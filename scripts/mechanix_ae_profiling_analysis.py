"""MechanixAE Descriptive-Profiling Validation Pipeline.

Implements the five descriptive-profiling gates defined in
``docs/models/mechanix_ae_validation_spec.md`` (reframed 2026-04-18 after the
injury-EWS framing was retired).  This script does NOT retrain anything --
it loads each per-pitcher VAE checkpoint under
``models/mechanix_ae/per_pitcher/<pid>.pt``, replays inference on that
pitcher's healthy windows, and produces the five spec-defined artifacts in
``results/mechanix_ae_profiling/``.

Outputs (mirrors the convention of ``scripts/mechanix_ae_roc_analysis.py``):

  - ``per_pitcher_fit.csv``                  -- Gate 1
  - ``mdi_distribution.csv``                 -- Gate 2
  - ``coverage.json``                        -- Gate 3
  - ``intra_pitcher_stability.csv`` /
    ``stability_summary.json``               -- Gate 4
  - ``feature_attribution_top_decile.csv`` /
    ``attribution_summary.json``             -- Gate 5

Read-only DuckDB access; checkpoints are read-only.  Seeds fixed at 42.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, skew

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytics.mechanix_ae import (  # noqa: E402
    FEATURE_COLS,
    N_FEATURES,
    RAW_FEATURE_COLS,
    WINDOW_SIZE,
    MechanixVAE,
    _get_device,
    build_sliding_windows,
    normalize_pitcher_pitch_type,
    prepare_features,
)

# ── Constants ────────────────────────────────────────────────────────────

DB_PATH: Path = ROOT / "data" / "baseball.duckdb"
CHECKPOINTS_DIR: Path = ROOT / "models" / "mechanix_ae" / "per_pitcher"
RESULTS_DIR: Path = ROOT / "results" / "mechanix_ae_profiling"

SEED: int = 42

MIN_HEALTHY_PITCHES: int = 200            # Gate 3 qualified-pitcher threshold
MIN_GAME_WINDOWS_FOR_STABILITY: int = 3   # spec: skip degenerate < 3-window starts
TOP_DECILE_FRACTION: float = 0.10
MECHANICAL_FEATURE_SET: set[str] = {
    "release_pos_x",
    "release_pos_z",
    "arm_angle",
    "release_extension",
    "spin_axis",
}

# Gate 1 / Gate 2 well-formedness thresholds (verbatim from spec).
GATE1_RECON_MSE_MAX: float = 0.50
GATE2_STD_MIN: float = 0.10
GATE2_SKEW_MAX_ABS: float = 5.0
GATE2_HIGH_Z_THRESHOLD: float = 1.0
GATE2_LOW_Z_THRESHOLD: float = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mechanix_profiling")


# ── Utilities ────────────────────────────────────────────────────────────


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def connect_db_readonly(
    retries: int = 6, base_delay: float = 1.0,
) -> duckdb.DuckDBPyConnection:
    """Open a read-only DuckDB connection with exponential backoff on lock."""
    delay = base_delay
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return duckdb.connect(str(DB_PATH), read_only=True)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "DB lock/connect failure (attempt %d): %s", attempt + 1, exc,
            )
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(
        f"Could not connect to DB after {retries} attempts: {last_exc}",
    )


def fetch_pitcher_pitches(
    conn: duckdb.DuckDBPyConnection, pitcher_id: int,
) -> pd.DataFrame:
    """Return the full pitch history for a pitcher with the raw feature cols."""
    cols = ", ".join(RAW_FEATURE_COLS)
    query = f"""
        SELECT
            pitcher_id, game_pk, game_date, pitch_type,
            at_bat_number, pitch_number,
            {cols}
        FROM pitches
        WHERE pitcher_id = ?
          AND release_speed IS NOT NULL
          AND pitch_type IS NOT NULL
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
    """
    return conn.execute(query, [pitcher_id]).fetchdf()


def df_to_feature_matrix(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the same standardisation pipeline used at training time.

    Returns ``(features, game_dates, game_pks)`` aligned row-wise with the
    incoming DataFrame after preprocessing.
    """
    if df.empty:
        return (
            np.empty((0, N_FEATURES), dtype=np.float32),
            np.array([], dtype="datetime64[ns]"),
            np.array([], dtype=np.int64),
        )
    df = prepare_features(df)
    df, _ = normalize_pitcher_pitch_type(df, columns=FEATURE_COLS)
    available = [c for c in FEATURE_COLS if c in df.columns]
    feat = df[available].fillna(0.0).astype(np.float32).values
    col_std = feat.std(axis=0)
    col_std[col_std < 1e-8] = 1.0
    feat = feat / col_std
    feat = np.clip(feat, -10.0, 10.0)
    if feat.shape[1] < N_FEATURES:
        feat = np.pad(feat, ((0, 0), (0, N_FEATURES - feat.shape[1])))
    dates = pd.to_datetime(df["game_date"]).values
    game_pks = df["game_pk"].astype(np.int64).values
    return feat, dates, game_pks


def windows_with_metadata(
    feats: np.ndarray, dates: np.ndarray, game_pks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows AND align end-of-window metadata.

    A window ending at pitch index ``i + W - 1`` carries that pitch's
    ``game_date`` and ``game_pk`` as its anchor metadata.
    """
    windows = build_sliding_windows(feats, WINDOW_SIZE)
    if len(windows) == 0:
        return windows, np.array([], dtype="datetime64[ns]"), np.array([], dtype=np.int64)
    end_idx = np.arange(WINDOW_SIZE - 1, WINDOW_SIZE - 1 + len(windows))
    return windows, dates[end_idx], game_pks[end_idx]


def load_checkpoint(pitcher_id: int) -> tuple[MechanixVAE, dict] | None:
    """Load a per-pitcher checkpoint.  Returns ``None`` on any failure."""
    path = CHECKPOINTS_DIR / f"{pitcher_id}.pt"
    if not path.exists():
        return None
    device = _get_device()
    try:
        try:
            ck = torch.load(path, map_location=device, weights_only=True)
        except Exception:
            ck = torch.load(path, map_location=device, weights_only=False)
        model = MechanixVAE(
            n_features=ck.get("n_features", N_FEATURES),
            window_size=ck.get("window_size", WINDOW_SIZE),
            latent_dim=ck.get("latent_dim", 6),
        )
        model.load_state_dict(ck["model_state_dict"])
        model.to(device)
        model.eval()
        return model, ck
    except Exception as exc:
        logger.warning("Failed to load checkpoint %s: %s", path, exc)
        return None


def reconstruction_errors(
    model: MechanixVAE, windows: np.ndarray,
) -> np.ndarray:
    """Per-window mean MSE on the standardised feature space."""
    if len(windows) == 0:
        return np.array([], dtype=np.float32)
    device = next(model.parameters()).device
    t = torch.tensor(windows, dtype=torch.float32).permute(0, 2, 1).to(device)
    with torch.no_grad():
        recon, _, _ = model(t)
    return ((t - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()


def per_feature_reconstruction_errors(
    model: MechanixVAE, windows: np.ndarray,
) -> np.ndarray:
    """Per-window, per-feature MSE.  Mirrors ``_render_feature_attribution``.

    Returns shape ``(n_windows, n_features)`` -- mean over the time axis.
    """
    if len(windows) == 0:
        return np.empty((0, N_FEATURES), dtype=np.float32)
    device = next(model.parameters()).device
    t = torch.tensor(windows, dtype=torch.float32).permute(0, 2, 1).to(device)
    with torch.no_grad():
        recon, _, _ = model(t)
    # (batch, features, time) -> mean over time
    return ((t - recon) ** 2).mean(dim=2).cpu().numpy()


# ── Per-pitcher inference ───────────────────────────────────────────────


def process_pitcher(
    conn: duckdb.DuckDBPyConnection, pitcher_id: int,
) -> dict | None:
    """Run inference for one pitcher; return all artifact-row payloads.

    Returns ``None`` if the pitcher cannot be processed (no checkpoint or no
    windows).  The caller is responsible for accumulating per-gate rows.
    """
    loaded = load_checkpoint(pitcher_id)
    if loaded is None:
        return None
    model, ck = loaded

    df = fetch_pitcher_pitches(conn, pitcher_id)
    if df.empty or len(df) < WINDOW_SIZE:
        return None

    feats, dates, game_pks = df_to_feature_matrix(df)
    windows, win_dates, win_game_pks = windows_with_metadata(feats, dates, game_pks)
    if len(windows) == 0:
        return None

    errors = reconstruction_errors(model, windows)
    if len(errors) == 0:
        return None

    # Per-feature errors, used for Gate 5 attribution.
    feat_errors = per_feature_reconstruction_errors(model, windows)

    # Z-score MDI: anchor against this pitcher's own healthy baseline.
    # Per-pitcher checkpoints persisted before the z-score era do NOT carry
    # ``healthy_recon_mean``/``healthy_recon_std``; in that case we follow
    # the same self-baseline fallback that ``calculate_mdi`` does.
    healthy_mean = ck.get("healthy_recon_mean")
    healthy_std = ck.get("healthy_recon_std")
    if healthy_mean is None or healthy_std is None:
        healthy_mean = float(np.mean(errors))
        healthy_std = float(np.std(errors))
    denom = max(float(healthy_std), 1e-6)
    z_scores = (errors - float(healthy_mean)) / denom

    return {
        "pitcher_id": pitcher_id,
        "n_windows": int(len(windows)),
        "errors": errors,
        "z_scores": z_scores,
        "feat_errors": feat_errors,
        "win_dates": win_dates,
        "win_game_pks": win_game_pks,
        "checkpoint_recon_loss": ck.get("final_recon_loss"),
    }


# ── Gate computations ───────────────────────────────────────────────────


def gate1_per_pitcher_fit(payload: dict) -> dict:
    """Per-pitcher reconstruction fit row for ``per_pitcher_fit.csv``."""
    errs = payload["errors"]
    mean_mse = float(np.mean(errs))
    std_mse = float(np.std(errs))
    return {
        "pitcher_id": payload["pitcher_id"],
        "n_windows": payload["n_windows"],
        "mean_recon_mse": round(mean_mse, 6),
        "std_recon_mse": round(std_mse, 6),
        "recon_mse": round(mean_mse, 6),  # spec column alias
        "checkpoint_final_recon_loss": (
            round(float(payload["checkpoint_recon_loss"]), 6)
            if payload["checkpoint_recon_loss"] is not None else None
        ),
        "pass": bool(mean_mse <= GATE1_RECON_MSE_MAX),
    }


def gate2_mdi_distribution(payload: dict) -> dict:
    """MDI distribution well-formedness row for ``mdi_distribution.csv``."""
    z = payload["z_scores"]
    n = len(z)
    std_z = float(np.std(z))
    skew_z = float(skew(z, bias=False)) if n >= 3 else 0.0
    n_high = int(np.sum(np.abs(z) >= GATE2_HIGH_Z_THRESHOLD))
    n_low = int(np.sum(np.abs(z) <= GATE2_LOW_Z_THRESHOLD))
    well_formed = bool(
        std_z > GATE2_STD_MIN
        and abs(skew_z) <= GATE2_SKEW_MAX_ABS
        and n_high >= 1
        and n_low >= 1
    )
    return {
        "pitcher_id": payload["pitcher_id"],
        "n_windows": n,
        "mdi_std": round(std_z, 6),
        "mdi_skew": round(skew_z, 6),
        "n_high_z": n_high,
        "n_low_z": n_low,
        "frac_high_z": round(n_high / max(n, 1), 6),
        "frac_low_z": round(n_low / max(n, 1), 6),
        "well_formed": well_formed,
        "pass": well_formed,
        "std": round(std_z, 6),     # spec column aliases
        "skew": round(skew_z, 6),
    }


def gate4_intra_pitcher_stability(payload: dict) -> dict | None:
    """Pearson correlation of consecutive within-game MDI windows.

    Aggregates across all of this pitcher's games, computing one Pearson r
    per game with >= ``MIN_GAME_WINDOWS_FOR_STABILITY`` consecutive windows.
    The pitcher-level row reports the median r across qualifying games.
    """
    z = payload["z_scores"]
    pks = payload["win_game_pks"]
    if len(z) < 2:
        return None
    df = pd.DataFrame({"game_pk": pks, "z": z})
    per_game_rs: list[float] = []
    n_pairs_total = 0
    for game_pk, grp in df.groupby("game_pk", sort=False):
        if len(grp) < MIN_GAME_WINDOWS_FOR_STABILITY:
            continue
        a = grp["z"].values[:-1]
        b = grp["z"].values[1:]
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            continue
        try:
            r, _ = pearsonr(a, b)
        except Exception:
            continue
        if not math.isnan(r):
            per_game_rs.append(float(r))
            n_pairs_total += len(a)
    if not per_game_rs:
        return None
    median_r = float(np.median(per_game_rs))
    return {
        "pitcher_id": payload["pitcher_id"],
        "n_qualifying_games": len(per_game_rs),
        "n_pairs": n_pairs_total,
        "pearson_r": round(median_r, 6),
        "pass": bool(median_r >= 0.70),
    }


def gate5_top_decile_attribution(
    payload: dict, mechanical_indices: list[int],
) -> list[dict]:
    """Identify the top-attributed feature for top-decile MDI windows.

    Mirrors ``_render_feature_attribution``: for each window, the dominant
    attributed feature is the largest positive entry of
    ``latest_errors - avg_errors`` (here ``avg_errors`` is the per-pitcher
    mean per-feature error excluding the focal window itself).
    """
    z = payload["z_scores"]
    feat_errs = payload["feat_errors"]
    n = len(z)
    if n == 0 or feat_errs.size == 0:
        return []
    k = max(1, int(math.ceil(n * TOP_DECILE_FRACTION)))
    # Indices of the top-decile windows by raw z (signed -- highest drift).
    top_idx = np.argsort(-z)[:k]

    # avg_errors per feature across the whole series (one vector per pitcher)
    cohort_avg = feat_errs.mean(axis=0)

    rows: list[dict] = []
    for w_idx in top_idx:
        latest = feat_errs[w_idx]
        # leave-one-out average: stable approximation for large n
        if n > 1:
            avg = (cohort_avg * n - latest) / (n - 1)
        else:
            avg = cohort_avg
        delta = latest - avg
        # Spec language: "largest positive entry of latest - avg"
        if not np.any(delta > 0):
            top_feat_idx = int(np.argmax(delta))  # least-negative fallback
        else:
            top_feat_idx = int(np.argmax(delta))
        feat_name = FEATURE_COLS[top_feat_idx] if top_feat_idx < len(FEATURE_COLS) \
            else f"feat_{top_feat_idx}"
        rows.append({
            "pitcher_id": payload["pitcher_id"],
            "window_idx": int(w_idx),
            "mdi": round(float(z[w_idx]), 6),
            "top_feature": feat_name,
            "top_feature_delta": round(float(delta[top_feat_idx]), 6),
            "in_mechanical_set": bool(feat_name in MECHANICAL_FEATURE_SET),
        })
    return rows


# ── Coverage (Gate 3) ───────────────────────────────────────────────────


def compute_coverage(
    conn: duckdb.DuckDBPyConnection,
    processed_pitchers: set[int],
) -> dict:
    """Cross-reference qualified pitchers vs available checkpoints."""
    qualified_df = conn.execute(
        """
        SELECT pitcher_id, COUNT(*) AS n
        FROM pitches
        WHERE release_speed IS NOT NULL
          AND pitch_type IS NOT NULL
        GROUP BY pitcher_id
        HAVING COUNT(*) >= ?
        """,
        [MIN_HEALTHY_PITCHES],
    ).fetchdf()
    qualified_ids = {int(p) for p in qualified_df["pitcher_id"].tolist()}
    n_qualified = len(qualified_ids)

    on_disk_ids: set[int] = set()
    for path in CHECKPOINTS_DIR.glob("*.pt"):
        try:
            on_disk_ids.add(int(path.stem))
        except ValueError:
            continue

    n_with_ckpt = len(qualified_ids & on_disk_ids)
    n_loadable_and_mdi = len(qualified_ids & processed_pitchers)
    coverage_pct = (
        n_loadable_and_mdi / n_qualified if n_qualified else 0.0
    )
    return {
        "n_qualified_pitchers": n_qualified,
        "n_with_checkpoint": n_with_ckpt,
        "n_loadable": n_loadable_and_mdi,
        "n_produces_mdi": n_loadable_and_mdi,
        "coverage_pct": round(float(coverage_pct), 6),
        "min_healthy_pitches": MIN_HEALTHY_PITCHES,
        "n_checkpoints_on_disk": len(on_disk_ids),
        "pass": bool(coverage_pct >= 0.75),
    }


# ── Pipeline ────────────────────────────────────────────────────────────


def run_pipeline() -> dict:
    set_seeds(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    conn = connect_db_readonly()

    checkpoint_files = sorted(CHECKPOINTS_DIR.glob("*.pt"))
    logger.info("Found %d per-pitcher checkpoints", len(checkpoint_files))

    fit_rows: list[dict] = []
    mdi_rows: list[dict] = []
    stability_rows: list[dict] = []
    attribution_rows: list[dict] = []
    processed_pitchers: set[int] = set()

    mechanical_indices = [
        FEATURE_COLS.index(f) for f in MECHANICAL_FEATURE_SET
        if f in FEATURE_COLS
    ]

    for i, ck_path in enumerate(checkpoint_files, start=1):
        try:
            pid = int(ck_path.stem)
        except ValueError:
            logger.warning("Skipping non-numeric checkpoint stem: %s", ck_path)
            continue

        try:
            payload = process_pitcher(conn, pid)
        except Exception as exc:
            logger.warning("Pitcher %d failed: %s", pid, exc)
            continue
        if payload is None:
            continue
        processed_pitchers.add(pid)

        fit_rows.append(gate1_per_pitcher_fit(payload))
        mdi_rows.append(gate2_mdi_distribution(payload))
        stab = gate4_intra_pitcher_stability(payload)
        if stab is not None:
            stability_rows.append(stab)
        attribution_rows.extend(gate5_top_decile_attribution(payload, mechanical_indices))

        if i % 10 == 0 or i == len(checkpoint_files):
            logger.info(
                "Processed %d/%d checkpoints (%.1fs elapsed)",
                i, len(checkpoint_files), time.time() - t0,
            )

    coverage = compute_coverage(conn, processed_pitchers)
    conn.close()

    # ── Write artifacts ────────────────────────────────────────────────
    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(RESULTS_DIR / "per_pitcher_fit.csv", index=False)

    mdi_df = pd.DataFrame(mdi_rows)
    mdi_df.to_csv(RESULTS_DIR / "mdi_distribution.csv", index=False)

    (RESULTS_DIR / "coverage.json").write_text(
        json.dumps(coverage, indent=2),
    )

    stab_df = pd.DataFrame(stability_rows)
    stab_df.to_csv(RESULTS_DIR / "intra_pitcher_stability.csv", index=False)
    if not stab_df.empty:
        stab_summary = {
            "n_pitchers_with_stability": int(len(stab_df)),
            "median_pearson_r": round(float(stab_df["pearson_r"].median()), 6),
            "mean_pearson_r": round(float(stab_df["pearson_r"].mean()), 6),
            "p25_pearson_r": round(float(stab_df["pearson_r"].quantile(0.25)), 6),
            "p75_pearson_r": round(float(stab_df["pearson_r"].quantile(0.75)), 6),
            "n_total_game_pairs": int(stab_df["n_qualifying_games"].sum()),
            "min_game_windows": MIN_GAME_WINDOWS_FOR_STABILITY,
            "pass": bool(stab_df["pearson_r"].median() >= 0.70),
        }
    else:
        stab_summary = {
            "n_pitchers_with_stability": 0,
            "median_pearson_r": None,
            "mean_pearson_r": None,
            "min_game_windows": MIN_GAME_WINDOWS_FOR_STABILITY,
            "pass": False,
            "note": "no pitchers had qualifying multi-window starts",
        }
    (RESULTS_DIR / "stability_summary.json").write_text(
        json.dumps(stab_summary, indent=2),
    )

    attr_df = pd.DataFrame(attribution_rows)
    attr_df.to_csv(
        RESULTS_DIR / "feature_attribution_top_decile.csv", index=False,
    )
    if not attr_df.empty:
        in_set_pct = float(attr_df["in_mechanical_set"].mean())
        feat_counts = attr_df["top_feature"].value_counts().to_dict()
        attr_summary = {
            "n_top_decile_windows": int(len(attr_df)),
            "n_pitchers_contributing": int(attr_df["pitcher_id"].nunique()),
            "cohort_pct_in_mechanical_set": round(in_set_pct, 6),
            "mechanical_feature_set": sorted(MECHANICAL_FEATURE_SET),
            "top_feature_counts": {k: int(v) for k, v in feat_counts.items()},
            "pass": bool(in_set_pct >= 0.60),
        }
    else:
        attr_summary = {
            "n_top_decile_windows": 0,
            "cohort_pct_in_mechanical_set": None,
            "pass": False,
            "note": "no top-decile windows produced -- pipeline degenerate",
        }
    (RESULTS_DIR / "attribution_summary.json").write_text(
        json.dumps(attr_summary, indent=2),
    )

    # ── Aggregate gate verdicts ────────────────────────────────────────
    gate1_pct = (
        float(fit_df["pass"].mean()) if not fit_df.empty else 0.0
    )
    gate2_pct = (
        float(mdi_df["well_formed"].mean()) if not mdi_df.empty else 0.0
    )
    gate3_pct = coverage["coverage_pct"]
    gate4_median_r = stab_summary.get("median_pearson_r")
    gate5_pct = attr_summary.get("cohort_pct_in_mechanical_set")

    summary = {
        "n_checkpoints_processed": int(len(processed_pitchers)),
        "wall_clock_seconds": round(time.time() - t0, 2),
        "gates": {
            "gate1_recon_fit": {
                "threshold": ">=80% with mean_recon_mse <= 0.50",
                "measured_pct_passing": round(gate1_pct, 6),
                "pass": bool(gate1_pct >= 0.80),
                "n_pitchers": int(len(fit_df)),
            },
            "gate2_mdi_well_formedness": {
                "threshold": ">=90% well-formed",
                "measured_pct_passing": round(gate2_pct, 6),
                "pass": bool(gate2_pct >= 0.90),
                "n_pitchers": int(len(mdi_df)),
            },
            "gate3_coverage": {
                "threshold": "coverage_pct >= 0.75",
                "measured_pct": gate3_pct,
                "pass": bool(coverage["pass"]),
                "n_qualified": coverage["n_qualified_pitchers"],
                "n_loadable": coverage["n_loadable"],
            },
            "gate4_intra_pitcher_stability": {
                "threshold": "median_pearson_r >= 0.70",
                "measured_median_r": gate4_median_r,
                "pass": bool(stab_summary.get("pass", False)),
                "n_pitchers": int(stab_summary.get("n_pitchers_with_stability", 0)),
            },
            "gate5_attribution": {
                "threshold": "cohort_pct_in_mechanical_set >= 0.60",
                "measured_pct": gate5_pct,
                "pass": bool(attr_summary.get("pass", False)),
                "n_top_decile_windows": int(attr_summary.get("n_top_decile_windows", 0)),
            },
        },
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="MechanixAE descriptive-profiling validation pipeline.",
    )
    p.parse_args()  # no flags yet; reserved for future smoke modes
    summary = run_pipeline()
    print(json.dumps(summary, indent=2))
