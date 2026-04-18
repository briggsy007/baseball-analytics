"""MechanixAE Validation Pipeline (Tickets #3, #4, #5, #6 of the validation spec).

Runs the following end-to-end on 2015-2016 arm-injury cohort from
``data/injury_labels.parquet``:

  Step 1.  Per-pitcher VAE training on healthy pre-IL-30 pitches.
  Step 2.  Lead-time analysis (first MDI threshold breach vs IL date).
  Step 3.  ROC/AUC at 30-day pre-IL window.
  Step 4.  Velocity-drop baseline ROC/AUC.
  Step 5.  False-positive rate on healthy pitcher-seasons.

All outputs land in ``results/mechanix_ae/`` and per-pitcher checkpoints
land in ``models/mechanix_ae/per_pitcher/``.

Read-only DB access.  Seeds fixed at 42 for reproducibility.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytics.mechanix_ae import (  # noqa: E402
    BETA_KL,
    FEATURE_COLS,
    LATENT_DIM,
    N_FEATURES,
    RAW_FEATURE_COLS,
    WINDOW_SIZE,
    MechanixVAE,
    _get_device,
    build_sliding_windows,
    normalize_pitcher_pitch_type,
    prepare_features,
    vae_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("mechanix_roc")

ARM_TYPES = [
    "tommy_john",
    "ucl_sprain",
    "shoulder",
    "elbow",
    "rotator_cuff",
    "labrum",
    "forearm",
    "other_arm",
]

DB_PATH = ROOT / "data" / "baseball.duckdb"
LABELS_PATH = ROOT / "data" / "injury_labels.parquet"
RESULTS_DIR = ROOT / "results" / "mechanix_ae"
CHECKPOINTS_DIR = ROOT / "models" / "mechanix_ae" / "per_pitcher"

SEED = 42

THRESHOLDS_LEAD_TIME_PERCENTILE = [60, 70, 80]
ROC_THRESHOLDS_PERCENTILE = list(range(0, 101, 5))

# Z-score variant: 1σ, 2σ, 3σ above the healthy baseline mean are the
# "percentile-like" buckets.  ROC sweep is a dense grid in the same range.
THRESHOLDS_LEAD_TIME_ZSCORE = [1.0, 2.0, 3.0]
ROC_THRESHOLDS_ZSCORE = [round(x, 2) for x in np.linspace(-2.0, 6.0, 41).tolist()]

# Backwards-compatible aliases — set on argparse based on --score-mode.
THRESHOLDS_LEAD_TIME = THRESHOLDS_LEAD_TIME_PERCENTILE
ROC_THRESHOLDS = ROC_THRESHOLDS_PERCENTILE

# Train-time hyperparameters.  A per-pitcher VAE converges fast.
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MIN_HEALTHY_PITCHES = 200
MIN_PRE_IL_PITCHES = WINDOW_SIZE  # need at least one window


# ──────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def connect_db_readonly(retries: int = 6, base_delay: float = 1.0) -> duckdb.DuckDBPyConnection:
    """Open a read-only DuckDB connection with exponential backoff."""
    delay = base_delay
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return duckdb.connect(str(DB_PATH), read_only=True)
        except Exception as exc:  # duckdb.IOException or similar
            last_exc = exc
            logger.warning("DB lock/connect failure (attempt %d): %s", attempt + 1, exc)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Could not connect to DB after {retries} attempts: {last_exc}")


def load_cohort() -> pd.DataFrame:
    df = pd.read_parquet(LABELS_PATH)
    arm = df[df["injury_type"].isin(ARM_TYPES)].copy()
    arm["il_date"] = pd.to_datetime(arm["il_date"])
    arm = arm.dropna(subset=["pitcher_id", "il_date"]).copy()
    arm["pitcher_id"] = arm["pitcher_id"].astype(int)
    # Earliest IL placement per pitcher is the injury of interest.
    arm = arm.sort_values("il_date").drop_duplicates("pitcher_id", keep="first")
    arm = arm.reset_index(drop=True)
    return arm


def fetch_pitches(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    date_lt: pd.Timestamp | None = None,
    date_ge: pd.Timestamp | None = None,
    date_le: pd.Timestamp | None = None,
) -> pd.DataFrame:
    cols = ", ".join(RAW_FEATURE_COLS)
    conds = [
        "pitcher_id = ?",
        "release_speed IS NOT NULL",
        "pitch_type IS NOT NULL",
    ]
    params: list = [pitcher_id]
    if date_lt is not None:
        conds.append("game_date < ?")
        params.append(date_lt.date() if hasattr(date_lt, "date") else date_lt)
    if date_ge is not None:
        conds.append("game_date >= ?")
        params.append(date_ge.date() if hasattr(date_ge, "date") else date_ge)
    if date_le is not None:
        conds.append("game_date <= ?")
        params.append(date_le.date() if hasattr(date_le, "date") else date_le)
    where = " AND ".join(conds)
    query = f"""
        SELECT
            pitcher_id, game_pk, game_date, pitch_type,
            at_bat_number, pitch_number,
            {cols}
        FROM pitches
        WHERE {where}
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
    """
    return conn.execute(query, params).fetchdf()


def df_to_feature_matrix(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (features, game_dates, release_speeds) aligned with ``df`` rows.

    Applies identical preprocessing as the training pipeline (normalize,
    standardise, clip) so inference and training live in the same space.
    """
    if df.empty:
        return (
            np.empty((0, N_FEATURES), dtype=np.float32),
            np.array([], dtype="datetime64[ns]"),
            np.array([], dtype=np.float32),
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
    speeds = df["release_speed"].astype(float).fillna(0.0).values.astype(np.float32)
    return feat, dates, speeds


# ──────────────────────────────────────────────────────────────────────────
# Step 1 — per-pitcher training
# ──────────────────────────────────────────────────────────────────────────


def train_per_pitcher_model(
    feat_matrix: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
) -> tuple[MechanixVAE, float]:
    """Train a VAE on healthy windows; return the trained model and final recon loss."""
    set_seeds(SEED)
    windows = build_sliding_windows(feat_matrix, WINDOW_SIZE)
    if len(windows) < 1:
        raise ValueError("No windows")
    device = _get_device()
    tensor = torch.tensor(windows, dtype=torch.float32).permute(0, 2, 1)
    # Work on CPU DataLoader (avoid shuffling nondeterminism via seeded generator).
    g = torch.Generator()
    g.manual_seed(SEED)
    dataset = torch.utils.data.TensorDataset(tensor, tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=g,
    )
    model = MechanixVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    final_recon = float("nan")
    for epoch in range(epochs):
        running_recon = 0.0
        n_batches = 0
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            opt.zero_grad()
            recon, mu, logvar = model(batch_x)
            loss, recon_l, _ = vae_loss(recon, batch_x, mu, logvar, beta=BETA_KL)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_recon += recon_l.item()
            n_batches += 1
        final_recon = running_recon / max(n_batches, 1)
    model.eval()
    return model, final_recon


def recon_errors_for_windows(model: MechanixVAE, windows: np.ndarray) -> np.ndarray:
    if len(windows) == 0:
        return np.array([], dtype=np.float32)
    device = next(model.parameters()).device
    t = torch.tensor(windows, dtype=torch.float32).permute(0, 2, 1).to(device)
    with torch.no_grad():
        recon, _, _ = model(t)
    return ((t - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()


def compute_mdi_from_errors(
    errors: np.ndarray,
    baseline_errors: np.ndarray,
) -> np.ndarray:
    """MDI = percentile rank of each error against the (healthy) baseline distribution.

    Returns a 1-D float array in [0, 100].
    """
    if len(errors) == 0:
        return np.array([], dtype=np.float32)
    if len(baseline_errors) == 0:
        # Fall back to self-rank
        baseline_errors = errors
    # Searchsorted for percentile rank of each error in the sorted baseline
    srt = np.sort(baseline_errors)
    ranks = np.searchsorted(srt, errors, side="right")
    return np.clip(ranks / max(len(srt), 1) * 100.0, 0.0, 100.0)


def compute_zscore_from_errors(
    errors: np.ndarray,
    baseline_errors: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Magnitude-based MDI: z = (error - baseline_mean) / max(baseline_std, eps).

    Higher z = more mechanical drift above the pitcher's healthy baseline.
    Unbounded; z=1/2/3 are the recommended 1σ/2σ/3σ threshold buckets.
    """
    if len(errors) == 0:
        return np.array([], dtype=np.float32)
    if len(baseline_errors) == 0:
        # Degenerate case — self-normalise (returns zero-centred z)
        baseline_errors = errors
    mu = float(np.mean(baseline_errors))
    sigma = float(np.std(baseline_errors))
    denom = max(sigma, eps)
    return (np.asarray(errors, dtype=float) - mu) / denom


# ──────────────────────────────────────────────────────────────────────────
# Step 2 — lead-time analysis
# ──────────────────────────────────────────────────────────────────────────


def rolling_window_errors_per_pitch(errors: np.ndarray) -> np.ndarray:
    """Given per-window errors of length (n_pitches - W + 1), each value is the
    error whose last pitch is at index (i + W - 1).  This helper is a no-op —
    kept for readability in the caller."""
    return errors


def per_day_mdi_series(
    dates: np.ndarray,
    window_errors: np.ndarray,
    baseline_errors: np.ndarray,
    score_mode: str = "percentile",
) -> pd.DataFrame:
    """Aggregate window-level errors into per-game-day MDI.

    ``window_errors`` is aligned such that window i ends at pitch index (i + W - 1).
    The MDI for a given date is the max score of any window ending on that date.

    Args:
        dates: pitch-level game dates aligned with the underlying feature matrix.
        window_errors: per-window reconstruction errors.
        baseline_errors: healthy training-set reconstruction errors.
        score_mode: "percentile" (rank vs baseline) or "zscore" (magnitude).
    """
    if len(window_errors) == 0:
        return pd.DataFrame(columns=["game_date", "mdi", "recon_error"])
    # end_date of window i = dates[i + W - 1]
    end_dates = dates[WINDOW_SIZE - 1 : WINDOW_SIZE - 1 + len(window_errors)]
    if len(end_dates) != len(window_errors):
        # Shouldn't happen with correctly sized windows; align defensively.
        m = min(len(end_dates), len(window_errors))
        end_dates = end_dates[:m]
        window_errors = window_errors[:m]
    if score_mode == "zscore":
        mdi_vals = compute_zscore_from_errors(window_errors, baseline_errors)
    else:
        mdi_vals = compute_mdi_from_errors(window_errors, baseline_errors)
    df = pd.DataFrame({
        "game_date": pd.to_datetime(end_dates),
        "mdi": mdi_vals,
        "recon_error": window_errors,
    })
    daily = df.groupby("game_date", as_index=False).agg(
        mdi=("mdi", "max"),
        recon_error=("recon_error", "max"),
    )
    return daily.sort_values("game_date").reset_index(drop=True)


def first_breach(
    daily: pd.DataFrame, threshold: float, il_date: pd.Timestamp
) -> float | None:
    """Return lead-time in days (IL_date - first_breach_date), or None if never breached."""
    breaches = daily[daily["mdi"] >= threshold]
    if breaches.empty:
        return None
    first = breaches.iloc[0]["game_date"]
    lead = (il_date - pd.Timestamp(first)).days
    return float(lead)


# ──────────────────────────────────────────────────────────────────────────
# Step 3 — ROC/AUC
# ──────────────────────────────────────────────────────────────────────────


def roc_from_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    thresholds: list[int] | list[float] | None = None,
) -> dict:
    """Compute ROC curve (TPR/FPR for each threshold) and AUC via trapezoid.

    Scores are MDI-like values.  A prediction is positive iff ``score >= threshold``.
    Thresholds are treated as float; percentile-mode passes ints in [0,100],
    z-score mode passes floats spanning roughly [-2, 6].
    """
    if thresholds is None:
        thresholds = ROC_THRESHOLDS
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return {
            "thresholds": thresholds,
            "tpr": [0.0] * len(thresholds),
            "fpr": [0.0] * len(thresholds),
            "auc": float("nan"),
            "n_pos": int(pos),
            "n_neg": int(neg),
        }
    tpr_list, fpr_list = [], []
    for t in thresholds:
        pred = (s >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        tpr_list.append(float(tp / pos))
        fpr_list.append(float(fp / neg))
    # Full-resolution AUC using unique-score sort (mann-whitney equivalent).
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    # Tie-aware TPR/FPR curve
    tps, fps = [0], [0]
    i = 0
    while i < len(y_sorted):
        j = i
        while j < len(y_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        tp_block = int((y_sorted[i:j] == 1).sum())
        fp_block = int((y_sorted[i:j] == 0).sum())
        tps.append(tps[-1] + tp_block)
        fps.append(fps[-1] + fp_block)
        i = j
    tpr_curve = np.array(tps, dtype=float) / pos
    fpr_curve = np.array(fps, dtype=float) / neg
    trapz = getattr(np, "trapezoid", np.trapz)
    auc = float(trapz(tpr_curve, fpr_curve))
    return {
        "thresholds": thresholds,
        "tpr": tpr_list,
        "fpr": fpr_list,
        "auc": auc,
        "n_pos": int(pos),
        "n_neg": int(neg),
    }


def bootstrap_auc_ci(
    y: np.ndarray,
    s: np.ndarray,
    n_iter: int = 200,
    rng: np.random.RandomState | None = None,
) -> tuple[float, float]:
    """Return 95% percentile bootstrap CI for AUC."""
    if rng is None:
        rng = np.random.RandomState(SEED)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n_iter):
        sample = rng.choice(idx, size=len(idx), replace=True)
        ys = y[sample]
        ss = s[sample]
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        r = roc_from_scores(ys, ss)
        if not math.isnan(r["auc"]):
            aucs.append(r["auc"])
    if not aucs:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(aucs, 2.5))
    hi = float(np.percentile(aucs, 97.5))
    return (lo, hi)


# ──────────────────────────────────────────────────────────────────────────
# Step 4 — velocity-drop baseline
# ──────────────────────────────────────────────────────────────────────────


def velocity_drop_score(
    speeds: np.ndarray, window: int = 10
) -> np.ndarray:
    """Return a score per pitch: number of standard deviations BELOW the rolling mean.

    A higher score ↔ more suspicious drop.  The first ``window`` pitches get 0.
    """
    s = pd.Series(speeds, dtype=float)
    roll_mean = s.rolling(window, min_periods=window).mean().shift(1)
    roll_std = s.rolling(window, min_periods=window).std().shift(1)
    score = (roll_mean - s) / roll_std.replace(0, np.nan)
    return score.fillna(0.0).clip(lower=0.0).values.astype(float)


def daily_velocity_score(
    dates: np.ndarray, speeds: np.ndarray, window: int = 10
) -> pd.DataFrame:
    if len(speeds) == 0:
        return pd.DataFrame(columns=["game_date", "vel_score"])
    raw = velocity_drop_score(speeds, window=window)
    df = pd.DataFrame({
        "game_date": pd.to_datetime(dates),
        "score": raw,
    })
    daily = df.groupby("game_date", as_index=False).agg(vel_score=("score", "max"))
    return daily.sort_values("game_date").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────
# Step 5 — False-positive rate on healthy seasons
# ──────────────────────────────────────────────────────────────────────────


def fetch_healthy_pool(
    conn: duckdb.DuckDBPyConnection,
    injured_ids: set[int],
    seasons: tuple[int, ...] = (2015, 2016),
    min_pitches: int = 1000,
    max_n: int = 120,
) -> list[tuple[int, int]]:
    """Return a list of (pitcher_id, season) never-injured in the label set, ≥min_pitches."""
    placeholders = ",".join("?" * len(seasons))
    q = f"""
        SELECT pitcher_id, EXTRACT(YEAR FROM game_date)::INT AS season, COUNT(*) AS n
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) IN ({placeholders})
          AND release_speed IS NOT NULL
          AND pitch_type IS NOT NULL
        GROUP BY pitcher_id, season
        HAVING COUNT(*) >= ?
        ORDER BY n DESC
    """
    params = list(seasons) + [min_pitches]
    df = conn.execute(q, params).fetchdf()
    out = [
        (int(pid), int(season))
        for pid, season in zip(df["pitcher_id"], df["season"])
        if int(pid) not in injured_ids
    ]
    return out[:max_n]


# ──────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────


def run_pipeline(
    max_pitchers: int | None = None,
    skip_train: bool = False,
    n_healthy_roc: int = 50,
    n_healthy_fpr: int = 100,
    score_mode: str = "percentile",
) -> dict:
    """Run the MechanixAE validation pipeline.

    Args:
        score_mode: "percentile" (legacy, self-rank) or "zscore" (magnitude
            normalised by healthy baseline mean/std).  Only the scoring
            function changes between modes; the checkpoints and per-pitcher
            windows are identical.
    """
    if score_mode not in ("percentile", "zscore"):
        raise ValueError(f"Unknown score_mode: {score_mode!r}")

    set_seeds(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # Mode-specific thresholds and file suffixes so both runs can coexist.
    if score_mode == "zscore":
        lead_thresholds: list = list(THRESHOLDS_LEAD_TIME_ZSCORE)
        roc_thresholds: list = list(ROC_THRESHOLDS_ZSCORE)
        suffix = "_zscore"
        # FPR buckets map to the same z-values for "breached_60/70/80" naming.
        # We re-label them below so the legacy column names stay intact and
        # the JSON summary gets z=1/2/3 keys as well.
    else:
        lead_thresholds = list(THRESHOLDS_LEAD_TIME_PERCENTILE)
        roc_thresholds = list(ROC_THRESHOLDS_PERCENTILE)
        suffix = "_percentile"

    t0_total = time.time()
    logger.info("Loading cohort from %s", LABELS_PATH)
    cohort = load_cohort()
    logger.info("Cohort rows (arm injuries): %d", len(cohort))

    conn = connect_db_readonly()

    # Pre-filter to pitchers with sufficient healthy pre-IL data so the
    # --max-pitchers smoke test hits trainable cases and the production
    # run skips obvious TJ-recovery non-starters up front.
    elig_rows = []
    for _, row in cohort.iterrows():
        pid = int(row["pitcher_id"])
        cutoff = row["il_date"] - pd.Timedelta(days=30)
        try:
            n = conn.execute(
                "SELECT COUNT(*) FROM pitches WHERE pitcher_id=? AND game_date < ? "
                "AND release_speed IS NOT NULL AND pitch_type IS NOT NULL",
                [pid, cutoff.date()],
            ).fetchone()[0]
        except Exception:
            n = 0
        if n >= MIN_HEALTHY_PITCHES:
            elig_rows.append(row)
    cohort = pd.DataFrame(elig_rows).reset_index(drop=True)
    logger.info("Cohort eligible (>=%d healthy): %d", MIN_HEALTHY_PITCHES, len(cohort))

    trained_records: list[dict] = []
    skipped_records: list[dict] = []
    # Cache of per-pitcher artifacts for downstream steps
    artifacts: dict[int, dict] = {}

    if max_pitchers is not None:
        cohort = cohort.head(max_pitchers).copy()

    t_train_start = time.time()
    for idx, row in cohort.iterrows():
        pid = int(row["pitcher_id"])
        il_date = pd.Timestamp(row["il_date"])
        cutoff = il_date - pd.Timedelta(days=30)
        try:
            healthy_df = fetch_pitches(conn, pid, date_lt=cutoff)
        except Exception as exc:
            logger.warning("Failed to fetch pitches for %d: %s", pid, exc)
            skipped_records.append({
                "pitcher_id": pid, "injury_type": row["injury_type"],
                "reason": "fetch_failed", "detail": str(exc),
            })
            continue
        if len(healthy_df) < MIN_HEALTHY_PITCHES:
            skipped_records.append({
                "pitcher_id": pid, "injury_type": row["injury_type"],
                "reason": "insufficient_healthy_pitches",
                "n_healthy": int(len(healthy_df)),
            })
            continue

        healthy_feats, healthy_dates, healthy_speeds = df_to_feature_matrix(healthy_df)
        if len(healthy_feats) < MIN_HEALTHY_PITCHES:
            skipped_records.append({
                "pitcher_id": pid, "injury_type": row["injury_type"],
                "reason": "feature_build_short", "n_healthy": int(len(healthy_feats)),
            })
            continue

        ckpt_path = CHECKPOINTS_DIR / f"{pid}.pt"
        try:
            if skip_train and ckpt_path.exists():
                # Load from disk
                device = _get_device()
                ck = torch.load(ckpt_path, map_location=device, weights_only=True)
                model = MechanixVAE().to(device)
                model.load_state_dict(ck["model_state_dict"])
                model.eval()
                final_recon = float(ck.get("final_recon_loss", float("nan")))
            else:
                model, final_recon = train_per_pitcher_model(healthy_feats)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "final_recon_loss": final_recon,
                    "n_features": N_FEATURES,
                    "window_size": WINDOW_SIZE,
                    "latent_dim": LATENT_DIM,
                    "injury_type": row["injury_type"],
                    "il_date": str(il_date.date()),
                    "n_healthy_pitches": int(len(healthy_feats)),
                }, ckpt_path)
        except Exception as exc:
            logger.warning("Train failed for %d: %s", pid, exc)
            skipped_records.append({
                "pitcher_id": pid, "injury_type": row["injury_type"],
                "reason": "train_failed", "detail": str(exc),
            })
            continue

        # Healthy baseline error distribution = reconstruction errors on healthy windows.
        healthy_windows = build_sliding_windows(healthy_feats, WINDOW_SIZE)
        baseline_errors = recon_errors_for_windows(model, healthy_windows)

        artifacts[pid] = {
            "model": model,
            "il_date": il_date,
            "injury_type": row["injury_type"],
            "baseline_errors": baseline_errors,
            "healthy_speeds": healthy_speeds,
            "n_healthy_pitches": int(len(healthy_feats)),
        }
        trained_records.append({
            "pitcher_id": pid,
            "injury_type": row["injury_type"],
            "il_date": str(il_date.date()),
            "n_healthy_pitches": int(len(healthy_feats)),
            "final_recon_loss": float(final_recon),
        })

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t_train_start
            logger.info(
                "Trained %d/%d pitchers so far (%.1f min elapsed)",
                len(trained_records), idx + 1, elapsed / 60,
            )

    t_train_end = time.time()
    logger.info(
        "Training step done.  Trained=%d, skipped=%d, wall=%.1f min",
        len(trained_records), len(skipped_records),
        (t_train_end - t_train_start) / 60,
    )

    # Coverage JSON
    coverage = {
        "n_cohort": int(len(cohort)),
        "n_trained": len(trained_records),
        "n_skipped": len(skipped_records),
        "wall_train_minutes": round((t_train_end - t_train_start) / 60, 2),
        "per_injury_trained": (
            pd.DataFrame(trained_records)["injury_type"].value_counts().to_dict()
            if trained_records else {}
        ),
        "per_injury_skipped": (
            pd.DataFrame(skipped_records)["injury_type"].value_counts().to_dict()
            if skipped_records else {}
        ),
        "skipped_reasons": (
            pd.DataFrame(skipped_records)["reason"].value_counts().to_dict()
            if skipped_records else {}
        ),
        "training_records": trained_records,
        "skipped_records": skipped_records,
    }
    (RESULTS_DIR / "training_coverage.json").write_text(json.dumps(coverage, indent=2, default=str))

    # ── Step 2: lead-time analysis ────────────────────────────────────────
    logger.info("Step 2: lead-time analysis")
    lead_rows: list[dict] = []
    pre_il_windowed: dict[int, pd.DataFrame] = {}  # for ROC later
    for pid, art in artifacts.items():
        il_date = art["il_date"]
        start = il_date - pd.Timedelta(days=90)
        try:
            pre_df = fetch_pitches(conn, pid, date_ge=start, date_le=il_date)
        except Exception as exc:
            logger.warning("pre-IL fetch fail %d: %s", pid, exc)
            continue
        if len(pre_df) < MIN_PRE_IL_PITCHES:
            continue
        pre_feats, pre_dates, pre_speeds = df_to_feature_matrix(pre_df)
        pre_windows = build_sliding_windows(pre_feats, WINDOW_SIZE)
        if len(pre_windows) == 0:
            continue
        pre_errors = recon_errors_for_windows(art["model"], pre_windows)
        daily = per_day_mdi_series(
            pre_dates, pre_errors, art["baseline_errors"], score_mode=score_mode,
        )
        if daily.empty:
            continue
        # Velocity daily for baseline
        daily_vel = daily_velocity_score(pre_dates, pre_speeds, window=10)
        daily = daily.merge(daily_vel, on="game_date", how="left").fillna({"vel_score": 0.0})

        rec: dict = {"pitcher_id": pid, "injury_type": art["injury_type"],
                     "il_date": str(il_date.date()), "n_pre_il_days": int(len(daily))}
        for th in lead_thresholds:
            rec[f"lead_time_t{th}"] = first_breach(daily, th, il_date)
        lead_rows.append(rec)
        pre_il_windowed[pid] = daily

    lead_df = pd.DataFrame(lead_rows)
    # Ensure required columns exist even when empty
    for th in lead_thresholds:
        col = f"lead_time_t{th}"
        if col not in lead_df.columns:
            lead_df[col] = pd.Series(dtype=float)
    lead_df.to_csv(RESULTS_DIR / f"lead_time_per_pitcher{suffix}.csv", index=False)

    lead_summary: dict = {}
    for th in lead_thresholds:
        key = f"lead_time_t{th}"
        vals = lead_df[key].dropna()
        lead_summary[key] = {
            "n_breached": int(len(vals)),
            "n_total": int(len(lead_df)),
            "fraction_breached": float(len(vals) / max(len(lead_df), 1)),
            "median_days": float(vals.median()) if len(vals) else None,
            "mean_days": float(vals.mean()) if len(vals) else None,
            "p25_days": float(vals.quantile(0.25)) if len(vals) else None,
            "p75_days": float(vals.quantile(0.75)) if len(vals) else None,
        }
    (RESULTS_DIR / f"lead_time_distribution{suffix}.json").write_text(
        json.dumps(lead_summary, indent=2)
    )

    # Plotly histogram
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for th in lead_thresholds:
            vals = lead_df[f"lead_time_t{th}"].dropna().values
            if len(vals) == 0:
                continue
            label_prefix = "z" if score_mode == "zscore" else "MDI"
            fig.add_trace(go.Histogram(
                x=vals, name=f"{label_prefix}>={th}", nbinsx=20, opacity=0.6,
            ))
        mode_title = "z-score" if score_mode == "zscore" else "percentile"
        fig.update_layout(
            barmode="overlay",
            title=f"MechanixAE Lead-Time Distribution ({mode_title}) "
                  "(days before IL placement)",
            xaxis_title="Lead time (days)",
            yaxis_title="Number of pitchers",
        )
        fig.write_html(RESULTS_DIR / f"lead_time_distribution{suffix}.html")
    except Exception as exc:
        logger.warning("Plotly lead-time plot failed: %s", exc)

    # ── Step 3: ROC at 30-day pre-IL window (MDI) ─────────────────────────
    logger.info("Step 3: ROC/AUC (MDI)")

    # Positives: injured pitcher-days within 30 days pre-IL.
    # Negatives: random sample of healthy pitcher-seasons.
    pos_mdi: list[float] = []
    pos_vel: list[float] = []
    for pid, daily in pre_il_windowed.items():
        il_date = artifacts[pid]["il_date"]
        mask = (daily["game_date"] >= il_date - pd.Timedelta(days=30)) & \
               (daily["game_date"] <= il_date)
        pos_mdi.extend(daily.loc[mask, "mdi"].tolist())
        pos_vel.extend(daily.loc[mask, "vel_score"].tolist())

    # Negative cohort: healthy pitcher-seasons
    injured_ids = {int(p) for p in cohort["pitcher_id"].tolist()}
    healthy_pool = fetch_healthy_pool(
        conn, injured_ids, seasons=(2015, 2016),
        min_pitches=1000, max_n=max(n_healthy_roc, n_healthy_fpr) * 2,
    )
    rng = np.random.RandomState(SEED)
    rng.shuffle(healthy_pool)
    healthy_roc_pool = healthy_pool[:n_healthy_roc]
    healthy_fpr_pool = healthy_pool[:n_healthy_fpr]

    # For negatives, train one VAE per healthy pitcher on their own season data
    # then score the full season's daily MDI and sample days.
    # This is expensive (~50-100 trainings).  Keep epochs modest.
    logger.info("Training %d healthy-baseline VAEs for ROC", len(healthy_roc_pool))

    neg_mdi_all: list[float] = []
    neg_vel_all: list[float] = []
    healthy_artifacts: dict[tuple[int, int], dict] = {}
    for i, (pid, season) in enumerate(healthy_roc_pool):
        start = pd.Timestamp(f"{season}-01-01")
        end = pd.Timestamp(f"{season}-12-31")
        try:
            h_df = fetch_pitches(conn, pid, date_ge=start, date_le=end)
        except Exception as exc:
            logger.warning("fetch failed healthy %d: %s", pid, exc)
            continue
        if len(h_df) < 400:
            continue
        # Split: train on first 60%, evaluate on remaining 40% (gives "healthy" negatives)
        split_i = int(len(h_df) * 0.6)
        train_df = h_df.iloc[:split_i]
        eval_df = h_df.iloc[split_i:]
        train_feats, _, _ = df_to_feature_matrix(train_df)
        eval_feats, eval_dates, eval_speeds = df_to_feature_matrix(eval_df)
        if len(train_feats) < MIN_HEALTHY_PITCHES or len(eval_feats) < WINDOW_SIZE:
            continue
        try:
            model, _ = train_per_pitcher_model(train_feats, epochs=EPOCHS)
        except Exception as exc:
            logger.warning("healthy train fail %d,%d: %s", pid, season, exc)
            continue
        train_windows = build_sliding_windows(train_feats, WINDOW_SIZE)
        base = recon_errors_for_windows(model, train_windows)
        eval_windows = build_sliding_windows(eval_feats, WINDOW_SIZE)
        errs = recon_errors_for_windows(model, eval_windows)
        daily = per_day_mdi_series(eval_dates, errs, base, score_mode=score_mode)
        if daily.empty:
            continue
        daily_vel = daily_velocity_score(eval_dates, eval_speeds, window=10)
        daily = daily.merge(daily_vel, on="game_date", how="left").fillna({"vel_score": 0.0})
        neg_mdi_all.extend(daily["mdi"].tolist())
        neg_vel_all.extend(daily["vel_score"].tolist())
        healthy_artifacts[(pid, season)] = daily
        if (i + 1) % 10 == 0:
            logger.info("healthy trained %d/%d", i + 1, len(healthy_roc_pool))

    # Downsample negatives to roughly balance or 10:1 to reflect real prevalence
    rng = np.random.RandomState(SEED)
    if len(neg_mdi_all) > len(pos_mdi) * 10 and len(pos_mdi) > 0:
        idx = rng.choice(len(neg_mdi_all), size=len(pos_mdi) * 10, replace=False)
        neg_mdi = np.array(neg_mdi_all)[idx]
        neg_vel = np.array(neg_vel_all)[idx]
    else:
        neg_mdi = np.array(neg_mdi_all)
        neg_vel = np.array(neg_vel_all)

    y = np.concatenate([np.ones(len(pos_mdi)), np.zeros(len(neg_mdi))])
    s_mdi = np.concatenate([np.array(pos_mdi), neg_mdi])
    s_vel = np.concatenate([np.array(pos_vel), neg_vel])

    # Scale velocity scores to a comparable range for plotting only.  AUC is
    # invariant to monotone transforms, so the reported AUC doesn't change.
    if score_mode == "zscore":
        # Leave vel scores on native scale; sweep uses wide threshold grid.
        s_vel_scaled = s_vel
        vel_thresholds = [round(v, 2) for v in np.linspace(0.0, 6.0, 31).tolist()]
    else:
        if len(s_vel) > 0 and np.nanmax(s_vel) > 0:
            p99 = np.nanpercentile(s_vel, 99)
            s_vel_scaled = 100.0 * (s_vel / p99 if p99 > 0 else 1.0)
            s_vel_scaled = np.clip(s_vel_scaled, 0, 100)
        else:
            s_vel_scaled = s_vel
        vel_thresholds = roc_thresholds

    roc_mdi = roc_from_scores(y, s_mdi, thresholds=roc_thresholds)
    roc_vel = roc_from_scores(y, s_vel_scaled, thresholds=vel_thresholds)
    ci_lo, ci_hi = bootstrap_auc_ci(y, s_mdi, n_iter=200)
    ci_vel_lo, ci_vel_hi = bootstrap_auc_ci(y, s_vel_scaled, n_iter=200)

    roc_out = {
        "mdi": {
            "auc": roc_mdi["auc"],
            "auc_ci_95": [ci_lo, ci_hi],
            "tpr": roc_mdi["tpr"],
            "fpr": roc_mdi["fpr"],
            "thresholds": roc_mdi["thresholds"],
            "n_pos": roc_mdi["n_pos"],
            "n_neg": roc_mdi["n_neg"],
        },
        "velocity_drop": {
            "auc": roc_vel["auc"],
            "auc_ci_95": [ci_vel_lo, ci_vel_hi],
            "tpr": roc_vel["tpr"],
            "fpr": roc_vel["fpr"],
            "thresholds": roc_vel["thresholds"],
            "n_pos": roc_vel["n_pos"],
            "n_neg": roc_vel["n_neg"],
        },
        "delta_auc": (
            None if (math.isnan(roc_mdi["auc"]) or math.isnan(roc_vel["auc"]))
            else roc_mdi["auc"] - roc_vel["auc"]
        ),
    }
    roc_out["score_mode"] = score_mode
    (RESULTS_DIR / f"roc_curve{suffix}.json").write_text(json.dumps(roc_out, indent=2))

    # Plotly ROC
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        mdi_label = "MDI (z)" if score_mode == "zscore" else "MDI"
        fig.add_trace(go.Scatter(
            x=roc_mdi["fpr"], y=roc_mdi["tpr"], mode="lines+markers",
            name=f"{mdi_label} (AUC={roc_mdi['auc']:.3f})",
        ))
        fig.add_trace(go.Scatter(
            x=roc_vel["fpr"], y=roc_vel["tpr"], mode="lines+markers",
            name=f"Velocity-drop (AUC={roc_vel['auc']:.3f})",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random",
            line=dict(dash="dash", color="grey"),
        ))
        mode_title = "z-score" if score_mode == "zscore" else "percentile"
        fig.update_layout(
            title=f"MechanixAE 30-day pre-IL ROC ({mode_title}) "
                  f"(n_pos={roc_mdi['n_pos']}, n_neg={roc_mdi['n_neg']})",
            xaxis_title="False-positive rate",
            yaxis_title="True-positive rate",
        )
        fig.write_html(RESULTS_DIR / f"roc_curve{suffix}.html")
    except Exception as exc:
        logger.warning("Plotly ROC plot failed: %s", exc)

    # ── Step 5: FPR on healthy pitcher-seasons ───────────────────────────
    logger.info("Step 5: FPR on healthy seasons")
    # For z-score mode, the "percentile-like" buckets are z=1, 2, 3.
    # For percentile mode, the legacy 60/70/80 buckets are retained.
    if score_mode == "zscore":
        fpr_thresholds = [1.0, 2.0, 3.0]
        th_labels = ["z1", "z2", "z3"]
    else:
        fpr_thresholds = [60.0, 70.0, 80.0]
        th_labels = ["60", "70", "80"]

    fpr_rows = []
    for key, daily in healthy_artifacts.items():
        pid, season = key
        row: dict = {
            "pitcher_id": pid,
            "season": season,
            "max_mdi": float(daily["mdi"].max()) if len(daily) else None,
        }
        for th, lab in zip(fpr_thresholds, th_labels):
            row[f"pct_days_above_{lab}"] = float((daily["mdi"] >= th).mean() * 100) if len(daily) else 0.0
            row[f"breached_{lab}"] = bool((daily["mdi"] >= th).any()) if len(daily) else False
        fpr_rows.append(row)

    fpr_df = pd.DataFrame(fpr_rows)
    fpr_df.to_csv(RESULTS_DIR / f"fpr_healthy_seasons{suffix}.csv", index=False)
    fpr_summary: dict = {"n_healthy_seasons": int(len(fpr_df)), "score_mode": score_mode}
    for lab in th_labels:
        col = f"breached_{lab}"
        fpr_summary[f"pct_breach_{lab}"] = (
            float(fpr_df[col].mean() * 100) if len(fpr_df) else None
        )
    (RESULTS_DIR / f"fpr_summary{suffix}.json").write_text(json.dumps(fpr_summary, indent=2))

    # ── Summary JSON ─────────────────────────────────────────────────────
    summary = {
        "cohort": coverage,
        "score_mode": score_mode,
        "lead_time_summary": lead_summary,
        "roc": roc_out,
        "fpr": fpr_summary,
        "wall_clock_minutes": round((time.time() - t0_total) / 60, 2),
    }
    (RESULTS_DIR / f"summary{suffix}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    # Also write the legacy unsuffixed summary.json to point to the latest run,
    # preserving the "current result" expectation of downstream readers.
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    conn.close()
    return summary


# ──────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-pitchers", type=int, default=None,
                   help="Cap trained pitchers (smoke test).")
    p.add_argument("--skip-train", action="store_true",
                   help="Reuse existing checkpoints when available.")
    p.add_argument("--n-healthy-roc", type=int, default=50)
    p.add_argument("--n-healthy-fpr", type=int, default=100)
    p.add_argument("--score-mode", type=str, default="zscore",
                   choices=["percentile", "zscore"],
                   help="MDI scoring formulation.  Default: zscore (post-rescue).")
    args = p.parse_args()
    summary = run_pipeline(
        max_pitchers=args.max_pitchers,
        skip_train=args.skip_train,
        n_healthy_roc=args.n_healthy_roc,
        n_healthy_fpr=args.n_healthy_fpr,
        score_mode=args.score_mode,
    )
    # Pick the "primary" thresholds for the printed summary based on mode.
    if args.score_mode == "zscore":
        lead_key = "lead_time_t2.0"
        fpr_key = "pct_breach_z2"
    else:
        lead_key = "lead_time_t70"
        fpr_key = "pct_breach_70"
    print(json.dumps({
        "score_mode": args.score_mode,
        "n_trained": summary["cohort"]["n_trained"],
        f"median_{lead_key}": summary["lead_time_summary"].get(lead_key, {}).get("median_days"),
        "auc_mdi": summary["roc"]["mdi"]["auc"],
        "auc_vel": summary["roc"]["velocity_drop"]["auc"],
        "delta_auc": summary["roc"]["delta_auc"],
        fpr_key: summary["fpr"].get(fpr_key),
        "wall_minutes": summary["wall_clock_minutes"],
    }, indent=2, default=str))
