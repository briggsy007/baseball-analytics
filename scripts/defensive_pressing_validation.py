#!/usr/bin/env python
"""
Defensive Pressing Intensity (DPI) validation harness.

Trains the xOut HistGradientBoosting model on 2015-2022 BIP rows, scores
team-season DPI for the 30 teams across 2023 and 2024, then evaluates
five hard gates from ``docs/models/defensive_pressing_validation_spec.md``:

  Gate 1 - xOut AUC on a held-out 2023-2024 BIP sample (>= 0.70)
  Gate 2 - team DPI vs run-prevention proxy (Pearson r >= 0.40)
  Gate 3 - team DPI vs BABIP-against           (Pearson r <= -0.50)
  Gate 4 - team DPI year-over-year stability   (Pearson r >= 0.30)
  Gate 5 - season-disjoint train/test leakage  (== 0)
  Gate 6 - team DPI vs Statcast OAA (external) (Pearson r >= 0.45)

Usage
-----
    python scripts/defensive_pressing_validation.py \
        --train-start 2015 --train-end 2022 \
        --test-start 2023 --test-end 2024 \
        --output-dir results/validate_defensive_pressing_<ts>/

Outputs (into ``--output-dir``)
-------------------------------
    defensive_pressing_validation_metrics.json
    defensive_pressing_team_seasons.csv
    defensive_pressing_xout_holdout_sample.csv  (subsample for inspection)
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics import defensive_pressing as dp  # noqa: E402
from src.analytics.defensive_pressing import (  # noqa: E402
    DPIConfig,
    OUT_EVENTS,
    HIT_EVENTS,
    build_bip_features,
    _is_out,
    _is_hit,
    calculate_team_dpi,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("defensive_pressing_validation")

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPIValidationConfig:
    """Configuration for the DPI validation run."""

    train_start: int = 2015
    train_end: int = 2022
    test_start: int = 2023
    test_end: int = 2024
    output_dir: Path = ROOT / "results"
    n_bootstrap_ci: int = 1000
    random_state: int = 42
    holdout_sample_size: int = 50000  # cap on rows for AUC-on-test computation
    external_oaa_path: Path = (
        ROOT / "data" / "baselines" / "team_defense_2023_2024.parquet"
    )
    use_persisted_xout: bool = True  # load checkpoint if present
    persist_xout: bool = True  # write checkpoint after fit


# ---------------------------------------------------------------------------
# xOut training / evaluation
# ---------------------------------------------------------------------------

def _fit_xout_on_train_window(
    conn: duckdb.DuckDBPyConnection,
    cfg: DPIValidationConfig,
) -> dict[str, Any]:
    """Load the persisted xOut checkpoint or fit one on the train window.

    Mutates the module-level ``dp._xout_model`` cache so downstream
    ``calculate_team_dpi`` calls use the fitted model. Honors the
    validation config's ``use_persisted_xout`` / ``persist_xout`` flags.
    """
    seasons = list(range(cfg.train_start, cfg.train_end + 1))
    chk_path = dp.DEFAULT_XOUT_CHECKPOINT

    # Try to use a checkpoint with matching train_seasons
    if cfg.use_persisted_xout and chk_path.exists():
        loaded_model, loaded_meta = dp.load_xout_checkpoint(chk_path)
        chk_seasons = (loaded_meta or {}).get("train_seasons") or []
        if loaded_model is not None and sorted(chk_seasons) == sorted(seasons):
            logger.info(
                "Loaded xOut checkpoint from %s (matches train window %d-%d)",
                chk_path, cfg.train_start, cfg.train_end,
            )
            return {
                "status": "loaded_checkpoint",
                "loaded_from_checkpoint": True,
                "persist_path": str(chk_path),
                "train_seasons": chk_seasons,
                "auc": loaded_meta.get("auc"),
                "train_auc": loaded_meta.get("train_auc"),
                "n_samples": loaded_meta.get("n_samples"),
                "out_rate": loaded_meta.get("out_rate"),
                "use_park": loaded_meta.get("use_park", False),
                "feature_columns": loaded_meta.get("feature_columns"),
                "fitted_at": loaded_meta.get("fitted_at"),
            }
        logger.info(
            "Existing checkpoint train_seasons=%s did not match requested %s; refitting",
            chk_seasons, seasons,
        )

    # Fit (and optionally persist)
    logger.info(
        "Fitting xOut classifier on BIP rows from seasons %d-%d",
        cfg.train_start, cfg.train_end,
    )
    persist_path = chk_path if cfg.persist_xout else None
    metrics = dp.fit_xout(
        conn, seasons=seasons, persist_path=persist_path, use_park=False,
    )
    metrics["train_seasons"] = seasons
    metrics["loaded_from_checkpoint"] = False
    return metrics


def _evaluate_xout_on_holdout(
    conn: duckdb.DuckDBPyConnection,
    cfg: DPIValidationConfig,
) -> dict[str, Any]:
    """Compute xOut AUC on a 2023-2024 holdout BIP sample."""
    from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

    if dp._xout_model is None:
        return {"status": "no_model", "test_auc": None, "test_n": 0}

    # Pull the test BIP cohort. Cap rows for memory.
    seasons_in = ", ".join(str(s) for s in range(cfg.test_start, cfg.test_end + 1))
    query = f"""
        SELECT
            launch_speed, launch_angle, hc_x, hc_y, bb_type, events
        FROM pitches
        WHERE type = 'X'
          AND EXTRACT(YEAR FROM game_date) IN ({seasons_in})
          AND launch_speed IS NOT NULL
          AND launch_angle IS NOT NULL
          AND hc_x IS NOT NULL
          AND hc_y IS NOT NULL
          AND bb_type IS NOT NULL
          AND events IS NOT NULL
    """
    df = conn.execute(query).fetchdf()

    if df.empty:
        return {"status": "no_test_data", "test_auc": None, "test_n": 0}

    if len(df) > cfg.holdout_sample_size > 0:
        df = df.sample(
            n=cfg.holdout_sample_size,
            random_state=cfg.random_state,
        ).reset_index(drop=True)

    features = build_bip_features(df)
    target = _is_out(df["events"]).astype(int)
    mask = features.notna().all(axis=1)
    features = features[mask]
    target = target[mask]

    if len(features) < 100:
        return {
            "status": "insufficient_holdout",
            "test_auc": None,
            "test_n": int(len(features)),
        }

    proba = dp._xout_model.predict_proba(features)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(target, proba))
    acc = float(accuracy_score(target, pred))
    try:
        ll = float(log_loss(target, np.clip(proba, 1e-9, 1 - 1e-9)))
    except Exception:
        ll = None

    sample_df = pd.DataFrame({
        "launch_speed": features["launch_speed"].values,
        "launch_angle": features["launch_angle"].values,
        "spray_angle": features["spray_angle"].values,
        "bb_type_encoded": features["bb_type_encoded"].values,
        "actual_out": target.values,
        "pred_out_prob": proba,
    })

    return {
        "status": "ok",
        "test_auc": round(auc, 4),
        "test_accuracy": round(acc, 4),
        "test_log_loss": round(ll, 4) if ll is not None else None,
        "test_n": int(len(features)),
        "test_out_rate": round(float(target.mean()), 4),
        "_sample_df": sample_df,
    }


# ---------------------------------------------------------------------------
# Team-season DPI
# ---------------------------------------------------------------------------

def _list_teams_in_season(
    conn: duckdb.DuckDBPyConnection, season: int,
) -> list[str]:
    """All teams that appear in pitches for the given season."""
    df = conn.execute(
        """
        SELECT DISTINCT t FROM (
            SELECT home_team AS t FROM pitches
            WHERE EXTRACT(YEAR FROM game_date) = $1
            UNION
            SELECT away_team AS t FROM pitches
            WHERE EXTRACT(YEAR FROM game_date) = $1
        ) ORDER BY t
        """,
        [int(season)],
    ).fetchdf()
    return [str(t) for t in df["t"].tolist()]


def _compute_team_season_dpi(
    conn: duckdb.DuckDBPyConnection,
    cfg: DPIValidationConfig,
) -> pd.DataFrame:
    """Compute DPI profile for every (team, season) in the test window."""
    rows: list[dict[str, Any]] = []
    config = DPIConfig(random_state=cfg.random_state)

    seasons = list(range(cfg.test_start, cfg.test_end + 1))
    for season in seasons:
        teams = _list_teams_in_season(conn, season)
        logger.info(
            "Scoring DPI for season %d across %d teams", season, len(teams),
        )
        t0 = time.monotonic()
        for i, team in enumerate(teams):
            profile = calculate_team_dpi(conn, team, season, config)
            if profile.get("dpi_mean") is None:
                continue
            rows.append({
                "team_id": str(team),
                "season": int(season),
                "dpi_mean": float(profile["dpi_mean"]),
                "dpi_total": float(profile["dpi_total"]),
                "dpi_std": float(profile.get("dpi_std", 0.0)),
                "consistency": float(profile["consistency"]),
                "extra_base_prevention": (
                    float(profile["extra_base_prevention"])
                    if profile["extra_base_prevention"] is not None else None
                ),
                "n_games": int(profile["n_games"]),
            })
            if (i + 1) % 5 == 0:
                rate = (i + 1) / max(time.monotonic() - t0, 1e-6)
                logger.info(
                    "  ...%d/%d teams (%.2f teams/s)",
                    i + 1, len(teams), rate,
                )
        logger.info(
            "Season %d DPI complete in %.1fs", season, time.monotonic() - t0,
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run-prevention proxies (computed from the same DB)
# ---------------------------------------------------------------------------

def _team_run_prevention_proxy(
    conn: duckdb.DuckDBPyConnection,
    seasons: list[int],
) -> pd.DataFrame:
    """Per (team, season) defensive run-prevention proxy.

    Sum of ``delta_run_exp`` over all pitches where the team was on
    defense (opposing batters at the plate). Higher = MORE runs added by
    opponents = WORSE defense. We invert sign in the metric to align with
    DPI direction (higher = better defense).
    """
    seasons_in = ", ".join(str(s) for s in seasons)
    query = f"""
        SELECT
            CASE WHEN inning_topbot = 'Top' THEN home_team ELSE away_team END
                AS team_id,
            EXTRACT(YEAR FROM game_date)::INT AS season,
            COUNT(*) AS n_pitches_on_def,
            COUNT(DISTINCT game_pk) AS n_def_games,
            SUM(delta_run_exp) AS sum_delta_run_exp_on_def,
            SUM(CASE WHEN type='X' THEN 1 ELSE 0 END) AS n_bip_on_def
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) IN ({seasons_in})
          AND delta_run_exp IS NOT NULL
        GROUP BY team_id, season
        ORDER BY team_id, season
    """
    df = conn.execute(query).fetchdf()
    df["rp_proxy"] = -df["sum_delta_run_exp_on_def"].astype(float)
    return df


def _team_babip_against(
    conn: duckdb.DuckDBPyConnection,
    seasons: list[int],
) -> pd.DataFrame:
    """Per (team, season) BABIP-against.

    Standard BABIP-against = (H - HR) / (AB - K - HR + SF). We approximate
    with the BIP cohort directly:

        BABIP_against = non_HR_hits_on_def / (BIP_on_def - HR_on_def)
                      = (single + double + triple) / BIP_in_play

    where BIP rows are ``type='X'`` (ball put in play, includes HR).
    """
    seasons_in = ", ".join(str(s) for s in seasons)
    query = f"""
        SELECT
            CASE WHEN inning_topbot = 'Top' THEN home_team ELSE away_team END
                AS team_id,
            EXTRACT(YEAR FROM game_date)::INT AS season,
            COUNT(*) AS n_bip,
            SUM(CASE WHEN events IN ('single','double','triple') THEN 1 ELSE 0 END)
                AS n_non_hr_hits,
            SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END)
                AS n_hr,
            SUM(CASE WHEN events = 'single' THEN 1 ELSE 0 END) AS n_singles,
            SUM(CASE WHEN events = 'double' THEN 1 ELSE 0 END) AS n_doubles,
            SUM(CASE WHEN events = 'triple' THEN 1 ELSE 0 END) AS n_triples
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) IN ({seasons_in})
          AND type = 'X'
          AND events IS NOT NULL
        GROUP BY team_id, season
        ORDER BY team_id, season
    """
    df = conn.execute(query).fetchdf()
    bip_in_play = (df["n_bip"] - df["n_hr"]).astype(float)
    df["babip_against"] = np.where(
        bip_in_play > 0,
        df["n_non_hr_hits"].astype(float) / bip_in_play,
        np.nan,
    )
    return df


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    sx = pd.Series(x).rank(method="average")
    sy = pd.Series(y).rank(method="average")
    if sx.std() == 0 or sy.std() == 0:
        return float("nan")
    return float(sx.corr(sy))


def _bootstrap_ci(
    x: np.ndarray, y: np.ndarray, *,
    n_boot: int, random_state: int, fn=None,
) -> Optional[list[float]]:
    """Generic 95% percentile bootstrap CI on a function of two paired arrays."""
    if fn is None:
        fn = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    n = int(len(x))
    if n < 3 or n_boot < 10:
        return None
    rng = np.random.RandomState(random_state)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        a = x[idx]
        b = y[idx]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        try:
            boots.append(float(fn(a, b)))
        except Exception:
            continue
    if len(boots) < max(10, n_boot // 4):
        return None
    return [
        round(float(np.percentile(boots, 2.5)), 4),
        round(float(np.percentile(boots, 97.5)), 4),
    ]


def _pair_metrics(
    a: np.ndarray, b: np.ndarray, *,
    n_boot: int, random_state: int,
) -> dict[str, Any]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    n = int(len(a))
    out: dict[str, Any] = {
        "n": n,
        "pearson_r": None,
        "pearson_r_ci": None,
        "spearman_rho": None,
        "spearman_rho_ci": None,
    }
    if n < 3 or np.std(a) == 0 or np.std(b) == 0:
        return out
    r = float(np.corrcoef(a, b)[0, 1])
    rho = float(_spearman_rho(a, b))
    r_ci = _bootstrap_ci(
        a, b, n_boot=n_boot, random_state=random_state,
    )
    rho_ci = _bootstrap_ci(
        a, b, n_boot=n_boot, random_state=random_state + 1,
        fn=_spearman_rho,
    )
    out.update({
        "pearson_r": round(r, 4),
        "pearson_r_ci": r_ci,
        "spearman_rho": round(rho, 4),
        "spearman_rho_ci": rho_ci,
    })
    return out


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def _leakage_audit(
    conn: duckdb.DuckDBPyConnection, cfg: DPIValidationConfig,
) -> dict[str, Any]:
    train_pks = set(int(x) for x in conn.execute(
        f"""
        SELECT DISTINCT game_pk FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) BETWEEN {cfg.train_start} AND {cfg.train_end}
        """
    ).fetchdf()["game_pk"].tolist())
    test_pks = set(int(x) for x in conn.execute(
        f"""
        SELECT DISTINCT game_pk FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) BETWEEN {cfg.test_start} AND {cfg.test_end}
        """
    ).fetchdf()["game_pk"].tolist())
    overlap = train_pks & test_pks
    return {
        "train_seasons": [cfg.train_start, cfg.train_end],
        "test_seasons": [cfg.test_start, cfg.test_end],
        "n_train_game_pks": int(len(train_pks)),
        "n_test_game_pks": int(len(test_pks)),
        "shared_game_pks_train_test": int(len(overlap)),
        "season_disjoint": bool(cfg.train_end < cfg.test_start),
        "pass": bool(len(overlap) == 0 and cfg.train_end < cfg.test_start),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_validation(cfg: DPIValidationConfig) -> dict[str, Any]:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = get_connection(read_only=True)

    logger.info(
        "DPI validation: xOut train %d-%d, team-DPI test %d-%d",
        cfg.train_start, cfg.train_end, cfg.test_start, cfg.test_end,
    )

    # ---- Step 1: leakage audit -------------------------------------------
    leakage = _leakage_audit(conn, cfg)
    logger.info(
        "Leakage: %d train pks, %d test pks, %d shared",
        leakage["n_train_game_pks"], leakage["n_test_game_pks"],
        leakage["shared_game_pks_train_test"],
    )

    # ---- Step 2: train xOut on train window ------------------------------
    t0 = time.monotonic()
    xout_train_metrics = _fit_xout_on_train_window(conn, cfg)
    xout_train_secs = time.monotonic() - t0
    logger.info(
        "xOut train (train-window AUC) = %s, %d samples (%.1fs)",
        xout_train_metrics.get("auc"),
        xout_train_metrics.get("n_samples"),
        xout_train_secs,
    )

    # ---- Step 3: eval xOut on test holdout -------------------------------
    t0 = time.monotonic()
    xout_test_metrics = _evaluate_xout_on_holdout(conn, cfg)
    xout_test_secs = time.monotonic() - t0
    sample_df = xout_test_metrics.pop("_sample_df", None)
    logger.info(
        "xOut test holdout AUC = %s on n=%d (%.1fs)",
        xout_test_metrics.get("test_auc"),
        xout_test_metrics.get("test_n"),
        xout_test_secs,
    )

    # ---- Step 4: compute team-season DPI ---------------------------------
    t0 = time.monotonic()
    dpi_df = _compute_team_season_dpi(conn, cfg)
    dpi_secs = time.monotonic() - t0
    logger.info(
        "Team DPI computed for %d team-seasons in %.1fs",
        len(dpi_df), dpi_secs,
    )

    # ---- Step 5: compute proxies -----------------------------------------
    seasons = list(range(cfg.test_start, cfg.test_end + 1))
    rp_df = _team_run_prevention_proxy(conn, seasons)
    babip_df = _team_babip_against(conn, seasons)

    merged = dpi_df.merge(
        rp_df[["team_id", "season", "rp_proxy",
               "sum_delta_run_exp_on_def", "n_def_games", "n_bip_on_def"]],
        on=["team_id", "season"], how="left",
    ).merge(
        babip_df[["team_id", "season", "babip_against", "n_bip"]],
        on=["team_id", "season"], how="left",
    )

    # External Statcast OAA / FRP (FanGraphs DRS endpoint blocked w/ 403;
    # Statcast OAA is the equivalent gold-standard team-defense baseline).
    external_oaa_status: dict[str, Any] = {
        "available": False,
        "path": str(cfg.external_oaa_path),
    }
    try:
        if cfg.external_oaa_path.exists():
            ext = pd.read_parquet(cfg.external_oaa_path)
            ext = ext.rename(columns={"team_abbr": "team_id"})
            keep = ["team_id", "season", "team_oaa", "team_frp", "n_players"]
            keep = [c for c in keep if c in ext.columns]
            merged = merged.merge(
                ext[keep], on=["team_id", "season"], how="left",
            )
            external_oaa_status.update({
                "available": True,
                "n_rows_loaded": int(len(ext)),
                "n_team_seasons_with_oaa": int(merged["team_oaa"].notna().sum())
                if "team_oaa" in merged.columns else 0,
            })
            logger.info(
                "Loaded external Statcast OAA from %s (%d rows, %d merged)",
                cfg.external_oaa_path,
                external_oaa_status["n_rows_loaded"],
                external_oaa_status["n_team_seasons_with_oaa"],
            )
        else:
            logger.warning(
                "External OAA parquet not found at %s; Gate 6 will FAIL/SKIP",
                cfg.external_oaa_path,
            )
    except Exception as exc:
        logger.warning("Could not load external OAA baseline: %s", exc)
        external_oaa_status["error"] = str(exc)

    # ---- Step 6: gate metrics --------------------------------------------
    rng_state = cfg.random_state

    metrics_dpi_vs_rp = _pair_metrics(
        merged["dpi_mean"].to_numpy(),
        merged["rp_proxy"].to_numpy(),
        n_boot=cfg.n_bootstrap_ci, random_state=rng_state,
    )
    metrics_dpi_vs_babip = _pair_metrics(
        merged["dpi_mean"].to_numpy(),
        merged["babip_against"].to_numpy(),
        n_boot=cfg.n_bootstrap_ci, random_state=rng_state + 10,
    )

    # External Statcast OAA correlation (Gate 6)
    if "team_oaa" in merged.columns:
        metrics_dpi_vs_oaa = _pair_metrics(
            merged["dpi_mean"].to_numpy(),
            merged["team_oaa"].to_numpy(),
            n_boot=cfg.n_bootstrap_ci, random_state=rng_state + 30,
        )
        metrics_dpi_vs_frp = _pair_metrics(
            merged["dpi_mean"].to_numpy(),
            merged["team_frp"].to_numpy(),
            n_boot=cfg.n_bootstrap_ci, random_state=rng_state + 31,
        )
    else:
        metrics_dpi_vs_oaa = {
            "n": 0, "pearson_r": None, "pearson_r_ci": None,
            "spearman_rho": None, "spearman_rho_ci": None,
        }
        metrics_dpi_vs_frp = dict(metrics_dpi_vs_oaa)

    # YoY stability
    pivot = (
        merged.pivot(index="team_id", columns="season", values="dpi_mean")
        .dropna(how="any")
    )
    if cfg.test_start in pivot.columns and cfg.test_end in pivot.columns:
        a = pivot[cfg.test_start].to_numpy()
        b = pivot[cfg.test_end].to_numpy()
    else:
        a = np.array([])
        b = np.array([])
    metrics_yoy = _pair_metrics(
        a, b, n_boot=cfg.n_bootstrap_ci, random_state=rng_state + 20,
    )

    # ---- Step 7: gate evaluation -----------------------------------------
    gates: list[dict[str, Any]] = []

    # Gate 1 - xOut AUC on holdout
    g1_meas = xout_test_metrics.get("test_auc")
    gates.append({
        "name": "xout_test_auc",
        "threshold": ">= 0.70",
        "measured": g1_meas,
        "operator": ">=",
        "pass": bool(g1_meas is not None and g1_meas >= 0.70),
        "hard": True,
        "source": "metrics.xout.test_auc",
    })

    # Gate 2 - DPI vs RP proxy
    g2_meas = metrics_dpi_vs_rp.get("pearson_r")
    gates.append({
        "name": "dpi_vs_rp_proxy_pearson_r",
        "threshold": ">= 0.40",
        "measured": g2_meas,
        "measured_ci": metrics_dpi_vs_rp.get("pearson_r_ci"),
        "operator": ">=",
        "pass": bool(g2_meas is not None and g2_meas >= 0.40),
        "hard": True,
        "source": "metrics.consumer_claim.dpi_vs_rp_proxy.pearson_r",
    })

    # Gate 3 - DPI vs BABIP-against (negative correlation expected)
    g3_meas = metrics_dpi_vs_babip.get("pearson_r")
    gates.append({
        "name": "dpi_vs_babip_against_pearson_r",
        "threshold": "<= -0.50",
        "measured": g3_meas,
        "measured_ci": metrics_dpi_vs_babip.get("pearson_r_ci"),
        "operator": "<=",
        "pass": bool(g3_meas is not None and g3_meas <= -0.50),
        "hard": True,
        "source": "metrics.consumer_claim.dpi_vs_babip_against.pearson_r",
    })

    # Gate 4 - YoY stability
    g4_meas = metrics_yoy.get("pearson_r")
    gates.append({
        "name": "dpi_yoy_pearson_r",
        "threshold": ">= 0.30",
        "measured": g4_meas,
        "measured_ci": metrics_yoy.get("pearson_r_ci"),
        "operator": ">=",
        "pass": bool(g4_meas is not None and g4_meas >= 0.30),
        "hard": True,
        "source": "metrics.stability.yoy_pearson_r",
    })

    # Gate 5 - Leakage
    gates.append({
        "name": "leakage_audit",
        "threshold": "0 game_pks shared train/test",
        "measured": leakage.get("shared_game_pks_train_test"),
        "operator": "==",
        "pass": bool(leakage.get("pass")),
        "hard": True,
        "source": "leakage_audit.shared_game_pks_train_test",
    })

    # Gate 6 - External Statcast OAA correlation
    g6_meas = metrics_dpi_vs_oaa.get("pearson_r")
    gates.append({
        "name": "dpi_vs_external_oaa_pearson_r",
        "threshold": ">= 0.45",
        "measured": g6_meas,
        "measured_ci": metrics_dpi_vs_oaa.get("pearson_r_ci"),
        "operator": ">=",
        "pass": bool(g6_meas is not None and g6_meas >= 0.45),
        "hard": True,
        "source": "metrics.external_baseline.dpi_vs_oaa.pearson_r",
        "note": (
            "Statcast Outs Above Average (Baseball Savant), aggregated to "
            "team-season. FanGraphs DRS endpoint returned 403; OAA is the "
            "equivalent gold-standard team-defense baseline."
        ),
    })

    overall_pass = all(g["pass"] for g in gates if g["hard"])

    # ---- Step 8: persist artifacts ---------------------------------------
    team_seasons_path = output_dir / "defensive_pressing_team_seasons.csv"
    merged.to_csv(team_seasons_path, index=False)

    sample_path = None
    if sample_df is not None:
        sample_path = output_dir / "defensive_pressing_xout_holdout_sample.csv"
        # Cap on disk for inspection
        sample_df.sample(
            n=min(5000, len(sample_df)),
            random_state=cfg.random_state,
        ).to_csv(sample_path, index=False)

    payload: dict[str, Any] = {
        "config": {
            "train_start": cfg.train_start,
            "train_end": cfg.train_end,
            "test_start": cfg.test_start,
            "test_end": cfg.test_end,
            "n_bootstrap_ci": cfg.n_bootstrap_ci,
            "random_state": cfg.random_state,
            "holdout_sample_size": cfg.holdout_sample_size,
        },
        "leakage_audit": leakage,
        "metrics": {
            "xout": {
                "train": xout_train_metrics,
                "test_auc": xout_test_metrics.get("test_auc"),
                "test_accuracy": xout_test_metrics.get("test_accuracy"),
                "test_log_loss": xout_test_metrics.get("test_log_loss"),
                "test_n": xout_test_metrics.get("test_n"),
                "test_out_rate": xout_test_metrics.get("test_out_rate"),
            },
            "consumer_claim": {
                "dpi_vs_rp_proxy": metrics_dpi_vs_rp,
                "dpi_vs_babip_against": metrics_dpi_vs_babip,
            },
            "external_baseline": {
                "source": external_oaa_status,
                "dpi_vs_oaa": metrics_dpi_vs_oaa,
                "dpi_vs_frp": metrics_dpi_vs_frp,
            },
            "stability": {
                "yoy_pearson_r": metrics_yoy.get("pearson_r"),
                "yoy_pearson_r_ci": metrics_yoy.get("pearson_r_ci"),
                "yoy_spearman_rho": metrics_yoy.get("spearman_rho"),
                "yoy_n_teams": metrics_yoy.get("n"),
            },
            "data_quality": {
                "n_team_seasons": int(len(merged)),
                "n_team_seasons_with_dpi": int(merged["dpi_mean"].notna().sum()),
                "n_team_seasons_with_rp": int(merged["rp_proxy"].notna().sum()),
                "n_team_seasons_with_babip": int(
                    merged["babip_against"].notna().sum()
                ),
            },
        },
        "gates": gates,
        "overall_pass": bool(overall_pass),
        "wall_clock_seconds": {
            "xout_train": round(xout_train_secs, 2),
            "xout_test": round(xout_test_secs, 2),
            "team_dpi": round(dpi_secs, 2),
        },
        "artifacts": {
            "team_seasons_csv": str(team_seasons_path),
            "xout_holdout_sample_csv": (
                str(sample_path) if sample_path is not None else None
            ),
        },
    }

    metrics_path = output_dir / "defensive_pressing_validation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
    payload["artifacts"]["metrics_json"] = str(metrics_path)
    logger.info("Wrote %s", metrics_path)

    conn.close()

    _print_summary(payload)
    return payload


def _print_summary(payload: dict[str, Any]) -> None:
    m = payload["metrics"]
    g = payload["gates"]
    print()
    print("=" * 72)
    print("Defensive Pressing (DPI) Validation -- Summary")
    print("=" * 72)
    xt = m["xout"]
    print(
        f"xOut: train_auc={xt['train'].get('auc')}  "
        f"test_auc={xt.get('test_auc')}  test_n={xt.get('test_n')}  "
        f"test_out_rate={xt.get('test_out_rate')}"
    )
    cc = m["consumer_claim"]
    rp = cc["dpi_vs_rp_proxy"]
    bb = cc["dpi_vs_babip_against"]
    print(
        f"DPI vs RP proxy:        n={rp['n']}  "
        f"r={rp['pearson_r']}  CI={rp['pearson_r_ci']}  "
        f"rho={rp['spearman_rho']}"
    )
    print(
        f"DPI vs BABIP-against:   n={bb['n']}  "
        f"r={bb['pearson_r']}  CI={bb['pearson_r_ci']}  "
        f"rho={bb['spearman_rho']}"
    )
    st = m["stability"]
    print(
        f"DPI YoY stability:      n_teams={st['yoy_n_teams']}  "
        f"r={st['yoy_pearson_r']}  CI={st['yoy_pearson_r_ci']}"
    )
    if "external_baseline" in m:
        ext = m["external_baseline"]
        oaa = ext.get("dpi_vs_oaa", {})
        print(
            f"DPI vs Statcast OAA:    n={oaa.get('n')}  "
            f"r={oaa.get('pearson_r')}  CI={oaa.get('pearson_r_ci')}  "
            f"rho={oaa.get('spearman_rho')}"
        )
    print("-" * 72)
    for gate in g:
        verdict = "PASS" if gate["pass"] else "FAIL"
        print(
            f"  [{verdict}] {gate['name']:34s} thr={gate['threshold']:30s} "
            f"meas={gate['measured']}"
        )
    print("-" * 72)
    print(f"OVERALL: {'PASS' if payload['overall_pass'] else 'FAIL'}")
    print("=" * 72)
    print()


def _json_safe(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        if isinstance(v, np.floating) and (math.isnan(v) or math.isinf(v)):
            return None
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, Path):
        return str(v)
    return str(v)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__ or "",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--train-start", type=int, default=2015)
    p.add_argument("--train-end", type=int, default=2022)
    p.add_argument("--test-start", type=int, default=2023)
    p.add_argument("--test-end", type=int, default=2024)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument("--n-bootstrap-ci", type=int, default=1000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--holdout-sample-size", type=int, default=50000,
        help="Cap on test BIP rows for AUC computation (0 = use all).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = DPIValidationConfig(
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        output_dir=args.output_dir,
        n_bootstrap_ci=args.n_bootstrap_ci,
        random_state=args.random_state,
        holdout_sample_size=args.holdout_sample_size,
    )
    try:
        payload = run_validation(cfg)
        if "error" in payload:
            return 2
        return 0 if payload.get("overall_pass") else 1
    except Exception as exc:
        logger.exception("defensive_pressing_validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
