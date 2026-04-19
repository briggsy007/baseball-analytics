#!/usr/bin/env python
"""
ChemNet validation harness.

Loads ``models/chemnet_v1.pt`` (the GNN + baseline pair) and evaluates it on
the held-out 2023-2024 game-side cohort. Computes Pearson r, RMSE, MAE for
both the GNN and the baseline; the synergy lift (GNN edge over baseline);
the rank correlation between synergy_score and residual_wOBA (the
"synergy predicts overshoot" claim); a leakage audit; and decile lift
diagnostics.

Implements the ChemNet validation gates from
``docs/models/chemnet_validation_spec.md``.

Usage
-----
    python scripts/chemnet_validation.py \
        --train-start 2015 --train-end 2022 \
        --test-start 2023 --test-end 2024 \
        --output-dir results/validate_chemnet_<ts>/

Outputs (into ``--output-dir``)
--------------------------------
    chemnet_validation_metrics.json
    chemnet_validation_per_game.csv
    chemnet_validation_decile_lift.csv
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics.chemnet import (  # noqa: E402
    NUM_LINEUP_SLOTS,
    SLOT_PA_SCHEDULE,
    _load_models,
    _load_models_v2,
    build_game_graph,
    build_game_graph_v2,
    _pad_graph,
    _pad_graph_v2,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("chemnet_validation")

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True,
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ChemNetValidationConfig:
    """Configuration for the ChemNet validation run."""

    train_start: int = 2015
    train_end: int = 2022
    test_start: int = 2023
    test_end: int = 2024
    output_dir: Path = ROOT / "results"
    max_test_games: int = 0   # 0 = use all qualifying games
    n_bootstrap_ci: int = 1000
    random_state: int = 42
    model_version: str = "1"
    pa_min_per_side: int = 25  # require at least 25 PAs in the side to be a "real" lineup game
    checkpoint_path: Optional[Path] = None  # if set, load this v2 checkpoint
    model_mode: str = "v1"  # 'v1' or 'v2' (residual + pitcher graph)


# ---------------------------------------------------------------------------
# Cohort selection + leakage audit
# ---------------------------------------------------------------------------

def _list_test_game_pks(
    conn: duckdb.DuckDBPyConnection,
    test_start: int,
    test_end: int,
) -> list[int]:
    """List 2023-2024 game_pks with enough batters to build a 9-node lineup."""
    df = conn.execute(
        f"""
        SELECT game_pk
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) BETWEEN {int(test_start)} AND {int(test_end)}
        GROUP BY game_pk
        HAVING COUNT(DISTINCT batter_id) >= 18
        ORDER BY game_pk
        """
    ).fetchdf()
    return [int(x) for x in df["game_pk"].tolist()]


def _list_train_game_pks(
    conn: duckdb.DuckDBPyConnection,
    train_start: int,
    train_end: int,
) -> set[int]:
    """List 2015-2022 game_pks (the assumed training reference range)."""
    df = conn.execute(
        f"""
        SELECT DISTINCT game_pk
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) BETWEEN {int(train_start)} AND {int(train_end)}
        """
    ).fetchdf()
    return set(int(x) for x in df["game_pk"].tolist())


def _leakage_audit(
    conn: duckdb.DuckDBPyConnection, cfg: ChemNetValidationConfig,
    test_pks: list[int],
) -> dict[str, Any]:
    """Verify season-disjoint train/test cohorts."""
    train_pks = _list_train_game_pks(conn, cfg.train_start, cfg.train_end)
    test_set = set(int(x) for x in test_pks)
    overlap = train_pks & test_set
    return {
        "train_seasons": [cfg.train_start, cfg.train_end],
        "test_seasons": [cfg.test_start, cfg.test_end],
        "n_train_game_pks": int(len(train_pks)),
        "n_test_game_pks": int(len(test_set)),
        "shared_game_pks_train_test": int(len(overlap)),
        "season_disjoint": bool(cfg.train_end < cfg.test_start),
        "pass": bool(len(overlap) == 0 and cfg.train_end < cfg.test_start),
    }


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def _score_one_game_side(
    conn: duckdb.DuckDBPyConnection,
    game_pk: int,
    side: str,
    gnn,
    baseline,
    device: torch.device,
) -> Optional[dict[str, Any]]:
    """Run ChemNet v1 + baseline on one game-side and return both predictions
    plus the actual SUM(woba_value)."""
    g = build_game_graph(conn, game_pk, side)
    if g is None:
        return None
    if g["n_nodes"] < 2:
        return None
    g = _pad_graph(g)
    nf = g["node_features"].to(device)
    adj = g["adj"].to(device)
    with torch.no_grad():
        pred_gnn, _ = gnn(nf, adj)
        pred_base = baseline(nf)
    actual = float(g["target"])
    return {
        "game_pk": int(game_pk),
        "side": side,
        "n_nodes": int(g["n_nodes"]),
        "gnn_pred": float(pred_gnn.item()),
        "baseline_pred": float(pred_base.item()),
        "actual_total_woba": actual,
        "synergy_score": float(pred_gnn.item()) - float(pred_base.item()),
    }


def _score_one_game_side_v2(
    conn: duckdb.DuckDBPyConnection,
    game_pk: int,
    side: str,
    gnn,
    baseline,
    pitcher_norm: dict,
    device: torch.device,
) -> Optional[dict[str, Any]]:
    """Run ChemNet v2 (residual objective + pitcher graph) on one game-side.

    Returns BOTH the residual predictions and the absolute-equivalent
    predictions (sum_of_parts + residual_pred). The downstream gates apply
    to the absolute-equivalent values so v1/v2 are directly comparable.
    """
    g = build_game_graph_v2(conn, game_pk, side, pitcher_norm=pitcher_norm)
    if g is None or g["n_lineup"] < 2:
        return None
    g = _pad_graph_v2(g)
    nf = g["node_features"].to(device)
    adj = g["adj"].to(device)
    with torch.no_grad():
        pred_gnn_res, _ = gnn(nf, adj)
        pred_base_res = baseline(nf)
    actual = float(g["target"])
    sop = float(g["sum_of_parts"])
    gnn_res = float(pred_gnn_res.item())
    base_res = float(pred_base_res.item())
    gnn_abs = sop + gnn_res
    base_abs = sop + base_res
    return {
        "game_pk": int(game_pk),
        "side": side,
        "n_nodes": int(g["n_nodes"]),
        "gnn_pred": gnn_abs,
        "baseline_pred": base_abs,
        "gnn_pred_residual": gnn_res,
        "baseline_pred_residual": base_res,
        "sum_of_parts": sop,
        "actual_total_woba": actual,
        "synergy_score": gnn_abs - base_abs,  # equivalently gnn_res - base_res
    }


def _filter_by_pa(
    conn: duckdb.DuckDBPyConnection,
    rows: list[dict[str, Any]],
    pa_min: int,
) -> tuple[list[dict[str, Any]], int]:
    """Drop game-sides with fewer than ``pa_min`` recorded PAs.

    Reason: late-season blowouts and shortened games can produce sparse
    targets that distort the per-game RMSE without changing the model's
    quality. Returns (kept, dropped_count).
    """
    if not rows:
        return [], 0

    # batch query: SUM(woba_denom) per game_pk x inning_topbot
    pks = list({r["game_pk"] for r in rows})
    if not pks:
        return rows, 0
    pk_list = ", ".join(str(int(p)) for p in pks)
    df = conn.execute(
        f"""
        SELECT game_pk, inning_topbot,
               COALESCE(SUM(woba_denom), 0) AS pas
        FROM pitches
        WHERE game_pk IN ({pk_list}) AND woba_denom IS NOT NULL AND woba_denom > 0
        GROUP BY game_pk, inning_topbot
        """
    ).fetchdf()
    pa_map: dict[tuple[int, str], int] = {}
    for _, r in df.iterrows():
        side_key = "home" if str(r["inning_topbot"]) == "Bot" else "away"
        pa_map[(int(r["game_pk"]), side_key)] = int(r["pas"])

    kept: list[dict[str, Any]] = []
    dropped = 0
    for r in rows:
        pas = pa_map.get((int(r["game_pk"]), str(r["side"])), 0)
        if pas >= pa_min:
            r2 = dict(r)
            r2["n_pa"] = pas
            kept.append(r2)
        else:
            dropped += 1
    return kept, dropped


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation via pandas (avoids hard scipy dep)."""
    sx = pd.Series(x).rank(method="average")
    sy = pd.Series(y).rank(method="average")
    if sx.std() == 0 or sy.std() == 0:
        return float("nan")
    return float(sx.corr(sy))


def _bootstrap_ci(
    fn,
    n: int,
    *,
    n_boot: int,
    rng: np.random.RandomState,
) -> Optional[list[float]]:
    """Generic 95% percentile bootstrap CI on a scalar function of paired
    indices. ``fn(idx)`` returns a float."""
    if n < 3 or n_boot < 10:
        return None
    boots: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.choice(n, size=n, replace=True)
        try:
            v = float(fn(idx))
        except Exception:
            continue
        if not math.isfinite(v):
            continue
        boots.append(v)
    if len(boots) < max(10, n_boot // 4):
        return None
    return [
        round(float(np.percentile(boots, 2.5)), 4),
        round(float(np.percentile(boots, 97.5)), 4),
    ]


def _basic_metrics(
    pred: np.ndarray, actual: np.ndarray, *, n_boot: int, random_state: int,
) -> dict[str, Any]:
    """Pearson r, Spearman rho, RMSE, MAE with bootstrap CIs on r."""
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    mask = np.isfinite(pred) & np.isfinite(actual)
    pred = pred[mask]
    actual = actual[mask]
    n = int(len(pred))

    out: dict[str, Any] = {
        "n": n,
        "pearson_r": None,
        "pearson_r_ci": None,
        "spearman_rho": None,
        "rmse": None,
        "mae": None,
    }
    if n < 3 or np.std(pred) == 0 or np.std(actual) == 0:
        return out

    r = float(np.corrcoef(pred, actual)[0, 1])
    rho = float(_spearman_rho(pred, actual))
    rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
    mae = float(np.mean(np.abs(pred - actual)))

    rng = np.random.RandomState(random_state)

    def _f(idx):
        p = pred[idx]
        a = actual[idx]
        if np.std(p) == 0 or np.std(a) == 0:
            return float("nan")
        return float(np.corrcoef(p, a)[0, 1])

    ci = _bootstrap_ci(_f, n, n_boot=n_boot, rng=rng)

    out.update({
        "pearson_r": round(r, 4),
        "pearson_r_ci": ci,
        "spearman_rho": round(rho, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
    })
    return out


def _synergy_residual_metrics(
    synergy: np.ndarray,
    residual: np.ndarray,
    *,
    n_boot: int,
    random_state: int,
) -> dict[str, Any]:
    """Spearman rho (synergy, residual) with bootstrap CI."""
    synergy = np.asarray(synergy, dtype=float)
    residual = np.asarray(residual, dtype=float)
    mask = np.isfinite(synergy) & np.isfinite(residual)
    synergy = synergy[mask]
    residual = residual[mask]
    n = int(len(synergy))
    out: dict[str, Any] = {
        "n": n,
        "spearman_rho": None,
        "spearman_rho_ci": None,
        "pearson_r": None,
    }
    if n < 3 or np.std(synergy) == 0 or np.std(residual) == 0:
        return out

    rho = float(_spearman_rho(synergy, residual))
    r = float(np.corrcoef(synergy, residual)[0, 1])
    rng = np.random.RandomState(random_state)

    def _f(idx):
        s = synergy[idx]
        r_ = residual[idx]
        if np.std(s) == 0 or np.std(r_) == 0:
            return float("nan")
        return float(_spearman_rho(s, r_))

    ci = _bootstrap_ci(_f, n, n_boot=n_boot, rng=rng)

    out.update({
        "spearman_rho": round(rho, 4),
        "spearman_rho_ci": ci,
        "pearson_r": round(r, 4),
    })
    return out


def _decile_lift(
    df: pd.DataFrame,
    score_col: str,
    residual_col: str,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """Bucket df by score_col into ``n_buckets`` quantiles; return mean
    residual per bucket."""
    if df.empty or len(df) < n_buckets:
        return pd.DataFrame(columns=["bucket", "n", "mean_score", "mean_residual"])
    d = df.copy()
    d["bucket"] = pd.qcut(d[score_col], n_buckets, labels=False, duplicates="drop")
    out = (
        d.groupby("bucket")
        .agg(
            n=(score_col, "size"),
            mean_score=(score_col, "mean"),
            mean_residual=(residual_col, "mean"),
        )
        .reset_index()
    )
    out["mean_score"] = out["mean_score"].round(4)
    out["mean_residual"] = out["mean_residual"].round(4)
    return out


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_validation(cfg: ChemNetValidationConfig) -> dict[str, Any]:
    """Execute the full ChemNet validation and return the metrics dict."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = get_connection(read_only=True)

    logger.info(
        "ChemNet validation: train ref %d-%d (assumed), test %d-%d",
        cfg.train_start, cfg.train_end, cfg.test_start, cfg.test_end,
    )

    # ---- 1. Load models ---------------------------------------------------
    t0 = time.monotonic()
    pitcher_norm = None
    if cfg.model_mode == "v2":
        ckpt = cfg.checkpoint_path or (ROOT / "models" / "chemnet_v2.pt")
        gnn, baseline, pitcher_norm = _load_models_v2(ckpt)
        ckpt_label = str(ckpt)
    else:
        gnn, baseline = _load_models(cfg.model_version)
        ckpt_label = f"chemnet_v{cfg.model_version}.pt"
    device = next(gnn.parameters()).device
    load_secs = time.monotonic() - t0
    logger.info(
        "Loaded %s onto %s in %.2fs (mode=%s)",
        ckpt_label, device, load_secs, cfg.model_mode,
    )

    # ---- 2. Cohort + leakage audit ----------------------------------------
    t0 = time.monotonic()
    test_pks = _list_test_game_pks(conn, cfg.test_start, cfg.test_end)
    if cfg.max_test_games > 0 and len(test_pks) > cfg.max_test_games:
        rng = np.random.RandomState(cfg.random_state)
        idx = rng.choice(len(test_pks), size=cfg.max_test_games, replace=False)
        test_pks = [test_pks[i] for i in sorted(idx.tolist())]
    leakage = _leakage_audit(conn, cfg, test_pks)
    logger.info(
        "Cohort: %d test game_pks (leakage overlap with train range = %d)",
        leakage["n_test_game_pks"], leakage["shared_game_pks_train_test"],
    )

    # ---- 3. Inference loop -------------------------------------------------
    t0 = time.monotonic()
    rows: list[dict[str, Any]] = []
    n_skipped = 0
    for i, gp in enumerate(test_pks):
        for side in ("home", "away"):
            if cfg.model_mode == "v2":
                r = _score_one_game_side_v2(
                    conn, gp, side, gnn, baseline, pitcher_norm, device,
                )
            else:
                r = _score_one_game_side(conn, gp, side, gnn, baseline, device)
            if r is None:
                n_skipped += 1
                continue
            rows.append(r)
        if (i + 1) % 200 == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            logger.info(
                "Inferred %d/%d games (%.1f games/s); kept rows=%d skipped=%d",
                i + 1, len(test_pks), rate, len(rows), n_skipped,
            )
    inference_secs = time.monotonic() - t0
    logger.info(
        "Inference complete: %d game-side rows in %.1fs (skipped %d for build_graph failure)",
        len(rows), inference_secs, n_skipped,
    )

    # ---- 4. PA filter ----------------------------------------------------
    rows, dropped_short = _filter_by_pa(conn, rows, cfg.pa_min_per_side)
    logger.info(
        "PA filter: kept %d rows, dropped %d short-PA sides (< %d PAs)",
        len(rows), dropped_short, cfg.pa_min_per_side,
    )

    if not rows:
        logger.error("No qualifying game-sides survived. Aborting.")
        conn.close()
        return {"error": "no_data", "leakage_audit": leakage}

    df = pd.DataFrame(rows)
    df["residual_woba"] = df["actual_total_woba"] - df["baseline_pred"]

    # ---- 5. Metrics --------------------------------------------------------
    t0 = time.monotonic()
    rng = np.random.RandomState(cfg.random_state)

    metrics_gnn = _basic_metrics(
        df["gnn_pred"].to_numpy(),
        df["actual_total_woba"].to_numpy(),
        n_boot=cfg.n_bootstrap_ci,
        random_state=cfg.random_state,
    )
    metrics_baseline = _basic_metrics(
        df["baseline_pred"].to_numpy(),
        df["actual_total_woba"].to_numpy(),
        n_boot=cfg.n_bootstrap_ci,
        random_state=cfg.random_state + 1,
    )

    # Synergy edge over baseline (RMSE-pct improvement)
    rmse_gnn = metrics_gnn["rmse"] or float("nan")
    rmse_base = metrics_baseline["rmse"] or float("nan")
    if rmse_base and rmse_base > 0 and math.isfinite(rmse_gnn) and math.isfinite(rmse_base):
        rmse_pct_improvement = float((rmse_base - rmse_gnn) / rmse_base)
    else:
        rmse_pct_improvement = float("nan")

    synergy_lift = {
        "rmse_gnn": rmse_gnn,
        "rmse_baseline": rmse_base,
        "rmse_abs_improvement": (
            round(rmse_base - rmse_gnn, 4)
            if math.isfinite(rmse_gnn) and math.isfinite(rmse_base)
            else None
        ),
        "rmse_pct_improvement": (
            round(rmse_pct_improvement, 4) if math.isfinite(rmse_pct_improvement) else None
        ),
    }

    # Synergy score predicts residual
    metrics_synergy_residual = _synergy_residual_metrics(
        df["synergy_score"].to_numpy(),
        df["residual_woba"].to_numpy(),
        n_boot=cfg.n_bootstrap_ci,
        random_state=cfg.random_state + 2,
    )

    # Decile lift
    decile_lift = _decile_lift(df, "synergy_score", "residual_woba", n_buckets=10)
    if not decile_lift.empty:
        top = float(decile_lift.iloc[-1]["mean_residual"])
        bot = float(decile_lift.iloc[0]["mean_residual"])
        decile_lift_summary = {
            "top_decile_mean_residual": round(top, 4),
            "bottom_decile_mean_residual": round(bot, 4),
            "lift_top_minus_bottom": round(top - bot, 4),
            "n_top_decile": int(decile_lift.iloc[-1]["n"]),
            "n_bottom_decile": int(decile_lift.iloc[0]["n"]),
        }
    else:
        decile_lift_summary = {
            "top_decile_mean_residual": None,
            "bottom_decile_mean_residual": None,
            "lift_top_minus_bottom": None,
            "n_top_decile": 0,
            "n_bottom_decile": 0,
        }

    metrics_secs = time.monotonic() - t0
    logger.info("Metrics computed in %.1fs", metrics_secs)

    # v2-only: also report metrics on the residual scale (predicting deviation
    # from sum_of_parts directly). Useful for diagnosing whether the residual
    # objective changed the achievable signal vs the absolute-equivalent gates.
    metrics_residual_only: dict[str, Any] = {}
    if cfg.model_mode == "v2" and "gnn_pred_residual" in df.columns:
        actual_res = (df["actual_total_woba"] - df["sum_of_parts"]).to_numpy()
        metrics_residual_only["gnn_residual"] = _basic_metrics(
            df["gnn_pred_residual"].to_numpy(), actual_res,
            n_boot=cfg.n_bootstrap_ci, random_state=cfg.random_state + 10,
        )
        metrics_residual_only["baseline_residual"] = _basic_metrics(
            df["baseline_pred_residual"].to_numpy(), actual_res,
            n_boot=cfg.n_bootstrap_ci, random_state=cfg.random_state + 11,
        )
        metrics_residual_only["sum_of_parts_only"] = _basic_metrics(
            df["sum_of_parts"].to_numpy(),
            df["actual_total_woba"].to_numpy(),
            n_boot=cfg.n_bootstrap_ci, random_state=cfg.random_state + 12,
        )

    # ---- 6. Gate evaluation -----------------------------------------------
    gates = _evaluate_gates(
        metrics_gnn=metrics_gnn,
        metrics_baseline=metrics_baseline,
        synergy_lift=synergy_lift,
        synergy_residual=metrics_synergy_residual,
        leakage=leakage,
    )

    overall_pass = all(g["pass"] for g in gates if g["hard"])

    # ---- 7. Persist artifacts ---------------------------------------------
    per_game_path = output_dir / "chemnet_validation_per_game.csv"
    df_sorted = df.sort_values(["game_pk", "side"]).reset_index(drop=True)
    base_cols = [
        "game_pk", "side", "n_nodes", "n_pa",
        "gnn_pred", "baseline_pred", "actual_total_woba",
        "synergy_score", "residual_woba",
    ]
    v2_cols = ["sum_of_parts", "gnn_pred_residual", "baseline_pred_residual"]
    csv_cols = base_cols + [c for c in v2_cols if c in df_sorted.columns]
    df_sorted[csv_cols].to_csv(per_game_path, index=False)

    decile_path = output_dir / "chemnet_validation_decile_lift.csv"
    if not decile_lift.empty:
        decile_lift.to_csv(decile_path, index=False)

    metrics_path = output_dir / "chemnet_validation_metrics.json"
    payload: dict[str, Any] = {
        "config": {
            "train_start": cfg.train_start,
            "train_end": cfg.train_end,
            "test_start": cfg.test_start,
            "test_end": cfg.test_end,
            "max_test_games": cfg.max_test_games,
            "n_bootstrap_ci": cfg.n_bootstrap_ci,
            "random_state": cfg.random_state,
            "model_version": cfg.model_version,
            "pa_min_per_side": cfg.pa_min_per_side,
        },
        "device": str(device),
        "leakage_audit": leakage,
        "data_quality": {
            "n_test_game_pks": int(leakage["n_test_game_pks"]),
            "n_game_sides_inferred": int(len(rows) + dropped_short),
            "n_game_sides_after_pa_filter": int(len(rows)),
            "dropped_for_short_lineup": int(n_skipped),
            "dropped_for_short_pa": int(dropped_short),
        },
        "metrics": {
            "gnn": metrics_gnn,
            "baseline": metrics_baseline,
            "synergy_lift": synergy_lift,
            "synergy_residual": metrics_synergy_residual,
            "decile_lift": decile_lift_summary,
            **({"residual_only": metrics_residual_only}
               if metrics_residual_only else {}),
        },
        "model_mode": cfg.model_mode,
        "checkpoint_path": str(cfg.checkpoint_path) if cfg.checkpoint_path else None,
        "gates": gates,
        "overall_pass": bool(overall_pass),
        "wall_clock_seconds": {
            "model_load": round(load_secs, 2),
            "inference": round(inference_secs, 2),
            "metrics": round(metrics_secs, 2),
        },
        "artifacts": {
            "per_game_csv": str(per_game_path),
            "decile_lift_csv": str(decile_path) if not decile_lift.empty else None,
            "metrics_json": str(metrics_path),
        },
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
    logger.info("Wrote %s", metrics_path)

    conn.close()

    _print_summary(payload)
    return payload


def _evaluate_gates(
    *,
    metrics_gnn: dict[str, Any],
    metrics_baseline: dict[str, Any],
    synergy_lift: dict[str, Any],
    synergy_residual: dict[str, Any],
    leakage: dict[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate the five hard gates and the one informational gate."""
    gates: list[dict[str, Any]] = []

    # Gate 1 — Pearson r (HEADLINE)
    r = metrics_gnn.get("pearson_r")
    gate1_pass = bool(r is not None and r >= 0.30)
    gates.append({
        "name": "gnn_pearson_r_test",
        "threshold": ">= 0.30",
        "measured": r,
        "measured_ci": metrics_gnn.get("pearson_r_ci"),
        "operator": ">=",
        "pass": gate1_pass,
        "hard": True,
        "source": "metrics.gnn.pearson_r",
    })

    # Gate 2 — GNN beats baseline by >= 2% RMSE
    pct = synergy_lift.get("rmse_pct_improvement")
    gate2_pass = bool(pct is not None and pct >= 0.02)
    gates.append({
        "name": "gnn_beats_baseline_rmse",
        "threshold": ">= 2% RMSE reduction",
        "measured": pct,
        "operator": ">=",
        "pass": gate2_pass,
        "hard": True,
        "source": "metrics.synergy_lift.rmse_pct_improvement",
    })

    # Gate 3 — Synergy predicts residual (rho >= 0.10 with CI lower > 0)
    rho = synergy_residual.get("spearman_rho")
    rho_ci = synergy_residual.get("spearman_rho_ci")
    rho_lower = rho_ci[0] if (rho_ci and len(rho_ci) == 2) else None
    gate3_pass = bool(
        rho is not None and rho >= 0.10
        and rho_lower is not None and rho_lower > 0
    )
    gates.append({
        "name": "synergy_predicts_residual",
        "threshold": ">= 0.10 with CI lower > 0",
        "measured": rho,
        "measured_ci": rho_ci,
        "operator": ">=",
        "pass": gate3_pass,
        "hard": True,
        "source": "metrics.synergy_residual.spearman_rho",
    })

    # Gate 4 — Calibration RMSE <= 4.0
    rmse = metrics_gnn.get("rmse")
    gate4_pass = bool(rmse is not None and rmse <= 4.0)
    gates.append({
        "name": "gnn_rmse_calibration",
        "threshold": "<= 4.0 wOBA-points",
        "measured": rmse,
        "operator": "<=",
        "pass": gate4_pass,
        "hard": True,
        "source": "metrics.gnn.rmse",
    })

    # Gate 5 — Leakage
    gates.append({
        "name": "leakage_audit",
        "threshold": "0 game_pks shared train/test",
        "measured": leakage.get("shared_game_pks_train_test"),
        "operator": "==",
        "pass": bool(leakage.get("pass")),
        "hard": True,
        "source": "leakage_audit.shared_game_pks_train_test",
    })

    return gates


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_summary(payload: dict[str, Any]) -> None:
    m_gnn = payload["metrics"]["gnn"]
    m_base = payload["metrics"]["baseline"]
    syn = payload["metrics"]["synergy_lift"]
    sres = payload["metrics"]["synergy_residual"]
    dec = payload["metrics"]["decile_lift"]
    gates = payload["gates"]

    print("\n" + "=" * 72)
    print("ChemNet Validation -- Summary")
    print("=" * 72)
    print(
        f"GNN:      n={m_gnn['n']:<5d}  r={m_gnn['pearson_r']}  "
        f"rho={m_gnn['spearman_rho']}  RMSE={m_gnn['rmse']}  MAE={m_gnn['mae']}"
    )
    print(
        f"Baseline: n={m_base['n']:<5d}  r={m_base['pearson_r']}  "
        f"rho={m_base['spearman_rho']}  RMSE={m_base['rmse']}  MAE={m_base['mae']}"
    )
    print(
        f"Synergy edge: dRMSE={syn.get('rmse_abs_improvement')}  "
        f"({syn.get('rmse_pct_improvement')} fractional)"
    )
    print(
        f"Synergy -> residual: rho={sres.get('spearman_rho')}  "
        f"CI={sres.get('spearman_rho_ci')}"
    )
    print(
        f"Decile lift (top - bottom mean residual): "
        f"{dec.get('lift_top_minus_bottom')}"
    )
    print("-" * 72)
    for g in gates:
        verdict = "PASS" if g["pass"] else "FAIL"
        print(
            f"  [{verdict}] {g['name']:32s}  thr={g['threshold']:25s}  "
            f"meas={g['measured']}"
        )
    print("-" * 72)
    print(f"OVERALL: {'PASS' if payload['overall_pass'] else 'FAIL'}")
    print("=" * 72 + "\n")


def _json_safe(v: Any) -> Any:
    """Coerce numpy / pandas scalars to JSON-serialisable types."""
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
    p = argparse.ArgumentParser(description=__doc__ or "", formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-start", type=int, default=2015)
    p.add_argument("--train-end", type=int, default=2022)
    p.add_argument("--test-start", type=int, default=2023)
    p.add_argument("--test-end", type=int, default=2024)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument(
        "--max-test-games", type=int, default=0,
        help="Cap on test game_pks (0 = use all qualifying games).",
    )
    p.add_argument("--n-bootstrap-ci", type=int, default=1000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--model-version", type=str, default="1")
    p.add_argument(
        "--pa-min-per-side", type=int, default=25,
        help="Minimum PAs per game-side (filters out shortened/blowout sides).",
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="Optional explicit checkpoint path. If set and the file is a v2 "
             "checkpoint, also pass --model-mode v2.",
    )
    p.add_argument(
        "--model-mode", type=str, default="v1", choices=["v1", "v2"],
        help="Which architecture pipeline to run: 'v1' (9-node lineup, "
             "absolute objective) or 'v2' (10-node lineup+pitcher, residual "
             "objective). Default v1.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = ChemNetValidationConfig(
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        output_dir=args.output_dir,
        max_test_games=args.max_test_games,
        n_bootstrap_ci=args.n_bootstrap_ci,
        random_state=args.random_state,
        model_version=args.model_version,
        pa_min_per_side=args.pa_min_per_side,
        checkpoint_path=args.checkpoint,
        model_mode=args.model_mode,
    )
    try:
        payload = run_validation(cfg)
        if "error" in payload:
            return 2
        return 0 if payload.get("overall_pass") else 1
    except Exception as exc:
        logger.exception("chemnet_validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
