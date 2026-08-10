#!/usr/bin/env python
"""WS4.1 (2026-08-10): CausalWAR v3 covariates -- retrain + honest gate re-run.

What this does
--------------
1. **Reproduces the v1 and v2 correlation gates predict-only** from the
   existing frozen checkpoints (`causal_war_trainsplit_2015_2022.pkl`,
   `..._v2_umpweather.pkl`) on today's DB, through the same
   merge/metrics path as the documented April runs (floors pa>=300/ip>=50,
   1000-rep CI bootstrap, seed 42).  This validates the scoring pipeline
   (and the sklearn-unpickle path) against the documented numbers
   (v1 r=0.7089 / rho=0.6314; v2 r=0.6995 / rho=0.6165; n=968) before any
   new number is produced.
2. **Retrains the nuisance with the v3 covariates** (WS4.1): season-lagged
   pitcher wOBA-against (`opp_woba_lag`) + park via `pitches.home_team`
   (`park_code`, fixed 30-team ordinal map) on top of the v2 layout.
   Train 2015-2022, test 2023-2024, `n_bootstrap=50` (mirrors the
   historical `baseline_comparison` config; the bootstrap affects only
   train-effect CIs, never the gates).  Artifact goes to the NEW path
   `models/causal_war/causal_war_trainsplit_2015_2022_v3_oppq_park.pkl`
   (the module's write-once guard diverts on collision; no existing
   checkpoint is ever overwritten).
3. **Re-runs the correlation gates for v3** on the identical basis and
   REPORTS WHERE THEY LAND.  Pre-commitment (written before results
   exist, per the plan's WS4.1): the audit expects a DROP -- the reverted
   2026-04-18 venue fix cost combined Spearman 0.63->0.49.  Whatever the
   numbers are, they are published unrounded and unsoftened.  Gate-metric
   optimization over correctness is the audit finding, not a practice to
   repeat.  Claims-registry updates happen in Batch D, not here.

Reads the DB strictly read_only.  Writes:
  * results/causal_war/v3_gates_2026-08-10/*.json/csv (new files)
  * models/causal_war/causal_war_trainsplit_2015_2022_v3_oppq_park.pkl (new)

Usage:  python scripts/causal_war_v3_retrain.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics.causal_war import (  # noqa: E402
    CausalWARConfig,
    CausalWARModel,
    _aggregate_player_effects,
    _build_features,
    _effects_to_df,
    _extract_pa_data,
)
from scripts.baseline_comparison import (  # noqa: E402
    _fetch_batter_traditional_war,
    _fetch_pitcher_traditional_war,
    _known_pitcher_ids,
    compute_metrics,
    merge_with_traditional,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("v3_retrain")

TRAIN = (2015, 2022)
TEST = (2023, 2024)
PA_MIN, IP_MIN = 300, 50          # the documented gate basis (171420Z/194415Z)
N_BOOT_CI = 1000                  # correlation-CI bootstrap, seed 42
N_BOOT_TRAIN = 50                 # mirrors historical baseline_comparison runs
OUTDIR = ROOT / "results" / "causal_war" / "v3_gates_2026-08-10"

V1_PKL = ROOT / "models" / "causal_war" / "causal_war_trainsplit_2015_2022.pkl"
V2_PKL = ROOT / "models" / "causal_war" / "causal_war_trainsplit_2015_2022_v2_umpweather.pkl"

# Documented April 2026 numbers (causal_war_baseline_results.md) the
# predict-only reproduction is checked against.
DOCUMENTED = {
    "v1": {"pearson_r": 0.7089, "spearman_rho": 0.6314, "n": 968},
    "v2": {"pearson_r": 0.6995, "spearman_rho": 0.6165, "n": 968},
}

_V2_ONLY_COLS = [
    "ump_accuracy_above_x", "ump_favor", "temp_f", "wind_speed_mph",
    "wind_dir_deg", "humidity_pct", "pressure_mb", "roof_status",
]


def _score_checkpoint(
    pkl_path: Path,
    df_test: pd.DataFrame,
    *,
    drop_cols: list[str],
    expect_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Predict-only test-window effects for a frozen checkpoint."""
    art = joblib.load(pkl_path)
    nuisance = art["nuisance_outcome"]
    df = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns]).copy()
    W, Y, batter_ids, dfb = _build_features(df)
    if W.shape[1] != expect_features:
        raise RuntimeError(
            f"{pkl_path.name}: expected {expect_features} confounders, built "
            f"{W.shape[1]} -- feature layout drifted; refusing to score."
        )
    if getattr(nuisance, "n_features_in_", None) != expect_features:
        raise RuntimeError(
            f"{pkl_path.name}: checkpoint n_features_in_="
            f"{getattr(nuisance, 'n_features_in_', None)} != {expect_features}"
        )
    Y_pred = nuisance.predict(W)
    Y_res = Y - Y_pred
    from sklearn.metrics import r2_score
    test_r2 = float(r2_score(Y, Y_pred))
    bat = _effects_to_df(
        _aggregate_player_effects(Y_res, batter_ids, dfb, pa_min=10),
        season_label=f"{TEST[0]}-{TEST[1]}", pa_min_qualifying=50,
    )
    pit_ids = dfb["pitcher_id"].to_numpy().astype(int)
    pit = _effects_to_df(
        _aggregate_player_effects(-Y_res, pit_ids, dfb, pa_min=50),
        season_label=f"{TEST[0]}-{TEST[1]}", pa_min_qualifying=50,
    )
    return bat, pit, {"test_nuisance_r2": round(test_r2, 4), "n_test_pa": int(len(Y))}


def _gates(
    bat: pd.DataFrame,
    pit: pd.DataFrame,
    trad_b: pd.DataFrame,
    trad_p: pd.DataFrame,
    pitcher_ids: set[int],
) -> dict[str, Any]:
    merged = merge_with_traditional(
        bat, pit, trad_b, trad_p, pa_min=PA_MIN, ip_min=IP_MIN,
        pitcher_ids=pitcher_ids,
    )
    out: dict[str, Any] = {"n_merged": int(len(merged))}
    for label, subset in (
        ("combined", merged),
        ("batters", merged[merged["position"] == "batter"]),
        ("pitchers", merged[merged["position"] == "pitcher"]),
    ):
        m = compute_metrics(
            subset["causal_war"].to_numpy(), subset["trad_war"].to_numpy(),
            n_bootstrap=N_BOOT_CI, random_state=42,
        )
        out[label] = m
    rc = out["combined"]["pearson_r"] or 0.0
    sc = out["combined"]["spearman_rho"] or 0.0
    out["spec_gate"] = {
        "pearson_r_required": 0.5,
        "spearman_rho_required": 0.6,
        "pearson_pass": bool(rc >= 0.5),
        "spearman_pass": bool(sc >= 0.6),
        "pass": bool(rc >= 0.5 and sc >= 0.6),
    }
    return out, merged


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection(read_only=True)
    results: dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "basis": {
            "train": TRAIN, "test": TEST, "pa_min": PA_MIN, "ip_min": IP_MIN,
            "n_bootstrap_ci": N_BOOT_CI, "random_state": 42,
            "trad_war_source": "season_batting_stats.war / season_pitching_stats.war (real bWAR)",
        },
        "documented_april_numbers": DOCUMENTED,
    }
    try:
        trad_b = _fetch_batter_traditional_war(conn, *TEST)
        trad_p = _fetch_pitcher_traditional_war(conn, *TEST)
        pitcher_ids = _known_pitcher_ids(conn, *TEST)
        results["basis"]["batter_war_source"] = (
            trad_b["war_source"].iloc[0] if len(trad_b) else None
        )
        results["basis"]["pitcher_war_source"] = (
            trad_p["war_source"].iloc[0] if len(trad_p) else None
        )

        # ---- Step 1+2: predict-only v1 / v2 reproduction -----------------
        logger.info("Extracting test PAs %s (default layout) ...", TEST)
        df_test = _extract_pa_data(conn, year_range=TEST)
        logger.info("  %d test PAs", len(df_test))

        for tag, pkl, drop, nfeat in (
            ("v1", V1_PKL, _V2_ONLY_COLS, 12),
            ("v2", V2_PKL, ["roof_status"], 19),
        ):
            logger.info("Scoring %s checkpoint predict-only ...", tag)
            bat, pit, info = _score_checkpoint(
                pkl, df_test, drop_cols=drop if tag == "v1" else [],
                expect_features=nfeat,
            )
            g, merged = _gates(bat, pit, trad_b, trad_p, pitcher_ids)
            g.update(info)
            doc = DOCUMENTED[tag]
            g["reproduction_delta"] = {
                "pearson_r": round((g["combined"]["pearson_r"] or 0) - doc["pearson_r"], 4),
                "spearman_rho": round((g["combined"]["spearman_rho"] or 0) - doc["spearman_rho"], 4),
            }
            results[tag] = g
            merged.to_csv(OUTDIR / f"gate_pairs_{tag}.csv", index=False)
            logger.info(
                "  %s: n=%d r=%.4f rho=%.4f (documented r=%.4f rho=%.4f)",
                tag, g["combined"]["n"], g["combined"]["pearson_r"],
                g["combined"]["spearman_rho"], doc["pearson_r"], doc["spearman_rho"],
            )

        # ---- Step 3: v3 retrain ------------------------------------------
        logger.info("Retraining v3 (opp_woba_lag + park_code) ...")
        cfg = CausalWARConfig(
            n_bootstrap=N_BOOT_TRAIN,
            train_start_year=TRAIN[0], train_end_year=TRAIN[1],
            test_start_year=TEST[0], test_end_year=TEST[1],
            model_version="v3_oppq_park",
            include_opp_quality_park=True,
        )
        model = CausalWARModel(config=cfg)
        split = model.train_test_split(conn, train_split=TRAIN, test_split=TEST)
        g3, merged3 = _gates(
            split["test_player_effects"], split["test_pitcher_effects"],
            trad_b, trad_p, pitcher_ids,
        )
        g3["train_metrics"] = split["train_metrics"]
        g3["test_metrics"] = split["test_metrics"]
        g3["artifact_path"] = split["artifact_path"]
        results["v3"] = g3
        merged3.to_csv(OUTDIR / "gate_pairs_v3.csv", index=False)
        logger.info(
            "  v3: n=%d r=%s rho=%s (train_r2=%s test_r2=%s)",
            g3["combined"]["n"], g3["combined"]["pearson_r"],
            g3["combined"]["spearman_rho"],
            split["train_metrics"].get("outcome_nuisance_r2"),
            split["test_metrics"].get("test_nuisance_r2"),
        )
    finally:
        conn.close()

    (OUTDIR / "v3_gates_metrics.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8",
    )
    logger.info("Wrote %s", OUTDIR / "v3_gates_metrics.json")

    print("\n" + "=" * 78)
    print("WS4.1 gate re-run -- numbers exactly as they land")
    print("=" * 78)
    for tag in ("v1", "v2", "v3"):
        c = results[tag]["combined"]
        gate = results[tag]["spec_gate"]
        print(
            f"{tag}: n={c['n']} r={c['pearson_r']} {c['pearson_r_ci']} "
            f"rho={c['spearman_rho']} {c['spearman_rho_ci']} "
            f"-> spec gate (r>=0.5 & rho>=0.6): {'PASS' if gate['pass'] else 'FAIL'}"
        )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
