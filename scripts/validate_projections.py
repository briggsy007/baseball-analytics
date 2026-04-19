"""
Backtest the projection model: project 2024 from 2021-2023, compare to actuals.

Writes:
    {run_dir}/projections_2024.csv
    {run_dir}/backtest_metrics.json
    {run_dir}/top_movers.csv
    {run_dir}/validation_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.analytics.projections import (  # noqa: E402
    ProjectionModel,
    ProjectionConfig,
    _load_actual_war,
    _score_predictions,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("validate_projections")


GATES = {
    "rmse_combined": {"threshold": 1.5, "operator": "<="},
    "pearson_r_combined": {"threshold": 0.55, "operator": ">="},
    "spearman_rho_combined": {"threshold": 0.50, "operator": ">="},
    "rmse_delta_vs_marcel": {"threshold": 0.0, "operator": "<="},
    "leakage_check": {"threshold": "all priors < target_season", "operator": "disjoint"},
}


def _evaluate_gate(name: str, value: float, gate: dict) -> bool:
    op = gate["operator"]
    thr = gate["threshold"]
    if op == ">=":
        return value >= thr
    if op == "<=":
        return value <= thr
    if op == "disjoint":
        return bool(value)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-seasons", nargs="+", type=int, default=[2021, 2022, 2023])
    parser.add_argument("--target-season", type=int, default=2024)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--db-path", type=Path, default=None)
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Enable v2 features: TJ flag + calibrated age curve + role change.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    db_path = str(args.db_path) if args.db_path else None
    conn = get_connection(db_path, read_only=True)

    t0 = time.time()

    # ---- Leakage check ------------------------------------------------------
    leakage_pass = all(s < args.target_season for s in args.train_seasons)
    logger.info(
        "Leakage check: priors=%s vs target=%d -> pass=%s",
        args.train_seasons, args.target_season, leakage_pass,
    )

    # ---- Fit & project ------------------------------------------------------
    if args.v2:
        config = ProjectionConfig(
            enable_tj_flag=True,
            enable_calibrated_age_curve=True,
            enable_role_change=True,
            age_curve_train_year_lo=2015,
            age_curve_train_year_hi=int(args.target_season) - 1,
        )
    else:
        config = ProjectionConfig()
    model = ProjectionModel(config=config)
    fit_metrics = model.fit(conn, list(args.train_seasons))
    proj = model.project(conn, target_season=args.target_season)
    logger.info("Projected %d players", len(proj))

    proj_path = run_dir / f"projections_{args.target_season}.csv"
    proj.to_csv(proj_path, index=False)

    # ---- Load actuals & score ----------------------------------------------
    actual = _load_actual_war(conn, args.target_season)
    merged = proj.merge(actual, on=["player_id", "position"], how="inner")
    logger.info("Merged %d projection-actual pairs", len(merged))
    merged.to_csv(run_dir / "backtest_pairs.csv", index=False)

    metrics = _score_predictions(merged)
    metrics["leakage_check"] = leakage_pass
    metrics["target_season"] = args.target_season
    metrics["train_seasons"] = list(args.train_seasons)
    (run_dir / "backtest_metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str),
    )

    # ---- Top movers (largest |statcast_adjustment|) ------------------------
    movers = proj.copy()
    movers["abs_adj"] = movers["statcast_adjustment"].abs()
    movers = movers.sort_values("abs_adj", ascending=False)
    movers.head(40).to_csv(run_dir / "top_movers.csv", index=False)

    # ---- Build validation_summary.json (skill schema) -----------------------
    combined = metrics.get("combined", {})
    rmse = combined.get("rmse")
    pearson_r = combined.get("pearson_r")
    spearman = combined.get("spearman_rho")
    rmse_delta = combined.get("rmse_delta_vs_marcel")

    gate_results = []
    gate_results.append({
        "name": "leakage_check",
        "threshold": GATES["leakage_check"]["threshold"],
        "measured": (
            f"train={args.train_seasons}, target={args.target_season}; "
            f"all priors strictly < target = {leakage_pass}"
        ),
        "operator": GATES["leakage_check"]["operator"],
        "pass": leakage_pass,
        "source": "scripts/validate_projections.py invocation",
    })
    gate_results.append({
        "name": "rmse_combined",
        "threshold": GATES["rmse_combined"]["threshold"],
        "measured": rmse,
        "operator": GATES["rmse_combined"]["operator"],
        "pass": _evaluate_gate("rmse_combined", rmse, GATES["rmse_combined"]) if rmse is not None else False,
        "source": "backtest_metrics.json (combined.rmse)",
    })
    gate_results.append({
        "name": "pearson_r_combined",
        "threshold": GATES["pearson_r_combined"]["threshold"],
        "measured": pearson_r,
        "operator": GATES["pearson_r_combined"]["operator"],
        "pass": _evaluate_gate("pearson_r_combined", pearson_r, GATES["pearson_r_combined"]) if pearson_r is not None else False,
        "source": "backtest_metrics.json (combined.pearson_r)",
    })
    gate_results.append({
        "name": "spearman_rho_combined",
        "threshold": GATES["spearman_rho_combined"]["threshold"],
        "measured": spearman,
        "operator": GATES["spearman_rho_combined"]["operator"],
        "pass": _evaluate_gate("spearman_rho_combined", spearman, GATES["spearman_rho_combined"]) if spearman is not None else False,
        "source": "backtest_metrics.json (combined.spearman_rho)",
    })
    gate_results.append({
        "name": "rmse_delta_vs_marcel",
        "threshold": GATES["rmse_delta_vs_marcel"]["threshold"],
        "measured": rmse_delta,
        "operator": GATES["rmse_delta_vs_marcel"]["operator"],
        "pass": _evaluate_gate("rmse_delta_vs_marcel", rmse_delta, GATES["rmse_delta_vs_marcel"]) if rmse_delta is not None else False,
        "source": "backtest_metrics.json (combined.rmse_delta_vs_marcel)",
    })

    overall_pass = all(g["pass"] for g in gate_results)
    failed = [g["name"] for g in gate_results if not g["pass"]]

    summary = {
        "model": "projections",
        "invocation": "/validate-model projections (initial release backtest)",
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spec_path": "docs/models/projections_validation_spec.md",
        "gates": gate_results,
        "overall_pass": overall_pass,
        "failed_gates": failed,
        "artifacts": [
            f"{run_dir.as_posix()}/projections_{args.target_season}.csv",
            f"{run_dir.as_posix()}/backtest_metrics.json",
            f"{run_dir.as_posix()}/backtest_pairs.csv",
            f"{run_dir.as_posix()}/top_movers.csv",
            f"{run_dir.as_posix()}/validation_summary.json",
        ],
        "wall_clock_seconds": {"total": round(time.time() - t0, 2)},
        "cohort_metrics": {
            "batters": metrics.get("batters", {}),
            "pitchers": metrics.get("pitchers", {}),
            "combined": metrics.get("combined", {}),
            "calibration": metrics.get("calibration_3war_to_2p5_actual", {}),
        },
        "fit_metrics": fit_metrics,
    }
    (run_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
    )
    logger.info("Validation summary written. overall_pass=%s", overall_pass)
    logger.info("Failed gates: %s", failed if failed else "none")

    conn.close()
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
