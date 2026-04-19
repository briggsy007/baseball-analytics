"""
Build next-season WAR projections from historical seasonal stats.

Usage:
    python scripts/build_projections.py \
        --train-seasons 2021 2022 2023 \
        --target-season 2024 \
        --output-path results/projections/

Emits:
    {output-path}/projections_{target}.csv
    {output-path}/model_metadata.json
    {output-path}/projection_model.pkl  (the fitted model)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make src importable when run as a script
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.analytics.projections import (  # noqa: E402
    ProjectionModel,
    ProjectionConfig,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("build_projections")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-seasons",
        nargs="+",
        type=int,
        required=True,
        help="Seasons to use as input history (e.g. 2021 2022 2023).",
    )
    parser.add_argument(
        "--target-season",
        type=int,
        required=True,
        help="Season to project (must be > all train seasons).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Directory to write projections CSV + metadata JSON.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help="Optional override for the DuckDB path.",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Enable v2 features: TJ flag + calibrated age curve + role change.",
    )
    args = parser.parse_args()

    output_dir = args.output_path
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = str(args.db_path) if args.db_path else None
    conn = get_connection(db_path, read_only=True)

    t0 = time.time()
    if args.v2:
        # Age-curve calibration uses every season strictly < target_season.
        # 2015-2023 is the default window for target=2024.
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

    logger.info("Fitting on seasons: %s", args.train_seasons)
    fit_metrics = model.fit(conn, list(args.train_seasons))
    logger.info("Fit complete: %s", fit_metrics)

    logger.info("Projecting season %d", args.target_season)
    proj = model.project(conn, target_season=args.target_season)
    logger.info("Projected %d player-seasons", len(proj))

    csv_path = output_dir / f"projections_{args.target_season}.csv"
    proj.to_csv(csv_path, index=False)
    logger.info("Projections written to %s", csv_path)

    model_pkl = output_dir / "projection_model.pkl"
    model.save(model_pkl)
    logger.info("Model saved to %s", model_pkl)

    elapsed = time.time() - t0
    metadata = {
        "model_name": model.model_name,
        "version": model.version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_seasons": list(args.train_seasons),
        "target_season": args.target_season,
        "fit_metrics": fit_metrics,
        "n_projections": int(len(proj)),
        "n_batters": int((proj["position"] == "batter").sum()),
        "n_pitchers": int((proj["position"] == "pitcher").sum()),
        "wall_clock_seconds": round(elapsed, 2),
        "config": {
            "weights": list(config.weights),
            "batter_regression_pa": config.batter_regression_pa,
            "pitcher_regression_ip": config.pitcher_regression_ip,
            "overlay_cap_war": config.overlay_cap_war,
            "batter_xwoba_threshold": config.batter_xwoba_threshold,
            "pitcher_xfip_threshold": config.pitcher_xfip_threshold,
        },
    }
    meta_path = output_dir / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    logger.info("Metadata written to %s", meta_path)

    logger.info("Done in %.1fs", elapsed)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
