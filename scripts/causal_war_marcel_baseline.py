#!/usr/bin/env python
"""WS4.3 sanity check: Marcel projects 2024 batter wOBA from 2021-2023.

Protocol (declared before results): PA-weighted RMSE of the Marcel wOBA
projection vs the actual 2024 wOBA, compared against the naive constant
forecast (= the 2023 PA-weighted league wOBA -- the most recent completed
season, matching the frozen spec section 5.6 M2 convention).  Weights are
the ACTUAL 2024 PA.  Reported at three PA floors (502 = official
qualification, 300, 100), all published, none a gate -- this is a sanity
check that the implementation behaves like a real projection system
(pre-committed ceiling from plan 4.4: good public systems beat naive by
roughly .02 wOBA), not a validation gate.

Age adjustment uses MLB Stats API birth dates (pre-registered source,
frozen spec section 5.6), parquet-cached at
``data/cache/player_birthdates.parquet``.  Players whose birth date the
API does not return keep multiplier 1.0 and are counted in
``n_age_unknown`` (NULL stays NULL).

Reads the DB strictly read_only; writes only
``results/causal_war/marcel_sanity_2026-08-10/``.

Usage:  python scripts/causal_war_marcel_baseline.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics.marcel import (  # noqa: E402
    fetch_birthdates,
    league_woba_by_season,
    load_batter_season_inputs,
    pa_weighted_rmse,
    project_batters,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("marcel_sanity")

TARGET = 2024
LAGS = (2021, 2022, 2023)
OUTDIR = ROOT / "results" / "causal_war" / "marcel_sanity_2026-08-10"


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection(read_only=True)
    try:
        inputs = load_batter_season_inputs(conn, LAGS)
        actual = load_batter_season_inputs(conn, [TARGET])
    finally:
        conn.close()
    logger.info("Inputs: %d player-season rows (%s); actuals: %d rows (%d)",
                len(inputs), LAGS, len(actual), TARGET)

    pool_ids = sorted(inputs["player_id"].unique().tolist())
    bd = fetch_birthdates(pool_ids)
    n_bd_known = int(bd["birth_date"].notna().sum())
    logger.info("Birth dates: %d/%d known", n_bd_known, len(pool_ids))

    proj = project_batters(inputs, season=TARGET, birthdates=bd)
    logger.info("Projected %d batters for %d", len(proj), TARGET)

    naive_lg_2023 = league_woba_by_season(inputs)[TARGET - 1]

    merged = proj.merge(
        actual.rename(columns={"pa": "actual_pa", "woba": "actual_woba"}),
        on="player_id", how="inner", suffixes=("", "_a"),
    )
    logger.info("Projection-actual overlap: %d batters", len(merged))

    results = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "target_season": TARGET,
        "lag_seasons": list(LAGS),
        "naive_forecast": {
            "definition": "constant = 2023 PA-weighted league wOBA "
                          "(most recent completed season)",
            "value": round(float(naive_lg_2023), 6),
        },
        "woba_source": "pitches SUM(woba_value)/SUM(woba_denom) per "
                       "batter-season (season_batting_stats.woba is 100% "
                       "NULL, verified 2026-08-10); PA from "
                       "season_batting_stats.pa",
        "n_projected": int(len(proj)),
        "n_overlap_with_2024": int(len(merged)),
        "n_age_unknown_in_pool": int(len(pool_ids) - n_bd_known),
        "floors": {},
        "dataset": "in-sample-free: projection inputs are 2021-2023 only; "
                   "2024 outcomes are OOS relative to every Marcel input",
    }

    for floor in (502, 300, 100):
        sub = merged[merged["actual_pa"] >= floor]
        w = sub["actual_pa"].to_numpy()
        a = sub["actual_woba"].to_numpy()
        m = sub["proj_woba"].to_numpy()
        n = np.full(len(sub), naive_lg_2023)
        rm = pa_weighted_rmse(a, m, w)
        rn = pa_weighted_rmse(a, n, w)
        results["floors"][f"pa_ge_{floor}"] = {
            "n_batters": int(len(sub)),
            "rmse_marcel": round(float(rm), 6),
            "rmse_naive_league": round(float(rn), 6),
            "marcel_minus_naive": round(float(rm - rn), 6),
            "mean_abs_err_marcel": round(float(
                np.average(np.abs(a - m), weights=w)), 6),
            "mean_signed_err_marcel": round(float(
                np.average(m - a, weights=w)), 6),
        }
        logger.info(
            "PA>=%d: n=%d  RMSE marcel=%.4f  naive=%.4f  delta=%+.4f",
            floor, len(sub), rm, rn, rm - rn,
        )

    proj.to_csv(OUTDIR / "marcel_2024_projections.csv", index=False)
    (OUTDIR / "marcel_2024_sanity.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8",
    )
    logger.info("Wrote %s", OUTDIR / "marcel_2024_sanity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
