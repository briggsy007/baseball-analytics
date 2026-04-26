"""Freeze the A3 matchup-pair eval cohort for reproducibility.

Per ``docs/pitchgpt_sim_engine/EXECUTION_PLAN.md`` §6 A3 Gates 2/3:
    Gate 2: observed-vs-simulated wOBA on 2025 matchup pairs with
            >=50 real PAs - calibration scatter plot, Pearson r >= 0.4.
    Gate 3: any pair with fewer than 20 real matchup PAs is marked
            "rollout only, no empirical cross-check" in the UI.

This script:
- Pulls all 2025 (batter, pitcher) pairs from the ``pitches`` table.
- Splits into three bands: ge50, 20-49, <20.
- Writes the row-key tuples to
  ``results/a3_matchup_cohort_2026_04_26/{ge50.parquet, band20_49.parquet,
  lt20.parquet, manifest.json}``.

SCHEMA-MISMATCH NOTE (surfaced in manifest):
  EXECUTION_PLAN A3 Gate 2 specifies ">=50 real matchup PAs" but in 2025
  alone the maximum batter x pitcher pair has only ~13 PAs - the ge50
  band is effectively empty in single-season data. The manifest captures
  this so the future Phase 1 grader knows to either:
    (a) widen to multi-season (e.g., 2022-2025) for the Gate 2 backtest, or
    (b) lower the Gate 2 floor with explicit user sign-off.

  We still emit ``ge50.parquet`` (likely empty) so the pipeline shape is
  stable; downstream code must handle the empty case.

Read-only DuckDB. No rollouts here - this just freezes the cohort.

Usage:
    python scripts/a3_matchup_cohort.py
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.db.schema import get_connection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEASON_START = "2025-03-28"
SEASON_END = "2025-09-29"

GE50_FLOOR = 50
BAND_LO = 20
BAND_HI = 49
LT_FLOOR = 20

SEED = 20260426

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "a3_matchup_cohort_2026_04_26"


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

PAIR_PA_COUNTS_SQL = """
WITH pa_counts AS (
    SELECT pitcher_id, batter_id,
           COUNT(DISTINCT game_pk * 1000 + at_bat_number) AS pa_count
    FROM pitches
    WHERE game_date BETWEEN ? AND ?
      AND pitcher_id IS NOT NULL
      AND batter_id IS NOT NULL
    GROUP BY pitcher_id, batter_id
)
SELECT pitcher_id, batter_id, pa_count
FROM pa_counts
ORDER BY pa_count DESC, pitcher_id, batter_id
"""


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def fetch_pair_pa_counts(conn) -> pd.DataFrame:
    """Return DataFrame [pitcher_id, batter_id, pa_count] for 2025 pairs."""
    rows = conn.execute(PAIR_PA_COUNTS_SQL, [SEASON_START, SEASON_END]).fetchall()
    return pd.DataFrame(rows, columns=["pitcher_id", "batter_id", "pa_count"])


def split_into_bands(pairs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split pair table into ge50 / band 20-49 / lt20 bands per A3 Gate 3."""
    ge50 = pairs[pairs["pa_count"] >= GE50_FLOOR].copy()
    band_20_49 = pairs[
        (pairs["pa_count"] >= BAND_LO) & (pairs["pa_count"] <= BAND_HI)
    ].copy()
    lt20 = pairs[pairs["pa_count"] < LT_FLOOR].copy()
    return {"ge50": ge50, "band20_49": band_20_49, "lt20": lt20}


def _band_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "n_pairs": 0,
            "n_unique_pitchers": 0,
            "n_unique_batters": 0,
            "pa_count_min": None,
            "pa_count_max": None,
            "pa_count_total": 0,
        }
    return {
        "n_pairs": int(len(df)),
        "n_unique_pitchers": int(df["pitcher_id"].nunique()),
        "n_unique_batters": int(df["batter_id"].nunique()),
        "pa_count_min": int(df["pa_count"].min()),
        "pa_count_max": int(df["pa_count"].max()),
        "pa_count_total": int(df["pa_count"].sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Output directory (default: %(default)s)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = get_connection(read_only=True)

    print(f"[a3_matchup_cohort] Pulling 2025 (batter, pitcher) pair PA counts...")
    pairs = fetch_pair_pa_counts(conn)
    print(f"  -> {len(pairs)} unique pairs (2025 season)")

    if pairs.empty:
        print("[a3_matchup_cohort] FAIL: no pairs found. Aborting.")
        return 1

    bands = split_into_bands(pairs)
    band_paths = {}
    for name, df in bands.items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        band_paths[name] = str(path.relative_to(PROJECT_ROOT))
        print(f"  -> {name}: {len(df)} pairs ({path.name})")

    manifest = {
        "schema_version": 1,
        "cohort_name": "a3_matchup_cohort_2026_04_26",
        "purpose": (
            "Matchup sim with CIs (A3) - eval cohort. Bands derive from "
            "EXECUTION_PLAN A3 Gate 2 (>=50 real PAs for empirical cross-check) "
            "and Gate 3 (<20 real PAs flagged 'rollout only, no empirical "
            "cross-check')."
        ),
        "frozen_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "seed": SEED,
        "season_window": {"start": SEASON_START, "end": SEASON_END},
        "thresholds": {
            "ge50_floor": GE50_FLOOR,
            "band_20_49_lo": BAND_LO,
            "band_20_49_hi": BAND_HI,
            "lt20_floor": LT_FLOOR,
        },
        "bands": {
            "ge50": {
                "parquet": band_paths["ge50"],
                "purpose": "Gate 2 empirical cross-check cohort (>=50 real PAs).",
                **_band_summary(bands["ge50"]),
            },
            "band20_49": {
                "parquet": band_paths["band20_49"],
                "purpose": "Gate 3 caveat band (20-49 real PAs).",
                **_band_summary(bands["band20_49"]),
            },
            "lt20": {
                "parquet": band_paths["lt20"],
                "purpose": "Gate 3 rollout-only flag (<20 real PAs).",
                **_band_summary(bands["lt20"]),
            },
        },
        "schema_mismatch_note": {
            "issue": (
                "EXECUTION_PLAN A3 Gate 2 specifies '>=50 real matchup PAs' "
                "for empirical cross-check, but in 2025 alone the maximum "
                "batter x pitcher pair has only ~13 PAs."
            ),
            "implication": (
                "The ge50 band is empty in single-season data. Phase 1 "
                "implementer must either widen to multi-season (e.g., "
                "2022-2025 stacked) for the Gate 2 backtest, or obtain "
                "explicit user sign-off to lower the floor (e.g., to >=10)."
            ),
            "recommended_followup": (
                "Add scripts/a3_matchup_cohort_multiseason.py that pulls "
                "2022-2025 union with the same band logic; expect ge50 "
                "to populate at multi-season scale."
            ),
        },
        "downstream_consumers": [
            "src/dashboard/views/matchup_sim.py (A3 view)",
            "src/analytics/pitchgpt_matchup.py (future Phase 1 module)",
        ],
        "notes": (
            "Read-only DuckDB pull. Row-key per pair is (pitcher_id, "
            "batter_id, pa_count); the parquet files do NOT include the "
            "underlying pitch-level rows - those can be re-fetched via "
            "pitches WHERE pitcher_id IN (...) AND batter_id IN (...) AND "
            "game_date BETWEEN ... AND ...."
        ),
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[a3_matchup_cohort] Wrote {manifest_path}")
    print(
        f"[a3_matchup_cohort] band counts: ge50={len(bands['ge50'])}, "
        f"20-49={len(bands['band20_49'])}, <20={len(bands['lt20'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
