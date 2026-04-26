"""Freeze the A1 evaluation cohort (2025 H1 vs H2) for reproducibility.

Per ``docs/pitchgpt_sim_engine/EXECUTION_PLAN.md`` §6 A1 Gate 3:
    "split 2025 into first-half (grade computation) and second-half
    (calibration check). Grades on first-half must predict second-half
    pitcher-level xwOBA allowed with rank correlation >= 0.3."

This script:
- Pulls candidate plate appearances (PAs) from the ``pitches`` table.
- Restricts to pitchers with >=200 PAs in 2025 H1 AND >=100 PAs in 2025 H2
  (cohort-stability floor; tighter than the EXECUTION_PLAN's per-half
  threshold so grades are only computed for pitchers we can also
  back-validate).
- Writes the row-ID lists to
  ``results/a1_eval_cohort_2026_04_26/{h1.parquet, h2.parquet, manifest.json}``.

Read-only DuckDB. Does NOT compute grades — that's Phase 1 work. Freezes
the cohort so future eval runs are reproducible.

Usage:
    python scripts/a1_eval_cohort.py

The output directory is created if it does not exist. Re-running overwrites
``h1.parquet``, ``h2.parquet``, and ``manifest.json``.
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

# 2025 season-half bounds (per EXECUTION_PLAN A1 Gate 3).
# Approximate All-Star break = 2025-07-15; we use end-of-June as a clean
# split that is independent of week-of-year.
H1_START = "2025-03-28"   # Approximate 2025 Opening Day
H1_END = "2025-06-30"
H2_START = "2025-07-01"
H2_END = "2025-09-29"     # Approximate 2025 regular-season close

# Pitcher floor: 200 PAs in H1 AND 100 PAs in H2.
H1_PA_FLOOR = 200
H2_PA_FLOOR = 100

# Reproducibility seed (no random sampling here, but recorded for downstream
# stages that may sub-sample within the cohort).
SEED = 20260426

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "results" / "a1_eval_cohort_2026_04_26"


# ---------------------------------------------------------------------------
# SQL
# ---------------------------------------------------------------------------

# Returns the set of qualifying pitcher_ids for the H1+H2 stability cohort.
# A "PA" is a unique (game_pk, at_bat_number) tuple within the half-window.
QUALIFYING_PITCHERS_SQL = """
WITH h1_pas AS (
    SELECT pitcher_id,
           COUNT(DISTINCT game_pk * 1000 + at_bat_number) AS pa_count
    FROM pitches
    WHERE game_date BETWEEN ? AND ?
      AND pitcher_id IS NOT NULL
    GROUP BY pitcher_id
),
h2_pas AS (
    SELECT pitcher_id,
           COUNT(DISTINCT game_pk * 1000 + at_bat_number) AS pa_count
    FROM pitches
    WHERE game_date BETWEEN ? AND ?
      AND pitcher_id IS NOT NULL
    GROUP BY pitcher_id
)
SELECT h1.pitcher_id, h1.pa_count AS h1_pa, h2.pa_count AS h2_pa
FROM h1_pas h1
JOIN h2_pas h2 USING (pitcher_id)
WHERE h1.pa_count >= ? AND h2.pa_count >= ?
ORDER BY h1.pa_count DESC
"""

# Returns ALL pitches in the half-window for the qualifying pitchers.
# We freeze game_pk + at_bat_number + pitch_number as the row-key tuple.
HALF_WINDOW_PITCHES_SQL = """
SELECT
    game_pk,
    game_date,
    at_bat_number,
    pitch_number,
    pitcher_id,
    batter_id,
    balls,
    strikes,
    outs_when_up,
    inning,
    inning_topbot,
    on_1b, on_2b, on_3b,
    stand,
    p_throws,
    pitch_type,
    zone,
    description,
    events,
    woba_value,
    woba_denom,
    delta_run_exp
FROM pitches
WHERE game_date BETWEEN ? AND ?
  AND pitcher_id IN (
      SELECT pitcher_id FROM ({qualifying_sql_inline}) qp
  )
ORDER BY game_pk, at_bat_number, pitch_number
"""


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def fetch_qualifying_pitchers(conn) -> pd.DataFrame:
    """Return DataFrame of qualifying (pitcher_id, h1_pa, h2_pa)."""
    rows = conn.execute(
        QUALIFYING_PITCHERS_SQL,
        [H1_START, H1_END, H2_START, H2_END, H1_PA_FLOOR, H2_PA_FLOOR],
    ).fetchall()
    return pd.DataFrame(rows, columns=["pitcher_id", "h1_pa", "h2_pa"])


def fetch_half_window_pitches(
    conn, start: str, end: str, qualifying_ids: list[int]
) -> pd.DataFrame:
    """Return DataFrame of pitches in [start, end] for the qualifying pitchers."""
    if not qualifying_ids:
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(qualifying_ids))
    sql = f"""
    SELECT
        game_pk,
        game_date,
        at_bat_number,
        pitch_number,
        pitcher_id,
        batter_id,
        balls,
        strikes,
        outs_when_up,
        inning,
        inning_topbot,
        on_1b, on_2b, on_3b,
        stand,
        p_throws,
        pitch_type,
        zone,
        description,
        events,
        woba_value,
        woba_denom,
        delta_run_exp
    FROM pitches
    WHERE game_date BETWEEN ? AND ?
      AND pitcher_id IN ({placeholders})
    ORDER BY game_pk, at_bat_number, pitch_number
    """
    params = [start, end, *qualifying_ids]
    return conn.execute(sql, params).fetch_df()


def _fingerprint_dataframe(df: pd.DataFrame) -> dict:
    """Return summary stats for the manifest (no PII / no full row dump)."""
    if df.empty:
        return {"n_rows": 0, "n_pitchers": 0, "n_pas": 0, "n_games": 0}
    return {
        "n_rows": int(len(df)),
        "n_pitchers": int(df["pitcher_id"].nunique()),
        "n_pas": int(df.groupby(["game_pk", "at_bat_number"]).ngroups),
        "n_games": int(df["game_pk"].nunique()),
        "date_min": str(df["game_date"].min()),
        "date_max": str(df["game_date"].max()),
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

    print(f"[a1_eval_cohort] Pulling qualifying pitcher cohort...")
    qualifying = fetch_qualifying_pitchers(conn)
    print(f"  -> {len(qualifying)} qualifying pitchers")

    if qualifying.empty:
        print("[a1_eval_cohort] FAIL: no qualifying pitchers. Aborting.")
        return 1

    pitcher_ids = qualifying["pitcher_id"].astype(int).tolist()

    print(f"[a1_eval_cohort] Fetching H1 ({H1_START} -> {H1_END}) pitches...")
    h1 = fetch_half_window_pitches(conn, H1_START, H1_END, pitcher_ids)
    print(f"  -> {len(h1)} pitches")

    print(f"[a1_eval_cohort] Fetching H2 ({H2_START} -> {H2_END}) pitches...")
    h2 = fetch_half_window_pitches(conn, H2_START, H2_END, pitcher_ids)
    print(f"  -> {len(h2)} pitches")

    h1_path = out_dir / "h1.parquet"
    h2_path = out_dir / "h2.parquet"
    manifest_path = out_dir / "manifest.json"

    h1.to_parquet(h1_path, index=False)
    h2.to_parquet(h2_path, index=False)

    manifest = {
        "schema_version": 1,
        "cohort_name": "a1_eval_cohort_2026_04_26",
        "purpose": (
            "Counterfactual pitch-call grades (A1) — grade-computation cohort "
            "(H1) + calibration-check cohort (H2). See "
            "docs/pitchgpt_sim_engine/EXECUTION_PLAN.md §6 A1 Gate 3."
        ),
        "frozen_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "seed": SEED,
        "splits": {
            "h1": {
                "date_start": H1_START,
                "date_end": H1_END,
                "pa_floor_per_pitcher": H1_PA_FLOOR,
                "parquet": str(h1_path.relative_to(PROJECT_ROOT)),
                **_fingerprint_dataframe(h1),
            },
            "h2": {
                "date_start": H2_START,
                "date_end": H2_END,
                "pa_floor_per_pitcher": H2_PA_FLOOR,
                "parquet": str(h2_path.relative_to(PROJECT_ROOT)),
                **_fingerprint_dataframe(h2),
            },
        },
        "qualifying_pitchers": {
            "n": int(len(qualifying)),
            "h1_pa_floor": H1_PA_FLOOR,
            "h2_pa_floor": H2_PA_FLOOR,
            "pitcher_id_min": int(qualifying["pitcher_id"].min()),
            "pitcher_id_max": int(qualifying["pitcher_id"].max()),
            "h1_pa_total": int(qualifying["h1_pa"].sum()),
            "h2_pa_total": int(qualifying["h2_pa"].sum()),
        },
        "downstream_consumers": [
            "src/dashboard/views/pitch_call_grades.py (A1 view)",
            "results/pitchgpt_sim/pitch_call_grades_2025/leaderboard.csv "
            "(future Phase 1 leaderboard build)",
        ],
        "notes": (
            "Read-only DuckDB pull. No grade computation here — only the row-ID "
            "freeze. Downstream Phase 1 grader can drive its rollouts off the "
            "row-key tuple (game_pk, at_bat_number, pitch_number)."
        ),
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[a1_eval_cohort] Wrote {h1_path} ({len(h1)} rows)")
    print(f"[a1_eval_cohort] Wrote {h2_path} ({len(h2)} rows)")
    print(f"[a1_eval_cohort] Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
