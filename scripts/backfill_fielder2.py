#!/usr/bin/env python
"""
Backfill fielder_2 (catcher ID) into existing pitches data.

Adds the fielder_2 column to the pitches table if it does not exist,
then downloads Statcast data month-by-month for 2024-2026 and updates
the fielder_2 value for matching rows based on
(game_pk, at_bat_number, pitch_number).

Usage
-----
    python scripts/backfill_fielder2.py
    python scripts/backfill_fielder2.py --start-year 2024 --end-year 2026
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure ``src`` is importable regardless of the working directory.
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pybaseball  # noqa: E402

from src.db.schema import get_connection  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_fielder2")

# Monthly windows matching backfill.py
_MONTH_WINDOWS: list[tuple[str, str]] = [
    ("03-20", "03-31"),
    ("04-01", "04-30"),
    ("05-01", "05-31"),
    ("06-01", "06-30"),
    ("07-01", "07-31"),
    ("08-01", "08-31"),
    ("09-01", "09-30"),
    ("10-01", "10-31"),
]


def _enable_pybaseball_cache() -> None:
    """Enable pybaseball's disk cache in ``data/cache/``."""
    cache_dir = ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        pybaseball.cache.enable()
    except Exception:
        pass


def _alter_table_add_fielder2(conn: duckdb.DuckDBPyConnection) -> None:
    """Add fielder_2 column to pitches table if it does not already exist."""
    # Check if column already exists
    try:
        conn.execute("SELECT fielder_2 FROM pitches LIMIT 1")
        logger.info("fielder_2 column already exists in pitches table.")
    except duckdb.BinderException:
        logger.info("Adding fielder_2 column to pitches table...")
        conn.execute("ALTER TABLE pitches ADD COLUMN fielder_2 INTEGER")
        logger.info("fielder_2 column added successfully.")


def _backfill_month(
    conn: duckdb.DuckDBPyConnection,
    year: int,
    start_md: str,
    end_md: str,
) -> int:
    """Download Statcast data for one month and update fielder_2 in existing rows.

    Returns the number of rows updated.
    """
    start_date = f"{year}-{start_md}"
    end_date = f"{year}-{end_md}"

    logger.info("Fetching Statcast data for %s to %s ...", start_date, end_date)
    try:
        raw = pybaseball.statcast(start_dt=start_date, end_dt=end_date)
    except Exception:
        logger.exception("Failed to fetch Statcast data for %s - %s", start_date, end_date)
        return 0

    if raw is None or raw.empty:
        logger.info("No data returned for %s - %s", start_date, end_date)
        return 0

    # We only need the join keys + fielder_2
    needed_cols = ["game_pk", "at_bat_number", "pitch_number", "fielder_2"]
    missing = [c for c in needed_cols if c not in raw.columns]
    if missing:
        logger.warning("Missing columns in Statcast data: %s", missing)
        return 0

    update_df = raw[needed_cols].copy()
    update_df = update_df.dropna(subset=["fielder_2"])
    update_df["game_pk"] = pd.to_numeric(update_df["game_pk"], errors="coerce")
    update_df["at_bat_number"] = pd.to_numeric(update_df["at_bat_number"], errors="coerce")
    update_df["pitch_number"] = pd.to_numeric(update_df["pitch_number"], errors="coerce")
    update_df["fielder_2"] = pd.to_numeric(update_df["fielder_2"], errors="coerce")
    update_df = update_df.dropna()
    update_df["game_pk"] = update_df["game_pk"].astype(int)
    update_df["at_bat_number"] = update_df["at_bat_number"].astype(int)
    update_df["pitch_number"] = update_df["pitch_number"].astype(int)
    update_df["fielder_2"] = update_df["fielder_2"].astype(int)

    if update_df.empty:
        logger.info("No fielder_2 data for %s - %s", start_date, end_date)
        return 0

    logger.info("Updating fielder_2 for %d pitches (%s - %s)...", len(update_df), start_date, end_date)

    # Register as temp view and run UPDATE ... SET ... FROM
    conn.register("_stg_fielder2", update_df)
    try:
        conn.execute("""
            UPDATE pitches
            SET fielder_2 = s.fielder_2
            FROM _stg_fielder2 AS s
            WHERE pitches.game_pk = s.game_pk
              AND pitches.at_bat_number = s.at_bat_number
              AND pitches.pitch_number = s.pitch_number
              AND pitches.fielder_2 IS NULL
        """)
    finally:
        conn.unregister("_stg_fielder2")

    # Count how many are now populated for this date range
    count = conn.execute(f"""
        SELECT COUNT(*) FROM pitches
        WHERE fielder_2 IS NOT NULL
          AND game_date >= '{start_date}'
          AND game_date <= '{end_date}'
    """).fetchone()[0]

    logger.info("  %d rows now have fielder_2 for %s - %s", count, start_date, end_date)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill fielder_2 (catcher ID) into pitches")
    parser.add_argument("--start-year", type=int, default=2024, help="First season to backfill (default: 2024)")
    parser.add_argument("--end-year", type=int, default=2026, help="Last season to backfill (default: 2026)")
    args = parser.parse_args()

    _enable_pybaseball_cache()

    logger.info("Opening database...")
    conn = get_connection()

    # Step 1: ALTER TABLE to add column
    _alter_table_add_fielder2(conn)

    # Step 2: Backfill month-by-month
    total_updated = 0
    for year in range(args.start_year, args.end_year + 1):
        logger.info("=== Backfilling fielder_2 for %d ===", year)
        for start_md, end_md in _MONTH_WINDOWS:
            try:
                count = _backfill_month(conn, year, start_md, end_md)
                total_updated += count
            except Exception:
                logger.exception("Error backfilling %d %s-%s", year, start_md, end_md)

    # Summary
    total_f2 = conn.execute("SELECT COUNT(*) FROM pitches WHERE fielder_2 IS NOT NULL").fetchone()[0]
    total_all = conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
    logger.info(
        "Backfill complete. %d / %d pitches have fielder_2 (%.1f%%)",
        total_f2, total_all, (total_f2 / total_all * 100) if total_all > 0 else 0,
    )

    conn.close()


if __name__ == "__main__":
    main()
