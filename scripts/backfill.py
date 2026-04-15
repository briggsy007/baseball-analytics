#!/usr/bin/env python
"""
Historical data backfill script for the baseball analytics platform.

Downloads Statcast pitch-by-pitch data, FanGraphs season stats, and the
Chadwick player ID crosswalk for a range of MLB seasons and loads them
into DuckDB.

Usage
-----
    python scripts/backfill.py                          # defaults: 2020-2025
    python scripts/backfill.py --start-year 2015 --end-year 2025
    python scripts/backfill.py --stats-only             # skip pitch-level data
    python scripts/backfill.py --pitches-only            # skip season aggregates

Baseball Savant caps queries at ~40 K rows and pybaseball chunks
automatically, but loading a full season in one call is slow and
memory-hungry.  This script breaks each season into monthly windows
(March – October) and wraps every network call in try/except so a
single failed chunk does not abort the whole run.
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

from tqdm import tqdm  # noqa: E402

from src.db.schema import init_db  # noqa: E402
from src.ingest.statcast_loader import (  # noqa: E402
    check_data_freshness,
    insert_pitches,
    load_player_id_map,
    load_season_batting_stats,
    load_season_pitching_stats,
    load_statcast_range,
    update_data_freshness,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill")

# ---------------------------------------------------------------------------
# Monthly date windows for a single MLB season.
# March 20 – October 31 covers spring training through the World Series.
# ---------------------------------------------------------------------------
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
        import pybaseball  # noqa: F811

        pybaseball.cache.enable()
        logger.info("pybaseball cache enabled (default location)")
    except Exception:
        logger.warning("Could not enable pybaseball cache — continuing without it")


def backfill_pitches(start_year: int, end_year: int, conn) -> int:
    """Download and insert Statcast pitches for every month of every season.

    Args:
        start_year: First season to load (inclusive).
        end_year:   Last season to load (inclusive).
        conn:       Open DuckDB connection.

    Returns:
        Total number of pitches inserted.
    """
    total_pitches = 0

    years = list(range(start_year, end_year + 1))
    pbar = tqdm(years, desc="Seasons (pitches)", unit="yr")

    for year in pbar:
        season_pitches = 0
        for month_start, month_end in _MONTH_WINDOWS:
            start_dt = f"{year}-{month_start}"
            end_dt = f"{year}-{month_end}"
            pbar.set_postfix_str(f"{start_dt}")

            # Skip if we already have data through this window
            if check_data_freshness(conn, "pitches", end_dt):
                logger.info("  %s – %s: already fresh, skipping", start_dt, end_dt)
                continue

            try:
                df = load_statcast_range(start_dt, end_dt)
                if df is not None and not df.empty:
                    inserted = insert_pitches(conn, df)
                    season_pitches += inserted
                    logger.info(
                        "  %s – %s: %d pitches fetched, %d inserted",
                        start_dt, end_dt, len(df), inserted,
                    )
            except Exception:
                logger.exception("  FAILED: %s – %s (skipping)", start_dt, end_dt)

        total_pitches += season_pitches
        logger.info("Season %d complete: %d pitches inserted", year, season_pitches)

    return total_pitches


def backfill_stats(start_year: int, end_year: int, conn) -> tuple[int, int]:
    """Download and insert FanGraphs season batting and pitching stats.

    Returns:
        Tuple of (batting_rows, pitching_rows) inserted across all seasons.
    """
    total_batting = 0
    total_pitching = 0

    years = list(range(start_year, end_year + 1))
    pbar = tqdm(years, desc="Seasons (stats) ", unit="yr")

    for year in pbar:
        pbar.set_postfix_str(f"batting {year}")
        try:
            bat_df = load_season_batting_stats(year, conn=conn)
            total_batting += len(bat_df)
            logger.info("  %d batting: %d rows", year, len(bat_df))
        except Exception:
            logger.exception("  FAILED batting stats for %d (skipping)", year)

        pbar.set_postfix_str(f"pitching {year}")
        try:
            pit_df = load_season_pitching_stats(year, conn=conn)
            total_pitching += len(pit_df)
            logger.info("  %d pitching: %d rows", year, len(pit_df))
        except Exception:
            logger.exception("  FAILED pitching stats for %d (skipping)", year)

    return total_batting, total_pitching


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill historical Statcast and FanGraphs data into DuckDB.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=2020,
        help="First season to load (inclusive, default: 2020).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=2025,
        help="Last season to load (inclusive, default: 2025).",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only load FanGraphs season stats (skip pitch-level data).",
    )
    parser.add_argument(
        "--pitches-only",
        action="store_true",
        help="Only load Statcast pitches (skip season stats).",
    )
    args = parser.parse_args()

    if args.stats_only and args.pitches_only:
        parser.error("--stats-only and --pitches-only are mutually exclusive")

    # ── Enable caching ────────────────────────────────────────────────────
    _enable_pybaseball_cache()

    # ── Initialise database ───────────────────────────────────────────────
    logger.info("Initializing database …")
    conn = init_db()

    total_pitches = 0
    total_batting = 0
    total_pitching = 0

    # ── Pitch-level data ──────────────────────────────────────────────────
    if not args.stats_only:
        logger.info(
            "=== Backfilling Statcast pitches: %d – %d ===",
            args.start_year, args.end_year,
        )
        total_pitches = backfill_pitches(args.start_year, args.end_year, conn)

    # ── Season-level stats ────────────────────────────────────────────────
    if not args.pitches_only:
        logger.info(
            "=== Backfilling FanGraphs stats: %d – %d ===",
            args.start_year, args.end_year,
        )
        total_batting, total_pitching = backfill_stats(args.start_year, args.end_year, conn)

    # ── Player ID crosswalk ───────────────────────────────────────────────
    logger.info("=== Loading Chadwick player ID map ===")
    try:
        players_df = load_player_id_map(conn=conn)
        total_players = len(players_df)
    except Exception:
        logger.exception("Failed to load player ID map")
        total_players = 0

    # ── Summary ───────────────────────────────────────────────────────────
    total_seasons = args.end_year - args.start_year + 1
    pitch_count = conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]

    print("\n" + "=" * 60)
    print("  BACKFILL COMPLETE")
    print("=" * 60)
    print(f"  Seasons processed:      {total_seasons} ({args.start_year}–{args.end_year})")
    print(f"  Pitches inserted (run): {total_pitches:,}")
    print(f"  Total pitches in DB:    {pitch_count:,}")
    print(f"  Batting stat rows:      {total_batting:,}")
    print(f"  Pitching stat rows:     {total_pitching:,}")
    print(f"  Players in crosswalk:   {total_players:,}")
    print("=" * 60)

    conn.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
