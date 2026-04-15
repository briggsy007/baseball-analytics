#!/usr/bin/env python
"""
Daily ETL for the baseball analytics platform.

Designed to run each morning during the MLB season (e.g. via cron at 6 AM ET).
It refreshes yesterday's Statcast pitches, current-season aggregate stats from
FanGraphs, and the player ID crosswalk.

Usage
-----
    python -m src.ingest.daily_etl          # from the project root
    python src/ingest/daily_etl.py          # direct invocation
"""

from __future__ import annotations

import logging
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import duckdb

# ---------------------------------------------------------------------------
# Ensure the project root is on ``sys.path`` so ``src.*`` imports work when
# the script is invoked directly.
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.ingest.statcast_loader import (  # noqa: E402
    check_data_freshness,
    insert_pitches,
    load_player_id_map,
    load_season_batting_stats,
    load_season_pitching_stats,
    load_statcast_range,
    update_data_freshness,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("daily_etl")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_conn() -> duckdb.DuckDBPyConnection:
    """Open a connection to the project DuckDB database.

    Attempts to import the canonical helper from ``src.db.schema``; falls
    back to a plain ``duckdb.connect()`` if the schema module is
    unavailable.
    """
    try:
        from src.db.schema import init_db
        return init_db()
    except Exception:
        db_path = ROOT / "data" / "baseball.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(db_path))


def _enable_cache() -> None:
    """Enable pybaseball disk caching."""
    cache_dir = ROOT / "data" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        import pybaseball  # noqa: F811
        pybaseball.cache.enable()
    except Exception:
        logger.warning("Could not enable pybaseball cache")


def _refresh_matchup_cache(conn: duckdb.DuckDBPyConnection) -> None:
    """Rebuild the ``matchup_summary`` table from the current ``pitches`` data.

    This is a lightweight aggregation that powers the matchup explorer
    on the dashboard.
    """
    logger.info("Refreshing matchup_summary cache …")
    try:
        # Check if the pitches table has data.
        pitch_count = conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
        if pitch_count == 0:
            logger.info("No pitches in database — skipping matchup cache refresh")
            return

        conn.execute("BEGIN TRANSACTION")
        try:
            conn.execute("DELETE FROM matchup_summary")
            conn.execute("""
                INSERT INTO matchup_summary
                SELECT
                    pitcher_id,
                    batter_id,
                    pitch_type,
                    COUNT(*)                                           AS num_pitches,
                    AVG(release_speed)                                 AS avg_speed,
                    AVG(release_spin_rate)                             AS avg_spin,
                    AVG(pfx_x)                                        AS avg_pfx_x,
                    AVG(pfx_z)                                        AS avg_pfx_z,
                    AVG(CASE WHEN description IN (
                        'swinging_strike', 'swinging_strike_blocked',
                        'foul_tip'
                    ) THEN 1.0 ELSE 0.0 END) * 100.0                  AS whiff_rate,
                    AVG(CASE
                        WHEN events IN ('single') THEN 1.0
                        WHEN events IN ('double') THEN 1.0
                        WHEN events IN ('triple') THEN 1.0
                        WHEN events IN ('home_run') THEN 1.0
                        WHEN events IS NOT NULL
                             AND events NOT IN ('walk', 'hit_by_pitch',
                                                'catcher_interf', 'sac_fly',
                                                'sac_bunt', 'sac_fly_double_play',
                                                'sac_bunt_double_play')
                        THEN 0.0
                        ELSE NULL
                    END)                                               AS ba,
                    AVG(CASE
                        WHEN events = 'single' THEN 1.0
                        WHEN events = 'double' THEN 2.0
                        WHEN events = 'triple' THEN 3.0
                        WHEN events = 'home_run' THEN 4.0
                        WHEN events IS NOT NULL
                             AND events NOT IN ('walk', 'hit_by_pitch',
                                                'catcher_interf', 'sac_fly',
                                                'sac_bunt', 'sac_fly_double_play',
                                                'sac_bunt_double_play')
                        THEN 0.0
                        ELSE NULL
                    END)                                               AS slg,
                    SUM(woba_value) / NULLIF(SUM(woba_denom), 0)       AS woba
                FROM pitches
                WHERE pitch_type IS NOT NULL
                GROUP BY pitcher_id, batter_id, pitch_type
            """)
            conn.execute("COMMIT")
            row_count = conn.execute("SELECT COUNT(*) FROM matchup_summary").fetchone()[0]
            logger.info("matchup_summary rebuilt: %d rows", row_count)
        except Exception:
            conn.execute("ROLLBACK")
            raise
    except duckdb.CatalogException:
        logger.warning("matchup_summary table does not exist — skipping cache refresh")
    except Exception:
        logger.exception("Failed to refresh matchup_summary")


# ---------------------------------------------------------------------------
# Main ETL routine
# ---------------------------------------------------------------------------


def run_daily_etl(conn: Optional[duckdb.DuckDBPyConnection] = None) -> dict:
    """Execute the full daily ETL pipeline.

    Steps:
        1. Load yesterday's Statcast pitch data.
        2. Insert into the ``pitches`` table.
        3. Load current-season batting stats from FanGraphs.
        4. Load current-season pitching stats from FanGraphs.
        5. Refresh the player ID crosswalk for any new players.
        6. Rebuild the ``matchup_summary`` cache.
        7. Log a summary of what was updated.

    Args:
        conn: Optional pre-existing DuckDB connection. If ``None``, one
              will be opened (and closed at the end).

    Returns:
        Summary dict with keys ``pitches``, ``batting_rows``,
        ``pitching_rows``, ``players``.
    """
    own_conn = conn is None
    if own_conn:
        conn = _get_conn()

    _enable_cache()

    yesterday: date = date.today() - timedelta(days=1)
    yesterday_str: str = yesterday.isoformat()
    current_year: int = datetime.now().year

    summary: dict = {
        "date": yesterday_str,
        "season": current_year,
        "pitches": 0,
        "batting_rows": 0,
        "pitching_rows": 0,
        "players": 0,
    }

    # ── 1. Yesterday's Statcast pitches ───────────────────────────────────
    logger.info("Step 1/6: Loading Statcast data for %s", yesterday_str)
    if check_data_freshness(conn, "pitches", yesterday_str):
        logger.info("  Pitches already fresh through %s — skipping", yesterday_str)
    else:
        try:
            pitch_df = load_statcast_range(yesterday_str, yesterday_str)
            if pitch_df is not None and not pitch_df.empty:
                inserted = insert_pitches(conn, pitch_df)
                summary["pitches"] = inserted
                logger.info("  Inserted %d pitches for %s", inserted, yesterday_str)
            else:
                logger.info("  No pitches returned for %s (off-day?)", yesterday_str)
        except Exception:
            logger.exception("  Failed to load Statcast data for %s", yesterday_str)

    # ── 2. Season batting stats ───────────────────────────────────────────
    logger.info("Step 2/6: Loading %d batting stats", current_year)
    try:
        bat_df = load_season_batting_stats(current_year, conn=conn)
        summary["batting_rows"] = len(bat_df)
        logger.info("  Loaded %d batting stat rows", len(bat_df))
    except Exception:
        logger.exception("  Failed to load batting stats for %d", current_year)

    # ── 3. Season pitching stats ──────────────────────────────────────────
    logger.info("Step 3/6: Loading %d pitching stats", current_year)
    try:
        pit_df = load_season_pitching_stats(current_year, conn=conn)
        summary["pitching_rows"] = len(pit_df)
        logger.info("  Loaded %d pitching stat rows", len(pit_df))
    except Exception:
        logger.exception("  Failed to load pitching stats for %d", current_year)

    # ── 4. Player ID crosswalk ────────────────────────────────────────────
    logger.info("Step 4/6: Refreshing player ID map")
    try:
        players_df = load_player_id_map(conn=conn)
        summary["players"] = len(players_df)
        logger.info("  Player map: %d entries", len(players_df))
    except Exception:
        logger.exception("  Failed to load player ID map")

    # ── 5. Matchup cache ─────────────────────────────────────────────────
    logger.info("Step 5/6: Refreshing matchup cache")
    _refresh_matchup_cache(conn)

    # ── 6. Summary ────────────────────────────────────────────────────────
    logger.info("Step 6/6: ETL complete")
    logger.info(
        "  Summary — date: %s | pitches: %d | batting: %d | pitching: %d | players: %d",
        summary["date"],
        summary["pitches"],
        summary["batting_rows"],
        summary["pitching_rows"],
        summary["players"],
    )

    if own_conn:
        conn.close()

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    summary = run_daily_etl()
    print("\n=== Daily ETL Summary ===")
    for key, val in summary.items():
        print(f"  {key}: {val}")
