#!/usr/bin/env python
"""Backfill Statcast pitches for 2026-04-09 through 2026-04-20.

Fills the 12-day gap since the last refresh on 2026-04-12 (data through
2026-04-08). Idempotent via (game_pk, at_bat_number, pitch_number) dedup
in insert_pitches().
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_gap")


def main() -> None:
    from src.db.schema import get_connection
    from src.ingest.statcast_loader import load_statcast_range

    start, end = "2026-04-09", "2026-04-20"
    conn = get_connection(read_only=False)

    pre_max = conn.execute("SELECT MAX(game_date) FROM pitches").fetchone()[0]
    pre_total = conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
    logger.info("Pre-backfill: max=%s, total=%d", pre_max, pre_total)

    df = load_statcast_range(start, end, conn=conn)
    logger.info("Returned df: %d rows", 0 if df is None else len(df))

    post_max = conn.execute("SELECT MAX(game_date) FROM pitches").fetchone()[0]
    post_total = conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
    games_added = conn.execute(
        "SELECT COUNT(DISTINCT game_pk) FROM pitches WHERE game_date >= ?",
        [start],
    ).fetchone()[0]

    logger.info("Post-backfill: max=%s, total=%d", post_max, post_total)
    logger.info("Rows added: %d  |  Games in range: %d", post_total - pre_total, games_added)
    conn.close()


if __name__ == "__main__":
    main()
