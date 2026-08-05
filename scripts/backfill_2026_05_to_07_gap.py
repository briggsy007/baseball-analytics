#!/usr/bin/env python
"""Backfill Statcast pitches for 2026-05-05 through 2026-07-23.

Fills the 81-day gap since the last refresh (data through 2026-05-04).
Pulls in 7-day chunks so a mid-run network failure only loses the current
week, and re-running resumes cleanly (insert_pitches dedups on
(game_pk, at_bat_number, pitch_number)).

After pitches, refreshes 2026 season batting/pitching stats, the player ID
crosswalk, rebuilds the matchup_summary cache, and updates the freshness
watermark. Writes a JSON status file so the run can be monitored.

Usage
-----
    python scripts/backfill_2026_05_to_07_gap.py [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_gap")

START = "2026-05-05"
END = "2026-07-23"
CHUNK_DAYS = 7

STATUS_PATH = Path(
    r"C:\Users\hunte\AppData\Local\Temp\claude"
    r"\C--Users-hunte-projects-baseball"
    r"\1284a947-3226-496c-b90a-c86e975dce67\scratchpad\backfill_status.json"
)


def _write_status(status: dict) -> None:
    try:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(json.dumps(status, indent=2, default=str))
    except Exception:
        logger.exception("Could not write status file")


def _chunks(start: str, end: str, days: int):
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    cur = d0
    while cur <= d1:
        chunk_end = min(cur + timedelta(days=days - 1), d1)
        yield cur.isoformat(), chunk_end.isoformat()
        cur = chunk_end + timedelta(days=1)


def main() -> None:
    global START, END
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    args = ap.parse_args()
    START, END = args.start, args.end

    from src.db.schema import get_connection
    from src.ingest.statcast_loader import (
        load_statcast_range,
        load_season_batting_stats,
        load_season_pitching_stats,
        load_player_id_map,
        update_data_freshness,
    )

    t0 = time.time()
    conn = get_connection(read_only=False)

    pre_max = conn.execute("SELECT MAX(game_date) FROM pitches").fetchone()[0]
    pre_total = conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
    logger.info("Pre-backfill: max=%s, total=%d", pre_max, pre_total)

    status = {
        "phase": "pitches",
        "started": datetime.now().isoformat(),
        "pre_max_game_date": str(pre_max),
        "pre_total_pitches": pre_total,
        "chunks_done": [],
        "done": False,
    }
    _write_status(status)

    chunk_list = list(_chunks(START, END, CHUNK_DAYS))
    for i, (cs, ce) in enumerate(chunk_list, 1):
        try:
            logger.info("[chunk %d/%d] Statcast %s .. %s", i, len(chunk_list), cs, ce)
            df = load_statcast_range(cs, ce, conn=conn)
            n = 0 if df is None else len(df)
            cur_total = conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
            logger.info("[chunk %d/%d] fetched=%d, table_total=%d", i, len(chunk_list), n, cur_total)
            status["chunks_done"].append({"start": cs, "end": ce, "fetched": n, "table_total": cur_total})
            _write_status(status)
        except Exception:
            logger.exception("[chunk %d/%d] FAILED for %s..%s (continuing)", i, len(chunk_list), cs, ce)
            status["chunks_done"].append({"start": cs, "end": ce, "error": True})
            _write_status(status)

    # Season aggregate stats (current season, full refresh)
    current_year = 2026
    status["phase"] = "season_stats"
    _write_status(status)
    try:
        logger.info("Loading %d batting stats", current_year)
        bat = load_season_batting_stats(current_year, conn=conn)
        status["batting_rows"] = len(bat)
    except Exception:
        logger.exception("batting stats failed")
        status["batting_rows"] = -1
    try:
        logger.info("Loading %d pitching stats", current_year)
        pit = load_season_pitching_stats(current_year, conn=conn)
        status["pitching_rows"] = len(pit)
    except Exception:
        logger.exception("pitching stats failed")
        status["pitching_rows"] = -1
    try:
        logger.info("Refreshing player ID map")
        players = load_player_id_map(conn=conn)
        status["players"] = len(players)
    except Exception:
        logger.exception("player map failed")
        status["players"] = -1
    _write_status(status)

    # Rebuild matchup cache from refreshed pitches
    status["phase"] = "matchup_cache"
    _write_status(status)
    try:
        from src.ingest.daily_etl import _refresh_matchup_cache
        _refresh_matchup_cache(conn)
        status["matchup_cache"] = True
    except Exception:
        logger.exception("matchup cache rebuild failed")
        status["matchup_cache"] = False

    post_max = conn.execute("SELECT MAX(game_date) FROM pitches").fetchone()[0]
    post_total = conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
    games_added = conn.execute(
        "SELECT COUNT(DISTINCT game_pk) FROM pitches WHERE game_date >= ?",
        [START],
    ).fetchone()[0]

    try:
        update_data_freshness(conn, "pitches", str(post_max), post_total)
    except Exception:
        logger.exception("freshness watermark update failed")

    conn.close()

    status.update({
        "phase": "done",
        "done": True,
        "post_max_game_date": str(post_max),
        "post_total_pitches": post_total,
        "rows_added": post_total - pre_total,
        "games_in_range": games_added,
        "elapsed_sec": round(time.time() - t0, 1),
        "finished": datetime.now().isoformat(),
    })
    _write_status(status)

    logger.info("Post-backfill: max=%s, total=%d", post_max, post_total)
    logger.info("Rows added: %d | Games in range: %d | Elapsed: %.1fs",
                post_total - pre_total, games_added, time.time() - t0)
    logger.info("BACKFILL_COMPLETE")


if __name__ == "__main__":
    main()
