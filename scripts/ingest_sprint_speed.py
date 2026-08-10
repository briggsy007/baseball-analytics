#!/usr/bin/env python
"""
Ingest per-player Statcast sprint speed from Baseball Savant into DuckDB.

Pulls ``pybaseball.statcast_sprint_speed(year, min_opp)`` per season and
writes a ``sprint_speed`` table keyed on (season, player_id). Intended
consumer: WS3.3 (DPI v2 sprint-speed feature on topped/weak contact).
Pattern follows ``scripts/ingest_team_oaa.py``.

Semantics
---------
- **Replace-by-season** (idempotent): each ingested season is DELETEd then
  re-INSERTed inside one transaction, so re-runs converge to source state.
  Seasons whose fetch fails (after one retry) are left untouched.
- **Single short write window**: all HTTP fetches complete BEFORE the
  read-write DuckDB connection opens; the connection closes immediately
  after the inserts (single-writer discipline).
- **No fabrication**: players absent from the Savant leaderboard are simply
  absent; NaN fields (e.g. ``hp_to_1b``) land as NULL. League-mean
  imputation happens at feature time, never in the DB.
- Duplicate (season, player_id) rows from the source (mid-season trades)
  are collapsed to the row with the most ``competitive_runs``; the drop
  count is logged in the audit JSON.

Audit trail
-----------
``results/ingest/sprint_speed_ingest_<UTCstamp>.json``: per-season row
counts, min_opp, fetch timestamps, retry/gap record, and (post-write,
via a separate read-only connection) batted-ball join coverage vs
``pitches``.

Usage
-----
    python scripts/ingest_sprint_speed.py                    # 2015-2026, min_opp=10
    python scripts/ingest_sprint_speed.py --seasons 2026     # refresh one season
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_sprint_speed")

DEFAULT_SEASONS: tuple[int, ...] = tuple(range(2015, 2027))
DEFAULT_MIN_OPP: int = 10
LOG_DIR: Path = ROOT / "results" / "ingest"

DDL = """
CREATE TABLE IF NOT EXISTS sprint_speed (
    season           INTEGER NOT NULL,
    player_id        BIGINT  NOT NULL,
    sprint_speed     DOUBLE,
    competitive_runs INTEGER,
    hp_to_1b         DOUBLE,
    fetched_utc      TIMESTAMP
)
"""

KEEP_COLS = ["season", "player_id", "sprint_speed", "competitive_runs", "hp_to_1b", "fetched_utc"]


def fetch_season(year: int, min_opp: int, retries: int = 1, retry_wait_s: float = 10.0) -> pd.DataFrame | None:
    """Fetch one season's sprint-speed leaderboard; retry once, else None."""
    import pybaseball

    for attempt in range(retries + 1):
        try:
            df = pybaseball.statcast_sprint_speed(year, min_opp)
        except Exception as exc:  # noqa: BLE001 — network layer raises many types
            logger.warning("year=%d fetch attempt %d failed: %s", year, attempt + 1, exc)
            if attempt < retries:
                time.sleep(retry_wait_s)
            continue
        if df is None or df.empty:
            logger.warning("year=%d returned empty frame (attempt %d)", year, attempt + 1)
            if attempt < retries:
                time.sleep(retry_wait_s)
            continue
        return df
    return None


def prepare_season(df: pd.DataFrame, year: int, fetched_utc: datetime) -> tuple[pd.DataFrame, int]:
    """Normalize one season's frame to the table schema.

    Returns (frame, n_duplicates_dropped). Duplicate (season, player_id)
    rows keep the entry with the most competitive_runs.
    """
    out = df.copy()
    out["season"] = int(year)
    out["fetched_utc"] = fetched_utc

    for col in ("sprint_speed", "hp_to_1b"):
        out[col] = pd.to_numeric(out.get(col), errors="coerce")
    out["competitive_runs"] = pd.to_numeric(out.get("competitive_runs"), errors="coerce").astype("Int64")
    out["player_id"] = pd.to_numeric(out["player_id"], errors="coerce").astype("int64")

    n_before = len(out)
    out = (
        out.sort_values("competitive_runs", ascending=False)
        .drop_duplicates(subset=["season", "player_id"], keep="first")
        .reset_index(drop=True)
    )
    n_dupes = n_before - len(out)
    if n_dupes:
        logger.info("year=%d dropped %d duplicate (season, player_id) rows", year, n_dupes)
    return out[KEEP_COLS], n_dupes


def write_seasons(frames: dict[int, pd.DataFrame], db_path: str | None) -> None:
    """Open ONE read-write connection, replace-by-season, close promptly."""
    con = get_connection(db_path=db_path, read_only=False)
    try:
        con.execute(DDL)
        con.execute("BEGIN TRANSACTION")
        for season, frame in sorted(frames.items()):
            con.execute("DELETE FROM sprint_speed WHERE season = ?", [season])
            con.register("_ss_batch", frame)
            con.execute(
                "INSERT INTO sprint_speed "
                "SELECT season, player_id, sprint_speed, competitive_runs, hp_to_1b, fetched_utc "
                "FROM _ss_batch"
            )
            con.unregister("_ss_batch")
            logger.info("season=%d wrote %d rows", season, len(frame))
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.close()


def coverage_check(seasons: list[int], db_path: str | None) -> list[dict]:
    """Read-only: % of batted-ball events (pitches.type='X') with a
    same-season sprint_speed row for the batter."""
    con = get_connection(db_path=db_path, read_only=True)
    try:
        rows = con.execute(
            """
            SELECT
                CAST(EXTRACT(year FROM p.game_date) AS INTEGER) AS season,
                COUNT(*)                                        AS n_bip,
                SUM(CASE WHEN ss.player_id IS NOT NULL THEN 1 ELSE 0 END) AS n_matched
            FROM pitches p
            LEFT JOIN sprint_speed ss
              ON ss.player_id = p.batter_id
             AND ss.season    = CAST(EXTRACT(year FROM p.game_date) AS INTEGER)
            WHERE p.type = 'X'
              AND CAST(EXTRACT(year FROM p.game_date) AS INTEGER) BETWEEN ? AND ?
            GROUP BY 1
            ORDER BY 1
            """,
            [min(seasons), max(seasons)],
        ).fetchall()
    finally:
        con.close()
    return [
        {
            "season": int(s),
            "n_bip": int(n),
            "n_matched": int(m),
            "pct_matched": round(100.0 * m / n, 2) if n else None,
        }
        for s, n, m in rows
    ]


def run(seasons: Iterable[int], min_opp: int, db_path: str | None = None) -> dict:
    seasons = sorted(set(int(s) for s in seasons))

    # Phase 1: fetch everything BEFORE touching the DB (short write window).
    frames: dict[int, pd.DataFrame] = {}
    season_log: dict[str, dict] = {}
    for year in seasons:
        logger.info("Fetching sprint speed for season %d (min_opp=%d)", year, min_opp)
        fetched_utc = datetime.now(timezone.utc)
        raw = fetch_season(year, min_opp)
        if raw is None:
            season_log[str(year)] = {"status": "FETCH_FAILED", "rows": 0}
            logger.error("season=%d: fetch failed after retry — leaving existing rows untouched", year)
            continue
        frame, n_dupes = prepare_season(raw, year, fetched_utc)
        frames[year] = frame
        season_log[str(year)] = {
            "status": "ok",
            "rows": int(len(frame)),
            "duplicates_dropped": int(n_dupes),
            "fetched_utc": fetched_utc.isoformat(),
        }

    # Phase 2: single short read-write window.
    if frames:
        write_seasons(frames, db_path)
    else:
        logger.error("No seasons fetched successfully; nothing written.")

    # Phase 3: read-only coverage check (write connection already closed).
    coverage = coverage_check(seasons, db_path) if frames else []

    audit = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": (
            "Baseball Savant sprint-speed leaderboard via "
            f"pybaseball.statcast_sprint_speed(year, min_opp={min_opp}). "
            "Sprint speed = ft/s in the player's fastest one-second window "
            "(mean of roughly the top two-thirds of competitive runs)."
        ),
        "min_opp": min_opp,
        "seasons_requested": seasons,
        "seasons_written": sorted(frames.keys()),
        "seasons_failed": [int(y) for y, v in season_log.items() if v["status"] != "ok"],
        "semantics": "replace-by-season (DELETE season then INSERT, one transaction)",
        "table": "sprint_speed(season, player_id, sprint_speed, competitive_runs, hp_to_1b, fetched_utc)",
        "per_season": season_log,
        "join_coverage_vs_pitches_bip": {
            "definition": (
                "share of pitches rows with type='X' (ball in play) whose "
                "batter_id has a same-season sprint_speed row"
            ),
            "per_season": coverage,
        },
        "no_fabrication_note": (
            "Players below min_opp or absent from the leaderboard are absent "
            "from the table; hp_to_1b NaN stays NULL. Imputation (league mean "
            "~27 ft/s) is a feature-time decision, not a DB write."
        ),
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOG_DIR / f"sprint_speed_ingest_{stamp}.json"
    log_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    logger.info("Wrote audit log %s", log_path)
    return audit


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seasons", nargs="+", type=int, default=list(DEFAULT_SEASONS),
        help="Seasons to ingest (default: 2015-2026).",
    )
    p.add_argument(
        "--min-opp", type=int, default=DEFAULT_MIN_OPP,
        help="Savant minimum competitive-run opportunities (default 10; low to maximize coverage).",
    )
    p.add_argument(
        "--db-path", type=str, default=None,
        help="Optional DuckDB path override (default: data/baseball.duckdb).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        audit = run(args.seasons, args.min_opp, args.db_path)
    except Exception as exc:
        logger.exception("ingest_sprint_speed failed: %s", exc)
        return 1
    if audit["seasons_failed"]:
        logger.warning("Completed with gaps: %s", audit["seasons_failed"])
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
