#!/usr/bin/env python
"""
Backfill ``players.position`` for rows where it is NULL.

Source
------
MLB Stats API ``/api/v1/people?personIds=...`` returns ``primaryPosition``
for each player.  The endpoint accepts batched ids (URL-length is the only
limit; we chunk at 100 ids/request to stay well under any limit and to keep
retries cheap).

Strategy
--------
1. Read all NULL-position players from ``players``.
2. Batch-fetch their primary position abbreviations.
3. For ids the API returns nothing for (retired/minor-league/spring-only
   non-roster invitees), leave the row NULL.
4. UPDATE in a single atomic transaction.

Idempotent: re-running only touches still-NULL rows.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

import duckdb
import requests

ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db.schema import DEFAULT_DB_PATH  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_player_positions")

PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
USER_AGENT = "BaseballAnalyticsPlatform/1.0 (contact: analytics@example.com)"
REQUEST_TIMEOUT = 30
INTER_REQUEST_SLEEP = 0.25
BATCH_SIZE = 100


def _chunk(seq: list[int], size: int) -> Iterable[list[int]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def fetch_positions(ids: list[int], session: requests.Session) -> dict[int, str]:
    """Return {player_id -> position abbreviation} for ids resolved by StatsAPI."""
    out: dict[int, str] = {}
    total = len(ids)
    for batch_idx, chunk in enumerate(_chunk(ids, BATCH_SIZE), start=1):
        params = {"personIds": ",".join(str(x) for x in chunk)}
        for attempt in range(1, 4):
            try:
                resp = session.get(
                    PEOPLE_URL,
                    params=params,
                    timeout=REQUEST_TIMEOUT,
                    headers={"User-Agent": USER_AGENT},
                )
                if resp.status_code >= 500:
                    raise RuntimeError(f"server {resp.status_code}")
                resp.raise_for_status()
                payload = resp.json()
                break
            except Exception as exc:
                wait = 2 * attempt
                logger.warning(
                    "batch %d attempt %d failed: %s (sleep %ds)",
                    batch_idx, attempt, exc, wait,
                )
                time.sleep(wait)
        else:
            logger.error("batch %d exhausted retries -- skipping", batch_idx)
            continue

        for person in payload.get("people", []):
            pid = person.get("id")
            pos = (person.get("primaryPosition") or {}).get("abbreviation")
            if pid is not None and pos:
                out[int(pid)] = str(pos)

        if batch_idx % 10 == 0 or batch_idx * BATCH_SIZE >= total:
            logger.info(
                "fetched batch %d (%d/%d players resolved so far)",
                batch_idx, len(out), total,
            )
        time.sleep(INTER_REQUEST_SLEEP)
    return out


def _connect_write(db_path: Path, max_wait_seconds: int = 15 * 60) -> duckdb.DuckDBPyConnection:
    start = time.time()
    backoff = 30
    attempts = 0
    while True:
        attempts += 1
        try:
            return duckdb.connect(str(db_path), read_only=False)
        except duckdb.IOException as exc:
            elapsed = time.time() - start
            if elapsed > max_wait_seconds:
                raise RuntimeError(
                    f"Could not acquire DB write lock after {elapsed:.0f}s "
                    f"({attempts} attempts): {exc}"
                ) from exc
            logger.warning(
                "DB write lock busy (attempt %d, %.0fs elapsed): %s -- sleeping %ds",
                attempts, elapsed, exc, backoff,
            )
            time.sleep(backoff)


def run(db_path: Path = DEFAULT_DB_PATH) -> dict:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        before_total, before_with_pos = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN position IS NOT NULL THEN 1 ELSE 0 END) FROM players"
        ).fetchone()
        ids = [int(r[0]) for r in conn.execute(
            "SELECT player_id FROM players WHERE position IS NULL ORDER BY player_id"
        ).fetchall()]
    logger.info(
        "before: %d total, %d with position; %d NULL to resolve",
        before_total, before_with_pos, len(ids),
    )

    session = requests.Session()
    pos_map = fetch_positions(ids, session)
    logger.info("StatsAPI resolved %d / %d ids", len(pos_map), len(ids))

    if not pos_map:
        logger.warning("no positions to write -- exiting")
        return {"resolved": 0}

    import pandas as pd
    upd_df = pd.DataFrame(
        [{"player_id": int(k), "position": v} for k, v in pos_map.items()]
    )

    conn = _connect_write(db_path)
    try:
        conn.register("pos_updates", upd_df)
        conn.execute("BEGIN TRANSACTION")
        conn.execute(
            """
            UPDATE players AS p
            SET position = u.position
            FROM pos_updates AS u
            WHERE p.player_id = u.player_id AND p.position IS NULL
            """
        )
        conn.execute("COMMIT")
        conn.unregister("pos_updates")
        after_total, after_with_pos = conn.execute(
            "SELECT COUNT(*), SUM(CASE WHEN position IS NOT NULL THEN 1 ELSE 0 END) FROM players"
        ).fetchone()
    finally:
        conn.close()

    logger.info(
        "after: %d total, %d with position (delta=%d, still NULL=%d)",
        after_total, after_with_pos,
        after_with_pos - before_with_pos, after_total - after_with_pos,
    )
    return {
        "before_with_pos": int(before_with_pos),
        "after_with_pos": int(after_with_pos),
        "delta": int(after_with_pos - before_with_pos),
        "still_null": int(after_total - after_with_pos),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        run(db_path=args.db_path)
        return 0
    except Exception as exc:
        logger.exception("backfill_player_positions failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
