#!/usr/bin/env python
"""
Backfill roster transactions from the MLB Stats API into the ``transactions``
DuckDB table for the requested date window.

This is the unblocker for MechanixAE flagship validation: the supervised
injury-label set needs a league-wide, multi-season sample of IL placements.

Usage (from project root):

    python scripts/ingest_transactions.py --start-date 2015-01-01 \
                                          --end-date   2024-12-31
    python scripts/ingest_transactions.py --start-date 2023-03-01 \
                                          --end-date   2023-03-31 --dry-run

Endpoint
--------
    https://statsapi.mlb.com/api/v1/transactions
        ?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD

The API has no documented monthly row cap but we chunk by month anyway for
reliability (single-day windows are wasteful; season-long windows occasionally
time out).

Idempotency
-----------
The ``transactions`` table uses ``transaction_id`` as PRIMARY KEY. We issue a
delete-by-ids + insert on every chunk so re-running is safe and converges to
the latest API state for every id seen.

No team filter: research/validation work is now league-wide.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

import requests

# Ensure the project root is on sys.path so ``src`` is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.schema import DEFAULT_DB_PATH, init_db  # noqa: E402


# ── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_transactions")


# ── Constants ──────────────────────────────────────────────────────────────

BASE_URL = "https://statsapi.mlb.com"
TRANSACTIONS_URL = f"{BASE_URL}/api/v1/transactions"
USER_AGENT = "BaseballAnalyticsPlatform/1.0 (contact: analytics@example.com)"
REQUEST_TIMEOUT = 30  # seconds — transaction windows can be large
INTER_REQUEST_SLEEP = 0.5  # be polite: ~2 req/sec max

# Columns that every row written to the transactions table must carry.
TRANSACTION_COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "player_id",
    "player_name",
    "team",
    "from_team",
    "to_team",
    "transaction_type",
    "description",
    "transaction_date",
)


# ── HTTP helpers ───────────────────────────────────────────────────────────


def _fetch_window(
    session: requests.Session,
    start: date,
    end: date,
    *,
    max_retries: int = 2,
) -> list[dict]:
    """Fetch a single date window from the StatsAPI.

    Retries once on 5xx / connection errors with exponential backoff. A 4xx
    response propagates immediately (caller decides whether to skip).

    Returns the ``transactions`` list from the response (possibly empty).
    """
    params = {
        "startDate": start.strftime("%Y-%m-%d"),
        "endDate": end.strftime("%Y-%m-%d"),
    }
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            resp = session.get(
                TRANSACTIONS_URL, params=params, timeout=REQUEST_TIMEOUT
            )
            # 422 / 4xx: log and bail out of retries; the window is skipped.
            if 400 <= resp.status_code < 500:
                logger.warning(
                    "  %s..%s: HTTP %d, skipping window",
                    start, end, resp.status_code,
                )
                return []
            resp.raise_for_status()
            return resp.json().get("transactions", []) or []
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    "  %s..%s: transient error (%s); retry %d/%d in %ds",
                    start, end, type(exc).__name__, attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
            else:
                logger.error("  %s..%s: gave up after %d retries (%s)",
                             start, end, max_retries, exc)
                return []
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            last_exc = exc
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    "  %s..%s: HTTP %s; retry %d/%d in %ds",
                    start, end, code, attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
            else:
                logger.error("  %s..%s: gave up after %d retries (HTTP %s)",
                             start, end, max_retries, code)
                return []
    # Unreachable, but keep mypy / linters quiet.
    if last_exc is not None:
        logger.error("  %s..%s: final failure %s", start, end, last_exc)
    return []


def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


# ── Payload mapping ────────────────────────────────────────────────────────


def _parse_iso_date(s: Optional[str]) -> Optional[date]:
    """Parse an ISO-8601 date/datetime string, tolerating trailing ``Z``.

    Returns ``None`` for empty / unparsable input.
    """
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).date()
    except (ValueError, TypeError):
        # Some responses ship bare ``YYYY-MM-DD``.
        try:
            return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return None


def _pick_team_abbrev(txn: dict) -> str:
    """Best-effort team abbreviation / name for the ``team`` column.

    Preference order: toTeam > team > fromTeam. Falls back to empty string.
    The legacy column historically stored PHI-style abbreviations, but the
    StatsAPI exposes only ``name`` on the transactions endpoint — we store
    the full name so we never silently truncate (downstream readers match
    on substring / LIKE queries).
    """
    for key in ("toTeam", "team", "fromTeam"):
        sub = txn.get(key) or {}
        nm = sub.get("name") or sub.get("abbreviation")
        if nm:
            return str(nm)
    return ""


def map_transaction(txn: dict) -> Optional[dict]:
    """Map one raw StatsAPI transaction dict into a row for the DB.

    Returns ``None`` if the transaction lacks the minimum identifiers the
    table requires (``transaction_id`` — it is the PRIMARY KEY and NOT NULL).
    """
    tid = txn.get("id")
    if tid is None:
        return None
    try:
        tid_int = int(tid)
    except (TypeError, ValueError):
        return None

    person = txn.get("person") or {}
    from_team = (txn.get("fromTeam") or {}).get("name") or ""
    to_team = (txn.get("toTeam") or {}).get("name") or ""

    # Date preference: effectiveDate > date > resolutionDate. StatsAPI
    # usually ships all three and they agree. effectiveDate is the closest
    # thing to "when it happened" for injured-list placements.
    txn_date = (
        _parse_iso_date(txn.get("effectiveDate"))
        or _parse_iso_date(txn.get("date"))
        or _parse_iso_date(txn.get("resolutionDate"))
    )

    pid_raw = person.get("id")
    try:
        pid = int(pid_raw) if pid_raw is not None else None
    except (TypeError, ValueError):
        pid = None

    return {
        "transaction_id": tid_int,
        "player_id": pid,
        "player_name": (person.get("fullName") or "") or None,
        "team": _pick_team_abbrev(txn),
        "from_team": from_team,
        "to_team": to_team,
        "transaction_type": (txn.get("typeDesc") or "") or None,
        "description": (txn.get("description") or "") or None,
        "transaction_date": txn_date,
    }


def map_transactions(payload: Iterable[dict]) -> list[dict]:
    """Map an iterable of raw StatsAPI transactions, dropping any that lack an id.

    Also deduplicates by ``transaction_id`` (keeps the last occurrence,
    matching ``INSERT OR REPLACE`` semantics).
    """
    seen: dict[int, dict] = {}
    for txn in payload:
        row = map_transaction(txn)
        if row is None:
            continue
        seen[row["transaction_id"]] = row
    return list(seen.values())


# ── Date-window helpers ────────────────────────────────────────────────────


def iter_month_windows(start: date, end: date) -> Iterable[tuple[date, date]]:
    """Yield ``(window_start, window_end)`` pairs that tile ``[start, end]``
    in calendar-month chunks.  The last chunk may be shorter.

    Example: ``iter_month_windows(2023-03-15, 2023-05-10)`` →
        (2023-03-15, 2023-03-31),
        (2023-04-01, 2023-04-30),
        (2023-05-01, 2023-05-10).
    """
    cur = start
    while cur <= end:
        # Last day of current month
        if cur.month == 12:
            next_month_first = date(cur.year + 1, 1, 1)
        else:
            next_month_first = date(cur.year, cur.month + 1, 1)
        month_end = next_month_first - timedelta(days=1)
        chunk_end = min(month_end, end)
        yield (cur, chunk_end)
        cur = chunk_end + timedelta(days=1)


# ── DB helpers ─────────────────────────────────────────────────────────────


def upsert_rows(conn, rows: list[dict]) -> int:
    """Idempotently upsert ``rows`` into the transactions table.

    Strategy: DELETE by the set of transaction_ids, then INSERT. Wrapped in
    a single transaction so readers never see a hole.

    Returns the number of rows inserted (== ``len(rows)`` on success).
    """
    if not rows:
        return 0

    ids = [r["transaction_id"] for r in rows]

    conn.execute("BEGIN TRANSACTION")
    try:
        # DELETE the incoming ids in batches (DuckDB can handle large IN
        # clauses but we stay conservative for portability).
        BATCH = 500
        for i in range(0, len(ids), BATCH):
            chunk = ids[i : i + BATCH]
            placeholders = ",".join(["?"] * len(chunk))
            conn.execute(
                f"DELETE FROM transactions WHERE transaction_id IN ({placeholders})",
                chunk,
            )

        # INSERT all rows in one parameterised executemany.
        insert_sql = (
            "INSERT INTO transactions ("
            + ", ".join(TRANSACTION_COLUMNS)
            + ") VALUES ("
            + ", ".join(["?"] * len(TRANSACTION_COLUMNS))
            + ")"
        )
        params = [
            [r.get(c) for c in TRANSACTION_COLUMNS] for r in rows
        ]
        conn.executemany(insert_sql, params)
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    return len(rows)


# ── Stats tracking ─────────────────────────────────────────────────────────


@dataclass
class RunStats:
    windows_ok: int = 0
    windows_failed: int = 0
    raw_fetched: int = 0
    rows_written: int = 0
    per_year: dict[int, int] = field(default_factory=dict)
    per_type: dict[str, int] = field(default_factory=dict)
    unique_players: set[int] = field(default_factory=set)
    unique_teams: set[str] = field(default_factory=set)

    def record(self, rows: list[dict]) -> None:
        self.rows_written += len(rows)
        for r in rows:
            d = r.get("transaction_date")
            if isinstance(d, date):
                self.per_year[d.year] = self.per_year.get(d.year, 0) + 1
            t = r.get("transaction_type") or "(unknown)"
            self.per_type[t] = self.per_type.get(t, 0) + 1
            if r.get("player_id") is not None:
                self.unique_players.add(int(r["player_id"]))
            for k in ("team", "from_team", "to_team"):
                v = r.get(k)
                if v:
                    self.unique_teams.add(str(v))


# ── CLI driver ─────────────────────────────────────────────────────────────


def run_ingest(
    start_date: date,
    end_date: date,
    *,
    db_path: Optional[Path] = None,
    dry_run: bool = False,
    sleep_seconds: float = INTER_REQUEST_SLEEP,
) -> RunStats:
    """Core entry point — fetch StatsAPI month windows and upsert into DuckDB.

    Args:
        start_date: Inclusive lower bound.
        end_date: Inclusive upper bound.
        db_path: Optional override for the DuckDB file.
        dry_run: If True, fetch & parse but skip the DB write.
        sleep_seconds: Politeness delay between HTTP requests.

    Returns a populated ``RunStats``.
    """
    session = _new_session()
    stats = RunStats()
    conn = None
    if not dry_run:
        conn = init_db(str(db_path) if db_path else None)

    try:
        windows = list(iter_month_windows(start_date, end_date))
        logger.info(
            "Backfilling transactions for %s..%s across %d monthly windows (dry_run=%s)",
            start_date, end_date, len(windows), dry_run,
        )

        for i, (ws, we) in enumerate(windows, start=1):
            raw = _fetch_window(session, ws, we)
            if not raw:
                # Could be a legitimate empty window (off-season early days)
                # or a failure we already logged.
                stats.windows_failed += 1 if raw is None else 0
                # Still mark ok if HTTP 200 returned zero rows.
                stats.windows_ok += 1
                logger.info("  [%d/%d] %s..%s: 0 raw txns",
                            i, len(windows), ws, we)
                time.sleep(sleep_seconds)
                continue

            stats.windows_ok += 1
            stats.raw_fetched += len(raw)
            mapped = map_transactions(raw)

            if conn is not None:
                try:
                    written = upsert_rows(conn, mapped)
                except Exception:
                    logger.exception(
                        "  [%d/%d] %s..%s: DB write FAILED (skipping)",
                        i, len(windows), ws, we,
                    )
                    time.sleep(sleep_seconds)
                    continue
            else:
                written = len(mapped)

            stats.record(mapped)
            logger.info(
                "  [%d/%d] %s..%s: fetched=%d mapped=%d written=%d",
                i, len(windows), ws, we, len(raw), len(mapped), written,
            )

            time.sleep(sleep_seconds)

    finally:
        if conn is not None:
            conn.close()

    return stats


def _print_summary(stats: RunStats, start_date: date, end_date: date) -> None:
    print("\n" + "=" * 72)
    print("  TRANSACTION INGEST — SUMMARY")
    print("=" * 72)
    print(f"  Window:                 {start_date} .. {end_date}")
    print(f"  HTTP windows OK:        {stats.windows_ok}")
    print(f"  HTTP windows failed:    {stats.windows_failed}")
    print(f"  Raw transactions seen:  {stats.raw_fetched:,}")
    print(f"  Rows written (mapped):  {stats.rows_written:,}")
    print(f"  Unique players:         {len(stats.unique_players):,}")
    print(f"  Unique teams observed:  {len(stats.unique_teams):,}")

    print("\n  Per-year counts:")
    for y in sorted(stats.per_year):
        print(f"    {y}   {stats.per_year[y]:>8,d}")

    print("\n  Top 15 typeDesc values:")
    top = sorted(stats.per_type.items(), key=lambda kv: kv[1], reverse=True)[:15]
    for t, c in top:
        print(f"    {t:<34s} {c:>8,d}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill MLB StatsAPI roster transactions into the DuckDB"
                    " transactions table.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Inclusive start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Inclusive end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"DuckDB path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and parse but do not write to the database.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=INTER_REQUEST_SLEEP,
        help=f"Seconds to sleep between HTTP requests "
             f"(default: {INTER_REQUEST_SLEEP}).",
    )
    args = parser.parse_args(argv)

    try:
        start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    except ValueError as exc:
        print(f"ERROR: bad date arg: {exc}", file=sys.stderr)
        return 2

    if end < start:
        print("ERROR: --end-date is before --start-date", file=sys.stderr)
        return 2

    stats = run_ingest(
        start,
        end,
        db_path=Path(args.db),
        dry_run=args.dry_run,
        sleep_seconds=args.sleep,
    )
    _print_summary(stats, start, end)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
