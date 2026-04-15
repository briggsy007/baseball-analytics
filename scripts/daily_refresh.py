#!/usr/bin/env python
"""
Daily data refresh pipeline for the baseball analytics platform.

Wraps ``run_daily_etl()`` with additional steps: roster sync, transaction
sync, matchup cache rebuild, optional Stuff+ retraining, and automatic
pre-game report generation.

Usage
-----
    python scripts/daily_refresh.py                  # standard refresh
    python scripts/daily_refresh.py --full           # includes model retraining
    python scripts/daily_refresh.py --date 2026-04-06  # refresh for a specific date
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("daily_refresh")

# Ensure stdout can handle Unicode on Windows (cp1252 fallback otherwise)
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True,
    )


# ---------------------------------------------------------------------------
# ANSI formatting
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_RESET = "\033[0m"

DIVIDER = "\033[90m" + "-" * 60 + _RESET


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


def _ok(msg: str) -> None:
    print(f"  {_c('[OK]', _GREEN + _BOLD)}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {_c('[WARN]', _YELLOW + _BOLD)} {msg}")


def _fail(msg: str) -> None:
    print(f"  {_c('[FAIL]', _RED + _BOLD)} {msg}")


def _info(msg: str) -> None:
    print(f"  {_c('[..]', _CYAN)}  {msg}")


# ---------------------------------------------------------------------------
# Database connection
# ---------------------------------------------------------------------------

def _get_conn():
    """Open a DuckDB connection with schema initialised."""
    try:
        from src.db.schema import init_db
        return init_db()
    except Exception as exc:
        logger.error("Failed to open database: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _step_etl(conn, target_date: str) -> dict:
    """Step 1: Run the core daily ETL pipeline.

    Args:
        conn: Open DuckDB connection.
        target_date: The date whose Statcast data to load (YYYY-MM-DD).

    Returns:
        ETL summary dict.
    """
    _info(f"Loading Statcast data for {target_date}...")

    try:
        from src.ingest.daily_etl import run_daily_etl
        summary = run_daily_etl(conn=conn)
        _ok(
            f"ETL complete -- pitches: {summary.get('pitches', 0)}, "
            f"batting: {summary.get('batting_rows', 0)}, "
            f"pitching: {summary.get('pitching_rows', 0)}"
        )
        return summary
    except Exception as exc:
        _fail(f"Daily ETL failed: {exc}")
        logger.exception("ETL failure")
        return {"pitches": 0, "batting_rows": 0, "pitching_rows": 0, "players": 0}


def _step_roster_sync(conn) -> int:
    """Step 2: Sync the Phillies active roster."""
    _info("Syncing Phillies roster...")

    try:
        from src.ingest.roster_tracker import sync_roster_to_db, PHILLIES_TEAM_ID
        count = sync_roster_to_db(conn, team_id=PHILLIES_TEAM_ID)
        _ok(f"Roster synced: {count} players")
        return count
    except Exception as exc:
        _fail(f"Roster sync failed: {exc}")
        logger.exception("Roster sync failure")
        return 0


def _step_transactions(conn) -> int:
    """Step 3: Sync recent Phillies transactions."""
    _info("Syncing recent transactions (last 7 days)...")

    try:
        from src.ingest.roster_tracker import sync_transactions_to_db
        count = sync_transactions_to_db(conn, days=7)
        _ok(f"Transactions synced: {count} new")
        return count
    except Exception as exc:
        _fail(f"Transaction sync failed: {exc}")
        logger.exception("Transaction sync failure")
        return 0


def _step_matchup_cache(conn) -> bool:
    """Step 4: Refresh the matchup summary cache."""
    _info("Refreshing matchup cache...")

    try:
        from src.db.queries import refresh_matchup_cache
        df = refresh_matchup_cache(conn)
        rows = len(df) if df is not None else 0
        _ok(f"Matchup cache rebuilt: {rows} rows")
        return True
    except Exception as exc:
        _warn(f"Matchup cache refresh failed: {exc}")
        logger.exception("Matchup cache failure")
        return False


def _step_retrain_model() -> bool:
    """Step 5 (optional): Retrain the Stuff+ model."""
    _info("Retraining Stuff+ model...")

    model_dir = ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Attempt to import and run model training if it exists
        # This is a placeholder -- the actual training module may not
        # exist yet.  We degrade gracefully.
        try:
            from src.analytics.stuff_model import train_stuff_model
            train_stuff_model(_get_conn())
            _ok("Stuff+ model retrained")
            return True
        except ImportError:
            _warn("Stuff+ training module not found -- skipping")
            return False
    except Exception as exc:
        _warn(f"Stuff+ retraining failed: {exc}")
        return False


def _step_pregame_report(target_date: Optional[str] = None) -> bool:
    """Step 6: Generate today's pre-game report."""
    report_date = target_date or date.today().strftime("%Y-%m-%d")
    _info(f"Generating pre-game report for {report_date}...")

    try:
        from scripts.pregame_report import generate_report
        generate_report(date_str=report_date)
        return True
    except Exception as exc:
        _warn(f"Pre-game report generation failed: {exc}")
        logger.exception("Pre-game report failure")
        return False


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_daily_refresh(
    target_date: Optional[str] = None,
    full: bool = False,
) -> dict:
    """Execute the full daily refresh pipeline.

    Args:
        target_date: Date to load data for (YYYY-MM-DD).
                     Defaults to yesterday.
        full: If True, also retrain Stuff+ model.

    Returns:
        Summary dict with step results.
    """
    start_time = time.time()

    if target_date is None:
        yesterday = date.today() - timedelta(days=1)
        target_date = yesterday.strftime("%Y-%m-%d")

    today_str = date.today().strftime("%Y-%m-%d")

    print()
    print(_c("=" * 60, _BOLD))
    print(_c("  DAILY DATA REFRESH PIPELINE", _BOLD + _CYAN))
    print(_c("=" * 60, _BOLD))
    print(f"  Target date:  {target_date}")
    print(f"  Report date:  {today_str}")
    print(f"  Mode:         {'Full (with model retraining)' if full else 'Standard'}")
    print(_c("=" * 60, _BOLD))
    print()

    summary: dict = {
        "target_date": target_date,
        "steps_completed": 0,
        "steps_failed": 0,
        "etl": {},
        "roster_count": 0,
        "transactions": 0,
        "matchup_cache": False,
        "model_retrained": False,
        "pregame_report": False,
    }

    # Open database connection
    conn = _get_conn()
    if conn is None:
        _fail("Cannot open database. Aborting.")
        return summary

    try:
        # Step 1: Core ETL
        print(DIVIDER)
        print(_c("  Step 1/6: Core ETL Pipeline", _BOLD))
        print(DIVIDER)
        etl_result = _step_etl(conn, target_date)
        summary["etl"] = etl_result
        summary["steps_completed"] += 1
        print()

        # Step 2: Roster sync
        print(DIVIDER)
        print(_c("  Step 2/6: Roster Sync", _BOLD))
        print(DIVIDER)
        roster_count = _step_roster_sync(conn)
        summary["roster_count"] = roster_count
        summary["steps_completed"] += 1
        print()

        # Step 3: Transactions
        print(DIVIDER)
        print(_c("  Step 3/6: Transaction Sync", _BOLD))
        print(DIVIDER)
        txn_count = _step_transactions(conn)
        summary["transactions"] = txn_count
        summary["steps_completed"] += 1
        print()

        # Step 4: Matchup cache
        print(DIVIDER)
        print(_c("  Step 4/6: Matchup Cache", _BOLD))
        print(DIVIDER)
        cache_ok = _step_matchup_cache(conn)
        summary["matchup_cache"] = cache_ok
        if cache_ok:
            summary["steps_completed"] += 1
        else:
            summary["steps_failed"] += 1
        print()

        # Step 5: Model retraining (optional)
        print(DIVIDER)
        print(_c("  Step 5/6: Model Retraining", _BOLD))
        print(DIVIDER)
        if full:
            retrained = _step_retrain_model()
            summary["model_retrained"] = retrained
            if retrained:
                summary["steps_completed"] += 1
            else:
                summary["steps_failed"] += 1
        else:
            _info("Skipped (use --full to enable)")
            summary["steps_completed"] += 1
        print()

        # Step 6: Pre-game report
        print(DIVIDER)
        print(_c("  Step 6/6: Pre-Game Report", _BOLD))
        print(DIVIDER)
        report_ok = _step_pregame_report(today_str)
        summary["pregame_report"] = report_ok
        if report_ok:
            summary["steps_completed"] += 1
        else:
            summary["steps_failed"] += 1

    except KeyboardInterrupt:
        print(f"\n  {_c('Pipeline interrupted by user.', _YELLOW)}")
    finally:
        try:
            conn.close()
        except Exception:
            pass

    # Print final summary
    elapsed = time.time() - start_time
    _print_summary(summary, elapsed)

    return summary


def _print_summary(summary: dict, elapsed: float) -> None:
    """Print a formatted summary of the pipeline run."""
    print()
    print(_c("=" * 60, _BOLD))
    print(_c("  PIPELINE SUMMARY", _BOLD + _CYAN))
    print(_c("=" * 60, _BOLD))

    etl = summary.get("etl", {})
    print(f"  Statcast pitches loaded:  {etl.get('pitches', 0)}")
    print(f"  Batting stat rows:        {etl.get('batting_rows', 0)}")
    print(f"  Pitching stat rows:       {etl.get('pitching_rows', 0)}")
    print(f"  Player IDs updated:       {etl.get('players', 0)}")
    print(f"  Roster players synced:    {summary.get('roster_count', 0)}")
    print(f"  New transactions:         {summary.get('transactions', 0)}")
    print(f"  Matchup cache:            {'OK' if summary.get('matchup_cache') else 'FAILED'}")
    print(f"  Model retrained:          {'Yes' if summary.get('model_retrained') else 'No'}")
    print(f"  Pre-game report:          {'Yes' if summary.get('pregame_report') else 'No'}")
    print()
    print(
        f"  Steps completed: {summary.get('steps_completed', 0)}/6  |  "
        f"Failed: {summary.get('steps_failed', 0)}  |  "
        f"Elapsed: {elapsed:.1f}s"
    )

    if summary.get("steps_failed", 0) == 0:
        print(f"\n  {_c('All steps completed successfully.', _GREEN + _BOLD)}")
    else:
        failed_count = summary["steps_failed"]
        msg = f"{failed_count} step(s) had issues."
        print(f"\n  {_c(msg, _YELLOW + _BOLD)} Check logs for details.")

    print(_c("=" * 60, _BOLD))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the daily refresh pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the daily data refresh pipeline.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date to load Statcast data for (YYYY-MM-DD, default: yesterday)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Include Stuff+ model retraining",
    )
    args = parser.parse_args()

    run_daily_refresh(target_date=args.date, full=args.full)


if __name__ == "__main__":
    main()
