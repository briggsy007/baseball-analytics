#!/usr/bin/env python
"""
Backfill 2026 season-to-date Baseball-Reference WAR (bWAR) into
``season_batting_stats.war`` / ``season_pitching_stats.war``.

Why
---
As of 2026-08-03 the DB carries 613 ``season_batting_stats`` rows and 777
``season_pitching_stats`` rows for the 2026 season, but the ``war`` column is
100% NULL on both.  Every downstream consumer that ranks players by "real"
WAR (the Contrarian Leaderboards mid-season board, projections, etc.) is
blocked until 2026 WAR lands.

Design
------
This script is a thin 2026-only wrapper around the *already-tested* pipeline
in ``scripts/backfill_fwar.py``.  It reuses that module's ``fetch_war_for_years``
(pybaseball ``bwar_bat`` / ``bwar_pitch`` with retry + stint aggregation),
``match_to_db_players`` (MLBAM-id crosswalk against ``players``),
``audit_staged`` and the atomic, idempotent ``merge_into_db``.  The only
things that differ are:

* the fetch is pinned to the 2026 season only;
* staging + audit land under ``data/staging/`` with 2026-specific names;
* a ``--dry-run`` flag that fetches, stages and reports the match-rate but
  never touches the DB (used to verify the pipeline without a writer);
* a hard KILL CRITERION: if Baseball-Reference is unreachable OR returns no
  2026 rows, the script exits non-zero with a clear message and NEVER
  substitutes another metric.  NULL stays NULL.

By default (no ``--dry-run``) the script executes the DB write.  The
orchestrator is expected to run the write step; the analytics agent that
authored this file only ever runs ``--dry-run`` (read-only) itself.

Idempotent: running twice produces the same DB state (delegated to
``merge_into_db``).

Examples
--------
    # Verify fetch + crosswalk without writing (safe, read-only)::
    python scripts/backfill_2026_war.py --dry-run

    # Execute the write (orchestrator step, dashboard must be stopped)::
    python scripts/backfill_2026_war.py

    # Re-use an already-staged parquet and only write::
    python scripts/backfill_2026_war.py --skip-fetch
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the battle-tested 2015-2024 backfill machinery verbatim.
from scripts import backfill_fwar as bfwar  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_2026_war")

SEASON = 2026

DB_PATH = ROOT / "data" / "baseball.duckdb"
STAGING_PATH = ROOT / "data" / "staging" / "war_2026_staging.parquet"
UNMATCHED_PATH = ROOT / "data" / "staging" / "war_2026_unmatched.csv"
AUDIT_PATH = ROOT / "data" / "staging" / "war_2026_staging_audit.json"


class KillCriterion(RuntimeError):
    """Raised when the bWAR source is unreachable or has no 2026 data."""


# ---------------------------------------------------------------------------
# Fetch (2026-only) with the kill criterion enforced
# ---------------------------------------------------------------------------

def fetch_2026_war(staging_path: Path, *, skip_fetch: bool) -> pd.DataFrame:
    """Fetch (or reload) the 2026 bWAR staging frame.

    Enforces the KILL CRITERION: unreachable source or zero 2026 rows raises
    :class:`KillCriterion`.  Never fabricates or substitutes a proxy metric.
    """
    if skip_fetch and staging_path.exists():
        logger.info("skip_fetch=True -- reading staged parquet from %s", staging_path)
        staged = pd.read_parquet(staging_path)
    else:
        try:
            staged = bfwar.fetch_war_for_years((SEASON,))
        except Exception as exc:  # network / retry exhaustion / parse failure
            raise KillCriterion(
                f"Baseball-Reference bWAR source is unreachable for {SEASON}: "
                f"{exc!r}. Refusing to substitute another metric -- no DB write."
            ) from exc

    # Defensively restrict to the target season (fetch already filters, but the
    # skip_fetch path could load a wider parquet).
    staged = staged[staged["season"] == SEASON].copy()

    n_nonnull = int(staged["war"].notna().sum()) if not staged.empty else 0
    if staged.empty or n_nonnull == 0:
        raise KillCriterion(
            f"Baseball-Reference returned no {SEASON} bWAR rows "
            f"(rows={len(staged)}, non-null WAR={n_nonnull}). "
            f"B-Ref may not yet publish an in-season {SEASON} daily file. "
            "Refusing to write -- NULL stays NULL."
        )

    if not (skip_fetch and staging_path.exists()):
        staging_path.parent.mkdir(parents=True, exist_ok=True)
        staged.to_parquet(staging_path, index=False)
        logger.info("Wrote staged parquet: %s (%d rows, %d non-null WAR)",
                    staging_path, len(staged), n_nonnull)

    return staged


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(
    *,
    dry_run: bool = False,
    skip_fetch: bool = False,
    db_path: Path = DB_PATH,
    staging_path: Path = STAGING_PATH,
    unmatched_path: Path = UNMATCHED_PATH,
    audit_path: Path = AUDIT_PATH,
) -> dict[str, Any]:
    """Fetch 2026 bWAR, stage, crosswalk, and (unless dry-run) write to the DB."""
    t0 = time.time()

    staged = fetch_2026_war(staging_path, skip_fetch=skip_fetch)

    matched, unmatched = bfwar.match_to_db_players(staged, db_path=db_path)
    logger.info("Matched: %d / Unmatched: %d", len(matched), len(unmatched))
    if len(unmatched):
        unmatched_path.parent.mkdir(parents=True, exist_ok=True)
        unmatched.to_csv(unmatched_path, index=False)
        logger.info("Wrote unmatched audit: %s", unmatched_path)

    audit = bfwar.audit_staged(matched, unmatched)
    audit["season"] = SEASON
    audit["dry_run"] = dry_run
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, default=str)
    logger.info("Wrote audit JSON: %s", audit_path)

    match_rate = audit.get("match_rate")
    logger.info(
        "Match rate: %s (%d matched / %d fetched)",
        f"{match_rate:.4f}" if match_rate is not None else "N/A",
        audit.get("n_matched"), audit.get("n_fetched_total"),
    )

    merge_report: dict[str, Any] = {}
    if dry_run:
        logger.info("DRY-RUN: fetch + stage + crosswalk complete; NO DB write performed.")
    else:
        merge_report = bfwar.merge_into_db(matched, db_path=db_path)
        logger.info(
            "Merge done: batter_delta=%s pitcher_delta=%s",
            merge_report.get("batter_updates_delta"),
            merge_report.get("pitcher_updates_delta"),
        )

    elapsed = time.time() - t0
    logger.info("Total wall-clock: %.1fs", elapsed)
    return {
        "season": SEASON,
        "dry_run": dry_run,
        "audit": audit,
        "merge": merge_report,
        "elapsed_seconds": elapsed,
        "staging_path": str(staging_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Fetch + stage + report match-rate, but do NOT write to the DB.",
    )
    p.add_argument(
        "--skip-fetch", action="store_true",
        help="Reuse the existing staged parquet instead of re-downloading.",
    )
    p.add_argument("--db-path", type=Path, default=DB_PATH)
    p.add_argument("--staging-path", type=Path, default=STAGING_PATH)
    p.add_argument("--unmatched-path", type=Path, default=UNMATCHED_PATH)
    p.add_argument("--audit-path", type=Path, default=AUDIT_PATH)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        run(
            dry_run=args.dry_run,
            skip_fetch=args.skip_fetch,
            db_path=args.db_path,
            staging_path=args.staging_path,
            unmatched_path=args.unmatched_path,
            audit_path=args.audit_path,
        )
        return 0
    except KillCriterion as exc:
        logger.error("KILL CRITERION TRIPPED: %s", exc)
        return 2
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("backfill_2026_war failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
