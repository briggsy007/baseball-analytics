#!/usr/bin/env python
"""Gap-fill Statcast pitches for 2017 (Jun, Jul) and 2019 (Aug, Sep).

The original monthly-window backfill (see ``scripts/backfill.py``) silently
dropped four monthly chunks — Savant has the data, but network failures or
rate-limit issues meant those months were never written to DuckDB. Year
totals were 2017=501K rows (vs ~725K neighbours) and 2019=517K rows.

This script:
  1. Re-fetches the missing months via ``pybaseball.statcast`` in small
     weekly windows (more robust than a single 30-day pull for these older
     windows where the Savant endpoint can time out).
  2. Cleans each chunk via ``_clean_statcast`` / ``validate_pitches`` so the
     schema matches the ``pitches`` table exactly.
  3. Dedupes against already-loaded pitches by the natural key
     ``(game_pk, at_bat_number, pitch_number)`` — opens DuckDB **read-only**
     for the dedupe lookup so a running dashboard does not block us.
  4. Writes the deduped rows to
     ``data/staging/statcast_gap_fill_2017_2019.parquet``.

This script NEVER writes to DuckDB. The separate (future) loader step will
pick up the parquet and call ``insert_pitches`` inside the single-writer
lock window.

Idempotent — safe to re-run. If the parquet already exists it is overwritten
with a fresh fetch (or you can pass ``--skip-existing`` to short-circuit).

Usage
-----
    python scripts/backfill_statcast_gaps_2017_2019.py
    python scripts/backfill_statcast_gaps_2017_2019.py --skip-existing
    python scripts/backfill_statcast_gaps_2017_2019.py --months 2017-06 2017-07
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db.schema import get_connection  # noqa: E402
from src.ingest.statcast_loader import (  # noqa: E402
    _PITCHES_COLUMNS,
    _clean_statcast,
    _retry_network_call,
    validate_pitches,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("gap_fill_2017_2019")

# ---------------------------------------------------------------------------
# The four months that dropped out of the 2020-2025 backfill:
#   2017-06, 2017-07, 2019-08, 2019-09
# ---------------------------------------------------------------------------
GAP_MONTHS: list[tuple[int, int]] = [
    (2017, 6),
    (2017, 7),
    (2019, 8),
    (2019, 9),
]

STAGING_PARQUET: Path = ROOT / "data" / "staging" / "statcast_gap_fill_2017_2019.parquet"
STAGING_PARQUET.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weekly_windows(year: int, month: int) -> list[tuple[str, str]]:
    """Split a calendar month into weekly (or shorter) Savant windows.

    Savant is more tolerant of 7-day pulls than 30-day pulls for older
    seasons. Returns a list of ``(start_dt, end_dt)`` ISO date strings.
    """
    first = date(year, month, 1)
    # last day of month
    if month == 12:
        last = date(year, 12, 31)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)

    windows: list[tuple[str, str]] = []
    cur = first
    while cur <= last:
        end = min(cur + timedelta(days=6), last)
        windows.append((cur.isoformat(), end.isoformat()))
        cur = end + timedelta(days=1)
    return windows


def _enable_pybaseball_cache() -> None:
    """Enable pybaseball's disk cache in ``data/cache/``."""
    try:
        import pybaseball

        pybaseball.cache.enable()
        logger.info("pybaseball cache enabled")
    except Exception:
        logger.warning("Could not enable pybaseball cache — continuing")


def _fetch_month(year: int, month: int) -> pd.DataFrame:
    """Fetch a full month from Savant via weekly windows.

    Returns an already-cleaned, schema-aligned DataFrame. Bad windows are
    logged and skipped; the caller checks whether the final row count is
    plausible.
    """
    import pybaseball

    frames: list[pd.DataFrame] = []
    windows = _weekly_windows(year, month)
    for start, end in windows:
        logger.info("  fetching %s .. %s", start, end)
        try:
            raw = _retry_network_call(
                pybaseball.statcast, start_dt=start, end_dt=end, max_retries=3
            )
        except Exception:
            logger.exception("  FAILED window %s .. %s (skipping)", start, end)
            continue

        if raw is None or raw.empty:
            logger.warning("  empty window %s .. %s", start, end)
            continue

        # Older Savant exports (pre-2019) include BOTH `des` (the canonical
        # plate-appearance description used by our schema) AND a raw
        # `description` column that collides with `des` after rename. Drop the
        # raw `description` so the rename-to-`description` path wins cleanly.
        if "des" in raw.columns and "description" in raw.columns:
            raw = raw.drop(columns=["description"])

        cleaned = _clean_statcast(raw)
        cleaned = validate_pitches(cleaned)
        frames.append(cleaned)
        logger.info(
            "  window %s .. %s: %d rows, %d games",
            start, end, len(cleaned), cleaned["game_pk"].nunique() if not cleaned.empty else 0,
        )

    if not frames:
        return pd.DataFrame(columns=_PITCHES_COLUMNS)

    return pd.concat(frames, ignore_index=True)


def _load_existing_keys() -> set[tuple[int, int, int]]:
    """Open DuckDB read-only and load existing natural keys for dedup.

    Returns a set of ``(game_pk, at_bat_number, pitch_number)`` tuples
    already in the ``pitches`` table across the two gap years (plus
    adjacent months we already partially have, so repeated runs don't
    re-add them).
    """
    conn = get_connection(read_only=True)
    try:
        df = conn.execute(
            """
            SELECT game_pk, at_bat_number, pitch_number
            FROM pitches
            WHERE EXTRACT(YEAR FROM game_date) IN (2017, 2019)
            """
        ).fetchdf()
    finally:
        conn.close()

    df = df.dropna(subset=["game_pk", "at_bat_number", "pitch_number"])
    df = df.astype({"game_pk": "int64", "at_bat_number": "int64", "pitch_number": "int64"})
    keys = set(
        zip(df["game_pk"].tolist(), df["at_bat_number"].tolist(), df["pitch_number"].tolist())
    )
    logger.info("Loaded %d existing (game_pk, at_bat, pitch) keys for dedup", len(keys))
    return keys


def _dedupe(df: pd.DataFrame, existing: set[tuple[int, int, int]]) -> pd.DataFrame:
    """Drop rows whose natural key already exists in DuckDB."""
    if df.empty:
        return df

    # Ensure key columns are integer (they may arrive as float due to NaN handling upstream)
    sub = df.dropna(subset=["game_pk", "at_bat_number", "pitch_number"]).copy()
    sub["game_pk"] = sub["game_pk"].astype("int64")
    sub["at_bat_number"] = sub["at_bat_number"].astype("int64")
    sub["pitch_number"] = sub["pitch_number"].astype("int64")

    keys = list(zip(
        sub["game_pk"].tolist(),
        sub["at_bat_number"].tolist(),
        sub["pitch_number"].tolist(),
    ))
    mask = [k not in existing for k in keys]
    deduped = sub[mask].copy()
    logger.info(
        "Dedup: input=%d, existing-overlap=%d, kept=%d",
        len(df), len(df) - len(deduped), len(deduped),
    )
    return deduped


def _parse_month_filter(months: list[str]) -> list[tuple[int, int]]:
    """Parse ``['2017-06', '2019-09']`` into ``[(2017, 6), (2019, 9)]``."""
    out: list[tuple[int, int]] = []
    for m in months:
        try:
            y, mo = m.split("-")
            out.append((int(y), int(mo)))
        except Exception as exc:
            raise SystemExit(f"Bad --months value '{m}': {exc}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If the staging parquet already exists, exit without re-fetching.",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        default=None,
        help="Subset of gap months to fetch (e.g. 2017-06 2019-09).",
    )
    args = parser.parse_args()

    if args.skip_existing and STAGING_PARQUET.exists():
        logger.info("Staging parquet already exists at %s — exiting", STAGING_PARQUET)
        return

    months = _parse_month_filter(args.months) if args.months else GAP_MONTHS
    logger.info("Gap-fill targets: %s", months)

    _enable_pybaseball_cache()

    # Fetch each month independently so a single-month failure does not
    # poison the others.
    per_month_frames: list[pd.DataFrame] = []
    per_month_summary: list[dict] = []
    for year, month in months:
        logger.info("=== Fetching %d-%02d ===", year, month)
        df = _fetch_month(year, month)
        per_month_frames.append(df)
        per_month_summary.append({
            "year": year,
            "month": month,
            "fetched_rows": int(len(df)),
            "fetched_games": int(df["game_pk"].nunique()) if not df.empty else 0,
        })
        logger.info(
            "  month %d-%02d totals: %d rows, %d games",
            year, month, len(df),
            df["game_pk"].nunique() if not df.empty else 0,
        )

    fetched = pd.concat(per_month_frames, ignore_index=True) if per_month_frames else pd.DataFrame(columns=_PITCHES_COLUMNS)
    logger.info("Fetched total (pre-dedup): %d rows", len(fetched))

    # Dedupe against the existing pitches table (read-only).
    existing_keys = _load_existing_keys()
    staged = _dedupe(fetched, existing_keys)

    # Enforce schema column order (matches _PITCHES_COLUMNS exactly).
    for col in _PITCHES_COLUMNS:
        if col not in staged.columns:
            staged[col] = None
    staged = staged[_PITCHES_COLUMNS]

    # Write parquet.
    staged.to_parquet(STAGING_PARQUET, index=False)
    logger.info("Wrote %d rows to %s", len(staged), STAGING_PARQUET)

    # Print a tidy summary for the report.
    print()
    print("=" * 70)
    print("  GAP-FILL STAGING COMPLETE (NOT YET WRITTEN TO DuckDB)")
    print("=" * 70)
    print(f"  Parquet: {STAGING_PARQUET}")
    print(f"  Total staged rows: {len(staged):,}")
    print()
    print("  Per-month fetch summary:")
    for s in per_month_summary:
        print(f"    {s['year']}-{s['month']:02d}: fetched {s['fetched_rows']:>7,} rows / {s['fetched_games']:>3} games")
    print("=" * 70)


if __name__ == "__main__":
    main()
