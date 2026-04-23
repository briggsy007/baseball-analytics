#!/usr/bin/env python
"""
Stage a clean Tommy John (TJ) surgery-date roster to
``data/staging/tj_surgery_dates.parquet``.

Purpose
-------
Unblocks the dormant Projections v2 ``had_tj_surgery`` / ``tj_surgery_date``
feature flag (see NORTH_STAR 5C, "TJ surgery dates 2017-2025" depth priority).
The feature is wired into the projections module but receives NULL for every
pitcher because no clean surgery-date roster exists today.

Scope
-----
- READ-ONLY with respect to DuckDB. Reads ``data/injury_labels.parquet`` plus
  the ``transactions`` and ``pitches`` tables (all read-only).
- STAGING ONLY: writes ``data/staging/tj_surgery_dates.parquet``. Does **not**
  touch DuckDB. Does **not** modify any model code.

Method
------
1. Start from the ``injury_labels.parquet`` rows where
   ``tj_classification_tier`` is populated — i.e. ``explicit_surgical`` or
   ``keyword_adjacent``. The first set is TJ-anchored by explicit surgery
   language; the second is TJ-adjacent (UCL sprain, elbow sprain, elbow
   inflammation, elbow surgery) and requires a follow-up signal before
   admission to the staged roster.
2. Re-scan the ``transactions`` table for any "Tommy John" / "UCL
   reconstruction" / "internal brace" mention we may have missed upstream
   (e.g. position-player UCL repairs the pitcher filter drops).
3. Admission rules by classification tier (NEVER FABRICATE):
     (a) ``explicit_surgical`` -> admit. Extract surgery_date:
         * Explicit "<Month> <YYYY>" reference -> day-1-of-month,
           confidence="high" if ≤ 36mo prior to il_date else "medium".
         * No explicit month, current-event phrasing -> il_date as the
           date, confidence="medium".
         * Recovery phrasing with no month -> surgery_date=NULL,
           confidence="low", source='manual_review_needed'.
     (b) ``keyword_adjacent`` -> admit ONLY if a follow-up signal is
         present (IL duration > 400 days OR a subsequent
         ``explicit_surgical`` transaction from the SAME pitcher). Admitted
         rows are assigned confidence="low" regardless of date extraction
         and are always marked source='manual_review_needed' unless an
         explicit surgery date can be extracted (rare — adjacent-tier
         descriptions rarely include a month/year). Rows without a
         follow-up signal are DROPPED (not fabricated into a TJ event).
4. Dedupe by (player_id, surgery_date ±30d). Multi-season recovery placements
   collapse to one row per surgery event.
5. Infer ``return_date_est`` as the pitcher's first ``pitches.game_date``
   strictly after ``surgery_date`` (MLB return). NULL if no post-surgery MLB
   appearance (still on IL or retired).
6. Stage to ``data/staging/tj_surgery_dates.parquet``.

Output schema
-------------
    mlb_id          BIGINT
    player_name     VARCHAR
    surgery_date    DATE                           (nullable)
    return_date_est DATE                           (nullable)
    confidence      VARCHAR  ('high'|'medium'|'low')
    source          VARCHAR  ('transactions_kw'|'mlb_official'|
                              'retrosheet'|'manual_review_needed')
    notes           VARCHAR

Never fabricate a date. When ambiguous, surgery_date=NULL and
source='manual_review_needed'.

Usage
-----
    python scripts/ingest_tj_dates.py
    python scripts/ingest_tj_dates.py --db /path/to/baseball.duckdb
    python scripts/ingest_tj_dates.py --out data/staging/tj_surgery_dates.parquet
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure the project root is on sys.path so ``src`` is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.schema import DEFAULT_DB_PATH  # noqa: E402


# ── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_tj_dates")


# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_INJURY_LABELS_PATH: Path = PROJECT_ROOT / "data" / "injury_labels.parquet"
DEFAULT_OUTPUT_PATH: Path = PROJECT_ROOT / "data" / "staging" / "tj_surgery_dates.parquet"

# Per the task: focus window 2017-2025 inclusive.
START_YEAR = 2017
END_YEAR = 2025

OUTPUT_COLUMNS: list[str] = [
    "mlb_id",
    "player_name",
    "surgery_date",
    "return_date_est",
    "confidence",
    "source",
    "notes",
]

# Dedupe window — two TJ entries within this many days collapse to one event
# (per task spec: ±30d).
DEDUPE_WINDOW_DAYS = 30

# Confidence enum values.
CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

# Source enum values.
SOURCE_TRANSACTIONS_KW = "transactions_kw"
SOURCE_TRANSACTIONS_KW_ADJACENT = "transactions_kw_adjacent"  # 2026-04-23
SOURCE_MLB_OFFICIAL = "mlb_official"  # reserved for future hand-curation
SOURCE_RETROSHEET = "retrosheet"  # reserved
SOURCE_MANUAL = "manual_review_needed"

# TJ classification tier values from ingest_injury_labels.py.
TJ_TIER_EXPLICIT = "explicit_surgical"
TJ_TIER_ADJACENT = "keyword_adjacent"

# Follow-up signal thresholds for admitting ``keyword_adjacent``-tier rows.
# Either threshold below alone is sufficient; both are tested independently
# per pitcher + il_date.
ADJACENT_IL_DURATION_THRESHOLD_DAYS = 400  # TJ recovery is typically 12-18mo
ADJACENT_FOLLOWUP_WINDOW_DAYS = 730  # 24mo window to look for subsequent
                                    # explicit TJ transaction from same player


# ── Text patterns ─────────────────────────────────────────────────────────

_MONTH_NAMES = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

_MONTH_YEAR_PATTERN = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b",
    re.IGNORECASE,
)

# "Recovering from" / "Recovery from" = surgery is in the past, not at IL date.
_RECOVERY_PATTERN = re.compile(
    r"\b(recovering from|recovery from|rehab(?:ilitating)? from|tommy john surgery rehab)\b",
    re.IGNORECASE,
)

# Anchor patterns that tell us this transaction is unambiguously about a TJ
# (or hybrid UCL-repair / internal-brace) surgery. Consumed both from the
# injury labels parquet and when re-scanning transactions. The
# "elbow reconstruction/repair" phrasing is how MLB.com describes the Bryce
# Harper-style hybrid (UCL repair with internal brace) — we catch it here and
# flag it in notes so downstream consumers can filter it out of classic-TJ
# analyses if desired.
_TJ_ANCHOR = re.compile(
    r"(tommy john|tj surgery|ucl reconstruction|ucl reconstructive|"
    r"ulnar collateral .{0,20}reconstruct|internal brace|"
    r"elbow reconstruction|elbow reconstructive|ucl repair)",
    re.IGNORECASE,
)

# Sub-pattern for "hybrid / UCL-repair / internal-brace" surgeries (not
# classic TJ reconstructions). When this fires and the base _TJ_ANCHOR also
# fires we tag the event with a UCL-REPAIR marker in notes.
_HYBRID_BRACE_PATTERN = re.compile(
    r"(internal brace|ucl repair|elbow reconstruction/repair|elbow repair)",
    re.IGNORECASE,
)


# ── Helpers ───────────────────────────────────────────────────────────────


@dataclass
class StageStats:
    tj_events_staged: int
    by_confidence: dict[str, int]
    by_source: dict[str, int]
    by_year: dict[int, int]
    unique_pitchers: int
    return_dates_inferred: int
    dedupe_collapsed: int


def _extract_month_year(description: str) -> Optional[date]:
    """Return the first ``<Month> <YYYY>`` reference in ``description`` as a
    day-1-of-month ``date``, or ``None`` if no such reference exists.
    """
    if not description:
        return None
    m = _MONTH_YEAR_PATTERN.search(description)
    if not m:
        return None
    month_name = m.group(1).lower()
    year = int(m.group(2))
    month = _MONTH_NAMES.get(month_name)
    if month is None:
        return None
    try:
        return date(year, month, 1)
    except ValueError:
        return None


def _is_recovery_phrasing(description: str) -> bool:
    """True if the description is backward-referential ("Recovering from ...",
    "Recovery from ...", or "Tommy John surgery rehab.").
    """
    if not description:
        return False
    return bool(_RECOVERY_PATTERN.search(description))


def _classify_surgery_date(
    description: str,
    il_date: date,
) -> tuple[Optional[date], str, str, str]:
    """Classify a TJ transaction row into (surgery_date, confidence, source, notes).

    Rules (in precedence order):

    1. Explicit ``<Month> <YYYY>`` reference in the description
       ("Recovering from March 2022 Tommy John surgery.") -> surgery_date
       is day 1 of that month. confidence="high" if the extracted year is
       within 24 months of ``il_date`` (consistent with a realistic TJ
       recovery window); else "medium" (likely a stray typo / transcription
       error; we keep the date but downgrade).

    2. No explicit date, and description is NOT recovery-phrased
       ("Tommy John surgery." without "Recovery from"): the IL placement
       coincides with the surgery event. surgery_date = il_date,
       confidence="medium", notes=...

    3. No explicit date, description IS recovery-phrased, surgery is in the
       recent past but we cannot pin a date: surgery_date=NULL,
       confidence="low", source='manual_review_needed'.

    Never fabricates.
    """
    desc = description or ""
    extracted = _extract_month_year(desc)
    if extracted is not None:
        # Sanity window: TJ recovery is typically 12-18 months, sometimes up to
        # 24 months for revisions. Reject references > 36 months before IL
        # placement (almost certainly not the surgery being recovered-from).
        delta_days = (il_date - extracted).days
        if 0 <= delta_days <= 36 * 31:
            return (
                extracted,
                CONFIDENCE_HIGH,
                SOURCE_TRANSACTIONS_KW,
                f"Extracted month/year from description; il_date={il_date.isoformat()}",
            )
        # Outside the plausible window — downgrade but keep.
        return (
            extracted,
            CONFIDENCE_MEDIUM,
            SOURCE_TRANSACTIONS_KW,
            f"Month/year extracted but {delta_days}d from il_date — review",
        )

    # No extracted month/year.
    if _is_recovery_phrasing(desc):
        # Backward-referential but no date. Cannot pin. Do not fabricate.
        return (
            None,
            CONFIDENCE_LOW,
            SOURCE_MANUAL,
            "Description indicates TJ recovery but no Month/YYYY reference; "
            "surgery date unknown",
        )

    # Current-event phrasing ("Tommy John surgery." at IL placement).
    # CAVEAT: A February/March placement reading "TJ surgery." often reflects
    # a surgery performed in the prior-year off-season (ranging 1–6 months
    # before the IL date). We still default to il_date as the best proxy but
    # flag that it is approximate. Downstream consumers needing true surgery
    # date should cross-reference mlb_official source when available.
    month = il_date.month if isinstance(il_date, date) else pd.Timestamp(il_date).month
    if month in (1, 2, 3):
        return (
            il_date,
            CONFIDENCE_MEDIUM,
            SOURCE_TRANSACTIONS_KW,
            "Early-season IL placement with TJ-anchor text; actual surgery "
            "likely 1–6 months earlier (off-season). il_date is an upper bound",
        )
    return (
        il_date,
        CONFIDENCE_MEDIUM,
        SOURCE_TRANSACTIONS_KW,
        "IL placement coincides with TJ surgery event (no 'Recovery from' phrasing)",
    )


# ── Transactions re-scan ─────────────────────────────────────────────────


def _rescan_transactions_for_tj(
    conn: duckdb.DuckDBPyConnection,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Re-scan the transactions table for any TJ-anchor mention (Tommy John /
    UCL reconstruction / internal brace) in the focus window. Returns a
    DataFrame with columns ``player_id, player_name, transaction_date,
    description``.

    This catches keyword-matched rows the upstream pitcher-filter drops
    (e.g. position-player UCL reconstructions such as Bryce Harper's 2022
    brace). Classification is still pitcher-leaning but we don't filter by
    presence in the pitches table here — we want the broadest net of TJ
    mentions and dedupe downstream.
    """
    query = """
        SELECT player_id,
               player_name,
               transaction_date,
               description
        FROM   transactions
        WHERE  transaction_date BETWEEN ? AND ?
          AND  description IS NOT NULL
          AND  LENGTH(TRIM(description)) > 0
          AND  (
                LOWER(description) LIKE '%tommy john%'
             OR LOWER(description) LIKE '%ucl reconstruction%'
             OR LOWER(description) LIKE '%ucl reconstructive%'
             OR LOWER(description) LIKE '%ucl repair%'
             OR LOWER(description) LIKE '%internal brace%'
             OR LOWER(description) LIKE '%ulnar collateral%reconstruct%'
             OR LOWER(description) LIKE '%elbow reconstruction%'
             OR LOWER(description) LIKE '%elbow reconstructive%'
          )
        ORDER  BY player_id, transaction_date
    """
    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"
    df = conn.execute(query, [start, end]).fetchdf()
    # Filter out 'activated from ... injured list' rows — those are RETURNS,
    # not placements.
    if not df.empty:
        df = df[~df["description"].str.contains(
            r"activated .*? from the", flags=re.IGNORECASE, regex=True, na=False
        )].copy()
    return df


# ── Return-date inference ────────────────────────────────────────────────


def _infer_return_dates(
    conn: duckdb.DuckDBPyConnection,
    events: pd.DataFrame,
) -> pd.Series:
    """For each (mlb_id, surgery_date), return the first ``pitches.game_date``
    strictly after ``surgery_date`` as ``return_date_est``.

    NULL when:
      - surgery_date is NULL (we won't guess a return with no anchor)
      - the pitcher has no post-surgery appearance in ``pitches``
    """
    out = pd.Series([pd.NaT] * len(events), index=events.index, dtype="object")
    # Collect distinct pitcher ids (non-null).
    ids = events["mlb_id"].dropna().astype("int64").unique().tolist()
    if not ids:
        return out

    # Fetch (pitcher_id, game_date) tuples once, sort, index by pitcher.
    placeholders = ",".join(["?"] * len(ids))
    q = f"""
        SELECT pitcher_id, MIN(game_date) AS first_date
        FROM   pitches
        WHERE  pitcher_id IN ({placeholders})
        GROUP  BY pitcher_id, game_date
        ORDER  BY pitcher_id, game_date
    """
    # ^ We actually want ALL (pitcher_id, game_date) pairs, not just min per
    # date. Use a simpler query:
    q = f"""
        SELECT pitcher_id, game_date
        FROM   pitches
        WHERE  pitcher_id IN ({placeholders})
        GROUP  BY pitcher_id, game_date
        ORDER  BY pitcher_id, game_date
    """
    df = conn.execute(q, ids).fetchdf()
    if df.empty:
        return out

    # Index: pitcher_id -> sorted list of game_date (python date objects).
    by_pitcher: dict[int, list[date]] = {}
    for pid, grp in df.groupby("pitcher_id"):
        by_pitcher[int(pid)] = [
            pd.Timestamp(d).date() for d in grp["game_date"].tolist()
        ]

    for idx, row in events.iterrows():
        sd = row["surgery_date"]
        if sd is None or (isinstance(sd, float) and np.isnan(sd)):
            out.loc[idx] = None
            continue
        pid_raw = row["mlb_id"]
        if pd.isna(pid_raw):
            out.loc[idx] = None
            continue
        pid = int(pid_raw)
        dates = by_pitcher.get(pid)
        if not dates:
            out.loc[idx] = None
            continue
        # Linear scan — dates are sorted; lists rarely exceed a few thousand.
        target = sd if isinstance(sd, date) else pd.Timestamp(sd).date()
        found: Optional[date] = None
        for d in dates:
            if d > target:
                found = d
                break
        out.loc[idx] = found
    return out


# ── Dedupe ────────────────────────────────────────────────────────────────


def _dedupe_events(events: pd.DataFrame, window_days: int) -> tuple[pd.DataFrame, int]:
    """Collapse multiple rows for the same pitcher whose surgery_date falls
    within ``window_days`` of each other into a single row. Prefers the
    highest-confidence row; ties broken by earliest surgery_date, then by
    richest description.

    Rows with NULL surgery_date are NOT deduped by date (can't window them).
    They are deduped only on (mlb_id, notes) to avoid exact-duplicate blanks.

    Returns (deduped_df, number_of_rows_collapsed).
    """
    if events.empty:
        return events, 0

    _CONF_RANK = {CONFIDENCE_HIGH: 0, CONFIDENCE_MEDIUM: 1, CONFIDENCE_LOW: 2}

    original_n = len(events)

    # Split: rows with non-null surgery_date vs NULL.
    ev = events.copy()
    ev["_conf_rank"] = ev["confidence"].map(_CONF_RANK).fillna(99).astype(int)
    ev["_desc_len"] = ev["notes"].fillna("").astype(str).str.len()

    has_date = ev[ev["surgery_date"].notna()].copy()
    no_date = ev[ev["surgery_date"].isna()].copy()

    kept_rows: list[pd.Series] = []

    # Dedupe rows with dates by sliding window per pitcher.
    for pid, grp in has_date.groupby("mlb_id"):
        rows = grp.sort_values(
            ["surgery_date", "_conf_rank", "_desc_len"],
            ascending=[True, True, False],
        ).reset_index(drop=True)
        if len(rows) == 1:
            kept_rows.append(rows.iloc[0])
            continue
        # Greedy pass: build clusters where consecutive rows within
        # window_days collapse.
        cluster_start_idx = 0
        cluster_indices = [0]
        for i in range(1, len(rows)):
            prev_date = rows.loc[cluster_indices[-1], "surgery_date"]
            this_date = rows.loc[i, "surgery_date"]
            delta = abs((this_date - prev_date).days)
            if delta <= window_days:
                cluster_indices.append(i)
            else:
                # Close cluster: pick best row.
                best = rows.iloc[cluster_indices].sort_values(
                    ["_conf_rank", "surgery_date", "_desc_len"],
                    ascending=[True, True, False],
                ).iloc[0]
                kept_rows.append(best)
                cluster_indices = [i]
        # Flush final cluster.
        best = rows.iloc[cluster_indices].sort_values(
            ["_conf_rank", "surgery_date", "_desc_len"],
            ascending=[True, True, False],
        ).iloc[0]
        kept_rows.append(best)

    # For NULL-date rows: collapse all NULL-date rows for a given pitcher to
    # a SINGLE row (the earliest by implicit il_date in notes, or the first).
    # Rationale: if we can't pin the surgery date, multiple "Recovery from Tommy
    # John surgery" placements across consecutive seasons describe the SAME
    # event. Keeping N copies inflates the event count.
    # Also: if a pitcher has BOTH a dated row and a null-date row, drop the
    # null-date row (the dated one supersedes).
    if not no_date.empty:
        # First collapse per-pitcher NULL rows to one.
        dated_pitchers = set(has_date["mlb_id"].dropna().astype("int64").tolist())

        def _is_superseded(pid) -> bool:
            if pd.isna(pid):
                return False
            return int(pid) in dated_pitchers

        # Group by mlb_id, keep first row per pitcher (stable order ⇒ the
        # earliest il_date wins because rows are inserted in chronological
        # order upstream).
        # Rows with NULL mlb_id (rare — team-only transaction rows) keep all
        # since we can't group them safely.
        with_pid = no_date[no_date["mlb_id"].notna()].copy()
        without_pid = no_date[no_date["mlb_id"].isna()].copy()
        collapsed_by_pid = with_pid.drop_duplicates(subset=["mlb_id"], keep="first")

        # Drop pitchers who already have a dated row.
        collapsed_by_pid = collapsed_by_pid[
            ~collapsed_by_pid["mlb_id"].apply(_is_superseded)
        ]

        for _, row in collapsed_by_pid.iterrows():
            kept_rows.append(row)
        for _, row in without_pid.iterrows():
            kept_rows.append(row)

    if not kept_rows:
        return events.iloc[0:0].copy(), original_n

    deduped = pd.DataFrame(kept_rows).drop(columns=["_conf_rank", "_desc_len"], errors="ignore")
    deduped = deduped.reset_index(drop=True)
    collapsed = original_n - len(deduped)
    return deduped, collapsed


# ── Main pipeline ────────────────────────────────────────────────────────


def _find_explicit_tj_dates_per_pitcher(
    explicit_df: pd.DataFrame,
) -> dict[int, list[date]]:
    """Return a mapping pitcher_id -> sorted list of explicit-TJ IL dates.

    Used as a follow-up signal gate for ``keyword_adjacent``-tier rows: an
    adjacent row is admitted only if the same pitcher has an explicit-TJ
    transaction within ``ADJACENT_FOLLOWUP_WINDOW_DAYS``.
    """
    out: dict[int, list[date]] = {}
    if explicit_df.empty:
        return out
    for _, row in explicit_df.iterrows():
        pid_raw = row.get("mlb_id")
        if pd.isna(pid_raw):
            continue
        pid = int(pid_raw)
        il_d = row["il_date"]
        if il_d is None or (isinstance(il_d, float) and np.isnan(il_d)):
            continue
        d = il_d if isinstance(il_d, date) else pd.Timestamp(il_d).date()
        out.setdefault(pid, []).append(d)
    for pid in out:
        out[pid] = sorted(out[pid])
    return out


def _adjacent_row_has_followup(
    pid: Optional[int],
    il_date: Optional[date],
    il_end_date: Optional[date],
    explicit_dates_by_pid: dict[int, list[date]],
    aggregate_il_span_by_pid: Optional[dict[int, dict[date, int]]] = None,
) -> tuple[bool, str]:
    """Apply the follow-up-signal gate to a keyword-adjacent-tier row.

    Returns (passed, reason_short). Admitted iff ANY of:

    - Observed single-placement IL duration >= 400 days, OR
    - Aggregated IL span for the same pitcher starting at this row (chaining
      consecutive IL placements within 45 days of each other, including
      30-day IL -> 60-day IL transfers) >= 400 days, OR
    - An explicit-TJ transaction from the same pitcher exists within
      ``ADJACENT_FOLLOWUP_WINDOW_DAYS`` days after this row's il_date.

    Otherwise the row is dropped (NEVER FABRICATE: an elbow sprain or UCL
    sprain alone is not proof of TJ).
    """
    if pid is None or il_date is None:
        return False, "no pid/il_date"
    # Signal 1: single-placement IL duration.
    if il_end_date is not None and isinstance(il_end_date, date) and isinstance(il_date, date):
        duration = (il_end_date - il_date).days
        if duration >= ADJACENT_IL_DURATION_THRESHOLD_DAYS:
            return True, f"IL duration {duration}d >= {ADJACENT_IL_DURATION_THRESHOLD_DAYS}d"

    # Signal 2: aggregated IL span across chained placements.
    if aggregate_il_span_by_pid is not None:
        span = aggregate_il_span_by_pid.get(int(pid), {}).get(il_date)
        if span is not None and span >= ADJACENT_IL_DURATION_THRESHOLD_DAYS:
            return True, f"aggregated IL span {span}d >= {ADJACENT_IL_DURATION_THRESHOLD_DAYS}d"

    # Signal 3: subsequent explicit-TJ transaction from same player.
    dates = explicit_dates_by_pid.get(int(pid), [])
    for d in dates:
        if d <= il_date:
            continue  # Must be SUBSEQUENT, not prior.
        gap = (d - il_date).days
        if gap <= ADJACENT_FOLLOWUP_WINDOW_DAYS:
            return True, (
                f"subsequent explicit TJ {d.isoformat()} "
                f"({gap}d after il_date)"
            )
        break  # sorted; no closer match possible.
    return False, "no follow-up signal"


def _build_aggregate_il_spans(
    adjacent_labels: pd.DataFrame,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
    chain_gap_days: int = 45,
    chain_il_to_il_gap_days: int = 540,
) -> dict[int, dict[date, int]]:
    """For each pitcher, chain consecutive adjacent-tier IL placements into
    a single "recovery window" and score the total span in days. Returns:

        { pitcher_id: { earliest_chain_il_date: total_span_days } }

    Chaining logic (applied in order per pitcher):

    1. If two placements' il_end_date/il_date-to-next-il_date gap is within
       ``chain_gap_days`` (default 45d) — the classic 15-day → 60-day IL
       transfer pattern — they collapse into one chain.
    2. If two adjacent placements sit within ``chain_il_to_il_gap_days``
       (default 540d / 18 months) il_date-to-il_date AND at least one has
       a missing end date (an unmatched activation in our transactions
       pairing), they also chain. This captures deGrom 2023-04-29 (end
       unpaired) → 2024-03-22 (end unpaired) where each placement alone
       fails the 400d gate but the combined span is ~500d.

    3. A chain's ``chain_end`` falls back to the NEXT post-placement
       activation pulled from the transactions table (via ``conn``) when
       an il_end_date is NULL and no subsequent chain-continuing placement
       exists. This catches pitchers whose activation exists in MLB
       transactions but was consumed by an upstream placement during
       greedy pairing.

    The chain span is attributed to EVERY il_date in the chain so the gate
    sees the aggregated number regardless of which row it evaluates.

    A span of zero days is still recorded so that non-chained adjacent rows
    don't silently pass due to dict.get returning None.
    """
    out: dict[int, dict[date, int]] = {}
    if adjacent_labels is None or adjacent_labels.empty:
        return out

    df = adjacent_labels[["pitcher_id", "il_date", "il_end_date"]].copy()
    df = df.dropna(subset=["pitcher_id"])

    def _to_pydate(v):
        if v is None:
            return None
        if isinstance(v, float) and np.isnan(v):
            return None
        if pd.isna(v):
            return None
        if isinstance(v, date):
            return v
        try:
            return pd.Timestamp(v).date()
        except Exception:  # noqa: BLE001
            return None

    df["il_date"] = df["il_date"].apply(_to_pydate)
    df["il_end_date"] = df["il_end_date"].apply(_to_pydate)
    df = df.dropna(subset=["il_date"])

    # Optionally fetch ALL activations per pitcher so chain_end can fall
    # back to the next activation after a NULL il_end_date.
    activations_by_pid: dict[int, list[date]] = {}
    if conn is not None:
        pids = [int(p) for p in df["pitcher_id"].dropna().unique().tolist()]
        if pids:
            placeholders = ",".join(["?"] * len(pids))
            q = f"""
                SELECT player_id, transaction_date
                FROM   transactions
                WHERE  player_id IN ({placeholders})
                  AND  description IS NOT NULL
                  AND  (
                        LOWER(description) LIKE '%activated%from the%injured list%'
                     OR LOWER(description) LIKE '%reinstated%from the%injured list%'
                     OR LOWER(description) LIKE '%activated%from the%disabled list%'
                  )
                ORDER BY player_id, transaction_date
            """
            try:
                adf = conn.execute(q, pids).fetchdf()
                for pid, grp in adf.groupby("player_id"):
                    activations_by_pid[int(pid)] = sorted(
                        pd.Timestamp(d).date() for d in grp["transaction_date"]
                    )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not fetch activations per pid: %s", exc)

    def _first_activation_on_or_after(pid: int, d: date) -> Optional[date]:
        acts = activations_by_pid.get(pid, [])
        for a in acts:
            if a >= d:
                return a
        return None

    for pid, grp in df.groupby("pitcher_id"):
        pid_int = int(pid)
        rows = grp.sort_values("il_date").reset_index(drop=True)
        spans: dict[date, int] = {}
        if rows.empty:
            continue
        chain_start: date = rows.iloc[0]["il_date"]
        chain_end: Optional[date] = rows.iloc[0]["il_end_date"]
        indices_in_chain: list[int] = [0]
        for i in range(1, len(rows)):
            this_start = rows.iloc[i]["il_date"]
            prev_end = chain_end
            this_end = rows.iloc[i]["il_end_date"]
            # Rule 1: consecutive IL placements within 45d of prior END.
            rule_1 = (
                prev_end is not None
                and (this_start - prev_end).days <= chain_gap_days
            )
            # Rule 2: placement-to-placement proximity (≤ chain_il_to_il_gap_days)
            # when at least one end date is NULL (unmatched activation).
            prior_start_or_end = prev_end if prev_end is not None else chain_start
            rule_2 = (
                (prev_end is None or this_end is None)
                and (this_start - prior_start_or_end).days <= chain_il_to_il_gap_days
                and (this_start - prior_start_or_end).days >= 0
            )
            if rule_1 or rule_2:
                indices_in_chain.append(i)
                # Extend chain_end to the LATER of current end and this end.
                candidates = [d for d in (chain_end, this_end) if d is not None]
                chain_end = max(candidates) if candidates else None
            else:
                _flush_chain(
                    pid_int, rows, indices_in_chain, chain_start, chain_end,
                    activations_by_pid, spans,
                )
                chain_start = this_start
                chain_end = this_end
                indices_in_chain = [i]
        _flush_chain(
            pid_int, rows, indices_in_chain, chain_start, chain_end,
            activations_by_pid, spans,
        )
        out[pid_int] = spans
    return out


def _flush_chain(
    pid: int,
    rows: pd.DataFrame,
    indices_in_chain: list[int],
    chain_start: date,
    chain_end: Optional[date],
    activations_by_pid: dict[int, list[date]],
    spans: dict[date, int],
) -> None:
    """Close a recovery-window chain and record its span for every placement
    in the chain. When chain_end is NULL (no paired end date for any
    placement in the chain), fall back to the FIRST activation that occurs
    after the LATEST il_date in the chain. This catches deGrom's case where
    both 2023-04-29 and 2024-03-22 have NULL end dates but 2024-09-13 is an
    activation in the transactions table (> all placements → chain end)."""
    effective_end = chain_end
    if effective_end is None:
        # Pick the latest il_date in the chain and find the first activation
        # strictly after it.
        latest_il_date = chain_start
        for j in indices_in_chain:
            d = rows.iloc[j]["il_date"]
            if d > latest_il_date:
                latest_il_date = d
        acts = activations_by_pid.get(pid, [])
        for a in acts:
            if a > latest_il_date:
                effective_end = a
                break
    if effective_end is None:
        total_span = 0
    else:
        total_span = max(0, (effective_end - chain_start).days)
    for j in indices_in_chain:
        spans[rows.iloc[j]["il_date"]] = total_span


def build_tj_events(
    injury_labels_path: Path,
    conn: duckdb.DuckDBPyConnection,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
) -> tuple[pd.DataFrame, StageStats]:
    """Build the staged TJ-events DataFrame and summary stats.

    Raises:
        FileNotFoundError: if ``data/injury_labels.parquet`` does not exist.
    """
    if not injury_labels_path.exists():
        raise FileNotFoundError(
            f"Injury labels parquet not found at {injury_labels_path}. "
            "Run scripts/ingest_injury_labels.py first."
        )

    # 1. Start from keyword-classified TJ rows. Prefer the new
    # ``tj_classification_tier`` column when present (2026-04-23 classifier
    # gap closure); fall back to ``injury_type == 'tommy_john'`` if the
    # upstream parquet hasn't been regenerated yet.
    labels = pd.read_parquet(injury_labels_path)
    has_tier = "tj_classification_tier" in labels.columns
    season_mask = (labels["season"] >= start_year) & (labels["season"] <= end_year)

    if has_tier:
        explicit_labels = labels[
            season_mask & (labels["tj_classification_tier"] == TJ_TIER_EXPLICIT)
        ].copy()
        adjacent_labels = labels[
            season_mask & (labels["tj_classification_tier"] == TJ_TIER_ADJACENT)
        ].copy()
    else:
        # Legacy / pre-2026-04-23 parquet — behave as before.
        explicit_labels = labels[season_mask & (labels["injury_type"] == "tommy_john")].copy()
        adjacent_labels = labels.iloc[0:0].copy()

    logger.info(
        "Loaded %d explicit-tier and %d adjacent-tier rows from %s (%d-%d window)",
        len(explicit_labels), len(adjacent_labels),
        injury_labels_path.name, start_year, end_year,
    )

    # 2. Re-scan transactions for any TJ-anchor mentions we may have missed.
    #    (Pitcher filter drops position players; re-scan recovers Harper 2022
    #    and similar.)
    rescan = _rescan_transactions_for_tj(conn, start_year, end_year)
    logger.info("Re-scanned transactions: %d TJ-anchor rows", len(rescan))

    # 3a. Build the EXPLICIT-tier union.
    frames: list[pd.DataFrame] = []
    if not explicit_labels.empty:
        frames.append(pd.DataFrame({
            "mlb_id": explicit_labels["pitcher_id"],
            "player_name": explicit_labels["pitcher_name"],
            "il_date": pd.to_datetime(explicit_labels["il_date"]).dt.date,
            "il_end_date": pd.to_datetime(explicit_labels["il_end_date"]).dt.date
                if "il_end_date" in explicit_labels.columns else pd.NaT,
            "description": explicit_labels["injury_description_raw"].astype(str),
            "_tier": TJ_TIER_EXPLICIT,
        }))
    if not rescan.empty:
        frames.append(pd.DataFrame({
            "mlb_id": rescan["player_id"],
            "player_name": rescan["player_name"],
            "il_date": pd.to_datetime(rescan["transaction_date"]).dt.date,
            "il_end_date": pd.NaT,
            "description": rescan["description"].astype(str),
            "_tier": TJ_TIER_EXPLICIT,
        }))
    if not frames:
        explicit_union = pd.DataFrame(
            columns=["mlb_id", "player_name", "il_date", "il_end_date",
                     "description", "_tier"]
        )
    else:
        explicit_union = pd.concat(frames, ignore_index=True)
    explicit_union = explicit_union.drop_duplicates(
        subset=["mlb_id", "il_date", "description"], keep="first"
    ).reset_index(drop=True)
    # Double-check explicit anchor regex fires (defensive).
    explicit_union = explicit_union[
        explicit_union["description"].apply(lambda s: bool(_TJ_ANCHOR.search(s or "")))
    ].reset_index(drop=True)

    # 3b. Build the ADJACENT-tier candidate set and gate it on follow-up
    # signals. DROP rows that fail the gate — never fabricate.
    adjacent_frame = pd.DataFrame(
        columns=["mlb_id", "player_name", "il_date", "il_end_date",
                 "description", "_tier", "_gate_reason"]
    )
    adjacent_admitted_count = 0
    adjacent_dropped_count = 0
    if not adjacent_labels.empty:
        explicit_dates_by_pid = _find_explicit_tj_dates_per_pitcher(explicit_union)
        # Aggregate chains of consecutive adjacent-tier IL placements. deGrom
        # 2023-04 → 2024-09, Strider 2024-04 → 2025-04: single placements
        # each fail the 400d gate, but the chained span clears it.
        aggregate_il_span_by_pid = _build_aggregate_il_spans(
            adjacent_labels, conn=conn,
        )
        adj_rows = []
        for _, row in adjacent_labels.iterrows():
            pid_raw = row.get("pitcher_id")
            pid = None if pd.isna(pid_raw) else int(pid_raw)
            il_d = row.get("il_date")
            il_e = row.get("il_end_date")
            # Coerce to python dates for the gate.
            def _pyd(v):
                if v is None or pd.isna(v):
                    return None
                if isinstance(v, date):
                    return v
                try:
                    return pd.Timestamp(v).date()
                except Exception:  # noqa: BLE001
                    return None
            passed, reason = _adjacent_row_has_followup(
                pid,
                _pyd(il_d),
                _pyd(il_e),
                explicit_dates_by_pid,
                aggregate_il_span_by_pid=aggregate_il_span_by_pid,
            )
            if passed:
                adjacent_admitted_count += 1
                adj_rows.append({
                    "mlb_id": pid_raw,
                    "player_name": row["pitcher_name"],
                    "il_date": il_d,
                    "il_end_date": il_e if not pd.isna(il_e) else None,
                    "description": str(row["injury_description_raw"]),
                    "_tier": TJ_TIER_ADJACENT,
                    "_gate_reason": reason,
                })
            else:
                adjacent_dropped_count += 1
        if adj_rows:
            adjacent_frame = pd.DataFrame(adj_rows)
    logger.info(
        "Adjacent-tier gate: admitted %d, dropped %d (no follow-up signal)",
        adjacent_admitted_count, adjacent_dropped_count,
    )

    # 3c. Union explicit + admitted adjacent rows.
    if explicit_union.empty and adjacent_frame.empty:
        union = pd.DataFrame(columns=["mlb_id", "player_name", "il_date",
                                      "il_end_date", "description", "_tier",
                                      "_gate_reason"])
    else:
        if "_gate_reason" not in explicit_union.columns:
            explicit_union["_gate_reason"] = ""
        if "_gate_reason" not in adjacent_frame.columns:
            adjacent_frame["_gate_reason"] = ""
        union = pd.concat([explicit_union, adjacent_frame], ignore_index=True)

    logger.info("Unioned sources: %d total rows before date extraction", len(union))

    # 5. Extract surgery dates. Adjacent-tier rows are locked to
    # confidence='low' and source='manual_review_needed' unless an explicit
    # surgery date can be extracted from the text (rare).
    records = []
    for _, row in union.iterrows():
        tier = row.get("_tier", TJ_TIER_EXPLICIT)
        sd, conf, src, notes = _classify_surgery_date(row["description"], row["il_date"])
        if tier == TJ_TIER_ADJACENT:
            # Adjacent rows are NEVER high-confidence, regardless of any
            # spurious date extraction. The description language is not
            # explicit surgical text.
            if sd is None:
                conf = CONFIDENCE_LOW
                src = SOURCE_MANUAL
            else:
                # Rare: adjacent row had a "<Month> <YYYY>" reference anyway.
                # Keep the date but cap confidence at 'low' and tag the
                # source explicitly so consumers can filter.
                conf = CONFIDENCE_LOW
                src = SOURCE_TRANSACTIONS_KW_ADJACENT
            notes = (
                f"ADJACENT_TIER (keyword_adjacent, gated); "
                f"gate_signal={row.get('_gate_reason', 'n/a')}; "
                f"{notes}"
            )
        # Flag hybrid UCL-repair / internal-brace surgeries so downstream
        # analyses can distinguish them from classic TJ reconstructions
        # (Bryce Harper 2022, many position-player "UCL repairs").
        hybrid_flag = ""
        if _HYBRID_BRACE_PATTERN.search(row["description"] or ""):
            hybrid_flag = "UCL_REPAIR_BRACE (hybrid, not classic TJ); "
        # Always include the IL date in notes for audit.
        enriched_notes = (
            f"{hybrid_flag}{notes}; "
            f"il_date={row['il_date'].isoformat() if row['il_date'] else 'NA'}; "
            f"desc_excerpt={str(row['description'])[:140]!r}"
        )
        records.append({
            "mlb_id": row["mlb_id"],
            "player_name": row["player_name"],
            "surgery_date": sd,
            "confidence": conf,
            "source": src,
            "notes": enriched_notes,
        })
    events = pd.DataFrame.from_records(records)

    # 6. Dedupe by (mlb_id, surgery_date ±30d).
    events, collapsed = _dedupe_events(events, window_days=DEDUPE_WINDOW_DAYS)
    logger.info("Deduped: collapsed %d rows within %dd window", collapsed, DEDUPE_WINDOW_DAYS)

    # 7. Infer return_date_est from pitches.
    events = events.reset_index(drop=True)
    return_series = _infer_return_dates(conn, events)
    events["return_date_est"] = return_series.values
    inferred_count = int(events["return_date_est"].apply(lambda v: v is not None and not (isinstance(v, float) and np.isnan(v))).sum())

    # 8. Coerce column order and types.
    events = events[OUTPUT_COLUMNS]

    # 9. Stats.
    by_conf: dict[str, int] = events["confidence"].value_counts().to_dict()
    by_src: dict[str, int] = events["source"].value_counts().to_dict()
    by_year: dict[int, int] = {}
    for sd in events["surgery_date"]:
        if sd is not None and not (isinstance(sd, float) and np.isnan(sd)):
            y = sd.year if isinstance(sd, date) else pd.Timestamp(sd).year
            by_year[y] = by_year.get(y, 0) + 1

    stats = StageStats(
        tj_events_staged=len(events),
        by_confidence={k: int(v) for k, v in by_conf.items()},
        by_source={k: int(v) for k, v in by_src.items()},
        by_year=dict(sorted(by_year.items())),
        unique_pitchers=int(events["mlb_id"].dropna().nunique()),
        return_dates_inferred=inferred_count,
        dedupe_collapsed=collapsed,
    )
    return events, stats


# ── Write ────────────────────────────────────────────────────────────────


def write_staging(events: pd.DataFrame, out_path: Path) -> None:
    """Write the staged DataFrame to parquet with an explicit schema."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schema = pa.schema([
        pa.field("mlb_id", pa.int64()),
        pa.field("player_name", pa.string()),
        pa.field("surgery_date", pa.date32()),
        pa.field("return_date_est", pa.date32()),
        pa.field("confidence", pa.string()),
        pa.field("source", pa.string()),
        pa.field("notes", pa.string()),
    ])

    if events.empty:
        table = pa.table(
            {
                "mlb_id": pa.array([], type=pa.int64()),
                "player_name": pa.array([], type=pa.string()),
                "surgery_date": pa.array([], type=pa.date32()),
                "return_date_est": pa.array([], type=pa.date32()),
                "confidence": pa.array([], type=pa.string()),
                "source": pa.array([], type=pa.string()),
                "notes": pa.array([], type=pa.string()),
            },
            schema=schema,
        )
    else:
        df = events.copy()
        # Convert NaN to None explicitly for nullable columns.
        df["mlb_id"] = df["mlb_id"].astype("Int64")

        # surgery_date / return_date_est must be python dates or None.
        def _to_py_date(v):
            if v is None:
                return None
            if isinstance(v, float) and np.isnan(v):
                return None
            if isinstance(v, pd.Timestamp):
                return v.date()
            if isinstance(v, date):
                return v
            try:
                return pd.Timestamp(v).date()
            except Exception:  # noqa: BLE001
                return None

        df["surgery_date"] = df["surgery_date"].apply(_to_py_date)
        df["return_date_est"] = df["return_date_est"].apply(_to_py_date)
        df["player_name"] = df["player_name"].fillna("").astype(str)
        df["confidence"] = df["confidence"].astype(str)
        df["source"] = df["source"].astype(str)
        df["notes"] = df["notes"].fillna("").astype(str)

        table = pa.Table.from_pandas(df[OUTPUT_COLUMNS], schema=schema, preserve_index=False)

    pq.write_table(table, str(out_path))


# ── Reporting ────────────────────────────────────────────────────────────


# Known real-world TJ sanity-check cases (public record). We don't fabricate
# these dates into our parquet — this list is consulted ONLY for reporting
# "did we recover the expected rows?".
SANITY_CHECK_CASES: list[dict] = [
    {
        "name": "Shohei Ohtani",
        "mlb_id": 660271,
        "expected_surgery_date": date(2018, 10, 1),
        "label": "TJ 2018",
        "note": "First TJ October 2018 (public record); transactions text says "
                "'Right elbow UCL sprain' (6/7/2018, 26-day IL) and 'Right "
                "elbow UCL injury' (3/25/2019, 43-day IL). Post-2026-04-23 "
                "classifier: the 6/7 row is tj_classification_tier="
                "'keyword_adjacent'. The follow-up-signal gate requires IL "
                "duration or aggregated span >= 400d OR a subsequent "
                "explicit-TJ transaction from the same player. Ohtani's "
                "chained IL spans total only ~60d and the MLB transactions "
                "table has NO 'Tommy John' mention for him — so the gate "
                "correctly DROPS this row. Expected MISS. Unrecoverable from "
                "the transactions source; an mlb_official / Roegele-CSV pass "
                "would close this case.",
    },
    {
        "name": "Shohei Ohtani",
        "mlb_id": 660271,
        "expected_surgery_date": date(2023, 9, 19),
        "label": "Revision TJ 2023",
        "note": "Second/revision surgery Sept 2023 (public record); transaction "
                "text was 'Oblique' in Sept 2023, with no TJ / UCL-recon label. "
                "Expected MISS — NOT a classifier gap (the injury at time of "
                "IL placement really was oblique; the TJ revision was later).",
    },
    {
        "name": "Jacob deGrom",
        "mlb_id": 594798,
        "expected_surgery_date": date(2023, 6, 12),
        "label": "TJ 2023",
        "note": "TJ June 2023 (public record). Transactions text is "
                "'Right elbow inflammation' then 'Right elbow surgery'. "
                "Post-2026-04-23 classifier: both phrases match the new "
                "elbow-adjacent keyword set and are tagged "
                "tj_classification_tier='keyword_adjacent'. Admitted only if "
                "follow-up-signal gate passes (IL duration > 400d from 2023 IL "
                "clears it — deGrom missed all of 2024). Expected HIT at "
                "confidence=low.",
    },
    {
        "name": "Spencer Strider",
        "mlb_id": 675911,
        "expected_surgery_date": date(2024, 4, 12),
        "label": "Internal-brace 2024",
        "note": "April 2024 internal brace (public record). Transactions text "
                "was 'Right elbow sprain' then 'elbow surgery / internal brace'. "
                "Post-2026-04-23 classifier: 'internal brace' is now in the "
                "explicit-surgical TJ pattern; 'elbow sprain' is keyword_adjacent. "
                "Expected HIT via the internal-brace explicit-tier hit; flagged "
                "UCL_REPAIR_BRACE in notes.",
    },
    {
        "name": "Walker Buehler",
        "mlb_id": 621111,
        "expected_surgery_date": date(2022, 8, 23),
        "label": "Revision TJ 2022",
        "note": "Second/revision TJ Aug 2022 (public record). 2023-02-16 IL "
                "placement in transactions reads 'Right elbow UCL reconstruction' "
                "— a TJ-anchor hit. Expected HIT, surgery_date=~2023-02-16 "
                "(il_date), confidence=medium. True Aug 2022 date requires "
                "mlb_official source.",
    },
    {
        "name": "Bryce Harper",
        "mlb_id": 547180,
        "expected_surgery_date": date(2022, 11, 23),
        "label": "UCL repair w/ brace 2022 (not classic TJ)",
        "note": "UCL REPAIR with internal brace (hybrid, not classic TJ "
                "reconstruction). 2023-03-27 IL text says 'right elbow "
                "reconstruction/repair'. Internal-brace pattern is in our "
                "TJ-anchor regex so this should be captured; row flagged in "
                "notes as UCL-repair/brace, not classic TJ.",
    },
]


def print_summary(events: pd.DataFrame, stats: StageStats, out_path: Path) -> None:
    print("\n" + "=" * 72)
    print("  TJ SURGERY DATE STAGING — SUMMARY")
    print("=" * 72)
    print(f"  Output:                   {out_path}")
    print(f"  Total TJ events staged:   {stats.tj_events_staged:,}")
    print(f"  Unique pitchers:          {stats.unique_pitchers:,}")
    print(f"  Return dates inferred:    {stats.return_dates_inferred:,}")
    print(f"  Dedupe rows collapsed:    {stats.dedupe_collapsed:,}")

    print("\n  Confidence distribution:")
    for conf in (CONFIDENCE_HIGH, CONFIDENCE_MEDIUM, CONFIDENCE_LOW):
        c = stats.by_confidence.get(conf, 0)
        print(f"    {conf:8s} {c:6,d}")

    print("\n  Source distribution:")
    for src, c in sorted(stats.by_source.items(), key=lambda t: -t[1]):
        print(f"    {src:24s} {c:6,d}")

    print("\n  Year distribution (surgery_date):")
    for y in range(START_YEAR, END_YEAR + 1):
        c = stats.by_year.get(y, 0)
        flag = ""
        if c == 0:
            flag = "  <-- EMPTY"
        elif c < 5:
            flag = "  <-- LOW"
        print(f"    {y}   {c:6,d}{flag}")


def print_sanity_checks(events: pd.DataFrame) -> list[dict]:
    """Check a hand-curated list of known real-world TJ cases against the
    staged dataset. Returns a list of dicts summarizing hits/misses.
    """
    print("\n" + "=" * 72)
    print("  SANITY-CHECK KNOWN CASES")
    print("=" * 72)

    results: list[dict] = []
    for case in SANITY_CHECK_CASES:
        mlb_id = case["mlb_id"]
        expected = case["expected_surgery_date"]
        name = case["name"]
        label = case["label"]
        matches = events[events["mlb_id"] == mlb_id]
        if matches.empty:
            status = "MISS"
            detail = f"no staged row for mlb_id={mlb_id}"
        else:
            # Accept any row within 24 months of the expected surgery_date.
            in_window = matches[matches["surgery_date"].apply(
                lambda d: d is not None and not (isinstance(d, float) and np.isnan(d))
                          and abs((d - expected).days) <= 24 * 31
            )]
            if not in_window.empty:
                row = in_window.iloc[0]
                status = "HIT"
                detail = (
                    f"staged surgery_date={row['surgery_date']} "
                    f"confidence={row['confidence']} "
                    f"(expected ~{expected})"
                )
            else:
                # Row exists but outside window.
                row = matches.iloc[0]
                status = "PARTIAL"
                detail = (
                    f"row exists but surgery_date={row['surgery_date']} is "
                    f">24mo from expected {expected}"
                )

        print(f"  [{status:7s}] {name:20s} ({label:30s}) "
              f"mlb_id={mlb_id} -> {detail}")
        print(f"              {case['note']}")
        results.append({
            "name": name,
            "mlb_id": mlb_id,
            "label": label,
            "expected_surgery_date": expected,
            "status": status,
            "detail": detail,
            "note": case["note"],
        })
    return results


# ── CLI entry-point ──────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--db", type=str, default=str(DEFAULT_DB_PATH),
        help=f"DuckDB path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--labels", type=str, default=str(DEFAULT_INJURY_LABELS_PATH),
        help=f"Injury labels parquet path (default: {DEFAULT_INJURY_LABELS_PATH})",
    )
    parser.add_argument(
        "--out", type=str, default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output parquet path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--start-year", type=int, default=START_YEAR,
        help=f"Inclusive lower bound on il_date year (default: {START_YEAR})",
    )
    parser.add_argument(
        "--end-year", type=int, default=END_YEAR,
        help=f"Inclusive upper bound on il_date year (default: {END_YEAR})",
    )
    args = parser.parse_args(argv)

    db_path = Path(args.db)
    labels_path = Path(args.labels)
    out_path = Path(args.out)

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}", file=sys.stderr)
        return 1
    if not labels_path.exists():
        print(f"ERROR: Injury labels parquet not found at {labels_path}", file=sys.stderr)
        return 1

    logger.info("Connecting to %s (read_only=True)", db_path)
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        events, stats = build_tj_events(
            injury_labels_path=labels_path,
            conn=conn,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass

    write_staging(events, out_path)
    print_summary(events, stats, out_path)
    print_sanity_checks(events)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
