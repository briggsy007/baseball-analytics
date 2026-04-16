#!/usr/bin/env python
"""
Ingest pitcher injured-list (IL) stints from the ``transactions`` table and
write a labeled dataset to ``data/injury_labels.parquet``.

This feeds MechanixAE Ticket #2 (the supervised label set required for ROC /
AUC validation of the Mechanical Drift Index).  Labels are classified via
keyword matching on the free-text ``description`` column.

Schema of the output parquet (exactly these columns):

    pitcher_id              INTEGER    -- MLB player ID (joins ``pitches.pitcher_id``)
    pitcher_name            VARCHAR    -- for human audit
    season                  INTEGER    -- 2015-2024 inclusive
    il_date                 DATE       -- IL placement date (first day on IL)
    il_end_date             DATE       -- IL return date (nullable)
    injury_type             VARCHAR    -- tommy_john | ucl_sprain | shoulder |
                                         elbow | rotator_cuff | labrum |
                                         forearm | other_arm | non_arm
    injury_description_raw  VARCHAR    -- original transaction description
    source                  VARCHAR    -- 'transactions' or 'manual'

Usage (from project root):

    python scripts/ingest_injury_labels.py
    python scripts/ingest_injury_labels.py --db /path/to/baseball.duckdb
    python scripts/ingest_injury_labels.py --out data/injury_labels.parquet
    python scripts/ingest_injury_labels.py --no-audit     # skip 50-row sample print
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date
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


# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_PATH: Path = PROJECT_ROOT / "data" / "injury_labels.parquet"
START_YEAR = 2015
END_YEAR = 2024

OUTPUT_COLUMNS: list[str] = [
    "pitcher_id",
    "pitcher_name",
    "season",
    "il_date",
    "il_end_date",
    "injury_type",
    "injury_description_raw",
    "source",
]

INJURY_TYPES: tuple[str, ...] = (
    "tommy_john",
    "ucl_sprain",
    "shoulder",
    "elbow",
    "rotator_cuff",
    "labrum",
    "forearm",
    "other_arm",
    "non_arm",
)


# ── Injury classification ──────────────────────────────────────────────────

# Order matters: evaluate more-specific patterns first.
# Each entry: (label, compiled_regex).  Regexes are applied to ``description``
# after lowercasing.
_TJ_PATTERNS = re.compile(
    r"(tommy john|tj surgery|ucl reconstruction|ucl reconstructive|"
    r"ulnar collateral .{0,20}reconstruct)"
)
_UCL_PATTERNS = re.compile(
    r"(ucl (sprain|tear|strain|inflammation)|sprained ucl|"
    r"ulnar collateral (ligament )?(sprain|strain|tear|inflammation))"
)
# Rotator cuff is a shoulder structure but we split it out per the spec.
_ROTATOR_PATTERNS = re.compile(r"(rotator cuff|rotator)")
_LABRUM_PATTERNS = re.compile(r"(labrum|slap tear)")
# Shoulder -- but filter out "shoulder blade" which is the scapula (back, not arm).
_SHOULDER_PATTERNS = re.compile(r"(shoulder)")
_SHOULDER_BLADE_PATTERNS = re.compile(r"(shoulder blade|scapula)")
# Elbow -- catch-all after TJ/UCL.
_ELBOW_PATTERNS = re.compile(r"(elbow)")
# Forearm / flexor bundle.
_FOREARM_PATTERNS = re.compile(r"(forearm|flexor|pronator)")
# Other arm: biceps, triceps, lat (latissimus), pec, generic "arm".
_OTHER_ARM_PATTERNS = re.compile(
    r"(\bbicep|\btricep|\blat\b|latissimus|\bpec\b|pectoral|"
    r"\barm\b|arm strain|arm fatigue|arm inflammation)"
)


def classify_injury(description: Optional[str]) -> str:
    """Return one of ``INJURY_TYPES`` based on keyword matching.

    Order of precedence (most specific first):
        1. tommy_john   -- TJ surgery / UCL reconstruction
        2. ucl_sprain   -- UCL sprain / tear (non-surgical)
        3. rotator_cuff
        4. labrum
        5. shoulder     -- any other shoulder (excluding shoulder-blade)
        6. forearm      -- forearm / flexor / pronator
        7. elbow        -- any other elbow
        8. other_arm    -- biceps / triceps / lat / pec / generic arm
        9. non_arm      -- everything else (knee, hamstring, oblique, COVID,
                           paternity, bereavement, etc.)

    NULL / empty descriptions are classified as ``non_arm``.
    """
    if description is None:
        return "non_arm"
    text = str(description).lower().strip()
    if not text:
        return "non_arm"

    if _TJ_PATTERNS.search(text):
        return "tommy_john"
    if _UCL_PATTERNS.search(text):
        return "ucl_sprain"
    if _ROTATOR_PATTERNS.search(text):
        return "rotator_cuff"
    if _LABRUM_PATTERNS.search(text):
        return "labrum"
    if _SHOULDER_PATTERNS.search(text) and not _SHOULDER_BLADE_PATTERNS.search(text):
        return "shoulder"
    if _FOREARM_PATTERNS.search(text):
        return "forearm"
    if _ELBOW_PATTERNS.search(text):
        return "elbow"
    if _OTHER_ARM_PATTERNS.search(text):
        return "other_arm"
    return "non_arm"


# ── Placement / activation detection ───────────────────────────────────────

_PLACEMENT_PATTERNS = re.compile(
    r"(placed .*? on the .*?injured list|"
    r"placed .*? on the .*?disabled list|"
    r"transferred .*? to the .*?injured list|"
    r"transferred .*? to the .*?disabled list)"
)
_ACTIVATION_PATTERNS = re.compile(
    r"(activated .*? from the .*?injured list|"
    r"activated .*? from the .*?disabled list|"
    r"reinstated .*? from the .*?injured list|"
    r"reinstated .*? from the .*?disabled list)"
)
# Skip lists that are not medical (per-spec "phantom" classes we want to drop
# before even writing labels).  Anything matched here is REMOVED entirely.
_NON_INJURY_LISTS = re.compile(
    r"(paternity list|bereavement list|restricted list|family medical emergency list|"
    r"suspended list|administrative leave)"
)


def is_placement(description: Optional[str]) -> bool:
    """True if the description indicates an IL placement (not an activation)."""
    if description is None:
        return False
    text = str(description).lower()
    if _NON_INJURY_LISTS.search(text):
        return False
    if _ACTIVATION_PATTERNS.search(text):
        return False
    return bool(_PLACEMENT_PATTERNS.search(text))


def is_activation(description: Optional[str]) -> bool:
    """True if the description indicates an IL activation (return from IL)."""
    if description is None:
        return False
    text = str(description).lower()
    if _NON_INJURY_LISTS.search(text):
        return False
    return bool(_ACTIVATION_PATTERNS.search(text))


# ── Pitcher filter ─────────────────────────────────────────────────────────

# The ``players`` table has a ``position`` column but it's NULL for most
# historical rows.  We therefore use presence in the ``pitches`` table as the
# canonical pitcher filter, and fall back to "RHP"/"LHP" substring detection
# in the description for pitchers who never recorded a pitch in our pitches
# table (e.g. minor-league call-ups that were placed on IL before debuting).

_PITCHER_CONTEXT = re.compile(r"\b(RHP|LHP|rhp|lhp|pitcher)\b")


def _fetch_pitcher_ids_from_pitches(conn: duckdb.DuckDBPyConnection) -> set[int]:
    """Return the set of every ``pitcher_id`` that appears in the pitches table."""
    rows = conn.execute("SELECT DISTINCT pitcher_id FROM pitches WHERE pitcher_id IS NOT NULL").fetchall()
    return {int(r[0]) for r in rows}


# ── Core ingestion ─────────────────────────────────────────────────────────


@dataclass
class IngestStats:
    total_tx_rows: int
    il_placements: int
    pitcher_placements: int
    activations_seen: int
    activations_matched: int
    years_covered: tuple[Optional[date], Optional[date]]
    usable_sample_size: int  # placements whose pitcher_id joins to pitches
    per_type: dict[str, int]
    per_year: dict[int, int]
    joinable_arm_injuries: int  # critical award number


def _run_transactions_query(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Fetch every transaction in the 2015-2024 window with a non-empty description.

    Returns a DataFrame with columns:
        transaction_id, player_id, player_name, transaction_type,
        description, transaction_date.
    """
    query = """
        SELECT transaction_id,
               player_id,
               player_name,
               transaction_type,
               description,
               transaction_date
        FROM   transactions
        WHERE  transaction_date BETWEEN ? AND ?
          AND  description IS NOT NULL
          AND  LENGTH(TRIM(description)) > 0
        ORDER  BY transaction_date, transaction_id
    """
    start = f"{START_YEAR}-01-01"
    end = f"{END_YEAR}-12-31"
    return conn.execute(query, [start, end]).fetchdf()


def _match_activations(
    placements: pd.DataFrame,
    activations: pd.DataFrame,
) -> pd.Series:
    """For each placement row, return the first activation date ≥ placement date
    for the same player_id.  Returns a Series aligned to ``placements`` index.

    Uses the earliest activation at-or-after the placement (simple greedy match).
    Unused activations are ignored.  When multiple placements share a player
    (multi-IL seasons), activations are consumed in chronological order.
    """
    if placements.empty:
        return pd.Series([pd.NaT] * 0, dtype="datetime64[ns]")

    # Work on copies sorted chronologically per player.  Drop rows whose
    # player_id is NULL — not every transaction involves a player (e.g. team
    # "Status Change" rows), and we cannot match those to anyone.
    pls = placements.dropna(subset=["player_id"]).sort_values(["player_id", "il_date"]).copy()
    acts = activations.dropna(subset=["player_id"]).sort_values(
        ["player_id", "transaction_date"]
    ).copy()

    il_end: dict[int, pd.Timestamp] = {}

    # Group activations by player for O(P + A) matching.
    acts_by_player: dict[int, list[pd.Timestamp]] = {}
    for pid, grp in acts.groupby("player_id"):
        if pd.isna(pid):
            continue
        acts_by_player[int(pid)] = sorted(pd.to_datetime(grp["transaction_date"]).tolist())

    used_counts: dict[int, int] = {pid: 0 for pid in acts_by_player}

    for idx, row in pls.iterrows():
        raw_pid = row["player_id"]
        if pd.isna(raw_pid):
            # Defensive: dropna above should have removed these, but guard
            # anyway so a schema/dtype change can never crash the script.
            il_end[idx] = pd.NaT
            continue
        pid = int(raw_pid)
        il_d = pd.Timestamp(row["il_date"])
        match: Optional[pd.Timestamp] = None
        if pid in acts_by_player:
            start_from = used_counts[pid]
            for i in range(start_from, len(acts_by_player[pid])):
                cand = acts_by_player[pid][i]
                if cand >= il_d:
                    match = cand
                    used_counts[pid] = i + 1
                    break
        il_end[idx] = match if match is not None else pd.NaT

    # Re-align back to the original placements index (rows that were dropped
    # due to NULL player_id will surface as NaT, same as unmatched rows).
    return pd.Series(il_end).reindex(placements.index)


def build_labels(conn: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, IngestStats]:
    """Build the labeled IL-stints DataFrame and summary statistics.

    Raises:
        RuntimeError: if the transactions table is missing or empty.
    """
    # 1. Sanity-check the table exists and has rows.
    try:
        total_tx = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    except Exception as exc:
        raise RuntimeError(
            "Cannot read `transactions` table: "
            f"{exc}\n\nRemediation: run scripts/ingest_transactions.py first "
            "(see MLB StatsAPI roster-transactions endpoint: "
            "https://statsapi.mlb.com/api/v1/transactions)."
        ) from exc
    if total_tx == 0:
        raise RuntimeError(
            "Transactions table is empty. "
            "Remediation: ingest MLB StatsAPI roster-transactions 2015-01-01 "
            "through 2024-12-31 (endpoint: "
            "https://statsapi.mlb.com/api/v1/transactions?startDate=YYYY-MM-DD&endDate=YYYY-MM-DD). "
            "See scripts/backfill.py for a template."
        )

    # 2. Pull the window.
    tx = _run_transactions_query(conn)
    if tx.empty:
        raise RuntimeError(
            f"No transactions in {START_YEAR}-{END_YEAR}. "
            f"Total rows in table: {total_tx}. "
            "Remediation: backfill historical transactions via MLB StatsAPI "
            "(scripts/ingest_transactions.py needed — does not yet exist)."
        )

    # 3. Pitcher-ID whitelist from the pitches table.
    pitcher_ids = _fetch_pitcher_ids_from_pitches(conn)

    # 4. Split placements vs activations.
    tx["_is_placement"] = tx["description"].apply(is_placement)
    tx["_is_activation"] = tx["description"].apply(is_activation)

    il_placements_df = tx[tx["_is_placement"]].copy()
    activations_df = tx[tx["_is_activation"]].copy()

    # 5. Filter to pitchers only: prefer the pitches-table whitelist, fall
    # back to in-description "RHP"/"LHP"/"pitcher" mentions for players who
    # never threw an MLB pitch.
    def _is_pitcher(row: pd.Series) -> bool:
        pid = row["player_id"]
        if pid is not None and not pd.isna(pid):
            try:
                if int(pid) in pitcher_ids:
                    return True
            except (TypeError, ValueError):
                pass
        desc = row.get("description") or ""
        return bool(_PITCHER_CONTEXT.search(str(desc)))

    if not il_placements_df.empty:
        il_placements_df = il_placements_df[il_placements_df.apply(_is_pitcher, axis=1)].copy()
    if not activations_df.empty:
        activations_df = activations_df[activations_df.apply(_is_pitcher, axis=1)].copy()

    # 6. Build the output frame.
    if il_placements_df.empty:
        labels = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        il_placements_df = il_placements_df.rename(columns={"transaction_date": "il_date"})
        il_placements_df["il_date"] = pd.to_datetime(il_placements_df["il_date"]).dt.date
        il_placements_df["season"] = pd.to_datetime(il_placements_df["il_date"]).dt.year
        il_placements_df["injury_type"] = il_placements_df["description"].apply(classify_injury)

        # 7. Pair activations to placements (nullable il_end_date).
        activations_df = activations_df.rename(columns={"transaction_date": "_act_date"})
        activations_df["_act_date"] = pd.to_datetime(activations_df["_act_date"])
        placements_for_match = il_placements_df[["player_id", "il_date"]].copy()
        placements_for_match["il_date"] = pd.to_datetime(placements_for_match["il_date"])
        activations_for_match = activations_df.rename(columns={"_act_date": "transaction_date"})
        il_end = _match_activations(placements_for_match, activations_for_match)
        il_placements_df["il_end_date"] = il_end.apply(
            lambda v: v.date() if pd.notna(v) else None
        )

        labels = pd.DataFrame(
            {
                "pitcher_id": il_placements_df["player_id"].astype("Int64"),
                "pitcher_name": il_placements_df["player_name"].fillna("").astype(str),
                "season": il_placements_df["season"].astype("Int64"),
                "il_date": il_placements_df["il_date"],
                "il_end_date": il_placements_df["il_end_date"],
                "injury_type": il_placements_df["injury_type"].astype(str),
                "injury_description_raw": il_placements_df["description"].astype(str),
                "source": "transactions",
            }
        )

        # 8. Idempotency: drop exact duplicates (same pitcher, same il_date,
        # same description).
        labels = labels.drop_duplicates(
            subset=["pitcher_id", "il_date", "injury_description_raw"],
            keep="first",
        ).reset_index(drop=True)

    # 9. Compute stats.
    usable_mask = labels["pitcher_id"].apply(
        lambda pid: (not pd.isna(pid)) and int(pid) in pitcher_ids
    ) if not labels.empty else pd.Series([], dtype=bool)
    usable_sample = int(usable_mask.sum()) if not labels.empty else 0
    arm_types = {"tommy_john", "ucl_sprain", "shoulder", "elbow", "rotator_cuff",
                 "labrum", "forearm", "other_arm"}
    joinable_arm = (
        int(((labels["injury_type"].isin(arm_types)) & usable_mask).sum())
        if not labels.empty else 0
    )

    per_type = {t: 0 for t in INJURY_TYPES}
    if not labels.empty:
        vc = labels["injury_type"].value_counts().to_dict()
        for t, c in vc.items():
            per_type[t] = int(c)
    per_year: dict[int, int] = {}
    if not labels.empty:
        per_year = (
            labels["il_date"]
            .apply(lambda d: d.year if d is not None else None)
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .to_dict()
        )

    years_cov: tuple[Optional[date], Optional[date]] = (None, None)
    if not labels.empty:
        dates = [d for d in labels["il_date"] if d is not None]
        if dates:
            years_cov = (min(dates), max(dates))

    stats = IngestStats(
        total_tx_rows=int(total_tx),
        il_placements=int(tx["_is_placement"].sum()),
        pitcher_placements=int(len(labels)),
        activations_seen=int(tx["_is_activation"].sum()),
        activations_matched=int(labels["il_end_date"].notna().sum()) if not labels.empty else 0,
        years_covered=years_cov,
        usable_sample_size=usable_sample,
        per_type=per_type,
        per_year=per_year,
        joinable_arm_injuries=joinable_arm,
    )
    return labels, stats


def write_labels(labels: pd.DataFrame, out_path: Path) -> None:
    """Write the labeled DataFrame to parquet (idempotent -- overwrites cleanly)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Explicit schema so downstream readers get stable types, even for 0 rows.
    schema = pa.schema(
        [
            pa.field("pitcher_id", pa.int64()),
            pa.field("pitcher_name", pa.string()),
            pa.field("season", pa.int64()),
            pa.field("il_date", pa.date32()),
            pa.field("il_end_date", pa.date32()),
            pa.field("injury_type", pa.string()),
            pa.field("injury_description_raw", pa.string()),
            pa.field("source", pa.string()),
        ]
    )

    if labels.empty:
        table = pa.table(
            {
                "pitcher_id": pa.array([], type=pa.int64()),
                "pitcher_name": pa.array([], type=pa.string()),
                "season": pa.array([], type=pa.int64()),
                "il_date": pa.array([], type=pa.date32()),
                "il_end_date": pa.array([], type=pa.date32()),
                "injury_type": pa.array([], type=pa.string()),
                "injury_description_raw": pa.array([], type=pa.string()),
                "source": pa.array([], type=pa.string()),
            },
            schema=schema,
        )
    else:
        # Coerce to pyarrow-compatible types.
        df = labels.copy()
        df["pitcher_id"] = df["pitcher_id"].astype("Int64")
        df["season"] = df["season"].astype("Int64")
        table = pa.Table.from_pandas(df[OUTPUT_COLUMNS], schema=schema, preserve_index=False)

    pq.write_table(table, str(out_path))


# ── Reporting ──────────────────────────────────────────────────────────────


def print_summary(labels: pd.DataFrame, stats: IngestStats, out_path: Path) -> None:
    print("\n" + "=" * 72)
    print("  INJURY LABEL INGESTION — SUMMARY")
    print("=" * 72)
    print(f"  transactions table rows (total):   {stats.total_tx_rows:,}")
    print(f"  IL placements (all positions):     {stats.il_placements:,}")
    print(f"  Pitcher IL placements (output):    {stats.pitcher_placements:,}")
    print(f"  Activations seen:                  {stats.activations_seen:,}")
    print(f"  Activations matched (il_end_date): {stats.activations_matched:,}")

    years_min, years_max = stats.years_covered
    print(f"  Year coverage:                     {years_min} -> {years_max}")
    print(f"\n  Unique pitchers (in label set):    "
          f"{labels['pitcher_id'].nunique() if not labels.empty else 0:,}")
    print(f"  Usable (joins to pitches table):   {stats.usable_sample_size:,}")
    print(f"  Joinable arm injuries [CRITICAL]:  {stats.joinable_arm_injuries:,}")
    print(f"  Output parquet:                    {out_path}")

    print("\n  Distribution by injury_type:")
    for t in INJURY_TYPES:
        c = stats.per_type.get(t, 0)
        print(f"    {t:14s} {c:6,d}")

    print("\n  Distribution by year:")
    for y in range(START_YEAR, END_YEAR + 1):
        c = stats.per_year.get(y, 0)
        flag = "  <-- SUSPICIOUSLY LOW" if 0 < c < 20 else ("  <-- EMPTY" if c == 0 else "")
        print(f"    {y}    {c:6,d}{flag}")


def print_audit_sample(labels: pd.DataFrame, n: int = 50, seed: int = 42) -> None:
    """Print a random sample of rows for manual QA."""
    if labels.empty:
        print("\n  [AUDIT] Label set is empty — nothing to sample.")
        return
    sample_n = min(n, len(labels))
    sample = labels.sample(n=sample_n, random_state=seed).sort_values(["il_date", "pitcher_name"])
    print("\n" + "=" * 72)
    print(f"  AUDIT SAMPLE ({sample_n} rows of {len(labels)}) — spot-check the labels")
    print("=" * 72)
    for _, row in sample.iterrows():
        pid = row["pitcher_id"]
        name = row["pitcher_name"]
        dte = row["il_date"]
        lbl = row["injury_type"]
        desc = str(row["injury_description_raw"])[:120]
        print(f"  [{dte}]  {name:28s} (id={pid})  -> {lbl:12s}  | {desc}")


# ── CLI entry-point ────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    global START_YEAR, END_YEAR  # noqa: PLW0603
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"DuckDB path (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help=f"Output parquet path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--no-audit",
        action="store_true",
        help="Skip the 50-row audit sample print.",
    )
    parser.add_argument(
        "--audit-n",
        type=int,
        default=50,
        help="Size of the random audit sample to print (default: 50).",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        default=None,
        help=f"Inclusive lower bound on season (default: {START_YEAR}).",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help=f"Inclusive upper bound on season (default: {END_YEAR}).",
    )
    args = parser.parse_args(argv)

    # Apply CLI overrides to module-level bounds so downstream helpers pick them up.
    if args.start_year is not None:
        START_YEAR = int(args.start_year)
    if args.end_year is not None:
        END_YEAR = int(args.end_year)

    db_path = Path(args.db)
    out_path = Path(args.out)

    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}", file=sys.stderr)
        return 1

    print(f"Connecting to {db_path} ...")
    conn = duckdb.connect(str(db_path), read_only=True)

    try:
        labels, stats = build_labels(conn)
    except RuntimeError as exc:
        print("\nFATAL: " + str(exc), file=sys.stderr)
        conn.close()
        return 2
    finally:
        # conn may already be closed; ignore double-close errors.
        try:
            conn.close()
        except Exception:
            pass

    write_labels(labels, out_path)
    print_summary(labels, stats, out_path)
    if not args.no_audit:
        print_audit_sample(labels, n=args.audit_n)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
