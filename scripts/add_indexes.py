#!/usr/bin/env python
"""
Add performance indexes to an existing pitches table.

This is a one-time migration script intended for environments where the
DuckDB database already exists (e.g. the Hetzner VPS).  Every statement
uses CREATE INDEX IF NOT EXISTS, so it is safe to run repeatedly.

Usage (from the project root):

    python scripts/add_indexes.py
    python scripts/add_indexes.py --db /path/to/baseball.duckdb
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path so ``src`` is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.schema import DEFAULT_DB_PATH, PITCHES_INDEXES  # noqa: E402

import duckdb  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Add indexes to the pitches table.")
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help=f"Path to the DuckDB file (default: {DEFAULT_DB_PATH})",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        sys.exit(1)

    print(f"Connecting to {db_path} ...")
    conn = duckdb.connect(str(db_path))

    # Quick sanity check: does the pitches table exist?
    tables = [
        row[0]
        for row in conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
    ]
    if "pitches" not in tables:
        print("ERROR: 'pitches' table does not exist in this database.")
        conn.close()
        sys.exit(1)

    row_count = conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
    print(f"pitches table has {row_count:,} rows\n")

    total_start = time.perf_counter()
    success_count = 0
    fail_count = 0

    for stmt in PITCHES_INDEXES:
        # Extract a friendly index name from the SQL for logging.
        idx_name = stmt.split("idx_")[1].split(" ")[0] if "idx_" in stmt else stmt[:60]
        idx_name = f"idx_{idx_name}"

        print(f"  Creating {idx_name} ...", end=" ", flush=True)
        start = time.perf_counter()
        try:
            conn.execute(stmt)
            elapsed = time.perf_counter() - start
            print(f"OK  ({elapsed:.2f}s)")
            success_count += 1
        except Exception as exc:
            elapsed = time.perf_counter() - start
            print(f"FAILED  ({elapsed:.2f}s) — {exc}")
            fail_count += 1

    total_elapsed = time.perf_counter() - total_start
    print(f"\nDone in {total_elapsed:.2f}s — {success_count} succeeded, {fail_count} failed.")

    # Report final index state.
    indexes = conn.execute(
        "SELECT index_name, table_name FROM duckdb_indexes() ORDER BY table_name, index_name"
    ).fetchall()
    print(f"\nAll indexes ({len(indexes)}):")
    for idx_name, tbl_name in indexes:
        print(f"  - {tbl_name}.{idx_name}")

    conn.close()


if __name__ == "__main__":
    main()
