#!/usr/bin/env python
"""
Initialise the baseball analytics DuckDB database.

Creates the ``data/`` directory (if needed) and all schema tables.
Run from the project root:

    python scripts/setup_db.py
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path so ``src`` is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.schema import DEFAULT_DB_PATH, init_db  # noqa: E402


def main() -> None:
    """Entry-point: create the data directory, initialise the database, and report."""
    data_dir = DEFAULT_DB_PATH.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data directory ready: {data_dir}")

    conn = init_db()

    # List tables that were created / already exist.
    tables = conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchdf()

    print(f"Database initialised at: {DEFAULT_DB_PATH}")
    print(f"Tables ({len(tables)}):")
    for name in sorted(tables["table_name"].tolist()):
        print(f"  - {name}")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
