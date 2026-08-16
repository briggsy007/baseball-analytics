"""Regression tests for the ``matchup_summary`` rebuild (2026-08-16).

Context: the rebuild used to be a transaction-wrapped DELETE + INSERT against a
table carrying ``PRIMARY KEY (pitcher_id, batter_id, pitch_type)``. Committing
that against the PK's ART index hard-aborted the process with a Windows
fast-fail (0xC0000409) and raised nothing Python could catch, so the rebuild
rolled back silently and the cache drifted 923,797 pitches stale while every
caller believed it had refreshed.

Two defects are pinned here:
  1. the rebuild must actually land and leave the cache in sync, and
  2. there must be exactly ONE rebuild implementation -- ``daily_etl`` used to
     carry a second copy whose BA/SLG denominators disagreed with the
     dashboard-facing one.

These tests build their own temporary DuckDB file. They never open
``data/baseball.duckdb``.
"""

from __future__ import annotations

import duckdb
import pytest

from src.db.queries import refresh_matchup_cache
from src.db.schema import _create_matchup_summary

# (pitcher_id, batter_id, pitch_type, release_speed, release_spin_rate,
#  pfx_x, pfx_z, description, events, woba_value, woba_denom)
_ROWS = [
    # Pitcher 1 vs batter 10, four-seam: 3 pitches, one a swinging strike, one a single.
    (1, 10, "FF", 95.0, 2200.0, -0.5, 1.5, "swinging_strike", None, 0.0, 0),
    (1, 10, "FF", 96.0, 2250.0, -0.4, 1.6, "called_strike", None, 0.0, 0),
    (1, 10, "FF", 94.0, 2150.0, -0.6, 1.4, "hit_into_play", "single", 0.9, 1),
    # Same pitcher/batter, different pitch type -> its own row.
    (1, 10, "SL", 85.0, 2400.0, 0.8, 0.2, "swinging_strike", None, 0.0, 0),
    # Different batter -> its own row.
    (1, 11, "FF", 97.0, 2300.0, -0.3, 1.7, "hit_into_play", "home_run", 2.0, 1),
    # NULL pitch_type must be excluded from the cache entirely.
    (1, 10, None, 90.0, 2000.0, 0.0, 1.0, "ball", None, 0.0, 0),
]


def _seed(db_path) -> duckdb.DuckDBPyConnection:
    """Create a temp DB with a minimal ``pitches`` table plus the cache table."""
    conn = duckdb.connect(str(db_path), read_only=False)
    conn.execute(
        """
        CREATE TABLE pitches (
            pitcher_id        INTEGER,
            batter_id         INTEGER,
            pitch_type        VARCHAR,
            release_speed     FLOAT,
            release_spin_rate FLOAT,
            pfx_x             FLOAT,
            pfx_z             FLOAT,
            description       VARCHAR,
            events            VARCHAR,
            woba_value        FLOAT,
            woba_denom        INTEGER
        )
        """
    )
    conn.executemany(
        "INSERT INTO pitches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", _ROWS
    )
    _create_matchup_summary(conn)
    return conn


@pytest.fixture()
def seeded(tmp_path):
    conn = _seed(tmp_path / "matchup_test.duckdb")
    yield conn
    conn.close()


def _sync_state(conn) -> tuple[int, int]:
    cached = conn.execute("SELECT SUM(num_pitches) FROM matchup_summary").fetchone()[0]
    eligible = conn.execute(
        "SELECT COUNT(*) FROM pitches WHERE pitch_type IS NOT NULL"
    ).fetchone()[0]
    return cached, eligible


def test_rebuild_leaves_cache_in_sync(seeded):
    """The invariant the nightly effect-check now asserts in production."""
    rows = refresh_matchup_cache(seeded)

    cached, eligible = _sync_state(seeded)
    assert cached == eligible == 5, "cache must cover every pitch_type-bearing pitch"
    assert rows == 3, "expected one row per (pitcher, batter, pitch_type) group"


def test_rebuild_is_idempotent(seeded):
    """A nightly runs this every night; twice must equal once."""
    first_rows = refresh_matchup_cache(seeded)
    first_state = _sync_state(seeded)
    first_dump = seeded.execute(
        "SELECT * FROM matchup_summary ORDER BY pitcher_id, batter_id, pitch_type"
    ).fetchall()

    second_rows = refresh_matchup_cache(seeded)

    assert second_rows == first_rows
    assert _sync_state(seeded) == first_state
    assert (
        seeded.execute(
            "SELECT * FROM matchup_summary ORDER BY pitcher_id, batter_id, pitch_type"
        ).fetchall()
        == first_dump
    )


def test_null_pitch_type_excluded_and_grouping_is_correct(seeded):
    """A naive implementation would double-count or admit the NULL row."""
    refresh_matchup_cache(seeded)

    assert (
        seeded.execute(
            "SELECT COUNT(*) FROM matchup_summary WHERE pitch_type IS NULL"
        ).fetchone()[0]
        == 0
    )
    ff = seeded.execute(
        "SELECT num_pitches FROM matchup_summary "
        "WHERE pitcher_id = 1 AND batter_id = 10 AND pitch_type = 'FF'"
    ).fetchone()
    assert ff[0] == 3, "the three FF pitches must collapse into one row of 3"


def test_rebuild_survives_a_preexisting_primary_key(tmp_path):
    """The live table still carried the old PK -- the crash mechanism itself.

    ``schema.py`` governs table CREATION only, so fixing the DDL does not fix an
    existing database. The rebuild must replace the table wholesale, dropping the
    PK along with it, or the fix works only on fresh installs.
    """
    conn = duckdb.connect(str(tmp_path / "pk.duckdb"), read_only=False)
    conn.execute(
        """
        CREATE TABLE pitches (
            pitcher_id        INTEGER, batter_id         INTEGER,
            pitch_type        VARCHAR, release_speed     FLOAT,
            release_spin_rate FLOAT,   pfx_x             FLOAT,
            pfx_z             FLOAT,   description       VARCHAR,
            events            VARCHAR, woba_value        FLOAT,
            woba_denom        INTEGER
        )
        """
    )
    conn.executemany(
        "INSERT INTO pitches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", _ROWS
    )
    # The OLD DDL, PK and all.
    conn.execute(
        """
        CREATE TABLE matchup_summary (
            pitcher_id INTEGER, batter_id INTEGER, pitch_type VARCHAR,
            num_pitches INTEGER, avg_speed FLOAT, avg_spin FLOAT,
            avg_pfx_x FLOAT, avg_pfx_z FLOAT, whiff_rate FLOAT,
            ba FLOAT, slg FLOAT, woba FLOAT,
            PRIMARY KEY (pitcher_id, batter_id, pitch_type)
        )
        """
    )

    refresh_matchup_cache(conn)

    cached, eligible = _sync_state(conn)
    assert cached == eligible
    pk_count = conn.execute(
        "SELECT COUNT(*) FROM duckdb_constraints() "
        "WHERE table_name = 'matchup_summary' AND constraint_type = 'PRIMARY KEY'"
    ).fetchone()[0]
    assert pk_count == 0, "rebuild must migrate the table off its PRIMARY KEY"
    conn.close()


def test_daily_etl_delegates_to_the_single_implementation(seeded, monkeypatch):
    """Regression guard against the two-divergent-copies root cause."""
    from src.ingest import daily_etl

    calls = []

    def _spy(conn):
        calls.append(conn)
        return 0

    monkeypatch.setattr("src.db.queries.refresh_matchup_cache", _spy)
    daily_etl._refresh_matchup_cache(seeded)

    assert len(calls) == 1, (
        "daily_etl must delegate to src.db.queries.refresh_matchup_cache rather "
        "than carry its own copy of the aggregation"
    )


def test_missing_table_is_tolerated(tmp_path):
    """A fresh DB without the cache table must not crash the ETL."""
    from src.ingest import daily_etl

    conn = duckdb.connect(str(tmp_path / "notable.duckdb"), read_only=False)
    conn.execute("CREATE TABLE pitches (pitcher_id INTEGER, pitch_type VARCHAR)")
    conn.execute("INSERT INTO pitches VALUES (1, 'FF')")

    # Must swallow the missing-table condition, not raise.
    daily_etl._refresh_matchup_cache(conn)
    conn.close()
