"""
Tests for the hardened data ingestion pipeline.

Covers:
  - Transaction rollback on bad data
  - Idempotent re-insertion (no duplicate rows)
  - Validation gate catches bad data
  - Data freshness tracking
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import duckdb

from src.db.schema import create_tables
from src.ingest.statcast_loader import (
    _PITCHES_COLUMNS,
    check_data_freshness,
    insert_pitches,
    insert_season_batting,
    insert_season_pitching,
    update_data_freshness,
    validate_pitches,
)


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_pitch_df(n: int = 5, game_pk: int = 999999, **overrides) -> pd.DataFrame:
    """Create a minimal valid pitch DataFrame for testing."""
    rows = []
    for i in range(n):
        row = {col: None for col in _PITCHES_COLUMNS}
        row["game_pk"] = game_pk
        row["game_date"] = "2025-06-15"
        row["pitcher_id"] = 100001
        row["batter_id"] = 200001
        row["pitch_type"] = "FF"
        row["pitch_name"] = "4-Seam Fastball"
        row["release_speed"] = 95.0
        row["release_spin_rate"] = 2300.0
        row["inning"] = 1
        row["at_bat_number"] = 1
        row["pitch_number"] = i + 1
        row["description"] = "called_strike"
        # Apply any per-row overrides
        for k, v in overrides.items():
            row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


@pytest.fixture
def fresh_conn():
    """In-memory DuckDB connection with a clean schema per test."""
    conn = duckdb.connect(":memory:")
    create_tables(conn)
    yield conn
    conn.close()


# ── Test: Validation gate catches bad data ────────────────────────────────


class TestValidationGate:
    """validate_pitches should drop rows with invalid fields."""

    def test_null_game_pk_dropped(self):
        df = _make_pitch_df(3)
        df.loc[0, "game_pk"] = None
        result = validate_pitches(df)
        assert len(result) == 2

    def test_invalid_pitch_type_dropped(self):
        df = _make_pitch_df(4)
        df.loc[1, "pitch_type"] = "ZZ"  # not a real pitch type
        result = validate_pitches(df)
        assert len(result) == 3
        assert "ZZ" not in result["pitch_type"].values

    def test_velocity_out_of_range_dropped(self):
        df = _make_pitch_df(5)
        df.loc[0, "release_speed"] = 150.0  # too fast
        df.loc[1, "release_speed"] = 10.0   # too slow
        result = validate_pitches(df)
        assert len(result) == 3

    def test_null_pitch_type_kept(self):
        """Null pitch_type is acceptable (data may be missing)."""
        df = _make_pitch_df(3)
        df.loc[0, "pitch_type"] = None
        result = validate_pitches(df)
        assert len(result) == 3

    def test_null_velocity_kept(self):
        """Null velocity is acceptable (data may be missing)."""
        df = _make_pitch_df(3)
        df.loc[0, "release_speed"] = None
        result = validate_pitches(df)
        assert len(result) == 3

    def test_empty_df_passthrough(self):
        df = pd.DataFrame(columns=_PITCHES_COLUMNS)
        result = validate_pitches(df)
        assert result.empty

    def test_all_valid_unchanged(self):
        df = _make_pitch_df(5)
        result = validate_pitches(df)
        assert len(result) == 5


# ── Test: Idempotent re-insertion ─────────────────────────────────────────


class TestIdempotentInsertion:
    """Re-running insert_pitches with the same data should not create duplicates."""

    def test_double_insert_no_duplicates(self, fresh_conn):
        df = _make_pitch_df(5)
        first_count = insert_pitches(fresh_conn, df)
        assert first_count == 5

        # Insert the exact same data again
        second_count = insert_pitches(fresh_conn, df)
        assert second_count == 0

        total = fresh_conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
        assert total == 5

    def test_partial_overlap_inserts_only_new(self, fresh_conn):
        df1 = _make_pitch_df(3, game_pk=111111)
        insert_pitches(fresh_conn, df1)

        # Create a second batch that overlaps on the first 3 and adds 2 new
        df2 = _make_pitch_df(5, game_pk=111111)
        second_count = insert_pitches(fresh_conn, df2)
        assert second_count == 2  # only pitch_number 4 and 5 are new

        total = fresh_conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
        assert total == 5

    def test_season_batting_upsert(self, fresh_conn):
        """insert_season_batting replaces existing rows for same (player, season)."""
        df1 = pd.DataFrame({
            "player_id": [100, 101],
            "season": [2025, 2025],
            "pa": [500, 400],
            "ab": [450, 360],
            "h": [130, 100],
        })
        insert_season_batting(fresh_conn, df1)

        row_count = fresh_conn.execute(
            "SELECT COUNT(*) FROM season_batting_stats"
        ).fetchone()[0]
        assert row_count == 2

        # Update with new stats for the same players
        df2 = pd.DataFrame({
            "player_id": [100, 101],
            "season": [2025, 2025],
            "pa": [550, 450],
            "ab": [490, 400],
            "h": [145, 115],
        })
        insert_season_batting(fresh_conn, df2)

        row_count = fresh_conn.execute(
            "SELECT COUNT(*) FROM season_batting_stats"
        ).fetchone()[0]
        assert row_count == 2  # still 2 rows, not 4

        pa_100 = fresh_conn.execute(
            "SELECT pa FROM season_batting_stats WHERE player_id = 100"
        ).fetchone()[0]
        assert pa_100 == 550  # updated value

    def test_season_pitching_upsert(self, fresh_conn):
        """insert_season_pitching replaces existing rows for same (player, season)."""
        df1 = pd.DataFrame({
            "player_id": [200],
            "season": [2025],
            "w": [10],
            "l": [5],
            "era": [3.50],
        })
        insert_season_pitching(fresh_conn, df1)

        df2 = pd.DataFrame({
            "player_id": [200],
            "season": [2025],
            "w": [12],
            "l": [5],
            "era": [3.20],
        })
        insert_season_pitching(fresh_conn, df2)

        row_count = fresh_conn.execute(
            "SELECT COUNT(*) FROM season_pitching_stats"
        ).fetchone()[0]
        assert row_count == 1

        era = fresh_conn.execute(
            "SELECT era FROM season_pitching_stats WHERE player_id = 200"
        ).fetchone()[0]
        assert abs(era - 3.20) < 0.01


# ── Test: Transaction rollback on bad data ────────────────────────────────


class TestTransactionRollback:
    """Database should remain consistent if an insert fails partway through."""

    def test_pitches_rollback_on_schema_violation(self, fresh_conn):
        """If insert_pitches raises, the DB should have no partial data."""
        # Insert some good data first
        good_df = _make_pitch_df(3, game_pk=888888)
        insert_pitches(fresh_conn, good_df)

        initial_count = fresh_conn.execute(
            "SELECT COUNT(*) FROM pitches"
        ).fetchone()[0]
        assert initial_count == 3

        # Now try to insert using a rigged scenario: drop the pitches table
        # mid-transaction to force a failure.  We simulate by corrupting the
        # staging view.
        bad_df = _make_pitch_df(2, game_pk=777777)
        bad_df = bad_df.rename(columns={"game_pk": "WRONG_COL"})

        # The insert should fail because the columns don't match
        with pytest.raises(Exception):
            insert_pitches(fresh_conn, bad_df)

        # Original data should be intact
        after_count = fresh_conn.execute(
            "SELECT COUNT(*) FROM pitches"
        ).fetchone()[0]
        assert after_count == initial_count

    def test_batting_rollback_preserves_old_data(self, fresh_conn):
        """If insert_season_batting fails, old data should remain."""
        df1 = pd.DataFrame({
            "player_id": [300],
            "season": [2025],
            "pa": [400],
        })
        insert_season_batting(fresh_conn, df1)

        # Attempt an insert that will fail (register with wrong schema)
        bad_df = pd.DataFrame({
            "player_id": [300],
            "season": [2025],
            "INVALID_COL": ["not_a_number"],
        })
        with pytest.raises(Exception):
            insert_season_batting(fresh_conn, bad_df)

        # Old data should still be there
        row = fresh_conn.execute(
            "SELECT pa FROM season_batting_stats WHERE player_id = 300"
        ).fetchone()
        assert row is not None
        assert row[0] == 400


# ── Test: Data freshness tracking ─────────────────────────────────────────


class TestDataFreshness:
    """data_freshness table should track ETL watermarks."""

    def test_update_and_check_freshness(self, fresh_conn):
        update_data_freshness(fresh_conn, "pitches", "2025-06-15", 1000)

        # Should report fresh for dates at or before the watermark
        assert check_data_freshness(fresh_conn, "pitches", "2025-06-15") is True
        assert check_data_freshness(fresh_conn, "pitches", "2025-06-14") is True

        # Should report stale for dates after the watermark
        assert check_data_freshness(fresh_conn, "pitches", "2025-06-16") is False

    def test_freshness_update_overwrites(self, fresh_conn):
        update_data_freshness(fresh_conn, "pitches", "2025-06-10", 500)
        assert check_data_freshness(fresh_conn, "pitches", "2025-06-12") is False

        update_data_freshness(fresh_conn, "pitches", "2025-06-15", 1000)
        assert check_data_freshness(fresh_conn, "pitches", "2025-06-12") is True

    def test_freshness_for_unknown_table(self, fresh_conn):
        assert check_data_freshness(fresh_conn, "nonexistent_table", "2025-01-01") is False

    def test_insert_pitches_updates_freshness(self, fresh_conn):
        """Successful pitch insertion should update the freshness watermark."""
        df = _make_pitch_df(3)
        insert_pitches(fresh_conn, df)

        row = fresh_conn.execute(
            "SELECT max_game_date, row_count FROM data_freshness WHERE table_name = 'pitches'"
        ).fetchone()
        assert row is not None
        assert row[1] == 3  # row_count should match inserted count

    def test_freshness_prevents_redundant_work(self, fresh_conn):
        """check_data_freshness should allow skipping already-loaded dates."""
        df = _make_pitch_df(3)
        insert_pitches(fresh_conn, df)

        # The freshness watermark should now cover 2025-06-15
        assert check_data_freshness(fresh_conn, "pitches", "2025-06-15") is True

        # A hypothetical ETL run for 2025-06-15 can be skipped
        # (the daily_etl.py code checks this before loading)
