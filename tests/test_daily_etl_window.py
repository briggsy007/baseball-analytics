"""Regression tests for the gap-aware Statcast ETL window (2026-09-19).

Incident: ``run_daily_etl`` always loaded a single fixed day ("yesterday"),
so when the nightly job missed a run (as it did for 27 game dates between
2026-08-19 and 2026-09-18) those dates were never ingested at all --
permanently, until this fix, since the next run only ever asked for its own
"yesterday" again rather than walking forward from wherever ``pitches``
actually left off.

``pitch_load_window`` is the pure function behind the fix: it computes the
inclusive date range to (re-)load from the DB's own ``MAX(game_date)``
watermark through the target end date, capped at ``max_days`` so a
months-stale DB doesn't trigger an enormous inline pull from a routine
nightly run (that is what ``scripts/backfill_*`` is for).
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.ingest.daily_etl as daily_etl  # noqa: E402
from src.ingest.daily_etl import pitch_load_window  # noqa: E402


# ---------------------------------------------------------------------------
# pitch_load_window (pure)
# ---------------------------------------------------------------------------


def test_normal_one_day_behind_loads_only_end():
    end = date(2026, 9, 18)
    watermark = end - timedelta(days=1)
    assert pitch_load_window(watermark, end) == [end]


def test_five_day_gap_loads_five_days():
    end = date(2026, 9, 18)
    watermark = end - timedelta(days=5)
    window = pitch_load_window(watermark, end)
    assert window == [end - timedelta(days=i) for i in (4, 3, 2, 1, 0)]
    assert len(window) == 5


def test_gap_beyond_max_days_is_capped_ending_at_end():
    end = date(2026, 9, 18)
    watermark = end - timedelta(days=40)
    window = pitch_load_window(watermark, end, max_days=21)
    assert len(window) == 21
    assert window[0] == end - timedelta(days=20)
    assert window[-1] == end


def test_none_watermark_loads_only_end():
    end = date(2026, 9, 18)
    assert pitch_load_window(None, end) == [end]


def test_watermark_equal_to_end_is_empty():
    end = date(2026, 9, 18)
    assert pitch_load_window(end, end) == []


def test_watermark_after_end_is_empty():
    end = date(2026, 9, 18)
    assert pitch_load_window(end + timedelta(days=3), end) == []


# ---------------------------------------------------------------------------
# run_daily_etl integration (in-memory DuckDB, everything else mocked)
# ---------------------------------------------------------------------------


@pytest.fixture()
def requested_dates(monkeypatch):
    """Patch every non-pitches ETL step to a no-op; record loader calls."""
    requested: list[str] = []

    def _fake_load_statcast_range(start_date, end_date):
        assert start_date == end_date
        requested.append(start_date)
        return pd.DataFrame({"a": [1, 2]})

    monkeypatch.setattr(daily_etl, "load_statcast_range", _fake_load_statcast_range)
    monkeypatch.setattr(daily_etl, "insert_pitches", lambda conn, df: len(df))
    monkeypatch.setattr(
        daily_etl, "load_season_batting_stats", lambda year, conn=None: pd.DataFrame()
    )
    monkeypatch.setattr(
        daily_etl, "load_season_pitching_stats", lambda year, conn=None: pd.DataFrame()
    )
    monkeypatch.setattr(
        daily_etl, "load_player_id_map", lambda conn=None: pd.DataFrame()
    )
    monkeypatch.setattr(daily_etl, "_refresh_matchup_cache", lambda conn: None)
    monkeypatch.setattr(daily_etl, "_enable_cache", lambda: None)
    return requested


def test_run_daily_etl_gap_fills_from_watermark_and_sums_pitches(requested_dates):
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE pitches (game_date DATE)")
    end = date(2026, 9, 18)
    watermark_day = end - timedelta(days=4)
    conn.execute("INSERT INTO pitches VALUES (?)", [watermark_day])

    summary = daily_etl.run_daily_etl(conn=conn, target_date=end.isoformat())

    expected_dates = [(end - timedelta(days=i)).isoformat() for i in (3, 2, 1, 0)]
    assert requested_dates == expected_dates
    assert summary["dates_loaded"] == expected_dates
    assert summary["pitches"] == 2 * len(expected_dates)  # 2 rows/day fake df
    assert summary["watermark_before"] == watermark_day.isoformat()
    assert summary["date"] == end.isoformat()


def test_run_daily_etl_target_date_shifts_end(requested_dates):
    conn = duckdb.connect(":memory:")
    conn.execute("CREATE TABLE pitches (game_date DATE)")

    d1 = date(2026, 9, 5)
    summary1 = daily_etl.run_daily_etl(conn=conn, target_date=d1.isoformat())
    assert summary1["date"] == d1.isoformat()
    assert summary1["dates_loaded"] == [d1.isoformat()]

    d2 = date(2026, 9, 12)
    summary2 = daily_etl.run_daily_etl(conn=conn, target_date=d2.isoformat())
    assert summary2["date"] == d2.isoformat()
    assert summary2["dates_loaded"] == [d2.isoformat()]

    assert requested_dates == [d1.isoformat(), d2.isoformat()]
