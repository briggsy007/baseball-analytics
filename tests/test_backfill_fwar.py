"""
Unit tests for ``scripts/backfill_fwar.py``.

These tests are fully offline: ``pybaseball`` is monkey-patched with a
synthetic fixture so no network traffic happens in CI.  Merges run against an
in-memory DuckDB with the full schema via ``src.db.schema.create_tables``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.db.schema import create_tables  # noqa: E402

# Import the module under test.
import scripts.backfill_fwar as bfwar  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Five known players with real MLBAM ids + made-up FanGraphs ids.
_KNOWN_PLAYERS = [
    # player_id (MLBAM), full_name, fg_id
    (592450, "Aaron Judge", "15640"),
    (660271, "Shohei Ohtani", "19755"),
    (545361, "Mike Trout", "10155"),
    (554430, "Zack Wheeler", "11122"),
    (607074, "Austin Hedges", "14213"),
]


def _write_db_with_players(db_path: Path) -> None:
    """Initialise the schema and insert five canonical players."""
    conn = duckdb.connect(str(db_path))
    try:
        create_tables(conn)
        rows = []
        for pid, name, fg in _KNOWN_PLAYERS:
            rows.append({
                "player_id": pid,
                "full_name": name,
                "team": "XXX",
                "position": "OF",
                "throws": "R",
                "bats": "R",
                "mlbam_id": pid,
                "fg_id": fg,
                "bref_id": f"bref{pid}",
            })
        pdf = pd.DataFrame(rows)
        conn.execute("INSERT INTO players SELECT * FROM pdf")

        # Seed some season_batting_stats rows (2023 + 2024 for Judge/Trout/Hedges).
        bat_rows = []
        for pid, name, _ in _KNOWN_PLAYERS[:3] + [_KNOWN_PLAYERS[4]]:
            for season in (2023, 2024):
                bat_rows.append({
                    "player_id": pid, "season": season,
                    "pa": 600, "ab": 550, "h": 150, "doubles": 30, "triples": 1,
                    "hr": 25, "rbi": 80, "bb": 40, "so": 120, "sb": 5, "cs": 1,
                    "ba": 0.27, "obp": 0.35, "slg": 0.45, "ops": 0.80,
                    "woba": 0.34, "wrc_plus": 110.0, "war": None,
                    "babip": 0.30, "iso": 0.18, "k_pct": 22.0, "bb_pct": 8.0,
                    "hard_hit_pct": None, "barrel_pct": None,
                    "xba": None, "xslg": None, "xwoba": None,
                })
        bdf = pd.DataFrame(bat_rows)
        conn.execute("INSERT INTO season_batting_stats SELECT * FROM bdf")

        # season_pitching_stats: Ohtani + Wheeler for 2023 + 2024.
        pit_rows = []
        for pid, name, _ in (_KNOWN_PLAYERS[1], _KNOWN_PLAYERS[3]):
            for season in (2023, 2024):
                pit_rows.append({
                    "player_id": pid, "season": season,
                    "w": 10, "l": 5, "sv": 0, "g": 25, "gs": 25, "ip": 150.0,
                    "era": 3.50, "fip": None, "xfip": None, "siera": None,
                    "whip": 1.10, "k_pct": 28.0, "bb_pct": 7.0, "hr_per_9": 1.1,
                    "k_per_9": 10.0, "bb_per_9": 2.5,
                    "war": None, "stuff_plus": None, "location_plus": None,
                    "pitching_plus": None, "avg_fastball_velo": None,
                    "avg_spin_rate": None, "gb_pct": None, "fb_pct": None,
                    "ld_pct": None, "xba": None, "xslg": None, "xwoba": None,
                })
        pdf2 = pd.DataFrame(pit_rows)
        conn.execute("INSERT INTO season_pitching_stats SELECT * FROM pdf2")
    finally:
        conn.close()


def _fake_bwar_bat() -> pd.DataFrame:
    """Mimic pybaseball.bwar_bat(return_all=True) output for our 5 players."""
    # Two stints for Hedges in 2023 to exercise the aggregator.
    rows = [
        dict(mlb_ID=592450, year_ID=2023, name_common="Aaron Judge", team_ID="NYY", stint_ID=1, PA=458, WAR=5.4),
        dict(mlb_ID=592450, year_ID=2024, name_common="Aaron Judge", team_ID="NYY", stint_ID=1, PA=704, WAR=10.9),
        dict(mlb_ID=660271, year_ID=2023, name_common="Shohei Ohtani", team_ID="LAA", stint_ID=1, PA=599, WAR=6.0),
        dict(mlb_ID=660271, year_ID=2024, name_common="Shohei Ohtani", team_ID="LAD", stint_ID=1, PA=731, WAR=9.3),
        dict(mlb_ID=545361, year_ID=2023, name_common="Mike Trout", team_ID="LAA", stint_ID=1, PA=361, WAR=2.9),
        dict(mlb_ID=545361, year_ID=2024, name_common="Mike Trout", team_ID="LAA", stint_ID=1, PA=124, WAR=0.8),
        dict(mlb_ID=607074, year_ID=2023, name_common="Austin Hedges", team_ID="PIT", stint_ID=1, PA=200, WAR=-0.3),
        dict(mlb_ID=607074, year_ID=2023, name_common="Austin Hedges", team_ID="TEX", stint_ID=2, PA=50,  WAR=-0.1),
        dict(mlb_ID=607074, year_ID=2024, name_common="Austin Hedges", team_ID="CLE", stint_ID=1, PA=180, WAR=0.2),
        # Pitcher-only rows with 0 PA (should be filtered out).
        dict(mlb_ID=554430, year_ID=2023, name_common="Zack Wheeler", team_ID="PHI", stint_ID=1, PA=0, WAR=None),
        dict(mlb_ID=554430, year_ID=2024, name_common="Zack Wheeler", team_ID="PHI", stint_ID=1, PA=0, WAR=None),
    ]
    return pd.DataFrame(rows)


def _fake_bwar_pitch() -> pd.DataFrame:
    rows = [
        dict(mlb_ID=554430, year_ID=2023, name_common="Zack Wheeler", team_ID="PHI", stint_ID=1, IPouts=576, WAR=5.7),
        dict(mlb_ID=554430, year_ID=2024, name_common="Zack Wheeler", team_ID="PHI", stint_ID=1, IPouts=600, WAR=6.1),
        dict(mlb_ID=660271, year_ID=2023, name_common="Shohei Ohtani", team_ID="LAA", stint_ID=1, IPouts=396, WAR=4.3),
        # Missing 2024 for Ohtani (injury year) -- tests nullable WAR.
        dict(mlb_ID=660271, year_ID=2024, name_common="Shohei Ohtani", team_ID="LAD", stint_ID=1, IPouts=0, WAR=None),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temp DB with schema + seed rows."""
    db_path = tmp_path / "test_backfill.duckdb"
    _write_db_with_players(db_path)
    return db_path


@pytest.fixture
def mocked_pybaseball():
    """Patch pybaseball inside the backfill module to our fixtures."""
    fake = type("FakeModule", (), {})()
    fake.bwar_bat = lambda return_all=True: _fake_bwar_bat()
    fake.bwar_pitch = lambda return_all=True: _fake_bwar_pitch()
    with patch.dict(sys.modules, {"pybaseball": fake}):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_map_fg_id_to_mlb_id(tmp_db):
    """The fg_id -> mlb_id helper resolves known players and drops unknown ones."""
    fg_ids = [fg for _, _, fg in _KNOWN_PLAYERS] + ["999999_unknown"]
    mapping = bfwar._map_fg_id_to_mlb_id(fg_ids, db_path=tmp_db)
    assert len(mapping) == 5
    for pid, _, fg in _KNOWN_PLAYERS:
        assert mapping[fg] == pid
    assert "999999_unknown" not in mapping


def test_fetch_war_aggregates_stints(mocked_pybaseball):
    """Two Hedges stints in 2023 combine into a single row with PA=250, WAR=-0.4."""
    staged = bfwar.fetch_war_for_years((2023, 2024))
    assert "player_id" in staged.columns
    assert set(staged["position_type"].unique()) <= {"batter", "pitcher"}

    # Hedges 2023 batter row
    h23 = staged[
        (staged["player_id"] == 607074)
        & (staged["season"] == 2023)
        & (staged["position_type"] == "batter")
    ]
    assert len(h23) == 1
    assert h23["pa_or_ip"].iloc[0] == pytest.approx(250.0)
    assert h23["war"].iloc[0] == pytest.approx(-0.4, abs=1e-6)

    # Wheeler's zero-PA batter stints must be dropped.
    wb = staged[(staged["player_id"] == 554430) & (staged["position_type"] == "batter")]
    assert len(wb) == 0

    # Wheeler's pitcher rows kept.
    wp = staged[(staged["player_id"] == 554430) & (staged["position_type"] == "pitcher")]
    assert len(wp) == 2

    # Ohtani 2024 pitcher row has zero IP -> dropped (we only keep ip > 0).
    o24 = staged[
        (staged["player_id"] == 660271)
        & (staged["season"] == 2024)
        & (staged["position_type"] == "pitcher")
    ]
    assert len(o24) == 0


def test_merge_updates_war_columns(mocked_pybaseball, tmp_db):
    """End-to-end: fetch, stage, merge, verify both tables updated."""
    staged = bfwar.fetch_war_for_years((2023, 2024))
    matched, _ = bfwar.match_to_db_players(staged, db_path=tmp_db)
    report = bfwar.merge_into_db(matched, db_path=tmp_db)

    assert report["batter_updates_delta"] >= 4
    assert report["pitcher_updates_delta"] >= 2

    with duckdb.connect(str(tmp_db), read_only=True) as conn:
        judge = conn.execute(
            "SELECT war FROM season_batting_stats WHERE player_id = 592450 AND season = 2024"
        ).fetchone()
        assert judge[0] == pytest.approx(10.9, abs=1e-6)

        wheeler = conn.execute(
            "SELECT war FROM season_pitching_stats WHERE player_id = 554430 AND season = 2023"
        ).fetchone()
        assert wheeler[0] == pytest.approx(5.7, abs=1e-6)


def test_merge_is_idempotent(mocked_pybaseball, tmp_db):
    """Running the merge twice produces exactly the same DB state."""
    staged = bfwar.fetch_war_for_years((2023, 2024))
    matched, _ = bfwar.match_to_db_players(staged, db_path=tmp_db)

    bfwar.merge_into_db(matched, db_path=tmp_db)
    with duckdb.connect(str(tmp_db), read_only=True) as conn:
        bat1 = conn.execute(
            "SELECT player_id, season, war FROM season_batting_stats ORDER BY player_id, season"
        ).fetchdf()
        pit1 = conn.execute(
            "SELECT player_id, season, war FROM season_pitching_stats ORDER BY player_id, season"
        ).fetchdf()

    bfwar.merge_into_db(matched, db_path=tmp_db)
    with duckdb.connect(str(tmp_db), read_only=True) as conn:
        bat2 = conn.execute(
            "SELECT player_id, season, war FROM season_batting_stats ORDER BY player_id, season"
        ).fetchdf()
        pit2 = conn.execute(
            "SELECT player_id, season, war FROM season_pitching_stats ORDER BY player_id, season"
        ).fetchdf()

    pd.testing.assert_frame_equal(bat1, bat2)
    pd.testing.assert_frame_equal(pit1, pit2)


def test_match_to_db_players_reports_unmatched(tmp_db):
    """Rows with MLBAM ids not present in ``players`` land in the unmatched frame."""
    staged = pd.DataFrame([
        {"player_id": 592450, "player_name": "Aaron Judge", "season": 2023,
         "position_type": "batter", "war": 5.4, "pa_or_ip": 458.0, "war_source": "bref_bwar"},
        {"player_id": 111111, "player_name": "Unknown Player", "season": 2023,
         "position_type": "batter", "war": 0.1, "pa_or_ip": 100.0, "war_source": "bref_bwar"},
    ])
    matched, unmatched = bfwar.match_to_db_players(staged, db_path=tmp_db)
    assert len(matched) == 1
    assert len(unmatched) == 1
    assert int(unmatched["player_id"].iloc[0]) == 111111


def test_audit_summary_shape(mocked_pybaseball):
    """Audit dict has the expected keys + per-year breakdown."""
    staged = bfwar.fetch_war_for_years((2023, 2024))
    unmatched = staged.iloc[0:0]
    audit = bfwar.audit_staged(staged, unmatched)
    assert audit["n_matched"] == len(staged)
    assert audit["n_unmatched"] == 0
    assert audit["match_rate"] == 1.0
    assert any(y["season"] == 2023 for y in audit["per_year"])
    assert any(y["season"] == 2024 for y in audit["per_year"])
    for y in audit["per_year"]:
        assert y["batters"] >= 0
        assert y["pitchers"] >= 0
