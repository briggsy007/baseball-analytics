"""
Unit tests for ``scripts/backfill_2026_war.py``.

Fully offline: ``pybaseball`` is monkey-patched with a synthetic 2026 fixture,
and merges run against an in-memory DuckDB built via
``src.db.schema.create_tables``.  Verifies the 2026-only filter, the dry-run
(no DB write) path, the KILL CRITERION, and the executing-write path.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import duckdb
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.db.schema import create_tables  # noqa: E402
import scripts.backfill_2026_war as b26  # noqa: E402


_KNOWN_PLAYERS = [
    (592450, "Aaron Judge"),
    (660271, "Shohei Ohtani"),
    (554430, "Zack Wheeler"),
]


def _write_db(db_path: Path) -> None:
    # NOTE: seed via explicit-column INSERTs (no pandas->duckdb arrow bridge).
    # Registering a pandas frame into duckdb registers the ``pandas.period``
    # arrow extension, which later collides with pandas ``to_parquet`` (pyarrow)
    # inside ``run()`` and raises ArrowKeyError. Plain INSERTs avoid that.
    conn = duckdb.connect(str(db_path))
    try:
        create_tables(conn)
        conn.executemany(
            "INSERT INTO players (player_id, full_name, mlbam_id) VALUES (?, ?, ?)",
            [(pid, name, pid) for pid, name in _KNOWN_PLAYERS],
        )
        # 2026 batting rows (war left NULL) for Judge + Ohtani.
        conn.executemany(
            "INSERT INTO season_batting_stats (player_id, season, pa, ops) "
            "VALUES (?, 2026, 400, 0.86)",
            [(592450,), (660271,)],
        )
        # 2026 pitching row (war left NULL) for Wheeler.
        conn.execute(
            "INSERT INTO season_pitching_stats (player_id, season, ip, era) "
            "VALUES (554430, 2026, 130.0, 3.10)"
        )
    finally:
        conn.close()


def _fake_bwar_bat() -> pd.DataFrame:
    # Includes a 2025 row that MUST be filtered out.
    return pd.DataFrame([
        dict(mlb_ID=592450, year_ID=2025, name_common="Aaron Judge", stint_ID=1, PA=700, WAR=9.0),
        dict(mlb_ID=592450, year_ID=2026, name_common="Aaron Judge", stint_ID=1, PA=400, WAR=5.0),
        dict(mlb_ID=660271, year_ID=2026, name_common="Shohei Ohtani", stint_ID=1, PA=450, WAR=6.0),
        # Unknown player -> unmatched.
        dict(mlb_ID=111111, year_ID=2026, name_common="Nobody", stint_ID=1, PA=300, WAR=1.0),
    ])


def _fake_bwar_pitch() -> pd.DataFrame:
    return pd.DataFrame([
        dict(mlb_ID=554430, year_ID=2025, name_common="Zack Wheeler", stint_ID=1, IPouts=600, WAR=6.0),
        dict(mlb_ID=554430, year_ID=2026, name_common="Zack Wheeler", stint_ID=1, IPouts=390, WAR=3.5),
    ])


def _empty_bwar_bat() -> pd.DataFrame:
    # Only pre-2026 batting data -> no 2026 rows after filtering.
    return pd.DataFrame([
        dict(mlb_ID=592450, year_ID=2025, name_common="Aaron Judge", stint_ID=1, PA=700, WAR=9.0),
    ])


def _empty_bwar_pitch() -> pd.DataFrame:
    # Only pre-2026 pitching data -> no 2026 rows after filtering.
    return pd.DataFrame([
        dict(mlb_ID=554430, year_ID=2025, name_common="Zack Wheeler", stint_ID=1, IPouts=600, WAR=6.0),
    ])


@pytest.fixture
def tmp_db(tmp_path):
    p = tmp_path / "t.duckdb"
    _write_db(p)
    return p


@pytest.fixture
def mocked_pb():
    fake = type("FakeModule", (), {})()
    fake.bwar_bat = lambda return_all=True: _fake_bwar_bat()
    fake.bwar_pitch = lambda return_all=True: _fake_bwar_pitch()
    with patch.dict(sys.modules, {"pybaseball": fake}):
        yield


@pytest.fixture
def mocked_pb_empty():
    fake = type("FakeModule", (), {})()
    fake.bwar_bat = lambda return_all=True: _empty_bwar_bat()
    fake.bwar_pitch = lambda return_all=True: _empty_bwar_pitch()
    with patch.dict(sys.modules, {"pybaseball": fake}):
        yield


def test_kill_criterion_no_2026(mocked_pb_empty, tmp_path):
    """Unreachable/absent 2026 data trips the kill criterion (no parquet write)."""
    with pytest.raises(b26.KillCriterion):
        b26.fetch_2026_war(tmp_path / "s.parquet", skip_fetch=False)


def test_dry_run_then_write(mocked_pb, tmp_db, tmp_path):
    """End-to-end: dry-run stages + filters to 2026 + leaves DB NULL; then the
    executing write (re-using the staged parquet) populates WAR.

    NOTE: this is deliberately ONE test doing a single ``to_parquet`` (the
    dry-run stage) and then ``skip_fetch=True`` for the write phase.  duckdb
    1.2.2 + pyarrow 19 clash on ``pandas.period`` extension re-registration if
    two independent ``to_parquet`` calls happen in the same process; production
    is unaffected because the orchestrator runs each script in a fresh process.
    """
    staging = tmp_path / "s.parquet"
    audit = tmp_path / "a.json"
    unmatched = tmp_path / "u.csv"

    # --- Dry-run: stage only, no DB write ---
    res = b26.run(
        dry_run=True, db_path=tmp_db,
        staging_path=staging, unmatched_path=unmatched, audit_path=audit,
    )
    assert res["dry_run"] is True
    assert res["audit"]["match_rate"] is not None
    assert audit.exists()

    staged = pd.read_parquet(staging)
    assert set(staged["season"].unique()) == {2026}  # 2025 rows filtered out
    j = staged[(staged["player_id"] == 592450) & (staged["position_type"] == "batter")]
    assert len(j) == 1 and j["war"].iloc[0] == pytest.approx(5.0)

    with duckdb.connect(str(tmp_db), read_only=True) as conn:
        n_bat = conn.execute(
            "SELECT COUNT(*) FROM season_batting_stats WHERE season=2026 AND war IS NOT NULL"
        ).fetchone()[0]
        n_pit = conn.execute(
            "SELECT COUNT(*) FROM season_pitching_stats WHERE season=2026 AND war IS NOT NULL"
        ).fetchone()[0]
    assert n_bat == 0 and n_pit == 0  # dry-run wrote nothing

    # --- Executing write: reuse staged parquet (skip_fetch -> no 2nd to_parquet) ---
    res2 = b26.run(
        dry_run=False, skip_fetch=True, db_path=tmp_db,
        staging_path=staging, unmatched_path=unmatched, audit_path=audit,
    )
    assert res2["merge"]["batter_updates_delta"] >= 2
    assert res2["merge"]["pitcher_updates_delta"] >= 1
    with duckdb.connect(str(tmp_db), read_only=True) as conn:
        judge = conn.execute(
            "SELECT war FROM season_batting_stats WHERE player_id=592450 AND season=2026"
        ).fetchone()[0]
        wheeler = conn.execute(
            "SELECT war FROM season_pitching_stats WHERE player_id=554430 AND season=2026"
        ).fetchone()[0]
    assert judge == pytest.approx(5.0)
    assert wheeler == pytest.approx(3.5)
