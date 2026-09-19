"""Regression tests for the hit-parlay leg ingest-gap fix (2026-09-19).

Incident: ``run_daily_etl`` only ever loaded "yesterday", so 27 game dates
between 2026-08-19 and 2026-09-18 were never ingested into ``pitches``. The
old resolver gated only on the GLOBAL watermark (``MAX(game_date)``), so once
any later day was ingested the watermark advanced past the missing dates and
every hit-parlay leg on those dates was resolved ``void`` (``void-no-ab``,
ab=0) even though the day itself had never been loaded -- 16 permanent wrong
voids, since resolutions are append-only.

These tests cover the two independent gates the fix adds ahead of the
existing watermark check:

  1. ``_date_ingested`` -- is the pick's own game_date present in ``pitches``
     at all (not just "does the watermark sit past it").
  2. ``_game_ingested`` -- given the date IS ingested, is this pick's
     specific game present yet, with a 3-day grace period before a still-
     absent game is treated as postponed/cancelled (``void-game-absent``,
     distinct from the ordinary ``void-no-ab`` outcome for a game that
     happened but where this batter had zero at-bats).

Loads ``scripts/resolve_picks.py`` by path because ``scripts/`` is not a
package (mirrors ``tests/test_nightly_effect_checks.py``).
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from src.pick_ledger import (  # noqa: E402
    build_hit_parlay_leg_pick,
    build_hit_parlay_parlay_pick,
    emit_pick,
)

_SCRIPT = ROOT / "scripts" / "resolve_picks.py"


@pytest.fixture(scope="module")
def rp():
    spec = importlib.util.spec_from_file_location("resolve_picks_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "picks.jsonl", tmp_path / "resolutions.jsonl"


def _conn() -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE pitches (
            game_pk    INTEGER,
            game_date  DATE,
            home_team  VARCHAR,
            away_team  VARCHAR,
            batter_id  INTEGER,
            events     VARCHAR
        )
        """
    )
    return conn


def _insert(conn, *, game_pk: int, day: str, home: str, away: str,
            batter_id: int, events: str | None) -> None:
    conn.execute(
        "INSERT INTO pitches VALUES (?, ?, ?, ?, ?, ?)",
        [game_pk, day, home, away, batter_id, events],
    )


def _leg(day: str, game: str, player_id: int, *, leg_index: int = 1,
         name: str = "Test Player", p: float = 0.6) -> dict:
    leg = {
        "name": name, "team": "PHI", "game": game, "opp_sp": "Someone",
        "slot": 1, "p_hit_game": p, "player_id": player_id, "projected": False,
    }
    return build_hit_parlay_leg_pick(
        day, leg_index, leg,
        frozen_utc="2026-08-01T00:00:00+00:00", git_sha="deadbeef",
    )


def _resolutions(res_path: Path) -> list[dict]:
    """Read the resolutions JSONL directly (never via ``load_ledger``'s
    default picks path -- these tests must not touch the real
    ``predictions/picks.jsonl``)."""
    if not res_path.exists():
        return []
    with open(res_path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# (a) watermark past the leg day, but the day itself has zero rows
# ---------------------------------------------------------------------------


def test_watermark_past_but_day_absent_is_skipped_not_voided(rp, paths):
    picks_path, res_path = paths
    conn = _conn()
    leg_day = "2026-08-20"
    # Advance the watermark with a LATER day's ingest; leg_day has 0 rows.
    _insert(conn, game_pk=1, day="2026-08-25", home="PHI", away="NYM",
            batter_id=999, events="single")

    pick = _leg(leg_day, "TOR@PHI", player_id=42)
    emit_pick(pick, picks_path=picks_path)

    rc = rp.run(date(2026, 8, 26), conn=conn,
                picks_path=picks_path, resolutions_path=res_path)

    assert rc == 0
    assert _resolutions(res_path) == []  # nothing appended -- not a void


# ---------------------------------------------------------------------------
# (b) day ingested, game absent, within the 3-day grace period
# ---------------------------------------------------------------------------


def test_day_ingested_game_absent_within_grace_is_skipped(rp, paths):
    picks_path, res_path = paths
    conn = _conn()
    leg_day = "2026-08-20"
    # The day IS ingested, but only for a different game.
    _insert(conn, game_pk=1, day=leg_day, home="ATL", away="NYM",
            batter_id=999, events="single")

    pick = _leg(leg_day, "TOR@PHI", player_id=42)
    emit_pick(pick, picks_path=picks_path)

    rc = rp.run(date(2026, 8, 21), conn=conn,  # as_of = leg_day + 1
                picks_path=picks_path, resolutions_path=res_path)

    assert rc == 0
    assert _resolutions(res_path) == []


# ---------------------------------------------------------------------------
# (c) day ingested, game absent, grace period has elapsed -> void-game-absent
# ---------------------------------------------------------------------------


def test_day_ingested_game_absent_past_grace_voids_as_game_absent(rp, paths):
    picks_path, res_path = paths
    conn = _conn()
    leg_day = "2026-08-20"
    _insert(conn, game_pk=1, day=leg_day, home="ATL", away="NYM",
            batter_id=999, events="single")

    pick = _leg(leg_day, "TOR@PHI", player_id=42)
    emit_pick(pick, picks_path=picks_path)

    rc = rp.run(date(2026, 8, 23), conn=conn,  # as_of = leg_day + 3
                picks_path=picks_path, resolutions_path=res_path)

    assert rc == 0
    resolutions = _resolutions(res_path)
    assert len(resolutions) == 1
    r = resolutions[0]
    assert r["outcome"] == "void"
    assert r["resolution_branch"] == "void-game-absent"
    assert r["game_ingested"] is False
    assert r["date_ingested"] is True


# ---------------------------------------------------------------------------
# (d) game ingested, batter has an AB and no hit -> no
# ---------------------------------------------------------------------------


def test_game_ingested_ab_no_hit_resolves_no(rp, paths):
    picks_path, res_path = paths
    conn = _conn()
    leg_day = "2026-08-20"
    _insert(conn, game_pk=1, day=leg_day, home="PHI", away="TOR",
            batter_id=42, events="strikeout")

    pick = _leg(leg_day, "TOR@PHI", player_id=42, p=0.6)
    emit_pick(pick, picks_path=picks_path)

    rc = rp.run(date(2026, 8, 21), conn=conn,
                picks_path=picks_path, resolutions_path=res_path)

    assert rc == 0
    resolutions = _resolutions(res_path)
    assert len(resolutions) == 1
    r = resolutions[0]
    assert r["outcome"] == "no"
    assert r["resolution_branch"] == "no"
    assert r["game_ingested"] is True
    assert r["ab"] == 1
    assert r["hits"] == 0
    assert r["brier"] == pytest.approx((0.6 - 0.0) ** 2)


# ---------------------------------------------------------------------------
# (e) a single -> yes, with brier fields
# ---------------------------------------------------------------------------


def test_game_ingested_hit_resolves_yes_with_brier(rp, paths):
    picks_path, res_path = paths
    conn = _conn()
    leg_day = "2026-08-20"
    _insert(conn, game_pk=1, day=leg_day, home="PHI", away="TOR",
            batter_id=42, events="single")

    pick = _leg(leg_day, "TOR@PHI", player_id=42, p=0.6)
    emit_pick(pick, picks_path=picks_path)

    rc = rp.run(date(2026, 8, 21), conn=conn,
                picks_path=picks_path, resolutions_path=res_path)

    assert rc == 0
    resolutions = _resolutions(res_path)
    assert len(resolutions) == 1
    r = resolutions[0]
    assert r["outcome"] == "yes"
    assert r["resolution_branch"] == "yes"
    assert r["game_ingested"] is True
    assert r["hits"] == 1
    assert r["p"] == 0.6
    assert r["y"] == 1.0
    assert r["brier"] == pytest.approx((0.6 - 1.0) ** 2)


# ---------------------------------------------------------------------------
# (f) game ingested, batter has no rows -> void-no-ab
# ---------------------------------------------------------------------------


def test_game_ingested_batter_absent_voids_no_ab(rp, paths):
    picks_path, res_path = paths
    conn = _conn()
    leg_day = "2026-08-20"
    # The game IS ingested (another batter has rows), but our subject
    # batter never appears -- did not play, not "game absent".
    _insert(conn, game_pk=1, day=leg_day, home="PHI", away="TOR",
            batter_id=999, events="single")

    pick = _leg(leg_day, "TOR@PHI", player_id=42)
    emit_pick(pick, picks_path=picks_path)

    rc = rp.run(date(2026, 8, 21), conn=conn,
                picks_path=picks_path, resolutions_path=res_path)

    assert rc == 0
    resolutions = _resolutions(res_path)
    assert len(resolutions) == 1
    r = resolutions[0]
    assert r["outcome"] == "void"
    assert r["resolution_branch"] == "void-no-ab"
    assert r["game_ingested"] is True
    assert r["ab"] == 0
    assert r["hits"] == 0


# ---------------------------------------------------------------------------
# (g) parlay resolves only after all legs resolve
# ---------------------------------------------------------------------------


def test_parlay_waits_for_all_legs_then_resolves(rp, paths):
    picks_path, res_path = paths
    conn = _conn()
    d1, d2 = "2026-08-20", "2026-08-21"

    leg1 = _leg(d1, "TOR@PHI", player_id=1, leg_index=1, name="Leg One")
    leg2 = _leg(d2, "NYM@PHI", player_id=2, leg_index=2, name="Leg Two")
    parlay = build_hit_parlay_parlay_pick(
        d1, [leg1["pick_id"], leg2["pick_id"]], 0.36,
        frozen_utc="2026-08-01T00:00:00+00:00", git_sha="deadbeef",
    )
    emit_pick(leg1, picks_path=picks_path)
    emit_pick(leg2, picks_path=picks_path)
    emit_pick(parlay, picks_path=picks_path)

    # Only d1 has been ingested so far; as_of sits right at d2 (leg2's own
    # day not yet "complete", so leg2 is skipped for that ordinary reason).
    _insert(conn, game_pk=1, day=d1, home="PHI", away="TOR",
            batter_id=1, events="single")
    rc = rp.run(date(2026, 8, 21), conn=conn,
                picks_path=picks_path, resolutions_path=res_path)
    assert rc == 0
    resolved_ids = {r["pick_id"] for r in _resolutions(res_path)}
    assert leg1["pick_id"] in resolved_ids
    assert leg2["pick_id"] not in resolved_ids
    assert parlay["pick_id"] not in resolved_ids

    # Now d2 ingests and as_of moves past it -- leg2 and then the parlay
    # both resolve within this second run.
    _insert(conn, game_pk=2, day=d2, home="PHI", away="NYM",
            batter_id=2, events="single")
    rc = rp.run(date(2026, 8, 22), conn=conn,
                picks_path=picks_path, resolutions_path=res_path)
    assert rc == 0
    resolutions_by_id = {r["pick_id"]: r for r in _resolutions(res_path)}
    assert resolutions_by_id[leg2["pick_id"]]["outcome"] == "yes"
    assert resolutions_by_id[parlay["pick_id"]]["outcome"] == "yes"
