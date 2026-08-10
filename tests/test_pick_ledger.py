"""Tests for the append-only pick ledger (plan WS1.2).

Covers: emit/resolve/load round-trip, append-only refusals (duplicate
pick_id, unknown pick, double resolution, bad outcome), track-record
computation against fixture ledgers, pick builders' frozen rule hashes,
and the repo ledger's own integrity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.pick_ledger import (
    HIT_PARLAY_LEG_RULE,
    HIT_PARLAY_PARLAY_RULE,
    PICKS_PATH,
    RESOLUTIONS_PATH,
    LedgerError,
    build_hit_parlay_leg_pick,
    build_hit_parlay_parlay_pick,
    compute_track_record,
    emit_pick,
    load_ledger,
    resolve_pick,
)


def _pick(pick_id: str = "test-pick-1", p: float = 0.7) -> dict:
    return {
        "pick_id": pick_id,
        "product": "test_product",
        "model": "test_model",
        "artifact_version": "v-test",
        "git_sha": "deadbeef",
        "subject": {"name": "Test Player"},
        "market_or_claim": "test claim",
        "p_or_direction": {"p": p},
        "resolution_rule": "yes iff test",
        "resolution_source": "test source",
        "resolve_by": "2026-01-02",
        "rule_hash": "abc123",
        "evidence_class": "prospective",
        "frozen_utc": "2026-01-01T00:00:00+00:00",
    }


@pytest.fixture()
def paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "picks.jsonl", tmp_path / "resolutions.jsonl"


class TestEmitPick:
    def test_emit_and_load_round_trip(self, paths):
        picks_path, res_path = paths
        emit_pick(_pick(), picks_path=picks_path)
        ledger = load_ledger(picks_path=picks_path, resolutions_path=res_path)
        assert len(ledger["picks"]) == 1
        assert ledger["picks"][0]["pick_id"] == "test-pick-1"
        assert ledger["resolutions"] == []

    def test_refuses_duplicate_pick_id(self, paths):
        picks_path, _ = paths
        emit_pick(_pick(), picks_path=picks_path)
        with pytest.raises(LedgerError, match="append-only"):
            emit_pick(_pick(), picks_path=picks_path)
        # the file was not corrupted by the refusal
        assert len(load_ledger(picks_path=picks_path)["picks"]) == 1

    def test_refuses_missing_required_fields(self, paths):
        picks_path, _ = paths
        bad = _pick()
        del bad["resolution_rule"]
        with pytest.raises(LedgerError, match="resolution_rule"):
            emit_pick(bad, picks_path=picks_path)
        assert not picks_path.exists()

    def test_autofills_frozen_utc_and_git_sha(self, paths):
        picks_path, _ = paths
        p = _pick()
        del p["frozen_utc"]
        del p["git_sha"]
        entry = emit_pick(p, picks_path=picks_path)
        assert entry["frozen_utc"]  # now-UTC filled
        assert "git_sha" in entry


class TestResolvePick:
    def test_resolve_round_trip(self, paths):
        picks_path, res_path = paths
        emit_pick(_pick(), picks_path=picks_path)
        entry = resolve_pick(
            "test-pick-1", "yes", score_fields={"brier": 0.09},
            picks_path=picks_path, resolutions_path=res_path,
        )
        assert entry["outcome"] == "yes"
        ledger = load_ledger(picks_path=picks_path, resolutions_path=res_path)
        assert ledger["resolutions"][0]["brier"] == 0.09

    def test_refuses_unknown_pick(self, paths):
        picks_path, res_path = paths
        emit_pick(_pick(), picks_path=picks_path)
        with pytest.raises(LedgerError, match="unknown pick_id"):
            resolve_pick("nope", "yes",
                         picks_path=picks_path, resolutions_path=res_path)

    def test_refuses_double_resolution(self, paths):
        picks_path, res_path = paths
        emit_pick(_pick(), picks_path=picks_path)
        resolve_pick("test-pick-1", "no",
                     picks_path=picks_path, resolutions_path=res_path)
        with pytest.raises(LedgerError, match="already resolved"):
            resolve_pick("test-pick-1", "yes",
                         picks_path=picks_path, resolutions_path=res_path)

    def test_refuses_invalid_outcome(self, paths):
        picks_path, res_path = paths
        emit_pick(_pick(), picks_path=picks_path)
        with pytest.raises(LedgerError, match="outcome"):
            resolve_pick("test-pick-1", "maybe",
                         picks_path=picks_path, resolutions_path=res_path)


class TestTrackRecord:
    """The track-record view's loader math, against fixture ledgers."""

    def _fixture(self, paths):
        picks_path, res_path = paths
        for i, p in enumerate([0.9, 0.8, 0.6, 0.55], 1):
            emit_pick(_pick(f"pk-{i}", p=p), picks_path=picks_path)
        # directional pick (no probability)
        d = _pick("pk-dir")
        d["p_or_direction"] = {"direction": "bullish"}
        emit_pick(d, picks_path=picks_path)
        # unresolved pick
        emit_pick(_pick("pk-open", p=0.5), picks_path=picks_path)
        resolve_pick("pk-1", "yes", picks_path=picks_path,
                     resolutions_path=res_path,
                     resolved_utc="2026-01-02T00:00:00+00:00")
        resolve_pick("pk-2", "no", picks_path=picks_path,
                     resolutions_path=res_path,
                     resolved_utc="2026-01-03T00:00:00+00:00")
        resolve_pick("pk-3", "yes", picks_path=picks_path,
                     resolutions_path=res_path,
                     resolved_utc="2026-01-04T00:00:00+00:00")
        resolve_pick("pk-4", "void", picks_path=picks_path,
                     resolutions_path=res_path,
                     resolved_utc="2026-01-05T00:00:00+00:00")
        resolve_pick("pk-dir", "no", picks_path=picks_path,
                     resolutions_path=res_path,
                     resolved_utc="2026-01-06T00:00:00+00:00")
        return load_ledger(picks_path=picks_path, resolutions_path=res_path)

    def test_summary_counts_losses_like_wins(self, paths):
        ledger = self._fixture(paths)
        tr = compute_track_record(ledger["picks"], ledger["resolutions"])
        s = tr["summary"]
        assert s["picks_total"] == 6
        assert s["resolved"] == 5
        assert s["unresolved"] == 1
        assert s["hits"] == 2
        assert s["misses"] == 2  # a miss is never dropped
        assert s["voids"] == 1
        assert s["hit_rate_itt"] == pytest.approx(2 / 4)
        assert s["hit_rate_worst_case"] == pytest.approx(2 / 5)

    def test_brier_math(self, paths):
        ledger = self._fixture(paths)
        tr = compute_track_record(ledger["picks"], ledger["resolutions"])
        # scored picks: 0.9->yes, 0.8->no, 0.6->yes (void and directional
        # excluded from Brier)
        expected = ((0.9 - 1) ** 2 + (0.8 - 0) ** 2 + (0.6 - 1) ** 2) / 3
        assert tr["summary"]["mean_brier"] == pytest.approx(expected, abs=1e-6)
        assert tr["summary"]["n_brier_scored"] == 3
        roll = tr["rolling_brier"]
        assert [r["n"] for r in roll] == [1, 2, 3]
        assert roll[-1]["cumulative_mean_brier"] == pytest.approx(
            expected, abs=1e-6
        )

    def test_calibration_buckets(self, paths):
        ledger = self._fixture(paths)
        tr = compute_track_record(ledger["picks"], ledger["resolutions"])
        buckets = {c["bucket"]: c for c in tr["calibration"]}
        # 0.9 and 0.8 land in [0.80, 1.00); one hit of two
        b = buckets["[0.80, 1.00)"]
        assert b["n"] == 2
        assert b["observed_rate"] == pytest.approx(0.5)
        # 0.6 lands in [0.60, 0.70) with a hit
        assert buckets["[0.60, 0.70)"]["observed_rate"] == pytest.approx(1.0)

    def test_ledger_files_are_the_only_input(self, paths):
        """compute_track_record accepts plain lists — nothing else."""
        tr = compute_track_record([], [])
        assert tr["summary"]["picks_total"] == 0
        assert tr["rolling_brier"] == []
        assert tr["calibration"] == []


class TestHitParlayBuilders:
    def test_leg_rule_hash_is_sha256_of_frozen_text(self):
        leg = {"name": "Test Guy", "team": "PHI", "game": "NYY@PHI",
               "opp_sp": "Somebody", "slot": 1, "p_hit_game": 0.75,
               "player_id": 123, "projected": False}
        pick = build_hit_parlay_leg_pick("2026-08-09", 1, leg)
        assert pick["rule_hash"] == hashlib.sha256(
            HIT_PARLAY_LEG_RULE.encode()).hexdigest()
        assert pick["pick_id"] == "hitparlay-2026-08-09-leg1-test-guy"
        assert pick["resolve_by"] == "2026-08-10"
        assert pick["p_or_direction"]["p"] == 0.75
        assert pick["evidence_class"] == "prospective"
        # a public resolution source is named
        assert "MLB Stats API" in pick["resolution_source"]

    def test_parlay_rule_hash_and_legs(self):
        pick = build_hit_parlay_parlay_pick(
            "2026-08-09", ["a", "b"], 0.5)
        assert pick["rule_hash"] == hashlib.sha256(
            HIT_PARLAY_PARLAY_RULE.encode()).hexdigest()
        assert pick["subject"]["legs"] == ["a", "b"]


class TestRepoLedgerIntegrity:
    """The committed predictions/ ledgers must stay parseable + coherent."""

    def test_repo_picks_ledger_valid(self):
        if not PICKS_PATH.exists():
            pytest.skip("predictions/picks.jsonl not present")
        ledger = load_ledger()
        picks = ledger["picks"]
        assert picks, "picks.jsonl exists but is empty"
        ids = [p["pick_id"] for p in picks]
        assert len(ids) == len(set(ids)), "duplicate pick_ids in ledger"
        for p in picks:
            # every pick names a rule, a source, a deadline, and a rule hash
            assert p["resolution_rule"]
            assert p["resolution_source"]
            assert p["resolve_by"]
            assert p["rule_hash"]
            assert p["evidence_class"]

    def test_repo_resolutions_reference_known_picks(self):
        if not RESOLUTIONS_PATH.exists():
            pytest.skip("predictions/resolutions.jsonl not present")
        ledger = load_ledger()
        known = {p["pick_id"] for p in ledger["picks"]}
        rids = [r["pick_id"] for r in ledger["resolutions"]]
        assert len(rids) == len(set(rids)), "double resolution in ledger"
        for r in ledger["resolutions"]:
            assert r["pick_id"] in known
            assert r["outcome"] in ("yes", "no", "void")

    def test_repo_contrarian_picks_pin_frozen_spec_hash(self):
        if not PICKS_PATH.exists():
            pytest.skip("predictions/picks.jsonl not present")
        spec_hash = ("1a27cd0e2d9b7d08c69c5a8a5944602585931121d9396be922e"
                     "1e519557c760e")
        ledger = load_ledger()
        boards = [p for p in ledger["picks"]
                  if p["product"] == "contrarian_board_2026_midseason"]
        assert len(boards) == 50
        assert all(p["rule_hash"] == spec_hash for p in boards)

    def test_view_module_imports_and_exposes_render(self):
        track_record = pytest.importorskip(
            "src.dashboard.views.track_record",
            reason="streamlit not installed",
        )
        assert callable(track_record.render)


def test_malformed_jsonl_raises(tmp_path):
    bad = tmp_path / "picks.jsonl"
    bad.write_text('{"pick_id": "ok"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(LedgerError, match="line 2"):
        load_ledger(picks_path=bad, resolutions_path=tmp_path / "r.jsonl")


def test_emitted_lines_are_single_line_json(paths):
    picks_path, _ = paths
    p = _pick()
    p["market_or_claim"] = "claim with\nnewline"
    emit_pick(p, picks_path=picks_path)
    lines = picks_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    assert json.loads(lines[0])["pick_id"] == "test-pick-1"
