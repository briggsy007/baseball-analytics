"""Tests for the holdout-contact ledger (plan WS2.3).

Covers: @holdout_access refusal past budget, override logging, sealed
lockbox refusal, hash verification, contact numbering; the committed
docs/holdout_ledger.jsonl's integrity; and the CI guard that fails any
eval script touching a registered holdout identifier without routing
through the ledger.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.holdout import (
    LEDGER_PATH,
    HoldoutBudgetExceeded,
    HoldoutHashMismatch,
    HoldoutSealedError,
    contact_count,
    dataset_policy,
    holdout_access,
    record_contact,
)

ROOT = Path(__file__).resolve().parents[1]


def _write_header(path: Path, policy: dict) -> None:
    entry = {"entry_type": "header", "ts": "2026-01-01T00:00:00+00:00",
             "tier_policy": policy}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


class TestDecorator:
    def test_contact_logged_with_metrics(self, tmp_path):
        led = tmp_path / "ledger.jsonl"

        @holdout_access("ds", "unit test", budget=2,
                        metrics=["K%"], ledger_path=led)
        def evaluate():
            return {"BB%": 0.08}

        out = evaluate()
        assert out == {"BB%": 0.08}
        assert contact_count("ds", led) == 1
        entry = json.loads(led.read_text().strip().split("\n")[-1])
        assert entry["entry_type"] == "contact"
        assert entry["contact_number"] == 1
        assert sorted(entry["metrics_revealed"]) == ["BB%", "K%"]
        assert "git_sha" in entry

    def test_refuses_past_budget(self, tmp_path):
        led = tmp_path / "ledger.jsonl"
        record_contact("ds", "spend 1", ledger_path=led)
        record_contact("ds", "spend 2", ledger_path=led)

        @holdout_access("ds", "one too many", budget=2, ledger_path=led)
        def evaluate():  # pragma: no cover - must not run
            raise AssertionError("function body must not execute past budget")

        with pytest.raises(HoldoutBudgetExceeded, match="budget is 2"):
            evaluate()
        # refusal appended nothing
        assert contact_count("ds", led) == 2

    def test_override_runs_and_is_logged(self, tmp_path):
        led = tmp_path / "ledger.jsonl"
        record_contact("ds", "spend 1", ledger_path=led)

        @holdout_access("ds", "overridden", budget=1, ledger_path=led,
                        override=True, override_reason="pre-registered amendment")
        def evaluate():
            return 42

        assert evaluate() == 42
        entries = [json.loads(x) for x in led.read_text().strip().split("\n")]
        last = entries[-1]
        assert last["override"] is True
        assert last["override_reason"] == "pre-registered amendment"
        assert last["contact_number"] == 2

    def test_header_budget_caps_decorator_budget(self, tmp_path):
        led = tmp_path / "ledger.jsonl"
        _write_header(led, {"ds": {"tier": "budgeted", "budget": 1}})
        record_contact("ds", "spend 1", ledger_path=led)

        @holdout_access("ds", "decorator says 99, header says 1",
                        budget=99, ledger_path=led)
        def evaluate():  # pragma: no cover
            raise AssertionError("must not run")

        with pytest.raises(HoldoutBudgetExceeded):
            evaluate()

    def test_sealed_lockbox_refused(self, tmp_path):
        led = tmp_path / "ledger.jsonl"
        _write_header(led, {"box": {"tier": "lockbox", "unsealed": False}})

        @holdout_access("box", "peeking", budget=1, ledger_path=led)
        def evaluate():  # pragma: no cover
            raise AssertionError("must not run")

        with pytest.raises(HoldoutSealedError, match="sealed LOCKBOX"):
            evaluate()

    def test_hash_mismatch_refused(self, tmp_path, monkeypatch):
        led = tmp_path / "ledger.jsonl"
        data = tmp_path / "cohort.npz"
        data.write_bytes(b"cohort-bytes")
        import src.holdout as h

        monkeypatch.setitem(h.DATASET_HASHES, "hashed_ds", "0" * 64)
        with pytest.raises(HoldoutHashMismatch, match="does not match"):
            record_contact("hashed_ds", "tampered", dataset_path=data,
                           ledger_path=led)
        # registered hash + no file offered is also refused
        with pytest.raises(HoldoutHashMismatch, match="pass dataset_path"):
            record_contact("hashed_ds", "no file", ledger_path=led)

    def test_contact_numbers_increment(self, tmp_path):
        led = tmp_path / "ledger.jsonl"
        e1 = record_contact("ds", "a", ledger_path=led)
        e2 = record_contact("ds", "b", ledger_path=led)
        e3 = record_contact("other", "c", ledger_path=led)
        assert (e1["contact_number"], e2["contact_number"]) == (1, 2)
        assert e3["contact_number"] == 1  # per-dataset numbering


class TestRepoLedgerIntegrity:
    def test_ledger_exists_and_parses(self):
        assert LEDGER_PATH.exists(), "docs/holdout_ledger.jsonl missing"
        lines = [ln for ln in
                 LEDGER_PATH.read_text(encoding="utf-8").split("\n")
                 if ln.strip()]
        entries = [json.loads(ln) for ln in lines]
        assert entries[0]["entry_type"] == "header"

    def test_tier_policy_declares_three_tiers(self):
        pol25 = dataset_policy("pitchgpt_2025_pitcher_disjoint")
        assert pol25 and pol25["tier"] == "budgeted"
        assert pol25["budget"] == 14
        pol24 = dataset_policy("pitchgpt_2024")
        assert pol24 and pol24["tier"] == "burned-dev"
        box = dataset_policy("lockbox_2026_full_season")
        assert box and box["tier"] == "lockbox"
        assert box["unsealed"] is False

    def test_backfilled_2025_contacts_present(self):
        """12 reconstructed historical contacts, then (at most) the two
        pre-registered Phase 0.6.2 contacts (#13 evaluation, #14 attribution
        diagnostic).  Nothing else, ever, without a logged override."""
        n = contact_count("pitchgpt_2025_pitcher_disjoint")
        assert 12 <= n <= 14, (
            f"expected 12 backfilled contacts plus at most the two "
            f"pre-registered 0.6.2 contacts, found {n}"
        )
        entries = [
            json.loads(ln)
            for ln in LEDGER_PATH.read_text(encoding="utf-8").split("\n")
            if ln.strip()
        ]
        contacts = [e for e in entries if e.get("entry_type") == "contact"
                    and e.get("dataset") == "pitchgpt_2025_pitcher_disjoint"]
        nums = [c["contact_number"] for c in contacts]
        assert nums == list(range(1, n + 1)), f"contact numbering must be 1..{n}"
        assert all(
            c.get("backfilled_reconstructed") for c in contacts[:12]
        ), "the first 12 contacts are the reconstructed history"
        # the pos-0 fit-on-holdout taint is registered as a FIT contact
        assert any("FIT" in c["purpose"] for c in contacts[:12])
        # contacts 13/14 must be the live, pre-registered Phase 0.6.2 runs
        for c in contacts[12:]:
            assert not c.get("backfilled_reconstructed"), (
                "contacts past #12 must be live entries, not backfills"
            )
            assert "0.6.2" in c.get("purpose", ""), (
                f"contact #{c['contact_number']} is not one of the two "
                f"pre-registered Phase 0.6.2 contacts: {c.get('purpose')!r}"
            )

    def test_lockbox_contact_refused_against_repo_ledger(self):
        """The real 2026 lockbox refuses contacts today."""

        @holdout_access("lockbox_2026_full_season", "must refuse", budget=1)
        def evaluate():  # pragma: no cover
            raise AssertionError("must not run")

        with pytest.raises(HoldoutSealedError):
            evaluate()

    def test_budgeted_2025_within_budget_but_finite(self):
        """The budget is 14 (12 backfills + 0.6.2 eval + attribution
        diagnostic).  Fill a COPY of the real ledger up to 14 and verify the
        15th contact is refused — the real ledger gains no entries from the
        test suite."""
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            led = Path(td) / "ledger.jsonl"
            shutil.copy(LEDGER_PATH, led)
            # top up to the full budget of 14 (works whether the live 0.6.2
            # contacts have landed yet or not)
            while contact_count("pitchgpt_2025_pitcher_disjoint", led) < 14:
                record_contact("pitchgpt_2025_pitcher_disjoint",
                               "simulated budget filler (test copy only)",
                               ledger_path=led)

            @holdout_access("pitchgpt_2025_pitcher_disjoint",
                            "15th contact", budget=14, ledger_path=led)
            def evaluate():  # pragma: no cover
                raise AssertionError("must not run")

            with pytest.raises(HoldoutBudgetExceeded):
                evaluate()


class TestCIGuard:
    """Any eval code touching a registered holdout identifier must either
    route through src.holdout or carry an explicit `# holdout-contact:`
    annotation tying it to the ledger.

    Scope notes (tuned for zero false positives, 2026-08-10):
      * ``TEST_RANGE = (2024, 2024)`` is NOT flagged — 2024 is the declared
        burned/dev tier (unbudgeted).
      * VWR's ``HOLDOUT_YEAR = 2025`` is NOT flagged — retracted model,
        different dataset, not registered in the ledger.
      * The ``val_seasons=[2025, 2026]`` legacy default in
        src/analytics/pitchgpt.py was remediated by WS5.0 (2026-08-10):
        default is now [2024] and holdout seasons require an explicit
        ``allow_holdout_val=True`` opt-in.
    """

    PATTERNS = [
        re.compile(r"^\s*HOLDOUT_SEASON\s*=\s*202[56]\b"),
        re.compile(r"^\s*HOLDOUT_RANGE\s*=\s*\(\s*202[56]\b"),
        re.compile(r"^\s*TEST_SEASONS\s*=\s*\[\s*202[56]\b"),
        re.compile(r"^\s*TEST_RANGE\s*=\s*\(\s*202[56]\b"),
    ]
    IMPORT_RE = re.compile(
        r"(from\s+src\.holdout\s+import|import\s+src\.holdout)"
    )
    ANNOTATION = "holdout-contact:"

    def _scan_files(self):
        yield from (ROOT / "scripts").glob("*.py")
        yield from (ROOT / "src").rglob("*.py")

    def test_no_unregistered_holdout_touch(self):
        violations = []
        for path in self._scan_files():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = path.read_text(encoding="latin-1")
            file_imports_holdout = bool(self.IMPORT_RE.search(text))
            for lineno, line in enumerate(text.split("\n"), 1):
                if not any(p.search(line) for p in self.PATTERNS):
                    continue
                if self.ANNOTATION in line or file_imports_holdout:
                    continue
                violations.append(
                    f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}"
                )
        assert not violations, (
            "HOLDOUT GUARD: the following lines touch a registered holdout "
            "(2025 pitcher-disjoint tier / 2026 lockbox) without routing "
            "through src.holdout (@holdout_access / record_contact) or an "
            "explicit `# holdout-contact:` annotation referencing "
            "docs/holdout_ledger.jsonl. Every evaluation contact must be "
            "budgeted and logged (audit failure class F-C):\n  "
            + "\n  ".join(violations)
        )

    def test_annotated_sites_exist(self):
        """The nine known historical eval scripts stay annotated — if the
        annotation is deleted, the guard above must catch them again."""
        annotated = 0
        for path in (ROOT / "scripts").glob("pitchgpt_*.py"):
            if self.ANNOTATION in path.read_text(encoding="utf-8"):
                annotated += 1
        assert annotated >= 9, (
            f"expected >= 9 annotated pitchgpt eval scripts, found {annotated}"
        )
