"""
PitchGPT v3 — the §5.5 sealed-2026 lockbox grading entry point.

**THIS SCRIPT MUST NOT BE RUN NOW.**  It exists so the one authorised contact
has a single, pre-registered, ledger-gated door, and so ``§8.3`` test 6 can
prove that door is locked.  Two independent guards stand in front of it:

1. ``@holdout_access(dataset="lockbox_2026_full_season", …)`` — refuses with
   :class:`src.holdout.HoldoutSealedError` while the ledger header carries
   ``unsealed: false``.  The refusal happens *before* the function body runs,
   so no 2026 row is read.
2. ``assert_season_policy(..., allow_lockbox=True)`` — the only call site in
   the repo permitted to pass that flag, and it is inside the body that guard
   1 protects.

Per spec §5.5 the contact is atomic and unrepeatable: full 2026 pitcher-
disjoint cohort (no subsample — the seed-42 10K-PA subsample is banned here,
audit finding F-C), 100 samples/PA, horizon 6, T=1.0, empirical baselines
computed in the same pass, one run, no peeking-and-refitting.  Per §7.4 the
verdict is final either way.

Preconditions the operator must satisfy before this is ever invoked:

* the 2026 regular season is over and an ``entry_type=unseal`` line has been
  appended to ``docs/holdout_ledger.jsonl`` with the cohort hash;
* the §6.7 anti-unfailability tightening has already been executed on dev and
  recorded in the spec's §9 deviations log;
* the v3 artifacts and their calibration sidecars are registry-pinned;
* the §6.3 kernel bandwidths are pinned to
  ``models/kce_bandwidths_pitchgpt_v3_dev2024.json`` (§9 entry 14).  The G2
  arm of the grading run MUST go through
  :func:`src.analytics.pitchgpt_v3_gates.graded_skce_test` with those pinned
  values.  §6.3 fixes the bandwidth "before the contact"; the default
  ``skce_test`` path would refit the median heuristic on the 2026
  probabilities, which is a direct spec violation and is why the graded entry
  point raises :class:`BandwidthNotPinnedError` instead of falling back.
  Note also §9 entry 13: the SKCE is computed on a seeded
  ``KCE_SUBSAMPLE``-row draw, so the G2 arm alone does NOT satisfy §5.5's
  "full cohort — no subsample"; every other §6 statistic does, and the G2
  ``n_used`` must be published with its p-values.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.holdout import holdout_access  # noqa: E402

LOCKBOX_DATASET = "lockbox_2026_full_season"

#: Marker the §8.3 test asserts on, so the gating cannot be silently removed.
LOCKBOX_ENTRY_POINT_IS_LEDGER_GATED = True

#: §9 entry 14.  The graded G2 arm may only use bandwidths loaded from the
#: dev-pinned artifact, via ``graded_skce_test`` (which passes
#: ``allow_bandwidth_fit=False``).  Asserted by
#: ``tests/test_pitchgpt_v3.py::test_lockbox_entry_point_pins_kce_bandwidth``.
LOCKBOX_G2_REQUIRES_PINNED_BANDWIDTH = True


def load_graded_kce_bandwidths() -> dict[str, float]:
    """The only bandwidth source the §5.5 grading run may use.

    Reads the dev-pinned §6.3 artifact and refuses anything whose provenance
    is not the 2024 dev cohort.  Safe to call while sealed: it touches no
    2026 data, only the recorded dev artifact.
    """
    from src.analytics.pitchgpt_v3_gates import (
        KCE_PINNED_BANDWIDTH_FILENAME,
        load_pinned_bandwidths,
    )

    path = _ROOT / "models" / KCE_PINNED_BANDWIDTH_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"§6.3 requires the kernel bandwidth to be fixed on the dev cohort "
            f"and recorded BEFORE the contact. {path} is missing, so there is "
            "nothing to replay and the contact must not proceed."
        )
    bandwidths, _ = load_pinned_bandwidths(path)
    return bandwidths


@holdout_access(
    dataset=LOCKBOX_DATASET,
    purpose=(
        "PITCHGPT_V2_SPEC.md §5.5 single pre-registered grading contact for the "
        "v3 factorized stack (G1-G5, §6.8 verdict)"
    ),
    budget=1,
    metrics=[
        "classwise_ECE_per_head", "TACE_per_head", "KCE_p_values",
        "PIT_KS", "PA_marginals", "per_count_state_ECE",
        "position_marginal_gap", "decision_calibration_deciles",
    ],
)
def run_lockbox_2026_grading(**kwargs) -> dict:
    """The one §6 grading run.  Unreachable while the lockbox is sealed."""
    from src.analytics.pitchgpt_v3_data import assert_season_policy

    # Only call site in the repo allowed to pass allow_lockbox=True, and it is
    # behind the ledger gate above.
    assert_season_policy([2026], allow_lockbox=True)
    raise NotImplementedError(
        "The §5.5 grading run body is deliberately not implemented before the "
        "unseal. It must be written against the dev-tier gate-suite module "
        "(src/analytics/pitchgpt_v3_gates.py) with the §6.7-tightened "
        "thresholds recorded in the spec's §9 log, reviewed, and only then "
        "executed exactly once. Its G2 arm must call graded_skce_test() with "
        "load_graded_kce_bandwidths() (§9 entry 14) — never plain skce_test(), "
        "which would refit the §6.3 kernel on the 2026 cohort — and must "
        "publish n_used with every p-value (§9 entry 13)."
    )


def main() -> int:  # pragma: no cover - never run in this program
    raise SystemExit(
        "Refusing to run: the 2026 lockbox is sealed (PITCHGPT_V2_SPEC §5.5, "
        "kill criterion K5). One contact, at season end, after an unseal entry."
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
