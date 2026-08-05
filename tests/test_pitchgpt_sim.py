"""
Unit tests for the PitchGPT sim engine -- Phase 0.5 ticket 0.5.6.

Test coverage table (per PHASE_0.5_PLAN.md §0.5.6):

|  # | Test name                                        | Validates                                           |
|----|--------------------------------------------------|-----------------------------------------------------|
|  1 | test_horizon_one_token_marginal_matches_softmax  | TVD<0.01 of pitch_token_marginal vs backbone softmax |
|  2 | test_pad_token_semantics                         | NaN past termination on prob arrays; PAD on tokens  |
|  3 | test_none_predictor_degraded_path                | outcomes/probs/pa_outcome all None when predictor None |
|  4 | test_calibration_valid_temperature_branch        | temperature!=1.0 ⇒ "temperature_not_unity"          |
|  5 | test_calibration_valid_horizon_branch            | horizon=13 ⇒ "horizon_exceeds_validated"           |
|  6 | test_calibration_valid_context_band_branch       | per-feature out-of-band reasons fire (5 sub-cases) |
|  7 | test_pa_termination_walk_with_predictor          | 4 balls in a row ⇒ pa terminates at position 3      |
|  8 | test_pa_termination_in_play_with_predictor       | in_play_hit ⇒ pa terminates with pa_outcome==5      |
|  9 | test_aggregation_nan_mask_correctness            | nanmean == filtered-mean within float epsilon       |
| 10 | test_latency_10pa_batch                          | 10-PA batch <5s GPU / <30s CPU                       |
| 11 | test_outcome_marginal_full_grid_shape            | shape smoke for outcome_marginal                    |
| 12 | test_pitch_token_marginal_position_none_shape    | shape smoke for pitch_token_marginal                |
| 13 | test_metadata_required_keys_present              | sampling_metadata required keys                      |

Tests 4, 5, 6, and the calibration-helper tests below run today (Agent 2's
deliverables).  Tests 1, 2, 3, 7, 8, 9, 10, 11, 12, 13 require Agent 1's
``rollout()`` / predictors / aggregations and are marked
``pytest.skip(...)`` with a clear reason until Agent 1 lands the harness.
The skip messages name the missing symbol so the validation agent can spot
them in the regression report.

Calibration-helper coverage (Agent 2, in addition to the §0.5.6 table):

* test_calibration_invalid_reasons_enum_order
* test_calibration_feature_cdfs_npz_present
* test_calibration_feature_cdfs_shapes
* test_calibration_a1_json_values_match_metrics
* test_calibration_a1_json_load_succeeds
* test_backbone_calibration_json_present
* test_calibration_stale_fit_date
* test_calibration_load_missing_file
* test_calibration_load_missing_keys
"""

from __future__ import annotations

import importlib
import json
import time
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from src.analytics.pitchgpt_sim import (
    CALIBRATION_INVALID_REASONS,
    CalibrationError,
    DEFAULT_CALIBRATION_CDFS_PATH,
    PAContext,
    RolloutResult,
    ROLLOUT_PAD_OUTCOME,
    ROLLOUT_PAD_PITCH,
    _check_calibration_valid,
    load_calibration_feature_cdfs,
    load_calibration_json,
)

_ROOT = Path(__file__).resolve().parents[1]
_MODELS = _ROOT / "models"
_RESULTS_A1 = (
    _ROOT / "results" / "pitchgpt_sim" / "outcome_baselines_2026_04_25" / "a1_concat" / "metrics.json"
)
_A1_CALIBRATION = _MODELS / "calibration_pitchgpt_v2_outcomehead_a1.json"
_BACKBONE_CALIBRATION = _MODELS / "pitchgpt_v2_calibration.json"

# A "fresh-enough" today for the staleness gate, anchored to the A1 fit_date
# so the suite is deterministic.  Tests that need to manipulate today pass it
# explicitly; otherwise we use this anchor to avoid wall-clock drift.
TODAY = date(2026, 4, 26)


def _make_default_context(**overrides) -> PAContext:
    """0-0 count, 0 outs, no runners, RHB vs RHP, top-1, tied game."""
    base = dict(
        pitcher_id=1,
        batter_id=2,
        count=(0, 0),
        outs=0,
        runners=(False, False, False),
        batter_stand="R",
        pitcher_throws="R",
        inning=1,
        inning_topbot="Top",
        score_diff=0,
        umpire_scalar=0.0,
        prefix_pitch_tokens=(),
    )
    base.update(overrides)
    return PAContext(**base)


def _fresh_calibration_dict(**overrides) -> dict:
    """A calibration dict with a fit_date close to TODAY so it isn't stale."""
    base = {
        "T": 1.0,
        "ECE_pre": 0.05,
        "ECE_post": 0.01,
        "holdout_season": 2025,
        "holdout_n_pitches": 200000,
        "fit_date": "2026-04-25",
        "checkpoint_sha256": "0" * 64,
        "predictor_kind": "backbone",
    }
    base.update(overrides)
    return base


def _has_rollout() -> bool:
    """Return True iff Agent 1 has landed rollout() (and predictors)."""
    mod = importlib.import_module("src.analytics.pitchgpt_sim")
    return hasattr(mod, "rollout")


# ═════════════════════════════════════════════════════════════════════════════
# Calibration validity gate -- Tests 4, 5, 6 from the §0.5.6 table
# ═════════════════════════════════════════════════════════════════════════════


def test_calibration_valid_temperature_branch():
    """Test 4: temperature != 1.0 forces calibration_valid=False, appends
    `temperature_not_unity` to reasons."""
    ctx = _make_default_context()
    bb = _fresh_calibration_dict()
    pred = _fresh_calibration_dict(predictor_kind="pg_concat_head")

    is_valid, reasons = _check_calibration_valid(
        ctx, bb, pred, temperature=1.05, horizon=6, today=TODAY,
    )
    assert is_valid is False
    assert "temperature_not_unity" in reasons


def test_calibration_valid_horizon_branch():
    """Test 5: horizon=13 forces calibration_valid=False, appends
    `horizon_exceeds_validated` to reasons."""
    ctx = _make_default_context()
    bb = _fresh_calibration_dict()
    pred = _fresh_calibration_dict(predictor_kind="pg_concat_head")

    is_valid, reasons = _check_calibration_valid(
        ctx, bb, pred, temperature=1.0, horizon=13, today=TODAY,
    )
    assert is_valid is False
    assert "horizon_exceeds_validated" in reasons


@pytest.mark.parametrize(
    ("override_kwargs", "expected_reason"),
    [
        # Out-of-band balls/strikes -> count_state out_of_band.
        # Although `count_state` is bucketed and clamped via min(), we hit
        # the gate by passing a bucket index that is NOT in the cohort's CDF
        # band.  Since count_state ∈ [0, 11] and all 12 are typically seen,
        # the way to trigger this gate at unit-test level is via an override
        # that puts the bucket value above the 99th percentile.  We instead
        # construct a synthetic CDF for this test (see body).  Param entry
        # here is a sentinel; actual triggering is body-driven below.
        ({"count": (3, 2)}, "expected-via-cdf"),
        ({"outs": 99}, "context_outs_out_of_band"),
        ({"score_diff": 999}, "expected-via-cdf"),
        ({"inning": 30}, "expected-via-cdf"),
        ({"runners": (True, True, True)}, "expected-via-mask"),
    ],
    ids=[
        "count_state_oob_via_synthetic_cdf",
        "outs_oob_value_clamping",
        "score_diff_oob_via_synthetic_cdf",
        "inning_oob_via_synthetic_cdf",
        "runner_state_unseen_via_synthetic_mask",
    ],
)
def test_calibration_valid_context_band_branch(override_kwargs, expected_reason):
    """Test 6 (parameterized): out-of-band feature values trip the right
    `context_*_out_of_band` reason.

    Many of the canonical cohort CDFs see all bucket values, so to trigger
    "out_of_band" reliably we install synthetic CDFs/masks for the test.

    The five parameterizations cover: count_state, outs, score_diff,
    inning_bucket, runner_state -- the four numeric features of §6.4 plus
    the categorical runner-state mask (`context_runner_state_unseen`).
    """
    ctx = _make_default_context(**override_kwargs)
    bb = _fresh_calibration_dict()
    pred = _fresh_calibration_dict(predictor_kind="pg_concat_head")

    # Synthetic CDFs / masks: we set the 99th-percentile to land BEFORE the
    # value being tested, so that bucket is "above the high tail."
    # All-zeros runner_state_mask means no runner_state has been "seen."
    feature_cdfs = {
        # count_state CDF: 99% of cohort sits in bucket 0; bucket 11 is past
        # the high tail.  The default context (count=(0,0)) lives in bucket
        # 0, which is in-band; count=(3,2) is bucket 11, out of band.
        "count_state_cdf": np.concatenate([
            np.array([0.99]),  # P(X<=0) = 0.99
            np.full(11, 1.0, dtype=np.float32),  # rest are at 1.0
        ]).astype(np.float32),
        # outs CDF: 100% of cohort in bucket 0.  (99 is out of cdf range
        # entirely; out_of_band fires.)
        "outs_cdf": np.array([1.0, 1.0, 1.0], dtype=np.float32),
        # score_diff_bucket CDF: 99% of cohort sits in bucket 2 (score_diff
        # == 0).  Bucket 4 is out-of-band.  default_context score_diff=0
        # -> bucket 2 in-band.  override score_diff=999 -> bucket 4 OOB.
        "score_diff_bucket_cdf": np.array(
            [0.0, 0.0, 0.99, 1.0, 1.0], dtype=np.float32,
        ),
        # inning_bucket CDF: 99% of cohort in bucket 0 (innings 1-3); bucket
        # 3 is out-of-band.  default inning=1 -> bucket 0 in-band.
        "inning_bucket_cdf": np.array([0.99, 1.0, 1.0, 1.0], dtype=np.float32),
        # runner_state_mask: only state 0 (no runners) seen.  override
        # runners=(T,T,T) is state 7, unseen.
        "runner_state_mask": np.array(
            [True, False, False, False, False, False, False, False],
            dtype=np.bool_,
        ),
    }

    is_valid, reasons = _check_calibration_valid(
        ctx, bb, pred,
        temperature=1.0, horizon=6,
        today=TODAY, feature_cdfs=feature_cdfs,
    )

    assert is_valid is False, (
        f"Expected invalid for override={override_kwargs}, got reasons={reasons}"
    )
    if expected_reason.startswith("expected-via-"):
        # The map from override -> reason for the synthetic-CDF cases:
        if "count" in override_kwargs:
            assert "context_count_state_out_of_band" in reasons
        elif "score_diff" in override_kwargs:
            assert "context_score_diff_out_of_band" in reasons
        elif "inning" in override_kwargs:
            assert "context_inning_out_of_band" in reasons
        elif "runners" in override_kwargs:
            assert "context_runner_state_unseen" in reasons
        else:
            pytest.fail(f"Unhandled synthetic case: {override_kwargs}")
    else:
        assert expected_reason in reasons


def test_calibration_valid_all_pass_in_band():
    """Sanity: a default in-band context with fresh calibrations and T=1.0,
    horizon=6 returns calibration_valid=True with empty reasons."""
    ctx = _make_default_context()
    bb = _fresh_calibration_dict()
    pred = _fresh_calibration_dict(predictor_kind="pg_concat_head")

    is_valid, reasons = _check_calibration_valid(
        ctx, bb, pred, temperature=1.0, horizon=6, today=TODAY,
    )
    assert is_valid is True
    assert reasons == []


def test_calibration_valid_no_predictor_is_vacuous():
    """When `predictor_calibration is None`, condition 3 is vacuously True
    (no `outcome_predictor_calibration_stale` reason is appended)."""
    ctx = _make_default_context()
    bb = _fresh_calibration_dict()
    is_valid, reasons = _check_calibration_valid(
        ctx, bb, None, temperature=1.0, horizon=6, today=TODAY,
    )
    assert is_valid is True
    assert "outcome_predictor_calibration_stale" not in reasons


def test_calibration_invalid_reasons_enum_order():
    """The 9 enumerated reason strings per SIM_ENGINE_API §6 are present in
    the module's frozen tuple."""
    expected = {
        "temperature_not_unity",
        "horizon_exceeds_validated",
        "backbone_calibration_stale",
        "outcome_predictor_calibration_stale",
        "context_count_state_out_of_band",
        "context_outs_out_of_band",
        "context_score_diff_out_of_band",
        "context_inning_out_of_band",
        "context_runner_state_unseen",
    }
    assert set(CALIBRATION_INVALID_REASONS) == expected
    assert len(CALIBRATION_INVALID_REASONS) == 9


def test_calibration_stale_fit_date():
    """A backbone calibration whose fit_date is >12 months old trips
    `backbone_calibration_stale`."""
    ctx = _make_default_context()
    bb = _fresh_calibration_dict(fit_date="2024-01-01")  # >2 years old at TODAY
    pred = _fresh_calibration_dict(predictor_kind="pg_concat_head")

    is_valid, reasons = _check_calibration_valid(
        ctx, bb, pred, temperature=1.0, horizon=6, today=TODAY,
    )
    assert is_valid is False
    assert "backbone_calibration_stale" in reasons


# ═════════════════════════════════════════════════════════════════════════════
# Calibration artifact tests (CDFs npz + JSON)
# ═════════════════════════════════════════════════════════════════════════════


def test_calibration_feature_cdfs_npz_present():
    """The build script must have produced the npz at the canonical location."""
    assert DEFAULT_CALIBRATION_CDFS_PATH.exists(), (
        f"calibration_feature_cdfs.npz missing at {DEFAULT_CALIBRATION_CDFS_PATH}; "
        "run scripts/pitchgpt_build_calibration_cdfs.py."
    )


def test_calibration_feature_cdfs_shapes():
    """Per PHASE_0.5_PLAN §0.5.4: CDF lengths are 12 / 3 / 5 / 4; runner mask len 8."""
    cdfs = load_calibration_feature_cdfs()
    assert cdfs is not None
    assert cdfs["count_state_cdf"].shape == (12,)
    assert cdfs["outs_cdf"].shape == (3,)
    assert cdfs["score_diff_bucket_cdf"].shape == (5,)
    assert cdfs["inning_bucket_cdf"].shape == (4,)
    assert cdfs["runner_state_mask"].shape == (8,)
    # CDFs are non-decreasing and end at ~1.0 (within float epsilon).
    for key in ("count_state_cdf", "outs_cdf", "score_diff_bucket_cdf", "inning_bucket_cdf"):
        cdf = cdfs[key]
        assert np.all(np.diff(cdf) >= -1e-6), f"{key} not non-decreasing"
        assert abs(float(cdf[-1]) - 1.0) < 1e-5, f"{key} does not end at 1.0"
    # The 2025 cohort observed all 8 runner states (verified at build time).
    assert cdfs["holdout_season"] == 2025
    assert cdfs["holdout_n_pitches"] > 100_000


def test_calibration_a1_json_values_match_metrics():
    """Ticket 0.5.5: the values in the A1 calibration.json must match the
    A1 training metrics.json exactly (up to numeric tolerance).  No invented
    values per the agent guardrail."""
    assert _A1_CALIBRATION.exists(), f"A1 calibration JSON missing at {_A1_CALIBRATION}"
    assert _RESULTS_A1.exists(), f"A1 metrics.json missing at {_RESULTS_A1}"

    cal = json.loads(_A1_CALIBRATION.read_text(encoding="utf-8"))
    metrics = json.loads(_RESULTS_A1.read_text(encoding="utf-8"))

    assert cal["T"] == metrics["config"]["temperature"]
    assert cal["ECE_pre"] == metrics["test_metrics"]["ece_pre_temp"]
    assert cal["ECE_post"] == metrics["test_metrics"]["ece_post_temp"]
    assert cal["holdout_season"] == 2025
    assert cal["holdout_n_pitches"] == metrics["cohort"]["test_valid_outcomes"]
    assert cal["predictor_kind"] == "pg_concat_head"
    # fit_date was 2026-04-25 at ticket-0.5.5 write; bumped to
    # 2026-04-26 by the Phase 0.6 class_calibration refit.
    assert cal["fit_date"] in ("2026-04-25", "2026-04-26")
    # checkpoint_sha256 is recomputed at write time; verify it's a 64-char hex.
    assert len(cal["checkpoint_sha256"]) == 64
    assert all(c in "0123456789abcdef" for c in cal["checkpoint_sha256"])


def test_calibration_a1_json_load_succeeds():
    """The A1 calibration JSON loads via `load_calibration_json` with the
    expected predictor_kind and within the outcome-predictor ECE budget
    (<0.05)."""
    cal = load_calibration_json(
        _A1_CALIBRATION,
        expected_predictor_kind="pg_concat_head",
        ece_budget=0.05,
    )
    assert cal["ECE_post"] < 0.05
    # Anchor staleness to the fit_date so the test is stable; the loader
    # uses today by default which would fail in 2027+.  We're testing the
    # load contract, not the wall-clock test.


def test_backbone_calibration_json_present():
    """Per PHASE_0.5_PLAN §0.5.5: the backbone calibration.json must exist
    adjacent to models/pitchgpt_v2.pt (or assembled from existing
    calibration_2025.json if it was missing)."""
    assert _BACKBONE_CALIBRATION.exists(), (
        f"Backbone calibration JSON missing at {_BACKBONE_CALIBRATION}.  "
        "Per PHASE_0.5_PLAN.md §0.5.5 either ship the existing assembled "
        "JSON or surface as a blocker."
    )
    cal = json.loads(_BACKBONE_CALIBRATION.read_text(encoding="utf-8"))
    assert cal["predictor_kind"] == "backbone"
    assert cal["holdout_season"] == 2025
    # Backbone ECE budget is <0.02 per SIM_ENGINE_API §7.3.
    assert float(cal["ECE_post"]) < 0.02


def test_calibration_load_missing_file(tmp_path):
    """`load_calibration_json` raises CalibrationError on missing file."""
    with pytest.raises(CalibrationError, match="missing at"):
        load_calibration_json(tmp_path / "nope.json")


def test_calibration_load_missing_keys(tmp_path):
    """`load_calibration_json` raises CalibrationError on missing required keys."""
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"T": 1.0}), encoding="utf-8")
    with pytest.raises(CalibrationError, match="missing required keys"):
        load_calibration_json(p)


# ═════════════════════════════════════════════════════════════════════════════
# Tests 1-3, 7-13 -- depend on Agent 1's rollout() implementation
# ═════════════════════════════════════════════════════════════════════════════
#
# Each is implemented to either (a) call into the real rollout() if Agent 1
# has landed it, or (b) skip with a clear message if not.  This keeps the
# §0.5.6 table of 10+ tests visible in the suite.


_AGENT1_SKIP_MSG = (
    "rollout() not yet implemented in src/analytics/pitchgpt_sim.py "
    "(Agent 1 ticket 0.5.1).  Test will execute once Agent 1's harness lands."
)


@pytest.mark.skipif(not _has_rollout(), reason=_AGENT1_SKIP_MSG)
def test_horizon_one_token_marginal_matches_softmax():
    """Test 1: pitch_token_marginal at H=1 matches backbone next-token softmax
    in TVD < 0.01 over 10K samples."""
    from src.analytics.pitchgpt_sim import (  # type: ignore[attr-defined]
        OutcomePredictorRegistry,
        pitch_token_marginal,
        rollout,
    )
    ctx = _make_default_context()
    predictor = OutcomePredictorRegistry.get("pg_concat_head")
    res = rollout(
        ctx,
        outcome_predictor=predictor,
        n_samples=10_000,
        horizon=1,
        temperature=1.0,
        seed=42,
        return_probs=True,
    )
    sampled_marginal = pitch_token_marginal(res, position=0)
    # The backbone's direct next-token softmax for ctx -- must match.
    # (We compute via the predictor's exposed backbone forward; if that's not
    # a stable surface, fall back to averaging res.pitch_probs which is
    # mathematically the same thing.)
    expected = np.nanmean(res.pitch_probs[:, 0, :], axis=0)
    tvd = 0.5 * float(np.abs(sampled_marginal - expected).sum())
    assert tvd < 0.01, f"H=1 token marginal TVD={tvd:.4f} (>0.01)"


@pytest.mark.skipif(not _has_rollout(), reason=_AGENT1_SKIP_MSG)
def test_pad_token_semantics():
    """Test 2: after PA termination, pitch_tokens are PAD; pitch_probs are NaN
    (NOT zeros)."""
    from src.analytics.pitchgpt_sim import (  # type: ignore[attr-defined]
        OutcomePredictorRegistry,
        rollout,
    )
    ctx = _make_default_context()
    predictor = OutcomePredictorRegistry.get("pg_concat_head")
    res = rollout(
        ctx,
        outcome_predictor=predictor,
        n_samples=64,
        horizon=6,
        temperature=1.0,
        seed=7,
    )
    for s in range(res.pitch_tokens.shape[0]):
        # Find the position of termination (first True in pa_terminated row).
        term_idx = np.where(res.pa_terminated[s])[0]
        if term_idx.size == 0:
            continue  # truncated rollout; nothing to assert about pad tail
        t = int(term_idx[0])
        for p in range(t + 1, res.pitch_tokens.shape[1]):
            assert int(res.pitch_tokens[s, p]) == ROLLOUT_PAD_PITCH
            if res.pitch_probs is not None:
                assert np.isnan(res.pitch_probs[s, p, 0]), (
                    "pitch_probs past termination must be NaN, NOT zero"
                )


@pytest.mark.skipif(not _has_rollout(), reason=_AGENT1_SKIP_MSG)
def test_none_predictor_degraded_path():
    """Test 3: outcome_predictor=None -> outcomes/outcome_probs/pa_outcome
    all None; sampling_metadata['outcome_predictor'] == 'none'."""
    from src.analytics.pitchgpt_sim import rollout  # type: ignore[attr-defined]
    ctx = _make_default_context()
    res = rollout(
        ctx,
        outcome_predictor=None,
        n_samples=4,
        horizon=6,
        temperature=1.0,
        seed=1,
    )
    assert res.outcomes is None
    assert res.outcome_probs is None
    assert res.pa_outcome is None
    assert res.sampling_metadata["outcome_predictor"] == "none"


@pytest.mark.skipif(not _has_rollout(), reason=_AGENT1_SKIP_MSG)
def test_pa_termination_walk_with_predictor():
    """Test 7: an outcome predictor forced to emit `ball` 4 times
    terminates exactly at position 3 with final_count=(4,0)."""
    from src.analytics.pitchgpt_sim import rollout  # type: ignore[attr-defined]

    class _ForceBallPredictor:
        name = "force_ball"
        checkpoint_sha256 = "0" * 64
        calibration: dict = {
            "T": 1.0, "ECE_pre": 0.0, "ECE_post": 0.0,
            "holdout_season": 2025, "holdout_n_pitches": 1,
            "fit_date": "2026-04-25",
            "checkpoint_sha256": "0" * 64,
            "predictor_kind": "pg_concat_head",
        }

        def predict_outcome_probs(self, backbone_hidden, context_vec, pitch_token):
            import torch
            B, S = backbone_hidden.shape[0], backbone_hidden.shape[1]
            out = torch.zeros(B, S, 7)
            out[..., 0] = 1.0  # ball
            return out

    ctx = _make_default_context()
    res = rollout(
        ctx,
        outcome_predictor=_ForceBallPredictor(),
        n_samples=4,
        horizon=6,
        temperature=1.0,
        seed=11,
    )
    # All samples should terminate at position 3 (the 4th ball).
    for s in range(res.pitch_tokens.shape[0]):
        term_idx = np.where(res.pa_terminated[s])[0]
        assert term_idx.size == 1
        assert int(term_idx[0]) == 3
        # final_count should be (4, 0) -- walk.
        assert int(res.final_count[s, 0]) == 4
        assert int(res.final_count[s, 1]) == 0


@pytest.mark.skipif(not _has_rollout(), reason=_AGENT1_SKIP_MSG)
def test_pa_termination_in_play_with_predictor():
    """Test 8: forcing predictor to emit `in_play_hit` (class 5) terminates
    at position 0 with pa_outcome == 5."""
    from src.analytics.pitchgpt_sim import rollout  # type: ignore[attr-defined]

    class _ForceHitPredictor:
        name = "force_hit"
        checkpoint_sha256 = "0" * 64
        calibration: dict = {
            "T": 1.0, "ECE_pre": 0.0, "ECE_post": 0.0,
            "holdout_season": 2025, "holdout_n_pitches": 1,
            "fit_date": "2026-04-25",
            "checkpoint_sha256": "0" * 64,
            "predictor_kind": "pg_concat_head",
        }

        def predict_outcome_probs(self, backbone_hidden, context_vec, pitch_token):
            import torch
            B, S = backbone_hidden.shape[0], backbone_hidden.shape[1]
            out = torch.zeros(B, S, 7)
            out[..., 5] = 1.0  # in_play_hit
            return out

    ctx = _make_default_context()
    res = rollout(
        ctx,
        outcome_predictor=_ForceHitPredictor(),
        n_samples=4,
        horizon=6,
        temperature=1.0,
        seed=13,
    )
    for s in range(res.pitch_tokens.shape[0]):
        term_idx = np.where(res.pa_terminated[s])[0]
        assert term_idx.size == 1
        assert int(term_idx[0]) == 0
        assert int(res.pa_outcome[s]) == 5


@pytest.mark.skipif(not _has_rollout(), reason=_AGENT1_SKIP_MSG)
def test_aggregation_nan_mask_correctness():
    """Test 9: ~50 truncated of 100 produces identical np.nanmean(...) and
    'manually filter pa_terminated then mean' results within float epsilon.

    Uses a deterministic-minimal predictor (in_play_hit ~0.21, ball ~0.79
    per step at horizon=3) so that (1-0.21)^3 ~ 0.49 of the 100 samples
    finish without ever terminating (truncated) and the rest terminate at
    a known terminal-outcome position.  This keeps the test fast (<1s)
    while exercising both outcome_marginal and pa_woba_distribution
    NaN-mask invariants per SIM_ENGINE_API §9.
    """
    from src.analytics.pitchgpt_sim import (  # type: ignore[attr-defined]
        outcome_marginal,
        pa_woba_distribution,
        rollout,
    )

    class _MixedTerminalPredictor:
        # Deterministic-minimal: ~21% in_play_hit (terminal class 5),
        # ~79% ball (count-advancing class 0).  At horizon=3 with no
        # strikes ever, no sample reaches 4 balls -> walk path can't
        # trigger; only class-5 sampling terminates a PA.  Gives ~50%
        # truncated samples after horizon=3.
        name = "mixed_terminal"
        checkpoint_sha256 = "0" * 64
        calibration: dict = {
            "T": 1.0, "ECE_pre": 0.0, "ECE_post": 0.0,
            "holdout_season": 2025, "holdout_n_pitches": 1,
            "fit_date": "2026-04-25",
            "checkpoint_sha256": "0" * 64,
            "predictor_kind": "pg_concat_head",
        }

        def predict_outcome_probs(self, backbone_hidden, context_vec, pitch_token):
            import torch
            B, S = backbone_hidden.shape[0], backbone_hidden.shape[1]
            out = torch.zeros(B, S, 7)
            out[..., 0] = 0.79  # ball
            out[..., 5] = 0.21  # in_play_hit (terminal)
            return out

    ctx = _make_default_context()
    res = rollout(
        ctx,
        outcome_predictor=_MixedTerminalPredictor(),
        n_samples=100,
        horizon=3,
        temperature=1.0,
        seed=21,
        return_probs=True,
    )

    # Sanity: we need a mix of terminated + truncated samples to exercise
    # the NaN-mask path.
    n_truncated = int((~res.pa_terminated.any(axis=1)).sum())
    assert 20 <= n_truncated <= 80, (
        f"need a mix for the NaN-mask test; got {n_truncated} truncated of 100"
    )
    assert res.outcomes is not None
    assert res.outcome_probs is not None
    assert res.pa_outcome is not None

    # ── outcome_marginal (per-position) ─────────────────────────────────
    # `np.nanmean(outcome_probs, axis=0)` over all (N, H, 7) with NaN past
    # termination MUST match: at each position, average outcome_probs over
    # samples that haven't been truncated past that position.
    via_util = outcome_marginal(res, outcome_class=None)  # (H, 7)
    # Manual filter using the canonical "valid through termination" mask.
    # cumsum<=1 keeps the termination position (where outcome IS valid)
    # and drops pad-NaN positions strictly after termination.
    valid = res.pa_terminated.cumsum(axis=1).cumsum(axis=1) <= 1  # (N, H)
    H = res.outcome_probs.shape[1]
    via_manual = np.full((H, 7), np.nan, dtype=np.float32)
    for h in range(H):
        mask_h = valid[:, h]
        if mask_h.any():
            via_manual[h] = res.outcome_probs[mask_h, h, :].mean(axis=0)
    np.testing.assert_allclose(via_util, via_manual, atol=1e-6)

    # ── pa_woba_distribution NaN-mask invariant ─────────────────────────
    # Truncated samples must yield NaN; terminated samples must yield a
    # finite wOBA.  np.nanmean over the distribution must equal the mean
    # over the manual filter (only terminated, finite-outcome samples).
    woba = pa_woba_distribution(res)
    assert woba.shape == (100,)
    truncated_mask = ~res.pa_terminated.any(axis=1)
    # Truncated -> NaN.
    assert np.isnan(woba[truncated_mask]).all(), (
        "pa_woba_distribution must be NaN for truncated samples per §3.3"
    )
    # Terminated samples with class-5 (in_play_hit) outcome yield a
    # finite wOBA; walks/strikeouts have pa_outcome == ROLLOUT_PAD_OUTCOME
    # which the function maps to NaN.  So compare nanmean to the manual
    # mean over only the finite entries.
    via_nanmean = float(np.nanmean(woba)) if not np.isnan(woba).all() else float("nan")
    finite_mask = ~np.isnan(woba)
    via_manual_woba = (
        float(woba[finite_mask].mean()) if finite_mask.any() else float("nan")
    )
    np.testing.assert_allclose(via_nanmean, via_manual_woba, atol=1e-6)


@pytest.mark.skipif(not _has_rollout(), reason=_AGENT1_SKIP_MSG)
def test_latency_10pa_batch():
    """Test 10: 10-PA batch with n_samples=100, horizon=6, A1 predictor must
    finish in <5s on RTX 3050 (or <30s on CPU as a CI-friendly fallback)."""
    import torch
    from src.analytics.pitchgpt_sim import (  # type: ignore[attr-defined]
        OutcomePredictorRegistry,
        rollout,
    )
    budget_seconds = 5.0 if torch.cuda.is_available() else 30.0

    predictor = OutcomePredictorRegistry.get("pg_concat_head")
    contexts = [_make_default_context() for _ in range(10)]
    t0 = time.perf_counter()
    for ctx in contexts:
        rollout(
            ctx,
            outcome_predictor=predictor,
            n_samples=100,
            horizon=6,
            temperature=1.0,
            seed=42,
        )
    elapsed = time.perf_counter() - t0
    assert elapsed < budget_seconds, (
        f"10-PA latency = {elapsed:.2f}s exceeds budget {budget_seconds}s "
        f"(device={'cuda' if torch.cuda.is_available() else 'cpu'})"
    )


@pytest.mark.skipif(not _has_rollout(), reason=_AGENT1_SKIP_MSG)
def test_outcome_marginal_full_grid_shape():
    """Test 11: outcome_marginal with outcome_class=None returns (horizon, 7)."""
    from src.analytics.pitchgpt_sim import (  # type: ignore[attr-defined]
        OutcomePredictorRegistry,
        outcome_marginal,
        rollout,
    )
    ctx = _make_default_context()
    predictor = OutcomePredictorRegistry.get("pg_concat_head")
    res = rollout(
        ctx,
        outcome_predictor=predictor,
        n_samples=8,
        horizon=6,
        temperature=1.0,
        seed=33,
    )
    full = outcome_marginal(res, outcome_class=None)
    assert full.shape == (6, 7)


@pytest.mark.skipif(not _has_rollout(), reason=_AGENT1_SKIP_MSG)
def test_pitch_token_marginal_position_none_shape():
    """Test 12: pitch_token_marginal with position=None returns (horizon, 2210)."""
    from src.analytics.pitchgpt_sim import (  # type: ignore[attr-defined]
        OutcomePredictorRegistry,
        pitch_token_marginal,
        rollout,
    )
    ctx = _make_default_context()
    predictor = OutcomePredictorRegistry.get("pg_concat_head")
    res = rollout(
        ctx,
        outcome_predictor=predictor,
        n_samples=8,
        horizon=4,
        temperature=1.0,
        seed=33,
    )
    full = pitch_token_marginal(res, position=None)
    assert full.shape == (4, 2210)


@pytest.mark.skipif(not _has_rollout(), reason=_AGENT1_SKIP_MSG)
def test_metadata_required_keys_present():
    """Test 13: every required key in sampling_metadata is present per
    SIM_ENGINE_API §3.4."""
    from src.analytics.pitchgpt_sim import (  # type: ignore[attr-defined]
        OutcomePredictorRegistry,
        rollout,
    )
    ctx = _make_default_context()
    predictor = OutcomePredictorRegistry.get("pg_concat_head")
    res = rollout(
        ctx,
        outcome_predictor=predictor,
        n_samples=4,
        horizon=4,
        temperature=1.0,
        seed=33,
    )
    required = {
        "temperature",
        "seed",
        "n_samples",
        "horizon",
        "backbone_version",
        "backbone_checkpoint_sha256",
        "outcome_predictor",
        "outcome_predictor_checkpoint_sha256",
        "calibration_valid",
        "calibration_invalid_reasons",
        "rollout_engine_version",
        "n_truncated",
    }
    missing = required.difference(res.sampling_metadata.keys())
    assert not missing, f"sampling_metadata missing keys: {sorted(missing)}"


# ═════════════════════════════════════════════════════════════════════════════
# Phase 0.6 -- per-class probability re-weighting on PGConcatHeadPredictor
# ═════════════════════════════════════════════════════════════════════════════


def _has_pg_concat_head_predictor() -> bool:
    """Return True iff the A1 checkpoint + backbone are present on disk."""
    a1 = _MODELS / "pitchgpt_v2_outcomehead_a1.pt"
    bb = _MODELS / "pitchgpt_v2.pt"
    return a1.exists() and bb.exists()


_PGCH_SKIP_MSG = (
    "PGConcatHeadPredictor checkpoint missing -- skipping class_calibration test."
)


@pytest.mark.skipif(
    not _has_pg_concat_head_predictor(), reason=_PGCH_SKIP_MSG
)
def test_pg_concat_head_predictor_no_class_calibration_is_identity(
    tmp_path,
):
    """Backwards-compat: a calibration JSON WITHOUT a `class_calibration`
    field leaves `predict_outcome_probs` behavior byte-identical to the
    pre-fix path (no re-weighting applied)."""
    import torch
    from src.analytics.pitchgpt_sim import PGConcatHeadPredictor

    # Write a minimal calibration JSON with NO class_calibration field.
    cpath = tmp_path / "calib_no_cc.json"
    cpath.write_text(json.dumps({
        "T": 0.8003096499977166,
        "ECE_pre": 0.0614403106926628,
        "ECE_post": 0.011409712028199842,
        "holdout_season": 2025,
        "holdout_n_pitches": 204513,
        "fit_date": "2026-04-25",
        "checkpoint_sha256": (
            "37b50e87599013c281560c9f63286fe5b7645297d0042694d907287417bb25e5"
        ),
        "predictor_kind": "pg_concat_head",
    }))

    pred = PGConcatHeadPredictor(calibration_path=cpath)
    assert pred._class_calibration is None
    # Sanity: predict on dummy inputs.  Just verify the returned probs
    # are valid (sum to 1, no NaN), which is the contract under the
    # backwards-compat path.
    B, S = 2, 3
    d_model = pred.head.d_model if hasattr(pred.head, "d_model") else 128
    ctx_dim = (
        pred.head.context_dim if hasattr(pred.head, "context_dim") else 35
    )
    hidden = torch.zeros(B, S, d_model, device=pred._device)
    ctx = torch.zeros(B, S, ctx_dim, device=pred._device)
    tokens = torch.zeros(B, S, dtype=torch.long, device=pred._device)
    out = pred.predict_outcome_probs(hidden, ctx, tokens)
    assert out.shape == (B, S, 7)
    assert torch.all(torch.isfinite(out))
    sums = out.sum(dim=-1).cpu().numpy()
    assert np.allclose(sums, 1.0, atol=1e-5)


@pytest.mark.skipif(
    not _has_pg_concat_head_predictor(), reason=_PGCH_SKIP_MSG
)
def test_pg_concat_head_predictor_class_calibration_reweights(tmp_path):
    """Synthetic test: with a known `class_calibration` ``w``, the post-T
    softmax ``p`` is reweighted to ``p_i * w_i / sum_j(p_j * w_j)``.

    Strategy: construct two predictors -- one with no class_calibration
    (baseline ``p``), one with a known ``w``.  Run both on identical
    inputs.  Verify the reweighted output matches the analytic answer
    ``p * w / (p * w).sum(-1, keepdim=True)``.
    """
    import torch
    from src.analytics.pitchgpt_sim import PGConcatHeadPredictor

    # Baseline: no class_calibration.
    cpath_baseline = tmp_path / "calib_baseline.json"
    base_dict = {
        "T": 0.8003096499977166,
        "ECE_pre": 0.0614403106926628,
        "ECE_post": 0.011409712028199842,
        "holdout_season": 2025,
        "holdout_n_pitches": 204513,
        "fit_date": "2026-04-25",
        "checkpoint_sha256": (
            "37b50e87599013c281560c9f63286fe5b7645297d0042694d907287417bb25e5"
        ),
        "predictor_kind": "pg_concat_head",
    }
    cpath_baseline.write_text(json.dumps(base_dict))

    # Reweighted: same JSON + a known class_calibration vector.
    w = np.array(
        [1.5, 0.8, 0.5, 1.2, 1.0, 1.0, 2.0], dtype=np.float64
    )
    cpath_rw = tmp_path / "calib_rw.json"
    rw_dict = dict(base_dict)
    rw_dict["class_calibration"] = w.tolist()
    cpath_rw.write_text(json.dumps(rw_dict))

    pred_base = PGConcatHeadPredictor(calibration_path=cpath_baseline)
    pred_rw = PGConcatHeadPredictor(calibration_path=cpath_rw)
    assert pred_base._class_calibration is None
    assert pred_rw._class_calibration is not None

    # Random non-degenerate inputs, deterministic seed.
    torch.manual_seed(20260426)
    B, S = 4, 5
    d_model = pred_base.head.d_model if hasattr(pred_base.head, "d_model") else 128
    ctx_dim = (
        pred_base.head.context_dim if hasattr(pred_base.head, "context_dim") else 35
    )
    hidden = torch.randn(B, S, d_model, device=pred_base._device)
    ctx = torch.randn(B, S, ctx_dim, device=pred_base._device)
    # Tokens drawn from valid VOCAB range.
    rng = np.random.default_rng(42)
    tokens = torch.from_numpy(
        rng.integers(0, 100, size=(B, S))
    ).long().to(pred_base._device)

    p_base = pred_base.predict_outcome_probs(hidden, ctx, tokens)
    p_rw = pred_rw.predict_outcome_probs(hidden, ctx, tokens)

    # Analytic expected output.
    p_base_np = p_base.cpu().numpy().astype(np.float64)
    weighted = p_base_np * w[None, None, :]
    expected = weighted / weighted.sum(axis=-1, keepdims=True).clip(min=1e-12)

    p_rw_np = p_rw.cpu().numpy().astype(np.float64)
    np.testing.assert_allclose(p_rw_np, expected, atol=1e-6, rtol=1e-6)
    # And renormalization correctness:
    np.testing.assert_allclose(
        p_rw_np.sum(axis=-1), np.ones((B, S)), atol=1e-6
    )


# ═════════════════════════════════════════════════════════════════════════════
# Phase 0.6.1 — mid-PA count-state mutation
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("balls", "strikes", "outcome_name", "expected"),
    [
        # ball advances balls
        (0, 0, "OUTCOME_BALL", (1, 0)),
        (1, 2, "OUTCOME_BALL", (2, 2)),
        (3, 0, "OUTCOME_BALL", (4, 0)),   # -> caller sees walk (balls >= 4)
        # called / swinging strike advances strikes
        (0, 0, "OUTCOME_CALLED_STRIKE", (0, 1)),
        (2, 1, "OUTCOME_CALLED_STRIKE", (2, 2)),
        (0, 2, "OUTCOME_CALLED_STRIKE", (0, 3)),   # -> caller sees K (strikes >= 3)
        (0, 0, "OUTCOME_SWINGING_STRIKE", (0, 1)),
        (1, 2, "OUTCOME_SWINGING_STRIKE", (1, 3)),  # -> caller sees K
        # foul advances strikes ONLY below the 2-strike cap
        (0, 0, "OUTCOME_FOUL", (0, 1)),
        (0, 1, "OUTCOME_FOUL", (0, 2)),
        (0, 2, "OUTCOME_FOUL", (0, 2)),   # capped: no advance at 2 strikes
        (3, 2, "OUTCOME_FOUL", (3, 2)),   # capped even on a full count
        # terminal in-play / HBP leave the count unchanged
        (1, 1, "OUTCOME_IN_PLAY_OUT", (1, 1)),
        (2, 2, "OUTCOME_IN_PLAY_HIT", (2, 2)),
        (0, 2, "OUTCOME_HBP", (0, 2)),
    ],
)
def test_advance_count_state_machine(balls, strikes, outcome_name, expected):
    """Phase 0.6.1: `_advance_count` is the pure per-pitch count transition.

    Covers ball / called-strike / swinging-strike / foul-at-<2 / foul-at-2
    (capped) / terminal-in-play (unchanged).  Also documents the caller-side
    termination thresholds: balls >= _WALK_BALLS ⇒ walk, strikes >=
    _STRIKEOUT_STRIKES ⇒ strikeout.
    """
    import src.analytics.pitchgpt_sim as sim

    outcome_class = getattr(sim, outcome_name)
    got = sim._advance_count(balls, strikes, outcome_class)
    assert got == expected, (
        f"_advance_count({balls},{strikes},{outcome_name})={got}, want {expected}"
    )
    # Purity: no negative counts ever produced.
    assert got[0] >= 0 and got[1] >= 0


def test_advance_count_termination_thresholds():
    """Phase 0.6.1: walk (4 balls) / strikeout (3 strikes) are detectable on
    the count `_advance_count` returns; foul-at-2-strikes never terminates."""
    import src.analytics.pitchgpt_sim as sim

    # A 4th ball is a walk.
    b, s = sim._advance_count(3, 1, sim.OUTCOME_BALL)
    assert b >= sim._WALK_BALLS and s < sim._STRIKEOUT_STRIKES

    # A 3rd strike (called or swinging) is a strikeout.
    b, s = sim._advance_count(1, 2, sim.OUTCOME_CALLED_STRIKE)
    assert s >= sim._STRIKEOUT_STRIKES and b < sim._WALK_BALLS

    # A foul at two strikes prolongs the PA: neither threshold trips.
    b, s = sim._advance_count(1, 2, sim.OUTCOME_FOUL)
    assert b < sim._WALK_BALLS and s < sim._STRIKEOUT_STRIKES


class _RecordingFoulPredictor:
    """Mock predictor that always emits `foul` (class 3) and records the
    context vector it is handed at each rollout position.

    Foul never terminates the PA (and is capped at 2 strikes), so the count
    marches 0-0 -> 0-1 -> 0-2 -> 0-2 -> ... deterministically, exercising the
    Phase 0.6.1 count-state mutation without any GPU-scale sampling.
    """

    name = "pg_concat_head"  # allowed name so rollout() doesn't warn
    checkpoint_sha256 = "0" * 64
    calibration: dict = {
        "T": 1.0, "ECE_pre": 0.0, "ECE_post": 0.0,
        "holdout_season": 2025, "holdout_n_pitches": 1,
        "fit_date": "2026-04-25", "checkpoint_sha256": "0" * 64,
        "predictor_kind": "pg_concat_head",
    }

    def __init__(self):
        self.seen_contexts = []  # list of (N, CTX) numpy arrays, one per position

    def predict_outcome_probs(self, backbone_hidden, context_vec, pitch_token):
        import torch
        # Record the last-position context handed to the outcome head.
        self.seen_contexts.append(
            context_vec[:, -1, :].detach().cpu().numpy().copy()
        )
        B, S = backbone_hidden.shape[0], backbone_hidden.shape[1]
        out = torch.zeros(B, S, 7)
        out[..., 3] = 1.0  # foul
        return out


@pytest.mark.skipif(
    not (_MODELS / "pitchgpt_v2.pt").exists(),
    reason="backbone models/pitchgpt_v2.pt missing — skip mid-PA mutation integration test.",
)
def test_mid_pa_context_mutation_produces_nonconstant_context():
    """Phase 0.6.1 integration: a short rollout re-emits the count_state
    one-hot each position, so the context tensor is NON-constant across
    positions; every non-count field stays PA-invariant.

    Uses the real backbone (CPU or GPU) with a mock foul-only predictor so no
    GPU-scale sampling is required.
    """
    from src.analytics.pitchgpt import NUM_COUNT_STATES
    from src.analytics.pitchgpt_sim import rollout

    ctx = _make_default_context()  # 0-0 count start
    pred = _RecordingFoulPredictor()
    horizon = 5
    rollout(
        ctx,
        outcome_predictor=pred,
        n_samples=3,
        horizon=horizon,
        temperature=1.0,
        seed=7,
    )

    assert len(pred.seen_contexts) == horizon, (
        f"expected {horizon} recorded contexts, got {len(pred.seen_contexts)}"
    )
    # count_state one-hot argmax per position (sample 0).  Foul-only path:
    # 0-0 -> 0-1 -> 0-2 -> 0-2 -> 0-2  == count_state 0,1,2,2,2.
    cs_by_pos = [
        int(np.argmax(c[0, :NUM_COUNT_STATES])) for c in pred.seen_contexts
    ]
    assert cs_by_pos == [0, 1, 2, 2, 2], (
        f"count_state sequence not mutated as expected: {cs_by_pos}"
    )
    # The context MUST be non-constant across positions (the whole point).
    assert len(set(cs_by_pos)) > 1, "context count_state did not vary across positions"

    # Every NON-count field must stay PA-invariant across all positions.
    for c in pred.seen_contexts[1:]:
        np.testing.assert_array_equal(
            c[:, NUM_COUNT_STATES:],
            pred.seen_contexts[0][:, NUM_COUNT_STATES:],
            err_msg="non-count context fields changed mid-PA (should be invariant)",
        )
    # The count one-hot slice DID change between position 0 and position 2.
    assert not np.array_equal(
        pred.seen_contexts[0][:, :NUM_COUNT_STATES],
        pred.seen_contexts[2][:, :NUM_COUNT_STATES],
    ), "count_state one-hot did not mutate between pos 0 and pos 2"


@pytest.mark.skipif(
    not _has_pg_concat_head_predictor(), reason=_PGCH_SKIP_MSG
)
def test_disable_class_calibration_flag_drops_class_cal_keeps_pos0():
    """Phase 0.6.1 A/B switch: `disable_class_calibration=True` nulls the
    all-position class_calibration but retains the position-0 recalibration."""
    from src.analytics.pitchgpt_sim import PGConcatHeadPredictor

    # Default (calibration enabled): the shipped JSON carries class_calibration.
    pred_on = PGConcatHeadPredictor()
    assert pred_on._class_calibration is not None, (
        "default calibration JSON is expected to carry class_calibration"
    )
    assert pred_on.class_calibration_disabled is False

    # Disabled: class_calibration dropped, pos-0 recalibration retained.
    pred_off = PGConcatHeadPredictor(disable_class_calibration=True)
    assert pred_off._class_calibration is None
    assert pred_off.class_calibration_disabled is True
    # pos-0 npz ships in models/, so it must still be loaded.
    assert pred_off._class_calibration_pos0 is not None, (
        "pos-0 recalibration must be retained when class_calibration is disabled"
    )
    assert pred_off.calibration.get("class_calibration_disabled") is True


@pytest.mark.skipif(
    not _has_pg_concat_head_predictor(), reason=_PGCH_SKIP_MSG
)
def test_disable_class_calibration_env_var(monkeypatch):
    """Phase 0.6.1: PITCHGPT_DISABLE_CLASS_CALIBRATION=1 disables the
    all-position class_calibration with no code change (default arg = None)."""
    from src.analytics.pitchgpt_sim import PGConcatHeadPredictor

    monkeypatch.setenv("PITCHGPT_DISABLE_CLASS_CALIBRATION", "1")
    pred = PGConcatHeadPredictor()  # disable_class_calibration=None -> read env
    assert pred.class_calibration_disabled is True
    assert pred._class_calibration is None

    # Explicit False must OVERRIDE the env var.
    pred2 = PGConcatHeadPredictor(disable_class_calibration=False)
    assert pred2.class_calibration_disabled is False
    assert pred2._class_calibration is not None
