"""Unit tests for the A1 pitch-call grades view + fixture.

These tests do NOT require ``pitchgpt_sim`` to exist. They cover:
    1. Fixture shape + dtype contracts (mirrors SIM_ENGINE_API §3.3).
    2. Fixture pad-NaN convention past PA termination.
    3. Sampling-metadata required keys per SIM_ENGINE_API §3.4.
    4. Leaderboard table builder integrity.
    5. Grade-letter mapping contract (high percentile -> A; low -> F).
    6. View module imports cleanly (no streamlit context required).

Do NOT run ``pytest`` from this scaffolding session; the parallel
Phase 0.5 session may be writing tests too. The user / a future
validation-agent run is the canonical pytest invoker.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.dashboard.fixtures.pitch_call_grades_fixture import (
    OUTCOME_LABELS,
    OUTCOME_TO_WOBA_DEMO,
    ROLLOUT_PAD_OUTCOME,
    ROLLOUT_PAD_PITCH,
    SAMPLE_PAS,
    _percentile_to_grade,
    build_leaderboard_table,
    build_rollout_fixture,
    make_sampling_metadata,
    schema_preview,
)


# ---------------------------------------------------------------------------
# Fixture shape + dtype tests (mirrors SIM_ENGINE_API §3.3)
# ---------------------------------------------------------------------------


def test_rollout_fixture_shape_and_dtype():
    """Fixture must match the SIM_ENGINE_API §3.3 shape contract."""
    fx = build_rollout_fixture(seed=42)

    n_samples = 100
    horizon = 6

    assert fx.pitch_tokens.shape == (n_samples, horizon)
    assert fx.pitch_tokens.dtype == np.int64

    # pitch_probs is None when return_probs=False (per §3.3 contract)
    assert fx.pitch_probs is None

    assert fx.outcomes.shape == (n_samples, horizon)
    assert fx.outcomes.dtype == np.int64

    assert fx.outcome_probs.shape == (n_samples, horizon, 7)
    assert fx.outcome_probs.dtype == np.float32

    assert fx.pa_terminated.shape == (n_samples, horizon)
    assert fx.pa_terminated.dtype == bool

    assert fx.pa_outcome.shape == (n_samples,)
    assert fx.pa_outcome.dtype == np.int64

    assert fx.final_count.shape == (n_samples, 2)
    assert fx.final_count.dtype == np.int64

    assert isinstance(fx.sampling_metadata, dict)


def test_rollout_fixture_pad_nan_convention():
    """Past-termination outcome_probs MUST be NaN, not zero (SIM_ENGINE_API §3.3).

    Silent zeros would propagate to KL/Wasserstein aggregates. The fixture
    MUST honor this exactly so the view paths NaN-mask correctly.
    """
    fx = build_rollout_fixture(seed=42)

    found_truncation_pair = False
    for i in range(fx.outcomes.shape[0]):
        if not fx.pa_terminated[i].any():
            # Truncated sample (no termination); pa_outcome must be PAD per §3.3
            assert fx.pa_outcome[i] == ROLLOUT_PAD_OUTCOME, (
                f"sample {i}: untruncated mark expected PAD outcome"
            )
            continue
        # Sample terminated -> find term position
        term_pos = int(np.argmax(fx.pa_terminated[i]))
        # Past-termination outcome_probs must be NaN
        if term_pos + 1 < fx.outcome_probs.shape[1]:
            past = fx.outcome_probs[i, term_pos + 1 :]
            assert np.isnan(past).all(), (
                f"sample {i}: outcome_probs past pos {term_pos} must be NaN"
            )
            found_truncation_pair = True
        # Past-termination pitch_tokens must be the PAD sentinel
        if term_pos + 1 < fx.pitch_tokens.shape[1]:
            assert (fx.pitch_tokens[i, term_pos + 1 :] == ROLLOUT_PAD_PITCH).all()
        # Past-termination outcomes must be PAD outcome (7)
        if term_pos + 1 < fx.outcomes.shape[1]:
            assert (fx.outcomes[i, term_pos + 1 :] == ROLLOUT_PAD_OUTCOME).all()

    assert found_truncation_pair, (
        "fixture should contain at least one terminated-then-padded sample "
        "to exercise the NaN-mask convention"
    )


def test_rollout_fixture_outcome_indices_in_range():
    """Outcome indices live in [0, 6] (real outcomes) or 7 (PAD)."""
    fx = build_rollout_fixture(seed=42)
    valid_with_pad = set(range(8))   # 0..6 + 7 PAD
    actual_set = set(np.unique(fx.outcomes).tolist())
    assert actual_set.issubset(valid_with_pad), (
        f"outcomes contains illegal values: {actual_set - valid_with_pad}"
    )
    assert len(OUTCOME_LABELS) == 7
    assert ROLLOUT_PAD_OUTCOME == 7


# ---------------------------------------------------------------------------
# Sampling metadata schema (SIM_ENGINE_API §3.4)
# ---------------------------------------------------------------------------


def test_sampling_metadata_required_keys():
    """SIM_ENGINE_API §3.4 lists the required keys; fixture must carry all."""
    md = make_sampling_metadata(seed=42)
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
    missing = required - set(md.keys())
    assert not missing, f"missing required metadata keys: {missing}"

    # Plan B winner is locked at pg_concat_head (SIM_ENGINE_API §4.4)
    assert md["outcome_predictor"] == "pg_concat_head"
    # Phase 0.5 fixture defaults to T=1.0 (calibrated regime)
    assert md["temperature"] == 1.0
    # Backbone v2 is locked per SIM_ENGINE_API §2 Locked context
    assert md["backbone_version"] == "v2"


def test_sampling_metadata_calibration_invalid_reasons_consistency():
    """If calibration_valid is True, calibration_invalid_reasons must be empty."""
    md = make_sampling_metadata(seed=42)
    if md["calibration_valid"]:
        assert md["calibration_invalid_reasons"] == [], (
            "SIM_ENGINE_API §6: calibration_invalid_reasons must be empty when "
            "calibration_valid is True"
        )


# ---------------------------------------------------------------------------
# Leaderboard table builder
# ---------------------------------------------------------------------------


def test_leaderboard_table_nonempty_and_typed():
    rows = build_leaderboard_table(seed=42)
    assert len(rows) > 0
    required_cols = {
        "pitcher_id",
        "pitcher_name",
        "week",
        "n_pitches",
        "mean_grade_percentile",
        "grade_letter",
        "woba_delta_per_pitch_x1000",
    }
    for row in rows:
        missing = required_cols - set(row.keys())
        assert not missing, f"leaderboard row missing keys: {missing}"
        # Sanity: percentile in [0, 1], n_pitches positive
        assert 0.0 <= row["mean_grade_percentile"] <= 1.0
        assert row["n_pitches"] > 0
        assert row["grade_letter"] in {"A", "B", "C", "D", "F"}


# ---------------------------------------------------------------------------
# Grade-letter mapping
# ---------------------------------------------------------------------------


def test_percentile_to_grade_monotone():
    """Higher percentile -> better grade. SIM_ENGINE_API + EXECUTION_PLAN A1
    methodology: high percentile = pitcher gave up LESS than expected."""
    grades_in_order = [_percentile_to_grade(p) for p in (0.05, 0.40, 0.60, 0.80, 0.95)]
    # F < D < C < B < A
    rank_map = {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4}
    ranks = [rank_map[g] for g in grades_in_order]
    assert ranks == sorted(ranks), (
        f"grade-letter mapping must be monotone in percentile; got {grades_in_order}"
    )


def test_percentile_to_grade_boundaries():
    """Spot-check boundary cases."""
    assert _percentile_to_grade(0.85) == "A"
    assert _percentile_to_grade(0.84999) == "B"
    assert _percentile_to_grade(0.30) == "D"
    assert _percentile_to_grade(0.29999) == "F"


# ---------------------------------------------------------------------------
# View module import test
# ---------------------------------------------------------------------------


def test_view_imports_cleanly_without_pitchgpt_sim(monkeypatch):
    """The A1 view must import even when ``pitchgpt_sim`` is unavailable.

    This is the load-bearing graceful-degrade: the view is imported by
    the dashboard router; failing here would crash the whole dashboard.
    """
    # Force the import attempt to fail by temporarily blocking pitchgpt_sim
    import sys

    # Remove any previously-cached pitchgpt_sim
    sys.modules.pop("src.analytics.pitchgpt_sim", None)
    # MetaPathFinder only fires on a fresh find_spec call; if a prior test
    # populated the parent-package attribute, `from src.analytics import
    # pitchgpt_sim` short-circuits via attribute access and the blocker
    # never sees it. Strip the attribute so the blocker actually fires.
    _parent = sys.modules.get("src.analytics")
    _had_attr = _parent is not None and hasattr(_parent, "pitchgpt_sim")
    _saved_attr = getattr(_parent, "pitchgpt_sim", None) if _had_attr else None
    if _had_attr:
        delattr(_parent, "pitchgpt_sim")

    # Insert a finder that raises ImportError for pitchgpt_sim only
    import importlib.abc
    import importlib.machinery

    class _BlockSim(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "src.analytics.pitchgpt_sim":
                raise ImportError("pitchgpt_sim blocked for test")
            return None

    blocker = _BlockSim()
    sys.meta_path.insert(0, blocker)

    try:
        # Reload the view module fresh
        sys.modules.pop("src.dashboard.views.pitch_call_grades", None)
        from src.dashboard.views import pitch_call_grades  # noqa: F401

        # The probe import is wrapped in try/except, so import should succeed
        # even with pitchgpt_sim blocked.
        assert hasattr(pitch_call_grades, "render")

        # The probe records the import error
        ok = pitch_call_grades._try_import_pitchgpt_sim()
        assert ok is False
        assert pitch_call_grades._SIM_IMPORT_ERROR is not None
    finally:
        sys.meta_path.remove(blocker)
        # Restore the parent-package attribute so subsequent tests in the
        # same session that rely on `src.analytics.pitchgpt_sim` aren't
        # broken by our cleanup.
        if _had_attr and _parent is not None:
            setattr(_parent, "pitchgpt_sim", _saved_attr)


# ---------------------------------------------------------------------------
# Schema preview convenience
# ---------------------------------------------------------------------------


def test_schema_preview_returns_known_keys():
    sp = schema_preview()
    assert "RolloutResult.pitch_tokens" in sp
    assert "RolloutResult.sampling_metadata" in sp
    # Each value is a non-empty string
    for k, v in sp.items():
        assert isinstance(v, str) and v.strip()


# ---------------------------------------------------------------------------
# SAMPLE_PAS sanity
# ---------------------------------------------------------------------------


def test_sample_pas_have_valid_outcome_indices():
    """Each sample PA's real_outcome_idx must be in [0, 6]."""
    for pa in SAMPLE_PAS:
        assert 0 <= pa["real_outcome_idx"] <= 6, (
            f"PA {pa['pitcher_name']} vs {pa['batter_name']} has illegal "
            f"outcome idx {pa['real_outcome_idx']}"
        )
        assert pa["grade_letter"] in {"A", "B", "C", "D", "F", "A-", "B-", "C-", "D-"}
        # Demo percentile must be in [0, 1]
        assert 0.0 <= pa["demo_percentile"] <= 1.0


def test_outcome_to_woba_demo_keys():
    """OUTCOME_TO_WOBA_DEMO must have keys for all 7 outcome classes."""
    assert set(OUTCOME_TO_WOBA_DEMO.keys()) == set(range(7))
    # All values are non-negative floats
    for k, v in OUTCOME_TO_WOBA_DEMO.items():
        assert isinstance(v, float) and v >= 0.0


# ---------------------------------------------------------------------------
# Dependency-light: don't require pytest plugins beyond defaults
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
