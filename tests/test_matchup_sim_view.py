"""Unit tests for the A3 matchup-sim view + fixture.

Same constraints as test_pitch_call_grades_view.py:
- No pitchgpt_sim required (graceful-degrade).
- No streamlit context required at import time.
- Do NOT run pytest from this scaffolding session.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.dashboard.fixtures.matchup_sim_fixture import (
    SAMPLE_MATCHUPS,
    build_matchup_fixture,
    list_matchup_options,
    make_sampling_metadata,
    schema_preview,
)


# ---------------------------------------------------------------------------
# Fixture shape + content tests
# ---------------------------------------------------------------------------


def test_matchup_fixture_shape_and_dtype():
    """build_matchup_fixture must return a populated MatchupSimFixture."""
    fx = build_matchup_fixture(matchup_idx=0, seed=99)

    # n_simulated_pas matches metadata
    assert fx.n_simulated_pas == 10_000
    assert fx.sampling_metadata["n_samples"] == 10_000

    # woba_samples shape + dtype
    assert fx.woba_samples.shape == (10_000,)
    assert fx.woba_samples.dtype == np.float32

    # All wOBA values are non-negative and finite
    assert np.all(fx.woba_samples >= 0.0)
    assert np.isfinite(fx.woba_samples).all()

    # Percentile dict has the five canonical bands per A3 methodology
    for k in ("p05", "p25", "p50", "p75", "p95"):
        assert k in fx.percentiles
        assert isinstance(fx.percentiles[k], float)
    # Monotone (p05 <= p25 <= p50 <= p75 <= p95)
    pcts = [fx.percentiles[k] for k in ("p05", "p25", "p50", "p75", "p95")]
    assert pcts == sorted(pcts), f"percentiles not monotone: {pcts}"


def test_matchup_fixture_small_sample_flag_consistency():
    """small_sample_flag must equal (n_real_pas < 20) per Gate 3 rule."""
    for i, _ in enumerate(SAMPLE_MATCHUPS):
        fx = build_matchup_fixture(matchup_idx=i, seed=99)
        expected = fx.n_real_pas < 20
        assert fx.small_sample_flag == expected, (
            f"matchup {i}: small_sample_flag mismatch (n_real_pas={fx.n_real_pas})"
        )


def test_matchup_fixture_in_play_hit_share_in_unit_interval():
    """in_play_hit_share is a probability in [0, 1]."""
    for i, _ in enumerate(SAMPLE_MATCHUPS):
        fx = build_matchup_fixture(matchup_idx=i, seed=99)
        assert 0.0 <= fx.in_play_hit_share <= 1.0


# ---------------------------------------------------------------------------
# Sampling metadata tests
# ---------------------------------------------------------------------------


def test_matchup_sampling_metadata_required_keys():
    md = make_sampling_metadata(seed=99)
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
    assert md["outcome_predictor"] == "pg_concat_head"
    assert md["temperature"] == 1.0
    assert md["backbone_version"] == "v2"


# ---------------------------------------------------------------------------
# list_matchup_options
# ---------------------------------------------------------------------------


def test_list_matchup_options_returns_correct_count():
    opts = list_matchup_options()
    assert len(opts) == len(SAMPLE_MATCHUPS)
    for label, idx in opts:
        assert isinstance(label, str) and "vs" in label
        assert 0 <= idx < len(SAMPLE_MATCHUPS)


# ---------------------------------------------------------------------------
# View module imports cleanly
# ---------------------------------------------------------------------------


def test_view_imports_cleanly_without_pitchgpt_sim():
    """The A3 view must import even when pitchgpt_sim is unavailable."""
    import sys
    import importlib.abc
    import importlib.machinery

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

    class _BlockSim(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == "src.analytics.pitchgpt_sim":
                raise ImportError("pitchgpt_sim blocked for test")
            return None

    blocker = _BlockSim()
    sys.meta_path.insert(0, blocker)
    try:
        sys.modules.pop("src.dashboard.views.matchup_sim", None)
        from src.dashboard.views import matchup_sim  # noqa: F401

        assert hasattr(matchup_sim, "render")
        ok = matchup_sim._try_import_pitchgpt_sim()
        assert ok is False
        assert matchup_sim._SIM_IMPORT_ERROR is not None
    finally:
        sys.meta_path.remove(blocker)
        # Restore the parent-package attribute so subsequent tests in the
        # same session that rely on `src.analytics.pitchgpt_sim` aren't
        # broken by our cleanup.
        if _had_attr and _parent is not None:
            setattr(_parent, "pitchgpt_sim", _saved_attr)


# ---------------------------------------------------------------------------
# Schema preview
# ---------------------------------------------------------------------------


def test_schema_preview_returns_known_keys():
    sp = schema_preview()
    assert "MatchupSim.woba_samples" in sp
    assert "MatchupSim.percentiles" in sp
    assert "MatchupSim.small_sample_flag" in sp
    for k, v in sp.items():
        assert isinstance(v, str) and v.strip()


# ---------------------------------------------------------------------------
# Sample matchups self-consistency
# ---------------------------------------------------------------------------


def test_sample_matchups_have_required_keys():
    """Each entry in SAMPLE_MATCHUPS must declare the canonical keys."""
    required = {
        "batter_name",
        "batter_id",
        "pitcher_name",
        "pitcher_id",
        "n_real_pas",
        "small_sample_flag",
        "mean_woba",
        "std_woba",
    }
    for m in SAMPLE_MATCHUPS:
        missing = required - set(m.keys())
        assert not missing, f"matchup missing keys: {missing}"
        assert m["n_real_pas"] >= 0
        assert m["mean_woba"] >= 0.0
        assert m["std_woba"] >= 0.0


def test_matchup_fixture_reproducible_under_same_seed():
    """Re-building with the same seed must yield identical wOBA arrays."""
    fx_a = build_matchup_fixture(matchup_idx=0, seed=99)
    fx_b = build_matchup_fixture(matchup_idx=0, seed=99)
    np.testing.assert_array_equal(fx_a.woba_samples, fx_b.woba_samples)
    assert fx_a.percentiles == fx_b.percentiles


# ---------------------------------------------------------------------------
# K5 consequence: no simulated wOBA level may reach the page (2026-08-10)
# ---------------------------------------------------------------------------


def test_simulated_median_is_zero_so_median_centring_is_a_no_op():
    """Pins the fact that motivated withholding rather than re-centring.

    PA-terminal wOBA is zero for every PA ending in an out, so the simulated
    median is exactly zero for every pair. Any "delta vs own median" display
    would therefore render the absolute levels unchanged under a delta label.
    If this ever stops holding, the module docstring's rejected-alternative
    rationale needs revisiting -- it does not license re-centring.
    """
    for i, _ in enumerate(SAMPLE_MATCHUPS):
        fx = build_matchup_fixture(matchup_idx=i, seed=99)
        assert fx.percentiles["p50"] == 0.0, (
            f"matchup {i}: median {fx.percentiles['p50']} is no longer zero"
        )


def test_cohort_rank_products_publish_integer_ordinals_only():
    """The A3 payload may carry ordinals and labels -- never a wOBA quantity.

    This is the structural guarantee behind the "absolute levels are
    withheld" copy: if a float ever re-enters the payload, it is a wOBA-scale
    number and this fails.
    """
    from src.dashboard.views import matchup_sim

    n = len(SAMPLE_MATCHUPS)
    seen_ranks = []
    for i in range(n):
        products = matchup_sim.cohort_rank_products(i)
        assert set(products) == {"rank", "n_pairs", "ordering"}
        assert type(products["rank"]) is int
        assert type(products["n_pairs"]) is int
        assert products["n_pairs"] == n
        assert 1 <= products["rank"] <= n
        seen_ranks.append(products["rank"])

        assert len(products["ordering"]) == n
        assert sum(row["selected"] for row in products["ordering"]) == 1
        assert [row["rank"] for row in products["ordering"]] == list(range(1, n + 1))
        for row in products["ordering"]:
            assert set(row) == {"rank", "matchup", "selected"}
            assert type(row["rank"]) is int
            assert type(row["selected"]) is bool
            assert isinstance(row["matchup"], str) and row["matchup"].strip()
        # The selected row is the pair's own rank.
        selected = [row for row in products["ordering"] if row["selected"]][0]
        assert selected["rank"] == products["rank"]

    # Ranks are a permutation: the ordering is total, not degenerate.
    assert sorted(seen_ranks) == list(range(1, n + 1))


def test_render_path_never_touches_woba_samples_or_percentiles():
    """Source guard: the rendered products may not be built from wOBA values.

    ``_cohort_order`` is the single sanctioned reader of ``woba_samples`` (it
    reduces them to an ordering and discards the levels). Neither the public
    product builder nor the renderer may read a wOBA level.
    """
    import inspect

    from src.dashboard.views import matchup_sim

    for fn in (matchup_sim.cohort_rank_products, matchup_sim._render_ordinal_products):
        src = inspect.getsource(fn)
        assert "percentiles" not in src, f"{fn.__name__} reads percentile bands"
        assert "woba_samples" not in src, f"{fn.__name__} reads wOBA samples"


def test_kill_quantities_render_as_text_not_a_dict_repr():
    """The kill claim's value is a mapping; banners must format its sub-keys."""
    from src.claims import get_claim
    from src.dashboard.views import matchup_sim

    kill = get_claim("pitchgpt_phase062_kill").value
    for text in (matchup_sim._KILL_QUANTITIES, matchup_sim._PA_ABSOLUTE_DROPPED_BANNER):
        assert "{'" not in text, "raw Python dict repr leaked into the banner"
        for key in kill:
            assert key not in text, f"dict key {key!r} leaked into user-facing copy"
    for key in ("iteration_1_max_abs_delta_pp", "iteration_2_max_abs_delta_pp"):
        assert f"{kill[key]:.3f}pp" in matchup_sim._KILL_QUANTITIES
    assert kill["verdict"] in matchup_sim._KILL_QUANTITIES


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
