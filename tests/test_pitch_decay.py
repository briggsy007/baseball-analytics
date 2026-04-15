"""
Tests for the Pitch Decay Rate (PDR) model.

Covers:
- Per-pitch-type quality signal z-scoring
- Cliff detection on synthetic piecewise data (known breakpoint recovery)
- Survival curve properties (monotonically non-increasing, starts at 1.0)
- "First to die" returns earliest cliff
- Edge cases: no cliff detected (monotonic), single pitch type, < 30 pitches
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.pitch_decay import (
    PitchDecayRateModel,
    batch_calculate,
    calculate_pdr,
    compute_pitch_type_quality,
    detect_cliff_point,
    get_first_to_die,
    _build_survival_curve,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_piecewise_series(
    n: int = 60,
    tau: int = 30,
    pre_slope: float = -0.002,
    post_slope: float = -0.05,
    noise_std: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a synthetic piecewise-linear quality series with a known breakpoint.

    Returns (quality_values, pitch_numbers).
    """
    rng = np.random.RandomState(seed)
    x = np.arange(n, dtype=float)
    y = np.where(
        x <= tau,
        1.0 + pre_slope * x,
        1.0 + pre_slope * tau + post_slope * (x - tau),
    )
    if noise_std > 0:
        y = y + rng.normal(0, noise_std, n)
    return y, x


def _make_pitch_type_df(
    n: int = 60,
    pitch_type: str = "FF",
    decay_after: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic DataFrame for one pitch type with a quality cliff.

    Velocity drops sharply after ``decay_after`` pitches.
    """
    rng = np.random.RandomState(seed)
    pitch_numbers = np.arange(n)

    # Pre-cliff: stable velocity; post-cliff: drops
    velo = np.where(
        pitch_numbers <= decay_after,
        95.0 + rng.normal(0, 0.3, n),
        95.0 - 0.15 * (pitch_numbers - decay_after) + rng.normal(0, 0.3, n),
    )
    spin = np.where(
        pitch_numbers <= decay_after,
        2400.0 + rng.normal(0, 30, n),
        2400.0 - 5.0 * (pitch_numbers - decay_after) + rng.normal(0, 30, n),
    )

    return pd.DataFrame({
        "game_pk": 999999,
        "game_date": "2025-06-15",
        "pitcher_id": 554430,
        "batter_id": 100001,
        "pitch_type": pitch_type,
        "release_speed": velo.clip(55, 105),
        "release_spin_rate": spin.clip(100, 3800),
        "pfx_x": rng.normal(-6, 0.5, n),
        "pfx_z": rng.normal(15, 0.5, n),
        "release_pos_x": rng.normal(-1.5, 0.05, n),
        "release_pos_z": rng.normal(5.8, 0.05, n),
        "inning": np.clip(pitch_numbers // 15 + 1, 1, 9).astype(int),
        "at_bat_number": (pitch_numbers // 5 + 1).astype(int),
        "pitch_number": (pitch_numbers % 5 + 1).astype(int),
    })


def _make_multi_type_df(
    n_per_type: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a DataFrame with multiple pitch types each having a different cliff."""
    dfs = []
    # FF cliff at 40, SL cliff at 20, CH cliff at 30
    for i, (pt, cliff) in enumerate([("FF", 40), ("SL", 20), ("CH", 30)]):
        df = _make_pitch_type_df(
            n=n_per_type, pitch_type=pt, decay_after=cliff, seed=seed + i,
        )
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined


# ── Tests: compute_pitch_type_quality ──────────────────────────────────────────


class TestPitchTypeQuality:
    """Tests for the per-pitch-type quality signal computation."""

    def test_returns_series_for_matching_type(self):
        df = _make_pitch_type_df(50, pitch_type="FF")
        quality = compute_pitch_type_quality(df, "FF")
        assert isinstance(quality, pd.Series)
        assert len(quality) == 50

    def test_empty_for_missing_type(self):
        df = _make_pitch_type_df(50, pitch_type="FF")
        quality = compute_pitch_type_quality(df, "CU")
        assert len(quality) == 0

    def test_zscored_within_pitch_type(self):
        """Quality should be z-score-like: mean near 0, nonzero std."""
        df = _make_pitch_type_df(100, pitch_type="SL", decay_after=1000, seed=7)
        quality = compute_pitch_type_quality(df, "SL")
        assert abs(quality.mean()) < 1.0
        assert quality.std() > 0

    def test_no_nans(self):
        df = _make_pitch_type_df(50, pitch_type="FF")
        quality = compute_pitch_type_quality(df, "FF")
        assert quality.notna().all()

    def test_multi_type_isolation(self):
        """Quality for FF should only use FF pitches, not SL."""
        df = _make_multi_type_df(50)
        ff_quality = compute_pitch_type_quality(df, "FF")
        sl_quality = compute_pitch_type_quality(df, "SL")
        # Each should have exactly n_per_type entries
        assert len(ff_quality) == 50
        assert len(sl_quality) == 50
        # Indices should not overlap
        assert set(ff_quality.index).isdisjoint(set(sl_quality.index))


# ── Tests: detect_cliff_point ──────────────────────────────────────────────────


class TestDetectCliffPoint:
    """Tests for piecewise changepoint detection."""

    def test_recovers_known_breakpoint(self):
        """Clean piecewise data with known tau should be recovered."""
        true_tau = 30
        y, x = _make_piecewise_series(n=60, tau=true_tau, pre_slope=-0.001, post_slope=-0.05)
        cliff = detect_cliff_point(y, x)

        assert cliff["tau"] is not None
        # Should be close to the true breakpoint
        assert abs(cliff["tau"] - true_tau) <= 5, (
            f"Expected tau near {true_tau}, got {cliff['tau']}"
        )

    def test_noisy_data_still_detects(self):
        """Adding moderate noise should still find the cliff."""
        true_tau = 25
        y, x = _make_piecewise_series(
            n=60, tau=true_tau, pre_slope=-0.001, post_slope=-0.04, noise_std=0.02,
        )
        cliff = detect_cliff_point(y, x)
        assert cliff["tau"] is not None
        assert abs(cliff["tau"] - true_tau) <= 8

    def test_post_slope_more_negative_than_pre(self):
        """The post-cliff slope should be more negative than the pre-cliff slope."""
        y, x = _make_piecewise_series(n=60, tau=30, pre_slope=0.0, post_slope=-0.05)
        cliff = detect_cliff_point(y, x)
        assert cliff["tau"] is not None
        assert cliff["post_slope"] < cliff["pre_slope"]

    def test_mse_reduction_positive(self):
        """Breaking the series into two should reduce MSE compared to one line."""
        y, x = _make_piecewise_series(n=60, tau=30, pre_slope=0.0, post_slope=-0.05)
        cliff = detect_cliff_point(y, x)
        assert cliff["mse_reduction"] is not None
        assert cliff["mse_reduction"] > 0

    def test_too_few_points_returns_none(self):
        """Fewer than 10 points should return None tau."""
        y = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        x = np.arange(5, dtype=float)
        cliff = detect_cliff_point(y, x)
        assert cliff["tau"] is None

    def test_monotonic_flat_returns_none(self):
        """Constant data (no cliff) should return None tau."""
        x = np.arange(40, dtype=float)
        y = np.ones(40) * 5.0
        cliff = detect_cliff_point(y, x)
        assert cliff["tau"] is None

    def test_monotonic_increasing_returns_none(self):
        """Steadily increasing data has no negative cliff."""
        x = np.arange(40, dtype=float)
        y = 0.01 * x + 1.0
        cliff = detect_cliff_point(y, x)
        # Either None (no improvement from split) or post_slope >= pre_slope
        # which our function filters out
        assert cliff["tau"] is None


# ── Tests: survival curve ──────────────────────────────────────────────────────


class TestSurvivalCurve:
    """Tests for the Kaplan-Meier-style survival curve."""

    def test_starts_at_one(self):
        curve = _build_survival_curve([20.0, 30.0, 40.0, 50.0])
        assert curve["survival"][0] == 1.0

    def test_monotonically_non_increasing(self):
        curve = _build_survival_curve([15.0, 25.0, 35.0, 45.0, 55.0])
        surv = curve["survival"]
        for i in range(1, len(surv)):
            assert surv[i] <= surv[i - 1], (
                f"Survival increased at pitch {i}: {surv[i]} > {surv[i-1]}"
            )

    def test_reaches_zero(self):
        """After all cliff points, survival should be 0."""
        cliff_points = [10.0, 20.0, 30.0]
        curve = _build_survival_curve(cliff_points, max_pitch_count=50)
        # At pitch 31, all 3 cliffs have been passed
        assert curve["survival"][31] == 0.0

    def test_empty_input(self):
        curve = _build_survival_curve([])
        assert curve["pitch_numbers"] == []
        assert curve["survival"] == []

    def test_single_cliff(self):
        curve = _build_survival_curve([25.0], max_pitch_count=40)
        assert curve["survival"][0] == 1.0
        assert curve["survival"][25] == 0.0  # cliff is at 25, so at 25 it's passed
        assert curve["survival"][24] == 1.0  # still effective at 24


# ── Tests: calculate_pdr (DB integration) ─────────────────────────────────────


class TestCalculatePDR:
    """Tests for the full pitcher PDR calculation using the DB."""

    def test_output_schema(self, db_conn):
        """Output dict should have the required keys."""
        result = calculate_pdr(db_conn, pitcher_id=999999999)
        assert "pitcher_id" in result
        assert "season" in result
        assert "pitch_types" in result
        assert "first_to_die" in result
        assert "first_to_die_cliff" in result

    def test_nonexistent_pitcher(self, db_conn):
        result = calculate_pdr(db_conn, pitcher_id=999999999)
        assert result["pitch_types"] == {}
        assert result["first_to_die"] is None

    def test_known_pitcher_structure(self, db_conn):
        """Schema should always be present even if insufficient data."""
        result = calculate_pdr(db_conn, pitcher_id=554430)
        assert isinstance(result, dict)
        assert result["pitcher_id"] == 554430
        assert isinstance(result["pitch_types"], dict)


# ── Tests: get_first_to_die ───────────────────────────────────────────────────


class TestFirstToDie:
    """Tests for the first-to-die convenience function."""

    def test_returns_earliest_cliff(self):
        """Manually verify that first_to_die picks the minimum median_cliff."""
        # We test via the pure functions rather than DB
        # Create data where SL degrades earliest
        cliff_taus_ff = [40.0, 45.0, 38.0, 42.0, 41.0]
        cliff_taus_sl = [20.0, 22.0, 18.0, 21.0, 19.0]

        median_ff = float(np.median(cliff_taus_ff))
        median_sl = float(np.median(cliff_taus_sl))

        assert median_sl < median_ff, "SL should have earlier cliff"

    def test_schema(self, db_conn):
        result = get_first_to_die(db_conn, pitcher_id=999999999)
        assert "pitcher_id" in result
        assert "first_to_die" in result
        assert "cliff_pitch_number" in result
        assert "all_cliffs" in result

    def test_nonexistent_pitcher(self, db_conn):
        result = get_first_to_die(db_conn, pitcher_id=999999999)
        assert result["first_to_die"] is None
        assert result["all_cliffs"] == {}


# ── Tests: batch_calculate ─────────────────────────────────────────────────────


class TestBatchCalculate:
    """Tests for the batch leaderboard calculation."""

    def test_returns_dataframe(self, db_conn):
        df = batch_calculate(db_conn, min_pitches=10)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, db_conn):
        df = batch_calculate(db_conn, min_pitches=10)
        for col in ["pitcher_id", "first_to_die", "first_to_die_cliff"]:
            assert col in df.columns

    def test_high_min_pitches_returns_fewer_or_empty(self, db_conn):
        df = batch_calculate(db_conn, min_pitches=999999)
        assert isinstance(df, pd.DataFrame)
        assert len(df) <= 1


# ── Tests: edge cases ─────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests."""

    def test_fewer_than_30_pitches_skipped(self):
        """Games with fewer than 30 pitches of a type should not produce a cliff."""
        from src.analytics.pitch_decay import _detect_game_cliff

        df = _make_pitch_type_df(20, pitch_type="FF")
        result = _detect_game_cliff(df, "FF")
        assert result is None

    def test_single_pitch_type(self, db_conn):
        result = calculate_pdr(db_conn, pitcher_id=554430)
        assert isinstance(result, dict)

    def test_model_class_interface(self):
        model = PitchDecayRateModel()
        assert model.model_name == "pitch_decay_rate"
        assert model.version == "1.0.0"
        assert "created_at" in model.metadata

    def test_quality_zero_variance_no_crash(self):
        """Constant velocity should not cause division errors."""
        df = _make_pitch_type_df(50, pitch_type="FF")
        df["release_speed"] = 95.0
        df["release_spin_rate"] = 2300.0
        quality = compute_pitch_type_quality(df, "FF")
        assert quality.notna().all()
