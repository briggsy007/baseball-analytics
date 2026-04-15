"""
Tests for src.analytics.features — shared feature engineering functions.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.features import (
    classify_pitch_outcome,
    compute_run_value,
    compute_z_scores,
    encode_count_state,
    exponential_decay,
    pitch_quality_vector,
    release_point_distance,
    rolling_mean,
    shannon_entropy,
    tunnel_distance,
)


# ── compute_z_scores ─────────────────────────────────────────────────────


class TestComputeZScores:
    """Tests for z-score normalisation."""

    def test_basic_z_scores(self):
        df = pd.DataFrame({"velo": [90.0, 93.0, 96.0, 99.0]})
        result = compute_z_scores(df, ["velo"])
        # Mean = 94.5, values should be symmetric around 0
        assert "velo_z" in result.columns
        assert abs(result["velo_z"].mean()) < 1e-10

    def test_grouped_z_scores(self):
        df = pd.DataFrame({
            "pitch_type": ["FF", "FF", "SL", "SL"],
            "velo": [94.0, 96.0, 84.0, 86.0],
        })
        result = compute_z_scores(df, ["velo"], group_by="pitch_type")
        # Within each group the z-scores should be symmetric
        ff_z = result.loc[result["pitch_type"] == "FF", "velo_z"]
        assert abs(ff_z.mean()) < 1e-10

    def test_zero_variance_produces_zero(self):
        """When all values are identical, z-scores should be 0."""
        df = pd.DataFrame({"velo": [93.0, 93.0, 93.0]})
        result = compute_z_scores(df, ["velo"])
        assert (result["velo_z"] == 0.0).all()

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"velo": [90.0, 95.0]})
        result = compute_z_scores(df, ["nonexistent"])
        assert "nonexistent_z" not in result.columns

    def test_single_value(self):
        """A single-row DataFrame should produce z=0 (std is NaN)."""
        df = pd.DataFrame({"velo": [95.0]})
        result = compute_z_scores(df, ["velo"])
        assert result["velo_z"].iloc[0] == 0.0

    def test_preserves_original_columns(self):
        df = pd.DataFrame({"velo": [90.0, 95.0], "spin": [2200, 2400]})
        result = compute_z_scores(df, ["velo"])
        assert list(result.columns) == ["velo", "spin", "velo_z"]
        assert (result["velo"] == df["velo"]).all()


# ── rolling_mean ──────────────────────────────────────────────────────────


class TestRollingMean:
    """Tests for rolling average."""

    def test_basic_rolling(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = rolling_mean(s, window=3)
        # Last value: mean(3,4,5) = 4.0
        assert result.iloc[-1] == pytest.approx(4.0)

    def test_min_periods_default(self):
        """Default min_periods = max(1, window//3), so first values are not NaN."""
        s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        result = rolling_mean(s, window=3)
        # min_periods = max(1, 1) = 1, so first value should be 10.0
        assert not pd.isna(result.iloc[0])

    def test_custom_min_periods(self):
        s = pd.Series([1.0, 2.0, 3.0])
        result = rolling_mean(s, window=5, min_periods=1)
        assert not pd.isna(result.iloc[0])

    def test_all_nan_input(self):
        s = pd.Series([np.nan, np.nan, np.nan])
        result = rolling_mean(s, window=2)
        assert result.isna().all()


# ── exponential_decay ─────────────────────────────────────────────────────


class TestExponentialDecay:
    """Tests for EWMA."""

    def test_basic_ewma(self):
        values = pd.Series([1.0, 1.0, 1.0, 1.0])
        result = exponential_decay(values, half_life=2.0)
        # Constant series => EWMA = constant
        assert all(abs(v - 1.0) < 1e-6 for v in result)

    def test_recent_values_weighted_more(self):
        """A jump at the end should pull the EWMA up."""
        values = pd.Series([1.0, 1.0, 1.0, 1.0, 10.0])
        result = exponential_decay(values, half_life=2.0)
        # The last EWMA value should be pulled towards 10
        assert result.iloc[-1] > result.iloc[-2]

    def test_numpy_array_input(self):
        arr = np.array([5.0, 5.0, 5.0])
        result = exponential_decay(arr, half_life=1.0)
        assert len(result) == 3

    def test_single_value(self):
        result = exponential_decay(pd.Series([42.0]), half_life=3.0)
        assert result.iloc[0] == pytest.approx(42.0)


# ── shannon_entropy ───────────────────────────────────────────────────────


class TestShannonEntropy:
    """Tests for entropy computation."""

    def test_uniform_distribution(self):
        """Entropy of uniform dist over n outcomes = ln(n)."""
        n = 4
        probs = [1.0 / n] * n
        result = shannon_entropy(probs)
        expected = math.log(n)
        assert result == pytest.approx(expected, abs=1e-8)

    def test_certain_outcome(self):
        """Entropy of a certain event is 0."""
        result = shannon_entropy([1.0])
        assert result == pytest.approx(0.0)

    def test_binary_fair_coin(self):
        result = shannon_entropy([0.5, 0.5])
        assert result == pytest.approx(math.log(2), abs=1e-8)

    def test_negative_probability_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            shannon_entropy([-0.5, 1.5])

    def test_probabilities_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum to"):
            shannon_entropy([0.3, 0.3])

    def test_with_zeros(self):
        """Zeros in the distribution are handled safely."""
        result = shannon_entropy([0.0, 0.0, 1.0])
        assert result == pytest.approx(0.0)


# ── pitch_quality_vector ─────────────────────────────────────────────────


class TestPitchQualityVector:
    """Tests for composite pitch quality signal."""

    def test_league_average_pitch(self):
        """A league-average pitch should score around 50."""
        row = pd.Series({
            "release_speed": 93.0,
            "release_spin_rate": 2300.0,
            "pfx_x": -6.0,
            "pfx_z": 6.0,
        })
        result = pitch_quality_vector(row)
        assert 40 <= result["composite"] <= 60

    def test_elite_fastball(self):
        """A 100 mph / 2800 rpm fastball should score well above 50."""
        row = pd.Series({
            "release_speed": 100.0,
            "release_spin_rate": 2800.0,
            "pfx_x": -8.0,
            "pfx_z": 16.0,
        })
        result = pitch_quality_vector(row)
        assert result["velocity_score"] > 70
        assert result["composite"] > 60

    def test_missing_values_get_50(self):
        """Missing attributes default to score 50."""
        row = pd.Series({})
        result = pitch_quality_vector(row)
        assert result["velocity_score"] == 50.0
        assert result["spin_score"] == 50.0
        assert result["movement_score"] == 50.0

    def test_output_keys(self):
        row = pd.Series({
            "release_speed": 95.0,
            "release_spin_rate": 2400.0,
            "pfx_x": -5.0,
            "pfx_z": 12.0,
        })
        result = pitch_quality_vector(row)
        assert set(result.keys()) == {
            "velocity_score", "spin_score", "movement_score", "composite"
        }

    def test_scores_clipped_to_0_100(self):
        """Extreme values should be clipped to [0, 100]."""
        # Extremely slow pitch
        row = pd.Series({
            "release_speed": 55.0,
            "release_spin_rate": 500.0,
            "pfx_x": 0.0,
            "pfx_z": 0.0,
        })
        result = pitch_quality_vector(row)
        assert result["velocity_score"] >= 0.0
        assert result["spin_score"] >= 0.0


# ── encode_count_state ────────────────────────────────────────────────────


class TestEncodeCountState:
    """Tests for count encoding."""

    def test_all_valid_counts(self):
        """All 12 valid counts map to unique indices 0-11."""
        indices = set()
        for b in range(4):
            for s in range(3):
                idx = encode_count_state(b, s)
                indices.add(idx)
                assert 0 <= idx <= 11
        assert len(indices) == 12

    def test_specific_counts(self):
        assert encode_count_state(0, 0) == 0
        assert encode_count_state(3, 2) == 11
        assert encode_count_state(1, 1) == 4

    def test_balls_out_of_range(self):
        with pytest.raises(ValueError, match="balls"):
            encode_count_state(4, 0)
        with pytest.raises(ValueError, match="balls"):
            encode_count_state(-1, 0)

    def test_strikes_out_of_range(self):
        with pytest.raises(ValueError, match="strikes"):
            encode_count_state(0, 3)
        with pytest.raises(ValueError, match="strikes"):
            encode_count_state(0, -1)


# ── tunnel_distance ───────────────────────────────────────────────────────


class TestTunnelDistance:
    """Tests for tunnel distance between pitch pairs."""

    def test_identical_location(self):
        p1 = pd.Series({"plate_x": 0.0, "plate_z": 2.5})
        p2 = pd.Series({"plate_x": 0.0, "plate_z": 2.5})
        assert tunnel_distance(p1, p2) == pytest.approx(0.0)

    def test_known_distance(self):
        p1 = pd.Series({"plate_x": 0.0, "plate_z": 0.0})
        p2 = pd.Series({"plate_x": 3.0, "plate_z": 4.0})
        assert tunnel_distance(p1, p2) == pytest.approx(5.0)

    def test_missing_values_return_nan(self):
        p1 = pd.Series({"plate_x": 0.0, "plate_z": 2.5})
        p2 = pd.Series({"plate_x": np.nan, "plate_z": 2.5})
        assert np.isnan(tunnel_distance(p1, p2))


# ── release_point_distance ───────────────────────────────────────────────


class TestReleasePointDistance:
    """Tests for release point drift."""

    def test_no_drift(self):
        row = pd.Series({
            "release_pos_x": -1.5,
            "release_pos_y": 55.0,
            "release_pos_z": 5.8,
        })
        baseline = pd.Series({
            "release_pos_x": -1.5,
            "release_pos_y": 55.0,
            "release_pos_z": 5.8,
        })
        assert release_point_distance(row, baseline) == pytest.approx(0.0)

    def test_known_distance(self):
        row = pd.Series({
            "release_pos_x": 0.0,
            "release_pos_y": 0.0,
            "release_pos_z": 0.0,
        })
        baseline = pd.Series({
            "release_pos_x": 1.0,
            "release_pos_y": 2.0,
            "release_pos_z": 2.0,
        })
        assert release_point_distance(row, baseline) == pytest.approx(3.0)

    def test_missing_dimension_returns_nan(self):
        row = pd.Series({"release_pos_x": 1.0})
        baseline = pd.Series({
            "release_pos_x": 0.0,
            "release_pos_y": 0.0,
            "release_pos_z": 0.0,
        })
        assert np.isnan(release_point_distance(row, baseline))


# ── classify_pitch_outcome ────────────────────────────────────────────────


class TestClassifyPitchOutcome:
    """Tests for outcome classification."""

    def test_whiff_descriptions(self):
        for desc in ["swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"]:
            assert classify_pitch_outcome(desc) == "whiff"

    def test_called_strike(self):
        assert classify_pitch_outcome("called_strike") == "called_strike"

    def test_ball_descriptions(self):
        for desc in ["ball", "blocked_ball", "intent_ball", "pitchout"]:
            assert classify_pitch_outcome(desc) == "ball"

    def test_in_play(self):
        for desc in ["hit_into_play", "hit_into_play_no_out", "hit_into_play_score"]:
            assert classify_pitch_outcome(desc) == "in_play"

    def test_foul(self):
        assert classify_pitch_outcome("foul") == "foul"
        assert classify_pitch_outcome("foul_bunt") == "foul"

    def test_hbp(self):
        assert classify_pitch_outcome("hit_by_pitch") == "hbp"

    def test_none_returns_unknown(self):
        assert classify_pitch_outcome(None) == "unknown"

    def test_unrecognised_returns_unknown(self):
        assert classify_pitch_outcome("some_weird_event") == "unknown"

    def test_case_insensitive(self):
        assert classify_pitch_outcome("Swinging_Strike") == "whiff"

    def test_whitespace_handling(self):
        assert classify_pitch_outcome("  ball  ") == "ball"


# ── compute_run_value ─────────────────────────────────────────────────────


class TestComputeRunValue:
    """Tests for run value computation."""

    def test_league_average_woba(self):
        """League average wOBA should produce ~0 run value."""
        result = compute_run_value(0.320, 1.0)
        assert result == pytest.approx(0.0)

    def test_zero_denom_returns_none(self):
        assert compute_run_value(0.5, 0.0) is None

    def test_none_denom_returns_none(self):
        assert compute_run_value(0.5, None) is None

    def test_none_value_returns_none(self):
        assert compute_run_value(None, 1.0) is None

    def test_home_run_positive(self):
        """A home run (high wOBA) should produce positive run value."""
        result = compute_run_value(2.0, 1.0)
        assert result is not None
        assert result > 0

    def test_strikeout_negative(self):
        """A strikeout (wOBA=0) should produce negative run value."""
        result = compute_run_value(0.0, 1.0)
        assert result is not None
        assert result < 0


# ── Integration with real sampled data ───────────────────────────────────


class TestFeaturesWithRealData:
    """Tests that use the sampled test database."""

    def test_z_scores_on_real_data(self, sample_pitches_df):
        """z-scores on real velocity data should have mean ~0, std ~1."""
        if "release_speed" not in sample_pitches_df.columns:
            pytest.skip("No release_speed column in test data.")
        df = sample_pitches_df.dropna(subset=["release_speed"])
        if len(df) < 5:
            pytest.skip("Not enough non-null velocities.")
        result = compute_z_scores(df, ["release_speed"])
        assert abs(result["release_speed_z"].mean()) < 0.1
        assert abs(result["release_speed_z"].std() - 1.0) < 0.1

    def test_pitch_quality_on_real_rows(self, sample_pitches_df):
        """pitch_quality_vector should return valid dicts on real data."""
        for _, row in sample_pitches_df.head(5).iterrows():
            result = pitch_quality_vector(row)
            assert 0 <= result["composite"] <= 100

    def test_classify_outcome_on_real_data(self, sample_pitches_df):
        """All real descriptions should map to known categories."""
        valid_categories = {"whiff", "called_strike", "ball", "foul", "in_play", "hbp", "unknown"}
        for desc in sample_pitches_df["description"].dropna().unique():
            cat = classify_pitch_outcome(desc)
            assert cat in valid_categories, f"Unmapped description: {desc} -> {cat}"
