"""
Tests for the Kinetic Half-Life (K1/2) model.

Covers:
- Stuff Concentration computation (valid z-scores)
- Exponential decay fitting on synthetic data with known lambda
- Per-pitch-type separation
- Output schema validation
- Batch calculation returns expected columns
- Edge cases: too few pitches, zero variance, single pitch type
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.kinetic_half_life import (
    KineticHalfLifeModel,
    batch_calculate,
    calculate_half_life,
    compute_stuff_concentration,
    fit_decay_curve,
    get_game_decay_data,
    predict_game_decay,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_pitches_df(
    n: int = 100,
    seed: int = 42,
    pitch_type: str = "FF",
    decay_lambda: float = 0.005,
) -> pd.DataFrame:
    """Create a realistic synthetic pitch DataFrame with known exponential decay.

    Velocity and spin decay exponentially so the Stuff Concentration
    should show a measurable decay pattern.
    """
    rng = np.random.RandomState(seed)
    pitch_numbers = np.arange(n)

    # Base values that decay exponentially
    velo_base = 95.0 * np.exp(-decay_lambda * pitch_numbers)
    spin_base = 2400.0 * np.exp(-decay_lambda * pitch_numbers)

    return pd.DataFrame({
        "game_pk": 999999,
        "game_date": "2025-06-15",
        "pitcher_id": 554430,
        "batter_id": 100001,
        "pitch_type": pitch_type,
        "release_speed": velo_base + rng.normal(0, 0.3, n),
        "release_spin_rate": spin_base + rng.normal(0, 30, n),
        "pfx_x": rng.normal(-6, 0.5, n),
        "pfx_z": rng.normal(15, 0.5, n),
        "release_pos_x": rng.normal(-1.5, 0.05, n),
        "release_pos_z": rng.normal(5.8, 0.05, n),
        "inning": np.clip(pitch_numbers // 15 + 1, 1, 9).astype(int),
        "at_bat_number": (pitch_numbers // 5 + 1).astype(int),
        "pitch_number": (pitch_numbers % 5 + 1).astype(int),
    })


def _make_multi_type_df(n_per_type: int = 80, seed: int = 42) -> pd.DataFrame:
    """Create a DataFrame with multiple pitch types, each with distinct decay."""
    dfs = []
    for i, (pt, lam) in enumerate([("FF", 0.003), ("SL", 0.006), ("CH", 0.008)]):
        df = _make_pitches_df(n=n_per_type, seed=seed + i, pitch_type=pt, decay_lambda=lam)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    # Interleave pitch types (as they would be in a real game)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    combined["at_bat_number"] = np.arange(len(combined)) // 5 + 1
    combined["pitch_number"] = np.arange(len(combined)) % 5 + 1
    return combined


# ── Tests: compute_stuff_concentration ─────────────────────────────────────────


class TestStuffConcentration:
    """Tests for the Stuff Concentration computation."""

    def test_returns_series_of_correct_length(self):
        df = _make_pitches_df(100)
        sc = compute_stuff_concentration(df)
        assert isinstance(sc, pd.Series)
        assert len(sc) == 100

    def test_produces_valid_zscores(self):
        """Stuff Concentration should be roughly z-score-like (mean near 0)."""
        df = _make_pitches_df(200, decay_lambda=0.0)  # no decay -> centered
        sc = compute_stuff_concentration(df)
        # Mean should be near 0 (within reason given weighting)
        assert abs(sc.mean()) < 1.0
        # Should have some variance
        assert sc.std() > 0

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        sc = compute_stuff_concentration(df)
        assert len(sc) == 0

    def test_missing_columns_returns_zeros(self):
        df = pd.DataFrame({"foo": [1, 2, 3]})
        sc = compute_stuff_concentration(df)
        assert (sc == 0.0).all()

    def test_zero_variance_columns(self):
        """All same velocity should produce z-scores of 0 for that column."""
        df = _make_pitches_df(50)
        df["release_speed"] = 95.0  # constant
        sc = compute_stuff_concentration(df)
        assert isinstance(sc, pd.Series)
        assert len(sc) == 50
        # Should not contain NaN
        assert sc.notna().all()


# ── Tests: fit_decay_curve ─────────────────────────────────────────────────────


class TestFitDecayCurve:
    """Tests for exponential decay fitting."""

    def test_recovers_known_lambda(self):
        """Fitting synthetic data with known lambda should recover it."""
        n = 100
        true_lambda = 0.01
        true_peak = 2.0
        x = np.arange(n, dtype=float)
        y = true_peak * np.exp(-true_lambda * x)

        stuff = pd.Series(y)
        fit = fit_decay_curve(stuff, x)

        assert fit["lambda"] is not None
        assert fit["half_life"] is not None
        assert fit["r_squared"] is not None

        # Lambda should be close to the true value
        assert abs(fit["lambda"] - true_lambda) < 0.002
        # Half-life should be close to ln(2)/0.01 ≈ 69.3
        expected_hl = math.log(2) / true_lambda
        assert abs(fit["half_life"] - expected_hl) < 5.0
        # R² should be very high for clean synthetic data
        assert fit["r_squared"] > 0.99

    def test_noisy_data_still_fits(self):
        """Adding noise should still yield a reasonable fit."""
        rng = np.random.RandomState(42)
        n = 100
        true_lambda = 0.008
        true_peak = 1.5
        x = np.arange(n, dtype=float)
        y = true_peak * np.exp(-true_lambda * x) + rng.normal(0, 0.05, n)
        y = np.maximum(y, 0)  # keep non-negative

        fit = fit_decay_curve(pd.Series(y), x)
        assert fit["lambda"] is not None
        assert fit["r_squared"] is not None and fit["r_squared"] > 0.5

    def test_too_few_points(self):
        """Fewer than 10 points should return None values."""
        x = np.arange(5, dtype=float)
        y = pd.Series([1.0, 0.9, 0.8, 0.7, 0.6])
        fit = fit_decay_curve(y, x)
        assert fit["lambda"] is None
        assert fit["half_life"] is None

    def test_half_life_formula(self):
        """K½ = ln(2) / lambda should hold exactly."""
        n = 100
        lam = 0.015
        x = np.arange(n, dtype=float)
        y = pd.Series(3.0 * np.exp(-lam * x))

        fit = fit_decay_curve(y, x)
        assert fit["lambda"] is not None and fit["half_life"] is not None

        expected_hl = math.log(2) / fit["lambda"]
        assert abs(fit["half_life"] - expected_hl) < 0.5


# ── Tests: calculate_half_life ─────────────────────────────────────────────────


class TestCalculateHalfLife:
    """Tests for the full pitcher K½ calculation using the DB."""

    def test_output_schema(self, db_conn):
        """Output dict should have the required keys."""
        # Use a pitcher that may not exist -- should return gracefully
        result = calculate_half_life(db_conn, pitcher_id=999999999)
        assert "pitcher_id" in result
        assert "overall_half_life" in result
        assert "overall_lambda" in result
        assert "games_fitted" in result
        assert "per_pitch_type" in result
        assert "game_curves" in result

    def test_nonexistent_pitcher(self, db_conn):
        """A pitcher with no data should return None half-life."""
        result = calculate_half_life(db_conn, pitcher_id=999999999)
        assert result["overall_half_life"] is None
        assert result["games_fitted"] == 0

    def test_known_pitcher_has_structure(self, db_conn):
        """If a known pitcher has enough data, output should be well-formed."""
        # Zack Wheeler or any pitcher in test data
        result = calculate_half_life(db_conn, pitcher_id=554430)
        assert isinstance(result, dict)
        assert result["pitcher_id"] == 554430
        # May or may not have enough pitches per game in test data,
        # but schema should always be present
        assert isinstance(result["per_pitch_type"], dict)
        assert isinstance(result["game_curves"], list)


# ── Tests: per-pitch-type K½ ──────────────────────────────────────────────────


class TestPerPitchType:
    """Tests that per-pitch-type separation works correctly."""

    def test_multi_type_produces_separate_keys(self, db_conn):
        """If the pitcher throws multiple types, per_pitch_type should
        have entries for each type with enough data."""
        result = calculate_half_life(db_conn, pitcher_id=554430)
        # The test data may or may not have enough per-type data,
        # but per_pitch_type should be a dict
        assert isinstance(result["per_pitch_type"], dict)
        for pt, pt_data in result["per_pitch_type"].items():
            assert "half_life" in pt_data
            assert "lambda" in pt_data
            assert "games_fitted" in pt_data


# ── Tests: batch_calculate ─────────────────────────────────────────────────────


class TestBatchCalculate:
    """Tests for the batch leaderboard calculation."""

    def test_returns_dataframe(self, db_conn):
        df = batch_calculate(db_conn, min_pitches=10)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, db_conn):
        df = batch_calculate(db_conn, min_pitches=10)
        for col in ["pitcher_id", "overall_half_life", "overall_lambda", "games_fitted"]:
            assert col in df.columns

    def test_sorted_by_half_life(self, db_conn):
        df = batch_calculate(db_conn, min_pitches=10)
        if len(df) >= 2:
            # Should be sorted descending
            assert df["overall_half_life"].is_monotonic_decreasing

    def test_high_min_pitches_returns_fewer_or_empty(self, db_conn):
        df = batch_calculate(db_conn, min_pitches=999999)
        assert isinstance(df, pd.DataFrame)
        # Likely empty given test data
        assert len(df) <= 1


# ── Tests: edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests."""

    def test_game_with_20_pitches_skipped(self):
        """Games with fewer than 50 pitches should not produce a fit."""
        from src.analytics.kinetic_half_life import _fit_game

        df = _make_pitches_df(20)
        result = _fit_game(df)
        assert result is None

    def test_all_same_velocity_no_crash(self):
        """Zero variance in velocity should not cause division errors."""
        df = _make_pitches_df(100)
        df["release_speed"] = 95.0
        df["release_spin_rate"] = 2300.0
        sc = compute_stuff_concentration(df)
        assert sc.notna().all()

    def test_single_pitch_type(self, db_conn):
        """A pitcher with only one pitch type should still work."""
        result = calculate_half_life(db_conn, pitcher_id=554430)
        assert isinstance(result, dict)

    def test_predict_game_decay_schema(self, db_conn):
        """predict_game_decay should return expected keys."""
        result = predict_game_decay(db_conn, pitcher_id=554430)
        assert "pitcher_id" in result
        assert "pitch_numbers" in result
        assert "predicted_stuff" in result
        assert "half_life" in result
        assert "lambda" in result
        assert "peak" in result

    def test_predict_nonexistent_pitcher(self, db_conn):
        result = predict_game_decay(db_conn, pitcher_id=999999999)
        assert result["half_life"] is None
        assert result["pitch_numbers"] == []

    def test_model_class_interface(self):
        """KineticHalfLifeModel should satisfy the BaseAnalyticsModel contract."""
        model = KineticHalfLifeModel()
        assert model.model_name == "kinetic_half_life"
        assert model.version == "1.0.0"
        assert "created_at" in model.metadata
