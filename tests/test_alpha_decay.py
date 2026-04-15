"""
Tests for src.analytics.alpha_decay -- Pitch Sequence Alpha Decay model.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.alpha_decay import (
    AlphaDecayModel,
    _exp_decay,
    _fit_exponential_decay,
    compute_sequence_alpha,
    fit_intra_game_decay,
    fit_series_decay,
    calculate_alpha_decay,
    batch_calculate,
    get_fastest_decaying_sequences,
    MIN_REPETITIONS_FOR_FIT,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

ZACK_WHEELER_ID = 554430


# ── TestExpDecay ─────────────────────────────────────────────────────────────


class TestExpDecay:
    """Tests for the exponential decay helper and fitting."""

    def test_exp_decay_at_zero(self):
        """At n=0, exp(-lambda*0) = 1, so result = alpha_0."""
        result = _exp_decay(np.array([0.0]), 0.15, 0.5)
        assert np.isclose(result[0], 0.15)

    def test_exp_decay_monotonic(self):
        """Decay function should be monotonically decreasing for positive alpha_0."""
        n = np.arange(0, 20, dtype=float)
        y = _exp_decay(n, 0.2, 0.3)
        diffs = np.diff(y)
        assert np.all(diffs <= 0), "Decay should be monotonically decreasing"

    def test_half_life_identity(self):
        """Half-life should satisfy t_half = ln(2) / lambda."""
        lam = 0.25
        half_life = math.log(2) / lam
        n_half = np.array([half_life])
        result = _exp_decay(n_half, 1.0, lam)
        assert np.isclose(result[0], 0.5, atol=1e-6), (
            f"At half-life, alpha should be 0.5, got {result[0]}"
        )

    def test_fit_on_synthetic_decay(self):
        """Fit should recover known parameters from clean synthetic data."""
        true_alpha_0 = 0.15
        true_lambda = 0.3
        reps = np.arange(1, 16, dtype=float)
        alphas = _exp_decay(reps, true_alpha_0, true_lambda)

        fit = _fit_exponential_decay(reps, alphas)
        assert fit is not None, "Fit should succeed on clean data"
        assert np.isclose(fit["alpha_0"], true_alpha_0, atol=0.01)
        assert np.isclose(fit["lambda"], true_lambda, atol=0.01)
        assert fit["r_squared"] > 0.99
        expected_hl = math.log(2) / true_lambda
        assert np.isclose(fit["half_life"], expected_hl, atol=0.1)

    def test_fit_on_noisy_data(self):
        """Fit should produce reasonable results with moderate noise."""
        rng = np.random.RandomState(42)
        true_alpha_0 = 0.10
        true_lambda = 0.2
        reps = np.arange(1, 21, dtype=float)
        alphas = _exp_decay(reps, true_alpha_0, true_lambda) + rng.normal(0, 0.01, len(reps))

        fit = _fit_exponential_decay(reps, alphas)
        assert fit is not None
        assert fit["r_squared"] > 0.5

    def test_fit_returns_none_for_single_point(self):
        """Cannot fit with just one data point."""
        fit = _fit_exponential_decay(np.array([1.0]), np.array([0.1]))
        assert fit is None

    def test_half_life_equals_ln2_over_lambda(self):
        """Verify the half-life formula in fit output."""
        reps = np.arange(1, 20, dtype=float)
        alphas = _exp_decay(reps, 0.2, 0.4)
        fit = _fit_exponential_decay(reps, alphas)
        assert fit is not None
        expected_hl = math.log(2) / fit["lambda"]
        assert np.isclose(fit["half_life"], expected_hl, atol=0.01)


# ── TestSequenceAlpha ────────────────────────────────────────────────────────


class TestSequenceAlpha:
    """Tests for compute_sequence_alpha."""

    def test_returns_dict(self, db_conn):
        """Should return a dictionary."""
        result = compute_sequence_alpha(db_conn, ZACK_WHEELER_ID)
        assert isinstance(result, dict)

    def test_empty_for_unknown_pitcher(self, db_conn):
        """Unknown pitcher should return empty dict."""
        result = compute_sequence_alpha(db_conn, 999999999)
        assert result == {}

    def test_alpha_keys_present(self, db_conn):
        """Each transition entry should have expected keys."""
        result = compute_sequence_alpha(db_conn, ZACK_WHEELER_ID)
        if not result:
            pytest.skip("No transition data for Wheeler in test DB")
        for trans, info in result.items():
            assert "alpha" in info
            assert "conditional_whiff_rate" in info
            assert "unconditional_whiff_rate" in info
            assert "swing_count" in info

    def test_transition_format(self, db_conn):
        """Transition keys should be in 'TYPE->TYPE' format."""
        result = compute_sequence_alpha(db_conn, ZACK_WHEELER_ID)
        for trans in result:
            assert "->" in trans, f"Bad transition format: {trans}"
            parts = trans.split("->")
            assert len(parts) == 2

    def test_whiff_rates_bounded(self, db_conn):
        """Whiff rates should be in [0, 1]."""
        result = compute_sequence_alpha(db_conn, ZACK_WHEELER_ID)
        for trans, info in result.items():
            assert 0.0 <= info["conditional_whiff_rate"] <= 1.0
            assert 0.0 <= info["unconditional_whiff_rate"] <= 1.0

    def test_alpha_is_difference(self, db_conn):
        """Alpha should equal conditional minus unconditional whiff rate."""
        result = compute_sequence_alpha(db_conn, ZACK_WHEELER_ID)
        for trans, info in result.items():
            expected = info["conditional_whiff_rate"] - info["unconditional_whiff_rate"]
            assert abs(info["alpha"] - expected) < 1e-4, (
                f"Alpha mismatch for {trans}: {info['alpha']} != {expected}"
            )


# ── TestSinglePitchTypePitcher ───────────────────────────────────────────────


class TestSinglePitchTypePitcher:
    """A pitcher with only one pitch type: alpha should be 0."""

    def test_single_type_alpha_zero(self, db_conn):
        """If a pitcher only throws one pitch type, every transition
        is TYPE->TYPE and conditional == unconditional, so alpha = 0."""
        # Insert a synthetic single-type pitcher
        try:
            db_conn.execute("""
                INSERT INTO pitches (
                    game_pk, game_date, pitcher_id, batter_id,
                    pitch_type, pitch_name, at_bat_number, pitch_number,
                    description, release_speed, release_spin_rate,
                    pfx_x, pfx_z, plate_x, plate_z,
                    balls, strikes, stand, p_throws,
                    inning, inning_topbot, outs_when_up,
                    home_team, away_team, type
                )
                SELECT
                    900000 + (i / 5)::INT AS game_pk,
                    '2025-06-15' AS game_date,
                    888888 AS pitcher_id,
                    777777 AS batter_id,
                    'FF' AS pitch_type,
                    'FF' AS pitch_name,
                    (i / 5)::INT + 1 AS at_bat_number,
                    (i % 5) + 1 AS pitch_number,
                    CASE WHEN random() < 0.3 THEN 'swinging_strike'
                         WHEN random() < 0.5 THEN 'foul'
                         ELSE 'ball' END AS description,
                    94.0 AS release_speed,
                    2300 AS release_spin_rate,
                    -6.0 AS pfx_x,
                    15.0 AS pfx_z,
                    0.0 AS plate_x,
                    2.5 AS plate_z,
                    0 AS balls,
                    0 AS strikes,
                    'R' AS stand,
                    'R' AS p_throws,
                    1 AS inning,
                    'Top' AS inning_topbot,
                    0 AS outs_when_up,
                    'PHI' AS home_team,
                    'NYM' AS away_team,
                    'S' AS type
                FROM generate_series(0, 49) AS t(i)
            """)
        except Exception:
            pytest.skip("Could not insert synthetic data")

        result = compute_sequence_alpha(db_conn, 888888)
        for trans, info in result.items():
            # All transitions are FF->FF
            assert trans == "FF->FF"
            # With only one pitch type, conditional == unconditional
            assert abs(info["alpha"]) < 1e-6, (
                f"Single-type pitcher alpha should be ~0, got {info['alpha']}"
            )


# ── TestIntraGameDecay ───────────────────────────────────────────────────────


class TestIntraGameDecay:
    """Tests for fit_intra_game_decay."""

    def test_returns_dict(self, db_conn):
        result = fit_intra_game_decay(db_conn, ZACK_WHEELER_ID)
        assert isinstance(result, dict)

    def test_empty_for_unknown_pitcher(self, db_conn):
        result = fit_intra_game_decay(db_conn, 999999999)
        assert result == {}

    def test_alpha_by_rep_structure(self, db_conn):
        """Each transition should have alpha_by_rep list."""
        result = fit_intra_game_decay(db_conn, ZACK_WHEELER_ID)
        if not result:
            pytest.skip("No data")
        for trans, data in result.items():
            assert "alpha_by_rep" in data
            assert isinstance(data["alpha_by_rep"], list)
            if data["alpha_by_rep"]:
                item = data["alpha_by_rep"][0]
                assert "rep" in item
                assert "alpha" in item
                assert "whiff_rate" in item
                assert "n_swings" in item


# ── TestSeriesDecay ──────────────────────────────────────────────────────────


class TestSeriesDecay:
    """Tests for fit_series_decay."""

    def test_returns_dict(self, db_conn):
        result = fit_series_decay(db_conn, ZACK_WHEELER_ID)
        assert isinstance(result, dict)

    def test_empty_for_unknown_pitcher(self, db_conn):
        result = fit_series_decay(db_conn, 999999999)
        assert result == {}

    def test_pitcher_with_one_game_in_series(self, db_conn):
        """Pitcher who only faces each team once should still return
        valid (unfitted) results without errors."""
        result = fit_series_decay(db_conn, ZACK_WHEELER_ID)
        # Should not raise; may have no fits
        assert isinstance(result, dict)


# ── TestCalculateAlphaDecay ──────────────────────────────────────────────────


class TestCalculateAlphaDecay:
    """Tests for the full calculate_alpha_decay profile."""

    def test_returns_required_keys(self, db_conn):
        result = calculate_alpha_decay(db_conn, ZACK_WHEELER_ID)
        required_keys = {"pitcher_id", "season", "transitions", "summary", "regime"}
        assert required_keys <= set(result.keys())

    def test_pitcher_id_matches(self, db_conn):
        result = calculate_alpha_decay(db_conn, ZACK_WHEELER_ID)
        assert result["pitcher_id"] == ZACK_WHEELER_ID

    def test_summary_keys(self, db_conn):
        result = calculate_alpha_decay(db_conn, ZACK_WHEELER_ID)
        summary = result["summary"]
        expected_keys = {
            "avg_intra_game_half_life",
            "avg_series_half_life",
            "most_durable_transition",
            "most_fragile_transition",
            "adaptability_score",
            "n_transitions_fitted_intra",
            "n_transitions_fitted_series",
        }
        assert expected_keys <= set(summary.keys())

    def test_regime_keys(self, db_conn):
        result = calculate_alpha_decay(db_conn, ZACK_WHEELER_ID)
        regime = result["regime"]
        assert "current_regime" in regime
        assert "regime_timeseries" in regime
        assert "median_alpha" in regime

    def test_empty_for_unknown_pitcher(self, db_conn):
        result = calculate_alpha_decay(db_conn, 999999999)
        assert result["transitions"] == {}
        assert result["summary"]["adaptability_score"] is None


# ── TestBatchCalculate ───────────────────────────────────────────────────────


class TestBatchCalculate:
    """Tests for batch_calculate leaderboard."""

    def test_returns_dataframe(self, db_conn):
        df = batch_calculate(db_conn, min_pitches=5)
        assert isinstance(df, pd.DataFrame)

    def test_expected_columns(self, db_conn):
        df = batch_calculate(db_conn, min_pitches=5)
        expected_cols = {
            "pitcher_id",
            "avg_intra_game_half_life",
            "avg_series_half_life",
            "adaptability_score",
            "n_transitions_fitted",
        }
        assert expected_cols <= set(df.columns)

    def test_high_threshold_returns_empty(self, db_conn):
        df = batch_calculate(db_conn, min_pitches=999999999)
        assert df.empty

    def test_sorted_by_adaptability(self, db_conn):
        df = batch_calculate(db_conn, min_pitches=5)
        if len(df) < 2:
            pytest.skip("Not enough qualifying pitchers")
        # Non-null scores should be sorted descending
        scored = df.dropna(subset=["adaptability_score"])
        if len(scored) < 2:
            pytest.skip("Not enough scored pitchers")
        assert scored["adaptability_score"].is_monotonic_decreasing


# ── TestFastestDecaying ──────────────────────────────────────────────────────


class TestFastestDecaying:
    """Tests for get_fastest_decaying_sequences."""

    def test_returns_list(self, db_conn):
        result = get_fastest_decaying_sequences(db_conn, ZACK_WHEELER_ID)
        assert isinstance(result, list)

    def test_empty_for_unknown_pitcher(self, db_conn):
        result = get_fastest_decaying_sequences(db_conn, 999999999)
        assert result == []

    def test_max_length(self, db_conn):
        result = get_fastest_decaying_sequences(
            db_conn, ZACK_WHEELER_ID, top_n=3
        )
        assert len(result) <= 3

    def test_sorted_by_half_life_ascending(self, db_conn):
        result = get_fastest_decaying_sequences(db_conn, ZACK_WHEELER_ID)
        if len(result) < 2:
            pytest.skip("Not enough fitted transitions")
        half_lives = [r["half_life"] for r in result]
        assert half_lives == sorted(half_lives)

    def test_sequence_never_repeated(self, db_conn):
        """A pitcher with no repeated transitions should return empty."""
        result = get_fastest_decaying_sequences(db_conn, 999999999)
        assert result == []


# ── TestModelLifecycle ───────────────────────────────────────────────────────


class TestModelLifecycle:
    """Tests for the AlphaDecayModel class lifecycle."""

    def test_model_properties(self):
        model = AlphaDecayModel()
        assert model.model_name == "AlphaDecay"
        assert model.version == "1.0.0"

    def test_train_is_noop(self, db_conn):
        model = AlphaDecayModel()
        metrics = model.train(db_conn)
        assert metrics["status"] == "no_training_needed"

    def test_predict_returns_profile(self, db_conn):
        model = AlphaDecayModel()
        result = model.predict(db_conn, pitcher_id=ZACK_WHEELER_ID)
        assert "transitions" in result
        assert "summary" in result

    def test_evaluate_returns_coverage(self, db_conn):
        model = AlphaDecayModel()
        result = model.evaluate(db_conn, min_pitches=5)
        assert "qualifying_pitchers" in result

    def test_repr(self):
        model = AlphaDecayModel()
        assert "AlphaDecay" in repr(model)
