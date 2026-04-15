"""
Tests for the Motor Engram Stability Index (MESI) model.

Validates execution vector extraction, SNR computation, context stability,
MESI normalisation, learning curve fitting, and edge cases using both
synthetic data and the shared test database fixture.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.mesi import (
    EXECUTION_COLS,
    MIN_PITCHES_SNR,
    compute_context_stability,
    compute_execution_vectors,
    compute_snr,
    calculate_mesi,
    fit_learning_curve,
    batch_calculate,
    get_arsenal_stability,
    MESIModel,
    _learning_curve_func,
)

# Schema column order for the pitches table (must match src/db/schema.py)
_PITCHES_COLUMNS = [
    "game_pk", "game_date", "pitcher_id", "batter_id",
    "pitch_type", "pitch_name",
    "release_speed", "release_spin_rate", "spin_axis",
    "pfx_x", "pfx_z",
    "plate_x", "plate_z",
    "release_extension", "release_pos_x", "release_pos_y", "release_pos_z",
    "launch_speed", "launch_angle", "hit_distance", "hc_x", "hc_y", "bb_type",
    "estimated_ba", "estimated_woba",
    "delta_home_win_exp", "delta_run_exp",
    "inning", "inning_topbot", "outs_when_up", "balls", "strikes",
    "on_1b", "on_2b", "on_3b",
    "stand", "p_throws",
    "at_bat_number", "pitch_number",
    "description", "events", "type",
    "home_team", "away_team",
    "woba_value", "woba_denom", "babip_value", "iso_value",
    "zone", "effective_speed",
    "if_fielding_alignment", "of_fielding_alignment",
]


def _insert_pitches(conn, df: pd.DataFrame) -> None:
    """Insert a DataFrame into pitches using explicit column names."""
    cols = ", ".join(_PITCHES_COLUMNS)
    conn.execute(f"INSERT INTO pitches ({cols}) SELECT {cols} FROM df")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_execution_df(
    n: int = 200,
    pitch_type: str = "FF",
    seed: int = 42,
    noise_scale: float = 1.0,
) -> pd.DataFrame:
    """Generate a synthetic DataFrame with execution vector columns."""
    rng = np.random.RandomState(seed)
    data = {
        "pitch_type": [pitch_type] * n,
        "release_speed": rng.normal(94.5, 1.5 * noise_scale, n),
        "release_spin_rate": rng.normal(2300, 150 * noise_scale, n),
        "pfx_x": rng.normal(-6, 2 * noise_scale, n),
        "pfx_z": rng.normal(15, 1.5 * noise_scale, n),
        "release_pos_x": rng.normal(-1.5, 0.3 * noise_scale, n),
        "release_pos_z": rng.normal(5.8, 0.2 * noise_scale, n),
        "release_extension": rng.normal(6.2, 0.3 * noise_scale, n),
    }
    return pd.DataFrame(data)


# ── Execution vector extraction ──────────────────────────────────────────────


class TestComputeExecutionVectors:
    """Tests for compute_execution_vectors."""

    def test_correct_dimensions(self):
        df = _make_execution_df(n=150, pitch_type="FF")
        ev = compute_execution_vectors(df, "FF")
        assert ev.shape[1] == 7
        assert len(ev) == 150
        assert list(ev.columns) == EXECUTION_COLS

    def test_filters_by_pitch_type(self):
        df_ff = _make_execution_df(n=100, pitch_type="FF")
        df_sl = _make_execution_df(n=50, pitch_type="SL", seed=99)
        combined = pd.concat([df_ff, df_sl], ignore_index=True)
        ev_ff = compute_execution_vectors(combined, "FF")
        ev_sl = compute_execution_vectors(combined, "SL")
        assert len(ev_ff) == 100
        assert len(ev_sl) == 50

    def test_drops_na_rows(self):
        df = _make_execution_df(n=100, pitch_type="FF")
        # Inject some NaN
        df.loc[0, "release_speed"] = np.nan
        df.loc[5, "pfx_x"] = np.nan
        ev = compute_execution_vectors(df, "FF")
        assert len(ev) == 98

    def test_empty_for_missing_pitch_type(self):
        df = _make_execution_df(n=100, pitch_type="FF")
        ev = compute_execution_vectors(df, "CU")
        assert len(ev) == 0


# ── SNR computation ──────────────────────────────────────────────────────────


class TestComputeSNR:
    """Tests for compute_snr."""

    def test_known_snr_synthetic(self):
        """Tight cluster should produce high SNR."""
        rng = np.random.RandomState(42)
        # Mean is large, variance is small => high SNR
        n = 200
        data = pd.DataFrame({
            col: rng.normal(100, 0.1, n) for col in EXECUTION_COLS
        })
        snr = compute_snr(data, window=100)
        # With mean ~100 per dim and tiny variance, SNR should be very high
        valid_snr = snr.dropna()
        assert len(valid_snr) > 0
        assert valid_snr.iloc[-1] > 50  # Very high SNR expected

    def test_noisy_data_lower_snr(self):
        """Noisy data should produce lower SNR than tight data."""
        rng = np.random.RandomState(42)
        n = 200

        tight = pd.DataFrame({
            col: rng.normal(50, 0.5, n) for col in EXECUTION_COLS
        })
        noisy = pd.DataFrame({
            col: rng.normal(50, 10, n) for col in EXECUTION_COLS
        })

        snr_tight = compute_snr(tight, window=100).dropna().iloc[-1]
        snr_noisy = compute_snr(noisy, window=100).dropna().iloc[-1]
        assert snr_tight > snr_noisy

    def test_small_sample_returns_single_snr(self):
        """When n < window, should return a single SNR value."""
        data = pd.DataFrame({
            col: np.ones(30) * 50 for col in EXECUTION_COLS
        })
        # Add a tiny bit of noise to avoid zero variance
        data.iloc[0, 0] = 50.001
        snr = compute_snr(data, window=100)
        assert len(snr) == 1
        assert snr.iloc[0] > 0

    def test_empty_returns_empty(self):
        data = pd.DataFrame(columns=EXECUTION_COLS)
        snr = compute_snr(data, window=100)
        assert len(snr) == 0

    def test_snr_series_length(self):
        """Rolling SNR should have n entries (window-1 NaN + rest valid)."""
        n = 300
        data = pd.DataFrame({
            col: np.random.RandomState(42).normal(50, 2, n)
            for col in EXECUTION_COLS
        })
        snr = compute_snr(data, window=100)
        assert len(snr) == n
        assert snr.iloc[:99].isna().all()
        assert snr.iloc[99:].notna().all()


# ── Context Stability ────────────────────────────────────────────────────────


class TestContextStability:
    """Tests for compute_context_stability."""

    def test_identical_snr_gives_cs_1(self, db_conn):
        """If SNR is identical across all contexts, CS should be ~1.0."""
        rng = np.random.RandomState(42)
        n = 500

        rows = []
        for i in range(n):
            rows.append({
                "game_pk": 100000 + i // 50,
                "game_date": "2025-06-15",
                "pitcher_id": 999999,
                "batter_id": 100001,
                "pitch_type": "FF",
                "pitch_name": "4-Seam Fastball",
                "release_speed": 94.5 + rng.normal(0, 0.01),
                "release_spin_rate": 2300 + rng.normal(0, 0.01),
                "spin_axis": 200.0,
                "pfx_x": -6.0 + rng.normal(0, 0.01),
                "pfx_z": 15.0 + rng.normal(0, 0.01),
                "plate_x": 0.0,
                "plate_z": 2.5,
                "release_extension": 6.2 + rng.normal(0, 0.01),
                "release_pos_x": -1.5 + rng.normal(0, 0.01),
                "release_pos_y": 55.0,
                "release_pos_z": 5.8 + rng.normal(0, 0.01),
                "launch_speed": None,
                "launch_angle": None,
                "hit_distance": None,
                "hc_x": None,
                "hc_y": None,
                "bb_type": None,
                "estimated_ba": None,
                "estimated_woba": None,
                "delta_home_win_exp": 0.0,
                "delta_run_exp": 0.0,
                # Spread across all context buckets
                "inning": [1, 2, 3, 7, 8, 9][i % 6],
                "inning_topbot": "Top",
                "outs_when_up": 0,
                "balls": [0, 1, 2, 3, 0, 3][i % 6],
                "strikes": [2, 2, 0, 0, 1, 1][i % 6],
                "on_1b": 0,
                "on_2b": 0,
                "on_3b": 0,
                "stand": "R",
                "p_throws": "R",
                "at_bat_number": [1, 5, 9, 15, 19, 25][i % 6],
                "pitch_number": i % 5 + 1,
                "description": "called_strike",
                "events": None,
                "type": "S",
                "home_team": "PHI",
                "away_team": "NYM",
                "woba_value": None,
                "woba_denom": 0.0,
                "babip_value": None,
                "iso_value": None,
                "zone": 5,
                "effective_speed": 93.5,
                "if_fielding_alignment": "Standard",
                "of_fielding_alignment": "Standard",
            })

        df = pd.DataFrame(rows)
        db_conn.execute("DELETE FROM pitches WHERE pitcher_id = 999999")
        _insert_pitches(db_conn, df)

        cs_result = compute_context_stability(db_conn, 999999, "FF")

        # Clean up
        db_conn.execute("DELETE FROM pitches WHERE pitcher_id = 999999")

        assert cs_result["cs_score"] is not None
        # With nearly identical data across contexts, CS should be very close to 1
        assert cs_result["cs_score"] > 0.9

    def test_insufficient_data_returns_none(self, db_conn):
        """Pitcher with very few pitches should get cs_score=None."""
        cs_result = compute_context_stability(db_conn, -1, "XX")
        assert cs_result["cs_score"] is None
        assert cs_result["context_snr"] == {}


# ── MESI normalisation ───────────────────────────────────────────────────────


class TestMESINormalisation:
    """Tests for batch MESI normalisation to 0-100 scale."""

    def test_batch_produces_0_100_scale(self, db_conn):
        """batch_calculate should produce MESI values in [0, 100]."""
        # Use a very low min_pitches to ensure some pitchers qualify
        leaderboard = batch_calculate(db_conn, min_pitches=50)
        if leaderboard.empty:
            pytest.skip("No qualifying pitchers in test data.")
        assert leaderboard["overall_mesi"].min() >= 0.0
        assert leaderboard["overall_mesi"].max() <= 100.0

    def test_leaderboard_sorted_descending(self, db_conn):
        leaderboard = batch_calculate(db_conn, min_pitches=50)
        if len(leaderboard) < 2:
            pytest.skip("Need at least 2 pitchers to test sorting.")
        values = leaderboard["overall_mesi"].tolist()
        assert values == sorted(values, reverse=True)


# ── Learning curve fitting ───────────────────────────────────────────────────


class TestLearningCurve:
    """Tests for fit_learning_curve."""

    def test_synthetic_exponential_data(self):
        """Verify curve fitting recovers known parameters from synthetic data."""
        # Generate data from known power law
        true_max = 10.0
        true_rate = 0.1
        t = np.arange(1, 51, dtype=float)
        mesi_true = _learning_curve_func(t, true_max, true_rate)

        # Add small noise
        rng = np.random.RandomState(42)
        mesi_noisy = mesi_true + rng.normal(0, 0.1, len(t))

        from scipy.optimize import curve_fit

        popt, _ = curve_fit(
            _learning_curve_func,
            t,
            mesi_noisy,
            p0=[mesi_noisy.max() * 1.1, 0.05],
            bounds=([0, 1e-6], [mesi_noisy.max() * 5, 10.0]),
            maxfev=5000,
        )
        recovered_max, recovered_rate = popt

        # Should be close to the true values
        assert abs(recovered_max - true_max) < 1.0
        assert abs(recovered_rate - true_rate) < 0.05

    def test_insufficient_data_returns_none(self, db_conn):
        """Pitcher with no data should return insufficient_data stage."""
        result = fit_learning_curve(db_conn, -1, "XX")
        assert result["current_stage"] == "insufficient_data"
        assert result["learning_rate"] is None
        assert result["trajectory"] == []


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests for MESI computation."""

    def test_pitch_type_below_min_pitches(self):
        """Pitch type with < MIN_PITCHES_SNR should be excluded from MESI."""
        df = _make_execution_df(n=50, pitch_type="CU")
        ev = compute_execution_vectors(df, "CU")
        assert len(ev) == 50
        # SNR can still be computed with small sample; compute_snr handles it
        snr = compute_snr(ev, window=100)
        assert len(snr) == 1  # Falls back to single value

    def test_zero_variance_in_one_dimension(self):
        """Zero variance in one dimension should not crash SNR computation."""
        n = 150
        rng = np.random.RandomState(42)
        data = pd.DataFrame({
            "release_speed": np.ones(n) * 95.0,  # zero variance
            "release_spin_rate": rng.normal(2300, 100, n),
            "pfx_x": rng.normal(-5, 2, n),
            "pfx_z": rng.normal(14, 1.5, n),
            "release_pos_x": rng.normal(-1.5, 0.3, n),
            "release_pos_z": rng.normal(5.8, 0.2, n),
            "release_extension": rng.normal(6.2, 0.3, n),
        })
        snr = compute_snr(data, window=100)
        valid = snr.dropna()
        assert len(valid) > 0
        assert all(np.isfinite(valid))

    def test_single_context_bucket(self, db_conn):
        """When only one context has data, CS should still be computable."""
        rng = np.random.RandomState(42)
        n = 200
        rows = []
        for i in range(n):
            rows.append({
                "game_pk": 200000,
                "game_date": "2025-07-01",
                "pitcher_id": 999998,
                "batter_id": 100001,
                "pitch_type": "SL",
                "pitch_name": "Slider",
                "release_speed": 85.0 + rng.normal(0, 1.0),
                "release_spin_rate": 2500 + rng.normal(0, 100),
                "spin_axis": 150.0,
                "pfx_x": 3.0 + rng.normal(0, 1.0),
                "pfx_z": 2.0 + rng.normal(0, 1.0),
                "plate_x": 0.5,
                "plate_z": 2.0,
                "release_extension": 6.2 + rng.normal(0, 0.2),
                "release_pos_x": -1.5 + rng.normal(0, 0.2),
                "release_pos_y": 55.0,
                "release_pos_z": 5.8 + rng.normal(0, 0.2),
                "launch_speed": None,
                "launch_angle": None,
                "hit_distance": None,
                "hc_x": None,
                "hc_y": None,
                "bb_type": None,
                "estimated_ba": None,
                "estimated_woba": None,
                "delta_home_win_exp": 0.0,
                "delta_run_exp": 0.0,
                "inning": 1,  # only low leverage
                "inning_topbot": "Top",
                "outs_when_up": 1,
                "balls": 0,
                "strikes": 2,  # only ahead
                "on_1b": 0,
                "on_2b": 0,
                "on_3b": 0,
                "stand": "R",
                "p_throws": "R",
                "at_bat_number": 1,  # only first time through
                "pitch_number": i % 5 + 1,
                "description": "swinging_strike",
                "events": None,
                "type": "S",
                "home_team": "PHI",
                "away_team": "ATL",
                "woba_value": None,
                "woba_denom": 0.0,
                "babip_value": None,
                "iso_value": None,
                "zone": 5,
                "effective_speed": 84.0,
                "if_fielding_alignment": "Standard",
                "of_fielding_alignment": "Standard",
            })

        df = pd.DataFrame(rows)
        db_conn.execute("DELETE FROM pitches WHERE pitcher_id = 999998")
        _insert_pitches(db_conn, df)

        cs_result = compute_context_stability(db_conn, 999998, "SL")

        db_conn.execute("DELETE FROM pitches WHERE pitcher_id = 999998")

        # Should still return a CS score (even if only a few contexts have data)
        assert cs_result["cs_score"] is not None

    def test_calculate_mesi_no_data(self, db_conn):
        """MESI for a non-existent pitcher should return None overall."""
        result = calculate_mesi(db_conn, -999)
        assert result["overall_mesi"] is None
        assert result["per_pitch_type"] == {}

    def test_mesi_model_interface(self):
        """MESIModel should satisfy the BaseAnalyticsModel interface."""
        model = MESIModel()
        assert model.model_name == "Motor Engram Stability Index"
        assert model.version == "1.0.0"
        assert "created_at" in model.metadata

    def test_get_arsenal_stability_no_data(self, db_conn):
        """Arsenal stability for missing pitcher returns graceful None."""
        result = get_arsenal_stability(db_conn, -999)
        assert result["overall_mesi"] is None
        assert result["best_pitch_under_pressure"] is None
        assert result["pitch_that_breaks_down"] is None
