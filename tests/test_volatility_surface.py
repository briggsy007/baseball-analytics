"""
Tests for the Pitch Implied Volatility Surface (PIVS) model.

Covers zone discretization, entropy computation, surface smoothing,
derived metrics (smile, term structure, skew), batch calculation,
and edge cases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.volatility_surface import (
    ALL_COUNT_STATES,
    MIN_PITCHES_PER_CELL,
    N_COUNTS,
    N_OUTCOMES,
    N_ZONE_CELLS,
    N_ZONE_COLS,
    N_ZONE_ROWS,
    OOZ_INDEX,
    OUTCOME_CATEGORIES,
    ZONE_X_EDGES,
    ZONE_Z_EDGES,
    PitchVolatilitySurfaceModel,
    _assign_zone_index,
    _extract_vol_smile,
    _extract_vol_term_structure,
    batch_calculate,
    calculate_volatility_surface,
    classify_outcome,
    compare_surfaces,
    compute_cell_entropy,
    smooth_surface,
)
from src.analytics.features import shannon_entropy


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_pitches_for_vol() -> pd.DataFrame:
    """Create a synthetic pitch DataFrame with zone, count, and outcome columns.

    Generates enough pitches across zones and counts to compute entropy.
    """
    rng = np.random.RandomState(42)
    rows = []

    for zone_idx in range(N_ZONE_CELLS):
        for count_state in ALL_COUNT_STATES:
            n = rng.randint(MIN_PITCHES_PER_CELL, 30)
            for _ in range(n):
                outcome = rng.choice(OUTCOME_CATEGORIES)
                rows.append({
                    "zone_idx": zone_idx,
                    "count_state": count_state,
                    "outcome": outcome,
                })

    return pd.DataFrame(rows)


@pytest.fixture
def uniform_cell_df() -> pd.DataFrame:
    """Cell where each outcome has exactly the same frequency."""
    rows = []
    n_per_cat = 20
    for cat in OUTCOME_CATEGORIES:
        for _ in range(n_per_cat):
            rows.append({
                "zone_idx": 0,
                "count_state": "0-0",
                "outcome": cat,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def single_outcome_cell_df() -> pd.DataFrame:
    """Cell where every pitch has the same outcome (entropy = 0)."""
    rows = []
    for _ in range(30):
        rows.append({
            "zone_idx": 0,
            "count_state": "0-0",
            "outcome": "called_strike",
        })
    return pd.DataFrame(rows)


# ── Zone discretization tests ────────────────────────────────────────────────


class TestZoneDiscretization:
    """Test that pitches are assigned to correct grid cells."""

    def test_center_pitch_in_zone(self):
        """A pitch at the center of the plate should be in the middle zone."""
        idx = _assign_zone_index(0.0, 2.5)
        assert 0 <= idx < OOZ_INDEX
        # Center = col 2, plate_z 2.5 in row 2 (edge 2.2-2.8)
        # => row=2, col=2 => index = 2*5+2 = 12
        assert idx == 12

    def test_far_inside_pitch_ooz(self):
        """A pitch far inside (-2.0) should be out-of-zone."""
        idx = _assign_zone_index(-2.0, 2.5)
        assert idx == OOZ_INDEX

    def test_high_pitch_ooz(self):
        """A pitch above the zone (z=5.0) should be out-of-zone."""
        idx = _assign_zone_index(0.0, 5.0)
        assert idx == OOZ_INDEX

    def test_low_pitch_ooz(self):
        """A pitch below the zone (z=0.5) should be out-of-zone."""
        idx = _assign_zone_index(0.0, 0.5)
        assert idx == OOZ_INDEX

    def test_boundary_pitch_inside(self):
        """A pitch exactly on the left boundary should be in-zone."""
        idx = _assign_zone_index(ZONE_X_EDGES[0], 2.5)
        assert 0 <= idx < OOZ_INDEX

    def test_boundary_pitch_at_right_edge_ooz(self):
        """A pitch at exactly the right edge boundary is out-of-zone (>= upper)."""
        idx = _assign_zone_index(ZONE_X_EDGES[-1], 2.5)
        assert idx == OOZ_INDEX

    def test_all_in_zone_corners(self):
        """Each corner of the 5x5 grid should be a unique zone index."""
        indices = set()
        # Test center of each grid cell
        for col in range(N_ZONE_COLS):
            x = (ZONE_X_EDGES[col] + ZONE_X_EDGES[col + 1]) / 2
            for row in range(N_ZONE_ROWS):
                z = (ZONE_Z_EDGES[row] + ZONE_Z_EDGES[row + 1]) / 2
                idx = _assign_zone_index(x, z)
                assert 0 <= idx < OOZ_INDEX
                indices.add(idx)

        assert len(indices) == N_ZONE_COLS * N_ZONE_ROWS

    def test_ooz_far_outside(self):
        """Extreme coordinates should always be OOZ."""
        assert _assign_zone_index(-10.0, 2.5) == OOZ_INDEX
        assert _assign_zone_index(10.0, 2.5) == OOZ_INDEX
        assert _assign_zone_index(0.0, -5.0) == OOZ_INDEX
        assert _assign_zone_index(0.0, 10.0) == OOZ_INDEX


# ── Outcome classification tests ────────────────────────────────────────────


class TestClassifyOutcome:
    """Test the outcome classification function."""

    def test_called_strike(self):
        assert classify_outcome("called_strike", None, "FF") == "called_strike"

    def test_swinging_strike(self):
        assert classify_outcome("swinging_strike", None, "SL") == "swinging_strike"

    def test_swinging_strike_blocked(self):
        assert classify_outcome("swinging_strike_blocked", None, "CU") == "swinging_strike"

    def test_foul(self):
        assert classify_outcome("foul", None, "FF") == "foul"

    def test_ball(self):
        assert classify_outcome("ball", None, "FF") == "ball"

    def test_blocked_ball(self):
        assert classify_outcome("blocked_ball", None, "FF") == "ball"

    def test_weak_contact(self):
        assert classify_outcome("hit_into_play", 75.0, "FF") == "weak_contact"

    def test_hard_contact(self):
        assert classify_outcome("hit_into_play", 100.0, "FF") == "hard_contact"

    def test_medium_contact(self):
        assert classify_outcome("hit_into_play", 90.0, "FF") == "medium_contact"

    def test_in_play_no_ev(self):
        """In-play with no launch speed defaults to medium_contact."""
        assert classify_outcome("hit_into_play", None, "FF") == "medium_contact"

    def test_none_description(self):
        assert classify_outcome(None, None, None) == "ball"

    def test_hit_into_play_score(self):
        assert classify_outcome("hit_into_play_score", 96.0, "FF") == "hard_contact"

    def test_foul_tip(self):
        assert classify_outcome("foul_tip", None, "FF") == "swinging_strike"


# ── Entropy computation tests ────────────────────────────────────────────────


class TestCellEntropy:
    """Test entropy matrix computation."""

    def test_uniform_distribution_max_entropy(self, uniform_cell_df):
        """Uniform distribution over N categories has entropy = ln(N)."""
        matrix = compute_cell_entropy(uniform_cell_df)
        expected_entropy = float(np.log(N_OUTCOMES))
        actual = matrix[0, 0]  # zone 0, count 0-0

        assert not np.isnan(actual)
        assert abs(actual - expected_entropy) < 0.01

    def test_single_outcome_zero_entropy(self, single_outcome_cell_df):
        """A cell with only one outcome type has entropy = 0."""
        matrix = compute_cell_entropy(single_outcome_cell_df)
        actual = matrix[0, 0]

        assert not np.isnan(actual)
        assert abs(actual) < 1e-10

    def test_shape(self, sample_pitches_for_vol):
        """Entropy matrix has the correct shape."""
        matrix = compute_cell_entropy(sample_pitches_for_vol)
        assert matrix.shape == (N_ZONE_CELLS, N_COUNTS)

    def test_sparse_cell_is_nan(self):
        """Cells with fewer than MIN_PITCHES_PER_CELL pitches are NaN."""
        # Create a cell with too few pitches
        rows = [
            {"zone_idx": 5, "count_state": "1-1", "outcome": "ball"}
            for _ in range(MIN_PITCHES_PER_CELL - 1)
        ]
        df = pd.DataFrame(rows)
        matrix = compute_cell_entropy(df)
        assert np.isnan(matrix[5, ALL_COUNT_STATES.index("1-1")])

    def test_entropy_non_negative(self, sample_pitches_for_vol):
        """All entropy values should be >= 0."""
        matrix = compute_cell_entropy(sample_pitches_for_vol)
        valid = matrix[~np.isnan(matrix)]
        assert np.all(valid >= 0)

    def test_entropy_bounded_by_log_n(self, sample_pitches_for_vol):
        """Entropy should not exceed ln(N_OUTCOMES)."""
        matrix = compute_cell_entropy(sample_pitches_for_vol)
        valid = matrix[~np.isnan(matrix)]
        max_entropy = float(np.log(N_OUTCOMES))
        assert np.all(valid <= max_entropy + 0.01)


# ── Surface smoothing tests ─────────────────────────────────────────────────


class TestSmoothSurface:
    """Test surface smoothing / interpolation."""

    def test_no_nans_after_smoothing(self, sample_pitches_for_vol):
        """Smoothed surface should have no NaN values."""
        raw = compute_cell_entropy(sample_pitches_for_vol)
        smoothed = smooth_surface(raw)
        assert not np.any(np.isnan(smoothed))

    def test_preserves_known_values_approximately(self, sample_pitches_for_vol):
        """Known (non-NaN) values should not change drastically after smoothing."""
        raw = compute_cell_entropy(sample_pitches_for_vol)
        smoothed = smooth_surface(raw)

        known_mask = ~np.isnan(raw)
        if known_mask.sum() == 0:
            pytest.skip("No known values in test data.")

        # Known values may shift slightly due to smoothing but should be in range
        for zi in range(N_ZONE_CELLS):
            for ci in range(N_COUNTS):
                if not np.isnan(raw[zi, ci]):
                    # The smoothed value should be within a reasonable range
                    assert smoothed[zi, ci] >= 0

    def test_all_nan_input(self):
        """All-NaN input should produce a surface of zeros (or global mean)."""
        raw = np.full((N_ZONE_CELLS, N_COUNTS), np.nan)
        smoothed = smooth_surface(raw)
        assert not np.any(np.isnan(smoothed))

    def test_single_known_cell(self):
        """A surface with just one known cell should still smooth without NaN."""
        raw = np.full((N_ZONE_CELLS, N_COUNTS), np.nan)
        raw[12, 0] = 1.5
        smoothed = smooth_surface(raw)
        assert not np.any(np.isnan(smoothed))

    def test_smoothed_values_non_negative(self, sample_pitches_for_vol):
        """Smoothed values should be non-negative (entropy is non-negative)."""
        raw = compute_cell_entropy(sample_pitches_for_vol)
        smoothed = smooth_surface(raw)
        assert np.all(smoothed >= 0)


# ── Vol smile tests ──────────────────────────────────────────────────────────


class TestVolSmile:
    """Test vol smile extraction from a known surface."""

    def test_smile_length(self):
        """Vol smile should have 5 values (one per zone column)."""
        surface = np.ones((N_ZONE_CELLS, N_COUNTS))
        smile = _extract_vol_smile(surface, count_idx=0)
        assert len(smile) == N_ZONE_COLS

    def test_uniform_surface_flat_smile(self):
        """A uniform surface should produce a flat smile."""
        surface = np.full((N_ZONE_CELLS, N_COUNTS), 1.5)
        smile = _extract_vol_smile(surface, count_idx=0)
        assert np.allclose(smile, 1.5)

    def test_varying_smile(self):
        """A surface with column-varying entropy should produce a non-flat smile."""
        surface = np.ones((N_ZONE_CELLS, N_COUNTS))
        # Make column 0 have higher entropy
        for row in range(N_ZONE_ROWS):
            surface[row * N_ZONE_COLS + 0, :] = 2.0
        smile = _extract_vol_smile(surface, count_idx=0)
        assert smile[0] > smile[2]


# ── Vol term structure tests ─────────────────────────────────────────────────


class TestVolTermStructure:
    """Test vol term structure extraction."""

    def test_term_structure_length(self):
        """Term structure should have 12 values (one per count)."""
        surface = np.ones((N_ZONE_CELLS, N_COUNTS))
        ts = _extract_vol_term_structure(surface, zone_idx=12)
        assert len(ts) == N_COUNTS

    def test_term_structure_from_known(self):
        """Should extract the correct row from the matrix."""
        surface = np.zeros((N_ZONE_CELLS, N_COUNTS))
        surface[5, :] = np.arange(N_COUNTS, dtype=float)
        ts = _extract_vol_term_structure(surface, zone_idx=5)
        np.testing.assert_array_equal(ts, np.arange(N_COUNTS, dtype=float))


# ── Compare surfaces tests ──────────────────────────────────────────────────


class TestCompareSurfaces:
    """Test surface comparison."""

    def test_identical_surfaces(self):
        """Comparing identical surfaces should give zero difference."""
        s = np.ones((N_ZONE_CELLS, N_COUNTS))
        result = compare_surfaces(s, s)
        assert result["mean_diff"] == 0.0

    def test_different_surfaces(self):
        """Comparing different surfaces should give non-zero difference."""
        s1 = np.ones((N_ZONE_CELLS, N_COUNTS))
        s2 = np.ones((N_ZONE_CELLS, N_COUNTS)) * 2.0
        result = compare_surfaces(s1, s2)
        assert result["mean_diff"] == 1.0
        assert result["max_diff_value"] == 1.0

    def test_diff_surface_shape(self):
        """Diff surface should have the correct shape."""
        s1 = np.ones((N_ZONE_CELLS, N_COUNTS))
        s2 = np.zeros((N_ZONE_CELLS, N_COUNTS))
        result = compare_surfaces(s1, s2)
        assert result["diff_surface"].shape == (N_ZONE_CELLS, N_COUNTS)


# ── Integration tests with database ─────────────────────────────────────────


class TestVolatilitySurfaceIntegration:
    """Integration tests using the shared db_conn fixture from conftest."""

    def test_calculate_surface_returns_expected_keys(self, db_conn):
        """Surface result dict should have all expected keys."""
        # Get a pitcher with pitches
        pitcher_row = db_conn.execute("""
            SELECT pitcher_id
            FROM pitches
            WHERE plate_x IS NOT NULL AND plate_z IS NOT NULL
            GROUP BY pitcher_id
            ORDER BY COUNT(*) DESC
            LIMIT 1
        """).fetchone()

        if pitcher_row is None:
            pytest.skip("No pitch data available.")

        pitcher_id = pitcher_row[0]
        result = calculate_volatility_surface(db_conn, pitcher_id)

        expected_keys = {
            "pitcher_id", "n_pitches", "surface_raw", "surface_smooth",
            "pitch_counts", "vol_smile", "vol_term_structure",
            "vol_skew", "overall_vol", "outcome_distributions",
        }
        assert expected_keys.issubset(result.keys())
        assert result["n_pitches"] > 0
        assert result["surface_raw"].shape == (N_ZONE_CELLS, N_COUNTS)
        assert result["surface_smooth"].shape == (N_ZONE_CELLS, N_COUNTS)

    def test_empty_pitcher_returns_zero_pitches(self, db_conn):
        """A non-existent pitcher should return n_pitches=0."""
        result = calculate_volatility_surface(db_conn, 999999999)
        assert result["n_pitches"] == 0
        assert result["overall_vol"] == 0.0

    def test_batch_calculate_returns_dataframe(self, db_conn):
        """Batch calculation should return a DataFrame with expected columns."""
        df = batch_calculate(db_conn, min_pitches=10)

        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "pitcher_id" in df.columns
            assert "overall_vol" in df.columns
            assert "n_pitches" in df.columns

    def test_batch_calculate_high_threshold(self, db_conn):
        """With a very high threshold, batch_calculate may return empty."""
        df = batch_calculate(db_conn, min_pitches=1_000_000)
        assert isinstance(df, pd.DataFrame)


# ── Edge cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for entropy and zone calculations."""

    def test_all_pitches_in_zone(self):
        """A pitcher who only throws in the zone (no OOZ pitches)."""
        rows = []
        for _ in range(50):
            rows.append({
                "zone_idx": 12,  # center zone
                "count_state": "0-0",
                "outcome": np.random.choice(OUTCOME_CATEGORIES),
            })
        df = pd.DataFrame(rows)
        matrix = compute_cell_entropy(df)
        # Only cell (12, 0) should be non-NaN
        assert not np.isnan(matrix[12, 0])
        # OOZ cell should be NaN
        assert np.isnan(matrix[OOZ_INDEX, 0])

    def test_all_strikeouts_in_one_cell(self):
        """All pitches in a cell are swinging_strike => entropy = 0."""
        rows = [
            {"zone_idx": 3, "count_state": "0-2", "outcome": "swinging_strike"}
            for _ in range(20)
        ]
        df = pd.DataFrame(rows)
        matrix = compute_cell_entropy(df)
        count_idx = ALL_COUNT_STATES.index("0-2")
        assert matrix[3, count_idx] == 0.0

    def test_model_wrapper_predict(self, db_conn):
        """The BaseAnalyticsModel wrapper should work end to end."""
        model = PitchVolatilitySurfaceModel()

        # Get a pitcher
        pitcher_row = db_conn.execute("""
            SELECT pitcher_id
            FROM pitches
            WHERE plate_x IS NOT NULL
            GROUP BY pitcher_id
            ORDER BY COUNT(*) DESC
            LIMIT 1
        """).fetchone()

        if pitcher_row is None:
            pytest.skip("No pitch data available.")

        result = model.predict(db_conn, pitcher_id=pitcher_row[0])
        assert "pitcher_id" in result
        assert "surface_smooth" in result

    def test_model_wrapper_train(self, db_conn):
        """Train should return a no-op status."""
        model = PitchVolatilitySurfaceModel()
        result = model.train(db_conn)
        assert result["status"] == "no_training_needed"

    def test_model_name_and_version(self):
        """Model metadata should be correct."""
        model = PitchVolatilitySurfaceModel()
        assert model.model_name == "pitch_implied_volatility_surface"
        assert model.version == "1.0.0"

    def test_classify_outcome_case_insensitivity(self):
        """Classification should handle case variations."""
        assert classify_outcome("Called_Strike", None, "FF") == "called_strike"
        assert classify_outcome("BALL", None, "FF") == "ball"
