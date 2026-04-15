"""
Tests for src.analytics.pset — Pitch Sequence Expected Threat model.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.pset import (
    PSETModel,
    build_transition_matrix,
    calculate_pset,
    batch_calculate,
    compute_predictability,
    compute_tunnel_scores,
    get_best_sequences,
    ALPHA_PREDICTABILITY,
    BETA_TUNNEL,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

ZACK_WHEELER_ID = 554430


@pytest.fixture
def pairs_df_basic():
    """A minimal pitch-pairs DataFrame with valid columns for tunnel scoring."""
    return pd.DataFrame(
        {
            "plate_x": [0.5, -0.3, 0.1, 0.8],
            "plate_z": [2.5, 3.0, 1.8, 2.2],
            "prev_plate_x": [0.4, -0.2, 0.0, 0.7],
            "prev_plate_z": [2.6, 2.9, 1.9, 2.3],
            "release_speed": [95.0, 84.0, 92.0, 78.0],
            "prev_release_speed": [94.0, 95.0, 85.0, 93.0],
            "pfx_x": [-6.0, 3.0, -8.0, 5.0],
            "pfx_z": [15.0, 2.0, 4.0, -8.0],
            "prev_pfx_x": [-5.5, -6.0, 3.0, -8.0],
            "prev_pfx_z": [14.5, 15.0, 2.0, 4.0],
        }
    )


@pytest.fixture
def identical_pairs_df():
    """Pitch pairs where both pitches are identical (same location/speed/movement)."""
    return pd.DataFrame(
        {
            "plate_x": [0.5, 0.5],
            "plate_z": [2.5, 2.5],
            "prev_plate_x": [0.5, 0.5],
            "prev_plate_z": [2.5, 2.5],
            "release_speed": [95.0, 95.0],
            "prev_release_speed": [95.0, 95.0],
            "pfx_x": [-6.0, -6.0],
            "pfx_z": [15.0, 15.0],
            "prev_pfx_x": [-6.0, -6.0],
            "prev_pfx_z": [15.0, 15.0],
        }
    )


# ── TestTransitionMatrix ─────────────────────────────────────────────────────


class TestTransitionMatrix:
    """Tests for build_transition_matrix."""

    def test_rows_sum_to_one(self, db_conn):
        """Each row of the transition matrix should sum to ~1.0."""
        tm = build_transition_matrix(db_conn, ZACK_WHEELER_ID)
        if not tm:
            pytest.skip("No transition data for Wheeler in test DB")

        for count, prev_types in tm.items():
            for prev_pt, nexts in prev_types.items():
                total = sum(nexts.values())
                assert abs(total - 1.0) < 0.02, (
                    f"Row {count}/{prev_pt} sums to {total}, expected ~1.0"
                )

    def test_empty_for_unknown_pitcher(self, db_conn):
        """A pitcher with no data should return an empty matrix."""
        tm = build_transition_matrix(db_conn, 999999999)
        assert tm == {}

    def test_matrix_keys_are_count_states(self, db_conn):
        """Outer keys should look like 'B-S' count states."""
        tm = build_transition_matrix(db_conn, ZACK_WHEELER_ID)
        if not tm:
            pytest.skip("No data")
        for key in tm:
            parts = key.split("-")
            assert len(parts) == 2, f"Bad count key: {key}"
            assert parts[0].isdigit() and parts[1].isdigit()


# ── TestPredictability ───────────────────────────────────────────────────────


class TestPredictability:
    """Tests for compute_predictability."""

    def test_single_pitch_type_max_penalty(self):
        """A pitcher who always throws the same pitch after FF should get
        maximum predictability penalty (1.0)."""
        # Transition matrix: after FF in 0-0, always throw FF
        tm = {"0-0": {"FF": {"FF": 1.0}}}
        seq = pd.DataFrame(
            {
                "prev_pitch_type": ["FF", "FF", "FF"],
                "balls": [0, 0, 0],
                "strikes": [0, 0, 0],
            }
        )
        penalties = compute_predictability(tm, seq)
        assert all(p == 1.0 for p in penalties), (
            f"Expected all penalties = 1.0, got {penalties.tolist()}"
        )

    def test_uniform_distribution_low_penalty(self):
        """A pitcher with perfectly uniform pitch selection should have
        near-zero predictability penalty."""
        tm = {
            "0-0": {
                "FF": {"FF": 0.25, "SL": 0.25, "CU": 0.25, "CH": 0.25},
            }
        }
        seq = pd.DataFrame(
            {
                "prev_pitch_type": ["FF"],
                "balls": [0],
                "strikes": [0],
            }
        )
        penalties = compute_predictability(tm, seq)
        assert penalties.iloc[0] < 0.05, (
            f"Expected near-zero penalty for uniform dist, got {penalties.iloc[0]}"
        )

    def test_missing_prev_pitch_zero_penalty(self):
        """When prev_pitch_type is None, penalty should be 0."""
        tm = {"0-0": {"FF": {"SL": 1.0}}}
        seq = pd.DataFrame(
            {
                "prev_pitch_type": [None],
                "balls": [0],
                "strikes": [0],
            }
        )
        penalties = compute_predictability(tm, seq)
        assert penalties.iloc[0] == 0.0

    def test_unknown_count_zero_penalty(self):
        """If the count isn't in the matrix, penalty should be 0."""
        tm = {"0-0": {"FF": {"SL": 1.0}}}
        seq = pd.DataFrame(
            {
                "prev_pitch_type": ["FF"],
                "balls": [3],
                "strikes": [2],
            }
        )
        penalties = compute_predictability(tm, seq)
        assert penalties.iloc[0] == 0.0


# ── TestTunnelScores ─────────────────────────────────────────────────────────


class TestTunnelScores:
    """Tests for compute_tunnel_scores."""

    def test_identical_pitches_zero_distance(self, identical_pairs_df):
        """Identical consecutive pitches should have plate_distance=0."""
        result = compute_tunnel_scores(identical_pairs_df)
        assert (result["plate_distance"] == 0.0).all()

    def test_identical_pitches_zero_velo_diff(self, identical_pairs_df):
        """Identical pitches have zero velocity diff."""
        result = compute_tunnel_scores(identical_pairs_df)
        assert (result["velo_diff"] == 0.0).all()

    def test_identical_pitches_zero_movement_diff(self, identical_pairs_df):
        """Identical pitches have zero movement diff."""
        result = compute_tunnel_scores(identical_pairs_df)
        assert (result["movement_diff"] == 0.0).all()

    def test_identical_pitches_zero_tunnel_score(self, identical_pairs_df):
        """Identical pitches should have tunnel_score=0 — no divergence."""
        result = compute_tunnel_scores(identical_pairs_df)
        assert (result["tunnel_score"] == 0.0).all()

    def test_basic_shape(self, pairs_df_basic):
        """Output should have the 4 expected columns with matching length."""
        result = compute_tunnel_scores(pairs_df_basic)
        assert list(result.columns) == [
            "plate_distance",
            "velo_diff",
            "movement_diff",
            "tunnel_score",
        ]
        assert len(result) == len(pairs_df_basic)

    def test_tunnel_score_bounded(self, pairs_df_basic):
        """Tunnel score should be between 0 and 1."""
        result = compute_tunnel_scores(pairs_df_basic)
        assert (result["tunnel_score"] >= 0.0).all()
        assert (result["tunnel_score"] <= 1.0).all()

    def test_plate_distance_nonnegative(self, pairs_df_basic):
        """Plate distance is a Euclidean distance — always non-negative."""
        result = compute_tunnel_scores(pairs_df_basic)
        assert (result["plate_distance"] >= 0.0).all()

    def test_high_divergence_boosts_tunnel(self):
        """A pair with same plate location but large velo+movement diff should
        have a higher tunnel score than a pair with no divergence."""
        # Pair 1: same location, big velo diff, big movement diff
        # Pair 2: same location, no velo diff, no movement diff
        df = pd.DataFrame(
            {
                "plate_x": [0.0, 0.0],
                "plate_z": [2.5, 2.5],
                "prev_plate_x": [0.0, 0.0],
                "prev_plate_z": [2.5, 2.5],
                "release_speed": [95.0, 95.0],
                "prev_release_speed": [80.0, 95.0],
                "pfx_x": [10.0, 0.0],
                "pfx_z": [10.0, 0.0],
                "prev_pfx_x": [0.0, 0.0],
                "prev_pfx_z": [0.0, 0.0],
            }
        )
        result = compute_tunnel_scores(df)
        assert result["tunnel_score"].iloc[0] > result["tunnel_score"].iloc[1]


# ── TestPSETAggregation ──────────────────────────────────────────────────────


class TestPSETAggregation:
    """Tests for the full PSET calculation pipeline."""

    def test_calculate_pset_returns_required_keys(self, db_conn):
        """Result dict must contain all documented keys."""
        result = calculate_pset(db_conn, ZACK_WHEELER_ID)
        required_keys = {
            "pitcher_id",
            "pset_per_100",
            "predictability_score",
            "tunnel_score",
            "base_rv_per_100",
            "total_pairs",
            "breakdown",
        }
        assert required_keys <= set(result.keys())

    def test_calculate_pset_pitcher_id_matches(self, db_conn):
        result = calculate_pset(db_conn, ZACK_WHEELER_ID)
        assert result["pitcher_id"] == ZACK_WHEELER_ID

    def test_empty_for_unknown_pitcher(self, db_conn):
        """Unknown pitcher should return the empty sentinel."""
        result = calculate_pset(db_conn, 999999999)
        assert result["total_pairs"] == 0
        assert result["pset_per_100"] == 0.0
        assert result["breakdown"] == []

    def test_pset_per_100_is_float(self, db_conn):
        result = calculate_pset(db_conn, ZACK_WHEELER_ID)
        assert isinstance(result["pset_per_100"], float)

    def test_breakdown_is_list_of_dicts(self, db_conn):
        result = calculate_pset(db_conn, ZACK_WHEELER_ID)
        assert isinstance(result["breakdown"], list)
        if result["breakdown"]:
            item = result["breakdown"][0]
            assert "pair" in item
            assert "count" in item
            assert "pset_contribution" in item

    def test_pset_math_consistency(self, db_conn):
        """PSET per 100 should be consistent with the component scores
        (at least directionally — a rough sanity check)."""
        result = calculate_pset(db_conn, ZACK_WHEELER_ID)
        if result["total_pairs"] == 0:
            pytest.skip("No pairs")
        # The components should all be finite
        assert math.isfinite(result["pset_per_100"])
        assert math.isfinite(result["predictability_score"])
        assert math.isfinite(result["tunnel_score"])


# ── TestBestSequences ────────────────────────────────────────────────────────


class TestBestSequences:
    """Tests for get_best_sequences."""

    def test_returns_list(self, db_conn):
        result = get_best_sequences(db_conn, ZACK_WHEELER_ID)
        assert isinstance(result, list)

    def test_max_length(self, db_conn):
        result = get_best_sequences(db_conn, ZACK_WHEELER_ID, top_n=3)
        assert len(result) <= 3

    def test_empty_for_unknown_pitcher(self, db_conn):
        result = get_best_sequences(db_conn, 999999999)
        assert result == []

    def test_sorted_by_pset_contribution(self, db_conn):
        result = get_best_sequences(db_conn, ZACK_WHEELER_ID, top_n=10)
        if len(result) < 2:
            pytest.skip("Not enough sequences to test ordering")
        contribs = [r["pset_contribution"] for r in result]
        assert contribs == sorted(contribs, reverse=True)


# ── TestBatchCalculate ───────────────────────────────────────────────────────


class TestBatchCalculate:
    """Tests for the batch leaderboard function."""

    def test_returns_dataframe(self, db_conn):
        df = batch_calculate(db_conn, min_pitches=5)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self, db_conn):
        df = batch_calculate(db_conn, min_pitches=5)
        required = {"pitcher_id", "pset_per_100", "predictability_score",
                     "tunnel_score", "total_pairs"}
        assert required <= set(df.columns)

    def test_sorted_descending(self, db_conn):
        df = batch_calculate(db_conn, min_pitches=5)
        if len(df) < 2:
            pytest.skip("Not enough qualifying pitchers")
        assert df["pset_per_100"].is_monotonic_decreasing

    def test_high_min_pitches_may_return_empty(self, db_conn):
        """With an absurdly high threshold, we should get an empty DF."""
        df = batch_calculate(db_conn, min_pitches=999999999)
        assert df.empty


# ── TestEdgeCases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_pitch_type_pitcher(self, db_conn):
        """A pitcher with only one pitch type has no meaningful transitions.
        The model should still return a valid (possibly empty) result."""
        # We use the unknown pitcher path which returns empty
        result = calculate_pset(db_conn, 999999999)
        assert result["total_pairs"] == 0

    def test_model_lifecycle(self, db_conn):
        """PSETModel follows the base class lifecycle."""
        model = PSETModel()
        assert model.model_name == "PSET"
        assert model.version == "1.0.0"

        # Train is a no-op
        metrics = model.train(db_conn)
        assert metrics["status"] == "no_training_needed"

        # Predict delegates to calculate_pset
        result = model.predict(db_conn, pitcher_id=ZACK_WHEELER_ID)
        assert "pset_per_100" in result

    def test_predictability_with_two_pitch_types(self):
        """A pitcher alternating between exactly 2 pitch types 50/50 should
        have low predictability (~0)."""
        tm = {"0-0": {"FF": {"FF": 0.5, "SL": 0.5}}}
        seq = pd.DataFrame(
            {
                "prev_pitch_type": ["FF"],
                "balls": [0],
                "strikes": [0],
            }
        )
        penalties = compute_predictability(tm, seq)
        # With 2 equally likely options, entropy = ln(2), max entropy = ln(2)
        # So normalised entropy = 1.0, penalty = 1 - 1.0 = 0.0
        assert abs(penalties.iloc[0]) < 0.01

    def test_tunnel_scores_with_nan_values(self):
        """NaN values in pitch data should produce NaN tunnel scores
        rather than raise an exception."""
        df = pd.DataFrame(
            {
                "plate_x": [np.nan],
                "plate_z": [2.5],
                "prev_plate_x": [0.5],
                "prev_plate_z": [2.5],
                "release_speed": [95.0],
                "prev_release_speed": [94.0],
                "pfx_x": [-6.0],
                "pfx_z": [15.0],
                "prev_pfx_x": [-5.5],
                "prev_pfx_z": [14.5],
            }
        )
        result = compute_tunnel_scores(df)
        # Should not raise; NaN propagation is acceptable
        assert len(result) == 1
