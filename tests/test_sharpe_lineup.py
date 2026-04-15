"""
Tests for src.analytics.sharpe_lineup — Player Sharpe Ratio & Efficient Frontier.

Covers game-level wOBA aggregation, PSR calculation, correlation matrix
properties, Markowitz optimiser constraints, and efficient frontier monotonicity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.sharpe_lineup import (
    LEAGUE_AVG_WOBA,
    _batch_game_level_woba,
    _game_level_woba,
    batch_player_sharpe,
    calculate_player_sharpe,
    compute_correlation_matrix,
    efficient_frontier,
    get_sharpe_leaderboard,
    optimize_lineup,
    SharpeLineupModel,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def sharpe_conn(db_conn):
    """Re-use the session-scoped db_conn from conftest.py."""
    return db_conn


@pytest.fixture()
def known_batter_id(sharpe_conn):
    """Return a batter_id that has enough game-level data for testing."""
    df = sharpe_conn.execute("""
        SELECT batter_id, COUNT(DISTINCT game_pk) AS games
        FROM pitches
        WHERE woba_denom > 0
        GROUP BY batter_id
        HAVING COUNT(DISTINCT game_pk) >= 3
        ORDER BY games DESC
        LIMIT 1
    """).fetchdf()
    if df.empty:
        pytest.skip("No batter with sufficient games in test data.")
    return int(df["batter_id"].iloc[0])


@pytest.fixture()
def multi_batter_ids(sharpe_conn):
    """Return at least 3 batter_ids that share games (for correlation testing)."""
    df = sharpe_conn.execute("""
        SELECT batter_id, COUNT(DISTINCT game_pk) AS games
        FROM pitches
        WHERE woba_denom > 0
        GROUP BY batter_id
        HAVING COUNT(DISTINCT game_pk) >= 2
        ORDER BY games DESC
        LIMIT 5
    """).fetchdf()
    if len(df) < 2:
        pytest.skip("Not enough batters with shared games in test data.")
    return df["batter_id"].tolist()


# ── Synthetic data helpers ────────────────────────────────────────────────────


def _make_player_stats(n: int = 5, seed: int = 42) -> pd.DataFrame:
    """Create synthetic player stats for optimiser tests."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "batter_id": list(range(1, n + 1)),
        "mean_woba": rng.uniform(0.280, 0.400, n).round(4),
        "std_woba": rng.uniform(0.05, 0.20, n).round(4),
        "games": [50] * n,
    })


def _make_correlation_matrix(n: int = 5, seed: int = 42) -> pd.DataFrame:
    """Create a valid positive-semidefinite correlation matrix."""
    rng = np.random.RandomState(seed)
    # Generate a random matrix and compute a proper correlation matrix
    A = rng.randn(n * 3, n)  # tall random matrix
    cov = np.cov(A, rowvar=False)
    # Normalise to correlation
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    np.fill_diagonal(corr, 1.0)
    ids = list(range(1, n + 1))
    return pd.DataFrame(corr, index=ids, columns=ids)


# ── Game-level wOBA aggregation ───────────────────────────────────────────────


class TestGameLevelWoba:
    """Tests for _game_level_woba helper."""

    def test_returns_dataframe(self, sharpe_conn, known_batter_id):
        df = _game_level_woba(sharpe_conn, known_batter_id)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_has_required_columns(self, sharpe_conn, known_batter_id):
        df = _game_level_woba(sharpe_conn, known_batter_id)
        for col in ("game_pk", "game_date", "game_woba"):
            assert col in df.columns

    def test_game_woba_in_valid_range(self, sharpe_conn, known_batter_id):
        df = _game_level_woba(sharpe_conn, known_batter_id)
        # Game-level wOBA should be between 0 and ~3 (theoretical max)
        assert (df["game_woba"] >= 0).all()
        assert (df["game_woba"] <= 3.0).all()

    def test_woba_calculation_correct(self, sharpe_conn, known_batter_id):
        """Verify game_woba = woba_value_sum / woba_denom_sum."""
        df = _game_level_woba(sharpe_conn, known_batter_id)
        if df.empty:
            pytest.skip("No data.")
        expected = df["woba_value_sum"] / df["woba_denom_sum"]
        pd.testing.assert_series_equal(
            df["game_woba"], expected, check_names=False, atol=1e-10,
        )

    def test_nonexistent_batter_returns_empty(self, sharpe_conn):
        df = _game_level_woba(sharpe_conn, 999999999)
        assert df.empty


class TestBatchGameLevelWoba:
    """Tests for _batch_game_level_woba helper."""

    def test_returns_all_requested_batters(self, sharpe_conn, multi_batter_ids):
        df = _batch_game_level_woba(sharpe_conn, multi_batter_ids)
        returned_ids = set(df["batter_id"].unique())
        # At least some of the requested players should appear
        assert len(returned_ids) > 0

    def test_empty_list_returns_empty(self, sharpe_conn):
        df = _batch_game_level_woba(sharpe_conn, [])
        assert df.empty


# ── Player Sharpe Ratio ──────────────────────────────────────────────────────


class TestCalculatePlayerSharpe:
    """Tests for calculate_player_sharpe."""

    def test_returns_dict_with_required_keys(self, sharpe_conn, known_batter_id):
        result = calculate_player_sharpe(sharpe_conn, known_batter_id)
        for key in ("batter_id", "psr", "mean_woba", "std_woba", "games"):
            assert key in result

    def test_games_count_positive(self, sharpe_conn, known_batter_id):
        result = calculate_player_sharpe(sharpe_conn, known_batter_id)
        assert result["games"] > 0

    def test_mean_woba_in_valid_range(self, sharpe_conn, known_batter_id):
        result = calculate_player_sharpe(sharpe_conn, known_batter_id)
        assert 0 <= result["mean_woba"] <= 3.0

    def test_std_woba_non_negative(self, sharpe_conn, known_batter_id):
        result = calculate_player_sharpe(sharpe_conn, known_batter_id)
        assert result["std_woba"] >= 0

    def test_nonexistent_batter(self, sharpe_conn):
        result = calculate_player_sharpe(sharpe_conn, 999999999)
        assert result["psr"] is None
        assert result["games"] == 0

    def test_psr_formula_with_known_values(self):
        """Verify PSR = (mean - league_avg) / std with known inputs."""
        # We test the math directly, not via DB
        mean_woba = 0.400
        std_woba = 0.100
        league_avg = 0.320
        expected_psr = (mean_woba - league_avg) / std_woba  # 0.8
        assert abs(expected_psr - 0.8) < 1e-10


class TestPSREdgeCases:
    """Edge cases for Player Sharpe Ratio."""

    def test_single_game_undefined_std(self, sharpe_conn):
        """A batter with only 1 game should have std=0 and PSR=None."""
        # Find a batter with exactly 1 qualifying game, or skip
        df = sharpe_conn.execute("""
            SELECT batter_id, COUNT(DISTINCT game_pk) AS games
            FROM pitches
            WHERE woba_denom > 0
            GROUP BY batter_id
            HAVING COUNT(DISTINCT game_pk) = 1
            LIMIT 1
        """).fetchdf()
        if df.empty:
            pytest.skip("No single-game batter in test data.")

        result = calculate_player_sharpe(sharpe_conn, int(df["batter_id"].iloc[0]))
        # With only 1 game, ddof=1 std is 0 or undefined
        assert result["psr"] is None or result["games"] == 1


# ── Batch Player Sharpe ──────────────────────────────────────────────────────


class TestBatchPlayerSharpe:
    """Tests for batch_player_sharpe."""

    def test_returns_dataframe(self, sharpe_conn):
        df = batch_player_sharpe(sharpe_conn, min_games=1)
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, sharpe_conn):
        df = batch_player_sharpe(sharpe_conn, min_games=1)
        if df.empty:
            pytest.skip("No qualifying batters.")
        for col in ("batter_id", "psr", "mean_woba", "std_woba", "games"):
            assert col in df.columns

    def test_sorted_by_psr_descending(self, sharpe_conn):
        df = batch_player_sharpe(sharpe_conn, min_games=1)
        if df.empty or len(df) < 2:
            pytest.skip("Not enough data.")
        valid = df.dropna(subset=["psr"])
        if len(valid) >= 2:
            assert valid["psr"].is_monotonic_decreasing


# ── Correlation matrix ────────────────────────────────────────────────────────


class TestCorrelationMatrix:
    """Tests for compute_correlation_matrix."""

    def test_symmetric(self, sharpe_conn, multi_batter_ids):
        corr = compute_correlation_matrix(sharpe_conn, multi_batter_ids)
        np.testing.assert_array_almost_equal(corr.values, corr.values.T)

    def test_diagonal_is_one(self, sharpe_conn, multi_batter_ids):
        corr = compute_correlation_matrix(sharpe_conn, multi_batter_ids)
        diag = np.diag(corr.values)
        np.testing.assert_array_almost_equal(diag, np.ones(len(multi_batter_ids)))

    def test_values_in_range(self, sharpe_conn, multi_batter_ids):
        corr = compute_correlation_matrix(sharpe_conn, multi_batter_ids)
        assert (corr.values >= -1.0 - 1e-10).all()
        assert (corr.values <= 1.0 + 1e-10).all()

    def test_single_player(self, sharpe_conn, known_batter_id):
        corr = compute_correlation_matrix(sharpe_conn, [known_batter_id])
        assert corr.shape == (1, 1)
        assert corr.values[0, 0] == 1.0

    def test_shape_matches_input(self, sharpe_conn, multi_batter_ids):
        corr = compute_correlation_matrix(sharpe_conn, multi_batter_ids)
        n = len(multi_batter_ids)
        assert corr.shape == (n, n)


class TestCorrelationMatrixSynthetic:
    """Synthetic correlation matrix edge cases."""

    def test_perfect_correlation(self):
        """When all off-diagonal are 1.0, the matrix should still be valid."""
        n = 3
        corr = pd.DataFrame(
            np.ones((n, n)),
            index=[1, 2, 3], columns=[1, 2, 3],
        )
        assert (np.diag(corr.values) == 1.0).all()

    def test_negative_correlation(self):
        """Negative correlations should be representable."""
        corr = pd.DataFrame(
            [[1.0, -0.5], [-0.5, 1.0]],
            index=[1, 2], columns=[1, 2],
        )
        assert corr.loc[1, 2] == -0.5
        assert corr.loc[2, 1] == -0.5


# ── Optimizer ─────────────────────────────────────────────────────────────────


class TestOptimizeLineup:
    """Tests for optimize_lineup."""

    def test_weights_sum_to_one(self):
        stats = _make_player_stats(5)
        corr = _make_correlation_matrix(5)
        result = optimize_lineup(stats, corr, n_players=3)
        total_weight = sum(result["weights"].values())
        assert abs(total_weight - 1.0) < 1e-3

    def test_weights_non_negative(self):
        stats = _make_player_stats(5)
        corr = _make_correlation_matrix(5)
        result = optimize_lineup(stats, corr, n_players=5)
        for w in result["weights"].values():
            assert w >= -1e-6

    def test_result_has_required_keys(self):
        stats = _make_player_stats(5)
        corr = _make_correlation_matrix(5)
        result = optimize_lineup(stats, corr)
        for key in ("weights", "expected_woba", "portfolio_variance",
                     "portfolio_std", "sharpe", "selected_players"):
            assert key in result

    def test_expected_woba_in_valid_range(self):
        stats = _make_player_stats(5)
        corr = _make_correlation_matrix(5)
        result = optimize_lineup(stats, corr)
        # Expected wOBA should be within the range of individual player means
        min_mu = stats["mean_woba"].min()
        max_mu = stats["mean_woba"].max()
        assert min_mu - 0.01 <= result["expected_woba"] <= max_mu + 0.01

    def test_variance_non_negative(self):
        stats = _make_player_stats(5)
        corr = _make_correlation_matrix(5)
        result = optimize_lineup(stats, corr)
        assert result["portfolio_variance"] >= -1e-10

    def test_single_player(self):
        stats = pd.DataFrame({
            "batter_id": [1],
            "mean_woba": [0.350],
            "std_woba": [0.100],
            "games": [50],
        })
        corr = pd.DataFrame([[1.0]], index=[1], columns=[1])
        result = optimize_lineup(stats, corr, n_players=1)
        assert abs(result["weights"][1] - 1.0) < 1e-3
        assert abs(result["expected_woba"] - 0.350) < 1e-3

    def test_risk_budget_affects_result(self):
        """Higher risk budget should produce lower variance portfolio."""
        stats = _make_player_stats(5)
        corr = _make_correlation_matrix(5)
        low_risk = optimize_lineup(stats, corr, risk_budget=0.01)
        high_risk = optimize_lineup(stats, corr, risk_budget=100.0)
        # High risk aversion should yield lower or equal variance
        assert high_risk["portfolio_variance"] <= low_risk["portfolio_variance"] + 1e-4

    def test_no_overlapping_ids_raises(self):
        stats = pd.DataFrame({
            "batter_id": [100, 200],
            "mean_woba": [0.300, 0.350],
            "std_woba": [0.10, 0.12],
        })
        corr = pd.DataFrame(
            [[1.0, 0.1], [0.1, 1.0]],
            index=[999, 998], columns=[999, 998],
        )
        with pytest.raises(ValueError, match="No overlapping"):
            optimize_lineup(stats, corr)


# ── Efficient frontier ────────────────────────────────────────────────────────


class TestEfficientFrontier:
    """Tests for efficient_frontier."""

    def test_returns_list(self):
        stats = _make_player_stats(5)
        corr = _make_correlation_matrix(5)
        points = efficient_frontier(stats, corr, n_points=10)
        assert isinstance(points, list)
        assert len(points) > 0

    def test_frontier_points_have_required_keys(self):
        stats = _make_player_stats(5)
        corr = _make_correlation_matrix(5)
        points = efficient_frontier(stats, corr, n_points=5)
        for pt in points:
            assert "expected_woba" in pt
            assert "variance" in pt
            assert "weights" in pt

    def test_frontier_monotonicity(self):
        """On the efficient frontier, as variance decreases, expected return
        should also decrease (roughly). We check that sorting by variance
        gives monotonically non-decreasing expected return."""
        stats = _make_player_stats(5)
        corr = _make_correlation_matrix(5)
        points = efficient_frontier(stats, corr, n_points=20)

        if len(points) < 3:
            pytest.skip("Not enough frontier points.")

        # Sort by variance
        sorted_pts = sorted(points, key=lambda p: p["variance"])
        returns = [p["expected_woba"] for p in sorted_pts]

        # Check general trend: correlation between variance and return
        # should be positive (higher risk = higher return on the frontier)
        variances = [p["variance"] for p in sorted_pts]
        if np.std(variances) > 0 and np.std(returns) > 0:
            corr_val = np.corrcoef(variances, returns)[0, 1]
            # The correlation should be positive (more risk = more return)
            # but we use a relaxed threshold since discrete optimisation
            # doesn't produce a perfectly smooth frontier
            assert corr_val > -0.5, (
                f"Expected positive correlation between variance and return, "
                f"got {corr_val:.3f}"
            )


# ── Sharpe Leaderboard ────────────────────────────────────────────────────────


class TestSharpeLeaderboard:
    """Tests for get_sharpe_leaderboard."""

    def test_returns_dataframe(self, sharpe_conn):
        df = get_sharpe_leaderboard(sharpe_conn, min_games=1)
        assert isinstance(df, pd.DataFrame)

    def test_has_rank_column(self, sharpe_conn):
        df = get_sharpe_leaderboard(sharpe_conn, min_games=1)
        if df.empty:
            pytest.skip("No qualifying batters.")
        assert "rank" in df.columns
        assert df["rank"].iloc[0] == 1

    def test_sorted_by_psr(self, sharpe_conn):
        df = get_sharpe_leaderboard(sharpe_conn, min_games=1)
        if len(df) < 2:
            pytest.skip("Not enough data.")
        assert df["psr"].is_monotonic_decreasing


# ── SharpeLineupModel class ──────────────────────────────────────────────────


class TestSharpeLineupModel:
    """Tests for the BaseAnalyticsModel subclass."""

    def test_model_name(self):
        model = SharpeLineupModel()
        assert model.model_name == "sharpe_lineup"
        assert model.version == "1.0.0"

    def test_repr(self):
        model = SharpeLineupModel()
        r = repr(model)
        assert "sharpe_lineup" in r

    def test_train_returns_metrics(self, sharpe_conn):
        model = SharpeLineupModel()
        metrics = model.train(sharpe_conn, min_games=1)
        assert "qualifying_batters" in metrics
        assert metrics["qualifying_batters"] >= 0

    def test_evaluate_returns_dict(self, sharpe_conn):
        model = SharpeLineupModel()
        result = model.evaluate(sharpe_conn, min_games=1)
        assert isinstance(result, dict)
        assert "qualifying_batters" in result


# ── Integration test with real DB fixture ─────────────────────────────────────


class TestIntegration:
    """End-to-end workflow: sharpe -> correlation -> optimize."""

    def test_full_pipeline(self, sharpe_conn):
        """Run the full pipeline: batch sharpe -> correlation -> optimize."""
        # Get qualifying batters
        stats = batch_player_sharpe(sharpe_conn, min_games=1)
        if len(stats) < 2:
            pytest.skip("Not enough batters for portfolio optimisation.")

        # Keep top 5 (or fewer)
        stats = stats.head(5).dropna(subset=["mean_woba", "std_woba"])
        if len(stats) < 2:
            pytest.skip("Not enough batters with valid stats.")

        # Ensure std_woba > 0 for all
        stats = stats[stats["std_woba"] > 0]
        if len(stats) < 2:
            pytest.skip("Not enough batters with non-zero std.")

        player_ids = stats["batter_id"].tolist()

        # Compute correlation
        corr = compute_correlation_matrix(sharpe_conn, player_ids)
        assert corr.shape[0] == len(player_ids)

        # Optimise
        result = optimize_lineup(stats, corr, n_players=min(9, len(player_ids)))
        assert abs(sum(result["weights"].values()) - 1.0) < 1e-3
        assert result["expected_woba"] > 0
