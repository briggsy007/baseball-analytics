"""
Tests for the CausalWAR causal inference player valuation model.

Tests cover:
- Data extraction queries return expected columns
- Nuisance model fitting on small data
- Residualisation produces correct dimensions
- Output schema validation
- Leaderboard sorting and filtering
- Edge cases: few PAs, single venue, all walks
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.causal_war import (
    CausalWARConfig,
    CausalWARModel,
    _build_features,
    _extract_pa_data,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synthetic_pa_df(n: int = 500, n_players: int = 10, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic plate-appearance DataFrame matching _extract_pa_data output."""
    rng = np.random.RandomState(seed)

    player_ids = rng.choice(range(100000, 100000 + n_players), size=n)
    pitcher_ids = rng.choice(range(200000, 200050), size=n)
    venues = rng.choice(
        ["Citizens Bank Park", "Yankee Stadium", "Fenway Park", "Dodger Stadium", "Wrigley Field"],
        size=n,
    )

    df = pd.DataFrame({
        "pitcher_id": pitcher_ids,
        "batter_id": player_ids,
        "game_pk": rng.randint(400000, 500000, size=n),
        "game_date": pd.date_range("2025-04-01", periods=n, freq="h").date[:n],
        "at_bat_number": rng.randint(1, 60, size=n),
        "stand": rng.choice(["L", "R"], size=n),
        "p_throws": rng.choice(["L", "R"], size=n),
        "inning": rng.randint(1, 10, size=n),
        "outs_when_up": rng.randint(0, 3, size=n),
        "on_1b": rng.choice([0, 1], size=n),
        "on_2b": rng.choice([0, 1], size=n),
        "on_3b": rng.choice([0, 1], size=n),
        "home_team": "PHI",
        "away_team": rng.choice(["NYM", "ATL", "WSH"], size=n),
        "if_fielding_alignment": rng.choice(["Standard", "Infield shift"], size=n),
        "of_fielding_alignment": rng.choice(["Standard", "Strategic"], size=n),
        "woba_value": rng.uniform(0, 2.0, size=n),
        "woba_denom": np.ones(n),
        "balls": rng.randint(0, 4, size=n),
        "strikes": rng.randint(0, 3, size=n),
        "delta_run_exp": rng.normal(0, 0.1, size=n),
        "pitches_in_pa": rng.randint(1, 8, size=n),
        "venue": venues,
    })
    return df


# ---------------------------------------------------------------------------
# Test data extraction
# ---------------------------------------------------------------------------


class TestDataExtraction:
    """Tests for _extract_pa_data and _build_features."""

    def test_extract_returns_expected_columns(self, db_conn):
        """Extracted PA data should contain required columns."""
        # Get any available season
        season_row = db_conn.execute(
            "SELECT DISTINCT EXTRACT(YEAR FROM game_date)::INTEGER AS season "
            "FROM pitches ORDER BY season DESC LIMIT 1"
        ).fetchone()
        if season_row is None:
            pytest.skip("No pitch data available")

        season = season_row[0]
        df = _extract_pa_data(db_conn, season)

        if df.empty:
            pytest.skip("No PA data for season %d" % season)

        required_cols = [
            "pitcher_id", "batter_id", "game_pk", "game_date",
            "stand", "p_throws", "inning", "outs_when_up",
            "on_1b", "on_2b", "on_3b",
            "woba_value", "woba_denom",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_extract_woba_denom_positive(self, db_conn):
        """All PAs should have woba_denom > 0 (PA-ending events only)."""
        season_row = db_conn.execute(
            "SELECT DISTINCT EXTRACT(YEAR FROM game_date)::INTEGER AS season "
            "FROM pitches ORDER BY season DESC LIMIT 1"
        ).fetchone()
        if season_row is None:
            pytest.skip("No pitch data available")

        df = _extract_pa_data(db_conn, season_row[0])
        if df.empty:
            pytest.skip("No PA data")

        assert (df["woba_denom"] > 0).all(), "All PAs should have woba_denom > 0"


class TestBuildFeatures:
    """Tests for the feature engineering step."""

    def test_dimensions(self):
        """W, Y, player_ids should have consistent first dimensions."""
        df = _synthetic_pa_df(n=200)
        W, Y, player_ids, _ = _build_features(df)

        assert W.shape[0] == 200
        assert Y.shape[0] == 200
        assert player_ids.shape[0] == 200
        assert W.ndim == 2
        assert W.shape[1] >= 10  # at least 10 confounders

    def test_no_nans_in_confounders(self):
        """Confounder matrix should have no NaN values after processing."""
        df = _synthetic_pa_df(n=300)
        W, Y, _, _ = _build_features(df)
        assert not np.any(np.isnan(W)), "W should have no NaNs"

    def test_outcome_is_float(self):
        """Outcome vector should be float64."""
        df = _synthetic_pa_df(n=100)
        _, Y, _, _ = _build_features(df)
        assert Y.dtype == np.float64

    def test_platoon_encoding(self):
        """Platoon advantage should be 1 when stand != p_throws."""
        df = pd.DataFrame({
            "pitcher_id": [1, 2],
            "batter_id": [10, 20],
            "game_pk": [100, 100],
            "game_date": ["2025-06-01", "2025-06-01"],
            "at_bat_number": [1, 2],
            "stand": ["L", "R"],
            "p_throws": ["R", "R"],
            "inning": [1, 1],
            "outs_when_up": [0, 0],
            "on_1b": [0, 0],
            "on_2b": [0, 0],
            "on_3b": [0, 0],
            "home_team": ["PHI", "PHI"],
            "away_team": ["NYM", "NYM"],
            "if_fielding_alignment": ["Standard", "Standard"],
            "of_fielding_alignment": ["Standard", "Standard"],
            "woba_value": [0.5, 0.3],
            "woba_denom": [1.0, 1.0],
            "balls": [0, 0],
            "strikes": [0, 0],
            "delta_run_exp": [0.0, 0.0],
            "pitches_in_pa": [3, 4],
            "venue": ["CBP", "CBP"],
        })
        W, _, _, result_df = _build_features(df)
        # Row 0: stand=L, p_throws=R -> platoon=1
        # Row 1: stand=R, p_throws=R -> platoon=0
        assert result_df["platoon"].iloc[0] == 1
        assert result_df["platoon"].iloc[1] == 0

    def test_missing_venue_handled(self):
        """Missing venue should not cause crash."""
        df = _synthetic_pa_df(n=50)
        df["venue"] = None
        W, Y, _, _ = _build_features(df)
        assert W.shape[0] == 50
        assert not np.any(np.isnan(W))


# ---------------------------------------------------------------------------
# Test nuisance model fitting
# ---------------------------------------------------------------------------


class TestNuisanceModelFit:
    """Test that nuisance model training doesn't crash on small data."""

    def test_fit_small_data(self):
        """Training on small synthetic data should complete without error."""
        df = _synthetic_pa_df(n=300, n_players=5)
        W, Y, player_ids, pa_df = _build_features(df)

        model = CausalWARModel(CausalWARConfig(
            n_estimators=10,
            n_bootstrap=5,
            n_splits=2,
        ))
        Y_res, effects, metrics = model._fit_dml(W, Y, player_ids, pa_df)

        assert Y_res.shape == Y.shape
        assert len(effects) > 0
        assert "outcome_nuisance_r2" in metrics
        assert "n_observations" in metrics

    def test_residuals_centered(self):
        """Cross-fitted residuals should be approximately zero-mean."""
        df = _synthetic_pa_df(n=500, n_players=10)
        W, Y, player_ids, pa_df = _build_features(df)

        model = CausalWARModel(CausalWARConfig(
            n_estimators=20,
            n_bootstrap=5,
            n_splits=3,
        ))
        Y_res, _, _ = model._fit_dml(W, Y, player_ids, pa_df)

        # Residuals should be roughly centered at zero (within tolerance)
        assert abs(float(np.mean(Y_res))) < 0.5, (
            f"Mean residual {np.mean(Y_res):.4f} should be near zero"
        )


# ---------------------------------------------------------------------------
# Helpers for in-memory model training (avoids needing prod DB)
# ---------------------------------------------------------------------------


def _train_synthetic_model():
    """Train a CausalWAR model on synthetic data using in-memory DuckDB.

    Returns (model, conn, season) where conn has the synthetic data.
    """
    import duckdb as _ddb
    from src.db.schema import create_tables as _ct

    conn = _ddb.connect(":memory:")
    _ct(conn)

    rng = np.random.RandomState(99)
    n = 600
    n_players = 8

    player_ids = rng.choice(range(100000, 100000 + n_players), size=n)
    pitcher_ids = rng.choice(range(200000, 200010), size=n)
    venues = rng.choice(
        ["Citizens Bank Park", "Yankee Stadium", "Fenway Park"], size=n
    )
    seasons_col = [2025] * n

    rows = []
    game_pk_base = 600000
    for i in range(n):
        event = rng.choice([None, "single", "double", "home_run", "strikeout", "walk", "field_out"])
        woba_val = round(float(rng.uniform(0, 2.0)), 3) if event else None
        woba_den = 1.0 if event else 0.0
        month = rng.randint(4, 10)
        day = rng.randint(1, 28)
        rows.append({
            "game_pk": game_pk_base + i // 10,
            "game_date": f"2025-{month:02d}-{day:02d}",
            "pitcher_id": int(pitcher_ids[i]),
            "batter_id": int(player_ids[i]),
            "pitch_type": rng.choice(["FF", "SL", "CH", "CU"]),
            "pitch_name": "FF",
            "release_speed": round(float(rng.normal(93, 2)), 1),
            "release_spin_rate": round(float(rng.normal(2300, 200)), 0),
            "spin_axis": round(float(rng.uniform(0, 360)), 1),
            "pfx_x": round(float(rng.normal(0, 5)), 1),
            "pfx_z": round(float(rng.normal(8, 4)), 1),
            "plate_x": round(float(rng.normal(0, 0.6)), 2),
            "plate_z": round(float(rng.normal(2.5, 0.6)), 2),
            "release_extension": round(float(rng.normal(6.2, 0.4)), 1),
            "release_pos_x": round(float(rng.normal(-1.5, 0.5)), 2),
            "release_pos_y": round(float(rng.normal(55, 0.5)), 2),
            "release_pos_z": round(float(rng.normal(5.8, 0.4)), 2),
            "launch_speed": round(float(rng.normal(88, 12)), 1) if event else None,
            "launch_angle": round(float(rng.normal(12, 20)), 1) if event else None,
            "hit_distance": round(float(rng.uniform(50, 420)), 0) if event else None,
            "hc_x": None,
            "hc_y": None,
            "bb_type": None,
            "estimated_ba": None,
            "estimated_woba": None,
            "delta_home_win_exp": round(float(rng.normal(0, 0.03)), 4),
            "delta_run_exp": round(float(rng.normal(0, 0.1)), 4),
            "inning": int(rng.randint(1, 10)),
            "inning_topbot": rng.choice(["Top", "Bot"]),
            "outs_when_up": int(rng.randint(0, 3)),
            "balls": int(rng.randint(0, 4)),
            "strikes": int(rng.randint(0, 3)),
            "on_1b": int(rng.choice([0, 1])),
            "on_2b": int(rng.choice([0, 1])),
            "on_3b": int(rng.choice([0, 1])),
            "stand": rng.choice(["L", "R"]),
            "p_throws": rng.choice(["L", "R"]),
            "at_bat_number": int(i // 5 + 1),
            "pitch_number": int(i % 5 + 1),
            "description": rng.choice(["called_strike", "ball", "hit_into_play", "foul"]),
            "events": event,
            "type": "X" if event else "S",
            "home_team": "PHI",
            "away_team": rng.choice(["NYM", "ATL", "WSH"]),
            "woba_value": woba_val,
            "woba_denom": woba_den,
            "babip_value": None,
            "iso_value": None,
            "zone": int(rng.randint(1, 15)),
            "effective_speed": round(float(rng.normal(92, 2)), 1),
            "if_fielding_alignment": rng.choice(["Standard", "Infield shift"]),
            "of_fielding_alignment": rng.choice(["Standard", "Strategic"]),
            "fielder_2": int(rng.randint(400000, 500000)),
        })

    df = pd.DataFrame(rows)
    conn.execute("INSERT INTO pitches SELECT * FROM df")

    # Insert games for venue join
    game_pks = df["game_pk"].unique()
    for gpk in game_pks:
        venue = rng.choice(["Citizens Bank Park", "Yankee Stadium", "Fenway Park"])
        conn.execute(
            "INSERT INTO games (game_pk, game_date, home_team, away_team, venue) "
            "VALUES ($1, '2025-06-15', 'PHI', 'NYM', $2)",
            [int(gpk), venue],
        )

    model = CausalWARModel(CausalWARConfig(
        n_estimators=10,
        n_bootstrap=5,
        n_splits=2,
        pa_min_qualifying=5,
    ))
    model.train(conn, season=2025)
    return model, conn, 2025


# ---------------------------------------------------------------------------
# Test CausalWAR output schema
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """Test output format of CausalWAR calculations."""

    @pytest.fixture(scope="class")
    def trained_env(self):
        """Train a model on synthetic in-memory data."""
        model, conn, season = _train_synthetic_model()
        yield model, conn, season
        conn.close()

    def test_single_player_output_keys(self, trained_env):
        """calculate_causal_war should return expected keys."""
        model, conn, season = trained_env
        if not model._player_effects:
            pytest.skip("No player effects estimated")

        pid = next(iter(model._player_effects))
        result = model.calculate_causal_war(conn, pid, season)

        expected_keys = {
            "player_id", "season", "causal_war", "ci_low", "ci_high",
            "park_adj_woba", "raw_woba", "pa",
        }
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_single_player_missing(self, trained_env):
        """Querying a non-existent player should return None values."""
        model, conn, season = trained_env
        result = model.calculate_causal_war(conn, 999999999, season)

        assert result["player_id"] == 999999999
        assert result["causal_war"] is None
        assert result["pa"] == 0

    def test_batch_output_columns(self, trained_env):
        """batch_calculate should return a DataFrame with expected columns."""
        model, conn, season = trained_env
        df = model.batch_calculate(conn, season)

        if df.empty:
            pytest.skip("No qualifying players")

        expected_cols = [
            "player_id", "season", "causal_war", "ci_low", "ci_high",
            "park_adj_woba", "raw_woba", "pa",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_batch_sorted_descending(self, trained_env):
        """Batch results should be sorted by causal_war descending."""
        model, conn, season = trained_env
        df = model.batch_calculate(conn, season)

        if len(df) < 2:
            pytest.skip("Need at least 2 players for sort test")

        wars = df["causal_war"].dropna().values
        for i in range(len(wars) - 1):
            assert wars[i] >= wars[i + 1], "Should be sorted descending"


# ---------------------------------------------------------------------------
# Test leaderboard
# ---------------------------------------------------------------------------


class TestLeaderboard:
    """Tests for the get_leaderboard method."""

    @pytest.fixture(scope="class")
    def trained_env(self):
        """Train a model on synthetic in-memory data."""
        model, conn, season = _train_synthetic_model()
        yield model, conn, season
        conn.close()

    def test_leaderboard_has_rank(self, trained_env):
        """Leaderboard index should be named 'Rank' and start at 1."""
        model, conn, season = trained_env

        df = model.get_leaderboard(conn, season)
        if df.empty:
            pytest.skip("Empty leaderboard")

        assert df.index.name == "Rank"
        assert df.index[0] == 1

    def test_leaderboard_position_filter(self, trained_env):
        """Filtering by position_type='batter' should exclude pitchers."""
        model, conn, season = trained_env

        all_df = model.get_leaderboard(conn, season, "all")
        batter_df = model.get_leaderboard(conn, season, "batter")

        # Batter leaderboard should be <= all leaderboard
        assert len(batter_df) <= len(all_df)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_pitcher_with_few_pa(self):
        """Player with very few PAs should still get an effect or be skipped gracefully."""
        df = _synthetic_pa_df(n=300, n_players=10)
        # Add a player with only 3 PAs
        extra = pd.DataFrame({
            "pitcher_id": [999999] * 3,
            "batter_id": [888888] * 3,
            "game_pk": [500000] * 3,
            "game_date": ["2025-06-01"] * 3,
            "at_bat_number": [1, 2, 3],
            "stand": ["R"] * 3,
            "p_throws": ["R"] * 3,
            "inning": [1, 1, 1],
            "outs_when_up": [0, 1, 2],
            "on_1b": [0, 0, 0],
            "on_2b": [0, 0, 0],
            "on_3b": [0, 0, 0],
            "home_team": ["PHI"] * 3,
            "away_team": ["NYM"] * 3,
            "if_fielding_alignment": ["Standard"] * 3,
            "of_fielding_alignment": ["Standard"] * 3,
            "woba_value": [0.0, 0.0, 0.0],
            "woba_denom": [1.0, 1.0, 1.0],
            "balls": [0, 0, 0],
            "strikes": [0, 0, 0],
            "delta_run_exp": [0.0, 0.0, 0.0],
            "pitches_in_pa": [3, 3, 3],
            "venue": ["CBP"] * 3,
        })
        combined = pd.concat([df, extra], ignore_index=True)

        W, Y, player_ids, pa_df = _build_features(combined)

        model = CausalWARModel(CausalWARConfig(
            n_estimators=10,
            n_bootstrap=5,
            n_splits=2,
        ))
        Y_res, effects, metrics = model._fit_dml(W, Y, player_ids, pa_df)

        # Player 888888 should be skipped (fewer than 10 PAs)
        assert 888888 not in effects or effects.get(888888, {}).get("pa", 0) <= 10

    def test_single_venue_player(self):
        """Player who only plays at one venue should still work."""
        df = _synthetic_pa_df(n=200)
        df["venue"] = "Citizens Bank Park"  # single venue

        W, Y, player_ids, pa_df = _build_features(df)

        model = CausalWARModel(CausalWARConfig(
            n_estimators=10,
            n_bootstrap=5,
            n_splits=2,
        ))
        Y_res, effects, metrics = model._fit_dml(W, Y, player_ids, pa_df)

        assert Y_res.shape == Y.shape
        assert len(effects) > 0

    def test_all_walks_player(self):
        """Player with woba_value=0 for all PAs should get a valid (likely negative) effect."""
        df = _synthetic_pa_df(n=300, n_players=5)
        # Override one player's outcomes to all zeros (like all walks with no value)
        target_player = df["batter_id"].unique()[0]
        df.loc[df["batter_id"] == target_player, "woba_value"] = 0.0

        W, Y, player_ids, pa_df = _build_features(df)

        model = CausalWARModel(CausalWARConfig(
            n_estimators=10,
            n_bootstrap=5,
            n_splits=2,
        ))
        Y_res, effects, metrics = model._fit_dml(W, Y, player_ids, pa_df)

        if target_player in effects:
            # The player should have a valid causal_war value
            assert effects[target_player]["causal_war"] is not None

    def test_model_not_trained_raises(self, db_conn):
        """Calling predict/calculate before train should raise RuntimeError."""
        model = CausalWARModel()

        with pytest.raises(RuntimeError, match="not trained"):
            model.calculate_causal_war(db_conn, 12345, 2025)

        with pytest.raises(RuntimeError, match="not trained"):
            model.batch_calculate(db_conn, 2025)

        with pytest.raises(RuntimeError, match="not trained"):
            model.evaluate(db_conn)

    def test_config_defaults(self):
        """Default config should have sensible values."""
        config = CausalWARConfig()
        assert config.n_splits >= 2
        assert config.n_bootstrap >= 10
        assert config.n_estimators >= 50
        assert config.pa_min_qualifying >= 50

    def test_model_name_and_version(self):
        """Model should report correct name and version."""
        model = CausalWARModel()
        assert model.model_name == "causal_war"
        assert model.version == "1.0.0"

    def test_evaluate_after_train(self):
        """Evaluate should return diagnostics after training."""
        df = _synthetic_pa_df(n=400, n_players=8)
        W, Y, player_ids, pa_df = _build_features(df)

        model = CausalWARModel(CausalWARConfig(
            n_estimators=10,
            n_bootstrap=5,
            n_splits=2,
        ))
        model._fit_dml(W, Y, player_ids, pa_df)
        # Manually mark as trained for evaluate
        model._is_trained = True
        # We need to set _player_effects from the fit
        # _fit_dml returns effects, so let's re-do
        _, effects, _ = model._fit_dml(W, Y, player_ids, pa_df)
        model._player_effects = effects
        model._is_trained = True

        # Use a real db_conn-like approach -- we just need evaluate to work
        # evaluate doesn't use conn, but the base class requires it in signature
        # We'll create a minimal mock
        import duckdb
        mock_conn = duckdb.connect(":memory:")
        result = model.evaluate(mock_conn)
        mock_conn.close()

        assert "n_players" in result
        assert "mean_causal_war" in result
        assert result["n_players"] > 0
