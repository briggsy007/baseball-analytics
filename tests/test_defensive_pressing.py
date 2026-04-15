"""
Tests for the Defensive Pressing Intensity (DPI) model.

Tests cover:
- Spray angle computation from hc_x, hc_y
- Expected out model: higher EV + higher LA -> lower out probability (fly balls)
- DPI computation: perfect defense (all outs) -> high DPI, all hits -> low DPI
- Team DPI aggregation
- Edge cases: game with 0 BIP, missing launch data, bunts
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import duckdb

from src.db.schema import create_tables
from src.analytics.defensive_pressing import (
    DPIConfig,
    DefensivePressingModel,
    compute_spray_angle,
    build_bip_features,
    train_expected_out_model,
    compute_expected_outs,
    calculate_game_dpi,
    calculate_team_dpi,
    batch_calculate,
    get_player_dpi,
    get_team_game_dpi_timeline,
    OUT_EVENTS,
    HIT_EVENTS,
    _is_out,
    _is_hit,
    _is_xbh,
)


# ---------------------------------------------------------------------------
# Helpers -- synthetic data with controlled defensive scenarios
# ---------------------------------------------------------------------------


def _build_dpi_test_db(
    n_games: int = 10,
    bip_per_game: int = 30,
    out_rate: float = 0.65,
    seed: int = 42,
):
    """Build an in-memory DuckDB with synthetic BIP data for DPI testing.

    Creates data where:
    - Each game has ~bip_per_game balls in play
    - Ground balls have higher out rate than fly balls
    - Popups are almost always outs
    - Higher launch speeds on fly balls = harder to catch

    Returns (conn, season, team_ids).
    """
    rng = np.random.RandomState(seed)
    conn = duckdb.connect(":memory:")
    create_tables(conn)

    season = 2025
    team_ids = ["PHI", "NYM", "ATL", "WSH", "MIA"]
    pitcher_ids = list(range(200000, 200010))
    batter_ids = list(range(400000, 400020))

    rows = []
    game_pk_base = 800000

    for game_idx in range(n_games):
        home_team = team_ids[game_idx % len(team_ids)]
        away_team = team_ids[(game_idx + 1) % len(team_ids)]
        game_pk = game_pk_base + game_idx

        for bip_idx in range(bip_per_game):
            pitcher_id = int(rng.choice(pitcher_ids))
            batter_id = int(rng.choice(batter_ids))

            # Generate batted ball profile
            bb_type = rng.choice(
                ["ground_ball", "line_drive", "fly_ball", "popup"],
                p=[0.44, 0.20, 0.30, 0.06],
            )

            if bb_type == "ground_ball":
                ls = rng.normal(85, 8)
                la = rng.normal(-10, 8)
                is_out_val = rng.random() < 0.75
            elif bb_type == "line_drive":
                ls = rng.normal(95, 8)
                la = rng.normal(15, 5)
                is_out_val = rng.random() < 0.25
            elif bb_type == "fly_ball":
                ls = rng.normal(92, 10)
                la = rng.normal(30, 10)
                # Higher EV + higher LA = harder to catch (for fly balls)
                catch_prob = 0.7 - (ls - 90) * 0.01 - max(0, la - 35) * 0.005
                is_out_val = rng.random() < max(0.1, min(0.95, catch_prob))
            else:  # popup
                ls = rng.normal(60, 10)
                la = rng.normal(60, 10)
                is_out_val = rng.random() < 0.97

            ls = float(np.clip(ls, 20, 120))
            la = float(np.clip(la, -80, 85))

            # Spray angle encoded as hc_x, hc_y
            spray_deg = rng.normal(0, 30)
            hc_x = 125.42 + 80 * math.sin(math.radians(spray_deg))
            hc_y = 198.27 - 80 * math.cos(math.radians(spray_deg))

            if is_out_val:
                if bb_type == "ground_ball" and rng.random() < 0.1:
                    event = "grounded_into_double_play"
                elif bb_type == "fly_ball" and rng.random() < 0.05:
                    event = "sac_fly"
                else:
                    event = "field_out"
            else:
                # Hit
                if rng.random() < 0.6:
                    event = "single"
                elif rng.random() < 0.6:
                    event = "double"
                elif rng.random() < 0.5:
                    event = "triple"
                else:
                    event = "home_run"

            # Alternate top/bot so both teams bat
            is_top = bip_idx % 2 == 0
            month = rng.randint(4, 10)
            day = rng.randint(1, 28)

            woba_val = round(float(rng.uniform(0, 2.0)), 3)
            woba_den = 1.0

            rows.append({
                "game_pk": game_pk,
                "game_date": f"{season}-{month:02d}-{day:02d}",
                "pitcher_id": pitcher_id,
                "batter_id": batter_id,
                "pitch_type": rng.choice(["FF", "SI", "SL", "CU", "CH"]),
                "pitch_name": "Fastball",
                "release_speed": round(float(rng.normal(93, 2)), 1),
                "release_spin_rate": round(float(rng.normal(2300, 200)), 0),
                "spin_axis": round(float(rng.uniform(0, 360)), 1),
                "pfx_x": round(float(rng.normal(0, 5)), 1),
                "pfx_z": round(float(rng.normal(8, 4)), 1),
                "plate_x": round(float(rng.normal(0, 0.6)), 2),
                "plate_z": round(float(np.clip(rng.normal(2.5, 0.6), 0.5, 5.0)), 2),
                "release_extension": round(float(rng.normal(6.2, 0.4)), 1),
                "release_pos_x": round(float(rng.normal(-1.5, 0.5)), 2),
                "release_pos_y": round(float(rng.normal(55, 0.5)), 2),
                "release_pos_z": round(float(rng.normal(5.8, 0.4)), 2),
                "launch_speed": round(ls, 1),
                "launch_angle": round(la, 1),
                "hit_distance": round(float(rng.uniform(50, 400)), 0),
                "hc_x": round(hc_x, 1),
                "hc_y": round(hc_y, 1),
                "bb_type": bb_type,
                "estimated_ba": round(float(rng.uniform(0, 1)), 3),
                "estimated_woba": round(float(rng.uniform(0, 2)), 3),
                "delta_home_win_exp": round(float(rng.normal(0, 0.03)), 4),
                "delta_run_exp": round(float(rng.normal(0, 0.1)), 4),
                "inning": int(rng.randint(1, 10)),
                "inning_topbot": "Top" if is_top else "Bot",
                "outs_when_up": int(rng.randint(0, 3)),
                "balls": int(rng.randint(0, 4)),
                "strikes": int(rng.randint(0, 3)),
                "on_1b": 0,
                "on_2b": 0,
                "on_3b": 0,
                "stand": rng.choice(["L", "R"]),
                "p_throws": rng.choice(["L", "R"]),
                "at_bat_number": bip_idx + 1,
                "pitch_number": 1,
                "description": "hit_into_play",
                "events": event,
                "type": "X",
                "home_team": home_team,
                "away_team": away_team,
                "woba_value": woba_val,
                "woba_denom": woba_den,
                "babip_value": None,
                "iso_value": None,
                "zone": int(rng.randint(1, 15)),
                "effective_speed": round(float(rng.normal(92, 2)), 1),
                "if_fielding_alignment": "Standard",
                "of_fielding_alignment": "Standard",
                "fielder_2": int(rng.randint(400000, 500000)),
            })

    df = pd.DataFrame(rows)
    conn.execute("INSERT INTO pitches SELECT * FROM df")
    return conn, season, team_ids


# ---------------------------------------------------------------------------
# Test spray angle computation
# ---------------------------------------------------------------------------


class TestSprayAngle:
    """Tests for compute_spray_angle."""

    def test_center_field_is_zero(self):
        """A ball hit straight to center field should have spray angle near 0."""
        hc_x = pd.Series([125.42])
        hc_y = pd.Series([100.0])  # far from home plate, straight up
        angle = compute_spray_angle(hc_x, hc_y)
        assert abs(float(angle.iloc[0])) < 2.0, f"Expected ~0, got {angle.iloc[0]}"

    def test_pull_side_positive(self):
        """A ball pulled to the right side should have positive spray angle."""
        hc_x = pd.Series([200.0])  # well to the right
        hc_y = pd.Series([130.0])
        angle = compute_spray_angle(hc_x, hc_y)
        assert float(angle.iloc[0]) > 10.0

    def test_opposite_field_negative(self):
        """A ball hit to the left (opposite for RHB) should have negative spray angle."""
        hc_x = pd.Series([50.0])  # well to the left
        hc_y = pd.Series([130.0])
        angle = compute_spray_angle(hc_x, hc_y)
        assert float(angle.iloc[0]) < -10.0

    def test_nan_input_returns_nan(self):
        """NaN coordinates should produce NaN angle."""
        hc_x = pd.Series([np.nan])
        hc_y = pd.Series([100.0])
        angle = compute_spray_angle(hc_x, hc_y)
        assert pd.isna(angle.iloc[0])

    def test_multiple_values(self):
        """Should handle multiple values in a vectorized fashion."""
        hc_x = pd.Series([125.42, 200.0, 50.0])
        hc_y = pd.Series([100.0, 130.0, 130.0])
        angles = compute_spray_angle(hc_x, hc_y)
        assert len(angles) == 3


# ---------------------------------------------------------------------------
# Test expected out model
# ---------------------------------------------------------------------------


class TestExpectedOutModel:
    """Tests for the xOut model training and prediction."""

    @pytest.fixture(scope="class")
    def trained_env(self):
        conn, season, team_ids = _build_dpi_test_db(n_games=20, bip_per_game=40)
        metrics = train_expected_out_model(conn, config=DPIConfig())
        yield conn, season, team_ids, metrics
        conn.close()

    def test_training_succeeds(self, trained_env):
        """Model should train successfully with enough data."""
        _, _, _, metrics = trained_env
        assert metrics["status"] == "trained"
        assert metrics["n_samples"] > 0

    def test_auc_above_chance(self, trained_env):
        """AUC should be above 0.5 (better than random)."""
        _, _, _, metrics = trained_env
        assert metrics["auc"] > 0.5, f"AUC {metrics['auc']} not above chance"

    def test_popup_high_out_probability(self, trained_env):
        """Popups (low EV, high LA) should have high expected out probability."""
        bip = pd.DataFrame({
            "launch_speed": [55.0, 50.0, 60.0],
            "launch_angle": [65.0, 70.0, 60.0],
            "hc_x": [125.0, 130.0, 120.0],
            "hc_y": [160.0, 155.0, 165.0],
            "bb_type": ["popup", "popup", "popup"],
        })
        probs = compute_expected_outs(bip)
        # Popups should have high out probability
        assert probs.mean() > 0.5, f"Popup xOut {probs.mean():.3f} too low"

    def test_hard_hit_line_drive_lower_out_prob(self, trained_env):
        """Hard-hit line drives should have lower out probability than popups."""
        popups = pd.DataFrame({
            "launch_speed": [55.0] * 5,
            "launch_angle": [65.0] * 5,
            "hc_x": [125.0] * 5,
            "hc_y": [160.0] * 5,
            "bb_type": ["popup"] * 5,
        })
        line_drives = pd.DataFrame({
            "launch_speed": [100.0] * 5,
            "launch_angle": [15.0] * 5,
            "hc_x": [125.0] * 5,
            "hc_y": [130.0] * 5,
            "bb_type": ["line_drive"] * 5,
        })

        popup_probs = compute_expected_outs(popups)
        ld_probs = compute_expected_outs(line_drives)

        assert popup_probs.mean() > ld_probs.mean(), (
            f"Popup xOut ({popup_probs.mean():.3f}) should be higher than "
            f"line drive xOut ({ld_probs.mean():.3f})"
        )

    def test_predictions_in_0_1_range(self, trained_env):
        """All predictions should be probabilities in [0, 1]."""
        bip = pd.DataFrame({
            "launch_speed": [85.0, 95.0, 100.0, 55.0],
            "launch_angle": [10.0, 25.0, -5.0, 70.0],
            "hc_x": [125.0, 180.0, 70.0, 125.0],
            "hc_y": [150.0, 130.0, 170.0, 165.0],
            "bb_type": ["ground_ball", "fly_ball", "ground_ball", "popup"],
        })
        probs = compute_expected_outs(bip)
        valid = probs.dropna()
        assert (valid >= 0).all() and (valid <= 1).all(), "Probabilities out of range"


# ---------------------------------------------------------------------------
# Test DPI computation
# ---------------------------------------------------------------------------


class TestGameDPI:
    """Tests for calculate_game_dpi."""

    @pytest.fixture(scope="class")
    def env(self):
        conn, season, team_ids = _build_dpi_test_db(n_games=10, bip_per_game=30)
        train_expected_out_model(conn, config=DPIConfig())
        yield conn, season, team_ids
        conn.close()

    def test_dpi_returns_expected_keys(self, env):
        """Game DPI result should contain all expected keys."""
        conn, season, team_ids = env
        game_pk = conn.execute("SELECT DISTINCT game_pk FROM pitches LIMIT 1").fetchone()[0]
        result = calculate_game_dpi(conn, game_pk, team_ids[0])

        expected_keys = {
            "game_pk", "team_id", "dpi", "n_bip", "actual_outs",
            "expected_outs", "xbh_count", "hit_count", "extra_base_prevention",
        }
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_dpi_numeric(self, env):
        """DPI should be a numeric value for games with enough BIP."""
        conn, season, team_ids = env
        game_pk = conn.execute("SELECT DISTINCT game_pk FROM pitches LIMIT 1").fetchone()[0]
        result = calculate_game_dpi(conn, game_pk, team_ids[0])
        if result["n_bip"] >= MIN_BIP_PER_GAME:
            assert result["dpi"] is not None
            assert isinstance(result["dpi"], float)

    def test_nonexistent_game_returns_none_dpi(self, env):
        """A game with no BIP should return None DPI."""
        conn, _, team_ids = env
        result = calculate_game_dpi(conn, 999999, team_ids[0])
        assert result["dpi"] is None
        assert result["n_bip"] == 0


MIN_BIP_PER_GAME = 5  # match the module default


# ---------------------------------------------------------------------------
# Test perfect defense vs all-hits scenarios
# ---------------------------------------------------------------------------


class TestDPIScenarios:
    """Test DPI direction: perfect defense should score higher than all-hits."""

    def _make_scenario_db(self, all_outs: bool, seed: int = 99):
        """Build a DB where outcomes are controlled."""
        rng = np.random.RandomState(seed)
        conn = duckdb.connect(":memory:")
        create_tables(conn)

        n = 60
        rows = []
        for i in range(n):
            bb_type = rng.choice(["ground_ball", "fly_ball", "line_drive", "popup"])
            if all_outs:
                event = "field_out"
            else:
                event = "single"

            spray_deg = rng.normal(0, 20)
            hc_x = 125.42 + 60 * math.sin(math.radians(spray_deg))
            hc_y = 198.27 - 60 * math.cos(math.radians(spray_deg))

            rows.append({
                "game_pk": 900000,
                "game_date": "2025-06-15",
                "pitcher_id": 200000,
                "batter_id": 400000 + i,
                "pitch_type": "FF",
                "pitch_name": "Fastball",
                "release_speed": round(float(rng.normal(93, 2)), 1),
                "release_spin_rate": round(float(rng.normal(2300, 200)), 0),
                "spin_axis": round(float(rng.uniform(0, 360)), 1),
                "pfx_x": round(float(rng.normal(0, 5)), 1),
                "pfx_z": round(float(rng.normal(8, 4)), 1),
                "plate_x": round(float(rng.normal(0, 0.6)), 2),
                "plate_z": round(float(np.clip(rng.normal(2.5, 0.6), 0.5, 5.0)), 2),
                "release_extension": round(float(rng.normal(6.2, 0.4)), 1),
                "release_pos_x": round(float(rng.normal(-1.5, 0.5)), 2),
                "release_pos_y": round(float(rng.normal(55, 0.5)), 2),
                "release_pos_z": round(float(rng.normal(5.8, 0.4)), 2),
                "launch_speed": round(float(rng.normal(88, 10)), 1),
                "launch_angle": round(float(rng.normal(15, 15)), 1),
                "hit_distance": round(float(rng.uniform(50, 400)), 0),
                "hc_x": round(hc_x, 1),
                "hc_y": round(hc_y, 1),
                "bb_type": bb_type,
                "estimated_ba": 0.5,
                "estimated_woba": 0.5,
                "delta_home_win_exp": 0.0,
                "delta_run_exp": 0.0,
                "inning": (i // 6) + 1,
                "inning_topbot": "Top",
                "outs_when_up": i % 3,
                "balls": 0,
                "strikes": 0,
                "on_1b": 0, "on_2b": 0, "on_3b": 0,
                "stand": "R",
                "p_throws": "R",
                "at_bat_number": i + 1,
                "pitch_number": 1,
                "description": "hit_into_play",
                "events": event,
                "type": "X",
                "home_team": "PHI",
                "away_team": "NYM",
                "woba_value": 0.5,
                "woba_denom": 1.0,
                "babip_value": None,
                "iso_value": None,
                "zone": 5,
                "effective_speed": 92.0,
                "if_fielding_alignment": "Standard",
                "of_fielding_alignment": "Standard",
                "fielder_2": int(rng.randint(400000, 500000)),
            })

        df = pd.DataFrame(rows)
        conn.execute("INSERT INTO pitches SELECT * FROM df")
        return conn

    def test_all_outs_higher_dpi_than_all_hits(self):
        """Perfect defense (all outs) should have higher DPI than all hits."""
        # First, train xOut model on the training DB
        train_conn, _, _ = _build_dpi_test_db(n_games=20, bip_per_game=40)
        train_expected_out_model(train_conn, config=DPIConfig())

        outs_conn = self._make_scenario_db(all_outs=True)
        hits_conn = self._make_scenario_db(all_outs=False, seed=100)

        outs_dpi = calculate_game_dpi(outs_conn, 900000, "PHI")
        hits_dpi = calculate_game_dpi(hits_conn, 900000, "PHI")

        assert outs_dpi["dpi"] is not None
        assert hits_dpi["dpi"] is not None
        assert outs_dpi["dpi"] > hits_dpi["dpi"], (
            f"All-outs DPI ({outs_dpi['dpi']}) should exceed all-hits DPI ({hits_dpi['dpi']})"
        )

        train_conn.close()
        outs_conn.close()
        hits_conn.close()

    def test_all_outs_positive_dpi(self):
        """Perfect defense should have positive DPI (more outs than expected)."""
        train_conn, _, _ = _build_dpi_test_db(n_games=20, bip_per_game=40)
        train_expected_out_model(train_conn, config=DPIConfig())

        outs_conn = self._make_scenario_db(all_outs=True)
        result = calculate_game_dpi(outs_conn, 900000, "PHI")
        assert result["dpi"] is not None
        assert result["dpi"] > 0, f"All-outs DPI should be positive, got {result['dpi']}"

        train_conn.close()
        outs_conn.close()

    def test_all_hits_negative_dpi(self):
        """All-hits defense should have negative DPI (fewer outs than expected)."""
        train_conn, _, _ = _build_dpi_test_db(n_games=20, bip_per_game=40)
        train_expected_out_model(train_conn, config=DPIConfig())

        hits_conn = self._make_scenario_db(all_outs=False)
        result = calculate_game_dpi(hits_conn, 900000, "PHI")
        assert result["dpi"] is not None
        assert result["dpi"] < 0, f"All-hits DPI should be negative, got {result['dpi']}"

        train_conn.close()
        hits_conn.close()


# ---------------------------------------------------------------------------
# Test team DPI aggregation
# ---------------------------------------------------------------------------


class TestTeamDPI:
    """Tests for calculate_team_dpi and batch_calculate."""

    @pytest.fixture(scope="class")
    def env(self):
        conn, season, team_ids = _build_dpi_test_db(n_games=15, bip_per_game=30)
        train_expected_out_model(conn, config=DPIConfig())
        yield conn, season, team_ids
        conn.close()

    def test_team_dpi_has_games(self, env):
        """Team DPI should report games played."""
        conn, season, team_ids = env
        result = calculate_team_dpi(conn, team_ids[0], season)
        assert result["n_games"] > 0

    def test_team_dpi_consistency_bounded(self, env):
        """Consistency should be in (0, 1]."""
        conn, season, team_ids = env
        result = calculate_team_dpi(conn, team_ids[0], season)
        if result["consistency"] is not None:
            assert 0 < result["consistency"] <= 1.0

    def test_team_dpi_returns_game_dpis(self, env):
        """Team profile should include per-game DPI list."""
        conn, season, team_ids = env
        result = calculate_team_dpi(conn, team_ids[0], season)
        assert isinstance(result["game_dpis"], list)
        if result["n_games"] > 0:
            assert len(result["game_dpis"]) == result["n_games"]

    def test_nonexistent_team_returns_zero_games(self, env):
        """A team not in the data should return n_games=0."""
        conn, season, _ = env
        result = calculate_team_dpi(conn, "ZZZ", season)
        assert result["n_games"] == 0
        assert result["dpi_mean"] is None

    def test_batch_returns_dataframe(self, env):
        """Batch calculation should return a sorted DataFrame."""
        conn, season, _ = env
        df = batch_calculate(conn, season)
        assert isinstance(df, pd.DataFrame)
        if not df.empty:
            assert "team_id" in df.columns
            assert "rank" in df.columns
            assert "dpi_mean" in df.columns
            assert "percentile" in df.columns

    def test_batch_sorted_descending(self, env):
        """Batch results should be sorted by dpi_mean descending."""
        conn, season, _ = env
        df = batch_calculate(conn, season)
        if len(df) >= 2:
            dpis = df["dpi_mean"].values
            for i in range(len(dpis) - 1):
                assert dpis[i] >= dpis[i + 1], "Should be sorted descending"

    def test_batch_percentile_range(self, env):
        """Percentile should be in [0, 100]."""
        conn, season, _ = env
        df = batch_calculate(conn, season)
        if not df.empty:
            assert df["percentile"].min() >= 0
            assert df["percentile"].max() <= 100


# ---------------------------------------------------------------------------
# Test player DPI proxy
# ---------------------------------------------------------------------------


class TestPlayerDPI:
    """Tests for get_player_dpi."""

    @pytest.fixture(scope="class")
    def env(self):
        conn, season, team_ids = _build_dpi_test_db(n_games=15, bip_per_game=30)
        train_expected_out_model(conn, config=DPIConfig())
        yield conn, season
        conn.close()

    def test_nonexistent_player_returns_none(self, env):
        """A player not in the data should return None DPI."""
        conn, season = env
        result = get_player_dpi(conn, 999999, season)
        assert result["dpi_mean"] is None

    def test_valid_pitcher_returns_result(self, env):
        """A pitcher with enough BIP should return a numeric DPI."""
        conn, season = env
        # Find a pitcher with at least some BIP
        row = conn.execute("""
            SELECT pitcher_id, COUNT(*) AS cnt
            FROM pitches
            WHERE type = 'X' AND EXTRACT(YEAR FROM game_date) = $1
            GROUP BY pitcher_id
            ORDER BY cnt DESC
            LIMIT 1
        """, [season]).fetchone()
        if row is None:
            pytest.skip("No pitchers with BIP")
        pitcher_id, cnt = row
        config = DPIConfig(min_bip_per_season=5)
        result = get_player_dpi(conn, pitcher_id, season, config)
        assert result["n_bip"] > 0


# ---------------------------------------------------------------------------
# Test timeline
# ---------------------------------------------------------------------------


class TestTimeline:
    """Tests for get_team_game_dpi_timeline."""

    @pytest.fixture(scope="class")
    def env(self):
        conn, season, team_ids = _build_dpi_test_db(n_games=10, bip_per_game=30)
        train_expected_out_model(conn, config=DPIConfig())
        yield conn, season, team_ids
        conn.close()

    def test_timeline_returns_dataframe(self, env):
        """Timeline should return a DataFrame with expected columns."""
        conn, season, team_ids = env
        df = get_team_game_dpi_timeline(conn, team_ids[0], season)
        if not df.empty:
            assert "game_date" in df.columns
            assert "dpi" in df.columns
            assert "n_bip" in df.columns


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_game_with_zero_bip(self):
        """A game with 0 BIP should return None DPI."""
        conn = duckdb.connect(":memory:")
        create_tables(conn)
        result = calculate_game_dpi(conn, 999999, "PHI")
        assert result["dpi"] is None
        assert result["n_bip"] == 0
        conn.close()

    def test_missing_launch_data(self):
        """BIP rows with NULL launch_speed should still compute (use available rows)."""
        conn = duckdb.connect(":memory:")
        create_tables(conn)

        rng = np.random.RandomState(42)
        rows = []
        for i in range(20):
            has_data = i >= 10  # first 10 have NULL launch data
            rows.append({
                "game_pk": 950000,
                "game_date": "2025-06-15",
                "pitcher_id": 200000,
                "batter_id": 400000 + i,
                "pitch_type": "FF",
                "pitch_name": "Fastball",
                "release_speed": 93.0,
                "release_spin_rate": 2300.0,
                "spin_axis": 180.0,
                "pfx_x": 0.0, "pfx_z": 8.0,
                "plate_x": 0.0, "plate_z": 2.5,
                "release_extension": 6.2,
                "release_pos_x": -1.5,
                "release_pos_y": 55.0,
                "release_pos_z": 5.8,
                "launch_speed": 88.0 if has_data else None,
                "launch_angle": 15.0 if has_data else None,
                "hit_distance": 200.0 if has_data else None,
                "hc_x": 125.0 if has_data else None,
                "hc_y": 150.0 if has_data else None,
                "bb_type": "ground_ball" if has_data else None,
                "estimated_ba": 0.5,
                "estimated_woba": 0.5,
                "delta_home_win_exp": 0.0,
                "delta_run_exp": 0.0,
                "inning": 1,
                "inning_topbot": "Top",
                "outs_when_up": 0,
                "balls": 0, "strikes": 0,
                "on_1b": 0, "on_2b": 0, "on_3b": 0,
                "stand": "R", "p_throws": "R",
                "at_bat_number": i + 1,
                "pitch_number": 1,
                "description": "hit_into_play",
                "events": "field_out",
                "type": "X",
                "home_team": "PHI",
                "away_team": "NYM",
                "woba_value": 0.0, "woba_denom": 1.0,
                "babip_value": None, "iso_value": None,
                "zone": 5,
                "effective_speed": 92.0,
                "if_fielding_alignment": "Standard",
                "of_fielding_alignment": "Standard",
                "fielder_2": int(rng.randint(400000, 500000)),
            })

        df = pd.DataFrame(rows)
        conn.execute("INSERT INTO pitches SELECT * FROM df")

        # Train model first
        train_conn, _, _ = _build_dpi_test_db(n_games=20, bip_per_game=40)
        train_expected_out_model(train_conn, config=DPIConfig())

        result = calculate_game_dpi(conn, 950000, "PHI")
        # Should still compute with partial data
        assert result["n_bip"] == 20
        assert result["actual_outs"] == 20

        train_conn.close()
        conn.close()

    def test_bunts_counted_as_outs(self):
        """Sac bunts should be counted in OUT_EVENTS."""
        assert "sac_bunt" in OUT_EVENTS

    def test_is_out_classification(self):
        """_is_out should correctly identify out events."""
        events = pd.Series(["field_out", "single", "grounded_into_double_play", "home_run"])
        result = _is_out(events)
        assert result.tolist() == [1, 0, 1, 0]

    def test_is_hit_classification(self):
        """_is_hit should correctly identify hit events."""
        events = pd.Series(["single", "double", "triple", "home_run", "field_out"])
        result = _is_hit(events)
        assert result.tolist() == [1, 1, 1, 1, 0]

    def test_is_xbh_classification(self):
        """_is_xbh should correctly identify extra-base hit events."""
        events = pd.Series(["single", "double", "triple", "home_run", "field_out"])
        result = _is_xbh(events)
        assert result.tolist() == [0, 1, 1, 1, 0]

    def test_model_name_and_version(self):
        """Model should report correct name and version."""
        model = DefensivePressingModel()
        assert model.model_name == "defensive_pressing"
        assert model.version == "1.0.0"

    def test_evaluate_not_trained_raises(self):
        """Calling evaluate before training should raise RuntimeError."""
        model = DefensivePressingModel()
        conn = duckdb.connect(":memory:")
        create_tables(conn)
        with pytest.raises(RuntimeError, match="not trained"):
            model.evaluate(conn)
        conn.close()

    def test_config_defaults(self):
        """Default config should have sensible values."""
        config = DPIConfig()
        assert config.min_bip_per_game >= 1
        assert config.min_bip_per_season >= 10
        assert config.xout_n_estimators > 0
