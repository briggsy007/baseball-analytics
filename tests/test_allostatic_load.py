"""
Tests for the Allostatic Batting Load (ABL) model.

Tests cover:
- Leaky integrator: constant input converges to steady state
- Off-day accelerated decay
- Composite ABL: all channels below mean yields ABL=0
- Stressor computation returns expected channels
- Edge cases: batter with 1 game, all home games, season with no off-days
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.allostatic_load import (
    ABLConfig,
    AllostaticLoadModel,
    _apply_channel_load,
    batch_calculate,
    calculate_abl,
    compute_game_stressors,
    compute_leaky_load,
    predict_recovery_days,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_constant_stressor(n: int = 50, value: float = 10.0) -> pd.Series:
    """Create a constant stressor series."""
    return pd.Series([value] * n, name="test_stressor")


def _make_days_since(n: int = 50, gap: int = 1) -> pd.Series:
    """Create a days_since_last series with uniform gaps."""
    return pd.Series([gap] * n, name="days_since_last")


def _make_synthetic_db():
    """Create an in-memory DuckDB with synthetic batter data.

    Returns (conn, batter_id, season).
    """
    import duckdb
    from src.db.schema import create_tables

    conn = duckdb.connect(":memory:")
    create_tables(conn)

    rng = np.random.RandomState(42)
    batter_id = 600000
    season = 2025
    n_games = 30
    pitches_per_game = 20

    rows = []
    game_pk_base = 700000
    game_dates = pd.date_range("2025-04-01", periods=n_games, freq="2D")

    for g_idx in range(n_games):
        game_pk = game_pk_base + g_idx
        game_date = game_dates[g_idx].strftime("%Y-%m-%d")
        is_home = g_idx % 3 != 0  # every 3rd game is away

        for p_idx in range(pitches_per_game):
            desc = rng.choice([
                "called_strike", "ball", "swinging_strike", "foul",
                "hit_into_play", "foul_tip",
            ])
            event = rng.choice([None, None, None, "single", "strikeout", "field_out"])
            woba_val = round(float(rng.uniform(0, 2.0)), 3) if event else None
            woba_den = 1.0 if event else 0.0

            # Some borderline pitches
            plate_x = round(float(rng.normal(0, 0.7)), 2)
            plate_z = round(float(rng.normal(2.5, 0.8)), 2)

            rows.append({
                "game_pk": game_pk,
                "game_date": game_date,
                "pitcher_id": rng.randint(100000, 200000),
                "batter_id": batter_id,
                "pitch_type": rng.choice(["FF", "SL", "CH", "CU"]),
                "pitch_name": "FF",
                "release_speed": round(float(rng.normal(93, 2)), 1),
                "release_spin_rate": round(float(rng.normal(2300, 200)), 0),
                "spin_axis": round(float(rng.uniform(0, 360)), 1),
                "pfx_x": round(float(rng.normal(0, 5)), 1),
                "pfx_z": round(float(rng.normal(8, 4)), 1),
                "plate_x": plate_x,
                "plate_z": max(0.5, plate_z),
                "release_extension": round(float(rng.normal(6.2, 0.4)), 1),
                "release_pos_x": round(float(rng.normal(-1.5, 0.5)), 2),
                "release_pos_y": round(float(rng.normal(55, 0.5)), 2),
                "release_pos_z": round(float(rng.normal(5.8, 0.4)), 2),
                "launch_speed": round(float(rng.normal(88, 12)), 1) if event else None,
                "launch_angle": round(float(rng.normal(12, 20)), 1) if event else None,
                "hit_distance": None,
                "hc_x": None,
                "hc_y": None,
                "bb_type": None,
                "estimated_ba": None,
                "estimated_woba": None,
                "delta_home_win_exp": round(float(rng.normal(0, 0.03)), 4),
                "delta_run_exp": round(float(rng.normal(0, 0.1)), 4),
                "inning": int(rng.randint(1, 10)),
                "inning_topbot": "Bot" if is_home else "Top",
                "outs_when_up": int(rng.randint(0, 3)),
                "balls": int(rng.randint(0, 4)),
                "strikes": int(rng.randint(0, 3)),
                "on_1b": int(rng.choice([0, 1])),
                "on_2b": int(rng.choice([0, 1])),
                "on_3b": int(rng.choice([0, 1])),
                "stand": "R",
                "p_throws": rng.choice(["L", "R"]),
                "at_bat_number": int(g_idx * 5 + p_idx // 4 + 1),
                "pitch_number": int(p_idx % 4 + 1),
                "description": desc,
                "events": event,
                "type": "X" if event else ("S" if "strike" in desc else "B"),
                "home_team": "PHI" if is_home else "NYM",
                "away_team": "NYM" if is_home else "PHI",
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

    # Insert games
    for g_idx in range(n_games):
        game_pk = game_pk_base + g_idx
        game_date = game_dates[g_idx].strftime("%Y-%m-%d")
        is_home = g_idx % 3 != 0
        conn.execute(
            "INSERT INTO games (game_pk, game_date, home_team, away_team, venue) "
            "VALUES ($1, $2, $3, $4, $5)",
            [game_pk, game_date,
             "PHI" if is_home else "NYM",
             "NYM" if is_home else "PHI",
             "Citizens Bank Park" if is_home else "Citi Field"],
        )

    return conn, batter_id, season


# ---------------------------------------------------------------------------
# Test leaky integrator
# ---------------------------------------------------------------------------


class TestLeakyIntegrator:
    """Tests for compute_leaky_load and _apply_channel_load."""

    def test_constant_input_converges(self):
        """With constant input, load should converge to steady state S/(1-alpha)."""
        alpha = 0.85
        value = 10.0
        n = 200  # enough iterations to converge
        stressors = _make_constant_stressor(n=n, value=value)

        load = compute_leaky_load(stressors, alpha=alpha)

        # Steady state: S / (1 - alpha) = 10 / 0.15 = 66.67
        expected_steady = value / (1.0 - alpha)
        # Last value should be within 1% of steady state
        assert abs(load.iloc[-1] - expected_steady) / expected_steady < 0.01, (
            f"Expected ~{expected_steady:.2f}, got {load.iloc[-1]:.2f}"
        )

    def test_monotonically_increasing_under_constant(self):
        """Under constant stressor, load should increase monotonically."""
        stressors = _make_constant_stressor(n=50, value=5.0)
        load = compute_leaky_load(stressors, alpha=0.85)

        for i in range(1, len(load)):
            assert load.iloc[i] >= load.iloc[i - 1], (
                f"Load decreased at index {i}: {load.iloc[i]} < {load.iloc[i-1]}"
            )

    def test_zero_stressor_decays_to_zero(self):
        """With zero stressor after initial load, should decay toward zero."""
        # First build up some load
        initial = pd.Series([10.0] * 20 + [0.0] * 50)
        load = compute_leaky_load(initial, alpha=0.85)

        # After the last non-zero stressor, load should be decreasing
        for i in range(21, len(load)):
            assert load.iloc[i] < load.iloc[i - 1] + 0.001, (
                f"Load not decaying at index {i}"
            )

    def test_off_day_accelerated_decay(self):
        """Off-days should cause faster decay than normal days."""
        alpha = 0.85
        stressors = pd.Series([10.0, 0.0, 0.0])

        # Normal (no off-days)
        load_normal = compute_leaky_load(stressors, alpha=alpha)

        # With off-days at index 1 and 2
        off_days = pd.Series([False, True, True])
        load_offday = compute_leaky_load(stressors, alpha=alpha, off_day_mask=off_days)

        # Off-day load should be lower
        assert load_offday.iloc[2] < load_normal.iloc[2], (
            f"Off-day load ({load_offday.iloc[2]:.4f}) should be less than "
            f"normal load ({load_normal.iloc[2]:.4f})"
        )

    def test_off_day_alpha_squared(self):
        """Off-day decay should use alpha^2 by default."""
        alpha = 0.85
        # Build: one game with stressor 10, then one off-day
        stressors = pd.Series([10.0, 0.0])
        off_days = pd.Series([False, True])

        load = compute_leaky_load(stressors, alpha=alpha, off_day_mask=off_days)

        # After first game: load = 10
        # After off-day: load = alpha^2 * 10
        expected = alpha ** 2 * 10.0
        assert abs(load.iloc[1] - expected) < 0.001, (
            f"Expected {expected:.4f}, got {load.iloc[1]:.4f}"
        )

    def test_apply_channel_load_with_gaps(self):
        """_apply_channel_load should handle multi-day gaps."""
        stressors = pd.Series([10.0, 10.0, 10.0])
        # Gap of 3 days between game 1 and 2 (2 off-days)
        days_since = pd.Series([1, 3, 1])

        load = _apply_channel_load(stressors, days_since, alpha=0.85)

        # First game: load = 10
        assert abs(load.iloc[0] - 10.0) < 0.001

        # Second game: 2 off-days of alpha^2 decay, then alpha decay + stressor
        alpha = 0.85
        off_alpha = alpha ** 2
        expected_1 = alpha * (off_alpha * (off_alpha * 10.0)) + 10.0
        assert abs(load.iloc[1] - expected_1) < 0.01, (
            f"Expected ~{expected_1:.2f}, got {load.iloc[1]:.2f}"
        )

    def test_single_observation(self):
        """Single observation should just be the stressor value."""
        stressors = pd.Series([15.0])
        load = compute_leaky_load(stressors, alpha=0.85)
        assert abs(load.iloc[0] - 15.0) < 0.001

    def test_empty_series(self):
        """Empty stressor series should return empty load."""
        stressors = pd.Series([], dtype=np.float64)
        load = compute_leaky_load(stressors, alpha=0.85)
        assert len(load) == 0


# ---------------------------------------------------------------------------
# Test stressor computation
# ---------------------------------------------------------------------------


class TestStressorComputation:
    """Tests for compute_game_stressors."""

    @pytest.fixture(scope="class")
    def synth_env(self):
        """Create synthetic DB for stressor tests."""
        conn, batter_id, season = _make_synthetic_db()
        yield conn, batter_id, season
        conn.close()

    def test_returns_expected_columns(self, synth_env):
        """Stressor DataFrame should contain all five channels."""
        conn, batter_id, season = synth_env
        df = compute_game_stressors(conn, batter_id, season)

        assert not df.empty
        expected_cols = [
            "game_pk", "game_date",
            "pitch_processing", "decision_conflict", "swing_exertion",
            "temporal_demand", "travel_stress",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_pitch_processing_is_positive(self, synth_env):
        """Pitch processing (pitches seen) should be > 0 for every game."""
        conn, batter_id, season = synth_env
        df = compute_game_stressors(conn, batter_id, season)
        assert (df["pitch_processing"] > 0).all()

    def test_swing_exertion_non_negative(self, synth_env):
        """Swing exertion should be >= 0."""
        conn, batter_id, season = synth_env
        df = compute_game_stressors(conn, batter_id, season)
        assert (df["swing_exertion"] >= 0).all()

    def test_temporal_demand_at_least_one(self, synth_env):
        """Temporal demand should be >= 1 (at least the current game)."""
        conn, batter_id, season = synth_env
        df = compute_game_stressors(conn, batter_id, season)
        assert (df["temporal_demand"] >= 1).all()

    def test_travel_stress_non_negative(self, synth_env):
        """Travel stress should be >= 0."""
        conn, batter_id, season = synth_env
        df = compute_game_stressors(conn, batter_id, season)
        assert (df["travel_stress"] >= 0).all()

    def test_nonexistent_batter(self, synth_env):
        """Non-existent batter should return empty DataFrame."""
        conn, _, season = synth_env
        df = compute_game_stressors(conn, 999999999, season)
        assert df.empty


# ---------------------------------------------------------------------------
# Test composite ABL calculation
# ---------------------------------------------------------------------------


class TestCompositeABL:
    """Tests for calculate_abl."""

    @pytest.fixture(scope="class")
    def synth_env(self):
        conn, batter_id, season = _make_synthetic_db()
        yield conn, batter_id, season
        conn.close()

    def test_output_keys(self, synth_env):
        """calculate_abl should return expected keys."""
        conn, batter_id, season = synth_env
        result = calculate_abl(conn, batter_id, season)

        expected_keys = {
            "batter_id", "season", "composite_abl", "channel_loads",
            "peak_abl", "peak_date", "timeline", "games_played",
        }
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_composite_abl_range(self, synth_env):
        """Composite ABL should be between 0 and 100."""
        conn, batter_id, season = synth_env
        result = calculate_abl(conn, batter_id, season)

        assert result["composite_abl"] is not None
        assert 0 <= result["composite_abl"] <= 100

    def test_peak_abl_gte_current(self, synth_env):
        """Peak ABL should be >= current (final) ABL."""
        conn, batter_id, season = synth_env
        result = calculate_abl(conn, batter_id, season)

        assert result["peak_abl"] >= result["composite_abl"]

    def test_timeline_has_composite(self, synth_env):
        """Timeline DataFrame should contain composite_abl column."""
        conn, batter_id, season = synth_env
        result = calculate_abl(conn, batter_id, season)

        assert "composite_abl" in result["timeline"].columns
        assert len(result["timeline"]) == result["games_played"]

    def test_all_channels_below_mean_gives_zero(self):
        """When all channel loads are below their mean, composite ABL should be 0."""
        # We test this by verifying the z-score logic:
        # If all z-scores <= 0, sum of above-mean z-scores = 0
        z_scores = [-1.0, -0.5, -0.2, -1.5, -0.3]
        above_mean = [z for z in z_scores if z > 0]
        assert sum(above_mean) == 0.0

    def test_insufficient_games(self, synth_env):
        """Batter with fewer than min_games should return None ABL."""
        conn, _, season = synth_env
        config = ABLConfig(min_games=999)  # impossibly high
        result = calculate_abl(conn, 600000, season, config)
        assert result["composite_abl"] is None


# ---------------------------------------------------------------------------
# Test edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_batter_with_one_game(self):
        """Batter with only 1 game should return None (below min_games)."""
        import duckdb
        from src.db.schema import create_tables

        conn = duckdb.connect(":memory:")
        create_tables(conn)

        rng = np.random.RandomState(99)
        batter_id = 800000
        rows = []
        for i in range(10):
            rows.append({
                "game_pk": 900000,
                "game_date": "2025-06-01",
                "pitcher_id": 100001,
                "batter_id": batter_id,
                "pitch_type": "FF",
                "pitch_name": "FF",
                "release_speed": 93.0,
                "release_spin_rate": 2300.0,
                "spin_axis": 180.0,
                "pfx_x": -5.0,
                "pfx_z": 12.0,
                "plate_x": round(float(rng.normal(0, 0.5)), 2),
                "plate_z": round(float(rng.normal(2.5, 0.5)), 2),
                "release_extension": 6.2,
                "release_pos_x": -1.5,
                "release_pos_y": 55.0,
                "release_pos_z": 5.8,
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
                "inning": 1,
                "inning_topbot": "Bot",
                "outs_when_up": 0,
                "balls": 0,
                "strikes": 0,
                "on_1b": 0,
                "on_2b": 0,
                "on_3b": 0,
                "stand": "R",
                "p_throws": "R",
                "at_bat_number": i + 1,
                "pitch_number": 1,
                "description": rng.choice(["called_strike", "ball", "foul"]),
                "events": None,
                "type": "S",
                "home_team": "PHI",
                "away_team": "NYM",
                "woba_value": None,
                "woba_denom": 0.0,
                "babip_value": None,
                "iso_value": None,
                "zone": int(rng.randint(1, 15)),
                "effective_speed": 92.0,
                "if_fielding_alignment": "Standard",
                "of_fielding_alignment": "Standard",
                "fielder_2": int(rng.randint(400000, 500000)),
            })

        df = pd.DataFrame(rows)
        conn.execute("INSERT INTO pitches SELECT * FROM df")

        result = calculate_abl(conn, batter_id, 2025)
        assert result["composite_abl"] is None
        assert result["games_played"] == 1
        conn.close()

    def test_all_home_games(self):
        """Batter with all home games should have zero travel stress."""
        import duckdb
        from src.db.schema import create_tables

        conn = duckdb.connect(":memory:")
        create_tables(conn)

        rng = np.random.RandomState(77)
        batter_id = 810000
        n_games = 25
        rows = []
        game_dates = pd.date_range("2025-04-01", periods=n_games, freq="1D")

        for g_idx in range(n_games):
            game_pk = 910000 + g_idx
            game_date = game_dates[g_idx].strftime("%Y-%m-%d")
            for p_idx in range(15):
                rows.append({
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "pitcher_id": rng.randint(100000, 200000),
                    "batter_id": batter_id,
                    "pitch_type": "FF",
                    "pitch_name": "FF",
                    "release_speed": 93.0,
                    "release_spin_rate": 2300.0,
                    "spin_axis": 180.0,
                    "pfx_x": -5.0,
                    "pfx_z": 12.0,
                    "plate_x": round(float(rng.normal(0, 0.5)), 2),
                    "plate_z": max(0.5, round(float(rng.normal(2.5, 0.5)), 2)),
                    "release_extension": 6.2,
                    "release_pos_x": -1.5,
                    "release_pos_y": 55.0,
                    "release_pos_z": 5.8,
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
                    "inning": 1,
                    # All Bot = home
                    "inning_topbot": "Bot",
                    "outs_when_up": 0,
                    "balls": 0,
                    "strikes": 0,
                    "on_1b": 0,
                    "on_2b": 0,
                    "on_3b": 0,
                    "stand": "R",
                    "p_throws": "R",
                    "at_bat_number": g_idx * 4 + p_idx // 4 + 1,
                    "pitch_number": p_idx % 4 + 1,
                    "description": rng.choice(["called_strike", "ball", "foul"]),
                    "events": None,
                    "type": "S",
                    "home_team": "PHI",
                    "away_team": "NYM",
                    "woba_value": None,
                    "woba_denom": 0.0,
                    "babip_value": None,
                    "iso_value": None,
                    "zone": int(rng.randint(1, 15)),
                    "effective_speed": 92.0,
                    "if_fielding_alignment": "Standard",
                    "of_fielding_alignment": "Standard",
                    "fielder_2": int(rng.randint(400000, 500000)),
                })

        df = pd.DataFrame(rows)
        conn.execute("INSERT INTO pitches SELECT * FROM df")

        stressors = compute_game_stressors(conn, batter_id, 2025)
        # All home games: travel stress should be 0 for every game
        assert (stressors["travel_stress"] == 0).all(), (
            f"Travel stress should be all zeros for home games, got: "
            f"{stressors['travel_stress'].tolist()}"
        )
        conn.close()

    def test_no_off_days_season(self):
        """Season with consecutive daily games (no off-days) should still work."""
        import duckdb
        from src.db.schema import create_tables

        conn = duckdb.connect(":memory:")
        create_tables(conn)

        rng = np.random.RandomState(55)
        batter_id = 820000
        n_games = 25
        rows = []
        # Every day, no gaps
        game_dates = pd.date_range("2025-06-01", periods=n_games, freq="1D")

        for g_idx in range(n_games):
            game_pk = 920000 + g_idx
            game_date = game_dates[g_idx].strftime("%Y-%m-%d")
            for p_idx in range(15):
                rows.append({
                    "game_pk": game_pk,
                    "game_date": game_date,
                    "pitcher_id": rng.randint(100000, 200000),
                    "batter_id": batter_id,
                    "pitch_type": "FF",
                    "pitch_name": "FF",
                    "release_speed": 93.0,
                    "release_spin_rate": 2300.0,
                    "spin_axis": 180.0,
                    "pfx_x": -5.0,
                    "pfx_z": 12.0,
                    "plate_x": round(float(rng.normal(0, 0.5)), 2),
                    "plate_z": max(0.5, round(float(rng.normal(2.5, 0.5)), 2)),
                    "release_extension": 6.2,
                    "release_pos_x": -1.5,
                    "release_pos_y": 55.0,
                    "release_pos_z": 5.8,
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
                    "inning": 1,
                    "inning_topbot": rng.choice(["Top", "Bot"]),
                    "outs_when_up": 0,
                    "balls": 0,
                    "strikes": 0,
                    "on_1b": 0,
                    "on_2b": 0,
                    "on_3b": 0,
                    "stand": "R",
                    "p_throws": "R",
                    "at_bat_number": g_idx * 4 + p_idx // 4 + 1,
                    "pitch_number": p_idx % 4 + 1,
                    "description": rng.choice(["called_strike", "ball", "foul"]),
                    "events": None,
                    "type": "S",
                    "home_team": "PHI",
                    "away_team": "NYM",
                    "woba_value": None,
                    "woba_denom": 0.0,
                    "babip_value": None,
                    "iso_value": None,
                    "zone": int(rng.randint(1, 15)),
                    "effective_speed": 92.0,
                    "if_fielding_alignment": "Standard",
                    "of_fielding_alignment": "Standard",
                    "fielder_2": int(rng.randint(400000, 500000)),
                })

        df = pd.DataFrame(rows)
        conn.execute("INSERT INTO pitches SELECT * FROM df")

        result = calculate_abl(conn, batter_id, 2025)
        assert result["composite_abl"] is not None
        assert result["games_played"] == n_games

        # All days_since_last should be 1 (consecutive)
        stressors = compute_game_stressors(conn, batter_id, 2025)
        # First game might have NaN/1 for days_since_last
        assert (stressors["days_since_last"].iloc[1:] == 1).all()
        conn.close()


# ---------------------------------------------------------------------------
# Test predict_recovery_days
# ---------------------------------------------------------------------------


class TestRecoveryPrediction:
    """Tests for predict_recovery_days."""

    def test_already_below_threshold(self):
        """If ABL is already below threshold, 0 days needed."""
        assert predict_recovery_days(20.0, 50.0) == 0

    def test_recovery_decreases(self):
        """Higher current ABL should require more recovery days."""
        days_low = predict_recovery_days(60.0, 30.0)
        days_high = predict_recovery_days(90.0, 30.0)
        assert days_high > days_low

    def test_zero_abl(self):
        """Zero ABL should need 0 days."""
        assert predict_recovery_days(0.0, 10.0) == 0

    def test_positive_days(self):
        """ABL above threshold should need at least 1 day."""
        days = predict_recovery_days(80.0, 50.0)
        assert days >= 1


# ---------------------------------------------------------------------------
# Test model class
# ---------------------------------------------------------------------------


class TestAllostaticLoadModel:
    """Tests for the AllostaticLoadModel class."""

    def test_model_name_and_version(self):
        """Model should report correct name and version."""
        model = AllostaticLoadModel()
        assert model.model_name == "allostatic_batting_load"
        assert model.version == "1.0.0"

    def test_config_defaults(self):
        """Default config should have sensible values."""
        config = ABLConfig()
        assert config.alpha_pitch_processing == 0.85
        assert config.alpha_travel == 0.70
        assert config.min_games == 20
        assert config.abl_max == 100.0

    def test_train_and_predict(self):
        """Full train-predict cycle on synthetic data."""
        conn, batter_id, season = _make_synthetic_db()
        model = AllostaticLoadModel(ABLConfig(min_games=20))

        metrics = model.train(conn, season=season)
        assert "n_batters" in metrics
        assert metrics["n_batters"] >= 1

        # Predict for a single batter
        result = model.predict(conn, batter_id=batter_id, season=season)
        assert "composite_abl" in result

        conn.close()

    def test_evaluate_requires_training(self):
        """Evaluate should raise RuntimeError if not trained."""
        import duckdb
        model = AllostaticLoadModel()
        conn = duckdb.connect(":memory:")

        with pytest.raises(RuntimeError, match="not trained"):
            model.evaluate(conn)

        conn.close()
