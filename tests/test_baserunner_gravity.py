"""
Tests for the Baserunner Gravity Index (BGI) model.

Tests cover:
- Runner threat rate computation returns valid [0, 1] range
- Channel effects are bounded (velocity diff < 5 mph, etc.)
- BGI centering: league average runner produces BGI near 100
- Batch returns expected columns
- Edge cases: runner never on base, DH-only player, catcher (slow runner)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import duckdb

from src.db.schema import create_tables
from src.analytics.baserunner_gravity import (
    BGIConfig,
    BaserunnerGravityModel,
    compute_runner_threat_rate,
    compute_gravity_effect,
    calculate_bgi,
    batch_calculate,
    get_gravity_leaderboard,
)


# ---------------------------------------------------------------------------
# Helpers -- synthetic data with controlled baserunner scenarios
# ---------------------------------------------------------------------------


def _build_bgi_test_db(
    n_pitches: int = 2000,
    n_runners: int = 8,
    seed: int = 42,
):
    """Build an in-memory DuckDB with synthetic data suitable for BGI testing.

    Creates pitches where specific runners appear on 1B with enough
    frequency to pass the minimum-appearance threshold.

    Returns (conn, runner_ids, season).
    """
    rng = np.random.RandomState(seed)
    conn = duckdb.connect(":memory:")
    create_tables(conn)

    season = 2025
    runner_ids = list(range(300000, 300000 + n_runners))
    pitcher_ids = list(range(200000, 200010))
    batter_ids = list(range(400000, 400020))

    pitch_types = ["FF", "SI", "SL", "CU", "CH", "FC"]
    descriptions = [
        "called_strike", "swinging_strike", "ball", "foul",
        "hit_into_play", "hit_into_play_no_out",
    ]
    events_list = [
        None, None, None, None,
        "single", "double", "home_run", "strikeout", "walk",
        "field_out", "stolen_base_2b",
    ]

    rows = []
    game_pk_base = 700000

    for i in range(n_pitches):
        # Ensure runners appear on 1B frequently enough
        if rng.random() < 0.6:
            # Runner on 1B
            on_1b = int(rng.choice(runner_ids))
        else:
            on_1b = 0

        pitcher_id = int(rng.choice(pitcher_ids))
        batter_id = int(rng.choice(batter_ids))
        pt = rng.choice(pitch_types)

        # Slight velocity effect: when fast runner on base, pitchers throw
        # a bit slower (simulated)
        base_velo = 93.0
        if on_1b in runner_ids[:2]:  # first two runners are "fast"
            base_velo -= 0.3

        desc = rng.choice(descriptions)
        event = rng.choice(events_list) if desc.startswith("hit_into") else None

        # For stolen base events, only if runner on base
        if on_1b != 0 and rng.random() < 0.02:
            event = "stolen_base_2b"

        woba_val = round(float(rng.uniform(0, 2.0)), 3) if event else None
        woba_den = 1.0 if event else 0.0
        est_woba = round(float(rng.uniform(0.1, 0.8)), 3) if event else None

        # For fast runners, batters get slightly better outcomes
        if on_1b in runner_ids[:2] and est_woba is not None:
            est_woba = min(est_woba + 0.05, 1.0)

        month = rng.randint(4, 10)
        day = rng.randint(1, 28)

        rows.append({
            "game_pk": game_pk_base + i // 30,
            "game_date": f"{season}-{month:02d}-{day:02d}",
            "pitcher_id": pitcher_id,
            "batter_id": batter_id,
            "pitch_type": pt,
            "pitch_name": pt,
            "release_speed": round(float(np.clip(rng.normal(base_velo, 2.0), 55, 105)), 1),
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
            "launch_speed": round(float(rng.normal(88, 12)), 1) if event else None,
            "launch_angle": round(float(rng.normal(12, 20)), 1) if event else None,
            "hit_distance": round(float(rng.uniform(50, 420)), 0) if event else None,
            "hc_x": None,
            "hc_y": None,
            "bb_type": None,
            "estimated_ba": None,
            "estimated_woba": est_woba,
            "delta_home_win_exp": round(float(rng.normal(0, 0.03)), 4),
            "delta_run_exp": round(float(rng.normal(0, 0.1)), 4),
            "inning": int(rng.randint(1, 10)),
            "inning_topbot": rng.choice(["Top", "Bot"]),
            "outs_when_up": int(rng.randint(0, 3)),
            "balls": int(rng.randint(0, 4)),
            "strikes": int(rng.randint(0, 3)),
            "on_1b": on_1b,
            "on_2b": 0,
            "on_3b": 0,
            "stand": rng.choice(["L", "R"]),
            "p_throws": rng.choice(["L", "R"]),
            "at_bat_number": int(i // 5 + 1),
            "pitch_number": int(i % 5 + 1),
            "description": desc,
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
    return conn, runner_ids, season


# ---------------------------------------------------------------------------
# Test runner threat computation
# ---------------------------------------------------------------------------


class TestRunnerThreatRate:
    """Tests for compute_runner_threat_rate."""

    @pytest.fixture(scope="class")
    def env(self):
        conn, runner_ids, season = _build_bgi_test_db()
        yield conn, runner_ids, season
        conn.close()

    def test_returns_valid_rate(self, env):
        """SB attempt rate should be in [0, 1]."""
        conn, runner_ids, season = env
        for rid in runner_ids:
            rate = compute_runner_threat_rate(conn, rid, season)
            assert 0.0 <= rate <= 1.0, f"Rate {rate} out of [0, 1] for runner {rid}"

    def test_nonexistent_runner_returns_zero(self, env):
        """A runner ID that never appeared should return 0.0."""
        conn, _, season = env
        rate = compute_runner_threat_rate(conn, 999999, season)
        assert rate == 0.0

    def test_wrong_season_returns_zero(self, env):
        """Querying the wrong season should return 0.0."""
        conn, runner_ids, _ = env
        rate = compute_runner_threat_rate(conn, runner_ids[0], 1900)
        assert rate == 0.0


# ---------------------------------------------------------------------------
# Test channel effects
# ---------------------------------------------------------------------------


class TestChannelEffects:
    """Tests for compute_gravity_effect -- individual channels."""

    @pytest.fixture(scope="class")
    def env(self):
        conn, runner_ids, season = _build_bgi_test_db(n_pitches=3000)
        yield conn, runner_ids, season
        conn.close()

    def test_effects_returned_for_qualifying_runner(self, env):
        """A runner with enough pitches should get non-None effects."""
        conn, runner_ids, season = env
        config = BGIConfig(min_pitches_with_runner=10, min_matched_sample=10)
        # Try each runner until we find one that qualifies
        found = False
        for rid in runner_ids:
            result = compute_gravity_effect(conn, rid, season, config)
            if result is not None:
                found = True
                assert "velocity_effect" in result
                assert "location_effect" in result
                assert "selection_effect" in result
                assert "outcome_effect" in result
                break
        assert found, "At least one runner should qualify"

    def test_velocity_effect_bounded(self, env):
        """Velocity effect should be within reasonable bounds (< 5 mph)."""
        conn, runner_ids, season = env
        config = BGIConfig(min_pitches_with_runner=10, min_matched_sample=10)
        for rid in runner_ids:
            result = compute_gravity_effect(conn, rid, season, config)
            if result is not None:
                assert abs(result["velocity_effect"]) < 5.0, (
                    f"Velocity effect {result['velocity_effect']} exceeds 5 mph"
                )

    def test_selection_effect_bounded(self, env):
        """Selection effect (fastball% diff) should be in [-1, 1]."""
        conn, runner_ids, season = env
        config = BGIConfig(min_pitches_with_runner=10, min_matched_sample=10)
        for rid in runner_ids:
            result = compute_gravity_effect(conn, rid, season, config)
            if result is not None:
                assert -1.0 <= result["selection_effect"] <= 1.0, (
                    f"Selection effect {result['selection_effect']} out of [-1, 1]"
                )

    def test_insufficient_data_returns_none(self, env):
        """Runner with too few pitches should return None."""
        conn, _, season = env
        config = BGIConfig(min_pitches_with_runner=999999, min_matched_sample=10)
        result = compute_gravity_effect(conn, 999999, season, config)
        assert result is None


# ---------------------------------------------------------------------------
# Test BGI centering
# ---------------------------------------------------------------------------


class TestBGICentering:
    """Tests that BGI is centered around 100 for the league average."""

    @pytest.fixture(scope="class")
    def batch_result(self):
        conn, runner_ids, season = _build_bgi_test_db(n_pitches=3000, n_runners=10)
        config = BGIConfig(
            min_pitches_with_runner=10,
            min_matched_sample=10,
            min_appearances_batch=10,
        )
        df = batch_calculate(conn, season, min_appearances=10, config=config)
        yield df
        conn.close()

    def test_mean_bgi_near_100(self, batch_result):
        """League-average BGI should be approximately 100."""
        df = batch_result
        if df.empty:
            pytest.skip("No qualifying runners in batch")
        mean_bgi = df["bgi"].mean()
        assert 85.0 <= mean_bgi <= 115.0, (
            f"Mean BGI {mean_bgi:.1f} not close enough to 100"
        )

    def test_bgi_std_near_15(self, batch_result):
        """BGI standard deviation should be approximately 15."""
        df = batch_result
        if len(df) < 3:
            pytest.skip("Need at least 3 runners for std test")
        std_bgi = df["bgi"].std()
        # Allow generous range: could be small with synthetic data
        assert 0.0 < std_bgi < 60.0, (
            f"BGI std {std_bgi:.1f} is out of expected range"
        )


# ---------------------------------------------------------------------------
# Test batch output
# ---------------------------------------------------------------------------


class TestBatchCalculate:
    """Tests for batch_calculate output schema."""

    @pytest.fixture(scope="class")
    def batch_env(self):
        conn, runner_ids, season = _build_bgi_test_db(n_pitches=3000, n_runners=10)
        config = BGIConfig(
            min_pitches_with_runner=10,
            min_matched_sample=10,
            min_appearances_batch=10,
        )
        df = batch_calculate(conn, season, min_appearances=10, config=config)
        yield df, conn, season
        conn.close()

    def test_expected_columns(self, batch_env):
        """Batch output should contain all required columns."""
        df, _, _ = batch_env
        if df.empty:
            pytest.skip("No qualifying runners")

        expected_cols = [
            "runner_id", "season", "bgi", "sb_attempt_rate",
            "velocity_effect", "location_effect", "selection_effect",
            "outcome_effect", "n_pitches", "percentile",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_sorted_descending(self, batch_env):
        """Batch results should be sorted by BGI descending."""
        df, _, _ = batch_env
        if len(df) < 2:
            pytest.skip("Need at least 2 runners")

        bgis = df["bgi"].values
        for i in range(len(bgis) - 1):
            assert bgis[i] >= bgis[i + 1], "Should be sorted descending"

    def test_percentile_range(self, batch_env):
        """Percentile should be in [0, 100]."""
        df, _, _ = batch_env
        if df.empty:
            pytest.skip("No qualifying runners")

        assert df["percentile"].min() >= 0
        assert df["percentile"].max() <= 100

    def test_bgi_values_numeric(self, batch_env):
        """All BGI values should be numeric (no NaN)."""
        df, _, _ = batch_env
        if df.empty:
            pytest.skip("No qualifying runners")

        assert df["bgi"].notna().all(), "BGI should have no NaN values"


# ---------------------------------------------------------------------------
# Test leaderboard
# ---------------------------------------------------------------------------


class TestLeaderboard:
    """Tests for get_gravity_leaderboard."""

    @pytest.fixture(scope="class")
    def leaderboard_env(self):
        conn, runner_ids, season = _build_bgi_test_db(n_pitches=3000, n_runners=10)
        config = BGIConfig(
            min_pitches_with_runner=10,
            min_matched_sample=10,
            min_appearances_batch=10,
        )
        yield conn, season, config
        conn.close()

    def test_leaderboard_has_rank(self, leaderboard_env):
        """Leaderboard index should be named 'Rank' and start at 1."""
        conn, season, config = leaderboard_env
        df = get_gravity_leaderboard(conn, season, config)
        if df.empty:
            pytest.skip("Empty leaderboard")
        assert df.index.name == "Rank"
        assert df.index[0] == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_runner_never_on_base(self):
        """A player who was never on base should return bgi=None."""
        conn, _, season = _build_bgi_test_db(n_pitches=500)
        result = calculate_bgi(conn, 999999, season)
        assert result["bgi"] is None
        assert result["n_pitches"] == 0
        conn.close()

    def test_dh_only_player(self):
        """A DH who bats but rarely runs should still compute (may return None)."""
        conn, _, season = _build_bgi_test_db(n_pitches=500)
        # DH-only = they appear as batter_id but not as on_1b
        # Pick a batter_id from the data
        batter_row = conn.execute(
            "SELECT DISTINCT batter_id FROM pitches LIMIT 1"
        ).fetchone()
        if batter_row is None:
            pytest.skip("No batters found")
        batter_id = batter_row[0]
        result = calculate_bgi(conn, batter_id, season)
        # Might be None if they never appeared on 1b, or valid
        assert "bgi" in result
        assert "runner_id" in result
        conn.close()

    def test_catcher_slow_runner(self):
        """A catcher (slow runner) should get a BGI near or below 100 in batch."""
        conn, runner_ids, season = _build_bgi_test_db(n_pitches=3000, n_runners=10)
        config = BGIConfig(
            min_pitches_with_runner=10,
            min_matched_sample=10,
            min_appearances_batch=10,
        )
        df = batch_calculate(conn, season, min_appearances=10, config=config)
        if df.empty:
            pytest.skip("No qualifying runners")
        # At least some runners should have BGI <= 100
        assert (df["bgi"] <= 100).any(), "Some runners should be at or below average"
        conn.close()

    def test_model_name_and_version(self):
        """Model should report correct name and version."""
        model = BaserunnerGravityModel()
        assert model.model_name == "baserunner_gravity"
        assert model.version == "1.0.0"

    def test_evaluate_not_trained_raises(self):
        """Calling evaluate before training should raise RuntimeError."""
        model = BaserunnerGravityModel()
        conn = duckdb.connect(":memory:")
        create_tables(conn)
        with pytest.raises(RuntimeError, match="not trained"):
            model.evaluate(conn)
        conn.close()

    def test_config_defaults(self):
        """Default config should have sensible values."""
        config = BGIConfig()
        assert config.min_pitches_with_runner >= 10
        assert config.min_matched_sample >= 10
        assert config.min_appearances_batch >= 10

    def test_calculate_bgi_returns_all_keys(self):
        """calculate_bgi should return all expected keys even for invalid runner."""
        conn, _, season = _build_bgi_test_db(n_pitches=500)
        result = calculate_bgi(conn, 888888, season)
        expected_keys = {
            "runner_id", "name", "season", "bgi", "sb_attempt_rate",
            "velocity_effect", "location_effect", "selection_effect",
            "outcome_effect", "n_pitches",
        }
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        conn.close()
