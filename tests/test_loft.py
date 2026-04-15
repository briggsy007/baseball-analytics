"""
Tests for src.analytics.loft -- Lineup Order Flow Toxicity model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.loft import (
    LOFTModel,
    classify_pitch_flow,
    compute_game_loft,
    compute_pitcher_baseline,
    detect_toxicity_events,
    batch_game_analysis,
    calculate_loft,
    _build_buckets,
    _rolling_loft,
    BUCKET_SIZE,
    EWMA_ALPHA,
    HARD_CONTACT_THRESHOLD,
    WEAK_CONTACT_THRESHOLD,
    MIN_PITCHES_GAME,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

ZACK_WHEELER_ID = 554430


def _make_pitch_df(n: int, **overrides) -> pd.DataFrame:
    """Create a synthetic pitch DataFrame with sensible defaults."""
    base = {
        "game_pk": [100000] * n,
        "game_date": ["2025-06-15"] * n,
        "pitcher_id": [ZACK_WHEELER_ID] * n,
        "batter_id": [600000 + i for i in range(n)],
        "at_bat_number": [(i // 5) + 1 for i in range(n)],
        "pitch_number": [(i % 5) + 1 for i in range(n)],
        "pitch_type": ["FF"] * n,
        "description": ["called_strike"] * n,
        "events": [None] * n,
        "type": ["S"] * n,
        "zone": [5] * n,
        "launch_speed": [None] * n,
        "launch_angle": [None] * n,
        "release_speed": [95.0] * n,
        "stand": ["R"] * n,
        "p_throws": ["R"] * n,
        "inning": [1] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


@pytest.fixture
def all_buy_df():
    """30 pitches that should all classify as 'buy': hard contact."""
    return _make_pitch_df(
        30,
        description=["hit_into_play"] * 30,
        type=["X"] * 30,
        launch_speed=[100.0] * 30,
        events=["single"] * 30,
        zone=[5] * 30,
    )


@pytest.fixture
def all_sell_df():
    """30 pitches that should all classify as 'sell': swinging strikes in zone."""
    return _make_pitch_df(
        30,
        description=["swinging_strike"] * 30,
        type=["S"] * 30,
        launch_speed=[None] * 30,
        zone=[5] * 30,
    )


@pytest.fixture
def mixed_df():
    """30 pitches: 15 buy + 15 sell for LOFT=0 in each bucket."""
    buy_descs = ["hit_into_play"] * 15
    sell_descs = ["swinging_strike"] * 15
    # Interleave so each bucket of 15 gets ~equal buy/sell
    descs = []
    types = []
    launch_speeds = []
    events_list = []
    zones = []
    for i in range(30):
        if i % 2 == 0 and len([d for d in descs if d == "hit_into_play"]) < 15:
            descs.append("hit_into_play")
            types.append("X")
            launch_speeds.append(100.0)
            events_list.append("single")
            zones.append(5)
        else:
            descs.append("swinging_strike")
            types.append("S")
            launch_speeds.append(None)
            events_list.append(None)
            zones.append(5)

    return _make_pitch_df(
        30,
        description=descs,
        type=types,
        launch_speed=launch_speeds,
        events=events_list,
        zone=zones,
    )


@pytest.fixture
def equal_bucket_df():
    """15 pitches: exactly 7 buy + 7 sell + 1 neutral for one complete bucket."""
    descs = (
        ["hit_into_play"] * 7
        + ["swinging_strike"] * 7
        + ["foul"]  # foul in zone = neutral (swing in zone, not a whiff in this context)
    )
    types = ["X"] * 7 + ["S"] * 7 + ["S"]
    ls = [100.0] * 7 + [None] * 7 + [None]
    evts = ["single"] * 7 + [None] * 7 + [None]
    zones = [5] * 15
    return _make_pitch_df(
        15,
        description=descs,
        type=types,
        launch_speed=ls,
        events=evts,
        zone=zones,
    )


# ── TestPitchClassification ──────────────────────────────────────────────────


class TestPitchClassification:
    """Tests for classify_pitch_flow."""

    def test_hard_contact_is_buy(self):
        """A pitch with launch_speed >= 95 should be classified as 'buy'."""
        df = _make_pitch_df(1, description=["hit_into_play"], type=["X"],
                            launch_speed=[98.0], events=["single"], zone=[5])
        result = classify_pitch_flow(df)
        assert result["flow"].iloc[0] == "buy"

    def test_walk_is_buy(self):
        """A walk event should be 'buy'."""
        df = _make_pitch_df(1, description=["ball"], type=["B"],
                            events=["walk"], zone=[12])
        result = classify_pitch_flow(df)
        assert result["flow"].iloc[0] == "buy"

    def test_hbp_is_buy(self):
        """Hit-by-pitch should be 'buy'."""
        df = _make_pitch_df(1, description=["hit_by_pitch"], type=["B"],
                            events=["hit_by_pitch"], zone=[12])
        result = classify_pitch_flow(df)
        assert result["flow"].iloc[0] == "buy"

    def test_good_take_outside_zone_is_buy(self):
        """Taking a ball outside the zone is a 'buy' (good plate discipline)."""
        df = _make_pitch_df(1, description=["ball"], type=["B"],
                            zone=[13])
        result = classify_pitch_flow(df)
        assert result["flow"].iloc[0] == "buy"

    def test_ball_in_zone_is_buy(self):
        """A ball called inside the zone means the pitcher missed -- 'buy'."""
        df = _make_pitch_df(1, description=["ball"], type=["B"],
                            zone=[5])
        result = classify_pitch_flow(df)
        assert result["flow"].iloc[0] == "buy"

    def test_swinging_strike_is_sell(self):
        """Swinging strike should be 'sell'."""
        df = _make_pitch_df(1, description=["swinging_strike"], type=["S"],
                            zone=[5])
        result = classify_pitch_flow(df)
        assert result["flow"].iloc[0] == "sell"

    def test_chase_outside_zone_is_sell(self):
        """Swinging at a pitch outside the zone (foul) is a 'sell' (chase)."""
        df = _make_pitch_df(1, description=["foul"], type=["S"],
                            zone=[14])
        result = classify_pitch_flow(df)
        assert result["flow"].iloc[0] == "sell"

    def test_weak_contact_is_sell(self):
        """Weak contact (launch_speed < 85) is 'sell'."""
        df = _make_pitch_df(1, description=["hit_into_play"], type=["X"],
                            launch_speed=[70.0], events=["field_out"], zone=[5])
        result = classify_pitch_flow(df)
        assert result["flow"].iloc[0] == "sell"

    def test_called_strike_in_zone_is_sell(self):
        """Called strike in the zone is 'sell'."""
        df = _make_pitch_df(1, description=["called_strike"], type=["S"],
                            zone=[5])
        result = classify_pitch_flow(df)
        assert result["flow"].iloc[0] == "sell"

    def test_flow_column_added(self):
        """classify_pitch_flow adds a 'flow' column."""
        df = _make_pitch_df(5)
        result = classify_pitch_flow(df)
        assert "flow" in result.columns

    def test_all_values_valid(self):
        """All flow values should be 'buy', 'sell', or 'neutral'."""
        df = _make_pitch_df(20)
        result = classify_pitch_flow(df)
        assert set(result["flow"].unique()) <= {"buy", "sell", "neutral"}

    def test_hard_contact_beats_whiff_classification(self):
        """foul_tip with high exit velo should be 'buy' (hard contact priority)."""
        df = _make_pitch_df(1, description=["foul_tip"], type=["S"],
                            launch_speed=[100.0], zone=[5])
        result = classify_pitch_flow(df)
        assert result["flow"].iloc[0] == "buy"


# ── TestBucketConstruction ───────────────────────────────────────────────────


class TestBucketConstruction:
    """Tests for _build_buckets."""

    def test_bucket_size(self, all_buy_df):
        """Each bucket should contain exactly BUCKET_SIZE pitches (except possibly last)."""
        classified = classify_pitch_flow(all_buy_df)
        buckets = _build_buckets(classified, bucket_size=15)
        assert len(buckets) == 2  # 30 pitches / 15 = 2 buckets

    def test_partial_last_bucket(self):
        """20 pitches with bucket_size=15 should give 2 buckets (15 + 5)."""
        df = _make_pitch_df(20)
        classified = classify_pitch_flow(df)
        buckets = _build_buckets(classified, bucket_size=15)
        assert len(buckets) == 2

    def test_empty_df(self):
        """Empty DataFrame should produce empty buckets."""
        df = pd.DataFrame(columns=["flow"])
        buckets = _build_buckets(df)
        assert len(buckets) == 0

    def test_single_bucket(self):
        """Exactly 15 pitches should make 1 bucket."""
        df = _make_pitch_df(15)
        classified = classify_pitch_flow(df)
        buckets = _build_buckets(classified, bucket_size=15)
        assert len(buckets) == 1

    def test_bucket_has_required_columns(self, all_buy_df):
        """Each bucket row should have the expected fields."""
        classified = classify_pitch_flow(all_buy_df)
        buckets = _build_buckets(classified)
        required = {"bucket_id", "buy_count", "sell_count", "total_flow", "loft", "start_idx", "end_idx"}
        assert required <= set(buckets.columns)


# ── TestLOFTFormula ──────────────────────────────────────────────────────────


class TestLOFTFormula:
    """Tests for the LOFT formula: |V_buy - V_sell| / (V_buy + V_sell)."""

    def test_all_buys_loft_equals_one(self, all_buy_df):
        """When every pitch is a buy, LOFT = |N-0| / N = 1.0."""
        classified = classify_pitch_flow(all_buy_df)
        buckets = _build_buckets(classified, bucket_size=15)
        for _, row in buckets.iterrows():
            if row["total_flow"] > 0:
                assert abs(row["loft"] - 1.0) < 0.01

    def test_all_sells_loft_equals_one(self, all_sell_df):
        """When every pitch is a sell, LOFT = |0-N| / N = 1.0."""
        classified = classify_pitch_flow(all_sell_df)
        buckets = _build_buckets(classified, bucket_size=15)
        for _, row in buckets.iterrows():
            if row["total_flow"] > 0:
                assert abs(row["loft"] - 1.0) < 0.01

    def test_equal_buy_sell_loft_zero(self):
        """Equal buys and sells should give LOFT = 0."""
        flow = pd.Series(["buy"] * 7 + ["sell"] * 7 + ["neutral"])
        df = pd.DataFrame({"flow": flow})
        buckets = _build_buckets(df, bucket_size=15)
        assert len(buckets) == 1
        assert abs(buckets["loft"].iloc[0]) < 0.01

    def test_loft_bounded_zero_one(self, all_buy_df):
        """LOFT values should always be in [0, 1]."""
        classified = classify_pitch_flow(all_buy_df)
        buckets = _build_buckets(classified, bucket_size=15)
        assert (buckets["loft"] >= 0.0).all()
        assert (buckets["loft"] <= 1.0).all()

    def test_all_neutral_loft_zero(self):
        """If all pitches are neutral, total_flow=0 and LOFT=0."""
        flow = pd.Series(["neutral"] * 15)
        df = pd.DataFrame({"flow": flow})
        buckets = _build_buckets(df, bucket_size=15)
        assert buckets["loft"].iloc[0] == 0.0


# ── TestEWMASmoothing ────────────────────────────────────────────────────────


class TestEWMASmoothing:
    """Tests for _rolling_loft EWMA properties."""

    def test_single_bucket_equals_raw(self):
        """With one bucket, EWMA should equal the raw LOFT value."""
        bucket_df = pd.DataFrame({"loft": [0.8]})
        result = _rolling_loft(bucket_df, alpha=0.3)
        assert abs(result.iloc[0] - 0.8) < 1e-6

    def test_constant_input_stays_constant(self):
        """If all bucket LOFT values are the same, EWMA should be that value."""
        bucket_df = pd.DataFrame({"loft": [0.5, 0.5, 0.5, 0.5, 0.5]})
        result = _rolling_loft(bucket_df, alpha=0.3)
        for val in result:
            assert abs(val - 0.5) < 1e-6

    def test_ewma_smooths_spike(self):
        """A spike in LOFT should be smoothed (EWMA value < spike value)."""
        bucket_df = pd.DataFrame({"loft": [0.2, 0.2, 0.2, 1.0, 0.2]})
        result = _rolling_loft(bucket_df, alpha=0.3)
        # The smoothed value at the spike should be less than 1.0
        assert result.iloc[3] < 1.0
        # But higher than the pre-spike level
        assert result.iloc[3] > 0.2

    def test_ewma_respects_alpha(self):
        """Higher alpha = more responsive to recent values."""
        bucket_df = pd.DataFrame({"loft": [0.0, 0.0, 0.0, 1.0]})
        result_high = _rolling_loft(bucket_df, alpha=0.9)
        result_low = _rolling_loft(bucket_df, alpha=0.1)
        # High alpha should be closer to the spike at index 3
        assert result_high.iloc[3] > result_low.iloc[3]

    def test_empty_input(self):
        """Empty bucket DataFrame should return empty series."""
        bucket_df = pd.DataFrame(columns=["loft"])
        result = _rolling_loft(bucket_df)
        assert len(result) == 0


# ── TestThresholdDetection ───────────────────────────────────────────────────


class TestThresholdDetection:
    """Tests for detect_toxicity_events."""

    def test_no_alerts_below_threshold(self):
        """When all buckets are below threshold, no alerts should fire."""
        game_loft = {
            "buckets": [
                {"bucket_id": 0, "loft": 0.3, "rolling_loft": 0.3,
                 "buy_count": 5, "sell_count": 8, "start_idx": 0, "end_idx": 14},
            ],
        }
        baseline = {"mean_loft": 0.5, "std_loft": 0.2}
        alerts = detect_toxicity_events(game_loft, baseline, sigma=2.0)
        assert len(alerts) == 0

    def test_alerts_above_threshold(self):
        """A bucket with rolling LOFT > mean + 2*std should trigger an alert."""
        game_loft = {
            "buckets": [
                {"bucket_id": 0, "loft": 1.0, "rolling_loft": 1.0,
                 "buy_count": 15, "sell_count": 0, "start_idx": 0, "end_idx": 14},
            ],
        }
        baseline = {"mean_loft": 0.3, "std_loft": 0.1}
        alerts = detect_toxicity_events(game_loft, baseline, sigma=2.0)
        assert len(alerts) == 1
        assert alerts[0]["bucket_id"] == 0

    def test_alert_contains_required_fields(self):
        """Each alert should contain bucket_id, rolling_loft, threshold, excess_sigma."""
        game_loft = {
            "buckets": [
                {"bucket_id": 0, "loft": 1.0, "rolling_loft": 1.0,
                 "buy_count": 15, "sell_count": 0, "start_idx": 0, "end_idx": 14},
            ],
        }
        baseline = {"mean_loft": 0.3, "std_loft": 0.1}
        alerts = detect_toxicity_events(game_loft, baseline, sigma=2.0)
        alert = alerts[0]
        assert "bucket_id" in alert
        assert "rolling_loft" in alert
        assert "threshold" in alert
        assert "excess_sigma" in alert

    def test_no_alerts_with_zero_std(self):
        """If baseline std is 0, no alerts should fire (avoid division by zero)."""
        game_loft = {
            "buckets": [
                {"bucket_id": 0, "loft": 1.0, "rolling_loft": 1.0,
                 "buy_count": 15, "sell_count": 0, "start_idx": 0, "end_idx": 14},
            ],
        }
        baseline = {"mean_loft": 0.3, "std_loft": 0.0}
        alerts = detect_toxicity_events(game_loft, baseline, sigma=2.0)
        assert len(alerts) == 0

    def test_empty_buckets_no_alerts(self):
        """Empty bucket list should produce no alerts."""
        game_loft = {"buckets": []}
        baseline = {"mean_loft": 0.5, "std_loft": 0.2}
        alerts = detect_toxicity_events(game_loft, baseline)
        assert len(alerts) == 0

    def test_multiple_alerts(self):
        """Multiple buckets breaching threshold should produce multiple alerts."""
        game_loft = {
            "buckets": [
                {"bucket_id": 0, "loft": 1.0, "rolling_loft": 1.0,
                 "buy_count": 15, "sell_count": 0, "start_idx": 0, "end_idx": 14},
                {"bucket_id": 1, "loft": 0.9, "rolling_loft": 0.95,
                 "buy_count": 14, "sell_count": 1, "start_idx": 15, "end_idx": 29},
                {"bucket_id": 2, "loft": 0.1, "rolling_loft": 0.2,
                 "buy_count": 5, "sell_count": 10, "start_idx": 30, "end_idx": 44},
            ],
        }
        baseline = {"mean_loft": 0.3, "std_loft": 0.1}
        alerts = detect_toxicity_events(game_loft, baseline, sigma=2.0)
        # Threshold = 0.3 + 2*0.1 = 0.5
        # Buckets 0 and 1 breach, bucket 2 does not
        assert len(alerts) == 2


# ── TestEdgeCases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_fewer_than_bucket_size_pitches(self):
        """A game with fewer than 15 pitches should still produce 1 bucket."""
        df = _make_pitch_df(10, description=["swinging_strike"] * 10,
                            type=["S"] * 10, zone=[5] * 10)
        classified = classify_pitch_flow(df)
        buckets = _build_buckets(classified, bucket_size=15)
        assert len(buckets) == 1

    def test_all_same_classification_buy(self, all_buy_df):
        """All buys: LOFT should be 1.0 for each bucket."""
        classified = classify_pitch_flow(all_buy_df)
        buckets = _build_buckets(classified, bucket_size=15)
        for _, row in buckets.iterrows():
            if row["total_flow"] > 0:
                assert abs(row["loft"] - 1.0) < 0.01

    def test_all_same_classification_sell(self, all_sell_df):
        """All sells: LOFT should be 1.0 for each bucket."""
        classified = classify_pitch_flow(all_sell_df)
        buckets = _build_buckets(classified, bucket_size=15)
        for _, row in buckets.iterrows():
            if row["total_flow"] > 0:
                assert abs(row["loft"] - 1.0) < 0.01

    def test_classify_preserves_original_columns(self):
        """classify_pitch_flow should not drop original columns."""
        df = _make_pitch_df(5)
        original_cols = set(df.columns)
        result = classify_pitch_flow(df)
        assert original_cols <= set(result.columns)

    def test_large_bucket_size(self):
        """Bucket size larger than total pitches should make 1 bucket."""
        df = _make_pitch_df(10, description=["ball"] * 10,
                            type=["B"] * 10, zone=[13] * 10)
        classified = classify_pitch_flow(df)
        buckets = _build_buckets(classified, bucket_size=100)
        assert len(buckets) == 1


# ── TestComputeGameLoft (with DB) ────────────────────────────────────────────


class TestComputeGameLoft:
    """Integration tests for compute_game_loft using the test database."""

    def test_returns_dict_with_required_keys(self, db_conn):
        """Result should always be a dict with expected keys."""
        # Get a game_pk that has Wheeler pitches
        result_df = db_conn.execute(
            "SELECT game_pk FROM pitches WHERE pitcher_id = $1 LIMIT 1",
            [ZACK_WHEELER_ID],
        ).fetchdf()
        if result_df.empty:
            pytest.skip("No Wheeler data in test DB")
        gpk = int(result_df["game_pk"].iloc[0])

        result = compute_game_loft(db_conn, gpk, ZACK_WHEELER_ID)
        required = {
            "pitcher_id", "game_pk", "total_pitches", "total_buckets",
            "buy_total", "sell_total", "neutral_total", "mean_loft",
            "max_loft", "final_rolling_loft", "buckets",
            "cumulative_buy", "cumulative_sell",
        }
        assert required <= set(result.keys())

    def test_unknown_game_returns_empty(self, db_conn):
        """A non-existent game_pk should return the empty sentinel."""
        result = compute_game_loft(db_conn, 999999999, ZACK_WHEELER_ID)
        assert result["total_buckets"] == 0

    def test_pitcher_id_preserved(self, db_conn):
        result = compute_game_loft(db_conn, 999999999, ZACK_WHEELER_ID)
        assert result["pitcher_id"] == ZACK_WHEELER_ID


# ── TestComputePitcherBaseline (with DB) ─────────────────────────────────────


class TestComputePitcherBaseline:
    """Integration tests for compute_pitcher_baseline."""

    def test_returns_dict(self, db_conn):
        baseline = compute_pitcher_baseline(db_conn, ZACK_WHEELER_ID, 2025)
        assert isinstance(baseline, dict)
        assert "pitcher_id" in baseline
        assert "mean_loft" in baseline
        assert "std_loft" in baseline

    def test_unknown_pitcher_empty(self, db_conn):
        baseline = compute_pitcher_baseline(db_conn, 999999999, 2025)
        assert baseline["n_games"] == 0


# ── TestModelLifecycle ───────────────────────────────────────────────────────


class TestModelLifecycle:
    """Tests for the LOFTModel BaseAnalyticsModel subclass."""

    def test_model_name(self):
        model = LOFTModel()
        assert model.model_name == "LOFT"

    def test_version(self):
        model = LOFTModel()
        assert model.version == "1.0.0"

    def test_train_is_noop(self, db_conn):
        model = LOFTModel()
        metrics = model.train(db_conn)
        assert metrics["status"] == "no_training_needed"

    def test_repr(self):
        model = LOFTModel()
        r = repr(model)
        assert "LOFT" in r
        assert "1.0.0" in r
