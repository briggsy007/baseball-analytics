"""Unit tests for the literal Marcel implementation + WS4.4 scoring protocol.

Fixture expectations are HAND-COMPUTED from the tangotiger.net/marcel
formulas (5/4/3 weights, +1200 PA regression, age +/-0.006/0.003 around
29, PA projection 0.5*y1 + 0.1*y2 + 200); the derivations are inline.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.analytics.marcel import (
    MARCEL_REGRESSION_PA,
    MARCEL_WEIGHTS,
    MarcelProjection,
    age_asof_june30,
    head_to_head_vs_marcel,
    league_woba_by_season,
    marcel_batter_woba,
    pa_weighted_paired_t,
    pa_weighted_rmse,
    project_batters,
    score_forecast_vs_protocol,
)


# ---------------------------------------------------------------------------
# Core formula fixtures
# ---------------------------------------------------------------------------

def _std_projection(age):
    """PA (600, 500, 400), wOBA (.360, .340, .320), league (.320, .315, .310).

    Hand computation:
      wPA  = 5*600 + 4*500 + 3*400 = 6200
      rel  = 6200 / 7400            = 0.8378378378378378
      r    = (5*600*.360 + 4*500*.340 + 3*400*.320) / 6200
           = (1080 + 680 + 384) / 6200 = 2144/6200 = 0.34580645161290324
      lg   = (5*600*.320 + 4*500*.315 + 3*400*.310) / 6200
           = (960 + 630 + 372) / 6200 = 1962/6200 = 0.3164516129032258
      raw  = rel*r + (1-rel)*lg      = 0.34104620749782044
    """
    return marcel_batter_woba(
        player_id=1, season=2024,
        pa_by_lag=(600.0, 500.0, 400.0),
        woba_by_lag=(0.360, 0.340, 0.320),
        league_woba_by_lag=(0.320, 0.315, 0.310),
        age=age,
    )


def test_weights_and_regression_constants():
    assert MARCEL_WEIGHTS == (5.0, 4.0, 3.0)
    assert MARCEL_REGRESSION_PA == 1200.0


def test_reliability_and_rates_hand_computed():
    p = _std_projection(age=None)
    assert p.w_pa == pytest.approx(6200.0)
    assert p.reliability == pytest.approx(0.8378378378378378, abs=1e-12)
    assert p.weighted_player_rate == pytest.approx(0.34580645161290324, abs=1e-12)
    assert p.weighted_league_rate == pytest.approx(0.3164516129032258, abs=1e-12)
    assert p.proj_woba_raw == pytest.approx(0.34104620749782044, abs=1e-12)


def test_age_unknown_leaves_multiplier_one():
    p = _std_projection(age=None)
    assert p.age_known is False
    assert p.age_multiplier == 1.0
    assert p.proj_woba == pytest.approx(p.proj_woba_raw, abs=1e-15)


def test_age_27_young_side():
    # mult = 1 + 0.006*(29-27) = 1.012 -> 0.34104620749782044 * 1.012
    p = _std_projection(age=27.0)
    assert p.age_multiplier == pytest.approx(1.012, abs=1e-12)
    assert p.proj_woba == pytest.approx(0.3451387619877943, abs=1e-12)


def test_age_33_old_side():
    # mult = 1 + 0.003*(29-33) = 0.988 -> 0.34104620749782044 * 0.988
    p = _std_projection(age=33.0)
    assert p.age_multiplier == pytest.approx(0.988, abs=1e-12)
    assert p.proj_woba == pytest.approx(0.3369536530078466, abs=1e-12)


def test_age_exactly_29_no_adjustment():
    p = _std_projection(age=29.0)
    assert p.age_multiplier == pytest.approx(1.0, abs=1e-15)


def test_pa_projection_rule():
    # 0.5*600 + 0.1*500 + 200 = 550
    p = _std_projection(age=None)
    assert p.proj_pa == pytest.approx(550.0)


def test_partial_history_only_y2():
    """Only season y2 played: PA=350, wOBA .300; league (.320, .315, .310).

    wPA = 4*350 = 1400; rel = 1400/2600 = 0.5384615384615384
    r = .300; lg blend = .315 (all weight on y2)
    raw = rel*.300 + (1-rel)*.315 = 0.3069230769230769
    PA proj = 0.5*0 + 0.1*350 + 200 = 235
    """
    p = marcel_batter_woba(
        player_id=2, season=2024,
        pa_by_lag=(0.0, 350.0, 0.0),
        woba_by_lag=(None, 0.300, None),
        league_woba_by_lag=(0.320, 0.315, 0.310),
        age=None,
    )
    assert p.w_pa == pytest.approx(1400.0)
    assert p.reliability == pytest.approx(0.5384615384615384, abs=1e-12)
    assert p.proj_woba_raw == pytest.approx(0.3069230769230769, abs=1e-12)
    assert p.proj_pa == pytest.approx(235.0)


def test_no_history_projects_league_average():
    p = marcel_batter_woba(
        player_id=3, season=2024,
        pa_by_lag=(0.0, 0.0, 0.0),
        woba_by_lag=(None, None, None),
        league_woba_by_lag=(0.320, 0.315, 0.310),
        age=None,
    )
    assert p.reliability == 0.0
    assert p.weighted_player_rate is None
    assert p.proj_woba_raw == pytest.approx(0.320)  # most recent league rate
    assert p.proj_pa == pytest.approx(200.0)


def test_regression_pulls_toward_league():
    """A .400-wOBA hitter over 300 PA projects far below .400 (rel small)."""
    p = marcel_batter_woba(
        player_id=4, season=2024,
        pa_by_lag=(300.0, 0.0, 0.0),
        woba_by_lag=(0.400, None, None),
        league_woba_by_lag=(0.320, 0.315, 0.310),
        age=None,
    )
    # wPA = 1500, rel = 1500/2700 = 0.5555..; raw = rel*.400+(1-rel)*.320
    assert p.reliability == pytest.approx(1500.0 / 2700.0, abs=1e-12)
    expected = (1500.0 / 2700.0) * 0.400 + (1200.0 / 2700.0) * 0.320
    assert p.proj_woba_raw == pytest.approx(expected, abs=1e-12)
    assert p.proj_woba_raw < 0.365


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_age_asof_june30():
    # Born 1994-06-30 -> exactly 30.0 years on 2024-06-30 (leap-cycle avg).
    age = age_asof_june30(pd.Timestamp("1994-06-30"), 2024)
    assert age == pytest.approx(30.0, abs=0.01)
    assert age_asof_june30(None, 2024) is None
    assert age_asof_june30(pd.NaT, 2024) is None


def test_league_woba_by_season_pa_weighted():
    df = pd.DataFrame({
        "player_id": [1, 2, 3, 4],
        "season": [2023, 2023, 2022, 2022],
        "pa": [600.0, 200.0, 100.0, 300.0],
        "woba": [0.360, 0.280, 0.300, 0.340],
    })
    lg = league_woba_by_season(df)
    assert lg[2023] == pytest.approx((600 * .36 + 200 * .28) / 800.0, abs=1e-12)
    assert lg[2022] == pytest.approx((100 * .30 + 300 * .34) / 400.0, abs=1e-12)


def test_project_batters_end_to_end_synthetic():
    rows = []
    for season, pa, woba in ((2023, 600, .360), (2022, 500, .340), (2021, 400, .320)):
        rows.append({"player_id": 10, "season": season, "pa": float(pa), "woba": woba})
    # League context batters so league rates equal (.320, .315, .310):
    # add a big anchor batter per season pulling PA-weighted mean to target.
    for season, target in ((2023, .320), (2022, .315), (2021, .310)):
        me = [r for r in rows if r["season"] == season and r["player_id"] == 10]
        my_pa, my_woba = me[0]["pa"], me[0]["woba"]
        anchor_pa = 1_000_000.0
        anchor_woba = (target * (my_pa + anchor_pa) - my_woba * my_pa) / anchor_pa
        rows.append({"player_id": 99, "season": season, "pa": anchor_pa,
                     "woba": anchor_woba})
    df = pd.DataFrame(rows)
    out = project_batters(df, season=2024)
    me = out[out["player_id"] == 10].iloc[0]
    # League rates match the hand fixture to ~1e-9 -> same expected raw.
    assert me["proj_woba_raw"] == pytest.approx(0.34104620749782044, abs=1e-6)
    assert me["proj_pa"] == pytest.approx(550.0)
    assert bool(me["age_known"]) is False


def test_project_batters_requires_league_lags():
    df = pd.DataFrame({
        "player_id": [1], "season": [2023], "pa": [500.0], "woba": [0.330],
    })
    with pytest.raises(ValueError, match="lag seasons"):
        project_batters(df, season=2024)


# ---------------------------------------------------------------------------
# WS4.4 scoring protocol
# ---------------------------------------------------------------------------

def test_pa_weighted_rmse_hand_computed():
    actual = np.array([0.350, 0.300])
    pred = np.array([0.340, 0.320])
    pa = np.array([600.0, 200.0])
    # sum(pa*err^2) = 600*1e-4 + 200*4e-4 = 0.06 + 0.08 = 0.14; /800 = 1.75e-4
    assert pa_weighted_rmse(actual, pred, pa) == pytest.approx(
        math.sqrt(1.75e-4), abs=1e-12
    )


def test_head_to_head_tie_band():
    actual = np.array([0.300, 0.300, 0.300])
    system = np.array([0.305, 0.330, 0.300])   # |err| .005, .030, .000
    marcel = np.array([0.320, 0.305, 0.309])   # |err| .020, .005, .009
    # gaps m-s: +.015 win, -.025 loss, +.009 tie (<= .010)
    h = head_to_head_vs_marcel(system, marcel, actual)
    assert h == {"wins": 1, "losses": 1, "ties": 1, "n": 3}


def test_paired_t_direction():
    rng = np.random.RandomState(0)
    actual = rng.uniform(0.28, 0.38, size=200)
    system = actual + rng.normal(0, 0.005, size=200)   # much better
    marcel = actual + rng.normal(0, 0.030, size=200)
    pa = np.full(200, 400.0)
    t = pa_weighted_paired_t(system, marcel, actual, pa)
    assert t["mean_diff"] > 0
    assert t["confidence_system_better"] > 0.99


def test_superiority_blocked_on_single_season():
    """Even a dominant single season cannot ground a superiority claim."""
    rng = np.random.RandomState(1)
    actual = rng.uniform(0.28, 0.38, size=100)
    system = actual + rng.normal(0, 0.004, size=100)
    marcel = actual + rng.normal(0, 0.030, size=100)
    naive = np.full(100, 0.315)
    pa = np.full(100, 500.0)
    v = score_forecast_vs_protocol(
        system_name="test_sys", actual_woba=actual, system_pred=system,
        marcel_pred=marcel, naive_pred=naive, followup_pa=pa,
        seasons_evaluated=1,
    )
    assert v["superiority_claim_allowed"] is False
    assert any("resolved season" in r for r in v["superiority_blocked_because"])


def test_superiority_allowed_two_seasons_strong():
    rng = np.random.RandomState(2)
    actual = rng.uniform(0.28, 0.38, size=200)
    system = actual + rng.normal(0, 0.004, size=200)
    marcel = actual + rng.normal(0, 0.030, size=200)
    naive = np.full(200, 0.315)
    pa = np.full(200, 500.0)
    v = score_forecast_vs_protocol(
        system_name="test_sys", actual_woba=actual, system_pred=system,
        marcel_pred=marcel, naive_pred=naive, followup_pa=pa,
        seasons_evaluated=2,
    )
    assert v["superiority_claim_allowed"] is True
    assert v["rmse_system"] < v["rmse_marcel"]
