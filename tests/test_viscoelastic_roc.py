"""Unit tests for Viscoelastic Workload ROC / lead-time / seasonal-control helpers.

Pure numeric tests — no DB access.  Seeds fixed at 42.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.viscoelastic_workload_roc_analysis import (  # noqa: E402
    bootstrap_auc_ci,
    compute_daily_vwr,
    daily_velocity_score,
    fit_parameters_from_df,
    first_breach_day,
    partial_out_auc,
    roc_from_scores,
    season_day_from_date,
    velocity_drop_score,
)


SEED = 42


# ─── ROC ──────────────────────────────────────────────────────────────────


def test_roc_perfect_separator():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([10.0, 20.0, 30.0, 70.0, 80.0, 90.0])
    r = roc_from_scores(y, s)
    assert r["auc"] == pytest.approx(1.0, abs=1e-6)
    idx = r["thresholds"].index(60)
    assert r["tpr"][idx] == pytest.approx(1.0)
    assert r["fpr"][idx] == pytest.approx(0.0)


def test_roc_inverse_separator_is_zero_auc():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([90.0, 80.0, 70.0, 30.0, 20.0, 10.0])
    r = roc_from_scores(y, s)
    assert r["auc"] == pytest.approx(0.0, abs=1e-6)


def test_roc_random_labels_around_half():
    rng = np.random.RandomState(SEED)
    y = rng.randint(0, 2, size=400)
    s = rng.rand(400) * 100.0
    r = roc_from_scores(y, s)
    assert 0.35 < r["auc"] < 0.65


def test_roc_all_one_class_returns_nan():
    y = np.array([1, 1, 1])
    s = np.array([10.0, 20.0, 30.0])
    r = roc_from_scores(y, s)
    assert math.isnan(r["auc"])


def test_bootstrap_ci_brackets_perfect_separator():
    rng = np.random.RandomState(SEED)
    y = np.array([0] * 50 + [1] * 50)
    s = np.concatenate([rng.rand(50) * 30, 60 + rng.rand(50) * 30])
    lo, hi = bootstrap_auc_ci(y, s, n_iter=200)
    assert lo >= 0.9
    assert hi <= 1.0


# ─── Lead-time ────────────────────────────────────────────────────────────


def test_first_breach_returns_lead_time():
    daily = pd.DataFrame({
        "game_date": pd.to_datetime(["2015-05-01", "2015-05-10", "2015-05-20", "2015-06-01"]),
        "vwr": [40.0, 60.0, 88.0, 92.0],
    })
    il_date = pd.Timestamp("2015-06-15")
    # threshold 85 first breached on 2015-05-20 → 26 days lead
    assert first_breach_day(daily, 85.0, il_date) == 26.0
    # threshold 50 first breached on 2015-05-10 → 36 days lead
    assert first_breach_day(daily, 50.0, il_date) == 36.0


def test_first_breach_returns_none_when_never_breached():
    daily = pd.DataFrame({
        "game_date": pd.to_datetime(["2015-05-01", "2015-05-10"]),
        "vwr": [40.0, 50.0],
    })
    il_date = pd.Timestamp("2015-06-15")
    assert first_breach_day(daily, 85.0, il_date) is None


# ─── Velocity-drop baseline ───────────────────────────────────────────────


def test_velocity_drop_score_flags_drops():
    # Constant 95 mph until a sudden drop to 85 after pitch 15.
    speeds = np.array([95.0] * 15 + [85.0] * 5)
    score = velocity_drop_score(speeds, window=10)
    # Pre-drop: roll std ≈ 0 → score is 0 (NaN→0). Post-drop: should be > 0.
    assert score[-1] > 0
    assert score[5] == 0.0


def test_daily_velocity_score_aggregates_max_per_day():
    df = pd.DataFrame({
        "game_date": pd.to_datetime(["2015-05-01"] * 15 + ["2015-05-02"] * 10),
        "release_speed": [95.0] * 15 + [80.0] * 10,
    })
    daily = daily_velocity_score(df, window=10)
    assert len(daily) == 2
    # day 2 has a big drop relative to day 1's rolling mean
    assert daily.iloc[-1]["vel_score"] > 0


# ─── Seasonal-control ─────────────────────────────────────────────────────


def test_partial_out_auc_collapses_when_signal_is_season_day():
    """If the 'score' is just season_day noised a bit, residual AUC should be ~0.5."""
    rng = np.random.RandomState(SEED)
    n = 400
    season_day = rng.randint(60, 270, size=n).astype(float)
    # Labels correlated only with season_day (later in season → more likely injured)
    p = 1 / (1 + np.exp(-(season_day - 160) / 30))
    y = (rng.rand(n) < p).astype(int)
    # score = season_day + noise
    score = season_day + rng.normal(0, 2.0, size=n)
    out = partial_out_auc(y, score, season_day)
    # season_day itself should be predictive
    assert out["season_day_auc"] > 0.6
    # Residual AUC should collapse toward 0.5
    assert abs(out["residual_auc"] - 0.5) < 0.15


def test_partial_out_auc_holds_up_when_signal_is_orthogonal():
    """If score is orthogonal to season_day but predictive, residual should stay high."""
    rng = np.random.RandomState(SEED)
    n = 400
    season_day = rng.randint(60, 270, size=n).astype(float)
    y = rng.randint(0, 2, size=n)
    score = 2.0 * y + rng.normal(0, 0.5, size=n)  # strong signal independent of season_day
    out = partial_out_auc(y, score, season_day)
    assert out["residual_auc"] > 0.85


# ─── Season day ────────────────────────────────────────────────────────────


def test_season_day_from_date():
    assert season_day_from_date(pd.Timestamp("2015-01-01")) == 1
    assert season_day_from_date(pd.Timestamp("2015-04-01")) == 91  # non-leap


# ─── compute_daily_vwr & fit on synthetic frame ──────────────────────────


def _synthetic_pitcher_df(n_games=30, n_per_game=20, base_velo=93.0, seed=SEED):
    rng = np.random.RandomState(seed)
    rows = []
    gpk = 100
    for g in range(n_games):
        date = pd.Timestamp("2015-04-01") + pd.Timedelta(days=g * 4)
        gpk += 1
        for p in range(n_per_game):
            rows.append({
                "game_pk": gpk,
                "game_date": date,
                "pitch_type": rng.choice(["FF", "SL", "CH"], p=[0.6, 0.25, 0.15]),
                "release_speed": base_velo + rng.normal(0, 1.5),
                "release_pos_x": rng.normal(-1.5, 0.1),
                "release_pos_z": rng.normal(6.0, 0.1),
                "at_bat_number": p // 4,
                "pitch_number": p % 4 + 1,
            })
    return pd.DataFrame(rows)


def test_fit_parameters_insufficient_pitches_returns_defaults():
    df = _synthetic_pitcher_df(n_games=5, n_per_game=10)
    result = fit_parameters_from_df(df, pitcher_id=1)
    assert "reason" in result
    assert result["E1"] == 100.0  # default


def test_fit_parameters_converges_on_synthetic():
    df = _synthetic_pitcher_df(n_games=40, n_per_game=20)
    assert len(df) >= 500
    result = fit_parameters_from_df(df, pitcher_id=1)
    assert "reason" not in result, result.get("reason")
    assert result["converged"] is True
    assert 20.0 <= result["E1"] <= 500.0
    assert 10.0 <= result["E2"] <= 300.0
    assert 6.0 <= result["tau"] <= 168.0


def test_compute_daily_vwr_returns_0_100_scores():
    df = _synthetic_pitcher_df(n_games=40, n_per_game=20)
    params = {"E1": 100.0, "E2": 50.0, "tau": 48.0}
    # Build a "career strain distribution" from the same data
    from src.analytics.viscoelastic_workload import (
        compute_pitch_stress, compute_strain_state, _compute_time_deltas,
    )
    stress = compute_pitch_stress(df).values
    tds = _compute_time_deltas(df)
    career = compute_strain_state(stress, tds, **params)

    daily = compute_daily_vwr(df, params, career)
    assert len(daily) > 0
    assert (daily["vwr"] >= 0).all() and (daily["vwr"] <= 100).all()
