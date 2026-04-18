"""Unit tests for MechanixAE ROC / lead-time / FPR helpers.

Covers:
  - ROC computation on synthetic known data
  - Lead-time computation
  - Velocity-drop baseline detector
  - FPR computation
  - AUC bootstrap CI sanity

No DB access; pure numeric tests.
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

from scripts.mechanix_ae_roc_analysis import (  # noqa: E402
    bootstrap_auc_ci,
    compute_mdi_from_errors,
    daily_velocity_score,
    first_breach,
    per_day_mdi_series,
    roc_from_scores,
    velocity_drop_score,
)


# ── ROC ───────────────────────────────────────────────────────────────────


def test_roc_perfect_separator():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([10.0, 20.0, 30.0, 70.0, 80.0, 90.0])
    r = roc_from_scores(y, s)
    assert r["auc"] == pytest.approx(1.0, abs=1e-6)
    # At threshold 60 we get perfect TPR with zero FPR
    idx = r["thresholds"].index(60)
    assert r["tpr"][idx] == pytest.approx(1.0)
    assert r["fpr"][idx] == pytest.approx(0.0)


def test_roc_random_labels_around_half():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, size=400)
    s = rng.rand(400) * 100.0
    r = roc_from_scores(y, s)
    assert 0.35 < r["auc"] < 0.65


def test_roc_inverse_separator_is_flipped_auc():
    y = np.array([0, 0, 0, 1, 1, 1])
    # Higher score for negatives -> AUC should be 0
    s = np.array([90.0, 80.0, 70.0, 30.0, 20.0, 10.0])
    r = roc_from_scores(y, s)
    assert r["auc"] == pytest.approx(0.0, abs=1e-6)


def test_roc_handles_all_one_class():
    y = np.array([1, 1, 1])
    s = np.array([10.0, 20.0, 30.0])
    r = roc_from_scores(y, s)
    assert math.isnan(r["auc"])


def test_bootstrap_ci_is_bracket():
    rng = np.random.RandomState(1)
    y = rng.randint(0, 2, size=200)
    # scores correlated with y
    s = y * 50.0 + rng.rand(200) * 30.0
    point = roc_from_scores(y, s)["auc"]
    lo, hi = bootstrap_auc_ci(y, s, n_iter=200)
    assert lo <= point <= hi
    assert hi - lo > 0.0


# ── MDI / lead-time ───────────────────────────────────────────────────────


def test_compute_mdi_from_errors_monotonic():
    baseline = np.linspace(0.0, 1.0, 101)  # 101 evenly spaced
    errors = np.array([0.0, 0.5, 1.0, 2.0])
    mdi = compute_mdi_from_errors(errors, baseline)
    # error 0 -> roughly 1% rank (first bin), 0.5 -> ~50%, 1.0 -> ~100%, 2.0 -> 100%
    assert mdi[0] < mdi[1] < mdi[2]
    assert mdi[2] == pytest.approx(100.0, abs=1.0)
    assert mdi[3] == pytest.approx(100.0, abs=1e-6)


def test_per_day_mdi_aggregates_max():
    from scripts.mechanix_ae_roc_analysis import WINDOW_SIZE
    # W=20.  We need (WINDOW_SIZE + n - 1) "dates" so n errors align.
    n = 5
    dates = np.array([np.datetime64("2020-05-01")] * (WINDOW_SIZE - 1)
                     + [np.datetime64("2020-05-01")] * 3
                     + [np.datetime64("2020-05-02")] * (n - 3))
    assert len(dates) == WINDOW_SIZE - 1 + n
    errors = np.array([0.1, 0.5, 0.2, 0.9, 0.4])
    baseline = np.linspace(0.0, 1.0, 101)
    daily = per_day_mdi_series(dates, errors, baseline)
    assert list(daily["game_date"].dt.strftime("%Y-%m-%d")) == ["2020-05-01", "2020-05-02"]
    # Day 1 max error 0.5 -> ~50%; Day 2 max 0.9 -> ~90%
    assert daily.iloc[0]["mdi"] < daily.iloc[1]["mdi"]
    assert daily.iloc[1]["mdi"] >= 80.0


def test_first_breach_returns_days():
    daily = pd.DataFrame({
        "game_date": pd.to_datetime(["2020-05-01", "2020-05-05", "2020-05-10"]),
        "mdi": [50.0, 72.0, 85.0],
    })
    il = pd.Timestamp("2020-05-15")
    # threshold 70 first breached on 2020-05-05 -> 10 days lead
    assert first_breach(daily, 70, il) == pytest.approx(10.0)
    # threshold 80 breached on 2020-05-10 -> 5 days lead
    assert first_breach(daily, 80, il) == pytest.approx(5.0)
    # threshold 90 never breached
    assert first_breach(daily, 90, il) is None


# ── Velocity-drop baseline ────────────────────────────────────────────────


def test_velocity_drop_detects_sudden_drop():
    # 20 steady pitches at 95, then 10 pitches at 88 (big drop)
    speeds = np.array([95.0] * 20 + [88.0] * 10)
    # Need some variance so rolling std > 0
    rng = np.random.RandomState(0)
    speeds = speeds + rng.normal(0, 0.3, size=speeds.shape)
    scores = velocity_drop_score(speeds, window=10)
    # First 10 are zero (not enough history)
    assert scores[:10].sum() == 0
    # Drop region (indices 20+) should have a positive score (below rolling mean)
    assert scores[20:].max() > 1.0


def test_velocity_drop_no_drop_gives_low_scores():
    rng = np.random.RandomState(1)
    speeds = 95.0 + rng.normal(0, 0.5, size=60)
    scores = velocity_drop_score(speeds, window=10)
    # Mean score over latter half should be modest (< 2σ roughly)
    assert scores[20:].mean() < 1.0


def test_daily_velocity_score_groups_correctly():
    speeds = np.array([95.0] * 15 + [88.0] * 15)
    rng = np.random.RandomState(0)
    speeds = speeds + rng.normal(0, 0.3, size=speeds.shape)
    dates = np.array([np.datetime64("2020-05-01")] * 15
                     + [np.datetime64("2020-05-08")] * 15)
    daily = daily_velocity_score(dates, speeds, window=10)
    assert len(daily) == 2
    # Second day shows a drop
    assert daily.iloc[1]["vel_score"] > daily.iloc[0]["vel_score"]


# ── FPR computation ───────────────────────────────────────────────────────


def test_fpr_computation_matches_spec():
    # Synthetic: 10 healthy pitcher-seasons, 3 ever exceed MDI 80
    seasons = []
    for i in range(10):
        n_days = 30
        if i < 3:
            mdi = np.array([50.0] * 25 + [85.0] * 5)  # breaches 80
        elif i < 6:
            mdi = np.array([50.0] * 25 + [72.0] * 5)  # breaches 70 but not 80
        else:
            mdi = np.array([50.0] * n_days)  # never breaches
        seasons.append({
            "breached_80": bool((mdi >= 80).any()),
            "breached_70": bool((mdi >= 70).any()),
            "breached_60": bool((mdi >= 60).any()),
        })
    df = pd.DataFrame(seasons)
    assert df["breached_80"].mean() * 100 == pytest.approx(30.0)
    assert df["breached_70"].mean() * 100 == pytest.approx(60.0)
    assert df["breached_60"].mean() * 100 == pytest.approx(60.0)


# ── Smoke test for the synthetic end-to-end ROC shape ─────────────────────


def test_roc_with_partial_overlap_is_between_random_and_perfect():
    rng = np.random.RandomState(42)
    # Positives centered at 70, negatives at 40 — partial overlap
    pos = rng.normal(70, 15, size=200)
    neg = rng.normal(40, 15, size=600)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    r = roc_from_scores(y, s)
    assert 0.7 < r["auc"] < 0.98
