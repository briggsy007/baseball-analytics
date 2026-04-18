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
    compute_zscore_from_errors,
    daily_velocity_score,
    first_breach,
    per_day_mdi_series,
    roc_from_scores,
    run_pipeline,
    velocity_drop_score,
)
from src.analytics.mechanix_ae import (  # noqa: E402
    calculate_mdi,
    compute_mdi_zscore,
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


# ── Z-score MDI (magnitude-based scoring) ────────────────────────────────


def test_compute_zscore_from_errors_known_distribution():
    """Error equal to baseline mean -> z=0; 1σ above -> z≈1; 2σ above -> z≈2."""
    baseline = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    mu = baseline.mean()      # 3.0
    sigma = baseline.std()    # ≈1.4142
    errors = np.array([mu, mu + sigma, mu + 2 * sigma, mu - sigma])
    z = compute_zscore_from_errors(errors, baseline)
    assert z[0] == pytest.approx(0.0, abs=1e-6)
    assert z[1] == pytest.approx(1.0, abs=1e-6)
    assert z[2] == pytest.approx(2.0, abs=1e-6)
    assert z[3] == pytest.approx(-1.0, abs=1e-6)


def test_compute_zscore_empty_errors_returns_empty():
    z = compute_zscore_from_errors(np.array([]), np.array([1.0, 2.0, 3.0]))
    assert z.size == 0


def test_compute_zscore_zero_std_falls_back_to_eps():
    # Degenerate: all baseline values identical -> std=0 -> use eps
    baseline = np.ones(10)
    errors = np.array([1.0, 2.0])
    z = compute_zscore_from_errors(errors, baseline, eps=1e-6)
    # error=1.0 -> z = 0 / eps = 0; error=2.0 -> z = 1.0 / eps = 1e6
    assert z[0] == pytest.approx(0.0, abs=1e-6)
    assert z[1] > 1e5  # huge but finite


def test_compute_mdi_zscore_api_matches_script_helper():
    """The library-level helper and script-level helper should agree."""
    baseline = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    errors = np.array([0.3, 0.6, 0.9])
    z_script = compute_zscore_from_errors(errors, baseline)
    z_lib = compute_mdi_zscore(errors, float(baseline.mean()), float(baseline.std()))
    np.testing.assert_allclose(z_script, z_lib, atol=1e-9)


def test_per_day_mdi_zscore_mode_returns_magnitudes():
    from scripts.mechanix_ae_roc_analysis import WINDOW_SIZE
    n = 5
    dates = np.array([np.datetime64("2020-05-01")] * (WINDOW_SIZE - 1 + 3)
                     + [np.datetime64("2020-05-02")] * (n - 3))
    assert len(dates) == WINDOW_SIZE - 1 + n
    # Baseline mean=0.5, std≈0.29; drifted day-2 has error 2.0 -> very high z.
    errors = np.array([0.4, 0.6, 0.5, 2.0, 0.4])
    baseline = np.linspace(0.0, 1.0, 101)
    daily = per_day_mdi_series(dates, errors, baseline, score_mode="zscore")
    assert list(daily["game_date"].dt.strftime("%Y-%m-%d")) == ["2020-05-01", "2020-05-02"]
    # Day 2 max error of 2.0 is far above baseline mean/std -> z well above 3
    assert daily.iloc[1]["mdi"] > 3.0
    # Day 1 max error 0.6 is ~0.3σ above mean -> small positive
    assert abs(daily.iloc[0]["mdi"]) < 1.0


def test_per_day_mdi_percentile_mode_still_works():
    """Ensure legacy percentile path is unchanged."""
    from scripts.mechanix_ae_roc_analysis import WINDOW_SIZE
    n = 3
    dates = np.array([np.datetime64("2020-05-01")] * (WINDOW_SIZE - 1 + n))
    errors = np.array([0.1, 0.5, 0.9])
    baseline = np.linspace(0.0, 1.0, 101)
    daily_pct = per_day_mdi_series(dates, errors, baseline, score_mode="percentile")
    daily_z = per_day_mdi_series(dates, errors, baseline, score_mode="zscore")
    # Percentile top ~90, zscore for 0.9 vs baseline mean 0.5, std ≈ 0.29 -> z≈1.4
    assert daily_pct.iloc[0]["mdi"] > 80.0  # percentile saturates high
    assert 0.5 < daily_z.iloc[0]["mdi"] < 3.0  # zscore is bounded by magnitude


def test_roc_zscore_with_partial_overlap_gives_meaningful_auc():
    """Z-score mode must produce AUC well above random when signal present."""
    rng = np.random.RandomState(7)
    # Simulate injured pitchers having recon errors around z=2, healthy around z=0
    pos = rng.normal(2.0, 0.8, size=300)  # elevated drift
    neg = rng.normal(0.0, 1.0, size=900)  # healthy baseline
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    r = roc_from_scores(y, s, thresholds=[round(x, 2) for x in np.linspace(-2, 6, 21)])
    # Separation ~2σ should give AUC ~0.9+
    assert r["auc"] > 0.85


def test_calculate_mdi_accepts_score_mode_parameter():
    """API-level smoke test: calculate_mdi signature includes score_mode."""
    import inspect
    sig = inspect.signature(calculate_mdi)
    assert "score_mode" in sig.parameters
    # Default must be percentile (backwards compatibility).
    assert sig.parameters["score_mode"].default == "percentile"


def test_run_pipeline_accepts_score_mode_parameter():
    """API-level smoke test: run_pipeline signature includes score_mode."""
    import inspect
    sig = inspect.signature(run_pipeline)
    assert "score_mode" in sig.parameters


def test_run_pipeline_rejects_unknown_score_mode():
    with pytest.raises(ValueError, match="Unknown score_mode"):
        run_pipeline(score_mode="rank")
