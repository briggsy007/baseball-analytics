"""Smoke tests for Plan B outcome-prediction baselines (A3 / A4 / A5).

These are import-and-shape tests; they verify the new modules load,
expose the documented public APIs, and that the metric helpers work
correctly on small synthetic data.  They do NOT exercise the full
training loop (that needs DuckDB + multi-minute wall-clock).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_common_module_imports():
    """The shared cohort-builder + metrics module must import cleanly."""
    from scripts import pitchgpt_outcome_baselines_common as common  # noqa: F401

    # Sanity-check the public surface.
    for name in (
        "build_cohort", "fetch_features_for_games", "fetch_pitcher_ids",
        "sample_game_pks", "log_loss_from_probs", "ece_10bin",
        "fit_temperature", "softmax", "bootstrap_log_loss",
        "per_class_log_loss", "top1_accuracy", "freq_prior_log_loss_on",
        "class_frequency_prior", "per_pitcher_log_loss",
        "FEATURE_COLUMNS_NUMERIC", "FEATURE_COLUMNS_CATEGORICAL",
        "TRAIN_SEASONS", "VAL_SEASONS", "TEST_SEASONS",
        "DEFAULT_TRAIN_GAMES", "DEFAULT_SEED",
    ):
        assert hasattr(common, name), (
            f"pitchgpt_outcome_baselines_common is missing public attribute "
            f"{name!r}"
        )


def test_a3_xgboost_module_imports():
    from scripts import pitchgpt_outcome_a3_xgboost as a3  # noqa: F401

    assert callable(a3.main)
    assert callable(a3.train_xgb)
    assert callable(a3.make_xgb_features)


def test_a4_logistic_module_imports():
    from scripts import pitchgpt_outcome_a4_logistic as a4  # noqa: F401

    assert callable(a4.main)
    assert callable(a4.make_features)
    assert hasattr(a4, "TargetEncoder7Class")


def test_a5_empirical_module_imports():
    from scripts import pitchgpt_outcome_a5_empirical as a5  # noqa: F401

    assert callable(a5.main)
    assert callable(a5.build_lookup)
    assert callable(a5.predict_fast)


def test_metrics_log_loss_correctness():
    """Log-loss helper matches the textbook formula."""
    from scripts.pitchgpt_outcome_baselines_common import log_loss_from_probs

    # Three rows, 7 classes, perfect prediction → 0 nats.
    n_classes = 7
    probs = np.eye(n_classes)
    y = np.arange(n_classes)
    ll = log_loss_from_probs(probs, y)
    assert ll < 1e-5, f"perfect prediction should yield 0 nats, got {ll}"

    # Uniform prediction → log(7) ≈ 1.9459.
    probs_uniform = np.full((n_classes, n_classes), 1.0 / n_classes)
    ll = log_loss_from_probs(probs_uniform, y)
    assert abs(ll - np.log(n_classes)) < 1e-5


def test_metrics_softmax_correctness():
    """Softmax helper preserves shape and sums to 1."""
    from scripts.pitchgpt_outcome_baselines_common import softmax

    z = np.array([[1.0, 2.0, 3.0, 0.0, -1.0, 0.5, 0.5]])
    s = softmax(z)
    assert s.shape == z.shape
    assert abs(s.sum() - 1.0) < 1e-9


def test_metrics_ece_correctness():
    """ECE is 0 when confidence equals empirical accuracy in every bin."""
    from scripts.pitchgpt_outcome_baselines_common import ece_10bin

    # Construct logits that put 1.0 prob on the correct class for every
    # row → top1_p=1.0, correct=1.0 → ECE=0.
    n = 100
    rng = np.random.default_rng(42)
    y = rng.integers(0, 7, size=n)
    probs = np.zeros((n, 7))
    probs[np.arange(n), y] = 1.0
    ece = ece_10bin(probs, y, n_bins=10)
    assert ece < 1e-5, f"perfect calibration should give ECE=0, got {ece}"


def test_metrics_temperature_scale_returns_positive_scalar():
    """Temperature fit returns a positive scalar (basic sanity).

    Tighter ranges depend on the calibration of the input logits;
    we don't synthesise calibrated logits here, so we just assert the
    optimiser returns a finite, positive number within the bounds we
    pass to scipy.optimize.minimize_scalar.
    """
    from scripts.pitchgpt_outcome_baselines_common import fit_temperature

    rng = np.random.default_rng(42)
    n = 1000
    y = rng.integers(0, 7, size=n)
    logits = rng.normal(scale=0.3, size=(n, 7))
    logits[np.arange(n), y] += 1.5
    T = fit_temperature(logits, y)
    assert np.isfinite(T)
    assert T > 0
    assert T <= 10.0  # upper bound passed to scipy


def test_metrics_freq_prior_log_loss_correctness():
    """Frequency prior log-loss equals the entropy of the prior on its
    own labels (when targets are sampled from prior)."""
    from scripts.pitchgpt_outcome_baselines_common import (
        class_frequency_prior, freq_prior_log_loss_on,
    )

    rng = np.random.default_rng(42)
    n = 10_000
    # sample y from a non-uniform 7-class distribution.
    p_true = np.array([0.35, 0.16, 0.10, 0.19, 0.11, 0.06, 0.03])
    y = rng.choice(7, size=n, p=p_true)
    prior = class_frequency_prior(y)  # empirical
    ll = freq_prior_log_loss_on(y, prior)
    # The log-loss equals -sum(p_emp * log p_emp) within sampling noise.
    expected = float(-(prior * np.log(np.clip(prior, 1e-12, 1.0))).sum())
    assert abs(ll - expected) < 1e-3


def test_a4_target_encoder_basic():
    """TargetEncoder7Class fits per-bucket distribution and falls back
    to global prior on unseen values."""
    from scripts.pitchgpt_outcome_a4_logistic import TargetEncoder7Class

    enc = TargetEncoder7Class(alpha=1.0)
    # Three pitchers, 7-class outcomes.
    vals = np.array([1, 1, 1, 2, 2, 3])
    y = np.array([0, 0, 0, 3, 3, 5])
    enc.fit(vals, y)

    # Pitcher 1: all class 0 (with smoothing); pitcher 3: all class 5.
    out = enc.transform(np.array([1, 3, 999]))
    assert out.shape == (3, 7)
    # pitcher 1 should mass-weight class 0 highly.
    assert out[0, 0] > out[0, 5]
    # pitcher 3 should mass-weight class 5 highly.
    assert out[1, 5] > out[1, 0]
    # unseen pitcher 999 → global prior (sums to 1).
    assert abs(out[2].sum() - 1.0) < 1e-9


def test_a5_lookup_global_fallback():
    """When all bucket levels are empty, the global prior should fire
    on every test row."""
    import pandas as pd

    from scripts.pitchgpt_outcome_a5_empirical import build_lookup, predict_fast

    df_train = pd.DataFrame({
        "pitcher_id": [1, 1, 2],
        "balls": [0, 0, 1],
        "strikes": [0, 0, 1],
        "pitch_type": ["FF", "FF", "CU"],
        "stand": ["L", "L", "R"],
        "outcome_label": [0, 0, 3],
    })
    levels = [
        ["pitcher_id", "balls", "strikes", "pitch_type", "stand"],
        [],
    ]
    lookups = build_lookup(df_train, levels, alpha=1.0)
    global_prior = np.array([0.5, 0, 0, 0.5, 0, 0, 0])  # arbitrary

    df_test = pd.DataFrame({
        "pitcher_id": [99],
        "balls": [3],
        "strikes": [2],
        "pitch_type": ["SL"],
        "stand": ["R"],
    })
    probs, depth = predict_fast(df_test, lookups, levels, global_prior)
    # Should fall through to global level (level 1).
    assert probs.shape == (1, 7)
    assert depth[0] == 1
    assert abs(probs[0].sum() - 1.0) < 1e-9


def test_classify_pitch_outcome_locked_class_set():
    """The 7-class label rules from Phase 0.3 are still locked - the
    common module imports them from src.analytics.pitchgpt_outcome_head
    so any drift here would be caught."""
    from src.analytics.pitchgpt_outcome_head import (
        NUM_OUTCOME_CLASSES, OUTCOME_CLASSES, classify_pitch_outcome,
    )

    assert NUM_OUTCOME_CLASSES == 7
    assert OUTCOME_CLASSES == (
        "ball", "called_strike", "swinging_strike", "foul",
        "in_play_out", "in_play_hit", "hbp",
    )

    # Spot-check rule application.
    assert classify_pitch_outcome("ball", None) == 0  # ball
    assert classify_pitch_outcome("called_strike", None) == 1
    assert classify_pitch_outcome("swinging_strike", None) == 2
    assert classify_pitch_outcome("foul", None) == 3
    assert classify_pitch_outcome("hit_into_play", "field_out") == 4
    assert classify_pitch_outcome("hit_into_play", "single") == 5
    assert classify_pitch_outcome("hit_by_pitch", None) == 6
