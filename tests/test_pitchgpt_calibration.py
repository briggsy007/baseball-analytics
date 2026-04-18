"""Tests for PitchGPT calibration utilities (Ticket #7)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analytics.pitchgpt_calibration import (
    apply_temperature,
    compute_reliability_curve,
    expected_calibration_error,
    temperature_scale,
)


# ── Reliability curve / ECE ─────────────────────────────────────────────


def test_ece_perfectly_calibrated_is_zero():
    """If empirical accuracy exactly matches predicted confidence in
    each bin, ECE should be ≈ 0.
    """
    # Construct a synthetic distribution: in every bin, correctness
    # matches the midpoint confidence of that bin.
    n_bins = 10
    n_per_bin = 1000
    rng = np.random.default_rng(42)

    top1 = []
    correct = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        # Sample confidences uniformly within the bin.
        p_in_bin = rng.uniform(lo, hi, size=n_per_bin)
        # For each sample, mark correct with probability == that sample's confidence.
        # This way mean_conf == mean_correctness in expectation.
        r = rng.uniform(0, 1, size=n_per_bin)
        c_in_bin = r < p_in_bin
        top1.append(p_in_bin)
        correct.append(c_in_bin)

    top1 = np.concatenate(top1)
    correct = np.concatenate(correct)

    curve = compute_reliability_curve(top1, correct, n_bins=n_bins)
    ece = expected_calibration_error(curve)
    # With 10K samples the per-bin gap is ~1/sqrt(1000) ≈ 0.03; ECE
    # averaged across bins will be much smaller.
    assert ece < 0.02, f"perfectly calibrated ECE should be near 0, got {ece:.4f}"


def test_ece_maximally_miscalibrated_matches_gap():
    """Predict 1.0 every time but be correct only 50% — ECE ≈ 0.5."""
    n = 2000
    top1 = np.full(n, 0.999)  # put in the top bin
    # Alternate correct/incorrect.
    correct = np.zeros(n, dtype=bool)
    correct[::2] = True  # exactly 50% correct

    curve = compute_reliability_curve(top1, correct, n_bins=10)
    ece = expected_calibration_error(curve)
    # Top bin's mean_conf ≈ 0.999, empirical ≈ 0.5 → gap ≈ 0.499, and
    # all samples land in that bin, so ECE ≈ 0.499.
    assert 0.45 <= ece <= 0.55, f"maximally miscalibrated ECE should ≈ 0.5, got {ece:.4f}"


def test_reliability_curve_empty_bins_zero_weight():
    """Bins with zero samples must contribute nothing to ECE."""
    n = 100
    top1 = np.full(n, 0.85)  # everything in bin 8
    correct = np.full(n, True)
    curve = compute_reliability_curve(top1, correct, n_bins=10)
    # 9 empty bins + 1 bin with 100 samples.
    empty = [b for b in curve if b["n_samples"] == 0]
    assert len(empty) == 9
    ece = expected_calibration_error(curve)
    # Top bin: mean_conf 0.85, emp_acc 1.0 → gap 0.15, weight 1.0.
    assert abs(ece - 0.15) < 1e-6


def test_reliability_curve_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_reliability_curve(np.zeros(10), np.zeros(5, dtype=bool))


def test_ece_empty_curve_returns_zero():
    assert expected_calibration_error([]) == 0.0


# ── Temperature scaling ─────────────────────────────────────────────────


def test_apply_temperature_scales_logits():
    logits = torch.tensor([[2.0, 0.0, -2.0]])
    out = apply_temperature(logits, T=2.0)
    torch.testing.assert_close(out, logits / 2.0)


def test_apply_temperature_rejects_nonpositive():
    with pytest.raises(ValueError):
        apply_temperature(torch.zeros(3), T=0.0)
    with pytest.raises(ValueError):
        apply_temperature(torch.zeros(3), T=-1.0)


def test_temperature_scaling_reduces_confidence_on_overconfident_model():
    """Construct logits that are *far* too spiky for the actual label
    distribution.  Temperature scaling should push T > 1 to widen them.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    V = 5
    N = 2000
    # True distribution: class 0 preferred but not by much (logit 1.0).
    # Emitted logits: class 0 preferred by a lot (logit 10.0) — i.e. the
    # model is *overconfident*.
    true_logits = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 0.0]), (N, 1))
    probs = np.exp(true_logits) / np.exp(true_logits).sum(axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    targets = np.array([rng.choice(V, p=probs[i]) for i in range(N)], dtype=np.int64)

    overconf_logits = np.tile(np.array([10.0, 0.0, 0.0, 0.0, 0.0]), (N, 1))

    T = temperature_scale(overconf_logits, targets, n_iters=200)
    assert T > 1.5, (
        f"overconfident model should need T > 1 to soften; got T={T:.3f}"
    )


def test_temperature_scaling_T_gt_0():
    """Temperature scaling should always return a strictly positive T."""
    torch.manual_seed(0)
    V = 3
    N = 100
    logits = np.random.randn(N, V).astype(np.float32)
    targets = np.random.randint(0, V, size=N, dtype=np.int64)
    T = temperature_scale(logits, targets, n_iters=30)
    assert T > 0


def test_temperature_scaling_well_calibrated_leaves_T_near_1():
    """If logits already match the label distribution, T should be ≈ 1."""
    torch.manual_seed(0)
    V = 4
    N = 4000
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 1.0, size=(1, V)).astype(np.float32).repeat(N, axis=0)
    probs = np.exp(true_logits) / np.exp(true_logits).sum(axis=1, keepdims=True)
    targets = np.array([rng.choice(V, p=probs[i]) for i in range(N)], dtype=np.int64)
    T = temperature_scale(true_logits, targets, n_iters=200)
    assert 0.75 < T < 1.3, f"well-calibrated model should give T near 1; got {T:.3f}"


def test_temperature_scaling_shape_validation():
    with pytest.raises(ValueError):
        temperature_scale(np.zeros((3,)), np.zeros((3,), dtype=np.int64))
    with pytest.raises(ValueError):
        temperature_scale(np.zeros((3, 4)), np.zeros((5,), dtype=np.int64))
