"""
PitchGPT calibration utilities — Ticket #7.

Produces a reliability diagram, Expected Calibration Error (ECE), and a
fitted temperature-scaling scalar for a trained PitchGPT checkpoint.

Calibration is computed on *per-pitch* next-token predictions: for every
(input, target) pair in the supplied dataset, we take the predicted
probability distribution over the ``VOCAB_SIZE`` token space, pick the
top-1 probability as the "confidence", and compare it to the empirical
frequency with which the top-1 prediction is correct.

Public API
----------
- ``gather_predictions``           -- run a model over a dataset and return
                                      (top1_prob, is_correct) arrays.
- ``compute_reliability_curve``    -- bin predictions and compute
                                      per-bin (mean_conf, empirical_acc).
- ``expected_calibration_error``   -- ECE from a reliability curve.
- ``temperature_scale``            -- fit scalar T maximising log-
                                      likelihood on a val set (LBFGS).
- ``apply_temperature``            -- return ``logits / T``.

Notes
-----
- Calibration is measured on the *argmax* prediction only (top-1
  reliability), which is standard for multi-class classifiers.  This is
  what ECE is typically defined against.
- Temperature scaling is a single-parameter post-hoc fix; it does not
  retrain the model and is numerically stable.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.analytics.pitchgpt import (
    PAD_TOKEN,
    VOCAB_SIZE,
    PitchGPTModel,
    _collate_fn,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# Inference — collect per-pitch predictions
# ═════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def gather_predictions(
    model: PitchGPTModel,
    dataset: Dataset,
    batch_size: int = 32,
    device: torch.device | None = None,
    return_logits: bool = False,
) -> dict:
    """Run ``model`` over ``dataset`` and return per-pitch predictions.

    Returns a dict with:
      * ``top1_prob`` : np.ndarray[float], shape (N,)
      * ``is_correct``: np.ndarray[bool],  shape (N,)
      * ``target``    : np.ndarray[int64], shape (N,)
      * ``target_log_prob``: np.ndarray[float], shape (N,) — log p(target)
      * ``logits`` (optional): np.ndarray[float], shape (N, VOCAB_SIZE)
        — only populated when ``return_logits=True`` (memory-hungry).

    Only non-PAD tokens are included.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn,
    )

    top1_probs: list[np.ndarray] = []
    is_correct: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    tgt_logp: list[np.ndarray] = []
    logits_chunks: list[np.ndarray] = []

    # Match the model's context schema: v1 models have context_dim=34
    # while datasets default to 35.  Slice the tail (ump scalar) if
    # needed so v1 can be scored against a 35-dim dataset.
    from src.analytics.pitchgpt import CONTEXT_DIM as _DATASET_CTX_DIM
    m_ctx_dim = getattr(model, "context_dim", _DATASET_CTX_DIM)

    for tokens, ctx, target in loader:
        tokens = tokens.to(device)
        ctx = ctx.to(device)
        target = target.to(device)
        if ctx.size(-1) > m_ctx_dim:
            ctx = ctx[..., :m_ctx_dim]

        logits = model(tokens, ctx)  # (B, S, V)
        log_probs = F.log_softmax(logits, dim=-1)

        # Mask out pad positions (target == PAD_TOKEN)
        flat_logits = logits.reshape(-1, VOCAB_SIZE)
        flat_logp = log_probs.reshape(-1, VOCAB_SIZE)
        flat_target = target.reshape(-1)
        valid = flat_target != PAD_TOKEN
        flat_logits = flat_logits[valid]
        flat_logp = flat_logp[valid]
        flat_target = flat_target[valid]

        if flat_target.numel() == 0:
            continue

        probs = flat_logp.exp()  # (N, V) — numerically OK since logp is bounded.
        top1_p, top1_idx = probs.max(dim=-1)
        correct = (top1_idx == flat_target)
        t_logp = flat_logp.gather(1, flat_target.unsqueeze(1)).squeeze(1)

        top1_probs.append(top1_p.cpu().numpy())
        is_correct.append(correct.cpu().numpy())
        targets.append(flat_target.cpu().numpy())
        tgt_logp.append(t_logp.cpu().numpy())
        if return_logits:
            logits_chunks.append(flat_logits.cpu().numpy())

    result = {
        "top1_prob": np.concatenate(top1_probs) if top1_probs else np.array([]),
        "is_correct": np.concatenate(is_correct) if is_correct else np.array([], dtype=bool),
        "target": np.concatenate(targets) if targets else np.array([], dtype=np.int64),
        "target_log_prob": np.concatenate(tgt_logp) if tgt_logp else np.array([]),
    }
    if return_logits:
        result["logits"] = np.concatenate(logits_chunks) if logits_chunks else np.zeros((0, VOCAB_SIZE))
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Reliability curve + ECE
# ═════════════════════════════════════════════════════════════════════════════


def compute_reliability_curve(
    top1_prob: np.ndarray,
    is_correct: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Bin predictions by top-1 confidence and return the reliability curve.

    Each bin covers ``[b/n_bins, (b+1)/n_bins)``; the final bin is
    closed on the right to include probability 1.0.

    Args:
        top1_prob: (N,) float array — top-1 predicted probability.
        is_correct: (N,) bool array — whether the top-1 was the target.
        n_bins: number of equal-width bins over [0, 1].

    Returns:
        List of dicts, one per bin::

            {"bin": i,
             "lo": float,
             "hi": float,
             "mean_conf": float,
             "empirical_acc": float,
             "n_samples": int}
    """
    if top1_prob.shape != is_correct.shape:
        raise ValueError(
            f"top1_prob and is_correct must have the same shape; "
            f"got {top1_prob.shape} and {is_correct.shape}"
        )
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    curve: list[dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (top1_prob >= lo) & (top1_prob <= hi)
        else:
            mask = (top1_prob >= lo) & (top1_prob < hi)
        n = int(mask.sum())
        if n == 0:
            mean_conf = float((lo + hi) / 2)
            emp_acc = 0.0
        else:
            mean_conf = float(top1_prob[mask].mean())
            emp_acc = float(is_correct[mask].mean())
        curve.append({
            "bin": i,
            "lo": float(lo),
            "hi": float(hi),
            "mean_conf": mean_conf,
            "empirical_acc": emp_acc,
            "n_samples": n,
        })
    return curve


def expected_calibration_error(reliability_curve: Iterable[dict]) -> float:
    """Expected Calibration Error.

    ``ECE = sum_b (n_b / n_total) * |mean_conf_b - empirical_acc_b|``

    Bins with zero samples contribute zero to the sum.
    """
    bins = list(reliability_curve)
    n_total = sum(b["n_samples"] for b in bins)
    if n_total == 0:
        return 0.0
    ece = 0.0
    for b in bins:
        if b["n_samples"] == 0:
            continue
        w = b["n_samples"] / n_total
        ece += w * abs(b["mean_conf"] - b["empirical_acc"])
    return float(ece)


# ═════════════════════════════════════════════════════════════════════════════
# Temperature scaling
# ═════════════════════════════════════════════════════════════════════════════


def apply_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    """Return ``logits / T``.  ``T > 1`` softens (reduces confidence)."""
    if T <= 0:
        raise ValueError(f"Temperature must be > 0; got {T}")
    return logits / float(T)


def temperature_scale(
    logits: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    n_iters: int = 100,
    lr: float = 0.01,
) -> float:
    """Fit a single temperature ``T`` that minimises NLL on
    ``softmax(logits / T)`` with respect to ``targets``.

    Uses LBFGS for stability (convex problem, converges in a few steps).

    Args:
        logits: (N, V) raw pre-softmax scores.
        targets: (N,) integer class labels in ``[0, V)``.
        n_iters: max LBFGS iterations.
        lr: optimiser learning rate.

    Returns:
        Optimal temperature (float).  Returned T is always > 0.
    """
    if isinstance(logits, np.ndarray):
        logits_t = torch.from_numpy(logits).float()
    else:
        logits_t = logits.detach().float()
    if isinstance(targets, np.ndarray):
        targets_t = torch.from_numpy(targets).long()
    else:
        targets_t = targets.detach().long()

    if logits_t.dim() != 2:
        raise ValueError(f"logits must be 2-D (N, V); got shape {tuple(logits_t.shape)}")
    if targets_t.dim() != 1 or targets_t.shape[0] != logits_t.shape[0]:
        raise ValueError(
            f"targets must be 1-D with shape[0] matching logits; "
            f"got {tuple(targets_t.shape)} vs {tuple(logits_t.shape)}"
        )

    # Parameterise T = softplus(raw) + eps so T stays > 0 without
    # needing a hard constraint.  Initialise raw so T ≈ 1.
    raw = torch.zeros(1, requires_grad=True)  # softplus(0) ≈ 0.6931
    # Shift so initial T ≈ 1.0 (softplus(raw_init) + eps = 1 → raw_init = log(e^{1-eps} - 1))
    eps = 1e-3
    with torch.no_grad():
        raw.fill_(float(np.log(np.exp(1.0 - eps) - 1.0)))

    nll = nn.CrossEntropyLoss()
    opt = torch.optim.LBFGS([raw], lr=lr, max_iter=n_iters, line_search_fn="strong_wolfe")

    def _closure():
        opt.zero_grad()
        T = F.softplus(raw) + eps
        loss = nll(logits_t / T, targets_t)
        loss.backward()
        return loss

    opt.step(_closure)
    T_final = float((F.softplus(raw) + eps).item())
    return T_final


# ═════════════════════════════════════════════════════════════════════════════
# Convenience: full calibration audit
# ═════════════════════════════════════════════════════════════════════════════


def full_calibration_audit(
    model: PitchGPTModel,
    val_dataset: Dataset,
    test_dataset: Dataset,
    n_bins: int = 10,
    batch_size: int = 32,
    device: torch.device | None = None,
) -> dict:
    """End-to-end audit: fit T on val logits, evaluate ECE pre/post on
    test logits.  Returns a single dict ready to serialise.
    """
    # 1. Test-set predictions (with logits for potential rescaling).
    test_preds = gather_predictions(
        model, test_dataset, batch_size=batch_size, device=device,
        return_logits=True,
    )
    n_test = int(test_preds["top1_prob"].shape[0])

    pre_curve = compute_reliability_curve(
        test_preds["top1_prob"], test_preds["is_correct"], n_bins=n_bins,
    )
    ece_pre = expected_calibration_error(pre_curve)

    # 2. Val-set logits → fit T.
    val_preds = gather_predictions(
        model, val_dataset, batch_size=batch_size, device=device,
        return_logits=True,
    )
    if val_preds["logits"].shape[0] == 0:
        logger.warning("Empty val set — temperature scaling skipped, T=1.")
        T_opt = 1.0
    else:
        T_opt = temperature_scale(
            val_preds["logits"], val_preds["target"],
        )

    # 3. Apply T to test logits and recompute.
    if n_test > 0 and abs(T_opt - 1.0) > 1e-6:
        scaled_logits = test_preds["logits"] / T_opt
        scaled_probs = np.exp(
            scaled_logits - scaled_logits.max(axis=1, keepdims=True)
        )
        scaled_probs /= scaled_probs.sum(axis=1, keepdims=True)
        top1_idx = scaled_probs.argmax(axis=1)
        top1_prob_post = scaled_probs.max(axis=1)
        is_correct_post = (top1_idx == test_preds["target"])
    else:
        top1_prob_post = test_preds["top1_prob"]
        is_correct_post = test_preds["is_correct"]

    post_curve = compute_reliability_curve(
        top1_prob_post, is_correct_post, n_bins=n_bins,
    )
    ece_post = expected_calibration_error(post_curve)

    # Accuracy is unchanged by temperature scaling (argmax is invariant
    # under monotonic rescaling), but we record it.
    accuracy = float(test_preds["is_correct"].mean()) if n_test > 0 else 0.0

    return {
        "n_test_tokens": n_test,
        "n_val_tokens": int(val_preds["top1_prob"].shape[0]),
        "accuracy": accuracy,
        "ece_pre_temp": float(ece_pre),
        "ece_post_temp": float(ece_post),
        "optimal_temperature": float(T_opt),
        "reliability_curve_pre": pre_curve,
        "reliability_curve_post": post_curve,
        "n_bins": int(n_bins),
    }
