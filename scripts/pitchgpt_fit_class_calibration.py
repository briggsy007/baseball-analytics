"""
Fit per-class probability re-weighting for A1 concat outcome head.

Phase 0.6 fix (2026-04-26): A1 was trained with inverse-frequency CE
class weights (cap=10), which pulled marginal mass out of `ball` and
into `swinging_strike` + `in_play_out`.  PHASE_0.6_DIAGNOSIS.md
documents the bias (-11.63pp on `ball`, +8.63pp on `swinging_strike`).
Temperature scaling (a scalar) cannot redistribute mass between
classes; this script fits a 7-vector ``class_calibration`` ``w`` such
that the post-T softmax is reweighted ``p_i -> p_i * w_i / sum_j(p_j *
w_j)``.

The fit minimizes ``KL(rebalanced_marginal || empirical_marginal)`` on
the 2023 pitcher-disjoint validation cohort -- the SAME split that A1
used to fit temperature.  We do NOT touch 2025 (test holdout) for
fitting.

Closed-form solution
--------------------
Let ``q_c = E[p_c]`` be A1's predicted class marginal (mean of post-T
softmax probs over val pitches), and ``p_c`` be the empirical
frequency.  Define ``w_c proportional to p_c / q_c``; we further
renormalize ``w`` so that ``geometric_mean(w) = 1`` (a convenience
that keeps numerical scale tame -- the renormalization step in
``predict_outcome_probs`` makes the absolute scale of ``w``
irrelevant).

This closed form is exact only when the per-pitch softmax is constant
(degenerate).  For real per-pitch heterogeneity the closed form is a
strong first approximation.  We verify post-fit that the rebalanced
marginal is within 1pp of empirical for every class.  If not, we
fall through to L-BFGS minimizing the KL divergence directly.

Outputs
-------
* Updates ``models/calibration_pitchgpt_v2_outcomehead_a1.json`` with
  a new field ``class_calibration``: list[float] length 7.
* Bumps ``fit_date`` to ``2026-04-26`` (current date).

Guardrails
----------
* DuckDB read-only.  No checkpoint mutation.  No changes to
  ``pitchgpt_v2.pt`` or ``pitchgpt_v2_outcomehead_a1.pt`` (SHA256
  asserted byte-identical pre/post).
* Reuses the FixedGamesSequenceDataset / collect_logits machinery in
  ``scripts.pitchgpt_outcome_a1_concat`` so the cohort matches A1
  training byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt import (  # noqa: E402
    NUM_PITCH_TYPES,
    NUM_VELO_BUCKETS,
    NUM_ZONES,
    _get_device,
    _load_model,
)
from src.analytics.pitchgpt_outcome_head import (  # noqa: E402
    NUM_OUTCOME_CLASSES,
    OUTCOME_CLASSES,
    OUTCOME_UNK,
    FrozenOutcomeHeadConcat,
    extract_backbone_hidden_states,
    freeze_backbone,
)
from scripts.pitchgpt_outcome_a1_concat import (  # noqa: E402
    A1_CKPT,
    BACKBONE_PATH,
    DEFAULT_BATCH,
    FixedGamesSequenceDataset,
    _a1_collate,
    _backbone_param_checksum,
    _sha256_file,
)
from scripts.pitchgpt_outcome_baselines_common import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_VAL_GAMES,
    TRAIN_SEASONS,
    VAL_SEASONS,
    fetch_pitcher_ids,
    sample_game_pks,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fit_class_calibration")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CALIB_PATH = (
    _ROOT / "models" / "calibration_pitchgpt_v2_outcomehead_a1.json"
)
FIT_DATE = "2026-04-26"
WITHIN_BAND_PP = 0.01  # 1pp closed-form acceptance band.


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _collect_post_T_probs(
    backbone,
    head: FrozenOutcomeHeadConcat,
    ds: FixedGamesSequenceDataset,
    temperature: float,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run A1 forward + softmax(logits / T) per valid pitch position.

    Returns (probs (N, 7), targets (N,)).  Mirrors the A1 training
    script's ``collect_logits`` but skips row-key bookkeeping (we only
    need probs / labels for marginal calibration).
    """
    from torch.utils.data import DataLoader

    backbone.eval()
    head.eval()
    backbone.to(device)
    head.to(device)

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=_a1_collate,
    )
    probs_chunks: list[np.ndarray] = []
    targets_chunks: list[np.ndarray] = []

    for batch in loader:
        (tokens, ctx, _tgt, outcomes,
         pt_idx, zn_idx, vl_idx, _rk_list) = batch
        tokens = tokens.to(device)
        ctx = ctx.to(device)
        outcomes = outcomes.to(device)
        pt_idx = pt_idx.to(device)
        zn_idx = zn_idx.to(device)
        vl_idx = vl_idx.to(device)

        hidden = extract_backbone_hidden_states(backbone, tokens, ctx)
        pt_oh = F.one_hot(pt_idx, num_classes=NUM_PITCH_TYPES).float()
        zn_oh = F.one_hot(zn_idx, num_classes=NUM_ZONES).float()
        vl_oh = F.one_hot(vl_idx, num_classes=NUM_VELO_BUCKETS).float()
        logits = head(hidden, ctx, pt_oh, zn_oh, vl_oh)  # (B, S, 7)
        scaled = logits / max(float(temperature), 1e-8)
        probs = F.softmax(scaled, dim=-1)

        out_flat = outcomes.reshape(-1)
        prob_flat = probs.reshape(-1, NUM_OUTCOME_CLASSES)
        valid = out_flat != OUTCOME_UNK
        if int(valid.sum()) == 0:
            continue
        probs_chunks.append(prob_flat[valid].cpu().numpy())
        targets_chunks.append(out_flat[valid].cpu().numpy())

    if not probs_chunks:
        return (
            np.zeros((0, NUM_OUTCOME_CLASSES), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.concatenate(probs_chunks, axis=0).astype(np.float32),
        np.concatenate(targets_chunks, axis=0).astype(np.int64),
    )


def _empirical_marginal(targets: np.ndarray) -> np.ndarray:
    """Per-class empirical frequency on the valid pitches."""
    counts = np.bincount(targets, minlength=NUM_OUTCOME_CLASSES).astype(
        np.float64
    )
    total = counts.sum()
    if total == 0:
        return np.full(NUM_OUTCOME_CLASSES, 1.0 / NUM_OUTCOME_CLASSES)
    return counts / total


def _predicted_marginal(probs: np.ndarray) -> np.ndarray:
    """Per-class mean predicted probability."""
    if probs.shape[0] == 0:
        return np.full(NUM_OUTCOME_CLASSES, 1.0 / NUM_OUTCOME_CLASSES)
    return probs.mean(axis=0).astype(np.float64)


def _apply_class_weights(
    probs: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Element-wise re-weight then renormalize per-row."""
    weighted = probs.astype(np.float64) * weights[None, :]
    denom = weighted.sum(axis=1, keepdims=True)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return weighted / denom


def _closed_form_weights(
    pred_marginal: np.ndarray, emp_marginal: np.ndarray
) -> np.ndarray:
    """``w_c ∝ p_emp_c / p_pred_c``, renormalize so geometric_mean(w)=1.

    The renormalization keeps numerical scale tame; the absolute
    scale of ``w`` is irrelevant because ``predict_outcome_probs``
    renormalizes per-pitch.  Geometric-mean=1 means
    ``sum(log w) = 0``.
    """
    safe_pred = np.where(pred_marginal > 1e-12, pred_marginal, 1e-12)
    raw = emp_marginal / safe_pred
    raw = np.where(raw > 1e-12, raw, 1e-12)
    log_w = np.log(raw)
    log_w = log_w - log_w.mean()  # geometric mean -> 1
    return np.exp(log_w)


def _kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) for two distributions over 7 classes."""
    p = np.where(p > 1e-12, p, 1e-12)
    q = np.where(q > 1e-12, q, 1e-12)
    return float(np.sum(p * np.log(p / q)))


def _lbfgs_refine_weights(
    probs: np.ndarray,
    emp_marginal: np.ndarray,
    init_weights: np.ndarray,
    max_iter: int = 200,
) -> np.ndarray:
    """Refine ``w`` via L-BFGS on ``KL(empirical || rebalanced_marginal)``.

    Parameterize ``w = exp(log_w)`` to keep positivity; remove gauge
    freedom by subtracting mean log_w each step (geometric_mean=1).
    """
    log_w = torch.tensor(np.log(init_weights), dtype=torch.float64,
                         requires_grad=True)
    P = torch.tensor(probs, dtype=torch.float64)         # (N, 7)
    p_emp = torch.tensor(emp_marginal, dtype=torch.float64)  # (7,)

    optimizer = torch.optim.LBFGS(
        [log_w],
        lr=1.0,
        max_iter=max_iter,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        history_size=20,
        line_search_fn="strong_wolfe",
    )

    def _closure():
        optimizer.zero_grad()
        # Center log_w (geometric_mean=1) ⇒ removes gauge freedom.
        lw = log_w - log_w.mean()
        w = torch.exp(lw)
        weighted = P * w[None, :]
        denom = weighted.sum(dim=1, keepdim=True).clamp_min(1e-12)
        re_p = weighted / denom
        q = re_p.mean(dim=0).clamp_min(1e-12)
        # KL(p_emp || q)
        kl = (p_emp * (torch.log(p_emp.clamp_min(1e-12)) - torch.log(q))).sum()
        kl.backward()
        return kl

    optimizer.step(_closure)
    final_log_w = (log_w - log_w.mean()).detach().cpu().numpy()
    return np.exp(final_log_w)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-games", type=int, default=DEFAULT_VAL_GAMES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Optional override for DuckDB path.",
    )
    args = parser.parse_args()

    # ── 1. SHA256-verify both checkpoints (pre).  ───────────────────────────
    bb_sha_pre = _sha256_file(BACKBONE_PATH)
    a1_sha_pre = _sha256_file(A1_CKPT)
    logger.info("v2.pt SHA256 (pre): %s", bb_sha_pre)
    logger.info("a1.pt SHA256 (pre): %s", a1_sha_pre)

    # ── 2. Load backbone (frozen).  ─────────────────────────────────────────
    logger.info("Loading PitchGPT v2 backbone ...")
    backbone = _load_model(version="2")
    freeze_backbone(backbone)
    bb_param_sha_pre = _backbone_param_checksum(backbone)
    device = _get_device()

    # ── 3. Load A1 head + temperature.  ─────────────────────────────────────
    logger.info("Loading A1 outcome head ...")
    ckpt = torch.load(A1_CKPT, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    head = FrozenOutcomeHeadConcat(
        d_model=cfg["d_model"],
        context_dim=cfg["context_dim"],
        n_pitch_types=cfg["n_pitch_types"],
        n_zones=cfg["n_zones"],
        n_velo_buckets=cfg["n_velo_buckets"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=cfg["dropout"],
        n_classes=cfg["n_classes"],
    )
    head.load_state_dict(ckpt["head_state_dict"])
    head.to(device).eval()
    for p in head.parameters():
        p.requires_grad = False
    temperature = float(ckpt["temperature"])
    logger.info("A1 temperature (locked): %.4f", temperature)

    # ── 4. Build 2023 val cohort exactly as in A1 training. ────────────────
    conn = get_connection(args.db_path, read_only=True)
    try:
        logger.info("Fetching train pitcher cohort (2015-2022) ...")
        train_pitchers = fetch_pitcher_ids(conn, TRAIN_SEASONS)
        logger.info("Train cohort pitchers: %d", len(train_pitchers))

        logger.info("Sampling val games (seed=%d, pitcher-disjoint) ...",
                    args.seed + 1)
        val_pks = sample_game_pks(
            conn, VAL_SEASONS, max_games=args.val_games,
            exclude_pitcher_ids=train_pitchers, seed=args.seed + 1,
        )
        logger.info("Val games selected: %d", len(val_pks))

        t0 = time.time()
        logger.info("Building val sequence dataset ...")
        val_ds = FixedGamesSequenceDataset(
            conn, game_pks=val_pks,
            exclude_pitcher_ids=train_pitchers,
            max_seq_len=256, context_dim=backbone.context_dim,
        )
        logger.info("Val ds built: %d sequences  (%.1fs)",
                    len(val_ds), time.time() - t0)
    finally:
        conn.close()

    # ── 5. Score post-T probs + empirical marginal on val.  ────────────────
    t0 = time.time()
    logger.info("Scoring A1 on 2023 val ...")
    probs, targets = _collect_post_T_probs(
        backbone, head, val_ds, temperature, args.batch_size, device,
    )
    logger.info("Val probs shape=%s  targets=%d  (%.1fs)",
                probs.shape, int(targets.size), time.time() - t0)
    if probs.shape[0] == 0:
        raise RuntimeError("Val cohort produced 0 valid pitches; abort.")

    pred_pre = _predicted_marginal(probs)
    emp = _empirical_marginal(targets)

    logger.info("--- 2023 val: pre-fit predicted vs empirical marginals ---")
    logger.info("%-18s %10s %10s %10s", "class", "pred", "emp", "delta_pp")
    for c, name in enumerate(OUTCOME_CLASSES):
        d_pp = (pred_pre[c] - emp[c]) * 100.0
        logger.info("%-18s %10.4f %10.4f  %+9.2fpp",
                    name, pred_pre[c], emp[c], d_pp)
    kl_pre = _kl_div(emp, pred_pre)
    logger.info("KL(emp || pred_pre) = %.6f", kl_pre)

    # ── 6. Fit closed-form weights; verify within 1pp; else L-BFGS.  ───────
    w_cf = _closed_form_weights(pred_pre, emp)
    probs_cf = _apply_class_weights(probs, w_cf)
    pred_cf = _predicted_marginal(probs_cf)
    max_dev_cf = float(np.max(np.abs(pred_cf - emp)))
    logger.info("Closed-form weights: %s", np.array2string(w_cf, precision=5))
    logger.info("Closed-form max |delta| post-fit: %.4fpp",
                max_dev_cf * 100.0)

    if max_dev_cf <= WITHIN_BAND_PP:
        weights = w_cf
        method = "closed_form"
    else:
        logger.info(
            "Closed-form residual %.4fpp > %.4fpp band; "
            "falling through to L-BFGS refinement ...",
            max_dev_cf * 100.0, WITHIN_BAND_PP * 100.0,
        )
        weights = _lbfgs_refine_weights(probs, emp, w_cf)
        method = "closed_form_then_lbfgs"

    probs_post = _apply_class_weights(probs, weights)
    pred_post = _predicted_marginal(probs_post)
    kl_post = _kl_div(emp, pred_post)

    logger.info("--- 2023 val: post-fit predicted vs empirical marginals ---")
    logger.info("%-18s %10s %10s %10s", "class", "pred_post", "emp",
                "delta_pp")
    for c, name in enumerate(OUTCOME_CLASSES):
        d_pp = (pred_post[c] - emp[c]) * 100.0
        logger.info("%-18s %10.4f %10.4f  %+9.2fpp",
                    name, pred_post[c], emp[c], d_pp)
    logger.info("KL(emp || pred_post) = %.6f  (improvement: %.6f -> %.6f)",
                kl_post, kl_pre, kl_post)
    logger.info("class_calibration vector (length 7, %s): %s",
                method, np.array2string(weights, precision=6))
    max_dev_final = float(np.max(np.abs(pred_post - emp)))
    logger.info("Final max |marginal delta|: %.4fpp", max_dev_final * 100.0)

    # ── 7. SHA256-verify both checkpoints (post).  ─────────────────────────
    bb_sha_post = _sha256_file(BACKBONE_PATH)
    a1_sha_post = _sha256_file(A1_CKPT)
    bb_param_sha_post = _backbone_param_checksum(backbone)
    assert bb_sha_pre == bb_sha_post, (
        f"v2.pt SHA drifted! pre={bb_sha_pre} post={bb_sha_post}"
    )
    assert a1_sha_pre == a1_sha_post, (
        f"a1.pt SHA drifted! pre={a1_sha_pre} post={a1_sha_post}"
    )
    assert bb_param_sha_pre == bb_param_sha_post, (
        f"Backbone param drift! pre={bb_param_sha_pre} "
        f"post={bb_param_sha_post}"
    )
    logger.info("Checkpoint byte-identity preserved (v2.pt + a1.pt).")

    # ── 8. Update calibration JSON.  ───────────────────────────────────────
    with open(CALIB_PATH, "r", encoding="utf-8") as fh:
        calib = json.load(fh)
    logger.info("Loaded existing calibration JSON from %s", CALIB_PATH)
    calib["class_calibration"] = [float(x) for x in weights.tolist()]
    calib["fit_date"] = FIT_DATE
    calib["class_calibration_method"] = method
    calib["class_calibration_val_kl_pre"] = float(kl_pre)
    calib["class_calibration_val_kl_post"] = float(kl_post)
    calib["class_calibration_val_max_delta_pp"] = float(
        max_dev_final * 100.0
    )
    calib["class_calibration_val_n_pitches"] = int(probs.shape[0])
    calib["class_calibration_val_season"] = int(VAL_SEASONS[0])
    with open(CALIB_PATH, "w", encoding="utf-8") as fh:
        json.dump(calib, fh, indent=2)
        fh.write("\n")
    logger.info("Wrote class_calibration to %s (fit_date=%s)",
                CALIB_PATH, FIT_DATE)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
