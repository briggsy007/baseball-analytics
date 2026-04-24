"""
PitchGPT 2025 holdout evidence run (Path 2, evidence-over-gates).

Loads the pitcher-disjoint PitchGPT checkpoint (trained 2015-2022, ran
through validate_pitchgpt_20260418T150305Z), trains a fresh LSTM
baseline on the same train split, fits Markov-2 and Heuristic on the
same training sequences, then scores all four on a 2025 holdout.

Strict holdout semantics:
  * Training window: 2015-2022 (same as validation run 150305Z).
  * Validation window: 2023 (used only for temperature scaling).
  * Holdout window: 2025 (NO model has seen 2025 pitches).
  * The 2025 test pitcher set is restricted to pitchers NOT in the
    train cohort — this is the pitcher-disjoint requirement inherited
    from the leakage fix in commit 11d74c5.

Outputs (results/pitchgpt/2025_holdout/):
  - perplexity_comparison.json  — headline: ppl for each model + CIs.
  - calibration_2025.json       — ECE pre/post temperature on 2025.
  - reliability_2025.html       — reliability diagram.
  - report.md                   — human-readable summary.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt import (  # noqa: E402
    PAD_TOKEN,
    TOTAL_VOCAB,
    VOCAB_SIZE,
    PitchGPTModel,
    PitchSequenceDataset,
    _collate_fn,
    _get_device,
    audit_no_game_overlap,
)
from src.analytics.pitch_lstm import PitchLSTMNetwork  # noqa: E402
from src.analytics.pitch_markov import (  # noqa: E402
    HeuristicBaseline,
    MarkovChainOrder2,
)
from src.analytics.pitchgpt_calibration import (  # noqa: E402
    compute_reliability_curve,
    expected_calibration_error,
    gather_predictions,
    temperature_scale,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitchgpt_2025_holdout")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_RANGE = (2015, 2022)
VAL_RANGE = (2023, 2023)
HOLDOUT_RANGE = (2025, 2025)

DEFAULT_BATCH = 32
DEFAULT_LR = 1e-3
GRAD_CLIP = 1.0

THRESH_LSTM_PCT = 15.0
THRESH_MARKOV2_PCT = 20.0
THRESH_HEURISTIC_PCT = 25.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Load PitchGPT checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_pitchgpt_checkpoint(path: Path, device: torch.device) -> PitchGPTModel:
    ck = torch.load(str(path), map_location=device, weights_only=True)
    cfg = ck["config"]
    # Infer the checkpoint's context_dim from the shape of
    # ``context_proj.weight`` rather than trusting ``config``.  v1
    # checkpoints predate the ``context_dim`` config key and were
    # trained with CONTEXT_DIM=34 (no ump scalar); v2 is 35.
    ctx_weight = ck["model_state_dict"]["context_proj.weight"]
    context_dim = int(ctx_weight.shape[1])
    model = PitchGPTModel(
        vocab_size=cfg.get("vocab_size", TOTAL_VOCAB),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        context_dim=context_dim,
    )
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    model.eval()
    model.context_dim = context_dim  # for downstream dataset-building
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation: per-pitch NLL array (needed for bootstrap CIs)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def per_pitch_nll_neural(
    model: nn.Module,
    dataset,
    device: torch.device,
    output_vocab: int,
    batch_size: int = DEFAULT_BATCH,
) -> np.ndarray:
    """Return a flat array of per-pitch NLLs over all non-PAD targets."""
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn,
    )
    # v1 PitchGPT has context_dim=34 while datasets default to 35 — slice
    # the tail (ump scalar) so v1 can be scored against a 35-dim dataset.
    from src.analytics.pitchgpt import CONTEXT_DIM as _DATASET_CTX_DIM
    m_ctx_dim = getattr(model, "context_dim", _DATASET_CTX_DIM)
    nlls: list[np.ndarray] = []
    for tokens, ctx, target in loader:
        tokens = tokens.to(device)
        ctx = ctx.to(device)
        target = target.to(device)
        if ctx.size(-1) > m_ctx_dim:
            ctx = ctx[..., :m_ctx_dim]

        logits = model(tokens, ctx)  # (B, S, V)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        flat_logp = log_probs.reshape(-1, output_vocab)
        flat_target = target.reshape(-1)
        valid = flat_target != PAD_TOKEN
        if valid.sum().item() == 0:
            continue
        flat_logp = flat_logp[valid]
        flat_target = flat_target[valid]

        # Some flat_target entries could equal the PAD_TOKEN value for
        # a PitchGPT output head that has output_vocab == VOCAB_SIZE
        # (< PAD_TOKEN), but PAD is already filtered.  Clamp only to be
        # safe with the LSTM variant.
        in_range = (flat_target >= 0) & (flat_target < output_vocab)
        flat_logp = flat_logp[in_range]
        flat_target = flat_target[in_range]
        if flat_target.numel() == 0:
            continue

        tgt_logp = flat_logp.gather(1, flat_target.unsqueeze(1)).squeeze(1)
        nlls.append((-tgt_logp).cpu().numpy())

    return np.concatenate(nlls) if nlls else np.array([])


def per_pitch_nll_closed_form(model, dataset) -> np.ndarray:
    """Return per-pitch NLL tensor from a closed-form baseline."""
    t = model.score_sequences(dataset)
    return t.cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI on perplexity and on the Δppl gap
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ppl(nll: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict:
    """Return bootstrap CI for perplexity = exp(mean(nll)).

    Resamples per-pitch NLLs with replacement; CI is the 2.5/97.5%
    quantiles of the bootstrap distribution of mean(nll).
    """
    rng = np.random.default_rng(seed)
    n = len(nll)
    point = float(math.exp(nll.mean()) if n > 0 else float("inf"))
    boot_ppls = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_ppls[i] = math.exp(nll[idx].mean())
    lo, hi = np.percentile(boot_ppls, [2.5, 97.5])
    return {
        "point": round(point, 3),
        "ci95_lo": round(float(lo), 3),
        "ci95_hi": round(float(hi), 3),
        "n_pitches": int(n),
        "n_bootstrap": n_boot,
    }


def bootstrap_delta_pct(
    nll_challenger: np.ndarray,
    nll_baseline: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Bootstrap the improvement-pct gap between challenger and baseline.

    Resamples the two NLL arrays independently (they may have different
    lengths — same dataset but different skipping on OOV edge cases).
    Uses `100 * (ppl_baseline - ppl_challenger) / ppl_baseline`.
    """
    rng = np.random.default_rng(seed)
    nc, nb = len(nll_challenger), len(nll_baseline)
    ppl_c_point = math.exp(nll_challenger.mean()) if nc else float("inf")
    ppl_b_point = math.exp(nll_baseline.mean()) if nb else float("inf")
    point_pct = 100.0 * (ppl_b_point - ppl_c_point) / ppl_b_point if ppl_b_point > 0 else 0.0

    gaps = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx_c = rng.integers(0, nc, size=nc)
        idx_b = rng.integers(0, nb, size=nb)
        ppl_c = math.exp(nll_challenger[idx_c].mean())
        ppl_b = math.exp(nll_baseline[idx_b].mean())
        gaps[i] = 100.0 * (ppl_b - ppl_c) / ppl_b
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {
        "improvement_pct_point": round(float(point_pct), 2),
        "improvement_pct_ci95_lo": round(float(lo), 2),
        "improvement_pct_ci95_hi": round(float(hi), 2),
        "n_bootstrap": n_boot,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Train LSTM with same train/val split as PitchGPT
# ─────────────────────────────────────────────────────────────────────────────

def train_lstm(
    train_ds,
    val_ds,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    context_dim: int | None = None,
) -> tuple[PitchLSTMNetwork, dict]:
    _set_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn,
    )

    if context_dim is None:
        model = PitchLSTMNetwork().to(device)
    else:
        model = PitchLSTMNetwork(context_dim=context_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total_loss, total_tok = 0.0, 0
        for tokens, ctx, target in train_loader:
            tokens = tokens.to(device)
            ctx = ctx.to(device)
            target = target.to(device)
            logits = model(tokens, ctx)
            loss = criterion(
                logits.reshape(-1, model.output_vocab),
                target.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            n_tok = (target != PAD_TOKEN).sum().item()
            total_loss += loss.item() * n_tok
            total_tok += n_tok
        train_loss = total_loss / max(total_tok, 1)

        model.eval()
        v_loss, v_tok = 0.0, 0
        with torch.no_grad():
            for tokens, ctx, target in val_loader:
                tokens = tokens.to(device)
                ctx = ctx.to(device)
                target = target.to(device)
                logits = model(tokens, ctx)
                loss = criterion(
                    logits.reshape(-1, model.output_vocab),
                    target.reshape(-1),
                )
                n_tok = (target != PAD_TOKEN).sum().item()
                v_loss += loss.item() * n_tok
                v_tok += n_tok
        val_loss = v_loss / max(v_tok, 1)

        dt = time.perf_counter() - t0
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_ppl": round(math.exp(min(train_loss, 20)), 3),
            "val_ppl": round(math.exp(min(val_loss, 20)), 3),
            "wall_clock_sec": round(dt, 1),
        }
        history.append(entry)
        logger.info(
            "[LSTM] ep %d/%d  train=%.4f (ppl %.2f)  val=%.4f (ppl %.2f)  %.1fs",
            epoch, epochs,
            train_loss, entry["train_ppl"],
            val_loss, entry["val_ppl"], dt,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, {
        "params": sum(p.numel() for p in model.parameters()),
        "epoch_best": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "history": history,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_ROOT / "results" / "validate_pitchgpt_20260418T150305Z" / "pitchgpt_full.pt",
        help="PitchGPT checkpoint (pitcher-disjoint, trained on 2015-2022).",
    )
    parser.add_argument("--max-train-games", type=int, default=1000)
    parser.add_argument("--max-val-games", type=int, default=300)
    parser.add_argument("--max-holdout-games", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "results" / "pitchgpt" / "2025_holdout",
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
    )
    parser.add_argument(
        "--context-dim",
        type=int,
        default=None,
        help=("Context tensor width for datasets + LSTM retrain.  "
              "Default = inferred from PitchGPT checkpoint (v1=34, v2=35) "
              "so LSTM and baselines match what PitchGPT saw at training "
              "time.  Pass explicitly to override."),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)
    device = _get_device()
    logger.info("device=%s  seed=%d  checkpoint=%s", device, args.seed, args.checkpoint)

    # Infer context_dim from the checkpoint if not explicitly set — so
    # LSTM + baselines + datasets all see the same context width that
    # PitchGPT saw at training time.  For v1 (34) this hides the ump
    # scalar; for v2 (35) the full width is used.
    if args.context_dim is None:
        _ck_peek = torch.load(str(args.checkpoint), map_location="cpu", weights_only=True)
        args.context_dim = int(_ck_peek["model_state_dict"]["context_proj.weight"].shape[1])
        del _ck_peek
    logger.info("context_dim resolved to %d", args.context_dim)

    # ── 1. Load datasets with the SAME pitcher-disjoint split as the
    #      training run that produced the PitchGPT checkpoint.  The
    #      holdout dataset is then constructed with the train cohort's
    #      pitcher_ids as the exclusion set so no train pitcher appears
    #      in the holdout.
    conn = get_connection(args.db_path, read_only=True)
    try:
        logger.info(
            "Loading train (2015-2022, pitcher-disjoint cohort)  "
            "val (2023, pitcher-disjoint)  holdout (2025, pitcher-disjoint)",
        )
        # Build train first — we need its pitcher set to exclude from val/holdout.
        train_pitchers_all = PitchSequenceDataset.fetch_pitcher_ids_for_seasons(
            conn, range(TRAIN_RANGE[0], TRAIN_RANGE[1] + 1),
        )
        logger.info("Train cohort: %d pitchers (2015-2022 full)", len(train_pitchers_all))

        train_ds = PitchSequenceDataset(
            conn,
            split_mode="train",
            train_range=TRAIN_RANGE,
            val_range=VAL_RANGE,
            test_range=HOLDOUT_RANGE,
            max_games_per_split=args.max_train_games,
            context_dim=args.context_dim,
        )
        val_ds = PitchSequenceDataset(
            conn,
            split_mode="val",
            train_range=TRAIN_RANGE,
            val_range=VAL_RANGE,
            test_range=HOLDOUT_RANGE,
            max_games_per_split=args.max_val_games,
            exclude_pitcher_ids=train_pitchers_all,
            context_dim=args.context_dim,
        )
        # Holdout = 2025 pitches, strictly pitcher-disjoint from train.
        holdout_ds = PitchSequenceDataset(
            conn,
            split_mode="test",
            train_range=TRAIN_RANGE,
            val_range=VAL_RANGE,
            test_range=HOLDOUT_RANGE,
            max_games_per_split=args.max_holdout_games,
            exclude_pitcher_ids=train_pitchers_all,
            context_dim=args.context_dim,
        )
    finally:
        conn.close()

    logger.info(
        "Dataset sizes: train=%d  val=%d  holdout=%d",
        len(train_ds), len(val_ds), len(holdout_ds),
    )
    if len(train_ds) == 0 or len(val_ds) == 0 or len(holdout_ds) == 0:
        logger.error("Empty dataset — aborting.")
        return 1

    audit = audit_no_game_overlap(train_ds, val_ds, holdout_ds)
    logger.info("Leakage audit: %s", json.dumps(audit))
    if audit["shared_game_pks"] != 0 or audit["shared_pitcher_ids_train_test"] != 0:
        logger.error(
            "LEAKAGE: shared game_pks=%d  shared train/holdout pitchers=%d",
            audit["shared_game_pks"], audit["shared_pitcher_ids_train_test"],
        )
        return 2

    # ── 2. Load PitchGPT.
    pitchgpt = load_pitchgpt_checkpoint(args.checkpoint, device)
    logger.info("PitchGPT loaded (%d params)",
                sum(p.numel() for p in pitchgpt.parameters()))

    # ── 3. Train LSTM on the same split with the same context_dim as
    #      PitchGPT — keeps the baseline apples-to-apples with the
    #      v1 (34) vs v2 (35) runs.
    lstm, lstm_meta = train_lstm(
        train_ds, val_ds, device,
        epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, seed=args.seed,
        context_dim=args.context_dim,
    )

    # ── 4. Fit closed-form baselines on train.
    logger.info("Fitting Markov-2 on train set...")
    t0 = time.perf_counter()
    markov2 = MarkovChainOrder2()
    markov2.fit(train_ds)
    m2_fit = time.perf_counter() - t0
    logger.info("Markov-2 fit in %.1fs", m2_fit)

    logger.info("Fitting Heuristic on train set...")
    t0 = time.perf_counter()
    heuristic = HeuristicBaseline()
    heuristic.fit(train_ds)
    h_fit = time.perf_counter() - t0
    logger.info("Heuristic fit in %.1fs", h_fit)

    # ── 5. Compute per-pitch NLL on 2025 holdout for all models.
    logger.info("Scoring PitchGPT on 2025 holdout...")
    nll_pg = per_pitch_nll_neural(pitchgpt, holdout_ds, device, VOCAB_SIZE)
    ppl_pg = bootstrap_ppl(nll_pg, n_boot=args.n_bootstrap, seed=args.seed)
    logger.info("PitchGPT: ppl=%.3f (95%%CI %.3f-%.3f), n=%d",
                ppl_pg["point"], ppl_pg["ci95_lo"], ppl_pg["ci95_hi"], ppl_pg["n_pitches"])

    logger.info("Scoring LSTM on 2025 holdout...")
    nll_lstm = per_pitch_nll_neural(lstm, holdout_ds, device, lstm.output_vocab)
    ppl_lstm = bootstrap_ppl(nll_lstm, n_boot=args.n_bootstrap, seed=args.seed + 1)
    logger.info("LSTM: ppl=%.3f (95%%CI %.3f-%.3f)",
                ppl_lstm["point"], ppl_lstm["ci95_lo"], ppl_lstm["ci95_hi"])

    logger.info("Scoring Markov-2 on 2025 holdout...")
    nll_m2 = per_pitch_nll_closed_form(markov2, holdout_ds)
    ppl_m2 = bootstrap_ppl(nll_m2, n_boot=args.n_bootstrap, seed=args.seed + 2)
    logger.info("Markov-2: ppl=%.3f (95%%CI %.3f-%.3f)",
                ppl_m2["point"], ppl_m2["ci95_lo"], ppl_m2["ci95_hi"])

    logger.info("Scoring Heuristic on 2025 holdout...")
    nll_h = per_pitch_nll_closed_form(heuristic, holdout_ds)
    ppl_h = bootstrap_ppl(nll_h, n_boot=args.n_bootstrap, seed=args.seed + 3)
    logger.info("Heuristic: ppl=%.3f (95%%CI %.3f-%.3f)",
                ppl_h["point"], ppl_h["ci95_lo"], ppl_h["ci95_hi"])

    # ── 6. Gate checks with CIs.
    gap_lstm = bootstrap_delta_pct(nll_pg, nll_lstm, args.n_bootstrap, args.seed)
    gap_m2 = bootstrap_delta_pct(nll_pg, nll_m2, args.n_bootstrap, args.seed + 10)
    gap_h = bootstrap_delta_pct(nll_pg, nll_h, args.n_bootstrap, args.seed + 20)

    def _gate(gap: dict, thresh: float) -> dict:
        passed = gap["improvement_pct_point"] >= thresh
        # Conservative read: lower 95% CI bound must meet the threshold
        # for a "confidently passes" verdict.
        ci_passes = gap["improvement_pct_ci95_lo"] >= thresh
        return {
            "threshold_pct": thresh,
            "point_pct": gap["improvement_pct_point"],
            "ci95_lo": gap["improvement_pct_ci95_lo"],
            "ci95_hi": gap["improvement_pct_ci95_hi"],
            "pass_point": bool(passed),
            "pass_ci_lower_bound": bool(ci_passes),
        }

    gate_lstm = _gate(gap_lstm, THRESH_LSTM_PCT)
    gate_m2 = _gate(gap_m2, THRESH_MARKOV2_PCT)
    gate_h = _gate(gap_h, THRESH_HEURISTIC_PCT)

    all_pass_point = gate_lstm["pass_point"] and gate_m2["pass_point"] and gate_h["pass_point"]
    all_pass_ci = gate_lstm["pass_ci_lower_bound"] and gate_m2["pass_ci_lower_bound"] and gate_h["pass_ci_lower_bound"]

    # ── 7. Calibration on 2025.
    logger.info("Computing calibration on 2025...")
    test_preds = gather_predictions(
        pitchgpt, holdout_ds, batch_size=args.batch_size, device=device,
        return_logits=True,
    )
    n_holdout_tokens = int(test_preds["top1_prob"].shape[0])
    pre_curve = compute_reliability_curve(
        test_preds["top1_prob"], test_preds["is_correct"], n_bins=10,
    )
    ece_pre = expected_calibration_error(pre_curve)

    val_preds = gather_predictions(
        pitchgpt, val_ds, batch_size=args.batch_size, device=device,
        return_logits=True,
    )
    if val_preds["logits"].shape[0] == 0:
        T_opt = 1.0
    else:
        T_opt = temperature_scale(val_preds["logits"], val_preds["target"])

    if n_holdout_tokens > 0 and abs(T_opt - 1.0) > 1e-6:
        scaled = test_preds["logits"] / T_opt
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        probs_post = np.exp(scaled)
        probs_post /= probs_post.sum(axis=1, keepdims=True)
        top1_idx = probs_post.argmax(axis=1)
        top1_prob_post = probs_post.max(axis=1)
        is_correct_post = (top1_idx == test_preds["target"])
    else:
        top1_prob_post = test_preds["top1_prob"]
        is_correct_post = test_preds["is_correct"]
    post_curve = compute_reliability_curve(
        top1_prob_post, is_correct_post, n_bins=10,
    )
    ece_post = expected_calibration_error(post_curve)
    accuracy = float(test_preds["is_correct"].mean()) if n_holdout_tokens > 0 else 0.0

    calibration = {
        "n_holdout_tokens": n_holdout_tokens,
        "n_val_tokens": int(val_preds["top1_prob"].shape[0]),
        "accuracy_top1": round(accuracy, 4),
        "ece_pre_temp": round(float(ece_pre), 4),
        "ece_post_temp": round(float(ece_post), 4),
        "optimal_temperature": round(float(T_opt), 4),
        "reliability_curve_pre": pre_curve,
        "reliability_curve_post": post_curve,
        "gate_ece_pre": bool(ece_pre < 0.10),
        "gate_ece_post": bool(ece_post < 0.10),
    }
    logger.info(
        "Calibration 2025: ECE pre=%.4f post=%.4f T=%.3f acc=%.3f n=%d",
        ece_pre, ece_post, T_opt, accuracy, n_holdout_tokens,
    )

    # ── 8. Emit artifacts.
    perplexity_payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "train_range": f"{TRAIN_RANGE[0]}-{TRAIN_RANGE[1]}",
        "val_range": f"{VAL_RANGE[0]}",
        "holdout_range": f"{HOLDOUT_RANGE[0]}",
        "max_train_games": args.max_train_games,
        "max_val_games": args.max_val_games,
        "max_holdout_games": args.max_holdout_games,
        "epochs_lstm": args.epochs,
        "leakage_audit": audit,
        "n_train_sequences": len(train_ds),
        "n_val_sequences": len(val_ds),
        "n_holdout_sequences": len(holdout_ds),
        "models": {
            "pitchgpt": {
                "source": "results/validate_pitchgpt_20260418T150305Z/pitchgpt_full.pt (pitcher-disjoint)",
                "params": int(sum(p.numel() for p in pitchgpt.parameters())),
                "holdout_perplexity": ppl_pg,
            },
            "lstm": {
                "source": "freshly trained on train_ds (pitcher-disjoint from 2025)",
                "params": lstm_meta["params"],
                "epoch_best": lstm_meta["epoch_best"],
                "best_val_loss": lstm_meta["best_val_loss"],
                "history": lstm_meta["history"],
                "holdout_perplexity": ppl_lstm,
            },
            "markov2": {
                "source": "fitted on train_ds (2015-2022, pitcher-disjoint)",
                "params": 0,
                "fit_seconds": round(m2_fit, 1),
                "holdout_perplexity": ppl_m2,
            },
            "heuristic": {
                "source": "fitted on train_ds (2015-2022, pitcher-disjoint)",
                "params": 0,
                "fit_seconds": round(h_fit, 1),
                "holdout_perplexity": ppl_h,
            },
        },
        "gates": {
            "vs_lstm": {**gate_lstm, "name": "PitchGPT vs LSTM"},
            "vs_markov2": {**gate_m2, "name": "PitchGPT vs Markov-2"},
            "vs_heuristic": {**gate_h, "name": "PitchGPT vs Heuristic"},
            "overall_pass_point": all_pass_point,
            "overall_pass_ci_lower_bound": all_pass_ci,
        },
    }

    perplexity_path = args.output_dir / "perplexity_comparison.json"
    with perplexity_path.open("w", encoding="utf-8") as f:
        json.dump(perplexity_payload, f, indent=2)
    logger.info("Wrote %s", perplexity_path)

    calibration_path = args.output_dir / "calibration_2025.json"
    with calibration_path.open("w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    logger.info("Wrote %s", calibration_path)

    # Reliability diagram
    reliability_path = args.output_dir / "reliability_2025.html"
    try:
        _write_reliability_html(reliability_path, pre_curve, post_curve, T_opt, ece_pre, ece_post)
    except Exception as exc:
        logger.warning("Failed to write reliability diagram: %s", exc)

    # Report markdown
    report_path = args.output_dir / "report.md"
    _write_report(
        report_path,
        perplexity_payload=perplexity_payload,
        calibration=calibration,
    )
    logger.info("Wrote %s", report_path)

    # ── 9. Print summary.
    print("\n" + "=" * 72)
    print("PitchGPT 2025 Holdout -- Summary")
    print("=" * 72)
    print(f"PitchGPT   ppl = {ppl_pg['point']}  (95% CI {ppl_pg['ci95_lo']}-{ppl_pg['ci95_hi']})")
    print(f"LSTM       ppl = {ppl_lstm['point']}  (95% CI {ppl_lstm['ci95_lo']}-{ppl_lstm['ci95_hi']})")
    print(f"Markov-2   ppl = {ppl_m2['point']}  (95% CI {ppl_m2['ci95_lo']}-{ppl_m2['ci95_hi']})")
    print(f"Heuristic  ppl = {ppl_h['point']}  (95% CI {ppl_h['ci95_lo']}-{ppl_h['ci95_hi']})")
    print("-" * 72)
    for key, gate in [("vs_lstm", gate_lstm), ("vs_markov2", gate_m2), ("vs_heuristic", gate_h)]:
        v = "PASS" if gate["pass_point"] else "FAIL"
        ci = "PASS" if gate["pass_ci_lower_bound"] else "FAIL"
        print(
            f"{key:12s}  {gate['point_pct']:+.2f}% improvement  "
            f"(CI {gate['ci95_lo']:+.2f} / {gate['ci95_hi']:+.2f})  "
            f"threshold {gate['threshold_pct']:.0f}%  point={v}  CI={ci}"
        )
    print(f"Calibration 2025: ECE pre={ece_pre:.4f} post={ece_post:.4f} (T={T_opt:.3f}) n={n_holdout_tokens}")
    print("=" * 72)
    return 0


def _write_reliability_html(
    path: Path,
    pre_curve: list[dict],
    post_curve: list[dict],
    T_opt: float,
    ece_pre: float,
    ece_post: float,
) -> None:
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError:
        path.write_text("plotly not installed", encoding="utf-8")
        return
    fig = go.Figure()
    for name, curve, color in [
        (f"pre-temp (ECE={ece_pre:.4f})", pre_curve, "#2a6df4"),
        (f"post-temp T={T_opt:.3f} (ECE={ece_post:.4f})", post_curve, "#e04e4e"),
    ]:
        mean_conf = [b["mean_conf"] for b in curve if b["n_samples"] > 0]
        emp_acc = [b["empirical_acc"] for b in curve if b["n_samples"] > 0]
        fig.add_trace(go.Scatter(
            x=mean_conf, y=emp_acc, mode="lines+markers", name=name,
            line=dict(color=color, width=2),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="ideal (y=x)",
        line=dict(color="grey", dash="dash", width=1),
    ))
    fig.update_layout(
        title="PitchGPT 2025 Holdout Reliability",
        xaxis_title="Mean predicted confidence",
        yaxis_title="Empirical accuracy",
        template="plotly_white",
        width=700, height=500,
    )
    fig.write_html(str(path), include_plotlyjs="cdn")


def _write_report(path: Path, perplexity_payload: dict, calibration: dict) -> None:
    lines = []
    lines.append("# PitchGPT 2025 Holdout Report\n")
    lines.append(f"Generated: {perplexity_payload['timestamp_utc']}\n")
    lines.append(f"Checkpoint: `{perplexity_payload['checkpoint']}`\n")
    lines.append(f"Train: {perplexity_payload['train_range']} ({perplexity_payload['n_train_sequences']} sequences)\n")
    lines.append(f"Val:   {perplexity_payload['val_range']} ({perplexity_payload['n_val_sequences']} sequences, used only for temperature scaling)\n")
    lines.append(f"Holdout: {perplexity_payload['holdout_range']} ({perplexity_payload['n_holdout_sequences']} sequences, pitcher-disjoint from train)\n")
    lines.append("")
    audit = perplexity_payload["leakage_audit"]
    lines.append("## Leakage audit\n")
    lines.append(f"- shared game_pks across splits: **{audit['shared_game_pks']}**")
    lines.append(f"- shared pitchers train/holdout: **{audit['shared_pitcher_ids_train_test']}**")
    lines.append(f"- train pitchers: {audit['n_train_pitchers']}  val: {audit['n_val_pitchers']}  holdout: {audit['n_test_pitchers']}")
    lines.append("")
    lines.append("## Holdout perplexity (lower is better)\n")
    lines.append("| Model | Params | Holdout PPL | 95% CI | N pitches |")
    lines.append("|---|---|---|---|---|")
    models = perplexity_payload["models"]
    for key in ["pitchgpt", "lstm", "markov2", "heuristic"]:
        m = models[key]
        p = m["holdout_perplexity"]
        lines.append(
            f"| {key} | {m.get('params', '-'):,} | {p['point']} | {p['ci95_lo']} – {p['ci95_hi']} | {p['n_pitches']} |"
        )
    lines.append("")
    lines.append("## Gates\n")
    lines.append("| Comparison | Spec | Point | 95% CI | Point verdict | CI-lower verdict |")
    lines.append("|---|---|---|---|---|---|")
    gates = perplexity_payload["gates"]
    for key in ["vs_lstm", "vs_markov2", "vs_heuristic"]:
        g = gates[key]
        p_v = "PASS" if g["pass_point"] else "FAIL"
        ci_v = "PASS" if g["pass_ci_lower_bound"] else "FAIL"
        lines.append(
            f"| {g['name']} | ≥{g['threshold_pct']:.0f}% | {g['point_pct']:+.2f}% | "
            f"{g['ci95_lo']:+.2f} / {g['ci95_hi']:+.2f} | {p_v} | {ci_v} |"
        )
    lines.append("")
    lines.append(f"**Overall (point)**: {'PASS' if gates['overall_pass_point'] else 'FAIL'}")
    lines.append(f"**Overall (CI lower bound)**: {'PASS' if gates['overall_pass_ci_lower_bound'] else 'FAIL'}")
    lines.append("")
    lines.append("## Calibration on 2025 (out-of-sample)\n")
    lines.append(f"- ECE pre-temperature: **{calibration['ece_pre_temp']}** (gate <0.10: {'PASS' if calibration['gate_ece_pre'] else 'FAIL'})")
    lines.append(f"- ECE post-temperature (T={calibration['optimal_temperature']}): **{calibration['ece_post_temp']}** (gate <0.10: {'PASS' if calibration['gate_ece_post'] else 'FAIL'})")
    lines.append(f"- Top-1 accuracy: {calibration['accuracy_top1']}")
    lines.append(f"- Non-PAD pitches scored: {calibration['n_holdout_tokens']}")
    lines.append("")
    lines.append("## Data provenance\n")
    lines.append("- PitchGPT checkpoint: trained 2015-2022 (pitcher-disjoint) per validate_pitchgpt_20260418T150305Z.")
    lines.append("- LSTM: freshly trained on the same 2015-2022 pitcher-disjoint cohort.")
    lines.append("- Markov-2, Heuristic: fit on the same training sequences.")
    lines.append("- 2025 was NOT observed during training or validation of any model.")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
