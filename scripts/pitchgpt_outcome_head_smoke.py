"""
Phase 0.2 smoke harness — PitchGPT PA-outcome head design comparison.

Trains two candidate head architectures at 500-game scale on a
pitcher-disjoint 2015-2022 cohort and evaluates both on a 100-game
pitcher-disjoint 2023 slice:

    (a) JOINT  : backbone unfrozen, joint next-token CE + outcome CE loss.
                 Checkpoint: models/pitchgpt_outcome_smoke_joint.pt
    (b) FROZEN : backbone frozen (gradients disabled), 2-layer MLP head.
                 Checkpoint: models/pitchgpt_outcome_smoke_frozen.pt

Decision criterion (from EXECUTION_PLAN §4.1, locked 2026-04-24):
    Lower OOS 7-class log-loss wins, subject to the hard constraint that
    the backbone's next-token ECE on the same eval slice does NOT degrade
    by more than +0.005 absolute relative to the locked v2 checkpoint.

Usage (serial; does NOT run concurrently with the 0.1 sampling-fidelity
agent at default batch sizes — check nvidia-smi first):

    python scripts/pitchgpt_outcome_head_smoke.py

Guardrails:
    * Does NOT overwrite models/pitchgpt_v2.pt.
    * Uses read-only DuckDB connections.
    * Writes results to
      results/pitchgpt/outcome_head_smoke_2026_04_24/metrics.json
      with both options' log-loss + ECE numbers.
    * Can be re-run idempotently; smoke checkpoints are overwritten each
      invocation.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt import (  # noqa: E402
    CONTEXT_DIM,
    PAD_TOKEN,
    VOCAB_SIZE,
    PitchGPTModel,
    PitchSequenceDataset,
    _collate_fn,
    _get_device,
    _load_model,
)
from src.analytics.pitchgpt_outcome_head import (  # noqa: E402
    NUM_OUTCOME_CLASSES,
    OUTCOME_UNK,
    FrozenOutcomeHead,
    JointOutcomeHead,
    PitchGPTWithOutcomeHead,
    class_frequency_prior,
    classify_pitch_outcome,
    extract_backbone_hidden_states,
    freeze_backbone,
    prior_log_loss,
    seven_class_ece,
    seven_class_log_loss,
)
from src.analytics.pitchgpt_calibration import (  # noqa: E402
    compute_reliability_curve,
    expected_calibration_error,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitchgpt_outcome_head_smoke")


# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════

OUT_DIR = _ROOT / "results" / "pitchgpt" / "outcome_head_smoke_2026_04_24"
MODELS_DIR = _ROOT / "models"
CHKPT_JOINT = MODELS_DIR / "pitchgpt_outcome_smoke_joint.pt"
CHKPT_FROZEN = MODELS_DIR / "pitchgpt_outcome_smoke_frozen.pt"

TRAIN_RANGE = (2015, 2022)
EVAL_RANGE = (2023, 2023)

# Smoke scale: small enough to complete in ~15-30 min wall-clock even when
# sharing GPU with the 0.1 sampling-fidelity training run.
DEFAULT_TRAIN_GAMES = 500
DEFAULT_EVAL_GAMES = 100

DEFAULT_BATCH = 16
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS_JOINT = 2
DEFAULT_EPOCHS_FROZEN = 3
DEFAULT_LAMBDA = 0.5  # outcome loss weight in joint objective

SEED = 42


# ═════════════════════════════════════════════════════════════════════════════
# Outcome-labelled dataset wrapper
# ═════════════════════════════════════════════════════════════════════════════


class OutcomeLabelledDataset:
    """Wraps a :class:`PitchSequenceDataset` and attaches per-pitch 7-class
    labels reconstructed from the underlying DuckDB rows.

    Rather than modifying :class:`PitchSequenceDataset` (which would
    perturb existing behaviour), we re-query the pitches we already
    loaded and join on ``(game_pk, pitcher_id, at_bat_number,
    pitch_number)`` to fetch the ``description`` + ``events`` needed for
    labelling.  The labels are aligned to the *target* sequence (tokens
    shifted by one) so loss computation remains trivially consistent with
    the backbone's next-token CE.

    The dataset exposes a ``__getitem__`` that returns
    ``(tokens, context, target_tokens, outcome_labels)``.
    """

    def __init__(
        self,
        base: PitchSequenceDataset,
        conn: Any,
    ):
        self.base = base
        self._labels: list[torch.Tensor] = []
        self._build_labels(conn)

    def _build_labels(self, conn: Any) -> None:
        """Re-fetch (description, events) per game and compute 7-class labels.

        We re-query the pitches for the exact set of ``game_pk`` values
        used by the base dataset, sort them in the **same** order the
        base loader's SQL used (``game_pk, at_bat_number, pitch_number``
        — NOT by ``pitcher_id``, which the base does not sort by), and
        apply the same ``groupby(["game_pk","pitcher_id"], sort=False)``
        pass.  Because ``sort=False`` iterates groups in
        first-appearance order, the resulting group sequence is
        identical to the order in which the base loader constructed its
        sequences.

        Alignment: ``base.sequences[i]`` stores
        ``(input_tokens[:T-1], context[:T-1], target_tokens[1:T])`` for
        the ``(game_pk, pitcher_id)`` group's pitch stream of length
        ``T = min(raw_T, max_seq_len)``.  We want an ``outcome_labels``
        tensor of length ``T - 1`` whose ``j``-th entry is the outcome
        class of the pitch at position ``j+1`` in the original stream
        (i.e., aligned with ``target_tokens[j]``).

        The base loader drops groups with ``len(game_df) < 2`` and also
        groups whose pitcher is in the exclusion set.  We replicate
        both filters here.
        """
        if not self.base.sequences:
            return

        gpks_set = set(self.base.game_pks)
        if not gpks_set:
            logger.warning(
                "OutcomeLabelledDataset: base has sequences but empty game_pks; "
                "cannot reproduce grouping — falling back to UNK labels.",
            )
            for _, _, tgt in self.base.sequences:
                self._labels.append(
                    torch.full((tgt.shape[0],), OUTCOME_UNK, dtype=torch.long)
                )
            return

        gpks = ", ".join(str(int(g)) for g in gpks_set)
        # Match the base loader's SQL: no pitcher-id in ORDER BY, since
        # the base SQL orders only by (game_pk, at_bat_number, pitch_number).
        query = f"""
            SELECT game_pk, pitcher_id, at_bat_number, pitch_number,
                   description, events
            FROM pitches
            WHERE pitch_type IS NOT NULL
              AND game_pk IN ({gpks})
            ORDER BY game_pk, at_bat_number, pitch_number
        """
        df = conn.execute(query).fetchdf()
        if df.empty:
            logger.warning("OutcomeLabelledDataset: re-query returned 0 rows.")
            for _, _, tgt in self.base.sequences:
                self._labels.append(
                    torch.full((tgt.shape[0],), OUTCOME_UNK, dtype=torch.long)
                )
            return

        # Apply pitcher-exclusion filter (same as base).
        if self.base.exclude_pitcher_ids:
            excl = self.base.exclude_pitcher_ids
            df = df[~df["pitcher_id"].isin(excl)].reset_index(drop=True)

        # Vectorised labelling (faster than .apply for 100K+ rows).
        descr_vals = df["description"].tolist()
        events_vals = df["events"].tolist()
        labels = [classify_pitch_outcome(d, e) for d, e in zip(descr_vals, events_vals)]
        df = df.assign(_outcome=labels)

        # Reproduce the base loader's groupby ordering. sort=False means
        # groups appear in first-pitch order, same as the base.
        max_seq_len = self.base.max_seq_len
        seq_idx = 0
        for key, group_df in df.groupby(["game_pk", "pitcher_id"], sort=False):
            if len(group_df) < 2:
                continue
            if seq_idx >= len(self.base.sequences):
                # More groups than sequences — base truncated or filtered
                # further (unlikely given we mirror the filters); stop.
                break
            raw_outcomes = group_df["_outcome"].tolist()
            truncated = raw_outcomes[:max_seq_len]
            # target_tokens[j] corresponds to pitch j+1 in the truncated
            # stream; outcome_targets[j] must align with the same pitch.
            outcome_targets = truncated[1:]  # length = min(raw_T, max_seq_len) - 1

            _, _, base_target = self.base.sequences[seq_idx]
            T_minus_1 = int(base_target.shape[0])
            if len(outcome_targets) != T_minus_1:
                # Graceful fallback: length mismatch (rare; could happen if
                # the base filtered additional rows we did not reproduce).
                # Truncate or pad to match.
                if len(outcome_targets) > T_minus_1:
                    outcome_targets = outcome_targets[:T_minus_1]
                else:
                    outcome_targets = outcome_targets + [OUTCOME_UNK] * (
                        T_minus_1 - len(outcome_targets)
                    )
            self._labels.append(
                torch.tensor(outcome_targets, dtype=torch.long)
            )
            seq_idx += 1

        # If the loop did not produce a label for every base sequence,
        # pad the remainder with UNK so downstream code never sees a
        # size mismatch.
        while len(self._labels) < len(self.base.sequences):
            _, _, tgt = self.base.sequences[len(self._labels)]
            self._labels.append(
                torch.full((tgt.shape[0],), OUTCOME_UNK, dtype=torch.long)
            )

        n_total = sum(t.numel() for t in self._labels)
        n_unk = sum((t == OUTCOME_UNK).sum().item() for t in self._labels)
        logger.info(
            "OutcomeLabelledDataset: attached %d label tensors  "
            "(base sequences=%d, total tokens=%d, UNK=%d [%.2f%%])",
            len(self._labels), len(self.base.sequences),
            n_total, n_unk, 100.0 * n_unk / max(n_total, 1),
        )

    def __len__(self) -> int:
        return len(self.base.sequences)

    def __getitem__(self, idx: int):
        tokens, ctx, targets = self.base.sequences[idx]
        outcome_labels = self._labels[idx]
        return tokens, ctx, targets, outcome_labels


def _outcome_collate(batch):
    """Collate that extends ``_collate_fn`` with outcome labels."""
    tokens_batch, ctx_batch, target_batch, outcome_batch = [], [], [], []
    max_len = max(item[0].size(0) for item in batch)

    for tokens, ctx, target, outcomes in batch:
        seq_len = tokens.size(0)
        pad_len = max_len - seq_len

        tokens_padded = torch.cat(
            [tokens, torch.full((pad_len,), PAD_TOKEN, dtype=torch.long)]
        )
        target_padded = torch.cat(
            [target, torch.full((pad_len,), PAD_TOKEN, dtype=torch.long)]
        )
        outcomes_padded = torch.cat(
            [outcomes, torch.full((pad_len,), OUTCOME_UNK, dtype=torch.long)]
        )
        if pad_len > 0:
            ctx_padded = torch.cat([ctx, torch.zeros(pad_len, ctx.size(-1))])
        else:
            ctx_padded = ctx

        tokens_batch.append(tokens_padded)
        ctx_batch.append(ctx_padded)
        target_batch.append(target_padded)
        outcome_batch.append(outcomes_padded)

    return (
        torch.stack(tokens_batch),
        torch.stack(ctx_batch),
        torch.stack(target_batch),
        torch.stack(outcome_batch),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Training — option (a) joint
# ═════════════════════════════════════════════════════════════════════════════


def train_joint(
    backbone_init: PitchGPTModel,
    train_ds: OutcomeLabelledDataset,
    val_ds: OutcomeLabelledDataset | None,
    epochs: int,
    batch_size: int,
    lr: float,
    lambda_outcome: float,
    device: torch.device,
    max_batches_per_epoch: int | None = None,
) -> tuple[PitchGPTWithOutcomeHead, list[dict]]:
    """Train backbone + joint outcome head end-to-end.

    backbone_init is deep-copied so the caller's checkpoint stays intact.
    """
    backbone = copy.deepcopy(backbone_init).to(device)
    model = PitchGPTWithOutcomeHead(backbone).to(device)

    # Unfreeze everything — joint training.
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    token_crit = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    outcome_crit = nn.CrossEntropyLoss(ignore_index=OUTCOME_UNK)

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_outcome_collate,
    )

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_tok_loss, total_out_loss, total_batches = 0.0, 0.0, 0
        t0 = time.time()
        for batch_idx, (tokens, ctx, targets, outcomes) in enumerate(loader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break
            tokens = tokens.to(device)
            ctx = ctx.to(device)
            targets = targets.to(device)
            outcomes = outcomes.to(device)

            token_logits, outcome_logits = model(tokens, ctx)
            tok_loss = token_crit(
                token_logits.reshape(-1, VOCAB_SIZE),
                targets.reshape(-1),
            )
            out_loss = outcome_crit(
                outcome_logits.reshape(-1, NUM_OUTCOME_CLASSES),
                outcomes.reshape(-1),
            )
            loss = tok_loss + lambda_outcome * out_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_tok_loss += float(tok_loss.item())
            total_out_loss += float(out_loss.item())
            total_batches += 1

        avg_tok = total_tok_loss / max(total_batches, 1)
        avg_out = total_out_loss / max(total_batches, 1)
        dt = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_token_loss": round(avg_tok, 4),
            "train_outcome_loss": round(avg_out, 4),
            "seconds": round(dt, 1),
        })
        logger.info(
            "  [joint] epoch %d/%d  tok_loss=%.4f  out_loss=%.4f  %.1fs",
            epoch, epochs, avg_tok, avg_out, dt,
        )
    return model, history


# ═════════════════════════════════════════════════════════════════════════════
# Training — option (b) frozen backbone
# ═════════════════════════════════════════════════════════════════════════════


def train_frozen(
    backbone: PitchGPTModel,
    train_ds: OutcomeLabelledDataset,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    max_batches_per_epoch: int | None = None,
) -> tuple[FrozenOutcomeHead, list[dict]]:
    """Train the MLP head on a frozen backbone's hidden states."""
    freeze_backbone(backbone)
    backbone.eval()
    backbone.to(device)

    head = FrozenOutcomeHead(d_model=backbone.d_model).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(ignore_index=OUTCOME_UNK)

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_outcome_collate,
    )

    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        head.train()
        total_loss, total_batches = 0.0, 0
        t0 = time.time()
        for batch_idx, (tokens, ctx, _targets, outcomes) in enumerate(loader):
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break
            tokens = tokens.to(device)
            ctx = ctx.to(device)
            outcomes = outcomes.to(device)

            with torch.no_grad():
                hidden = extract_backbone_hidden_states(backbone, tokens, ctx)
            # hidden is detached already — forward through head tracks grad.
            logits = head(hidden)
            loss = crit(
                logits.reshape(-1, NUM_OUTCOME_CLASSES),
                outcomes.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_batches += 1

        avg = total_loss / max(total_batches, 1)
        dt = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_outcome_loss": round(avg, 4),
            "seconds": round(dt, 1),
        })
        logger.info(
            "  [frozen] epoch %d/%d  out_loss=%.4f  %.1fs",
            epoch, epochs, avg, dt,
        )
    return head, history


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def eval_joint(
    model: PitchGPTWithOutcomeHead,
    eval_ds: OutcomeLabelledDataset,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Compute 7-class log-loss + ECE, and backbone next-token ECE."""
    model.eval()
    loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_outcome_collate,
    )
    all_outcome_logits: list[np.ndarray] = []
    all_outcome_targets: list[np.ndarray] = []
    all_token_top1: list[np.ndarray] = []
    all_token_correct: list[np.ndarray] = []

    for tokens, ctx, targets, outcomes in loader:
        tokens = tokens.to(device)
        ctx = ctx.to(device)
        targets = targets.to(device)
        outcomes = outcomes.to(device)

        token_logits, outcome_logits = model(tokens, ctx)

        flat_out_logits = outcome_logits.reshape(-1, NUM_OUTCOME_CLASSES)
        flat_out_targets = outcomes.reshape(-1)
        out_valid = flat_out_targets != OUTCOME_UNK
        if int(out_valid.sum()) > 0:
            all_outcome_logits.append(flat_out_logits[out_valid].cpu().numpy())
            all_outcome_targets.append(flat_out_targets[out_valid].cpu().numpy())

        flat_tok_logits = token_logits.reshape(-1, VOCAB_SIZE)
        flat_tok_targets = targets.reshape(-1)
        tok_valid = flat_tok_targets != PAD_TOKEN
        if int(tok_valid.sum()) > 0:
            tok_logits_v = flat_tok_logits[tok_valid]
            tok_targets_v = flat_tok_targets[tok_valid]
            tok_probs = F.softmax(tok_logits_v, dim=-1)
            top1_p, top1_idx = tok_probs.max(dim=-1)
            tok_correct = (top1_idx == tok_targets_v).cpu().numpy().astype(bool)
            all_token_top1.append(top1_p.cpu().numpy())
            all_token_correct.append(tok_correct)

    if not all_outcome_logits:
        return {
            "log_loss": float("nan"),
            "ece": float("nan"),
            "backbone_token_ece": float("nan"),
            "n_tokens": 0,
            "n_outcome_labels": 0,
        }
    out_logits = np.concatenate(all_outcome_logits, axis=0)
    out_targets = np.concatenate(all_outcome_targets, axis=0)
    tok_top1 = np.concatenate(all_token_top1, axis=0)
    tok_correct = np.concatenate(all_token_correct, axis=0)

    ll = seven_class_log_loss(out_logits, out_targets)
    ece = seven_class_ece(out_logits, out_targets, n_bins=10)
    tok_curve = compute_reliability_curve(tok_top1, tok_correct, n_bins=10)
    tok_ece = expected_calibration_error(tok_curve)
    tok_acc = float(tok_correct.mean()) if tok_correct.size else 0.0

    # Per-class log-loss (to flag class-specific collapse).
    per_class_ll = {}
    for c in range(NUM_OUTCOME_CLASSES):
        mask = (out_targets == c)
        if mask.sum() == 0:
            per_class_ll[c] = None
        else:
            per_class_ll[c] = seven_class_log_loss(out_logits[mask], out_targets[mask])

    return {
        "log_loss": ll,
        "ece": ece,
        "backbone_token_ece": float(tok_ece),
        "backbone_token_accuracy": tok_acc,
        "n_tokens": int(tok_correct.size),
        "n_outcome_labels": int(out_targets.size),
        "per_class_log_loss": per_class_ll,
    }


@torch.no_grad()
def eval_frozen(
    backbone: PitchGPTModel,
    head: FrozenOutcomeHead,
    eval_ds: OutcomeLabelledDataset,
    batch_size: int,
    device: torch.device,
) -> dict:
    """Compute 7-class log-loss + ECE via frozen backbone + head.

    Backbone next-token ECE is also computed (it is *by construction* the
    v2 checkpoint's ECE because the head cannot perturb it, but we
    measure on the same eval slice for apples-to-apples reporting).
    """
    backbone.eval()
    head.eval()
    backbone.to(device)
    head.to(device)

    loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_outcome_collate,
    )
    all_outcome_logits: list[np.ndarray] = []
    all_outcome_targets: list[np.ndarray] = []
    all_token_top1: list[np.ndarray] = []
    all_token_correct: list[np.ndarray] = []

    for tokens, ctx, targets, outcomes in loader:
        tokens = tokens.to(device)
        ctx = ctx.to(device)
        targets = targets.to(device)
        outcomes = outcomes.to(device)

        hidden = extract_backbone_hidden_states(backbone, tokens, ctx)
        outcome_logits = head(hidden)
        token_logits = backbone.output_head(hidden)

        flat_out_logits = outcome_logits.reshape(-1, NUM_OUTCOME_CLASSES)
        flat_out_targets = outcomes.reshape(-1)
        out_valid = flat_out_targets != OUTCOME_UNK
        if int(out_valid.sum()) > 0:
            all_outcome_logits.append(flat_out_logits[out_valid].cpu().numpy())
            all_outcome_targets.append(flat_out_targets[out_valid].cpu().numpy())

        flat_tok_logits = token_logits.reshape(-1, VOCAB_SIZE)
        flat_tok_targets = targets.reshape(-1)
        tok_valid = flat_tok_targets != PAD_TOKEN
        if int(tok_valid.sum()) > 0:
            tok_logits_v = flat_tok_logits[tok_valid]
            tok_targets_v = flat_tok_targets[tok_valid]
            tok_probs = F.softmax(tok_logits_v, dim=-1)
            top1_p, top1_idx = tok_probs.max(dim=-1)
            tok_correct = (top1_idx == tok_targets_v).cpu().numpy().astype(bool)
            all_token_top1.append(top1_p.cpu().numpy())
            all_token_correct.append(tok_correct)

    if not all_outcome_logits:
        return {
            "log_loss": float("nan"),
            "ece": float("nan"),
            "backbone_token_ece": float("nan"),
            "n_tokens": 0,
            "n_outcome_labels": 0,
        }
    out_logits = np.concatenate(all_outcome_logits, axis=0)
    out_targets = np.concatenate(all_outcome_targets, axis=0)
    tok_top1 = np.concatenate(all_token_top1, axis=0)
    tok_correct = np.concatenate(all_token_correct, axis=0)

    ll = seven_class_log_loss(out_logits, out_targets)
    ece = seven_class_ece(out_logits, out_targets, n_bins=10)
    tok_curve = compute_reliability_curve(tok_top1, tok_correct, n_bins=10)
    tok_ece = expected_calibration_error(tok_curve)
    tok_acc = float(tok_correct.mean()) if tok_correct.size else 0.0

    per_class_ll = {}
    for c in range(NUM_OUTCOME_CLASSES):
        mask = (out_targets == c)
        if mask.sum() == 0:
            per_class_ll[c] = None
        else:
            per_class_ll[c] = seven_class_log_loss(out_logits[mask], out_targets[mask])

    return {
        "log_loss": ll,
        "ece": ece,
        "backbone_token_ece": float(tok_ece),
        "backbone_token_accuracy": tok_acc,
        "n_tokens": int(tok_correct.size),
        "n_outcome_labels": int(out_targets.size),
        "per_class_log_loss": per_class_ll,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-games", type=int, default=DEFAULT_TRAIN_GAMES)
    p.add_argument("--eval-games", type=int, default=DEFAULT_EVAL_GAMES)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--epochs-joint", type=int, default=DEFAULT_EPOCHS_JOINT)
    p.add_argument("--epochs-frozen", type=int, default=DEFAULT_EPOCHS_FROZEN)
    p.add_argument("--lambda-outcome", type=float, default=DEFAULT_LAMBDA)
    p.add_argument("--max-batches-per-epoch", type=int, default=None,
                   help="cap batches/epoch for GPU time-slicing (joint + frozen).")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--skip-joint", action="store_true",
                   help="skip option (a) (joint) — useful if GPU is contended")
    p.add_argument("--skip-frozen", action="store_true",
                   help="skip option (b) (frozen)")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = _get_device()
    logger.info("device=%s  train_games=%d  eval_games=%d  batch=%d  seed=%d",
                device, args.train_games, args.eval_games, args.batch_size, args.seed)

    # -------- Load v2 backbone (base for both options) --------------------
    logger.info("Loading PitchGPT v2 backbone from models/pitchgpt_v2.pt")
    backbone_v2 = _load_model(version="2")
    logger.info("Backbone loaded: d_model=%d, context_dim=%d",
                backbone_v2.d_model, backbone_v2.context_dim)

    # -------- Build pitcher-disjoint train + eval cohorts -----------------
    conn = get_connection(read_only=True)
    try:
        train_pitchers = PitchSequenceDataset.fetch_pitcher_ids_for_seasons(
            conn, seasons=list(range(TRAIN_RANGE[0], TRAIN_RANGE[1] + 1)),
        )
        logger.info("Train cohort candidate pitchers: %d", len(train_pitchers))

        # Train: 500-game sample from 2015-2022 (no pitcher-disjoint because
        # we are not doing OOS here — train and eval are from *different*
        # season ranges and we want eval's pitcher-disjointness, not train's).
        logger.info("Building train dataset (seasons %s, max_games=%d)...",
                    list(range(TRAIN_RANGE[0], TRAIN_RANGE[1] + 1)), args.train_games)
        base_train = PitchSequenceDataset(
            conn,
            seasons=list(range(TRAIN_RANGE[0], TRAIN_RANGE[1] + 1)),
            max_games=args.train_games,
            context_dim=backbone_v2.context_dim,
        )
        logger.info("Train sequences: %d", len(base_train))

        # Eval: 100-game sample from 2023 filtered to pitcher-disjoint
        # (pitchers NOT in train cohort).
        logger.info("Building eval dataset (seasons %s, pitcher-disjoint, max_games=%d)...",
                    list(range(EVAL_RANGE[0], EVAL_RANGE[1] + 1)), args.eval_games)
        base_eval = PitchSequenceDataset(
            conn,
            seasons=list(range(EVAL_RANGE[0], EVAL_RANGE[1] + 1)),
            max_games=args.eval_games,
            exclude_pitcher_ids=train_pitchers,
            context_dim=backbone_v2.context_dim,
        )
        logger.info("Eval sequences: %d  (%d unique pitchers)",
                    len(base_eval), len(base_eval.pitcher_ids))

        # Attach outcome labels.
        logger.info("Attaching outcome labels to train...")
        train_ds = OutcomeLabelledDataset(base_train, conn)
        logger.info("Attaching outcome labels to eval...")
        eval_ds = OutcomeLabelledDataset(base_eval, conn)
    finally:
        conn.close()

    # -------- Baseline: frequency prior log-loss on eval ------------------
    eval_outcome_targets = np.concatenate(
        [lbl.cpu().numpy() for lbl in eval_ds._labels]
    )
    eval_valid_outcomes = eval_outcome_targets[eval_outcome_targets != OUTCOME_UNK]
    train_outcome_targets = np.concatenate(
        [lbl.cpu().numpy() for lbl in train_ds._labels]
    )
    train_valid_outcomes = train_outcome_targets[train_outcome_targets != OUTCOME_UNK]
    prior = class_frequency_prior(train_valid_outcomes)
    prior_ll = prior_log_loss(eval_valid_outcomes, prior=prior)
    uniform_ll = float(-np.log(1.0 / NUM_OUTCOME_CLASSES))
    logger.info("Eval valid outcomes: %d / %d  (UNK dropped)",
                len(eval_valid_outcomes), len(eval_outcome_targets))
    logger.info("Baselines — uniform ll=%.4f  train-prior ll=%.4f",
                uniform_ll, prior_ll)
    # Class distribution sanity for both splits.
    train_dist = np.bincount(train_valid_outcomes, minlength=NUM_OUTCOME_CLASSES) / max(
        len(train_valid_outcomes), 1,
    )
    eval_dist = np.bincount(eval_valid_outcomes, minlength=NUM_OUTCOME_CLASSES) / max(
        len(eval_valid_outcomes), 1,
    )
    logger.info("Train outcome distribution:  %s",
                {c: round(float(train_dist[i]), 4) for i, c in enumerate(
                    ("ball","called_strike","swinging_strike","foul","in_play_out","in_play_hit","hbp")
                )})
    logger.info("Eval  outcome distribution:  %s",
                {c: round(float(eval_dist[i]), 4) for i, c in enumerate(
                    ("ball","called_strike","swinging_strike","foul","in_play_out","in_play_hit","hbp")
                )})

    # -------- Also measure backbone ECE on the v2 checkpoint alone --------
    # (Reference number for the +0.005 degradation budget.)
    logger.info("Measuring v2 backbone token ECE on eval slice (reference)...")
    with torch.no_grad():
        backbone_v2.eval()
        backbone_v2.to(device)
        loader = DataLoader(
            eval_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=_outcome_collate,
        )
        top1_p_list, correct_list = [], []
        for tokens, ctx, targets, _outcomes in loader:
            tokens = tokens.to(device)
            ctx = ctx.to(device)
            targets = targets.to(device)
            logits = backbone_v2(tokens, ctx)
            flat = logits.reshape(-1, VOCAB_SIZE)
            flat_t = targets.reshape(-1)
            valid = flat_t != PAD_TOKEN
            if int(valid.sum()) == 0:
                continue
            flat = flat[valid]; flat_t = flat_t[valid]
            probs = F.softmax(flat, dim=-1)
            top1_p, top1_idx = probs.max(dim=-1)
            top1_p_list.append(top1_p.cpu().numpy())
            correct_list.append((top1_idx == flat_t).cpu().numpy().astype(bool))
    if top1_p_list:
        ref_top1 = np.concatenate(top1_p_list)
        ref_corr = np.concatenate(correct_list)
        ref_curve = compute_reliability_curve(ref_top1, ref_corr, n_bins=10)
        ref_ece = float(expected_calibration_error(ref_curve))
        ref_acc = float(ref_corr.mean())
    else:
        ref_ece, ref_acc = float("nan"), float("nan")
    logger.info("Reference v2 backbone on eval: ECE=%.4f  acc=%.4f", ref_ece, ref_acc)

    results: dict[str, Any] = {
        "config": {
            "train_games": args.train_games,
            "eval_games": args.eval_games,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs_joint": args.epochs_joint,
            "epochs_frozen": args.epochs_frozen,
            "lambda_outcome": args.lambda_outcome,
            "train_range": TRAIN_RANGE,
            "eval_range": EVAL_RANGE,
            "seed": args.seed,
            "device": str(device),
        },
        "baselines": {
            "uniform_log_loss": uniform_ll,
            "train_prior_log_loss": prior_ll,
            "train_outcome_distribution": {
                c: float(train_dist[i]) for i, c in enumerate((
                    "ball","called_strike","swinging_strike","foul",
                    "in_play_out","in_play_hit","hbp",
                ))
            },
            "eval_outcome_distribution": {
                c: float(eval_dist[i]) for i, c in enumerate((
                    "ball","called_strike","swinging_strike","foul",
                    "in_play_out","in_play_hit","hbp",
                ))
            },
            "reference_v2_backbone_ece": ref_ece,
            "reference_v2_backbone_accuracy": ref_acc,
            "n_train_valid_outcomes": int(len(train_valid_outcomes)),
            "n_eval_valid_outcomes": int(len(eval_valid_outcomes)),
        },
        "joint": None,
        "frozen": None,
    }

    # -------- Option (a): joint -----------------------------------------
    if not args.skip_joint:
        logger.info("═══ Option (a): JOINT (backbone unfrozen + aux head) ═══")
        torch.cuda.empty_cache() if device.type == "cuda" else None
        t0 = time.time()
        joint_model, joint_hist = train_joint(
            backbone_v2, train_ds, None,
            epochs=args.epochs_joint,
            batch_size=args.batch_size,
            lr=args.lr,
            lambda_outcome=args.lambda_outcome,
            device=device,
            max_batches_per_epoch=args.max_batches_per_epoch,
        )
        train_dt = time.time() - t0
        logger.info("Saving joint smoke checkpoint to %s", CHKPT_JOINT)
        torch.save({
            "model_state_dict": joint_model.state_dict(),
            "config": {
                "d_model": joint_model.backbone.d_model,
                "context_dim": joint_model.backbone.context_dim,
                "n_classes": NUM_OUTCOME_CLASSES,
            },
        }, CHKPT_JOINT)

        logger.info("Evaluating joint model on %d-game eval slice...", args.eval_games)
        joint_eval = eval_joint(joint_model, eval_ds, args.batch_size, device)
        joint_eval["train_seconds"] = round(train_dt, 1)
        joint_eval["history"] = joint_hist
        results["joint"] = joint_eval
        logger.info(
            "  JOINT  ll=%.4f  ece=%.4f  backbone_tok_ece=%.4f  acc=%.4f  n=%d",
            joint_eval["log_loss"], joint_eval["ece"],
            joint_eval["backbone_token_ece"],
            joint_eval["backbone_token_accuracy"],
            joint_eval["n_outcome_labels"],
        )
        del joint_model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # -------- Option (b): frozen ----------------------------------------
    if not args.skip_frozen:
        logger.info("═══ Option (b): FROZEN (backbone frozen + MLP head) ═══")
        # Deep-copy to keep the reference backbone intact for the ref ECE
        # measurement (though we already did it above).  Saves on re-loads.
        frozen_backbone = copy.deepcopy(backbone_v2).to(device)
        t0 = time.time()
        frozen_head, frozen_hist = train_frozen(
            frozen_backbone, train_ds,
            epochs=args.epochs_frozen,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            max_batches_per_epoch=args.max_batches_per_epoch,
        )
        train_dt = time.time() - t0
        logger.info("Saving frozen smoke checkpoint to %s", CHKPT_FROZEN)
        torch.save({
            "head_state_dict": frozen_head.state_dict(),
            "config": {
                "d_model": frozen_backbone.d_model,
                "hidden_dim": 64,
                "n_classes": NUM_OUTCOME_CLASSES,
            },
        }, CHKPT_FROZEN)

        logger.info("Evaluating frozen model on %d-game eval slice...", args.eval_games)
        frozen_eval = eval_frozen(frozen_backbone, frozen_head, eval_ds,
                                  args.batch_size, device)
        frozen_eval["train_seconds"] = round(train_dt, 1)
        frozen_eval["history"] = frozen_hist
        results["frozen"] = frozen_eval
        logger.info(
            "  FROZEN  ll=%.4f  ece=%.4f  backbone_tok_ece=%.4f  acc=%.4f  n=%d",
            frozen_eval["log_loss"], frozen_eval["ece"],
            frozen_eval["backbone_token_ece"],
            frozen_eval["backbone_token_accuracy"],
            frozen_eval["n_outcome_labels"],
        )
        del frozen_backbone, frozen_head
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # -------- Verdict ---------------------------------------------------
    verdict = _compute_verdict(results, ref_ece)
    results["verdict"] = verdict

    out_path = OUT_DIR / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)
    logger.info("Wrote %s", out_path)

    logger.info("═══ Verdict: %s ═══", verdict["decision"])
    for line in verdict["rationale"]:
        logger.info("  %s", line)

    return 0


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _compute_verdict(results: dict, ref_ece: float) -> dict:
    """Decide joint vs frozen per the locked criterion.

    Lower 7-class log-loss wins, subject to backbone-token-ECE budget of
    +0.005 absolute over the reference v2 ECE on the same eval slice.
    """
    j = results.get("joint")
    f_ = results.get("frozen")
    ece_budget = 0.005
    decision = "UNDECIDED"
    rationale: list[str] = []

    if f_ is None and j is None:
        return {"decision": "BOTH_SKIPPED", "rationale": ["both options skipped"]}
    if f_ is None:
        decision = "JOINT (only option run)"
        rationale.append(f"frozen skipped; joint ll={j['log_loss']:.4f}, ece={j['ece']:.4f}")
        return {"decision": decision, "rationale": rationale}
    if j is None:
        decision = "FROZEN (only option run)"
        rationale.append(f"joint skipped; frozen ll={f_['log_loss']:.4f}, ece={f_['ece']:.4f}")
        return {"decision": decision, "rationale": rationale}

    # Joint degradation check.
    joint_ece_degrade = j["backbone_token_ece"] - ref_ece
    rationale.append(
        f"ref v2 backbone token ECE = {ref_ece:.4f} (eval slice); budget = +{ece_budget:.4f}"
    )
    rationale.append(
        f"joint  backbone token ECE = {j['backbone_token_ece']:.4f}  "
        f"(delta vs ref = {joint_ece_degrade:+.4f})"
    )
    rationale.append(
        f"frozen backbone token ECE = {f_['backbone_token_ece']:.4f}  "
        f"(delta vs ref = {f_['backbone_token_ece'] - ref_ece:+.4f})"
    )
    rationale.append(
        f"joint  7-class log-loss   = {j['log_loss']:.4f}  (ece={j['ece']:.4f})"
    )
    rationale.append(
        f"frozen 7-class log-loss   = {f_['log_loss']:.4f}  (ece={f_['ece']:.4f})"
    )

    joint_eligible = joint_ece_degrade <= ece_budget
    if not joint_eligible:
        decision = "FROZEN"
        rationale.append(
            f"joint exceeded ECE budget ({joint_ece_degrade:+.4f} > +{ece_budget:.4f}); "
            "falling back to frozen per §4.1 hard constraint."
        )
    else:
        # Compare 7-class log-loss.
        delta_ll = j["log_loss"] - f_["log_loss"]
        if delta_ll < -0.005:
            decision = "JOINT"
            rationale.append(
                f"joint wins on 7-class log-loss by {-delta_ll:.4f} (> 0.005 threshold); "
                "joint is within ECE budget."
            )
        elif delta_ll > 0.005:
            decision = "FROZEN"
            rationale.append(
                f"frozen wins on 7-class log-loss by {delta_ll:.4f} (> 0.005 threshold)."
            )
        else:
            # Tie-breaker: pick frozen (preserves paper checkpoint guarantees).
            decision = "FROZEN (tie-breaker: calibration preservation)"
            rationale.append(
                f"log-loss delta is within noise (|{delta_ll:.4f}| <= 0.005); "
                "tie-breaker favours frozen because it preserves the v2 checkpoint's "
                "calibration guarantees exactly and cannot cause downstream drift."
            )
    return {"decision": decision, "rationale": rationale,
            "joint_ece_degradation": joint_ece_degrade,
            "frozen_ece_degradation": f_["backbone_token_ece"] - ref_ece,
            "log_loss_delta_joint_minus_frozen": j["log_loss"] - f_["log_loss"]}


if __name__ == "__main__":
    sys.exit(main())
