"""
PitchGPT downstream-utility evaluation.

Question: does PitchGPT's calibrated per-pitch predicted distribution
drive better downstream decisions than the LSTM baseline's distribution,
even though PitchGPT's raw perplexity edge over LSTM is only 13.80%
(below the 15% spec gate)?

Test
----
* Load the pitcher-disjoint PitchGPT checkpoint (trained 2015-2022).
* Re-train a fresh LSTM baseline on the same pitcher-disjoint training
  data (same hyperparameters as ``scripts/pitchgpt_2025_holdout.py``).
* Score a 2025 pitcher-disjoint holdout (identical construction as the
  2025_holdout run).  For every non-PAD target pitch, compute
  streaming-summarised features over the 2,210-token vocabulary:
     - pitch-type marginals (17 probs)
     - zone marginals (26 probs)
     - velo-bucket marginals (5 probs)
     - top-1 prob, top-1 log-prob, predictive entropy
  (=51 dims per model) — the full (N, 2210) distribution is never
  materialised, which keeps peak memory under 2 GB.
* Train three XGBoost classifiers on the 2015-2022 train split:
     1. ``null`` — situational-context only (34 dims).
     2. ``pitchgpt`` — context + PitchGPT distribution summary (85 dims).
     3. ``lstm`` — context + LSTM distribution summary (85 dims).
* Score each on the 2025 holdout.  Bootstrap 95% CIs on log-loss /
  Brier / accuracy for each variant AND on pairwise deltas.
* Restrict to high-confidence (PitchGPT top-1 prob > 0.5) pitches and
  repeat.

Ground-truth outcome buckets
----------------------------
The per-pitch ``description`` column is mapped into 5 buckets:

    called_strike, ball, foul, in_play, swinging_strike

with these fold-ins:
    blocked_ball                    -> ball
    hit_by_pitch                    -> ball
    pitchout                        -> ball
    foul_tip, foul_bunt,
      bunt_foul_tip                 -> foul
    swinging_strike_blocked,
      missed_bunt                   -> swinging_strike
    hit_into_play                   -> in_play

Any description outside this set is dropped from the downstream training
and eval sets (counts are near-zero in 2015-2025 anyway).

Outputs
-------
Written to ``results/pitchgpt/downstream_utility/``:

    downstream_comparison.json          -- headline metrics + CIs
    calibration_utility_subset.json     -- metrics restricted to the
                                           high-confidence PitchGPT subset
    report.md                           -- methodology-paper-section write-up
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
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import log_loss

try:
    import xgboost as xgb
except ImportError as exc:
    raise SystemExit(
        "xgboost is required for this script. pip install xgboost"
    ) from exc

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt import (  # noqa: E402
    CONTEXT_DIM,
    NUM_PITCH_TYPES,
    NUM_VELO_BUCKETS,
    NUM_ZONES,
    PAD_TOKEN,
    PitchGPTModel,
    PitchSequenceDataset,
    PitchTokenizer,
    TOTAL_VOCAB,
    VOCAB_SIZE,
    _collate_fn,
    _compute_per_pitch_score_diff,
    _get_device,
    _safe_bool,
    _safe_int,
    _safe_str,
    audit_no_game_overlap,
)
from src.analytics.pitch_lstm import PitchLSTMNetwork  # noqa: E402
from src.db.schema import get_connection  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitchgpt_downstream_utility")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_RANGE = (2015, 2022)
VAL_RANGE = (2023, 2023)
HOLDOUT_RANGE = (2025, 2025)

DEFAULT_BATCH = 32
DEFAULT_LR = 1e-3
GRAD_CLIP = 1.0

# 5-class pitch-outcome bucket
OUTCOME_CLASSES = ["called_strike", "ball", "foul", "in_play", "swinging_strike"]
_DESCRIPTION_TO_BUCKET = {
    "called_strike": "called_strike",
    "ball": "ball",
    "blocked_ball": "ball",
    "hit_by_pitch": "ball",
    "pitchout": "ball",
    "foul": "foul",
    "foul_tip": "foul",
    "foul_bunt": "foul",
    "bunt_foul_tip": "foul",
    "hit_into_play": "in_play",
    "swinging_strike": "swinging_strike",
    "swinging_strike_blocked": "swinging_strike",
    "missed_bunt": "swinging_strike",
}
_OUTCOME_TO_IDX = {c: i for i, c in enumerate(OUTCOME_CLASSES)}
_OUTCOME_SKIP = -1  # position sentinel

SUMMARY_DIM = (
    NUM_PITCH_TYPES    # 17 pitch-type marginal probs
    + NUM_ZONES        # 26 zone marginal probs
    + NUM_VELO_BUCKETS # 5 velo marginal probs
    + 3                # top-1 prob, top-1 log-prob, entropy
)  # 17 + 26 + 5 + 3 = 51


# Precompute indexed mapping (token -> (pt, zone, velo)) so we can scatter-
# aggregate marginals with index_add on-device, avoiding any (N, 2210)
# materialisation.
def _build_vocab_index() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens = torch.arange(VOCAB_SIZE, dtype=torch.long)
    velo = tokens % NUM_VELO_BUCKETS
    zone = (tokens // NUM_VELO_BUCKETS) % NUM_ZONES
    pt = tokens // (NUM_VELO_BUCKETS * NUM_ZONES)
    return pt, zone, velo


_PT_IDX_CPU, _ZONE_IDX_CPU, _VELO_IDX_CPU = _build_vocab_index()


def _summarise_probs_torch(probs: torch.Tensor) -> torch.Tensor:
    """Return (N, SUMMARY_DIM) summary of per-pitch distributions without
    materialising any intermediate (N, V') tensor besides the input.

    ``probs`` is ``(N, VOCAB_SIZE)`` on the same device as the model.
    """
    if probs.numel() == 0:
        return probs.new_zeros((0, SUMMARY_DIM))

    device = probs.device
    pt_idx = _PT_IDX_CPU.to(device)
    zone_idx = _ZONE_IDX_CPU.to(device)
    velo_idx = _VELO_IDX_CPU.to(device)

    # scatter-add along the class axis to aggregate marginals.
    pt_m = probs.new_zeros((probs.shape[0], NUM_PITCH_TYPES))
    zone_m = probs.new_zeros((probs.shape[0], NUM_ZONES))
    velo_m = probs.new_zeros((probs.shape[0], NUM_VELO_BUCKETS))
    pt_m.scatter_add_(1, pt_idx.unsqueeze(0).expand_as(probs), probs)
    zone_m.scatter_add_(1, zone_idx.unsqueeze(0).expand_as(probs), probs)
    velo_m.scatter_add_(1, velo_idx.unsqueeze(0).expand_as(probs), probs)

    top1_prob, _ = probs.max(dim=1, keepdim=True)
    top1_logprob = torch.log(torch.clamp(top1_prob, min=1e-12))
    ent = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=1, keepdim=True)

    return torch.cat([pt_m, zone_m, velo_m, top1_prob, top1_logprob, ent], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Labeled dataset — mirrors PitchSequenceDataset but also emits outcome indices.
# ─────────────────────────────────────────────────────────────────────────────


class LabeledPitchSequenceDataset(Dataset):
    """Like ``PitchSequenceDataset`` but also carries per-target outcome
    indices (5-class pitch-outcome bucket from the ``description`` column).

    ``__getitem__`` returns ``(tokens, context, target_tokens, target_outcomes)``.
    ``target_outcomes[i]`` is the outcome idx of the pitch at
    ``target_tokens[i]``, or -1 if it falls outside the supported bucket
    set.  The collate fn pads outcomes with -1.

    IMPORTANT: to guarantee sequence construction identical to
    ``PitchSequenceDataset`` (and thus identical tokens / contexts / target
    alignment, so the existing PitchGPT checkpoint scores exactly as it
    would have in the 2025 holdout run), we reuse the same SQL / grouping
    /truncation logic.
    """

    def __init__(
        self,
        conn,
        seasons,
        max_seq_len: int,
        max_games: int | None,
        exclude_pitcher_ids: set[int] | None,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.game_pks: set[int] = set()
        self.pitcher_ids: set[int] = set()
        excl = set(int(p) for p in exclude_pitcher_ids) if exclude_pitcher_ids else set()

        season_filter = ""
        seasons_list = list(seasons) if seasons is not None else []
        if seasons_list:
            s_str = ", ".join(str(int(s)) for s in seasons_list)
            season_filter = f"AND EXTRACT(YEAR FROM game_date) IN ({s_str})"

        pitcher_exclude = ""
        if excl:
            pids_str = ", ".join(str(int(p)) for p in excl)
            pitcher_exclude = f"AND pitcher_id NOT IN ({pids_str})"

        game_filter = ""
        if max_games is not None:
            game_filter = f"""
                AND game_pk IN (
                    SELECT game_pk FROM (
                        SELECT DISTINCT game_pk
                        FROM pitches
                        WHERE pitch_type IS NOT NULL {season_filter}
                          {pitcher_exclude}
                    ) USING SAMPLE {int(max_games)} ROWS
                )
            """

        query = f"""
            SELECT game_pk, pitcher_id, pitch_type, plate_x, plate_z,
                   release_speed, balls, strikes, outs_when_up,
                   on_1b, on_2b, on_3b, stand, inning,
                   inning_topbot, events, delta_run_exp,
                   at_bat_number, pitch_number, description
            FROM pitches
            WHERE pitch_type IS NOT NULL
              {season_filter}
              {pitcher_exclude}
              {game_filter}
            ORDER BY game_pk, at_bat_number, pitch_number
        """
        df = conn.execute(query).fetchdf()
        if df.empty:
            logger.warning("LabeledPitchSequenceDataset: no rows.")
            return

        df = df.assign(_score_diff=_compute_per_pitch_score_diff(df))
        grouped = df.groupby(["game_pk", "pitcher_id"], sort=False)
        for _key, game_df in grouped:
            if len(game_df) < 2:
                continue
            try:
                _pid = int(_key[1])
            except (TypeError, ValueError):
                _pid = -1
            if excl and _pid in excl:
                continue
            try:
                self.game_pks.add(int(_key[0]))
                self.pitcher_ids.add(_pid)
            except (TypeError, ValueError):
                pass

            tokens: list[int] = []
            contexts: list[torch.Tensor] = []
            outcomes: list[int] = []

            for _, row in game_df.iterrows():
                tok = PitchTokenizer.encode(
                    row.get("pitch_type"),
                    row.get("plate_x"),
                    row.get("plate_z"),
                    row.get("release_speed"),
                )
                tokens.append(tok)

                ctx_list = PitchTokenizer.encode_context(
                    balls=_safe_int(row.get("balls"), 0),
                    strikes=_safe_int(row.get("strikes"), 0),
                    outs=_safe_int(row.get("outs_when_up"), 0),
                    on_1b=_safe_bool(row.get("on_1b")),
                    on_2b=_safe_bool(row.get("on_2b")),
                    on_3b=_safe_bool(row.get("on_3b")),
                    stand=_safe_str(row.get("stand"), "R"),
                    inning=_safe_int(row.get("inning"), 1),
                    score_diff=_safe_int(row.get("_score_diff"), 0),
                )
                contexts.append(PitchTokenizer.context_to_tensor(ctx_list))

                desc = row.get("description")
                bucket = _DESCRIPTION_TO_BUCKET.get(str(desc) if desc is not None else "", None)
                outcomes.append(_OUTCOME_TO_IDX[bucket] if bucket is not None else _OUTCOME_SKIP)

            tokens = tokens[: self.max_seq_len]
            contexts = contexts[: self.max_seq_len]
            outcomes = outcomes[: self.max_seq_len]

            input_tokens = torch.tensor(tokens[:-1], dtype=torch.long)
            target_tokens = torch.tensor(tokens[1:], dtype=torch.long)
            target_outcomes = torch.tensor(outcomes[1:], dtype=torch.long)
            context_tensor = torch.stack(contexts[:-1])

            self.sequences.append((input_tokens, context_tensor, target_tokens, target_outcomes))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def _labeled_collate(batch):
    """Pad tokens/context/target as _collate_fn does, plus outcomes with -1."""
    max_len = max(item[0].size(0) for item in batch)
    tokens_b, ctx_b, target_b, outcome_b = [], [], [], []
    for tokens, ctx, target, outcomes in batch:
        seq_len = tokens.size(0)
        pad = max_len - seq_len
        tokens_padded = torch.cat([tokens, torch.full((pad,), PAD_TOKEN, dtype=torch.long)])
        target_padded = torch.cat([target, torch.full((pad,), PAD_TOKEN, dtype=torch.long)])
        outcomes_padded = torch.cat([outcomes, torch.full((pad,), _OUTCOME_SKIP, dtype=torch.long)])
        if pad > 0:
            ctx_padded = torch.cat([ctx, torch.zeros(pad, CONTEXT_DIM)])
        else:
            ctx_padded = ctx
        tokens_b.append(tokens_padded)
        ctx_b.append(ctx_padded)
        target_b.append(target_padded)
        outcome_b.append(outcomes_padded)
    return (torch.stack(tokens_b), torch.stack(ctx_b), torch.stack(target_b), torch.stack(outcome_b))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_pitchgpt(path: Path, device: torch.device) -> PitchGPTModel:
    ck = torch.load(str(path), map_location=device, weights_only=True)
    cfg = ck["config"]
    model = PitchGPTModel(
        vocab_size=cfg.get("vocab_size", TOTAL_VOCAB),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
    )
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _train_lstm(
    train_ds: PitchSequenceDataset,
    val_ds: PitchSequenceDataset,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> tuple[PitchLSTMNetwork, dict]:
    """Identical training loop to scripts/pitchgpt_2025_holdout.train_lstm."""
    _set_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn)

    model = PitchLSTMNetwork().to(device)
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
            tokens, ctx, target = tokens.to(device), ctx.to(device), target.to(device)
            logits = model(tokens, ctx)
            loss = criterion(logits.reshape(-1, model.output_vocab), target.reshape(-1))
            optimizer.zero_grad(); loss.backward()
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
                tokens, ctx, target = tokens.to(device), ctx.to(device), target.to(device)
                logits = model(tokens, ctx)
                loss = criterion(logits.reshape(-1, model.output_vocab), target.reshape(-1))
                n_tok = (target != PAD_TOKEN).sum().item()
                v_loss += loss.item() * n_tok; v_tok += n_tok
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
            epoch, epochs, train_loss, entry["train_ppl"], val_loss, entry["val_ppl"], dt,
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
# Batched summary scoring — streams probs through the summary function.
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def score_summaries(
    model: nn.Module,
    dataset: LabeledPitchSequenceDataset,
    device: torch.device,
    output_vocab: int,
    batch_size: int,
    progress_label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run ``model`` over the labeled dataset and return
    ``(summary, context, outcomes, top1_prob)``, with:
        * summary: (N_valid, SUMMARY_DIM) float32 — 51-d per-pitch summary of
          the predictive distribution.
        * context: (N_valid, CONTEXT_DIM) float32 — situational-context
          one-hot at the input position (i.e. tokens[:-1]).  This is the
          context at the *target* pitch from the model's input feed.
        * outcomes: (N_valid,) int64 — 5-class bucket index.
        * top1_prob: (N_valid,) float32.

    Only positions where ``outcomes != -1`` AND ``target != PAD_TOKEN`` are
    retained.  (PAD positions are already guaranteed to have outcome=-1 via
    ``_labeled_collate``.)
    """
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_labeled_collate)
    logger.info("[%s] batched scoring — %d sequences", progress_label, len(dataset))

    summaries: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    outcomes: list[np.ndarray] = []
    top1s: list[np.ndarray] = []

    total_batches = len(loader)
    log_every = max(total_batches // 20, 1)

    for i, (tokens, ctx, target, outcome) in enumerate(loader):
        tokens = tokens.to(device)
        ctx = ctx.to(device)
        target = target.to(device)
        outcome = outcome.to(device)

        logits = model(tokens, ctx)  # (B, S, V_out)
        if logits.shape[-1] != output_vocab:
            logits = logits[..., :output_vocab]
        probs = F.softmax(logits, dim=-1)

        flat_probs = probs.reshape(-1, output_vocab)
        flat_ctx = ctx.reshape(-1, CONTEXT_DIM)
        flat_target = target.reshape(-1)
        flat_outcome = outcome.reshape(-1)

        # Validity: non-PAD target AND non-skip outcome.
        valid = (flat_target != PAD_TOKEN) & (flat_outcome != _OUTCOME_SKIP)
        if valid.sum().item() == 0:
            continue

        v_probs = flat_probs[valid]
        v_ctx = flat_ctx[valid]
        v_outcome = flat_outcome[valid]

        summary = _summarise_probs_torch(v_probs)
        top1 = v_probs.max(dim=1).values

        summaries.append(summary.cpu().numpy().astype(np.float32))
        contexts.append(v_ctx.cpu().numpy().astype(np.float32))
        outcomes.append(v_outcome.cpu().numpy().astype(np.int64))
        top1s.append(top1.cpu().numpy().astype(np.float32))

        if (i + 1) % log_every == 0:
            logger.info("[%s] batch %d / %d", progress_label, i + 1, total_batches)

    summary_arr = np.concatenate(summaries, axis=0) if summaries else np.zeros((0, SUMMARY_DIM), dtype=np.float32)
    ctx_arr = np.concatenate(contexts, axis=0) if contexts else np.zeros((0, CONTEXT_DIM), dtype=np.float32)
    out_arr = np.concatenate(outcomes, axis=0) if outcomes else np.zeros((0,), dtype=np.int64)
    top1_arr = np.concatenate(top1s, axis=0) if top1s else np.zeros((0,), dtype=np.float32)

    logger.info("[%s] done — %d valid targets", progress_label, len(out_arr))
    return summary_arr, ctx_arr, out_arr, top1_arr


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost + metrics
# ─────────────────────────────────────────────────────────────────────────────


def _fit_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    seed: int,
    variant_name: str,
) -> xgb.XGBClassifier:
    clf = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=seed,
        n_jobs=0,
        early_stopping_rounds=20,
    )
    t0 = time.perf_counter()
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    dt = time.perf_counter() - t0
    logger.info("[%s] xgb fit in %.1fs  (n_train=%d  best_iter=%s)",
                variant_name, dt, len(X_train), getattr(clf, "best_iteration", "n/a"))
    return clf


def _metric_pack(probs: np.ndarray, y: np.ndarray, n_classes: int) -> dict:
    if len(y) == 0:
        return {"log_loss": float("nan"), "brier": float("nan"), "accuracy": float("nan"), "n": 0}
    ll = log_loss(y, probs, labels=list(range(n_classes)))
    onehot = np.zeros_like(probs, dtype=np.float32)
    onehot[np.arange(len(y)), y] = 1.0
    brier = float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))
    acc = float((probs.argmax(axis=1) == y).mean())
    return {"log_loss": round(float(ll), 6), "brier": round(brier, 6),
            "accuracy": round(acc, 6), "n": int(len(y))}


def _bootstrap_metric_ci(
    probs: np.ndarray, y: np.ndarray, n_classes: int, metric: str,
    n_boot: int, seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    onehot = np.zeros_like(probs, dtype=np.float32)
    onehot[np.arange(n), y] = 1.0
    vals = np.empty(n_boot, dtype=np.float64)
    labels_arr = np.arange(n_classes)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        p_b = probs[idx]; y_b = y[idx]
        if metric == "log_loss":
            vals[i] = log_loss(y_b, p_b, labels=labels_arr)
        elif metric == "brier":
            oh = onehot[idx]
            vals[i] = float(np.mean(np.sum((p_b - oh) ** 2, axis=1)))
        elif metric == "accuracy":
            vals[i] = float((p_b.argmax(axis=1) == y_b).mean())
        else:
            raise ValueError(metric)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"ci95_lo": round(float(lo), 6), "ci95_hi": round(float(hi), 6)}


def _bootstrap_delta_ci(
    probs_a: np.ndarray, probs_b: np.ndarray, y: np.ndarray,
    n_classes: int, metric: str, n_boot: int, seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    onehot = np.zeros_like(probs_a, dtype=np.float32)
    onehot[np.arange(n), y] = 1.0
    vals = np.empty(n_boot, dtype=np.float64)
    labels_arr = np.arange(n_classes)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pa = probs_a[idx]; pb = probs_b[idx]; y_b = y[idx]
        if metric == "log_loss":
            ma = log_loss(y_b, pa, labels=labels_arr)
            mb = log_loss(y_b, pb, labels=labels_arr)
        elif metric == "brier":
            oh = onehot[idx]
            ma = float(np.mean(np.sum((pa - oh) ** 2, axis=1)))
            mb = float(np.mean(np.sum((pb - oh) ** 2, axis=1)))
        elif metric == "accuracy":
            ma = float((pa.argmax(axis=1) == y_b).mean())
            mb = float((pb.argmax(axis=1) == y_b).mean())
        else:
            raise ValueError(metric)
        vals[i] = ma - mb
    lo, hi = np.percentile(vals, [2.5, 97.5])
    if metric == "log_loss":
        ma = log_loss(y, probs_a, labels=labels_arr)
        mb = log_loss(y, probs_b, labels=labels_arr)
    elif metric == "brier":
        ma = float(np.mean(np.sum((probs_a - onehot) ** 2, axis=1)))
        mb = float(np.mean(np.sum((probs_b - onehot) ** 2, axis=1)))
    else:
        ma = float((probs_a.argmax(axis=1) == y).mean())
        mb = float((probs_b.argmax(axis=1) == y).mean())
    return {
        "delta_point": round(float(ma - mb), 6),
        "delta_ci95_lo": round(float(lo), 6),
        "delta_ci95_hi": round(float(hi), 6),
        "n_bootstrap": int(n_boot),
    }


def _with_cis(probs, y, n_classes, seed_offset, n_bootstrap, base_seed):
    pack = _metric_pack(probs, y, n_classes)
    return {
        "log_loss": {
            "point": pack["log_loss"],
            **_bootstrap_metric_ci(probs, y, n_classes, "log_loss",
                                   n_bootstrap, base_seed + seed_offset),
        },
        "brier": {
            "point": pack["brier"],
            **_bootstrap_metric_ci(probs, y, n_classes, "brier",
                                   n_bootstrap, base_seed + seed_offset + 1),
        },
        "accuracy": {
            "point": pack["accuracy"],
            **_bootstrap_metric_ci(probs, y, n_classes, "accuracy",
                                   n_bootstrap, base_seed + seed_offset + 2),
        },
        "n": int(len(y)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=_ROOT / "results" / "validate_pitchgpt_20260418T150305Z" / "pitchgpt_full.pt",
        help="PitchGPT checkpoint (pitcher-disjoint, trained on 2015-2022).",
    )
    parser.add_argument("--max-train-games", type=int, default=1000)
    parser.add_argument("--max-val-games", type=int, default=300)
    parser.add_argument("--max-holdout-games", type=int, default=500)
    parser.add_argument("--lstm-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--high-conf-threshold", type=float, default=0.5,
                        help="Absolute top-1 prob threshold (pg top-1 > X).")
    parser.add_argument("--high-conf-percentile", type=float, default=90.0,
                        help="Alternative: use the X percentile of PitchGPT "
                             "top-1 probs as the high-confidence cutoff. "
                             "The more permissive of (absolute threshold, "
                             "percentile) is used.")
    parser.add_argument(
        "--output-dir", type=Path,
        default=_ROOT / "results" / "pitchgpt" / "downstream_utility",
    )
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--max-train-targets", type=int, default=300_000,
                        help="Cap training targets per XGB model (uniform subsample).")
    parser.add_argument("--cache-dir", type=Path,
                        default=_ROOT / "results" / "pitchgpt" / "downstream_utility" / "_cache",
                        help="Cache directory for summary arrays — skip dataset "
                             "build + model scoring on re-runs.")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Ignore cache and rebuild from scratch.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)
    device = _get_device()
    logger.info("device=%s  seed=%d  checkpoint=%s", device, args.seed, args.checkpoint)

    cache_path = args.cache_dir / (
        f"summaries_seed{args.seed}_tgames{args.max_train_games}_"
        f"hgames{args.max_holdout_games}_seq{args.max_seq_len}.npz"
    )
    audit: dict = {}
    lstm_meta: dict = {}

    if cache_path.exists() and not args.force_refresh:
        logger.info("Loading cached summary arrays from %s", cache_path)
        cache = np.load(cache_path, allow_pickle=True)
        s_tr_pg = cache["s_tr_pg"]; s_tr_lstm = cache["s_tr_lstm"]
        ctx_tr = cache["ctx_tr"]; y_tr = cache["y_tr"]; top1_tr_pg = cache["top1_tr_pg"]
        s_ho_pg = cache["s_ho_pg"]; s_ho_lstm = cache["s_ho_lstm"]
        ctx_ho = cache["ctx_ho"]; y_ho = cache["y_ho"]; top1_ho_pg = cache["top1_ho_pg"]
        audit = json.loads(str(cache["audit_json"]))
        lstm_meta = json.loads(str(cache["lstm_meta_json"]))
        logger.info("Cache loaded — train targets=%d  holdout targets=%d",
                    len(y_tr), len(y_ho))
    else:
        s_tr_pg = s_tr_lstm = ctx_tr = y_tr = top1_tr_pg = None
        s_ho_pg = s_ho_lstm = ctx_ho = y_ho = top1_ho_pg = None

    if s_tr_pg is None:
        # ── 1. Build pitcher-disjoint datasets (same as 2025 holdout script).
        conn = get_connection(args.db_path, read_only=True)
        try:
            logger.info("Loading train (2015-2022)  val (2023)  holdout (2025) datasets…")
            train_pitchers_all = PitchSequenceDataset.fetch_pitcher_ids_for_seasons(
                conn, range(TRAIN_RANGE[0], TRAIN_RANGE[1] + 1),
            )
            logger.info("Train cohort size: %d pitchers", len(train_pitchers_all))

            train_ds = PitchSequenceDataset(
                conn, split_mode="train",
                train_range=TRAIN_RANGE, val_range=VAL_RANGE, test_range=HOLDOUT_RANGE,
                max_games_per_split=args.max_train_games,
            )
            val_ds = PitchSequenceDataset(
                conn, split_mode="val",
                train_range=TRAIN_RANGE, val_range=VAL_RANGE, test_range=HOLDOUT_RANGE,
                max_games_per_split=args.max_val_games,
                exclude_pitcher_ids=train_pitchers_all,
            )
            holdout_ds = PitchSequenceDataset(
                conn, split_mode="test",
                train_range=TRAIN_RANGE, val_range=VAL_RANGE, test_range=HOLDOUT_RANGE,
                max_games_per_split=args.max_holdout_games,
                exclude_pitcher_ids=train_pitchers_all,
            )
            audit = audit_no_game_overlap(train_ds, val_ds, holdout_ds)
            logger.info("Leakage audit: %s", json.dumps(audit))
            if audit["shared_game_pks"] != 0 or audit["shared_pitcher_ids_train_test"] != 0:
                logger.error("LEAKAGE — aborting.")
                return 2

            def _build_labeled(ref_ds: PitchSequenceDataset, label: str) -> LabeledPitchSequenceDataset:
                t0 = time.perf_counter()
                gpk_set = set(int(g) for g in ref_ds.game_pks)
                if not gpk_set:
                    return LabeledPitchSequenceDataset(conn, seasons=[], max_seq_len=args.max_seq_len,
                                                        max_games=None, exclude_pitcher_ids=None)
                g_str = ", ".join(str(int(g)) for g in gpk_set)
                pitcher_exclude = ""
                if label != "train" and train_pitchers_all:
                    pids_str = ", ".join(str(int(p)) for p in train_pitchers_all)
                    pitcher_exclude = f"AND pitcher_id NOT IN ({pids_str})"

                query = f"""
                    SELECT game_pk, pitcher_id, pitch_type, plate_x, plate_z,
                           release_speed, balls, strikes, outs_when_up,
                           on_1b, on_2b, on_3b, stand, inning,
                           inning_topbot, events, delta_run_exp,
                           at_bat_number, pitch_number, description
                    FROM pitches
                    WHERE pitch_type IS NOT NULL
                      AND game_pk IN ({g_str})
                      {pitcher_exclude}
                    ORDER BY game_pk, at_bat_number, pitch_number
                """
                df = conn.execute(query).fetchdf()
                if df.empty:
                    return LabeledPitchSequenceDataset(conn, seasons=[], max_seq_len=args.max_seq_len,
                                                        max_games=None, exclude_pitcher_ids=None)
                df = df.assign(_score_diff=_compute_per_pitch_score_diff(df))
                logger.info("[%s] labeled frame: %d rows  %d game_pks", label, len(df), df["game_pk"].nunique())
                ds = _build_labeled_from_frame(df, args.max_seq_len)
                logger.info("[%s] labeled dataset: %d sequences in %.1fs",
                            label, len(ds), time.perf_counter() - t0)
                return ds

            labeled_train = _build_labeled(train_ds, "train")
            labeled_holdout = _build_labeled(holdout_ds, "holdout")
        finally:
            conn.close()

        # ── 2. Load PitchGPT, train LSTM.
        logger.info("Loading PitchGPT checkpoint…")
        pitchgpt = _load_pitchgpt(args.checkpoint, device)
        logger.info("PitchGPT loaded (%d params)", sum(p.numel() for p in pitchgpt.parameters()))

        logger.info("Training LSTM baseline (5 epochs, same hyperparams as 2025 holdout)…")
        lstm, lstm_meta = _train_lstm(
            train_ds, val_ds, device,
            epochs=args.lstm_epochs, lr=args.lr,
            batch_size=args.batch_size, seed=args.seed,
        )

        # ── 3. Batched summary scoring.
        logger.info("Scoring PitchGPT on TRAIN…")
        s_tr_pg, ctx_tr, y_tr, top1_tr_pg = score_summaries(
            pitchgpt, labeled_train, device, VOCAB_SIZE, args.batch_size, "PGpT/train")
        logger.info("Scoring LSTM on TRAIN…")
        s_tr_lstm, ctx_tr_lstm, y_tr_lstm, top1_tr_lstm = score_summaries(
            lstm, labeled_train, device, lstm.output_vocab, args.batch_size, "LSTM/train")

        assert y_tr.shape == y_tr_lstm.shape, f"train label shape mismatch {y_tr.shape} vs {y_tr_lstm.shape}"
        assert np.array_equal(y_tr, y_tr_lstm), "train outcome label mismatch between PGpT and LSTM"
        assert np.allclose(ctx_tr, ctx_tr_lstm, atol=1e-5), "train context mismatch between PGpT and LSTM"

        logger.info("Scoring PitchGPT on HOLDOUT…")
        s_ho_pg, ctx_ho, y_ho, top1_ho_pg = score_summaries(
            pitchgpt, labeled_holdout, device, VOCAB_SIZE, args.batch_size, "PGpT/holdout")
        logger.info("Scoring LSTM on HOLDOUT…")
        s_ho_lstm, ctx_ho_lstm, y_ho_lstm, top1_ho_lstm = score_summaries(
            lstm, labeled_holdout, device, lstm.output_vocab, args.batch_size, "LSTM/holdout")
        assert y_ho.shape == y_ho_lstm.shape
        assert np.array_equal(y_ho, y_ho_lstm)
        assert np.allclose(ctx_ho, ctx_ho_lstm, atol=1e-5)

        # Save cache.
        np.savez_compressed(
            cache_path,
            s_tr_pg=s_tr_pg, s_tr_lstm=s_tr_lstm, ctx_tr=ctx_tr, y_tr=y_tr, top1_tr_pg=top1_tr_pg,
            s_ho_pg=s_ho_pg, s_ho_lstm=s_ho_lstm, ctx_ho=ctx_ho, y_ho=y_ho, top1_ho_pg=top1_ho_pg,
            audit_json=json.dumps(audit),
            lstm_meta_json=json.dumps(lstm_meta),
        )
        logger.info("Cached summary arrays at %s", cache_path)

    # ── 4. Optional training subsample.
    rng = np.random.default_rng(args.seed)
    n_tr = len(y_tr)
    if n_tr > args.max_train_targets > 0:
        idx = rng.choice(n_tr, size=args.max_train_targets, replace=False)
        idx.sort()
        s_tr_pg = s_tr_pg[idx]; s_tr_lstm = s_tr_lstm[idx]
        ctx_tr = ctx_tr[idx]; y_tr = y_tr[idx]; top1_tr_pg = top1_tr_pg[idx]
        logger.info("Subsampled training targets %d → %d", n_tr, len(y_tr))

    # ── 5. Feature matrices.
    X_tr_null = ctx_tr
    X_tr_pg = np.hstack([ctx_tr, s_tr_pg])
    X_tr_lstm = np.hstack([ctx_tr, s_tr_lstm])
    X_ho_null = ctx_ho
    X_ho_pg = np.hstack([ctx_ho, s_ho_pg])
    X_ho_lstm = np.hstack([ctx_ho, s_ho_lstm])
    logger.info("Feature dims  null=%d  pgpt=%d  lstm=%d",
                X_tr_null.shape[1], X_tr_pg.shape[1], X_tr_lstm.shape[1])

    tr_counts = Counter(y_tr.tolist())
    ho_counts = Counter(y_ho.tolist())
    logger.info("Train outcome counts: %s",
                {OUTCOME_CLASSES[k]: v for k, v in tr_counts.items()})
    logger.info("Holdout outcome counts: %s",
                {OUTCOME_CLASSES[k]: v for k, v in ho_counts.items()})

    # 10% slice for XGB early stopping.
    rng2 = np.random.default_rng(args.seed + 7)
    perm = rng2.permutation(len(y_tr))
    val_cut = int(0.1 * len(perm))
    val_pos = perm[:val_cut]; tr_pos = perm[val_cut:]

    def _split(X): return X[tr_pos], X[val_pos]
    X_tr_null_t, X_tr_null_v = _split(X_tr_null)
    X_tr_pg_t, X_tr_pg_v = _split(X_tr_pg)
    X_tr_lstm_t, X_tr_lstm_v = _split(X_tr_lstm)
    y_tr_t, y_tr_v = y_tr[tr_pos], y_tr[val_pos]

    # ── 6. Train XGBs.
    logger.info("Training XGB (null)…")
    clf_null = _fit_xgb(X_tr_null_t, y_tr_t, X_tr_null_v, y_tr_v,
                       n_classes=len(OUTCOME_CLASSES), seed=args.seed, variant_name="null")
    logger.info("Training XGB (PitchGPT)…")
    clf_pg = _fit_xgb(X_tr_pg_t, y_tr_t, X_tr_pg_v, y_tr_v,
                     n_classes=len(OUTCOME_CLASSES), seed=args.seed + 1, variant_name="pgpt")
    logger.info("Training XGB (LSTM)…")
    clf_lstm = _fit_xgb(X_tr_lstm_t, y_tr_t, X_tr_lstm_v, y_tr_v,
                       n_classes=len(OUTCOME_CLASSES), seed=args.seed + 2, variant_name="lstm")

    # ── 7. Score on holdout.
    probs_null = clf_null.predict_proba(X_ho_null)
    probs_pg = clf_pg.predict_proba(X_ho_pg)
    probs_lstm = clf_lstm.predict_proba(X_ho_lstm)

    variants = {
        "null_situational_only": _with_cis(probs_null, y_ho, len(OUTCOME_CLASSES), 100, args.n_bootstrap, args.seed),
        "pitchgpt_plus_situational": _with_cis(probs_pg, y_ho, len(OUTCOME_CLASSES), 200, args.n_bootstrap, args.seed),
        "lstm_plus_situational": _with_cis(probs_lstm, y_ho, len(OUTCOME_CLASSES), 300, args.n_bootstrap, args.seed),
    }
    logger.info("Headline variants: %s", {k: v["log_loss"] for k, v in variants.items()})

    def _deltas(probs_a, probs_b, y, offset):
        return {
            "log_loss": _bootstrap_delta_ci(probs_a, probs_b, y, len(OUTCOME_CLASSES),
                                            "log_loss", args.n_bootstrap, args.seed + offset),
            "brier": _bootstrap_delta_ci(probs_a, probs_b, y, len(OUTCOME_CLASSES),
                                         "brier", args.n_bootstrap, args.seed + offset + 1),
            "accuracy": _bootstrap_delta_ci(probs_a, probs_b, y, len(OUTCOME_CLASSES),
                                            "accuracy", args.n_bootstrap, args.seed + offset + 2),
        }

    pair_deltas = {
        "pitchgpt_vs_lstm": _deltas(probs_pg, probs_lstm, y_ho, 500),
        "pitchgpt_vs_null": _deltas(probs_pg, probs_null, y_ho, 510),
        "lstm_vs_null": _deltas(probs_lstm, probs_null, y_ho, 520),
    }

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "downstream_target": "pitch_outcome_bucket_5class (option b)",
        "outcome_classes": OUTCOME_CLASSES,
        "class_counts_train": {OUTCOME_CLASSES[k]: int(v) for k, v in tr_counts.items()},
        "class_counts_holdout": {OUTCOME_CLASSES[k]: int(v) for k, v in ho_counts.items()},
        "feature_dims": {
            "null_situational_only": X_tr_null.shape[1],
            "pitchgpt_plus_situational": X_tr_pg.shape[1],
            "lstm_plus_situational": X_tr_lstm.shape[1],
        },
        "n_train_targets": int(len(y_tr)),
        "n_holdout_targets": int(len(y_ho)),
        "variants": variants,
        "pair_deltas": pair_deltas,
        "leakage_audit": audit,
        "lstm_meta": lstm_meta,
        "n_bootstrap": args.n_bootstrap,
        "max_train_games": args.max_train_games,
        "max_val_games": args.max_val_games,
        "max_holdout_games": args.max_holdout_games,
        "max_train_targets": args.max_train_targets,
    }
    (args.output_dir / "downstream_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %s/downstream_comparison.json", args.output_dir)

    # ── 8. High-confidence subsets (multiple thresholds + percentile).
    #   Given PitchGPT's ~36% top-1 accuracy over a 2,210-vocab, an
    #   absolute top-1 > 0.5 cutoff would isolate near-zero pitches.
    #   We report: (a) the requested absolute threshold (usually 0.5),
    #   (b) a percentile-based cutoff, and (c) absolute thresholds of
    #   0.10, 0.20, 0.30 for coverage-vs-signal sweeps.
    pct_cutoff = float(np.percentile(top1_ho_pg, args.high_conf_percentile)) if len(top1_ho_pg) else 0.0
    logger.info("PitchGPT top-1 prob on holdout — min=%.4f  mean=%.4f  max=%.4f  "
                "p50=%.4f  p90=%.4f  p95=%.4f  p99=%.4f",
                float(top1_ho_pg.min()) if len(top1_ho_pg) else 0.0,
                float(top1_ho_pg.mean()) if len(top1_ho_pg) else 0.0,
                float(top1_ho_pg.max()) if len(top1_ho_pg) else 0.0,
                float(np.percentile(top1_ho_pg, 50)) if len(top1_ho_pg) else 0.0,
                float(np.percentile(top1_ho_pg, 90)) if len(top1_ho_pg) else 0.0,
                float(np.percentile(top1_ho_pg, 95)) if len(top1_ho_pg) else 0.0,
                float(np.percentile(top1_ho_pg, 99)) if len(top1_ho_pg) else 0.0)

    # Build a list of subsets.  The `label` is saved in the output JSON.
    subsets: list[tuple[str, float]] = [
        (f"abs_threshold_{args.high_conf_threshold}", args.high_conf_threshold),
        (f"abs_threshold_0.10", 0.10),
        (f"abs_threshold_0.20", 0.20),
        (f"abs_threshold_0.30", 0.30),
        (f"top_{100 - args.high_conf_percentile:.0f}pct_by_top1", pct_cutoff),
    ]

    hc_payload = {
        "primary_threshold": args.high_conf_threshold,
        "percentile_cutoff_value": pct_cutoff,
        "percentile": args.high_conf_percentile,
        "top1_prob_holdout_stats": {
            "min": float(top1_ho_pg.min()) if len(top1_ho_pg) else 0.0,
            "mean": float(top1_ho_pg.mean()) if len(top1_ho_pg) else 0.0,
            "max": float(top1_ho_pg.max()) if len(top1_ho_pg) else 0.0,
            "p50": float(np.percentile(top1_ho_pg, 50)) if len(top1_ho_pg) else 0.0,
            "p75": float(np.percentile(top1_ho_pg, 75)) if len(top1_ho_pg) else 0.0,
            "p90": float(np.percentile(top1_ho_pg, 90)) if len(top1_ho_pg) else 0.0,
            "p95": float(np.percentile(top1_ho_pg, 95)) if len(top1_ho_pg) else 0.0,
            "p99": float(np.percentile(top1_ho_pg, 99)) if len(top1_ho_pg) else 0.0,
        },
        "n_total_holdout": int(len(y_ho)),
        "subsets": {},
    }

    for label, thr in subsets:
        hc_mask = top1_ho_pg > thr
        n_hc = int(hc_mask.sum())
        hc_frac = float(hc_mask.mean()) if len(hc_mask) else 0.0
        logger.info("Subset %s  thr=%.4f  n=%d (%.1f%%)",
                    label, thr, n_hc, 100 * hc_frac)
        entry: dict = {
            "threshold": float(thr),
            "n": n_hc,
            "pct_of_total": round(100 * hc_frac, 2),
        }
        if n_hc > 100:
            y_hc = y_ho[hc_mask]
            p_null_hc = probs_null[hc_mask]
            p_pg_hc = probs_pg[hc_mask]
            p_lstm_hc = probs_lstm[hc_mask]
            entry["variants"] = {
                "null_situational_only": _with_cis(p_null_hc, y_hc, len(OUTCOME_CLASSES), 1100,
                                                   args.n_bootstrap, args.seed),
                "pitchgpt_plus_situational": _with_cis(p_pg_hc, y_hc, len(OUTCOME_CLASSES), 1200,
                                                       args.n_bootstrap, args.seed),
                "lstm_plus_situational": _with_cis(p_lstm_hc, y_hc, len(OUTCOME_CLASSES), 1300,
                                                   args.n_bootstrap, args.seed),
            }
            entry["pair_deltas"] = {
                "pitchgpt_vs_lstm": _deltas(p_pg_hc, p_lstm_hc, y_hc, 1500),
                "pitchgpt_vs_null": _deltas(p_pg_hc, p_null_hc, y_hc, 1510),
                "lstm_vs_null": _deltas(p_lstm_hc, p_null_hc, y_hc, 1520),
            }
            entry["class_counts"] = {
                OUTCOME_CLASSES[k]: int(v) for k, v in Counter(y_hc.tolist()).items()
            }
        hc_payload["subsets"][label] = entry
    (args.output_dir / "calibration_utility_subset.json").write_text(
        json.dumps(hc_payload, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %s/calibration_utility_subset.json", args.output_dir)

    # ── 9. Markdown report.
    _write_report(args.output_dir / "report.md", payload=payload, hc_payload=hc_payload)

    # ── 10. Console summary.
    print("\n" + "=" * 72)
    print("PitchGPT Downstream Utility -- Summary")
    print("=" * 72)
    print(f"Holdout targets: {payload['n_holdout_targets']}  Train targets: {payload['n_train_targets']}")
    for name, v in variants.items():
        print(f"  {name:32s}  ll={v['log_loss']['point']:.4f} ({v['log_loss']['ci95_lo']:.4f},"
              f"{v['log_loss']['ci95_hi']:.4f})  brier={v['brier']['point']:.4f}  acc={v['accuracy']['point']:.4f}")
    print("-" * 72)
    print("Pair deltas (A-B; negative log-loss/brier = A better; positive accuracy = A better):")
    for pair, d in pair_deltas.items():
        ll = d["log_loss"]; br = d["brier"]; ac = d["accuracy"]
        print(f"  {pair:22s}  ll={ll['delta_point']:+.4f} [{ll['delta_ci95_lo']:+.4f}, {ll['delta_ci95_hi']:+.4f}]  "
              f"brier={br['delta_point']:+.4f} [{br['delta_ci95_lo']:+.4f}, {br['delta_ci95_hi']:+.4f}]  "
              f"acc={ac['delta_point']:+.4f} [{ac['delta_ci95_lo']:+.4f}, {ac['delta_ci95_hi']:+.4f}]")
    print("-" * 72)
    print(f"High-confidence subsets (total holdout {hc_payload['n_total_holdout']}):")
    for label, entry in hc_payload["subsets"].items():
        n = entry["n"]
        if "pair_deltas" in entry:
            ll = entry["pair_deltas"]["pitchgpt_vs_lstm"]["log_loss"]
            llp = entry["pair_deltas"]["pitchgpt_vs_null"]["log_loss"]
            print(f"  {label:32s}  thr={entry['threshold']:.3f}  n={n:>6d} "
                  f"({entry['pct_of_total']:.1f}%)  "
                  f"pg-lstm ll={ll['delta_point']:+.4f} [{ll['delta_ci95_lo']:+.4f},{ll['delta_ci95_hi']:+.4f}]  "
                  f"pg-null ll={llp['delta_point']:+.4f} [{llp['delta_ci95_lo']:+.4f},{llp['delta_ci95_hi']:+.4f}]")
        else:
            print(f"  {label:32s}  thr={entry['threshold']:.3f}  n={n} (too few for CI)")
    print("=" * 72)
    return 0


def _build_labeled_from_frame(df: pd.DataFrame, max_seq_len: int) -> LabeledPitchSequenceDataset:
    """Construct a LabeledPitchSequenceDataset directly from a pre-loaded frame
    (skips the SQL query — used when we already have the exact game_pk set).
    """
    out = LabeledPitchSequenceDataset.__new__(LabeledPitchSequenceDataset)
    out.max_seq_len = max_seq_len
    out.sequences = []
    out.game_pks = set()
    out.pitcher_ids = set()
    if df.empty:
        return out
    grouped = df.groupby(["game_pk", "pitcher_id"], sort=False)
    for _key, game_df in grouped:
        if len(game_df) < 2:
            continue
        try:
            _pid = int(_key[1])
        except (TypeError, ValueError):
            _pid = -1
        try:
            out.game_pks.add(int(_key[0]))
            out.pitcher_ids.add(_pid)
        except (TypeError, ValueError):
            pass

        tokens: list[int] = []
        contexts: list[torch.Tensor] = []
        outcomes: list[int] = []
        for _, row in game_df.iterrows():
            tok = PitchTokenizer.encode(
                row.get("pitch_type"), row.get("plate_x"), row.get("plate_z"),
                row.get("release_speed"),
            )
            tokens.append(tok)
            ctx_list = PitchTokenizer.encode_context(
                balls=_safe_int(row.get("balls"), 0),
                strikes=_safe_int(row.get("strikes"), 0),
                outs=_safe_int(row.get("outs_when_up"), 0),
                on_1b=_safe_bool(row.get("on_1b")),
                on_2b=_safe_bool(row.get("on_2b")),
                on_3b=_safe_bool(row.get("on_3b")),
                stand=_safe_str(row.get("stand"), "R"),
                inning=_safe_int(row.get("inning"), 1),
                score_diff=_safe_int(row.get("_score_diff"), 0),
            )
            contexts.append(PitchTokenizer.context_to_tensor(ctx_list))
            desc = row.get("description")
            bucket = _DESCRIPTION_TO_BUCKET.get(str(desc) if desc is not None else "", None)
            outcomes.append(_OUTCOME_TO_IDX[bucket] if bucket is not None else _OUTCOME_SKIP)
        tokens = tokens[:max_seq_len]; contexts = contexts[:max_seq_len]; outcomes = outcomes[:max_seq_len]
        input_tokens = torch.tensor(tokens[:-1], dtype=torch.long)
        target_tokens = torch.tensor(tokens[1:], dtype=torch.long)
        target_outcomes = torch.tensor(outcomes[1:], dtype=torch.long)
        context_tensor = torch.stack(contexts[:-1])
        out.sequences.append((input_tokens, context_tensor, target_tokens, target_outcomes))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────


def _fmt(val, k=4):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    return f"{val:.{k}f}"


def _write_report(path: Path, payload: dict, hc_payload: dict) -> None:
    lines: list[str] = []
    lines.append("# PitchGPT Downstream Utility Report\n")
    lines.append(f"Generated: {payload['timestamp_utc']}\n")
    lines.append(f"Checkpoint: `{payload['checkpoint']}`\n")
    lines.append(f"Device: {payload['device']}  Seed: {payload['seed']}\n")
    lines.append("")
    lines.append("## Question")
    lines.append(
        "Does PitchGPT's calibrated per-pitch next-token distribution drive better "
        "downstream *pitch-outcome* prediction than the LSTM baseline's distribution "
        "— even though PitchGPT's raw perplexity edge is only 13.80% "
        "(below the 15% spec gate)?\n"
    )
    lines.append("## Methodology")
    lines.append(
        "- Same pitcher-disjoint splits as `scripts/pitchgpt_2025_holdout.py`: "
        f"train {payload['max_train_games']} games (2015-2022), "
        f"val {payload['max_val_games']} games (2023, for LSTM early-stop), "
        f"holdout {payload['max_holdout_games']} games (2025, pitchers never seen in train)."
    )
    lines.append("- Both models score the same sequences. For every non-PAD "
                 f"target pitch we softmax over the {VOCAB_SIZE:,}-token vocab "
                 "and summarise on-device into a 51-dim vector: 17 pitch-type "
                 "marginals + 26 zone marginals + 5 velocity-bucket marginals "
                 "+ top-1 prob + top-1 log-prob + entropy.")
    lines.append("- Situational context (34-d one-hot: count, outs, runners, "
                 "batter-hand, inning, score-diff) is identical to the model's "
                 "input context.")
    lines.append("- Downstream model: XGBoost `multi:softprob`, 400 trees, "
                 "depth 6, lr 0.08, early-stopping on a 10% slice of the train-"
                 "year targets (the 2025 holdout is never seen during training).")
    lines.append(
        "- Outcome target (5 classes): " + ", ".join(OUTCOME_CLASSES) +
        ".  Low-count descriptions are folded in (`blocked_ball`→`ball`, "
        "`foul_tip`/`foul_bunt`/`bunt_foul_tip`→`foul`, "
        "`swinging_strike_blocked`/`missed_bunt`→`swinging_strike`, "
        "`hit_by_pitch`/`pitchout`→`ball`).")
    lines.append(f"- Bootstrap CIs: {payload['n_bootstrap']} resamples with "
                 "replacement; paired resamples for pairwise deltas.")
    lines.append("")
    lines.append("## Leakage audit")
    audit = payload["leakage_audit"]
    lines.append(f"- shared `game_pk` across splits: **{audit['shared_game_pks']}**")
    lines.append(f"- shared pitchers train/holdout: **{audit['shared_pitcher_ids_train_test']}**")
    lines.append(f"- train pitchers={audit['n_train_pitchers']}  "
                 f"val pitchers={audit['n_val_pitchers']}  "
                 f"holdout pitchers={audit['n_test_pitchers']}")
    lines.append("")
    lines.append("## Dataset sizes")
    lines.append(f"- Train targets (post-subsample): {payload['n_train_targets']:,}")
    lines.append(f"- Holdout targets: {payload['n_holdout_targets']:,}")
    lines.append("- Train outcome counts: " +
                 ", ".join(f"`{k}`={v:,}" for k, v in payload["class_counts_train"].items()))
    lines.append("- Holdout outcome counts: " +
                 ", ".join(f"`{k}`={v:,}" for k, v in payload["class_counts_holdout"].items()))
    lines.append("")
    lines.append("## Headline results (2025 holdout)\n")
    lines.append("| Variant | Feature dim | Log-loss (95% CI) | Brier (95% CI) | Accuracy (95% CI) |")
    lines.append("|---|---|---|---|---|")
    for key in ["null_situational_only", "pitchgpt_plus_situational", "lstm_plus_situational"]:
        v = payload["variants"][key]; dim = payload["feature_dims"][key]
        ll = v["log_loss"]; br = v["brier"]; ac = v["accuracy"]
        lines.append(
            f"| {key} | {dim} | {_fmt(ll['point'])} ({_fmt(ll['ci95_lo'])}, {_fmt(ll['ci95_hi'])}) | "
            f"{_fmt(br['point'])} ({_fmt(br['ci95_lo'])}, {_fmt(br['ci95_hi'])}) | "
            f"{_fmt(ac['point'])} ({_fmt(ac['ci95_lo'])}, {_fmt(ac['ci95_hi'])}) |"
        )
    lines.append("")
    lines.append("## Pairwise deltas (A − B)\n")
    lines.append("Negative log-loss/Brier delta ⇒ A better. Positive accuracy delta ⇒ A better.\n")
    lines.append("| Comparison (A vs B) | Δ Log-loss (95% CI) | Δ Brier (95% CI) | Δ Accuracy (95% CI) |")
    lines.append("|---|---|---|---|")
    for pair, d in payload["pair_deltas"].items():
        ll = d["log_loss"]; br = d["brier"]; ac = d["accuracy"]
        lines.append(
            f"| {pair} | {_fmt(ll['delta_point'])} ({_fmt(ll['delta_ci95_lo'])}, {_fmt(ll['delta_ci95_hi'])}) | "
            f"{_fmt(br['delta_point'])} ({_fmt(br['delta_ci95_lo'])}, {_fmt(br['delta_ci95_hi'])}) | "
            f"{_fmt(ac['delta_point'])} ({_fmt(ac['delta_ci95_lo'])}, {_fmt(ac['delta_ci95_hi'])}) |"
        )
    lines.append("")
    lines.append("## High-confidence subsets\n")
    stats = hc_payload["top1_prob_holdout_stats"]
    lines.append(
        f"PitchGPT top-1 probability on the 2025 holdout ranges from "
        f"{stats['min']:.4f} to {stats['max']:.4f} (mean {stats['mean']:.4f}, "
        f"p50 {stats['p50']:.4f}, p90 {stats['p90']:.4f}, p99 {stats['p99']:.4f}). "
        f"The wide 2,210-token vocabulary keeps absolute confidences low; we "
        f"therefore report several cutoffs: the requested absolute 0.5 "
        f"threshold, a sweep at 0.10/0.20/0.30, and a top-"
        f"{100 - hc_payload['percentile']:.0f}% percentile cutoff "
        f"({hc_payload['percentile_cutoff_value']:.4f})."
    )
    lines.append("")
    lines.append("| Subset | Threshold | n (coverage) | Δ log-loss PGpT-LSTM (95% CI) | Δ log-loss PGpT-Null (95% CI) |")
    lines.append("|---|---|---|---|---|")
    for label, entry in hc_payload["subsets"].items():
        n = entry["n"]
        pct = entry.get("pct_of_total", 0.0)
        if "pair_deltas" in entry:
            llpl = entry["pair_deltas"]["pitchgpt_vs_lstm"]["log_loss"]
            llpn = entry["pair_deltas"]["pitchgpt_vs_null"]["log_loss"]
            lines.append(
                f"| {label} | {entry['threshold']:.4f} | {n:,} ({pct:.1f}%) | "
                f"{_fmt(llpl['delta_point'])} ({_fmt(llpl['delta_ci95_lo'])}, {_fmt(llpl['delta_ci95_hi'])}) | "
                f"{_fmt(llpn['delta_point'])} ({_fmt(llpn['delta_ci95_lo'])}, {_fmt(llpn['delta_ci95_hi'])}) |"
            )
        else:
            lines.append(f"| {label} | {entry['threshold']:.4f} | {n:,} ({pct:.1f}%) | — (n ≤ 100) | — (n ≤ 100) |")
    lines.append("")
    lines.append("## Interpretation\n")
    pd_pg_lstm = payload["pair_deltas"]["pitchgpt_vs_lstm"]["log_loss"]
    pd_pg_null = payload["pair_deltas"]["pitchgpt_vs_null"]["log_loss"]
    pd_lstm_null = payload["pair_deltas"]["lstm_vs_null"]["log_loss"]

    def _ci_excludes_zero(d):
        return (d["delta_ci95_lo"] > 0) or (d["delta_ci95_hi"] < 0)

    if pd_pg_lstm["delta_point"] < 0 and _ci_excludes_zero(pd_pg_lstm):
        lines.append(
            f"- PitchGPT beats LSTM downstream in log-loss by "
            f"{abs(pd_pg_lstm['delta_point']):.4f} with CI excluding zero — "
            "signal survives the null-delta test."
        )
    elif pd_pg_lstm["delta_point"] > 0 and _ci_excludes_zero(pd_pg_lstm):
        lines.append(
            f"- LSTM beats PitchGPT downstream by {pd_pg_lstm['delta_point']:.4f} "
            "log-loss — PitchGPT's calibration does NOT help here."
        )
    else:
        lines.append(
            "- PitchGPT vs LSTM is a statistical tie downstream (CI spans zero). "
            "PitchGPT's perplexity/calibration advantage does not translate to "
            "better outcome prediction at this sample size."
        )
    if pd_pg_null["delta_point"] < 0 and _ci_excludes_zero(pd_pg_null):
        lines.append(
            "- Adding PitchGPT features over situational-only context reduces "
            "log-loss with CI excluding zero — the distribution adds real "
            "downstream information beyond count/outs/inning."
        )
    else:
        lines.append(
            "- Adding PitchGPT features over the situational-only null does NOT "
            "significantly reduce log-loss (CI spans zero)."
        )
    if pd_lstm_null["delta_point"] < 0 and _ci_excludes_zero(pd_lstm_null):
        lines.append("- The LSTM distribution also beats the null.")
    else:
        lines.append("- The LSTM distribution does not significantly beat the null.")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
