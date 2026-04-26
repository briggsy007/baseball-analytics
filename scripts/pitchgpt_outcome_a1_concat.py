"""
A1 — Frozen PitchGPT v2 backbone + concat MLP head (Plan B Step 2).

Plan B §3 hypothesis A1.  Tests whether the PitchGPT v2 backbone hidden
state carries any marginal information beyond engineered features for
the 7-class PA-outcome target.

Architecture (Plan B §3 hypothesis A1, locked):
    backbone(v2, FROZEN) -> hidden_state[B,S,128]
    concat with raw_context[B,S,35] + pitch_type_oh[B,S,17]
            + zone_oh[B,S,26] + velo_oh[B,S,5] = 211-dim
    MLP: 211 -> 128 -> 64 -> 7 (ReLU + dropout 0.1)

Loss:
    Weighted CE with inverse-frequency class weights cap 10 (matches
    Phase 0.3 + Plan B Step 1 protocol).

Training:
    AdamW lr=1e-3, batch=32, 5 epochs, early-stop on val log-loss
    (patience=2).  Identical to Phase 0.3 protocol.

Cohort (locked, identical to Phase 0.3 + Plan B Step 1):
    Train: 2015-2022 pitcher-disjoint, 10K-game subset.
    Val:   2023 pitcher-disjoint.
    Test:  2025 pitcher-disjoint.

    Game-pks sampled via the same deterministic numpy-seeded routine as
    `pitchgpt_outcome_baselines_common.sample_game_pks` (seeds 42 / 43 /
    44 for train / val / test) — guarantees the SAME games appear in
    both A1 and A3 (paired bootstrap on the row-aligned delta is valid).

Outputs (artifacts):
    - models/pitchgpt_v2_outcomehead_a1.pt           (NEW path)
    - results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/
        metrics.json
        report.md
        train.log

Guardrails (non-negotiable):
    - models/pitchgpt_v2.pt: byte-identity verified pre/post via SHA256.
    - models/pitchgpt_v2_outcomehead.pt: NEVER overwritten (Phase 0.3
      FAIL artifact, kept for diff).
    - DuckDB read-only.

Usage:
    python scripts/pitchgpt_outcome_a1_concat.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt import (  # noqa: E402
    NUM_PITCH_TYPES,
    NUM_VELO_BUCKETS,
    NUM_ZONES,
    PAD_TOKEN,
    VOCAB_SIZE,
    PitchGPTModel,
    PitchSequenceDataset,
    PitchTokenizer,
    _get_device,
    _load_model,
)
from src.analytics.pitchgpt_outcome_head import (  # noqa: E402
    NUM_OUTCOME_CLASSES,
    OUTCOME_CLASSES,
    OUTCOME_UNK,
    FrozenOutcomeHeadConcat,
    classify_pitch_outcome,
    extract_backbone_hidden_states,
    freeze_backbone,
    prior_log_loss,
    seven_class_ece,
    seven_class_log_loss,
)
from scripts.pitchgpt_outcome_baselines_common import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_TEST_GAMES,
    DEFAULT_TRAIN_GAMES,
    DEFAULT_VAL_GAMES,
    TEST_SEASONS,
    TRAIN_SEASONS,
    VAL_SEASONS,
    bootstrap_log_loss,
    class_frequency_prior,
    ece_10bin,
    fetch_pitcher_ids,
    fit_temperature,
    freq_prior_log_loss_on,
    log_loss_from_probs,
    per_class_log_loss,
    per_pitcher_log_loss,
    sample_game_pks,
    softmax,
    top1_accuracy,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("a1_concat")


# ═════════════════════════════════════════════════════════════════════════════
# Config
# ═════════════════════════════════════════════════════════════════════════════

OUT_DIR = (
    _ROOT / "results" / "pitchgpt_sim" / "outcome_baselines_2026_04_25"
    / "a1_concat"
)
MODELS_DIR = _ROOT / "models"
BACKBONE_PATH = MODELS_DIR / "pitchgpt_v2.pt"
PHASE03_CKPT = MODELS_DIR / "pitchgpt_v2_outcomehead.pt"  # do NOT overwrite
A1_CKPT = MODELS_DIR / "pitchgpt_v2_outcomehead_a1.pt"

A3_METRICS_PATH = (
    _ROOT / "results" / "pitchgpt_sim" / "outcome_baselines_2026_04_25"
    / "a3_xgboost" / "metrics.json"
)
A3_CACHE_DIR = _ROOT / "data" / "staging" / "outcome_baselines_cache"


# A1 protocol numbers (locked per RESEARCH_PLAN §3 A1 + research-plan §6).
DEFAULT_BATCH = 32
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 5
DEFAULT_PATIENCE = 2
DEFAULT_CLASS_WEIGHT_CAP = 10.0


# ═════════════════════════════════════════════════════════════════════════════
# Utilities
# ═════════════════════════════════════════════════════════════════════════════


def _sha256_file(path: Path, chunk: int = 1 << 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _backbone_param_checksum(backbone: PitchGPTModel) -> str:
    h = hashlib.sha256()
    for name, p in backbone.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    for name, b in backbone.named_buffers():
        h.update(name.encode())
        h.update(b.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _decode_token_components(tokens: torch.Tensor) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Decode a (B, S) tensor of composite tokens into (pitch_type_idx,
    zone_idx, velo_bucket_idx) tensors of the same shape.

    Mirror of PitchTokenizer.decode but vectorised on torch tensors.

    For PAD_TOKEN (== VOCAB_SIZE) and BOS_TOKEN (== VOCAB_SIZE+1) and
    any out-of-range value, we return (NUM_PITCH_TYPES-1, NUM_ZONES-1,
    middle_velo=2) — the "unknown" buckets.  The downstream consumer
    masks those positions out via the outcome target == OUTCOME_UNK
    check, so the choice of unknown values is irrelevant.

    Token formula: t = pt * (NUM_ZONES * NUM_VELO_BUCKETS) + zone *
    NUM_VELO_BUCKETS + velo.
    """
    # Replace any out-of-range token (pad / bos) with 0 to avoid
    # negative indices, then we don't care about those positions.
    safe = torch.where(
        (tokens >= 0) & (tokens < VOCAB_SIZE),
        tokens,
        torch.zeros_like(tokens),
    )
    velo = safe % NUM_VELO_BUCKETS
    rem = safe // NUM_VELO_BUCKETS
    zone = rem % NUM_ZONES
    pt = rem // NUM_ZONES
    return pt, zone, velo


# ═════════════════════════════════════════════════════════════════════════════
# Sequence dataset filtered to a fixed set of game_pks
# ═════════════════════════════════════════════════════════════════════════════


class FixedGamesSequenceDataset(Dataset):
    """PitchSequenceDataset variant restricted to a fixed list of game_pks
    and filtered to exclude a fixed pitcher cohort.

    Mirrors the Plan B `sample_game_pks` cohort selection so that A1's
    sequences cover the SAME games as A3's per-row cache — paired
    bootstrap on the A1 / A3 delta is then well-defined on the
    intersection of (game_pk, at_bat, pitch_number) row keys.

    Yields: (input_tokens [T-1], context [T-1, 35], target_tokens [T-1],
             outcome_labels [T-1], target_pitch_type_idx [T-1],
             target_zone_idx [T-1], target_velo_bucket [T-1],
             target_row_keys: list of tuples (game_pk, at_bat, pitch_number)
             of length T-1)
    """

    def __init__(
        self,
        conn: Any,
        game_pks: list[int],
        exclude_pitcher_ids: set[int] | None = None,
        max_seq_len: int = 256,
        context_dim: int = 35,
    ):
        self.max_seq_len = max_seq_len
        self.context_dim = context_dim
        self.exclude_pitcher_ids = (
            {int(p) for p in exclude_pitcher_ids}
            if exclude_pitcher_ids else set()
        )

        # 1) Use PitchSequenceDataset's own loader for per-pitch tokenisation
        #    and context, but constrain to the requested game_pks.  We can't
        #    use PitchSequenceDataset directly because its load API is
        #    keyed on (seasons, max_games) USING SAMPLE — not deterministic
        #    against our pre-sampled game_pks.  Reuse its private loader by
        #    passing a one-game-at-a-time WHERE clause.
        if not game_pks:
            self.sequences: list[tuple] = []
            self.row_key_arrays: list[np.ndarray] = []
            self.target_pt_idx: list[np.ndarray] = []
            self.target_zone_idx: list[np.ndarray] = []
            self.target_velo_idx: list[np.ndarray] = []
            self.outcome_labels: list[np.ndarray] = []
            self.game_pks: set[int] = set()
            self.pitcher_ids: set[int] = set()
            return

        # Use PitchSequenceDataset by building a lightweight in-memory
        # filter — easier to import its parsing logic by querying with
        # the full WHERE filter ourselves.
        from src.analytics.pitchgpt import (
            _compute_per_pitch_score_diff,
            _fetch_hp_umpire_by_game,
            _fetch_prior_season_ump_tendency,
            _fetch_season_league_median,
            _safe_bool,
            _safe_int,
            _safe_str,
        )

        gpks_str = ", ".join(str(int(g)) for g in game_pks)
        exclude_filter = ""
        if self.exclude_pitcher_ids:
            pids_str = ", ".join(str(int(p)) for p in self.exclude_pitcher_ids)
            exclude_filter = f"AND pitcher_id NOT IN ({pids_str})"

        # Chunked SQL fetch for very large IN lists.
        chunks: list[pd.DataFrame] = []
        chunk_size = 5_000
        for start in range(0, len(game_pks), chunk_size):
            end = min(start + chunk_size, len(game_pks))
            gpk_chunk = ", ".join(str(int(g)) for g in game_pks[start:end])
            q = f"""
                SELECT
                    game_pk,
                    pitcher_id,
                    pitch_type,
                    plate_x,
                    plate_z,
                    release_speed,
                    balls,
                    strikes,
                    outs_when_up,
                    on_1b,
                    on_2b,
                    on_3b,
                    stand,
                    inning,
                    inning_topbot,
                    events,
                    description,
                    delta_run_exp,
                    at_bat_number,
                    pitch_number,
                    game_date
                FROM pitches
                WHERE pitch_type IS NOT NULL
                  AND game_pk IN ({gpk_chunk})
                  {exclude_filter}
                ORDER BY game_pk, at_bat_number, pitch_number
            """
            cdf = conn.execute(q).fetchdf()
            if not cdf.empty:
                chunks.append(cdf)
        if not chunks:
            self.sequences = []
            self.row_key_arrays = []
            self.target_pt_idx = []
            self.target_zone_idx = []
            self.target_velo_idx = []
            self.outcome_labels = []
            self.game_pks = set()
            self.pitcher_ids = set()
            return
        df = pd.concat(chunks, ignore_index=True)

        # Reconstruct score_diff (pitcher POV).
        df = df.assign(_score_diff=_compute_per_pitch_score_diff(df))

        # Umpire scalar (mirrors PitchSequenceDataset).
        df_seasons = (
            sorted({int(s) for s in df["game_date"].dt.year.unique()})
            if "game_date" in df.columns else []
        )
        ump_by_game = _fetch_hp_umpire_by_game(conn, df_seasons)
        tendency_map = _fetch_prior_season_ump_tendency(conn, df_seasons)
        median_by_season = _fetch_season_league_median(conn, df_seasons)

        def _ump_scalar(row) -> float:
            try:
                gdate = row.get("game_date") if "game_date" in df.columns else None
                if gdate is not None and not (isinstance(gdate, float) and np.isnan(gdate)):
                    try:
                        game_season = int(pd.Timestamp(gdate).year)
                    except (TypeError, ValueError):
                        game_season = -1
                else:
                    game_season = -1
            except Exception:
                game_season = -1
            median = median_by_season.get(game_season, 0.0)
            try:
                gpk = int(row.get("game_pk")) if row.get("game_pk") is not None else None
            except (TypeError, ValueError):
                gpk = None
            if gpk is None:
                return median
            ump_name = ump_by_game.get(gpk)
            if ump_name is None:
                return median
            val = tendency_map.get((ump_name, game_season))
            if val is None:
                return median
            return float(val)

        df = df.assign(_ump_scalar=df.apply(_ump_scalar, axis=1))

        # Group by (game, pitcher) — same as PitchSequenceDataset.
        sequences: list[tuple] = []
        row_keys: list[np.ndarray] = []
        tgt_pt: list[np.ndarray] = []
        tgt_zn: list[np.ndarray] = []
        tgt_vl: list[np.ndarray] = []
        outc_lbl: list[np.ndarray] = []

        grouped = df.groupby(["game_pk", "pitcher_id"], sort=False)
        for _key, gdf in grouped:
            if len(gdf) < 2:
                continue
            try:
                _pid = int(_key[1])
            except (TypeError, ValueError):
                _pid = -1
            if self.exclude_pitcher_ids and _pid in self.exclude_pitcher_ids:
                continue

            tokens: list[int] = []
            contexts: list[torch.Tensor] = []
            row_keys_g: list[tuple] = []
            outcomes_g: list[int] = []

            for _, row in gdf.iterrows():
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
                ump_raw = row.get("_ump_scalar")
                try:
                    ump_val = 0.0 if ump_raw is None else float(ump_raw)
                except (TypeError, ValueError):
                    ump_val = 0.0
                contexts.append(
                    PitchTokenizer.context_to_tensor(ctx_list, ump_scalar=ump_val)
                )

                row_keys_g.append((
                    int(row.get("game_pk")),
                    int(_pid),
                    int(_safe_int(row.get("at_bat_number"), 0)),
                    int(_safe_int(row.get("pitch_number"), 0)),
                ))
                outcomes_g.append(
                    classify_pitch_outcome(row.get("description"), row.get("events"))
                )

            tokens = tokens[: self.max_seq_len]
            contexts = contexts[: self.max_seq_len]
            row_keys_g = row_keys_g[: self.max_seq_len]
            outcomes_g = outcomes_g[: self.max_seq_len]

            input_tokens = torch.tensor(tokens[:-1], dtype=torch.long)
            target_tokens = torch.tensor(tokens[1:], dtype=torch.long)
            context_tensor = torch.stack(contexts[:-1])
            if self.context_dim < context_tensor.shape[-1]:
                context_tensor = context_tensor[:, : self.context_dim]

            # outcome_labels[j] aligns with target_tokens[j] (outcome of
            # pitch at position j+1 in the original stream).
            outcomes_aligned = outcomes_g[1:]
            row_keys_aligned = row_keys_g[1:]

            # Decode target token components (used as concat features at j).
            t_pt = []
            t_zn = []
            t_vl = []
            for tok in tokens[1:]:
                if 0 <= tok < VOCAB_SIZE:
                    velo = tok % NUM_VELO_BUCKETS
                    rem = tok // NUM_VELO_BUCKETS
                    zone = rem % NUM_ZONES
                    pt = rem // NUM_ZONES
                    t_pt.append(pt)
                    t_zn.append(zone)
                    t_vl.append(velo)
                else:
                    t_pt.append(NUM_PITCH_TYPES - 1)
                    t_zn.append(NUM_ZONES - 1)
                    t_vl.append(2)

            sequences.append((input_tokens, context_tensor, target_tokens))
            row_keys.append(np.asarray(row_keys_aligned, dtype=np.int64))
            tgt_pt.append(np.asarray(t_pt, dtype=np.int64))
            tgt_zn.append(np.asarray(t_zn, dtype=np.int64))
            tgt_vl.append(np.asarray(t_vl, dtype=np.int64))
            outc_lbl.append(np.asarray(outcomes_aligned, dtype=np.int64))

        self.sequences = sequences
        self.row_key_arrays = row_keys
        self.target_pt_idx = tgt_pt
        self.target_zone_idx = tgt_zn
        self.target_velo_idx = tgt_vl
        self.outcome_labels = outc_lbl
        self.game_pks = {int(rk[0, 0]) for rk in row_keys if len(rk) > 0}
        self.pitcher_ids = {int(rk[0, 1]) for rk in row_keys if len(rk) > 0}

        n_total = sum(len(o) for o in outc_lbl)
        n_unk = sum(int((o == OUTCOME_UNK).sum()) for o in outc_lbl)
        logger.info(
            "FixedGamesSequenceDataset: %d sequences, %d positions, %d UNK (%.2f%%)",
            len(sequences), n_total, n_unk, 100.0 * n_unk / max(n_total, 1),
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        tokens, ctx, targets = self.sequences[idx]
        return (
            tokens,
            ctx,
            targets,
            torch.from_numpy(self.outcome_labels[idx]).long(),
            torch.from_numpy(self.target_pt_idx[idx]).long(),
            torch.from_numpy(self.target_zone_idx[idx]).long(),
            torch.from_numpy(self.target_velo_idx[idx]).long(),
            self.row_key_arrays[idx],  # numpy, not stacked in collate
        )


def _a1_collate(batch):
    """Pad variable-length sequences to the longest in the batch."""
    max_len = max(item[0].size(0) for item in batch)

    tok_b, ctx_b, tgt_b, out_b = [], [], [], []
    pt_b, zn_b, vl_b = [], [], []
    rk_b = []  # list of numpy arrays (untouched)

    for tokens, ctx, targets, outcomes, pt, zn, vl, rk in batch:
        seq_len = tokens.size(0)
        pad_len = max_len - seq_len

        tokens_p = torch.cat(
            [tokens, torch.full((pad_len,), PAD_TOKEN, dtype=torch.long)]
        )
        targets_p = torch.cat(
            [targets, torch.full((pad_len,), PAD_TOKEN, dtype=torch.long)]
        )
        outcomes_p = torch.cat(
            [outcomes, torch.full((pad_len,), OUTCOME_UNK, dtype=torch.long)]
        )
        pt_p = torch.cat(
            [pt, torch.full((pad_len,), NUM_PITCH_TYPES - 1, dtype=torch.long)]
        )
        zn_p = torch.cat(
            [zn, torch.full((pad_len,), NUM_ZONES - 1, dtype=torch.long)]
        )
        vl_p = torch.cat(
            [vl, torch.full((pad_len,), 2, dtype=torch.long)]
        )
        if pad_len > 0:
            ctx_p = torch.cat([ctx, torch.zeros(pad_len, ctx.size(-1))])
        else:
            ctx_p = ctx

        tok_b.append(tokens_p)
        ctx_b.append(ctx_p)
        tgt_b.append(targets_p)
        out_b.append(outcomes_p)
        pt_b.append(pt_p)
        zn_b.append(zn_p)
        vl_b.append(vl_p)
        rk_b.append(rk)  # not stacked; per-batch list

    return (
        torch.stack(tok_b),
        torch.stack(ctx_b),
        torch.stack(tgt_b),
        torch.stack(out_b),
        torch.stack(pt_b),
        torch.stack(zn_b),
        torch.stack(vl_b),
        rk_b,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Class weights helper
# ═════════════════════════════════════════════════════════════════════════════


def compute_class_weights(
    train_targets: np.ndarray,
    cap: float = DEFAULT_CLASS_WEIGHT_CAP,
) -> np.ndarray:
    t = train_targets[train_targets != OUTCOME_UNK]
    counts = np.bincount(t, minlength=NUM_OUTCOME_CLASSES).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return np.ones(NUM_OUTCOME_CLASSES, dtype=np.float32)
    freqs = counts / total
    inv = np.where(freqs > 0, 1.0 / np.clip(freqs, 1e-12, None), cap)
    inv = np.minimum(inv, cap)
    inv = inv * (NUM_OUTCOME_CLASSES / inv.sum())
    return inv.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════


def train_a1(
    backbone: PitchGPTModel,
    train_ds: FixedGamesSequenceDataset,
    val_ds: FixedGamesSequenceDataset,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    class_weights: torch.Tensor,
    device: torch.device,
) -> tuple[FrozenOutcomeHeadConcat, list[dict], int]:
    """Train the A1 concat head on a frozen backbone with early-stopping."""
    freeze_backbone(backbone)
    backbone.eval()
    backbone.to(device)
    n_bb_grad = sum(1 for p in backbone.parameters() if p.requires_grad)
    assert n_bb_grad == 0, f"Backbone has {n_bb_grad} trainable params"

    head = FrozenOutcomeHeadConcat(
        d_model=backbone.d_model,
        context_dim=backbone.context_dim,
        n_pitch_types=NUM_PITCH_TYPES,
        n_zones=NUM_ZONES,
        n_velo_buckets=NUM_VELO_BUCKETS,
        hidden_dims=(128, 64),
        dropout=0.1,
        n_classes=NUM_OUTCOME_CLASSES,
    ).to(device)

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss(
        ignore_index=OUTCOME_UNK,
        weight=class_weights.to(device),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_a1_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_a1_collate,
    )

    history: list[dict] = []
    best_val_ll = float("inf")
    best_state = copy.deepcopy(head.state_dict())
    best_epoch = 0
    epochs_since_improve = 0

    for epoch in range(1, epochs + 1):
        head.train()
        total_loss, total_batches = 0.0, 0
        t0 = time.time()
        for batch_idx, batch in enumerate(train_loader):
            (tokens, ctx, _tgt, outcomes,
             pt_idx, zn_idx, vl_idx, _rk) = batch
            tokens = tokens.to(device)
            ctx = ctx.to(device)
            outcomes = outcomes.to(device)
            pt_idx = pt_idx.to(device)
            zn_idx = zn_idx.to(device)
            vl_idx = vl_idx.to(device)

            with torch.no_grad():
                hidden = extract_backbone_hidden_states(backbone, tokens, ctx)

            pt_oh = F.one_hot(pt_idx, num_classes=NUM_PITCH_TYPES).float()
            zn_oh = F.one_hot(zn_idx, num_classes=NUM_ZONES).float()
            vl_oh = F.one_hot(vl_idx, num_classes=NUM_VELO_BUCKETS).float()

            logits = head(hidden, ctx, pt_oh, zn_oh, vl_oh)
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

            if batch_idx % 200 == 0:
                logger.info(
                    "  [epoch %d] batch %d/%d  train_loss=%.4f",
                    epoch, batch_idx, len(train_loader),
                    float(loss.item()),
                )

        train_loss = total_loss / max(total_batches, 1)

        # Val unweighted log-loss for reporting.
        head.eval()
        total_val_ll = 0.0
        total_val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                (tokens, ctx, _tgt, outcomes,
                 pt_idx, zn_idx, vl_idx, _rk) = batch
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
                logits = head(hidden, ctx, pt_oh, zn_oh, vl_oh)

                flat_logits = logits.reshape(-1, NUM_OUTCOME_CLASSES)
                flat_out = outcomes.reshape(-1)
                valid = flat_out != OUTCOME_UNK
                n = int(valid.sum())
                if n == 0:
                    continue
                logits_v = flat_logits[valid]
                out_v = flat_out[valid]
                log_probs = F.log_softmax(logits_v, dim=-1)
                nll = -log_probs.gather(1, out_v.unsqueeze(1)).squeeze(1)
                total_val_ll += float(nll.sum().item())
                total_val_n += n
        val_ll = total_val_ll / max(total_val_n, 1)

        dt = time.time() - t0
        history.append({
            "epoch": epoch,
            "train_loss_weighted": round(train_loss, 4),
            "val_log_loss": round(val_ll, 4),
            "seconds": round(dt, 1),
        })
        logger.info(
            "Epoch %d/%d  train_loss(w)=%.4f  val_ll=%.4f  %.1fs  (best %.4f @ ep%d)",
            epoch, epochs, train_loss, val_ll, dt,
            best_val_ll if best_val_ll < float("inf") else float("nan"),
            best_epoch,
        )

        if val_ll < best_val_ll - 1e-5:
            best_val_ll = val_ll
            best_state = copy.deepcopy(head.state_dict())
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d, best %.4f @ ep%d).",
                    epoch, patience, best_val_ll, best_epoch,
                )
                break

    head.load_state_dict(best_state)
    return head, history, best_epoch


# ═════════════════════════════════════════════════════════════════════════════
# Inference: collect per-row logits + row_keys
# ═════════════════════════════════════════════════════════════════════════════


@torch.no_grad()
def collect_logits(
    backbone: PitchGPTModel,
    head: FrozenOutcomeHeadConcat,
    ds: FixedGamesSequenceDataset,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Produce a flat (N, 7) logits matrix + (N,) targets + (N, 4) row_keys.

    row_keys columns: (game_pk, pitcher_id, at_bat_number, pitch_number).
    """
    backbone.eval()
    head.eval()
    backbone.to(device)
    head.to(device)

    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, collate_fn=_a1_collate,
    )
    logits_chunks: list[np.ndarray] = []
    targets_chunks: list[np.ndarray] = []
    row_key_chunks: list[np.ndarray] = []

    for batch in loader:
        (tokens, ctx, _tgt, outcomes,
         pt_idx, zn_idx, vl_idx, rk_list) = batch
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
        out_flat = outcomes.reshape(-1)
        log_flat = logits.reshape(-1, NUM_OUTCOME_CLASSES)
        valid = out_flat != OUTCOME_UNK
        if int(valid.sum()) == 0:
            continue

        # Match row_keys by reconstructing per-position keys (one row per
        # B,S position).  rk_list[i] is the (S_i, 4) numpy array (no
        # padding); the collate doesn't pad row_keys.  We reconstruct
        # per-position by building flat indices in the same order
        # (B, S) flatten matches sequence iteration.
        B, S = outcomes.shape
        # Build per-position row_keys of shape (B, S, 4) with -1 fill.
        rk_full = np.full((B, S, 4), -1, dtype=np.int64)
        for bi in range(B):
            arr = rk_list[bi]
            n_i = arr.shape[0]
            rk_full[bi, :n_i, :] = arr
        rk_flat = rk_full.reshape(-1, 4)

        valid_np = valid.cpu().numpy()
        logits_v = log_flat[valid].cpu().numpy()
        targets_v = out_flat[valid].cpu().numpy()
        rk_v = rk_flat[valid_np]

        logits_chunks.append(logits_v)
        targets_chunks.append(targets_v)
        row_key_chunks.append(rk_v)

    if not logits_chunks:
        return {
            "logits": np.zeros((0, NUM_OUTCOME_CLASSES), dtype=np.float32),
            "targets": np.zeros((0,), dtype=np.int64),
            "row_keys": np.zeros((0, 4), dtype=np.int64),
        }
    return {
        "logits": np.concatenate(logits_chunks, axis=0).astype(np.float32),
        "targets": np.concatenate(targets_chunks, axis=0).astype(np.int64),
        "row_keys": np.concatenate(row_key_chunks, axis=0).astype(np.int64),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Paired bootstrap on A1 vs A3 delta
# ═════════════════════════════════════════════════════════════════════════════


def merge_against_a3_cache(
    a1_logits: np.ndarray,
    a1_targets: np.ndarray,
    a1_row_keys: np.ndarray,  # (N, 4) game_pk, pitcher_id, ab, pnum
    a3_test_df: pd.DataFrame,
    a3_test_probs: np.ndarray,  # (M, 7) — A3's row-aligned post-T probs
) -> dict[str, Any]:
    """Align A1 predictions to A3's cached test rows.

    A3's cohort cache has the same rows as `a3_test_df` IN ORDER.
    Returns a paired set of rows where (game_pk, at_bat, pitch_number)
    appears in both A1 and A3.  Note: A3's cached parquet does NOT carry
    at_bat / pitch_number columns directly — but the cohort builder's
    SQL ORDER BY guarantees row-by-row order matches what we'd get from
    a fresh re-fetch using the same game_pks.  We need to rebuild the
    A3 row keys from a fresh SQL fetch (or store them in a side table).

    This function takes a fully-keyed A3 frame (test_df_with_keys).
    """
    # Build an index from A3 (game_pk, at_bat, pnum) -> row index.
    # We're agnostic to pitcher_id at the join key (game_pk + at_bat +
    # pitch_number is unique).
    a3_keys = a3_test_df[
        ["game_pk", "at_bat_number", "pitch_number"]
    ].astype(np.int64).to_numpy()
    a3_idx = {}
    for i in range(len(a3_keys)):
        k = (int(a3_keys[i, 0]), int(a3_keys[i, 1]), int(a3_keys[i, 2]))
        a3_idx[k] = i

    matched_a1 = []
    matched_a3 = []
    for i in range(len(a1_row_keys)):
        gk = int(a1_row_keys[i, 0])
        ab = int(a1_row_keys[i, 2])
        pn = int(a1_row_keys[i, 3])
        k = (gk, ab, pn)
        if k in a3_idx:
            matched_a1.append(i)
            matched_a3.append(a3_idx[k])

    matched_a1 = np.asarray(matched_a1, dtype=np.int64)
    matched_a3 = np.asarray(matched_a3, dtype=np.int64)

    return {
        "n_matched": int(len(matched_a1)),
        "n_a1": int(len(a1_targets)),
        "n_a3": int(len(a3_keys)),
        "a1_idx": matched_a1,
        "a3_idx": matched_a3,
    }


def paired_bootstrap_lift_delta(
    a1_probs: np.ndarray, a3_probs: np.ndarray,
    y: np.ndarray, freq_prior: np.ndarray,
    n_boot: int = 1000, seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for (lift_a1 - lift_a3) on paired rows.

    Pairing is preserved by sampling indices used by both probs
    matrices.  Lift = 1 - log_loss(model) / log_loss(freq_prior).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    if n == 0:
        return {
            "delta_point": float("nan"),
            "delta_ci_lo": float("nan"),
            "delta_ci_hi": float("nan"),
            "lift_a1_point": float("nan"),
            "lift_a3_point": float("nan"),
        }

    def lift_for(probs: np.ndarray, y_b: np.ndarray) -> float:
        m_ll = log_loss_from_probs(probs, y_b)
        p_ll = freq_prior_log_loss_on(y_b, freq_prior)
        if p_ll <= 0:
            return float("nan")
        return 1.0 - m_ll / p_ll

    lift_a1_point = lift_for(a1_probs, y)
    lift_a3_point = lift_for(a3_probs, y)
    delta_point = lift_a1_point - lift_a3_point

    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ll_a1 = log_loss_from_probs(a1_probs[idx], y[idx])
        ll_a3 = log_loss_from_probs(a3_probs[idx], y[idx])
        ll_p = freq_prior_log_loss_on(y[idx], freq_prior)
        if ll_p <= 0:
            deltas[b] = float("nan")
        else:
            l_a1 = 1.0 - ll_a1 / ll_p
            l_a3 = 1.0 - ll_a3 / ll_p
            deltas[b] = l_a1 - l_a3
    deltas = deltas[~np.isnan(deltas)]
    if len(deltas) == 0:
        return {
            "delta_point": float(delta_point),
            "delta_ci_lo": float("nan"),
            "delta_ci_hi": float("nan"),
            "lift_a1_point": float(lift_a1_point),
            "lift_a3_point": float(lift_a3_point),
        }
    lo = float(np.percentile(deltas, 2.5))
    hi = float(np.percentile(deltas, 97.5))
    return {
        "delta_point": float(delta_point),
        "delta_ci_lo": lo,
        "delta_ci_hi": hi,
        "lift_a1_point": float(lift_a1_point),
        "lift_a3_point": float(lift_a3_point),
        "boot_mean": float(np.mean(deltas)),
        "boot_std": float(np.std(deltas)),
    }


def paired_bootstrap_log_loss_delta(
    a1_probs: np.ndarray, a3_probs: np.ndarray, y: np.ndarray,
    n_boot: int = 1000, seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for (log_loss_a1 - log_loss_a3) on paired rows."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if n == 0:
        return {"delta_point": float("nan"), "delta_ci_lo": float("nan"),
                "delta_ci_hi": float("nan")}
    point = log_loss_from_probs(a1_probs, y) - log_loss_from_probs(a3_probs, y)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[b] = (
            log_loss_from_probs(a1_probs[idx], y[idx])
            - log_loss_from_probs(a3_probs[idx], y[idx])
        )
    return {
        "delta_point": float(point),
        "delta_ci_lo": float(np.percentile(deltas, 2.5)),
        "delta_ci_hi": float(np.percentile(deltas, 97.5)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-games", type=int, default=DEFAULT_TRAIN_GAMES)
    p.add_argument("--val-games", type=int, default=DEFAULT_VAL_GAMES)
    p.add_argument("--test-games", type=int, default=DEFAULT_TEST_GAMES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--class-weight-cap", type=float,
                   default=DEFAULT_CLASS_WEIGHT_CAP)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--checkpoint", type=str, default=str(A1_CKPT))
    args = p.parse_args(argv)

    _set_seed(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = _get_device()
    logger.info("device=%s  train_games=%d  val_games=%d  test_games=%d  seed=%d",
                device, args.train_games, args.val_games, args.test_games, args.seed)

    # Step 1: pre-run SHA256 of v2.pt (frozen-ness guard).
    logger.info("Pre-run SHA256 of %s ...", BACKBONE_PATH)
    v2_sha_pre = _sha256_file(BACKBONE_PATH)
    v2_size_pre = BACKBONE_PATH.stat().st_size
    logger.info("v2.pt SHA256 (pre): %s  size=%d bytes",
                v2_sha_pre, v2_size_pre)

    # Step 2: load backbone, freeze.
    logger.info("Loading PitchGPT v2 backbone...")
    backbone = _load_model(version="2")
    backbone_param_sha_pre = _backbone_param_checksum(backbone)
    logger.info("Backbone d_model=%d ctx_dim=%d  param-SHA pre: %s",
                backbone.d_model, backbone.context_dim, backbone_param_sha_pre)

    # Step 3: cohort — replicate Plan B Step 1 game_pks deterministically.
    conn = get_connection(read_only=True)
    try:
        logger.info("Fetching train pitcher cohort (2015-2022)...")
        train_pitchers = fetch_pitcher_ids(conn, TRAIN_SEASONS)
        logger.info("Train cohort pitchers: %d", len(train_pitchers))

        logger.info("Sampling train games (seed=%d)...", args.seed)
        train_pks = sample_game_pks(
            conn, TRAIN_SEASONS, max_games=args.train_games, seed=args.seed,
        )
        logger.info("Train games selected: %d", len(train_pks))

        logger.info("Sampling val games (seed=%d, pitcher-disjoint)...", args.seed + 1)
        val_pks = sample_game_pks(
            conn, VAL_SEASONS, max_games=args.val_games,
            exclude_pitcher_ids=train_pitchers, seed=args.seed + 1,
        )
        logger.info("Val games selected: %d", len(val_pks))

        logger.info("Sampling test games (seed=%d, pitcher-disjoint)...", args.seed + 2)
        test_pks = sample_game_pks(
            conn, TEST_SEASONS, max_games=args.test_games,
            exclude_pitcher_ids=train_pitchers, seed=args.seed + 2,
        )
        logger.info("Test games selected: %d", len(test_pks))

        # Step 4: build sequence datasets restricted to those games.
        t0 = time.time()
        logger.info("Building train sequence dataset (%d games, "
                    "pitcher-disjoint NOT applied for train per protocol)...",
                    len(train_pks))
        train_ds = FixedGamesSequenceDataset(
            conn, game_pks=train_pks,
            exclude_pitcher_ids=None,
            max_seq_len=256, context_dim=backbone.context_dim,
        )
        logger.info("Train ds built: %d sequences  (%.1fs)",
                    len(train_ds), time.time() - t0)

        t0 = time.time()
        logger.info("Building val sequence dataset (drop train pitchers)...")
        val_ds = FixedGamesSequenceDataset(
            conn, game_pks=val_pks,
            exclude_pitcher_ids=train_pitchers,
            max_seq_len=256, context_dim=backbone.context_dim,
        )
        logger.info("Val ds built: %d sequences  (%.1fs)",
                    len(val_ds), time.time() - t0)

        t0 = time.time()
        logger.info("Building test sequence dataset (drop train pitchers)...")
        test_ds = FixedGamesSequenceDataset(
            conn, game_pks=test_pks,
            exclude_pitcher_ids=train_pitchers,
            max_seq_len=256, context_dim=backbone.context_dim,
        )
        logger.info("Test ds built: %d sequences  (%.1fs)",
                    len(test_ds), time.time() - t0)

        # Also fetch A3's test_df row keys for paired join.
        logger.info("Re-fetching A3 test row keys for paired bootstrap...")
        gpks_str_chunks = []
        for start in range(0, len(test_pks), 5000):
            end = min(start + 5000, len(test_pks))
            gpks_str_chunks.append(", ".join(str(int(g)) for g in test_pks[start:end]))
        # We'll match by (game_pk, at_bat_number, pitch_number) — re-query.
        a3_test_keys = []
        excl_str = ""
        if train_pitchers:
            pids_str = ", ".join(str(int(p)) for p in train_pitchers)
            excl_str = f"AND p.pitcher_id NOT IN ({pids_str})"
        for gchunk in gpks_str_chunks:
            q = f"""
                SELECT
                    p.game_pk, p.at_bat_number, p.pitch_number,
                    p.description, p.events
                FROM pitches p
                WHERE p.pitch_type IS NOT NULL
                  AND p.game_pk IN ({gchunk})
                  {excl_str}
                ORDER BY p.game_pk, p.at_bat_number, p.pitch_number
            """
            cdf = conn.execute(q).fetchdf()
            if not cdf.empty:
                a3_test_keys.append(cdf)
        a3_keys_df = pd.concat(a3_test_keys, ignore_index=True) if a3_test_keys else pd.DataFrame()
        # Apply same UNK filter as Plan B common.
        if not a3_keys_df.empty:
            labels = [
                classify_pitch_outcome(d, e)
                for d, e in zip(a3_keys_df["description"].tolist(),
                                 a3_keys_df["events"].tolist())
            ]
            a3_keys_df["outcome_label"] = labels
            a3_keys_df = a3_keys_df[
                a3_keys_df["outcome_label"] != OUTCOME_UNK
            ].reset_index(drop=True)
        logger.info("A3 test rows reconstructed: %d", len(a3_keys_df))
    finally:
        conn.close()

    # Step 5: collect targets, build class weights.
    train_targets_all = np.concatenate(
        [t for t in train_ds.outcome_labels]
    ) if train_ds.outcome_labels else np.array([], dtype=np.int64)
    val_targets_all = np.concatenate(
        [t for t in val_ds.outcome_labels]
    ) if val_ds.outcome_labels else np.array([], dtype=np.int64)

    train_valid = train_targets_all[train_targets_all != OUTCOME_UNK]
    val_valid = val_targets_all[val_targets_all != OUTCOME_UNK]
    logger.info("Train valid outcomes: %d  Val valid outcomes: %d",
                len(train_valid), len(val_valid))

    class_weights_np = compute_class_weights(
        train_valid, cap=args.class_weight_cap,
    )
    class_weights_t = torch.from_numpy(class_weights_np)
    logger.info("Class weights (inv-freq cap=%g): %s",
                args.class_weight_cap,
                {OUTCOME_CLASSES[i]: round(float(class_weights_np[i]), 3)
                 for i in range(NUM_OUTCOME_CLASSES)})

    train_dist = np.bincount(train_valid, minlength=NUM_OUTCOME_CLASSES) / max(
        len(train_valid), 1,
    )
    frequency_prior = train_dist.copy()

    # Step 6: train.
    logger.info("=== Starting A1 (frozen + concat MLP head) training ===")
    t_train_start = time.time()
    head, history, best_epoch = train_a1(
        backbone, train_ds, val_ds,
        epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size, lr=args.lr,
        class_weights=class_weights_t,
        device=device,
    )
    train_wall = time.time() - t_train_start
    logger.info("Training done in %.1fs.  Best epoch=%d", train_wall, best_epoch)

    # Step 7: backbone byte-identity check.
    backbone_param_sha_post = _backbone_param_checksum(backbone)
    bb_param_match = backbone_param_sha_pre == backbone_param_sha_post
    assert bb_param_match, (
        f"Backbone param drift! pre={backbone_param_sha_pre} "
        f"post={backbone_param_sha_post}"
    )
    logger.info("Backbone param-SHA post: %s (MATCH=%s)",
                backbone_param_sha_post, bb_param_match)

    # Step 8: collect val + test logits.
    logger.info("Collecting val logits ...")
    val_dump = collect_logits(backbone, head, val_ds, args.batch_size, device)
    logger.info("Val logits: %s targets: %d",
                val_dump["logits"].shape, val_dump["targets"].size)

    logger.info("Collecting test logits ...")
    test_dump = collect_logits(backbone, head, test_ds, args.batch_size, device)
    logger.info("Test logits: %s targets: %d",
                test_dump["logits"].shape, test_dump["targets"].size)

    # Step 9: temperature scaling on val (probs → log probs → temperature).
    if val_dump["logits"].shape[0] > 0:
        T_opt = fit_temperature(val_dump["logits"], val_dump["targets"])
    else:
        T_opt = 1.0
    logger.info("Fitted temperature: %.4f", T_opt)

    val_probs_pre = softmax(val_dump["logits"])
    test_probs_pre = softmax(test_dump["logits"])
    val_probs_post = softmax(val_dump["logits"] / T_opt)
    test_probs_post = softmax(test_dump["logits"] / T_opt)

    # Step 10: metrics on val + test.
    val_freq_ll = freq_prior_log_loss_on(val_dump["targets"], frequency_prior)
    test_freq_ll = freq_prior_log_loss_on(test_dump["targets"], frequency_prior)

    val_ll_pre = log_loss_from_probs(val_probs_pre, val_dump["targets"])
    val_ll_post = log_loss_from_probs(val_probs_post, val_dump["targets"])
    test_ll_pre = log_loss_from_probs(test_probs_pre, test_dump["targets"])
    test_ll_post = log_loss_from_probs(test_probs_post, test_dump["targets"])

    val_ece_pre = ece_10bin(val_probs_pre, val_dump["targets"])
    val_ece_post = ece_10bin(val_probs_post, val_dump["targets"])
    test_ece_pre = ece_10bin(test_probs_pre, test_dump["targets"])
    test_ece_post = ece_10bin(test_probs_post, test_dump["targets"])

    val_lift_post = 1.0 - val_ll_post / max(val_freq_ll, 1e-12)
    test_lift_pre = 1.0 - test_ll_pre / max(test_freq_ll, 1e-12)
    test_lift_post = 1.0 - test_ll_post / max(test_freq_ll, 1e-12)

    test_acc = top1_accuracy(test_probs_post, test_dump["targets"])
    test_per_class = per_class_log_loss(test_probs_post, test_dump["targets"])

    pitcher_ids_test = test_dump["row_keys"][:, 1].astype(np.int64)
    pitcher_ll = per_pitcher_log_loss(
        test_probs_post, test_dump["targets"], pitcher_ids_test, top_k=50,
    )
    if pitcher_ll:
        plls = [r["log_loss"] for r in pitcher_ll]
        pitcher_var = float(np.var(plls))
        pitcher_min = float(min(plls))
        pitcher_max = float(max(plls))
        pitcher_mean = float(np.mean(plls))
    else:
        pitcher_var = pitcher_min = pitcher_max = pitcher_mean = float("nan")

    # Step 11: bootstrap CI on test log-loss + lift.
    logger.info("Bootstrapping log-loss + lift CI (B=%d)...", args.n_boot)
    t1 = time.time()
    test_ll_pt, test_ll_lo, test_ll_hi = bootstrap_log_loss(
        test_probs_post, test_dump["targets"],
        n_boot=args.n_boot, seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    n_t = len(test_dump["targets"])
    lift_samples = np.empty(args.n_boot)
    for b in range(args.n_boot):
        idx = rng.integers(0, n_t, size=n_t)
        ll_m = log_loss_from_probs(test_probs_post[idx], test_dump["targets"][idx])
        ll_p = freq_prior_log_loss_on(test_dump["targets"][idx], frequency_prior)
        lift_samples[b] = 1.0 - ll_m / max(ll_p, 1e-12)
    test_lift_lo = float(np.percentile(lift_samples, 2.5))
    test_lift_hi = float(np.percentile(lift_samples, 97.5))
    logger.info("Bootstrap done in %.1fs.", time.time() - t1)

    # Step 12: paired bootstrap on A1 - A3 delta.
    logger.info("Loading A3 metrics + cache for paired bootstrap...")
    with open(A3_METRICS_PATH, "r", encoding="utf-8") as f:
        a3_metrics = json.load(f)
    a3_test_freq_ll = a3_metrics["test_metrics"]["freq_prior_log_loss"]
    a3_test_lift_post = a3_metrics["test_metrics"]["lift_post_temp"]

    # We need A3's per-row probs.  Re-run A3 inference would be expensive;
    # instead reconstruct A3 row predictions by re-fitting an XGB at the
    # locked best params is also expensive.  Acceptable shortcut: use the
    # A3 temperature + cohort cache parquet to re-infer test_probs.  Load
    # A3 cache + retrain best model (or load from disk).  The A3 script
    # didn't dump probs to disk, so we must re-run inference.
    # Simpler: refit A3 at the locked best_max_depth/best_lr on the cached
    # cohort to get test_probs.  We refit because Plan B's metrics.json
    # gives us best params, and refitting is reproducible.
    logger.info("Re-fitting A3 at locked best params for paired-bootstrap probs...")
    a3_probs_post, a3_y, a3_test_df_with_keys = _refit_a3_for_probs(
        a3_metrics=a3_metrics, seed=args.seed,
    )
    a3_freq_prior = np.asarray(
        [a3_metrics["cohort"]["train_freq_prior"][c] for c in OUTCOME_CLASSES],
        dtype=np.float64,
    )

    # Match A1 rows to A3 rows.
    merge = merge_against_a3_cache(
        a1_logits=test_dump["logits"], a1_targets=test_dump["targets"],
        a1_row_keys=test_dump["row_keys"],
        a3_test_df=a3_test_df_with_keys, a3_test_probs=a3_probs_post,
    )
    logger.info("Paired rows: A1=%d  A3=%d  matched=%d",
                merge["n_a1"], merge["n_a3"], merge["n_matched"])

    if merge["n_matched"] >= 100:
        a1_paired_probs = test_probs_post[merge["a1_idx"]]
        a3_paired_probs = a3_probs_post[merge["a3_idx"]]
        y_paired = test_dump["targets"][merge["a1_idx"]]
        delta_lift = paired_bootstrap_lift_delta(
            a1_paired_probs, a3_paired_probs, y_paired,
            freq_prior=a3_freq_prior,
            n_boot=args.n_boot, seed=args.seed,
        )
        delta_ll = paired_bootstrap_log_loss_delta(
            a1_paired_probs, a3_paired_probs, y_paired,
            n_boot=args.n_boot, seed=args.seed,
        )
        logger.info("Paired delta-lift point: %+.4f%% CI [%+.4f%%, %+.4f%%]",
                    100 * delta_lift["delta_point"],
                    100 * delta_lift["delta_ci_lo"],
                    100 * delta_lift["delta_ci_hi"])
        logger.info("Paired log-loss delta (A1 - A3): %+.4f CI [%+.4f, %+.4f]",
                    delta_ll["delta_point"],
                    delta_ll["delta_ci_lo"], delta_ll["delta_ci_hi"])
    else:
        delta_lift = {
            "delta_point": float("nan"),
            "delta_ci_lo": float("nan"), "delta_ci_hi": float("nan"),
            "lift_a1_point": float("nan"), "lift_a3_point": float("nan"),
        }
        delta_ll = {
            "delta_point": float("nan"),
            "delta_ci_lo": float("nan"), "delta_ci_hi": float("nan"),
        }
        logger.warning("Insufficient paired rows (%d) for paired bootstrap.",
                       merge["n_matched"])

    # Step 13: gate evaluation.
    GATE_PASS_LIFT = 0.10
    GATE_PASS_CI_LO = 0.05
    GATE_WEAK_LIFT = 0.05
    GATE_WEAK_CI_LO = 0.02
    GATE_ECE_MAX = 0.05
    GATE_HIT_LL_PASS = 2.0
    GATE_HIT_LL_WEAK = 2.5
    GATE_HBP_LL_PASS = 4.0
    GATE_HBP_LL_WEAK = 5.0

    hit_ll = test_per_class.get("in_play_hit", float("nan"))
    hbp_ll = test_per_class.get("hbp", float("nan"))

    if (test_lift_post >= GATE_PASS_LIFT
            and test_lift_lo >= GATE_PASS_CI_LO
            and test_ece_post < GATE_ECE_MAX
            and hit_ll < GATE_HIT_LL_PASS
            and hbp_ll < GATE_HBP_LL_PASS):
        verdict_a1 = "PASS"
    elif (test_lift_post >= GATE_WEAK_LIFT
            and test_lift_lo >= GATE_WEAK_CI_LO
            and test_ece_post < GATE_ECE_MAX
            and hit_ll < GATE_HIT_LL_WEAK
            and hbp_ll < GATE_HBP_LL_WEAK):
        verdict_a1 = "WEAKER PASS"
    else:
        verdict_a1 = "FAIL"

    # Ship verdict (per RESEARCH_PLAN §7 kill criteria).
    # If A1 beats A3 by ≥ +1pp lift on 2025 holdout (paired CI lo > 0):
    #   SHIP A1
    # If A1 < A3 + 1pp OR CI includes 0:
    #   SHIP A3 (the protocol-locked outcome — confirms PG backbone adds
    #   no marginal value)
    a1_beats_a3_pp = 100.0 * (delta_lift.get("delta_point", float("nan")))
    a1_beats_ci_lo = 100.0 * (delta_lift.get("delta_ci_lo", float("nan")))
    if (not np.isnan(a1_beats_a3_pp)
            and a1_beats_a3_pp >= 1.0
            and a1_beats_ci_lo > 0.0):
        ship_verdict = "SHIP A1"
    else:
        # The A4 logistic at +17.35% is the actual leader vs A3 +16.12%, so
        # if A1 doesn't beat A3 then we should compare to A4.  But the
        # instructions explicitly: "Beats A3 by <+1pp OR A3 wins → SHIP A3".
        # We follow the instruction literally; downstream operators may
        # prefer A4.
        ship_verdict = "SHIP A3"

    logger.info("=" * 60)
    logger.info("A1 RESULTS")
    logger.info("=" * 60)
    logger.info("VAL  pre/post log-loss: %.4f / %.4f  freq=%.4f  lift_post=%.2f%%",
                val_ll_pre, val_ll_post, val_freq_ll, 100 * val_lift_post)
    logger.info("VAL  pre/post ECE:      %.4f / %.4f", val_ece_pre, val_ece_post)
    logger.info("TEST pre/post log-loss: %.4f / %.4f  freq=%.4f  "
                "lift_post=%.2f%% [%.2f%%, %.2f%%]",
                test_ll_pre, test_ll_post, test_freq_ll,
                100 * test_lift_post,
                100 * test_lift_lo, 100 * test_lift_hi)
    logger.info("TEST pre/post ECE:      %.4f / %.4f", test_ece_pre, test_ece_post)
    logger.info("TEST per-class log-loss: %s",
                {k: round(v, 4) for k, v in test_per_class.items()})
    logger.info("TEST top-1 accuracy: %.4f", test_acc)
    logger.info("Per-pitcher (top-50) ll: mean=%.4f var=%.4f range=[%.4f, %.4f]",
                pitcher_mean, pitcher_var, pitcher_min, pitcher_max)
    logger.info("PAIRED A1 - A3 LIFT DELTA: %+.4f%% CI [%+.4f%%, %+.4f%%]",
                100 * delta_lift["delta_point"],
                100 * delta_lift["delta_ci_lo"],
                100 * delta_lift["delta_ci_hi"])
    logger.info("PAIRED log-loss delta (A1 - A3): %+.4f CI [%+.4f, %+.4f]",
                delta_ll["delta_point"],
                delta_ll["delta_ci_lo"], delta_ll["delta_ci_hi"])
    logger.info("BACKBONE byte-identity verified: %s", bb_param_match)
    logger.info("VERDICT (A1 standalone): %s", verdict_a1)
    logger.info("FINAL SHIP VERDICT: %s", ship_verdict)

    # Step 14: save checkpoint.
    ckpt_path = Path(args.checkpoint)
    logger.info("Saving A1 head checkpoint to %s", ckpt_path)
    ckpt = {
        "head_state_dict": head.state_dict(),
        "config": {
            "head_class": "FrozenOutcomeHeadConcat",
            "d_model": backbone.d_model,
            "context_dim": backbone.context_dim,
            "n_pitch_types": NUM_PITCH_TYPES,
            "n_zones": NUM_ZONES,
            "n_velo_buckets": NUM_VELO_BUCKETS,
            "hidden_dims": [128, 64],
            "dropout": 0.1,
            "n_classes": NUM_OUTCOME_CLASSES,
            "class_names": list(OUTCOME_CLASSES),
            "backbone_path": str(BACKBONE_PATH),
            "backbone_sha256": v2_sha_pre,
            "architecture": "frozen_v2_concat_211_128_64_7",
            "train_games": args.train_games,
            "val_games": args.val_games,
            "test_games": args.test_games,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs_requested": args.epochs,
            "patience": args.patience,
            "seed": args.seed,
        },
        "class_weights": class_weights_np,
        "frequency_prior": frequency_prior,
        "temperature": float(T_opt),
        "best_epoch": best_epoch,
        "training_history": history,
        "val_metrics": {
            "log_loss_pre_temp": val_ll_pre,
            "log_loss_post_temp": val_ll_post,
            "ece_pre_temp": val_ece_pre,
            "ece_post_temp": val_ece_post,
            "freq_prior_log_loss": val_freq_ll,
            "lift_post_temp": val_lift_post,
            "temperature": T_opt,
        },
        "test_metrics": {
            "log_loss_pre_temp": test_ll_pre,
            "log_loss_post_temp": test_ll_post,
            "ece_pre_temp": test_ece_pre,
            "ece_post_temp": test_ece_post,
            "freq_prior_log_loss": test_freq_ll,
            "lift_pre_temp": test_lift_pre,
            "lift_post_temp": test_lift_post,
            "log_loss_post_ci_lo": test_ll_lo,
            "log_loss_post_ci_hi": test_ll_hi,
            "lift_ci_lo": test_lift_lo,
            "lift_ci_hi": test_lift_hi,
            "per_class_log_loss": test_per_class,
            "top1_accuracy": test_acc,
            "per_pitcher_log_loss": pitcher_ll,
            "per_pitcher_summary": {
                "mean": pitcher_mean, "var": pitcher_var,
                "min": pitcher_min, "max": pitcher_max,
                "n_pitchers": len(pitcher_ll),
            },
        },
        "paired_vs_a3": {
            "n_a1": merge["n_a1"], "n_a3": merge["n_a3"],
            "n_matched": merge["n_matched"],
            "delta_lift": delta_lift,
            "delta_log_loss": delta_ll,
        },
        "train_wall_seconds": round(train_wall, 1),
        "backbone_param_sha256_pre": backbone_param_sha_pre,
        "backbone_param_sha256_post": backbone_param_sha_post,
        "backbone_byte_identity": bb_param_match,
    }
    torch.save(ckpt, ckpt_path)
    logger.info("Saved A1 checkpoint (%.1f MB)",
                ckpt_path.stat().st_size / (1024 * 1024))

    # Step 15: post-run SHA256 of v2.pt (must equal pre).
    v2_sha_post = _sha256_file(BACKBONE_PATH)
    v2_size_post = BACKBONE_PATH.stat().st_size
    assert v2_sha_pre == v2_sha_post, (
        f"v2.pt modified during training! pre={v2_sha_pre} post={v2_sha_post}"
    )
    assert v2_size_pre == v2_size_post
    logger.info("v2.pt byte-identity verified: PRE == POST")

    # Step 15b: confirm Phase 0.3 checkpoint untouched.
    phase03_exists = PHASE03_CKPT.exists()
    if phase03_exists:
        phase03_sha = _sha256_file(PHASE03_CKPT)
        logger.info("Phase 0.3 ckpt %s exists.  SHA256: %s",
                    PHASE03_CKPT, phase03_sha)
    else:
        phase03_sha = None
        logger.warning("Phase 0.3 ckpt %s missing!", PHASE03_CKPT)

    # Step 16: write metrics.json + report.md.
    metrics = {
        "variant": "a1_concat",
        "config": {
            "train_games": args.train_games,
            "val_games": args.val_games,
            "test_games": args.test_games,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "epochs_requested": args.epochs,
            "patience": args.patience,
            "class_weight_cap": args.class_weight_cap,
            "seed": args.seed,
            "n_boot": args.n_boot,
            "device": str(device),
            "best_epoch": best_epoch,
            "temperature": float(T_opt),
            "class_weights": {
                OUTCOME_CLASSES[i]: float(class_weights_np[i])
                for i in range(NUM_OUTCOME_CLASSES)
            },
        },
        "cohort": {
            "train_sequences": len(train_ds),
            "val_sequences": len(val_ds),
            "test_sequences": len(test_ds),
            "train_valid_outcomes": int(len(train_valid)),
            "val_valid_outcomes": int(len(val_valid)),
            "test_valid_outcomes": int(test_dump["targets"].size),
            "train_freq_prior": {
                OUTCOME_CLASSES[i]: float(frequency_prior[i])
                for i in range(NUM_OUTCOME_CLASSES)
            },
            "n_unique_test_pitchers": int(len(set(pitcher_ids_test.tolist()))),
        },
        "training": {
            "history": history,
            "best_epoch": best_epoch,
            "wall_clock_seconds": round(train_wall, 1),
        },
        "val_metrics": {
            "log_loss_pre_temp": val_ll_pre,
            "log_loss_post_temp": val_ll_post,
            "ece_pre_temp": val_ece_pre,
            "ece_post_temp": val_ece_post,
            "freq_prior_log_loss": val_freq_ll,
            "lift_post_temp": val_lift_post,
            "temperature": float(T_opt),
        },
        "test_metrics": {
            "log_loss_pre_temp": test_ll_pre,
            "log_loss_post_temp": test_ll_post,
            "ece_pre_temp": test_ece_pre,
            "ece_post_temp": test_ece_post,
            "freq_prior_log_loss": test_freq_ll,
            "lift_pre_temp": test_lift_pre,
            "lift_post_temp": test_lift_post,
            "log_loss_post_ci_lo": test_ll_lo,
            "log_loss_post_ci_hi": test_ll_hi,
            "lift_ci_lo": test_lift_lo,
            "lift_ci_hi": test_lift_hi,
            "per_class_log_loss": test_per_class,
            "top1_accuracy": test_acc,
            "per_pitcher_log_loss": pitcher_ll,
            "per_pitcher_summary": {
                "mean": pitcher_mean, "var": pitcher_var,
                "min": pitcher_min, "max": pitcher_max,
                "n_pitchers": len(pitcher_ll),
            },
        },
        "vs_a3": {
            "a3_metrics_path": str(A3_METRICS_PATH),
            "a3_test_freq_ll": a3_test_freq_ll,
            "a3_test_lift_post": a3_test_lift_post,
            "n_paired_rows": merge["n_matched"],
            "n_a1_rows": merge["n_a1"],
            "n_a3_rows": merge["n_a3"],
            "delta_lift": delta_lift,
            "delta_log_loss": delta_ll,
        },
        "gates": {
            "verdict_a1": verdict_a1,
            "verdict_ship": ship_verdict,
            "PASS_lift_threshold": GATE_PASS_LIFT,
            "PASS_ci_lo_threshold": GATE_PASS_CI_LO,
            "WEAK_lift_threshold": GATE_WEAK_LIFT,
            "WEAK_ci_lo_threshold": GATE_WEAK_CI_LO,
            "ECE_max": GATE_ECE_MAX,
            "hit_ll_PASS_max": GATE_HIT_LL_PASS,
            "hit_ll_WEAK_max": GATE_HIT_LL_WEAK,
            "hbp_ll_PASS_max": GATE_HBP_LL_PASS,
            "hbp_ll_WEAK_max": GATE_HBP_LL_WEAK,
            "kill_check_a1_beats_a3_min_pp": 1.0,
            "a1_beats_a3_pp": float(a1_beats_a3_pp)
                if not np.isnan(a1_beats_a3_pp) else None,
        },
        "backbone_byte_identity": {
            "v2_pt_sha256_pre": v2_sha_pre,
            "v2_pt_sha256_post": v2_sha_post,
            "v2_pt_size_pre": v2_size_pre,
            "v2_pt_size_post": v2_size_post,
            "backbone_param_sha256_pre": backbone_param_sha_pre,
            "backbone_param_sha256_post": backbone_param_sha_post,
            "identity_verified": bool(bb_param_match
                                      and v2_sha_pre == v2_sha_post),
            "phase03_ckpt_present": phase03_exists,
            "phase03_ckpt_sha256": phase03_sha,
        },
        "checkpoint": {
            "path": str(ckpt_path),
            "size_bytes": int(ckpt_path.stat().st_size),
        },
    }
    out_path = OUT_DIR / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    logger.info("Wrote %s", out_path)

    _write_report(OUT_DIR / "report.md", metrics)
    logger.info("Wrote %s", OUT_DIR / "report.md")
    logger.info("DONE.  Verdict: %s | Ship: %s", verdict_a1, ship_verdict)
    return 0


def _refit_a3_for_probs(
    a3_metrics: dict, seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Refit A3 XGBoost at the locked best params, return (probs_post, y, test_df_with_keys).

    A3's metrics.json gives us best_max_depth, best_learning_rate,
    best_iteration, and the temperature.  Refit on the same cached
    cohort (parquet) and apply the same temperature to produce
    test_probs.

    Note: A3's cached parquet does NOT include game_pk / at_bat /
    pitch_number columns — we re-fetch those by re-running the cohort
    SQL with the same game_pks + same UNK filter, in the same order.
    """
    from scripts.pitchgpt_outcome_baselines_common import (  # noqa: WPS433
        build_cohort,
    )

    train_df, val_df, test_df = build_cohort(
        train_games=a3_metrics["config"]["train_games"],
        val_games=a3_metrics["config"]["val_games"],
        test_games=a3_metrics["config"]["test_games"],
        seed=a3_metrics["config"]["seed"],
        cache_dir=A3_CACHE_DIR,
    )

    # Re-fetch row keys for test_df in the same SQL order.
    conn = get_connection(read_only=True)
    try:
        train_pitchers = fetch_pitcher_ids(conn, TRAIN_SEASONS)
        test_pks = sample_game_pks(
            conn, TEST_SEASONS,
            max_games=a3_metrics["config"]["test_games"],
            exclude_pitcher_ids=train_pitchers,
            seed=a3_metrics["config"]["seed"] + 2,
        )
        excl_str = ""
        if train_pitchers:
            pids_str = ", ".join(str(int(p)) for p in train_pitchers)
            excl_str = f"AND p.pitcher_id NOT IN ({pids_str})"
        chunks = []
        for start in range(0, len(test_pks), 5000):
            end = min(start + 5000, len(test_pks))
            gpk_chunk = ", ".join(str(int(g)) for g in test_pks[start:end])
            q = f"""
                SELECT
                    p.game_pk, p.pitcher_id, p.batter_id,
                    p.at_bat_number, p.pitch_number,
                    p.description, p.events
                FROM pitches p
                WHERE p.pitch_type IS NOT NULL
                  AND p.game_pk IN ({gpk_chunk})
                  {excl_str}
                ORDER BY p.game_pk, p.at_bat_number, p.pitch_number
            """
            cdf = conn.execute(q).fetchdf()
            if not cdf.empty:
                chunks.append(cdf)
        keys_df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        if not keys_df.empty:
            labels = [
                classify_pitch_outcome(d, e)
                for d, e in zip(keys_df["description"].tolist(),
                                 keys_df["events"].tolist())
            ]
            keys_df["outcome_label"] = labels
            keys_df = keys_df[
                keys_df["outcome_label"] != OUTCOME_UNK
            ].reset_index(drop=True)
    finally:
        conn.close()

    # Sanity: keys_df row count must match test_df row count.
    assert len(keys_df) == len(test_df), (
        f"A3 row count mismatch: keys={len(keys_df)}  test={len(test_df)}"
    )

    # Refit XGB at best params.
    import xgboost as xgb
    from scripts.pitchgpt_outcome_a3_xgboost import make_xgb_features

    feats = make_xgb_features(train_df, val_df, test_df)
    counts = np.bincount(
        feats["y_train"], minlength=NUM_OUTCOME_CLASSES,
    ).astype(np.float64)
    freq = counts / counts.sum()
    inv = np.minimum(1.0 / np.clip(freq, 1e-12, None), 10.0)
    inv = inv * (NUM_OUTCOME_CLASSES / inv.sum())
    sample_weight = inv[feats["y_train"]].astype(np.float32)

    best_md = a3_metrics["config"]["best_max_depth"]
    best_lr = a3_metrics["config"]["best_learning_rate"]
    n_est = a3_metrics["config"]["n_estimators"]
    early_stop = a3_metrics["config"]["early_stop"]
    a3_T = a3_metrics["val_metrics"]["temperature"]

    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=NUM_OUTCOME_CLASSES,
        max_depth=best_md, learning_rate=best_lr,
        n_estimators=n_est, min_child_weight=3,
        tree_method="hist", enable_categorical=True,
        random_state=seed, early_stopping_rounds=early_stop,
        eval_metric="mlogloss", n_jobs=-1, verbosity=0,
    )
    clf.fit(
        feats["X_train"], feats["y_train"],
        sample_weight=sample_weight,
        eval_set=[(feats["X_val"], feats["y_val"])],
        verbose=False,
    )
    test_probs_pre = clf.predict_proba(feats["X_test"])
    eps = 1e-12
    test_logits = np.log(np.clip(test_probs_pre, eps, 1.0))
    test_probs_post = softmax(test_logits / a3_T)

    # Augment test_df with keys.
    test_df_aug = test_df.copy()
    test_df_aug["game_pk"] = keys_df["game_pk"].astype(np.int64).to_numpy()
    test_df_aug["at_bat_number"] = keys_df["at_bat_number"].astype(np.int64).to_numpy()
    test_df_aug["pitch_number"] = keys_df["pitch_number"].astype(np.int64).to_numpy()

    return test_probs_post, feats["y_test"], test_df_aug


def _write_report(path: Path, m: dict) -> None:
    cfg = m["config"]
    coh = m["cohort"]
    val = m["val_metrics"]
    tst = m["test_metrics"]
    g = m["gates"]
    bb = m["backbone_byte_identity"]
    vs = m["vs_a3"]
    tr = m["training"]
    dl = vs["delta_lift"]
    dll = vs["delta_log_loss"]

    lift_post = 100 * tst["lift_post_temp"]
    lift_lo = 100 * tst["lift_ci_lo"]
    lift_hi = 100 * tst["lift_ci_hi"]

    lines = []
    lines.append("# A1 — Frozen v2 Backbone + Concat MLP Head")
    lines.append("")
    lines.append(f"**Verdict (A1 standalone):** **{g['verdict_a1']}**")
    lines.append(f"**Ship verdict (A1 vs A3 paired):** **{g['verdict_ship']}**")
    lines.append("")

    lines.append("## TL;DR")
    lines.append("")
    lines.append(f"- 7-class log-loss (post-T): **{tst['log_loss_post_temp']:.4f}**  "
                 f"(freq prior: {tst['freq_prior_log_loss']:.4f})")
    lines.append(f"- Lift vs prior: **{lift_post:+.2f}%** "
                 f"(95% CI [{lift_lo:+.2f}%, {lift_hi:+.2f}%])")
    lines.append(f"- 10-bin ECE post-T: **{tst['ece_post_temp']:.4f}**")
    lines.append(f"- Top-1 accuracy: {tst['top1_accuracy']:.4f}")
    lines.append(f"- Temperature: {val['temperature']:.4f}")
    lines.append(f"- Best epoch: {tr['best_epoch']}  Wall-clock: "
                 f"{tr['wall_clock_seconds']:.1f}s")
    lines.append("")
    lines.append("### A1 vs A3 paired-bootstrap delta")
    lines.append("")
    lines.append(f"- Paired rows: **{vs['n_paired_rows']:,}** "
                 f"(A1={vs['n_a1_rows']:,}, A3={vs['n_a3_rows']:,})")
    if not np.isnan(dl["delta_point"]):
        lines.append(f"- Lift delta (A1 - A3): **{100 * dl['delta_point']:+.4f}%** "
                     f"95% CI [{100 * dl['delta_ci_lo']:+.4f}%, "
                     f"{100 * dl['delta_ci_hi']:+.4f}%]")
        lines.append(f"- Log-loss delta (A1 - A3): **{dll['delta_point']:+.4f}** "
                     f"95% CI [{dll['delta_ci_lo']:+.4f}, "
                     f"{dll['delta_ci_hi']:+.4f}]  "
                     f"(negative = A1 wins)")
        lines.append(f"- A1 paired lift: {100*dl['lift_a1_point']:+.2f}%, "
                     f"A3 paired lift: {100*dl['lift_a3_point']:+.2f}%")
    else:
        lines.append("- Paired bootstrap unavailable (insufficient matched rows).")
    lines.append("")

    lines.append("## Architecture")
    lines.append("")
    lines.append("- **Backbone:** `models/pitchgpt_v2.pt` — FROZEN by construction "
                 "(byte-identity verified pre/post).")
    lines.append("- **Head input:** concat(hidden[128] + context[35] + "
                 "pitch_type_oh[17] + zone_oh[26] + velo_oh[5]) = 211d")
    lines.append("- **Head:** MLP `211 -> 128 -> 64 -> 7` (ReLU + dropout 0.1).")
    lines.append("- **Loss:** weighted CE, inverse-frequency class weights cap 10.")
    lines.append("")

    lines.append("## Cohort")
    lines.append("")
    lines.append(f"- Train sequences: {coh['train_sequences']:,}  "
                 f"valid outcomes: {coh['train_valid_outcomes']:,}")
    lines.append(f"- Val sequences: {coh['val_sequences']:,}  "
                 f"valid outcomes: {coh['val_valid_outcomes']:,}")
    lines.append(f"- Test sequences: {coh['test_sequences']:,}  "
                 f"valid outcomes: {coh['test_valid_outcomes']:,}")
    lines.append(f"- Test unique pitchers: {coh['n_unique_test_pitchers']}")
    lines.append("")

    lines.append("## Training history")
    lines.append("")
    lines.append("| epoch | train loss (wtd) | val log-loss | seconds |")
    lines.append("|-------|------------------|-------------:|--------:|")
    for h in tr["history"]:
        marker = "  **<-best**" if h["epoch"] == tr["best_epoch"] else ""
        lines.append(f"| {h['epoch']} | {h['train_loss_weighted']:.4f} | "
                     f"{h['val_log_loss']:.4f} | {h['seconds']:.1f} |{marker}")
    lines.append("")

    lines.append("## Per-class log-loss (test, post-temp)")
    lines.append("")
    lines.append("| class | log-loss |")
    lines.append("|-------|---------:|")
    for c in OUTCOME_CLASSES:
        ll = tst["per_class_log_loss"].get(c)
        ll_str = (f"{ll:.4f}" if ll is not None
                  and not (isinstance(ll, float) and np.isnan(ll)) else "NA")
        lines.append(f"| {c} | {ll_str} |")
    lines.append("")

    lines.append("## Val metrics")
    lines.append("")
    lines.append(f"- Log-loss pre/post: {val['log_loss_pre_temp']:.4f} / "
                 f"{val['log_loss_post_temp']:.4f}")
    lines.append(f"- ECE pre/post: {val['ece_pre_temp']:.4f} / "
                 f"{val['ece_post_temp']:.4f}")
    lines.append(f"- Freq prior log-loss: {val['freq_prior_log_loss']:.4f}")
    lines.append(f"- Lift post-T: {100*val['lift_post_temp']:+.2f}%")
    lines.append("")

    lines.append("## Per-pitcher stability (top-50 most-frequent test pitchers)")
    lines.append("")
    pp = tst["per_pitcher_summary"]
    lines.append(f"- n pitchers (n>=30 rows each): {pp['n_pitchers']}")
    lines.append(f"- Log-loss mean / var / range: {pp['mean']:.4f} / "
                 f"{pp['var']:.4f} / [{pp['min']:.4f}, {pp['max']:.4f}]")
    lines.append("")

    lines.append("## Backbone byte-identity verification")
    lines.append("")
    lines.append(f"- v2.pt SHA256 pre:  `{bb['v2_pt_sha256_pre']}`")
    lines.append(f"- v2.pt SHA256 post: `{bb['v2_pt_sha256_post']}`")
    lines.append(f"- Backbone param-SHA pre:  `{bb['backbone_param_sha256_pre']}`")
    lines.append(f"- Backbone param-SHA post: `{bb['backbone_param_sha256_post']}`")
    lines.append(f"- **Byte-identity verified:** "
                 f"{'YES' if bb['identity_verified'] else 'NO'}")
    lines.append(f"- Phase 0.3 ckpt present (untouched): "
                 f"{'YES' if bb['phase03_ckpt_present'] else 'NO'}")
    if bb.get("phase03_ckpt_sha256"):
        lines.append(f"- Phase 0.3 ckpt SHA256: `{bb['phase03_ckpt_sha256']}`")
    lines.append("")

    lines.append("## Final ship recommendation")
    lines.append("")
    lines.append(f"**{g['verdict_ship']}**")
    if g["verdict_ship"] == "SHIP A3":
        lines.append("")
        lines.append("Per RESEARCH_PLAN §7 kill criterion: A1 did not beat A3 by "
                     "+1pp lift on 2025 holdout (paired CI). "
                     "**The PG v2 backbone provides no marginal value for the "
                     "7-class outcome target beyond engineered features.** "
                     "Ship A3 (XGBoost) per the protocol-locked outcome.")
    elif g["verdict_ship"] == "SHIP A1":
        lines.append("")
        lines.append("A1 beat A3 by >= +1pp lift on 2025 holdout with paired CI "
                     "lo > 0. The PG v2 backbone adds marginal value beyond "
                     "engineered features. Ship A1.")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
