"""
PitchLSTMBaseline -- LSTM baseline for the PitchGPT transformer.

This module exists solely as a *baseline* for Ticket #3 of the PitchGPT
validation spec.  The deliverable is a model with the same inputs,
vocabulary, loss, and training interface as ``PitchGPT`` but with a
plain 2-layer LSTM in place of the causal-masked transformer.  The
eventual baseline-comparison agent will train both on identical splits
and report perplexity head-to-head.

We deliberately do NOT include attention, positional encoding, or any
"transformer-y" inductive bias — the whole point is to prove that the
transformer beats a pure sequential model.

Public API
----------
- ``PitchLSTMNetwork``   -- bare ``nn.Module``
- ``PitchLSTMBaseline``  -- lifecycle wrapper (same shape as ``PitchGPT``)
- ``train_pitch_lstm``   -- end-to-end training loop
- ``calculate_perplexity_lstm`` -- perplexity on a held-out split
- ``score_sequences_lstm``      -- per-pitch NLL (matches ``_score_sequences``)
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.analytics.base import BaseAnalyticsModel
from src.analytics.pitchgpt import (
    CONTEXT_DIM,
    PAD_TOKEN,
    TOTAL_VOCAB,
    VOCAB_SIZE,
    PitchSequenceDataset,
    PitchTokenizer,
    _collate_fn,
    _compute_per_pitch_score_diff,
    _get_device,
    _safe_bool,
    _safe_int,
    _safe_str,
)

logger = logging.getLogger(__name__)


# ── Paths ────────────────────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

# ── Training defaults (kept symmetric with PitchGPT) ─────────────────────────
DEFAULT_DIM = 128
DEFAULT_LAYERS = 2
DEFAULT_MAX_SEQ = 256
DEFAULT_EPOCHS = 5
DEFAULT_BATCH = 32
DEFAULT_LR = 1e-3


# ═════════════════════════════════════════════════════════════════════════════
# Network
# ═════════════════════════════════════════════════════════════════════════════

class PitchLSTMNetwork(nn.Module):
    """A deliberately-simple 2-layer LSTM baseline for next-pitch
    prediction.

    Architecture
    ------------
    - ``token_embedding`` (TOTAL_VOCAB → d_model)  -- PAD-aware.
    - ``context_proj``    (CONTEXT_DIM → d_model)  -- same 34-dim one-hot
      as the transformer.
    - Fused input = ``tok_emb + ctx_emb`` (additive, like PitchGPT).
    - 2-layer LSTM, hidden size = d_model.
    - LayerNorm on the LSTM output.
    - Linear head → VOCAB_SIZE (not TOTAL_VOCAB — we never predict
      PAD / BOS, only real pitch tokens, matching PitchGPT).

    No attention, no positional encoding: the LSTM's recurrence is the
    only source of sequential information.  Naturally causal.
    """

    def __init__(
        self,
        vocab_size: int = TOTAL_VOCAB,
        context_dim: int = CONTEXT_DIM,
        d_model: int = DEFAULT_DIM,
        num_layers: int = DEFAULT_LAYERS,
        dropout: float = 0.1,
        output_vocab: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        # Default output vocab = VOCAB_SIZE (no PAD/BOS in output logits).
        self.output_vocab = output_vocab if output_vocab is not None else VOCAB_SIZE

        # PAD index is TOTAL_VOCAB - 2 normally, but for unit tests the
        # caller may pass a smaller vocab.  Use ``padding_idx`` only when
        # PAD_TOKEN fits inside the supplied ``vocab_size``.
        pad_idx = PAD_TOKEN if PAD_TOKEN < vocab_size else None

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.context_proj = nn.Linear(context_dim, d_model)

        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, self.output_vocab)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tokens: torch.Tensor,          # (B, S)
        context: torch.Tensor,         # (B, S, CONTEXT_DIM)
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Return logits of shape (B, S, output_vocab)."""
        tok_emb = self.token_embedding(tokens)      # (B, S, D)
        ctx_emb = self.context_proj(context)        # (B, S, D)
        x = tok_emb + ctx_emb                        # (B, S, D)

        out, _ = self.lstm(x, hidden)                # (B, S, D)
        out = self.layer_norm(out)
        logits = self.output_head(out)               # (B, S, output_vocab)
        return logits


# ═════════════════════════════════════════════════════════════════════════════
# BaseAnalyticsModel wrapper
# ═════════════════════════════════════════════════════════════════════════════

class PitchLSTMBaseline(BaseAnalyticsModel):
    """Lifecycle wrapper around :class:`PitchLSTMNetwork`.

    Deliberately mirrors :class:`src.analytics.pitchgpt.PitchGPT` so the
    eventual baseline-comparison harness can swap one for the other
    without branching.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model: PitchLSTMNetwork | None = None
        self._version = "1.0.0"

    @property
    def model_name(self) -> str:
        return "PitchLSTMBaseline"

    @property
    def version(self) -> str:
        return self._version

    def train(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        return train_pitch_lstm(conn, **kwargs)

    def predict(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        raise NotImplementedError(
            "PitchLSTMBaseline is a baseline for perplexity comparison; "
            "use calculate_perplexity_lstm() instead."
        )

    def evaluate(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        seasons = kwargs.get("seasons", [2024])
        version = kwargs.get("version", "1")
        return calculate_perplexity_lstm(conn, seasons=seasons, version=version)

    # ── Convenience for the comparison harness ──────────────────────────
    def score_sequences(
        self,
        conn: duckdb.DuckDBPyConnection,
        pitcher_id: int,
        season: int | None = None,
        version: str = "1",
    ) -> list[dict]:
        """Per-pitch NLLs for one pitcher (matches PitchGPT ``_score_sequences``)."""
        model = _load_lstm_model(version)
        return score_sequences_lstm(model, conn, pitcher_id, season=season)

    def calculate_perplexity(
        self,
        conn: duckdb.DuckDBPyConnection,
        seasons: list[int] | None = None,
        version: str = "1",
    ) -> dict:
        """Perplexity on a held-out split (matches the spec's head-to-head API)."""
        return calculate_perplexity_lstm(conn, seasons=seasons, version=version)


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

def train_pitch_lstm(
    conn: duckdb.DuckDBPyConnection,
    seasons: range | list[int] | None = None,
    val_seasons: list[int] | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    d_model: int = DEFAULT_DIM,
    num_layers: int = DEFAULT_LAYERS,
    max_seq_len: int = DEFAULT_MAX_SEQ,
    version: str = "1",
    max_games: int = 3000,
) -> dict:
    """Train the LSTM baseline.  Mirrors :func:`train_pitchgpt` one-for-one.

    Reuses :class:`PitchSequenceDataset` so the data pipeline (including
    the score_diff fix) is identical to PitchGPT's.
    """
    if seasons is None:
        seasons = list(range(2015, 2025))
    if val_seasons is None:
        val_seasons = [2025, 2026]

    device = _get_device()

    logger.info("Loading training data (seasons %s, max_games=%d)...", list(seasons), max_games)
    train_ds = PitchSequenceDataset(conn, seasons=list(seasons), max_seq_len=max_seq_len, max_games=max_games)
    if len(train_ds) == 0:
        logger.warning("No training sequences found.")
        return {"status": "no_data", "train_loss": float("nan")}

    val_ds = PitchSequenceDataset(conn, seasons=val_seasons, max_seq_len=max_seq_len, max_games=500)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn,
    ) if len(val_ds) > 0 else None

    model = PitchLSTMNetwork(
        d_model=d_model,
        num_layers=num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    best_val_loss = float("inf")
    metrics_history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0

        for tokens, ctx, target in train_loader:
            tokens, ctx, target = (
                tokens.to(device), ctx.to(device), target.to(device),
            )
            logits = model(tokens, ctx)  # (B, S, output_vocab)
            # Target tokens in [0, VOCAB_SIZE) — output head predicts VOCAB_SIZE
            # classes; PAD targets are ignored via ``ignore_index``.  However,
            # PAD_TOKEN == VOCAB_SIZE which is *out of bounds* for the output
            # head (output_vocab = VOCAB_SIZE, valid class indices 0..V-1).
            # We therefore clamp PAD positions in the *target* to 0 before
            # cross_entropy — ``ignore_index=PAD_TOKEN`` is still honored
            # against the original target, so we must keep PAD_TOKEN in the
            # target tensor, not clamp it.  Instead, CrossEntropyLoss with
            # ignore_index does the right thing when the target equals the
            # ignore_index even if it's out of class range — but PyTorch
            # checks class range for non-ignored targets.  Safer: mask PAD
            # positions out of the loss entirely by replacing them with
            # ignore_index (already PAD_TOKEN) and computing CE over the
            # valid targets only.  The default behavior of CrossEntropyLoss
            # with ignore_index == PAD_TOKEN is correct as long as no real
            # target == VOCAB_SIZE — which is guaranteed by the tokenizer.
            loss = criterion(
                logits.reshape(-1, model.output_vocab),
                target.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_tok = (target != PAD_TOKEN).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok

        train_loss = total_loss / max(total_tokens, 1)

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            v_loss, v_tok = 0.0, 0
            with torch.no_grad():
                for tokens, ctx, target in val_loader:
                    tokens, ctx, target = (
                        tokens.to(device), ctx.to(device), target.to(device),
                    )
                    logits = model(tokens, ctx)
                    loss = criterion(
                        logits.reshape(-1, model.output_vocab),
                        target.reshape(-1),
                    )
                    n_tok = (target != PAD_TOKEN).sum().item()
                    v_loss += loss.item() * n_tok
                    v_tok += n_tok
            val_loss = v_loss / max(v_tok, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        perplexity = math.exp(min(train_loss, 20))
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4) if not math.isnan(val_loss) else None,
            "perplexity": round(perplexity, 2),
        }
        metrics_history.append(epoch_metrics)
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%s  ppl=%.2f",
            epoch, epochs, train_loss,
            f"{val_loss:.4f}" if not math.isnan(val_loss) else "N/A",
            perplexity,
        )

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = _MODEL_DIR / f"pitch_lstm_v{version}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "d_model": d_model,
                "num_layers": num_layers,
                "max_seq_len": max_seq_len,
                "vocab_size": TOTAL_VOCAB,
                "context_dim": CONTEXT_DIM,
                "output_vocab": VOCAB_SIZE,
            },
            "version": version,
        },
        save_path,
    )
    logger.info("LSTM baseline saved to %s", save_path)

    return {
        "status": "trained",
        "epochs": epochs,
        "final_train_loss": metrics_history[-1]["train_loss"],
        "final_val_loss": metrics_history[-1].get("val_loss"),
        "best_val_loss": round(best_val_loss, 4) if best_val_loss < float("inf") else None,
        "final_perplexity": metrics_history[-1]["perplexity"],
        "n_train_sequences": len(train_ds),
        "n_val_sequences": len(val_ds),
        "save_path": str(save_path),
        "history": metrics_history,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Inference
# ═════════════════════════════════════════════════════════════════════════════

def _load_lstm_model(version: str = "1") -> PitchLSTMNetwork:
    device = _get_device()
    path = _MODEL_DIR / f"pitch_lstm_v{version}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained PitchLSTM model at {path}. Run train_pitch_lstm() first."
        )
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    model = PitchLSTMNetwork(
        vocab_size=cfg.get("vocab_size", TOTAL_VOCAB),
        context_dim=cfg.get("context_dim", CONTEXT_DIM),
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        output_vocab=cfg.get("output_vocab", VOCAB_SIZE),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def score_sequences_lstm(
    model: PitchLSTMNetwork,
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: int | None = None,
) -> list[dict]:
    """Per-pitch NLL for a single pitcher, mirroring PitchGPT ``_score_sequences``."""
    # Build the same query the transformer inference path uses so the
    # downstream numbers are directly comparable.
    params: list = [pitcher_id]
    season_filter = ""
    if season is not None:
        season_filter = "AND EXTRACT(YEAR FROM game_date) = $2"
        params.append(season)

    df = conn.execute(f"""
        SELECT
            game_pk, pitcher_id, pitch_type, plate_x, plate_z,
            release_speed, balls, strikes, outs_when_up,
            on_1b, on_2b, on_3b, stand, inning,
            inning_topbot, events, delta_run_exp,
            at_bat_number, pitch_number
        FROM pitches
        WHERE pitcher_id = $1 AND pitch_type IS NOT NULL
          {season_filter}
        ORDER BY game_pk, at_bat_number, pitch_number
    """, params).fetchdf()

    if df.empty:
        return []

    df = df.assign(_score_diff=_compute_per_pitch_score_diff(df))
    device = next(model.parameters()).device
    results: list[dict] = []

    for game_pk, game_df in df.groupby("game_pk", sort=False):
        if len(game_df) < 2:
            continue

        tokens_list: list[int] = []
        contexts_list: list[torch.Tensor] = []
        pitch_types: list[str] = []

        for _, row in game_df.iterrows():
            tok = PitchTokenizer.encode(
                row.get("pitch_type"),
                row.get("plate_x"),
                row.get("plate_z"),
                row.get("release_speed"),
            )
            tokens_list.append(tok)
            pitch_types.append(str(row.get("pitch_type", "UN")))

            ctx = PitchTokenizer.encode_context(
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
            contexts_list.append(PitchTokenizer.context_to_tensor(ctx))

        max_len = DEFAULT_MAX_SEQ
        tokens_list = tokens_list[:max_len]
        contexts_list = contexts_list[:max_len]
        pitch_types = pitch_types[:max_len]

        if len(tokens_list) < 2:
            continue

        input_tokens = torch.tensor(tokens_list[:-1], dtype=torch.long).unsqueeze(0).to(device)
        target_tokens = torch.tensor(tokens_list[1:], dtype=torch.long).to(device)
        context_tensor = torch.stack(contexts_list[:-1]).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tokens, context_tensor).squeeze(0)  # (S, V)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            per_pitch_nll = []
            for i in range(len(target_tokens)):
                t = target_tokens[i].item()
                if 0 <= t < model.output_vocab:
                    nll = -log_probs[i, t].item()
                else:
                    nll = 0.0
                per_pitch_nll.append(round(nll, 4))

        results.append({
            "game_pk": int(game_pk),
            "per_pitch_nll": per_pitch_nll,
            "mean_nll": round(float(np.mean(per_pitch_nll)), 4) if per_pitch_nll else 0.0,
            "pitch_types": pitch_types[1:],
        })

    return results


def calculate_perplexity_lstm(
    conn: duckdb.DuckDBPyConnection,
    seasons: list[int] | None = None,
    version: str = "1",
    max_games: int = 500,
) -> dict:
    """Compute held-out perplexity on a season set.  Used by the
    baseline-comparison harness to produce a number directly
    comparable to PitchGPT's perplexity.
    """
    if seasons is None:
        seasons = [2024]

    model = _load_lstm_model(version)
    device = next(model.parameters()).device

    ds = PitchSequenceDataset(
        conn, seasons=seasons, max_seq_len=DEFAULT_MAX_SEQ, max_games=max_games,
    )
    if len(ds) == 0:
        return {"status": "no_data", "perplexity": float("inf")}

    loader = DataLoader(ds, batch_size=DEFAULT_BATCH, shuffle=False, collate_fn=_collate_fn)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    total_loss, total_tok = 0.0, 0
    model.eval()
    with torch.no_grad():
        for tokens, ctx, target in loader:
            tokens, ctx, target = tokens.to(device), ctx.to(device), target.to(device)
            logits = model(tokens, ctx)
            loss = criterion(
                logits.reshape(-1, model.output_vocab),
                target.reshape(-1),
            )
            n_tok = (target != PAD_TOKEN).sum().item()
            total_loss += loss.item() * n_tok
            total_tok += n_tok

    mean_nll = total_loss / max(total_tok, 1)
    return {
        "status": "evaluated",
        "mean_nll": round(mean_nll, 4),
        "perplexity": round(math.exp(min(mean_nll, 20)), 2),
        "n_tokens": total_tok,
        "n_sequences": len(ds),
        "seasons": list(seasons),
    }
