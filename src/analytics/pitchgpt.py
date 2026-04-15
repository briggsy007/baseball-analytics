"""
PitchGPT -- Transformer-based pitch sequence model.

Treats each game as a "pitch sentence" and learns to predict the next
pitch token given the sequence so far plus situational context (count,
outs, runners, batter hand, inning, score differential).

Public API
----------
- ``PitchTokenizer``          -- encode/decode composite pitch tokens
- ``PitchSequenceDataset``    -- PyTorch Dataset of game pitch sequences
- ``PitchGPTModel``           -- small decoder-only transformer
- ``train_pitchgpt``          -- end-to-end training loop
- ``calculate_predictability`` -- Pitch Predictability Score (perplexity)
- ``calculate_disruption_index`` -- per-pitch surprise for a game
- ``batch_calculate``         -- PPS leaderboard for a season

Inherits from ``BaseAnalyticsModel`` for lifecycle / serialisation hooks.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.analytics.base import BaseAnalyticsModel

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    """Return the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _safe_int(val, default: int = 0) -> int:
    """Convert a possibly-NA/None value to int safely."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_bool(val) -> bool:
    """Convert a possibly-NA/None value to bool safely."""
    if val is None:
        return False
    try:
        return bool(int(val))
    except (TypeError, ValueError):
        return False


def _safe_str(val, default: str = "R") -> str:
    """Convert a possibly-NA/None value to str safely."""
    if val is None:
        return default
    s = str(val)
    if s in ("", "nan", "None", "<NA>"):
        return default
    return s


# ── Paths ────────────────────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

# ── Constants ────────────────────────────────────────────────────────────────
# Pitch type vocabulary (indices 0-15 for known types, 16 = unknown)
PITCH_TYPE_MAP: dict[str, int] = {
    "FF": 0, "SI": 1, "FC": 2, "SL": 3, "CU": 4, "CH": 5,
    "FS": 6, "KC": 7, "KN": 8, "EP": 9, "CS": 10, "SV": 11,
    "ST": 12, "SC": 13, "FO": 14, "FA": 15,
}
NUM_PITCH_TYPES = 17  # 16 known + 1 unknown
NUM_ZONES = 26        # 5x5 grid (0-24) + 1 for out-of-zone
NUM_VELO_BUCKETS = 5  # <80, 80-85, 85-90, 90-95, 95+
VOCAB_SIZE = NUM_PITCH_TYPES * NUM_ZONES * NUM_VELO_BUCKETS  # 17*26*5 = 2210

# Context dimensions
NUM_COUNT_STATES = 12   # 0-0 .. 3-2
NUM_OUTS = 3            # 0, 1, 2
NUM_RUNNER_STATES = 8   # 2^3 base combinations
NUM_BATTER_HANDS = 2    # L, R
NUM_INNING_BUCKETS = 4  # early(1-3), mid(4-6), late(7-9), extra(10+)
NUM_SCORE_DIFF_BUCKETS = 5  # big deficit, small deficit, tie, small lead, big lead
CONTEXT_DIM = (
    NUM_COUNT_STATES + NUM_OUTS + NUM_RUNNER_STATES
    + NUM_BATTER_HANDS + NUM_INNING_BUCKETS + NUM_SCORE_DIFF_BUCKETS
)  # 34

# Special tokens
PAD_TOKEN = VOCAB_SIZE      # padding
BOS_TOKEN = VOCAB_SIZE + 1  # beginning of sequence
TOTAL_VOCAB = VOCAB_SIZE + 2

# Training defaults
DEFAULT_DIM = 128
DEFAULT_HEADS = 4
DEFAULT_LAYERS = 4
DEFAULT_MAX_SEQ = 256
DEFAULT_EPOCHS = 5
DEFAULT_BATCH = 32
DEFAULT_LR = 1e-3
MIN_PITCHES_LEADERBOARD = 200


# ═════════════════════════════════════════════════════════════════════════════
# Tokenizer
# ═════════════════════════════════════════════════════════════════════════════

class PitchTokenizer:
    """Encode / decode a pitch into a single composite integer token.

    token = pitch_type_idx * (NUM_ZONES * NUM_VELO_BUCKETS)
            + zone_idx * NUM_VELO_BUCKETS
            + velo_bucket
    """

    # ── Encoding helpers ─────────────────────────────────────────────────

    @staticmethod
    def pitch_type_to_idx(pitch_type: str | None) -> int:
        if pitch_type is None:
            return NUM_PITCH_TYPES - 1  # unknown
        return PITCH_TYPE_MAP.get(pitch_type, NUM_PITCH_TYPES - 1)

    @staticmethod
    def idx_to_pitch_type(idx: int) -> str:
        inv = {v: k for k, v in PITCH_TYPE_MAP.items()}
        return inv.get(idx, "UN")

    @staticmethod
    def location_to_zone(plate_x: float | None, plate_z: float | None) -> int:
        """Discretise plate location into a 5x5 grid (0-24) or 25 for missing."""
        if plate_x is None or plate_z is None or np.isnan(plate_x) or np.isnan(plate_z):
            return NUM_ZONES - 1  # out-of-zone / missing

        # Plate is roughly -1.5 to 1.5 feet wide, 1.0 to 4.0 feet tall
        x_edges = np.linspace(-1.5, 1.5, 6)
        z_edges = np.linspace(1.0, 4.0, 6)

        x_bin = int(np.clip(np.digitize(plate_x, x_edges) - 1, 0, 4))
        z_bin = int(np.clip(np.digitize(plate_z, z_edges) - 1, 0, 4))
        return z_bin * 5 + x_bin

    @staticmethod
    def velocity_to_bucket(release_speed: float | None) -> int:
        """Map velocity to one of 5 buckets."""
        if release_speed is None or np.isnan(release_speed):
            return 2  # default to middle bucket
        if release_speed < 80:
            return 0
        elif release_speed < 85:
            return 1
        elif release_speed < 90:
            return 2
        elif release_speed < 95:
            return 3
        else:
            return 4

    # ── Main encode / decode ─────────────────────────────────────────────

    @classmethod
    def encode(
        cls,
        pitch_type: str | None,
        plate_x: float | None,
        plate_z: float | None,
        release_speed: float | None,
    ) -> int:
        pt_idx = cls.pitch_type_to_idx(pitch_type)
        zone = cls.location_to_zone(plate_x, plate_z)
        velo = cls.velocity_to_bucket(release_speed)
        return pt_idx * (NUM_ZONES * NUM_VELO_BUCKETS) + zone * NUM_VELO_BUCKETS + velo

    @classmethod
    def decode(cls, token: int) -> dict[str, Any]:
        if token < 0 or token >= VOCAB_SIZE:
            return {"pitch_type": "UN", "zone": -1, "velo_bucket": -1}
        velo = token % NUM_VELO_BUCKETS
        remaining = token // NUM_VELO_BUCKETS
        zone = remaining % NUM_ZONES
        pt_idx = remaining // NUM_ZONES
        return {
            "pitch_type": cls.idx_to_pitch_type(pt_idx),
            "zone": zone,
            "velo_bucket": velo,
        }

    @classmethod
    def encode_context(
        cls,
        balls: int,
        strikes: int,
        outs: int,
        on_1b: bool,
        on_2b: bool,
        on_3b: bool,
        stand: str,
        inning: int,
        score_diff: int,
    ) -> list[int]:
        """Encode situational context as a list of categorical indices.

        Returns a list of 6 integers (one per context feature).
        """
        count_state = min(balls, 3) * 3 + min(strikes, 2)
        outs_idx = min(outs, 2)
        runner_state = int(bool(on_1b)) * 4 + int(bool(on_2b)) * 2 + int(bool(on_3b))
        batter_hand = 0 if stand == "L" else 1

        if inning <= 3:
            inning_bucket = 0
        elif inning <= 6:
            inning_bucket = 1
        elif inning <= 9:
            inning_bucket = 2
        else:
            inning_bucket = 3

        if score_diff <= -4:
            score_bucket = 0
        elif score_diff < 0:
            score_bucket = 1
        elif score_diff == 0:
            score_bucket = 2
        elif score_diff <= 3:
            score_bucket = 3
        else:
            score_bucket = 4

        return [count_state, outs_idx, runner_state, batter_hand,
                inning_bucket, score_bucket]

    @classmethod
    def context_to_tensor(cls, context_list: list[int]) -> torch.Tensor:
        """One-hot encode a context list into a float tensor of size CONTEXT_DIM."""
        vec = torch.zeros(CONTEXT_DIM, dtype=torch.float32)
        offsets = [0, NUM_COUNT_STATES, NUM_COUNT_STATES + NUM_OUTS,
                   NUM_COUNT_STATES + NUM_OUTS + NUM_RUNNER_STATES,
                   NUM_COUNT_STATES + NUM_OUTS + NUM_RUNNER_STATES + NUM_BATTER_HANDS,
                   NUM_COUNT_STATES + NUM_OUTS + NUM_RUNNER_STATES + NUM_BATTER_HANDS + NUM_INNING_BUCKETS]
        for i, val in enumerate(context_list):
            vec[offsets[i] + val] = 1.0
        return vec


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

class PitchSequenceDataset(Dataset):
    """Loads game pitch sequences from the database, tokenises them, and
    returns ``(token_seq, context_seq, target_seq)`` where target is
    token_seq shifted by 1.

    Each *game* by a single pitcher is one sequence (all at-bats
    concatenated in pitch order).
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        seasons: list[int] | range | None = None,
        max_seq_len: int = DEFAULT_MAX_SEQ,
        max_games: int | None = None,
    ) -> None:
        self.max_seq_len = max_seq_len
        self.sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._load(conn, seasons, max_games=max_games)

    # ── Private ──────────────────────────────────────────────────────────

    def _load(self, conn: duckdb.DuckDBPyConnection, seasons, max_games: int | None = None) -> None:
        season_filter = ""
        if seasons is not None:
            season_list = list(seasons)
            if season_list:
                s_str = ", ".join(str(int(s)) for s in season_list)
                season_filter = f"AND EXTRACT(YEAR FROM game_date) IN ({s_str})"

        # When max_games is set, restrict to a random sample of game_pks
        # to avoid loading the entire 7M+ row table into memory.
        game_filter = ""
        if max_games is not None:
            game_filter = f"""
                AND game_pk IN (
                    SELECT game_pk FROM (
                        SELECT DISTINCT game_pk
                        FROM pitches
                        WHERE pitch_type IS NOT NULL {season_filter}
                    ) USING SAMPLE {int(max_games)} ROWS
                )
            """

        query = f"""
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
                inning
            FROM pitches
            WHERE pitch_type IS NOT NULL
              {season_filter}
              {game_filter}
            ORDER BY game_pk, pitcher_id, at_bat_number, pitch_number
        """
        df = conn.execute(query).fetchdf()

        if df.empty:
            logger.warning("No pitch data found for PitchGPT dataset.")
            return

        # Group by (game, pitcher) to form sequences
        grouped = df.groupby(["game_pk", "pitcher_id"])
        for _key, game_df in grouped:
            if len(game_df) < 2:
                continue  # need at least 2 pitches

            tokens: list[int] = []
            contexts: list[torch.Tensor] = []

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
                    score_diff=0,  # not directly available; use 0
                )
                contexts.append(PitchTokenizer.context_to_tensor(ctx_list))

            # Truncate to max_seq_len
            tokens = tokens[: self.max_seq_len]
            contexts = contexts[: self.max_seq_len]

            # Input = tokens[:-1], target = tokens[1:]
            input_tokens = torch.tensor(tokens[:-1], dtype=torch.long)
            target_tokens = torch.tensor(tokens[1:], dtype=torch.long)
            context_tensor = torch.stack(contexts[:-1])  # aligned with input

            self.sequences.append((input_tokens, context_tensor, target_tokens))

        logger.info(
            "PitchGPT dataset: %d sequences from %d pitches.",
            len(self.sequences),
            len(df),
        )

    # ── Dataset interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx]


def _collate_fn(batch):
    """Pad variable-length sequences to the longest in the batch."""
    max_len = max(item[0].size(0) for item in batch)
    tokens_batch, ctx_batch, target_batch = [], [], []

    for tokens, ctx, target in batch:
        seq_len = tokens.size(0)
        pad_len = max_len - seq_len

        tokens_padded = torch.cat(
            [tokens, torch.full((pad_len,), PAD_TOKEN, dtype=torch.long)]
        )
        target_padded = torch.cat(
            [target, torch.full((pad_len,), PAD_TOKEN, dtype=torch.long)]
        )
        if pad_len > 0:
            ctx_padded = torch.cat(
                [ctx, torch.zeros(pad_len, CONTEXT_DIM)]
            )
        else:
            ctx_padded = ctx

        tokens_batch.append(tokens_padded)
        ctx_batch.append(ctx_padded)
        target_batch.append(target_padded)

    return (
        torch.stack(tokens_batch),
        torch.stack(ctx_batch),
        torch.stack(target_batch),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Model
# ═════════════════════════════════════════════════════════════════════════════

class PitchGPTModel(nn.Module):
    """Small decoder-only transformer for next-pitch prediction.

    Architecture:
        token_embedding(TOTAL_VOCAB, dim) + context_projection(CONTEXT_DIM, dim)
        + positional_encoding
        → N TransformerDecoderLayers (self-attn only, causal mask)
        → linear head → vocab_size logits
    """

    def __init__(
        self,
        vocab_size: int = TOTAL_VOCAB,
        d_model: int = DEFAULT_DIM,
        nhead: int = DEFAULT_HEADS,
        num_layers: int = DEFAULT_LAYERS,
        max_seq_len: int = DEFAULT_MAX_SEQ,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.context_proj = nn.Linear(CONTEXT_DIM, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer decoder layers (self-attention only)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # Output projection
        self.output_head = nn.Linear(d_model, VOCAB_SIZE)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tokens: torch.Tensor,       # (B, S)
        context: torch.Tensor,      # (B, S, CONTEXT_DIM)
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns logits of shape (B, S, VOCAB_SIZE).
        """
        B, S = tokens.shape
        device = tokens.device

        # Embeddings
        tok_emb = self.token_embedding(tokens)               # (B, S, D)
        ctx_emb = self.context_proj(context)                  # (B, S, D)
        pos_ids = torch.arange(S, device=device).unsqueeze(0) # (1, S)
        pos_ids = pos_ids.clamp(max=self.max_seq_len - 1)    # prevent OOB
        pos_emb = self.pos_embedding(pos_ids)                 # (1, S, D)

        x = tok_emb + ctx_emb + pos_emb  # (B, S, D)

        # Causal mask: True means *masked* for nn.TransformerEncoder
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(
                S, device=device
            )

        # Padding mask: True means *ignore*
        padding_mask = (tokens == PAD_TOKEN)  # (B, S)

        x = self.transformer(
            x,
            mask=mask,
            src_key_padding_mask=padding_mask,
        )

        logits = self.output_head(x)  # (B, S, VOCAB_SIZE)
        return logits


# ═════════════════════════════════════════════════════════════════════════════
# BaseAnalyticsModel wrapper
# ═════════════════════════════════════════════════════════════════════════════

class PitchGPT(BaseAnalyticsModel):
    """Lifecycle wrapper around :class:`PitchGPTModel`."""

    def __init__(self) -> None:
        super().__init__()
        self.model: PitchGPTModel | None = None
        self._version = "1.0.0"

    @property
    def model_name(self) -> str:
        return "PitchGPT"

    @property
    def version(self) -> str:
        return self._version

    def train(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        return train_pitchgpt(conn, **kwargs)

    def predict(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        pitcher_id = kwargs.get("pitcher_id")
        season = kwargs.get("season")
        if pitcher_id is None:
            raise ValueError("pitcher_id is required for predict().")
        return calculate_predictability(conn, pitcher_id, season)

    def evaluate(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        season = kwargs.get("season")
        df = batch_calculate(conn, season=season)
        if df.empty:
            return {"qualifying_pitchers": 0, "mean_pps": 0.0}
        return {
            "qualifying_pitchers": len(df),
            "mean_pps": round(float(df["pps"].mean()), 4),
            "std_pps": round(float(df["pps"].std()), 4),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

def train_pitchgpt(
    conn: duckdb.DuckDBPyConnection,
    seasons: range | list[int] | None = None,
    val_seasons: list[int] | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    d_model: int = DEFAULT_DIM,
    nhead: int = DEFAULT_HEADS,
    num_layers: int = DEFAULT_LAYERS,
    max_seq_len: int = DEFAULT_MAX_SEQ,
    version: str = "1",
    max_games: int = 3000,
) -> dict:
    """Train PitchGPT end-to-end.

    Args:
        conn: DuckDB connection.
        seasons: Training seasons (default 2015-2024).
        val_seasons: Validation seasons (default [2025, 2026]).
        epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        max_seq_len: Maximum sequence length.
        version: Model version tag.
        max_games: Cap on training games to sample (default 3000, yielding
                   ~750-900K pitch tokens).  Prevents loading the full 7M+
                   row table into memory.

    Returns:
        Dictionary of training metrics.
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

    model = PitchGPTModel(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
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
            logits = model(tokens, ctx)  # (B, S, V)
            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                target.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Count non-pad tokens
            n_tok = (target != PAD_TOKEN).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok

        train_loss = total_loss / max(total_tokens, 1)

        # Validation
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
                        logits.reshape(-1, VOCAB_SIZE),
                        target.reshape(-1),
                    )
                    n_tok = (target != PAD_TOKEN).sum().item()
                    v_loss += loss.item() * n_tok
                    v_tok += n_tok
            val_loss = v_loss / max(v_tok, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        perplexity = math.exp(min(train_loss, 20))  # cap to avoid overflow
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

    # Save model
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = _MODEL_DIR / f"pitchgpt_v{version}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "max_seq_len": max_seq_len,
                "vocab_size": TOTAL_VOCAB,
            },
            "version": version,
        },
        save_path,
    )
    logger.info("Model saved to %s", save_path)

    final_metrics = {
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
    return final_metrics


# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════

def _load_model(version: str = "1") -> PitchGPTModel:
    """Load a trained PitchGPT checkpoint."""
    device = _get_device()
    path = _MODEL_DIR / f"pitchgpt_v{version}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained PitchGPT model at {path}. Run train_pitchgpt() first."
        )
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    model = PitchGPTModel(
        vocab_size=cfg.get("vocab_size", TOTAL_VOCAB),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Inference: shared query
# ═════════════════════════════════════════════════════════════════════════════

def _get_pitcher_game_sequences(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: int | None = None,
    game_pk: int | None = None,
) -> pd.DataFrame:
    """Fetch ordered pitches for a pitcher, optionally filtered by season/game."""
    filters = ["pitcher_id = $1", "pitch_type IS NOT NULL"]
    params: list = [pitcher_id]
    idx = 2

    if season is not None:
        filters.append(f"EXTRACT(YEAR FROM game_date) = ${idx}")
        params.append(season)
        idx += 1
    if game_pk is not None:
        filters.append(f"game_pk = ${idx}")
        params.append(game_pk)
        idx += 1

    where = " AND ".join(filters)
    query = f"""
        SELECT
            game_pk, pitcher_id, pitch_type, plate_x, plate_z,
            release_speed, balls, strikes, outs_when_up,
            on_1b, on_2b, on_3b, stand, inning, at_bat_number, pitch_number
        FROM pitches
        WHERE {where}
        ORDER BY game_pk, at_bat_number, pitch_number
    """
    return conn.execute(query, params).fetchdf()


def _score_sequences(
    model: PitchGPTModel,
    df: pd.DataFrame,
) -> list[dict]:
    """Score each game sequence and return per-pitch NLL values.

    Returns a list of dicts, one per game:
        {"game_pk": ..., "per_pitch_nll": [...], "mean_nll": ..., "pitch_types": [...]}
    """
    device = next(model.parameters()).device
    results = []

    for game_pk, game_df in df.groupby("game_pk"):
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
                score_diff=0,
            )
            contexts_list.append(PitchTokenizer.context_to_tensor(ctx))

        # Truncate to model's max sequence length
        max_len = model.max_seq_len
        tokens_list = tokens_list[:max_len]
        contexts_list = contexts_list[:max_len]
        pitch_types = pitch_types[:max_len]

        if len(tokens_list) < 2:
            continue

        input_tokens = torch.tensor(tokens_list[:-1], dtype=torch.long).unsqueeze(0).to(device)
        target_tokens = torch.tensor(tokens_list[1:], dtype=torch.long).to(device)
        context_tensor = torch.stack(contexts_list[:-1]).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tokens, context_tensor)  # (1, S, V)
            logits = logits.squeeze(0)  # (S, V)

            # Per-pitch cross entropy
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            per_pitch_nll = []
            for i in range(len(target_tokens)):
                t = target_tokens[i].item()
                if 0 <= t < VOCAB_SIZE:
                    nll = -log_probs[i, t].item()
                else:
                    nll = 0.0
                per_pitch_nll.append(round(nll, 4))

        results.append({
            "game_pk": int(game_pk),
            "per_pitch_nll": per_pitch_nll,
            "mean_nll": round(float(np.mean(per_pitch_nll)), 4) if per_pitch_nll else 0.0,
            "pitch_types": pitch_types[1:],  # aligned with target
        })

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Public inference API
# ═════════════════════════════════════════════════════════════════════════════

def calculate_predictability(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: int | None = None,
    model_version: str = "1",
) -> dict:
    """Pitch Predictability Score (PPS): average NLL across all games.

    Lower PPS (lower perplexity) = more predictable pitcher.
    Higher PPS = less predictable / harder to model.

    Returns:
        Dict with ``pitcher_id``, ``season``, ``pps`` (mean NLL),
        ``perplexity`` (exp(pps)), ``n_games``, ``n_pitches``,
        ``game_details`` (per-game breakdown).
    """
    model = _load_model(model_version)
    df = _get_pitcher_game_sequences(conn, pitcher_id, season=season)

    if df.empty:
        return {
            "pitcher_id": pitcher_id,
            "season": season,
            "pps": 0.0,
            "perplexity": 1.0,
            "n_games": 0,
            "n_pitches": 0,
            "game_details": [],
        }

    game_results = _score_sequences(model, df)

    all_nll = []
    for g in game_results:
        all_nll.extend(g["per_pitch_nll"])

    mean_nll = float(np.mean(all_nll)) if all_nll else 0.0
    perplexity = math.exp(min(mean_nll, 20))

    return {
        "pitcher_id": pitcher_id,
        "season": season,
        "pps": round(mean_nll, 4),
        "perplexity": round(perplexity, 2),
        "n_games": len(game_results),
        "n_pitches": len(all_nll),
        "game_details": game_results,
    }


def calculate_disruption_index(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    game_pk: int,
    model_version: str = "1",
) -> dict:
    """Per-pitch surprise scores for a specific game.

    Spikes in the returned NLL values indicate unexpected / disruptive
    pitch selections.

    Returns:
        Dict with ``pitcher_id``, ``game_pk``, ``per_pitch_surprise``,
        ``pitch_types``, ``mean_surprise``, ``max_surprise``.
    """
    model = _load_model(model_version)
    df = _get_pitcher_game_sequences(conn, pitcher_id, game_pk=game_pk)

    if df.empty or len(df) < 2:
        return {
            "pitcher_id": pitcher_id,
            "game_pk": game_pk,
            "per_pitch_surprise": [],
            "pitch_types": [],
            "mean_surprise": 0.0,
            "max_surprise": 0.0,
        }

    game_results = _score_sequences(model, df)

    if not game_results:
        return {
            "pitcher_id": pitcher_id,
            "game_pk": game_pk,
            "per_pitch_surprise": [],
            "pitch_types": [],
            "mean_surprise": 0.0,
            "max_surprise": 0.0,
        }

    result = game_results[0]
    surprise = result["per_pitch_nll"]

    return {
        "pitcher_id": pitcher_id,
        "game_pk": game_pk,
        "per_pitch_surprise": surprise,
        "pitch_types": result["pitch_types"],
        "mean_surprise": round(float(np.mean(surprise)), 4) if surprise else 0.0,
        "max_surprise": round(float(np.max(surprise)), 4) if surprise else 0.0,
    }


def calculate_predictability_by_catcher(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: int | None = None,
    model_version: str = "1",
) -> list[dict]:
    """Compute PPS grouped by catcher for a pitcher.

    For each catcher that caught games for this pitcher, computes a
    separate PPS score by filtering pitch sequences to only those games.

    Args:
        conn: DuckDB connection.
        pitcher_id: MLB pitcher ID.
        season: Optional season filter.
        model_version: Model checkpoint version.

    Returns:
        List of dicts, one per catcher, each containing:
        - catcher_id: int
        - catcher_name: str (from players table, or str(id))
        - games_caught: int
        - pps: float (mean NLL for games with this catcher)
        - perplexity: float
        - game_details: list of per-game results
    """
    model = _load_model(model_version)

    # Find distinct catchers for this pitcher
    season_filter = ""
    params: list = [pitcher_id]
    idx = 2
    if season is not None:
        season_filter = f"AND EXTRACT(YEAR FROM game_date) = ${idx}"
        params.append(season)
        idx += 1

    catchers_df = conn.execute(f"""
        SELECT DISTINCT fielder_2 AS catcher_id
        FROM pitches
        WHERE pitcher_id = $1
          AND fielder_2 IS NOT NULL
          {season_filter}
    """, params).fetchdf()

    if catchers_df.empty:
        return []

    # Get catcher names from players table
    catcher_ids = catchers_df["catcher_id"].tolist()
    names_map: dict[int, str] = {}
    try:
        if catcher_ids:
            ids_str = ", ".join(str(int(c)) for c in catcher_ids)
            names_df = conn.execute(f"""
                SELECT player_id, full_name FROM players
                WHERE player_id IN ({ids_str})
            """).fetchdf()
            for _, row in names_df.iterrows():
                names_map[int(row["player_id"])] = str(row["full_name"])
    except Exception:
        pass

    # For each catcher, get games and score them
    results: list[dict] = []
    for catcher_id in catcher_ids:
        catcher_id = int(catcher_id)

        # Get game_pks where this catcher caught for this pitcher
        game_params: list = [pitcher_id, catcher_id]
        gp_idx = 3
        game_season_filter = ""
        if season is not None:
            game_season_filter = f"AND EXTRACT(YEAR FROM game_date) = ${gp_idx}"
            game_params.append(season)

        games_df = conn.execute(f"""
            SELECT DISTINCT game_pk
            FROM pitches
            WHERE pitcher_id = $1
              AND fielder_2 = $2
              {game_season_filter}
        """, game_params).fetchdf()

        if games_df.empty:
            continue

        game_pks = games_df["game_pk"].tolist()

        # Fetch all pitch sequences for this pitcher in these specific games
        pks_str = ", ".join(str(int(g)) for g in game_pks)
        pitch_query = f"""
            SELECT
                game_pk, pitcher_id, pitch_type, plate_x, plate_z,
                release_speed, balls, strikes, outs_when_up,
                on_1b, on_2b, on_3b, stand, inning, at_bat_number, pitch_number
            FROM pitches
            WHERE pitcher_id = $1
              AND pitch_type IS NOT NULL
              AND game_pk IN ({pks_str})
            ORDER BY game_pk, at_bat_number, pitch_number
        """
        pitches_df = conn.execute(pitch_query, [pitcher_id]).fetchdf()

        if pitches_df.empty or len(pitches_df) < 2:
            continue

        game_results = _score_sequences(model, pitches_df)

        all_nll: list[float] = []
        for g in game_results:
            all_nll.extend(g["per_pitch_nll"])

        if not all_nll:
            continue

        mean_nll = float(np.mean(all_nll))
        results.append({
            "catcher_id": catcher_id,
            "catcher_name": names_map.get(catcher_id, str(catcher_id)),
            "games_caught": len(game_results),
            "pps": round(mean_nll, 4),
            "perplexity": round(math.exp(min(mean_nll, 20)), 2),
            "n_pitches": len(all_nll),
            "game_details": game_results,
        })

    # Sort by PPS ascending (most predictable first)
    results.sort(key=lambda x: x["pps"])
    return results


def batch_calculate(
    conn: duckdb.DuckDBPyConnection,
    season: int | None = None,
    min_pitches: int = MIN_PITCHES_LEADERBOARD,
    model_version: str = "1",
) -> pd.DataFrame:
    """PPS leaderboard for all qualifying pitchers.

    Args:
        conn: DuckDB connection.
        season: Season filter.
        min_pitches: Minimum pitch count to qualify.
        model_version: Checkpoint version tag.

    Returns:
        DataFrame with columns: pitcher_id, pps, perplexity, n_games, n_pitches.
    """
    cols = ["pitcher_id", "pps", "perplexity", "n_games", "n_pitches"]

    try:
        model = _load_model(model_version)
    except FileNotFoundError:
        logger.warning("PitchGPT model not found. Train first via train_pitchgpt().")
        return pd.DataFrame(columns=cols)

    season_filter = ""
    params: list = []
    if season is not None:
        season_filter = "AND EXTRACT(YEAR FROM game_date) = $1"
        params = [season]

    qualifying_query = f"""
        SELECT pitcher_id, COUNT(*) AS pitch_count
        FROM pitches
        WHERE pitch_type IS NOT NULL
          {season_filter}
        GROUP BY pitcher_id
        HAVING COUNT(*) >= {int(min_pitches)}
    """
    qualifying = conn.execute(qualifying_query, params).fetchdf()

    if qualifying.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for _, row in qualifying.iterrows():
        pid = int(row["pitcher_id"])
        try:
            df = _get_pitcher_game_sequences(conn, pid, season=season)
            if df.empty or len(df) < 2:
                continue
            game_results = _score_sequences(model, df)
            all_nll = []
            for g in game_results:
                all_nll.extend(g["per_pitch_nll"])
            if not all_nll:
                continue
            mean_nll = float(np.mean(all_nll))
            rows.append({
                "pitcher_id": pid,
                "pps": round(mean_nll, 4),
                "perplexity": round(math.exp(min(mean_nll, 20)), 2),
                "n_games": len(game_results),
                "n_pitches": len(all_nll),
            })
        except Exception as exc:
            logger.warning("PPS failed for pitcher %d: %s", pid, exc)

    if not rows:
        return pd.DataFrame(columns=cols)

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("pps", ascending=True).reset_index(drop=True)
    return result_df
