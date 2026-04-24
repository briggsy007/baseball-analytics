"""
PitchGPT sampling-fidelity evaluation vs matched LSTM baseline.

The perplexity gate on the 2025 pitcher-disjoint holdout collapsed at
matched 10K scale (v2 +3.13%, v1 +2.57%, both FAIL the 15% gate).  This
script reframes PitchGPT's claim around **simulation fidelity** rather
than LSTM-beating next-token accuracy:

    Hypothesis: PitchGPT, by virtue of its richer contextual representation
    and transformer architecture, generates pitch sequences whose marginal
    and joint distributions more closely match empirical 2025 distributions
    than a matched LSTM baseline's generations do.

We draw real 2025 starting contexts (game_pk, pitcher_id, batter_id,
first-pitch situation) from pitcher-disjoint PAs (pitchers not in the
2015-2022 train cohort), then auto-regressively sample N continuations
per PA from each model at temperature=1.0.  Each model produces N*PAs
sampled sequences.  The empirical 2025 distribution of those same PAs
is the reference.

Metrics (each reported as PitchGPT-vs-empirical, LSTM-vs-empirical, and
the signed difference with bootstrap CI):

  1. Marginal pitch-type distribution   -- KL + chi-square
  2. Marginal zone distribution         -- KL + chi-square
  3. Velocity distribution              -- 1-D Wasserstein
  4. Sequential pitch-type 2-gram       -- Frobenius norm of (sampled - empirical)
  5. Implied at-bat outcome distribution -- strikeout/walk/contact rates

Guardrails (see user's brief):
  * DO NOT commit.  Leave everything unstaged.
  * DO NOT touch existing model checkpoints; only write pitch_lstm_10k.pt.
  * Use read-only DB connections for sampling.
  * Report null results honestly — the null result (PitchGPT == LSTM on
    sampling fidelity) is scientifically valuable for the narrowing
    of the flagship claim.

Output:
  - results/pitchgpt/sampling_fidelity_2026_04_24/metrics.json
  - results/pitchgpt/sampling_fidelity_2026_04_24/report.md
  - results/pitchgpt/sampling_fidelity_2026_04_24/sampled_sequences_pg.csv (optional)
  - results/pitchgpt/sampling_fidelity_2026_04_24/sampled_sequences_lstm.csv (optional)
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
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt import (  # noqa: E402
    BOS_TOKEN,
    CONTEXT_DIM,
    NUM_PITCH_TYPES,
    NUM_VELO_BUCKETS,
    NUM_ZONES,
    PAD_TOKEN,
    PITCH_TYPE_MAP,
    TOTAL_VOCAB,
    VOCAB_SIZE,
    PitchGPTModel,
    PitchSequenceDataset,
    PitchTokenizer,
    _collate_fn,
    _get_device,
    _safe_bool,
    _safe_int,
    _safe_str,
    _score_diff_for_pitch,
    _runs_scored_on_event,
    _fetch_hp_umpire_by_game,
    _fetch_prior_season_ump_tendency,
    _fetch_season_league_median,
    audit_no_game_overlap,
)
from src.analytics.pitch_lstm import PitchLSTMNetwork  # noqa: E402
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitchgpt_sampling_fidelity")

# ═════════════════════════════════════════════════════════════════════════════
# Constants
# ═════════════════════════════════════════════════════════════════════════════
TRAIN_RANGE = (2015, 2022)
VAL_RANGE = (2023, 2023)
HOLDOUT_RANGE = (2025, 2025)

DEFAULT_BATCH = 32
DEFAULT_LR = 1e-3
GRAD_CLIP = 1.0

# Use a fixed 6-pitch horizon for sampling.  Rationale: PA length
# distribution in 2025 peaks at 3-5 pitches with long tail; 6 covers
# the 76th percentile and keeps sequences long enough to compute
# 2-gram transitions.  PA termination on events is awkward for
# generative models (would require an event-prediction head), so we
# use a fixed horizon as a simpler, defensible operationalisation.
SAMPLE_HORIZON = 6

# Number of sample sequences per (model, PA start).
N_SAMPLES_PER_PA = 10
TEMPERATURE = 1.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═════════════════════════════════════════════════════════════════════════════
# Load PitchGPT checkpoint
# ═════════════════════════════════════════════════════════════════════════════

def load_pitchgpt_checkpoint(path: Path, device: torch.device) -> PitchGPTModel:
    ck = torch.load(str(path), map_location=device, weights_only=True)
    cfg = ck["config"]
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
    model.context_dim = context_dim
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Train / load LSTM at 10K
# ═════════════════════════════════════════════════════════════════════════════

def train_lstm(
    train_ds,
    val_ds,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    context_dim: int,
) -> tuple[PitchLSTMNetwork, dict]:
    _set_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn,
    )

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


def load_lstm_checkpoint(path: Path, device: torch.device) -> PitchLSTMNetwork:
    """Load an LSTM checkpoint from disk."""
    ck = torch.load(str(path), map_location=device, weights_only=True)
    cfg = ck["config"]
    model = PitchLSTMNetwork(
        vocab_size=cfg.get("vocab_size", TOTAL_VOCAB),
        context_dim=cfg.get("context_dim", CONTEXT_DIM),
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        output_vocab=cfg.get("output_vocab", VOCAB_SIZE),
    )
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# 2025 PA starting-context sampler
# ═════════════════════════════════════════════════════════════════════════════

def fetch_2025_pa_starts(
    conn,
    n_pas: int,
    exclude_pitcher_ids: set[int],
    seed: int,
    context_dim: int,
) -> tuple[list[dict], pd.DataFrame]:
    """Sample starting contexts from real 2025 PAs (pitcher-disjoint).

    A PA is a ``(game_pk, at_bat_number, pitcher_id, batter_id)`` group.
    We require the PA to have at least 2 pitches (so we have something
    to evaluate 2-grams against).  The "start context" is the first
    pitch's situational context (count 0-0, outs, runners, stand,
    inning, score, ump tendency).  The empirical continuation is the
    actual pitch sequence that followed in-game.

    Returns:
        (pa_starts, pa_df)
        pa_starts: list of dicts with keys
            game_pk, at_bat_number, pitcher_id, batter_id,
            start_context_tensor (CPU tensor, shape (context_dim,)),
            empirical_tokens (list of int tokens, real continuation),
            empirical_pitch_types (list of str),
            empirical_zones (list of int),
            empirical_velos (list of float),
            empirical_outcome (str: strikeout / walk / in_play / other)
        pa_df: full pitch-level DataFrame for provenance / CSVs.
    """
    rng = np.random.default_rng(seed)

    # Sample PA keys FIRST, at SQL level, to avoid pulling full 2025 frame.
    pids_str = ", ".join(str(int(p)) for p in exclude_pitcher_ids) if exclude_pitcher_ids else "-1"
    pa_key_query = f"""
        WITH pa_counts AS (
            SELECT game_pk, at_bat_number, pitcher_id, COUNT(*) AS n_pitches
            FROM pitches
            WHERE pitch_type IS NOT NULL
              AND EXTRACT(YEAR FROM game_date) = 2025
              AND pitcher_id NOT IN ({pids_str})
              AND pitcher_id IS NOT NULL
              AND batter_id IS NOT NULL
            GROUP BY game_pk, at_bat_number, pitcher_id
        )
        SELECT game_pk, at_bat_number, pitcher_id
        FROM pa_counts
        WHERE n_pitches >= 2
    """
    pa_keys_df = conn.execute(pa_key_query).fetchdf()
    logger.info("Total eligible PAs in 2025 (pitcher-disjoint, >=2 pitches): %d", len(pa_keys_df))
    if pa_keys_df.empty:
        return [], pd.DataFrame()

    n_draw = min(n_pas, len(pa_keys_df))
    draw_idx = rng.choice(len(pa_keys_df), size=n_draw, replace=False)
    pa_keys_df = pa_keys_df.iloc[draw_idx].reset_index(drop=True)

    # Temp-table join to pull ONLY the sampled PAs' pitch rows.
    conn.register("_pgfs_pa_keys", pa_keys_df)
    try:
        query = """
            SELECT p.game_pk, p.pitcher_id, p.batter_id, p.pitch_type,
                   p.plate_x, p.plate_z, p.release_speed, p.balls, p.strikes,
                   p.outs_when_up, p.on_1b, p.on_2b, p.on_3b, p.stand, p.inning,
                   p.inning_topbot, p.events, p.description,
                   p.delta_run_exp, p.at_bat_number, p.pitch_number, p.game_date
            FROM pitches p
            INNER JOIN _pgfs_pa_keys k
                ON p.game_pk = k.game_pk
                AND p.at_bat_number = k.at_bat_number
                AND p.pitcher_id = k.pitcher_id
            WHERE p.pitch_type IS NOT NULL
              AND EXTRACT(YEAR FROM p.game_date) = 2025
            ORDER BY p.game_pk, p.at_bat_number, p.pitch_number
        """
        df = conn.execute(query).fetchdf()
    finally:
        conn.unregister("_pgfs_pa_keys")
    if df.empty:
        return [], df

    # Reconstruct per-pitch score_diff for score bucket.  Since we
    # filtered to sampled PAs only, score reconstruction uses a per-PA
    # zero-start (not exact but fine for the starting-context's 5-bucket
    # discretisation — the first pitch of a PA inherits the score from
    # prior PAs in the game, which we cannot reconstruct without the
    # prior events.  Use 0 for the starting-context score_diff — a
    # documented simplification.  The bucket is robust: abs(diff)<=3
    # → "balanced game" bucket which covers the majority of cases.
    from src.analytics.pitchgpt import _compute_per_pitch_score_diff
    # _compute_per_pitch_score_diff handles within-game running; since
    # we only have the sampled PAs' pitches, the "running score" is
    # approximate.  For the STARTING context (first pitch of each PA)
    # we'll use the in-batch approximation.
    df = df.assign(_score_diff=_compute_per_pitch_score_diff(df))

    # Attach ump scalar for context width 35 (v2) compatibility.
    # Vectorised version (per-game mapping, not per-row .apply).
    seasons_present = sorted({int(s) for s in df["game_date"].dt.year.unique()})
    ump_by_game = _fetch_hp_umpire_by_game(conn, seasons_present)
    tendency_map = _fetch_prior_season_ump_tendency(conn, seasons_present)
    median_by_season = _fetch_season_league_median(conn, seasons_present)

    gpk_arr = df["game_pk"].astype("int64").to_numpy()
    year_arr = df["game_date"].dt.year.astype("int64").to_numpy()
    scalars = np.zeros(len(df), dtype=np.float64)
    # Per-game lookup cache (much faster than per-row).
    game_cache: dict[tuple[int, int], float] = {}
    for i in range(len(df)):
        key = (int(gpk_arr[i]), int(year_arr[i]))
        v = game_cache.get(key)
        if v is None:
            season = key[1]
            median = median_by_season.get(season, 0.0)
            ump_name = ump_by_game.get(key[0])
            if ump_name is None:
                v = median
            else:
                v = float(tendency_map.get((ump_name, season), median))
            game_cache[key] = v
        scalars[i] = v
    df["_ump_scalar"] = scalars

    # df is already restricted to sampled PAs.  Keep pa_df as alias for
    # backward-compat naming.
    pa_df = df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)

    results: list[dict] = []
    pa_grouped = pa_df.groupby(["game_pk", "at_bat_number", "pitcher_id"], sort=False)
    for (gpk, abn, pid), pa_rows in pa_grouped:
        pa_rows = pa_rows.sort_values("pitch_number")
        if len(pa_rows) < 2:
            continue
        first_row = pa_rows.iloc[0]

        # Build context tensor for the FIRST pitch (count 0-0, the start).
        # This is the context from which both models will roll forward.
        ctx_list = PitchTokenizer.encode_context(
            balls=_safe_int(first_row.get("balls"), 0),
            strikes=_safe_int(first_row.get("strikes"), 0),
            outs=_safe_int(first_row.get("outs_when_up"), 0),
            on_1b=_safe_bool(first_row.get("on_1b")),
            on_2b=_safe_bool(first_row.get("on_2b")),
            on_3b=_safe_bool(first_row.get("on_3b")),
            stand=_safe_str(first_row.get("stand"), "R"),
            inning=_safe_int(first_row.get("inning"), 1),
            score_diff=_safe_int(first_row.get("_score_diff"), 0),
        )
        ump_val = float(first_row.get("_ump_scalar", 0.0))
        start_ctx = PitchTokenizer.context_to_tensor(ctx_list, ump_scalar=ump_val)
        if context_dim < CONTEXT_DIM:
            start_ctx = start_ctx[:context_dim]

        # Empirical continuation tokens.
        emp_tokens: list[int] = []
        emp_pt: list[str] = []
        emp_zones: list[int] = []
        emp_velos: list[float] = []
        for _, r in pa_rows.iterrows():
            tok = PitchTokenizer.encode(
                r.get("pitch_type"),
                r.get("plate_x"), r.get("plate_z"),
                r.get("release_speed"),
            )
            emp_tokens.append(tok)
            emp_pt.append(str(r.get("pitch_type") or "UN"))
            emp_zones.append(PitchTokenizer.location_to_zone(r.get("plate_x"), r.get("plate_z")))
            v = r.get("release_speed")
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                emp_velos.append(float(v))

        # Empirical outcome: look at last pitch's ``events`` column.
        last_event = pa_rows.iloc[-1].get("events")
        last_descr = pa_rows.iloc[-1].get("description")
        outcome = _classify_outcome(last_event, last_descr)

        # Level / count / inning / leverage stratification metadata.
        inning = _safe_int(first_row.get("inning"), 1)
        count_str = f"{_safe_int(first_row.get('balls'), 0)}-{_safe_int(first_row.get('strikes'), 0)}"
        score_diff = _safe_int(first_row.get("_score_diff"), 0)
        leverage = _classify_leverage(inning, score_diff, first_row)

        results.append({
            "game_pk": int(gpk),
            "at_bat_number": int(abn),
            "pitcher_id": int(pid),
            "batter_id": int(first_row.get("batter_id", 0) or 0),
            "start_context_tensor": start_ctx.cpu(),
            "empirical_tokens": emp_tokens,
            "empirical_pitch_types": emp_pt,
            "empirical_zones": emp_zones,
            "empirical_velos": emp_velos,
            "empirical_outcome": outcome,
            "inning": inning,
            "count": count_str,
            "leverage": leverage,
            "score_diff": score_diff,
            "context_dim": context_dim,
        })

    logger.info("Built %d PA starts", len(results))
    return results, pa_df


def _classify_outcome(events: Any, descr: Any) -> str:
    """Classify the PA-ending outcome from Statcast events/description.

    Categories: strikeout, walk, in_play_out, in_play_hit, in_play_other,
    hit_by_pitch, other.
    """
    ev = str(events or "").lower()
    if ev in {"strikeout", "strikeout_double_play"}:
        return "strikeout"
    if ev in {"walk", "intent_walk"}:
        return "walk"
    if ev == "hit_by_pitch":
        return "hit_by_pitch"
    if ev in {"single", "double", "triple", "home_run"}:
        return "in_play_hit"
    if ev in {
        "field_out", "force_out", "grounded_into_double_play",
        "sac_fly", "sac_fly_double_play", "sac_bunt",
        "double_play", "triple_play", "fielders_choice",
        "fielders_choice_out", "field_error",
    }:
        return "in_play_out"
    d = str(descr or "").lower()
    if d.startswith("hit_into_play") or d == "in_play":
        return "in_play_other"
    return "other"


def _classify_leverage(inning: int, score_diff: int, row) -> str:
    """Lightweight leverage classifier.

    - high: 7th+ inning and |score_diff| <= 2
    - medium: 4th-6th inning and |score_diff| <= 3
    - low: else
    """
    if inning >= 7 and abs(score_diff) <= 2:
        return "high"
    if 4 <= inning <= 6 and abs(score_diff) <= 3:
        return "medium"
    return "low"


# ═════════════════════════════════════════════════════════════════════════════
# Autoregressive sampling
# ═════════════════════════════════════════════════════════════════════════════

def _update_context_after_pitch(
    prev_ctx: torch.Tensor,
    prev_tok: int,
    prev_balls: int,
    prev_strikes: int,
    start_meta: dict,
    context_dim: int,
) -> tuple[torch.Tensor, int, int, bool]:
    """Given the previous context and the (just-sampled) prev token,
    return the next context tensor plus updated (balls, strikes).

    We only update the count-state dimension; outs/runners/inning/ump
    stay frozen within a PA (true in real baseball most of the time
    except for end-of-inning / steals which we ignore for this
    simulation).

    Returns (new_ctx, new_balls, new_strikes, terminated).
    ``terminated`` = True iff the count reached a PA-ending state (walk
    on 4 balls or strikeout on 3 strikes) — the caller may choose to
    respect or ignore this (we currently use a fixed horizon regardless).
    """
    decoded = PitchTokenizer.decode(prev_tok)
    # We don't know the actual called ball/strike outcome of the sampled
    # pitch (that requires an outcome head).  Approximation: infer from
    # zone — if it's within the strike-zone 5x5 grid AND the decode
    # suggests a strike region we bump strikes; otherwise bump balls.
    # This is a heuristic; we document it in the report.
    zone = decoded["zone"]
    if 0 <= zone <= 24:
        # In-strike-zone grid; assume strike (simplification).
        new_balls = prev_balls
        new_strikes = min(prev_strikes + 1, 2)  # cap at 2 — a 3rd strike is out
    else:
        # Out-of-zone (25); assume ball.
        new_balls = min(prev_balls + 1, 3)  # cap at 3 — a 4th ball is a walk
        new_strikes = prev_strikes
    terminated = (new_balls == 3 and prev_balls == 3) or (new_strikes == 2 and prev_strikes == 2)

    # Rebuild context with updated count.
    ctx_list = PitchTokenizer.encode_context(
        balls=new_balls,
        strikes=new_strikes,
        outs=_safe_int(start_meta.get("outs_when_up", 0), 0),
        on_1b=_safe_bool(start_meta.get("on_1b", 0)),
        on_2b=_safe_bool(start_meta.get("on_2b", 0)),
        on_3b=_safe_bool(start_meta.get("on_3b", 0)),
        stand=_safe_str(start_meta.get("stand", "R"), "R"),
        inning=_safe_int(start_meta.get("inning", 1), 1),
        score_diff=_safe_int(start_meta.get("score_diff", 0), 0),
    )
    ump_scalar = float(start_meta.get("ump_scalar", 0.0))
    new_ctx = PitchTokenizer.context_to_tensor(ctx_list, ump_scalar=ump_scalar)
    if context_dim < CONTEXT_DIM:
        new_ctx = new_ctx[:context_dim]
    return new_ctx, new_balls, new_strikes, terminated


def _batched_context_update(
    prev_tokens: torch.Tensor,     # (B,) last sampled token per rollout
    balls: torch.Tensor,           # (B,) current balls count
    strikes: torch.Tensor,         # (B,) current strikes count
    start_ctxs: torch.Tensor,      # (B, context_dim) starting context per rollout
    start_meta: list[dict],        # per-rollout start-meta bag (inning, score_diff, etc.)
    context_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorised per-step context update.  Returns (new_ctx, new_balls, new_strikes).

    Counts advance using the zone-based ball/strike heuristic (zone 25
    = ball, 0..24 = strike).  Capped at 3 balls / 2 strikes within the
    context encoding (PA-termination is handled by the outcome heuristic
    in implied_outcome_distribution_from_samples).
    """
    B = prev_tokens.shape[0]
    # Decode zone from token: zone = (tok // NUM_VELO_BUCKETS) % NUM_ZONES.
    # But tokens can be PAD/BOS which we treat as out-of-zone (zone 25).
    toks = prev_tokens.to(torch.long)
    in_vocab = (toks >= 0) & (toks < VOCAB_SIZE)
    zone = torch.where(
        in_vocab,
        (toks // NUM_VELO_BUCKETS) % NUM_ZONES,
        torch.full_like(toks, NUM_ZONES - 1),  # out-of-zone
    )
    is_strike = (zone >= 0) & (zone <= 24)
    new_balls = torch.where(is_strike, balls, torch.clamp(balls + 1, max=3))
    new_strikes = torch.where(is_strike, torch.clamp(strikes + 1, max=2), strikes)

    # Build context vectors vectorially.  We rebuild the count / outs /
    # runners / stand / inning / score_diff / ump cells fresh each step.
    # For speed, we compute only the COUNT one-hot block and copy the
    # rest from start_ctxs.  The count state is balls*3 + strikes (0..11).
    new_ctx = start_ctxs.clone()
    # Zero out the first NUM_COUNT_STATES=12 bits (the count one-hot).
    new_ctx[:, :12] = 0.0
    count_idx = (new_balls.clamp(max=3).to(torch.long) * 3 + new_strikes.clamp(max=2).to(torch.long))
    new_ctx.scatter_(1, count_idx.unsqueeze(1), 1.0)
    return new_ctx, new_balls, new_strikes


@torch.no_grad()
def _sample_continuations_batched(
    model: nn.Module,
    pa_starts: list[dict],
    n_samples: int,
    horizon: int,
    temperature: float,
    device: torch.device,
    seed: int,
    batch_size: int,
    model_type: str,   # "pitchgpt" or "lstm"
) -> list[list[list[int]]]:
    """Unified batched autoregressive sampler for both PitchGPT and LSTM.

    Rollouts are flattened across (PA, sample): we materialise
    ``len(pa_starts) * n_samples`` rollouts, process them in batches of
    ``batch_size``, and reassemble at the end.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.eval()
    if model_type == "pitchgpt":
        m_ctx_dim = getattr(model, "context_dim", CONTEXT_DIM)
        out_vocab = VOCAB_SIZE
    elif model_type == "lstm":
        m_ctx_dim = int(model.context_proj.in_features)
        out_vocab = int(model.output_vocab)
    else:
        raise ValueError(model_type)

    # Materialise per-rollout start contexts + meta.
    n_pa = len(pa_starts)
    total = n_pa * n_samples

    start_ctx_stack = torch.stack([
        pa["start_context_tensor"][:m_ctx_dim]
        for pa in pa_starts
    ])  # (n_pa, m_ctx_dim)
    # Expand to rollouts:
    start_ctx_all = start_ctx_stack.repeat_interleave(n_samples, dim=0).to(device)  # (total, m_ctx_dim)

    # Output tokens: shape (total, horizon) filled after sampling.
    sampled_tokens_all = torch.full((total, horizon), fill_value=PAD_TOKEN, dtype=torch.long, device=device)

    # Loop over rollout batches.
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        B = end - start
        ctx_start = start_ctx_all[start:end]  # (B, m_ctx_dim)

        # Current running-token sequence for this batch.  Begin with BOS.
        token_seq = torch.full((B, 1), fill_value=BOS_TOKEN, dtype=torch.long, device=device)
        # Context sequence aligned with tokens.
        ctx_seq = ctx_start.unsqueeze(1)  # (B, 1, m_ctx_dim)
        balls = torch.zeros(B, dtype=torch.long, device=device)
        strikes = torch.zeros(B, dtype=torch.long, device=device)

        for step in range(horizon):
            if model_type == "pitchgpt":
                logits = model(token_seq, ctx_seq)  # (B, S, VOCAB_SIZE)
            else:
                logits = model(token_seq, ctx_seq)  # (B, S, out_vocab)
            last_logits = logits[:, -1, :] / temperature
            probs = torch.nn.functional.softmax(last_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (B,)
            # Record.
            sampled_tokens_all[start:end, step] = sampled
            # Update tokens + context for next step.
            new_ctx, balls, strikes = _batched_context_update(
                sampled, balls, strikes, ctx_start, None, m_ctx_dim, device,
            )
            token_seq = torch.cat([token_seq, sampled.unsqueeze(1)], dim=1)
            ctx_seq = torch.cat([ctx_seq, new_ctx.unsqueeze(1)], dim=1)
            # Safety: if sequence gets too long for PitchGPT max_seq, truncate.
            if model_type == "pitchgpt":
                max_s = int(getattr(model, "max_seq_len", 256))
                if token_seq.size(1) > max_s:
                    token_seq = token_seq[:, -max_s:]
                    ctx_seq = ctx_seq[:, -max_s:]

    # Reshape back to (n_pa, n_samples, horizon)
    sampled_cpu = sampled_tokens_all.cpu().numpy()  # (total, horizon)
    all_pa_samples: list[list[list[int]]] = []
    for i in range(n_pa):
        pa_samples = []
        for s in range(n_samples):
            row = sampled_cpu[i * n_samples + s].tolist()
            pa_samples.append([int(t) for t in row])
        all_pa_samples.append(pa_samples)
    return all_pa_samples


def sample_continuations_pitchgpt(
    model: PitchGPTModel,
    pa_starts: list[dict],
    n_samples: int,
    horizon: int,
    temperature: float,
    device: torch.device,
    seed: int,
    batch_size: int = 512,
) -> list[list[list[int]]]:
    """Sample ``n_samples`` continuations per PA-start using batched rollouts."""
    return _sample_continuations_batched(
        model, pa_starts, n_samples, horizon, temperature, device,
        seed, batch_size, "pitchgpt",
    )


def sample_continuations_lstm(
    model: PitchLSTMNetwork,
    pa_starts: list[dict],
    n_samples: int,
    horizon: int,
    temperature: float,
    device: torch.device,
    seed: int,
    batch_size: int = 512,
) -> list[list[list[int]]]:
    """Sample ``n_samples`` continuations per PA-start using batched rollouts."""
    return _sample_continuations_batched(
        model, pa_starts, n_samples, horizon, temperature, device,
        seed, batch_size, "lstm",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Sample decoding helpers
# ═════════════════════════════════════════════════════════════════════════════

def decode_samples_to_flat_arrays(
    samples: list[list[list[int]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a sampled-tokens structure into flat arrays of pitch_type
    index, zone, and velo bucket."""
    pt_flat: list[int] = []
    zone_flat: list[int] = []
    velo_flat: list[int] = []
    for pa_samples in samples:
        for seq in pa_samples:
            for tok in seq:
                if tok < 0 or tok >= VOCAB_SIZE:
                    continue
                d = PitchTokenizer.decode(tok)
                pt_flat.append(int(d["pitch_type"] and PITCH_TYPE_MAP.get(d["pitch_type"], NUM_PITCH_TYPES - 1)))
                zone_flat.append(int(d["zone"]))
                velo_flat.append(int(d["velo_bucket"]))
    return np.asarray(pt_flat), np.asarray(zone_flat), np.asarray(velo_flat)


def decode_empirical_to_flat_arrays(
    pa_starts: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten empirical per-PA pitch arrays into arrays matching the
    decoded-sample arrays (same per-PA weighting — each PA contributes
    its real pitch count, not inflated by N_SAMPLES)."""
    pt_flat: list[int] = []
    zone_flat: list[int] = []
    velo_cont: list[float] = []
    for pa in pa_starts:
        for pt_str, z in zip(pa["empirical_pitch_types"], pa["empirical_zones"]):
            pt_flat.append(PITCH_TYPE_MAP.get(pt_str, NUM_PITCH_TYPES - 1))
            zone_flat.append(int(z))
        velo_cont.extend(pa["empirical_velos"])
    return np.asarray(pt_flat), np.asarray(zone_flat), np.asarray(velo_cont)


def velocity_bucket_to_midpoint(bucket: int) -> float:
    """Map a velo-bucket index to a rough midpoint (mph).

    Mapping matches PitchTokenizer.velocity_to_bucket:
      0 -> <80 (midpoint 75)
      1 -> 80-85 (82.5)
      2 -> 85-90 (87.5)
      3 -> 90-95 (92.5)
      4 -> 95+   (97.5)
    """
    mapping = {0: 75.0, 1: 82.5, 2: 87.5, 3: 92.5, 4: 97.5}
    return mapping.get(int(bucket), 87.5)


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════

def dist_histogram(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Categorical histogram with Laplace smoothing for KL."""
    alpha = 1e-6
    counts = np.full(n_bins, alpha, dtype=np.float64)
    if values.size == 0:
        return counts / counts.sum()
    for v in values:
        if 0 <= int(v) < n_bins:
            counts[int(v)] += 1
    return counts / counts.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q).  Both must sum to 1; small floor on q to avoid inf."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    q = np.clip(q, 1e-12, None)
    p = np.clip(p, 1e-12, None)
    return float(np.sum(p * np.log(p / q)))


def chi_square_stat(p_counts: np.ndarray, q_expected: np.ndarray) -> float:
    """Pearson chi-square with q treated as expected probabilities.

    p_counts: observed counts from sampled dist.
    q_expected: probability distribution (sums to 1).
    Returns chi^2 test statistic (NOT p-value).
    """
    n = p_counts.sum()
    expected = q_expected * n
    expected = np.clip(expected, 1.0, None)  # avoid divide-by-0 in sparse bins
    return float(np.sum((p_counts - expected) ** 2 / expected))


def wasserstein_1d(p_samples: np.ndarray, q_samples: np.ndarray) -> float:
    """1-D Wasserstein (earth mover's) distance between two samples.

    Implementation: sort both, equalize via empirical CDF, integrate
    |F_p(x) - F_q(x)| dx.  Uses scipy if available for exactness;
    falls back to a numpy quantile-based proxy.
    """
    if p_samples.size == 0 or q_samples.size == 0:
        return float("nan")
    try:
        from scipy.stats import wasserstein_distance  # type: ignore
        return float(wasserstein_distance(p_samples, q_samples))
    except ImportError:
        # Fallback: quantile matching on 200 quantiles.
        qs = np.linspace(0.005, 0.995, 200)
        p_q = np.quantile(p_samples, qs)
        q_q = np.quantile(q_samples, qs)
        return float(np.mean(np.abs(p_q - q_q)))


def transition_matrix_pt(sequences: list[list[int]]) -> np.ndarray:
    """Build a (NUM_PITCH_TYPES x NUM_PITCH_TYPES) row-stochastic
    transition matrix from pitch_type → pitch_type transitions in the
    given token sequences.
    """
    alpha = 1e-6
    counts = np.full((NUM_PITCH_TYPES, NUM_PITCH_TYPES), alpha, dtype=np.float64)
    for seq in sequences:
        if len(seq) < 2:
            continue
        for a, b in zip(seq[:-1], seq[1:]):
            if a < 0 or a >= VOCAB_SIZE or b < 0 or b >= VOCAB_SIZE:
                continue
            a_d = PitchTokenizer.decode(a)
            b_d = PitchTokenizer.decode(b)
            a_pt = PITCH_TYPE_MAP.get(a_d["pitch_type"], NUM_PITCH_TYPES - 1)
            b_pt = PITCH_TYPE_MAP.get(b_d["pitch_type"], NUM_PITCH_TYPES - 1)
            counts[a_pt, b_pt] += 1
    # Row-normalize
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums = np.clip(row_sums, 1e-12, None)
    return counts / row_sums


def transition_matrix_from_empirical(pa_starts: list[dict]) -> np.ndarray:
    seqs = []
    for pa in pa_starts:
        pt_seq = [
            PITCH_TYPE_MAP.get(pt, NUM_PITCH_TYPES - 1)
            for pt in pa["empirical_pitch_types"]
        ]
        seqs.append(_pt_indices_to_token_seq(pt_seq))
    return transition_matrix_pt(seqs)


def _pt_indices_to_token_seq(pt_indices: list[int]) -> list[int]:
    """Encode a pitch-type-only sequence as full tokens using a default
    zone/velo (middle of 5x5 grid, middle velo bucket).  This is fine
    for 2-gram pt-only transitions since we strip zone/velo in the
    decoder anyway."""
    out = []
    for pt in pt_indices:
        zone = 12  # center of 5x5 grid
        velo = 2
        out.append(pt * (NUM_ZONES * NUM_VELO_BUCKETS) + zone * NUM_VELO_BUCKETS + velo)
    return out


def sample_sequences_by_pa(samples: list[list[list[int]]]) -> list[list[int]]:
    """Flatten sample structure to one sequence per (PA, sample) pair."""
    flat = []
    for pa in samples:
        for seq in pa:
            flat.append(list(seq))
    return flat


# ═════════════════════════════════════════════════════════════════════════════
# Implied outcome distribution
# ═════════════════════════════════════════════════════════════════════════════

def implied_outcome_distribution_from_samples(
    samples: list[list[list[int]]],
    p_strike_per_pitch: float = 0.47,
    p_in_play_per_pitch: float = 0.18,
    seed: int = 12345,
) -> dict[str, float]:
    """Classify each sampled sequence into {strikeout, walk, in_play}.

    Empirical-calibrated heuristic:
      * Per-pitch, the 2025 MLB base rates (see ``description`` counts in
        the 2025 pitches table) are roughly: 35% balls (ball / blocked /
        auto), 47% strikes (called / swinging / foul / foul_tip etc.),
        18% in-play.
      * Within-zone (zones 0-24) we bump the strike probability to 0.60
        and ball prob to 0.22 (contact and fouls shift mass toward
        strikes when the pitch is near the plate).  Out-of-zone (zone
        25 / missing) we flip to 0.72 ball, 0.16 strike, 0.12 in-play.
      * Outcomes are sampled IID using the per-pitch class probs and
        walked forward until a terminal event (3 strikes, 4 balls,
        contact).  This is still a crude proxy — real Statcast
        classification depends on batter swing decision, umpire bias,
        and foul-ball dynamics — but it at least honours the empirical
        base rates rather than forcing every in-zone pitch to be a
        called strike.

    The seed makes outcome sampling reproducible.
    """
    rng = np.random.default_rng(seed)
    counts = {"strikeout": 0, "walk": 0, "in_play": 0}
    total_pa = 0

    # Per-zone empirical rates (rough MLB 2025 aggregates).
    in_zone_rates = np.array([0.22, 0.60, 0.18])  # ball, strike, in_play
    out_zone_rates = np.array([0.72, 0.16, 0.12])
    classes = ["ball", "strike", "in_play"]

    for pa_samples in samples:
        for seq in pa_samples:
            total_pa += 1
            balls, strikes = 0, 0
            cls = None
            for tok in seq:
                if tok < 0 or tok >= VOCAB_SIZE:
                    continue
                zone = PitchTokenizer.decode(tok)["zone"]
                rates = in_zone_rates if 0 <= zone <= 24 else out_zone_rates
                draw = rng.choice(3, p=rates)
                ev = classes[draw]
                if ev == "in_play":
                    cls = "in_play"
                    break
                if ev == "strike":
                    strikes += 1
                    if strikes >= 3:
                        cls = "strikeout"
                        break
                else:  # ball
                    balls += 1
                    if balls >= 4:
                        cls = "walk"
                        break
            if cls is None:
                # Hit horizon without termination — call it in_play
                # (the PA would have continued in reality).
                cls = "in_play"
            counts[cls] += 1
    if total_pa == 0:
        return {"strikeout": 0.0, "walk": 0.0, "in_play": 0.0}
    return {k: v / total_pa for k, v in counts.items()}


def empirical_outcome_distribution(pa_starts: list[dict]) -> dict[str, float]:
    """Collapse empirical outcomes into the three classes used by the
    sampling heuristic: strikeout, walk, in_play, plus everything else
    lumped into in_play (HBPs and other rare endings are rare enough
    to not meaningfully shift the distribution)."""
    counts = {"strikeout": 0, "walk": 0, "in_play": 0}
    total = 0
    for pa in pa_starts:
        o = pa["empirical_outcome"]
        total += 1
        if o == "strikeout":
            counts["strikeout"] += 1
        elif o == "walk":
            counts["walk"] += 1
        else:
            counts["in_play"] += 1
    if total == 0:
        return {"strikeout": 0.0, "walk": 0.0, "in_play": 0.0}
    return {k: v / total for k, v in counts.items()}


# ═════════════════════════════════════════════════════════════════════════════
# Bootstrap CI on a metric function
# ═════════════════════════════════════════════════════════════════════════════

def bootstrap_ci_on_metric(
    metric_fn,                  # callable(samples_pg, samples_lstm, pa_starts) -> (pg_val, lstm_val, diff)
    samples_pg: list[list[list[int]]],
    samples_lstm: list[list[list[int]]],
    pa_starts: list[dict],
    n_boot: int,
    seed: int,
) -> dict:
    """Resample PAs with replacement and recompute a metric.

    Returns dict with pg_point/lstm_point/diff_point plus CIs.
    """
    rng = np.random.default_rng(seed)
    n_pa = len(pa_starts)

    pg_vals, lstm_vals, diff_vals = [], [], []
    # Point estimate on the full dataset:
    pg_full, lstm_full, diff_full = metric_fn(samples_pg, samples_lstm, pa_starts)

    # Pre-index lookups for speed.
    for b in range(n_boot):
        idx = rng.integers(0, n_pa, size=n_pa)
        sub_pg = [samples_pg[i] for i in idx]
        sub_lstm = [samples_lstm[i] for i in idx]
        sub_pa = [pa_starts[i] for i in idx]
        pg_v, lstm_v, diff_v = metric_fn(sub_pg, sub_lstm, sub_pa)
        pg_vals.append(pg_v)
        lstm_vals.append(lstm_v)
        diff_vals.append(diff_v)

    def _ci(arr):
        arr = np.asarray(arr, dtype=np.float64)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return round(float(lo), 4), round(float(hi), 4)

    pg_lo, pg_hi = _ci(pg_vals)
    lstm_lo, lstm_hi = _ci(lstm_vals)
    diff_lo, diff_hi = _ci(diff_vals)
    return {
        "pg_point": round(float(pg_full), 4),
        "pg_ci95_lo": pg_lo,
        "pg_ci95_hi": pg_hi,
        "lstm_point": round(float(lstm_full), 4),
        "lstm_ci95_lo": lstm_lo,
        "lstm_ci95_hi": lstm_hi,
        "diff_point_pg_minus_lstm": round(float(diff_full), 4),
        "diff_ci95_lo": diff_lo,
        "diff_ci95_hi": diff_hi,
        "n_bootstrap": n_boot,
    }


# ─── Fast-bootstrap path: pre-decode once, then resample indices ─────────────

def _decode_per_pa_tokens(
    samples: list[list[list[int]]],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """For each PA, pre-decode its sampled tokens to (pt, zone, velo_bucket)
    flat arrays across all samples for that PA.  Returns three parallel
    lists of length n_pa, each entry shape (n_samples*horizon,).
    """
    pt_by_pa: list[np.ndarray] = []
    z_by_pa: list[np.ndarray] = []
    v_by_pa: list[np.ndarray] = []
    for pa_samples in samples:
        pts, zs, vs = [], [], []
        for seq in pa_samples:
            for tok in seq:
                if 0 <= tok < VOCAB_SIZE:
                    d = PitchTokenizer.decode(tok)
                    pts.append(PITCH_TYPE_MAP.get(d["pitch_type"], NUM_PITCH_TYPES - 1))
                    zs.append(int(d["zone"]))
                    vs.append(int(d["velo_bucket"]))
        pt_by_pa.append(np.asarray(pts, dtype=np.int32))
        z_by_pa.append(np.asarray(zs, dtype=np.int32))
        v_by_pa.append(np.asarray(vs, dtype=np.int32))
    return pt_by_pa, z_by_pa, v_by_pa


def _transition_counts_per_pa(
    samples: list[list[list[int]]],
) -> list[np.ndarray]:
    """For each PA, return a (NUM_PITCH_TYPES x NUM_PITCH_TYPES) raw count
    matrix of pt->pt transitions across all sampled sequences for that PA."""
    counts_by_pa: list[np.ndarray] = []
    for pa_samples in samples:
        m = np.zeros((NUM_PITCH_TYPES, NUM_PITCH_TYPES), dtype=np.float64)
        for seq in pa_samples:
            prev_pt = None
            for tok in seq:
                if tok < 0 or tok >= VOCAB_SIZE:
                    prev_pt = None
                    continue
                d = PitchTokenizer.decode(tok)
                cur_pt = PITCH_TYPE_MAP.get(d["pitch_type"], NUM_PITCH_TYPES - 1)
                if prev_pt is not None:
                    m[prev_pt, cur_pt] += 1
                prev_pt = cur_pt
        counts_by_pa.append(m)
    return counts_by_pa


def _outcome_counts_per_pa(
    samples: list[list[list[int]]],
    seed: int = 12345,
) -> list[np.ndarray]:
    """For each PA, return a 3-vector of (strikeout, walk, in_play)
    counts across its sampled sequences, using the empirical-calibrated
    outcome heuristic (match :func:`implied_outcome_distribution_from_samples`).
    """
    rng = np.random.default_rng(seed)
    in_zone_rates = np.array([0.22, 0.60, 0.18])  # ball, strike, in_play
    out_zone_rates = np.array([0.72, 0.16, 0.12])
    out: list[np.ndarray] = []
    for pa_samples in samples:
        vec = np.zeros(3, dtype=np.float64)
        for seq in pa_samples:
            balls, strikes = 0, 0
            cls = None  # 0=K, 1=BB, 2=IP
            for tok in seq:
                if tok < 0 or tok >= VOCAB_SIZE:
                    continue
                zone = PitchTokenizer.decode(tok)["zone"]
                rates = in_zone_rates if 0 <= zone <= 24 else out_zone_rates
                draw = rng.choice(3, p=rates)
                if draw == 2:  # in_play
                    cls = 2
                    break
                if draw == 1:  # strike
                    strikes += 1
                    if strikes >= 3:
                        cls = 0
                        break
                else:  # ball
                    balls += 1
                    if balls >= 4:
                        cls = 1
                        break
            if cls is None:
                cls = 2  # hit horizon → in_play
            vec[cls] += 1
        out.append(vec)
    return out


def _empirical_flat_by_pa(
    pa_starts: list[dict],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Pre-decode empirical per-PA arrays.  Returns (pt, zone, velo_mph, outcome_vec)."""
    pt, z, v, outc = [], [], [], []
    for pa in pa_starts:
        pt.append(np.asarray([
            PITCH_TYPE_MAP.get(s, NUM_PITCH_TYPES - 1) for s in pa["empirical_pitch_types"]
        ], dtype=np.int32))
        z.append(np.asarray(pa["empirical_zones"], dtype=np.int32))
        v.append(np.asarray(pa["empirical_velos"], dtype=np.float64))
        # Empirical outcome vector (same 3-class mapping as sampled)
        o = pa["empirical_outcome"]
        if o == "strikeout":
            outc_vec = np.array([1, 0, 0], dtype=np.float64)
        elif o == "walk":
            outc_vec = np.array([0, 1, 0], dtype=np.float64)
        else:
            outc_vec = np.array([0, 0, 1], dtype=np.float64)
        outc.append(outc_vec)
    return pt, z, v, outc


def _empirical_transition_counts_per_pa(
    pa_starts: list[dict],
) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for pa in pa_starts:
        m = np.zeros((NUM_PITCH_TYPES, NUM_PITCH_TYPES), dtype=np.float64)
        pts = [PITCH_TYPE_MAP.get(p, NUM_PITCH_TYPES - 1) for p in pa["empirical_pitch_types"]]
        for a, b in zip(pts[:-1], pts[1:]):
            m[a, b] += 1
        out.append(m)
    return out


def _fast_bootstrap_all_metrics(
    pg_pt, pg_z, pg_v,
    lstm_pt, lstm_z, lstm_v,
    emp_pt, emp_z, emp_v, emp_outc,
    pg_trans, lstm_trans, emp_trans,
    pg_outc, lstm_outc,
    n_boot: int,
    seed: int,
) -> dict[str, dict]:
    """Bootstrap all five metrics at once using pre-decoded per-PA arrays.

    Optimisation: pre-bin counts per PA so each bootstrap iteration is a
    cheap axis-0 sum (O(n_pa * n_bins)) rather than a per-pitch scan.
    """
    rng = np.random.default_rng(seed)
    n_pa = len(emp_pt)

    alpha = 1e-6

    # Pre-compute per-PA histograms (counts only, unnormalised).
    pg_pt_hist = np.zeros((n_pa, NUM_PITCH_TYPES), dtype=np.float64)
    lstm_pt_hist = np.zeros((n_pa, NUM_PITCH_TYPES), dtype=np.float64)
    emp_pt_hist = np.zeros((n_pa, NUM_PITCH_TYPES), dtype=np.float64)
    for i in range(n_pa):
        for v in pg_pt[i]:
            if 0 <= int(v) < NUM_PITCH_TYPES:
                pg_pt_hist[i, int(v)] += 1
        for v in lstm_pt[i]:
            if 0 <= int(v) < NUM_PITCH_TYPES:
                lstm_pt_hist[i, int(v)] += 1
        for v in emp_pt[i]:
            if 0 <= int(v) < NUM_PITCH_TYPES:
                emp_pt_hist[i, int(v)] += 1

    pg_z_hist = np.zeros((n_pa, NUM_ZONES), dtype=np.float64)
    lstm_z_hist = np.zeros((n_pa, NUM_ZONES), dtype=np.float64)
    emp_z_hist = np.zeros((n_pa, NUM_ZONES), dtype=np.float64)
    for i in range(n_pa):
        for v in pg_z[i]:
            if 0 <= int(v) < NUM_ZONES:
                pg_z_hist[i, int(v)] += 1
        for v in lstm_z[i]:
            if 0 <= int(v) < NUM_ZONES:
                lstm_z_hist[i, int(v)] += 1
        for v in emp_z[i]:
            if 0 <= int(v) < NUM_ZONES:
                emp_z_hist[i, int(v)] += 1

    # Pre-stack transition count matrices into 3D arrays.
    pg_trans_stack = np.stack(pg_trans, axis=0)   # (n_pa, PT, PT)
    lstm_trans_stack = np.stack(lstm_trans, axis=0)
    emp_trans_stack = np.stack(emp_trans, axis=0)

    # Pre-stack outcome vectors.
    pg_outc_stack = np.stack(pg_outc, axis=0)     # (n_pa, 3)
    lstm_outc_stack = np.stack(lstm_outc, axis=0)
    emp_outc_stack = np.stack(emp_outc, axis=0)

    # Per-PA velocity arrays (left as ragged lists for Wasserstein).
    vel_pg_arrs = [np.array([velocity_bucket_to_midpoint(b) for b in a]) for a in pg_v]
    vel_lstm_arrs = [np.array([velocity_bucket_to_midpoint(b) for b in a]) for a in lstm_v]

    def _kl_from_hists(mod_hist: np.ndarray, emp_hist: np.ndarray) -> float:
        # Aggregate then normalise with alpha smoothing.
        p_mod = mod_hist + alpha
        p_emp = emp_hist + alpha
        p_mod = p_mod / p_mod.sum()
        p_emp = p_emp / p_emp.sum()
        return float(np.sum(p_emp * np.log(p_emp / np.clip(p_mod, 1e-12, None))))

    def _frob_from_stacks(mod_stack: np.ndarray, emp_stack: np.ndarray) -> float:
        M = mod_stack.sum(axis=0) + alpha
        E = emp_stack.sum(axis=0) + alpha
        M = M / np.clip(M.sum(axis=1, keepdims=True), 1e-12, None)
        E = E / np.clip(E.sum(axis=1, keepdims=True), 1e-12, None)
        return float(np.linalg.norm(M - E, ord="fro"))

    def _chi2_from_stacks(mod_stack: np.ndarray, emp_stack: np.ndarray) -> float:
        M = mod_stack.sum(axis=0)
        E = emp_stack.sum(axis=0)
        p_mod = M / max(M.sum(), 1e-12)
        p_emp = E / max(E.sum(), 1e-12)
        p_emp = np.clip(p_emp, 1e-6, None)
        p_emp = p_emp / p_emp.sum()
        return float(np.sum((p_mod - p_emp) ** 2 / p_emp))

    def _wass_on_indices(idx: np.ndarray, mod_arrs: list[np.ndarray], emp_arrs: list[np.ndarray]) -> float:
        mod_flat = np.concatenate([mod_arrs[i] for i in idx]) if len(idx) else np.array([])
        emp_flat = np.concatenate([emp_arrs[i] for i in idx]) if len(idx) else np.array([])
        if mod_flat.size == 0 or emp_flat.size == 0:
            return float("nan")
        try:
            from scipy.stats import wasserstein_distance  # type: ignore
            return float(wasserstein_distance(mod_flat, emp_flat))
        except ImportError:
            qs = np.linspace(0.005, 0.995, 200)
            return float(np.mean(np.abs(np.quantile(mod_flat, qs) - np.quantile(emp_flat, qs))))

    # Point estimates on full dataset (sum across all PAs).
    point = {
        "pitch_type_kl_pg": _kl_from_hists(pg_pt_hist.sum(axis=0), emp_pt_hist.sum(axis=0)),
        "pitch_type_kl_lstm": _kl_from_hists(lstm_pt_hist.sum(axis=0), emp_pt_hist.sum(axis=0)),
        "zone_kl_pg": _kl_from_hists(pg_z_hist.sum(axis=0), emp_z_hist.sum(axis=0)),
        "zone_kl_lstm": _kl_from_hists(lstm_z_hist.sum(axis=0), emp_z_hist.sum(axis=0)),
        "velo_w_pg": _wass_on_indices(np.arange(n_pa), vel_pg_arrs, emp_v),
        "velo_w_lstm": _wass_on_indices(np.arange(n_pa), vel_lstm_arrs, emp_v),
        "trans_f_pg": _frob_from_stacks(pg_trans_stack, emp_trans_stack),
        "trans_f_lstm": _frob_from_stacks(lstm_trans_stack, emp_trans_stack),
        "outcome_chi2_pg": _chi2_from_stacks(pg_outc_stack, emp_outc_stack),
        "outcome_chi2_lstm": _chi2_from_stacks(lstm_outc_stack, emp_outc_stack),
    }

    # Bootstrap loop (fast path uses pre-stacked 3D arrays + indexing).
    boot_store = {k: np.empty(n_boot, dtype=np.float64) for k in point.keys()}
    for b in range(n_boot):
        idx = rng.integers(0, n_pa, size=n_pa)
        # KL — sum over sampled PAs.
        boot_store["pitch_type_kl_pg"][b] = _kl_from_hists(pg_pt_hist[idx].sum(axis=0), emp_pt_hist[idx].sum(axis=0))
        boot_store["pitch_type_kl_lstm"][b] = _kl_from_hists(lstm_pt_hist[idx].sum(axis=0), emp_pt_hist[idx].sum(axis=0))
        boot_store["zone_kl_pg"][b] = _kl_from_hists(pg_z_hist[idx].sum(axis=0), emp_z_hist[idx].sum(axis=0))
        boot_store["zone_kl_lstm"][b] = _kl_from_hists(lstm_z_hist[idx].sum(axis=0), emp_z_hist[idx].sum(axis=0))
        # Frobenius on transition matrices.
        boot_store["trans_f_pg"][b] = _frob_from_stacks(pg_trans_stack[idx], emp_trans_stack[idx])
        boot_store["trans_f_lstm"][b] = _frob_from_stacks(lstm_trans_stack[idx], emp_trans_stack[idx])
        # Chi-square on outcome.
        boot_store["outcome_chi2_pg"][b] = _chi2_from_stacks(pg_outc_stack[idx], emp_outc_stack[idx])
        boot_store["outcome_chi2_lstm"][b] = _chi2_from_stacks(lstm_outc_stack[idx], emp_outc_stack[idx])
        # Wasserstein (still O(n_pa) per metric; dominating cost).
        boot_store["velo_w_pg"][b] = _wass_on_indices(idx, vel_pg_arrs, emp_v)
        boot_store["velo_w_lstm"][b] = _wass_on_indices(idx, vel_lstm_arrs, emp_v)

    def _ci(arr: np.ndarray) -> tuple[float, float]:
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return round(float(lo), 4), round(float(hi), 4)

    def _pack(name: str, pg_key: str, lstm_key: str) -> dict:
        pg_full = point[pg_key]
        lstm_full = point[lstm_key]
        diff_full = pg_full - lstm_full
        pg_boot = boot_store[pg_key]
        lstm_boot = boot_store[lstm_key]
        diff_boot = pg_boot - lstm_boot
        pg_lo, pg_hi = _ci(pg_boot)
        lstm_lo, lstm_hi = _ci(lstm_boot)
        diff_lo, diff_hi = _ci(diff_boot)
        return {
            "pg_point": round(float(pg_full), 4),
            "pg_ci95_lo": pg_lo,
            "pg_ci95_hi": pg_hi,
            "lstm_point": round(float(lstm_full), 4),
            "lstm_ci95_lo": lstm_lo,
            "lstm_ci95_hi": lstm_hi,
            "diff_point_pg_minus_lstm": round(float(diff_full), 4),
            "diff_ci95_lo": diff_lo,
            "diff_ci95_hi": diff_hi,
            "n_bootstrap": n_boot,
        }

    return {
        "pitch_type_kl": _pack("pitch_type_kl", "pitch_type_kl_pg", "pitch_type_kl_lstm"),
        "zone_kl": _pack("zone_kl", "zone_kl_pg", "zone_kl_lstm"),
        "velocity_wasserstein": _pack("velocity_wasserstein", "velo_w_pg", "velo_w_lstm"),
        "transition_frobenius": _pack("transition_frobenius", "trans_f_pg", "trans_f_lstm"),
        "outcome_chi2": _pack("outcome_chi2", "outcome_chi2_pg", "outcome_chi2_lstm"),
    }


# ─── Metric factories ────────────────────────────────────────────────────────

def metric_pitch_type_kl(samples_pg, samples_lstm, pa_starts):
    pg_pt, _, _ = decode_samples_to_flat_arrays(samples_pg)
    lstm_pt, _, _ = decode_samples_to_flat_arrays(samples_lstm)
    emp_pt, _, _ = decode_empirical_to_flat_arrays(pa_starts)
    p_pg = dist_histogram(pg_pt, NUM_PITCH_TYPES)
    p_lstm = dist_histogram(lstm_pt, NUM_PITCH_TYPES)
    p_emp = dist_histogram(emp_pt, NUM_PITCH_TYPES)
    kl_pg = kl_divergence(p_emp, p_pg)  # KL(empirical || model) — how bad is model at covering empirical
    kl_lstm = kl_divergence(p_emp, p_lstm)
    return kl_pg, kl_lstm, kl_pg - kl_lstm  # negative diff = PitchGPT closer to empirical


def metric_zone_kl(samples_pg, samples_lstm, pa_starts):
    _, pg_z, _ = decode_samples_to_flat_arrays(samples_pg)
    _, lstm_z, _ = decode_samples_to_flat_arrays(samples_lstm)
    _, emp_z, _ = decode_empirical_to_flat_arrays(pa_starts)
    p_pg = dist_histogram(pg_z, NUM_ZONES)
    p_lstm = dist_histogram(lstm_z, NUM_ZONES)
    p_emp = dist_histogram(emp_z, NUM_ZONES)
    kl_pg = kl_divergence(p_emp, p_pg)
    kl_lstm = kl_divergence(p_emp, p_lstm)
    return kl_pg, kl_lstm, kl_pg - kl_lstm


def metric_velocity_wasserstein(samples_pg, samples_lstm, pa_starts):
    _, _, pg_vb = decode_samples_to_flat_arrays(samples_pg)
    _, _, lstm_vb = decode_samples_to_flat_arrays(samples_lstm)
    pg_v = np.array([velocity_bucket_to_midpoint(b) for b in pg_vb])
    lstm_v = np.array([velocity_bucket_to_midpoint(b) for b in lstm_vb])
    emp_v = np.concatenate([np.asarray(pa["empirical_velos"]) for pa in pa_starts])
    w_pg = wasserstein_1d(pg_v, emp_v)
    w_lstm = wasserstein_1d(lstm_v, emp_v)
    return w_pg, w_lstm, w_pg - w_lstm


def metric_transition_frobenius(samples_pg, samples_lstm, pa_starts):
    pg_seqs = sample_sequences_by_pa(samples_pg)
    lstm_seqs = sample_sequences_by_pa(samples_lstm)
    T_pg = transition_matrix_pt(pg_seqs)
    T_lstm = transition_matrix_pt(lstm_seqs)
    T_emp = transition_matrix_from_empirical(pa_starts)
    f_pg = float(np.linalg.norm(T_pg - T_emp, ord="fro"))
    f_lstm = float(np.linalg.norm(T_lstm - T_emp, ord="fro"))
    return f_pg, f_lstm, f_pg - f_lstm


def metric_outcome_chi2(samples_pg, samples_lstm, pa_starts):
    p_pg = implied_outcome_distribution_from_samples(samples_pg)
    p_lstm = implied_outcome_distribution_from_samples(samples_lstm)
    p_emp = empirical_outcome_distribution(pa_starts)
    cats = ["strikeout", "walk", "in_play"]
    obs_pg = np.array([p_pg[c] for c in cats])
    obs_lstm = np.array([p_lstm[c] for c in cats])
    emp = np.array([p_emp[c] for c in cats])
    emp = np.clip(emp, 1e-6, None)
    emp = emp / emp.sum()
    chi2_pg = float(np.sum((obs_pg - emp) ** 2 / emp))
    chi2_lstm = float(np.sum((obs_lstm - emp) ** 2 / emp))
    return chi2_pg, chi2_lstm, chi2_pg - chi2_lstm


# Chi-square p-values (for reporting — we flag the MC issue in the report)
def chi2_pvalue(chi2: float, dof: int) -> float:
    try:
        from scipy.stats import chi2 as chi2_dist  # type: ignore
        return float(1.0 - chi2_dist.cdf(chi2, dof))
    except Exception:
        return float("nan")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pitchgpt-checkpoint",
        type=Path,
        default=_ROOT / "models" / "pitchgpt_v2.pt",
    )
    parser.add_argument(
        "--lstm-checkpoint",
        type=Path,
        default=_ROOT / "models" / "pitch_lstm_10k.pt",
    )
    parser.add_argument(
        "--train-lstm",
        action="store_true",
        help="Force (re)training of the LSTM checkpoint even if present.",
    )
    parser.add_argument("--max-train-games", type=int, default=10000)
    parser.add_argument("--max-val-games", type=int, default=1000)
    parser.add_argument("--n-pas", type=int, default=2000)
    parser.add_argument("--n-samples-per-pa", type=int, default=N_SAMPLES_PER_PA)
    parser.add_argument("--horizon", type=int, default=SAMPLE_HORIZON)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "results" / "pitchgpt" / "sampling_fidelity_2026_04_24",
    )
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument(
        "--save-sequences",
        action="store_true",
        help="Write sampled sequences to CSV for reproducibility.",
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip bootstrap CI computation (faster; point estimates only).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)
    device = _get_device()
    logger.info("device=%s  seed=%d  output=%s", device, args.seed, args.output_dir)

    # ── 1. Load PitchGPT checkpoint + derive context_dim.
    pg = load_pitchgpt_checkpoint(args.pitchgpt_checkpoint, device)
    context_dim = int(pg.context_dim)
    logger.info(
        "PitchGPT loaded: %d params, context_dim=%d",
        sum(p.numel() for p in pg.parameters()), context_dim,
    )

    # ── 2. Build the same pitcher-disjoint 10K train/val split the
    #      holdout used, so the LSTM sees exactly the matched training set.
    conn = get_connection(args.db_path, read_only=True)
    try:
        logger.info("Building pitcher-disjoint train (10K) / val (1K) for LSTM retrain + PA sampling")
        train_pitchers_all = PitchSequenceDataset.fetch_pitcher_ids_for_seasons(
            conn, range(TRAIN_RANGE[0], TRAIN_RANGE[1] + 1),
        )
        logger.info("Train cohort: %d pitchers (2015-2022 full)", len(train_pitchers_all))

        lstm_path = args.lstm_checkpoint
        if args.train_lstm or not lstm_path.exists():
            # Train LSTM fresh with matching seed/split.
            t_build = time.perf_counter()
            train_ds = PitchSequenceDataset(
                conn,
                split_mode="train",
                train_range=TRAIN_RANGE,
                val_range=VAL_RANGE,
                test_range=HOLDOUT_RANGE,
                max_games_per_split=args.max_train_games,
                context_dim=context_dim,
            )
            val_ds = PitchSequenceDataset(
                conn,
                split_mode="val",
                train_range=TRAIN_RANGE,
                val_range=VAL_RANGE,
                test_range=HOLDOUT_RANGE,
                max_games_per_split=args.max_val_games,
                exclude_pitcher_ids=train_pitchers_all,
                context_dim=context_dim,
            )
            logger.info(
                "Dataset build: train=%d sequences, val=%d sequences (%.1fs)",
                len(train_ds), len(val_ds), time.perf_counter() - t_build,
            )

            t_train = time.perf_counter()
            lstm, lstm_meta = train_lstm(
                train_ds, val_ds, device,
                epochs=args.epochs, lr=args.lr,
                batch_size=args.batch_size, seed=args.seed,
                context_dim=context_dim,
            )
            dt_train = time.perf_counter() - t_train

            # Persist so we don't retrain next time.
            lstm_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": lstm.state_dict(),
                    "config": {
                        "d_model": lstm.d_model,
                        "num_layers": lstm.num_layers,
                        "max_seq_len": 256,
                        "vocab_size": TOTAL_VOCAB,
                        "context_dim": context_dim,
                        "output_vocab": lstm.output_vocab,
                    },
                    "version": "sampling_fidelity_10k",
                    "training_meta": {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "seed": args.seed,
                        "max_train_games": args.max_train_games,
                        "max_val_games": args.max_val_games,
                        "wall_clock_sec": round(dt_train, 1),
                        **lstm_meta,
                    },
                },
                lstm_path,
            )
            logger.info("Saved LSTM to %s  (train wall-clock %.1fs)", lstm_path, dt_train)
        else:
            logger.info("Loading LSTM from %s (already exists)", lstm_path)
            lstm = load_lstm_checkpoint(lstm_path, device)
            lstm_meta = {"params": sum(p.numel() for p in lstm.parameters())}

        # ── 3. Draw PA-start contexts from 2025 pitcher-disjoint pool.
        t_pas = time.perf_counter()
        pa_starts, pa_df = fetch_2025_pa_starts(
            conn,
            n_pas=args.n_pas,
            exclude_pitcher_ids=train_pitchers_all,
            seed=args.seed,
            context_dim=context_dim,
        )
        dt_pas = time.perf_counter() - t_pas
        logger.info("Drew %d PA starts in %.1fs", len(pa_starts), dt_pas)
    finally:
        conn.close()

    if not pa_starts:
        logger.error("No PA starts available — aborting.")
        return 1

    # ── 4. Sample continuations from each model.
    t_pg = time.perf_counter()
    samples_pg = sample_continuations_pitchgpt(
        pg, pa_starts,
        n_samples=args.n_samples_per_pa,
        horizon=args.horizon,
        temperature=args.temperature,
        device=device,
        seed=args.seed,
    )
    dt_pg = time.perf_counter() - t_pg
    logger.info("PitchGPT sampling done in %.1fs", dt_pg)

    t_ls = time.perf_counter()
    samples_lstm = sample_continuations_lstm(
        lstm, pa_starts,
        n_samples=args.n_samples_per_pa,
        horizon=args.horizon,
        temperature=args.temperature,
        device=device,
        seed=args.seed + 1,
    )
    dt_ls = time.perf_counter() - t_ls
    logger.info("LSTM sampling done in %.1fs", dt_ls)

    # ── 5. Compute metrics with bootstrap CIs.  Fast path: pre-decode
    # samples once per PA, then resample PA indices and recompute from
    # the pre-decoded arrays.  This is ~100x faster than the naive
    # decode-per-iteration approach for n_boot=1000 on 2000 PAs.
    t_m = time.perf_counter()
    logger.info("Pre-decoding sampled tokens for fast bootstrap...")
    pg_pt_pa, pg_z_pa, pg_v_pa = _decode_per_pa_tokens(samples_pg)
    lstm_pt_pa, lstm_z_pa, lstm_v_pa = _decode_per_pa_tokens(samples_lstm)
    emp_pt_pa, emp_z_pa, emp_v_pa, emp_outc_pa = _empirical_flat_by_pa(pa_starts)
    pg_trans_pa = _transition_counts_per_pa(samples_pg)
    lstm_trans_pa = _transition_counts_per_pa(samples_lstm)
    emp_trans_pa = _empirical_transition_counts_per_pa(pa_starts)
    pg_outc_pa = _outcome_counts_per_pa(samples_pg)
    lstm_outc_pa = _outcome_counts_per_pa(samples_lstm)

    if args.skip_bootstrap:
        metric_fns = {
            "pitch_type_kl": metric_pitch_type_kl,
            "zone_kl": metric_zone_kl,
            "velocity_wasserstein": metric_velocity_wasserstein,
            "transition_frobenius": metric_transition_frobenius,
            "outcome_chi2": metric_outcome_chi2,
        }
        metrics: dict[str, dict] = {}
        for name, fn in metric_fns.items():
            pg_v, lstm_v, diff_v = fn(samples_pg, samples_lstm, pa_starts)
            metrics[name] = {
                "pg_point": round(float(pg_v), 4),
                "lstm_point": round(float(lstm_v), 4),
                "diff_point_pg_minus_lstm": round(float(diff_v), 4),
                "n_bootstrap": 0,
            }
    else:
        logger.info("Running %d bootstrap iterations for 5 metrics...", args.n_bootstrap)
        metrics = _fast_bootstrap_all_metrics(
            pg_pt_pa, pg_z_pa, pg_v_pa,
            lstm_pt_pa, lstm_z_pa, lstm_v_pa,
            emp_pt_pa, emp_z_pa, emp_v_pa, emp_outc_pa,
            pg_trans_pa, lstm_trans_pa, emp_trans_pa,
            pg_outc_pa, lstm_outc_pa,
            n_boot=args.n_bootstrap,
            seed=args.seed + 7,
        )
    dt_m = time.perf_counter() - t_m

    # Additional one-shot diagnostics (not bootstrapped; for context).
    pg_pt_flat, pg_z_flat, _ = decode_samples_to_flat_arrays(samples_pg)
    lstm_pt_flat, lstm_z_flat, _ = decode_samples_to_flat_arrays(samples_lstm)
    emp_pt_flat, emp_z_flat, emp_v_cont = decode_empirical_to_flat_arrays(pa_starts)

    emp_pt_dist = dist_histogram(emp_pt_flat, NUM_PITCH_TYPES).tolist()
    pg_pt_dist = dist_histogram(pg_pt_flat, NUM_PITCH_TYPES).tolist()
    lstm_pt_dist = dist_histogram(lstm_pt_flat, NUM_PITCH_TYPES).tolist()

    pg_pt_counts = np.zeros(NUM_PITCH_TYPES, dtype=np.float64)
    for v in pg_pt_flat:
        if 0 <= int(v) < NUM_PITCH_TYPES:
            pg_pt_counts[int(v)] += 1
    lstm_pt_counts = np.zeros(NUM_PITCH_TYPES, dtype=np.float64)
    for v in lstm_pt_flat:
        if 0 <= int(v) < NUM_PITCH_TYPES:
            lstm_pt_counts[int(v)] += 1

    emp_pt_probs = dist_histogram(emp_pt_flat, NUM_PITCH_TYPES)
    chi2_pg_pt = chi_square_stat(pg_pt_counts, emp_pt_probs)
    chi2_lstm_pt = chi_square_stat(lstm_pt_counts, emp_pt_probs)

    emp_z_counts = dist_histogram(emp_z_flat, NUM_ZONES)
    pg_z_counts = np.zeros(NUM_ZONES, dtype=np.float64)
    for v in pg_z_flat:
        if 0 <= int(v) < NUM_ZONES:
            pg_z_counts[int(v)] += 1
    lstm_z_counts = np.zeros(NUM_ZONES, dtype=np.float64)
    for v in lstm_z_flat:
        if 0 <= int(v) < NUM_ZONES:
            lstm_z_counts[int(v)] += 1
    chi2_pg_z = chi_square_stat(pg_z_counts, emp_z_counts)
    chi2_lstm_z = chi_square_stat(lstm_z_counts, emp_z_counts)

    # Outcome distributions (one-shot point values).
    outcome_pg = implied_outcome_distribution_from_samples(samples_pg)
    outcome_lstm = implied_outcome_distribution_from_samples(samples_lstm)
    outcome_emp = empirical_outcome_distribution(pa_starts)

    # ── 6. Calibration-by-context (bonus).  Stratify PA-starts by count,
    #      leverage, inning and compute per-stratum pitch-type KL for
    #      each model.  Cheap; point estimates only to keep within budget.
    stratified = {}
    for strat_key in ("count", "leverage", "inning_bucket"):
        stratified[strat_key] = {}
        if strat_key == "inning_bucket":
            def keyfn(pa):
                i = pa["inning"]
                if i <= 3:
                    return "early(1-3)"
                if i <= 6:
                    return "mid(4-6)"
                if i <= 9:
                    return "late(7-9)"
                return "extra(10+)"
        else:
            def keyfn(pa, k=strat_key):
                return pa[k]
        # Group PA indices by stratum.
        groups: dict[str, list[int]] = {}
        for i, pa in enumerate(pa_starts):
            groups.setdefault(keyfn(pa), []).append(i)
        for stratum_val, idxs in groups.items():
            if len(idxs) < 5:
                continue
            sub_pg = [samples_pg[i] for i in idxs]
            sub_lstm = [samples_lstm[i] for i in idxs]
            sub_pa = [pa_starts[i] for i in idxs]
            kl_pg, kl_lstm, diff = metric_pitch_type_kl(sub_pg, sub_lstm, sub_pa)
            stratified[strat_key][stratum_val] = {
                "n_pas": len(idxs),
                "pg_kl": round(float(kl_pg), 4),
                "lstm_kl": round(float(kl_lstm), 4),
                "diff_pg_minus_lstm": round(float(diff), 4),
            }

    # ── 7. Write artifacts.
    output_payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": args.seed,
        "device": str(device),
        "pitchgpt_checkpoint": str(args.pitchgpt_checkpoint),
        "lstm_checkpoint": str(args.lstm_checkpoint),
        "context_dim": context_dim,
        "config": {
            "n_pas_requested": args.n_pas,
            "n_pas_drawn": len(pa_starts),
            "n_samples_per_pa": args.n_samples_per_pa,
            "horizon": args.horizon,
            "temperature": args.temperature,
            "max_train_games_lstm": args.max_train_games,
            "epochs_lstm": args.epochs,
        },
        "metrics": metrics,
        "diagnostics": {
            "chi_square": {
                "pitch_type_pg": round(chi2_pg_pt, 3),
                "pitch_type_lstm": round(chi2_lstm_pt, 3),
                "zone_pg": round(chi2_pg_z, 3),
                "zone_lstm": round(chi2_lstm_z, 3),
                "pitch_type_p_pg_uncorrected": round(chi2_pvalue(chi2_pg_pt, NUM_PITCH_TYPES - 1), 4),
                "pitch_type_p_lstm_uncorrected": round(chi2_pvalue(chi2_lstm_pt, NUM_PITCH_TYPES - 1), 4),
                "zone_p_pg_uncorrected": round(chi2_pvalue(chi2_pg_z, NUM_ZONES - 1), 4),
                "zone_p_lstm_uncorrected": round(chi2_pvalue(chi2_lstm_z, NUM_ZONES - 1), 4),
                "note": "Chi-square stat with expected counts from empirical distribution; p-values uncorrected for multiple tests.",
            },
            "pitch_type_distribution": {
                "labels": [k for k, _ in sorted(PITCH_TYPE_MAP.items(), key=lambda x: x[1])] + ["UN"],
                "empirical": [round(x, 4) for x in emp_pt_dist],
                "pitchgpt": [round(x, 4) for x in pg_pt_dist],
                "lstm": [round(x, 4) for x in lstm_pt_dist],
            },
            "outcome_distribution": {
                "empirical": {k: round(v, 4) for k, v in outcome_emp.items()},
                "pitchgpt": {k: round(v, 4) for k, v in outcome_pg.items()},
                "lstm": {k: round(v, 4) for k, v in outcome_lstm.items()},
            },
        },
        "stratified_kl_by_context": stratified,
        "wall_clock": {
            "pg_sampling_sec": round(dt_pg, 1),
            "lstm_sampling_sec": round(dt_ls, 1),
            "pa_fetch_sec": round(dt_pas, 1),
            "metric_computation_sec": round(dt_m, 1),
        },
        "lstm_meta": lstm_meta if isinstance(lstm_meta, dict) else {},
    }

    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    logger.info("Wrote %s", metrics_path)

    # Write reviewer-grade markdown report.
    report_path = args.output_dir / "report.md"
    _write_report(report_path, output_payload)
    logger.info("Wrote %s", report_path)

    if args.save_sequences:
        _write_sampled_sequences_csv(
            args.output_dir / "sampled_sequences_pg.csv", samples_pg, pa_starts, "pitchgpt",
        )
        _write_sampled_sequences_csv(
            args.output_dir / "sampled_sequences_lstm.csv", samples_lstm, pa_starts, "lstm",
        )

    # ── 8. Print summary.  Use ASCII-only text for Windows cp1252 consoles.
    print("\n" + "=" * 78)
    print("PitchGPT Sampling-Fidelity vs LSTM (10K) -- 2025 holdout")
    print("=" * 78)
    print(f"n_pas = {len(pa_starts)}  n_samples/pa = {args.n_samples_per_pa}  horizon = {args.horizon}")
    print(f"seed = {args.seed}  temp = {args.temperature}  context_dim = {context_dim}")
    print("-" * 78)
    print(f"{'Metric':30s} {'PitchGPT':>12s} {'LSTM':>12s} {'delta (PG-LSTM)':>18s} {'CI':>18s}")
    for name, m in metrics.items():
        ci = f"[{m.get('diff_ci95_lo','?')}, {m.get('diff_ci95_hi','?')}]" if "diff_ci95_lo" in m else "-"
        print(f"{name:30s} {m['pg_point']:>12.4f} {m['lstm_point']:>12.4f} "
              f"{m['diff_point_pg_minus_lstm']:>+18.4f} {ci:>18s}")
    print("-" * 78)
    print("NOTE: smaller distance = closer to empirical 2025.  delta<0 means PitchGPT wins.")
    print("=" * 78)
    return 0


# ═════════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═════════════════════════════════════════════════════════════════════════════

def _write_sampled_sequences_csv(path: Path, samples, pa_starts, model_name: str) -> None:
    rows = []
    for pa_idx, pa in enumerate(pa_starts):
        pa_samples = samples[pa_idx]
        for samp_idx, seq in enumerate(pa_samples):
            for step, tok in enumerate(seq):
                d = PitchTokenizer.decode(tok)
                rows.append({
                    "model": model_name,
                    "pa_idx": pa_idx,
                    "game_pk": pa["game_pk"],
                    "at_bat_number": pa["at_bat_number"],
                    "pitcher_id": pa["pitcher_id"],
                    "batter_id": pa["batter_id"],
                    "sample_idx": samp_idx,
                    "step": step,
                    "token": int(tok),
                    "pitch_type": d["pitch_type"],
                    "zone": d["zone"],
                    "velo_bucket": d["velo_bucket"],
                })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_report(path: Path, payload: dict) -> None:
    lines: list[str] = []
    lines.append("# PitchGPT Sampling-Fidelity vs LSTM — 2025 Holdout\n")
    lines.append(f"Generated: {payload['timestamp_utc']}\n")
    lines.append(f"PitchGPT checkpoint: `{payload['pitchgpt_checkpoint']}`")
    lines.append(f"LSTM checkpoint: `{payload['lstm_checkpoint']}`")
    lines.append(f"Context schema: v2, context_dim={payload['context_dim']}  (umpire scalar present)")
    lines.append("")
    lines.append(f"Seed: {payload['seed']}  Device: {payload['device']}")
    lines.append("")

    cfg = payload["config"]
    lines.append("## Configuration\n")
    lines.append(f"- PAs drawn from 2025 (pitcher-disjoint cohort): **{cfg['n_pas_drawn']}** (requested {cfg['n_pas_requested']})")
    lines.append(f"- Samples per PA per model: **{cfg['n_samples_per_pa']}**")
    lines.append(f"- Fixed horizon per sample: **{cfg['horizon']}** pitches")
    lines.append(f"- Sampling temperature: **{cfg['temperature']}** (autoregressive, top-k=none)")
    lines.append(f"- LSTM training: {cfg['epochs_lstm']} epochs, max_train_games={cfg['max_train_games_lstm']} (matches v2 10K holdout split)")
    lines.append("")

    lines.append("## Hypothesis\n")
    lines.append("PitchGPT's decoder-only transformer, by virtue of its attention over the "
                 "full context plus richer contextual conditioning (count × outs × runners × "
                 "stand × inning × score × ump), generates pitch sequences whose marginal and "
                 "joint distributions are closer to empirical 2025 distributions than a matched "
                 "LSTM baseline's.  This is the first test of PitchGPT's claim as a **calibrated "
                 "simulation engine** (not a next-token-accuracy king — that claim was already "
                 "narrowed: at matched 10K scale, v2 beats LSTM by only 3.13% on holdout perplexity).")
    lines.append("")

    lines.append("## Method\n")
    lines.append("1. **Checkpoint**: v2 PitchGPT (10K games, context_dim=35, umpire scalar included) per user-specified "
                 "`models/pitchgpt_v2.pt`.")
    lines.append("2. **Matched LSTM**: 2-layer LSTM (d_model=128), trained fresh on the same 10K pitcher-disjoint "
                 "2015-2022 train split with seed=42, epochs=5, lr=1e-3, batch=32.  Context width matches PitchGPT.")
    lines.append("3. **PA starts**: Uniformly sampled from 2025 plate appearances thrown by pitchers NOT in the "
                 "2015-2022 train cohort (pitcher-disjoint).  A PA = `(game_pk, at_bat_number, pitcher_id)` group "
                 "with ≥2 pitches.  We use the real first-pitch situational context (count 0-0, outs, runners, "
                 "stand, inning, score_diff, ump tendency).")
    lines.append("4. **Sampling**: Temperature-1 multinomial sampling on model logits over the full 2210-class "
                 "token vocab (pitch_type × zone × velo_bucket).  Fixed horizon = 6 pitches (covers the 76th "
                 "percentile of 2025 PAs by pitch count).  Context updates between steps use a zone-based "
                 "ball/strike heuristic (in-zone → strike, out-of-zone → ball).")
    lines.append("5. **Reference**: Empirical 2025 per-pitch distributions over the SAME PAs' real continuations.")
    lines.append("")
    lines.append("**Limitations documented up front:**")
    lines.append("- Fixed horizon does not model PA termination; implied outcome distribution is a heuristic proxy based on "
                 "running counts, not a learned outcome head.  Real Statcast PA ends on events not observable from the "
                 "token stream alone.")
    lines.append("- Context updates hold runners/outs/inning fixed, because the models don't predict those.")
    lines.append("- The LSTM is trained to the same 5-epoch schedule as PitchGPT, not tuned further.")
    lines.append("- Chi-square p-values are uncorrected; ≥5 tests run so Bonferroni α=0.01 is the appropriate floor.")
    lines.append("")

    lines.append("## Results\n")
    lines.append("**Convention: smaller distance = closer to empirical.  Δ(PG − LSTM) < 0 means PitchGPT is closer.**\n")
    lines.append("| Metric | Description | PitchGPT | LSTM | Δ(PG − LSTM) | 95% CI on Δ |")
    lines.append("|---|---|---:|---:|---:|---|")
    m = payload["metrics"]
    meta = {
        "pitch_type_kl": ("Marginal pitch-type KL(empirical ‖ model)", "KL"),
        "zone_kl": ("Marginal zone KL (26-class, 5×5 + out-of-zone)", "KL"),
        "velocity_wasserstein": ("Velocity 1-D Wasserstein (mph)", "EMD"),
        "transition_frobenius": ("Pitch-type 2-gram transition matrix Frobenius norm", "||·||_F"),
        "outcome_chi2": ("Implied at-bat outcome χ² (K/BB/IP vs empirical)", "χ²"),
    }
    for k, (desc, _unit) in meta.items():
        row = m[k]
        ci = f"[{row.get('diff_ci95_lo', '—')}, {row.get('diff_ci95_hi', '—')}]" if "diff_ci95_lo" in row else "—"
        lines.append(
            f"| `{k}` | {desc} | {row['pg_point']:.4f} | {row['lstm_point']:.4f} | "
            f"{row['diff_point_pg_minus_lstm']:+.4f} | {ci} |"
        )
    lines.append("")

    lines.append("### Diagnostics (one-shot, no CI)\n")
    dx = payload["diagnostics"]["chi_square"]
    lines.append("| Test | PitchGPT | LSTM | PG p (uncorr.) | LSTM p (uncorr.) |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| χ² pitch-type vs empirical | {dx['pitch_type_pg']:.1f} | {dx['pitch_type_lstm']:.1f} | "
        f"{dx['pitch_type_p_pg_uncorrected']:.4f} | {dx['pitch_type_p_lstm_uncorrected']:.4f} |"
    )
    lines.append(
        f"| χ² zone vs empirical | {dx['zone_pg']:.1f} | {dx['zone_lstm']:.1f} | "
        f"{dx['zone_p_pg_uncorrected']:.4f} | {dx['zone_p_lstm_uncorrected']:.4f} |"
    )
    lines.append("")
    lines.append("_Chi² p-values are uncorrected for multiple testing.  With ~5 tests, Bonferroni α=0.01 is the floor "
                 "for significance; values below this are suggestive, not confirmatory._")
    lines.append("")

    lines.append("### Outcome-implied distribution (3-class: strikeout / walk / in_play)\n")
    od = payload["diagnostics"]["outcome_distribution"]
    lines.append("| Class | Empirical | PitchGPT | LSTM |")
    lines.append("|---|---:|---:|---:|")
    for cls in ["strikeout", "walk", "in_play"]:
        lines.append(f"| {cls} | {od['empirical'][cls]:.4f} | {od['pitchgpt'][cls]:.4f} | {od['lstm'][cls]:.4f} |")
    lines.append("")
    lines.append("_Note: sampling-generated outcomes use a zone-based heuristic (in-zone→strike, out-of-zone→ball).  "
                 "Real Statcast outcome classification uses pitch-description labels not present in the token stream._")
    lines.append("")

    lines.append("### Marginal pitch-type distribution (top 7 by empirical mass)\n")
    ptd = payload["diagnostics"]["pitch_type_distribution"]
    labels = ptd["labels"]
    emp = ptd["empirical"]
    pg_v = ptd["pitchgpt"]
    lstm_v = ptd["lstm"]
    ranked = sorted(zip(labels, emp, pg_v, lstm_v), key=lambda r: -r[1])[:7]
    lines.append("| Pitch | Empirical | PitchGPT | LSTM | PG−Emp | LSTM−Emp |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for lbl, e, p, l in ranked:
        lines.append(f"| {lbl} | {e:.4f} | {p:.4f} | {l:.4f} | {p-e:+.4f} | {l-e:+.4f} |")
    lines.append("")

    # Stratified
    strat = payload.get("stratified_kl_by_context", {})
    if strat:
        lines.append("### Calibration-by-context: pitch-type KL stratified\n")
        for key, buckets in strat.items():
            if not buckets:
                continue
            lines.append(f"#### {key}")
            lines.append("| stratum | n PAs | PG KL | LSTM KL | Δ(PG−LSTM) |")
            lines.append("|---|---:|---:|---:|---:|")
            for stratum_val, stats in sorted(buckets.items(), key=lambda kv: -kv[1]["n_pas"])[:12]:
                lines.append(
                    f"| {stratum_val} | {stats['n_pas']} | {stats['pg_kl']:.4f} | "
                    f"{stats['lstm_kl']:.4f} | {stats['diff_pg_minus_lstm']:+.4f} |"
                )
            lines.append("")

    lines.append("## Interpretation\n")
    # Auto-interpret: for each metric, win/loss/tie.
    wins = 0
    losses = 0
    ties = 0
    per_metric_msg = []
    for k, row in m.items():
        diff = row["diff_point_pg_minus_lstm"]
        lo = row.get("diff_ci95_lo")
        hi = row.get("diff_ci95_hi")
        if lo is None or hi is None:
            label = "PitchGPT lower" if diff < 0 else ("Tie" if abs(diff) < 1e-6 else "LSTM lower")
            per_metric_msg.append(f"- `{k}`: Δ={diff:+.4f}  ({label}, no CI computed)")
        else:
            if hi < 0:
                wins += 1
                per_metric_msg.append(f"- `{k}`: Δ={diff:+.4f}, CI [{lo}, {hi}] — **PitchGPT significantly closer to empirical**.")
            elif lo > 0:
                losses += 1
                per_metric_msg.append(f"- `{k}`: Δ={diff:+.4f}, CI [{lo}, {hi}] — **LSTM significantly closer**.")
            else:
                ties += 1
                per_metric_msg.append(f"- `{k}`: Δ={diff:+.4f}, CI [{lo}, {hi}] — CI crosses 0, **no significant difference**.")
    lines.extend(per_metric_msg)
    lines.append("")
    lines.append(f"**Scorecard**: PitchGPT wins {wins} / LSTM wins {losses} / ties {ties} out of {len(m)} metrics (95% CI on Δ).")
    lines.append("")
    if wins > losses and wins >= 2:
        verdict = "PitchGPT shows real sampling-fidelity advantages on multiple metrics; the generative-engine claim survives."
    elif losses > wins:
        verdict = "LSTM wins sampling fidelity more often than PitchGPT; the generative claim does NOT hold, consistent with the 3.13% holdout-perplexity gap."
    elif wins > 0 and losses == 0:
        verdict = "PitchGPT wins some metrics and ties elsewhere (no losses); a defensible but narrow generative-fidelity edge."
    elif wins == 0 and losses == 0:
        verdict = "PitchGPT and LSTM are statistically indistinguishable on every sampling-fidelity metric.  The generative claim does not find empirical support at this scale."
    else:
        verdict = "Mixed picture; PitchGPT is not clearly better on sampling fidelity overall."
    lines.append(f"**Verdict**: {verdict}")
    lines.append("")

    lines.append("## Limitations\n")
    lines.append("- **Horizon truncation**: 6-pitch horizon covers the modal PA (3-5 pitches) but truncates long PAs.")
    lines.append("- **Outcome heuristic**: Implied strike/ball counting uses zone position as a proxy; this biases both models "
                 "uniformly but does not reflect Statcast's actual ball/strike logic (umpire effects, foul-ball dynamics, "
                 "check-swings, chase rates).")
    lines.append("- **Empirical reference is finite**: At ~2K PAs × mean 3.7 pitches = ~7.4K empirical tokens, KL/χ² estimates "
                 "have non-trivial noise.")
    lines.append("- **Temperature=1 only**: We do not sweep temperature.  Higher temperatures will systematically widen both "
                 "models' distributions; lower will sharpen.  The tradeoff may differ between architectures.")
    lines.append("- **Multiple-comparison issue**: 5 CI tests + strata → ~25 comparisons.  Bonferroni α/N reduces effective α "
                 "to 0.002, tightening any significance claims.")
    lines.append("- **Matching to 10K**: We use the same 10K game subset as the matched holdout.  Scaling could change "
                 "any winner's margin.")
    lines.append("")

    lines.append("## Reproducibility\n")
    wc = payload.get("wall_clock", {})
    lines.append(f"- PA fetch: {wc.get('pa_fetch_sec', '?')}s")
    lines.append(f"- PitchGPT sampling: {wc.get('pg_sampling_sec', '?')}s")
    lines.append(f"- LSTM sampling: {wc.get('lstm_sampling_sec', '?')}s")
    lines.append(f"- Metric computation (incl. bootstrap): {wc.get('metric_computation_sec', '?')}s")
    lines.append("")
    lines.append("Exact command:")
    lines.append("```")
    lines.append("python scripts/pitchgpt_sampling_fidelity.py \\")
    lines.append(f"  --pitchgpt-checkpoint models/pitchgpt_v2.pt \\")
    lines.append(f"  --lstm-checkpoint models/pitch_lstm_10k.pt \\")
    lines.append(f"  --n-pas {cfg['n_pas_drawn']} \\")
    lines.append(f"  --n-samples-per-pa {cfg['n_samples_per_pa']} \\")
    lines.append(f"  --horizon {cfg['horizon']} \\")
    lines.append(f"  --temperature {cfg['temperature']} \\")
    lines.append(f"  --seed {payload['seed']}")
    lines.append("```")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
