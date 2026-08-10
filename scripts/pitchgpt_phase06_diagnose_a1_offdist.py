"""
Phase 0.6 PitchGPT diagnostic: A1 outcome head off-distribution at
counterfactually-sampled pitches.

Hypothesis under test
---------------------
A1 was trained on real (hidden, context, pitch_token) tuples from the
2015-2022 cohort.  At rollout time the backbone samples pitches that may be
plausible at the token level but rare/absent in the (hidden, context, pitch)
joint distribution A1 saw.  At those "rare joints" A1's outcome probabilities
may be miscalibrated -- biased toward common outcomes (called/swinging
strikes), against rare outcomes (balls).

If TRUE we expect:
    * Counterfactual A1 outcome distribution (over rollout-sampled pitches)
      shifts toward strikes vs the in-distribution (real-pitch) A1 outcome
      distribution.
    * Tokens that are rare in 2015-2022 train but common in 2025 rollout
      sampling carry the bulk of the bias.
    * ECE balloons under counterfactual sampling because temperature
      T = 0.8003 was fit on in-distribution test pitches.

What this script does NOT do
----------------------------
* Modify ``src/analytics/pitchgpt_sim.py``, ``src/analytics/pitchgpt.py``,
  ``src/analytics/pitchgpt_outcome_head.py`` or any model checkpoint.
* Touch DuckDB write -- read-only throughout.
* Commit anything (PM commits after review).
* Test static-context bias (parallel diagnostic agent).
* Test pitch-token sampling bias (parallel diagnostic agent).

Outputs
-------
    results/pitchgpt/rollout_sanity_2025/diagnose_a1_offdist.json
    results/pitchgpt/rollout_sanity_2025/diagnose_a1_offdist.md

Verdict in JSON ``"verdict"`` field: CONFIRMED / FALSIFIED / INDETERMINATE.

Time budget
-----------
~30 min.  If wall-clock projects past 45 min, the script auto-falls-back
to n_pa = 2000 (logged in the report).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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
    VOCAB_SIZE,
    PitchTokenizer,
)
from src.analytics.pitchgpt_outcome_head import (  # noqa: E402
    NUM_OUTCOME_CLASSES,
    OUTCOME_CLASSES,
    OUTCOME_UNK,
    classify_pitch_outcome,
    extract_backbone_hidden_states,
)
from src.analytics.pitchgpt_sim import (  # noqa: E402
    PAContext,
    PGConcatHeadPredictor,
    ROLLOUT_PAD_PITCH,
    _decompose_pitch_token,
    _load_backbone,
    rollout,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase06_diagnose_a1_offdist")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_RANGE = (2015, 2022)
HOLDOUT_SEASON = 2025  # holdout-contact: historical contacts registered in docs/holdout_ledger.jsonl; any NEW run must append a new contact via src.holdout (@holdout_access / record_contact)
N_PA_DEFAULT = 5_000
N_PA_FALLBACK = 2_000
N_SAMPLES_DEFAULT = 20
HORIZON = 6
TEMPERATURE = 1.0
SEED_ROOT = 42

# Threshold time to trigger fallback (seconds).
WALL_CLOCK_FALLBACK_TRIGGER_SEC = 45 * 60

# Outcome class indices (mirror src/analytics/pitchgpt_outcome_head.py).
OUTCOME_BALL = 0
OUTCOME_CALLED_STRIKE = 1
OUTCOME_SWINGING_STRIKE = 2
OUTCOME_FOUL = 3
OUTCOME_IN_PLAY_OUT = 4
OUTCOME_IN_PLAY_HIT = 5
OUTCOME_HBP = 6

# A1 locked temperature -- replicate the predictor's behavior, but allow
# T=1 evaluation for the calibration drift comparison.
A1_LOCKED_TEMPERATURE = 0.8003

# Rare/common token thresholds for the joint-distribution drift test.
# "rare" = token freq < 1e-3 in train, but >= 1e-2 in rollout sample.
TRAIN_RARE_FREQ_THRESH = 1e-3
ROLLOUT_COMMON_FREQ_THRESH = 1e-2

# Per-class TVD threshold (across the 7 outcome classes) we use as the
# per-class tipping point for the verdict.  Total Variation Distance
# is half-sum of |p - q|; a TVD > 0.05 between in-dist and counterfactual
# A1 distributions is "large" relative to the on-task ECE_post = 0.0114
# the head was tuned to.  KL divergence > 0.05 also flags bias.
TVD_CONFIRM_THRESH = 0.05
KL_CONFIRM_THRESH = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# SQL helpers (mirror pitchgpt_rollout_sanity_2025.py — read-only DuckDB)
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_train_pitcher_ids(conn) -> set[int]:
    """Pitchers who appeared in 2015-2022 (the train cohort)."""
    s_str = ", ".join(str(s) for s in range(TRAIN_RANGE[0], TRAIN_RANGE[1] + 1))
    rows = conn.execute(
        f"""
        SELECT DISTINCT pitcher_id
        FROM pitches
        WHERE pitch_type IS NOT NULL
          AND pitcher_id IS NOT NULL
          AND EXTRACT(YEAR FROM game_date) IN ({s_str})
        """,
    ).fetchdf()
    return {int(p) for p in rows["pitcher_id"].tolist() if p is not None}


def _fetch_2025_pitcher_disjoint_pa_pitches(
    conn,
    train_pitchers: set[int],
) -> pd.DataFrame:
    """Return one row per real 2025 pitcher-disjoint pitch with PA grouping
    keys + per-pitch fields needed to (a) tokenise and (b) classify outcomes.

    Mirrors the cohort definition in pitchgpt_rollout_sanity_2025.py but
    keeps every pitch (not just the PA start) so we can stream backbone
    hidden states position-by-position.
    """
    pids_str = ", ".join(str(int(p)) for p in train_pitchers)
    query = f"""
        WITH all_2025 AS (
            SELECT
                game_pk, at_bat_number, pitcher_id, batter_id,
                pitch_number,
                pitch_type, plate_x, plate_z, release_speed,
                balls, strikes, outs_when_up,
                on_1b, on_2b, on_3b,
                stand, p_throws,
                inning, inning_topbot,
                events, woba_value, description,
                game_date
            FROM pitches
            WHERE EXTRACT(YEAR FROM game_date) = {HOLDOUT_SEASON}
              AND pitch_type IS NOT NULL
              AND pitcher_id IS NOT NULL
              AND batter_id IS NOT NULL
              AND at_bat_number IS NOT NULL
              AND pitch_number IS NOT NULL
              AND pitcher_id NOT IN ({pids_str})
        )
        SELECT *
        FROM all_2025
        ORDER BY game_pk, at_bat_number, pitch_number
    """
    return conn.execute(query).fetchdf()


def _fetch_train_pitch_token_freqs(conn, train_pitchers: set[int]) -> np.ndarray:
    """Per-token frequency over the 2015-2022 train cohort.

    Vectorised: we let DuckDB do the bucketing + tokenisation entirely
    in SQL, mirroring ``PitchTokenizer.encode`` / ``location_to_zone`` /
    ``velocity_to_bucket``.  This avoids ~4M Python iterrows.

    Returns a (VOCAB_SIZE,) float64 array summing to 1.0.
    """
    s_str = ", ".join(str(s) for s in range(TRAIN_RANGE[0], TRAIN_RANGE[1] + 1))
    pids_str = ", ".join(str(int(p)) for p in train_pitchers)

    # Replicate the PitchTokenizer mappings.  We rely on DuckDB CASE
    # for pitch_type -> idx and floor()/least()/greatest() for binning.
    # plate range: x in [-1.5, 1.5] (5 bins, width 0.6); z in [1.0, 4.0]
    # (5 bins, width 0.6).  Out-of-zone => 25 (NUM_ZONES - 1).
    # NULL plate / velo => default zone 25, default velo 2.
    query = f"""
        WITH src AS (
            SELECT
                pitch_type,
                plate_x,
                plate_z,
                release_speed
            FROM pitches
            WHERE pitch_type IS NOT NULL
              AND EXTRACT(YEAR FROM game_date) IN ({s_str})
              AND pitcher_id IN ({pids_str})
        ),
        bucketed AS (
            SELECT
                CASE pitch_type
                    WHEN 'FF' THEN 0
                    WHEN 'SI' THEN 1
                    WHEN 'FC' THEN 2
                    WHEN 'SL' THEN 3
                    WHEN 'CU' THEN 4
                    WHEN 'CH' THEN 5
                    WHEN 'FS' THEN 6
                    WHEN 'KC' THEN 7
                    WHEN 'KN' THEN 8
                    WHEN 'EP' THEN 9
                    WHEN 'CS' THEN 10
                    WHEN 'SV' THEN 11
                    WHEN 'ST' THEN 12
                    WHEN 'SC' THEN 13
                    WHEN 'FO' THEN 14
                    WHEN 'FA' THEN 15
                    ELSE 16
                END AS pt_idx,
                CASE
                    WHEN plate_x IS NULL OR plate_z IS NULL THEN 25
                    ELSE
                        LEAST(GREATEST(CAST(FLOOR((plate_z - 1.0) / 0.6) AS INTEGER), 0), 4) * 5
                      + LEAST(GREATEST(CAST(FLOOR((plate_x + 1.5) / 0.6) AS INTEGER), 0), 4)
                END AS zone_idx,
                CASE
                    WHEN release_speed IS NULL THEN 2
                    WHEN release_speed < 80 THEN 0
                    WHEN release_speed < 85 THEN 1
                    WHEN release_speed < 90 THEN 2
                    WHEN release_speed < 95 THEN 3
                    ELSE 4
                END AS velo_idx
            FROM src
        )
        SELECT
            pt_idx * 130 + zone_idx * 5 + velo_idx AS token,
            COUNT(*) AS n
        FROM bucketed
        GROUP BY 1
        ORDER BY 1
    """
    logger.info("Vectorised SQL fetch of train cohort token counts...")
    df = conn.execute(query).fetchdf()
    logger.info(
        "Train cohort: %d distinct tokens (sum=%d)",
        len(df), int(df["n"].sum()) if len(df) else 0,
    )
    counts = np.zeros(VOCAB_SIZE, dtype=np.float64)
    for _, r in df.iterrows():
        tok = int(r["token"])
        if 0 <= tok < VOCAB_SIZE:
            counts[tok] = float(r["n"])
    total = counts.sum()
    if total <= 0:
        raise RuntimeError("Empty train cohort -- frequency table cannot be computed.")
    return counts / total


# ─────────────────────────────────────────────────────────────────────────────
# Build per-PA real-pitch sequence (tokens + context vec + descr/events/etc.)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RealPASequence:
    pa_key: tuple[int, int, int, int]
    tokens: np.ndarray  # (n_pitches,) int64
    context_vec: np.ndarray  # (CONTEXT_DIM,) float32  -- constant per PA
    outcome_targets: np.ndarray  # (n_pitches,) int64
    pitcher_id: int
    batter_id: int
    pa_start_row: pd.Series  # for PAContext construction


def _safe_int(v, default=0):
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    try:
        if isinstance(v, float) and np.isnan(v):
            return default
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_bool(v):
    if v is None:
        return False
    # Pandas NA (BIGINT NULL on DuckDB columns) -> False.
    try:
        if pd.isna(v):
            return False
    except (TypeError, ValueError):
        pass
    try:
        if isinstance(v, float) and np.isnan(v):
            return False
        return bool(int(v))
    except (TypeError, ValueError):
        try:
            return bool(v)
        except (TypeError, ValueError):
            return False


def _safe_str(v, default="R"):
    if v is None:
        return default
    try:
        if pd.isna(v):
            return default
    except (TypeError, ValueError):
        pass
    s = str(v).strip()
    return s if s else default


def _coerce_int_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-coerce DuckDB BIGINT-NULL columns (which surface as pandas NA)
    to plain int64 with 0 for missing -- matches the pre-coercion in
    ``scripts/pitchgpt_rollout_sanity_2025.py``.  Without this, the
    ``PAContext.from_pitches_row`` factory in the sim module hits
    ``NAType`` and raises.
    """
    for col in (
        "on_1b", "on_2b", "on_3b",
        "balls", "strikes", "outs_when_up",
        "inning", "pitcher_id", "batter_id",
        "at_bat_number", "pitch_number",
    ):
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0)
                .astype(np.int64)
            )
    return df


def _build_real_pa_sequences(
    df: pd.DataFrame, max_n_pa: int, league_median_ump: float, seed: int,
) -> list[RealPASequence]:
    """Build n_pa real PA sequences from the streamed-pitch DataFrame.

    For each PA we:
        * tokenise every pitch via ``PitchTokenizer.encode``.
        * build a constant 35-D context vector at the PA start
          (matching ``rollout()``'s constant-context-within-PA behavior).
        * label each pitch's outcome via ``classify_pitch_outcome``.

    Returns a list of size ``min(max_n_pa, n_pa_available)`` selected
    deterministically with seed=seed.
    """
    rng = np.random.default_rng(seed)
    grouped = df.groupby(
        ["game_pk", "at_bat_number", "pitcher_id", "batter_id"], sort=False,
    )
    all_groups = list(grouped)
    n_avail = len(all_groups)
    logger.info("Real PA groups available: %d", n_avail)
    if n_avail < max_n_pa:
        logger.warning(
            "Only %d PAs available; using all of them (requested %d).",
            n_avail, max_n_pa,
        )
        target_idx = np.arange(n_avail)
    else:
        target_idx = rng.choice(n_avail, size=max_n_pa, replace=False)
    target_set = set(int(i) for i in target_idx)
    out: list[RealPASequence] = []
    for i, (key, gdf) in enumerate(all_groups):
        if i not in target_set:
            continue
        gdf_sorted = gdf.sort_values("pitch_number")
        first_row = gdf_sorted.iloc[0]
        # Build the constant 35-D context vector at the PA start.  We use
        # the 2025 league-median ump scalar (mirrors PAContext.from_pitches_row
        # default; rollout_sanity_2025 uses the same).
        ctx_list = PitchTokenizer.encode_context(
            balls=_safe_int(first_row.get("balls"), 0),
            strikes=_safe_int(first_row.get("strikes"), 0),
            outs=_safe_int(first_row.get("outs_when_up"), 0),
            on_1b=_safe_bool(first_row.get("on_1b")),
            on_2b=_safe_bool(first_row.get("on_2b")),
            on_3b=_safe_bool(first_row.get("on_3b")),
            stand=_safe_str(first_row.get("stand"), "R"),
            inning=_safe_int(first_row.get("inning"), 1),
            score_diff=0,  # match rollout's default; no per-pitch score diff
        )
        ctx_t = PitchTokenizer.context_to_tensor(
            ctx_list, ump_scalar=float(league_median_ump),
        ).numpy()
        tokens = []
        outcomes = []
        for _, row in gdf_sorted.iterrows():
            tok = PitchTokenizer.encode(
                row.get("pitch_type"),
                row.get("plate_x"),
                row.get("plate_z"),
                row.get("release_speed"),
            )
            tokens.append(int(tok) if 0 <= int(tok) < VOCAB_SIZE else -1)
            outcomes.append(int(classify_pitch_outcome(
                row.get("description"), row.get("events"),
            )))
        out.append(RealPASequence(
            pa_key=(int(key[0]), int(key[1]), int(key[2]), int(key[3])),
            tokens=np.asarray(tokens, dtype=np.int64),
            context_vec=ctx_t.astype(np.float32),
            outcome_targets=np.asarray(outcomes, dtype=np.int64),
            pitcher_id=int(key[2]),
            batter_id=int(key[3]),
            pa_start_row=first_row,
        ))
    logger.info("Real PA sequences built: %d", len(out))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Run A1 head on real (in-distribution) sequences
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _run_a1_on_real_pa(
    pa: RealPASequence,
    backbone,
    head_predictor: PGConcatHeadPredictor,
    device: torch.device,
    use_temperature: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each pitch position j>=1 in the real PA, run A1 outcome head on
    (hidden_at_position_j_predicting_pitch, context, real_pitch_token_j).

    The context is held constant within a PA, which mirrors how the rollout
    feeds A1.  The hidden state is computed by the FROZEN backbone on the
    sequence [BOS, t_0, ..., t_{j-1}], and we read off position -1 as the
    "predict-next" representation.  We then concat (hidden, context, t_j_oh)
    and pass through the A1 head -- this is the SAME forward pass A1 saw
    at training time on real data.

    Returns
    -------
    probs : (n_eval, 7) float32  -- A1 calibrated probabilities (post-T)
    targets : (n_eval,) int64    -- 7-class outcome targets (-1 if UNK)
    pitch_tokens : (n_eval,) int64 -- the actual pitch token at position j
    valid_mask : (n_eval,) bool  -- target != UNK and token in [0, VOCAB_SIZE)
    """
    n = pa.tokens.shape[0]
    if n < 1:
        empty = np.empty((0, NUM_OUTCOME_CLASSES), dtype=np.float32)
        return (empty,
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=bool))

    # Build the input sequence = [BOS, t_0, t_1, ..., t_{n-1}]
    # We need hidden states at positions 0, 1, ..., n-1 (each predicts the
    # next pitch in [t_0, ..., t_{n-1}]).
    valid_tokens = np.where(
        (pa.tokens >= 0) & (pa.tokens < VOCAB_SIZE), pa.tokens, PAD_TOKEN,
    )
    seq = np.concatenate(([BOS_TOKEN], valid_tokens), axis=0).astype(np.int64)
    seq_t = torch.from_numpy(seq).long().unsqueeze(0).to(device)

    ctx_t = torch.from_numpy(pa.context_vec).float().unsqueeze(0).unsqueeze(0)
    ctx_t = ctx_t.expand(1, seq.shape[0], -1).contiguous().to(device)

    # Get hidden states from the backbone's transformer.
    hidden = extract_backbone_hidden_states(
        backbone, seq_t, ctx_t,
    )  # (1, n+1, d_model)

    # We want hidden state at positions 0..n-1 (these are the "predict next"
    # positions for pitch tokens at sequence positions 1..n which == pa.tokens
    # at index 0..n-1).
    pred_hidden = hidden[:, :n, :]  # (1, n, d_model)
    pred_ctx = ctx_t[:, :n, :]      # (1, n, CONTEXT_DIM)

    # Pitch tokens to score against (the actual real tokens at those positions).
    pitch_tok_t = torch.from_numpy(valid_tokens.astype(np.int64)).long().unsqueeze(0).to(device)

    # Forward through the A1 head.  PGConcatHeadPredictor.predict_outcome_probs
    # already applies temperature scaling and class calibration.  For the
    # "use_temperature=False" path we want raw post-softmax probs (T=1 ECE
    # comparison).  The predictor doesn't expose that directly, so we
    # replicate its forward inline -- this is read-only access to the
    # head module weights.
    pt_idx, zone_idx, velo_idx = _decompose_pitch_token(pitch_tok_t)
    pt_oh = F.one_hot(pt_idx, num_classes=head_predictor._n_pitch_types).float()
    zn_oh = F.one_hot(zone_idx, num_classes=head_predictor._n_zones).float()
    vl_oh = F.one_hot(velo_idx, num_classes=head_predictor._n_velo_buckets).float()
    logits = head_predictor.head(pred_hidden, pred_ctx, pt_oh, zn_oh, vl_oh)
    if use_temperature:
        scaled = logits / max(head_predictor.temperature, 1e-8)
    else:
        scaled = logits  # T=1.0
    probs = F.softmax(scaled, dim=-1)
    # Apply class calibration if present (mirror predict_outcome_probs).
    if head_predictor._class_calibration is not None:
        weighted = probs * head_predictor._class_calibration
        denom = weighted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = weighted / denom
    probs_np = probs.squeeze(0).cpu().numpy().astype(np.float32)

    targets = pa.outcome_targets
    valid_mask = (targets != OUTCOME_UNK) & (
        (pa.tokens >= 0) & (pa.tokens < VOCAB_SIZE)
    )
    return probs_np, targets, pa.tokens.copy(), valid_mask


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation utilities
# ─────────────────────────────────────────────────────────────────────────────


def _outcome_marginal(probs: np.ndarray) -> np.ndarray:
    """Marginal distribution over the 7 outcome classes from a (N, 7)
    probability matrix.  Returns (7,) float64 summing to 1.0.
    """
    if probs.shape[0] == 0:
        return np.full(NUM_OUTCOME_CLASSES, 1.0 / NUM_OUTCOME_CLASSES)
    return probs.mean(axis=0).astype(np.float64)


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total Variation Distance between two discrete distributions."""
    return float(0.5 * np.sum(np.abs(p - q)))


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) -- treats p as the reference, q as approximation."""
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))


def _compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Top-1 Expected Calibration Error.  UNK rows must be filtered out
    by the caller.
    """
    if probs.shape[0] == 0:
        return float("nan")
    top1_p = probs.max(axis=-1)
    top1_idx = probs.argmax(axis=-1)
    correct = (top1_idx == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_total = probs.shape[0]
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (top1_p >= lo) & (top1_p <= hi)
        else:
            mask = (top1_p >= lo) & (top1_p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_conf = float(top1_p[mask].mean())
        emp_acc = float(correct[mask].mean())
        ece += (n / n_total) * abs(mean_conf - emp_acc)
    return float(ece)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pa", type=int, default=N_PA_DEFAULT)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES_DEFAULT)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--seed", type=int, default=SEED_ROOT)
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "results" / "pitchgpt" / "rollout_sanity_2025",
    )
    parser.add_argument(
        "--skip-train-freq",
        action="store_true",
        help="Skip the train cohort frequency fetch (debug; fast).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    logger.info(
        "Phase 0.6 A1 off-distribution diagnostic start: "
        "n_pa=%d  n_samples=%d  horizon=%d  seed=%d",
        args.n_pa, args.n_samples, args.horizon, args.seed,
    )

    # ── 1. Pull cohort + train freq table (read-only) ──────────────────────
    conn = get_connection(args.db_path, read_only=True)
    try:
        train_pitchers = _fetch_train_pitcher_ids(conn)
        logger.info("Train pitcher set: %d", len(train_pitchers))
        df_pitches = _fetch_2025_pitcher_disjoint_pa_pitches(
            conn, train_pitchers,
        )
        logger.info(
            "Streamed 2025 pitcher-disjoint pitches: %d", len(df_pitches),
        )
        df_pitches = _coerce_int_columns(df_pitches)
        league_median_ump = 0.0  # rollout_sanity_2025 uses the 2024 prior
        # season median; for the diagnostic we keep the same default 0.0
        # the rollout is broadcasting since both arms see identical context.

        if not args.skip_train_freq:
            train_token_freqs = _fetch_train_pitch_token_freqs(
                conn, train_pitchers,
            )
            logger.info(
                "Train token freq table built (n_unique=%d, sum=%.4f)",
                int((train_token_freqs > 0).sum()), float(train_token_freqs.sum()),
            )
        else:
            train_token_freqs = None
            logger.warning("--skip-train-freq: token-frequency analysis will be skipped.")
    finally:
        conn.close()

    # ── 2. Build real PA sequences ─────────────────────────────────────────
    real_pas = _build_real_pa_sequences(
        df_pitches, args.n_pa, league_median_ump, args.seed,
    )

    # ── 3. Load backbone + A1 predictor ────────────────────────────────────
    cuda = torch.cuda.is_available()
    logger.info("Device: %s (CUDA available=%s)", "cuda" if cuda else "cpu", cuda)
    backbone, backbone_sha = _load_backbone("v2")
    device = next(backbone.parameters()).device
    a1 = PGConcatHeadPredictor()
    logger.info(
        "A1 head loaded; T_locked = %.4f, ECE_post = %.4f",
        a1.temperature, a1.calibration.get("ECE_post", float("nan")),
    )

    # ── 4. PASS A: in-distribution probs on REAL pitches ──────────────────
    logger.info("Pass A: scoring %d real PAs in-distribution...", len(real_pas))
    t_a = time.perf_counter()
    in_probs_chunks = []
    in_targets_chunks = []
    in_tokens_chunks = []
    in_probs_T1_chunks = []
    log_every = max(len(real_pas) // 20, 50)
    for i, pa in enumerate(real_pas):
        probs, targets, pa_tokens, valid = _run_a1_on_real_pa(
            pa, backbone, a1, device, use_temperature=True,
        )
        if valid.any():
            in_probs_chunks.append(probs[valid])
            in_targets_chunks.append(targets[valid])
            in_tokens_chunks.append(pa_tokens[valid])
        # Also compute T=1.0 probs for ECE-without-temperature comparison.
        probs_T1, targets_T1, _, valid_T1 = _run_a1_on_real_pa(
            pa, backbone, a1, device, use_temperature=False,
        )
        if valid_T1.any():
            in_probs_T1_chunks.append(probs_T1[valid_T1])
        if (i + 1) % log_every == 0:
            dt = time.perf_counter() - t_a
            rate = (i + 1) / dt
            eta = (len(real_pas) - (i + 1)) / max(rate, 1e-6)
            logger.info(
                "[Pass A] %d/%d PAs  rate=%.1f PA/s  eta=%.0fs",
                i + 1, len(real_pas), rate, eta,
            )
            # Time budget guard.
            elapsed_total = time.perf_counter() - t_start
            if elapsed_total > WALL_CLOCK_FALLBACK_TRIGGER_SEC and len(real_pas) > N_PA_FALLBACK:
                logger.warning(
                    "Wall-clock %ds exceeds %ds threshold; truncating to first %d PAs.",
                    int(elapsed_total), WALL_CLOCK_FALLBACK_TRIGGER_SEC, N_PA_FALLBACK,
                )
                real_pas = real_pas[: max(i + 1, 1)]
                break

    in_probs = (
        np.concatenate(in_probs_chunks, axis=0)
        if in_probs_chunks else np.empty((0, NUM_OUTCOME_CLASSES), dtype=np.float32)
    )
    in_targets = (
        np.concatenate(in_targets_chunks, axis=0)
        if in_targets_chunks else np.empty((0,), dtype=np.int64)
    )
    in_tokens = (
        np.concatenate(in_tokens_chunks, axis=0)
        if in_tokens_chunks else np.empty((0,), dtype=np.int64)
    )
    in_probs_T1 = (
        np.concatenate(in_probs_T1_chunks, axis=0)
        if in_probs_T1_chunks else np.empty((0, NUM_OUTCOME_CLASSES), dtype=np.float32)
    )
    t_a_total = time.perf_counter() - t_a
    logger.info(
        "Pass A done in %.1fs.  In-dist eval rows: n=%d", t_a_total, len(in_probs),
    )

    # ── 5. PASS B: rollout-sampled (counterfactual) A1 outcome probs ──────
    logger.info(
        "Pass B: rolling out %d PAs (n_samples=%d, H=%d) -- counterfactual...",
        len(real_pas), args.n_samples, args.horizon,
    )
    t_b = time.perf_counter()
    cf_probs_chunks = []
    cf_tokens_chunks = []
    log_every_b = max(len(real_pas) // 20, 50)
    for i, pa in enumerate(real_pas):
        ctx = PAContext.from_pitches_row(pa.pa_start_row, ump_scalar=league_median_ump)
        per_row_seed = (args.seed * 1_000_003 + i) & 0x7FFFFFFF
        result = rollout(
            ctx,
            outcome_predictor=a1,
            n_samples=args.n_samples,
            horizon=args.horizon,
            temperature=TEMPERATURE,
            seed=per_row_seed,
            return_probs=True,
        )
        # outcome_probs shape (n_samples, horizon, 7); pitch_tokens shape
        # (n_samples, horizon).  Filter PAD positions.
        outc_probs = result.outcome_probs  # (n_samples, horizon, 7)
        toks = result.pitch_tokens  # (n_samples, horizon)
        if outc_probs is None:
            continue
        valid_mask = toks != ROLLOUT_PAD_PITCH
        cf_probs_chunks.append(outc_probs[valid_mask])
        cf_tokens_chunks.append(toks[valid_mask])
        if (i + 1) % log_every_b == 0:
            dt = time.perf_counter() - t_b
            rate = (i + 1) / dt
            eta = (len(real_pas) - (i + 1)) / max(rate, 1e-6)
            logger.info(
                "[Pass B] %d/%d PAs  rate=%.1f PA/s  eta=%.0fs",
                i + 1, len(real_pas), rate, eta,
            )

    cf_probs = (
        np.concatenate(cf_probs_chunks, axis=0).astype(np.float32)
        if cf_probs_chunks else np.empty((0, NUM_OUTCOME_CLASSES), dtype=np.float32)
    )
    cf_tokens = (
        np.concatenate(cf_tokens_chunks, axis=0).astype(np.int64)
        if cf_tokens_chunks else np.empty((0,), dtype=np.int64)
    )
    t_b_total = time.perf_counter() - t_b
    logger.info(
        "Pass B done in %.1fs.  Counterfactual eval rows: n=%d", t_b_total, len(cf_probs),
    )

    # ── 6. Outcome marginal comparison ─────────────────────────────────────
    in_marginal = _outcome_marginal(in_probs)
    cf_marginal = _outcome_marginal(cf_probs)
    tvd_overall = _tvd(in_marginal, cf_marginal)
    kl_in_to_cf = _kl_divergence(in_marginal, cf_marginal)
    kl_cf_to_in = _kl_divergence(cf_marginal, in_marginal)
    js_div = 0.5 * (
        _kl_divergence(in_marginal, 0.5 * (in_marginal + cf_marginal))
        + _kl_divergence(cf_marginal, 0.5 * (in_marginal + cf_marginal))
    )

    # Per-class delta in pp.
    per_class_delta = (cf_marginal - in_marginal) * 100.0  # in pp

    logger.info(
        "Outcome marginal -- in-dist:  %s",
        ", ".join(f"{c}={p:.4f}" for c, p in zip(OUTCOME_CLASSES, in_marginal)),
    )
    logger.info(
        "Outcome marginal -- counterf: %s",
        ", ".join(f"{c}={p:.4f}" for c, p in zip(OUTCOME_CLASSES, cf_marginal)),
    )
    logger.info(
        "TVD = %.4f  KL(in||cf) = %.4f  KL(cf||in) = %.4f  JS = %.4f",
        tvd_overall, kl_in_to_cf, kl_cf_to_in, js_div,
    )

    # ── 7. Per-token off-distribution analysis ────────────────────────────
    if train_token_freqs is not None:
        # Rollout token freq.
        rollout_freqs = np.zeros(VOCAB_SIZE, dtype=np.float64)
        if cf_tokens.size > 0:
            valid_cf_tokens = cf_tokens[(cf_tokens >= 0) & (cf_tokens < VOCAB_SIZE)]
            counts = np.bincount(valid_cf_tokens, minlength=VOCAB_SIZE)
            rollout_freqs = counts.astype(np.float64) / max(counts.sum(), 1)

        # Rare = low train freq, high rollout freq -> potential off-dist.
        rare_mask = (
            (train_token_freqs < TRAIN_RARE_FREQ_THRESH)
            & (rollout_freqs >= ROLLOUT_COMMON_FREQ_THRESH)
        )
        common_mask = (
            (train_token_freqs >= TRAIN_RARE_FREQ_THRESH)
            & (rollout_freqs >= TRAIN_RARE_FREQ_THRESH)
        )
        rare_token_set = set(int(t) for t in np.where(rare_mask)[0])
        n_rare_tokens = len(rare_token_set)

        # Aggregate counterfactual probs over rare-token rows vs common-token rows.
        if cf_tokens.size > 0 and n_rare_tokens > 0:
            is_rare = np.fromiter(
                (int(t) in rare_token_set for t in cf_tokens),
                dtype=bool, count=cf_tokens.size,
            )
            n_rare_eval = int(is_rare.sum())
            n_common_eval = int((~is_rare).sum())
            rare_marg = (
                _outcome_marginal(cf_probs[is_rare])
                if n_rare_eval > 0 else np.full(NUM_OUTCOME_CLASSES, np.nan)
            )
            common_marg = (
                _outcome_marginal(cf_probs[~is_rare])
                if n_common_eval > 0 else np.full(NUM_OUTCOME_CLASSES, np.nan)
            )
            tvd_rare_vs_common = (
                _tvd(rare_marg, common_marg) if n_rare_eval > 0 and n_common_eval > 0 else float("nan")
            )
        else:
            n_rare_eval = 0
            n_common_eval = int(cf_tokens.size)
            rare_marg = np.full(NUM_OUTCOME_CLASSES, np.nan)
            common_marg = (
                _outcome_marginal(cf_probs) if cf_tokens.size > 0 else np.full(NUM_OUTCOME_CLASSES, np.nan)
            )
            tvd_rare_vs_common = float("nan")

        # Top 10 over-sampled tokens (largest rollout - train delta among
        # those with rollout_freq >= 0.005).
        delta = rollout_freqs - train_token_freqs
        candidate_mask = rollout_freqs >= 0.005
        candidate_idx = np.where(candidate_mask)[0]
        if candidate_idx.size > 0:
            sorted_idx = candidate_idx[np.argsort(-delta[candidate_idx])][:15]
            top_oversampled = []
            for tok in sorted_idx:
                top_oversampled.append({
                    "token": int(tok),
                    "train_freq": round(float(train_token_freqs[tok]), 6),
                    "rollout_freq": round(float(rollout_freqs[tok]), 6),
                    "delta": round(float(delta[tok]), 6),
                })
        else:
            top_oversampled = []
    else:
        rollout_freqs = None
        n_rare_tokens = None
        n_rare_eval = None
        n_common_eval = None
        rare_marg = None
        common_marg = None
        tvd_rare_vs_common = None
        top_oversampled = []

    # ── 8. ECE drift (in-dist vs counterfactual; with vs without T) ───────
    # In-distribution ECE -- with T_locked + class_calibration (production),
    # also with T=1.0 (no temperature scaling), both vs the real targets.
    valid_in_T = in_targets >= 0
    ece_in_T_locked = _compute_ece(in_probs[valid_in_T], in_targets[valid_in_T])
    valid_in_T1 = in_targets >= 0
    ece_in_T1 = _compute_ece(in_probs_T1[valid_in_T1], in_targets[valid_in_T1])

    # Counterfactual ECE -- we have NO ground-truth labels for sampled
    # pitches (they are counterfactual).  But we can compute a proxy:
    # the model's top-1 confidence vs its OWN argmax-correctness, which
    # for unlabeled data reduces to the confidence vs accuracy gap if we
    # use the most-frequent outcome over all rollouts at each position
    # as a pseudo-target.  This is loose; we surface it explicitly.  The
    # core comparison is the in-dist ECE with T=0.8003 vs T=1.0 -- if
    # the gap is large in-distribution AND the marginal shift is large
    # under counterfactual sampling, the temperature is over-fit
    # in-distribution.

    # ── 9. Verdict ────────────────────────────────────────────────────────
    # CONFIRMED: TVD(in, cf) > TVD_CONFIRM_THRESH OR KL > KL_CONFIRM_THRESH,
    #            AND the bias is in the strikes-up / balls-down direction
    #            (consistent with the +8.9pp K% / -4.5pp BB% Phase 0.6 FAIL).
    # FALSIFIED: TVD < 0.02 AND KL < 0.02 -- no meaningful drift.
    # INDETERMINATE: in between.
    delta_ball_pp = float(per_class_delta[OUTCOME_BALL])
    delta_cs_pp = float(per_class_delta[OUTCOME_CALLED_STRIKE])
    delta_ss_pp = float(per_class_delta[OUTCOME_SWINGING_STRIKE])
    direction_consistent = (
        delta_ball_pp < 0.0  # counterfactual under-predicts ball
        and (delta_cs_pp + delta_ss_pp) > 0.0  # over-predicts strikes
    )
    # Verdict refinement: if the per-class drift direction is OPPOSITE to
    # the Phase 0.6 K% bias (counterfactual under-predicts strikes / over-
    # predicts balls), the off-distribution-toward-strikes hypothesis is
    # FALSIFIED regardless of magnitude -- the drift cannot be the cause
    # of a +K%/-BB% gate FAIL because A1's counterfactual distribution
    # actually leans the other way.
    direction_opposite = (
        delta_ball_pp > 0.0  # counterfactual OVER-predicts balls
        and (delta_cs_pp + delta_ss_pp) < 0.0  # UNDER-predicts strikes
    )
    if (tvd_overall > TVD_CONFIRM_THRESH or kl_in_to_cf > KL_CONFIRM_THRESH) and direction_consistent:
        verdict = "CONFIRMED"
    elif direction_opposite:
        # The hypothesis predicts strikes-up under counterfactual sampling.
        # If the data shows balls-up + strikes-down, the hypothesis is
        # falsified by direction even when TVD is non-negligible.
        verdict = "FALSIFIED"
    elif tvd_overall < 0.02 and kl_in_to_cf < 0.02:
        verdict = "FALSIFIED"
    else:
        verdict = "INDETERMINATE"

    # K% bias attribution (rough estimate).  The Phase 0.6 K% gap is
    # +8.9pp.  Counterfactual A1 over-predicts strikes by
    # (delta_cs + delta_ss) pp at the per-pitch level.  Strikes per PA
    # roughly compound across positions to drive K% -- so a per-pitch
    # bias of (delta_cs + delta_ss) / 100 multiplied by mean-PA-len
    # approximates the per-PA strike-mass shift.  Calibrate by mapping
    # per-pitch strike-marginal shift -> per-PA K-rate via the empirical
    # 2025 K-per-PA / strike-per-pitch ratio.  Numeric estimate is
    # approximate; we surface the formula.
    strike_marg_in = float(in_marginal[OUTCOME_CALLED_STRIKE] + in_marginal[OUTCOME_SWINGING_STRIKE])
    strike_marg_cf = float(cf_marginal[OUTCOME_CALLED_STRIKE] + cf_marginal[OUTCOME_SWINGING_STRIKE])
    delta_strike_marg = strike_marg_cf - strike_marg_in
    # Naive amplification: per-PA K-rate ~ P(get to strike 3 before ball 4)
    # ~ proportional to (strike per pitch / (strike + ball per pitch))^3.
    # We'll just report the per-pitch strike marginal shift in pp and let
    # the PM scale.
    k_pct_attribution_naive_pp = delta_strike_marg * 100.0
    # An upper bound on the K% gap explicable by this single channel:
    # if per-pitch strike marginal increases by X pp, expected K% shift
    # is roughly X/2 to X*1.5 depending on PA length distribution.

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec": "Phase 0.6 hypothesis: A1 outcome head off-distribution at counterfactually-sampled pitches",
        "args": {
            "n_pa": args.n_pa,
            "n_pa_actually_processed": len(real_pas),
            "n_samples": args.n_samples,
            "horizon": args.horizon,
            "seed": args.seed,
        },
        "device": "cuda" if cuda else "cpu",
        "wall_clock_seconds": {
            "pass_a_in_distribution": round(t_a_total, 1),
            "pass_b_counterfactual": round(t_b_total, 1),
            "total": round(time.perf_counter() - t_start, 1),
        },
        "n_eval_rows": {
            "in_distribution_pitches": int(in_probs.shape[0]),
            "counterfactual_pitches": int(cf_probs.shape[0]),
        },
        "outcome_marginals": {
            "in_distribution": {
                c: round(float(p), 6) for c, p in zip(OUTCOME_CLASSES, in_marginal)
            },
            "counterfactual": {
                c: round(float(p), 6) for c, p in zip(OUTCOME_CLASSES, cf_marginal)
            },
            "per_class_delta_pp": {
                c: round(float(d), 4) for c, d in zip(OUTCOME_CLASSES, per_class_delta)
            },
        },
        "distributional_distances": {
            "tvd": round(tvd_overall, 6),
            "kl_in_to_cf": round(kl_in_to_cf, 6),
            "kl_cf_to_in": round(kl_cf_to_in, 6),
            "js_divergence": round(js_div, 6),
            "tvd_confirm_threshold": TVD_CONFIRM_THRESH,
            "kl_confirm_threshold": KL_CONFIRM_THRESH,
        },
        "rare_token_analysis": {
            "skipped": train_token_freqs is None,
            "train_rare_threshold": TRAIN_RARE_FREQ_THRESH,
            "rollout_common_threshold": ROLLOUT_COMMON_FREQ_THRESH,
            "n_rare_tokens": int(n_rare_tokens) if n_rare_tokens is not None else None,
            "n_rare_eval_rows": int(n_rare_eval) if n_rare_eval is not None else None,
            "n_common_eval_rows": int(n_common_eval) if n_common_eval is not None else None,
            "rare_token_outcome_marginal": (
                {c: round(float(p), 6) for c, p in zip(OUTCOME_CLASSES, rare_marg)}
                if rare_marg is not None and not np.isnan(rare_marg).all() else None
            ),
            "common_token_outcome_marginal": (
                {c: round(float(p), 6) for c, p in zip(OUTCOME_CLASSES, common_marg)}
                if common_marg is not None and not np.isnan(common_marg).all() else None
            ),
            "tvd_rare_vs_common": (
                round(float(tvd_rare_vs_common), 6)
                if tvd_rare_vs_common is not None and not (
                    isinstance(tvd_rare_vs_common, float) and np.isnan(tvd_rare_vs_common)
                ) else None
            ),
            "top_oversampled_tokens": top_oversampled,
        },
        "calibration_drift": {
            "ece_in_distribution_T_locked": round(float(ece_in_T_locked), 6),
            "ece_in_distribution_T1": round(float(ece_in_T1), 6),
            "ece_locked_T_value": float(a1.temperature),
            "note": (
                "Counterfactual ECE cannot be computed without ground-truth "
                "labels for sampled pitches.  We report in-distribution ECE "
                "with T=0.8003 (production) vs T=1.0 (no scaling).  If the "
                "T=0.8003 ECE is much smaller than T=1.0 ECE in-distribution "
                "AND the marginal TVD(in, cf) > 0.05, the temperature is "
                "over-fit in-distribution and contributes to the off-dist "
                "bias under counterfactual sampling."
            ),
        },
        "k_pct_attribution_estimate": {
            "in_dist_strike_marginal": round(strike_marg_in, 6),
            "counterfactual_strike_marginal": round(strike_marg_cf, 6),
            "delta_strike_marginal_pp": round(delta_strike_marg * 100.0, 4),
            "naive_k_pct_pp_attribution": round(k_pct_attribution_naive_pp, 4),
            "phase06_observed_k_pct_pp_gap": 8.88,
            "fraction_of_observed_gap_explained_naive": round(
                k_pct_attribution_naive_pp / 8.88
                if abs(8.88) > 1e-9 else float("nan"),
                4,
            ),
            "note": (
                "Naive attribution: per-pitch strike-marginal shift in pp "
                "is an upper-bound estimate of per-PA K% shift contribution.  "
                "Compounding across PA-length distribution may amplify "
                "(strike-per-pitch -> K-per-PA is a positive-feedback loop).  "
                "Use this value as a rough order-of-magnitude only."
            ),
        },
        "verdict": verdict,
        "verdict_explanation": _verdict_text(
            verdict, tvd_overall, kl_in_to_cf,
            delta_ball_pp, delta_cs_pp, delta_ss_pp, direction_consistent,
        ),
        "checkpoints": {
            "backbone_v2_sha256": backbone_sha,
            "a1_head_sha256": a1.checkpoint_sha256,
            "a1_temperature_locked": float(a1.temperature),
        },
    }

    out_json = args.output_dir / "diagnose_a1_offdist.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    logger.info("Wrote %s", out_json)

    out_md = args.output_dir / "diagnose_a1_offdist.md"
    _write_report(out_md, payload)
    logger.info("Wrote %s", out_md)

    # Console summary.
    print("\n" + "=" * 78)
    print(f"Phase 0.6 A1 Off-Distribution Diagnostic -- VERDICT: {verdict}")
    print("=" * 78)
    print(f"n_pa processed: {len(real_pas)}  n_in_eval: {in_probs.shape[0]}  n_cf_eval: {cf_probs.shape[0]}")
    print(f"TVD(in_dist, counterfactual) = {tvd_overall:.4f}  (threshold {TVD_CONFIRM_THRESH})")
    print(f"KL(in||cf) = {kl_in_to_cf:.4f}  KL(cf||in) = {kl_cf_to_in:.4f}")
    print(f"Per-class delta (counterfactual - in_dist), pp:")
    for c, d in zip(OUTCOME_CLASSES, per_class_delta):
        print(f"  {c:<18}: {d:+.4f}")
    print(f"Strike marginal shift (per-pitch): {delta_strike_marg * 100:+.4f} pp")
    print(f"Phase 0.6 observed K% gap: +8.88 pp  (naive attribution: {k_pct_attribution_naive_pp:+.4f} pp)")
    if rare_marg is not None and not np.isnan(rare_marg).all():
        print(f"\nRare-vs-common token outcome TVD: {tvd_rare_vs_common:.4f}")
    print(f"\nIn-distribution ECE (T=0.8003): {ece_in_T_locked:.4f}")
    print(f"In-distribution ECE (T=1.0):    {ece_in_T1:.4f}")
    print(f"\nWall clock total: {time.perf_counter() - t_start:.1f}s")
    print("=" * 78)
    return 0


def _verdict_text(
    verdict: str,
    tvd: float,
    kl: float,
    delta_ball_pp: float,
    delta_cs_pp: float,
    delta_ss_pp: float,
    direction_consistent: bool,
) -> str:
    if verdict == "CONFIRMED":
        return (
            f"CONFIRMED: TVD={tvd:.4f} > {TVD_CONFIRM_THRESH} (or KL={kl:.4f} > {KL_CONFIRM_THRESH}). "
            f"Per-class drift is in the strikes-up/balls-down direction "
            f"(ball: {delta_ball_pp:+.2f}pp, called_strike: {delta_cs_pp:+.2f}pp, "
            f"swinging_strike: {delta_ss_pp:+.2f}pp), consistent with the "
            f"Phase 0.6 +8.9pp K% / -4.5pp BB% bias.  A1 is off-distribution "
            f"under counterfactual sampling and the temperature scaling does "
            f"not generalize off-task."
        )
    if verdict == "FALSIFIED":
        # Two falsifying paths: (a) drift below threshold, (b) drift in
        # the OPPOSITE direction (balls-up / strikes-down).
        direction_opposite = (
            delta_ball_pp > 0.0
            and (delta_cs_pp + delta_ss_pp) < 0.0
        )
        if direction_opposite:
            return (
                f"FALSIFIED (direction-opposite): TVD={tvd:.4f}, KL={kl:.4f}.  "
                f"Counterfactual A1 outcome distribution drifts in the "
                f"OPPOSITE direction from the Phase 0.6 K%/BB% bias: "
                f"ball: {delta_ball_pp:+.2f}pp (OVER-predicts ball), "
                f"called_strike: {delta_cs_pp:+.2f}pp, "
                f"swinging_strike: {delta_ss_pp:+.2f}pp "
                f"(net strike-marginal goes DOWN under counterfactual "
                f"sampling, not UP).  Whatever drift exists in A1 under "
                f"counterfactual sampling is in the WRONG direction to "
                f"explain the +8.9pp K% / -4.5pp BB% gate FAIL.  The bias "
                f"must come from elsewhere -- pitch-token sampling, "
                f"PA-termination logic, or static-context (parallel agents)."
            )
        return (
            f"FALSIFIED (small drift): TVD={tvd:.4f} < 0.02 and KL={kl:.4f} "
            f"< 0.02.  A1's outcome distribution under counterfactual "
            f"rollout sampling is essentially identical to its "
            f"in-distribution distribution.  The Phase 0.6 K%/BB% bias "
            f"must come from elsewhere (static-context or pitch-token "
            f"sampling -- parallel agents)."
        )
    return (
        f"INDETERMINATE: TVD={tvd:.4f}, KL={kl:.4f}.  Drift exists but is "
        f"either small ({tvd:.4f} < {TVD_CONFIRM_THRESH}) or in an "
        f"unexpected direction (consistent={direction_consistent}).  Need "
        f"a parallel agent's findings to triangulate."
    )


def _write_report(path: Path, payload: dict) -> None:
    args = payload["args"]
    om = payload["outcome_marginals"]
    dd = payload["distributional_distances"]
    rt = payload["rare_token_analysis"]
    cd = payload["calibration_drift"]
    ka = payload["k_pct_attribution_estimate"]
    chk = payload["checkpoints"]
    wc = payload["wall_clock_seconds"]

    lines: list[str] = []
    lines.append("# Phase 0.6 PitchGPT Diagnostic -- A1 Off-Distribution at Counterfactually-Sampled Pitches\n")
    lines.append(f"Generated: {payload['timestamp_utc']}\n")
    lines.append(f"Verdict: **{payload['verdict']}**\n\n")
    lines.append(payload["verdict_explanation"] + "\n")
    lines.append("")
    lines.append("## Configuration\n")
    lines.append(f"- n_pa requested: **{args['n_pa']}**, processed: **{args['n_pa_actually_processed']}**")
    lines.append(f"- n_samples per PA (rollout): **{args['n_samples']}**, horizon: **{args['horizon']}**, seed: {args['seed']}")
    lines.append(f"- Device: {payload['device']}")
    lines.append(f"- Backbone v2 SHA: `{chk['backbone_v2_sha256'][:16]}...`")
    lines.append(f"- A1 head SHA: `{chk['a1_head_sha256'][:16]}...`")
    lines.append(f"- A1 locked temperature: {chk['a1_temperature_locked']:.4f}")
    lines.append("")
    lines.append("## Wall clock\n")
    lines.append(f"- Pass A (in-distribution): {wc['pass_a_in_distribution']:.1f}s")
    lines.append(f"- Pass B (counterfactual rollout): {wc['pass_b_counterfactual']:.1f}s")
    lines.append(f"- Total: {wc['total']:.1f}s")
    lines.append("")
    lines.append("## Eval set sizes\n")
    lines.append(f"- In-distribution pitches scored: {payload['n_eval_rows']['in_distribution_pitches']:,}")
    lines.append(f"- Counterfactual sampled pitches scored: {payload['n_eval_rows']['counterfactual_pitches']:,}")
    lines.append("")
    lines.append("## A1 outcome marginal -- in-distribution vs counterfactual\n")
    lines.append("| Class | In-distribution | Counterfactual | Delta (pp) |")
    lines.append("|---|---|---|---|")
    for c in OUTCOME_CLASSES:
        in_v = om["in_distribution"][c]
        cf_v = om["counterfactual"][c]
        d_v = om["per_class_delta_pp"][c]
        lines.append(f"| {c} | {in_v:.4f} | {cf_v:.4f} | {d_v:+.4f} |")
    lines.append("")
    lines.append("## Distributional distances\n")
    lines.append(f"- TVD(in_dist, counterfactual) = **{dd['tvd']:.4f}** (threshold {dd['tvd_confirm_threshold']:.4f})")
    lines.append(f"- KL(in || cf) = **{dd['kl_in_to_cf']:.4f}** (threshold {dd['kl_confirm_threshold']:.4f})")
    lines.append(f"- KL(cf || in) = {dd['kl_cf_to_in']:.4f}")
    lines.append(f"- JS divergence = {dd['js_divergence']:.4f}")
    lines.append("")
    lines.append("## Rare-token off-distribution analysis\n")
    if rt["skipped"]:
        lines.append("(Skipped via --skip-train-freq.)")
    else:
        lines.append(f"- Train rare threshold: freq < {rt['train_rare_threshold']:.4f}")
        lines.append(f"- Rollout common threshold: freq >= {rt['rollout_common_threshold']:.4f}")
        lines.append(f"- N rare tokens (low train, high rollout): **{rt['n_rare_tokens']}**")
        lines.append(f"- N counterfactual eval rows on rare tokens: {rt['n_rare_eval_rows']:,}")
        lines.append(f"- N counterfactual eval rows on common tokens: {rt['n_common_eval_rows']:,}")
        lines.append(f"- TVD(rare_marg, common_marg) = **{rt['tvd_rare_vs_common']}**")
        if rt["rare_token_outcome_marginal"] is not None:
            lines.append("")
            lines.append("Rare-token A1 outcome marginal (counterfactual):")
            lines.append("")
            lines.append("| Class | Rare-token | Common-token | Delta (pp) |")
            lines.append("|---|---|---|---|")
            for c in OUTCOME_CLASSES:
                r = rt["rare_token_outcome_marginal"][c]
                co = rt["common_token_outcome_marginal"][c]
                lines.append(f"| {c} | {r:.4f} | {co:.4f} | {(r - co) * 100:+.4f} |")
        if rt["top_oversampled_tokens"]:
            lines.append("")
            lines.append("Top 15 over-sampled tokens (rollout - train, with rollout_freq >= 0.005):")
            lines.append("")
            lines.append("| Token | Train freq | Rollout freq | Delta |")
            lines.append("|---|---|---|---|")
            for r in rt["top_oversampled_tokens"]:
                lines.append(f"| {r['token']} | {r['train_freq']:.6f} | {r['rollout_freq']:.6f} | {r['delta']:+.6f} |")
    lines.append("")
    lines.append("## Calibration drift (in-distribution ECE with vs without temperature)\n")
    lines.append(f"- ECE in-distribution, T=0.8003 (production): **{cd['ece_in_distribution_T_locked']:.4f}**")
    lines.append(f"- ECE in-distribution, T=1.0 (no scaling):    **{cd['ece_in_distribution_T1']:.4f}**")
    lines.append("")
    lines.append(cd["note"])
    lines.append("")
    lines.append("## K% attribution estimate (rough)\n")
    lines.append(f"- In-distribution strike marginal (CS+SS, per-pitch): {ka['in_dist_strike_marginal']:.4f}")
    lines.append(f"- Counterfactual strike marginal (CS+SS, per-pitch):  {ka['counterfactual_strike_marginal']:.4f}")
    lines.append(f"- Delta strike marginal: **{ka['delta_strike_marginal_pp']:+.4f}pp**")
    lines.append(f"- Naive K% pp attribution (upper-bound order-of-magnitude): {ka['naive_k_pct_pp_attribution']:+.4f}pp")
    lines.append(f"- Phase 0.6 observed K% gap: +{ka['phase06_observed_k_pct_pp_gap']:.2f}pp")
    lines.append(f"- Naive fraction of observed gap explained: {ka['fraction_of_observed_gap_explained_naive']:.4f}")
    lines.append("")
    lines.append(ka["note"])
    lines.append("")
    lines.append("## Cross-references\n")
    lines.append("- Hypothesis source: PHASE_0.6_DIAGNOSIS or comparable diagnostic plan.")
    lines.append("- A1 head: `src/analytics/pitchgpt_outcome_head.py::FrozenOutcomeHeadConcat`")
    lines.append("- Rollout: `src/analytics/pitchgpt_sim.py::rollout`")
    lines.append("- Phase 0.6 gate-fail metrics: `results/pitchgpt/rollout_sanity_2025/metrics.json`")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
