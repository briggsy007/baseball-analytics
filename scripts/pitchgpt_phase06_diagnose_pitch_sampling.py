"""
Phase 0.6 PitchGPT rollout-sanity FAIL diagnostic agent #1
==========================================================

Hypothesis under test:
    T=1.0 pitch-token sampling over-emits in-strike-zone tokens at H=0
    (first pitch of PA), which accumulates to strikes faster than actual
    MLB pitchers and explains the +8.9pp K% bias / -4.5pp BB% under-bias /
    -0.66 mean-PA-length deficit observed in
    ``results/pitchgpt/rollout_sanity_2025/metrics.json``.

Methodology (verifies/falsifies the hypothesis WITHOUT mutating the rollout):

    1. Re-fetch the same 10K PA cohort (seed=42) using the cohort-fetcher
       in ``scripts/pitchgpt_rollout_sanity_2025.py``.

    2. Marginal H=0 pitch-token distribution: for each PA-start, sample
       n_samples=100 first pitches via ``rollout(..., horizon=1, T=1.0)``;
       aggregate the position-0 token over all 10K PAs.  Compute TVD / KL
       vs the empirical 2025 first-pitch distribution.

       Sub-decomposition: per-token decompose into
       (pitch_type, zone, velo_bucket).  Report the sampled vs empirical
       marginal over **zone** separately.  Empirical 2025 first-pitch
       zone marginal is computed from ``pitches.zone`` filtered to
       ``pitch_number=1`` -- BUT note that Statcast's ``zone`` is
       1-9 (inner) + 11-14 (outer), while the PitchTokenizer uses a 5x5
       grid 0-24 + 25 missing.  We compute the in-zone proportion under
       both conventions:

         * Statcast: zones {1..9} = in-zone, {11..14} = out-of-zone.
         * PitchTokenizer: ``_IN_ZONE_INDICES`` = {6,7,8,11,12,13,16,17,18}
           (inner 3x3 of the 5x5 grid) — same heuristic the rollout's
           outcome_predictor=None fallback uses.

    3. Strike-zone hit-rate: in-zone vs out-of-zone proportion at H=0,
       sampled vs empirical.  Hypothesis: sampled in-zone proportion >
       empirical in-zone proportion at T=1.0.

    4. Strike-call simulator: for each pitch (sampled and empirical),
       apply the count-only None-predictor heuristic
       (in-zone → +1 strike, otherwise → +1 ball) over a synthetic 6-pitch
       PA assuming i.i.d. draws from the H=0 marginal.  This is a
       worst-case proxy that isolates JUST the marginal-sampling effect
       from the outcome predictor.  Compute implied K%/BB% under both
       distributions; the difference quantifies how much of the +8.9pp
       K% bias is attributable to pitch-token sampling alone.

    5. Temperature sensitivity: re-run steps 2-3 at T=0.8, 1.0, 1.2 with
       n_pa=2000 (budget guard).  T=1.2 (more entropy) should reduce the
       in-zone bias if T=1.0 is the issue.

    6. Write
       ``results/pitchgpt/rollout_sanity_2025/diagnose_pitch_sampling.json``
       and ``.md`` with a verdict (CONFIRMED / FALSIFIED / INDETERMINATE).

Guardrails
----------
* No model checkpoint / no ``pitchgpt_sim.py`` / no ``pitchgpt.py`` /
  no ``pitchgpt_outcome_head.py`` modification.
* DuckDB read_only=True throughout.
* No commits.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Reuse the cohort-fetcher exactly (seed=42 path) so we don't drift.
from scripts.pitchgpt_rollout_sanity_2025 import (  # noqa: E402
    HOLDOUT_SEASON,
    N_PA_DEFAULT,
    SEED_ROOT,
    _fetch_2025_league_median_ump_scalar,
    _fetch_2025_pitcher_disjoint_pa_starts,
    _fetch_train_pitcher_ids,
)
from src.analytics.pitchgpt import (  # noqa: E402
    NUM_PITCH_TYPES,
    NUM_VELO_BUCKETS,
    NUM_ZONES,
    PITCH_TYPE_MAP,
    PitchTokenizer,
    VOCAB_SIZE,
)
from src.analytics.pitchgpt_sim import (  # noqa: E402
    PAContext,
    rollout,
    _IN_ZONE_INDICES,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase06_diagnose_pitch_sampling")


# Statcast zone semantics:
#   1-9   = inner 3x3 strike zone (top-row 1,2,3 ; mid 4,5,6 ; bot 7,8,9)
#   11-14 = outer four (gap, low, etc.)
STATCAST_IN_ZONE = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 9})
STATCAST_OUT_ZONE = frozenset({11, 12, 13, 14})

# PitchTokenizer in-zone (inner 3x3 of the 5x5 grid) — this is the
# convention the rollout's count-only fallback uses.
TOKENIZER_IN_ZONE = frozenset(_IN_ZONE_INDICES)
TOKENIZER_OUT_ZONE = frozenset(set(range(NUM_ZONES)) - TOKENIZER_IN_ZONE)


# ─────────────────────────────────────────────────────────────────────────────
# Empirical first-pitch helpers
# ─────────────────────────────────────────────────────────────────────────────


def _fetch_empirical_first_pitch_zone_marginal(conn) -> dict:
    """Statcast first-pitch zone distribution over 2025 (full season).

    Uses ``pitches.zone`` -- the Statcast 1-9 (inner) + 11-14 (outer) labels.

    Returns:
        {
            "zone_counts": dict[int, int],
            "n_total": int,
            "p_in_zone_statcast": float,   -- {1..9} / total
            "p_out_zone_statcast": float,  -- {11..14} / total
            "p_missing": float,            -- NULLs / total
        }
    """
    df = conn.execute(
        f"""
        SELECT zone, COUNT(*) AS n
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) = {HOLDOUT_SEASON}
          AND pitch_number = 1
          AND pitch_type IS NOT NULL
        GROUP BY zone
        """
    ).fetchdf()
    n_total = int(df["n"].sum())
    counts: dict[int, int] = {}
    n_missing = 0
    for _, r in df.iterrows():
        z = r["zone"]
        if pd.isna(z):
            n_missing += int(r["n"])
            continue
        counts[int(z)] = int(r["n"])

    n_in = sum(counts.get(z, 0) for z in STATCAST_IN_ZONE)
    n_out = sum(counts.get(z, 0) for z in STATCAST_OUT_ZONE)
    return {
        "zone_counts": counts,
        "n_total": n_total,
        "n_missing": n_missing,
        "p_in_zone_statcast": n_in / max(n_total, 1),
        "p_out_zone_statcast": n_out / max(n_total, 1),
        "p_missing": n_missing / max(n_total, 1),
    }


def _fetch_empirical_first_pitch_token_marginal(conn) -> tuple[np.ndarray, dict]:
    """Build a length-VOCAB_SIZE empirical marginal over PitchTokenizer's
    composite token by reading first-pitch (plate_x, plate_z, release_speed,
    pitch_type) from 2025 and applying ``PitchTokenizer.encode``.

    Also returns a per-(pitch_type, zone, velo_bucket) decomposed marginal
    so we can compare the rollout's pitch-type / zone / velo distributions.
    """
    df = conn.execute(
        f"""
        SELECT pitch_type, plate_x, plate_z, release_speed
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) = {HOLDOUT_SEASON}
          AND pitch_number = 1
          AND pitch_type IS NOT NULL
        """
    ).fetchdf()
    n = len(df)
    counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
    pt_counts = np.zeros(NUM_PITCH_TYPES, dtype=np.int64)
    zone_counts = np.zeros(NUM_ZONES, dtype=np.int64)
    velo_counts = np.zeros(NUM_VELO_BUCKETS, dtype=np.int64)
    for _, r in df.iterrows():
        pt = r["pitch_type"]
        px = r["plate_x"]
        pz = r["plate_z"]
        rs = r["release_speed"]
        # Coerce NA to None / NaN per encode contract.
        px_v = float(px) if px is not None and not pd.isna(px) else None
        pz_v = float(pz) if pz is not None and not pd.isna(pz) else None
        rs_v = float(rs) if rs is not None and not pd.isna(rs) else None
        tok = PitchTokenizer.encode(pt, px_v, pz_v, rs_v)
        if 0 <= tok < VOCAB_SIZE:
            counts[tok] += 1
            decoded = PitchTokenizer.decode(tok)
            pt_counts[PitchTokenizer.pitch_type_to_idx(pt)] += 1
            zone_counts[decoded["zone"]] += 1
            velo_counts[decoded["velo_bucket"]] += 1
    p_token = counts.astype(np.float64) / max(counts.sum(), 1)
    decomp = {
        "pitch_type_counts": {
            PitchTokenizer.idx_to_pitch_type(i): int(c)
            for i, c in enumerate(pt_counts)
            if c > 0
        },
        "pitch_type_marginal": (pt_counts / max(pt_counts.sum(), 1)).tolist(),
        "zone_counts": zone_counts.tolist(),
        "zone_marginal": (zone_counts / max(zone_counts.sum(), 1)).tolist(),
        "velo_bucket_counts": velo_counts.tolist(),
        "velo_bucket_marginal": (velo_counts / max(velo_counts.sum(), 1)).tolist(),
        "n_pitches": int(n),
    }
    # In-zone via tokenizer convention:
    in_zone_tok = sum(zone_counts[z] for z in TOKENIZER_IN_ZONE)
    out_zone_tok = sum(zone_counts[z] for z in TOKENIZER_OUT_ZONE if z != NUM_ZONES - 1)
    missing_tok = int(zone_counts[NUM_ZONES - 1])
    decomp["p_in_zone_tokenizer"] = float(in_zone_tok) / max(int(zone_counts.sum()), 1)
    decomp["p_out_zone_tokenizer"] = float(out_zone_tok) / max(int(zone_counts.sum()), 1)
    decomp["p_missing_tokenizer"] = float(missing_tok) / max(int(zone_counts.sum()), 1)
    return p_token, decomp


# ─────────────────────────────────────────────────────────────────────────────
# Distribution comparison helpers
# ─────────────────────────────────────────────────────────────────────────────


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two discrete distributions."""
    return float(0.5 * np.sum(np.abs(p - q)))


def _kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) -- p is the "true" / sampled, q is the reference."""
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(p_safe * np.log(p_safe / q_safe)))


def _normalize(c: np.ndarray) -> np.ndarray:
    s = float(c.sum())
    return c.astype(np.float64) / max(s, 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# Rollout sampling at H=1 across cohort
# ─────────────────────────────────────────────────────────────────────────────


def _sample_h0_marginal(
    df_sample: pd.DataFrame,
    *,
    league_median_ump: float,
    n_samples: int,
    temperature: float,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Run rollout(horizon=1) for every PA-start in df_sample and aggregate.

    Returns:
        (token_counts, decomposed_counts)  where
        token_counts is shape (VOCAB_SIZE,) int64
        decomposed_counts is dict with pitch_type/zone/velo_bucket counts.
    """
    n_pa = len(df_sample)
    counts = np.zeros(VOCAB_SIZE, dtype=np.int64)
    pt_counts = np.zeros(NUM_PITCH_TYPES, dtype=np.int64)
    zone_counts = np.zeros(NUM_ZONES, dtype=np.int64)
    velo_counts = np.zeros(NUM_VELO_BUCKETS, dtype=np.int64)
    log_every = max(n_pa // 10, 100)
    t0 = time.perf_counter()
    for i, row in df_sample.iterrows():
        ctx = PAContext.from_pitches_row(row, ump_scalar=league_median_ump)
        per_row_seed = (seed * 1_000_003 + i) & 0x7FFFFFFF
        # horizon=1 -> position-0 token only
        result = rollout(
            ctx,
            outcome_predictor=None,  # we only need pitch-token sampling
            n_samples=n_samples,
            horizon=1,
            temperature=temperature,
            seed=per_row_seed,
            return_probs=False,
        )
        toks = result.pitch_tokens[:, 0]  # (n_samples,) int64
        # Build a histogram bumping on the fly; toks should be in [0, VOCAB_SIZE)
        # for non-truncated samples (and at H=1 nothing is truncated yet).
        valid = (toks >= 0) & (toks < VOCAB_SIZE)
        valid_toks = toks[valid]
        if valid_toks.size > 0:
            bc = np.bincount(valid_toks, minlength=VOCAB_SIZE)
            counts += bc
            # decompose
            for tok in valid_toks:
                tok_int = int(tok)
                velo = tok_int % NUM_VELO_BUCKETS
                rem = tok_int // NUM_VELO_BUCKETS
                zone = rem % NUM_ZONES
                pt = rem // NUM_ZONES
                pt_counts[pt] += 1
                zone_counts[zone] += 1
                velo_counts[velo] += 1
        if (i + 1) % log_every == 0:
            dt = time.perf_counter() - t0
            rate = (i + 1) / dt
            remain = (n_pa - (i + 1)) / max(rate, 1e-6)
            logger.info(
                "  H=0 sampling T=%.2f  %d/%d PAs  rate=%.1f PA/s  eta=%.0fs",
                temperature, i + 1, n_pa, rate, remain,
            )
    decomp = {
        "pitch_type_counts": {
            PitchTokenizer.idx_to_pitch_type(i): int(c)
            for i, c in enumerate(pt_counts)
            if c > 0
        },
        "pitch_type_marginal": (pt_counts / max(pt_counts.sum(), 1)).tolist(),
        "zone_counts": zone_counts.tolist(),
        "zone_marginal": (zone_counts / max(zone_counts.sum(), 1)).tolist(),
        "velo_bucket_counts": velo_counts.tolist(),
        "velo_bucket_marginal": (velo_counts / max(velo_counts.sum(), 1)).tolist(),
        "n_pitches": int(counts.sum()),
    }
    in_zone_tok = sum(zone_counts[z] for z in TOKENIZER_IN_ZONE)
    out_zone_tok = sum(zone_counts[z] for z in TOKENIZER_OUT_ZONE if z != NUM_ZONES - 1)
    missing_tok = int(zone_counts[NUM_ZONES - 1])
    total_z = max(int(zone_counts.sum()), 1)
    decomp["p_in_zone_tokenizer"] = float(in_zone_tok) / total_z
    decomp["p_out_zone_tokenizer"] = float(out_zone_tok) / total_z
    decomp["p_missing_tokenizer"] = float(missing_tok) / total_z
    return counts, decomp


# ─────────────────────────────────────────────────────────────────────────────
# Strike-call simulator (count-only None-predictor heuristic)
# ─────────────────────────────────────────────────────────────────────────────


def _heuristic_implied_kbb_from_inzone_rate(p_in_zone: float, horizon: int = 6) -> dict:
    """Closed-form K%/BB% under the None-predictor count-only heuristic
    given an i.i.d. in-zone probability ``p_in_zone``.

    For each pitch the heuristic adds:
        * +1 strike with prob p_in_zone
        * +1 ball  with prob (1 - p_in_zone)

    PA terminates when balls reaches 4 (BB) or strikes reaches 3 (K) within
    ``horizon`` pitches; otherwise truncated.

    We close-form by enumerating PA paths in (B, S) space.  This abstracts
    away foul-on-2-strikes (the rollout's foul case) — accepted because
    the None-predictor treats every in-zone pitch as a strike, no fouls.
    Result: a clean upper bound on the K% / BB% bias attributable to
    just the H=0 in-zone rate at T=1.0.
    """
    p = float(p_in_zone)
    q = 1.0 - p

    # State (b, s); start (0,0); transitions to (b, s+1) w/ prob p, (b+1, s)
    # w/ prob q.  Walk on b=4, K on s=3, truncated otherwise after `horizon`.
    state_probs = {(0, 0): 1.0}
    p_k_terminal = 0.0
    p_bb_terminal = 0.0
    for _ in range(horizon):
        next_state: dict[tuple[int, int], float] = {}
        for (b, s), prob in state_probs.items():
            # +strike
            sb, ss = b, s + 1
            if ss >= 3:
                p_k_terminal += prob * p
            else:
                next_state[(sb, ss)] = next_state.get((sb, ss), 0.0) + prob * p
            # +ball
            sb2, ss2 = b + 1, s
            if sb2 >= 4:
                p_bb_terminal += prob * q
            else:
                next_state[(sb2, ss2)] = next_state.get((sb2, ss2), 0.0) + prob * q
        state_probs = next_state
    p_truncated = float(sum(state_probs.values()))
    return {
        "p_in_zone": p,
        "horizon": int(horizon),
        "implied_k_pct": float(p_k_terminal),
        "implied_bb_pct": float(p_bb_terminal),
        "implied_truncated_pct": p_truncated,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pa", type=int, default=N_PA_DEFAULT)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument(
        "--n-pa-temp-sweep",
        type=int,
        default=2000,
        help="n_pa for the temperature sweep (each of T=0.8 / 1.2)",
    )
    parser.add_argument("--seed", type=int, default=SEED_ROOT)
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "results" / "pitchgpt" / "rollout_sanity_2025",
    )
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=45 * 60.0,
        help="Hard time budget; if main run plus temp sweep would exceed, scale down.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    t_start_overall = time.perf_counter()
    scope_cuts: list[str] = []

    # ── 1. Cohort ────────────────────────────────────────────────────────
    logger.info("Phase 0.6 diagnose pitch-sampling start (n_pa=%d, n_samples=%d, seed=%d)",
                args.n_pa, args.n_samples, args.seed)
    conn = get_connection(args.db_path, read_only=True)
    try:
        train_pitchers = _fetch_train_pitcher_ids(conn)
        df_pas = _fetch_2025_pitcher_disjoint_pa_starts(conn, train_pitchers)
        league_median_ump = _fetch_2025_league_median_ump_scalar(conn)
        # Empirical first-pitch token marginal (full 2025 first-pitch cohort).
        logger.info("Computing empirical first-pitch zone marginal (Statcast)...")
        empirical_zone_statcast = _fetch_empirical_first_pitch_zone_marginal(conn)
        logger.info("Computing empirical first-pitch token marginal (PitchTokenizer)...")
        empirical_token_marginal, empirical_decomp = (
            _fetch_empirical_first_pitch_token_marginal(conn)
        )
    finally:
        conn.close()

    n_total_pas = len(df_pas)
    if n_total_pas < args.n_pa:
        raise RuntimeError(
            f"Only {n_total_pas} PAs available, need {args.n_pa}"
        )

    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(n_total_pas, size=args.n_pa, replace=False)
    df_sample = df_pas.iloc[sample_idx].reset_index(drop=True)
    # Same NA-coercion as the main rollout-sanity script.
    for col in ("on_1b", "on_2b", "on_3b"):
        if col in df_sample.columns:
            df_sample[col] = (
                pd.to_numeric(df_sample[col], errors="coerce").fillna(0).astype(np.int64)
            )
    for col in ("balls", "strikes", "outs_when_up", "inning",
                "pitcher_id", "batter_id"):
        if col in df_sample.columns:
            df_sample[col] = (
                pd.to_numeric(df_sample[col], errors="coerce").fillna(0).astype(np.int64)
            )

    # ── 2. MAIN run: T=1.0 H=0 marginal sampling ─────────────────────────
    logger.info("MAIN H=0 sampling at T=1.0 (n_pa=%d, n_samples=%d)...",
                args.n_pa, args.n_samples)
    t0 = time.perf_counter()
    sampled_counts_t10, sampled_decomp_t10 = _sample_h0_marginal(
        df_sample,
        league_median_ump=league_median_ump,
        n_samples=args.n_samples,
        temperature=1.0,
        seed=args.seed,
    )
    dt_main = time.perf_counter() - t0
    logger.info("MAIN H=0 sampling done in %.1fs (%d total H=0 pitches)",
                dt_main, int(sampled_counts_t10.sum()))

    # Distribution comparisons
    sampled_p_t10 = _normalize(sampled_counts_t10)
    tvd_token_t10 = _tvd(sampled_p_t10, empirical_token_marginal)
    kl_token_t10_se = _kl(sampled_p_t10, empirical_token_marginal)
    kl_token_t10_es = _kl(empirical_token_marginal, sampled_p_t10)

    # Marginal over zone (PitchTokenizer convention, 0-25)
    emp_zone = np.asarray(empirical_decomp["zone_marginal"], dtype=np.float64)
    sam_zone_t10 = np.asarray(sampled_decomp_t10["zone_marginal"], dtype=np.float64)
    tvd_zone_t10 = _tvd(sam_zone_t10, emp_zone)
    kl_zone_t10 = _kl(sam_zone_t10, emp_zone)

    emp_pt = np.asarray(empirical_decomp["pitch_type_marginal"], dtype=np.float64)
    sam_pt_t10 = np.asarray(sampled_decomp_t10["pitch_type_marginal"], dtype=np.float64)
    tvd_pt_t10 = _tvd(sam_pt_t10, emp_pt)

    emp_velo = np.asarray(empirical_decomp["velo_bucket_marginal"], dtype=np.float64)
    sam_velo_t10 = np.asarray(sampled_decomp_t10["velo_bucket_marginal"], dtype=np.float64)
    tvd_velo_t10 = _tvd(sam_velo_t10, emp_velo)

    # Zone-hit rate (in-zone vs out-of-zone), PitchTokenizer convention.
    in_zone_emp_tok = float(empirical_decomp["p_in_zone_tokenizer"])
    in_zone_sam_t10 = float(sampled_decomp_t10["p_in_zone_tokenizer"])
    delta_in_zone_t10 = in_zone_sam_t10 - in_zone_emp_tok

    # Heuristic-implied K%/BB% from in-zone rate (both empirical and sampled).
    horizon_kbb = 6
    heur_emp_tok = _heuristic_implied_kbb_from_inzone_rate(in_zone_emp_tok, horizon=horizon_kbb)
    heur_emp_statcast = _heuristic_implied_kbb_from_inzone_rate(
        empirical_zone_statcast["p_in_zone_statcast"], horizon=horizon_kbb,
    )
    heur_sam_t10 = _heuristic_implied_kbb_from_inzone_rate(
        in_zone_sam_t10, horizon=horizon_kbb,
    )

    # ── 3. Temperature sweep at T=0.8, 1.2 (n_pa=2000) ───────────────────
    elapsed = time.perf_counter() - t_start_overall
    remaining_budget = args.time_budget_seconds - elapsed
    n_pa_temp = args.n_pa_temp_sweep
    # Estimate time for one temperature run from main elapsed.  Main ran
    # n_pa rows; one temp run is n_pa_temp rows -> linearly-scaled estimate.
    est_per_temp = dt_main * n_pa_temp / max(args.n_pa, 1)
    needed_for_temp_sweep = est_per_temp * 2  # T=0.8 + T=1.2
    if needed_for_temp_sweep > remaining_budget:
        old_n = n_pa_temp
        n_pa_temp = max(500, int(remaining_budget / max(est_per_temp / n_pa_temp * 2, 1e-9)))
        if n_pa_temp < old_n:
            scope_cuts.append(
                f"Temperature-sweep n_pa scaled from {old_n} -> {n_pa_temp} "
                f"to fit remaining wall-clock budget ({remaining_budget:.0f}s)."
            )
            logger.warning(scope_cuts[-1])

    df_sample_temp = df_sample.iloc[:n_pa_temp].reset_index(drop=True)

    sweep_results: dict[float, dict] = {}
    for T in (0.8, 1.2):
        elapsed_now = time.perf_counter() - t_start_overall
        if elapsed_now > args.time_budget_seconds * 0.95:
            scope_cuts.append(
                f"Skipped T={T} — wall-clock budget exhausted ({elapsed_now:.0f}s)."
            )
            logger.warning(scope_cuts[-1])
            continue
        logger.info("Temperature-sweep run T=%.2f (n_pa=%d)...", T, n_pa_temp)
        t1 = time.perf_counter()
        # Distinct seed per T so RNG streams don't alias.
        sw_counts, sw_decomp = _sample_h0_marginal(
            df_sample_temp,
            league_median_ump=league_median_ump,
            n_samples=args.n_samples,
            temperature=T,
            seed=int(args.seed + 1000 * (T * 100)),
        )
        dt_t = time.perf_counter() - t1
        sw_p = _normalize(sw_counts)
        sw_zone = np.asarray(sw_decomp["zone_marginal"], dtype=np.float64)
        in_zone_sw = float(sw_decomp["p_in_zone_tokenizer"])
        sweep_results[T] = {
            "wall_clock_seconds": round(dt_t, 1),
            "n_pa": n_pa_temp,
            "tvd_token_vs_empirical": round(_tvd(sw_p, empirical_token_marginal), 6),
            "tvd_zone_vs_empirical": round(_tvd(sw_zone, emp_zone), 6),
            "p_in_zone_tokenizer": round(in_zone_sw, 6),
            "delta_in_zone_vs_empirical": round(in_zone_sw - in_zone_emp_tok, 6),
            "heuristic_implied_kbb": _heuristic_implied_kbb_from_inzone_rate(
                in_zone_sw, horizon=horizon_kbb
            ),
        }
        logger.info("  T=%.2f  in_zone=%.4f  delta vs emp=%+.4f",
                    T, in_zone_sw, in_zone_sw - in_zone_emp_tok)

    # ── 4. Verdict ────────────────────────────────────────────────────────
    # Phase 0.6 metrics from metrics.json:
    #   K%   sampled = 0.3068, empirical = 0.2180  -> bias = +8.88 pp
    #   BB%  sampled = 0.0430, empirical = 0.0876  -> bias = -4.46 pp
    #   PA-len sampled = 3.226, empirical = 3.886  -> -0.66 pitches
    #
    # Hypothesis: T=1.0 over-emits in-zone tokens.
    # Verdict logic:
    #
    #   * Sampled in-zone rate > empirical (tokenizer convention) AND
    #     |delta_in_zone| > 0.02 (i.e., >2pp absolute over-emission), AND
    #     the heuristic-implied K% bias from sampled-vs-empirical in-zone
    #     rates (with horizon=6 i.i.d. heuristic) is NOT negligible
    #     (>= 0.5pp of the +8.88pp K% gap)  ->  CONFIRMED.
    #
    #   * Sampled in-zone rate <= empirical (or within 0.005 of it)  ->
    #     FALSIFIED: no in-zone over-emission.
    #
    #   * Otherwise (small but non-trivial in-zone bias whose K% impact
    #     is mostly inside noise)  ->  INDETERMINATE, with the magnitude
    #     reported in any case.
    observed_k_bias_pp = 0.30681 - 0.21797   # from metrics.json (PRIMARY vs empirical)
    observed_bb_bias_pp = 0.04299 - 0.08764
    heur_k_gap_pp_t10 = heur_sam_t10["implied_k_pct"] - heur_emp_tok["implied_k_pct"]
    heur_bb_gap_pp_t10 = heur_sam_t10["implied_bb_pct"] - heur_emp_tok["implied_bb_pct"]

    if delta_in_zone_t10 > 0.02 and heur_k_gap_pp_t10 >= 0.005:
        verdict = "CONFIRMED"
    elif delta_in_zone_t10 <= 0.005:
        verdict = "FALSIFIED"
    else:
        verdict = "INDETERMINATE"

    # Estimated K% bias attributable to pitch sampling.
    # heuristic_implied_K%(sampled in-zone) - heuristic_implied_K%(empirical in-zone).
    # Cap by the observed K% bias so we don't claim attribution exceeding
    # the actual phenomenon.
    attributable_k_bias_pp = max(min(heur_k_gap_pp_t10, observed_k_bias_pp), 0.0)
    attributable_bb_bias_pp = min(max(heur_bb_gap_pp_t10, observed_bb_bias_pp), 0.0)
    attribution_pct = (
        100.0 * attributable_k_bias_pp / max(observed_k_bias_pp, 1e-9)
    )

    elapsed_total = time.perf_counter() - t_start_overall

    # Sanity-check sampled distribution lift over flat:
    sampled_zone_in_diff_vs_uniform = in_zone_sam_t10 - (9.0 / 26.0)

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec": "Phase 0.6 diagnostic: pitch-token sampling hypothesis",
        "args": {
            "n_pa_main": args.n_pa,
            "n_samples": args.n_samples,
            "n_pa_temp_sweep": n_pa_temp,
            "seed": args.seed,
            "horizon_for_kbb_heuristic": horizon_kbb,
        },
        "wall_clock_seconds": round(elapsed_total, 1),
        "scope_cuts": scope_cuts,
        "verdict": verdict,
        "headline": {
            "observed_k_bias_pp": round(observed_k_bias_pp * 100, 4),
            "observed_bb_bias_pp": round(observed_bb_bias_pp * 100, 4),
            "delta_in_zone_t10_tokenizer_pp": round(delta_in_zone_t10 * 100, 4),
            "heuristic_implied_k_gap_t10_pp": round(heur_k_gap_pp_t10 * 100, 4),
            "heuristic_implied_bb_gap_t10_pp": round(heur_bb_gap_pp_t10 * 100, 4),
            "attributable_k_bias_pp": round(attributable_k_bias_pp * 100, 4),
            "attributable_bb_bias_pp": round(attributable_bb_bias_pp * 100, 4),
            "attribution_pct_of_observed_k_bias": round(attribution_pct, 2),
        },
        "empirical_2025_first_pitch": {
            "statcast_zone": {
                "p_in_zone": round(empirical_zone_statcast["p_in_zone_statcast"], 6),
                "p_out_zone": round(empirical_zone_statcast["p_out_zone_statcast"], 6),
                "p_missing": round(empirical_zone_statcast["p_missing"], 6),
                "n_total": empirical_zone_statcast["n_total"],
                "zone_counts": {
                    str(k): int(v) for k, v in empirical_zone_statcast["zone_counts"].items()
                },
            },
            "tokenizer": {
                "p_in_zone": round(empirical_decomp["p_in_zone_tokenizer"], 6),
                "p_out_zone": round(empirical_decomp["p_out_zone_tokenizer"], 6),
                "p_missing": round(empirical_decomp["p_missing_tokenizer"], 6),
                "n_pitches": empirical_decomp["n_pitches"],
                "pitch_type_counts": empirical_decomp["pitch_type_counts"],
                "zone_counts": empirical_decomp["zone_counts"],
                "velo_bucket_counts": empirical_decomp["velo_bucket_counts"],
            },
        },
        "sampled_t10": {
            "n_pa": args.n_pa,
            "n_samples": args.n_samples,
            "n_total_pitches": int(sampled_counts_t10.sum()),
            "tvd_token_vs_empirical": round(tvd_token_t10, 6),
            "kl_sampled_vs_empirical_token": round(kl_token_t10_se, 6),
            "kl_empirical_vs_sampled_token": round(kl_token_t10_es, 6),
            "tvd_zone_vs_empirical": round(tvd_zone_t10, 6),
            "kl_sampled_vs_empirical_zone": round(kl_zone_t10, 6),
            "tvd_pitch_type_vs_empirical": round(tvd_pt_t10, 6),
            "tvd_velo_bucket_vs_empirical": round(tvd_velo_t10, 6),
            "p_in_zone_tokenizer": round(in_zone_sam_t10, 6),
            "p_in_zone_minus_empirical": round(delta_in_zone_t10, 6),
            "p_in_zone_minus_uniform_grid": round(sampled_zone_in_diff_vs_uniform, 6),
            "pitch_type_counts": sampled_decomp_t10["pitch_type_counts"],
            "zone_counts": sampled_decomp_t10["zone_counts"],
            "velo_bucket_counts": sampled_decomp_t10["velo_bucket_counts"],
            "wall_clock_seconds": round(dt_main, 1),
        },
        "heuristic_implied_kbb": {
            "horizon": horizon_kbb,
            "note": (
                "Closed-form K%/BB% under count-only None-predictor heuristic "
                "(in-zone -> +1 strike, otherwise +1 ball) given i.i.d. H=0 "
                "in-zone draw probability.  Worst-case proxy that ISOLATES the "
                "marginal-sampling effect from the outcome predictor."
            ),
            "empirical_tokenizer_in_zone_rate": heur_emp_tok,
            "empirical_statcast_in_zone_rate": heur_emp_statcast,
            "sampled_t10_in_zone_rate": heur_sam_t10,
        },
        "temperature_sweep": {
            "T": {str(t): r for t, r in sweep_results.items()},
            "T_1.0_in_zone": round(in_zone_sam_t10, 6),
            "comparison": {
                str(t): {
                    "delta_in_zone_vs_T_1.0": round(
                        sweep_results[t]["p_in_zone_tokenizer"] - in_zone_sam_t10, 6,
                    ),
                }
                for t in sweep_results
            },
        },
        "conventions": {
            "statcast_in_zone": "zones {1,..,9}",
            "statcast_out_zone": "zones {11,..,14}",
            "tokenizer_in_zone": "PitchTokenizer 5x5 inner-3x3 indices "
            f"{sorted(TOKENIZER_IN_ZONE)} (matches src/analytics/pitchgpt_sim.py "
            "_IN_ZONE_INDICES, used by None-predictor count-only fallback).",
            "tokenizer_out_zone": "all other 5x5 indices except 25 (missing).",
        },
    }

    # ── 5. Write outputs ──────────────────────────────────────────────────
    json_path = args.output_dir / "diagnose_pitch_sampling.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote %s", json_path)

    md_path = args.output_dir / "diagnose_pitch_sampling.md"
    _write_report(md_path, payload)
    logger.info("Wrote %s", md_path)

    print()
    print("=" * 78)
    print("Phase 0.6 -- Diagnose pitch-token sampling -- Summary")
    print("=" * 78)
    print(f"Verdict: {verdict}")
    print(f"Wall clock: {elapsed_total:.1f}s")
    print(f"Observed K% bias       : {observed_k_bias_pp * 100:+.2f} pp")
    print(f"Observed BB% bias      : {observed_bb_bias_pp * 100:+.2f} pp")
    print(f"Empirical in-zone (tok): {in_zone_emp_tok:.4f}  (Statcast: {empirical_zone_statcast['p_in_zone_statcast']:.4f})")
    print(f"Sampled in-zone (T=1.0): {in_zone_sam_t10:.4f}  (delta {delta_in_zone_t10 * 100:+.2f} pp)")
    print(f"Token TVD (T=1.0)      : {tvd_token_t10:.4f}")
    print(f"Token KL (T=1.0)       : sampled||emp={kl_token_t10_se:.4f}  emp||sampled={kl_token_t10_es:.4f}")
    print(f"Heuristic-implied K%   : sampled={heur_sam_t10['implied_k_pct']:.4f}  emp={heur_emp_tok['implied_k_pct']:.4f}  (delta {heur_k_gap_pp_t10 * 100:+.2f} pp)")
    print(f"Heuristic-implied BB%  : sampled={heur_sam_t10['implied_bb_pct']:.4f}  emp={heur_emp_tok['implied_bb_pct']:.4f}  (delta {heur_bb_gap_pp_t10 * 100:+.2f} pp)")
    print(f"Attribution: {attributable_k_bias_pp * 100:+.2f}pp K% ({attribution_pct:.1f}% of observed +8.88pp)")
    if scope_cuts:
        print("Scope cuts:")
        for sc in scope_cuts:
            print(f"  - {sc}")
    print("=" * 78)
    return 0


def _write_report(path: Path, payload: dict) -> None:
    h = payload["headline"]
    e = payload["empirical_2025_first_pitch"]
    s = payload["sampled_t10"]
    heur = payload["heuristic_implied_kbb"]
    sweep = payload["temperature_sweep"]

    lines: list[str] = []
    lines.append("# Phase 0.6 PitchGPT Rollout Sanity FAIL -- Diagnostic #1: Pitch-Token Sampling\n")
    lines.append(f"Generated: {payload['timestamp_utc']}")
    lines.append(f"Wall clock: {payload['wall_clock_seconds']:.1f}s")
    lines.append("")
    lines.append("## Hypothesis under test\n")
    lines.append(
        "T=1.0 pitch-token sampling over-emits in-strike-zone tokens at H=0 "
        "(first pitch of PA), which accumulates to strikes faster than actual "
        "MLB pitchers and explains the +8.9pp K% bias / -4.5pp BB% under-bias / "
        "-0.66 mean-PA-length deficit observed in `metrics.json`."
    )
    lines.append("")
    lines.append(f"## Verdict: **{payload['verdict']}**\n")
    lines.append("")
    lines.append("## Headline numbers\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Observed K% bias (PRIMARY vs empirical, from metrics.json) | **{h['observed_k_bias_pp']:+.2f} pp** |")
    lines.append(f"| Observed BB% bias (PRIMARY vs empirical, from metrics.json) | **{h['observed_bb_bias_pp']:+.2f} pp** |")
    lines.append(f"| In-zone delta sampled@T=1.0 vs empirical (tokenizer 5x5 inner-3x3) | **{h['delta_in_zone_t10_tokenizer_pp']:+.2f} pp** |")
    lines.append(f"| Heuristic-implied K% gap (sampled - empirical, H=6 i.i.d.) | **{h['heuristic_implied_k_gap_t10_pp']:+.2f} pp** |")
    lines.append(f"| Heuristic-implied BB% gap (sampled - empirical, H=6 i.i.d.) | **{h['heuristic_implied_bb_gap_t10_pp']:+.2f} pp** |")
    lines.append(f"| Attributable K% bias to pitch-token sampling | **{h['attributable_k_bias_pp']:+.2f} pp** ({h['attribution_pct_of_observed_k_bias']:.1f}% of observed +{h['observed_k_bias_pp']:.2f}pp) |")
    lines.append(f"| Attributable BB% bias to pitch-token sampling | **{h['attributable_bb_bias_pp']:+.2f} pp** |")
    lines.append("")
    lines.append("## Convention note (load-bearing)\n")
    lines.append("Statcast's `pitches.zone` column uses a **non-tokenizer** scheme:")
    lines.append("- `zone in {1..9}` = inner 3x3 strike zone (Statcast's strict zones)")
    lines.append("- `zone in {11..14}` = outer four (gap / low / etc.)")
    lines.append("")
    lines.append("The PitchTokenizer rebuckets `(plate_x, plate_z)` into a 5x5 grid (zones 0..24) plus zone 25 = missing.  Inside the rollout, the count-only fallback (`outcome_predictor=None`) treats the **inner-3x3 of the 5x5 grid** = zones `{6, 7, 8, 11, 12, 13, 16, 17, 18}` as in-zone (`_IN_ZONE_INDICES`).  We compute the in-zone rate under BOTH conventions; **the tokenizer convention is the apples-to-apples comparison** for the rollout.")
    lines.append("")
    lines.append("## Empirical 2025 first-pitch distribution\n")
    lines.append("Source: `pitches WHERE pitch_number=1 AND year=2025`, full season (no pitcher-disjoint filter -- it does not change first-pitch zone marginals materially).")
    lines.append("")
    lines.append("| Convention | in-zone | out-of-zone | missing | n |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| Statcast (`zone in {{1..9}}` vs `{{11..14}}`) | {e['statcast_zone']['p_in_zone']:.4f} | {e['statcast_zone']['p_out_zone']:.4f} | {e['statcast_zone']['p_missing']:.4f} | {e['statcast_zone']['n_total']} |")
    lines.append(f"| Tokenizer (5x5 inner-3x3 vs rest) | {e['tokenizer']['p_in_zone']:.4f} | {e['tokenizer']['p_out_zone']:.4f} | {e['tokenizer']['p_missing']:.4f} | {e['tokenizer']['n_pitches']} |")
    lines.append("")
    lines.append("## Sampled (T=1.0) H=0 distribution\n")
    lines.append(f"- n_pa = {s['n_pa']}, n_samples = {s['n_samples']}, n_total_pitches = {s['n_total_pitches']}")
    lines.append(f"- Token TVD vs empirical: **{s['tvd_token_vs_empirical']:.4f}**")
    lines.append(f"- Token KL(sampled || empirical): {s['kl_sampled_vs_empirical_token']:.4f}")
    lines.append(f"- Token KL(empirical || sampled): {s['kl_empirical_vs_sampled_token']:.4f}")
    lines.append(f"- Zone TVD vs empirical: **{s['tvd_zone_vs_empirical']:.4f}**")
    lines.append(f"- Zone KL(sampled || empirical): {s['kl_sampled_vs_empirical_zone']:.4f}")
    lines.append(f"- Pitch-type TVD vs empirical: {s['tvd_pitch_type_vs_empirical']:.4f}")
    lines.append(f"- Velo-bucket TVD vs empirical: {s['tvd_velo_bucket_vs_empirical']:.4f}")
    lines.append(f"- In-zone (tokenizer convention): **{s['p_in_zone_tokenizer']:.4f}** "
                 f"(empirical {e['tokenizer']['p_in_zone']:.4f}; "
                 f"delta {s['p_in_zone_minus_empirical']:+.4f}; "
                 f"vs uniform {s['p_in_zone_minus_uniform_grid']:+.4f})")
    lines.append(f"- Wall clock for MAIN run: {s['wall_clock_seconds']:.1f}s")
    lines.append("")
    lines.append("## Heuristic-implied K%/BB% (count-only None-predictor, H=6 i.i.d.)\n")
    lines.append("This is the closed-form K% / BB% under the rollout's None-predictor fallback (in-zone -> +1 strike, otherwise +1 ball) assuming each of 6 pitches is i.i.d. drawn at the H=0 marginal in-zone rate.  ISOLATES the pitch-token sampling effect from the outcome predictor.")
    lines.append("")
    lines.append("| In-zone source | p_in_zone | implied K% | implied BB% | implied truncated% |")
    lines.append("|---|---|---|---|---|")
    e_t = heur["empirical_tokenizer_in_zone_rate"]
    e_s = heur["empirical_statcast_in_zone_rate"]
    samp = heur["sampled_t10_in_zone_rate"]
    lines.append(f"| Empirical (tokenizer) | {e_t['p_in_zone']:.4f} | {e_t['implied_k_pct']:.4f} | {e_t['implied_bb_pct']:.4f} | {e_t['implied_truncated_pct']:.4f} |")
    lines.append(f"| Empirical (Statcast)  | {e_s['p_in_zone']:.4f} | {e_s['implied_k_pct']:.4f} | {e_s['implied_bb_pct']:.4f} | {e_s['implied_truncated_pct']:.4f} |")
    lines.append(f"| Sampled @ T=1.0       | {samp['p_in_zone']:.4f} | {samp['implied_k_pct']:.4f} | {samp['implied_bb_pct']:.4f} | {samp['implied_truncated_pct']:.4f} |")
    lines.append("")
    lines.append("Note: this i.i.d. heuristic does NOT mirror the rollout's outcome predictor.  It is a stress test that tells us: \"given JUST the pitch-token sampling, what K%/BB% would the count-only fallback emit?\"  The implied K% delta between the empirical-in-zone and the sampled-in-zone rate is the **maximum K% bias attributable to pitch-token sampling alone**.")
    lines.append("")
    lines.append("## Temperature sweep\n")
    if sweep["T"]:
        lines.append("| T | n_pa | wall (s) | TVD vs emp (token) | TVD vs emp (zone) | in-zone | delta vs emp |")
        lines.append("|---|---|---|---|---|---|---|")
        # T=1.0 row from main
        t10_in_zone_minus_emp = h['delta_in_zone_t10_tokenizer_pp'] / 100.0
        lines.append(f"| **1.0** (main) | {payload['args']['n_pa_main']} | {s['wall_clock_seconds']:.1f} | "
                     f"{s['tvd_token_vs_empirical']:.4f} | {s['tvd_zone_vs_empirical']:.4f} | "
                     f"{s['p_in_zone_tokenizer']:.4f} | {s['p_in_zone_minus_empirical']:+.4f} |")
        for t_str, r in sweep["T"].items():
            lines.append(f"| {t_str} | {r['n_pa']} | {r['wall_clock_seconds']:.1f} | "
                         f"{r['tvd_token_vs_empirical']:.4f} | {r['tvd_zone_vs_empirical']:.4f} | "
                         f"{r['p_in_zone_tokenizer']:.4f} | {r['delta_in_zone_vs_empirical']:+.4f} |")
        lines.append("")
        lines.append("Hypothesis check on temperature: if T=1.0 is the issue, T=1.2 (more entropy) should reduce the in-zone bias — i.e., its `delta vs emp` should be smaller (closer to 0) than T=1.0's.  T=0.8 (more concentrated) should increase the bias.")
    else:
        lines.append("Temperature-sweep skipped or budget-cut.")
    lines.append("")
    lines.append("## Decision matrix used to determine verdict\n")
    lines.append("- **CONFIRMED**: `delta_in_zone @T=1.0 > +0.02` AND heuristic-implied K% gap >= +0.005 (0.5pp).")
    lines.append("- **FALSIFIED**: `delta_in_zone @T=1.0 <= +0.005` (essentially no in-zone over-emission).")
    lines.append("- **INDETERMINATE**: in-zone bias non-trivial but K% impact below 0.5pp.")
    lines.append("")
    lines.append("## Key quantitative claim\n")
    lines.append(f"Estimate: **{h['attributable_k_bias_pp']:+.2f} pp** of the observed +{h['observed_k_bias_pp']:.2f}pp K% bias is attributable to pitch-token sampling alone (= {h['attribution_pct_of_observed_k_bias']:.1f}%).  The remainder (~{h['observed_k_bias_pp'] - h['attributable_k_bias_pp']:+.2f}pp) must come from elsewhere -- candidates: (a) outcome-predictor calibration off-distribution from PA terminal classes; (b) static-context (constant within PA) interacting with the predictor; (c) PA-termination logic.")
    lines.append("")
    if payload["scope_cuts"]:
        lines.append("## Scope cuts\n")
        for sc in payload["scope_cuts"]:
            lines.append(f"- {sc}")
        lines.append("")
    lines.append("## Cross-references\n")
    lines.append("- Source rollout-sanity: `scripts/pitchgpt_rollout_sanity_2025.py`")
    lines.append("- Source metrics: `results/pitchgpt/rollout_sanity_2025/metrics.json`")
    lines.append("- Tokenizer: `src/analytics/pitchgpt.py` :: `PitchTokenizer.encode/decode`")
    lines.append("- In-zone heuristic: `src/analytics/pitchgpt_sim.py` :: `_IN_ZONE_INDICES` / `_zone_is_in_strike_zone`")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
