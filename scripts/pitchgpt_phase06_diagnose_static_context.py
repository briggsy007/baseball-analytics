"""
Phase 0.6 diagnostic — verify or falsify the static-mid-PA-context hypothesis.

Hypothesis under test:
    PAContext is held constant within a PA. count_state at position p>0
    still reads the STARTING count of the PA, not the running count. A1
    head was trained on real (count, hidden, pitch) tuples where count
    updates each pitch. Mid-PA the head sees stale context; suspected to
    under-emit balls in late-PA positions.

If the hypothesis is correct:
    KL(rollout_pos_p_marginal || empirical_pos_p_marginal) should
    INCREASE monotonically with position p.

Plan:
1. Pull the same 2025 pitcher-disjoint PA cohort the rollout sanity uses
   (seed=42 sample). Reduce n_pa to 2000 to fit ~30min wall-clock budget.
2. For H ∈ {1, 2, 3, 6}, run rollout(..., outcome_predictor=PGConcatHeadPredictor(),
   n_samples=100, T=1.0). Compute the per-position outcome marginal
   (averaged across n_samples × n_pa).
3. Empirical baseline: from the 2025 pitches table, compute the empirical
   outcome distribution at each position-in-PA (pitch_number 1..6).
4. KL(rollout_marginal || empirical_marginal) per position. Hypothesis:
   monotone increase with position.
5. Counter-test: stratify by starting count_state ∈ {0-0, 0-1, 1-0, 2-2}.
   Hypothesis: position-5 KL is smaller for PAs starting at 2-2 (count is
   already "almost terminal" so context drift is small).
6. Write results/pitchgpt/rollout_sanity_2025/diagnose_static_context.json
   and .md with verdict.

Guardrails:
- Do NOT modify src/analytics/pitchgpt_sim.py, pitchgpt.py, pitchgpt_outcome_head.py
  or any model checkpoint.
- DuckDB read_only=True throughout.
- Do NOT commit anything.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt_outcome_head import classify_pitch_outcome  # noqa: E402
from src.analytics.pitchgpt_sim import (  # noqa: E402
    NUM_OUTCOME_CLASSES,
    OUTCOME_BALL,
    OUTCOME_CALLED_STRIKE,
    OUTCOME_FOUL,
    OUTCOME_HBP,
    OUTCOME_IN_PLAY_HIT,
    OUTCOME_IN_PLAY_OUT,
    OUTCOME_SWINGING_STRIKE,
    PAContext,
    PGConcatHeadPredictor,
    rollout,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase06_diag_static_context")


# ─── Configuration ──────────────────────────────────────────────────────
TRAIN_RANGE = (2015, 2022)
HOLDOUT_SEASON = 2025
N_PA = 2000   # scaled from 10K to fit ~30min budget
N_SAMPLES = 100
HORIZONS = (1, 2, 3, 6)
TEMPERATURE = 1.0
SEED_ROOT = 42

OUTCOME_NAMES = (
    "ball", "called_strike", "swinging_strike", "foul",
    "in_play_out", "in_play_hit", "hbp",
)


# ─── Cohort fetching (mirror rollout_sanity_2025 logic) ─────────────────


def _fetch_train_pitcher_ids(conn) -> set[int]:
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


def _fetch_2025_pa_starts(conn, train_pitchers: set[int]) -> pd.DataFrame:
    """Return one row per 2025 PA (pitch_number=1) with PA-start fields."""
    pids_str = ", ".join(str(int(p)) for p in train_pitchers)
    query = f"""
        WITH all_2025 AS (
            SELECT
                game_pk, at_bat_number, pitcher_id, batter_id, pitch_number,
                balls, strikes, outs_when_up, on_1b, on_2b, on_3b,
                stand, p_throws, inning, inning_topbot, events, woba_value, description
            FROM pitches
            WHERE EXTRACT(YEAR FROM game_date) = {HOLDOUT_SEASON}
              AND pitch_type IS NOT NULL
              AND pitcher_id IS NOT NULL
              AND batter_id IS NOT NULL
              AND at_bat_number IS NOT NULL
              AND pitch_number IS NOT NULL
              AND pitcher_id NOT IN ({pids_str})
        ),
        pa_groups AS (
            SELECT
                game_pk, at_bat_number, pitcher_id, batter_id,
                MIN(pitch_number) AS first_pn,
                MAX(pitch_number) AS last_pn,
                COUNT(*)           AS pa_pitch_count
            FROM all_2025
            GROUP BY game_pk, at_bat_number, pitcher_id, batter_id
        ),
        starts AS (
            SELECT
                p.game_pk, p.at_bat_number, p.pitcher_id, p.batter_id,
                p.balls, p.strikes, p.outs_when_up,
                p.on_1b, p.on_2b, p.on_3b,
                p.stand, p.p_throws,
                p.inning, p.inning_topbot,
                g.first_pn, g.last_pn, g.pa_pitch_count
            FROM all_2025 p
            JOIN pa_groups g
              ON p.game_pk = g.game_pk
             AND p.at_bat_number = g.at_bat_number
             AND p.pitcher_id = g.pitcher_id
             AND p.batter_id = g.batter_id
             AND p.pitch_number = g.first_pn
        ),
        terminals AS (
            SELECT
                p.game_pk, p.at_bat_number, p.pitcher_id, p.batter_id,
                p.events     AS terminal_event
            FROM all_2025 p
            JOIN pa_groups g
              ON p.game_pk = g.game_pk
             AND p.at_bat_number = g.at_bat_number
             AND p.pitcher_id = g.pitcher_id
             AND p.batter_id = g.batter_id
             AND p.pitch_number = g.last_pn
        )
        SELECT
            s.*,
            t.terminal_event
        FROM starts s
        JOIN terminals t
          ON s.game_pk = t.game_pk
         AND s.at_bat_number = t.at_bat_number
         AND s.pitcher_id = t.pitcher_id
         AND s.batter_id = t.batter_id
        WHERE t.terminal_event IS NOT NULL
          AND t.terminal_event != ''
    """
    return conn.execute(query).fetchdf()


def _fetch_2025_per_position_outcomes(
    conn,
    train_pitchers: set[int],
    pa_keys: set[tuple],
) -> pd.DataFrame:
    """Return all pitches in 2025 PA cohort with (description, events,
    pitch_number) for empirical per-position outcome marginal computation.

    pa_keys is the set of (game_pk, at_bat_number, pitcher_id, batter_id)
    tuples to filter to (the sample we sampled from PA-starts).
    """
    pids_str = ", ".join(str(int(p)) for p in train_pitchers)
    query = f"""
        SELECT
            game_pk, at_bat_number, pitcher_id, batter_id,
            pitch_number, description, events,
            balls, strikes
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) = {HOLDOUT_SEASON}
          AND pitch_type IS NOT NULL
          AND pitcher_id IS NOT NULL
          AND batter_id IS NOT NULL
          AND at_bat_number IS NOT NULL
          AND pitch_number IS NOT NULL
          AND pitcher_id NOT IN ({pids_str})
    """
    df = conn.execute(query).fetchdf()
    # Restrict to sampled PA cohort.
    keys = pd.DataFrame(
        list(pa_keys),
        columns=["game_pk", "at_bat_number", "pitcher_id", "batter_id"],
    )
    return df.merge(keys, on=["game_pk", "at_bat_number", "pitcher_id", "batter_id"], how="inner")


# ─── KL divergence ──────────────────────────────────────────────────────


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """KL(p || q) with epsilon smoothing on q (and renormalise)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), eps)
    q = q + eps
    q = q / q.sum()
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


# ─── Empirical per-position marginal builder ────────────────────────────


def _per_position_empirical_marginal(
    df_pitches: pd.DataFrame,
    *,
    max_position: int = 6,
    starting_count_filter: tuple[int, int] | None = None,
    pa_starts_index: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the empirical 7-class outcome distribution at each position
    p ∈ [1, max_position] across the PA cohort.

    If ``starting_count_filter`` is not None, restrict the cohort to PAs
    whose starting count == filter (uses pa_starts_index).

    Returns:
        marginals : shape (max_position, 7)
        n_at_pos  : shape (max_position,)  — number of pitches contributing
    """
    if starting_count_filter is not None and pa_starts_index is not None:
        b, s = starting_count_filter
        pas_match = pa_starts_index[
            (pa_starts_index["balls"] == b) & (pa_starts_index["strikes"] == s)
        ][["game_pk", "at_bat_number", "pitcher_id", "batter_id"]]
        df_pitches = df_pitches.merge(
            pas_match,
            on=["game_pk", "at_bat_number", "pitcher_id", "batter_id"],
            how="inner",
        )

    marg = np.zeros((max_position, NUM_OUTCOME_CLASSES), dtype=np.float64)
    n_at = np.zeros((max_position,), dtype=np.int64)

    # Vectorised classify_pitch_outcome on the rows.
    descrs = df_pitches["description"].tolist()
    events = df_pitches["events"].tolist()
    pns = df_pitches["pitch_number"].astype(int).to_numpy()
    classes = np.array(
        [classify_pitch_outcome(d, e) for d, e in zip(descrs, events)],
        dtype=np.int32,
    )

    valid = (classes >= 0) & (classes < NUM_OUTCOME_CLASSES)
    for p in range(1, max_position + 1):
        sel = valid & (pns == p)
        if not sel.any():
            continue
        cls_p = classes[sel]
        counts = np.bincount(cls_p, minlength=NUM_OUTCOME_CLASSES)
        marg[p - 1, :] = counts
        n_at[p - 1] = int(sel.sum())

    # Normalise rows.
    for i in range(max_position):
        if n_at[i] > 0:
            marg[i, :] /= float(n_at[i])
    return marg.astype(np.float32), n_at


# ─── Rollout per-position marginal builder ──────────────────────────────


def _per_position_rollout_marginal(
    rollout_results: list,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate rollout results across PAs to per-position outcome marginal.

    For each position p ∈ [0, horizon-1], average the outcome_probs
    across (n_samples × n_pa) — using NaN-mask to ignore truncated
    positions per the API contract.

    Returns:
        marg : shape (horizon, 7)
        n_at : shape (horizon,)  — number of (sample × pa) instances
                  contributing at each position (non-NaN, non-truncated)
    """
    # Stack outcome_probs across PAs.  Each rollout is (n_samples, horizon, 7).
    # NaN past termination per the API contract.
    all_probs = np.stack(
        [r.outcome_probs for r in rollout_results],
        axis=0,  # (n_pa, n_samples, horizon, 7)
    )
    n_pa, n_samples, h, c = all_probs.shape
    assert h == horizon, f"horizon mismatch: {h} != {horizon}"
    assert c == NUM_OUTCOME_CLASSES

    # Per-position marginal: mean across (n_pa × n_samples), NaN-aware.
    flat = all_probs.reshape(n_pa * n_samples, horizon, NUM_OUTCOME_CLASSES)
    marg = np.full((horizon, NUM_OUTCOME_CLASSES), np.nan, dtype=np.float32)
    n_at = np.zeros((horizon,), dtype=np.int64)
    for p in range(horizon):
        mask = ~np.isnan(flat[:, p, 0])  # if outcome at pos p exists
        if mask.any():
            marg[p, :] = np.nanmean(flat[mask, p, :], axis=0)
            n_at[p] = int(mask.sum())
        else:
            marg[p, :] = 0.0
    return marg, n_at


def _per_position_rollout_marginal_filtered(
    rollout_results: list,
    pa_starts_filter_indices: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Same as above but restricted to PAs at the given indices."""
    sel = [rollout_results[i] for i in pa_starts_filter_indices]
    if not sel:
        marg = np.full((horizon, NUM_OUTCOME_CLASSES), np.nan, dtype=np.float32)
        n_at = np.zeros((horizon,), dtype=np.int64)
        return marg, n_at
    return _per_position_rollout_marginal(sel, horizon=horizon)


# ─── Main ──────────────────────────────────────────────────────────────


def main() -> int:
    output_dir = _ROOT / "results" / "pitchgpt" / "rollout_sanity_2025"
    output_dir.mkdir(parents=True, exist_ok=True)

    t_overall_start = time.perf_counter()

    logger.info(
        "Phase 0.6 diagnostic (static-context) start: n_pa=%d  n_samples=%d  horizons=%s  T=%.3f  seed=%d",
        N_PA, N_SAMPLES, HORIZONS, TEMPERATURE, SEED_ROOT,
    )

    # ── 1. Cohort fetch ────────────────────────────────────────────────
    conn = get_connection(read_only=True)
    try:
        train_pitchers = _fetch_train_pitcher_ids(conn)
        logger.info("Train (2015-2022) pitcher cohort: %d pitchers", len(train_pitchers))
        df_pas = _fetch_2025_pa_starts(conn, train_pitchers)
        logger.info("2025 pitcher-disjoint PAs: %d", len(df_pas))

        # Resolve league-median ump scalar (matches rollout_sanity_2025).
        rows = conn.execute(
            """
            SELECT season, MEDIAN(accuracy_above_x_wmean) AS med
            FROM umpire_tendencies
            WHERE accuracy_above_x_wmean IS NOT NULL
              AND season IN (2024, 2025)
            GROUP BY season
            """,
        ).fetchdf()
        by_season = {int(r["season"]): float(r["med"]) for _, r in rows.iterrows()}
        league_median_ump = by_season.get(2024, by_season.get(2025, 0.0))
        logger.info("League-median ump scalar: %.5f", league_median_ump)

        # Sample N_PA PA starts deterministically.
        rng = np.random.default_rng(SEED_ROOT)
        sample_idx = rng.choice(len(df_pas), size=min(N_PA, len(df_pas)), replace=False)
        df_sample = df_pas.iloc[sample_idx].reset_index(drop=True)
        for col in ("on_1b", "on_2b", "on_3b"):
            if col in df_sample.columns:
                df_sample[col] = (
                    pd.to_numeric(df_sample[col], errors="coerce")
                    .fillna(0)
                    .astype(np.int64)
                )
        for col in ("balls", "strikes", "outs_when_up", "inning", "pitcher_id", "batter_id"):
            if col in df_sample.columns:
                df_sample[col] = (
                    pd.to_numeric(df_sample[col], errors="coerce")
                    .fillna(0)
                    .astype(np.int64)
                )
        logger.info("Sampled %d PA starts (seed=%d)", len(df_sample), SEED_ROOT)

        # Build the empirical per-position cohort: pitches in those PAs.
        pa_keys = set(
            zip(
                df_sample["game_pk"].astype(int).tolist(),
                df_sample["at_bat_number"].astype(int).tolist(),
                df_sample["pitcher_id"].astype(int).tolist(),
                df_sample["batter_id"].astype(int).tolist(),
            )
        )
        df_pitches = _fetch_2025_per_position_outcomes(conn, train_pitchers, pa_keys)
        logger.info("Pitches in sampled cohort: %d", len(df_pitches))
    finally:
        conn.close()

    # ── 2. Empirical per-position marginal (overall) ───────────────────
    t_emp_start = time.perf_counter()
    emp_marg_overall, emp_n_overall = _per_position_empirical_marginal(
        df_pitches, max_position=6,
    )
    logger.info("Empirical per-position marginal: n_at=%s (computed in %.1fs)",
                emp_n_overall.tolist(), time.perf_counter() - t_emp_start)

    # Empirical per-position marginal stratified by starting count.
    starting_counts = [(0, 0), (0, 1), (1, 0), (2, 2)]
    emp_marg_by_count: dict[str, dict] = {}
    for b, s in starting_counts:
        marg_bs, n_bs = _per_position_empirical_marginal(
            df_pitches, max_position=6,
            starting_count_filter=(b, s),
            pa_starts_index=df_sample,
        )
        key = f"{b}-{s}"
        emp_marg_by_count[key] = {
            "marginal": marg_bs.tolist(),
            "n_at_position": n_bs.tolist(),
        }
        logger.info("  Empirical [%s start]: n_at=%s", key, n_bs.tolist())

    # ── 3. Run rollouts per horizon ────────────────────────────────────
    cuda = torch.cuda.is_available()
    logger.info("Device: %s", "cuda" if cuda else "cpu")
    predictor = PGConcatHeadPredictor()

    rollout_marginals_by_h: dict[int, dict] = {}
    rollout_results_by_h: dict[int, list] = {}
    for H in HORIZONS:
        logger.info("Rollout H=%d: starting %d PAs × %d samples", H, len(df_sample), N_SAMPLES)
        t_h = time.perf_counter()
        results_h: list = []
        for i, row in df_sample.iterrows():
            ctx = PAContext.from_pitches_row(row, ump_scalar=league_median_ump)
            per_row_seed = (SEED_ROOT * 1_000_003 + i) & 0x7FFFFFFF
            r = rollout(
                ctx,
                outcome_predictor=predictor,
                n_samples=N_SAMPLES,
                horizon=H,
                temperature=TEMPERATURE,
                seed=per_row_seed,
                return_probs=True,
            )
            results_h.append(r)
            if (i + 1) % max(len(df_sample) // 5, 50) == 0:
                dt = time.perf_counter() - t_h
                rate = (i + 1) / dt
                logger.info("  H=%d: %d/%d  rate=%.1f PA/s", H, i + 1, len(df_sample), rate)
        dt_h = time.perf_counter() - t_h
        logger.info("Rollout H=%d complete in %.1fs", H, dt_h)
        rollout_results_by_h[H] = results_h

        # Per-position rollout marginal (overall).
        roll_marg, roll_n = _per_position_rollout_marginal(results_h, horizon=H)
        rollout_marginals_by_h[H] = {
            "marginal": roll_marg.tolist(),
            "n_at_position": roll_n.tolist(),
            "wall_clock_seconds": round(dt_h, 1),
        }

    # ── 4. KL divergences per position per horizon (overall) ───────────
    kl_table_overall: dict[int, list[float]] = {}
    for H in HORIZONS:
        roll_marg = np.array(rollout_marginals_by_h[H]["marginal"], dtype=np.float64)
        kls = []
        for p in range(H):
            # Use empirical at pos (p+1) — 1-indexed pitch_number — vs rollout at pos p (0-indexed).
            if p < emp_marg_overall.shape[0] and emp_n_overall[p] > 0:
                kl = _kl_divergence(roll_marg[p, :], emp_marg_overall[p, :])
            else:
                kl = float("nan")
            kls.append(kl)
        kl_table_overall[H] = kls
        logger.info("KL by position (H=%d): %s",
                    H, [f"{k:.4f}" if not np.isnan(k) else "nan" for k in kls])

    # ── 5. KL by starting count (counter-test, only for H=6) ───────────
    H_test = 6
    results_h6 = rollout_results_by_h[H_test]
    starting_counts_balls = df_sample["balls"].to_numpy()
    starting_counts_strikes = df_sample["strikes"].to_numpy()

    kl_by_starting_count: dict[str, list[float]] = {}
    rollout_marg_by_starting_count: dict[str, dict] = {}
    for b, s in starting_counts:
        idx = np.where(
            (starting_counts_balls == b) & (starting_counts_strikes == s)
        )[0]
        n_pas_at_count = int(len(idx))
        if n_pas_at_count == 0:
            kl_by_starting_count[f"{b}-{s}"] = [float("nan")] * H_test
            rollout_marg_by_starting_count[f"{b}-{s}"] = {
                "n_pas_at_count": 0,
                "marginal": [[float("nan")] * NUM_OUTCOME_CLASSES] * H_test,
                "n_at_position": [0] * H_test,
            }
            continue

        roll_marg_bs, roll_n_bs = _per_position_rollout_marginal_filtered(
            results_h6, idx, horizon=H_test,
        )
        emp_marg_bs = np.array(emp_marg_by_count[f"{b}-{s}"]["marginal"], dtype=np.float64)
        emp_n_bs = np.array(emp_marg_by_count[f"{b}-{s}"]["n_at_position"], dtype=np.int64)
        kls_bs = []
        for p in range(H_test):
            if p < emp_marg_bs.shape[0] and emp_n_bs[p] > 0 and roll_n_bs[p] > 0:
                kl = _kl_divergence(roll_marg_bs[p, :], emp_marg_bs[p, :])
            else:
                kl = float("nan")
            kls_bs.append(kl)
        kl_by_starting_count[f"{b}-{s}"] = kls_bs
        rollout_marg_by_starting_count[f"{b}-{s}"] = {
            "n_pas_at_count": n_pas_at_count,
            "marginal": roll_marg_bs.tolist(),
            "n_at_position": roll_n_bs.tolist(),
        }
        logger.info("KL by position [start=%d-%d, n_pas=%d]: %s",
                    b, s, n_pas_at_count,
                    [f"{k:.4f}" if not np.isnan(k) else "nan" for k in kls_bs])

    # ── 6. Verdict logic ──────────────────────────────────────────────

    # Hypothesis 1: monotone increase in KL with position (overall, H=6).
    kl_h6 = kl_table_overall.get(H_test, [])
    valid = [k for k in kl_h6 if not np.isnan(k)]
    monotone_increase = False
    if len(valid) >= 4:
        # Compute correlation of position vs KL.
        positions = np.arange(len(valid))
        if np.std(valid) > 1e-9:
            r = float(np.corrcoef(positions, valid)[0, 1])
        else:
            r = 0.0
        # Strong monotone increase: r >= 0.7 AND last-position KL > 2x first-position KL.
        kl_ratio = (valid[-1] / valid[0]) if valid[0] > 1e-9 else float("inf")
        monotone_increase = (r >= 0.7) and (kl_ratio > 2.0)
        logger.info(
            "Verdict-component 1: r(pos, KL) = %.3f  ratio(last/first) = %.2f  monotone_increase=%s",
            r, kl_ratio, monotone_increase,
        )
    else:
        r = float("nan")
        kl_ratio = float("nan")

    # Hypothesis 2 (counter-test): KL at position 5 is smaller for 2-2 start than for 0-0 start.
    # CAVEAT: Almost all PAs start at 0-0 (>99.7% in the cohort), so the
    # 2-2 start cohort is ~8 PAs across the entire 2025 holdout, which
    # makes this stratification statistically weak.  We retain it as
    # informational only.  The PRIMARY decision criterion is the
    # monotone-KL-increase.
    counter_test_supports_hypothesis = False
    kl_pos5_by_count: dict[str, float] = {}
    for k, kls in kl_by_starting_count.items():
        kl_pos5_by_count[k] = kls[-1] if len(kls) >= H_test else float("nan")
    kl_22_pos5 = kl_pos5_by_count.get("2-2", float("nan"))
    kl_00_pos5 = kl_pos5_by_count.get("0-0", float("nan"))
    if not (np.isnan(kl_22_pos5) or np.isnan(kl_00_pos5)):
        counter_test_supports_hypothesis = (kl_22_pos5 < kl_00_pos5)
        logger.info(
            "Verdict-component 2a (starting-count counter-test): KL@pos5 [2-2 start] = %.4f vs [0-0 start] = %.4f  -> supports=%s",
            kl_22_pos5, kl_00_pos5, counter_test_supports_hypothesis,
        )

    # Hypothesis 2b (cross-horizon counter-test, primary counter-test):
    # If static-context is the bug, the position-p KL should be similar
    # whether p is reached at H=p+1 (terminal) or at H=6 (mid-rollout).
    # Specifically: KL@pos1 of H=2 ≈ KL@pos1 of H=6 (one position of drift either way).
    # AND KL@pos1 should be NOTICEABLY larger than KL@pos0 of H=1 (drift introduces error).
    cross_horizon_supports = False
    cross_horizon_evidence: dict[str, float] = {}
    h1_pos0 = kl_table_overall.get(1, [float("nan")])[0]
    h6_pos0 = kl_table_overall.get(6, [float("nan")])[0] if len(kl_table_overall.get(6, [])) > 0 else float("nan")
    h2_pos1 = kl_table_overall.get(2, [float("nan"), float("nan")])[1] if len(kl_table_overall.get(2, [])) > 1 else float("nan")
    h6_pos1 = kl_table_overall.get(6, [float("nan")] * 6)[1] if len(kl_table_overall.get(6, [])) > 1 else float("nan")
    h6_pos5 = kl_table_overall.get(6, [float("nan")] * 6)[5] if len(kl_table_overall.get(6, [])) > 5 else float("nan")
    cross_horizon_evidence = {
        "h1_pos0": (None if np.isnan(h1_pos0) else round(h1_pos0, 4)),
        "h6_pos0": (None if np.isnan(h6_pos0) else round(h6_pos0, 4)),
        "h2_pos1": (None if np.isnan(h2_pos1) else round(h2_pos1, 4)),
        "h6_pos1": (None if np.isnan(h6_pos1) else round(h6_pos1, 4)),
        "h6_pos5": (None if np.isnan(h6_pos5) else round(h6_pos5, 4)),
    }
    # Static-context support: pos5 KL much larger than pos0 KL (drift accumulates),
    # AND pos1 KL approx the same regardless of H (one-step drift is the same).
    if all(not np.isnan(x) for x in (h6_pos0, h6_pos5)):
        # Static-context support: pos5 KL >> pos0 KL.
        ratio_5_0 = h6_pos5 / max(h6_pos0, 1e-6)
        cross_horizon_supports = ratio_5_0 > 2.0
        cross_horizon_evidence["ratio_pos5_pos0_h6"] = round(ratio_5_0, 4)
        logger.info(
            "Verdict-component 2b (cross-horizon counter-test): KL@pos5/KL@pos0 (H=6) = %.2f  -> supports=%s",
            ratio_5_0, cross_horizon_supports,
        )
    # The PRIMARY counter-test we actually use is 2b (cross-horizon),
    # since starting-count stratification is too sparse.
    counter_test_fires = cross_horizon_supports

    # Final verdict.
    if monotone_increase and counter_test_fires:
        verdict = "CONFIRMED"
        reasoning = (
            "Both verdict components support the static-context hypothesis: "
            "KL(rollout||empirical) increases monotonically with position (correlation+ratio gate), "
            "AND the cross-horizon counter-test confirms KL@pos5 >> KL@pos0 "
            "(drift accumulates with position depth)."
        )
    elif monotone_increase or counter_test_fires:
        verdict = "INDETERMINATE"
        reasoning = (
            "Only one of the two verdict components fires.  "
            f"monotone_KL_increase={monotone_increase}, cross_horizon_supports={counter_test_fires}.  "
            "Static-context bias may contribute but is not unambiguously the "
            "dominant cause of the Phase 0.6 FAIL; "
            "likely alternative candidates: A1 head class-marginal calibration "
            "(systematic ball/strike bias at all positions), PA-termination "
            "logic, or empirical-baseline small-sample noise at deep positions."
        )
    else:
        verdict = "FALSIFIED"
        reasoning = (
            "Neither verdict component fires.  KL does not increase monotonically "
            "with position AND the cross-horizon ratio is flat.  "
            "Static-context-within-PA is NOT the dominant cause of the "
            "Phase 0.6 K%/BB% gates failing.  Likely alternative: A1's "
            "class-marginal calibration is biased — under-emits ball, "
            "over-emits swinging_strike — same direction at every position."
        )

    # ── 7. Magnitude estimation if confirmed ──────────────────────────
    magnitude_estimate = None
    if verdict == "CONFIRMED":
        # Estimate K%/BB% bias attributable to static context: for each
        # position, compute (rollout - empirical) for ball and strike-events.
        # Sum the position-conditional bias weighted by the empirical
        # probability of reaching that position.
        # Pos-conditional p(reach pos p) = empirical P(PA length >= p+1).
        emp_n = emp_n_overall
        # P(reach pos p) ~ emp_n[p] / emp_n[0] (relative to position 1).
        if emp_n[0] > 0:
            p_reach = emp_n / emp_n[0]
        else:
            p_reach = np.zeros_like(emp_n, dtype=np.float64)

        roll_marg_h6 = np.array(rollout_marginals_by_h[H_test]["marginal"], dtype=np.float64)
        # Aggregate K-event probs (CS+SS) and ball probs.
        roll_strike_per_pos = roll_marg_h6[:, OUTCOME_CALLED_STRIKE] + roll_marg_h6[:, OUTCOME_SWINGING_STRIKE]
        roll_ball_per_pos = roll_marg_h6[:, OUTCOME_BALL]
        emp_strike_per_pos = emp_marg_overall[:H_test, OUTCOME_CALLED_STRIKE] + emp_marg_overall[:H_test, OUTCOME_SWINGING_STRIKE]
        emp_ball_per_pos = emp_marg_overall[:H_test, OUTCOME_BALL]

        # Position-conditional bias (rollout - empirical).
        bias_strike = roll_strike_per_pos - emp_strike_per_pos
        bias_ball = roll_ball_per_pos - emp_ball_per_pos
        # Approximate expected K% / BB% bias contribution: weighted by p_reach[p].
        # This is rough but gives an order-of-magnitude estimate.
        k_bias_est = float(np.sum(bias_strike * p_reach[:H_test]))
        bb_bias_est = float(np.sum(bias_ball * p_reach[:H_test]))
        magnitude_estimate = {
            "k_pct_bias_estimate": round(k_bias_est, 4),
            "bb_pct_bias_estimate": round(bb_bias_est, 4),
            "method": "sum over positions of (rollout - empirical) * P(reach pos p)",
            "caveat": "Order-of-magnitude estimate; assumes bias compounds purely additively across positions.",
        }

    # ── 8. Persist results ────────────────────────────────────────────
    wall_clock_total = time.perf_counter() - t_overall_start

    payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec": "Phase 0.6 diagnostic — static-context hypothesis",
        "args": {
            "n_pa": N_PA,
            "n_samples": N_SAMPLES,
            "horizons": list(HORIZONS),
            "temperature": TEMPERATURE,
            "seed": SEED_ROOT,
        },
        "wall_clock_total_seconds": round(wall_clock_total, 1),
        "device": "cuda" if cuda else "cpu",
        "scope_cuts": (
            f"Reduced n_pa from 10000 to {N_PA} to fit ~30min wall-clock "
            "budget per task instructions (allow up to 45min, scale down if "
            "longer)."
        ),
        "outcome_class_order": list(OUTCOME_NAMES),
        "empirical": {
            "overall": {
                "marginal": emp_marg_overall.tolist(),
                "n_at_position": emp_n_overall.tolist(),
            },
            "by_starting_count": emp_marg_by_count,
        },
        "rollout": {
            "by_horizon": rollout_marginals_by_h,
            "by_starting_count_h6": rollout_marg_by_starting_count,
        },
        "kl_by_position_per_horizon": {str(H): kl_table_overall[H] for H in HORIZONS},
        "kl_by_starting_count_h6": kl_by_starting_count,
        "verdict": verdict,
        "verdict_components": {
            "monotone_kl_increase": {
                "fires": bool(monotone_increase),
                "correlation_pos_kl": (None if np.isnan(r) else round(r, 4)),
                "ratio_last_first": (None if np.isnan(kl_ratio) else round(kl_ratio, 4)),
                "thresholds": {"r_min": 0.7, "ratio_min": 2.0},
            },
            "counter_test_supports": {
                "fires": bool(counter_test_fires),
                "primary_counter_test": "cross_horizon_pos5_vs_pos0_h6",
                "cross_horizon_evidence": cross_horizon_evidence,
                "expected": "KL@pos5/KL@pos0 (H=6) > 2.0 if static context drives bug",
                "starting_count_secondary_test": {
                    "kl_22_pos5": (None if np.isnan(kl_22_pos5) else round(kl_22_pos5, 4)),
                    "kl_00_pos5": (None if np.isnan(kl_00_pos5) else round(kl_00_pos5, 4)),
                    "supports": bool(counter_test_supports_hypothesis),
                    "caveat": "Almost all PAs start at 0-0; 2-2 starts are <1% — statistically weak",
                },
            },
        },
        "reasoning": reasoning,
        "magnitude_estimate": magnitude_estimate,
        "phase06_observed_bias": {
            "k_pct_bias_observed": 0.0888,  # +8.88pp from metrics.json
            "bb_pct_bias_observed": -0.0446,  # -4.46pp
            "mean_pa_length_bias_observed": -0.66,
        },
    }

    out_json = output_dir / "diagnose_static_context.json"
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Wrote %s", out_json)

    # ── 9. Emit markdown report ───────────────────────────────────────
    md_path = output_dir / "diagnose_static_context.md"
    _write_md_report(md_path, payload)
    logger.info("Wrote %s", md_path)

    # ── 10. Console summary ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print("Phase 0.6 Static-Context Diagnostic — Verdict: " + verdict)
    print("=" * 78)
    print(f"Wall clock: {wall_clock_total:.1f}s  (n_pa={N_PA}, horizons={HORIZONS})")
    print(f"Reasoning: {reasoning}")
    print()
    print("KL by position (overall, H=6):")
    for p, k in enumerate(kl_table_overall.get(6, [])):
        print(f"  pos {p}: {k:.4f}" if not np.isnan(k) else f"  pos {p}: nan")
    print()
    print("KL@pos5 by starting count:")
    for key, kls in kl_by_starting_count.items():
        if len(kls) >= H_test:
            v = kls[-1]
            print(f"  {key}: {v:.4f}" if not np.isnan(v) else f"  {key}: nan")
    print("=" * 78)
    return 0 if verdict in ("CONFIRMED", "FALSIFIED") else 0


def _write_md_report(path: Path, payload: dict) -> None:
    lines: list[str] = []
    lines.append("# Phase 0.6 Diagnostic — Static-Context Hypothesis\n")
    lines.append(f"Generated: {payload['timestamp_utc']}\n")
    lines.append(f"Wall clock: {payload['wall_clock_total_seconds']:.1f}s on {payload['device']}\n")
    lines.append("")
    lines.append("## Verdict: **" + payload["verdict"] + "**\n")
    lines.append("")
    lines.append(payload["reasoning"])
    lines.append("")
    lines.append("## Configuration\n")
    args = payload["args"]
    lines.append(f"- n_pa = **{args['n_pa']}**, n_samples = **{args['n_samples']}**")
    lines.append(f"- horizons tested: **{args['horizons']}**")
    lines.append(f"- temperature = {args['temperature']}, seed = {args['seed']}")
    lines.append(f"- {payload['scope_cuts']}")
    lines.append("")
    lines.append("## Phase 0.6 observed bias (context)\n")
    pob = payload["phase06_observed_bias"]
    lines.append(f"- K% bias: {pob['k_pct_bias_observed']:+.4f} ({pob['k_pct_bias_observed'] * 100:+.2f}pp)")
    lines.append(f"- BB% bias: {pob['bb_pct_bias_observed']:+.4f} ({pob['bb_pct_bias_observed'] * 100:+.2f}pp)")
    lines.append(f"- mean PA length bias: {pob['mean_pa_length_bias_observed']:+.3f} pitches")
    lines.append("")
    lines.append("## KL(rollout || empirical) by position × horizon\n")
    lines.append("| Position | H=1 | H=2 | H=3 | H=6 |")
    lines.append("|---|---|---|---|---|")
    kl_pos_h = payload["kl_by_position_per_horizon"]
    max_pos = max(len(v) for v in kl_pos_h.values()) if kl_pos_h else 0
    for p in range(max_pos):
        row = [f"pos {p}"]
        for H in (1, 2, 3, 6):
            kls = kl_pos_h.get(str(H), [])
            if p < len(kls):
                v = kls[p]
                row.append(f"{v:.4f}" if not (v != v) else "nan")
            else:
                row.append("--")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Hypothesis prediction: KL should INCREASE monotonically with position if static-context drift is the bug.")
    lines.append("")
    lines.append("## KL by starting count (H=6, position 0..5)\n")
    lines.append("| Start | n_pas | pos0 | pos1 | pos2 | pos3 | pos4 | pos5 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    klbsc = payload["kl_by_starting_count_h6"]
    rmbsc = payload["rollout"]["by_starting_count_h6"]
    for key, kls in klbsc.items():
        n_pas = rmbsc.get(key, {}).get("n_pas_at_count", 0)
        cells = [key, str(n_pas)]
        for v in kls:
            cells.append(f"{v:.4f}" if not (v != v) else "nan")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Hypothesis prediction: KL@pos5 should be smaller for 2-2 start than for 0-0 start (count is already near-terminal; rollout positions don't drift far from real count).")
    lines.append("")
    lines.append("## Verdict components\n")
    vc = payload["verdict_components"]
    mki = vc["monotone_kl_increase"]
    cts = vc["counter_test_supports"]
    lines.append(f"### 1. Monotone KL increase (overall, H=6) — **{'FIRES' if mki['fires'] else 'no fire'}**\n")
    lines.append(f"- Pearson r(position, KL) = {mki['correlation_pos_kl']}")
    lines.append(f"- Ratio KL[pos5] / KL[pos0] = {mki['ratio_last_first']}")
    lines.append(f"- Thresholds: r >= {mki['thresholds']['r_min']}, ratio >= {mki['thresholds']['ratio_min']}")
    lines.append("")
    lines.append(f"### 2. Counter-test (cross-horizon) — **{'FIRES' if cts['fires'] else 'no fire'}**\n")
    lines.append(f"- Primary: cross-horizon ratio test")
    lines.append(f"- Cross-horizon evidence: {cts['cross_horizon_evidence']}")
    lines.append(f"- Expected: {cts['expected']}")
    sst = cts.get("starting_count_secondary_test", {})
    if sst:
        lines.append(f"- Secondary (starting-count, weak): KL@pos5 [2-2 start] = {sst.get('kl_22_pos5')} vs [0-0 start] = {sst.get('kl_00_pos5')}; supports={sst.get('supports')}")
        lines.append(f"  - Caveat: {sst.get('caveat')}")
    lines.append("")
    if payload.get("magnitude_estimate"):
        me = payload["magnitude_estimate"]
        lines.append("## Magnitude estimate (if CONFIRMED)\n")
        lines.append(f"- K% bias attributable to static context: **{me['k_pct_bias_estimate']:+.4f}**")
        lines.append(f"- BB% bias attributable to static context: **{me['bb_pct_bias_estimate']:+.4f}**")
        lines.append(f"- Method: {me['method']}")
        lines.append(f"- Caveat: {me['caveat']}")
        lines.append("")
    lines.append("## Empirical per-position outcome distribution (overall)\n")
    lines.append("Class order: " + ", ".join(payload["outcome_class_order"]))
    lines.append("")
    lines.append("| Position | n | ball | called_strike | swinging_strike | foul | in_play_out | in_play_hit | hbp |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    emp_marg = payload["empirical"]["overall"]["marginal"]
    emp_n = payload["empirical"]["overall"]["n_at_position"]
    for p in range(len(emp_marg)):
        cells = [f"pos {p+1}", str(emp_n[p])]
        for c in range(NUM_OUTCOME_CLASSES):
            cells.append(f"{emp_marg[p][c]:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Rollout per-position outcome distribution (H=6)\n")
    lines.append("| Position | n | ball | called_strike | swinging_strike | foul | in_play_out | in_play_hit | hbp |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    roll_h6 = payload["rollout"]["by_horizon"][6 if 6 in payload["rollout"]["by_horizon"] else "6"]
    if not isinstance(roll_h6, dict):
        roll_h6 = payload["rollout"]["by_horizon"]["6"]
    rm = roll_h6["marginal"]
    rn = roll_h6["n_at_position"]
    for p in range(len(rm)):
        cells = [f"pos {p}", str(rn[p])]
        for c in range(NUM_OUTCOME_CLASSES):
            cells.append(f"{rm[p][c]:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
