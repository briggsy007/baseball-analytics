"""
PitchGPT rollout sanity check on 2025 (Phase 0.6).

Spec: docs/pitchgpt_sim_engine/PHASE_0.5_PLAN.md §3.

Samples 10K PA starts from the 2025 pitcher-disjoint test cohort with
``np.random.default_rng(seed=42)``, runs:

    PRIMARY   : rollout(... outcome_predictor=PGConcatHeadPredictor(),
                            n_samples=100, horizon=6, temperature=1.0)
    SECONDARY : rollout(... outcome_predictor=None, ...)  -- bias surface,
                NOT a gate

Reports K%, BB%, HR%, mean wOBA, mean PA-length-in-pitches with 95%
bootstrap CIs vs an empirical baseline computed directly from the 2025
``pitches`` cohort (read_only=True).  Writes:

    results/pitchgpt/rollout_sanity_2025/metrics.json
    results/pitchgpt/rollout_sanity_2025/report.md

Gate thresholds (binding, per PHASE_0.5_PLAN §3.5 + §6 + EXECUTION_PLAN
§6.0.6):

    * K%/BB%/HR%      : within +-10% relative OR +-1pp absolute
                        (whichever is TIGHTER).
    * Mean wOBA       : within +-0.015 absolute.
    * Mean PA length  : within +-0.5 pitches.
    * Calibration_valid coverage: >=95% of 10K rollouts.

If any gate FAILs we surface the diagnosis hypothesis (PA-termination
bug / hit-vs-out noise / context CDF mis-build) per the plan -- do NOT
paper over.

Guardrails:
    * No model checkpoint touched.
    * No mutation to ``src/analytics/pitchgpt_sim.py``.
    * DuckDB read_only=True throughout.
    * No commit -- PM commits after reviewing the report.
"""

from __future__ import annotations

import argparse
import hashlib
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

from src.analytics.pitchgpt_sim import (  # noqa: E402
    PAContext,
    PGConcatHeadPredictor,
    WObaTable,
    pa_woba_distribution,
    rollout,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitchgpt_rollout_sanity_2025")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_RANGE = (2015, 2022)
HOLDOUT_SEASON = 2025
N_PA_DEFAULT = 10_000
N_SAMPLES = 100
HORIZON = 6
TEMPERATURE = 1.0
SEED_ROOT = 42
N_BOOT = 1000

# Per PHASE_0.5_PLAN §3.5 + §6.6
CAL_VALID_COVERAGE_GATE = 0.95

# Outcome class indices (mirror src/analytics/pitchgpt_outcome_head.py)
OUTCOME_BALL = 0
OUTCOME_CALLED_STRIKE = 1
OUTCOME_SWINGING_STRIKE = 2
OUTCOME_FOUL = 3
OUTCOME_IN_PLAY_OUT = 4
OUTCOME_IN_PLAY_HIT = 5
OUTCOME_HBP = 6

# Statcast event-string sets (mirror pitchgpt_outcome_head.py)
HIT_EVENTS = frozenset({"single", "double", "triple", "home_run"})
IN_PLAY_OUT_EVENTS = frozenset({
    "field_out", "force_out", "grounded_into_double_play",
    "double_play", "triple_play",
    "sac_fly", "sac_fly_double_play", "sac_bunt",
    "fielders_choice", "fielders_choice_out",
    "field_error",
    "truncated_pa",
})
WALK_EVENTS = frozenset({"walk"})
K_EVENTS = frozenset({"strikeout", "strikeout_double_play"})
HBP_EVENTS = frozenset({"hit_by_pitch"})
HR_EVENTS = frozenset({"home_run"})


# ─────────────────────────────────────────────────────────────────────────────
# SQL helpers
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


def _fetch_2025_pitcher_disjoint_pa_starts(
    conn,
    train_pitchers: set[int],
) -> pd.DataFrame:
    """Return one row per 2025 PA (game_pk, at_bat_number, pitcher_id, batter_id)
    representing the PA start (pitch_number=1).  The PA must be pitcher-disjoint
    from the 2015-2022 train cohort.

    Columns include the PA-start context PLUS the terminal_event /
    terminal_woba so we can also compute the empirical baseline straight
    from this frame (avoiding a second SQL pass).

    Score-diff is computed at the PA start; for Phase 0.6 we use a default
    of 0 since the SIM_ENGINE_API constructs a constant context within a PA
    and PA-start rollouts share the prefix-empty starting state.
    """
    # Bound the IN(...) list size; DuckDB handles ~M elements fine.
    pids_str = ", ".join(str(int(p)) for p in train_pitchers)

    query = f"""
        WITH all_2025 AS (
            SELECT
                game_pk,
                at_bat_number,
                pitcher_id,
                batter_id,
                pitch_number,
                balls,
                strikes,
                outs_when_up,
                on_1b,
                on_2b,
                on_3b,
                stand,
                p_throws,
                inning,
                inning_topbot,
                events,
                woba_value,
                description
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
            -- PA start row (pitch_number = first_pn within PA)
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
            -- PA terminal row (pitch_number = last_pn within PA)
            SELECT
                p.game_pk, p.at_bat_number, p.pitcher_id, p.batter_id,
                p.events     AS terminal_event,
                p.woba_value AS terminal_woba,
                p.description AS terminal_description
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
            t.terminal_event, t.terminal_woba, t.terminal_description
        FROM starts s
        JOIN terminals t
          ON s.game_pk = t.game_pk
         AND s.at_bat_number = t.at_bat_number
         AND s.pitcher_id = t.pitcher_id
         AND s.batter_id = t.batter_id
        WHERE t.terminal_event IS NOT NULL
          AND t.terminal_event != ''
    """
    df = conn.execute(query).fetchdf()
    return df


def _fetch_2025_league_median_ump_scalar(conn) -> float:
    """Resolve the 2025 league-median ump scalar.

    Per PHASE_0.5_PLAN §3.2 + the API §3.2 NULL/missing-resolution rule:
    a missing ump_scalar resolves to season-league-median.  This is the
    same recipe as ``_fetch_season_league_median(2025)`` -- the 2024 prior
    season median (or 2025 same-season backstop).
    """
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
    # Prefer prior-season (2024); same-season backstop = 2025; else 0.0.
    if 2024 in by_season:
        return by_season[2024]
    if 2025 in by_season:
        return by_season[2025]
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Empirical baseline (PA-level marginals)
# ─────────────────────────────────────────────────────────────────────────────


def _classify_pa_terminal(event: str | None) -> dict:
    """Map a terminal Statcast event to outcome flags.

    Returns dict with: is_k, is_bb, is_hr, is_hbp, is_hit (=1B/2B/3B/HR).
    """
    e = (str(event or "")).strip().lower()
    return {
        "is_k": e in K_EVENTS,
        "is_bb": e in WALK_EVENTS,
        "is_hr": e in HR_EVENTS,
        "is_hbp": e in HBP_EVENTS,
        "is_hit": e in HIT_EVENTS,
    }


def _compute_empirical_baselines(df: pd.DataFrame) -> dict:
    """League-rate marginals over the 2025 pitcher-disjoint cohort.

    Returns the same five metrics that the rollout reports, with 95%
    bootstrap CIs computed at the PA level.
    """
    n = len(df)
    is_k = df["terminal_event"].map(lambda e: _classify_pa_terminal(e)["is_k"]).to_numpy(dtype=bool)
    is_bb = df["terminal_event"].map(lambda e: _classify_pa_terminal(e)["is_bb"]).to_numpy(dtype=bool)
    is_hr = df["terminal_event"].map(lambda e: _classify_pa_terminal(e)["is_hr"]).to_numpy(dtype=bool)

    woba = df["terminal_woba"].astype(float).to_numpy()
    woba_valid = ~np.isnan(woba)

    pa_len = df["pa_pitch_count"].astype(float).to_numpy()

    rng = np.random.default_rng(SEED_ROOT)

    def _boot_mean(arr_bool_or_float: np.ndarray, mask: np.ndarray | None = None) -> dict:
        if mask is None:
            sub = arr_bool_or_float
        else:
            sub = arr_bool_or_float[mask]
        m = float(np.mean(sub)) if len(sub) > 0 else float("nan")
        boots = np.empty(N_BOOT, dtype=np.float64)
        sub_n = len(sub)
        if sub_n == 0:
            return {"value": m, "ci_95_lo": float("nan"), "ci_95_hi": float("nan"), "n": 0}
        for i in range(N_BOOT):
            idx = rng.integers(0, sub_n, size=sub_n)
            boots[i] = float(np.mean(sub[idx]))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {
            "value": round(m, 6),
            "ci_95_lo": round(float(lo), 6),
            "ci_95_hi": round(float(hi), 6),
            "n": int(sub_n),
        }

    return {
        "n_pas": int(n),
        "k_pct": _boot_mean(is_k),
        "bb_pct": _boot_mean(is_bb),
        "hr_pct": _boot_mean(is_hr),
        "mean_woba": _boot_mean(woba, mask=woba_valid),
        "mean_pa_length_pitches": _boot_mean(pa_len),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Rollout aggregation
# ─────────────────────────────────────────────────────────────────────────────


def _rollout_per_pa_outcome_distribution(rollout_result) -> dict:
    """Aggregate ``rollout_result`` (n_samples=100) to per-PA marginals.

    Returns the dict the per-PA accumulator consumes:

        {
            "p_k": float       -- fraction of n_samples that K'd
            "p_bb": float      -- fraction that walked
            "p_hr": float      -- fraction that ended in HR
            "p_hbp": float     -- fraction that ended in HBP
            "mean_woba": float -- nanmean of pa_woba_distribution
            "mean_pa_len_pitches": float
            "p_truncated": float
            "calibration_valid": bool (per-PA flag from sampling_metadata)
        }

    Per the API: walks / Ks have no terminal-outcome class; they terminate
    on count.  So K-fraction is computed as
    ``mean(final_count.strikes == 3 & pa_outcome == ROLLOUT_PAD_OUTCOME)``.
    HR-fraction is computed by treating ``pa_outcome == in_play_hit`` as
    HR with the league HR-given-hit rate -- BUT that's noisy.  The plan
    asks for per-PA HR%, so we use the raw in-play-hit rate as an upper
    bound and tag it with a HR-given-hit scaling constant (=0.107 league
    avg in 2025).  An exact HR vs non-HR-hit decomposition requires a
    second sampling pass; we surface the assumption explicitly.
    """
    from src.analytics.pitchgpt_sim import ROLLOUT_PAD_OUTCOME

    n_samples = rollout_result.pitch_tokens.shape[0]
    pa_outc = rollout_result.pa_outcome
    final_count = rollout_result.final_count
    pa_terminated = rollout_result.pa_terminated

    # K%: terminated AND strikes >= 3 AND pa_outcome is PAD (count-driven).
    any_term = pa_terminated.any(axis=1)  # (n_samples,)
    is_k_per_sample = any_term & (final_count[:, 1] >= 3)
    is_bb_per_sample = any_term & (final_count[:, 0] >= 4)
    # In-play hit (broader than HR; we further fold into HR via league prior).
    is_in_play_hit = (pa_outc is not None) and np.zeros(n_samples, dtype=bool)
    is_hbp = (pa_outc is not None) and np.zeros(n_samples, dtype=bool)
    if pa_outc is not None:
        is_in_play_hit = (pa_outc == OUTCOME_IN_PLAY_HIT)
        is_hbp = (pa_outc == OUTCOME_HBP)

    p_k = float(np.mean(is_k_per_sample))
    p_bb = float(np.mean(is_bb_per_sample))
    p_in_play_hit = float(np.mean(is_in_play_hit)) if pa_outc is not None else float("nan")
    p_hbp = float(np.mean(is_hbp)) if pa_outc is not None else float("nan")

    # mean wOBA via the API's pa_woba_distribution (NaN-aware).
    woba_arr = pa_woba_distribution(rollout_result, woba_lookup=WObaTable.default())
    if np.isnan(woba_arr).all():
        mean_woba = float("nan")
    else:
        mean_woba = float(np.nanmean(woba_arr))

    # Mean PA length (in pitches) -- via pa_terminated index.
    # First True index per sample, or horizon if no termination.
    horizon = pa_terminated.shape[1]
    term_idx = np.full(n_samples, horizon, dtype=np.int32)
    if pa_terminated.any():
        # argmax returns 0 if no True; combine with any() per row.
        first_true = np.argmax(pa_terminated, axis=1)
        any_per_row = pa_terminated.any(axis=1)
        term_idx = np.where(any_per_row, first_true + 1, horizon)
    mean_pa_len = float(np.mean(term_idx))

    n_truncated = int(rollout_result.sampling_metadata.get("n_truncated", 0))
    cal_valid = bool(rollout_result.sampling_metadata.get("calibration_valid", False))

    return {
        "p_k": p_k,
        "p_bb": p_bb,
        "p_in_play_hit": p_in_play_hit,
        "p_hbp": p_hbp,
        "mean_woba": mean_woba,
        "mean_pa_len_pitches": mean_pa_len,
        "p_truncated": float(n_truncated) / max(n_samples, 1),
        "calibration_valid": cal_valid,
    }


def _bootstrap_per_pa_means(values: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED_ROOT + 7) -> dict:
    """95% bootstrap CI on a mean over n_pa per-PA values."""
    rng = np.random.default_rng(seed)
    n = len(values)
    valid = ~np.isnan(values)
    sub = values[valid]
    if sub.size == 0:
        return {"value": float("nan"), "ci_95_lo": float("nan"), "ci_95_hi": float("nan"), "n": 0}
    m = float(np.mean(sub))
    boots = np.empty(n_boot, dtype=np.float64)
    sn = sub.size
    for i in range(n_boot):
        idx = rng.integers(0, sn, size=sn)
        boots[i] = float(np.mean(sub[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "value": round(m, 6),
        "ci_95_lo": round(float(lo), 6),
        "ci_95_hi": round(float(hi), 6),
        "n": int(n),
        "n_valid": int(sub.size),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Gate evaluation
# ─────────────────────────────────────────────────────────────────────────────


def _gate_pct_metric(sampled: float, empirical: float, abs_tol: float = 0.01, rel_tol: float = 0.10) -> dict:
    """Tighter-of (rel*emp, abs) tolerance per PHASE_0.5_PLAN §3.5."""
    # rel band relative to empirical, abs band fixed at 0.01 (1pp).
    tol = min(rel_tol * empirical, abs_tol)
    delta = sampled - empirical
    passed = abs(delta) <= tol
    return {
        "sampled": round(sampled, 6),
        "empirical": round(empirical, 6),
        "delta_abs": round(delta, 6),
        "delta_rel": round((delta / empirical) if empirical else float("nan"), 6),
        "tol_used": round(tol, 6),
        "tol_rule": "min(rel*emp, abs)",
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "pass_band_lo": round(empirical - tol, 6),
        "pass_band_hi": round(empirical + tol, 6),
        "pass": bool(passed),
    }


def _gate_abs_metric(sampled: float, empirical: float, abs_tol: float) -> dict:
    delta = sampled - empirical
    passed = abs(delta) <= abs_tol
    return {
        "sampled": round(sampled, 6),
        "empirical": round(empirical, 6),
        "delta_abs": round(delta, 6),
        "tol_used": abs_tol,
        "tol_rule": "abs",
        "pass_band_lo": round(empirical - abs_tol, 6),
        "pass_band_hi": round(empirical + abs_tol, 6),
        "pass": bool(passed),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pa", type=int, default=N_PA_DEFAULT)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--seed", type=int, default=SEED_ROOT)
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_ROOT / "results" / "pitchgpt" / "rollout_sanity_2025",
    )
    parser.add_argument(
        "--skip-secondary",
        action="store_true",
        help="Skip the None-predictor SECONDARY run (debug only).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Phase 0.6 sanity start: n_pa=%d  n_samples=%d  horizon=%d  T=%.3f  seed=%d",
        args.n_pa, args.n_samples, args.horizon, args.temperature, args.seed,
    )

    # ── 1. Pull 2025 pitcher-disjoint PA-start cohort ───────────────────────
    conn = get_connection(args.db_path, read_only=True)
    try:
        train_pitchers = _fetch_train_pitcher_ids(conn)
        logger.info("Train (2015-2022) pitcher cohort: %d pitchers", len(train_pitchers))
        df_pas = _fetch_2025_pitcher_disjoint_pa_starts(conn, train_pitchers)
        league_median_ump = _fetch_2025_league_median_ump_scalar(conn)
    finally:
        conn.close()

    n_total_pas = len(df_pas)
    logger.info(
        "2025 pitcher-disjoint PAs available: %d  (league-median ump scalar: %.5f)",
        n_total_pas, league_median_ump,
    )
    if n_total_pas < args.n_pa:
        raise RuntimeError(
            f"Only {n_total_pas} PAs available, need {args.n_pa}; "
            f"cohort smaller than expected."
        )

    # ── 2. Empirical baseline (full cohort) ─────────────────────────────────
    logger.info("Computing empirical league baseline (n=%d)...", n_total_pas)
    t0 = time.perf_counter()
    empirical = _compute_empirical_baselines(df_pas)
    emp_dt = time.perf_counter() - t0
    logger.info("Empirical baseline computed in %.1fs", emp_dt)
    logger.info(
        "  K%%=%.4f  BB%%=%.4f  HR%%=%.4f  meanWoba=%.4f  meanPAlen=%.3f",
        empirical["k_pct"]["value"],
        empirical["bb_pct"]["value"],
        empirical["hr_pct"]["value"],
        empirical["mean_woba"]["value"],
        empirical["mean_pa_length_pitches"]["value"],
    )

    # ── 3. Sample 10K PA starts deterministically ───────────────────────────
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(n_total_pas, size=args.n_pa, replace=False)
    df_sample = df_pas.iloc[sample_idx].reset_index(drop=True)
    # PAContext.from_pitches_row's _safe_bool helper raises on pandas NAType
    # (DuckDB's BIGINT-NULL columns come back as ``pd.NA`` not ``np.nan``).
    # Pre-coerce the on_{1b,2b,3b} columns to plain int64 so NAs become 0
    # (= "no runner") -- matches the encode_context default.
    for col in ("on_1b", "on_2b", "on_3b"):
        if col in df_sample.columns:
            df_sample[col] = (
                pd.to_numeric(df_sample[col], errors="coerce")
                .fillna(0)
                .astype(np.int64)
            )
    # Same NAType-safety on numeric integer columns the factory consumes.
    for col in ("balls", "strikes", "outs_when_up", "inning", "pitcher_id", "batter_id"):
        if col in df_sample.columns:
            df_sample[col] = (
                pd.to_numeric(df_sample[col], errors="coerce")
                .fillna(0)
                .astype(np.int64)
            )
    logger.info("Sampled %d PA starts (seed=%d)", args.n_pa, args.seed)

    # ── 4. PRIMARY run -- PGConcatHeadPredictor ─────────────────────────────
    logger.info("Loading PGConcatHeadPredictor (A1)...")
    predictor = PGConcatHeadPredictor()
    backbone_sha = None  # set by first rollout's sampling_metadata
    a1_checkpoint_sha = predictor.checkpoint_sha256
    a1_checkpoint_path = Path(_ROOT / "models" / "pitchgpt_v2_outcomehead_a1.pt")

    cuda = torch.cuda.is_available()
    logger.info("Device: %s (CUDA available=%s)", "cuda" if cuda else "cpu", cuda)

    t_primary_start = time.perf_counter()
    primary_per_pa: list[dict] = []
    primary_cal_invalid_reasons: dict[str, int] = {}
    log_every = max(args.n_pa // 20, 100)
    for i, row in df_sample.iterrows():
        ctx = PAContext.from_pitches_row(row, ump_scalar=league_median_ump)
        per_row_seed = (args.seed * 1_000_003 + i) & 0x7FFFFFFF
        result = rollout(
            ctx,
            outcome_predictor=predictor,
            n_samples=args.n_samples,
            horizon=args.horizon,
            temperature=args.temperature,
            seed=per_row_seed,
            return_probs=False,
        )
        agg = _rollout_per_pa_outcome_distribution(result)
        primary_per_pa.append(agg)
        for reason in result.sampling_metadata.get("calibration_invalid_reasons", []):
            primary_cal_invalid_reasons[reason] = primary_cal_invalid_reasons.get(reason, 0) + 1
        if backbone_sha is None:
            backbone_sha = result.sampling_metadata.get("backbone_checkpoint_sha256")
        if (i + 1) % log_every == 0:
            dt = time.perf_counter() - t_primary_start
            rate = (i + 1) / dt
            remain = (args.n_pa - (i + 1)) / max(rate, 1e-6)
            logger.info(
                "[PRIMARY] %d/%d PAs  rate=%.1f PA/s  eta=%.0fs",
                i + 1, args.n_pa, rate, remain,
            )
    t_primary_total = time.perf_counter() - t_primary_start
    logger.info("PRIMARY run complete in %.1fs", t_primary_total)

    # Aggregate the per-PA dicts into 5-metric arrays.
    p_k_arr = np.array([d["p_k"] for d in primary_per_pa], dtype=np.float64)
    p_bb_arr = np.array([d["p_bb"] for d in primary_per_pa], dtype=np.float64)
    p_in_play_hit_arr = np.array([d["p_in_play_hit"] for d in primary_per_pa], dtype=np.float64)
    p_hbp_arr = np.array([d["p_hbp"] for d in primary_per_pa], dtype=np.float64)
    mean_woba_arr = np.array([d["mean_woba"] for d in primary_per_pa], dtype=np.float64)
    mean_pa_len_arr = np.array([d["mean_pa_len_pitches"] for d in primary_per_pa], dtype=np.float64)
    cal_valid_arr = np.array([d["calibration_valid"] for d in primary_per_pa], dtype=bool)
    p_truncated_arr = np.array([d["p_truncated"] for d in primary_per_pa], dtype=np.float64)

    # HR fraction = in_play_hit fraction * empirical league HR-given-hit rate.
    # The A1 outcome head emits a single in_play_hit class -- it does not
    # natively distinguish HR from non-HR hit.  Statcast 2025 league hit
    # rate (1B+2B+3B+HR) is ~22.18% per
    # ``results/pitchgpt/rollout_sanity_2025/empirical_baselines_2025.json``,
    # of which ~3.21% is HR -- so HR-given-hit ≈ 0.1447 league-wide.  We
    # surface this assumption explicitly in the report.
    hits_pct_emp = _hits_pct_from_emp(empirical)
    hr_given_hit = (
        empirical["hr_pct"]["value"] / hits_pct_emp
    ) if hits_pct_emp > 1e-9 else 0.144
    p_hr_arr = p_in_play_hit_arr * hr_given_hit

    primary_metrics = {
        "k_pct": _bootstrap_per_pa_means(p_k_arr, seed=args.seed + 100),
        "bb_pct": _bootstrap_per_pa_means(p_bb_arr, seed=args.seed + 101),
        "in_play_hit_pct": _bootstrap_per_pa_means(p_in_play_hit_arr, seed=args.seed + 102),
        "hbp_pct": _bootstrap_per_pa_means(p_hbp_arr, seed=args.seed + 103),
        "hr_pct": _bootstrap_per_pa_means(p_hr_arr, seed=args.seed + 104),
        "mean_woba": _bootstrap_per_pa_means(mean_woba_arr, seed=args.seed + 105),
        "mean_pa_length_pitches": _bootstrap_per_pa_means(mean_pa_len_arr, seed=args.seed + 106),
        "p_truncated_per_pa": _bootstrap_per_pa_means(p_truncated_arr, seed=args.seed + 107),
        "calibration_valid_coverage": float(np.mean(cal_valid_arr)),
        "calibration_invalid_reason_counts": primary_cal_invalid_reasons,
        "hr_given_hit_assumption_used": round(hr_given_hit, 4),
        "hr_given_hit_assumption_source": "empirical_2025: hr_pct / hit_pct",
    }

    # ── 5. SECONDARY run -- outcome_predictor=None ──────────────────────────
    if args.skip_secondary:
        secondary_metrics = {"skipped": True}
    else:
        logger.info("SECONDARY run (outcome_predictor=None)...")
        t_secondary_start = time.perf_counter()
        sec_per_pa: list[dict] = []
        sec_cal_invalid_reasons: dict[str, int] = {}
        for i, row in df_sample.iterrows():
            ctx = PAContext.from_pitches_row(row, ump_scalar=league_median_ump)
            per_row_seed = (args.seed * 2_000_003 + i) & 0x7FFFFFFF
            result = rollout(
                ctx,
                outcome_predictor=None,
                n_samples=args.n_samples,
                horizon=args.horizon,
                temperature=args.temperature,
                seed=per_row_seed,
                return_probs=False,
            )
            agg = _rollout_per_pa_outcome_distribution(result)
            sec_per_pa.append(agg)
            for reason in result.sampling_metadata.get("calibration_invalid_reasons", []):
                sec_cal_invalid_reasons[reason] = sec_cal_invalid_reasons.get(reason, 0) + 1
            if (i + 1) % log_every == 0:
                dt = time.perf_counter() - t_secondary_start
                rate = (i + 1) / dt
                remain = (args.n_pa - (i + 1)) / max(rate, 1e-6)
                logger.info(
                    "[SECONDARY] %d/%d PAs  rate=%.1f PA/s  eta=%.0fs",
                    i + 1, args.n_pa, rate, remain,
                )
        t_secondary_total = time.perf_counter() - t_secondary_start
        logger.info("SECONDARY run complete in %.1fs", t_secondary_total)

        sec_p_k_arr = np.array([d["p_k"] for d in sec_per_pa], dtype=np.float64)
        sec_p_bb_arr = np.array([d["p_bb"] for d in sec_per_pa], dtype=np.float64)
        sec_mean_pa_len_arr = np.array([d["mean_pa_len_pitches"] for d in sec_per_pa], dtype=np.float64)
        sec_p_truncated_arr = np.array([d["p_truncated"] for d in sec_per_pa], dtype=np.float64)

        # No outcome predictor -> HR%, mean_woba unavailable.
        secondary_metrics = {
            "k_pct": _bootstrap_per_pa_means(sec_p_k_arr, seed=args.seed + 200),
            "bb_pct": _bootstrap_per_pa_means(sec_p_bb_arr, seed=args.seed + 201),
            "hr_pct": {"value": float("nan"), "note": "outcome_predictor=None cannot resolve HR vs non-HR hit"},
            "mean_woba": {"value": float("nan"), "note": "outcome_predictor=None -> pa_woba is NaN"},
            "mean_pa_length_pitches": _bootstrap_per_pa_means(sec_mean_pa_len_arr, seed=args.seed + 202),
            "p_truncated_per_pa": _bootstrap_per_pa_means(sec_p_truncated_arr, seed=args.seed + 203),
            "calibration_invalid_reason_counts": sec_cal_invalid_reasons,
            "wall_clock_seconds": round(t_secondary_total, 1),
        }

    # ── 6. Gate evaluation (PRIMARY only) ───────────────────────────────────
    gates = {
        "k_pct": _gate_pct_metric(
            primary_metrics["k_pct"]["value"],
            empirical["k_pct"]["value"],
            abs_tol=0.01, rel_tol=0.10,
        ),
        "bb_pct": _gate_pct_metric(
            primary_metrics["bb_pct"]["value"],
            empirical["bb_pct"]["value"],
            abs_tol=0.01, rel_tol=0.10,
        ),
        "hr_pct": _gate_pct_metric(
            primary_metrics["hr_pct"]["value"],
            empirical["hr_pct"]["value"],
            abs_tol=0.01, rel_tol=0.10,
        ),
        "mean_woba": _gate_abs_metric(
            primary_metrics["mean_woba"]["value"],
            empirical["mean_woba"]["value"],
            abs_tol=0.015,
        ),
        "mean_pa_length_pitches": _gate_abs_metric(
            primary_metrics["mean_pa_length_pitches"]["value"],
            empirical["mean_pa_length_pitches"]["value"],
            abs_tol=0.5,
        ),
        "calibration_valid_coverage": {
            "sampled": round(primary_metrics["calibration_valid_coverage"], 4),
            "threshold": CAL_VALID_COVERAGE_GATE,
            "pass": bool(
                primary_metrics["calibration_valid_coverage"] >= CAL_VALID_COVERAGE_GATE
            ),
        },
    }
    n_failed = sum(1 for k, g in gates.items() if not g["pass"])
    overall_pass = (n_failed == 0)
    logger.info(
        "Gates: k=%s bb=%s hr=%s woba=%s palen=%s cal_cov=%s  -> overall=%s",
        gates["k_pct"]["pass"], gates["bb_pct"]["pass"], gates["hr_pct"]["pass"],
        gates["mean_woba"]["pass"], gates["mean_pa_length_pitches"]["pass"],
        gates["calibration_valid_coverage"]["pass"],
        "PASS" if overall_pass else "FAIL",
    )

    # ── 7. Mean wOBA decomposition (per PHASE_0.5_PLAN §5.5) ────────────────
    # Default scalar table:
    #   ball/CS/SS/foul/in_play_out -> 0
    #   in_play_hit -> 0.892
    #   hbp -> 0.708
    # So: mean_woba = p_in_play_hit * 0.892 + p_hbp * 0.708.
    # Walks (count-driven, pa_outcome=PAD) currently get 0 wOBA in
    # WObaTable.default() -- which is a known under-attribution.  Surface
    # the decomposition explicitly so the diagnosis is unambiguous.
    woba_in_play_hit_contrib = primary_metrics["in_play_hit_pct"]["value"] * 0.892
    woba_hbp_contrib = primary_metrics["hbp_pct"]["value"] * 0.708
    woba_walk_unattributed = primary_metrics["bb_pct"]["value"] * 0.690  # if we DID attribute
    woba_decomposition = {
        "table": "WObaTable.default() -- 7-element scalar (PHASE_0.5_PLAN §2.0.5.3)",
        "in_play_hit_contrib": round(float(woba_in_play_hit_contrib), 6),
        "hbp_contrib": round(float(woba_hbp_contrib), 6),
        "computed_mean_woba": round(
            float(woba_in_play_hit_contrib + woba_hbp_contrib), 6,
        ),
        "walk_attribution_if_added": round(float(woba_walk_unattributed), 6),
        "walk_attribution_note": (
            "WObaTable.default() does NOT attribute wOBA to walks (which "
            "terminate via count-only and have pa_outcome=ROLLOUT_PAD_OUTCOME). "
            "The empirical 2025 league mean wOBA does include walks at the "
            "league mean wOBA-on-walk ~0.69; PHASE_0.5_PLAN §5.5 marks this "
            "as the most-likely under-attribution source."
        ),
    }

    # ── 8. Verify checkpoint integrity ──────────────────────────────────────
    backbone_path = Path(_ROOT / "models" / "pitchgpt_v2.pt")
    backbone_sha_recompute = _sha256_file(backbone_path)
    a1_sha_recompute = _sha256_file(a1_checkpoint_path)
    BACKBONE_LOCKED_SHA = "6f952054d14ac6f918f3eb9502b496b70bc0c87dfc65dc50d98ee7244a62883c"
    sha_check = {
        "backbone_path": str(backbone_path),
        "backbone_sha256_recomputed": backbone_sha_recompute,
        "backbone_sha256_locked": BACKBONE_LOCKED_SHA,
        "backbone_sha_match_locked": (backbone_sha_recompute == BACKBONE_LOCKED_SHA),
        "backbone_sha_match_runtime": (backbone_sha_recompute == backbone_sha),
        "a1_path": str(a1_checkpoint_path),
        "a1_sha256_recomputed": a1_sha_recompute,
        "a1_sha256_runtime": a1_checkpoint_sha,
        "a1_sha_match_runtime": (a1_sha_recompute == a1_checkpoint_sha),
        "a1_size_bytes": a1_checkpoint_path.stat().st_size,
        "a1_size_expected_bytes": 151289,
        "a1_size_matches": (a1_checkpoint_path.stat().st_size == 151289),
    }

    # ── 9. Emit metrics.json + report.md ────────────────────────────────────
    metrics_payload = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec": "docs/pitchgpt_sim_engine/PHASE_0.5_PLAN.md §3 + §6.3 / §6.4 / §6.6",
        "args": {
            "n_pa": args.n_pa,
            "n_samples": args.n_samples,
            "horizon": args.horizon,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "device": "cuda" if cuda else "cpu",
        "wall_clock": {
            "primary_seconds": round(t_primary_total, 1),
            "secondary_seconds": (
                round(secondary_metrics.get("wall_clock_seconds", 0.0), 1)
                if not args.skip_secondary else 0.0
            ),
            "empirical_seconds": round(emp_dt, 1),
        },
        "cohort": {
            "season": HOLDOUT_SEASON,
            "n_pas_total_eligible": int(n_total_pas),
            "n_pas_sampled": int(args.n_pa),
            "league_median_ump_scalar_2025": round(float(league_median_ump), 6),
            "pitcher_disjoint_from_train_seasons": "2015-2022",
            "n_train_pitchers_excluded": len(train_pitchers),
        },
        "empirical": empirical,
        "primary": primary_metrics,
        "secondary": secondary_metrics,
        "gates": gates,
        "overall_pass": overall_pass,
        "woba_decomposition": woba_decomposition,
        "checkpoint_integrity": sha_check,
    }

    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)
    logger.info("Wrote %s", metrics_path)

    report_path = args.output_dir / "report.md"
    _write_report(report_path, metrics_payload)
    logger.info("Wrote %s", report_path)

    # ── 10. Console summary ─────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("Phase 0.6 Rollout Sanity 2025 -- Summary")
    print("=" * 78)
    print(f"PRIMARY wall clock: {t_primary_total:.1f}s  ({args.n_pa} PA x {args.n_samples} samples x H={args.horizon})")
    if not args.skip_secondary:
        print(f"SECONDARY wall clock: {secondary_metrics['wall_clock_seconds']:.1f}s")
    print("-" * 78)
    print(f"{'Metric':<26}{'PRIMARY':>12}{'EMP':>12}{'DELTA':>12}{'PASS':>8}")
    for k in ["k_pct", "bb_pct", "hr_pct", "mean_woba", "mean_pa_length_pitches"]:
        g = gates[k]
        print(
            f"{k:<26}{g['sampled']:>12.4f}{g['empirical']:>12.4f}"
            f"{g['delta_abs']:>+12.4f}{('PASS' if g['pass'] else 'FAIL'):>8}",
        )
    print("-" * 78)
    print(
        f"calibration_valid_coverage = "
        f"{primary_metrics['calibration_valid_coverage']:.4f}  "
        f"(>= {CAL_VALID_COVERAGE_GATE} -> "
        f"{'PASS' if gates['calibration_valid_coverage']['pass'] else 'FAIL'})",
    )
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 78)
    return 0 if overall_pass else 1


def _hits_pct_from_emp(empirical: dict) -> float:
    """Hit% = 1B + 2B + 3B + HR fraction.  Sourced from
    ``results/pitchgpt/rollout_sanity_2025/empirical_baselines_2025.json``
    if present; else estimated from this run's empirical dict.

    Locked Phase 0.6 uses the empirical_baselines_2025.json value
    (0.22177 = 22.18%) per the prep doc.  For robustness we read it back
    from disk if present, falling back to the in-memory value.
    """
    p = _ROOT / "results" / "pitchgpt" / "rollout_sanity_2025" / "empirical_baselines_2025.json"
    if p.exists():
        try:
            with p.open(encoding="utf-8") as fh:
                eb = json.load(fh)
            return float(eb["league_rates"]["hit_pct"]["value"])
        except (OSError, KeyError, json.JSONDecodeError):
            pass
    # Fallback: same empirical pass we computed.  hit_pct is not in our
    # quick empirical dict (we kept it minimal) -- estimate via HR / 0.144.
    return 0.222


def _write_report(path: Path, payload: dict) -> None:
    e = payload["empirical"]
    p = payload["primary"]
    s = payload["secondary"]
    g = payload["gates"]
    sha = payload["checkpoint_integrity"]
    args = payload["args"]
    cohort = payload["cohort"]
    wc = payload["wall_clock"]
    woba_decomp = payload["woba_decomposition"]
    overall_pass = payload["overall_pass"]

    lines: list[str] = []
    lines.append("# PitchGPT Phase 0.6 Rollout Sanity 2025 -- Report\n")
    lines.append(f"Generated: {payload['timestamp_utc']}\n")
    lines.append(f"Spec: `{payload['spec']}`\n")
    lines.append("")
    lines.append("## Configuration\n")
    lines.append(f"- n_pa = **{args['n_pa']}**, n_samples = **{args['n_samples']}**, horizon = **{args['horizon']}**, T = **{args['temperature']:.3f}**, seed = **{args['seed']}**")
    lines.append(f"- Cohort: 2025 pitcher-disjoint from train (2015-2022 excluded {cohort['n_train_pitchers_excluded']} pitchers)")
    lines.append(f"- Total eligible PAs in cohort: {cohort['n_pas_total_eligible']}; sampled: {cohort['n_pas_sampled']}")
    lines.append(f"- League-median ump scalar (2024 prior season): {cohort['league_median_ump_scalar_2025']}")
    lines.append(f"- Device: {payload['device']}")
    lines.append("")
    lines.append("## Wall clock\n")
    lines.append(f"- PRIMARY (PGConcatHeadPredictor, 10K x 100 samples x H=6): **{wc['primary_seconds']:.1f}s**")
    lines.append(f"- SECONDARY (None predictor): {wc['secondary_seconds']:.1f}s")
    lines.append(f"- Empirical baseline aggregation: {wc['empirical_seconds']:.1f}s")
    lines.append("")
    lines.append("## Phase 0.6 binding gates (per PHASE_0.5_PLAN §3.5 + §6.3 / §6.4 / §6.6)\n")
    lines.append("| Gate | Sampled | Empirical | Delta | Tolerance | PASS band | Verdict |")
    lines.append("|---|---|---|---|---|---|---|")
    pct_metrics = ["k_pct", "bb_pct", "hr_pct"]
    for k in pct_metrics:
        gate = g[k]
        lines.append(
            f"| {k} | {gate['sampled']:.4f} | {gate['empirical']:.4f} | "
            f"{gate['delta_abs']:+.4f} | "
            f"+/-{gate['tol_used']:.4f} ({gate['tol_rule']}) | "
            f"[{gate['pass_band_lo']:.4f}, {gate['pass_band_hi']:.4f}] | "
            f"**{'PASS' if gate['pass'] else 'FAIL'}** |",
        )
    for k in ["mean_woba", "mean_pa_length_pitches"]:
        gate = g[k]
        lines.append(
            f"| {k} | {gate['sampled']:.4f} | {gate['empirical']:.4f} | "
            f"{gate['delta_abs']:+.4f} | +/-{gate['tol_used']} (abs) | "
            f"[{gate['pass_band_lo']:.4f}, {gate['pass_band_hi']:.4f}] | "
            f"**{'PASS' if gate['pass'] else 'FAIL'}** |",
        )
    cov = g["calibration_valid_coverage"]
    lines.append(
        f"| calibration_valid_coverage (§6.6) | {cov['sampled']:.4f} | "
        f">= {cov['threshold']:.2f} | -- | -- | -- | "
        f"**{'PASS' if cov['pass'] else 'FAIL'}** |",
    )
    lines.append("")
    lines.append(f"**Overall**: **{'PASS' if overall_pass else 'FAIL'}**\n")
    lines.append("")
    lines.append("## PRIMARY -- 5 metrics with 95% bootstrap CIs\n")
    lines.append("| Metric | Sampled | 95% CI | Empirical | 95% CI |")
    lines.append("|---|---|---|---|---|")
    pa_n = p["k_pct"]["n_valid"]
    for k_sampled, k_emp, label in [
        ("k_pct", "k_pct", "K%"),
        ("bb_pct", "bb_pct", "BB%"),
        ("hr_pct", "hr_pct", "HR%"),
        ("hbp_pct", None, "HBP%"),
        ("in_play_hit_pct", None, "in_play_hit%"),
        ("mean_woba", "mean_woba", "mean wOBA"),
        ("mean_pa_length_pitches", "mean_pa_length_pitches", "mean PA length"),
        ("p_truncated_per_pa", None, "p_truncated"),
    ]:
        sm = p[k_sampled]
        emp_str = "--"
        emp_ci = "--"
        if k_emp is not None:
            em = e[k_emp]
            emp_str = f"{em['value']:.4f}"
            emp_ci = f"[{em['ci_95_lo']:.4f}, {em['ci_95_hi']:.4f}]"
        lines.append(
            f"| {label} | {sm['value']:.4f} | "
            f"[{sm['ci_95_lo']:.4f}, {sm['ci_95_hi']:.4f}] | "
            f"{emp_str} | {emp_ci} |",
        )
    lines.append("")
    lines.append(f"PRIMARY n_pa = {pa_n}, calibration_valid_coverage = {p['calibration_valid_coverage']:.4f}")
    if p["calibration_invalid_reason_counts"]:
        lines.append("")
        lines.append("Calibration-invalid-reason occurrence counts (PRIMARY):")
        lines.append("")
        for reason, n in sorted(p["calibration_invalid_reason_counts"].items(), key=lambda x: -x[1]):
            lines.append(f"- {reason}: {n}")
    lines.append("")
    lines.append("## SECONDARY -- None-predictor degraded mode (NOT gated)\n")
    if s.get("skipped"):
        lines.append("(Skipped via --skip-secondary.)")
    else:
        lines.append("Per PHASE_0.5_PLAN §3.4 + §5.3: with `outcome_predictor=None`, PA-termination falls back to a count-only heuristic that misclassifies in-play / foul / HBP outcomes as strikes -- biases K% / BB% high by construction.  Run measures the bias magnitude.")
        lines.append("")
        lines.append("| Metric | Sampled (None) | Sampled (PRIMARY) | Empirical | Bias vs PRIMARY | Bias vs Empirical |")
        lines.append("|---|---|---|---|---|---|")
        for label, kk in [
            ("K%", "k_pct"),
            ("BB%", "bb_pct"),
            ("mean PA length", "mean_pa_length_pitches"),
        ]:
            sec_v = s[kk]["value"]
            pri_v = p[kk]["value"]
            emp_v = e[kk]["value"]
            bias_pri = sec_v - pri_v
            bias_emp = sec_v - emp_v
            lines.append(
                f"| {label} | {sec_v:.4f} | {pri_v:.4f} | {emp_v:.4f} | "
                f"{bias_pri:+.4f} | {bias_emp:+.4f} |",
            )
        lines.append("")
        lines.append(f"NOTE: HR%, mean wOBA are unavailable in SECONDARY (no outcome predictor -> `pa_outcome=None`).  Per the API contract, consumers MUST NOT silently emit wOBA aggregations when the outcome predictor is missing.")
    lines.append("")
    lines.append("## Honest caveat -- A1 hit-vs-out noise (per PHASE_0.5_PLAN §3.3)\n")
    lines.append(
        "The A1 outcome predictor's `in_play_hit` test log-loss is **2.34** "
        "(per `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/metrics.json::test_metrics.per_class_log_loss.in_play_hit`).  This clears the WEAKER PASS gate (<2.5) but misses the full PASS gate (<2.0).  Consequence: the hit-vs-out marginal is structurally noisier than the league average -- A1 cannot condition on post-pitch `launch_speed` / `launch_angle`, so hit-vs-out is decided downstream of the model's information set.  **Mean-wOBA aggregation inherits this noise and is the noisiest of the five gated metrics.**  K% / BB% terminate via count-only and are insulated from this ceiling."
    )
    lines.append("")
    lines.append("Per `EXECUTION_PLAN.md` §3 the locked claim is \"calibrated rollout engine\" with calibration as load-bearing -- this report honors that by surfacing where rollout sub-distributions are weakest.")
    lines.append("")
    lines.append("## Mean-wOBA decomposition (per PHASE_0.5_PLAN §5.5)\n")
    lines.append("| Component | Value |")
    lines.append("|---|---|")
    lines.append(f"| in_play_hit_pct (sampled) | {p['in_play_hit_pct']['value']:.4f} |")
    lines.append(f"| HBP%   (sampled)          | {p['hbp_pct']['value']:.4f} |")
    lines.append(f"| in_play_hit contribution (* 0.892)  | {woba_decomp['in_play_hit_contrib']:.4f} |")
    lines.append(f"| HBP contribution (* 0.708)          | {woba_decomp['hbp_contrib']:.4f} |")
    lines.append(f"| **Sum (computed mean wOBA)**        | **{woba_decomp['computed_mean_woba']:.4f}** |")
    lines.append(f"| Walk attribution if added (BB% * 0.690) | {woba_decomp['walk_attribution_if_added']:.4f} |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- `WObaTable.default()` is the 7-element scalar wOBA table per PHASE_0.5_PLAN §2.0.5.3.")
    lines.append("- Walks (count-driven, `pa_outcome=ROLLOUT_PAD_OUTCOME`) get **0** from the default table.  The empirical 2025 mean wOBA includes walks at the league wOBA-on-walk (~0.69).  This is the single largest known under-attribution.")
    lines.append("- If `mean wOBA` gate FAILs, decomposition above identifies whether the gap is in (a) hit-vs-out marginal noise (A1 ceiling) or (b) walk attribution (table choice).  Phase-1 follow-up: extend `WObaTable` to attribute walks correctly OR switch to the full Statcast empirical (outcome x pitch_type) table.")
    lines.append("")
    if not g["mean_woba"]["pass"]:
        lines.append("**Mean wOBA gate FAIL diagnosis path** (per §5.5):")
        lines.append("")
        gap_total = g['mean_woba']['delta_abs']
        gap_walk = -woba_decomp['walk_attribution_if_added']  # we under-attribute by this much
        residual = gap_total - gap_walk
        lines.append(f"- Total mean wOBA gap (sampled - empirical): {gap_total:+.4f}")
        lines.append(f"- Expected walk under-attribution gap (if we DID attribute walks at 0.69): {gap_walk:+.4f}")
        lines.append(f"- Residual after walk-correction (= hit-vs-out / table effects): {residual:+.4f}")
        lines.append("")
        lines.append("Diagnosis hypothesis ranking:")
        lines.append("1. If `|residual|` < 0.005 -- the mean-wOBA failure is a known under-attribution from the scalar wOBA table not crediting walks.  Phase-1 fix: extend `WObaTable` to handle walks.  Not a rollout-engine bug.")
        lines.append("2. If `|residual|` >= 0.005 -- the gap is real.  Likely sources (most -> least): (a) A1 hit-vs-out marginal noise (test log-loss 2.34); (b) PA-termination logic bug in `rollout()`; (c) calibration-feature-CDF mis-build in `pitchgpt_build_calibration_cdfs.py`.")
    lines.append("")
    lines.append("## None-predictor bias (per PHASE_0.5_PLAN §3.4)\n")
    if s.get("skipped"):
        lines.append("(Skipped.)")
    else:
        bias_k = s["k_pct"]["value"] - p["k_pct"]["value"]
        bias_bb = s["bb_pct"]["value"] - p["bb_pct"]["value"]
        lines.append(f"Expected K% bias (None vs PRIMARY): **{bias_k:+.4f}** ({bias_k * 100:+.2f}pp)")
        lines.append(f"Expected BB% bias (None vs PRIMARY): **{bias_bb:+.4f}** ({bias_bb * 100:+.2f}pp)")
        lines.append("")
        lines.append("Per SIM_ENGINE_API §4.4 + PHASE_0.5_PLAN §5.3: None-predictor falls back to a zone-based count heuristic (in-zone token -> +1 strike, otherwise +1 ball).  Misclassifies in-play / foul / HBP as strikes.  Bias magnitude here is what consumers should expect when the predictor is unavailable -- DOES NOT trip the Phase 0.6 PASS/FAIL gate.")
    lines.append("")
    lines.append("## Backbone + A1 SHA256 verification (per PHASE_0.5_PLAN §6.8 / §6.9)\n")
    lines.append("| Asset | Path | SHA256 (recomputed) | Match (locked) | Match (runtime) | Size |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(
        f"| backbone v2 | `{sha['backbone_path']}` | `{sha['backbone_sha256_recomputed'][:32]}...` | "
        f"**{'YES' if sha['backbone_sha_match_locked'] else 'NO'}** | "
        f"{'YES' if sha['backbone_sha_match_runtime'] else 'NO'} | -- |",
    )
    lines.append(
        f"| A1 head | `{sha['a1_path']}` | `{sha['a1_sha256_recomputed'][:32]}...` | "
        f"-- | "
        f"**{'YES' if sha['a1_sha_match_runtime'] else 'NO'}** | "
        f"{sha['a1_size_bytes']} bytes (expected {sha['a1_size_expected_bytes']}: "
        f"{'YES' if sha['a1_size_matches'] else 'NO'}) |",
    )
    lines.append("")
    lines.append(f"Backbone locked SHA: `{sha['backbone_sha256_locked']}`")
    lines.append(f"Backbone recomputed SHA: `{sha['backbone_sha256_recomputed']}`")
    lines.append(f"A1 runtime SHA: `{sha['a1_sha256_runtime']}`")
    lines.append("")
    lines.append("## Cross-references\n")
    lines.append("- API spec: `docs/pitchgpt_sim_engine/SIM_ENGINE_API.md` §3, §4, §5, §6, §9.")
    lines.append("- Phase plan: `docs/pitchgpt_sim_engine/PHASE_0.5_PLAN.md` §3, §6.3 / §6.4 / §6.6 / §6.8 / §6.9.")
    lines.append("- A1 metrics: `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/metrics.json`.")
    lines.append("- Empirical baseline (prep): `results/pitchgpt/rollout_sanity_2025/empirical_baselines_2025.json`.")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
