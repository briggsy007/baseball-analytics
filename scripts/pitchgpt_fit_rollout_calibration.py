"""
Phase 0.6.2 — fit the rollout-regime per-position calibration table W.

Pre-registered protocol: ``docs/pitchgpt_sim_engine/PHASE_0.6.2_PLAN.md`` §4
(fit on 2023 ONLY — 2025 is never read) with the §10.A7 measurement-semantics
operationalization (2026-08-10 amendments, recorded pre-execution):

    roll-0 (raw T-softmax, no class_calibration, no pos-0)  ->  W1
    roll-1 (W1 applied)  -> converged?  (every 2023 per-position class
                            marginal within 1pp of empirical)
        yes -> final W = W1, n_iterations = 1
        no  -> W2 = W1 * (empirical / roll-1), roll-2 (W2) -> converged?
            yes -> final W = W2, n_iterations = 2
            no  -> KILL SIGNAL (PHASE_0.6.2_PLAN.md §6): NO artifact ships
                   to models/; the failed fit is archived under results/
                   and Phase 0.6 closes as FAIL.

W construction per §4 + §10.A7 (guard order):
    ratio[pos, c] = empirical[pos, c] / rollout[pos, c]
    1. any (pos, c) with < 500 rollout observations inherits ratio = 1.0
    2. floor/cap ratios at [0.2, 5.0]
    3. per-row geometric-mean normalization (cosmetic: the
       renormalize-after-multiply application is scale-invariant)
The iteration-2 update factor gets the same guard/cap; the combined W is
re-capped at [0.2, 5.0] and re-normalized.

Cohort (§4 step 1): 2023 pitcher-disjoint PA starts (pitchers seen in the
2015-2022 train split EXCLUDED — the same pitcher-disjoint recipe as A1's
temperature fit), subsampled to 10K PAs with seed 42, via the
season-parameterized cohort fetchers of ``scripts.pitchgpt_rollout_sanity_2025``
(Ticket 1).

Empirical target (§4 step 2): per-position class marginals of the REAL 2023
pitch sequences of the sampled PAs (within-PA pitch index 1..6 -> positions
0..5, truncated at the horizon), classified via ``classify_pitch_outcome``.

Outputs
-------
* ``models/calibration_rollout_perpos.npz``  (ONLY on convergence; write-once;
  must be committed together with this script — §3)
  Sidecar provenance schema (§10.A4): W, fit_cohort_season, fit_seed,
  fit_n_pas, fit_n_samples, horizon, n_iterations, converged,
  empirical_perpos, rollout_raw_perpos, rollout_final_perpos,
  rollout_obs_counts_raw, temperature, fit_date, backbone_sha256,
  a1_head_sha256, produced_by.
* ``results/pitchgpt/rollout_calibration_fit_2023/fit_audit.json`` + ``report.md``
  (always, converged or killed).

Guardrails
----------
* DuckDB read_only=True throughout.
* v2.pt + a1.pt SHA256 byte-identity asserted pre/post.
* 2025 never read (fit cohort is 2023; the artifact declares it, and
  ``PGConcatHeadPredictor`` refuses W artifacts declaring 2025/2026).
* No third fixed-point iteration under any circumstances (§4 step 5).
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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt_outcome_head import (  # noqa: E402
    NUM_OUTCOME_CLASSES,
    OUTCOME_CLASSES,
    classify_pitch_outcome,
)
from src.analytics.pitchgpt_sim import (  # noqa: E402
    PAContext,
    PGConcatHeadPredictor,
    rollout,
)
from src.db.schema import get_connection  # noqa: E402
from scripts.pitchgpt_rollout_sanity_2025 import (  # noqa: E402
    TRAIN_RANGE,
    _fetch_league_median_ump_scalar,
    _fetch_pitcher_disjoint_pa_starts,
    _fetch_train_pitcher_ids,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitchgpt_fit_rollout_calibration")

FIT_SEASON = 2023          # §4: 2023 ONLY.  2025 is never read.
N_PA_DEFAULT = 10_000
N_SAMPLES_DEFAULT = 100
HORIZON = 6
SEED_DEFAULT = 42
MIN_OBS_GUARD = 500        # §4 step 4
RATIO_FLOOR, RATIO_CAP = 0.2, 5.0
CONVERGENCE_PP = 0.01      # §4 step 5: every marginal within 1pp
FIT_DATE = "2026-08-10"

OUT_NPZ = _ROOT / "models" / "calibration_rollout_perpos.npz"
OUT_DIR = _ROOT / "results" / "pitchgpt" / "rollout_calibration_fit_2023"
BACKBONE_PATH = _ROOT / "models" / "pitchgpt_v2.pt"
A1_PATH = _ROOT / "models" / "pitchgpt_v2_outcomehead_a1.pt"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _coerce_ints(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0)
                .astype(np.int64)
            )
    return df


def _fetch_cohort_pitches(conn, train_pitchers: set[int]) -> pd.DataFrame:
    """All pitches of the 2023 pitcher-disjoint cohort with within-PA position."""
    pids_str = ", ".join(str(int(p)) for p in train_pitchers)
    query = f"""
        WITH all_szn AS (
            SELECT
                game_pk, at_bat_number, pitcher_id, batter_id, pitch_number,
                description, events
            FROM pitches
            WHERE EXTRACT(YEAR FROM game_date) = {FIT_SEASON}
              AND pitch_type IS NOT NULL
              AND pitcher_id IS NOT NULL
              AND batter_id IS NOT NULL
              AND at_bat_number IS NOT NULL
              AND pitch_number IS NOT NULL
              AND pitcher_id NOT IN ({pids_str})
        ),
        firsts AS (
            SELECT game_pk, at_bat_number, pitcher_id, batter_id,
                   MIN(pitch_number) AS first_pn
            FROM all_szn
            GROUP BY game_pk, at_bat_number, pitcher_id, batter_id
        )
        SELECT
            p.game_pk, p.at_bat_number, p.pitcher_id, p.batter_id,
            (p.pitch_number - f.first_pn) AS pa_position,
            p.description, p.events
        FROM all_szn p
        JOIN firsts f
          ON p.game_pk = f.game_pk
         AND p.at_bat_number = f.at_bat_number
         AND p.pitcher_id = f.pitcher_id
         AND p.batter_id = f.batter_id
        WHERE (p.pitch_number - f.first_pn) BETWEEN 0 AND {HORIZON - 1}
    """
    return conn.execute(query).fetchdf()


def _empirical_perpos_marginals(
    pitches: pd.DataFrame, sampled_keys: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-position (H x 7) empirical class marginals for the sampled PAs.

    Returns (marginals, counts) where counts[pos] = pitches observed at pos.
    """
    merged = pitches.merge(
        sampled_keys,
        on=["game_pk", "at_bat_number", "pitcher_id", "batter_id"],
        how="inner",
    )
    counts = np.zeros((HORIZON, NUM_OUTCOME_CLASSES), dtype=np.float64)
    cls = np.array(
        [
            classify_pitch_outcome(d, e)
            for d, e in zip(merged["description"], merged["events"])
        ],
        dtype=np.int64,
    )
    pos = merged["pa_position"].to_numpy(dtype=np.int64)
    valid = (cls >= 0) & (cls < NUM_OUTCOME_CLASSES)
    for p in range(HORIZON):
        sel = valid & (pos == p)
        if sel.any():
            counts[p] = np.bincount(
                cls[sel], minlength=NUM_OUTCOME_CLASSES,
            ).astype(np.float64)
    marg = counts / np.clip(counts.sum(axis=1, keepdims=True), 1.0, None)
    return marg, counts


def _roll_marginals(
    df_sample: pd.DataFrame,
    predictor,
    league_median_ump: float,
    n_samples: int,
    seed_base: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Rollout the sampled PA starts; per-position (H x 7) outcome marginals.

    Returns (marginals, counts) where counts[pos, c] = sampled outcomes of
    class c observed at position pos (alive samples only).
    """
    counts = np.zeros((HORIZON, NUM_OUTCOME_CLASSES), dtype=np.float64)
    t0 = time.perf_counter()
    n_pa = len(df_sample)
    log_every = max(n_pa // 10, 100)
    for i, row in df_sample.iterrows():
        ctx = PAContext.from_pitches_row(row, ump_scalar=league_median_ump)
        per_row_seed = (seed_base + i) & 0x7FFFFFFF
        result = rollout(
            ctx,
            outcome_predictor=predictor,
            n_samples=n_samples,
            horizon=HORIZON,
            temperature=1.0,
            seed=per_row_seed,
            return_probs=False,
        )
        if result.outcomes is None:
            continue
        outc = result.outcomes  # (n_samples, H); PAD past termination
        for p in range(HORIZON):
            col = outc[:, p]
            in_range = (col >= 0) & (col < NUM_OUTCOME_CLASSES)
            if in_range.any():
                counts[p] += np.bincount(
                    col[in_range].astype(np.int64),
                    minlength=NUM_OUTCOME_CLASSES,
                ).astype(np.float64)
        if (i + 1) % log_every == 0:
            dt = time.perf_counter() - t0
            rate = (i + 1) / dt
            eta = (n_pa - (i + 1)) / max(rate, 1e-6)
            logger.info(
                "[%s] %d/%d PAs  rate=%.1f PA/s  eta=%.0fs",
                label, i + 1, n_pa, rate, eta,
            )
    logger.info("[%s] complete in %.1fs", label, time.perf_counter() - t0)
    marg = counts / np.clip(counts.sum(axis=1, keepdims=True), 1.0, None)
    return marg, counts


def _guarded_ratio(
    empirical: np.ndarray, rollout_m: np.ndarray, rollout_counts: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """§4 step 4 ratio with guard order per §10.A7. Returns (ratio, audit)."""
    safe_roll = np.where(rollout_m > 1e-12, rollout_m, 1e-12)
    ratio = empirical / safe_roll
    guard_mask = rollout_counts < MIN_OBS_GUARD
    ratio = np.where(guard_mask, 1.0, ratio)
    n_capped = int(np.sum((ratio < RATIO_FLOOR) | (ratio > RATIO_CAP)))
    ratio = np.clip(ratio, RATIO_FLOOR, RATIO_CAP)
    # Per-row geometric-mean normalization (cosmetic; scale-invariant apply).
    log_w = np.log(np.clip(ratio, 1e-12, None))
    log_w = log_w - log_w.mean(axis=1, keepdims=True)
    ratio = np.exp(log_w)
    audit = {
        "n_guarded_lt_500_obs": int(guard_mask.sum()),
        "n_capped_before_norm": n_capped,
    }
    return ratio, audit


def _convergence(
    marg: np.ndarray, empirical: np.ndarray,
) -> tuple[bool, float, np.ndarray]:
    """§4 step 5: every per-position class marginal within 1pp of empirical."""
    delta = np.abs(marg - empirical)
    return bool(np.all(delta <= CONVERGENCE_PP)), float(delta.max()), delta


def _write_temp_w(w: np.ndarray, tag: str, tmp_dir: Path) -> Path:
    """Write an intermediate W npz (fit-cohort-declared) for the W rolls."""
    tmp_dir.mkdir(parents=True, exist_ok=True)
    p = tmp_dir / f"W_{tag}.npz"
    np.savez(
        p,
        W=w.astype(np.float64),
        fit_cohort_season=np.int64(FIT_SEASON),
        intermediate=np.bool_(True),
    )
    return p


def _marg_table(marg: np.ndarray) -> list[list[float]]:
    return [[round(float(v), 6) for v in row] for row in marg]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pa", type=int, default=N_PA_DEFAULT)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Fetch cohort + empirical target only; no GPU rolls, no artifact.",
    )
    parser.add_argument(
        "--tmp-dir", type=Path,
        default=OUT_DIR / "intermediate",
        help="Where intermediate W npz files for the measurement rolls go.",
    )
    args = parser.parse_args()

    if OUT_NPZ.exists():
        raise RuntimeError(
            f"{OUT_NPZ} already exists — the W artifact is write-once "
            "(PHASE_0.6.2_PLAN.md §3). Remove/rename manually only via an "
            "orchestrator decision."
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bb_sha_pre = _sha256_file(BACKBONE_PATH)
    a1_sha_pre = _sha256_file(A1_PATH)
    logger.info("v2.pt SHA256 (pre): %s", bb_sha_pre)
    logger.info("a1.pt SHA256 (pre): %s", a1_sha_pre)

    # ── 1. Cohort (2023 pitcher-disjoint; 2025 never read) ──────────────
    conn = get_connection(args.db_path, read_only=True)
    try:
        train_pitchers = _fetch_train_pitcher_ids(conn)
        logger.info(
            "Train (%d-%d) pitcher cohort: %d pitchers",
            TRAIN_RANGE[0], TRAIN_RANGE[1], len(train_pitchers),
        )
        df_pas = _fetch_pitcher_disjoint_pa_starts(
            conn, train_pitchers, FIT_SEASON,
        )
        league_median_ump = _fetch_league_median_ump_scalar(conn, FIT_SEASON)
        pitches = _fetch_cohort_pitches(conn, train_pitchers)
    finally:
        conn.close()
    n_total = len(df_pas)
    logger.info(
        "%d pitcher-disjoint PAs: %d (pitch rows in horizon: %d; "
        "ump scalar %.5f)",
        FIT_SEASON, n_total, len(pitches), league_median_ump,
    )

    # ── 2. Deterministic 10K-PA subsample (seed 42) ─────────────────────
    rng = np.random.default_rng(args.seed)
    n_sample = min(args.n_pa, n_total)
    sample_idx = rng.choice(n_total, size=n_sample, replace=False)
    df_sample = df_pas.iloc[sample_idx].reset_index(drop=True)
    df_sample = _coerce_ints(
        df_sample,
        ["balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
         "inning", "pitcher_id", "batter_id"],
    )
    logger.info("Sampled %d PA starts (seed=%d)", n_sample, args.seed)

    # ── 3. Empirical per-position target (real 2023 sequences) ──────────
    sampled_keys = df_sample[
        ["game_pk", "at_bat_number", "pitcher_id", "batter_id"]
    ].copy()
    empirical, emp_counts = _empirical_perpos_marginals(pitches, sampled_keys)
    logger.info("Empirical per-position marginals (rows=pos 0..%d):", HORIZON - 1)
    for p in range(HORIZON):
        logger.info(
            "  pos %d (n=%7d): %s", p, int(emp_counts[p].sum()),
            np.array2string(empirical[p], precision=4),
        )

    if args.dry_run:
        logger.info("--dry-run: stopping before GPU rolls.")
        return 0

    # ── 4. Roll-0: raw T-softmax (no class_cal, no pos-0) ───────────────
    predictor_raw = PGConcatHeadPredictor(
        disable_class_calibration=True,
        disable_pos0_calibration=True,
    )
    roll0_marg, roll0_counts = _roll_marginals(
        df_sample, predictor_raw, league_median_ump,
        args.n_samples, args.seed * 5_000_011, "roll-0 raw-T",
    )
    w1, w1_audit = _guarded_ratio(empirical, roll0_marg, roll0_counts)

    iterations: list[dict] = [{
        "iteration": 1,
        "roll": "roll-0 raw-T",
        "rollout_marginals": _marg_table(roll0_marg),
        "rollout_counts": _marg_table(roll0_counts),
        "W_after_update": _marg_table(w1),
        "ratio_audit": w1_audit,
    }]

    # ── 5. Roll-1: measure with W1 ──────────────────────────────────────
    w1_path = _write_temp_w(w1, "iter1", args.tmp_dir)
    predictor_w1 = PGConcatHeadPredictor(
        disable_class_calibration=True,
        disable_pos0_calibration=True,
        use_rollout_perpos_calibration=True,
        rollout_perpos_calibration_path=w1_path,
    )
    roll1_marg, roll1_counts = _roll_marginals(
        df_sample, predictor_w1, league_median_ump,
        args.n_samples, args.seed * 5_000_021, "roll-1 W1",
    )
    conv1, maxd1, _ = _convergence(roll1_marg, empirical)
    iterations[0]["measurement_marginals"] = _marg_table(roll1_marg)
    iterations[0]["max_abs_delta_pp"] = round(maxd1 * 100.0, 3)
    iterations[0]["converged"] = conv1
    logger.info(
        "Iteration 1 measurement: max |delta| = %.3fpp -> %s",
        maxd1 * 100.0, "CONVERGED" if conv1 else "not converged",
    )

    final_w = w1
    final_marg, final_counts = roll1_marg, roll1_counts
    n_iterations = 1
    converged = conv1

    if not conv1:
        # ── 6. Iteration 2 (the single §4 feedback update) ──────────────
        update, upd_audit = _guarded_ratio(empirical, roll1_marg, roll1_counts)
        w2 = np.clip(w1 * update, RATIO_FLOOR, RATIO_CAP)
        log_w = np.log(np.clip(w2, 1e-12, None))
        log_w = log_w - log_w.mean(axis=1, keepdims=True)
        w2 = np.exp(log_w)
        w2_path = _write_temp_w(w2, "iter2", args.tmp_dir)
        predictor_w2 = PGConcatHeadPredictor(
            disable_class_calibration=True,
            disable_pos0_calibration=True,
            use_rollout_perpos_calibration=True,
            rollout_perpos_calibration_path=w2_path,
        )
        roll2_marg, roll2_counts = _roll_marginals(
            df_sample, predictor_w2, league_median_ump,
            args.n_samples, args.seed * 5_000_031, "roll-2 W2",
        )
        conv2, maxd2, _ = _convergence(roll2_marg, empirical)
        iterations.append({
            "iteration": 2,
            "roll": "roll-2 W2",
            "W_after_update": _marg_table(w2),
            "ratio_audit": upd_audit,
            "measurement_marginals": _marg_table(roll2_marg),
            "rollout_counts": _marg_table(roll2_counts),
            "max_abs_delta_pp": round(maxd2 * 100.0, 3),
            "converged": conv2,
        })
        logger.info(
            "Iteration 2 measurement: max |delta| = %.3fpp -> %s",
            maxd2 * 100.0, "CONVERGED" if conv2 else "NOT CONVERGED — KILL",
        )
        final_w = w2
        final_marg, final_counts = roll2_marg, roll2_counts
        n_iterations = 2
        converged = conv2

    # ── 7. SHA post-check (no checkpoint mutation, ever) ────────────────
    assert _sha256_file(BACKBONE_PATH) == bb_sha_pre, "v2.pt SHA drifted!"
    assert _sha256_file(A1_PATH) == a1_sha_pre, "a1.pt SHA drifted!"
    logger.info("Checkpoint byte-identity preserved (v2.pt + a1.pt).")

    # ── 8. Persist ──────────────────────────────────────────────────────
    npz_payload = dict(
        W=final_w.astype(np.float64),
        fit_cohort_season=np.int64(FIT_SEASON),
        fit_seed=np.int64(args.seed),
        fit_n_pas=np.int64(n_sample),
        fit_n_samples=np.int64(args.n_samples),
        horizon=np.int64(HORIZON),
        n_iterations=np.int64(n_iterations),
        converged=np.bool_(converged),
        empirical_perpos=empirical.astype(np.float64),
        empirical_counts=emp_counts.astype(np.float64),
        rollout_raw_perpos=roll0_marg.astype(np.float64),
        rollout_obs_counts_raw=roll0_counts.astype(np.float64),
        rollout_final_perpos=final_marg.astype(np.float64),
        rollout_final_counts=final_counts.astype(np.float64),
        temperature=np.float64(predictor_raw.temperature),
        backbone_sha256=np.array([bb_sha_pre.encode()], dtype="S64"),
        a1_head_sha256=np.array([a1_sha_pre.encode()], dtype="S64"),
        fit_date=np.array([FIT_DATE.encode()], dtype="S10"),
        produced_by=np.array(
            [b"scripts/pitchgpt_fit_rollout_calibration.py"], dtype="S48",
        ),
    )
    if converged:
        np.savez(OUT_NPZ, **npz_payload)
        logger.info("Saved converged W -> %s", OUT_NPZ)
        artifact_path = str(OUT_NPZ)
        artifact_sha = _sha256_file(OUT_NPZ)
    else:
        # §6 kill signal: nothing ships to models/. Archive for the record.
        kill_path = OUT_DIR / "W_FAILED_FIT_quarantine.npz"
        np.savez(kill_path, **npz_payload)
        logger.error(
            "KILL SIGNAL (PHASE_0.6.2_PLAN.md §6): 2023 fit did not converge "
            "within 2 fixed-point iterations. NO artifact ships. Archived "
            "failed fit -> %s", kill_path,
        )
        artifact_path = str(kill_path)
        artifact_sha = _sha256_file(kill_path)

    audit = {
        "spec": "docs/pitchgpt_sim_engine/PHASE_0.6.2_PLAN.md §4 + §10.A7",
        "fit_date": FIT_DATE,
        "fit_cohort": {
            "season": FIT_SEASON,
            "pitcher_disjoint_from": f"{TRAIN_RANGE[0]}-{TRAIN_RANGE[1]}",
            "n_train_pitchers_excluded": len(train_pitchers),
            "n_pas_total_eligible": int(n_total),
            "n_pas_sampled": int(n_sample),
            "subsample_seed": int(args.seed),
            "n_samples_per_pa": int(args.n_samples),
            "horizon": HORIZON,
            "league_median_ump_scalar": round(float(league_median_ump), 6),
        },
        "guards": {
            "min_obs_guard": MIN_OBS_GUARD,
            "ratio_floor_cap": [RATIO_FLOOR, RATIO_CAP],
            "convergence_pp": CONVERGENCE_PP,
        },
        "outcome_classes": list(OUTCOME_CLASSES),
        "empirical_perpos": _marg_table(empirical),
        "empirical_counts": _marg_table(emp_counts),
        "iterations": iterations,
        "n_iterations": n_iterations,
        "converged": converged,
        "final_W": _marg_table(final_w),
        "artifact": {
            "path": artifact_path,
            "sha256": artifact_sha,
            "shipped_to_models": bool(converged),
        },
        "checkpoint_integrity": {
            "backbone_sha256": bb_sha_pre,
            "a1_head_sha256": a1_sha_pre,
            "mutated": False,
        },
    }
    audit_path = OUT_DIR / "fit_audit.json"
    with audit_path.open("w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)
    logger.info("Wrote %s", audit_path)

    lines = [
        "# Phase 0.6.2 rollout-regime W fit — 2023 (audit report)\n",
        f"Fit date: {FIT_DATE}.  Spec: PHASE_0.6.2_PLAN.md §4 + §10.A7.\n",
        f"Cohort: {FIT_SEASON} pitcher-disjoint, {n_sample} PAs (seed "
        f"{args.seed}), {args.n_samples} samples/PA, horizon {HORIZON}.\n",
        f"**Iterations used: {n_iterations}.  Converged: "
        f"{'YES' if converged else 'NO — KILL SIGNAL (§6)'}.**\n",
        "| iteration | roll | max |delta| (pp) | converged |",
        "|---|---|---|---|",
    ]
    for it in iterations:
        lines.append(
            f"| {it['iteration']} | {it['roll']} | "
            f"{it.get('max_abs_delta_pp', 'n/a')} | "
            f"{it.get('converged', 'n/a')} |"
        )
    lines.append("")
    lines.append(f"Final artifact: `{artifact_path}` (sha256 `{artifact_sha}`)")
    lines.append("")
    lines.append("Per-position final W rows (classes: "
                 + ", ".join(OUTCOME_CLASSES) + "):")
    lines.append("")
    for p in range(HORIZON):
        lines.append(
            f"- pos {p}: "
            + np.array2string(final_w[p], precision=4, separator=", ")
        )
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", OUT_DIR / "report.md")

    return 0 if converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
