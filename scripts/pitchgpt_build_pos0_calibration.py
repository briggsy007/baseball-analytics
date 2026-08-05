"""
Build a position-0 class-marginal recalibration vector for the A1 outcome head.

Phase 0.6 fix (2026-04-26).  The static-context diagnostic
(`results/pitchgpt/rollout_sanity_2025/diagnose_static_context.md`) showed
that even with the existing JSON `class_calibration` applied, the A1 head
under-emits `ball` at PA position 0 by ~7.8pp (and over-emits
`swinging_strike` / `in_play_out`), driving the BB% deficit observed at
Phase 0.6 gate FAIL.

This script computes a SECOND-ORDER calibration vector

    class_calibration_pos0[c] = empirical_pos0_marginal[c] / a1_pos0_marginal[c]

where ``a1_pos0_marginal`` is the rollout's H=1 (single-step) outcome
distribution at position 0 averaged across PAs (= PA-start) on a 2025
pitcher-disjoint cohort, AFTER the existing JSON `class_calibration` is
applied (so this calibration acts on top of the existing one).

The vector is saved to ``models/calibration_class_marginal_pos0.npz`` and
loaded by ``PGConcatHeadPredictor`` at construction time.  It is applied
multiplicatively after the existing class_calibration in
``predict_outcome_probs`` (with renormalization).  This closes the
position-0 ball-rate miss (~4pp BB% recovery; per the Phase 0.6
diagnosis decomposition).

Mid-PA drift at later positions (which drives the K% surplus) is fixed
separately by the per-position context mutation now wired into
``rollout()`` -- see PHASE_0.5_PLAN §5.2 + the Phase 0.6 diagnostic.

Cohort
------
* 2025 pitcher-disjoint test holdout (same recipe as
  ``scripts/pitchgpt_rollout_sanity_2025.py``: pitchers seen in 2015-2022
  EXCLUDED).
* Subsample to 5K PAs (deterministic seed=42) -- per-class marginal
  estimation does not need the full cohort; 5K * 100 samples = 500K
  pos-0 pitch outcomes is plenty.
* Empirical marginal computed from `pitch_number=1` rows of the same
  cohort.

Outputs
-------
* ``models/calibration_class_marginal_pos0.npz`` containing:
    - ``class_calibration_pos0`` : (7,) float64
    - ``empirical_pos0_marginal`` : (7,) float64
    - ``a1_pos0_marginal`` : (7,) float64 (post-existing-calibration)
    - ``cohort_n_pas`` : int
    - ``cohort_season`` : 2025
    - ``fit_date`` : "2026-04-26"

Guardrails
----------
* DuckDB read_only=True throughout.
* No model checkpoint mutation (v2.pt + a1.pt SHA256 byte-identity
  asserted pre/post).
* No edits to ``src/analytics/pitchgpt.py`` or ``pitchgpt_outcome_head.py``.
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

from src.analytics.pitchgpt_sim import (  # noqa: E402
    PAContext,
    PGConcatHeadPredictor,
    rollout,
)
from src.analytics.pitchgpt_outcome_head import (  # noqa: E402
    NUM_OUTCOME_CLASSES,
    OUTCOME_CLASSES,
    OUTCOME_UNK,
    classify_pitch_outcome,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_pos0_calibration")


TRAIN_RANGE = (2015, 2022)
HOLDOUT_SEASON = 2025
N_PA_DEFAULT = 5_000
N_SAMPLES_DEFAULT = 100
SEED_DEFAULT = 42
FIT_DATE = "2026-04-26"

OUT_PATH = _ROOT / "models" / "calibration_class_marginal_pos0.npz"
BACKBONE_PATH = _ROOT / "models" / "pitchgpt_v2.pt"
A1_PATH = _ROOT / "models" / "pitchgpt_v2_outcomehead_a1.pt"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def _fetch_2025_pa_starts_and_first_pitches(
    conn, train_pitchers: set[int],
) -> pd.DataFrame:
    """Return one row per 2025 PA with start context and first-pitch outcome."""
    pids_str = ", ".join(str(int(p)) for p in train_pitchers)
    query = f"""
        WITH all_2025 AS (
            SELECT
                game_pk, at_bat_number, pitcher_id, batter_id, pitch_number,
                balls, strikes, outs_when_up, on_1b, on_2b, on_3b,
                stand, p_throws, inning, inning_topbot,
                events, description, woba_value
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
                MIN(pitch_number) AS first_pn
            FROM all_2025
            GROUP BY game_pk, at_bat_number, pitcher_id, batter_id
        )
        SELECT
            p.game_pk, p.at_bat_number, p.pitcher_id, p.batter_id,
            p.balls, p.strikes, p.outs_when_up,
            p.on_1b, p.on_2b, p.on_3b,
            p.stand, p.p_throws,
            p.inning, p.inning_topbot,
            p.description, p.events
        FROM all_2025 p
        JOIN pa_groups g
          ON p.game_pk = g.game_pk
         AND p.at_bat_number = g.at_bat_number
         AND p.pitcher_id = g.pitcher_id
         AND p.batter_id = g.batter_id
         AND p.pitch_number = g.first_pn
    """
    return conn.execute(query).fetchdf()


def _fetch_2025_league_median_ump_scalar(conn) -> float:
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
    if 2024 in by_season:
        return by_season[2024]
    if 2025 in by_season:
        return by_season[2025]
    return 0.0


def _coerce_ints(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(0)
                .astype(np.int64)
            )
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pa", type=int, default=N_PA_DEFAULT)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--db-path", type=str, default=None)
    args = parser.parse_args()

    bb_sha_pre = _sha256_file(BACKBONE_PATH)
    a1_sha_pre = _sha256_file(A1_PATH)
    logger.info("v2.pt SHA256 (pre): %s", bb_sha_pre)
    logger.info("a1.pt SHA256 (pre): %s", a1_sha_pre)

    # ── 1. Pull cohort ──────────────────────────────────────────────────
    conn = get_connection(args.db_path, read_only=True)
    try:
        logger.info("Fetching 2015-2022 train pitcher cohort ...")
        train_pitchers = _fetch_train_pitcher_ids(conn)
        logger.info("Train cohort pitchers: %d", len(train_pitchers))

        logger.info("Fetching 2025 PA-start + first-pitch rows ...")
        df_pas = _fetch_2025_pa_starts_and_first_pitches(conn, train_pitchers)
        logger.info("2025 pitcher-disjoint PA-start rows: %d", len(df_pas))

        league_median_ump = _fetch_2025_league_median_ump_scalar(conn)
    finally:
        conn.close()

    # Coerce nullable int / bool columns BEFORE PAContext.from_pitches_row.
    df_pas = _coerce_ints(
        df_pas,
        ["balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b",
         "inning", "pitcher_id", "batter_id"],
    )

    # ── 2. Subsample 5K PAs deterministically ───────────────────────────
    rng = np.random.default_rng(args.seed)
    n_total = len(df_pas)
    n_sample = min(args.n_pa, n_total)
    if n_sample < args.n_pa:
        logger.warning(
            "Only %d PAs available, requested %d; using all.",
            n_total, args.n_pa,
        )
    sample_idx = rng.choice(n_total, size=n_sample, replace=False)
    df_sample = df_pas.iloc[sample_idx].reset_index(drop=True)

    # ── 3. Empirical pos-0 marginal on the sampled cohort ───────────────
    descrs = df_sample["description"].tolist()
    events = df_sample["events"].tolist()
    classes = np.array(
        [classify_pitch_outcome(d, e) for d, e in zip(descrs, events)],
        dtype=np.int32,
    )
    valid_mask = (classes >= 0) & (classes < NUM_OUTCOME_CLASSES)
    n_valid_emp = int(valid_mask.sum())
    counts = np.bincount(
        classes[valid_mask], minlength=NUM_OUTCOME_CLASSES,
    ).astype(np.float64)
    empirical_pos0 = counts / max(counts.sum(), 1)
    logger.info(
        "Empirical pos-0 marginal (n=%d): %s",
        n_valid_emp,
        np.array2string(empirical_pos0, precision=4),
    )

    # ── 4. A1 pos-0 marginal via H=1 rollouts ───────────────────────────
    logger.info("Loading PGConcatHeadPredictor (A1, with current JSON calibration) ...")
    predictor = PGConcatHeadPredictor()
    a1_class_calib_loaded = predictor._class_calibration is not None
    logger.info(
        "Existing class_calibration loaded from JSON: %s",
        a1_class_calib_loaded,
    )

    logger.info(
        "Running H=1 rollouts (n_pa=%d  n_samples=%d) to estimate A1 pos-0 marginal ...",
        n_sample, args.n_samples,
    )
    t0 = time.perf_counter()
    a1_counts = np.zeros(NUM_OUTCOME_CLASSES, dtype=np.float64)
    n_outcomes_total = 0
    log_every = max(n_sample // 20, 100)

    for i, row in df_sample.iterrows():
        ctx = PAContext.from_pitches_row(row, ump_scalar=league_median_ump)
        per_row_seed = (args.seed * 3_000_017 + i) & 0x7FFFFFFF
        result = rollout(
            ctx,
            outcome_predictor=predictor,
            n_samples=args.n_samples,
            horizon=1,
            temperature=1.0,
            seed=per_row_seed,
            return_probs=False,
        )
        # outcomes shape: (n_samples, 1)
        if result.outcomes is None:
            continue
        col = result.outcomes[:, 0]
        cls_in_range = (col >= 0) & (col < NUM_OUTCOME_CLASSES)
        if not cls_in_range.any():
            continue
        c = col[cls_in_range].astype(np.int64)
        bc = np.bincount(c, minlength=NUM_OUTCOME_CLASSES).astype(np.float64)
        a1_counts += bc
        n_outcomes_total += int(cls_in_range.sum())

        if (i + 1) % log_every == 0:
            dt = time.perf_counter() - t0
            rate = (i + 1) / dt
            remain = (n_sample - (i + 1)) / max(rate, 1e-6)
            logger.info(
                "[a1_pos0] %d/%d PAs  rate=%.1f PA/s  eta=%.0fs",
                i + 1, n_sample, rate, remain,
            )
    dt_total = time.perf_counter() - t0
    logger.info("A1 pos-0 estimation complete in %.1fs", dt_total)

    a1_pos0 = a1_counts / max(a1_counts.sum(), 1)
    logger.info(
        "A1 pos-0 marginal (n=%d): %s",
        n_outcomes_total,
        np.array2string(a1_pos0, precision=4),
    )

    # ── 5. Compute calibration vector ───────────────────────────────────
    safe_a1 = np.where(a1_pos0 > 1e-12, a1_pos0, 1e-12)
    class_calibration_pos0 = empirical_pos0 / safe_a1
    # Normalize so geometric_mean(w) = 1 (purely cosmetic; renormalize-
    # after-multiply makes scale irrelevant).
    log_w = np.log(np.clip(class_calibration_pos0, 1e-12, None))
    log_w = log_w - log_w.mean()
    class_calibration_pos0 = np.exp(log_w)

    logger.info("--- pos-0 marginals + class_calibration ---")
    logger.info("%-20s %10s %10s %10s %10s",
                "class", "empirical", "a1", "delta_pp", "calib")
    for c, name in enumerate(OUTCOME_CLASSES):
        d_pp = (a1_pos0[c] - empirical_pos0[c]) * 100.0
        logger.info(
            "%-20s %10.4f %10.4f  %+9.2fpp %10.4f",
            name, empirical_pos0[c], a1_pos0[c], d_pp,
            class_calibration_pos0[c],
        )

    # ── 6. Save npz ─────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_PATH,
        class_calibration_pos0=class_calibration_pos0.astype(np.float64),
        empirical_pos0_marginal=empirical_pos0.astype(np.float64),
        a1_pos0_marginal=a1_pos0.astype(np.float64),
        cohort_n_pas=np.int64(n_sample),
        cohort_season=np.int64(HOLDOUT_SEASON),
        n_emp_pitches=np.int64(n_valid_emp),
        n_a1_outcomes=np.int64(n_outcomes_total),
        fit_date=np.array([FIT_DATE.encode()], dtype="S10"),
    )
    logger.info("Saved %s", OUT_PATH)

    # ── 7. SHA256-verify both checkpoints (post). ──────────────────────
    bb_sha_post = _sha256_file(BACKBONE_PATH)
    a1_sha_post = _sha256_file(A1_PATH)
    assert bb_sha_pre == bb_sha_post, (
        f"v2.pt SHA drifted! pre={bb_sha_pre} post={bb_sha_post}"
    )
    assert a1_sha_pre == a1_sha_post, (
        f"a1.pt SHA drifted! pre={a1_sha_pre} post={a1_sha_post}"
    )
    logger.info("Checkpoint byte-identity preserved (v2.pt + a1.pt).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
