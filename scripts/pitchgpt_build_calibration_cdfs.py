"""
Build calibration-feature CDFs for the PitchGPT v2 sim engine.

Computes empirical CDFs over the 2025 pitcher-disjoint holdout cohort for the
five features used by `_check_calibration_valid` (SIM_ENGINE_API §6, condition
4 / context out-of-band gate):

    * `count_state`   (12 buckets, 0..11)  -- balls*3 + min(strikes,2)
    * `outs`          ( 3 buckets, 0..2 )
    * `score_diff_bucket` (5 buckets, 0..4) -- per `encode_context`
    * `inning_bucket` ( 4 buckets, 0..3 )  -- per `encode_context`
    * `runner_state`  ( 8 buckets, 0..7 )  -- 4*on_1b + 2*on_2b + on_3b

For each feature we save the empirical CDF (cumulative frequency, normalized to
[0, 1]) -- this lets the calibration validity gate cheaply test whether a
starting context's value lies within the [1st, 99th] percentile band of the
calibration cohort.  For `runner_state` (the categorical case), we additionally
emit an occurrence mask used by the `"context_runner_state_unseen"` reason.

The 2025 pitcher-disjoint holdout cohort matches `pitchgpt_2025_holdout.py`:
2015-2022 train cohort's pitcher set is excluded from the 2025 query so no
training pitcher leaks into the calibration cohort.

Output
------
``models/calibration_feature_cdfs.npz`` with keys:
    count_state_cdf          float32 (12,)
    outs_cdf                 float32 (3,)
    score_diff_bucket_cdf    float32 (5,)
    inning_bucket_cdf        float32 (4,)
    runner_state_mask        bool    (8,)   -- True iff value present
    holdout_season           int             -- 2025
    holdout_n_pitches        int             -- count of rows aggregated
    fit_date                 str             -- ISO-8601 build date

Read-only DuckDB use only.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.db.schema import get_connection  # noqa: E402
from src.analytics.pitchgpt import _compute_per_pitch_score_diff  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitchgpt_build_calibration_cdfs")


TRAIN_RANGE = (2015, 2022)
HOLDOUT_SEASON = 2025  # holdout-contact: historical contacts registered in docs/holdout_ledger.jsonl; any NEW run must append a new contact via src.holdout (@holdout_access / record_contact)

DEFAULT_OUTPUT = _ROOT / "models" / "calibration_feature_cdfs.npz"


def _bucket_score_diff(score_diff: int) -> int:
    """Replica of `PitchTokenizer.encode_context`'s score_diff bucketing."""
    if score_diff <= -4:
        return 0
    if score_diff < 0:
        return 1
    if score_diff == 0:
        return 2
    if score_diff <= 3:
        return 3
    return 4


def _bucket_inning(inning: int) -> int:
    if inning <= 3:
        return 0
    if inning <= 6:
        return 1
    if inning <= 9:
        return 2
    return 3


def fetch_train_pitcher_ids(conn) -> set[int]:
    """Pitchers appearing in 2015-2022 (the train cohort)."""
    s_str = ", ".join(str(y) for y in range(TRAIN_RANGE[0], TRAIN_RANGE[1] + 1))
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


def load_holdout_features(conn, train_pitcher_ids: set[int]) -> dict:
    """Return dict of arrays for the 5 features over the holdout cohort.

    Pitcher-disjoint: pitchers appearing in 2015-2022 are excluded from 2025.
    """
    logger.info(
        "Excluding %d train-cohort pitchers from %d-only holdout query",
        len(train_pitcher_ids),
        HOLDOUT_SEASON,
    )

    if train_pitcher_ids:
        # Use a temp table for the exclusion list; DuckDB IN-list on a million
        # elements is tractable but a temp join is cheaper.
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS _train_pitchers (pitcher_id BIGINT)")
        conn.execute("DELETE FROM _train_pitchers")
        if train_pitcher_ids:
            conn.executemany(
                "INSERT INTO _train_pitchers VALUES (?)",
                [(int(p),) for p in train_pitcher_ids],
            )
        join_clause = (
            "AND p.pitcher_id NOT IN (SELECT pitcher_id FROM _train_pitchers)"
        )
    else:
        join_clause = ""

    query = f"""
        SELECT
            p.game_pk                    AS game_pk,
            p.at_bat_number              AS at_bat_number,
            p.pitch_number               AS pitch_number,
            COALESCE(p.balls, 0)         AS balls,
            COALESCE(p.strikes, 0)       AS strikes,
            COALESCE(p.outs_when_up, 0)  AS outs,
            COALESCE(p.on_1b, 0)         AS on_1b,
            COALESCE(p.on_2b, 0)         AS on_2b,
            COALESCE(p.on_3b, 0)         AS on_3b,
            COALESCE(p.inning, 1)        AS inning,
            COALESCE(p.inning_topbot, 'Top') AS inning_topbot,
            p.events                     AS events,
            p.delta_run_exp              AS delta_run_exp
        FROM pitches p
        WHERE p.pitch_type IS NOT NULL
          AND p.pitcher_id IS NOT NULL
          AND EXTRACT(YEAR FROM p.game_date) = {HOLDOUT_SEASON}
          {join_clause}
        ORDER BY p.game_pk,
                 COALESCE(p.at_bat_number, 0),
                 COALESCE(p.pitch_number, 0)
    """
    t0 = time.perf_counter()
    df = conn.execute(query).fetchdf()
    dt = time.perf_counter() - t0
    logger.info("Fetched %d holdout rows in %.1fs", len(df), dt)

    if len(df) == 0:
        raise RuntimeError(
            f"No {HOLDOUT_SEASON} pitcher-disjoint holdout rows returned -- "
            "verify the database has 2025 pitches and the train cohort is "
            "present.",
        )

    balls = np.minimum(df["balls"].to_numpy(dtype=np.int32), 3)
    strikes = np.minimum(df["strikes"].to_numpy(dtype=np.int32), 2)
    outs = np.minimum(df["outs"].to_numpy(dtype=np.int32), 2)
    on_1b = (df["on_1b"].to_numpy(dtype=np.int32) > 0).astype(np.int32)
    on_2b = (df["on_2b"].to_numpy(dtype=np.int32) > 0).astype(np.int32)
    on_3b = (df["on_3b"].to_numpy(dtype=np.int32) > 0).astype(np.int32)
    inning = df["inning"].to_numpy(dtype=np.int32)

    # Score diff (pitcher POV) -- reconstructed via the canonical helper that
    # walks each game in pitch order and accumulates runs from `events` /
    # `delta_run_exp`.  See `_compute_per_pitch_score_diff` in pitchgpt.py.
    logger.info("Reconstructing per-pitch score_diff over %d rows...", len(df))
    t1 = time.perf_counter()
    score_diff = np.array(_compute_per_pitch_score_diff(df), dtype=np.int32)
    logger.info("score_diff reconstruction: %.1fs", time.perf_counter() - t1)

    count_state = balls * 3 + strikes
    runner_state = on_1b * 4 + on_2b * 2 + on_3b

    score_diff_bucket = np.empty_like(score_diff, dtype=np.int32)
    score_diff_bucket[:] = 2  # default (== 0 case)
    score_diff_bucket[score_diff <= -4] = 0
    score_diff_bucket[(score_diff > -4) & (score_diff < 0)] = 1
    score_diff_bucket[score_diff == 0] = 2
    score_diff_bucket[(score_diff > 0) & (score_diff <= 3)] = 3
    score_diff_bucket[score_diff > 3] = 4

    inning_bucket = np.empty_like(inning, dtype=np.int32)
    inning_bucket[inning <= 3] = 0
    inning_bucket[(inning > 3) & (inning <= 6)] = 1
    inning_bucket[(inning > 6) & (inning <= 9)] = 2
    inning_bucket[inning > 9] = 3

    return {
        "count_state": count_state.astype(np.int32),
        "outs": outs.astype(np.int32),
        "score_diff_bucket": score_diff_bucket.astype(np.int32),
        "inning_bucket": inning_bucket.astype(np.int32),
        "runner_state": runner_state.astype(np.int32),
        "n_rows": int(len(df)),
    }


def build_cdf(values: np.ndarray, n_categories: int) -> np.ndarray:
    """Cumulative empirical CDF over [0, n_categories-1].

    Returns a (n_categories,) float32 array where entry i = P(X <= i) on the
    holdout cohort.  The final entry is 1.0 (within float epsilon).
    """
    counts = np.bincount(values, minlength=n_categories).astype(np.float64)
    if values.size != int(counts.sum()):
        raise RuntimeError(
            f"Histogram count mismatch: values.size={values.size} "
            f"vs counts.sum()={int(counts.sum())} -- check that all "
            f"`values` lie in [0, {n_categories-1}]",
        )
    pmf = counts / max(counts.sum(), 1.0)
    cdf = np.cumsum(pmf)
    return cdf.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="DuckDB path (defaults to project canonical location).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output .npz path (default models/calibration_feature_cdfs.npz).",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection(args.db_path, read_only=True)
    try:
        train_pitcher_ids = fetch_train_pitcher_ids(conn)
        feats = load_holdout_features(conn, train_pitcher_ids)
    finally:
        conn.close()

    cdfs = {
        "count_state_cdf": build_cdf(feats["count_state"], 12),
        "outs_cdf": build_cdf(feats["outs"], 3),
        "score_diff_bucket_cdf": build_cdf(feats["score_diff_bucket"], 5),
        "inning_bucket_cdf": build_cdf(feats["inning_bucket"], 4),
    }
    runner_mask = (np.bincount(feats["runner_state"], minlength=8) > 0).astype(np.bool_)

    payload = {
        **cdfs,
        "runner_state_mask": runner_mask,
        "holdout_season": np.int32(HOLDOUT_SEASON),
        "holdout_n_pitches": np.int64(feats["n_rows"]),
        "fit_date": np.str_(date.today().isoformat()),
    }

    logger.info(
        "CDF lengths: count_state=%d outs=%d score_diff_bucket=%d "
        "inning_bucket=%d runner_state_mask=%d (n_rows=%d)",
        cdfs["count_state_cdf"].shape[0],
        cdfs["outs_cdf"].shape[0],
        cdfs["score_diff_bucket_cdf"].shape[0],
        cdfs["inning_bucket_cdf"].shape[0],
        runner_mask.shape[0],
        feats["n_rows"],
    )
    logger.info("count_state CDF tail: %s", cdfs["count_state_cdf"][-3:])
    logger.info("runner_state mask: %s", runner_mask.tolist())

    np.savez_compressed(args.output, **payload)
    sz = args.output.stat().st_size
    logger.info("Wrote %s (%d bytes)", args.output, sz)

    # Print a tiny diagnostic summary in JSON (handy for the build report).
    print(json.dumps({
        "output": str(args.output),
        "size_bytes": sz,
        "holdout_season": HOLDOUT_SEASON,
        "holdout_n_pitches": feats["n_rows"],
        "cdf_lengths": {
            "count_state": int(cdfs["count_state_cdf"].shape[0]),
            "outs": int(cdfs["outs_cdf"].shape[0]),
            "score_diff_bucket": int(cdfs["score_diff_bucket_cdf"].shape[0]),
            "inning_bucket": int(cdfs["inning_bucket_cdf"].shape[0]),
            "runner_state_mask": int(runner_mask.shape[0]),
        },
        "runner_state_seen": [int(x) for x in np.where(runner_mask)[0].tolist()],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
