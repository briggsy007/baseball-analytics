"""
Sanity check: load a sample of the PitchGPT dataset and print the
per-pitch score_diff bucket distribution.

The bug this covers: prior to ticket #2, ``score_diff`` was hard-coded
to ``0`` in ``PitchSequenceDataset._load()`` and in the inference path,
so bucket 2 ("tie") contained 100% of the mass.  After the fix, all
five buckets should be non-empty on a full-season sample; if any
bucket is empty, the reconstruction in ``_compute_per_pitch_score_diff``
is broken.

Usage:
    python scripts/verify_score_diff_distribution.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

# Ensure project root is on sys.path for direct script execution.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics.pitchgpt import (  # noqa: E402
    NUM_SCORE_DIFF_BUCKETS,
    PitchSequenceDataset,
    PitchTokenizer,
    NUM_COUNT_STATES,
    NUM_OUTS,
    NUM_RUNNER_STATES,
    NUM_BATTER_HANDS,
    NUM_INNING_BUCKETS,
)
from src.db.schema import get_connection  # noqa: E402


_BUCKET_LABELS = {
    0: "big deficit (<= -4)",
    1: "small deficit (-3..-1)",
    2: "tie (0)",
    3: "small lead (+1..+3)",
    4: "big lead (>= +4)",
}


def _score_bucket_offset() -> int:
    """Index within the 34-dim context tensor where the 5 score buckets start."""
    return (
        NUM_COUNT_STATES
        + NUM_OUTS
        + NUM_RUNNER_STATES
        + NUM_BATTER_HANDS
        + NUM_INNING_BUCKETS
    )


def main(seasons: list[int] | None = None, max_games: int = 300) -> int:
    if seasons is None:
        seasons = [2023]

    print(
        f"Loading PitchGPT dataset: seasons={seasons}, max_games={max_games} ..."
    )
    conn = get_connection(read_only=True)
    ds = PitchSequenceDataset(conn, seasons=seasons, max_seq_len=256, max_games=max_games)
    conn.close()

    if len(ds) == 0:
        print("ERROR: no sequences loaded — check that the DB has pitches for these seasons.")
        return 1

    offset = _score_bucket_offset()
    counts: Counter[int] = Counter()

    for tokens, ctx, _target in ds.sequences:
        # ctx is (S, 34); the score-diff one-hot sits at cols [offset : offset+5]
        block = ctx[:, offset : offset + NUM_SCORE_DIFF_BUCKETS]
        # argmax over the 5 buckets gives the bucket id per pitch
        ids = block.argmax(dim=-1).tolist()
        counts.update(ids)

    total = sum(counts.values())
    print(f"\nTotal pitches analysed: {total}")
    print("Score-diff bucket distribution:")
    missing: list[int] = []
    for b in range(NUM_SCORE_DIFF_BUCKETS):
        n = counts.get(b, 0)
        pct = 100.0 * n / total if total else 0.0
        print(f"  bucket {b} [{_BUCKET_LABELS[b]:<28}] {n:>8}  ({pct:5.2f}%)")
        if n == 0:
            missing.append(b)

    if missing:
        print(f"\nFAIL: buckets {missing} are empty — check the reconstruction.")
        return 2

    # Additional invariant: bucket 2 (tie) should NOT hold 100% of the mass
    # (the whole point of the ticket).
    if counts.get(2, 0) == total:
        print("\nFAIL: all pitches still map to bucket 2 (tie). score_diff still hard-coded?")
        return 3

    print("\nOK: all 5 buckets are populated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
