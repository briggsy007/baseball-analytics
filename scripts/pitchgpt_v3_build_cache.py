"""
Build the PitchGPT v3 (WS5.2) sequence caches.

Spec: ``docs/pitchgpt_sim_engine/PITCHGPT_V2_SPEC.md`` §5.1 / §5.2 / §5.3.

Cohorts built (2025 and 2026 are structurally refused by
``pitchgpt_v3_data.assert_season_policy``):

* ``train``  — 2015-2022, 10,000 games (§3.2 games budget), all pitchers.
* ``fit23``  — 2023 pitcher-disjoint (every 2015-2022 pitcher excluded),
               FULL season — the model-selection + calibration slice (§5.2,
               "19,625 eligible PAs as measured by the 0.6.2 fit"; a game cap
               would shrink that cohort below the spec's definition).
* ``dev24``  — 2024 pitcher-disjoint, FULL season — the burned dev tier (§5.3).

Read-only DuckDB throughout (§8.4).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt_v3_data import (  # noqa: E402
    DEV_SEASON,
    FIT_SEASON,
    TRAIN_SEASONS,
    build_sequence_cache,
    fetch_train_pitcher_ids,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitchgpt_v3_build_cache")

DEFAULT_CACHE_DIR = Path(
    r"C:\Users\hunte\AppData\Local\Temp\claude"
    r"\C--Users-hunte-projects-baseball"
    r"\5112f9f4-f7ff-4db3-bd24-90acc5fbef27\scratchpad\pitchgpt_v3_cache"
)

SEED = 42
TRAIN_GAMES = 10_000
#: 0 == "no game cap".  §5.2 defines the fit slice as *the* 2023 pitcher-
#: disjoint cohort, not a sample of it, so the eval cohorts are full-season.
EVAL_GAMES = 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument(
        "--cohorts", nargs="*", default=["train", "fit23", "dev24"],
    )
    ap.add_argument("--train-games", type=int, default=TRAIN_GAMES)
    ap.add_argument("--eval-games", type=int, default=EVAL_GAMES)
    args = ap.parse_args()
    # 0 == "no game cap" (full season).  The 2023 fit slice and the 2024 dev
    # slice are pitcher-disjoint and therefore small; capping them buys
    # nothing and costs statistical power on the calibration fit and gates.
    if args.eval_games <= 0:
        args.eval_games = None
    if args.train_games <= 0:
        args.train_games = None

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    conn = get_connection(read_only=True)
    manifest: dict = {}

    excl: set[int] | None = None
    if {"fit23", "dev24"} & set(args.cohorts):
        t = time.perf_counter()
        excl = fetch_train_pitcher_ids(conn, TRAIN_SEASONS)
        logger.info(
            "pitcher-disjoint exclusion set: %d pitchers (%.1fs)",
            len(excl), time.perf_counter() - t,
        )

    specs = {
        "train": dict(
            seasons=list(TRAIN_SEASONS), max_games=args.train_games,
            exclude_pitcher_ids=None,
        ),
        "fit23": dict(
            seasons=[FIT_SEASON], max_games=args.eval_games,
            exclude_pitcher_ids=excl,
        ),
        "dev24": dict(
            seasons=[DEV_SEASON], max_games=args.eval_games,
            exclude_pitcher_ids=excl,
        ),
    }

    for name in args.cohorts:
        spec = specs[name]
        path = args.cache_dir / f"{name}.npz"
        if path.exists():
            logger.info("%s already cached at %s — skipping", name, path)
            continue
        logger.info("building cohort %s …", name)
        payload = build_sequence_cache(
            conn,
            seasons=spec["seasons"],
            max_games=spec["max_games"],
            sample_seed=SEED,
            exclude_pitcher_ids=spec["exclude_pitcher_ids"],
            cache_path=path,
        )
        manifest[name] = payload["meta"]
        logger.info("%s -> %s", name, json.dumps(payload["meta"]))

    if manifest:
        mp = args.cache_dir / "manifest.json"
        existing = json.loads(mp.read_text()) if mp.exists() else {}
        existing.update(manifest)
        mp.write_text(json.dumps(existing, indent=2))
        logger.info("manifest -> %s", mp)
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
