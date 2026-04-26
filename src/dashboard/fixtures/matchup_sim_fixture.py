"""Hand-built fixture for the A3 batter x pitcher matchup sim view.

Mirrors a (batter, pitcher) -> wOBA-distribution histogram emitted by
``pitchgpt_matchup`` (Phase 1 module not yet built; see EXECUTION_PLAN
§6 A3). Does NOT import ``pitchgpt_sim`` so the matchup view renders
cleanly even when Phase 0.5 has not landed.

Numbers are illustrative; replace the histogram-builder with a real
``pitchgpt_sim.rollout()`` aggregation once the engine ships.
"""
from __future__ import annotations

from collections import namedtuple
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Mirror dataclass (no pitchgpt_sim import)
# ---------------------------------------------------------------------------

MatchupSimFixture = namedtuple(
    "MatchupSimFixture",
    [
        "batter_id",
        "batter_name",
        "pitcher_id",
        "pitcher_name",
        "n_real_pas",          # actual 2025 head-to-head PA count
        "n_simulated_pas",     # rollout count (e.g., 10_000)
        "woba_samples",        # (n_simulated_pas,) float32 array
        "percentiles",         # dict {p05, p25, p50, p75, p95}
        "small_sample_flag",   # True if n_real_pas < 20 (rollout-only)
        "in_play_hit_share",   # fraction of rollouts that ended in 'in_play_hit'
        "sampling_metadata",   # mirrors SIM_ENGINE_API §3.4
    ],
)


# Fixture matchups — five canonical pairs spanning the small-sample rule
# bands per EXECUTION_PLAN A3 Gate 3.
SAMPLE_MATCHUPS: list[dict[str, Any]] = [
    # Big-sample (rare in single-season MLB; in 2025 alone the max real-PA
    # batter x pitcher pair has 13 PAs — see scripts/a3_matchup_cohort.py
    # manifest "schema_mismatch_note"). Multi-season aggregation is needed
    # for true ge50 cells; this fixture pre-emptively shows the band.
    {
        "batter_name": "Trea Turner",
        "batter_id": 607208,
        "pitcher_name": "Spencer Strider",
        "pitcher_id": 675911,
        "n_real_pas": 8,            # realistic for 2025 alone
        "small_sample_flag": True,  # < 20 -> rollout only
        "mean_woba": 0.310,
        "std_woba": 0.18,
    },
    {
        "batter_name": "Bryce Harper",
        "batter_id": 547180,
        "pitcher_name": "Paul Skenes",
        "pitcher_id": 694973,
        "n_real_pas": 4,
        "small_sample_flag": True,
        "mean_woba": 0.355,
        "std_woba": 0.21,
    },
    {
        "batter_name": "Aaron Judge",
        "batter_id": 592450,
        "pitcher_name": "Garrett Crochet",
        "pitcher_id": 676979,
        "n_real_pas": 6,
        "small_sample_flag": True,
        "mean_woba": 0.385,
        "std_woba": 0.22,
    },
    # An illustrative "ge20" pair — synthetic count to show the
    # not-flagged path.
    {
        "batter_name": "Pete Alonso",
        "batter_id": 624413,
        "pitcher_name": "Zack Wheeler",
        "pitcher_id": 622491,
        "n_real_pas": 22,
        "small_sample_flag": False,  # >= 20 PAs -> empirical cross-check valid
        "mean_woba": 0.290,
        "std_woba": 0.16,
    },
    {
        "batter_name": "Juan Soto",
        "batter_id": 666182,
        "pitcher_name": "Aaron Nola",
        "pitcher_id": 605400,
        "n_real_pas": 31,
        "small_sample_flag": False,
        "mean_woba": 0.345,
        "std_woba": 0.19,
    },
]


def make_sampling_metadata(seed: int = 99) -> dict[str, Any]:
    """Mirror SIM_ENGINE_API §3.4 required keys for the matchup fixture."""
    return {
        "temperature": 1.0,
        "seed": seed,
        "n_samples": 10_000,
        "horizon": 6,
        "backbone_version": "v2",
        "backbone_checkpoint_sha256": (
            "6f952054d14ac6f918f3eb9502b496b70bc0c87dfc65dc50d98ee7244a62883c"
        ),
        "outcome_predictor": "pg_concat_head",
        "outcome_predictor_checkpoint_sha256": "<a1_head_sha_to_be_filled_phase_0.5>",
        "calibration_valid": True,
        "calibration_invalid_reasons": [],
        "rollout_engine_version": "0.5.0-fixture",
        "n_truncated": 412,   # ~4% truncated at horizon=6 (illustrative)
        "fixture_disclaimer": (
            "Hand-built illustrative fixture. Numbers are NOT from the "
            "real rollout engine. Replace with real RolloutResult aggregation "
            "once Phase 0.5 + pitchgpt_matchup land."
        ),
    }


def build_matchup_fixture(matchup_idx: int = 0, seed: int = 99) -> MatchupSimFixture:
    """Build one ``MatchupSimFixture`` with a synthetic wOBA histogram.

    The wOBA distribution is an illustrative heavy-right-tailed mixture
    (mostly zeros for outs/Ks/walks-without-RBI, with occasional 0.69 (BB),
    0.89 (1B), 1.27 (2B), 1.62 (3B), 2.10 (HR) hits). Real rollouts will
    produce a richer distribution; this is only a screen.
    """
    spec = SAMPLE_MATCHUPS[matchup_idx]
    rng = np.random.default_rng(seed + matchup_idx * 17)
    n = 10_000

    # Mix: 60% zeros, 20% near-mean, 15% small-positive, 5% big tail
    samples = np.zeros(n, dtype=np.float32)
    near = rng.normal(spec["mean_woba"], spec["std_woba"], size=n).astype(np.float32)
    near = np.clip(near, 0.0, 2.5)
    mask_zero = rng.random(n) < 0.55
    mask_tail = rng.random(n) < 0.05
    samples = np.where(mask_zero, 0.0, near)
    # Inject the heavy-right tail
    tail_vals = rng.choice(
        np.array([1.27, 1.62, 2.10], dtype=np.float32),
        size=n,
        p=[0.6, 0.25, 0.15],
    )
    samples = np.where(mask_tail, tail_vals, samples)

    p05, p25, p50, p75, p95 = np.percentile(samples, [5, 25, 50, 75, 95]).astype(float)
    in_play_hit_share = float(np.mean((samples >= 0.85) & (samples <= 2.30)))

    return MatchupSimFixture(
        batter_id=spec["batter_id"],
        batter_name=spec["batter_name"],
        pitcher_id=spec["pitcher_id"],
        pitcher_name=spec["pitcher_name"],
        n_real_pas=spec["n_real_pas"],
        n_simulated_pas=n,
        woba_samples=samples,
        percentiles={"p05": p05, "p25": p25, "p50": p50, "p75": p75, "p95": p95},
        small_sample_flag=spec["small_sample_flag"],
        in_play_hit_share=in_play_hit_share,
        sampling_metadata=make_sampling_metadata(seed=seed + matchup_idx),
    )


def list_matchup_options() -> list[tuple[str, int]]:
    """Return ``[(label, idx), ...]`` for the matchup-selector dropdown."""
    return [
        (f"{m['batter_name']} vs {m['pitcher_name']} (real PAs={m['n_real_pas']})", i)
        for i, m in enumerate(SAMPLE_MATCHUPS)
    ]


def schema_preview() -> dict[str, str]:
    return {
        "MatchupSim.woba_samples": "(n_simulated_pas=10000,) float32",
        "MatchupSim.percentiles": "{p05, p25, p50, p75, p95} dict[str, float]",
        "MatchupSim.n_real_pas": "int — real head-to-head PA count in 2025",
        "MatchupSim.small_sample_flag": "bool — True if n_real_pas < 20",
        "MatchupSim.in_play_hit_share": "float — fraction of rollouts ending in in-play hit",
        "MatchupSim.sampling_metadata": "dict — see SIM_ENGINE_API §3.4",
    }
