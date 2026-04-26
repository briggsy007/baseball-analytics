"""Hand-built fixture mirroring ``pitchgpt_sim.RolloutResult`` for A1.

The real ``RolloutResult`` lives in ``src/analytics/pitchgpt_sim.py``
(Phase 0.5; not yet implemented as of 2026-04-26). This module
intentionally does **NOT** import ``pitchgpt_sim`` — that lets the A1
dashboard view (`src/dashboard/views/pitch_call_grades.py`) render even
when Phase 0.5 has not yet landed.

Contract mirrored: ``SIM_ENGINE_API.md`` §3.3.

Fixture shape
-------------
- 5 sample plate appearances (PA contexts).
- Per-PA: 100 rollouts, horizon 6.
- ``outcome_predictor = "pg_concat_head"`` per Plan B winner
  (`SIM_ENGINE_API.md` §4.4).
- Calibration metadata reflects the locked A1 head:
  T = 0.8003, ECE_post = 0.0114 on 2025 holdout.

This fixture is illustrative — numbers are not drawn from the real model.
The view stub flags this prominently so screenshots aren't mistaken for
real grades.
"""
from __future__ import annotations

from collections import namedtuple
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Mirror dataclass shapes WITHOUT importing pitchgpt_sim
# ---------------------------------------------------------------------------

# These named tuples mirror the dataclasses from SIM_ENGINE_API.md §3.2 / §3.3.
# Using namedtuple (not dataclass) keeps the fixture module zero-dependency
# and avoids any chance of accidentally shadowing the real types.

PAContextFixture = namedtuple(
    "PAContextFixture",
    [
        "pitcher_id",
        "batter_id",
        "count",
        "outs",
        "runners",
        "batter_stand",
        "pitcher_throws",
        "inning",
        "inning_topbot",
        "score_diff",
        "umpire_scalar",
        "prefix_pitch_tokens",
    ],
)

RolloutResultFixture = namedtuple(
    "RolloutResultFixture",
    [
        "pitch_tokens",
        "pitch_probs",
        "outcomes",
        "outcome_probs",
        "pa_terminated",
        "pa_outcome",
        "final_count",
        "sampling_metadata",
    ],
)


# Outcome class indices per SIM_ENGINE_API.md §2 "7-class outcome target":
#   0 = ball, 1 = called_strike, 2 = swinging_strike, 3 = foul,
#   4 = in_play_out, 5 = in_play_hit, 6 = hbp
OUTCOME_LABELS = (
    "ball",
    "called_strike",
    "swinging_strike",
    "foul",
    "in_play_out",
    "in_play_hit",
    "hbp",
)

# wOBA-by-outcome (illustrative; real values live in WObaTable.default()
# per SIM_ENGINE_API.md §5.1).
OUTCOME_TO_WOBA_DEMO = {
    0: 0.69,   # ball   (BB ~0.69 on terminal walk; pre-terminal balls 0.0)
    1: 0.00,   # called strike
    2: 0.00,   # swinging strike
    3: 0.00,   # foul
    4: 0.00,   # in-play out
    5: 0.92,   # in-play hit (single-ish placeholder)
    6: 0.72,   # HBP
}

ROLLOUT_PAD_PITCH = 2210     # SIM_ENGINE_API §3.3
ROLLOUT_PAD_OUTCOME = 7      # SIM_ENGINE_API §3.3


def _make_pa_context(
    pitcher_id: int,
    batter_id: int,
    count_balls_strikes: tuple[int, int],
    inning: int,
    score_diff: int,
    inning_topbot: str = "Top",
) -> PAContextFixture:
    return PAContextFixture(
        pitcher_id=pitcher_id,
        batter_id=batter_id,
        count=count_balls_strikes,
        outs=1,
        runners=(False, False, False),
        batter_stand="R",
        pitcher_throws="R",
        inning=inning,
        inning_topbot=inning_topbot,
        score_diff=score_diff,
        umpire_scalar=0.0,
        prefix_pitch_tokens=(),
    )


# ---------------------------------------------------------------------------
# Fixture data — 5 PAs, illustrative
# ---------------------------------------------------------------------------

# Five hand-picked PA scenarios (pitcher_id, batter_id, scenario tag, real_outcome_idx, pitcher_name, batter_name)
SAMPLE_PAS: list[dict[str, Any]] = [
    {
        "pitcher_id": 605400,
        "batter_id": 547180,
        "pitcher_name": "Aaron Nola",
        "batter_name": "Bryce Harper",
        "scenario": "FF middle-middle in 2-1, 1 out, 0 runners",
        "real_outcome_idx": 5,   # in-play hit (the call was punished)
        "real_pitch_token": 1820,
        "context": _make_pa_context(605400, 547180, (2, 1), inning=4, score_diff=-1),
        # Demo grade percentile: actual outcome was in the 14th percentile of rollouts
        # (low = pitcher gave up MORE than expected -> bad call grade).
        "demo_percentile": 0.14,
        "grade_letter": "D",
    },
    {
        "pitcher_id": 605400,
        "batter_id": 660670,
        "pitcher_name": "Aaron Nola",
        "batter_name": "Ronald Acuna Jr.",
        "scenario": "CU low-away in 1-2, 2 out, 1 runner",
        "real_outcome_idx": 2,   # swinging strike (good call!)
        "real_pitch_token": 480,
        "context": _make_pa_context(605400, 660670, (1, 2), inning=2, score_diff=0),
        "demo_percentile": 0.92,
        "grade_letter": "A",
    },
    {
        "pitcher_id": 622491,
        "batter_id": 624413,
        "pitcher_name": "Zack Wheeler",
        "batter_name": "Pete Alonso",
        "scenario": "SI in-zone, 0-0, 0 out",
        "real_outcome_idx": 4,   # in-play out
        "real_pitch_token": 1120,
        "context": _make_pa_context(622491, 624413, (0, 0), inning=1, score_diff=0),
        "demo_percentile": 0.71,
        "grade_letter": "B",
    },
    {
        "pitcher_id": 622491,
        "batter_id": 666182,
        "pitcher_name": "Zack Wheeler",
        "batter_name": "Juan Soto",
        "scenario": "FF up-in 3-1, 2 out, no runners",
        "real_outcome_idx": 0,   # ball -> walk
        "real_pitch_token": 1700,
        "context": _make_pa_context(622491, 666182, (3, 1), inning=6, score_diff=2),
        "demo_percentile": 0.31,
        "grade_letter": "C-",
    },
    {
        "pitcher_id": 656232,
        "batter_id": 663656,
        "pitcher_name": "Cristopher Sanchez",
        "batter_name": "Kyle Tucker",
        "scenario": "CH down-away 2-2, 1 out",
        "real_outcome_idx": 3,   # foul (no count change at 2-2 ceiling)
        "real_pitch_token": 990,
        "context": _make_pa_context(656232, 663656, (2, 2), inning=7, score_diff=-2),
        "demo_percentile": 0.55,
        "grade_letter": "B-",
    },
]


def make_sampling_metadata(seed: int = 42) -> dict[str, Any]:
    """Return the canonical sampling_metadata block for the fixture.

    Mirrors ``SIM_ENGINE_API.md`` §3.4 required keys exactly.
    """
    return {
        "temperature": 1.0,
        "seed": seed,
        "n_samples": 100,
        "horizon": 6,
        "backbone_version": "v2",
        "backbone_checkpoint_sha256": (
            "6f952054d14ac6f918f3eb9502b496b70bc0c87dfc65dc50d98ee7244a62883c"
        ),
        # Plan B winner: PGConcatHeadPredictor backed by
        # models/pitchgpt_v2_outcomehead_a1.pt (SIM_ENGINE_API §4.4).
        "outcome_predictor": "pg_concat_head",
        "outcome_predictor_checkpoint_sha256": "<a1_head_sha_to_be_filled_phase_0.5>",
        "calibration_valid": True,
        "calibration_invalid_reasons": [],
        "rollout_engine_version": "0.5.0-fixture",
        "n_truncated": 4,   # 4 of 100 rollouts truncated at horizon=6 (illustrative)
        # NOTE: extra keys allowed per SIM_ENGINE_API §3.4 ("Implementations MAY add keys")
        "fixture_disclaimer": (
            "Hand-built illustrative fixture. Numbers are NOT from the real "
            "rollout engine. Replace with real RolloutResult from "
            "pitchgpt_sim.rollout() once Phase 0.5 lands."
        ),
    }


def build_rollout_fixture(seed: int = 42) -> RolloutResultFixture:
    """Construct one ``RolloutResultFixture`` with 100 samples × horizon 6.

    The numbers are reproducible via the seed. Outcome distribution is drawn
    from the empirical 7-class frequency prior on 2025 holdout (per
    `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/metrics.json`
    `cohort.train_freq_prior`):

        ball=0.365, called_strike=0.164, swinging_strike=0.109, foul=0.193,
        in_play_out=0.113, in_play_hit=0.054, hbp=0.003

    A few PAs in the fixture are pre-set to terminate early (walk / K / in-play)
    via ``pa_terminated`` so percentile aggregation has truncation to NaN-mask.
    """
    rng = np.random.default_rng(seed)
    n_samples = 100
    horizon = 6

    # Empirical freq-prior from a1_concat metrics.json
    freq_prior = np.array(
        [0.3650, 0.1638, 0.1087, 0.1931, 0.1126, 0.0540, 0.0027],
        dtype=np.float32,
    )

    # Sample pitch tokens uniformly from a small subset of the 2210-vocab so
    # the fixture stays tiny + readable; pad to ROLLOUT_PAD_PITCH past term.
    pitch_tokens = rng.integers(low=0, high=2210, size=(n_samples, horizon)).astype(np.int64)

    # Outcomes drawn IID from the freq prior (illustrative; real model is
    # state-conditional).
    outcomes = rng.choice(7, size=(n_samples, horizon), p=freq_prior).astype(np.int64)

    # PA termination: a sample terminates the first time outcome ∈ {4, 5, 6}
    # (in_play_out, in_play_hit, hbp) per SIM_ENGINE_API §3.3. Walk/K via
    # count tracking is approximated by a coin flip on (ball/strike) totals
    # — we just use the in-play termination here for fixture simplicity.
    pa_terminated = np.zeros((n_samples, horizon), dtype=bool)
    pa_outcome = np.full(n_samples, ROLLOUT_PAD_OUTCOME, dtype=np.int64)
    final_count = np.zeros((n_samples, 2), dtype=np.int64)
    n_truncated = 0
    for i in range(n_samples):
        terminated = False
        balls = 0
        strikes = 0
        for h in range(horizon):
            o = int(outcomes[i, h])
            if o == 0:
                balls += 1
            elif o in (1, 2):
                strikes = min(strikes + 1, 3)
            elif o == 3:
                if strikes < 2:
                    strikes += 1
            if balls >= 4 or strikes >= 3 or o in (4, 5, 6):
                pa_terminated[i, h] = True
                pa_outcome[i] = o if o in (4, 5, 6) else (0 if balls >= 4 else 2)
                final_count[i] = (min(balls, 4), min(strikes, 3))
                # Pad subsequent positions
                if h + 1 < horizon:
                    pitch_tokens[i, h + 1 :] = ROLLOUT_PAD_PITCH
                    outcomes[i, h + 1 :] = ROLLOUT_PAD_OUTCOME
                terminated = True
                break
        if not terminated:
            n_truncated += 1
            final_count[i] = (balls, strikes)
            # pa_outcome stays at PAD; pa_terminated stays all False

    # outcome_probs are calibrated 7-class probs per position. Fixture uses
    # frequency-prior smoothed by a small noise term, NaN-masked past
    # termination per SIM_ENGINE_API §3.3.
    outcome_probs = np.tile(freq_prior, (n_samples, horizon, 1)).astype(np.float32)
    noise = rng.normal(0.0, 0.02, size=outcome_probs.shape).astype(np.float32)
    outcome_probs = np.clip(outcome_probs + noise, 1e-4, None)
    outcome_probs = outcome_probs / outcome_probs.sum(axis=-1, keepdims=True)

    # NaN-mask past termination per SIM_ENGINE_API §3.3 ("rows past
    # termination are filled with NaN — silent zeros would propagate to
    # KL/Wasserstein aggregates").
    for i in range(n_samples):
        # Find termination position (first True in row, if any)
        term_idx = np.argmax(pa_terminated[i]) if pa_terminated[i].any() else horizon
        if term_idx + 1 < horizon and pa_terminated[i, term_idx]:
            outcome_probs[i, term_idx + 1 :] = np.nan

    metadata = make_sampling_metadata(seed=seed)
    metadata["n_truncated"] = n_truncated

    return RolloutResultFixture(
        pitch_tokens=pitch_tokens,
        pitch_probs=None,         # return_probs=False to keep fixture tiny
        outcomes=outcomes,
        outcome_probs=outcome_probs,
        pa_terminated=pa_terminated,
        pa_outcome=pa_outcome,
        final_count=final_count,
        sampling_metadata=metadata,
    )


def build_leaderboard_table(seed: int = 42) -> list[dict[str, Any]]:
    """Build a placeholder pitcher x week leaderboard table.

    Each row corresponds to one pitcher-week and reports the mean grade
    percentile + n_pitches + a fake "wOBA delta" attribution.

    Numbers are illustrative; real leaderboard build will live at
    ``results/pitchgpt_sim/pitch_call_grades_2025/leaderboard.csv``
    (see EXECUTION_PLAN §6 A1 Artifacts).
    """
    rng = np.random.default_rng(seed)
    pitchers = [
        ("Aaron Nola", 605400),
        ("Zack Wheeler", 622491),
        ("Cristopher Sanchez", 656232),
        ("Tarik Skubal", 669373),
        ("Garrett Crochet", 676979),
        ("Paul Skenes", 694973),
        ("Spencer Strider", 675911),
    ]
    weeks = ["2025-W14", "2025-W15", "2025-W16", "2025-W17"]
    rows: list[dict[str, Any]] = []
    for name, pid in pitchers:
        for wk in weeks:
            mean_pct = float(np.clip(rng.normal(0.50, 0.12), 0.05, 0.95))
            grade = _percentile_to_grade(mean_pct)
            rows.append(
                {
                    "pitcher_id": pid,
                    "pitcher_name": name,
                    "week": wk,
                    "n_pitches": int(rng.integers(80, 320)),
                    "mean_grade_percentile": round(mean_pct, 3),
                    "grade_letter": grade,
                    # Placeholder wOBA delta in milli-wOBA per pitch
                    "woba_delta_per_pitch_x1000": round(float(rng.normal(0.0, 4.5)), 2),
                }
            )
    return rows


def _percentile_to_grade(p: float) -> str:
    """Convert a percentile rank in [0, 1] to a coarse letter grade.

    Higher percentile = pitcher gave up LESS than expected = better grade.
    """
    if p >= 0.85:
        return "A"
    if p >= 0.70:
        return "B"
    if p >= 0.50:
        return "C"
    if p >= 0.30:
        return "D"
    return "F"


def schema_preview() -> dict[str, str]:
    """Return a human-readable schema preview for the view's fallback notice."""
    return {
        "RolloutResult.pitch_tokens": "(n_samples=100, horizon=6) int64",
        "RolloutResult.outcomes": "(n_samples=100, horizon=6) int64 [0..6]",
        "RolloutResult.outcome_probs": "(n_samples=100, horizon=6, 7) float32 [NaN past term]",
        "RolloutResult.pa_terminated": "(n_samples=100, horizon=6) bool",
        "RolloutResult.pa_outcome": "(n_samples=100,) int64 [7 = PAD past horizon]",
        "RolloutResult.final_count": "(n_samples=100, 2) int64 (balls, strikes)",
        "RolloutResult.sampling_metadata": "dict — see SIM_ENGINE_API §3.4",
    }
