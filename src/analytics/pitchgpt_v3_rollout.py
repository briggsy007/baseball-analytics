"""
PitchGPT v3 — PA rollout on the factorized backbone.

Spec: ``PITCHGPT_V2_SPEC.md`` §3.6 (per-field sampling masks), §4.1 (dynamic
mid-PA context, commit ``6111cd6``), §4.3.1 (horizon 6, existing termination
logic), §4.3.2 (seeds).

One state machine, three doors
------------------------------
:func:`rollout_pa_batch` is the **only** implementation of the v3 PA rollout.
:func:`rollout_v3` (single PA, ``RolloutResult`` contract) and
:func:`rollout_v3_optin` (the sim door) both call it, so "the training-time
rollout is the inference-time rollout" (§4.3) holds structurally rather than by
inspection.  The PA-termination logic, the ``count_state`` mutation and the
``RolloutResult`` contract are the production engine's own — they are imported
from :mod:`src.analytics.pitchgpt_sim`, never re-derived.

Production ``pitchgpt_sim.rollout()`` is **untouched** by this module: nothing
here is registered with ``OutcomePredictorRegistry`` and no production import
reaches v3.  Reaching v3 from a sim-shaped call site requires
:func:`rollout_v3_optin`, which refuses unless the caller passes
``enable_v3=True`` *or* sets ``PITCHGPT_V3_SIM_OPTIN=1`` (§8.1: no alias change;
``production`` stays pinned to ``v2026.04.23``).

Randomness (§4.3.2)
-------------------
Per-PA rollout seed ``42 + pa_index * 1000``.  Sampling is inverse-CDF from a
pre-drawn ``(n_samples, horizon, 4)`` uniform block produced by
``numpy.random.default_rng(pa_seed(pa_index))`` — one block per PA — so a PA's
sampled trajectory is **identical whether it is rolled alone or inside a batch
of any size**.  That invariance is asserted by
``tests/test_pitchgpt_v3.py::test_batched_and_single_pa_rollouts_agree``.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from src.analytics.pitchgpt import (
    BOS_TOKEN,
    CONTEXT_DIM,
    NUM_COUNT_STATES,
    PAD_TOKEN,
    PitchTokenizer,
)
from src.analytics.pitchgpt_outcome_head import NUM_OUTCOME_CLASSES
from src.analytics.pitchgpt_sim import (
    PAContext,
    RolloutResult,
    ROLLOUT_PAD_OUTCOME,
    ROLLOUT_PAD_PITCH,
    _TERMINAL_INPLAY_OUTCOMES,
    _advance_count,
)
from src.analytics.pitchgpt_v3 import (
    FactorizedPitchGPT,
    HeadTemperatures,
    MaskStats,
    V3OutcomeHead,
)

logger = logging.getLogger(__name__)

HORIZON = 6
#: §4.3.2 — per-PA rollout seed convention inherited from PHASE_0.6_PLAN §4.4.
SEED_STRIDE = 1000
SEED_BASE = 42
#: Number of uniforms consumed per (sample, position): type, zone, velo, outcome.
N_UNIFORM_FIELDS = 4

#: Terminal outcome classes as a dense boolean lookup (order = OUTCOME_CLASSES).
_TERMINAL_MASK = np.array(
    [c in _TERMINAL_INPLAY_OUTCOMES for c in range(NUM_OUTCOME_CLASSES)],
    dtype=bool,
)
#: Vectorised ``_advance_count``: ``_ADV[outcome] = (d_balls, d_strikes, foul)``.
#: ``foul`` marks the class whose strike increment is capped at 2 strikes.
_ADV_DBALLS = np.zeros(NUM_OUTCOME_CLASSES, dtype=np.int64)
_ADV_DSTRIKES = np.zeros(NUM_OUTCOME_CLASSES, dtype=np.int64)
_ADV_IS_FOUL = np.zeros(NUM_OUTCOME_CLASSES, dtype=bool)
for _c in range(NUM_OUTCOME_CLASSES):
    _b, _s = _advance_count(0, 0, _c)
    _ADV_DBALLS[_c], _ADV_DSTRIKES[_c] = _b, _s
    # A foul is the only class whose strike increment depends on the count.
    _ADV_IS_FOUL[_c] = _advance_count(0, 2, _c) == (0, 2) and _s == 1


def pa_seed(pa_index: int, base: int = SEED_BASE) -> int:
    """``42 + pa_index * 1000`` (§4.3.2)."""
    return (base + int(pa_index) * SEED_STRIDE) & 0x7FFFFFFF


def pa_uniform_block(
    pa_index: int, n_samples: int, horizon: int = HORIZON, base: int = SEED_BASE
) -> np.ndarray:
    """The ``(n_samples, horizon, 4)`` uniform block owned by one PA (§4.3.2)."""
    rng = np.random.default_rng(pa_seed(pa_index, base))
    return rng.random((n_samples, horizon, N_UNIFORM_FIELDS))


def pa_count_trajectory(
    outcomes: np.ndarray, start: tuple[int, int] = (0, 0)
) -> list[tuple[int, int]]:
    """Pre-pitch ``(balls, strikes)`` at each within-PA position.

    Driven by :func:`src.analytics.pitchgpt_sim._advance_count` — the same
    state machine the production rollout uses — so a Stage-B training context
    built from the *real* outcome sequence is literally the §4.1 trajectory.
    Position *i* carries the count as it stood **before** pitch *i*.
    """
    balls, strikes = int(start[0]), int(start[1])
    traj: list[tuple[int, int]] = []
    for o in np.asarray(outcomes).tolist():
        traj.append((balls, strikes))
        balls, strikes = _advance_count(balls, strikes, int(o))
    return traj


def count_state_index(balls: int, strikes: int) -> int:
    return min(balls, 3) * 3 + min(strikes, 2)


def sample_from_probs(probs: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """Inverse-CDF categorical draw: ``probs`` ``(N, C)``, ``u`` ``(N,)``."""
    cdf = probs.cumsum(dim=-1)
    cdf = cdf / cdf[:, -1:].clamp_min(1e-12)
    idx = (u.unsqueeze(-1) > cdf).sum(dim=-1)
    return idx.clamp_(max=probs.shape[-1] - 1)


def build_pa_start_tensors(
    ctx: PAContext, context_dim: int = CONTEXT_DIM
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """BOS + prefix tokens and the position-0 context (same shape as the v2 sim)."""
    balls, strikes = ctx.count
    on1, on2, on3 = ctx.runners
    ctx_list = PitchTokenizer.encode_context(
        balls=balls, strikes=strikes, outs=ctx.outs,
        on_1b=on1, on_2b=on2, on_3b=on3,
        stand=ctx.batter_stand, inning=ctx.inning, score_diff=ctx.score_diff,
    )
    vec = PitchTokenizer.context_to_tensor(ctx_list, ump_scalar=ctx.umpire_scalar)
    if context_dim < vec.shape[-1]:
        vec = vec[:context_dim]
    prefix = list(ctx.prefix_pitch_tokens)
    tokens = torch.tensor([BOS_TOKEN] + prefix, dtype=torch.long).unsqueeze(0)
    context = vec.unsqueeze(0).unsqueeze(0).expand(1, tokens.shape[1], -1).contiguous()
    return tokens, context, len(prefix)


@torch.no_grad()
def rollout_pa_batch(
    model: FactorizedPitchGPT,
    outcome_head: V3OutcomeHead | None,
    *,
    start_context: torch.Tensor,
    start_count: np.ndarray,
    uniforms: np.ndarray,
    prefix_tokens: torch.Tensor | None = None,
    temps: HeadTemperatures | None = None,
    horizon: int = HORIZON,
    mask_stats: MaskStats | None = None,
    return_probs: bool = False,
) -> dict:
    """Roll ``P`` PAs x ``S`` samples forward, fully vectorised.

    Parameters
    ----------
    start_context : ``(P, CONTEXT_DIM)``
        The PA-start context vector.  Only the ``count_state`` block is mutated
        during the rollout (§4.1); every other field is PA-invariant, exactly as
        ``pitchgpt_sim.rollout`` treats it.
    start_count : ``(P, 2)`` int
        Pre-PA ``(balls, strikes)``.
    uniforms : ``(P, S, horizon, 4)``
        Per-PA uniform blocks from :func:`pa_uniform_block` (§4.3.2).
    prefix_tokens : ``(P, K)`` optional
        Pitches already thrown in the PA; the rollout resumes after them.

    Returns a dict of numpy arrays shaped ``(P, S, ...)``.
    """
    device = next(model.parameters()).device
    P = start_context.shape[0]
    S = uniforms.shape[1]
    N = P * S
    t_t, t_z, t_v, t_o = (
        (1.0, 1.0, 1.0, 1.0) if temps is None else temps.as_tuple()
    )

    u = torch.as_tensor(uniforms, dtype=torch.float32, device=device).reshape(
        N, horizon, N_UNIFORM_FIELDS
    )
    ctx0 = start_context.to(device).unsqueeze(1).expand(P, S, -1).reshape(N, -1)
    running_context = ctx0.clone()

    if prefix_tokens is not None and prefix_tokens.shape[1] > 0:
        pre = prefix_tokens.to(device).unsqueeze(1).expand(P, S, -1).reshape(N, -1)
        cur_tokens = torch.cat(
            [torch.full((N, 1), BOS_TOKEN, dtype=torch.long, device=device), pre],
            dim=1,
        )
    else:
        cur_tokens = torch.full((N, 1), BOS_TOKEN, dtype=torch.long, device=device)
    cur_context = ctx0.unsqueeze(1).expand(N, cur_tokens.shape[1], -1).contiguous()

    balls = torch.as_tensor(
        np.repeat(start_count[:, 0], S), dtype=torch.long, device=device
    )
    strikes = torch.as_tensor(
        np.repeat(start_count[:, 1], S), dtype=torch.long, device=device
    )
    alive = torch.ones(N, dtype=torch.bool, device=device)

    adv_b = torch.as_tensor(_ADV_DBALLS, device=device)
    adv_s = torch.as_tensor(_ADV_DSTRIKES, device=device)
    adv_foul = torch.as_tensor(_ADV_IS_FOUL, device=device)
    terminal = torch.as_tensor(_TERMINAL_MASK, device=device)

    pitch_tokens = torch.full(
        (N, horizon), ROLLOUT_PAD_PITCH, dtype=torch.long, device=device
    )
    outcomes = torch.full(
        (N, horizon), ROLLOUT_PAD_OUTCOME, dtype=torch.long, device=device
    )
    terminated = torch.zeros((N, horizon), dtype=torch.bool, device=device)
    pa_outcome = torch.full(
        (N,), ROLLOUT_PAD_OUTCOME, dtype=torch.long, device=device
    )
    final_b = balls.clone()
    final_s = strikes.clone()
    probs_out = (
        torch.full((N, horizon, NUM_OUTCOME_CLASSES), float("nan"), device=device)
        if (return_probs and outcome_head is not None)
        else None
    )

    for pos in range(horizon):
        if not bool(alive.any()):
            break
        hidden = model.forward_hidden(cur_tokens, cur_context)[:, -1, :]

        tok, t_idx, z_idx, v_idx = model.sample_tokens(
            hidden, uniforms=u[:, pos, :3], temps=temps, stats=mask_stats,
        )
        pitch_tokens[:, pos] = torch.where(alive, tok, pitch_tokens[:, pos])

        if outcome_head is not None:
            logits = outcome_head(hidden, running_context, t_idx, z_idx, v_idx)
            p_out = F.softmax(logits / max(float(t_o), 1e-8), dim=-1)
            if probs_out is not None:
                probs_out[:, pos, :] = torch.where(
                    alive.unsqueeze(-1), p_out, probs_out[:, pos, :]
                )
            o = sample_from_probs(p_out, u[:, pos, 3])
            outcomes[:, pos] = torch.where(alive, o, outcomes[:, pos])

            is_term = terminal[o] & alive
            pa_outcome = torch.where(is_term, o, pa_outcome)
            final_b = torch.where(is_term, balls, final_b)
            final_s = torch.where(is_term, strikes, final_s)
            terminated[:, pos] |= is_term

            # Vectorised _advance_count on the still-alive, non-terminal rows.
            step = alive & ~is_term
            db = adv_b[o]
            ds = torch.where(
                adv_foul[o] & (strikes >= 2), torch.zeros_like(strikes), adv_s[o]
            )
            balls = torch.where(step, balls + db, balls)
            strikes = torch.where(step, strikes + ds, strikes)

            ends = step & ((balls >= 4) | (strikes >= 3))
            terminated[:, pos] |= ends
            final_b = torch.where(ends, balls.clamp(max=4), final_b)
            final_s = torch.where(ends, strikes.clamp(max=3), final_s)
            alive = alive & ~is_term & ~ends
        # With no outcome head the PA never terminates early; it runs to horizon.

        if pos + 1 >= horizon:
            break
        next_tok = torch.where(alive, tok, torch.full_like(tok, PAD_TOKEN))
        cur_tokens = torch.cat([cur_tokens, next_tok.unsqueeze(1)], dim=1)

        # §4.1 dynamic mid-PA context: re-emit the count_state one-hot.
        cs = balls.clamp(max=3) * 3 + strikes.clamp(max=2)
        running_context[:, :NUM_COUNT_STATES] = 0.0
        running_context.scatter_(
            1, cs.unsqueeze(1), torch.ones((N, 1), device=device)
        )
        cur_context = torch.cat([cur_context, running_context.unsqueeze(1)], dim=1)

    still = alive
    final_b = torch.where(still, balls.clamp(max=4), final_b)
    final_s = torch.where(still, strikes.clamp(max=3), final_s)

    def _np(t, dtype=np.int64):
        return t.detach().cpu().numpy().astype(dtype).reshape(P, S, *t.shape[1:])

    out = {
        "pitch_tokens": _np(pitch_tokens),
        "pa_terminated": _np(terminated, bool),
        "final_count": np.stack(
            [_np(final_b), _np(final_s)], axis=-1
        ),
        "n_truncated": int(still.sum().item()),
    }
    if outcome_head is not None:
        out["outcomes"] = _np(outcomes)
        out["pa_outcome"] = _np(pa_outcome)
    else:
        out["outcomes"] = None
        out["pa_outcome"] = None
    out["outcome_probs"] = (
        probs_out.detach().cpu().numpy().reshape(P, S, horizon, NUM_OUTCOME_CLASSES)
        if probs_out is not None
        else None
    )
    return out


@torch.no_grad()
def rollout_v3(
    starting_context: PAContext,
    *,
    model: FactorizedPitchGPT,
    outcome_head: V3OutcomeHead | None,
    temps: HeadTemperatures | None = None,
    n_samples: int = 100,
    horizon: int = HORIZON,
    pa_index: int = 0,
    seed: int | None = None,
    return_probs: bool = False,
    mask_stats: MaskStats | None = None,
) -> RolloutResult:
    """Single-PA door onto :func:`rollout_pa_batch`, in the v2 ``RolloutResult`` shape.

    ``seed`` overrides the §4.3.2 per-PA seed for callers that need an explicit
    stream; otherwise ``42 + pa_index * 1000`` is used.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    device = next(model.parameters()).device

    init_tokens, init_context, prefix_len = build_pa_start_tensors(
        starting_context, model.context_dim,
    )
    start_ctx = init_context[0, 0, :].unsqueeze(0).to(device)
    prefix = (
        init_tokens[:, 1:].to(device) if prefix_len else None
    )
    rng = np.random.default_rng(
        pa_seed(pa_index) if seed is None else int(seed) & 0x7FFFFFFF
    )
    uniforms = rng.random((1, n_samples, horizon, N_UNIFORM_FIELDS))
    start_count = np.array([starting_context.count], dtype=np.int64)

    res = rollout_pa_batch(
        model, outcome_head,
        start_context=start_ctx,
        start_count=start_count,
        uniforms=uniforms,
        prefix_tokens=prefix,
        temps=temps,
        horizon=horizon,
        mask_stats=mask_stats,
        return_probs=return_probs,
    )

    meta = {
        "temperature": 1.0,
        "seed": int(pa_seed(pa_index) if seed is None else seed),
        "pa_index": int(pa_index),
        "n_samples": int(n_samples),
        "horizon": int(horizon),
        "backbone_version": "v3_factorized",
        "outcome_predictor": (
            "pg_v3_factorized_head" if outcome_head is not None else "none"
        ),
        "rollout_engine_version": "v3-factorized-1",
        "n_truncated": int(res["n_truncated"]),
        "prefix_len": int(prefix_len),
        "head_temperatures": (temps.to_dict() if temps is not None else None),
        "sampling": "inverse-CDF from the per-PA uniform block (spec §4.3.2)",
    }
    return RolloutResult(
        pitch_tokens=res["pitch_tokens"][0],
        pitch_probs=None,
        outcomes=(res["outcomes"][0] if res["outcomes"] is not None else None),
        outcome_probs=(
            res["outcome_probs"][0] if res["outcome_probs"] is not None else None
        ),
        pa_terminated=res["pa_terminated"][0],
        pa_outcome=(res["pa_outcome"][0] if res["pa_outcome"] is not None else None),
        final_count=res["final_count"][0],
        sampling_metadata=meta,
    )


class V3SimOptInError(RuntimeError):
    """Raised when the v3 sim path is reached without an explicit opt-in."""


V3_SIM_OPTIN_ENV = "PITCHGPT_V3_SIM_OPTIN"


def v3_sim_optin_enabled(enable_v3: bool | None = None) -> bool:
    """True only on an explicit opt-in (§8.1 — production stays on v2026.04.23)."""
    if enable_v3 is not None:
        return bool(enable_v3)
    return str(os.environ.get(V3_SIM_OPTIN_ENV, "")).strip().lower() in {
        "1", "true", "yes", "on",
    }


def rollout_v3_optin(
    starting_context: PAContext,
    *,
    model: FactorizedPitchGPT,
    outcome_head: V3OutcomeHead | None,
    enable_v3: bool | None = None,
    **kwargs,
) -> RolloutResult:
    """The ONLY sanctioned sim-shaped door onto the v3 rollout.

    Refuses unless the caller explicitly opts in.  Production
    ``pitchgpt_sim.rollout()`` never reaches this function: no registry entry,
    no import, no default.  v3 is dev-tier until §6.8 returns PASS on the
    sealed-2026 contact and the alias move is separately reviewed (§8.1).
    """
    if not v3_sim_optin_enabled(enable_v3):
        raise V3SimOptInError(
            "PitchGPT v3 is a dev-tier artifact: its gate suite has been run on "
            "the 2024 BURNED dev tier only and the §5.5 lockbox contact has NOT "
            "been made. Production sim stays on registry alias v2026.04.23. "
            f"Pass enable_v3=True or set {V3_SIM_OPTIN_ENV}=1 to use it anyway."
        )
    return rollout_v3(
        starting_context, model=model, outcome_head=outcome_head, **kwargs,
    )
