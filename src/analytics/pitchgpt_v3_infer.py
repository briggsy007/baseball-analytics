"""
PitchGPT v3 — shared PA-scoped inference helpers.

Every §6 statistic, the §4.4 temperature fits and the §6.2 frozen-v2
comparison are computed in the **PA-scoped regime**: a sequence is
``BOS`` followed by the PA's real pitches, and the context at position *j* is
the pre-pitch-*j* context (PA-invariant fields from the PA start, mutating
``count_state``).  That is byte-for-byte the construction
``pitchgpt_sim.rollout`` and ``pitchgpt_v3_rollout.rollout_v3`` feed their
backbones, so the calibration that is fit here is the calibration the sim
actually consumes (spec §4.3: "the training-time rollout is the inference-time
rollout").

The one deliberate exception is the K-v2-FIT-A NLL comparison (§7.1), which is
scored on whole game sequences — the regime the frozen v2 backbone was
trained and previously reported in.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from src.analytics.pitchgpt import (
    BOS_TOKEN,
    CONTEXT_DIM,
    NUM_PITCH_TYPES,
    NUM_VELO_BUCKETS,
    NUM_ZONES,
    PAD_TOKEN,
)
from src.analytics.pitchgpt_v3 import HeadTemperatures, fields_from_token
from src.analytics.pitchgpt_v3_data import context_indices_to_tensor

logger = logging.getLogger(__name__)


def pa_batch_tensors(pa: dict, sl: slice, device, horizon: int = 6):
    """Build ``(inputs, contexts, targets, valid)`` for a slice of PAs."""
    tok = torch.from_numpy(pa["pa_tok"][sl].astype(np.int64)).to(device)
    ctxi = pa["pa_ctx_idx"][sl].astype(np.int64)
    ump = pa["pa_ump"][sl].astype(np.float32)
    lens = torch.from_numpy(pa["pa_len"][sl].astype(np.int64)).to(device)
    B = tok.shape[0]
    L = int(lens.max().item()) if B else 0
    if L < 1:
        return None
    ctx_full = context_indices_to_tensor(
        ctxi, np.repeat(ump[:, None], horizon, axis=1), CONTEXT_DIM,
    ).to(device)
    inp = torch.cat(
        [
            torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device),
            tok[:, : L - 1].clamp(min=0),
        ],
        dim=1,
    )
    pad = torch.arange(L, device=device).unsqueeze(0) >= lens.unsqueeze(1)
    inp = inp.masked_fill(pad, PAD_TOKEN)
    ctx = ctx_full[:, :L, :].masked_fill(pad.unsqueeze(-1), 0.0)
    return inp, ctx, tok[:, :L].clamp(min=0), ~pad


@torch.no_grad()
def collect_v3_head_probs(
    model,
    pa: dict,
    device,
    temps: HeadTemperatures | None = None,
    batch: int = 512,
    horizon: int = 6,
    return_hidden: bool = False,
) -> dict:
    """Per-pitch conditional head probabilities in the PA-scoped regime.

    Returns ``p_type (N,17)``, ``p_zone (N,26)`` conditioned on the *true*
    type, ``p_velo (N,5)`` conditioned on the true (type, zone), the three
    label arrays, the per-pitch within-PA position and the count state.
    """
    t_t, t_z, t_v, _ = (1.0, 1.0, 1.0, 1.0) if temps is None else temps.as_tuple()
    n_pa = len(pa["pa_len"])
    out: dict[str, list] = {
        "p_type": [], "p_zone": [], "p_velo": [],
        "y_type": [], "y_zone": [], "y_velo": [],
        "pa_pos": [], "count_state": [], "pa_index": [], "hidden": [],
    }
    for start in range(0, n_pa, batch):
        sl = slice(start, min(start + batch, n_pa))
        prepped = pa_batch_tensors(pa, sl, device, horizon)
        if prepped is None:
            continue
        inp, ctx, tgt, valid = prepped
        hidden = model.forward_hidden(inp, ctx)
        t_s, z_s, v_s = fields_from_token(tgt)
        lt = model.type_logits(hidden) / t_t
        lz = model.zone_logits(hidden, t_s) / t_z
        lv = model.velo_logits(hidden, t_s, z_s) / t_v
        m = valid
        out["p_type"].append(F.softmax(lt, -1)[m].cpu().numpy())
        out["p_zone"].append(F.softmax(lz, -1)[m].cpu().numpy())
        out["p_velo"].append(F.softmax(lv, -1)[m].cpu().numpy())
        out["y_type"].append(t_s[m].cpu().numpy())
        out["y_zone"].append(z_s[m].cpu().numpy())
        out["y_velo"].append(v_s[m].cpu().numpy())
        B, L = tgt.shape
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        out["pa_pos"].append(pos[m].cpu().numpy())
        cs = torch.from_numpy(
            pa["pa_ctx_idx"][sl, :L, 0].astype(np.int64)
        ).to(device)
        out["count_state"].append(cs[m].cpu().numpy())
        pai = torch.arange(start, start + B, device=device).unsqueeze(1).expand(B, L)
        out["pa_index"].append(pai[m].cpu().numpy())
        if return_hidden:
            out["hidden"].append(hidden[m].cpu().numpy().astype(np.float32))
    res = {k: (np.concatenate(v) if v else np.empty(0)) for k, v in out.items() if v}
    if not return_hidden:
        res.pop("hidden", None)
    return res


@torch.no_grad()
def collect_v2_marginal_head_probs(
    backbone,
    pa: dict,
    device,
    batch: int = 512,
    horizon: int = 6,
) -> dict:
    """§6.2 — project the frozen v2 flat 2,210-way softmax onto the three fields.

    ``p_type(t) = Σ_{z,v} p(token)``; the conditionals follow by division.
    Scored on the identical PA-scoped rows as :func:`collect_v3_head_probs`.
    """
    n_pa = len(pa["pa_len"])
    acc = {"p_type": [], "p_zone": [], "p_velo": []}
    for start in range(0, n_pa, batch):
        sl = slice(start, min(start + batch, n_pa))
        prepped = pa_batch_tensors(pa, sl, device, horizon)
        if prepped is None:
            continue
        inp, ctx, tgt, valid = prepped
        if backbone.context_dim < CONTEXT_DIM:
            ctx_in = ctx[..., : backbone.context_dim]
        else:
            ctx_in = ctx
        logits = backbone(inp, ctx_in)
        p = F.softmax(logits, dim=-1)[valid]  # (N, 2210)
        n = p.shape[0]
        p4 = p.view(n, NUM_PITCH_TYPES, NUM_ZONES, NUM_VELO_BUCKETS)
        t_s, z_s, v_s = fields_from_token(tgt)
        t_i, z_i = t_s[valid], z_s[valid]
        p_type = p4.sum(dim=(2, 3))                              # (N, 17)
        rows = torch.arange(n, device=p.device)
        p_tz = p4[rows, t_i]                                     # (N, 26, 5)
        p_zone = p_tz.sum(dim=-1)
        p_zone = p_zone / p_zone.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        p_velo = p4[rows, t_i, z_i]                              # (N, 5)
        p_velo = p_velo / p_velo.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        acc["p_type"].append(p_type.cpu().numpy())
        acc["p_zone"].append(p_zone.cpu().numpy())
        acc["p_velo"].append(p_velo.cpu().numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


@torch.no_grad()
def collect_outcome_probs(
    model,
    head,
    pa: dict,
    device,
    temperature: float = 1.0,
    batch: int = 512,
    horizon: int = 6,
) -> dict:
    """Per-pitch 7-class outcome probabilities in the PA-scoped regime."""
    n_pa = len(pa["pa_len"])
    probs, labels, positions, counts = [], [], [], []
    for start in range(0, n_pa, batch):
        sl = slice(start, min(start + batch, n_pa))
        prepped = pa_batch_tensors(pa, sl, device, horizon)
        if prepped is None:
            continue
        inp, ctx, tgt, valid = prepped
        hidden = model.forward_hidden(inp, ctx)
        t_s, z_s, v_s = fields_from_token(tgt)
        logits = head(hidden, ctx, t_s, z_s, v_s) / max(float(temperature), 1e-8)
        y = torch.from_numpy(
            pa["pa_outc"][sl, : tgt.shape[1]].astype(np.int64)
        ).to(device)
        m = valid & (y >= 0)
        probs.append(F.softmax(logits, -1)[m].cpu().numpy())
        labels.append(y[m].cpu().numpy())
        B, L = tgt.shape
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        positions.append(pos[m].cpu().numpy())
        cs = torch.from_numpy(
            pa["pa_ctx_idx"][sl, :L, 0].astype(np.int64)
        ).to(device)
        counts.append(cs[m].cpu().numpy())
    return {
        "probs": np.concatenate(probs) if probs else np.empty((0, 7)),
        "labels": np.concatenate(labels) if labels else np.empty(0, dtype=int),
        "pa_pos": np.concatenate(positions) if positions else np.empty(0, dtype=int),
        "count_state": np.concatenate(counts) if counts else np.empty(0, dtype=int),
    }


@torch.no_grad()
def collect_v2_outcome_probs(
    backbone, a1_head, pa: dict, device, temperature: float,
    batch: int = 512, horizon: int = 6,
) -> dict:
    """Same, for the frozen v2 backbone + frozen A1 concat head (§6.2)."""
    n_pa = len(pa["pa_len"])
    probs, labels, positions, counts = [], [], [], []
    for start in range(0, n_pa, batch):
        sl = slice(start, min(start + batch, n_pa))
        prepped = pa_batch_tensors(pa, sl, device, horizon)
        if prepped is None:
            continue
        inp, ctx, tgt, valid = prepped
        ctx_in = (
            ctx[..., : backbone.context_dim]
            if backbone.context_dim < CONTEXT_DIM else ctx
        )
        from src.analytics.pitchgpt_outcome_head import (
            extract_backbone_hidden_states,
        )

        hidden = extract_backbone_hidden_states(backbone, inp, ctx_in)
        t_s, z_s, v_s = fields_from_token(tgt)
        logits = a1_head(
            hidden,
            ctx,
            F.one_hot(t_s, NUM_PITCH_TYPES).float(),
            F.one_hot(z_s, NUM_ZONES).float(),
            F.one_hot(v_s, NUM_VELO_BUCKETS).float(),
        ) / max(float(temperature), 1e-8)
        y = torch.from_numpy(
            pa["pa_outc"][sl, : tgt.shape[1]].astype(np.int64)
        ).to(device)
        m = valid & (y >= 0)
        probs.append(F.softmax(logits, -1)[m].cpu().numpy())
        labels.append(y[m].cpu().numpy())
        B, L = tgt.shape
        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        positions.append(pos[m].cpu().numpy())
        cs = torch.from_numpy(
            pa["pa_ctx_idx"][sl, :L, 0].astype(np.int64)
        ).to(device)
        counts.append(cs[m].cpu().numpy())
    return {
        "probs": np.concatenate(probs) if probs else np.empty((0, 7)),
        "labels": np.concatenate(labels) if labels else np.empty(0, dtype=int),
        "pa_pos": np.concatenate(positions) if positions else np.empty(0, dtype=int),
        "count_state": np.concatenate(counts) if counts else np.empty(0, dtype=int),
    }


def fit_temperature(
    probs: np.ndarray, labels: np.ndarray, bounds: tuple[float, float] = (0.25, 5.0)
) -> dict:
    """Single-scalar temperature by NLL minimisation (§4.4).

    Operates on *probabilities*: logits are recovered as ``log p`` (softmax is
    shift-invariant, so this is exact) and rescaled by ``1/T``.
    """
    from scipy.optimize import minimize_scalar

    logp = np.log(np.clip(probs, 1e-30, None))
    y = labels.astype(int)
    rows = np.arange(len(y))

    def nll(T: float) -> float:
        z = logp / max(T, 1e-6)
        z = z - z.max(axis=1, keepdims=True)
        lse = np.log(np.exp(z).sum(axis=1))
        return float(np.mean(lse - z[rows, y]))

    res = minimize_scalar(nll, bounds=bounds, method="bounded")
    T = float(res.x)
    return {
        "T": T,
        "nll_at_T": nll(T),
        "nll_at_1": nll(1.0),
        "n": int(len(y)),
        "bounds": list(bounds),
        "converged": bool(res.success),
    }


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(probs, 1e-30, None)) / max(T, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ── PA-level rollout evaluation (G3 / G4 / G5 and K-v2-FIT-B) ────────────────
#
# One driver produces every PA-level statistic the spec gates, so the
# K-v2-FIT-B fit-cohort measurement (§7.2) and the dev-tier §6 suite are
# computed by identical code on different cohorts.

from src.analytics.pitchgpt_outcome_head import NUM_OUTCOME_CLASSES  # noqa: E402
from src.analytics.pitchgpt_sim import (  # noqa: E402
    OUTCOME_HBP,
    OUTCOME_IN_PLAY_HIT,
    ROLLOUT_PAD_OUTCOME,
)

HORIZON = 6

# PA-terminal wOBA map.  ``WObaTable.default()`` covers only the five
# pitch-outcome termini; walks and strikeouts terminate on COUNT and carry no
# pitch-outcome class, so the two count-terminal states are added with the
# standard linear-weights values.  Applied identically to model rollouts and to
# the empirical PA reconstruction, so the two sides are commensurable
# (deviations-log entry 7).
WOBA_K: float = 0.0
WOBA_BB: float = 0.690
WOBA_HBP: float = 0.708
WOBA_IN_PLAY_HIT: float = 0.892
WOBA_IN_PLAY_OUT: float = 0.0

#: PA terminal-state codes used by both sides of the comparison.
PA_K, PA_BB, PA_HBP, PA_HIT, PA_OUT, PA_TRUNC = 0, 1, 2, 3, 4, 5
PA_TERMINAL_NAMES = ("K", "BB", "HBP", "in_play_hit", "in_play_out", "truncated")
_PA_WOBA = np.array(
    [WOBA_K, WOBA_BB, WOBA_HBP, WOBA_IN_PLAY_HIT, WOBA_IN_PLAY_OUT, np.nan]
)


def empirical_position_marginals(pa: dict, horizon: int = HORIZON):
    """``(horizon, 7)`` empirical class marginals + counts of the real PAs.

    Identical construction to ``scripts/pitchgpt_fit_rollout_calibration.py::
    _empirical_perpos_marginals``: position *p* pools every real pitch thrown at
    within-PA index *p* whose 7-class outcome is labelled.
    """
    outc = pa["pa_outc"][:, :horizon]
    lens = pa["pa_len"]
    counts = np.zeros((horizon, NUM_OUTCOME_CLASSES), dtype=np.float64)
    for p in range(horizon):
        col = outc[:, p]
        sel = (col >= 0) & (col < NUM_OUTCOME_CLASSES) & (lens > p)
        if sel.any():
            counts[p] = np.bincount(
                col[sel].astype(np.int64), minlength=NUM_OUTCOME_CLASSES
            ).astype(np.float64)
    marg = counts / np.clip(counts.sum(axis=1, keepdims=True), 1.0, None)
    return marg, counts


def empirical_pa_terminals(pa: dict, horizon: int = HORIZON) -> np.ndarray:
    """Terminal state of each REAL PA, replayed through the sim state machine.

    The real per-pitch outcome sequence is fed to the production
    ``_advance_count`` with the production termination rules, so an empirical PA
    and a sampled PA are classified by exactly the same logic (and both are
    truncated at ``horizon``).
    """
    from src.analytics.pitchgpt_sim import _TERMINAL_INPLAY_OUTCOMES, _advance_count

    outc = pa["pa_outc"][:, :horizon]
    lens = pa["pa_len"]
    ctx0 = pa["pa_ctx_idx"][:, 0, 0].astype(int)
    out = np.full(len(lens), PA_TRUNC, dtype=np.int64)
    for i in range(len(lens)):
        b, s = int(ctx0[i]) // 3, int(ctx0[i]) % 3
        for j in range(min(int(lens[i]), horizon)):
            o = int(outc[i, j])
            if o < 0:
                break
            if o in _TERMINAL_INPLAY_OUTCOMES:
                out[i] = (
                    PA_HBP if o == OUTCOME_HBP
                    else (PA_HIT if o == OUTCOME_IN_PLAY_HIT else PA_OUT)
                )
                break
            b, s = _advance_count(b, s, o)
            if b >= 4:
                out[i] = PA_BB
                break
            if s >= 3:
                out[i] = PA_K
                break
    return out


def classify_rollout_terminals(
    pa_outcome: np.ndarray, final_count: np.ndarray, terminated: np.ndarray
) -> np.ndarray:
    """Terminal state of each sampled PA — mirror of :func:`empirical_pa_terminals`."""
    any_term = terminated.any(axis=-1)
    out = np.full(pa_outcome.shape, PA_TRUNC, dtype=np.int64)
    out = np.where(any_term & (final_count[..., 1] >= 3), PA_K, out)
    out = np.where(any_term & (final_count[..., 0] >= 4), PA_BB, out)
    out = np.where(pa_outcome == OUTCOME_HBP, PA_HBP, out)
    out = np.where(pa_outcome == OUTCOME_IN_PLAY_HIT, PA_HIT, out)
    out = np.where(
        (pa_outcome != ROLLOUT_PAD_OUTCOME)
        & (pa_outcome != OUTCOME_HBP)
        & (pa_outcome != OUTCOME_IN_PLAY_HIT),
        PA_OUT,
        out,
    )
    return out


def pa_length_from_terminated(terminated: np.ndarray, horizon: int = HORIZON):
    any_term = terminated.any(axis=-1)
    first = terminated.argmax(axis=-1) + 1
    return np.where(any_term, first, horizon)


@torch.no_grad()
def rollout_cohort(
    model,
    outcome_head,
    pa: dict,
    device,
    *,
    temps: HeadTemperatures | None = None,
    n_samples: int = 100,
    horizon: int = HORIZON,
    pa_batch: int = 48,
    log_every: int = 50,
    progress: str = "rollout",
) -> dict:
    """Roll every PA in ``pa`` and return every PA-level statistic the spec gates.

    Per-PA randomness is the §4.3.2 block ``42 + pa_index * 1000``, so the
    result does not depend on ``pa_batch``.
    """
    import time as _time

    from src.analytics.pitchgpt_v3 import MaskStats
    from src.analytics.pitchgpt_v3_data import context_indices_to_tensor
    from src.analytics.pitchgpt_v3_rollout import pa_uniform_block, rollout_pa_batch

    n_pa = len(pa["pa_len"])
    stats = MaskStats()
    pos_counts = np.zeros((horizon, NUM_OUTCOME_CLASSES), dtype=np.float64)
    term_all = np.empty((n_pa, n_samples), dtype=np.int8)
    palen_all = np.empty((n_pa, n_samples), dtype=np.int8)
    t0 = _time.perf_counter()

    for start in range(0, n_pa, pa_batch):
        end = min(start + pa_batch, n_pa)
        sl = slice(start, end)
        ctx_idx = pa["pa_ctx_idx"][sl, 0, :]
        ump = pa["pa_ump"][sl]
        start_ctx = context_indices_to_tensor(ctx_idx, ump, CONTEXT_DIM).to(device)
        cs0 = pa["pa_ctx_idx"][sl, 0, 0].astype(np.int64)
        start_count = np.stack([cs0 // 3, cs0 % 3], axis=1)
        uniforms = np.stack(
            [pa_uniform_block(i, n_samples, horizon) for i in range(start, end)]
        )
        res = rollout_pa_batch(
            model, outcome_head,
            start_context=start_ctx,
            start_count=start_count,
            uniforms=uniforms,
            temps=temps,
            horizon=horizon,
            mask_stats=stats,
        )
        o = res["outcomes"]                       # (P, S, H)
        for p in range(horizon):
            col = o[:, :, p].ravel()
            sel = (col >= 0) & (col < NUM_OUTCOME_CLASSES)
            if sel.any():
                pos_counts[p] += np.bincount(
                    col[sel], minlength=NUM_OUTCOME_CLASSES
                ).astype(np.float64)
        term_all[sl] = classify_rollout_terminals(
            res["pa_outcome"], res["final_count"], res["pa_terminated"]
        ).astype(np.int8)
        palen_all[sl] = pa_length_from_terminated(
            res["pa_terminated"], horizon
        ).astype(np.int8)
        if log_every and (start // pa_batch) % log_every == 0 and start:
            dt = _time.perf_counter() - t0
            logger.info(
                "[%s] %d/%d PAs  %.1f PA/s  eta %.0fs",
                progress, start, n_pa, start / dt,
                (n_pa - start) / max(start / dt, 1e-9),
            )

    pos_marg = pos_counts / np.clip(pos_counts.sum(axis=1, keepdims=True), 1.0, None)
    woba = _PA_WOBA[term_all]
    return {
        "position_marginals": pos_marg,
        "position_counts": pos_counts,
        "pa_terminal": term_all,
        "pa_length": palen_all,
        "pa_woba_samples": woba.astype(np.float32),
        "mask_events": stats.as_dict(),
        "n_pa": int(n_pa),
        "n_samples": int(n_samples),
        "horizon": int(horizon),
        "elapsed_sec": round(_time.perf_counter() - t0, 1),
    }


def terminal_share(codes: np.ndarray) -> dict:
    """Share of each PA terminal state (``truncated`` included)."""
    flat = np.asarray(codes).ravel()
    n = max(len(flat), 1)
    cnt = np.bincount(flat.astype(np.int64), minlength=len(PA_TERMINAL_NAMES))
    return {
        PA_TERMINAL_NAMES[i]: float(cnt[i] / n)
        for i in range(len(PA_TERMINAL_NAMES))
    }
