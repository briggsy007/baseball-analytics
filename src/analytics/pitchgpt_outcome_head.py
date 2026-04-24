"""
PitchGPT PA-outcome head (Phase 0.2 design-and-smoke deliverable).

Provides two candidate head architectures for producing a 7-class
pitch-level outcome distribution alongside the PitchGPT backbone's
next-token prediction:

    (a) ``JointOutcomeHead``   -- small linear head added to the live
                                  backbone; trained jointly with the
                                  next-token loss (backbone unfrozen).
    (b) ``FrozenOutcomeHead``  -- 2-layer MLP on top of a **frozen**
                                  backbone's per-pitch hidden state.
                                  Preserves the v2 checkpoint's next-token
                                  calibration guarantees exactly by
                                  construction (the backbone never sees a
                                  gradient).

The 7 outcome classes (locked 2026-04-24 per §3.1 of the sim-engine plan)::

    0 -> ball
    1 -> called_strike
    2 -> swinging_strike
    3 -> foul
    4 -> in_play_out
    5 -> in_play_hit
    6 -> hbp

The design/rationale is in ``docs/pitchgpt_sim_engine/pa_outcome_head_design.md``.
This module only implements the minimum code surface needed for the Phase
0.2 smoke harness (``scripts/pitchgpt_outcome_head_smoke.py``); Phase 0.3
will extend and productionise one of the two variants depending on the
smoke verdict.

Public API
----------
- ``OUTCOME_CLASSES`` / ``OUTCOME_TO_IDX`` / ``IDX_TO_OUTCOME``
- ``classify_pitch_outcome(description, events)``    -- label constructor
- ``JointOutcomeHead``                               -- option (a) head
- ``FrozenOutcomeHead``                              -- option (b) head
- ``PitchGPTWithOutcomeHead``                        -- option (a) wrapper
- ``extract_backbone_hidden_states(...)``            -- option (b) feature
                                                         extractor
- ``seven_class_log_loss(...)`` / ``seven_class_ece(...)`` -- eval utils

Per Phase-0.2 spec:
- Do NOT overwrite ``models/pitchgpt_v2.pt``.
- Smoke checkpoints land at ``models/pitchgpt_outcome_smoke_joint.pt`` and
  ``models/pitchgpt_outcome_smoke_frozen.pt``.
- Read-only DuckDB use only.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.analytics.pitchgpt import (
    CONTEXT_DIM,
    PAD_TOKEN,
    PitchGPTModel,
)

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 7-class outcome vocabulary
# ═════════════════════════════════════════════════════════════════════════════

OUTCOME_CLASSES: tuple[str, ...] = (
    "ball",
    "called_strike",
    "swinging_strike",
    "foul",
    "in_play_out",
    "in_play_hit",
    "hbp",
)
NUM_OUTCOME_CLASSES: int = len(OUTCOME_CLASSES)

OUTCOME_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(OUTCOME_CLASSES)}
IDX_TO_OUTCOME: dict[int, str] = {i: c for i, c in enumerate(OUTCOME_CLASSES)}

# Sentinel for "cannot label this pitch" — excluded from training/eval.
OUTCOME_UNK: int = -1


# Statcast event strings that map to "in_play_hit" (reached base via hit).
_HIT_EVENTS: frozenset[str] = frozenset({
    "single", "double", "triple", "home_run",
})

# Statcast event strings that map to "in_play_out" (ball in play, batter out
# or reached on error / fielder's choice).  ``catcher_interf`` is excluded
# because the ball is not typically put in play.
_IN_PLAY_OUT_EVENTS: frozenset[str] = frozenset({
    "field_out", "force_out", "grounded_into_double_play",
    "double_play", "triple_play",
    "sac_fly", "sac_fly_double_play", "sac_bunt",
    "fielders_choice", "fielders_choice_out",
    "field_error",
    # truncated_pa — PA ended early (game-ending walk-off, etc.); best-effort
    # assign to in_play_out when the pitch description was hit_into_play.
    "truncated_pa",
})


def _normalize_description(descr: Any) -> str:
    """Map raw ``pitches.description`` into a canonical snake_case token.

    Earlier seasons (2017, 2019 backfilled from a different source)
    occasionally carry Title-Case variants such as ``"Called Strike"`` and
    ``"Ball In Dirt"``.  We normalise those before pattern matching so the
    classifier is era-agnostic.  Unknown / free-text descriptions (e.g.,
    player-narration strings from a handful of 2017 rows) collapse to the
    empty string and the caller's fallback path handles them.
    """
    if descr is None:
        return ""
    s = str(descr).strip()
    if not s:
        return ""
    # Common Title-Case → snake_case mapping for the backfill variants.
    title_map = {
        "Ball": "ball",
        "Ball In Dirt": "ball",
        "Foul": "foul",
        "Foul Tip": "foul_tip",
        "Foul Bunt": "foul_bunt",
        "Called Strike": "called_strike",
        "Swinging Strike": "swinging_strike",
        "Swinging Strike (Blocked)": "swinging_strike_blocked",
        "Missed Bunt": "missed_bunt",
        "Hit Into Play": "hit_into_play",
        "Hit By Pitch": "hit_by_pitch",
        "Intent Ball": "intent_ball",
        "Pitchout": "pitchout",
    }
    if s in title_map:
        return title_map[s]
    return s.lower()


def classify_pitch_outcome(description: Any, events: Any) -> int:
    """Map a single pitch's ``(description, events)`` to a 7-class outcome.

    Returns
    -------
    int
        Integer class id in ``[0, 7)`` from :data:`OUTCOME_CLASSES`, or
        :data:`OUTCOME_UNK` (``-1``) if the pitch cannot be confidently
        labelled (e.g., free-text narration, non-standard description).

    Labelling rules (in priority order):

    1. ``description`` is the *pitch-level* signal and decides ball /
       called-strike / swinging-strike / foul / hbp **without needing the
       events column** — those five outcomes are fully determined at the
       pitch level.  The `events` column is only consulted when
       ``description`` is ``hit_into_play`` (i.e., a contact event that
       terminates the PA).

    2. For in-play terminal pitches, ``events`` distinguishes
       ``in_play_hit`` (single / double / triple / home_run) from
       ``in_play_out`` (field_out / force_out / sac / fielders_choice / …).

    3. HBP is captured both via ``description == 'hit_by_pitch'`` and via
       ``events == 'hit_by_pitch'``; both forms collapse to class
       ``hbp``.

    4. ``intent_ball``, ``pitchout``, ``blocked_ball`` are all grouped
       under ``ball`` — they are balls from the batter's POV, and the
       pitch-type-zone-velocity token already separates them.

    5. ``foul_tip``, ``foul_bunt``, ``bunt_foul_tip``,
       ``swinging_strike_blocked`` are all grouped under the appropriate
       base class: ``foul_tip`` → ``foul`` (it is a foul that ends the PA
       only on strike 3, but the batter's outcome is "contact foul"),
       ``swinging_strike_blocked`` → ``swinging_strike``, etc.

    6. ``missed_bunt`` is treated as ``swinging_strike`` (it is a strike
       by rule; the batter swung and missed).  ``bunt_foul_tip`` → foul.

    7. Narration-style descriptions (rare, ~<1% of rows in 2017 backfill)
       return ``OUTCOME_UNK`` — the caller filters them out.  Similarly
       ``catcher_interf`` (the ball is not usually put in play) returns
       UNK.
    """
    d = _normalize_description(description)
    e = (str(events or "")).strip().lower()

    # --- HBP first (both columns can signal it) ---------------------------
    if d == "hit_by_pitch" or e == "hit_by_pitch":
        return OUTCOME_TO_IDX["hbp"]

    # --- Ball variants ----------------------------------------------------
    # ``ball``, ``blocked_ball``, ``intent_ball``, ``pitchout`` are all balls
    # from the batter's POV.  ``ball_in_dirt`` (post-normalisation: "ball")
    # is already captured.
    if d in {"ball", "blocked_ball", "intent_ball", "pitchout"}:
        return OUTCOME_TO_IDX["ball"]

    # --- Called strike ----------------------------------------------------
    if d == "called_strike":
        return OUTCOME_TO_IDX["called_strike"]

    # --- Swinging strike (and blocked variant, missed bunt) ---------------
    if d in {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}:
        return OUTCOME_TO_IDX["swinging_strike"]

    # --- Foul (all flavours that do not end with contact in play) ---------
    if d in {"foul", "foul_tip", "foul_bunt", "bunt_foul_tip"}:
        return OUTCOME_TO_IDX["foul"]

    # --- In play ----------------------------------------------------------
    if d in {"hit_into_play", "in_play"}:
        # Decide hit vs out via ``events``.
        if e in _HIT_EVENTS:
            return OUTCOME_TO_IDX["in_play_hit"]
        if e in _IN_PLAY_OUT_EVENTS:
            return OUTCOME_TO_IDX["in_play_out"]
        # ``catcher_interf`` is technically in play but not a pitch outcome
        # we want to regress — fall through to UNK so it is excluded.
        if e in {"catcher_interf", "catcher_interference"}:
            return OUTCOME_UNK
        # If events is NULL/unknown on a hit_into_play row, best-effort
        # assume in_play_out (it is the dominant class) — BUT only when
        # ``events`` is truly missing.  An empty string counts as missing.
        if e == "":
            return OUTCOME_TO_IDX["in_play_out"]
        return OUTCOME_UNK

    # --- Unknown / narration / unhandled ---------------------------------
    return OUTCOME_UNK


# ═════════════════════════════════════════════════════════════════════════════
# Option (a) — joint head
# ═════════════════════════════════════════════════════════════════════════════


class JointOutcomeHead(nn.Module):
    """Linear head that projects backbone hidden states to 7-class logits.

    For option (a) we keep the outcome head tiny (single linear layer) and
    rely on the backbone's already-rich per-pitch representation.  The
    head's parameter count is ``d_model * 7 + 7`` = 903 at ``d_model=128``,
    which is negligible relative to the backbone's ~1.4M.
    """

    def __init__(self, d_model: int = 128, n_classes: int = NUM_OUTCOME_CLASSES):
        super().__init__()
        self.proj = nn.Linear(d_model, n_classes)
        # Xavier to match the backbone's init convention.
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : (B, S, d_model)
            Per-pitch pre-output-head hidden state from the backbone's
            transformer stack.

        Returns
        -------
        torch.Tensor of shape (B, S, 7)
            7-class outcome logits for every position in the sequence.
        """
        return self.proj(hidden_states)


class PitchGPTWithOutcomeHead(nn.Module):
    """Option (a) wrapper — PitchGPT backbone + joint outcome head.

    Exposes a ``forward`` that returns both next-token logits (for the
    backbone's existing CE) and outcome logits (for the aux CE).  The
    backbone's transformer stack is called once and its hidden state is
    reused by both heads — this is the parameter-efficient version of
    joint training.
    """

    def __init__(self, backbone: PitchGPTModel):
        super().__init__()
        self.backbone = backbone
        self.outcome_head = JointOutcomeHead(d_model=backbone.d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        (token_logits, outcome_logits) with shapes
            token_logits   : (B, S, VOCAB_SIZE)
            outcome_logits : (B, S, NUM_OUTCOME_CLASSES)
        """
        B, S = tokens.shape
        device = tokens.device

        # Mirror the backbone's forward without the final output head so
        # we can reuse the hidden state for both heads.
        tok_emb = self.backbone.token_embedding(tokens)
        ctx_emb = self.backbone.context_proj(context)
        pos_ids = torch.arange(S, device=device).unsqueeze(0)
        pos_ids = pos_ids.clamp(max=self.backbone.max_seq_len - 1)
        pos_emb = self.backbone.pos_embedding(pos_ids)
        x = tok_emb + ctx_emb + pos_emb

        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(S, device=device)
        padding_mask = (tokens == PAD_TOKEN)

        hidden = self.backbone.transformer(
            x, mask=mask, src_key_padding_mask=padding_mask,
        )  # (B, S, d_model)

        token_logits = self.backbone.output_head(hidden)       # (B, S, V)
        outcome_logits = self.outcome_head(hidden)             # (B, S, 7)
        return token_logits, outcome_logits


# ═════════════════════════════════════════════════════════════════════════════
# Option (b) — frozen-backbone MLP head
# ═════════════════════════════════════════════════════════════════════════════


class FrozenOutcomeHead(nn.Module):
    """2-layer MLP on top of frozen backbone hidden states.

    Topology: ``d_model -> hidden_dim -> n_classes`` with GELU + dropout
    between the two linear layers.  Slightly larger than the joint head
    because it has no backbone capacity to rely on — all outcome-specific
    learning happens here.
    """

    def __init__(
        self,
        d_model: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.1,
        n_classes: int = NUM_OUTCOME_CLASSES,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : (N, d_model) or (B, S, d_model)

        Returns
        -------
        Logits of shape (N, 7) or (B, S, 7).
        """
        return self.net(hidden_states)


@torch.no_grad()
def extract_backbone_hidden_states(
    backbone: PitchGPTModel,
    tokens: torch.Tensor,
    context: torch.Tensor,
) -> torch.Tensor:
    """Run the backbone through to the pre-output-head hidden state.

    Used by option (b): we extract per-pitch hidden vectors once (no
    gradient flow into the backbone) and train the MLP head on those
    features.  ``backbone`` must be in ``eval()`` mode; caller is
    responsible for that.
    """
    B, S = tokens.shape
    device = tokens.device

    tok_emb = backbone.token_embedding(tokens)
    ctx_emb = backbone.context_proj(context)
    pos_ids = torch.arange(S, device=device).unsqueeze(0)
    pos_ids = pos_ids.clamp(max=backbone.max_seq_len - 1)
    pos_emb = backbone.pos_embedding(pos_ids)
    x = tok_emb + ctx_emb + pos_emb

    mask = nn.Transformer.generate_square_subsequent_mask(S, device=device)
    padding_mask = (tokens == PAD_TOKEN)

    hidden = backbone.transformer(
        x, mask=mask, src_key_padding_mask=padding_mask,
    )  # (B, S, d_model)
    return hidden


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation utilities
# ═════════════════════════════════════════════════════════════════════════════


def seven_class_log_loss(
    logits: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
) -> float:
    """Multi-class log-loss on the 7-class outcome target.

    Returns mean NLL in nats.  Targets are integer class ids in
    ``[0, 7)``; rows with target == :data:`OUTCOME_UNK` (``-1``) are
    dropped (not penalised).
    """
    if isinstance(logits, np.ndarray):
        logits_t = torch.from_numpy(logits).float()
    else:
        logits_t = logits.detach().float()
    if isinstance(targets, np.ndarray):
        targets_t = torch.from_numpy(targets).long()
    else:
        targets_t = targets.detach().long()

    valid = targets_t != OUTCOME_UNK
    if valid.sum() == 0:
        return float("nan")
    logits_t = logits_t[valid]
    targets_t = targets_t[valid]

    log_probs = F.log_softmax(logits_t, dim=-1)
    nll = -log_probs.gather(1, targets_t.unsqueeze(1)).squeeze(1).mean()
    return float(nll.item())


def seven_class_ece(
    logits: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Top-1 Expected Calibration Error on 7-class outcome logits.

    Matches the definition used by
    :func:`src.analytics.pitchgpt_calibration.expected_calibration_error`
    but inlined so this module has no circular dependency on the full
    calibration harness.  UNK rows are dropped.
    """
    if isinstance(logits, np.ndarray):
        logits_t = torch.from_numpy(logits).float()
    else:
        logits_t = logits.detach().float()
    if isinstance(targets, np.ndarray):
        targets_t = torch.from_numpy(targets).long()
    else:
        targets_t = targets.detach().long()

    valid = targets_t != OUTCOME_UNK
    if valid.sum() == 0:
        return float("nan")
    logits_t = logits_t[valid]
    targets_t = targets_t[valid]

    probs = F.softmax(logits_t, dim=-1)
    top1_p, top1_idx = probs.max(dim=-1)
    correct = (top1_idx == targets_t).float()

    top1_p_np = top1_p.cpu().numpy()
    correct_np = correct.cpu().numpy()

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_total = top1_p_np.shape[0]
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (top1_p_np >= lo) & (top1_p_np <= hi)
        else:
            mask = (top1_p_np >= lo) & (top1_p_np < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        mean_conf = float(top1_p_np[mask].mean())
        emp_acc = float(correct_np[mask].mean())
        ece += (n / n_total) * abs(mean_conf - emp_acc)
    return float(ece)


def class_frequency_prior(targets: np.ndarray) -> np.ndarray:
    """Return a (7,) unit-sum vector of class frequencies.

    Used as a baseline reference: the 7-class log-loss of the frequency
    prior is ``-sum_c freq_c * log(freq_c)`` (the entropy of the class
    distribution).  The outcome head's lift is measured against this
    floor.
    """
    t = np.asarray(targets)
    t = t[t != OUTCOME_UNK]
    counts = np.bincount(t, minlength=NUM_OUTCOME_CLASSES).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return np.full(NUM_OUTCOME_CLASSES, 1.0 / NUM_OUTCOME_CLASSES)
    return counts / total


def prior_log_loss(targets: np.ndarray, prior: np.ndarray | None = None) -> float:
    """Log-loss of the frequency-prior baseline on ``targets``.

    Used as a naive floor for the "≥15% lift over frequency prior"
    success-criterion in Phase 0.3.
    """
    t = np.asarray(targets)
    t = t[t != OUTCOME_UNK]
    if prior is None:
        prior = class_frequency_prior(t)
    # Avoid log(0) — the bootstrap can produce zero-support classes.
    eps = 1e-12
    logp = np.log(np.clip(prior, eps, 1.0))
    ll = -logp[t].mean()
    return float(ll)


# ═════════════════════════════════════════════════════════════════════════════
# Convenience freeze helper (option b)
# ═════════════════════════════════════════════════════════════════════════════


def freeze_backbone(backbone: PitchGPTModel) -> PitchGPTModel:
    """Disable gradients for every parameter of ``backbone``.

    Caller is still responsible for ``backbone.eval()`` so dropout is
    disabled at feature-extraction time.
    """
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone
