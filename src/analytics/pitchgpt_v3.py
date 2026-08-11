"""
PitchGPT v3 — chain-rule factorized output heads (WS5.2).

Spec: ``docs/pitchgpt_sim_engine/PITCHGPT_V2_SPEC.md`` §3 (FROZEN at commit
``b61e05b``, body sha256 ``b19b54d96b496c5ffbff3f9af3070c180609003ec3d5d0aae71bfa84bb8d6d5b``).

The frozen v2 backbone emits a flat 2,210-way composite softmax
(``nn.Linear(128, 2210)`` = 285,090 parameters).  v3 keeps that backbone
byte-for-byte in shape (§3.2) and replaces **only** the output side with the
exact factorisation

.. code-block:: text

    p(token) = p_type(t | h) · p_zone(z | h, t) · p_velo(v | h, t, z)

over the same composite-token arithmetic identity
``token_id = type_idx * 130 + zone_idx * 5 + velo_idx``, so any v3 output
projects back onto the v2 token space.

Head shapes are **[FIXED]** by §3.3 and are asserted at construction:

======================  =============================================  ==========
head                    layers                                          params
======================  =============================================  ==========
``H_type``              ``Linear(128 → 17)``                            2,193
``H_zone``              ``E_type(17×32)`` + ``Lin(160→64)+GELU+Lin(64→26)``  12,538
``H_velo``              ``E_zone(26×32)`` + ``Lin(192→64)+GELU+Lin(64→5)``   13,509
**total**                                                               **28,240**
======================  =============================================  ==========

Nothing in this module touches the v1/v2 code paths; ``pitchgpt.py`` is
imported read-only for the backbone machinery, tokenizer constants and the
7-class outcome vocabulary.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.analytics.pitchgpt import (
    CONTEXT_DIM,
    DEFAULT_DIM,
    DEFAULT_HEADS,
    DEFAULT_LAYERS,
    DEFAULT_MAX_SEQ,
    NUM_PITCH_TYPES,
    NUM_VELO_BUCKETS,
    NUM_ZONES,
    PAD_TOKEN,
    TOTAL_VOCAB,
    VOCAB_SIZE,
)
from src.analytics.pitchgpt_outcome_head import NUM_OUTCOME_CLASSES

logger = logging.getLogger(__name__)

# ── §3.3 [FIXED] constants ───────────────────────────────────────────────────
FIELD_EMB_DIM: int = 32
HEAD_HIDDEN: int = 64
#: Flat head being replaced: ``nn.Linear(128, 2210)``.
FLAT_HEAD_PARAMS: int = DEFAULT_DIM * VOCAB_SIZE + VOCAB_SIZE  # 285,090
#: §3.3 hard ceiling — the build aborts if either assertion fails.
OUTPUT_STACK_PARAM_CEILING: int = 45_000
OUTPUT_STACK_PARAM_RATIO: float = 0.25
#: §3.3 expected total, used as a build-time self-check.
EXPECTED_OUTPUT_STACK_PARAMS: int = 28_240

#: §3.6.1 — ``unknown`` pitch type is a data-coverage artifact, never emitted.
UNKNOWN_TYPE_IDX: int = NUM_PITCH_TYPES - 1  # 16
#: §3.6.2/§3.6.3 minimum training support for a field value to be sampleable.
MIN_SUPPORT: int = 10

#: Spec freeze provenance, stamped into every audit JSON (§8.2).
SPEC_PATH: str = "docs/pitchgpt_sim_engine/PITCHGPT_V2_SPEC.md"
SPEC_FREEZE_SHA: str = "b61e05b3729aca2fa4609fbdadf4e533c1cd814f"
SPEC_BODY_SHA256: str = (
    "b19b54d96b496c5ffbff3f9af3070c180609003ec3d5d0aae71bfa84bb8d6d5b"
)


def token_from_fields(
    t: torch.Tensor | np.ndarray, z: torch.Tensor | np.ndarray, v: torch.Tensor | np.ndarray
):
    """``token_id = t * (26*5) + z * 5 + v`` — the v2 arithmetic identity."""
    return t * (NUM_ZONES * NUM_VELO_BUCKETS) + z * NUM_VELO_BUCKETS + v


def fields_from_token(tok: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of :func:`token_from_fields` (clamped for PAD/BOS safety)."""
    safe = tok.clamp(0, VOCAB_SIZE - 1)
    v = safe % NUM_VELO_BUCKETS
    rest = torch.div(safe, NUM_VELO_BUCKETS, rounding_mode="floor")
    z = rest % NUM_ZONES
    t = torch.div(rest, NUM_ZONES, rounding_mode="floor")
    return t, z, v


# ── Support tables (§3.6) ────────────────────────────────────────────────────


@dataclass
class SupportCounts:
    """Training-split field-support counts backing the §3.6 sampling masks."""

    type_zone: np.ndarray        # (17, 26)
    type_zone_velo: np.ndarray   # (17, 26, 5)
    zone: np.ndarray             # (26,)     — zone fallback (drop type)
    type_velo: np.ndarray        # (17, 5)   — velo fallback (drop zone)
    velo: np.ndarray             # (5,)

    @classmethod
    def from_tokens(cls, tokens: np.ndarray) -> "SupportCounts":
        tok = np.asarray(tokens).astype(np.int64)
        tok = tok[(tok >= 0) & (tok < VOCAB_SIZE)]
        v = tok % NUM_VELO_BUCKETS
        rest = tok // NUM_VELO_BUCKETS
        z = rest % NUM_ZONES
        t = rest // NUM_ZONES
        tz = np.zeros((NUM_PITCH_TYPES, NUM_ZONES), dtype=np.int64)
        np.add.at(tz, (t, z), 1)
        tzv = np.zeros((NUM_PITCH_TYPES, NUM_ZONES, NUM_VELO_BUCKETS), dtype=np.int64)
        np.add.at(tzv, (t, z, v), 1)
        return cls(
            type_zone=tz,
            type_zone_velo=tzv,
            zone=tz.sum(axis=0),
            type_velo=tzv.sum(axis=1),
            velo=tzv.sum(axis=(0, 1)),
        )

    def to_dict(self) -> dict:
        return {
            "type_zone": self.type_zone,
            "type_zone_velo": self.type_zone_velo,
            "zone": self.zone,
            "type_velo": self.type_velo,
            "velo": self.velo,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SupportCounts":
        return cls(
            type_zone=np.asarray(d["type_zone"]),
            type_zone_velo=np.asarray(d["type_zone_velo"]),
            zone=np.asarray(d["zone"]),
            type_velo=np.asarray(d["type_velo"]),
            velo=np.asarray(d["velo"]),
        )

    def sha256(self) -> str:
        import hashlib

        h = hashlib.sha256()
        for k in ("type_zone", "type_zone_velo", "zone", "type_velo", "velo"):
            h.update(np.ascontiguousarray(getattr(self, k)).astype(np.int64).tobytes())
        return h.hexdigest()


class MaskStats:
    """Mutable counters for §3.6.4 fallback / degenerate events."""

    def __init__(self) -> None:
        self.zone_fallback = 0
        self.velo_fallback = 0
        self.zone_degenerate = 0
        self.velo_degenerate = 0

    def as_dict(self) -> dict:
        return {
            "mask_fallbacks": {
                "zone_dropped_type_condition": int(self.zone_fallback),
                "velo_dropped_zone_condition": int(self.velo_fallback),
            },
            "mask_degenerate": {
                "zone_uniform": int(self.zone_degenerate),
                "velo_uniform": int(self.velo_degenerate),
            },
        }


# ── Model ────────────────────────────────────────────────────────────────────


class FactorizedPitchGPT(nn.Module):
    """v2 backbone (§3.2 identical) + chain-rule factorized heads (§3.3).

    ``forward_hidden`` reproduces :meth:`PitchGPTModel.forward` up to (but not
    including) the flat output head, so the transformer stack is
    layer-for-layer comparable with ``models/pitchgpt_v2.pt``.
    """

    def __init__(
        self,
        vocab_size: int = TOTAL_VOCAB,
        d_model: int = DEFAULT_DIM,
        nhead: int = DEFAULT_HEADS,
        num_layers: int = DEFAULT_LAYERS,
        max_seq_len: int = DEFAULT_MAX_SEQ,
        dropout: float = 0.1,
        context_dim: int = CONTEXT_DIM,
        field_emb_dim: int = FIELD_EMB_DIM,
        head_hidden: int = HEAD_HIDDEN,
        factorized_input: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.context_dim = context_dim
        self.factorized_input = bool(factorized_input)

        # ── Input side (§3.4 primary: flat embedding) ────────────────────
        if self.factorized_input:
            # Ablation A-EMB: E_in_type + E_in_zone + E_in_velo (6,144 params
            # replacing 283,136).  PAD/BOS keep dedicated rows.
            self.in_type = nn.Embedding(NUM_PITCH_TYPES, d_model)
            self.in_zone = nn.Embedding(NUM_ZONES, d_model)
            self.in_velo = nn.Embedding(NUM_VELO_BUCKETS, d_model)
            self.in_special = nn.Embedding(2, d_model)  # PAD, BOS
            self.token_embedding = None
        else:
            self.token_embedding = nn.Embedding(
                vocab_size, d_model, padding_idx=PAD_TOKEN,
            )

        self.context_proj = nn.Linear(context_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── Output side: the factorisation (§3.3) ────────────────────────
        self.h_type = nn.Linear(d_model, NUM_PITCH_TYPES)
        self.e_type = nn.Embedding(NUM_PITCH_TYPES, field_emb_dim)
        self.h_zone = nn.Sequential(
            nn.Linear(d_model + field_emb_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, NUM_ZONES),
        )
        self.e_zone = nn.Embedding(NUM_ZONES, field_emb_dim)
        self.h_velo = nn.Sequential(
            nn.Linear(d_model + 2 * field_emb_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, NUM_VELO_BUCKETS),
        )

        self._init_weights()
        self.support: SupportCounts | None = None
        self._assert_param_budget()

    # ── §3.3 build-time assertions ───────────────────────────────────────

    def output_stack_parameters(self) -> int:
        mods = [self.h_type, self.e_type, self.h_zone, self.e_zone, self.h_velo]
        return int(sum(p.numel() for m in mods for p in m.parameters()))

    def _assert_param_budget(self) -> None:
        n = self.output_stack_parameters()
        if not n < OUTPUT_STACK_PARAM_CEILING:
            raise AssertionError(
                f"§3.3 build assertion failed: output stack has {n:,} parameters, "
                f"ceiling is {OUTPUT_STACK_PARAM_CEILING:,}."
            )
        if not n < OUTPUT_STACK_PARAM_RATIO * FLAT_HEAD_PARAMS:
            raise AssertionError(
                f"§3.3 build assertion failed: output stack has {n:,} parameters, "
                f"must be < {OUTPUT_STACK_PARAM_RATIO} × {FLAT_HEAD_PARAMS:,} = "
                f"{OUTPUT_STACK_PARAM_RATIO * FLAT_HEAD_PARAMS:,.0f}."
            )

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Backbone ─────────────────────────────────────────────────────────

    def _embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.factorized_input:
            return self.token_embedding(tokens)
        t, z, v = fields_from_token(tokens)
        emb = self.in_type(t) + self.in_zone(z) + self.in_velo(v)
        is_pad = tokens == PAD_TOKEN
        is_bos = tokens == PAD_TOKEN + 1
        if is_pad.any():
            emb = torch.where(
                is_pad.unsqueeze(-1), self.in_special.weight[0].expand_as(emb), emb,
            )
        if is_bos.any():
            emb = torch.where(
                is_bos.unsqueeze(-1), self.in_special.weight[1].expand_as(emb), emb,
            )
        return emb

    def forward_hidden(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pre-output-head hidden state, ``(B, S, d_model)``."""
        B, S = tokens.shape
        device = tokens.device
        tok_emb = self._embed_tokens(tokens)
        ctx_emb = self.context_proj(context)
        pos_ids = torch.arange(S, device=device).unsqueeze(0)
        pos_ids = pos_ids.clamp(max=self.max_seq_len - 1)
        pos_emb = self.pos_embedding(pos_ids)
        x = tok_emb + ctx_emb + pos_emb
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(S, device=device)
        padding_mask = tokens == PAD_TOKEN
        return self.transformer(x, mask=mask, src_key_padding_mask=padding_mask)

    # ── Heads ────────────────────────────────────────────────────────────

    def type_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.h_type(hidden)

    def zone_logits(self, hidden: torch.Tensor, type_idx: torch.Tensor) -> torch.Tensor:
        return self.h_zone(torch.cat([hidden, self.e_type(type_idx)], dim=-1))

    def velo_logits(
        self, hidden: torch.Tensor, type_idx: torch.Tensor, zone_idx: torch.Tensor
    ) -> torch.Tensor:
        return self.h_velo(
            torch.cat([hidden, self.e_type(type_idx), self.e_zone(zone_idx)], dim=-1)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        context: torch.Tensor,
        target_type: torch.Tensor | None = None,
        target_zone: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Teacher-forced forward.

        ``target_type`` / ``target_zone`` are the ground-truth conditioning
        fields (§4.2 — teacher forcing on the fields as well as the sequence).
        Returns ``(hidden, type_logits, zone_logits, velo_logits)``.
        """
        hidden = self.forward_hidden(tokens, context, mask=mask)
        lt = self.type_logits(hidden)
        if target_type is None:
            return hidden, lt, None, None
        lz = self.zone_logits(hidden, target_type)
        if target_zone is None:
            return hidden, lt, lz, None
        lv = self.velo_logits(hidden, target_type, target_zone)
        return hidden, lt, lz, lv

    # ── Chain-rule reconstruction onto the 2,210 token space (§6.2, §8.3.2) ──

    @torch.no_grad()
    def full_token_probs(
        self,
        hidden: torch.Tensor,
        temps: "HeadTemperatures | None" = None,
    ) -> torch.Tensor:
        """Reconstruct ``p(token)`` over all 2,210 tokens from the chain rule.

        ``hidden`` is ``(N, d_model)``.  Returns ``(N, 2210)`` in the v2 token
        ordering.  Memory scales as ``N × 2210``; callers chunk large N.
        """
        n = hidden.shape[0]
        dev = hidden.device
        t_t, t_z, t_v = (1.0, 1.0, 1.0) if temps is None else temps.as_tuple()[:3]

        p_type = F.softmax(self.type_logits(hidden) / t_t, dim=-1)  # (N, 17)

        all_types = torch.arange(NUM_PITCH_TYPES, device=dev)
        h_rep = hidden.unsqueeze(1).expand(n, NUM_PITCH_TYPES, hidden.shape[-1])
        e_t = self.e_type(all_types).unsqueeze(0).expand(n, -1, -1)
        p_zone = F.softmax(
            self.h_zone(torch.cat([h_rep, e_t], dim=-1)) / t_z, dim=-1,
        )  # (N, 17, 26)

        all_zones = torch.arange(NUM_ZONES, device=dev)
        h_rep2 = hidden.view(n, 1, 1, -1).expand(n, NUM_PITCH_TYPES, NUM_ZONES, hidden.shape[-1])
        e_t2 = self.e_type(all_types).view(1, NUM_PITCH_TYPES, 1, -1).expand(
            n, NUM_PITCH_TYPES, NUM_ZONES, -1,
        )
        e_z2 = self.e_zone(all_zones).view(1, 1, NUM_ZONES, -1).expand(
            n, NUM_PITCH_TYPES, NUM_ZONES, -1,
        )
        p_velo = F.softmax(
            self.h_velo(torch.cat([h_rep2, e_t2, e_z2], dim=-1)) / t_v, dim=-1,
        )  # (N, 17, 26, 5)

        joint = p_type.view(n, NUM_PITCH_TYPES, 1, 1) * p_zone.unsqueeze(-1) * p_velo
        return joint.reshape(n, VOCAB_SIZE)

    # ── Sampling with §3.6 per-field masks ───────────────────────────────

    def _mask_from_support(
        self,
        support_row: np.ndarray,
        fallback_row: np.ndarray | None,
        stats_fallback_attr: str | None,
        stats_degenerate_attr: str | None,
        stats: MaskStats | None,
    ) -> np.ndarray:
        """Return a boolean legality mask for one field, applying §3.6.4.

        Reference (row-at-a-time) implementation.  The hot path uses the
        precomputed tables built by :meth:`_build_mask_tables`, which this
        method defines the semantics of and is tested against.
        """
        legal = support_row >= MIN_SUPPORT
        if legal.any():
            return legal
        if fallback_row is not None:
            legal_fb = fallback_row >= MIN_SUPPORT
            if legal_fb.any():
                if stats is not None and stats_fallback_attr:
                    setattr(
                        stats, stats_fallback_attr,
                        getattr(stats, stats_fallback_attr) + 1,
                    )
                return legal_fb
        if stats is not None and stats_degenerate_attr:
            setattr(
                stats, stats_degenerate_attr,
                getattr(stats, stats_degenerate_attr) + 1,
            )
        return np.ones_like(legal, dtype=bool)

    def _build_mask_tables(self) -> None:
        """Precompute §3.6 legality masks and their fallback provenance.

        ``*_source`` is 0 = conditional support, 1 = fallback (one conditioning
        level dropped, §3.6.4), 2 = degenerate (uniform over the legal range).
        """
        s = self.support
        zone_mask = np.zeros((NUM_PITCH_TYPES, NUM_ZONES), dtype=bool)
        zone_source = np.zeros(NUM_PITCH_TYPES, dtype=np.int8)
        for t in range(NUM_PITCH_TYPES):
            cond = s.type_zone[t] >= MIN_SUPPORT
            if cond.any():
                zone_mask[t], zone_source[t] = cond, 0
                continue
            fb = s.zone >= MIN_SUPPORT
            if fb.any():
                zone_mask[t], zone_source[t] = fb, 1
            else:
                zone_mask[t], zone_source[t] = True, 2

        velo_mask = np.zeros(
            (NUM_PITCH_TYPES, NUM_ZONES, NUM_VELO_BUCKETS), dtype=bool,
        )
        velo_source = np.zeros((NUM_PITCH_TYPES, NUM_ZONES), dtype=np.int8)
        for t in range(NUM_PITCH_TYPES):
            fb_t = s.type_velo[t] >= MIN_SUPPORT
            for z in range(NUM_ZONES):
                cond = s.type_zone_velo[t, z] >= MIN_SUPPORT
                if cond.any():
                    velo_mask[t, z], velo_source[t, z] = cond, 0
                elif fb_t.any():
                    velo_mask[t, z], velo_source[t, z] = fb_t, 1
                else:
                    velo_mask[t, z], velo_source[t, z] = True, 2

        self._zone_mask = zone_mask
        self._zone_source = zone_source
        self._velo_mask = velo_mask
        self._velo_source = velo_source
        self._mask_cache = {}

    def _mask_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Device-resident §3.6 legality tables (built once, cached)."""
        key = str(device)
        cache = getattr(self, "_mask_cache", None)
        if cache is None:
            cache = {}
            self._mask_cache = cache
        if key not in cache:
            cache[key] = (
                torch.from_numpy(self._zone_mask).to(device),
                torch.from_numpy(self._velo_mask).to(device),
            )
        return cache[key]

    @torch.no_grad()
    def sample_tokens(
        self,
        hidden: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        uniforms: torch.Tensor | None = None,
        temps: "HeadTemperatures | None" = None,
        stats: MaskStats | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequential sub-token decode with the §3.6 masks.

        ``hidden`` is ``(N, d_model)``.  Returns
        ``(token, type_idx, zone_idx, velo_idx)`` each ``(N,)``.

        ``uniforms`` ``(N, 3)`` selects inverse-CDF sampling from a caller-owned
        random stream (the §4.3.2 per-PA blocks), which makes a PA's trajectory
        independent of the batch it is rolled in.  Without it the draw comes
        from ``torch.multinomial`` and ``generator``.
        """
        if self.support is None:
            raise RuntimeError(
                "sample_tokens() requires the §3.6 support table; load a v3 "
                "checkpoint or call attach_support()."
            )
        dev = hidden.device
        t_t, t_z, t_v = (1.0, 1.0, 1.0) if temps is None else temps.as_tuple()[:3]
        zone_mask_t, velo_mask_t = self._mask_tensors(dev)

        def _draw(p: torch.Tensor, col: int) -> torch.Tensor:
            if uniforms is None:
                return torch.multinomial(p, 1, generator=generator).squeeze(1)
            cdf = p.cumsum(dim=-1)
            cdf = cdf / cdf[:, -1:].clamp_min(1e-12)
            u = uniforms[:, col].to(p.device)
            return (u.unsqueeze(-1) > cdf).sum(dim=-1).clamp_(max=p.shape[-1] - 1)

        # 1. pitch type — index 16 (unknown) is always masked (§3.6.1).
        lt = self.type_logits(hidden) / t_t
        lt[:, UNKNOWN_TYPE_IDX] = -float("inf")
        t_idx = _draw(F.softmax(lt, dim=-1), 0)

        # 2. zone | type, support >= 10 (§3.6.2), fallback drops the type
        #    condition (§3.6.4).
        lz = self.zone_logits(hidden, t_idx) / t_z
        if stats is not None:
            src = self._zone_source[t_idx.detach().cpu().numpy()]
            stats.zone_fallback += int((src == 1).sum())
            stats.zone_degenerate += int((src == 2).sum())
        lz = lz.masked_fill(~zone_mask_t[t_idx], -float("inf"))
        z_idx = _draw(F.softmax(lz, dim=-1), 1)

        # 3. velo | type, zone (§3.6.3), fallback drops the zone condition.
        lv = self.velo_logits(hidden, t_idx, z_idx) / t_v
        if stats is not None:
            src = self._velo_source[
                t_idx.detach().cpu().numpy(), z_idx.detach().cpu().numpy()
            ]
            stats.velo_fallback += int((src == 1).sum())
            stats.velo_degenerate += int((src == 2).sum())
        lv = lv.masked_fill(~velo_mask_t[t_idx, z_idx], -float("inf"))
        v_idx = _draw(F.softmax(lv, dim=-1), 2)

        return token_from_fields(t_idx, z_idx, v_idx), t_idx, z_idx, v_idx

    @torch.no_grad()
    def greedy_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        """Sequential-greedy decode (no masks) — used by the §8.3.2 identity test."""
        t_idx = self.type_logits(hidden).argmax(dim=-1)
        z_idx = self.zone_logits(hidden, t_idx).argmax(dim=-1)
        v_idx = self.velo_logits(hidden, t_idx, z_idx).argmax(dim=-1)
        return token_from_fields(t_idx, z_idx, v_idx)

    def attach_support(self, support: SupportCounts) -> None:
        self.support = support
        self._build_mask_tables()


# ── Per-head temperatures (§4.4) ─────────────────────────────────────────────


@dataclass
class HeadTemperatures:
    """One scalar per head.  Vectors/matrices are banned by §0.2."""

    type_T: float = 1.0
    zone_T: float = 1.0
    velo_T: float = 1.0
    outcome_T: float = 1.0

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.type_T, self.zone_T, self.velo_T, self.outcome_T)

    def to_dict(self) -> dict:
        return {
            "T_type": float(self.type_T),
            "T_zone": float(self.zone_T),
            "T_velo": float(self.velo_T),
            "T_outcome": float(self.outcome_T),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HeadTemperatures":
        return cls(
            type_T=float(d["T_type"]),
            zone_T=float(d["T_zone"]),
            velo_T=float(d["T_velo"]),
            outcome_T=float(d["T_outcome"]),
        )


class CalibrationProvenanceError(RuntimeError):
    """Raised when a calibration sidecar violates the §7.5 provenance guard."""


#: Cohorts a calibration artifact may never declare (§7.5, K5 second clause).
FORBIDDEN_FIT_COHORTS: frozenset[int] = frozenset({2025, 2026})
REQUIRED_PROVENANCE_KEYS: tuple[str, ...] = (
    "fit_cohort_season",
    "fit_seed",
    "fit_n_pas",
    "produced_by",
)


def validate_calibration_provenance(payload: dict) -> dict:
    """Structural §7.5 guard — a violating run aborts rather than warns.

    Requires the four provenance keys the guard already enforces and refuses
    any artifact declaring ``fit_cohort_season`` in {2025, 2026}.
    """
    missing = [k for k in REQUIRED_PROVENANCE_KEYS if k not in payload]
    if missing:
        raise CalibrationProvenanceError(
            f"v3 calibration sidecar is missing required provenance keys "
            f"{missing}; schema is {list(REQUIRED_PROVENANCE_KEYS)} (§4.4)."
        )
    season = payload["fit_cohort_season"]
    try:
        season_i = int(season)
    except (TypeError, ValueError) as exc:
        raise CalibrationProvenanceError(
            f"fit_cohort_season={season!r} is not an integer season."
        ) from exc
    if season_i in FORBIDDEN_FIT_COHORTS:
        raise CalibrationProvenanceError(
            f"§7.5 provenance guard: calibration artifact declares "
            f"fit_cohort_season={season_i}, which is a gate-evaluation cohort "
            f"(2025 budgeted tier / 2026 lockbox). Refusing to load."
        )
    if season_i != 2023:
        raise CalibrationProvenanceError(
            f"§4.4 requires fit_cohort_season == 2023 (the pitcher-disjoint "
            f"slice); artifact declares {season_i}."
        )
    return payload


def load_calibration(path: Path | str) -> tuple[HeadTemperatures, dict]:
    """Load + provenance-check ``models/calibration_pitchgpt_v3.json``."""
    import json

    p = Path(path)
    with p.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    validate_calibration_provenance(payload)
    return HeadTemperatures.from_dict(payload["temperatures"]), payload


# ── Outcome head on the v3 backbone (§3.5) ───────────────────────────────────


class V3OutcomeHead(nn.Module):
    """[FIXED] unchanged from the A1 winner: ``211 → 128 → 64 → 7``.

    One deliberate change vs A1 lives in the *training* script, not here:
    **no inverse-frequency class weighting** (§3.5).
    """

    def __init__(self, d_model: int = DEFAULT_DIM, context_dim: int = CONTEXT_DIM,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.in_dim = (
            d_model + context_dim + NUM_PITCH_TYPES + NUM_ZONES + NUM_VELO_BUCKETS
        )
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, NUM_OUTCOME_CLASSES),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        hidden: torch.Tensor,
        context: torch.Tensor,
        type_idx: torch.Tensor,
        zone_idx: torch.Tensor,
        velo_idx: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat(
            [
                hidden,
                context,
                F.one_hot(type_idx, NUM_PITCH_TYPES).float(),
                F.one_hot(zone_idx, NUM_ZONES).float(),
                F.one_hot(velo_idx, NUM_VELO_BUCKETS).float(),
            ],
            dim=-1,
        )
        return self.net(x)


# ── Checkpoint I/O (write-once; §0.4, §8.1) ──────────────────────────────────


def save_v3_checkpoint(
    path: Path | str,
    model: FactorizedPitchGPT,
    *,
    meta: dict,
    allow_overwrite: bool = False,
) -> str:
    """Write a v3 checkpoint.  Refuses to clobber an existing file (§0.4)."""
    import hashlib

    p = Path(path)
    if p.exists() and not allow_overwrite:
        raise FileExistsError(
            f"§0.4 forbids overwriting artifacts: {p} already exists."
        )
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "config": {
            "d_model": model.d_model,
            "max_seq_len": model.max_seq_len,
            "context_dim": model.context_dim,
            "factorized_input": model.factorized_input,
            "field_emb_dim": FIELD_EMB_DIM,
            "head_hidden": HEAD_HIDDEN,
            "output_stack_params": model.output_stack_parameters(),
        },
        "support_counts": (
            model.support.to_dict() if model.support is not None else None
        ),
        "support_counts_sha256": (
            model.support.sha256() if model.support is not None else None
        ),
        "spec_path": SPEC_PATH,
        "spec_freeze_sha": SPEC_FREEZE_SHA,
        "spec_body_sha256": SPEC_BODY_SHA256,
        "meta": meta,
    }
    torch.save(payload, p)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_v3_checkpoint(
    path: Path | str, device: torch.device | str | None = None
) -> tuple[FactorizedPitchGPT, dict]:
    ck = torch.load(Path(path), map_location=device or "cpu", weights_only=False)
    cfg = ck["config"]
    model = FactorizedPitchGPT(
        d_model=cfg["d_model"],
        max_seq_len=cfg["max_seq_len"],
        context_dim=cfg["context_dim"],
        factorized_input=cfg.get("factorized_input", False),
        field_emb_dim=cfg.get("field_emb_dim", FIELD_EMB_DIM),
        head_hidden=cfg.get("head_hidden", HEAD_HIDDEN),
    )
    model.load_state_dict(ck["model_state_dict"])
    if ck.get("support_counts") is not None:
        model.attach_support(SupportCounts.from_dict(ck["support_counts"]))
    if device is not None:
        model.to(device)
    model.eval()
    return model, ck
