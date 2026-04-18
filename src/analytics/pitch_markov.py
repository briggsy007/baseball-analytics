"""
Markov and heuristic baselines for the PitchGPT transformer (spec
tickets #4 and #5).

These modules deliberately contain only the minimum machinery needed to
compute comparable *per-token negative log-likelihood* on the same
:class:`PitchSequenceDataset` used by PitchGPT and the LSTM baseline, so
that the resulting perplexity numbers are apples-to-apples.

The Markov chain conditions the next-pitch token only on **pitch type**
(the single most-predictive categorical feature of a pitch).  Zone and
velocity are factored as the empirical conditional distribution
``P(zone, velo_bucket | pitch_type)`` estimated on the training set.
This keeps the vocabulary identical to PitchGPT (``VOCAB_SIZE = 2210``)
so perplexity numbers compare directly.

Count-state (0-0 .. 3-2) enters the conditioning tuple as well, because
pitcher behavior is strongly count-dependent — a Markov baseline that
ignores count is easy to beat and would be an unconvincing foil.  The
count bucket is looked up straight out of the
:class:`PitchSequenceDataset` context tensor.

Three classes exposed:

* :class:`MarkovChainOrder1` — conditions on ``(last_pitch_type, count)``.
* :class:`MarkovChainOrder2` — conditions on
  ``(prev2_pitch_type, prev1_pitch_type, count)``.
* :class:`HeuristicBaseline`  — a fixed-by-family prior over pitch types
  (fastball / breaking / offspeed) refined with the league-wide empirical
  rate observed during ``fit``.

All three expose the same API:

    ``.fit(sequences) -> self``
    ``.score_sequences(sequences) -> torch.Tensor``  (flat per-token NLL)
    ``.calculate_perplexity(sequences) -> float``

where ``sequences`` is an iterable of ``(tokens, context, target)``
tuples — i.e. the output of :class:`PitchSequenceDataset.__getitem__`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from typing import Iterator

import numpy as np
import torch

from src.analytics.pitchgpt import (
    NUM_COUNT_STATES,
    NUM_PITCH_TYPES,
    NUM_VELO_BUCKETS,
    NUM_ZONES,
    PAD_TOKEN,
    VOCAB_SIZE,
)

logger = logging.getLogger(__name__)


# ─── Pitch-type groupings (spec ticket #5) ──────────────────────────────────
# Index into ``PITCH_TYPE_MAP`` (see pitchgpt.py).  The unknown slot (16)
# is treated as "other" — given zero prior mass but smoothed so its NLL is
# finite.
_FASTBALL_IDX = [0, 1, 2]              # FF, SI, FC
_BREAKING_IDX = [3, 4, 7, 12, 11]      # SL, CU, KC, ST, SV
_OFFSPEED_IDX = [5, 6, 14, 13]         # CH, FS, FO, SC
# KN (8), EP (9), CS (10), FA (15), UN (16) land in "other".

_FASTBALL_SHARE = 0.50
_BREAKING_SHARE = 0.30
_OFFSPEED_SHARE = 0.20

_DEFAULT_ALPHA = 0.1  # Laplace smoothing on all transition counts.


# ─── Helpers that decode a token into its (pitch_type, zone, velo) parts ────
# Token layout (from PitchTokenizer):
#   token = pt_idx * (NUM_ZONES * NUM_VELO_BUCKETS)
#           + zone * NUM_VELO_BUCKETS + velo
_ZV = NUM_ZONES * NUM_VELO_BUCKETS


def _token_to_pt(token: int) -> int:
    """Extract pitch_type index from a composite pitch token."""
    if token < 0 or token >= VOCAB_SIZE:
        return NUM_PITCH_TYPES - 1  # unknown
    return token // _ZV


def _count_from_context(context_row: torch.Tensor) -> int:
    """Recover the count-state index (0..NUM_COUNT_STATES-1) from the
    one-hot context tensor.

    The context layout in ``PitchTokenizer.context_to_tensor`` is a
    concatenation of six one-hot blocks; count is the first block,
    occupying indices ``[0, NUM_COUNT_STATES)``.
    """
    # ``context_row`` may be a (CONTEXT_DIM,) tensor — take argmax over
    # the count block.
    block = context_row[:NUM_COUNT_STATES]
    # If the block is all zeros (shouldn't happen in practice — one-hot
    # always has one 1), fall back to 0.
    if block.sum().item() == 0:
        return 0
    return int(torch.argmax(block).item())


# ═════════════════════════════════════════════════════════════════════════════
# Shared emission table: P(zone, velo_bucket | pitch_type)
# ═════════════════════════════════════════════════════════════════════════════

def _build_emission_table(
    sequences: Iterable[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    alpha: float,
) -> np.ndarray:
    """Return a (NUM_PITCH_TYPES, _ZV) row-stochastic emission matrix.

    ``sequences`` is the same iterable the Markov chain is trained on —
    we walk every non-PAD token in both the input and target streams to
    maximize coverage.
    """
    counts = np.full((NUM_PITCH_TYPES, _ZV), alpha, dtype=np.float64)
    for tokens, _ctx, target in sequences:
        for seq in (tokens, target):
            for tok in seq.tolist():
                if tok == PAD_TOKEN or tok < 0 or tok >= VOCAB_SIZE:
                    continue
                pt = tok // _ZV
                zv = tok % _ZV
                counts[pt, zv] += 1.0
    row_sum = counts.sum(axis=1, keepdims=True)
    return counts / np.clip(row_sum, 1e-12, None)


# ═════════════════════════════════════════════════════════════════════════════
# Base class — all three models share the perplexity / scoring machinery
# ═════════════════════════════════════════════════════════════════════════════

class _BaseChain:
    """Abstract base that computes token-level NLL from a
    ``P(next_pt | state)`` distribution and a factored emission
    ``P(zv | pt)``.

    Subclasses override :meth:`_next_pt_logprobs` to provide the
    state-conditional pitch-type log-probs for every position in a
    sequence.
    """

    def __init__(self, alpha: float = _DEFAULT_ALPHA) -> None:
        self.alpha = alpha
        # P(pitch_type) marginal on the training set — used as the
        # unconditional backoff distribution at the start of every
        # sequence.
        self._pt_marginal: np.ndarray = np.full(
            NUM_PITCH_TYPES, 1.0 / NUM_PITCH_TYPES, dtype=np.float64,
        )
        # P(zone, velo_bucket | pitch_type).
        self._emission: np.ndarray = np.full(
            (NUM_PITCH_TYPES, _ZV), 1.0 / _ZV, dtype=np.float64,
        )
        self._fitted = False

    # ── Public API ──────────────────────────────────────────────────────

    def fit(self, sequences) -> "_BaseChain":  # type: ignore[override]
        """Override in subclasses.  Must set ``self._fitted = True``."""
        raise NotImplementedError

    def score_sequences(self, sequences) -> torch.Tensor:
        """Return a flat tensor of per-token NLLs (one entry per non-PAD
        target token across the whole iterable).
        """
        if not self._fitted:
            raise RuntimeError("score_sequences() called before fit().")

        nlls: list[float] = []
        for tokens, ctx, target in sequences:
            pt_logp = self._next_pt_logprobs(tokens, ctx)   # (S, NUM_PT)
            for i, tgt_tok in enumerate(target.tolist()):
                if tgt_tok == PAD_TOKEN or tgt_tok < 0 or tgt_tok >= VOCAB_SIZE:
                    continue
                tgt_pt = tgt_tok // _ZV
                tgt_zv = tgt_tok % _ZV
                nll_pt = -float(pt_logp[i, tgt_pt])
                nll_zv = -math.log(max(self._emission[tgt_pt, tgt_zv], 1e-12))
                nlls.append(nll_pt + nll_zv)
        return torch.tensor(nlls, dtype=torch.float64)

    def calculate_perplexity(self, sequences) -> float:
        """Exp of mean per-token NLL — directly comparable to PitchGPT."""
        # We need to iterate twice if ``sequences`` is a one-shot
        # iterator.  Materialize a list if so.
        seq_list = _materialize(sequences)
        nlls = self.score_sequences(seq_list)
        if nlls.numel() == 0:
            return float("inf")
        mean_nll = float(nlls.mean().item())
        return math.exp(min(mean_nll, 20))

    # ── Hooks ───────────────────────────────────────────────────────────

    def _next_pt_logprobs(
        self,
        tokens: torch.Tensor,   # (S,)
        context: torch.Tensor,  # (S, CONTEXT_DIM)
    ) -> np.ndarray:
        """Return (S, NUM_PITCH_TYPES) matrix of log P(next_pt | state)
        for each position ``i`` in the sequence, where "next" is the
        pitch whose token is ``target[i]``."""
        raise NotImplementedError


def _materialize(sequences) -> list:
    """Materialize a one-shot iterable into a list if needed.

    We detect a torch ``Dataset`` (has ``__len__`` + ``__getitem__``)
    vs. a plain iterator.  Datasets can be iterated multiple times, so
    we return them as-is; everything else gets listed.
    """
    if hasattr(sequences, "__len__") and hasattr(sequences, "__getitem__"):
        return sequences
    if isinstance(sequences, list):
        return sequences
    return list(sequences)


# ═════════════════════════════════════════════════════════════════════════════
# Markov-1
# ═════════════════════════════════════════════════════════════════════════════

class MarkovChainOrder1(_BaseChain):
    """P(next_pt | last_pt, count_state).

    The conditioning state is 2-D, so we represent the transition table
    as a dense array of shape ``(NUM_PITCH_TYPES, NUM_COUNT_STATES,
    NUM_PITCH_TYPES)``.  Laplace smoothing ``alpha`` is added to every
    cell so unseen transitions do not produce infinite NLL.
    """

    def __init__(self, alpha: float = _DEFAULT_ALPHA) -> None:
        super().__init__(alpha=alpha)
        self._trans: np.ndarray | None = None

    def fit(self, sequences) -> "MarkovChainOrder1":
        seq_list = _materialize(sequences)
        counts = np.full(
            (NUM_PITCH_TYPES, NUM_COUNT_STATES, NUM_PITCH_TYPES),
            self.alpha, dtype=np.float64,
        )
        marg = np.full(NUM_PITCH_TYPES, self.alpha, dtype=np.float64)

        for tokens, ctx, target in seq_list:
            toks = tokens.tolist()
            tgts = target.tolist()
            for i, tgt_tok in enumerate(tgts):
                if tgt_tok == PAD_TOKEN:
                    continue
                prev_tok = toks[i]
                if prev_tok == PAD_TOKEN:
                    continue
                prev_pt = _token_to_pt(prev_tok)
                tgt_pt = _token_to_pt(tgt_tok)
                count_s = _count_from_context(ctx[i])
                counts[prev_pt, count_s, tgt_pt] += 1.0
                marg[tgt_pt] += 1.0

        # Normalize last axis.
        row_sum = counts.sum(axis=-1, keepdims=True)
        self._trans = counts / np.clip(row_sum, 1e-12, None)
        self._pt_marginal = marg / max(marg.sum(), 1e-12)
        self._emission = _build_emission_table(seq_list, self.alpha)
        self._fitted = True
        return self

    def _next_pt_logprobs(self, tokens, context):  # type: ignore[override]
        assert self._trans is not None
        S = tokens.size(0)
        out = np.zeros((S, NUM_PITCH_TYPES), dtype=np.float64)
        toks = tokens.tolist()
        for i in range(S):
            prev_tok = toks[i]
            if prev_tok == PAD_TOKEN:
                # Back off to unconditional — the eval loop will skip this
                # row anyway when the target is PAD, but be safe.
                probs = self._pt_marginal
            else:
                prev_pt = _token_to_pt(prev_tok)
                count_s = _count_from_context(context[i])
                probs = self._trans[prev_pt, count_s]
            out[i] = np.log(np.clip(probs, 1e-12, None))
        return out


# ═════════════════════════════════════════════════════════════════════════════
# Markov-2
# ═════════════════════════════════════════════════════════════════════════════

class MarkovChainOrder2(_BaseChain):
    """P(next_pt | prev2_pt, prev1_pt, count_state).

    For position ``i < 2`` in a sequence we *fall back* to a 1st-order
    chain (also fit during ``fit``).  Position ``i == 0`` further falls
    back to the unconditional marginal.
    """

    def __init__(self, alpha: float = _DEFAULT_ALPHA) -> None:
        super().__init__(alpha=alpha)
        self._trans2: np.ndarray | None = None     # (PT, PT, CS, PT)
        self._trans1: np.ndarray | None = None     # (PT, CS, PT)

    def fit(self, sequences) -> "MarkovChainOrder2":
        seq_list = _materialize(sequences)

        counts2 = np.full(
            (NUM_PITCH_TYPES, NUM_PITCH_TYPES, NUM_COUNT_STATES, NUM_PITCH_TYPES),
            self.alpha, dtype=np.float64,
        )
        counts1 = np.full(
            (NUM_PITCH_TYPES, NUM_COUNT_STATES, NUM_PITCH_TYPES),
            self.alpha, dtype=np.float64,
        )
        marg = np.full(NUM_PITCH_TYPES, self.alpha, dtype=np.float64)

        for tokens, ctx, target in seq_list:
            toks = tokens.tolist()
            tgts = target.tolist()
            for i, tgt_tok in enumerate(tgts):
                if tgt_tok == PAD_TOKEN:
                    continue
                prev_tok = toks[i]
                if prev_tok == PAD_TOKEN:
                    continue
                prev_pt = _token_to_pt(prev_tok)
                tgt_pt = _token_to_pt(tgt_tok)
                count_s = _count_from_context(ctx[i])
                counts1[prev_pt, count_s, tgt_pt] += 1.0
                marg[tgt_pt] += 1.0
                if i >= 1:
                    prev2_tok = toks[i - 1]
                    if prev2_tok != PAD_TOKEN:
                        prev2_pt = _token_to_pt(prev2_tok)
                        counts2[prev2_pt, prev_pt, count_s, tgt_pt] += 1.0

        self._trans2 = counts2 / np.clip(
            counts2.sum(axis=-1, keepdims=True), 1e-12, None,
        )
        self._trans1 = counts1 / np.clip(
            counts1.sum(axis=-1, keepdims=True), 1e-12, None,
        )
        self._pt_marginal = marg / max(marg.sum(), 1e-12)
        self._emission = _build_emission_table(seq_list, self.alpha)
        self._fitted = True
        return self

    def _next_pt_logprobs(self, tokens, context):  # type: ignore[override]
        assert self._trans2 is not None and self._trans1 is not None
        S = tokens.size(0)
        out = np.zeros((S, NUM_PITCH_TYPES), dtype=np.float64)
        toks = tokens.tolist()
        for i in range(S):
            prev_tok = toks[i]
            if prev_tok == PAD_TOKEN:
                probs = self._pt_marginal
            else:
                prev_pt = _token_to_pt(prev_tok)
                count_s = _count_from_context(context[i])
                if i >= 1 and toks[i - 1] != PAD_TOKEN:
                    prev2_pt = _token_to_pt(toks[i - 1])
                    probs = self._trans2[prev2_pt, prev_pt, count_s]
                else:
                    probs = self._trans1[prev_pt, count_s]
            out[i] = np.log(np.clip(probs, 1e-12, None))
        return out


# ═════════════════════════════════════════════════════════════════════════════
# Heuristic baseline (spec ticket #5)
# ═════════════════════════════════════════════════════════════════════════════

class HeuristicBaseline(_BaseChain):
    """Fixed-by-family prior over pitch types.

    * 50% fastball family (FF, SI, FC)
    * 30% breaking (SL, CU, KC, ST, SV)
    * 20% offspeed (CH, FS, FO, SC)
    * remainder (KN, EP, CS, FA, UN) shares the Laplace smoothing budget.

    ``fit`` additionally estimates the league-wide empirical pitch-type
    marginal on the training set; the final distribution used at
    inference time is the *mixture* ``0.5 * fixed + 0.5 * empirical``,
    which is stronger than either alone and matches the spec's intent of
    a "real" naive prior.
    """

    def __init__(self, alpha: float = _DEFAULT_ALPHA) -> None:
        super().__init__(alpha=alpha)
        self._fixed = self._fixed_prior()
        self._empirical: np.ndarray | None = None

    @staticmethod
    def _fixed_prior() -> np.ndarray:
        """Return the fixed 50/30/20 prior as a pt-level distribution."""
        p = np.zeros(NUM_PITCH_TYPES, dtype=np.float64)
        for idx in _FASTBALL_IDX:
            p[idx] = _FASTBALL_SHARE / len(_FASTBALL_IDX)
        for idx in _BREAKING_IDX:
            p[idx] = _BREAKING_SHARE / len(_BREAKING_IDX)
        for idx in _OFFSPEED_IDX:
            p[idx] = _OFFSPEED_SHARE / len(_OFFSPEED_IDX)
        # Smooth so no zero mass (unknown / rare types).
        residual = 1.0 - p.sum()
        if residual > 0:
            zero_idx = np.where(p == 0.0)[0]
            if len(zero_idx) > 0:
                p[zero_idx] = residual / len(zero_idx)
            else:
                p += residual / NUM_PITCH_TYPES
        p = p / p.sum()
        return p

    def fit(self, sequences) -> "HeuristicBaseline":
        seq_list = _materialize(sequences)
        # Empirical pitch-type marginal over all non-PAD tokens.
        counts = np.full(NUM_PITCH_TYPES, self.alpha, dtype=np.float64)
        for tokens, _ctx, target in seq_list:
            for seq in (tokens, target):
                for tok in seq.tolist():
                    if tok == PAD_TOKEN or tok < 0 or tok >= VOCAB_SIZE:
                        continue
                    counts[tok // _ZV] += 1.0
        self._empirical = counts / counts.sum()
        # Mixture: half fixed prior, half empirical.
        mix = 0.5 * self._fixed + 0.5 * self._empirical
        self._pt_marginal = mix / mix.sum()
        self._emission = _build_emission_table(seq_list, self.alpha)
        self._fitted = True
        return self

    def _next_pt_logprobs(self, tokens, context):  # type: ignore[override]
        # Stateless — every position uses the same marginal.
        S = tokens.size(0)
        logp = np.log(np.clip(self._pt_marginal, 1e-12, None))
        out = np.broadcast_to(logp, (S, NUM_PITCH_TYPES)).copy()
        return out
