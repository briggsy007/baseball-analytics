"""
Tests for :mod:`src.analytics.pitch_markov` — Markov-1, Markov-2 and
the heuristic baseline from spec tickets #4 and #5.

We test on synthetic sequences (never the production DB) so these tests
run fast and deterministically.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.analytics.pitchgpt import (
    CONTEXT_DIM,
    NUM_PITCH_TYPES,
    NUM_VELO_BUCKETS,
    NUM_ZONES,
    PAD_TOKEN,
    PitchTokenizer,
    VOCAB_SIZE,
)
from src.analytics.pitch_markov import (
    HeuristicBaseline,
    MarkovChainOrder1,
    MarkovChainOrder2,
)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

_ZV = NUM_ZONES * NUM_VELO_BUCKETS


def _pt_to_token(pt_idx: int, zone: int = 0, velo: int = 0) -> int:
    """Compose a valid pitch token from (pitch_type_idx, zone, velo)."""
    return pt_idx * _ZV + zone * NUM_VELO_BUCKETS + velo


def _context_row(count_state: int = 0) -> torch.Tensor:
    """Return a CONTEXT_DIM-length one-hot context row with the given
    count state set and every other block defaulting to index 0.
    """
    # Mirror PitchTokenizer.encode_context to keep this in sync.
    vec = torch.zeros(CONTEXT_DIM, dtype=torch.float32)
    # Count state block starts at offset 0.
    vec[count_state] = 1.0
    # Outs (0), runners (0), batter hand (0), inning (0), score_diff (0).
    # Use the known offset set from the tokenizer.
    from src.analytics.pitchgpt import (
        NUM_BATTER_HANDS,
        NUM_COUNT_STATES,
        NUM_INNING_BUCKETS,
        NUM_OUTS,
        NUM_RUNNER_STATES,
    )
    offs = [
        0,
        NUM_COUNT_STATES,
        NUM_COUNT_STATES + NUM_OUTS,
        NUM_COUNT_STATES + NUM_OUTS + NUM_RUNNER_STATES,
        NUM_COUNT_STATES + NUM_OUTS + NUM_RUNNER_STATES + NUM_BATTER_HANDS,
        NUM_COUNT_STATES + NUM_OUTS + NUM_RUNNER_STATES + NUM_BATTER_HANDS
        + NUM_INNING_BUCKETS,
    ]
    # Leave count to above; set default 0 for the rest.
    for o in offs[1:]:
        vec[o] = 1.0
    return vec


def _build_seq(pt_seq: list[int], count_state: int = 0) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,
]:
    """Build a (tokens, context, target) tuple for a pitch-type sequence."""
    tokens_full = [_pt_to_token(pt) for pt in pt_seq]
    tokens = torch.tensor(tokens_full[:-1], dtype=torch.long)
    target = torch.tensor(tokens_full[1:], dtype=torch.long)
    ctx = torch.stack([_context_row(count_state) for _ in range(len(tokens))])
    return tokens, ctx, target


# ═════════════════════════════════════════════════════════════════════════════
# Markov-1
# ═════════════════════════════════════════════════════════════════════════════


class TestMarkovChainOrder1:
    def test_markov1_perplexity_on_uniform_data(self):
        """Uniform random pitch sequence → perplexity in the ballpark of
        VOCAB_SIZE (specifically, between NUM_PITCH_TYPES * NUM_ZONES
        and VOCAB_SIZE — the Markov chain knows pitch type and zone is
        uniform per type).

        The ceiling is VOCAB_SIZE (2210); we just assert perplexity is
        strictly within an order of magnitude of it.
        """
        rng = np.random.default_rng(0)
        N_SEQ = 30
        SEQ_LEN = 40
        sequences = []
        for _ in range(N_SEQ):
            pts = rng.integers(0, NUM_PITCH_TYPES - 1, size=SEQ_LEN).tolist()
            sequences.append(_build_seq(pts))

        m = MarkovChainOrder1(alpha=0.1).fit(sequences)
        ppl = m.calculate_perplexity(sequences)

        # Sanity: strictly positive and finite.
        assert math.isfinite(ppl) and ppl > 1.0
        # Coarse bound: the Markov chain does about as well as the
        # unconditional + emission prior on uniform data.  We just want
        # to make sure we're roughly near the vocabulary size and not
        # accidentally at 1.0 or at 1e9.
        assert ppl < VOCAB_SIZE * 2, f"ppl {ppl} implausibly high"
        assert ppl > 10, f"ppl {ppl} implausibly low for uniform data"

    def test_markov1_better_than_random_on_structured_data(self):
        """A deterministic A→B→A→B pattern should give Markov-1 much
        lower perplexity than its own uniform-data perplexity."""
        pattern = [0, 3] * 25  # FF, SL, FF, SL, ...
        sequences = [_build_seq(pattern) for _ in range(10)]
        m = MarkovChainOrder1(alpha=0.01).fit(sequences)
        ppl = m.calculate_perplexity(sequences)
        # A→B→A is perfectly captured by 1st order — perplexity should
        # be very close to the emission entropy (since zone / velo are
        # fixed to 0 in the synthetic data).
        assert ppl < 5.0, f"ppl {ppl} too high for deterministic A-B pattern"


# ═════════════════════════════════════════════════════════════════════════════
# Markov-2
# ═════════════════════════════════════════════════════════════════════════════


class TestMarkovChainOrder2:
    def test_markov2_better_than_markov1_on_structured_data(self):
        """Synthetic data where the 1st-order chain is *ambiguous* but
        the 2nd-order chain is deterministic.

        Pattern:  (A, A) → B,  (A, B) → A,  (B, A) → A,  (B, B) → A
        In plain words: after seeing ``A`` the next pitch could be A or
        B (1st-order can't decide); but given the pair ``(A, A)`` the
        next is always B, ``(A, B)`` → A, etc.  So Markov-2 should
        dramatically outperform Markov-1.
        """
        A, B = 0, 3  # FF, SL
        base = [A, A, B, A, A, B]  # pair-driven pattern
        pts = (base * 20)          # 120 pitches per sequence
        sequences = [_build_seq(pts) for _ in range(20)]

        m1 = MarkovChainOrder1(alpha=0.01).fit(sequences)
        m2 = MarkovChainOrder2(alpha=0.01).fit(sequences)

        ppl1 = m1.calculate_perplexity(sequences)
        ppl2 = m2.calculate_perplexity(sequences)

        # Markov-2 must be at least 30% better than Markov-1.
        improvement = (ppl1 - ppl2) / ppl1
        assert improvement > 0.30, (
            f"Expected markov-2 to beat markov-1 by >30%; got "
            f"markov1_ppl={ppl1:.3f}, markov2_ppl={ppl2:.3f} "
            f"({improvement*100:.1f}%)"
        )

    def test_markov2_falls_back_to_markov1_at_position_0(self):
        """At sequence position 0 there is no ``prev2`` token — the
        model must back off to Markov-1 / the marginal and not crash."""
        sequences = [_build_seq([0, 3, 5, 0, 3])]
        m = MarkovChainOrder2(alpha=0.1).fit(sequences)
        nlls = m.score_sequences(sequences)
        assert nlls.numel() > 0
        assert torch.isfinite(nlls).all()


# ═════════════════════════════════════════════════════════════════════════════
# Smoothing behavior
# ═════════════════════════════════════════════════════════════════════════════


class TestSmoothing:
    def test_markov_smoothing_handles_unseen_transitions(self):
        """Train on ONE transition; score a sequence with novel
        transitions.  No nan / inf allowed."""
        train = [_build_seq([0, 3])]   # only FF→SL seen
        test = [_build_seq([1, 5, 2, 7])]  # SI→CH→FC→KC — all unseen

        m1 = MarkovChainOrder1(alpha=0.1).fit(train)
        m2 = MarkovChainOrder2(alpha=0.1).fit(train)

        for m in (m1, m2):
            nlls = m.score_sequences(test)
            assert torch.isfinite(nlls).all(), f"{m.__class__.__name__} produced non-finite NLL"
            assert nlls.numel() > 0

    def test_markov_lower_alpha_gives_sharper_predictions(self):
        """A more concentrated training set + smaller alpha → lower
        perplexity on the same data."""
        train = [_build_seq([0, 3] * 20) for _ in range(10)]
        m_small = MarkovChainOrder1(alpha=0.001).fit(train)
        m_big = MarkovChainOrder1(alpha=1.0).fit(train)
        ppl_small = m_small.calculate_perplexity(train)
        ppl_big = m_big.calculate_perplexity(train)
        assert ppl_small < ppl_big


# ═════════════════════════════════════════════════════════════════════════════
# Vocab consistency with PitchGPT
# ═════════════════════════════════════════════════════════════════════════════


class TestVocabConsistency:
    def test_markov_uses_same_vocab_as_pitchgpt(self):
        """Emission table shape must match ``(NUM_PITCH_TYPES, zones*velo)``
        so perplexity is computed over the *same* vocabulary as PitchGPT."""
        sequences = [_build_seq([0, 3, 5, 0, 3])]
        m = MarkovChainOrder1(alpha=0.1).fit(sequences)
        assert m._emission.shape == (NUM_PITCH_TYPES, NUM_ZONES * NUM_VELO_BUCKETS)
        # Row sums should be ~1.
        row_sums = m._emission.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_markov_respects_pad_token(self):
        """PAD tokens in the target must not contribute to perplexity."""
        tokens = torch.tensor(
            [_pt_to_token(0), _pt_to_token(3), _pt_to_token(5)],
            dtype=torch.long,
        )
        target = torch.tensor(
            [_pt_to_token(3), _pt_to_token(5), PAD_TOKEN],
            dtype=torch.long,
        )
        ctx = torch.stack([_context_row(0) for _ in range(3)])
        seq = [(tokens, ctx, target)]

        m = MarkovChainOrder1(alpha=0.1).fit(seq)
        nlls = m.score_sequences(seq)
        # Only 2 real targets — PAD position must be skipped.
        assert nlls.numel() == 2

    def test_markov_score_before_fit_raises(self):
        seq = [_build_seq([0, 3, 5])]
        m = MarkovChainOrder1()
        with pytest.raises(RuntimeError):
            m.score_sequences(seq)


# ═════════════════════════════════════════════════════════════════════════════
# Heuristic baseline
# ═════════════════════════════════════════════════════════════════════════════


class TestHeuristicBaseline:
    def test_heuristic_fixed_prior_sums_to_one(self):
        """The hard-coded 50/30/20 prior must be a valid distribution."""
        prior = HeuristicBaseline._fixed_prior()
        assert prior.shape == (NUM_PITCH_TYPES,)
        np.testing.assert_allclose(prior.sum(), 1.0, atol=1e-9)
        assert (prior >= 0).all()

    def test_heuristic_fit_combines_with_empirical(self):
        """After ``fit`` the per-token marginal is a mixture, not just
        the fixed prior — it should shift toward the training data."""
        # All training pitches are the SAME type (FF = 0).  The mixture
        # should push mass toward FF relative to the fixed prior.
        fixed = HeuristicBaseline._fixed_prior()
        train = [_build_seq([0] * 20) for _ in range(5)]
        h = HeuristicBaseline(alpha=0.1).fit(train)
        assert h._pt_marginal[0] > fixed[0], (
            "After fitting on all-FF data, FF mass should exceed the fixed prior."
        )
        # Should still sum to 1.
        np.testing.assert_allclose(h._pt_marginal.sum(), 1.0, atol=1e-9)

    def test_heuristic_perplexity_is_worse_than_markov(self):
        """A naive prior must lose to a Markov-1 chain on structured data."""
        cycle = [0, 3, 5] * 30
        train = [_build_seq(cycle) for _ in range(20)]

        h = HeuristicBaseline(alpha=0.1).fit(train)
        m = MarkovChainOrder1(alpha=0.1).fit(train)

        h_ppl = h.calculate_perplexity(train)
        m_ppl = m.calculate_perplexity(train)

        assert h_ppl > m_ppl, (
            f"Heuristic should be worse than Markov-1 on structured data; "
            f"got heuristic={h_ppl:.3f}, markov1={m_ppl:.3f}"
        )

    def test_heuristic_score_before_fit_raises(self):
        seq = [_build_seq([0, 3])]
        h = HeuristicBaseline()
        with pytest.raises(RuntimeError):
            h.score_sequences(seq)

    def test_heuristic_nlls_finite(self):
        """No inf / nan allowed anywhere in the output."""
        sequences = [_build_seq([0, 3, 5, 0, 3])]
        h = HeuristicBaseline(alpha=0.1).fit(sequences)
        nlls = h.score_sequences(sequences)
        assert torch.isfinite(nlls).all()
