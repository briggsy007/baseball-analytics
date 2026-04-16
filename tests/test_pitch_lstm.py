"""
Tests for the :mod:`src.analytics.pitch_lstm` baseline.

Ticket #3 deliverable: the baseline must match PitchGPT's tokenizer,
context, loss, and checkpoint conventions so the eventual baseline
comparison agent can run both with identical splits.  We don't train
on production data here — a separate agent will do that in Phase 2.
"""

from __future__ import annotations

import pytest
import torch

from src.analytics.pitch_lstm import (
    DEFAULT_DIM,
    DEFAULT_LAYERS,
    PitchLSTMBaseline,
    PitchLSTMNetwork,
)
from src.analytics.pitchgpt import (
    CONTEXT_DIM,
    PAD_TOKEN,
    TOTAL_VOCAB,
    VOCAB_SIZE,
)


# ═════════════════════════════════════════════════════════════════════════════
# Network tests
# ═════════════════════════════════════════════════════════════════════════════


class TestPitchLSTMNetwork:
    """Shape, grad, and causal guarantees."""

    def test_lstm_forward_shape(self):
        """Tiny network: output must be (B, S, output_vocab)."""
        B, S, V, C = 4, 16, 100, 10
        net = PitchLSTMNetwork(
            vocab_size=V + 1,  # +1 so PAD doesn't collide
            context_dim=C,
            d_model=32,
            num_layers=2,
            output_vocab=V,
        )
        tokens = torch.randint(0, V, (B, S))
        ctx = torch.randn(B, S, C)
        out = net(tokens, ctx)
        assert out.shape == (B, S, V), f"Got {out.shape}"

    def test_lstm_gradient_flow(self):
        """Every trainable parameter must receive a non-null gradient."""
        V, C = 50, 8
        net = PitchLSTMNetwork(
            vocab_size=V + 1, context_dim=C, d_model=16, num_layers=2, output_vocab=V,
        )
        tokens = torch.randint(0, V, (2, 8))
        ctx = torch.randn(2, 8, C)
        target = torch.randint(0, V, (2, 8))

        logits = net(tokens, ctx)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, V), target.reshape(-1),
        )
        loss.backward()

        missing = [
            name for name, p in net.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert not missing, f"Missing grads for: {missing}"

    def test_lstm_single_token(self):
        """A sequence of length 1 must not crash."""
        V, C = 20, 8
        net = PitchLSTMNetwork(
            vocab_size=V + 1, context_dim=C, d_model=16, num_layers=1, output_vocab=V,
        )
        tokens = torch.randint(0, V, (1, 1))
        ctx = torch.randn(1, 1, C)
        out = net(tokens, ctx)
        assert out.shape == (1, 1, V)

    def test_lstm_padding_ignored(self):
        """CrossEntropyLoss with ignore_index=PAD_TOKEN must drop PAD targets.

        We construct a batch where *every* target is PAD — the resulting
        loss should be ``nan`` (all entries ignored) rather than a finite
        number, confirming PAD positions are excluded.  The relevant
        invariant for training is that loss with any PAD targets equals
        loss computed on the non-PAD subset only; we check the strict
        all-PAD case here.
        """
        V, C = 20, 8
        net = PitchLSTMNetwork(
            vocab_size=V + 1, context_dim=C, d_model=16, num_layers=1, output_vocab=V,
        )
        tokens = torch.randint(0, V, (2, 6))
        ctx = torch.randn(2, 6, C)
        # PAD index here is V (== vocab_size - 1 with our +1).  Note the
        # network's output head emits V classes so PAD is naturally out
        # of range; CE with ignore_index must handle this.
        ignore_idx = V
        target_all_pad = torch.full((2, 6), ignore_idx, dtype=torch.long)
        logits = net(tokens, ctx)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, V),
            target_all_pad.reshape(-1),
            ignore_index=ignore_idx,
            reduction="mean",
        )
        # All ignored → PyTorch returns nan.
        assert torch.isnan(loss), "Expected NaN loss when every target is PAD."

        # Compare: if we set ONE position to a real class, loss is finite
        # and equal to the per-element cross-entropy on that single
        # position.
        target_one_valid = target_all_pad.clone()
        target_one_valid[0, 0] = 0  # a real class
        loss_valid = torch.nn.functional.cross_entropy(
            logits.reshape(-1, V),
            target_one_valid.reshape(-1),
            ignore_index=ignore_idx,
            reduction="mean",
        )
        # Should be finite and equal to -log_softmax(logits[0,0])[0]
        expected = -torch.log_softmax(logits[0, 0], dim=-1)[0]
        torch.testing.assert_close(loss_valid, expected, atol=1e-5, rtol=1e-5)

    def test_lstm_causal(self):
        """LSTMs are naturally causal: the hidden state at position t
        depends only on inputs [0..t], not on [t+1..T-1].

        We verify this by running the network on a prefix ``[A, B]`` and
        comparing its output at position 1 to the output at position 1
        when the network is run on the full sequence ``[A, B, C]``.
        They must match (within fp tolerance).
        """
        V, C = 30, 8
        net = PitchLSTMNetwork(
            vocab_size=V + 1, context_dim=C, d_model=16, num_layers=2, output_vocab=V,
        )
        net.eval()

        torch.manual_seed(0)
        full_tokens = torch.randint(0, V, (1, 3))
        full_ctx = torch.randn(1, 3, C)

        with torch.no_grad():
            full_out = net(full_tokens, full_ctx)            # (1, 3, V)
            prefix_out = net(full_tokens[:, :2], full_ctx[:, :2])  # (1, 2, V)

        # Position 1 of the full-sequence output must equal position 1
        # of the prefix-only output: no future leakage through the LSTM.
        torch.testing.assert_close(
            full_out[:, 1, :], prefix_out[:, 1, :], atol=1e-5, rtol=1e-5,
        )
        # Position 0 likewise.
        torch.testing.assert_close(
            full_out[:, 0, :], prefix_out[:, 0, :], atol=1e-5, rtol=1e-5,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Wrapper interface
# ═════════════════════════════════════════════════════════════════════════════


class TestPitchLSTMBaseline:
    """The wrapper must expose the same-shape API as ``PitchGPT``."""

    def test_wrapper_attributes(self):
        wrapper = PitchLSTMBaseline()
        assert wrapper.model_name == "PitchLSTMBaseline"
        assert wrapper.version.startswith("1.")
        # Has the train / evaluate / score_sequences / calculate_perplexity
        # methods the comparison harness will call.
        assert hasattr(wrapper, "train")
        assert hasattr(wrapper, "evaluate")
        assert hasattr(wrapper, "score_sequences")
        assert hasattr(wrapper, "calculate_perplexity")

    def test_wrapper_predict_raises(self):
        """A baseline for perplexity does not implement predict()."""
        wrapper = PitchLSTMBaseline()
        with pytest.raises(NotImplementedError):
            wrapper.predict(conn=None)


# ═════════════════════════════════════════════════════════════════════════════
# Smoke test: real-vocab network, tiny synthetic batch
# ═════════════════════════════════════════════════════════════════════════════


class TestSmoke:
    """Instantiate with the real PitchGPT vocab size and run a single
    forward pass on a tiny synthetic batch.  DO NOT train on prod data.
    """

    def test_real_vocab_forward_pass(self):
        net = PitchLSTMNetwork(
            vocab_size=TOTAL_VOCAB,
            context_dim=CONTEXT_DIM,
            d_model=DEFAULT_DIM,
            num_layers=DEFAULT_LAYERS,
        )
        B, S = 2, 4
        # Tokens must be < TOTAL_VOCAB; keep them well below VOCAB_SIZE
        # so they're valid pitch tokens.
        tokens = torch.randint(0, VOCAB_SIZE, (B, S))
        ctx = torch.randn(B, S, CONTEXT_DIM)
        out = net(tokens, ctx)
        assert out.shape == (B, S, VOCAB_SIZE)
        # Softmax row sums to ~1.
        probs = torch.softmax(out, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B, S), atol=1e-4)

    def test_real_vocab_param_count_reasonable(self):
        """Parameter count should be in a sane range for a 2-layer LSTM
        at d_model=128 with 2210 output classes + 2212 input classes.
        Rough expectation: < 2M params (vs PitchGPT ~3.2M)."""
        net = PitchLSTMNetwork(
            vocab_size=TOTAL_VOCAB,
            context_dim=CONTEXT_DIM,
            d_model=DEFAULT_DIM,
            num_layers=DEFAULT_LAYERS,
        )
        total = sum(p.numel() for p in net.parameters())
        # Hard floor/ceiling to catch drastic architecture regressions.
        assert 500_000 < total < 5_000_000, f"Unexpected param count: {total}"
