"""Tests for PitchGPT feature ablation (Ticket #6)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.pitchgpt_ablation import (  # noqa: E402
    COUNT_END,
    OUTS_END,
    IdentityTokenDataset,
    _IdentityPitchGPT,
    mask_context,
)
from src.analytics.pitchgpt import (  # noqa: E402
    CONTEXT_DIM,
    PitchGPTModel,
    TOTAL_VOCAB,
)


# ── Context masking ─────────────────────────────────────────────────────


def test_mask_context_full_is_identity():
    ctx = torch.randn(2, 5, CONTEXT_DIM)
    out = mask_context(ctx, "full")
    assert torch.equal(ctx, out)


def test_mask_context_tokens_only_zeros_all():
    ctx = torch.randn(2, 5, CONTEXT_DIM)
    out = mask_context(ctx, "tokens_only")
    assert torch.all(out == 0)


def test_mask_context_count_only_keeps_only_count_and_outs():
    ctx = torch.randn(2, 5, CONTEXT_DIM)
    out = mask_context(ctx, "count_only")
    # Count + outs dims preserved.
    assert torch.equal(out[..., :OUTS_END], ctx[..., :OUTS_END])
    # Everything after OUTS_END must be zero.
    assert torch.all(out[..., OUTS_END:] == 0)


def test_mask_context_invalid_mode_raises():
    with pytest.raises(ValueError):
        mask_context(torch.zeros(1, CONTEXT_DIM), "garbage")


# ── Tokens-only zeroing actually produces identical output ──────────────


def test_tokens_only_ignores_context_changes():
    """For the tokens_only mode, the model's forward pass should be
    invariant to the context values (because we zero them before
    feeding).  We simulate this by calling the model twice — once on
    real context, once on zeros — after passing both through
    mask_context.
    """
    torch.manual_seed(0)
    model = PitchGPTModel().eval()
    B, S = 2, 10
    tokens = torch.randint(0, 100, (B, S))
    ctx_a = torch.randn(B, S, CONTEXT_DIM)
    ctx_b = torch.randn(B, S, CONTEXT_DIM)

    with torch.no_grad():
        out_a = model(tokens, mask_context(ctx_a, "tokens_only"))
        out_b = model(tokens, mask_context(ctx_b, "tokens_only"))

    # With the context zeroed, both forward passes should produce
    # identical output.
    torch.testing.assert_close(out_a, out_b, atol=1e-6, rtol=1e-6)


def test_count_only_sensitive_to_count_but_not_to_runner():
    """For count_only mode, changing count/outs dims must change
    output; changing runner/hand/etc must NOT.
    """
    torch.manual_seed(0)
    model = PitchGPTModel().eval()
    B, S = 1, 5
    tokens = torch.randint(0, 100, (B, S))
    ctx = torch.zeros(B, S, CONTEXT_DIM)
    ctx[..., 0] = 1.0  # count_state = 0

    ctx_diff_count = ctx.clone()
    ctx_diff_count[..., 0] = 0.0
    ctx_diff_count[..., 5] = 1.0  # different count_state

    ctx_diff_runner = ctx.clone()
    ctx_diff_runner[..., 16] = 1.0  # change a runner dim (index in [15,23))

    with torch.no_grad():
        out0 = model(tokens, mask_context(ctx, "count_only"))
        out1 = model(tokens, mask_context(ctx_diff_count, "count_only"))
        out2 = model(tokens, mask_context(ctx_diff_runner, "count_only"))

    # Changing count → should change output.
    assert not torch.allclose(out0, out1, atol=1e-6)
    # Changing runner → must NOT change output (that dim is masked).
    torch.testing.assert_close(out0, out2, atol=1e-6, rtol=1e-6)


# ── Identity-only dataset vocabulary ────────────────────────────────────


class _FakeBaseDataset:
    """Minimal stand-in for PitchSequenceDataset that has the
    attributes the IdentityTokenDataset needs.
    """

    def __init__(self, sequences, pitcher_ids, sequence_pids, max_seq_len=16):
        self.sequences = sequences
        self.pitcher_ids = set(pitcher_ids)
        self.sequence_pids = sequence_pids
        self.max_seq_len = max_seq_len


def test_identity_dataset_token_vocabulary_swap():
    """IdentityTokenDataset must replace pitch tokens with pitcher-id
    tokens, one per sequence.
    """
    # Three fake sequences belonging to two pitchers.
    seq_a = (
        torch.tensor([1, 2, 3, 4], dtype=torch.long),
        torch.zeros(4, CONTEXT_DIM),
        torch.tensor([2, 3, 4, 5], dtype=torch.long),
    )
    seq_b = (
        torch.tensor([5, 6, 7], dtype=torch.long),
        torch.zeros(3, CONTEXT_DIM),
        torch.tensor([6, 7, 8], dtype=torch.long),
    )
    seq_c = (
        torch.tensor([9, 10], dtype=torch.long),
        torch.zeros(2, CONTEXT_DIM),
        torch.tensor([10, 11], dtype=torch.long),
    )
    base = _FakeBaseDataset(
        sequences=[seq_a, seq_b, seq_c],
        pitcher_ids={100, 200},
        sequence_pids=[100, 200, 100],
    )

    ds = IdentityTokenDataset(base)
    assert ds.n_known == 2
    assert ds.vocab_size == 3  # UNK + 2 pitchers
    # pid_to_idx should be deterministic (sorted, +1 offset).
    assert ds.pid_to_idx == {100: 1, 200: 2}

    # Sequence 0 (pid=100) → all tokens == 1.
    inp0, ctx0, tgt0 = ds[0]
    assert torch.all(inp0 == 1)
    assert torch.all(tgt0 == 1)
    assert inp0.shape == seq_a[0].shape
    # Sequence 1 (pid=200) → all tokens == 2.
    inp1, _, tgt1 = ds[1]
    assert torch.all(inp1 == 2)
    assert torch.all(tgt1 == 2)
    assert inp1.shape == seq_b[0].shape


def test_identity_dataset_unknown_pitcher_falls_back_to_unk():
    seq_a = (
        torch.tensor([1, 2], dtype=torch.long),
        torch.zeros(2, CONTEXT_DIM),
        torch.tensor([2, 3], dtype=torch.long),
    )
    base = _FakeBaseDataset(
        sequences=[seq_a],
        pitcher_ids={999},  # "known" pitcher 999
        sequence_pids=[12345],  # seq belongs to an UNKNOWN pitcher
    )
    # Reuse a pid_to_idx that does NOT include 12345.
    ds = IdentityTokenDataset(base, pid_to_idx={999: 1})
    inp, _, _ = ds[0]
    # Unknown pitcher → UNK_IDX (= 0).
    assert torch.all(inp == IdentityTokenDataset.UNK_IDX)


def test_identity_dataset_reuses_pid_to_idx_for_val_test():
    """Val/test datasets must use the train-derived mapping — same
    known pitchers map to the same indices.
    """
    seq = (
        torch.tensor([1], dtype=torch.long),
        torch.zeros(1, CONTEXT_DIM),
        torch.tensor([2], dtype=torch.long),
    )
    train_base = _FakeBaseDataset(
        sequences=[seq, seq],
        pitcher_ids={7, 13},
        sequence_pids=[7, 13],
    )
    train_ds = IdentityTokenDataset(train_base)

    val_base = _FakeBaseDataset(
        sequences=[seq],
        pitcher_ids={7},
        sequence_pids=[7],
    )
    val_ds = IdentityTokenDataset(val_base, pid_to_idx=train_ds.pid_to_idx)
    # Same mapping applied.
    assert val_ds.pid_to_idx[7] == train_ds.pid_to_idx[7]
    # And the produced token matches.
    inp, _, _ = val_ds[0]
    assert inp[0].item() == train_ds.pid_to_idx[7]


def test_identity_model_output_vocab_matches_requested():
    """The identity-only wrapper must project to the pitcher-id vocab."""
    m = _IdentityPitchGPT(vocab_size=10, output_vocab=5, pad_token=9).eval()
    tokens = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    ctx = torch.zeros(1, 4, CONTEXT_DIM)
    with torch.no_grad():
        out = m(tokens, ctx)
    assert out.shape == (1, 4, 5)
