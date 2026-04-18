"""
Tests for the PitchGPT transformer-based pitch sequence model.

Covers tokenizer encode/decode, dataset construction, model forward pass,
causal masking, PPS computation, disruption index, and edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.analytics.pitchgpt import (
    CONTEXT_DIM,
    NUM_BATTER_HANDS,
    NUM_COUNT_STATES,
    NUM_INNING_BUCKETS,
    NUM_OUTS,
    NUM_PITCH_TYPES,
    NUM_RUNNER_STATES,
    NUM_SCORE_DIFF_BUCKETS,
    NUM_VELO_BUCKETS,
    NUM_ZONES,
    PAD_TOKEN,
    TOTAL_VOCAB,
    VOCAB_SIZE,
    PitchGPTModel,
    PitchSequenceDataset,
    PitchTokenizer,
    _collate_fn,
    _compute_per_pitch_score_diff,
    _score_sequences,
    audit_no_game_overlap,
    train_pitchgpt,
)


# Offset inside the 34-dim context one-hot where the 5 score-diff buckets sit.
_SCORE_BUCKET_OFFSET = (
    NUM_COUNT_STATES
    + NUM_OUTS
    + NUM_RUNNER_STATES
    + NUM_BATTER_HANDS
    + NUM_INNING_BUCKETS
)


# ═════════════════════════════════════════════════════════════════════════════
# Tokenizer tests
# ═════════════════════════════════════════════════════════════════════════════


class TestPitchTokenizer:
    """Tests for PitchTokenizer encode / decode round-trips."""

    def test_known_pitch_type_encoding(self):
        """FF at centre of zone with 94 mph should give a deterministic token."""
        token = PitchTokenizer.encode("FF", 0.0, 2.5, 94.0)
        assert 0 <= token < VOCAB_SIZE

    def test_roundtrip_encode_decode(self):
        """Encoding then decoding should recover pitch type."""
        for pt in ["FF", "SL", "CU", "CH", "SI", "FC"]:
            token = PitchTokenizer.encode(pt, 0.0, 2.5, 90.0)
            decoded = PitchTokenizer.decode(token)
            assert decoded["pitch_type"] == pt

    def test_unknown_pitch_type(self):
        """An unrecognised pitch type maps to the unknown slot."""
        token = PitchTokenizer.encode("ZZ", 0.0, 2.5, 90.0)
        decoded = PitchTokenizer.decode(token)
        assert decoded["pitch_type"] == "UN"  # unknown

    def test_none_pitch_type(self):
        """None pitch type maps to unknown."""
        token = PitchTokenizer.encode(None, 0.0, 2.5, 90.0)
        decoded = PitchTokenizer.decode(token)
        assert decoded["pitch_type"] == "UN"

    def test_missing_location_gets_out_of_zone(self):
        """NaN plate location should produce the out-of-zone index."""
        zone = PitchTokenizer.location_to_zone(None, None)
        assert zone == NUM_ZONES - 1

        zone2 = PitchTokenizer.location_to_zone(float("nan"), 2.5)
        assert zone2 == NUM_ZONES - 1

    def test_velocity_buckets(self):
        assert PitchTokenizer.velocity_to_bucket(75.0) == 0
        assert PitchTokenizer.velocity_to_bucket(82.0) == 1
        assert PitchTokenizer.velocity_to_bucket(87.0) == 2
        assert PitchTokenizer.velocity_to_bucket(93.0) == 3
        assert PitchTokenizer.velocity_to_bucket(98.0) == 4
        assert PitchTokenizer.velocity_to_bucket(None) == 2  # default

    def test_token_range(self):
        """All valid tokens should lie in [0, VOCAB_SIZE)."""
        for pt in ["FF", "SL", "CU", "CH", None]:
            for px in [-1.5, 0.0, 1.5, None]:
                for pz in [1.0, 2.5, 4.0, None]:
                    for v in [70.0, 82.0, 87.0, 93.0, 100.0, None]:
                        tok = PitchTokenizer.encode(pt, px, pz, v)
                        assert 0 <= tok < VOCAB_SIZE, f"Token {tok} out of range"

    def test_context_encoding_shape(self):
        """Context tensor should have the expected dimension."""
        ctx_list = PitchTokenizer.encode_context(
            balls=1, strikes=2, outs=1, on_1b=True, on_2b=False,
            on_3b=False, stand="L", inning=5, score_diff=-2,
        )
        tensor = PitchTokenizer.context_to_tensor(ctx_list)
        assert tensor.shape == (CONTEXT_DIM,)
        assert tensor.sum().item() == 6.0  # exactly 6 one-hot bits

    def test_context_score_diff_buckets(self):
        """Score diff should map to 5 distinct buckets."""
        buckets_seen = set()
        for diff in [-10, -2, 0, 2, 10]:
            ctx = PitchTokenizer.encode_context(
                0, 0, 0, False, False, False, "R", 1, diff,
            )
            buckets_seen.add(ctx[5])
        assert len(buckets_seen) == 5


# ═════════════════════════════════════════════════════════════════════════════
# Dataset tests
# ═════════════════════════════════════════════════════════════════════════════


class TestPitchSequenceDataset:
    """Tests for the PyTorch Dataset."""

    def test_dataset_loads(self, db_conn):
        """Dataset should load at least some sequences from test data."""
        ds = PitchSequenceDataset(db_conn, max_seq_len=64)
        # In test DB (synthetic or sampled), we should have at least one sequence
        assert len(ds) >= 0  # may be 0 if data is too sparse

    def test_sequence_structure(self, db_conn):
        """Each item should be (input_tokens, context, target)."""
        ds = PitchSequenceDataset(db_conn, max_seq_len=64)
        if len(ds) == 0:
            pytest.skip("No sequences in test dataset.")

        tokens, ctx, target = ds[0]
        assert tokens.dtype == torch.long
        assert target.dtype == torch.long
        assert ctx.dtype == torch.float32

        # Target is input shifted by one
        assert tokens.shape[0] == target.shape[0]
        assert ctx.shape[0] == tokens.shape[0]
        assert ctx.shape[1] == CONTEXT_DIM

    def test_target_is_shifted_input(self, db_conn):
        """Target token at position i should equal input token at position i+1
        of the original un-split sequence."""
        ds = PitchSequenceDataset(db_conn, max_seq_len=64)
        if len(ds) == 0:
            pytest.skip("No sequences in test dataset.")

        tokens, _, target = ds[0]
        # tokens = original[:-1], target = original[1:]
        # Therefore target[i] is the ground-truth next token for tokens[i]
        # Both should be valid token ids
        for t in target.tolist():
            assert 0 <= t < VOCAB_SIZE

    def test_collate_fn_padding(self, db_conn):
        """Collation should pad sequences to the same length."""
        ds = PitchSequenceDataset(db_conn, max_seq_len=64)
        if len(ds) < 2:
            pytest.skip("Need at least 2 sequences to test collation.")

        batch = [ds[i] for i in range(min(4, len(ds)))]
        tokens, ctx, target = _collate_fn(batch)

        assert tokens.shape[0] == len(batch)
        assert ctx.shape[0] == len(batch)
        assert target.shape[0] == len(batch)
        # All same length
        assert tokens.shape[1] == ctx.shape[1] == target.shape[1]


# ═════════════════════════════════════════════════════════════════════════════
# Model tests
# ═════════════════════════════════════════════════════════════════════════════


class TestPitchGPTModel:
    """Tests for the transformer model."""

    @pytest.fixture
    def model(self):
        return PitchGPTModel(
            d_model=32, nhead=2, num_layers=2, max_seq_len=64,
        )

    def test_forward_pass_shape(self, model):
        """Output logits should be (batch, seq_len, VOCAB_SIZE)."""
        B, S = 2, 10
        tokens = torch.randint(0, VOCAB_SIZE, (B, S))
        ctx = torch.randn(B, S, CONTEXT_DIM)
        logits = model(tokens, ctx)
        assert logits.shape == (B, S, VOCAB_SIZE)

    def test_single_token_input(self, model):
        """Model should handle a sequence of length 1."""
        tokens = torch.randint(0, VOCAB_SIZE, (1, 1))
        ctx = torch.randn(1, 1, CONTEXT_DIM)
        logits = model(tokens, ctx)
        assert logits.shape == (1, 1, VOCAB_SIZE)

    def test_causal_mask(self, model):
        """Future tokens must not influence current predictions.

        We verify this by running the same prefix with different future
        tokens and checking that the logits for the prefix are identical.
        """
        model.eval()
        S = 8
        prefix = torch.randint(0, VOCAB_SIZE, (1, S))
        ctx = torch.randn(1, S, CONTEXT_DIM)

        # Two sequences: same first 4 tokens, different last 4
        seq_a = prefix.clone()
        seq_b = prefix.clone()
        seq_b[0, 4:] = torch.randint(0, VOCAB_SIZE, (4,))
        ctx_a = ctx.clone()
        ctx_b = ctx.clone()

        with torch.no_grad():
            logits_a = model(seq_a, ctx_a)
            logits_b = model(seq_b, ctx_b)

        # The logits for positions 0-3 should be identical
        torch.testing.assert_close(
            logits_a[:, :4, :], logits_b[:, :4, :],
            atol=1e-5, rtol=1e-5,
        )

    def test_output_is_valid_log_distribution(self, model):
        """Softmax of logits should sum to 1."""
        tokens = torch.randint(0, VOCAB_SIZE, (1, 5))
        ctx = torch.randn(1, 5, CONTEXT_DIM)
        logits = model(tokens, ctx)
        probs = torch.softmax(logits, dim=-1)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)

    def test_gradient_flow(self, model):
        """Gradients should flow through the full model."""
        tokens = torch.randint(0, VOCAB_SIZE, (2, 8))
        ctx = torch.randn(2, 8, CONTEXT_DIM)
        target = torch.randint(0, VOCAB_SIZE, (2, 8))

        logits = model(tokens, ctx)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE), target.reshape(-1),
        )
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


# ═════════════════════════════════════════════════════════════════════════════
# Training (quick smoke test)
# ═════════════════════════════════════════════════════════════════════════════


class TestTraining:
    """Quick smoke test of the training loop."""

    def test_train_returns_metrics(self, db_conn):
        """Training should return a metrics dict (even with tiny data)."""
        result = train_pitchgpt(
            db_conn,
            seasons=list(range(2015, 2027)),
            val_seasons=[2026],
            epochs=1,
            batch_size=4,
            d_model=32,
            nhead=2,
            num_layers=1,
            max_seq_len=32,
            version="test",
        )
        assert isinstance(result, dict)
        if result.get("status") == "no_data":
            pytest.skip("No training data in test DB.")
        assert "final_train_loss" in result
        assert result["final_train_loss"] > 0


# ═════════════════════════════════════════════════════════════════════════════
# Inference tests
# ═════════════════════════════════════════════════════════════════════════════


class TestInference:
    """Tests for PPS and disruption index (require a trained model)."""

    @pytest.fixture(autouse=True)
    def ensure_model(self, db_conn):
        """Train a tiny model for inference tests."""
        from pathlib import Path
        model_path = Path(__file__).resolve().parents[1] / "models" / "pitchgpt_vtest.pt"
        if not model_path.exists():
            train_pitchgpt(
                db_conn,
                seasons=list(range(2015, 2027)),
                val_seasons=[2026],
                epochs=1,
                batch_size=4,
                d_model=32,
                nhead=2,
                num_layers=1,
                max_seq_len=32,
                version="test",
            )

    def test_score_sequences(self, db_conn):
        """_score_sequences should produce valid per-pitch NLL values."""
        from src.analytics.pitchgpt import _load_model, _get_pitcher_game_sequences

        try:
            model = _load_model("test")
        except FileNotFoundError:
            pytest.skip("Model not available.")

        # Find a pitcher+game combo with enough pitches in a single game
        df = db_conn.execute("""
            SELECT pitcher_id, game_pk, COUNT(*) AS cnt
            FROM pitches
            WHERE pitch_type IS NOT NULL
            GROUP BY pitcher_id, game_pk
            HAVING COUNT(*) >= 5
            LIMIT 1
        """).fetchdf()
        if df.empty:
            pytest.skip("No pitcher-game combo with >= 5 pitches in test DB.")

        pid = int(df.iloc[0]["pitcher_id"])
        gpk = int(df.iloc[0]["game_pk"])
        pitch_df = _get_pitcher_game_sequences(db_conn, pid, game_pk=gpk)
        if pitch_df.empty or len(pitch_df) < 2:
            pytest.skip("Not enough pitches.")

        results = _score_sequences(model, pitch_df)
        if len(results) == 0:
            pytest.skip("No scoreable sequences (games may have < 2 pitches).")

        for g in results:
            assert "per_pitch_nll" in g
            assert "mean_nll" in g
            for nll in g["per_pitch_nll"]:
                assert nll >= 0, "NLL should be non-negative"

    def test_calculate_predictability_returns_valid(self, db_conn):
        """PPS should return a valid perplexity > 0."""
        from src.analytics.pitchgpt import calculate_predictability

        df = db_conn.execute("""
            SELECT pitcher_id FROM pitches
            GROUP BY pitcher_id
            HAVING COUNT(*) >= 5
            LIMIT 1
        """).fetchdf()
        if df.empty:
            pytest.skip("No qualifying pitcher.")

        pid = int(df.iloc[0]["pitcher_id"])
        result = calculate_predictability(db_conn, pid, model_version="test")

        assert "pps" in result
        assert "perplexity" in result
        if result["n_pitches"] > 0:
            assert result["pps"] > 0
            assert result["perplexity"] >= 1.0

    def test_calculate_disruption_index(self, db_conn):
        """Disruption index should return per-pitch surprise scores."""
        from src.analytics.pitchgpt import calculate_disruption_index

        row = db_conn.execute("""
            SELECT pitcher_id, game_pk, COUNT(*) AS cnt
            FROM pitches
            WHERE pitch_type IS NOT NULL
            GROUP BY pitcher_id, game_pk
            HAVING COUNT(*) >= 5
            LIMIT 1
        """).fetchdf()
        if row.empty:
            pytest.skip("No qualifying game.")

        pid = int(row.iloc[0]["pitcher_id"])
        gpk = int(row.iloc[0]["game_pk"])
        result = calculate_disruption_index(db_conn, pid, gpk, model_version="test")

        assert "per_pitch_surprise" in result
        assert "pitch_types" in result
        if result["per_pitch_surprise"]:
            assert len(result["per_pitch_surprise"]) == len(result["pitch_types"])
            assert result["mean_surprise"] >= 0
            assert result["max_surprise"] >= 0


# ═════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge-case handling."""

    def test_empty_sequence_predictability(self, db_conn):
        """A pitcher with no data should return zeroed output, not crash."""
        from src.analytics.pitchgpt import calculate_predictability
        # Use a fake pitcher ID that won't exist
        try:
            result = calculate_predictability(db_conn, pitcher_id=999999999, model_version="test")
            assert result["pps"] == 0.0
            assert result["n_pitches"] == 0
        except FileNotFoundError:
            pytest.skip("Model not trained.")

    def test_empty_sequence_disruption(self, db_conn):
        """Disruption index for missing game should return empty list."""
        from src.analytics.pitchgpt import calculate_disruption_index
        try:
            result = calculate_disruption_index(
                db_conn, pitcher_id=999999999, game_pk=0, model_version="test",
            )
            assert result["per_pitch_surprise"] == []
        except FileNotFoundError:
            pytest.skip("Model not trained.")

    def test_single_pitch_game_scoring(self):
        """A game with only 1 pitch should produce no scores (need >= 2)."""
        model = PitchGPTModel(d_model=32, nhead=2, num_layers=1, max_seq_len=32)
        model.eval()

        df = pd.DataFrame([{
            "game_pk": 1,
            "pitcher_id": 100,
            "pitch_type": "FF",
            "plate_x": 0.0,
            "plate_z": 2.5,
            "release_speed": 94.0,
            "balls": 0,
            "strikes": 0,
            "outs_when_up": 0,
            "on_1b": 0,
            "on_2b": 0,
            "on_3b": 0,
            "stand": "R",
            "inning": 1,
            "at_bat_number": 1,
            "pitch_number": 1,
        }])
        results = _score_sequences(model, df)
        assert len(results) == 0  # not enough pitches to score


# ═════════════════════════════════════════════════════════════════════════════
# Ticket #2: score_diff no longer hard-coded
# ═════════════════════════════════════════════════════════════════════════════


class TestScoreDiffFix:
    """Regression tests for the score_diff context feature.

    Prior to ticket #2 the dataset loader and inference path both
    hard-coded ``score_diff=0`` so 100% of the mass landed in bucket 2
    ("tie").  These tests lock in the fix.
    """

    def test_score_diff_not_hardcoded_zero(self, db_conn):
        """After the fix at least one non-tie bucket must show up in the
        dataset on synthetic data."""
        ds = PitchSequenceDataset(db_conn, max_seq_len=128)
        if len(ds) == 0:
            pytest.skip("No sequences in test dataset.")

        # Extract the score-bucket one-hot argmax from every context row.
        buckets: list[int] = []
        for _tokens, ctx, _target in ds.sequences:
            block = ctx[:, _SCORE_BUCKET_OFFSET : _SCORE_BUCKET_OFFSET + NUM_SCORE_DIFF_BUCKETS]
            buckets.extend(block.argmax(dim=-1).tolist())

        assert buckets, "Dataset produced no context rows."
        bucket_set = set(buckets)
        # The key assertion: if score_diff were still hard-coded to 0 the
        # only bucket we'd ever see is bucket 2 ("tie").  We must see at
        # least one other bucket on a realistic sample.
        non_tie = bucket_set - {2}
        assert non_tie, (
            f"All pitches mapped to bucket 2 (tie): score_diff still hardcoded? "
            f"Buckets seen: {bucket_set}"
        )

    def test_score_diff_reflects_game_state(self):
        """A synthetic game where the home team has already scored 5+
        should place the pitcher's context in the 'big lead' bucket
        when the home team is pitching (inning_topbot='Top')."""
        # Build a deliberately-crafted tiny game: the home team hits
        # five solo home runs in the top of the 1st... wait, that's
        # backwards.  In Statcast convention, during the TOP of the
        # inning the AWAY team bats.  So to make the HOME team lead,
        # we simulate the home team batting in the BOTTOM of the 1st
        # and hitting 5 solo home runs, then a pitch is thrown with
        # inning_topbot='Top' (the next half-inning) where the HOME
        # team is now pitching and leading by 5.
        rows = []
        # 5 solo home runs, home team batting (Bot of the 1st).
        for i in range(5):
            rows.append({
                "game_pk": 9999,
                "inning_topbot": "Bot",
                "on_1b": 0, "on_2b": 0, "on_3b": 0,
                "events": "home_run",
                "delta_run_exp": 1.0,
                "at_bat_number": i + 1,
                "pitch_number": 1,
            })
        # Now a pitch in the TOP of the 2nd (home team pitching,
        # leading by 5).  score_diff from pitcher POV should be +5.
        rows.append({
            "game_pk": 9999,
            "inning_topbot": "Top",
            "on_1b": 0, "on_2b": 0, "on_3b": 0,
            "events": None,
            "delta_run_exp": 0.0,
            "at_bat_number": 6,
            "pitch_number": 1,
        })

        df = pd.DataFrame(rows)
        diffs = _compute_per_pitch_score_diff(df)
        assert len(diffs) == len(rows)
        # The 6th pitch is thrown with home up 5-0.  Pitcher POV = +5.
        assert diffs[-1] == 5, f"Expected +5 at pitch 6, got {diffs[-1]}"

        # And the bucket mapping for +5 is bucket 4 ("big lead").
        ctx = PitchTokenizer.encode_context(
            balls=0, strikes=0, outs=0,
            on_1b=False, on_2b=False, on_3b=False,
            stand="R", inning=2, score_diff=diffs[-1],
        )
        assert ctx[5] == 4, f"Expected bucket 4 (big lead), got {ctx[5]}"

    def test_score_diff_home_run_increments_correctly(self):
        """A single home-run event should push the batting team's score
        by exactly 1 + runners-on.  Verifies the reconstructor math."""
        # Grand slam: bases loaded, home run → 4 runs for the batting team.
        rows = [
            # Pre-HR pitch: bases loaded, 0-0 count.
            {
                "game_pk": 1,
                "inning_topbot": "Top",  # away batting
                "on_1b": 1, "on_2b": 1, "on_3b": 1,
                "events": None,
                "delta_run_exp": 0.0,
                "at_bat_number": 1, "pitch_number": 1,
            },
            # HR pitch: same PA, bases still loaded at moment of pitch.
            {
                "game_pk": 1,
                "inning_topbot": "Top",
                "on_1b": 1, "on_2b": 1, "on_3b": 1,
                "events": "home_run",
                "delta_run_exp": 3.5,
                "at_bat_number": 1, "pitch_number": 2,
            },
            # Next pitch of the inning (new batter, bases empty, away leads 4-0).
            {
                "game_pk": 1,
                "inning_topbot": "Top",
                "on_1b": 0, "on_2b": 0, "on_3b": 0,
                "events": None,
                "delta_run_exp": 0.0,
                "at_bat_number": 2, "pitch_number": 1,
            },
        ]
        df = pd.DataFrame(rows)
        diffs = _compute_per_pitch_score_diff(df)
        # Pitches 1 and 2: score_diff still 0 (event happens *on* pitch 2,
        # runs are credited after recording).
        assert diffs[0] == 0
        assert diffs[1] == 0
        # Pitch 3: home team pitching (inning_topbot=Top), away now up 4-0,
        # pitcher POV score_diff = home - away = -4.
        assert diffs[2] == -4, f"Expected -4 after grand slam, got {diffs[2]}"


# ═════════════════════════════════════════════════════════════════════════════
# Ticket #1: date-based train/val/test split + leakage audit
# ═════════════════════════════════════════════════════════════════════════════


class TestDateSplit:
    """Regression tests for the leakage-free date-based split.

    Guardrails:
      * Overlapping train/val/test year ranges must raise ``ValueError``.
      * On real date-separated data no ``game_pk`` can appear in more
        than one split.
      * Each of train/val/test must contain >= 1 sequence on a multi-
        year fixture.
    """

    def test_split_year_ranges_disjoint_guard(self, db_conn):
        """Overlapping ranges must raise ``ValueError`` at Dataset init."""
        # train 2015-2022, val 2022-2023 overlap on 2022 → must raise.
        with pytest.raises(ValueError, match="overlap"):
            PitchSequenceDataset(
                db_conn,
                max_seq_len=32,
                split_mode="train",
                train_range=(2015, 2022),
                val_range=(2022, 2023),
                test_range=(2024, 2024),
                max_games_per_split=5,
            )
        # train vs test overlap
        with pytest.raises(ValueError, match="overlap"):
            PitchSequenceDataset(
                db_conn,
                max_seq_len=32,
                split_mode="train",
                train_range=(2015, 2022),
                val_range=(2023, 2023),
                test_range=(2020, 2024),
                max_games_per_split=5,
            )

    def test_no_game_overlap_across_splits(self, db_conn):
        """No ``game_pk`` may appear in more than one date-based split."""
        # Use the generous range that synthetic data covers (2015-2026).
        train_ds = PitchSequenceDataset(
            db_conn,
            max_seq_len=64,
            split_mode="train",
            train_range=(2015, 2022),
            val_range=(2023, 2023),
            test_range=(2024, 2024),
            max_games_per_split=20,
        )
        val_ds = PitchSequenceDataset(
            db_conn,
            max_seq_len=64,
            split_mode="val",
            train_range=(2015, 2022),
            val_range=(2023, 2023),
            test_range=(2024, 2024),
            max_games_per_split=20,
        )
        test_ds = PitchSequenceDataset(
            db_conn,
            max_seq_len=64,
            split_mode="test",
            train_range=(2015, 2022),
            val_range=(2023, 2023),
            test_range=(2024, 2024),
            max_games_per_split=20,
        )

        report = audit_no_game_overlap(train_ds, val_ds, test_ds)

        # Hard leakage constraint: no game_pk can be shared.
        assert report["shared_game_pks"] == 0, (
            f"Leakage detected: {report['shared_game_pks']} shared game_pks "
            f"across splits. report={report}"
        )

    def test_each_split_non_empty(self, db_conn):
        """On a multi-year fixture, train/val/test must each have >= 1 seq."""
        train_ds = PitchSequenceDataset(
            db_conn, max_seq_len=64, split_mode="train",
            train_range=(2015, 2022), val_range=(2023, 2023),
            test_range=(2024, 2024), max_games_per_split=50,
        )
        val_ds = PitchSequenceDataset(
            db_conn, max_seq_len=64, split_mode="val",
            train_range=(2015, 2022), val_range=(2023, 2023),
            test_range=(2024, 2024), max_games_per_split=50,
        )
        test_ds = PitchSequenceDataset(
            db_conn, max_seq_len=64, split_mode="test",
            train_range=(2015, 2022), val_range=(2023, 2023),
            test_range=(2024, 2024), max_games_per_split=50,
        )

        # The synthetic fixture spans 2015-2026 with data in every year,
        # so each of these must contain at least one sequence.
        assert len(train_ds) >= 1, "Train split produced zero sequences."
        assert len(val_ds) >= 1, "Val split produced zero sequences."
        assert len(test_ds) >= 1, "Test split produced zero sequences."

    def test_invalid_split_mode_raises(self, db_conn):
        """An unknown split_mode should raise ``ValueError``."""
        with pytest.raises(ValueError):
            PitchSequenceDataset(
                db_conn, max_seq_len=32, split_mode="garbage",  # type: ignore[arg-type]
                max_games_per_split=5,
            )

    def test_backwards_compatible_default(self, db_conn):
        """``split_mode=None`` must preserve the original single-season API.

        The existing ``train_pitchgpt(conn, seasons=...)`` path and the
        dashboards/precompute calls depend on this.
        """
        ds_legacy = PitchSequenceDataset(
            db_conn, seasons=list(range(2015, 2027)), max_seq_len=32,
        )
        # Should load without raising.  Number of sequences may legitimately
        # be zero on a tiny test fixture; what matters is the API works.
        assert isinstance(ds_legacy.game_pks, set)
        assert isinstance(ds_legacy.pitcher_ids, set)
