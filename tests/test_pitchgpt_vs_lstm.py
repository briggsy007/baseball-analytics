"""
Tests for :mod:`scripts.train_pitchgpt_vs_lstm`.

We deliberately do NOT run the training loop here — that's the job of
the script itself on real data.  These tests cover the metric-assembly
logic only: verdict construction, seed handling, and JSON shape.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add project root / scripts to sys.path so we can import the script as a module.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import train_pitchgpt_vs_lstm as tvl  # type: ignore  # noqa: E402


# ═════════════════════════════════════════════════════════════════════════════
# Verdict builder
# ═════════════════════════════════════════════════════════════════════════════


class TestVerdictBuilder:
    """The ≥15% threshold is a hard spec condition — lock it in."""

    def test_pass_threshold(self):
        passed, msg = tvl._build_verdict(improvement_pct=18.3)
        assert passed is True
        assert "PASS" in msg
        assert "18.3" in msg

    def test_fail_threshold(self):
        passed, msg = tvl._build_verdict(improvement_pct=8.3)
        assert passed is False
        assert "FAIL" in msg
        assert "8.3" in msg

    def test_exact_threshold_is_pass(self):
        """Exactly 15.0% should be a PASS (>= inclusive)."""
        passed, _ = tvl._build_verdict(improvement_pct=15.0)
        assert passed is True

    def test_negative_improvement_is_fail(self):
        """The transformer LOSING to the LSTM is obviously a FAIL."""
        passed, msg = tvl._build_verdict(improvement_pct=-3.0)
        assert passed is False
        assert "FAIL" in msg


# ═════════════════════════════════════════════════════════════════════════════
# Seed determinism
# ═════════════════════════════════════════════════════════════════════════════


class TestSeed:
    def test_set_seed_is_deterministic(self):
        """Calling _set_seed twice with the same seed should produce the
        same next random draw from python/numpy/torch."""
        tvl._set_seed(42)
        py_a = random.random()
        np_a = np.random.rand()
        th_a = torch.rand(1).item()

        tvl._set_seed(42)
        py_b = random.random()
        np_b = np.random.rand()
        th_b = torch.rand(1).item()

        assert py_a == py_b
        assert np_a == np_b
        assert math.isclose(th_a, th_b, rel_tol=0, abs_tol=0)


# ═════════════════════════════════════════════════════════════════════════════
# Metric JSON shape (mock inputs)
# ═════════════════════════════════════════════════════════════════════════════


class TestMetricSerialization:
    """Simulate what ``main()`` assembles and verify the JSON shape.

    We construct a mock ``out_json`` that mirrors the production dict,
    write it, re-read it, and check every field the spec demands.
    """

    @pytest.fixture
    def mock_result(self) -> dict:
        return {
            "seed": 42,
            "device": "cpu",
            "n_train_sequences": 45000,
            "n_val_sequences": 5000,
            "n_test_sequences": 5000,
            "train_range": "2015-2022",
            "val_range": "2023",
            "test_range": "2024",
            "max_train_games": 2000,
            "max_val_games": 500,
            "max_test_games": 500,
            "epochs": 5,
            "batch_size": 32,
            "lr": 0.001,
            "leakage_audit": {
                "shared_game_pks": 0,
                "shared_pitcher_ids_train_test": 320,
                "n_train_game_pks": 2000,
                "n_val_game_pks": 500,
                "n_test_game_pks": 500,
            },
            "pitchgpt": {
                "params": 1398562,
                "train_loss_final": 5.12,
                "val_loss_final": 5.43,
                "test_loss": 5.47,
                "test_perplexity": 5.60,
                "epoch_best": 4,
                "best_val_loss": 5.40,
                "wall_clock_sec": 412.3,
                "history": [
                    {"epoch": 1, "train_loss": 6.0, "val_loss": 5.9,
                     "train_perplexity": 403.4, "val_perplexity": 365.0,
                     "wall_clock_sec": 80.2},
                ],
            },
            "lstm": {
                "params": 837154,
                "train_loss_final": 5.45,
                "val_loss_final": 5.71,
                "test_loss": 5.74,
                "test_perplexity": 6.11,
                "epoch_best": 4,
                "best_val_loss": 5.70,
                "wall_clock_sec": 320.1,
                "history": [
                    {"epoch": 1, "train_loss": 6.2, "val_loss": 6.1,
                     "train_perplexity": 492.7, "val_perplexity": 445.9,
                     "wall_clock_sec": 65.0},
                ],
            },
            "perplexity_improvement_pct": 8.3,
            "spec_threshold_15pct": False,
            "verdict": "FAIL — transformer only 8.3% better than LSTM; "
                       "spec requires >=15%",
        }

    def test_required_top_level_keys(self, mock_result):
        required = {
            "seed", "n_train_sequences", "n_val_sequences", "n_test_sequences",
            "train_range", "val_range", "test_range", "max_train_games",
            "pitchgpt", "lstm", "perplexity_improvement_pct",
            "spec_threshold_15pct", "verdict",
        }
        assert required <= set(mock_result.keys())

    def test_per_model_shape(self, mock_result):
        for key in ("pitchgpt", "lstm"):
            sub = mock_result[key]
            for field in ("params", "train_loss_final", "val_loss_final",
                          "test_perplexity", "epoch_best"):
                assert field in sub, f"{key}.{field} missing"

    def test_roundtrip_via_json(self, mock_result, tmp_path):
        """Writing + reading JSON should preserve every field."""
        path = tmp_path / "metrics.json"
        path.write_text(json.dumps(mock_result, indent=2), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == mock_result

    def test_verdict_matches_threshold(self, mock_result):
        """The pre-computed verdict should agree with the threshold helper."""
        improvement = mock_result["perplexity_improvement_pct"]
        expected_pass, expected_msg = tvl._build_verdict(improvement)
        assert expected_pass == mock_result["spec_threshold_15pct"]
        # The verdict string prefix must match PASS / FAIL.
        prefix = "PASS" if expected_pass else "FAIL"
        assert mock_result["verdict"].startswith(prefix)


# ═════════════════════════════════════════════════════════════════════════════
# Training-curves HTML writer — smoke test only, no training
# ═════════════════════════════════════════════════════════════════════════════


class TestTrainingCurvesHtml:
    def test_writes_html_file(self, tmp_path):
        hist_tx = [
            {"epoch": 1, "train_loss": 6.0, "val_loss": 5.9,
             "train_perplexity": 403.4, "val_perplexity": 365.0,
             "wall_clock_sec": 80.0},
            {"epoch": 2, "train_loss": 5.5, "val_loss": 5.6,
             "train_perplexity": 244.7, "val_perplexity": 270.4,
             "wall_clock_sec": 80.0},
        ]
        hist_ls = [
            {"epoch": 1, "train_loss": 6.2, "val_loss": 6.1,
             "train_perplexity": 492.7, "val_perplexity": 445.9,
             "wall_clock_sec": 65.0},
            {"epoch": 2, "train_loss": 5.8, "val_loss": 5.9,
             "train_perplexity": 330.3, "val_perplexity": 365.0,
             "wall_clock_sec": 65.0},
        ]
        out = tmp_path / "curves.html"
        tvl._write_training_curves_html(out, hist_tx, hist_ls)
        assert out.exists()
        # File should be non-empty — whether plotly is installed or not
        # we emit at least a fallback HTML stub.
        assert out.stat().st_size > 0
