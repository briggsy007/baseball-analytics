"""
Tests for the automated model retraining pipeline.

Covers:
- Dry run doesn't save any models
- Training log is written correctly
- Model comparison logic works (improved / rejected / no-baseline)
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
import sys

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from scripts.retrain import (
    _append_training_log,
    compare_metrics,
    run_retrain,
)


# ---------------------------------------------------------------------------
# compare_metrics tests
# ---------------------------------------------------------------------------


class TestCompareMetrics:
    """Tests for the metric comparison helper."""

    def test_higher_is_better_improvement(self):
        baseline = {"r2_test": 0.75, "n_players": 100}
        candidate = {"r2_test": 0.82, "n_players": 120}
        improved, reason = compare_metrics(baseline, candidate)
        assert improved is True
        assert "improved" in reason.lower() or "2 improved" in reason

    def test_higher_is_better_degradation(self):
        baseline = {"r2_test": 0.85, "coverage": 95.0}
        candidate = {"r2_test": 0.70, "coverage": 80.0}
        improved, reason = compare_metrics(baseline, candidate)
        assert improved is False
        assert "degraded" in reason.lower()

    def test_lower_is_better_improvement(self):
        baseline = {"rmse_test": 1.5, "mae": 1.0}
        candidate = {"rmse_test": 1.2, "mae": 0.8}
        improved, reason = compare_metrics(baseline, candidate)
        assert improved is True

    def test_lower_is_better_degradation(self):
        baseline = {"rmse_test": 0.5}
        candidate = {"rmse_test": 1.5}
        improved, reason = compare_metrics(baseline, candidate)
        assert improved is False

    def test_mixed_metrics_majority_wins(self):
        baseline = {"r2_test": 0.80, "rmse_test": 1.0, "coverage": 90.0}
        candidate = {"r2_test": 0.85, "rmse_test": 0.9, "coverage": 88.0}
        # 2 improved (r2_test, rmse_test) vs 1 degraded (coverage) -> improved
        improved, reason = compare_metrics(baseline, candidate)
        assert improved is True

    def test_no_overlapping_numeric_keys(self):
        baseline = {"some_text": "abc"}
        candidate = {"other_text": "xyz"}
        improved, reason = compare_metrics(baseline, candidate)
        assert improved is True
        assert "no comparable" in reason.lower()

    def test_identical_metrics(self):
        baseline = {"r2_test": 0.80, "rmse_test": 1.0}
        candidate = {"r2_test": 0.80, "rmse_test": 1.0}
        # No improvements, no degradations -> accepted (no comparable changes)
        improved, reason = compare_metrics(baseline, candidate)
        assert improved is True

    def test_empty_dicts(self):
        improved, reason = compare_metrics({}, {})
        assert improved is True

    def test_non_numeric_values_ignored(self):
        baseline = {"r2_test": 0.80, "model_path": "/old/path"}
        candidate = {"r2_test": 0.85, "model_path": "/new/path"}
        improved, reason = compare_metrics(baseline, candidate)
        assert improved is True


# ---------------------------------------------------------------------------
# Training log tests
# ---------------------------------------------------------------------------


class TestTrainingLog:
    """Tests for training log I/O."""

    def test_append_creates_file_and_writes_json(self, tmp_path, monkeypatch):
        log_path = tmp_path / "logs" / "training_log.jsonl"
        monkeypatch.setattr("scripts.retrain.LOGS_DIR", tmp_path / "logs")
        monkeypatch.setattr("scripts.retrain.TRAINING_LOG_PATH", log_path)

        entry = {
            "model": "test_model",
            "timestamp": "2026-04-14T12:00:00+00:00",
            "status": "promoted",
            "metrics_new": {"r2_test": 0.9},
        }
        _append_training_log(entry)

        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["model"] == "test_model"
        assert parsed["status"] == "promoted"

    def test_append_multiple_entries(self, tmp_path, monkeypatch):
        log_path = tmp_path / "logs" / "training_log.jsonl"
        monkeypatch.setattr("scripts.retrain.LOGS_DIR", tmp_path / "logs")
        monkeypatch.setattr("scripts.retrain.TRAINING_LOG_PATH", log_path)

        for i in range(3):
            _append_training_log({"model": f"model_{i}", "iteration": i})

        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 3
        for i, line in enumerate(lines):
            parsed = json.loads(line)
            assert parsed["model"] == f"model_{i}"


# ---------------------------------------------------------------------------
# Dry run tests
# ---------------------------------------------------------------------------


class TestDryRun:
    """Verify that --dry-run never saves models."""

    def test_dry_run_does_not_save_models(self, tmp_path, monkeypatch):
        """Dry run should produce 'dry_run' or 'skipped' status for every model."""
        # Patch _get_conn to return a mock connection
        mock_conn = MagicMock()
        monkeypatch.setattr("scripts.retrain._get_conn", lambda: mock_conn)

        # Point training log to temp dir
        log_path = tmp_path / "logs" / "training_log.jsonl"
        monkeypatch.setattr("scripts.retrain.LOGS_DIR", tmp_path / "logs")
        monkeypatch.setattr("scripts.retrain.TRAINING_LOG_PATH", log_path)

        # Patch _get_registry so load_model raises FileNotFoundError
        mock_registry = MagicMock()
        mock_registry.load_model.side_effect = FileNotFoundError("no model")
        monkeypatch.setattr("scripts.retrain._get_registry", lambda: mock_registry)

        # Patch _is_base_analytics_model
        monkeypatch.setattr("scripts.retrain._is_base_analytics_model", lambda obj: True)

        # Patch importlib.import_module for analytics modules
        fake_model_instance = MagicMock()
        fake_model_instance.evaluate.return_value = {"r2_test": 0.5}
        fake_model_cls = MagicMock(return_value=fake_model_instance)

        fake_module = MagicMock(spec=[])

        # Set all class attributes that might be looked up
        all_class_names = [
            "PitchVolatilitySurfaceModel", "PSETModel", "SharpeLineupModel",
            "DefensivePressingModel", "MESIModel", "KineticHalfLifeModel",
            "AlphaDecayModel", "AllostaticLoadModel", "LOFTModel",
            "BaserunnerGravityModel", "PitchDecayRateModel",
            "ViscoelasticWorkloadModel", "CausalWARModel", "PitchGPT",
            "MechanixAEModel", "ChemNetModel", "train_stuff_model",
        ]
        for attr_name in all_class_names:
            setattr(fake_module, attr_name, fake_model_cls)

        original_import = importlib.import_module

        def patched_import(name, *args, **kwargs):
            if name.startswith("src.analytics."):
                return fake_module
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("scripts.retrain.importlib.import_module", patched_import)

        results = run_retrain(model_filter="all", dry_run=True)

        # Verify no model was saved or promoted
        for r in results:
            assert r["status"] in ("dry_run", "skipped"), (
                f"Model {r['model']} has status {r['status']} during dry run"
            )

        # Verify registry.save_model was never called
        mock_registry.save_model.assert_not_called()

        # Verify training log was written
        assert log_path.exists()
        lines = log_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == len(results)

    def test_dry_run_single_model(self, tmp_path, monkeypatch):
        """Dry run on a single BaseAnalyticsModel model."""
        mock_conn = MagicMock()
        monkeypatch.setattr("scripts.retrain._get_conn", lambda: mock_conn)

        log_path = tmp_path / "logs" / "training_log.jsonl"
        monkeypatch.setattr("scripts.retrain.LOGS_DIR", tmp_path / "logs")
        monkeypatch.setattr("scripts.retrain.TRAINING_LOG_PATH", log_path)

        mock_registry = MagicMock()
        mock_registry.load_model.side_effect = FileNotFoundError("no model")
        monkeypatch.setattr("scripts.retrain._get_registry", lambda: mock_registry)

        monkeypatch.setattr("scripts.retrain._is_base_analytics_model", lambda obj: True)

        fake_model_instance = MagicMock()
        fake_model_instance.evaluate.return_value = {"n_players": 50}
        fake_model_cls = MagicMock(return_value=fake_model_instance)

        fake_module = MagicMock(spec=[])
        setattr(fake_module, "CausalWARModel", fake_model_cls)

        original_import = importlib.import_module

        def patched_import(name, *args, **kwargs):
            if name.startswith("src.analytics."):
                return fake_module
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("scripts.retrain.importlib.import_module", patched_import)

        results = run_retrain(model_filter="causal_war", dry_run=True)

        assert len(results) == 1
        assert results[0]["status"] == "dry_run"
        mock_registry.save_model.assert_not_called()


# ---------------------------------------------------------------------------
# Unknown model name
# ---------------------------------------------------------------------------


class TestUnknownModel:
    def test_unknown_model_returns_empty(self):
        # Unknown model name returns empty results without needing a DB connection
        results = run_retrain(model_filter="nonexistent_model_xyz")
        assert results == []
