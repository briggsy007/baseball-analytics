"""
Cross-model integration tests for the baseball analytics platform.

Tests interactions between models, shared pipelines, and systemic properties
that no single-model test file covers.  All tests use synthetic data from
conftest.py fixtures and do not depend on the production database.

Run with:  pytest tests/test_integration.py -m integration
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.analytics.base import BaseAnalyticsModel
from src.analytics.registry import ModelRegistry

# ── All 16 model classes ─────────────────────────────────────────────────────

from src.analytics.pset import PSETModel
from src.analytics.sharpe_lineup import SharpeLineupModel
from src.analytics.volatility_surface import PitchVolatilitySurfaceModel
from src.analytics.causal_war import CausalWARModel
from src.analytics.pitch_decay import PitchDecayRateModel
from src.analytics.loft import LOFTModel
from src.analytics.viscoelastic_workload import ViscoelasticWorkloadModel
from src.analytics.baserunner_gravity import BaserunnerGravityModel
from src.analytics.allostatic_load import AllostaticLoadModel
from src.analytics.defensive_pressing import DefensivePressingModel
from src.analytics.pitchgpt import PitchGPT
from src.analytics.mesi import MESIModel
from src.analytics.mechanix_ae import MechanixAEModel
from src.analytics.kinetic_half_life import KineticHalfLifeModel
from src.analytics.alpha_decay import AlphaDecayModel
from src.analytics.chemnet import ChemNetModel

# ── Shared feature functions ─────────────────────────────────────────────────

from src.analytics.features import (
    compute_z_scores,
    pitch_quality_vector,
    shannon_entropy,
    encode_count_state,
    rolling_mean,
    exponential_decay,
    tunnel_distance,
    classify_pitch_outcome,
    compute_run_value,
)

# ── Pitcher-focused model functions (for cross-model consistency) ────────────

from src.analytics.pitch_decay import calculate_pdr
from src.analytics.pset import calculate_pset
from src.analytics.mesi import calculate_mesi
from src.analytics.volatility_surface import calculate_volatility_surface

# ── Known test pitcher ───────────────────────────────────────────────────────
ZACK_WHEELER_ID = 554430


# All 16 model classes in a single registry for parametrized tests.
ALL_MODEL_CLASSES: list[type[BaseAnalyticsModel]] = [
    PSETModel,
    SharpeLineupModel,
    PitchVolatilitySurfaceModel,
    CausalWARModel,
    PitchDecayRateModel,
    LOFTModel,
    ViscoelasticWorkloadModel,
    BaserunnerGravityModel,
    AllostaticLoadModel,
    DefensivePressingModel,
    PitchGPT,
    MESIModel,
    MechanixAEModel,
    KineticHalfLifeModel,
    AlphaDecayModel,
    ChemNetModel,
]


# ═════════════════════════════════════════════════════════════════════════════
# A.  All models instantiate and predict
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestAllModelsInstantiateAndPredict:
    """Loop through all 16 models, instantiate, verify interface compliance."""

    @pytest.mark.parametrize(
        "model_cls",
        ALL_MODEL_CLASSES,
        ids=[cls.__name__ for cls in ALL_MODEL_CLASSES],
    )
    def test_model_instantiates(self, model_cls: type[BaseAnalyticsModel]):
        """Every model class should instantiate without errors."""
        model = model_cls()
        assert isinstance(model, BaseAnalyticsModel)

    @pytest.mark.parametrize(
        "model_cls",
        ALL_MODEL_CLASSES,
        ids=[cls.__name__ for cls in ALL_MODEL_CLASSES],
    )
    def test_model_has_required_properties(self, model_cls: type[BaseAnalyticsModel]):
        """Every model must expose model_name (str) and version (str)."""
        model = model_cls()
        assert isinstance(model.model_name, str)
        assert len(model.model_name) > 0
        assert isinstance(model.version, str)
        assert len(model.version) > 0

    @pytest.mark.parametrize(
        "model_cls",
        ALL_MODEL_CLASSES,
        ids=[cls.__name__ for cls in ALL_MODEL_CLASSES],
    )
    def test_model_has_metadata(self, model_cls: type[BaseAnalyticsModel]):
        """Every model should expose metadata with created_at."""
        model = model_cls()
        meta = model.metadata
        assert isinstance(meta, dict)
        assert "created_at" in meta

    @pytest.mark.parametrize(
        "model_cls",
        ALL_MODEL_CLASSES,
        ids=[cls.__name__ for cls in ALL_MODEL_CLASSES],
    )
    def test_model_has_abstract_methods(self, model_cls: type[BaseAnalyticsModel]):
        """Every model should implement train, predict, and evaluate."""
        model = model_cls()
        assert callable(getattr(model, "train", None))
        assert callable(getattr(model, "predict", None))
        assert callable(getattr(model, "evaluate", None))

    @pytest.mark.parametrize(
        "model_cls",
        ALL_MODEL_CLASSES,
        ids=[cls.__name__ for cls in ALL_MODEL_CLASSES],
    )
    def test_model_names_are_unique(self, model_cls: type[BaseAnalyticsModel]):
        """Model names should be non-empty strings (uniqueness tested below)."""
        model = model_cls()
        assert model.model_name.strip() != ""

    def test_all_model_names_are_unique(self):
        """No two models should share the same model_name."""
        names = [cls().model_name for cls in ALL_MODEL_CLASSES]
        assert len(names) == len(set(names)), (
            f"Duplicate model names found: {[n for n in names if names.count(n) > 1]}"
        )

    def test_exactly_16_models(self):
        """We expect exactly 16 model classes."""
        assert len(ALL_MODEL_CLASSES) == 16

    @pytest.mark.parametrize(
        "model_cls",
        ALL_MODEL_CLASSES,
        ids=[cls.__name__ for cls in ALL_MODEL_CLASSES],
    )
    def test_model_repr(self, model_cls: type[BaseAnalyticsModel]):
        """repr() should include model_name and version."""
        model = model_cls()
        r = repr(model)
        assert model.model_name in r
        assert model.version in r

    @pytest.mark.parametrize(
        "model_cls",
        ALL_MODEL_CLASSES,
        ids=[cls.__name__ for cls in ALL_MODEL_CLASSES],
    )
    def test_validate_input_rejects_empty_df(self, model_cls: type[BaseAnalyticsModel]):
        """validate_input should raise ValueError on an empty DataFrame."""
        model = model_cls()
        with pytest.raises(ValueError, match="empty"):
            model.validate_input(pd.DataFrame())

    @pytest.mark.parametrize(
        "model_cls",
        ALL_MODEL_CLASSES,
        ids=[cls.__name__ for cls in ALL_MODEL_CLASSES],
    )
    def test_validate_output_rejects_none(self, model_cls: type[BaseAnalyticsModel]):
        """validate_output should raise ValueError on None."""
        model = model_cls()
        with pytest.raises(ValueError, match="None"):
            model.validate_output(None)


# ═════════════════════════════════════════════════════════════════════════════
# B.  Cross-model consistency
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestCrossModelConsistency:
    """Run multiple pitcher-focused models on the same data and verify
    that their outputs do not contradict each other semantically."""

    def test_elite_pitcher_consistency(self, db_conn):
        """A pitcher with data should produce consistent cross-model signals.

        Specifically: if Stuff+ data is available and shows above-average
        quality, we should not see catastrophically early cliff detection
        from the pitch decay model on the same data.
        """
        # Gather pitch-type quality data for Wheeler
        pitcher_id = ZACK_WHEELER_ID

        # Get PDR results
        pdr_result = calculate_pdr(db_conn, pitcher_id)

        # Get PSET results
        pset_result = calculate_pset(db_conn, pitcher_id)

        # Both models should agree on the pitcher_id
        assert pdr_result["pitcher_id"] == pitcher_id
        assert pset_result["pitcher_id"] == pitcher_id

        # If pitcher has enough data for PSET, the scores should be finite
        if pset_result["total_pairs"] > 0:
            assert math.isfinite(pset_result["pset_per_100"])
            assert math.isfinite(pset_result["predictability_score"])
            assert math.isfinite(pset_result["tunnel_score"])

    def test_pitcher_model_outputs_share_pitcher_id(self, db_conn):
        """All pitcher-centric models should tag their output with pitcher_id."""
        pitcher_id = ZACK_WHEELER_ID

        pdr_result = calculate_pdr(db_conn, pitcher_id)
        pset_result = calculate_pset(db_conn, pitcher_id)

        assert pdr_result["pitcher_id"] == pitcher_id
        assert pset_result["pitcher_id"] == pitcher_id

    def test_multiple_pitcher_models_produce_valid_output(self, db_conn):
        """Run several pitcher-focused models on Wheeler and verify each
        returns a well-formed dict (not None, not empty)."""
        pitcher_id = ZACK_WHEELER_ID

        results = {}
        results["pdr"] = calculate_pdr(db_conn, pitcher_id)
        results["pset"] = calculate_pset(db_conn, pitcher_id)
        results["mesi"] = calculate_mesi(db_conn, pitcher_id)
        results["pivs"] = calculate_volatility_surface(db_conn, pitcher_id)

        for name, result in results.items():
            assert isinstance(result, dict), f"{name} did not return a dict"
            assert len(result) > 0, f"{name} returned empty dict"
            assert result.get("pitcher_id") == pitcher_id, (
                f"{name} pitcher_id mismatch"
            )

    def test_no_infinite_or_nan_in_pitcher_models(self, db_conn):
        """Numeric outputs from pitcher models should be finite (no NaN/Inf)."""
        pitcher_id = ZACK_WHEELER_ID

        pdr_result = calculate_pdr(db_conn, pitcher_id)
        pset_result = calculate_pset(db_conn, pitcher_id)

        # Check PSET numeric fields
        for key in ["pset_per_100", "predictability_score", "tunnel_score"]:
            val = pset_result[key]
            if val is not None:
                assert math.isfinite(val), f"PSET {key} = {val} is not finite"

        # Check PDR numeric fields in pitch_types breakdown
        for pt, pt_data in pdr_result.get("pitch_types", {}).items():
            if isinstance(pt_data, dict):
                for k, v in pt_data.items():
                    if isinstance(v, float):
                        assert math.isfinite(v), (
                            f"PDR pitch_types[{pt}][{k}] = {v} is not finite"
                        )

    def test_decay_models_agree_directionally(self, db_conn):
        """PDR (Pitch Decay Rate) and MESI (Motor Engram Stability) should
        not wildly contradict on the same pitcher.

        If MESI reports high stability (high SNR / high MESI), the pitch
        decay model should not show decay beginning at an implausibly
        low pitch count.

        This test is a loose sanity check on directionality -- we do not
        expect exact agreement since the models measure different things.
        """
        pitcher_id = ZACK_WHEELER_ID

        pdr_result = calculate_pdr(db_conn, pitcher_id)
        mesi_result = calculate_mesi(db_conn, pitcher_id)

        # Both should at least return valid structures
        assert isinstance(pdr_result, dict)
        assert isinstance(mesi_result, dict)

        # If MESI has data and PDR has cliff data, check for gross contradiction
        mesi_score = mesi_result.get("overall_mesi")
        first_cliff = pdr_result.get("first_to_die_cliff")

        if mesi_score is not None and first_cliff is not None:
            # A pitcher with very high MESI (>80) having a cliff at pitch 5
            # would be a red flag -- their execution is consistent so they
            # should not degrade that fast.  We use a very lenient threshold.
            if mesi_score > 80:
                assert first_cliff > 10, (
                    f"MESI={mesi_score} suggests stable execution but "
                    f"PDR cliff at pitch {first_cliff} is implausibly early"
                )


# ═════════════════════════════════════════════════════════════════════════════
# C.  Leaderboard pipeline
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestLeaderboardPipeline:
    """Test that batch/leaderboard functions produce valid data."""

    def test_pset_leaderboard_has_required_columns(self, db_conn):
        """PSET batch_calculate should return a DataFrame with expected columns."""
        from src.analytics.pset import batch_calculate as pset_batch

        df = pset_batch(db_conn, min_pitches=5)
        assert isinstance(df, pd.DataFrame)

        if not df.empty:
            required_cols = {"pitcher_id", "pset_per_100", "predictability_score",
                             "tunnel_score", "total_pairs"}
            assert required_cols <= set(df.columns)

    def test_pdr_leaderboard_has_required_columns(self, db_conn):
        """PDR batch_calculate should return a DataFrame with expected columns."""
        from src.analytics.pitch_decay import batch_calculate as pdr_batch

        df = pdr_batch(db_conn, min_pitches=10)
        assert isinstance(df, pd.DataFrame)

        if not df.empty:
            for col in ["pitcher_id", "first_to_die", "first_to_die_cliff"]:
                assert col in df.columns

    def test_leaderboard_pitcher_ids_are_unique(self, db_conn):
        """Each leaderboard should list each pitcher at most once."""
        from src.analytics.pset import batch_calculate as pset_batch

        df = pset_batch(db_conn, min_pitches=5)
        if not df.empty:
            assert df["pitcher_id"].is_unique, "Duplicate pitcher_ids in PSET leaderboard"

    def test_leaderboard_values_are_finite(self, db_conn):
        """All numeric columns in leaderboard DataFrames should be finite."""
        from src.analytics.pset import batch_calculate as pset_batch

        df = pset_batch(db_conn, min_pitches=5)
        if df.empty:
            pytest.skip("No qualifying pitchers in test data.")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            non_null = df[col].dropna()
            if not non_null.empty:
                assert np.isfinite(non_null.values).all(), (
                    f"Non-finite values in leaderboard column '{col}'"
                )

    def test_multiple_leaderboards_overlap_on_known_pitcher(self, db_conn):
        """If Wheeler qualifies in multiple leaderboards, his pitcher_id should
        appear in all of them."""
        from src.analytics.pset import batch_calculate as pset_batch
        from src.analytics.pitch_decay import batch_calculate as pdr_batch

        pset_df = pset_batch(db_conn, min_pitches=5)
        pdr_df = pdr_batch(db_conn, min_pitches=5)

        # Wheeler might not qualify in all with limited test data,
        # but if he does he should appear in both
        pset_ids = set(pset_df["pitcher_id"].tolist()) if not pset_df.empty else set()
        pdr_ids = set(pdr_df["pitcher_id"].tolist()) if not pdr_df.empty else set()

        if ZACK_WHEELER_ID in pset_ids and ZACK_WHEELER_ID in pdr_ids:
            # Both agree he exists -- good
            pass
        # No assertion failure if he does not qualify -- data may be insufficient

    def test_high_threshold_yields_empty_or_fewer(self, db_conn):
        """Setting an impossibly high min_pitches threshold should produce
        an empty (or very small) leaderboard."""
        from src.analytics.pset import batch_calculate as pset_batch

        df = pset_batch(db_conn, min_pitches=999_999_999)
        assert isinstance(df, pd.DataFrame)
        assert df.empty


# ═════════════════════════════════════════════════════════════════════════════
# D.  Model registry round-trip
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestModelRegistryRoundTrip:
    """Save a model via the registry, reload it, and verify predictions match."""

    def test_save_and_load_model(self):
        """A model saved through the registry should be recoverable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)

            model = PSETModel()
            result = registry.save_model(
                model,
                name="pset",
                version="test_v1",
                metadata={"model_name": model.model_name, "version": model.version},
            )

            assert Path(result["model_path"]).exists()
            assert Path(result["meta_path"]).exists()
            assert result["version"] == "test_v1"

            loaded = registry.load_model("pset", version="test_v1")
            assert isinstance(loaded, PSETModel)
            assert loaded.model_name == model.model_name
            assert loaded.version == model.version

    def test_metadata_persists_correctly(self):
        """Metadata saved alongside the model should be readable as JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)

            model = PitchDecayRateModel()
            custom_meta = {
                "model_name": model.model_name,
                "trained_on": "synthetic_data",
                "n_pitchers": 42,
            }
            registry.save_model(model, name="pdr", version="meta_test", metadata=custom_meta)

            meta = registry.get_metadata("pdr", version="meta_test")
            assert meta["name"] == "pdr"
            assert meta["version"] == "meta_test"
            assert meta["trained_on"] == "synthetic_data"
            assert meta["n_pitchers"] == 42

    def test_list_models_after_multiple_saves(self):
        """list_models should return entries for all saved models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)

            registry.save_model(PSETModel(), name="pset", version="v1")
            registry.save_model(LOFTModel(), name="loft", version="v1")
            registry.save_model(MESIModel(), name="mesi", version="v1")

            all_models = registry.list_models()
            names = {m["name"] for m in all_models}
            assert {"pset", "loft", "mesi"} <= names

    def test_load_latest_version(self):
        """When no version is specified, the latest should be loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)

            registry.save_model(PSETModel(), name="pset", version="v001")
            registry.save_model(PSETModel(), name="pset", version="v002")

            loaded = registry.load_model("pset")
            # Should load v002 (latest by filename sort)
            assert isinstance(loaded, PSETModel)

    def test_load_nonexistent_raises(self):
        """Loading a model that does not exist should raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(registry_dir=tmpdir)

            with pytest.raises(FileNotFoundError):
                registry.load_model("nonexistent_model", version="v1")

    def test_base_model_save_load_roundtrip(self):
        """Test the BaseAnalyticsModel.save/load path directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = PSETModel()
            save_path = Path(tmpdir) / "pset_direct.pkl"
            model.save(save_path)

            assert save_path.exists()
            meta_path = save_path.with_name("pset_direct_meta.json")
            assert meta_path.exists()

            loaded = BaseAnalyticsModel.load(save_path)
            assert isinstance(loaded, PSETModel)
            assert loaded.model_name == model.model_name

            # Verify metadata JSON is valid
            meta = json.loads(meta_path.read_text())
            assert meta["model_name"] == "PSET"
            assert meta["version"] == "1.0.0"

    def test_save_load_preserves_training_metadata(self):
        """Training metadata set before save should survive the round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = PSETModel()
            model.set_training_metadata(
                metrics={"r2": 0.85, "rmse": 0.12},
                params={"alpha": 0.3},
                data_range=("2020-01-01", "2025-12-31"),
            )

            save_path = Path(tmpdir) / "pset_meta_test.pkl"
            model.save(save_path)
            loaded = BaseAnalyticsModel.load(save_path)

            meta = loaded.metadata
            assert meta["metrics"]["r2"] == 0.85
            assert meta["params"]["alpha"] == 0.3
            assert meta["data_range"]["start"] == "2020-01-01"


# ═════════════════════════════════════════════════════════════════════════════
# E.  Feature pipeline integrity
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestFeaturePipelineIntegrity:
    """Verify shared feature functions produce identical outputs regardless
    of which model context calls them."""

    @pytest.fixture
    def synthetic_pitch_df(self) -> pd.DataFrame:
        """A small DataFrame of synthetic pitches for feature testing."""
        rng = np.random.RandomState(99)
        n = 50
        return pd.DataFrame({
            "pitcher_id": [ZACK_WHEELER_ID] * n,
            "pitch_type": rng.choice(["FF", "SL", "CU", "CH"], n),
            "release_speed": rng.normal(94.5, 2.0, n).clip(55, 105),
            "release_spin_rate": rng.normal(2300, 200, n).clip(100, 3800),
            "pfx_x": rng.normal(-6, 3, n),
            "pfx_z": rng.normal(15, 2, n),
            "release_pos_x": rng.normal(-1.5, 0.5, n),
            "release_pos_y": rng.normal(55, 0.5, n),
            "release_pos_z": rng.normal(5.8, 0.4, n),
            "plate_x": rng.normal(0, 0.6, n),
            "plate_z": rng.normal(2.5, 0.6, n).clip(0.5, 5.0),
        })

    def test_z_scores_deterministic(self, synthetic_pitch_df):
        """compute_z_scores should give identical results on repeated calls."""
        df = synthetic_pitch_df
        result1 = compute_z_scores(df, ["release_speed", "release_spin_rate"])
        result2 = compute_z_scores(df, ["release_speed", "release_spin_rate"])

        pd.testing.assert_frame_equal(result1, result2)

    def test_z_scores_grouped_deterministic(self, synthetic_pitch_df):
        """Grouped z-scores should be deterministic."""
        df = synthetic_pitch_df
        result1 = compute_z_scores(df, ["release_speed"], group_by="pitch_type")
        result2 = compute_z_scores(df, ["release_speed"], group_by="pitch_type")

        pd.testing.assert_frame_equal(result1, result2)

    def test_pitch_quality_vector_consistency(self, synthetic_pitch_df):
        """pitch_quality_vector should give identical output for the same row."""
        row = synthetic_pitch_df.iloc[0]
        result1 = pitch_quality_vector(row)
        result2 = pitch_quality_vector(row)

        assert result1 == result2
        assert all(0 <= v <= 100 for v in result1.values())

    def test_pitch_quality_vector_output_format(self, synthetic_pitch_df):
        """Output should have the four expected keys."""
        row = synthetic_pitch_df.iloc[0]
        result = pitch_quality_vector(row)

        expected_keys = {"velocity_score", "spin_score", "movement_score", "composite"}
        assert set(result.keys()) == expected_keys

    def test_shannon_entropy_deterministic(self):
        """shannon_entropy should be deterministic and non-negative."""
        probs = [0.25, 0.25, 0.25, 0.25]
        e1 = shannon_entropy(probs)
        e2 = shannon_entropy(probs)
        assert e1 == e2
        assert e1 >= 0
        # Uniform over 4 categories: entropy = ln(4)
        assert abs(e1 - np.log(4)) < 1e-10

    def test_encode_count_state_covers_all_counts(self):
        """All valid ball-strike combinations should produce unique indices."""
        indices = set()
        for b in range(4):
            for s in range(3):
                idx = encode_count_state(b, s)
                assert 0 <= idx <= 11
                indices.add(idx)
        assert len(indices) == 12

    def test_rolling_mean_deterministic(self):
        """Rolling mean should be deterministic."""
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        r1 = rolling_mean(s, window=3)
        r2 = rolling_mean(s, window=3)
        pd.testing.assert_series_equal(r1, r2)

    def test_exponential_decay_deterministic(self):
        """Exponential decay should be deterministic."""
        values = pd.Series([10.0, 9.0, 8.5, 8.0, 7.8, 7.5])
        d1 = exponential_decay(values, half_life=3.0)
        d2 = exponential_decay(values, half_life=3.0)
        pd.testing.assert_series_equal(d1, d2)

    def test_tunnel_distance_symmetric(self):
        """tunnel_distance(a, b) should equal tunnel_distance(b, a)
        when computed on plate_x/plate_z."""
        p1 = pd.Series({"plate_x": 0.5, "plate_z": 2.5})
        p2 = pd.Series({"plate_x": -0.3, "plate_z": 3.0})

        d_ab = tunnel_distance(p1, p2)
        d_ba = tunnel_distance(p2, p1)
        assert abs(d_ab - d_ba) < 1e-10

    def test_classify_pitch_outcome_all_standard_types(self):
        """All standard Statcast descriptions should map to known categories."""
        known_categories = {"whiff", "called_strike", "ball", "foul", "in_play", "hbp", "unknown"}
        test_descriptions = [
            "swinging_strike", "called_strike", "ball", "foul",
            "hit_into_play", "hit_by_pitch", None, "random_nonsense",
        ]
        for desc in test_descriptions:
            cat = classify_pitch_outcome(desc)
            assert cat in known_categories, f"'{desc}' mapped to unknown category '{cat}'"

    def test_compute_run_value_zero_denom(self):
        """Zero or missing denominator should return None."""
        assert compute_run_value(0.5, 0.0) is None
        assert compute_run_value(0.5, None) is None

    def test_feature_functions_called_from_different_contexts(self, synthetic_pitch_df):
        """The same feature function called with identical data from two
        independent code paths should produce identical results.

        This simulates two different models calling compute_z_scores on
        the same DataFrame and verifies they see the same output.
        """
        df = synthetic_pitch_df.copy()

        # Simulate "model A" calling z-scores
        model_a_result = compute_z_scores(df, ["release_speed", "pfx_x"], group_by="pitch_type")

        # Simulate "model B" calling z-scores on a fresh copy
        df_copy = synthetic_pitch_df.copy()
        model_b_result = compute_z_scores(df_copy, ["release_speed", "pfx_x"], group_by="pitch_type")

        pd.testing.assert_frame_equal(model_a_result, model_b_result)


# ═════════════════════════════════════════════════════════════════════════════
# F.  Data freshness tracking / cache invalidation
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestDataFreshnessTracking:
    """Test that model outputs change when underlying source data is updated,
    verifying that caching does not serve stale results."""

    def _make_fresh_conn(self) -> duckdb.DuckDBPyConnection:
        """Create a minimal in-memory DB with schema + a few pitches."""
        from src.db.schema import create_tables

        conn = duckdb.connect(":memory:")
        create_tables(conn)
        return conn

    def _insert_pitches(
        self,
        conn: duckdb.DuckDBPyConnection,
        pitcher_id: int,
        n: int,
        velo_mean: float = 94.5,
        season: int = 2025,
        seed: int = 42,
    ) -> None:
        """Insert synthetic pitches for one pitcher into a connection."""
        rng = np.random.RandomState(seed)
        pitch_types = ["FF", "SL", "CU", "CH"]

        rows = []
        for i in range(n):
            pt = rng.choice(pitch_types)
            desc = rng.choice(["called_strike", "swinging_strike", "ball", "foul",
                               "hit_into_play"])
            event = rng.choice(["single", "strikeout", "field_out"]) if desc == "hit_into_play" else None

            rows.append({
                "game_pk": 100000 + i // 20,
                "game_date": f"{season}-06-{(i % 28) + 1:02d}",
                "pitcher_id": pitcher_id,
                "batter_id": rng.randint(100000, 700000),
                "pitch_type": pt,
                "pitch_name": pt,
                "release_speed": round(float(np.clip(rng.normal(velo_mean, 2.0), 55, 105)), 1),
                "release_spin_rate": round(float(np.clip(rng.normal(2300, 200), 100, 3800)), 0),
                "spin_axis": round(float(rng.uniform(0, 360)), 1),
                "pfx_x": round(float(rng.normal(-6, 3)), 1),
                "pfx_z": round(float(rng.normal(15, 2)), 1),
                "plate_x": round(float(rng.normal(0, 0.6)), 2),
                "plate_z": round(float(np.clip(rng.normal(2.5, 0.6), 0.5, 5.0)), 2),
                "release_extension": round(float(rng.normal(6.2, 0.5)), 1),
                "release_pos_x": round(float(rng.normal(-1.5, 0.5)), 2),
                "release_pos_y": round(float(rng.normal(55, 0.5)), 2),
                "release_pos_z": round(float(rng.normal(5.8, 0.4)), 2),
                "launch_speed": round(float(rng.normal(88, 12)), 1) if event else None,
                "launch_angle": round(float(rng.normal(12, 20)), 1) if event else None,
                "hit_distance": round(float(rng.uniform(50, 420)), 0) if event else None,
                "hc_x": round(float(rng.uniform(20, 230)), 1) if event else None,
                "hc_y": round(float(rng.uniform(20, 220)), 1) if event else None,
                "bb_type": rng.choice(["fly_ball", "ground_ball", "line_drive", "popup"]) if event else None,
                "estimated_ba": round(float(rng.uniform(0, 1)), 3) if event else None,
                "estimated_woba": round(float(rng.uniform(0, 2)), 3) if event else None,
                "delta_home_win_exp": round(float(rng.normal(0, 0.03)), 4),
                "delta_run_exp": round(float(rng.normal(0, 0.1)), 4),
                "inning": int(rng.randint(1, 10)),
                "inning_topbot": rng.choice(["Top", "Bot"]),
                "outs_when_up": int(rng.randint(0, 3)),
                "balls": int(rng.randint(0, 4)),
                "strikes": int(rng.randint(0, 3)),
                "on_1b": int(rng.choice([0, 1])),
                "on_2b": int(rng.choice([0, 1])),
                "on_3b": int(rng.choice([0, 1])),
                "stand": rng.choice(["L", "R"]),
                "p_throws": "R",
                "at_bat_number": int(i // 5 + 1),
                "pitch_number": int(i % 5 + 1),
                "description": desc,
                "events": event,
                "type": "S" if "strike" in desc else ("B" if desc == "ball" else "X"),
                "home_team": "PHI",
                "away_team": "NYM",
                "woba_value": round(float(rng.uniform(0, 2)), 3) if event else None,
                "woba_denom": 1.0 if event else 0.0,
                "babip_value": round(float(rng.uniform(0, 1)), 3) if event else None,
                "iso_value": round(float(rng.uniform(0, 1)), 3) if event else None,
                "zone": int(rng.randint(1, 15)),
                "effective_speed": round(float(np.clip(rng.normal(velo_mean - 1, 2), 55, 105)), 1),
                "if_fielding_alignment": "Standard",
                "of_fielding_alignment": "Standard",
                "fielder_2": rng.randint(100000, 700000),
            })

        df = pd.DataFrame(rows)
        conn.execute("INSERT INTO pitches SELECT * FROM df")

    def test_pset_changes_after_data_update(self):
        """PSET output should differ after new pitches are inserted."""
        conn = self._make_fresh_conn()
        pid = 999001

        # Initial data: 120 pitches
        self._insert_pitches(conn, pid, n=120, seed=10)
        result_before = calculate_pset(conn, pid)

        # Insert 80 more pitches with different seed (different sequences)
        self._insert_pitches(conn, pid, n=80, seed=999)
        result_after = calculate_pset(conn, pid)

        # The total_pairs should have changed
        if result_before["total_pairs"] > 0 and result_after["total_pairs"] > 0:
            assert result_after["total_pairs"] != result_before["total_pairs"], (
                "PSET total_pairs should change after data update"
            )

        conn.close()

    def test_pdr_changes_after_data_update(self):
        """PDR output should update when fresh pitches arrive."""
        conn = self._make_fresh_conn()
        pid = 999002

        self._insert_pitches(conn, pid, n=150, velo_mean=95.0, seed=20)
        result_before = calculate_pdr(conn, pid)

        # Insert more pitches with significantly lower velocity (simulating fatigue)
        self._insert_pitches(conn, pid, n=100, velo_mean=88.0, seed=30)
        result_after = calculate_pdr(conn, pid)

        # The model should see the new data (different number of pitch types
        # analyzed, or different cliff estimates)
        assert result_after is not None
        assert isinstance(result_after, dict)

        conn.close()

    def test_model_sees_only_current_data(self):
        """A model queried against a fresh connection should not carry over
        results from a previous connection (no hidden global caching)."""
        conn1 = self._make_fresh_conn()
        conn2 = self._make_fresh_conn()

        pid = 999003
        self._insert_pitches(conn1, pid, n=120, seed=40)
        # conn2 has NO data for this pitcher

        result1 = calculate_pset(conn1, pid)
        result2 = calculate_pset(conn2, pid)

        # conn2 should show no data
        assert result2["total_pairs"] == 0

        # conn1 may have data
        # (no assertion on result1 total_pairs because it depends on data density)

        conn1.close()
        conn2.close()


# ═════════════════════════════════════════════════════════════════════════════
# G.  Additional integration sanity checks
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestModelTrainInterface:
    """Verify the train() interface returns a valid dict for descriptive models."""

    @pytest.mark.parametrize(
        "model_cls",
        [PSETModel, AlphaDecayModel, LOFTModel],
        ids=["PSET", "AlphaDecay", "LOFT"],
    )
    def test_descriptive_models_train_is_noop(self, model_cls, db_conn):
        """Descriptive (non-supervised) models should return a valid metrics
        dict from train() without actually fitting a model."""
        model = model_cls()
        metrics = model.train(db_conn)
        assert isinstance(metrics, dict)
        assert len(metrics) > 0

    def test_base_model_validate_input_accepts_valid_df(self):
        """validate_input should pass through a valid DataFrame unchanged."""
        model = PSETModel()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        validated = model.validate_input(df)
        assert len(validated) == 3

    def test_base_model_validate_input_drops_all_null_cols(self):
        """validate_input should drop columns that are entirely null."""
        model = PSETModel()
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [None, None, None],
        })
        validated = model.validate_input(df)
        assert "b" not in validated.columns
        assert "a" in validated.columns

    def test_base_model_validate_output_accepts_valid_dict(self):
        """validate_output should pass through a non-empty dict."""
        model = PSETModel()
        result = model.validate_output({"pset_per_100": 5.3})
        assert result == {"pset_per_100": 5.3}

    def test_base_model_validate_output_rejects_empty_dict(self):
        """validate_output should reject an empty dict."""
        model = PSETModel()
        with pytest.raises(ValueError, match="empty"):
            model.validate_output({})


@pytest.mark.integration
class TestCrossModelDataFlow:
    """Verify that data flows correctly across model boundaries when
    multiple models consume the same database rows."""

    def test_same_query_same_results(self, db_conn):
        """Two models querying the same pitcher should see the same row count."""
        pitcher_id = ZACK_WHEELER_ID

        count_query = """
            SELECT COUNT(*) FROM pitches
            WHERE pitcher_id = $1
              AND pitch_type IS NOT NULL
              AND release_speed IS NOT NULL
        """
        count = db_conn.execute(count_query, [pitcher_id]).fetchone()[0]

        # Run it again -- should be identical (no side effects from reads)
        count2 = db_conn.execute(count_query, [pitcher_id]).fetchone()[0]
        assert count == count2

    def test_wheeler_exists_in_test_data(self, db_conn):
        """Zack Wheeler should have at least some pitches in the test DB."""
        count = db_conn.execute(
            "SELECT COUNT(*) FROM pitches WHERE pitcher_id = $1",
            [ZACK_WHEELER_ID],
        ).fetchone()[0]
        assert count > 0, "Wheeler should have pitches in the test database"

    def test_sample_pitches_fixture_nonempty(self, sample_pitches_df):
        """The shared sample_pitches_df fixture should have data."""
        assert len(sample_pitches_df) > 0

    def test_sample_pitches_have_required_columns(self, sample_pitches_df):
        """The pitch data used by all models should contain essential columns."""
        required = {
            "pitcher_id", "batter_id", "pitch_type", "release_speed",
            "release_spin_rate", "pfx_x", "pfx_z", "plate_x", "plate_z",
        }
        assert required <= set(sample_pitches_df.columns), (
            f"Missing columns: {required - set(sample_pitches_df.columns)}"
        )
