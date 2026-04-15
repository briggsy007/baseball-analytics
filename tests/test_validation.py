"""
Tests for src.analytics.validation — Pydantic schemas and decorators.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.analytics.validation import (
    VELOCITY_MIN,
    VELOCITY_MAX,
    SPIN_RATE_MIN,
    SPIN_RATE_MAX,
    PLATE_X_MIN,
    PLATE_X_MAX,
    PLATE_Z_MIN,
    PLATE_Z_MAX,
    GameState,
    ModelPrediction,
    PitchDataFrame,
    PitchRecord,
    validate_pitch_data,
)


# ── PitchRecord ───────────────────────────────────────────────────────────


class TestPitchRecord:
    """Tests for the PitchRecord validation schema."""

    def test_valid_pitch(self):
        """A well-formed pitch record passes validation."""
        record = PitchRecord(
            game_pk=717253,
            game_date="2025-04-15",
            pitcher_id=554430,
            batter_id=660271,
            pitch_type="FF",
            release_speed=96.5,
            release_spin_rate=2350.0,
            plate_x=0.15,
            plate_z=2.8,
        )
        assert record.pitcher_id == 554430
        assert record.pitch_type == "FF"

    def test_optional_fields_default_none(self):
        """Optional physical measurements default to None."""
        record = PitchRecord(
            game_pk=1, game_date="2025-01-01",
            pitcher_id=1, batter_id=2,
        )
        assert record.release_speed is None
        assert record.launch_speed is None

    def test_velocity_below_range(self):
        """Velocity below 50 mph raises ValidationError."""
        with pytest.raises(ValidationError, match="release_speed"):
            PitchRecord(
                game_pk=1, game_date="2025-01-01",
                pitcher_id=1, batter_id=2,
                release_speed=30.0,
            )

    def test_velocity_above_range(self):
        """Velocity above 110 mph raises ValidationError."""
        with pytest.raises(ValidationError, match="release_speed"):
            PitchRecord(
                game_pk=1, game_date="2025-01-01",
                pitcher_id=1, batter_id=2,
                release_speed=115.0,
            )

    def test_spin_rate_below_range(self):
        """Negative spin rate raises ValidationError."""
        with pytest.raises(ValidationError, match="release_spin_rate"):
            PitchRecord(
                game_pk=1, game_date="2025-01-01",
                pitcher_id=1, batter_id=2,
                release_spin_rate=-100.0,
            )

    def test_spin_rate_above_range(self):
        """Spin above 4000 raises ValidationError."""
        with pytest.raises(ValidationError, match="release_spin_rate"):
            PitchRecord(
                game_pk=1, game_date="2025-01-01",
                pitcher_id=1, batter_id=2,
                release_spin_rate=5000.0,
            )

    def test_plate_location_range(self):
        """Out-of-range plate locations fail validation."""
        with pytest.raises(ValidationError, match="plate_x"):
            PitchRecord(
                game_pk=1, game_date="2025-01-01",
                pitcher_id=1, batter_id=2,
                plate_x=-5.0,
            )

    def test_plate_z_below_zero(self):
        """plate_z below 0 fails."""
        with pytest.raises(ValidationError, match="plate_z"):
            PitchRecord(
                game_pk=1, game_date="2025-01-01",
                pitcher_id=1, batter_id=2,
                plate_z=-1.0,
            )

    def test_invalid_handedness(self):
        """Invalid handedness code raises ValidationError."""
        with pytest.raises(ValidationError, match="Handedness"):
            PitchRecord(
                game_pk=1, game_date="2025-01-01",
                pitcher_id=1, batter_id=2,
                stand="S",
            )

    def test_valid_handedness(self):
        """L and R are accepted for handedness."""
        record = PitchRecord(
            game_pk=1, game_date="2025-01-01",
            pitcher_id=1, batter_id=2,
            stand="L", p_throws="R",
        )
        assert record.stand == "L"
        assert record.p_throws == "R"

    def test_unknown_pitch_type_warns(self):
        """Unknown pitch type is accepted but logs a warning."""
        # Should not raise, just warn
        record = PitchRecord(
            game_pk=1, game_date="2025-01-01",
            pitcher_id=1, batter_id=2,
            pitch_type="ZZ",
        )
        assert record.pitch_type == "ZZ"

    def test_balls_strikes_range(self):
        """Balls must be 0-3, strikes must be 0-2."""
        with pytest.raises(ValidationError, match="balls"):
            PitchRecord(
                game_pk=1, game_date="2025-01-01",
                pitcher_id=1, batter_id=2,
                balls=4,
            )
        with pytest.raises(ValidationError, match="strikes"):
            PitchRecord(
                game_pk=1, game_date="2025-01-01",
                pitcher_id=1, batter_id=2,
                strikes=3,
            )

    def test_extra_fields_allowed(self):
        """Extra fields are accepted due to extra='allow'."""
        record = PitchRecord(
            game_pk=1, game_date="2025-01-01",
            pitcher_id=1, batter_id=2,
            custom_field="value",
        )
        assert record.model_extra.get("custom_field") == "value"


# ── GameState ─────────────────────────────────────────────────────────────


class TestGameState:
    """Tests for the GameState validation schema."""

    def test_valid_game_state(self):
        state = GameState(
            inning=5, inning_topbot="Top",
            outs=1, balls=2, strikes=1,
        )
        assert state.inning == 5
        assert state.outs == 1

    def test_inning_range(self):
        """Inning must be 1-18."""
        with pytest.raises(ValidationError, match="inning"):
            GameState(
                inning=0, inning_topbot="Top",
                outs=0, balls=0, strikes=0,
            )
        with pytest.raises(ValidationError, match="inning"):
            GameState(
                inning=19, inning_topbot="Top",
                outs=0, balls=0, strikes=0,
            )

    def test_outs_range(self):
        """Outs must be 0-2."""
        with pytest.raises(ValidationError, match="outs"):
            GameState(
                inning=1, inning_topbot="Top",
                outs=3, balls=0, strikes=0,
            )

    def test_balls_range(self):
        """Balls must be 0-3."""
        with pytest.raises(ValidationError, match="balls"):
            GameState(
                inning=1, inning_topbot="Top",
                outs=0, balls=4, strikes=0,
            )

    def test_strikes_range(self):
        """Strikes must be 0-2."""
        with pytest.raises(ValidationError, match="strikes"):
            GameState(
                inning=1, inning_topbot="Top",
                outs=0, balls=0, strikes=3,
            )

    def test_invalid_inning_topbot(self):
        """inning_topbot must be 'Top' or 'Bot'."""
        with pytest.raises(ValidationError, match="inning_topbot"):
            GameState(
                inning=1, inning_topbot="Middle",
                outs=0, balls=0, strikes=0,
            )

    def test_base_runners_default_false(self):
        state = GameState(
            inning=1, inning_topbot="Top",
            outs=0, balls=0, strikes=0,
        )
        assert state.on_1b is False
        assert state.on_2b is False
        assert state.on_3b is False

    def test_extra_innings_valid(self):
        """18th inning is valid (extended game)."""
        state = GameState(
            inning=18, inning_topbot="Bot",
            outs=2, balls=3, strikes=2,
        )
        assert state.inning == 18


# ── ModelPrediction ───────────────────────────────────────────────────────


class TestModelPrediction:
    """Tests for the ModelPrediction base schema."""

    def test_valid_prediction(self):
        pred = ModelPrediction(
            model_name="stuff_plus",
            model_version="1.0.0",
            generated_at="2025-04-15T12:00:00Z",
            prediction={"overall_stuff_plus": 112.5},
        )
        assert pred.model_name == "stuff_plus"

    def test_confidence_range(self):
        """Confidence must be 0-1."""
        with pytest.raises(ValidationError, match="confidence"):
            ModelPrediction(
                model_name="test",
                model_version="1.0",
                generated_at="2025-01-01",
                prediction=42,
                confidence=1.5,
            )

    def test_extra_fields_allowed(self):
        pred = ModelPrediction(
            model_name="test",
            model_version="1.0",
            generated_at="2025-01-01",
            prediction=42,
            extra_metric=0.95,
        )
        assert pred.model_extra.get("extra_metric") == 0.95


# ── PitchDataFrame ────────────────────────────────────────────────────────


class TestPitchDataFrame:
    """Tests for bulk DataFrame validation."""

    def _make_df(self, n: int = 10, **overrides) -> pd.DataFrame:
        """Helper to create a minimal valid pitch DataFrame."""
        data = {
            "pitcher_id": np.random.randint(100000, 700000, n),
            "batter_id": np.random.randint(100000, 700000, n),
            "game_pk": np.random.randint(100000, 999999, n),
            "release_speed": np.random.normal(93, 3, n),
        }
        data.update(overrides)
        return pd.DataFrame(data)

    def test_valid_dataframe(self):
        df = self._make_df()
        result = PitchDataFrame.from_dataframe(df)
        assert result.n_rows == 10
        assert "pitcher_id" in result.column_names

    def test_empty_dataframe_fails(self):
        df = pd.DataFrame(columns=["pitcher_id", "batter_id", "game_pk"])
        with pytest.raises(ValidationError, match="n_rows"):
            PitchDataFrame.from_dataframe(df)

    def test_missing_required_column(self):
        df = pd.DataFrame({"pitcher_id": [1], "some_col": [2]})
        with pytest.raises(ValidationError, match="Missing required columns"):
            PitchDataFrame.from_dataframe(df)

    def test_custom_required_columns(self):
        df = pd.DataFrame({"release_speed": [95.0], "pitcher_id": [1]})
        # Should pass with custom required columns
        result = PitchDataFrame.from_dataframe(
            df, required_columns=["release_speed"]
        )
        assert result.n_rows == 1

    def test_high_null_rate_fails(self):
        df = pd.DataFrame({
            "pitcher_id": [None] * 10,
            "batter_id": range(10),
            "game_pk": range(10),
        })
        with pytest.raises(ValidationError, match="null rate"):
            PitchDataFrame.from_dataframe(df, max_null_rate=0.5)

    def test_null_rate_in_non_required_column_ok(self):
        df = self._make_df()
        df["launch_speed"] = None  # All nulls in optional column
        result = PitchDataFrame.from_dataframe(df)
        assert result.null_rates["launch_speed"] == 1.0


# ── validate_pitch_data decorator ────────────────────────────────────────


class TestValidatePitchDataDecorator:
    """Tests for the @validate_pitch_data decorator."""

    def _make_df(self, n: int = 10) -> pd.DataFrame:
        return pd.DataFrame({
            "pitcher_id": np.random.randint(100000, 700000, n),
            "batter_id": np.random.randint(100000, 700000, n),
            "game_pk": np.random.randint(100000, 999999, n),
            "release_speed": np.random.normal(93, 3, n),
        })

    def test_decorator_passes_valid_df(self):
        @validate_pitch_data()
        def my_func(df: pd.DataFrame) -> int:
            return len(df)

        df = self._make_df()
        assert my_func(df) == 10

    def test_decorator_fails_empty_df(self):
        @validate_pitch_data(min_rows=1)
        def my_func(df: pd.DataFrame) -> int:
            return len(df)

        df = pd.DataFrame(columns=["pitcher_id", "batter_id", "game_pk"])
        with pytest.raises(ValueError, match="rows"):
            my_func(df)

    def test_decorator_fails_missing_columns(self):
        @validate_pitch_data(required_columns=["release_speed", "pitch_type"])
        def my_func(df: pd.DataFrame) -> int:
            return len(df)

        df = pd.DataFrame({"release_speed": [95.0]})
        with pytest.raises(ValidationError, match="Missing required columns"):
            my_func(df)

    def test_decorator_accepts_kwargs(self):
        @validate_pitch_data()
        def my_func(conn, data: pd.DataFrame) -> int:
            return len(data)

        df = self._make_df()
        assert my_func(None, data=df) == 10

    def test_decorator_no_dataframe_raises(self):
        @validate_pitch_data()
        def my_func(x: int) -> int:
            return x

        with pytest.raises(ValueError, match="No DataFrame"):
            my_func(42)

    def test_decorator_min_rows(self):
        @validate_pitch_data(min_rows=5)
        def my_func(df: pd.DataFrame) -> int:
            return len(df)

        df = self._make_df(n=3)
        with pytest.raises(ValueError, match="3 rows.*need at least 5"):
            my_func(df)


# ── Integration with real sampled data ───────────────────────────────────


class TestValidationWithRealData:
    """Tests that use the sampled test database."""

    def test_sample_pitches_validate_as_dataframe(self, sample_pitches_df):
        """The test fixture data passes PitchDataFrame validation."""
        result = PitchDataFrame.from_dataframe(
            sample_pitches_df,
            required_columns=["pitcher_id", "batter_id", "game_pk"],
        )
        assert result.n_rows > 0

    def test_sample_pitch_records_validate(self, sample_pitches_df):
        """Spot-check that individual rows pass PitchRecord validation."""
        # Validate first 10 rows
        for _, row in sample_pitches_df.head(10).iterrows():
            record = PitchRecord(
                game_pk=int(row["game_pk"]),
                game_date=str(row["game_date"]),
                pitcher_id=int(row["pitcher_id"]),
                batter_id=int(row["batter_id"]),
            )
            assert record.game_pk == int(row["game_pk"])
