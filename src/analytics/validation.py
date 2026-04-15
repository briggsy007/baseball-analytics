"""
Pydantic validation schemas and decorators for baseball analytics data.

Provides strict validation for Statcast pitch records, game states, model
predictions, and bulk DataFrames.  All range constants are derived from
historical MLB data and include generous safety margins.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# ── Physical range constants ──────────────────────────────────────────────
# Generous bounds that cover all historical Statcast data plus margin.

VELOCITY_MIN: float = 50.0
VELOCITY_MAX: float = 110.0

SPIN_RATE_MIN: float = 0.0
SPIN_RATE_MAX: float = 4000.0

SPIN_AXIS_MIN: float = 0.0
SPIN_AXIS_MAX: float = 360.0

PLATE_X_MIN: float = -3.0
PLATE_X_MAX: float = 3.0

PLATE_Z_MIN: float = 0.0
PLATE_Z_MAX: float = 6.0

PFX_MIN: float = -25.0
PFX_MAX: float = 25.0

EXTENSION_MIN: float = 3.0
EXTENSION_MAX: float = 9.0

LAUNCH_SPEED_MIN: float = 0.0
LAUNCH_SPEED_MAX: float = 125.0

LAUNCH_ANGLE_MIN: float = -90.0
LAUNCH_ANGLE_MAX: float = 90.0

RELEASE_POS_X_MIN: float = -4.0
RELEASE_POS_X_MAX: float = 4.0

RELEASE_POS_Z_MIN: float = 0.0
RELEASE_POS_Z_MAX: float = 8.0

# Valid Statcast pitch type codes
VALID_PITCH_TYPES: set[str] = {
    "FF",  # 4-seam fastball
    "SI",  # sinker
    "FC",  # cutter
    "SL",  # slider
    "CU",  # curveball
    "CH",  # changeup
    "FS",  # splitter
    "KC",  # knuckle curve
    "KN",  # knuckleball
    "EP",  # eephus
    "CS",  # slow curve
    "SV",  # sweeper
    "ST",  # sweeping curve / slurve
    "SC",  # screwball
    "FO",  # forkball
    "FA",  # fastball (generic)
    "PO",  # pitchout
    "IN",  # intentional ball
    "AB",  # automatic ball
    "UN",  # unknown
}


# ── Pydantic models ──────────────────────────────────────────────────────


class PitchRecord(BaseModel):
    """Validates a single Statcast pitch row.

    All physical measurements use generous ranges that cover the full
    historical distribution (plus outlier margin).  Optional fields
    default to None for pitches where the measurement is absent.
    """

    game_pk: int
    game_date: Any  # date or string
    pitcher_id: int
    batter_id: int

    pitch_type: Optional[str] = None
    release_speed: Optional[float] = Field(default=None, ge=VELOCITY_MIN, le=VELOCITY_MAX)
    release_spin_rate: Optional[float] = Field(default=None, ge=SPIN_RATE_MIN, le=SPIN_RATE_MAX)
    spin_axis: Optional[float] = Field(default=None, ge=SPIN_AXIS_MIN, le=SPIN_AXIS_MAX)

    pfx_x: Optional[float] = Field(default=None, ge=PFX_MIN, le=PFX_MAX)
    pfx_z: Optional[float] = Field(default=None, ge=PFX_MIN, le=PFX_MAX)

    plate_x: Optional[float] = Field(default=None, ge=PLATE_X_MIN, le=PLATE_X_MAX)
    plate_z: Optional[float] = Field(default=None, ge=PLATE_Z_MIN, le=PLATE_Z_MAX)

    release_extension: Optional[float] = Field(default=None, ge=EXTENSION_MIN, le=EXTENSION_MAX)
    release_pos_x: Optional[float] = Field(default=None, ge=RELEASE_POS_X_MIN, le=RELEASE_POS_X_MAX)
    release_pos_z: Optional[float] = Field(default=None, ge=RELEASE_POS_Z_MIN, le=RELEASE_POS_Z_MAX)

    launch_speed: Optional[float] = Field(default=None, ge=LAUNCH_SPEED_MIN, le=LAUNCH_SPEED_MAX)
    launch_angle: Optional[float] = Field(default=None, ge=LAUNCH_ANGLE_MIN, le=LAUNCH_ANGLE_MAX)

    inning: Optional[int] = Field(default=None, ge=1, le=30)
    outs_when_up: Optional[int] = Field(default=None, ge=0, le=2)
    balls: Optional[int] = Field(default=None, ge=0, le=3)
    strikes: Optional[int] = Field(default=None, ge=0, le=2)

    stand: Optional[str] = None
    p_throws: Optional[str] = None
    description: Optional[str] = None
    events: Optional[str] = None

    woba_value: Optional[float] = None
    woba_denom: Optional[float] = None

    model_config = {"extra": "allow"}

    @field_validator("pitch_type")
    @classmethod
    def validate_pitch_type(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_PITCH_TYPES:
            logger.warning("Unknown pitch type: %s", v)
        return v

    @field_validator("stand", "p_throws")
    @classmethod
    def validate_handedness(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("L", "R"):
            raise ValueError(f"Handedness must be 'L' or 'R', got {v!r}")
        return v


class GameState(BaseModel):
    """Validates the game situation at the time of a pitch."""

    inning: int = Field(ge=1, le=18)
    inning_topbot: str
    outs: int = Field(ge=0, le=2)
    balls: int = Field(ge=0, le=3)
    strikes: int = Field(ge=0, le=2)
    on_1b: bool = False
    on_2b: bool = False
    on_3b: bool = False
    home_score: int = Field(default=0, ge=0)
    away_score: int = Field(default=0, ge=0)

    @field_validator("inning_topbot")
    @classmethod
    def validate_inning_topbot(cls, v: str) -> str:
        if v not in ("Top", "Bot"):
            raise ValueError(f"inning_topbot must be 'Top' or 'Bot', got {v!r}")
        return v


class ModelPrediction(BaseModel):
    """Base schema for model output validation.

    All model predictions must include at minimum a model name,
    version, and a timestamp.  Subclass for model-specific fields.
    """

    model_name: str
    model_version: str
    generated_at: str
    prediction: Any
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    model_config = {"extra": "allow"}


class PitchDataFrame(BaseModel):
    """Validates a DataFrame of pitches for bulk operations.

    Checks column presence, null rates, and value ranges for the entire
    DataFrame rather than individual rows.
    """

    n_rows: int = Field(ge=1)
    n_columns: int = Field(ge=1)
    null_rates: dict[str, float] = Field(default_factory=dict)
    column_names: list[str] = Field(default_factory=list)

    # Maximum allowable null rate for critical columns
    max_null_rate: float = Field(default=0.5, ge=0.0, le=1.0)

    # Required columns that must be present
    required_columns: list[str] = Field(
        default_factory=lambda: [
            "pitcher_id",
            "batter_id",
            "game_pk",
        ]
    )

    @model_validator(mode="after")
    def check_required_columns(self) -> "PitchDataFrame":
        missing = [c for c in self.required_columns if c not in self.column_names]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return self

    @model_validator(mode="after")
    def check_null_rates(self) -> "PitchDataFrame":
        high_null = {
            col: rate
            for col, rate in self.null_rates.items()
            if rate > self.max_null_rate and col in self.required_columns
        }
        if high_null:
            raise ValueError(
                f"Required columns exceed max null rate ({self.max_null_rate}): {high_null}"
            )
        return self

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        required_columns: list[str] | None = None,
        max_null_rate: float = 0.5,
    ) -> "PitchDataFrame":
        """Create a validator from a pandas DataFrame.

        Args:
            df: The DataFrame to validate.
            required_columns: Override for required column list.
            max_null_rate: Maximum allowable null fraction for required columns.

        Returns:
            A validated ``PitchDataFrame`` instance.

        Raises:
            ValidationError: On validation failure.
        """
        null_rates = (df.isnull().sum() / len(df)).to_dict() if len(df) > 0 else {}
        kwargs: dict[str, Any] = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "null_rates": null_rates,
            "column_names": list(df.columns),
            "max_null_rate": max_null_rate,
        }
        if required_columns is not None:
            kwargs["required_columns"] = required_columns
        return cls(**kwargs)


# ── Validation decorator ─────────────────────────────────────────────────


def validate_pitch_data(
    required_columns: list[str] | None = None,
    max_null_rate: float = 0.5,
    min_rows: int = 1,
):
    """Decorator that validates the first DataFrame argument of a function.

    Scans positional and keyword arguments for the first ``pd.DataFrame``
    and runs ``PitchDataFrame`` validation on it before the function
    executes.

    Args:
        required_columns: Columns that must be present.
        max_null_rate: Maximum null rate for required columns.
        min_rows: Minimum number of rows required.

    Returns:
        The decorator.

    Raises:
        ValueError: If no DataFrame argument is found or validation fails.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Find the first DataFrame in positional or keyword args
            df = None
            for arg in args:
                if isinstance(arg, pd.DataFrame):
                    df = arg
                    break
            if df is None:
                for v in kwargs.values():
                    if isinstance(v, pd.DataFrame):
                        df = v
                        break

            if df is None:
                raise ValueError(
                    f"No DataFrame argument found for @validate_pitch_data "
                    f"on {func.__name__}."
                )

            if len(df) < min_rows:
                raise ValueError(
                    f"DataFrame has {len(df)} rows, need at least {min_rows}."
                )

            # Validate column presence and null rates
            req_cols = required_columns or ["pitcher_id", "batter_id", "game_pk"]
            PitchDataFrame.from_dataframe(
                df,
                required_columns=req_cols,
                max_null_rate=max_null_rate,
            )

            # Range checks on known physical columns
            _check_range(df, "release_speed", VELOCITY_MIN, VELOCITY_MAX)
            _check_range(df, "release_spin_rate", SPIN_RATE_MIN, SPIN_RATE_MAX)
            _check_range(df, "plate_x", PLATE_X_MIN, PLATE_X_MAX)
            _check_range(df, "plate_z", PLATE_Z_MIN, PLATE_Z_MAX)
            _check_range(df, "pfx_x", PFX_MIN, PFX_MAX)
            _check_range(df, "pfx_z", PFX_MIN, PFX_MAX)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _check_range(
    df: pd.DataFrame, col: str, lo: float, hi: float
) -> None:
    """Warn if any non-null values in *col* are outside [lo, hi]."""
    if col not in df.columns:
        return
    series = df[col].dropna()
    if series.empty:
        return
    n_below = (series < lo).sum()
    n_above = (series > hi).sum()
    if n_below > 0 or n_above > 0:
        logger.warning(
            "Column %s has %d values below %.1f and %d above %.1f",
            col,
            n_below,
            lo,
            n_above,
            hi,
        )
