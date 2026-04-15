"""
Shared feature engineering functions for all baseball analytics models.

Every function in this module is stateless and operates on pandas Series,
DataFrames, or scalar values.  They are the common building blocks that
the 16 proprietary models compose into their feature pipelines.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Statistical transforms ───────────────────────────────────────────────


def compute_z_scores(
    df: pd.DataFrame,
    columns: list[str],
    group_by: Optional[str | list[str]] = None,
) -> pd.DataFrame:
    """Compute z-score normalisation for the given columns.

    When *group_by* is provided the mean and standard deviation are
    computed within each group (e.g. per pitch type), otherwise they
    are computed across the entire DataFrame.

    Args:
        df: Input DataFrame.
        columns: Columns to normalise.
        group_by: Optional column(s) to group by before computing stats.

    Returns:
        A copy of *df* with new columns ``{col}_z`` appended.
    """
    result = df.copy()

    for col in columns:
        if col not in df.columns:
            logger.warning("Column %s not in DataFrame, skipping z-score.", col)
            continue

        if group_by is not None:
            grouped = df.groupby(group_by)[col]
            means = grouped.transform("mean")
            stds = grouped.transform("std")
            # Avoid division by zero: groups with zero variance get z=0
            stds = stds.replace(0, np.nan)
            result[f"{col}_z"] = (df[col] - means) / stds
            result[f"{col}_z"] = result[f"{col}_z"].fillna(0.0)
        else:
            mean = df[col].mean()
            std = df[col].std()
            if std == 0 or pd.isna(std):
                result[f"{col}_z"] = 0.0
            else:
                result[f"{col}_z"] = (df[col] - mean) / std

    return result


def rolling_mean(
    series: pd.Series,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    """Compute a rolling average with sensible min_periods default.

    Args:
        series: Input values.
        window: Rolling window size.
        min_periods: Minimum observations required.  Defaults to
                     ``max(1, window // 3)``.

    Returns:
        Rolling mean Series.
    """
    if min_periods is None:
        min_periods = max(1, window // 3)
    return series.rolling(window=window, min_periods=min_periods).mean()


def exponential_decay(
    values: pd.Series | np.ndarray,
    half_life: float,
) -> pd.Series:
    """Exponentially-weighted moving average with a given half-life.

    Args:
        values: Time-ordered values (most recent last).
        half_life: Number of periods for the signal to decay by 50 %.

    Returns:
        EWMA Series.
    """
    series = pd.Series(values) if not isinstance(values, pd.Series) else values
    return series.ewm(halflife=half_life, min_periods=1).mean()


# ── Information theory ───────────────────────────────────────────────────


def shannon_entropy(probs: np.ndarray | list[float]) -> float:
    """Compute Shannon entropy in nats for a discrete probability distribution.

    Zero-probability events are safely ignored.  The distribution must
    sum to approximately 1.0.

    Args:
        probs: Array of probabilities.

    Returns:
        Entropy value (non-negative float).

    Raises:
        ValueError: If any probability is negative or sum is not ~1.
    """
    p = np.asarray(probs, dtype=np.float64)

    if np.any(p < 0):
        raise ValueError("Probabilities must be non-negative.")

    total = p.sum()
    if not np.isclose(total, 1.0, atol=0.01):
        raise ValueError(
            f"Probabilities must sum to ~1.0, got {total:.4f}."
        )

    # Normalise to exactly 1.0 and filter zeros
    p = p / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


# ── Pitch quality ────────────────────────────────────────────────────────


def pitch_quality_vector(row: pd.Series) -> dict[str, float]:
    """Compute a composite quality signal from a single pitch's physical attributes.

    Combines velocity, spin rate, and induced movement into a single
    quality dictionary.  All values are normalised to a 0-100 scale
    using league-average reference points.

    Args:
        row: A Series containing at least ``release_speed``,
             ``release_spin_rate``, ``pfx_x``, ``pfx_z``.

    Returns:
        Dictionary with keys ``velocity_score``, ``spin_score``,
        ``movement_score``, and ``composite``.
    """
    # League-average reference points (approx MLB 2020-2025)
    VELO_REF = 93.0
    VELO_STD = 3.0
    SPIN_REF = 2300.0
    SPIN_STD = 300.0
    MOVE_REF = 8.0  # inches of total movement
    MOVE_STD = 3.0

    velo = float(row.get("release_speed", np.nan))
    spin = float(row.get("release_spin_rate", np.nan))
    pfx_x = float(row.get("pfx_x", np.nan))
    pfx_z = float(row.get("pfx_z", np.nan))

    # Velocity score: higher is better
    if np.isnan(velo):
        velo_score = 50.0
    else:
        velo_score = 50.0 + (velo - VELO_REF) / VELO_STD * 10.0
        velo_score = float(np.clip(velo_score, 0.0, 100.0))

    # Spin score
    if np.isnan(spin):
        spin_score = 50.0
    else:
        spin_score = 50.0 + (spin - SPIN_REF) / SPIN_STD * 10.0
        spin_score = float(np.clip(spin_score, 0.0, 100.0))

    # Movement score: total movement magnitude
    if np.isnan(pfx_x) or np.isnan(pfx_z):
        move_score = 50.0
    else:
        total_movement = np.sqrt(pfx_x**2 + pfx_z**2)
        move_score = 50.0 + (total_movement - MOVE_REF) / MOVE_STD * 10.0
        move_score = float(np.clip(move_score, 0.0, 100.0))

    composite = (velo_score * 0.35 + spin_score * 0.25 + move_score * 0.40)

    return {
        "velocity_score": round(velo_score, 1),
        "spin_score": round(spin_score, 1),
        "movement_score": round(move_score, 1),
        "composite": round(composite, 1),
    }


# ── Count / state encoding ──────────────────────────────────────────────


def encode_count_state(balls: int, strikes: int) -> int:
    """Encode a ball-strike count as a single integer index.

    The encoding is ``balls * 3 + strikes``, producing indices 0-11
    for all valid counts (0-0 through 3-2).

    Args:
        balls: Number of balls (0-3).
        strikes: Number of strikes (0-2).

    Returns:
        Integer index in [0, 11].

    Raises:
        ValueError: If balls or strikes are out of range.
    """
    if not (0 <= balls <= 3):
        raise ValueError(f"balls must be 0-3, got {balls}")
    if not (0 <= strikes <= 2):
        raise ValueError(f"strikes must be 0-2, got {strikes}")
    return balls * 3 + strikes


# ── Pitch pair / tunnel metrics ──────────────────────────────────────────


def tunnel_distance(pitch1: pd.Series, pitch2: pd.Series) -> float:
    """Compute the tunnel distance between two consecutive pitches.

    Tunnel distance is the Euclidean distance between the pitches at
    the "tunnel point" (approximately 23.9 feet from home plate, where
    the batter must commit to swing).  We approximate this using the
    plate location and release-point difference.

    Args:
        pitch1: First pitch (must have plate_x, plate_z).
        pitch2: Second pitch.

    Returns:
        Distance in feet (lower = better tunnel).
    """
    x1 = float(pitch1.get("plate_x", np.nan))
    z1 = float(pitch1.get("plate_z", np.nan))
    x2 = float(pitch2.get("plate_x", np.nan))
    z2 = float(pitch2.get("plate_z", np.nan))

    if any(np.isnan(v) for v in [x1, z1, x2, z2]):
        return np.nan

    return float(np.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2))


def release_point_distance(row: pd.Series, baseline: pd.Series) -> float:
    """Compute release point drift from a baseline.

    Uses 3D Euclidean distance in (release_pos_x, release_pos_y,
    release_pos_z) space.

    Args:
        row: Pitch record with release position columns.
        baseline: Reference release point (e.g. season average).

    Returns:
        Distance in feet.
    """
    dims = ["release_pos_x", "release_pos_y", "release_pos_z"]
    vals_row = []
    vals_base = []

    for d in dims:
        r = float(row.get(d, np.nan))
        b = float(baseline.get(d, np.nan))
        if np.isnan(r) or np.isnan(b):
            return np.nan
        vals_row.append(r)
        vals_base.append(b)

    return float(np.sqrt(sum((a - b) ** 2 for a, b in zip(vals_row, vals_base))))


# ── Outcome classification ───────────────────────────────────────────────

_OUTCOME_MAP: dict[str, str] = {
    # Whiffs
    "swinging_strike": "whiff",
    "swinging_strike_blocked": "whiff",
    "foul_tip": "whiff",
    "missed_bunt": "whiff",
    # Called strikes
    "called_strike": "called_strike",
    # Balls
    "ball": "ball",
    "blocked_ball": "ball",
    "intent_ball": "ball",
    "pitchout": "ball",
    "hit_by_pitch": "hbp",
    # Fouls
    "foul": "foul",
    "foul_bunt": "foul",
    # In play
    "hit_into_play": "in_play",
    "hit_into_play_no_out": "in_play",
    "hit_into_play_score": "in_play",
}


def classify_pitch_outcome(description: str | None) -> str:
    """Map a Statcast description string to a canonical outcome category.

    Args:
        description: The ``description`` column from the pitches table.

    Returns:
        One of ``"whiff"``, ``"called_strike"``, ``"ball"``, ``"foul"``,
        ``"in_play"``, ``"hbp"``, or ``"unknown"``.
    """
    if description is None:
        return "unknown"
    return _OUTCOME_MAP.get(description.strip().lower(), "unknown")


# ── Run value ────────────────────────────────────────────────────────────


def compute_run_value(
    woba_value: float | None,
    woba_denom: float | None,
) -> float | None:
    """Compute run value per plate appearance from wOBA components.

    Args:
        woba_value: The wOBA numerator for this event.
        woba_denom: The wOBA denominator (1.0 for PA-ending events, else 0).

    Returns:
        Run value, or None if denominator is zero/missing.
    """
    if woba_denom is None or woba_denom == 0:
        return None
    if woba_value is None:
        return None
    # League average wOBA is ~0.320; run value = (woba_value - 0.320) / 1.25
    LEAGUE_WOBA = 0.320
    WOBA_SCALE = 1.25
    return (woba_value - LEAGUE_WOBA) / WOBA_SCALE


# ── Database helpers (shared across models) ──────────────────────────────


def get_player_name(
    conn,
    player_id: int,
) -> str | None:
    """Look up a player's full name from the players table.

    Args:
        conn: Open DuckDB connection.
        player_id: MLB player ID.

    Returns:
        Full name string, or None if not found.
    """
    try:
        result = conn.execute(
            "SELECT full_name FROM players WHERE player_id = $1",
            [player_id],
        ).fetchone()
        return result[0] if result else None
    except Exception:
        return None


def get_latest_season(conn) -> int:
    """Return the most recent season year in the pitches table.

    Args:
        conn: Open DuckDB connection.

    Returns:
        Most recent season year, or 2025 as fallback.
    """
    result = conn.execute(
        "SELECT MAX(EXTRACT(YEAR FROM game_date)) FROM pitches"
    ).fetchone()
    if result and result[0] is not None:
        return int(result[0])
    return 2025


# ── Pitch quality composite (weighted z-score) ──────────────────────────

# Columns and weights used by the Stuff Concentration / pitch-type quality
# composite.  Shared by kinetic_half_life and pitch_decay models.
STUFF_QUALITY_COLUMNS: list[str] = [
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
]

STUFF_QUALITY_WEIGHTS: dict[str, float] = {
    "release_speed_z": 0.30,
    "release_spin_rate_z": 0.20,
    "pfx_x_z": 0.15,
    "pfx_z_z": 0.15,
    "release_pos_x_z": 0.10,
    "release_pos_z_z": 0.10,
}


def compute_stuff_quality(
    pitches_df: pd.DataFrame,
    columns: list[str] | None = None,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Compute a per-pitch composite quality score from z-scored physical attributes.

    The score is a weighted combination of z-scored physical attributes:
    velocity, spin rate, horizontal/vertical movement, and release-point
    consistency.  Higher values indicate better raw stuff on that pitch.

    For movement columns (``pfx_x_z``, ``pfx_z_z``) the absolute z-score
    is used (more movement = better).  For release-point columns
    (``release_pos_x_z``, ``release_pos_z_z``) the negative absolute
    z-score is used (less deviation = better).

    Args:
        pitches_df: DataFrame with at least the columns listed in *columns*.
        columns: Override for the quality columns.  Defaults to
                 ``STUFF_QUALITY_COLUMNS``.
        weights: Override for the z-score weights.  Defaults to
                 ``STUFF_QUALITY_WEIGHTS``.

    Returns:
        A Series of per-pitch quality scores aligned to the input index.
    """
    if columns is None:
        columns = STUFF_QUALITY_COLUMNS
    if weights is None:
        weights = STUFF_QUALITY_WEIGHTS

    if pitches_df.empty:
        return pd.Series(dtype=float)

    available_cols = [c for c in columns if c in pitches_df.columns]
    if not available_cols:
        logger.warning("No quality columns found in pitches_df; returning zeros.")
        return pd.Series(0.0, index=pitches_df.index)

    z_df = compute_z_scores(pitches_df, available_cols)

    score = pd.Series(0.0, index=pitches_df.index)
    total_weight = 0.0
    for z_col, weight in weights.items():
        if z_col in z_df.columns:
            # For movement columns use absolute z-score (more movement = better)
            if z_col in ("pfx_x_z", "pfx_z_z"):
                score += z_df[z_col].abs() * weight
            # For release point columns, *lower* deviation is better, so invert
            elif z_col in ("release_pos_x_z", "release_pos_z_z"):
                score += -z_df[z_col].abs() * weight
            else:
                score += z_df[z_col] * weight
            total_weight += weight

    if total_weight > 0:
        score = score / total_weight

    return score
