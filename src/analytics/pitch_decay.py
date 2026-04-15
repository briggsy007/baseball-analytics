"""
Pitch Decay Rate (PDR) model -- F1 tire-degradation-inspired per-pitch-type fatigue cliff detection.

Unlike the Kinetic Half-Life model (which fits a smooth exponential decay
across ALL pitch types), PDR focuses on **changepoint detection** within
each individual pitch type.  For every game appearance, it finds the pitch
count at which a specific pitch type's quality *drops off a cliff* using
piecewise linear regression.

Key outputs:
- Per-pitch-type cliff point (tau): the pitch number where quality breaks
- Pre-cliff and post-cliff slope
- Survival curve: P(pitch type still effective) vs pitch count
- "First to die" identification: which pitch type degrades earliest
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.analytics.base import BaseAnalyticsModel
from src.analytics.features import (
    compute_z_scores,
    compute_stuff_quality,
    STUFF_QUALITY_COLUMNS,
    STUFF_QUALITY_WEIGHTS,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_MIN_PITCHES_PER_TYPE_GAME: int = 30
_MIN_GAME_APPEARANCES: int = 5
_DEFAULT_MIN_PITCHES: int = 500
_ROLLING_WINDOW: int = 5
_MIN_ROLLING_PERIODS: int = 3

# Re-export shared quality columns/weights for backward compatibility.
# Z-scores are computed WITHIN each pitch type (not across all types).
_QUALITY_COLUMNS = STUFF_QUALITY_COLUMNS
_QUALITY_WEIGHTS = STUFF_QUALITY_WEIGHTS


# ── Pure functions ─────────────────────────────────────────────────────────────


def compute_pitch_type_quality(
    pitches_df: pd.DataFrame,
    pitch_type: str,
) -> pd.Series:
    """Compute per-pitch quality signal for a SINGLE pitch type.

    Z-scores are computed within the subset of pitches of that type,
    so the baseline is the pitcher's own average for that specific pitch.

    Delegates to :func:`~src.analytics.features.compute_stuff_quality`
    after filtering to the requested pitch type.

    Args:
        pitches_df: DataFrame containing at least the quality columns,
                    already filtered to a single game appearance.
        pitch_type: The pitch type code to isolate (e.g. ``"FF"``).

    Returns:
        A Series of quality scores aligned to the input index.
        Only rows matching *pitch_type* will have non-NaN values.
    """
    mask = pitches_df["pitch_type"] == pitch_type
    subset = pitches_df.loc[mask].copy()

    if subset.empty:
        return pd.Series(dtype=float)

    return compute_stuff_quality(subset, _QUALITY_COLUMNS, _QUALITY_WEIGHTS)


def detect_cliff_point(
    quality_series: pd.Series | np.ndarray,
    pitch_numbers: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Detect the changepoint (cliff) in a quality time-series.

    Fits a piecewise linear model with two segments:
        y = a1*x + b1   for x <= tau
        y = a2*x + b2   for x >  tau
    and finds tau via grid search that minimises total MSE.

    Args:
        quality_series: Quality signal values (possibly smoothed).
        pitch_numbers:  Corresponding ordinal pitch numbers within the
                        subset (0-indexed).

    Returns:
        Dictionary with keys ``tau``, ``pre_slope``, ``post_slope``,
        ``mse_reduction``, ``n_points``.  Returns ``None`` values when
        the series is too short or no meaningful cliff is found.
    """
    y = np.asarray(quality_series, dtype=float)
    x = np.asarray(pitch_numbers, dtype=float)

    # Drop NaN
    valid = ~np.isnan(y) & ~np.isnan(x)
    y = y[valid]
    x = x[valid]

    result: dict[str, Any] = {
        "tau": None,
        "pre_slope": None,
        "post_slope": None,
        "mse_reduction": None,
        "n_points": len(y),
    }

    if len(y) < 10:
        logger.debug("Too few valid points (%d) for cliff detection.", len(y))
        return result

    # ── Single-segment baseline MSE ──────────────────────────────────────
    if len(np.unique(x)) < 2:
        return result

    # Fit a single linear regression as baseline
    coeffs_full = np.polyfit(x, y, 1)
    y_pred_full = np.polyval(coeffs_full, x)
    mse_full = float(np.mean((y - y_pred_full) ** 2))

    # If the single-line fit is already near-perfect, there is no cliff
    # to detect (e.g. constant or perfectly linear data).
    if mse_full < 1e-10:
        return result

    # ── Grid search over candidate breakpoints ───────────────────────────
    # Require at least 5 points on each side
    min_side = 5
    if len(x) < 2 * min_side:
        return result

    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]

    best_mse = mse_full
    best_tau: float | None = None
    best_pre_slope: float | None = None
    best_post_slope: float | None = None

    for split in range(min_side, len(x_sorted) - min_side + 1):
        x_left, y_left = x_sorted[:split], y_sorted[:split]
        x_right, y_right = x_sorted[split:], y_sorted[split:]

        if len(np.unique(x_left)) < 2 or len(np.unique(x_right)) < 2:
            continue

        c_left = np.polyfit(x_left, y_left, 1)
        c_right = np.polyfit(x_right, y_right, 1)

        pred_left = np.polyval(c_left, x_left)
        pred_right = np.polyval(c_right, x_right)
        combined_mse = float(
            (np.sum((y_left - pred_left) ** 2) + np.sum((y_right - pred_right) ** 2))
            / len(x_sorted)
        )

        if combined_mse < best_mse:
            best_mse = combined_mse
            best_tau = float(x_sorted[split])
            best_pre_slope = float(c_left[0])
            best_post_slope = float(c_right[0])

    if best_tau is None:
        return result

    mse_reduction = (mse_full - best_mse) / mse_full if mse_full > 0 else 0.0

    # Require a meaningful MSE reduction (at least 5%) to avoid
    # spurious cliffs on flat or monotonic data.
    if mse_reduction < 0.05:
        return result

    # Only accept cliff if the post-slope is meaningfully more negative
    # than the pre-slope (i.e. there IS a drop-off).  The post-slope
    # must also be negative in absolute terms.
    if best_post_slope is not None and best_pre_slope is not None:
        if best_post_slope >= best_pre_slope:
            # No cliff: the quality didn't degrade more after the breakpoint
            return result
        if best_post_slope >= 0:
            # Post-cliff quality is not declining
            return result

    result["tau"] = round(best_tau, 1)
    result["pre_slope"] = round(best_pre_slope, 6) if best_pre_slope is not None else None
    result["post_slope"] = round(best_post_slope, 6) if best_post_slope is not None else None
    result["mse_reduction"] = round(mse_reduction, 4)

    return result


def _build_survival_curve(
    cliff_points: list[float],
    max_pitch_count: int = 120,
) -> dict[str, list[float]]:
    """Build a Kaplan-Meier-style survival curve from cliff points.

    P(still effective at pitch N) = fraction of games where the cliff
    has NOT yet been reached at pitch count N.

    Args:
        cliff_points: List of tau values from individual games.
        max_pitch_count: Maximum pitch count for the curve.

    Returns:
        Dictionary with ``pitch_numbers`` and ``survival`` lists.
    """
    if not cliff_points:
        return {"pitch_numbers": [], "survival": []}

    n_games = len(cliff_points)
    pitch_numbers = list(range(0, max_pitch_count + 1))
    survival = []
    for n in pitch_numbers:
        frac_surviving = sum(1 for tau in cliff_points if tau > n) / n_games
        survival.append(round(frac_surviving, 4))

    return {"pitch_numbers": pitch_numbers, "survival": survival}


def _query_pitcher_game_pitches(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """Retrieve pitch-level data for a pitcher, ordered within each game."""
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    params: list = [pitcher_id]
    if season:
        params.append(season)

    query = f"""
        SELECT
            game_pk,
            game_date,
            pitch_type,
            release_speed,
            release_spin_rate,
            pfx_x,
            pfx_z,
            release_pos_x,
            release_pos_z,
            inning,
            at_bat_number,
            pitch_number
        FROM pitches
        WHERE pitcher_id = $1
          AND release_speed IS NOT NULL
          AND release_spin_rate IS NOT NULL
          AND pfx_x IS NOT NULL
          AND pfx_z IS NOT NULL
          AND pitch_type IS NOT NULL
          {season_filter}
        ORDER BY game_date, at_bat_number, pitch_number
    """
    return conn.execute(query, params).fetchdf()


def _detect_game_cliff(
    game_df: pd.DataFrame,
    pitch_type: str,
) -> dict[str, Any] | None:
    """Detect the cliff point for one pitch type in one game.

    Returns None when there are fewer than ``_MIN_PITCHES_PER_TYPE_GAME``
    pitches of the requested type.
    """
    mask = game_df["pitch_type"] == pitch_type
    pt_df = game_df.loc[mask].copy()

    if len(pt_df) < _MIN_PITCHES_PER_TYPE_GAME:
        return None

    quality = compute_pitch_type_quality(pt_df, pitch_type)
    if quality.empty:
        return None

    # Smooth with a rolling window
    quality_smooth = quality.rolling(
        window=_ROLLING_WINDOW, min_periods=_MIN_ROLLING_PERIODS,
    ).mean()

    # Ordinal pitch numbers within this pitch type for this game
    ordinals = np.arange(len(quality))

    cliff = detect_cliff_point(quality_smooth, ordinals)

    if cliff["tau"] is None:
        return None

    cliff["game_pk"] = int(game_df["game_pk"].iloc[0])
    cliff["game_date"] = game_df["game_date"].iloc[0]
    cliff["pitch_type"] = pitch_type
    cliff["n_pitches_of_type"] = len(pt_df)
    return cliff


def calculate_pdr(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> dict[str, Any]:
    """Calculate the full Pitch Decay Rate profile for a pitcher.

    For each pitch type with sufficient data across multiple games,
    detects the cliff point per game and aggregates into median cliff,
    survival curves, and a "first to die" indicator.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        season: Restrict to this year (optional).

    Returns:
        Dictionary with keys:
        - ``pitcher_id``
        - ``season``
        - ``pitch_types``: dict mapping pitch type -> {
              ``median_cliff``, ``mean_cliff``, ``std_cliff``,
              ``games_analysed``, ``survival_curve``, ``game_cliffs``
          }
        - ``first_to_die``: pitch type code with earliest median cliff
        - ``first_to_die_cliff``: that cliff's median pitch number
    """
    logger.info("Calculating PDR for pitcher %d (season=%s).", pitcher_id, season)

    df = _query_pitcher_game_pitches(conn, pitcher_id, season)

    empty_result: dict[str, Any] = {
        "pitcher_id": pitcher_id,
        "season": season,
        "pitch_types": {},
        "first_to_die": None,
        "first_to_die_cliff": None,
    }

    if df.empty:
        logger.warning("No pitches found for pitcher %d.", pitcher_id)
        return empty_result

    pitch_types = df["pitch_type"].dropna().unique()

    per_type: dict[str, dict[str, Any]] = {}

    for pt in pitch_types:
        game_cliffs: list[dict] = []
        cliff_taus: list[float] = []

        for game_pk, game_df in df.groupby("game_pk"):
            cliff = _detect_game_cliff(game_df, pt)
            if cliff is not None:
                game_cliffs.append(cliff)
                cliff_taus.append(cliff["tau"])

        if len(game_cliffs) < _MIN_GAME_APPEARANCES:
            continue

        median_cliff = float(np.median(cliff_taus))
        mean_cliff = float(np.mean(cliff_taus))
        std_cliff = float(np.std(cliff_taus)) if len(cliff_taus) > 1 else 0.0
        survival = _build_survival_curve(cliff_taus)

        per_type[pt] = {
            "median_cliff": round(median_cliff, 1),
            "mean_cliff": round(mean_cliff, 1),
            "std_cliff": round(std_cliff, 1),
            "games_analysed": len(game_cliffs),
            "survival_curve": survival,
            "game_cliffs": game_cliffs,
        }

    if not per_type:
        return empty_result

    # Determine which pitch type's cliff comes earliest
    first_pt = min(per_type.items(), key=lambda kv: kv[1]["median_cliff"])

    return {
        "pitcher_id": pitcher_id,
        "season": season,
        "pitch_types": per_type,
        "first_to_die": first_pt[0],
        "first_to_die_cliff": first_pt[1]["median_cliff"],
    }


def get_first_to_die(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> dict[str, Any]:
    """Return which pitch type cliff comes earliest for a pitcher.

    Convenience wrapper around :func:`calculate_pdr` that returns only
    the "first to die" information.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        season: Optional season filter.

    Returns:
        Dictionary with ``pitcher_id``, ``first_to_die`` (pitch type code),
        ``cliff_pitch_number`` (median cliff), and ``all_cliffs`` (per-type
        median cliff dict).
    """
    pdr = calculate_pdr(conn, pitcher_id, season)

    all_cliffs = {
        pt: data["median_cliff"]
        for pt, data in pdr["pitch_types"].items()
    }

    return {
        "pitcher_id": pitcher_id,
        "first_to_die": pdr["first_to_die"],
        "cliff_pitch_number": pdr["first_to_die_cliff"],
        "all_cliffs": all_cliffs,
    }


def batch_calculate(
    conn: duckdb.DuckDBPyConnection,
    season: Optional[int] = None,
    min_pitches: int = _DEFAULT_MIN_PITCHES,
) -> pd.DataFrame:
    """Compute a PDR leaderboard for all qualifying pitchers.

    Args:
        conn: Open DuckDB connection.
        season: Restrict to this year (optional).
        min_pitches: Minimum total pitches to qualify.

    Returns:
        DataFrame with columns: pitcher_id, name, first_to_die,
        first_to_die_cliff, plus per-pitch-type cliff columns.
    """
    logger.info("Batch calculating PDR (season=%s, min_pitches=%d).", season, min_pitches)

    season_filter = "AND EXTRACT(YEAR FROM game_date) = $1" if season else ""
    params: list = []
    if season:
        params.append(season)

    query = f"""
        SELECT pitcher_id, COUNT(*) AS pitch_count
        FROM pitches
        WHERE release_speed IS NOT NULL
          AND release_spin_rate IS NOT NULL
          AND pfx_x IS NOT NULL
          AND pfx_z IS NOT NULL
          AND pitch_type IS NOT NULL
          {season_filter}
        GROUP BY pitcher_id
        HAVING COUNT(*) >= {int(min_pitches)}
    """
    qualifying = conn.execute(query, params).fetchdf()

    empty_cols = ["pitcher_id", "name", "first_to_die", "first_to_die_cliff"]
    if qualifying.empty:
        logger.warning("No pitchers qualify with min_pitches=%d.", min_pitches)
        return pd.DataFrame(columns=empty_cols)

    rows: list[dict] = []
    for pitcher_id in qualifying["pitcher_id"].tolist():
        pdr = calculate_pdr(conn, int(pitcher_id), season)
        if not pdr["pitch_types"]:
            continue

        row: dict[str, Any] = {
            "pitcher_id": int(pitcher_id),
            "first_to_die": pdr["first_to_die"],
            "first_to_die_cliff": pdr["first_to_die_cliff"],
        }
        for pt, pt_data in pdr["pitch_types"].items():
            row[f"{pt}_cliff"] = pt_data["median_cliff"]
            row[f"{pt}_games"] = pt_data["games_analysed"]
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=empty_cols)

    result_df = pd.DataFrame(rows)

    # Join player names
    try:
        names_df = conn.execute(
            "SELECT player_id AS pitcher_id, full_name AS name FROM players"
        ).fetchdf()
        result_df = result_df.merge(names_df, on="pitcher_id", how="left")
    except Exception:
        result_df["name"] = None

    # Reorder columns
    front_cols = ["pitcher_id", "name", "first_to_die", "first_to_die_cliff"]
    other_cols = sorted(c for c in result_df.columns if c not in front_cols)
    result_df = result_df[
        [c for c in front_cols if c in result_df.columns] + other_cols
    ]

    result_df = result_df.sort_values(
        "first_to_die_cliff", ascending=False,
    ).reset_index(drop=True)
    logger.info("Batch PDR complete: %d pitchers.", len(result_df))
    return result_df


def get_game_cliff_data(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    game_pk: int,
) -> dict[str, Any]:
    """Retrieve cliff detection data for a specific past game.

    Returns the quality signal and detected cliff for each pitch type
    present in that game, suitable for visualization.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        game_pk: Unique game identifier.

    Returns:
        Dictionary with per-pitch-type quality signals and cliff info.
    """
    query = """
        SELECT
            game_pk,
            game_date,
            pitch_type,
            release_speed,
            release_spin_rate,
            pfx_x,
            pfx_z,
            release_pos_x,
            release_pos_z,
            inning,
            at_bat_number,
            pitch_number
        FROM pitches
        WHERE pitcher_id = $1
          AND game_pk = $2
          AND release_speed IS NOT NULL
          AND release_spin_rate IS NOT NULL
          AND pfx_x IS NOT NULL
          AND pfx_z IS NOT NULL
          AND pitch_type IS NOT NULL
        ORDER BY at_bat_number, pitch_number
    """
    df = conn.execute(query, [pitcher_id, game_pk]).fetchdf()

    if df.empty:
        return {
            "pitcher_id": pitcher_id,
            "game_pk": game_pk,
            "pitch_types": {},
        }

    result: dict[str, Any] = {
        "pitcher_id": pitcher_id,
        "game_pk": game_pk,
        "pitch_types": {},
    }

    for pt in df["pitch_type"].dropna().unique():
        mask = df["pitch_type"] == pt
        pt_df = df.loc[mask]
        if len(pt_df) < _MIN_PITCHES_PER_TYPE_GAME:
            continue

        quality = compute_pitch_type_quality(pt_df, pt)
        quality_smooth = quality.rolling(
            window=_ROLLING_WINDOW, min_periods=_MIN_ROLLING_PERIODS,
        ).mean()

        ordinals = np.arange(len(quality))
        cliff = detect_cliff_point(quality_smooth, ordinals)

        result["pitch_types"][pt] = {
            "pitch_numbers": list(range(len(quality))),
            "raw_quality": quality.tolist(),
            "smooth_quality": quality_smooth.tolist(),
            "cliff": cliff,
        }

    return result


# ── BaseAnalyticsModel subclass ────────────────────────────────────────────────


class PitchDecayRateModel(BaseAnalyticsModel):
    """F1 tire-degradation-inspired pitch-type fatigue cliff model.

    Wraps the functional API above into the project's standard
    ``BaseAnalyticsModel`` lifecycle.
    """

    @property
    def model_name(self) -> str:
        return "pitch_decay_rate"

    @property
    def version(self) -> str:
        return "1.0.0"

    def train(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Train (batch-fit) PDR for all qualifying pitchers."""
        season = kwargs.get("season")
        min_pitches = kwargs.get("min_pitches", _DEFAULT_MIN_PITCHES)

        leaderboard = batch_calculate(conn, season=season, min_pitches=min_pitches)
        self._leaderboard = leaderboard

        metrics: dict[str, Any] = {
            "n_pitchers": len(leaderboard),
            "median_first_cliff": round(
                float(leaderboard["first_to_die_cliff"].median()), 1,
            ) if not leaderboard.empty and leaderboard["first_to_die_cliff"].notna().any() else None,
        }
        self.set_training_metadata(
            metrics=metrics,
            params={"season": season, "min_pitches": min_pitches},
        )
        return metrics

    def predict(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Compute PDR profile for a single pitcher."""
        pitcher_id = kwargs.get("pitcher_id")
        if pitcher_id is None:
            raise ValueError("pitcher_id is required for predict().")
        season = kwargs.get("season")
        return calculate_pdr(conn, pitcher_id, season)

    def evaluate(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Evaluate model quality across all pitcher cliff detections."""
        season = kwargs.get("season")
        min_pitches = kwargs.get("min_pitches", _DEFAULT_MIN_PITCHES)

        leaderboard = batch_calculate(conn, season=season, min_pitches=min_pitches)
        if leaderboard.empty:
            return {"n_pitchers": 0}

        return {
            "n_pitchers": len(leaderboard),
            "median_first_cliff": round(
                float(leaderboard["first_to_die_cliff"].median()), 1,
            ) if leaderboard["first_to_die_cliff"].notna().any() else None,
            "std_first_cliff": round(
                float(leaderboard["first_to_die_cliff"].std()), 1,
            ) if leaderboard["first_to_die_cliff"].notna().any() else None,
        }
