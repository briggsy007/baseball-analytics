"""
Kinetic Half-Life (K1/2) model -- pharmacokinetic-inspired pitcher stamina decay.

Models pitcher stamina as exponential decay of pitch quality ("Stuff Concentration")
over the course of a game.  For each game appearance the model computes a rolling
window of composite pitch quality, fits an exponential decay curve
S(n) = S_peak * exp(-lambda * n), and derives the half-life K1/2 = ln(2) / lambda.

Pitchers with higher K1/2 values maintain their stuff deeper into games.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.analytics.base import BaseAnalyticsModel
from src.analytics.features import (
    compute_z_scores,
    compute_stuff_quality,
    STUFF_QUALITY_COLUMNS,
    STUFF_QUALITY_WEIGHTS,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_MIN_PITCHES_PER_GAME: int = 50
_MIN_PITCHES_PER_GAME_PT: int = 15  # Lower threshold for per-pitch-type analysis
_ROLLING_WINDOW: int = 10
_MIN_ROLLING_PERIODS: int = 5
_MIN_FIT_R_SQUARED: float = 0.05
_DEFAULT_MIN_PITCHES: int = 500

# Re-export shared quality columns/weights for backward compatibility
_QUALITY_COLUMNS = STUFF_QUALITY_COLUMNS
_QUALITY_WEIGHTS = STUFF_QUALITY_WEIGHTS


# ── Pure functions ─────────────────────────────────────────────────────────────


def _exp_decay(n: np.ndarray, s_peak: float, lam: float) -> np.ndarray:
    """Exponential decay model: S(n) = s_peak * exp(-lam * n)."""
    return s_peak * np.exp(-lam * n)


def compute_stuff_concentration(pitches_df: pd.DataFrame) -> pd.Series:
    """Compute a per-pitch composite quality score (Stuff Concentration).

    The score is a weighted combination of z-scored physical attributes:
    velocity, spin rate, horizontal/vertical movement, and release-point
    consistency.  Higher values indicate better raw stuff on that pitch.

    Delegates to :func:`~src.analytics.features.compute_stuff_quality`.

    Args:
        pitches_df: DataFrame with at least the columns in ``_QUALITY_COLUMNS``.

    Returns:
        A Series of per-pitch Stuff Concentration scores aligned to the
        input index.
    """
    return compute_stuff_quality(pitches_df, _QUALITY_COLUMNS, _QUALITY_WEIGHTS)


def fit_decay_curve(
    stuff_series: pd.Series,
    pitch_numbers: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Fit an exponential decay curve to a Stuff Concentration time-series.

    Uses ``scipy.optimize.curve_fit`` to estimate S_peak and lambda in
    S(n) = S_peak * exp(-lambda * n).

    Args:
        stuff_series: Rolling-averaged Stuff Concentration values.
        pitch_numbers: Corresponding pitch ordinal numbers within the game
                       (0-indexed).

    Returns:
        Dictionary with keys ``lambda``, ``half_life``, ``r_squared``,
        ``peak``, and ``n_points``.  Returns ``None`` values if the fit
        fails or data is insufficient.
    """
    mask = stuff_series.notna()
    y = stuff_series[mask].values.astype(float)
    x = np.asarray(pitch_numbers)[mask.values].astype(float)

    result: dict[str, Any] = {
        "lambda": None,
        "half_life": None,
        "r_squared": None,
        "peak": None,
        "n_points": len(y),
    }

    if len(y) < 10:
        logger.debug("Too few points (%d) for decay fit.", len(y))
        return result

    # Initial guesses
    s_peak_guess = float(np.max(y)) if np.max(y) > 0 else 1.0
    lam_guess = 0.005  # gentle decay

    try:
        popt, _ = curve_fit(
            _exp_decay,
            x,
            y,
            p0=[s_peak_guess, lam_guess],
            bounds=([0, 0], [np.inf, 1.0]),
            maxfev=5000,
        )
        s_peak_fit, lam_fit = popt

        # Goodness of fit
        y_pred = _exp_decay(x, s_peak_fit, lam_fit)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        half_life = math.log(2) / lam_fit if lam_fit > 1e-10 else np.inf

        result["lambda"] = round(float(lam_fit), 6)
        result["half_life"] = round(float(half_life), 1) if np.isfinite(half_life) else None
        result["r_squared"] = round(float(r_squared), 4)
        result["peak"] = round(float(s_peak_fit), 4)

    except (RuntimeError, ValueError, TypeError) as exc:
        logger.debug("Decay curve fit failed: %s", exc)

    return result


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
          {season_filter}
        ORDER BY game_date, at_bat_number, pitch_number
    """
    return conn.execute(query, params).fetchdf()


def _fit_game(
    game_df: pd.DataFrame,
    pitch_type_filter: Optional[str] = None,
) -> dict[str, Any] | None:
    """Fit a decay curve for one game appearance.

    Returns None if the game does not have enough pitches or the
    exponential fit quality is too poor.
    """
    df = game_df.copy()
    if pitch_type_filter is not None:
        df = df[df["pitch_type"] == pitch_type_filter]

    min_pitches = _MIN_PITCHES_PER_GAME_PT if pitch_type_filter is not None else _MIN_PITCHES_PER_GAME
    if len(df) < min_pitches:
        return None

    # Compute stuff concentration
    sc = compute_stuff_concentration(df)
    # Rolling average to smooth noise
    sc_smooth = sc.rolling(window=_ROLLING_WINDOW, min_periods=_MIN_ROLLING_PERIODS).mean()

    pitch_ordinals = np.arange(len(df))

    fit = fit_decay_curve(sc_smooth, pitch_ordinals)

    if fit["r_squared"] is not None and fit["r_squared"] < _MIN_FIT_R_SQUARED:
        logger.debug(
            "Rejecting fit for game (r²=%.4f < %.2f).",
            fit["r_squared"],
            _MIN_FIT_R_SQUARED,
        )
        return None

    if fit["half_life"] is None:
        return None

    fit["n_pitches"] = len(df)
    fit["game_pk"] = int(df["game_pk"].iloc[0])
    fit["game_date"] = df["game_date"].iloc[0]
    return fit


def calculate_half_life(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> dict[str, Any]:
    """Calculate K1/2 (Kinetic Half-Life) for a single pitcher.

    Retrieves all game appearances, fits per-game decay curves, and
    aggregates into an overall and per-pitch-type K1/2.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        season: Restrict to this year (optional).

    Returns:
        Dictionary with keys:
        - ``pitcher_id``
        - ``overall_half_life``  (median across qualifying games)
        - ``overall_lambda``     (median lambda)
        - ``games_fitted``       (number of games used)
        - ``per_pitch_type``     dict mapping pitch type -> {half_life, lambda, games_fitted}
        - ``game_curves``        list of per-game fit dicts
    """
    logger.info("Calculating K½ for pitcher %d (season=%s).", pitcher_id, season)

    df = _query_pitcher_game_pitches(conn, pitcher_id, season)
    if df.empty:
        logger.warning("No pitches found for pitcher %d.", pitcher_id)
        return {
            "pitcher_id": pitcher_id,
            "overall_half_life": None,
            "overall_lambda": None,
            "games_fitted": 0,
            "per_pitch_type": {},
            "game_curves": [],
        }

    # ── Overall K½ across all pitch types ─────────────────────────────────
    game_fits: list[dict] = []
    for game_pk, game_df in df.groupby("game_pk"):
        fit = _fit_game(game_df)
        if fit is not None:
            game_fits.append(fit)

    overall_hl: float | None = None
    overall_lam: float | None = None
    if game_fits:
        half_lives = [f["half_life"] for f in game_fits if f["half_life"] is not None]
        lambdas = [f["lambda"] for f in game_fits if f["lambda"] is not None]
        if half_lives:
            overall_hl = round(float(np.median(half_lives)), 1)
        if lambdas:
            overall_lam = round(float(np.median(lambdas)), 6)

    # ── Per-pitch-type K½ ─────────────────────────────────────────────────
    per_pitch_type: dict[str, dict] = {}
    pitch_types = df["pitch_type"].dropna().unique()
    # Pre-compute per-game pitch type counts to avoid expensive _fit_game calls
    # on games that won't qualify anyway
    _pt_counts = df.groupby(["game_pk", "pitch_type"]).size().reset_index(name="_cnt")
    for pt in pitch_types:
        # Only iterate over games where this pitch type has enough pitches
        qualifying_games = set(
            _pt_counts.loc[
                (_pt_counts["pitch_type"] == pt) & (_pt_counts["_cnt"] >= _MIN_PITCHES_PER_GAME_PT),
                "game_pk",
            ]
        )
        pt_fits: list[dict] = []
        for game_pk, game_df in df.groupby("game_pk"):
            if game_pk not in qualifying_games:
                continue
            fit = _fit_game(game_df, pitch_type_filter=pt)
            if fit is not None:
                pt_fits.append(fit)
        if pt_fits:
            pt_hls = [f["half_life"] for f in pt_fits if f["half_life"] is not None]
            pt_lams = [f["lambda"] for f in pt_fits if f["lambda"] is not None]
            per_pitch_type[pt] = {
                "half_life": round(float(np.median(pt_hls)), 1) if pt_hls else None,
                "lambda": round(float(np.median(pt_lams)), 6) if pt_lams else None,
                "games_fitted": len(pt_fits),
            }

    return {
        "pitcher_id": pitcher_id,
        "overall_half_life": overall_hl,
        "overall_lambda": overall_lam,
        "games_fitted": len(game_fits),
        "per_pitch_type": per_pitch_type,
        "game_curves": game_fits,
    }


def batch_calculate(
    conn: duckdb.DuckDBPyConnection,
    season: Optional[int] = None,
    min_pitches: int = _DEFAULT_MIN_PITCHES,
) -> pd.DataFrame:
    """Compute a K1/2 leaderboard for all qualifying pitchers.

    Args:
        conn: Open DuckDB connection.
        season: Restrict to this year (optional).
        min_pitches: Minimum total pitches to qualify.

    Returns:
        DataFrame with columns: pitcher_id, name, overall_half_life,
        overall_lambda, games_fitted, plus per-pitch-type K½ columns.
    """
    logger.info("Batch calculating K½ (season=%s, min_pitches=%d).", season, min_pitches)

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
          {season_filter}
        GROUP BY pitcher_id
        HAVING COUNT(*) >= {int(min_pitches)}
    """
    qualifying = conn.execute(query, params).fetchdf()

    if qualifying.empty:
        logger.warning("No pitchers qualify with min_pitches=%d.", min_pitches)
        return pd.DataFrame(columns=[
            "pitcher_id", "name", "overall_half_life", "overall_lambda", "games_fitted",
        ])

    # Compute K½ for qualifying pitchers using the accurate per-game
    # exponential fitting model.  To keep latency reasonable we cap at
    # the top 75 pitchers by pitch volume and cache downstream.
    _MAX_LEADERBOARD = 75
    pitcher_ids = qualifying.sort_values("pitch_count", ascending=False)["pitcher_id"].tolist()
    pitcher_ids = pitcher_ids[:_MAX_LEADERBOARD]

    rows: list[dict] = []
    for pid in pitcher_ids:
        result = calculate_half_life(conn, int(pid), season)
        if result["overall_half_life"] is not None:
            row: dict[str, Any] = {
                "pitcher_id": int(pid),
                "overall_half_life": result["overall_half_life"],
                "overall_lambda": result["overall_lambda"],
                "games_fitted": result["games_fitted"],
            }
            for pt, pt_data in result.get("per_pitch_type", {}).items():
                row[f"{pt}_half_life"] = pt_data.get("half_life")
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=[
            "pitcher_id", "name", "overall_half_life", "overall_lambda", "games_fitted",
        ])

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
    front_cols = ["pitcher_id", "name", "overall_half_life", "overall_lambda", "games_fitted"]
    other_cols = sorted(c for c in result_df.columns if c not in front_cols)
    result_df = result_df[[c for c in front_cols if c in result_df.columns] + other_cols]

    result_df = result_df.sort_values("overall_half_life", ascending=False).reset_index(drop=True)
    logger.info("Batch K½ complete: %d pitchers.", len(result_df))
    return result_df


def predict_game_decay(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    context: Optional[dict] = None,
) -> dict[str, Any]:
    """Predict the decay curve for an upcoming start.

    Uses the pitcher's historical median lambda and peak to project
    Stuff Concentration over a typical 100-pitch outing.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        context: Optional dict with ``days_rest``, ``workload_30d``,
                 ``temperature`` for contextual adjustments.

    Returns:
        Dictionary with ``pitch_numbers``, ``predicted_stuff``,
        ``half_life``, ``lambda``, and ``peak``.
    """
    result = calculate_half_life(conn, pitcher_id)

    if result["overall_half_life"] is None:
        return {
            "pitcher_id": pitcher_id,
            "pitch_numbers": [],
            "predicted_stuff": [],
            "half_life": None,
            "lambda": None,
            "peak": None,
        }

    lam = result["overall_lambda"]
    # Estimate peak from game curves
    peaks = [
        g["peak"] for g in result["game_curves"]
        if g.get("peak") is not None
    ]
    peak = float(np.median(peaks)) if peaks else 1.0

    # Apply contextual adjustments
    if context:
        days_rest = context.get("days_rest", 4)
        # More rest -> slightly lower lambda (slower decay)
        rest_adj = max(0.8, 1.0 - (days_rest - 4) * 0.02)
        lam = lam * rest_adj

        workload = context.get("workload_30d")
        if workload is not None:
            # Higher workload -> faster decay
            workload_adj = 1.0 + max(0, workload - 400) * 0.0005
            lam = lam * workload_adj

    pitch_numbers = list(range(0, 110))
    predicted = [
        round(float(peak * math.exp(-lam * n)), 4) for n in pitch_numbers
    ]

    half_life = round(math.log(2) / lam, 1) if lam > 1e-10 else None

    return {
        "pitcher_id": pitcher_id,
        "pitch_numbers": pitch_numbers,
        "predicted_stuff": predicted,
        "half_life": half_life,
        "lambda": round(lam, 6),
        "peak": round(peak, 4),
    }


def get_game_decay_data(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    game_pk: int,
) -> dict[str, Any]:
    """Retrieve actual decay data for a specific past game.

    Returns the raw stuff concentration, rolling average, and the
    fitted exponential overlay for visualization.
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
        ORDER BY at_bat_number, pitch_number
    """
    df = conn.execute(query, [pitcher_id, game_pk]).fetchdf()

    if df.empty:
        return {
            "pitcher_id": pitcher_id,
            "game_pk": game_pk,
            "pitch_numbers": [],
            "raw_stuff": [],
            "rolling_stuff": [],
            "fitted_stuff": [],
            "fit": None,
        }

    sc = compute_stuff_concentration(df)
    sc_smooth = sc.rolling(window=_ROLLING_WINDOW, min_periods=_MIN_ROLLING_PERIODS).mean()
    pitch_ordinals = np.arange(len(df))

    fit = fit_decay_curve(sc_smooth, pitch_ordinals)

    fitted_values = []
    if fit["peak"] is not None and fit["lambda"] is not None:
        fitted_values = [
            float(fit["peak"] * math.exp(-fit["lambda"] * n))
            for n in range(len(df))
        ]

    return {
        "pitcher_id": pitcher_id,
        "game_pk": game_pk,
        "pitch_numbers": list(range(len(df))),
        "raw_stuff": sc.tolist(),
        "rolling_stuff": sc_smooth.tolist(),
        "fitted_stuff": fitted_values,
        "fit": fit,
    }


# ── BaseAnalyticsModel subclass ───────────────────────────────────────────────


class KineticHalfLifeModel(BaseAnalyticsModel):
    """Pharmacokinetic-inspired pitcher stamina decay model.

    Wraps the functional API above into the project's standard
    ``BaseAnalyticsModel`` lifecycle for consistency with other models.
    """

    @property
    def model_name(self) -> str:
        return "kinetic_half_life"

    @property
    def version(self) -> str:
        return "1.0.0"

    def train(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Train (batch-fit) K½ for all qualifying pitchers.

        Training here means computing the leaderboard and caching
        aggregate statistics.  Returns summary metrics.
        """
        season = kwargs.get("season")
        min_pitches = kwargs.get("min_pitches", _DEFAULT_MIN_PITCHES)

        leaderboard = batch_calculate(conn, season=season, min_pitches=min_pitches)
        self._leaderboard = leaderboard

        metrics: dict[str, Any] = {
            "n_pitchers": len(leaderboard),
            "median_half_life": round(float(leaderboard["overall_half_life"].median()), 1)
            if not leaderboard.empty else None,
            "mean_half_life": round(float(leaderboard["overall_half_life"].mean()), 1)
            if not leaderboard.empty else None,
        }
        self.set_training_metadata(
            metrics=metrics,
            params={"season": season, "min_pitches": min_pitches},
        )
        return metrics

    def predict(
        self,
        conn: duckdb.DuckDBPyConnection,
        **kwargs,
    ) -> dict:
        """Predict decay curve for a pitcher's next start."""
        pitcher_id = kwargs.get("pitcher_id")
        if pitcher_id is None:
            raise ValueError("pitcher_id is required for predict().")
        context = kwargs.get("context")
        return predict_game_decay(conn, pitcher_id, context)

    def evaluate(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Evaluate model fit quality across all pitcher-game fits."""
        season = kwargs.get("season")
        min_pitches = kwargs.get("min_pitches", _DEFAULT_MIN_PITCHES)

        leaderboard = batch_calculate(conn, season=season, min_pitches=min_pitches)
        if leaderboard.empty:
            return {"n_pitchers": 0, "avg_games_fitted": 0}

        return {
            "n_pitchers": len(leaderboard),
            "avg_games_fitted": round(float(leaderboard["games_fitted"].mean()), 1),
            "median_half_life": round(float(leaderboard["overall_half_life"].median()), 1),
            "std_half_life": round(float(leaderboard["overall_half_life"].std()), 1),
        }
