"""
Viscoelastic Workload Response (VWR) model -- biomechanical arm stress-strain
modeling for pitchers.

Models the pitcher's arm as a Standard Linear Solid (SLS) viscoelastic material.
Each pitch applies a stress proportional to (velocity / velo_max)^2 times a
pitch-type effort multiplier.  Cumulative strain is computed via Boltzmann
superposition with an SLS creep compliance function:

    J(dt) = 1/E1 + (1/E2)(1 - exp(-dt / tau))

where E1 governs permanent strain, E2 governs recoverable strain, and tau is
the recovery time constant (hours).  Pitcher-specific parameters are fitted
by minimizing error between predicted strain and observed release-point drift.

The VWR score is the percentile rank of a pitcher's current strain within
their own career distribution (0-100, higher = more strained).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import percentileofscore

from src.analytics.base import BaseAnalyticsModel

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_MIN_CAREER_PITCHES: int = 500

# Default SLS parameters
_DEFAULT_E1: float = 100.0    # Permanent strain modulus
_DEFAULT_E2: float = 50.0     # Recoverable strain modulus
_DEFAULT_TAU: float = 48.0    # Recovery time constant (hours)

# Time between consecutive pitches within a game (hours)
_WITHIN_GAME_DT: float = 30.0 / 3600.0  # 30 seconds in hours

# Pitch-type effort multipliers
EFFORT_MULTIPLIERS: dict[str, float] = {
    "FF": 1.00,  # 4-seam fastball
    "SI": 1.00,  # sinker
    "FC": 0.95,  # cutter
    "SL": 0.85,  # slider
    "CU": 0.80,  # curveball
    "CH": 0.75,  # changeup
    "FS": 0.85,  # splitter
    "ST": 0.85,  # sweeper
}
_DEFAULT_EFFORT: float = 0.85

# Parameter bounds for fitting
_E1_BOUNDS = (20.0, 500.0)
_E2_BOUNDS = (10.0, 300.0)
_TAU_BOUNDS = (6.0, 168.0)  # 6 hours to 7 days


# ── Pure functions ─────────────────────────────────────────────────────────────


def creep_compliance(dt: float | np.ndarray, E1: float, E2: float, tau: float) -> float | np.ndarray:
    """Standard Linear Solid creep compliance function.

    J(dt) = 1/E1 + (1/E2)(1 - exp(-dt / tau))

    Args:
        dt: Time delta in hours (scalar or array).
        E1: Permanent strain modulus.
        E2: Recoverable strain modulus.
        tau: Recovery time constant in hours.

    Returns:
        Creep compliance value(s).
    """
    return 1.0 / E1 + (1.0 / E2) * (1.0 - np.exp(-np.asarray(dt) / tau))


def compute_pitch_stress(pitches_df: pd.DataFrame) -> pd.Series:
    """Compute mechanical stress for each pitch.

    stress(i) = (velo(i) / velo_max)^2 * effort_multiplier(pitch_type)

    Args:
        pitches_df: DataFrame with ``release_speed`` and ``pitch_type`` columns.

    Returns:
        Series of per-pitch stress values aligned to the input index.
    """
    if pitches_df.empty:
        return pd.Series(dtype=float)

    velo = pitches_df["release_speed"].astype(float)
    velo_max = velo.max()
    if velo_max == 0 or pd.isna(velo_max):
        return pd.Series(0.0, index=pitches_df.index)

    velo_ratio_sq = (velo / velo_max) ** 2

    effort = pitches_df["pitch_type"].map(EFFORT_MULTIPLIERS).fillna(_DEFAULT_EFFORT)

    return (velo_ratio_sq * effort).astype(float)


def compute_strain_state(
    stress_events: np.ndarray,
    time_deltas: np.ndarray,
    E1: float = _DEFAULT_E1,
    E2: float = _DEFAULT_E2,
    tau: float = _DEFAULT_TAU,
) -> np.ndarray:
    """Compute cumulative strain via Boltzmann superposition.

    For each pitch i, the total strain is:
        epsilon(i) = sum_{j=0}^{i} sigma(j) * J(t_i - t_j)

    where t_i - t_j is the elapsed time from pitch j to pitch i.

    Args:
        stress_events: Array of per-pitch stress values.
        time_deltas: Array of time deltas (hours) between consecutive pitches.
                     Length = len(stress_events) - 1.
        E1: Permanent strain modulus.
        E2: Recoverable strain modulus.
        tau: Recovery time constant (hours).

    Returns:
        Array of cumulative strain values (same length as stress_events).
    """
    n = len(stress_events)
    if n == 0:
        return np.array([])

    # Build cumulative time from start for each pitch
    cum_time = np.zeros(n)
    if len(time_deltas) > 0:
        cum_time[1:] = np.cumsum(time_deltas[:n - 1])

    strain = np.zeros(n)
    for i in range(n):
        # Superposition: sum contributions from all pitches j <= i
        dt = cum_time[i] - cum_time[: i + 1]
        J_vals = creep_compliance(dt, E1, E2, tau)
        strain[i] = np.sum(stress_events[: i + 1] * J_vals)

    return strain


def _compute_time_deltas(pitches_df: pd.DataFrame) -> np.ndarray:
    """Compute time deltas (hours) between consecutive pitches.

    Within the same game, assumes ~30 seconds between pitches.
    Between games, uses (game_date difference) * 24 hours.

    Args:
        pitches_df: DataFrame with ``game_date`` and ``game_pk`` columns,
                    ordered chronologically.

    Returns:
        Array of time deltas with length len(pitches_df) - 1.
    """
    n = len(pitches_df)
    if n <= 1:
        return np.array([])

    game_pks = pitches_df["game_pk"].values
    game_dates = pd.to_datetime(pitches_df["game_date"]).values

    deltas = np.full(n - 1, _WITHIN_GAME_DT)

    for i in range(n - 1):
        if game_pks[i] != game_pks[i + 1]:
            # Different game -- compute time delta from dates
            d1 = pd.Timestamp(game_dates[i])
            d2 = pd.Timestamp(game_dates[i + 1])
            hours_diff = (d2 - d1).total_seconds() / 3600.0
            deltas[i] = max(hours_diff, 1.0)  # floor at 1 hour

    return deltas


def _query_pitcher_pitches(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    as_of_date: Optional[str] = None,
) -> pd.DataFrame:
    """Retrieve pitch-level data for a pitcher, ordered chronologically.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        as_of_date: Optional date cutoff (ISO string, inclusive).

    Returns:
        DataFrame ordered by game_date, at_bat_number, pitch_number.
    """
    date_filter = "AND game_date <= $2" if as_of_date else ""
    params: list = [pitcher_id]
    if as_of_date:
        params.append(as_of_date)

    query = f"""
        SELECT
            game_pk,
            game_date,
            pitch_type,
            release_speed,
            release_pos_x,
            release_pos_z,
            at_bat_number,
            pitch_number
        FROM pitches
        WHERE pitcher_id = $1
          AND release_speed IS NOT NULL
          AND pitch_type IS NOT NULL
          {date_filter}
        ORDER BY game_date, at_bat_number, pitch_number
    """
    return conn.execute(query, params).fetchdf()


def _compute_release_point_drift(pitches_df: pd.DataFrame) -> np.ndarray:
    """Compute per-pitch release-point drift from the baseline (career mean).

    Uses Euclidean distance in (release_pos_x, release_pos_z) space.
    This serves as a proxy for biomechanical strain.

    Args:
        pitches_df: DataFrame with ``release_pos_x`` and ``release_pos_z``.

    Returns:
        Array of drift values (same length as pitches_df).
    """
    has_x = "release_pos_x" in pitches_df.columns
    has_z = "release_pos_z" in pitches_df.columns

    if not has_x or not has_z:
        return np.full(len(pitches_df), np.nan)

    x = pitches_df["release_pos_x"].astype(float)
    z = pitches_df["release_pos_z"].astype(float)

    # Baseline is the mean of the first 50 pitches (fresh state)
    n_baseline = min(50, len(pitches_df))
    x_base = x.iloc[:n_baseline].mean()
    z_base = z.iloc[:n_baseline].mean()

    drift = np.sqrt((x - x_base) ** 2 + (z - z_base) ** 2)
    return drift.values


def fit_pitcher_parameters(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
) -> dict[str, Any]:
    """Fit pitcher-specific SLS parameters (E1, E2, tau) by minimizing
    error between predicted strain and observed release-point drift.

    Requires at least 500 career pitches.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.

    Returns:
        Dictionary with ``E1``, ``E2``, ``tau``, ``fit_error``,
        ``n_pitches``, and ``converged`` flag.
    """
    logger.info("Fitting VWR parameters for pitcher %d.", pitcher_id)

    df = _query_pitcher_pitches(conn, pitcher_id)
    n = len(df)

    result: dict[str, Any] = {
        "pitcher_id": pitcher_id,
        "E1": _DEFAULT_E1,
        "E2": _DEFAULT_E2,
        "tau": _DEFAULT_TAU,
        "fit_error": None,
        "n_pitches": n,
        "converged": False,
    }

    if n < _MIN_CAREER_PITCHES:
        logger.warning(
            "Pitcher %d has only %d pitches (need %d). Using defaults.",
            pitcher_id, n, _MIN_CAREER_PITCHES,
        )
        return result

    # Compute observed drift
    drift = _compute_release_point_drift(df)
    if np.all(np.isnan(drift)):
        logger.warning("No release point data for pitcher %d. Using defaults.", pitcher_id)
        return result

    # Compute stress events and time deltas
    stress = compute_pitch_stress(df).values
    time_deltas = _compute_time_deltas(df)

    # Normalise drift to [0, 1] range for fitting
    drift_valid = drift[~np.isnan(drift)]
    if len(drift_valid) == 0 or drift_valid.max() == 0:
        return result

    drift_norm = drift / drift_valid.max()
    drift_norm = np.nan_to_num(drift_norm, nan=0.0)

    # Subsample for performance (every Nth pitch)
    max_fit_points = 2000
    if n > max_fit_points:
        step = n // max_fit_points
        indices = np.arange(0, n, step)
    else:
        indices = np.arange(n)

    def objective(params):
        e1, e2, tau_p = params
        predicted_strain = compute_strain_state(stress, time_deltas, e1, e2, tau_p)
        # Normalise predicted strain
        ps_max = predicted_strain.max()
        if ps_max == 0:
            return 1e6
        ps_norm = predicted_strain / ps_max
        # MSE on subsampled points
        diff = ps_norm[indices] - drift_norm[indices]
        return float(np.mean(diff ** 2))

    try:
        res = minimize(
            objective,
            x0=[_DEFAULT_E1, _DEFAULT_E2, _DEFAULT_TAU],
            bounds=[_E1_BOUNDS, _E2_BOUNDS, _TAU_BOUNDS],
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-8},
        )
        result["E1"] = round(float(res.x[0]), 2)
        result["E2"] = round(float(res.x[1]), 2)
        result["tau"] = round(float(res.x[2]), 2)
        result["fit_error"] = round(float(res.fun), 6)
        result["converged"] = bool(res.success)

        logger.info(
            "Fitted VWR for pitcher %d: E1=%.1f, E2=%.1f, tau=%.1f hrs (error=%.6f).",
            pitcher_id, result["E1"], result["E2"], result["tau"], result["fit_error"],
        )
    except Exception as exc:
        logger.warning("VWR parameter fit failed for pitcher %d: %s", pitcher_id, exc)

    return result


def calculate_vwr(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    as_of_date: Optional[str] = None,
) -> dict[str, Any]:
    """Calculate the current VWR score for a pitcher.

    The VWR score is the percentile rank of the current strain value
    within the pitcher's career strain distribution.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        as_of_date: Optional date cutoff (ISO string).

    Returns:
        Dictionary with ``pitcher_id``, ``vwr_score``, ``current_strain``,
        ``strain_history``, ``parameters``, and ``game_appearances``.
    """
    logger.info("Calculating VWR for pitcher %d (as_of=%s).", pitcher_id, as_of_date)

    # Fit or use default parameters
    params = fit_pitcher_parameters(conn, pitcher_id)

    df = _query_pitcher_pitches(conn, pitcher_id, as_of_date)

    if df.empty:
        return {
            "pitcher_id": pitcher_id,
            "vwr_score": None,
            "current_strain": None,
            "strain_history": [],
            "parameters": params,
            "game_appearances": [],
        }

    stress = compute_pitch_stress(df).values
    time_deltas = _compute_time_deltas(df)
    strain = compute_strain_state(
        stress, time_deltas,
        E1=params["E1"], E2=params["E2"], tau=params["tau"],
    )

    # VWR score: percentile rank of current strain in career distribution
    current_strain = float(strain[-1])
    vwr_score = float(percentileofscore(strain, current_strain, kind="rank"))
    vwr_score = round(min(max(vwr_score, 0.0), 100.0), 1)

    # Build per-game summary for timeline
    game_appearances = []
    df_with_strain = df.copy()
    df_with_strain["strain"] = strain
    for game_pk, game_df in df_with_strain.groupby("game_pk", sort=False):
        game_appearances.append({
            "game_pk": int(game_pk),
            "game_date": str(game_df["game_date"].iloc[0]),
            "pitches": len(game_df),
            "peak_strain": round(float(game_df["strain"].max()), 4),
            "end_strain": round(float(game_df["strain"].iloc[-1]), 4),
        })

    return {
        "pitcher_id": pitcher_id,
        "vwr_score": vwr_score,
        "current_strain": round(current_strain, 4),
        "strain_history": strain.tolist(),
        "parameters": params,
        "game_appearances": game_appearances,
    }


def batch_calculate(
    conn: duckdb.DuckDBPyConnection,
    season: Optional[int] = None,
    min_pitches: int = _MIN_CAREER_PITCHES,
) -> pd.DataFrame:
    """Compute a VWR leaderboard for all qualifying pitchers in a season.

    Args:
        conn: Open DuckDB connection.
        season: Season year to restrict to (optional).
        min_pitches: Minimum career pitches to qualify.

    Returns:
        DataFrame with columns: pitcher_id, name, vwr_score,
        current_strain, E1, E2, tau, n_pitches.
    """
    logger.info("Batch calculating VWR (season=%s, min_pitches=%d).", season, min_pitches)

    season_filter = "AND EXTRACT(YEAR FROM game_date) = $1" if season else ""
    params: list = []
    if season:
        params.append(season)

    query = f"""
        SELECT pitcher_id, COUNT(*) AS pitch_count
        FROM pitches
        WHERE release_speed IS NOT NULL
          AND pitch_type IS NOT NULL
          {season_filter}
        GROUP BY pitcher_id
        HAVING COUNT(*) >= {int(min_pitches)}
    """
    qualifying = conn.execute(query, params).fetchdf()

    empty_cols = [
        "pitcher_id", "name", "vwr_score", "current_strain",
        "E1", "E2", "tau", "n_pitches",
    ]
    if qualifying.empty:
        logger.warning("No pitchers qualify with min_pitches=%d.", min_pitches)
        return pd.DataFrame(columns=empty_cols)

    rows: list[dict] = []
    for pitcher_id in qualifying["pitcher_id"].tolist():
        result = calculate_vwr(conn, int(pitcher_id))
        if result["vwr_score"] is not None:
            p = result["parameters"]
            rows.append({
                "pitcher_id": int(pitcher_id),
                "vwr_score": result["vwr_score"],
                "current_strain": result["current_strain"],
                "E1": p["E1"],
                "E2": p["E2"],
                "tau": p["tau"],
                "n_pitches": p["n_pitches"],
            })

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
    front_cols = ["pitcher_id", "name", "vwr_score", "current_strain", "E1", "E2", "tau", "n_pitches"]
    other_cols = sorted(c for c in result_df.columns if c not in front_cols)
    result_df = result_df[[c for c in front_cols if c in result_df.columns] + other_cols]

    result_df = result_df.sort_values("vwr_score", ascending=False).reset_index(drop=True)
    logger.info("Batch VWR complete: %d pitchers.", len(result_df))
    return result_df


def predict_recovery(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    rest_days: int,
) -> dict[str, Any]:
    """Predict the VWR score after a given number of rest days.

    Simulates recovery by advancing time from the last pitch and
    recomputing strain using the SLS model's natural decay.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        rest_days: Number of rest days to project.

    Returns:
        Dictionary with ``pitcher_id``, ``rest_days``,
        ``projected_vwr``, ``projected_strain``, ``current_vwr``,
        ``current_strain``, and ``recovery_curve`` (list of
        {day, vwr, strain} dicts for each rest day).
    """
    logger.info("Predicting recovery for pitcher %d (%d rest days).", pitcher_id, rest_days)

    # Get current state
    current = calculate_vwr(conn, pitcher_id)

    if current["vwr_score"] is None or not current["strain_history"]:
        return {
            "pitcher_id": pitcher_id,
            "rest_days": rest_days,
            "projected_vwr": None,
            "projected_strain": None,
            "current_vwr": None,
            "current_strain": None,
            "recovery_curve": [],
        }

    params = current["parameters"]
    E1 = params["E1"]
    E2 = params["E2"]
    tau = params["tau"]

    # Rebuild full state to project from
    df = _query_pitcher_pitches(conn, pitcher_id)
    stress = compute_pitch_stress(df).values
    time_deltas = _compute_time_deltas(df)

    # Build cumulative times from the original data
    n = len(stress)
    cum_time = np.zeros(n)
    if len(time_deltas) > 0:
        cum_time[1:] = np.cumsum(time_deltas[:n - 1])

    strain_history = np.array(current["strain_history"])

    # Project forward: for each rest day, compute strain as if no new
    # pitches are thrown but time advances
    recovery_curve: list[dict] = []
    for day in range(rest_days + 1):
        extra_hours = day * 24.0
        # Recompute strain at the "current time + extra_hours" point
        # using Boltzmann superposition from all past pitches
        t_now = cum_time[-1] + extra_hours if n > 0 else extra_hours
        dt_array = t_now - cum_time
        J_vals = creep_compliance(dt_array, E1, E2, tau)
        projected_strain = float(np.sum(stress * J_vals))

        # Percentile within career distribution
        projected_vwr = float(percentileofscore(
            strain_history, projected_strain, kind="rank",
        ))
        projected_vwr = round(min(max(projected_vwr, 0.0), 100.0), 1)

        recovery_curve.append({
            "day": day,
            "vwr": projected_vwr,
            "strain": round(projected_strain, 4),
        })

    return {
        "pitcher_id": pitcher_id,
        "rest_days": rest_days,
        "projected_vwr": recovery_curve[-1]["vwr"] if recovery_curve else None,
        "projected_strain": recovery_curve[-1]["strain"] if recovery_curve else None,
        "current_vwr": current["vwr_score"],
        "current_strain": current["current_strain"],
        "recovery_curve": recovery_curve,
    }


# ── BaseAnalyticsModel subclass ───────────────────────────────────────────────


class ViscoelasticWorkloadModel(BaseAnalyticsModel):
    """Biomechanical arm stress-strain model for pitchers.

    Wraps the functional API above into the project's standard
    ``BaseAnalyticsModel`` lifecycle for consistency with other models.
    """

    @property
    def model_name(self) -> str:
        return "viscoelastic_workload_response"

    @property
    def version(self) -> str:
        return "1.0.0"

    def train(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Train (batch-fit) VWR for all qualifying pitchers.

        Returns summary metrics.
        """
        season = kwargs.get("season")
        min_pitches = kwargs.get("min_pitches", _MIN_CAREER_PITCHES)

        leaderboard = batch_calculate(conn, season=season, min_pitches=min_pitches)
        self._leaderboard = leaderboard

        metrics: dict[str, Any] = {
            "n_pitchers": len(leaderboard),
            "median_vwr": round(float(leaderboard["vwr_score"].median()), 1)
            if not leaderboard.empty else None,
            "mean_strain": round(float(leaderboard["current_strain"].mean()), 4)
            if not leaderboard.empty else None,
        }
        self.set_training_metadata(
            metrics=metrics,
            params={"season": season, "min_pitches": min_pitches},
        )
        return metrics

    def predict(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Calculate VWR for a specific pitcher."""
        pitcher_id = kwargs.get("pitcher_id")
        if pitcher_id is None:
            raise ValueError("pitcher_id is required for predict().")
        as_of_date = kwargs.get("as_of_date")
        return calculate_vwr(conn, pitcher_id, as_of_date)

    def evaluate(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Evaluate model across all qualifying pitchers."""
        season = kwargs.get("season")
        min_pitches = kwargs.get("min_pitches", _MIN_CAREER_PITCHES)

        leaderboard = batch_calculate(conn, season=season, min_pitches=min_pitches)
        if leaderboard.empty:
            return {"n_pitchers": 0, "avg_strain": 0}

        return {
            "n_pitchers": len(leaderboard),
            "median_vwr": round(float(leaderboard["vwr_score"].median()), 1),
            "mean_strain": round(float(leaderboard["current_strain"].mean()), 4),
            "std_strain": round(float(leaderboard["current_strain"].std()), 4),
            "median_tau": round(float(leaderboard["tau"].median()), 1),
        }
