"""
Pitch Sequence Alpha Decay model.

Quantifies how quickly a pitcher's sequencing patterns lose deceptive
effectiveness.  Borrowed from quantitative trading: **alpha** is the
whiff-rate uplift a pitch-pair transition produces above the
unconditional baseline; **alpha decay** tracks how that uplift erodes
with repetition within a game, across a series against the same team,
and over the course of a season.

For each pitch-pair transition (e.g. FF->SL):

    alpha(seq) = E[whiff_rate | sequence] - E[whiff_rate | unconditional]

Decay is modelled as an exponential:

    alpha_t = alpha_0 * exp(-lambda * n)

where *n* is the repetition count and lambda is the decay rate.
Half-life = ln(2) / lambda tells us how many repetitions it takes for
the deceptive advantage to halve.

Seasonal regime detection uses a rolling-window approach: compute a
500-pitch rolling alpha and classify high/low regimes via a threshold
at the population median.

Inherits from ``BaseAnalyticsModel`` to plug into the standard
train / predict / evaluate lifecycle.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.analytics.base import BaseAnalyticsModel

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

MIN_REPETITIONS_FOR_FIT: int = 10
MIN_R_SQUARED: float = 0.01
MIN_PITCHES_DEFAULT: int = 300
ROLLING_WINDOW: int = 500

# Whiff descriptions from Statcast
_WHIFF_DESCRIPTIONS: set[str] = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
    "missed_bunt",
}

# Swing descriptions (whiffs + fouls + contact)
_SWING_DESCRIPTIONS: set[str] = _WHIFF_DESCRIPTIONS | {
    "foul",
    "foul_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}


# ── Exponential decay helper ─────────────────────────────────────────────────


def _exp_decay(n: np.ndarray, alpha_0: float, lam: float) -> np.ndarray:
    """Exponential decay: alpha_0 * exp(-lam * n)."""
    return alpha_0 * np.exp(-lam * n)


def _fit_exponential_decay(
    reps: np.ndarray,
    alphas: np.ndarray,
) -> Optional[dict]:
    """Fit alpha_t = alpha_0 * exp(-lambda * n) to observed data.

    Returns None if the fit fails or produces degenerate parameters.
    """
    if len(reps) < 2:
        return None

    try:
        popt, _ = curve_fit(
            _exp_decay,
            reps.astype(float),
            alphas.astype(float),
            p0=[alphas[0] if alphas[0] != 0 else 0.05, 0.3],
            bounds=([-1.0, 1e-6], [1.0, 10.0]),
            maxfev=5000,
        )
        alpha_0, lam = popt

        # Compute R-squared
        predicted = _exp_decay(reps.astype(float), alpha_0, lam)
        ss_res = np.sum((alphas - predicted) ** 2)
        ss_tot = np.sum((alphas - np.mean(alphas)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        half_life = math.log(2) / lam if lam > 0 else float("inf")

        return {
            "alpha_0": round(float(alpha_0), 6),
            "lambda": round(float(lam), 6),
            "half_life": round(float(half_life), 2),
            "r_squared": round(float(r_squared), 6),
        }

    except (RuntimeError, ValueError, TypeError) as exc:
        logger.debug("Exponential fit failed: %s", exc)
        return None


# ── Data extraction queries ──────────────────────────────────────────────────


def _get_pitch_pairs_with_whiff(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """Return consecutive pitch pairs within each at-bat, enriched with
    whiff/swing flags and intra-game repetition counts.
    """
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    params: list = [pitcher_id]
    if season:
        params.append(season)

    query = f"""
        WITH sequenced AS (
            SELECT
                game_pk,
                game_date,
                pitcher_id,
                batter_id,
                at_bat_number,
                pitch_number,
                pitch_type,
                description,
                stand,
                LAG(pitch_type) OVER (
                    PARTITION BY game_pk, at_bat_number
                    ORDER BY pitch_number
                ) AS prev_pitch_type,
                CASE WHEN description IN (
                    'swinging_strike', 'swinging_strike_blocked',
                    'foul_tip', 'missed_bunt'
                ) THEN 1 ELSE 0 END AS is_whiff,
                CASE WHEN description IN (
                    'swinging_strike', 'swinging_strike_blocked',
                    'foul_tip', 'missed_bunt', 'foul', 'foul_bunt',
                    'hit_into_play', 'hit_into_play_no_out',
                    'hit_into_play_score'
                ) THEN 1 ELSE 0 END AS is_swing
            FROM pitches
            WHERE pitcher_id = $1
              AND pitch_type IS NOT NULL
              {season_filter}
        )
        SELECT
            s.*,
            prev_pitch_type || '->' || pitch_type AS transition,
            ROW_NUMBER() OVER (
                PARTITION BY game_pk, prev_pitch_type, pitch_type
                ORDER BY at_bat_number, pitch_number
            ) AS game_transition_rep
        FROM sequenced s
        WHERE prev_pitch_type IS NOT NULL
        ORDER BY game_date, at_bat_number, pitch_number
    """
    return conn.execute(query, params).fetchdf()


def _get_series_info(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """Return game-level info to identify series vs the same opponent.

    A 'series' is defined as consecutive games against the same opponent
    with no more than 4 days between games.
    """
    season_filter = "AND EXTRACT(YEAR FROM p.game_date) = $2" if season else ""
    params: list = [pitcher_id]
    if season:
        params.append(season)

    query = f"""
        SELECT DISTINCT
            p.game_pk,
            p.game_date,
            g.home_team,
            g.away_team
        FROM pitches p
        LEFT JOIN games g ON p.game_pk = g.game_pk
        WHERE p.pitcher_id = $1
          {season_filter}
        ORDER BY p.game_date
    """
    return conn.execute(query, params).fetchdf()


# ── Core computation functions ───────────────────────────────────────────────


def compute_sequence_alpha(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> dict[str, dict]:
    """Compute alpha (whiff rate uplift) for each pitch-pair transition.

    For each transition (e.g. FF->SL):
        alpha = E[whiff_rate | transition] - E[whiff_rate | pitch_type unconditional]

    Only considers swings (whiff_rate = whiffs / swings).

    Returns:
        Dict keyed by transition string -> {
            "conditional_whiff_rate": float,
            "unconditional_whiff_rate": float,
            "alpha": float,
            "swing_count": int,
            "whiff_count": int,
        }
    """
    pairs = _get_pitch_pairs_with_whiff(conn, pitcher_id, season)

    if pairs.empty:
        return {}

    # Filter to swings only for whiff rate
    swings = pairs[pairs["is_swing"] == 1].copy()

    if swings.empty:
        return {}

    # Unconditional whiff rate per pitch type (the second pitch's type)
    uncond = (
        swings.groupby("pitch_type")
        .agg(total_swings=("is_swing", "sum"), total_whiffs=("is_whiff", "sum"))
        .reset_index()
    )
    uncond["unconditional_whiff_rate"] = uncond["total_whiffs"] / uncond["total_swings"]
    uncond_map = dict(zip(uncond["pitch_type"], uncond["unconditional_whiff_rate"]))

    # Conditional whiff rate per transition
    cond = (
        swings.groupby("transition")
        .agg(swing_count=("is_swing", "sum"), whiff_count=("is_whiff", "sum"))
        .reset_index()
    )
    cond["conditional_whiff_rate"] = cond["whiff_count"] / cond["swing_count"]

    result = {}
    for _, row in cond.iterrows():
        trans = row["transition"]
        parts = trans.split("->")
        if len(parts) != 2:
            continue
        curr_type = parts[1]
        uncond_rate = uncond_map.get(curr_type, 0.0)
        alpha = row["conditional_whiff_rate"] - uncond_rate

        result[trans] = {
            "conditional_whiff_rate": round(float(row["conditional_whiff_rate"]), 6),
            "unconditional_whiff_rate": round(float(uncond_rate), 6),
            "alpha": round(float(alpha), 6),
            "swing_count": int(row["swing_count"]),
            "whiff_count": int(row["whiff_count"]),
        }

    return result


def fit_intra_game_decay(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> dict[str, dict]:
    """Fit intra-game alpha decay for each transition.

    Groups pitch pairs by in-game repetition number (game_transition_rep),
    computes whiff rate at each rep, derives alpha, and fits an
    exponential decay curve.

    Returns:
        Dict keyed by transition string -> {
            "fit": {alpha_0, lambda, half_life, r_squared} or None,
            "alpha_by_rep": [{rep, alpha, whiff_rate, n_swings}, ...],
        }
    """
    pairs = _get_pitch_pairs_with_whiff(conn, pitcher_id, season)

    if pairs.empty:
        return {}

    swings = pairs[pairs["is_swing"] == 1].copy()
    if swings.empty:
        return {}

    # Unconditional whiff rate per pitch type
    uncond = swings.groupby("pitch_type").agg(
        total_swings=("is_swing", "sum"), total_whiffs=("is_whiff", "sum")
    )
    uncond["rate"] = uncond["total_whiffs"] / uncond["total_swings"]
    uncond_map = uncond["rate"].to_dict()

    result = {}
    for trans, grp in swings.groupby("transition"):
        parts = trans.split("->")
        if len(parts) != 2:
            continue
        curr_type = parts[1]
        base_rate = uncond_map.get(curr_type, 0.0)

        # Group by repetition number
        rep_stats = (
            grp.groupby("game_transition_rep")
            .agg(n_swings=("is_swing", "sum"), n_whiffs=("is_whiff", "sum"))
            .reset_index()
        )
        rep_stats["whiff_rate"] = rep_stats["n_whiffs"] / rep_stats["n_swings"]
        rep_stats["alpha"] = rep_stats["whiff_rate"] - base_rate

        alpha_by_rep = [
            {
                "rep": int(row["game_transition_rep"]),
                "alpha": round(float(row["alpha"]), 6),
                "whiff_rate": round(float(row["whiff_rate"]), 6),
                "n_swings": int(row["n_swings"]),
            }
            for _, row in rep_stats.iterrows()
        ]

        # Fit exponential decay if enough data
        fit_result = None
        total_reps = len(rep_stats)
        if total_reps >= MIN_REPETITIONS_FOR_FIT:
            reps_arr = rep_stats["game_transition_rep"].values
            alphas_arr = rep_stats["alpha"].values
            fit_result = _fit_exponential_decay(reps_arr, alphas_arr)
            if fit_result and fit_result["r_squared"] < MIN_R_SQUARED:
                fit_result = None

        result[trans] = {
            "fit": fit_result,
            "alpha_by_rep": alpha_by_rep,
        }

    return result


def fit_series_decay(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> dict[str, dict]:
    """Fit cross-game (series) alpha decay for each transition.

    Within a series against the same team, tracks alpha by game number
    and fits an exponential decay.

    Returns:
        Dict keyed by transition string -> {
            "fit": {alpha_0, lambda, half_life, r_squared} or None,
            "alpha_by_game": [{game_num, alpha, whiff_rate, n_swings}, ...],
        }
    """
    pairs = _get_pitch_pairs_with_whiff(conn, pitcher_id, season)
    series_info = _get_series_info(conn, pitcher_id, season)

    if pairs.empty or series_info.empty:
        return {}

    swings = pairs[pairs["is_swing"] == 1].copy()
    if swings.empty:
        return {}

    # Determine the opponent for each game.  We use a heuristic:
    # the pitcher's team is the one that appears most often.
    team_counts = {}
    for _, row in series_info.iterrows():
        for t in [row["home_team"], row["away_team"]]:
            if pd.notna(t):
                team_counts[t] = team_counts.get(t, 0) + 1
    pitcher_team = max(team_counts, key=team_counts.get) if team_counts else None

    if pitcher_team is None:
        return {}

    # Assign opponent per game_pk
    game_opponent = {}
    for _, row in series_info.iterrows():
        ht = row.get("home_team")
        at = row.get("away_team")
        if ht == pitcher_team:
            game_opponent[row["game_pk"]] = at
        else:
            game_opponent[row["game_pk"]] = ht

    swings = swings.copy()
    swings["opponent"] = swings["game_pk"].map(game_opponent)

    # Assign series IDs: consecutive games against the same opponent
    games_sorted = series_info.sort_values("game_date").copy()
    games_sorted["opponent"] = games_sorted["game_pk"].map(game_opponent)
    games_sorted = games_sorted.dropna(subset=["opponent"])

    if games_sorted.empty:
        return {}

    # Assign series_id by detecting opponent changes
    series_id = 0
    series_ids = {}
    prev_opp = None
    for _, row in games_sorted.iterrows():
        opp = row["opponent"]
        if opp != prev_opp:
            series_id += 1
            prev_opp = opp
        series_ids[row["game_pk"]] = series_id

    swings["series_id"] = swings["game_pk"].map(series_ids)
    swings = swings.dropna(subset=["series_id"])

    if swings.empty:
        return {}

    # Within each series, number the games
    game_dates = swings.groupby(["series_id", "game_pk"])["game_date"].first().reset_index()
    game_dates = game_dates.sort_values(["series_id", "game_date"])
    game_dates["game_num"] = game_dates.groupby("series_id").cumcount() + 1
    game_num_map = dict(zip(game_dates["game_pk"], game_dates["game_num"]))

    swings["game_num_in_series"] = swings["game_pk"].map(game_num_map)

    # Unconditional whiff rate per pitch type
    uncond = swings.groupby("pitch_type").agg(
        total_swings=("is_swing", "sum"), total_whiffs=("is_whiff", "sum")
    )
    uncond["rate"] = uncond["total_whiffs"] / uncond["total_swings"]
    uncond_map = uncond["rate"].to_dict()

    result = {}
    for trans, grp in swings.groupby("transition"):
        parts = trans.split("->")
        if len(parts) != 2:
            continue
        curr_type = parts[1]
        base_rate = uncond_map.get(curr_type, 0.0)

        # Group by game number within series
        game_stats = (
            grp.groupby("game_num_in_series")
            .agg(n_swings=("is_swing", "sum"), n_whiffs=("is_whiff", "sum"))
            .reset_index()
        )
        game_stats["whiff_rate"] = game_stats["n_whiffs"] / game_stats["n_swings"]
        game_stats["alpha"] = game_stats["whiff_rate"] - base_rate

        alpha_by_game = [
            {
                "game_num": int(row["game_num_in_series"]),
                "alpha": round(float(row["alpha"]), 6),
                "whiff_rate": round(float(row["whiff_rate"]), 6),
                "n_swings": int(row["n_swings"]),
            }
            for _, row in game_stats.iterrows()
        ]

        fit_result = None
        if len(game_stats) >= MIN_REPETITIONS_FOR_FIT:
            reps_arr = game_stats["game_num_in_series"].values
            alphas_arr = game_stats["alpha"].values
            fit_result = _fit_exponential_decay(reps_arr, alphas_arr)
            if fit_result and fit_result["r_squared"] < MIN_R_SQUARED:
                fit_result = None

        result[trans] = {
            "fit": fit_result,
            "alpha_by_game": alpha_by_game,
        }

    return result


def _compute_regime_detection(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> dict:
    """Simple rolling-window regime detection for sequencing effectiveness.

    Computes a rolling alpha over ROLLING_WINDOW pitches and classifies
    the pitcher's regime as 'high_alpha' or 'low_alpha' based on whether
    the rolling value is above or below the population median.

    Returns:
        {
            "current_regime": "high_alpha" | "low_alpha" | None,
            "regime_timeseries": [{game_date, rolling_alpha, regime}, ...],
            "median_alpha": float,
        }
    """
    pairs = _get_pitch_pairs_with_whiff(conn, pitcher_id, season)

    if pairs.empty:
        return {"current_regime": None, "regime_timeseries": [], "median_alpha": 0.0}

    swings = pairs[pairs["is_swing"] == 1].copy()
    if len(swings) < ROLLING_WINDOW:
        return {"current_regime": None, "regime_timeseries": [], "median_alpha": 0.0}

    # Unconditional whiff rate
    overall_whiff_rate = swings["is_whiff"].mean()

    # Per-pitch alpha: is_whiff - overall_whiff_rate
    swings = swings.sort_values(["game_date", "at_bat_number", "pitch_number"])
    swings["alpha_raw"] = swings["is_whiff"].astype(float) - overall_whiff_rate

    # Rolling window
    swings["rolling_alpha"] = (
        swings["alpha_raw"]
        .rolling(window=ROLLING_WINDOW, min_periods=ROLLING_WINDOW // 2)
        .mean()
    )

    valid = swings.dropna(subset=["rolling_alpha"]).copy()
    if valid.empty:
        return {"current_regime": None, "regime_timeseries": [], "median_alpha": 0.0}

    median_alpha = float(valid["rolling_alpha"].median())

    valid["regime"] = np.where(
        valid["rolling_alpha"] >= median_alpha, "high_alpha", "low_alpha"
    )

    # Aggregate per game_date
    per_game = (
        valid.groupby("game_date")
        .agg(rolling_alpha=("rolling_alpha", "last"), regime=("regime", "last"))
        .reset_index()
    )

    timeseries = [
        {
            "game_date": str(row["game_date"]),
            "rolling_alpha": round(float(row["rolling_alpha"]), 6),
            "regime": row["regime"],
        }
        for _, row in per_game.iterrows()
    ]

    current_regime = timeseries[-1]["regime"] if timeseries else None

    return {
        "current_regime": current_regime,
        "regime_timeseries": timeseries,
        "median_alpha": round(median_alpha, 6),
    }


def calculate_alpha_decay(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> dict:
    """Full alpha decay profile for a pitcher.

    Combines sequence alpha computation, intra-game decay, series decay,
    and regime detection into a single comprehensive result.

    Returns:
        {
            "pitcher_id": int,
            "season": int | None,
            "transitions": {transition -> {alpha, intra_game, series}},
            "summary": {avg half-lives, most durable/fragile, adaptability_score},
            "regime": {current_regime, timeseries, median_alpha},
        }
    """
    alphas = compute_sequence_alpha(conn, pitcher_id, season)
    intra = fit_intra_game_decay(conn, pitcher_id, season)
    series = fit_series_decay(conn, pitcher_id, season)
    regime = _compute_regime_detection(conn, pitcher_id, season)

    # Merge into per-transition structure
    all_transitions = set(alphas.keys()) | set(intra.keys()) | set(series.keys())

    transitions = {}
    intra_half_lives = []
    series_half_lives = []

    for trans in sorted(all_transitions):
        alpha_info = alphas.get(trans, {})
        intra_info = intra.get(trans, {})
        series_info = series.get(trans, {})

        entry = {
            "alpha": alpha_info.get("alpha", 0.0),
            "conditional_whiff_rate": alpha_info.get("conditional_whiff_rate", 0.0),
            "unconditional_whiff_rate": alpha_info.get("unconditional_whiff_rate", 0.0),
            "swing_count": alpha_info.get("swing_count", 0),
            "intra_game": intra_info,
            "series": series_info,
        }
        transitions[trans] = entry

        # Collect half-lives for summary
        intra_fit = intra_info.get("fit")
        if intra_fit and intra_fit.get("r_squared", 0) >= MIN_R_SQUARED:
            intra_half_lives.append((trans, intra_fit["half_life"]))

        series_fit = series_info.get("fit")
        if series_fit and series_fit.get("r_squared", 0) >= MIN_R_SQUARED:
            series_half_lives.append((trans, series_fit["half_life"]))

    # Build summary
    avg_intra_hl = (
        round(float(np.mean([hl for _, hl in intra_half_lives])), 2)
        if intra_half_lives else None
    )
    avg_series_hl = (
        round(float(np.mean([hl for _, hl in series_half_lives])), 2)
        if series_half_lives else None
    )

    most_durable = max(intra_half_lives, key=lambda x: x[1])[0] if intra_half_lives else None
    most_fragile = min(intra_half_lives, key=lambda x: x[1])[0] if intra_half_lives else None

    # Adaptability score: longer half-lives = harder to decode = higher score
    # Scale to 0-100 by capping half-life at 20 reps
    if intra_half_lives:
        avg_hl = np.mean([hl for _, hl in intra_half_lives])
        adaptability_score = round(float(min(avg_hl / 20.0, 1.0) * 100), 1)
    else:
        adaptability_score = None

    summary = {
        "avg_intra_game_half_life": avg_intra_hl,
        "avg_series_half_life": avg_series_hl,
        "most_durable_transition": most_durable,
        "most_fragile_transition": most_fragile,
        "adaptability_score": adaptability_score,
        "n_transitions_fitted_intra": len(intra_half_lives),
        "n_transitions_fitted_series": len(series_half_lives),
    }

    return {
        "pitcher_id": pitcher_id,
        "season": season,
        "transitions": transitions,
        "summary": summary,
        "regime": regime,
    }


def batch_calculate(
    conn: duckdb.DuckDBPyConnection,
    season: Optional[int] = None,
    min_pitches: int = MIN_PITCHES_DEFAULT,
) -> pd.DataFrame:
    """Compute alpha decay leaderboard for all qualifying pitchers.

    Args:
        conn: Open DuckDB connection.
        season: Optional season filter.
        min_pitches: Minimum total pitches to qualify.

    Returns:
        DataFrame with columns: pitcher_id, avg_intra_game_half_life,
        avg_series_half_life, adaptability_score, n_transitions_fitted,
        sorted by adaptability_score descending (hardest to decode first).
    """
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $1" if season else ""
    params: list = [season] if season else []

    query = f"""
        SELECT pitcher_id, COUNT(*) AS pitch_count
        FROM pitches
        WHERE pitch_type IS NOT NULL
          {season_filter}
        GROUP BY pitcher_id
        HAVING COUNT(*) >= {int(min_pitches)}
    """
    qualifying = conn.execute(query, params).fetchdf()

    columns = [
        "pitcher_id",
        "avg_intra_game_half_life",
        "avg_series_half_life",
        "adaptability_score",
        "n_transitions_fitted",
    ]

    if qualifying.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for _, row in qualifying.iterrows():
        pid = int(row["pitcher_id"])
        try:
            result = calculate_alpha_decay(conn, pid, season=season)
            summary = result["summary"]
            rows.append({
                "pitcher_id": pid,
                "avg_intra_game_half_life": summary.get("avg_intra_game_half_life"),
                "avg_series_half_life": summary.get("avg_series_half_life"),
                "adaptability_score": summary.get("adaptability_score"),
                "n_transitions_fitted": summary.get("n_transitions_fitted_intra", 0),
            })
        except Exception as exc:
            logger.warning("Alpha decay failed for pitcher %d: %s", pid, exc)

    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    df = df.sort_values("adaptability_score", ascending=False, na_position="last")
    df = df.reset_index(drop=True)
    return df


def get_fastest_decaying_sequences(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
    top_n: int = 5,
) -> list[dict]:
    """Return the top N transitions with the shortest intra-game half-lives.

    These are the sequences that batters adapt to fastest.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        season: Optional season filter.
        top_n: Number of results.

    Returns:
        List of dicts with keys: transition, half_life, alpha_0, r_squared.
    """
    intra = fit_intra_game_decay(conn, pitcher_id, season)

    fitted = []
    for trans, data in intra.items():
        fit = data.get("fit")
        if fit and fit.get("r_squared", 0) >= MIN_R_SQUARED:
            fitted.append({
                "transition": trans,
                "half_life": fit["half_life"],
                "alpha_0": fit["alpha_0"],
                "r_squared": fit["r_squared"],
            })

    fitted.sort(key=lambda x: x["half_life"])
    return fitted[:top_n]


# ── Model class ──────────────────────────────────────────────────────────────


class AlphaDecayModel(BaseAnalyticsModel):
    """Pitch Sequence Alpha Decay model.

    Measures how quickly a pitcher's pitch-pair transitions lose their
    deceptive effectiveness through within-game and cross-series
    repetition.
    """

    @property
    def model_name(self) -> str:
        return "AlphaDecay"

    @property
    def version(self) -> str:
        return "1.0.0"

    def train(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """No training step required -- alpha decay is a descriptive metric."""
        metrics = {"status": "no_training_needed"}
        self.set_training_metadata(metrics)
        return metrics

    def predict(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        pitcher_id: int,
        season: Optional[int] = None,
    ) -> dict:
        """Compute full alpha decay profile.  Delegates to ``calculate_alpha_decay``."""
        return calculate_alpha_decay(conn, pitcher_id, season=season)

    def evaluate(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Evaluate model coverage: how many pitchers qualify."""
        season = kwargs.get("season")
        min_pitches = kwargs.get("min_pitches", MIN_PITCHES_DEFAULT)
        df = batch_calculate(conn, season=season, min_pitches=min_pitches)
        fitted = df[df["n_transitions_fitted"] > 0]
        return {
            "qualifying_pitchers": len(df),
            "pitchers_with_fits": len(fitted),
            "mean_adaptability_score": round(
                float(fitted["adaptability_score"].mean()), 2
            ) if not fitted.empty else 0.0,
        }
