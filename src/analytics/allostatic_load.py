"""
Allostatic Batting Load (ABL) -- Multi-channel cumulative decision fatigue for hitters.

Models five independent fatigue channels, each as a leaky integrator that
accumulates game-level stressors and decays over rest days:

    1. Pitch Processing Load   -- pitches seen per game
    2. Decision Conflict Load  -- borderline pitches faced (near zone edge)
    3. Swing Exertion           -- total swings per game
    4. Temporal Demand          -- schedule density (games in trailing 7 days)
    5. Travel Stress            -- home/away switches and consecutive road games

The composite ABL is the sum of z-scored channel loads that exceed the
population mean, scaled to 0-100.  Validated against O-Swing% (chase rate)
and Z-Contact% (in-zone contact rate).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd

from src.analytics.base import BaseAnalyticsModel
from src.analytics.features import (
    compute_z_scores,
    get_player_name,
    get_latest_season,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "allostatic_load"

# Minimum number of games for meaningful load accumulation
_MIN_GAMES = 20

# Default leaky-integrator decay rates
_ALPHA_DEFAULT = 0.85
_ALPHA_TRAVEL = 0.70

# Swing descriptions in Statcast
_SWING_DESCRIPTIONS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
    "missed_bunt",
}

# Zone edge constants (feet)
_ZONE_X_INNER = 0.58
_ZONE_X_OUTER = 1.08


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ABLConfig:
    """Hyperparameters for the Allostatic Batting Load model."""

    alpha_pitch_processing: float = _ALPHA_DEFAULT
    alpha_decision_conflict: float = _ALPHA_DEFAULT
    alpha_swing_exertion: float = _ALPHA_DEFAULT
    alpha_temporal_demand: float = _ALPHA_DEFAULT
    alpha_travel: float = _ALPHA_TRAVEL

    # On off-days the decay is alpha squared (faster recovery)
    off_day_uses_alpha_squared: bool = True

    # Minimum games for calculation
    min_games: int = _MIN_GAMES

    # ABL scale (output is 0 to abl_max)
    abl_max: float = 100.0

    # Zone edge for borderline pitches
    zone_x_inner: float = _ZONE_X_INNER
    zone_x_outer: float = _ZONE_X_OUTER

    # Trailing window (days) for temporal demand
    temporal_window_days: int = 7


# ---------------------------------------------------------------------------
# Leaky integrator
# ---------------------------------------------------------------------------

def compute_leaky_load(
    stressors: pd.Series,
    alpha: float = _ALPHA_DEFAULT,
    off_day_alpha: float | None = None,
    off_day_mask: pd.Series | None = None,
) -> pd.Series:
    """Apply a leaky integrator to a time-ordered stressor series.

    L(d) = alpha * L(d-1) + stressor(d)

    On off-days (where ``off_day_mask`` is True), the decay uses
    ``off_day_alpha`` (typically alpha**2) and the stressor is 0.

    Args:
        stressors: Per-game stressor values, ordered chronologically.
        alpha: Retention factor (0 < alpha < 1).
        off_day_alpha: Decay on off-days.  Defaults to ``alpha ** 2``.
        off_day_mask: Boolean Series aligned with *stressors*;
                      True for off-day entries.  If None, no off-days.

    Returns:
        Cumulative load series of the same length.
    """
    if off_day_alpha is None:
        off_day_alpha = alpha ** 2

    values = stressors.values.astype(np.float64)
    n = len(values)
    load = np.zeros(n, dtype=np.float64)

    if off_day_mask is not None:
        off_days = off_day_mask.values.astype(bool)
    else:
        off_days = np.zeros(n, dtype=bool)

    for i in range(n):
        prev = load[i - 1] if i > 0 else 0.0
        if off_days[i]:
            # Off-day: faster decay, stressor is 0
            load[i] = off_day_alpha * prev
        else:
            load[i] = alpha * prev + values[i]

    return pd.Series(load, index=stressors.index, name=stressors.name)


# ---------------------------------------------------------------------------
# Stressor computation from database
# ---------------------------------------------------------------------------

def compute_game_stressors(
    conn: duckdb.DuckDBPyConnection,
    batter_id: int,
    season: int,
) -> pd.DataFrame:
    """Compute per-game stressor values for each fatigue channel.

    Args:
        conn: Open DuckDB connection.
        batter_id: MLB player ID.
        season: Season year.

    Returns:
        DataFrame indexed by game_date with columns:
        ``game_pk``, ``pitch_processing``, ``decision_conflict``,
        ``swing_exertion``, ``temporal_demand``, ``travel_stress``,
        ``is_off_day``.
    """
    # ---- Per-game pitch-level aggregation --------------------------------
    pitch_query = """
        SELECT
            p.game_pk,
            p.game_date,
            COUNT(*) AS pitches_seen,
            SUM(CASE
                WHEN (ABS(p.plate_x) BETWEEN $3 AND $4)
                  OR (p.plate_z IS NOT NULL
                      AND p.plate_z BETWEEN 1.0 AND 1.5)
                  OR (p.plate_z IS NOT NULL
                      AND p.plate_z BETWEEN 3.2 AND 3.7)
                THEN 1 ELSE 0
            END) AS borderline_pitches,
            SUM(CASE
                WHEN p.description IN (
                    'swinging_strike', 'swinging_strike_blocked',
                    'foul', 'foul_tip', 'foul_bunt',
                    'hit_into_play', 'hit_into_play_no_out',
                    'hit_into_play_score', 'missed_bunt'
                )
                THEN 1 ELSE 0
            END) AS swings
        FROM pitches p
        WHERE p.batter_id = $1
          AND EXTRACT(YEAR FROM p.game_date) = $2
        GROUP BY p.game_pk, p.game_date
        ORDER BY p.game_date
    """
    game_df = conn.execute(
        pitch_query,
        [batter_id, season, _ZONE_X_INNER, _ZONE_X_OUTER],
    ).fetchdf()

    if game_df.empty:
        return pd.DataFrame(columns=[
            "game_pk", "game_date",
            "pitch_processing", "decision_conflict", "swing_exertion",
            "temporal_demand", "travel_stress", "is_off_day",
        ])

    game_df["game_date"] = pd.to_datetime(game_df["game_date"])
    game_df = game_df.sort_values("game_date").reset_index(drop=True)

    # ---- Channel 1: Pitch Processing Load --------------------------------
    game_df["pitch_processing"] = game_df["pitches_seen"].astype(float)

    # ---- Channel 2: Decision Conflict Load -------------------------------
    game_df["decision_conflict"] = game_df["borderline_pitches"].astype(float)

    # ---- Channel 3: Swing Exertion ---------------------------------------
    game_df["swing_exertion"] = game_df["swings"].astype(float)

    # ---- Channel 4: Temporal Demand (games in trailing 7 days) -----------
    dates = game_df["game_date"].values
    temporal = np.zeros(len(game_df), dtype=np.float64)
    for i, d in enumerate(dates):
        window_start = d - np.timedelta64(7, "D")
        temporal[i] = float(np.sum((dates >= window_start) & (dates <= d)))
    game_df["temporal_demand"] = temporal

    # ---- Channel 5: Travel Stress ----------------------------------------
    # Determine home/away per game by joining games table and players table
    # to check if the batter's team matches the home team.
    try:
        team_result = conn.execute(
            "SELECT team FROM players WHERE player_id = $1",
            [batter_id],
        ).fetchone()
        batter_team = team_result[0] if team_result else None
    except Exception:
        batter_team = None

    if batter_team is not None:
        game_pks = game_df["game_pk"].tolist()
        if game_pks:
            gp_str = ", ".join(str(int(g)) for g in game_pks)
            home_away_df = conn.execute(f"""
                SELECT game_pk, home_team
                FROM games
                WHERE game_pk IN ({gp_str})
            """).fetchdf()
            home_map = dict(zip(home_away_df["game_pk"], home_away_df["home_team"]))
            game_df["is_home"] = game_df["game_pk"].map(
                lambda gp: home_map.get(gp) == batter_team
            )
        else:
            game_df["is_home"] = True
    else:
        # Fallback: assume home (travel stress will be 0)
        game_df["is_home"] = True

    travel = np.zeros(len(game_df), dtype=np.float64)
    for i in range(len(game_df)):
        if i == 0:
            travel[i] = 0.0 if game_df["is_home"].iloc[i] else 1.0
        else:
            # Count consecutive away games ending at this game
            if not game_df["is_home"].iloc[i]:
                # Away game: add 1 for being away, plus bonus for switch
                consec_away = 0
                for j in range(i, -1, -1):
                    if not game_df["is_home"].iloc[j]:
                        consec_away += 1
                    else:
                        break
                travel[i] = float(consec_away)
                # Add extra if there was a home/away switch
                if game_df["is_home"].iloc[i - 1] != game_df["is_home"].iloc[i]:
                    travel[i] += 1.0
            else:
                # Home game
                if game_df["is_home"].iloc[i - 1] != game_df["is_home"].iloc[i]:
                    travel[i] = 1.0  # just switched from away to home
                else:
                    travel[i] = 0.0
    game_df["travel_stress"] = travel

    # ---- Off-day detection -----------------------------------------------
    date_diffs = game_df["game_date"].diff()
    game_df["is_off_day"] = False  # game days are never off-days
    # We need to insert off-days into the timeline for the leaky integrator.
    # Instead, we mark whether there was a gap BEFORE this game, so the
    # integrator can decay extra for the gap.
    game_df["days_since_last"] = date_diffs.dt.days.fillna(1).astype(int)

    return game_df[[
        "game_pk", "game_date",
        "pitch_processing", "decision_conflict", "swing_exertion",
        "temporal_demand", "travel_stress",
        "days_since_last", "is_home",
    ]]


def _apply_channel_load(
    stressor_series: pd.Series,
    days_since_last: pd.Series,
    alpha: float,
) -> pd.Series:
    """Apply the leaky integrator with multi-day gap handling.

    When there is a gap of N days between games, we apply
    alpha^(2*gap_days) for the decay (off-days accelerate recovery).

    Args:
        stressor_series: Per-game stressor values.
        days_since_last: Days since previous game (1 = consecutive).
        alpha: Retention factor.

    Returns:
        Cumulative load series.
    """
    values = stressor_series.values.astype(np.float64)
    gaps = days_since_last.values.astype(np.float64)
    n = len(values)
    load = np.zeros(n, dtype=np.float64)

    off_day_alpha = alpha ** 2

    for i in range(n):
        prev = load[i - 1] if i > 0 else 0.0
        gap = int(gaps[i]) if i > 0 else 1

        if gap <= 1:
            # Normal consecutive game
            load[i] = alpha * prev + values[i]
        else:
            # Off-days: apply accelerated decay for each off-day
            # Then normal decay + new stressor for the game day
            decayed = prev
            for _ in range(gap - 1):
                decayed = off_day_alpha * decayed
            load[i] = alpha * decayed + values[i]

    return pd.Series(load, index=stressor_series.index)


# ---------------------------------------------------------------------------
# Core ABL calculation
# ---------------------------------------------------------------------------

def calculate_abl(
    conn: duckdb.DuckDBPyConnection,
    batter_id: int,
    season: int,
    config: ABLConfig | None = None,
) -> dict:
    """Calculate Allostatic Batting Load for a single batter-season.

    Args:
        conn: Open DuckDB connection.
        batter_id: MLB player ID.
        season: Season year.
        config: Optional configuration override.

    Returns:
        Dictionary with keys:
        - ``batter_id``, ``season``
        - ``composite_abl``: Current (final game) composite load (0-100)
        - ``channel_loads``: Dict of final load per channel
        - ``peak_abl``: Maximum composite ABL during the season
        - ``peak_date``: Date of peak ABL
        - ``timeline``: DataFrame of per-game channel loads and composite
        - ``games_played``: Number of games
    """
    if config is None:
        config = ABLConfig()

    stressors = compute_game_stressors(conn, batter_id, season)

    if stressors.empty or len(stressors) < config.min_games:
        return {
            "batter_id": batter_id,
            "season": season,
            "composite_abl": None,
            "channel_loads": {},
            "peak_abl": None,
            "peak_date": None,
            "timeline": pd.DataFrame(),
            "games_played": len(stressors),
        }

    days_since = stressors["days_since_last"]

    # ---- Compute channel loads -------------------------------------------
    channels = {
        "pitch_processing": config.alpha_pitch_processing,
        "decision_conflict": config.alpha_decision_conflict,
        "swing_exertion": config.alpha_swing_exertion,
        "temporal_demand": config.alpha_temporal_demand,
        "travel_stress": config.alpha_travel,
    }

    load_df = stressors[["game_pk", "game_date"]].copy()

    for channel, alpha in channels.items():
        load_df[channel] = _apply_channel_load(
            stressors[channel], days_since, alpha
        )

    # ---- Z-score each channel across the season --------------------------
    channel_names = list(channels.keys())
    load_df = compute_z_scores(load_df, channel_names)

    # ---- Composite ABL: sum of z-scores that are above 0 (mean) ---------
    z_cols = [f"{ch}_z" for ch in channel_names]

    def _composite_row(row):
        z_values = [row[c] for c in z_cols]
        # Sum only z-scores above 0 (above population mean)
        above_mean = [z for z in z_values if z > 0]
        return sum(above_mean)

    load_df["composite_raw"] = load_df.apply(_composite_row, axis=1)

    # Scale to 0-100
    raw_max = load_df["composite_raw"].max()
    if raw_max > 0:
        load_df["composite_abl"] = (
            load_df["composite_raw"] / raw_max * config.abl_max
        )
    else:
        load_df["composite_abl"] = 0.0

    # Clamp
    load_df["composite_abl"] = load_df["composite_abl"].clip(0, config.abl_max)

    # ---- Extract results -------------------------------------------------
    final_row = load_df.iloc[-1]
    peak_idx = load_df["composite_abl"].idxmax()
    peak_row = load_df.loc[peak_idx]

    channel_loads = {ch: round(float(final_row[ch]), 3) for ch in channel_names}

    return {
        "batter_id": batter_id,
        "season": season,
        "composite_abl": round(float(final_row["composite_abl"]), 2),
        "channel_loads": channel_loads,
        "peak_abl": round(float(peak_row["composite_abl"]), 2),
        "peak_date": str(peak_row["game_date"].date())
        if hasattr(peak_row["game_date"], "date")
        else str(peak_row["game_date"]),
        "timeline": load_df,
        "games_played": len(load_df),
    }


# ---------------------------------------------------------------------------
# Batch calculation
# ---------------------------------------------------------------------------

def batch_calculate(
    conn: duckdb.DuckDBPyConnection,
    season: int,
    min_games: int = _MIN_GAMES,
    config: ABLConfig | None = None,
) -> pd.DataFrame:
    """Calculate ABL for all qualifying batters in a season.

    Args:
        conn: Open DuckDB connection.
        season: Season year.
        min_games: Minimum games played to include.
        config: Optional configuration override.

    Returns:
        DataFrame with columns: batter_id, name, season, composite_abl,
        peak_abl, peak_date, games_played, plus per-channel final loads.
    """
    if config is None:
        config = ABLConfig()
    config.min_games = min_games

    # Get all batters with enough games
    batter_query = """
        SELECT batter_id, COUNT(DISTINCT game_pk) AS n_games
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) = $1
        GROUP BY batter_id
        HAVING COUNT(DISTINCT game_pk) >= $2
        ORDER BY n_games DESC
    """
    batters = conn.execute(batter_query, [season, min_games]).fetchdf()

    if batters.empty:
        return pd.DataFrame(columns=[
            "batter_id", "name", "season", "composite_abl",
            "peak_abl", "peak_date", "games_played",
            "pitch_processing", "decision_conflict", "swing_exertion",
            "temporal_demand", "travel_stress",
        ])

    rows = []
    for _, batter_row in batters.iterrows():
        bid = int(batter_row["batter_id"])
        result = calculate_abl(conn, bid, season, config)

        if result["composite_abl"] is None:
            continue

        name = _get_player_name(conn, bid)
        rows.append({
            "batter_id": bid,
            "name": name,
            "season": season,
            "composite_abl": result["composite_abl"],
            "peak_abl": result["peak_abl"],
            "peak_date": result["peak_date"],
            "games_played": result["games_played"],
            **result["channel_loads"],
        })

    if not rows:
        return pd.DataFrame(columns=[
            "batter_id", "name", "season", "composite_abl",
            "peak_abl", "peak_date", "games_played",
            "pitch_processing", "decision_conflict", "swing_exertion",
            "temporal_demand", "travel_stress",
        ])

    df = pd.DataFrame(rows)
    df = df.sort_values("composite_abl", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Validation against outcomes
# ---------------------------------------------------------------------------

def validate_against_outcomes(
    conn: duckdb.DuckDBPyConnection,
    batter_id: int,
    season: int,
    config: ABLConfig | None = None,
) -> dict:
    """Validate ABL by correlating with O-Swing% and Z-Contact%.

    Args:
        conn: Open DuckDB connection.
        batter_id: MLB player ID.
        season: Season year.
        config: Optional configuration override.

    Returns:
        Dictionary with correlation statistics:
        - ``chase_rate_corr``: Correlation of ABL with O-Swing%
        - ``zone_contact_corr``: Correlation of ABL with Z-Contact%
        - ``n_games``: Number of games used
        - ``p_values``: p-values (if scipy available)
    """
    result = calculate_abl(conn, batter_id, season, config)

    if result["composite_abl"] is None or result["timeline"].empty:
        return {
            "batter_id": batter_id,
            "season": season,
            "chase_rate_corr": None,
            "zone_contact_corr": None,
            "n_games": 0,
        }

    timeline = result["timeline"]
    game_pks = timeline["game_pk"].tolist()

    if not game_pks:
        return {
            "batter_id": batter_id,
            "season": season,
            "chase_rate_corr": None,
            "zone_contact_corr": None,
            "n_games": 0,
        }

    # Compute per-game O-Swing% and Z-Contact% from pitch-level data
    gp_str = ", ".join(str(int(g)) for g in game_pks)
    outcome_query = f"""
        SELECT
            game_pk,
            -- O-Swing%: swings at pitches outside zone / pitches outside zone
            SUM(CASE
                WHEN zone IS NOT NULL AND zone > 9
                 AND description IN (
                    'swinging_strike', 'swinging_strike_blocked',
                    'foul', 'foul_tip', 'hit_into_play',
                    'hit_into_play_no_out', 'hit_into_play_score',
                    'missed_bunt', 'foul_bunt'
                )
                THEN 1 ELSE 0
            END) AS o_swings,
            SUM(CASE WHEN zone IS NOT NULL AND zone > 9
                THEN 1 ELSE 0
            END) AS o_pitches,
            -- Z-Contact%: contact on pitches in zone / swings at pitches in zone
            SUM(CASE
                WHEN zone IS NOT NULL AND zone <= 9
                 AND description IN (
                    'foul', 'foul_bunt',
                    'hit_into_play', 'hit_into_play_no_out',
                    'hit_into_play_score'
                )
                THEN 1 ELSE 0
            END) AS z_contact,
            SUM(CASE
                WHEN zone IS NOT NULL AND zone <= 9
                 AND description IN (
                    'swinging_strike', 'swinging_strike_blocked',
                    'foul', 'foul_tip', 'hit_into_play',
                    'hit_into_play_no_out', 'hit_into_play_score',
                    'missed_bunt', 'foul_bunt'
                )
                THEN 1 ELSE 0
            END) AS z_swings
        FROM pitches
        WHERE batter_id = $1
          AND game_pk IN ({gp_str})
        GROUP BY game_pk
    """
    outcomes = conn.execute(outcome_query, [batter_id]).fetchdf()

    if outcomes.empty:
        return {
            "batter_id": batter_id,
            "season": season,
            "chase_rate_corr": None,
            "zone_contact_corr": None,
            "n_games": 0,
        }

    # Compute rates
    outcomes["chase_rate"] = np.where(
        outcomes["o_pitches"] > 0,
        outcomes["o_swings"] / outcomes["o_pitches"],
        np.nan,
    )
    outcomes["zone_contact_rate"] = np.where(
        outcomes["z_swings"] > 0,
        outcomes["z_contact"] / outcomes["z_swings"],
        np.nan,
    )

    # Merge with timeline
    merged = timeline.merge(outcomes[["game_pk", "chase_rate", "zone_contact_rate"]],
                            on="game_pk", how="inner")

    valid_chase = merged.dropna(subset=["composite_abl", "chase_rate"])
    valid_contact = merged.dropna(subset=["composite_abl", "zone_contact_rate"])

    chase_corr = None
    zone_contact_corr = None

    if len(valid_chase) >= 10:
        chase_corr = round(float(
            valid_chase["composite_abl"].corr(valid_chase["chase_rate"])
        ), 4)

    if len(valid_contact) >= 10:
        zone_contact_corr = round(float(
            valid_contact["composite_abl"].corr(valid_contact["zone_contact_rate"])
        ), 4)

    return {
        "batter_id": batter_id,
        "season": season,
        "chase_rate_corr": chase_corr,
        "zone_contact_corr": zone_contact_corr,
        "n_games": len(merged),
    }


# ---------------------------------------------------------------------------
# Model class (BaseAnalyticsModel)
# ---------------------------------------------------------------------------

class AllostaticLoadModel(BaseAnalyticsModel):
    """Allostatic Batting Load model.

    Inherits from ``BaseAnalyticsModel`` for consistent lifecycle management.
    """

    def __init__(self, config: ABLConfig | None = None) -> None:
        super().__init__()
        self.config = config or ABLConfig()
        self._results: dict[int, dict] = {}  # batter_id -> result dict
        self._is_trained = False
        self._training_season: int | None = None

    @property
    def model_name(self) -> str:
        return "allostatic_batting_load"

    @property
    def version(self) -> str:
        return "1.0.0"

    def train(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Train the ABL model (batch compute for all qualifying batters).

        Args:
            conn: Open DuckDB connection.
            season: Season year (keyword argument).
            min_games: Minimum games played (keyword argument).

        Returns:
            Dictionary of training metrics.
        """
        season = kwargs.get("season")
        if season is None:
            season = _get_latest_season(conn)
        min_games = kwargs.get("min_games", self.config.min_games)

        logger.info("Training ABL model for season %d (min_games=%d)", season, min_games)

        df = batch_calculate(conn, season, min_games, self.config)
        logger.info("Computed ABL for %d batters", len(df))

        # Store per-batter results
        self._results = {}
        for _, row in df.iterrows():
            bid = int(row["batter_id"])
            self._results[bid] = row.to_dict()

        self._is_trained = True
        self._training_season = season

        # Metrics
        if not df.empty:
            abl_vals = df["composite_abl"].dropna()
            metrics = {
                "n_batters": len(df),
                "mean_abl": round(float(abl_vals.mean()), 2),
                "std_abl": round(float(abl_vals.std()), 2),
                "median_abl": round(float(abl_vals.median()), 2),
                "max_abl": round(float(abl_vals.max()), 2),
                "season": season,
            }
        else:
            metrics = {"n_batters": 0, "season": season}

        self.set_training_metadata(metrics=metrics, params={"season": season, "min_games": min_games})
        return metrics

    def predict(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict | pd.DataFrame:
        """Return ABL for a specific batter or all batters.

        Args:
            conn: Open DuckDB connection.
            batter_id: Specific batter (optional).
            season: Season year (optional).

        Returns:
            Dict for single batter or DataFrame for all batters.
        """
        batter_id = kwargs.get("batter_id")
        season = kwargs.get("season", self._training_season)

        if batter_id is not None:
            return calculate_abl(conn, batter_id, season, self.config)

        if self._is_trained and self._results:
            return pd.DataFrame(list(self._results.values()))

        return batch_calculate(conn, season, self.config.min_games, self.config)

    def evaluate(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Evaluate model by validating against O-Swing% and Z-Contact%.

        Returns:
            Dictionary with evaluation metrics.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        season = self._training_season
        batter_ids = list(self._results.keys())[:20]  # sample for speed

        correlations = []
        for bid in batter_ids:
            v = validate_against_outcomes(conn, bid, season, self.config)
            if v["chase_rate_corr"] is not None:
                correlations.append(v)

        if not correlations:
            return self.validate_output({
                "n_evaluated": 0,
                "status": "no_validation_data",
            })

        chase_corrs = [c["chase_rate_corr"] for c in correlations if c["chase_rate_corr"] is not None]
        contact_corrs = [c["zone_contact_corr"] for c in correlations if c["zone_contact_corr"] is not None]

        return self.validate_output({
            "n_evaluated": len(correlations),
            "mean_chase_rate_corr": round(float(np.mean(chase_corrs)), 4) if chase_corrs else None,
            "mean_zone_contact_corr": round(float(np.mean(contact_corrs)), 4) if contact_corrs else None,
        })


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _get_latest_season(conn: duckdb.DuckDBPyConnection) -> int:
    """Return the most recent season year in the pitches table."""
    return get_latest_season(conn)


def _get_player_name(conn: duckdb.DuckDBPyConnection, player_id: int) -> str | None:
    """Look up a player's full name from the players table."""
    return get_player_name(conn, player_id)


def predict_recovery_days(
    current_abl: float,
    threshold: float,
    alpha: float = _ALPHA_DEFAULT,
) -> int:
    """Predict days until ABL drops below a threshold (no new stressor).

    Assumes pure rest (off-day decay: alpha^2 per day).

    Args:
        current_abl: Current composite ABL value.
        threshold: Target ABL level.
        alpha: Decay rate.

    Returns:
        Number of rest days needed, or 0 if already below threshold.
    """
    if current_abl <= threshold:
        return 0

    off_day_alpha = alpha ** 2
    days = 0
    load = current_abl
    while load > threshold and days < 365:
        load *= off_day_alpha
        days += 1
    return days
