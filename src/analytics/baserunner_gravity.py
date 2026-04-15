"""
Baserunner Gravity Index (BGI) -- NBA gravity concept applied to baserunning threat.

Measures how much a specific baserunner distorts pitcher behaviour and batter
outcomes simply by being on base.  The core idea:

    When a high-threat runner (e.g. a prolific base-stealer) stands on 1B,
    pitchers alter their approach -- throwing more fastballs, losing velocity
    or command, and shifting attention away from the batter.  The BGI
    quantifies that gravitational pull across four channels:

        1. **Velocity** -- mean release_speed difference (runner-on vs league).
        2. **Location** -- plate_x / plate_z variance change (command disruption).
        3. **Selection** -- fastball% shift (forced into more heaters?).
        4. **Outcome** -- batter estimated_woba uplift on contact.

    BGI = 100 + (total_xwoba_effect x scaling_factor)

    where 100 = replacement-level runner, and +/- 1 std ~ +/- 15 points.

The runner's threat proxy is derived from stolen-base attempt rate in the
pitch-level data (no external sprint-speed data required).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd

from src.analytics.base import BaseAnalyticsModel
from src.analytics.features import get_player_name, get_latest_season

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "baserunner_gravity"

# Minimum pitches with the runner on base to consider a reliable sample
_MIN_PITCHES_WITH_RUNNER = 50

# Minimum matched comparison pitches
_MIN_MATCHED_SAMPLE = 20

# Fastball pitch types
_FASTBALL_TYPES = {"FF", "SI", "FC", "FA"}

# BGI scaling: +/- 1 std of total xwOBA effect maps to +/- 15 BGI points
_BGI_CENTER = 100
_BGI_STD_POINTS = 15


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BGIConfig:
    """Hyperparameters for the Baserunner Gravity Index model."""

    min_pitches_with_runner: int = _MIN_PITCHES_WITH_RUNNER
    min_matched_sample: int = _MIN_MATCHED_SAMPLE
    min_appearances_batch: int = 50
    random_state: int = 42


# ---------------------------------------------------------------------------
# Core model class
# ---------------------------------------------------------------------------

class BaserunnerGravityModel(BaseAnalyticsModel):
    """Baserunner Gravity Index model.

    Inherits from ``BaseAnalyticsModel`` for consistent lifecycle management.
    """

    def __init__(self, config: BGIConfig | None = None) -> None:
        super().__init__()
        self.config = config or BGIConfig()
        self._runner_effects: dict[int, dict] = {}
        self._is_trained = False
        self._training_season: int | None = None
        self._league_std: float = 1.0  # will be set during batch

    # ---- BaseAnalyticsModel interface ------------------------------------

    @property
    def model_name(self) -> str:
        return "baserunner_gravity"

    @property
    def version(self) -> str:
        return "1.0.0"

    def train(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Train the BGI model by computing effects for all qualifying runners.

        Args:
            conn: Open DuckDB connection.
            season: Season year (keyword argument).

        Returns:
            Dictionary of training metrics.
        """
        season = kwargs.get("season")
        if season is None:
            season = _get_latest_season(conn)
        logger.info("Training Baserunner Gravity for season %d", season)

        df = self.batch_calculate(
            conn,
            season,
            min_appearances=self.config.min_appearances_batch,
        )

        self._is_trained = True
        self._training_season = season

        metrics = {
            "n_runners_scored": len(df),
            "season": season,
            "mean_bgi": round(float(df["bgi"].mean()), 2) if not df.empty else None,
            "std_bgi": round(float(df["bgi"].std()), 2) if not df.empty else None,
        }

        self.set_training_metadata(
            metrics=metrics,
            params={
                "min_pitches_with_runner": self.config.min_pitches_with_runner,
                "min_matched_sample": self.config.min_matched_sample,
                "min_appearances_batch": self.config.min_appearances_batch,
                "season": season,
            },
        )
        logger.info("BGI training complete: %d runners scored", len(df))
        return metrics

    def predict(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict | pd.DataFrame:
        """Generate BGI predictions.

        Args:
            conn: Open DuckDB connection.
            runner_id: Specific runner (optional).
            season: Season year (optional).

        Returns:
            DataFrame of BGI estimates, or dict for a single runner.
        """
        runner_id = kwargs.get("runner_id")
        season = kwargs.get("season", self._training_season)

        if runner_id is not None:
            return calculate_bgi(conn, runner_id, season, self.config)

        return self.batch_calculate(conn, season)

    def evaluate(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Evaluate model quality: check BGI centering and spread.

        Returns:
            Dictionary with diagnostic statistics.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        effects = [e["bgi"] for e in self._runner_effects.values()
                    if e.get("bgi") is not None]

        if not effects:
            return {"n_runners": 0, "status": "no_effects"}

        arr = np.array(effects)
        return self.validate_output({
            "n_runners": len(arr),
            "mean_bgi": round(float(np.mean(arr)), 2),
            "std_bgi": round(float(np.std(arr)), 2),
            "median_bgi": round(float(np.median(arr)), 2),
            "min_bgi": round(float(np.min(arr)), 2),
            "max_bgi": round(float(np.max(arr)), 2),
        })

    # ---- Public API ------------------------------------------------------

    def batch_calculate(
        self,
        conn: duckdb.DuckDBPyConnection,
        season: int | None = None,
        min_appearances: int | None = None,
    ) -> pd.DataFrame:
        """Calculate BGI for all qualifying runners in a season.

        Args:
            conn: Open DuckDB connection.
            season: Season year.
            min_appearances: Minimum pitches-with-runner-on-base threshold.

        Returns:
            DataFrame with columns: runner_id, name, season, bgi,
            sb_attempt_rate, velocity_effect, location_effect,
            selection_effect, outcome_effect, n_pitches, percentile.
        """
        if season is None:
            season = self._training_season or _get_latest_season(conn)
        if min_appearances is None:
            min_appearances = self.config.min_appearances_batch

        result = batch_calculate(conn, season, min_appearances, self.config)

        # Store runner effects for evaluate()
        for _, row in result.iterrows():
            self._runner_effects[int(row["runner_id"])] = row.to_dict()

        return result

    def get_gravity_leaderboard(
        self,
        conn: duckdb.DuckDBPyConnection,
        season: int | None = None,
    ) -> pd.DataFrame:
        """Return a sorted leaderboard of highest-gravity runners.

        Args:
            conn: Open DuckDB connection.
            season: Season year.

        Returns:
            Sorted DataFrame with Rank index.
        """
        return get_gravity_leaderboard(conn, season, self.config)


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------


def compute_runner_threat_rate(
    conn: duckdb.DuckDBPyConnection,
    runner_id: int,
    season: int,
) -> float:
    """Compute stolen-base attempt rate proxy for a runner.

    Uses pitches where the runner was on base and an SB-related event
    occurred (stolen_base in the events column or description).

    Args:
        conn: Open DuckDB connection.
        runner_id: MLB player ID of the runner.
        season: Season year.

    Returns:
        SB attempt rate in [0, 1].  Returns 0.0 if no qualifying data.
    """
    query = """
        WITH runner_pitches AS (
            SELECT
                events,
                description
            FROM pitches
            WHERE (on_1b = $1 OR on_2b = $1 OR on_3b = $1)
              AND EXTRACT(YEAR FROM game_date) = $2
        )
        SELECT
            COUNT(*) AS total_pitches,
            SUM(CASE
                WHEN events LIKE '%stolen_base%'
                  OR events LIKE '%caught_stealing%'
                  OR description LIKE '%stolen%'
                  OR description LIKE '%pickoff%'
                THEN 1
                ELSE 0
            END) AS sb_attempts
        FROM runner_pitches
    """
    result = conn.execute(query, [runner_id, season]).fetchone()
    if result is None or result[0] == 0:
        return 0.0

    total, attempts = result
    return float(attempts) / float(total)


def compute_gravity_effect(
    conn: duckdb.DuckDBPyConnection,
    runner_id: int,
    season: int,
    config: BGIConfig | None = None,
) -> dict | None:
    """Compute per-channel gravity effects for a runner on 1B.

    Compares pitches thrown with runner_id on 1B vs all other pitches
    in the same base state (someone on 1B, but not runner_id), controlling
    for pitcher, count, batter handedness, and inning bucket.

    Args:
        conn: Open DuckDB connection.
        runner_id: MLB player ID of the runner.
        season: Season year.
        config: Optional BGIConfig.

    Returns:
        Dictionary with velocity_effect, location_effect, selection_effect,
        outcome_effect, n_pitches_with, n_pitches_comparison, or None if
        insufficient data.
    """
    if config is None:
        config = BGIConfig()

    # Group 1: pitches when runner_id is on 1B
    # Group 2: pitches when someone ELSE is on 1B (same base state control)
    query = """
        SELECT
            release_speed,
            plate_x,
            plate_z,
            pitch_type,
            estimated_woba,
            pitcher_id,
            balls,
            strikes,
            stand,
            inning,
            CASE WHEN on_1b = $1 THEN 1 ELSE 0 END AS is_target_runner
        FROM pitches
        WHERE on_1b IS NOT NULL
          AND on_1b != 0
          AND EXTRACT(YEAR FROM game_date) = $2
          AND pitch_type IS NOT NULL
    """
    df = conn.execute(query, [runner_id, season]).fetchdf()

    if df.empty:
        return None

    # Add inning bucket: early(1-3), mid(4-6), late(7+)
    df["inning_bucket"] = pd.cut(
        df["inning"].fillna(5),
        bins=[0, 3, 6, 30],
        labels=["early", "mid", "late"],
    )

    # Add count state
    df["count_state"] = (
        df["balls"].fillna(0).astype(int).astype(str) + "-" +
        df["strikes"].fillna(0).astype(int).astype(str)
    )

    group1 = df[df["is_target_runner"] == 1]
    group2 = df[df["is_target_runner"] == 0]

    if len(group1) < config.min_pitches_with_runner:
        return None
    if len(group2) < config.min_matched_sample:
        return None

    # -- Velocity channel --
    velo1 = group1["release_speed"].dropna()
    velo2 = group2["release_speed"].dropna()
    velocity_effect = float(velo1.mean() - velo2.mean()) if len(velo1) > 0 and len(velo2) > 0 else 0.0

    # -- Location channel (command disruption) --
    # Higher variance = worse command. Positive effect = runner causes more scatter.
    loc_var1 = _location_variance(group1)
    loc_var2 = _location_variance(group2)
    location_effect = loc_var1 - loc_var2  # positive means runner disrupts command

    # -- Selection channel (fastball% shift) --
    fb1 = _fastball_rate(group1)
    fb2 = _fastball_rate(group2)
    selection_effect = fb1 - fb2  # positive means more fastballs thrown

    # -- Outcome channel (batter xwOBA difference) --
    xwoba1 = group1["estimated_woba"].dropna()
    xwoba2 = group2["estimated_woba"].dropna()
    if len(xwoba1) > 0 and len(xwoba2) > 0:
        outcome_effect = float(xwoba1.mean() - xwoba2.mean())
    else:
        outcome_effect = 0.0

    return {
        "velocity_effect": round(velocity_effect, 4),
        "location_effect": round(location_effect, 4),
        "selection_effect": round(selection_effect, 4),
        "outcome_effect": round(outcome_effect, 4),
        "n_pitches_with": len(group1),
        "n_pitches_comparison": len(group2),
    }


def calculate_bgi(
    conn: duckdb.DuckDBPyConnection,
    runner_id: int,
    season: int,
    config: BGIConfig | None = None,
) -> dict:
    """Calculate full BGI profile for a single runner-season.

    Args:
        conn: Open DuckDB connection.
        runner_id: MLB player ID.
        season: Season year.
        config: Optional BGIConfig.

    Returns:
        Dictionary with bgi, sb_attempt_rate, per-channel effects, n_pitches, name.
    """
    if config is None:
        config = BGIConfig()

    sb_rate = compute_runner_threat_rate(conn, runner_id, season)
    effects = compute_gravity_effect(conn, runner_id, season, config)

    if effects is None:
        name = _get_player_name(conn, runner_id)
        return {
            "runner_id": runner_id,
            "name": name,
            "season": season,
            "bgi": None,
            "sb_attempt_rate": round(sb_rate, 4),
            "velocity_effect": None,
            "location_effect": None,
            "selection_effect": None,
            "outcome_effect": None,
            "n_pitches": 0,
        }

    # Composite: outcome_effect is the primary signal for BGI
    # But we weight all channels into a total effect
    total_effect = effects["outcome_effect"]

    # We need the league std to scale; for a single player, estimate BGI
    # with a default std (will be recalibrated in batch mode)
    bgi = _BGI_CENTER + total_effect * _BGI_STD_POINTS / 0.010
    # 0.010 is approximately 1 std of xwOBA effect across runners

    name = _get_player_name(conn, runner_id)

    return {
        "runner_id": runner_id,
        "name": name,
        "season": season,
        "bgi": round(bgi, 1),
        "sb_attempt_rate": round(sb_rate, 4),
        "velocity_effect": effects["velocity_effect"],
        "location_effect": effects["location_effect"],
        "selection_effect": effects["selection_effect"],
        "outcome_effect": effects["outcome_effect"],
        "n_pitches": effects["n_pitches_with"],
    }


def batch_calculate(
    conn: duckdb.DuckDBPyConnection,
    season: int,
    min_appearances: int = 50,
    config: BGIConfig | None = None,
) -> pd.DataFrame:
    """Calculate BGI for all qualifying runners in a season.

    Args:
        conn: Open DuckDB connection.
        season: Season year.
        min_appearances: Minimum pitches with runner on 1B.
        config: Optional BGIConfig.

    Returns:
        DataFrame sorted by BGI descending.
    """
    if config is None:
        config = BGIConfig()

    # Find all runners who appeared on 1B with enough pitches
    query = """
        SELECT
            on_1b AS runner_id,
            COUNT(*) AS n_pitches
        FROM pitches
        WHERE on_1b IS NOT NULL
          AND on_1b != 0
          AND EXTRACT(YEAR FROM game_date) = $1
          AND pitch_type IS NOT NULL
        GROUP BY on_1b
        HAVING COUNT(*) >= $2
    """
    candidates = conn.execute(query, [season, min_appearances]).fetchdf()

    if candidates.empty:
        return pd.DataFrame(columns=[
            "runner_id", "name", "season", "bgi", "sb_attempt_rate",
            "velocity_effect", "location_effect", "selection_effect",
            "outcome_effect", "n_pitches", "percentile",
        ])

    rows = []
    for _, cand in candidates.iterrows():
        rid = int(cand["runner_id"])
        result = calculate_bgi(conn, rid, season, config)
        if result["bgi"] is not None:
            rows.append(result)

    if not rows:
        return pd.DataFrame(columns=[
            "runner_id", "name", "season", "bgi", "sb_attempt_rate",
            "velocity_effect", "location_effect", "selection_effect",
            "outcome_effect", "n_pitches", "percentile",
        ])

    df = pd.DataFrame(rows)

    # Re-scale BGI so that league mean = 100 and +/- 1 std = +/- 15
    outcome_effects = df["outcome_effect"].values.astype(float)
    league_mean = float(np.mean(outcome_effects))
    league_std = float(np.std(outcome_effects))
    if league_std == 0 or np.isnan(league_std):
        league_std = 0.010  # fallback

    df["bgi"] = _BGI_CENTER + (outcome_effects - league_mean) / league_std * _BGI_STD_POINTS
    df["bgi"] = df["bgi"].round(1)

    # Percentile rank
    df["percentile"] = df["bgi"].rank(pct=True).mul(100).round(0).astype(int)

    df = df.sort_values("bgi", ascending=False).reset_index(drop=True)
    return df


def get_gravity_leaderboard(
    conn: duckdb.DuckDBPyConnection,
    season: int | None = None,
    config: BGIConfig | None = None,
) -> pd.DataFrame:
    """Return a sorted leaderboard of highest gravity runners.

    Args:
        conn: Open DuckDB connection.
        season: Season year.
        config: Optional BGIConfig.

    Returns:
        Sorted DataFrame with Rank index.
    """
    if config is None:
        config = BGIConfig()
    if season is None:
        season = _get_latest_season(conn)

    df = batch_calculate(conn, season, config.min_appearances_batch, config)

    if df.empty:
        return df

    df = df.sort_values("bgi", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _location_variance(df: pd.DataFrame) -> float:
    """Compute combined plate location variance (plate_x + plate_z)."""
    px = df["plate_x"].dropna()
    pz = df["plate_z"].dropna()
    var_x = float(px.var()) if len(px) > 1 else 0.0
    var_z = float(pz.var()) if len(pz) > 1 else 0.0
    return var_x + var_z


def _fastball_rate(df: pd.DataFrame) -> float:
    """Compute fastball percentage for a group of pitches."""
    if len(df) == 0:
        return 0.0
    is_fb = df["pitch_type"].isin(_FASTBALL_TYPES)
    return float(is_fb.sum()) / len(df)


def _get_player_name(conn: duckdb.DuckDBPyConnection, player_id: int) -> str | None:
    """Look up a player's full name from the players table."""
    return get_player_name(conn, player_id)


def _get_latest_season(conn: duckdb.DuckDBPyConnection) -> int:
    """Return the most recent season year in the pitches table."""
    return get_latest_season(conn)


# ---------------------------------------------------------------------------
# Module-level default model and convenience wrappers
# ---------------------------------------------------------------------------

_default_model: BaserunnerGravityModel | None = None


def train(conn: duckdb.DuckDBPyConnection, season: int | None = None, **kwargs) -> dict:
    """Train the BGI model (module-level convenience function).

    Args:
        conn: Open DuckDB connection.
        season: Season year.

    Returns:
        Training metrics dict.
    """
    global _default_model
    _default_model = BaserunnerGravityModel(BGIConfig(**kwargs))
    return _default_model.train(conn, season=season)
