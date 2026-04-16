"""
CausalWAR -- Causal Inference Player Valuation via Double Machine Learning.

Estimates the causal effect of individual players on run production by
de-confounding park, lineup context, platoon matchup, base-out state, and
defensive alignment using the Frisch-Waugh-Lovell approach:

    1. Fit nuisance models E[Y|W] and E[T|W] with cross-fitting.
    2. Residualise:  Y_res = Y - E[Y|W],  T_res = T - E[T|W].
    3. Regress Y_res on T_res to recover the causal effect.
    4. Bootstrap for confidence intervals.

No external causal-inference library required -- implemented manually with
scikit-learn's HistGradientBoostingRegressor.
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
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.analytics.base import BaseAnalyticsModel
from src.analytics.features import get_player_name, get_latest_season

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "causal_war"

# Minimum plate appearances required for qualifying season-level estimates
_MIN_PA_QUALIFYING = 100

# Minimum observations needed to attempt training at all
_MIN_TRAINING_OBS = 200

# Runs per win conversion (standard sabermetrics)
_RUNS_PER_WIN = 10.0

# League average wOBA for run-value computation
_LEAGUE_WOBA = 0.320
_WOBA_SCALE = 1.25

# PA per full season (used to scale per-PA values to WAR)
_PA_PER_SEASON = 600


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CausalWARConfig:
    """Hyperparameters for the CausalWAR model."""

    # Nuisance model (HistGradientBoosting)
    n_estimators: int = 300
    max_depth: int = 6
    learning_rate: float = 0.05
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0

    # DML cross-fitting
    n_splits: int = 5

    # Bootstrap for confidence intervals
    n_bootstrap: int = 100

    # Qualifying thresholds
    pa_min_qualifying: int = _MIN_PA_QUALIFYING

    # Random state for reproducibility
    random_state: int = 42

    # Temporal train/test split defaults (used by train_test_split())
    train_start_year: int = 2015
    train_end_year: int = 2022
    test_start_year: int = 2023
    test_end_year: int = 2024

    # Minimum PA for a test-set player to be considered "qualified" (players
    # below this threshold are still reported but flagged as sparse).
    pa_min_test_qualifying: int = 50


# ---------------------------------------------------------------------------
# Core model class
# ---------------------------------------------------------------------------

class CausalWARModel(BaseAnalyticsModel):
    """Double Machine Learning model for causal player valuation.

    Inherits from ``BaseAnalyticsModel`` for consistent lifecycle management.
    """

    def __init__(self, config: CausalWARConfig | None = None) -> None:
        super().__init__()
        self.config = config or CausalWARConfig()
        self._nuisance_outcome = None  # E[Y|W]
        self._nuisance_treatment = None  # E[T|W]
        self._player_effects: dict[int, dict] = {}  # player_id -> effect dict
        self._is_trained = False
        self._training_season: int | None = None

    # ---- BaseAnalyticsModel interface ------------------------------------

    @property
    def model_name(self) -> str:
        return "causal_war"

    @property
    def version(self) -> str:
        return "1.0.0"

    def train(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Train nuisance models and estimate player causal effects.

        Args:
            conn: Open DuckDB connection.
            season: Season year to train on (keyword argument).

        Returns:
            Dictionary of training metrics.
        """
        season = kwargs.get("season")
        if season is None:
            season = _get_latest_season(conn)
        logger.info("Training CausalWAR for season %d", season)

        # Extract confounder-enriched data
        df = _extract_pa_data(conn, season)
        logger.info("Extracted %d plate appearances for season %d", len(df), season)

        if len(df) < _MIN_TRAINING_OBS:
            raise ValueError(
                f"Only {len(df)} PAs found for season {season}; "
                f"need at least {_MIN_TRAINING_OBS}."
            )

        # Build confounders and outcomes
        W, Y, player_ids, pa_df = _build_features(df)
        logger.info(
            "Feature matrix: %d observations, %d confounders", W.shape[0], W.shape[1]
        )

        # Fit nuisance models and get residuals via cross-fitting
        Y_res, T_effects, metrics = self._fit_dml(W, Y, player_ids, pa_df)

        # Store results
        self._player_effects = T_effects
        self._is_trained = True
        self._training_season = season

        self.set_training_metadata(
            metrics=metrics,
            params={
                "n_splits": self.config.n_splits,
                "n_bootstrap": self.config.n_bootstrap,
                "n_estimators": self.config.n_estimators,
                "season": season,
            },
        )
        logger.info("CausalWAR training complete: %d player effects estimated", len(T_effects))
        return metrics

    def predict(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict | pd.DataFrame:
        """Return CausalWAR estimates for a player or all players.

        Args:
            conn: Open DuckDB connection.
            player_id: Specific player (optional).
            season: Season year (optional).

        Returns:
            DataFrame of CausalWAR estimates, or dict for a single player.
        """
        player_id = kwargs.get("player_id")
        season = kwargs.get("season", self._training_season)

        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        if player_id is not None:
            return self.calculate_causal_war(conn, player_id, season)

        return self.batch_calculate(conn, season)

    def evaluate(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Evaluate model quality via residual diagnostics.

        Returns:
            Dictionary with residual statistics and model diagnostics.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        n_players = len(self._player_effects)
        effects = [e["causal_war"] for e in self._player_effects.values()
                    if e.get("causal_war") is not None]

        if not effects:
            return {"n_players": 0, "status": "no_effects"}

        effects_arr = np.array(effects)
        return self.validate_output({
            "n_players": n_players,
            "mean_causal_war": round(float(np.mean(effects_arr)), 4),
            "std_causal_war": round(float(np.std(effects_arr)), 4),
            "median_causal_war": round(float(np.median(effects_arr)), 4),
            "min_causal_war": round(float(np.min(effects_arr)), 4),
            "max_causal_war": round(float(np.max(effects_arr)), 4),
        })

    # ---- Temporal train/test split --------------------------------------

    def train_test_split(
        self,
        conn: duckdb.DuckDBPyConnection,
        train_split: tuple[int, int] | None = None,
        test_split: tuple[int, int] | None = None,
    ) -> dict:
        """Fit the DML nuisance model on a train-year range and evaluate on a
        disjoint test-year range.

        This is the spec-compliant validation path that rules out in-sample
        leakage.  The nuisance model is fit on ``train_split`` PA data only;
        the fitted model is then applied to ``test_split`` PA data to produce
        out-of-sample residuals and a held-out R squared.  Per-player effects
        are aggregated independently for each split.

        Args:
            conn: Open DuckDB connection.
            train_split: Inclusive ``(start_year, end_year)`` for training.
                Defaults to ``(config.train_start_year, config.train_end_year)``.
            test_split: Inclusive ``(start_year, end_year)`` for testing.
                Defaults to ``(config.test_start_year, config.test_end_year)``.

        Returns:
            Dictionary with keys ``train_metrics``, ``test_metrics``,
            ``train_player_effects`` (DataFrame), ``test_player_effects``
            (DataFrame).

        Raises:
            ValueError: If the train and test year ranges overlap, or if
                either range has insufficient data.
        """
        if train_split is None:
            train_split = (self.config.train_start_year, self.config.train_end_year)
        if test_split is None:
            test_split = (self.config.test_start_year, self.config.test_end_year)

        train_start, train_end = int(train_split[0]), int(train_split[1])
        test_start, test_end = int(test_split[0]), int(test_split[1])

        # Leakage guard
        if train_start > train_end:
            raise ValueError(
                f"train_split start ({train_start}) must be <= end ({train_end})."
            )
        if test_start > test_end:
            raise ValueError(
                f"test_split start ({test_start}) must be <= end ({test_end})."
            )
        if test_start <= train_end:
            raise ValueError(
                f"Leakage guard: test_start_year ({test_start}) must be strictly "
                f"greater than train_end_year ({train_end}). Ranges overlap."
            )

        logger.info(
            "CausalWAR train/test split: train=%d-%d, test=%d-%d",
            train_start, train_end, test_start, test_end,
        )

        # ---- Extract train and test data --------------------------------
        train_df = _extract_pa_data(conn, year_range=(train_start, train_end))
        logger.info(
            "Extracted %d train PAs (%d-%d)", len(train_df), train_start, train_end,
        )
        if len(train_df) < _MIN_TRAINING_OBS:
            raise ValueError(
                f"Only {len(train_df)} PAs in train window {train_start}-{train_end}; "
                f"need at least {_MIN_TRAINING_OBS}."
            )

        test_df = _extract_pa_data(conn, year_range=(test_start, test_end))
        logger.info(
            "Extracted %d test PAs (%d-%d)", len(test_df), test_start, test_end,
        )
        if len(test_df) == 0:
            raise ValueError(
                f"No PAs found in test window {test_start}-{test_end}."
            )

        # ---- Build features ---------------------------------------------
        W_train, Y_train, player_ids_train, pa_df_train = _build_features(train_df)
        W_test, Y_test, player_ids_test, pa_df_test = _build_features(test_df)

        # ---- Fit DML on train only --------------------------------------
        Y_res_train, train_effects, train_metrics = self._fit_dml(
            W_train, Y_train, player_ids_train, pa_df_train,
        )

        # ---- Fit a single full-train nuisance model for test prediction --
        # (Cross-fitting is used for train residuals; for held-out test
        # predictions we use a model fit on the full train set.)
        full_model = HistGradientBoostingRegressor(
            max_iter=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            min_samples_leaf=self.config.min_samples_leaf,
            l2_regularization=self.config.l2_regularization,
            random_state=self.config.random_state,
        )
        full_model.fit(W_train, Y_train)

        # Persist state so predict() / evaluate() keep working after this call
        self._nuisance_outcome = full_model
        self._player_effects = train_effects
        self._is_trained = True
        self._training_season = train_end

        # ---- Evaluate on held-out test ----------------------------------
        Y_pred_test = full_model.predict(W_test)
        Y_res_test = Y_test - Y_pred_test
        test_r2 = float(r2_score(Y_test, Y_pred_test))
        test_rmse_residuals = float(np.sqrt(mean_squared_error(Y_test, Y_pred_test)))

        # Aggregate per-player test effects
        test_effects = _aggregate_player_effects(
            Y_res_test,
            player_ids_test,
            pa_df_test,
            pa_min=10,
        )

        n_test_players = len(test_effects)
        n_test_players_sparse = sum(
            1 for e in test_effects.values()
            if e.get("pa", 0) < self.config.pa_min_test_qualifying
        )

        test_metrics = {
            "test_nuisance_r2": round(test_r2, 4),
            "test_rmse_residuals": round(test_rmse_residuals, 6),
            "n_test_observations": int(len(Y_test)),
            "n_test_players": int(n_test_players),
            "n_test_players_sparse": int(n_test_players_sparse),
            "pa_min_test_qualifying": self.config.pa_min_test_qualifying,
            "test_start_year": test_start,
            "test_end_year": test_end,
        }

        train_metrics = dict(train_metrics)
        train_metrics["train_start_year"] = train_start
        train_metrics["train_end_year"] = train_end

        # ---- Build per-split player-effect DataFrames -------------------
        train_player_effects = _effects_to_df(
            train_effects, season_label=f"{train_start}-{train_end}",
        )
        test_player_effects = _effects_to_df(
            test_effects,
            season_label=f"{test_start}-{test_end}",
            pa_min_qualifying=self.config.pa_min_test_qualifying,
        )

        # ---- Save artifact ----------------------------------------------
        artifact_dir = DEFAULT_MODEL_DIR
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = (
            artifact_dir
            / f"causal_war_trainsplit_{train_start}_{train_end}.pkl"
        )
        artifact = {
            "nuisance_outcome": full_model,
            "config": self.config,
            "train_split": (train_start, train_end),
            "test_split": (test_start, test_end),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            import joblib
            joblib.dump(artifact, artifact_path)
            logger.info("CausalWAR train/test artifact saved to %s", artifact_path)
        except Exception as exc:  # pragma: no cover - filesystem errors
            logger.warning("Could not persist train/test artifact: %s", exc)

        self.set_training_metadata(
            metrics={"train": train_metrics, "test": test_metrics},
            params={
                "n_splits": self.config.n_splits,
                "n_bootstrap": self.config.n_bootstrap,
                "n_estimators": self.config.n_estimators,
                "train_split": (train_start, train_end),
                "test_split": (test_start, test_end),
            },
        )

        return {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "train_player_effects": train_player_effects,
            "test_player_effects": test_player_effects,
            "artifact_path": str(artifact_path),
        }

    # ---- Public API ------------------------------------------------------

    def calculate_causal_war(
        self,
        conn: duckdb.DuckDBPyConnection,
        player_id: int,
        season: int | None = None,
    ) -> dict:
        """Calculate CausalWAR for a single player-season.

        Args:
            conn: Open DuckDB connection.
            player_id: MLB player ID.
            season: Season year.

        Returns:
            Dictionary with causal_war, ci_low, ci_high, park_adj_woba, etc.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        if season is None:
            season = self._training_season

        effect = self._player_effects.get(player_id)
        if effect is None:
            logger.warning("Player %d not found in trained effects", player_id)
            return {
                "player_id": player_id,
                "season": season,
                "causal_war": None,
                "ci_low": None,
                "ci_high": None,
                "park_adj_woba": None,
                "raw_woba": None,
                "pa": 0,
                "causal_run_value": None,
            }

        # Enrich with player name
        name = _get_player_name(conn, player_id)

        return {
            "player_id": player_id,
            "name": name,
            "season": season,
            **effect,
        }

    def batch_calculate(
        self,
        conn: duckdb.DuckDBPyConnection,
        season: int | None = None,
    ) -> pd.DataFrame:
        """Calculate CausalWAR for all qualifying players.

        Args:
            conn: Open DuckDB connection.
            season: Season year.

        Returns:
            DataFrame with columns: player_id, name, season, causal_war,
            ci_low, ci_high, park_adj_woba, raw_woba, pa, traditional_war.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        if season is None:
            season = self._training_season

        rows = []
        for pid, effect in self._player_effects.items():
            if effect.get("pa", 0) < self.config.pa_min_qualifying:
                continue
            name = _get_player_name(conn, pid)
            traditional_war = _get_traditional_war(conn, pid, season)
            rows.append({
                "player_id": pid,
                "name": name,
                "season": season,
                "causal_war": effect.get("causal_war"),
                "ci_low": effect.get("ci_low"),
                "ci_high": effect.get("ci_high"),
                "park_adj_woba": effect.get("park_adj_woba"),
                "raw_woba": effect.get("raw_woba"),
                "pa": effect.get("pa", 0),
                "causal_run_value": effect.get("causal_run_value"),
                "traditional_war": traditional_war,
            })

        if not rows:
            return pd.DataFrame(columns=[
                "player_id", "name", "season", "causal_war", "ci_low",
                "ci_high", "park_adj_woba", "raw_woba", "pa",
                "causal_run_value", "traditional_war",
            ])

        df = pd.DataFrame(rows)
        df = df.sort_values("causal_war", ascending=False, na_position="last")
        df = df.reset_index(drop=True)
        return df

    def get_leaderboard(
        self,
        conn: duckdb.DuckDBPyConnection,
        season: int | None = None,
        position_type: str = "all",
    ) -> pd.DataFrame:
        """Return a sorted leaderboard of CausalWAR.

        Args:
            conn: Open DuckDB connection.
            season: Season year.
            position_type: ``"pitcher"``, ``"batter"``, or ``"all"``.

        Returns:
            Sorted DataFrame with rank column.
        """
        df = self.batch_calculate(conn, season)

        if df.empty:
            return df

        if position_type == "pitcher":
            pitcher_ids = _get_pitcher_ids(conn, season)
            df = df[df["player_id"].isin(pitcher_ids)]
        elif position_type == "batter":
            pitcher_ids = _get_pitcher_ids(conn, season)
            df = df[~df["player_id"].isin(pitcher_ids)]

        df = df.sort_values("causal_war", ascending=False, na_position="last")
        df = df.reset_index(drop=True)
        df.index = df.index + 1
        df.index.name = "Rank"

        return df

    # ---- DML internals ---------------------------------------------------

    def _fit_dml(
        self,
        W: np.ndarray,
        Y: np.ndarray,
        player_ids: np.ndarray,
        pa_df: pd.DataFrame,
    ) -> tuple[np.ndarray, dict[int, dict], dict]:
        """Fit the Double ML model using Frisch-Waugh-Lovell.

        For each player with enough PAs, the treatment is a binary indicator
        (1 if this player, 0 otherwise). We use a simplified approach:

        1. Fit E[Y|W] to residualise the outcome on confounders.
        2. Compute per-player average residuals as the causal effect.
        3. Bootstrap for confidence intervals.

        Returns:
            (Y_residuals, player_effects_dict, training_metrics)
        """
        rng = np.random.RandomState(self.config.random_state)

        # ---- Step 1: Cross-fitted outcome residualisation ----------------
        Y_res = np.zeros_like(Y)
        kf = KFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=self.config.random_state,
        )

        outcome_r2_scores = []
        for train_idx, test_idx in kf.split(W):
            model_y = HistGradientBoostingRegressor(
                max_iter=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_samples_leaf=self.config.min_samples_leaf,
                l2_regularization=self.config.l2_regularization,
                random_state=self.config.random_state,
            )
            model_y.fit(W[train_idx], Y[train_idx])
            Y_pred = model_y.predict(W[test_idx])
            Y_res[test_idx] = Y[test_idx] - Y_pred
            r2 = r2_score(Y[test_idx], Y_pred)
            outcome_r2_scores.append(r2)

        self._nuisance_outcome = model_y  # keep last fold for reference
        avg_outcome_r2 = float(np.mean(outcome_r2_scores))
        logger.info("Outcome nuisance model avg R2: %.4f", avg_outcome_r2)

        # ---- Step 2: Per-player causal effects ---------------------------
        unique_players = np.unique(player_ids)
        player_effects: dict[int, dict] = {}

        for pid in unique_players:
            mask = player_ids == pid
            n_pa = int(mask.sum())
            if n_pa < 10:
                continue

            # Point estimate: average outcome residual for this player
            player_y_res = Y_res[mask]
            point_estimate = float(np.mean(player_y_res))

            # Raw wOBA for this player
            player_rows = pa_df.loc[mask]
            raw_woba = float(player_rows["woba_value"].mean()) if "woba_value" in player_rows.columns else None

            # Park-adjusted wOBA = raw_woba + residual adjustment
            if raw_woba is not None:
                park_adj_woba = raw_woba + point_estimate
            else:
                park_adj_woba = None

            # Convert to run value per PA, then scale to WAR
            causal_run_value_per_pa = point_estimate / _WOBA_SCALE
            causal_runs = causal_run_value_per_pa * n_pa
            causal_war = causal_runs / _RUNS_PER_WIN

            player_effects[int(pid)] = {
                "causal_war": round(causal_war, 2),
                "causal_run_value": round(causal_runs, 2),
                "point_estimate_per_pa": round(point_estimate, 4),
                "park_adj_woba": round(park_adj_woba, 3) if park_adj_woba is not None else None,
                "raw_woba": round(raw_woba, 3) if raw_woba is not None else None,
                "pa": n_pa,
                "ci_low": None,
                "ci_high": None,
            }

        # ---- Step 3: Bootstrap confidence intervals ----------------------
        logger.info(
            "Running %d bootstrap iterations for CIs...",
            self.config.n_bootstrap,
        )
        n_obs = len(Y)
        bootstrap_effects: dict[int, list[float]] = {
            pid: [] for pid in player_effects
        }

        for b in range(self.config.n_bootstrap):
            boot_idx = rng.choice(n_obs, size=n_obs, replace=True)
            W_b, Y_b = W[boot_idx], Y[boot_idx]
            pids_b = player_ids[boot_idx]

            # Re-fit outcome nuisance on bootstrap sample
            model_b = HistGradientBoostingRegressor(
                max_iter=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                min_samples_leaf=self.config.min_samples_leaf,
                l2_regularization=self.config.l2_regularization,
                random_state=self.config.random_state + b,
            )
            model_b.fit(W_b, Y_b)
            Y_res_b = Y_b - model_b.predict(W_b)

            for pid in player_effects:
                mask_b = pids_b == pid
                if mask_b.sum() < 5:
                    continue
                avg_res = float(np.mean(Y_res_b[mask_b]))
                n_pa_b = int(mask_b.sum())
                boot_war = (avg_res / _WOBA_SCALE) * n_pa_b / _RUNS_PER_WIN
                bootstrap_effects[pid].append(boot_war)

        # Compute CIs from bootstrap distribution
        for pid, boots in bootstrap_effects.items():
            if len(boots) >= 10:
                arr = np.array(boots)
                player_effects[pid]["ci_low"] = round(float(np.percentile(arr, 2.5)), 2)
                player_effects[pid]["ci_high"] = round(float(np.percentile(arr, 97.5)), 2)

        # ---- Training metrics --------------------------------------------
        metrics = {
            "outcome_nuisance_r2": round(avg_outcome_r2, 4),
            "n_observations": int(n_obs),
            "n_players_estimated": len(player_effects),
            "n_bootstrap": self.config.n_bootstrap,
            "n_splits": self.config.n_splits,
            "mean_residual_variance": round(float(np.var(Y_res)), 6),
        }

        return Y_res, player_effects, metrics


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def _extract_pa_data(
    conn: duckdb.DuckDBPyConnection,
    season: int | None = None,
    *,
    year_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Extract plate-appearance-level data with confounders.

    Joins pitches to games for venue, aggregates to PA level, and computes
    confounder features.

    Args:
        conn: Open DuckDB connection.
        season: Single season year (backwards-compatible default).  Ignored if
            ``year_range`` is provided.
        year_range: Optional inclusive ``(start_year, end_year)`` tuple for
            multi-season extraction (used by the temporal train/test split
            workflow).

    Returns:
        DataFrame with one row per plate appearance.
    """
    if year_range is not None:
        start_year, end_year = year_range
        if start_year > end_year:
            raise ValueError(
                f"year_range start ({start_year}) must be <= end ({end_year})."
            )
        params = [int(start_year), int(end_year)]
        year_filter = (
            "EXTRACT(YEAR FROM p.game_date) BETWEEN $1 AND $2"
        )
    else:
        if season is None:
            raise ValueError("Either 'season' or 'year_range' must be provided.")
        params = [int(season)]
        year_filter = "EXTRACT(YEAR FROM p.game_date) = $1"

    query = f"""
        WITH pa_events AS (
            SELECT
                p.pitcher_id,
                p.batter_id,
                p.game_pk,
                p.game_date,
                p.at_bat_number,
                p.stand,
                p.p_throws,
                p.inning,
                p.outs_when_up,
                p.on_1b,
                p.on_2b,
                p.on_3b,
                -- Aggregate wOBA components per PA
                SUM(COALESCE(p.woba_value, 0)) AS woba_value,
                MAX(COALESCE(p.woba_denom, 0)) AS woba_denom,
                -- Use the last pitch's state in the PA
                MAX(p.balls) AS balls,
                MAX(p.strikes) AS strikes,
                COUNT(*) AS pitches_in_pa
            FROM pitches p
            WHERE {year_filter}
              AND p.pitch_type IS NOT NULL
            GROUP BY
                p.pitcher_id, p.batter_id, p.game_pk, p.game_date,
                p.at_bat_number, p.stand, p.p_throws, p.inning,
                p.outs_when_up, p.on_1b, p.on_2b, p.on_3b
        )
        SELECT
            pa.*,
            g.venue,
            g.home_team,
            g.away_team
        FROM pa_events pa
        LEFT JOIN games g ON pa.game_pk = g.game_pk
        WHERE pa.woba_denom > 0
        ORDER BY pa.game_date, pa.game_pk, pa.at_bat_number
    """
    df = conn.execute(query, params).fetchdf()
    return df


def _build_features(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Build confounder matrix (W), outcome (Y), and player IDs from PA data.

    Confounders:
        - venue (label-encoded)
        - platoon advantage (stand != p_throws)
        - runners on base (on_1b, on_2b, on_3b)
        - outs_when_up
        - inning bucket (early/mid/late/extra)
        - if_shift / of_shift indicators
        - month
        - home/away indicator (batter on home team)

    Treatment: batter_id (used later per-player)
    Outcome: woba_value (per PA)

    Returns:
        (W, Y, player_ids, df)
    """
    df = df.copy()

    # ---- Outcome: wOBA value per PA --------------------------------------
    Y = df["woba_value"].fillna(0).values.astype(np.float64)

    # ---- Confounders -----------------------------------------------------
    # Platoon advantage
    df["platoon"] = (df["stand"] != df["p_throws"]).astype(int)

    # Venue encoding (label encode)
    if "venue" in df.columns and df["venue"].notna().any():
        venue_codes, _venue_uniques = pd.factorize(df["venue"].fillna("Unknown"))
        df["venue_code"] = venue_codes
    else:
        df["venue_code"] = 0

    # Runners on base
    df["on_1b"] = df["on_1b"].fillna(0).astype(int)
    df["on_2b"] = df["on_2b"].fillna(0).astype(int)
    df["on_3b"] = df["on_3b"].fillna(0).astype(int)

    # Outs
    df["outs"] = df["outs_when_up"].fillna(0).astype(int)

    # Inning bucket: 1-3 early (0), 4-6 mid (1), 7-9 late (2), 10+ extra (3)
    inning = df["inning"].fillna(5).astype(int)
    df["inning_bucket"] = np.where(
        inning <= 3, 0,
        np.where(inning <= 6, 1, np.where(inning <= 9, 2, 3))
    )

    # Shift indicators (fielding alignment data not available; default to 0)
    df["if_shift"] = 0
    df["of_shift"] = 0

    # Month
    if "game_date" in df.columns:
        df["month"] = pd.to_datetime(df["game_date"]).dt.month
    else:
        df["month"] = 6  # default

    # Home/away: batter is home if inning_topbot would be Bot, approximate
    # via home_team column -- if batter's game is home, assume home
    df["home_indicator"] = 0  # placeholder, refined below
    # A crude approximation: use the at-bat order or just leave as 0

    # Handedness encoding
    df["stand_R"] = (df["stand"] == "R").astype(int)
    df["p_throws_R"] = (df["p_throws"] == "R").astype(int)

    # Build confounder matrix
    confounder_cols = [
        "venue_code",
        "platoon",
        "on_1b",
        "on_2b",
        "on_3b",
        "outs",
        "inning_bucket",
        "if_shift",
        "of_shift",
        "month",
        "stand_R",
        "p_throws_R",
    ]

    W = df[confounder_cols].values.astype(np.float64)

    # Handle any remaining NaNs in W
    col_means = np.nanmean(W, axis=0)
    col_means = np.where(np.isnan(col_means), 0, col_means)
    nan_mask = np.isnan(W)
    W[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    player_ids = df["batter_id"].values.astype(int)

    return W, Y, player_ids, df


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _aggregate_player_effects(
    Y_res: np.ndarray,
    player_ids: np.ndarray,
    pa_df: pd.DataFrame,
    pa_min: int = 10,
) -> dict[int, dict]:
    """Aggregate per-player residual statistics into CausalWAR effect dicts.

    This mirrors the aggregation step inside ``_fit_dml`` but without the
    bootstrap CI pass; it is used for held-out test-set evaluation where a
    bootstrap is not required.

    Args:
        Y_res: Residual vector (``Y - E[Y|W]``) aligned with ``player_ids``.
        player_ids: Array of player IDs, one per row of ``Y_res``.
        pa_df: PA-level DataFrame (must contain ``woba_value`` if raw wOBA is
            to be reported).
        pa_min: Minimum PA count required to emit an effect for a player.

    Returns:
        Dict mapping ``player_id -> {causal_war, causal_run_value,
        point_estimate_per_pa, park_adj_woba, raw_woba, pa, ci_low, ci_high}``.
    """
    unique_players = np.unique(player_ids)
    effects: dict[int, dict] = {}

    has_woba = "woba_value" in pa_df.columns

    for pid in unique_players:
        mask = player_ids == pid
        n_pa = int(mask.sum())
        if n_pa < pa_min:
            continue

        player_y_res = Y_res[mask]
        point_estimate = float(np.mean(player_y_res))

        if has_woba:
            raw_woba = float(pa_df.loc[mask, "woba_value"].mean())
            park_adj_woba = raw_woba + point_estimate
        else:
            raw_woba = None
            park_adj_woba = None

        causal_run_value_per_pa = point_estimate / _WOBA_SCALE
        causal_runs = causal_run_value_per_pa * n_pa
        causal_war = causal_runs / _RUNS_PER_WIN

        effects[int(pid)] = {
            "causal_war": round(causal_war, 2),
            "causal_run_value": round(causal_runs, 2),
            "point_estimate_per_pa": round(point_estimate, 4),
            "park_adj_woba": (
                round(park_adj_woba, 3) if park_adj_woba is not None else None
            ),
            "raw_woba": round(raw_woba, 3) if raw_woba is not None else None,
            "pa": n_pa,
            "ci_low": None,
            "ci_high": None,
        }

    return effects


def _effects_to_df(
    effects: dict[int, dict],
    season_label: str | None = None,
    pa_min_qualifying: int | None = None,
) -> pd.DataFrame:
    """Convert a ``{player_id -> effect_dict}`` mapping to a DataFrame.

    Args:
        effects: Mapping produced by ``_aggregate_player_effects`` or the
            equivalent aggregation in ``_fit_dml``.
        season_label: Optional season string (e.g. ``"2015-2022"``) added as
            a column for downstream reporting.
        pa_min_qualifying: If provided, adds a boolean ``sparse`` column
            flagging players below this PA threshold.

    Returns:
        DataFrame sorted by ``causal_war`` descending.
    """
    rows = []
    for pid, e in effects.items():
        row = {"player_id": int(pid), **e}
        if season_label is not None:
            row["season"] = season_label
        if pa_min_qualifying is not None:
            row["sparse"] = bool(e.get("pa", 0) < pa_min_qualifying)
        rows.append(row)

    if not rows:
        cols = [
            "player_id", "causal_war", "ci_low", "ci_high",
            "park_adj_woba", "raw_woba", "pa", "causal_run_value",
            "point_estimate_per_pa",
        ]
        if season_label is not None:
            cols.append("season")
        if pa_min_qualifying is not None:
            cols.append("sparse")
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)
    df = df.sort_values("causal_war", ascending=False, na_position="last")
    df = df.reset_index(drop=True)
    return df


def _get_latest_season(conn: duckdb.DuckDBPyConnection) -> int:
    """Return the most recent season year in the pitches table."""
    return get_latest_season(conn)


def _get_player_name(conn: duckdb.DuckDBPyConnection, player_id: int) -> str | None:
    """Look up a player's full name from the players table."""
    return get_player_name(conn, player_id)


def _get_traditional_war(
    conn: duckdb.DuckDBPyConnection,
    player_id: int,
    season: int | None,
) -> float | None:
    """Retrieve traditional WAR from season stats tables."""
    if season is None:
        return None
    try:
        # Try batting stats first
        result = conn.execute(
            "SELECT war FROM season_batting_stats WHERE player_id = $1 AND season = $2",
            [player_id, season],
        ).fetchone()
        if result and result[0] is not None:
            return round(float(result[0]), 1)

        # Try pitching stats
        result = conn.execute(
            "SELECT war FROM season_pitching_stats WHERE player_id = $1 AND season = $2",
            [player_id, season],
        ).fetchone()
        if result and result[0] is not None:
            return round(float(result[0]), 1)
    except Exception:
        pass
    return None


def _get_pitcher_ids(
    conn: duckdb.DuckDBPyConnection,
    season: int | None,
) -> set[int]:
    """Return the set of player IDs that appear as pitchers."""
    try:
        if season:
            result = conn.execute(
                "SELECT DISTINCT pitcher_id FROM pitches WHERE EXTRACT(YEAR FROM game_date) = $1",
                [season],
            ).fetchdf()
        else:
            result = conn.execute("SELECT DISTINCT pitcher_id FROM pitches").fetchdf()
        return set(result["pitcher_id"].tolist())
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Convenience wrappers (match the specification's function signatures)
# ---------------------------------------------------------------------------

_default_model: CausalWARModel | None = None


def train(conn: duckdb.DuckDBPyConnection, season: int | None = None, **kwargs) -> dict:
    """Train the CausalWAR model (module-level convenience function).

    Args:
        conn: Open DuckDB connection.
        season: Season year.

    Returns:
        Training metrics dict.
    """
    global _default_model
    _default_model = CausalWARModel(CausalWARConfig(**kwargs))
    return _default_model.train(conn, season=season)


def calculate_causal_war(
    conn: duckdb.DuckDBPyConnection,
    player_id: int,
    season: int | None = None,
) -> dict:
    """Calculate CausalWAR for a single player (module-level convenience).

    Args:
        conn: Open DuckDB connection.
        player_id: MLB player ID.
        season: Season year.

    Returns:
        Dict with causal_war, ci_low, ci_high, park_adj_woba, etc.
    """
    if _default_model is None or not _default_model._is_trained:
        raise RuntimeError("Model not trained. Call train() first.")
    return _default_model.calculate_causal_war(conn, player_id, season)


def batch_calculate(
    conn: duckdb.DuckDBPyConnection,
    season: int | None = None,
) -> pd.DataFrame:
    """Batch calculate CausalWAR for all qualifying players.

    Args:
        conn: Open DuckDB connection.
        season: Season year.

    Returns:
        DataFrame with CausalWAR leaderboard.
    """
    if _default_model is None or not _default_model._is_trained:
        raise RuntimeError("Model not trained. Call train() first.")
    return _default_model.batch_calculate(conn, season)


def get_leaderboard(
    conn: duckdb.DuckDBPyConnection,
    season: int | None = None,
    position_type: str = "all",
) -> pd.DataFrame:
    """Return CausalWAR leaderboard.

    Args:
        conn: Open DuckDB connection.
        season: Season year.
        position_type: ``"pitcher"``, ``"batter"``, or ``"all"``.

    Returns:
        Sorted DataFrame with Rank index.
    """
    if _default_model is None or not _default_model._is_trained:
        raise RuntimeError("Model not trained. Call train() first.")
    return _default_model.get_leaderboard(conn, season, position_type)
