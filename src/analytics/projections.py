"""
ProjectionModel -- Next-Season WAR Projections (Marcel + Statcast Overlay).

Produces forward-looking next-season WAR projections for batters and pitchers
by combining a Marcel-style baseline with a Statcast forward-indicator overlay.

Design:

    1. Marcel baseline.  Weighted average of the prior 3 seasons (5/4/3
       weights for the most recent through oldest), regressed toward league
       mean using a PA/IP-equivalent prior (~1200 PA for batters, ~134 IP
       for pitchers).  Apply an age curve (+0.5 WAR/yr below 27, flat 27-31,
       -0.5 WAR/yr above 31).
    2. Statcast overlay.  For batters, if xwOBA exceeds wOBA over the prior
       2 seasons by >= 0.015, boost the projection (positive regression
       candidate).  For pitchers, if xFIP < FIP over the prior 2 seasons by
       >= 0.30, boost.  Symmetric on the other side.  Capped at +/- 0.5 WAR.
    3. Public API: ``fit(train_seasons)`` then ``project(target_season,
       players=None)``.  ``save(path)`` / ``load(path)`` mirror the CausalWAR
       persistence pattern.

The model is deliberately simple, defensible, and reproducible -- the bar
is "honest baseline that beats Marcel by leveraging Statcast forward
indicators", not "best-in-class projection system".

Data contract:

    season_batting_stats: player_id, season, pa, war, woba (often NULL),
        xwoba (NULL in current DB -- derived from pitches.estimated_woba).
    season_pitching_stats: player_id, season, ip, war, fip (NULL), xfip
        (NULL -- derived from pitches.estimated_woba against the pitcher).
    pitches: estimated_woba, woba_value, woba_denom, batter_id, pitcher_id,
        game_date.
    data/player_ages.parquet: pre-cached (player_id, season, age) from
        Baseball-Reference bwar_bat / bwar_pitch (Chadwick mlbam_id keyed).

Leakage discipline: ``fit`` only consumes seasons in ``train_seasons``;
``project(target_season)`` requires the prior 3 seasons to be in
``train_seasons`` (i.e. all priors are strictly < target_season).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb
import joblib
import numpy as np
import pandas as pd

from src.analytics.base import BaseAnalyticsModel

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "projections"
DEFAULT_AGE_CACHE = _PROJECT_ROOT / "data" / "player_ages.parquet"
DEFAULT_INJURY_LABELS = _PROJECT_ROOT / "data" / "injury_labels.parquet"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ProjectionConfig:
    """Hyperparameters for the projection model."""

    # Marcel weighted-average weights (most recent -> oldest)
    weights: tuple[float, float, float] = (5.0, 4.0, 3.0)

    # Regression-to-mean PA / IP equivalents
    batter_regression_pa: float = 1200.0
    pitcher_regression_ip: float = 134.0

    # Min PA/IP across the 3-yr window to be a qualified projection
    min_total_pa_batter: float = 100.0
    min_total_ip_pitcher: float = 30.0

    # Age curve
    peak_age_low: int = 27
    peak_age_high: int = 31
    age_slope_young: float = 0.5     # WAR per year below peak
    age_slope_old: float = -0.5      # WAR per year above peak
    default_age: float = 28.0        # fallback if age unknown

    # Statcast overlay thresholds
    batter_xwoba_threshold: float = 0.015   # 15 wOBA points
    pitcher_xfip_threshold: float = 0.30    # 0.30 ERA points
    overlay_cap_war: float = 0.5            # +/- cap

    # Overlay scale: convert (xwOBA - wOBA) gap into a WAR adjustment.
    # A 30-point xwOBA-wOBA gap historically projects ~+0.5 WAR.
    batter_overlay_war_per_woba_pt: float = 0.5 / 0.030
    pitcher_overlay_war_per_fip_pt: float = 0.5 / 0.60

    # Random state
    random_state: int = 42

    # ---- v2 features (toggle on for v2; default off keeps v1.1 behaviour) ----
    # v2.A: Tommy John surgery dampener.  If enabled, pitchers with a TJ surgery
    # within ``tj_window_months`` of the target season's start are dampened by
    # ``tj_dampener_war``.  Capped so projection cannot fall below the
    # *current performance baseline* (Marcel-only) by more than the dampener.
    enable_tj_flag: bool = False
    tj_window_months: int = 24
    tj_dampener_war: float = -0.75   # mid-point of -0.5 to -1.0 range

    # v2.B: Empirically-calibrated age curve.  When True, ``_age_adjustment``
    # consults the calibrated WAR-delta-by-age table (loaded into the model
    # at fit time) and falls back to the canonical curve when sample size
    # at a given age is below ``age_curve_min_sample``.
    enable_calibrated_age_curve: bool = False
    age_curve_min_sample: int = 30
    age_curve_smoothing_window: int = 3
    age_curve_train_year_lo: int = 2015
    age_curve_train_year_hi: int = 2023  # inclusive; must be < target_season

    # v2.C: Role-change feature.  SP<->RP transitions get a flat WAR adjustment.
    enable_role_change: bool = False
    role_change_war: float = -0.30
    role_change_cap_war: float = 0.50    # cap absolute role adjustment
    role_sp_threshold: float = 0.50      # gs / g >= 0.5 -> SP, else RP


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ProjectionModel(BaseAnalyticsModel):
    """Next-season WAR projection model (Marcel + Statcast overlay)."""

    def __init__(self, config: ProjectionConfig | None = None) -> None:
        super().__init__()
        self.config = config or ProjectionConfig()

        # Filled in by fit()
        self._train_seasons: list[int] = []
        self._batter_history: pd.DataFrame | None = None
        self._pitcher_history: pd.DataFrame | None = None
        self._age_table: pd.DataFrame | None = None
        self._batter_xstats: pd.DataFrame | None = None  # per (player_id, season): xwoba, woba
        self._pitcher_xstats: pd.DataFrame | None = None  # per (player_id, season): fip, xfip, ip
        self._league_war_per_pa_batter: float = 0.0
        self._league_war_per_ip_pitcher: float = 0.0
        self._is_fit: bool = False

        # v2 sidecar tables
        # TJ flag: pitcher_id -> list[date] of TJ surgery dates
        self._tj_surgeries: dict[int, list[pd.Timestamp]] = {}
        self._tj_coverage_years: tuple[int | None, int | None] = (None, None)
        # Calibrated age curve: dict[age_int] -> WAR-delta from age N to N+1
        # Separate curves for batters and pitchers.
        self._age_curve_batter: dict[int, float] = {}
        self._age_curve_pitcher: dict[int, float] = {}
        self._age_curve_diagnostics: dict[str, Any] = {}
        # Role table: (player_id, season) -> "SP" | "RP"
        self._role_table: pd.DataFrame | None = None

    # ---- BaseAnalyticsModel interface --------------------------------------

    @property
    def model_name(self) -> str:
        return "projections"

    @property
    def version(self) -> str:
        # v2.0.0 introduced 2026-04-18: TJ flag, calibrated age curve, role
        # change feature.  Behaviour identical to v1.1.0 when all three v2
        # config flags default to False; flagged on via build/validate scripts.
        return "2.0.0"

    def train(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """``BaseAnalyticsModel`` adapter -- delegates to :meth:`fit`."""
        train_seasons = kwargs.get("train_seasons")
        if not train_seasons:
            raise ValueError("train() requires kwarg 'train_seasons' (list[int]).")
        return self.fit(conn, train_seasons)

    def predict(self, conn: duckdb.DuckDBPyConnection, **kwargs):
        """``BaseAnalyticsModel`` adapter -- delegates to :meth:`project`."""
        target_season = kwargs.get("target_season")
        if target_season is None:
            raise ValueError("predict() requires kwarg 'target_season' (int).")
        players = kwargs.get("players")
        return self.project(conn, target_season=target_season, players=players)

    def evaluate(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Backtest: project ``target_season`` and compare to actual WAR.

        Returns a dict of headline metrics (rmse, pearson_r, spearman_rho,
        per-cohort breakdowns).
        """
        target_season = kwargs.get("target_season")
        if target_season is None:
            raise ValueError("evaluate() requires kwarg 'target_season' (int).")
        proj = self.project(conn, target_season=target_season)
        actual = _load_actual_war(conn, target_season)
        merged = proj.merge(actual, on=["player_id", "position"], how="inner")
        return self.validate_output(_score_predictions(merged))

    # ---- Core public API ---------------------------------------------------

    def fit(self, conn: duckdb.DuckDBPyConnection, train_seasons: list[int]) -> dict:
        """Load all historical season-level stats for ``train_seasons``.

        No leakage: only seasons present in ``train_seasons`` are loaded; the
        ``project`` call later enforces that the 3-yr lookback ends strictly
        before the target season.

        Args:
            conn: Open DuckDB connection (read-only is fine).
            train_seasons: List of seasons available for training (e.g.
                [2021, 2022, 2023]).

        Returns:
            Dictionary of fit metrics: number of batter / pitcher seasons,
            league baselines, etc.
        """
        seasons = sorted(set(int(s) for s in train_seasons))
        if not seasons:
            raise ValueError("train_seasons must be non-empty.")

        logger.info("Fitting ProjectionModel on seasons: %s", seasons)

        self._train_seasons = seasons

        # ---- Load season-level stats ---------------------------------------
        self._batter_history = _load_batter_history(conn, seasons)
        self._pitcher_history = _load_pitcher_history(conn, seasons)
        logger.info(
            "Loaded %d batter-seasons, %d pitcher-seasons",
            len(self._batter_history),
            len(self._pitcher_history),
        )

        # ---- Load age table ------------------------------------------------
        self._age_table = _load_age_table()
        logger.info("Loaded %d player-season age rows", len(self._age_table))

        # ---- Compute league baselines (used for regression-to-mean) --------
        self._league_war_per_pa_batter = _league_war_per_pa(
            self._batter_history, "pa",
        )
        self._league_war_per_ip_pitcher = _league_war_per_pa(
            self._pitcher_history, "ip",
        )
        logger.info(
            "League WAR/PA (bat) = %.5f, WAR/IP (pit) = %.5f",
            self._league_war_per_pa_batter,
            self._league_war_per_ip_pitcher,
        )

        # ---- Statcast forward-indicator tables -----------------------------
        # xwOBA and wOBA for batters; FIP-proxy and xFIP-proxy for pitchers.
        # season_batting_stats.xwoba is currently 100% NULL in this DB, so we
        # derive both wOBA and xwOBA from the per-PA pitches table directly,
        # restricted to the train seasons so no future data leaks.
        self._batter_xstats = _load_batter_xstats(conn, seasons)
        self._pitcher_xstats = _load_pitcher_xstats(conn, seasons)
        logger.info(
            "Loaded xstats: %d batter rows, %d pitcher rows",
            len(self._batter_xstats),
            len(self._pitcher_xstats),
        )

        # ---- v2 sidecars ---------------------------------------------------
        # All loads are best-effort: any failure logs and degrades to v1.1
        # behaviour for that feature without breaking the fit.
        if self.config.enable_tj_flag:
            self._tj_surgeries, self._tj_coverage_years = _load_tj_surgeries()
            logger.info(
                "Loaded TJ surgeries: %d pitchers, coverage years %s",
                len(self._tj_surgeries),
                self._tj_coverage_years,
            )

        if self.config.enable_role_change:
            self._role_table = _load_role_table(conn, seasons)
            logger.info(
                "Loaded role table: %d (player_id, season) rows",
                0 if self._role_table is None else len(self._role_table),
            )

        if self.config.enable_calibrated_age_curve:
            curves, diagnostics = _calibrate_age_curve(
                conn,
                year_lo=self.config.age_curve_train_year_lo,
                year_hi=self.config.age_curve_train_year_hi,
                min_sample=self.config.age_curve_min_sample,
                smoothing_window=self.config.age_curve_smoothing_window,
            )
            self._age_curve_batter = curves.get("batter", {})
            self._age_curve_pitcher = curves.get("pitcher", {})
            self._age_curve_diagnostics = diagnostics
            logger.info(
                "Calibrated age curves: batter ages=%s, pitcher ages=%s",
                sorted(self._age_curve_batter.keys()),
                sorted(self._age_curve_pitcher.keys()),
            )

        self._is_fit = True

        metrics = {
            "n_train_seasons": len(seasons),
            "train_seasons": seasons,
            "n_batter_seasons": int(len(self._batter_history)),
            "n_pitcher_seasons": int(len(self._pitcher_history)),
            "league_war_per_pa_batter": round(self._league_war_per_pa_batter, 6),
            "league_war_per_ip_pitcher": round(self._league_war_per_ip_pitcher, 6),
            "n_batter_xstat_rows": int(len(self._batter_xstats)),
            "n_pitcher_xstat_rows": int(len(self._pitcher_xstats)),
            "v2_enable_tj_flag": bool(self.config.enable_tj_flag),
            "v2_enable_calibrated_age_curve": bool(self.config.enable_calibrated_age_curve),
            "v2_enable_role_change": bool(self.config.enable_role_change),
            "v2_n_tj_surgeries": int(sum(len(v) for v in self._tj_surgeries.values())),
            "v2_tj_coverage_years": list(self._tj_coverage_years),
            "v2_age_curve_diagnostics": self._age_curve_diagnostics,
            "v2_n_role_rows": int(0 if self._role_table is None else len(self._role_table)),
        }
        self.set_training_metadata(
            metrics=metrics,
            params={
                "weights": list(self.config.weights),
                "batter_regression_pa": self.config.batter_regression_pa,
                "pitcher_regression_ip": self.config.pitcher_regression_ip,
                "overlay_cap_war": self.config.overlay_cap_war,
            },
        )
        return metrics

    def project(
        self,
        conn: duckdb.DuckDBPyConnection,
        target_season: int,
        players: list[int] | None = None,
    ) -> pd.DataFrame:
        """Produce next-season WAR projections.

        Args:
            conn: Open DuckDB connection (used only for player-name lookup).
            target_season: Season being projected (must be > all train_seasons).
            players: Optional list of player IDs to restrict the projection.

        Returns:
            DataFrame with columns:
                player_id, name, position ('batter'|'pitcher'),
                projected_war, marcel_war, statcast_adjustment, age,
                prior_3yr_war, prior_3yr_pa_or_ip
        """
        if not self._is_fit:
            raise RuntimeError("Model not fit. Call fit() first.")

        target_season = int(target_season)
        # Leakage guard: every prior season must be strictly < target_season
        if any(s >= target_season for s in self._train_seasons):
            raise ValueError(
                f"Leakage guard: train_seasons {self._train_seasons} contain a "
                f"season >= target_season ({target_season}). Refit with priors only."
            )
        prior_seasons = [target_season - 1, target_season - 2, target_season - 3]
        # Allow projection if we have at least 1 prior season available
        available_priors = [s for s in prior_seasons if s in self._train_seasons]
        if not available_priors:
            raise ValueError(
                f"No prior seasons in train set for target {target_season}. "
                f"Need any of {prior_seasons}, have {self._train_seasons}."
            )

        bat_proj = self._project_cohort(
            history=self._batter_history,
            xstats=self._batter_xstats,
            target_season=target_season,
            position_label="batter",
            volume_col="pa",
            min_total_volume=self.config.min_total_pa_batter,
            regression_volume=self.config.batter_regression_pa,
            league_war_per_volume=self._league_war_per_pa_batter,
            overlay_fn=self._batter_overlay,
            players=players,
        )
        pit_proj = self._project_cohort(
            history=self._pitcher_history,
            xstats=self._pitcher_xstats,
            target_season=target_season,
            position_label="pitcher",
            volume_col="ip",
            min_total_volume=self.config.min_total_ip_pitcher,
            regression_volume=self.config.pitcher_regression_ip,
            league_war_per_volume=self._league_war_per_pa_batter,  # not used; pitcher uses its own
            overlay_fn=self._pitcher_overlay,
            players=players,
        )
        # Re-set pitcher league baseline correctly
        pit_proj_correct = self._project_cohort(
            history=self._pitcher_history,
            xstats=self._pitcher_xstats,
            target_season=target_season,
            position_label="pitcher",
            volume_col="ip",
            min_total_volume=self.config.min_total_ip_pitcher,
            regression_volume=self.config.pitcher_regression_ip,
            league_war_per_volume=self._league_war_per_ip_pitcher,
            overlay_fn=self._pitcher_overlay,
            players=players,
        )

        proj = pd.concat([bat_proj, pit_proj_correct], ignore_index=True)
        if proj.empty:
            return proj

        # Attach names
        names = _load_player_names(conn, proj["player_id"].tolist())
        proj = proj.merge(names, on="player_id", how="left")
        cols = [
            "player_id", "name", "position", "projected_war",
            "marcel_war", "statcast_adjustment",
            "tj_adjustment", "role_change_adjustment",
            "age", "prior_3yr_war", "prior_3yr_pa_or_ip",
        ]
        # Tolerate older columns being absent (e.g. on a v1.1-era model load).
        cols = [c for c in cols if c in proj.columns]
        return proj[cols].sort_values("projected_war", ascending=False).reset_index(drop=True)

    # ---- Internals ---------------------------------------------------------

    def _project_cohort(
        self,
        history: pd.DataFrame,
        xstats: pd.DataFrame,
        target_season: int,
        position_label: str,
        volume_col: str,
        min_total_volume: float,
        regression_volume: float,
        league_war_per_volume: float,
        overlay_fn,
        players: list[int] | None,
    ) -> pd.DataFrame:
        """Project all qualified players in ``history`` for ``target_season``."""
        priors = [target_season - 1, target_season - 2, target_season - 3]
        sub = history[history["season"].isin(priors)].copy()
        if players is not None:
            sub = sub[sub["player_id"].isin(players)]
        if sub.empty:
            return pd.DataFrame()

        weights = dict(zip(priors, self.config.weights))  # season -> weight

        rows: list[dict] = []
        for pid, group in sub.groupby("player_id"):
            group = group.set_index("season")
            war_terms = 0.0
            vol_terms = 0.0
            war_total_unweighted = 0.0
            vol_total_unweighted = 0.0
            for s in priors:
                if s not in group.index:
                    continue
                w = weights[s]
                war_s = group.at[s, "war"]
                vol_s = group.at[s, volume_col]
                if pd.isna(war_s) or pd.isna(vol_s) or vol_s <= 0:
                    continue
                war_terms += w * float(war_s)
                vol_terms += w * float(vol_s)
                war_total_unweighted += float(war_s)
                vol_total_unweighted += float(vol_s)

            if vol_terms <= 0 or vol_total_unweighted < min_total_volume:
                continue

            # ---- Regress to league baseline -----------------------------
            # Marcel: rate = (sum(w * WAR) + R * league_rate) / (sum(w * vol) + R)
            # Then re-scale by projected next-season volume (use weighted-avg
            # volume as the next-season expectation).
            league_war_term = league_war_per_volume * regression_volume
            rate = (war_terms + league_war_term) / (vol_terms + regression_volume)
            # Projected next-season volume: weighted-average prior volume
            # (this implicitly regresses workload toward the recent average,
            # which is closer to true playing time than full-season default).
            proj_volume = vol_terms / sum(weights[s] for s in priors if s in group.index)
            marcel_war = rate * proj_volume

            # ---- Age adjustment -----------------------------------------
            age = self._lookup_age(int(pid), target_season)
            age_adj = self._age_adjustment(age, position_label)
            marcel_war_aged = marcel_war + age_adj

            # ---- Statcast overlay ----------------------------------------
            overlay = overlay_fn(int(pid), priors, xstats)
            overlay_capped = float(np.clip(
                overlay, -self.config.overlay_cap_war, self.config.overlay_cap_war,
            ))

            # ---- v2: TJ flag (pitchers only) -----------------------------
            tj_adj = 0.0
            if (
                self.config.enable_tj_flag
                and position_label == "pitcher"
                and self._has_recent_tj(int(pid), target_season)
            ):
                # Floor the dampener so projection doesn't drop below the
                # current Marcel-only baseline by more than the dampener.
                # Equivalent to: never project a pitcher more than |dampener|
                # below their Marcel performance baseline.
                tj_adj = float(self.config.tj_dampener_war)

            # ---- v2: Role change (pitchers only) -------------------------
            role_adj = 0.0
            if (
                self.config.enable_role_change
                and position_label == "pitcher"
                and self._role_changed(int(pid), target_season)
            ):
                role_adj = float(np.clip(
                    self.config.role_change_war,
                    -self.config.role_change_cap_war,
                    self.config.role_change_cap_war,
                ))

            projected = marcel_war_aged + overlay_capped + tj_adj + role_adj

            rows.append({
                "player_id": int(pid),
                "position": position_label,
                "projected_war": round(float(projected), 3),
                "marcel_war": round(float(marcel_war_aged), 3),
                "statcast_adjustment": round(overlay_capped, 3),
                "tj_adjustment": round(float(tj_adj), 3),
                "role_change_adjustment": round(float(role_adj), 3),
                "age": round(float(age), 1) if age is not None else None,
                "prior_3yr_war": round(float(war_total_unweighted), 2),
                "prior_3yr_pa_or_ip": round(float(vol_total_unweighted), 1),
            })

        return pd.DataFrame(rows)

    def _lookup_age(self, player_id: int, target_season: int) -> float:
        """Return the projected age for ``player_id`` in ``target_season``.

        Uses the cached Baseball-Reference age table (``data/player_ages.parquet``).
        Falls back to most-recent-age + delta-years, then to ``config.default_age``.
        """
        if self._age_table is None or self._age_table.empty:
            return float(self.config.default_age)

        sub = self._age_table[self._age_table["player_id"] == player_id]
        if sub.empty:
            return float(self.config.default_age)

        # Exact match
        exact = sub[sub["season"] == target_season]
        if not exact.empty:
            return float(exact["age"].iloc[0])

        # Most-recent prior + season delta
        prior = sub[sub["season"] < target_season].sort_values("season")
        if not prior.empty:
            last = prior.iloc[-1]
            return float(last["age"]) + (target_season - int(last["season"]))

        # Future-only entries (rare): subtract delta
        future = sub[sub["season"] > target_season].sort_values("season")
        if not future.empty:
            first = future.iloc[0]
            return float(first["age"]) - (int(first["season"]) - target_season)

        return float(self.config.default_age)

    def _age_adjustment(self, age: float | None, position_label: str = "batter") -> float:
        """Age curve adjustment.

        v1.1 behaviour (default): piecewise-linear +0.5/yr below 27, flat
        27-31, -0.5/yr above 31, capped at +/- 2-3 WAR.

        v2 behaviour (when ``enable_calibrated_age_curve`` is True): consult
        the empirically-fit age curve (per-position) for the next-season
        delta from the player's current age.  Falls back to the canonical
        slope for any age whose calibration sample was below threshold.
        """
        if age is None:
            age = float(self.config.default_age)
        if self.config.enable_calibrated_age_curve:
            curve = (
                self._age_curve_pitcher if position_label == "pitcher"
                else self._age_curve_batter
            )
            age_int = int(round(age))
            if age_int in curve:
                delta = float(curve[age_int])
                # Mild safety cap (calibrated curve is single-year delta only)
                return float(np.clip(delta, -1.5, 1.5))
            # else fall through to canonical curve

        if age < self.config.peak_age_low:
            yrs = self.config.peak_age_low - age
            return min(self.config.age_slope_young * yrs, 2.0)
        if age > self.config.peak_age_high:
            yrs = age - self.config.peak_age_high
            return max(self.config.age_slope_old * yrs, -3.0)
        return 0.0

    # ---- v2 helpers --------------------------------------------------------

    def _has_recent_tj(self, player_id: int, target_season: int) -> bool:
        """Return True if ``player_id`` had a TJ surgery within the
        configured window before the start of ``target_season``.
        """
        surgeries = self._tj_surgeries.get(int(player_id))
        if not surgeries:
            return False
        # Season starts ~April 1.  Compare date difference against window.
        season_start = pd.Timestamp(year=int(target_season), month=4, day=1)
        window = pd.Timedelta(days=int(self.config.tj_window_months * 30.5))
        for surgery_date in surgeries:
            delta = season_start - surgery_date
            if pd.Timedelta(0) <= delta <= window:
                return True
        return False

    def _role_changed(self, player_id: int, target_season: int) -> bool:
        """Return True if ``player_id``'s most recent role differs from the
        prior season's role (SP <-> RP transition).
        """
        if self._role_table is None or self._role_table.empty:
            return False
        sub = self._role_table[self._role_table["player_id"] == int(player_id)]
        if sub.empty:
            return False
        # Look at the two most recent prior seasons strictly < target_season.
        prior = sub[sub["season"] < target_season].sort_values("season")
        if len(prior) < 2:
            return False
        last_two = prior.tail(2)
        roles = last_two["role"].tolist()
        return roles[0] != roles[1]

    def _batter_overlay(
        self,
        player_id: int,
        priors: list[int],
        xstats: pd.DataFrame,
    ) -> float:
        """Statcast overlay for a batter.

        If the prior-2-season weighted xwOBA exceeds wOBA by ``threshold``,
        the batter is a positive-regression candidate (boost). If reversed,
        dampen.

        Returns a WAR adjustment (positive boost, negative dampen). Caller
        clips to +/- ``overlay_cap_war``.
        """
        if xstats is None or xstats.empty:
            return 0.0
        last2 = [priors[0], priors[1]]  # most recent two prior seasons
        sub = xstats[(xstats["player_id"] == player_id) & (xstats["season"].isin(last2))]
        if sub.empty:
            return 0.0
        total_pa = float(sub["pa_proxy"].sum())
        if total_pa < 50:
            return 0.0
        # PA-weighted means
        sub = sub.dropna(subset=["xwoba", "woba"])
        if sub.empty:
            return 0.0
        weighted_xwoba = float((sub["xwoba"] * sub["pa_proxy"]).sum() / sub["pa_proxy"].sum())
        weighted_woba = float((sub["woba"] * sub["pa_proxy"]).sum() / sub["pa_proxy"].sum())
        gap = weighted_xwoba - weighted_woba
        if abs(gap) < self.config.batter_xwoba_threshold:
            return 0.0
        return gap * self.config.batter_overlay_war_per_woba_pt

    def _pitcher_overlay(
        self,
        player_id: int,
        priors: list[int],
        xstats: pd.DataFrame,
    ) -> float:
        """Statcast overlay for a pitcher (canonical xFIP - FIP signal).

        Per the v1.1 design (post-FIP/xFIP backfill, 2026-04-18): use the
        IP-weighted ``fip`` and ``xfip`` over the prior 2 seasons. If
        ``xfip < fip`` by >= ``pitcher_xfip_threshold`` (default 0.30), the
        pitcher allowed worse contact outcomes than the underlying skill
        suggests -> positive regression candidate -> boost. If reversed
        (``xfip > fip``), dampen. Magnitude is linear in the gap, capped at
        +/- ``overlay_cap_war`` by the caller.

        Conversion: ~0.5 WAR per 0.60 FIP-points (``pitcher_overlay_war_per_fip_pt``),
        which equals the +/- 0.5 WAR cap at a 0.6 ERA-equivalent gap. The
        slope is the literature-canonical "1 run of FIP ~ 0.1 WAR per 9 IP
        on a 150 IP workload".
        """
        if xstats is None or xstats.empty:
            return 0.0
        last2 = [priors[0], priors[1]]  # most recent two prior seasons
        sub = xstats[(xstats["player_id"] == player_id) & (xstats["season"].isin(last2))]
        if sub.empty:
            return 0.0
        sub = sub.dropna(subset=["fip", "xfip"])
        if sub.empty:
            return 0.0
        total_ip = float(sub["ip"].sum())
        if total_ip < 30.0:
            return 0.0
        # IP-weighted means
        weighted_fip = float((sub["fip"] * sub["ip"]).sum() / sub["ip"].sum())
        weighted_xfip = float((sub["xfip"] * sub["ip"]).sum() / sub["ip"].sum())
        # Sign convention: xFIP < FIP -> pitcher was unlucky on HR/FB or
        # contact -> positive regression candidate -> boost (positive gap).
        gap = weighted_fip - weighted_xfip
        if abs(gap) < self.config.pitcher_xfip_threshold:
            return 0.0
        return gap * self.config.pitcher_overlay_war_per_fip_pt

    # ---- Persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        """Persist model + sidecar metadata JSON via base class joblib path."""
        super().save(Path(path))

    @classmethod
    def load(cls, path: Path) -> "ProjectionModel":
        return super().load(Path(path))


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_batter_history(
    conn: duckdb.DuckDBPyConnection,
    seasons: list[int],
) -> pd.DataFrame:
    """Pull (player_id, season, pa, war) for every batter-season in ``seasons``."""
    placeholder = ",".join(str(int(s)) for s in seasons)
    df = conn.execute(f"""
        SELECT player_id, season, pa, war
        FROM season_batting_stats
        WHERE season IN ({placeholder})
          AND pa IS NOT NULL AND pa > 0
          AND war IS NOT NULL
    """).fetchdf()
    df["player_id"] = df["player_id"].astype(int)
    df["season"] = df["season"].astype(int)
    return df


def _load_pitcher_history(
    conn: duckdb.DuckDBPyConnection,
    seasons: list[int],
) -> pd.DataFrame:
    """Pull (player_id, season, ip, war) for every pitcher-season in ``seasons``."""
    placeholder = ",".join(str(int(s)) for s in seasons)
    df = conn.execute(f"""
        SELECT player_id, season, ip, war
        FROM season_pitching_stats
        WHERE season IN ({placeholder})
          AND ip IS NOT NULL AND ip > 0
          AND war IS NOT NULL
    """).fetchdf()
    df["player_id"] = df["player_id"].astype(int)
    df["season"] = df["season"].astype(int)
    return df


def _load_age_table() -> pd.DataFrame:
    """Load the cached (player_id, season, age) parquet built from bWAR."""
    path = DEFAULT_AGE_CACHE
    if not path.exists():
        logger.warning("Age cache not found at %s; ages will fall back to default.", path)
        return pd.DataFrame(columns=["player_id", "season", "age"])
    df = pd.read_parquet(path)
    df["player_id"] = df["player_id"].astype(int)
    df["season"] = df["season"].astype(int)
    df["age"] = df["age"].astype(float)
    return df


def _load_batter_xstats(
    conn: duckdb.DuckDBPyConnection,
    seasons: list[int],
) -> pd.DataFrame:
    """Per (player_id, season) wOBA and xwOBA derived from the pitches table.

    Restricted to ``seasons`` so the overlay sees no future data.
    """
    placeholder = ",".join(str(int(s)) for s in seasons)
    df = conn.execute(f"""
        SELECT
            batter_id            AS player_id,
            EXTRACT(YEAR FROM game_date)::INTEGER AS season,
            SUM(woba_value)      AS sum_woba_value,
            SUM(woba_denom)      AS sum_woba_denom,
            SUM(estimated_woba * COALESCE(woba_denom, 1))
                                 AS sum_xwoba_weighted,
            SUM(CASE WHEN estimated_woba IS NOT NULL THEN COALESCE(woba_denom, 1) END)
                                 AS sum_xwoba_denom,
            COUNT(*)             AS pa_proxy
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) IN ({placeholder})
          AND woba_denom > 0
        GROUP BY batter_id, season
    """).fetchdf()
    if df.empty:
        return pd.DataFrame(columns=[
            "player_id", "season", "woba", "xwoba", "pa_proxy",
        ])
    df["woba"] = df["sum_woba_value"] / df["sum_woba_denom"].replace(0, np.nan)
    df["xwoba"] = df["sum_xwoba_weighted"] / df["sum_xwoba_denom"].replace(0, np.nan)
    df["player_id"] = df["player_id"].astype(int)
    df["season"] = df["season"].astype(int)
    return df[["player_id", "season", "woba", "xwoba", "pa_proxy"]]


def _load_pitcher_xstats(
    conn: duckdb.DuckDBPyConnection,
    seasons: list[int],
) -> pd.DataFrame:
    """Per (player_id, season) FIP / xFIP from ``season_pitching_stats``.

    As of the FIP/xFIP backfill (2026-04-18), ``season_pitching_stats.fip`` and
    ``.xfip`` are populated for ~97% of pitcher-seasons, so the pitcher overlay
    can use the canonical signal directly. Earlier versions of this loader
    derived a wOBA-against / xwOBA-against substitute from the ``pitches``
    table because both columns were 100% NULL; that fallback is retained at
    the call site (``_pitcher_overlay``) for any pitcher-season missing FIP/xFIP.

    Returns a frame with columns:
        player_id, season, fip, xfip, ip
    """
    placeholder = ",".join(str(int(s)) for s in seasons)
    df = conn.execute(f"""
        SELECT player_id, season, fip, xfip, ip
        FROM season_pitching_stats
        WHERE season IN ({placeholder})
          AND ip IS NOT NULL AND ip > 0
          AND (fip IS NOT NULL OR xfip IS NOT NULL)
    """).fetchdf()
    if df.empty:
        return pd.DataFrame(columns=["player_id", "season", "fip", "xfip", "ip"])
    df["player_id"] = df["player_id"].astype(int)
    df["season"] = df["season"].astype(int)
    return df[["player_id", "season", "fip", "xfip", "ip"]]


def _load_player_names(
    conn: duckdb.DuckDBPyConnection,
    player_ids: list[int],
) -> pd.DataFrame:
    """Look up full names for a list of player IDs."""
    if not player_ids:
        return pd.DataFrame(columns=["player_id", "name"])
    ids = ",".join(str(int(p)) for p in set(player_ids))
    df = conn.execute(f"""
        SELECT player_id, full_name AS name
        FROM players
        WHERE player_id IN ({ids})
    """).fetchdf()
    df["player_id"] = df["player_id"].astype(int)
    return df


def _load_actual_war(
    conn: duckdb.DuckDBPyConnection,
    season: int,
) -> pd.DataFrame:
    """Load actual WAR for ``season`` (batters and pitchers)."""
    bat = conn.execute(f"""
        SELECT player_id, war AS actual_war, pa AS actual_pa
        FROM season_batting_stats
        WHERE season = {int(season)} AND war IS NOT NULL
    """).fetchdf()
    bat["position"] = "batter"
    pit = conn.execute(f"""
        SELECT player_id, war AS actual_war, ip AS actual_ip
        FROM season_pitching_stats
        WHERE season = {int(season)} AND war IS NOT NULL
    """).fetchdf()
    pit["position"] = "pitcher"
    out = pd.concat([
        bat[["player_id", "position", "actual_war"]],
        pit[["player_id", "position", "actual_war"]],
    ], ignore_index=True)
    out["player_id"] = out["player_id"].astype(int)
    return out


# ---------------------------------------------------------------------------
# v2 sidecar loaders
# ---------------------------------------------------------------------------

def _load_tj_surgeries(
    path: Path = DEFAULT_INJURY_LABELS,
) -> tuple[dict[int, list[pd.Timestamp]], tuple[int | None, int | None]]:
    """Load TJ surgery dates from ``data/injury_labels.parquet``.

    Returns:
        (mapping, coverage_years)
        mapping: pitcher_id -> list of TJ surgery dates (one entry per IL
            placement classified as ``tommy_john``).
        coverage_years: (min_year, max_year) of the injury_labels parquet,
            documenting the data coverage limit.
    """
    if not path.exists():
        logger.warning("TJ surgery file not found at %s; TJ flag disabled.", path)
        return {}, (None, None)
    df = pd.read_parquet(path)
    if df.empty or "injury_type" not in df.columns:
        return {}, (None, None)
    coverage = (int(df["season"].min()), int(df["season"].max()))
    tj = df[df["injury_type"] == "tommy_john"].dropna(subset=["pitcher_id", "il_date"])
    out: dict[int, list[pd.Timestamp]] = {}
    for _, row in tj.iterrows():
        pid = int(row["pitcher_id"])
        out.setdefault(pid, []).append(pd.Timestamp(row["il_date"]))
    return out, coverage


def _load_role_table(
    conn: duckdb.DuckDBPyConnection,
    seasons: list[int],
) -> pd.DataFrame:
    """Load a per-(player_id, season) role label (SP / RP) for the train seasons.

    SP = ``gs >= 0.5 * g``, else RP.  Only includes pitchers with at least
    one game played.
    """
    placeholder = ",".join(str(int(s)) for s in seasons)
    df = conn.execute(f"""
        SELECT player_id, season, g, gs
        FROM season_pitching_stats
        WHERE season IN ({placeholder})
          AND g IS NOT NULL AND g > 0
    """).fetchdf()
    if df.empty:
        return pd.DataFrame(columns=["player_id", "season", "role"])
    df["player_id"] = df["player_id"].astype(int)
    df["season"] = df["season"].astype(int)
    df["gs"] = df["gs"].fillna(0).astype(int)
    df["role"] = np.where(df["gs"] >= 0.5 * df["g"], "SP", "RP")
    return df[["player_id", "season", "role"]]


def _calibrate_age_curve(
    conn: duckdb.DuckDBPyConnection,
    year_lo: int,
    year_hi: int,
    min_sample: int,
    smoothing_window: int,
) -> tuple[dict[str, dict[int, float]], dict[str, Any]]:
    """Compute empirical age curves: mean WAR-delta from age N to N+1.

    Pulls every (player_id, season, war, vol) from the batting and pitching
    stat tables in [year_lo, year_hi], joins to the age table, and computes
    the mean change in WAR for every (age, age+1) consecutive-season pair
    where both seasons cleared the qualifier (>= 100 PA / >= 50 IP).

    Smoothes the per-age means with a centered moving average of width
    ``smoothing_window``.  Ages with sample size below ``min_sample`` are
    left out of the returned dict (caller falls back to canonical slope).
    """
    age_table = _load_age_table()
    if age_table.empty:
        return ({"batter": {}, "pitcher": {}}, {"note": "age_table empty"})

    placeholder = ",".join(str(y) for y in range(int(year_lo), int(year_hi) + 1))
    bat = conn.execute(f"""
        SELECT player_id, season, war, pa
        FROM season_batting_stats
        WHERE season IN ({placeholder})
          AND war IS NOT NULL AND pa >= 100
    """).fetchdf()
    pit = conn.execute(f"""
        SELECT player_id, season, war, ip
        FROM season_pitching_stats
        WHERE season IN ({placeholder})
          AND war IS NOT NULL AND ip >= 50
    """).fetchdf()

    diagnostics: dict[str, Any] = {
        "year_range": [int(year_lo), int(year_hi)],
        "min_sample_per_age": int(min_sample),
        "smoothing_window": int(smoothing_window),
        "batter_pairs_per_age": {},
        "pitcher_pairs_per_age": {},
        "fallback_ages_batter": [],
        "fallback_ages_pitcher": [],
    }

    def _curve_for(df: pd.DataFrame, label: str) -> dict[int, float]:
        if df.empty:
            return {}
        df = df.merge(
            age_table[["player_id", "season", "age"]],
            on=["player_id", "season"], how="inner",
        )
        if df.empty:
            return {}
        # Build (age_n, war_n, war_n+1) triples
        df = df.sort_values(["player_id", "season"])
        df["next_war"] = df.groupby("player_id")["war"].shift(-1)
        df["next_season"] = df.groupby("player_id")["season"].shift(-1)
        consec = df[df["next_season"] == df["season"] + 1].copy()
        consec["delta_war"] = consec["next_war"] - consec["war"]
        consec["age_int"] = consec["age"].round().astype(int)
        per_age = consec.groupby("age_int").agg(
            mean_delta=("delta_war", "mean"),
            n=("delta_war", "size"),
        ).reset_index()
        # Persist sample counts for diagnostics
        sample_dict = {int(r.age_int): int(r.n) for r in per_age.itertuples()}
        if label == "batter":
            diagnostics["batter_pairs_per_age"] = sample_dict
        else:
            diagnostics["pitcher_pairs_per_age"] = sample_dict
        # Smoothing: rolling mean across age axis (sorted)
        per_age = per_age.sort_values("age_int").reset_index(drop=True)
        per_age["smoothed"] = (
            per_age["mean_delta"]
            .rolling(window=smoothing_window, min_periods=1, center=True)
            .mean()
        )
        # Filter by sample size
        passed = per_age[per_age["n"] >= min_sample]
        out = {int(r.age_int): float(r.smoothed) for r in passed.itertuples()}
        fallback = sorted(set(per_age["age_int"]) - set(out.keys()))
        if label == "batter":
            diagnostics["fallback_ages_batter"] = [int(a) for a in fallback]
        else:
            diagnostics["fallback_ages_pitcher"] = [int(a) for a in fallback]
        return out

    bat_curve = _curve_for(bat, "batter")
    pit_curve = _curve_for(pit, "pitcher")
    diagnostics["batter_curve"] = {str(k): round(v, 4) for k, v in sorted(bat_curve.items())}
    diagnostics["pitcher_curve"] = {str(k): round(v, 4) for k, v in sorted(pit_curve.items())}
    return ({"batter": bat_curve, "pitcher": pit_curve}, diagnostics)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _league_war_per_pa(history: pd.DataFrame, vol_col: str) -> float:
    """League average WAR per unit volume (PA or IP) from ``history``."""
    if history is None or history.empty:
        return 0.0
    total_war = float(history["war"].sum())
    total_vol = float(history[vol_col].sum())
    if total_vol <= 0:
        return 0.0
    return total_war / total_vol


def _score_predictions(merged: pd.DataFrame) -> dict:
    """Compute headline metrics on a (projected_war, actual_war) frame."""
    out: dict[str, Any] = {}
    if merged.empty:
        return {"n": 0}
    for label, sub in [
        ("combined", merged),
        ("batters", merged[merged["position"] == "batter"]),
        ("pitchers", merged[merged["position"] == "pitcher"]),
    ]:
        if sub.empty:
            out[label] = {"n": 0}
            continue
        proj = sub["projected_war"].astype(float).to_numpy()
        actual = sub["actual_war"].astype(float).to_numpy()
        marcel = sub["marcel_war"].astype(float).to_numpy()
        rmse = float(np.sqrt(np.mean((proj - actual) ** 2)))
        rmse_marcel = float(np.sqrt(np.mean((marcel - actual) ** 2)))
        mae = float(np.mean(np.abs(proj - actual)))
        # Pearson r
        if len(sub) >= 3 and np.std(proj) > 0 and np.std(actual) > 0:
            pearson = float(np.corrcoef(proj, actual)[0, 1])
            spearman = float(pd.Series(proj).corr(pd.Series(actual), method="spearman"))
            pearson_marcel = float(np.corrcoef(marcel, actual)[0, 1])
            spearman_marcel = float(pd.Series(marcel).corr(pd.Series(actual), method="spearman"))
        else:
            pearson = spearman = pearson_marcel = spearman_marcel = float("nan")
        out[label] = {
            "n": int(len(sub)),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "pearson_r": round(pearson, 4),
            "spearman_rho": round(spearman, 4),
            "rmse_marcel": round(rmse_marcel, 4),
            "rmse_delta_vs_marcel": round(rmse - rmse_marcel, 4),
            "pearson_marcel": round(pearson_marcel, 4),
            "spearman_marcel": round(spearman_marcel, 4),
        }
    # Calibration: hit rate at >=3 WAR projection -> >=2.5 actual
    pred_3plus = merged[merged["projected_war"] >= 3.0]
    if len(pred_3plus) > 0:
        hit_rate = float((pred_3plus["actual_war"] >= 2.5).mean())
        out["calibration_3war_to_2p5_actual"] = {
            "n_projected_geq_3": int(len(pred_3plus)),
            "hit_rate": round(hit_rate, 4),
            "mean_actual_war": round(float(pred_3plus["actual_war"].mean()), 3),
        }
    else:
        out["calibration_3war_to_2p5_actual"] = {
            "n_projected_geq_3": 0, "hit_rate": None, "mean_actual_war": None,
        }
    return out


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

def fit_and_project(
    conn: duckdb.DuckDBPyConnection,
    train_seasons: list[int],
    target_season: int,
    config: ProjectionConfig | None = None,
) -> tuple[ProjectionModel, pd.DataFrame]:
    """Fit a model and project ``target_season`` in one call.

    Returns ``(model, projections_df)``.
    """
    model = ProjectionModel(config=config)
    model.fit(conn, train_seasons)
    proj = model.project(conn, target_season)
    return model, proj
