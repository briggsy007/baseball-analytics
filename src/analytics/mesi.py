"""
Motor Engram Stability Index (MESI) model.

Applies motor learning theory to quantify pitch execution reliability.
A pitcher with a high MESI produces consistent execution vectors across
game contexts — indicating deeply ingrained "motor engrams" for each pitch.

The model measures:
  - **Signal-to-Noise Ratio (SNR)**: How tightly clustered a pitcher's
    execution vectors are in 7-dimensional physical space.
  - **Context Stability (CS)**: Whether that consistency holds across
    leverage, fatigue, count, and times-through-order contexts.
  - **MESI score**: SNR * CS, normalised to a 0-100 population scale.
  - **Learning curve**: Power-law fit of MESI over career appearances
    to estimate ceiling and developmental stage.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from src.analytics.base import BaseAnalyticsModel

logger = logging.getLogger(__name__)

# ── Execution vector columns ─────────────────────────────────────────────
EXECUTION_COLS: list[str] = [
    "release_speed",
    "release_spin_rate",
    "pfx_x",
    "pfx_z",
    "release_pos_x",
    "release_pos_z",
    "release_extension",
]

# Minimum sample sizes for valid computation
MIN_PITCHES_SNR: int = 100
MIN_PITCHES_CONTEXT: int = 50
MIN_PITCHES_QUALIFY: int = 200

# Rolling window size for SNR
DEFAULT_WINDOW: int = 100


# ─────────────────────────────────────────────────────────────────────────────
# Helper: power-law learning curve
# ─────────────────────────────────────────────────────────────────────────────

def _learning_curve_func(t: np.ndarray, mesi_max: float, rate: float) -> np.ndarray:
    """Power-law learning model: MESI(t) = mesi_max * (1 - exp(-rate * t))."""
    return mesi_max * (1.0 - np.exp(-rate * t))


# ─────────────────────────────────────────────────────────────────────────────
# Core computation functions
# ─────────────────────────────────────────────────────────────────────────────


def compute_execution_vectors(
    pitches_df: pd.DataFrame,
    pitch_type: str,
) -> pd.DataFrame:
    """Extract execution vectors for a given pitch type.

    Filters to rows matching *pitch_type* and returns a DataFrame
    containing only the 7 execution-vector dimensions, dropping any
    rows with missing values.

    Args:
        pitches_df: Full pitch-level DataFrame (must contain EXECUTION_COLS
                    and ``pitch_type``).
        pitch_type: Statcast pitch-type code (e.g. ``"FF"``).

    Returns:
        DataFrame of shape (n, 7) with execution-vector columns.
    """
    filtered = pitches_df[pitches_df["pitch_type"] == pitch_type].copy()
    available = [c for c in EXECUTION_COLS if c in filtered.columns]
    if not available:
        return pd.DataFrame(columns=EXECUTION_COLS)
    result = filtered[available].dropna()
    return result.reset_index(drop=True)


def compute_snr(
    execution_vectors: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
) -> pd.Series:
    """Compute rolling Signal-to-Noise Ratio over execution vectors.

    For each rolling window of *window* pitches:
      - mu = mean vector (7-d)
      - Sigma = covariance matrix (7x7)
      - SNR = ||mu|| / sqrt(trace(Sigma))

    Args:
        execution_vectors: DataFrame of shape (n, 7).
        window: Rolling window size.

    Returns:
        Series of SNR values (one per window position), with NaN for
        positions where the window is not yet full.
    """
    n = len(execution_vectors)
    if n < window:
        # Not enough data; return a single SNR for the whole sample
        if n == 0:
            return pd.Series(dtype=float)
        vals = execution_vectors.values.astype(float)
        mu = vals.mean(axis=0)
        cov = np.cov(vals, rowvar=False)
        if cov.ndim == 0:
            # Only one dimension or degenerate
            trace_val = float(cov)
        else:
            trace_val = float(np.trace(cov))
        noise = np.sqrt(max(trace_val, 1e-12))
        snr_val = float(np.linalg.norm(mu)) / noise
        return pd.Series([snr_val])

    snr_values = []
    for i in range(n - window + 1):
        chunk = execution_vectors.iloc[i : i + window].values.astype(float)
        mu = chunk.mean(axis=0)
        cov = np.cov(chunk, rowvar=False)
        trace_val = float(np.trace(cov))
        noise = np.sqrt(max(trace_val, 1e-12))
        snr_val = float(np.linalg.norm(mu)) / noise
        snr_values.append(snr_val)

    # Pad front with NaN so the series aligns with the original index
    padding = [np.nan] * (window - 1)
    return pd.Series(padding + snr_values)


def _compute_snr_for_subset(df_subset: pd.DataFrame) -> float | None:
    """Compute a single SNR value for a subset of execution vectors.

    Returns None if the subset has fewer than 10 pitches.
    """
    available = [c for c in EXECUTION_COLS if c in df_subset.columns]
    vals = df_subset[available].dropna().values.astype(float)
    if len(vals) < 10:
        return None
    mu = vals.mean(axis=0)
    cov = np.cov(vals, rowvar=False)
    if cov.ndim == 0:
        trace_val = float(cov)
    else:
        trace_val = float(np.trace(cov))
    noise = np.sqrt(max(trace_val, 1e-12))
    return float(np.linalg.norm(mu)) / noise


def compute_context_stability(
    conn,
    pitcher_id: int,
    pitch_type: str,
) -> dict:
    """Compute Context Stability score for a pitcher/pitch-type.

    Splits pitches into 4 context pairs and computes SNR in each
    context bucket. CS = 1 - CV(SNR across contexts), where CV is
    the coefficient of variation.

    Contexts:
      1. Low leverage (inning 1-3) vs high leverage (inning 7+)
      2. First time through order (at_bat_number <= 9) vs third (>= 19)
      3. Ahead in count vs behind in count
      4. Fresh (pitch 1-50 in game) vs fatigued (pitch 80+)

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        pitch_type: Pitch type code.

    Returns:
        Dictionary with ``cs_score`` and ``context_snr`` breakdown.
    """
    exec_cols = ", ".join(EXECUTION_COLS)

    # Fetch all pitches with context columns
    query = f"""
        SELECT
            {exec_cols},
            inning,
            at_bat_number,
            balls,
            strikes,
            ROW_NUMBER() OVER (
                PARTITION BY game_pk
                ORDER BY at_bat_number, pitch_number
            ) AS pitch_num_in_game
        FROM pitches
        WHERE pitcher_id = $1
          AND pitch_type = $2
          AND release_speed IS NOT NULL
          AND release_spin_rate IS NOT NULL
          AND pfx_x IS NOT NULL
          AND pfx_z IS NOT NULL
    """
    df = conn.execute(query, [pitcher_id, pitch_type]).fetchdf()

    if len(df) < MIN_PITCHES_CONTEXT:
        return {"cs_score": None, "context_snr": {}}

    context_snrs: dict[str, float | None] = {}

    # Context 1: Low leverage (inning 1-3)
    low_lev = df[df["inning"] <= 3]
    context_snrs["low_leverage"] = _compute_snr_for_subset(low_lev)

    # Context 1b: High leverage (inning 7+)
    high_lev = df[df["inning"] >= 7]
    context_snrs["high_leverage"] = _compute_snr_for_subset(high_lev)

    # Context 2: First time through order (at_bat_number roughly 1-9)
    first_tto = df[df["at_bat_number"] <= 9]
    context_snrs["first_time_through"] = _compute_snr_for_subset(first_tto)

    # Context 2b: Third time through order (at_bat_number >= 19)
    third_tto = df[df["at_bat_number"] >= 19]
    context_snrs["third_time_through"] = _compute_snr_for_subset(third_tto)

    # Context 3: Ahead in count (0-2, 1-2)
    ahead_mask = ((df["balls"] == 0) & (df["strikes"] == 2)) | (
        (df["balls"] == 1) & (df["strikes"] == 2)
    )
    context_snrs["ahead_in_count"] = _compute_snr_for_subset(df[ahead_mask])

    # Context 3b: Behind in count (2-0, 3-0, 3-1)
    behind_mask = (
        ((df["balls"] == 2) & (df["strikes"] == 0))
        | ((df["balls"] == 3) & (df["strikes"] == 0))
        | ((df["balls"] == 3) & (df["strikes"] == 1))
    )
    context_snrs["behind_in_count"] = _compute_snr_for_subset(df[behind_mask])

    # Context 4: Fresh (pitch 1-50 in game)
    fresh = df[df["pitch_num_in_game"] <= 50]
    context_snrs["fresh"] = _compute_snr_for_subset(fresh)

    # Context 4b: Fatigued (pitch 80+ in game)
    fatigued = df[df["pitch_num_in_game"] >= 80]
    context_snrs["fatigued"] = _compute_snr_for_subset(fatigued)

    # Collect all valid SNR values
    valid_snrs = [v for v in context_snrs.values() if v is not None]

    if len(valid_snrs) < 2:
        # Can't compute CV with fewer than 2 values
        cs_score = 1.0 if len(valid_snrs) == 1 else None
    else:
        arr = np.array(valid_snrs)
        mean_snr = arr.mean()
        if mean_snr == 0:
            cs_score = 0.0
        else:
            cv = float(arr.std() / mean_snr)
            cs_score = max(0.0, 1.0 - cv)

    return {
        "cs_score": cs_score,
        "context_snr": context_snrs,
    }


def calculate_mesi(
    conn,
    pitcher_id: int,
    season: Optional[int] = None,
) -> dict:
    """Calculate the full MESI profile for a pitcher.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        season: Optional season filter.

    Returns:
        Dictionary with per-pitch-type MESI, CS breakdown,
        and overall MESI score.
    """
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    params: list = [pitcher_id]
    if season:
        params.append(season)

    exec_cols = ", ".join(EXECUTION_COLS)
    query = f"""
        SELECT
            pitch_type,
            {exec_cols}
        FROM pitches
        WHERE pitcher_id = $1
          AND pitch_type IS NOT NULL
          AND release_speed IS NOT NULL
          AND release_spin_rate IS NOT NULL
          AND pfx_x IS NOT NULL
          AND pfx_z IS NOT NULL
          {season_filter}
    """
    df = conn.execute(query, params).fetchdf()

    if df.empty:
        return {
            "pitcher_id": pitcher_id,
            "per_pitch_type": {},
            "overall_mesi": None,
            "overall_snr": None,
            "overall_cs": None,
        }

    # Per pitch type
    per_pt: dict = {}
    mesi_scores: list[float] = []
    pitch_counts: list[int] = []

    for pt, grp in df.groupby("pitch_type"):
        if len(grp) < MIN_PITCHES_SNR:
            continue

        # Execution vectors
        ev = grp[EXECUTION_COLS].dropna()
        if len(ev) < MIN_PITCHES_SNR:
            continue

        # Overall SNR for this pitch type
        vals = ev.values.astype(float)
        mu = vals.mean(axis=0)
        cov = np.cov(vals, rowvar=False)
        trace_val = float(np.trace(cov))
        noise = np.sqrt(max(trace_val, 1e-12))
        snr = float(np.linalg.norm(mu)) / noise

        # Context stability
        cs_result = compute_context_stability(conn, pitcher_id, pt)
        cs_score = cs_result["cs_score"]

        # Raw MESI = SNR * CS
        if cs_score is not None:
            raw_mesi = snr * cs_score
        else:
            raw_mesi = snr  # fallback: assume CS = 1

        per_pt[pt] = {
            "snr": round(snr, 3),
            "cs_score": round(cs_score, 3) if cs_score is not None else None,
            "raw_mesi": round(raw_mesi, 3),
            "context_snr": cs_result["context_snr"],
            "n_pitches": len(ev),
        }
        mesi_scores.append(raw_mesi)
        pitch_counts.append(len(ev))

    if not mesi_scores:
        return {
            "pitcher_id": pitcher_id,
            "per_pitch_type": per_pt,
            "overall_mesi": None,
            "overall_snr": None,
            "overall_cs": None,
        }

    # Weighted average across pitch types
    weights = np.array(pitch_counts, dtype=float)
    weights /= weights.sum()
    overall_raw = float(np.average(mesi_scores, weights=weights))

    # Collect summary stats
    snr_values = [per_pt[pt]["snr"] for pt in per_pt]
    cs_values = [per_pt[pt]["cs_score"] for pt in per_pt if per_pt[pt]["cs_score"] is not None]

    overall_snr = float(np.mean(snr_values)) if snr_values else None
    overall_cs = float(np.mean(cs_values)) if cs_values else None

    return {
        "pitcher_id": pitcher_id,
        "per_pitch_type": per_pt,
        "overall_mesi": round(overall_raw, 3),
        "overall_snr": round(overall_snr, 3) if overall_snr is not None else None,
        "overall_cs": round(overall_cs, 3) if overall_cs is not None else None,
    }


def fit_learning_curve(
    conn,
    pitcher_id: int,
    pitch_type: str,
) -> dict:
    """Fit a power-law learning curve to MESI over career appearances.

    Partitions the pitcher's career pitches (of the given type) into
    appearance-level chunks, computes a rolling SNR for each appearance
    block, then fits ``MESI(t) = MESI_max * (1 - exp(-r * t))``.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        pitch_type: Pitch type code.

    Returns:
        Dictionary with ``learning_rate``, ``ceiling``,
        ``current_stage``, ``trajectory`` (list of (t, mesi) pairs),
        and fit quality ``r_squared``.
    """
    exec_cols = ", ".join(EXECUTION_COLS)
    query = f"""
        SELECT
            game_pk,
            game_date,
            {exec_cols}
        FROM pitches
        WHERE pitcher_id = $1
          AND pitch_type = $2
          AND release_speed IS NOT NULL
          AND release_spin_rate IS NOT NULL
          AND pfx_x IS NOT NULL
          AND pfx_z IS NOT NULL
        ORDER BY game_date, at_bat_number, pitch_number
    """
    df = conn.execute(query, [pitcher_id, pitch_type]).fetchdf()

    if len(df) < MIN_PITCHES_SNR:
        return {
            "learning_rate": None,
            "ceiling": None,
            "current_stage": "insufficient_data",
            "trajectory": [],
            "r_squared": None,
        }

    # Group by game appearance
    appearances = []
    for game_pk, grp in df.groupby("game_pk", sort=False):
        ev = grp[EXECUTION_COLS].dropna()
        if len(ev) >= 5:
            appearances.append(ev)

    if len(appearances) < 3:
        return {
            "learning_rate": None,
            "ceiling": None,
            "current_stage": "insufficient_data",
            "trajectory": [],
            "r_squared": None,
        }

    # Compute cumulative rolling SNR across appearances
    # Use expanding windows: first N appearances cumulated
    trajectory = []
    all_vecs = pd.DataFrame(columns=EXECUTION_COLS)
    for i, app_ev in enumerate(appearances):
        all_vecs = pd.concat([all_vecs, app_ev], ignore_index=True)
        if len(all_vecs) >= 20:
            vals = all_vecs.values.astype(float)
            mu = vals.mean(axis=0)
            cov = np.cov(vals, rowvar=False)
            trace_val = float(np.trace(cov))
            noise = np.sqrt(max(trace_val, 1e-12))
            snr_val = float(np.linalg.norm(mu)) / noise
            trajectory.append((i + 1, snr_val))

    if len(trajectory) < 3:
        return {
            "learning_rate": None,
            "ceiling": None,
            "current_stage": "insufficient_data",
            "trajectory": trajectory,
            "r_squared": None,
        }

    t_arr = np.array([p[0] for p in trajectory], dtype=float)
    mesi_arr = np.array([p[1] for p in trajectory], dtype=float)

    # Fit the learning curve
    try:
        popt, _ = curve_fit(
            _learning_curve_func,
            t_arr,
            mesi_arr,
            p0=[mesi_arr.max() * 1.1, 0.05],
            bounds=([0, 1e-6], [mesi_arr.max() * 5, 10.0]),
            maxfev=5000,
        )
        mesi_max, rate = popt

        # R-squared
        predicted = _learning_curve_func(t_arr, mesi_max, rate)
        ss_res = np.sum((mesi_arr - predicted) ** 2)
        ss_tot = np.sum((mesi_arr - mesi_arr.mean()) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    except (RuntimeError, ValueError):
        # curve_fit failed — use simple estimates
        mesi_max = float(mesi_arr.max())
        rate = 0.0
        r_squared = None

    # Determine developmental stage
    if len(trajectory) == 0:
        stage = "insufficient_data"
    else:
        current_mesi = trajectory[-1][1]
        if mesi_max > 0 and rate > 0:
            pct_ceiling = current_mesi / mesi_max
            if pct_ceiling < 0.5:
                stage = "developing"
            elif pct_ceiling < 0.85:
                stage = "emerging"
            elif pct_ceiling < 0.95:
                stage = "established"
            else:
                stage = "elite_stable"
        else:
            stage = "unknown"

    return {
        "learning_rate": round(float(rate), 6) if rate else None,
        "ceiling": round(float(mesi_max), 3),
        "current_stage": stage,
        "trajectory": [(int(t), round(float(m), 3)) for t, m in trajectory],
        "r_squared": round(float(r_squared), 4) if r_squared is not None else None,
    }


def batch_calculate(
    conn,
    season: Optional[int] = None,
    min_pitches: int = MIN_PITCHES_QUALIFY,
) -> pd.DataFrame:
    """Calculate MESI for all qualifying pitchers and return a leaderboard.

    Args:
        conn: Open DuckDB connection.
        season: Optional season filter.
        min_pitches: Minimum total pitches to qualify.

    Returns:
        DataFrame with columns: pitcher_id, name, overall_mesi,
        avg_snr, avg_cs, n_pitch_types, and MESI rank.
    """
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $1" if season else ""
    params: list = []
    if season:
        params.append(season)

    # Find qualifying pitchers
    query = f"""
        SELECT pitcher_id, COUNT(*) AS n_pitches
        FROM pitches
        WHERE pitch_type IS NOT NULL
          AND release_speed IS NOT NULL
          {season_filter}
        GROUP BY pitcher_id
        HAVING COUNT(*) >= {min_pitches}
    """
    pitcher_df = conn.execute(query, params).fetchdf()

    if pitcher_df.empty:
        return pd.DataFrame(
            columns=["pitcher_id", "name", "overall_mesi", "avg_snr", "avg_cs", "n_pitch_types"]
        )

    rows = []
    raw_mesi_values = []

    for _, row in pitcher_df.iterrows():
        pid = int(row["pitcher_id"])
        result = calculate_mesi(conn, pid, season=season)
        if result["overall_mesi"] is not None:
            rows.append({
                "pitcher_id": pid,
                "overall_mesi_raw": result["overall_mesi"],
                "avg_snr": result["overall_snr"],
                "avg_cs": result["overall_cs"],
                "n_pitch_types": len(result["per_pitch_type"]),
            })
            raw_mesi_values.append(result["overall_mesi"])

    if not rows:
        return pd.DataFrame(
            columns=["pitcher_id", "name", "overall_mesi", "avg_snr", "avg_cs", "n_pitch_types"]
        )

    leaderboard = pd.DataFrame(rows)

    # Normalise MESI to 0-100 scale within population
    raw = np.array(raw_mesi_values)
    min_raw = raw.min()
    max_raw = raw.max()
    if max_raw > min_raw:
        leaderboard["overall_mesi"] = (
            (leaderboard["overall_mesi_raw"] - min_raw) / (max_raw - min_raw) * 100.0
        ).round(1)
    else:
        leaderboard["overall_mesi"] = 50.0

    leaderboard = leaderboard.drop(columns=["overall_mesi_raw"])

    # Round numeric columns
    for col in ["avg_snr", "avg_cs"]:
        if col in leaderboard.columns:
            leaderboard[col] = leaderboard[col].round(3)

    # Join player names
    try:
        names_df = conn.execute(
            "SELECT player_id AS pitcher_id, full_name AS name FROM players"
        ).fetchdf()
        leaderboard = leaderboard.merge(names_df, on="pitcher_id", how="left")
    except Exception:
        leaderboard["name"] = None

    # Reorder and sort
    front_cols = ["pitcher_id", "name", "overall_mesi", "avg_snr", "avg_cs", "n_pitch_types"]
    leaderboard = leaderboard[[c for c in front_cols if c in leaderboard.columns]]
    leaderboard = leaderboard.sort_values("overall_mesi", ascending=False).reset_index(drop=True)

    return leaderboard


def get_arsenal_stability(
    conn,
    pitcher_id: int,
) -> dict:
    """Return a full arsenal MESI profile for a pitcher.

    Combines per-pitch-type MESI, context breakdowns, and learning
    curves into a single scouting-ready profile.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.

    Returns:
        Dictionary with overall MESI, per-pitch-type detail,
        scouting insights, and learning curves.
    """
    mesi_result = calculate_mesi(conn, pitcher_id)

    if mesi_result["overall_mesi"] is None:
        return {
            "pitcher_id": pitcher_id,
            "overall_mesi": None,
            "per_pitch_type": {},
            "best_pitch_under_pressure": None,
            "pitch_that_breaks_down": None,
            "learning_curves": {},
        }

    per_pt = mesi_result["per_pitch_type"]

    # Scouting insights: best pitch under pressure, pitch that breaks down
    best_pressure = None
    worst_breakdown = None
    best_pressure_score = -1.0
    worst_breakdown_score = float("inf")

    for pt, info in per_pt.items():
        ctx = info.get("context_snr", {})
        high_lev = ctx.get("high_leverage")
        fatigued = ctx.get("fatigued")

        # "Best under pressure": highest SNR in high leverage
        if high_lev is not None and high_lev > best_pressure_score:
            best_pressure_score = high_lev
            best_pressure = pt

        # "Breaks down": lowest CS score (or biggest SNR drop in fatigue)
        cs = info.get("cs_score")
        if cs is not None and cs < worst_breakdown_score:
            worst_breakdown_score = cs
            worst_breakdown = pt

    # Learning curves for each pitch type
    learning_curves = {}
    for pt in per_pt:
        lc = fit_learning_curve(conn, pitcher_id, pt)
        learning_curves[pt] = lc

    return {
        "pitcher_id": pitcher_id,
        "overall_mesi": mesi_result["overall_mesi"],
        "overall_snr": mesi_result["overall_snr"],
        "overall_cs": mesi_result["overall_cs"],
        "per_pitch_type": per_pt,
        "best_pitch_under_pressure": best_pressure,
        "pitch_that_breaks_down": worst_breakdown,
        "learning_curves": learning_curves,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BaseAnalyticsModel wrapper
# ─────────────────────────────────────────────────────────────────────────────


class MESIModel(BaseAnalyticsModel):
    """Motor Engram Stability Index model, conforming to BaseAnalyticsModel."""

    @property
    def model_name(self) -> str:
        return "Motor Engram Stability Index"

    @property
    def version(self) -> str:
        return "1.0.0"

    def train(self, conn, **kwargs) -> dict:
        """Train (compute) the population MESI leaderboard.

        MESI is not a supervised model — training is the batch
        computation of scores across all qualifying pitchers.

        Args:
            conn: Open DuckDB connection.
            **kwargs: Passed to ``batch_calculate`` (season, min_pitches).

        Returns:
            Training summary with population statistics.
        """
        season = kwargs.get("season")
        min_pitches = kwargs.get("min_pitches", MIN_PITCHES_QUALIFY)
        leaderboard = batch_calculate(conn, season=season, min_pitches=min_pitches)

        metrics = {
            "n_pitchers": len(leaderboard),
            "mean_mesi": round(float(leaderboard["overall_mesi"].mean()), 2)
            if not leaderboard.empty
            else 0.0,
            "std_mesi": round(float(leaderboard["overall_mesi"].std()), 2)
            if not leaderboard.empty
            else 0.0,
        }
        self.set_training_metadata(metrics=metrics, params={"min_pitches": min_pitches})
        return metrics

    def predict(self, conn, **kwargs) -> dict:
        """Predict MESI for a specific pitcher.

        Args:
            conn: Open DuckDB connection.
            **kwargs: Must include ``pitcher_id``. Optionally ``season``.

        Returns:
            MESI result dictionary.
        """
        pitcher_id = kwargs["pitcher_id"]
        season = kwargs.get("season")
        return calculate_mesi(conn, pitcher_id, season=season)

    def evaluate(self, conn, **kwargs) -> dict:
        """Evaluate the MESI model by computing population coverage.

        Returns:
            Dictionary with coverage and distribution statistics.
        """
        season = kwargs.get("season")
        leaderboard = batch_calculate(conn, season=season)
        n = len(leaderboard)
        if n == 0:
            return {"n_pitchers": 0, "coverage": 0.0}

        return {
            "n_pitchers": n,
            "coverage": round(n / max(1, n) * 100, 1),
            "mean_mesi": round(float(leaderboard["overall_mesi"].mean()), 2),
            "median_mesi": round(float(leaderboard["overall_mesi"].median()), 2),
            "std_mesi": round(float(leaderboard["overall_mesi"].std()), 2),
        }
