"""
Lineup Order Flow Toxicity (LOFT) model.

Adapted from market microstructure's VPIN (Volume-synchronized Probability
of Informed Trading): detects when a lineup has **decoded** a pitcher.

Each pitch is classified as a "buy" (hitter-favorable outcome) or "sell"
(pitcher-favorable outcome).  Pitches are grouped into volume-synchronized
buckets of N pitches, and order-flow imbalance is measured within each
bucket.  An EWMA-smoothed LOFT score tracks how one-sided the flow
becomes over time; when it breaches the pitcher's seasonal baseline by
2+ standard deviations, a toxicity alert fires.

Inherits from ``BaseAnalyticsModel`` to plug into the standard
train/predict/evaluate lifecycle.
"""

from __future__ import annotations

import logging
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from src.analytics.base import BaseAnalyticsModel

logger = logging.getLogger(__name__)

# ── Tuning constants ─────────────────────────────────────────────────────────

BUCKET_SIZE: int = 15
EWMA_ALPHA: float = 0.3
HARD_CONTACT_THRESHOLD: float = 95.0
WEAK_CONTACT_THRESHOLD: float = 85.0
ALERT_SIGMA: float = 2.0
MIN_PITCHES_GAME: int = 30
MIN_GAMES_BASELINE: int = 3

# Statcast zone mapping: zones 1-9 are inside the strike zone
_STRIKE_ZONES: set[int] = {1, 2, 3, 4, 5, 6, 7, 8, 9}

# Descriptions that indicate a swing
_SWING_DESCRIPTIONS: set[str] = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
    "missed_bunt",
    "foul",
    "foul_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

# Descriptions that indicate a whiff (swinging strike)
_WHIFF_DESCRIPTIONS: set[str] = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
    "missed_bunt",
}


# ── Pitch classification ─────────────────────────────────────────────────────


def classify_pitch_flow(pitches_df: pd.DataFrame) -> pd.DataFrame:
    """Classify each pitch as 'buy' (hitter-favorable) or 'sell' (pitcher-favorable).

    Classification rules:
      **Buy** (hitter wins the pitch):
        - Hard contact: launch_speed >= 95 mph
        - Walk (events == 'walk')
        - Hit by pitch (events == 'hit_by_pitch' or description == 'hit_by_pitch')
        - Good take: ball outside the zone (type == 'B' and zone not in 1-9)
        - Ball on hittable pitch: type == 'B' and zone in 1-9 (pitcher missed)

      **Sell** (pitcher wins the pitch):
        - Swinging strike / whiff
        - Chase: swing on pitch outside zone (zones 11-14)
        - Weak contact: launch_speed < 85 mph
        - Called strike in zone (description == 'called_strike' and zone in 1-9)

    Pitches that match neither are classified as 'neutral'.

    Args:
        pitches_df: DataFrame with Statcast pitch columns.

    Returns:
        Copy of the input with a new ``flow`` column ('buy', 'sell', or 'neutral').
    """
    df = pitches_df.copy()
    flow = pd.Series("neutral", index=df.index)

    # Normalise description to lowercase for safe comparison
    desc = df["description"].fillna("").str.strip().str.lower()
    # "type" column is B/S/X (ball/strike/in-play), NOT the pitch_type (FF/SL/CH etc.)
    bsx_type = df["type"].fillna("") if "type" in df.columns else pd.Series("", index=df.index)
    zone = df["zone"].fillna(0).astype(int) if "zone" in df.columns else pd.Series(0, index=df.index)
    launch_speed = df["launch_speed"] if "launch_speed" in df.columns else pd.Series(np.nan, index=df.index)
    events = df["events"].fillna("").str.strip().str.lower() if "events" in df.columns else pd.Series("", index=df.index)

    in_zone = zone.isin(_STRIKE_ZONES)
    outside_zone = ~in_zone & (zone > 0)

    # Determine if the pitch was swung at
    is_swing = desc.isin(_SWING_DESCRIPTIONS)

    # ── BUY signals ──────────────────────────────────────────────────────
    # Hard contact
    hard_contact = launch_speed.notna() & (launch_speed >= HARD_CONTACT_THRESHOLD)
    flow = flow.where(~hard_contact, "buy")

    # Walk
    is_walk = events == "walk"
    flow = flow.where(~is_walk, "buy")

    # HBP
    is_hbp = (events == "hit_by_pitch") | (desc == "hit_by_pitch")
    flow = flow.where(~is_hbp, "buy")

    # Good take: ball outside zone
    good_take = (bsx_type == "B") & outside_zone
    flow = flow.where(~good_take, "buy")

    # Ball on hittable pitch: ball called in zone (pitcher missed location)
    ball_in_zone = (bsx_type == "B") & in_zone
    flow = flow.where(~ball_in_zone, "buy")

    # ── SELL signals ─────────────────────────────────────────────────────
    # Swinging strike / whiff
    is_whiff = desc.isin(_WHIFF_DESCRIPTIONS)
    # Only mark as sell if not already a buy (hard contact on a foul_tip, etc.)
    sell_whiff = is_whiff & (flow != "buy")
    flow = flow.where(~sell_whiff, "sell")

    # Chase: swing outside zone
    is_chase = is_swing & outside_zone & (flow != "buy")
    flow = flow.where(~is_chase, "sell")

    # Weak contact
    weak_contact = launch_speed.notna() & (launch_speed < WEAK_CONTACT_THRESHOLD) & (flow != "buy")
    flow = flow.where(~weak_contact, "sell")

    # Called strike in zone
    called_strike_in_zone = (desc == "called_strike") & in_zone & (flow != "buy")
    flow = flow.where(~called_strike_in_zone, "sell")

    df["flow"] = flow
    return df


# ── Bucket construction and LOFT computation ─────────────────────────────────


def _build_buckets(classified_df: pd.DataFrame, bucket_size: int = BUCKET_SIZE) -> pd.DataFrame:
    """Group classified pitches into volume-synchronized buckets.

    Args:
        classified_df: DataFrame with a ``flow`` column.
        bucket_size: Number of pitches per bucket.

    Returns:
        DataFrame with one row per bucket containing buy/sell counts and LOFT.
    """
    df = classified_df.reset_index(drop=True)
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=[
            "bucket_id", "buy_count", "sell_count", "total_flow",
            "loft", "start_idx", "end_idx",
        ])

    records = []
    bucket_id = 0

    for start in range(0, n, bucket_size):
        end = min(start + bucket_size, n)
        chunk = df.iloc[start:end]
        buy_count = int((chunk["flow"] == "buy").sum())
        sell_count = int((chunk["flow"] == "sell").sum())
        total_flow = buy_count + sell_count

        if total_flow > 0:
            loft = abs(buy_count - sell_count) / total_flow
        else:
            loft = 0.0

        records.append({
            "bucket_id": bucket_id,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "total_flow": total_flow,
            "loft": round(loft, 4),
            "start_idx": start,
            "end_idx": end - 1,
        })
        bucket_id += 1

    return pd.DataFrame(records)


def _rolling_loft(bucket_df: pd.DataFrame, alpha: float = EWMA_ALPHA) -> pd.Series:
    """Compute EWMA-smoothed rolling LOFT over buckets.

    Args:
        bucket_df: DataFrame from ``_build_buckets``.
        alpha: EWMA smoothing factor (higher = more weight on recent).

    Returns:
        Series of rolling LOFT values aligned with bucket_df index.
    """
    if bucket_df.empty:
        return pd.Series(dtype=float)
    return bucket_df["loft"].ewm(alpha=alpha, min_periods=1).mean()


# ── Game-level analysis ──────────────────────────────────────────────────────


def compute_game_loft(
    conn: duckdb.DuckDBPyConnection,
    game_pk: int,
    pitcher_id: int,
    bucket_size: int = BUCKET_SIZE,
    alpha: float = EWMA_ALPHA,
) -> dict:
    """Compute per-bucket LOFT time series for a pitcher in a specific game.

    Args:
        conn: Open DuckDB connection.
        game_pk: Unique game identifier.
        pitcher_id: MLB player ID.
        bucket_size: Pitches per bucket.
        alpha: EWMA smoothing factor.

    Returns:
        Dictionary with game metadata, bucket time series, and summary stats.
        Returns empty sentinel if fewer than ``MIN_PITCHES_GAME`` pitches.
    """
    query = """
        SELECT *
        FROM pitches
        WHERE game_pk = $1
          AND pitcher_id = $2
        ORDER BY at_bat_number, pitch_number
    """
    pitches_df = conn.execute(query, [game_pk, pitcher_id]).fetchdf()

    if pitches_df.empty or len(pitches_df) < MIN_PITCHES_GAME:
        return _empty_game_loft(game_pk, pitcher_id, len(pitches_df))

    classified = classify_pitch_flow(pitches_df)
    buckets = _build_buckets(classified, bucket_size=bucket_size)

    if buckets.empty:
        return _empty_game_loft(game_pk, pitcher_id, len(pitches_df))

    buckets["rolling_loft"] = _rolling_loft(buckets, alpha=alpha)

    # Buy/sell cumulative totals for the stacked area chart
    classified_reset = classified.reset_index(drop=True)
    cum_buy = (classified_reset["flow"] == "buy").cumsum()
    cum_sell = (classified_reset["flow"] == "sell").cumsum()

    return {
        "pitcher_id": pitcher_id,
        "game_pk": game_pk,
        "game_date": str(pitches_df["game_date"].iloc[0]) if "game_date" in pitches_df.columns else None,
        "total_pitches": len(pitches_df),
        "total_buckets": len(buckets),
        "buy_total": int((classified["flow"] == "buy").sum()),
        "sell_total": int((classified["flow"] == "sell").sum()),
        "neutral_total": int((classified["flow"] == "neutral").sum()),
        "mean_loft": round(float(buckets["loft"].mean()), 4),
        "max_loft": round(float(buckets["loft"].max()), 4),
        "final_rolling_loft": round(float(buckets["rolling_loft"].iloc[-1]), 4),
        "buckets": buckets.to_dict("records"),
        "cumulative_buy": cum_buy.tolist(),
        "cumulative_sell": cum_sell.tolist(),
    }


# ── Pitcher baseline ─────────────────────────────────────────────────────────


def compute_pitcher_baseline(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: int,
    bucket_size: int = BUCKET_SIZE,
    alpha: float = EWMA_ALPHA,
) -> dict:
    """Compute season-level LOFT distribution for a pitcher.

    Iterates over all qualifying game appearances (>= MIN_PITCHES_GAME)
    and computes the LOFT across them.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        season: Season year.
        bucket_size: Pitches per bucket.
        alpha: EWMA smoothing factor.

    Returns:
        Dictionary with mean, std, and game-level LOFT values.
    """
    query = """
        SELECT game_pk, COUNT(*) AS pitch_count
        FROM pitches
        WHERE pitcher_id = $1
          AND EXTRACT(YEAR FROM game_date) = $2
        GROUP BY game_pk
        HAVING COUNT(*) >= $3
        ORDER BY game_pk
    """
    games_df = conn.execute(query, [pitcher_id, season, MIN_PITCHES_GAME]).fetchdf()

    if games_df.empty:
        return _empty_baseline(pitcher_id, season)

    game_loft_values = []
    all_bucket_lofts = []

    for _, row in games_df.iterrows():
        gp = int(row["game_pk"])
        result = compute_game_loft(conn, gp, pitcher_id, bucket_size=bucket_size, alpha=alpha)
        if result["total_buckets"] > 0:
            game_loft_values.append(result["mean_loft"])
            for b in result["buckets"]:
                all_bucket_lofts.append(b["loft"])

    if not all_bucket_lofts:
        return _empty_baseline(pitcher_id, season)

    arr = np.array(all_bucket_lofts)
    game_arr = np.array(game_loft_values)

    return {
        "pitcher_id": pitcher_id,
        "season": season,
        "n_games": len(game_loft_values),
        "n_buckets": len(all_bucket_lofts),
        "mean_loft": round(float(arr.mean()), 4),
        "std_loft": round(float(arr.std()), 4),
        "mean_game_loft": round(float(game_arr.mean()), 4),
        "std_game_loft": round(float(game_arr.std()), 4),
        "game_loft_values": [round(float(v), 4) for v in game_loft_values],
    }


# ── Toxicity event detection ─────────────────────────────────────────────────


def detect_toxicity_events(
    game_loft: dict,
    baseline: dict,
    sigma: float = ALERT_SIGMA,
) -> list[dict]:
    """Flag buckets where the rolling LOFT exceeds the pitcher's baseline.

    An alert fires when rolling LOFT > baseline_mean + sigma * baseline_std.

    Args:
        game_loft: Output of ``compute_game_loft``.
        baseline: Output of ``compute_pitcher_baseline``.
        sigma: Number of standard deviations for the threshold.

    Returns:
        List of alert dictionaries, one per breached bucket.
    """
    if not game_loft.get("buckets") or baseline.get("std_loft", 0) == 0:
        return []

    threshold = baseline["mean_loft"] + sigma * baseline["std_loft"]
    alerts = []

    for bucket in game_loft["buckets"]:
        rolling = bucket.get("rolling_loft", bucket["loft"])
        if rolling > threshold:
            alerts.append({
                "bucket_id": bucket["bucket_id"],
                "rolling_loft": round(rolling, 4),
                "threshold": round(threshold, 4),
                "excess_sigma": round(
                    (rolling - baseline["mean_loft"]) / baseline["std_loft"], 2
                ) if baseline["std_loft"] > 0 else 0.0,
                "buy_count": bucket["buy_count"],
                "sell_count": bucket["sell_count"],
                "start_pitch": bucket["start_idx"],
                "end_pitch": bucket["end_idx"],
            })

    return alerts


# ── Batch analysis ───────────────────────────────────────────────────────────


def batch_game_analysis(
    conn: duckdb.DuckDBPyConnection,
    season: int,
    min_games: int = MIN_GAMES_BASELINE,
) -> pd.DataFrame:
    """Compute toxicity events across an entire season for all pitchers.

    Args:
        conn: Open DuckDB connection.
        season: Season year.
        min_games: Minimum qualifying game appearances.

    Returns:
        DataFrame with columns: pitcher_id, game_pk, mean_loft,
        max_loft, n_alerts, decoded (bool).
    """
    query = """
        SELECT pitcher_id, COUNT(DISTINCT game_pk) AS n_games
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) = $1
        GROUP BY pitcher_id
        HAVING COUNT(DISTINCT game_pk) >= $2
    """
    pitchers_df = conn.execute(query, [season, min_games]).fetchdf()

    if pitchers_df.empty:
        return pd.DataFrame(columns=[
            "pitcher_id", "game_pk", "game_date", "mean_loft",
            "max_loft", "n_alerts", "decoded",
        ])

    rows = []
    for _, pitcher_row in pitchers_df.iterrows():
        pid = int(pitcher_row["pitcher_id"])
        try:
            baseline = compute_pitcher_baseline(conn, pid, season)
            if baseline["n_games"] == 0:
                continue

            # Get individual game results
            games_query = """
                SELECT DISTINCT game_pk
                FROM pitches
                WHERE pitcher_id = $1
                  AND EXTRACT(YEAR FROM game_date) = $2
            """
            games = conn.execute(games_query, [pid, season]).fetchdf()

            for _, g_row in games.iterrows():
                gpk = int(g_row["game_pk"])
                game_result = compute_game_loft(conn, gpk, pid)
                if game_result["total_buckets"] == 0:
                    continue

                alerts = detect_toxicity_events(game_result, baseline)
                rows.append({
                    "pitcher_id": pid,
                    "game_pk": gpk,
                    "game_date": game_result.get("game_date"),
                    "mean_loft": game_result["mean_loft"],
                    "max_loft": game_result["max_loft"],
                    "n_alerts": len(alerts),
                    "decoded": len(alerts) > 0,
                })
        except Exception as exc:
            logger.warning("LOFT batch failed for pitcher %d: %s", pid, exc)

    if not rows:
        return pd.DataFrame(columns=[
            "pitcher_id", "game_pk", "game_date", "mean_loft",
            "max_loft", "n_alerts", "decoded",
        ])

    return pd.DataFrame(rows)


# ── Full profile (main entry point) ──────────────────────────────────────────


def calculate_loft(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: int,
) -> dict:
    """Compute the full LOFT profile for a pitcher in a season.

    Combines baseline stats, per-game LOFT, and toxicity detection
    into one comprehensive result.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        season: Season year.

    Returns:
        Dictionary with baseline, game-level results, and leaderboard-ready
        summary stats.
    """
    baseline = compute_pitcher_baseline(conn, pitcher_id, season)

    if baseline["n_games"] == 0:
        return {
            "pitcher_id": pitcher_id,
            "season": season,
            "baseline": baseline,
            "games": [],
            "total_alerts": 0,
            "games_decoded": 0,
            "avg_loft": 0.0,
        }

    # Get all game appearances
    query = """
        SELECT DISTINCT game_pk
        FROM pitches
        WHERE pitcher_id = $1
          AND EXTRACT(YEAR FROM game_date) = $2
    """
    games_df = conn.execute(query, [pitcher_id, season]).fetchdf()

    game_results = []
    total_alerts = 0
    games_decoded = 0

    for _, g_row in games_df.iterrows():
        gpk = int(g_row["game_pk"])
        game_loft = compute_game_loft(conn, gpk, pitcher_id)
        if game_loft["total_buckets"] == 0:
            continue

        alerts = detect_toxicity_events(game_loft, baseline)
        total_alerts += len(alerts)
        if alerts:
            games_decoded += 1

        game_results.append({
            "game_pk": gpk,
            "game_date": game_loft.get("game_date"),
            "mean_loft": game_loft["mean_loft"],
            "max_loft": game_loft["max_loft"],
            "total_pitches": game_loft["total_pitches"],
            "buy_total": game_loft["buy_total"],
            "sell_total": game_loft["sell_total"],
            "n_alerts": len(alerts),
            "decoded": len(alerts) > 0,
        })

    avg_loft = (
        round(float(np.mean([g["mean_loft"] for g in game_results])), 4)
        if game_results else 0.0
    )

    return {
        "pitcher_id": pitcher_id,
        "season": season,
        "baseline": baseline,
        "games": game_results,
        "total_alerts": total_alerts,
        "games_decoded": games_decoded,
        "avg_loft": avg_loft,
    }


# ── BaseAnalyticsModel subclass ─────────────────────────────────────────────


class LOFTModel(BaseAnalyticsModel):
    """Lineup Order Flow Toxicity model."""

    @property
    def model_name(self) -> str:
        return "LOFT"

    @property
    def version(self) -> str:
        return "1.0.0"

    def train(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """No training step -- LOFT is a descriptive statistical model."""
        metrics = {"status": "no_training_needed"}
        self.set_training_metadata(metrics)
        return metrics

    def predict(
        self,
        conn: duckdb.DuckDBPyConnection,
        *,
        pitcher_id: int,
        season: int,
    ) -> dict:
        """Compute the full LOFT profile.  Delegates to ``calculate_loft``."""
        return calculate_loft(conn, pitcher_id, season)

    def evaluate(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Evaluate model coverage: how many pitchers have toxicity events."""
        season = kwargs.get("season", 2025)
        min_games = kwargs.get("min_games", MIN_GAMES_BASELINE)
        df = batch_game_analysis(conn, season=season, min_games=min_games)
        if df.empty:
            return {"qualifying_pitchers": 0, "total_games": 0, "decoded_pct": 0.0}

        n_pitchers = df["pitcher_id"].nunique()
        n_games = len(df)
        decoded_pct = round(float(df["decoded"].mean()) * 100, 2) if n_games > 0 else 0.0

        return {
            "qualifying_pitchers": n_pitchers,
            "total_games": n_games,
            "decoded_pct": decoded_pct,
        }


# ── Empty sentinels ──────────────────────────────────────────────────────────


def _empty_game_loft(game_pk: int, pitcher_id: int, n_pitches: int = 0) -> dict:
    """Return an empty game LOFT result."""
    return {
        "pitcher_id": pitcher_id,
        "game_pk": game_pk,
        "game_date": None,
        "total_pitches": n_pitches,
        "total_buckets": 0,
        "buy_total": 0,
        "sell_total": 0,
        "neutral_total": 0,
        "mean_loft": 0.0,
        "max_loft": 0.0,
        "final_rolling_loft": 0.0,
        "buckets": [],
        "cumulative_buy": [],
        "cumulative_sell": [],
    }


def _empty_baseline(pitcher_id: int, season: int) -> dict:
    """Return an empty baseline result."""
    return {
        "pitcher_id": pitcher_id,
        "season": season,
        "n_games": 0,
        "n_buckets": 0,
        "mean_loft": 0.0,
        "std_loft": 0.0,
        "mean_game_loft": 0.0,
        "std_game_loft": 0.0,
        "game_loft_values": [],
    }
