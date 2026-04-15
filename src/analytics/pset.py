"""
Pitch Sequence Expected Threat (PSET) model.

Applies tennis serve game theory to pitch sequencing: pitchers face a
strategic trade-off between *predictability* (repeating effective patterns
that batters can learn) and *tunnel quality* (making consecutive pitches
look identical as long as possible, then diverge).

PSET quantifies this by combining three signals for each consecutive
pitch pair within an at-bat:

    PSET = base_run_value + tunnel_bonus - predictability_penalty

The aggregate PSET per 100 pitches is scaled to runs above average,
giving a single number for how much value a pitcher's sequencing creates
(or destroys) independent of raw stuff.

Inherits from ``BaseAnalyticsModel`` to plug into the standard
train/predict/evaluate lifecycle.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from src.analytics.base import BaseAnalyticsModel
from src.analytics.features import compute_run_value, shannon_entropy

logger = logging.getLogger(__name__)

# ── Tuning constants ─────────────────────────────────────────────────────────

# Weight for the predictability penalty relative to raw run value
ALPHA_PREDICTABILITY: float = 0.15

# Weight for the tunnel bonus relative to raw run value
BETA_TUNNEL: float = 0.20

# Normalisation denominators for tunnel sub-components
_PLATE_DISTANCE_NORM: float = 2.0   # feet — typical max plate distance
_VELO_DIFF_NORM: float = 15.0       # mph — typical max velo spread
_MOVE_DIFF_NORM: float = 20.0       # inches — typical max movement spread

# Minimum sample sizes
MIN_PITCHES_DEFAULT: int = 300
MIN_PAIRS_PER_TYPE: int = 5

# League-average wOBA reference for run-value conversion
_LEAGUE_WOBA: float = 0.320
_WOBA_SCALE: float = 1.25


# ── Helper: get pitch pairs with run values ──────────────────────────────────


def _get_pitch_pairs_with_rv(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """Return consecutive pitch pairs within each at-bat, enriched with run
    value and movement data.

    Uses LAG window function to attach previous-pitch attributes to each
    row.  Only returns rows where both the current and previous pitch
    have a non-null pitch_type.
    """
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    params: list = [pitcher_id]
    if season:
        params.append(season)

    query = f"""
        WITH ordered AS (
            SELECT
                game_pk,
                game_date,
                at_bat_number,
                pitch_number,
                pitch_type,
                release_speed,
                release_spin_rate,
                pfx_x,
                pfx_z,
                plate_x,
                plate_z,
                description,
                events,
                balls,
                strikes,
                woba_value,
                woba_denom,
                LAG(pitch_type) OVER w      AS prev_pitch_type,
                LAG(release_speed) OVER w   AS prev_release_speed,
                LAG(pfx_x) OVER w           AS prev_pfx_x,
                LAG(pfx_z) OVER w           AS prev_pfx_z,
                LAG(plate_x) OVER w         AS prev_plate_x,
                LAG(plate_z) OVER w         AS prev_plate_z
            FROM pitches
            WHERE pitcher_id = $1
              AND pitch_type IS NOT NULL
              {season_filter}
            WINDOW w AS (
                PARTITION BY game_pk, at_bat_number
                ORDER BY pitch_number
            )
        )
        SELECT *
        FROM ordered
        WHERE prev_pitch_type IS NOT NULL
        ORDER BY game_date, at_bat_number, pitch_number
    """
    return conn.execute(query, params).fetchdf()


# ── Core model ───────────────────────────────────────────────────────────────


class PSETModel(BaseAnalyticsModel):
    """Pitch Sequence Expected Threat model."""

    @property
    def model_name(self) -> str:
        return "PSET"

    @property
    def version(self) -> str:
        return "1.0.0"

    def train(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """No training step required — PSET is a descriptive metric."""
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
        """Compute PSET for a single pitcher.  Delegates to ``calculate_pset``."""
        return calculate_pset(conn, pitcher_id, season=season)

    def evaluate(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Evaluate model coverage: how many pitchers qualify."""
        season = kwargs.get("season")
        min_pitches = kwargs.get("min_pitches", MIN_PITCHES_DEFAULT)
        df = batch_calculate(conn, season=season, min_pitches=min_pitches)
        return {
            "qualifying_pitchers": len(df),
            "mean_pset_per_100": round(float(df["pset_per_100"].mean()), 4)
            if not df.empty
            else 0.0,
            "std_pset_per_100": round(float(df["pset_per_100"].std()), 4)
            if not df.empty
            else 0.0,
        }


# ── Public functions ─────────────────────────────────────────────────────────


def build_transition_matrix(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Build a pitcher's conditional transition matrix: P(pitch_N | pitch_N-1, count).

    Returns:
        Nested dict keyed by ``count_state`` -> ``prev_pitch_type`` ->
        ``{next_type: probability, ...}``.  Each inner dict sums to ~1.0.
    """
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    params: list = [pitcher_id]
    if season:
        params.append(season)

    query = f"""
        WITH ordered AS (
            SELECT
                game_pk,
                at_bat_number,
                pitch_number,
                pitch_type,
                balls,
                strikes,
                LAG(pitch_type) OVER (
                    PARTITION BY game_pk, at_bat_number
                    ORDER BY pitch_number
                ) AS prev_pitch_type
            FROM pitches
            WHERE pitcher_id = $1
              AND pitch_type IS NOT NULL
              {season_filter}
        )
        SELECT
            balls || '-' || strikes AS count_state,
            prev_pitch_type,
            pitch_type,
            COUNT(*) AS cnt
        FROM ordered
        WHERE prev_pitch_type IS NOT NULL
        GROUP BY count_state, prev_pitch_type, pitch_type
    """
    df = conn.execute(query, params).fetchdf()

    if df.empty:
        logger.warning("No pitch pairs found for pitcher %d", pitcher_id)
        return {}

    matrix: dict[str, dict[str, dict[str, float]]] = {}
    for count, count_grp in df.groupby("count_state"):
        matrix[count] = {}
        for prev_pt, pt_grp in count_grp.groupby("prev_pitch_type"):
            total = pt_grp["cnt"].sum()
            if total > 0:
                matrix[count][prev_pt] = {
                    row["pitch_type"]: round(row["cnt"] / total, 4)
                    for _, row in pt_grp.iterrows()
                }

    return matrix


def compute_predictability(
    transition_matrix: dict[str, dict[str, dict[str, float]]],
    pitch_sequence: pd.DataFrame,
) -> pd.Series:
    """Compute a per-pitch predictability score (conditional entropy).

    For each pitch pair (prev -> curr) at a given count, look up the
    conditional distribution P(next | prev, count) and compute its
    Shannon entropy.  Lower entropy = more predictable = higher penalty.

    Args:
        transition_matrix: Output of ``build_transition_matrix``.
        pitch_sequence: DataFrame with columns ``prev_pitch_type``,
            ``balls``, ``strikes``.

    Returns:
        Series of predictability penalty values (higher = more predictable,
        which is worse for the pitcher).  Uses 1 - normalised_entropy so
        that a perfectly predictable pitcher gets penalty=1 and a maximally
        mixed pitcher gets penalty~0.
    """
    penalties = []

    for _, row in pitch_sequence.iterrows():
        count = f"{int(row['balls'])}-{int(row['strikes'])}"
        prev_pt = row.get("prev_pitch_type")

        if prev_pt is None or pd.isna(prev_pt):
            penalties.append(0.0)
            continue

        # Look up the distribution
        count_dist = transition_matrix.get(count, {})
        pt_dist = count_dist.get(prev_pt, {})

        if not pt_dist:
            penalties.append(0.0)
            continue

        probs = np.array(list(pt_dist.values()), dtype=np.float64)
        n_types = len(probs)

        if n_types <= 1:
            # Only one option => maximally predictable
            penalties.append(1.0)
            continue

        # Shannon entropy
        entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
        max_entropy = math.log(n_types)

        if max_entropy == 0:
            penalties.append(1.0)
        else:
            normalised = entropy / max_entropy  # 0 = predictable, 1 = mixed
            penalties.append(round(1.0 - normalised, 4))

    return pd.Series(penalties, index=pitch_sequence.index, name="predictability")


def compute_tunnel_scores(pitch_pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-pair tunnel metrics for consecutive pitches.

    Returns a DataFrame with columns:
        - ``plate_distance``: Euclidean distance between plate locations
        - ``velo_diff``: absolute velocity difference (mph)
        - ``movement_diff``: Euclidean distance in (pfx_x, pfx_z) space
        - ``tunnel_score``: similarity_at_decision * divergence_at_plate

    The tunnel score captures the idea that great sequences look the same
    early (similar plate approach) but end up in different places (high
    movement differential).
    """
    df = pitch_pairs_df.copy()

    # Plate distance (lower = pitches arrive in similar spots)
    dx = df["plate_x"].astype(float) - df["prev_plate_x"].astype(float)
    dz = df["plate_z"].astype(float) - df["prev_plate_z"].astype(float)
    df["plate_distance"] = np.sqrt(dx**2 + dz**2)

    # Velocity differential
    df["velo_diff"] = (
        df["release_speed"].astype(float) - df["prev_release_speed"].astype(float)
    ).abs()

    # Movement differential (induced movement in inches)
    dpfx_x = df["pfx_x"].astype(float) - df["prev_pfx_x"].astype(float)
    dpfx_z = df["pfx_z"].astype(float) - df["prev_pfx_z"].astype(float)
    df["movement_diff"] = np.sqrt(dpfx_x**2 + dpfx_z**2)

    # Tunnel score: similarity at approach * divergence at plate
    # similarity = 1 / (1 + normalised_plate_distance)
    # divergence = normalised(velo_diff + movement_diff)
    similarity = 1.0 / (1.0 + df["plate_distance"] / _PLATE_DISTANCE_NORM)
    divergence = (
        df["velo_diff"] / _VELO_DIFF_NORM + df["movement_diff"] / _MOVE_DIFF_NORM
    ) / 2.0
    df["tunnel_score"] = (similarity * divergence).clip(0.0, 1.0)

    return df[["plate_distance", "velo_diff", "movement_diff", "tunnel_score"]]


def calculate_pset(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> dict:
    """Calculate PSET for a single pitcher.

    Returns a dictionary with:
        - ``pitcher_id``
        - ``pset_per_100``: aggregate PSET scaled to per-100-pitches
        - ``predictability_score``: average predictability penalty
        - ``tunnel_score``: average tunnel bonus
        - ``base_rv_per_100``: average base run value per 100 pitches
        - ``total_pairs``: number of pitch pairs analysed
        - ``breakdown``: list of dicts per pitch-pair type with PSET contribution
    """
    pairs_df = _get_pitch_pairs_with_rv(conn, pitcher_id, season=season)

    if pairs_df.empty:
        logger.warning("No pitch pairs for pitcher %d", pitcher_id)
        return _empty_pset_result(pitcher_id)

    # ── Base run value per pitch ──────────────────────────────────────────
    pairs_df["run_value"] = pairs_df.apply(
        lambda r: compute_run_value(r.get("woba_value"), r.get("woba_denom")),
        axis=1,
    )
    # For non-PA-ending pitches, run_value is None => fill with 0
    pairs_df["run_value"] = pairs_df["run_value"].fillna(0.0)

    # ── Predictability penalty ────────────────────────────────────────────
    tm = build_transition_matrix(conn, pitcher_id, season=season)
    pred_scores = compute_predictability(tm, pairs_df)
    pairs_df["predictability"] = pred_scores

    # ── Tunnel bonus ──────────────────────────────────────────────────────
    tunnel_df = compute_tunnel_scores(pairs_df)
    pairs_df["tunnel_score"] = tunnel_df["tunnel_score"].values
    pairs_df["plate_distance"] = tunnel_df["plate_distance"].values
    pairs_df["velo_diff"] = tunnel_df["velo_diff"].values
    pairs_df["movement_diff"] = tunnel_df["movement_diff"].values

    # ── Per-pitch PSET ────────────────────────────────────────────────────
    # Flip sign on run_value: negative run value = good for pitcher
    # So pitcher-friendly PSET = -run_value + tunnel_bonus - predictability_penalty
    pairs_df["pset_raw"] = (
        -pairs_df["run_value"]
        + BETA_TUNNEL * pairs_df["tunnel_score"]
        - ALPHA_PREDICTABILITY * pairs_df["predictability"]
    )

    total_pairs = len(pairs_df)
    mean_pset = float(pairs_df["pset_raw"].mean()) if total_pairs > 0 else 0.0
    pset_per_100 = round(mean_pset * 100, 4)

    # ── Breakdown by pitch-pair type ──────────────────────────────────────
    pairs_df["pair_type"] = (
        pairs_df["prev_pitch_type"] + " -> " + pairs_df["pitch_type"]
    )
    breakdown = []
    for pair_type, grp in pairs_df.groupby("pair_type"):
        if len(grp) < MIN_PAIRS_PER_TYPE:
            continue
        breakdown.append(
            {
                "pair": pair_type,
                "count": len(grp),
                "pset_contribution": round(float(grp["pset_raw"].mean()) * 100, 4),
                "avg_tunnel_score": round(float(grp["tunnel_score"].mean()), 4),
                "avg_predictability": round(
                    float(grp["predictability"].mean()), 4
                ),
                "avg_plate_distance": round(
                    float(grp["plate_distance"].mean()), 4
                ),
                "avg_velo_diff": round(float(grp["velo_diff"].mean()), 2),
                "avg_movement_diff": round(float(grp["movement_diff"].mean()), 2),
            }
        )
    breakdown.sort(key=lambda x: x["pset_contribution"], reverse=True)

    return {
        "pitcher_id": pitcher_id,
        "pset_per_100": pset_per_100,
        "predictability_score": round(
            float(pairs_df["predictability"].mean()), 4
        ),
        "tunnel_score": round(float(pairs_df["tunnel_score"].mean()), 4),
        "base_rv_per_100": round(
            float(-pairs_df["run_value"].mean()) * 100, 4
        ),
        "total_pairs": total_pairs,
        "breakdown": breakdown,
    }


def batch_calculate(
    conn: duckdb.DuckDBPyConnection,
    season: Optional[int] = None,
    min_pitches: int = MIN_PITCHES_DEFAULT,
) -> pd.DataFrame:
    """Compute PSET leaderboard for all qualifying pitchers.

    Args:
        conn: Open DuckDB connection.
        season: Optional season filter.
        min_pitches: Minimum total pitches to qualify.

    Returns:
        DataFrame with columns: pitcher_id, pset_per_100,
        predictability_score, tunnel_score, total_pairs, sorted by
        pset_per_100 descending.
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

    if qualifying.empty:
        logger.info("No pitchers qualify with min_pitches=%d", min_pitches)
        return pd.DataFrame(
            columns=[
                "pitcher_id",
                "pset_per_100",
                "predictability_score",
                "tunnel_score",
                "total_pairs",
            ]
        )

    rows = []
    for _, row in qualifying.iterrows():
        pid = int(row["pitcher_id"])
        try:
            result = calculate_pset(conn, pid, season=season)
            if result["total_pairs"] > 0:
                rows.append(
                    {
                        "pitcher_id": pid,
                        "pset_per_100": result["pset_per_100"],
                        "predictability_score": result["predictability_score"],
                        "tunnel_score": result["tunnel_score"],
                        "total_pairs": result["total_pairs"],
                    }
                )
        except Exception as exc:
            logger.warning("PSET failed for pitcher %d: %s", pid, exc)

    if not rows:
        return pd.DataFrame(
            columns=[
                "pitcher_id",
                "pset_per_100",
                "predictability_score",
                "tunnel_score",
                "total_pairs",
            ]
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("pset_per_100", ascending=False).reset_index(drop=True)
    return df


def get_best_sequences(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
    top_n: int = 5,
) -> list[dict]:
    """Return the top N most effective pitch-pair transitions for a pitcher.

    "Effective" is defined as the pitch pairs with the highest average
    PSET contribution (best tunnel + lowest predictability + best run
    value).

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        season: Optional season filter.
        top_n: Number of top sequences to return.

    Returns:
        List of dicts with keys: pair, count, pset_contribution,
        avg_tunnel_score, avg_predictability.
    """
    result = calculate_pset(conn, pitcher_id, season=season)

    if not result["breakdown"]:
        return []

    return result["breakdown"][:top_n]


# ── Private helpers ──────────────────────────────────────────────────────────


def _empty_pset_result(pitcher_id: int) -> dict:
    """Return an empty PSET result dict when no data is available."""
    return {
        "pitcher_id": pitcher_id,
        "pset_per_100": 0.0,
        "predictability_score": 0.0,
        "tunnel_score": 0.0,
        "base_rv_per_100": 0.0,
        "total_pairs": 0,
        "breakdown": [],
    }
