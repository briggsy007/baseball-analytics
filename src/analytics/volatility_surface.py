"""
Pitch Implied Volatility Surface (PIVS) model.

Borrows the implied volatility surface concept from options pricing and applies
it to pitching: for each pitcher we compute **Shannon entropy** of pitch outcome
distributions across a 5x5 spatial zone grid and 12 ball-strike count states.

High entropy means outcomes are unpredictable (the pitcher keeps hitters
guessing); low entropy means the outcome is deterministic (hitters can sit on
a particular result).

Derived metrics include:
- **Vol Smile**: entropy profile across zones at a fixed count
- **Vol Term Structure**: entropy profile across counts at a fixed zone
- **Vol Skew**: inside vs outside entropy asymmetry
- **Overall Volatility**: weighted average entropy
"""

from __future__ import annotations

import logging
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from src.analytics.base import BaseAnalyticsModel
from src.analytics.features import shannon_entropy, encode_count_state

logger = logging.getLogger(__name__)

# ── Zone grid constants ──────────────────────────────────────────────────────
# 5 columns across plate_x (-1.5 to 1.5) and 5 rows across plate_z (1.0 to 4.0)
# Plus 1 out-of-zone bucket = 26 cells total.

ZONE_X_EDGES: list[float] = [-1.5, -0.9, -0.3, 0.3, 0.9, 1.5]
ZONE_Z_EDGES: list[float] = [1.0, 1.6, 2.2, 2.8, 3.4, 4.0]

N_ZONE_COLS: int = 5
N_ZONE_ROWS: int = 5
N_ZONE_CELLS: int = N_ZONE_COLS * N_ZONE_ROWS + 1  # 25 in-zone + 1 out-of-zone = 26
OOZ_INDEX: int = 25  # out-of-zone bucket index

# ── Count states ─────────────────────────────────────────────────────────────
# 12 states: (0,0) through (3,2)
ALL_COUNT_STATES: list[str] = [
    f"{b}-{s}" for b in range(4) for s in range(3)
]
N_COUNTS: int = len(ALL_COUNT_STATES)  # 12

# ── Outcome categories ──────────────────────────────────────────────────────
OUTCOME_CATEGORIES: list[str] = [
    "called_strike",
    "swinging_strike",
    "foul",
    "ball",
    "weak_contact",
    "hard_contact",
    "medium_contact",
]
N_OUTCOMES: int = len(OUTCOME_CATEGORIES)  # 7

# Minimum pitches per cell for stable entropy
MIN_PITCHES_PER_CELL: int = 5


# ── Outcome classification ──────────────────────────────────────────────────

_CALLED_STRIKE_DESCS = {"called_strike"}
_SWINGING_STRIKE_DESCS = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
    "missed_bunt",
}
_BALL_DESCS = {"ball", "blocked_ball", "intent_ball", "pitchout", "hit_by_pitch"}
_FOUL_DESCS = {"foul", "foul_bunt"}
_IN_PLAY_DESCS = {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}


def classify_outcome(
    description: str | None,
    launch_speed: float | None,
    pitch_type_code: str | None,
) -> str:
    """Map a Statcast pitch to one of seven outcome categories.

    Categories:
        called_strike, swinging_strike, foul, ball,
        weak_contact (EV < 85), hard_contact (EV >= 95), medium_contact

    Args:
        description: The Statcast ``description`` column value.
        launch_speed: Exit velocity (``launch_speed``), may be None.
        pitch_type_code: Pitch type code (unused currently, reserved).

    Returns:
        One of the seven ``OUTCOME_CATEGORIES`` strings.
    """
    if description is None:
        return "ball"

    desc = description.strip().lower()

    if desc in _CALLED_STRIKE_DESCS:
        return "called_strike"
    if desc in _SWINGING_STRIKE_DESCS:
        return "swinging_strike"
    if desc in _FOUL_DESCS:
        return "foul"
    if desc in _BALL_DESCS:
        return "ball"
    if desc in _IN_PLAY_DESCS:
        if launch_speed is not None and not np.isnan(launch_speed):
            if launch_speed < 85.0:
                return "weak_contact"
            elif launch_speed >= 95.0:
                return "hard_contact"
            else:
                return "medium_contact"
        return "medium_contact"

    # Fallback for unknown descriptions
    return "ball"


# ── Zone assignment ──────────────────────────────────────────────────────────


def _assign_zone_index(plate_x: float, plate_z: float) -> int:
    """Assign a pitch location to one of 26 zone cells.

    The in-zone grid is 5x5 (indices 0-24). Index 25 is out-of-zone.

    Args:
        plate_x: Horizontal plate coordinate.
        plate_z: Vertical plate coordinate.

    Returns:
        Integer zone index in [0, 25].
    """
    # Check out-of-zone
    if (plate_x < ZONE_X_EDGES[0] or plate_x >= ZONE_X_EDGES[-1] or
            plate_z < ZONE_Z_EDGES[0] or plate_z >= ZONE_Z_EDGES[-1]):
        return OOZ_INDEX

    # Find column
    col = 0
    for i in range(1, len(ZONE_X_EDGES)):
        if plate_x < ZONE_X_EDGES[i]:
            col = i - 1
            break

    # Find row
    row = 0
    for i in range(1, len(ZONE_Z_EDGES)):
        if plate_z < ZONE_Z_EDGES[i]:
            row = i - 1
            break

    return row * N_ZONE_COLS + col


def _zone_index_to_col_row(zone_idx: int) -> tuple[int, int] | None:
    """Convert flat zone index to (col, row) tuple. Returns None for OOZ."""
    if zone_idx == OOZ_INDEX:
        return None
    row = zone_idx // N_ZONE_COLS
    col = zone_idx % N_ZONE_COLS
    return (col, row)


# ── Entropy matrix computation ───────────────────────────────────────────────


def compute_cell_entropy(
    pitches_df: pd.DataFrame,
    zone_col: str = "zone_idx",
    count_col: str = "count_state",
) -> np.ndarray:
    """Compute Shannon entropy for each (zone, count) cell.

    Args:
        pitches_df: DataFrame with columns ``zone_col``, ``count_col``,
            and ``outcome``.
        zone_col: Column name for zone index (0-25).
        count_col: Column name for count state string.

    Returns:
        NumPy array of shape (26, 12) where element [z, c] is the
        Shannon entropy of outcome distribution for zone z and count
        index c. Cells with fewer than ``MIN_PITCHES_PER_CELL`` pitches
        are set to NaN.
    """
    entropy_matrix = np.full((N_ZONE_CELLS, N_COUNTS), np.nan)
    count_to_idx = {cs: i for i, cs in enumerate(ALL_COUNT_STATES)}

    for (zone, count_str), group in pitches_df.groupby([zone_col, count_col]):
        if count_str not in count_to_idx:
            continue

        zone_i = int(zone)
        count_i = count_to_idx[count_str]

        if len(group) < MIN_PITCHES_PER_CELL:
            continue

        # Compute outcome distribution
        outcome_counts = group["outcome"].value_counts()
        probs = np.zeros(N_OUTCOMES)
        for j, cat in enumerate(OUTCOME_CATEGORIES):
            probs[j] = outcome_counts.get(cat, 0)

        total = probs.sum()
        if total == 0:
            continue

        probs = probs / total
        entropy_matrix[zone_i, count_i] = shannon_entropy(probs)

    return entropy_matrix


# ── Surface smoothing ────────────────────────────────────────────────────────


def smooth_surface(entropy_matrix: np.ndarray) -> np.ndarray:
    """Smooth the entropy surface using interpolation to fill NaN cells.

    Uses scipy's RBF interpolation when available. Falls back to simple
    nearest-neighbor filling.

    Args:
        entropy_matrix: Array of shape (26, 12) with NaN for sparse cells.

    Returns:
        Array of same shape with NaN cells filled by interpolation.
        Remaining NaN values (if any) are filled with the global mean.
    """
    smoothed = entropy_matrix.copy()

    try:
        from scipy.interpolate import RBFInterpolator

        # Get known (non-NaN) points
        known_mask = ~np.isnan(smoothed)
        if known_mask.sum() < 4:
            # Not enough points; fill with global mean
            global_mean = np.nanmean(smoothed)
            if np.isnan(global_mean):
                global_mean = 0.0
            smoothed[np.isnan(smoothed)] = global_mean
            return smoothed

        # Create coordinate grids
        zone_indices = np.arange(N_ZONE_CELLS)
        count_indices = np.arange(N_COUNTS)
        zz, cc = np.meshgrid(zone_indices, count_indices, indexing="ij")

        known_points = np.column_stack([
            zz[known_mask],
            cc[known_mask],
        ])
        known_values = smoothed[known_mask]

        unknown_mask = np.isnan(smoothed)
        if not unknown_mask.any():
            return smoothed

        unknown_points = np.column_stack([
            zz[unknown_mask],
            cc[unknown_mask],
        ])

        # RBF interpolation with linear kernel
        rbf = RBFInterpolator(
            known_points,
            known_values,
            kernel="linear",
            smoothing=1.0,
        )
        interpolated = rbf(unknown_points)

        # Clip to valid entropy range
        interpolated = np.clip(interpolated, 0.0, np.log(N_OUTCOMES) + 0.1)
        smoothed[unknown_mask] = interpolated

    except ImportError:
        logger.warning(
            "scipy not available for RBF smoothing; using mean fill."
        )
        global_mean = np.nanmean(smoothed)
        if np.isnan(global_mean):
            global_mean = 0.0
        smoothed[np.isnan(smoothed)] = global_mean

    # Final safety: fill any remaining NaN
    if np.isnan(smoothed).any():
        global_mean = np.nanmean(smoothed)
        if np.isnan(global_mean):
            global_mean = 0.0
        smoothed[np.isnan(smoothed)] = global_mean

    return smoothed


# ── Derived metrics ──────────────────────────────────────────────────────────


def _extract_vol_smile(
    surface: np.ndarray,
    count_idx: int,
) -> np.ndarray:
    """Extract entropy profile across in-zone columns at a fixed count.

    For each of the 5 zone columns, averages entropy across the 5 rows
    at the given count index.

    Args:
        surface: Smoothed entropy matrix of shape (26, 12).
        count_idx: Index into the count dimension (0-11).

    Returns:
        Array of length 5 (one value per zone column).
    """
    smile = np.zeros(N_ZONE_COLS)
    for col in range(N_ZONE_COLS):
        vals = []
        for row in range(N_ZONE_ROWS):
            idx = row * N_ZONE_COLS + col
            vals.append(surface[idx, count_idx])
        smile[col] = np.mean(vals)
    return smile


def _extract_vol_term_structure(
    surface: np.ndarray,
    zone_idx: int,
) -> np.ndarray:
    """Extract entropy profile across counts at a fixed zone.

    Args:
        surface: Smoothed entropy matrix of shape (26, 12).
        zone_idx: Zone index (0-25).

    Returns:
        Array of length 12 (one value per count state).
    """
    return surface[zone_idx, :].copy()


def _compute_vol_skew(
    surface: np.ndarray,
    count_idx: int,
) -> float:
    """Compute inside vs outside entropy asymmetry at a fixed count.

    "Inside" is defined as zone columns 0 and 4 (edges).
    "Middle" is zone column 2 (center).
    Skew = mean(edge entropy) - mean(center entropy).

    Args:
        surface: Smoothed entropy matrix of shape (26, 12).
        count_idx: Index into the count dimension (0-11).

    Returns:
        Skew value (positive = edges more volatile than center).
    """
    edge_vals = []
    center_vals = []
    for row in range(N_ZONE_ROWS):
        # Edge columns: 0, 4
        edge_vals.append(surface[row * N_ZONE_COLS + 0, count_idx])
        edge_vals.append(surface[row * N_ZONE_COLS + 4, count_idx])
        # Center column: 2
        center_vals.append(surface[row * N_ZONE_COLS + 2, count_idx])

    return float(np.mean(edge_vals) - np.mean(center_vals))


def _compute_overall_volatility(
    surface: np.ndarray,
    pitch_counts: np.ndarray | None = None,
) -> float:
    """Compute weighted average entropy across all cells.

    Args:
        surface: Smoothed entropy matrix of shape (26, 12).
        pitch_counts: Optional array of same shape with pitch counts
            per cell (used as weights). If None, uniform weights.

    Returns:
        Scalar overall volatility.
    """
    if pitch_counts is not None and pitch_counts.sum() > 0:
        weights = pitch_counts.astype(float)
        weights = weights / weights.sum()
        return float(np.nansum(surface * weights))
    else:
        valid = surface[~np.isnan(surface)]
        if len(valid) == 0:
            return 0.0
        return float(np.mean(valid))


# ── Pitch count matrix ───────────────────────────────────────────────────────


def _build_pitch_count_matrix(
    pitches_df: pd.DataFrame,
    zone_col: str = "zone_idx",
    count_col: str = "count_state",
) -> np.ndarray:
    """Build a (26, 12) matrix of pitch counts per cell."""
    count_matrix = np.zeros((N_ZONE_CELLS, N_COUNTS), dtype=int)
    count_to_idx = {cs: i for i, cs in enumerate(ALL_COUNT_STATES)}

    for (zone, count_str), group in pitches_df.groupby([zone_col, count_col]):
        if count_str not in count_to_idx:
            continue
        zone_i = int(zone)
        count_i = count_to_idx[count_str]
        count_matrix[zone_i, count_i] = len(group)

    return count_matrix


# ── Main calculation ─────────────────────────────────────────────────────────


def _query_pitcher_pitches(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: int | None = None,
) -> pd.DataFrame:
    """Query pitch-level data for a single pitcher.

    Returns a DataFrame with plate_x, plate_z, balls, strikes,
    description, launch_speed, and pitch_type columns.
    """
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    params: list = [pitcher_id]
    if season:
        params.append(season)

    query = f"""
        SELECT
            plate_x,
            plate_z,
            balls,
            strikes,
            description,
            launch_speed,
            pitch_type,
            type
        FROM pitches
        WHERE pitcher_id = $1
          AND plate_x IS NOT NULL
          AND plate_z IS NOT NULL
          AND description IS NOT NULL
          {season_filter}
    """
    return conn.execute(query, params).fetchdf()


def _prepare_pitches(df: pd.DataFrame) -> pd.DataFrame:
    """Add zone_idx, count_state, and outcome columns to raw pitch data."""
    df = df.copy()

    # Assign zone
    df["zone_idx"] = df.apply(
        lambda r: _assign_zone_index(float(r["plate_x"]), float(r["plate_z"])),
        axis=1,
    )

    # Clamp balls/strikes to valid ranges for count_state
    df["balls_clamped"] = df["balls"].clip(0, 3).astype(int)
    df["strikes_clamped"] = df["strikes"].clip(0, 2).astype(int)
    df["count_state"] = (
        df["balls_clamped"].astype(str) + "-" + df["strikes_clamped"].astype(str)
    )

    # Classify outcome
    df["outcome"] = df.apply(
        lambda r: classify_outcome(
            r.get("description"),
            r.get("launch_speed"),
            r.get("pitch_type"),
        ),
        axis=1,
    )

    return df


def calculate_volatility_surface(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: int | None = None,
) -> dict:
    """Calculate the Pitch Implied Volatility Surface for a pitcher.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        season: Optional season filter.

    Returns:
        Dictionary with keys:
            - pitcher_id: int
            - n_pitches: int
            - surface_raw: (26, 12) entropy matrix (with NaN for sparse cells)
            - surface_smooth: (26, 12) smoothed entropy matrix
            - pitch_counts: (26, 12) pitch count matrix
            - vol_smile: dict mapping count_state -> array of 5 floats
            - vol_term_structure: dict mapping zone_idx -> array of 12 floats
            - vol_skew: dict mapping count_state -> float
            - overall_vol: float
            - outcome_distributions: dict of (zone, count) -> prob vector
    """
    raw_df = _query_pitcher_pitches(conn, pitcher_id, season)

    if raw_df.empty:
        logger.warning("No pitches found for pitcher %d", pitcher_id)
        return {
            "pitcher_id": pitcher_id,
            "n_pitches": 0,
            "surface_raw": np.full((N_ZONE_CELLS, N_COUNTS), np.nan),
            "surface_smooth": np.full((N_ZONE_CELLS, N_COUNTS), np.nan),
            "pitch_counts": np.zeros((N_ZONE_CELLS, N_COUNTS), dtype=int),
            "vol_smile": {},
            "vol_term_structure": {},
            "vol_skew": {},
            "overall_vol": 0.0,
            "outcome_distributions": {},
        }

    pitches = _prepare_pitches(raw_df)
    n_pitches = len(pitches)
    logger.info(
        "Computing PIVS for pitcher %d with %d pitches", pitcher_id, n_pitches
    )

    # Entropy matrix
    entropy_raw = compute_cell_entropy(pitches)
    pitch_counts = _build_pitch_count_matrix(pitches)

    # Smooth surface
    entropy_smooth = smooth_surface(entropy_raw)

    # Derived metrics
    count_to_idx = {cs: i for i, cs in enumerate(ALL_COUNT_STATES)}

    # Vol smile for each count
    vol_smile: dict[str, list[float]] = {}
    for cs, ci in count_to_idx.items():
        smile = _extract_vol_smile(entropy_smooth, ci)
        vol_smile[cs] = smile.tolist()

    # Vol term structure for center zone and OOZ
    center_zone = 2 * N_ZONE_COLS + 2  # row=2, col=2 => index 12
    vol_term: dict[int, list[float]] = {}
    for zi in [center_zone, OOZ_INDEX]:
        ts = _extract_vol_term_structure(entropy_smooth, zi)
        vol_term[zi] = ts.tolist()

    # Vol skew at first count (0-0) and full count (3-2)
    vol_skew: dict[str, float] = {}
    for cs in ["0-0", "3-2", "0-2", "3-0"]:
        if cs in count_to_idx:
            vol_skew[cs] = _compute_vol_skew(entropy_smooth, count_to_idx[cs])

    # Overall volatility
    overall_vol = _compute_overall_volatility(entropy_smooth, pitch_counts)

    # Build outcome distributions for hoverdata
    outcome_dists: dict[str, dict[str, float]] = {}
    for (zone, count_str), group in pitches.groupby(["zone_idx", "count_state"]):
        if len(group) < MIN_PITCHES_PER_CELL:
            continue
        oc = group["outcome"].value_counts()
        total = oc.sum()
        dist = {cat: round(oc.get(cat, 0) / total, 3) for cat in OUTCOME_CATEGORIES}
        outcome_dists[f"{int(zone)}-{count_str}"] = dist

    return {
        "pitcher_id": pitcher_id,
        "n_pitches": n_pitches,
        "surface_raw": entropy_raw,
        "surface_smooth": entropy_smooth,
        "pitch_counts": pitch_counts,
        "vol_smile": vol_smile,
        "vol_term_structure": vol_term,
        "vol_skew": vol_skew,
        "overall_vol": overall_vol,
        "outcome_distributions": outcome_dists,
    }


# ── Batch calculation ────────────────────────────────────────────────────────


def batch_calculate(
    conn: duckdb.DuckDBPyConnection,
    season: int | None = None,
    min_pitches: int = 300,
) -> pd.DataFrame:
    """Compute per-pitcher volatility metrics for all qualifying pitchers.

    Args:
        conn: Open DuckDB connection.
        season: Optional season filter.
        min_pitches: Minimum pitch count to qualify.

    Returns:
        DataFrame with columns: pitcher_id, name, n_pitches, overall_vol,
        vol_skew_0_0, vol_skew_3_2, and zone-level summaries.
    """
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $1" if season else ""
    params: list = []
    if season:
        params.append(season)

    # Find qualifying pitchers
    query = f"""
        SELECT pitcher_id, COUNT(*) AS n_pitches
        FROM pitches
        WHERE plate_x IS NOT NULL
          AND plate_z IS NOT NULL
          AND description IS NOT NULL
          {season_filter}
        GROUP BY pitcher_id
        HAVING COUNT(*) >= {int(min_pitches)}
        ORDER BY n_pitches DESC
    """
    pitcher_df = conn.execute(query, params).fetchdf()

    if pitcher_df.empty:
        logger.info("No qualifying pitchers for batch PIVS calculation.")
        return pd.DataFrame(columns=[
            "pitcher_id", "name", "n_pitches", "overall_vol",
            "vol_skew_0_0", "vol_skew_3_2",
        ])

    rows: list[dict] = []
    for _, prow in pitcher_df.iterrows():
        pid = int(prow["pitcher_id"])
        try:
            result = calculate_volatility_surface(conn, pid, season)
            rows.append({
                "pitcher_id": pid,
                "n_pitches": result["n_pitches"],
                "overall_vol": round(result["overall_vol"], 4),
                "vol_skew_0_0": round(result["vol_skew"].get("0-0", 0.0), 4),
                "vol_skew_3_2": round(result["vol_skew"].get("3-2", 0.0), 4),
            })
        except Exception as exc:
            logger.warning("PIVS failed for pitcher %d: %s", pid, exc)

    result_df = pd.DataFrame(rows)

    if result_df.empty:
        return result_df

    # Join player names
    try:
        names_df = conn.execute(
            "SELECT player_id AS pitcher_id, full_name AS name FROM players"
        ).fetchdf()
        result_df = result_df.merge(names_df, on="pitcher_id", how="left")
    except Exception:
        result_df["name"] = None

    # Reorder columns
    front_cols = ["pitcher_id", "name", "n_pitches", "overall_vol"]
    other_cols = [c for c in result_df.columns if c not in front_cols]
    result_df = result_df[front_cols + sorted(other_cols)]
    result_df = result_df.sort_values("overall_vol", ascending=False).reset_index(
        drop=True
    )

    return result_df


# ── Surface comparison ───────────────────────────────────────────────────────


def compare_surfaces(
    surface1: np.ndarray,
    surface2: np.ndarray,
) -> dict:
    """Compare two volatility surfaces and return key differences.

    Args:
        surface1: First smoothed entropy matrix (26, 12).
        surface2: Second smoothed entropy matrix (26, 12).

    Returns:
        Dictionary with:
            - diff_surface: (26, 12) array of surface1 - surface2
            - mean_diff: float mean of absolute differences
            - max_diff_cell: tuple (zone_idx, count_idx) of maximum difference
            - max_diff_value: float value of maximum absolute difference
    """
    diff = surface1 - surface2
    abs_diff = np.abs(diff)

    max_idx = np.unravel_index(np.nanargmax(abs_diff), abs_diff.shape)

    return {
        "diff_surface": diff,
        "mean_diff": float(np.nanmean(abs_diff)),
        "max_diff_cell": (int(max_idx[0]), int(max_idx[1])),
        "max_diff_value": float(abs_diff[max_idx]),
    }


# ── BaseAnalyticsModel wrapper ───────────────────────────────────────────────


class PitchVolatilitySurfaceModel(BaseAnalyticsModel):
    """BaseAnalyticsModel wrapper for the PIVS computation.

    This is a descriptive/statistical model (not ML-trained), so ``train``
    is a no-op and ``predict`` runs the surface calculation.
    """

    @property
    def model_name(self) -> str:
        return "pitch_implied_volatility_surface"

    @property
    def version(self) -> str:
        return "1.0.0"

    def train(self, conn, **kwargs) -> dict:
        """No training needed -- PIVS is a descriptive model."""
        logger.info("PIVS is a descriptive model; no training required.")
        return {"status": "no_training_needed"}

    def predict(self, conn, **kwargs) -> dict:
        """Calculate the volatility surface for a pitcher.

        Keyword Args:
            pitcher_id: MLB player ID (required).
            season: Optional season filter.

        Returns:
            Dictionary from ``calculate_volatility_surface``.
        """
        pitcher_id = kwargs.get("pitcher_id")
        if pitcher_id is None:
            raise ValueError("pitcher_id is required for PIVS prediction.")
        season = kwargs.get("season")
        result = calculate_volatility_surface(conn, pitcher_id, season)
        return self.validate_output(result)

    def evaluate(self, conn, **kwargs) -> dict:
        """Evaluate surface coverage and stability metrics.

        Keyword Args:
            pitcher_id: MLB player ID (required).
            season: Optional season filter.

        Returns:
            Dictionary with coverage statistics.
        """
        pitcher_id = kwargs.get("pitcher_id")
        if pitcher_id is None:
            raise ValueError("pitcher_id is required for PIVS evaluation.")
        season = kwargs.get("season")
        result = calculate_volatility_surface(conn, pitcher_id, season)

        raw = result["surface_raw"]
        counts = result["pitch_counts"]

        total_cells = N_ZONE_CELLS * N_COUNTS
        populated_cells = int(np.sum(~np.isnan(raw)))
        coverage = populated_cells / total_cells if total_cells > 0 else 0.0

        return {
            "pitcher_id": pitcher_id,
            "n_pitches": result["n_pitches"],
            "total_cells": total_cells,
            "populated_cells": populated_cells,
            "coverage_pct": round(coverage * 100, 1),
            "overall_vol": result["overall_vol"],
            "max_entropy": float(np.nanmax(raw)) if populated_cells > 0 else 0.0,
            "min_entropy": float(np.nanmin(raw)) if populated_cells > 0 else 0.0,
        }
