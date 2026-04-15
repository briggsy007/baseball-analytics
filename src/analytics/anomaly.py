"""
In-game and cross-game anomaly detection for player performance.

Detects anomalies in pitcher velocity, spin rate, release point, and pitch mix
during live games. Also provides cross-game trend analysis for batters
(exit velocity, chase rate, barrel rate).

All detector functions are designed to be lightweight enough to run on every
pitch during a live game -- expensive database queries are performed once at
initialisation time and cached in the ``GameAnomalyMonitor`` class.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

import duckdb


# ── Fastball pitch-type codes used for velocity tracking ────────────────────
_FASTBALL_TYPES: set[str] = {"FF", "SI", "FC"}

# Swinging-strike descriptions used throughout the platform
_WHIFF_DESCRIPTIONS: set[str] = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
    "missed_bunt",
}

# Zone 11-14 are outside the strike zone (chase zone) in Statcast
_CHASE_ZONES: set[int] = {11, 12, 13, 14}


# ─────────────────────────────────────────────────────────────────────────────
# Baseline computation
# ─────────────────────────────────────────────────────────────────────────────


def calculate_pitcher_baselines(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> dict:
    """Compute baseline statistics for a pitcher from their season data.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID of the pitcher.
        season: If provided, restrict to pitches thrown in this calendar year.
                When ``None``, all available data is used.

    Returns:
        Dictionary keyed by pitch type with mean/std for velocity, spin,
        horizontal and vertical movement, and usage percentage. Also includes
        aggregate release-point statistics and total pitch count.
    """
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    params: list = [pitcher_id]
    if season:
        params.append(season)

    # -- Per-pitch-type aggregates ----------------------------------------
    pitch_type_df: pd.DataFrame = conn.execute(
        f"""
        SELECT pitch_type,
               AVG(release_speed)       AS mean_velo,
               STDDEV(release_speed)    AS std_velo,
               AVG(release_spin_rate)   AS mean_spin,
               STDDEV(release_spin_rate) AS std_spin,
               AVG(pfx_x)              AS mean_pfx_x,
               STDDEV(pfx_x)           AS std_pfx_x,
               AVG(pfx_z)              AS mean_pfx_z,
               STDDEV(pfx_z)           AS std_pfx_z,
               COUNT(*)                 AS cnt
        FROM   pitches
        WHERE  pitcher_id = $1
               AND pitch_type IS NOT NULL
               {season_filter}
        GROUP  BY pitch_type
        """,
        params,
    ).fetchdf()

    total_pitches: int = int(pitch_type_df["cnt"].sum()) if not pitch_type_df.empty else 0

    by_pitch_type: dict = {}
    for _, row in pitch_type_df.iterrows():
        pt: str = row["pitch_type"]
        by_pitch_type[pt] = {
            "mean_velo": _safe_float(row["mean_velo"]),
            "std_velo": _safe_float(row["std_velo"], default=1.0),
            "mean_spin": _safe_float(row["mean_spin"]),
            "std_spin": _safe_float(row["std_spin"], default=50.0),
            "mean_pfx_x": _safe_float(row["mean_pfx_x"]),
            "std_pfx_x": _safe_float(row["std_pfx_x"], default=1.0),
            "mean_pfx_z": _safe_float(row["mean_pfx_z"]),
            "std_pfx_z": _safe_float(row["std_pfx_z"], default=1.0),
            "usage_pct": round(float(row["cnt"]) / total_pitches, 4) if total_pitches > 0 else 0.0,
        }

    # -- Release point aggregates -----------------------------------------
    rp_df: pd.DataFrame = conn.execute(
        f"""
        SELECT AVG(release_pos_x)    AS mean_x,
               STDDEV(release_pos_x) AS std_x,
               AVG(release_pos_z)    AS mean_z,
               STDDEV(release_pos_z) AS std_z
        FROM   pitches
        WHERE  pitcher_id = $1
               AND release_pos_x IS NOT NULL
               AND release_pos_z IS NOT NULL
               {season_filter}
        """,
        params,
    ).fetchdf()

    release_point: dict = {
        "mean_x": _safe_float(rp_df.iloc[0]["mean_x"]) if not rp_df.empty else 0.0,
        "std_x": _safe_float(rp_df.iloc[0]["std_x"], default=0.5) if not rp_df.empty else 0.5,
        "mean_z": _safe_float(rp_df.iloc[0]["mean_z"]) if not rp_df.empty else 0.0,
        "std_z": _safe_float(rp_df.iloc[0]["std_z"], default=0.5) if not rp_df.empty else 0.5,
    }

    return {
        "pitcher_id": pitcher_id,
        "by_pitch_type": by_pitch_type,
        "release_point": release_point,
        "pitches_analyzed": total_pitches,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pitcher in-game anomaly detectors
# ─────────────────────────────────────────────────────────────────────────────


def detect_velocity_anomaly(
    game_pitches: list[dict],
    baseline: dict,
    window: int = 8,
    threshold: float = 2.0,
) -> list[dict]:
    """Detect fastball velocity drops within a game.

    Computes a rolling average over the last *window* fastballs and flags
    when the rolling average falls more than *threshold* mph below the
    pitcher's season baseline mean.  Also flags individual pitches that are
    more than 2.5 standard deviations below baseline.

    Thresholds are automatically scaled up by 1.3x for relievers (pitchers
    who have thrown fewer than 40 pitches in the game) to account for their
    naturally higher variance.

    Args:
        game_pitches: Ordered list of pitch dicts from the current game.
        baseline: Return value of :func:`calculate_pitcher_baselines`.
        window: Number of recent fastballs for the rolling average.
        threshold: Minimum mph drop (rolling avg vs baseline) to trigger
                   a *warning*.  A *critical* alert fires at ``threshold + 1.0``.

    Returns:
        List of alert dicts with keys ``type``, ``severity``, ``pitch_number``,
        ``pitch_type``, ``current_value``, ``baseline_value``,
        ``deviation_mph``, and ``message``.
    """
    alerts: list[dict] = []
    by_pt: dict = baseline.get("by_pitch_type", {})

    # Reliever adjustment: if fewer than 40 pitches, assume reliever
    reliever_factor = 1.3 if len(game_pitches) < 40 else 1.0
    adj_threshold = threshold * reliever_factor

    for pt_code in _FASTBALL_TYPES:
        pt_baseline = by_pt.get(pt_code)
        if pt_baseline is None:
            continue
        mean_velo: float = pt_baseline["mean_velo"]
        std_velo: float = pt_baseline["std_velo"]
        if mean_velo == 0.0:
            continue

        # Collect in-game fastballs of this type
        fb_velos: list[tuple[int, float]] = []
        for idx, p in enumerate(game_pitches):
            if p.get("pitch_type") != pt_code:
                continue
            velo = p.get("release_speed")
            if velo is None or not _is_finite(velo):
                continue
            fb_velos.append((idx, float(velo)))

        for i, (pitch_idx, velo) in enumerate(fb_velos):
            pitch_num: int = pitch_idx + 1  # 1-based

            # --- Single-pitch outlier check (>2.5 std devs below) --------
            sd_threshold = 2.5 * reliever_factor
            if std_velo > 0 and velo < mean_velo - sd_threshold * std_velo:
                dev = round(mean_velo - velo, 1)
                alerts.append({
                    "type": "velocity_drop",
                    "severity": "critical",
                    "pitch_number": pitch_num,
                    "pitch_type": pt_code,
                    "current_value": round(velo, 1),
                    "baseline_value": round(mean_velo, 1),
                    "deviation_mph": dev,
                    "message": (
                        f"Single {pt_code} at {velo:.1f} mph is {dev:.1f} mph "
                        f"below season avg ({mean_velo:.1f}), "
                        f">{sd_threshold * std_velo:.1f} mph ({sd_threshold:.1f} SD) drop"
                    ),
                })

            # --- Rolling-window check ------------------------------------
            if i + 1 >= window:
                recent = [v for _, v in fb_velos[i + 1 - window : i + 1]]
                rolling_avg = sum(recent) / len(recent)
                drop = mean_velo - rolling_avg
                if drop >= adj_threshold:
                    severity = "critical" if drop >= adj_threshold + 1.0 else "warning"
                    alerts.append({
                        "type": "velocity_drop",
                        "severity": severity,
                        "pitch_number": pitch_num,
                        "pitch_type": pt_code,
                        "current_value": round(rolling_avg, 1),
                        "baseline_value": round(mean_velo, 1),
                        "deviation_mph": round(drop, 1),
                        "message": (
                            f"{pt_code} velocity dropped {drop:.1f} mph "
                            f"({mean_velo:.1f} -> {rolling_avg:.1f}) "
                            f"over last {window} pitches"
                        ),
                    })

    return alerts


def detect_spin_anomaly(
    game_pitches: list[dict],
    baseline: dict,
    threshold_pct: float = 8.0,
    window: int = 8,
) -> list[dict]:
    """Detect spin-rate deviations within a game.

    Flags when a rolling-window average spin rate deviates by more than
    *threshold_pct* percent from the pitcher's season baseline for a given
    pitch type.  Spin drops can indicate fatigue; spin spikes were historically
    associated with foreign-substance use.

    Thresholds are scaled up by 1.3x for relievers (fewer than 40 pitches).

    Args:
        game_pitches: Ordered list of pitch dicts from the current game.
        baseline: Return value of :func:`calculate_pitcher_baselines`.
        threshold_pct: Percentage deviation to trigger an alert.
        window: Number of recent pitches (per type) for rolling average.

    Returns:
        List of alert dicts.
    """
    alerts: list[dict] = []
    by_pt: dict = baseline.get("by_pitch_type", {})

    # Reliever adjustment
    reliever_factor = 1.3 if len(game_pitches) < 40 else 1.0
    adj_threshold = threshold_pct * reliever_factor

    for pt_code, pt_baseline in by_pt.items():
        mean_spin: float = pt_baseline["mean_spin"]
        if mean_spin == 0.0:
            continue

        spins: list[tuple[int, float]] = []
        for idx, p in enumerate(game_pitches):
            if p.get("pitch_type") != pt_code:
                continue
            spin = p.get("release_spin_rate")
            if spin is None or not _is_finite(spin):
                continue
            spins.append((idx, float(spin)))

        for i, (pitch_idx, _spin) in enumerate(spins):
            if i + 1 < window:
                continue
            recent = [s for _, s in spins[i + 1 - window : i + 1]]
            rolling_avg = sum(recent) / len(recent)
            pct_dev = abs(rolling_avg - mean_spin) / mean_spin * 100.0
            if pct_dev >= adj_threshold:
                direction = "drop" if rolling_avg < mean_spin else "spike"
                severity = "critical" if pct_dev >= adj_threshold * 1.5 else "warning"
                alerts.append({
                    "type": "spin_rate_change",
                    "severity": severity,
                    "pitch_number": pitch_idx + 1,
                    "pitch_type": pt_code,
                    "current_value": round(rolling_avg, 0),
                    "baseline_value": round(mean_spin, 0),
                    "deviation_pct": round(pct_dev, 1),
                    "message": (
                        f"{pt_code} spin rate {direction}: "
                        f"{rolling_avg:.0f} rpm vs {mean_spin:.0f} rpm season avg "
                        f"({pct_dev:.1f}% deviation over last {window} pitches)"
                    ),
                })

    return alerts


def detect_release_point_drift(
    game_pitches: list[dict],
    baseline: dict,
    threshold_inches: float = 3.0,
    recent_window: int = 10,
) -> list[dict]:
    """Detect release-point drift during a game.

    Computes the RMSE of recent release points (last *recent_window* pitches)
    relative to the game-start release point cluster, and flags when the drift
    exceeds *threshold_inches*.  Also checks for consistent directional trends.

    Thresholds are scaled up by 1.3x for relievers (fewer than 40 pitches).

    Args:
        game_pitches: Ordered list of pitch dicts from the current game.
        baseline: Return value of :func:`calculate_pitcher_baselines`.
        threshold_inches: RMSE threshold (in inches) to trigger an alert.
        recent_window: Number of recent pitches to compare.

    Returns:
        List of alert dicts.
    """
    alerts: list[dict] = []

    # Reliever adjustment
    reliever_factor = 1.3 if len(game_pitches) < 40 else 1.0
    adj_threshold = threshold_inches * reliever_factor

    # Collect valid release points
    rp_data: list[tuple[int, float, float]] = []
    for idx, p in enumerate(game_pitches):
        rpx = p.get("release_pos_x")
        rpz = p.get("release_pos_z")
        if rpx is None or rpz is None or not (_is_finite(rpx) and _is_finite(rpz)):
            continue
        rp_data.append((idx, float(rpx), float(rpz)))

    if len(rp_data) < recent_window + 5:
        return alerts  # not enough data yet

    # Use the first 5 pitches as the "game start" reference cluster
    early_x = np.mean([x for _, x, _ in rp_data[:5]])
    early_z = np.mean([z for _, _, z in rp_data[:5]])

    # Latest window
    recent = rp_data[-recent_window:]
    recent_xs = np.array([x for _, x, _ in recent])
    recent_zs = np.array([z for _, _, z in recent])

    # RMSE relative to game-start cluster
    rmse = float(np.sqrt(np.mean((recent_xs - early_x) ** 2 + (recent_zs - early_z) ** 2)))

    # Convert feet to inches for more intuitive thresholds (Statcast data is in feet)
    rmse_inches = rmse * 12.0

    if rmse_inches >= adj_threshold:
        severity = "critical" if rmse_inches >= adj_threshold * 1.5 else "warning"
        pitch_num = recent[-1][0] + 1

        # Check for consistent directional trend
        drift_x = float(np.mean(recent_xs) - early_x) * 12.0
        drift_z = float(np.mean(recent_zs) - early_z) * 12.0
        direction_parts: list[str] = []
        if abs(drift_x) > 0.5:
            direction_parts.append(f"{'glove' if drift_x > 0 else 'arm'}-side ({abs(drift_x):.1f} in)")
        if abs(drift_z) > 0.5:
            direction_parts.append(f"{'higher' if drift_z > 0 else 'lower'} ({abs(drift_z):.1f} in)")
        direction_str = " and ".join(direction_parts) if direction_parts else "scattered"

        alerts.append({
            "type": "release_point_drift",
            "severity": severity,
            "pitch_number": pitch_num,
            "pitch_type": "ALL",
            "rmse_inches": round(rmse_inches, 1),
            "drift_x_inches": round(drift_x, 1),
            "drift_z_inches": round(drift_z, 1),
            "message": (
                f"Release point drifting {rmse_inches:.1f} inches from game start "
                f"(last {recent_window} pitches). "
                f"Direction: {direction_str}"
            ),
        })

    return alerts


def detect_pitch_mix_anomaly(
    game_pitches: list[dict],
    baseline: dict,
    min_pitches: int = 30,
    deviation_threshold: float = 20.0,
) -> list[dict]:
    """Detect unusual pitch-type distribution within a game.

    Compares the current game's pitch-mix percentages to the pitcher's season
    baseline.  Only triggers after *min_pitches* have been thrown to avoid
    false positives from small samples early in the game.

    Thresholds are scaled up by 1.3x for relievers (fewer than 40 pitches).

    Args:
        game_pitches: Ordered list of pitch dicts from the current game.
        baseline: Return value of :func:`calculate_pitcher_baselines`.
        min_pitches: Minimum number of pitches before detection activates.
        deviation_threshold: Absolute percentage-point difference between
            current game usage and season baseline to trigger an alert.

    Returns:
        List of alert dicts.
    """
    alerts: list[dict] = []

    # Count pitch types in this game
    type_counts: dict[str, int] = {}
    for p in game_pitches:
        pt = p.get("pitch_type")
        if pt is None:
            continue
        type_counts[pt] = type_counts.get(pt, 0) + 1

    total = sum(type_counts.values())
    if total < min_pitches:
        return alerts

    # Reliever adjustment
    reliever_factor = 1.3 if len(game_pitches) < 40 else 1.0
    adj_threshold = deviation_threshold * reliever_factor

    by_pt: dict = baseline.get("by_pitch_type", {})

    # Check each pitch type in the baseline
    all_types = set(type_counts.keys()) | set(by_pt.keys())
    for pt_code in all_types:
        game_pct = (type_counts.get(pt_code, 0) / total) * 100.0
        baseline_pct = by_pt.get(pt_code, {}).get("usage_pct", 0.0) * 100.0

        # Skip very rare pitch types in both game and baseline
        if game_pct < 3.0 and baseline_pct < 3.0:
            continue

        diff = game_pct - baseline_pct
        abs_diff = abs(diff)

        if abs_diff >= adj_threshold:
            direction = "increased" if diff > 0 else "decreased"
            severity = "critical" if abs_diff >= adj_threshold * 1.5 else "warning"
            alerts.append({
                "type": "pitch_mix_shift",
                "severity": severity,
                "pitch_number": total,
                "pitch_type": pt_code,
                "current_pct": round(game_pct, 1),
                "baseline_pct": round(baseline_pct, 1),
                "deviation_pct_pts": round(abs_diff, 1),
                "message": (
                    f"{pt_code} usage {direction}: "
                    f"{game_pct:.1f}% today vs {baseline_pct:.1f}% season average "
                    f"({abs_diff:.1f} pct-pt difference over {total} pitches)"
                ),
            })

    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Batter cross-game anomaly detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_batter_anomalies(
    conn: duckdb.DuckDBPyConnection,
    batter_id: int,
    n_games: int = 10,
) -> list[dict]:
    """Cross-game anomaly detection for a batter.

    Queries the last *n_games* games and compares EWMA-smoothed rolling
    statistics (exit velocity, chase rate, barrel rate) against the season
    average.  Uses exponentially-weighted moving average with ``span=10``
    for smoothing.

    Args:
        conn: Open DuckDB connection.
        batter_id: MLB player ID of the batter.
        n_games: Number of recent games to analyze.

    Returns:
        List of alert dicts for any detected anomalies.
    """
    alerts: list[dict] = []

    # -- Per-game aggregates -----------------------------------------------
    game_df: pd.DataFrame = conn.execute(
        """
        WITH game_agg AS (
            SELECT game_pk,
                   game_date,
                   -- exit velocity (balls in play only)
                   AVG(CASE WHEN type = 'X' AND launch_speed IS NOT NULL
                            THEN launch_speed END)            AS avg_ev,
                   -- chase rate: swings at pitches outside the zone
                   SUM(CASE WHEN zone IN (11,12,13,14)
                             AND description IN (
                                 'swinging_strike','swinging_strike_blocked',
                                 'foul_tip','foul','hit_into_play',
                                 'hit_into_play_no_out','hit_into_play_score')
                            THEN 1 ELSE 0 END)                AS chases,
                   SUM(CASE WHEN zone IN (11,12,13,14)
                            THEN 1 ELSE 0 END)                AS chase_opps,
                   -- barrel rate (launch_speed >= 98 and optimal angle)
                   SUM(CASE WHEN type = 'X'
                             AND launch_speed >= 98
                             AND launch_angle BETWEEN 26 AND 30
                            THEN 1 ELSE 0 END)                AS barrels,
                   SUM(CASE WHEN type = 'X'
                            THEN 1 ELSE 0 END)                AS batted_balls
            FROM   pitches
            WHERE  batter_id = $1
            GROUP  BY game_pk, game_date
        )
        SELECT *
        FROM   game_agg
        ORDER  BY game_date ASC
        """,
        [batter_id],
    ).fetchdf()

    if game_df.empty or len(game_df) < 5:
        return alerts

    # Compute derived rates
    game_df["chase_rate"] = game_df["chases"] / game_df["chase_opps"].replace(0, np.nan) * 100.0
    game_df["barrel_rate"] = game_df["barrels"] / game_df["batted_balls"].replace(0, np.nan) * 100.0

    # Season averages
    season_ev = game_df["avg_ev"].mean()
    season_chase = game_df["chase_rate"].mean()
    season_barrel = game_df["barrel_rate"].mean()

    # EWMA (span=10)
    recent = game_df.tail(n_games).copy()
    ewma_ev = recent["avg_ev"].ewm(span=10, min_periods=3).mean()
    ewma_chase = recent["chase_rate"].ewm(span=10, min_periods=3).mean()
    ewma_barrel = recent["barrel_rate"].ewm(span=10, min_periods=3).mean()

    # Latest EWMA values
    latest_ev = ewma_ev.iloc[-1] if not ewma_ev.empty and _is_finite(ewma_ev.iloc[-1]) else None
    latest_chase = ewma_chase.iloc[-1] if not ewma_chase.empty and _is_finite(ewma_chase.iloc[-1]) else None
    latest_barrel = ewma_barrel.iloc[-1] if not ewma_barrel.empty and _is_finite(ewma_barrel.iloc[-1]) else None

    # --- Exit velocity drop (>3 mph below season avg) --------------------
    if latest_ev is not None and _is_finite(season_ev):
        ev_drop = season_ev - latest_ev
        if ev_drop > 3.0:
            severity = "critical" if ev_drop > 5.0 else "warning"
            alerts.append({
                "type": "exit_velocity_drop",
                "severity": severity,
                "batter_id": batter_id,
                "current_value": round(float(latest_ev), 1),
                "baseline_value": round(float(season_ev), 1),
                "deviation": round(float(ev_drop), 1),
                "message": (
                    f"Exit velocity trending down: {latest_ev:.1f} mph "
                    f"(EWMA) vs {season_ev:.1f} mph season avg "
                    f"({ev_drop:.1f} mph drop over last {n_games} games)"
                ),
            })

    # --- Chase rate spike (>5% above season avg) -------------------------
    if latest_chase is not None and _is_finite(season_chase):
        chase_increase = latest_chase - season_chase
        if chase_increase > 5.0:
            severity = "critical" if chase_increase > 10.0 else "warning"
            alerts.append({
                "type": "chase_rate_spike",
                "severity": severity,
                "batter_id": batter_id,
                "current_value": round(float(latest_chase), 1),
                "baseline_value": round(float(season_chase), 1),
                "deviation": round(float(chase_increase), 1),
                "message": (
                    f"Chase rate trending up: {latest_chase:.1f}% "
                    f"(EWMA) vs {season_chase:.1f}% season avg "
                    f"(+{chase_increase:.1f} pct-pt over last {n_games} games)"
                ),
            })

    # --- Barrel rate drop (>3% below season avg) -------------------------
    if latest_barrel is not None and _is_finite(season_barrel) and season_barrel > 0:
        barrel_drop = season_barrel - latest_barrel
        if barrel_drop > 3.0:
            severity = "critical" if barrel_drop > 6.0 else "warning"
            alerts.append({
                "type": "barrel_rate_drop",
                "severity": severity,
                "batter_id": batter_id,
                "current_value": round(float(latest_barrel), 1),
                "baseline_value": round(float(season_barrel), 1),
                "deviation": round(float(barrel_drop), 1),
                "message": (
                    f"Barrel rate trending down: {latest_barrel:.1f}% "
                    f"(EWMA) vs {season_barrel:.1f}% season avg "
                    f"({barrel_drop:.1f} pct-pt drop over last {n_games} games)"
                ),
            })

    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Live-game monitor class
# ─────────────────────────────────────────────────────────────────────────────


class GameAnomalyMonitor:
    """Stateful monitor that accumulates pitches and runs anomaly detection.

    Designed for real-time use: baselines are loaded once in ``__init__``,
    and each ``add_pitch`` / ``check_all`` cycle is O(n) in pitches thrown
    this game -- no database queries after initialisation.

    Args:
        pitcher_id: MLB player ID of the pitcher being monitored.
        conn: Open DuckDB connection used to load baselines. If ``None``,
              baselines must be supplied later via ``baseline`` attribute.
        season: Optional season year for baseline computation.
    """

    # Cooldown: minimum number of pitches between alerts of the same type
    ALERT_COOLDOWN_PITCHES: int = 10
    # Severity decay: alerts persisting this many pitches without worsening
    # are downgraded from "warning" to "info"
    SEVERITY_DECAY_PITCHES: int = 15

    def __init__(
        self,
        pitcher_id: int,
        conn: Optional[duckdb.DuckDBPyConnection] = None,
        season: Optional[int] = None,
    ) -> None:
        self.pitcher_id: int = pitcher_id
        self._pitches: list[dict] = []
        self._alerts: list[dict] = []
        # Track the last pitch number at which each alert type fired,
        # and the peak severity observed for ongoing alerts.
        self._last_alert_pitch: dict[str, int] = {}
        self._alert_first_seen: dict[str, int] = {}
        self._alert_peak_severity: dict[str, str] = {}

        if conn is not None:
            self.baseline: dict = calculate_pitcher_baselines(conn, pitcher_id, season)
        else:
            self.baseline = {
                "pitcher_id": pitcher_id,
                "by_pitch_type": {},
                "release_point": {"mean_x": 0.0, "std_x": 0.5, "mean_z": 0.0, "std_z": 0.5},
                "pitches_analyzed": 0,
            }

    # -- Pitch ingestion --------------------------------------------------

    def add_pitch(self, pitch: dict) -> None:
        """Append a pitch to the in-game log.

        Args:
            pitch: Dictionary with at least ``pitch_type``,
                   ``release_speed``, ``release_spin_rate``,
                   ``release_pos_x``, ``release_pos_z``.  Extra keys
                   are preserved but not required.
        """
        self._pitches.append(pitch)

    # -- Detection --------------------------------------------------------

    def check_all(self) -> list[dict]:
        """Run all pitcher anomaly detectors and return combined alerts.

        Applies three layers of noise reduction:

        1. **Cooldown** -- the same alert type cannot fire again within
           ``ALERT_COOLDOWN_PITCHES`` pitches of its last firing.
        2. **Deduplication** -- only the most recent (highest pitch-number)
           alert per type is kept; earlier duplicates are collapsed into it.
        3. **Severity decay** -- if an alert type has been continuously
           present for ``SEVERITY_DECAY_PITCHES`` pitches without escalating
           to a worse severity, it is downgraded from ``"warning"`` to
           ``"info"`` to reduce visual clutter.

        Returns:
            Deduplicated list of alert dicts from all detectors, sorted
            by pitch number (most recent first).
        """
        raw_alerts: list[dict] = []
        raw_alerts.extend(
            detect_velocity_anomaly(self._pitches, self.baseline)
        )
        raw_alerts.extend(
            detect_spin_anomaly(self._pitches, self.baseline)
        )
        raw_alerts.extend(
            detect_release_point_drift(self._pitches, self.baseline)
        )
        raw_alerts.extend(
            detect_pitch_mix_anomaly(self._pitches, self.baseline)
        )

        # --- Cooldown: suppress alerts that fired too recently -----------
        cooled: list[dict] = []
        for alert in raw_alerts:
            a_type = alert.get("type", "unknown")
            pitch_num = alert.get("pitch_number", 0)
            last_pitch = self._last_alert_pitch.get(a_type, -999)
            if pitch_num - last_pitch < self.ALERT_COOLDOWN_PITCHES:
                continue
            cooled.append(alert)

        # --- Deduplication: keep only the most recent alert per type -----
        best_by_type: dict[str, dict] = {}
        for alert in cooled:
            a_type = alert.get("type", "unknown")
            pitch_num = alert.get("pitch_number", 0)
            existing = best_by_type.get(a_type)
            if existing is None or pitch_num > existing.get("pitch_number", 0):
                best_by_type[a_type] = alert

        deduped = list(best_by_type.values())

        # --- Severity decay: downgrade stale warnings to info ------------
        _severity_rank = {"info": 0, "warning": 1, "critical": 2}
        for alert in deduped:
            a_type = alert.get("type", "unknown")
            pitch_num = alert.get("pitch_number", 0)
            severity = alert.get("severity", "warning")

            # Track first-seen pitch for this alert type
            if a_type not in self._alert_first_seen:
                self._alert_first_seen[a_type] = pitch_num
                self._alert_peak_severity[a_type] = severity
            else:
                # Update peak if severity escalated
                if _severity_rank.get(severity, 0) > _severity_rank.get(
                    self._alert_peak_severity.get(a_type, "info"), 0
                ):
                    self._alert_peak_severity[a_type] = severity
                    # Reset the clock when severity worsens
                    self._alert_first_seen[a_type] = pitch_num

            pitches_active = pitch_num - self._alert_first_seen[a_type]
            if (
                pitches_active >= self.SEVERITY_DECAY_PITCHES
                and severity == "warning"
            ):
                alert["severity"] = "info"

            # Record this firing for the cooldown tracker
            self._last_alert_pitch[a_type] = pitch_num

        # Clean up first-seen tracking for alert types that are no longer firing
        active_types = {a.get("type") for a in deduped}
        for a_type in list(self._alert_first_seen.keys()):
            if a_type not in active_types:
                self._alert_first_seen.pop(a_type, None)
                self._alert_peak_severity.pop(a_type, None)

        # Sort by pitch number, most recent first
        deduped.sort(key=lambda a: a.get("pitch_number", 0), reverse=True)
        self._alerts = deduped
        return self._alerts

    # -- Dashboard data ---------------------------------------------------

    def get_dashboard_data(self) -> dict:
        """Return current state formatted for front-end display.

        Returns:
            Dictionary containing pitcher identity, pitch count, active
            alerts, and time-series data for velocity, spin, release point,
            and pitch-mix comparison.
        """
        velocity_trend = self._build_velocity_trend()
        spin_trend = self._build_spin_trend()
        release_scatter = self._build_release_scatter()
        current_mix, baseline_mix = self._build_pitch_mix()

        return {
            "pitcher_id": self.pitcher_id,
            "pitches_thrown": len(self._pitches),
            "alerts": self._alerts,
            "velocity_trend": velocity_trend,
            "spin_trend": spin_trend,
            "release_point_scatter": release_scatter,
            "pitch_mix_current": current_mix,
            "pitch_mix_baseline": baseline_mix,
        }

    # -- Private helpers --------------------------------------------------

    def _build_velocity_trend(self) -> list[dict]:
        """Build pitch-by-pitch velocity series with rolling average."""
        window = 8
        fb_entries: list[dict] = []
        for idx, p in enumerate(self._pitches):
            if p.get("pitch_type") not in _FASTBALL_TYPES:
                continue
            velo = p.get("release_speed")
            if velo is None or not _is_finite(velo):
                continue
            fb_entries.append({
                "pitch_num": idx + 1,
                "velo": round(float(velo), 1),
            })

        # Compute rolling average
        for i, entry in enumerate(fb_entries):
            start = max(0, i + 1 - window)
            vals = [fb_entries[j]["velo"] for j in range(start, i + 1)]
            entry["rolling_avg"] = round(sum(vals) / len(vals), 1)

        return fb_entries

    def _build_spin_trend(self) -> list[dict]:
        """Build pitch-by-pitch spin-rate series with rolling average."""
        window = 8
        entries: list[dict] = []
        for idx, p in enumerate(self._pitches):
            spin = p.get("release_spin_rate")
            if spin is None or not _is_finite(spin):
                continue
            entries.append({
                "pitch_num": idx + 1,
                "spin": round(float(spin), 0),
                "pitch_type": p.get("pitch_type", ""),
            })

        for i, entry in enumerate(entries):
            start = max(0, i + 1 - window)
            vals = [entries[j]["spin"] for j in range(start, i + 1)]
            entry["rolling_avg"] = round(sum(vals) / len(vals), 0)

        return entries

    def _build_release_scatter(self) -> list[dict]:
        """Build release-point scatter data."""
        scatter: list[dict] = []
        for idx, p in enumerate(self._pitches):
            rpx = p.get("release_pos_x")
            rpz = p.get("release_pos_z")
            if rpx is None or rpz is None:
                continue
            if not (_is_finite(rpx) and _is_finite(rpz)):
                continue
            scatter.append({
                "x": round(float(rpx), 3),
                "z": round(float(rpz), 3),
                "pitch_num": idx + 1,
            })
        return scatter

    def _build_pitch_mix(self) -> tuple[dict, dict]:
        """Build current-game pitch-mix vs baseline."""
        counts: dict[str, int] = {}
        for p in self._pitches:
            pt = p.get("pitch_type")
            if pt is None:
                continue
            counts[pt] = counts.get(pt, 0) + 1

        total = sum(counts.values())
        current_mix: dict[str, float] = {}
        if total > 0:
            current_mix = {pt: round(cnt / total, 3) for pt, cnt in counts.items()}

        baseline_mix: dict[str, float] = {
            pt: round(info["usage_pct"], 3)
            for pt, info in self.baseline.get("by_pitch_type", {}).items()
        }

        return current_mix, baseline_mix


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────


def _safe_float(val, default: float = 0.0) -> float:
    """Convert a possibly-None / NaN value to a plain float."""
    if val is None:
        return default
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _is_finite(val) -> bool:
    """Return True if *val* is a finite number."""
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False
