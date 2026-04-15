"""
Pitch sequencing analysis engine.

Analyses how pitchers construct at-bats through pitch ordering, count-specific
tendencies, setup-knockout pairs, and pitch tunneling.  Also evaluates batter
vulnerabilities by pitch type and zone, and generates matchup-specific pitch
plans.

All public functions accept an open DuckDB connection as the first argument and
return plain dicts / DataFrames suitable for dashboard rendering.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Swinging-strike descriptions (Statcast conventions)
_WHIFF_DESCRIPTIONS: set[str] = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
    "missed_bunt",
}

_SWING_DESCRIPTIONS: set[str] = {
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "missed_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
}

_CALLED_STRIKE_DESCRIPTIONS: set[str] = {"called_strike"}

_STRIKE_DESCRIPTIONS: set[str] = (
    _WHIFF_DESCRIPTIONS | _CALLED_STRIKE_DESCRIPTIONS | {"foul", "foul_bunt"}
)

# Zone mapping: zones 1-9 are a 3x3 grid; 11-14 are outside the zone
# For RHB: 1=high-inside, 3=high-outside, 7=low-inside, 9=low-outside
# For LHB: inside/outside flipped
_ZONE_LABELS_RHB: dict[int, str] = {
    1: "high-inside",  2: "high-middle",  3: "high-outside",
    4: "mid-inside",   5: "middle",       6: "mid-outside",
    7: "low-inside",   8: "low-middle",   9: "low-outside",
}

_ZONE_LABELS_LHB: dict[int, str] = {
    1: "high-outside", 2: "high-middle",  3: "high-inside",
    4: "mid-outside",  5: "middle",       6: "mid-inside",
    7: "low-outside",  8: "low-middle",   9: "low-inside",
}

_OUTSIDE_ZONES: set[int] = {0, 11, 12, 13, 14}

# Count states used for analysis
_ALL_COUNTS: list[str] = [
    "0-0", "0-1", "0-2",
    "1-0", "1-1", "1-2",
    "2-0", "2-1", "2-2",
    "3-0", "3-1", "3-2",
]

_TWO_STRIKE_COUNTS: set[str] = {"0-2", "1-2", "2-2", "3-2"}


# ─────────────────────────────────────────────────────────────────────────────
# Pitch sequence extraction
# ─────────────────────────────────────────────────────────────────────────────


def get_pitch_sequence(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    game_pk: Optional[int] = None,
) -> pd.DataFrame:
    """Extract ordered pitch sequences from at-bats for a pitcher.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        game_pk: If provided, restrict to a single game.

    Returns:
        DataFrame with columns: game_pk, at_bat_number, pitch_number,
        pitch_type, release_speed, plate_x, plate_z, description, events,
        count_state, zone, stand.
    """
    game_filter = "AND game_pk = $2" if game_pk else ""
    params: list = [pitcher_id]
    if game_pk:
        params.append(game_pk)

    query = f"""
        SELECT
            game_pk,
            game_date,
            at_bat_number,
            pitch_number,
            pitch_type,
            release_speed,
            plate_x,
            plate_z,
            description,
            events,
            balls,
            strikes,
            zone,
            stand,
            batter_id,
            woba_value,
            woba_denom,
            type
        FROM pitches
        WHERE pitcher_id = $1
          AND pitch_type IS NOT NULL
          {game_filter}
        ORDER BY game_pk, at_bat_number, pitch_number
    """
    df = conn.execute(query, params).fetchdf()

    if df.empty:
        return pd.DataFrame()

    # Build the count state string (balls-strikes BEFORE this pitch)
    df["count_state"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Sequencing pattern analysis
# ─────────────────────────────────────────────────────────────────────────────


def analyze_sequencing_patterns(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> dict:
    """Analyse a pitcher's sequencing tendencies across their data.

    Computes transition probabilities, count-specific pitch mix, setup-knockout
    pairs, first-pitch tendencies, putaway pitch analysis, and a basic pitch
    tunneling score.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        season: Optional season year filter.

    Returns:
        Dictionary with keys: pitcher_id, transition_matrix, count_mix,
        setup_knockout_pairs, first_pitch, putaway, pitch_tunneling.
    """
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    params: list = [pitcher_id]
    if season:
        params.append(season)

    query = f"""
        SELECT
            game_pk,
            at_bat_number,
            pitch_number,
            pitch_type,
            release_speed,
            pfx_x,
            pfx_z,
            plate_x,
            plate_z,
            description,
            events,
            balls,
            strikes,
            zone,
            stand,
            type
        FROM pitches
        WHERE pitcher_id = $1
          AND pitch_type IS NOT NULL
          {season_filter}
        ORDER BY game_pk, at_bat_number, pitch_number
    """
    df = conn.execute(query, params).fetchdf()

    if df.empty:
        return _empty_sequencing_result(pitcher_id)

    df["count_state"] = df["balls"].astype(str) + "-" + df["strikes"].astype(str)
    df["is_whiff"] = df["description"].isin(_WHIFF_DESCRIPTIONS)
    df["is_strike"] = df["description"].isin(_STRIKE_DESCRIPTIONS)
    df["is_swing"] = df["description"].isin(_SWING_DESCRIPTIONS)
    df["is_chase"] = df["zone"].isin(_OUTSIDE_ZONES) & df["is_swing"]

    # ── Transition matrix ────────────────────────────────────────────────
    transition_matrix = _compute_transition_matrix(df)

    # ── Count-specific mix ───────────────────────────────────────────────
    count_mix = _compute_count_mix(df)

    # ── Setup-knockout pairs ─────────────────────────────────────────────
    setup_knockout_pairs = _compute_setup_knockout_pairs(df)

    # ── First pitch tendencies ───────────────────────────────────────────
    first_pitch = _compute_first_pitch_tendencies(df)

    # ── Putaway analysis ─────────────────────────────────────────────────
    putaway = _compute_putaway_analysis(df)

    # ── Pitch tunneling scores ───────────────────────────────────────────
    pitch_tunneling = _compute_tunneling_scores(df)

    return {
        "pitcher_id": pitcher_id,
        "transition_matrix": transition_matrix,
        "count_mix": count_mix,
        "setup_knockout_pairs": setup_knockout_pairs,
        "first_pitch": first_pitch,
        "putaway": putaway,
        "pitch_tunneling": pitch_tunneling,
    }


def _empty_sequencing_result(pitcher_id: int) -> dict:
    """Return an empty result dict when no data is available."""
    return {
        "pitcher_id": pitcher_id,
        "transition_matrix": {},
        "count_mix": {},
        "setup_knockout_pairs": [],
        "first_pitch": {"type_dist": {}, "strike_rate": 0.0},
        "putaway": {"pitch": None, "whiff_rate": 0.0, "k_rate": 0.0},
        "pitch_tunneling": {},
    }


def _compute_transition_matrix(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute P(next_pitch | current_pitch) transition probabilities."""
    transitions: dict[str, dict[str, int]] = {}

    for (_game, _ab), ab_group in df.groupby(["game_pk", "at_bat_number"]):
        pitches = ab_group.sort_values("pitch_number")["pitch_type"].tolist()
        for i in range(len(pitches) - 1):
            curr, nxt = pitches[i], pitches[i + 1]
            if curr not in transitions:
                transitions[curr] = {}
            transitions[curr][nxt] = transitions[curr].get(nxt, 0) + 1

    # Normalize to probabilities
    matrix: dict[str, dict[str, float]] = {}
    for curr, nexts in transitions.items():
        total = sum(nexts.values())
        if total > 0:
            matrix[curr] = {nxt: round(cnt / total, 3) for nxt, cnt in nexts.items()}

    return matrix


def _compute_count_mix(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute pitch-type distribution for each count state."""
    count_mix: dict[str, dict[str, float]] = {}

    for count, grp in df.groupby("count_state"):
        if count not in _ALL_COUNTS:
            continue
        type_counts = grp["pitch_type"].value_counts()
        total = type_counts.sum()
        if total > 0:
            count_mix[count] = {
                pt: round(cnt / total, 3) for pt, cnt in type_counts.items()
            }

    return count_mix


def _compute_setup_knockout_pairs(df: pd.DataFrame) -> list[dict]:
    """Find 2-pitch sequences that lead to whiffs or strikeouts.

    A "setup" pitch is the pitch before the final pitch of a 2-strike
    at-bat that results in a swinging strike or strikeout.
    """
    pairs: dict[tuple[str, str], dict] = {}

    for (_game, _ab), ab_group in df.groupby(["game_pk", "at_bat_number"]):
        pitches = ab_group.sort_values("pitch_number").reset_index(drop=True)
        if len(pitches) < 2:
            continue

        for i in range(1, len(pitches)):
            setup_pt = pitches.loc[i - 1, "pitch_type"]
            knockout_pt = pitches.loc[i, "pitch_type"]
            key = (setup_pt, knockout_pt)

            if key not in pairs:
                pairs[key] = {"total": 0, "whiffs": 0, "strikeouts": 0}

            pairs[key]["total"] += 1

            if pitches.loc[i, "description"] in _WHIFF_DESCRIPTIONS:
                pairs[key]["whiffs"] += 1

            events = pitches.loc[i, "events"]
            if isinstance(events, str) and "strikeout" in events.lower():
                pairs[key]["strikeouts"] += 1

    # Convert to list with rates
    result: list[dict] = []
    for (setup, knockout), stats in pairs.items():
        if stats["total"] < 5:
            continue
        result.append({
            "setup": setup,
            "knockout": knockout,
            "whiff_rate": round(stats["whiffs"] / stats["total"], 3),
            "k_rate": round(stats["strikeouts"] / stats["total"], 3),
            "count": stats["total"],
        })

    # Sort by whiff rate descending
    result.sort(key=lambda x: x["whiff_rate"], reverse=True)
    return result[:20]  # top 20


def _compute_first_pitch_tendencies(df: pd.DataFrame) -> dict:
    """Analyse pitch-type distribution and strike rate on first pitches."""
    first_pitches = df[df["pitch_number"] == 1]

    if first_pitches.empty:
        return {"type_dist": {}, "strike_rate": 0.0}

    type_counts = first_pitches["pitch_type"].value_counts()
    total = type_counts.sum()
    type_dist = {pt: round(cnt / total, 3) for pt, cnt in type_counts.items()} if total > 0 else {}

    strikes = first_pitches["is_strike"].sum()
    strike_rate = round(float(strikes / total), 3) if total > 0 else 0.0

    return {"type_dist": type_dist, "strike_rate": strike_rate}


def _compute_putaway_analysis(df: pd.DataFrame) -> dict:
    """Identify the primary putaway pitch on 2-strike counts."""
    two_strike = df[df["count_state"].isin(_TWO_STRIKE_COUNTS)]

    if two_strike.empty:
        return {"pitch": None, "whiff_rate": 0.0, "k_rate": 0.0}

    type_counts = two_strike["pitch_type"].value_counts()
    putaway_pitch = type_counts.index[0] if not type_counts.empty else None

    total = len(two_strike)
    whiffs = two_strike["is_whiff"].sum()
    whiff_rate = round(float(whiffs / total), 3) if total > 0 else 0.0

    # Strikeout rate on 2-strike pitches
    strikeouts = two_strike["events"].apply(
        lambda e: isinstance(e, str) and "strikeout" in e.lower()
    ).sum()
    k_rate = round(float(strikeouts / total), 3) if total > 0 else 0.0

    return {
        "pitch": putaway_pitch,
        "whiff_rate": whiff_rate,
        "k_rate": k_rate,
    }


def _compute_tunneling_scores(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compute a basic pitch-tunneling score for pitch-type pairs.

    The tunneling score measures how similar two pitch types look through
    the "decision point" (~23 feet from the plate).  We approximate this
    using the similarity of release_speed and initial trajectory (pfx_x,
    pfx_z at release).  A higher score means the pitches tunnel better.

    Score = 1 / (1 + distance), where distance combines normalised speed
    difference and movement difference.
    """
    pitch_types = df["pitch_type"].unique()
    if len(pitch_types) < 2:
        return {}

    pt_profiles: dict[str, dict[str, float]] = {}
    for pt in pitch_types:
        subset = df[df["pitch_type"] == pt]
        pt_profiles[pt] = {
            "avg_speed": float(subset["release_speed"].mean()),
            "avg_pfx_x": float(subset["pfx_x"].mean()) if "pfx_x" in subset.columns else 0.0,
            "avg_pfx_z": float(subset["pfx_z"].mean()) if "pfx_z" in subset.columns else 0.0,
        }

    tunneling: dict[str, dict[str, float]] = {}
    sorted_pts = sorted(pitch_types)
    for i in range(len(sorted_pts)):
        for j in range(i + 1, len(sorted_pts)):
            pt_a, pt_b = sorted_pts[i], sorted_pts[j]
            a, b = pt_profiles[pt_a], pt_profiles[pt_b]

            # Normalise speed diff (typically 0-15 mph range)
            speed_diff = abs(a["avg_speed"] - b["avg_speed"]) / 15.0
            # Normalise movement diff (typically 0-20 inches range)
            move_diff = (
                math.sqrt(
                    (a["avg_pfx_x"] - b["avg_pfx_x"]) ** 2
                    + (a["avg_pfx_z"] - b["avg_pfx_z"]) ** 2
                )
                / 20.0
            )

            distance = speed_diff + move_diff
            score = round(1.0 / (1.0 + distance), 3)

            key = f"{pt_a}_to_{pt_b}"
            tunneling[key] = {"early_tunnel_score": score}

    return tunneling


# ─────────────────────────────────────────────────────────────────────────────
# Sequencing anomaly detection
# ─────────────────────────────────────────────────────────────────────────────


def detect_sequencing_anomaly(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    game_pk: int,
    season: Optional[int] = None,
) -> list[dict]:
    """Compare in-game sequencing to season patterns and flag deviations.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        game_pk: Specific game to analyse.
        season: Season year for computing baseline patterns.

    Returns:
        List of alert dicts with type, severity, and message.
    """
    alerts: list[dict] = []

    # Get season patterns (baseline)
    baseline = analyze_sequencing_patterns(conn, pitcher_id, season)
    if not baseline["count_mix"] and not baseline["first_pitch"]["type_dist"]:
        return alerts  # no baseline to compare against

    # Get game data
    game_df = get_pitch_sequence(conn, pitcher_id, game_pk)
    if game_df.empty:
        return alerts

    game_df["count_state"] = game_df["balls"].astype(str) + "-" + game_df["strikes"].astype(str)
    game_df["is_strike"] = game_df["description"].isin(_STRIKE_DESCRIPTIONS)

    # ── First pitch approach ─────────────────────────────────────────────
    game_first = game_df[game_df["pitch_number"] == 1]
    if not game_first.empty and baseline["first_pitch"]["type_dist"]:
        game_first_dist = game_first["pitch_type"].value_counts(normalize=True).to_dict()
        baseline_first_dist = baseline["first_pitch"]["type_dist"]

        for pt, game_pct in game_first_dist.items():
            baseline_pct = baseline_first_dist.get(pt, 0.0)
            diff = abs(game_pct - baseline_pct)
            if diff >= 0.20 and len(game_first) >= 3:
                direction = "more" if game_pct > baseline_pct else "fewer"
                alerts.append({
                    "type": "first_pitch_shift",
                    "severity": "warning",
                    "pitch_type": pt,
                    "game_pct": round(game_pct * 100, 1),
                    "baseline_pct": round(baseline_pct * 100, 1),
                    "message": (
                        f"Throwing {direction} {pt} on first pitch: "
                        f"{game_pct * 100:.1f}% today vs {baseline_pct * 100:.1f}% season"
                    ),
                })

        # Strike rate on first pitch
        if len(game_first) >= 5:
            game_fp_strike = float(game_first["is_strike"].mean())
            baseline_fp_strike = baseline["first_pitch"]["strike_rate"]
            fp_diff = abs(game_fp_strike - baseline_fp_strike)
            if fp_diff >= 0.15:
                direction = "higher" if game_fp_strike > baseline_fp_strike else "lower"
                alerts.append({
                    "type": "first_pitch_strike_rate",
                    "severity": "info",
                    "game_value": round(game_fp_strike * 100, 1),
                    "baseline_value": round(baseline_fp_strike * 100, 1),
                    "message": (
                        f"First-pitch strike rate {direction}: "
                        f"{game_fp_strike * 100:.1f}% today vs "
                        f"{baseline_fp_strike * 100:.1f}% season"
                    ),
                })

    # ── Count-specific deviations ────────────────────────────────────────
    for count in _ALL_COUNTS:
        game_count = game_df[game_df["count_state"] == count]
        if len(game_count) < 3:
            continue

        baseline_mix = baseline["count_mix"].get(count, {})
        if not baseline_mix:
            continue

        game_mix = game_count["pitch_type"].value_counts(normalize=True).to_dict()

        for pt in set(list(game_mix.keys()) + list(baseline_mix.keys())):
            g_pct = game_mix.get(pt, 0.0)
            b_pct = baseline_mix.get(pt, 0.0)
            diff = abs(g_pct - b_pct)
            if diff >= 0.25 and (g_pct >= 0.05 or b_pct >= 0.05):
                direction = "more" if g_pct > b_pct else "fewer"
                alerts.append({
                    "type": "count_mix_shift",
                    "severity": "warning",
                    "count": count,
                    "pitch_type": pt,
                    "game_pct": round(g_pct * 100, 1),
                    "baseline_pct": round(b_pct * 100, 1),
                    "message": (
                        f"In {count} counts, throwing {direction} {pt}: "
                        f"{g_pct * 100:.1f}% today vs {b_pct * 100:.1f}% season"
                    ),
                })

    # ── Putaway strategy change ──────────────────────────────────────────
    two_strike_game = game_df[game_df["count_state"].isin(_TWO_STRIKE_COUNTS)]
    if len(two_strike_game) >= 5:
        game_putaway = two_strike_game["pitch_type"].mode()
        baseline_putaway = baseline["putaway"]["pitch"]
        if not game_putaway.empty and baseline_putaway:
            if game_putaway.iloc[0] != baseline_putaway:
                alerts.append({
                    "type": "putaway_shift",
                    "severity": "info",
                    "game_putaway": game_putaway.iloc[0],
                    "baseline_putaway": baseline_putaway,
                    "message": (
                        f"Putaway pitch shifted from {baseline_putaway} (season) "
                        f"to {game_putaway.iloc[0]} (today)"
                    ),
                })

    return alerts


# ─────────────────────────────────────────────────────────────────────────────
# Batter vulnerability analysis
# ─────────────────────────────────────────────────────────────────────────────


def get_batter_pitch_type_vulnerability(
    conn: duckdb.DuckDBPyConnection,
    batter_id: int,
    season: Optional[int] = None,
) -> dict:
    """Analyse which pitch types and locations a batter struggles against.

    Args:
        conn: Open DuckDB connection.
        batter_id: MLB player ID.
        season: Optional season year filter.

    Returns:
        Dictionary with worst/best pitch types, vulnerable zones,
        and a text recommendation.
    """
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    params: list = [batter_id]
    if season:
        params.append(season)

    query = f"""
        SELECT
            pitch_type,
            description,
            zone,
            stand,
            woba_value,
            woba_denom,
            type
        FROM pitches
        WHERE batter_id = $1
          AND pitch_type IS NOT NULL
          {season_filter}
    """
    df = conn.execute(query, params).fetchdf()

    if df.empty:
        return {
            "batter_id": batter_id,
            "worst_pitch_types": [],
            "best_pitch_types": [],
            "vulnerable_zones": [],
            "recommendation": "Insufficient data for recommendation.",
        }

    df["is_whiff"] = df["description"].isin(_WHIFF_DESCRIPTIONS)
    df["is_swing"] = df["description"].isin(_SWING_DESCRIPTIONS)
    df["is_chase"] = df["zone"].isin(_OUTSIDE_ZONES) & df["is_swing"]
    df["is_chase_opp"] = df["zone"].isin(_OUTSIDE_ZONES)

    # ── By pitch type ────────────────────────────────────────────────────
    pt_stats: list[dict] = []
    for pt, grp in df.groupby("pitch_type"):
        total = len(grp)
        if total < 10:
            continue

        swings = grp["is_swing"].sum()
        whiffs = grp["is_whiff"].sum()
        whiff_rate = float(whiffs / swings) if swings > 0 else 0.0

        # wOBA on PA-ending pitches
        pa_pitches = grp[grp["woba_denom"] > 0]
        woba_num = pa_pitches["woba_value"].sum()
        woba_den = pa_pitches["woba_denom"].sum()
        woba = float(woba_num / woba_den) if woba_den > 0 else None

        pt_stats.append({
            "pitch_type": pt,
            "woba": round(woba, 3) if woba is not None else None,
            "whiff_rate": round(whiff_rate, 3),
            "sample": total,
        })

    # Sort: worst = lowest wOBA (hardest to hit)
    pts_with_woba = [p for p in pt_stats if p["woba"] is not None]
    pts_with_woba.sort(key=lambda x: x["woba"])

    worst_pitch_types = pts_with_woba[:3]
    best_pitch_types = sorted(pts_with_woba, key=lambda x: x["woba"], reverse=True)[:3]

    # ── By zone ──────────────────────────────────────────────────────────
    # Use the predominant batter stand to map zones
    predominant_stand = df["stand"].mode().iloc[0] if not df["stand"].mode().empty else "R"
    zone_map = _ZONE_LABELS_RHB if predominant_stand == "R" else _ZONE_LABELS_LHB

    vulnerable_zones: list[dict] = []
    for zone_num, zone_desc in zone_map.items():
        zone_grp = df[df["zone"] == zone_num]
        if len(zone_grp) < 10:
            continue

        swings = zone_grp["is_swing"].sum()
        whiffs = zone_grp["is_whiff"].sum()
        whiff_rate = float(whiffs / swings) if swings > 0 else 0.0

        vulnerable_zones.append({
            "zone_desc": zone_desc,
            "zone": zone_num,
            "whiff_rate": round(whiff_rate, 3),
            "sample": int(len(zone_grp)),
        })

    # Chase zone analysis
    chase_opps = df[df["zone"].isin(_OUTSIDE_ZONES)]
    if len(chase_opps) >= 10:
        chase_swings = chase_opps["is_swing"].sum()
        chase_rate = float(chase_swings / len(chase_opps)) if len(chase_opps) > 0 else 0.0
        chase_whiffs = chase_opps["is_whiff"].sum()
        chase_whiff_rate = float(chase_whiffs / chase_swings) if chase_swings > 0 else 0.0

        vulnerable_zones.append({
            "zone_desc": "chase (outside zone)",
            "zone": -1,
            "whiff_rate": round(chase_whiff_rate, 3),
            "chase_rate": round(chase_rate, 3),
            "sample": int(len(chase_opps)),
        })

    # Sort zones by whiff rate descending
    vulnerable_zones.sort(key=lambda x: x["whiff_rate"], reverse=True)

    # ── Generate recommendation ──────────────────────────────────────────
    recommendation = _generate_vulnerability_recommendation(
        worst_pitch_types, best_pitch_types, vulnerable_zones
    )

    return {
        "batter_id": batter_id,
        "worst_pitch_types": worst_pitch_types,
        "best_pitch_types": best_pitch_types,
        "vulnerable_zones": vulnerable_zones[:5],
        "recommendation": recommendation,
    }


def _generate_vulnerability_recommendation(
    worst: list[dict],
    best: list[dict],
    zones: list[dict],
) -> str:
    """Build a natural-language recommendation from vulnerability data."""
    parts: list[str] = []

    if worst:
        top_weak = worst[0]
        parts.append(
            f"Attack with {top_weak['pitch_type']} "
            f"({top_weak['whiff_rate'] * 100:.0f}% whiff rate)"
        )

    if best:
        top_strong = best[0]
        parts.append(
            f"avoid {top_strong['pitch_type']} over the plate "
            f"(.{int(top_strong['woba'] * 1000):03d} wOBA)"
        )

    if zones:
        best_zone = zones[0]
        if best_zone["zone_desc"] != "chase (outside zone)":
            parts.append(f"target {best_zone['zone_desc']}")

    if not parts:
        return "Insufficient data for a targeted recommendation."

    return "; ".join(parts) + "."


# ─────────────────────────────────────────────────────────────────────────────
# Pitch plan recommendation
# ─────────────────────────────────────────────────────────────────────────────


def recommend_pitch_plan(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    batter_id: int,
    season: Optional[int] = None,
) -> dict:
    """Generate a matchup-specific pitching strategy recommendation.

    Cross-references the pitcher's arsenal strengths, the batter's pitch-type
    vulnerabilities, and sequencing patterns to produce actionable advice.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID of the pitcher.
        batter_id: MLB player ID of the batter.
        season: Optional season year filter.

    Returns:
        Dictionary with recommended_first_pitch, putaway_pitch, avoid,
        sequence_suggestion, and confidence level.
    """
    # ── Gather data ──────────────────────────────────────────────────────
    batter_vuln = get_batter_pitch_type_vulnerability(conn, batter_id, season)
    sequencing = analyze_sequencing_patterns(conn, pitcher_id, season)

    # Get pitcher's arsenal
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    params: list = [pitcher_id]
    if season:
        params.append(season)

    arsenal_query = f"""
        SELECT pitch_type, COUNT(*) AS cnt
        FROM pitches
        WHERE pitcher_id = $1
          AND pitch_type IS NOT NULL
          {season_filter}
        GROUP BY pitch_type
        ORDER BY cnt DESC
    """
    arsenal_df = conn.execute(arsenal_query, params).fetchdf()

    if arsenal_df.empty:
        return _empty_pitch_plan()

    pitcher_pitches = set(arsenal_df["pitch_type"].tolist())

    # Get platoon info
    platoon_params: list = [pitcher_id, batter_id]
    platoon_season_filter = "AND EXTRACT(YEAR FROM game_date) = $3" if season else ""
    if season:
        platoon_params.append(season)

    platoon_query = f"""
        SELECT DISTINCT p_throws, stand
        FROM pitches
        WHERE pitcher_id = $1 AND batter_id = $2
          {platoon_season_filter}
        LIMIT 1
    """
    platoon_df = conn.execute(platoon_query, platoon_params).fetchdf()

    platoon_advantage = False
    if not platoon_df.empty:
        p_throws = platoon_df.iloc[0]["p_throws"]
        stand = platoon_df.iloc[0]["stand"]
        platoon_advantage = (p_throws == "R" and stand == "R") or (
            p_throws == "L" and stand == "L"
        )

    # ── Determine sample size for confidence ─────────────────────────────
    total_samples = sum(
        p.get("sample", 0) for p in batter_vuln.get("worst_pitch_types", [])
    ) + sum(p.get("sample", 0) for p in batter_vuln.get("best_pitch_types", []))

    if total_samples >= 200:
        confidence = "high"
    elif total_samples >= 50:
        confidence = "medium"
    else:
        confidence = "low"

    # ── First pitch recommendation ───────────────────────────────────────
    first_pitch_rec = _recommend_first_pitch(
        sequencing, batter_vuln, pitcher_pitches
    )

    # ── Putaway pitch recommendation ─────────────────────────────────────
    putaway_rec = _recommend_putaway(
        sequencing, batter_vuln, pitcher_pitches
    )

    # ── Pitch to avoid ───────────────────────────────────────────────────
    avoid_rec = _recommend_avoid(batter_vuln, pitcher_pitches)

    # ── Sequence suggestion ──────────────────────────────────────────────
    sequence_suggestion = _build_sequence_suggestion(
        first_pitch_rec, putaway_rec, avoid_rec, sequencing
    )

    return {
        "recommended_first_pitch": first_pitch_rec,
        "putaway_pitch": putaway_rec,
        "avoid": avoid_rec,
        "sequence_suggestion": sequence_suggestion,
        "confidence": confidence,
        "platoon_advantage": platoon_advantage,
    }


def _empty_pitch_plan() -> dict:
    """Return an empty pitch plan when data is insufficient."""
    return {
        "recommended_first_pitch": {
            "type": None,
            "location": None,
            "reasoning": "Insufficient data",
        },
        "putaway_pitch": {
            "type": None,
            "location": None,
            "reasoning": "Insufficient data",
        },
        "avoid": {
            "type": None,
            "location": None,
            "reasoning": "Insufficient data",
        },
        "sequence_suggestion": "Not enough data to generate a recommendation.",
        "confidence": "low",
        "platoon_advantage": None,
    }


def _recommend_first_pitch(
    sequencing: dict,
    batter_vuln: dict,
    pitcher_pitches: set[str],
) -> dict:
    """Determine the best first-pitch option."""
    # Start with pitcher's natural first-pitch tendencies
    fp_dist = sequencing.get("first_pitch", {}).get("type_dist", {})
    fp_strike_rate = sequencing.get("first_pitch", {}).get("strike_rate", 0.0)

    # Prefer the pitcher's primary first-pitch option
    best_fp = None
    best_score = -1.0

    for pt, usage in fp_dist.items():
        if pt not in pitcher_pitches:
            continue

        score = usage  # base score from usage

        # Bonus if batter is weak against this pitch type
        for weak in batter_vuln.get("worst_pitch_types", []):
            if weak["pitch_type"] == pt:
                score += 0.3
                break

        # Penalty if batter crushes this pitch type
        for strong in batter_vuln.get("best_pitch_types", []):
            if strong["pitch_type"] == pt:
                score -= 0.2
                break

        if score > best_score:
            best_score = score
            best_fp = pt

    if best_fp is None:
        # Fall back to most-used pitch
        best_fp = max(fp_dist, key=fp_dist.get) if fp_dist else None

    # Determine location
    location = "over the plate for strike"
    reasoning_parts = []
    if fp_dist and best_fp:
        reasoning_parts.append(
            f"{fp_dist.get(best_fp, 0) * 100:.0f}% first-pitch usage"
        )
    if fp_strike_rate > 0:
        reasoning_parts.append(f"{fp_strike_rate * 100:.0f}% first-pitch strike rate")

    return {
        "type": best_fp,
        "location": location,
        "reasoning": "; ".join(reasoning_parts) if reasoning_parts else "Default approach",
    }


def _recommend_putaway(
    sequencing: dict,
    batter_vuln: dict,
    pitcher_pitches: set[str],
) -> dict:
    """Determine the best putaway pitch."""
    putaway_info = sequencing.get("putaway", {})
    pitcher_putaway = putaway_info.get("pitch")

    # Check if batter is vulnerable to the pitcher's natural putaway
    batter_weak_types = {
        p["pitch_type"] for p in batter_vuln.get("worst_pitch_types", [])
    }

    best_putaway = None
    best_reason = ""

    # Prefer a pitch that's both the pitcher's putaway AND the batter's weakness
    if pitcher_putaway and pitcher_putaway in batter_weak_types:
        best_putaway = pitcher_putaway
        whiff_rate = putaway_info.get("whiff_rate", 0)
        best_reason = (
            f"Pitcher's putaway pitch ({whiff_rate * 100:.0f}% whiff rate on 2-strike) "
            f"and batter weakness"
        )
    elif batter_weak_types & pitcher_pitches:
        # Use the batter's weakest pitch that the pitcher throws
        for weak in batter_vuln.get("worst_pitch_types", []):
            if weak["pitch_type"] in pitcher_pitches:
                best_putaway = weak["pitch_type"]
                best_reason = (
                    f"Batter's weakest pitch: "
                    f".{int(weak['woba'] * 1000):03d} wOBA, "
                    f"{weak['whiff_rate'] * 100:.0f}% whiff rate"
                    if weak["woba"] is not None
                    else f"Batter struggles vs {weak['pitch_type']}"
                )
                break
    elif pitcher_putaway:
        best_putaway = pitcher_putaway
        best_reason = f"Pitcher's primary putaway option"

    # Determine location based on pitch type
    location = "low-away"  # default putaway location
    if best_putaway in ("CU", "SL"):
        location = "low-away, below zone"
    elif best_putaway in ("CH", "FS"):
        location = "low, off the plate"
    elif best_putaway in ("FF", "SI"):
        location = "up-and-in"

    return {
        "type": best_putaway,
        "location": location,
        "reasoning": best_reason or "Default putaway approach",
    }


def _recommend_avoid(
    batter_vuln: dict,
    pitcher_pitches: set[str],
) -> dict:
    """Determine which pitch/location to avoid."""
    for strong in batter_vuln.get("best_pitch_types", []):
        if strong["pitch_type"] in pitcher_pitches:
            return {
                "type": strong["pitch_type"],
                "location": "middle",
                "reasoning": (
                    f"Batter's best pitch: "
                    f".{int(strong['woba'] * 1000):03d} wOBA"
                    if strong["woba"] is not None
                    else f"Batter excels vs {strong['pitch_type']}"
                ),
            }

    return {
        "type": None,
        "location": None,
        "reasoning": "No clear pitch to avoid (limited data)",
    }


def _build_sequence_suggestion(
    first_pitch: dict,
    putaway: dict,
    avoid: dict,
    sequencing: dict,
) -> str:
    """Build a natural-language sequence suggestion."""
    fp_type = first_pitch.get("type")
    pa_type = putaway.get("type")
    avoid_type = avoid.get("type")

    if not fp_type or not pa_type:
        return "Not enough data to generate a detailed sequence plan."

    parts: list[str] = []

    parts.append(f"Start with {fp_type} {first_pitch.get('location', 'in zone')}.")

    # Look for good tunneling pairs
    tunneling = sequencing.get("pitch_tunneling", {})
    tunnel_key_fwd = f"{fp_type}_to_{pa_type}"
    tunnel_key_rev = f"{pa_type}_to_{fp_type}"
    tunnel_info = tunneling.get(tunnel_key_fwd) or tunneling.get(tunnel_key_rev)

    if tunnel_info and tunnel_info.get("early_tunnel_score", 0) >= 0.5:
        parts.append(
            f"Then {pa_type} {putaway.get('location', 'low')} "
            f"for tunnel effect (score: {tunnel_info['early_tunnel_score']:.2f})."
        )
    else:
        parts.append(
            f"Follow with {pa_type} {putaway.get('location', 'low')} to change eye level."
        )

    parts.append(
        f"If ahead 0-2, go {pa_type} buried below the zone."
    )

    if avoid_type:
        parts.append(f"Avoid {avoid_type} over the plate.")

    return " ".join(parts)
