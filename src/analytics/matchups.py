"""
Bayesian pitcher-batter matchup analysis engine.

Uses hierarchical Bayesian models (Beta-Binomial conjugate) to estimate
matchup outcomes, naturally shrinking toward population priors when
direct matchup data is sparse (< 15 PA).

All public functions accept an open DuckDB connection as the first argument
and return plain dicts ready for serialization / dashboard rendering.
"""

from __future__ import annotations

import math
from datetime import date
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ── Constants ────────────────────────────────────────────────────────────────

LEAGUE_AVG_WOBA: float = 0.315
LEAGUE_AVG_BA: float = 0.245
LEAGUE_AVG_SLG: float = 0.400
LEAGUE_AVG_K_RATE: float = 0.225
LEAGUE_AVG_BB_RATE: float = 0.085

# Platoon adjustment factors (added to/subtracted from wOBA prior).
# Positive means advantage to that side.
PLATOON_ADJUSTMENT: float = 0.015  # ~15 pts wOBA for platoon advantage

# Prior strength for Beta-Binomial: controls how quickly data overrides prior.
# Larger = slower shrinkage toward observed data.  200 PA ≈ half-weight.
PRIOR_STRENGTH_PA: int = 200

# Minimum PA to trust raw matchup stats enough to display.
SAMPLE_WARNING_THRESHOLD: int = 15

DEFAULT_DB_PATH: Path = Path(r"C:\Users\hunte\projects\baseball\data\baseball.duckdb")

# Description values for swing / whiff classification (Statcast conventions).
WHIFF_DESCRIPTIONS: tuple[str, ...] = (
    "swinging_strike",
    "swinging_strike_blocked",
    "foul_tip",
    "missed_bunt",
)
SWING_DESCRIPTIONS: tuple[str, ...] = (
    "swinging_strike",
    "swinging_strike_blocked",
    "foul",
    "foul_tip",
    "foul_bunt",
    "missed_bunt",
    "hit_into_play",
    "hit_into_play_no_out",
    "hit_into_play_score",
)

# Zone 1-9 = strike zone 3x3 grid, 11-14 (and 0) = outside zone.
OUTSIDE_ZONE_VALUES: tuple[int, ...] = (0, 11, 12, 13, 14)

# Mapping from zone number to simplified 3x3 label.
# Statcast zones: 1-3 top row (L-R), 4-6 middle, 7-9 bottom.
ZONE_LABELS: dict[int, str] = {
    1: "high_inside",   2: "high_middle",   3: "high_outside",
    4: "mid_inside",    5: "mid_middle",    6: "mid_outside",
    7: "low_inside",    8: "low_middle",    9: "low_outside",
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division guarded against zero denominators.

    Always returns a plain Python ``float`` (never a numpy scalar) so that
    downstream ``round()`` / JSON-serialization behaves predictably.
    """
    if denominator == 0:
        return default
    return float(numerator / denominator)


def _get_default_conn() -> duckdb.DuckDBPyConnection:
    """Open a connection to the default DuckDB database.

    Ensures all schema tables exist before returning.
    """
    from src.db.schema import init_db
    return init_db(str(DEFAULT_DB_PATH))


def _compute_woba_from_events(df: pd.DataFrame) -> float:
    """Compute wOBA from a DataFrame that contains woba_value and woba_denom columns."""
    if df.empty:
        return 0.0
    woba_num = df["woba_value"].sum()
    woba_den = df["woba_denom"].sum()
    return _safe_div(woba_num, woba_den)


def _compute_ba_from_events(events: pd.Series) -> float:
    """Batting average from an events column (only rows where events is not null)."""
    ab_events = events.dropna()
    non_ab = {"walk", "hit_by_pitch", "sac_fly", "sac_bunt", "sac_fly_double_play",
              "catcher_interf", "intent_walk"}
    ab = ab_events[~ab_events.isin(non_ab)]
    hits = ab[ab.isin({"single", "double", "triple", "home_run"})]
    return _safe_div(len(hits), len(ab))


def _compute_slg_from_events(events: pd.Series) -> float:
    """Slugging percentage from an events column."""
    ab_events = events.dropna()
    non_ab = {"walk", "hit_by_pitch", "sac_fly", "sac_bunt", "sac_fly_double_play",
              "catcher_interf", "intent_walk"}
    ab = ab_events[~ab_events.isin(non_ab)]
    if len(ab) == 0:
        return 0.0
    weights = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    total_bases = sum(weights.get(e, 0) for e in ab)
    return total_bases / len(ab)


def _platoon_advantage(batter_hand: str, pitcher_hand: str) -> str:
    """Determine who has the platoon advantage.

    Opposite-hand matchups favour the batter; same-hand favour the pitcher.
    """
    if not batter_hand or not pitcher_hand:
        return "neutral"
    if batter_hand == pitcher_hand:
        return "pitcher"
    return "batter"


def _platoon_adjustment_value(batter_hand: str, pitcher_hand: str) -> float:
    """Signed wOBA adjustment for platoon splits.

    Positive = batter advantage (raises expected wOBA).
    """
    adv = _platoon_advantage(batter_hand, pitcher_hand)
    if adv == "batter":
        return PLATOON_ADJUSTMENT
    elif adv == "pitcher":
        return -PLATOON_ADJUSTMENT
    return 0.0


def _zone_label(zone: int, stand: str = "R") -> str:
    """Map a Statcast zone number to a human-readable label.

    Inside/outside depends on batter handedness.  For simplicity we label
    columns 1/4/7 as 'inside' for RHB and 'outside' for LHB.
    """
    if zone in ZONE_LABELS:
        label = ZONE_LABELS[zone]
        # Flip inside/outside for LHB.
        if stand == "L":
            label = label.replace("inside", "__tmp__").replace("outside", "inside").replace("__tmp__", "outside")
        return label
    return "outside_zone"


def _whiff_rate_from_desc(descriptions: pd.Series) -> float:
    """Compute whiff rate = swinging strikes / total swings."""
    swings = descriptions.isin(SWING_DESCRIPTIONS).sum()
    whiffs = descriptions.isin(WHIFF_DESCRIPTIONS).sum()
    return _safe_div(whiffs, swings)


def _chase_rate_from_df(df: pd.DataFrame) -> float:
    """Chase rate = swings outside zone / pitches outside zone."""
    outside = df[df["zone"].isin(OUTSIDE_ZONE_VALUES) | df["zone"].isna()]
    if outside.empty:
        return 0.0
    swings = outside["description"].isin(SWING_DESCRIPTIONS).sum()
    return _safe_div(swings, len(outside))


# ── Core Functions ───────────────────────────────────────────────────────────


def get_matchup_stats(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    batter_id: int,
) -> dict:
    """Aggregate all historical encounters between a pitcher and batter.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID of the pitcher.
        batter_id: MLB player ID of the batter.

    Returns:
        Dictionary of aggregated matchup statistics.  Returns sensible
        defaults when no matchup data exists.
    """
    df = conn.execute(
        """
        SELECT *
        FROM   pitches
        WHERE  pitcher_id = $1
          AND  batter_id  = $2
        ORDER  BY game_date, at_bat_number, pitch_number
        """,
        [pitcher_id, batter_id],
    ).fetchdf()

    result: dict = {
        "pitcher_id": pitcher_id,
        "batter_id": batter_id,
        "plate_appearances": 0,
        "pitches_seen": 0,
        "batting_avg": 0.0,
        "slug_pct": 0.0,
        "woba": 0.0,
        "strikeout_rate": 0.0,
        "walk_rate": 0.0,
        "avg_exit_velo": None,
        "avg_launch_angle": None,
        "whiff_rate": 0.0,
        "chase_rate": 0.0,
        "pitch_type_breakdown": {},
        "zone_results": {},
        "last_matchup_date": None,
        "sample_warning": True,
    }

    if df.empty:
        return result

    # ── Plate appearance counting ──
    # A plate appearance ends when an event is recorded.
    pa_df = df.dropna(subset=["events"])
    pa_count = len(pa_df)

    result["plate_appearances"] = pa_count
    result["pitches_seen"] = len(df)
    result["sample_warning"] = pa_count < SAMPLE_WARNING_THRESHOLD

    # ── Rate stats ──
    result["batting_avg"] = round(_compute_ba_from_events(pa_df["events"]), 3)
    result["slug_pct"] = round(_compute_slg_from_events(pa_df["events"]), 3)
    result["woba"] = round(_compute_woba_from_events(df), 3)

    strikeouts = (pa_df["events"] == "strikeout").sum()
    walks = pa_df["events"].isin({"walk", "intent_walk"}).sum()
    result["strikeout_rate"] = round(_safe_div(strikeouts, pa_count), 3)
    result["walk_rate"] = round(_safe_div(walks, pa_count), 3)

    # ── Batted ball stats ──
    bip = df[df["type"] == "X"]
    if not bip.empty:
        result["avg_exit_velo"] = round(bip["launch_speed"].mean(), 1) if bip["launch_speed"].notna().any() else None
        result["avg_launch_angle"] = round(bip["launch_angle"].mean(), 1) if bip["launch_angle"].notna().any() else None

    # ── Plate discipline ──
    result["whiff_rate"] = round(_whiff_rate_from_desc(df["description"]), 3)
    result["chase_rate"] = round(_chase_rate_from_df(df), 3)

    # ── Pitch type breakdown ──
    for pt, grp in df.groupby("pitch_type"):
        if pd.isna(pt):
            continue
        pt_events = grp.dropna(subset=["events"])
        pt_ba = _compute_ba_from_events(pt_events["events"]) if len(pt_events) > 0 else 0.0
        result["pitch_type_breakdown"][pt] = {
            "count": len(grp),
            "avg_velo": round(grp["release_speed"].mean(), 1) if grp["release_speed"].notna().any() else None,
            "whiff_rate": round(_whiff_rate_from_desc(grp["description"]), 3),
            "ba_against": round(pt_ba, 3),
        }

    # ── Zone results (simplified 3x3 grid) ──
    zone_df = df[df["zone"].between(1, 9)]
    for z in range(1, 10):
        z_data = zone_df[zone_df["zone"] == z]
        if z_data.empty:
            result["zone_results"][f"zone_{z}"] = {
                "pitches": 0, "swing_rate": 0.0, "contact_rate": 0.0, "ba": 0.0,
            }
            continue
        swings = z_data["description"].isin(SWING_DESCRIPTIONS).sum()
        whiffs = z_data["description"].isin(WHIFF_DESCRIPTIONS).sum()
        contact = swings - whiffs
        z_events = z_data.dropna(subset=["events"])
        result["zone_results"][f"zone_{z}"] = {
            "pitches": len(z_data),
            "swing_rate": round(_safe_div(swings, len(z_data)), 3),
            "contact_rate": round(_safe_div(contact, swings), 3),
            "ba": round(_compute_ba_from_events(z_events["events"]), 3) if len(z_events) > 0 else 0.0,
        }

    # ── Last matchup date ──
    result["last_matchup_date"] = str(df["game_date"].max()) if df["game_date"].notna().any() else None

    return result


def get_pitcher_profile(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> dict:
    """Build a comprehensive pitcher profile from pitch-level and season data.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        season: Optional season year.  Defaults to the most recent season
                with data.

    Returns:
        Dict with arsenal breakdown, platoon splits, season stats,
        release point metrics, and Stuff+.
    """
    # ── Player info ──
    player_row = conn.execute(
        "SELECT full_name, throws FROM players WHERE player_id = $1",
        [pitcher_id],
    ).fetchdf()

    name = player_row["full_name"].iloc[0] if not player_row.empty else "Unknown"
    throws = (
        player_row["throws"].iloc[0]
        if not player_row.empty and pd.notna(player_row["throws"].iloc[0])
        else "R"
    )

    # ── Pitch data ──
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    params: list = [pitcher_id]
    if season:
        params.append(season)

    pitches_df = conn.execute(
        f"""
        SELECT *
        FROM   pitches
        WHERE  pitcher_id = $1
               {season_filter}
        ORDER  BY game_date, at_bat_number, pitch_number
        """,
        params,
    ).fetchdf()

    # ── Arsenal ──
    arsenal: dict = {}
    if not pitches_df.empty:
        total_pitches = len(pitches_df)
        for pt, grp in pitches_df.groupby("pitch_type"):
            if pd.isna(pt):
                continue
            in_zone = grp[grp["zone"].between(1, 9)]
            pt_events = grp.dropna(subset=["events"])
            ba_against = _compute_ba_from_events(pt_events["events"]) if len(pt_events) > 0 else 0.0
            arsenal[pt] = {
                "usage_pct": round(len(grp) / total_pitches, 3),
                "avg_velo": round(grp["release_speed"].mean(), 1) if grp["release_speed"].notna().any() else None,
                "avg_spin": round(grp["release_spin_rate"].mean(), 0) if grp["release_spin_rate"].notna().any() else None,
                "avg_pfx_x": round(grp["pfx_x"].mean(), 2) if grp["pfx_x"].notna().any() else None,
                "avg_pfx_z": round(grp["pfx_z"].mean(), 2) if grp["pfx_z"].notna().any() else None,
                "whiff_rate": round(_whiff_rate_from_desc(grp["description"]), 3),
                "ba_against": round(ba_against, 3),
                "zone_rate": round(_safe_div(len(in_zone), len(grp)), 3),
            }

    # ── Platoon splits ──
    platoon: dict = {"vs_L": {}, "vs_R": {}}
    for hand, label in [("L", "vs_L"), ("R", "vs_R")]:
        hand_df = pitches_df[pitches_df["stand"] == hand] if not pitches_df.empty else pd.DataFrame()
        if hand_df.empty:
            platoon[label] = {"ba": 0.0, "woba": 0.0, "k_rate": 0.0, "bb_rate": 0.0}
            continue
        pa_df = hand_df.dropna(subset=["events"])
        pa_count = len(pa_df)
        ba = _compute_ba_from_events(pa_df["events"]) if pa_count > 0 else 0.0
        woba = _compute_woba_from_events(hand_df)
        ks = (pa_df["events"] == "strikeout").sum() if pa_count > 0 else 0
        bbs = pa_df["events"].isin({"walk", "intent_walk"}).sum() if pa_count > 0 else 0
        platoon[label] = {
            "ba": round(ba, 3),
            "woba": round(woba, 3),
            "k_rate": round(_safe_div(ks, pa_count), 3),
            "bb_rate": round(_safe_div(bbs, pa_count), 3),
        }

    # ── Season stats (from season_pitching_stats table) ──
    season_clause = "AND season = $2" if season else ""
    s_params: list = [pitcher_id]
    if season:
        s_params.append(season)
    season_df = conn.execute(
        f"""
        SELECT *
        FROM   season_pitching_stats
        WHERE  player_id = $1
               {season_clause}
        ORDER  BY season DESC
        LIMIT  1
        """,
        s_params,
    ).fetchdf()

    season_stats: dict = {
        "era": None, "fip": None, "xfip": None,
        "k_pct": None, "bb_pct": None, "whip": None,
    }
    stuff_plus: Optional[float] = None
    if not season_df.empty:
        row = season_df.iloc[0]
        season_stats = {
            "era": _maybe_round(row.get("era"), 2),
            "fip": _maybe_round(row.get("fip"), 2),
            "xfip": _maybe_round(row.get("xfip"), 2),
            "k_pct": _maybe_round(row.get("k_pct"), 3),
            "bb_pct": _maybe_round(row.get("bb_pct"), 3),
            "whip": _maybe_round(row.get("whip"), 2),
        }
        stuff_plus = _maybe_round(row.get("stuff_plus"), 1)

    # ── Release point ──
    release_x: Optional[float] = None
    release_z: Optional[float] = None
    release_consistency: Optional[float] = None
    if not pitches_df.empty and "release_pos_x" in pitches_df.columns:
        rpx = pitches_df["release_pos_x"].dropna()
        rpz = pitches_df["release_pos_z"].dropna()
        if len(rpx) > 0 and len(rpz) > 0:
            release_x = round(rpx.mean(), 2)
            release_z = round(rpz.mean(), 2)
            # RMSE of release point relative to mean (consistency measure).
            rmse_x = np.sqrt(((rpx - rpx.mean()) ** 2).mean())
            rmse_z = np.sqrt(((rpz - rpz.mean()) ** 2).mean())
            release_consistency = round(math.sqrt(rmse_x**2 + rmse_z**2), 3)

    # ── Pitch count today (for live game support) ──
    today_str = date.today().isoformat()
    today_count_df = conn.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM   pitches
        WHERE  pitcher_id = $1
          AND  game_date = CAST($2 AS DATE)
        """,
        [pitcher_id, today_str],
    ).fetchdf()
    pitch_count_today = int(today_count_df["cnt"].iloc[0]) if not today_count_df.empty else 0

    return {
        "pitcher_id": pitcher_id,
        "name": name,
        "throws": throws,
        "arsenal": arsenal,
        "platoon_splits": platoon,
        "season_stats": season_stats,
        "stuff_plus": stuff_plus,
        "avg_release_point": {"x": release_x, "z": release_z},
        "release_point_consistency": release_consistency,
        "pitch_count_today": pitch_count_today,
    }


def get_batter_profile(
    conn: duckdb.DuckDBPyConnection,
    batter_id: int,
    season: Optional[int] = None,
) -> dict:
    """Build a comprehensive batter profile.

    Args:
        conn: Open DuckDB connection.
        batter_id: MLB player ID.
        season: Optional season year filter.

    Returns:
        Dict with season stats, Statcast metrics, plate discipline,
        pitch-type performance, and platoon splits.
    """
    # ── Player info ──
    player_row = conn.execute(
        "SELECT full_name, bats FROM players WHERE player_id = $1",
        [batter_id],
    ).fetchdf()

    name = player_row["full_name"].iloc[0] if not player_row.empty else "Unknown"
    bats = (
        player_row["bats"].iloc[0]
        if not player_row.empty and pd.notna(player_row["bats"].iloc[0])
        else "R"
    )

    # ── Pitch-level data ──
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    params: list = [batter_id]
    if season:
        params.append(season)

    pitches_df = conn.execute(
        f"""
        SELECT *
        FROM   pitches
        WHERE  batter_id = $1
               {season_filter}
        """,
        params,
    ).fetchdf()

    # ── Season stats from season_batting_stats ──
    season_clause = "AND season = $2" if season else ""
    s_params: list = [batter_id]
    if season:
        s_params.append(season)
    season_df = conn.execute(
        f"""
        SELECT *
        FROM   season_batting_stats
        WHERE  player_id = $1
               {season_clause}
        ORDER  BY season DESC
        LIMIT  1
        """,
        s_params,
    ).fetchdf()

    season_stats: dict = {
        "ba": None, "obp": None, "slg": None, "woba": None,
        "wrc_plus": None, "war": None,
    }
    if not season_df.empty:
        row = season_df.iloc[0]
        season_stats = {
            "ba": _maybe_round(row.get("ba"), 3),
            "obp": _maybe_round(row.get("obp"), 3),
            "slg": _maybe_round(row.get("slg"), 3),
            "woba": _maybe_round(row.get("woba"), 3),
            "wrc_plus": _maybe_round(row.get("wrc_plus"), 1),
            "war": _maybe_round(row.get("war"), 1),
        }

    # ── Statcast metrics (computed from pitch data) ──
    bip = pitches_df[pitches_df["type"] == "X"] if not pitches_df.empty else pd.DataFrame()
    statcast: dict = {
        "avg_exit_velo": None, "avg_launch_angle": None,
        "barrel_pct": None, "hard_hit_pct": None,
        "xba": None, "xslg": None, "xwoba": None,
    }
    if not bip.empty:
        statcast["avg_exit_velo"] = round(bip["launch_speed"].mean(), 1) if bip["launch_speed"].notna().any() else None
        statcast["avg_launch_angle"] = round(bip["launch_angle"].mean(), 1) if bip["launch_angle"].notna().any() else None
        # Barrel: EV >= 98 AND 26 <= LA <= 30 (simplified — Statcast uses a sliding scale)
        if bip["launch_speed"].notna().any() and bip["launch_angle"].notna().any():
            barrels = bip[
                (bip["launch_speed"] >= 98)
                & (bip["launch_angle"] >= 26)
                & (bip["launch_angle"] <= 30)
            ]
            statcast["barrel_pct"] = round(_safe_div(len(barrels), len(bip)), 3)
            hard_hit = bip[bip["launch_speed"] >= 95]
            statcast["hard_hit_pct"] = round(_safe_div(len(hard_hit), len(bip)), 3)

    # xBA / xSLG / xwOBA from season stats if available.
    if not season_df.empty:
        row = season_df.iloc[0]
        statcast["xba"] = _maybe_round(row.get("xba"), 3)
        statcast["xslg"] = _maybe_round(row.get("xslg"), 3)
        statcast["xwoba"] = _maybe_round(row.get("xwoba"), 3)

    # ── Plate discipline ──
    plate_discipline: dict = {
        "chase_rate": 0.0, "zone_swing_rate": 0.0,
        "whiff_rate": 0.0, "zone_contact_rate": 0.0,
    }
    if not pitches_df.empty:
        plate_discipline["whiff_rate"] = round(_whiff_rate_from_desc(pitches_df["description"]), 3)
        plate_discipline["chase_rate"] = round(_chase_rate_from_df(pitches_df), 3)

        in_zone = pitches_df[pitches_df["zone"].between(1, 9)]
        if not in_zone.empty:
            z_swings = in_zone["description"].isin(SWING_DESCRIPTIONS).sum()
            plate_discipline["zone_swing_rate"] = round(_safe_div(z_swings, len(in_zone)), 3)
            z_whiffs = in_zone["description"].isin(WHIFF_DESCRIPTIONS).sum()
            z_contact = z_swings - z_whiffs
            plate_discipline["zone_contact_rate"] = round(_safe_div(z_contact, z_swings), 3)

    # ── Vs pitch types ──
    vs_pitch_types: dict = {}
    if not pitches_df.empty:
        for pt, grp in pitches_df.groupby("pitch_type"):
            if pd.isna(pt):
                continue
            pt_events = grp.dropna(subset=["events"])
            pa_count = len(pt_events)
            ba = _compute_ba_from_events(pt_events["events"]) if pa_count > 0 else 0.0
            woba = _compute_woba_from_events(grp)
            vs_pitch_types[pt] = {
                "pa": pa_count,
                "ba": round(ba, 3),
                "woba": round(woba, 3),
                "whiff_rate": round(_whiff_rate_from_desc(grp["description"]), 3),
            }

    # ── Platoon splits ──
    platoon: dict = {"vs_L": {}, "vs_R": {}}
    for hand, label in [("L", "vs_L"), ("R", "vs_R")]:
        hand_df = pitches_df[pitches_df["p_throws"] == hand] if not pitches_df.empty else pd.DataFrame()
        if hand_df.empty:
            platoon[label] = {"ba": 0.0, "woba": 0.0, "ops": 0.0}
            continue
        pa_df = hand_df.dropna(subset=["events"])
        pa_count = len(pa_df)
        ba = _compute_ba_from_events(pa_df["events"]) if pa_count > 0 else 0.0
        slg = _compute_slg_from_events(pa_df["events"]) if pa_count > 0 else 0.0
        woba = _compute_woba_from_events(hand_df)
        # OBP approximation: (H + BB + HBP) / PA
        hits = pa_df["events"].isin({"single", "double", "triple", "home_run"}).sum() if pa_count > 0 else 0
        bbs = pa_df["events"].isin({"walk", "intent_walk", "hit_by_pitch"}).sum() if pa_count > 0 else 0
        obp = _safe_div(hits + bbs, pa_count)
        platoon[label] = {
            "ba": round(ba, 3),
            "woba": round(woba, 3),
            "ops": round(obp + slg, 3),
        }

    return {
        "batter_id": batter_id,
        "name": name,
        "bats": bats,
        "season_stats": season_stats,
        "statcast": statcast,
        "plate_discipline": plate_discipline,
        "vs_pitch_types": vs_pitch_types,
        "platoon_splits": platoon,
    }


# ── Bayesian Matchup Estimation ─────────────────────────────────────────────


def _log5_estimate(batter_woba: float, pitcher_woba_against: float,
                   league_avg: float = LEAGUE_AVG_WOBA) -> float:
    """Classic log5 formula for matchup estimation.

    log5(B, P) = (B * P / L) / (B * P / L + (1 - B) * (1 - P) / (1 - L))

    This is the Odds Ratio method; we adapt it for wOBA by treating wOBA
    as a probability-like rate (valid for rates bounded roughly 0-0.5).
    """
    b = max(0.001, min(batter_woba, 0.600))
    p = max(0.001, min(pitcher_woba_against, 0.600))
    l_ = max(0.001, league_avg)

    numerator = (b * p) / l_
    denominator = numerator + ((1 - b) * (1 - p)) / (1 - l_)
    return _safe_div(numerator, denominator, default=l_)


def _get_batter_woba(conn: duckdb.DuckDBPyConnection, batter_id: int) -> float:
    """Retrieve the batter's season wOBA, falling back to pitch-level calculation."""
    row = conn.execute(
        "SELECT woba FROM season_batting_stats WHERE player_id = $1 ORDER BY season DESC LIMIT 1",
        [batter_id],
    ).fetchdf()
    if not row.empty and pd.notna(row["woba"].iloc[0]):
        return float(row["woba"].iloc[0])

    # Fallback: compute from pitch data.
    pitch_woba = conn.execute(
        """
        SELECT SUM(woba_value) / NULLIF(SUM(woba_denom), 0) AS woba
        FROM   pitches
        WHERE  batter_id = $1
        """,
        [batter_id],
    ).fetchdf()
    if not pitch_woba.empty and pd.notna(pitch_woba["woba"].iloc[0]):
        return float(pitch_woba["woba"].iloc[0])
    return LEAGUE_AVG_WOBA


def _get_pitcher_woba_against(conn: duckdb.DuckDBPyConnection, pitcher_id: int) -> float:
    """Retrieve the pitcher's season wOBA-against, falling back to pitch-level calculation."""
    row = conn.execute(
        "SELECT xwoba FROM season_pitching_stats WHERE player_id = $1 ORDER BY season DESC LIMIT 1",
        [pitcher_id],
    ).fetchdf()
    if not row.empty and pd.notna(row["xwoba"].iloc[0]):
        return float(row["xwoba"].iloc[0])

    # Fallback: compute from pitch data.
    pitch_woba = conn.execute(
        """
        SELECT SUM(woba_value) / NULLIF(SUM(woba_denom), 0) AS woba
        FROM   pitches
        WHERE  pitcher_id = $1
        """,
        [pitcher_id],
    ).fetchdf()
    if not pitch_woba.empty and pd.notna(pitch_woba["woba"].iloc[0]):
        return float(pitch_woba["woba"].iloc[0])
    return LEAGUE_AVG_WOBA


def _get_handedness(conn: duckdb.DuckDBPyConnection, player_id: int,
                    role: str = "batter") -> str:
    """Retrieve batter's stand or pitcher's throwing hand."""
    if role == "batter":
        row = conn.execute(
            "SELECT bats FROM players WHERE player_id = $1", [player_id],
        ).fetchdf()
        return row["bats"].iloc[0] if not row.empty and pd.notna(row["bats"].iloc[0]) else "R"
    else:
        row = conn.execute(
            "SELECT throws FROM players WHERE player_id = $1", [player_id],
        ).fetchdf()
        return row["throws"].iloc[0] if not row.empty and pd.notna(row["throws"].iloc[0]) else "R"


def _find_key_pitch_vulnerability(
    conn: duckdb.DuckDBPyConnection,
    batter_id: int,
    pitcher_id: int,
) -> Optional[str]:
    """Identify the batter's worst pitch-type matchup from the pitcher's arsenal.

    Looks at the pitcher's primary pitch types and the batter's performance
    against each.  Returns a human-readable summary of the most exploitable
    weakness, or None.
    """
    # Pitcher's arsenal (top pitch types).
    arsenal_df = conn.execute(
        """
        SELECT pitch_type, COUNT(*) AS cnt
        FROM   pitches
        WHERE  pitcher_id = $1
          AND  pitch_type IS NOT NULL
        GROUP  BY pitch_type
        HAVING COUNT(*) >= 20
        ORDER  BY cnt DESC
        """,
        [pitcher_id],
    ).fetchdf()

    if arsenal_df.empty:
        return None

    # Batter's performance vs each pitch type (across all pitchers).
    worst_pitch: Optional[str] = None
    worst_score: float = float("inf")  # lower is worse for batter

    for _, arow in arsenal_df.iterrows():
        pt = arow["pitch_type"]
        batter_vs = conn.execute(
            """
            SELECT SUM(CASE WHEN description IN
                        ('swinging_strike', 'swinging_strike_blocked', 'missed_bunt')
                        THEN 1 ELSE 0 END) AS whiffs,
                   SUM(CASE WHEN description IN
                        ('swinging_strike', 'swinging_strike_blocked',
                         'foul', 'foul_tip', 'foul_bunt',
                         'hit_into_play', 'hit_into_play_no_out', 'hit_into_play_score')
                        THEN 1 ELSE 0 END) AS swings,
                   SUM(CASE WHEN events IN ('single','double','triple','home_run')
                        THEN 1 ELSE 0 END) AS hits,
                   SUM(CASE WHEN events IS NOT NULL
                             AND events NOT IN ('walk','hit_by_pitch','intent_walk',
                                                'sac_fly','sac_bunt','catcher_interf')
                        THEN 1 ELSE 0 END) AS ab
            FROM   pitches
            WHERE  batter_id = $1
              AND  pitch_type = $2
            """,
            [batter_id, pt],
        ).fetchdf()

        if batter_vs.empty:
            continue
        r = batter_vs.iloc[0]
        ab = int(r["ab"]) if pd.notna(r["ab"]) else 0
        if ab < 10:
            continue
        hits = int(r["hits"]) if pd.notna(r["hits"]) else 0
        whiffs = int(r["whiffs"]) if pd.notna(r["whiffs"]) else 0
        swings = int(r["swings"]) if pd.notna(r["swings"]) else 0
        ba = _safe_div(hits, ab)
        whiff_rate = _safe_div(whiffs, swings)

        # Score: lower BA + higher whiff = worse for batter.
        score = ba - whiff_rate  # more negative = more vulnerable
        if score < worst_score:
            worst_score = score
            worst_pitch = (
                f"Weak vs {pt}: .{int(ba * 1000):03d} BA, "
                f"{int(whiff_rate * 100)}% whiff"
            )

    return worst_pitch


def estimate_matchup_woba(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    batter_id: int,
    method: str = "bayesian",
) -> dict:
    """Estimate the expected wOBA for a pitcher-batter matchup.

    Two methods are supported:

    **log5** -- The classic Odds Ratio formula:
        matchup = (B * P / L) / (B*P/L + (1-B)*(1-P)/(1-L))

    **bayesian** -- Beta-Binomial conjugate model.  The prior is derived
    from the log5 estimate (informed by overall batter/pitcher quality
    and platoon adjustment).  The prior strength is controlled by
    ``PRIOR_STRENGTH_PA``.  With zero historical matchup data the
    posterior equals the prior; as matchup plate appearances grow the
    posterior converges to the observed matchup rate.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID of the pitcher.
        batter_id: MLB player ID of the batter.
        method: ``"bayesian"`` (default) or ``"log5"``.

    Returns:
        Dict with point estimate, credible interval, and contextual metadata.
    """
    if method not in ("bayesian", "log5"):
        raise ValueError(f"Unknown method '{method}'; use 'bayesian' or 'log5'.")

    batter_woba = _get_batter_woba(conn, batter_id)
    pitcher_woba_ag = _get_pitcher_woba_against(conn, pitcher_id)

    batter_hand = _get_handedness(conn, batter_id, role="batter")
    pitcher_hand = _get_handedness(conn, pitcher_id, role="pitcher")
    platoon_adj = _platoon_adjustment_value(batter_hand, pitcher_hand)
    platoon_adv = _platoon_advantage(batter_hand, pitcher_hand)

    # Prior: log5 estimate with platoon adjustment.
    prior_woba = _log5_estimate(batter_woba, pitcher_woba_ag)
    prior_woba = max(0.001, min(prior_woba + platoon_adj, 0.600))

    # ── Historical matchup data ──
    matchup_df = conn.execute(
        """
        SELECT woba_value, woba_denom
        FROM   pitches
        WHERE  pitcher_id = $1
          AND  batter_id  = $2
          AND  woba_denom > 0
        """,
        [pitcher_id, batter_id],
    ).fetchdf()

    matchup_pa = int(matchup_df["woba_denom"].sum()) if not matchup_df.empty else 0
    actual_woba: Optional[float] = None
    if matchup_pa > 0:
        actual_woba = round(
            _safe_div(matchup_df["woba_value"].sum(), matchup_df["woba_denom"].sum()),
            3,
        )

    if method == "log5":
        # Pure log5 — no Bayesian updating.
        return {
            "estimated_woba": round(prior_woba, 3),
            "confidence_interval": (
                round(max(0, prior_woba - 0.060), 3),
                round(min(0.600, prior_woba + 0.060), 3),
            ),
            "prior_woba": round(prior_woba, 3),
            "actual_woba": actual_woba,
            "matchup_pa": matchup_pa,
            "method": "log5",
            "platoon_advantage": platoon_adv,
            "key_pitch_vulnerability": _find_key_pitch_vulnerability(conn, batter_id, pitcher_id),
        }

    # ── Bayesian Beta-Binomial approach ──
    # We model "success" as the batter achieving wOBA-equivalent outcomes.
    # The Beta prior is parameterised so that:
    #   E[prior] = prior_woba
    #   prior_strength = PRIOR_STRENGTH_PA (pseudo-observations)
    #
    # alpha_prior = prior_woba * PRIOR_STRENGTH_PA
    # beta_prior  = (1 - prior_woba) * PRIOR_STRENGTH_PA
    #
    # Posterior after observing matchup_woba_value / matchup_woba_denom:
    #   alpha_post = alpha_prior + sum(woba_value)
    #   beta_post  = beta_prior + (matchup_pa - sum(woba_value))

    alpha_prior = prior_woba * PRIOR_STRENGTH_PA
    beta_prior = (1 - prior_woba) * PRIOR_STRENGTH_PA

    if matchup_pa > 0:
        observed_success = matchup_df["woba_value"].sum()
        observed_failure = matchup_pa - observed_success
        # NOTE: wOBA is not a true proportion -- woba_value is a weighted
        # sum that can exceed woba_denom when the batter produces many
        # extra-base hits (weights: 1B=0.88, 2B=1.24, 3B=1.56, HR=2.01).
        # Clamping to zero prevents negative Beta parameters.  For typical
        # MLB wOBA ranges (~0.250-0.400) this approximation is adequate.
        alpha_post = alpha_prior + max(0, observed_success)
        beta_post = beta_prior + max(0, observed_failure)
    else:
        alpha_post = alpha_prior
        beta_post = beta_prior

    # Posterior mean.
    posterior_mean = alpha_post / (alpha_post + beta_post)

    # 90% credible interval from the Beta posterior.
    ci_low = sp_stats.beta.ppf(0.05, alpha_post, beta_post)
    ci_high = sp_stats.beta.ppf(0.95, alpha_post, beta_post)

    return {
        "estimated_woba": round(float(posterior_mean), 3),
        "confidence_interval": (round(float(ci_low), 3), round(float(ci_high), 3)),
        "prior_woba": round(prior_woba, 3),
        "actual_woba": actual_woba,
        "matchup_pa": matchup_pa,
        "method": "bayesian",
        "platoon_advantage": platoon_adv,
        "key_pitch_vulnerability": _find_key_pitch_vulnerability(conn, batter_id, pitcher_id),
    }


# ── Composite Functions ──────────────────────────────────────────────────────


def generate_matchup_card(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    batter_id: int,
) -> dict:
    """Generate a complete matchup card combining all analyses.

    Calls :func:`get_pitcher_profile`, :func:`get_batter_profile`,
    :func:`get_matchup_stats`, and :func:`estimate_matchup_woba` then
    assembles them into a single dict suitable for rendering on the
    dashboard.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID of the pitcher.
        batter_id: MLB player ID of the batter.

    Returns:
        Comprehensive matchup card dict.
    """
    pitcher_profile = get_pitcher_profile(conn, pitcher_id)
    batter_profile = get_batter_profile(conn, batter_id)
    matchup_stats = get_matchup_stats(conn, pitcher_id, batter_id)
    matchup_estimate = estimate_matchup_woba(conn, pitcher_id, batter_id, method="bayesian")

    return {
        "pitcher": pitcher_profile,
        "batter": batter_profile,
        "matchup_history": matchup_stats,
        "matchup_estimate": matchup_estimate,
    }


def get_lineup_matchups(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    lineup: list[int],
) -> list[dict]:
    """Generate matchup cards for an entire batting lineup vs one pitcher.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID of the pitcher.
        lineup: Ordered list of batter MLB player IDs.

    Returns:
        List of matchup dicts sorted by estimated wOBA (ascending = best
        for pitcher, descending = best for batters).  ``most_favorable``
        and ``most_unfavorable`` flags highlight the extreme matchups.
    """
    cards: list[dict] = []
    for order_idx, batter_id in enumerate(lineup, start=1):
        card = generate_matchup_card(conn, pitcher_id, batter_id)
        card["lineup_order"] = order_idx
        card["batter_id"] = batter_id
        card["estimated_woba"] = card["matchup_estimate"]["estimated_woba"]
        cards.append(card)

    # Sort by estimated wOBA descending (most dangerous batters first).
    cards.sort(key=lambda c: c["estimated_woba"], reverse=True)

    # Tag top/bottom 3.
    for i, card in enumerate(cards):
        card["most_favorable_for_batter"] = i < 3
        card["most_unfavorable_for_batter"] = i >= len(cards) - 3

    return cards


# ── Private utility ──────────────────────────────────────────────────────────


def _maybe_round(value, digits: int = 3):
    """Round a value if it is not None/NaN, otherwise return None."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(value), digits)
