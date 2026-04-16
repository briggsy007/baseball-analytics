"""
Bullpen strategy and fatigue analysis engine.

Provides workload tracking, Leverage Index computation, and reliever
recommendation logic that integrates with the Bayesian matchup model
in :mod:`src.analytics.matchups`.

All public functions accept an open DuckDB connection (where DB access is
needed) and return plain dicts/lists suitable for serialization.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from src.analytics.matchups import (
    LEAGUE_AVG_WOBA,
    _get_handedness,
    _platoon_advantage,
    _safe_div,
    estimate_matchup_woba,
)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_DB_PATH: Path = Path(r"C:\Users\hunte\projects\baseball\data\baseball.duckdb")

# Fatigue thresholds.
MAX_CONSECUTIVE_DAYS: int = 3           # unavailable after this many straight days
HIGH_PITCH_RECENT: int = 25             # 25+ pitches yesterday → tired
MODERATE_PITCH_3DAY: int = 40           # 40+ pitches in 3 days → moderate
FRESH_PITCH_7DAY: int = 30              # < 30 pitches in 7 days & 2+ days rest → fresh

# Reliever role pitch-count heuristics.
CLOSER_MIN_SAVES: int = 5
SETUP_MIN_HOLDS: int = 3

# ── Leverage Index lookup ────────────────────────────────────────────────────
#
# Simplified Leverage Index table based on Tom Tango's research.
# Keys: (inning_bucket, outs, base_state_code, score_bucket) → LI
#
# inning_bucket: "early" (1-3), "mid" (4-6), "late" (7-8), "final" (9+)
# base_state_code: 3-digit binary string, e.g. "100" = runner on 1B only
# score_bucket: clamped to [-3, +3] (negative = trailing)

_INNING_BUCKET_MAP: dict[int, str] = {}
for _i in range(1, 4):
    _INNING_BUCKET_MAP[_i] = "early"
for _i in range(4, 7):
    _INNING_BUCKET_MAP[_i] = "mid"
for _i in range(7, 9):
    _INNING_BUCKET_MAP[_i] = "late"
# 9+ handled in function


def _inning_bucket(inning: int) -> str:
    if inning >= 9:
        return "final"
    return _INNING_BUCKET_MAP.get(inning, "mid")


def _base_state_code(runners: dict) -> str:
    """Convert a runners dict to '000'-'111'.

    Accepts keys: ``{1: bool}``, ``{"1b": bool}``, ``{"first": bool}``,
    ``{"1B": bool}``, or ``{"on_1b": bool}`` (and similarly for 2nd/3rd).
    """
    return (
        ("1" if runners.get(1, False) or runners.get("1b", False)
               or runners.get("first", False) or runners.get("1B", False)
               or runners.get("on_1b", False) else "0")
        + ("1" if runners.get(2, False) or runners.get("2b", False)
                 or runners.get("second", False) or runners.get("2B", False)
                 or runners.get("on_2b", False) else "0")
        + ("1" if runners.get(3, False) or runners.get("3b", False)
                 or runners.get("third", False) or runners.get("3B", False)
                 or runners.get("on_3b", False) else "0")
    )


def _score_bucket(score_diff: int) -> int:
    return max(-3, min(3, score_diff))


# Pre-computed LI table (subset of Tango's values, linearly interpolated for
# states not explicitly listed).  Full table has ~672 entries; we store the
# most impactful states and default to 1.0.
_LI_TABLE: dict[tuple[str, int, str, int], float] = {
    # ── Final inning (9+), high-leverage situations ──
    ("final", 0, "000", 0): 1.9,
    ("final", 0, "000", -1): 2.8,
    ("final", 0, "100", 0): 2.4,
    ("final", 0, "100", -1): 3.3,
    ("final", 0, "110", 0): 2.9,
    ("final", 0, "110", -1): 3.8,
    ("final", 0, "111", 0): 3.4,
    ("final", 0, "111", -1): 4.3,
    ("final", 1, "000", 0): 1.6,
    ("final", 1, "000", -1): 2.3,
    ("final", 1, "100", 0): 2.1,
    ("final", 1, "100", -1): 2.9,
    ("final", 1, "110", 0): 2.6,
    ("final", 1, "110", -1): 3.4,
    ("final", 1, "111", 0): 3.0,
    ("final", 1, "111", -1): 3.9,
    ("final", 2, "000", 0): 1.0,
    ("final", 2, "000", -1): 1.8,
    ("final", 2, "100", 0): 1.5,
    ("final", 2, "100", -1): 2.3,
    ("final", 2, "110", 0): 2.0,
    ("final", 2, "110", -1): 2.8,
    ("final", 2, "111", 0): 2.3,
    ("final", 2, "111", -1): 3.2,
    # Large leads in final inning — low leverage.
    ("final", 0, "000", 3): 0.2,
    ("final", 1, "000", 3): 0.2,
    ("final", 2, "000", 3): 0.1,
    ("final", 0, "000", 2): 0.6,
    ("final", 1, "000", 2): 0.5,
    ("final", 2, "000", 2): 0.3,
    ("final", 0, "000", 1): 1.2,
    ("final", 1, "000", 1): 1.0,
    ("final", 2, "000", 1): 0.6,

    # ── Late innings (7-8) ──
    ("late", 0, "000", 0): 1.5,
    ("late", 0, "000", -1): 2.0,
    ("late", 0, "100", 0): 2.0,
    ("late", 0, "100", -1): 2.6,
    ("late", 0, "110", 0): 2.4,
    ("late", 0, "110", -1): 3.0,
    ("late", 0, "111", 0): 2.8,
    ("late", 0, "111", -1): 3.5,
    ("late", 1, "000", 0): 1.3,
    ("late", 1, "000", -1): 1.7,
    ("late", 1, "100", 0): 1.7,
    ("late", 1, "100", -1): 2.2,
    ("late", 1, "110", 0): 2.0,
    ("late", 1, "110", -1): 2.6,
    ("late", 1, "111", 0): 2.3,
    ("late", 1, "111", -1): 3.0,
    ("late", 2, "000", 0): 0.8,
    ("late", 2, "000", -1): 1.3,
    ("late", 2, "100", 0): 1.2,
    ("late", 2, "100", -1): 1.8,
    ("late", 2, "110", 0): 1.5,
    ("late", 2, "110", -1): 2.2,
    ("late", 2, "111", 0): 1.8,
    ("late", 2, "111", -1): 2.5,
    ("late", 0, "000", 1): 1.0,
    ("late", 1, "000", 1): 0.8,
    ("late", 2, "000", 1): 0.5,
    ("late", 0, "000", 2): 0.5,
    ("late", 1, "000", 2): 0.4,
    ("late", 2, "000", 2): 0.3,
    ("late", 0, "000", 3): 0.2,
    ("late", 1, "000", 3): 0.2,
    ("late", 2, "000", 3): 0.1,

    # ── Mid innings (4-6) ──
    ("mid", 0, "000", 0): 1.0,
    ("mid", 0, "000", -1): 1.2,
    ("mid", 0, "100", 0): 1.3,
    ("mid", 0, "100", -1): 1.6,
    ("mid", 0, "110", 0): 1.6,
    ("mid", 0, "110", -1): 1.9,
    ("mid", 0, "111", 0): 1.9,
    ("mid", 0, "111", -1): 2.2,
    ("mid", 1, "000", 0): 0.9,
    ("mid", 1, "000", -1): 1.0,
    ("mid", 1, "100", 0): 1.2,
    ("mid", 1, "100", -1): 1.4,
    ("mid", 1, "110", 0): 1.4,
    ("mid", 1, "110", -1): 1.7,
    ("mid", 1, "111", 0): 1.6,
    ("mid", 1, "111", -1): 1.9,
    ("mid", 2, "000", 0): 0.6,
    ("mid", 2, "000", -1): 0.8,
    ("mid", 2, "100", 0): 0.9,
    ("mid", 2, "100", -1): 1.1,
    ("mid", 2, "110", 0): 1.1,
    ("mid", 2, "110", -1): 1.3,
    ("mid", 2, "111", 0): 1.3,
    ("mid", 2, "111", -1): 1.5,
    ("mid", 0, "000", 1): 0.8,
    ("mid", 1, "000", 1): 0.7,
    ("mid", 2, "000", 1): 0.5,
    ("mid", 0, "000", 2): 0.5,
    ("mid", 1, "000", 2): 0.4,
    ("mid", 2, "000", 2): 0.3,
    ("mid", 0, "000", 3): 0.3,
    ("mid", 1, "000", 3): 0.2,
    ("mid", 2, "000", 3): 0.1,

    # ── Early innings (1-3) — generally low leverage ──
    ("early", 0, "000", 0): 0.9,
    ("early", 0, "100", 0): 1.1,
    ("early", 0, "110", 0): 1.3,
    ("early", 0, "111", 0): 1.5,
    ("early", 1, "000", 0): 0.7,
    ("early", 1, "100", 0): 0.9,
    ("early", 1, "110", 0): 1.1,
    ("early", 1, "111", 0): 1.3,
    ("early", 2, "000", 0): 0.5,
    ("early", 2, "100", 0): 0.7,
    ("early", 2, "110", 0): 0.9,
    ("early", 2, "111", 0): 1.0,
    ("early", 0, "000", -1): 1.0,
    ("early", 1, "000", -1): 0.8,
    ("early", 2, "000", -1): 0.6,
    ("early", 0, "000", 1): 0.7,
    ("early", 1, "000", 1): 0.6,
    ("early", 2, "000", 1): 0.4,
    ("early", 0, "000", 2): 0.4,
    ("early", 0, "000", 3): 0.2,
}


# ── Public Functions ─────────────────────────────────────────────────────────


def calculate_leverage_index(
    inning: int,
    outs: int,
    runners: dict,
    score_diff: int,
) -> float:
    """Compute the Leverage Index for a game situation.

    Uses a simplified lookup table derived from Tom Tango's LI research.
    States not found in the table return a heuristic interpolation.

    Args:
        inning: Current inning (1-based).  9+ treated as "final".
        outs: Number of outs (0, 1, or 2).
        runners: Runner positions.  Accepts ``{1: True, 2: False, 3: True}``
                 or ``{"1b": True, "2b": False, "3b": True}``.
        score_diff: Score differential from the perspective of the
                    pitching team.  Positive = leading; negative = trailing.

    Returns:
        Leverage Index as a float (1.0 = average situation).
    """
    ib = _inning_bucket(inning)
    bs = _base_state_code(runners)
    sb = _score_bucket(score_diff)
    outs = max(0, min(2, outs))

    key = (ib, outs, bs, sb)
    if key in _LI_TABLE:
        return _LI_TABLE[key]

    # Fallback: try with empty bases at the same score bucket.
    fallback_key = (ib, outs, "000", sb)
    if fallback_key in _LI_TABLE:
        # Add a runner premium proportional to number of runners.
        n_runners = bs.count("1")
        base_li = _LI_TABLE[fallback_key]
        return round(base_li + n_runners * 0.35, 1)

    # Second fallback: try tie game at this inning/out state.
    tie_key = (ib, outs, "000", 0)
    if tie_key in _LI_TABLE:
        base_li = _LI_TABLE[tie_key]
        # Adjust for score: trailing raises LI, leading lowers it.
        adjustment = -0.15 * sb  # sb > 0 → leading → lower LI
        n_runners = bs.count("1")
        return round(max(0.1, base_li + adjustment + n_runners * 0.35), 1)

    # Ultimate fallback: 1.0 (average).
    return 1.0


def calculate_fatigue_score(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    as_of_date: Optional[str] = None,
) -> dict:
    """Compute a pitcher's fatigue state based on recent workload.

    Queries the pitches table for the most recent 7 days of appearances
    and derives a composite fatigue score.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID of the pitcher.
        as_of_date: Reference date (ISO format, e.g. ``"2025-06-15"``).
                    Defaults to today.

    Returns:
        Dict with fatigue level, numeric score, and availability flag.
    """
    ref_date = as_of_date or date.today().isoformat()

    # Fetch per-game appearance data over the last 7 days.
    workload_df = conn.execute(
        """
        SELECT game_date,
               COUNT(*) AS pitches
        FROM   pitches
        WHERE  pitcher_id = $1
          AND  game_date BETWEEN CAST($2 AS DATE) - INTERVAL 7 DAY
                              AND CAST($2 AS DATE)
        GROUP  BY game_date
        ORDER  BY game_date DESC
        """,
        [pitcher_id, ref_date],
    ).fetchdf()

    # ── Compute workload metrics ──
    today = pd.Timestamp(ref_date).date()

    if workload_df.empty:
        return _build_fatigue_result(
            pitcher_id=pitcher_id,
            days_rest=999,
            pitches_1=0, pitches_3=0, pitches_7=0,
            appearances_7=0,
            consecutive=0,
        )

    # Normalise game_date to python date objects.
    workload_df["game_date"] = pd.to_datetime(workload_df["game_date"]).dt.date

    appearance_dates = sorted(workload_df["game_date"].unique(), reverse=True)
    most_recent = appearance_dates[0]
    days_rest = (today - most_recent).days

    # Pitches in rolling windows.
    def _pitches_in_window(days: int) -> int:
        cutoff = today - timedelta(days=days)
        return int(workload_df[workload_df["game_date"] > cutoff]["pitches"].sum())

    pitches_1 = _pitches_in_window(1)
    pitches_3 = _pitches_in_window(3)
    pitches_7 = _pitches_in_window(7)
    appearances_7 = len(appearance_dates)

    # Consecutive days pitched (counting backward from most recent).
    consecutive = 0
    check_date = most_recent
    while check_date in appearance_dates:
        consecutive += 1
        check_date = check_date - timedelta(days=1)
    # Only count as consecutive if the most recent was today or yesterday.
    if days_rest > 1:
        consecutive = 0

    return _build_fatigue_result(
        pitcher_id=pitcher_id,
        days_rest=days_rest,
        pitches_1=pitches_1,
        pitches_3=pitches_3,
        pitches_7=pitches_7,
        appearances_7=appearances_7,
        consecutive=consecutive,
    )


def _build_fatigue_result(
    pitcher_id: int,
    days_rest: int,
    pitches_1: int,
    pitches_3: int,
    pitches_7: int,
    appearances_7: int,
    consecutive: int,
) -> dict:
    """Compute fatigue score and level from raw workload numbers."""
    score = 0.0

    # Rule 1: 3+ consecutive days → unavailable.
    if consecutive >= MAX_CONSECUTIVE_DAYS:
        return {
            "pitcher_id": pitcher_id,
            "days_rest": days_rest,
            "pitches_last_1_day": pitches_1,
            "pitches_last_3_days": pitches_3,
            "pitches_last_7_days": pitches_7,
            "appearances_last_7_days": appearances_7,
            "consecutive_days_pitched": consecutive,
            "fatigue_level": "unavailable",
            "fatigue_score": 1.0,
            "availability": False,
        }

    # Rule 2: 0 days rest AND 25+ pitches yesterday.
    if days_rest == 0 and pitches_1 >= HIGH_PITCH_RECENT:
        score = max(score, 0.7 + min(0.3, (pitches_1 - 25) * 0.01))

    # Rule 3: 40+ pitches in last 3 days.
    if pitches_3 >= MODERATE_PITCH_3DAY:
        p3_score = 0.5 + min(0.3, (pitches_3 - 40) * 0.008)
        score = max(score, p3_score)

    # Rule 4: Each appearance in last 7 days adds fatigue.
    if appearances_7 >= 3:
        score = max(score, 0.4 + (appearances_7 - 3) * 0.15)

    # Rule 5: consecutive days (< 3 but still fatiguing).
    if consecutive == 2:
        score = max(score, 0.5)
    elif consecutive == 1 and days_rest == 0:
        score = max(score, 0.3)

    # Rule 6: Fresh — 2+ days rest, <30 pitches in 7 days.
    if days_rest >= 2 and pitches_7 < FRESH_PITCH_7DAY:
        score = min(score, 0.1)

    # Clamp to [0, 1].
    score = max(0.0, min(1.0, score))

    # Derive level.
    if score <= 0.15:
        level = "fresh"
    elif score <= 0.45:
        level = "moderate"
    elif score <= 0.75:
        level = "tired"
    else:
        level = "unavailable"

    availability = level != "unavailable"

    return {
        "pitcher_id": pitcher_id,
        "days_rest": days_rest,
        "pitches_last_1_day": pitches_1,
        "pitches_last_3_days": pitches_3,
        "pitches_last_7_days": pitches_7,
        "appearances_last_7_days": appearances_7,
        "consecutive_days_pitched": consecutive,
        "fatigue_level": level,
        "fatigue_score": round(score, 2),
        "availability": availability,
    }


def get_bullpen_state(
    conn: duckdb.DuckDBPyConnection,
    team: str = "PHI",
    as_of_date: Optional[str] = None,
) -> list[dict]:
    """Return the full bullpen state for a team, sorted by availability.

    For each reliever on the roster, calculates fatigue, infers role,
    and attaches season-level stats.

    Args:
        conn: Open DuckDB connection.
        team: Team abbreviation (e.g. ``"PHI"``).
        as_of_date: Reference date (ISO format). Defaults to today.

    Returns:
        List of dicts ordered from most available to least available.
    """
    ref_date = as_of_date or date.today().isoformat()

    # Get relievers on the roster.
    roster_df = conn.execute(
        """
        SELECT player_id, full_name, throws
        FROM   players
        WHERE  team = $1
          AND  position = 'RP'
        ORDER  BY full_name
        """,
        [team],
    ).fetchdf()

    if roster_df.empty:
        return []

    results: list[dict] = []

    for _, row in roster_df.iterrows():
        pid = int(row["player_id"])
        name = row["full_name"]
        throws = row["throws"] if pd.notna(row["throws"]) else "R"

        # Fatigue.
        fatigue = calculate_fatigue_score(conn, pid, as_of_date=ref_date)

        # Season stats.
        season_stats = conn.execute(
            """
            SELECT era, fip, k_pct, bb_pct, sv, g, ip, war
            FROM   season_pitching_stats
            WHERE  player_id = $1
            ORDER  BY season DESC
            LIMIT  1
            """,
            [pid],
        ).fetchdf()

        era: Optional[float] = None
        fip: Optional[float] = None
        k_pct: Optional[float] = None
        bb_pct: Optional[float] = None
        saves: int = 0
        appearances: int = 0

        if not season_stats.empty:
            s = season_stats.iloc[0]
            era = round(float(s["era"]), 2) if pd.notna(s["era"]) else None
            fip = round(float(s["fip"]), 2) if pd.notna(s["fip"]) else None
            k_pct = round(float(s["k_pct"]), 3) if pd.notna(s["k_pct"]) else None
            bb_pct = round(float(s["bb_pct"]), 3) if pd.notna(s["bb_pct"]) else None
            saves = int(s["sv"]) if pd.notna(s["sv"]) else 0
            appearances = int(s["g"]) if pd.notna(s["g"]) else 0

        # Holds: not in season_pitching_stats, estimate from pitches table.
        # We skip this if unavailable and default to 0.
        holds = 0

        # Role inference.
        role = _infer_role(saves=saves, holds=holds, appearances=appearances,
                           era=era, k_pct=k_pct)

        results.append({
            "pitcher_id": pid,
            "name": name,
            "throws": throws,
            "era": era,
            "fip": fip,
            "k_pct": k_pct,
            "bb_pct": bb_pct,
            "fatigue": fatigue,
            "role": role,
            "season_appearances": appearances,
            "season_saves": saves,
            "season_holds": holds,
        })

    # Sort: available first, then by fatigue score ascending.
    results.sort(key=lambda r: (
        0 if r["fatigue"]["availability"] else 1,
        r["fatigue"]["fatigue_score"],
    ))

    return results


def _infer_role(
    saves: int,
    holds: int,
    appearances: int,
    era: Optional[float],
    k_pct: Optional[float],
) -> str:
    """Heuristic role classification for a reliever.

    Returns one of: 'Closer', 'Setup', 'High-leverage', 'Middle', 'Long'.
    """
    if saves >= CLOSER_MIN_SAVES:
        return "Closer"
    if holds >= SETUP_MIN_HOLDS:
        return "Setup"
    # High-leverage: good ERA and K rate without closer/setup saves/holds.
    if era is not None and k_pct is not None:
        if era < 3.50 and k_pct > 0.25:
            return "High-leverage"
    if appearances > 0 and era is not None and era < 4.00:
        return "Middle"
    return "Long"


def get_upcoming_batters(
    lineup: list[dict],
    current_batter_idx: int,
    n: int = 6,
) -> list[dict]:
    """Return the next *n* batters in the lineup, wrapping around.

    Args:
        lineup: List of dicts with at least ``"batter_id"`` and ``"name"``.
        current_batter_idx: Zero-based index of the current batter in the
                            lineup.
        n: Number of upcoming batters to return (default 6).

    Returns:
        List of the next *n* batter dicts from the lineup (circular).
    """
    if not lineup:
        return []

    size = len(lineup)
    result: list[dict] = []
    for offset in range(1, n + 1):
        idx = (current_batter_idx + offset) % size
        result.append(lineup[idx])

    return result


def recommend_reliever(
    conn: duckdb.DuckDBPyConnection,
    team: str,
    game_state: dict,
) -> list[dict]:
    """Recommend the best reliever for the current game situation.

    Evaluates each available bullpen arm against the upcoming batters,
    weighting by Leverage Index and fatigue.

    Args:
        conn: Open DuckDB connection.
        team: Team abbreviation (e.g. ``"PHI"``).
        game_state: Dict describing the current situation::

            {
                "inning": int,
                "outs": int,
                "runners": {1: bool, 2: bool, 3: bool},
                "score_diff": int,       # positive = leading
                "lineup": [              # full opposing lineup
                    {"batter_id": int, "name": str, "bats": str}, ...
                ],
                "current_batter_idx": int,  # 0-based index into lineup
                "as_of_date": str | None,   # optional date override
            }

    Returns:
        List of reliever recommendations sorted best-first.
    """
    inning = game_state.get("inning", 7)
    outs = game_state.get("outs", 0)
    runners = game_state.get("runners", {})
    score_diff = game_state.get("score_diff", 0)
    lineup = game_state.get("lineup", [])
    current_idx = game_state.get("current_batter_idx", 0)
    as_of = game_state.get("as_of_date")

    li = calculate_leverage_index(inning, outs, runners, score_diff)

    # Get upcoming batters (next 3 for matchup scoring).
    upcoming = get_upcoming_batters(lineup, current_idx, n=3) if lineup else []

    # Bullpen availability.
    bullpen = get_bullpen_state(conn, team, as_of_date=as_of)

    recommendations: list[dict] = []

    for reliever in bullpen:
        if not reliever["fatigue"]["availability"]:
            continue

        pid = reliever["pitcher_id"]
        fatigue_score = reliever["fatigue"]["fatigue_score"]
        throws = reliever["throws"]

        # Evaluate vs upcoming batters.
        vs_batters: list[dict] = []
        total_matchup_woba = 0.0

        for batter_info in upcoming:
            bid = batter_info.get("batter_id")
            batter_name = batter_info.get("name", "Unknown")
            batter_hand = batter_info.get("bats", "R")

            if bid is None:
                continue

            matchup = estimate_matchup_woba(conn, pid, bid, method="bayesian")
            platoon_adv = _platoon_advantage(batter_hand, throws) == "pitcher"

            vs_batters.append({
                "batter_name": batter_name,
                "estimated_woba": matchup["estimated_woba"],
                "platoon_advantage": platoon_adv,
            })
            total_matchup_woba += matchup["estimated_woba"]

        # ── Recommendation score ──
        # Lower matchup wOBA is better for the pitcher → invert for scoring.
        if vs_batters:
            avg_matchup_woba = total_matchup_woba / len(vs_batters)
        else:
            avg_matchup_woba = LEAGUE_AVG_WOBA

        # Matchup quality: how far below league average (higher = better).
        matchup_quality = (LEAGUE_AVG_WOBA - avg_matchup_woba) / LEAGUE_AVG_WOBA

        # Fatigue penalty: fresh pitchers score higher.
        fatigue_penalty = fatigue_score  # 0-1, higher = worse

        # Leverage weight: high-leverage situations reward better arms more.
        leverage_weight = min(li / 2.0, 1.5)  # cap at 1.5x

        # Platoon bonus: fraction of upcoming batters with platoon advantage.
        platoon_bonus = 0.0
        if vs_batters:
            platoon_count = sum(1 for b in vs_batters if b["platoon_advantage"])
            platoon_bonus = 0.1 * (platoon_count / len(vs_batters))

        # Role fit: closers in save situations, etc.
        role_bonus = _role_fit_bonus(reliever["role"], inning, score_diff, li)

        # Composite score: higher is better.
        rec_score = (
            (1.0 + matchup_quality * leverage_weight)
            * (1.0 - 0.5 * fatigue_penalty)
            + platoon_bonus
            + role_bonus
        )

        # Build reasoning string.
        reasoning_parts: list[str] = []
        fl = reliever["fatigue"]["fatigue_level"]
        dr = reliever["fatigue"]["days_rest"]
        reasoning_parts.append(f"{fl.capitalize()} ({dr} days rest)")

        if platoon_bonus > 0:
            reasoning_parts.append(f"platoon advantage vs {platoon_count}/{len(vs_batters)} batters")

        if matchup_quality > 0.02:
            reasoning_parts.append("favorable matchups vs upcoming batters")
        elif matchup_quality < -0.02:
            reasoning_parts.append("unfavorable matchups vs upcoming batters")

        if role_bonus > 0:
            reasoning_parts.append(f"role fit ({reliever['role']})")

        recommendations.append({
            "pitcher_id": pid,
            "name": reliever["name"],
            "recommendation_score": round(rec_score, 3),
            "fatigue_level": fl,
            "vs_next_3_batters": vs_batters,
            "reasoning": "; ".join(reasoning_parts),
        })

    # Sort by recommendation score descending (best first).
    recommendations.sort(key=lambda r: r["recommendation_score"], reverse=True)

    return recommendations


def _role_fit_bonus(role: str, inning: int, score_diff: int, li: float) -> float:
    """Bonus for deploying a pitcher in their typical role.

    Closers get a bonus in save situations (9th inning, leading by 1-3).
    Setup men get a bonus in the 7th-8th.
    High-leverage arms get a bonus when LI > 1.5.
    """
    bonus = 0.0

    if role == "Closer":
        if inning >= 9 and 1 <= score_diff <= 3:
            bonus = 0.3  # save situation
        elif inning >= 9 and score_diff == 0:
            bonus = 0.15  # tie game, 9th
    elif role == "Setup":
        if inning in (7, 8) and abs(score_diff) <= 2:
            bonus = 0.15
    elif role == "High-leverage":
        if li >= 1.5:
            bonus = 0.1

    return bonus


# ---------------------------------------------------------------------------
# Batch precompute: team bullpen x opposing lineup matchup stats
# ---------------------------------------------------------------------------

import logging as _logging

_bp_logger = _logging.getLogger(__name__)


def batch_bullpen_matchups(
    conn: duckdb.DuckDBPyConnection,
    season: int,
) -> pd.DataFrame:
    """Precompute bullpen matchup stats for all teams in a season.

    For each team, fetches the bullpen state and computes aggregate matchup
    metrics against the most-frequently-faced opposing lineups.  The result
    is a flat DataFrame cached in ``leaderboard_cache`` for fast dashboard reads.

    Args:
        conn: Open DuckDB connection.
        season: Season year.

    Returns:
        DataFrame with columns: team_id, pitcher_id, pitcher_name, role,
        fatigue_level, fatigue_score, era, fip, k_pct, bb_pct,
        avg_matchup_woba, platoon_adv_pct, season_appearances, season_saves.
    """
    # Get all teams that played in this season
    teams_df = conn.execute(
        """
        SELECT DISTINCT home_team AS team_id FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) = $1
        UNION
        SELECT DISTINCT away_team AS team_id FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) = $1
        """,
        [season],
    ).fetchdf()

    if teams_df.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for team_id in teams_df["team_id"].tolist():
        try:
            bullpen = get_bullpen_state(conn, team_id)
        except Exception as exc:
            _bp_logger.debug("Could not get bullpen state for %s: %s", team_id, exc)
            continue

        if not bullpen:
            continue

        # Get the most common opposing batters for this team (top 20 by PA)
        opp_batters = conn.execute(
            """
            SELECT batter_id, stand, COUNT(*) AS pa
            FROM pitches
            WHERE EXTRACT(YEAR FROM game_date) = $1
              AND (
                  (home_team = $2 AND inning_topbot = 'Top')
                  OR (away_team = $2 AND inning_topbot = 'Bot')
              )
            GROUP BY batter_id, stand
            ORDER BY pa DESC
            LIMIT 20
            """,
            [season, team_id],
        ).fetchdf()

        for reliever in bullpen:
            pid = reliever["pitcher_id"]
            throws = reliever.get("throws", "R")

            # Compute average matchup wOBA against common opponents
            avg_woba = None
            platoon_adv_count = 0
            matchup_count = 0

            if not opp_batters.empty:
                woba_sum = 0.0
                for _, batter_row in opp_batters.iterrows():
                    bid = int(batter_row["batter_id"])
                    batter_hand = batter_row["stand"] if pd.notna(batter_row["stand"]) else "R"
                    try:
                        matchup = estimate_matchup_woba(conn, pid, bid, method="bayesian")
                        woba_sum += matchup["estimated_woba"]
                        matchup_count += 1
                        if _platoon_advantage(batter_hand, throws) == "pitcher":
                            platoon_adv_count += 1
                    except Exception:
                        continue

                if matchup_count > 0:
                    avg_woba = round(woba_sum / matchup_count, 3)

            platoon_pct = round(platoon_adv_count / max(matchup_count, 1), 3)

            rows.append({
                "team_id": team_id,
                "pitcher_id": pid,
                "pitcher_name": reliever.get("name"),
                "role": reliever.get("role"),
                "fatigue_level": reliever["fatigue"]["fatigue_level"],
                "fatigue_score": reliever["fatigue"]["fatigue_score"],
                "era": reliever.get("era"),
                "fip": reliever.get("fip"),
                "k_pct": reliever.get("k_pct"),
                "bb_pct": reliever.get("bb_pct"),
                "avg_matchup_woba": avg_woba,
                "platoon_adv_pct": platoon_pct,
                "season_appearances": reliever.get("season_appearances", 0),
                "season_saves": reliever.get("season_saves", 0),
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values(["team_id", "fatigue_score"]).reset_index(drop=True)
    return df
