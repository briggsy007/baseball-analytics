"""
Mock data generators for Diamond Analytics dashboard.

Provides realistic fake data for all dashboard components so the UI
can be developed and tested independently of the backend analytics modules.
All player names, velocities, spin rates, and stat lines are modeled on
real Phillies / NL East data ranges.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants – Phillies roster snapshots (2024-ish)
# ---------------------------------------------------------------------------

PHILLIES_BATTERS: list[dict[str, Any]] = [
    {"name": "Kyle Schwarber", "pos": "LF", "bats": "L", "number": 12},
    {"name": "Trea Turner", "pos": "SS", "bats": "R", "number": 7},
    {"name": "Bryce Harper", "pos": "1B", "bats": "L", "number": 3},
    {"name": "Nick Castellanos", "pos": "RF", "bats": "R", "number": 8},
    {"name": "Alec Bohm", "pos": "3B", "bats": "R", "number": 28},
    {"name": "Bryson Stott", "pos": "2B", "bats": "L", "number": 5},
    {"name": "Brandon Marsh", "pos": "CF", "bats": "L", "number": 16},
    {"name": "J.T. Realmuto", "pos": "C", "bats": "R", "number": 10},
    {"name": "Johan Rojas", "pos": "CF", "bats": "R", "number": 18},
]

PHILLIES_SP: list[dict[str, Any]] = [
    {"name": "Zack Wheeler", "throws": "R", "number": 45},
    {"name": "Aaron Nola", "throws": "R", "number": 27},
    {"name": "Ranger Suarez", "throws": "L", "number": 55},
    {"name": "Cristopher Sanchez", "throws": "L", "number": 61},
    {"name": "Taijuan Walker", "throws": "R", "number": 99},
]

PHILLIES_RP: list[dict[str, Any]] = [
    {"name": "Jeff Hoffman", "throws": "R", "role": "Setup", "number": 34},
    {"name": "Matt Strahm", "throws": "L", "role": "Setup", "number": 25},
    {"name": "Orion Kerkering", "throws": "R", "role": "Middle", "number": 47},
    {"name": "Jose Alvarado", "throws": "L", "role": "Closer", "number": 46},
    {"name": "Gregory Soto", "throws": "L", "role": "Middle", "number": 44},
    {"name": "Seranthony Dominguez", "throws": "R", "role": "Setup", "number": 58},
    {"name": "Carlos Estevez", "throws": "R", "role": "Closer", "number": 73},
]

OPPONENT_BATTERS: list[dict[str, Any]] = [
    {"name": "Francisco Lindor", "pos": "SS", "bats": "S", "number": 12},
    {"name": "Brandon Nimmo", "pos": "RF", "bats": "L", "number": 9},
    {"name": "Pete Alonso", "pos": "1B", "bats": "R", "number": 20},
    {"name": "Mark Vientos", "pos": "3B", "bats": "R", "number": 27},
    {"name": "Jesse Winker", "pos": "LF", "bats": "L", "number": 3},
    {"name": "Francisco Alvarez", "pos": "C", "bats": "R", "number": 4},
    {"name": "Jeff McNeil", "pos": "2B", "bats": "L", "number": 1},
    {"name": "Harrison Bader", "pos": "CF", "bats": "R", "number": 44},
    {"name": "Jose Iglesias", "pos": "SS", "bats": "R", "number": 11},
]

PITCH_TYPES: dict[str, dict[str, Any]] = {
    "FF": {"label": "4-Seam Fastball", "velo_mean": 95.5, "velo_std": 1.2, "spin_mean": 2350, "spin_std": 80,
            "pfx_x_mean": -5.0, "pfx_x_std": 2.0, "pfx_z_mean": 16.0, "pfx_z_std": 2.0, "color": "#E81828"},
    "SI": {"label": "Sinker", "velo_mean": 94.0, "velo_std": 1.1, "spin_mean": 2100, "spin_std": 70,
            "pfx_x_mean": -12.0, "pfx_x_std": 2.5, "pfx_z_mean": 8.0, "pfx_z_std": 2.0, "color": "#FF6B35"},
    "SL": {"label": "Slider", "velo_mean": 87.5, "velo_std": 1.5, "spin_mean": 2500, "spin_std": 100,
            "pfx_x_mean": 3.0, "pfx_x_std": 2.0, "pfx_z_mean": 1.0, "pfx_z_std": 2.0, "color": "#002D72"},
    "CU": {"label": "Curveball", "velo_mean": 80.0, "velo_std": 1.8, "spin_mean": 2700, "spin_std": 120,
            "pfx_x_mean": 6.0, "pfx_x_std": 2.5, "pfx_z_mean": -8.0, "pfx_z_std": 3.0, "color": "#6A0DAD"},
    "CH": {"label": "Changeup", "velo_mean": 87.0, "velo_std": 1.4, "spin_mean": 1800, "spin_std": 90,
            "pfx_x_mean": -10.0, "pfx_x_std": 3.0, "pfx_z_mean": 5.0, "pfx_z_std": 2.5, "color": "#2ECC71"},
    "FC": {"label": "Cutter", "velo_mean": 90.5, "velo_std": 1.3, "spin_mean": 2400, "spin_std": 80,
            "pfx_x_mean": 0.5, "pfx_x_std": 2.0, "pfx_z_mean": 8.0, "pfx_z_std": 2.5, "color": "#F39C12"},
    "FS": {"label": "Splitter", "velo_mean": 88.0, "velo_std": 1.2, "spin_mean": 1500, "spin_std": 70,
            "pfx_x_mean": -6.0, "pfx_x_std": 3.0, "pfx_z_mean": 2.0, "pfx_z_std": 2.0, "color": "#1ABC9C"},
}

ARSENALS: dict[str, list[str]] = {
    "Zack Wheeler": ["FF", "SL", "FC", "CH", "SI"],
    "Aaron Nola": ["FF", "CU", "CH", "SI", "FC"],
    "Ranger Suarez": ["SI", "CH", "FC", "FF"],
    "Cristopher Sanchez": ["SI", "CH", "SL", "FF"],
    "Taijuan Walker": ["FF", "SI", "SL", "CH"],
    "Jeff Hoffman": ["FF", "SL", "CU"],
    "Matt Strahm": ["SL", "SI", "CH"],
    "Jose Alvarado": ["SI", "SL", "FF"],
    "Carlos Estevez": ["FF", "SL", "CH"],
    "Orion Kerkering": ["FF", "SL"],
    "Seranthony Dominguez": ["FF", "SL", "CH"],
    "Gregory Soto": ["SI", "SL"],
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _rand(mean: float, std: float) -> float:
    """Return a Gaussian sample rounded to one decimal."""
    return round(np.random.normal(mean, std), 1)


def _generate_pitch(pitch_type: str, pitch_num: int, total_pitches: int) -> dict[str, Any]:
    """Generate a single realistic pitch record."""
    pt = PITCH_TYPES[pitch_type]
    # Slight fatigue effect: velocity drops ~0.5 mph per 30 pitches after 60
    fatigue_drop = max(0, (pitch_num - 60) * 0.015) if pitch_num > 60 else 0.0
    velo = round(np.random.normal(pt["velo_mean"] - fatigue_drop, pt["velo_std"]), 1)
    spin = int(np.random.normal(pt["spin_mean"], pt["spin_std"]))
    plate_x = round(np.random.normal(0, 0.7), 2)
    plate_z = round(np.random.normal(2.5, 0.6), 2)
    pfx_x = round(np.random.normal(pt["pfx_x_mean"], pt["pfx_x_std"]), 1)
    pfx_z = round(np.random.normal(pt["pfx_z_mean"], pt["pfx_z_std"]), 1)

    in_zone = abs(plate_x) <= 0.83 and 1.5 <= plate_z <= 3.5
    swing_prob = 0.70 if in_zone else 0.30
    swung = random.random() < swing_prob
    if swung:
        contact_prob = 0.75 if in_zone else 0.50
        contact = random.random() < contact_prob
        if contact:
            foul_prob = 0.40
            result = "foul" if random.random() < foul_prob else random.choice(
                ["in_play_out", "in_play_out", "in_play_out", "single", "double", "home_run"]
            )
        else:
            result = "swinging_strike"
    else:
        result = "called_strike" if in_zone else "ball"

    return {
        "pitch_number": pitch_num,
        "pitch_type": pitch_type,
        "pitch_type_label": pt["label"],
        "release_speed": velo,
        "spin_rate": spin,
        "plate_x": plate_x,
        "plate_z": plate_z,
        "pfx_x": pfx_x,
        "pfx_z": pfx_z,
        "result": result,
        "in_zone": in_zone,
        "color": pt["color"],
    }


# ---------------------------------------------------------------------------
# Public mock-data functions
# ---------------------------------------------------------------------------

def mock_game_state() -> dict[str, Any]:
    """Return a realistic in-progress game state dictionary.

    The format matches what the dashboard pages expect after adaptation
    from :func:`src.ingest.live_feed.parse_game_state`.  Keys used by
    the scoreboard renderer: ``home_team``, ``away_team``, ``home_score``,
    ``away_score``, ``inning``, ``half``, ``outs``, ``balls``, ``strikes``,
    ``on_1b``/``on_2b``/``on_3b``, ``pitcher``, ``batter``,
    ``home_win_prob``, ``leverage_index``.
    """
    return {
        "game_pk": 745231,
        "status": "In Progress",
        "home_team": "PHI",
        "away_team": "NYM",
        "home_score": 3,
        "away_score": 2,
        "inning": 5,
        "half": "top",
        "inning_half": "Top",
        "outs": 1,
        "balls": 1,
        "strikes": 2,
        "on_1b": False,
        "on_2b": True,
        "on_3b": False,
        "pitcher": {
            "name": "Zack Wheeler",
            "id": 554430,
            "throws": "R",
            "pitches_today": 68,
            "ip_today": 4.1,
            "hits_today": 4,
            "runs_today": 2,
            "k_today": 6,
            "bb_today": 1,
        },
        "batter": {
            "name": "Francisco Lindor",
            "id": 596019,
            "bats": "S",
            "today_abs": 2,
            "today_hits": 1,
            "season_avg": ".271",
        },
        "home_win_prob": 0.623,
        "leverage_index": 1.84,
        "venue": "Citizens Bank Park",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "start_time": "7:05 PM ET",
    }


def mock_pitch_log(num_pitches: int = 68) -> pd.DataFrame:
    """Return a DataFrame of pitches thrown in the current game."""
    rows: list[dict[str, Any]] = []
    arsenal = ARSENALS["Zack Wheeler"]
    weights = [0.35, 0.25, 0.15, 0.15, 0.10]
    for i in range(1, num_pitches + 1):
        pt = random.choices(arsenal, weights=weights, k=1)[0]
        rows.append(_generate_pitch(pt, i, num_pitches))
    return pd.DataFrame(rows)


def mock_matchup_stats(
    pitcher_name: str = "Zack Wheeler",
    batter_name: str = "Francisco Lindor",
) -> dict[str, Any]:
    """Return historical matchup stats between a pitcher and batter."""
    ab = random.randint(15, 35)
    h = random.randint(4, int(ab * 0.40))
    hr = random.randint(0, min(3, h))
    k = random.randint(3, int(ab * 0.35))
    bb = random.randint(1, 6)
    avg = round(h / ab, 3) if ab > 0 else 0.0
    slg = round((h + hr * 3) / ab, 3) if ab > 0 else 0.0
    woba_est = round(random.uniform(0.280, 0.400), 3)
    woba_low = round(woba_est - random.uniform(0.04, 0.08), 3)
    woba_high = round(woba_est + random.uniform(0.04, 0.08), 3)

    pitch_breakdown = []
    for pt in ARSENALS.get(pitcher_name, ["FF", "SL", "CH"]):
        n = random.randint(5, 25)
        pitch_breakdown.append({
            "pitch_type": pt,
            "label": PITCH_TYPES[pt]["label"],
            "count": n,
            "pct": 0,  # filled below
            "whiff_rate": round(random.uniform(0.15, 0.45), 3),
            "ba_against": round(random.uniform(0.150, 0.350), 3),
            "avg_velo": round(PITCH_TYPES[pt]["velo_mean"] + random.uniform(-1, 1), 1),
        })
    total = sum(p["count"] for p in pitch_breakdown)
    for p in pitch_breakdown:
        p["pct"] = round(p["count"] / total, 3) if total else 0

    # Determine a vulnerability
    weak_pitch = random.choice(pitch_breakdown)
    vulnerability = f"Struggles against {weak_pitch['label']} ({weak_pitch['ba_against']:.3f} BA, {weak_pitch['whiff_rate']:.0%} whiff)"

    return {
        "pitcher_name": pitcher_name,
        "batter_name": batter_name,
        "pa": ab + bb,
        "ab": ab,
        "h": h,
        "hr": hr,
        "k": k,
        "bb": bb,
        "avg": avg,
        "slg": slg,
        "estimated_woba": woba_est,
        "woba_ci_low": woba_low,
        "woba_ci_high": woba_high,
        "pitch_breakdown": pitch_breakdown,
        "vulnerability": vulnerability,
        "sample_size": ab + bb,
    }


def mock_pitcher_profile(pitcher_name: str = "Zack Wheeler") -> dict[str, Any]:
    """Return a full pitcher profile with arsenal and season stats."""
    arsenal = ARSENALS.get(pitcher_name, ["FF", "SL", "CH"])
    arsenal_data = []
    for pt in arsenal:
        info = PITCH_TYPES[pt]
        n_pitches = random.randint(100, 600)
        arsenal_data.append({
            "pitch_type": pt,
            "label": info["label"],
            "pct": 0,
            "avg_velo": round(info["velo_mean"] + random.uniform(-1.5, 1.5), 1),
            "avg_spin": int(info["spin_mean"] + random.uniform(-50, 50)),
            "pfx_x": round(info["pfx_x_mean"] + random.uniform(-1, 1), 1),
            "pfx_z": round(info["pfx_z_mean"] + random.uniform(-1, 1), 1),
            "whiff_rate": round(random.uniform(0.18, 0.42), 3),
            "ba_against": round(random.uniform(0.150, 0.300), 3),
            "usage": n_pitches,
            "color": info["color"],
        })
    total = sum(p["usage"] for p in arsenal_data)
    for p in arsenal_data:
        p["pct"] = round(p["usage"] / total, 3) if total else 0

    throws = "R"
    for sp in PHILLIES_SP + PHILLIES_RP:
        if sp["name"] == pitcher_name:
            throws = sp.get("throws", "R")
            break

    return {
        "name": pitcher_name,
        "throws": throws,
        "era": round(random.uniform(2.5, 4.5), 2),
        "fip": round(random.uniform(2.8, 4.2), 2),
        "whip": round(random.uniform(0.95, 1.30), 2),
        "k_per_9": round(random.uniform(8.0, 12.0), 1),
        "bb_per_9": round(random.uniform(1.5, 3.5), 1),
        "ip": round(random.uniform(50, 180), 1),
        "wins": random.randint(5, 16),
        "losses": random.randint(3, 10),
        "arsenal": arsenal_data,
        "vs_left": {
            "avg": round(random.uniform(0.200, 0.280), 3),
            "woba": round(random.uniform(0.280, 0.360), 3),
            "k_rate": round(random.uniform(0.20, 0.32), 3),
        },
        "vs_right": {
            "avg": round(random.uniform(0.200, 0.280), 3),
            "woba": round(random.uniform(0.280, 0.360), 3),
            "k_rate": round(random.uniform(0.20, 0.32), 3),
        },
        "recent_starts": [
            {"date": (datetime.now() - timedelta(days=d)).strftime("%m/%d"), "ip": round(random.uniform(5, 8), 1),
             "h": random.randint(2, 8), "er": random.randint(0, 4), "k": random.randint(3, 11),
             "bb": random.randint(0, 4)}
            for d in [5, 10, 15, 20, 25]
        ],
    }


def mock_batter_profile(batter_name: str = "Bryce Harper") -> dict[str, Any]:
    """Return a batter profile with hot/cold zone data."""
    # 3x3 zone grid values (avg or woba)
    hot_cold: list[list[float]] = [
        [round(random.uniform(0.200, 0.450), 3) for _ in range(3)]
        for _ in range(3)
    ]
    bats = "R"
    for b in PHILLIES_BATTERS + OPPONENT_BATTERS:
        if b["name"] == batter_name:
            bats = b.get("bats", "R")
            break

    return {
        "name": batter_name,
        "bats": bats,
        "avg": round(random.uniform(0.240, 0.310), 3),
        "obp": round(random.uniform(0.320, 0.410), 3),
        "slg": round(random.uniform(0.400, 0.580), 3),
        "woba": round(random.uniform(0.310, 0.410), 3),
        "hr": random.randint(8, 35),
        "k_rate": round(random.uniform(0.15, 0.30), 3),
        "bb_rate": round(random.uniform(0.06, 0.16), 3),
        "exit_velo": round(random.uniform(87.0, 93.0), 1),
        "hard_hit_rate": round(random.uniform(0.35, 0.55), 3),
        "zone_grid": hot_cold,  # 3x3 zone batting average
    }


def mock_bullpen_state(team: str = "PHI") -> list[dict[str, Any]]:
    """Return bullpen availability for a team."""
    relievers = PHILLIES_RP if team == "PHI" else [
        {"name": "Edwin Diaz", "throws": "R", "role": "Closer", "number": 39},
        {"name": "Reed Garrett", "throws": "R", "role": "Setup", "number": 50},
        {"name": "Adam Ottavino", "throws": "R", "role": "Middle", "number": 0},
        {"name": "Jake Diekman", "throws": "L", "role": "Middle", "number": 63},
        {"name": "Brooks Raley", "throws": "L", "role": "Setup", "number": 48},
        {"name": "Sean Reid-Foley", "throws": "R", "role": "Middle", "number": 61},
    ]

    result: list[dict[str, Any]] = []
    for rp in relievers:
        days_rest = random.choice([0, 1, 1, 2, 2, 3, 4])
        pitches_last3 = random.randint(0, 65)
        consec = random.randint(0, 3)
        fatigue_score = min(1.0, (pitches_last3 / 80) + (0.2 if days_rest == 0 else 0) + (consec * 0.15))
        if fatigue_score < 0.35:
            fatigue_level = "green"
        elif fatigue_score < 0.65:
            fatigue_level = "yellow"
        else:
            fatigue_level = "red"

        result.append({
            "name": rp["name"],
            "throws": rp["throws"],
            "role": rp["role"],
            "era": round(random.uniform(2.0, 5.0), 2),
            "fip": round(random.uniform(2.5, 4.5), 2),
            "days_rest": days_rest,
            "pitches_last_3_days": pitches_last3,
            "consecutive_appearances": consec,
            "fatigue_score": round(fatigue_score, 2),
            "fatigue_level": fatigue_level,
            "available": fatigue_level != "red",
        })
    return result


def mock_anomaly_alerts() -> list[dict[str, Any]]:
    """Return sample anomaly alert records."""
    return [
        {
            "type": "velocity_drop",
            "severity": "high",
            "icon": "warning",
            "player": "Zack Wheeler",
            "message": "FB velocity down 1.8 mph in last 5 pitches (93.2 vs 95.0 season avg)",
            "timestamp": datetime.now().strftime("%H:%M"),
            "metric_value": -1.8,
            "metric_unit": "mph",
        },
        {
            "type": "release_point_shift",
            "severity": "medium",
            "icon": "info",
            "player": "Zack Wheeler",
            "message": "Release point shifted 1.2 inches glove-side in inning 5",
            "timestamp": (datetime.now() - timedelta(minutes=8)).strftime("%H:%M"),
            "metric_value": 1.2,
            "metric_unit": "inches",
        },
        {
            "type": "spin_rate_change",
            "severity": "low",
            "icon": "info",
            "player": "Zack Wheeler",
            "message": "Slider spin rate up 120 RPM from season average (2620 vs 2500)",
            "timestamp": (datetime.now() - timedelta(minutes=22)).strftime("%H:%M"),
            "metric_value": 120,
            "metric_unit": "RPM",
        },
        {
            "type": "chase_rate_spike",
            "severity": "medium",
            "icon": "warning",
            "player": "Francisco Lindor",
            "message": "Chase rate at 42% today vs 28% season average",
            "timestamp": (datetime.now() - timedelta(minutes=15)).strftime("%H:%M"),
            "metric_value": 14,
            "metric_unit": "pct_pts",
        },
    ]


def mock_win_prob_curve(num_events: int = 85) -> pd.DataFrame:
    """Return a win probability curve for the current game.

    Output format matches :func:`src.analytics.win_probability.build_win_prob_curve`:
    each row has ``event_index``, ``home_win_prob``, ``inning``, ``description``,
    ``delta``, and ``leverage``.
    """
    events: list[dict[str, Any]] = []
    wp = 0.50
    prev_wp = 0.50
    for i in range(num_events):
        delta = np.random.normal(0, 0.04)
        # Occasional big swing
        if random.random() < 0.08:
            delta = random.choice([-1, 1]) * random.uniform(0.08, 0.18)
        wp = max(0.02, min(0.98, wp + delta))
        inning = (i // 8) + 1
        half = "top" if (i % 8) < 4 else "bottom"
        wp_delta = round(wp - prev_wp, 4)
        leverage = round(random.uniform(0.5, 2.5), 2)
        events.append({
            "event_index": i,
            "home_win_prob": round(wp, 3),
            "inning": inning,
            "half": half,
            "description": "",
            "delta": wp_delta,
            "leverage": leverage,
        })
        prev_wp = wp

    # Make the last entry match the mock game state
    if events:
        events[-1]["home_win_prob"] = 0.623

    # Annotate a couple of key events
    if len(events) > 20:
        events[18]["description"] = "Harper 2-run HR"
        events[18]["home_win_prob"] = 0.68
    if len(events) > 50:
        events[48]["description"] = "Alonso RBI double"
        events[48]["home_win_prob"] = 0.52

    return pd.DataFrame(events)


def mock_lineup(team: str = "PHI") -> list[dict[str, Any]]:
    """Return a lineup for the given team."""
    batters = PHILLIES_BATTERS if team == "PHI" else OPPONENT_BATTERS
    lineup = []
    for i, b in enumerate(batters):
        lineup.append({
            "order": i + 1,
            "name": b["name"],
            "pos": b["pos"],
            "bats": b["bats"],
            "number": b["number"],
            "avg": round(random.uniform(0.230, 0.310), 3),
            "obp": round(random.uniform(0.300, 0.400), 3),
            "slg": round(random.uniform(0.370, 0.560), 3),
            "ops": 0,  # filled below
            "hr": random.randint(3, 30),
            "rbi": random.randint(15, 85),
        })
        lineup[-1]["ops"] = round(lineup[-1]["obp"] + lineup[-1]["slg"], 3)
    return lineup


def mock_transactions() -> list[dict[str, Any]]:
    """Return recent roster transactions."""
    base = datetime.now()
    return [
        {"date": (base - timedelta(days=1)).strftime("%m/%d"), "type": "Activated",
         "player": "Trea Turner", "detail": "Activated from 10-day IL (hamstring)"},
        {"date": (base - timedelta(days=2)).strftime("%m/%d"), "type": "Optioned",
         "player": "Griff McGarry", "detail": "Optioned to Lehigh Valley (AAA)"},
        {"date": (base - timedelta(days=3)).strftime("%m/%d"), "type": "Recalled",
         "player": "Orion Kerkering", "detail": "Recalled from Lehigh Valley (AAA)"},
        {"date": (base - timedelta(days=5)).strftime("%m/%d"), "type": "Placed on IL",
         "player": "Spencer Turnbull", "detail": "Placed on 15-day IL (right lat strain)"},
    ]


def mock_schedule() -> list[dict[str, Any]]:
    """Return the next 7 games on the schedule."""
    base = datetime.now()
    opponents = ["NYM", "ATL", "ATL", "ATL", "WSH", "WSH", "WSH"]
    home_away = ["vs", "vs", "@", "@", "@", "vs", "vs"]
    times = ["7:05 PM", "1:35 PM", "7:20 PM", "7:20 PM", "7:05 PM", "7:05 PM", "1:35 PM"]
    starters = ["Zack Wheeler", "Aaron Nola", "Ranger Suarez", "Cristopher Sanchez",
                "Taijuan Walker", "Zack Wheeler", "Aaron Nola"]
    opp_starters = ["Sean Manaea", "Chris Sale", "Max Fried", "Reynaldo Lopez",
                    "MacKenzie Gore", "DJ Herz", "Jake Irvin"]
    result = []
    for i in range(7):
        result.append({
            "date": (base + timedelta(days=i)).strftime("%m/%d"),
            "day": (base + timedelta(days=i)).strftime("%a"),
            "opponent": opponents[i],
            "home_away": home_away[i],
            "time": times[i],
            "phillies_starter": starters[i],
            "opponent_starter": opp_starters[i],
        })
    return result


def mock_standings() -> dict[str, Any]:
    """Return NL East standings information."""
    return {
        "team": "PHI",
        "wins": 48,
        "losses": 27,
        "pct": ".640",
        "gb": "-",
        "streak": "W4",
        "l10": "7-3",
        "division": "NL East",
        "rank": 1,
        "division_standings": [
            {"team": "PHI", "w": 48, "l": 27, "pct": ".640", "gb": "-"},
            {"team": "ATL", "w": 42, "l": 32, "pct": ".568", "gb": "5.5"},
            {"team": "NYM", "w": 38, "l": 36, "pct": ".514", "gb": "9.5"},
            {"team": "MIA", "w": 29, "l": 45, "pct": ".392", "gb": "18.5"},
            {"team": "WSH", "w": 28, "l": 46, "pct": ".378", "gb": "19.5"},
        ],
    }


def mock_strike_zone_pitches(pitcher_name: str = "Zack Wheeler", n: int = 200) -> pd.DataFrame:
    """Generate a set of pitches with plate locations for strike-zone heatmaps."""
    arsenal = ARSENALS.get(pitcher_name, ["FF", "SL", "CH"])
    rows = []
    for _ in range(n):
        pt = random.choice(arsenal)
        plate_x = round(np.random.normal(0, 0.7), 2)
        plate_z = round(np.random.normal(2.5, 0.6), 2)
        in_zone = abs(plate_x) <= 0.83 and 1.5 <= plate_z <= 3.5
        is_hit = random.random() < (0.28 if in_zone else 0.15)
        rows.append({
            "pitch_type": pt,
            "plate_x": plate_x,
            "plate_z": plate_z,
            "is_hit": is_hit,
            "estimated_ba": round(random.uniform(0.150, 0.400), 3) if in_zone else round(random.uniform(0.050, 0.250), 3),
            "result": random.choice(["ball", "called_strike", "swinging_strike", "foul", "in_play_out", "single", "double", "home_run"]),
        })
    return pd.DataFrame(rows)


def mock_spray_chart_data(batter_name: str = "Bryce Harper", n: int = 80) -> pd.DataFrame:
    """Generate batted-ball data for spray charts."""
    rows = []
    for _ in range(n):
        # hc_x centered at 125.42 (home plate), hc_y around 160-200 (higher = closer to home)
        hc_x = round(np.random.normal(125.42, 35), 1)
        hc_y = round(np.random.normal(165, 30), 1)
        result = random.choices(
            ["out", "single", "double", "triple", "home_run"],
            weights=[0.55, 0.22, 0.10, 0.02, 0.11],
            k=1,
        )[0]
        rows.append({
            "hc_x": hc_x,
            "hc_y": hc_y,
            "result": result,
            "exit_velocity": round(np.random.normal(89, 7), 1),
            "launch_angle": round(np.random.normal(14, 14), 1),
        })
    return pd.DataFrame(rows)


def mock_velocity_trend(pitcher_name: str = "Zack Wheeler", n_games: int = 10) -> pd.DataFrame:
    """Generate velocity trend data across recent games."""
    base = datetime.now()
    rows = []
    base_velo = 95.5
    for g in range(n_games):
        game_date = (base - timedelta(days=(n_games - g) * 5)).strftime("%m/%d")
        for pitch_num in range(1, random.randint(70, 100)):
            fatigue = max(0, (pitch_num - 60) * 0.018)
            velo = round(np.random.normal(base_velo - fatigue, 1.0), 1)
            rows.append({
                "game_date": game_date,
                "pitch_number": pitch_num,
                "release_speed": velo,
                "pitch_type": "FF",
            })
    return pd.DataFrame(rows)


def mock_release_point_data(pitcher_name: str = "Zack Wheeler", n: int = 200) -> pd.DataFrame:
    """Generate release point data for scatter plots."""
    rows = []
    for _ in range(n):
        rows.append({
            "release_pos_x": round(np.random.normal(-1.8, 0.15), 2),
            "release_pos_z": round(np.random.normal(6.2, 0.12), 2),
            "pitch_type": random.choice(ARSENALS.get(pitcher_name, ["FF", "SL"])),
        })
    return pd.DataFrame(rows)
