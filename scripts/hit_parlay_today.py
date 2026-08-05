#!/usr/bin/env python
"""Transparent 'to record a hit' model for today's MLB slate.

NOT a validated flagship model. This is an explainable heuristic that ranks
batters by P(>=1 hit) so we can assemble a high-floor hit parlay. Every input
is real (fresh Statcast + 2026 season stats + live lineups); the *combination*
formula is a documented heuristic, not a backtested edge.

Model (per batter vs today's opposing starter, in today's park):

    skill   = empirical-Bayes shrink of the batter's per-AB hit rate,
              blending Statcast xBA (luck-stripped) with actual BA,
              regressed toward league xBA by sample size.
    opp     = 0.55 * (starter_xBA_against / league) + 0.45 * 1.0
              (a batter faces the starter for ~55% of PAs, the bullpen ~45%)
    platoon = batter's BA vs the starter's handedness (fresh pitches), lightly
              blended in when the sample supports it.
    park    = BABIP-leaning hit park factor (neutral 1.0 default).
    p_hit   = clamp(skill * opp * platoon * park, 0.12, 0.42)   # per AB
    AB_exp  = expected at-bats from lineup slot (PA table * (1 - bb% - ~2%))
    P(hit)  = 1 - (1 - p_hit) ** AB_exp

Usage
-----
    python scripts/hit_parlay_today.py [--date 2026-07-24] [--top 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import duckdb  # noqa: E402

DB_PATH = ROOT / "data" / "baseball.duckdb"
LEAGUE_XBA = 0.245
PRIOR_AB = 100  # empirical-Bayes shrinkage strength (AB)

# Expected plate appearances by batting-order slot (MLB averages).
PA_BY_SLOT = {1: 4.62, 2: 4.52, 3: 4.42, 4: 4.32, 5: 4.22,
              6: 4.12, 7: 4.02, 8: 3.92, 9: 3.82}
DEFAULT_PA = 4.05  # unknown slot: assume a regular

# BABIP-leaning hit park factors (1.00 = neutral). Conservative; extreme parks
# only. Keyed by home-team abbreviation.
PARK_HIT_FACTOR = {
    "COL": 1.08, "BOS": 1.04, "CIN": 1.03, "PHI": 1.02, "BAL": 1.02,
    "KC": 1.02, "TEX": 1.01, "ARI": 1.01,
    "SD": 0.98, "SF": 0.97, "SEA": 0.97, "NYM": 0.98, "MIA": 0.97,
    "DET": 0.98, "CLE": 0.98, "OAK": 0.97, "ATH": 0.97, "PIT": 0.99,
    "TB": 0.99, "LAD": 0.99, "MIL": 0.99,
}

STATUS_URL = "https://statsapi.mlb.com/api/v1/schedule"
LIVE_URL = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"


def _get(url: str, timeout: int = 15) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def fetch_slate(day: str) -> list[dict]:
    """Return today's games with team abbreviations + probable pitchers."""
    url = (f"{STATUS_URL}?sportId=1&date={day}"
           "&hydrate=probablePitcher,team,linescore")
    data = _get(url)
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            away = g["teams"]["away"]["team"]
            home = g["teams"]["home"]["team"]
            games.append({
                "game_pk": g["gamePk"],
                "away_abbr": away.get("abbreviation", "?"),
                "home_abbr": home.get("abbreviation", "?"),
                "away_name": away.get("name", "?"),
                "home_name": home.get("name", "?"),
                "away_sp": g["teams"]["away"].get("probablePitcher", {}),
                "home_sp": g["teams"]["home"].get("probablePitcher", {}),
                "state": g.get("status", {}).get("detailedState", ""),
            })
    return games


def fetch_lineup(game_pk: int, side: str) -> list[dict]:
    """Return posted batting order for `side` ('home'/'away'), or [] if unposted.

    Uses the live feed boxscore battingOrder + player handedness.
    """
    try:
        feed = _get(LIVE_URL.format(pk=game_pk))
        box = feed["liveData"]["boxscore"]["teams"][side]
        order = box.get("battingOrder", [])
        players = box.get("players", {})
        lineup = []
        for slot, pid in enumerate(order, 1):
            info = players.get(f"ID{pid}", {})
            person = info.get("person", {})
            bats = info.get("batSide", {}).get("code", "R")
            lineup.append({
                "slot": slot,
                "player_id": pid,
                "name": person.get("fullName", str(pid)),
                "bats": bats,
            })
        return lineup
    except Exception:
        return []


def projected_lineup(conn, team_abbr: str, day: str) -> list[dict]:
    """Fallback when a lineup isn't posted: the 9 position players with the most
    PA for `team_abbr` over the last 21 days, ranked as a slot proxy.
    """
    from datetime import datetime, timedelta
    cutoff = (datetime.strptime(day, "%Y-%m-%d") - timedelta(days=21)).date().isoformat()
    rows = conn.execute(
        """
        WITH recent AS (
            SELECT batter_id,
                   CASE WHEN inning_topbot = 'Top' THEN away_team ELSE home_team END AS bat_team,
                   game_pk::VARCHAR || '-' || at_bat_number::VARCHAR AS pa_key
            FROM pitches
            WHERE game_date >= ?
        )
        SELECT r.batter_id,
               COUNT(DISTINCT r.pa_key) AS pa
        FROM recent r
        WHERE r.bat_team = ?
        GROUP BY r.batter_id
        ORDER BY pa DESC
        LIMIT 9
        """,
        [cutoff, team_abbr],
    ).fetchall()
    lineup = []
    for slot, (bid, pa) in enumerate(rows, 1):
        nm = conn.execute("SELECT full_name, bats FROM players WHERE player_id=?",
                          [bid]).fetchone()
        lineup.append({
            "slot": slot,
            "player_id": bid,
            "name": nm[0] if nm else str(bid),
            "bats": (nm[1] if nm and nm[1] else "R"),
            "projected": True,
        })
    return lineup


def batter_skill(conn, pid: int) -> dict | None:
    """Empirical-Bayes per-AB hit rate from 2026 xBA/BA, shrunk by sample."""
    row = conn.execute(
        "SELECT ab, h, ba, xba, bb_pct, k_pct FROM season_batting_stats "
        "WHERE player_id=? AND season=2026", [pid]).fetchone()
    if not row or not row[0]:
        return None
    ab, h, ba, xba, bb_pct, k_pct = row
    ba = ba or (h / ab if ab else LEAGUE_XBA)
    # blend luck-stripped xBA with actual BA; xBA is the better predictor
    raw = 0.6 * (xba if xba else ba) + 0.4 * ba
    skill = (raw * ab + LEAGUE_XBA * PRIOR_AB) / (ab + PRIOR_AB)
    return {"skill": skill, "ab": ab, "ba": ba, "xba": xba,
            "bb_pct": bb_pct or 0.08, "k_pct": k_pct or 0.22}


def pitcher_factor(conn, pid: int) -> float:
    """starter xBA-against relative to league (from 2026 pitching stats)."""
    if not pid:
        return 1.0
    row = conn.execute(
        "SELECT xba FROM season_pitching_stats WHERE player_id=? AND season=2026",
        [pid]).fetchone()
    if row and row[0]:
        return max(0.80, min(1.20, row[0] / LEAGUE_XBA))
    return 1.0


def platoon_ba(conn, batter_id: int, p_throws: str) -> tuple[float, int]:
    """Batter's BA vs the starter's handedness (career fresh pitches)."""
    if p_throws not in ("L", "R"):
        return (0.0, 0)
    row = conn.execute(
        """
        SELECT
          SUM(CASE WHEN events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END) AS h,
          SUM(CASE WHEN events IS NOT NULL AND events NOT IN
              ('walk','hit_by_pitch','sac_fly','sac_bunt','catcher_interf',
               'intent_walk','sac_fly_double_play','truncated_pa') THEN 1 ELSE 0 END) AS ab
        FROM pitches
        WHERE batter_id=? AND p_throws=? AND events IS NOT NULL
        """, [batter_id, p_throws]).fetchone()
    if row and row[1]:
        return (row[0] / row[1], row[1])
    return (0.0, 0)


def pitcher_throws(conn, pid: int) -> str:
    if not pid:
        return "?"
    row = conn.execute("SELECT throws FROM players WHERE player_id=? AND throws IN ('L','R')",
                       [pid]).fetchone()
    if row:
        return row[0]
    # players.throws is sparsely populated league-wide; derive from fresh pitches
    row = conn.execute(
        "SELECT p_throws FROM pitches WHERE pitcher_id=? AND p_throws IN ('L','R') LIMIT 1",
        [pid]).fetchone()
    return (row[0] if row else "?")


def eval_batter(conn, b: dict, opp_sp_id: int, opp_throws: str, park_factor: float) -> dict | None:
    sk = batter_skill(conn, b["player_id"])
    if sk is None:
        return None
    opp = 0.55 * pitcher_factor(conn, opp_sp_id) + 0.45 * 1.0
    pl_ba, pl_n = platoon_ba(conn, b["player_id"], opp_throws)
    # blend platoon in only when supported; weight grows with sample (cap 0.30)
    if pl_n >= 50:
        w = min(0.30, pl_n / 800)
        skill = (1 - w) * sk["skill"] + w * pl_ba
    else:
        skill = sk["skill"]
    p_hit = skill * opp * park_factor
    p_hit = max(0.12, min(0.42, p_hit))
    slot = b.get("slot")
    pa = PA_BY_SLOT.get(slot, DEFAULT_PA)
    ab_exp = pa * (1 - (sk["bb_pct"] or 0.08) - 0.02)
    p_game = 1 - (1 - p_hit) ** ab_exp
    return {
        "name": b["name"], "slot": slot, "bats": b.get("bats"),
        "projected": b.get("projected", False), "ab_season": sk["ab"],
        "skill": round(sk["skill"], 4), "xba": sk["xba"], "ba": round(sk["ba"], 3),
        "k_pct": sk["k_pct"], "opp_factor": round(opp, 3),
        "platoon_ba": round(pl_ba, 3) if pl_n else None, "platoon_n": pl_n,
        "park": park_factor, "p_hit_per_ab": round(p_hit, 4),
        "ab_exp": round(ab_exp, 2), "p_hit_game": round(p_game, 4),
    }


def american_odds(p: float) -> str:
    if p <= 0 or p >= 1:
        return "n/a"
    if p >= 0.5:
        return f"-{round(100 * p / (1 - p))}"
    return f"+{round(100 * (1 - p) / p)}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()
    day = args.date

    conn = duckdb.connect(str(DB_PATH), read_only=True)
    slate = fetch_slate(day)
    print(f"\n=== Hit-probability model | {day} | {len(slate)} games ===\n")

    candidates = []
    for g in slate:
        for side, opp_side in (("home", "away"), ("away", "home")):
            opp_sp = g[f"{opp_side}_sp"]
            opp_sp_id = opp_sp.get("id", 0)
            opp_throws = pitcher_throws(conn, opp_sp_id) if opp_sp_id else "?"
            team_abbr = g[f"{side}_abbr"]
            home_abbr = g["home_abbr"]
            park = PARK_HIT_FACTOR.get(home_abbr, 1.0)

            lineup = fetch_lineup(g["game_pk"], side)
            if not lineup:
                lineup = projected_lineup(conn, team_abbr, day)

            for b in lineup:
                res = eval_batter(conn, b, opp_sp_id, opp_throws, park)
                if res is None:
                    continue
                res["team"] = team_abbr
                res["opp_sp"] = opp_sp.get("fullName", "TBD")
                res["opp_throws"] = opp_throws
                res["game"] = f"{g['away_abbr']}@{g['home_abbr']}"
                candidates.append(res)

    candidates.sort(key=lambda x: x["p_hit_game"], reverse=True)

    n_conf = sum(1 for c in candidates if not c["projected"])
    print(f"(lineups: {n_conf} batters from CONFIRMED lineups, "
          f"{len(candidates) - n_conf} from projected)\n")
    print(f"{'Batter':<22}{'Tm':<4}{'Slot':<5}{'vs SP':<18}{'T':<2}{'sAB':>5}"
          f"{'p/AB':>6}{'xAB':>5}{'P(hit)':>8}{'odds':>7}  proj")
    print("-" * 96)
    for c in candidates[:30]:
        print(f"{c['name'][:21]:<22}{c['team']:<4}{str(c['slot']):<5}"
              f"{c['opp_sp'][:17]:<18}{c['opp_throws']:<2}{str(c.get('ab_season','')):>5}"
              f"{c['p_hit_per_ab']:>6.3f}"
              f"{c['ab_exp']:>5.1f}{c['p_hit_game']:>8.3f}{american_odds(c['p_hit_game']):>7}"
              f"  {'proj' if c['projected'] else 'CONF'}")

    # pick top-N, one per game for closer-to-independent legs.
    # Gate on an established-regular sample so hot small-sample names don't win.
    MIN_AB = 250
    picks, seen = [], set()
    for c in candidates:
        if c["game"] in seen or (c.get("ab_season") or 0) < MIN_AB:
            continue
        picks.append(c)
        seen.add(c["game"])
        if len(picks) >= args.top:
            break

    combined = 1.0
    for c in picks:
        combined *= c["p_hit_game"]

    print(f"\n=== TOP {args.top} (one per game) ===")
    for i, c in enumerate(picks, 1):
        print(f"{i}. {c['name']} ({c['team']}, slot {c['slot']}) vs {c['opp_sp']} "
              f"[{c['opp_throws']}HP] -- P(hit)={c['p_hit_game']:.1%} ({american_odds(c['p_hit_game'])})")
    print(f"\nParlay combined P(all hit) = {combined:.1%}  ({american_odds(combined)})")

    out = ROOT / "results" / "hit_parlay" / f"{day}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"date": day, "picks": picks,
                               "combined_prob": combined,
                               "all_candidates": candidates[:60]}, indent=2, default=str))
    print(f"\nWrote {out}")
    conn.close()


if __name__ == "__main__":
    main()
