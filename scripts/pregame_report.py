#!/usr/bin/env python
"""
Pre-game scouting report generator for the Phillies.

Produces a richly-formatted terminal report covering opponent starting
pitcher scouting, lineup matchup previews, Phillies starter analysis,
and bullpen availability.

Usage
-----
    python scripts/pregame_report.py [--date 2026-04-07]
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pregame_report")

# Ensure stdout can handle Unicode on Windows (cp1252 fallback otherwise)
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True,
    )


# ---------------------------------------------------------------------------
# ANSI formatting
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"
_WHITE = "\033[97m"
_RESET = "\033[0m"
_UNDERLINE = "\033[4m"

DIVIDER = "\033[90m" + "\u2500" * 64 + _RESET
THICK_DIVIDER = "\033[1m" + "\u2550" * 64 + _RESET


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


def _pct_str(val: Optional[float], mult: float = 1.0) -> str:
    """Format a rate/percentage value."""
    if val is None:
        return "--"
    return f"{val * mult:.1f}%"


def _stat_str(val: Optional[float], decimals: int = 2) -> str:
    if val is None:
        return "--"
    return f"{val:.{decimals}f}"


# ---------------------------------------------------------------------------
# Database connection helper
# ---------------------------------------------------------------------------

def _get_conn():
    """Try to open a DuckDB connection; return None on failure."""
    try:
        from src.db.schema import init_db
        return init_db()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Section 1: Game Info
# ---------------------------------------------------------------------------

def _print_game_info(game: dict, feed: Optional[dict]) -> None:
    """Print game metadata: opponent, time, venue, weather."""
    print()
    print(THICK_DIVIDER)
    print(_c("  PRE-GAME SCOUTING REPORT", _BOLD + _CYAN))
    print(THICK_DIVIDER)
    print()

    home = game.get("home", "?")
    away = game.get("away", "?")
    start = game.get("start_time", "TBD")
    is_home = home == "PHI"
    opponent = away if is_home else home
    location = "HOME" if is_home else "AWAY"

    print(f"  {_c('Matchup:', _BOLD)}   {away} @ {home}")
    print(f"  {_c('Phillies:', _BOLD)}  {location}")
    print(f"  {_c('Opponent:', _BOLD)}  {opponent}")
    print(f"  {_c('First Pitch:', _BOLD)} {start}")

    # Venue and weather from GUMBO feed
    if feed:
        game_data = feed.get("gameData", {})
        venue = game_data.get("venue", {}).get("name", "")
        weather = game_data.get("weather", {})
        if venue:
            print(f"  {_c('Venue:', _BOLD)}    {venue}")
        if weather:
            temp = weather.get("temp", "")
            condition = weather.get("condition", "")
            wind = weather.get("wind", "")
            parts = [p for p in [
                f"{temp}F" if temp else "",
                condition,
                wind,
            ] if p]
            if parts:
                print(f"  {_c('Weather:', _BOLD)}  {', '.join(parts)}")

    print()


# ---------------------------------------------------------------------------
# Section 2: Opponent Starting Pitcher
# ---------------------------------------------------------------------------

def _get_starter_from_feed(feed: dict, team_side: str) -> Optional[dict]:
    """Extract the probable pitcher from the GUMBO feed."""
    if not feed:
        return None
    try:
        game_data = feed.get("gameData", {})
        probables = game_data.get("probablePitchers", {})
        pitcher_info = probables.get(team_side, {})
        if pitcher_info:
            return {
                "id": pitcher_info.get("id", 0),
                "name": pitcher_info.get("fullName", "TBD"),
            }
    except Exception:
        pass
    return None


def _print_opponent_starter(
    starter: Optional[dict],
    conn,
    season: int,
) -> None:
    """Print opponent starter scouting report."""
    print(DIVIDER)
    print(_c("  OPPONENT STARTING PITCHER", _BOLD + _RED))
    print(DIVIDER)

    if not starter or not starter.get("id"):
        print(f"\n  Probable pitcher: {_c('TBD', _DIM)}")
        print()
        return

    pid = starter["id"]
    name = starter["name"]
    print(f"\n  {_c(name, _BOLD + _WHITE)} (ID: {pid})")

    # Try to get season stats from DB
    season_stats = None
    throws = "?"
    if conn is not None:
        try:
            row = conn.execute(
                """SELECT era, fip, k_pct, bb_pct, w, l, ip, gs, whip,
                          avg_fastball_velo, avg_spin_rate, gb_pct
                   FROM season_pitching_stats WHERE player_id = $1 AND season = $2""",
                [pid, season],
            ).fetchone()
            if row:
                season_stats = {
                    "era": row[0], "fip": row[1], "k_pct": row[2],
                    "bb_pct": row[3], "w": row[4], "l": row[5],
                    "ip": row[6], "gs": row[7], "whip": row[8],
                    "avg_fb_velo": row[9], "avg_spin": row[10],
                    "gb_pct": row[11],
                }
        except Exception:
            pass

        try:
            t_row = conn.execute(
                "SELECT throws FROM players WHERE player_id = $1", [pid],
            ).fetchone()
            if t_row:
                throws = t_row[0] or "?"
        except Exception:
            pass

    # Try to get season stats from API if DB is empty
    if season_stats is None:
        try:
            from src.ingest.roster_tracker import _api_get, PERSON_URL
            data = _api_get(
                PERSON_URL.format(player_id=pid),
                params={"hydrate": f"stats(type=season,season={season})"},
            )
            people = data.get("people", [])
            if people:
                person = people[0]
                throws = person.get("pitchHand", {}).get("code", "?")
                for sg in person.get("stats", []):
                    for split in sg.get("splits", []):
                        st = split.get("stat", {})
                        if "era" in st:
                            season_stats = {
                                "era": _try_float(st.get("era")),
                                "fip": None,
                                "k_pct": None,
                                "bb_pct": None,
                                "w": st.get("wins", 0),
                                "l": st.get("losses", 0),
                                "ip": _try_float(st.get("inningsPitched")),
                                "gs": st.get("gamesStarted", 0),
                                "whip": _try_float(st.get("whip")),
                                "avg_fb_velo": None,
                                "avg_spin": None,
                                "gb_pct": None,
                            }
                            # Compute K% and BB% from counting stats
                            bf = st.get("battersFaced", 0)
                            if bf and bf > 0:
                                ks = st.get("strikeOuts", 0)
                                bbs = st.get("baseOnBalls", 0)
                                season_stats["k_pct"] = ks / bf if ks else None
                                season_stats["bb_pct"] = bbs / bf if bbs else None
                            break
        except Exception:
            pass

    print(f"  Throws: {_c(throws, _BOLD)}")

    if season_stats:
        print()
        print(f"  {_c('Season Stats:', _UNDERLINE)}")
        w = season_stats.get("w", 0) or 0
        l = season_stats.get("l", 0) or 0
        era = _stat_str(season_stats.get("era"))
        fip = _stat_str(season_stats.get("fip"))
        kp = _pct_str(season_stats.get("k_pct"), 100)
        bbp = _pct_str(season_stats.get("bb_pct"), 100)
        ip = _stat_str(season_stats.get("ip"), 1)
        whip = _stat_str(season_stats.get("whip"))
        print(f"    Record: {w}-{l}  |  ERA: {era}  |  FIP: {fip}")
        print(f"    IP: {ip}  |  WHIP: {whip}")
        print(f"    K%: {kp}  |  BB%: {bbp}")

        if season_stats.get("avg_fb_velo"):
            print(f"    FB Velo: {season_stats['avg_fb_velo']:.1f} mph", end="")
            if season_stats.get("gb_pct"):
                print(f"  |  GB%: {season_stats['gb_pct']:.1f}%", end="")
            print()
    else:
        print(f"\n  {_c('No season stats available', _DIM)}")

    # Arsenal breakdown from DB
    if conn is not None:
        try:
            from src.db.queries import get_pitcher_arsenal
            arsenal = get_pitcher_arsenal(conn, pid, season=season)
            if not arsenal.empty:
                print()
                print(f"  {_c('Arsenal Breakdown:', _UNDERLINE)}")
                print(
                    f"    {'Pitch':<12} {'Velo':>6} {'Spin':>7} "
                    f"{'Usage':>7} {'Whiff%':>7} {'HBreak':>7} {'VBreak':>7}"
                )
                print(f"    {'-' * 55}")
                total_pitches = arsenal["num_pitches"].sum()
                for _, row in arsenal.iterrows():
                    pname = row.get("pitch_name", row.get("pitch_type", "?"))
                    if len(pname) > 11:
                        pname = pname[:11]
                    velo = f"{row['avg_velocity']:.1f}" if row.get("avg_velocity") else "--"
                    spin = f"{row['avg_spin_rate']:.0f}" if row.get("avg_spin_rate") else "--"
                    usage = f"{row['num_pitches'] / total_pitches * 100:.1f}%" if total_pitches else "--"
                    whiff = f"{row['whiff_pct']:.1f}%" if row.get("whiff_pct") is not None else "--"
                    hb = f"{row['avg_horz_break']:.1f}" if row.get("avg_horz_break") is not None else "--"
                    vb = f"{row['avg_vert_break']:.1f}" if row.get("avg_vert_break") is not None else "--"
                    print(f"    {pname:<12} {velo:>6} {spin:>7} {usage:>7} {whiff:>7} {hb:>7} {vb:>7}")
        except Exception:
            pass

    # Recent starts from DB
    if conn is not None:
        try:
            from src.db.queries import get_pitcher_game_log
            game_log = get_pitcher_game_log(conn, pid, n_games=3)
            if not game_log.empty:
                print()
                print(f"  {_c('Last 3 Starts:', _UNDERLINE)}")
                print(
                    f"    {'Date':<12} {'Pitches':>8} {'BF':>4} "
                    f"{'Velo':>6} {'Whiff%':>7} {'K':>3} {'BB':>3} {'HR':>3}"
                )
                print(f"    {'-' * 50}")
                for _, row in game_log.iterrows():
                    gd = str(row.get("game_date", ""))[:10]
                    tp = int(row.get("total_pitches", 0))
                    bf = int(row.get("batters_faced", 0))
                    av = f"{row['avg_velo']:.1f}" if row.get("avg_velo") else "--"
                    wp = f"{row['whiff_pct']:.1f}%" if row.get("whiff_pct") is not None else "--"
                    k = int(row.get("strikeouts", 0))
                    bb = int(row.get("walks", 0))
                    hr = int(row.get("home_runs", 0))
                    print(f"    {gd:<12} {tp:>8} {bf:>4} {av:>6} {wp:>7} {k:>3} {bb:>3} {hr:>3}")
        except Exception:
            pass

    print()


# ---------------------------------------------------------------------------
# Section 3: Phillies Lineup Matchup Preview
# ---------------------------------------------------------------------------

def _print_lineup_matchups(
    feed: Optional[dict],
    opp_starter_id: Optional[int],
    is_home: bool,
    conn,
) -> None:
    """Print matchup preview for each Phillies batter vs opponent starter."""
    print(DIVIDER)
    print(_c("  PHILLIES LINEUP MATCHUP PREVIEW", _BOLD + _GREEN))
    print(DIVIDER)

    if not opp_starter_id:
        print(f"\n  {_c('Opponent starter TBD -- cannot generate matchups', _DIM)}")
        print()
        return

    # Try to get lineup from GUMBO feed
    lineup: list[dict] = []
    if feed:
        try:
            from src.ingest.live_feed import parse_lineup
            phi_side = "home" if is_home else "away"
            lineup = parse_lineup(feed, team=phi_side)
        except Exception:
            pass

    # Fall back to roster if lineup not available
    if not lineup:
        print(f"\n  {_c('Lineup not yet announced -- using roster projection', _DIM)}\n")
        if conn is not None:
            try:
                rows = conn.execute(
                    """SELECT player_id, full_name, bats, position
                       FROM players WHERE team = 'PHI'
                       AND position NOT IN ('SP', 'RP', 'P')
                       ORDER BY full_name LIMIT 9""",
                ).fetchall()
                for i, r in enumerate(rows, 1):
                    lineup.append({
                        "order": i, "id": r[0], "name": r[1],
                        "bats": r[2] or "R", "position": r[3] or "?",
                    })
            except Exception:
                pass

    if not lineup:
        print(f"  {_c('No lineup data available', _DIM)}")
        print()
        return

    # Print each batter matchup
    best_matchup: Optional[dict] = None
    worst_matchup: Optional[dict] = None
    best_woba = -1.0
    worst_woba = 2.0

    print(
        f"\n    {'#':<3} {'Batter':<22} {'B':>2} {'Pos':>4} "
        f"{'Est wOBA':>9} {'Matchup':>8} {'Note'}"
    )
    print(f"    {'-' * 62}")

    for batter in lineup:
        bid = batter.get("id", 0)
        name = batter.get("name", "?")
        bats = batter.get("bats", "?")
        pos = batter.get("position", "?")
        order = batter.get("order", "?")

        est_woba = None
        matchup_label = "--"
        note = ""

        if conn is not None and bid and opp_starter_id:
            try:
                from src.analytics.matchups import estimate_matchup_woba
                result = estimate_matchup_woba(conn, opp_starter_id, bid, method="bayesian")
                est_woba = result.get("estimated_woba")
                pa = result.get("direct_pa", 0)
                if pa >= 15:
                    matchup_label = f"{pa} PA"
                elif pa > 0:
                    matchup_label = f"{pa} PA*"
                else:
                    matchup_label = "prior"

                if est_woba is not None:
                    if est_woba > best_woba:
                        best_woba = est_woba
                        best_matchup = {"name": name, "woba": est_woba}
                    if est_woba < worst_woba:
                        worst_woba = est_woba
                        worst_matchup = {"name": name, "woba": est_woba}
            except Exception:
                pass

        woba_str = f".{int(est_woba * 1000):03d}" if est_woba is not None else "  --"

        # Color code: green for good matchup (>.350), red for bad (<.280)
        if est_woba is not None:
            if est_woba >= 0.350:
                woba_str = _c(woba_str, _GREEN)
                note = "favorable"
            elif est_woba <= 0.280:
                woba_str = _c(woba_str, _RED)
                note = "tough"

        if len(name) > 21:
            name = name[:21]

        print(
            f"    {order:<3} {name:<22} {bats:>2} {pos:>4} "
            f"{woba_str:>9} {matchup_label:>8} {_c(note, _DIM)}"
        )

    # Highlight best/worst
    if best_matchup and worst_matchup:
        print()
        print(
            f"    {_c('Best matchup:', _GREEN + _BOLD)}  "
            f"{best_matchup['name']} (.{int(best_matchup['woba'] * 1000):03d} wOBA)"
        )
        print(
            f"    {_c('Worst matchup:', _RED + _BOLD)} "
            f"{worst_matchup['name']} (.{int(worst_matchup['woba'] * 1000):03d} wOBA)"
        )

    print()


# ---------------------------------------------------------------------------
# Section 4: Phillies Starter
# ---------------------------------------------------------------------------

def _print_phillies_starter(
    starter: Optional[dict],
    conn,
    season: int,
) -> None:
    """Print a brief scouting summary for the Phillies starter."""
    print(DIVIDER)
    print(_c("  PHILLIES STARTING PITCHER", _BOLD + _CYAN))
    print(DIVIDER)

    if not starter or not starter.get("id"):
        print(f"\n  Probable pitcher: {_c('TBD', _DIM)}")
        print()
        return

    pid = starter["id"]
    name = starter["name"]
    print(f"\n  {_c(name, _BOLD + _WHITE)} (ID: {pid})")

    # Season stats
    if conn is not None:
        try:
            row = conn.execute(
                """SELECT era, fip, k_pct, bb_pct, w, l, ip, whip
                   FROM season_pitching_stats WHERE player_id = $1 AND season = $2""",
                [pid, season],
            ).fetchone()
            if row:
                era = _stat_str(row[0])
                fip = _stat_str(row[1])
                kp = _pct_str(row[2], 100)
                bbp = _pct_str(row[3], 100)
                w, l = row[4] or 0, row[5] or 0
                ip = _stat_str(row[6], 1)
                whip = _stat_str(row[7])
                print(f"    {w}-{l}  |  ERA: {era}  |  FIP: {fip}  |  WHIP: {whip}")
                print(f"    IP: {ip}  |  K%: {kp}  |  BB%: {bbp}")
        except Exception:
            pass

        # Arsenal summary
        try:
            from src.db.queries import get_pitcher_arsenal
            arsenal = get_pitcher_arsenal(conn, pid, season=season)
            if not arsenal.empty:
                print()
                print(f"  {_c('Arsenal:', _UNDERLINE)}")
                total = arsenal["num_pitches"].sum()
                for _, row in arsenal.iterrows():
                    pname = row.get("pitch_name", row.get("pitch_type", "?"))
                    velo = f"{row['avg_velocity']:.1f}" if row.get("avg_velocity") else "--"
                    usage = f"{row['num_pitches'] / total * 100:.0f}%" if total else "--"
                    whiff = f"{row['whiff_pct']:.1f}%" if row.get("whiff_pct") is not None else "--"
                    print(f"    {pname:<16} {velo:>6} mph  {usage:>5} usage  {whiff:>6} whiff")
        except Exception:
            pass

        # Recent form (last 3 games)
        try:
            from src.db.queries import get_pitcher_game_log
            gl = get_pitcher_game_log(conn, pid, n_games=3)
            if not gl.empty:
                print()
                print(f"  {_c('Recent Form (last 3):', _UNDERLINE)}")
                for _, row in gl.iterrows():
                    gd = str(row.get("game_date", ""))[:10]
                    tp = int(row.get("total_pitches", 0))
                    k = int(row.get("strikeouts", 0))
                    bb = int(row.get("walks", 0))
                    print(f"    {gd}: {tp} pitches, {k} K, {bb} BB")
        except Exception:
            pass

    print()


# ---------------------------------------------------------------------------
# Section 5: Bullpen Status
# ---------------------------------------------------------------------------

def _print_bullpen_status(conn, season: int) -> None:
    """Print bullpen availability and fatigue status."""
    print(DIVIDER)
    print(_c("  BULLPEN STATUS", _BOLD + _YELLOW))
    print(DIVIDER)

    if conn is None:
        print(f"\n  {_c('No database connection -- cannot assess bullpen', _DIM)}")
        print()
        return

    # Try analytics bullpen state first
    try:
        from src.analytics.bullpen import get_bullpen_state
        bp_state = get_bullpen_state(conn, team="PHI")
        if bp_state:
            print(
                f"\n    {'Pitcher':<22} {'Throws':>6} {'ERA':>6} {'Role':<14} "
                f"{'Fatigue':<12} {'Status'}"
            )
            print(f"    {'-' * 68}")

            for rp in bp_state:
                name = rp.get("name", "?")
                if len(name) > 21:
                    name = name[:21]
                throws = rp.get("throws", "?")
                era = f"{rp['era']:.2f}" if rp.get("era") is not None else "--"
                role = rp.get("role", "?")
                fatigue = rp.get("fatigue", {})
                level = fatigue.get("fatigue_level", "?")
                avail = fatigue.get("availability", True)
                days_rest = fatigue.get("days_rest", "?")
                p7 = fatigue.get("pitches_last_7_days", 0)

                # Color code availability
                if not avail:
                    status_str = _c("UNAVAILABLE", _RED + _BOLD)
                elif level == "fresh":
                    status_str = _c("FRESH", _GREEN + _BOLD)
                elif level == "moderate":
                    status_str = _c("AVAILABLE", _YELLOW)
                else:
                    status_str = _c("TIRED", _RED)

                level_str = f"{level} ({days_rest}d rest)"

                print(
                    f"    {name:<22} {throws:>6} {era:>6} {role:<14} "
                    f"{level_str:<12} {status_str}"
                )

            # Summary
            avail_count = sum(1 for r in bp_state if r.get("fatigue", {}).get("availability"))
            fresh_count = sum(
                1 for r in bp_state
                if r.get("fatigue", {}).get("fatigue_level") == "fresh"
            )
            print(
                f"\n    {_c('Summary:', _BOLD)} "
                f"{avail_count} available, {fresh_count} fully rested"
            )
            print()
            return
    except Exception:
        pass

    # Fallback: API-based roster view
    try:
        from src.ingest.roster_tracker import get_active_roster, PHILLIES_TEAM_ID
        roster = get_active_roster(PHILLIES_TEAM_ID)
        pitchers = [p for p in roster if p["position"] in ("RP", "P") and p.get("season_stats")]
        if pitchers:
            print(
                f"\n    {'Pitcher':<22} {'Throws':>6} {'ERA':>6} {'G':>4} {'SV':>4}"
            )
            print(f"    {'-' * 44}")
            for p in pitchers:
                name = p["name"][:21]
                throws = p.get("throws", "?")
                ss = p.get("season_stats", {})
                era = f"{ss.get('era', 0.0):.2f}"
                g = ss.get("g", 0)
                sv = ss.get("sv", 0)
                print(f"    {name:<22} {throws:>6} {era:>6} {g:>4} {sv:>4}")
            print()
        else:
            print(f"\n  {_c('No bullpen data available', _DIM)}")
            print()
    except Exception:
        print(f"\n  {_c('Could not fetch bullpen data', _DIM)}")
        print()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _try_float(val) -> Optional[float]:
    """Safely convert to float."""
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Main report orchestration
# ---------------------------------------------------------------------------

def generate_report(date_str: Optional[str] = None) -> None:
    """Generate and print the full pre-game report.

    Args:
        date_str: Date to generate report for (YYYY-MM-DD, default today).
    """
    if date_str is None:
        date_str = date.today().strftime("%Y-%m-%d")

    try:
        season = int(date_str[:4])
        # Validate the date is parseable
        datetime.strptime(date_str, "%Y-%m-%d")
    except (ValueError, IndexError):
        print(f"\n  Invalid date format: '{date_str}'. Expected YYYY-MM-DD.")
        return

    # Step 1: Find the Phillies game
    print(f"\n  Fetching schedule for {date_str}...")

    try:
        from src.ingest.live_feed import get_phillies_game, fetch_live_feed
        game = get_phillies_game(date_str)
    except Exception as exc:
        print(f"  Error fetching schedule: {exc}")
        return

    if game is None:
        print(f"\n  No Phillies game scheduled for {date_str}.")
        return

    game_pk = game["game_pk"]
    home = game.get("home", "")
    away = game.get("away", "")
    is_home = home == "PHI"

    # Step 2: Fetch GUMBO feed for game details
    feed: Optional[dict] = None
    try:
        feed = fetch_live_feed(game_pk)
    except Exception:
        logger.warning("Could not fetch GUMBO feed for game_pk=%d", game_pk)

    # Step 3: Get DB connection
    conn = _get_conn()

    # Step 4: Identify starters
    opp_side = "away" if is_home else "home"
    phi_side = "home" if is_home else "away"
    opp_starter = _get_starter_from_feed(feed, opp_side)
    phi_starter = _get_starter_from_feed(feed, phi_side)

    # Section 1: Game Info
    _print_game_info(game, feed)

    # Section 2: Opponent Starter Scouting
    _print_opponent_starter(opp_starter, conn, season)

    # Section 3: Lineup Matchup Preview
    _print_lineup_matchups(
        feed,
        opp_starter.get("id") if opp_starter else None,
        is_home,
        conn,
    )

    # Section 4: Phillies Starter
    _print_phillies_starter(phi_starter, conn, season)

    # Section 5: Bullpen Status
    _print_bullpen_status(conn, season)

    # Footer
    print(THICK_DIVIDER)
    print(
        _c(f"  Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", _DIM)
    )
    print(THICK_DIVIDER)
    print()

    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and generate the pre-game report."""
    parser = argparse.ArgumentParser(
        description="Generate a pre-game scouting report for the Phillies.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date for the report (YYYY-MM-DD, default: today)",
    )
    args = parser.parse_args()
    generate_report(date_str=args.date)


if __name__ == "__main__":
    main()
