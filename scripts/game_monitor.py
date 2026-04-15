#!/usr/bin/env python
"""
Real-time Phillies game monitor with live analytics.

Polls the MLB Stats API for live game data and prints pitch-level info,
anomaly alerts, at-bat results, win probability updates, and pitching
changes to the console.  Optionally writes every event as a JSON line
to a log file for post-game analysis.

Usage
-----
    python scripts/game_monitor.py [--team PHI] [--log-file game_log.jsonl]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import IO, Optional

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingest.live_feed import (
    LiveGameTracker,
    fetch_live_feed,
    get_phillies_game,
    get_todays_games,
    parse_game_state,
    parse_pitch_events,
)
from src.analytics.win_probability import calculate_win_probability

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("game_monitor")

# Ensure stdout can handle Unicode on Windows (cp1252 fallback otherwise)
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True,
    )

# ---------------------------------------------------------------------------
# ANSI colour helpers (degrade gracefully on terminals without support)
# ---------------------------------------------------------------------------

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_MAGENTA = "\033[95m"
_RESET = "\033[0m"


def _c(text: str, code: str) -> str:
    """Wrap *text* in an ANSI escape sequence."""
    return f"{code}{text}{_RESET}"


# ---------------------------------------------------------------------------
# Unicode symbols
# ---------------------------------------------------------------------------

SYM_PITCH = "\u25C6"    # diamond
SYM_AB = "\u25B6"       # right-pointing triangle
SYM_ALERT = "\u26A0"    # warning
SYM_CHANGE = "\u21BB"   # clockwise arrow
SYM_FINAL = "\u2605"    # star
SYM_WAIT = "\u23F3"     # hourglass
SYM_BALL = "\u25CB"     # circle
SYM_STRIKE = "\u25CF"   # filled circle


# ---------------------------------------------------------------------------
# Log-file helper
# ---------------------------------------------------------------------------

class JsonLineLogger:
    """Append JSON-line records to a file handle."""

    def __init__(self, fh: Optional[IO[str]] = None) -> None:
        self._fh = fh

    def log(self, event_type: str, data: dict) -> None:
        if self._fh is None:
            return
        record = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "event": event_type,
            **data,
        }
        try:
            self._fh.write(json.dumps(record, default=str) + "\n")
            self._fh.flush()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Anomaly monitor (lazy init)
# ---------------------------------------------------------------------------

def _try_create_anomaly_monitor(pitcher_id: int):
    """Attempt to create a GameAnomalyMonitor; return None on failure."""
    try:
        from src.analytics.anomaly import GameAnomalyMonitor
        from src.db.schema import init_db
        conn = init_db()
        monitor = GameAnomalyMonitor(pitcher_id=pitcher_id, conn=conn)
        conn.close()
        return monitor
    except Exception:
        # DB may not exist or be empty -- degrade gracefully
        try:
            from src.analytics.anomaly import GameAnomalyMonitor
            return GameAnomalyMonitor(pitcher_id=pitcher_id, conn=None)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_header(home: str, away: str) -> None:
    print()
    print(_c("=" * 60, _BOLD))
    print(_c(f"  {SYM_PITCH} GAME MONITOR: {away} @ {home}", _BOLD + _CYAN))
    print(_c("=" * 60, _BOLD))
    print()


def _print_scoreboard(state: dict) -> None:
    """Print a compact live scoreboard."""
    half = "TOP" if state["inning_half"] == "Top" else "BOT"
    inning = state["inning"]
    outs = state["outs"]
    away = state["away_team"]
    home = state["home_team"]
    a_score = state["away_score"]
    h_score = state["home_score"]

    outs_str = SYM_STRIKE * outs + SYM_BALL * (3 - outs)
    runners = state.get("runners", {})
    r1 = _c("[1B]", _YELLOW) if runners.get("first") else "    "
    r2 = _c("[2B]", _YELLOW) if runners.get("second") else "    "
    r3 = _c("[3B]", _YELLOW) if runners.get("third") else "    "

    print(
        f"  {_c(half, _BOLD)} {inning}  "
        f"{_c(away, _BOLD)} {a_score} - {h_score} {_c(home, _BOLD)}  "
        f"Outs: {outs_str}  {r1}{r2}{r3}"
    )


def _print_pitch(pitch: dict) -> None:
    """Print a single pitch event."""
    pt = pitch.get("pitch_name") or pitch.get("pitch_type") or "?"
    speed = pitch.get("start_speed")
    speed_str = f"{speed:.1f} mph" if speed is not None else "-- mph"
    call = pitch.get("call_description", "")
    count = f"{pitch.get('count_balls', 0)}-{pitch.get('count_strikes', 0)}"

    # Colour the call description
    if "strike" in call.lower() or "foul" in call.lower():
        call_colored = _c(call, _RED)
    elif "ball" in call.lower():
        call_colored = _c(call, _GREEN)
    elif "in play" in call.lower():
        call_colored = _c(call, _MAGENTA)
    else:
        call_colored = call

    print(
        f"    {SYM_PITCH} [{count}] {_c(pt, _BOLD)} {_c(speed_str, _CYAN)}  "
        f"{call_colored}"
    )


def _print_alerts(alerts: list[dict]) -> None:
    """Print anomaly alerts (only new ones)."""
    for alert in alerts:
        sev = alert.get("severity", "warning")
        if sev == "critical":
            prefix = _c(f"  {SYM_ALERT} CRITICAL:", _RED + _BOLD)
        else:
            prefix = _c(f"  {SYM_ALERT} WARNING:", _YELLOW)
        print(f"{prefix} {alert.get('message', '')}")


def _print_ab_result(play: dict, state: dict) -> None:
    """Print a completed at-bat result with win probability."""
    result = play.get("result", {})
    event = result.get("event", "")
    desc = result.get("description", "")

    matchup = play.get("matchup", {})
    batter_name = matchup.get("batter", {}).get("fullName", "?")

    print()
    print(
        f"  {SYM_AB} {_c('AT-BAT:', _BOLD)} {_c(batter_name, _CYAN)}  "
        f"-> {_c(event, _BOLD + _GREEN)}"
    )
    if desc:
        print(f"    {_c(desc, _DIM)}")

    # Win probability
    try:
        wp = calculate_win_probability(
            inning=state.get("inning", 1),
            inning_half=state.get("inning_half", "Top"),
            outs=state.get("outs", 0),
            runners=state.get("runners", {}),
            home_score=state.get("home_score", 0),
            away_score=state.get("away_score", 0),
        )
        home = state.get("home_team", "HOME")
        away = state.get("away_team", "AWAY")
        hp = wp["home_win_prob"] * 100
        ap = wp["away_win_prob"] * 100
        li = wp["leverage_index"]
        print(
            f"    WP: {home} {hp:.1f}% | {away} {ap:.1f}%  "
            f"LI: {li:.2f}  {wp['base_out_state']}"
        )
    except Exception:
        pass
    print()


def _print_pitching_change(change: dict) -> None:
    """Print a pitching change event."""
    name = change.get("new_pitcher_name", "Unknown")
    side = change.get("team", "?")
    print()
    print(
        f"  {SYM_CHANGE} {_c('PITCHING CHANGE', _BOLD + _MAGENTA)} "
        f"({side.upper()}): {_c(name, _BOLD)} now pitching"
    )
    print()


def _print_final(state: dict) -> None:
    """Print the final score and summary."""
    home = state.get("home_team", "HOME")
    away = state.get("away_team", "AWAY")
    hs = state.get("home_score", 0)
    aws = state.get("away_score", 0)

    print()
    print(_c("=" * 60, _BOLD))
    print(
        f"  {SYM_FINAL} {_c('FINAL', _BOLD + _GREEN)}  "
        f"{_c(away, _BOLD)} {aws} - {hs} {_c(home, _BOLD)}"
    )

    # Print innings summary if available
    innings = state.get("innings", [])
    if innings:
        header_nums = "".join(f" {i['num']:>2}" for i in innings)
        away_runs = "".join(f" {i.get('away_runs', 0):>2}" for i in innings)
        home_runs = "".join(f" {i.get('home_runs', 0):>2}" for i in innings)
        print(f"         {header_nums}   R")
        print(f"  {away:>4}  {away_runs}  {aws:>2}")
        print(f"  {home:>4}  {home_runs}  {hs:>2}")

    # Scoring plays
    scoring = state.get("scoring_plays", [])
    if scoring:
        print()
        print(_c("  Scoring Plays:", _BOLD))
        for sp in scoring:
            print(f"    {_c('-', _DIM)} {sp}")

    print(_c("=" * 60, _BOLD))
    print()


# ---------------------------------------------------------------------------
# Main monitor loop
# ---------------------------------------------------------------------------

def monitor_game(
    team: str = "PHI",
    log_file: Optional[str] = None,
    date_str: Optional[str] = None,
) -> None:
    """Run the game monitor for the specified team.

    Args:
        team: Team abbreviation to track (default PHI).
        log_file: Optional path to write JSONL events.
        date_str: Optional date override (YYYY-MM-DD).
    """
    # Open log file if requested
    fh: Optional[IO[str]] = None
    if log_file:
        fh = open(log_file, "a", encoding="utf-8")
    jl = JsonLineLogger(fh)

    try:
        _run_monitor(team, jl, date_str)
    except KeyboardInterrupt:
        print(f"\n{_c('Monitor stopped by user.', _YELLOW)}")
    finally:
        if fh:
            fh.close()


def _run_monitor(team: str, jl: JsonLineLogger, date_str: Optional[str]) -> None:
    """Core monitor logic."""
    # Step 1: Find the game
    print(f"\n  {SYM_WAIT} Looking for {team} game...")

    try:
        games = get_todays_games(date_str)
    except Exception as exc:
        print(f"  {_c('Error fetching schedule:', _RED)} {exc}")
        return

    game = None
    for g in games:
        if g.get("home") == team or g.get("away") == team:
            game = g
            break

    if game is None:
        print(f"\n  No {team} game today.")
        jl.log("no_game", {"team": team})
        return

    game_pk = game["game_pk"]
    home = game["home"]
    away = game["away"]
    status = game.get("status", "Preview")
    start_time = game.get("start_time", "TBD")

    _print_header(home, away)
    jl.log("game_found", {"game_pk": game_pk, "home": home, "away": away,
                           "status": status, "start_time": start_time})

    # Step 2: Wait for game to start if Preview
    if status == "Preview":
        print(f"  Game starts at {_c(start_time, _BOLD)}.")
        print(f"  {SYM_WAIT} Waiting for game to go live (polling every 60s)...\n")

        while True:
            time.sleep(60)
            try:
                g = None
                for gg in get_todays_games(date_str):
                    if gg.get("game_pk") == game_pk:
                        g = gg
                        break
                if g and g.get("status") == "Live":
                    print(f"\n  {_c('Game is now LIVE!', _GREEN + _BOLD)}\n")
                    break
                elif g and g.get("status") == "Final":
                    print(f"\n  {_c('Game went Final before we caught it live.', _YELLOW)}")
                    # Fetch final state
                    try:
                        feed = fetch_live_feed(game_pk)
                        state = parse_game_state(feed)
                        _print_final(state)
                        jl.log("final", state)
                    except Exception:
                        pass
                    return
                else:
                    now = datetime.now().strftime("%H:%M:%S")
                    print(f"  [{now}] Still in Preview...", end="\r")
            except Exception:
                pass

    # Step 3: Create tracker and monitors
    tracker = LiveGameTracker(game_pk=game_pk, poll_interval=10)

    # Anomaly monitor -- keyed by pitcher_id, created lazily
    anomaly_monitors: dict[int, object] = {}
    _seen_alert_keys: set[str] = set()

    def _get_anomaly_monitor(pitcher_id: int):
        if pitcher_id not in anomaly_monitors:
            anomaly_monitors[pitcher_id] = _try_create_anomaly_monitor(pitcher_id)
        return anomaly_monitors[pitcher_id]

    # Step 4: Register callbacks
    last_state: dict = {}

    def on_pitch(pitch: dict) -> None:
        nonlocal last_state
        _print_pitch(pitch)
        jl.log("pitch", pitch)

        # Anomaly detection on the current pitcher
        pitcher_id = pitch.get("pitcher_id", 0)
        if pitcher_id:
            monitor = _get_anomaly_monitor(pitcher_id)
            if monitor is not None:
                # Map live-feed pitch dict keys to what the anomaly monitor expects
                monitor_pitch = {
                    "pitch_type": pitch.get("pitch_type"),
                    "release_speed": pitch.get("start_speed"),
                    "release_spin_rate": None,  # not in live feed
                    "release_pos_x": None,
                    "release_pos_z": None,
                }
                monitor.add_pitch(monitor_pitch)
                try:
                    alerts = monitor.check_all()
                    # Only print alerts we haven't seen
                    new_alerts = []
                    for a in alerts:
                        key = f"{a.get('type')}_{a.get('pitch_number')}_{a.get('pitch_type')}"
                        if key not in _seen_alert_keys:
                            _seen_alert_keys.add(key)
                            new_alerts.append(a)
                    if new_alerts:
                        _print_alerts(new_alerts)
                        for a in new_alerts:
                            jl.log("alert", a)
                except Exception:
                    pass

    def on_play_complete(play: dict) -> None:
        nonlocal last_state
        _print_ab_result(play, last_state)
        result = play.get("result", {})
        jl.log("play_complete", {
            "event": result.get("event", ""),
            "description": result.get("description", ""),
        })

    def on_pitching_change(change: dict) -> None:
        _print_pitching_change(change)
        jl.log("pitching_change", change)

    tracker.on_new_pitch(on_pitch)
    tracker.on_play_complete(on_play_complete)
    tracker.on_pitching_change(on_pitching_change)

    # Step 5: Run the tracker loop
    print(f"  {_c('Monitoring live game...', _GREEN)} (Ctrl+C to stop)\n")

    while True:
        try:
            state = tracker.poll_once()
            last_state = state

            # Print scoreboard periodically (every poll)
            _print_scoreboard(state)

            if state.get("status") == "Final":
                _print_final(state)
                jl.log("final", {
                    "home_team": state.get("home_team"),
                    "away_team": state.get("away_team"),
                    "home_score": state.get("home_score"),
                    "away_score": state.get("away_score"),
                })
                break

        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.warning("Poll error: %s", exc)

        time.sleep(tracker.poll_interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and launch the monitor."""
    parser = argparse.ArgumentParser(
        description="Monitor a live MLB game with real-time analytics.",
    )
    parser.add_argument(
        "--team",
        default="PHI",
        help="Team abbreviation to monitor (default: PHI)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to write JSONL event log (optional)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Date to check schedule for (YYYY-MM-DD, default: today)",
    )
    args = parser.parse_args()

    monitor_game(team=args.team, log_file=args.log_file, date_str=args.date)


if __name__ == "__main__":
    main()
