"""
Live Game Dashboard page.

The crown jewel -- displays real-time game state, win probability,
pitch log, velocity tracker, and anomaly alerts during an active
Phillies game.
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Graceful imports -- fall back to mock data when backend isn't ready
# ---------------------------------------------------------------------------

# -- Live feed --
_USE_MOCK_FEED = False
try:
    from src.ingest.live_feed import (
        get_phillies_game,
        fetch_live_feed,
        parse_game_state as _real_parse_game_state,
        parse_pitch_events as _real_parse_pitch_events,
    )
except ImportError:
    _USE_MOCK_FEED = True

# -- Win probability --
_USE_MOCK_WP = False
try:
    from src.analytics.win_probability import (
        calculate_win_probability,
        build_win_prob_curve as _real_build_win_prob_curve,
    )
except ImportError:
    _USE_MOCK_WP = True

# -- Anomaly detection --
_USE_MOCK_ANOMALY = False
try:
    from src.analytics.anomaly import GameAnomalyMonitor
except ImportError:
    _USE_MOCK_ANOMALY = True

# -- Matchup --
_USE_MOCK_MATCHUP = False
try:
    from src.analytics.matchups import get_matchup_stats as _real_get_matchup_stats
except ImportError:
    _USE_MOCK_MATCHUP = True

# -- DB connection helper --
from src.dashboard.db_helper import get_db_connection, has_data, get_player_id_by_name

from src.dashboard.components.matchup_card import create_matchup_card
from src.dashboard.components.win_prob import create_win_prob_chart
from src.dashboard.constants import PITCH_COLORS, PITCH_LABELS, FASTBALL_TYPES


# ---------------------------------------------------------------------------
# Adapter functions that unify real vs mock signatures
# ---------------------------------------------------------------------------

def _get_game_state() -> dict[str, Any] | None:
    """Return game state from the real API.

    Falls back to a 'No Game' state on error rather than showing mock data.
    Mock data should NEVER be shown as if it were real game data.
    """
    if _USE_MOCK_FEED:
        # live_feed module unavailable -- show no game rather than fake data
        return {"status": "No Game"}
    try:
        game_info = get_phillies_game()
        if game_info is None:
            return {"status": "No Game"}
        game_pk = game_info["game_pk"]
        feed = fetch_live_feed(game_pk)
        state = _real_parse_game_state(feed)
        # Store the feed in session_state so pitch events can be extracted
        st.session_state["_live_feed"] = feed
        st.session_state["_game_pk"] = game_pk
        # Adapt the real state to include keys the dashboard expects
        return _adapt_game_state(state)
    except Exception as exc:
        st.warning(f"Could not fetch live game data: {exc}")
        return {"status": "No Game"}


def _adapt_game_state(state: dict) -> dict[str, Any]:
    """Adapt parse_game_state() output to the format the dashboard expects.

    The real parse_game_state returns:
      - runners: {first: bool, second: bool, third: bool}
      - current_batter: {id, name, side}
      - current_pitcher: {id, name, hand}
      - count: {balls, strikes}
      - inning_half: "Top"/"Bot"

    The dashboard expects:
      - on_1b/on_2b/on_3b: bool
      - pitcher: {name, id, throws, ...}
      - batter: {name, id, bats, ...}
      - balls/strikes: int
      - half: "top"/"bottom"
    """
    runners = state.get("runners", {})
    pitcher_raw = state.get("current_pitcher", {})
    batter_raw = state.get("current_batter", {})
    count = state.get("count", {})

    # Map inning_half from "Top"/"Bot" to "top"/"bottom"
    inning_half = state.get("inning_half", "Top")
    half = "top" if inning_half == "Top" else "bottom"

    adapted = {
        "game_pk": state.get("game_pk"),
        "status": state.get("status", "Preview"),
        "home_team": state.get("home_team", ""),
        "away_team": state.get("away_team", ""),
        "home_score": state.get("home_score", 0),
        "away_score": state.get("away_score", 0),
        "inning": state.get("inning", 1),
        "half": half,
        "outs": state.get("outs", 0),
        "balls": count.get("balls", 0),
        "strikes": count.get("strikes", 0),
        "on_1b": runners.get("first", False),
        "on_2b": runners.get("second", False),
        "on_3b": runners.get("third", False),
        "pitcher": {
            "name": pitcher_raw.get("name", ""),
            "id": pitcher_raw.get("id", 0),
            "throws": pitcher_raw.get("hand", ""),
            "pitches_today": 0,
            "ip_today": 0,
            "k_today": 0,
        },
        "batter": {
            "name": batter_raw.get("name", ""),
            "id": batter_raw.get("id", 0),
            "bats": batter_raw.get("side", ""),
            "season_avg": "",
        },
        "venue": state.get("venue", ""),
    }

    # Compute win probability for the current state
    if not _USE_MOCK_WP:
        try:
            wp_result = calculate_win_probability(
                inning=state.get("inning", 1),
                inning_half=inning_half,
                outs=state.get("outs", 0),
                runners={
                    "on_1b": runners.get("first", False),
                    "on_2b": runners.get("second", False),
                    "on_3b": runners.get("third", False),
                },
                home_score=state.get("home_score", 0),
                away_score=state.get("away_score", 0),
            )
            adapted["home_win_prob"] = wp_result["home_win_prob"]
            adapted["leverage_index"] = wp_result["leverage_index"]
        except Exception:
            adapted["home_win_prob"] = 0.5
            adapted["leverage_index"] = 1.0
    else:
        adapted["home_win_prob"] = 0.5
        adapted["leverage_index"] = 1.0

    return adapted


@st.cache_data(ttl=15)
def _get_pitch_log() -> pd.DataFrame:
    """Return the pitch log as a DataFrame."""
    if _USE_MOCK_FEED:
        return pd.DataFrame()
    try:
        feed = st.session_state.get("_live_feed")
        if feed is None:
            return pd.DataFrame()
        events = _real_parse_pitch_events(feed)
        if not events:
            return pd.DataFrame()
        df = pd.DataFrame(events)
        # Rename columns to match what the dashboard charts expect
        rename_map = {
            "start_speed": "release_speed",
            "pitch_name": "pitch_type_label",
            "px": "plate_x",
            "pz": "plate_z",
            "call_description": "result",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
        return df
    except Exception:
        return pd.DataFrame()


def _get_win_prob_curve(game: dict[str, Any]) -> pd.DataFrame:
    """Return the win probability curve as a DataFrame."""
    if _USE_MOCK_WP:
        return pd.DataFrame()
    try:
        feed = st.session_state.get("_live_feed")
        if feed is None:
            # No live feed -- return empty rather than mock
            return pd.DataFrame()

        # Build game events from the feed's all plays
        plays = feed.get("liveData", {}).get("plays", {})
        all_plays = plays.get("allPlays", [])

        game_events: list[dict] = []
        for play in all_plays:
            about = play.get("about", {})
            result = play.get("result", {})

            half_inning = about.get("halfInning", "top")
            inning_half = "Top" if half_inning == "top" else "Bot"

            # Determine runners from play data
            runners_dict: dict[str, bool] = {}

            game_events.append({
                "inning": about.get("inning", 1),
                "inning_half": inning_half,
                "outs": about.get("outs", 0) if "outs" in about else 0,
                "runners": runners_dict,
                "home_score": result.get("homeScore", 0),
                "away_score": result.get("awayScore", 0),
                "description": result.get("description", ""),
            })

        if not game_events:
            return pd.DataFrame()

        curve = _real_build_win_prob_curve(game_events)
        return pd.DataFrame(curve)
    except Exception:
        return pd.DataFrame()


def _get_live_anomalies() -> list[dict[str, Any]]:
    """Return anomaly alerts for the current game.

    When a DB connection is available and a live feed is active, creates
    a ``GameAnomalyMonitor`` with real baselines and feeds it the current
    game's pitches.  Returns empty list if no live game or insufficient data.
    """
    if _USE_MOCK_ANOMALY:
        # No anomaly module -- just return empty, not mock alerts
        return []

    conn = get_db_connection()
    if not has_data(conn):
        # No DB data for baselines
        return []

    try:
        game_state = st.session_state.get("_live_game_state")
        feed = st.session_state.get("_live_feed")
        if game_state is None or feed is None:
            return []

        pitcher = game_state.get("pitcher", {})
        pitcher_id = pitcher.get("id")
        if not pitcher_id:
            return []

        # Create or retrieve the monitor from session state
        monitor_key = f"_anomaly_monitor_{pitcher_id}"
        if monitor_key not in st.session_state:
            monitor = GameAnomalyMonitor(pitcher_id=pitcher_id, conn=conn)
            st.session_state[monitor_key] = monitor
        else:
            monitor = st.session_state[monitor_key]

        # Feed all current-game pitches from the live feed
        pitch_events = _real_parse_pitch_events(feed)
        # Reset and re-feed to keep in sync
        monitor._pitches = []
        for pe in pitch_events:
            # Note: release_spin_rate, release_pos_x, release_pos_z are
            # not available from the live feed (parse_pitch_events doesn't
            # provide them).  The anomaly detectors (detect_spin_anomaly,
            # detect_release_point_drift) gracefully skip pitches where
            # these are None, so passing None here is safe — those
            # detectors simply won't fire until richer data is available.
            monitor.add_pitch({
                "pitch_type": pe.get("pitch_type", ""),
                "release_speed": pe.get("start_speed"),
                "release_spin_rate": pe.get("release_spin_rate"),
                "release_pos_x": pe.get("release_pos_x"),
                "release_pos_z": pe.get("release_pos_z"),
            })

        alerts = monitor.check_all()
        if alerts:
            # Adapt alert format to what the dashboard expects
            adapted: list[dict[str, Any]] = []
            for a in alerts:
                adapted.append({
                    "type": a.get("type", "unknown"),
                    "severity": "high" if a.get("severity") == "critical" else a.get("severity", "low"),
                    "icon": "warning" if a.get("severity") in ("critical", "warning") else "info",
                    "player": pitcher.get("name", "Unknown"),
                    "message": a.get("message", ""),
                    "timestamp": "",
                    "metric_value": a.get("deviation_mph", a.get("deviation_pct", 0)),
                    "metric_unit": "mph",
                })
            return adapted
        return []
    except Exception as exc:
        st.warning(f"Anomaly detection error: {exc}")
        return []


def _get_matchup_stats(pitcher_name: str, batter_name: str) -> dict[str, Any] | None:
    """Get matchup stats -- resolves names to IDs via DB, returns None if unavailable."""
    if _USE_MOCK_MATCHUP:
        return None

    conn = get_db_connection()
    if not has_data(conn):
        return None

    try:
        pitcher_id = get_player_id_by_name(conn, pitcher_name)
        batter_id = get_player_id_by_name(conn, batter_name)
        if pitcher_id is None or batter_id is None:
            return None

        real_stats = _real_get_matchup_stats(conn, pitcher_id, batter_id)

        # Adapt real format to what the dashboard matchup card expects
        return _adapt_matchup_stats(real_stats, pitcher_name, batter_name)
    except Exception as e:
        st.warning(f"Matchup query failed: {e}")
        return None


def _adapt_matchup_stats(real: dict, pitcher_name: str, batter_name: str) -> dict[str, Any]:
    """Adapt the real ``get_matchup_stats`` return to the dashboard format.

    The real module returns keys like ``plate_appearances``, ``batting_avg``,
    ``slug_pct``, ``pitch_type_breakdown`` etc.  The dashboard expects
    ``pa``, ``avg``, ``slg``, ``estimated_woba``, ``pitch_breakdown``, etc.
    """
    pa = real.get("plate_appearances", 0)
    # Build pitch_breakdown list from the real pitch_type_breakdown dict
    pitch_breakdown: list[dict] = []
    for pt, info in real.get("pitch_type_breakdown", {}).items():
        pitch_breakdown.append({
            "pitch_type": pt,
            "label": pt,  # real data doesn't have the label; use code
            "count": info.get("count", 0),
            "pct": 0,  # filled below
            "whiff_rate": info.get("whiff_rate", 0.0),
            "ba_against": info.get("ba_against", 0.0),
            "avg_velo": info.get("avg_velo"),
        })
    total_pitches = sum(p["count"] for p in pitch_breakdown)
    for p in pitch_breakdown:
        p["pct"] = round(p["count"] / total_pitches, 3) if total_pitches else 0

    # Determine vulnerability from the real data
    vulnerability = ""
    if pitch_breakdown:
        weak = max(pitch_breakdown, key=lambda p: p.get("ba_against", 0) if p.get("ba_against") else 0)
        if weak.get("ba_against"):
            vulnerability = (
                f"Struggles against {weak['label']} "
                f"({weak['ba_against']:.3f} BA, {weak['whiff_rate']:.0%} whiff)"
            )

    return {
        "pitcher_name": pitcher_name,
        "batter_name": batter_name,
        "pa": pa,
        "ab": pa,  # approximate
        "h": 0,
        "hr": 0,
        "k": 0,
        "bb": 0,
        "avg": real.get("batting_avg", 0.0),
        "slg": real.get("slug_pct", 0.0),
        "estimated_woba": real.get("woba", 0.0),
        "woba_ci_low": 0.0,
        "woba_ci_high": 0.0,
        "pitch_breakdown": pitch_breakdown,
        "vulnerability": vulnerability,
        "sample_size": pa,
    }


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHILLIES_RED = "#E81828"
PHILLIES_BLUE = "#002D72"

PITCH_RESULT_COLORS: dict[str, str] = {
    "called_strike": "#E74C3C",
    "swinging_strike": "#C0392B",
    "foul": "#E67E22",
    "ball": "#3498DB",
    "in_play_out": "#95A5A6",
    "single": "#2ECC71",
    "double": "#27AE60",
    "triple": "#F39C12",
    "home_run": "#E81828",
}


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Live Game page."""
    st.title("Live Game")
    st.info("""
**What this shows:** Real-time game state, win probability, pitch log, velocity tracking, and anomaly alerts during an active Phillies game.

- **Win Probability** swings toward 1.0 (Phillies win) or 0.0 (loss) with every play — big swings mean high-leverage moments
- **Velocity Tracker** flags fatigue — a starter losing 1-2 mph late in a game is a bullpen trigger
- **Anomaly Alerts** fire when a pitcher's mechanics or stuff deviate from their baseline — early warning for blowups or injuries
""")

    # Refresh controls
    _render_refresh_controls()

    # Fetch data
    game = _get_game_state()
    # Store for other adapters (anomaly detection) to access
    st.session_state["_live_game_state"] = game
    if game is None or game.get("status") == "No Game":
        st.info("No active game right now. Check back at game time!")
        return

    # Map abstract game state "Preview"/"Final" for display
    status = game.get("status", "")
    if status == "Preview":
        st.info("Game hasn't started yet. Check back at game time!")
        return

    # ----- Scoreboard -----
    _render_scoreboard(game)

    st.markdown("---")

    # ----- Win probability + Matchup card -----
    col_wp, col_mu = st.columns([3, 2])

    with col_wp:
        wp_df = _get_win_prob_curve(game)
        fig_wp = create_win_prob_chart(
            wp_df,
            home_team=game.get("home_team", "PHI"),
            away_team=game.get("away_team", "OPP"),
        )
        st.plotly_chart(fig_wp, use_container_width=True, key="live_wp_chart")

    with col_mu:
        pitcher = game.get("pitcher", {})
        batter = game.get("batter", {})
        if pitcher and batter:
            matchup = _get_matchup_stats(
                pitcher_name=pitcher.get("name", ""),
                batter_name=batter.get("name", ""),
            )
            st.markdown("#### Current Matchup")
            if matchup is not None:
                create_matchup_card(matchup)
            else:
                st.info("No historical matchup data available.")
        else:
            st.info("Matchup data unavailable.")

    st.markdown("---")

    # ----- Pitch log + Velocity tracker -----
    pitch_log = _get_pitch_log()

    _render_velocity_tracker(pitch_log)

    _render_pitch_log(pitch_log)

    st.markdown("---")

    # ----- Anomaly alerts -----
    _render_anomaly_alerts()


# ---------------------------------------------------------------------------
# Sub-renderers
# ---------------------------------------------------------------------------

def _render_refresh_controls() -> None:
    """Render the manual refresh button and auto-refresh toggle."""
    r1, r2, r3 = st.columns([1, 1, 4])
    with r1:
        if st.button("Refresh", type="primary"):
            # Clear cached data so next call re-fetches
            _get_pitch_log.clear()
            st.rerun()
    with r2:
        auto = st.toggle("Auto-refresh", value=False, key="auto_refresh_toggle")
    if auto:
        time.sleep(15)
        _get_pitch_log.clear()
        st.rerun()


def _render_scoreboard(game: dict[str, Any]) -> None:
    """Render the top scoreboard banner."""
    away = game.get("away_team", "AWAY")
    home = game.get("home_team", "HOME")
    away_score = game.get("away_score", 0)
    home_score = game.get("home_score", 0)
    inning = game.get("inning", 1)
    half = game.get("half", "top")
    arrow = "\u2191" if half == "top" else "\u2193"
    outs = game.get("outs", 0)
    balls = game.get("balls", 0)
    strikes = game.get("strikes", 0)

    pitcher = game.get("pitcher", {})
    batter = game.get("batter", {})

    # Score row
    s1, s2, s3, s4, s5 = st.columns([2, 1, 2, 1, 2])
    with s1:
        st.metric(away, away_score)
    with s2:
        st.markdown(
            f"<div style='text-align:center; padding-top:12px;'>"
            f"<span style='font-size:1.8rem; font-weight:bold;'>"
            f"{inning}{_ordinal(inning)} {arrow}</span></div>",
            unsafe_allow_html=True,
        )
    with s3:
        st.metric(home, home_score)
    with s4:
        st.metric("Outs", outs)
    with s5:
        st.metric("Count", f"{balls}-{strikes}")

    # Runners / pitcher-batter info
    i1, i2, i3 = st.columns([2, 3, 3])
    with i1:
        _render_bases(game)
    with i2:
        if pitcher:
            p_name = pitcher.get("name", "")
            p_hand = pitcher.get("throws", "")
            p_pitches = pitcher.get("pitches_today", 0)
            p_ip = pitcher.get("ip_today", 0)
            p_k = pitcher.get("k_today", 0)
            st.markdown(
                f"**Pitching:** {p_name} ({p_hand})  \n"
                f"{p_pitches} pitches | {p_ip} IP | {p_k} K"
            )
    with i3:
        if batter:
            b_name = batter.get("name", "")
            b_hand = batter.get("bats", "")
            b_avg = batter.get("season_avg", "")
            st.markdown(
                f"**Batting:** {b_name} ({b_hand})  \n"
                f"Season AVG: {b_avg}"
            )

    # Win probability headline
    wp = game.get("home_win_prob", 0.5)
    li = game.get("leverage_index", 1.0)
    m1, m2 = st.columns(2)
    m1.metric(f"{home} Win Probability", f"{wp:.1%}")
    m2.metric("Leverage Index", f"{li:.2f}")


def _render_bases(game: dict[str, Any]) -> None:
    """Render a simple text-based base diagram."""
    on1 = game.get("on_1b", False)
    on2 = game.get("on_2b", False)
    on3 = game.get("on_3b", False)

    def _dot(on: bool) -> str:
        return "\U0001F7E1" if on else "\u26AA"  # yellow circle vs white circle

    # Diamond layout using markdown
    st.markdown(
        f"<div style='text-align:center; line-height:1.8; font-size:1.3rem;'>"
        f"{_dot(on2)}<br>"
        f"{_dot(on3)} &nbsp;&nbsp;&nbsp;&nbsp; {_dot(on1)}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_pitch_log(pitch_log: pd.DataFrame) -> None:
    """Render a clean, scannable pitch-by-pitch log table.

    Columns shown: #, Type, Velo, Spin, Result, Count, Zone.
    Most recent pitches appear first.  Rows are color-coded by outcome:
      - Strikes (called/swinging): light red
      - Balls: light blue
      - In play (hits): light green
      - In play (outs): light gray
      - Fouls: light yellow
    """
    st.markdown("#### Pitch Log")

    if pitch_log is None or pitch_log.empty:
        st.info("No pitches recorded yet.")
        return

    # --- Build the display DataFrame (most-recent first) ---------------
    display = pitch_log.tail(30).iloc[::-1].copy().reset_index(drop=True)

    # Resolve the velocity column (real data uses release_speed after rename)
    velo_col = next(
        (c for c in ("release_speed", "start_speed") if c in display.columns), None
    )

    # Resolve pitch type label
    type_col = next(
        (c for c in ("pitch_type_label", "pitch_name") if c in display.columns), None
    )
    if type_col is None and "pitch_type" in display.columns:
        type_col = "pitch_type"

    has_count = "count_balls" in display.columns and "count_strikes" in display.columns
    has_zone = "zone" in display.columns
    has_in_zone = "in_zone" in display.columns

    # --- Assemble clean columns ----------------------------------------
    out = pd.DataFrame()

    if "pitch_number" in display.columns:
        out["#"] = display["pitch_number"].values

    if "pitch_type" in display.columns:
        out["Type"] = display["pitch_type"].values
    elif type_col is not None:
        out["Type"] = display[type_col].values

    if velo_col is not None:
        out["Velo"] = pd.to_numeric(display[velo_col], errors="coerce").round(1).values

    if "spin_rate" in display.columns:
        out["Spin"] = (
            pd.to_numeric(display["spin_rate"], errors="coerce")
            .round(0)
            .astype("Int64")
            .values
        )

    # Human-readable result labels
    _RESULT_LABELS: dict[str, str] = {
        "called_strike": "Called Strike",
        "swinging_strike": "Swinging Strike",
        "swinging_strike_blocked": "Swinging Strike",
        "ball": "Ball",
        "blocked_ball": "Ball",
        "foul": "Foul",
        "foul_tip": "Foul Tip",
        "in_play_out": "In Play (Out)",
        "single": "Single",
        "double": "Double",
        "triple": "Triple",
        "home_run": "Home Run",
        "hit_by_pitch": "HBP",
        "pitchout": "Pitchout",
    }
    raw_results: list[str] = []
    if "result" in display.columns:
        raw_results = (
            display["result"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .tolist()
        )
        out["Result"] = [_RESULT_LABELS.get(r, r.replace("_", " ").title()) for r in raw_results]
    else:
        raw_results = [""] * len(display)

    if has_count:
        out["Count"] = (
            display["count_balls"].astype(str) + "-" + display["count_strikes"].astype(str)
        ).values

    if has_zone:
        out["Zone"] = display["zone"].values
    elif has_in_zone:
        out["Zone"] = display["in_zone"].map({True: "In", False: "Out"}).values

    out = out.reset_index(drop=True)

    # --- Row color-coding ----------------------------------------------
    def _style_row(row: pd.Series) -> list[str]:
        idx = row.name
        res = raw_results[idx] if idx < len(raw_results) else ""
        if res in ("called_strike", "swinging_strike", "swinging_strike_blocked", "foul_tip"):
            bg = "background-color: rgba(231, 76, 60, 0.18)"
        elif res in ("ball", "blocked_ball"):
            bg = "background-color: rgba(52, 152, 219, 0.18)"
        elif res in ("single", "double", "triple", "home_run"):
            bg = "background-color: rgba(46, 204, 113, 0.22)"
        elif res == "in_play_out":
            bg = "background-color: rgba(149, 165, 166, 0.18)"
        elif res == "foul":
            bg = "background-color: rgba(241, 196, 15, 0.15)"
        else:
            bg = ""
        return [bg] * len(row)

    styled = out.style.apply(_style_row, axis=1)

    # --- Compact column widths -----------------------------------------
    col_config: dict[str, Any] = {
        "#": st.column_config.NumberColumn(width="small"),
        "Type": st.column_config.TextColumn(width="small"),
        "Velo": st.column_config.NumberColumn(format="%.1f", width="small"),
        "Spin": st.column_config.NumberColumn(width="small"),
        "Result": st.column_config.TextColumn(width="medium"),
        "Count": st.column_config.TextColumn(width="small"),
        "Zone": st.column_config.TextColumn(width="small"),
    }
    col_config = {k: v for k, v in col_config.items() if k in out.columns}

    st.dataframe(
        styled,
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config=col_config,
    )


def _build_velo_figure(
    df: pd.DataFrame,
    title: str,
    chart_key: str,
) -> None:
    """Build and render a single velocity tracker chart for one side (home/away).

    Adds vertical dashed lines at pitcher changes with the pitcher's last name.
    """
    if df.empty:
        st.info(f"No {title.lower()} data yet.")
        return

    # Sequential x-axis for this subset
    df = df.copy()
    df["seq"] = range(1, len(df) + 1)

    fig = go.Figure()

    # --- Pitcher change vertical lines ---
    if "pitcher_name" in df.columns:
        pitcher_col = "pitcher_name"
    elif "pitcher_id" in df.columns:
        pitcher_col = "pitcher_id"
    else:
        pitcher_col = None

    annotations: list[dict] = []
    if pitcher_col is not None:
        prev_pitcher = None
        for _, row in df.iterrows():
            cur = row[pitcher_col]
            if cur != prev_pitcher:
                x_pos = row["seq"]
                # Derive a short display name (last name)
                if isinstance(cur, str) and cur:
                    parts = cur.split()
                    short_name = parts[-1] if parts else cur
                else:
                    short_name = str(cur)
                fig.add_vline(
                    x=x_pos, line_dash="dash",
                    line_color="rgba(255,255,255,0.45)", line_width=1.5,
                )
                annotations.append(dict(
                    x=x_pos, y=1.0, yref="paper",
                    text=short_name, showarrow=False,
                    font=dict(size=10, color="rgba(255,255,255,0.8)"),
                    textangle=-35, xanchor="left", yanchor="bottom",
                    xshift=3, yshift=2,
                ))
                prev_pitcher = cur

    # --- Pitch type traces ---
    pitch_types_in_data = sorted(df["pitch_type"].dropna().unique())
    for pt in pitch_types_in_data:
        if not pt:
            continue
        subset = df[df["pitch_type"] == pt]
        color = PITCH_COLORS.get(pt, "#95A5A6")
        label = PITCH_LABELS.get(pt, pt)

        is_fb = pt in FASTBALL_TYPES
        marker_size = 8 if is_fb else 5
        line_width = 2 if is_fb else 1

        hover_texts = []
        for _, row in subset.iterrows():
            velo = row["release_speed"]
            num = int(row["seq"])
            pitcher = row.get("pitcher_name", "")
            result = row.get("result", "")
            parts = [f"Pitch #{num}, {pt} {velo:.1f} mph"]
            if pitcher:
                parts.append(pitcher)
            if result:
                parts.append(result)
            hover_texts.append(", ".join(parts))

        fig.add_trace(go.Scatter(
            x=subset["seq"],
            y=subset["release_speed"],
            mode="markers+lines",
            marker=dict(size=marker_size, color=color, opacity=0.85,
                        line=dict(width=0.5, color="white")),
            line=dict(color=color, width=line_width),
            name=label,
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        ))

    # --- Layout ---
    all_velos = df["release_speed"].dropna()
    if not all_velos.empty:
        y_min = max(60, all_velos.min() - 5)
        y_max = all_velos.max() + 3
    else:
        y_min, y_max = 75, 102

    fig.update_layout(
        title=dict(text=title, font=dict(size=15)),
        xaxis=dict(title="Pitch #", showgrid=True,
                    gridcolor="rgba(255,255,255,0.1)", dtick=10),
        yaxis=dict(title="Velocity (mph)", range=[y_min, y_max],
                    showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        template="plotly_dark",
        height=420,
        margin=dict(l=55, r=20, t=60, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.06,
                    xanchor="center", x=0.5, font=dict(size=10)),
        annotations=annotations,
    )

    st.plotly_chart(fig, use_container_width=True, key=chart_key)


def _render_velocity_tracker(pitch_log: pd.DataFrame) -> None:
    """Render two velocity-by-pitch charts — one for each team's pitching.

    Each chart shows pitches color-coded by type with vertical dashed lines
    at pitcher changes, labeled with the pitcher's last name.
    """
    if pitch_log is None or pitch_log.empty:
        st.info("No velocity data available.")
        return

    if "pitch_type" not in pitch_log.columns or "release_speed" not in pitch_log.columns:
        st.info("No velocity data available.")
        return

    # Determine home/away split.
    # In baseball: Top of inning → away team bats → HOME team is pitching.
    #              Bot of inning → home team bats → AWAY team is pitching.
    has_half = "inning_topbot" in pitch_log.columns
    game_state = st.session_state.get("_live_game_state", {})
    home = game_state.get("home_team", "Home")
    away = game_state.get("away_team", "Away")

    if has_half:
        home_pitching = pitch_log[pitch_log["inning_topbot"].isin(["Top"])].copy()
        away_pitching = pitch_log[pitch_log["inning_topbot"].isin(["Bot", "Bottom"])].copy()
    else:
        # Can't split — show one combined chart
        pitch_log = pitch_log.copy()
        pitch_log["seq"] = range(1, len(pitch_log) + 1)
        _build_velo_figure(pitch_log, "Pitch Velocity by Type", "velo_combined")
        return

    _build_velo_figure(home_pitching, f"{home} Pitching", "velo_home")
    _build_velo_figure(away_pitching, f"{away} Pitching", "velo_away")


def _render_anomaly_alerts() -> None:
    """Render the top 3 most severe / most recent anomaly alerts.

    Alerts are first deduplicated by type (keeping the most recent per
    type), then sorted by severity (critical > warning > info).  Only
    the top 3 are shown to keep the live game view uncluttered; the full
    list is available on the Anomalies page.
    """
    st.markdown("#### Anomaly Alerts")
    alerts = _get_live_anomalies()

    if not alerts:
        st.success("No anomalies detected.")
        return

    # --- Group by type, keeping only the most recent per type ------------
    best_by_type: dict[str, dict] = {}
    for alert in alerts:
        a_type = alert.get("type", "unknown")
        if a_type not in best_by_type:
            # alerts arrive sorted most-recent-first from the monitor,
            # so the first one we see per type is the most recent.
            best_by_type[a_type] = alert

    deduped = list(best_by_type.values())

    # --- Sort by severity (critical first) then recency ------------------
    _sev_order = {"high": 0, "critical": 0, "medium": 1, "warning": 1, "low": 2, "info": 2}
    deduped.sort(key=lambda a: _sev_order.get(a.get("severity", "low"), 3))

    # --- Limit to top 3 --------------------------------------------------
    top_alerts = deduped[:3]
    remaining = len(deduped) - len(top_alerts)

    for alert in top_alerts:
        severity = alert.get("severity", "low")
        msg = alert.get("message", "")
        player = alert.get("player", "")
        a_type = alert.get("type", "unknown").replace("_", " ").title()

        if severity in ("high", "critical"):
            st.error(f"**{a_type}** | {player}: {msg}")
        elif severity in ("medium", "warning"):
            st.warning(f"**{a_type}** | {player}: {msg}")
        else:
            st.info(f"**{a_type}** | {player}: {msg}")

    if remaining > 0:
        st.caption(f"+{remaining} more alert{'s' if remaining != 1 else ''} -- see Anomalies page for full list.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ordinal(n: int) -> str:
    """Return ordinal suffix for an integer (1 -> 'st', 2 -> 'nd', etc.)."""
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
