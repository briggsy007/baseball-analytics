"""
Bullpen Strategy page.

Displays reliever availability, fatigue levels, and matchup grids
for optimal bullpen deployment.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_USE_MOCK_BULLPEN = False
try:
    from src.analytics.bullpen import (
        get_bullpen_state as _real_get_bullpen_state,
        recommend_reliever as _real_recommend_reliever,
    )
except ImportError:
    _USE_MOCK_BULLPEN = True

_USE_MOCK_MATCHUP = False
try:
    from src.analytics.matchups import get_matchup_stats as _real_get_matchup_stats
except ImportError:
    _USE_MOCK_MATCHUP = True

_HAS_LIVE_FEED = False
try:
    from src.ingest.live_feed import (
        parse_lineup as _real_parse_lineup,
    )
    _HAS_LIVE_FEED = True
except ImportError:
    pass

_HAS_ROSTER = False
try:
    from src.ingest.roster_tracker import get_active_roster as _real_get_active_roster
    _HAS_ROSTER = True
except ImportError:
    pass

from src.dashboard.mock_data import (
    OPPONENT_BATTERS,
    PHILLIES_BATTERS,
    mock_bullpen_state,
)
from src.dashboard.db_helper import (
    get_db_connection,
    has_data,
    get_player_id_by_name,
)


# ---------------------------------------------------------------------------
# Cached wrappers (no conn argument -- get conn inside for st.cache_data)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _cached_bullpen_state(team: str) -> list[dict] | None:
    """Cached real bullpen state (TTL 300s)."""
    conn = get_db_connection()
    if conn is None or _USE_MOCK_BULLPEN:
        return None
    try:
        return _real_get_bullpen_state(conn, team)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _cached_batch_matchup_stats(
    pitcher_ids: tuple[int | None, ...],
    batter_ids: tuple[int | None, ...],
) -> dict[tuple[int, int], dict]:
    """Batch-fetch matchup stats for ALL reliever-batter pairs in ONE query.

    Returns a dict keyed by (pitcher_id, batter_id) -> matchup stats dict.
    """
    conn = get_db_connection()
    if conn is None or _USE_MOCK_MATCHUP:
        return {}

    # Filter out None IDs
    valid_pitcher_ids = [pid for pid in pitcher_ids if pid is not None]
    valid_batter_ids = [bid for bid in batter_ids if bid is not None]
    if not valid_pitcher_ids or not valid_batter_ids:
        return {}

    try:
        # Single query: aggregate wOBA-relevant stats for all pairs at once
        df = conn.execute(
            """
            WITH pa_events AS (
                SELECT pitcher_id, batter_id, events,
                       launch_speed, description, zone,
                       type
                FROM   pitches
                WHERE  pitcher_id IN (SELECT UNNEST($1::INT[]))
                  AND  batter_id  IN (SELECT UNNEST($2::INT[]))
                  AND  events IS NOT NULL
            )
            SELECT pitcher_id,
                   batter_id,
                   COUNT(*)                                       AS plate_appearances,
                   AVG(CASE WHEN events IN ('single','double','triple','home_run')
                            THEN 1.0 ELSE 0.0 END)               AS batting_avg,
                   -- Simplified wOBA weights
                   AVG(CASE
                       WHEN events = 'walk'       THEN 0.690
                       WHEN events = 'hit_by_pitch' THEN 0.720
                       WHEN events = 'single'     THEN 0.880
                       WHEN events = 'double'     THEN 1.247
                       WHEN events = 'triple'     THEN 1.578
                       WHEN events = 'home_run'   THEN 2.031
                       ELSE 0.0
                   END)                                           AS woba
            FROM   pa_events
            GROUP  BY pitcher_id, batter_id
            """,
            [valid_pitcher_ids, valid_batter_ids],
        ).fetchdf()

        results: dict[tuple[int, int], dict] = {}
        if not df.empty:
            for _, row in df.iterrows():
                key = (int(row["pitcher_id"]), int(row["batter_id"]))
                results[key] = {
                    "woba": round(float(row["woba"]), 3),
                    "batting_avg": round(float(row["batting_avg"]), 3),
                    "plate_appearances": int(row["plate_appearances"]),
                }
        return results
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Adapter functions
# ---------------------------------------------------------------------------

def _get_bullpen_state(team: str = "PHI") -> list[dict[str, Any]]:
    """Get bullpen state from real module, API roster, or show warning.

    Priority: 1) DB analytics, 2) API roster (without fatigue), 3) warning.
    """
    conn = get_db_connection()
    use_real = has_data(conn)

    # Try real DB analytics first
    if use_real and not _USE_MOCK_BULLPEN:
        real = _cached_bullpen_state(team)
        if real is not None and len(real) > 0:
            return _adapt_bullpen_state(real)

    # Try API roster as second fallback (at least shows real names)
    if _HAS_ROSTER and team == "PHI":
        try:
            roster = _real_get_active_roster(143)
            if roster:
                relievers = [
                    p for p in roster
                    if p.get("position", {}).get("abbreviation") == "P"
                    or p.get("position", {}).get("type") == "Pitcher"
                ]
                if relievers:
                    adapted = []
                    for rp in relievers:
                        adapted.append({
                            "name": rp.get("fullName", rp.get("name", "")),
                            "throws": rp.get("pitchHand", {}).get("code", ""),
                            "role": "",
                            "era": 0.0,
                            "fip": 0.0,
                            "days_rest": 0,
                            "pitches_last_3_days": 0,
                            "consecutive_appearances": 0,
                            "fatigue_score": 0.0,
                            "fatigue_level": "green",
                            "available": True,
                            "pitcher_id": rp.get("id"),
                        })
                    st.info("Showing roster from MLB API. Fatigue data unavailable -- load pitch data for fatigue tracking.")
                    return adapted
        except Exception:
            pass

    st.warning("Showing demo data -- real bullpen data unavailable")
    return mock_bullpen_state(team=team)


def _adapt_bullpen_state(real_list: list[dict]) -> list[dict[str, Any]]:
    """Flatten the real bullpen state to the dashboard's expected format.

    The real module nests fatigue data under a ``fatigue`` key.  The dashboard
    expects flat keys like ``fatigue_level``, ``fatigue_score``, ``days_rest``,
    ``pitches_last_3_days``, ``consecutive_appearances``, ``available``.
    """
    adapted: list[dict[str, Any]] = []
    for rp in real_list:
        fatigue = rp.get("fatigue", {})
        fatigue_level = fatigue.get("fatigue_level", "moderate")
        # Map real levels to dashboard color scheme
        if fatigue_level in ("fresh",):
            display_level = "green"
        elif fatigue_level in ("moderate",):
            display_level = "yellow"
        elif fatigue_level in ("tired",):
            display_level = "orange"
        else:
            display_level = "red"

        adapted.append({
            "name": rp.get("name", ""),
            "throws": rp.get("throws", ""),
            "role": rp.get("role", ""),
            "era": rp.get("era") or 0.0,
            "fip": rp.get("fip") or 0.0,
            "days_rest": fatigue.get("days_rest", 0),
            "pitches_last_3_days": fatigue.get("pitches_last_3_days", 0),
            "consecutive_appearances": fatigue.get("consecutive_days_pitched", 0),
            "fatigue_score": fatigue.get("fatigue_score", 0.0),
            "fatigue_level": display_level,
            "available": fatigue.get("availability", True),
            # Keep pitcher_id for matchup lookups
            "pitcher_id": rp.get("pitcher_id"),
        })
    return adapted


def _get_batch_matchups_for_grid(
    pitcher_ids: list[int | None],
    batter_ids: list[int | None],
) -> dict[tuple[int, int], dict]:
    """Fetch all matchup stats for the grid in a single batch call."""
    return _cached_batch_matchup_stats(tuple(pitcher_ids), tuple(batter_ids))


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Bullpen Strategy page."""
    st.title("Bullpen Strategy")
    st.caption("Analyze reliever availability, fatigue, and batter matchups.")
    st.info("""
**What this shows:** Reliever availability, fatigue levels, and optimal matchup deployment for the bullpen.

- **Green** = fresh and available, **Yellow** = moderate fatigue, **Orange** = tired but available, **Red** = unavailable
- **Fatigue Score** (0-100%) accounts for days rest, recent pitch counts, and consecutive appearances — not just "how many pitches"
- **Impact:** Managers who match fresh relievers to high-leverage spots win 3-5 more games per season vs. gut-feel bullpen management
""")

    # Determine data source
    conn = get_db_connection()
    use_real = has_data(conn)

    # Data source indicator is shown inside _get_bullpen_state

    # Team selector
    team = st.radio(
        "Team",
        options=["PHI", "Opponent"],
        horizontal=True,
        key="bullpen_team",
    )
    # Determine opponent from today's real schedule
    opp_abbr = "OPP"
    if _HAS_LIVE_FEED:
        try:
            from src.ingest.live_feed import get_phillies_game
            phi_game = get_phillies_game()
            if phi_game:
                if phi_game.get("home") == "PHI":
                    opp_abbr = phi_game.get("away", "OPP")
                else:
                    opp_abbr = phi_game.get("home", "OPP")
        except Exception:
            pass
    team_code = "PHI" if team == "PHI" else opp_abbr

    # ----- Availability table -----
    st.markdown("---")
    st.subheader("Bullpen Availability")

    bullpen = _get_bullpen_state(team=team_code)
    if not bullpen:
        st.info("No bullpen data available.")
        return

    _render_availability_table(bullpen)

    st.markdown("---")

    # ----- Live game context -----
    # Only show live context if a real game is in progress (no mock fallback)
    game = st.session_state.get("_live_game_state")
    if game and game.get("status") not in (None, "No Game", "Preview"):
        _render_live_context(game, bullpen)
        st.markdown("---")

    # ----- Matchup grid -----
    st.subheader("Reliever vs Upcoming Batters")
    _render_matchup_grid(bullpen, team_code)


# ---------------------------------------------------------------------------
# Sub-renderers
# ---------------------------------------------------------------------------

def _render_availability_table(bullpen: list[dict[str, Any]]) -> None:
    """Render the bullpen availability table with fatigue indicators."""
    rows = []
    for rp in bullpen:
        fatigue = rp.get("fatigue_level", "green")
        if fatigue in ("green", "fresh"):
            indicator = "\U0001F7E2"  # green circle
        elif fatigue in ("yellow", "moderate"):
            indicator = "\U0001F7E1"  # yellow circle
        elif fatigue in ("orange", "tired"):
            indicator = "\U0001F7E0"  # orange circle
        else:
            indicator = "\U0001F534"  # red circle (unavailable)

        rows.append({
            "Status": indicator,
            "Pitcher": rp["name"],
            "Role": rp.get("role", ""),
            "Throws": rp.get("throws", ""),
            "ERA": rp.get("era", 0),
            "FIP": rp.get("fip", 0),
            "Days Rest": rp.get("days_rest", 0),
            "Pitches (3d)": rp.get("pitches_last_3_days", 0),
            "Consec. App.": rp.get("consecutive_appearances", 0),
            "Fatigue": f"{rp.get('fatigue_score', 0):.0%}",
            "Available": "Yes" if rp.get("available", True) else "No",
        })

    df = pd.DataFrame(rows)

    def _highlight_avail(row: pd.Series) -> list[str]:
        if row["Available"] == "No":
            return ["background-color: rgba(231, 76, 60, 0.2)"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_highlight_avail, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # Summary counts
    avail_count = sum(1 for rp in bullpen if rp.get("available", True))
    total = len(bullpen)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Available", f"{avail_count}/{total}")
    fresh = sum(1 for rp in bullpen if rp.get("fatigue_level") in ("green", "fresh"))
    c2.metric("Fresh (Green)", fresh)
    tired = sum(1 for rp in bullpen if rp.get("fatigue_level") in ("orange", "tired"))
    c3.metric("Tired (Orange)", tired)
    taxed = sum(1 for rp in bullpen if rp.get("fatigue_level") in ("red", "unavailable"))
    c4.metric("Unavailable (Red)", taxed)


def _render_live_context(game: dict[str, Any], bullpen: list[dict[str, Any]]) -> None:
    """Show leverage index and recommended reliever during a live game."""
    st.subheader("Live Game Context")

    li = game.get("leverage_index", 1.0)
    inning = game.get("inning", 1)
    half = game.get("half", "top")

    c1, c2, c3 = st.columns(3)
    c1.metric("Leverage Index", f"{li:.2f}")
    c2.metric("Inning", f"{inning} {'Top' if half == 'top' else 'Bot'}")

    # Try real recommendation engine
    conn = get_db_connection()
    recommendation_used = False
    if has_data(conn) and not _USE_MOCK_BULLPEN:
        try:
            # Build game_state dict for the recommendation engine
            game_state = {
                "inning": inning,
                "outs": game.get("outs", 0),
                "runners": {
                    1: game.get("on_1b", False),
                    2: game.get("on_2b", False),
                    3: game.get("on_3b", False),
                },
                "score_diff": game.get("home_score", 0) - game.get("away_score", 0),
                "lineup": [],
                "current_batter_idx": 0,
            }
            recommendations = _real_recommend_reliever(conn, "PHI", game_state)
            if recommendations:
                rec = recommendations[0]
                c3.metric("Recommended", rec["name"])
                st.success(
                    f"**{rec['name']}** -- "
                    f"Score: {rec.get('recommendation_score', 0):.3f}, "
                    f"{rec.get('fatigue_level', 'unknown')} fatigue. "
                    f"{rec.get('reasoning', '')}"
                )
                recommendation_used = True
        except Exception:
            pass

    if not recommendation_used:
        # Simple heuristic fallback: lowest ERA among available green/yellow
        available = [rp for rp in bullpen if rp.get("available", True)]
        if available:
            available_sorted = sorted(available, key=lambda r: r.get("era", 99))
            rec = available_sorted[0]
            c3.metric("Recommended", rec["name"])
            st.success(
                f"**{rec['name']}** ({rec.get('throws', '')}HP) -- "
                f"ERA {rec.get('era', 0):.2f}, FIP {rec.get('fip', 0):.2f}, "
                f"{rec.get('days_rest', 0)} days rest, "
                f"fatigue {rec.get('fatigue_score', 0):.0%}"
            )
        else:
            c3.metric("Recommended", "None available")
            st.error("All relievers are currently unavailable or fatigued.")


def _render_matchup_grid(bullpen: list[dict[str, Any]], team_code: str) -> None:
    """Render a reliever-vs-batter matchup heatmap grid."""
    available = [rp for rp in bullpen if rp.get("available", True)]
    if not available:
        st.info("No available relievers for matchup grid.")
        return

    # Choose which batters to show (next 6 from the opposing team)
    # Try live lineup from feed if available
    batters: list[dict] = []
    feed = st.session_state.get("_live_feed")
    if feed is not None and _HAS_LIVE_FEED:
        try:
            side = "away" if team_code == "PHI" else "home"
            live_lineup = _real_parse_lineup(feed, team=side)
            if live_lineup:
                batters = [{"name": b["name"], "batter_id": b.get("id")} for b in live_lineup[:6]]
        except Exception:
            pass

    if not batters:
        st.caption("No live lineup available -- matchup grid requires a live game or database data.")
        return

    # Resolve batter IDs from DB if needed
    conn = get_db_connection()
    for b in batters:
        if b.get("batter_id") is None and has_data(conn):
            b["batter_id"] = get_player_id_by_name(conn, b["name"])

    pitcher_names = [rp["name"] for rp in available[:6]]
    pitcher_ids = [rp.get("pitcher_id") for rp in available[:6]]
    batter_names = [b["name"] for b in batters]
    batter_ids = [b.get("batter_id") for b in batters]

    # Batch-fetch ALL matchup stats in ONE query instead of N*M individual queries
    batch_results = _get_batch_matchups_for_grid(pitcher_ids, batter_ids)

    # Build wOBA matrix from the batch results
    z: list[list[float]] = []
    text: list[list[str]] = []
    for rp_idx, rp_name in enumerate(pitcher_names):
        row_z = []
        row_t = []
        pid = pitcher_ids[rp_idx]
        for bat_idx, bat_name in enumerate(batter_names):
            bid = batter_ids[bat_idx]
            if pid is not None and bid is not None:
                result = batch_results.get((pid, bid), {})
                woba = result.get("woba", 0.320)
            else:
                woba = 0.320
            row_z.append(woba)
            row_t.append(f"{woba:.3f}")
        z.append(row_z)
        text.append(row_t)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=batter_names,
        y=pitcher_names,
        text=text,
        texttemplate="%{text}",
        colorscale=[
            [0.0, "#2ECC71"],   # green = favorable for pitcher (low wOBA)
            [0.5, "#F7DC6F"],   # neutral
            [1.0, "#E74C3C"],   # red = unfavorable for pitcher (high wOBA)
        ],
        zmin=0.250,
        zmax=0.420,
        colorbar=dict(title="Est. wOBA"),
        hovertemplate="Pitcher: %{y}<br>Batter: %{x}<br>wOBA: %{text}<extra></extra>",
    ))

    fig.update_layout(
        title="Matchup Grid (Est. wOBA -- lower = pitcher advantage)",
        xaxis=dict(title="Batter", tickangle=-30),
        yaxis=dict(title="Reliever", autorange="reversed"),
        template="plotly_dark",
        height=max(300, len(pitcher_names) * 55 + 100),
        margin=dict(l=140, r=40, t=60, b=100),
    )

    st.plotly_chart(fig, use_container_width=True, key="bullpen_matchup_grid")
