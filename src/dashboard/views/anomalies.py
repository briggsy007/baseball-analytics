"""
Anomaly Detection page.

Displays real-time and historical anomaly alerts, velocity trends,
release point scatter, spin rate trends, pitch mix comparisons,
and batter anomaly panels.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_USE_MOCK_ANOMALY = False
try:
    from src.analytics.anomaly import (
        GameAnomalyMonitor,
        calculate_pitcher_baselines as _real_calculate_pitcher_baselines,
        detect_batter_anomalies as _real_detect_batter_anomalies,
    )
except ImportError:
    _USE_MOCK_ANOMALY = True

from src.dashboard.mock_data import (
    ARSENALS,
    PHILLIES_SP,
    PITCH_TYPES,
)
from src.dashboard.db_helper import (
    get_db_connection,
    has_data,
    get_player_id_by_name,
    get_all_pitchers,
    get_all_batters,
)


# ---------------------------------------------------------------------------
# Cached wrappers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _cached_player_id_by_name(name: str) -> int | None:
    """Cached player ID lookup (TTL 1 hour) -- avoids repeated DB queries."""
    conn = get_db_connection()
    return get_player_id_by_name(conn, name)


@st.cache_data(ttl=300)
def _cached_pitcher_baselines(pitcher_id: int) -> dict | None:
    """Cached pitcher baselines from the real analytics module (TTL 5 min)."""
    conn = get_db_connection()
    if conn is None or _USE_MOCK_ANOMALY:
        return None
    try:
        return _real_calculate_pitcher_baselines(conn, pitcher_id)
    except Exception:
        return None


@st.cache_data(ttl=300)
def _cached_batter_anomalies(batter_id: int) -> list[dict] | None:
    """Cached batter anomaly alerts from the real analytics module (TTL 5 min)."""
    conn = get_db_connection()
    if conn is None or _USE_MOCK_ANOMALY:
        return None
    try:
        return _real_detect_batter_anomalies(conn, batter_id)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _cached_all_pitchers() -> list[dict]:
    """Cached list of all pitchers in the DB (TTL 1 hour)."""
    conn = get_db_connection()
    return get_all_pitchers(conn)


@st.cache_data(ttl=3600)
def _cached_all_batters() -> list[dict]:
    """Cached list of all batters in the DB (TTL 1 hour)."""
    conn = get_db_connection()
    return get_all_batters(conn)


# ---------------------------------------------------------------------------
# Adapter functions
# ---------------------------------------------------------------------------

def _get_live_anomalies() -> list[dict[str, Any]]:
    """Return live anomaly alerts from the active game session.

    Only returns mock data if the anomaly module is missing AND there
    is no live game state available.
    """
    if _USE_MOCK_ANOMALY:
        # Module not importable -- no way to produce real alerts
        return []

    # Try to pull live anomalies from the live_game page's session data
    # The live_game page runs the GameAnomalyMonitor and stores alerts
    game = st.session_state.get("_live_game_state")
    if game is None or game.get("status") in (None, "No Game", "Preview"):
        return []  # No live game -- no alerts to show

    conn = get_db_connection()
    if not has_data(conn):
        return []

    try:
        feed = st.session_state.get("_live_feed")
        if feed is None:
            return []

        pitcher = game.get("pitcher", {})
        pitcher_id = pitcher.get("id")
        if not pitcher_id:
            return []

        from src.ingest.live_feed import parse_pitch_events as _parse_pitch_events

        # Reuse monitor from session_state; only create a new one when
        # the pitcher (i.e. game context) changes.
        monitor_key = "_anomaly_monitor"
        prev_pitcher_key = "_anomaly_monitor_pitcher_id"

        prev_pitcher_id = st.session_state.get(prev_pitcher_key)
        if prev_pitcher_id != pitcher_id or monitor_key not in st.session_state:
            monitor = GameAnomalyMonitor(pitcher_id=pitcher_id, conn=conn)
            st.session_state[monitor_key] = monitor
            st.session_state[prev_pitcher_key] = pitcher_id
        else:
            monitor = st.session_state[monitor_key]

        # Only re-process pitches if the feed has new data since last check
        pitch_events = _parse_pitch_events(feed)
        feed_pitch_count = len(pitch_events)
        last_count_key = "_anomaly_last_pitch_count"
        last_count = st.session_state.get(last_count_key, 0)

        if feed_pitch_count != last_count:
            monitor._pitches = []
            for pe in pitch_events:
                monitor.add_pitch({
                    "pitch_type": pe.get("pitch_type", ""),
                    "release_speed": pe.get("start_speed"),
                    "release_spin_rate": pe.get("release_spin_rate"),
                    "release_pos_x": pe.get("release_pos_x"),
                    "release_pos_z": pe.get("release_pos_z"),
                })
            st.session_state[last_count_key] = feed_pitch_count
            alerts = monitor.check_all()
            st.session_state["_anomaly_cached_alerts"] = alerts
        else:
            alerts = st.session_state.get("_anomaly_cached_alerts", [])

        if alerts:
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
    except Exception:
        return []


def _get_historical_anomalies() -> list[dict[str, Any]]:
    """Return historical anomaly alerts.

    When real DB data is available, queries for recent anomalies.
    Falls back to mock only if no real data source is available.
    """
    conn = get_db_connection()
    if not has_data(conn) or _USE_MOCK_ANOMALY:
        # No DB or no anomaly module -- return empty instead of mock
        return []
    # Real historical anomalies would need a stored alert table;
    # for now return empty rather than fake data
    return []


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Anomaly Detection page."""
    st.title("Anomaly Detection")
    st.caption("Monitor pitcher and batter performance deviations in real time.")
    st.info("""
**What this shows:** Statistical outliers in pitcher and batter performance — things that deviate significantly from their established baselines.

- **Velocity drops** of 1.5+ mph from a pitcher's game average often precede blowup innings or signal injury
- **Spin rate changes** of 5%+ can indicate mechanical adjustments, fatigue, or (historically) substance usage changes
- **Release point drift** means the pitcher's arm slot is changing — a leading indicator of both declining effectiveness and injury risk
""")

    # Determine data source
    conn = get_db_connection()
    use_real = has_data(conn)

    if use_real and not _USE_MOCK_ANOMALY:
        st.info("Using real data from database for anomaly baselines.")
    else:
        st.warning("Anomaly detection requires pitch data in the database. Run backfill.py to load data.")

    # ----- Active alerts panel -----
    st.subheader("Active Alerts (Live Game)")
    _render_active_alerts()

    st.markdown("---")

    # ----- Historical anomaly log -----
    st.subheader("Recent Anomaly Log")
    _render_historical_log()

    st.markdown("---")

    # ----- Pitcher anomaly deep-dive -----
    st.subheader("Pitcher Deep Dive")

    # Build pitcher list from DB or mock
    if use_real and not _USE_MOCK_ANOMALY:
        db_pitchers = _cached_all_pitchers()
    else:
        db_pitchers = []

    if db_pitchers:
        pitcher_options = [p["full_name"] for p in db_pitchers]
        pitcher_id_map = {p["full_name"]: p["player_id"] for p in db_pitchers}
    else:
        st.info("Insufficient data for pitcher baselines. Load pitch data to enable anomaly deep dive.")
        return

    selected_pitcher = st.selectbox(
        "Select Pitcher",
        options=pitcher_options,
        index=0,
        key="anomaly_pitcher",
    )

    # Resolve pitcher ID
    pitcher_id = pitcher_id_map.get(selected_pitcher)
    if pitcher_id is None and use_real:
        pitcher_id = _cached_player_id_by_name(selected_pitcher)

    # Try to get real baselines
    real_baselines: dict | None = None
    if pitcher_id is not None and use_real and not _USE_MOCK_ANOMALY:
        real_baselines = _cached_pitcher_baselines(pitcher_id)

    tab_velo, tab_rp, tab_spin, tab_mix = st.tabs(
        ["Velocity Trend", "Release Point", "Spin Rate", "Pitch Mix"]
    )

    with tab_velo:
        _render_velocity_trend(selected_pitcher, real_baselines)

    with tab_rp:
        _render_release_point(selected_pitcher, real_baselines)

    with tab_spin:
        _render_spin_rate_trend(selected_pitcher, real_baselines)

    with tab_mix:
        _render_pitch_mix_comparison(selected_pitcher, real_baselines)

    st.markdown("---")

    # ----- Batter anomalies -----
    st.subheader("Batter Anomalies")
    _render_batter_anomalies(use_real)


# ---------------------------------------------------------------------------
# Sub-renderers
# ---------------------------------------------------------------------------

def _render_active_alerts() -> None:
    """Show current live game anomaly alerts with grouping and severity filter.

    Alerts of the same type are grouped into a single display entry with
    a count badge.  A severity filter lets the user toggle between seeing
    all alerts or only critical ones.
    """
    game = st.session_state.get("_live_game_state")
    if game is None or game.get("status") in (None, "No Game", "Preview"):
        st.info("Anomaly detection active during live games.")
        return

    alerts = _get_live_anomalies()
    if not alerts:
        st.success("No active anomalies detected.")
        return

    # --- Severity filter -------------------------------------------------
    severity_filter = st.radio(
        "Alert filter",
        options=["All alerts", "Critical only"],
        horizontal=True,
        key="anomaly_severity_filter",
    )
    if severity_filter == "Critical only":
        alerts = [a for a in alerts if a.get("severity") in ("high", "critical")]
    if not alerts:
        st.success("No critical anomalies detected.")
        return

    # --- Group alerts by type --------------------------------------------
    grouped: dict[str, list[dict]] = {}
    for alert in alerts:
        a_type = alert.get("type", "unknown")
        grouped.setdefault(a_type, []).append(alert)

    # --- Count badge -----------------------------------------------------
    total_active = len(grouped)
    st.markdown(f"**{total_active} active alert{'s' if total_active != 1 else ''}**")

    # --- Render one entry per group --------------------------------------
    for a_type, group in grouped.items():
        # Use the most recent (first in the list, since alerts arrive
        # sorted by pitch number descending) alert as the representative.
        representative = group[0]
        severity = representative.get("severity", "low")
        msg = representative.get("message", "")
        player = representative.get("player", "")
        display_type = a_type.replace("_", " ").title()
        count = len(group)
        count_badge = f" (x{count})" if count > 1 else ""

        col_icon, col_body = st.columns([1, 8])
        with col_icon:
            if severity in ("high", "critical"):
                st.markdown(
                    "<div style='font-size:2rem; text-align:center;'>!!!</div>",
                    unsafe_allow_html=True,
                )
            elif severity in ("medium", "warning"):
                st.markdown(
                    "<div style='font-size:2rem; text-align:center;'>(i)</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='font-size:2rem; text-align:center;'>--</div>",
                    unsafe_allow_html=True,
                )
        with col_body:
            st.markdown(f"**{display_type}{count_badge}** | {player}")
            st.caption(msg)


def _render_historical_log() -> None:
    """Show a table of recent historical anomalies."""
    historical = _get_historical_anomalies()
    if not historical:
        st.info("No recent anomalies in the log.")
        return

    df = pd.DataFrame(historical)
    cols = ["game_date", "severity", "player", "type", "message"]
    available = [c for c in cols if c in df.columns]
    df_show = df[available].rename(columns={
        "game_date": "Date",
        "severity": "Severity",
        "player": "Player",
        "type": "Type",
        "message": "Details",
    })
    if "Type" in df_show.columns:
        df_show["Type"] = df_show["Type"].str.replace("_", " ").str.title()
    st.dataframe(df_show, use_container_width=True, hide_index=True)


def _render_velocity_trend(pitcher_name: str, baselines: dict | None = None) -> None:
    """Velocity trend chart across recent games.

    If real baselines are available, uses the season average from the
    baselines; otherwise falls back to mock data.
    """
    st.markdown(f"**{pitcher_name} -- Fastball Velocity Trend**")

    # Try real pitch data from DB first
    velo_df = pd.DataFrame()
    conn = get_db_connection()
    if has_data(conn) and not _USE_MOCK_ANOMALY:
        pitcher_id = _cached_player_id_by_name(pitcher_name)
        if pitcher_id is not None:
            try:
                _df = conn.execute("""
                    SELECT game_date, pitch_number,
                           release_speed, pitch_type
                    FROM pitches
                    WHERE pitcher_id = $1 AND pitch_type IN ('FF', 'SI')
                    ORDER BY game_date DESC, pitch_number
                    LIMIT 1000
                """, [pitcher_id]).fetchdf()
                if not _df.empty:
                    _df["game_date"] = _df["game_date"].astype(str)
                    velo_df = _df
            except Exception:
                pass

    if velo_df.empty:
        st.info("No velocity data available for this pitcher.")
        return

    fig = go.Figure()

    game_dates = velo_df["game_date"].unique()
    colors = ["#E81828", "#FF6B35", "#F39C12", "#2ECC71", "#3498DB",
              "#9B59B6", "#1ABC9C", "#E74C3C", "#2980B9", "#8E44AD"]
    for idx, gd in enumerate(game_dates):
        subset = velo_df[velo_df["game_date"] == gd]
        fig.add_trace(go.Scatter(
            x=subset["pitch_number"],
            y=subset["release_speed"],
            mode="markers",
            marker=dict(size=4, color=colors[idx % len(colors)], opacity=0.6),
            name=gd,
            hovertemplate=f"Game {gd}<br>Pitch #%{{x}}<br>%{{y:.1f}} mph<extra></extra>",
        ))

    # Season average line -- prefer real baselines
    if baselines is not None:
        by_pt = baselines.get("by_pitch_type", {})
        ff_baseline = by_pt.get("FF", {})
        season_avg = ff_baseline.get("mean_velo", 0.0)
        if season_avg > 0:
            fig.add_hline(
                y=season_avg,
                line=dict(color="cyan", width=2, dash="dash"),
                annotation_text=f"DB Baseline: {season_avg:.1f}",
                annotation_position="top right",
            )
    else:
        season_avg = velo_df["release_speed"].mean()
        fig.add_hline(y=season_avg, line=dict(color="white", width=1, dash="dash"),
                      annotation_text=f"Season Avg: {season_avg:.1f}", annotation_position="top right")

    fig.update_layout(
        xaxis=dict(title="Pitch Number in Game"),
        yaxis=dict(title="Velocity (mph)"),
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=30, t=30, b=50),
        legend=dict(title="Game Date", orientation="v"),
    )
    st.plotly_chart(fig, use_container_width=True, key="anomaly_velo_trend")


def _render_release_point(pitcher_name: str, baselines: dict | None = None) -> None:
    """Release point scatter plot."""
    st.markdown(f"**{pitcher_name} -- Release Point**")

    # Try real pitch data from DB first
    rp_df = pd.DataFrame()
    conn = get_db_connection()
    if has_data(conn) and not _USE_MOCK_ANOMALY:
        pitcher_id = _cached_player_id_by_name(pitcher_name)
        if pitcher_id is not None:
            try:
                _df = conn.execute("""
                    SELECT release_pos_x, release_pos_z, pitch_type
                    FROM pitches
                    WHERE pitcher_id = $1
                      AND release_pos_x IS NOT NULL
                      AND release_pos_z IS NOT NULL
                    ORDER BY game_date DESC
                    LIMIT 200
                """, [pitcher_id]).fetchdf()
                if not _df.empty:
                    rp_df = _df
            except Exception:
                pass

    if rp_df.empty:
        st.info("No release point data available for this pitcher.")
        return

    fig = go.Figure()
    pitch_types = rp_df["pitch_type"].unique()
    for pt in pitch_types:
        subset = rp_df[rp_df["pitch_type"] == pt]
        color = PITCH_TYPES.get(pt, {}).get("color", "#FFFFFF")
        label = PITCH_TYPES.get(pt, {}).get("label", pt)
        fig.add_trace(go.Scatter(
            x=subset["release_pos_x"],
            y=subset["release_pos_z"],
            mode="markers",
            marker=dict(size=5, color=color, opacity=0.6),
            name=label,
            hovertemplate=f"{label}<br>X: %{{x:.2f}}<br>Z: %{{y:.2f}}<extra></extra>",
        ))

    # Add real baseline centroid if available
    if baselines is not None:
        rp_baseline = baselines.get("release_point", {})
        mean_x = rp_baseline.get("mean_x", 0)
        mean_z = rp_baseline.get("mean_z", 0)
        if mean_x != 0 and mean_z != 0:
            fig.add_trace(go.Scatter(
                x=[mean_x], y=[mean_z],
                mode="markers",
                marker=dict(size=15, color="cyan", symbol="x", line=dict(width=2)),
                name="DB Baseline Avg",
                hovertemplate=f"Baseline<br>X: {mean_x:.2f}<br>Z: {mean_z:.2f}<extra></extra>",
            ))

    fig.update_layout(
        xaxis=dict(title="Release X (ft)"),
        yaxis=dict(title="Release Z (ft)", scaleanchor="x", scaleratio=1),
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=30, t=30, b=50),
    )
    st.plotly_chart(fig, use_container_width=True, key="anomaly_release_pt")


def _render_spin_rate_trend(pitcher_name: str, baselines: dict | None = None) -> None:
    """Spin rate trend by game -- uses real DB data."""
    st.markdown(f"**{pitcher_name} -- Spin Rate by Game**")

    # Try real DB data first
    conn = get_db_connection()
    spin_df = pd.DataFrame()
    if has_data(conn) and not _USE_MOCK_ANOMALY:
        pitcher_id = _cached_player_id_by_name(pitcher_name)
        if pitcher_id is not None:
            try:
                spin_df = conn.execute("""
                    SELECT game_date, pitch_type,
                           AVG(release_spin_rate) AS avg_spin
                    FROM pitches
                    WHERE pitcher_id = $1
                      AND release_spin_rate IS NOT NULL
                    GROUP BY game_date, pitch_type
                    ORDER BY game_date DESC
                    LIMIT 100
                """, [pitcher_id]).fetchdf()
            except Exception:
                pass

    if spin_df.empty:
        st.info("No spin rate data available for this pitcher.")
        return

    fig = go.Figure()
    for pt in spin_df["pitch_type"].unique():
        subset = spin_df[spin_df["pitch_type"] == pt].sort_values("game_date")
        color = PITCH_TYPES.get(pt, {}).get("color", "#FFFFFF")
        label = PITCH_TYPES.get(pt, {}).get("label", pt)
        fig.add_trace(go.Scatter(
            x=subset["game_date"].astype(str),
            y=subset["avg_spin"],
            mode="lines+markers",
            marker=dict(size=7, color=color),
            line=dict(color=color, width=2),
            name=label,
            hovertemplate="%{y:.0f} RPM<extra></extra>",
        ))

        # Add baseline horizontal line if real baselines available
        if baselines is not None:
            real_pt_baseline = baselines.get("by_pitch_type", {}).get(pt)
            if real_pt_baseline is not None:
                spin_mean = real_pt_baseline.get("mean_spin", 0)
                if spin_mean > 0:
                    fig.add_hline(
                        y=spin_mean,
                        line=dict(color=color, width=1, dash="dot"),
                        annotation_text=f"{pt} baseline: {spin_mean:.0f}",
                    )

    fig.update_layout(
        xaxis=dict(title="Game"),
        yaxis=dict(title="Avg Spin Rate (RPM)"),
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=30, t=30, b=50),
    )
    st.plotly_chart(fig, use_container_width=True, key="anomaly_spin_trend")


def _render_pitch_mix_comparison(pitcher_name: str, baselines: dict | None = None) -> None:
    """Compare pitch mix using real DB data."""
    st.markdown(f"**{pitcher_name} -- Pitch Mix: Season Usage**")

    # Try real pitch mix from DB
    conn = get_db_connection()
    mix_df = pd.DataFrame()
    if has_data(conn) and not _USE_MOCK_ANOMALY:
        pitcher_id = _cached_player_id_by_name(pitcher_name)
        if pitcher_id is not None:
            try:
                mix_df = conn.execute("""
                    SELECT pitch_type,
                           COUNT(*) AS cnt,
                           COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS pct
                    FROM pitches
                    WHERE pitcher_id = $1
                      AND pitch_type IS NOT NULL
                    GROUP BY pitch_type
                    ORDER BY cnt DESC
                """, [pitcher_id]).fetchdf()
            except Exception:
                pass

    if mix_df.empty:
        st.info("No pitch mix data available for this pitcher.")
        return

    labels = [PITCH_TYPES.get(pt, {}).get("label", pt) for pt in mix_df["pitch_type"]]
    season_pcts = mix_df["pct"].values.round(1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=season_pcts, name="Season",
        marker_color=[PITCH_TYPES.get(pt, {}).get("color", "#FFFFFF") for pt in mix_df["pitch_type"]],
    ))

    fig.update_layout(
        xaxis=dict(title="Pitch Type"),
        yaxis=dict(title="Usage %", ticksuffix="%"),
        template="plotly_dark",
        height=350,
        margin=dict(l=50, r=30, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True, key="anomaly_pitch_mix")


def _render_batter_anomalies(use_real: bool = False) -> None:
    """Show batter-side anomaly panels: exit velo and chase rate trends.

    When real DB data is available, also runs ``detect_batter_anomalies``
    and displays any alerts.
    """
    # Show real batter anomaly alerts if available
    if use_real and not _USE_MOCK_ANOMALY:
        db_batters = _cached_all_batters()
        if db_batters:
            batter_name = st.selectbox(
                "Select Batter for Anomaly Check",
                options=[b["full_name"] for b in db_batters],
                index=0,
                key="anomaly_batter_select",
            )
            batter_id = next(
                (b["player_id"] for b in db_batters if b["full_name"] == batter_name),
                None,
            )
            if batter_id is not None:
                alerts = _cached_batter_anomalies(batter_id)
                if alerts:
                    for alert in alerts:
                        severity = alert.get("severity", "low")
                        msg = alert.get("message", "")
                        a_type = alert.get("type", "").replace("_", " ").title()
                        if severity in ("critical", "high"):
                            st.error(f"**{a_type}**: {msg}")
                        elif severity in ("warning", "medium"):
                            st.warning(f"**{a_type}**: {msg}")
                        else:
                            st.info(f"**{a_type}**: {msg}")
                else:
                    st.success(f"No anomalies detected for {batter_name}.")
            st.markdown("---")

    if not use_real:
        st.info("Batter anomaly trends require pitch data in the database.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Exit Velocity Trend (Last 10 Games)**")
        st.info("Exit velocity trend requires batted ball data in the database.")

    with col2:
        st.markdown("**Chase Rate Trend (Last 10 Games)**")
        st.info("Chase rate trend requires pitch data in the database.")

    # Placeholder -- keeping layout consistent. Charts would be rendered
    # here from real DB data once batted ball and plate discipline data
    # is loaded.
