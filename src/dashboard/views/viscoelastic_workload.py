"""
Viscoelastic Workload Response (VWR) dashboard view.

Visualizes biomechanical arm stress-strain modeling for pitchers using a
Standard Linear Solid viscoelastic model.  Shows strain timelines, VWR
gauge, recovery projections, season workload charts, and a strained-arms
leaderboard.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.db_helper import get_db_connection, has_data, get_all_pitchers

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_VWR_AVAILABLE = False
try:
    from src.analytics.viscoelastic_workload import (
        batch_calculate,
        calculate_vwr,
        predict_recovery,
    )
    _VWR_AVAILABLE = True
except ImportError:
    pass

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

_CACHE_AVAILABLE = False
try:
    from src.dashboard.cache_reader import get_cached_leaderboard, cache_age_display
    _CACHE_AVAILABLE = True
except Exception:
    pass


# VWR colour zones
_VWR_ZONES = [
    (0, 25, "#2ECC71", "Fresh"),       # green
    (25, 50, "#F1C40F", "Moderate"),    # yellow
    (50, 75, "#E67E22", "Elevated"),    # orange
    (75, 100, "#E74C3C", "High Risk"),  # red
]


def _vwr_color(score: float) -> str:
    """Return the colour for a VWR score."""
    for lo, hi, color, _ in _VWR_ZONES:
        if lo <= score < hi:
            return color
    return "#E74C3C"  # default red for >= 75


def _vwr_label(score: float) -> str:
    """Return the zone label for a VWR score."""
    for lo, hi, _, label in _VWR_ZONES:
        if lo <= score < hi:
            return label
    return "High Risk"


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Viscoelastic Workload Response analysis page."""
    st.title("Viscoelastic Workload Response (VWR)")
    st.caption(
        "Biomechanical arm stress-strain model using Standard Linear Solid "
        "viscoelasticity.  Higher VWR = more accumulated arm strain."
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**VWR models a pitcher's arm like a biomechanical material** — each pitch adds strain that partially recovers over time, following real tissue physics.

- **VWR 0-40** = low strain (well-rested, recovered from recent outings)
- **VWR 40-70** = moderate strain (accumulating load, monitor closely)
- **VWR 70-90** = high strain (increased injury risk — consider extra rest)
- **VWR 90+** = critical (tissue-level strain hasn't recovered — strong recommendation to skip a start or extend rest)
- **Key difference from pitch count:** Two pitchers at 90 pitches can have wildly different VWR based on rest days, pitch effort, and recovery time between outings
- **Recovery predictor** shows exactly how many rest days are needed to return to a safe VWR level
- **Impact:** Monitoring VWR during workload ramp-ups (especially for young arms) can prevent the overuse injuries that end careers
""")

    if not _VWR_AVAILABLE:
        st.error(
            "The `viscoelastic_workload` analytics module could not be imported. "
            "Ensure `scipy` is installed:\n\n```\npip install scipy\n```"
        )
        return

    if not _PLOTLY_AVAILABLE:
        st.error("Plotly is required for charts. Install with `pip install plotly`.")
        return

    conn = get_db_connection()
    if conn is None or not has_data(conn):
        st.warning(
            "No pitch data available.  Run the data backfill pipeline first "
            "(`python scripts/backfill.py`)."
        )
        return

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_pitcher, tab_recovery, tab_season, tab_leaderboard = st.tabs([
        "Pitcher Analysis",
        "Recovery Predictor",
        "Season Workload",
        "Strain Leaderboard",
    ])

    with tab_pitcher:
        _render_pitcher_analysis(conn)

    with tab_recovery:
        _render_recovery_predictor(conn)

    with tab_season:
        _render_season_workload(conn)

    with tab_leaderboard:
        _render_leaderboard(conn)


# ---------------------------------------------------------------------------
# Pitcher selector helper
# ---------------------------------------------------------------------------


def _get_pitcher_selector(conn, key_suffix: str) -> int | None:
    """Render a pitcher selector and return the selected pitcher_id."""
    pitchers = get_all_pitchers(conn)
    if not pitchers:
        try:
            fallback = conn.execute(
                "SELECT DISTINCT pitcher_id FROM pitches LIMIT 500"
            ).fetchdf()
            pitchers = [
                {"player_id": int(pid), "full_name": f"Pitcher {pid}"}
                for pid in fallback["pitcher_id"]
            ]
        except Exception:
            st.info("No pitcher data available.")
            return None

    pitcher_names = {
        p.get("full_name", f"ID {p['player_id']}"): p["player_id"]
        for p in pitchers
    }
    selected_name = st.selectbox(
        "Select Pitcher",
        options=sorted(pitcher_names.keys()),
        key=f"vwr_pitcher_{key_suffix}",
    )
    if not selected_name:
        return None
    return pitcher_names[selected_name]


# ---------------------------------------------------------------------------
# Pitcher Analysis tab
# ---------------------------------------------------------------------------


def _render_pitcher_analysis(conn) -> None:
    """Individual pitcher VWR analysis with strain timeline and gauge."""
    st.subheader("Pitcher Strain Profile")

    pitcher_id = _get_pitcher_selector(conn, "analysis")
    if pitcher_id is None:
        return

    with st.spinner("Computing VWR..."):
        try:
            result = calculate_vwr(conn, pitcher_id)
        except Exception as exc:
            st.error(f"Error computing VWR: {exc}")
            return

    vwr = result.get("vwr_score")
    strain = result.get("current_strain")
    params = result.get("parameters", {})

    # ── Key metrics row ───────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    if vwr is not None:
        col1.metric("VWR Score", f"{vwr:.1f}")
    else:
        col1.metric("VWR Score", "N/A")
    col2.metric("Current Strain", f"{strain:.4f}" if strain else "N/A")
    col3.metric("Recovery Tau", f"{params.get('tau', 'N/A')} hrs")
    col4.metric("Pitches Analyzed", params.get("n_pitches", 0))

    if vwr is None:
        st.info(
            "Not enough pitch data for this pitcher to compute VWR. "
            "Need at least 500 career pitches for parameter fitting."
        )
        return

    # ── VWR Gauge ──────────────────────────────────────────────────────────
    st.markdown("---")
    _render_vwr_gauge(vwr)

    # ── Strain Timeline ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Cumulative Strain Timeline**")
    _render_strain_timeline(result)


def _render_vwr_gauge(vwr_score: float) -> None:
    """Render a gauge-style VWR indicator with colour zones."""
    color = _vwr_color(vwr_score)
    label = _vwr_label(vwr_score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=vwr_score,
        title={"text": f"VWR Score -- {label}", "font": {"size": 18}},
        number={"font": {"size": 48}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 25], "color": "rgba(46, 204, 113, 0.2)"},
                {"range": [25, 50], "color": "rgba(241, 196, 15, 0.2)"},
                {"range": [50, 75], "color": "rgba(230, 126, 34, 0.2)"},
                {"range": [75, 100], "color": "rgba(231, 76, 60, 0.2)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": vwr_score,
            },
        },
    ))
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    st.plotly_chart(fig, use_container_width=True, key="vwr_gauge")


def _render_strain_timeline(result: dict) -> None:
    """Line chart of cumulative strain over the season with game markers."""
    appearances = result.get("game_appearances", [])
    if not appearances:
        st.info("No game appearances to display.")
        return

    dates = [a["game_date"] for a in appearances]
    peak_strains = [a["peak_strain"] for a in appearances]
    end_strains = [a["end_strain"] for a in appearances]
    pitches = [a["pitches"] for a in appearances]

    fig = go.Figure()

    # End-of-game strain line
    fig.add_trace(go.Scatter(
        x=dates,
        y=end_strains,
        mode="lines+markers",
        line=dict(color="#E81828", width=2.5),
        marker=dict(
            size=[max(4, p // 10) for p in pitches],
            color="#E81828",
        ),
        name="End-of-Game Strain",
        hovertemplate=(
            "Date: %{x}<br>"
            "Strain: %{y:.4f}<br>"
            "<extra></extra>"
        ),
    ))

    # Peak strain markers
    fig.add_trace(go.Scatter(
        x=dates,
        y=peak_strains,
        mode="markers",
        marker=dict(
            size=6,
            color="#FFD600",
            symbol="diamond",
        ),
        name="Peak Strain",
    ))

    fig.update_layout(
        xaxis=dict(title="Game Date"),
        yaxis=dict(title="Cumulative Strain"),
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=30, t=30, b=50),
        legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True, key="vwr_timeline")


# ---------------------------------------------------------------------------
# Recovery Predictor tab
# ---------------------------------------------------------------------------


def _render_recovery_predictor(conn) -> None:
    """Recovery prediction with rest-day slider and projected VWR curve."""
    st.subheader("Recovery Predictor")
    st.caption(
        "Project how VWR will change with additional rest days. "
        "Uses the SLS model's natural strain dissipation."
    )

    pitcher_id = _get_pitcher_selector(conn, "recovery")
    if pitcher_id is None:
        return

    rest_days = st.slider(
        "Rest Days to Project",
        min_value=1,
        max_value=14,
        value=5,
        key="vwr_rest_days",
    )

    with st.spinner("Predicting recovery..."):
        try:
            result = predict_recovery(conn, pitcher_id, rest_days=rest_days)
        except Exception as exc:
            st.error(f"Error predicting recovery: {exc}")
            return

    if not result["recovery_curve"]:
        st.info("Not enough data to predict recovery for this pitcher.")
        return

    # ── Current vs Projected metrics ──────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Current VWR", f"{result['current_vwr']:.1f}" if result['current_vwr'] else "N/A")
    col2.metric(
        f"Projected VWR (Day {rest_days})",
        f"{result['projected_vwr']:.1f}" if result['projected_vwr'] else "N/A",
    )
    if result['current_strain'] and result['projected_strain']:
        delta = result['projected_strain'] - result['current_strain']
        col3.metric(
            "Strain Change",
            f"{result['projected_strain']:.4f}",
            delta=f"{delta:+.4f}",
            delta_color="inverse",
        )

    # ── Recovery curve chart ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**Projected Recovery Curve**")

    curve = result["recovery_curve"]
    days = [pt["day"] for pt in curve]
    vwr_vals = [pt["vwr"] for pt in curve]
    strain_vals = [pt["strain"] for pt in curve]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("VWR Score Over Rest", "Raw Strain Over Rest"),
    )

    # VWR curve
    colors = [_vwr_color(v) for v in vwr_vals]
    fig.add_trace(
        go.Scatter(
            x=days, y=vwr_vals,
            mode="lines+markers",
            line=dict(color="#3498DB", width=2.5),
            marker=dict(size=8, color=colors),
            name="VWR",
        ),
        row=1, col=1,
    )

    # Zone bands on VWR chart
    for lo, hi, color, label in _VWR_ZONES:
        fig.add_hrect(
            y0=lo, y1=hi,
            fillcolor=color, opacity=0.08,
            line_width=0,
            row=1, col=1,
        )

    # Strain curve
    fig.add_trace(
        go.Scatter(
            x=days, y=strain_vals,
            mode="lines+markers",
            line=dict(color="#E81828", width=2.5),
            marker=dict(size=8, color="#E81828"),
            name="Strain",
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Rest Days", row=1, col=1)
    fig.update_xaxes(title_text="Rest Days", row=1, col=2)
    fig.update_yaxes(title_text="VWR Score", range=[0, 100], row=1, col=1)
    fig.update_yaxes(title_text="Strain", row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=30, t=50, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="vwr_recovery_chart")


# ---------------------------------------------------------------------------
# Season Workload tab
# ---------------------------------------------------------------------------


def _render_season_workload(conn) -> None:
    """Season-long strain peaks/valleys overlaid with game appearances."""
    st.subheader("Season Workload Profile")

    pitcher_id = _get_pitcher_selector(conn, "season")
    if pitcher_id is None:
        return

    with st.spinner("Computing season workload..."):
        try:
            result = calculate_vwr(conn, pitcher_id)
        except Exception as exc:
            st.error(f"Error: {exc}")
            return

    appearances = result.get("game_appearances", [])
    if not appearances:
        st.info("No game appearance data available.")
        return

    dates = [a["game_date"] for a in appearances]
    peak_strains = [a["peak_strain"] for a in appearances]
    end_strains = [a["end_strain"] for a in appearances]
    pitches = [a["pitches"] for a in appearances]

    fig = go.Figure()

    # Strain range (end-of-game to peak as a filled area)
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1],
        y=peak_strains + end_strains[::-1],
        fill="toself",
        fillcolor="rgba(232, 24, 40, 0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Strain Range",
        showlegend=True,
    ))

    # End-of-game strain
    fig.add_trace(go.Scatter(
        x=dates,
        y=end_strains,
        mode="lines",
        line=dict(color="#E81828", width=2),
        name="End-of-Game Strain",
    ))

    # Peak strain
    fig.add_trace(go.Scatter(
        x=dates,
        y=peak_strains,
        mode="lines",
        line=dict(color="#FFD600", width=1.5, dash="dot"),
        name="Peak Strain",
    ))

    # Pitch count bar chart on secondary y-axis
    fig.add_trace(go.Bar(
        x=dates,
        y=pitches,
        marker_color="rgba(52, 152, 219, 0.4)",
        name="Pitch Count",
        yaxis="y2",
    ))

    fig.update_layout(
        xaxis=dict(title="Game Date"),
        yaxis=dict(title="Cumulative Strain", side="left"),
        yaxis2=dict(
            title="Pitch Count",
            overlaying="y",
            side="right",
            showgrid=False,
        ),
        template="plotly_dark",
        height=450,
        margin=dict(l=50, r=60, t=30, b=50),
        legend=dict(x=0.01, y=0.99),
        barmode="overlay",
    )
    st.plotly_chart(fig, use_container_width=True, key="vwr_season_chart")

    # Summary stats
    if len(appearances) >= 2:
        st.markdown("**Season Summary**")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Games", len(appearances))
        col2.metric("Total Pitches", sum(pitches))
        col3.metric("Max Strain", f"{max(peak_strains):.4f}")
        col4.metric("Avg End Strain", f"{np.mean(end_strains):.4f}")


# ---------------------------------------------------------------------------
# Leaderboard tab
# ---------------------------------------------------------------------------


def _render_leaderboard(conn) -> None:
    """Display pitchers with highest current VWR (most strained arms)."""
    st.subheader("VWR Strain Leaderboard")
    st.caption("Ranked by current VWR score (highest = most accumulated strain).")

    min_pitches = st.slider(
        "Minimum pitches to qualify",
        50, 2000, 500, step=50,
        key="vwr_min_pitches",
    )

    # Try cache first
    df = None
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "viscoelastic_workload", None)
            if cached is not None:
                df = cached
                age_info = cache_age_display(conn, "viscoelastic_workload", None)
                if age_info:
                    st.caption(age_info)
        except Exception:
            pass

    if df is None:
        with st.spinner("Computing... Run `python scripts/precompute.py` for instant loading."):
            try:
                df = batch_calculate(conn, min_pitches=min_pitches)
            except Exception as exc:
                st.error(f"Error computing leaderboard: {exc}")
                return

    if df.empty:
        st.info("No qualifying pitchers found with the current filters.")
        return

    df = df.reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"

    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "vwr_score": st.column_config.NumberColumn("VWR Score", format="%.1f"),
            "current_strain": st.column_config.NumberColumn("Strain", format="%.4f"),
            "E1": st.column_config.NumberColumn("E1", format="%.1f"),
            "E2": st.column_config.NumberColumn("E2", format="%.1f"),
            "tau": st.column_config.NumberColumn("Tau (hrs)", format="%.1f"),
            "n_pitches": st.column_config.NumberColumn("Pitches"),
            "name": st.column_config.TextColumn("Pitcher"),
        },
    )

    # VWR distribution chart
    if len(df) >= 5:
        st.markdown("**VWR Score Distribution**")
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=df["vwr_score"],
            nbinsx=20,
            marker_color="#E81828",
            opacity=0.75,
        ))

        median_vwr = df["vwr_score"].median()
        fig.add_vline(
            x=median_vwr,
            line=dict(color="white", width=2, dash="dash"),
            annotation_text=f"Median ({median_vwr:.1f})",
            annotation_position="top right",
        )

        # Zone backgrounds
        for lo, hi, color, label in _VWR_ZONES:
            fig.add_vrect(
                x0=lo, x1=hi,
                fillcolor=color, opacity=0.08,
                line_width=0,
            )

        fig.update_layout(
            xaxis=dict(title="VWR Score", range=[0, 100]),
            yaxis=dict(title="Count"),
            template="plotly_dark",
            height=350,
            margin=dict(l=50, r=30, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True, key="vwr_dist")
