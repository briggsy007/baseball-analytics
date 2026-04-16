"""
Pitch Decay Rate (PDR) dashboard view.

Visualises per-pitch-type fatigue cliff detection inspired by F1 tire
degradation models.  Shows cliff comparison across pitch types, survival
curves, game-specific cliff overlays, "which pitch dies first" callouts,
and a most-durable leaderboard.
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

_PDR_AVAILABLE = False
try:
    from src.analytics.pitch_decay import (
        batch_calculate,
        calculate_pdr,
        get_first_to_die,
        get_game_cliff_data,
    )
    _PDR_AVAILABLE = True
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

# Pitch type colour map (consistent with other views)
_PITCH_COLORS: dict[str, str] = {
    "FF": "#E81828", "SI": "#FF6B35", "SL": "#002D72", "CU": "#6A0DAD",
    "CH": "#2ECC71", "FC": "#F39C12", "FS": "#1ABC9C", "KC": "#9B59B6",
    "ST": "#E74C3C", "SV": "#3498DB",
}

_PITCH_LABELS: dict[str, str] = {
    "FF": "4-Seam", "SI": "Sinker", "SL": "Slider", "CU": "Curve",
    "CH": "Change", "FC": "Cutter", "FS": "Splitter", "KC": "Knuckle-Curve",
    "ST": "Sweeper", "SV": "Slurve", "CS": "Slow Curve", "FA": "Fastball",
}


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def _cached_pdr(pitcher_id: int, season: int) -> dict:
    """Cached PDR calculation."""
    conn = get_db_connection()
    return calculate_pdr(conn, pitcher_id, season=season)


@st.cache_data(ttl=3600)
def _cached_game_cliff_data(pitcher_id: int, game_pk: int) -> dict:
    """Cached game cliff data."""
    conn = get_db_connection()
    return get_game_cliff_data(conn, pitcher_id, game_pk)


def render() -> None:
    """Render the Pitch Decay Rate analysis page."""
    st.title("Pitch Decay Rate (PDR) -- Fatigue Cliff Detection")
    st.caption(
        "F1 tire-degradation model applied to per-pitch-type fatigue.  "
        "Finds the pitch count where each pitch type's quality drops off a cliff."
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**PDR finds the exact pitch count where each pitch type "falls off a cliff"** — inspired by F1 tire degradation modeling.

- **Cliff at pitch 80+** = durable pitch type (holds quality deep into games)
- **Cliff at pitch 50-60** = concerning — this pitch should be shelved in later innings
- **"Which pitch dies first"** is the key insight — if a starter's slider cliff is at pitch 55 but his fastball lasts to pitch 90, stop calling the slider after the 5th inning
- **Survival curves** show the probability each pitch type is still effective at any given pitch count
- **Impact:** Managing pitch mix based on per-pitch-type decay (not just total count) can extend a starter's effectiveness by 1-2 innings per game
""")

    if not _PDR_AVAILABLE:
        st.error(
            "The `pitch_decay` analytics module could not be imported.  "
            "Ensure `scipy` is installed:\n\n```\npip install scipy\n```"
        )
        return

    if not _PLOTLY_AVAILABLE:
        st.error("Plotly is required for charts.  Install with `pip install plotly`.")
        return

    conn = get_db_connection()
    if conn is None or not has_data(conn):
        st.warning(
            "No pitch data available.  Run the data backfill pipeline first "
            "(`python scripts/backfill.py`)."
        )
        return

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_pitcher, tab_leaderboard, tab_game = st.tabs([
        "Pitcher Analysis",
        "Durability Leaderboard",
        "Game Cliff Replay",
    ])

    with tab_pitcher:
        _render_pitcher_analysis(conn)

    with tab_leaderboard:
        _render_leaderboard(conn)

    with tab_game:
        _render_game_view(conn)


# ---------------------------------------------------------------------------
# Pitcher analysis tab
# ---------------------------------------------------------------------------


def _render_pitcher_analysis(conn) -> None:
    """Individual pitcher PDR analysis with cliff comparison and survival curves."""
    st.subheader("Pitcher Decay Profile")

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
            return

    pitcher_names = {
        p.get("full_name", f"ID {p['player_id']}"): p["player_id"]
        for p in pitchers
    }

    col_sel, col_season = st.columns([3, 1])
    with col_sel:
        selected_name = st.selectbox(
            "Select Pitcher",
            options=sorted(pitcher_names.keys()),
            key="pdr_pitcher_select",
        )
    with col_season:
        season = st.number_input(
            "Season",
            min_value=2015,
            max_value=2026,
            value=2025,
            key="pdr_season_input",
        )

    if not selected_name:
        return

    pitcher_id = pitcher_names[selected_name]

    with st.spinner("Detecting pitch-type fatigue cliffs..."):
        try:
            result = _cached_pdr(pitcher_id, season=int(season))
        except Exception as exc:
            st.error(f"Error computing PDR: {exc}")
            return

    per_pt = result.get("pitch_types", {})

    if not per_pt:
        st.info(
            "Not enough qualifying game appearances (need 30+ pitches of a type "
            "per game, across 5+ games) to detect cliffs for this pitcher/season."
        )
        return

    # ── "Which pitch dies first" callout ──────────────────────────────────
    first_to_die = result.get("first_to_die")
    first_cliff = result.get("first_to_die_cliff")
    if first_to_die:
        label = _PITCH_LABELS.get(first_to_die, first_to_die)
        # Find most durable
        most_durable_pt = max(per_pt.items(), key=lambda kv: kv[1]["median_cliff"])
        durable_label = _PITCH_LABELS.get(most_durable_pt[0], most_durable_pt[0])
        st.warning(
            f"**First to die:** {label} (cliff at pitch **{first_cliff:.0f}**)  |  "
            f"**Most durable:** {durable_label} "
            f"(cliff at pitch **{most_durable_pt[1]['median_cliff']:.0f}**)"
        )

    st.markdown("---")

    # ── Pitch-type cliff comparison bar chart ─────────────────────────────
    st.markdown("**Cliff Pitch Number by Pitch Type**")
    _render_cliff_bar_chart(per_pt)

    st.markdown("---")

    # ── Survival curves ───────────────────────────────────────────────────
    st.markdown("**Survival Curves: P(Pitch Type Still Effective) vs Pitch Count**")
    _render_survival_curves(per_pt)

    st.markdown("---")

    # ── Summary table ─────────────────────────────────────────────────────
    st.markdown("**Per-Pitch-Type Summary**")
    rows = []
    for pt in sorted(per_pt.keys()):
        d = per_pt[pt]
        rows.append({
            "Pitch": _PITCH_LABELS.get(pt, pt),
            "Code": pt,
            "Median Cliff": d["median_cliff"],
            "Mean Cliff": d["mean_cliff"],
            "Std": d["std_cliff"],
            "Games": d["games_analysed"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_cliff_bar_chart(per_pt: dict[str, dict]) -> None:
    """Grouped bar chart showing cliff pitch number per pitch type."""
    sorted_pts = sorted(per_pt.items(), key=lambda kv: kv[1]["median_cliff"])
    labels = [_PITCH_LABELS.get(pt, pt) for pt, _ in sorted_pts]
    cliffs = [d["median_cliff"] for _, d in sorted_pts]
    colors = [_PITCH_COLORS.get(pt, "#FFFFFF") for pt, _ in sorted_pts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=cliffs,
        marker_color=colors,
        text=[f"{c:.0f}" for c in cliffs],
        textposition="outside",
    ))
    fig.update_layout(
        xaxis=dict(title="Pitch Type"),
        yaxis=dict(title="Cliff Pitch Number (Median)"),
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=30, t=30, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="pdr_cliff_bar")


def _render_survival_curves(per_pt: dict[str, dict]) -> None:
    """Overlaid line charts per pitch type showing survival probability."""
    fig = go.Figure()

    for pt, pt_data in sorted(per_pt.items()):
        curve = pt_data.get("survival_curve", {})
        x = curve.get("pitch_numbers", [])
        y = curve.get("survival", [])
        if not x:
            continue

        color = _PITCH_COLORS.get(pt, "#FFFFFF")
        label = _PITCH_LABELS.get(pt, pt)
        cliff = pt_data["median_cliff"]

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color=color, width=2.5),
            name=f"{label} (cliff={cliff:.0f})",
        ))

    fig.add_hline(
        y=0.5,
        line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"),
        annotation_text="50%",
        annotation_position="bottom left",
    )

    fig.update_layout(
        xaxis=dict(title="Pitch Count (within type)"),
        yaxis=dict(title="P(Still Effective)", range=[-0.05, 1.05]),
        template="plotly_dark",
        height=450,
        margin=dict(l=50, r=30, t=30, b=50),
        legend=dict(x=0.6, y=0.95),
    )
    st.plotly_chart(fig, use_container_width=True, key="pdr_survival")


# ---------------------------------------------------------------------------
# Leaderboard tab
# ---------------------------------------------------------------------------


def _render_leaderboard(conn) -> None:
    """Display most durable pitchers per pitch type."""
    st.subheader("PDR Durability Leaderboard")
    st.caption(
        "Pitchers ranked by latest first-cliff point (higher = more durable).  "
        "The 'first to die' column shows which pitch type degrades earliest."
    )

    min_pitches = st.slider(
        "Minimum pitches to qualify",
        50, 2000, 500, step=50,
        key="pdr_min_pitches",
    )

    # Try cache first
    df = None
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "pitch_decay", None)
            if cached is not None:
                df = cached
                age_info = cache_age_display(conn, "pitch_decay", None)
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
            "first_to_die_cliff": st.column_config.NumberColumn(
                "First Cliff (pitches)", format="%.0f",
            ),
            "first_to_die": st.column_config.TextColumn("First to Die"),
            "name": st.column_config.TextColumn("Pitcher"),
        },
    )

    # Distribution chart
    if len(df) >= 5 and df["first_to_die_cliff"].notna().any():
        st.markdown("**First-Cliff Distribution**")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["first_to_die_cliff"].dropna(),
            nbinsx=25,
            marker_color="#E81828",
            opacity=0.75,
        ))
        median_cliff = df["first_to_die_cliff"].median()
        fig.add_vline(
            x=median_cliff,
            line=dict(color="white", width=2, dash="dash"),
            annotation_text=f"Median ({median_cliff:.0f})",
            annotation_position="top right",
        )
        fig.update_layout(
            xaxis=dict(title="First Cliff (pitch count)"),
            yaxis=dict(title="Count"),
            template="plotly_dark",
            height=350,
            margin=dict(l=50, r=30, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True, key="pdr_dist")


# ---------------------------------------------------------------------------
# Game replay tab
# ---------------------------------------------------------------------------


def _render_game_view(conn) -> None:
    """Select a specific game and see quality signal with cliff marked."""
    st.subheader("Game-Specific Cliff Analysis")

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
            return

    pitcher_names = {
        p.get("full_name", f"ID {p['player_id']}"): p["player_id"]
        for p in pitchers
    }
    selected_name = st.selectbox(
        "Select Pitcher",
        options=sorted(pitcher_names.keys()),
        key="pdr_game_pitcher_select",
    )
    if not selected_name:
        return

    pitcher_id = pitcher_names[selected_name]

    # Get list of games for this pitcher
    try:
        games_df = conn.execute("""
            SELECT DISTINCT game_pk, game_date, COUNT(*) AS pitches
            FROM pitches
            WHERE pitcher_id = $1
              AND release_speed IS NOT NULL
              AND pitch_type IS NOT NULL
            GROUP BY game_pk, game_date
            HAVING COUNT(*) >= 30
            ORDER BY game_date DESC
            LIMIT 50
        """, [pitcher_id]).fetchdf()
    except Exception:
        games_df = pd.DataFrame()

    if games_df.empty:
        st.info("No qualifying games found for this pitcher.")
        return

    game_options = {
        f"{row['game_date']} (pk={row['game_pk']}, {row['pitches']} pitches)": int(row["game_pk"])
        for _, row in games_df.iterrows()
    }
    selected_game = st.selectbox(
        "Select Game",
        options=list(game_options.keys()),
        key="pdr_game_select",
    )
    if not selected_game:
        return

    game_pk = game_options[selected_game]

    with st.spinner("Analysing game cliffs..."):
        try:
            data = _cached_game_cliff_data(pitcher_id, game_pk)
        except Exception as exc:
            st.error(f"Error: {exc}")
            return

    pt_data = data.get("pitch_types", {})
    if not pt_data:
        st.info("Not enough data per pitch type in this game (need 30+ pitches per type).")
        return

    # One chart per pitch type
    for pt in sorted(pt_data.keys()):
        info = pt_data[pt]
        label = _PITCH_LABELS.get(pt, pt)
        color = _PITCH_COLORS.get(pt, "#FFFFFF")
        cliff = info.get("cliff", {})

        st.markdown(f"**{label} ({pt})**")

        fig = go.Figure()

        # Raw quality (thin, transparent)
        fig.add_trace(go.Scatter(
            x=info["pitch_numbers"],
            y=info["raw_quality"],
            mode="lines",
            line=dict(color=f"rgba(255,255,255,0.2)", width=1),
            name="Raw Quality",
        ))

        # Smoothed quality (bold)
        fig.add_trace(go.Scatter(
            x=info["pitch_numbers"],
            y=info["smooth_quality"],
            mode="lines",
            line=dict(color=color, width=2.5),
            name="Smoothed Quality",
        ))

        # Mark cliff
        tau = cliff.get("tau")
        if tau is not None:
            fig.add_vline(
                x=tau,
                line=dict(color="#FFD600", width=2, dash="dash"),
                annotation_text=f"Cliff = {tau:.0f}",
                annotation_position="top left",
                annotation_font_color="#FFD600",
            )

        fig.update_layout(
            xaxis=dict(title=f"Pitch Number (within {pt})"),
            yaxis=dict(title="Quality Signal"),
            template="plotly_dark",
            height=350,
            margin=dict(l=50, r=30, t=30, b=50),
            legend=dict(x=0.6, y=0.95),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"pdr_game_{pt}")

        # Cliff metrics
        if tau is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cliff", f"{tau:.0f}")
            c2.metric("Pre-slope", f"{cliff.get('pre_slope', 0):.5f}")
            c3.metric("Post-slope", f"{cliff.get('post_slope', 0):.5f}")
            c4.metric("MSE Reduction", f"{cliff.get('mse_reduction', 0):.3f}")
