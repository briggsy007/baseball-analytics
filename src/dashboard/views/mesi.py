"""
Motor Engram Stability Index (MESI) dashboard view.

Visualises pitch execution reliability through radar charts, context
stability breakdowns, learning curves, leaderboards, and pitcher
comparison mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.db_helper import get_db_connection, has_data, get_all_pitchers

# ---------------------------------------------------------------------------
# Graceful imports for analytics modules
# ---------------------------------------------------------------------------

_MESI_AVAILABLE = False
try:
    from src.analytics.mesi import (
        batch_calculate,
        calculate_mesi,
        compute_context_stability,
        fit_learning_curve,
        get_arsenal_stability,
        EXECUTION_COLS,
    )
    _MESI_AVAILABLE = True
except ImportError:
    pass

_CACHE_AVAILABLE = False
try:
    from src.dashboard.cache_reader import get_cached_leaderboard, get_cached_entity, cache_age_display
    _CACHE_AVAILABLE = True
except Exception:
    pass

_PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    pass

# Pitch type metadata
_PITCH_COLORS: dict[str, str] = {
    "FF": "#E81828", "SI": "#FF6B35", "SL": "#002D72", "CU": "#6A0DAD",
    "CH": "#2ECC71", "FC": "#F39C12", "FS": "#1ABC9C", "KC": "#9B59B6",
    "ST": "#E74C3C", "SV": "#3498DB",
}

_PITCH_LABELS: dict[str, str] = {
    "FF": "4-Seam", "SI": "Sinker", "SL": "Slider", "CU": "Curve",
    "CH": "Change", "FC": "Cutter", "FS": "Splitter", "KC": "Knuckle-Curve",
    "ST": "Sweeper", "SV": "Slurve", "CS": "Slow Curve", "SC": "Screwball",
    "FA": "Fastball",
}

_CONTEXT_LABELS: dict[str, str] = {
    "low_leverage": "Low Leverage (Inn 1-3)",
    "high_leverage": "High Leverage (Inn 7+)",
    "first_time_through": "1st Time Through",
    "third_time_through": "3rd Time Through",
    "ahead_in_count": "Ahead in Count",
    "behind_in_count": "Behind in Count",
    "fresh": "Fresh (P# 1-50)",
    "fatigued": "Fatigued (P# 80+)",
}


# ---------------------------------------------------------------------------
# Cached computation wrappers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_mesi_leaderboard(_conn, min_pitches: int) -> pd.DataFrame:
    """Cached wrapper for MESI leaderboard computation."""
    return batch_calculate(_conn, min_pitches=min_pitches)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_arsenal_stability(_conn, pitcher_id: int) -> dict:
    """Cached wrapper for get_arsenal_stability computation."""
    return get_arsenal_stability(_conn, pitcher_id)


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_context_stability(_conn, pitcher_id: int) -> dict:
    """Cached wrapper for compute_context_stability computation."""
    return compute_context_stability(_conn, pitcher_id)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the MESI Analysis page."""
    st.title("Motor Engram Stability Index (MESI)")
    st.caption(
        "Quantify pitch execution reliability using motor learning theory. "
        "Higher MESI = more consistent, deeply-ingrained pitch mechanics."
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**MESI measures how "grooved" each pitch is** — not how good it is (that's Stuff+), but how reliably a pitcher can execute it under any conditions.

- **MESI 80+** = autonomous pitch (deeply ingrained, consistent under pressure and fatigue)
- **MESI 50–80** = associative stage (developing — reliable in low-stress but breaks down under pressure)
- **MESI < 50** = cognitive stage (still learning — high variability, unreliable)
- **Context Stability** shows if a pitch holds up under pressure, fatigue, and times-through-the-order — a pitch with high SNR but low context stability is a "flat track bully"
- **Learning Curves** reveal if a pitch is still improving (steep curve) or has plateaued — critical for prospect evaluation
- **Impact:** A pitcher shouldn't throw a MESI-40 pitch in a 3-2 count with the bases loaded, regardless of what the scouting report says
""")

    if not _MESI_AVAILABLE:
        st.error(
            "The `mesi` analytics module could not be imported. "
            "Ensure `scipy` is installed:\n\n"
            "```\npip install scipy\n```"
        )
        return

    if not _PLOTLY_AVAILABLE:
        st.error(
            "Plotly is required for MESI visualisations:\n\n"
            "```\npip install plotly\n```"
        )
        return

    conn = get_db_connection()

    if conn is None or not has_data(conn):
        st.warning(
            "No pitch data available. Run the data backfill pipeline first "
            "(`python scripts/backfill.py`)."
        )
        return

    # ── Main content tabs ─────────────────────────────────────────────────
    tab_leaderboard, tab_pitcher, tab_compare = st.tabs(
        ["Leaderboard", "Pitcher Deep Dive", "Comparison Mode"]
    )

    with tab_leaderboard:
        _render_leaderboard(conn)

    with tab_pitcher:
        _render_pitcher_deep_dive(conn)

    with tab_compare:
        _render_comparison(conn)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def _render_leaderboard(conn) -> None:
    """Display the MESI leaderboard for all qualifying pitchers."""
    st.subheader("Most Reliable Arsenals")

    min_pitches = st.slider(
        "Minimum pitches to qualify", 50, 1000, 200, step=50,
        key="mesi_min_pitches",
    )

    # Try cache first
    leaderboard = None
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "mesi", None)
            if cached is not None:
                leaderboard = cached
                age_info = cache_age_display(conn, "mesi", None)
                if age_info:
                    st.caption(age_info)
        except Exception:
            pass

    if leaderboard is None:
        try:
            with st.spinner("Computing... Run `python scripts/precompute.py` for instant loading."):
                leaderboard = _cached_mesi_leaderboard(conn, min_pitches=min_pitches)
        except Exception as exc:
            st.error(f"Error computing leaderboard: {exc}")
            return

    if leaderboard.empty:
        st.info("No qualifying pitchers found with the current filters.")
        return

    leaderboard = leaderboard.reset_index(drop=True)
    leaderboard.index = leaderboard.index + 1
    leaderboard.index.name = "Rank"

    st.dataframe(
        leaderboard,
        use_container_width=True,
        column_config={
            "overall_mesi": st.column_config.NumberColumn("MESI", format="%.1f"),
            "avg_snr": st.column_config.NumberColumn("Avg SNR", format="%.3f"),
            "avg_cs": st.column_config.NumberColumn("Avg CS", format="%.3f"),
            "n_pitch_types": st.column_config.NumberColumn("Pitch Types"),
            "name": st.column_config.TextColumn("Pitcher"),
        },
    )

    # Distribution chart
    if len(leaderboard) >= 5:
        st.markdown("**MESI Distribution**")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=leaderboard["overall_mesi"],
            nbinsx=20,
            marker_color="#3498DB",
            opacity=0.75,
        ))
        fig.add_vline(
            x=50, line=dict(color="white", width=2, dash="dash"),
            annotation_text="Population Median",
            annotation_position="top right",
        )
        fig.update_layout(
            xaxis=dict(title="MESI Score (0-100)"),
            yaxis=dict(title="Count"),
            template="plotly_dark",
            height=350,
            margin=dict(l=50, r=30, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True, key="mesi_dist")


# ---------------------------------------------------------------------------
# Pitcher Deep Dive
# ---------------------------------------------------------------------------


def _render_pitcher_deep_dive(conn) -> None:
    """Full MESI profile for a single pitcher."""
    st.subheader("Pitcher MESI Profile")

    pitcher_id, pitcher_name = _select_pitcher(conn, key_suffix="deep_dive")
    if pitcher_id is None:
        return

    try:
        # Try entity cache first
        profile = None
        if _CACHE_AVAILABLE:
            cached = get_cached_entity(conn, "mesi", pitcher_id, season=2025)
            if cached is not None:
                profile = cached
                st.caption("Using pre-computed results")
        if profile is None:
            with st.spinner("Computing MESI profile... Run `python scripts/precompute.py` for instant loading."):
                profile = _cached_arsenal_stability(conn, pitcher_id)
    except Exception as exc:
        st.error(f"Error computing MESI profile: {exc}")
        return

    if profile is None or profile.get("overall_mesi") is None:
        st.info("Insufficient data for this pitcher (need 100+ pitches per pitch type).")
        return

    # ── Header metrics ───────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall MESI", f"{profile['overall_mesi']:.3f}")
    snr_val = profile.get('overall_snr') or 0
    cs_val = profile.get('overall_cs') or 0
    col2.metric("Avg SNR", f"{snr_val:.3f}")
    col3.metric("Avg CS", f"{cs_val:.3f}")

    st.markdown("---")

    # ── Scouting insights ────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        best = profile.get("best_pitch_under_pressure")
        if best:
            st.success(
                f"**Best Pitch Under Pressure:** "
                f"{_PITCH_LABELS.get(best, best)} ({best})"
            )
        else:
            st.info("Not enough high-leverage data for pressure insight.")

    with c2:
        worst = profile.get("pitch_that_breaks_down")
        if worst:
            st.warning(
                f"**Pitch That Breaks Down:** "
                f"{_PITCH_LABELS.get(worst, worst)} ({worst})"
            )
        else:
            st.info("No pitch breakdown detected.")

    st.markdown("---")

    # ── Arsenal MESI radar chart ─────────────────────────────────────────
    per_pt = profile.get("per_pitch_type", {})
    if per_pt:
        _render_radar_chart(per_pt, pitcher_name)

    st.markdown("---")

    # ── Context stability breakdown ──────────────────────────────────────
    if per_pt:
        _render_context_breakdown(per_pt)

    st.markdown("---")

    # ── Learning curves ──────────────────────────────────────────────────
    learning_curves = profile.get("learning_curves", {})
    if learning_curves:
        _render_learning_curves(learning_curves)


def _render_radar_chart(per_pt: dict, pitcher_name: str) -> None:
    """Radar/polar chart showing MESI components per pitch type."""
    st.markdown("**Arsenal MESI Radar**")

    pitch_types = sorted(per_pt.keys())
    snr_values = [per_pt[pt]["snr"] for pt in pitch_types]
    cs_values = [
        per_pt[pt]["cs_score"] if per_pt[pt]["cs_score"] is not None else 0
        for pt in pitch_types
    ]
    labels = [_PITCH_LABELS.get(pt, pt) for pt in pitch_types]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=snr_values,
        theta=labels,
        fill="toself",
        name="SNR",
        line_color="#3498DB",
        fillcolor="rgba(52, 152, 219, 0.2)",
    ))

    fig.add_trace(go.Scatterpolar(
        r=cs_values,
        theta=labels,
        fill="toself",
        name="Context Stability",
        line_color="#2ECC71",
        fillcolor="rgba(46, 204, 113, 0.2)",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True),
            bgcolor="rgba(0,0,0,0)",
        ),
        template="plotly_dark",
        height=450,
        title=f"{pitcher_name} - Arsenal Stability",
        margin=dict(l=60, r=60, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, key="mesi_radar")

    # Summary table
    rows = []
    for pt in pitch_types:
        info = per_pt[pt]
        rows.append({
            "Pitch": _PITCH_LABELS.get(pt, pt),
            "Code": pt,
            "SNR": round(info["snr"], 3),
            "CS": round(info["cs_score"], 3) if info["cs_score"] is not None else None,
            "Raw MESI": round(info["raw_mesi"], 3),
            "Pitches": info["n_pitches"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_context_breakdown(per_pt: dict) -> None:
    """Bar chart showing SNR across contexts for each pitch type."""
    st.markdown("**Context Stability Breakdown**")

    selected_pt = st.selectbox(
        "Select pitch type",
        options=sorted(per_pt.keys()),
        format_func=lambda x: f"{_PITCH_LABELS.get(x, x)} ({x})",
        key="mesi_context_pt",
    )

    if not selected_pt:
        return

    ctx_snr = per_pt[selected_pt].get("context_snr", {})
    if not ctx_snr:
        st.info("No context data available for this pitch type.")
        return

    labels = []
    values = []
    colors = []

    pair_colors = [
        ("#3498DB", "#1A5276"),  # leverage
        ("#2ECC71", "#1D8348"),  # TTO
        ("#F39C12", "#B7950B"),  # count
        ("#E74C3C", "#922B21"),  # fatigue
    ]

    context_pairs = [
        ("low_leverage", "high_leverage"),
        ("first_time_through", "third_time_through"),
        ("ahead_in_count", "behind_in_count"),
        ("fresh", "fatigued"),
    ]

    for i, (ctx_a, ctx_b) in enumerate(context_pairs):
        for ctx in [ctx_a, ctx_b]:
            val = ctx_snr.get(ctx)
            if val is not None:
                labels.append(_CONTEXT_LABELS.get(ctx, ctx))
                values.append(val)
                idx = 0 if ctx == ctx_a else 1
                colors.append(pair_colors[i][idx])

    if not labels:
        st.info("No context SNR data available.")
        return

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="auto",
    ))

    fig.update_layout(
        yaxis=dict(title="SNR"),
        xaxis=dict(tickangle=-45),
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=30, t=30, b=100),
        title=f"Context SNR - {_PITCH_LABELS.get(selected_pt, selected_pt)}",
    )
    st.plotly_chart(fig, use_container_width=True, key="mesi_context_bars")


def _render_learning_curves(learning_curves: dict) -> None:
    """MESI trajectory over career with fitted power law overlay."""
    st.markdown("**Learning Curve (MESI Trajectory)**")

    available_pts = [
        pt for pt, lc in learning_curves.items()
        if lc.get("trajectory") and len(lc["trajectory"]) >= 3
    ]

    if not available_pts:
        st.info("Not enough appearance data to plot learning curves.")
        return

    selected_pt = st.selectbox(
        "Select pitch type for learning curve",
        options=sorted(available_pts),
        format_func=lambda x: f"{_PITCH_LABELS.get(x, x)} ({x})",
        key="mesi_lc_pt",
    )

    if not selected_pt:
        return

    lc = learning_curves[selected_pt]
    trajectory = lc["trajectory"]

    t_vals = [p[0] for p in trajectory]
    mesi_vals = [p[1] for p in trajectory]

    fig = go.Figure()

    # Actual trajectory
    fig.add_trace(go.Scatter(
        x=t_vals,
        y=mesi_vals,
        mode="lines+markers",
        name="Observed MESI",
        line=dict(color="#3498DB", width=2),
        marker=dict(size=4),
    ))

    # Fitted curve overlay
    if lc.get("ceiling") and lc.get("learning_rate") and lc["learning_rate"] > 0:
        from src.analytics.mesi import _learning_curve_func
        t_fit = np.linspace(1, max(t_vals) * 1.2, 100)
        mesi_fit = _learning_curve_func(t_fit, lc["ceiling"], lc["learning_rate"])
        fig.add_trace(go.Scatter(
            x=t_fit.tolist(),
            y=mesi_fit.tolist(),
            mode="lines",
            name="Fitted Power Law",
            line=dict(color="#E74C3C", width=2, dash="dash"),
        ))

        # Ceiling line
        fig.add_hline(
            y=lc["ceiling"],
            line=dict(color="#2ECC71", width=1, dash="dot"),
            annotation_text=f"Ceiling: {lc['ceiling']:.2f}",
            annotation_position="top right",
        )

    fig.update_layout(
        xaxis=dict(title="Appearances"),
        yaxis=dict(title="MESI (SNR)"),
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=30, t=30, b=50),
    )
    st.plotly_chart(fig, use_container_width=True, key="mesi_lc_chart")

    # Learning curve stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Learning Rate", f"{lc['learning_rate']:.4f}" if lc["learning_rate"] else "N/A")
    col2.metric("Ceiling", f"{lc['ceiling']:.3f}" if lc["ceiling"] else "N/A")
    col3.metric(
        "Stage",
        lc["current_stage"].replace("_", " ").title(),
    )

    if lc.get("r_squared") is not None:
        st.caption(f"Fit quality (R-squared): {lc['r_squared']:.4f}")


# ---------------------------------------------------------------------------
# Comparison Mode
# ---------------------------------------------------------------------------


def _render_comparison(conn) -> None:
    """Side-by-side MESI comparison of two pitchers."""
    st.subheader("Pitcher Comparison")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Pitcher A**")
        pid_a, name_a = _select_pitcher(conn, key_suffix="comp_a")

    with col_b:
        st.markdown("**Pitcher B**")
        pid_b, name_b = _select_pitcher(conn, key_suffix="comp_b")

    if pid_a is None or pid_b is None:
        st.info("Select two pitchers to compare.")
        return

    if pid_a == pid_b:
        st.warning("Select two different pitchers.")
        return

    # Compute each profile independently so one failure doesn't kill both
    profile_a = None
    profile_b = None

    with st.spinner("Computing MESI profiles..."):
        try:
            profile_a = _cached_arsenal_stability(conn, pid_a)
        except Exception as exc:
            st.warning(f"Could not compute profile for {name_a}: {exc}")

        try:
            profile_b = _cached_arsenal_stability(conn, pid_b)
        except Exception as exc:
            st.warning(f"Could not compute profile for {name_b}: {exc}")

    has_a = profile_a is not None and profile_a.get("overall_mesi") is not None
    has_b = profile_b is not None and profile_b.get("overall_mesi") is not None

    if not has_a and not has_b:
        st.info("Insufficient data for both pitchers. Try selecting pitchers with 100+ pitches of at least one pitch type.")
        return

    # ── Summary comparison ───────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        if has_a:
            _render_comparison_card(name_a, profile_a)
        else:
            st.info(f"{name_a}: insufficient data")
    with col2:
        if has_b:
            _render_comparison_card(name_b, profile_b)
        else:
            st.info(f"{name_b}: insufficient data")

    st.markdown("---")

    # ── Overlaid radar chart ─────────────────────────────────────────────
    if has_a and has_b:
        _render_comparison_radar(name_a, profile_a, name_b, profile_b)
    elif has_a:
        st.info(f"Radar chart requires data for both pitchers. {name_b} has insufficient data.")
    elif has_b:
        st.info(f"Radar chart requires data for both pitchers. {name_a} has insufficient data.")


def _render_comparison_card(name: str, profile: dict) -> None:
    """Render a summary card for one pitcher in comparison mode."""
    if profile["overall_mesi"] is None:
        st.info(f"{name}: insufficient data")
        return

    st.metric("Overall MESI", f"{profile['overall_mesi']:.3f}")

    best = profile.get("best_pitch_under_pressure")
    worst = profile.get("pitch_that_breaks_down")

    if best:
        st.markdown(f"**Best under pressure:** {_PITCH_LABELS.get(best, best)}")
    if worst:
        st.markdown(f"**Breaks down:** {_PITCH_LABELS.get(worst, worst)}")

    # Pitch type table
    per_pt = profile.get("per_pitch_type", {})
    if per_pt:
        rows = []
        for pt in sorted(per_pt.keys()):
            info = per_pt[pt]
            rows.append({
                "Pitch": _PITCH_LABELS.get(pt, pt),
                "MESI": round(info["raw_mesi"], 3),
                "SNR": round(info["snr"], 3),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_comparison_radar(
    name_a: str, profile_a: dict,
    name_b: str, profile_b: dict,
) -> None:
    """Overlay radar charts for two pitchers."""
    st.markdown("**Arsenal Comparison**")

    # Collect all pitch types from both pitchers
    pts_a = set(profile_a.get("per_pitch_type", {}).keys())
    pts_b = set(profile_b.get("per_pitch_type", {}).keys())
    all_pts = sorted(pts_a | pts_b)

    if not all_pts:
        st.info("No pitch types to compare.")
        return

    labels = [_PITCH_LABELS.get(pt, pt) for pt in all_pts]

    def _get_snr(profile, pt):
        info = profile.get("per_pitch_type", {}).get(pt)
        return info["snr"] if info else 0

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[_get_snr(profile_a, pt) for pt in all_pts],
        theta=labels,
        fill="toself",
        name=name_a,
        line_color="#3498DB",
        fillcolor="rgba(52, 152, 219, 0.15)",
    ))

    fig.add_trace(go.Scatterpolar(
        r=[_get_snr(profile_b, pt) for pt in all_pts],
        theta=labels,
        fill="toself",
        name=name_b,
        line_color="#E74C3C",
        fillcolor="rgba(231, 76, 60, 0.15)",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True),
            bgcolor="rgba(0,0,0,0)",
        ),
        template="plotly_dark",
        height=450,
        margin=dict(l=60, r=60, t=40, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, key="mesi_compare_radar")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=600)
def _cached_pitcher_list(_conn) -> list[dict]:
    """Cache the pitcher list so it doesn't re-query on every rerun."""
    return get_all_pitchers(_conn)


def _select_pitcher(conn, key_suffix: str = "") -> tuple[int | None, str]:
    """Render a pitcher selector and return (pitcher_id, name)."""
    pitchers = _cached_pitcher_list(conn)
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
            return None, ""

    if not pitchers:
        st.info("No pitchers found.")
        return None, ""

    pitcher_names = {
        p.get("full_name", f"ID {p['player_id']}"): p["player_id"]
        for p in pitchers
    }
    selected_name = st.selectbox(
        "Select Pitcher",
        options=sorted(pitcher_names.keys()),
        key=f"mesi_pitcher_select_{key_suffix}",
    )

    if not selected_name:
        return None, ""

    return pitcher_names[selected_name], selected_name
