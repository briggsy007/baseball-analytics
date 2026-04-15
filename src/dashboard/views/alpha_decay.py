"""
Pitch Sequence Alpha Decay dashboard view.

Visualises how quickly a pitcher's sequencing patterns lose their
deceptive effectiveness.  Shows intra-game decay curves, half-life
comparisons across time scales, a "Hardest to Decode" leaderboard,
and a per-transition breakdown table.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.db_helper import get_db_connection, has_data, get_all_pitchers

# ---------------------------------------------------------------------------
# Graceful imports for the alpha decay analytics module
# ---------------------------------------------------------------------------

_AD_AVAILABLE = False
try:
    from src.analytics.alpha_decay import (
        calculate_alpha_decay,
        batch_calculate,
        get_fastest_decaying_sequences,
        compute_sequence_alpha,
        fit_intra_game_decay,
        _exp_decay,
    )
    _AD_AVAILABLE = True
except ImportError:
    pass

_CACHE_AVAILABLE = False
try:
    from src.dashboard.cache_reader import get_cached_leaderboard, cache_age_display
    _CACHE_AVAILABLE = True
except Exception:
    pass

try:
    import plotly.graph_objects as go
    import plotly.express as px
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


# Pitch type display helpers
_PITCH_COLORS: dict[str, str] = {
    "FF": "#E81828", "SI": "#FF6B35", "SL": "#002D72", "CU": "#6A0DAD",
    "CH": "#2ECC71", "FC": "#F39C12", "FS": "#1ABC9C", "KC": "#9B59B6",
    "ST": "#E74C3C", "SV": "#3498DB",
}

_PITCH_LABELS: dict[str, str] = {
    "FF": "4-Seam", "SI": "Sinker", "SL": "Slider", "CU": "Curve",
    "CH": "Change", "FC": "Cutter", "FS": "Splitter", "KC": "Knuckle-Curve",
    "ST": "Sweeper", "SV": "Slurve", "FA": "Fastball", "CS": "Slow Curve",
}


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Pitch Sequence Alpha Decay analysis page."""
    st.title("Pitch Sequence Alpha Decay")
    st.caption(
        "How quickly do batters decode a pitcher's sequencing patterns? "
        "Alpha measures whiff-rate uplift from pitch-pair transitions; "
        "decay tracks how that edge erodes with repetition."
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**Alpha Decay measures how quickly hitters "figure out" a pitcher's patterns** — borrowed from quant finance where trading signals lose edge over time.

- **Long half-life (50+ repetitions)** = pitcher stays deceptive even when batters see the same sequences repeatedly — hard to decode
- **Short half-life (< 20 repetitions)** = patterns get decoded quickly — effectiveness drops sharply with exposure
- **Intra-game decay** = within a single start (why starters get worse the 3rd time through the order)
- **Series decay** = across games vs. the same team (why familiarity breeds success for hitters)
- **Impact:** A starter with short half-life should be pulled earlier or have their pitch mix reshuffled. A reliever with short half-life is fine — they only face each hitter once
""")

    if not _AD_AVAILABLE:
        st.error(
            "The `alpha_decay` analytics module could not be imported. "
            "Ensure the module is installed and all dependencies are available."
        )
        return

    if not _PLOTLY_AVAILABLE:
        st.error("Plotly is required for charts. Install with `pip install plotly`.")
        return

    conn = get_db_connection()

    if conn is None or not has_data(conn):
        st.warning(
            "No pitch data available. Run the data backfill pipeline first "
            "(`python scripts/backfill.py`)."
        )
        return

    # -- Sidebar controls ------------------------------------------------------
    with st.sidebar:
        st.markdown("### Alpha Decay Options")

        try:
            seasons = conn.execute(
                "SELECT DISTINCT EXTRACT(YEAR FROM game_date)::INT AS season "
                "FROM pitches ORDER BY season DESC"
            ).fetchdf()["season"].tolist()
        except Exception:
            seasons = []

        season_options = ["All Seasons"] + [str(s) for s in seasons]
        selected_season = st.selectbox(
            "Season", season_options, key="ad_season"
        )
        season = int(selected_season) if selected_season != "All Seasons" else None

        min_pitches = st.slider(
            "Min pitches to qualify",
            50, 2000, 300, step=50,
            key="ad_min_pitches",
        )

    # -- Tabs ------------------------------------------------------------------
    tab_pitcher, tab_leaderboard = st.tabs([
        "Pitcher Deep Dive",
        "Hardest to Decode Leaderboard",
    ])

    with tab_pitcher:
        _render_pitcher_analysis(conn, season)

    with tab_leaderboard:
        _render_leaderboard(conn, season, min_pitches)


# ---------------------------------------------------------------------------
# Pitcher deep-dive
# ---------------------------------------------------------------------------


def _render_pitcher_analysis(conn, season) -> None:
    """Individual pitcher alpha decay analysis."""
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
    selected_name = st.selectbox(
        "Select Pitcher",
        options=sorted(pitcher_names.keys()),
        key="ad_pitcher_select",
    )
    if not selected_name:
        return

    pitcher_id = pitcher_names[selected_name]

    with st.spinner("Computing alpha decay profile..."):
        try:
            result = calculate_alpha_decay(conn, pitcher_id, season=season)
        except Exception as exc:
            st.error(f"Error computing alpha decay: {exc}")
            return

    if not result["transitions"]:
        st.info("Not enough pitch-pair data for this pitcher/season combination.")
        return

    summary = result["summary"]

    # -- Key metrics -----------------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    hl_intra = summary.get("avg_intra_game_half_life")
    hl_series = summary.get("avg_series_half_life")
    adapt = summary.get("adaptability_score")

    c1.metric(
        "Avg In-Game Half-Life",
        f"{hl_intra:.1f} reps" if hl_intra is not None else "N/A",
    )
    c2.metric(
        "Avg Series Half-Life",
        f"{hl_series:.1f} games" if hl_series is not None else "N/A",
    )
    c3.metric(
        "Adaptability Score",
        f"{adapt:.0f} / 100" if adapt is not None else "N/A",
    )
    c4.metric("Transitions Analyzed", len(result["transitions"]))

    # Callout for most durable / fragile
    durable = summary.get("most_durable_transition")
    fragile = summary.get("most_fragile_transition")
    if durable and fragile:
        st.info(
            f"**Most deceptive:** {durable} (longest half-life)  |  "
            f"**Easiest to read:** {fragile} (shortest half-life)"
        )

    st.markdown("---")

    # -- Intra-game decay chart ------------------------------------------------
    _render_intra_game_decay_chart(result)

    st.markdown("---")

    # -- Half-life comparison bar chart ----------------------------------------
    _render_half_life_comparison(result)

    st.markdown("---")

    # -- Sequence-specific breakdown table -------------------------------------
    _render_transition_table(result)

    st.markdown("---")

    # -- Regime timeline -------------------------------------------------------
    _render_regime_timeline(result)


def _render_intra_game_decay_chart(result: dict) -> None:
    """Plot alpha vs repetition count with fitted curve for each transition."""
    st.markdown("**Intra-Game Alpha Decay**")
    st.caption(
        "How quickly each pitch-pair sequence loses its whiff-rate edge "
        "as it gets repeated within a game."
    )

    transitions = result.get("transitions", {})

    # Collect transitions that have alpha_by_rep data
    plottable = {}
    for trans, data in transitions.items():
        intra = data.get("intra_game", {})
        by_rep = intra.get("alpha_by_rep", [])
        if len(by_rep) >= 3:
            plottable[trans] = data

    if not plottable:
        st.info("Not enough repetition data to plot decay curves.")
        return

    # Let user select a transition
    trans_options = sorted(plottable.keys())
    selected_trans = st.selectbox(
        "Select transition",
        trans_options,
        key="ad_decay_trans_select",
    )

    data = plottable[selected_trans]
    intra = data["intra_game"]
    by_rep = intra["alpha_by_rep"]

    reps = [p["rep"] for p in by_rep]
    alphas = [p["alpha"] for p in by_rep]
    n_swings = [p["n_swings"] for p in by_rep]

    fig = go.Figure()

    # Observed alpha points
    fig.add_trace(go.Scatter(
        x=reps,
        y=alphas,
        mode="markers",
        marker=dict(
            size=[max(5, min(s, 30)) for s in n_swings],
            color="#3498DB",
            opacity=0.8,
        ),
        name="Observed Alpha",
        hovertemplate="Rep %{x}<br>Alpha: %{y:.4f}<br>Swings: %{customdata}<extra></extra>",
        customdata=n_swings,
    ))

    # Fitted curve
    fit = intra.get("fit")
    if fit:
        x_fit = np.linspace(min(reps), max(reps), 100)
        y_fit = _exp_decay(x_fit, fit["alpha_0"], fit["lambda"])
        fig.add_trace(go.Scatter(
            x=x_fit.tolist(),
            y=y_fit.tolist(),
            mode="lines",
            line=dict(color="#E81828", width=2.5, dash="dash"),
            name=f"Fitted (t1/2={fit['half_life']:.1f}, R2={fit['r_squared']:.3f})",
        ))

        # Mark half-life
        if fit["half_life"] <= max(reps) * 1.5:
            hl_y = fit["alpha_0"] * 0.5
            fig.add_trace(go.Scatter(
                x=[fit["half_life"]],
                y=[hl_y],
                mode="markers+text",
                marker=dict(size=12, color="#FFD600", symbol="diamond"),
                text=[f"t1/2 = {fit['half_life']:.1f}"],
                textposition="top right",
                textfont=dict(color="#FFD600", size=12),
                name="Half-Life",
            ))

    fig.add_hline(y=0, line=dict(color="gray", width=1, dash="dot"))

    fig.update_layout(
        xaxis=dict(title="In-Game Repetition Count"),
        yaxis=dict(title="Alpha (Whiff Rate Uplift)"),
        template="plotly_dark",
        height=450,
        margin=dict(l=50, r=30, t=30, b=50),
        legend=dict(x=0.6, y=0.95),
    )
    st.plotly_chart(fig, use_container_width=True, key="ad_intra_game_chart")


def _render_half_life_comparison(result: dict) -> None:
    """Bar chart comparing intra-game vs series half-lives per transition."""
    st.markdown("**Half-Life Comparison: Intra-Game vs Series**")

    transitions = result.get("transitions", {})

    rows = []
    for trans, data in transitions.items():
        intra_fit = data.get("intra_game", {}).get("fit")
        series_fit = data.get("series", {}).get("fit")

        intra_hl = intra_fit["half_life"] if intra_fit else None
        series_hl = series_fit["half_life"] if series_fit else None

        if intra_hl is not None or series_hl is not None:
            rows.append({
                "Transition": trans,
                "Intra-Game Half-Life": intra_hl,
                "Series Half-Life": series_hl,
            })

    if not rows:
        st.info("No fitted half-lives available for comparison.")
        return

    df = pd.DataFrame(rows)

    fig = go.Figure()

    # Intra-game bars
    fig.add_trace(go.Bar(
        x=df["Transition"],
        y=df["Intra-Game Half-Life"],
        name="Intra-Game",
        marker_color="#E81828",
    ))

    # Series bars
    fig.add_trace(go.Bar(
        x=df["Transition"],
        y=df["Series Half-Life"],
        name="Cross-Series",
        marker_color="#3498DB",
    ))

    fig.update_layout(
        barmode="group",
        xaxis=dict(title="Pitch Transition", tickangle=-45),
        yaxis=dict(title="Half-Life (repetitions / games)"),
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=30, t=30, b=80),
        legend=dict(x=0.7, y=0.95),
    )
    st.plotly_chart(fig, use_container_width=True, key="ad_hl_comparison")


def _render_transition_table(result: dict) -> None:
    """Sequence-specific breakdown table."""
    st.markdown("**Sequence Breakdown**")

    transitions = result.get("transitions", {})

    rows = []
    for trans, data in transitions.items():
        intra_fit = data.get("intra_game", {}).get("fit")
        series_fit = data.get("series", {}).get("fit")

        rows.append({
            "Transition": trans,
            "Alpha": round(data.get("alpha", 0.0), 4),
            "Cond. Whiff%": round(data.get("conditional_whiff_rate", 0.0) * 100, 1),
            "Uncond. Whiff%": round(data.get("unconditional_whiff_rate", 0.0) * 100, 1),
            "Swings": data.get("swing_count", 0),
            "Game t1/2": round(intra_fit["half_life"], 1) if intra_fit else None,
            "Game R2": round(intra_fit["r_squared"], 3) if intra_fit else None,
            "Series t1/2": round(series_fit["half_life"], 1) if series_fit else None,
            "Series R2": round(series_fit["r_squared"], 3) if series_fit else None,
        })

    if not rows:
        st.info("No transition data available.")
        return

    df = pd.DataFrame(rows).sort_values("Alpha", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_regime_timeline(result: dict) -> None:
    """Show regime detection timeline if available."""
    regime = result.get("regime", {})
    timeseries = regime.get("regime_timeseries", [])

    if not timeseries:
        return

    st.markdown("**Sequencing Effectiveness Regime**")
    st.caption(
        "Rolling alpha across the season.  'High alpha' means sequencing "
        "patterns are generating above-median deception."
    )

    current = regime.get("current_regime")
    if current:
        color = "green" if current == "high_alpha" else "orange"
        st.markdown(
            f"Current regime: :{color}[**{current.replace('_', ' ').title()}**]"
        )

    dates = [r["game_date"] for r in timeseries]
    rolling_alphas = [r["rolling_alpha"] for r in timeseries]
    regimes = [r["regime"] for r in timeseries]
    colors = ["#2ECC71" if r == "high_alpha" else "#E74C3C" for r in regimes]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=dates,
        y=rolling_alphas,
        mode="lines+markers",
        marker=dict(size=5, color=colors),
        line=dict(color="rgba(255,255,255,0.3)", width=1),
        name="Rolling Alpha",
    ))

    median_alpha = regime.get("median_alpha", 0.0)
    fig.add_hline(
        y=median_alpha,
        line=dict(color="white", width=1, dash="dash"),
        annotation_text=f"Median ({median_alpha:.4f})",
        annotation_position="top right",
    )

    fig.update_layout(
        xaxis=dict(title="Game Date"),
        yaxis=dict(title="Rolling Alpha"),
        template="plotly_dark",
        height=350,
        margin=dict(l=50, r=30, t=30, b=50),
    )
    st.plotly_chart(fig, use_container_width=True, key="ad_regime_timeline")


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def _render_leaderboard(conn, season, min_pitches) -> None:
    """Display the 'Hardest to Decode' leaderboard."""
    st.subheader("Hardest to Decode Leaderboard")
    st.caption(
        "Pitchers whose sequencing patterns maintain their deceptive edge "
        "the longest (highest adaptability score / longest half-lives)."
    )

    # Try cache first
    df = None
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "alpha_decay", season)
            if cached is not None:
                df = cached
                age_info = cache_age_display(conn, "alpha_decay", season)
                if age_info:
                    st.caption(age_info)
        except Exception:
            pass

    if df is None:
        with st.spinner("Computing... Run `python scripts/precompute.py` for instant loading."):
            try:
                df = batch_calculate(conn, season=season, min_pitches=min_pitches)
            except Exception as exc:
                st.error(f"Error computing leaderboard: {exc}")
                return

    if df.empty:
        st.info("No qualifying pitchers found with the current filters.")
        return

    # Try to join pitcher names
    try:
        names_df = conn.execute(
            "SELECT player_id AS pitcher_id, full_name FROM players"
        ).fetchdf()
        df = df.merge(names_df, on="pitcher_id", how="left")
        df["full_name"] = df["full_name"].fillna(df["pitcher_id"].astype(str))
    except Exception:
        df["full_name"] = df["pitcher_id"].astype(str)

    # Format for display
    display_df = df[[
        "full_name", "adaptability_score", "avg_intra_game_half_life",
        "avg_series_half_life", "n_transitions_fitted",
    ]].copy()
    display_df = display_df.rename(columns={
        "full_name": "Pitcher",
        "adaptability_score": "Adaptability",
        "avg_intra_game_half_life": "Avg Game t1/2",
        "avg_series_half_life": "Avg Series t1/2",
        "n_transitions_fitted": "Transitions Fitted",
    })

    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = "Rank"

    st.dataframe(display_df, use_container_width=True)

    # Distribution chart
    scored = df.dropna(subset=["adaptability_score"])
    if len(scored) >= 3:
        st.markdown("**Adaptability Score Distribution**")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=scored["adaptability_score"],
            nbinsx=20,
            marker_color="#E81828",
            opacity=0.75,
        ))
        median_score = scored["adaptability_score"].median()
        fig.add_vline(
            x=median_score,
            line=dict(color="white", width=2, dash="dash"),
            annotation_text=f"Median ({median_score:.0f})",
            annotation_position="top right",
        )
        fig.update_layout(
            xaxis=dict(title="Adaptability Score (0-100)"),
            yaxis=dict(title="Count"),
            template="plotly_dark",
            height=350,
            margin=dict(l=50, r=30, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True, key="ad_adapt_dist")
