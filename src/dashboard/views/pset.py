"""
Pitch Sequence Expected Threat (PSET) dashboard view.

Visualises how effectively pitchers sequence their pitches using the
game-theory-inspired PSET model: transition matrix heatmaps,
leaderboards, pitch-pair effectiveness charts, and predictability vs
tunnel scatter plots.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.db_helper import get_db_connection, has_data, get_all_pitchers

# ---------------------------------------------------------------------------
# Graceful imports for the PSET analytics module
# ---------------------------------------------------------------------------

_PSET_AVAILABLE = False
try:
    from src.analytics.pset import (
        build_transition_matrix,
        calculate_pset,
        batch_calculate,
        get_best_sequences,
    )
    _PSET_AVAILABLE = True
except ImportError:
    pass

_CACHE_AVAILABLE = False
try:
    from src.dashboard.cache_reader import get_cached_leaderboard, cache_age_display
    _CACHE_AVAILABLE = True
except Exception:
    pass

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
# Cached computation helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=600)
def _cached_pset(pitcher_id: int, season=None) -> dict:
    """Try DB entity cache first, then live computation."""
    conn = get_db_connection()
    # Check per-entity precomputed cache
    if _CACHE_AVAILABLE and season is not None:
        from src.dashboard.cache_reader import get_cached_entity
        cached = get_cached_entity(conn, "pset", pitcher_id, season)
        if cached is not None:
            return cached
    return calculate_pset(conn, pitcher_id, season=season)


@st.cache_data(ttl=600)
def _cached_best_sequences(pitcher_id: int, season=None, top_n: int = 5) -> list:
    """Cache best sequences lookup."""
    conn = get_db_connection()
    return get_best_sequences(conn, pitcher_id, season=season, top_n=top_n)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the PSET Analysis page."""
    st.title("Pitch Sequence Expected Threat (PSET)")
    st.caption(
        "Game-theory sequencing model: how much value does a pitcher's pitch "
        "ordering create through tunneling and unpredictability?"
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**PSET measures how much run value a pitcher creates through sequencing** — the art of pitch ordering, not just individual pitch quality.

- **PSET > 0** = sequencing adds value (tunneling and unpredictability make individual pitches play up)
- **PSET < 0** = sequencing destroys value (predictable patterns let hitters sit on pitches)
- **Tunnel Bonus** rewards pitch pairs that look identical early in flight but diverge at the plate — pure deception
- **Predictability Penalty** punishes repetitive patterns that hitters can anticipate
- **"Sequencing Genius"** = high PSET but average Stuff+ (think Greg Maddux — made average stuff elite through pitch calling)
- **Impact:** The gap between the best and worst sequencers is worth ~15-20 runs per season — equivalent to 1.5-2 WAR
""")

    if not _PSET_AVAILABLE:
        st.error(
            "The `pset` analytics module could not be imported. "
            "Ensure the module is installed and all dependencies are available."
        )
        return

    conn = get_db_connection()

    if conn is None or not has_data(conn):
        st.warning(
            "No pitch data available. Run the data backfill pipeline first "
            "(`python scripts/backfill.py`)."
        )
        return

    # ── Sidebar controls ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### PSET Options")

        # Season filter
        try:
            seasons = conn.execute(
                "SELECT DISTINCT EXTRACT(YEAR FROM game_date)::INT AS season "
                "FROM pitches ORDER BY season DESC"
            ).fetchdf()["season"].tolist()
        except Exception:
            seasons = []

        season_options = ["All Seasons"] + [str(s) for s in seasons]
        selected_season = st.selectbox(
            "Season", season_options, key="pset_season"
        )
        season = int(selected_season) if selected_season != "All Seasons" else None

        min_pitches = st.slider(
            "Min pitches to qualify",
            50, 2000, 300, step=50,
            key="pset_min_pitches",
        )

    # ── Main content ──────────────────────────────────────────────────────
    tab_board, tab_pitcher, tab_compare = st.tabs(
        ["Leaderboard", "Pitcher Deep Dive", "Stuff vs Sequencing"]
    )

    with tab_board:
        _render_leaderboard(conn, season, min_pitches)

    with tab_pitcher:
        _render_pitcher_analysis(conn, season)

    with tab_compare:
        _render_stuff_vs_sequencing(conn, season, min_pitches)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def _render_leaderboard(conn, season, min_pitches) -> None:
    """Display the PSET leaderboard for all qualifying pitchers."""
    st.subheader("PSET Leaderboard")

    # Try cache first
    df = None
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "pset", season)
            if cached is not None:
                df = cached
                age_info = cache_age_display(conn, "pset", season)
                if age_info:
                    st.caption(age_info)
        except Exception:
            pass

    if df is None:
        try:
            with st.spinner("Computing... Run `python scripts/precompute.py` for instant loading."):
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
        df["full_name"] = df["full_name"].fillna(
            df["pitcher_id"].astype(str)
        )
    except Exception:
        df["full_name"] = df["pitcher_id"].astype(str)

    # Reorder columns for display
    display_cols = [
        "full_name", "pset_per_100", "predictability_score",
        "tunnel_score", "total_pairs",
    ]
    display_df = df[[c for c in display_cols if c in df.columns]].copy()
    display_df = display_df.rename(columns={
        "full_name": "Pitcher",
        "pset_per_100": "PSET / 100",
        "predictability_score": "Predictability",
        "tunnel_score": "Tunnel Score",
        "total_pairs": "Pitch Pairs",
    })

    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = "Rank"

    st.dataframe(display_df, use_container_width=True)

    # Distribution chart
    if len(df) > 2:
        try:
            import plotly.express as px

            fig = px.histogram(
                df,
                x="pset_per_100",
                nbins=30,
                title="PSET per 100 Distribution",
                labels={"pset_per_100": "PSET / 100 Pitches"},
                template="plotly_dark",
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Pitcher deep-dive
# ---------------------------------------------------------------------------


def _render_pitcher_analysis(conn, season) -> None:
    """Individual pitcher PSET analysis with transition heatmap, best
    sequences, and pitch-pair effectiveness chart."""
    st.subheader("Pitcher Deep Dive")

    # Pitcher selector
    pitchers = get_all_pitchers(conn)
    if not pitchers:
        # Fallback: grab pitcher IDs from data
        try:
            pitchers_df = conn.execute(
                "SELECT DISTINCT pitcher_id FROM pitches LIMIT 200"
            ).fetchdf()
            pitchers = [
                {"player_id": int(r), "full_name": str(int(r))}
                for r in pitchers_df["pitcher_id"]
            ]
        except Exception:
            st.info("No pitcher data available.")
            return

    pitcher_names = [
        f"{p.get('full_name', p.get('player_id', '?'))}" for p in pitchers
    ]
    selected_idx = st.selectbox(
        "Select pitcher",
        range(len(pitcher_names)),
        format_func=lambda i: pitcher_names[i],
        key="pset_pitcher_select",
    )

    if selected_idx is None:
        st.info("Select a pitcher to analyze.")
        return

    pitcher_id = int(pitchers[selected_idx].get(
        "player_id", pitchers[selected_idx].get("pitcher_id", 0)
    ))

    if pitcher_id == 0:
        st.warning("Invalid pitcher selection.")
        return

    # Calculate PSET (cached so re-renders are instant)
    try:
        result = _cached_pset(pitcher_id, season)
    except Exception as exc:
        st.error(f"Error computing PSET: {exc}")
        return

    if result is None or result.get("total_pairs", 0) == 0:
        st.info("Not enough pitch-pair data for this pitcher/season combination.")
        return

    # ── Summary metrics ───────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PSET / 100", f"{result['pset_per_100']:+.2f}")
    c2.metric("Predictability", f"{result['predictability_score']:.3f}")
    c3.metric("Tunnel Score", f"{result['tunnel_score']:.3f}")
    c4.metric("Pitch Pairs", f"{result['total_pairs']:,}")

    # ── Best sequences callout ────────────────────────────────────────────
    best = _cached_best_sequences(pitcher_id, season=season, top_n=5)
    if best:
        top = best[0]
        pair_display = top["pair"].replace(" -> ", " --> ")
        st.success(
            f"Deadliest combo: **{pair_display}** "
            f"({top['pset_contribution']:+.2f} runs per 100 pitches, "
            f"n={top['count']})"
        )

    # ── Transition matrix heatmap ─────────────────────────────────────────
    _render_transition_heatmap(conn, pitcher_id, season)

    # ── Pitch pair effectiveness chart ────────────────────────────────────
    _render_pair_effectiveness(result)

    # ── Predictability vs Tunnel scatter ──────────────────────────────────
    _render_pred_vs_tunnel_scatter(result)


def _render_transition_heatmap(conn, pitcher_id, season) -> None:
    """Render the P(next | current) transition matrix as a Plotly heatmap."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("Plotly not available for heatmap rendering.")
        return

    tm = build_transition_matrix(conn, pitcher_id, season=season)
    if not tm:
        return

    # Collapse across counts for an overall matrix
    combined: dict[str, dict[str, float]] = {}
    combined_counts: dict[str, dict[str, int]] = {}

    for _count, prev_types in tm.items():
        for prev_pt, nexts in prev_types.items():
            if prev_pt not in combined_counts:
                combined_counts[prev_pt] = {}
            for nxt, prob in nexts.items():
                combined_counts[prev_pt][nxt] = (
                    combined_counts[prev_pt].get(nxt, 0) + 1
                )

    # Re-estimate probabilities from aggregated counts
    for prev_pt, nexts in combined_counts.items():
        total = sum(nexts.values())
        combined[prev_pt] = {
            nxt: round(cnt / total, 3) for nxt, cnt in nexts.items()
        }

    if not combined:
        return

    all_types = sorted(
        set(combined.keys()) | {n for d in combined.values() for n in d}
    )
    labels = [_PITCH_LABELS.get(pt, pt) for pt in all_types]

    z = []
    text = []
    for pt in all_types:
        row = []
        text_row = []
        for nxt in all_types:
            val = combined.get(pt, {}).get(nxt, 0.0)
            row.append(val)
            text_row.append(f"{val:.1%}")
        z.append(row)
        text.append(text_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Viridis",
            colorbar_title="P(next)",
        )
    )
    fig.update_layout(
        title="Pitch Transition Matrix: P(Next Pitch | Current Pitch)",
        xaxis_title="Next Pitch",
        yaxis_title="Current Pitch",
        template="plotly_dark",
        height=450,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_pair_effectiveness(result: dict) -> None:
    """Bar chart of PSET contribution by pitch-pair type."""
    breakdown = result.get("breakdown", [])
    if not breakdown:
        return

    try:
        import plotly.express as px
    except ImportError:
        return

    bd_df = pd.DataFrame(breakdown)

    fig = px.bar(
        bd_df,
        x="pair",
        y="pset_contribution",
        color="pset_contribution",
        color_continuous_scale=["#FF1744", "#FFD600", "#00C853"],
        title="PSET Contribution by Pitch Pair",
        labels={
            "pair": "Pitch Pair",
            "pset_contribution": "PSET / 100",
        },
        template="plotly_dark",
    )
    fig.update_layout(
        height=400,
        xaxis_tickangle=-45,
        coloraxis_showscale=False,
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_pred_vs_tunnel_scatter(result: dict) -> None:
    """Scatter: x=predictability, y=tunnel score, per pitch-pair type."""
    breakdown = result.get("breakdown", [])
    if not breakdown:
        return

    try:
        import plotly.express as px
    except ImportError:
        return

    bd_df = pd.DataFrame(breakdown)
    if "avg_predictability" not in bd_df.columns:
        return

    fig = px.scatter(
        bd_df,
        x="avg_predictability",
        y="avg_tunnel_score",
        size="count",
        color="pset_contribution",
        color_continuous_scale=["#FF1744", "#FFD600", "#00C853"],
        hover_name="pair",
        title="Predictability vs Tunnel Quality by Pitch Pair",
        labels={
            "avg_predictability": "Predictability (higher = more predictable)",
            "avg_tunnel_score": "Tunnel Score (higher = better tunnel)",
            "pset_contribution": "PSET / 100",
        },
        template="plotly_dark",
    )
    fig.update_layout(height=450)

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Stuff vs Sequencing comparison
# ---------------------------------------------------------------------------


def _render_stuff_vs_sequencing(conn, season, min_pitches) -> None:
    """Compare Stuff+ (raw quality) vs PSET (sequencing value).

    Highlights 'Sequencing Geniuses' (high PSET, average Stuff+) and
    'Stuff Guys' (high Stuff+, low PSET).
    """
    st.subheader("Sequencing Genius vs Stuff Guy")
    st.caption(
        "Pitchers with high PSET but average physical stuff prove that "
        "pitch ordering creates real value. The reverse shows elite stuff "
        "being undercut by predictable patterns."
    )

    # Get PSET leaderboard
    try:
        pset_df = batch_calculate(conn, season=season, min_pitches=min_pitches)
    except Exception as exc:
        st.error(f"Error computing PSET: {exc}")
        return

    if pset_df.empty:
        st.info("No qualifying pitchers.")
        return

    # Try to get Stuff+ data
    stuff_available = False
    try:
        from src.analytics.stuff_model import batch_calculate_stuff_plus
        stuff_df = batch_calculate_stuff_plus(conn, min_pitches=min_pitches)
        if not stuff_df.empty and "pitcher_id" in stuff_df.columns:
            stuff_available = True
    except Exception:
        pass

    if not stuff_available:
        st.info(
            "Stuff+ model not available or not trained. Train it from the "
            "Stuff+ page to enable this comparison."
        )
        # Still show PSET-only scatter
        _show_pset_only_scatter(pset_df, conn)
        return

    # Merge PSET and Stuff+
    merged = pset_df.merge(stuff_df, on="pitcher_id", how="inner")
    if merged.empty:
        st.info("No pitchers overlap between PSET and Stuff+ datasets.")
        return

    # Try to join names
    try:
        names_df = conn.execute(
            "SELECT player_id AS pitcher_id, full_name FROM players"
        ).fetchdf()
        merged = merged.merge(names_df, on="pitcher_id", how="left")
        merged["full_name"] = merged["full_name"].fillna(
            merged["pitcher_id"].astype(str)
        )
    except Exception:
        merged["full_name"] = merged["pitcher_id"].astype(str)

    # Determine the Stuff+ score column
    stuff_col = None
    for candidate in ["stuff_plus", "stuff_plus_mean", "overall_stuff_plus"]:
        if candidate in merged.columns:
            stuff_col = candidate
            break

    if stuff_col is None:
        st.info("Stuff+ score column not found in merged data.")
        return

    try:
        import plotly.express as px

        fig = px.scatter(
            merged,
            x=stuff_col,
            y="pset_per_100",
            hover_name="full_name",
            size="total_pairs",
            color="pset_per_100",
            color_continuous_scale=["#FF1744", "#FFD600", "#00C853"],
            title="Stuff Quality vs Sequencing Value",
            labels={
                stuff_col: "Stuff+ (100 = average)",
                "pset_per_100": "PSET / 100 Pitches",
            },
            template="plotly_dark",
        )
        fig.update_layout(height=500)

        # Add quadrant lines at medians
        med_stuff = float(merged[stuff_col].median())
        med_pset = float(merged["pset_per_100"].median())
        fig.add_hline(y=med_pset, line_dash="dash", line_color="gray",
                       opacity=0.5)
        fig.add_vline(x=med_stuff, line_dash="dash", line_color="gray",
                       opacity=0.5)

        # Quadrant annotations
        fig.add_annotation(
            x=merged[stuff_col].quantile(0.25),
            y=merged["pset_per_100"].quantile(0.75),
            text="Sequencing<br>Genius",
            showarrow=False, font=dict(color="lime", size=12),
        )
        fig.add_annotation(
            x=merged[stuff_col].quantile(0.75),
            y=merged["pset_per_100"].quantile(0.25),
            text="Stuff<br>Guy",
            showarrow=False, font=dict(color="orange", size=12),
        )

        st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        st.info("Plotly not available.")


def _show_pset_only_scatter(pset_df: pd.DataFrame, conn) -> None:
    """Fallback scatter: predictability vs tunnel, colored by PSET."""
    try:
        import plotly.express as px
    except ImportError:
        return

    try:
        names_df = conn.execute(
            "SELECT player_id AS pitcher_id, full_name FROM players"
        ).fetchdf()
        pset_df = pset_df.merge(names_df, on="pitcher_id", how="left")
        pset_df["full_name"] = pset_df["full_name"].fillna(
            pset_df["pitcher_id"].astype(str)
        )
    except Exception:
        pset_df["full_name"] = pset_df["pitcher_id"].astype(str)

    fig = px.scatter(
        pset_df,
        x="predictability_score",
        y="tunnel_score",
        color="pset_per_100",
        color_continuous_scale=["#FF1744", "#FFD600", "#00C853"],
        hover_name="full_name",
        size="total_pairs",
        title="Predictability vs Tunnel Score (colored by PSET)",
        labels={
            "predictability_score": "Predictability (higher = more predictable)",
            "tunnel_score": "Tunnel Score (higher = better)",
            "pset_per_100": "PSET / 100",
        },
        template="plotly_dark",
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)
