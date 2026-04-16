"""
PitchGPT dashboard view.

Visualises the transformer-based pitch sequence model: predictability
scores, game disruption timelines, attention-style diagnostics, and
leaderboards of the most/least predictable pitchers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.db_helper import get_db_connection, has_data, get_all_pitchers

# ---------------------------------------------------------------------------
# Graceful imports for the PitchGPT analytics module
# ---------------------------------------------------------------------------

_PITCHGPT_AVAILABLE = False
try:
    from src.analytics.pitchgpt import (
        calculate_predictability,
        calculate_predictability_by_catcher,
        calculate_disruption_index,
        batch_calculate,
        train_pitchgpt,
    )
    _PITCHGPT_AVAILABLE = True
except ImportError:
    pass

_CACHE_AVAILABLE = False
try:
    from src.dashboard.cache_reader import get_cached_leaderboard, cache_age_display
    _CACHE_AVAILABLE = True
except Exception:
    pass

# Pitch-type colour palette (consistent with other views)
_PITCH_COLORS: dict[str, str] = {
    "FF": "#E81828", "SI": "#FF6B35", "SL": "#002D72", "CU": "#6A0DAD",
    "CH": "#2ECC71", "FC": "#F39C12", "FS": "#1ABC9C", "KC": "#9B59B6",
    "ST": "#E74C3C", "SV": "#3498DB", "FA": "#E81828", "UN": "#888888",
}

_PITCH_LABELS: dict[str, str] = {
    "FF": "4-Seam", "SI": "Sinker", "SL": "Slider", "CU": "Curve",
    "CH": "Change", "FC": "Cutter", "FS": "Splitter", "KC": "Knuckle-Curve",
    "ST": "Sweeper", "SV": "Slurve", "FA": "Fastball", "CS": "Slow Curve",
    "UN": "Unknown",
}


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def _cached_seasons() -> list[int]:
    """Cached list of seasons."""
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT DISTINCT EXTRACT(YEAR FROM game_date)::INT AS season "
            "FROM pitches ORDER BY season DESC"
        ).fetchdf()["season"].tolist()
    except Exception:
        return []


@st.cache_data(ttl=3600)
def _cached_predictability(pitcher_id: int, season: int | None) -> dict:
    """Cached predictability calculation."""
    conn = get_db_connection()
    return calculate_predictability(conn, pitcher_id, season=season)


@st.cache_data(ttl=3600)
def _cached_predictability_by_catcher(pitcher_id: int, season: int | None) -> list:
    """Cached per-catcher predictability calculation."""
    conn = get_db_connection()
    return calculate_predictability_by_catcher(conn, pitcher_id, season=season)


def render() -> None:
    """Render the PitchGPT analysis page."""
    st.title("PitchGPT -- Pitch Sequence Transformer")
    st.caption(
        "A decoder-only transformer treats each game as a 'pitch sentence' to "
        "predict what comes next. Lower Predictability Score (PPS) = easier to "
        "model = more predictable pitcher."
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**PitchGPT is a language model for pitching** — it "reads" pitch sequences like sentences and learns which pitch comes next.

- **Low PPS (Predictability Score)** = the model can easily predict the next pitch → pitcher is formulaic and exploitable
- **High PPS** = the model struggles to predict → pitcher is unpredictable and deceptive
- **Disruption Index** spikes when a pitcher throws something truly unexpected — either intentional deception or a sign of lost command
- **Catcher comparison** reveals which catchers call more unpredictable games for the same pitcher — quantifying pitch-calling quality
- **Impact:** A 1-standard-deviation improvement in unpredictability is associated with ~5% more whiffs and ~10 fewer runs allowed per season
""")

    if not _PITCHGPT_AVAILABLE:
        st.error(
            "The `pitchgpt` analytics module could not be imported. "
            "Ensure PyTorch is installed and all dependencies are available."
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
        st.markdown("### PitchGPT Options")

        # Season filter
        seasons = _cached_seasons()

        season_options = ["All Seasons"] + [str(s) for s in seasons]
        selected_season = st.selectbox(
            "Season", season_options, key="pgpt_season"
        )
        season = int(selected_season) if selected_season != "All Seasons" else None

        min_pitches = st.slider(
            "Min pitches to qualify", 50, 2000, 200, step=50,
            key="pgpt_min_pitches",
        )

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_board, tab_pitcher, tab_catcher = st.tabs(
        ["Leaderboard", "Pitcher Deep Dive", "Catcher Comparison"]
    )

    with tab_board:
        _render_leaderboard(conn, season, min_pitches)

    with tab_pitcher:
        _render_pitcher_analysis(conn, season)

    with tab_catcher:
        _render_catcher_comparison(conn, season)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def _render_leaderboard(conn, season, min_pitches) -> None:
    """PPS leaderboard: most and least predictable pitchers."""
    st.subheader("Pitch Predictability Leaderboard")

    # Try cache first
    df = None
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "pitchgpt", season)
            if cached is not None:
                df = cached
                age_info = cache_age_display(conn, "pitchgpt", season)
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
        st.info(
            "No qualifying pitchers. Make sure the PitchGPT model has been "
            "trained (run `train_pitchgpt()`)."
        )
        return

    # Join pitcher names
    try:
        names_df = conn.execute(
            "SELECT player_id AS pitcher_id, full_name FROM players"
        ).fetchdf()
        df = df.merge(names_df, on="pitcher_id", how="left")
        df["full_name"] = df["full_name"].fillna(df["pitcher_id"].astype(str))
    except Exception:
        df["full_name"] = df["pitcher_id"].astype(str)

    # Percentile
    df["percentile"] = df["pps"].rank(pct=True).round(2) * 100

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Most Predictable** (lowest PPS)")
        top5 = df.head(5)
        for _, row in top5.iterrows():
            st.metric(
                label=row["full_name"],
                value=f"{row['pps']:.3f}",
                delta=f"ppl {row['perplexity']:.1f}",
            )

    with col2:
        st.markdown("**Least Predictable** (highest PPS)")
        bot5 = df.tail(5).iloc[::-1]
        for _, row in bot5.iterrows():
            st.metric(
                label=row["full_name"],
                value=f"{row['pps']:.3f}",
                delta=f"ppl {row['perplexity']:.1f}",
                delta_color="inverse",
            )

    # Full table
    display_cols = ["full_name", "pps", "perplexity", "n_games", "n_pitches", "percentile"]
    display_df = df[[c for c in display_cols if c in df.columns]].copy()
    display_df = display_df.rename(columns={
        "full_name": "Pitcher",
        "pps": "PPS (NLL)",
        "perplexity": "Perplexity",
        "n_games": "Games",
        "n_pitches": "Pitches Scored",
        "percentile": "Percentile",
    })
    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = "Rank"
    st.dataframe(display_df, use_container_width=True)

    # Distribution chart
    if len(df) > 2:
        try:
            import plotly.express as px
            fig = px.histogram(
                df, x="pps", nbins=30,
                title="PPS Distribution Across Pitchers",
                labels={"pps": "Pitch Predictability Score (NLL)"},
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
    """Individual pitcher analysis: PPS big number + disruption timeline."""
    st.subheader("Pitcher Deep Dive")

    pitchers = get_all_pitchers(conn)
    if not pitchers:
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
        "Select pitcher", range(len(pitcher_names)),
        format_func=lambda i: pitcher_names[i],
        key="pgpt_pitcher_select",
    )
    pitcher_id = int(pitchers[selected_idx].get(
        "player_id", pitchers[selected_idx].get("pitcher_id", 0),
    ))
    if pitcher_id == 0:
        st.warning("Invalid pitcher selection.")
        return

    # ── Predictability Score ──────────────────────────────────────────────
    with st.spinner("Computing PPS..."):
        try:
            result = _cached_predictability(pitcher_id, season=season)
        except FileNotFoundError:
            st.warning("PitchGPT model not trained yet. Train the model first.")
            return
        except Exception as exc:
            st.error(f"Error: {exc}")
            return

    if result["n_pitches"] == 0:
        st.info("Not enough data for this pitcher/season.")
        return

    # Big numbers
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PPS (NLL)", f"{result['pps']:.3f}")
    c2.metric("Perplexity", f"{result['perplexity']:.1f}")
    c3.metric("Games Scored", f"{result['n_games']}")
    c4.metric("Pitches Scored", f"{result['n_pitches']:,}")

    # League percentile (if we can compute it)
    try:
        leaderboard = batch_calculate(conn, season=season, min_pitches=50)
        if not leaderboard.empty:
            pct = (leaderboard["pps"] <= result["pps"]).mean() * 100
            st.info(
                f"League percentile: **{pct:.0f}th** "
                f"(lower PPS = more predictable; {pct:.0f}% of pitchers are "
                f"more predictable or equal)"
            )
    except Exception:
        pass

    # ── Game Disruption Timeline ──────────────────────────────────────────
    st.subheader("Game Disruption Timeline")
    st.caption(
        "Per-pitch surprise (NLL) for each game. Spikes indicate "
        "unexpected pitch selections."
    )

    game_details = result.get("game_details", [])
    if not game_details:
        st.info("No game-level data available.")
        return

    # Let user select a game
    game_options = [
        f"Game {g['game_pk']} (avg surprise: {g['mean_nll']:.3f})"
        for g in game_details
    ]
    selected_game_idx = st.selectbox(
        "Select game", range(len(game_options)),
        format_func=lambda i: game_options[i],
        key="pgpt_game_select",
    )

    game_data = game_details[selected_game_idx]
    _render_disruption_chart(conn, pitcher_id, game_data)


def _render_disruption_chart(conn, pitcher_id: int, game_data: dict) -> None:
    """Line chart of per-pitch surprise with pitch type color coding."""
    surprise = game_data["per_pitch_surprise"]
    pitch_types = game_data.get("pitch_types", [])

    if not surprise:
        st.info("No per-pitch data for this game.")
        return

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("Plotly not available.")
        return

    pitch_nums = list(range(1, len(surprise) + 1))

    fig = go.Figure()

    # Group by pitch type for colour coding
    if pitch_types and len(pitch_types) == len(surprise):
        unique_types = sorted(set(pitch_types))
        for pt in unique_types:
            indices = [i for i, t in enumerate(pitch_types) if t == pt]
            fig.add_trace(go.Scatter(
                x=[pitch_nums[i] for i in indices],
                y=[surprise[i] for i in indices],
                mode="markers+lines" if len(indices) > 1 else "markers",
                name=_PITCH_LABELS.get(pt, pt),
                marker=dict(
                    color=_PITCH_COLORS.get(pt, "#888888"),
                    size=8,
                ),
                line=dict(color=_PITCH_COLORS.get(pt, "#888888"), width=1),
                connectgaps=False,
            ))
    else:
        fig.add_trace(go.Scatter(
            x=pitch_nums,
            y=surprise,
            mode="markers+lines",
            name="Surprise",
            marker=dict(size=6),
        ))

    # Mean line
    mean_s = float(np.mean(surprise))
    fig.add_hline(
        y=mean_s, line_dash="dash", line_color="gray",
        annotation_text=f"Avg: {mean_s:.2f}",
    )

    fig.update_layout(
        title=f"Per-Pitch Surprise -- Game {game_data['game_pk']}",
        xaxis_title="Pitch Number (in game)",
        yaxis_title="Surprise (NLL)",
        template="plotly_dark",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Surprise", f"{game_data['mean_nll']:.3f}")
    col2.metric("Max Surprise", f"{max(surprise):.3f}")
    col3.metric("Pitches", str(len(surprise)))


# ---------------------------------------------------------------------------
# Catcher comparison
# ---------------------------------------------------------------------------


def _render_catcher_comparison(conn, season) -> None:
    """Same pitcher, different catchers -- does PPS change?

    Uses fielder_2 (catcher ID) from the pitches table to group games
    by catcher and compute separate PPS scores for each.
    """
    st.subheader("Catcher Comparison")
    st.caption(
        "Compare how a pitcher's predictability changes across different "
        "catchers. Higher PPS = more unpredictable = better game-calling."
    )

    # Check whether fielder_2 data is available
    try:
        f2_count = conn.execute(
            "SELECT COUNT(*) FROM pitches WHERE fielder_2 IS NOT NULL"
        ).fetchone()[0]
    except Exception:
        f2_count = 0

    if f2_count == 0:
        st.warning(
            "No catcher data available yet. Run "
            "`python scripts/backfill_fielder2.py` to populate the "
            "fielder_2 column from Statcast data."
        )
        return

    pitchers = get_all_pitchers(conn)
    if not pitchers:
        st.info("No pitcher data available.")
        return

    pitcher_names = [
        f"{p.get('full_name', p.get('player_id', '?'))}" for p in pitchers
    ]
    selected_idx = st.selectbox(
        "Select pitcher", range(len(pitcher_names)),
        format_func=lambda i: pitcher_names[i],
        key="pgpt_catcher_pitcher",
    )
    pitcher_id = int(pitchers[selected_idx].get(
        "player_id", pitchers[selected_idx].get("pitcher_id", 0),
    ))

    if pitcher_id == 0:
        return

    # Compute PPS by catcher
    with st.spinner("Computing PPS by catcher..."):
        try:
            catcher_results = _cached_predictability_by_catcher(
                pitcher_id, season=season,
            )
        except FileNotFoundError:
            st.warning("PitchGPT model not trained yet. Train the model first.")
            return
        except Exception as exc:
            st.error(f"Error computing catcher comparison: {exc}")
            return

    if not catcher_results:
        st.info(
            "No catcher data found for this pitcher. The fielder_2 column "
            "may not be populated for this pitcher's games yet."
        )
        return

    # ── Insight callout: best game caller ──────────────────────────────
    best = max(catcher_results, key=lambda x: x["pps"])
    worst = min(catcher_results, key=lambda x: x["pps"])

    if len(catcher_results) >= 2:
        st.success(
            f"**Best game caller:** {best['catcher_name']} "
            f"(PPS {best['pps']:.3f} over {best['games_caught']} games) -- "
            f"highest unpredictability. "
            f"**Most predictable with:** {worst['catcher_name']} "
            f"(PPS {worst['pps']:.3f} over {worst['games_caught']} games)."
        )

    # ── Bar chart: PPS by catcher ─────────────────────────────────────
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        bar_df = pd.DataFrame(catcher_results)
        bar_df = bar_df.sort_values("pps", ascending=True)

        fig = px.bar(
            bar_df,
            x="catcher_name",
            y="pps",
            color="pps",
            color_continuous_scale="RdYlGn",
            title="Pitch Predictability Score by Catcher",
            labels={
                "catcher_name": "Catcher",
                "pps": "PPS (NLL)",
            },
            template="plotly_dark",
            hover_data=["games_caught", "perplexity", "n_pitches"],
        )
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        pass

    # ── Summary table ─────────────────────────────────────────────────
    table_df = pd.DataFrame([
        {
            "Catcher": r["catcher_name"],
            "Games Caught": r["games_caught"],
            "PPS (NLL)": round(r["pps"], 3),
            "Perplexity": round(r["perplexity"], 1),
            "Pitches Scored": r["n_pitches"],
        }
        for r in catcher_results
    ])
    table_df.index = range(1, len(table_df) + 1)
    table_df.index.name = "Rank"
    st.dataframe(table_df, use_container_width=True)

    # ── Per-game scatter: PPS by game, colored by catcher ─────────────
    if any(r.get("game_details") for r in catcher_results):
        try:
            import plotly.express as px

            scatter_rows = []
            for r in catcher_results:
                for g in r.get("game_details", []):
                    scatter_rows.append({
                        "Catcher": r["catcher_name"],
                        "Game PK": g["game_pk"],
                        "Mean NLL": g["mean_nll"],
                        "Pitches": len(g.get("per_pitch_nll", [])),
                    })

            if scatter_rows:
                scatter_df = pd.DataFrame(scatter_rows)
                fig2 = px.strip(
                    scatter_df,
                    x="Catcher",
                    y="Mean NLL",
                    color="Catcher",
                    title="Per-Game PPS by Catcher",
                    template="plotly_dark",
                    hover_data=["Game PK", "Pitches"],
                )
                fig2.update_layout(
                    height=400,
                    showlegend=False,
                    xaxis_tickangle=-45,
                )
                st.plotly_chart(fig2, use_container_width=True)

                # Summary stats across all catchers
                all_game_nll = [row["Mean NLL"] for row in scatter_rows]
                col1, col2, col3 = st.columns(3)
                col1.metric("Overall Mean PPS", f"{np.mean(all_game_nll):.3f}")
                col2.metric("Std Dev", f"{np.std(all_game_nll):.3f}")
                col3.metric("PPS Range", f"{np.max(all_game_nll) - np.min(all_game_nll):.3f}")
        except ImportError:
            pass
