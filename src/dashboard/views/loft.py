"""
Lineup Order Flow Toxicity (LOFT) dashboard view.

Visualises the market-microstructure-inspired LOFT model: real-time-style
toxicity meter, buy/sell flow charts, alert timelines, and leaderboards
for the most decodable and most resilient pitchers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.db_helper import get_db_connection, has_data, get_all_pitchers

# ---------------------------------------------------------------------------
# Graceful imports for the LOFT analytics module
# ---------------------------------------------------------------------------

_LOFT_AVAILABLE = False
try:
    from src.analytics.loft import (
        classify_pitch_flow,
        compute_game_loft,
        compute_pitcher_baseline,
        detect_toxicity_events,
        batch_game_analysis,
        calculate_loft,
        LOFTModel,
        BUCKET_SIZE,
        EWMA_ALPHA,
        ALERT_SIGMA,
    )
    _LOFT_AVAILABLE = True
except ImportError:
    pass

_CACHE_AVAILABLE = False
try:
    from src.dashboard.cache_reader import get_cached_leaderboard, cache_age_display
    _CACHE_AVAILABLE = True
except Exception:
    pass


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the LOFT Analysis page."""
    st.title("Lineup Order Flow Toxicity (LOFT)")
    st.caption(
        "Market microstructure signal: detect when a lineup has decoded a "
        "pitcher by tracking the buy/sell imbalance of pitch outcomes."
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**LOFT detects the moment a lineup has collectively "figured out" a pitcher** — like detecting informed trading in financial markets.

- **LOFT near 0** = balanced outcomes (pitcher and hitters trading blows evenly)
- **LOFT near 1** = one-sided flow (either all hitter-favorable or all pitcher-favorable outcomes)
- **Toxicity threshold breach** = the rolling LOFT exceeds the pitcher's seasonal baseline by 2+ standard deviations — hitters have decoded him
- **Buy signals** = hard contact, walks, good takes (hitter-favorable)
- **Sell signals** = whiffs, chases, weak contact (pitcher-favorable)
- **Impact:** LOFT provides a pitch-by-pitch "pull him now" signal that's more precise than arbitrary pitch count thresholds — it captures *when* hitters have adapted, not just *how many pitches* have been thrown
""")

    if not _LOFT_AVAILABLE:
        st.error(
            "The `loft` analytics module could not be imported. "
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
        st.markdown("### LOFT Options")

        # Season filter
        try:
            seasons = conn.execute(
                "SELECT DISTINCT EXTRACT(YEAR FROM game_date)::INT AS season "
                "FROM pitches ORDER BY season DESC"
            ).fetchdf()["season"].tolist()
        except Exception:
            seasons = []

        if not seasons:
            st.info("No season data found.")
            return

        selected_season = st.selectbox(
            "Season", [str(s) for s in seasons], key="loft_season"
        )
        season = int(selected_season)

        min_games = st.slider(
            "Min games to qualify", 1, 30, 3, step=1,
            key="loft_min_games",
        )

    # ── Main tabs ─────────────────────────────────────────────────────────
    tab_game, tab_pitcher, tab_board = st.tabs(
        ["Game Analysis", "Pitcher Profile", "Leaderboard"]
    )

    with tab_game:
        _render_game_analysis(conn, season)

    with tab_pitcher:
        _render_pitcher_profile(conn, season)

    with tab_board:
        _render_leaderboard(conn, season, min_games)


# ---------------------------------------------------------------------------
# Game Analysis
# ---------------------------------------------------------------------------


def _render_game_analysis(conn, season: int) -> None:
    """Game-level LOFT analysis: toxicity meter and buy/sell flow."""
    st.subheader("Game Toxicity Analysis")

    # Pitcher selector
    pitchers = _get_pitcher_list(conn)
    if not pitchers:
        st.info("No pitcher data available.")
        return

    pitcher_names = [
        f"{p.get('full_name', str(p.get('player_id', '?')))}" for p in pitchers
    ]
    selected_idx = st.selectbox(
        "Select pitcher",
        range(len(pitcher_names)),
        format_func=lambda i: pitcher_names[i],
        key="loft_game_pitcher",
    )
    pitcher_id = int(pitchers[selected_idx].get(
        "player_id", pitchers[selected_idx].get("pitcher_id", 0)
    ))

    if pitcher_id == 0:
        st.warning("Invalid pitcher selection.")
        return

    # Game selector
    try:
        games_df = conn.execute("""
            SELECT DISTINCT game_pk, game_date,
                   home_team, away_team, COUNT(*) AS pitch_count
            FROM pitches
            WHERE pitcher_id = $1
              AND EXTRACT(YEAR FROM game_date) = $2
            GROUP BY game_pk, game_date, home_team, away_team
            HAVING COUNT(*) >= 15
            ORDER BY game_date DESC
        """, [pitcher_id, season]).fetchdf()
    except Exception:
        games_df = pd.DataFrame()

    if games_df.empty:
        st.info("No qualifying games found for this pitcher/season.")
        return

    game_options = [
        f"{row['game_date']} -- {row['away_team']} @ {row['home_team']} "
        f"({row['pitch_count']} pitches)"
        for _, row in games_df.iterrows()
    ]
    game_idx = st.selectbox("Select game", range(len(game_options)),
                            format_func=lambda i: game_options[i],
                            key="loft_game_select")
    game_pk = int(games_df.iloc[game_idx]["game_pk"])

    # Compute
    with st.spinner("Computing LOFT..."):
        game_loft = compute_game_loft(conn, game_pk, pitcher_id)
        baseline = compute_pitcher_baseline(conn, pitcher_id, season)

    if game_loft["total_buckets"] == 0:
        st.info("Not enough pitches for LOFT analysis in this game.")
        return

    # ── Summary metrics ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pitches", f"{game_loft['total_pitches']}")
    c2.metric("Mean LOFT", f"{game_loft['mean_loft']:.3f}")
    c3.metric("Max LOFT", f"{game_loft['max_loft']:.3f}")
    c4.metric("Buy / Sell", f"{game_loft['buy_total']} / {game_loft['sell_total']}")

    # ── Toxicity meter: line chart of rolling LOFT ────────────────────────
    _render_toxicity_meter(game_loft, baseline)

    # ── Buy/Sell flow: stacked area chart ─────────────────────────────────
    _render_buy_sell_flow(game_loft)

    # ── Alert timeline ───────────────────────────────────────────────────
    if baseline["n_games"] > 0:
        alerts = detect_toxicity_events(game_loft, baseline)
        _render_alert_timeline(game_loft, alerts, baseline)


def _render_toxicity_meter(game_loft: dict, baseline: dict) -> None:
    """Line chart of rolling LOFT across a game with threshold line."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.info("Plotly not available.")
        return

    buckets = game_loft.get("buckets", [])
    if not buckets:
        return

    bucket_ids = [b["bucket_id"] for b in buckets]
    rolling_loft = [b.get("rolling_loft", b["loft"]) for b in buckets]
    raw_loft = [b["loft"] for b in buckets]

    fig = go.Figure()

    # Raw LOFT
    fig.add_trace(go.Scatter(
        x=bucket_ids, y=raw_loft,
        mode="markers+lines",
        name="Raw LOFT",
        line=dict(color="#555555", width=1, dash="dot"),
        marker=dict(size=4),
        opacity=0.6,
    ))

    # Rolling LOFT (main signal)
    fig.add_trace(go.Scatter(
        x=bucket_ids, y=rolling_loft,
        mode="lines+markers",
        name="Rolling LOFT (EWMA)",
        line=dict(color="#FF6B35", width=3),
        marker=dict(size=6),
    ))

    # Threshold line
    if baseline.get("std_loft", 0) > 0:
        threshold = baseline["mean_loft"] + ALERT_SIGMA * baseline["std_loft"]
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#E81828",
            annotation_text=f"Alert Threshold ({threshold:.3f})",
            annotation_position="top right",
        )

    # Baseline mean
    if baseline.get("mean_loft", 0) > 0:
        fig.add_hline(
            y=baseline["mean_loft"],
            line_dash="dot",
            line_color="#2ECC71",
            annotation_text=f"Season Baseline ({baseline['mean_loft']:.3f})",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title="Toxicity Meter: Rolling LOFT Across the Game",
        xaxis_title="Bucket (15 pitches each)",
        yaxis_title="LOFT Score",
        yaxis_range=[0, 1.05],
        template="plotly_dark",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_buy_sell_flow(game_loft: dict) -> None:
    """Stacked area chart showing cumulative buy vs sell pitch counts."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return

    cum_buy = game_loft.get("cumulative_buy", [])
    cum_sell = game_loft.get("cumulative_sell", [])

    if not cum_buy:
        return

    pitch_nums = list(range(1, len(cum_buy) + 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pitch_nums, y=cum_buy,
        mode="lines",
        name="Cumulative Buy (Hitter)",
        fill="tozeroy",
        line=dict(color="#2ECC71", width=2),
        fillcolor="rgba(46, 204, 113, 0.3)",
    ))

    fig.add_trace(go.Scatter(
        x=pitch_nums, y=cum_sell,
        mode="lines",
        name="Cumulative Sell (Pitcher)",
        fill="tozeroy",
        line=dict(color="#E81828", width=2),
        fillcolor="rgba(232, 24, 40, 0.3)",
    ))

    fig.update_layout(
        title="Buy/Sell Order Flow",
        xaxis_title="Pitch Number",
        yaxis_title="Cumulative Count",
        template="plotly_dark",
        height=350,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_alert_timeline(game_loft: dict, alerts: list[dict], baseline: dict) -> None:
    """Markers on the pitch sequence where toxicity threshold was breached."""
    if not alerts:
        st.success("No toxicity alerts -- this pitcher was not decoded in this game.")
        return

    st.warning(
        f"**{len(alerts)} toxicity alert(s)** detected. "
        f"The lineup may have decoded this pitcher."
    )

    try:
        import plotly.graph_objects as go
    except ImportError:
        alert_df = pd.DataFrame(alerts)
        st.dataframe(alert_df, use_container_width=True)
        return

    # Show alerts as markers on the rolling LOFT chart
    buckets = game_loft.get("buckets", [])
    bucket_ids = [b["bucket_id"] for b in buckets]
    rolling_loft = [b.get("rolling_loft", b["loft"]) for b in buckets]

    alert_ids = [a["bucket_id"] for a in alerts]
    alert_vals = [a["rolling_loft"] for a in alerts]
    alert_text = [f"Bucket {a['bucket_id']}: {a['excess_sigma']:.1f} sigma" for a in alerts]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bucket_ids, y=rolling_loft,
        mode="lines",
        name="Rolling LOFT",
        line=dict(color="#FF6B35", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=alert_ids, y=alert_vals,
        mode="markers",
        name="ALERT",
        marker=dict(size=14, color="#E81828", symbol="triangle-up",
                    line=dict(width=2, color="white")),
        text=alert_text,
        hoverinfo="text+y",
    ))

    threshold = baseline["mean_loft"] + ALERT_SIGMA * baseline["std_loft"]
    fig.add_hline(y=threshold, line_dash="dash", line_color="#E81828",
                  annotation_text="Threshold")

    fig.update_layout(
        title="Alert Timeline",
        xaxis_title="Bucket",
        yaxis_title="Rolling LOFT",
        template="plotly_dark",
        height=350,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Alert details table
    alert_df = pd.DataFrame(alerts)
    display_cols = ["bucket_id", "rolling_loft", "threshold", "excess_sigma",
                    "buy_count", "sell_count"]
    display_df = alert_df[[c for c in display_cols if c in alert_df.columns]].copy()
    display_df = display_df.rename(columns={
        "bucket_id": "Bucket",
        "rolling_loft": "Rolling LOFT",
        "threshold": "Threshold",
        "excess_sigma": "Excess Sigma",
        "buy_count": "Buys",
        "sell_count": "Sells",
    })
    st.dataframe(display_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Pitcher Profile
# ---------------------------------------------------------------------------


def _render_pitcher_profile(conn, season: int) -> None:
    """Full pitcher LOFT profile for a season."""
    st.subheader("Pitcher Toxicity Profile")

    pitchers = _get_pitcher_list(conn)
    if not pitchers:
        st.info("No pitcher data available.")
        return

    pitcher_names = [
        f"{p.get('full_name', str(p.get('player_id', '?')))}" for p in pitchers
    ]
    selected_idx = st.selectbox(
        "Select pitcher",
        range(len(pitcher_names)),
        format_func=lambda i: pitcher_names[i],
        key="loft_profile_pitcher",
    )
    pitcher_id = int(pitchers[selected_idx].get(
        "player_id", pitchers[selected_idx].get("pitcher_id", 0)
    ))

    if pitcher_id == 0:
        return

    with st.spinner("Computing season LOFT profile..."):
        profile = calculate_loft(conn, pitcher_id, season)

    baseline = profile.get("baseline", {})
    games = profile.get("games", [])

    if not games:
        st.info("No qualifying game appearances for this pitcher/season.")
        return

    # ── Summary metrics ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg LOFT", f"{profile['avg_loft']:.3f}")
    c2.metric("Games Analyzed", f"{len(games)}")
    c3.metric("Games Decoded", f"{profile['games_decoded']}")
    c4.metric("Total Alerts", f"{profile['total_alerts']}")

    # ── Game-by-game LOFT chart ──────────────────────────────────────────
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        games_df = pd.DataFrame(games)
        if not games_df.empty and "game_date" in games_df.columns:
            games_df = games_df.sort_values("game_date")

            fig = go.Figure()

            # LOFT per game
            colors = ["#E81828" if d else "#2ECC71" for d in games_df["decoded"]]
            fig.add_trace(go.Bar(
                x=games_df["game_date"].astype(str),
                y=games_df["mean_loft"],
                marker_color=colors,
                name="Mean LOFT",
                text=[f"Alerts: {a}" for a in games_df["n_alerts"]],
                hoverinfo="text+y",
            ))

            # Baseline line
            if baseline.get("mean_game_loft", 0) > 0:
                fig.add_hline(
                    y=baseline["mean_game_loft"],
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Season Avg",
                )

            fig.update_layout(
                title="Game-by-Game LOFT (red = decoded)",
                xaxis_title="Game Date",
                yaxis_title="Mean LOFT",
                template="plotly_dark",
                height=400,
                showlegend=False,
            )

            st.plotly_chart(fig, use_container_width=True)

    except ImportError:
        pass

    # ── Games table ──────────────────────────────────────────────────────
    if games:
        gdf = pd.DataFrame(games)
        display_cols = ["game_date", "game_pk", "total_pitches", "buy_total",
                        "sell_total", "mean_loft", "max_loft", "n_alerts", "decoded"]
        display_df = gdf[[c for c in display_cols if c in gdf.columns]]
        st.dataframe(display_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def _render_leaderboard(conn, season: int, min_games: int) -> None:
    """Leaderboards: most decodable and most resilient pitchers."""
    st.subheader("LOFT Leaderboards")

    # Try cache first
    df = None
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "loft", season)
            if cached is not None:
                df = cached
                age_info = cache_age_display(conn, "loft", season)
                if age_info:
                    st.caption(age_info)
        except Exception:
            pass

    if df is None:
        with st.spinner("Computing... Run `python scripts/precompute.py` for instant loading."):
            try:
                df = batch_game_analysis(conn, season=season, min_games=min_games)
            except Exception as exc:
                st.error(f"Error computing leaderboard: {exc}")
                return

    if df.empty:
        st.info("No qualifying pitchers found with current filters.")
        return

    # Aggregate to pitcher-level
    pitcher_agg = df.groupby("pitcher_id").agg(
        n_games=("game_pk", "nunique"),
        avg_loft=("mean_loft", "mean"),
        max_loft=("max_loft", "max"),
        total_alerts=("n_alerts", "sum"),
        decoded_games=("decoded", "sum"),
    ).reset_index()

    pitcher_agg["decoded_pct"] = (
        pitcher_agg["decoded_games"] / pitcher_agg["n_games"] * 100
    ).round(1)

    # Try to join names
    try:
        names_df = conn.execute(
            "SELECT player_id AS pitcher_id, full_name FROM players"
        ).fetchdf()
        pitcher_agg = pitcher_agg.merge(names_df, on="pitcher_id", how="left")
        pitcher_agg["full_name"] = pitcher_agg["full_name"].fillna(
            pitcher_agg["pitcher_id"].astype(str)
        )
    except Exception:
        pitcher_agg["full_name"] = pitcher_agg["pitcher_id"].astype(str)

    # ── Most decodable ───────────────────────────────────────────────────
    st.markdown("#### Most Decodable Pitchers (Highest Avg LOFT)")
    decodable = pitcher_agg.sort_values("avg_loft", ascending=False).head(15).copy()
    decodable.index = range(1, len(decodable) + 1)
    decodable.index.name = "Rank"
    display_cols_dec = ["full_name", "avg_loft", "max_loft", "n_games",
                        "total_alerts", "decoded_games", "decoded_pct"]
    dec_display = decodable[[c for c in display_cols_dec if c in decodable.columns]].rename(columns={
        "full_name": "Pitcher",
        "avg_loft": "Avg LOFT",
        "max_loft": "Max LOFT",
        "n_games": "Games",
        "total_alerts": "Alerts",
        "decoded_games": "Decoded",
        "decoded_pct": "Decoded %",
    })
    st.dataframe(dec_display, use_container_width=True)

    # ── Most resilient ───────────────────────────────────────────────────
    st.markdown("#### Most Resilient Pitchers (Lowest Avg LOFT)")
    resilient = pitcher_agg.sort_values("avg_loft", ascending=True).head(15).copy()
    resilient.index = range(1, len(resilient) + 1)
    resilient.index.name = "Rank"
    res_display = resilient[[c for c in display_cols_dec if c in resilient.columns]].rename(columns={
        "full_name": "Pitcher",
        "avg_loft": "Avg LOFT",
        "max_loft": "Max LOFT",
        "n_games": "Games",
        "total_alerts": "Alerts",
        "decoded_games": "Decoded",
        "decoded_pct": "Decoded %",
    })
    st.dataframe(res_display, use_container_width=True)

    # ── Distribution chart ───────────────────────────────────────────────
    if len(pitcher_agg) > 2:
        try:
            import plotly.express as px

            fig = px.histogram(
                pitcher_agg,
                x="avg_loft",
                nbins=25,
                title="Distribution of Average LOFT Scores",
                labels={"avg_loft": "Average LOFT"},
                template="plotly_dark",
                color_discrete_sequence=["#FF6B35"],
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_pitcher_list(conn) -> list[dict]:
    """Get list of pitchers for dropdown selectors."""
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
            pitchers = []
    return pitchers
