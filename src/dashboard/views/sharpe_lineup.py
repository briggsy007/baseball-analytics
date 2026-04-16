"""
Player Sharpe Ratio & Efficient Lineup Frontier dashboard view.

Displays risk-adjusted offensive consistency metrics (PSR), pairwise wOBA
correlation heatmaps, the Markowitz efficient frontier for lineup
construction, and an interactive lineup builder.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.db_helper import get_db_connection, has_data

# ── Graceful imports ──────────────────────────────────────────────────────────

_SHARPE_AVAILABLE = False
try:
    from src.analytics.sharpe_lineup import (
        LEAGUE_AVG_WOBA,
        batch_player_sharpe,
        calculate_player_sharpe,
        compute_correlation_matrix,
        efficient_frontier,
        get_sharpe_leaderboard,
        optimize_lineup,
    )
    _SHARPE_AVAILABLE = True
except ImportError:
    pass

_CACHE_AVAILABLE = False
try:
    from src.dashboard.cache_reader import get_cached_leaderboard, cache_age_display
    _CACHE_AVAILABLE = True
except Exception:
    pass

# ── Colour palette ────────────────────────────────────────────────────────────

_PSR_COLOR_SCALE = [
    [0.0, "#FF1744"],    # poor
    [0.25, "#FF9100"],   # below avg
    [0.5, "#FFD600"],    # average
    [0.75, "#64DD17"],   # above avg
    [1.0, "#00C853"],    # elite
]


# ── Cached data helpers ──────────────────────────────────────────────────────


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
def _cached_teams() -> list[str]:
    """Cached list of teams."""
    conn = get_db_connection()
    try:
        return conn.execute(
            "SELECT DISTINCT home_team FROM pitches "
            "WHERE home_team IS NOT NULL ORDER BY home_team"
        ).fetchdf()["home_team"].tolist()
    except Exception:
        return []


@st.cache_data(ttl=3600)
def _cached_batch_player_sharpe(
    team_id: str | None, season: int | None, min_games: int,
) -> pd.DataFrame:
    """Cached batch player sharpe computation."""
    conn = get_db_connection()
    return batch_player_sharpe(conn, team_id=team_id, season=season, min_games=min_games)


@st.cache_data(ttl=3600)
def _cached_correlation_matrix(
    player_ids: tuple, season: int | None,
) -> pd.DataFrame:
    """Cached correlation matrix computation."""
    conn = get_db_connection()
    return compute_correlation_matrix(conn, list(player_ids), season)


# ── Page entry point ─────────────────────────────────────────────────────────


def render() -> None:
    """Render the Player Sharpe Ratio & Efficient Frontier page."""
    st.title("Player Sharpe Ratio & Efficient Lineup Frontier")
    st.caption(
        "Markowitz portfolio theory applied to baseball lineups. "
        "Measure each hitter's risk-adjusted consistency and build "
        "statistically optimal lineup combinations."
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**Player Sharpe Ratio (PSR)** measures consistency, not just talent. Borrowed from finance: high return with low volatility = high Sharpe.

- **PSR > 1.5** = elite consistency (rare — top ~5% of hitters)
- **PSR 0.5–1.5** = solid contributor with acceptable variance
- **PSR < 0.5** = boom-or-bust hitter — big games mixed with invisible ones
- **Correlation Matrix** shows which hitters' games are linked — negative correlation = good diversification (when one slumps, the other produces)
- **Efficient Frontier** finds lineups that maximize expected production for a given risk tolerance — the "optimal portfolio" of hitters
- **Impact:** Lineup construction that accounts for correlations can add 10-20 runs over a season vs. just sorting by OPS
""")

    conn = get_db_connection()

    if not _SHARPE_AVAILABLE:
        st.error(
            "The `sharpe_lineup` analytics module could not be imported. "
            "Ensure `scipy` is installed:\n\n"
            "```\npip install scipy\n```"
        )
        return

    if conn is None or not has_data(conn):
        st.warning(
            "No pitch data available. Run the data backfill pipeline first "
            "(`python scripts/backfill.py`)."
        )
        return

    # ── Sidebar controls ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Sharpe / Frontier Options")

        # Season selector
        seasons = _cached_seasons()

        season = st.selectbox(
            "Season",
            options=[None] + seasons,
            format_func=lambda x: "All Seasons" if x is None else str(x),
            key="sharpe_season",
        )

        # Team selector
        teams = _cached_teams()

        team = st.selectbox(
            "Team",
            options=[None] + teams,
            format_func=lambda x: "All Teams" if x is None else x,
            key="sharpe_team",
        )

        min_games = st.slider(
            "Minimum games to qualify",
            min_value=5, max_value=100, value=40, step=5,
            key="sharpe_min_games",
        )

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_board, tab_corr, tab_frontier, tab_builder = st.tabs([
        "PSR Leaderboard",
        "Correlation Heatmap",
        "Efficient Frontier",
        "Lineup Builder",
    ])

    with tab_board:
        _render_leaderboard(conn, season, team, min_games)

    with tab_corr:
        _render_correlation_heatmap(conn, season, team, min_games)

    with tab_frontier:
        _render_efficient_frontier(conn, season, team, min_games)

    with tab_builder:
        _render_lineup_builder(conn, season, team, min_games)


# ── PSR Leaderboard ──────────────────────────────────────────────────────────


def _render_leaderboard(conn, season, team, min_games) -> None:
    """Display the PSR leaderboard with consistency indicators."""
    st.subheader("Player Sharpe Ratio Leaderboard")

    # Try cache first
    df = None
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "sharpe_lineup", season)
            if cached is not None:
                df = cached
                age_info = cache_age_display(conn, "sharpe_lineup", season)
                if age_info:
                    st.caption(age_info)
        except Exception:
            pass

    if df is None:
        try:
            with st.spinner("Computing... Run `python scripts/precompute.py` for instant loading."):
                df = _cached_batch_player_sharpe(team_id=team, season=season, min_games=min_games)
        except Exception as exc:
            st.error(f"Error computing leaderboard: {exc}")
            return

    if df.empty:
        st.info(
            "No qualifying batters found with the current filters. "
            "Try lowering the minimum games threshold."
        )
        return

    # Add rank
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"

    # Display the table
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "psr": st.column_config.NumberColumn("PSR", format="%.3f"),
            "mean_woba": st.column_config.NumberColumn("Mean wOBA", format="%.4f"),
            "std_woba": st.column_config.NumberColumn("Std wOBA", format="%.4f"),
            "games": st.column_config.NumberColumn("Games"),
            "name": st.column_config.TextColumn("Player"),
        },
    )

    # ── Key insight callouts ─────────────────────────────────────────────
    valid_df = df.dropna(subset=["psr"])
    if not valid_df.empty:
        best = valid_df.iloc[0]
        best_name = best.get("name") or f"ID {best['batter_id']}"
        st.success(
            f"**Most consistent hitter:** {best_name} "
            f"(PSR {best['psr']:.3f}, {best['mean_woba']:.3f} mean wOBA, "
            f"{best['std_woba']:.3f} std)"
        )

    # PSR distribution chart
    if len(valid_df) >= 5:
        st.markdown("**PSR Distribution**")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=valid_df["psr"],
            nbinsx=20,
            marker_color="#2196F3",
            opacity=0.75,
        ))
        fig.add_vline(
            x=0, line=dict(color="white", width=2, dash="dash"),
            annotation_text="League Average",
            annotation_position="top right",
        )
        fig.update_layout(
            xaxis=dict(title="Player Sharpe Ratio"),
            yaxis=dict(title="Count"),
            template="plotly_dark",
            height=350,
            margin=dict(l=50, r=30, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True, key="psr_dist")


# ── Correlation heatmap ───────────────────────────────────────────────────────


def _render_correlation_heatmap(conn, season, team, min_games) -> None:
    """Display a pairwise wOBA correlation heatmap."""
    st.subheader("Game-Level wOBA Correlation Heatmap")

    try:
        stats = _cached_batch_player_sharpe(team_id=team, season=season, min_games=min_games)
    except Exception as exc:
        st.error(f"Error: {exc}")
        return

    if len(stats) < 2:
        st.info("Need at least 2 qualifying players to compute correlations.")
        return

    # Limit to top 15 players for readability
    display_stats = stats.head(15)
    player_ids = display_stats["batter_id"].tolist()

    try:
        corr = _cached_correlation_matrix(tuple(player_ids), season)
    except Exception as exc:
        st.error(f"Error computing correlation matrix: {exc}")
        return

    # Build labels
    labels = []
    for pid in player_ids:
        row = display_stats[display_stats["batter_id"] == pid]
        if not row.empty and pd.notna(row.iloc[0].get("name")):
            labels.append(str(row.iloc[0]["name"]))
        else:
            labels.append(str(pid))

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=labels,
        y=labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=max(400, len(labels) * 40),
        margin=dict(l=120, r=30, t=30, b=120),
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")

    # ── Best diversifier insight ─────────────────────────────────────────
    corr_values = corr.values.copy()
    np.fill_diagonal(corr_values, np.nan)
    mean_corr_per_player = np.nanmean(corr_values, axis=1)

    if len(mean_corr_per_player) > 0:
        best_div_idx = int(np.nanargmin(mean_corr_per_player))
        best_div_name = labels[best_div_idx]
        best_div_corr = mean_corr_per_player[best_div_idx]
        st.info(
            f"**Best diversifier:** {best_div_name} "
            f"(avg correlation with teammates: {best_div_corr:.3f}). "
            f"Adding this hitter reduces portfolio variance the most."
        )


# ── Efficient frontier ────────────────────────────────────────────────────────


def _render_efficient_frontier(conn, season, team, min_games) -> None:
    """Display the Markowitz efficient frontier chart."""
    st.subheader("Efficient Lineup Frontier")

    try:
        stats = _cached_batch_player_sharpe(team_id=team, season=season, min_games=min_games)
    except Exception as exc:
        st.error(f"Error: {exc}")
        return

    if len(stats) < 2:
        st.info("Need at least 2 qualifying players to build an efficient frontier.")
        return

    # Use top N players
    n_pool = st.slider(
        "Player pool size", min_value=3, max_value=min(20, len(stats)),
        value=min(9, len(stats)), step=1, key="frontier_pool",
    )
    pool = stats.head(n_pool).dropna(subset=["mean_woba", "std_woba"])
    pool = pool[pool["std_woba"] > 0]

    if len(pool) < 2:
        st.info("Not enough players with valid stats.")
        return

    player_ids = pool["batter_id"].tolist()

    try:
        corr = _cached_correlation_matrix(tuple(player_ids), season)
        points = efficient_frontier(pool, corr, n_points=30, n_players=min(9, len(pool)))
    except Exception as exc:
        st.error(f"Error computing frontier: {exc}")
        return

    if not points:
        st.info("Could not compute efficient frontier points.")
        return

    # Build scatter
    variances = [p["variance"] for p in points]
    stds = [p["std"] for p in points]
    returns = [p["expected_woba"] for p in points]

    fig = go.Figure()

    # Frontier line
    fig.add_trace(go.Scatter(
        x=stds,
        y=returns,
        mode="lines+markers",
        marker=dict(size=6, color="#2196F3"),
        line=dict(color="#2196F3", width=2),
        name="Efficient Frontier",
        hovertemplate=(
            "Std: %{x:.4f}<br>Expected wOBA: %{y:.4f}<extra></extra>"
        ),
    ))

    # Individual players
    fig.add_trace(go.Scatter(
        x=pool["std_woba"].tolist(),
        y=pool["mean_woba"].tolist(),
        mode="markers+text",
        marker=dict(size=10, color="#FF9100", symbol="diamond"),
        text=[
            str(row.get("name", row["batter_id"]))[:12]
            for _, row in pool.iterrows()
        ],
        textposition="top center",
        textfont=dict(size=9),
        name="Individual Players",
        hovertemplate=(
            "<b>%{text}</b><br>Std: %{x:.4f}<br>Mean wOBA: %{y:.4f}<extra></extra>"
        ),
    ))

    # Highlight optimal (max Sharpe) portfolio
    sharpe_vals = []
    for p in points:
        if p["std"] > 0:
            sharpe_vals.append((p["expected_woba"] - LEAGUE_AVG_WOBA) / p["std"])
        else:
            sharpe_vals.append(0)

    if sharpe_vals:
        opt_idx = int(np.argmax(sharpe_vals))
        fig.add_trace(go.Scatter(
            x=[stds[opt_idx]],
            y=[returns[opt_idx]],
            mode="markers",
            marker=dict(size=16, color="#00C853", symbol="star", line=dict(width=2, color="white")),
            name="Optimal Portfolio",
            hovertemplate=(
                "OPTIMAL<br>Std: %{x:.4f}<br>Expected wOBA: %{y:.4f}<extra></extra>"
            ),
        ))

    fig.update_layout(
        xaxis=dict(title="Portfolio Std Dev (Risk)"),
        yaxis=dict(title="Expected wOBA (Return)"),
        template="plotly_dark",
        height=500,
        margin=dict(l=60, r=30, t=30, b=60),
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
    )
    st.plotly_chart(fig, use_container_width=True, key="frontier_chart")

    # Display optimal lineup composition
    if sharpe_vals:
        opt_weights = points[opt_idx]["weights"]
        selected = {k: v for k, v in opt_weights.items() if v > 0.01}
        if selected:
            st.markdown("**Optimal Portfolio Composition**")
            rows = []
            for pid, weight in sorted(selected.items(), key=lambda x: -x[1]):
                prow = pool[pool["batter_id"] == pid]
                name = str(prow.iloc[0].get("name", pid)) if not prow.empty else str(pid)
                rows.append({
                    "Player": name,
                    "Weight": f"{weight:.1%}",
                    "Mean wOBA": f"{prow.iloc[0]['mean_woba']:.4f}" if not prow.empty else "",
                })
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ── Lineup builder ────────────────────────────────────────────────────────────


def _render_lineup_builder(conn, season, team, min_games) -> None:
    """Interactive lineup builder: select 9 players and see portfolio stats."""
    st.subheader("Lineup Builder")

    try:
        stats = _cached_batch_player_sharpe(
            team_id=team, season=season, min_games=max(1, min_games // 2),
        )
    except Exception as exc:
        st.error(f"Error: {exc}")
        return

    if stats.empty:
        st.info("No qualifying players found.")
        return

    # Build player selection options
    options = {}
    for _, row in stats.iterrows():
        label = f"{row.get('name', '')} ({row['batter_id']})" if pd.notna(row.get("name")) else str(row["batter_id"])
        options[label] = int(row["batter_id"])

    selected_labels = st.multiselect(
        "Select lineup players (up to 9)",
        options=list(options.keys()),
        max_selections=9,
        key="lineup_builder_select",
    )

    if len(selected_labels) < 2:
        st.info("Select at least 2 players to compute portfolio statistics.")
        return

    selected_ids = [options[lbl] for lbl in selected_labels]
    selected_stats = stats[stats["batter_id"].isin(selected_ids)].copy()
    selected_stats = selected_stats.dropna(subset=["mean_woba", "std_woba"])
    selected_stats = selected_stats[selected_stats["std_woba"] > 0]

    if len(selected_stats) < 2:
        st.warning("Not enough selected players have valid stats.")
        return

    # Compute portfolio stats
    try:
        corr = _cached_correlation_matrix(tuple(selected_stats["batter_id"].tolist()), season)
        portfolio = optimize_lineup(
            selected_stats, corr, n_players=len(selected_stats),
        )
    except Exception as exc:
        st.error(f"Error optimising lineup: {exc}")
        return

    # Display results
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected wOBA", f"{portfolio['expected_woba']:.4f}")
    col2.metric("Portfolio Std", f"{portfolio['portfolio_std']:.4f}")
    col3.metric(
        "Portfolio Sharpe",
        f"{portfolio['sharpe']:.3f}" if portfolio["sharpe"] is not None else "N/A",
    )
    col4.metric("Players", len(portfolio["selected_players"]))

    # Show optimal weights
    st.markdown("**Optimal Weight Allocation**")
    weight_rows = []
    for pid, weight in sorted(
        portfolio["weights"].items(), key=lambda x: -x[1]
    ):
        if weight < 0.01:
            continue
        prow = selected_stats[selected_stats["batter_id"] == pid]
        name = str(prow.iloc[0].get("name", pid)) if not prow.empty else str(pid)
        psr = prow.iloc[0].get("psr", None) if not prow.empty else None
        weight_rows.append({
            "Player": name,
            "Weight": f"{weight:.1%}",
            "Mean wOBA": f"{prow.iloc[0]['mean_woba']:.4f}" if not prow.empty else "",
            "PSR": f"{psr:.3f}" if psr is not None else "N/A",
        })

    if weight_rows:
        st.dataframe(pd.DataFrame(weight_rows), hide_index=True, use_container_width=True)

    # Compare to equal-weight portfolio
    n_sel = len(selected_stats)
    equal_weights = np.ones(n_sel) / n_sel
    mu_arr = selected_stats["mean_woba"].values
    sigma_arr = selected_stats["std_woba"].values
    corr_arr = corr.loc[
        selected_stats["batter_id"].tolist(),
        selected_stats["batter_id"].tolist(),
    ].values
    cov_arr = np.outer(sigma_arr, sigma_arr) * corr_arr
    eq_ret = float(equal_weights @ mu_arr)
    eq_var = float(equal_weights @ cov_arr @ equal_weights)
    eq_std = float(np.sqrt(max(eq_var, 0)))

    st.markdown("---")
    st.markdown("**Optimal vs Equal-Weight Portfolio**")
    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.metric("Optimal wOBA", f"{portfolio['expected_woba']:.4f}")
        st.metric("Optimal Std", f"{portfolio['portfolio_std']:.4f}")
    with comp_col2:
        st.metric("Equal-Weight wOBA", f"{eq_ret:.4f}")
        st.metric("Equal-Weight Std", f"{eq_std:.4f}")

    improvement = portfolio["expected_woba"] - eq_ret
    risk_reduction = eq_std - portfolio["portfolio_std"]
    if improvement > 0:
        st.success(
            f"Optimisation improves expected wOBA by {improvement:+.4f} "
            f"and reduces risk by {risk_reduction:+.4f} std dev."
        )
    else:
        st.info(
            f"Equal-weight portfolio is close to optimal "
            f"(difference: {improvement:+.4f} wOBA)."
        )
