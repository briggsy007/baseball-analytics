"""
ChemNet Lineup Synergy dashboard view.

Visualises the GNN-based lineup synergy model: team/game synergy scores,
protection-network graphs with attention weights, pairwise interaction
heatmaps, lineup optimiser, and a team leaderboard.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.db_helper import get_db_connection, has_data

# ── Graceful imports ──────────────────────────────────────────────────────────

_CHEMNET_AVAILABLE = False
try:
    from src.analytics.chemnet import (
        ChemNetModel,
        batch_calculate,
        build_game_graph,
        calculate_synergy,
        get_protection_coefficients,
        optimize_lineup_order,
        _get_rolling_player_features,
        _build_adjacency,
        _pad_graph,
        NUM_LINEUP_SLOTS,
    )
    _CHEMNET_AVAILABLE = True
except ImportError:
    pass

_CACHE_AVAILABLE = False
try:
    from src.dashboard.cache_reader import get_cached_leaderboard, cache_age_display
    _CACHE_AVAILABLE = True
except Exception:
    pass

# ── Colour palette ────────────────────────────────────────────────────────────

_SYNERGY_POSITIVE = "#00C853"
_SYNERGY_NEGATIVE = "#FF1744"
_SYNERGY_NEUTRAL = "#FFD600"
_EDGE_COLOR = "#2196F3"
_NODE_COLOR = "#FF9100"
_BG_DARK = "#0E1117"


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


# ── Page entry point ─────────────────────────────────────────────────────────


def render() -> None:
    """Render the ChemNet Lineup Synergy page."""
    st.title("ChemNet: Lineup Synergy & Protection Effects")
    st.caption(
        "Graph Neural Network that measures batting-order synergy. "
        "Attention weights reveal protection effects between consecutive hitters."
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**ChemNet measures lineup synergy** — the interaction effects between hitters that make a lineup more (or less) than the sum of its parts.

- **Positive synergy** = hitters in adjacent lineup spots complement each other (protection effects, platoon cascades)
- **Negative synergy** = adjacent hitters are redundant (similar weaknesses that pitchers exploit in sequence)
- **Protection coefficients** quantify the specific "lineup protection" effect — does having Hitter B on deck change how Hitter A gets pitched?
- **Lineup optimizer** finds the batting order that maximizes synergy, not just individual talent
- **Impact:** Optimal lineup ordering based on interaction effects is worth 10-20 runs per season compared to a talent-only ordering — roughly 1-2 wins
""")

    conn = get_db_connection()

    if not _CHEMNET_AVAILABLE:
        st.error(
            "The `chemnet` analytics module could not be imported. "
            "Ensure PyTorch is installed:\n\n"
            "```\npip install torch --index-url https://download.pytorch.org/whl/cpu\n```"
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
        st.markdown("### ChemNet Options")

        seasons = _cached_seasons()

        season = st.selectbox(
            "Season",
            options=seasons if seasons else [2024],
            index=0,
            key="chemnet_season",
        )

        teams = _cached_teams()

        team = st.selectbox(
            "Team",
            options=[None] + teams,
            format_func=lambda x: "All Teams" if x is None else x,
            key="chemnet_team",
        )

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_score, tab_network, tab_heatmap, tab_optimizer, tab_leaders = st.tabs([
        "Synergy Score",
        "Protection Network",
        "Pairwise Interactions",
        "Lineup Optimizer",
        "Team Leaderboard",
    ])

    with tab_score:
        _render_synergy_score(conn, season, team)

    with tab_network:
        _render_protection_network(conn, season, team)

    with tab_heatmap:
        _render_pairwise_heatmap(conn, season, team)

    with tab_optimizer:
        _render_lineup_optimizer(conn, season, team)

    with tab_leaders:
        _render_team_leaderboard(conn, season)


# ── Synergy Score ────────────────────────────────────────────────────────────


@st.cache_data(ttl=3600)
def _get_game_options(season: int, team: str | None) -> list[dict]:
    """Get available games for the selectors."""
    conn = get_db_connection()
    params: list = [int(season)]
    if team:
        team_filter = "AND (home_team = $2 OR away_team = $2)"
        params.append(team)
    else:
        team_filter = ""

    query = f"""
        SELECT DISTINCT game_pk, game_date, home_team, away_team
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) = $1
          {team_filter}
        ORDER BY game_date DESC
        LIMIT 200
    """
    try:
        df = conn.execute(query, params).fetchdf()
        return df.to_dict("records")
    except Exception:
        return []


def _render_synergy_score(conn, season, team) -> None:
    """Display the synergy score for a selected game."""
    st.subheader("Lineup Synergy Score")

    games = _get_game_options(season, team)
    if not games:
        st.info("No games found for the selected season/team.")
        return

    game_labels = {
        f"{g['game_date']} | {g['away_team']} @ {g['home_team']} (pk={g['game_pk']})": g["game_pk"]
        for g in games
    }

    selected_label = st.selectbox(
        "Select game",
        options=list(game_labels.keys()),
        key="chemnet_game_select",
    )
    if not selected_label:
        return

    game_pk = game_labels[selected_label]
    side = st.radio("Team side", ["home", "away"], horizontal=True, key="chemnet_side")

    try:
        result = calculate_synergy(conn, game_pk, side)
    except FileNotFoundError:
        st.warning(
            "ChemNet model has not been trained yet. "
            "Run `ChemNetModel().train(conn)` first."
        )
        return
    except Exception as exc:
        st.error(f"Error computing synergy: {exc}")
        return

    if result["synergy_score"] is None:
        st.info("Could not compute synergy for this game/side (insufficient lineup data).")
        return

    synergy = result["synergy_score"]
    color = _SYNERGY_POSITIVE if synergy > 0 else (_SYNERGY_NEGATIVE if synergy < 0 else _SYNERGY_NEUTRAL)

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Synergy Score",
        f"{synergy:+.4f}",
        delta=f"{'Positive' if synergy > 0 else 'Negative'} synergy",
        delta_color="normal" if synergy > 0 else "inverse",
    )
    col2.metric("GNN Prediction", f"{result['gnn_pred']:.4f}")
    col3.metric("Baseline Prediction", f"{result['baseline_pred']:.4f}")

    st.markdown(
        f"**Interpretation:** A synergy score of **{synergy:+.4f}** means the "
        f"batting order {'adds' if synergy > 0 else 'subtracts'} value "
        f"beyond what individual player quality alone would predict."
    )

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=synergy,
        title={"text": "Lineup Synergy"},
        delta={"reference": 0},
        gauge={
            "axis": {"range": [-2, 2]},
            "bar": {"color": color},
            "steps": [
                {"range": [-2, -0.5], "color": "rgba(255, 23, 68, 0.2)"},
                {"range": [-0.5, 0.5], "color": "rgba(255, 214, 0, 0.2)"},
                {"range": [0.5, 2], "color": "rgba(0, 200, 83, 0.2)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.75,
                "value": synergy,
            },
        },
    ))
    fig.update_layout(template="plotly_dark", height=300)
    st.plotly_chart(fig, use_container_width=True, key="synergy_gauge")


# ── Protection Network ───────────────────────────────────────────────────────


def _render_protection_network(conn, season, team) -> None:
    """Visualise the lineup as a graph with attention-weighted edges."""
    st.subheader("Protection Network")

    games = _get_game_options(season, team)
    if not games:
        st.info("No games found.")
        return

    game_labels = {
        f"{g['game_date']} | {g['away_team']} @ {g['home_team']}": g["game_pk"]
        for g in games
    }
    selected_label = st.selectbox(
        "Select game", list(game_labels.keys()), key="chemnet_network_game"
    )
    if not selected_label:
        return

    game_pk = game_labels[selected_label]
    side = st.radio("Side", ["home", "away"], horizontal=True, key="chemnet_network_side")

    try:
        coeff = get_protection_coefficients(conn, game_pk, side)
    except FileNotFoundError:
        st.warning("ChemNet model not trained. Run training first.")
        return
    except Exception as exc:
        st.error(f"Error: {exc}")
        return

    pairs = coeff.get("pairs", [])
    if not pairs:
        st.info("No protection data available for this game.")
        return

    # Get player names
    batter_ids = list({p["batter_from"] for p in pairs} | {p["batter_to"] for p in pairs})
    try:
        names_df = conn.execute(
            f"SELECT player_id, full_name FROM players "
            f"WHERE player_id IN ({','.join(str(b) for b in batter_ids)})"
        ).fetchdf()
        name_map = dict(zip(names_df["player_id"], names_df["full_name"]))
    except Exception:
        name_map = {}

    # Build graph visualisation using plotly
    n = len(pairs) + 1  # number of nodes
    # Arrange nodes in a horizontal line
    x_coords = list(range(n))
    y_coords = [0] * n

    # Node labels
    node_labels = []
    all_batters = []
    for p in pairs:
        if p["batter_from"] not in all_batters:
            all_batters.append(p["batter_from"])
    all_batters.append(pairs[-1]["batter_to"])

    for i, bid in enumerate(all_batters):
        name = name_map.get(bid, str(bid))
        if isinstance(name, str) and len(name) > 15:
            name = name[:13] + ".."
        node_labels.append(f"#{i+1} {name}")

    fig = go.Figure()

    # Edges with thickness proportional to attention
    max_att = max(p["avg_attention"] for p in pairs) if pairs else 1.0
    for p in pairs:
        i = p["slot_from"] - 1
        j = p["slot_to"] - 1
        width = max(1, (p["avg_attention"] / max(max_att, 1e-6)) * 10)
        fig.add_trace(go.Scatter(
            x=[x_coords[i], x_coords[j]],
            y=[y_coords[i], y_coords[j]],
            mode="lines",
            line=dict(color=_EDGE_COLOR, width=width),
            hoverinfo="text",
            text=f"Attention: {p['avg_attention']:.4f}",
            showlegend=False,
        ))

    # Nodes
    fig.add_trace(go.Scatter(
        x=x_coords[:n],
        y=y_coords[:n],
        mode="markers+text",
        marker=dict(size=30, color=_NODE_COLOR, line=dict(width=2, color="white")),
        text=node_labels,
        textposition="top center",
        textfont=dict(size=10),
        hoverinfo="text",
        showlegend=False,
    ))

    fig.update_layout(
        template="plotly_dark",
        height=350,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1.5]),
        margin=dict(l=30, r=30, t=30, b=30),
    )
    st.plotly_chart(fig, use_container_width=True, key="protection_network")

    # Edge table
    st.markdown("**Pairwise Protection Weights**")
    edge_rows = []
    for p in pairs:
        name_from = name_map.get(p["batter_from"], str(p["batter_from"]))
        name_to = name_map.get(p["batter_to"], str(p["batter_to"]))
        edge_rows.append({
            "Slot": f"#{p['slot_from']} -> #{p['slot_to']}",
            "From": name_from,
            "To": name_to,
            "Attention": f"{p['avg_attention']:.4f}",
        })
    st.dataframe(pd.DataFrame(edge_rows), hide_index=True, use_container_width=True)


# ── Pairwise Interactions Heatmap ────────────────────────────────────────────


def _render_pairwise_heatmap(conn, season, team) -> None:
    """Display a heatmap of attention between batting-order slot pairs."""
    st.subheader("Pairwise Interaction Heatmap")

    games = _get_game_options(season, team)
    if not games:
        st.info("No games found.")
        return

    game_labels = {
        f"{g['game_date']} | {g['away_team']} @ {g['home_team']}": g["game_pk"]
        for g in games
    }
    selected_label = st.selectbox(
        "Select game", list(game_labels.keys()), key="chemnet_heatmap_game"
    )
    if not selected_label:
        return

    game_pk = game_labels[selected_label]
    side = st.radio("Side", ["home", "away"], horizontal=True, key="chemnet_heatmap_side")

    try:
        coeff = get_protection_coefficients(conn, game_pk, side)
    except FileNotFoundError:
        st.warning("ChemNet model not trained.")
        return
    except Exception as exc:
        st.error(f"Error: {exc}")
        return

    att_matrix = coeff.get("attention_matrix", [])
    if not att_matrix:
        st.info("No attention data available.")
        return

    att_np = np.array(att_matrix)
    n = min(att_np.shape[0], NUM_LINEUP_SLOTS)
    att_np = att_np[:n, :n]
    labels = [f"Slot {i+1}" for i in range(n)]

    fig = go.Figure(data=go.Heatmap(
        z=att_np,
        x=labels,
        y=labels,
        colorscale="Blues",
        zmin=0,
        text=np.round(att_np, 3),
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="<b>%{y}</b> -> <b>%{x}</b><br>Attention: %{z:.4f}<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_dark",
        height=500,
        margin=dict(l=80, r=30, t=30, b=80),
        xaxis=dict(title="To Slot"),
        yaxis=dict(title="From Slot", autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True, key="att_heatmap")

    st.markdown(
        "**Reading the heatmap:** Brighter cells indicate stronger "
        "attention from one batting-order slot to another. Strong "
        "adjacent-cell values indicate meaningful protection effects."
    )


# ── Lineup Optimizer ─────────────────────────────────────────────────────────


def _render_lineup_optimizer(conn, season, team) -> None:
    """Select players and find the highest-synergy batting order."""
    st.subheader("Lineup Optimizer")

    # Get available batters
    team_filter = ""
    if team:
        team_filter = f"""
            AND (
                (inning_topbot = 'Top' AND away_team = '{team}')
                OR (inning_topbot = 'Bot' AND home_team = '{team}')
            )
        """

    try:
        batters_df = conn.execute(f"""
            SELECT
                batter_id,
                COUNT(DISTINCT game_pk) AS games
            FROM pitches
            WHERE EXTRACT(YEAR FROM game_date) = {int(season)}
              {team_filter}
            GROUP BY batter_id
            HAVING COUNT(DISTINCT game_pk) >= 10
            ORDER BY games DESC
            LIMIT 40
        """).fetchdf()
    except Exception as exc:
        st.error(f"Error loading batters: {exc}")
        return

    if batters_df.empty:
        st.info("No qualifying batters found.")
        return

    # Get names
    batter_ids = batters_df["batter_id"].tolist()
    try:
        names_df = conn.execute(
            f"SELECT player_id, full_name FROM players "
            f"WHERE player_id IN ({','.join(str(b) for b in batter_ids)})"
        ).fetchdf()
        name_map = dict(zip(names_df["player_id"], names_df["full_name"]))
    except Exception:
        name_map = {}

    options = {}
    for _, row in batters_df.iterrows():
        bid = int(row["batter_id"])
        name = name_map.get(bid, str(bid))
        label = f"{name} ({bid}) - {int(row['games'])}G"
        options[label] = bid

    selected_labels = st.multiselect(
        "Select players for lineup (up to 9)",
        options=list(options.keys()),
        max_selections=9,
        key="chemnet_optimizer_select",
    )

    if len(selected_labels) < 2:
        st.info("Select at least 2 players to optimise batting order.")
        return

    selected_ids = [options[lbl] for lbl in selected_labels]

    if st.button("Optimize Batting Order", key="chemnet_optimize_btn"):
        with st.spinner("Computing optimal batting order..."):
            try:
                # Get features
                features = _get_rolling_player_features(
                    conn, selected_ids, f"{season}-12-31"
                )
                result = optimize_lineup_order(selected_ids, features)
            except FileNotFoundError:
                st.warning("ChemNet model not trained.")
                return
            except Exception as exc:
                st.error(f"Error: {exc}")
                return

        col1, col2, col3 = st.columns(3)
        col1.metric("Optimal GNN Score", f"{result['gnn_score']:.4f}")
        col2.metric("Initial Score", f"{result['initial_score']:.4f}")
        improvement = result["improvement"]
        col3.metric(
            "Improvement",
            f"{improvement:+.4f}",
            delta_color="normal" if improvement > 0 else "inverse",
        )

        st.markdown("**Optimal Batting Order**")
        order_rows = []
        for i, bid in enumerate(result["optimal_order"]):
            name = name_map.get(bid, str(bid))
            order_rows.append({"Slot": i + 1, "Player": name, "ID": bid})
        st.dataframe(pd.DataFrame(order_rows), hide_index=True, use_container_width=True)


# ── Team Leaderboard ─────────────────────────────────────────────────────────


def _render_team_leaderboard(conn, season) -> None:
    """Display average synergy by team for the season."""
    st.subheader("Team Synergy Leaderboard")

    # Try cache first
    df = None
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "chemnet", season)
            if cached is not None:
                df = cached
                age_info = cache_age_display(conn, "chemnet", season)
                if age_info:
                    st.caption(age_info)
        except Exception:
            pass

    if df is None:
        try:
            with st.spinner("Computing... Run `python scripts/precompute.py` for instant loading."):
                df = batch_calculate(conn, season, max_games=200)
        except FileNotFoundError:
            st.warning("ChemNet model not trained. Run training first.")
            return
        except Exception as exc:
            st.error(f"Error computing leaderboard: {exc}")
            return

    if df.empty:
        st.info("No synergy data available for this season.")
        return

    # Display table
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"

    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "team": st.column_config.TextColumn("Team"),
            "avg_synergy": st.column_config.NumberColumn("Avg Synergy", format="%.4f"),
            "games": st.column_config.NumberColumn("Games"),
        },
    )

    # Bar chart
    colors = [
        _SYNERGY_POSITIVE if v > 0 else _SYNERGY_NEGATIVE
        for v in df["avg_synergy"]
    ]

    fig = go.Figure(data=go.Bar(
        x=df["team"],
        y=df["avg_synergy"],
        marker_color=colors,
        text=df["avg_synergy"].apply(lambda v: f"{v:.4f}"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Avg Synergy: %{y:.4f}<br>Games: %{customdata}<extra></extra>",
        customdata=df["games"],
    ))
    fig.update_layout(
        xaxis=dict(title="Team"),
        yaxis=dict(title="Average Synergy Score"),
        template="plotly_dark",
        height=450,
        margin=dict(l=60, r=30, t=30, b=60),
    )
    fig.add_hline(y=0, line=dict(color="white", width=1, dash="dash"))
    st.plotly_chart(fig, use_container_width=True, key="team_synergy_bar")

    # Top/bottom callouts
    top = df.iloc[0]
    st.success(
        f"**Highest synergy:** {top['team']} "
        f"(avg {top['avg_synergy']:.4f} over {int(top['games'])} games)"
    )
    if len(df) > 1:
        bottom = df.iloc[-1]
        st.error(
            f"**Lowest synergy:** {bottom['team']} "
            f"(avg {bottom['avg_synergy']:.4f} over {int(bottom['games'])} games)"
        )
