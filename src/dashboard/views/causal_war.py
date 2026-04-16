"""
CausalWAR dashboard view -- Causal Inference Player Valuation.

Displays a leaderboard of de-confounded player value (CausalWAR),
comparison scatter plots against traditional WAR, forest plots with
confidence intervals, and "Biggest Movers" analysis.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.db_helper import get_db_connection, has_data

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_CAUSAL_AVAILABLE = False
try:
    from src.analytics.causal_war import (
        CausalWARConfig,
        CausalWARModel,
        train,
        batch_calculate,
        calculate_causal_war,
        get_leaderboard,
    )
    _CAUSAL_AVAILABLE = True
except ImportError:
    pass

_CACHE_AVAILABLE = False
try:
    from src.dashboard.cache_reader import get_cached_leaderboard, cache_age_display
    _CACHE_AVAILABLE = True
except Exception:
    pass

# Mock flag -- set True to render the page with synthetic data when the
# analytics module is unavailable or the model hasn't been trained.
_USE_MOCK = False

# Phillies red / blue palette
_PHILLIES_RED = "#E81828"
_PHILLIES_BLUE = "#002D72"
_PHILLIES_LIGHT = "#B0B7BC"
_POSITIVE_GREEN = "#2ECC71"
_NEGATIVE_RED = "#E74C3C"
_CI_BAND = "rgba(232, 24, 40, 0.25)"


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

def _generate_mock_leaderboard() -> pd.DataFrame:
    """Generate mock CausalWAR data for development / demo mode."""
    rng = np.random.RandomState(42)
    n = 30
    names = [f"Player {chr(65 + i)}" for i in range(n)]
    causal_war = rng.normal(2.0, 2.5, size=n)
    traditional_war = causal_war + rng.normal(0, 1.2, size=n)

    df = pd.DataFrame({
        "player_id": range(100000, 100000 + n),
        "name": names,
        "season": 2025,
        "causal_war": np.round(causal_war, 2),
        "ci_low": np.round(causal_war - rng.uniform(0.5, 1.5, n), 2),
        "ci_high": np.round(causal_war + rng.uniform(0.5, 1.5, n), 2),
        "park_adj_woba": np.round(rng.uniform(0.280, 0.400, n), 3),
        "raw_woba": np.round(rng.uniform(0.270, 0.410, n), 3),
        "pa": rng.randint(200, 650, size=n),
        "causal_run_value": np.round(causal_war * 10, 2),
        "traditional_war": np.round(traditional_war, 1),
    })
    return df.sort_values("causal_war", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the CausalWAR Analysis page."""
    st.title("CausalWAR: Causal Inference Player Valuation")
    st.caption(
        "De-confounds park, lineup, platoon, and game-state effects using "
        "Double Machine Learning to isolate each player's true causal impact "
        "on run production."
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**CausalWAR isolates a player's true impact** by statistically removing the effects of park, lineup, platoon, and game situation — things traditional WAR only crudely adjusts for.

- **CausalWAR > traditional WAR** → player is undervalued (park or context is suppressing their raw stats)
- **CausalWAR < traditional WAR** → player is overvalued (benefiting from favorable park/lineup context)
- **Confidence intervals** show how certain the estimate is — wider CI = more uncertainty, treat with caution
- **Biggest Movers** are the players where context effects are largest — these are the market inefficiencies
- **Impact:** A 1.0 CausalWAR difference on a free agent is worth ~$8M/year in contract value
""")

    conn = get_db_connection()

    if not _CAUSAL_AVAILABLE and not _USE_MOCK:
        st.error(
            "The `causal_war` analytics module could not be imported. "
            "Ensure `scikit-learn` is installed:\n\n"
            "```\npip install scikit-learn\n```"
        )
        return

    if not _USE_MOCK and (conn is None or not has_data(conn)):
        st.warning(
            "No pitch data available. Run the data backfill pipeline first "
            "(`python scripts/backfill.py`)."
        )
        return

    # ---- Sidebar controls ------------------------------------------------
    with st.sidebar:
        st.markdown("### CausalWAR Options")

        season = st.selectbox(
            "Season",
            options=_get_available_seasons(conn),
            key="causal_war_season",
        )

        position_filter = st.radio(
            "Position Filter",
            options=["All", "Batters", "Pitchers"],
            key="causal_war_position",
        )

        position_type = {
            "All": "all",
            "Batters": "batter",
            "Pitchers": "pitcher",
        }[position_filter]

        if _CAUSAL_AVAILABLE and not _USE_MOCK:
            if st.button("Train CausalWAR Model", type="primary"):
                _train_model_ui(conn, season)
                st.rerun()

    # ---- Load data -------------------------------------------------------
    df = _load_leaderboard(conn, season, position_type)

    if df is None or df.empty:
        st.warning("**Model not trained.** CausalWAR data is not available for this season.")
        st.markdown(
            "Click **Train CausalWAR Model** in the sidebar, or run "
            "`python scripts/precompute.py` to generate cached results."
        )
        if _CAUSAL_AVAILABLE and not _USE_MOCK:
            if st.button("Train CausalWAR Model Now", type="primary", key="causal_train_main"):
                _train_model_ui(conn, season)
                st.rerun()
        return

    # ---- Tabs ------------------------------------------------------------
    tab_board, tab_detail, tab_compare, tab_movers = st.tabs([
        "Leaderboard",
        "Player Detail",
        "CausalWAR vs Traditional WAR",
        "Biggest Movers",
    ])

    with tab_board:
        _render_leaderboard(df)

    with tab_detail:
        _render_player_detail(conn, df, season)

    with tab_compare:
        _render_comparison_scatter(df)

    with tab_movers:
        _render_biggest_movers(df)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _get_available_seasons(conn) -> list[int]:
    """Return list of seasons with pitch data."""
    if conn is None:
        return [2025]
    try:
        result = conn.execute(
            "SELECT DISTINCT EXTRACT(YEAR FROM game_date)::INTEGER AS season "
            "FROM pitches ORDER BY season DESC"
        ).fetchdf()
        seasons = result["season"].tolist()
        return seasons if seasons else [2025]
    except Exception:
        return [2025]


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_causal_leaderboard(_conn, season: int, position_type: str) -> pd.DataFrame | None:
    """Cached wrapper for CausalWAR leaderboard computation."""
    return get_leaderboard(_conn, season, position_type)


def _load_leaderboard(
    conn,
    season: int,
    position_type: str,
) -> pd.DataFrame | None:
    """Load CausalWAR leaderboard, using cache then live computation."""
    if _USE_MOCK:
        return _generate_mock_leaderboard()

    if not _CAUSAL_AVAILABLE:
        return None

    # Try precomputed cache first
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "causal_war", season)
            if cached is not None:
                age_info = cache_age_display(conn, "causal_war", season)
                if age_info:
                    st.caption(age_info)
                return cached
        except Exception:
            pass

    try:
        with st.spinner("Computing... Run `python scripts/precompute.py` for instant loading."):
            return _cached_causal_leaderboard(conn, season, position_type)
    except RuntimeError:
        # Model not trained
        return None
    except Exception as exc:
        st.error(f"Error loading CausalWAR data: {exc}")
        return None


def _train_model_ui(conn, season: int) -> None:
    """Train the CausalWAR model with UI feedback."""
    with st.spinner("Training CausalWAR model... this may take a few minutes."):
        try:
            metrics = train(conn, season=season, n_bootstrap=50, n_estimators=100)
            st.success("CausalWAR model trained successfully!")

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Outcome R2",
                f"{metrics.get('outcome_nuisance_r2', 0):.4f}",
            )
            col2.metric("Players Estimated", metrics.get("n_players_estimated", 0))
            col3.metric("Observations", f"{metrics.get('n_observations', 0):,}")

        except Exception as exc:
            st.error(f"Training failed: {exc}")


# ---------------------------------------------------------------------------
# Leaderboard tab
# ---------------------------------------------------------------------------


def _render_leaderboard(df: pd.DataFrame) -> None:
    """Display the CausalWAR leaderboard table."""
    st.subheader("CausalWAR Leaderboard")

    display_df = df.copy()
    display_df = display_df.reset_index(drop=True)
    display_df.index = display_df.index + 1
    display_df.index.name = "Rank"

    # Select display columns
    display_cols = [
        "name", "causal_war", "ci_low", "ci_high",
        "park_adj_woba", "raw_woba", "pa",
    ]
    if "traditional_war" in display_df.columns:
        display_cols.append("traditional_war")

    available_cols = [c for c in display_cols if c in display_df.columns]

    st.dataframe(
        display_df[available_cols],
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Player"),
            "causal_war": st.column_config.NumberColumn("CausalWAR", format="%.2f"),
            "ci_low": st.column_config.NumberColumn("CI Low", format="%.2f"),
            "ci_high": st.column_config.NumberColumn("CI High", format="%.2f"),
            "park_adj_woba": st.column_config.NumberColumn("Park-Adj wOBA", format="%.3f"),
            "raw_woba": st.column_config.NumberColumn("Raw wOBA", format="%.3f"),
            "pa": st.column_config.NumberColumn("PA", format="%d"),
            "traditional_war": st.column_config.NumberColumn("Trad. WAR", format="%.1f"),
        },
    )

    # Distribution chart
    if len(df) >= 5 and "causal_war" in df.columns:
        st.markdown("**CausalWAR Distribution**")
        _render_distribution(df)


def _render_distribution(df: pd.DataFrame) -> None:
    """Histogram of CausalWAR values."""
    values = df["causal_war"].dropna()
    if values.empty:
        return

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=25,
        marker_color=_PHILLIES_RED,
        opacity=0.75,
        name="CausalWAR",
    ))
    fig.add_vline(
        x=0,
        line=dict(color="white", width=2, dash="dash"),
        annotation_text="Replacement (0)",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis=dict(title="CausalWAR"),
        yaxis=dict(title="Count"),
        template="plotly_dark",
        height=350,
        margin=dict(l=50, r=30, t=30, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="causal_war_dist")


# ---------------------------------------------------------------------------
# Player detail tab
# ---------------------------------------------------------------------------


def _render_player_detail(conn, df: pd.DataFrame, season: int) -> None:
    """Show detailed CausalWAR breakdown for a single player."""
    st.subheader("Player Detail")

    if df.empty or "name" not in df.columns:
        st.info("No player data available.")
        return

    # Build player selector
    player_options = df["name"].dropna().tolist()
    if not player_options:
        player_options = [f"ID {pid}" for pid in df["player_id"].tolist()]

    selected = st.selectbox(
        "Select Player",
        options=player_options,
        key="causal_war_player_select",
    )

    if not selected:
        return

    # Find the player row
    if selected.startswith("ID "):
        pid = int(selected.split(" ")[1])
        player_row = df[df["player_id"] == pid]
    else:
        player_row = df[df["name"] == selected]

    if player_row.empty:
        st.warning("Player not found.")
        return

    row = player_row.iloc[0]

    # ---- Metric cards ----------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("CausalWAR", f"{row.get('causal_war', 'N/A'):.2f}"
                if pd.notna(row.get('causal_war')) else "N/A")
    col2.metric("Park-Adj wOBA", f"{row.get('park_adj_woba', 'N/A'):.3f}"
                if pd.notna(row.get('park_adj_woba')) else "N/A")
    col3.metric("Plate Appearances", f"{int(row.get('pa', 0)):,}")

    trad_war = row.get("traditional_war")
    if pd.notna(trad_war):
        col4.metric("Traditional WAR", f"{trad_war:.1f}")
    else:
        col4.metric("Traditional WAR", "N/A")

    st.markdown("---")

    # ---- Confidence interval forest plot ---------------------------------
    st.markdown("**Confidence Interval**")
    _render_single_forest_plot(row)

    st.markdown("---")

    # ---- Component breakdown ---------------------------------------------
    st.markdown("**CausalWAR Components**")
    _render_component_breakdown(row)


def _render_single_forest_plot(row: pd.Series) -> None:
    """Forest plot for a single player's CausalWAR CI."""
    causal_war = row.get("causal_war")
    ci_low = row.get("ci_low")
    ci_high = row.get("ci_high")
    name = row.get("name", "Unknown")

    if pd.isna(causal_war):
        st.caption("No CausalWAR estimate available for this player.")
        return

    fig = go.Figure()

    # CI error bar
    if pd.notna(ci_low) and pd.notna(ci_high):
        fig.add_trace(go.Scatter(
            x=[causal_war],
            y=[name],
            error_x=dict(
                type="data",
                symmetric=False,
                array=[ci_high - causal_war],
                arrayminus=[causal_war - ci_low],
                color=_PHILLIES_RED,
                thickness=3,
                width=10,
            ),
            mode="markers",
            marker=dict(size=14, color=_PHILLIES_RED, symbol="diamond"),
            name="CausalWAR",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[causal_war],
            y=[name],
            mode="markers",
            marker=dict(size=14, color=_PHILLIES_RED, symbol="diamond"),
            name="CausalWAR",
        ))

    fig.add_vline(x=0, line=dict(color="white", width=1, dash="dash"))
    fig.update_layout(
        xaxis=dict(title="CausalWAR"),
        template="plotly_dark",
        height=180,
        margin=dict(l=120, r=30, t=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="causal_war_single_forest")


def _render_component_breakdown(row: pd.Series) -> None:
    """Show waterfall of CausalWAR components."""
    raw_woba = row.get("raw_woba")
    park_adj = row.get("park_adj_woba")
    causal_war = row.get("causal_war")
    pa = row.get("pa", 0)

    components = []
    if pd.notna(raw_woba):
        components.append(("Raw wOBA", round(float(raw_woba), 3)))
    if pd.notna(park_adj) and pd.notna(raw_woba):
        adj_delta = round(float(park_adj - raw_woba), 3)
        components.append(("Park Adjustment", adj_delta))
    if pd.notna(causal_war):
        components.append(("CausalWAR (wins)", round(float(causal_war), 2)))
    components.append(("Plate Appearances", int(pa)))

    if not components:
        st.caption("No component data available.")
        return

    comp_df = pd.DataFrame(components, columns=["Component", "Value"])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# CausalWAR vs Traditional WAR scatter
# ---------------------------------------------------------------------------


def _render_comparison_scatter(df: pd.DataFrame) -> None:
    """Scatter plot comparing CausalWAR with traditional WAR."""
    st.subheader("CausalWAR vs Traditional WAR")

    if "traditional_war" not in df.columns:
        st.info(
            "Traditional WAR data not available. "
            "Ensure season stats are loaded in the database."
        )
        return

    plot_df = df.dropna(subset=["causal_war", "traditional_war"])
    if plot_df.empty:
        st.info("No players with both CausalWAR and traditional WAR.")
        return

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=plot_df["traditional_war"],
        y=plot_df["causal_war"],
        mode="markers+text",
        marker=dict(
            size=10,
            color=_PHILLIES_RED,
            line=dict(width=1, color="white"),
        ),
        text=plot_df["name"].fillna(""),
        textposition="top center",
        textfont=dict(size=9),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Traditional WAR: %{x:.1f}<br>"
            "CausalWAR: %{y:.2f}<br>"
            "<extra></extra>"
        ),
        name="Players",
    ))

    # Perfect correlation line
    all_vals = pd.concat([plot_df["traditional_war"], plot_df["causal_war"]])
    min_val = float(all_vals.min()) - 0.5
    max_val = float(all_vals.max()) + 0.5

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color=_PHILLIES_LIGHT, dash="dash", width=1),
        name="y = x",
    ))

    fig.update_layout(
        xaxis=dict(title="Traditional WAR"),
        yaxis=dict(title="CausalWAR"),
        template="plotly_dark",
        height=500,
        margin=dict(l=60, r=30, t=30, b=60),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True, key="causal_war_scatter")

    # Correlation stat
    if len(plot_df) >= 3:
        corr = plot_df["causal_war"].corr(plot_df["traditional_war"])
        st.caption(f"Pearson correlation: {corr:.3f} (n={len(plot_df)} players)")


# ---------------------------------------------------------------------------
# Biggest Movers
# ---------------------------------------------------------------------------


def _render_biggest_movers(df: pd.DataFrame) -> None:
    """Show players whose CausalWAR differs most from traditional metrics."""
    st.subheader("Biggest Movers: CausalWAR vs Traditional WAR")

    if "traditional_war" not in df.columns:
        st.info("Traditional WAR data not available for comparison.")
        return

    movers_df = df.dropna(subset=["causal_war", "traditional_war"]).copy()
    if movers_df.empty:
        st.info("No players with both metrics available.")
        return

    movers_df["war_diff"] = movers_df["causal_war"] - movers_df["traditional_war"]
    movers_df["abs_diff"] = movers_df["war_diff"].abs()
    movers_df = movers_df.sort_values("abs_diff", ascending=False).head(20)

    if movers_df.empty:
        return

    # ---- Two-column layout: gainers and losers ---------------------------
    col1, col2 = st.columns(2)

    gainers = movers_df[movers_df["war_diff"] > 0].head(10)
    losers = movers_df[movers_df["war_diff"] < 0].head(10)

    with col1:
        st.markdown("**Undervalued by Traditional WAR**")
        st.caption("CausalWAR > Traditional WAR")
        if not gainers.empty:
            _render_movers_table(gainers)
        else:
            st.caption("None found.")

    with col2:
        st.markdown("**Overvalued by Traditional WAR**")
        st.caption("CausalWAR < Traditional WAR")
        if not losers.empty:
            _render_movers_table(losers)
        else:
            st.caption("None found.")

    st.markdown("---")

    # ---- Divergence bar chart --------------------------------------------
    st.markdown("**WAR Divergence (CausalWAR - Traditional WAR)**")
    _render_divergence_bars(movers_df)

    # ---- Forest plot for top movers --------------------------------------
    st.markdown("**Confidence Intervals for Top Movers**")
    _render_forest_plot(movers_df.head(15))


def _render_movers_table(df: pd.DataFrame) -> None:
    """Display a table of biggest movers."""
    display = df[["name", "causal_war", "traditional_war", "war_diff", "pa"]].copy()
    display.columns = ["Player", "CausalWAR", "Trad. WAR", "Difference", "PA"]
    display = display.reset_index(drop=True)
    display.index = display.index + 1

    st.dataframe(
        display,
        use_container_width=True,
        column_config={
            "CausalWAR": st.column_config.NumberColumn(format="%.2f"),
            "Trad. WAR": st.column_config.NumberColumn(format="%.1f"),
            "Difference": st.column_config.NumberColumn(format="%+.2f"),
            "PA": st.column_config.NumberColumn(format="%d"),
        },
    )


def _render_divergence_bars(df: pd.DataFrame) -> None:
    """Horizontal bar chart of WAR differences."""
    plot_df = df.sort_values("war_diff").copy()

    colors = [
        _POSITIVE_GREEN if v > 0 else _NEGATIVE_RED
        for v in plot_df["war_diff"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=plot_df["name"].fillna("Unknown"),
        x=plot_df["war_diff"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in plot_df["war_diff"]],
        textposition="auto",
    ))
    fig.add_vline(x=0, line=dict(color="white", width=1))
    fig.update_layout(
        xaxis=dict(title="CausalWAR - Traditional WAR"),
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=max(300, len(plot_df) * 30),
        margin=dict(l=120, r=30, t=20, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="causal_war_divergence")


def _render_forest_plot(df: pd.DataFrame) -> None:
    """Forest plot with confidence intervals for multiple players."""
    if not all(c in df.columns for c in ["causal_war", "ci_low", "ci_high"]):
        st.caption("Confidence interval data not available.")
        return

    plot_df = df.dropna(subset=["causal_war"]).copy()
    if plot_df.empty:
        return

    # Sort by CausalWAR descending for display
    plot_df = plot_df.sort_values("causal_war", ascending=True)

    fig = go.Figure()

    names = plot_df["name"].fillna("Unknown").tolist()
    values = plot_df["causal_war"].tolist()
    ci_lows = plot_df["ci_low"].tolist()
    ci_highs = plot_df["ci_high"].tolist()

    # CI bars
    for i, (name, val, lo, hi) in enumerate(zip(names, values, ci_lows, ci_highs)):
        show_legend = i == 0
        if pd.notna(lo) and pd.notna(hi):
            fig.add_trace(go.Scatter(
                x=[lo, hi],
                y=[name, name],
                mode="lines",
                line=dict(color=_PHILLIES_RED, width=3),
                showlegend=False,
            ))

    # Point estimates
    fig.add_trace(go.Scatter(
        x=values,
        y=names,
        mode="markers",
        marker=dict(size=10, color=_PHILLIES_RED, symbol="diamond"),
        name="CausalWAR",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "CausalWAR: %{x:.2f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.add_vline(x=0, line=dict(color="white", width=1, dash="dash"))
    fig.update_layout(
        xaxis=dict(title="CausalWAR"),
        template="plotly_dark",
        height=max(300, len(plot_df) * 35),
        margin=dict(l=140, r=30, t=20, b=50),
    )
    st.plotly_chart(fig, use_container_width=True, key="causal_war_forest")
