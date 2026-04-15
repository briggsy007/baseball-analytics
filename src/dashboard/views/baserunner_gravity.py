"""
Baserunner Gravity Index (BGI) dashboard view.

Displays a BGI dashboard with runner selection, score display with
percentile context, channel decomposition charts, impact comparison
metrics, leaderboard, and team-level roster view.
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

_BGI_AVAILABLE = False
try:
    from src.analytics.baserunner_gravity import (
        BGIConfig,
        BaserunnerGravityModel,
        calculate_bgi,
        batch_calculate,
        get_gravity_leaderboard,
        compute_runner_threat_rate,
        compute_gravity_effect,
        train,
    )
    _BGI_AVAILABLE = True
except ImportError:
    pass

_CACHE_AVAILABLE = False
try:
    from src.dashboard.cache_reader import get_cached_leaderboard, cache_age_display
    _CACHE_AVAILABLE = True
except Exception:
    pass

_USE_MOCK = False

# Phillies red / blue palette
_PHILLIES_RED = "#E81828"
_PHILLIES_BLUE = "#002D72"
_PHILLIES_LIGHT = "#B0B7BC"
_POSITIVE_GREEN = "#2ECC71"
_NEGATIVE_RED = "#E74C3C"
_NEUTRAL_GRAY = "#95A5A6"


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------


def _generate_mock_leaderboard() -> pd.DataFrame:
    """Generate mock BGI data for development / demo mode."""
    rng = np.random.RandomState(42)
    n = 25
    names = [f"Runner {chr(65 + i)}" for i in range(n)]
    bgi = rng.normal(100, 15, size=n)

    df = pd.DataFrame({
        "runner_id": range(300000, 300000 + n),
        "name": names,
        "season": 2025,
        "bgi": np.round(bgi, 1),
        "sb_attempt_rate": np.round(rng.uniform(0, 0.15, n), 4),
        "velocity_effect": np.round(rng.normal(0, 0.3, n), 4),
        "location_effect": np.round(rng.normal(0, 0.02, n), 4),
        "selection_effect": np.round(rng.normal(0, 0.03, n), 4),
        "outcome_effect": np.round(rng.normal(0, 0.01, n), 4),
        "n_pitches": rng.randint(100, 800, size=n),
        "percentile": np.round(rng.uniform(5, 99, n), 0).astype(int),
    })
    return df.sort_values("bgi", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Baserunner Gravity Index page."""
    st.title("Baserunner Gravity Index (BGI)")
    st.caption(
        "Measures how much a baserunner distorts pitcher behaviour and batter "
        "outcomes by their mere presence on base -- the baseball equivalent of "
        "NBA offensive gravity."
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**BGI quantifies the invisible value of a baserunner's threat** — how much they distort the pitcher-batter matchup just by standing on base.

- **BGI > 115** = elite gravity (pitcher significantly changes behavior — more fastballs, less command, worse outcomes for the batter at the plate)
- **BGI 100-115** = above-average threat
- **BGI 85-100** = below-average threat (pitcher largely ignores this runner)
- **BGI < 85** = negative gravity (pitcher is *more* effective with this runner on — possibly because they simplify their approach)
- **4 channels:** velocity change, command scatter, pitch selection shift, batter outcome impact
- **Impact:** A high-BGI pinch runner who never attempts a steal can still be worth 0.5+ WAR/season in "gravity runs" — value that's completely invisible in traditional stats
""")

    conn = get_db_connection()

    if not _BGI_AVAILABLE and not _USE_MOCK:
        st.error(
            "The `baserunner_gravity` analytics module could not be imported. "
            "Check that all dependencies are installed."
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
        st.markdown("### BGI Options")

        season = st.selectbox(
            "Season",
            options=_get_available_seasons(conn),
            key="bgi_season",
        )

        if _BGI_AVAILABLE and not _USE_MOCK:
            if st.button("Compute BGI Scores", type="primary"):
                _train_model_ui(conn, season)
                st.rerun()

    # ---- Load data -------------------------------------------------------
    df = _load_leaderboard(conn, season)

    if df is None or df.empty:
        st.info(
            "No BGI data available for this season. "
            "Click **Compute BGI Scores** in the sidebar to build it."
        )
        return

    # ---- Tabs ------------------------------------------------------------
    tab_dashboard, tab_channels, tab_leader, tab_team = st.tabs([
        "Runner Dashboard",
        "Channel Decomposition",
        "Leaderboard",
        "Team View",
    ])

    with tab_dashboard:
        _render_runner_dashboard(conn, df, season)

    with tab_channels:
        _render_channel_decomposition(conn, df, season)

    with tab_leader:
        _render_leaderboard(df)

    with tab_team:
        _render_team_view(conn, df, season)


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


def _load_leaderboard(conn, season: int) -> pd.DataFrame | None:
    """Load BGI leaderboard, using cache then live computation."""
    if _USE_MOCK:
        return _generate_mock_leaderboard()

    if not _BGI_AVAILABLE:
        return None

    # Try cache first
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "baserunner_gravity", season)
            if cached is not None:
                age_info = cache_age_display(conn, "baserunner_gravity", season)
                if age_info:
                    st.caption(age_info)
                return cached
        except Exception:
            pass

    try:
        with st.spinner("Computing... Run `python scripts/precompute.py` for instant loading."):
            config = BGIConfig(min_appearances_batch=50)
            return batch_calculate(conn, season, min_appearances=50, config=config)
    except Exception as exc:
        st.error(f"Error computing BGI data: {exc}")
        return None


def _train_model_ui(conn, season: int) -> None:
    """Compute BGI scores with UI feedback."""
    with st.spinner("Computing Baserunner Gravity Index... this may take a minute."):
        try:
            metrics = train(conn, season=season)
            st.success("BGI scores computed successfully!")

            col1, col2, col3 = st.columns(3)
            col1.metric("Runners Scored", metrics.get("n_runners_scored", 0))
            col2.metric(
                "Mean BGI",
                f"{metrics.get('mean_bgi', 0):.1f}"
                if metrics.get("mean_bgi") else "N/A",
            )
            col3.metric(
                "Std BGI",
                f"{metrics.get('std_bgi', 0):.1f}"
                if metrics.get("std_bgi") else "N/A",
            )
        except Exception as exc:
            st.error(f"Computation failed: {exc}")


# ---------------------------------------------------------------------------
# Runner Dashboard tab
# ---------------------------------------------------------------------------


def _render_runner_dashboard(conn, df: pd.DataFrame, season: int) -> None:
    """Show BGI big number + percentile context for a selected runner."""
    st.subheader("Runner BGI Dashboard")

    if df.empty:
        st.info("No runner data available.")
        return

    # Build player selector
    player_options = []
    for _, row in df.iterrows():
        label = row.get("name") or f"ID {row['runner_id']}"
        player_options.append(label)

    selected = st.selectbox(
        "Select Runner",
        options=player_options,
        key="bgi_runner_select",
    )
    if not selected:
        return

    # Find the selected row
    idx = player_options.index(selected)
    row = df.iloc[idx]

    bgi = row.get("bgi")
    pct = row.get("percentile", 50)

    # ---- Big number display + percentile gauge ----------------------------
    col1, col2, col3, col4 = st.columns(4)

    bgi_color = _PHILLIES_RED if bgi and bgi > 100 else _PHILLIES_BLUE
    col1.metric(
        "BGI Score",
        f"{bgi:.1f}" if pd.notna(bgi) else "N/A",
    )
    col2.metric("Percentile", f"{int(pct)}th" if pd.notna(pct) else "N/A")
    col3.metric(
        "SB Attempt Rate",
        f"{row.get('sb_attempt_rate', 0):.1%}",
    )
    col4.metric("Pitches Tracked", f"{int(row.get('n_pitches', 0)):,}")

    st.markdown("---")

    # ---- Impact comparison: "When X is on 1st, batters see..." -----------
    st.markdown(f"**When {selected} is on 1st, batters see...**")

    velo_eff = row.get("velocity_effect", 0) or 0
    loc_eff = row.get("location_effect", 0) or 0
    sel_eff = row.get("selection_effect", 0) or 0
    out_eff = row.get("outcome_effect", 0) or 0

    icol1, icol2, icol3, icol4 = st.columns(4)
    icol1.metric(
        "Velocity Change",
        f"{velo_eff:+.2f} mph",
        delta=f"{velo_eff:+.2f}",
        delta_color="inverse",
    )
    icol2.metric(
        "Command Disruption",
        f"{loc_eff:+.4f}",
        delta="More scatter" if loc_eff > 0 else "Less scatter",
        delta_color="normal" if loc_eff > 0 else "inverse",
    )
    icol3.metric(
        "Fastball% Shift",
        f"{sel_eff:+.1%}",
        delta=f"{sel_eff:+.1%}",
        delta_color="normal" if sel_eff > 0 else "inverse",
    )
    icol4.metric(
        "xwOBA Impact",
        f"{out_eff:+.4f}",
        delta=f"{out_eff:+.4f}",
        delta_color="normal" if out_eff > 0 else "inverse",
    )

    # ---- BGI gauge chart --------------------------------------------------
    st.markdown("---")
    st.markdown("**BGI Scale**")
    _render_bgi_gauge(bgi, selected)


def _render_bgi_gauge(bgi: float | None, name: str) -> None:
    """Render a gauge chart for BGI score."""
    if bgi is None or pd.isna(bgi):
        st.caption("BGI not available.")
        return

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bgi,
        title={"text": name, "font": {"size": 16}},
        gauge={
            "axis": {"range": [50, 150], "tickwidth": 1},
            "bar": {"color": _PHILLIES_RED},
            "steps": [
                {"range": [50, 85], "color": "#1a3a5c"},
                {"range": [85, 100], "color": "#2c5f8a"},
                {"range": [100, 115], "color": "#8a3030"},
                {"range": [115, 150], "color": "#c02020"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8,
                "value": 100,
            },
        },
        number={"suffix": "", "font": {"size": 36}},
    ))
    fig.update_layout(
        template="plotly_dark",
        height=280,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    st.plotly_chart(fig, use_container_width=True, key="bgi_gauge")


# ---------------------------------------------------------------------------
# Channel Decomposition tab
# ---------------------------------------------------------------------------


def _render_channel_decomposition(conn, df: pd.DataFrame, season: int) -> None:
    """Horizontal bar chart showing per-channel effects for selected runner."""
    st.subheader("Channel Decomposition")

    if df.empty:
        st.info("No data available.")
        return

    player_options = []
    for _, row in df.iterrows():
        label = row.get("name") or f"ID {row['runner_id']}"
        player_options.append(label)

    selected = st.selectbox(
        "Select Runner",
        options=player_options,
        key="bgi_channel_runner",
    )
    if not selected:
        return

    idx = player_options.index(selected)
    row = df.iloc[idx]

    channels = {
        "Velocity": row.get("velocity_effect", 0) or 0,
        "Location (Command)": row.get("location_effect", 0) or 0,
        "Selection (FB%)": row.get("selection_effect", 0) or 0,
        "Outcome (xwOBA)": row.get("outcome_effect", 0) or 0,
    }

    names = list(channels.keys())
    values = list(channels.values())

    colors = [
        _POSITIVE_GREEN if v > 0 else _NEGATIVE_RED if v < 0 else _NEUTRAL_GRAY
        for v in values
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names,
        x=values,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.4f}" for v in values],
        textposition="auto",
    ))

    fig.add_vline(x=0, line=dict(color="white", width=1))
    fig.update_layout(
        xaxis=dict(title="Effect (runner on base vs league)"),
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=300,
        margin=dict(l=160, r=30, t=20, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="bgi_channels")

    st.caption(
        "Positive velocity = faster pitches when runner is on base. "
        "Positive location = more scatter (worse command). "
        "Positive selection = higher fastball%. "
        "Positive outcome = higher batter xwOBA."
    )


# ---------------------------------------------------------------------------
# Leaderboard tab
# ---------------------------------------------------------------------------


def _render_leaderboard(df: pd.DataFrame) -> None:
    """Display the BGI leaderboard table."""
    st.subheader("BGI Leaderboard -- Highest Gravity Runners")

    if df.empty:
        st.info("No data available.")
        return

    display_df = df.copy().sort_values("bgi", ascending=False)
    display_df = display_df.reset_index(drop=True)
    display_df.index = display_df.index + 1
    display_df.index.name = "Rank"

    display_cols = [
        "name", "bgi", "percentile", "sb_attempt_rate",
        "velocity_effect", "outcome_effect", "n_pitches",
    ]
    available_cols = [c for c in display_cols if c in display_df.columns]

    st.dataframe(
        display_df[available_cols],
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Runner"),
            "bgi": st.column_config.NumberColumn("BGI", format="%.1f"),
            "percentile": st.column_config.NumberColumn("Pctl", format="%d"),
            "sb_attempt_rate": st.column_config.NumberColumn("SB Att Rate", format="%.3f"),
            "velocity_effect": st.column_config.NumberColumn("Velo Eff", format="%+.3f"),
            "outcome_effect": st.column_config.NumberColumn("xwOBA Eff", format="%+.4f"),
            "n_pitches": st.column_config.NumberColumn("Pitches", format="%d"),
        },
    )

    # Distribution histogram
    if len(df) >= 5 and "bgi" in df.columns:
        st.markdown("**BGI Distribution**")
        _render_distribution(df)


def _render_distribution(df: pd.DataFrame) -> None:
    """Histogram of BGI values."""
    values = df["bgi"].dropna()
    if values.empty:
        return

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=20,
        marker_color=_PHILLIES_RED,
        opacity=0.75,
        name="BGI",
    ))
    fig.add_vline(
        x=100,
        line=dict(color="white", width=2, dash="dash"),
        annotation_text="Neutral (100)",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis=dict(title="BGI Score"),
        yaxis=dict(title="Count"),
        template="plotly_dark",
        height=350,
        margin=dict(l=50, r=30, t=30, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="bgi_dist")


# ---------------------------------------------------------------------------
# Team View tab
# ---------------------------------------------------------------------------


def _render_team_view(conn, df: pd.DataFrame, season: int) -> None:
    """All runners on a roster ranked by BGI."""
    st.subheader("Team BGI Roster View")

    if df.empty:
        st.info("No runner data available.")
        return

    # Get available teams
    teams = _get_teams(conn, season)
    if not teams:
        teams = ["PHI"]

    selected_team = st.selectbox(
        "Select Team",
        options=teams,
        key="bgi_team_select",
    )

    # Filter runners by team (via their batting appearances)
    team_runner_ids = _get_team_runner_ids(conn, selected_team, season)

    if not team_runner_ids:
        st.info(f"No runners found for {selected_team} in {season}.")
        return

    team_df = df[df["runner_id"].isin(team_runner_ids)].copy()

    if team_df.empty:
        st.info(f"No BGI data for {selected_team} runners.")
        return

    team_df = team_df.sort_values("bgi", ascending=False).reset_index(drop=True)

    # Horizontal bar chart of team runners
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=team_df["name"].fillna(team_df["runner_id"].astype(str)),
        x=team_df["bgi"],
        orientation="h",
        marker_color=[
            _PHILLIES_RED if b > 100 else _PHILLIES_BLUE
            for b in team_df["bgi"]
        ],
        text=[f"{b:.1f}" for b in team_df["bgi"]],
        textposition="auto",
    ))
    fig.add_vline(x=100, line=dict(color="white", width=2, dash="dash"))
    fig.update_layout(
        xaxis=dict(title="BGI Score"),
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=max(300, len(team_df) * 35),
        margin=dict(l=140, r=30, t=20, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="bgi_team_chart")

    # Table
    st.dataframe(
        team_df[["name", "bgi", "percentile", "sb_attempt_rate", "n_pitches"]].reset_index(drop=True),
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Runner"),
            "bgi": st.column_config.NumberColumn("BGI", format="%.1f"),
            "percentile": st.column_config.NumberColumn("Pctl", format="%d"),
            "sb_attempt_rate": st.column_config.NumberColumn("SB Att Rate", format="%.3f"),
            "n_pitches": st.column_config.NumberColumn("Pitches", format="%d"),
        },
    )


def _get_teams(conn, season: int) -> list[str]:
    """Return list of teams in the season."""
    if conn is None:
        return []
    try:
        result = conn.execute(
            "SELECT DISTINCT home_team FROM pitches "
            "WHERE EXTRACT(YEAR FROM game_date) = $1 "
            "ORDER BY home_team",
            [season],
        ).fetchdf()
        return result["home_team"].dropna().tolist()
    except Exception:
        return []


def _get_team_runner_ids(conn, team: str, season: int) -> list[int]:
    """Return list of runner IDs (players who appeared as batters for this team)."""
    if conn is None:
        return []
    try:
        result = conn.execute("""
            SELECT DISTINCT batter_id
            FROM pitches
            WHERE (
                (inning_topbot = 'Top' AND away_team = $1)
                OR (inning_topbot = 'Bot' AND home_team = $1)
            )
            AND EXTRACT(YEAR FROM game_date) = $2
        """, [team, season]).fetchdf()
        return result["batter_id"].tolist()
    except Exception:
        return []
