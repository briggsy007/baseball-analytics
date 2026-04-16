"""
Defensive Pressing Intensity (DPI) dashboard view.

Displays team-level defensive efficiency metrics:
- Team DPI big-number with league rank & consistency score
- BIP outcome chart: launch_speed vs launch_angle colored by actual vs expected
- Team leaderboard: all teams ranked by DPI with bar chart
- Extra-base prevention: team ranking on limiting advancement
- Game-by-game DPI: timeline chart across the season
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.dashboard.db_helper import get_db_connection, has_data

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_DPI_AVAILABLE = False
try:
    from src.analytics.defensive_pressing import (
        DPIConfig,
        DefensivePressingModel,
        train_expected_out_model,
        compute_expected_outs,
        calculate_game_dpi,
        calculate_team_dpi,
        batch_calculate,
        get_player_dpi,
        get_team_game_dpi_timeline,
        build_bip_features,
        compute_spray_angle,
    )
    _DPI_AVAILABLE = True
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
_NEUTRAL_GOLD = "#FFC145"


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

def _generate_mock_leaderboard() -> pd.DataFrame:
    """Generate mock team DPI data for development / demo mode."""
    rng = np.random.RandomState(42)
    teams = [
        "PHI", "NYM", "ATL", "WSH", "MIA", "LAD", "SDP", "SFG", "ARI", "COL",
        "NYY", "BOS", "TBR", "TOR", "BAL", "CLE", "MIN", "DET", "CHW", "KCR",
        "HOU", "SEA", "TEX", "LAA", "OAK", "STL", "CHC", "MIL", "CIN", "PIT",
    ]
    n = len(teams)
    dpi_mean = np.round(rng.normal(0, 2, size=n), 3)

    df = pd.DataFrame({
        "team_id": teams,
        "season": 2025,
        "dpi_mean": sorted(dpi_mean, reverse=True),
        "dpi_total": np.round(sorted(dpi_mean, reverse=True) * rng.randint(100, 162, n), 1),
        "dpi_std": np.round(rng.uniform(1.0, 4.0, n), 3),
        "consistency": np.round(rng.uniform(0.2, 0.6, n), 4),
        "extra_base_prevention": np.round(rng.uniform(0.55, 0.80, n), 3),
        "n_games": rng.randint(100, 162, n),
    })
    df = df.sort_values("dpi_mean", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["percentile"] = np.round(100 * (1 - (df["rank"] - 1) / len(df)), 1)
    return df


def _generate_mock_timeline() -> pd.DataFrame:
    """Generate mock game-by-game DPI timeline."""
    rng = np.random.RandomState(42)
    n = 100
    dates = pd.date_range("2025-04-01", periods=n, freq="2D")
    dpi_vals = np.cumsum(rng.normal(0, 1.5, n)) / np.arange(1, n + 1) * 3

    return pd.DataFrame({
        "game_date": dates,
        "game_pk": range(800000, 800000 + n),
        "dpi": np.round(dpi_vals, 3),
        "n_bip": rng.randint(15, 40, n),
        "actual_outs": rng.randint(8, 28, n),
        "expected_outs": np.round(rng.uniform(8, 25, n), 1),
        "extra_base_prevention": np.round(rng.uniform(0.5, 0.9, n), 3),
    })


def _generate_mock_bip_scatter() -> pd.DataFrame:
    """Generate mock BIP scatter data for the outcome chart."""
    rng = np.random.RandomState(42)
    n = 300
    launch_speed = rng.normal(88, 12, n)
    launch_angle = rng.normal(12, 20, n)
    expected_out = 1 / (1 + np.exp(0.03 * (launch_speed - 85) + 0.02 * launch_angle))
    actual_out = (rng.random(n) < expected_out).astype(int)

    return pd.DataFrame({
        "launch_speed": np.round(launch_speed, 1),
        "launch_angle": np.round(launch_angle, 1),
        "expected_out_prob": np.round(expected_out, 3),
        "actual_out": actual_out,
    })


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _cached_dpi_timeline(team_id: str, season: int) -> pd.DataFrame | None:
    """Cached DPI timeline for a team/season."""
    conn = get_db_connection()
    return get_team_game_dpi_timeline(conn, team_id, season)


def render() -> None:
    """Render the Defensive Pressing Intensity (DPI) Analysis page."""
    st.title("Defensive Pressing Intensity (DPI)")
    st.caption(
        "Soccer gegenpressing metrics applied to baseball defense. Measures how "
        "effectively a team converts batted balls in play into outs compared to "
        "expectation, using a HistGradientBoosting model on BIP features."
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**DPI measures how well a defense converts batted balls into outs** compared to what an average defense would do on the same batted balls.

- **Positive DPI** = defense is making more outs than expected (elite fielding, good positioning, efficient transitions)
- **DPI near 0** = league-average defense
- **Negative DPI** = defense is leaking hits on balls that should be caught (poor range, bad positioning, slow reactions)
- **Consistency score** matters — a defense with high average DPI but wild variance is unreliable in big moments
- **Extra-base prevention** measures how well a defense limits damage on hits (holding singles to singles, preventing doubles)
- **Impact:** The gap between the best and worst defensive teams is worth 30-50 runs per season — equivalent to a top free agent signing
""")

    conn = get_db_connection()

    if not _DPI_AVAILABLE and not _USE_MOCK:
        st.error(
            "The `defensive_pressing` analytics module could not be imported. "
            "Check that `src/analytics/defensive_pressing.py` exists."
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
        st.markdown("### DPI Options")

        season = st.selectbox(
            "Season",
            options=_get_available_seasons(),
            key="dpi_season",
        )

    # ---- Load leaderboard ------------------------------------------------
    leaderboard = _load_leaderboard(conn, season)

    if leaderboard is None or leaderboard.empty:
        st.info(
            "No DPI data available for this season. Ensure enough batted-ball "
            "data is loaded."
        )
        return

    # ---- Team selector ---------------------------------------------------
    team_options = leaderboard["team_id"].tolist()
    default_idx = team_options.index("PHI") if "PHI" in team_options else 0
    selected_team = st.selectbox(
        "Select Team",
        options=team_options,
        index=default_idx,
        key="dpi_team_select",
    )

    # ---- Tabs ------------------------------------------------------------
    tab_overview, tab_scatter, tab_board, tab_ebp, tab_timeline = st.tabs([
        "Team DPI",
        "BIP Outcome Chart",
        "Team Leaderboard",
        "Extra-Base Prevention",
        "Game-by-Game",
    ])

    with tab_overview:
        _render_team_overview(conn, selected_team, season, leaderboard)

    with tab_scatter:
        _render_bip_scatter(conn, selected_team, season)

    with tab_board:
        _render_leaderboard(leaderboard)

    with tab_ebp:
        _render_extra_base_prevention(leaderboard)

    with tab_timeline:
        _render_timeline(conn, selected_team, season)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _get_available_seasons() -> list[int]:
    """Return list of seasons with pitch data."""
    conn = get_db_connection()
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
    """Load DPI leaderboard, using cache then live computation."""
    if _USE_MOCK:
        return _generate_mock_leaderboard()

    if not _DPI_AVAILABLE:
        return None

    # Try cache first
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "defensive_pressing", season)
            if cached is not None:
                age_info = cache_age_display(conn, "defensive_pressing", season)
                if age_info:
                    st.caption(age_info)
                return cached
        except Exception:
            pass

    try:
        with st.spinner("Computing... Run `python scripts/precompute.py` for instant loading."):
            return batch_calculate(conn, season)
    except Exception as exc:
        st.error(f"Error computing DPI leaderboard: {exc}")
        return None


# ---------------------------------------------------------------------------
# Tab: Team DPI Overview
# ---------------------------------------------------------------------------

def _render_team_overview(
    conn, team_id: str, season: int, leaderboard: pd.DataFrame,
) -> None:
    """Big-number DPI card with league rank and consistency score."""
    st.subheader(f"{team_id} Defensive Pressing Intensity")

    team_row = leaderboard[leaderboard["team_id"] == team_id]
    if team_row.empty:
        st.warning(f"No DPI data for {team_id}")
        return

    row = team_row.iloc[0]
    dpi_mean = row["dpi_mean"]
    rank = int(row["rank"])
    n_teams = len(leaderboard)
    consistency = row.get("consistency", 0)
    n_games = int(row.get("n_games", 0))
    percentile = row.get("percentile", 0)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        color = _POSITIVE_GREEN if dpi_mean > 0 else _NEGATIVE_RED
        st.metric(
            "DPI (avg per game)",
            f"{dpi_mean:+.3f}",
            delta=f"Rank #{rank} of {n_teams}",
        )

    with col2:
        st.metric("Percentile", f"{percentile:.0f}th")

    with col3:
        st.metric("Consistency", f"{consistency:.3f}")

    with col4:
        st.metric("Games", str(n_games))

    # Gauge chart for DPI
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=dpi_mean,
        title={"text": "DPI (Outs Above Expected / Game)"},
        delta={"reference": 0, "increasing": {"color": _POSITIVE_GREEN},
               "decreasing": {"color": _NEGATIVE_RED}},
        gauge={
            "axis": {"range": [-5, 5]},
            "bar": {"color": _PHILLIES_RED if dpi_mean > 0 else _NEGATIVE_RED},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [-5, -1], "color": "rgba(231,76,60,0.2)"},
                {"range": [-1, 1], "color": "rgba(255,193,69,0.2)"},
                {"range": [1, 5], "color": "rgba(46,204,113,0.2)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8,
                "value": 0,
            },
        },
    ))
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(t=50, b=20, l=30, r=30),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: BIP Outcome Chart
# ---------------------------------------------------------------------------

def _render_bip_scatter(conn, team_id: str, season: int) -> None:
    """Scatter of launch_speed vs launch_angle, colored by outcome vs expected."""
    st.subheader("Batted Ball Outcomes vs Expected")

    if _USE_MOCK:
        scatter_df = _generate_mock_bip_scatter()
    elif _DPI_AVAILABLE:
        try:
            scatter_df = _load_bip_data(team_id, season)
        except Exception as exc:
            st.error(f"Error loading BIP data: {exc}")
            return
    else:
        return

    if scatter_df is None or scatter_df.empty:
        st.info("No batted-ball data available for this team/season.")
        return

    fig = go.Figure()

    # Split by outcome
    outs = scatter_df[scatter_df["actual_out"] == 1]
    hits = scatter_df[scatter_df["actual_out"] == 0]

    fig.add_trace(go.Scatter(
        x=outs["launch_speed"],
        y=outs["launch_angle"],
        mode="markers",
        name="Out Made",
        marker=dict(
            color=outs["expected_out_prob"],
            colorscale="RdYlGn",
            cmin=0, cmax=1,
            size=6,
            opacity=0.7,
            symbol="circle",
            colorbar=dict(title="xOut Prob", x=1.05),
        ),
        text=[f"xOut: {p:.2f}" for p in outs["expected_out_prob"]],
        hovertemplate=(
            "EV: %{x:.1f} mph<br>"
            "LA: %{y:.1f} deg<br>"
            "%{text}<br>"
            "Result: Out<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=hits["launch_speed"],
        y=hits["launch_angle"],
        mode="markers",
        name="Hit Allowed",
        marker=dict(
            color=_NEGATIVE_RED,
            size=7,
            opacity=0.8,
            symbol="x",
        ),
        hovertemplate=(
            "EV: %{x:.1f} mph<br>"
            "LA: %{y:.1f} deg<br>"
            "Result: Hit<extra></extra>"
        ),
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Exit Velocity (mph)",
        yaxis_title="Launch Angle (deg)",
        height=500,
        margin=dict(t=30, b=40, l=50, r=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Each dot is a batted ball in play against this team's defense. "
        "Color intensity shows expected out probability (green = easy out, "
        "red = likely hit). X marks are hits allowed."
    )


@st.cache_data(ttl=3600)
def _load_bip_data(team_id: str, season: int) -> pd.DataFrame:
    """Load BIP data with expected out probabilities for the scatter chart."""
    conn = get_db_connection()
    query = """
        SELECT
            launch_speed, launch_angle, hc_x, hc_y, bb_type, events
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) = $1
          AND type = 'X'
          AND events IS NOT NULL
          AND launch_speed IS NOT NULL
          AND launch_angle IS NOT NULL
          AND hc_x IS NOT NULL
          AND hc_y IS NOT NULL
          AND bb_type IS NOT NULL
          AND (
              (home_team = $2 AND inning_topbot = 'Top')
              OR (away_team = $2 AND inning_topbot = 'Bot')
          )
    """
    df = conn.execute(query, [season, team_id]).fetchdf()

    if df.empty:
        return pd.DataFrame()

    from src.analytics.defensive_pressing import _is_out

    xout = compute_expected_outs(df)
    actual = _is_out(df["events"])

    return pd.DataFrame({
        "launch_speed": df["launch_speed"],
        "launch_angle": df["launch_angle"],
        "expected_out_prob": xout,
        "actual_out": actual,
    })


# ---------------------------------------------------------------------------
# Tab: Team Leaderboard
# ---------------------------------------------------------------------------

def _render_leaderboard(leaderboard: pd.DataFrame) -> None:
    """All teams ranked by DPI with horizontal bar chart."""
    st.subheader("Team DPI Leaderboard")

    if leaderboard.empty:
        st.info("No leaderboard data.")
        return

    # Bar chart
    sorted_df = leaderboard.sort_values("dpi_mean", ascending=True)

    colors = [
        _POSITIVE_GREEN if v > 0 else _NEGATIVE_RED
        for v in sorted_df["dpi_mean"]
    ]

    fig = go.Figure(go.Bar(
        x=sorted_df["dpi_mean"],
        y=sorted_df["team_id"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.3f}" for v in sorted_df["dpi_mean"]],
        textposition="outside",
        hovertemplate=(
            "%{y}<br>"
            "DPI: %{x:+.3f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="DPI (Outs Above Expected / Game)",
        yaxis_title="",
        height=max(400, len(sorted_df) * 28),
        margin=dict(t=30, b=40, l=60, r=60),
        xaxis=dict(zeroline=True, zerolinecolor="white", zerolinewidth=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Data table
    display_cols = [
        "rank", "team_id", "dpi_mean", "dpi_total", "consistency",
        "extra_base_prevention", "n_games", "percentile",
    ]
    available = [c for c in display_cols if c in leaderboard.columns]
    st.dataframe(
        leaderboard[available].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Tab: Extra-Base Prevention
# ---------------------------------------------------------------------------

def _render_extra_base_prevention(leaderboard: pd.DataFrame) -> None:
    """Team ranking on limiting advancement (extra-base prevention rate)."""
    st.subheader("Extra-Base Prevention")
    st.caption(
        "Fraction of hits that are kept to singles. Higher = better at "
        "preventing doubles, triples, and home runs on batted balls."
    )

    if "extra_base_prevention" not in leaderboard.columns:
        st.info("Extra-base prevention data not available.")
        return

    ebp_df = leaderboard.dropna(subset=["extra_base_prevention"]).copy()
    if ebp_df.empty:
        st.info("No extra-base prevention data.")
        return

    ebp_df = ebp_df.sort_values("extra_base_prevention", ascending=True)

    fig = go.Figure(go.Bar(
        x=ebp_df["extra_base_prevention"],
        y=ebp_df["team_id"],
        orientation="h",
        marker_color=[
            _POSITIVE_GREEN if v >= ebp_df["extra_base_prevention"].median()
            else _NEUTRAL_GOLD
            for v in ebp_df["extra_base_prevention"]
        ],
        text=[f"{v:.1%}" for v in ebp_df["extra_base_prevention"]],
        textposition="outside",
        hovertemplate=(
            "%{y}<br>"
            "EBP: %{x:.1%}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Extra-Base Prevention Rate",
        yaxis_title="",
        height=max(400, len(ebp_df) * 28),
        margin=dict(t=30, b=40, l=60, r=80),
        xaxis=dict(tickformat=".0%"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Game-by-Game Timeline
# ---------------------------------------------------------------------------

def _render_timeline(conn, team_id: str, season: int) -> None:
    """Game-by-game DPI timeline across the season."""
    st.subheader(f"{team_id} DPI Timeline ({season})")

    if _USE_MOCK:
        timeline = _generate_mock_timeline()
    elif _DPI_AVAILABLE:
        try:
            timeline = _cached_dpi_timeline(team_id, season)
        except Exception as exc:
            st.error(f"Error loading timeline: {exc}")
            return
    else:
        return

    if timeline is None or timeline.empty:
        st.info("No game-by-game data available.")
        return

    # DPI line chart with rolling average
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=("Game DPI", "BIP per Game"),
    )

    # Game DPI bars
    colors = [
        _POSITIVE_GREEN if v > 0 else _NEGATIVE_RED
        for v in timeline["dpi"]
    ]
    fig.add_trace(
        go.Bar(
            x=timeline["game_date"],
            y=timeline["dpi"],
            marker_color=colors,
            name="Game DPI",
            opacity=0.6,
            hovertemplate="Date: %{x}<br>DPI: %{y:+.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Rolling average
    if len(timeline) >= 5:
        rolling = timeline["dpi"].rolling(window=10, min_periods=3).mean()
        fig.add_trace(
            go.Scatter(
                x=timeline["game_date"],
                y=rolling,
                mode="lines",
                name="10-Game Avg",
                line=dict(color="white", width=2),
                hovertemplate="10-game avg: %{y:+.3f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # BIP per game
    fig.add_trace(
        go.Bar(
            x=timeline["game_date"],
            y=timeline["n_bip"],
            marker_color=_PHILLIES_LIGHT,
            name="BIP",
            opacity=0.5,
            hovertemplate="BIP: %{y}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=550,
        margin=dict(t=50, b=40, l=50, r=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        showlegend=True,
    )

    fig.update_yaxes(
        title_text="DPI",
        zeroline=True,
        zerolinecolor="rgba(255,255,255,0.3)",
        row=1, col=1,
    )
    fig.update_yaxes(title_text="BIP Count", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Season DPI Avg", f"{timeline['dpi'].mean():+.3f}")
    with col2:
        st.metric("Best Game", f"{timeline['dpi'].max():+.3f}")
    with col3:
        st.metric("Worst Game", f"{timeline['dpi'].min():+.3f}")
