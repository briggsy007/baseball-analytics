"""
Allostatic Batting Load (ABL) dashboard view.

Displays multi-channel cumulative decision fatigue for hitters:
- Load timeline with all 5 channels + composite ABL
- Current load gauges for each channel
- Fatigue vs performance scatter (ABL vs chase rate / wOBA)
- Rest optimisation: predicted recovery days
- Leaderboard of most / least fatigued batters
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

_ABL_AVAILABLE = False
try:
    from src.analytics.allostatic_load import (
        ABLConfig,
        AllostaticLoadModel,
        calculate_abl,
        batch_calculate,
        compute_game_stressors,
        validate_against_outcomes,
        predict_recovery_days,
    )
    _ABL_AVAILABLE = True
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
_CHANNEL_COLORS = {
    "pitch_processing": "#E81828",    # red
    "decision_conflict": "#FF6B35",   # orange
    "swing_exertion": "#FFC145",      # gold
    "temporal_demand": "#2ECC71",     # green
    "travel_stress": "#3498DB",       # blue
    "composite_abl": "#FFFFFF",       # white
}
_CHANNEL_LABELS = {
    "pitch_processing": "Pitch Processing",
    "decision_conflict": "Decision Conflict",
    "swing_exertion": "Swing Exertion",
    "temporal_demand": "Temporal Demand",
    "travel_stress": "Travel Stress",
    "composite_abl": "Composite ABL",
}


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

def _generate_mock_leaderboard() -> pd.DataFrame:
    """Generate mock ABL data for development / demo mode."""
    rng = np.random.RandomState(42)
    n = 25
    names = [f"Player {chr(65 + i)}" for i in range(n)]
    abls = np.clip(rng.normal(45, 20, size=n), 0, 100)

    df = pd.DataFrame({
        "batter_id": range(100000, 100000 + n),
        "name": names,
        "season": 2025,
        "composite_abl": np.round(abls, 2),
        "peak_abl": np.round(np.clip(abls + rng.uniform(5, 20, n), 0, 100), 2),
        "peak_date": "2025-07-15",
        "games_played": rng.randint(60, 140, size=n),
        "pitch_processing": np.round(rng.uniform(20, 80, n), 1),
        "decision_conflict": np.round(rng.uniform(5, 30, n), 1),
        "swing_exertion": np.round(rng.uniform(15, 60, n), 1),
        "temporal_demand": np.round(rng.uniform(3, 7, n), 1),
        "travel_stress": np.round(rng.uniform(0, 15, n), 1),
    })
    return df.sort_values("composite_abl", ascending=False).reset_index(drop=True)


def _generate_mock_timeline() -> pd.DataFrame:
    """Generate mock timeline for a single batter."""
    rng = np.random.RandomState(42)
    n = 80
    dates = pd.date_range("2025-04-01", periods=n, freq="2D")

    # Simulate gradual buildup with noise
    base = np.cumsum(rng.normal(0.5, 0.3, n))
    base = np.clip(base, 0, None)

    df = pd.DataFrame({
        "game_date": dates,
        "pitch_processing": np.clip(base * 1.2 + rng.normal(0, 3, n), 0, None),
        "decision_conflict": np.clip(base * 0.5 + rng.normal(0, 1.5, n), 0, None),
        "swing_exertion": np.clip(base * 0.8 + rng.normal(0, 2, n), 0, None),
        "temporal_demand": np.clip(rng.uniform(3, 7, n), 0, None),
        "travel_stress": np.clip(rng.exponential(3, n), 0, None),
        "composite_abl": np.clip(base * 3 + rng.normal(0, 5, n), 0, 100),
    })
    return df


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Allostatic Batting Load Analysis page."""
    st.title("Allostatic Batting Load (ABL)")
    st.caption(
        "Multi-channel cumulative decision fatigue model for hitters. "
        "Tracks pitch processing, decision conflict, swing exertion, "
        "schedule density, and travel stress as leaky integrators over "
        "the season."
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**ABL tracks cumulative cognitive and physical fatigue** across a season — not just "is he tired today" but "how much wear has accumulated without adequate recovery."

- **ABL 0-30** = fresh — decision-making is sharp, swing decisions are clean
- **ABL 30-60** = moderate load — slight uptick in chase rate and bad contact
- **ABL 60+** = high load — expect measurably worse plate discipline, especially on borderline pitches
- **5 channels tracked:** pitch processing load, decision conflict (borderline pitches), swing exertion, schedule density, travel stress
- **The key insight:** A hitter can look physically fine (barrel rate holds up) but make worse decisions (chase rate spikes) — ABL catches the cognitive fatigue that box scores miss
- **Impact:** Resting a hitter *before* they hit ABL 60 (not after they're slumping) prevents 2-3 week cold streaks that cost 5-10 runs per season
""")

    conn = get_db_connection()

    if not _ABL_AVAILABLE and not _USE_MOCK:
        st.error(
            "The `allostatic_load` analytics module could not be imported. "
            "Check that `src/analytics/allostatic_load.py` exists."
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
        st.markdown("### ABL Options")

        available_seasons = _get_available_seasons()
        default_idx = available_seasons.index(2025) if 2025 in available_seasons else 0
        season = st.selectbox(
            "Season",
            options=available_seasons,
            index=default_idx,
            key="abl_season",
        )

        min_games = st.slider(
            "Minimum Games",
            min_value=10,
            max_value=100,
            value=20,
            step=5,
            key="abl_min_games",
        )

        abl_threshold = st.slider(
            "Fatigue Threshold (ABL)",
            min_value=10,
            max_value=90,
            value=60,
            step=5,
            key="abl_threshold",
            help="Threshold for rest optimisation predictions.",
        )

    # ---- Load leaderboard ------------------------------------------------
    season = int(season)  # Ensure Python int (not numpy int32) for DuckDB
    leaderboard = _load_leaderboard(conn, season, min_games)

    if leaderboard is None or leaderboard.empty:
        st.info(
            "No ABL data available for this season. Ensure enough pitch data "
            "is loaded and batters have played at least the minimum number of games."
        )
        return

    # ---- Batter selector -------------------------------------------------
    batter_options = _build_batter_options(leaderboard)
    selected_batter = st.selectbox(
        "Select Batter",
        options=list(batter_options.keys()),
        key="abl_batter_select",
    )
    batter_id = batter_options.get(selected_batter)

    if batter_id is None:
        st.info("Please select a valid batter to view ABL data.")
        return

    # ---- Tabs ------------------------------------------------------------
    tab_timeline, tab_gauges, tab_scatter, tab_rest, tab_board = st.tabs([
        "Load Timeline",
        "Current Load",
        "Fatigue vs Performance",
        "Rest Optimisation",
        "Leaderboard",
    ])

    with tab_timeline:
        _render_load_timeline(conn, batter_id, season)

    with tab_gauges:
        _render_current_gauges(conn, batter_id, season)

    with tab_scatter:
        _render_fatigue_scatter(conn, batter_id, season)

    with tab_rest:
        _render_rest_optimisation(conn, batter_id, season, abl_threshold)

    with tab_board:
        _render_leaderboard(leaderboard)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _get_available_seasons(_conn_id: int = 0) -> list[int]:
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


def _load_leaderboard(
    conn,
    season: int,
    min_games: int,
) -> pd.DataFrame | None:
    """Load ABL leaderboard, using cache then live computation."""
    if _USE_MOCK:
        return _generate_mock_leaderboard()

    if not _ABL_AVAILABLE:
        return None

    # Try cache first
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "allostatic_load", season)
            if cached is not None:
                age_info = cache_age_display(conn, "allostatic_load", season)
                if age_info:
                    st.caption(age_info)
                return cached
        except Exception:
            pass

    try:
        with st.spinner("Computing... Run `python scripts/precompute.py` for instant loading."):
            return batch_calculate(conn, season, min_games)
    except Exception as exc:
        st.error(f"Error computing ABL leaderboard: {exc}")
        return None


@st.cache_data(ttl=3600)
def _cached_calculate_abl(batter_id: int, season: int) -> dict:
    """Cached ABL calculation for a single batter."""
    conn = get_db_connection()
    return calculate_abl(conn, batter_id, season)


@st.cache_data(ttl=3600)
def _cached_validate_against_outcomes(batter_id: int, season: int) -> dict:
    """Cached ABL validation against outcomes."""
    conn = get_db_connection()
    return validate_against_outcomes(conn, batter_id, season)


def _build_batter_options(df: pd.DataFrame) -> dict[str, int]:
    """Build batter display name -> batter_id mapping."""
    options = {}
    for _, row in df.iterrows():
        name = row.get("name") or f"ID {row['batter_id']}"
        label = f"{name} (ABL: {row['composite_abl']:.1f})"
        options[label] = int(row["batter_id"])
    return options


# ---------------------------------------------------------------------------
# Tab: Load Timeline
# ---------------------------------------------------------------------------

def _render_load_timeline(conn, batter_id: int, season: int) -> None:
    """Multi-line chart showing all 5 channels + composite ABL over the season."""
    st.subheader("Fatigue Load Timeline")

    if _USE_MOCK:
        timeline = _generate_mock_timeline()
    elif _ABL_AVAILABLE:
        result = _cached_calculate_abl(batter_id, season)
        timeline = result.get("timeline")
        if timeline is None or timeline.empty:
            st.info("No timeline data available for this batter.")
            return
    else:
        st.info("ABL module not available.")
        return

    fig = go.Figure()

    channels = [
        "pitch_processing", "decision_conflict", "swing_exertion",
        "temporal_demand", "travel_stress",
    ]

    # Add channel lines
    for ch in channels:
        if ch in timeline.columns:
            fig.add_trace(go.Scatter(
                x=timeline["game_date"],
                y=timeline[ch],
                mode="lines",
                name=_CHANNEL_LABELS.get(ch, ch),
                line=dict(color=_CHANNEL_COLORS.get(ch, "#888"), width=1.5),
                opacity=0.7,
            ))

    # Add composite ABL (thicker, white)
    if "composite_abl" in timeline.columns:
        fig.add_trace(go.Scatter(
            x=timeline["game_date"],
            y=timeline["composite_abl"],
            mode="lines",
            name="Composite ABL",
            line=dict(color=_CHANNEL_COLORS["composite_abl"], width=3),
        ))

    fig.update_layout(
        xaxis=dict(title="Date"),
        yaxis=dict(title="Load"),
        template="plotly_dark",
        height=500,
        margin=dict(l=60, r=30, t=30, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True, key="abl_timeline")


# ---------------------------------------------------------------------------
# Tab: Current Load Gauges
# ---------------------------------------------------------------------------

def _render_current_gauges(conn, batter_id: int, season: int) -> None:
    """Gauge charts for each channel + composite."""
    st.subheader("Current Fatigue Load")

    if _USE_MOCK:
        channel_loads = {
            "pitch_processing": 45.2,
            "decision_conflict": 18.7,
            "swing_exertion": 32.1,
            "temporal_demand": 5.8,
            "travel_stress": 8.3,
        }
        composite = 62.4
    elif _ABL_AVAILABLE:
        result = _cached_calculate_abl(batter_id, season)
        channel_loads = result.get("channel_loads", {})
        composite = result.get("composite_abl")
        if composite is None:
            st.info("No ABL data available for this batter.")
            return
    else:
        st.info("ABL module not available.")
        return

    # Composite gauge (full width)
    fig_composite = go.Figure(go.Indicator(
        mode="gauge+number",
        value=composite,
        title={"text": "Composite ABL", "font": {"size": 20}},
        gauge=dict(
            axis=dict(range=[0, 100]),
            bar=dict(color=_PHILLIES_RED),
            steps=[
                dict(range=[0, 33], color="#1a472a"),
                dict(range=[33, 66], color="#4a3f00"),
                dict(range=[66, 100], color="#4a0000"),
            ],
            threshold=dict(
                line=dict(color="white", width=3),
                thickness=0.8,
                value=composite,
            ),
        ),
    ))
    fig_composite.update_layout(
        template="plotly_dark",
        height=250,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    st.plotly_chart(fig_composite, use_container_width=True, key="abl_composite_gauge")

    # Channel gauges (5 columns)
    cols = st.columns(5)
    channels = [
        "pitch_processing", "decision_conflict", "swing_exertion",
        "temporal_demand", "travel_stress",
    ]

    for i, ch in enumerate(channels):
        val = channel_loads.get(ch, 0)
        # Determine a reasonable max for the gauge
        gauge_max = max(val * 1.5, 50) if val > 0 else 50

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            title={"text": _CHANNEL_LABELS.get(ch, ch), "font": {"size": 12}},
            number={"font": {"size": 18}},
            gauge=dict(
                axis=dict(range=[0, gauge_max]),
                bar=dict(color=_CHANNEL_COLORS.get(ch, "#888")),
                bgcolor="#1a1a2e",
            ),
        ))
        fig.update_layout(
            template="plotly_dark",
            height=200,
            margin=dict(l=15, r=15, t=50, b=10),
        )
        with cols[i]:
            st.plotly_chart(fig, use_container_width=True, key=f"abl_gauge_{ch}")


# ---------------------------------------------------------------------------
# Tab: Fatigue vs Performance scatter
# ---------------------------------------------------------------------------

def _render_fatigue_scatter(conn, batter_id: int, season: int) -> None:
    """Scatter of ABL vs chase rate or contact rate with regression line."""
    st.subheader("Fatigue vs Performance")

    if _USE_MOCK:
        rng = np.random.RandomState(42)
        n = 60
        abl = np.clip(rng.normal(50, 20, n), 0, 100)
        chase = 0.25 + 0.002 * abl + rng.normal(0, 0.05, n)
        scatter_df = pd.DataFrame({
            "composite_abl": abl,
            "chase_rate": chase,
        })
    elif _ABL_AVAILABLE:
        validation = _cached_validate_against_outcomes(batter_id, season)
        # Also get the per-game timeline for scatter
        result = _cached_calculate_abl(batter_id, season)
        timeline = result.get("timeline")
        if timeline is None or timeline.empty:
            st.info("No timeline data available for this batter.")
            return

        # Compute per-game chase rate
        game_pks = timeline["game_pk"].tolist()
        if not game_pks:
            st.info("No games available.")
            return

        gp_str = ", ".join(str(int(g)) for g in game_pks)
        try:
            outcome_query = f"""
                SELECT
                    game_pk,
                    SUM(CASE
                        WHEN zone IS NOT NULL AND zone > 9
                         AND description IN (
                            'swinging_strike', 'swinging_strike_blocked',
                            'foul', 'foul_tip', 'hit_into_play',
                            'hit_into_play_no_out', 'hit_into_play_score',
                            'missed_bunt', 'foul_bunt'
                        )
                        THEN 1 ELSE 0
                    END) AS o_swings,
                    SUM(CASE WHEN zone IS NOT NULL AND zone > 9
                        THEN 1 ELSE 0
                    END) AS o_pitches
                FROM pitches
                WHERE batter_id = $1
                  AND game_pk IN ({gp_str})
                GROUP BY game_pk
            """
            outcomes = conn.execute(outcome_query, [batter_id]).fetchdf()
        except Exception:
            st.info("Could not query outcome data.")
            return

        outcomes["chase_rate"] = np.where(
            outcomes["o_pitches"] > 0,
            outcomes["o_swings"] / outcomes["o_pitches"],
            np.nan,
        )
        scatter_df = timeline[["game_pk", "composite_abl"]].merge(
            outcomes[["game_pk", "chase_rate"]], on="game_pk", how="inner"
        ).dropna()

        if scatter_df.empty:
            st.info("No overlapping ABL and chase rate data.")
            return

        # Show validation stats
        if validation.get("chase_rate_corr") is not None:
            st.metric("ABL-Chase Rate Correlation", f"{validation['chase_rate_corr']:.3f}")
        if validation.get("zone_contact_corr") is not None:
            st.metric("ABL-Zone Contact Correlation", f"{validation['zone_contact_corr']:.3f}")
    else:
        st.info("ABL module not available.")
        return

    if scatter_df.empty:
        st.info("Not enough data for scatter plot.")
        return

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=scatter_df["composite_abl"],
        y=scatter_df["chase_rate"],
        mode="markers",
        marker=dict(
            size=8,
            color=_PHILLIES_RED,
            opacity=0.7,
            line=dict(width=1, color="white"),
        ),
        name="Game-level",
        hovertemplate=(
            "ABL: %{x:.1f}<br>"
            "Chase Rate: %{y:.3f}<br>"
            "<extra></extra>"
        ),
    ))

    # Add regression line
    if len(scatter_df) >= 5:
        x = scatter_df["composite_abl"].values
        y = scatter_df["chase_rate"].values
        valid = ~np.isnan(x) & ~np.isnan(y)
        if valid.sum() >= 5:
            coeffs = np.polyfit(x[valid], y[valid], 1)
            x_line = np.linspace(x[valid].min(), x[valid].max(), 50)
            y_line = np.polyval(coeffs, x_line)
            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color=_PHILLIES_LIGHT, dash="dash", width=2),
                name=f"Trend (slope={coeffs[0]:.4f})",
            ))

    fig.update_layout(
        xaxis=dict(title="Composite ABL"),
        yaxis=dict(title="Chase Rate (O-Swing%)"),
        template="plotly_dark",
        height=450,
        margin=dict(l=60, r=30, t=30, b=60),
    )
    st.plotly_chart(fig, use_container_width=True, key="abl_scatter")


# ---------------------------------------------------------------------------
# Tab: Rest Optimisation
# ---------------------------------------------------------------------------

def _render_rest_optimisation(
    conn, batter_id: int, season: int, threshold: float
) -> None:
    """Predict days until ABL drops below threshold."""
    st.subheader("Rest Optimisation")

    if _USE_MOCK:
        current_abl = 72.5
        days_needed = 4
    elif _ABL_AVAILABLE:
        result = _cached_calculate_abl(batter_id, season)
        current_abl = result.get("composite_abl")
        if current_abl is None:
            st.info("No ABL data available for this batter.")
            return
        days_needed = predict_recovery_days(current_abl, threshold)
    else:
        st.info("ABL module not available.")
        return

    col1, col2, col3 = st.columns(3)
    col1.metric("Current ABL", f"{current_abl:.1f}")
    col2.metric("Target Threshold", f"{threshold:.0f}")
    col3.metric("Rest Days Needed", f"{days_needed}")

    st.markdown("---")

    # Decay projection chart
    st.markdown("**Projected ABL Decay (Pure Rest)**")
    alpha = 0.85
    off_alpha = alpha ** 2
    days_project = min(days_needed + 10, 30)

    projected = [current_abl]
    for d in range(1, days_project + 1):
        projected.append(projected[-1] * off_alpha)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(days_project + 1)),
        y=projected,
        mode="lines+markers",
        line=dict(color=_PHILLIES_RED, width=2),
        marker=dict(size=6),
        name="Projected ABL",
    ))

    # Threshold line
    fig.add_hline(
        y=threshold,
        line=dict(color=_POSITIVE_GREEN, width=2, dash="dash"),
        annotation_text=f"Threshold ({threshold:.0f})",
        annotation_position="top right",
    )

    if days_needed > 0:
        fig.add_vline(
            x=days_needed,
            line=dict(color=_PHILLIES_LIGHT, width=1, dash="dot"),
            annotation_text=f"Day {days_needed}",
            annotation_position="top left",
        )

    fig.update_layout(
        xaxis=dict(title="Rest Days"),
        yaxis=dict(title="Composite ABL", range=[0, 100]),
        template="plotly_dark",
        height=350,
        margin=dict(l=60, r=30, t=30, b=50),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True, key="abl_rest_projection")

    # Context
    if days_needed == 0:
        st.success("This batter's ABL is already below the fatigue threshold.")
    elif days_needed <= 2:
        st.info(f"A short rest period ({days_needed} day(s)) should bring ABL below threshold.")
    else:
        st.warning(
            f"This batter needs approximately {days_needed} rest days to recover "
            f"below an ABL of {threshold:.0f}."
        )


# ---------------------------------------------------------------------------
# Tab: Leaderboard
# ---------------------------------------------------------------------------

def _render_leaderboard(df: pd.DataFrame) -> None:
    """Display ABL leaderboard with most and least fatigued batters."""
    st.subheader("ABL Leaderboard")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Most Fatigued (Highest ABL)**")
        top = df.head(15).copy()
        _render_leaderboard_table(top)

    with col2:
        st.markdown("**Least Fatigued (Lowest ABL)**")
        bottom = df.tail(15).sort_values("composite_abl", ascending=True).copy()
        _render_leaderboard_table(bottom)

    st.markdown("---")

    # Distribution histogram
    st.markdown("**ABL Distribution**")
    if len(df) >= 5:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["composite_abl"],
            nbinsx=20,
            marker_color=_PHILLIES_RED,
            opacity=0.75,
            name="ABL",
        ))
        fig.update_layout(
            xaxis=dict(title="Composite ABL"),
            yaxis=dict(title="Count"),
            template="plotly_dark",
            height=300,
            margin=dict(l=50, r=30, t=30, b=50),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key="abl_distribution")


def _render_leaderboard_table(df: pd.DataFrame) -> None:
    """Display a formatted leaderboard table."""
    display = df.copy().reset_index(drop=True)
    display.index = display.index + 1
    display.index.name = "Rank"

    display_cols = [
        "name", "composite_abl", "peak_abl", "games_played",
    ]
    available_cols = [c for c in display_cols if c in display.columns]

    st.dataframe(
        display[available_cols],
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Player"),
            "composite_abl": st.column_config.NumberColumn("ABL", format="%.1f"),
            "peak_abl": st.column_config.NumberColumn("Peak ABL", format="%.1f"),
            "games_played": st.column_config.NumberColumn("Games", format="%d"),
        },
    )
