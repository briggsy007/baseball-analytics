"""
Kinetic Half-Life (K1/2) dashboard view.

Visualizes pitcher stamina decay using pharmacokinetic-inspired
exponential decay curves.  Shows per-game Stuff Concentration decay,
per-pitch-type breakdown, durability leaderboard, and game-specific
actual-vs-predicted overlays.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.db_helper import get_db_connection, has_data, get_all_pitchers

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_KHL_AVAILABLE = False
try:
    from src.analytics.kinetic_half_life import (
        batch_calculate,
        calculate_half_life,
        get_game_decay_data,
        predict_game_decay,
    )
    _KHL_AVAILABLE = True
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
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

# Pitch type colour map
_PITCH_COLORS: dict[str, str] = {
    "FF": "#E81828", "SI": "#FF6B35", "SL": "#002D72", "CU": "#6A0DAD",
    "CH": "#2ECC71", "FC": "#F39C12", "FS": "#1ABC9C", "KC": "#9B59B6",
    "ST": "#E74C3C", "SV": "#3498DB",
}

_PITCH_LABELS: dict[str, str] = {
    "FF": "4-Seam", "SI": "Sinker", "SL": "Slider", "CU": "Curve",
    "CH": "Change", "FC": "Cutter", "FS": "Splitter", "KC": "Knuckle-Curve",
    "ST": "Sweeper", "SV": "Slurve", "CS": "Slow Curve", "FA": "Fastball",
}


# ---------------------------------------------------------------------------
# Cached computation wrappers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=600, show_spinner=False)
def _cached_calculate_half_life(pitcher_id: int, season: int | None = None) -> dict:
    """Cached wrapper around calculate_half_life (avoids recomputing on every rerun)."""
    conn = get_db_connection()
    return calculate_half_life(conn, pitcher_id, season)


@st.cache_data(ttl=600, show_spinner=False)
def _cached_batch_calculate(season: int | None = None, min_pitches: int = 500) -> pd.DataFrame:
    """Cached wrapper around batch_calculate (leaderboard)."""
    conn = get_db_connection()
    return batch_calculate(conn, season=season, min_pitches=min_pitches)


def _build_prediction_from_result(result: dict) -> dict:
    """Build a prediction dict from an already-computed calculate_half_life result.

    Avoids calling predict_game_decay (which internally re-calls
    calculate_half_life), eliminating redundant computation.
    """
    import math as _math

    if result["overall_half_life"] is None:
        return {
            "pitcher_id": result.get("pitcher_id"),
            "pitch_numbers": [],
            "predicted_stuff": [],
            "half_life": None,
            "lambda": None,
            "peak": None,
        }

    lam = result["overall_lambda"]
    peaks = [
        g["peak"] for g in result.get("game_curves", [])
        if g.get("peak") is not None
    ]
    peak = float(np.median(peaks)) if peaks else 1.0

    pitch_numbers = list(range(0, 110))
    predicted = [
        round(float(peak * _math.exp(-lam * n)), 4) for n in pitch_numbers
    ]
    half_life = round(_math.log(2) / lam, 1) if lam > 1e-10 else None

    return {
        "pitcher_id": result.get("pitcher_id"),
        "pitch_numbers": pitch_numbers,
        "predicted_stuff": predicted,
        "half_life": half_life,
        "lambda": round(lam, 6),
        "peak": round(peak, 4),
    }


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Kinetic Half-Life analysis page."""
    st.title("Kinetic Half-Life (K\u00bd) -- Pitcher Stamina Decay")
    st.caption(
        "Pharmacokinetic-inspired model of how a pitcher's raw stuff "
        "decays over the course of a game.  Higher K\u00bd = more durable."
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**K\u00bd is the number of pitches until a starter's stuff quality drops by 50%** — like a drug's half-life but for pitch effectiveness.

- **K\u00bd > 90 pitches** = elite durability (can go deep into games with quality stuff)
- **K\u00bd 70–90** = average durability (typical starter)
- **K\u00bd < 70** = stuff degrades quickly (better suited as bulk reliever or needs early hook)
- **Per-pitch-type K\u00bd** reveals which pitch dies first — a starter whose slider cliff comes at pitch 55 should stop throwing it in the 6th inning
- **Impact:** Pulling a starter 10 pitches before their K\u00bd cliff prevents the "one batter too many" blowup that costs 2-3 wins per season
""")

    if not _KHL_AVAILABLE:
        st.error(
            "The `kinetic_half_life` analytics module could not be imported. "
            "Ensure `scipy` is installed:\n\n```\npip install scipy\n```"
        )
        return

    if not _PLOTLY_AVAILABLE:
        st.error("Plotly is required for charts. Install with `pip install plotly`.")
        return

    conn = get_db_connection()
    if conn is None or not has_data(conn):
        st.warning(
            "No pitch data available.  Run the data backfill pipeline first "
            "(`python scripts/backfill.py`)."
        )
        return

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_pitcher, tab_leaderboard, tab_game = st.tabs([
        "Pitcher Analysis",
        "Durability Leaderboard",
        "Game Replay",
    ])

    with tab_pitcher:
        _render_pitcher_analysis(conn)

    with tab_leaderboard:
        _render_leaderboard(conn)

    with tab_game:
        _render_game_view(conn)


# ---------------------------------------------------------------------------
# Pitcher analysis tab
# ---------------------------------------------------------------------------


def _render_pitcher_analysis(conn) -> None:
    """Individual pitcher K1/2 analysis with decay curve and per-type breakdown."""
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
        key="khl_pitcher_select",
    )
    if not selected_name:
        return

    pitcher_id = pitcher_names[selected_name]

    with st.spinner("Computing K\u00bd..."):
        try:
            result = _cached_calculate_half_life(pitcher_id)
        except Exception as exc:
            st.error(f"Error computing K\u00bd: {exc}")
            return

    # ── Key metrics row ───────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    hl = result.get("overall_half_life")
    col1.metric("Overall K\u00bd", f"{hl:.0f} pitches" if hl else "N/A")
    col2.metric("Games Fitted", result.get("games_fitted", 0))

    lam = result.get("overall_lambda")
    col3.metric("Decay Rate (\u03bb)", f"{lam:.5f}" if lam else "N/A")

    if hl is None:
        st.info(
            "Not enough qualifying game appearances (need 50+ pitches per game) "
            "to fit a decay curve for this pitcher."
        )
        return

    # ── "Which pitch dies first" callout ──────────────────────────────────
    per_pt = result.get("per_pitch_type", {})
    pts_with_hl: dict = {}
    if per_pt:
        pts_with_hl = {pt: d for pt, d in per_pt.items() if d.get("half_life") is not None}
        if pts_with_hl:
            worst = min(pts_with_hl.items(), key=lambda x: x[1]["half_life"])
            best = max(pts_with_hl.items(), key=lambda x: x[1]["half_life"])
            st.info(
                f"**Weakest stamina:** {_PITCH_LABELS.get(worst[0], worst[0])} "
                f"(K\u00bd = {worst[1]['half_life']:.0f} pitches)  |  "
                f"**Most durable:** {_PITCH_LABELS.get(best[0], best[0])} "
                f"(K\u00bd = {best[1]['half_life']:.0f} pitches)"
            )

    st.markdown("---")

    # ── Predicted decay curve ─────────────────────────────────────────────
    st.markdown("**Projected Stuff Decay (Typical Start)**")
    prediction = _build_prediction_from_result(result)
    if prediction["predicted_stuff"]:
        _render_decay_chart(prediction, hl)

    st.markdown("---")

    # ── Per-pitch-type breakdown ──────────────────────────────────────────
    if pts_with_hl:
        st.markdown("**Per-Pitch-Type Decay Curves**")
        _render_per_type_chart(pts_with_hl)


def _render_decay_chart(prediction: dict, half_life: float | None) -> None:
    """Plotly line chart showing predicted stuff concentration vs pitch number."""
    x = prediction["pitch_numbers"]
    y = prediction["predicted_stuff"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color="#E81828", width=3),
        name="Predicted Stuff",
    ))

    # Mark the half-life point
    if half_life is not None and half_life < max(x):
        import math
        peak = prediction.get("peak", y[0])
        hl_y = peak * 0.5
        fig.add_trace(go.Scatter(
            x=[half_life],
            y=[hl_y],
            mode="markers+text",
            marker=dict(size=12, color="#FFD600", symbol="diamond"),
            text=[f"K\u00bd = {half_life:.0f}"],
            textposition="top right",
            textfont=dict(color="#FFD600", size=13),
            name="Half-Life",
        ))
        fig.add_vline(
            x=half_life,
            line=dict(color="#FFD600", width=1, dash="dash"),
        )

    fig.update_layout(
        xaxis=dict(title="Pitch Number"),
        yaxis=dict(title="Stuff Concentration"),
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=30, t=30, b=50),
        showlegend=True,
        legend=dict(x=0.75, y=0.95),
    )
    st.plotly_chart(fig, use_container_width=True, key="khl_decay_curve")


def _render_per_type_chart(pts_with_hl: dict) -> None:
    """Side-by-side decay curves for each pitch type."""
    fig = go.Figure()

    for pt, pt_data in sorted(pts_with_hl.items()):
        hl = pt_data["half_life"]
        lam = pt_data["lambda"]
        if lam is None or lam <= 0:
            continue
        x = list(range(0, 120))
        y = [1.0 * np.exp(-lam * n) for n in x]
        color = _PITCH_COLORS.get(pt, "#FFFFFF")
        label = _PITCH_LABELS.get(pt, pt)

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode="lines",
            line=dict(color=color, width=2.5),
            name=f"{label} (K\u00bd={hl:.0f})",
        ))

    fig.update_layout(
        xaxis=dict(title="Pitch Number"),
        yaxis=dict(title="Relative Stuff (Normalized)"),
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=30, t=30, b=50),
        legend=dict(x=0.65, y=0.95),
    )
    st.plotly_chart(fig, use_container_width=True, key="khl_per_type")

    # Table summary
    rows = []
    for pt in sorted(pts_with_hl.keys()):
        d = pts_with_hl[pt]
        rows.append({
            "Pitch": _PITCH_LABELS.get(pt, pt),
            "Code": pt,
            "K\u00bd (pitches)": d["half_life"],
            "\u03bb": d["lambda"],
            "Games Fitted": d["games_fitted"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Leaderboard tab
# ---------------------------------------------------------------------------


def _render_leaderboard(conn) -> None:
    """Display top starters by K1/2 (durability ranking)."""
    st.subheader("K\u00bd Durability Leaderboard")
    st.caption("Higher K\u00bd = pitcher maintains stuff longer into a game.")

    min_pitches = st.slider(
        "Minimum pitches to qualify",
        50, 2000, 500, step=50,
        key="khl_min_pitches",
    )

    # Try cache first
    df = None
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "kinetic_half_life", None)
            if cached is not None:
                df = cached
                age_info = cache_age_display(conn, "kinetic_half_life", None)
                if age_info:
                    st.caption(age_info)
        except Exception:
            pass

    if df is None:
        with st.spinner("Computing... Run `python scripts/precompute.py` for instant loading."):
            try:
                df = _cached_batch_calculate(min_pitches=min_pitches)
            except Exception as exc:
                st.error(f"Error computing leaderboard: {exc}")
                return

    if df.empty:
        st.info("No qualifying pitchers found with the current filters.")
        return

    df = df.reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"

    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "overall_half_life": st.column_config.NumberColumn(
                "K\u00bd (pitches)", format="%.0f",
            ),
            "overall_lambda": st.column_config.NumberColumn(
                "\u03bb", format="%.5f",
            ),
            "games_fitted": st.column_config.NumberColumn("Games"),
            "name": st.column_config.TextColumn("Pitcher"),
        },
    )

    # Distribution chart
    if len(df) >= 5:
        st.markdown("**K\u00bd Distribution**")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["overall_half_life"],
            nbinsx=25,
            marker_color="#E81828",
            opacity=0.75,
        ))
        median_hl = df["overall_half_life"].median()
        fig.add_vline(
            x=median_hl,
            line=dict(color="white", width=2, dash="dash"),
            annotation_text=f"Median ({median_hl:.0f})",
            annotation_position="top right",
        )
        fig.update_layout(
            xaxis=dict(title="K\u00bd (pitches)"),
            yaxis=dict(title="Count"),
            template="plotly_dark",
            height=350,
            margin=dict(l=50, r=30, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True, key="khl_dist")


# ---------------------------------------------------------------------------
# Game replay tab
# ---------------------------------------------------------------------------


def _render_game_view(conn) -> None:
    """Select a specific game and see actual vs predicted decay."""
    st.subheader("Game-Specific Decay Analysis")

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
        key="khl_game_pitcher_select",
    )
    if not selected_name:
        return

    pitcher_id = pitcher_names[selected_name]

    # Get list of games for this pitcher
    try:
        games_df = conn.execute("""
            SELECT DISTINCT game_pk, game_date, COUNT(*) AS pitches
            FROM pitches
            WHERE pitcher_id = $1
              AND release_speed IS NOT NULL
            GROUP BY game_pk, game_date
            HAVING COUNT(*) >= 50
            ORDER BY game_date DESC
            LIMIT 50
        """, [pitcher_id]).fetchdf()
    except Exception:
        games_df = pd.DataFrame()

    if games_df.empty:
        st.info("No qualifying games (50+ pitches) found for this pitcher.")
        return

    game_options = {
        f"{row['game_date']} (pk={row['game_pk']}, {row['pitches']} pitches)": int(row["game_pk"])
        for _, row in games_df.iterrows()
    }
    selected_game = st.selectbox(
        "Select Game",
        options=list(game_options.keys()),
        key="khl_game_select",
    )
    if not selected_game:
        return

    game_pk = game_options[selected_game]

    with st.spinner("Analyzing game..."):
        try:
            data = get_game_decay_data(conn, pitcher_id, game_pk)
        except Exception as exc:
            st.error(f"Error: {exc}")
            return

    if not data["pitch_numbers"]:
        st.info("Not enough data for this game.")
        return

    # ── Chart: actual vs fitted ───────────────────────────────────────────
    fig = go.Figure()

    # Raw stuff concentration (light, thin)
    fig.add_trace(go.Scatter(
        x=data["pitch_numbers"],
        y=data["raw_stuff"],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.2)", width=1),
        name="Raw Stuff",
    ))

    # Rolling average (solid)
    fig.add_trace(go.Scatter(
        x=data["pitch_numbers"],
        y=data["rolling_stuff"],
        mode="lines",
        line=dict(color="#3498DB", width=2.5),
        name="Rolling Avg (10-pitch)",
    ))

    # Fitted exponential
    if data["fitted_stuff"]:
        fig.add_trace(go.Scatter(
            x=data["pitch_numbers"],
            y=data["fitted_stuff"],
            mode="lines",
            line=dict(color="#E81828", width=2.5, dash="dash"),
            name="Fitted Decay",
        ))

    fit = data.get("fit")
    if fit and fit.get("half_life") is not None:
        fig.add_vline(
            x=fit["half_life"],
            line=dict(color="#FFD600", width=1, dash="dot"),
            annotation_text=f"K\u00bd = {fit['half_life']:.0f}",
            annotation_position="top left",
        )

    fig.update_layout(
        xaxis=dict(title="Pitch Number"),
        yaxis=dict(title="Stuff Concentration"),
        template="plotly_dark",
        height=450,
        margin=dict(l=50, r=30, t=30, b=50),
        legend=dict(x=0.6, y=0.95),
    )
    st.plotly_chart(fig, use_container_width=True, key="khl_game_chart")

    # Fit metrics
    if fit:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("K\u00bd", f"{fit['half_life']:.0f}" if fit.get("half_life") else "N/A")
        col2.metric("\u03bb", f"{fit['lambda']:.5f}" if fit.get("lambda") else "N/A")
        col3.metric("R\u00b2", f"{fit['r_squared']:.3f}" if fit.get("r_squared") else "N/A")
        col4.metric("Pitches", fit.get("n_points", 0))
