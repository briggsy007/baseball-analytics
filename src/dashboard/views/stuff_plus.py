"""
Stuff+ leaderboard and individual pitcher analysis page.

Displays a sortable leaderboard of all pitchers ranked by Stuff+, with
drill-down into individual pitcher breakdowns: per-pitch-type grades,
feature contribution charts, and league-average comparisons.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.db_helper import get_db_connection, has_data, get_all_pitchers

# ---------------------------------------------------------------------------
# Graceful imports for analytics modules
# ---------------------------------------------------------------------------

_STUFF_AVAILABLE = False
try:
    from src.analytics.stuff_model import (
        batch_calculate_stuff_plus,
        calculate_stuff_plus,
        get_pitch_grade,
        train_stuff_model,
    )
    _STUFF_AVAILABLE = True
except ImportError:
    pass

# Pitch type metadata for colours and labels
_PITCH_COLORS: dict[str, str] = {
    "FF": "#E81828", "SI": "#FF6B35", "SL": "#002D72", "CU": "#6A0DAD",
    "CH": "#2ECC71", "FC": "#F39C12", "FS": "#1ABC9C", "KC": "#9B59B6",
    "ST": "#E74C3C", "SV": "#3498DB",
}

_PITCH_LABELS: dict[str, str] = {
    "FF": "4-Seam", "SI": "Sinker", "SL": "Slider", "CU": "Curve",
    "CH": "Change", "FC": "Cutter", "FS": "Splitter", "KC": "Knuckle-Curve",
    "ST": "Sweeper", "SV": "Slurve", "IN": "Int. Ball", "PO": "Pitchout",
    "CS": "Slow Curve", "SC": "Screwball", "FA": "Fastball", "AB": "Auto Ball",
}

_GRADE_COLORS: dict[str, str] = {
    "Elite": "#00C853",
    "Plus-Plus": "#64DD17",
    "Above Average": "#AEEA00",
    "Average": "#FFD600",
    "Below Average": "#FF9100",
    "Poor": "#FF1744",
}


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Stuff+ Analysis page."""
    st.title("Stuff+ Pitch Quality Model")
    st.caption(
        "Rate every pitch's physical nastiness on a 100-centered scale. "
        "Higher = better for the pitcher."
    )
    st.info("""
**What this shows:** Each pitch graded purely on physical characteristics (velocity, spin, movement) — independent of whether the batter got a hit.

- **100 = league average.** Every 10 points above/below = 1 standard deviation of quality
- **130+ = elite** (top ~2%), **115+ = plus-plus**, **105+ = above average**, **85-95 = below average**, **<85 = poor**
- **Why it matters:** Stuff+ predicts future performance better than ERA because it measures what the pitcher controls, not luck or defense
""")

    conn = get_db_connection()

    if not _STUFF_AVAILABLE:
        st.error(
            "The `stuff_model` analytics module could not be imported. "
            "Ensure `scikit-learn` and `joblib` are installed:\n\n"
            "```\npip install scikit-learn joblib\n```"
        )
        return

    if conn is None or not has_data(conn):
        st.warning(
            "No pitch data available. Run the data backfill pipeline first "
            "(`python scripts/backfill.py`)."
        )
        return

    # Check if model is trained
    model_trained = _check_model_exists()

    # ── Sidebar controls ─────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Stuff+ Options")
        if not model_trained:
            if st.button("Train Stuff+ Model", type="primary"):
                _train_model_ui(conn)
                st.rerun()
        else:
            if st.button("Re-train Model"):
                _train_model_ui(conn)
                st.rerun()

    if not model_trained:
        st.info(
            "The Stuff+ model has not been trained yet. "
            "Click **Train Stuff+ Model** in the sidebar to build it from "
            "your pitch data. Training typically takes 10-30 seconds."
        )
        return

    # ── Main content ─────────────────────────────────────────────────────
    tab_board, tab_pitcher = st.tabs(["Leaderboard", "Pitcher Deep Dive"])

    with tab_board:
        _render_leaderboard(conn)

    with tab_pitcher:
        _render_pitcher_analysis(conn)


# ---------------------------------------------------------------------------
# Model management helpers
# ---------------------------------------------------------------------------


def _check_model_exists() -> bool:
    """Return True if a trained Stuff+ model file exists."""
    from src.analytics.stuff_model import DEFAULT_MODEL_PATH
    return DEFAULT_MODEL_PATH.exists()


def _train_model_ui(conn) -> None:
    """Train the model and show progress / results in the UI."""
    with st.spinner("Training Stuff+ model... this may take a moment."):
        try:
            metrics = train_stuff_model(conn)
            st.success("Model trained successfully!")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Train R\u00b2", f"{metrics['r2_train']:.4f}")
            col2.metric("Test R\u00b2", f"{metrics['r2_test']:.4f}")
            col3.metric("Train RMSE", f"{metrics['rmse_train']:.4f}")
            col4.metric("Test RMSE", f"{metrics['rmse_test']:.4f}")

            st.caption(
                f"Trained on {metrics['n_train']:,} pitches, "
                f"tested on {metrics['n_test']:,} pitches."
            )

            # Feature importances
            if metrics.get("feature_importances"):
                st.markdown("**Feature Importances**")
                imp_df = pd.DataFrame(
                    list(metrics["feature_importances"].items()),
                    columns=["Feature", "Importance"],
                ).sort_values("Importance", ascending=False)
                st.dataframe(imp_df, use_container_width=True, hide_index=True)

        except Exception as exc:
            st.error(f"Training failed: {exc}")


# ---------------------------------------------------------------------------
# Leaderboard
# ---------------------------------------------------------------------------


def _render_leaderboard(conn) -> None:
    """Display the Stuff+ leaderboard for all qualifying pitchers."""
    st.subheader("Stuff+ Leaderboard")

    min_pitches = st.slider(
        "Minimum pitches to qualify", 50, 1000, 200, step=50,
        key="stuff_min_pitches",
    )

    try:
        df = batch_calculate_stuff_plus(conn, min_pitches=min_pitches)
    except FileNotFoundError:
        st.warning("Model not found. Train it from the sidebar.")
        return
    except Exception as exc:
        st.error(f"Error computing leaderboard: {exc}")
        return

    if df.empty:
        st.info("No qualifying pitchers found with the current filters.")
        return

    # Rank column
    df = df.reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"

    # Colour-code the grade column
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "overall_stuff_plus": st.column_config.NumberColumn(
                "Stuff+", format="%.1f",
            ),
            "grade": st.column_config.TextColumn("Grade"),
            "name": st.column_config.TextColumn("Pitcher"),
        },
    )

    # ── Distribution chart ───────────────────────────────────────────────
    if len(df) >= 5:
        st.markdown("**Stuff+ Distribution**")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["overall_stuff_plus"],
            nbinsx=30,
            marker_color="#E81828",
            opacity=0.75,
        ))
        fig.add_vline(
            x=100, line=dict(color="white", width=2, dash="dash"),
            annotation_text="League Avg (100)",
            annotation_position="top right",
        )
        fig.update_layout(
            xaxis=dict(title="Stuff+"),
            yaxis=dict(title="Count"),
            template="plotly_dark",
            height=350,
            margin=dict(l=50, r=30, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True, key="stuff_dist")


# ---------------------------------------------------------------------------
# Individual pitcher analysis
# ---------------------------------------------------------------------------


def _render_pitcher_analysis(conn) -> None:
    """Deep-dive view for a single pitcher."""
    st.subheader("Pitcher Deep Dive")

    pitchers = get_all_pitchers(conn)
    if not pitchers:
        # Fallback: pull distinct pitcher IDs with name lookup
        try:
            fallback = conn.execute("""
                SELECT DISTINCT pi.pitcher_id AS player_id,
                       COALESCE(pl.full_name, 'Pitcher ' || CAST(pi.pitcher_id AS VARCHAR)) AS full_name
                FROM (SELECT DISTINCT pitcher_id FROM pitches LIMIT 500) pi
                LEFT JOIN players pl ON pl.player_id = pi.pitcher_id
                ORDER BY full_name
            """).fetchdf()
            pitchers = fallback.to_dict("records")
        except Exception:
            st.info("No pitcher data available.")
            return

    pitcher_names = {
        p.get("full_name") or f"ID {p['player_id']}": p["player_id"]
        for p in pitchers
    }
    selected_name = st.selectbox(
        "Select Pitcher",
        options=sorted(pitcher_names.keys()),
        key="stuff_pitcher_select",
    )

    if not selected_name:
        return

    pitcher_id = pitcher_names[selected_name]

    try:
        result = calculate_stuff_plus(conn, pitcher_id)
    except FileNotFoundError:
        st.warning("Model not found. Train it from the sidebar.")
        return
    except Exception as exc:
        st.error(f"Error: {exc}")
        return

    if result["overall_stuff_plus"] is None:
        st.info("No scoreable pitches found for this pitcher.")
        return

    overall = result["overall_stuff_plus"]
    grade = get_pitch_grade(overall)
    grade_color = _GRADE_COLORS.get(grade, "#FFFFFF")

    # ── Header metrics ───────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Stuff+", f"{overall:.1f}")
    col2.markdown(
        f"<h3 style='color:{grade_color};'>{grade}</h3>",
        unsafe_allow_html=True,
    )
    if result["percentile_rank"] is not None:
        col3.metric("Percentile", f"{result['percentile_rank']:.0f}th")

    st.markdown("---")

    # ── Per-pitch-type breakdown ─────────────────────────────────────────
    by_pt = result.get("by_pitch_type", {})
    if by_pt:
        st.markdown("**Stuff+ by Pitch Type**")
        _render_pitch_type_bars(by_pt)
        _render_pitch_type_table(by_pt)

    st.markdown("---")

    # ── Feature contributions ────────────────────────────────────────────
    contributions = result.get("feature_contributions", {})
    if contributions:
        st.markdown("**What Drives This Pitcher's Stuff**")
        _render_feature_contributions(contributions)

    st.markdown("---")

    # ── Comparison to league average ─────────────────────────────────────
    st.markdown("**vs League Average**")
    _render_league_comparison(overall, by_pt)


def _render_pitch_type_bars(by_pt: dict) -> None:
    """Horizontal bar chart of Stuff+ by pitch type."""
    pts = sorted(by_pt.keys())
    values = [by_pt[pt]["stuff_plus"] for pt in pts]
    colors = [_PITCH_COLORS.get(pt, "#FFFFFF") for pt in pts]
    labels = [_PITCH_LABELS.get(pt, pt) for pt in pts]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition="auto",
    ))

    # Reference line at 100
    fig.add_vline(
        x=100, line=dict(color="white", width=2, dash="dash"),
        annotation_text="Avg (100)",
    )

    fig.update_layout(
        xaxis=dict(title="Stuff+", range=[min(70, min(values) - 5), max(130, max(values) + 5)]),
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=max(200, len(pts) * 50),
        margin=dict(l=100, r=30, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, key="stuff_pt_bars")


def _render_pitch_type_table(by_pt: dict) -> None:
    """Table of per-pitch-type details."""
    rows = []
    for pt in sorted(by_pt.keys()):
        info = by_pt[pt]
        rows.append({
            "Pitch": _PITCH_LABELS.get(pt, pt),
            "Code": pt,
            "Stuff+": info["stuff_plus"],
            "Grade": get_pitch_grade(info["stuff_plus"]),
            "Count": info["count"],
            "Avg Velo": info["avg_velo"],
            "Avg Spin": int(info["avg_spin"]),
            "HB (in)": info.get("avg_pfx_x", ""),
            "IVB (in)": info.get("avg_pfx_z", ""),
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


def _render_feature_contributions(contributions: dict) -> None:
    """Bar chart showing how each physical attribute contributes to Stuff+."""
    labels_map = {
        "velocity": "Velocity",
        "movement": "Movement",
        "spin": "Spin",
        "extension": "Extension",
        "release_point": "Release Point",
    }

    names = []
    values = []
    for key, label in labels_map.items():
        if key in contributions:
            names.append(label)
            values.append(contributions[key])

    if not names:
        st.caption("Feature contribution data not available.")
        return

    colors = ["#2ECC71" if v >= 0 else "#E74C3C" for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=names,
        y=values,
        marker_color=colors,
        text=[f"{v:+.1f}" for v in values],
        textposition="auto",
    ))

    fig.add_hline(y=0, line=dict(color="white", width=1))
    fig.update_layout(
        yaxis=dict(title="Stuff+ Contribution"),
        template="plotly_dark",
        height=350,
        margin=dict(l=50, r=30, t=20, b=50),
    )
    st.plotly_chart(fig, use_container_width=True, key="stuff_contributions")

    st.caption(
        "Positive values mean this feature adds to the pitcher's Stuff+; "
        "negative values mean it detracts."
    )


def _render_league_comparison(overall: float, by_pt: dict) -> None:
    """Show a simple gauge / comparison to league average."""
    col1, col2 = st.columns(2)

    with col1:
        # Gauge-like indicator
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall,
            title={"text": "Overall Stuff+"},
            gauge={
                "axis": {"range": [60, 140]},
                "bar": {"color": "#E81828"},
                "steps": [
                    {"range": [60, 85], "color": "rgba(255, 23, 68, 0.2)"},
                    {"range": [85, 95], "color": "rgba(255, 145, 0, 0.2)"},
                    {"range": [95, 105], "color": "rgba(255, 214, 0, 0.2)"},
                    {"range": [105, 115], "color": "rgba(174, 234, 0, 0.2)"},
                    {"range": [115, 130], "color": "rgba(100, 221, 23, 0.2)"},
                    {"range": [130, 140], "color": "rgba(0, 200, 83, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.8,
                    "value": 100,
                },
            },
        ))
        fig.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=30, r=30, t=50, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, key="stuff_gauge")

    with col2:
        # Quick stats summary
        st.markdown("**Pitch Arsenal Summary**")
        n_pitches = sum(info["count"] for info in by_pt.values())
        n_types = len(by_pt)
        best_pt = max(by_pt.items(), key=lambda x: x[1]["stuff_plus"]) if by_pt else None
        worst_pt = min(by_pt.items(), key=lambda x: x[1]["stuff_plus"]) if by_pt else None

        st.markdown(f"- **Pitch types:** {n_types}")
        st.markdown(f"- **Pitches scored:** {n_pitches:,}")
        if best_pt:
            st.markdown(
                f"- **Best pitch:** {_PITCH_LABELS.get(best_pt[0], best_pt[0])} "
                f"({best_pt[1]['stuff_plus']:.1f} Stuff+)"
            )
        if worst_pt:
            st.markdown(
                f"- **Weakest pitch:** {_PITCH_LABELS.get(worst_pt[0], worst_pt[0])} "
                f"({worst_pt[1]['stuff_plus']:.1f} Stuff+)"
            )
        st.markdown(
            f"- **Overall vs avg:** {overall - 100:+.1f} Stuff+ points"
        )
