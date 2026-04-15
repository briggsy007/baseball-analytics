"""
Pitch Implied Volatility Surface (PIVS) dashboard view.

Visualises a pitcher's outcome-entropy surface across a 5x5 zone grid and
12 ball-strike count states.  Includes 3D surface plots, heatmaps, vol smile
and term structure line charts, a leaderboard, and scouting callouts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.db_helper import get_db_connection, has_data, get_all_pitchers

# ── Graceful imports ─────────────────────────────────────────────────────────

_PIVS_AVAILABLE = False
try:
    from src.analytics.volatility_surface import (
        ALL_COUNT_STATES,
        MIN_PITCHES_PER_CELL,
        N_COUNTS,
        N_OUTCOMES,
        N_ZONE_CELLS,
        N_ZONE_COLS,
        N_ZONE_ROWS,
        OOZ_INDEX,
        OUTCOME_CATEGORIES,
        ZONE_X_EDGES,
        ZONE_Z_EDGES,
        batch_calculate,
        calculate_volatility_surface,
        compare_surfaces,
    )
    _PIVS_AVAILABLE = True
except ImportError:
    pass

_PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    pass

_CACHE_AVAILABLE = False
try:
    from src.dashboard.cache_reader import get_cached_leaderboard, cache_age_display
    _CACHE_AVAILABLE = True
except Exception:
    pass


# ── Page entry point ─────────────────────────────────────────────────────────


def render() -> None:
    """Render the Pitch Implied Volatility Surface page."""
    st.title("Pitch Implied Volatility Surface")
    st.caption(
        "Options-pricing-inspired view of pitch outcome uncertainty. "
        "Maps Shannon entropy across zone and count to reveal where a "
        "pitcher is most unpredictable."
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**The volatility surface maps how unpredictable a pitcher's outcomes are** across every zone-count combination — like an options trader's implied vol surface.

- **High entropy (red)** = unpredictable outcomes → good for the pitcher inside the zone (hitters can't sit on one outcome), bad outside (command issues)
- **Low entropy (blue)** = predictable outcomes → exploitable if hitters identify the pattern ("he always gets a chase swing here")
- **Vol Smile** shows if a pitcher is more unpredictable at the zone edges (good — deceptive) or the middle (bad — hittable)
- **Vol Term Structure** reveals if a pitcher becomes more predictable as counts deepen (bad — hitters can narrow their approach in full counts)
- **Impact:** Identifying the 2-3 zone-count cells where a pitcher is most predictable gives hitters a concrete plan of attack
""")

    if not _PIVS_AVAILABLE:
        st.error(
            "The `volatility_surface` analytics module could not be imported. "
            "Ensure `scipy` and `numpy` are installed:\n\n"
            "```\npip install scipy numpy\n```"
        )
        return

    if not _PLOTLY_AVAILABLE:
        st.error(
            "Plotly is required for this page. Install with:\n\n"
            "```\npip install plotly\n```"
        )
        return

    conn = get_db_connection()
    if conn is None or not has_data(conn):
        st.warning(
            "No pitch data available. Run the data backfill pipeline first "
            "(`python scripts/backfill.py`)."
        )
        return

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_pitcher, tab_board = st.tabs(
        ["Pitcher Analysis", "Volatility Leaderboard"]
    )

    with tab_pitcher:
        _render_pitcher_analysis(conn)

    with tab_board:
        _render_leaderboard(conn)


# ── Pitcher analysis ─────────────────────────────────────────────────────────


def _render_pitcher_analysis(conn) -> None:
    """Single-pitcher PIVS deep dive."""
    st.subheader("Pitcher Volatility Surface")

    # Pitcher selector
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

    col_sel, col_season = st.columns([3, 1])
    with col_sel:
        selected_name = st.selectbox(
            "Select Pitcher",
            options=sorted(pitcher_names.keys()),
            key="pivs_pitcher_select",
        )
    with col_season:
        season = st.number_input(
            "Season (0 = all)",
            min_value=0,
            max_value=2030,
            value=0,
            key="pivs_season",
        )

    if not selected_name:
        return

    pitcher_id = pitcher_names[selected_name]
    season_val = int(season) if season > 0 else None

    with st.spinner("Computing volatility surface..."):
        result = calculate_volatility_surface(conn, pitcher_id, season_val)

    if result["n_pitches"] == 0:
        st.info("No scoreable pitches found for this pitcher.")
        return

    # ── Header metrics ───────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Pitches Analyzed", f"{result['n_pitches']:,}")
    col2.metric("Overall Volatility", f"{result['overall_vol']:.3f}")

    # Most exploitable cell
    exploitable = _find_most_exploitable(result)
    if exploitable:
        col3.metric(
            "Most Exploitable",
            exploitable["label"],
            help=f"Lowest entropy cell: {exploitable['entropy']:.3f}",
        )

    st.markdown("---")

    # ── Visualizations ───────────────────────────────────────────────────
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "3D Surface", "Heatmap", "Vol Smile", "Term Structure",
    ])

    with viz_tab1:
        _render_3d_surface(result)

    with viz_tab2:
        _render_heatmap(result)

    with viz_tab3:
        _render_vol_smile(result)

    with viz_tab4:
        _render_term_structure(result)

    # ── Scouting callout ─────────────────────────────────────────────────
    st.markdown("---")
    _render_scouting_callout(result)


# ── 3D surface plot ──────────────────────────────────────────────────────────


def _render_3d_surface(result: dict) -> None:
    """3D Plotly surface: x=zone_col, y=count, z=entropy."""
    surface = result["surface_smooth"]

    # Extract just the 5x5 in-zone grid, averaged across rows for each column
    # to create a (5 zone_cols x 12 counts) matrix
    zone_count_matrix = np.zeros((N_ZONE_COLS, N_COUNTS))
    for col in range(N_ZONE_COLS):
        for ci in range(N_COUNTS):
            vals = [surface[row * N_ZONE_COLS + col, ci] for row in range(N_ZONE_ROWS)]
            zone_count_matrix[col, ci] = np.mean(vals)

    x_labels = [f"Col {i}" for i in range(N_ZONE_COLS)]
    y_labels = ALL_COUNT_STATES

    fig = go.Figure(data=[go.Surface(
        z=zone_count_matrix.T,
        x=list(range(N_ZONE_COLS)),
        y=list(range(N_COUNTS)),
        colorscale="RdBu_r",
        colorbar=dict(title="Entropy"),
        hovertemplate=(
            "Zone Col: %{x}<br>"
            "Count: %{customdata}<br>"
            "Entropy: %{z:.3f}<extra></extra>"
        ),
        customdata=np.array([[ALL_COUNT_STATES[ci] for _ in range(N_ZONE_COLS)]
                             for ci in range(N_COUNTS)]),
    )])

    fig.update_layout(
        scene=dict(
            xaxis=dict(title="Zone Column", tickvals=list(range(N_ZONE_COLS)),
                       ticktext=[f"Col {i}" for i in range(N_ZONE_COLS)]),
            yaxis=dict(title="Count", tickvals=list(range(N_COUNTS)),
                       ticktext=ALL_COUNT_STATES),
            zaxis=dict(title="Entropy (nats)"),
        ),
        template="plotly_dark",
        height=550,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig, use_container_width=True, key="pivs_3d")


# ── Heatmap ──────────────────────────────────────────────────────────────────


def _render_heatmap(result: dict) -> None:
    """2D heatmap: 5x5 zone grid x 12 counts."""
    surface = result["surface_smooth"]
    pitch_counts = result["pitch_counts"]

    # Let user select a count
    selected_count = st.selectbox(
        "Select count state",
        ALL_COUNT_STATES,
        key="pivs_heatmap_count",
    )
    count_idx = ALL_COUNT_STATES.index(selected_count)

    # Build 5x5 grid for the selected count
    grid = np.zeros((N_ZONE_ROWS, N_ZONE_COLS))
    hover_text = [['' for _ in range(N_ZONE_COLS)] for _ in range(N_ZONE_ROWS)]

    for row in range(N_ZONE_ROWS):
        for col in range(N_ZONE_COLS):
            zi = row * N_ZONE_COLS + col
            grid[row, col] = surface[zi, count_idx]
            n = pitch_counts[zi, count_idx]

            # Build hover with outcome distribution
            key = f"{zi}-{selected_count}"
            dist_str = ""
            if key in result.get("outcome_distributions", {}):
                dist = result["outcome_distributions"][key]
                dist_str = "<br>".join(
                    f"  {cat}: {prob:.1%}" for cat, prob in dist.items()
                )

            hover_text[row][col] = (
                f"Zone ({col}, {row})<br>"
                f"Count: {selected_count}<br>"
                f"Entropy: {grid[row, col]:.3f}<br>"
                f"Pitches: {n}<br>"
                f"{dist_str}"
            )

    # Zone labels
    x_labels = [
        f"{ZONE_X_EDGES[i]:.1f} to {ZONE_X_EDGES[i+1]:.1f}"
        for i in range(N_ZONE_COLS)
    ]
    y_labels = [
        f"{ZONE_Z_EDGES[i]:.1f} to {ZONE_Z_EDGES[i+1]:.1f}"
        for i in range(N_ZONE_ROWS)
    ]

    fig = go.Figure(data=go.Heatmap(
        z=grid,
        x=x_labels,
        y=y_labels,
        colorscale="RdBu_r",
        colorbar=dict(title="Entropy"),
        text=hover_text,
        hoverinfo="text",
        zmin=0,
        zmax=float(np.log(N_OUTCOMES)),
    ))

    fig.update_layout(
        xaxis=dict(title="Plate X (ft)"),
        yaxis=dict(title="Plate Z (ft)", autorange="reversed"),
        template="plotly_dark",
        height=450,
        margin=dict(l=80, r=30, t=30, b=60),
    )

    # Add strike zone outline (approximately cols 1-3, rows 1-3)
    fig.add_shape(
        type="rect",
        x0=x_labels[1], x1=x_labels[3],
        y0=y_labels[1], y1=y_labels[3],
        line=dict(color="white", width=2, dash="dash"),
    )

    st.plotly_chart(fig, use_container_width=True, key="pivs_heatmap")

    # OOZ entropy
    ooz_entropy = surface[OOZ_INDEX, count_idx]
    st.caption(
        f"Out-of-zone entropy at {selected_count} count: **{ooz_entropy:.3f}**"
    )


# ── Vol smile chart ──────────────────────────────────────────────────────────


def _render_vol_smile(result: dict) -> None:
    """Line chart of entropy across zones at a selected count."""
    vol_smile = result.get("vol_smile", {})
    if not vol_smile:
        st.info("Vol smile data not available.")
        return

    # Let user select count(s)
    selected_counts = st.multiselect(
        "Select count states to compare",
        ALL_COUNT_STATES,
        default=["0-0", "0-2", "3-0", "3-2"],
        key="pivs_smile_counts",
    )

    if not selected_counts:
        st.info("Select at least one count state.")
        return

    x_labels = [
        f"{ZONE_X_EDGES[i]:.1f} to {ZONE_X_EDGES[i+1]:.1f}"
        for i in range(N_ZONE_COLS)
    ]

    fig = go.Figure()
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
              "#DDA0DD", "#FF8C00", "#00CED1", "#FF69B4", "#32CD32",
              "#BA55D3", "#FF4500"]

    for i, cs in enumerate(selected_counts):
        if cs in vol_smile:
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=x_labels,
                y=vol_smile[cs],
                mode="lines+markers",
                name=f"Count {cs}",
                line=dict(color=color, width=2),
                marker=dict(size=8),
            ))

    fig.update_layout(
        xaxis=dict(title="Zone Column (Plate X)"),
        yaxis=dict(title="Entropy (nats)", rangemode="tozero"),
        template="plotly_dark",
        height=400,
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=60, r=30, t=30, b=60),
    )
    st.plotly_chart(fig, use_container_width=True, key="pivs_smile")
    st.caption(
        "The **vol smile** shows how outcome uncertainty varies across "
        "horizontal zones at a fixed count. A 'smile' shape (edges higher "
        "than center) suggests the pitcher is harder to read at the edges."
    )


# ── Vol term structure chart ─────────────────────────────────────────────────


def _render_term_structure(result: dict) -> None:
    """Line chart of entropy across counts at a fixed zone."""
    vol_ts = result.get("vol_term_structure", {})
    if not vol_ts:
        st.info("Vol term structure data not available.")
        return

    fig = go.Figure()

    for zone_idx, values in vol_ts.items():
        label = "Center Zone" if zone_idx != OOZ_INDEX else "Out of Zone"
        color = "#4ECDC4" if zone_idx != OOZ_INDEX else "#FF6B6B"
        fig.add_trace(go.Scatter(
            x=ALL_COUNT_STATES,
            y=values,
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=8),
        ))

    fig.update_layout(
        xaxis=dict(title="Count State"),
        yaxis=dict(title="Entropy (nats)", rangemode="tozero"),
        template="plotly_dark",
        height=400,
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=60, r=30, t=30, b=60),
    )
    st.plotly_chart(fig, use_container_width=True, key="pivs_term_structure")
    st.caption(
        "The **vol term structure** shows how outcome uncertainty evolves "
        "across count states for a fixed zone. Analogous to how volatility "
        "changes across maturities in options pricing."
    )


# ── Scouting callout ─────────────────────────────────────────────────────────


def _find_most_exploitable(result: dict) -> dict | None:
    """Find the count-zone combination with the lowest entropy (most predictable)."""
    surface = result["surface_smooth"]
    pitch_counts = result["pitch_counts"]

    # Only consider cells with actual pitches
    min_entropy = float("inf")
    best_zone = None
    best_count = None

    for zi in range(N_ZONE_CELLS):
        for ci in range(N_COUNTS):
            if pitch_counts[zi, ci] >= MIN_PITCHES_PER_CELL:
                if surface[zi, ci] < min_entropy:
                    min_entropy = surface[zi, ci]
                    best_zone = zi
                    best_count = ci

    if best_zone is None:
        return None

    if best_zone == OOZ_INDEX:
        zone_label = "Out of Zone"
    else:
        row = best_zone // N_ZONE_COLS
        col = best_zone % N_ZONE_COLS
        zone_label = f"Zone ({col}, {row})"

    count_label = ALL_COUNT_STATES[best_count]

    return {
        "label": f"{zone_label} @ {count_label}",
        "entropy": min_entropy,
        "zone_idx": best_zone,
        "count_idx": best_count,
    }


def _render_scouting_callout(result: dict) -> None:
    """Highlight the most exploitable count-zone combination."""
    st.subheader("Scouting Report")

    exploitable = _find_most_exploitable(result)
    if exploitable is None:
        st.info("Not enough data for a scouting callout.")
        return

    # Get outcome distribution for this cell
    key = f"{exploitable['zone_idx']}-{ALL_COUNT_STATES[exploitable['count_idx']]}"
    dist = result.get("outcome_distributions", {}).get(key, {})

    st.markdown(
        f"**Most Exploitable Spot:** {exploitable['label']} "
        f"(entropy = {exploitable['entropy']:.3f})"
    )

    if dist:
        dominant = max(dist.items(), key=lambda x: x[1])
        st.markdown(
            f"At this location and count, the most likely outcome is "
            f"**{dominant[0]}** ({dominant[1]:.1%} probability). "
            f"A hitter can anticipate this result with high confidence."
        )

        # Show distribution as a bar chart
        dist_df = pd.DataFrame([
            {"Outcome": k, "Probability": v}
            for k, v in sorted(dist.items(), key=lambda x: -x[1])
        ])
        fig = go.Figure(go.Bar(
            x=dist_df["Outcome"],
            y=dist_df["Probability"],
            marker_color=[
                "#FF6B6B" if o == dominant[0] else "#4A4A4A"
                for o in dist_df["Outcome"]
            ],
            text=[f"{v:.1%}" for v in dist_df["Probability"]],
            textposition="auto",
        ))
        fig.update_layout(
            yaxis=dict(title="Probability", tickformat=".0%"),
            template="plotly_dark",
            height=300,
            margin=dict(l=60, r=30, t=20, b=60),
        )
        st.plotly_chart(fig, use_container_width=True, key="pivs_scouting_bar")

    # Vol skew summary
    vol_skew = result.get("vol_skew", {})
    if vol_skew:
        st.markdown("**Volatility Skew Summary**")
        skew_cols = st.columns(len(vol_skew))
        for i, (cs, skew_val) in enumerate(vol_skew.items()):
            with skew_cols[i]:
                direction = "Edges" if skew_val > 0 else "Center"
                st.metric(
                    f"Skew @ {cs}",
                    f"{skew_val:+.3f}",
                    help=f"Positive = edges more volatile; negative = center more volatile",
                )


# ── Leaderboard ──────────────────────────────────────────────────────────────


def _render_leaderboard(conn) -> None:
    """Display the volatility leaderboard: most/least volatile pitchers."""
    st.subheader("Volatility Leaderboard")

    min_pitches = st.slider(
        "Minimum pitches to qualify",
        50, 2000, 300, step=50,
        key="pivs_lb_min_pitches",
    )

    # Try cache first
    df = None
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "volatility_surface", None)
            if cached is not None:
                df = cached
                age_info = cache_age_display(conn, "volatility_surface", None)
                if age_info:
                    st.caption(age_info)
        except Exception:
            pass

    if df is None:
        with st.spinner("Computing... Run `python scripts/precompute.py` for instant loading."):
            try:
                df = batch_calculate(conn, min_pitches=min_pitches)
            except Exception as exc:
                st.error(f"Error computing leaderboard: {exc}")
                return

    if df.empty:
        st.info("No qualifying pitchers found with the current filters.")
        return

    # Show top and bottom
    col_high, col_low = st.columns(2)

    with col_high:
        st.markdown("**Most Volatile (Unpredictable)**")
        top_df = df.head(15).copy()
        top_df.index = range(1, len(top_df) + 1)
        top_df.index.name = "Rank"
        st.dataframe(
            top_df[["name", "n_pitches", "overall_vol"]],
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Pitcher"),
                "n_pitches": st.column_config.NumberColumn("Pitches"),
                "overall_vol": st.column_config.NumberColumn(
                    "Overall Vol", format="%.4f",
                ),
            },
        )

    with col_low:
        st.markdown("**Least Volatile (Predictable)**")
        bottom_df = df.tail(15).sort_values("overall_vol").copy()
        bottom_df.index = range(1, len(bottom_df) + 1)
        bottom_df.index.name = "Rank"
        st.dataframe(
            bottom_df[["name", "n_pitches", "overall_vol"]],
            use_container_width=True,
            column_config={
                "name": st.column_config.TextColumn("Pitcher"),
                "n_pitches": st.column_config.NumberColumn("Pitches"),
                "overall_vol": st.column_config.NumberColumn(
                    "Overall Vol", format="%.4f",
                ),
            },
        )

    # Distribution chart
    if len(df) >= 5:
        st.markdown("**Volatility Distribution**")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=df["overall_vol"],
            nbinsx=30,
            marker_color="#4ECDC4",
            opacity=0.75,
        ))
        mean_vol = df["overall_vol"].mean()
        fig.add_vline(
            x=mean_vol,
            line=dict(color="white", width=2, dash="dash"),
            annotation_text=f"Mean ({mean_vol:.3f})",
            annotation_position="top right",
        )
        fig.update_layout(
            xaxis=dict(title="Overall Volatility (entropy)"),
            yaxis=dict(title="Count"),
            template="plotly_dark",
            height=350,
            margin=dict(l=50, r=30, t=30, b=50),
        )
        st.plotly_chart(fig, use_container_width=True, key="pivs_dist")
