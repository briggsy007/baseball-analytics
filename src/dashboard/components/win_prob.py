"""
Win probability chart visualization.

Step-line chart showing home win probability over the course of a game,
with fill colors, inning markers, and smart annotations for high-WPA events.
Designed to look like the FanGraphs win probability chart.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go


# Team colors for common NL East teams
TEAM_COLORS: dict[str, str] = {
    "PHI": "#E81828",
    "NYM": "#002D72",
    "ATL": "#CE1141",
    "WSH": "#AB0003",
    "MIA": "#00A3E0",
    "OPP": "#7F8C8D",
}

# Maximum number of annotations to display
_MAX_ANNOTATIONS = 10

# Minimum horizontal gap (in event_index units) before we reset the stagger
_HORIZONTAL_GAP = 5


def create_win_prob_chart(
    wp_curve: pd.DataFrame,
    home_team: str = "PHI",
    away_team: str = "OPP",
) -> go.Figure:
    """Build a win probability line chart.

    Parameters
    ----------
    wp_curve : pd.DataFrame
        Must contain ``event_index`` and ``home_win_prob`` columns.
        Optionally ``description``, ``delta``, ``inning``, and ``half``
        for annotation text and inning markers.
    home_team : str
        Home team abbreviation (for color and label).
    away_team : str
        Away team abbreviation.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    if wp_curve is None or wp_curve.empty:
        fig.add_annotation(
            text="No win probability data available",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color="gray"),
        )
        fig.update_layout(template="plotly_dark", height=380)
        return fig

    x = wp_curve["event_index"]
    y = wp_curve["home_win_prob"] * 100  # convert to percent

    home_color = TEAM_COLORS.get(home_team, "#E81828")
    away_color = TEAM_COLORS.get(away_team, "#7F8C8D")

    # ------------------------------------------------------------------
    # 1. Inning boundary markers (vertical dashed lines + top labels)
    # ------------------------------------------------------------------
    inning_shapes, inning_annotations = _build_inning_markers(wp_curve)

    # ------------------------------------------------------------------
    # 2. Baseline at 50%  (must come BEFORE the fill trace for "tonexty")
    # ------------------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=[x.min(), x.max()],
        y=[50, 50],
        mode="lines",
        line=dict(color="rgba(255,255,255,0.45)", width=1.5, dash="dash"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Label the 50 % line on the left
    inning_annotations.append(dict(
        x=x.min(), y=50,
        xanchor="right", yanchor="middle",
        text="50 %",
        showarrow=False,
        font=dict(size=9, color="rgba(255,255,255,0.55)"),
        xshift=-6,
    ))

    # ------------------------------------------------------------------
    # 3. Filled area between the WP line and the 50 % baseline
    # ------------------------------------------------------------------
    fig.add_trace(go.Scatter(
        x=x, y=y,
        fill="tonexty",
        mode="none",
        fillcolor=f"rgba({_hex_to_rgb(home_color)}, 0.15)",
        showlegend=False,
        hoverinfo="skip",
    ))

    # ------------------------------------------------------------------
    # 4. Main WP line with rich hover
    # ------------------------------------------------------------------
    hover_texts = _build_hover_texts(wp_curve, home_team)

    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode="lines",
        line=dict(color=home_color, width=2.5, shape="hv"),
        name=f"{home_team} Win Prob",
        hovertext=hover_texts,
        hoverinfo="text",
    ))

    # ------------------------------------------------------------------
    # 5. Smart annotations for high-impact events only
    # ------------------------------------------------------------------
    event_annotations = _build_event_annotations(
        wp_curve, home_color, away_color,
    )

    # Merge all annotation lists
    all_annotations = inning_annotations + event_annotations

    # ------------------------------------------------------------------
    # 6. Layout
    # ------------------------------------------------------------------
    fig.update_layout(
        title=dict(text="Win Probability", font=dict(size=16)),
        xaxis=dict(
            title="Game Events",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Home Win %",
            range=[0, 100],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.1)",
            ticksuffix="%",
        ),
        template="plotly_dark",
        height=380,
        margin=dict(l=60, r=30, t=50, b=50),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="center", x=0.5,
        ),
        shapes=inning_shapes,
        annotations=all_annotations,
    )

    return fig


# ======================================================================
# Internal helpers
# ======================================================================

def _build_inning_markers(
    wp_curve: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return Plotly shapes (vertical dashed lines) and annotations for inning boundaries.

    An inning boundary is defined as the first event of each new inning
    (i.e. the first time we see ``inning`` change in the data).
    """
    shapes: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []

    if "inning" not in wp_curve.columns:
        return shapes, annotations

    seen_innings: set[int] = set()
    for _, row in wp_curve.iterrows():
        inning = int(row["inning"])
        if inning in seen_innings:
            continue
        seen_innings.add(inning)
        ev_idx = row["event_index"]

        # Vertical dashed line
        shapes.append(dict(
            type="line",
            x0=ev_idx, x1=ev_idx,
            y0=0, y1=100,
            yref="y", xref="x",
            line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
        ))

        # Inning label at the top of the chart area
        annotations.append(dict(
            x=ev_idx, y=100,
            xanchor="center", yanchor="bottom",
            text=str(inning),
            showarrow=False,
            font=dict(size=9, color="rgba(255,255,255,0.5)"),
            yshift=4,
        ))

    return shapes, annotations


def _build_hover_texts(wp_curve: pd.DataFrame, home_team: str) -> list[str]:
    """Build per-point hover labels with full event details."""
    texts: list[str] = []
    has_desc = "description" in wp_curve.columns
    has_delta = "delta" in wp_curve.columns
    has_inning = "inning" in wp_curve.columns
    has_half = "half" in wp_curve.columns

    for _, row in wp_curve.iterrows():
        wp_pct = row["home_win_prob"] * 100
        parts = [f"<b>{home_team} Win Prob: {wp_pct:.1f}%</b>"]

        if has_inning:
            half_str = ""
            if has_half:
                half_str = "Top " if row["half"] == "top" else "Bot "
            parts.append(f"{half_str}{int(row['inning'])}")

        if has_delta:
            delta = row["delta"]
            sign = "+" if delta > 0 else ""
            parts.append(f"WPA: {sign}{delta * 100:.1f}%")

        if has_desc and row.get("description"):
            parts.append(f"<i>{row['description']}</i>")

        texts.append("<br>".join(parts))

    return texts


def _build_event_annotations(
    wp_curve: pd.DataFrame,
    home_color: str,
    away_color: str,
) -> list[dict[str, Any]]:
    """Select high-impact events and return smartly positioned annotations."""
    annotations: list[dict[str, Any]] = []

    if "delta" not in wp_curve.columns:
        return annotations

    abs_deltas = wp_curve["delta"].abs()

    # Adaptive threshold: at least |0.05|, but raise if there are many events
    threshold = max(0.05, abs_deltas.quantile(0.85))
    big_events = wp_curve[abs_deltas >= threshold].copy()

    if big_events.empty:
        return annotations

    # Cap at _MAX_ANNOTATIONS, keeping the highest-impact events
    if len(big_events) > _MAX_ANNOTATIONS:
        big_events = big_events.assign(_abs=big_events["delta"].abs())
        big_events = big_events.nlargest(_MAX_ANNOTATIONS, "_abs")
        big_events = big_events.drop(columns=["_abs"])

    # Sort by event_index so we can detect horizontal proximity
    big_events = big_events.sort_values("event_index")

    # Separate home (positive delta) and away (negative delta)
    home_events = big_events[big_events["delta"] > 0].copy()
    away_events = big_events[big_events["delta"] <= 0].copy()

    # Build annotations for each group
    annotations.extend(
        _staggered_annotations(home_events, direction="above", color=home_color)
    )
    annotations.extend(
        _staggered_annotations(away_events, direction="below", color=away_color)
    )

    return annotations


def _staggered_annotations(
    events: pd.DataFrame,
    direction: str,
    color: str,
) -> list[dict[str, Any]]:
    """Return annotation dicts with staggered vertical offsets.

    ``direction`` is "above" (ay < 0) or "below" (ay > 0).
    Consecutive annotations that are within ``_HORIZONTAL_GAP`` event
    indices of each other get increasingly offset so they do not overlap.
    """
    result: list[dict[str, Any]] = []
    base_offset = -40 if direction == "above" else 40
    step = -30 if direction == "above" else 30  # additional offset per level

    prev_x: float | None = None
    stagger_level = 0

    for _, row in events.iterrows():
        cur_x = row["event_index"]
        if prev_x is not None and abs(cur_x - prev_x) < _HORIZONTAL_GAP:
            stagger_level += 1
        else:
            stagger_level = 0
        prev_x = cur_x

        ay = base_offset + step * (stagger_level % 3)
        text = _shorten_event(row.get("description", ""), row["delta"])

        result.append(dict(
            x=cur_x,
            y=row["home_win_prob"] * 100,
            text=text,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1,
            arrowcolor=color,
            ax=0,
            ay=ay,
            font=dict(size=10, color=color),
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor=color,
            borderwidth=1,
            borderpad=3,
        ))

    return result


def _shorten_event(description: str, delta: float) -> str:
    """Shorten an event description to ~25 chars and append the WPA value."""
    if not description:
        sign = "+" if delta > 0 else ""
        return f"{sign}{delta * 100:.1f}% WPA"

    short = description[:25].rstrip()
    if len(description) > 25:
        short += "..."
    sign = "+" if delta > 0 else ""
    return f"{short} ({sign}{delta * 100:.1f}%)"


def _hex_to_rgb(hex_color: str) -> str:
    """Convert '#RRGGBB' to 'R, G, B' string for rgba()."""
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"{r}, {g}, {b}"
