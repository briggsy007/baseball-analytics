"""
Spray chart visualization.

Renders batted-ball locations on a baseball field outline using Plotly
scatter, colored by hit result.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Home plate center in Statcast coordinates
HC_X_CENTER = 125.42
HC_Y_HOME = 204.0  # approximate y for home plate

RESULT_COLORS: dict[str, str] = {
    "single": "#3498DB",
    "double": "#2ECC71",
    "triple": "#F39C12",
    "home_run": "#E81828",
    "out": "#7F8C8D",
}

RESULT_LABELS: dict[str, str] = {
    "single": "Single",
    "double": "Double",
    "triple": "Triple",
    "home_run": "Home Run",
    "out": "Out",
}


def _field_outline() -> list[dict[str, Any]]:
    """Return Plotly shape dicts for the baseball field outline."""
    shapes: list[dict[str, Any]] = []

    # Infield diamond
    diamond_pts = [
        (HC_X_CENTER, HC_Y_HOME),               # Home
        (HC_X_CENTER + 63.64, HC_Y_HOME - 63.64),  # First base (approx)
        (HC_X_CENTER, HC_Y_HOME - 127.28),      # Second base
        (HC_X_CENTER - 63.64, HC_Y_HOME - 63.64),  # Third base
    ]
    path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in diamond_pts) + " Z"
    shapes.append(dict(type="path", path=path,
                       line=dict(color="rgba(255,255,255,0.5)", width=1),
                       fillcolor="rgba(255,255,255,0.02)"))

    # Outfield arc
    angles = np.linspace(-45, 225, 100)
    radius = 250
    arc_pts = [(HC_X_CENTER + radius * np.cos(np.radians(a)),
                HC_Y_HOME - radius * np.sin(np.radians(a))) for a in angles]
    arc_path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in arc_pts)
    shapes.append(dict(type="path", path=arc_path,
                       line=dict(color="rgba(255,255,255,0.3)", width=1)))

    return shapes


def create_spray_chart(
    hit_data: pd.DataFrame,
    title: str = "",
) -> go.Figure:
    """Build a spray chart figure.

    Parameters
    ----------
    hit_data : pd.DataFrame
        Must contain ``hc_x``, ``hc_y``, and ``result`` columns.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    if hit_data is None or hit_data.empty:
        fig.add_annotation(text="No batted-ball data available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray"))
        fig.update_layout(title=dict(text=title), template="plotly_dark", height=500, width=500)
        return fig

    # Add field outline shapes
    for shape in _field_outline():
        fig.add_shape(**shape)

    # Plot each result type as a separate trace for legend
    for result_type in ["out", "single", "double", "triple", "home_run"]:
        subset = hit_data[hit_data["result"] == result_type]
        if subset.empty:
            continue
        fig.add_trace(go.Scatter(
            x=subset["hc_x"],
            y=subset["hc_y"],
            mode="markers",
            marker=dict(
                size=7,
                color=RESULT_COLORS.get(result_type, "#95A5A6"),
                opacity=0.8,
                line=dict(width=0.5, color="white"),
            ),
            name=RESULT_LABELS.get(result_type, result_type),
            hovertemplate=(
                f"Result: {RESULT_LABELS.get(result_type, result_type)}<br>"
                "x: %{x:.1f}<br>y: %{y:.1f}"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="", range=[0, 250], showgrid=False, zeroline=False,
                   showticklabels=False),
        yaxis=dict(title="", range=[220, -30], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x", scaleratio=1),
        template="plotly_dark",
        height=500,
        width=500,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )

    return fig
