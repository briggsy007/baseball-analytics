"""
Strike zone heatmap visualization.

Renders a Plotly heatmap overlaid on a standard strike zone rectangle,
divided into a 3x3 grid, with color encoding for any metric (estimated BA,
wOBA, whiff rate, etc.).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Strike zone boundaries (in feet from plate center / above ground)
ZONE_LEFT = -0.83
ZONE_RIGHT = 0.83
ZONE_BOTTOM = 1.5
ZONE_TOP = 3.5

# Grid resolution for the heatmap
N_BINS_X = 12
N_BINS_Y = 12


def create_strike_zone_heatmap(
    pitch_data: pd.DataFrame,
    metric: str = "estimated_ba",
    title: str = "",
    colorscale: str = "RdBu_r",
) -> go.Figure:
    """Build a strike-zone heatmap figure.

    Parameters
    ----------
    pitch_data : pd.DataFrame
        Must contain columns ``plate_x``, ``plate_z``, and *metric*.
    metric : str
        Column name to aggregate inside each zone bin.
    title : str
        Figure title.
    colorscale : str
        Plotly colorscale name.

    Returns
    -------
    go.Figure
    """
    if pitch_data is None or pitch_data.empty:
        return _empty_figure(title)

    needed = {"plate_x", "plate_z", metric}
    if not needed.issubset(pitch_data.columns):
        return _empty_figure(title)

    # Bin the data into a grid that spans wider than the zone
    x_edges = np.linspace(-1.5, 1.5, N_BINS_X + 1)
    y_edges = np.linspace(0.5, 4.5, N_BINS_Y + 1)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    df = pitch_data.copy()
    df["xbin"] = pd.cut(df["plate_x"], bins=x_edges, labels=False)
    df["ybin"] = pd.cut(df["plate_z"], bins=y_edges, labels=False)
    df = df.dropna(subset=["xbin", "ybin"])

    grid = np.full((N_BINS_Y, N_BINS_X), np.nan)
    grouped = df.groupby(["ybin", "xbin"])[metric].mean()
    for (yi, xi), val in grouped.items():
        yi_int, xi_int = int(yi), int(xi)
        if 0 <= yi_int < N_BINS_Y and 0 <= xi_int < N_BINS_X:
            grid[yi_int][xi_int] = round(val, 3)

    fig = go.Figure()

    # Heatmap layer
    fig.add_trace(go.Heatmap(
        z=grid,
        x=x_centers,
        y=y_centers,
        colorscale=colorscale,
        zmin=0.100,
        zmax=0.400,
        colorbar=dict(title=metric.replace("_", " ").title(), tickformat=".3f"),
        hovertemplate="x: %{x:.2f}<br>z: %{y:.2f}<br>value: %{z:.3f}<extra></extra>",
        opacity=0.85,
    ))

    # Strike zone rectangle
    fig.add_shape(
        type="rect",
        x0=ZONE_LEFT, x1=ZONE_RIGHT,
        y0=ZONE_BOTTOM, y1=ZONE_TOP,
        line=dict(color="white", width=2),
    )

    # 3x3 grid lines inside the zone
    zone_w = (ZONE_RIGHT - ZONE_LEFT) / 3
    zone_h = (ZONE_TOP - ZONE_BOTTOM) / 3
    for i in range(1, 3):
        # Vertical
        fig.add_shape(type="line",
                      x0=ZONE_LEFT + i * zone_w, x1=ZONE_LEFT + i * zone_w,
                      y0=ZONE_BOTTOM, y1=ZONE_TOP,
                      line=dict(color="white", width=1, dash="dot"))
        # Horizontal
        fig.add_shape(type="line",
                      x0=ZONE_LEFT, x1=ZONE_RIGHT,
                      y0=ZONE_BOTTOM + i * zone_h, y1=ZONE_BOTTOM + i * zone_h,
                      line=dict(color="white", width=1, dash="dot"))

    # Home plate polygon (trapezoid)
    plate_y = 0.3
    fig.add_shape(type="path",
                  path=f"M -0.71 {plate_y} L 0.71 {plate_y} L 0.83 {plate_y + 0.15} "
                       f"L 0 {plate_y + 0.35} L -0.83 {plate_y + 0.15} Z",
                  line=dict(color="white", width=1),
                  fillcolor="rgba(255,255,255,0.1)")

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Plate X (ft)", range=[-1.6, 1.6], constrain="domain",
                   showgrid=False, zeroline=False),
        yaxis=dict(title="Plate Z (ft)", range=[0.2, 4.6], scaleanchor="x", scaleratio=1,
                   showgrid=False, zeroline=False),
        template="plotly_dark",
        height=500,
        width=450,
        margin=dict(l=50, r=50, t=60, b=50),
    )

    return fig


def _empty_figure(title: str) -> go.Figure:
    """Return a placeholder figure when no data is available."""
    fig = go.Figure()
    fig.add_annotation(text="No data available", xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=18, color="gray"))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_dark",
        height=500, width=450,
    )
    return fig
