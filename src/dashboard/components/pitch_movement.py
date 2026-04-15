"""
Pitch movement chart visualization.

Scatter plot of horizontal vs vertical pitch movement (pfx_x vs pfx_z),
colored by pitch type.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from src.dashboard.constants import PITCH_COLORS, PITCH_LABELS

# Alias kept for any downstream code that may reference the old name
DEFAULT_PITCH_COLORS = PITCH_COLORS


def create_pitch_movement_chart(
    arsenal_data: list[dict[str, Any]] | pd.DataFrame,
    title: str = "",
) -> go.Figure:
    """Build a pitch movement scatter plot.

    Parameters
    ----------
    arsenal_data : list[dict] or pd.DataFrame
        Each record should have ``pitch_type``, ``pfx_x``, ``pfx_z``, and
        optionally ``label`` and ``color``.  If a list of dicts is supplied
        (e.g. from a pitcher profile), each dict represents an aggregate
        point per pitch type.  If a DataFrame, individual pitches are plotted.
    title : str
        Figure title.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    if arsenal_data is None:
        return _empty(title)

    # Normalise to DataFrame
    if isinstance(arsenal_data, list):
        if len(arsenal_data) == 0:
            return _empty(title)
        df = pd.DataFrame(arsenal_data)
    else:
        df = arsenal_data.copy()

    if df.empty or "pfx_x" not in df.columns or "pfx_z" not in df.columns:
        return _empty(title)

    # Crosshair lines at origin
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"))
    fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"))

    pitch_types = df["pitch_type"].unique()
    for pt in pitch_types:
        subset = df[df["pitch_type"] == pt]
        color = PITCH_COLORS.get(pt, "#95A5A6")
        label = PITCH_LABELS.get(pt, pt)
        # Use per-row color if present
        if "color" in subset.columns and not subset["color"].isna().all():
            color = subset["color"].iloc[0]
        if "label" in subset.columns and not subset["label"].isna().all():
            label = str(subset["label"].iloc[0])

        fig.add_trace(go.Scatter(
            x=subset["pfx_x"],
            y=subset["pfx_z"],
            mode="markers",
            marker=dict(
                size=10 if len(subset) < 20 else 6,
                color=color,
                opacity=0.85,
                line=dict(width=0.5, color="white"),
            ),
            name=label,
            hovertemplate=(
                f"{label}<br>"
                "H-Break: %{x:.1f} in<br>"
                "V-Break: %{y:.1f} in"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis=dict(title="Horizontal Movement (in)", zeroline=False, showgrid=True,
                   gridcolor="rgba(255,255,255,0.1)"),
        yaxis=dict(title="Vertical Movement (in)", zeroline=False, showgrid=True,
                   gridcolor="rgba(255,255,255,0.1)"),
        template="plotly_dark",
        height=480,
        width=520,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.02),
    )

    return fig


def _empty(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text="No arsenal data available", xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="gray"))
    fig.update_layout(title=dict(text=title), template="plotly_dark", height=480, width=520)
    return fig
