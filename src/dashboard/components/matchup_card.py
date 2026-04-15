"""
Matchup card component.

Renders a pitcher-vs-batter matchup card using Streamlit layout components
(columns, metrics, dataframes). Not a Plotly figure -- renders directly.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


def create_matchup_card(matchup_data: dict[str, Any] | None) -> None:
    """Render a matchup card in the current Streamlit container.

    Parameters
    ----------
    matchup_data : dict
        Output of ``mock_matchup_stats`` or the real analytics equivalent.
    """
    if matchup_data is None:
        st.info("No matchup data available.")
        return

    pitcher = matchup_data.get("pitcher_name", "Unknown Pitcher")
    batter = matchup_data.get("batter_name", "Unknown Batter")

    # Top row: key counting stats
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("PA", matchup_data.get("pa", 0))
    c2.metric("H", matchup_data.get("h", 0))
    c3.metric("HR", matchup_data.get("hr", 0))
    c4.metric("K", matchup_data.get("k", 0))
    c5.metric("BB", matchup_data.get("bb", 0))
    c6.metric("AVG", f"{matchup_data.get('avg', 0):.3f}")

    # Estimated wOBA with confidence interval
    st.markdown("---")
    w1, w2 = st.columns([2, 3])
    with w1:
        woba = matchup_data.get("estimated_woba", 0)
        ci_low = matchup_data.get("woba_ci_low", 0)
        ci_high = matchup_data.get("woba_ci_high", 0)
        st.metric("Est. wOBA (Bayesian)", f"{woba:.3f}")
        st.caption(f"95% CI: [{ci_low:.3f} , {ci_high:.3f}]")
        sample = matchup_data.get("sample_size", 0)
        if sample < 30:
            st.warning(f"Small sample size ({sample} PA) -- estimate relies heavily on priors")
        else:
            st.caption(f"Based on {sample} plate appearances")

    with w2:
        # Confidence interval bullet chart
        fig = _woba_ci_chart(woba, ci_low, ci_high)
        st.plotly_chart(fig, use_container_width=True, key=f"woba_ci_{pitcher}_{batter}")

    # Vulnerability callout
    vuln = matchup_data.get("vulnerability", "")
    if vuln:
        st.info(f"**Key vulnerability:** {vuln}")

    # Pitch type breakdown table
    breakdown = matchup_data.get("pitch_breakdown", [])
    if breakdown:
        st.markdown("**Pitch Type Breakdown**")
        df = pd.DataFrame(breakdown)
        display_cols = ["label", "count", "pct", "whiff_rate", "ba_against", "avg_velo"]
        available = [c for c in display_cols if c in df.columns]
        df_display = df[available].copy()
        col_rename = {
            "label": "Pitch",
            "count": "N",
            "pct": "Usage %",
            "whiff_rate": "Whiff %",
            "ba_against": "BA Against",
            "avg_velo": "Avg Velo",
        }
        df_display = df_display.rename(columns=col_rename)
        if "Usage %" in df_display.columns:
            df_display["Usage %"] = (df_display["Usage %"] * 100).round(1).astype(str) + "%"
        if "Whiff %" in df_display.columns:
            df_display["Whiff %"] = (df_display["Whiff %"] * 100).round(1).astype(str) + "%"
        if "BA Against" in df_display.columns:
            df_display["BA Against"] = df_display["BA Against"].apply(lambda v: f"{v:.3f}")
        st.dataframe(df_display, use_container_width=True, hide_index=True)


def _woba_ci_chart(woba: float, ci_low: float, ci_high: float) -> go.Figure:
    """Create a small bullet / range chart for the wOBA confidence interval."""
    fig = go.Figure()

    # League average reference band
    fig.add_shape(type="rect", x0=0.300, x1=0.330, y0=-0.4, y1=0.4,
                  fillcolor="rgba(255,255,255,0.08)", line=dict(width=0))

    # CI bar
    fig.add_trace(go.Scatter(
        x=[ci_low, ci_high],
        y=[0, 0],
        mode="lines",
        line=dict(color="#E81828", width=6),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Point estimate
    fig.add_trace(go.Scatter(
        x=[woba],
        y=[0],
        mode="markers",
        marker=dict(size=14, color="white", symbol="diamond",
                    line=dict(color="#E81828", width=2)),
        showlegend=False,
        hovertemplate=f"Est. wOBA: {woba:.3f}<extra></extra>",
    ))

    fig.add_annotation(text="Lg Avg", x=0.315, y=0.45, showarrow=False,
                       font=dict(size=9, color="gray"))

    fig.update_layout(
        xaxis=dict(title="wOBA", range=[0.200, 0.500], showgrid=False),
        yaxis=dict(visible=False, range=[-1, 1]),
        template="plotly_dark",
        height=120,
        margin=dict(l=40, r=20, t=10, b=30),
    )
    return fig
