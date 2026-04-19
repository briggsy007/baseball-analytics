#!/usr/bin/env python
"""Plot year-over-year stability of CausalWAR contrarian leaderboards.

Reads `hit_rates_by_year.json` + `hit_rates_reproduction.json` from
`results/causal_war/contrarian_stability/` and produces:
    - hit_rates_by_year.html  (Plotly)
    - hit_rates_by_year.png   (static, if kaleido is available)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "results" / "causal_war" / "contrarian_stability"


def main() -> int:
    with open(OUTDIR / "hit_rates_by_year.json", "r", encoding="utf-8") as f:
        hr = json.load(f)
    with open(OUTDIR / "hit_rates_reproduction.json", "r", encoding="utf-8") as f:
        repro = json.load(f)

    try:
        import plotly.graph_objects as go
    except Exception:
        print("plotly not available; skipping plot")
        return 0

    labels = []
    bl_rates, bl_lo, bl_hi, bl_n = [], [], [], []
    ov_rates, ov_lo, ov_hi, ov_n = [], [], [], []
    bl_naive, ov_naive = [], []
    for r in hr:
        labels.append(f"{r['baseline_year']} -> {r['followup_year']}")
        bl = r["buy_low"]
        ov = r["over_valued"]
        bl_rates.append((bl["hit_rate"] or 0) * 100)
        bl_lo.append((bl["ci_95_low"] or 0) * 100)
        bl_hi.append((bl["ci_95_high"] or 0) * 100)
        bl_n.append(bl["n_evaluated"])
        ov_rates.append((ov["hit_rate"] or 0) * 100)
        ov_lo.append((ov["ci_95_low"] or 0) * 100)
        ov_hi.append((ov["ci_95_high"] or 0) * 100)
        ov_n.append(ov["n_evaluated"])
        nb = r.get("naive_baseline", {})
        bl_naive.append((nb.get("buy_low", {}).get("matched_naive_rate") or 0) * 100)
        ov_naive.append((nb.get("over_valued", {}).get("matched_naive_rate") or 0) * 100)

    # Add the 2-year aggregate anchor row (dashboard replication)
    labels.append("2023-24 -> 2025 (dashboard)")
    bl_rates.append((repro["buy_low"]["hit_rate"] or 0) * 100)
    bl_lo.append((repro["buy_low"]["ci_95_low"] or 0) * 100)
    bl_hi.append((repro["buy_low"]["ci_95_high"] or 0) * 100)
    bl_n.append(repro["buy_low"]["n_evaluated"])
    ov_rates.append((repro["over_valued"]["hit_rate"] or 0) * 100)
    ov_lo.append((repro["over_valued"]["ci_95_low"] or 0) * 100)
    ov_hi.append((repro["over_valued"]["ci_95_high"] or 0) * 100)
    ov_n.append(repro["over_valued"]["n_evaluated"])
    # No naive baseline for the 2-year aggregate anchor (reported but not comparable)
    bl_naive.append(None)
    ov_naive.append(None)

    # Error bars are absolute, not offsets, so convert
    def _err(rates, lo, hi):
        return dict(
            type="data",
            symmetric=False,
            array=[h - r for h, r in zip(hi, rates)],
            arrayminus=[r - l for l, r in zip(lo, rates)],
            color="rgba(0,0,0,0.6)",
            thickness=1.5,
        )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Buy-Low",
        x=labels, y=bl_rates,
        marker_color="#002D72",  # Phillies blue
        error_y=_err(bl_rates, bl_lo, bl_hi),
        text=[f"{r:.0f}% (n={n})" for r, n in zip(bl_rates, bl_n)],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Over-Valued",
        x=labels, y=ov_rates,
        marker_color="#E81828",  # Phillies red
        error_y=_err(ov_rates, ov_lo, ov_hi),
        text=[f"{r:.0f}% (n={n})" for r, n in zip(ov_rates, ov_n)],
        textposition="outside",
    ))

    # Overlay naive baseline markers for each window (scatter)
    valid_idx = [i for i, v in enumerate(bl_naive) if v is not None]
    fig.add_trace(go.Scatter(
        name="Buy-Low naive (WAR-matched)",
        x=[labels[i] for i in valid_idx],
        y=[bl_naive[i] for i in valid_idx],
        mode="markers",
        marker=dict(symbol="diamond-open", size=14, color="#002D72",
                    line=dict(color="#002D72", width=2)),
        xaxis="x", yaxis="y",
    ))
    valid_idx2 = [i for i, v in enumerate(ov_naive) if v is not None]
    fig.add_trace(go.Scatter(
        name="Over-Valued naive (WAR-matched)",
        x=[labels[i] for i in valid_idx2],
        y=[ov_naive[i] for i in valid_idx2],
        mode="markers",
        marker=dict(symbol="diamond-open", size=14, color="#E81828",
                    line=dict(color="#E81828", width=2)),
        xaxis="x", yaxis="y",
    ))

    fig.add_shape(
        type="line",
        x0=-0.5, x1=len(labels) - 0.5,
        y0=50, y1=50,
        line=dict(color="grey", dash="dash", width=1),
        layer="below",
    )
    fig.add_annotation(
        x=len(labels) - 0.5,
        y=50,
        text="chance (50%)",
        showarrow=False,
        yshift=10,
        xanchor="right",
        font=dict(color="grey", size=11),
    )

    fig.update_layout(
        title="CausalWAR Contrarian Leaderboards: Year-over-Year Hit Rates",
        xaxis_title="Baseline -> Follow-up window",
        yaxis_title="Hit rate (%)",
        yaxis=dict(range=[0, 105]),
        barmode="group",
        template="plotly_white",
        width=900, height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )

    html_path = OUTDIR / "hit_rates_by_year.html"
    fig.write_html(str(html_path), include_plotlyjs="cdn")
    print(f"Wrote {html_path}")

    # PNG export is optional and skipped by default (kaleido hangs on this
    # Windows/Chrome build). HTML is self-contained for the report.

    return 0


if __name__ == "__main__":
    sys.exit(main())
