"""
Projections dashboard view -- Next-Season WAR projections (Marcel + Statcast
overlay, v2 with TJ flag / calibrated age curve / role change).

Reads the most-recent projections CSV from ``results/projections/`` and
surfaces the ``tj_adjustment`` column as a visible post-TJ badge on the
pitcher rows. Kept minimal on purpose: the authoritative rebuild runs via
``scripts/build_projections.py --v2``; this view is a consumer, not a
trainer.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PROJECTIONS_DIR = _PROJECT_ROOT / "results" / "projections"


@st.cache_data(ttl=3600)
def _load_latest_projections() -> tuple[pd.DataFrame, str | None]:
    candidates = sorted(_PROJECTIONS_DIR.glob("projections_*.csv"))
    if not candidates:
        return pd.DataFrame(), None
    latest = candidates[-1]
    return pd.read_csv(latest), latest.name


def _badge(v: float) -> str:
    if v is None or pd.isna(v) or float(v) == 0.0:
        return ""
    return "Post-TJ"


def render() -> None:
    st.title("Projections (v2)")
    st.caption(
        "Next-season WAR projections = Marcel baseline + Statcast overlay + "
        "age curve + TJ flag + role change. TJ flag fires when a pitcher's "
        "surgery is within 24 months of the target-season start."
    )

    df, src = _load_latest_projections()
    if df.empty:
        st.warning(
            "No projections artifact found in results/projections/. "
            "Run `python scripts/build_projections.py --v2 "
            "--train-seasons 2021 2022 2023 --target-season 2024 "
            "--output-path results/projections/`."
        )
        return

    st.caption(f"Source: `results/projections/{src}`")

    pos = st.radio("Cohort", ["pitcher", "batter"], horizontal=True, index=0)
    sub = df[df["position"] == pos].copy()

    if pos == "pitcher" and "tj_adjustment" in sub.columns:
        sub["TJ flag"] = sub["tj_adjustment"].apply(_badge)
        flagged = int((sub["TJ flag"] == "Post-TJ").sum())
        st.metric("Pitchers flagged post-TJ", flagged)
        cols = [
            "name", "projected_war", "marcel_war",
            "statcast_adjustment", "tj_adjustment", "TJ flag",
            "age", "prior_3yr_war",
        ]
    else:
        cols = [
            "name", "projected_war", "marcel_war",
            "statcast_adjustment", "age", "prior_3yr_war",
        ]

    cols = [c for c in cols if c in sub.columns]
    sub = sub.sort_values("projected_war", ascending=False).reset_index(drop=True)
    st.dataframe(sub[cols], use_container_width=True, hide_index=True)
