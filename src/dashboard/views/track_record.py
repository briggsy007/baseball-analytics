"""Track Record view (plan WS1.2 / WS6.1 seed).

Renders the platform's prospective pick track record computed ONLY from the
two append-only ledger files:

    predictions/picks.jsonl
    predictions/resolutions.jsonl

No model artifact, cache, or doc is consulted — if a number is not derivable
from the ledgers, it does not appear here. Losses render with the same
prominence as wins (kill criterion K4: "the miss is published on the
track-record page with the same prominence a win would get").
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.pick_ledger import compute_track_record, load_ledger

# Chart inks (dataviz skill: single categorical series + ink reference marks;
# text/reference marks wear ink tokens, never a series hue).
_SERIES_BLUE = "#4269D0"
_INK_GRAY = "#6B7280"

_BANNER = (
    "**Every number on this page is computed from the append-only pick "
    "ledger and nothing else** (`predictions/picks.jsonl` + "
    "`predictions/resolutions.jsonl`). Picks are frozen with their "
    "resolution rule, source, and deadline *before* outcomes exist; "
    "resolutions are separate appends. Misses are shown with the same "
    "prominence as hits — this page has no mechanism to hide a loss. "
    "The hit-parlay model is an explicitly non-flagship, unvalidated "
    "heuristic; its record here is the honest test of it."
)


@st.cache_data(ttl=300)
def _load() -> dict:
    ledger = load_ledger()
    return compute_track_record(ledger["picks"], ledger["resolutions"])


def _rolling_brier_chart(rolling: list[dict]):
    import altair as alt

    df = pd.DataFrame(rolling)
    base = alt.Chart(df).encode(
        x=alt.X("n:Q", title="Resolved picks (cumulative)",
                axis=alt.Axis(tickMinStep=1)),
    )
    line = base.mark_line(color=_SERIES_BLUE, strokeWidth=2).encode(
        y=alt.Y("cumulative_mean_brier:Q", title="Cumulative mean Brier",
                scale=alt.Scale(domain=[0, max(0.5, df["cumulative_mean_brier"].max() * 1.1)])),
        tooltip=["n", "pick_id", "brier", "cumulative_mean_brier",
                 "resolved_utc"],
    )
    pts = base.mark_point(color=_SERIES_BLUE, size=70, filled=True).encode(
        y="cumulative_mean_brier:Q",
        tooltip=["n", "pick_id", "brier", "cumulative_mean_brier",
                 "resolved_utc"],
    )
    ref = (
        alt.Chart(pd.DataFrame({"y": [0.25]}))
        .mark_rule(color=_INK_GRAY, strokeDash=[4, 4], strokeWidth=1.5)
        .encode(y="y:Q")
    )
    return (line + pts + ref).properties(height=260)


def _calibration_chart(calibration: list[dict]):
    import altair as alt

    df = pd.DataFrame(calibration)
    base = alt.Chart(df).encode(
        x=alt.X("bucket:N", title="Frozen probability bucket", sort=None),
    )
    bars = base.mark_bar(color=_SERIES_BLUE, cornerRadiusTopLeft=4,
                         cornerRadiusTopRight=4, size=38).encode(
        y=alt.Y("observed_rate:Q", title="Rate",
                scale=alt.Scale(domain=[0, 1])),
        tooltip=["bucket", "n", "mean_predicted", "observed_rate", "gap"],
    )
    ticks = base.mark_tick(color=_INK_GRAY, thickness=3, size=44).encode(
        y="mean_predicted:Q",
        tooltip=["bucket", "n", "mean_predicted", "observed_rate", "gap"],
    )
    return (bars + ticks).properties(height=260)


def render() -> None:
    st.title("Track Record")
    st.warning(_BANNER)

    try:
        tr = _load()
    except Exception as exc:
        st.error(f"Ledger unreadable: {exc}")
        return

    s = tr["summary"]

    # ── Summary tiles: losses sit beside wins at identical prominence ──
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Resolved picks", s["resolved"])
    c2.metric("Hits", s["hits"])
    c3.metric("Misses", s["misses"])
    c4.metric("Voids", s["voids"])
    c5.metric(
        "ITT hit rate",
        f"{s['hit_rate_itt']:.1%}" if s["hit_rate_itt"] is not None else "n/a",
    )
    c6.metric(
        "Mean Brier",
        f"{s['mean_brier']:.4f}" if s["mean_brier"] is not None else "n/a",
        help="Lower is better. 0.25 = always guessing p=0.5. Computed over "
             f"{s['n_brier_scored']} probabilistic resolved picks.",
    )
    st.caption(
        f"{s['picks_total']} picks frozen in the ledger; "
        f"{s['unresolved']} awaiting resolution. Worst-case hit rate "
        f"(voids counted as losses): "
        + (f"{s['hit_rate_worst_case']:.1%}"
           if s["hit_rate_worst_case"] is not None else "n/a")
        + "."
    )

    # ── Per-product accounting ──
    st.subheader("By product")
    prod_rows = []
    for name, d in tr["by_product"].items():
        prod_rows.append(
            {
                "product": name,
                "picks": d["picks"],
                "resolved": d["resolved"],
                "W": d["yes"],
                "L": d["no"],
                "void": d["void"],
                "ITT hit rate": d["hit_rate_itt"],
                "worst-case rate": d["hit_rate_worst_case"],
                "mean Brier": d["mean_brier"],
            }
        )
    st.dataframe(pd.DataFrame(prod_rows), use_container_width=True,
                 hide_index=True)
    st.caption(
        "The 2026 contrarian boards resolve once, at season end + 7 days, "
        "per the frozen spec (`docs/models/contrarian_2026_resolution_spec.md`); "
        "until then they are counted as unresolved — no interim rate is a "
        "resolution number."
    )

    # ── Rolling Brier ──
    st.subheader("Rolling Brier score")
    if tr["rolling_brier"]:
        st.altair_chart(_rolling_brier_chart(tr["rolling_brier"]),
                        use_container_width=True)
        st.caption(
            "Blue line: cumulative mean Brier over probabilistic resolved "
            "picks, in resolution order. Dashed gray reference: 0.25, the "
            "score of always guessing 50%. Below the line = better than a "
            "coin flip; above = worse."
        )
    else:
        st.info("No probabilistic picks resolved yet.")

    # ── Calibration buckets ──
    st.subheader("Calibration")
    if tr["calibration"]:
        st.altair_chart(_calibration_chart(tr["calibration"]),
                        use_container_width=True)
        st.caption(
            "Blue bars: observed hit rate per frozen-probability bucket. "
            "Gray ticks: mean frozen probability of the bucket (the target "
            "a calibrated forecaster would match). Bars far below their "
            "tick = overconfidence, published as-is."
        )
        st.dataframe(pd.DataFrame(tr["calibration"]),
                     use_container_width=True, hide_index=True)
    else:
        st.info("No probabilistic picks resolved yet.")

    # ── Full resolution log: misses never collapse ──
    st.subheader("All resolved picks")
    resolved_rows = [r for r in tr["rows"] if r["outcome"] is not None]
    if resolved_rows:
        df = pd.DataFrame(resolved_rows)
        df["subject"] = df["subject"].map(
            lambda x: (x or {}).get("name") if isinstance(x, dict) else x
        )
        df = df.sort_values("resolved_utc", ascending=False)
        cols = ["pick_id", "product", "subject", "market_or_claim", "p",
                "direction", "outcome", "brier", "resolved_utc",
                "evidence_class"]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
    else:
        st.info("Nothing resolved yet.")

    # ── Open picks ──
    st.subheader("Open picks (frozen, awaiting resolution)")
    open_rows = [r for r in tr["rows"] if r["outcome"] is None]
    if open_rows:
        df = pd.DataFrame(open_rows)
        df["subject"] = df["subject"].map(
            lambda x: (x or {}).get("name") if isinstance(x, dict) else x
        )
        cols = ["pick_id", "product", "subject", "p", "direction",
                "frozen_utc", "resolve_by", "evidence_class"]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)
        non_prospective = sorted(
            {
                str(r["evidence_class"])
                for r in tr["rows"]
                if r.get("evidence_class")
                and not str(r["evidence_class"]).startswith("prospective")
            }
        )
        if non_prospective:
            st.error(
                "Evidence classes present that are NOT prospective: "
                f"{non_prospective} — these picks were registered after "
                "their outcomes were knowable and must not be quoted as a "
                "prospective record."
            )
        st.caption(
            "`prospective-backfilled_2026-08-10` = pick provably frozen "
            "(committed / file-dated) before its outcome, registered into "
            "the ledger on 2026-08-10 when the ledger was created."
        )
    else:
        st.info("No open picks.")
