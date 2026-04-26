"""Batter x pitcher matchup simulator — A3 dossier view (Phase 0.5 stub).

Real distributions are emitted by ``pitchgpt_sim.rollout()`` aggregated
through a future ``pitchgpt_matchup`` module (see EXECUTION_PLAN §6 A3).
Until both land, this view renders a hand-built fixture.

Methodology (from EXECUTION_PLAN §6 A3, copied so the view is self-explanatory):
    For each (batter, pitcher) pair:
    1. Sample 10K PA rollouts conditioned on batter.stand x pitcher.p_throws
       and the pitcher's recent distribution.
    2. Per rollout, compute PA-outcome + wOBA contribution.
    3. Emit histogram with p05/p25/p50/p75/p95.

Small-sample guard (Gate 3 disclosure):
    Pairs with fewer than 20 real matchup PAs are flagged
    "rollout only, no empirical cross-check." The flag is shown in the
    UI; a public consumer should treat the histogram as a model
    projection only, not a backtested estimate.
"""
from __future__ import annotations

import streamlit as st

from src.dashboard.fixtures.matchup_sim_fixture import (
    build_matchup_fixture,
    list_matchup_options,
    schema_preview,
)


# ---------------------------------------------------------------------------
# Lazy import of pitchgpt_sim
# ---------------------------------------------------------------------------

_SIM_IMPORT_ERROR: Exception | None = None


def _try_import_pitchgpt_sim():
    global _SIM_IMPORT_ERROR
    try:
        from src.analytics import pitchgpt_sim  # noqa: F401  (probe import)
        _SIM_IMPORT_ERROR = None
        return True
    except Exception as exc:
        _SIM_IMPORT_ERROR = exc
        return False


# ---------------------------------------------------------------------------
# Methodology + disclosure text (single source of truth)
# ---------------------------------------------------------------------------

_METHODOLOGY_FOOTNOTE = (
    "**Methodology.** For the selected pair, sample 10,000 PA rollouts "
    "from a 0-0 start under the **calibrated** PitchGPT v2 backbone + "
    "PGConcatHead outcome predictor (Plan B winner; see "
    "`docs/pitchgpt_sim_engine/SIM_ENGINE_API.md` §4.4). Per rollout, "
    "compute the PA-terminal wOBA via the empirical wOBA-by-outcome lookup. "
    "Histogram with p05/p25/p50/p75/p95 bands."
)

_SMALL_SAMPLE_DISCLOSURE = (
    ":warning: **Small-sample guard (EXECUTION_PLAN §6 A3 Gate 3).** "
    "Pairs with fewer than 20 real matchup PAs are flagged "
    "**rollout only, no empirical cross-check**. Treat the histogram as a "
    "model projection rather than a backtested estimate."
)

_HIT_CEILING_DISCLOSURE = (
    ":warning: **Inherited in-play hit-vs-out ceiling.** The PGConcatHead "
    "outcome predictor lands at `in_play_hit` log-loss 2.34 (WEAKER PASS) "
    "on 2025 holdout. Hit-vs-out at pitch-time has a structural ceiling "
    "because launch_speed/launch_angle are post-pitch. Treat any wOBA "
    "splits dominated by hit/out outcomes (i.e., balls in play) with "
    "elevated uncertainty."
)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _render_phase_0_5_unavailable_notice() -> None:
    st.warning(
        ":construction: **Phase 0.5 rollout harness not yet available.** "
        "The `src/analytics/pitchgpt_sim.py` module is being implemented in "
        "a parallel session (see `docs/pitchgpt_sim_engine/EXECUTION_PLAN.md` "
        "§6.0.5). This view renders a hand-built fixture so the A3 "
        "scaffolding can be reviewed."
    )

    if _SIM_IMPORT_ERROR is not None:
        with st.expander("Import error detail (debug)"):
            st.code(repr(_SIM_IMPORT_ERROR))

    st.markdown("**Schema preview:**")
    sp = schema_preview()
    st.table([{"field": k, "shape / dtype": v} for k, v in sp.items()])


def _render_woba_histogram(fx) -> None:
    """Render the wOBA distribution + percentile bands."""
    st.subheader(f"Simulated wOBA distribution: {fx.batter_name} vs {fx.pitcher_name}")

    # Bands row
    cols = st.columns(5)
    band_labels = ["p05", "p25", "p50", "p75", "p95"]
    for c, lbl in zip(cols, band_labels):
        c.metric(lbl, f"{fx.percentiles[lbl]:.3f}")

    # Histogram (built in via Streamlit's bar_chart for zero-dependency)
    import numpy as np

    hist, edges = np.histogram(fx.woba_samples, bins=40)
    centers = 0.5 * (edges[:-1] + edges[1:])
    chart_data = {"wOBA": centers, "count": hist}
    st.bar_chart(chart_data, x="wOBA", y="count", height=280)

    cols2 = st.columns(3)
    cols2[0].metric("Simulated PAs", f"{fx.n_simulated_pas:,}")
    cols2[1].metric(
        "Real 2025 PAs",
        f"{fx.n_real_pas}",
        help="Actual head-to-head PA count in the 2025 regular season.",
    )
    cols2[2].metric(
        "In-play-hit share",
        f"{fx.in_play_hit_share:.3f}",
        help="Fraction of rollouts that terminated in an in-play hit. Interpret with hit/out ceiling.",
    )


def _render_disclosures(small_sample: bool) -> None:
    st.markdown("---")
    st.markdown("### Disclosures")
    if small_sample:
        st.warning(_SMALL_SAMPLE_DISCLOSURE)
    st.info(_HIT_CEILING_DISCLOSURE)
    st.caption(
        "Methodology paper v2 §3.7 details the Plan B WEAKER PASS verdict "
        "and the structural ceiling. See "
        "`docs/awards/methodology_paper_pitchgpt_v2.md`."
    )


def _render_selector_and_summary() -> None:
    """Render the batter x pitcher selector and summary card."""
    options = list_matchup_options()
    labels = [lbl for lbl, _ in options]
    idx_to_label = {idx: lbl for lbl, idx in options}

    label = st.selectbox(
        "Matchup",
        options=labels,
        index=0,
        key="matchup_sim_selector",
    )
    selected_idx = labels.index(label)
    fx = build_matchup_fixture(matchup_idx=selected_idx)

    _render_woba_histogram(fx)

    st.caption(_METHODOLOGY_FOOTNOTE)
    _render_disclosures(small_sample=fx.small_sample_flag)

    with st.expander("Sampling metadata (SIM_ENGINE_API §3.4)"):
        st.json(fx.sampling_metadata)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the A3 Matchup Sim view."""
    st.title("Matchup Sim (Batter x Pitcher)")
    st.caption(
        "A3 dossier — `EXECUTION_PLAN.md` §6 A3. Simulated wOBA distribution "
        "for any batter x pitcher pair under the calibrated rollout engine."
    )

    sim_available = _try_import_pitchgpt_sim()
    if not sim_available:
        _render_phase_0_5_unavailable_notice()
        st.markdown("---")

    # TODO(phase-0.5): replace ``build_matchup_fixture(...)`` inside
    # ``_render_selector_and_summary`` with a real
    # ``pitchgpt_matchup.simulate(batter_id, pitcher_id)`` that wraps
    # ``pitchgpt_sim.rollout(...)`` per SIM_ENGINE_API §5.1
    # (``pa_woba_distribution``).
    _render_selector_and_summary()
