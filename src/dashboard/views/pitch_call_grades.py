"""Counterfactual pitch-call grades — A1 dossier view (Phase 0.5 stub).

Real grades are computed by the rollout engine in
``src/analytics/pitchgpt_sim.py``. That module is being implemented in a
parallel session (Phase 0.5; see ``docs/pitchgpt_sim_engine/EXECUTION_PLAN.md``
§6.0.5 + §6 A1 dossier). Until it lands, this view renders a placeholder
fixture so the surrounding dashboard scaffolding can be reviewed.

Methodology (from EXECUTION_PLAN §6 A1, copied so the view is self-explanatory):
    For each real pitch in the 2025 cohort,
    1. Freeze the PA prefix up to (but not including) that pitch.
    2. Sample N=100 rollouts for the remaining horizon.
    3. Compute expected wOBA per rollout via the outcome head + WObaTable.
    4. The "call grade" is the percentile rank of the actual pitch's
       simulated wOBA outcome within the rollout distribution
       (low percentile = pitcher gave up MORE than expected = worse call).

Disclosure (load-bearing per SIM_ENGINE_API §4):
    The hit-vs-out distinction inherits a structural ceiling: pitch-time
    features cannot disambiguate balls in play that become hits vs outs
    without launch_speed / launch_angle. Grades that depend on this
    distinction are flagged.
"""
from __future__ import annotations

import streamlit as st

from src.dashboard.fixtures.pitch_call_grades_fixture import (
    OUTCOME_LABELS,
    SAMPLE_PAS,
    build_leaderboard_table,
    build_rollout_fixture,
    schema_preview,
)

# ---------------------------------------------------------------------------
# Lazy import of pitchgpt_sim
# ---------------------------------------------------------------------------

# Phase 0.5 deliverable. Until that session lands, the import will fail and
# the view falls back to fixture-only mode. Wrapped in a function so the
# import is re-tried if Streamlit hot-reloads after the module appears.

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
# Methodology footnote text (single source of truth)
# ---------------------------------------------------------------------------

_METHODOLOGY_FOOTNOTE = (
    "**Methodology.** Grade = rollout-percentile-of-actual-outcome, "
    "**calibrated**. For each pitch, we freeze the PA prefix, sample "
    "N=100 rollouts of the remaining at-bat under the calibrated "
    "PitchGPT v2 backbone + PGConcatHead outcome predictor, score each "
    "rollout's terminal wOBA, and report where the actual pitch's "
    "outcome falls in that distribution. **High percentile = pitcher "
    "allowed less than expected (good call); low percentile = allowed "
    "more (bad call).** Calibration is post-temperature ECE 0.0114 on "
    "2025 pitcher-disjoint holdout per "
    "`results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/metrics.json`."
)

_HIT_CEILING_DISCLOSURE = (
    ":warning: **In-play hit-vs-out ceiling.** Hit-vs-out distinction at "
    "pitch-time has a structural ceiling: pitch-time features cannot "
    "disambiguate balls in play that become hits vs outs without "
    "`launch_speed` / `launch_angle` (post-pitch). Grades that depend on "
    "this distinction are flagged. The Plan B winner (A1 PGConcatHead) "
    "lands at `in_play_hit` log-loss 2.34 on 2025 holdout — clears "
    "WEAKER PASS (<2.5) but misses full PASS (<2.0). Treat hit/out splits "
    "with caution."
)


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def _render_phase_0_5_unavailable_notice() -> None:
    """Render the fallback notice when ``pitchgpt_sim`` is not yet available."""
    st.warning(
        ":construction: **Phase 0.5 rollout harness not yet available.** "
        "The `src/analytics/pitchgpt_sim.py` module is being implemented in "
        "a parallel session (see `docs/pitchgpt_sim_engine/EXECUTION_PLAN.md` "
        "§6.0.5). This view is rendering a hand-built fixture so the "
        "downstream A1 scaffolding can be reviewed."
    )

    if _SIM_IMPORT_ERROR is not None:
        with st.expander("Import error detail (debug)"):
            st.code(repr(_SIM_IMPORT_ERROR))

    st.markdown(
        "**Schema preview (mirrors `SIM_ENGINE_API.md` §3.3):**"
    )
    schema = schema_preview()
    st.table([{"field": k, "shape / dtype": v} for k, v in schema.items()])


def _render_leaderboard_table() -> None:
    """Render the pitcher × week leaderboard table from fixture data."""
    rows = build_leaderboard_table()
    st.subheader("Pitcher × week leaderboard (placeholder)")
    st.caption(
        "Lower mean percentile = poorer pitch-call decision-making over "
        "the week. This is fixture data; once Phase 0.5 lands, the "
        "leaderboard rebuilds from `results/pitchgpt_sim/pitch_call_grades_2025/`."
    )
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "pitcher_id": st.column_config.NumberColumn("Pitcher ID", format="%d"),
            "pitcher_name": st.column_config.TextColumn("Pitcher"),
            "week": st.column_config.TextColumn("Week"),
            "n_pitches": st.column_config.NumberColumn("Pitches", format="%d"),
            "mean_grade_percentile": st.column_config.NumberColumn(
                "Mean grade pct", format="%.3f"
            ),
            "grade_letter": st.column_config.TextColumn("Grade"),
            "woba_delta_per_pitch_x1000": st.column_config.NumberColumn(
                "wOBA delta / pitch (×1000)", format="%+0.2f"
            ),
        },
    )


def _render_per_pitch_detail() -> None:
    """Render the per-pitch grade detail panel (5 sample PAs)."""
    st.subheader("Per-pitch grade detail (5 sample PAs)")
    st.caption(
        "Each PA is rolled out N=100 times from the pre-pitch state. "
        "The actual pitch's wOBA outcome is located within the simulated "
        "wOBA distribution; that percentile is the grade."
    )

    # Fixture rollout (used for bootstrapping the schema preview / sanity)
    fixture = build_rollout_fixture()

    for pa in SAMPLE_PAS:
        with st.expander(
            f"{pa['pitcher_name']} vs {pa['batter_name']} — {pa['scenario']} "
            f"→ grade **{pa['grade_letter']}** "
            f"(percentile {pa['demo_percentile']:.2f})"
        ):
            ctx = pa["context"]
            real_outcome = OUTCOME_LABELS[pa["real_outcome_idx"]]

            cols = st.columns(3)
            with cols[0]:
                st.markdown("**PA context (pre-pitch)**")
                st.write(
                    {
                        "count": f"{ctx.count[0]}-{ctx.count[1]}",
                        "outs": ctx.outs,
                        "inning": f"{ctx.inning} ({ctx.inning_topbot})",
                        "score_diff": ctx.score_diff,
                        "umpire_scalar": ctx.umpire_scalar,
                    }
                )
            with cols[1]:
                st.markdown("**Actual call**")
                st.write(
                    {
                        "pitch_token": pa["real_pitch_token"],
                        "outcome_observed": real_outcome,
                    }
                )
            with cols[2]:
                st.markdown("**Rollout summary**")
                st.write(
                    {
                        "n_samples": fixture.sampling_metadata["n_samples"],
                        "horizon": fixture.sampling_metadata["horizon"],
                        "n_truncated": fixture.sampling_metadata["n_truncated"],
                        "calibration_valid": fixture.sampling_metadata[
                            "calibration_valid"
                        ],
                    }
                )

            # Flag in_play_hit explicitly per SIM_ENGINE_API §4 disclosure
            if pa["real_outcome_idx"] == 5:  # in_play_hit
                st.warning(
                    ":warning: This grade depends on the hit-vs-out distinction "
                    "(see disclosure below). Treat the percentile as a coarse "
                    "indicator only."
                )

            st.caption(_METHODOLOGY_FOOTNOTE)


def _render_disclosure_block() -> None:
    """Render the load-bearing disclosure block (in_play_hit ceiling)."""
    st.markdown("---")
    st.markdown("### Disclosures")
    st.info(_HIT_CEILING_DISCLOSURE)
    st.caption(
        "Methodology paper v2 §3.7 details the Plan B WEAKER PASS verdict and "
        "the structural ceiling. See `docs/awards/methodology_paper_pitchgpt_v2.md`."
    )


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Counterfactual Pitch-Call Grades view (A1)."""
    st.title("Counterfactual Pitch-Call Grades")
    st.caption(
        "A1 dossier — `EXECUTION_PLAN.md` §6 A1. Grade each pitch by where "
        "its actual outcome falls in a distribution of N=100 calibrated "
        "rollouts of the remaining at-bat."
    )

    sim_available = _try_import_pitchgpt_sim()

    if not sim_available:
        _render_phase_0_5_unavailable_notice()
        st.markdown("---")

    # TODO(phase-0.5): replace `build_rollout_fixture()` and
    # `build_leaderboard_table()` calls below with real
    # `pitchgpt_sim.rollout(...)` invocations. The replacement happens at
    # the boundary of `_render_per_pitch_detail` and `_render_leaderboard_table`
    # — those functions become thin wrappers around the real engine, with
    # the fixture-builder retained only as a smoke test.
    _render_leaderboard_table()
    st.markdown("---")
    _render_per_pitch_detail()
    _render_disclosure_block()
