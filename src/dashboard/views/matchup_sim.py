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
    UI; a public consumer should treat the rendered product as a model
    projection only, not a backtested estimate.

PRODUCT SCOPE — K5 consequence (2026-08-10; display corrected same day):
    Phase 0.6.2 was KILLED at the fit-convergence gate
    (``docs/models/pitchgpt_phase062_results.md``; claim
    ``pitchgpt_phase062_kill``). ``PHASE_0.6.2_PLAN.md`` §6 names this view
    explicitly: "PA-level absolute-rate products (A3 matchup K%/BB%
    displays) are dropped from Tier-A scope; rank/differential products (A1
    grades, A2 projection *distribution shapes*) proceed with the
    marginal-bias disclosure." An A3 matchup wOBA *level* is precisely the
    dropped product, so this view publishes NO simulated wOBA quantity at
    all: no p05/p25/p50/p75/p95 bands, no histogram, no mean, no in-play-hit
    share, no K%/BB%/HR%.

    Rejected alternative, recorded so it is not retried: re-centring the
    distribution on its own simulated median. PA-terminal wOBA is zero for
    every PA that ends in an out, which is more than half of all PAs, so the
    simulated median is exactly zero and median-centring is a numerical
    no-op — it renders byte-identical numbers under a "delta" label, which
    is a worse honesty posture than showing the level plainly. Rescaling by
    some other statistic leaks just as badly here: the wOBA-by-outcome atoms
    (walk / single / double / triple / home run) are public constants, so a
    reader who can see rescaled atom positions can solve for the scale.

    What survives is a pure ORDINAL — where this pair sits in the ordering
    of the matchup cohort currently loaded. Both sides of any such
    comparison come from the same biased engine, so the shared component of
    the marginal bias largely cancels in the ordering; that is the same
    argument §6 uses to let the A1 grades ship. It does not cancel exactly
    (the bias is outcome-class-dependent and pairs differ in their class
    mix), so the ordering is ordinal only and carries the marginal-bias
    disclosure. Absolute levels return only if a future retrain passes its
    own pre-registered gates
    (``docs/pitchgpt_sim_engine/PITCHGPT_V2_SPEC.md``).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import streamlit as st

from src.claims import get_claim
from src.dashboard.fixtures.matchup_sim_fixture import (
    build_matchup_fixture,
    list_matchup_options,
    schema_preview,
)

# Registry-backed claims (WS2.2): retracted claims raise at import, so a
# retracted number is structurally unable to render on this page.
_CLAIM_ECE = get_claim("pitchgpt_per_pitch_ece")
_CLAIM_IN_PLAY_HIT = get_claim("pitchgpt_outcome_head_in_play_hit")
_CLAIM_PA_RATES = get_claim("pitchgpt_pa_rates_fail")
_CLAIM_062_KILL = get_claim("pitchgpt_phase062_kill")

# The kill claim's ``value`` is a mapping. Interpolating it whole renders a
# raw Python dict repr on the page, so index the sub-keys and format them --
# the same convention the other dict-valued claims use (see
# contrarian_leaderboards.py ``_render_k3_evidence`` and causal_war.py).
_K062: dict[str, Any] = _CLAIM_062_KILL.value
_KILL_QUANTITIES = (
    "Verdict quantities (2023 fit-convergence gate; max per-position "
    "class-marginal absolute deviation from empirical): iteration 1 "
    f"{_K062['iteration_1_max_abs_delta_pp']:.3f}pp, iteration 2 "
    f"{_K062['iteration_2_max_abs_delta_pp']:.3f}pp, against a pre-registered "
    f"{_K062['threshold_pp']:.1f}pp convergence threshold (pre-fit reference "
    f"{_K062['pre_fit_reference_max_abs_delta_pp']:.2f}pp). "
    f"{_K062['verdict']}."
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
    "from a 0-0 start under the **calibrated** (scoped -- see below) "
    "PitchGPT v2 backbone + "
    "PGConcatHead outcome predictor (Plan B winner; see "
    "`docs/pitchgpt_sim_engine/SIM_ENGINE_API.md` §4.4). Per rollout, "
    "compute the PA-terminal wOBA via the empirical wOBA-by-outcome lookup. "
    "**Every simulated wOBA quantity is withheld** (see the scope banner): "
    "no level, no p05/p25/p50/p75/p95 band, no histogram, no mean, no "
    "in-play-hit share. Only the pair's ordinal position within the loaded "
    "matchup cohort is published. "
    f"**Calibration scope (2026-08-10 audit):** 'calibrated' means per-pitch "
    f"post-temperature ECE {_CLAIM_ECE.value}. {_CLAIM_ECE.caveat} "  # claim:pitchgpt_per_pitch_ece
    f"PA-level absolute rates from this same rollout engine FAIL their "
    f"fidelity gates: {_CLAIM_PA_RATES.caveat} Read the cohort ordering as "
    "ordinal, not as a validated absolute outcome rate."
)

_PA_ABSOLUTE_DROPPED_BANNER = (
    ":no_entry: **PA-level absolute-rate products are DROPPED from this "
    "view (Tier-A scope change, 2026-08-10).** Phase 0.6.2 -- the "
    "pre-registered attempt to make rollout PA-level marginals honest -- was "
    "**KILLED at its fit-convergence gate**, so the absolute simulated wOBA "
    "level, the percentile bands, the histogram, the simulated K%/BB%/HR% "
    "and the absolute in-play-hit rate are not shown on this page in any "
    "form, and may not be quoted from this engine. "
    f"{_KILL_QUANTITIES} {_CLAIM_062_KILL.caveat} "
    "What survives is a pure **ordinal**: this pair's position in the "
    "ordering of the matchup cohort loaded below (a rank/differential "
    "product) under the marginal-bias disclosure. Absolute levels return "
    "only if the pre-registered v2 retrain "
    "(`docs/pitchgpt_sim_engine/PITCHGPT_V2_SPEC.md`) clears its own gates."
)

_WITHHELD_INVENTORY = (
    ":lock: **Withheld from this panel** (Tier-A scope drop, not a loading "
    "error): the simulated wOBA level and its p05/p25/p50/p75/p95 bands, the "
    "rollout histogram, the mean, the in-play-hit share, and the simulated "
    "K%/BB%/HR%. Median-centring was evaluated as an alternative to "
    "withholding and **rejected**: PA-terminal wOBA is zero for every PA "
    "ending in an out (more than half of all PAs), so the simulated median "
    "is exactly zero and a median-centred display renders the identical "
    "absolute numbers under a 'delta' label."
)

_SMALL_SAMPLE_DISCLOSURE = (
    ":warning: **Small-sample guard (EXECUTION_PLAN §6 A3 Gate 3).** "
    "Pairs with fewer than 20 real matchup PAs are flagged "
    "**rollout only, no empirical cross-check**. Treat the cohort ordering "
    "as a model projection rather than a backtested estimate."
)

_HIT_CEILING_DISCLOSURE = (
    ":warning: **Inherited in-play hit-vs-out ceiling.** The PGConcatHead "
    f"outcome predictor lands at `in_play_hit` log-loss {_CLAIM_IN_PLAY_HIT.value} (WEAKER PASS) "  # claim:pitchgpt_outcome_head_in_play_hit
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


@lru_cache(maxsize=8)
def _cohort_order(seed: int = 99) -> tuple[int, ...]:
    """Cohort matchup indices, most- to least- batter-favorable.

    The sort key is the pair's mean simulated PA-terminal wOBA. That mean is
    an absolute PA-level quantity, dropped from Tier-A scope by the Phase
    0.6.2 kill: it is computed here purely to induce an ordering and is
    never returned, logged or rendered. Ties break by cohort index, so the
    ordering is deterministic for a given seed.
    """
    n_pairs = len(list_matchup_options())
    keyed = []
    for idx in range(n_pairs):
        fx = build_matchup_fixture(matchup_idx=idx, seed=seed)
        keyed.append((-float(np.mean(fx.woba_samples)), idx))
    return tuple(idx for _, idx in sorted(keyed))


def cohort_rank_products(matchup_idx: int, seed: int = 99) -> dict[str, Any]:
    """Ordinal-only products for one matchup (K5 consequence, 2026-08-10).

    Every number in the returned payload is an integer ordinal: the pair's
    rank, the cohort size, and the rank of each cohort member. No wOBA-unit
    quantity -- no level, percentile band, histogram edge, mean or rate --
    appears anywhere in the return value. ``tests/test_matchup_sim_view.py``
    asserts that invariant, which is what makes the "absolute levels are
    withheld" copy on this page structurally true rather than merely
    advertised.
    """
    order = _cohort_order(seed)
    labels = [lbl for lbl, _ in list_matchup_options()]
    rank_of = {idx: pos + 1 for pos, idx in enumerate(order)}
    return {
        "rank": rank_of[matchup_idx],
        "n_pairs": len(order),
        "ordering": [
            {
                "rank": pos + 1,
                "matchup": labels[idx],
                "selected": idx == matchup_idx,
            }
            for pos, idx in enumerate(order)
        ],
    }


def _render_ordinal_products(fx, products: dict[str, Any]) -> None:
    """Render the ordinal product; every simulated wOBA quantity is withheld.

    K5 consequence (Phase 0.6.2 KILL, 2026-08-10): ``PHASE_0.6.2_PLAN.md`` §6
    drops PA-level absolute-rate products from Tier-A scope, and an A3
    matchup wOBA level is exactly that. Nothing derived from the wOBA scale
    reaches the page -- see the module docstring for why median-centring was
    rejected as a substitute for withholding.
    """
    st.subheader(
        f"{fx.batter_name} vs {fx.pitcher_name} "
        "— simulated wOBA levels withheld"
    )
    st.warning(_WITHHELD_INVENTORY)

    cols = st.columns(3)
    cols[0].metric(
        "Cohort rank (batter-favorable)",
        f"{products['rank']} of {products['n_pairs']}",
        help=(
            "Ordinal position of this pair among the matchup pairs loaded "
            "below. Rank 1 = the most batter-favorable simulated matchup in "
            "the cohort. A rank is a comparison between two runs of the same "
            "biased engine, so the shared marginal bias largely cancels; it "
            "does not cancel exactly, so read the ordering as ordinal only."
        ),
    )
    cols[1].metric("Simulated PAs", f"{fx.n_simulated_pas:,}")
    cols[2].metric(
        "Real 2025 PAs",
        f"{fx.n_real_pas}",
        help="Actual head-to-head PA count in the 2025 regular season.",
    )

    st.markdown("**Cohort ordering (ordinal only — no levels)**")
    st.dataframe(
        [
            {
                "Rank": row["rank"],
                "Matchup": row["matchup"],
                "Selected": row["selected"],
            }
            for row in products["ordering"]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Cohort = the {products['n_pairs']} pairs currently loaded in this "
        "view, not a league-wide population. The ordering is the only "
        "product published from the rollout distribution."
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

    _render_ordinal_products(fx, cohort_rank_products(selected_idx))

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
        "A3 dossier — `EXECUTION_PLAN.md` §6 A3. Ordinal standing of a batter "
        "x pitcher pair within the loaded cohort under the "
        "per-pitch-calibrated rollout engine. Calibration is scoped — see the "
        "methodology footnote: production-path ECE unmeasured and now "
        "stranded, PA-level absolute rates FAIL and are dropped from Tier-A "
        "scope, so no simulated wOBA level is published on this page."
    )
    st.error(_PA_ABSOLUTE_DROPPED_BANNER)

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
