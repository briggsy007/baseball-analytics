"""DPI dashboard-view guards (WS3.6 claims rewrite).

Three things this pins, all file/pure-function level (no Streamlit runtime,
no DB):

1. The unsourced impact line and the fielder-level mechanism copy the
   2026-08-10 audit flagged (DPI finding 10) stay deleted.
2. Every claims-registry id the evidence panel renders exists in
   ``docs/claims/claims.yaml`` and is renderable (not retracted).
3. Regressed DPI is computed by shrinking toward the league mean with the
   published split-half reliability, and is DEFERRED -- not faked -- for
   seasons the WS3.1 reliability artifact does not cover.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from src.claims import ClaimRetracted, get_claim
from src.dashboard.views import defensive_pressing as view

_VIEW_SRC = Path(view.__file__).read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 1. Audit finding 10: unsourced impact + fielder-mechanism copy is gone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "banned",
    [
        "30-50 runs",
        "30–50 runs",
        "equivalent to a top free agent",
        "elite fielding, good positioning",
        "poor range, bad positioning",
        # 2026-08-10 review: an unregistered SABR BABIP variance-decomposition
        # parenthetical shipped here; its "fielding 17%" term reads as a fact
        # about DPI's own signal composition. No registry entry, no dataset,
        # no sha -- banned outright rather than re-registered.
        "luck 44%",
        "fielding 17%",
        "Solving DIPS",
    ],
)
def test_unsourced_mechanism_copy_absent(banned):
    assert banned not in _VIEW_SRC, (
        f"audit DPI finding 10 copy is back in the DPI view: {banned!r}"
    )


def test_extra_base_prevention_is_labelled_unmodelled():
    """EBP is a raw XBH share -- the view must say so (audit finding 10)."""
    assert "no expected-XBH" in _VIEW_SRC


# ---------------------------------------------------------------------------
# 2. Evidence panel resolves through the registry
# ---------------------------------------------------------------------------

def test_evidence_panel_claims_all_resolve_and_are_renderable():
    assert view._DPI_EVIDENCE_CLAIMS, "evidence panel lists no claims"
    for claim_id in view._DPI_EVIDENCE_CLAIMS:
        claim = get_claim(claim_id)  # raises if unknown or retracted
        assert claim.caveat.strip(), f"{claim_id} renders without a caveat"


def test_defensible_core_leads_the_evidence_panel():
    assert view._DPI_EVIDENCE_CLAIMS[0] == "dpi_v2_partial_r_oaa_given_babip"
    assert "dpi_positioning_alignment_ab" in view._DPI_EVIDENCE_CLAIMS


def test_retracted_babip_corroboration_cannot_be_panelled():
    assert "dpi_babip_corroboration" not in view._DPI_EVIDENCE_CLAIMS
    with pytest.raises(ClaimRetracted):
        get_claim("dpi_babip_corroboration")


def test_view_quotes_no_bare_metric_literals():
    """Numbers in this view come from get_claim, not from the source text."""
    # -0.80 (retracted) and the v2 headline must not be hand-copied here.
    for literal in ("-0.80", "0.4698", "0.6406", "0.584"):
        assert literal not in _VIEW_SRC, (
            f"hand-copied metric literal {literal!r} in the DPI view"
        )


# ---------------------------------------------------------------------------
# 3. Regressed DPI: shrink where reliability is published, defer where not
# ---------------------------------------------------------------------------

def _fake_board() -> pd.DataFrame:
    return pd.DataFrame({
        "team_id": ["PHI", "ATL", "NYM", "WSH"],
        "dpi_mean": [0.60, -0.40, 0.20, -0.40],
        "rank": [1, 3, 2, 4],
    })


def test_regressed_dpi_shrinks_toward_the_league_mean():
    board, reliability, note = view._with_regressed_dpi(_fake_board(), 2025)
    assert reliability == pytest.approx(0.584, abs=1e-9)
    assert "regressed_dpi" in board.columns
    league_mean = board["dpi_mean"].mean()
    for _, row in board.iterrows():
        raw_gap = abs(row["dpi_mean"] - league_mean)
        reg_gap = abs(row["regressed_dpi"] - league_mean)
        assert reg_gap <= raw_gap + 1e-9
        assert reg_gap == pytest.approx(reliability * raw_gap, abs=1e-3)
    assert "reliability_2026-08-10" in note


def test_regressed_dpi_deferred_with_a_reason_for_uncovered_seasons():
    board, reliability, note = view._with_regressed_dpi(_fake_board(), 2026)
    assert reliability is None
    assert "regressed_dpi" not in board.columns
    assert "no published split-half reliability for 2026" in note
    assert "overstate" in note  # says WHY, not just that it is missing


def test_season_reliability_uses_the_seasons_own_estimate_for_2020():
    """2020 is a 60-game season and carries its own reliability, not the
    full-season Fisher mean."""
    r_2020, _ = view._season_reliability(2020)
    r_2025, _ = view._season_reliability(2025)
    assert r_2020 is not None and r_2025 is not None
    assert r_2020 != r_2025


def test_reliability_artifact_path_matches_the_registered_dataset():
    """The view's shrinkage source is the artifact the claim points at."""
    claim = get_claim("dpi_split_half_reliability")
    claim_dir = Path(claim.dataset["path"]).parent.name
    assert view._RELIABILITY_CSV.parent.name == claim_dir
    assert view._RELIABILITY_CSV.exists()


def test_leaderboard_table_lists_regressed_next_to_raw():
    src = re.search(
        r"display_cols = \[(.*?)\]", _VIEW_SRC, re.DOTALL,
    )
    assert src, "leaderboard display_cols not found"
    cols = src.group(1)
    assert "dpi_mean" in cols and "regressed_dpi" in cols
    assert cols.index("dpi_mean") < cols.index("regressed_dpi")
