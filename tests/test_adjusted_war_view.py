"""AdjustedWAR dashboard-view guards (Batch D rename + ridge promotion).

Pins the two things the Batch D review found unenforced on the AdjustedWAR
page, plus the uncertainty gate:

1. **Producing-model disclosure.** ``scripts/precompute.py`` stamps
   ``scoring_model`` / ``scoring_artifact_version`` / ``scoring_artifact``
   onto every cached row precisely so a surface can say which estimator
   produced the numbers. The page must actually read them back -- the
   promotion of AdjustedWAR v3 does not license rendering legacy-produced
   cache rows under a promoted-model banner, and an UNSTAMPED frame must be
   reported as legacy-produced rather than passed over in silence.
2. **K6 framing** renders from the single claims-registry entry, not from a
   hand-typed paraphrase that can drift away from the board surfaces.
3. **WS4.7 CI gate**: intervals are withheld by code while
   ``adjusted_war_v3_ci_coverage`` says no construction may ship.

No Streamlit runtime and no DB: ``st`` is mocked, frames are synthetic.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from src.claims import get_claim
from src.dashboard.views import causal_war as view

_VIEW_SRC = Path(view.__file__).read_text(encoding="utf-8", errors="replace")
_PRECOMPUTE_SRC = (
    Path(__file__).resolve().parents[1] / "scripts" / "precompute.py"
).read_text(encoding="utf-8", errors="replace")


def _captions_from(frame: pd.DataFrame | None) -> list[str]:
    """Run the provenance renderer against a mocked ``st`` and return copy."""
    captured: list[str] = []
    with mock.patch.object(view, "st") as fake_st:
        fake_st.caption.side_effect = lambda s: captured.append(s)
        view._render_scoring_provenance(frame)
    return captured


# ---------------------------------------------------------------------------
# 1. Producing-model disclosure
# ---------------------------------------------------------------------------

_RIDGE_FRAME = pd.DataFrame({
    "player_id": [1, 2],
    "causal_war": [2.0, 1.0],
    "scoring_model": ["adjusted_war_v3"] * 2,
    "scoring_artifact_version": ["v2026.08.10"] * 2,
    "scoring_artifact": ["adjusted_war_v3_2026_08_10.pkl"] * 2,
})
_LEGACY_FRAME = pd.DataFrame({
    "player_id": [1],
    "causal_war": [2.0],
    "scoring_model": ["causal_war_legacy"],
    "scoring_artifact_version": [None],
    "scoring_artifact": [None],
})
_UNSTAMPED_FRAME = pd.DataFrame({"player_id": [1], "causal_war": [2.0]})


@pytest.mark.parametrize(
    "frame", [_RIDGE_FRAME, _LEGACY_FRAME, _UNSTAMPED_FRAME],
    ids=["ridge-stamped", "legacy-stamped", "unstamped"],
)
def test_every_frame_shape_states_a_producing_model(frame):
    captions = _captions_from(frame)
    assert len(captions) == 1, "provenance must be stated exactly once"
    assert "Scored by" in captions[0]


def test_ridge_frame_names_the_promoted_model_and_version():
    text = _captions_from(_RIDGE_FRAME)[0]
    assert "AdjustedWAR v3" in text
    assert "v2026.08.10" in text
    assert "adjusted_war_v3_2026_08_10.pkl" in text


def test_legacy_frame_says_the_numbers_are_not_from_the_promoted_model():
    text = _captions_from(_LEGACY_FRAME)[0]
    assert "LEGACY" in text
    assert "did not come from it" in text


def test_unstamped_cache_is_reported_as_legacy_not_unknown():
    """The blocking Batch D review finding.

    Stamping and the ridge promotion landed the same day, so a frame with no
    stamp was necessarily produced by the legacy formulation. Reporting it as
    "unknown" -- or not at all -- would let promoted-model framing attach to
    legacy numbers, which is exactly the live cache state on 2026-08-10.
    """
    text = _captions_from(_UNSTAMPED_FRAME)[0]
    assert "LEGACY" in text
    assert "NOT produced by the promoted model" in text


def test_empty_and_none_frames_still_disclose():
    assert _captions_from(None)
    assert _captions_from(pd.DataFrame())


def test_render_calls_the_provenance_renderer():
    assert "_render_scoring_provenance(df)" in _VIEW_SRC, (
        "the page must call the provenance renderer, not merely define it"
    )


# ---------------------------------------------------------------------------
# 2. K6 framing comes from the registry
# ---------------------------------------------------------------------------

def test_k6_framing_is_rendered_from_the_claim_verbatim():
    claim_value = get_claim("adjusted_war_boards_k6_framing").value
    assert claim_value.strip() in view._K6_FRAMING
    # ...and the mandated content is intact.
    for fragment in ("matched-naive", "Marcel-picker", "no edge claim vs Marcel"):
        assert fragment in view._K6_FRAMING


def test_no_hand_typed_marcel_paraphrase_survives():
    """The K6 sentence must not be duplicated in prose that can drift."""
    assert "ties Marcel and does not beat it" not in _VIEW_SRC


# ---------------------------------------------------------------------------
# 3. WS4.7 CI gate
# ---------------------------------------------------------------------------

def test_ci_gate_tracks_the_coverage_claim():
    claim = get_claim("adjusted_war_v3_ci_coverage")
    assert view._CI_MAY_SHIP is bool(claim.value.get("any_ci_may_ship", False))


def test_ci_columns_are_gated_not_footnoted():
    assert 'if _CI_MAY_SHIP:\n        display_cols += ["ci_low", "ci_high"]' in _VIEW_SRC


# ---------------------------------------------------------------------------
# 4. Column parity across the two scoring paths
# ---------------------------------------------------------------------------

def test_view_can_display_either_adjusted_woba_construction():
    """Legacy emits ``park_adj_woba``; ridge emits ``context_neutral_woba``.

    Different constructions, so different labels -- the page must be able to
    show whichever the producing model actually emitted.
    """
    assert "context_neutral_woba" in _VIEW_SRC
    assert "Ctx-Neutral wOBA" in _VIEW_SRC


def test_ridge_scoring_path_emits_the_columns_the_page_needs():
    assert 'df["context_neutral_woba"] = float(fit.intercept)' in _PRECOMPUTE_SRC
    assert "_traditional_war_frame(conn, season), on=\"player_id\"" in _PRECOMPUTE_SRC


# ---------------------------------------------------------------------------
# 5. Rename policy
# ---------------------------------------------------------------------------

def test_page_states_the_rename_and_disclaims_causal_identification():
    assert "renamed 2026-08-10" in view._RENAME_NOTE
    assert "not causal identification" in view._RENAME_NOTE


def test_no_causal_identification_claim_on_the_page():
    """`causal identification` may appear only inside an explicit negation."""
    for lineno, line in enumerate(_VIEW_SRC.splitlines(), start=1):
        if "causal identification" in line:
            assert "not causal identification" in line, (
                f"src/dashboard/views/causal_war.py:{lineno} asserts causal "
                f"identification: {line.strip()!r}"
            )
