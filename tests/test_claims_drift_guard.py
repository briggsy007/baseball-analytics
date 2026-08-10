"""Claims-registry unit tests + dashboard drift guard (WS2.2).

Two jobs:

1. Contract tests for ``src/claims.py``: retracted claims raise
   :class:`ClaimRetracted` (a retracted number is structurally unable to
   render through the API), every entry carries a mandatory caveat, and the
   mtime cache picks up registry edits.

2. Drift guard (audit failure class F-B, "claims drift"): scan
   ``src/dashboard/views/`` for hardcoded numeric literals adjacent to metric
   keywords. Every such line must either render through ``get_claim`` or
   carry a ``claim:<id>`` annotation whose id exists in
   ``docs/claims/claims.yaml``. New hand-copied headline numbers fail CI
   until they are registered.

The guard is purely file-based: no DB access, no dashboard launch.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from src.claims import (
    REGISTRY_PATH,
    ClaimNotFound,
    ClaimRetracted,
    RegistryInvalid,
    claim_ids,
    get_claim,
    load_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
VIEWS_DIR = REPO_ROOT / "src" / "dashboard" / "views"

# ---------------------------------------------------------------------------
# Drift-guard scanner
# ---------------------------------------------------------------------------

# Metric keywords (plan §2.2: "ECE|AUC|hit rate|r=|rho=|correlation|calibrat|%").
# The bare "%" trigger is implemented as the decimal-percent / ratio number
# rules below so that Streamlit format specs (":.1%", "%.3f") and UI prose
# ("0-100%") do not false-positive.
_KEYWORD = re.compile(
    r"(\bECE\b|\bAUC\b|hit[ -]rate|\br\s*=\s*[-+0-9.]|\brho\s*=\s*[-+0-9.]"
    r"|correlat|calibrat|log[-_ ]loss|\bCI\s*\[)",
    re.IGNORECASE,
)
# Hardcoded decimal literal (0.6406, 2.34, ...). Excludes %-suffixed matches
# (handled by _PCT_DECIMAL) and the trivial 0.0 / 1.0 defaults.
_DECIMAL = re.compile(r"(?<![\w.])\d+\.\d+(?![\d%])")
_TRIVIAL_DECIMALS = {"0.0", "1.0"}
# Decimal percent literal (68.4%, 77.5%). The lookbehind rejects format
# specs like ":.1%" and attribute-ish contexts.
_PCT_DECIMAL = re.compile(r"(?<![:.\w])\d+\.\d+\s*%")
# Quoted-ratio pattern ("13/19 = 68.4%", "13/25 = 52%").
_RATIO = re.compile(r"\b\d+\s*/\s*\d+\s*=")
# Annotation: "# claim:<id>" in code, "(claim:<id>)" / "*(claim:<id>)*" in
# strings and markdown (renders as a provenance tag -- intentional).
_ANNOTATION = re.compile(r"claim:\s*([A-Za-z0-9_]+)")


def _line_is_flagged(line: str) -> bool:
    """True if the line carries a hardcoded metric literal."""
    if _PCT_DECIMAL.search(line) or _RATIO.search(line):
        return True
    if _KEYWORD.search(line):
        decimals = [m for m in _DECIMAL.findall(line) if m not in _TRIVIAL_DECIMALS]
        if decimals:
            return True
    return False


def _scan_views() -> list[tuple[str, int, str, list[str]]]:
    """Yield (relpath, lineno, line, annotation_ids) for every flagged line."""
    flagged: list[tuple[str, int, str, list[str]]] = []
    for path in sorted(VIEWS_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _line_is_flagged(line):
                flagged.append((rel, lineno, line, _ANNOTATION.findall(line)))
    return flagged


# ---------------------------------------------------------------------------
# Registry contract tests
# ---------------------------------------------------------------------------


def test_registry_loads_and_every_claim_has_mandatory_caveat():
    registry = load_registry()
    assert len(registry) >= 20, "seed registry unexpectedly small"
    for claim in registry.values():
        assert claim.caveat.strip(), f"claim '{claim.id}' has an empty caveat"
        assert claim.status in {"active", "narrowed", "retracted", "superseded"}


def test_get_claim_returns_value_plus_caveat():
    claim = get_claim("dpi_oaa_2025_r")
    assert claim.value == 0.6406
    assert "0.45" in claim.caveat  # threshold context travels with the value
    assert claim.text().startswith("0.6406 (")

    buy_low = get_claim("causal_war_buy_low_68_4")
    assert buy_low.status == "narrowed"
    assert buy_low.value == "13/19 = 68.4%"
    assert "post-hoc" in buy_low.caveat
    assert "52%" in buy_low.caveat  # ITT bound is part of the mandatory caveat


def test_retracted_lstm_13_80_raises():
    """Plan §2.2 acceptance: a retracted claim cannot render on the dashboard."""
    with pytest.raises(ClaimRetracted) as excinfo:
        get_claim("lstm_13_80")
    assert "RETRACTED" in str(excinfo.value)
    assert excinfo.value.superseded_by == "pitchgpt_vs_lstm_10k"


@pytest.mark.parametrize(
    "retracted_id",
    [
        "lstm_13_80",
        "dpi_babip_corroboration",
        "causal_war_buy_low_validated",
        "pitchgpt_woba_pa_pass_pre062",
        "pitchgpt_perfect_calibration",
        "contrarian_70_80_pct",
        "mechanix_ae_injury_prediction",
        "vwr_flagship_promotion",
    ],
)
def test_all_seeded_retractions_raise(retracted_id):
    with pytest.raises(ClaimRetracted):
        get_claim(retracted_id)


def test_retracted_set_matches_registry():
    """Every registry entry with status=retracted actually raises."""
    registry = load_registry()
    retracted = [c.id for c in registry.values() if c.status == "retracted"]
    assert retracted, "registry should seed retracted entries"
    for cid in retracted:
        with pytest.raises(ClaimRetracted):
            get_claim(cid)


def test_unknown_claim_raises_claim_not_found():
    with pytest.raises(ClaimNotFound):
        get_claim("definitely_not_a_registered_claim")


@pytest.mark.parametrize(
    "body, fragment",
    [
        ("claims: {}", "expected a mapping with a 'claims' list"),
        ("claims:\n  - 42\n", "non-mapping claim entry"),
        ("claims:\n  - status: active\n    caveat: c\n", "missing string 'id'"),
        (
            "claims:\n"
            "  - id: dup\n    status: active\n    caveat: c\n"
            "  - id: dup\n    status: active\n    caveat: c\n",
            "duplicate claim id",
        ),
        ("claims:\n  - id: x\n    status: bogus\n    caveat: c\n", "invalid status"),
        ("claims:\n  - id: x\n    status: active\n", "no caveat"),
        (
            "claims:\n"
            "  - id: x\n    status: retracted\n    caveat: c\n"
            "    superseded_by: missing_id\n",
            "superseded_by unknown id",
        ),
    ],
)
def test_registry_validation_rejects_malformed_files(tmp_path, body, fragment):
    reg = tmp_path / "claims.yaml"
    reg.write_text(body, encoding="utf-8")
    with pytest.raises(RegistryInvalid, match=re.escape(fragment)):
        load_registry(reg)


def test_mtime_cache_reloads_on_change(tmp_path):
    reg = tmp_path / "claims.yaml"
    template = (
        "schema_version: 1\n"
        "claims:\n"
        "  - id: demo\n"
        "    model: demo\n"
        "    metric: demo metric\n"
        "    value: {value}\n"
        "    status: active\n"
        "    caveat: demo caveat\n"
    )
    reg.write_text(template.format(value="1"), encoding="utf-8")
    assert get_claim("demo", path=reg).value == 1

    reg.write_text(template.format(value="2"), encoding="utf-8")
    # Force an mtime bump in case the writes land inside one clock tick.
    stat = reg.stat()
    os.utime(reg, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert get_claim("demo", path=reg).value == 2


# ---------------------------------------------------------------------------
# Drift guard
# ---------------------------------------------------------------------------


def test_drift_guard_every_metric_literal_is_claim_annotated():
    """Hardcoded metric literals in views must be registered.

    A flagged line complies if it (a) carries a ``claim:<id>`` annotation for
    a registered id, or (b) renders through ``get_claim`` on the same line.
    Anything else is claims drift (audit failure class F-B) and fails here.
    """
    known_ids = set(claim_ids())
    violations: list[str] = []
    for rel, lineno, line, annotations in _scan_views():
        if "get_claim(" in line:
            continue
        valid = [a for a in annotations if a in known_ids]
        if not valid:
            violations.append(f"{rel}:{lineno}: {line.strip()[:120]}")
    assert not violations, (
        "Unregistered metric literals in dashboard views (add a claims.yaml "
        "entry and a 'claim:<id>' annotation, or render via get_claim):\n"
        + "\n".join(violations)
    )


def test_drift_guard_annotations_reference_real_claims():
    """Every claim:<id> annotation anywhere in views must exist in the registry."""
    known_ids = set(claim_ids())
    bad: list[str] = []
    for path in sorted(VIEWS_DIR.glob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            for annotation in _ANNOTATION.findall(line):
                if annotation not in known_ids:
                    bad.append(f"{rel}:{lineno}: claim:{annotation}")
    assert not bad, "Annotations referencing unknown claim ids:\n" + "\n".join(bad)


def test_drift_guard_scanner_still_has_teeth():
    """Self-test: the scanner flags a canonical hand-copied claim line."""
    assert _line_is_flagged('"Buy-Low hit rate is 68.4% on 2025 outcomes"')
    assert _line_is_flagged('"post-temperature ECE 0.0114 on holdout"')
    assert _line_is_flagged('"intention-to-treat is 13/25 = 52%"')
    assert _line_is_flagged('"DPI vs OAA r = 0.6406 in 2025"')
    # ...and does not flag routine formatting / UI code.
    assert not _line_is_flagged('f"{row.get(\'sb_attempt_rate\', 0):.1%}"')
    assert not _line_is_flagged('"Fatigue Score (0-100%) accounts for rest"')
    assert not _line_is_flagged("marker=dict(size=4, opacity=0.6),")


# ---------------------------------------------------------------------------
# Retracted numbers may not appear unbounded in views
# ---------------------------------------------------------------------------

_BANNED_SUBSTRINGS = [
    "13.80",  # retired LSTM margin (claim lstm_13_80) -- banned outright
    "13.8%",
    "70-80%",
    "70–80%",  # en-dash variant
    "perfect mathematical calibration",
    "68.4% validated",
    "validated 68.4",
]


def test_banned_retracted_strings_absent_from_views():
    hits: list[str] = []
    for path in sorted(VIEWS_DIR.glob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8", errors="replace")
        for banned in _BANNED_SUBSTRINGS:
            if banned in text:
                hits.append(f"{rel}: contains banned string {banned!r}")
    assert not hits, "\n".join(hits)


@pytest.mark.parametrize("marker", ["68.4", "0.387", "0.768"])
def test_bounded_retracted_numbers_only_on_annotated_lines(marker):
    """Numbers tied to retracted claims may appear only on claim-annotated
    lines (demotion banners / bounded evidence copy)."""
    known_ids = set(claim_ids())
    unbounded: list[str] = []
    for path in sorted(VIEWS_DIR.glob("*.py")):
        rel = path.relative_to(REPO_ROOT).as_posix()
        for lineno, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if marker in line and "get_claim(" not in line:
                if not any(a in known_ids for a in _ANNOTATION.findall(line)):
                    unbounded.append(f"{rel}:{lineno}: {line.strip()[:120]}")
    assert not unbounded, (
        f"'{marker}' appears outside claim-annotated lines:\n" + "\n".join(unbounded)
    )


def test_registry_file_lives_at_expected_path():
    assert REGISTRY_PATH == REPO_ROOT / "docs" / "claims" / "claims.yaml"
    assert REGISTRY_PATH.exists()
