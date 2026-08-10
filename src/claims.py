"""Claims-registry accessor (WS2.2, 2026-08-10 platform improvement plan §2.2).

Single sanctioned code path between ``docs/claims/claims.yaml`` and any
surface that renders a reportable number (kill criterion K6: no claim ships
to dashboard/docs without a registry entry).

Contract
--------
- ``get_claim(claim_id)`` returns a :class:`Claim` whose ``caveat`` is a
  MANDATORY non-empty string -- render the value only alongside its caveat.
- Retracted entries raise :class:`ClaimRetracted`; a retracted claim is
  structurally unable to render through this API. Views may reference a
  retracted id only inside bounding copy (demotion banners, errata) via a
  ``claim:<id>`` annotation, never via ``get_claim``.
- The YAML is parsed once and cached; the cache invalidates on file
  mtime/size change, so registry edits are picked up without a restart.

The drift-guard test (``tests/test_claims_drift_guard.py``) enforces that
hardcoded metric literals in ``src/dashboard/views/`` either render through
``get_claim`` or carry a ``claim:<id>`` annotation whose id exists here.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

__all__ = [
    "Claim",
    "ClaimError",
    "ClaimNotFound",
    "ClaimRetracted",
    "RegistryInvalid",
    "REGISTRY_PATH",
    "claim_ids",
    "get_claim",
    "load_registry",
]

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "docs" / "claims" / "claims.yaml"

_VALID_STATUSES = frozenset({"active", "narrowed", "retracted", "superseded"})


class ClaimError(Exception):
    """Base class for claims-registry errors."""


class ClaimNotFound(ClaimError, KeyError):
    """Raised when a claim id does not exist in the registry."""


class ClaimRetracted(ClaimError):
    """Raised when a retracted claim is requested for rendering."""

    def __init__(self, claim_id: str, reason: str, superseded_by: str | None):
        self.claim_id = claim_id
        self.reason = reason
        self.superseded_by = superseded_by
        pointer = f" Superseded by: {superseded_by}." if superseded_by else ""
        super().__init__(
            f"Claim '{claim_id}' is RETRACTED and cannot render. {reason}{pointer}"
        )


class RegistryInvalid(ClaimError):
    """Raised when claims.yaml fails structural validation."""


@dataclass(frozen=True)
class Claim:
    """One registry entry. ``value`` + ``caveat`` travel together."""

    id: str
    model: str
    artifact_version: str | None
    metric: str
    value: Any
    ci: str | None
    dataset: Mapping[str, Any] | None
    source_doc: str | None
    status: str
    superseded_by: str | None
    effective: str | None
    caveat: str

    def text(self) -> str:
        """Render ``value (caveat)`` -- the minimum honest quotable string."""
        return f"{self.value} ({self.caveat})"


# --------------------------------------------------------------------------
# Cached loading
# --------------------------------------------------------------------------

_lock = threading.Lock()
_cache: dict[Path, tuple[tuple[int, int], dict[str, Claim]]] = {}


def _parse(path: Path) -> dict[str, Claim]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("claims"), list):
        raise RegistryInvalid(f"{path}: expected a mapping with a 'claims' list")

    claims: dict[str, Claim] = {}
    for entry in raw["claims"]:
        if not isinstance(entry, dict):
            raise RegistryInvalid(f"{path}: non-mapping claim entry: {entry!r}")
        cid = entry.get("id")
        if not cid or not isinstance(cid, str):
            raise RegistryInvalid(f"{path}: claim entry missing string 'id': {entry!r}")
        if cid in claims:
            raise RegistryInvalid(f"{path}: duplicate claim id '{cid}'")
        status = entry.get("status")
        if status not in _VALID_STATUSES:
            raise RegistryInvalid(
                f"{path}: claim '{cid}' has invalid status {status!r} "
                f"(must be one of {sorted(_VALID_STATUSES)})"
            )
        caveat = entry.get("caveat")
        if not isinstance(caveat, str) or not caveat.strip():
            raise RegistryInvalid(
                f"{path}: claim '{cid}' has no caveat -- the caveat is mandatory "
                "for every status (it is the retraction reason for retracted "
                "entries)"
            )
        claims[cid] = Claim(
            id=cid,
            model=str(entry.get("model", "")),
            artifact_version=entry.get("artifact_version"),
            metric=str(entry.get("metric", "")),
            value=entry.get("value"),
            ci=entry.get("ci"),
            dataset=entry.get("dataset"),
            source_doc=entry.get("source_doc"),
            status=status,
            superseded_by=entry.get("superseded_by"),
            effective=str(entry["effective"]) if entry.get("effective") else None,
            caveat=" ".join(caveat.split()),
        )

    # Referential integrity for supersession pointers.
    for claim in claims.values():
        if claim.superseded_by and claim.superseded_by not in claims:
            raise RegistryInvalid(
                f"{path}: claim '{claim.id}' superseded_by unknown id "
                f"'{claim.superseded_by}'"
            )
    return claims


def load_registry(path: Path | None = None) -> dict[str, Claim]:
    """Load (and cache by mtime/size) the full registry as ``{id: Claim}``."""
    p = Path(path) if path is not None else REGISTRY_PATH
    stat = p.stat()
    key = (stat.st_mtime_ns, stat.st_size)
    with _lock:
        cached = _cache.get(p)
        if cached is not None and cached[0] == key:
            return cached[1]
        claims = _parse(p)
        _cache[p] = (key, claims)
        return claims


def claim_ids(path: Path | None = None) -> tuple[str, ...]:
    """All registered claim ids (any status)."""
    return tuple(load_registry(path).keys())


def get_claim(claim_id: str, path: Path | None = None) -> Claim:
    """Return the claim for rendering; raise :class:`ClaimRetracted` if retracted.

    The returned :class:`Claim` carries the mandatory caveat -- surfaces must
    render it (or the substance of it) alongside the value.
    """
    registry = load_registry(path)
    claim = registry.get(claim_id)
    if claim is None:
        raise ClaimNotFound(
            f"Unknown claim id '{claim_id}' -- register it in {REGISTRY_PATH} "
            "before quoting the number (kill criterion K6)."
        )
    if claim.status == "retracted":
        raise ClaimRetracted(claim_id, claim.caveat, claim.superseded_by)
    return claim
