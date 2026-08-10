"""Holdout-contact ledger + budget enforcement (plan WS2.3, 2026-08-10).

The ledger lives at ``docs/holdout_ledger.jsonl`` (append-only, tracked in
git). It exists because the platform exhausted its 2025 pitcher-disjoint
holdout (~13 evaluation contacts, all on the same seed-42 subsample) without
any record of the spending — audit failure class **F-C: holdout exhaustion**.

Tier policy (declared in the ledger's header entry; summarized here):

* ``pitchgpt_2025_pitcher_disjoint`` — **budgeted validation tier.** The
  historical contacts are backfilled-reconstructed entries; the remaining
  budget is what the pre-registered Phase 0.6.2 protocol allows and nothing
  more. Contacts past budget require an explicit, logged override.
* ``pitchgpt_2024`` — **BURNED (dev tier).** Three committed eval scripts
  used ``TEST_RANGE = (2024, 2024)``; 2024 carries no validation authority.
  Dev-tier contacts are not budget-limited and not required to be logged.
* ``lockbox_2026_full_season`` — **LOCKBOX.** Sealed until the 2026 MLB
  regular season ends; then ONE pre-registered contact per frozen spec
  version. Any pre-unseal contact attempt is refused (override is logged
  and shames the caller in the ledger forever).

Usage::

    from src.holdout import holdout_access

    @holdout_access(
        dataset="pitchgpt_2025_pitcher_disjoint",
        purpose="Phase 0.6.2 single evaluation run (pre-registered)",
        budget=14,
        metrics=["K%", "BB%", "HR%", "wOBA", "PA-length"],
    )
    def run_2025_eval(...):
        ...

The decorator refuses to run when the dataset's recorded contact count has
reached the budget, unless called with ``override=True`` — and the override
itself is appended to the ledger. Every completed call appends::

    {ts, dataset, sha256, purpose, git_sha, metrics_revealed, contact_number}
"""

from __future__ import annotations

import functools
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from src.pick_ledger import current_git_sha  # same provenance helper

ROOT: Path = Path(__file__).resolve().parents[1]
LEDGER_PATH: Path = ROOT / "docs" / "holdout_ledger.jsonl"

#: Datasets with a registered artifact hash get it verified on every contact.
#: The 2025 pitcher-disjoint cohort is derived live from DuckDB (no frozen
#: file exists yet), so no hash is registered for it; when a cohort file is
#: frozen, register it here AND in a ledger entry.
DATASET_HASHES: dict[str, str] = {}


class HoldoutBudgetExceeded(RuntimeError):
    """Raised when a contact would exceed the dataset's declared budget."""


class HoldoutSealedError(RuntimeError):
    """Raised on any contact attempt against a still-sealed lockbox tier."""


class HoldoutHashMismatch(RuntimeError):
    """Raised when a registered dataset hash does not match the file on disk."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_ledger(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"{path} line {lineno} is not valid JSON: {exc}"
                ) from exc
    return entries


def _append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_of(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def contact_count(dataset: str, ledger_path: Path | str | None = None) -> int:
    """Number of recorded contacts (including backfilled) for ``dataset``."""
    path = Path(ledger_path) if ledger_path else LEDGER_PATH
    return sum(
        1
        for e in _read_ledger(path)
        if e.get("entry_type") == "contact" and e.get("dataset") == dataset
    )


def dataset_policy(
    dataset: str, ledger_path: Path | str | None = None
) -> dict | None:
    """Tier policy for ``dataset`` from the ledger header entry, if declared."""
    path = Path(ledger_path) if ledger_path else LEDGER_PATH
    for e in _read_ledger(path):
        if e.get("entry_type") == "header":
            pol = (e.get("tier_policy") or {}).get(dataset)
            if pol is not None:
                return pol
    return None


def record_contact(
    dataset: str,
    purpose: str,
    metrics_revealed: Iterable[str] | None = None,
    dataset_path: Path | str | None = None,
    ledger_path: Path | str | None = None,
    override: bool = False,
    override_reason: str | None = None,
    extra: dict | None = None,
    ts: str | None = None,
) -> dict:
    """Append one contact entry (non-decorator form, e.g. for orchestrators)."""
    path = Path(ledger_path) if ledger_path else LEDGER_PATH

    sha: str | None = None
    registered = DATASET_HASHES.get(dataset)
    if dataset_path is not None:
        sha = sha256_of(dataset_path)
        if registered and sha != registered:
            raise HoldoutHashMismatch(
                f"dataset {dataset!r}: file {dataset_path} sha256 {sha} does "
                f"not match registered hash {registered} — refusing to log a "
                "contact against a mutated dataset."
            )
    elif registered:
        # A hash is registered but no file was offered for verification.
        raise HoldoutHashMismatch(
            f"dataset {dataset!r} has a registered hash; pass dataset_path "
            "so it can be verified."
        )

    n = contact_count(dataset, path) + 1
    entry = {
        "entry_type": "contact",
        "ts": ts or _utcnow(),
        "dataset": dataset,
        "sha256": sha,
        "purpose": purpose,
        "git_sha": current_git_sha(),
        "metrics_revealed": sorted(metrics_revealed) if metrics_revealed else [],
        "contact_number": n,
    }
    if override:
        entry["override"] = True
        entry["override_reason"] = override_reason or "unspecified"
    if extra:
        for k, v in extra.items():
            entry.setdefault(k, v)
    _append(path, entry)
    return entry


def _enforce_budget(
    dataset: str,
    budget: int | None,
    override: bool,
    ledger_path: Path,
) -> None:
    policy = dataset_policy(dataset, ledger_path) or {}
    tier = policy.get("tier")

    if tier == "lockbox" and not policy.get("unsealed", False):
        if not override:
            raise HoldoutSealedError(
                f"dataset {dataset!r} is a sealed LOCKBOX (policy: "
                f"{policy.get('seal_rule', 'sealed until unseal entry')}). "
                "No contact is permitted before the unseal entry is appended. "
                "override=True would log the violation permanently."
            )
        return  # override falls through; the contact entry records it

    effective: int | None = budget
    header_budget = policy.get("budget")
    if isinstance(header_budget, int):
        effective = (
            header_budget if effective is None else min(effective, header_budget)
        )
    if effective is None:
        return  # no budget declared anywhere (dev tier)

    n = contact_count(dataset, ledger_path)
    if n >= effective and not override:
        raise HoldoutBudgetExceeded(
            f"dataset {dataset!r} has {n} recorded contacts; budget is "
            f"{effective}. Refusing contact #{n + 1}. If this contact is "
            "genuinely pre-registered, raise the budget via a dated ledger "
            "entry + spec amendment, or pass override=True (the override is "
            "itself logged, permanently)."
        )


def holdout_access(
    dataset: str,
    purpose: str,
    budget: int | None = None,
    metrics: Iterable[str] | None = None,
    dataset_path: Path | str | None = None,
    ledger_path: Path | str | None = None,
    override: bool = False,
    override_reason: str | None = None,
) -> Callable:
    """Decorator: gate a holdout-touching evaluation behind the ledger.

    * verifies the dataset hash where one is registered;
    * refuses when contacts >= budget (or the ledger header's budget,
      whichever is smaller) unless ``override=True``;
    * refuses sealed-lockbox contacts outright unless ``override=True``;
    * on completion appends a contact entry (overrides included — the
      override is logged, not hidden);
    * ``metrics_revealed`` = the ``metrics`` argument, plus the returned
      dict's keys when the wrapped function returns a dict.
    """

    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            lpath = Path(ledger_path) if ledger_path else LEDGER_PATH
            _enforce_budget(dataset, budget, override, lpath)
            result = fn(*args, **kwargs)
            revealed = set(metrics or [])
            if isinstance(result, dict):
                revealed.update(str(k) for k in result.keys())
            record_contact(
                dataset=dataset,
                purpose=purpose,
                metrics_revealed=revealed,
                dataset_path=dataset_path,
                ledger_path=lpath,
                override=override,
                override_reason=override_reason,
                extra={"function": getattr(fn, "__qualname__", str(fn))},
            )
            return result

        return wrapper

    return deco
