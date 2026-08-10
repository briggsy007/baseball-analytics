#!/usr/bin/env python
"""Verify every registered model artifact against its manifest (WS2.1).

Recomputes the sha256 of each artifact registered in ``models/<name>/v*/
manifest.json`` and checks alias integrity in ``models/registry.json``:

  * ``hash_policy: pinned``  — missing file or hash mismatch is a FAILURE
    (frozen validation evidence must be byte-identical);
  * ``hash_policy: advisory`` — existence-checked only; hash drift is
    expected (nightly in-season retrains) and reported informationally;
  * every alias must point at a registered version, and
    ``frozen_validated`` must point at a *pinned* version.

Exit codes: 0 = all green (warnings allowed), 1 = at least one failure.

The nightly chain (``scripts/nightly_refresh.py``) runs this FIRST (abort
the run if frozen evidence was tampered with) and LAST (prove the run
itself modified nothing pinned).  ``tests/test_model_registry.py`` runs the
identical check in pytest via ``src.analytics.registry.verify_registry``.

Usage
-----
    python scripts/verify_artifacts.py                # verify models/
    python scripts/verify_artifacts.py --registry-dir <dir>
    python scripts/verify_artifacts.py --json         # machine-readable
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics.registry import verify_registry  # noqa: E402

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True,
    )

_STATUS_ORDER = {"fail": 0, "warn": 1, "ok": 2}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify registered model artifacts against their manifests."
    )
    ap.add_argument(
        "--registry-dir",
        default=None,
        help="Registry root (default: models/, or $BASEBALL_MODEL_REGISTRY_DIR).",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of the table.",
    )
    args = ap.parse_args()

    results = verify_registry(registry_dir=args.registry_dir)
    n_fail = sum(1 for r in results if r["status"] == "fail")
    n_warn = sum(1 for r in results if r["status"] == "warn")
    n_ok = sum(1 for r in results if r["status"] == "ok")

    if args.json:
        print(
            json.dumps(
                {
                    "results": results,
                    "n_ok": n_ok,
                    "n_warn": n_warn,
                    "n_fail": n_fail,
                    "verdict": "fail" if n_fail else "ok",
                },
                indent=2,
            )
        )
        return 1 if n_fail else 0

    print("ARTIFACT VERIFICATION (models/registry.json)")
    print("-" * 78)
    for r in sorted(
        results, key=lambda r: (_STATUS_ORDER[r["status"]], r["model"], r["version"])
    ):
        tag = {"ok": "[OK]  ", "warn": "[WARN]", "fail": "[FAIL]"}[r["status"]]
        print(f"{tag} {r['model']:<22s} {r['version']:<22s} {r['check']:<18s} {r['detail']}")
    print("-" * 78)
    print(f"ok={n_ok}  warn={n_warn}  fail={n_fail}")
    if n_fail:
        print(
            "VERDICT: FAIL — a pinned (frozen-validated) artifact or the "
            "registry index does not match its manifest. Do NOT run "
            "training/scoring until resolved."
        )
        return 1
    print("VERDICT: OK — all registered artifacts verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
