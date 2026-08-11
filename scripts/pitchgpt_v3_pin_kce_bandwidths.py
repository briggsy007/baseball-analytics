"""PitchGPT v3 — serialize the §6.3 pinned kernel bandwidths from a G2 audit.

**This script measures nothing.**  It is a pure serialization step: it reads
the per-head bandwidths that ``scripts/pitchgpt_v3_g2_bandwidth_fix.py``
actually measured on the 2024 BURNED-dev cohort, writes them write-once to
``models/kce_bandwidths_pitchgpt_v3_dev2024.json`` with the full §6.3
provenance block, and drops a ``pin_record.json`` beside the source audit
carrying the pin's sha256 so the audit directory records the artifact hash.

Why it exists as a separate step: the measuring run wrote the pin before it
knew its own hash, so its ``audit.json`` names the pinned values but not the
pinned file's digest.  ``audit.json`` is never rewritten — the digest lands in
a NEW sibling file instead, and this script records exactly which audit it read
and which fields it copied, so the pin can be re-derived from the audit by
anyone.

Nothing here is a judgement call: every bandwidth is copied verbatim from
``per_head[<head>].v3_observed.bandwidth`` in the cited audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt_v3_data import DEV_SEASON  # noqa: E402
from src.analytics.pitchgpt_v3_gates import (  # noqa: E402
    KCE_BANDWIDTH_RULE,
    KCE_PINNED_BANDWIDTH_FILENAME,
    KCE_SEED,
    save_pinned_bandwidths,
)

DEFAULT_AUDIT = (
    "results/pitchgpt_v3/dev_2024_g2_bandwidth_fix_20260811T074725Z/audit.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit", type=Path, default=_ROOT / DEFAULT_AUDIT)
    args = ap.parse_args()

    audit_path = Path(args.audit)
    audit_sha = sha256_file(audit_path)
    audit = json.loads(audit_path.read_text())
    if audit.get("tier") != "burned-dev":
        raise SystemExit(
            f"refusing to pin from a non-dev audit: tier={audit.get('tier')!r}"
        )

    per_head = audit["per_head"]
    bandwidths = {h: v["v3_observed"]["bandwidth"] for h, v in per_head.items()}
    n_used = {h: v["v3_observed"]["n_used"] for h, v in per_head.items()}
    n_avail = {h: v["v3_observed"]["n_available"] for h, v in per_head.items()}

    # Every head's three SKCE evaluations in that audit must already share the
    # bandwidth being pinned, or the pin does not describe what was measured.
    for h, v in per_head.items():
        bw = bandwidths[h]
        for key in ("v3_observed", "power_probe", "frozen_v2_reference_SAME_kernel"):
            got = v[key]["bandwidth"]
            if abs(got - bw) > 1e-12:
                raise SystemExit(
                    f"{h}/{key} was measured at bandwidth {got}, not the pinned "
                    f"{bw}; the audit is not a single-kernel measurement."
                )

    bw_path = _ROOT / "models" / KCE_PINNED_BANDWIDTH_FILENAME
    payload = save_pinned_bandwidths(
        bw_path,
        bandwidths,
        {
            "fit_cohort_season": DEV_SEASON,
            "fit_cohort": "2024 pitcher-disjoint BURNED dev cohort",
            "fit_seed": KCE_SEED,
            "fit_n_used_per_head": n_used,
            "fit_n_available_per_head": n_avail,
            "fitted_on_model": "v3-factorized (the model under test)",
            "measured_by": "scripts/pitchgpt_v3_g2_bandwidth_fix.py",
            "produced_by": "scripts/pitchgpt_v3_pin_kce_bandwidths.py",
            "serialized_from_audit": str(
                audit_path.relative_to(_ROOT).as_posix()
            ),
            "serialized_from_audit_sha256": audit_sha,
            "spec_freeze_sha": audit["spec_freeze_sha"],
            "measurement_git_sha": audit["git_sha"],
        },
    )
    pin_sha = sha256_file(bw_path)
    print(f"pinned -> {bw_path}\n  sha256 {pin_sha}")

    record_path = audit_path.parent / "pin_record.json"
    if record_path.exists():
        raise SystemExit(f"{record_path} exists; refusing to overwrite.")
    record_path.write_text(
        json.dumps(
            {
                "what": (
                    "sha256 of the §6.3 pinned-bandwidth artifact produced from "
                    "this audit's measured bandwidths. It lives here rather "
                    "than inside audit.json because the measuring run wrote the "
                    "pin after assembling its payload and audit.json is never "
                    "rewritten."
                ),
                "audit": audit_path.name,
                "audit_sha256": audit_sha,
                "pinned_artifact": f"models/{KCE_PINNED_BANDWIDTH_FILENAME}",
                "pinned_artifact_sha256": pin_sha,
                "pinned_bandwidths": bandwidths,
                "rule": KCE_BANDWIDTH_RULE,
                "pinned_payload": payload,
                "written_by": "scripts/pitchgpt_v3_pin_kce_bandwidths.py",
                "written_utc": datetime.now(timezone.utc).isoformat(),
                "deviations_log_entry": 16,
            },
            indent=2,
        )
    )
    print(f"pin record -> {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
