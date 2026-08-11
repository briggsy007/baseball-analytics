"""
PitchGPT v3 — write-once registry registration (spec §8.1).

    "Registry versions are `pitchgpt/v<YYYY.MM.DD>-factorized` etc., registered
    write-once with `manifest.json` per WS2.1; **no alias change**
    (`production` / `frozen_validated` stay pointed at `v2026.04.23`) unless and
    until §6.8 returns PASS, at which point the alias move is its own reviewed
    step."

This script registers the three v3 artifacts and **never touches an alias**.
It refuses to run if any alias would move, and it re-verifies that the frozen
v2 blobs are byte-identical before and after.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.registry import ModelRegistry, sha256_of_file  # noqa: E402

SPEC = "docs/pitchgpt_sim_engine/PITCHGPT_V2_SPEC.md"
RESULTS = "docs/pitchgpt_sim_engine/V2_BUILD_RESULTS_2026-08.md"
FROZEN = {
    "models/pitchgpt_v2.pt":
        "6f952054d14ac6f918f3eb9502b496b70bc0c87dfc65dc50d98ee7244a62883c",
    "models/pitchgpt_v2_outcomehead_a1.pt":
        "37b50e87599013c281560c9f63286fe5b7645297d0042694d907287417bb25e5",
}

NOTES_COMMON = (
    "WS5.2 chain-rule factorized successor (spec {spec}, frozen at b61e05b). "
    "DEV-TIER ONLY: graded on the 2024 BURNED dev cohort; the single §5.5 "
    "sealed-2026 lockbox contact has NOT been made, so this artifact carries no "
    "validation authority and no alias. 2025 untouched (12/14 contacts stand)."
).format(spec=SPEC)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version-date", default="v2026.08.11")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--only",
        action="append",
        default=None,
        help=(
            "Register only the named artifact file(s). Registration is "
            "write-once, so an artifact added to the plan after the first run "
            "(e.g. the §6.3 pinned-bandwidth sidecar) needs this to avoid "
            "re-registering — and re-raising FileExistsError on — the ones "
            "already recorded."
        ),
    )
    args = ap.parse_args()

    for rel, want in FROZEN.items():
        got = sha256_of_file(_ROOT / rel)
        if got != want:
            raise SystemExit(
                f"REFUSING TO REGISTER: frozen artifact {rel} has sha256 {got}, "
                f"expected the registry-pinned {want}."
            )

    reg = ModelRegistry(_ROOT / "models")
    before = json.loads((_ROOT / "models" / "registry.json").read_text())
    aliases_before = {
        k: v for k, v in before.get("pitchgpt", {}).items() if k != "history"
    }

    plan = [
        (
            f"{args.version_date}-factorized",
            "pitchgpt_v3_factorized.pt",
            "scripts/pitchgpt_v3_train_stage_b.py",
            "Factorized backbone + chain-rule heads after Stage A pretrain "
            "(2015-2022, 5 epochs) and the §4.3 B1/B2/B3 curriculum "
            "rollout-aware fine-tune. Output stack 28,240 params vs the flat "
            "head's 285,090 (10.1x drop, §3.3 ceiling 45,000).",
        ),
        (
            f"{args.version_date}-factorized-outcomehead",
            "pitchgpt_v3_outcomehead.pt",
            "scripts/pitchgpt_v3_train_outcome_head.py",
            "7-class outcome head on the frozen v3 backbone. A1 topology "
            "(211->128->64->7) with §3.5's one [FIXED] change: NO "
            "inverse-frequency class weighting.",
        ),
        (
            f"{args.version_date}-factorized-calibration",
            "calibration_pitchgpt_v3.json",
            "scripts/pitchgpt_v3_calibrate.py",
            "§4.4 per-head temperature scalars (T_type/T_zone/T_velo/"
            "T_outcome), fit by NLL minimisation on the 2023 pitcher-disjoint "
            "slice ONLY. No vectors, no matrices, no per-position or per-count "
            "tables (§0.2). Provenance sidecar declares fit_cohort_season=2023; "
            "the loader refuses 2025/2026 (§7.5).",
        ),
        (
            f"{args.version_date}-factorized-kce-bandwidths",
            "kce_bandwidths_pitchgpt_v3_dev2024.json",
            "scripts/pitchgpt_v3_g2_bandwidth_fix.py",
            "§6.3 per-head KCE kernel bandwidths ('set by the median heuristic "
            "on the dev cohort, fixed before the contact and recorded'), fitted "
            "once on v3's 2024 BURNED-dev probabilities and written write-once. "
            "The §5.5 graded run replays these via graded_skce_test() instead of "
            "refitting the kernel on the cohort under grade; the loader refuses "
            "any artifact not declaring fit_cohort_season=2024. §9 entry 16.",
        ),
    ]
    if args.only:
        wanted = set(args.only)
        plan = [p for p in plan if p[1] in wanted]
        if not plan:
            raise SystemExit(f"--only matched no planned artifact: {sorted(wanted)}")

    out = []
    for version, artifact, script, note in plan:
        path = _ROOT / "models" / artifact
        if not path.exists():
            raise SystemExit(f"Artifact missing: {path}")
        entry = {
            "version": version,
            "artifact": artifact,
            "sha256": sha256_of_file(path),
        }
        if args.dry_run:
            out.append({**entry, "dry_run": True})
            continue
        m = reg.register_version(
            "pitchgpt",
            version,
            path,
            hash_policy="pinned",
            train_window="2015-2022 train / 2023 fit+calibration / 2024 burned-dev eval",
            training_script=script,
            spec_version=SPEC,
            validation_results_ref=RESULTS,
            notes=NOTES_COMMON + " " + note,
        )
        out.append(m)
        print(f"registered pitchgpt/{version}  sha256 {m['sha256'][:16]}")

    after = json.loads((_ROOT / "models" / "registry.json").read_text())
    aliases_after = {
        k: v for k, v in after.get("pitchgpt", {}).items() if k != "history"
    }
    if aliases_before != aliases_after:
        raise SystemExit(
            f"ALIAS DRIFT — §8.1 forbids it: {aliases_before} -> {aliases_after}"
        )
    print(f"aliases unchanged: {aliases_after}")

    for rel, want in FROZEN.items():
        assert sha256_of_file(_ROOT / rel) == want, rel
    print("frozen v2 blobs byte-identical.")
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
