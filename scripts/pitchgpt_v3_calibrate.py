"""
PitchGPT v3 — per-head temperature calibration (spec §4.4).

    "[FIXED] One temperature scalar per head (T_type, T_zone, T_velo,
    T_outcome), each fit by NLL minimization on the 2023 pitcher-disjoint slice
    only (§5.2). No vectors, no matrices, no per-position or per-count tables —
    those are banned by §0.2. Each temperature is written into a sidecar
    carrying the provenance schema already enforced by the guard
    (fit_cohort_season, fit_seed, fit_n_pas, produced_by), and
    fit_cohort_season must be 2023."

The fit cohort is 2023; every gate cohort is 2024 (dev) or the sealed 2026
lockbox, so no calibration vector is ever fit on a cohort a gate is evaluated
on (§7.5 / K5 second clause).  ``validate_calibration_provenance`` is called on
the payload before it is written, so a run that would violate the guard aborts
rather than warns.

Writes ``models/calibration_pitchgpt_v3.json`` (write-once) and a run audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt_v3 import (  # noqa: E402
    SPEC_BODY_SHA256,
    SPEC_FREEZE_SHA,
    SPEC_PATH,
    HeadTemperatures,
    V3OutcomeHead,
    load_v3_checkpoint,
    validate_calibration_provenance,
)
from src.analytics.pitchgpt_v3_data import (  # noqa: E402
    FIT_SEASON,
    build_pa_dataset,
    load_sequence_cache,
)
from src.analytics.pitchgpt_v3_infer import (  # noqa: E402
    collect_outcome_probs,
    collect_v3_head_probs,
    fit_temperature,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitchgpt_v3_calibrate")

DEFAULT_CACHE_DIR = Path(
    r"C:\Users\hunte\AppData\Local\Temp\claude"
    r"\C--Users-hunte-projects-baseball"
    r"\5112f9f4-f7ff-4db3-bd24-90acc5fbef27\scratchpad\pitchgpt_v3_cache"
)
SEED = 42
HORIZON = 6


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument(
        "--backbone", type=Path,
        default=_ROOT / "models" / "pitchgpt_v3_factorized.pt",
    )
    ap.add_argument(
        "--outcome-head", type=Path,
        default=_ROOT / "models" / "pitchgpt_v3_outcomehead.pt",
    )
    ap.add_argument(
        "--out", type=Path,
        default=_ROOT / "models" / "calibration_pitchgpt_v3.json",
    )
    ap.add_argument("--tag", type=str, default="calibration")
    args = ap.parse_args()

    if args.out.exists():
        raise SystemExit(f"§0.4 forbids overwriting artifacts: {args.out} exists.")

    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, bb_ck = load_v3_checkpoint(args.backbone, device=device)
    model.to(device).eval()

    head_ck = torch.load(args.outcome_head, map_location=device, weights_only=False)
    head = V3OutcomeHead().to(device)
    head.load_state_dict(head_ck["head_state_dict"])
    head.eval()

    pa_fit = build_pa_dataset(
        load_sequence_cache(args.cache_dir / "fit23.npz"), horizon=HORIZON,
    )
    n_pas = int(len(pa_fit["pa_len"]))
    logger.info("2023 pitcher-disjoint fit cohort: %d PAs", n_pas)

    hp = collect_v3_head_probs(model, pa_fit, device)
    logger.info("collected %d per-pitch head rows", len(hp["y_type"]))
    fits = {
        "type": fit_temperature(hp["p_type"], hp["y_type"]),
        "zone": fit_temperature(hp["p_zone"], hp["y_zone"]),
        "velo": fit_temperature(hp["p_velo"], hp["y_velo"]),
    }
    op = collect_outcome_probs(model, head, pa_fit, device)
    fits["outcome"] = fit_temperature(op["probs"], op["labels"])

    for k, v in fits.items():
        logger.info(
            "T_%-7s = %.4f   NLL %.5f -> %.5f  (n=%d)",
            k, v["T"], v["nll_at_1"], v["nll_at_T"], v["n"],
        )

    temps = HeadTemperatures(
        type_T=fits["type"]["T"],
        zone_T=fits["zone"]["T"],
        velo_T=fits["velo"]["T"],
        outcome_T=fits["outcome"]["T"],
    )

    payload = {
        "artifact": "calibration_pitchgpt_v3",
        "temperatures": temps.to_dict(),
        # §4.4 provenance schema (the guard enforces all four keys).
        "fit_cohort_season": FIT_SEASON,
        "fit_seed": SEED,
        "fit_n_pas": n_pas,
        "produced_by": "scripts/pitchgpt_v3_calibrate.py",
        "fit_cohort": "2023 pitcher-disjoint (2015-2022 pitchers excluded)",
        "fit_cohort_meta": pa_fit["meta"],
        "form": "one scalar per head; vectors/matrices banned by §0.2",
        "spec_path": SPEC_PATH,
        "spec_freeze_sha": SPEC_FREEZE_SHA,
        "spec_body_sha256": SPEC_BODY_SHA256,
        "git_sha": git_sha(),
        "backbone_checkpoint": str(args.backbone),
        "outcome_head_checkpoint": str(args.outcome_head),
        "fit_detail": fits,
        "gate_cohorts_never_fit_on": ["2024 (dev)", "2025 (budgeted)", "2026 (lockbox)"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    # Structural §7.5 check BEFORE the artifact exists on disk.
    validate_calibration_provenance(payload)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sha = hashlib.sha256(args.out.read_bytes()).hexdigest()
    logger.info("calibration -> %s (sha256 %s)", args.out, sha[:16])

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _ROOT / "results" / "pitchgpt_v3" / f"fit_2023_{args.tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(
        json.dumps(
            {
                **payload,
                "artifact_path": str(args.out),
                "artifact_sha256": sha,
                "backbone_meta": bb_ck["meta"],
                "elapsed_sec": round(time.perf_counter() - t0, 1),
                "duckdb_read_only": True,
                "tier": "fit-2023",
            },
            indent=2,
            default=str,
        )
    )
    logger.info("audit -> %s", out_dir / "audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
