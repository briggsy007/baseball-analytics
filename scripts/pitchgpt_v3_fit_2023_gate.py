"""
PitchGPT v3 — K-v2-FIT-B: does exposure bias actually close on the fit cohort?

Spec ``PITCHGPT_V2_SPEC.md`` §7.2:

    After Stage B3, on the 2023 fit cohort (§5.2), measure the same quantity
    Phase 0.6.2 died on: max over within-PA positions 0-5 and outcome classes
    of |rollout marginal - empirical marginal|.

    KILL if that maximum is > 1.0pp.  Reference points: 16.37pp for the v2-era
    stack under raw T, 2.625pp after two rounds of the (now-banned) output
    reweighting.  No post-hoc reweighting layer may be added to rescue this
    number (§0.2).

§4.3.3: the measurement is a kill gate, not a tuning signal — it is looked at
exactly once per curriculum run.  Exit codes: ``0`` = no kill, ``2`` = KILL.

The gated configuration is the SHIPPED one: the §4.4 per-head temperatures
applied.  The raw ``T = 1.0`` number is computed in the same pass and reported
as the direct analogue of Phase 0.6.2's roll-0 reference (16.37pp), so the two
programs are comparable at the same operating point.

2023 is a FIT cohort, never a gate cohort (§5.2).  2025/2026 are never read.
"""

from __future__ import annotations

import argparse
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

from src.analytics.pitchgpt_outcome_head import OUTCOME_CLASSES  # noqa: E402
from src.analytics.pitchgpt_v3 import (  # noqa: E402
    SPEC_BODY_SHA256,
    SPEC_FREEZE_SHA,
    SPEC_PATH,
    V3OutcomeHead,
    load_calibration,
    load_v3_checkpoint,
)
from src.analytics.pitchgpt_v3_data import (  # noqa: E402
    build_pa_dataset,
    load_sequence_cache,
)
from src.analytics.pitchgpt_v3_gates import (  # noqa: E402
    POSITION_GAP_MAX,
    position_kl_spearman,
    position_marginal_gap,
)
from src.analytics.pitchgpt_v3_infer import (  # noqa: E402
    HORIZON,
    empirical_pa_terminals,
    empirical_position_marginals,
    rollout_cohort,
    terminal_share,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitchgpt_v3_fit_2023_gate")

DEFAULT_CACHE_DIR = Path(
    r"C:\Users\hunte\AppData\Local\Temp\claude"
    r"\C--Users-hunte-projects-baseball"
    r"\5112f9f4-f7ff-4db3-bd24-90acc5fbef27\scratchpad\pitchgpt_v3_cache"
)
N_SAMPLES = 100


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _table(m: np.ndarray) -> list[dict]:
    return [
        {"position": int(p), **{OUTCOME_CLASSES[c]: float(m[p, c]) for c in range(m.shape[1])}}
        for p in range(m.shape[0])
    ]


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
        "--calibration", type=Path,
        default=_ROOT / "models" / "calibration_pitchgpt_v3.json",
    )
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--pa-batch", type=int, default=48)
    ap.add_argument("--tag", type=str, default="killB")
    ap.add_argument("--limit-pas", type=int, default=0, help="smoke only")
    args = ap.parse_args()

    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, bb_ck = load_v3_checkpoint(args.backbone, device=device)
    model.to(device).eval()
    head = V3OutcomeHead().to(device)
    head.load_state_dict(
        torch.load(args.outcome_head, map_location=device, weights_only=False)[
            "head_state_dict"
        ]
    )
    head.eval()
    # §7.5: refuses any sidecar declaring a gate-evaluation cohort.
    temps, cal_payload = load_calibration(args.calibration)
    logger.info("temperatures: %s", temps.to_dict())

    pa = build_pa_dataset(
        load_sequence_cache(args.cache_dir / "fit23.npz"), horizon=HORIZON,
    )
    if args.limit_pas:
        n = int(args.limit_pas)
        pa = {
            **{k: v[:n] for k, v in pa.items() if k != "meta"},
            "meta": {**pa["meta"], "SMOKE_limited_to_pas": n},
        }
    logger.info("2023 fit cohort: %d PAs (full cohort, no subsample)", len(pa["pa_len"]))

    emp_marg, emp_counts = empirical_position_marginals(pa, HORIZON)
    emp_term = empirical_pa_terminals(pa, HORIZON)

    logger.info("rolling with the SHIPPED per-head temperatures ...")
    shipped = rollout_cohort(
        model, head, pa, device, temps=temps, n_samples=args.n_samples,
        horizon=HORIZON, pa_batch=args.pa_batch, progress="shipped-T",
    )
    logger.info("rolling with raw T=1.0 (0.6.2 roll-0 analogue) ...")
    raw = rollout_cohort(
        model, head, pa, device, temps=None, n_samples=args.n_samples,
        horizon=HORIZON, pa_batch=args.pa_batch, progress="raw-T",
    )

    gap_shipped = position_marginal_gap(shipped["position_marginals"], emp_marg)
    gap_raw = position_marginal_gap(raw["position_marginals"], emp_marg)
    kl_shipped = position_kl_spearman(shipped["position_marginals"], emp_marg)

    kill = not gap_shipped["pass"]
    logger.info(
        "K-v2-FIT-B: max |rollout - empirical| = %.4fpp (shipped T) / %.4fpp (raw T) "
        "vs threshold %.2fpp -> %s",
        100 * gap_shipped["max_abs_gap"], 100 * gap_raw["max_abs_gap"],
        100 * POSITION_GAP_MAX, "KILL" if kill else "no kill",
    )
    logger.info(
        "worst cell: position %d, class %s",
        gap_shipped["max_at_position"], OUTCOME_CLASSES[gap_shipped["max_at_class"]],
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _ROOT / "results" / "pitchgpt_v3" / f"fit_2023_{args.tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate": "K-v2-FIT-B",
        "tier": "fit-2023",
        "spec_path": SPEC_PATH,
        "spec_freeze_sha": SPEC_FREEZE_SHA,
        "spec_body_sha256": SPEC_BODY_SHA256,
        "git_sha": git_sha(),
        "cohort": {**pa["meta"], "subsample": "none (full 2023 pitcher-disjoint cohort)"},
        "n_samples_per_pa": args.n_samples,
        "horizon": HORIZON,
        "seed_convention": "42 + pa_index * 1000 (§4.3.2), per-PA uniform block",
        "temperatures": temps.to_dict(),
        "calibration_provenance": {
            k: cal_payload[k]
            for k in ("fit_cohort_season", "fit_seed", "fit_n_pas", "produced_by")
        },
        "backbone_checkpoint": str(args.backbone),
        "backbone_stage": bb_ck["meta"]["stage"],
        "threshold_pp": 100 * POSITION_GAP_MAX,
        "kill": bool(kill),
        "shipped_T": {
            "max_abs_gap_pp": 100 * gap_shipped["max_abs_gap"],
            "max_at_position": gap_shipped["max_at_position"],
            "max_at_class": OUTCOME_CLASSES[gap_shipped["max_at_class"]],
            "per_position_max_pp": [100 * x for x in gap_shipped["per_position_max"]],
            "pass": gap_shipped["pass"],
            "rollout_marginals": _table(shipped["position_marginals"]),
            "position_counts": shipped["position_counts"].tolist(),
            "mask_events": shipped["mask_events"],
            "pa_terminal_share": terminal_share(shipped["pa_terminal"]),
            "mean_pa_length": float(np.mean(shipped["pa_length"])),
            "elapsed_sec": shipped["elapsed_sec"],
        },
        "raw_T": {
            "max_abs_gap_pp": 100 * gap_raw["max_abs_gap"],
            "max_at_position": gap_raw["max_at_position"],
            "max_at_class": OUTCOME_CLASSES[gap_raw["max_at_class"]],
            "per_position_max_pp": [100 * x for x in gap_raw["per_position_max"]],
            "rollout_marginals": _table(raw["position_marginals"]),
            "pa_terminal_share": terminal_share(raw["pa_terminal"]),
            "elapsed_sec": raw["elapsed_sec"],
        },
        "empirical_marginals": _table(emp_marg),
        "empirical_counts": emp_counts.tolist(),
        "empirical_pa_terminal_share": terminal_share(emp_term),
        "empirical_mean_pa_length_capped": float(
            np.mean(np.minimum(pa["pa_len"], HORIZON))
        ),
        "secondary_position_kl": kl_shipped,
        "v2_era_reference": {
            "raw_T_pp": 16.37,
            "after_two_reweighting_rounds_pp": 2.625,
            "source": "docs/models/pitchgpt_phase062_results.md §1",
        },
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "duckdb_read_only": True,
        "looked_at_once": True,
    }
    (out_dir / "audit.json").write_text(json.dumps(payload, indent=2, default=str))
    logger.info("audit -> %s", out_dir / "audit.json")
    return 2 if kill else 0


if __name__ == "__main__":
    raise SystemExit(main())
