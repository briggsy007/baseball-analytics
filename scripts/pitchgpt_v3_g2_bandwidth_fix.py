"""
PitchGPT v3 — G2 (§6.3) re-measurement under a single pinned kernel bandwidth.

**Why this run exists.**  The first dev-gate run computed the §6.2 frozen-v2
SKCE reference with ``bandwidth=None``, so the median heuristic was refitted on
frozen v2's own probabilities.  Two SKCE values computed under *different*
kernels are not comparable, which invalidated the published side-by-side (and
the "29x smaller SKCE" reading of the outcome head).  §6.3 also requires the
bandwidth to be "fixed before the contact and recorded"; nothing pinned it, so
the sealed-2026 grading run would have refitted the kernel on 2026 data.
See §9 entries 13 and 14.

**What this run changes and does not change.**  The v3 numbers are recomputed
identically (same rows, same seed, same bandwidth that v3's own probabilities
produce) and are asserted to reproduce the first run's values EXACTLY — the
gate cannot move, and the assertion proves it.  Only the frozen-v2 *reference*
numbers change, and those are explicitly "reported, never gated" (§6.2).  The
§6.8 verdict is untouched.

Scope: 2024 BURNED DEV TIER only (§5.3 — unlimited iteration, no budget, no
validation authority).  2025 is never read.  2026 is never read.  No rollout is
performed: G2 needs only teacher-forced per-pitch probabilities.
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

from src.analytics.pitchgpt import CONTEXT_DIM, PitchGPTModel  # noqa: E402
from src.analytics.pitchgpt_outcome_head import (  # noqa: E402
    FrozenOutcomeHeadConcat,
)
from src.analytics.pitchgpt_v3 import (  # noqa: E402
    SPEC_BODY_SHA256,
    SPEC_FREEZE_SHA,
    SPEC_PATH,
    V3OutcomeHead,
    load_calibration,
    load_v3_checkpoint,
)
from src.analytics.pitchgpt_v3_data import DEV_SEASON, build_pa_dataset  # noqa: E402
from src.analytics.pitchgpt_v3_data import load_sequence_cache  # noqa: E402
from src.analytics.pitchgpt_v3_gates import (  # noqa: E402
    KCE_ALPHA,
    KCE_BANDWIDTH_RULE,
    KCE_PINNED_BANDWIDTH_FILENAME,
    KCE_PROBE_LOGIT_SCALE,
    KCE_SEED,
    KCE_SUBSAMPLE,
    load_pinned_bandwidths,
    save_pinned_bandwidths,
    skce_test,
)
from src.analytics.pitchgpt_v3_infer import (  # noqa: E402
    HORIZON,
    collect_outcome_probs,
    collect_v2_marginal_head_probs,
    collect_v2_outcome_probs,
    collect_v3_head_probs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitchgpt_v3_g2_bandwidth_fix")

DEFAULT_CACHE_DIR = Path(
    r"C:\Users\hunte\AppData\Local\Temp\claude"
    r"\C--Users-hunte-projects-baseball"
    r"\5112f9f4-f7ff-4db3-bd24-90acc5fbef27\scratchpad\pitchgpt_v3_cache"
)
A1_TEMPERATURE = 0.8003

#: The first dev-gate run's G2 values, for the no-movement assertion.
PRIOR_RUN = "results/pitchgpt_v3/dev_2024_gates_20260811T062956Z/audit.json"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    args = ap.parse_args()

    t_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v2_path = _ROOT / "models" / "pitchgpt_v2.pt"
    a1_path = _ROOT / "models" / "pitchgpt_v2_outcomehead_a1.pt"
    frozen_sha_pre = {p.name: sha256_file(p) for p in (v2_path, a1_path)}

    model, bb_ck = load_v3_checkpoint(args.backbone, device=device)
    model.to(device).eval()
    head = V3OutcomeHead().to(device)
    head.load_state_dict(
        torch.load(args.outcome_head, map_location=device, weights_only=False)[
            "head_state_dict"
        ]
    )
    head.eval()
    temps, cal_payload = load_calibration(args.calibration)   # §7.5 guard

    v2_ck = torch.load(v2_path, map_location=device, weights_only=False)
    v2 = PitchGPTModel(
        vocab_size=v2_ck["config"]["vocab_size"],
        d_model=v2_ck["config"]["d_model"],
        nhead=v2_ck["config"]["nhead"],
        num_layers=v2_ck["config"]["num_layers"],
        max_seq_len=v2_ck["config"]["max_seq_len"],
        context_dim=v2_ck["config"].get("context_dim", CONTEXT_DIM),
    )
    v2.load_state_dict(v2_ck["model_state_dict"])
    v2.to(device).eval()
    a1_ck = torch.load(a1_path, map_location=device, weights_only=False)
    a1 = FrozenOutcomeHeadConcat()
    a1.load_state_dict(a1_ck["head_state_dict"])
    a1.to(device).eval()
    a1_temperature = float(a1_ck.get("temperature", A1_TEMPERATURE))

    pa = build_pa_dataset(
        load_sequence_cache(args.cache_dir / "dev24.npz"), horizon=HORIZON,
    )
    n_pa = len(pa["pa_len"])
    logger.info("2024 BURNED-DEV cohort: %d PAs", n_pa)

    v3p = collect_v3_head_probs(model, pa, device, temps=temps)
    v2p = collect_v2_marginal_head_probs(v2, pa, device)
    v3o = collect_outcome_probs(model, head, pa, device, temperature=temps.outcome_T)
    v2o = collect_v2_outcome_probs(v2, a1, pa, device, temperature=a1_temperature)

    heads = {
        "type": (v3p["p_type"], v3p["y_type"], v2p["p_type"]),
        "zone": (v3p["p_zone"], v3p["y_zone"], v2p["p_zone"]),
        "velo": (v3p["p_velo"], v3p["y_velo"], v2p["p_velo"]),
        "outcome": (v3o["probs"], v3o["labels"], v2o["probs"]),
    }
    labels_v2_outcome = v2o["labels"]

    prior = json.loads((_ROOT / PRIOR_RUN).read_text())["gates"]["G2"]["per_head"]

    out: dict = {}
    for name, (p3, y3, p2) in heads.items():
        y2 = labels_v2_outcome if name == "outcome" else y3
        logger.info("G2 %s: fitting the pinned bandwidth on v3 dev probs ...", name)
        base = skce_test(p3, y3)
        bw = base["bandwidth"]

        # Byte-identical to ``pitchgpt_v3_dev_gates.head_g2``'s probe path, so
        # the probe reproduces the first run rather than merely agreeing to
        # three digits.
        logit = np.log(np.clip(p3, 1e-30, None)) * KCE_PROBE_LOGIT_SCALE
        logit -= logit.max(axis=1, keepdims=True)
        e = np.exp(logit)
        probe = skce_test(e / e.sum(axis=1, keepdims=True), y3, bandwidth=bw)

        # THE FIX: the frozen-v2 reference now shares v3's kernel.
        v2_same = skce_test(p2, y2, bandwidth=bw)
        # Retained for the record: what the old, invalid call produced.
        v2_own = skce_test(p2, y2)

        # The gate cannot have moved: v3's own numbers must reproduce exactly.
        pr = prior[name]["observed"]
        assert abs(base["skce"] - pr["skce"]) < 1e-12, (name, base["skce"], pr["skce"])
        assert abs(base["p_value"] - pr["p_value"]) < 1e-12, name
        assert abs(bw - pr["bandwidth"]) < 1e-12, name

        # Reported, not asserted: how exactly the two *other* prior-run
        # quantities re-derive.  The probe shares v3's kernel in both runs, so
        # it should reproduce; the own-kernel v2 number is the superseded one
        # and is re-derived here purely so the audit can carry it as measured
        # rather than as a quote from another file.
        repro = {}
        for key, now, was in (
            ("power_probe", probe, prior[name]["power_probe"]),
            (
                "frozen_v2_reference_own_kernel",
                v2_own,
                prior[name]["frozen_v2_reference"],
            ),
        ):
            repro[key] = {
                "prior_skce": was["skce"],
                "rerun_skce": now["skce"],
                "abs_delta_skce": abs(now["skce"] - was["skce"]),
                "prior_p_value": was["p_value"],
                "rerun_p_value": now["p_value"],
                "prior_bandwidth": was["bandwidth"],
                "rerun_bandwidth": now["bandwidth"],
                "bit_for_bit": bool(
                    abs(now["skce"] - was["skce"]) < 1e-12
                    and abs(now["p_value"] - was["p_value"]) < 1e-12
                    and abs(now["bandwidth"] - was["bandwidth"]) < 1e-12
                ),
            }

        out[name] = {
            "v3_observed": base,
            "power_probe": probe,
            "probe_detects_planted_error": bool(probe["p_value"] < KCE_ALPHA),
            "frozen_v2_reference_SAME_kernel": v2_same,
            "frozen_v2_reference_OWN_kernel_SUPERSEDED": v2_own,
            "v3_reproduces_prior_run_exactly": True,
            "reproduction_vs_prior_run": repro,
            "outcome_ratio_v2_over_v3_same_kernel": (
                v2_same["skce"] / base["skce"] if base["skce"] else None
            ),
            "outcome_ratio_v2_over_v3_own_kernel_SUPERSEDED": (
                v2_own["skce"] / base["skce"] if base["skce"] else None
            ),
        }
        logger.info(
            "G2 %-8s bw=%.6f | v3 SKCE %.4e p=%.4f | v2 SAME-kernel %.4e p=%.4f "
            "| v2 own-kernel (invalid) %.4e p=%.4f bw=%.6f | probe p=%.4f",
            name, bw, base["skce"], base["p_value"],
            v2_same["skce"], v2_same["p_value"],
            v2_own["skce"], v2_own["p_value"], v2_own["bandwidth"],
            probe["p_value"],
        )

    # ── pin the bandwidths (§6.3 "fixed before the contact and recorded") ──
    pinned = {k: v["v3_observed"]["bandwidth"] for k, v in out.items()}
    bw_path = _ROOT / "models" / KCE_PINNED_BANDWIDTH_FILENAME
    pin_pre_existing = bw_path.exists()
    if bw_path.exists():
        existing, _ = load_pinned_bandwidths(bw_path)
        drift = {
            k: (existing.get(k), pinned[k]) for k in pinned
            if k not in existing or abs(existing[k] - pinned[k]) > 1e-9
        }
        assert not drift, f"pinned bandwidths drifted: {drift}"
        logger.info("pinned bandwidths already recorded and reproduce exactly")
    else:
        save_pinned_bandwidths(
            bw_path, pinned,
            {
                "fit_cohort_season": DEV_SEASON,
                "fit_cohort": "2024 pitcher-disjoint BURNED dev cohort",
                "fit_seed": KCE_SEED,
                "fit_n_used_per_head": {
                    k: v["v3_observed"]["n_used"] for k, v in out.items()
                },
                "fit_n_available_per_head": {
                    k: v["v3_observed"]["n_available"] for k, v in out.items()
                },
                "fitted_on_model": "v3-factorized (the model under test)",
                "produced_by": "scripts/pitchgpt_v3_g2_bandwidth_fix.py",
                "spec_freeze_sha": SPEC_FREEZE_SHA,
            },
        )

    frozen_sha_post = {p.name: sha256_file(p) for p in (v2_path, a1_path)}
    assert frozen_sha_pre == frozen_sha_post, "a frozen artifact was mutated!"

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _ROOT / "results" / "pitchgpt_v3" / f"dev_2024_g2_bandwidth_fix_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tier": "burned-dev",
        "tier_warning": (
            "2024 is the BURNED dev tier (§5.3). NO validation authority. The "
            "one graded contact is against sealed 2026 (§5.5) and was NOT made."
        ),
        "purpose": (
            "Re-measure G2 (§6.3) with ONE pinned kernel bandwidth per head so "
            "the §6.2 frozen-v2 side-by-side is like-for-like, and record the "
            "bandwidths so the graded run replays rather than refits them. "
            "§9 entries 13 and 14."
        ),
        "lockbox_2026_contacted": False,
        "holdout_2025_contacted": False,
        "spec_path": SPEC_PATH,
        "spec_freeze_sha": SPEC_FREEZE_SHA,
        "spec_body_sha256": SPEC_BODY_SHA256,
        "git_sha": git_sha(),
        "cohort": pa["meta"],
        "n_pa": n_pa,
        "prior_run": PRIOR_RUN,
        "gate_verdict_unchanged": True,
        "gate_verdict_unchanged_reason": (
            "Every v3 SKCE/p-value reproduces the prior run bit-for-bit "
            "(asserted in-run). Only the frozen-v2 REFERENCE moved, and §6.2 "
            "reports it without gating on it. G2 remains FAIL, not VOID."
        ),
        "subsample": {
            "deviations_log_entry": 13,
            "cap": KCE_SUBSAMPLE,
            "seed": KCE_SEED,
            "note": (
                "§6.3 authorizes no subsample; the O(n^2) U-statistic forces "
                "one. Publish n_used with every SKCE and p-value."
            ),
        },
        "bandwidth_pinning": {
            "deviations_log_entry": 14,
            "superseding_deviations_log_entry": 16,
            "artifact": f"models/{KCE_PINNED_BANDWIDTH_FILENAME}",
            "artifact_sha256": sha256_file(bw_path),
            "artifact_pre_existed": pin_pre_existing,
            "rule": KCE_BANDWIDTH_RULE,
            "pinned": pinned,
            "pinned_provenance": json.loads(bw_path.read_text()),
        },
        "temperatures": temps.to_dict(),
        "calibration_provenance": {
            k: cal_payload[k]
            for k in ("fit_cohort_season", "fit_seed", "fit_n_pas", "produced_by")
        },
        "backbone_stage": bb_ck["meta"]["stage"],
        "backbone_sha256": sha256_file(args.backbone),
        "outcome_head_sha256": sha256_file(args.outcome_head),
        "frozen_artifact_sha256_pre": frozen_sha_pre,
        "frozen_artifact_sha256_post": frozen_sha_post,
        "a1_temperature": a1_temperature,
        "per_head": out,
        "elapsed_sec": round(time.perf_counter() - t_start, 1),
        "duckdb_read_only": True,
        "duckdb_opened": False,
    }
    (out_dir / "audit.json").write_text(json.dumps(payload, indent=2, default=str))
    logger.info("audit -> %s", out_dir / "audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
