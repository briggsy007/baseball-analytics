"""
PitchGPT v3 — the §6 gate suite, run on the 2024 BURNED DEV TIER.

**THESE ARE DEV NUMBERS, NOT THE LOCKBOX VALIDATION.**  §5.3: "2024 carries no
validation authority ... No 2024 number may be published as a validation
result — dev numbers appear in the results doc explicitly labelled
``tier=burned-dev``."  The one graded contact is against sealed 2026 at season
end (§5.5), and this program does not make it.

What runs here (spec §6):

* **G1** classwise-ECE + TACE per factor head (15 equal-mass bins; TACE
  threshold 1e-3), for v3 AND for the FROZEN v2 stack marginalised onto the
  same three fields (§6.2).
* **G2** SKCE hypothesis test per head with a 1,000-resample bootstrap, plus
  the **mandatory power probe** (logits x1.10; if the test cannot reject that
  planted distortion at p<0.01 the gate is recorded VOID, not PASS).
* **G3** randomized PIT of PA-terminal wOBA (KS <= 0.03, 20 bins in
  [0.0335, 0.0750]) + the Phase-0.6 marginal tolerances.
* **G4** per-count-state top-1 ECE of the outcome head (<= 0.02 where n >= 500)
  + the position gate (max |rollout - empirical| <= 1.0pp) — the quantity 0.6.2
  died on.
* **G5** decision calibration against K / BB / HR deciles.
* **§6.7** anti-unfailability: every G1 threshold and the G4 per-count-state
  threshold that the FROZEN v2 stack already satisfies on this dev cohort is
  tightened to 0.6 x the frozen-v2 value (floor 0.005) and v3 is graded on the
  tightened value.  Executed before any thresholds are applied.

Marginal-rate comparators (both are computed and BOTH must pass, so neither
framing can quietly rescue a gate):

* **A — like-for-like**: empirical PAs replayed through the production
  ``_advance_count`` state machine and truncated at horizon 6, exactly as the
  rollout is.  Same estimator on both sides.
* **B — spec-literal**: the PHASE_0.6 §3.3 SQL-shape empirical baselines
  (uncapped, terminal-event driven).  The rollout's horizon-6 truncation is a
  known bias against this comparator and is reported, not hidden.

2025 is never read.  2026 is never read.
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
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt import CONTEXT_DIM, PitchGPTModel  # noqa: E402
from src.analytics.pitchgpt_outcome_head import (  # noqa: E402
    NUM_OUTCOME_CLASSES,
    OUTCOME_CLASSES,
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
from src.analytics.pitchgpt_v3_data import (  # noqa: E402
    DEV_SEASON,
    FIT_SEASON,
    TRAIN_SEASONS,
    build_pa_dataset,
    fetch_pa_terminal_facts,
    fetch_train_pitcher_ids,
    load_sequence_cache,
)
from src.analytics.pitchgpt_v3_gates import (  # noqa: E402
    CAL_VALID_COVERAGE_MIN,
    COUNT_STATE_ECE_MAX,
    COUNT_STATE_MIN_OBS,
    DECISION_DECILE_GAP_MAX,
    DECISION_WEIGHTED_GAP_MAX,
    G1_THRESHOLDS,
    KCE_ALPHA,
    KCE_PINNED_BANDWIDTH_FILENAME,
    KCE_PROBE_LOGIT_SCALE,
    KCE_SEED,
    KCE_SUBSAMPLE,
    PA_LEN_ABS_TOL,
    PIT_BIN_HI,
    PIT_BIN_LO,
    PIT_KS_MAX,
    POSITION_GAP_MAX,
    WOBA_ABS_TOL,
    GateResult,
    classwise_ece,
    decision_deciles,
    gate_abs,
    gate_pct,
    ks_uniform,
    load_pinned_bandwidths,
    pit_bin_masses,
    position_kl_spearman,
    position_marginal_gap,
    randomized_pit,
    save_pinned_bandwidths,
    skce_test,
    tace,
    tighten_threshold,
    top1_ece,
)
from src.analytics.pitchgpt_v3_infer import (  # noqa: E402
    HORIZON,
    PA_BB,
    PA_HIT,
    PA_K,
    PA_TERMINAL_NAMES,
    _PA_WOBA,
    collect_outcome_probs,
    collect_v2_marginal_head_probs,
    collect_v2_outcome_probs,
    collect_v3_head_probs,
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
logger = logging.getLogger("pitchgpt_v3_dev_gates")

DEFAULT_CACHE_DIR = Path(
    r"C:\Users\hunte\AppData\Local\Temp\claude"
    r"\C--Users-hunte-projects-baseball"
    r"\5112f9f4-f7ff-4db3-bd24-90acc5fbef27\scratchpad\pitchgpt_v3_cache"
)
#: The A1 head's shipped temperature (docs/models/pitchgpt_validation_spec.md;
#: claim ``pitchgpt_per_pitch_ece``: post-T ECE 0.0114 at T = 0.8003).
A1_TEMPERATURE = 0.8003
HIT_EVENTS = frozenset({"single", "double", "triple", "home_run"})
HR_EVENTS = frozenset({"home_run"})
K_EVENTS = frozenset({"strikeout", "strikeout_double_play"})
BB_EVENTS = frozenset({"walk", "intent_walk"})
HBP_EVENTS = frozenset({"hit_by_pitch"})


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


# ── G1 / G2 per-head machinery ───────────────────────────────────────────────


def head_g1(probs: np.ndarray, labels: np.ndarray) -> dict:
    return {"cwECE": classwise_ece(probs, labels), "TACE": tace(probs, labels)}


def head_g2(probs: np.ndarray, labels: np.ndarray, name: str) -> dict:
    logger.info("  G2 SKCE: %s (n=%d, C=%d) ...", name, len(labels), probs.shape[1])
    # THE ONE call permitted to fit the §6.3 median heuristic: it establishes
    # the pin, on the DEV cohort, on the model under test (§9 entry 14).
    # ``bandwidth=None`` is written explicitly — every other call site must
    # name a bandwidth, and a test enforces that no call relies on the default.
    base = skce_test(probs, labels, bandwidth=None)
    # §6.3 power probe: distort the logits by x1.10 and re-test on the same rows.
    logit = np.log(np.clip(probs, 1e-30, None)) * KCE_PROBE_LOGIT_SCALE
    logit -= logit.max(axis=1, keepdims=True)
    e = np.exp(logit)
    probe = skce_test(e / e.sum(axis=1, keepdims=True), labels, bandwidth=base["bandwidth"])
    return {
        "observed": base,
        "power_probe": probe,
        "probe_detects_planted_error": bool(probe["p_value"] < KCE_ALPHA),
    }


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
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--pa-batch", type=int, default=48)
    ap.add_argument("--tag", type=str, default="gates")
    ap.add_argument("--limit-pas", type=int, default=0, help="smoke only")
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
    logger.info("v3 temperatures: %s", temps.to_dict())

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
    # The A1 head's own shipped temperature, read from its frozen checkpoint.
    a1_temperature = float(a1_ck.get("temperature", A1_TEMPERATURE))
    logger.info("frozen A1 outcome-head temperature = %.6f", a1_temperature)

    pa = build_pa_dataset(
        load_sequence_cache(args.cache_dir / "dev24.npz"), horizon=HORIZON,
    )
    if args.limit_pas:
        n = int(args.limit_pas)
        pa = {
            **{k: v[:n] for k, v in pa.items() if k != "meta"},
            "meta": {**pa["meta"], "SMOKE_limited_to_pas": n},
        }
    n_pa = len(pa["pa_len"])
    logger.info("2024 BURNED-DEV cohort: %d PAs (full pitcher-disjoint season)", n_pa)

    # ── per-pitch head probabilities (teacher-forced, PA-scoped) ─────────
    logger.info("collecting v3 head probabilities ...")
    v3p = collect_v3_head_probs(model, pa, device, temps=temps)
    logger.info("collecting frozen-v2 marginalised head probabilities (§6.2) ...")
    v2p = collect_v2_marginal_head_probs(v2, pa, device)
    logger.info("collecting outcome probabilities ...")
    v3o = collect_outcome_probs(model, head, pa, device, temperature=temps.outcome_T)
    v2o = collect_v2_outcome_probs(v2, a1, pa, device, temperature=a1_temperature)

    heads = {
        "type": (v3p["p_type"], v3p["y_type"], v2p["p_type"]),
        "zone": (v3p["p_zone"], v3p["y_zone"], v2p["p_zone"]),
        "velo": (v3p["p_velo"], v3p["y_velo"], v2p["p_velo"]),
        "outcome": (v3o["probs"], v3o["labels"], v2o["probs"]),
    }
    labels_v2_outcome = v2o["labels"]

    # ── G1 + §6.7 tightening ─────────────────────────────────────────────
    g1: dict = {}
    for name, (p3, y3, p2) in heads.items():
        y2 = labels_v2_outcome if name == "outcome" else y3
        r3 = head_g1(p3, y3)
        r2 = head_g1(p2, y2)
        spec_cw = G1_THRESHOLDS[name]["cwECE"]
        spec_ta = G1_THRESHOLDS[name]["TACE"]
        t_cw = tighten_threshold(spec_cw, r2["cwECE"]["classwise_ece"])
        t_ta = tighten_threshold(spec_ta, r2["TACE"]["tace"])
        g1[name] = {
            "C": G1_THRESHOLDS[name]["C"],
            "n": int(len(y3)),
            "v3_cwECE": r3["cwECE"]["classwise_ece"],
            "v3_TACE": r3["TACE"]["tace"],
            "frozen_v2_cwECE": r2["cwECE"]["classwise_ece"],
            "frozen_v2_TACE": r2["TACE"]["tace"],
            "cwECE_threshold": t_cw,
            "TACE_threshold": t_ta,
            "cwECE_pass": bool(r3["cwECE"]["classwise_ece"] <= t_cw["effective_threshold"]),
            "TACE_pass": bool(r3["TACE"]["tace"] <= t_ta["effective_threshold"]),
            "v3_per_class_cwECE": r3["cwECE"]["per_class"],
            "class_counts": r3["cwECE"]["class_counts"],
            "excluded_classes_lt_100_obs": r3["cwECE"]["excluded_classes"],
        }
        logger.info(
            "G1 %-8s cwECE v3 %.5f vs v2 %.5f  -> thr %.5f (%s)  %s | "
            "TACE v3 %.5f vs v2 %.5f -> thr %.5f  %s",
            name, g1[name]["v3_cwECE"], g1[name]["frozen_v2_cwECE"],
            t_cw["effective_threshold"], "tightened" if t_cw["tightened"] else "as-written",
            "PASS" if g1[name]["cwECE_pass"] else "FAIL",
            g1[name]["v3_TACE"], g1[name]["frozen_v2_TACE"],
            t_ta["effective_threshold"],
            "PASS" if g1[name]["TACE_pass"] else "FAIL",
        )
    G1 = GateResult(
        "G1",
        all(v["cwECE_pass"] and v["TACE_pass"] for v in g1.values()),
        {"per_head": g1},
    )

    # ── G2 + power probe ─────────────────────────────────────────────────
    g2: dict = {}
    for name, (p3, y3, p2) in heads.items():
        y2 = labels_v2_outcome if name == "outcome" else y3
        g2[name] = head_g2(p3, y3, name)
        # Reported, never gated: the same test on the FROZEN v2 stack, so a
        # rejection can be read against the incumbent rather than in a vacuum.
        # It MUST reuse v3's bandwidth (§9 entry 14): two SKCE values computed
        # under different kernels are not comparable, and letting this call
        # refit the median heuristic on v2's probabilities is what made the
        # earlier side-by-side invalid.
        g2[name]["frozen_v2_reference"] = skce_test(
            p2, y2, bandwidth=g2[name]["observed"]["bandwidth"],
        )
        logger.info(
            "G2 %-8s SKCE %.3e  p=%.4f  | v2 SKCE %.3e p=%.4f | probe p=%.4f "
            "(detects planted error: %s)",
            name, g2[name]["observed"]["skce"], g2[name]["observed"]["p_value"],
            g2[name]["frozen_v2_reference"]["skce"],
            g2[name]["frozen_v2_reference"]["p_value"],
            g2[name]["power_probe"]["p_value"],
            g2[name]["probe_detects_planted_error"],
        )
    # §6.3 "fixed before the contact and recorded" — pin the dev-fitted
    # bandwidths so the graded run replays them instead of refitting the
    # kernel on the cohort it is grading (§9 entry 14).
    pinned_bw = {k: v["observed"]["bandwidth"] for k, v in g2.items()}
    bw_path = _ROOT / "models" / KCE_PINNED_BANDWIDTH_FILENAME
    if bw_path.exists():
        existing, _ = load_pinned_bandwidths(bw_path)
        drift = {
            k: (existing.get(k), pinned_bw[k])
            for k in pinned_bw
            if k not in existing or abs(existing[k] - pinned_bw[k]) > 1e-9
        }
        assert not drift, f"pinned §6.3 bandwidths drifted: {drift}"
        logger.info("pinned §6.3 bandwidths reproduce exactly: %s", bw_path)
    else:
        save_pinned_bandwidths(
            bw_path, pinned_bw,
            {
                "fit_cohort_season": DEV_SEASON,
                "fit_cohort": "2024 pitcher-disjoint BURNED dev cohort",
                "fit_seed": KCE_SEED,
                "fit_n_used_per_head": {
                    k: v["observed"]["n_used"] for k, v in g2.items()
                },
                "fit_n_available_per_head": {
                    k: v["observed"]["n_available"] for k, v in g2.items()
                },
                "fitted_on_model": "v3-factorized (the model under test)",
                "produced_by": "scripts/pitchgpt_v3_dev_gates.py",
                "spec_freeze_sha": SPEC_FREEZE_SHA,
            },
        )

    probe_ok = all(v["probe_detects_planted_error"] for v in g2.values())
    g2_rejects = any(v["observed"]["p_value"] < KCE_ALPHA for v in g2.values())
    G2 = GateResult(
        "G2",
        passed=(not g2_rejects),
        detail={
            "per_head": g2,
            "alpha": KCE_ALPHA,
            "protocol_deviations": {
                "subsample": {
                    "deviations_log_entry": 13,
                    "n_used_per_head": KCE_SUBSAMPLE,
                    "seed": KCE_SEED,
                    "note": (
                        "§6.3 authorizes no subsample. The unbiased SKCE is an "
                        "O(n^2) U-statistic and each of the 1,000 bootstrap "
                        "replicates is O(n^2) again, so every SKCE and p-value "
                        "here is computed on a fixed seeded draw of "
                        f"{KCE_SUBSAMPLE} rows, NOT the full cohort. Publish "
                        "n_used with every number."
                    ),
                },
                "bandwidth_pinning": {
                    "deviations_log_entry": 14,
                    "artifact": KCE_PINNED_BANDWIDTH_FILENAME,
                    "note": (
                        "§6.3 fixes the bandwidth on the dev cohort before the "
                        "contact. It is fitted once on v3's dev probabilities, "
                        "pinned to the artifact above, and reused for the power "
                        "probe and the §6.2 frozen-v2 reference so all three "
                        "are computed under the SAME kernel."
                    ),
                },
            },
            "pinned_bandwidths": pinned_bw,
            "power_probe_all_heads_detect": probe_ok,
            "void_reason": (
                None if probe_ok
                else "§6.3: the SKCE test failed to reject a planted x1.10 logit "
                     "distortion on at least one head, so it is uninformative"
            ),
        },
        void=not probe_ok,
    )

    # ── rollout the dev cohort ───────────────────────────────────────────
    logger.info("rolling the 2024 dev cohort (%d PAs x %d samples) ...", n_pa, args.n_samples)
    roll = rollout_cohort(
        model, head, pa, device, temps=temps, n_samples=args.n_samples,
        horizon=HORIZON, pa_batch=args.pa_batch, progress="dev24",
    )
    emp_marg, emp_counts = empirical_position_marginals(pa, HORIZON)
    emp_term = empirical_pa_terminals(pa, HORIZON)

    # ── database-side (spec-literal) empirical baselines ─────────────────
    from src.db.schema import get_connection

    conn = get_connection(read_only=True)
    excl = fetch_train_pitcher_ids(conn, TRAIN_SEASONS)
    dev_facts = fetch_pa_terminal_facts(conn, DEV_SEASON, excl)
    fit_facts = fetch_pa_terminal_facts(conn, FIT_SEASON, excl)
    conn.close()

    def _flags(df):
        ev = df["terminal_event"].astype(str).str.lower()
        return {
            "K": ev.isin(K_EVENTS).to_numpy(),
            "BB": ev.isin(BB_EVENTS).to_numpy(),
            "HR": ev.isin(HR_EVENTS).to_numpy(),
            "HBP": ev.isin(HBP_EVENTS).to_numpy(),
            "hit": ev.isin(HIT_EVENTS).to_numpy(),
        }

    dev_f = _flags(dev_facts)
    fit_f = _flags(fit_facts)
    hr_given_hit_2023 = float(fit_f["HR"].sum() / max(fit_f["hit"].sum(), 1))
    logger.info(
        "HR|hit on the 2023 FIT cohort = %.4f (used to decompose the rollout's "
        "single in-play-hit channel; never fit on a gate cohort)",
        hr_given_hit_2023,
    )
    empirical_B = {
        "K": float(dev_f["K"].mean()),
        "BB": float(dev_f["BB"].mean()),
        "HR": float(dev_f["HR"].mean()),
        "HBP": float(dev_f["HBP"].mean()),
        "hit": float(dev_f["hit"].mean()),
        "mean_woba_db": float(np.nanmean(dev_facts["terminal_woba"].astype(float))),
        "mean_pa_length": float(dev_facts["pa_pitch_count"].mean()),
        "n_pa": int(len(dev_facts)),
    }

    # ── join the cache PAs to the DB PA facts (for G5 actuals) ───────────
    import pandas as pd

    key = pd.DataFrame(
        {"game_pk": pa["pa_game_pk"], "at_bat_number": pa["pa_abn"],
         "pitcher_id": pa["pa_pitcher_id"], "row": np.arange(n_pa)}
    )
    dev_small = dev_facts[
        ["game_pk", "at_bat_number", "pitcher_id", "terminal_event"]
    ].copy()
    for c in ("game_pk", "at_bat_number", "pitcher_id"):
        dev_small[c] = pd.to_numeric(dev_small[c], errors="coerce").astype("Int64")
        key[c] = key[c].astype("Int64")
    joined = key.merge(
        dev_small.drop_duplicates(["game_pk", "at_bat_number", "pitcher_id"]),
        on=["game_pk", "at_bat_number", "pitcher_id"], how="left",
    ).sort_values("row")
    ev = joined["terminal_event"].astype(str).str.lower()
    actual = {
        "K": ev.isin(K_EVENTS).to_numpy().astype(float),
        "BB": ev.isin(BB_EVENTS).to_numpy().astype(float),
        "HR": ev.isin(HR_EVENTS).to_numpy().astype(float),
    }
    matched = joined["terminal_event"].notna().to_numpy()
    logger.info(
        "PA join to DB terminal facts: %d/%d matched (%.2f%%)",
        int(matched.sum()), n_pa, 100 * matched.mean(),
    )
    for k in actual:
        actual[k] = np.where(matched, actual[k], np.nan)

    # ── model per-PA decision probabilities ──────────────────────────────
    term = roll["pa_terminal"]
    p_k = (term == PA_K).mean(axis=1)
    p_bb = (term == PA_BB).mean(axis=1)
    p_hit = (term == PA_HIT).mean(axis=1)
    p_hr = p_hit * hr_given_hit_2023

    model_A = {
        "K": float(p_k.mean()),
        "BB": float(p_bb.mean()),
        "HR": float(p_hr.mean()),
        "HBP": float((term == 2).mean()),
        "hit": float(p_hit.mean()),
    }
    emp_A = {
        "K": float((emp_term == PA_K).mean()),
        "BB": float((emp_term == PA_BB).mean()),
        "HR": float((emp_term == PA_HIT).mean() * hr_given_hit_2023),
        "HBP": float((emp_term == 2).mean()),
        "hit": float((emp_term == PA_HIT).mean()),
    }
    model_woba = float(np.nanmean(roll["pa_woba_samples"]))
    emp_woba = float(np.nanmean(_PA_WOBA[emp_term]))
    model_palen = float(np.mean(roll["pa_length"]))
    emp_palen_capped = float(np.mean(np.minimum(pa["pa_len"], HORIZON)))

    # ── §6.4 calibration_valid coverage (v3 analogue) ────────────────────
    fit_pa = build_pa_dataset(
        load_sequence_cache(args.cache_dir / "fit23.npz"), horizon=HORIZON,
    )
    fit_ctx = fit_pa["pa_ctx_idx"][:, 0, :]
    seen = [set(np.unique(fit_ctx[:, j]).tolist()) for j in range(fit_ctx.shape[1])]
    ump_lo, ump_hi = float(fit_pa["pa_ump"].min()), float(fit_pa["pa_ump"].max())
    dev_ctx = pa["pa_ctx_idx"][:, 0, :]
    in_support = np.ones(n_pa, dtype=bool)
    for j in range(dev_ctx.shape[1]):
        in_support &= np.isin(dev_ctx[:, j], list(seen[j]))
    in_support &= (pa["pa_ump"] >= ump_lo) & (pa["pa_ump"] <= ump_hi)
    cal_valid_cov = float(in_support.mean())

    marg_rows = []
    for q in ("K", "BB", "HR", "HBP", "hit"):
        a = gate_pct(model_A[q], emp_A[q])
        b = gate_pct(model_A[q], empirical_B[q])
        marg_rows.append({
            "quantity": q + "%",
            "model": model_A[q],
            "comparator_A_like_for_like": a,
            "comparator_B_spec_literal_sql": b,
            "pass": bool(a["pass"] and b["pass"]),
        })
    woba_row = gate_abs(model_woba, emp_woba, WOBA_ABS_TOL)
    woba_row_db = gate_abs(model_woba, empirical_B["mean_woba_db"], WOBA_ABS_TOL)
    palen_row = gate_abs(model_palen, emp_palen_capped, PA_LEN_ABS_TOL)
    palen_row_db = gate_abs(model_palen, empirical_B["mean_pa_length"], PA_LEN_ABS_TOL)

    # ── G3 PIT ───────────────────────────────────────────────────────────
    observed_woba = _PA_WOBA[emp_term]
    u = randomized_pit(roll["pa_woba_samples"], observed_woba)
    ks = ks_uniform(u)
    masses = pit_bin_masses(u)
    bins_ok = bool(np.all((masses >= PIT_BIN_LO) & (masses <= PIT_BIN_HI)))
    pit_pass = bool(ks <= PIT_KS_MAX and bins_ok)
    logger.info(
        "G3 PIT: KS %.4f (<= %.3f) %s | bins in [%.4f, %.4f] %s",
        ks, PIT_KS_MAX, "PASS" if ks <= PIT_KS_MAX else "FAIL",
        PIT_BIN_LO, PIT_BIN_HI, "PASS" if bins_ok else "FAIL",
    )
    marg_all_pass = (
        all(r["pass"] for r in marg_rows)
        and woba_row["pass"] and woba_row_db["pass"]
        and palen_row["pass"] and palen_row_db["pass"]
        and cal_valid_cov >= CAL_VALID_COVERAGE_MIN
    )
    G3 = GateResult(
        "G3", bool(pit_pass and marg_all_pass),
        {
            "PIT": {
                "ks_distance": ks, "ks_threshold": PIT_KS_MAX,
                "bin_masses": [float(x) for x in masses],
                "bin_band": [PIT_BIN_LO, PIT_BIN_HI],
                "bins_within_band": bins_ok, "n": int(u.size),
                "pass": pit_pass,
            },
            "marginals": marg_rows,
            "mean_woba": {
                "comparator_A_like_for_like": woba_row,
                "comparator_B_db_woba_value": woba_row_db,
            },
            "mean_pa_length": {
                "comparator_A_horizon6_capped": palen_row,
                "comparator_B_uncapped_sql": palen_row_db,
            },
            "calibration_valid_coverage": {
                "value": cal_valid_cov, "threshold": CAL_VALID_COVERAGE_MIN,
                "pass": bool(cal_valid_cov >= CAL_VALID_COVERAGE_MIN),
                "definition": (
                    "share of dev PA starts whose every categorical context "
                    "field value was observed in the 2023 fit cohort and whose "
                    "umpire scalar lies inside the fit cohort's range"
                ),
            },
            "hr_given_hit_2023_fit": hr_given_hit_2023,
            "rollout_truncated_share": float((term == 5).mean()),
        },
    )

    # ── G4 ───────────────────────────────────────────────────────────────
    cs_rows = []
    v2_cs = {}
    for cs in range(12):
        sel3 = v3o["count_state"] == cs
        sel2 = v2o["count_state"] == cs
        n3 = int(sel3.sum())
        e3 = top1_ece(v3o["probs"][sel3], v3o["labels"][sel3]) if n3 else float("nan")
        e2 = (
            top1_ece(v2o["probs"][sel2], v2o["labels"][sel2])
            if int(sel2.sum()) else float("nan")
        )
        v2_cs[cs] = e2
        thr = tighten_threshold(COUNT_STATE_ECE_MAX, e2) if np.isfinite(e2) else {
            "effective_threshold": COUNT_STATE_ECE_MAX, "tightened": False,
            "spec_threshold": COUNT_STATE_ECE_MAX, "frozen_v2_dev_value": None,
        }
        gated = n3 >= COUNT_STATE_MIN_OBS
        cs_rows.append({
            "count_state": cs,
            "count": f"{cs // 3}-{cs % 3}",
            "n": n3,
            "v3_top1_ece": e3,
            "frozen_v2_top1_ece": e2,
            "threshold": thr,
            "gated": gated,
            "pass": bool(e3 <= thr["effective_threshold"]) if gated else None,
        })
    pos_gap = position_marginal_gap(roll["position_marginals"], emp_marg)
    kl = position_kl_spearman(roll["position_marginals"], emp_marg)
    cs_pass = all(r["pass"] for r in cs_rows if r["gated"])
    logger.info(
        "G4 position gate: max |rollout - empirical| = %.4fpp (thr %.2fpp) at "
        "pos %d class %s -> %s",
        100 * pos_gap["max_abs_gap"], 100 * POSITION_GAP_MAX,
        pos_gap["max_at_position"], OUTCOME_CLASSES[pos_gap["max_at_class"]],
        "PASS" if pos_gap["pass"] else "FAIL",
    )
    G4 = GateResult(
        "G4", bool(cs_pass and pos_gap["pass"]),
        {
            "per_count_state": cs_rows,
            "per_count_state_pass": cs_pass,
            "position_gate": {
                "max_abs_gap_pp": 100 * pos_gap["max_abs_gap"],
                "threshold_pp": 100 * POSITION_GAP_MAX,
                "max_at_position": pos_gap["max_at_position"],
                "max_at_class": OUTCOME_CLASSES[pos_gap["max_at_class"]],
                "per_position_max_pp": [100 * x for x in pos_gap["per_position_max"]],
                "pass": pos_gap["pass"],
            },
            "secondary_position_kl_spearman": kl,
            "rollout_position_marginals": roll["position_marginals"].tolist(),
            "empirical_position_marginals": emp_marg.tolist(),
            "empirical_position_counts": emp_counts.tolist(),
        },
    )

    # ── G5 ───────────────────────────────────────────────────────────────
    g5 = {
        "K": decision_deciles(p_k, actual["K"]),
        "BB": decision_deciles(p_bb, actual["BB"]),
        "HR": decision_deciles(p_hr, actual["HR"]),
    }
    for d, r in g5.items():
        logger.info(
            "G5 %s: max gated decile gap %.4fpp (thr %.2fpp), weighted %.4fpp "
            "(thr %.2fpp) -> %s",
            d, 100 * r["max_gated_gap"], 100 * DECISION_DECILE_GAP_MAX,
            100 * r["weighted_mean_abs_gap"], 100 * DECISION_WEIGHTED_GAP_MAX,
            "PASS" if r["pass"] else "FAIL",
        )
    G5 = GateResult("G5", all(r["pass"] for r in g5.values()), {"per_decision": g5})

    # ── verdict (§6.8) ───────────────────────────────────────────────────
    gates = [G1, G2, G3, G4, G5]
    verdict = (
        "PASS"
        if (G1.passed and G3.passed and G4.passed and G5.passed
            and (G2.passed or G2.void))
        else "FAIL"
    )
    for g in gates:
        logger.info("%s -> %s", g.name, g.verdict())
    logger.info("DEV-TIER (2024, BURNED) VERDICT: %s", verdict)

    frozen_sha_post = {p.name: sha256_file(p) for p in (v2_path, a1_path)}
    assert frozen_sha_pre == frozen_sha_post, "a frozen artifact was mutated!"

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _ROOT / "results" / "pitchgpt_v3" / f"dev_2024_{args.tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tier": "burned-dev",
        "tier_warning": (
            "2024 is the BURNED dev tier (§5.3). These numbers carry NO "
            "validation authority. The one graded contact is against sealed "
            "2026 at season end (§5.5) and was NOT made."
        ),
        "lockbox_2026_contacted": False,
        "holdout_2025_contacted": False,
        "spec_path": SPEC_PATH,
        "spec_freeze_sha": SPEC_FREEZE_SHA,
        "spec_body_sha256": SPEC_BODY_SHA256,
        "git_sha": git_sha(),
        "cohort": pa["meta"],
        "n_pa": n_pa,
        "n_samples_per_pa": args.n_samples,
        "horizon": HORIZON,
        "seed_convention": "42 + pa_index * 1000 (§4.3.2)",
        "temperatures": temps.to_dict(),
        "calibration_provenance": {
            k: cal_payload[k]
            for k in ("fit_cohort_season", "fit_seed", "fit_n_pas", "produced_by")
        },
        "backbone_checkpoint": str(args.backbone),
        "backbone_stage": bb_ck["meta"]["stage"],
        "backbone_sha256": sha256_file(args.backbone),
        "outcome_head_sha256": sha256_file(args.outcome_head),
        "frozen_artifact_sha256_pre": frozen_sha_pre,
        "frozen_artifact_sha256_post": frozen_sha_post,
        "a1_temperature": a1_temperature,
        "verdict": verdict,
        "gates": {g.name: {"verdict": g.verdict(), **g.detail} for g in gates},
        "empirical_spec_literal_sql": empirical_B,
        "empirical_like_for_like": {
            **emp_A,
            "mean_woba": emp_woba,
            "mean_pa_length_capped": emp_palen_capped,
            "pa_terminal_share": terminal_share(emp_term),
        },
        "model_pa_terminal_share": terminal_share(term),
        "mask_events": roll["mask_events"],
        "pa_terminal_names": list(PA_TERMINAL_NAMES),
        "rollout_elapsed_sec": roll["elapsed_sec"],
        "elapsed_sec": round(time.perf_counter() - t_start, 1),
        "duckdb_read_only": True,
    }
    (out_dir / "audit.json").write_text(json.dumps(payload, indent=2, default=str))
    logger.info("audit -> %s", out_dir / "audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
