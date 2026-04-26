"""
A5 — per-pitcher × per-count × per-pitch-type empirical-prior baseline.

Plan B §3 hypothesis A5.  Builds a Dirichlet-smoothed (alpha=1) lookup
table on the train cohort, with a hierarchical fallback chain:

    (pitcher_id, count, pitch_type, batter_hand)   -- depth 4
    (count, pitch_type, batter_hand)               -- depth 3 (no pid)
    (pitch_type, batter_hand)                      -- depth 2
    overall train frequency prior                   -- depth 0

This is non-parametric: at inference we look up the deepest available
bucket per row.  No model fit beyond the Dirichlet smoothing.

For temperature scaling on this baseline: because outputs are
already-calibrated proportions (frequency tables), we still fit a
single scalar via log-temperature so we can compare ECE
apples-to-apples with A3/A4.

Outputs: results/pitchgpt_sim/outcome_baselines_2026_04_25/a5_empirical/{metrics.json, report.md}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.pitchgpt_outcome_baselines_common import (  # noqa: E402
    DEFAULT_SEED,
    DEFAULT_TEST_GAMES,
    DEFAULT_TRAIN_GAMES,
    DEFAULT_VAL_GAMES,
    bootstrap_log_loss,
    build_cohort,
    class_frequency_prior,
    ece_10bin,
    fit_temperature,
    freq_prior_log_loss_on,
    log_loss_from_probs,
    per_class_log_loss,
    per_pitcher_log_loss,
    softmax,
    top1_accuracy,
)
from src.analytics.pitchgpt_outcome_head import (  # noqa: E402
    NUM_OUTCOME_CLASSES,
    OUTCOME_CLASSES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("a5_empirical")

OUT_DIR = (
    _ROOT / "results" / "pitchgpt_sim" / "outcome_baselines_2026_04_25"
    / "a5_empirical"
)


# ═════════════════════════════════════════════════════════════════════════════
# Smoothed lookup-table builder
# ═════════════════════════════════════════════════════════════════════════════


def build_lookup(
    df: pd.DataFrame, levels: list[list[str]], alpha: float = 1.0,
) -> list[dict[tuple, np.ndarray]]:
    """Build a Dirichlet-smoothed lookup at each level.

    Each level is a list of column names defining the bucket.  Returns a
    list of dicts: each dict maps the bucket tuple to a smoothed
    probability vector over 7 classes.

    The hierarchy is fall-through: caller iterates in order from deepest
    to shallowest; first hit wins.
    """
    out: list[dict[tuple, np.ndarray]] = []
    K = NUM_OUTCOME_CLASSES
    for cols in levels:
        d: dict[tuple, np.ndarray] = {}
        if not cols:
            # Global prior — single (None,) bucket.
            counts = np.bincount(df["outcome_label"].to_numpy(), minlength=K).astype(
                np.float64,
            )
            counts = counts + alpha
            d[("__global__",)] = counts / counts.sum()
        else:
            grouped = df.groupby(cols, sort=False)["outcome_label"].apply(
                lambda s: np.bincount(s.to_numpy(), minlength=K).astype(np.float64),
            )
            for key, counts in grouped.items():
                if not isinstance(key, tuple):
                    key = (key,)
                smoothed = counts + alpha
                d[key] = smoothed / smoothed.sum()
        out.append(d)
    return out


def predict_with_lookup(
    df: pd.DataFrame,
    lookups: list[dict[tuple, np.ndarray]],
    levels: list[list[str]],
    global_prior: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Predict probabilities by hierarchical fall-through.

    Returns
    -------
    probs : (n, 7) array
    depth_used : list of which level depth (index into levels) was used
                 for each row.  -1 means fell off the bottom and the
                 global prior was used.
    """
    n = len(df)
    K = NUM_OUTCOME_CLASSES
    probs = np.empty((n, K), dtype=np.float64)
    depth = np.full(n, -1, dtype=np.int8)

    # Pre-compute key tuples per level (vectorised on numpy arrays).
    level_arrays = []
    for cols in levels:
        if not cols:
            # Global level — empty key tuple.
            level_arrays.append(None)
            continue
        # Build tuples row-by-row.  For ~150K rows × ~5 levels this is
        # tolerable; pandas itertuples is faster than apply.
        arr = []
        for col in cols:
            arr.append(df[col].to_numpy())
        level_arrays.append(arr)

    n_levels = len(levels)
    for i in range(n):
        filled = False
        for d_idx in range(n_levels):
            cols = levels[d_idx]
            lookup = lookups[d_idx]
            if not cols:
                # Global level — use the global prior.
                probs[i] = global_prior
                depth[i] = d_idx
                filled = True
                break
            key = tuple(int(level_arrays[d_idx][k][i]) if isinstance(level_arrays[d_idx][k][i], (np.integer, int))
                        else level_arrays[d_idx][k][i]
                        for k in range(len(cols)))
            v = lookup.get(key)
            if v is not None:
                probs[i] = v
                depth[i] = d_idx
                filled = True
                break
        if not filled:
            probs[i] = global_prior
            depth[i] = -1
    return probs, depth.tolist()


def predict_fast(
    df: pd.DataFrame,
    lookups: list[dict[tuple, np.ndarray]],
    levels: list[list[str]],
    global_prior: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised version using pandas merge-style joins.

    For each level, build a DataFrame from the lookup, then merge.
    Take the first non-null per row.
    """
    n = len(df)
    K = NUM_OUTCOME_CLASSES
    probs = np.full((n, K), np.nan, dtype=np.float64)
    depth = np.full(n, -1, dtype=np.int8)

    used = np.zeros(n, dtype=bool)
    for d_idx, (cols, lookup) in enumerate(zip(levels, lookups)):
        if used.all():
            break
        if not cols:
            # Global level — fill remaining.
            mask = ~used
            probs[mask] = global_prior
            depth[mask] = d_idx
            used[mask] = True
            continue

        # Convert lookup to a DataFrame for merging.
        if not lookup:
            continue
        keys = list(lookup.keys())
        vals = np.stack([lookup[k] for k in keys], axis=0)
        ld_data = {col: [k[i] for k in keys] for i, col in enumerate(cols)}
        for c in range(K):
            ld_data[f"_p{c}"] = vals[:, c]
        ld = pd.DataFrame(ld_data)

        # Restrict to unfilled rows.
        idx_unfilled = np.where(~used)[0]
        if len(idx_unfilled) == 0:
            break
        sub = df.iloc[idx_unfilled].reset_index(drop=True)
        merged = sub.merge(ld, on=cols, how="left")

        # Find rows where we got a hit.
        hit_mask = merged["_p0"].notna().to_numpy()
        if hit_mask.sum() == 0:
            continue

        for c in range(K):
            probs[idx_unfilled[hit_mask], c] = merged.loc[hit_mask, f"_p{c}"].to_numpy()
        depth[idx_unfilled[hit_mask]] = d_idx
        used[idx_unfilled[hit_mask]] = True

    # Anyone still unfilled gets global prior.
    if not used.all():
        mask = ~used
        probs[mask] = global_prior
        depth[mask] = -1
        used[mask] = True
    return probs, depth


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-games", type=int, default=DEFAULT_TRAIN_GAMES)
    p.add_argument("--val-games", type=int, default=DEFAULT_VAL_GAMES)
    p.add_argument("--test-games", type=int, default=DEFAULT_TEST_GAMES)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="Dirichlet smoothing alpha (default 1)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    logger.info("Building cohort...")
    train_df, val_df, test_df = build_cohort(
        train_games=args.train_games,
        val_games=args.val_games,
        test_games=args.test_games,
        seed=args.seed,
        cache_dir=_ROOT / "data" / "staging" / "outcome_baselines_cache",
    )
    cohort_dt = time.time() - t0
    logger.info("Cohort built in %.1fs.  train=%d  val=%d  test=%d",
                cohort_dt, len(train_df), len(val_df), len(test_df))

    # Hierarchy levels (deepest first, falling through to shallowest).
    LEVELS = [
        ["pitcher_id", "balls", "strikes", "pitch_type", "stand"],
        ["balls", "strikes", "pitch_type", "stand"],
        ["pitch_type", "stand"],
        [],  # global prior fallback
    ]

    logger.info("Building hierarchical Dirichlet-smoothed lookups (alpha=%g)...",
                args.alpha)
    t1 = time.time()
    lookups = build_lookup(train_df, LEVELS, alpha=args.alpha)
    build_dt = time.time() - t1
    logger.info("Lookups built in %.1fs.", build_dt)
    for i, (cols, d) in enumerate(zip(LEVELS, lookups)):
        logger.info("  Level %d (cols=%s): %d unique buckets", i, cols, len(d))

    train_freq = class_frequency_prior(train_df["outcome_label"].to_numpy())
    logger.info("Train frequency prior: %s",
                {OUTCOME_CLASSES[i]: round(float(train_freq[i]), 4)
                 for i in range(NUM_OUTCOME_CLASSES)})

    # Predict on val + test.
    logger.info("Predicting on val (n=%d)...", len(val_df))
    t1 = time.time()
    val_probs, val_depth = predict_fast(val_df, lookups, LEVELS, train_freq)
    logger.info("Val predict in %.1fs.", time.time() - t1)

    logger.info("Predicting on test (n=%d)...", len(test_df))
    t1 = time.time()
    test_probs, test_depth = predict_fast(test_df, lookups, LEVELS, train_freq)
    logger.info("Test predict in %.1fs.", time.time() - t1)

    val_y = val_df["outcome_label"].to_numpy()
    test_y = test_df["outcome_label"].to_numpy()

    # Pre-temp metrics.
    val_ll_pre = log_loss_from_probs(val_probs, val_y)
    test_ll_pre = log_loss_from_probs(test_probs, test_y)
    val_ece_pre = ece_10bin(val_probs, val_y)
    test_ece_pre = ece_10bin(test_probs, test_y)

    # Fit temperature on val (treat probs as logits via log).
    eps = 1e-12
    val_logits = np.log(np.clip(val_probs, eps, 1.0))
    test_logits = np.log(np.clip(test_probs, eps, 1.0))
    T_opt = fit_temperature(val_logits, val_y)
    logger.info("Fitted temperature on val: %.4f", T_opt)

    val_probs_post = softmax(val_logits / T_opt)
    test_probs_post = softmax(test_logits / T_opt)

    val_ll_post = log_loss_from_probs(val_probs_post, val_y)
    test_ll_post = log_loss_from_probs(test_probs_post, test_y)
    val_ece_post = ece_10bin(val_probs_post, val_y)
    test_ece_post = ece_10bin(test_probs_post, test_y)

    val_freq_ll = freq_prior_log_loss_on(val_y, train_freq)
    test_freq_ll = freq_prior_log_loss_on(test_y, train_freq)

    val_lift_pre = 1.0 - val_ll_pre / val_freq_ll
    val_lift_post = 1.0 - val_ll_post / val_freq_ll
    test_lift_pre = 1.0 - test_ll_pre / test_freq_ll
    test_lift_post = 1.0 - test_ll_post / test_freq_ll

    # Bootstrap on test set (the headline number).
    logger.info("Bootstrapping log-loss CI (n=%d, B=%d)...",
                len(test_y), args.n_boot)
    t1 = time.time()
    test_ll_post_pt, test_ll_post_lo, test_ll_post_hi = bootstrap_log_loss(
        test_probs_post, test_y, n_boot=args.n_boot, seed=args.seed,
    )
    # CI on the LIFT vs prior (paired by row).
    rng = np.random.default_rng(args.seed)
    n = len(test_y)
    lift_samples = np.empty(args.n_boot)
    for b in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        ll_m = log_loss_from_probs(test_probs_post[idx], test_y[idx])
        ll_p = freq_prior_log_loss_on(test_y[idx], train_freq)
        lift_samples[b] = 1.0 - ll_m / max(ll_p, 1e-12)
    test_lift_post_lo = float(np.percentile(lift_samples, 2.5))
    test_lift_post_hi = float(np.percentile(lift_samples, 97.5))
    logger.info("Bootstrap done in %.1fs", time.time() - t1)

    test_per_class = per_class_log_loss(test_probs_post, test_y)
    test_acc = top1_accuracy(test_probs_post, test_y)

    pitcher_ll = per_pitcher_log_loss(
        test_probs_post, test_y, test_df["pitcher_id"].to_numpy(), top_k=50,
    )
    if pitcher_ll:
        pitcher_lls = [r["log_loss"] for r in pitcher_ll]
        pitcher_var = float(np.var(pitcher_lls))
        pitcher_min = float(min(pitcher_lls))
        pitcher_max = float(max(pitcher_lls))
        pitcher_mean = float(np.mean(pitcher_lls))
    else:
        pitcher_var = pitcher_min = pitcher_max = pitcher_mean = float("nan")

    # Depth utilisation diagnostic.
    depth_arr = np.asarray(test_depth)
    depth_dist = {
        f"level_{i}": int((depth_arr == i).sum()) for i in range(len(LEVELS))
    }
    depth_dist["fallback_global"] = int((depth_arr == -1).sum())

    # Gates.
    GATE_PASS_LIFT = 0.10
    GATE_PASS_CI_LO = 0.05
    GATE_WEAK_LIFT = 0.05
    GATE_WEAK_CI_LO = 0.02
    GATE_ECE_MAX = 0.05
    GATE_HIT_LL_PASS = 2.0
    GATE_HIT_LL_WEAK = 2.5
    GATE_HBP_LL_PASS = 4.0
    GATE_HBP_LL_WEAK = 5.0

    hit_ll = test_per_class.get("in_play_hit", float("nan"))
    hbp_ll = test_per_class.get("hbp", float("nan"))

    if (test_lift_post >= GATE_PASS_LIFT
            and test_lift_post_lo >= GATE_PASS_CI_LO
            and test_ece_post < GATE_ECE_MAX
            and hit_ll < GATE_HIT_LL_PASS
            and hbp_ll < GATE_HBP_LL_PASS):
        verdict = "PASS"
    elif (test_lift_post >= GATE_WEAK_LIFT
            and test_lift_post_lo >= GATE_WEAK_CI_LO
            and test_ece_post < GATE_ECE_MAX
            and hit_ll < GATE_HIT_LL_WEAK
            and hbp_ll < GATE_HBP_LL_WEAK):
        verdict = "WEAKER PASS"
    else:
        verdict = "FAIL"

    logger.info("-" * 60)
    logger.info("A5 RESULTS")
    logger.info("-" * 60)
    logger.info("Val   pre/post log-loss: %.4f / %.4f  freq=%.4f  lift_post=%.2f%%",
                val_ll_pre, val_ll_post, val_freq_ll, 100 * val_lift_post)
    logger.info("Val   pre/post ECE:      %.4f / %.4f", val_ece_pre, val_ece_post)
    logger.info("Test  pre/post log-loss: %.4f / %.4f  freq=%.4f  "
                "lift_post=%.2f%% [%.2f%%, %.2f%%]",
                test_ll_pre, test_ll_post, test_freq_ll,
                100 * test_lift_post,
                100 * test_lift_post_lo, 100 * test_lift_post_hi)
    logger.info("Test  pre/post ECE:      %.4f / %.4f", test_ece_pre, test_ece_post)
    logger.info("Test  log-loss CI:       %.4f [%.4f, %.4f]",
                test_ll_post_pt, test_ll_post_lo, test_ll_post_hi)
    logger.info("Test  per-class log-loss: %s",
                {k: round(v, 4) for k, v in test_per_class.items()})
    logger.info("Test  top-1 accuracy: %.4f", test_acc)
    logger.info("Per-pitcher log-loss across top-50 pitchers: "
                "mean=%.4f  var=%.4f  range=[%.4f, %.4f]",
                pitcher_mean, pitcher_var, pitcher_min, pitcher_max)
    logger.info("Lookup depth utilisation (test): %s", depth_dist)
    logger.info("VERDICT: %s", verdict)

    # Write metrics.json.
    metrics = {
        "variant": "a5_empirical",
        "config": {
            "train_games": args.train_games,
            "val_games": args.val_games,
            "test_games": args.test_games,
            "alpha": args.alpha,
            "seed": args.seed,
            "n_boot": args.n_boot,
            "levels": [str(c) for c in LEVELS],
        },
        "cohort": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_freq_prior": {
                OUTCOME_CLASSES[i]: float(train_freq[i])
                for i in range(NUM_OUTCOME_CLASSES)
            },
        },
        "val_metrics": {
            "log_loss_pre_temp": val_ll_pre,
            "log_loss_post_temp": val_ll_post,
            "ece_pre_temp": val_ece_pre,
            "ece_post_temp": val_ece_post,
            "freq_prior_log_loss": val_freq_ll,
            "lift_pre_temp": val_lift_pre,
            "lift_post_temp": val_lift_post,
            "temperature": T_opt,
        },
        "test_metrics": {
            "log_loss_pre_temp": test_ll_pre,
            "log_loss_post_temp": test_ll_post,
            "ece_pre_temp": test_ece_pre,
            "ece_post_temp": test_ece_post,
            "freq_prior_log_loss": test_freq_ll,
            "lift_pre_temp": test_lift_pre,
            "lift_post_temp": test_lift_post,
            "log_loss_post_ci_lo": test_ll_post_lo,
            "log_loss_post_ci_hi": test_ll_post_hi,
            "lift_ci_lo": test_lift_post_lo,
            "lift_ci_hi": test_lift_post_hi,
            "per_class_log_loss": test_per_class,
            "top1_accuracy": test_acc,
            "per_pitcher_log_loss": pitcher_ll,
            "per_pitcher_summary": {
                "mean": pitcher_mean, "var": pitcher_var,
                "min": pitcher_min, "max": pitcher_max,
                "n_pitchers": len(pitcher_ll),
            },
            "depth_utilisation": depth_dist,
        },
        "gates": {
            "verdict": verdict,
            "PASS_lift_threshold": GATE_PASS_LIFT,
            "PASS_ci_lo_threshold": GATE_PASS_CI_LO,
            "WEAK_lift_threshold": GATE_WEAK_LIFT,
            "WEAK_ci_lo_threshold": GATE_WEAK_CI_LO,
            "ECE_max": GATE_ECE_MAX,
            "hit_ll_PASS_max": GATE_HIT_LL_PASS,
            "hit_ll_WEAK_max": GATE_HIT_LL_WEAK,
            "hbp_ll_PASS_max": GATE_HBP_LL_PASS,
            "hbp_ll_WEAK_max": GATE_HBP_LL_WEAK,
        },
        "wall_clock": {
            "cohort_seconds": cohort_dt,
            "build_seconds": build_dt,
            "total_seconds": time.time() - t0,
        },
    }
    out_path = OUT_DIR / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    logger.info("Wrote %s", out_path)

    # Write report.md.
    _write_report(OUT_DIR / "report.md", metrics)
    logger.info("Wrote %s", OUT_DIR / "report.md")
    logger.info("DONE.  Verdict: %s", verdict)
    return 0


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    return str(obj)


def _write_report(path: Path, m: dict) -> None:
    v = m["test_metrics"]
    val = m["val_metrics"]
    g = m["gates"]
    cfg = m["config"]
    cohort = m["cohort"]

    lift = 100 * v["lift_post_temp"]
    lift_lo = 100 * v["lift_ci_lo"]
    lift_hi = 100 * v["lift_ci_hi"]
    val_lift = 100 * val["lift_post_temp"]

    lines = []
    lines.append("# A5 — Per-Pitcher × Per-Count × Per-Pitch-Type Empirical Priors")
    lines.append("")
    lines.append(f"**Verdict:** **{g['verdict']}**")
    lines.append("")
    lines.append("## Headline (2025 holdout, post-temperature)")
    lines.append("")
    lines.append(f"- 7-class log-loss: **{v['log_loss_post_temp']:.4f}**  "
                 f"(freq prior: {v['freq_prior_log_loss']:.4f})")
    lines.append(f"- Lift vs frequency prior: **{lift:+.2f}%** "
                 f"(95% CI [{lift_lo:+.2f}%, {lift_hi:+.2f}%])")
    lines.append(f"- 10-bin ECE post-temp: **{v['ece_post_temp']:.4f}**")
    lines.append(f"- Top-1 accuracy: {v['top1_accuracy']:.4f}")
    lines.append(f"- Temperature: {val['temperature']:.4f} (fit on 2023 val)")
    lines.append("")
    lines.append("## Cohort")
    lines.append("")
    lines.append(f"- Train rows: {cohort['train_rows']:,}")
    lines.append(f"- Val rows (2023 pitcher-disjoint): {cohort['val_rows']:,}")
    lines.append(f"- Test rows (2025 pitcher-disjoint): {cohort['test_rows']:,}")
    lines.append("")
    lines.append("## Per-class log-loss (test, post-temp)")
    lines.append("")
    lines.append("| class | log-loss |")
    lines.append("|-------|---------:|")
    for c in OUTCOME_CLASSES:
        ll = v["per_class_log_loss"].get(c)
        ll_str = f"{ll:.4f}" if ll is not None and not (isinstance(ll, float) and np.isnan(ll)) else "NA"
        lines.append(f"| {c} | {ll_str} |")
    lines.append("")
    lines.append("## Val metrics (2023 pitcher-disjoint, used for hyperparam + T)")
    lines.append("")
    lines.append(f"- Log-loss pre/post temp: {val['log_loss_pre_temp']:.4f} / "
                 f"{val['log_loss_post_temp']:.4f}")
    lines.append(f"- ECE pre/post temp: {val['ece_pre_temp']:.4f} / "
                 f"{val['ece_post_temp']:.4f}")
    lines.append(f"- Lift vs freq prior: {val_lift:+.2f}%")
    lines.append("")
    lines.append("## Per-pitcher stability (test)")
    lines.append("")
    pp = v["per_pitcher_summary"]
    lines.append(f"- Top-50 pitchers (n>=30 rows each): n={pp['n_pitchers']}")
    lines.append(f"- Log-loss mean / var / range: {pp['mean']:.4f} / "
                 f"{pp['var']:.4f} / [{pp['min']:.4f}, {pp['max']:.4f}]")
    lines.append("")
    lines.append("## Lookup utilisation (test)")
    lines.append("")
    for k, n in v["depth_utilisation"].items():
        lines.append(f"- {k}: {n:,}")
    lines.append("")
    lines.append("## Hierarchy levels")
    lines.append("")
    for i, lev in enumerate(cfg["levels"]):
        lines.append(f"- Level {i}: {lev}")
    lines.append("")
    lines.append("## Gate criteria")
    lines.append("")
    lines.append(f"- PASS: lift >= {100*g['PASS_lift_threshold']:.0f}% "
                 f"AND CI lower >= {100*g['PASS_ci_lo_threshold']:.0f}% "
                 f"AND ECE < {g['ECE_max']:.2f} "
                 f"AND hit ll < {g['hit_ll_PASS_max']:.1f} "
                 f"AND hbp ll < {g['hbp_ll_PASS_max']:.1f}")
    lines.append(f"- WEAKER PASS: lift >= {100*g['WEAK_lift_threshold']:.0f}% "
                 f"AND CI lower >= {100*g['WEAK_ci_lo_threshold']:.0f}% "
                 f"AND ECE < {g['ECE_max']:.2f} "
                 f"AND hit ll < {g['hit_ll_WEAK_max']:.1f} "
                 f"AND hbp ll < {g['hbp_ll_WEAK_max']:.1f}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
