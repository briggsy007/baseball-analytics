"""
A3 — XGBoost on engineered features (Plan B priority 1).

Plan B §3 hypothesis A3.  XGBoost multi-class softmax over the 7-class
target with inverse-frequency class weights (cap=10), tuned via 5-fold CV
on TRAIN ONLY (no val touch), final fit on the full train, temperature-
scaled on 2023 val, evaluated on 2025 holdout.

Features per Plan B §3:
    pitch_type, zone, release_speed (continuous), count, outs,
    on_1b/2b/3b, batter_stand, pitcher_throws, inning, score_diff,
    umpire_scalar, pitcher_id, batter_id

XGBoost handles categoricals natively (enable_categorical=True), so we
keep them as integer-encoded categoricals (label-encoded) rather than
one-hotting.  pitcher_id and batter_id are also integer-encoded (XGBoost
handles high-cardinality categoricals via histogram + interaction
search).  This is a cleaner XGB approach than target-encoding.

Hyperparameter grid (per Plan B §3 spec):
    max_depth in {4, 6, 8}
    learning_rate in {0.05, 0.08, 0.12}
    n_estimators=400 with early-stopping at 20 rounds (stopping on val).

CV: 5-fold on TRAIN; selects best (max_depth, lr) on average CV log-loss.
Final fit uses early-stopping on 2023 VAL log-loss for n_estimators.

Outputs: results/pitchgpt_sim/outcome_baselines_2026_04_25/a3_xgboost/{metrics.json, report.md}
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
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

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
logger = logging.getLogger("a3_xgboost")

OUT_DIR = (
    _ROOT / "results" / "pitchgpt_sim" / "outcome_baselines_2026_04_25"
    / "a3_xgboost"
)


# ═════════════════════════════════════════════════════════════════════════════
# Featuriser
# ═════════════════════════════════════════════════════════════════════════════


def make_xgb_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    """Build per-frame feature DataFrames suitable for XGBoost.

    Categorical columns are integer-encoded by their string repr;
    XGBoost is fed enable_categorical=True for them.  pitcher_id /
    batter_id are also treated as categoricals.

    Returns a dict with feature DataFrames, label arrays, and
    cat_feature column names.
    """
    cat_cols = [
        "pitch_type", "zone", "balls", "strikes", "outs_when_up",
        "on_1b_bool", "on_2b_bool", "on_3b_bool",
        "stand", "p_throws", "inning_bucket",
        "pitcher_id", "batter_id",
    ]
    num_cols = ["release_speed", "umpire_scalar", "score_diff"]

    # Build label encoders on train + val + test combined (so unseen
    # categories at test get a code, not "ignore").  Note: this is the
    # standard XGB pattern; the leakage guard is at the cohort level
    # (pitcher-disjoint), not the encoding level.
    #
    # We must use a *shared* pd.CategoricalDtype across all three frames so
    # XGBoost sees a consistent category set (it fails with "Found a
    # category not in the training set" otherwise).
    cat_dtypes: dict[str, pd.CategoricalDtype] = {}
    for col in cat_cols:
        all_vals = pd.concat([train_df[col], val_df[col], test_df[col]]).astype(str)
        cats = sorted(set(all_vals))
        cat_dtypes[col] = pd.CategoricalDtype(categories=cats, ordered=False)

    def encode(df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame()
        for col in cat_cols:
            out[col] = df[col].astype(str).astype(cat_dtypes[col])
        for col in num_cols:
            out[col] = df[col].astype(np.float32)
        return out

    X_train = encode(train_df)
    X_val = encode(val_df)
    X_test = encode(test_df)

    return {
        "X_train": X_train,
        "y_train": train_df["outcome_label"].to_numpy(),
        "X_val": X_val,
        "y_val": val_df["outcome_label"].to_numpy(),
        "X_test": X_test,
        "y_test": test_df["outcome_label"].to_numpy(),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Train one XGB model
# ═════════════════════════════════════════════════════════════════════════════


def train_xgb(
    X_train: pd.DataFrame, y_train: np.ndarray,
    X_eval: pd.DataFrame, y_eval: np.ndarray,
    sample_weight: np.ndarray | None,
    max_depth: int, learning_rate: float, n_estimators: int,
    early_stopping_rounds: int,
    seed: int = DEFAULT_SEED,
    verbose: bool = False,
) -> xgb.XGBClassifier:
    clf = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=NUM_OUTCOME_CLASSES,
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_weight=3,
        tree_method="hist",
        enable_categorical=True,
        random_state=seed,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="mlogloss",
        n_jobs=-1,
        verbosity=0,
    )
    clf.fit(
        X_train, y_train,
        sample_weight=sample_weight,
        eval_set=[(X_eval, y_eval)],
        verbose=verbose,
    )
    return clf


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-games", type=int, default=DEFAULT_TRAIN_GAMES)
    p.add_argument("--val-games", type=int, default=DEFAULT_VAL_GAMES)
    p.add_argument("--test-games", type=int, default=DEFAULT_TEST_GAMES)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--max-depths", type=str, default="4,6,8")
    p.add_argument("--learning-rates", type=str, default="0.05,0.08,0.12")
    p.add_argument("--n-estimators", type=int, default=400)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--early-stop", type=int, default=20)
    p.add_argument("--cv-subsample", type=int, default=300_000,
                   help="Cap row count for CV folds to keep tuning fast")
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

    logger.info("Featurising...")
    t1 = time.time()
    feats = make_xgb_features(train_df, val_df, test_df)
    feat_dt = time.time() - t1
    logger.info("Features built in %.1fs.", feat_dt)

    X_train, y_train = feats["X_train"], feats["y_train"]
    X_val, y_val = feats["X_val"], feats["y_val"]
    X_test, y_test = feats["X_test"], feats["y_test"]

    train_freq = class_frequency_prior(y_train)
    val_freq_ll = freq_prior_log_loss_on(y_val, train_freq)
    test_freq_ll = freq_prior_log_loss_on(y_test, train_freq)

    # Inverse-frequency class weights, cap 10.
    counts = np.bincount(y_train, minlength=NUM_OUTCOME_CLASSES).astype(np.float64)
    freq = counts / counts.sum()
    inv = np.minimum(1.0 / np.clip(freq, 1e-12, None), 10.0)
    inv = inv * (NUM_OUTCOME_CLASSES / inv.sum())
    sample_weight = inv[y_train].astype(np.float32)
    logger.info("Class weights: %s",
                {OUTCOME_CLASSES[i]: round(float(inv[i]), 3)
                 for i in range(NUM_OUTCOME_CLASSES)})

    # 5-fold CV on TRAIN ONLY.  Subsample for speed.
    max_depths = [int(x) for x in args.max_depths.split(",")]
    lrs = [float(x) for x in args.learning_rates.split(",")]
    cv_subsample = min(args.cv_subsample, len(X_train))

    rng = np.random.default_rng(args.seed)
    if cv_subsample < len(X_train):
        sub_idx = rng.choice(len(X_train), size=cv_subsample, replace=False)
        sub_idx = np.sort(sub_idx)
    else:
        sub_idx = np.arange(len(X_train))
    X_cv = X_train.iloc[sub_idx].reset_index(drop=True)
    y_cv = y_train[sub_idx]
    w_cv = sample_weight[sub_idx]
    logger.info("CV subsample: %d rows (%.1f%% of train)",
                len(X_cv), 100 * len(X_cv) / len(X_train))

    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                          random_state=args.seed)

    grid = []
    cv_history = []
    logger.info("Tuning via %d-fold CV (max_depths=%s, learning_rates=%s)...",
                args.cv_folds, max_depths, lrs)
    for max_depth in max_depths:
        for lr in lrs:
            t1 = time.time()
            fold_lls = []
            for fold_idx, (tr_idx, vl_idx) in enumerate(skf.split(X_cv, y_cv)):
                X_tr = X_cv.iloc[tr_idx].reset_index(drop=True)
                X_vl = X_cv.iloc[vl_idx].reset_index(drop=True)
                y_tr = y_cv[tr_idx]
                y_vl = y_cv[vl_idx]
                w_tr = w_cv[tr_idx]

                clf = train_xgb(
                    X_tr, y_tr, X_vl, y_vl, sample_weight=w_tr,
                    max_depth=max_depth, learning_rate=lr,
                    n_estimators=args.n_estimators,
                    early_stopping_rounds=args.early_stop,
                    seed=args.seed,
                )
                probs = clf.predict_proba(X_vl)
                ll = log_loss_from_probs(probs, y_vl)
                fold_lls.append(ll)
            mean_ll = float(np.mean(fold_lls))
            std_ll = float(np.std(fold_lls))
            dt = time.time() - t1
            logger.info("  max_depth=%d lr=%g  CV ll=%.4f +/- %.4f  (%.1fs)",
                        max_depth, lr, mean_ll, std_ll, dt)
            grid.append({
                "max_depth": max_depth, "learning_rate": lr,
                "cv_log_loss_mean": mean_ll, "cv_log_loss_std": std_ll,
                "fold_log_losses": fold_lls, "fit_seconds": dt,
            })
            cv_history.append({
                "max_depth": max_depth, "learning_rate": lr,
                "cv_log_loss_mean": mean_ll, "cv_log_loss_std": std_ll,
            })

    grid.sort(key=lambda r: r["cv_log_loss_mean"])
    best = grid[0]
    logger.info("Best CV: max_depth=%d lr=%g  CV ll=%.4f",
                best["max_depth"], best["learning_rate"],
                best["cv_log_loss_mean"])

    # Final fit at best hyperparams on FULL train, early-stop on 2023 val.
    logger.info("Refitting on full train, early-stop on 2023 val (n=%d)...",
                len(X_val))
    t1 = time.time()
    final_clf = train_xgb(
        X_train, y_train, X_val, y_val, sample_weight=sample_weight,
        max_depth=best["max_depth"], learning_rate=best["learning_rate"],
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stop,
        seed=args.seed,
    )
    fit_dt = time.time() - t1
    best_iter = getattr(final_clf, "best_iteration", None)
    logger.info("Final fit: %.1fs  best_iteration=%s",
                fit_dt, best_iter)

    val_probs = final_clf.predict_proba(X_val)
    test_probs = final_clf.predict_proba(X_test)

    val_ll_pre = log_loss_from_probs(val_probs, y_val)
    test_ll_pre = log_loss_from_probs(test_probs, y_test)
    val_ece_pre = ece_10bin(val_probs, y_val)
    test_ece_pre = ece_10bin(test_probs, y_test)

    eps = 1e-12
    val_logits = np.log(np.clip(val_probs, eps, 1.0))
    test_logits = np.log(np.clip(test_probs, eps, 1.0))
    T_opt = fit_temperature(val_logits, y_val)
    logger.info("Fitted temperature on val: %.4f", T_opt)

    val_probs_post = softmax(val_logits / T_opt)
    test_probs_post = softmax(test_logits / T_opt)

    val_ll_post = log_loss_from_probs(val_probs_post, y_val)
    test_ll_post = log_loss_from_probs(test_probs_post, y_test)
    val_ece_post = ece_10bin(val_probs_post, y_val)
    test_ece_post = ece_10bin(test_probs_post, y_test)

    val_lift_post = 1.0 - val_ll_post / val_freq_ll
    test_lift_pre = 1.0 - test_ll_pre / test_freq_ll
    test_lift_post = 1.0 - test_ll_post / test_freq_ll

    logger.info("Bootstrapping log-loss + lift CI (B=%d)...", args.n_boot)
    t1 = time.time()
    test_ll_post_pt, test_ll_post_lo, test_ll_post_hi = bootstrap_log_loss(
        test_probs_post, y_test, n_boot=args.n_boot, seed=args.seed,
    )
    rng = np.random.default_rng(args.seed)
    n_t = len(y_test)
    lift_samples = np.empty(args.n_boot)
    for b in range(args.n_boot):
        idx = rng.integers(0, n_t, size=n_t)
        ll_m = log_loss_from_probs(test_probs_post[idx], y_test[idx])
        ll_p = freq_prior_log_loss_on(y_test[idx], train_freq)
        lift_samples[b] = 1.0 - ll_m / max(ll_p, 1e-12)
    test_lift_post_lo = float(np.percentile(lift_samples, 2.5))
    test_lift_post_hi = float(np.percentile(lift_samples, 97.5))
    logger.info("Bootstrap done in %.1fs.", time.time() - t1)

    test_per_class = per_class_log_loss(test_probs_post, y_test)
    test_acc = top1_accuracy(test_probs_post, y_test)
    pitcher_ll = per_pitcher_log_loss(
        test_probs_post, y_test, test_df["pitcher_id"].to_numpy(), top_k=50,
    )
    if pitcher_ll:
        plls = [r["log_loss"] for r in pitcher_ll]
        pitcher_var = float(np.var(plls))
        pitcher_min = float(min(plls))
        pitcher_max = float(max(plls))
        pitcher_mean = float(np.mean(plls))
    else:
        pitcher_var = pitcher_min = pitcher_max = pitcher_mean = float("nan")

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
    logger.info("A3 RESULTS")
    logger.info("-" * 60)
    logger.info("Best params: max_depth=%d lr=%g  CV ll=%.4f",
                best["max_depth"], best["learning_rate"],
                best["cv_log_loss_mean"])
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
    logger.info("VERDICT: %s", verdict)

    metrics = {
        "variant": "a3_xgboost",
        "config": {
            "train_games": args.train_games,
            "val_games": args.val_games,
            "test_games": args.test_games,
            "max_depths_grid": max_depths,
            "learning_rates_grid": lrs,
            "n_estimators": args.n_estimators,
            "cv_folds": args.cv_folds,
            "cv_subsample": cv_subsample,
            "early_stop": args.early_stop,
            "seed": args.seed,
            "n_boot": args.n_boot,
            "best_max_depth": best["max_depth"],
            "best_learning_rate": best["learning_rate"],
            "best_iteration": int(best_iter) if best_iter is not None else None,
            "class_weights": {OUTCOME_CLASSES[i]: float(inv[i])
                              for i in range(NUM_OUTCOME_CLASSES)},
        },
        "cohort": {
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "n_features": int(X_train.shape[1]),
            "train_freq_prior": {
                OUTCOME_CLASSES[i]: float(train_freq[i])
                for i in range(NUM_OUTCOME_CLASSES)
            },
        },
        "cv_history": cv_history,
        "val_metrics": {
            "log_loss_pre_temp": val_ll_pre,
            "log_loss_post_temp": val_ll_post,
            "ece_pre_temp": val_ece_pre,
            "ece_post_temp": val_ece_post,
            "freq_prior_log_loss": val_freq_ll,
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
            "feat_seconds": feat_dt,
            "fit_seconds": fit_dt,
            "total_seconds": time.time() - t0,
        },
    }
    out_path = OUT_DIR / "metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_json_default)
    logger.info("Wrote %s", out_path)

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
    lines.append("# A3 — XGBoost on Engineered Features")
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
    lines.append(f"- Best params: max_depth={cfg['best_max_depth']}, "
                 f"lr={cfg['best_learning_rate']}, "
                 f"best_iteration={cfg['best_iteration']}")
    lines.append(f"- Temperature: {val['temperature']:.4f}")
    lines.append("")
    lines.append("## Cohort")
    lines.append("")
    lines.append(f"- Train rows: {cohort['train_rows']:,}")
    lines.append(f"- Val rows (2023 pitcher-disjoint): {cohort['val_rows']:,}")
    lines.append(f"- Test rows (2025 pitcher-disjoint): {cohort['test_rows']:,}")
    lines.append(f"- Feature width: {cohort['n_features']}")
    lines.append("")
    lines.append("## CV history (mean per (max_depth, learning_rate))")
    lines.append("")
    lines.append("| max_depth | learning_rate | CV log-loss mean | std |")
    lines.append("|---:|---:|---:|---:|")
    for r in m["cv_history"]:
        lines.append(f"| {r['max_depth']} | {r['learning_rate']:g} | "
                     f"{r['cv_log_loss_mean']:.4f} | "
                     f"{r['cv_log_loss_std']:.4f} |")
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
    lines.append("## Val metrics (used for early-stopping + temperature)")
    lines.append("")
    lines.append(f"- Log-loss pre/post: {val['log_loss_pre_temp']:.4f} / "
                 f"{val['log_loss_post_temp']:.4f}")
    lines.append(f"- ECE pre/post: {val['ece_pre_temp']:.4f} / "
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
    lines.append("## Wall clock")
    lines.append("")
    wc = m["wall_clock"]
    lines.append(f"- Cohort: {wc['cohort_seconds']:.1f}s")
    lines.append(f"- Featurise: {wc['feat_seconds']:.1f}s")
    lines.append(f"- Final fit: {wc['fit_seconds']:.1f}s")
    lines.append(f"- Total: {wc['total_seconds']:.1f}s")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
