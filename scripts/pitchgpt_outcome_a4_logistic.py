"""
A4 — multinomial logistic regression on engineered features.

Plan B §3 hypothesis A4.  sklearn.LogisticRegression(solver='lbfgs',
max_iter=2000), tune C on 2023 val.

Notes:
- Spec called for solver='saga' at full scale (3M rows × 68 features),
  but saga did not converge in 300 iterations and lbfgs is materially
  faster on this dense problem.  Penalty stays L2 (the default for
  lbfgs).  All other tuning identical to the spec.
- For tractable wall-clock at full scale, train fitting subsamples to
  500K rows by default (--train-subsample arg).  Empirical: log-loss
  on val plateaus past ~250K rows for this 68-feature problem.

Features (Plan B §3): pitch_type, zone, release_speed, count, outs,
on_1b/2b/3b, batter_stand, pitcher_throws, inning, score_diff,
umpire_scalar, pitcher_id (one-hot via target encoding), batter_id (TE).

For high-cardinality pitcher_id / batter_id we use **target-mean encoding
fitted on train**: each pitcher_id gets a 7-D probability vector of train
outcomes, used as 7 numeric features per row at inference.  This avoids
the catastrophic dimensionality blow-up of one-hotting ~3K pitchers.

Outputs: results/pitchgpt_sim/outcome_baselines_2026_04_25/a4_logistic/{metrics.json, report.md}
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
logger = logging.getLogger("a4_logistic")

OUT_DIR = (
    _ROOT / "results" / "pitchgpt_sim" / "outcome_baselines_2026_04_25"
    / "a4_logistic"
)


# ═════════════════════════════════════════════════════════════════════════════
# Target encoding for high-cardinality categoricals
# ═════════════════════════════════════════════════════════════════════════════


class TargetEncoder7Class:
    """Empirical-prior encoder.  For each unique value of ``col``, store
    a smoothed 7-D probability vector over outcomes from the train set.
    Unknown values at inference fall back to the global prior."""

    def __init__(self, alpha: float = 5.0):
        self.alpha = alpha
        self.lookup: dict[Any, np.ndarray] = {}
        self.global_prior: np.ndarray = np.zeros(NUM_OUTCOME_CLASSES)

    def fit(self, vals: np.ndarray, y: np.ndarray) -> None:
        K = NUM_OUTCOME_CLASSES
        global_counts = np.bincount(y, minlength=K).astype(np.float64) + self.alpha
        self.global_prior = global_counts / global_counts.sum()

        df = pd.DataFrame({"v": vals, "y": y})
        grp = df.groupby("v")["y"].apply(
            lambda s: np.bincount(s.to_numpy(), minlength=K).astype(np.float64),
        )
        for k, c in grp.items():
            smoothed = c + self.alpha * self.global_prior
            self.lookup[k] = smoothed / smoothed.sum()

    def transform(self, vals: np.ndarray) -> np.ndarray:
        out = np.empty((len(vals), NUM_OUTCOME_CLASSES), dtype=np.float64)
        for i, v in enumerate(vals):
            out[i] = self.lookup.get(v, self.global_prior)
        return out


# ═════════════════════════════════════════════════════════════════════════════
# Featuriser
# ═════════════════════════════════════════════════════════════════════════════


def make_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build (X_train, X_val, X_test) numpy arrays from feature DataFrames.

    Categorical features are one-hot encoded (sklearn OneHotEncoder).
    pitcher_id and batter_id are target-encoded (7 features each).
    Continuous features are standardised.
    """
    y_train = train_df["outcome_label"].to_numpy()
    y_val = val_df["outcome_label"].to_numpy()
    y_test = test_df["outcome_label"].to_numpy()

    # Target encoders fit on train.
    pid_enc = TargetEncoder7Class(alpha=5.0)
    bid_enc = TargetEncoder7Class(alpha=5.0)
    pid_enc.fit(train_df["pitcher_id"].to_numpy(), y_train)
    bid_enc.fit(train_df["batter_id"].to_numpy(), y_train)

    pid_train = pid_enc.transform(train_df["pitcher_id"].to_numpy())
    pid_val = pid_enc.transform(val_df["pitcher_id"].to_numpy())
    pid_test = pid_enc.transform(test_df["pitcher_id"].to_numpy())

    bid_train = bid_enc.transform(train_df["batter_id"].to_numpy())
    bid_val = bid_enc.transform(val_df["batter_id"].to_numpy())
    bid_test = bid_enc.transform(test_df["batter_id"].to_numpy())

    # One-hot the lower-cardinality categoricals.
    cat_cols = [
        "pitch_type", "stand", "p_throws",
    ]
    ohe = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, dtype=np.float32,
    )
    ohe.fit(train_df[cat_cols].astype(str))
    cat_train = ohe.transform(train_df[cat_cols].astype(str))
    cat_val = ohe.transform(val_df[cat_cols].astype(str))
    cat_test = ohe.transform(test_df[cat_cols].astype(str))

    # Treat zone, balls, strikes, outs, inning_bucket as one-hot too
    # (low cardinality, ordinal isn't quite right for them).
    int_cat_cols = ["zone", "balls", "strikes", "outs_when_up", "inning_bucket"]
    ohe_int = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False, dtype=np.float32,
    )
    ohe_int.fit(train_df[int_cat_cols].astype(int))
    int_train = ohe_int.transform(train_df[int_cat_cols].astype(int))
    int_val = ohe_int.transform(val_df[int_cat_cols].astype(int))
    int_test = ohe_int.transform(test_df[int_cat_cols].astype(int))

    # Boolean runners as plain 0/1.
    bool_cols = ["on_1b_bool", "on_2b_bool", "on_3b_bool"]
    bool_train = train_df[bool_cols].to_numpy(dtype=np.float32)
    bool_val = val_df[bool_cols].to_numpy(dtype=np.float32)
    bool_test = test_df[bool_cols].to_numpy(dtype=np.float32)

    # Continuous: scale on train.
    num_cols = ["release_speed", "umpire_scalar", "score_diff"]
    scaler = StandardScaler()
    scaler.fit(train_df[num_cols].to_numpy())
    num_train = scaler.transform(train_df[num_cols].to_numpy()).astype(np.float32)
    num_val = scaler.transform(val_df[num_cols].to_numpy()).astype(np.float32)
    num_test = scaler.transform(test_df[num_cols].to_numpy()).astype(np.float32)

    X_train = np.hstack([
        cat_train, int_train, bool_train, num_train,
        pid_train.astype(np.float32), bid_train.astype(np.float32),
    ])
    X_val = np.hstack([
        cat_val, int_val, bool_val, num_val,
        pid_val.astype(np.float32), bid_val.astype(np.float32),
    ])
    X_test = np.hstack([
        cat_test, int_test, bool_test, num_test,
        pid_test.astype(np.float32), bid_test.astype(np.float32),
    ])

    feature_names = (
        [f"oh_cat_{c}" for c in ohe.get_feature_names_out(cat_cols)]
        + [f"oh_int_{c}" for c in ohe_int.get_feature_names_out(int_cat_cols)]
        + bool_cols
        + num_cols
        + [f"pid_te_{OUTCOME_CLASSES[c]}" for c in range(NUM_OUTCOME_CLASSES)]
        + [f"bid_te_{OUTCOME_CLASSES[c]}" for c in range(NUM_OUTCOME_CLASSES)]
    )
    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names,
    }


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
    p.add_argument("--C-grid", type=str, default="0.1,0.3,1.0,3.0",
                   help="Comma-separated C values to tune on val")
    p.add_argument("--max-iter", type=int, default=2000)
    p.add_argument("--train-subsample", type=int, default=500_000,
                   help="Cap rows for LR fit to keep wall-clock tractable. "
                        "0 = no cap.  Default 500K balances accuracy vs time.")
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
    feats = make_features(train_df, val_df, test_df, seed=args.seed)
    feat_dt = time.time() - t1
    logger.info("Features built in %.1fs.  X_train=%s  X_val=%s  X_test=%s",
                feat_dt, feats["X_train"].shape, feats["X_val"].shape,
                feats["X_test"].shape)

    X_train, y_train = feats["X_train"], feats["y_train"]
    X_val, y_val = feats["X_val"], feats["y_val"]
    X_test, y_test = feats["X_test"], feats["y_test"]

    # Subsample train for tractable LR fit (lbfgs on 3M x 68 takes hours).
    train_freq = class_frequency_prior(y_train)
    if args.train_subsample > 0 and args.train_subsample < len(y_train):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(y_train), size=args.train_subsample, replace=False)
        idx = np.sort(idx)
        X_train_fit = X_train[idx]
        y_train_fit = y_train[idx]
        logger.info("Subsampled train to %d rows for LR fit "
                    "(of %d original)", args.train_subsample, len(y_train))
    else:
        X_train_fit = X_train
        y_train_fit = y_train
    val_freq_ll = freq_prior_log_loss_on(y_val, train_freq)
    test_freq_ll = freq_prior_log_loss_on(y_test, train_freq)

    # Class weights — inverse-frequency cap 10.
    counts = np.bincount(y_train, minlength=NUM_OUTCOME_CLASSES).astype(np.float64)
    freq = counts / counts.sum()
    inv = np.minimum(1.0 / np.clip(freq, 1e-12, None), 10.0)
    inv = inv * (NUM_OUTCOME_CLASSES / inv.sum())
    class_weight = {i: float(inv[i]) for i in range(NUM_OUTCOME_CLASSES)}

    # Tune C on val.
    Cs = [float(c) for c in args.C_grid.split(",")]
    best_C = None
    best_val_ll = float("inf")
    tune_history = []
    logger.info("Tuning C on 2023 val (grid: %s)...", Cs)
    for C in Cs:
        t1 = time.time()
        clf = LogisticRegression(
            C=C, solver="lbfgs",
            max_iter=args.max_iter,
            class_weight=class_weight,
            random_state=args.seed,
            verbose=0,
            tol=1e-3,
        )
        clf.fit(X_train_fit, y_train_fit)
        # Val log-loss.
        val_probs = clf.predict_proba(X_val)
        ll = log_loss_from_probs(val_probs, y_val)
        dt = time.time() - t1
        logger.info("  C=%g  val_ll=%.4f  iter=%d  fit=%.1fs",
                    C, ll, getattr(clf, "n_iter_", [None])[0]
                    if hasattr(clf, "n_iter_") else 0, dt)
        tune_history.append({"C": C, "val_log_loss": ll, "fit_seconds": dt})
        if ll < best_val_ll:
            best_val_ll = ll
            best_C = C

    logger.info("Best C: %g (val_ll=%.4f).  Refitting...", best_C, best_val_ll)

    # Refit at best C and evaluate.
    t1 = time.time()
    clf = LogisticRegression(
        C=best_C, solver="lbfgs",
        max_iter=args.max_iter,
        class_weight=class_weight,
        random_state=args.seed,
        verbose=0,
        tol=1e-3,
    )
    clf.fit(X_train_fit, y_train_fit)
    fit_dt = time.time() - t1
    logger.info("Final fit in %.1fs.", fit_dt)

    # Get val + test probs.
    val_probs = clf.predict_proba(X_val)
    test_probs = clf.predict_proba(X_test)

    # Pre-temp metrics.
    val_ll_pre = log_loss_from_probs(val_probs, y_val)
    test_ll_pre = log_loss_from_probs(test_probs, y_test)
    val_ece_pre = ece_10bin(val_probs, y_val)
    test_ece_pre = ece_10bin(test_probs, y_test)

    # Fit temperature on val (use logits via decision_function-style log).
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

    # Bootstrap CI on test.
    logger.info("Bootstrapping test log-loss + lift CI (B=%d)...", args.n_boot)
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
    logger.info("A4 RESULTS")
    logger.info("-" * 60)
    logger.info("Best C: %g  (val ll: %.4f)", best_C, best_val_ll)
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
        "variant": "a4_logistic",
        "config": {
            "train_games": args.train_games,
            "val_games": args.val_games,
            "test_games": args.test_games,
            "C_grid": Cs,
            "C_best": best_C,
            "max_iter": args.max_iter,
            "seed": args.seed,
            "n_boot": args.n_boot,
            "class_weight": class_weight,
            "train_subsample": args.train_subsample,
            "n_train_fit": int(len(y_train_fit)),
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
        "tune_history": tune_history,
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
    lines.append("# A4 — Multinomial Logistic Regression on Engineered Features")
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
    lines.append(f"- Best C: {cfg['C_best']}  Temperature: {val['temperature']:.4f}")
    lines.append("")
    lines.append("## Cohort")
    lines.append("")
    lines.append(f"- Train rows: {cohort['train_rows']:,}")
    lines.append(f"- Val rows (2023 pitcher-disjoint): {cohort['val_rows']:,}")
    lines.append(f"- Test rows (2025 pitcher-disjoint): {cohort['test_rows']:,}")
    lines.append(f"- Feature width: {cohort['n_features']}")
    lines.append("")
    lines.append("## Tune history (val log-loss per C)")
    lines.append("")
    lines.append("| C | val log-loss | fit (s) |")
    lines.append("|---|---:|---:|")
    for r in m["tune_history"]:
        lines.append(f"| {r['C']:g} | {r['val_log_loss']:.4f} | "
                     f"{r['fit_seconds']:.1f} |")
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
    lines.append("## Val metrics (used for hyperparam + T)")
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
    lines.append(f"- Top-50 pitchers: n={pp['n_pitchers']}")
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
