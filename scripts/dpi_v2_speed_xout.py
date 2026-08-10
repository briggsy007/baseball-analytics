"""DPI v2: sprint-speed xOut variant -- WS3.3 of the 2026-08-10 platform
improvement plan (task C1a).

Retrains the xOut classifier on the FROZEN recipe's train window
(2015-2022, HistGB 200/depth6/lr0.05/seed42, stratified 80/20 holdout)
with ONE added feature: the batter's same-season Savant sprint speed,
active ONLY on grounders/weak contact (Savant xBA recipe; MLB Tech Blog /
fantasy.fangraphs.com sprint-speed xBA adjustment), neutral-filled with a
MEASURED league mean elsewhere, and constrained monotonic-decreasing
(``monotonic_cst=-1``) so a faster runner can never RAISE the predicted
out probability.

Design decisions (pre-declared):
  * League mean is MEASURED (plan: "impute league mean ~27 ft/s (measure
    the actual league mean, don't assume)"): the BIP-weighted mean sprint
    speed over matched applicable rows in the 2015-2022 train window.  ONE
    global value is used for both imputation and the neutral fill so the
    feature encodes zero season-level information.
  * Comparison design: the frozen checkpoint's own 80/20 split indices are
    not reconstructible (DB row order may have changed since 2026-04-18),
    so scoring the frozen model on this run's holdout may include rows the
    frozen model trained on (overlap inflates its AUC, biasing the delta
    AGAINST the speed variant -- conservative).  A 4-feature control model
    retrained on the IDENTICAL train rows/seed is therefore the clean
    apples-to-apples baseline; the frozen-model row ships as reference.

Artifacts: NEW path models/defensive_pressing/xout_v2_speed_2026_08_10.pkl
(never overwrites anything), registered as experimental version
``v2026.08.10-speed`` in models/registry.json (write-once dir, hash_policy
pinned).  NO alias changes: production/frozen_validated are untouched.

Read-only vs the DB; writes the new artifact + results/ outputs only.

Usage:
    python scripts/dpi_v2_speed_xout.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")  # shared box: 3 lanes on CPU

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.db.schema import get_connection  # noqa: E402
from src.analytics import defensive_pressing as dp  # noqa: E402
from src.analytics.registry import ModelRegistry, sha256_of_file  # noqa: E402

FROZEN_CHECKPOINT = REPO_ROOT / "models" / "defensive_pressing" / "xout_v1.pkl"
FROZEN_SHA256 = "e689bff6ab069474c57df6950ba3ed7d376de8b0a3a7a71861d2a96dc3d3bb39"
NEW_ARTIFACT = (
    REPO_ROOT / "models" / "defensive_pressing" / "xout_v2_speed_2026_08_10.pkl"
)
REGISTRY_VERSION = "v2026.08.10-speed"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "defensive_pressing" / "v2_2026-08"

TRAIN_SEASONS = list(range(2015, 2023))  # 2015-2022, frozen recipe
CORE_FEATURES = ["launch_speed", "launch_angle", "spray_angle", "bb_type_encoded"]
SPEED_FEATURE = "sprint_speed_gbweak"


def sha256_of(path: Path) -> str:
    return sha256_of_file(path)


def pull_train_bip(conn) -> pd.DataFrame:
    """BIP rows for the frozen recipe cohort, plus batter_id + season."""
    seasons = ", ".join(str(s) for s in TRAIN_SEASONS)
    return conn.execute(
        f"""
        SELECT
            CAST(EXTRACT(YEAR FROM game_date) AS INT) AS season,
            batter_id, launch_speed, launch_angle, hc_x, hc_y,
            bb_type, events
        FROM pitches
        WHERE type = 'X'
          AND launch_speed IS NOT NULL
          AND launch_angle IS NOT NULL
          AND hc_x IS NOT NULL
          AND hc_y IS NOT NULL
          AND bb_type IS NOT NULL
          AND events IS NOT NULL
          AND EXTRACT(YEAR FROM game_date) IN ({seasons})
        """
    ).fetchdf()


def pull_sprint_speed(conn) -> pd.DataFrame:
    return conn.execute(
        "SELECT season, player_id, sprint_speed FROM sprint_speed "
        "WHERE sprint_speed IS NOT NULL"
    ).fetchdf()


def bootstrap_delta_auc(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray,
    n_boot: int, rng: np.random.Generator,
) -> dict:
    """Paired bootstrap over test rows: CI for AUC(b) - AUC(a)."""
    from sklearn.metrics import roc_auc_score

    n = len(y)
    deltas = np.full(n_boot, np.nan)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        deltas[i] = roc_auc_score(yb, p_b[idx]) - roc_auc_score(yb, p_a[idx])
    deltas = deltas[~np.isnan(deltas)]
    return {
        "delta_ci_lo": round(float(np.percentile(deltas, 2.5)), 5),
        "delta_ci_hi": round(float(np.percentile(deltas, 97.5)), 5),
        "n_boot": int(len(deltas)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--skip-registry", action="store_true")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    import joblib
    import sklearn

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if NEW_ARTIFACT.exists():
        raise SystemExit(
            f"{NEW_ARTIFACT} already exists -- refusing to overwrite ANY "
            "existing model artifact. Choose a new path."
        )

    # --- frozen checkpoint: verify hash BEFORE loading ---------------------
    actual = sha256_of(FROZEN_CHECKPOINT)
    if actual != FROZEN_SHA256:
        raise SystemExit(
            f"Frozen checkpoint hash mismatch: {actual} != {FROZEN_SHA256}. "
            "Refusing to use it as the comparison baseline."
        )
    frozen_bundle = joblib.load(FROZEN_CHECKPOINT)
    frozen_model = frozen_bundle["model"]
    frozen_meta = frozen_bundle.get("metadata", {})
    print(f"Frozen model verified (sha256 {actual[:12]}...), "
          f"train_seasons={frozen_meta.get('train_seasons')}, "
          f"recorded holdout AUC={frozen_meta.get('auc')}")

    # --- data ---------------------------------------------------------------
    conn = get_connection(read_only=True)
    try:
        print("Pulling 2015-2022 BIP cohort (read-only) ...")
        df = pull_train_bip(conn)
        print(f"  {len(df)} BIP rows")
        ss = pull_sprint_speed(conn)
        print(f"  {len(ss)} sprint-speed player-season rows")
    finally:
        conn.close()

    target = dp._is_out(df["events"])
    features_core = dp.build_bip_features(
        df, include_park=False, include_weather=False
    )
    # Core-clean mask (parity with train_expected_out_model): SQL already
    # requires non-null inputs, so this only guards derived-NaN edge cases.
    mask = features_core[CORE_FEATURES].notna().all(axis=1)
    df = df.loc[mask].reset_index(drop=True)
    features_core = features_core.loc[mask].reset_index(drop=True)
    target = target.loc[mask].reset_index(drop=True)
    n_samples = len(df)
    print(f"  {n_samples} clean rows (manifest n_samples for frozen: 837571)")

    # --- sprint-speed feature ------------------------------------------------
    speed_map = {
        (int(r.season), int(r.player_id)): float(r.sprint_speed)
        for r in ss.itertuples()
    }
    applicable = dp.sprint_speed_applicable_mask(df["bb_type"], df["launch_speed"])
    keys = list(zip(df["season"].astype(int), df["batter_id"].astype(int)))
    matched_speed = pd.Series(
        [speed_map.get(k, np.nan) for k in keys], index=df.index, dtype=float
    )
    matched = matched_speed.notna()

    # League means: MEASURED. Global = BIP-weighted mean over matched
    # applicable train rows (used for impute + neutral fill). Per-season
    # player-level means reported for the record.
    league_mean_global = float(matched_speed[applicable & matched].mean())
    per_season_player_mean = (
        ss[ss["season"].isin(TRAIN_SEASONS)]
        .groupby("season")["sprint_speed"].mean().round(3).to_dict()
    )
    per_season_all = ss.groupby("season")["sprint_speed"].mean().round(3).to_dict()
    coverage = pd.DataFrame({
        "season": df["season"],
        "applicable": applicable.astype(int),
        "matched": matched.astype(int),
    })
    cov_tbl = coverage.groupby("season").agg(
        n_bip=("applicable", "size"),
        n_applicable=("applicable", "sum"),
        n_matched_all=("matched", "sum"),
    ).reset_index()
    cov_app = (
        coverage[coverage["applicable"] == 1]
        .groupby("season")["matched"].agg(["size", "sum"]).reset_index()
        .rename(columns={"size": "n_applicable_rows", "sum": "n_applicable_matched"})
    )
    cov_tbl = cov_tbl.merge(cov_app, on="season")
    cov_tbl["pct_applicable"] = (100 * cov_tbl["n_applicable"] / cov_tbl["n_bip"]).round(2)
    cov_tbl["pct_applicable_matched"] = (
        100 * cov_tbl["n_applicable_matched"] / cov_tbl["n_applicable_rows"]
    ).round(2)
    cov_tbl.to_csv(out_dir / "speed_feature_coverage_by_season.csv", index=False)
    print(f"League mean sprint speed (global, BIP-weighted, matched applicable "
          f"train rows): {league_mean_global:.4f} ft/s")

    features_speed = features_core.copy()
    features_speed[SPEED_FEATURE] = dp.build_sprint_speed_feature(
        df["batter_id"], df["season"], df["bb_type"], df["launch_speed"],
        speed_map, league_mean_global,
    )

    # --- split (frozen recipe: stratified 80/20, seed 42) --------------------
    cfg = dp.DPIConfig()
    train_idx, test_idx = train_test_split(
        np.arange(n_samples), test_size=0.2,
        random_state=cfg.random_state, stratify=target,
    )
    y_train = target.iloc[train_idx].to_numpy()
    y_test = target.iloc[test_idx].to_numpy()

    def fit_hgb(X_train, monotonic_cst=None):
        m = HistGradientBoostingClassifier(
            max_iter=cfg.xout_n_estimators,
            max_depth=cfg.xout_max_depth,
            learning_rate=cfg.xout_learning_rate,
            random_state=cfg.random_state,
            monotonic_cst=monotonic_cst,
        )
        m.fit(X_train, y_train)
        return m

    print("Training 4-feature control (identical rows/seed) ...")
    Xc_train = features_core.iloc[train_idx]
    Xc_test = features_core.iloc[test_idx]
    control = fit_hgb(Xc_train)

    print("Training 5-feature speed variant (monotonic_cst=-1 on speed) ...")
    mono = [0, 0, 0, 0, -1]
    Xs_train = features_speed.iloc[train_idx]
    Xs_test = features_speed.iloc[test_idx]
    variant = fit_hgb(Xs_train, monotonic_cst=mono)

    # --- scoring --------------------------------------------------------------
    frozen_cols = list(frozen_meta.get("feature_columns") or CORE_FEATURES)
    p_frozen = frozen_model.predict_proba(Xc_test[frozen_cols])[:, 1]
    p_control = control.predict_proba(Xc_test)[:, 1]
    p_variant = variant.predict_proba(Xs_test)[:, 1]

    auc_frozen = float(roc_auc_score(y_test, p_frozen))
    auc_control = float(roc_auc_score(y_test, p_control))
    auc_variant = float(roc_auc_score(y_test, p_variant))

    rng = np.random.default_rng(args.seed)
    d_vs_control = bootstrap_delta_auc(y_test, p_control, p_variant, args.n_boot, rng)
    rng = np.random.default_rng(args.seed)
    d_vs_frozen = bootstrap_delta_auc(y_test, p_frozen, p_variant, args.n_boot, rng)

    app_test = applicable.iloc[test_idx].to_numpy()
    subset = {}
    for name, m in [("applicable_gb_weak", app_test), ("not_applicable", ~app_test)]:
        subset[name] = {
            "n": int(m.sum()),
            "out_rate": round(float(y_test[m].mean()), 4),
            "auc_frozen": round(float(roc_auc_score(y_test[m], p_frozen[m])), 5),
            "auc_control": round(float(roc_auc_score(y_test[m], p_control[m])), 5),
            "auc_variant": round(float(roc_auc_score(y_test[m], p_variant[m])), 5),
        }

    # --- monotonicity verification --------------------------------------------
    rng2 = np.random.default_rng(args.seed)
    check_idx = rng2.choice(np.flatnonzero(app_test), size=min(2000, int(app_test.sum())), replace=False)
    grid = np.array([23.0, 24.5, 26.0, 27.5, 29.0, 30.5])
    base_rows = Xs_test.iloc[check_idx].copy()
    prev = None
    mono_ok = True
    for g in grid:
        rows = base_rows.copy()
        rows[SPEED_FEATURE] = g
        p = variant.predict_proba(rows)[:, 1]
        if prev is not None and np.any(p > prev + 1e-9):
            mono_ok = False
        prev = p
    print(f"Monotonicity check (P(out) non-increasing in speed on {len(check_idx)} "
          f"applicable rows x {len(grid)} speeds): {'PASS' if mono_ok else 'FAIL'}")

    # --- persist NEW artifact ---------------------------------------------------
    metadata = {
        "variant": "xout_v2_speed (WS3.3, experimental)",
        "train_seasons": TRAIN_SEASONS,
        "n_samples": int(n_samples),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "auc": round(auc_variant, 4),
        "out_rate": round(float(target.mean()), 4),
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": CORE_FEATURES + [SPEED_FEATURE],
        "monotonic_cst": mono,
        "speed_feature": {
            "definition": "batter same-season Savant sprint speed on "
                          "grounders (bb_type='ground_ball') OR weak contact "
                          f"(EV < {dp.SPRINT_SPEED_WEAK_EV_MAX}); league-mean "
                          "neutral fill elsewhere and league-mean imputation "
                          "for unmatched batters",
            "league_mean_used": round(league_mean_global, 4),
            "league_mean_basis": "BIP-weighted mean over matched applicable "
                                 "rows, 2015-2022 train window",
            "source_table": "sprint_speed (Savant leaderboard, min_opp=10)",
        },
        "config": {
            "xout_n_estimators": cfg.xout_n_estimators,
            "xout_max_depth": cfg.xout_max_depth,
            "xout_learning_rate": cfg.xout_learning_rate,
            "random_state": cfg.random_state,
        },
        "use_park": False,
        "use_weather": False,
        "sklearn_version": sklearn.__version__,
        "comparison_note": "frozen xOut scored on this run's holdout may "
                           "overlap its own 2026-04-18 train split (split "
                           "indices not reconstructible); the 4-feature "
                           "same-split control is the clean baseline.",
    }
    NEW_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": variant, "metadata": metadata}, NEW_ARTIFACT)
    artifact_sha = sha256_of(NEW_ARTIFACT)
    print(f"Persisted {NEW_ARTIFACT} (sha256 {artifact_sha[:12]}...)")

    registry_result = None
    if not args.skip_registry:
        reg = ModelRegistry()
        try:
            manifest = reg.register_version(
                "defensive_pressing",
                REGISTRY_VERSION,
                artifact=NEW_ARTIFACT,
                sha256=artifact_sha,
                hash_policy="pinned",
                train_window="2015-2022",
                data_snapshot={
                    "tables": {
                        "pitches": {"train_bip_rows": int(n_samples)},
                        "sprint_speed": {
                            "player_season_rows": int(len(ss)),
                            "ingest_log": "results/ingest/sprint_speed_ingest_20260810T154343Z.json",
                        },
                    }
                },
                training_script="scripts/dpi_v2_speed_xout.py",
                spec_version="docs/plans/2026-08-10_platform_improvement_plan.md WS3.3",
                validation_results_ref="results/defensive_pressing/v2_2026-08/speed_xout_summary.json",
                notes="EXPERIMENTAL DPI v2 speed variant (WS3.3). Not "
                      "production, not frozen_validated; aliases untouched.",
            )
            registry_result = {"registered": True, "version": REGISTRY_VERSION,
                               "sha256": manifest["sha256"]}
        except FileExistsError as exc:
            registry_result = {"registered": False, "note": str(exc)}
        print(f"Registry: {registry_result}")

    # --- summary -------------------------------------------------------------
    summary = {
        "task": "WS3.3 sprint-speed xOut variant -- C1a",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "frozen_checkpoint": {
            "path": str(FROZEN_CHECKPOINT),
            "sha256": actual,
            "recorded_holdout_auc": frozen_meta.get("auc"),
        },
        "environment": {
            "python": sys.version.split()[0],
            "sklearn": sklearn.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "frozen_pickle_note": "frozen checkpoint pickled under sklearn "
                                  "1.8.0, scored here under the installed "
                                  "version; A5 parity check verified fidelity",
        },
        "cohort": {
            "train_window": "2015-2022",
            "n_samples": int(n_samples),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "split": "stratified 80/20, random_state 42 (frozen recipe)",
            "manifest_frozen_n_samples": 837571,
        },
        "league_mean_sprint_speed": {
            "global_bip_weighted_matched_applicable_2015_2022": round(league_mean_global, 4),
            "per_season_player_level_mean_train_window": per_season_player_mean,
            "per_season_player_level_mean_all": per_season_all,
            "used_for": "imputation of unmatched batters AND neutral fill on "
                        "non-applicable rows (one global value: zero "
                        "season-level information in the feature)",
        },
        "holdout_auc": {
            "frozen_as_loaded": round(auc_frozen, 5),
            "control_4feat_same_split": round(auc_control, 5),
            "variant_5feat_speed": round(auc_variant, 5),
            "delta_variant_minus_control": round(auc_variant - auc_control, 5),
            "delta_variant_minus_control_ci95": [
                d_vs_control["delta_ci_lo"], d_vs_control["delta_ci_hi"]],
            "delta_variant_minus_frozen": round(auc_variant - auc_frozen, 5),
            "delta_variant_minus_frozen_ci95": [
                d_vs_frozen["delta_ci_lo"], d_vs_frozen["delta_ci_hi"]],
            "n_boot": args.n_boot,
            "status": "holdout within the 2015-2022 train window (frozen "
                      "recipe convention), not future-season OOS",
        },
        "subset_auc": subset,
        "monotonicity_check": {
            "constraint": "monotonic_cst=-1 on sprint_speed_gbweak",
            "n_rows_checked": int(len(check_idx)),
            "speed_grid": grid.tolist(),
            "pass": bool(mono_ok),
        },
        "new_artifact": {
            "path": str(NEW_ARTIFACT),
            "sha256": artifact_sha,
            "registry": registry_result,
        },
    }
    with open(out_dir / "speed_xout_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(f"Wrote {out_dir / 'speed_xout_summary.json'}")
    print(json.dumps(summary["holdout_auc"], indent=1))


if __name__ == "__main__":
    main()
