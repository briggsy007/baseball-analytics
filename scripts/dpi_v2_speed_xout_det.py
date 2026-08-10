"""DPI v2 sprint-speed xOut variant, DETERMINISTIC re-run -- WS3.3 (C1a).

Supersedes scripts/dpi_v2_speed_xout.py's first attempt.  Why a re-run:
the first attempt's data pull had no ORDER BY, and DuckDB's parallel scan
returns rows in connection-dependent order, so the stratified 80/20 split
selects different physical rows per run.  The first run's numbers are
internally valid (all models scored on that run's in-memory holdout) but
not reproducible; a follow-up subset-CI attempt (dpi_v2_speed_subset_ci.py)
was INVALID because it scored the persisted variant on a differently-split
test set overlapping the variant's own training rows (leakage).  This
script fixes the defect at the root:

  * deterministic total order: ORDER BY game_pk, at_bat_number (unique on
    the PA-ending BIP cohort; asserted);
  * control (4-feature) and speed variant (5-feature, monotonic_cst=-1)
    trained in the SAME run on the SAME split;
  * overall AND subset AUC deltas with paired-bootstrap CIs from the same
    predictions;
  * test-set predictions persisted (parquet keyed by game_pk,
    at_bat_number) so every number is recomputable without retraining;
  * NEW artifact models/defensive_pressing/xout_v2_speed_det_2026_08_10.pkl
    registered as version v2026.08.10-speed.det (write-once, pinned).
    The first attempt's artifact/version remain in the registry history;
    nothing is overwritten and no alias moves.

Read-only vs the DB.

Usage:
    python scripts/dpi_v2_speed_xout_det.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.db.schema import get_connection  # noqa: E402
from src.analytics import defensive_pressing as dp  # noqa: E402
from src.analytics.registry import ModelRegistry, sha256_of_file  # noqa: E402
from scripts.dpi_v2_speed_xout import (  # noqa: E402
    CORE_FEATURES, SPEED_FEATURE, FROZEN_CHECKPOINT, FROZEN_SHA256,
    TRAIN_SEASONS, DEFAULT_OUT_DIR, pull_sprint_speed, bootstrap_delta_auc,
)

NEW_ARTIFACT = (
    REPO_ROOT / "models" / "defensive_pressing"
    / "xout_v2_speed_det_2026_08_10.pkl"
)
REGISTRY_VERSION = "v2026.08.10-speed.det"


def pull_train_bip_ordered(conn) -> pd.DataFrame:
    seasons = ", ".join(str(s) for s in TRAIN_SEASONS)
    return conn.execute(
        f"""
        SELECT
            CAST(EXTRACT(YEAR FROM game_date) AS INT) AS season,
            game_pk, at_bat_number,
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
        ORDER BY game_pk, at_bat_number
        """
    ).fetchdf()


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
        raise SystemExit(f"{NEW_ARTIFACT} already exists -- refusing to overwrite.")

    actual = sha256_of_file(FROZEN_CHECKPOINT)
    if actual != FROZEN_SHA256:
        raise SystemExit(f"Frozen checkpoint hash mismatch: {actual}")
    frozen_bundle = joblib.load(FROZEN_CHECKPOINT)
    frozen_model = frozen_bundle["model"]
    frozen_meta = frozen_bundle.get("metadata", {})
    print(f"Frozen model verified (sha256 {actual[:12]}...)")

    conn = get_connection(read_only=True)
    try:
        print("Pulling 2015-2022 BIP cohort, deterministic order ...")
        df = pull_train_bip_ordered(conn)
        ss = pull_sprint_speed(conn)
    finally:
        conn.close()
    print(f"  {len(df)} BIP rows, {len(ss)} speed player-seasons")

    dupes = df.duplicated(subset=["game_pk", "at_bat_number"]).sum()
    assert dupes == 0, f"(game_pk, at_bat_number) not unique: {dupes} dupes"

    target = dp._is_out(df["events"])
    features_core = dp.build_bip_features(df, include_park=False, include_weather=False)
    mask = features_core[CORE_FEATURES].notna().all(axis=1)
    df = df.loc[mask].reset_index(drop=True)
    features_core = features_core.loc[mask].reset_index(drop=True)
    target = target.loc[mask].reset_index(drop=True)
    n_samples = len(df)
    print(f"  {n_samples} clean rows")

    speed_map = {(int(r.season), int(r.player_id)): float(r.sprint_speed)
                 for r in ss.itertuples()}
    applicable = dp.sprint_speed_applicable_mask(df["bb_type"], df["launch_speed"])
    keys = list(zip(df["season"].astype(int), df["batter_id"].astype(int)))
    matched_speed = pd.Series(
        [speed_map.get(k, np.nan) for k in keys], index=df.index, dtype=float
    )
    league_mean = float(matched_speed[applicable & matched_speed.notna()].mean())
    print(f"League mean sprint speed (BIP-weighted, matched applicable): "
          f"{league_mean:.4f} ft/s")

    features_speed = features_core.copy()
    features_speed[SPEED_FEATURE] = dp.build_sprint_speed_feature(
        df["batter_id"], df["season"], df["bb_type"], df["launch_speed"],
        speed_map, league_mean,
    )

    cfg = dp.DPIConfig()
    train_idx, test_idx = train_test_split(
        np.arange(n_samples), test_size=0.2,
        random_state=cfg.random_state, stratify=target,
    )
    y_train = target.iloc[train_idx].to_numpy()
    y_test = target.iloc[test_idx].to_numpy()

    def fit_hgb(X_train, monotonic_cst=None):
        m = HistGradientBoostingClassifier(
            max_iter=cfg.xout_n_estimators, max_depth=cfg.xout_max_depth,
            learning_rate=cfg.xout_learning_rate, random_state=cfg.random_state,
            monotonic_cst=monotonic_cst,
        )
        m.fit(X_train, y_train)
        return m

    print("Training 4-feature control ...")
    control = fit_hgb(features_core.iloc[train_idx])
    print("Training 5-feature speed variant (monotonic_cst=-1) ...")
    mono = [0, 0, 0, 0, -1]
    variant = fit_hgb(features_speed.iloc[train_idx], monotonic_cst=mono)

    frozen_cols = list(frozen_meta.get("feature_columns") or CORE_FEATURES)
    Xc_test = features_core.iloc[test_idx]
    Xs_test = features_speed.iloc[test_idx]
    p_frozen = frozen_model.predict_proba(Xc_test[frozen_cols])[:, 1]
    p_control = control.predict_proba(Xc_test)[:, 1]
    p_variant = variant.predict_proba(Xs_test)[:, 1]

    auc = {
        "frozen_as_loaded": float(roc_auc_score(y_test, p_frozen)),
        "control_4feat": float(roc_auc_score(y_test, p_control)),
        "variant_5feat_speed": float(roc_auc_score(y_test, p_variant)),
    }
    app_test = applicable.iloc[test_idx].to_numpy()

    deltas = {}
    specs = [
        ("overall_variant_minus_control", None, p_control, p_variant),
        ("overall_variant_minus_frozen", None, p_frozen, p_variant),
        ("applicable_variant_minus_control", app_test, p_control, p_variant),
        ("not_applicable_variant_minus_control", ~app_test, p_control, p_variant),
    ]
    for name, m, pa, pb in specs:
        if m is None:
            yb, pab, pbb = y_test, pa, pb
        else:
            yb, pab, pbb = y_test[m], pa[m], pb[m]
        boot = bootstrap_delta_auc(yb, pab, pbb, args.n_boot,
                                   np.random.default_rng(args.seed))
        deltas[name] = {
            "n": int(len(yb)),
            "delta": round(float(roc_auc_score(yb, pbb) - roc_auc_score(yb, pab)), 5),
            "ci95": [boot["delta_ci_lo"], boot["delta_ci_hi"]],
        }
        print(name, deltas[name])

    subset_auc = {}
    for name, m in [("applicable_gb_weak", app_test), ("not_applicable", ~app_test)]:
        subset_auc[name] = {
            "n": int(m.sum()),
            "out_rate": round(float(y_test[m].mean()), 4),
            "auc_frozen": round(float(roc_auc_score(y_test[m], p_frozen[m])), 5),
            "auc_control": round(float(roc_auc_score(y_test[m], p_control[m])), 5),
            "auc_variant": round(float(roc_auc_score(y_test[m], p_variant[m])), 5),
        }

    # Monotonicity verification
    rng2 = np.random.default_rng(args.seed)
    check_idx = rng2.choice(np.flatnonzero(app_test),
                            size=min(2000, int(app_test.sum())), replace=False)
    grid = np.array([23.0, 24.5, 26.0, 27.5, 29.0, 30.5])
    base_rows = Xs_test.iloc[check_idx].copy()
    prev, mono_ok = None, True
    for g in grid:
        rows = base_rows.copy()
        rows[SPEED_FEATURE] = g
        p = variant.predict_proba(rows)[:, 1]
        if prev is not None and np.any(p > prev + 1e-9):
            mono_ok = False
        prev = p
    print(f"Monotonicity check: {'PASS' if mono_ok else 'FAIL'}")

    # Persist test predictions for exact recomputability.
    preds = pd.DataFrame({
        "game_pk": df["game_pk"].iloc[test_idx].to_numpy(),
        "at_bat_number": df["at_bat_number"].iloc[test_idx].to_numpy(),
        "season": df["season"].iloc[test_idx].to_numpy(),
        "y_out": y_test,
        "applicable_gb_weak": app_test,
        "p_frozen": p_frozen, "p_control": p_control, "p_variant": p_variant,
    })
    preds_path = out_dir / "speed_xout_det_test_predictions.parquet"
    preds.to_parquet(preds_path, index=False)
    print(f"Persisted test predictions: {preds_path}")

    metadata = {
        "variant": "xout_v2_speed_det (WS3.3, experimental, deterministic rerun)",
        "supersedes": "v2026.08.10-speed (row-order-nondeterministic split)",
        "train_seasons": TRAIN_SEASONS,
        "row_order": "ORDER BY game_pk, at_bat_number (unique, asserted)",
        "n_samples": int(n_samples),
        "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "auc": round(auc["variant_5feat_speed"], 5),
        "out_rate": round(float(target.mean()), 4),
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": CORE_FEATURES + [SPEED_FEATURE],
        "monotonic_cst": mono,
        "speed_feature": {
            "definition": "batter same-season Savant sprint speed on "
                          "grounders OR weak contact "
                          f"(EV < {dp.SPRINT_SPEED_WEAK_EV_MAX}); league-mean "
                          "neutral fill / imputation",
            "league_mean_used": round(league_mean, 4),
        },
        "config": {
            "xout_n_estimators": cfg.xout_n_estimators,
            "xout_max_depth": cfg.xout_max_depth,
            "xout_learning_rate": cfg.xout_learning_rate,
            "random_state": cfg.random_state,
        },
        "use_park": False, "use_weather": False,
        "sklearn_version": sklearn.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
    NEW_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": variant, "metadata": metadata}, NEW_ARTIFACT)
    artifact_sha = sha256_of_file(NEW_ARTIFACT)
    print(f"Persisted {NEW_ARTIFACT} (sha256 {artifact_sha[:12]}...)")

    registry_result = None
    if not args.skip_registry:
        reg = ModelRegistry()
        try:
            manifest = reg.register_version(
                "defensive_pressing", REGISTRY_VERSION,
                artifact=NEW_ARTIFACT, sha256=artifact_sha,
                hash_policy="pinned", train_window="2015-2022",
                data_snapshot={"tables": {
                    "pitches": {"train_bip_rows": int(n_samples)},
                    "sprint_speed": {"player_season_rows": int(len(ss))},
                }},
                training_script="scripts/dpi_v2_speed_xout_det.py",
                spec_version="docs/plans/2026-08-10_platform_improvement_plan.md WS3.3",
                validation_results_ref="results/defensive_pressing/v2_2026-08/speed_xout_det_summary.json",
                notes="EXPERIMENTAL deterministic rerun; supersedes "
                      "v2026.08.10-speed (nondeterministic row order in its "
                      "split). Aliases untouched.",
            )
            registry_result = {"registered": True, "version": REGISTRY_VERSION,
                               "sha256": manifest["sha256"]}
        except FileExistsError as exc:
            registry_result = {"registered": False, "note": str(exc)}
        print(f"Registry: {registry_result}")

    summary = {
        "task": "WS3.3 sprint-speed xOut variant, deterministic rerun -- C1a",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "why_rerun": "first attempt's pull had no ORDER BY; DuckDB scan "
                     "order is connection-dependent, so its 80/20 split is "
                     "not reproducible. This run is the canonical WS3.3 "
                     "measurement; the first attempt is reported alongside "
                     "as attempt #1 (internally valid, not reproducible).",
        "frozen_checkpoint": {"path": str(FROZEN_CHECKPOINT), "sha256": actual,
                              "recorded_holdout_auc": frozen_meta.get("auc")},
        "environment": {"python": sys.version.split()[0],
                        "sklearn": sklearn.__version__,
                        "omp_num_threads": os.environ.get("OMP_NUM_THREADS")},
        "cohort": {
            "train_window": "2015-2022",
            "row_order": "ORDER BY game_pk, at_bat_number (unique, asserted)",
            "n_samples": int(n_samples),
            "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
            "split": "stratified 80/20, random_state 42 (frozen recipe)",
            "note": "918,292 feature-complete rows today vs 837,571 at the "
                    "2026-04-18 freeze (interim backfills); frozen model's "
                    "recorded 0.8936 AUC belongs to the April snapshot",
        },
        "league_mean_sprint_speed": round(league_mean, 4),
        "holdout_auc": {k: round(v, 5) for k, v in auc.items()},
        "auc_deltas": deltas,
        "subset_auc": subset_auc,
        "monotonicity_check": {"n_rows_checked": int(len(check_idx)),
                               "speed_grid": grid.tolist(),
                               "pass": bool(mono_ok)},
        "test_predictions": str(preds_path),
        "new_artifact": {"path": str(NEW_ARTIFACT), "sha256": artifact_sha,
                         "registry": registry_result},
        "in_sample_oos_status": "holdout within the 2015-2022 train window "
                                "(frozen recipe convention), not "
                                "future-season OOS; frozen-as-loaded row may "
                                "overlap its own April train split "
                                "(reference only)",
    }
    with open(out_dir / "speed_xout_det_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(f"Wrote {out_dir / 'speed_xout_det_summary.json'}")


if __name__ == "__main__":
    main()
