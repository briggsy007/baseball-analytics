"""DPI v2 positioning-thesis test (xOut-A vs xOut-B) -- WS3.5 of the
2026-08-10 platform improvement plan (task C1b).

THE MODEL'S NAME IS ON THE LINE (audit DPI finding 8: the
"pressing"/positioning framing was never tested against the
if_fielding_alignment / of_fielding_alignment columns).  DRS-PART blueprint
(SIS 2020: Positioning = team credit, Range = player credit).

Design (pre-declared BEFORE any result was computed):

  * xOut-A: ball features only -- the frozen recipe's 4 features
    (launch_speed, launch_angle, spray_angle, bb_type_encoded).
  * xOut-B: A + alignment block:
      - stand_R           (1 = RHB, 0 = LHB; batter stand)
      - if_shift, if_shade, if_strategic   (one-hot vs Standard baseline;
        NaN where if_fielding_alignment is NULL -- HistGB routes missing)
      - of_strategic, of_other             (of_other = "4th outfielder" OR
        "Extreme outfield shift", 742 rows league-wide; NaN where NULL)
      - pull_spray = spray_angle * (+1 if LHB else -1)  -- degrees toward
        the batter's pull side (stand x spray interaction; positive spray
        is toward RF, RHB pull = LF)
      - if_shift_x_pull, if_shade_x_pull, if_strategic_x_pull,
        of_strategic_x_pull                (alignment x spray interactions)
  * Era split EXACTLY at 2023-03-30 (shift ban).  DB check: earliest 2023
    game is 2023-04-01, so pre-ban = 2015-2022, post-ban = 2023-2025.
    2026 (in-season, lockbox year for other lanes) is EXCLUDED.
  * Same train/holdout protocol for both models, per era: deterministic
    total row order (ORDER BY game_pk, at_bat_number; uniqueness asserted),
    stratified 80/20 split, seed 42, frozen-recipe HistGB
    (200 iter / depth 6 / lr 0.05).  CPU only, OMP_NUM_THREADS=6.
  * K1 quantities (kill criterion, plan section 8 -- MEASURED here,
    adjudicated in Batch D):
      - holdout AUC delta (B minus A) per era, paired bootstrap CI
        (1,000 draws, seed 42);
      - per-team positioning value = mean(P_B - P_A) over the defending
        team's BIP rows per team-season (ALL era rows, train+test --
        descriptive, partially in-sample; documented);
      - split-half reliability of the per-team positioning value:
        game_pk % 2 halves within team-season, per-season cross-team
        Pearson r, Spearman-Brown corrected; era alpha = Fisher-z mean of
        the per-season Spearman-Brown values (A5/WS3.1 convention).
        Sensitivity: era-pooled season-centered variant.

Artifacts: the four era models are persisted together in ONE new bundle
(models/defensive_pressing/xout_v2_alignment_ab_2026_08_10.pkl) and
registered once as experimental version v2026.08.10-alignment-ab (pinned,
write-once).  NO alias changes.  Read-only vs the DB.

Usage:
    python scripts/dpi_v2_alignment_ab.py
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
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.db.schema import get_connection  # noqa: E402
from src.analytics import defensive_pressing as dp  # noqa: E402
from src.analytics.registry import ModelRegistry, sha256_of_file  # noqa: E402
from scripts.dpi_v2_speed_xout import bootstrap_delta_auc  # noqa: E402

NEW_ARTIFACT = (
    REPO_ROOT / "models" / "defensive_pressing"
    / "xout_v2_alignment_ab_2026_08_10.pkl"
)
REGISTRY_VERSION = "v2026.08.10-alignment-ab"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "defensive_pressing" / "v2_2026-08"

ERA_SPLIT_DATE = "2023-03-30"  # shift ban -- plan 3.5, exact
CORE_FEATURES = ["launch_speed", "launch_angle", "spray_angle", "bb_type_encoded"]
ALIGN_FEATURES = [
    "stand_R",
    "if_shift", "if_shade", "if_strategic",
    "of_strategic", "of_other",
    "pull_spray",
    "if_shift_x_pull", "if_shade_x_pull", "if_strategic_x_pull",
    "of_strategic_x_pull",
]


def pull_bip(conn) -> pd.DataFrame:
    """All core-complete BIP rows 2015-2025, deterministic order."""
    return conn.execute(
        """
        SELECT
            CAST(EXTRACT(YEAR FROM game_date) AS INT) AS season,
            game_date, game_pk, at_bat_number,
            home_team, away_team, inning_topbot, stand,
            if_fielding_alignment, of_fielding_alignment,
            launch_speed, launch_angle, hc_x, hc_y, bb_type, events
        FROM pitches
        WHERE type = 'X'
          AND launch_speed IS NOT NULL
          AND launch_angle IS NOT NULL
          AND hc_x IS NOT NULL
          AND hc_y IS NOT NULL
          AND bb_type IS NOT NULL
          AND events IS NOT NULL
          AND EXTRACT(YEAR FROM game_date) BETWEEN 2015 AND 2025
        ORDER BY game_pk, at_bat_number
        """
    ).fetchdf()


def build_alignment_block(df: pd.DataFrame, spray: pd.Series) -> pd.DataFrame:
    """The pre-declared xOut-B alignment feature block."""
    out = pd.DataFrame(index=df.index)
    stand = df["stand"].astype(str)
    out["stand_R"] = (stand == "R").astype(float)

    if_al = df["if_fielding_alignment"]
    if_known = if_al.notna()
    out["if_shift"] = np.where(if_known, (if_al == "Infield shift").astype(float), np.nan)
    out["if_shade"] = np.where(if_known, (if_al == "Infield shade").astype(float), np.nan)
    out["if_strategic"] = np.where(if_known, (if_al == "Strategic").astype(float), np.nan)

    of_al = df["of_fielding_alignment"]
    of_known = of_al.notna()
    out["of_strategic"] = np.where(of_known, (of_al == "Strategic").astype(float), np.nan)
    out["of_other"] = np.where(
        of_known,
        of_al.isin(["4th outfielder", "Extreme outfield shift"]).astype(float),
        np.nan,
    )

    # positive spray = toward RF; RHB pull side = LF -> pull_spray =
    # spray * (+1 for LHB, -1 for RHB): degrees toward the batter's pull side.
    pull_sign = np.where(stand == "L", 1.0, -1.0)
    out["pull_spray"] = spray.to_numpy(float) * pull_sign

    for col in ("if_shift", "if_shade", "if_strategic", "of_strategic"):
        out[f"{col}_x_pull"] = out[col] * out["pull_spray"]
    return out


def spearman_brown(r_half: float) -> float:
    return 2 * r_half / (1 + r_half) if r_half > -1 else np.nan


def fisher_z_mean(vals: list[float]) -> float:
    z = np.arctanh(np.clip(np.asarray(vals, dtype=float), -0.999999, 0.999999))
    return float(np.tanh(np.nanmean(z)))


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

    conn = get_connection(read_only=True)
    try:
        print("Pulling 2015-2025 BIP cohort (read-only, deterministic order) ...")
        df = pull_bip(conn)
    finally:
        conn.close()
    print(f"  {len(df)} BIP rows")

    dupes = int(df.duplicated(subset=["game_pk", "at_bat_number"]).sum())
    assert dupes == 0, f"(game_pk, at_bat_number) not unique: {dupes} dupes"

    n_bad_topbot = int((~df["inning_topbot"].isin(["Top", "Bot"])).sum())
    df["team_id"] = np.where(
        df["inning_topbot"] == "Top", df["home_team"], df["away_team"]
    )

    # Era assignment -- EXACTLY at 2023-03-30.
    game_date = pd.to_datetime(df["game_date"])
    df["era"] = np.where(
        game_date < pd.Timestamp(ERA_SPLIT_DATE), "pre_ban", "post_ban"
    )

    target_all = dp._is_out(df["events"])
    feats_core = dp.build_bip_features(df, include_park=False, include_weather=False)
    align_block = build_alignment_block(df, feats_core["spray_angle"])
    feats_b_all = pd.concat([feats_core, align_block], axis=1)

    cfg = dp.DPIConfig()

    def fit_hgb(X_train, y_train):
        m = HistGradientBoostingClassifier(
            max_iter=cfg.xout_n_estimators, max_depth=cfg.xout_max_depth,
            learning_rate=cfg.xout_learning_rate, random_state=cfg.random_state,
        )
        m.fit(X_train, y_train)
        return m

    era_results: dict[str, dict] = {}
    models: dict[str, object] = {}
    pos_frames = []

    for era in ("pre_ban", "post_ban"):
        idx_era = np.flatnonzero((df["era"] == era).to_numpy())
        sub = df.iloc[idx_era]
        Xa = feats_core.iloc[idx_era][CORE_FEATURES].reset_index(drop=True)
        Xb = feats_b_all.iloc[idx_era][CORE_FEATURES + ALIGN_FEATURES].reset_index(drop=True)
        y = target_all.iloc[idx_era].reset_index(drop=True)
        n = len(sub)
        seasons_in_era = sorted(sub["season"].unique().tolist())
        print(f"\n=== era {era}: {n} rows, seasons {seasons_in_era} ===")

        align_missing = float(sub["if_fielding_alignment"].isna().mean())
        train_idx, test_idx = train_test_split(
            np.arange(n), test_size=0.2,
            random_state=cfg.random_state, stratify=y,
        )
        y_tr = y.iloc[train_idx].to_numpy()
        y_te = y.iloc[test_idx].to_numpy()

        print(f"  training xOut-A ({len(CORE_FEATURES)} features) ...")
        model_a = fit_hgb(Xa.iloc[train_idx], y_tr)
        print(f"  training xOut-B ({len(CORE_FEATURES) + len(ALIGN_FEATURES)} features) ...")
        model_b = fit_hgb(Xb.iloc[train_idx], y_tr)
        models[f"{era}_A"] = model_a
        models[f"{era}_B"] = model_b

        pa_te = model_a.predict_proba(Xa.iloc[test_idx])[:, 1]
        pb_te = model_b.predict_proba(Xb.iloc[test_idx])[:, 1]
        auc_a = float(roc_auc_score(y_te, pa_te))
        auc_b = float(roc_auc_score(y_te, pb_te))
        boot = bootstrap_delta_auc(
            y_te, pa_te, pb_te, args.n_boot, np.random.default_rng(args.seed)
        )
        print(f"  holdout AUC: A={auc_a:.5f} B={auc_b:.5f} "
              f"delta={auc_b - auc_a:+.5f} CI[{boot['delta_ci_lo']}, {boot['delta_ci_hi']}]")

        # persist test predictions for recomputability
        preds = pd.DataFrame({
            "game_pk": sub["game_pk"].iloc[test_idx].to_numpy(),
            "at_bat_number": sub["at_bat_number"].iloc[test_idx].to_numpy(),
            "season": sub["season"].iloc[test_idx].to_numpy(),
            "y_out": y_te, "p_a": pa_te, "p_b": pb_te,
        })
        preds_path = out_dir / f"alignment_ab_test_predictions_{era}.parquet"
        preds.to_parquet(preds_path, index=False)

        # positioning value: ALL era rows scored with the era models
        pa_all = model_a.predict_proba(Xa)[:, 1]
        pb_all = model_b.predict_proba(Xb)[:, 1]
        pos = pd.DataFrame({
            "era": era,
            "season": sub["season"].to_numpy(),
            "team_id": sub["team_id"].to_numpy(),
            "game_pk": sub["game_pk"].to_numpy(),
            "diff": pb_all - pa_all,
        })
        pos = pos[sub["inning_topbot"].isin(["Top", "Bot"]).to_numpy()]
        pos_frames.append(pos)

        era_results[era] = {
            "n_rows": int(n),
            "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
            "seasons": seasons_in_era,
            "pct_if_alignment_missing": round(100 * align_missing, 3),
            "holdout_auc_A": round(auc_a, 5),
            "holdout_auc_B": round(auc_b, 5),
            "delta_B_minus_A": round(auc_b - auc_a, 5),
            "delta_ci95": [boot["delta_ci_lo"], boot["delta_ci_hi"]],
            "n_boot": args.n_boot,
            "test_predictions": str(preds_path),
        }

    # ---- positioning value per team-season + split-half ------------------
    pos_all = pd.concat(pos_frames, ignore_index=True)
    pos_all["half"] = (pos_all["game_pk"].astype(np.int64) % 2).map(
        {0: "even", 1: "odd"})

    team_season = pos_all.groupby(["era", "season", "team_id"]).agg(
        positioning_value=("diff", "mean"),
        n_bip=("diff", "size"),
    ).reset_index()
    team_season.to_csv(out_dir / "positioning_value_team_season.csv", index=False)

    halves = pos_all.groupby(["era", "season", "team_id", "half"]).agg(
        pv_half=("diff", "mean"), n_bip=("diff", "size"),
    ).reset_index()
    halves.to_csv(out_dir / "positioning_half_aggregates.csv", index=False)

    wide = halves.pivot_table(
        index=["era", "season", "team_id"], columns="half", values="pv_half",
    ).reset_index()

    sh_rows = []
    for (era, season), grp in wide.groupby(["era", "season"]):
        grp = grp.dropna(subset=["even", "odd"])
        r_half = float(stats.pearsonr(grp["even"], grp["odd"])[0])
        sh_rows.append({
            "era": era, "season": int(season), "n_teams": int(len(grp)),
            "split_half_r": round(r_half, 4),
            "spearman_brown": round(spearman_brown(r_half), 4),
        })
    sh_df = pd.DataFrame(sh_rows).sort_values(["era", "season"])
    sh_df.to_csv(out_dir / "positioning_split_half.csv", index=False)
    print("\nPositioning split-half per season:")
    print(sh_df.to_string(index=False))

    alphas = {}
    for era in ("pre_ban", "post_ban"):
        vals = sh_df[sh_df["era"] == era]["spearman_brown"].tolist()
        alphas[era] = {
            "alpha_fisher_mean_spearman_brown": round(fisher_z_mean(vals), 4),
            "per_season_spearman_brown": vals,
            "n_seasons": len(vals),
        }
        # sensitivity: era-pooled, season-centered halves
        w = wide[wide["era"] == era].dropna(subset=["even", "odd"]).copy()
        for col in ("even", "odd"):
            w[col] = w[col] - w.groupby("season")[col].transform("mean")
        r_pooled = float(stats.pearsonr(w["even"], w["odd"])[0])
        alphas[era]["pooled_season_centered_split_half_r"] = round(r_pooled, 4)
        alphas[era]["pooled_season_centered_spearman_brown"] = round(
            spearman_brown(r_pooled), 4)
        alphas[era]["n_team_seasons_pooled"] = int(len(w))

    # positioning-value scale, for the record
    pv_scale = team_season.groupby("era")["positioning_value"].agg(
        ["mean", "std", "min", "max"]).round(5).to_dict(orient="index")

    # ---- persist model bundle + register ---------------------------------
    metadata = {
        "variant": "xout_v2 alignment A/B bundle (WS3.5, experimental probe)",
        "era_split_date": ERA_SPLIT_DATE,
        "models": {
            k: {"features": CORE_FEATURES if k.endswith("_A")
                else CORE_FEATURES + ALIGN_FEATURES}
            for k in models
        },
        "train_protocol": "per era: deterministic ORDER BY game_pk, "
                          "at_bat_number; stratified 80/20 seed 42; frozen "
                          "recipe HistGB 200/6/0.05",
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "sklearn_version": sklearn.__version__,
        "era_results": era_results,
        "config": {
            "xout_n_estimators": cfg.xout_n_estimators,
            "xout_max_depth": cfg.xout_max_depth,
            "xout_learning_rate": cfg.xout_learning_rate,
            "random_state": cfg.random_state,
        },
    }
    NEW_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "metadata": metadata}, NEW_ARTIFACT)
    artifact_sha = sha256_of_file(NEW_ARTIFACT)
    print(f"\nPersisted {NEW_ARTIFACT} (sha256 {artifact_sha[:12]}...)")

    registry_result = None
    if not args.skip_registry:
        reg = ModelRegistry()
        try:
            manifest = reg.register_version(
                "defensive_pressing", REGISTRY_VERSION,
                artifact=NEW_ARTIFACT, sha256=artifact_sha,
                hash_policy="pinned",
                train_window="pre_ban 2015..2023-03-29 / post_ban 2023-03-30..2025",
                data_snapshot={"tables": {"pitches": {
                    "bip_rows_2015_2025": int(len(df)),
                }}},
                training_script="scripts/dpi_v2_alignment_ab.py",
                spec_version="docs/plans/2026-08-10_platform_improvement_plan.md WS3.5",
                validation_results_ref="results/defensive_pressing/v2_2026-08/alignment_ab_summary.json",
                notes="EXPERIMENTAL WS3.5 positioning-thesis probe: 4 era "
                      "models (pre/post x A/B) in one bundle. K1 kill "
                      "criterion measurement artifact. Not production; "
                      "aliases untouched.",
            )
            registry_result = {"registered": True, "version": REGISTRY_VERSION,
                               "sha256": manifest["sha256"]}
        except FileExistsError as exc:
            registry_result = {"registered": False, "note": str(exc)}
        print(f"Registry: {registry_result}")

    summary = {
        "task": "WS3.5 positioning thesis (xOut-A vs xOut-B) -- C1b",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "era_split_date": ERA_SPLIT_DATE,
        "db_era_check": "earliest 2023 game in DB is 2023-04-01; pre_ban = "
                        "2015-2022, post_ban = 2023-2025; 2026 excluded "
                        "(in-season lockbox)",
        "n_rows_topbot_invalid": n_bad_topbot,
        "feature_sets": {"A": CORE_FEATURES,
                         "B": CORE_FEATURES + ALIGN_FEATURES},
        "environment": {"python": sys.version.split()[0],
                        "sklearn": sklearn.__version__,
                        "omp_num_threads": os.environ.get("OMP_NUM_THREADS")},
        "seed": args.seed,
        "k1_auc_deltas": {
            era: {
                "delta_B_minus_A": era_results[era]["delta_B_minus_A"],
                "ci95": era_results[era]["delta_ci95"],
                "auc_A": era_results[era]["holdout_auc_A"],
                "auc_B": era_results[era]["holdout_auc_B"],
                "n_test": era_results[era]["n_test"],
            } for era in era_results
        },
        "k1_split_half_alpha": alphas,
        "positioning_value_scale_per_bip": pv_scale,
        "era_results": era_results,
        "in_sample_oos_status": {
            "auc": "per-era internal 20% holdout (frozen recipe convention), "
                   "not future-season OOS",
            "positioning_value": "descriptive: ALL era rows scored with "
                                 "era models (train rows included -- "
                                 "partially in-sample); split-half operates "
                                 "on these descriptive values",
        },
        "new_artifact": {"path": str(NEW_ARTIFACT), "sha256": artifact_sha,
                         "registry": registry_result},
    }
    with open(out_dir / "alignment_ab_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(f"Wrote {out_dir / 'alignment_ab_summary.json'}")
    print("\nK1 quantities:")
    print(json.dumps({"k1_auc_deltas": summary["k1_auc_deltas"],
                      "k1_split_half_alpha": alphas}, indent=1))


if __name__ == "__main__":
    main()
