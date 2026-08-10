"""DPI v2 stage 2: park jointly estimated (MixedLM) + final K2 measurement
-- WS3.4 of the 2026-08-10 platform improvement plan (task C1b).

Plan 3.4 (preferred method, adopted): mixed model on xOut-GBM residuals,

    residual ~ (1 | home_park) + (1 | fielding_team_season)

with empirical-Bayes shrinkage (statsmodels MixedLM; crossed random effects
via the single-group variance-components trick).  DPI_v2 = the team-season
BLUP; park falls out as its own estimand (2026 arXiv 2603.21163 does this
simultaneous park+defense estimation).  Park is NOT stacked independently on
batted-ball mix (they confound, plan-cited 0.696): the response already
conditions on EV/LA/spray/bb_type through the xOut model, and park + team
are estimated JOINTLY here.

Design decisions (pre-declared BEFORE any result was computed):

  * Grain: game-level stage-1 residuals (the C1a pitching-strip output
    scale) -- 52,498 team-games, NOT 1.2M BIP rows.  MixedLM at BIP grain
    is intractable (dense n x 360 VC design); the game-level response
    carries the identical team/park information because DPI is additive
    over BIP.  This is the plan's "subsample/aggregate if intractable at
    1.2M rows" branch, resolved by aggregation rather than subsampling.
  * Unweighted game rows (production team-season aggregation is an
    unweighted mean over games; n_bip per game is roughly constant).
  * park proxy = the game's home_team (100% populated; production's own
    park proxy).  Teams that moved parks in-window (TEX/ATL/OAK-ATH) are
    absorbed into one park level per franchise -- documented limitation.
  * TWO pipelines:
      P1 "strip+park" (frozen):   C1a stage-1 residuals (frozen xOut) ->
          MixedLM -> team BLUP.  Isolates the park increment on the C1a
          trajectory.
      P2 "strip+park+speed" (FINAL for K2): game DPI re-scored 2015-2025
          with the registered speed variant v2026.08.10-speed.det
          (hash-verified), same production-parity game aggregation as A5's
          score_game_dpi, staff peripherals joined from the C1a stage-1
          CSV, per-season stage-1 OLS refit, then MixedLM -> team BLUP
          = DPI_v2_final.
  * PRE-COMMITTED PRIMARY K2 NUMBER: partial r(P2 team BLUP, team OAA |
    BABIP-against), pooled 2023-25, from the FULL-WINDOW (2015-2025)
    MixedLM fit.  Sensitivity (reported, not primary): MixedLM refit on
    2023-25 rows only.  This choice is declared here, before any model
    was fit.
  * K2 CI machinery: identical to C1a (Fisher z + pairs cluster bootstrap
    over the 30 franchises, 5,000 draws, seed 42) via
    scripts.dpi_v2_pitching_strip imports.

Read-only vs the DB; writes only results/ outputs.  No model artifact is
created (MixedLM estimates are published as CSVs; the script is the
recipe).

Usage:
    python scripts/dpi_v2_park_mixedlm.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
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
from scripts.dpi_v2_pitching_strip import (  # noqa: E402
    PERIPHERALS, RANKINGS_CSV, k2_block, sha256_of,
)

STAGE1_CSV = (
    REPO_ROOT / "results" / "defensive_pressing" / "v2_2026-08"
    / "game_dpi_v2_stage1_2015_2025.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "defensive_pressing" / "v2_2026-08"
SPEED_MODEL_NAME = "defensive_pressing"
SPEED_MODEL_VERSION = "v2026.08.10-speed.det"
SEASONS = list(range(2015, 2026))
K2_SEASONS = [2023, 2024, 2025]
MIN_BIP_PER_GAME = dp.MIN_BIP_PER_GAME  # 5, production constant


# ---------------------------------------------------------------------------
# P2: game DPI re-scored with the speed variant (production-parity)
# ---------------------------------------------------------------------------

def load_speed_model():
    """Load the registered speed variant, hash-verified against its manifest."""
    import joblib

    reg = ModelRegistry()
    manifest = reg.get_manifest_v2(SPEED_MODEL_NAME, SPEED_MODEL_VERSION)
    path = reg.artifact_path_for(SPEED_MODEL_NAME, SPEED_MODEL_VERSION)
    actual = sha256_of_file(path)
    if actual != manifest["sha256"]:
        raise SystemExit(
            f"speed variant hash mismatch: {actual} != {manifest['sha256']}"
        )
    bundle = joblib.load(path)
    return bundle["model"], bundle["metadata"], {
        "path": str(path), "sha256": actual, "version": SPEED_MODEL_VERSION,
    }


def score_game_dpi_speed(conn, model, meta) -> pd.DataFrame:
    """Per-game per-defending-team DPI scored with the speed variant.

    Parity with scripts/dpi_reliability_2026.py::score_game_dpi (A5), except
    the expectation model is the 5-feature speed variant: same BIP cohort,
    same n_bip >= 5 filter, same round-3 and zero-feature-row conventions.
    """
    from scripts.dpi_v2_speed_xout import pull_sprint_speed

    league_mean = float(meta["speed_feature"]["league_mean_used"])
    feature_cols = list(meta["feature_columns"])
    ss = pull_sprint_speed(conn)
    speed_map = {(int(r.season), int(r.player_id)): float(r.sprint_speed)
                 for r in ss.itertuples()}

    frames = []
    for season in SEASONS:
        df = conn.execute(
            """
            SELECT game_pk, home_team, away_team, inning_topbot, batter_id,
                   launch_speed, launch_angle, hc_x, hc_y, bb_type, events
            FROM pitches
            WHERE type = 'X'
              AND events IS NOT NULL
              AND inning_topbot IN ('Top', 'Bot')
              AND EXTRACT(YEAR FROM game_date) = ?
            """,
            [season],
        ).fetchdf()
        df["team_id"] = np.where(
            df["inning_topbot"] == "Top", df["home_team"], df["away_team"]
        )
        df["is_out"] = dp._is_out(df["events"])
        has_data = (
            df["launch_speed"].notna() & df["launch_angle"].notna()
            & df["hc_x"].notna() & df["hc_y"].notna() & df["bb_type"].notna()
        )
        xout = pd.Series(np.nan, index=df.index)
        if has_data.any():
            sub = df.loc[has_data]
            feats = dp.build_bip_features(
                sub, include_park=False, include_weather=False)
            feats["sprint_speed_gbweak"] = dp.build_sprint_speed_feature(
                sub["batter_id"], pd.Series(season, index=sub.index),
                sub["bb_type"], sub["launch_speed"], speed_map, league_mean,
            )
            xout.loc[has_data] = model.predict_proba(feats[feature_cols])[:, 1]
        df["xout"] = xout

        grp = df.groupby(["game_pk", "team_id"], sort=False).agg(
            n_bip=("events", "size"),
            actual_outs=("is_out", "sum"),
            expected_outs=("xout", "sum"),
            n_feature_rows=("xout", "count"),
        ).reset_index()
        grp = grp[grp["n_bip"] >= MIN_BIP_PER_GAME].copy()
        grp["dpi_speed"] = np.round(grp["actual_outs"] - grp["expected_outs"], 3)
        grp.loc[grp["n_feature_rows"] == 0, "dpi_speed"] = 0.0
        grp["season"] = season
        frames.append(grp)
        print(f"  scored season {season}: {len(grp)} team-games")
    out = pd.concat(frames, ignore_index=True)
    return out[["season", "team_id", "game_pk", "n_bip", "dpi_speed"]]


def stage1_strip(df: pd.DataFrame, dpi_col: str) -> tuple[pd.DataFrame, list[dict]]:
    """Per-season OLS of dpi_col on the C1a staff peripherals; residual out.

    Identical functional form to scripts/dpi_v2_pitching_strip.py stage 1.
    """
    feat_cols = [f"staff_{c}" for c in PERIPHERALS]
    frames, rows = [], []
    for season, sub in df.groupby("season"):
        sub = sub.copy()
        X = np.column_stack(
            [np.ones(len(sub))] + [sub[c].to_numpy(float) for c in feat_cols]
        )
        y = sub[dpi_col].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        ss_res = float(np.sum(resid ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        sub["stage1_resid"] = resid
        frames.append(sub)
        rows.append({"season": int(season), "n_games": int(len(sub)),
                     "r2_game_level": round(1 - ss_res / ss_tot, 5)})
    return pd.concat(frames, ignore_index=True), rows


# ---------------------------------------------------------------------------
# MixedLM: resid ~ (1|park) + (1|team_season), EB shrinkage
# ---------------------------------------------------------------------------

def fit_park_team_mixedlm(df: pd.DataFrame, resid_col: str, label: str) -> dict:
    """Crossed random effects via the single-group VC trick."""
    import statsmodels.formula.api as smf

    data = df[[resid_col, "park", "team_season"]].copy()
    data = data.rename(columns={resid_col: "resid"})
    data["one"] = 1

    model = smf.mixedlm(
        "resid ~ 1", data, groups="one",
        vc_formula={"park": "0 + C(park)", "team_season": "0 + C(team_season)"},
        re_formula="0",
    )
    # Optimizer choice validated on synthetic data before the real fit:
    # lbfgs stopped short of the REML optimum (converged=False, worse llf);
    # bfgs/powell/cg agreed on the optimum and recovered the realized truth.
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        result = model.fit(reml=True, method="bfgs", maxiter=500)
    warn_msgs = sorted({str(w.message) for w in wlist})

    vcomp = dict(zip(model.exog_vc.names, [float(v) for v in result.vcomp]))
    sigma2 = float(result.scale)

    re_series = result.random_effects[1]  # single group labeled 1
    park_blup, team_blup = {}, {}
    pat = re.compile(r"^(park|team_season)\[C\((?:park|team_season)\)\[(.+)\]\]$")
    for name, val in re_series.items():
        m = pat.match(str(name))
        if not m:
            continue
        comp, level = m.group(1), m.group(2)
        if comp == "park":
            park_blup[level] = float(val)
        else:
            team_blup[level] = float(val)

    total_var = sum(vcomp.values()) + sigma2
    return {
        "label": label,
        "n_games": int(len(data)),
        "converged": bool(getattr(result, "converged", True)),
        "warnings": warn_msgs,
        "variance_components": {k: round(v, 6) for k, v in vcomp.items()},
        "residual_variance": round(sigma2, 6),
        "variance_share_game_level": {
            **{k: round(v / total_var, 6) for k, v in vcomp.items()},
            "residual": round(sigma2 / total_var, 6),
        },
        "intercept": round(float(result.params["Intercept"]), 6),
        "park_blup": park_blup,
        "team_season_blup": team_blup,
    }


def blup_to_season_df(team_blup: dict[str, float]) -> pd.DataFrame:
    rows = []
    for key, val in team_blup.items():
        team_id, season = key.rsplit("_", 1)
        rows.append({"team_id": team_id, "season": int(season), "blup": val})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--reuse-speed-scores", action="store_true",
                    help="Reuse <out-dir>/game_dpi_speed_2015_2025.csv")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- inputs ----------------------------------------------------------
    stage1 = pd.read_csv(STAGE1_CSV)
    stage1_sha = sha256_of(STAGE1_CSV)
    print(f"Loaded C1a stage-1 CSV: {len(stage1)} team-games "
          f"(sha256 {stage1_sha[:12]}...)")

    speed_model, speed_meta, speed_prov = load_speed_model()
    print(f"Speed variant verified: {speed_prov['version']} "
          f"(sha256 {speed_prov['sha256'][:12]}...)")

    speed_csv = out_dir / "game_dpi_speed_2015_2025.csv"
    conn = get_connection(read_only=True)
    try:
        park_map = conn.execute(
            "SELECT DISTINCT game_pk, home_team FROM pitches "
            "WHERE EXTRACT(YEAR FROM game_date) BETWEEN 2015 AND 2025"
        ).fetchdf()
        if args.reuse_speed_scores and speed_csv.exists():
            print(f"Reusing {speed_csv}")
            speed_games = pd.read_csv(speed_csv)
        else:
            print("Scoring game DPI 2015-2025 with the speed variant ...")
            speed_games = score_game_dpi_speed(conn, speed_model, speed_meta)
            speed_games.to_csv(speed_csv, index=False)
            print(f"Wrote {speed_csv} ({len(speed_games)} rows)")
    finally:
        conn.close()

    assert not park_map.duplicated("game_pk").any(), "game_pk -> multiple home_team"

    # ---- assemble P1 and P2 game tables ----------------------------------
    def attach(df: pd.DataFrame) -> pd.DataFrame:
        df = df.merge(park_map, on="game_pk", how="left")
        assert df["home_team"].notna().all()
        df = df.rename(columns={"home_team": "park"})
        df["team_season"] = df["team_id"] + "_" + df["season"].astype(str)
        return df

    p1 = attach(stage1.copy())
    p1["stage1_resid"] = p1["dpi_v2_stage1"]

    # P2: speed-scored DPI + C1a staff peripherals -> stage-1 refit
    staff_cols = ["season", "game_pk", "team_id"] + [f"staff_{c}" for c in PERIPHERALS]
    p2 = speed_games.merge(stage1[staff_cols], on=["season", "game_pk", "team_id"],
                           how="inner")
    n_dropped = len(speed_games) - len(p2)
    print(f"P2 join vs C1a staff peripherals: {len(p2)} matched, "
          f"{n_dropped} dropped (no staff row)")
    corr_frozen_speed = float(np.corrcoef(
        p2.merge(stage1[["season", "game_pk", "team_id", "dpi"]],
                 on=["season", "game_pk", "team_id"])[["dpi_speed", "dpi"]]
        .T.to_numpy())[0, 1])
    p2, p2_stage1_rows = stage1_strip(p2, "dpi_speed")
    p2 = attach(p2)

    # ---- MixedLM fits ----------------------------------------------------
    print("\nFitting MixedLM P1 (frozen strip residuals, 2015-2025) ...")
    fit_p1 = fit_park_team_mixedlm(p1, "stage1_resid", "P1_strip_park_frozen")
    print(f"  converged={fit_p1['converged']} "
          f"vc={fit_p1['variance_components']} sigma2={fit_p1['residual_variance']}")

    print("Fitting MixedLM P2 (speed strip residuals, 2015-2025) ...")
    fit_p2 = fit_park_team_mixedlm(p2, "stage1_resid", "P2_final_strip_park_speed")
    print(f"  converged={fit_p2['converged']} "
          f"vc={fit_p2['variance_components']} sigma2={fit_p2['residual_variance']}")

    print("Fitting MixedLM P2-sensitivity (2023-25 rows only) ...")
    fit_p2s = fit_park_team_mixedlm(
        p2[p2["season"].isin(K2_SEASONS)], "stage1_resid",
        "P2_sensitivity_2023_25_fit")
    print(f"  converged={fit_p2s['converged']} "
          f"vc={fit_p2s['variance_components']} sigma2={fit_p2s['residual_variance']}")

    # ---- outputs: park effects + team-season BLUPs -----------------------
    park_rows = []
    for label, fit in [("P1", fit_p1), ("P2", fit_p2), ("P2_2023_25", fit_p2s)]:
        for park, val in sorted(fit["park_blup"].items()):
            park_rows.append({"fit": label, "park": park,
                              "blup_dpi_per_game": round(val, 5)})
    pd.DataFrame(park_rows).to_csv(out_dir / "park_effects_mixedlm.csv", index=False)

    frames = []
    for label, fit in [("v2_strip_park_frozen", fit_p1),
                       ("v2_final_strip_park_speed", fit_p2),
                       ("v2_final_sensitivity_2023_25_fit", fit_p2s)]:
        sdf = blup_to_season_df(fit["team_season_blup"])
        sdf["pipeline"] = label
        frames.append(sdf)
    blup_df = pd.concat(frames, ignore_index=True)
    blup_df["blup"] = blup_df["blup"].round(5)
    blup_df.to_csv(out_dir / "team_season_dpi_v2_blup.csv", index=False)

    # ---- K2: partial r(DPI_v2, OAA | BABIP-against) ----------------------
    recorded = pd.read_csv(RANKINGS_CSV)
    k2_rows = []
    for label, fit in [("v2_strip_park_frozen", fit_p1),
                       ("v2_final_strip_park_speed", fit_p2),
                       ("v2_final_sensitivity_2023_25_fit", fit_p2s)]:
        sdf = blup_to_season_df(fit["team_season_blup"]).rename(
            columns={"blup": "dpi_v2"})
        k2_rows += k2_block(sdf, recorded, "dpi_v2", label,
                            args.seed, args.n_boot)
    k2_df = pd.DataFrame(k2_rows)
    k2_df.to_csv(out_dir / "k2_final_partial_r.csv", index=False)
    print("\nK2 partial r(DPI_v2, OAA | BABIP-against):")
    print(k2_df.to_string(index=False))

    summary = {
        "task": "WS3.4 park (MixedLM, jointly estimated) + final K2 -- C1b",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "stage1_csv": str(STAGE1_CSV), "stage1_csv_sha256": stage1_sha,
            "speed_variant": speed_prov,
            "rankings_csv": str(RANKINGS_CSV),
            "rankings_csv_sha256": sha256_of(RANKINGS_CSV),
        },
        "primary_pre_commitment": (
            "PRIMARY K2 number = v2_final_strip_park_speed pooled_2023_2025 "
            "(full-window 2015-2025 MixedLM fit), declared in the script "
            "docstring before any fit was run. 2023-25-only refit is "
            "sensitivity."
        ),
        "p2_speed_rescore": {
            "n_team_games": int(len(speed_games)),
            "n_matched_staff": int(len(p2)),
            "n_dropped_no_staff": int(n_dropped),
            "r_game_dpi_speed_vs_frozen": round(corr_frozen_speed, 5),
            "stage1_r2_by_season": p2_stage1_rows,
        },
        "mixedlm_fits": {
            "P1_strip_park_frozen": {k: v for k, v in fit_p1.items()
                                     if k not in ("park_blup", "team_season_blup")},
            "P2_final_strip_park_speed": {k: v for k, v in fit_p2.items()
                                          if k not in ("park_blup", "team_season_blup")},
            "P2_sensitivity_2023_25_fit": {k: v for k, v in fit_p2s.items()
                                           if k not in ("park_blup", "team_season_blup")},
        },
        "park_blups": {"P1": fit_p1["park_blup"], "P2": fit_p2["park_blup"],
                       "P2_2023_25": fit_p2s["park_blup"]},
        "k2_final_partial_r": k2_rows,
        "in_sample_oos_status": (
            "DPI game scores 2023-25 are OOS w.r.t. both expectation models' "
            "2015-2022 train windows (speed variant additionally holds out "
            "20% of 2015-2022); 2015-2022 game scores overlap the models' "
            "own train rows (in-sample; they inform the park RE and their "
            "own seasons' BLUPs only). Stage-1 OLS and the MixedLM are "
            "within-window descriptive decompositions, not forecasts."
        ),
        "seed": args.seed, "n_boot": args.n_boot,
    }
    with open(out_dir / "park_mixedlm_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(f"\nWrote {out_dir / 'park_mixedlm_summary.json'}")


if __name__ == "__main__":
    main()
