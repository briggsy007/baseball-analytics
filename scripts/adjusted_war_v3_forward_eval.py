#!/usr/bin/env python
"""WS4.2 / K3 half 1 (2026-08-10): AdjustedWAR v3 ridge vs the current
CausalWAR formulation on SEASON-FORWARD prediction.

PRE-REGISTERED PROTOCOL -- this docstring was written in full before the
evaluation ran.  Nothing below was chosen after seeing a result.  K3
adjudication itself belongs to Batch D; this script only measures.

Held-out seasons (>= 2 per WS4.4): **2024 and 2025** (baselines b = 2023
and b = 2024).  Neither season enters lambda/C selection in any form.

Hyperparameter selection (season-forward, Sill 2010; never random K-fold):
* lambda* for the ridge identity blocks: pairs (b_v -> b_v+1) for
  b_v in 2015..2022 (follow-ups 2016..2023, all strictly before the
  held-out seasons); PA-weighted RMSE, follow-up PA >= 100; grid
  ``DEFAULT_LAM_GRID``.  Full path recorded.
* C* for the shrunk-raw comparator (below): same pairs, same objective,
  grid (100, 200, 300, 450, 600, 800, 1100, 1500, 2000, 3000).

Systems (batter channel; every prediction for follow-up season f = b+1
uses information from seasons <= b only):
* ``ridge_1yr`` (PRIMARY, information parity with the current
  formulation): AdjustedWAR v3 fit on season b alone at lambda*;
  pred = league_b + coef_centered.
* ``current_v1``: the production CausalWAR formulation -- frozen
  2015-2022 v1 nuisance (fully OOS for b in {2023, 2024}), per-batter
  mean residual (pa_min=10); pred = league_b + point_estimate_per_pa.
* ``marcel``: literal Marcel (WS4.3), 3-season history -- the mandatory
  floor.
* ``naive``: constant league_b (baseline-season PA-weighted league wOBA).
* ``shrunk_raw`` (DIAGNOSTIC, the DRC+ shrunk-to-shrunk lesson):
  pred = league_b + (woba_b - league_b) * pa_b / (pa_b + C*).  If ridge
  only matches this, its win over raw wOBA is shrinkage, not adjustment.
  Not a K3 quantity; published anyway.
* ``ridge_3yr`` (SECONDARY, declared now): AdjustedWAR v3 fit on pooled
  seasons {b-2, b-1, b} at lambda* (single coefficient per player) --
  information parity with Marcel rather than with current_v1.

Scoring pool (per follow-up season): batters having (a) actual follow-up
wOBA with follow-up PA >= 100 (primary floor; 300 / 502 / all as labeled
sensitivities), (b) predictions from ridge_1yr, current_v1 AND marcel
(intersection; current_v1 implies baseline pa >= 10).  Weights =
follow-up PA.  wOBA source: Statcast pitch rows (marcel.py loader;
``season_batting_stats.woba`` is 100% NULL).

Reported (WS4.4 protocol, exactly as landing):
* PA-weighted RMSE per system, per season and pooled.
* Head-to-head W-L vs Marcel (0.010 tie band) for ridge_1yr and
  current_v1; PA-weighted paired t.
* DIRECT ridge_1yr vs current_v1: RMSE delta, W-L, paired t -- the K3
  half-1 quantity.
* Diagnostics: identity-block centering offsets; park-skill correlation
  (corr over batters between skill estimate and the batter's primary
  park's raw scoring environment, season b) for ridge_1yr / current_v1 /
  raw wOBA; per-park mean fit residual for the ridge.

Reads the DB strictly read_only; writes only
``results/adjusted_war_v3/forward_eval_2026-08-10/`` and the scratch
extraction cache.  Touches no existing artifact.

Usage:  python scripts/adjusted_war_v3_forward_eval.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics.adjusted_war_v3 import (  # noqa: E402
    DEFAULT_LAM_GRID,
    build_design,
    extract_ridge_pa,
    season_forward_lambda_cv,
)
from src.analytics.causal_war import (  # noqa: E402
    _aggregate_player_effects,
    _build_features,
    _extract_pa_data,
)
from src.analytics.marcel import (  # noqa: E402
    fetch_birthdates,
    head_to_head_vs_marcel,
    league_woba_by_season,
    load_batter_season_inputs,
    pa_weighted_paired_t,
    pa_weighted_rmse,
    project_batters,
    score_forecast_vs_protocol,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("awv3_forward")

SCRATCH = Path(
    "C:/Users/hunte/AppData/Local/Temp/claude/"
    "C--Users-hunte-projects-baseball/"
    "5112f9f4-f7ff-4db3-bd24-90acc5fbef27/scratchpad/ridge_cache"
)
OUTDIR = ROOT / "results" / "adjusted_war_v3" / "forward_eval_2026-08-10"
FROZEN_V1 = ROOT / "models" / "causal_war" / "causal_war_trainsplit_2015_2022.pkl"

CV_PAIRS = [(b, b + 1) for b in range(2015, 2023)]   # follow-ups 2016..2023
HOLDOUTS = [(2023, 2024), (2024, 2025)]              # (b, f)
C_GRID = (100.0, 200.0, 300.0, 450.0, 600.0, 800.0,
          1100.0, 1500.0, 2000.0, 3000.0)
PRIMARY_FLOOR = 100.0
SENSITIVITY_FLOORS = (300.0, 502.0, 0.0)             # 0.0 = all pa_f > 0

_V2_ONLY_COLS = [
    "ump_accuracy_above_x", "ump_favor", "temp_f", "wind_speed_mph",
    "wind_dir_deg", "humidity_pct", "pressure_mb", "roof_status",
]


def current_v1_effects(conn, season: int) -> pd.DataFrame:
    """Frozen v1 nuisance point estimates for one season (batter channel)."""
    art = joblib.load(FROZEN_V1)
    nuisance = art["nuisance_outcome"]
    df = _extract_pa_data(conn, year_range=(season, season))
    df = df.drop(columns=[c for c in _V2_ONLY_COLS if c in df.columns])
    W, Y, bat_ids, dfb = _build_features(df)
    if W.shape[1] != 12:
        raise RuntimeError(f"v1 layout expects 12 confounders, got {W.shape[1]}")
    Y_res = Y - nuisance.predict(W)
    eff = _aggregate_player_effects(Y_res, bat_ids, dfb, pa_min=10)
    rows = [{"player_id": int(pid),
             "point_estimate_per_pa": float(e["point_estimate_per_pa"]),
             "pa_b": int(e["pa"])} for pid, e in eff.items()]
    return pd.DataFrame(rows)


def shrunk_raw_pred(inputs: pd.DataFrame, b: int, lg_b: float,
                    C: float) -> pd.DataFrame:
    base = inputs[inputs["season"] == b][["player_id", "pa", "woba"]].copy()
    base = base.rename(columns={"pa": "pa_b", "woba": "woba_b"})
    base["pred_woba"] = lg_b + (base["woba_b"] - lg_b) * base["pa_b"] / (
        base["pa_b"] + C)
    return base[["player_id", "pred_woba"]]


def shrunk_raw_cv(inputs: pd.DataFrame, league: dict[int, float]) -> dict:
    path = []
    for C in C_GRID:
        per_pair = {}
        for b, f in CV_PAIRS:
            pred = shrunk_raw_pred(inputs, b, league[b], C)
            act = inputs[(inputs["season"] == f) & (inputs["pa"] >= 100.0)]
            m = act.merge(pred, on="player_id", how="inner")
            per_pair[f"{b}->{f}"] = pa_weighted_rmse(
                m["woba"].to_numpy(), m["pred_woba"].to_numpy(),
                m["pa"].to_numpy())
        vals = [v for v in per_pair.values() if v == v]
        path.append({"C": C, "mean_rmse": float(np.mean(vals)),
                     "per_pair": {k: round(v, 6) for k, v in per_pair.items()}})
    best = min(path, key=lambda r: r["mean_rmse"])
    return {"C_star": best["C"], "path": path}


def park_skill_corr(pa_frame: pd.DataFrame, skill: pd.DataFrame,
                    skill_col: str, lg_b: float) -> dict[str, Any]:
    """Corr over batters between a skill estimate and the raw scoring
    environment (mean per-PA wOBA - league) of the batter's primary park."""
    park_env = (pa_frame.groupby("park")["woba_value"].mean() - lg_b)
    primary = (
        pa_frame.groupby(["batter_id", "park"]).size().reset_index(name="n")
        .sort_values(["batter_id", "n"], ascending=[True, False])
        .drop_duplicates("batter_id")
        .rename(columns={"batter_id": "player_id"})
    )
    m = primary.merge(skill, on="player_id", how="inner")
    m["park_env"] = m["park"].map(park_env)
    m = m.dropna(subset=["park_env", skill_col])
    if len(m) < 10:
        return {"r": None, "n": int(len(m))}
    r = float(np.corrcoef(m[skill_col], m["park_env"])[0, 1])
    return {"r": round(r, 4), "n": int(len(m))}


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    conn = get_connection(read_only=True)

    results: dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "pre-registered in scripts/adjusted_war_v3_forward_eval.py "
                    "docstring (written before the evaluation ran)",
        "holdout_seasons": [f for _, f in HOLDOUTS],
        "cv_pairs": [list(p) for p in CV_PAIRS],
    }

    # ---- Inputs (shared with Marcel; real data only) ----------------------
    inputs = load_batter_season_inputs(conn, range(2015, 2026))
    league = league_woba_by_season(inputs)
    results["league_woba_by_season"] = {
        int(k): round(v, 6) for k, v in sorted(league.items())}

    # ---- Ridge designs 2015..2024 ----------------------------------------
    frames = {s: extract_ridge_pa(conn, s, cache_dir=SCRATCH)
              for s in range(2015, 2025)}
    designs = {s: build_design([frames[s]], [s]) for s in frames}

    # ---- lambda* by season-forward CV ------------------------------------
    logger.info("Season-forward lambda CV over %d pairs x %d grid points ...",
                len(CV_PAIRS), len(DEFAULT_LAM_GRID))
    cv = season_forward_lambda_cv(
        designs, inputs, league, CV_PAIRS,
        lam_grid=DEFAULT_LAM_GRID, pa_floor=PRIMARY_FLOOR,
    )
    lam_star = cv["lambda_star"]
    results["lambda_cv"] = cv
    logger.info("lambda* = %s", lam_star)

    # ---- C* for the shrunk-raw comparator --------------------------------
    sr = shrunk_raw_cv(inputs, league)
    results["shrunk_raw_cv"] = sr
    logger.info("C* = %s", sr["C_star"])

    # ---- Marcel projections for the holdout follow-ups -------------------
    bd = fetch_birthdates(sorted(inputs["player_id"].unique().tolist()))
    marcel = {}
    for b, f in HOLDOUTS:
        sub = inputs[inputs["season"].isin([f - 1, f - 2, f - 3])]
        proj = project_batters(sub, season=f, birthdates=bd)
        marcel[f] = proj[["player_id", "proj_woba"]].rename(
            columns={"proj_woba": "pred_woba"})

    # ---- Per-holdout evaluation ------------------------------------------
    per_season: dict[int, dict[str, Any]] = {}
    pooled_rows: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {}

    for b, f in HOLDOUTS:
        lg_b = league[b]
        logger.info("Holdout %d -> %d (league_b=%.4f)", b, f, lg_b)

        fit1 = designs[b].solve(lam_star)
        ridge1 = fit1.batter[["player_id", "coef_centered"]].copy()
        ridge1["pred_woba"] = lg_b + ridge1["coef_centered"]

        pooled_design = build_design(
            [frames[s] for s in (b - 2, b - 1, b)], [b - 2, b - 1, b])
        fit3 = pooled_design.solve(lam_star)
        ridge3 = fit3.batter[["player_id", "coef_centered"]].copy()
        ridge3["pred_woba"] = lg_b + ridge3["coef_centered"]

        cur = current_v1_effects(conn, b)
        cur["pred_woba"] = lg_b + cur["point_estimate_per_pa"]

        srp = shrunk_raw_pred(inputs, b, lg_b, sr["C_star"])

        act = inputs[inputs["season"] == f][["player_id", "pa", "woba"]].copy()
        act = act.rename(columns={"pa": "pa_f", "woba": "woba_f"})

        # Intersection pool: actual + ridge_1yr + current_v1 + marcel.
        pool = (act
                .merge(ridge1[["player_id", "pred_woba"]]
                       .rename(columns={"pred_woba": "pred_ridge_1yr"}),
                       on="player_id", how="inner")
                .merge(cur[["player_id", "pred_woba"]]
                       .rename(columns={"pred_woba": "pred_current_v1"}),
                       on="player_id", how="inner")
                .merge(marcel[f].rename(columns={"pred_woba": "pred_marcel"}),
                       on="player_id", how="inner")
                .merge(ridge3[["player_id", "pred_woba"]]
                       .rename(columns={"pred_woba": "pred_ridge_3yr"}),
                       on="player_id", how="left")
                .merge(srp.rename(columns={"pred_woba": "pred_shrunk_raw"}),
                       on="player_id", how="left"))
        pool["pred_naive"] = lg_b
        pool["followup_season"] = f
        pool.to_csv(OUTDIR / f"predictions_{b}_to_{f}.csv", index=False)
        pooled_rows.append(pool)

        season_out: dict[str, Any] = {
            "baseline": b, "followup": f,
            "n_pool_all": int(len(pool)),
            "oos_note": "current_v1 nuisance trained 2015-2022 (fully OOS); "
                        "ridge fit on baseline-season PAs only; lambda*/C* "
                        "selected on follow-ups <= 2023",
        }
        for floor in (PRIMARY_FLOOR,) + SENSITIVITY_FLOORS:
            sub = pool[pool["pa_f"] >= floor] if floor > 0 else pool[pool["pa_f"] > 0]
            key = f"floor_pa_ge_{int(floor)}" if floor > 0 else "floor_all"
            season_out[key] = {
                "n": int(len(sub)),
                "rmse": {
                    name: round(pa_weighted_rmse(
                        sub["woba_f"].to_numpy(),
                        sub[f"pred_{name}"].to_numpy(),
                        sub["pa_f"].to_numpy()), 6)
                    for name in ("ridge_1yr", "current_v1", "ridge_3yr",
                                 "shrunk_raw", "marcel", "naive")
                },
            }
            if floor == PRIMARY_FLOOR:
                for name in ("ridge_1yr", "current_v1"):
                    season_out[key][f"{name}_vs_marcel"] = {
                        "h2h": head_to_head_vs_marcel(
                            sub[f"pred_{name}"].to_numpy(),
                            sub["pred_marcel"].to_numpy(),
                            sub["woba_f"].to_numpy()),
                        "paired_t": pa_weighted_paired_t(
                            sub[f"pred_{name}"].to_numpy(),
                            sub["pred_marcel"].to_numpy(),
                            sub["woba_f"].to_numpy(),
                            sub["pa_f"].to_numpy()),
                    }
                season_out[key]["ridge_vs_current_direct"] = {
                    "h2h": head_to_head_vs_marcel(
                        sub["pred_ridge_1yr"].to_numpy(),
                        sub["pred_current_v1"].to_numpy(),
                        sub["woba_f"].to_numpy()),
                    "paired_t": pa_weighted_paired_t(
                        sub["pred_ridge_1yr"].to_numpy(),
                        sub["pred_current_v1"].to_numpy(),
                        sub["woba_f"].to_numpy(),
                        sub["pa_f"].to_numpy()),
                    "rmse_delta_ridge_minus_current": round(
                        pa_weighted_rmse(sub["woba_f"].to_numpy(),
                                         sub["pred_ridge_1yr"].to_numpy(),
                                         sub["pa_f"].to_numpy())
                        - pa_weighted_rmse(sub["woba_f"].to_numpy(),
                                           sub["pred_current_v1"].to_numpy(),
                                           sub["pa_f"].to_numpy()), 6),
                }
        per_season[f] = season_out

        # ---- Diagnostics per baseline fit --------------------------------
        pa_b = frames[b]
        resid = designs[b].y_centered - designs[b].X @ np.concatenate([
            fit1.batter["coef"].to_numpy(), fit1.pitcher["coef"].to_numpy(),
            fit1.parkstand["coef"].to_numpy(), fit1.context["coef"].to_numpy(),
        ])
        park_resid = (
            pd.DataFrame({"park": pa_b["park"], "resid": resid})
            .groupby("park")["resid"].mean().round(5)
        )
        diagnostics[f"fit_{b}"] = {
            "lambda": lam_star,
            "centering_offset_batter": round(fit1.centering_offset_batter, 6),
            "centering_offset_pitcher": round(fit1.centering_offset_pitcher, 6),
            "park_mean_residual_max_abs": float(park_resid.abs().max()),
            "park_skill_corr": {
                "ridge_1yr": park_skill_corr(
                    pa_b, fit1.batter[["player_id", "coef_centered"]],
                    "coef_centered", lg_b),
                "current_v1": park_skill_corr(
                    pa_b, cur[["player_id", "point_estimate_per_pa"]],
                    "point_estimate_per_pa", lg_b),
                "raw_woba_reference": park_skill_corr(
                    pa_b,
                    pa_b.groupby("batter_id")["woba_value"].mean()
                    .reset_index()
                    .rename(columns={"batter_id": "player_id",
                                     "woba_value": "raw_skill"}),
                    "raw_skill", lg_b),
            },
        }

    conn.close()

    # ---- Pooled (both holdout seasons) -----------------------------------
    allp = pd.concat(pooled_rows, ignore_index=True)
    prim = allp[allp["pa_f"] >= PRIMARY_FLOOR]
    pooled: dict[str, Any] = {"n": int(len(prim)), "seasons_evaluated": 2}
    for name in ("ridge_1yr", "current_v1", "ridge_3yr", "shrunk_raw",
                 "marcel", "naive"):
        pooled[f"rmse_{name}"] = round(pa_weighted_rmse(
            prim["woba_f"].to_numpy(), prim[f"pred_{name}"].to_numpy(),
            prim["pa_f"].to_numpy()), 6)
    for name in ("ridge_1yr", "current_v1"):
        pooled[f"{name}_protocol_verdict"] = score_forecast_vs_protocol(
            system_name=f"adjusted_war_v3::{name}" if "ridge" in name
            else "causal_war::current_v1",
            actual_woba=prim["woba_f"].to_numpy(),
            system_pred=prim[f"pred_{name}"].to_numpy(),
            marcel_pred=prim["pred_marcel"].to_numpy(),
            naive_pred=prim["pred_naive"].to_numpy(),
            followup_pa=prim["pa_f"].to_numpy(),
            seasons_evaluated=2,
        )
    pooled["ridge_vs_current_direct"] = {
        "h2h": head_to_head_vs_marcel(
            prim["pred_ridge_1yr"].to_numpy(),
            prim["pred_current_v1"].to_numpy(),
            prim["woba_f"].to_numpy()),
        "paired_t": pa_weighted_paired_t(
            prim["pred_ridge_1yr"].to_numpy(),
            prim["pred_current_v1"].to_numpy(),
            prim["woba_f"].to_numpy(), prim["pa_f"].to_numpy()),
        "rmse_delta_ridge_minus_current": round(
            pooled["rmse_ridge_1yr"] - pooled["rmse_current_v1"], 6),
    }

    results["per_season"] = per_season
    results["pooled_primary_floor_100"] = pooled
    results["diagnostics"] = diagnostics

    (OUTDIR / "forward_eval.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", OUTDIR / "forward_eval.json")

    # ---- Console summary --------------------------------------------------
    print("\nK3 half 1 -- season-forward prediction, exactly as it lands")
    print("=" * 84)
    print(f"lambda* = {lam_star}   C*(shrunk_raw) = {sr['C_star']}   "
          f"pool floor: follow-up PA >= {int(PRIMARY_FLOOR)}")
    for f, so in per_season.items():
        k = f"floor_pa_ge_{int(PRIMARY_FLOOR)}"
        r = so[k]["rmse"]
        print(f"  {so['baseline']}->{f}  n={so[k]['n']:4d}  "
              + "  ".join(f"{n}={r[n]:.5f}" for n in
                          ("ridge_1yr", "current_v1", "ridge_3yr",
                           "shrunk_raw", "marcel", "naive")))
    print(f"POOLED n={pooled['n']}: "
          + "  ".join(f"{n}={pooled[f'rmse_{n}']:.5f}" for n in
                      ("ridge_1yr", "current_v1", "ridge_3yr",
                       "shrunk_raw", "marcel", "naive")))
    d = pooled["ridge_vs_current_direct"]
    print(f"ridge vs current: RMSE delta {d['rmse_delta_ridge_minus_current']:+.6f} "
          f"(negative = ridge better); h2h {d['h2h']}; "
          f"paired-t conf {d['paired_t'].get('confidence_system_better')}")
    for name in ("ridge_1yr", "current_v1"):
        v = pooled[f"{name}_protocol_verdict"]
        print(f"{name} vs Marcel: W-L-T "
              f"{v['head_to_head_vs_marcel']['wins']}-"
              f"{v['head_to_head_vs_marcel']['losses']}-"
              f"{v['head_to_head_vs_marcel']['ties']}; "
              f"superiority_claim_allowed={v['superiority_claim_allowed']}")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    sys.exit(main())
