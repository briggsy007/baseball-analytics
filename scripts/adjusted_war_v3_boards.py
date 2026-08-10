#!/usr/bin/env python
"""WS4.2 / K3 half 2 (2026-08-10): rebuild EVERY WS4.5 backfill window's
contrarian boards from AdjustedWAR v3 ridge scores and measure the same
fully-OOS lifts (vs ITT-consistent matched-naive AND Marcel) that the
current formulation was measured on.

PRE-REGISTERED PROTOCOL -- written in full before any ridge board was
scored.  Everything except the ranking score is IDENTICAL to the WS4.5
current-formulation run (scripts/causal_war_backfill_windows.py):

* Same windows: single-season b in 2016..2024 (f = b+1), 2-yr-aggregate
  baselines {b-1, b} for b in 2017..2024.
* Same per-window qualified pools -- loaded verbatim from the WS4.5 pool
  parquets (``pool_single_<b>.parquet`` / ``pool_twoyr_*.parquet``), so
  the two formulations rank the exact same players.
* Same hit rules (``row_hit_verdict`` single-season; frozen spec 5.5.3
  half-of-aggregate rule 2-yr), same ITT accounting (spec 5.3: Buy-Low
  exit = MISS, Over-Valued exit = VOID), same matched-naive construction
  (ITT-consistent, +/-0.3 / +/-0.6 WAR, >= 3 neighbors), same Marcel
  boards (loaded from the WS4.5 ``marcel_proj_<f>.parquet`` dumps).
* Ridge score: AdjustedWAR v3 fit at the board's baseline --
  single-season config fits season b alone; 2-yr config fits the pooled
  baseline pair {b-1, b} (one coefficient per player).  Batters ranked
  by batter value, pitchers by pitcher value (``adjusted_war_v3``
  column; ranks are shift/scale-invariant).
* lambda per window (pre-registered EXPANDING rule, no follow-up data):
  lambda_b = argmin over the DEFAULT_LAM_GRID of the mean season-forward
  CV RMSE across pairs (b_v -> b_v+1) with b_v+1 <= b (information
  available when the board would have been built).  b=2016 has exactly
  one pair (2015->2016).  The follow-up season f = b+1 NEVER enters
  lambda selection or the fit.
* OOS status: the ridge coefficients are jointly estimated on the
  baseline season(s) being valued -- like the current formulation's
  residual means, the VALUE is in-sample to the baseline year by
  construction; the board's predictive claim (follow-up f) is fully OOS
  for both formulations.  lambda_b uses seasons <= b only.

COVID handling identical to WS4.5 (primary mean includes COVID windows;
COVID-excluded published as a labeled sensitivity, never the headline).

Writes ``results/adjusted_war_v3/boards_2026-08-10/`` only; DB access
strictly read_only (follow-up outcomes + no other queries).

Usage:  python scripts/adjusted_war_v3_boards.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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
from src.analytics.marcel import (  # noqa: E402
    league_woba_by_season,
    load_batter_season_inputs,
)
from scripts.causal_war_backfill_windows import (  # noqa: E402
    SINGLE_WINDOWS,
    TWOYR_WINDOWS,
    build_u2,
    matched_naive_itt_2yr,
    matched_naive_itt_single,
    rank_boards,
    score_board_itt,
    score_board_itt_2yr,
    summarize_side,
)
from scripts.causal_war_contrarian_stability import (  # noqa: E402
    fetch_followup_outcomes,
    load_fg_war,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("awv3_boards")

SCRATCH = Path(
    "C:/Users/hunte/AppData/Local/Temp/claude/"
    "C--Users-hunte-projects-baseball/"
    "5112f9f4-f7ff-4db3-bd24-90acc5fbef27/scratchpad/ridge_cache"
)
WS45 = ROOT / "results" / "causal_war" / "backfill_windows_2026-08-10"
OUTDIR = ROOT / "results" / "adjusted_war_v3" / "boards_2026-08-10"

CV_PAIRS_ALL = [(b, b + 1) for b in range(2015, 2024)]  # 2015->2016 .. 2023->2024


def lambda_for_board(cv_path: list[dict], max_followup: int) -> tuple[float, int]:
    """Expanding-rule lambda: argmin mean RMSE over pairs with f_v <= max_followup."""
    keys = [f"{b}->{f}" for b, f in CV_PAIRS_ALL if f <= max_followup]
    best_lam, best_rmse = None, None
    for row in cv_path:
        vals = [row["per_pair"][k] for k in keys if row["per_pair"].get(k) == row["per_pair"].get(k)]
        if not vals:
            continue
        m = float(np.mean(vals))
        if best_rmse is None or m < best_rmse:
            best_lam, best_rmse = row["lambda"], m
    return float(best_lam), len(keys)


def ridge_scores(fit, position_col_map: dict[str, str]) -> pd.DataFrame:
    """Stack batter and pitcher values into [player_id, <pos_col>, adjusted_war_v3]."""
    pos_col = position_col_map["col"]
    bat = fit.batter[["player_id", "adjusted_war_v3"]].copy()
    bat[pos_col] = position_col_map["batter"]
    pit = fit.pitcher[["player_id", "adjusted_war_v3"]].copy()
    pit[pos_col] = position_col_map["pitcher"]
    return pd.concat([bat, pit], ignore_index=True)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection(read_only=True)
    fg = load_fg_war()

    # ---- Designs + full expanding CV grid (pairs 2015->16 .. 2023->24) ----
    frames = {s: extract_ridge_pa(conn, s, cache_dir=SCRATCH)
              for s in range(2015, 2025)}
    designs = {s: build_design([frames[s]], [s]) for s in frames}
    inputs = load_batter_season_inputs(conn, range(2015, 2025))
    league = league_woba_by_season(inputs)
    logger.info("Season-forward CV grid over %d pairs (for expanding lambda rule) ...",
                len(CV_PAIRS_ALL))
    cv = season_forward_lambda_cv(
        designs, inputs, league, CV_PAIRS_ALL,
        lam_grid=DEFAULT_LAM_GRID, pa_floor=100.0,
    )

    results: dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "pre-registered in scripts/adjusted_war_v3_boards.py "
                    "docstring (written before any ridge board was scored); "
                    "pools/rules/controls verbatim from WS4.5",
        "cv_grid": cv["path"],
        "single_season_windows": [],
        "two_year_windows": [],
    }

    # ---------------- single-season windows ----------------------------
    for b, f in SINGLE_WINDOWS:
        lam_b, n_pairs = lambda_for_board(cv["path"], max_followup=b)
        logger.info("=" * 60)
        logger.info("Single window %d -> %d: lambda_b=%s (from %d pairs)",
                    b, f, lam_b, n_pairs)
        pool = pd.read_parquet(WS45 / f"pool_single_{b}.parquet")
        fit = designs[b].solve(lam_b)
        scores = ridge_scores(fit, {"col": "position", "batter": "batter",
                                    "pitcher": "pitcher"})
        merged = pool.merge(scores, on=["player_id", "position"], how="inner")
        n_lost = len(pool) - len(merged)

        buy_low, over_valued = rank_boards(merged, "trad_war", "adjusted_war_v3")
        outcomes = fetch_followup_outcomes(
            conn, set(buy_low["player_id"]) | set(over_valued["player_id"]), f)

        win: dict[str, Any] = {
            "baseline_year": b, "followup_year": f,
            "lambda_b": lam_b, "n_cv_pairs": n_pairs,
            "covid_touched": bool(b == 2020 or f == 2020),
            "n_qualified_pool": int(len(merged)),
            "n_pool_rows_without_ridge_score": int(n_lost),
        }
        for side, board in (("buy_low", buy_low), ("over_valued", over_valued)):
            itt, scored = score_board_itt(board, outcomes, side)
            scored.to_csv(OUTDIR / f"ridge_single_{side}_{b}_to_{f}.csv", index=False)
            naive_itt = matched_naive_itt_single(board, fg, b, f, side)
            win[side] = summarize_side(itt, naive_itt)

        # ---- batter channel: ridge-as-batter-picker vs Marcel ----------
        mp = WS45 / f"marcel_proj_{f}.parquet"
        if not mp.exists():
            win["batter_channel"] = {"status": f"no WS4.5 marcel dump for {f}"}
        else:
            mproj = pd.read_parquet(mp)
            bat_pool = merged[merged["position"] == "batter"].merge(
                mproj[["player_id", "proj_woba"]], on="player_id", how="inner")
            bat_pool["_rank_group"] = "batter"
            ch: dict[str, Any] = {"n_batter_pool": int(len(bat_pool))}
            model_bl, model_ov = rank_boards(bat_pool, "trad_war", "adjusted_war_v3")
            marcel_bl, marcel_ov = rank_boards(bat_pool, "trad_war", "proj_woba")
            out_b = fetch_followup_outcomes(
                conn,
                set(model_bl["player_id"]) | set(model_ov["player_id"])
                | set(marcel_bl["player_id"]) | set(marcel_ov["player_id"]), f)
            for side, mb, kb in (("buy_low", model_bl, marcel_bl),
                                 ("over_valued", model_ov, marcel_ov)):
                m_itt, m_scored = score_board_itt(mb, out_b, side)
                k_itt, _ = score_board_itt(kb, out_b, side)
                m_scored.to_csv(
                    OUTDIR / f"ridge_single_batter_{side}_{b}_to_{f}.csv",
                    index=False)
                naive_b = matched_naive_itt_single(mb, fg, b, f, side,
                                                   batters_only=True)
                mr, kr = m_itt["itt"]["rate"], k_itt["itt"]["rate"]
                ch[side] = {
                    "ridge_batter_board": m_itt,
                    "marcel_board": k_itt,
                    "marcel_lift_pp": (round((mr - kr) * 100, 2)
                                       if mr is not None and kr is not None else None),
                    "batter_matched_naive_itt": naive_b,
                }
            win["batter_channel"] = ch
        results["single_season_windows"].append(win)

    # ---------------- 2-yr-aggregate windows ---------------------------
    for b, f in TWOYR_WINDOWS:
        years = (b - 1, b)
        lam_b, n_pairs = lambda_for_board(cv["path"], max_followup=b)
        logger.info("=" * 60)
        logger.info("2-yr window {%d,%d} -> %d: lambda_b=%s (from %d pairs)",
                    years[0], years[1], f, lam_b, n_pairs)
        pool = pd.read_parquet(WS45 / f"pool_twoyr_{years[0]}_{years[1]}.parquet")
        pooled_design = build_design([frames[years[0]], frames[years[1]]],
                                     list(years))
        fit = pooled_design.solve(lam_b)
        scores = ridge_scores(fit, {"col": "position_type", "batter": "batter",
                                    "pitcher": "pitcher"})
        merged = pool.merge(scores, on=["player_id", "position_type"], how="inner")
        n_lost = len(pool) - len(merged)

        u2 = build_u2(fg, years)
        fg_f = fg[fg["season"] == f][["player_id", "war"]].rename(
            columns={"war": "war_f"})
        buy_low, over_valued = rank_boards(merged, "war_agg", "adjusted_war_v3")

        win = {
            "baseline_years": list(years), "followup_year": f,
            "lambda_b": lam_b, "n_cv_pairs": n_pairs,
            "covid_touched": bool(2020 in years or f == 2020),
            "n_qualified_pool": int(len(merged)),
            "n_pool_rows_without_ridge_score": int(n_lost),
        }
        for side, board in (("buy_low", buy_low), ("over_valued", over_valued)):
            itt, scored = score_board_itt_2yr(board, side)
            scored.to_csv(
                OUTDIR / f"ridge_twoyr_{side}_{years[0]}_{years[1]}_to_{f}.csv",
                index=False)
            naive_itt = matched_naive_itt_2yr(board, u2, fg_f, side)
            win[side] = summarize_side(itt, naive_itt)

        mp = WS45 / f"marcel_proj_{f}.parquet"
        if not mp.exists():
            win["batter_channel"] = {"status": f"no WS4.5 marcel dump for {f}"}
        else:
            mproj = pd.read_parquet(mp)
            bat_pool = merged[merged["position_type"] == "batter"].merge(
                mproj[["player_id", "proj_woba"]], on="player_id", how="inner")
            bat_pool["_rank_group"] = "batter"
            ch = {"n_batter_pool": int(len(bat_pool))}
            model_bl, model_ov = rank_boards(bat_pool, "war_agg", "adjusted_war_v3")
            marcel_bl, marcel_ov = rank_boards(bat_pool, "war_agg", "proj_woba")
            for side, mb, kb in (("buy_low", model_bl, marcel_bl),
                                 ("over_valued", model_ov, marcel_ov)):
                m_itt, _ = score_board_itt_2yr(mb, side)
                k_itt, _ = score_board_itt_2yr(kb, side)
                naive_b = matched_naive_itt_2yr(mb, u2, fg_f, side,
                                                batters_only=True)
                mr, kr = m_itt["itt"]["rate"], k_itt["itt"]["rate"]
                ch[side] = {
                    "ridge_batter_board": m_itt,
                    "marcel_board": k_itt,
                    "marcel_lift_pp": (round((mr - kr) * 100, 2)
                                       if mr is not None and kr is not None else None),
                    "batter_matched_naive_itt": naive_b,
                }
            win["batter_channel"] = ch
        results["two_year_windows"].append(win)

    conn.close()

    # ---------------- K3 aggregates (identical construction to WS4.5) ----
    def _mean(vals: list[Optional[float]]) -> Optional[float]:
        vals = [v for v in vals if v is not None and v == v]
        return round(float(np.mean(vals)), 2) if vals else None

    agg: dict[str, Any] = {"note": (
        "Ridge-board K3 inputs, aggregate construction identical to WS4.5: "
        "primary mean includes COVID-touched windows exactly as they land; "
        "covid_excluded is the pre-declared labeled sensitivity. Marcel "
        "lifts are batter-channel."
    )}
    for cfg_key, windows in (("single_season", results["single_season_windows"]),
                             ("two_year_aggregate", results["two_year_windows"])):
        cfg_agg: dict[str, Any] = {}
        for side in ("buy_low", "over_valued"):
            naive_lifts = [w[side]["lift_itt_pp"] for w in windows]
            naive_lifts_nc = [w[side]["lift_itt_pp"] for w in windows
                              if not w["covid_touched"]]
            marcel_lifts = [
                w["batter_channel"].get(side, {}).get("marcel_lift_pp")
                for w in windows if side in w.get("batter_channel", {})]
            marcel_lifts_nc = [
                w["batter_channel"].get(side, {}).get("marcel_lift_pp")
                for w in windows
                if side in w.get("batter_channel", {}) and not w["covid_touched"]]
            cfg_agg[side] = {
                "mean_fully_oos_lift_vs_matched_naive_pp": _mean(naive_lifts),
                "per_window_naive_lifts_pp": naive_lifts,
                "mean_fully_oos_marcel_lift_pp_batter_channel": _mean(marcel_lifts),
                "per_window_marcel_lifts_pp": marcel_lifts,
                "covid_excluded_sensitivity": {
                    "mean_naive_lift_pp": _mean(naive_lifts_nc),
                    "mean_marcel_lift_pp": _mean(marcel_lifts_nc),
                },
            }
        agg[cfg_key] = cfg_agg
    results["k3_ridge_aggregates"] = agg

    (OUTDIR / "ridge_boards_summary.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", OUTDIR / "ridge_boards_summary.json")

    # ---------------- console table ------------------------------------
    print("\nK3 half 2 -- ridge-scored boards, exactly as they land")
    print("=" * 100)
    for cfg_key, windows in (("single", results["single_season_windows"]),
                             ("2yr", results["two_year_windows"])):
        for w in windows:
            base = w.get("baseline_year", w.get("baseline_years"))
            covid = " [COVID]" if w["covid_touched"] else ""
            for side in ("buy_low", "over_valued"):
                s = w[side]
                mr = s["model"]["itt"]["rate"]
                nr = s["matched_naive_itt"]["rate_itt"]
                ml = w["batter_channel"].get(side, {}).get("marcel_lift_pp") \
                    if isinstance(w.get("batter_channel"), dict) else None
                print(f"{cfg_key:6s} {str(base):12s}->{w['followup_year']}{covid:8s} "
                      f"{side:11s} ridge_ITT={mr if mr is not None else float('nan'):.3f} "
                      f"naive_ITT={nr:.3f} lift={s['lift_itt_pp']}pp "
                      f"marcel_lift(batter)={ml}pp")
    print("=" * 100)
    print(json.dumps(results["k3_ridge_aggregates"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
