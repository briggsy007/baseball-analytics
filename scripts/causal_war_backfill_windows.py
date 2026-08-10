#!/usr/bin/env python
"""WS4.5 (2026-08-10): backfill the contrarian windows 2015->2025 with
HONEST per-window nuisance retrains + matched-naive AND Marcel controls.

PRE-REGISTERED PROTOCOL -- written in full BEFORE any window was computed.
Nothing below was chosen by looking at a result (kill criterion K3 feeds
on these numbers; no post-hoc subgroup rescues).

Windows and nuisance checkpoints
--------------------------------
Baseline b -> follow-up f = b+1.  ``T(X)`` = HistGB nuisance trained on
seasons 2015..X with the FROZEN v1 recipe (12-confounder layout, 300
iters, depth 6, lr 0.05, min_samples_leaf 20, l2 1.0, seed 42 -- the
exact recipe of ``causal_war_trainsplit_2015_2022.pkl``, which slots in
as T(2022) byte-identical, no retrain).

* Single-season config: b in 2016..2024 (9 windows), nuisance T(b-1).
  Every window is FULLY OOS (train strictly before the baseline year).
  b=2022 is the honest retrain (T(2021)) of the previously in-sample
  window; b in {2023, 2024} reuse the frozen T(2022) exactly as the
  existing evidence runs did.
* 2-yr-aggregate config (baselines {b-1, b}): b in 2017..2024 (8
  windows), nuisance T(b-2) -- fully OOS for BOTH baseline years.
  b=2016 has no possible honest checkpoint (would need T(2014); no
  pre-2015 pitch data) and is reported as structurally unavailable.

COVID disclosure (declared now): windows touching 2020 are structurally
distorted (60-game season: follow-up 2020 crushes every bullish pick;
baseline 2020 halves baselines).  The distortion hits model AND controls
symmetrically, so lifts remain meaningful; the PRIMARY K3 aggregate
includes every fully-OOS window exactly as it lands, and a
COVID-excluded mean (dropping any window whose baseline or follow-up
involves 2020) is published as a labeled, pre-declared sensitivity --
never as the headline.

Scoring rules
-------------
* Single-season config: verbatim historical convention
  (``causal_war_contrarian_stability.py``): pool floors PA>=100/IP>=20 on
  baseline bWAR (A8 parquet), ranks within position, top-25 per side by
  rank_diff; hit rule ``row_hit_verdict`` (WAR-delta primary, ERA/OPS
  surrogate fallback from season stats).
* 2-yr-aggregate config: the frozen resolution spec section 5.5.3 protocol,
  applied self-consistently to BOTH the model board and the naive
  control: universe per player = sum of bWAR / volume over {b-1, b}
  (position by larger vol, tie -> batter), qualification batter
  vol>=200 / pitcher vol>=40; hit = ``war_f >= war_agg/2`` (Buy-Low,
  ``<`` Over-Valued); no surrogate branch; absent follow-up row = exit.
  (The historical A7 board used per-year-mean baselines with 300/50
  floors; its continuity numbers live in the WS4.6 rescoring. The
  backfill series needs ONE internally consistent rule for model and
  control -- the frozen section 5.5.3 rule is that rule.)
* ITT accounting everywhere per spec section 5.3: Buy-Low exit = MISS
  (denominator = all picks); Over-Valued exit = VOID (excluded,
  disclosed; worst-case = hits/25 published).  Per-protocol reported
  alongside, never as the headline.

Controls (every window, every config)
-------------------------------------
1. **Matched-naive, ITT-consistent**: neighbors of pick i = same
   position type, baseline WAR within +/-0.3 (single-season) / +/-0.6
   (2-yr, = 0.3 per baseline season, spec section 5.5.2), j != i, >= 3
   neighbors else the pick contributes nothing; each neighbor scored
   with the SAME rule and the SAME ITT exit treatment as the picks;
   naive rate = mean over contributing picks of mean neighbor score.
   The historical WAR-only (per-protocol) naive is also computed for
   continuity via ``compute_matched_naive``.
2. **Marcel-picker (batter channel)**: Marcel (WS4.3, literal Tango
   constants) projects follow-up wOBA from the three seasons <= b --
   outcome-blind by construction.  Because pitcher Marcel is deferred to
   the marcelR pin (frozen spec section 6.7), the Marcel control runs on the
   BATTER channel: within the window's qualified batter pool
   (intersected with the batters Marcel can project), build (a) the
   model-as-batter-picker board (ranks recomputed within that pool) and
   (b) the Marcel-picker board (rank_trad - rank_marcel, spec M1
   mirror), both top-25 per side, both scored with the identical rule +
   ITT.  Marcel lift = model batter-board ITT - Marcel board ITT.  The
   batter-pool matched-naive ITT is published next to both.
   Window b=2016 (f=2017) needs lag season 2014, which has no
   pitch-level wOBA source -> its Marcel control uses truncated 2-season
   history (2015-16) and is labeled as such.

Artifacts
---------
* Per-window nuisance checkpoints (write-once; loaded if they already
  exist): ``models/causal_war/backfill/causal_war_nuisance_2015_<X>_v1layout.pkl``
* ``results/causal_war/backfill_windows_2026-08-10/``:
  ``windows_summary.json`` (every number + K3 interim aggregates),
  per-window board CSVs.

Reads the DB strictly read_only; never touches any existing checkpoint.

Usage:  python scripts/causal_war_backfill_windows.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from src.analytics.causal_war import (  # noqa: E402
    _aggregate_player_effects,
    _build_features,
    _extract_pa_data,
)
from scripts.causal_war_contrarian_stability import (  # noqa: E402
    IP_MIN,
    PA_MIN,
    TOP_N,
    compute_matched_naive,
    fetch_followup_outcomes,
    known_pitcher_ids,
    load_fg_war,
    load_player_names,
    merge_with_bwar,
    row_hit_verdict,
)
from scripts.causal_war_itt_rescore import score_side  # noqa: E402
from src.analytics.marcel import (  # noqa: E402
    fetch_birthdates,
    load_batter_season_inputs,
    project_batters,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill_windows")

BF_DIR = ROOT / "models" / "causal_war" / "backfill"
OUTDIR = ROOT / "results" / "causal_war" / "backfill_windows_2026-08-10"
FROZEN_2015_2022 = ROOT / "models" / "causal_war" / "causal_war_trainsplit_2015_2022.pkl"

SINGLE_WINDOWS = [(b, b + 1) for b in range(2016, 2025)]      # 9 windows
TWOYR_WINDOWS = [(b, b + 1) for b in range(2017, 2025)]       # 8 windows
COVID_TOUCHED = {2019, 2020}  # window b: single touches 2020 iff b or f == 2020;
                              # 2-yr iff 2020 in {b-1, b, f}

_V2_ONLY_COLS = [
    "ump_accuracy_above_x", "ump_favor", "temp_f", "wind_speed_mph",
    "wind_dir_deg", "humidity_pct", "pressure_mb", "roof_status",
]

# v1 recipe hyperparameters (must equal the frozen checkpoint's config).
HGB_KW = dict(max_iter=300, max_depth=6, learning_rate=0.05,
              min_samples_leaf=20, l2_regularization=1.0, random_state=42)

NAIVE_WINDOW_1YR = 0.3
NAIVE_WINDOW_2YR = 0.6
MIN_NEIGHBORS = 3


# ---------------------------------------------------------------------------
# Season cache: 12-col v1 features per season (scoring path)
# ---------------------------------------------------------------------------

class SeasonCache:
    def __init__(self, conn):
        self.conn = conn
        self._cache: dict[int, dict[str, Any]] = {}

    def get(self, season: int) -> dict[str, Any]:
        if season not in self._cache:
            logger.info("Extracting season %d PAs (v1 12-col layout) ...", season)
            df = _extract_pa_data(self.conn, year_range=(season, season))
            df = df.drop(columns=[c for c in _V2_ONLY_COLS if c in df.columns])
            W, Y, bat_ids, dfb = _build_features(df)
            if W.shape[1] != 12:
                raise RuntimeError(
                    f"v1 layout expects 12 confounders, built {W.shape[1]}"
                )
            slim = pd.DataFrame({
                "woba_value": dfb["woba_value"].to_numpy(),
            })
            self._cache[season] = {
                "W": W, "Y": Y,
                "bat_ids": bat_ids.astype(int),
                "pit_ids": dfb["pitcher_id"].to_numpy().astype(int),
                "slim": slim,
            }
            logger.info("  season %d: %d PAs cached", season, len(Y))
        return self._cache[season]


# ---------------------------------------------------------------------------
# Nuisance ladder T(2015)..T(2021) + frozen T(2022)
# ---------------------------------------------------------------------------

def get_nuisance(conn, train_end: int) -> Any:
    """Load or train the ladder checkpoint for train window (2015, train_end)."""
    if train_end == 2022:
        art = joblib.load(FROZEN_2015_2022)
        logger.info("T(2022) = frozen checkpoint %s (no retrain)", FROZEN_2015_2022.name)
        return art["nuisance_outcome"]
    path = BF_DIR / f"causal_war_nuisance_2015_{train_end}_v1layout.pkl"
    if path.exists():
        logger.info("T(%d): loading existing %s (write-once)", train_end, path.name)
        return joblib.load(path)["nuisance_outcome"]
    logger.info("T(%d): training on 2015-%d (v1 recipe) ...", train_end, train_end)
    df = _extract_pa_data(conn, year_range=(2015, train_end))
    df = df.drop(columns=[c for c in _V2_ONLY_COLS if c in df.columns])
    W, Y, _bat, _dfb = _build_features(df)
    if W.shape[1] != 12:
        raise RuntimeError(f"v1 layout expects 12 confounders, built {W.shape[1]}")
    model = HistGradientBoostingRegressor(**HGB_KW)
    model.fit(W, Y)
    from sklearn.metrics import r2_score
    fit_r2 = float(r2_score(Y, model.predict(W)))
    BF_DIR.mkdir(parents=True, exist_ok=True)
    art = {
        "nuisance_outcome": model,
        "train_split": (2015, train_end),
        "feature_layout": "v1_12_confounders",
        "hyperparams": HGB_KW,
        "in_fit_r2": round(fit_r2, 5),
        "n_train_pa": int(len(Y)),
        "recipe": "WS4.5 honest per-window retrain; exact frozen-checkpoint "
                  "recipe (single full-train fit; no cross-fit/bootstrap "
                  "needed for scoring)",
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(art, path)
    logger.info("  T(%d) fit on %d PAs (in-fit R2 %.5f) -> %s",
                train_end, len(Y), fit_r2, path.name)
    return model


def effects_for(cache: SeasonCache, nuisance: Any,
                seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Batter/pitcher effect frames for *seasons* under one nuisance."""
    Yres_all, bat_all, pit_all, slim_all = [], [], [], []
    for s in seasons:
        c = cache.get(s)
        Yres = c["Y"] - nuisance.predict(c["W"])
        Yres_all.append(Yres)
        bat_all.append(c["bat_ids"])
        pit_all.append(c["pit_ids"])
        slim_all.append(c["slim"])
    Yres = np.concatenate(Yres_all)
    bat_ids = np.concatenate(bat_all)
    pit_ids = np.concatenate(pit_all)
    slim = pd.concat(slim_all, ignore_index=True)

    def _to_df(effects: dict[int, dict]) -> pd.DataFrame:
        rows = [{"player_id": int(pid),
                 "causal_war": float(e["causal_war"]),
                 "point_estimate_per_pa": float(e["point_estimate_per_pa"]),
                 "pa": int(e["pa"])} for pid, e in effects.items()]
        return pd.DataFrame(rows)

    bat = _to_df(_aggregate_player_effects(Yres, bat_ids, slim, pa_min=10))
    pit = _to_df(_aggregate_player_effects(-Yres, pit_ids, slim, pa_min=50))
    return bat, pit


# ---------------------------------------------------------------------------
# ITT board scoring + ITT-consistent matched-naive
# ---------------------------------------------------------------------------

def score_board_itt(board: pd.DataFrame, outcomes: dict[int, dict],
                    side: str) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = []
    for _, r in board.iterrows():
        hit, basis = row_hit_verdict(r, outcomes, side=side)
        rows.append({**r.to_dict(), "hit": hit, "basis": basis})
    scored = pd.DataFrame(rows)
    hits = [None if pd.isna(h) else bool(h) for h in scored["hit"]]
    return score_side(hits, side, len(scored)), scored


def matched_naive_itt_single(
    board: pd.DataFrame, fg: pd.DataFrame, b: int, f: int, side: str,
    batters_only: bool = False,
) -> dict[str, Any]:
    """ITT-consistent matched-naive for the single-season config."""
    fg_b = fg[fg["season"] == b][["player_id", "war", "pa_or_ip", "position_type"]]
    fg_b = fg_b.rename(columns={"war": "war_b", "pa_or_ip": "vol_b",
                                "position_type": "pos_type"})
    q = fg_b[((fg_b["pos_type"] == "batter") & (fg_b["vol_b"] >= PA_MIN))
             | ((fg_b["pos_type"] == "pitcher") & (fg_b["vol_b"] >= IP_MIN))].copy()
    if batters_only:
        q = q[q["pos_type"] == "batter"].copy()
    fg_f = fg[fg["season"] == f][["player_id", "war"]].rename(columns={"war": "war_f"})
    q = q.merge(fg_f, on="player_id", how="left")  # NaN war_f = exit

    per_pick, n_contributing = [], 0
    for _, p in board.iterrows():
        pos = "pitcher" if p["position"] == "pitcher" else "batter"
        cand = q[(q["pos_type"] == pos)
                 & ((q["war_b"] - float(p["trad_war"])).abs() <= NAIVE_WINDOW_1YR)
                 & (q["player_id"] != int(p["player_id"]))]
        if side == "buy_low":
            scores = np.where(
                cand["war_f"].notna() & (cand["war_f"] >= cand["war_b"]), 1.0, 0.0
            )  # exit -> 0 (miss), stays in denominator
        else:
            resolved = cand[cand["war_f"].notna()]
            scores = (resolved["war_f"] < resolved["war_b"]).astype(float).to_numpy()
        if len(scores) < MIN_NEIGHBORS:
            continue
        per_pick.append(float(np.mean(scores)))
        n_contributing += 1
    rate = float(np.mean(per_pick)) if per_pick else float("nan")
    return {"rate_itt": rate, "n_contributing_picks": n_contributing,
            "window": NAIVE_WINDOW_1YR,
            "convention": f"spec 5.3 ITT ({side}); >= {MIN_NEIGHBORS} neighbors"}


# ---------------------------------------------------------------------------
# 2-yr-aggregate config (frozen spec 5.5.3, self-consistent)
# ---------------------------------------------------------------------------

def build_u2(fg: pd.DataFrame, years: tuple[int, int]) -> pd.DataFrame:
    rows = fg[fg["season"].isin(years)]
    g = (rows.groupby(["player_id", "position_type"])
         .agg(war_agg=("war", "sum"), vol_agg=("pa_or_ip", "sum"))
         .reset_index())
    # Dual-type players: larger vol_agg governs; exact tie -> batter.
    g["_batter_first"] = (g["position_type"] == "batter").astype(int)
    g = (g.sort_values(["player_id", "vol_agg", "_batter_first"],
                       ascending=[True, False, False])
         .drop_duplicates("player_id", keep="first")
         .drop(columns="_batter_first"))
    qual = (((g["position_type"] == "batter") & (g["vol_agg"] >= 200))
            | ((g["position_type"] == "pitcher") & (g["vol_agg"] >= 40)))
    return g[qual].reset_index(drop=True)


def score_board_itt_2yr(board: pd.DataFrame, side: str) -> tuple[dict[str, Any], pd.DataFrame]:
    """Spec 5.5.3 rule: hit iff war_f >= war_agg/2 (BL) / < (OV); no surrogate."""
    rows = []
    for _, r in board.iterrows():
        if pd.isna(r["war_f"]):
            hit: Optional[bool] = None
            basis = "exit_no_followup_row"
        else:
            target = float(r["war_agg"]) / 2.0
            hit = bool(r["war_f"] >= target) if side == "buy_low" else bool(r["war_f"] < target)
            basis = "war_agg_half"
        rows.append({**r.to_dict(), "hit": hit, "basis": basis})
    scored = pd.DataFrame(rows)
    hits = [None if h is None or pd.isna(h) else bool(h) for h in scored["hit"]]
    return score_side(hits, side, len(scored)), scored


def matched_naive_itt_2yr(board: pd.DataFrame, u2: pd.DataFrame,
                          fg_f: pd.DataFrame, side: str,
                          batters_only: bool = False) -> dict[str, Any]:
    q = u2.merge(fg_f, on="player_id", how="left")
    if batters_only:
        q = q[q["position_type"] == "batter"].copy()
    per_pick, n_contributing = [], 0
    for _, p in board.iterrows():
        cand = q[(q["position_type"] == p["position_type"])
                 & ((q["war_agg"] - float(p["war_agg"])).abs() <= NAIVE_WINDOW_2YR)
                 & (q["player_id"] != int(p["player_id"]))]
        if side == "buy_low":
            scores = np.where(
                cand["war_f"].notna() & (cand["war_f"] >= cand["war_agg"] / 2.0),
                1.0, 0.0,
            )
        else:
            resolved = cand[cand["war_f"].notna()]
            scores = (resolved["war_f"] < resolved["war_agg"] / 2.0).astype(float).to_numpy()
        if len(scores) < MIN_NEIGHBORS:
            continue
        per_pick.append(float(np.mean(scores)))
        n_contributing += 1
    rate = float(np.mean(per_pick)) if per_pick else float("nan")
    return {"rate_itt": rate, "n_contributing_picks": n_contributing,
            "window": NAIVE_WINDOW_2YR,
            "convention": f"spec 5.5.3 + 5.3 ITT ({side}); >= {MIN_NEIGHBORS} neighbors"}


# ---------------------------------------------------------------------------
# Boards
# ---------------------------------------------------------------------------

def rank_boards(pool: pd.DataFrame, rank_col_trad: str, rank_col_model: str,
                top_n: int = TOP_N) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = pool.copy()
    pool["rank_trad"] = pool.groupby("_rank_group")[rank_col_trad].rank(
        ascending=False, method="min").astype(int)
    pool["rank_model"] = pool.groupby("_rank_group")[rank_col_model].rank(
        ascending=False, method="min").astype(int)
    pool["rank_diff"] = pool["rank_trad"] - pool["rank_model"]
    buy_low = pool.sort_values("rank_diff", ascending=False).head(top_n).copy()
    over_valued = pool.sort_values("rank_diff", ascending=True).head(top_n).copy()
    return buy_low, over_valued


def summarize_side(itt: dict[str, Any], naive_itt: dict[str, Any],
                   naive_war_only: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    model_rate = itt["itt"]["rate"]
    naive_rate = naive_itt["rate_itt"]
    lift = (
        round((model_rate - naive_rate) * 100, 2)
        if model_rate is not None and naive_rate == naive_rate else None
    )
    out = {
        "model": itt,
        "matched_naive_itt": naive_itt,
        "lift_itt_pp": lift,
    }
    if naive_war_only is not None:
        out["matched_naive_war_only_historical_convention"] = naive_war_only
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection(read_only=True)
    fg = load_fg_war()
    fg_seasons = sorted(fg["season"].unique())
    assert set(range(2015, 2026)).issubset(set(int(s) for s in fg_seasons)), \
        f"A8 parquet must cover 2015-2025; has {fg_seasons}"

    cache = SeasonCache(conn)
    results: dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "pre-registered in scripts/causal_war_backfill_windows.py "
                    "docstring (written before any window was computed)",
        "single_season_windows": [],
        "two_year_windows": [],
    }

    # Marcel inputs once (2015-2024 batter pa/woba) + birthdates.
    marcel_inputs = load_batter_season_inputs(conn, range(2015, 2025))
    bd = fetch_birthdates(sorted(marcel_inputs["player_id"].unique().tolist()))
    marcel_proj: dict[int, pd.DataFrame] = {}   # target season -> projections

    def get_marcel(f: int) -> tuple[Optional[pd.DataFrame], str]:
        if f in marcel_proj:
            pass
        else:
            lags = [f - 1, f - 2, f - 3]
            have = [y for y in lags if y in set(marcel_inputs["season"])]
            if f - 1 not in have:
                return None, "no lag-1 season available"
            sub = marcel_inputs[marcel_inputs["season"].isin(have)]
            if len(have) < 3:
                # Truncated history (only b=2016/f=2017: lag 2014 predates
                # pitch-level wOBA).  Pad the frame so league lags resolve;
                # players simply have PA=0 in missing lags.
                pad = pd.DataFrame({
                    "player_id": [-1] * (3 - len(have)),
                    "season": [y for y in lags if y not in have],
                    "pa": [1.0] * (3 - len(have)),
                    "woba": [float(np.average(sub[sub['season'] == max(have)]['woba'], weights=sub[sub['season'] == max(have)]['pa']))] * (3 - len(have)),
                })
                sub = pd.concat([sub, pad], ignore_index=True)
                label = f"TRUNCATED history: lags {sorted(have)} only (no pitch-level wOBA before 2015)"
            else:
                label = f"full 3-season history {sorted(have)}"
            proj = project_batters(sub, season=f, birthdates=bd)
            proj = proj[proj["player_id"] > 0]
            marcel_proj[f] = proj
            marcel_proj[f].attrs["label"] = label
            # C2b additive dump (mechanics only, protocol unchanged): the
            # WS4.2 ridge-board rebuild reuses the identical Marcel boards.
            proj.to_parquet(OUTDIR / f"marcel_proj_{f}.parquet", index=False)
        return marcel_proj[f], marcel_proj[f].attrs.get("label", "")

    # ---------------- single-season windows ----------------------------
    for b, f in SINGLE_WINDOWS:
        train_end = min(b - 1, 2022)
        logger.info("=" * 64)
        logger.info("Single-season window %d -> %d (nuisance T(%d), fully OOS)",
                    b, f, train_end)
        nuisance = get_nuisance(conn, train_end)
        bat_eff, pit_eff = effects_for(cache, nuisance, [b])

        pitcher_ids = known_pitcher_ids(conn, b)
        names = load_player_names(conn, set(bat_eff["player_id"]) | set(pit_eff["player_id"]))
        merged = merge_with_bwar(bat_eff, pit_eff, fg, baseline_year=b,
                                 pitcher_ids=pitcher_ids, names=names)
        merged["_rank_group"] = merged["position"]
        merged = merged.drop(columns=["rank_causal", "rank_trad", "rank_diff"])
        # C2b additive dump (mechanics only, protocol unchanged): per-window
        # qualified pool, reused verbatim by the WS4.2 ridge-board rebuild.
        merged.to_parquet(OUTDIR / f"pool_single_{b}.parquet", index=False)
        buy_low, over_valued = rank_boards(merged, "trad_war", "causal_war")

        outcomes = fetch_followup_outcomes(
            conn, set(buy_low["player_id"]) | set(over_valued["player_id"]), f)

        win: dict[str, Any] = {
            "baseline_year": b, "followup_year": f,
            "nuisance_train": [2015, train_end],
            "oos_status": "fully-OOS (train strictly before baseline)",
            "covid_touched": bool(b == 2020 or f == 2020),
            "n_qualified_pool": int(len(merged)),
        }
        for side, board in (("buy_low", buy_low), ("over_valued", over_valued)):
            itt, scored = score_board_itt(board, outcomes, side)
            scored.to_csv(OUTDIR / f"single_{side}_{b}_to_{f}.csv", index=False)
            naive_itt = matched_naive_itt_single(board, fg, b, f, side)
            naive_wo = compute_matched_naive(board, fg, b, f, side)
            win[side] = summarize_side(itt, naive_itt, naive_wo)

        # ---- batter channel: model-as-batter-picker vs Marcel ----------
        mproj, mlabel = get_marcel(f)
        if mproj is None:
            win["batter_channel"] = {"status": f"Marcel unavailable: {mlabel}"}
        else:
            bat_pool = merged[merged["position"] == "batter"].merge(
                mproj[["player_id", "proj_woba"]], on="player_id", how="inner")
            bat_pool["_rank_group"] = "batter"
            ch: dict[str, Any] = {
                "marcel_history": mlabel,
                "n_batter_pool": int(len(bat_pool)),
                "note": "model and Marcel boards drawn from the IDENTICAL "
                        "batter pool (qualified, causal-scored, "
                        "Marcel-projectable); pitcher Marcel deferred to "
                        "the marcelR pin (frozen spec 6.7)",
            }
            model_bl, model_ov = rank_boards(bat_pool, "trad_war", "causal_war")
            marcel_bl, marcel_ov = rank_boards(bat_pool, "trad_war", "proj_woba")
            out_b = fetch_followup_outcomes(
                conn,
                set(model_bl["player_id"]) | set(model_ov["player_id"])
                | set(marcel_bl["player_id"]) | set(marcel_ov["player_id"]),
                f,
            )
            for side, mb, kb in (("buy_low", model_bl, marcel_bl),
                                 ("over_valued", model_ov, marcel_ov)):
                m_itt, m_scored = score_board_itt(mb, out_b, side)
                k_itt, k_scored = score_board_itt(kb, out_b, side)
                m_scored.to_csv(OUTDIR / f"single_batter_model_{side}_{b}_to_{f}.csv", index=False)
                k_scored.to_csv(OUTDIR / f"single_batter_marcel_{side}_{b}_to_{f}.csv", index=False)
                naive_b = matched_naive_itt_single(mb, fg, b, f, side, batters_only=True)
                model_rate = m_itt["itt"]["rate"]
                marcel_rate = k_itt["itt"]["rate"]
                ch[side] = {
                    "model_batter_board": m_itt,
                    "marcel_board": k_itt,
                    "marcel_lift_pp": (
                        round((model_rate - marcel_rate) * 100, 2)
                        if model_rate is not None and marcel_rate is not None else None
                    ),
                    "batter_matched_naive_itt": naive_b,
                }
            win["batter_channel"] = ch
        results["single_season_windows"].append(win)

    # ---------------- 2-yr-aggregate windows ---------------------------
    for b, f in TWOYR_WINDOWS:
        train_end = min(b - 2, 2022)
        years = (b - 1, b)
        logger.info("=" * 64)
        logger.info("2-yr window {%d,%d} -> %d (nuisance T(%d), fully OOS)",
                    years[0], years[1], f, train_end)
        nuisance = get_nuisance(conn, train_end)
        bat_eff, pit_eff = effects_for(cache, nuisance, list(years))

        u2 = build_u2(fg, years)
        fg_f = fg[fg["season"] == f][["player_id", "war"]].rename(columns={"war": "war_f"})

        eff = pd.concat([
            bat_eff.assign(position_type="batter"),
            pit_eff.assign(position_type="pitcher"),
        ], ignore_index=True)
        pool = u2.merge(eff, on=["player_id", "position_type"], how="inner")
        pool = pool.merge(fg_f, on="player_id", how="left")
        names = load_player_names(conn, pool["player_id"].tolist())
        pool["name"] = pool["player_id"].map(lambda p: names.get(int(p), f"pid_{p}"))
        pool["_rank_group"] = pool["position_type"]
        # C2b additive dump (mechanics only, protocol unchanged).
        pool.to_parquet(
            OUTDIR / f"pool_twoyr_{years[0]}_{years[1]}.parquet", index=False)
        buy_low, over_valued = rank_boards(pool, "war_agg", "causal_war")

        win = {
            "baseline_years": list(years), "followup_year": f,
            "nuisance_train": [2015, train_end],
            "oos_status": "fully-OOS (train strictly before both baseline years)",
            "covid_touched": bool(2020 in years or f == 2020),
            "n_qualified_pool": int(len(pool)),
            "rule": "spec 5.5.3: hit iff war_f >= war_agg/2 (BL) / < (OV); "
                    "absent follow-up row = exit",
        }
        for side, board in (("buy_low", buy_low), ("over_valued", over_valued)):
            itt, scored = score_board_itt_2yr(board, side)
            scored.to_csv(OUTDIR / f"twoyr_{side}_{years[0]}_{years[1]}_to_{f}.csv", index=False)
            naive_itt = matched_naive_itt_2yr(board, u2, fg_f, side)
            win[side] = summarize_side(itt, naive_itt)

        mproj, mlabel = get_marcel(f)
        if mproj is None:
            win["batter_channel"] = {"status": f"Marcel unavailable: {mlabel}"}
        else:
            bat_pool = pool[pool["position_type"] == "batter"].merge(
                mproj[["player_id", "proj_woba"]], on="player_id", how="inner")
            bat_pool["_rank_group"] = "batter"
            ch = {
                "marcel_history": mlabel,
                "n_batter_pool": int(len(bat_pool)),
            }
            model_bl, model_ov = rank_boards(bat_pool, "war_agg", "causal_war")
            marcel_bl, marcel_ov = rank_boards(bat_pool, "war_agg", "proj_woba")
            for side, mb, kb in (("buy_low", model_bl, marcel_bl),
                                 ("over_valued", model_ov, marcel_ov)):
                m_itt, _ = score_board_itt_2yr(mb, side)
                k_itt, _ = score_board_itt_2yr(kb, side)
                naive_b = matched_naive_itt_2yr(mb, u2, fg_f, side, batters_only=True)
                model_rate = m_itt["itt"]["rate"]
                marcel_rate = k_itt["itt"]["rate"]
                ch[side] = {
                    "model_batter_board": m_itt,
                    "marcel_board": k_itt,
                    "marcel_lift_pp": (
                        round((model_rate - marcel_rate) * 100, 2)
                        if model_rate is not None and marcel_rate is not None else None
                    ),
                    "batter_matched_naive_itt": naive_b,
                }
            win["batter_channel"] = ch
        results["two_year_windows"].append(win)

    conn.close()

    # ---------------- K3 interim aggregates ----------------------------
    def _mean(vals: list[Optional[float]]) -> Optional[float]:
        vals = [v for v in vals if v is not None and v == v]
        return round(float(np.mean(vals)), 2) if vals else None

    agg: dict[str, Any] = {"note": (
        "K3 interim inputs: every window fully-OOS; primary mean includes "
        "COVID-touched windows exactly as they land (pre-declared); "
        "covid_excluded is the pre-declared labeled sensitivity, never the "
        "headline. Marcel lifts are batter-channel (pitcher Marcel deferred "
        "per frozen spec 6.7)."
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
                for w in windows if side in w.get("batter_channel", {})
            ]
            marcel_lifts_nc = [
                w["batter_channel"].get(side, {}).get("marcel_lift_pp")
                for w in windows
                if side in w.get("batter_channel", {}) and not w["covid_touched"]
            ]
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
    results["k3_interim_aggregates"] = agg

    (OUTDIR / "windows_summary.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8",
    )
    logger.info("Wrote %s", OUTDIR / "windows_summary.json")

    # ---------------- console table ------------------------------------
    print("\nWS4.5 backfill windows -- ITT rates and lifts, exactly as they land")
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
                      f"{side:11s} model_ITT={mr if mr is not None else float('nan'):.3f} "
                      f"naive_ITT={nr:.3f} lift={s['lift_itt_pp']}pp "
                      f"marcel_lift(batter)={ml}pp")
    print("=" * 100)
    print(json.dumps(results["k3_interim_aggregates"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
