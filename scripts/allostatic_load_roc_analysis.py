"""Allostatic Load (ABL) validation pipeline -- flagship decision gate.

Mirrors ``scripts/viscoelastic_workload_roc_analysis.py`` but uses the
multi-channel decision-fatigue model from
``src/analytics/allostatic_load.py``.

Decision gate (from the ABL validation spec):
    AUC >= 0.60 on 30-day pre-IL window  -> PASS
    Median lead-time >= 14 days at operating threshold ABL=75 -> PASS
    FPR at operating threshold <= 30 %   -> PASS
    delta_AUC vs games-played baseline >= 0.05 -> clinically meaningful

Additional artifact tests:
    1. Seasonal-monotonicity control (partial out season_day via logistic).
    2. Percentile-saturation check on healthy seasons.

Read-only DB access. Seeds fixed at 42.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analytics.allostatic_load import (  # noqa: E402
    ABLConfig,
    calculate_abl,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("abl_roc")

DB_PATH = ROOT / "data" / "baseball.duckdb"
LABELS_PATH = ROOT / "data" / "injury_labels.parquet"
RESULTS_DIR = ROOT / "results" / "allostatic_load"
CHECKPOINTS_DIR = ROOT / "models" / "allostatic_load" / "per_batter"

SEED = 42

# ROC threshold sweep (ABL is 0-100, so match that range)
ROC_THRESHOLDS = list(range(0, 101, 2))
LEAD_TIME_THRESHOLDS = [60.0, 75.0, 85.0]
OPERATING_THRESHOLD = 75.0

# Flagship decision gate
GATE_AUC = 0.60
GATE_LEAD_TIME_DAYS = 14.0
GATE_FPR = 0.30
GATE_DELTA_AUC = 0.05
GATE_RESIDUAL_AUC = 0.55

# Cohort eligibility
PRE_IL_WINDOW_DAYS = 90
ROC_WINDOW_DAYS = 30
FIT_BUFFER_DAYS = 14
MIN_GAMES_BEFORE_BUFFER = 15
HEALTHY_MIN_GAMES = 60
HEALTHY_EVAL_WINDOW_DAYS = 90

WORKLOAD_BASELINE_WINDOW_DAYS = 30


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def connect_db_readonly(retries: int = 6, base_delay: float = 1.0) -> duckdb.DuckDBPyConnection:
    delay = base_delay
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return duckdb.connect(str(DB_PATH), read_only=True)
        except Exception as exc:
            last_exc = exc
            logger.warning("DB lock failure (attempt %d): %s", attempt + 1, exc)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"Could not connect to DB after {retries} attempts: {last_exc}")


def load_cohort() -> pd.DataFrame:
    """Load non_arm position-player IL placements 2015-2016."""
    df = pd.read_parquet(LABELS_PATH)
    df = df.dropna(subset=["pitcher_id", "il_date"]).copy()
    df["il_date"] = pd.to_datetime(df["il_date"])
    df["pitcher_id"] = df["pitcher_id"].astype(int)
    non_arm = df[df["injury_type"] == "non_arm"].copy()
    non_arm = non_arm.sort_values("il_date").drop_duplicates("pitcher_id", keep="first")
    non_arm = non_arm[(non_arm["il_date"] >= "2015-01-01") &
                      (non_arm["il_date"] <= "2016-12-31")].copy()
    return non_arm.reset_index(drop=True)


def all_injury_ids() -> set[int]:
    """All player IDs that appear anywhere in injury_labels (any year, any type)."""
    df = pd.read_parquet(LABELS_PATH).dropna(subset=["pitcher_id"])
    return set(df["pitcher_id"].astype(int).tolist())


def fetch_eligible_injured(
    conn: duckdb.DuckDBPyConnection,
    cohort: pd.DataFrame,
) -> tuple[list[dict], list[dict]]:
    """Filter cohort to those with sufficient pre-IL game volume as a batter."""
    keepers = []
    skipped = []
    for _, row in cohort.iterrows():
        pid = int(row["pitcher_id"])
        il_date = pd.Timestamp(row["il_date"])
        season = int(il_date.year)
        fit_cutoff = il_date - pd.Timedelta(days=FIT_BUFFER_DAYS)
        q = f"""
            SELECT COUNT(DISTINCT game_pk) AS n
            FROM pitches
            WHERE batter_id = {pid}
              AND EXTRACT(YEAR FROM game_date) = {season}
              AND game_date < DATE '{fit_cutoff.date()}'
        """
        try:
            n = int(conn.execute(q).fetchone()[0])
        except Exception as exc:
            skipped.append({"player_id": pid, "reason": "fetch_failed", "detail": str(exc)})
            continue
        if n < MIN_GAMES_BEFORE_BUFFER:
            skipped.append({
                "player_id": pid,
                "season": season,
                "reason": "insufficient_pre_il_games",
                "n_games": n,
            })
            continue
        keepers.append({
            "player_id": pid,
            "season": season,
            "il_date": il_date,
            "injury_description_raw": row.get("injury_description_raw"),
            "n_pre_il_games": n,
        })
    return keepers, skipped


def fetch_healthy_pool(
    conn: duckdb.DuckDBPyConnection,
    excluded_ids: set[int],
    seasons: tuple[int, ...] = (2015, 2016),
    min_games: int = HEALTHY_MIN_GAMES,
) -> list[tuple[int, int]]:
    placeholders = ",".join("?" * len(seasons))
    q = f"""
        SELECT batter_id, EXTRACT(YEAR FROM game_date)::INT AS season,
               COUNT(DISTINCT game_pk) AS n_games
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) IN ({placeholders})
        GROUP BY batter_id, season
        HAVING COUNT(DISTINCT game_pk) >= ?
        ORDER BY n_games DESC
    """
    df = conn.execute(q, list(seasons) + [min_games]).fetchdf()
    return [
        (int(bid), int(season))
        for bid, season in zip(df["batter_id"], df["season"])
        if int(bid) not in excluded_ids
    ]


# --------------------------------------------------------------------------
# ABL trajectory computation (delegate to production calculate_abl)
# --------------------------------------------------------------------------


def compute_abl_timeline(
    conn: duckdb.DuckDBPyConnection,
    batter_id: int,
    season: int,
    config: ABLConfig | None = None,
) -> pd.DataFrame:
    """Return per-game timeline DataFrame: ['game_date', 'composite_abl', 'game_pk']."""
    config = config or ABLConfig()
    # Use min_games=1 because we eligibility-filter outside this function;
    # here we want the full timeline regardless of count.
    cfg = ABLConfig(
        alpha_pitch_processing=config.alpha_pitch_processing,
        alpha_decision_conflict=config.alpha_decision_conflict,
        alpha_swing_exertion=config.alpha_swing_exertion,
        alpha_temporal_demand=config.alpha_temporal_demand,
        alpha_travel=config.alpha_travel,
        min_games=1,
        abl_max=config.abl_max,
    )
    res = calculate_abl(conn, batter_id, season, cfg)
    timeline = res.get("timeline")
    if timeline is None or timeline.empty:
        return pd.DataFrame(columns=["game_date", "composite_abl", "game_pk"])
    out = timeline[["game_pk", "game_date", "composite_abl"]].copy()
    out["game_date"] = pd.to_datetime(out["game_date"])
    return out.sort_values("game_date").reset_index(drop=True)


def compute_workload_baseline(timeline: pd.DataFrame, window_days: int = WORKLOAD_BASELINE_WINDOW_DAYS) -> pd.DataFrame:
    """For each game date, count games in trailing window_days. Scaled to 0-100."""
    if timeline.empty:
        return pd.DataFrame(columns=["game_date", "workload_score"])
    dates = pd.to_datetime(timeline["game_date"]).values
    counts = np.zeros(len(dates), dtype=np.float64)
    for i, d in enumerate(dates):
        ws = d - np.timedelta64(window_days, "D")
        counts[i] = float(np.sum((dates >= ws) & (dates <= d)))
    # Scale to 0-100 using max=30 (every day a game in 30-day window)
    scaled = np.clip(counts / 30.0 * 100.0, 0, 100)
    return pd.DataFrame({
        "game_date": pd.to_datetime(timeline["game_date"]).values,
        "workload_score": scaled,
    })


# --------------------------------------------------------------------------
# ROC helpers (mirrors VWR)
# --------------------------------------------------------------------------


def roc_from_scores(y_true, scores, thresholds=None):
    if thresholds is None:
        thresholds = ROC_THRESHOLDS
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return {
            "thresholds": list(thresholds),
            "tpr": [0.0] * len(thresholds),
            "fpr": [0.0] * len(thresholds),
            "auc": float("nan"),
            "n_pos": pos,
            "n_neg": neg,
        }
    tpr_list, fpr_list = [], []
    for t in thresholds:
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tpr_list.append(float(tp / pos))
        fpr_list.append(float(fp / neg))
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tps, fps = [0], [0]
    i = 0
    while i < len(y_sorted):
        j = i
        while j < len(y_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        tps.append(tps[-1] + int((y_sorted[i:j] == 1).sum()))
        fps.append(fps[-1] + int((y_sorted[i:j] == 0).sum()))
        i = j
    tpr_curve = np.array(tps, dtype=float) / pos
    fpr_curve = np.array(fps, dtype=float) / neg
    trapz = getattr(np, "trapezoid", np.trapz)
    auc = float(trapz(tpr_curve, fpr_curve))
    return {
        "thresholds": list(thresholds),
        "tpr": tpr_list,
        "fpr": fpr_list,
        "auc": auc,
        "n_pos": pos,
        "n_neg": neg,
    }


def bootstrap_auc_ci(y, s, n_iter=1000, rng=None):
    if rng is None:
        rng = np.random.RandomState(SEED)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n_iter):
        sample = rng.choice(idx, size=len(idx), replace=True)
        ys = y[sample]
        ss = s[sample]
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        r = roc_from_scores(ys, ss)
        if not math.isnan(r["auc"]):
            aucs.append(r["auc"])
    if not aucs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def first_breach_day(daily: pd.DataFrame, threshold: float, il_date: pd.Timestamp) -> float | None:
    breaches = daily[daily["composite_abl"] >= threshold]
    if breaches.empty:
        return None
    first = pd.Timestamp(breaches.iloc[0]["game_date"])
    return float((il_date - first).days)


def season_day_from_date(d: pd.Timestamp | np.datetime64) -> int:
    return int(pd.Timestamp(d).dayofyear)


def partial_out_auc(y: np.ndarray, score: np.ndarray, season_day: np.ndarray) -> dict:
    from sklearn.linear_model import LogisticRegression, LinearRegression

    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    sd = np.asarray(season_day, dtype=float).reshape(-1, 1)
    if y.sum() == 0 or y.sum() == len(y):
        return {"residual_auc": float("nan"), "season_day_auc": float("nan")}
    sd_roc = roc_from_scores(y, sd.ravel())
    season_day_auc = sd_roc["auc"]
    lin = LinearRegression().fit(sd, score)
    score_hat = lin.predict(sd)
    residual_score = score - score_hat
    res_roc = roc_from_scores(y, residual_score)
    X = np.column_stack([score, sd.ravel()])
    lr = LogisticRegression(max_iter=1000)
    try:
        lr.fit(X, y)
        p = lr.predict_proba(X)[:, 1]
        joint_auc = roc_from_scores(y, p)["auc"]
        coefs = lr.coef_.ravel().tolist()
    except Exception:
        joint_auc = float("nan")
        coefs = [float("nan"), float("nan")]
    return {
        "season_day_auc": season_day_auc,
        "residual_auc": res_roc["auc"],
        "joint_auc": joint_auc,
        "logistic_coef_abl": coefs[0],
        "logistic_coef_season_day": coefs[1],
    }


# --------------------------------------------------------------------------
# Per-injury sub-bucket tokenization (best-effort)
# --------------------------------------------------------------------------

_BODY_PART_TOKENS = [
    "oblique", "hamstring", "knee", "ankle", "hand", "wrist",
    "back", "hip", "groin", "concussion", "illness", "foot", "neck",
    "thumb", "finger", "calf", "quad",
]


def tokenize_injury(desc: str | None) -> list[str]:
    if not isinstance(desc, str):
        return ["unspecified"]
    d = desc.lower()
    hits = [t for t in _BODY_PART_TOKENS if t in d]
    return hits or ["unspecified"]


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------


def run_pipeline(
    max_pitchers: int | None = None,
    n_healthy: int = 200,
    skip_cache: bool = False,
) -> dict:
    set_seeds(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    conn = connect_db_readonly()

    # ---- Step 1: cohorts ------------------------------------------------
    logger.info("Loading injury cohort (non_arm 2015-16)")
    cohort = load_cohort()
    logger.info("Raw non_arm unique players (2015-16): %d", len(cohort))
    if max_pitchers is not None:
        cohort = cohort.head(max_pitchers).copy()

    injured_eligible, injured_skipped = fetch_eligible_injured(conn, cohort)
    logger.info("Eligible injured (>=%d games before il_date - %dd): %d",
                MIN_GAMES_BEFORE_BUFFER, FIT_BUFFER_DAYS, len(injured_eligible))

    excluded = all_injury_ids()
    healthy_pool = fetch_healthy_pool(conn, excluded, seasons=(2015, 2016),
                                      min_games=HEALTHY_MIN_GAMES)
    logger.info("Healthy pool candidates: %d", len(healthy_pool))
    rng = np.random.RandomState(SEED)
    rng.shuffle(healthy_pool)
    healthy_sample = healthy_pool[:n_healthy]
    logger.info("Healthy cohort selected: %d", len(healthy_sample))

    # ---- Step 2: compute injured trajectories ----------------------------
    logger.info("Computing injured ABL trajectories")
    injured_daily: dict[int, pd.DataFrame] = {}
    injured_meta: dict[int, dict] = {}
    t_inj_start = time.time()
    for i, rec in enumerate(injured_eligible):
        pid = rec["player_id"]
        season = rec["season"]
        il_date = rec["il_date"]
        ck = CHECKPOINTS_DIR / f"inj_{pid}_{season}.pkl"
        if not skip_cache and ck.exists():
            with open(ck, "rb") as f:
                payload = pickle.load(f)
            injured_daily[pid] = payload["timeline"]
            injured_meta[pid] = payload["meta"]
            continue
        try:
            timeline = compute_abl_timeline(conn, pid, season)
        except Exception as exc:
            logger.warning("ABL compute failed for pid=%d season=%d: %s", pid, season, exc)
            continue
        if timeline.empty:
            continue
        meta = {
            "player_id": pid,
            "season": season,
            "il_date": il_date,
            "injury_description_raw": rec.get("injury_description_raw"),
            "injury_tokens": tokenize_injury(rec.get("injury_description_raw")),
        }
        with open(ck, "wb") as f:
            pickle.dump({"timeline": timeline, "meta": meta}, f)
        injured_daily[pid] = timeline
        injured_meta[pid] = meta
        if (i + 1) % 5 == 0:
            logger.info("Injured ABL %d/%d, %.1f s elapsed",
                        i + 1, len(injured_eligible), time.time() - t_inj_start)
    logger.info("Injured ABL done: n=%d", len(injured_daily))

    # ---- Step 3: compute healthy trajectories ----------------------------
    logger.info("Computing healthy ABL trajectories")
    healthy_daily: dict[tuple[int, int], pd.DataFrame] = {}
    healthy_meta: dict[tuple[int, int], dict] = {}
    healthy_skipped: list[dict] = []
    t_hlt_start = time.time()
    rng_h = np.random.RandomState(SEED + 1)
    for i, (pid, season) in enumerate(healthy_sample):
        ck = CHECKPOINTS_DIR / f"hlt_{pid}_{season}.pkl"
        if not skip_cache and ck.exists():
            with open(ck, "rb") as f:
                payload = pickle.load(f)
            healthy_daily[(pid, season)] = payload["timeline"]
            healthy_meta[(pid, season)] = payload["meta"]
            continue
        try:
            timeline = compute_abl_timeline(conn, pid, season)
        except Exception as exc:
            healthy_skipped.append({"player_id": pid, "season": season,
                                    "reason": "compute_failed", "detail": str(exc)})
            continue
        if timeline.empty or len(timeline) < 20:
            healthy_skipped.append({"player_id": pid, "season": season,
                                    "reason": "insufficient_timeline",
                                    "n": int(len(timeline))})
            continue
        # Random 90-day window inside this season
        dmin = timeline["game_date"].min()
        dmax = timeline["game_date"].max()
        span = (dmax - dmin).days
        if span < HEALTHY_EVAL_WINDOW_DAYS:
            sub = timeline.copy()
        else:
            offset = int(rng_h.randint(0, max(span - HEALTHY_EVAL_WINDOW_DAYS, 1)))
            eval_start = dmin + pd.Timedelta(days=offset)
            eval_end = eval_start + pd.Timedelta(days=HEALTHY_EVAL_WINDOW_DAYS)
            sub = timeline[(timeline["game_date"] >= eval_start) &
                           (timeline["game_date"] <= eval_end)].copy()
        if sub.empty:
            healthy_skipped.append({"player_id": pid, "season": season,
                                    "reason": "empty_eval_window"})
            continue
        meta = {"player_id": pid, "season": season,
                "eval_start": str(sub["game_date"].min().date()),
                "eval_end": str(sub["game_date"].max().date())}
        with open(ck, "wb") as f:
            pickle.dump({"timeline": sub, "meta": meta}, f)
        healthy_daily[(pid, season)] = sub
        healthy_meta[(pid, season)] = meta
        if (i + 1) % 25 == 0:
            logger.info("Healthy ABL %d/%d, %.1f s elapsed",
                        i + 1, len(healthy_sample), time.time() - t_hlt_start)
    logger.info("Healthy ABL done: n=%d", len(healthy_daily))

    # ---- Coverage JSON ---------------------------------------------------
    coverage = {
        "n_cohort_raw_non_arm": int(len(cohort)),
        "n_injured_eligible": len(injured_eligible),
        "n_injured_skipped": len(injured_skipped),
        "n_injured_with_trajectory": len(injured_daily),
        "n_healthy_selected": len(healthy_sample),
        "n_healthy_with_trajectory": len(healthy_daily),
        "n_healthy_skipped": len(healthy_skipped),
        "wall_compute_minutes": round((time.time() - t_inj_start) / 60, 2),
        "skipped_reasons_injured": (
            pd.DataFrame(injured_skipped)["reason"].value_counts().to_dict()
            if injured_skipped else {}
        ),
        "skipped_reasons_healthy": (
            pd.DataFrame(healthy_skipped)["reason"].value_counts().to_dict()
            if healthy_skipped else {}
        ),
    }
    (RESULTS_DIR / "training_coverage.json").write_text(json.dumps(coverage, indent=2, default=str))

    # ---- Step 4: assemble ROC frame -------------------------------------
    logger.info("Assembling ROC frame at %d-day pre-IL window", ROC_WINDOW_DAYS)
    pos_rows = []
    for pid, daily in injured_daily.items():
        meta = injured_meta[pid]
        il_date = pd.Timestamp(meta["il_date"])
        mask = (daily["game_date"] >= il_date - pd.Timedelta(days=ROC_WINDOW_DAYS)) & \
               (daily["game_date"] <= il_date)
        sub = daily.loc[mask].copy()
        # Workload baseline: per-game count of games in trailing 30 days, on the FULL
        # season timeline so the rolling window has context, then filter to the ROC window.
        wb = compute_workload_baseline(daily)
        sub = sub.merge(wb, on="game_date", how="left").fillna({"workload_score": 0.0})
        for _, r in sub.iterrows():
            pos_rows.append({
                "player_id": pid,
                "game_date": r["game_date"],
                "abl": float(r["composite_abl"]),
                "workload_score": float(r["workload_score"]),
                "season_day": season_day_from_date(r["game_date"]),
                "injury_tokens": ",".join(meta.get("injury_tokens", ["unspecified"])),
                "label": 1,
            })

    neg_rows = []
    for (pid, season), daily in healthy_daily.items():
        wb = compute_workload_baseline(daily)
        sub = daily.merge(wb, on="game_date", how="left").fillna({"workload_score": 0.0})
        for _, r in sub.iterrows():
            neg_rows.append({
                "player_id": pid,
                "game_date": r["game_date"],
                "abl": float(r["composite_abl"]),
                "workload_score": float(r["workload_score"]),
                "season_day": season_day_from_date(r["game_date"]),
                "injury_tokens": "healthy",
                "label": 0,
            })

    pos_df = pd.DataFrame(pos_rows)
    neg_df = pd.DataFrame(neg_rows)
    all_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # 10:1 negative cap
    rng_neg = np.random.RandomState(SEED)
    n_pos = int(all_df["label"].sum())
    if len(neg_df) > n_pos * 10 and n_pos > 0:
        neg_df = neg_df.sample(n=n_pos * 10, random_state=rng_neg).reset_index(drop=True)
        all_df = pd.concat([pos_df, neg_df], ignore_index=True)

    all_df.to_csv(RESULTS_DIR / "roc_daily_scores.csv", index=False)

    y = all_df["label"].values.astype(int)
    s_abl = all_df["abl"].values.astype(float)
    s_wl = all_df["workload_score"].values.astype(float)
    season_day = all_df["season_day"].values.astype(float)

    roc_abl = roc_from_scores(y, s_abl)
    roc_wl = roc_from_scores(y, s_wl)
    ci_lo, ci_hi = bootstrap_auc_ci(y, s_abl, n_iter=1000)
    ci_wl_lo, ci_wl_hi = bootstrap_auc_ci(y, s_wl, n_iter=1000)

    delta_auc = (None if math.isnan(roc_abl["auc"]) or math.isnan(roc_wl["auc"])
                 else roc_abl["auc"] - roc_wl["auc"])

    # ---- Seasonal-monotonicity ----
    logger.info("Partialling out season_day")
    partial = partial_out_auc(y, s_abl, season_day)

    # ---- Per-injury-token AUC ----
    per_token_auc: dict[str, dict] = {}
    if len(pos_df) > 0:
        token_counter: dict[str, int] = {}
        for _, r in pos_df.iterrows():
            for tok in str(r["injury_tokens"]).split(","):
                token_counter[tok] = token_counter.get(tok, 0) + 1
        for tok, n in token_counter.items():
            if n < 5:
                continue
            mask_pos = pos_df["injury_tokens"].str.contains(tok, regex=False)
            sub_pos = pos_df[mask_pos]
            sub = pd.concat([sub_pos, neg_df], ignore_index=True)
            if int(sub["label"].sum()) < 2:
                continue
            r = roc_from_scores(sub["label"].values, sub["abl"].values)
            per_token_auc[tok] = {
                "auc": r["auc"],
                "n_pos": int(mask_pos.sum()),
                "n_neg": int(len(neg_df)),
            }

    roc_out = {
        "abl": {
            "auc": roc_abl["auc"],
            "auc_ci_95": [ci_lo, ci_hi],
            "tpr": roc_abl["tpr"],
            "fpr": roc_abl["fpr"],
            "thresholds": roc_abl["thresholds"],
            "n_pos": roc_abl["n_pos"],
            "n_neg": roc_abl["n_neg"],
        },
        "workload_baseline": {
            "auc": roc_wl["auc"],
            "auc_ci_95": [ci_wl_lo, ci_wl_hi],
            "tpr": roc_wl["tpr"],
            "fpr": roc_wl["fpr"],
            "n_pos": roc_wl["n_pos"],
            "n_neg": roc_wl["n_neg"],
        },
        "delta_auc": delta_auc,
        "seasonal_control": partial,
        "per_injury_token": per_token_auc,
    }
    (RESULTS_DIR / "roc_curve.json").write_text(json.dumps(roc_out, indent=2, default=str))

    # Plotly ROC
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc_abl["fpr"], y=roc_abl["tpr"],
                                  mode="lines+markers",
                                  name=f"ABL (AUC={roc_abl['auc']:.3f})"))
        fig.add_trace(go.Scatter(x=roc_wl["fpr"], y=roc_wl["tpr"],
                                  mode="lines+markers",
                                  name=f"Workload-baseline (AUC={roc_wl['auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  name="Random",
                                  line=dict(dash="dash", color="grey")))
        fig.update_layout(
            title=f"ABL 30-day pre-IL ROC (n_pos={roc_abl['n_pos']}, n_neg={roc_abl['n_neg']})",
            xaxis_title="False-positive rate",
            yaxis_title="True-positive rate",
        )
        fig.write_html(RESULTS_DIR / "roc_curve.html")
    except Exception as exc:
        logger.warning("ROC plot failed: %s", exc)

    # ---- Step 5: Lead-time -----------------------------------------------
    logger.info("Lead-time analysis")
    lead_rows = []
    for pid, daily in injured_daily.items():
        meta = injured_meta[pid]
        il_date = pd.Timestamp(meta["il_date"])
        mask = (daily["game_date"] >= il_date - pd.Timedelta(days=PRE_IL_WINDOW_DAYS)) & \
               (daily["game_date"] <= il_date)
        sub = daily.loc[mask]
        if sub.empty:
            continue
        rec = {
            "player_id": pid,
            "season": meta["season"],
            "il_date": str(il_date.date()),
            "injury_tokens": ",".join(meta.get("injury_tokens", ["unspecified"])),
            "n_pre_il_days": int(len(sub)),
        }
        for th in LEAD_TIME_THRESHOLDS:
            rec[f"lead_time_t{int(th)}"] = first_breach_day(sub, th, il_date)
        lead_rows.append(rec)

    lead_df = pd.DataFrame(lead_rows)
    for th in LEAD_TIME_THRESHOLDS:
        c = f"lead_time_t{int(th)}"
        if c not in lead_df.columns:
            lead_df[c] = pd.Series(dtype=float)
    lead_df.to_csv(RESULTS_DIR / "lead_time_per_pitcher.csv", index=False)

    lead_summary: dict = {}
    for th in LEAD_TIME_THRESHOLDS:
        key = f"lead_time_t{int(th)}"
        vals = lead_df[key].dropna()
        lead_summary[key] = {
            "n_breached": int(len(vals)),
            "n_total": int(len(lead_df)),
            "fraction_breached": float(len(vals) / max(len(lead_df), 1)),
            "median_days": float(vals.median()) if len(vals) else None,
            "mean_days": float(vals.mean()) if len(vals) else None,
            "p25_days": float(vals.quantile(0.25)) if len(vals) else None,
            "p75_days": float(vals.quantile(0.75)) if len(vals) else None,
        }
    (RESULTS_DIR / "lead_time_distribution.json").write_text(json.dumps(lead_summary, indent=2))

    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for th in LEAD_TIME_THRESHOLDS:
            vals = lead_df[f"lead_time_t{int(th)}"].dropna().values
            if len(vals) == 0:
                continue
            fig.add_trace(go.Histogram(x=vals, name=f"ABL>={int(th)}",
                                        nbinsx=20, opacity=0.6))
        fig.update_layout(barmode="overlay",
                          title="ABL Lead-Time Distribution (days before IL)",
                          xaxis_title="Lead time (days)",
                          yaxis_title="Number of batters")
        fig.write_html(RESULTS_DIR / "lead_time_distribution.html")
    except Exception as exc:
        logger.warning("Lead-time plot failed: %s", exc)

    # ---- Step 6: FPR on healthy seasons ----------------------------------
    logger.info("FPR on healthy seasons")
    fpr_rows = []
    saturation_max = []
    for key, daily in healthy_daily.items():
        pid, season = key
        if len(daily) == 0:
            continue
        max_abl = float(daily["composite_abl"].max())
        saturation_max.append(max_abl)
        row = {"player_id": pid, "season": season, "max_abl": max_abl}
        for th in LEAD_TIME_THRESHOLDS:
            t_int = int(th)
            row[f"pct_days_above_{t_int}"] = float((daily["composite_abl"] >= th).mean() * 100)
            row[f"breached_{t_int}"] = bool((daily["composite_abl"] >= th).any())
        fpr_rows.append(row)

    fpr_df = pd.DataFrame(fpr_rows)
    fpr_df.to_csv(RESULTS_DIR / "fpr_healthy_seasons.csv", index=False)
    fpr_summary: dict = {"n_healthy_seasons": int(len(fpr_df))}
    for th in LEAD_TIME_THRESHOLDS:
        t_int = int(th)
        col = f"breached_{t_int}"
        fpr_summary[f"pct_breach_{t_int}"] = (
            float(fpr_df[col].mean() * 100) if len(fpr_df) else None
        )
    if saturation_max:
        sat = np.array(saturation_max)
        fpr_summary["saturation_pct_ge_75"] = float((sat >= 75.0).mean() * 100)
        fpr_summary["saturation_pct_ge_85"] = float((sat >= 85.0).mean() * 100)
        fpr_summary["healthy_max_abl_median"] = float(np.median(sat))
    (RESULTS_DIR / "fpr_summary.json").write_text(json.dumps(fpr_summary, indent=2))

    # ---- Decision gate ---------------------------------------------------
    auc_pass = (not math.isnan(roc_abl["auc"])) and roc_abl["auc"] >= GATE_AUC
    median_lead = lead_summary.get(f"lead_time_t{int(OPERATING_THRESHOLD)}", {}).get("median_days")
    lead_pass = median_lead is not None and median_lead >= GATE_LEAD_TIME_DAYS
    fpr_at_op = fpr_summary.get(f"pct_breach_{int(OPERATING_THRESHOLD)}")
    fpr_pass = fpr_at_op is not None and (fpr_at_op / 100.0) <= GATE_FPR
    delta_pass = delta_auc is not None and delta_auc >= GATE_DELTA_AUC
    residual_auc = partial.get("residual_auc")
    seasonal_artifact = (
        residual_auc is not None
        and not math.isnan(residual_auc)
        and residual_auc < GATE_RESIDUAL_AUC
    )

    hard_passes = sum([auc_pass, lead_pass, fpr_pass])
    if hard_passes == 3 and delta_pass and not seasonal_artifact:
        verdict = "FLAGSHIP"
    elif hard_passes >= 2 and not seasonal_artifact:
        verdict = "MARGINAL"
    else:
        verdict = "NOT_FLAGSHIP"

    decision = {
        "auc_gate": {"value": roc_abl["auc"], "threshold": GATE_AUC, "pass": auc_pass},
        "lead_time_gate": {"value": median_lead, "threshold": GATE_LEAD_TIME_DAYS,
                            "pass": lead_pass, "operating_threshold": OPERATING_THRESHOLD},
        "fpr_gate": {"value_pct": fpr_at_op, "threshold_pct": GATE_FPR * 100,
                     "pass": fpr_pass},
        "delta_auc_gate": {"value": delta_auc, "threshold": GATE_DELTA_AUC,
                            "pass": delta_pass},
        "residual_auc_gate": {"value": residual_auc, "threshold": GATE_RESIDUAL_AUC,
                              "seasonal_artifact": seasonal_artifact},
        "verdict": verdict,
    }

    summary = {
        "cohort": coverage,
        "roc": roc_out,
        "lead_time": lead_summary,
        "fpr": fpr_summary,
        "decision_gate": decision,
        "wall_clock_minutes": round((time.time() - t0) / 60, 2),
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    conn.close()
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-pitchers", type=int, default=None,
                   help="Limit injured cohort size (debug)")
    p.add_argument("--n-healthy", type=int, default=200)
    p.add_argument("--skip-cache", action="store_true",
                   help="Recompute trajectories even if cached")
    args = p.parse_args()
    summary = run_pipeline(
        max_pitchers=args.max_pitchers,
        n_healthy=args.n_healthy,
        skip_cache=args.skip_cache,
    )
    d = summary["decision_gate"]
    print(json.dumps({
        "verdict": d["verdict"],
        "auc": d["auc_gate"]["value"],
        "auc_ci": summary["roc"]["abl"]["auc_ci_95"],
        "delta_auc": d["delta_auc_gate"]["value"],
        "median_lead": d["lead_time_gate"]["value"],
        "fpr_pct": d["fpr_gate"]["value_pct"],
        "residual_auc": summary["roc"]["seasonal_control"].get("residual_auc"),
        "season_day_auc": summary["roc"]["seasonal_control"].get("season_day_auc"),
        "n_injured": summary["cohort"]["n_injured_with_trajectory"],
        "n_healthy": summary["cohort"]["n_healthy_with_trajectory"],
        "wall_min": summary["wall_clock_minutes"],
    }, indent=2, default=str))
