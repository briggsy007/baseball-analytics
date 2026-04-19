"""Viscoelastic Workload (VWR) validation pipeline — flagship decision gate.

Mirrors ``scripts/mechanix_ae_roc_analysis.py`` but uses the SLS creep
model from ``src/analytics/viscoelastic_workload.py``.

Decision gate (from the VWR deep dive):
    AUC >= 0.65 on 30-day pre-IL window  → PASS
    Median lead-time >= 30 days           → PASS
    FPR at operating threshold <= 30 %   → PASS
    delta_AUC vs velocity-drop >= 0.10   → clinically meaningful

Additional artifact tests:
    1. Seasonal-monotonicity control (partial out season_day via logistic).
    2. Percentile-saturation check on healthy seasons.

Read-only DB access.  Seeds fixed at 42.
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

from src.analytics.viscoelastic_workload import (  # noqa: E402
    _DEFAULT_E1,
    _DEFAULT_E2,
    _DEFAULT_TAU,
    _E1_BOUNDS,
    _E2_BOUNDS,
    _TAU_BOUNDS,
    _MIN_CAREER_PITCHES,
    _compute_release_point_drift,
    _compute_time_deltas,
    compute_pitch_stress,
    compute_strain_state,
)
from scipy.optimize import minimize  # noqa: E402
from scipy.stats import percentileofscore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vwr_roc")

ARM_TYPES = [
    "tommy_john",
    "ucl_sprain",
    "shoulder",
    "elbow",
    "rotator_cuff",
    "labrum",
    "forearm",
    "other_arm",
]

DB_PATH = ROOT / "data" / "baseball.duckdb"
LABELS_PATH = ROOT / "data" / "injury_labels.parquet"
RESULTS_DIR = ROOT / "results" / "viscoelastic_workload"
CHECKPOINTS_DIR = ROOT / "models" / "viscoelastic" / "per_pitcher"

SEED = 42

# ROC threshold sweep (VWR is 0-100 percentile, so match that range)
ROC_THRESHOLDS = list(range(0, 101, 2))
LEAD_TIME_THRESHOLDS = [50.0, 70.0, 85.0]
OPERATING_THRESHOLD = 85.0

# Flagship decision gate
GATE_AUC = 0.65
GATE_LEAD_TIME_DAYS = 30.0
GATE_FPR = 0.30
GATE_DELTA_AUC = 0.10

# Windows
PRE_IL_WINDOW_DAYS = 90       # full trajectory window
ROC_WINDOW_DAYS = 30          # positive class window
FIT_BUFFER_DAYS = 60          # don't fit on pitches closer than this to IL
MIN_FIT_PITCHES = _MIN_CAREER_PITCHES  # 500
HEALTHY_MIN_PITCHES = 500
HEALTHY_FIT_WINDOW_DAYS = 60


# ──────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────


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
    df = pd.read_parquet(LABELS_PATH)
    arm = df[df["injury_type"].isin(ARM_TYPES)].copy()
    arm["il_date"] = pd.to_datetime(arm["il_date"])
    arm = arm.dropna(subset=["pitcher_id", "il_date"]).copy()
    arm["pitcher_id"] = arm["pitcher_id"].astype(int)
    arm = arm.sort_values("il_date").drop_duplicates("pitcher_id", keep="first")
    arm = arm[(arm["il_date"] >= "2015-01-01") & (arm["il_date"] <= "2016-12-31")].copy()
    return arm.reset_index(drop=True)


def fetch_pitches(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    date_lt: pd.Timestamp | None = None,
    date_ge: pd.Timestamp | None = None,
    date_le: pd.Timestamp | None = None,
) -> pd.DataFrame:
    conds = [
        "pitcher_id = ?",
        "release_speed IS NOT NULL",
        "pitch_type IS NOT NULL",
    ]
    params: list = [pitcher_id]
    if date_lt is not None:
        conds.append("game_date < ?")
        params.append(date_lt.date() if hasattr(date_lt, "date") else date_lt)
    if date_ge is not None:
        conds.append("game_date >= ?")
        params.append(date_ge.date() if hasattr(date_ge, "date") else date_ge)
    if date_le is not None:
        conds.append("game_date <= ?")
        params.append(date_le.date() if hasattr(date_le, "date") else date_le)
    where = " AND ".join(conds)
    q = f"""
        SELECT
            game_pk, game_date, pitch_type, release_speed,
            release_pos_x, release_pos_z,
            at_bat_number, pitch_number
        FROM pitches
        WHERE {where}
        ORDER BY game_date, at_bat_number, pitch_number
    """
    return conn.execute(q, params).fetchdf()


def fetch_healthy_pool(
    conn: duckdb.DuckDBPyConnection,
    injured_ids: set[int],
    seasons: tuple[int, ...] = (2015, 2016),
    min_pitches: int = HEALTHY_MIN_PITCHES,
) -> list[tuple[int, int]]:
    """Return (pitcher_id, season) pairs for pitchers with NO IL stint in labels."""
    placeholders = ",".join("?" * len(seasons))
    q = f"""
        SELECT pitcher_id, EXTRACT(YEAR FROM game_date)::INT AS season, COUNT(*) AS n
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) IN ({placeholders})
          AND release_speed IS NOT NULL
          AND pitch_type IS NOT NULL
        GROUP BY pitcher_id, season
        HAVING COUNT(*) >= ?
        ORDER BY n DESC
    """
    params = list(seasons) + [min_pitches]
    df = conn.execute(q, params).fetchdf()
    return [
        (int(pid), int(season))
        for pid, season in zip(df["pitcher_id"], df["season"])
        if int(pid) not in injured_ids
    ]


# ──────────────────────────────────────────────────────────────────────────
# Parameter fitting on a pre-filtered DataFrame (so we can control horizon)
# ──────────────────────────────────────────────────────────────────────────


def fit_parameters_from_df(df: pd.DataFrame, pitcher_id: int) -> dict[str, Any]:
    """Fit (E1, E2, tau) on a DataFrame already filtered to the desired horizon.

    Mirrors ``src.analytics.viscoelastic_workload.fit_pitcher_parameters`` but
    operates on the caller-provided frame so we can cleanly train on
    pre-IL-60 pitches only (no leakage) without hitting the built-in
    "all pitches" query.
    """
    n = len(df)
    result: dict[str, Any] = {
        "pitcher_id": pitcher_id,
        "E1": _DEFAULT_E1,
        "E2": _DEFAULT_E2,
        "tau": _DEFAULT_TAU,
        "fit_error": None,
        "n_pitches": n,
        "converged": False,
    }
    if n < MIN_FIT_PITCHES:
        result["reason"] = f"insufficient_pitches ({n} < {MIN_FIT_PITCHES})"
        return result

    drift = _compute_release_point_drift(df)
    if np.all(np.isnan(drift)):
        result["reason"] = "no_release_point"
        return result

    stress = compute_pitch_stress(df).values
    time_deltas = _compute_time_deltas(df)

    drift_valid = drift[~np.isnan(drift)]
    if len(drift_valid) == 0 or drift_valid.max() == 0:
        result["reason"] = "flat_drift"
        return result

    drift_norm = drift / drift_valid.max()
    drift_norm = np.nan_to_num(drift_norm, nan=0.0)

    max_fit_points = 2000
    if n > max_fit_points:
        step = n // max_fit_points
        indices = np.arange(0, n, step)
    else:
        indices = np.arange(n)

    def objective(params):
        e1, e2, tau_p = params
        predicted = compute_strain_state(stress, time_deltas, e1, e2, tau_p)
        ps_max = predicted.max()
        if ps_max == 0:
            return 1e6
        ps_norm = predicted / ps_max
        diff = ps_norm[indices] - drift_norm[indices]
        return float(np.mean(diff ** 2))

    try:
        res = minimize(
            objective,
            x0=[_DEFAULT_E1, _DEFAULT_E2, _DEFAULT_TAU],
            bounds=[_E1_BOUNDS, _E2_BOUNDS, _TAU_BOUNDS],
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-8},
        )
        result["E1"] = round(float(res.x[0]), 2)
        result["E2"] = round(float(res.x[1]), 2)
        result["tau"] = round(float(res.x[2]), 2)
        result["fit_error"] = round(float(res.fun), 6)
        result["converged"] = bool(res.success)
    except Exception as exc:
        result["reason"] = f"optimizer_failed: {exc}"

    return result


def compute_daily_vwr(
    pitches_df: pd.DataFrame,
    params: dict[str, Any],
    career_strain_distribution: np.ndarray,
) -> pd.DataFrame:
    """Compute per-pitch strain then aggregate to per-day VWR (max percentile)."""
    if pitches_df.empty:
        return pd.DataFrame(columns=["game_date", "vwr", "max_strain"])
    stress = compute_pitch_stress(pitches_df).values
    time_deltas = _compute_time_deltas(pitches_df)
    strain = compute_strain_state(
        stress, time_deltas,
        E1=params["E1"], E2=params["E2"], tau=params["tau"],
    )
    # VWR = percentile rank of each strain against the career distribution
    srt = np.sort(career_strain_distribution)
    if len(srt) == 0:
        srt = np.sort(strain)
    ranks = np.searchsorted(srt, strain, side="right")
    vwr = np.clip(ranks / max(len(srt), 1) * 100.0, 0.0, 100.0)

    tmp = pd.DataFrame({
        "game_date": pd.to_datetime(pitches_df["game_date"]).values,
        "vwr": vwr,
        "strain": strain,
    })
    daily = tmp.groupby("game_date", as_index=False).agg(
        vwr=("vwr", "max"),
        max_strain=("strain", "max"),
    )
    return daily.sort_values("game_date").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────
# ROC helpers
# ──────────────────────────────────────────────────────────────────────────


def roc_from_scores(y_true, scores, thresholds=None):
    if thresholds is None:
        thresholds = ROC_THRESHOLDS
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pos = y.sum()
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return {
            "thresholds": list(thresholds),
            "tpr": [0.0] * len(thresholds),
            "fpr": [0.0] * len(thresholds),
            "auc": float("nan"),
            "n_pos": int(pos),
            "n_neg": int(neg),
        }
    tpr_list, fpr_list = [], []
    for t in thresholds:
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tpr_list.append(float(tp / pos))
        fpr_list.append(float(fp / neg))
    # Full-resolution AUC
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tps, fps = [0], [0]
    i = 0
    while i < len(y_sorted):
        j = i
        while j < len(y_sorted) and s_sorted[j] == s_sorted[i]:
            j += 1
        tp_block = int((y_sorted[i:j] == 1).sum())
        fp_block = int((y_sorted[i:j] == 0).sum())
        tps.append(tps[-1] + tp_block)
        fps.append(fps[-1] + fp_block)
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
        "n_pos": int(pos),
        "n_neg": int(neg),
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
    breaches = daily[daily["vwr"] >= threshold]
    if breaches.empty:
        return None
    first = pd.Timestamp(breaches.iloc[0]["game_date"])
    return float((il_date - first).days)


# ──────────────────────────────────────────────────────────────────────────
# Velocity-drop baseline
# ──────────────────────────────────────────────────────────────────────────


def velocity_drop_score(speeds: np.ndarray, window: int = 10) -> np.ndarray:
    s = pd.Series(speeds, dtype=float)
    roll_mean = s.rolling(window, min_periods=window).mean().shift(1)
    roll_std = s.rolling(window, min_periods=window).std().shift(1)
    score = (roll_mean - s) / roll_std.replace(0, np.nan)
    return score.fillna(0.0).clip(lower=0.0).values.astype(float)


def daily_velocity_score(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["game_date", "vel_score"])
    raw = velocity_drop_score(df["release_speed"].values, window=window)
    tmp = pd.DataFrame({
        "game_date": pd.to_datetime(df["game_date"]).values,
        "score": raw,
    })
    return tmp.groupby("game_date", as_index=False).agg(vel_score=("score", "max")) \
              .sort_values("game_date").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────
# Seasonal-monotonicity control (partial out season_day via logistic regression)
# ──────────────────────────────────────────────────────────────────────────


def season_day_from_date(d: pd.Timestamp | np.datetime64) -> int:
    """Day-of-year (1-366) — proxy for "how far into the season"."""
    ts = pd.Timestamp(d)
    return int(ts.dayofyear)


def _residualize(score: np.ndarray, sd: np.ndarray) -> np.ndarray:
    """Linearly partial out season_day from a score, return residuals."""
    from sklearn.linear_model import LinearRegression
    sd2d = np.asarray(sd, dtype=float).reshape(-1, 1)
    lin = LinearRegression().fit(sd2d, score)
    return np.asarray(score, dtype=float) - lin.predict(sd2d)


def bootstrap_residual_auc_ci(
    y: np.ndarray,
    score: np.ndarray,
    season_day: np.ndarray,
    n_iter: int = 1000,
    rng: np.random.RandomState | None = None,
) -> tuple[float, float]:
    """Bootstrap 95% CI for the seasonal-residual AUC.

    Resample (y, score, season_day) tuples; refit the linear partial-out on each
    resample and recompute residual AUC. Mirrors :func:`bootstrap_auc_ci`.
    """
    if rng is None:
        rng = np.random.RandomState(SEED)
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    sd = np.asarray(season_day, dtype=float)
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n_iter):
        sample = rng.choice(idx, size=len(idx), replace=True)
        ys, ss, sds = y[sample], score[sample], sd[sample]
        if ys.sum() == 0 or ys.sum() == len(ys):
            continue
        try:
            res = _residualize(ss, sds)
        except Exception:
            continue
        r = roc_from_scores(ys, res)
        if not math.isnan(r["auc"]):
            aucs.append(r["auc"])
    if not aucs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


def partial_out_auc(
    y: np.ndarray,
    score: np.ndarray,
    season_day: np.ndarray,
    score_compare: np.ndarray | None = None,
) -> dict:
    """Fit linear regression score ~ season_day, residualize, recompute AUC.

    A simple partial-correlation proxy: if the score only tracks season_day, the
    residual AUC will collapse toward 0.5. We also compute the joint logistic
    (score + season_day) AUC and (optionally) the residual AUC of a comparison
    score (e.g., velocity-drop) so that the **residual-space delta_AUC** between
    VWR and a baseline can be reported as a clinical-delta gate in the same
    confounding-controlled space as the headline AUC gate.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    sd = np.asarray(season_day, dtype=float).reshape(-1, 1)

    if y.sum() == 0 or y.sum() == len(y):
        return {"residual_auc": float("nan"), "season_day_auc": float("nan")}

    # (a) AUC of season_day alone — how much does "calendar time" itself predict IL?
    sd_roc = roc_from_scores(y, sd.ravel())
    season_day_auc = sd_roc["auc"]

    # (b) Linear residualization of the headline score wrt season_day.
    residual_score = _residualize(score, sd.ravel())
    res_roc = roc_from_scores(y, residual_score)

    # (c) Combined logistic model: logit(y) = a*score + b*season_day.
    # The partial effect of VWR is captured by AUC of predicted prob.
    X = np.column_stack([score, sd.ravel()])
    lr = LogisticRegression(max_iter=1000)
    try:
        lr.fit(X, y)
        p = lr.predict_proba(X)[:, 1]
        joint_roc = roc_from_scores(y, p)
        joint_auc = joint_roc["auc"]
        coefs = lr.coef_.ravel().tolist()
    except Exception:
        joint_auc = float("nan")
        coefs = [float("nan"), float("nan")]

    out = {
        "season_day_auc": season_day_auc,
        "residual_auc": res_roc["auc"],
        "joint_auc": joint_auc,
        "logistic_coef_vwr": coefs[0],
        "logistic_coef_season_day": coefs[1],
    }

    # (d) Optional: residual-space AUC for a comparison score (velocity-drop).
    # Used to build the delta_AUC-in-residual-space clinical-delta gate.
    if score_compare is not None:
        sc2 = np.asarray(score_compare, dtype=float)
        residual_compare = _residualize(sc2, sd.ravel())
        res2_roc = roc_from_scores(y, residual_compare)
        out["compare_residual_auc"] = res2_roc["auc"]
        out["compare_raw_auc"] = roc_from_scores(y, sc2)["auc"]
        if not math.isnan(out["residual_auc"]) and not math.isnan(out["compare_residual_auc"]):
            out["delta_residual_auc"] = float(out["residual_auc"] - out["compare_residual_auc"])
        else:
            out["delta_residual_auc"] = float("nan")

    return out


# ──────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────


def run_pipeline(
    max_pitchers: int | None = None,
    n_healthy: int = 300,
    skip_fit: bool = False,
) -> dict:
    set_seeds(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    conn = connect_db_readonly()

    # ── Step 1: cohorts ────────────────────────────────────────────────
    logger.info("Loading injury cohort")
    cohort = load_cohort()
    logger.info("Arm-injury unique pitchers (2015-16): %d", len(cohort))
    if max_pitchers is not None:
        cohort = cohort.head(max_pitchers).copy()

    injured_ids = set(cohort["pitcher_id"].tolist())
    logger.info("Loading healthy pool")
    healthy_pool = fetch_healthy_pool(conn, injured_ids, seasons=(2015, 2016), min_pitches=HEALTHY_MIN_PITCHES)
    logger.info("Healthy pool candidates: %d", len(healthy_pool))
    rng = np.random.RandomState(SEED)
    rng.shuffle(healthy_pool)
    healthy_sample = healthy_pool[:n_healthy]
    logger.info("Healthy cohort selected: %d", len(healthy_sample))

    # ── Step 2: fit parameters ────────────────────────────────────────
    logger.info("Fitting VWR parameters on injured cohort")
    injured_params: dict[int, dict] = {}
    injured_skipped: list[dict] = []
    t_fit_start = time.time()

    for idx, row in cohort.iterrows():
        pid = int(row["pitcher_id"])
        il_date = pd.Timestamp(row["il_date"])
        fit_cutoff = il_date - pd.Timedelta(days=FIT_BUFFER_DAYS)

        ck_path = CHECKPOINTS_DIR / f"inj_{pid}.pkl"
        if skip_fit and ck_path.exists():
            with open(ck_path, "rb") as f:
                payload = pickle.load(f)
            injured_params[pid] = payload
            continue

        try:
            fit_df = fetch_pitches(conn, pid, date_lt=fit_cutoff)
        except Exception as exc:
            injured_skipped.append({"pitcher_id": pid, "injury_type": row["injury_type"],
                                     "reason": "fetch_failed", "detail": str(exc)})
            continue
        if len(fit_df) < MIN_FIT_PITCHES:
            injured_skipped.append({"pitcher_id": pid, "injury_type": row["injury_type"],
                                     "reason": "insufficient_fit_pitches",
                                     "n_fit": int(len(fit_df))})
            continue

        params = fit_parameters_from_df(fit_df, pid)
        if "reason" in params:
            injured_skipped.append({"pitcher_id": pid, "injury_type": row["injury_type"],
                                     "reason": params["reason"]})
            continue

        # Build career strain distribution on the fit df (for VWR percentiles)
        stress = compute_pitch_stress(fit_df).values
        tds = _compute_time_deltas(fit_df)
        career_strain = compute_strain_state(stress, tds, params["E1"], params["E2"], params["tau"])

        payload = {
            "pitcher_id": pid,
            "injury_type": row["injury_type"],
            "il_date": str(il_date.date()),
            "params": params,
            "career_strain": career_strain,
            "n_fit_pitches": int(len(fit_df)),
        }
        with open(ck_path, "wb") as f:
            pickle.dump(payload, f)
        injured_params[pid] = payload

        if (idx + 1) % 20 == 0:
            logger.info("Fitted %d injured so far, %.1f min elapsed",
                        len(injured_params), (time.time() - t_fit_start) / 60)

    logger.info("Injured fit: n=%d, skipped=%d, %.1f min",
                len(injured_params), len(injured_skipped),
                (time.time() - t_fit_start) / 60)

    # Healthy cohort fit on a random 60-day window
    logger.info("Fitting VWR parameters on healthy cohort")
    healthy_params: dict[tuple[int, int], dict] = {}
    healthy_skipped: list[dict] = []
    t_healthy_start = time.time()
    rng_h = np.random.RandomState(SEED + 1)

    for i, (pid, season) in enumerate(healthy_sample):
        ck_path = CHECKPOINTS_DIR / f"hlt_{pid}_{season}.pkl"
        if skip_fit and ck_path.exists():
            with open(ck_path, "rb") as f:
                payload = pickle.load(f)
            healthy_params[(pid, season)] = payload
            continue

        start = pd.Timestamp(f"{season}-01-01")
        end = pd.Timestamp(f"{season}-12-31")
        try:
            season_df = fetch_pitches(conn, pid, date_ge=start, date_le=end)
        except Exception as exc:
            healthy_skipped.append({"pitcher_id": pid, "season": season,
                                     "reason": "fetch_failed", "detail": str(exc)})
            continue
        if len(season_df) < HEALTHY_MIN_PITCHES:
            healthy_skipped.append({"pitcher_id": pid, "season": season,
                                     "reason": "insufficient_season",
                                     "n": int(len(season_df))})
            continue

        # Fit on all season pitches (the "healthy baseline")
        params = fit_parameters_from_df(season_df, pid)
        if "reason" in params:
            healthy_skipped.append({"pitcher_id": pid, "season": season,
                                     "reason": params["reason"]})
            continue

        stress = compute_pitch_stress(season_df).values
        tds = _compute_time_deltas(season_df)
        career_strain = compute_strain_state(stress, tds, params["E1"], params["E2"], params["tau"])

        # For healthy evaluation we use a random 90-day window within the season
        season_dates = pd.to_datetime(season_df["game_date"])
        dmin, dmax = season_dates.min(), season_dates.max()
        span = (dmax - dmin).days
        if span < 90:
            eval_start = dmin
            eval_end = dmax
        else:
            offset = int(rng_h.randint(0, max(span - 90, 1)))
            eval_start = dmin + pd.Timedelta(days=offset)
            eval_end = eval_start + pd.Timedelta(days=90)

        payload = {
            "pitcher_id": pid,
            "season": season,
            "params": params,
            "career_strain": career_strain,
            "eval_start": eval_start,
            "eval_end": eval_end,
            "n_season_pitches": int(len(season_df)),
        }
        with open(ck_path, "wb") as f:
            pickle.dump(payload, f)
        healthy_params[(pid, season)] = payload

        if (i + 1) % 25 == 0:
            logger.info("Healthy fit %d/%d, %.1f min elapsed",
                        i + 1, len(healthy_sample),
                        (time.time() - t_healthy_start) / 60)

    logger.info("Healthy fit: n=%d, skipped=%d, %.1f min",
                len(healthy_params), len(healthy_skipped),
                (time.time() - t_healthy_start) / 60)

    # Coverage JSON
    coverage = {
        "n_cohort": int(len(cohort)),
        "n_injured_fit": len(injured_params),
        "n_injured_skipped": len(injured_skipped),
        "n_healthy_selected": len(healthy_sample),
        "n_healthy_fit": len(healthy_params),
        "n_healthy_skipped": len(healthy_skipped),
        "wall_fit_minutes": round((time.time() - t_fit_start) / 60, 2),
        "per_injury_fit": (
            pd.DataFrame([{"injury_type": p["injury_type"]} for p in injured_params.values()])
              ["injury_type"].value_counts().to_dict()
            if injured_params else {}
        ),
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

    # ── Step 3: compute trajectories ───────────────────────────────────
    logger.info("Computing injured trajectories (pre-IL 90 days)")
    injured_daily: dict[int, pd.DataFrame] = {}
    injured_daily_vel: dict[int, pd.DataFrame] = {}
    for pid, payload in injured_params.items():
        il_date = pd.Timestamp(payload["il_date"])
        start = il_date - pd.Timedelta(days=PRE_IL_WINDOW_DAYS)
        try:
            pre_df = fetch_pitches(conn, pid, date_ge=start, date_le=il_date)
        except Exception:
            continue
        if pre_df.empty:
            continue
        daily = compute_daily_vwr(pre_df, payload["params"], payload["career_strain"])
        if daily.empty:
            continue
        dvel = daily_velocity_score(pre_df)
        injured_daily[pid] = daily
        injured_daily_vel[pid] = dvel

    logger.info("Injured with trajectories: %d", len(injured_daily))

    logger.info("Computing healthy trajectories (random 90-day window)")
    healthy_daily: dict[tuple[int, int], pd.DataFrame] = {}
    healthy_daily_vel: dict[tuple[int, int], pd.DataFrame] = {}
    for key, payload in healthy_params.items():
        pid, season = key
        try:
            eval_df = fetch_pitches(conn, pid, date_ge=payload["eval_start"], date_le=payload["eval_end"])
        except Exception:
            continue
        if eval_df.empty:
            continue
        daily = compute_daily_vwr(eval_df, payload["params"], payload["career_strain"])
        if daily.empty:
            continue
        dvel = daily_velocity_score(eval_df)
        healthy_daily[key] = daily
        healthy_daily_vel[key] = dvel

    logger.info("Healthy with trajectories: %d", len(healthy_daily))

    # ── Step 4: ROC / AUC at 30-day pre-IL window ──────────────────────
    logger.info("Computing ROC/AUC at 30-day window")

    pos_rows = []
    for pid, daily in injured_daily.items():
        il_date = pd.Timestamp(injured_params[pid]["il_date"])
        mask = (daily["game_date"] >= il_date - pd.Timedelta(days=ROC_WINDOW_DAYS)) & \
               (daily["game_date"] <= il_date)
        vel = injured_daily_vel[pid]
        sub = daily.loc[mask, ["game_date", "vwr"]].merge(vel, on="game_date", how="left") \
                                                     .fillna({"vel_score": 0.0})
        for _, r in sub.iterrows():
            pos_rows.append({
                "pitcher_id": pid,
                "game_date": r["game_date"],
                "vwr": float(r["vwr"]),
                "vel_score": float(r["vel_score"]),
                "season_day": season_day_from_date(r["game_date"]),
                "injury_type": injured_params[pid]["injury_type"],
                "label": 1,
            })

    neg_rows = []
    for (pid, season), daily in healthy_daily.items():
        vel = healthy_daily_vel[(pid, season)]
        sub = daily.merge(vel, on="game_date", how="left").fillna({"vel_score": 0.0})
        for _, r in sub.iterrows():
            neg_rows.append({
                "pitcher_id": pid,
                "game_date": r["game_date"],
                "vwr": float(r["vwr"]),
                "vel_score": float(r["vel_score"]),
                "season_day": season_day_from_date(r["game_date"]),
                "injury_type": "healthy",
                "label": 0,
            })

    pos_df = pd.DataFrame(pos_rows)
    neg_df = pd.DataFrame(neg_rows)
    all_df = pd.concat([pos_df, neg_df], ignore_index=True)

    # Balance negatives to 10:1 cap
    rng_neg = np.random.RandomState(SEED)
    n_pos = int(all_df["label"].sum())
    if len(neg_df) > n_pos * 10 and n_pos > 0:
        neg_df = neg_df.sample(n=n_pos * 10, random_state=rng_neg).reset_index(drop=True)
        all_df = pd.concat([pos_df, neg_df], ignore_index=True)

    (RESULTS_DIR / "roc_daily_scores.csv").write_text("")  # overwritten below
    all_df.to_csv(RESULTS_DIR / "roc_daily_scores.csv", index=False)

    y = all_df["label"].values.astype(int)
    s_vwr = all_df["vwr"].values.astype(float)
    s_vel = all_df["vel_score"].values.astype(float)
    season_day = all_df["season_day"].values.astype(float)

    # Scale velocity to 0-100 for comparable plotting (AUC invariant anyway)
    if len(s_vel) > 0 and np.nanmax(s_vel) > 0:
        p99 = np.nanpercentile(s_vel, 99)
        s_vel_scaled = 100.0 * (s_vel / p99) if p99 > 0 else s_vel
        s_vel_scaled = np.clip(s_vel_scaled, 0, 100)
    else:
        s_vel_scaled = s_vel

    roc_vwr = roc_from_scores(y, s_vwr)
    roc_vel = roc_from_scores(y, s_vel_scaled)
    ci_lo, ci_hi = bootstrap_auc_ci(y, s_vwr, n_iter=1000)
    ci_vel_lo, ci_vel_hi = bootstrap_auc_ci(y, s_vel_scaled, n_iter=1000)

    delta_auc = (None if math.isnan(roc_vwr["auc"]) or math.isnan(roc_vel["auc"])
                 else roc_vwr["auc"] - roc_vel["auc"])

    # ── Seasonal-monotonicity control ─────────────────────────────────
    logger.info("Partialling out season_day")
    partial = partial_out_auc(y, s_vwr, season_day, score_compare=s_vel_scaled)
    # Bootstrap CI for the residual AUC — this is the headline gate number
    # under the seasonal-residual framing.
    res_ci_lo, res_ci_hi = bootstrap_residual_auc_ci(y, s_vwr, season_day, n_iter=1000)
    partial["residual_auc_ci_95"] = [res_ci_lo, res_ci_hi]

    # ── Per-injury-type breakdown ─────────────────────────────────────
    per_type_auc: dict[str, dict] = {}
    if len(pos_df) > 0:
        all_types = pos_df["injury_type"].unique().tolist()
        for t in all_types:
            mask_pos = (all_df["injury_type"] == t) & (all_df["label"] == 1)
            mask_neg = (all_df["label"] == 0)
            sub = all_df[mask_pos | mask_neg]
            if int(sub["label"].sum()) < 2 or int((sub["label"] == 0).sum()) < 2:
                per_type_auc[t] = {"auc": None, "n_pos": int(mask_pos.sum())}
                continue
            r = roc_from_scores(sub["label"].values, sub["vwr"].values)
            per_type_auc[t] = {
                "auc": r["auc"],
                "n_pos": int(mask_pos.sum()),
                "n_neg": int(mask_neg.sum()),
            }

    roc_out = {
        "vwr": {
            "auc": roc_vwr["auc"],
            "auc_ci_95": [ci_lo, ci_hi],
            "tpr": roc_vwr["tpr"],
            "fpr": roc_vwr["fpr"],
            "thresholds": roc_vwr["thresholds"],
            "n_pos": roc_vwr["n_pos"],
            "n_neg": roc_vwr["n_neg"],
        },
        "velocity_drop": {
            "auc": roc_vel["auc"],
            "auc_ci_95": [ci_vel_lo, ci_vel_hi],
            "tpr": roc_vel["tpr"],
            "fpr": roc_vel["fpr"],
            "n_pos": roc_vel["n_pos"],
            "n_neg": roc_vel["n_neg"],
        },
        "delta_auc": delta_auc,
        "seasonal_control": partial,
        "per_injury_type": per_type_auc,
    }
    (RESULTS_DIR / "roc_curve.json").write_text(json.dumps(roc_out, indent=2, default=str))

    # Plotly ROC
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roc_vwr["fpr"], y=roc_vwr["tpr"], mode="lines+markers",
            name=f"VWR (AUC={roc_vwr['auc']:.3f})",
        ))
        fig.add_trace(go.Scatter(
            x=roc_vel["fpr"], y=roc_vel["tpr"], mode="lines+markers",
            name=f"Velocity-drop (AUC={roc_vel['auc']:.3f})",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random",
            line=dict(dash="dash", color="grey"),
        ))
        fig.update_layout(
            title=f"VWR 30-day pre-IL ROC (n_pos={roc_vwr['n_pos']}, n_neg={roc_vwr['n_neg']})",
            xaxis_title="False-positive rate",
            yaxis_title="True-positive rate",
        )
        fig.write_html(RESULTS_DIR / "roc_curve.html")
    except Exception as exc:
        logger.warning("Plotly ROC plot failed: %s", exc)

    # ── Step 5: Lead-time ─────────────────────────────────────────────
    logger.info("Lead-time analysis")
    lead_rows = []
    for pid, daily in injured_daily.items():
        il_date = pd.Timestamp(injured_params[pid]["il_date"])
        mask = (daily["game_date"] >= il_date - pd.Timedelta(days=PRE_IL_WINDOW_DAYS)) & \
               (daily["game_date"] <= il_date)
        sub = daily.loc[mask]
        if sub.empty:
            continue
        rec = {
            "pitcher_id": pid,
            "injury_type": injured_params[pid]["injury_type"],
            "il_date": str(il_date.date()),
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
            fig.add_trace(go.Histogram(x=vals, name=f"VWR>={int(th)}", nbinsx=20, opacity=0.6))
        fig.update_layout(
            barmode="overlay",
            title="VWR Lead-Time Distribution (days before IL placement)",
            xaxis_title="Lead time (days)",
            yaxis_title="Number of pitchers",
        )
        fig.write_html(RESULTS_DIR / "lead_time_distribution.html")
    except Exception as exc:
        logger.warning("Plotly lead-time failed: %s", exc)

    # ── Step 7: FPR on healthy seasons ───────────────────────────────
    logger.info("FPR on healthy seasons")
    fpr_rows = []
    saturation_ever_max = []
    for key, daily in healthy_daily.items():
        pid, season = key
        row = {"pitcher_id": pid, "season": season,
               "max_vwr": float(daily["vwr"].max()) if len(daily) else None}
        saturation_ever_max.append(row["max_vwr"] if row["max_vwr"] is not None else 0.0)
        for th in LEAD_TIME_THRESHOLDS:
            th_int = int(th)
            row[f"pct_days_above_{th_int}"] = float((daily["vwr"] >= th).mean() * 100) if len(daily) else 0.0
            row[f"breached_{th_int}"] = bool((daily["vwr"] >= th).any()) if len(daily) else False
        fpr_rows.append(row)

    fpr_df = pd.DataFrame(fpr_rows)
    fpr_df.to_csv(RESULTS_DIR / "fpr_healthy_seasons.csv", index=False)
    fpr_summary: dict = {"n_healthy_seasons": int(len(fpr_df))}
    for th in LEAD_TIME_THRESHOLDS:
        th_int = int(th)
        col = f"breached_{th_int}"
        fpr_summary[f"pct_breach_{th_int}"] = (
            float(fpr_df[col].mean() * 100) if len(fpr_df) else None
        )
    # Saturation check
    if saturation_ever_max:
        sat_arr = np.array(saturation_ever_max)
        fpr_summary["saturation_pct_ge_85"] = float((sat_arr >= 85.0).mean() * 100)
        fpr_summary["saturation_pct_ge_95"] = float((sat_arr >= 95.0).mean() * 100)
        fpr_summary["healthy_max_vwr_median"] = float(np.median(sat_arr))
    (RESULTS_DIR / "fpr_summary.json").write_text(json.dumps(fpr_summary, indent=2))

    # ── Decision gate verdict ─────────────────────────────────────────
    # Raw-space metrics (kept for transparency / informational).
    auc_pass_raw = (not math.isnan(roc_vwr["auc"])) and roc_vwr["auc"] >= GATE_AUC
    delta_pass_raw = delta_auc is not None and delta_auc >= GATE_DELTA_AUC

    # Headline gates — seasonal-residual framing.
    residual_auc = partial.get("residual_auc")
    residual_delta = partial.get("delta_residual_auc")
    auc_pass = (residual_auc is not None and not math.isnan(residual_auc)
                and residual_auc >= GATE_AUC)
    delta_pass = (residual_delta is not None and not math.isnan(residual_delta)
                  and residual_delta >= GATE_DELTA_AUC)

    median_lead = lead_summary.get(f"lead_time_t{int(OPERATING_THRESHOLD)}", {}).get("median_days")
    lead_pass = median_lead is not None and median_lead >= GATE_LEAD_TIME_DAYS
    fpr_at_op = fpr_summary.get(f"pct_breach_{int(OPERATING_THRESHOLD)}")
    fpr_pass = fpr_at_op is not None and (fpr_at_op / 100.0) <= GATE_FPR

    # Seasonal-control sanity — residual must NOT collapse below raw AUC after
    # partialling out season_day. If it does, the signal is a calendar artifact.
    seasonal_artifact = (residual_auc is not None and not math.isnan(residual_auc)
                         and roc_vwr["auc"] is not None and not math.isnan(roc_vwr["auc"])
                         and residual_auc < roc_vwr["auc"])
    seasonal_sanity_pass = not seasonal_artifact

    passes = sum([auc_pass, lead_pass, fpr_pass, delta_pass, seasonal_sanity_pass])
    if passes == 5:
        verdict = "FLAGSHIP"
    elif passes >= 4:
        verdict = "MARGINAL"
    else:
        verdict = "NOT_FLAGSHIP"

    decision = {
        # Headline (seasonal-residual framing).
        "auc_residual_gate": {
            "value": residual_auc,
            "ci_95": partial.get("residual_auc_ci_95"),
            "threshold": GATE_AUC,
            "pass": auc_pass,
            "framing": "seasonal-residual (linear partial-out of season_day)",
        },
        "delta_auc_residual_gate": {
            "value": residual_delta,
            "vwr_residual_auc": residual_auc,
            "veldrop_residual_auc": partial.get("compare_residual_auc"),
            "threshold": GATE_DELTA_AUC,
            "pass": delta_pass,
            "framing": "delta(residual_auc): VWR vs velocity-drop, both partialled-out of season_day",
        },
        "lead_time_gate": {
            "value": median_lead,
            "threshold": GATE_LEAD_TIME_DAYS,
            "pass": lead_pass,
            "operating_threshold": OPERATING_THRESHOLD,
        },
        "fpr_gate": {
            "value_pct": fpr_at_op,
            "threshold_pct": GATE_FPR * 100,
            "pass": fpr_pass,
        },
        "seasonal_sanity_gate": {
            "raw_auc": roc_vwr["auc"],
            "residual_auc": residual_auc,
            "season_day_alone_auc": partial.get("season_day_auc"),
            "rule": "residual_auc must NOT be lower than raw_auc",
            "pass": seasonal_sanity_pass,
        },
        # Raw-space numbers (informational / transparency).
        "auc_raw": {
            "value": roc_vwr["auc"],
            "ci_95": [ci_lo, ci_hi],
            "threshold": GATE_AUC,
            "pass": auc_pass_raw,
            "note": "informational — superseded by seasonal-residual headline gate",
        },
        "delta_auc_raw": {
            "value": delta_auc,
            "threshold": GATE_DELTA_AUC,
            "pass": delta_pass_raw,
            "note": "informational — superseded by delta-in-residual-space gate",
        },
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
    p.add_argument("--max-pitchers", type=int, default=None)
    p.add_argument("--n-healthy", type=int, default=300)
    p.add_argument("--skip-fit", action="store_true")
    args = p.parse_args()
    summary = run_pipeline(
        max_pitchers=args.max_pitchers,
        n_healthy=args.n_healthy,
        skip_fit=args.skip_fit,
    )
    d = summary["decision_gate"]
    print(json.dumps({
        "verdict": d["verdict"],
        "auc_residual": d["auc_residual_gate"]["value"],
        "auc_residual_ci": d["auc_residual_gate"].get("ci_95"),
        "delta_auc_residual": d["delta_auc_residual_gate"]["value"],
        "median_lead": d["lead_time_gate"]["value"],
        "fpr_pct": d["fpr_gate"]["value_pct"],
        "auc_raw": d["auc_raw"]["value"],
        "delta_auc_raw": d["delta_auc_raw"]["value"],
        "season_day_auc": summary["roc"]["seasonal_control"].get("season_day_auc"),
        "joint_auc": summary["roc"]["seasonal_control"].get("joint_auc"),
        "veldrop_residual_auc": summary["roc"]["seasonal_control"].get("compare_residual_auc"),
        "n_injured": summary["cohort"]["n_injured_fit"],
        "n_healthy": summary["cohort"]["n_healthy_fit"],
        "wall_min": summary["wall_clock_minutes"],
    }, indent=2, default=str))
