"""VWR 2025 out-of-sample holdout validation.

Builds on ``scripts/viscoelastic_workload_roc_analysis.py`` but extends the
cohort beyond the original 64 fits:

    1. Expanded training cohort: all qualifying arm-injured MLB pitchers
       whose IL date is in 2017-2024 (inclusive). Same eligibility rules:
       >= 500 pitches before (il_date - 60 days).

    2. 2025 holdout: arm-injured pitchers whose IL date is in 2025. Their
       VWR parameters are fit on pitches STRICTLY PRIOR to (il_date - 60d).
       In practice, because the holdout IL is in 2025, most pre-IL-60
       pitches come from <= 2024 data. No leakage from the holdout season
       into the fit. A healthy control cohort from 2024-2025 is drawn for
       the ROC negatives.

    3. Gates are computed the same way as the 2026-04-18 seasonal-residual
       framing (gate headlines are residual-space).

Artifacts land in ``results/viscoelastic_workload/2025_holdout/``.

Read-only DB, seeds at 42.
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

from scripts.viscoelastic_workload_roc_analysis import (  # noqa: E402
    ARM_TYPES,
    CHECKPOINTS_DIR,
    FIT_BUFFER_DAYS,
    GATE_AUC,
    GATE_DELTA_AUC,
    GATE_FPR,
    GATE_LEAD_TIME_DAYS,
    HEALTHY_FIT_WINDOW_DAYS,
    HEALTHY_MIN_PITCHES,
    LEAD_TIME_THRESHOLDS,
    MIN_FIT_PITCHES,
    OPERATING_THRESHOLD,
    PRE_IL_WINDOW_DAYS,
    ROC_WINDOW_DAYS,
    SEED,
    bootstrap_auc_ci,
    bootstrap_residual_auc_ci,
    compute_daily_vwr,
    daily_velocity_score,
    fetch_pitches,
    fit_parameters_from_df,
    first_breach_day,
    partial_out_auc,
    roc_from_scores,
    season_day_from_date,
    set_seeds,
)
from src.analytics.viscoelastic_workload import (  # noqa: E402
    _compute_time_deltas,
    compute_pitch_stress,
    compute_strain_state,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vwr_2025_holdout")


DB_PATH = ROOT / "data" / "baseball.duckdb"
LABELS_PATH = ROOT / "data" / "injury_labels.parquet"
RESULTS_DIR = ROOT / "results" / "viscoelastic_workload" / "2025_holdout"
CHECKPOINTS_DIR_EXPANDED = ROOT / "models" / "viscoelastic" / "per_pitcher"

TRAIN_YEARS = (2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024)
HOLDOUT_YEAR = 2025
HEALTHY_CONTROL_SEASONS = (2024, 2025)  # for holdout negatives


def _connect_db_readonly(retries: int = 6, base_delay: float = 1.0) -> duckdb.DuckDBPyConnection:
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


def load_cohort_window(start_year: int, end_year: int) -> pd.DataFrame:
    df = pd.read_parquet(LABELS_PATH)
    arm = df[df["injury_type"].isin(ARM_TYPES)].copy()
    arm["il_date"] = pd.to_datetime(arm["il_date"])
    arm = arm.dropna(subset=["pitcher_id", "il_date"]).copy()
    arm["pitcher_id"] = arm["pitcher_id"].astype(int)
    arm = arm.sort_values("il_date").drop_duplicates("pitcher_id", keep="first")
    arm = arm[
        (arm["il_date"] >= f"{start_year}-01-01")
        & (arm["il_date"] <= f"{end_year}-12-31")
    ].copy()
    return arm.reset_index(drop=True)


def fetch_healthy_pool(
    conn: duckdb.DuckDBPyConnection,
    injured_ids: set[int],
    seasons: tuple[int, ...],
    min_pitches: int = HEALTHY_MIN_PITCHES,
) -> list[tuple[int, int]]:
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


def fit_injured_cohort(
    conn: duckdb.DuckDBPyConnection,
    cohort: pd.DataFrame,
    cache_prefix: str,
) -> tuple[dict[int, dict], list[dict]]:
    """Fit VWR params for each injured pitcher on pre-IL-60 pitches.

    Writes checkpoint pickles under ``models/viscoelastic/per_pitcher/``
    with the given cache_prefix (e.g. "inj2025h"). Reuses if present.
    """
    CHECKPOINTS_DIR_EXPANDED.mkdir(parents=True, exist_ok=True)

    fits: dict[int, dict] = {}
    skipped: list[dict] = []
    t0 = time.time()

    for idx, row in cohort.iterrows():
        pid = int(row["pitcher_id"])
        il_date = pd.Timestamp(row["il_date"])
        fit_cutoff = il_date - pd.Timedelta(days=FIT_BUFFER_DAYS)

        ck_path = CHECKPOINTS_DIR_EXPANDED / f"{cache_prefix}_{pid}.pkl"
        if ck_path.exists():
            try:
                with open(ck_path, "rb") as f:
                    payload = pickle.load(f)
                fits[pid] = payload
                continue
            except Exception:
                # fall through to re-fit
                pass

        try:
            fit_df = fetch_pitches(conn, pid, date_lt=fit_cutoff)
        except Exception as exc:
            skipped.append({"pitcher_id": pid, "injury_type": row["injury_type"],
                            "reason": "fetch_failed", "detail": str(exc)})
            continue

        if len(fit_df) < MIN_FIT_PITCHES:
            skipped.append({"pitcher_id": pid, "injury_type": row["injury_type"],
                            "reason": "insufficient_fit_pitches",
                            "n_fit": int(len(fit_df))})
            continue

        params = fit_parameters_from_df(fit_df, pid)
        if "reason" in params:
            skipped.append({"pitcher_id": pid, "injury_type": row["injury_type"],
                            "reason": params["reason"]})
            continue

        stress = compute_pitch_stress(fit_df).values
        tds = _compute_time_deltas(fit_df)
        career_strain = compute_strain_state(
            stress, tds, params["E1"], params["E2"], params["tau"],
        )

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
        fits[pid] = payload

        if (idx + 1) % 20 == 0:
            logger.info(
                "Injured fit: %d done (skipped=%d), %.1f min elapsed",
                len(fits), len(skipped), (time.time() - t0) / 60,
            )
    logger.info(
        "Injured cohort fit complete: n=%d, skipped=%d, wall=%.1f min",
        len(fits), len(skipped), (time.time() - t0) / 60,
    )
    return fits, skipped


def fit_healthy_cohort(
    conn: duckdb.DuckDBPyConnection,
    healthy_sample: list[tuple[int, int]],
    cache_prefix: str,
) -> tuple[dict[tuple[int, int], dict], list[dict]]:
    CHECKPOINTS_DIR_EXPANDED.mkdir(parents=True, exist_ok=True)
    fits: dict[tuple[int, int], dict] = {}
    skipped: list[dict] = []
    t0 = time.time()
    rng_h = np.random.RandomState(SEED + 1)

    for i, (pid, season) in enumerate(healthy_sample):
        ck_path = CHECKPOINTS_DIR_EXPANDED / f"{cache_prefix}_{pid}_{season}.pkl"
        if ck_path.exists():
            try:
                with open(ck_path, "rb") as f:
                    payload = pickle.load(f)
                fits[(pid, season)] = payload
                continue
            except Exception:
                pass

        start = pd.Timestamp(f"{season}-01-01")
        end = pd.Timestamp(f"{season}-12-31")
        try:
            season_df = fetch_pitches(conn, pid, date_ge=start, date_le=end)
        except Exception as exc:
            skipped.append({"pitcher_id": pid, "season": season,
                            "reason": "fetch_failed", "detail": str(exc)})
            continue
        if len(season_df) < HEALTHY_MIN_PITCHES:
            skipped.append({"pitcher_id": pid, "season": season,
                            "reason": "insufficient_season",
                            "n": int(len(season_df))})
            continue

        params = fit_parameters_from_df(season_df, pid)
        if "reason" in params:
            skipped.append({"pitcher_id": pid, "season": season,
                            "reason": params["reason"]})
            continue

        stress = compute_pitch_stress(season_df).values
        tds = _compute_time_deltas(season_df)
        career_strain = compute_strain_state(
            stress, tds, params["E1"], params["E2"], params["tau"],
        )

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
        fits[(pid, season)] = payload

        if (i + 1) % 50 == 0:
            logger.info(
                "Healthy fit: %d/%d done, %.1f min elapsed",
                i + 1, len(healthy_sample), (time.time() - t0) / 60,
            )
    logger.info(
        "Healthy cohort fit complete: n=%d, skipped=%d, wall=%.1f min",
        len(fits), len(skipped), (time.time() - t0) / 60,
    )
    return fits, skipped


def compute_trajectories(
    conn: duckdb.DuckDBPyConnection,
    injured_params: dict[int, dict],
    healthy_params: dict[tuple[int, int], dict],
) -> tuple[dict, dict, dict, dict]:
    """Return (injured_daily, injured_vel, healthy_daily, healthy_vel)."""
    injured_daily: dict[int, pd.DataFrame] = {}
    injured_vel: dict[int, pd.DataFrame] = {}
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
        injured_vel[pid] = dvel

    healthy_daily: dict[tuple[int, int], pd.DataFrame] = {}
    healthy_vel: dict[tuple[int, int], pd.DataFrame] = {}
    for key, payload in healthy_params.items():
        try:
            eval_df = fetch_pitches(
                conn, payload["pitcher_id"],
                date_ge=payload["eval_start"], date_le=payload["eval_end"],
            )
        except Exception:
            continue
        if eval_df.empty:
            continue
        daily = compute_daily_vwr(eval_df, payload["params"], payload["career_strain"])
        if daily.empty:
            continue
        dvel = daily_velocity_score(eval_df)
        healthy_daily[key] = daily
        healthy_vel[key] = dvel

    return injured_daily, injured_vel, healthy_daily, healthy_vel


def build_score_table(
    injured_params: dict[int, dict],
    injured_daily: dict[int, pd.DataFrame],
    injured_vel: dict[int, pd.DataFrame],
    healthy_params: dict[tuple[int, int], dict],
    healthy_daily: dict[tuple[int, int], pd.DataFrame],
    healthy_vel: dict[tuple[int, int], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (pos_df, neg_df, all_df) per 2026-04-18 gate framing."""
    pos_rows = []
    for pid, daily in injured_daily.items():
        il_date = pd.Timestamp(injured_params[pid]["il_date"])
        mask = (
            (daily["game_date"] >= il_date - pd.Timedelta(days=ROC_WINDOW_DAYS))
            & (daily["game_date"] <= il_date)
        )
        vel = injured_vel[pid]
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
    for key, daily in healthy_daily.items():
        vel = healthy_vel[key]
        sub = daily.merge(vel, on="game_date", how="left").fillna({"vel_score": 0.0})
        for _, r in sub.iterrows():
            neg_rows.append({
                "pitcher_id": key[0],
                "game_date": r["game_date"],
                "vwr": float(r["vwr"]),
                "vel_score": float(r["vel_score"]),
                "season_day": season_day_from_date(r["game_date"]),
                "injury_type": "healthy",
                "label": 0,
            })

    pos_df = pd.DataFrame(pos_rows)
    neg_df = pd.DataFrame(neg_rows)

    # Balance negatives 10:1 if needed
    rng_neg = np.random.RandomState(SEED)
    n_pos = int(pos_df["label"].sum()) if not pos_df.empty else 0
    if len(neg_df) > n_pos * 10 and n_pos > 0:
        neg_df = neg_df.sample(n=n_pos * 10, random_state=rng_neg).reset_index(drop=True)
    all_df = pd.concat([pos_df, neg_df], ignore_index=True)
    return pos_df, neg_df, all_df


def run_gates(all_df: pd.DataFrame, pos_df: pd.DataFrame) -> dict:
    y = all_df["label"].values.astype(int)
    s_vwr = all_df["vwr"].values.astype(float)
    s_vel = all_df["vel_score"].values.astype(float)
    season_day = all_df["season_day"].values.astype(float)

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

    delta_auc = (
        None if math.isnan(roc_vwr["auc"]) or math.isnan(roc_vel["auc"])
        else roc_vwr["auc"] - roc_vel["auc"]
    )

    partial = partial_out_auc(y, s_vwr, season_day, score_compare=s_vel_scaled)
    res_ci_lo, res_ci_hi = bootstrap_residual_auc_ci(y, s_vwr, season_day, n_iter=1000)
    partial["residual_auc_ci_95"] = [res_ci_lo, res_ci_hi]

    per_type_auc: dict[str, dict] = {}
    if not pos_df.empty:
        for t in pos_df["injury_type"].unique().tolist():
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

    return {
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


def run_lead_time(
    injured_params: dict[int, dict],
    injured_daily: dict[int, pd.DataFrame],
) -> tuple[pd.DataFrame, dict]:
    lead_rows = []
    for pid, daily in injured_daily.items():
        il_date = pd.Timestamp(injured_params[pid]["il_date"])
        mask = (
            (daily["game_date"] >= il_date - pd.Timedelta(days=PRE_IL_WINDOW_DAYS))
            & (daily["game_date"] <= il_date)
        )
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
    summary: dict = {}
    for th in LEAD_TIME_THRESHOLDS:
        key = f"lead_time_t{int(th)}"
        vals = lead_df[key].dropna()
        summary[key] = {
            "n_breached": int(len(vals)),
            "n_total": int(len(lead_df)),
            "fraction_breached": float(len(vals) / max(len(lead_df), 1)),
            "median_days": float(vals.median()) if len(vals) else None,
            "mean_days": float(vals.mean()) if len(vals) else None,
            "p25_days": float(vals.quantile(0.25)) if len(vals) else None,
            "p75_days": float(vals.quantile(0.75)) if len(vals) else None,
        }
    return lead_df, summary


def run_fpr(healthy_daily: dict[tuple[int, int], pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    fpr_rows = []
    sat = []
    for (pid, season), daily in healthy_daily.items():
        row = {
            "pitcher_id": pid, "season": season,
            "max_vwr": float(daily["vwr"].max()) if len(daily) else None,
        }
        sat.append(row["max_vwr"] if row["max_vwr"] is not None else 0.0)
        for th in LEAD_TIME_THRESHOLDS:
            th_int = int(th)
            row[f"pct_days_above_{th_int}"] = (
                float((daily["vwr"] >= th).mean() * 100) if len(daily) else 0.0
            )
            row[f"breached_{th_int}"] = (
                bool((daily["vwr"] >= th).any()) if len(daily) else False
            )
        fpr_rows.append(row)
    fpr_df = pd.DataFrame(fpr_rows)
    summary: dict = {"n_healthy_seasons": int(len(fpr_df))}
    for th in LEAD_TIME_THRESHOLDS:
        th_int = int(th)
        col = f"breached_{th_int}"
        summary[f"pct_breach_{th_int}"] = (
            float(fpr_df[col].mean() * 100) if len(fpr_df) else None
        )
    if sat:
        sat_arr = np.array(sat)
        summary["saturation_pct_ge_85"] = float((sat_arr >= 85.0).mean() * 100)
        summary["saturation_pct_ge_95"] = float((sat_arr >= 95.0).mean() * 100)
        summary["healthy_max_vwr_median"] = float(np.median(sat_arr))
    return fpr_df, summary


def build_decision(
    roc_out: dict,
    lead_summary: dict,
    fpr_summary: dict,
) -> dict:
    partial = roc_out["seasonal_control"]
    roc_vwr = roc_out["vwr"]
    raw_auc = roc_vwr["auc"]
    delta_raw = roc_out["delta_auc"]
    residual_auc = partial.get("residual_auc")
    residual_delta = partial.get("delta_residual_auc")

    auc_pass = (
        residual_auc is not None and not math.isnan(residual_auc)
        and residual_auc >= GATE_AUC
    )
    delta_pass = (
        residual_delta is not None and not math.isnan(residual_delta)
        and residual_delta >= GATE_DELTA_AUC
    )
    auc_pass_raw = (not math.isnan(raw_auc)) and raw_auc >= GATE_AUC
    delta_pass_raw = delta_raw is not None and delta_raw >= GATE_DELTA_AUC

    median_lead = lead_summary.get(
        f"lead_time_t{int(OPERATING_THRESHOLD)}", {}
    ).get("median_days")
    lead_pass = median_lead is not None and median_lead >= GATE_LEAD_TIME_DAYS
    fpr_at_op = fpr_summary.get(f"pct_breach_{int(OPERATING_THRESHOLD)}")
    fpr_pass = fpr_at_op is not None and (fpr_at_op / 100.0) <= GATE_FPR

    seasonal_artifact = (
        residual_auc is not None and not math.isnan(residual_auc)
        and raw_auc is not None and not math.isnan(raw_auc)
        and residual_auc < raw_auc
    )
    seasonal_sanity_pass = not seasonal_artifact
    passes = sum([auc_pass, lead_pass, fpr_pass, delta_pass, seasonal_sanity_pass])
    if passes == 5:
        verdict = "FLAGSHIP"
    elif passes >= 4:
        verdict = "MARGINAL"
    else:
        verdict = "NOT_FLAGSHIP"

    return {
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
            "raw_auc": raw_auc,
            "residual_auc": residual_auc,
            "season_day_alone_auc": partial.get("season_day_auc"),
            "rule": "residual_auc must NOT be lower than raw_auc",
            "pass": seasonal_sanity_pass,
        },
        "auc_raw": {
            "value": raw_auc,
            "ci_95": roc_vwr["auc_ci_95"],
            "threshold": GATE_AUC,
            "pass": auc_pass_raw,
        },
        "delta_auc_raw": {
            "value": delta_raw,
            "threshold": GATE_DELTA_AUC,
            "pass": delta_pass_raw,
        },
        "verdict": verdict,
    }


# ──────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────


def run_pipeline(
    max_train_pitchers: int | None = None,
    max_holdout_pitchers: int | None = None,
    n_healthy_holdout: int = 400,
) -> dict:
    set_seeds(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    conn = _connect_db_readonly()

    # ── Step 1: cohorts ───────────────────────────────────────────────
    logger.info("Loading expanded training cohort 2017-2024")
    train_cohort = load_cohort_window(TRAIN_YEARS[0], TRAIN_YEARS[-1])
    logger.info("Training arm-injury cohort 2017-2024: n=%d", len(train_cohort))
    if max_train_pitchers is not None:
        train_cohort = train_cohort.head(max_train_pitchers).copy()

    logger.info("Loading 2025 holdout cohort")
    holdout_cohort = load_cohort_window(HOLDOUT_YEAR, HOLDOUT_YEAR)
    logger.info("Holdout arm-injury cohort 2025: n=%d", len(holdout_cohort))
    if max_holdout_pitchers is not None:
        holdout_cohort = holdout_cohort.head(max_holdout_pitchers).copy()

    # ── Step 2: fit injured cohorts ─────────────────────────────────
    logger.info("Fitting VWR params: expanded training cohort 2017-2024")
    train_fits, train_skipped = fit_injured_cohort(
        conn, train_cohort, cache_prefix="inj_ext",
    )

    logger.info("Fitting VWR params: 2025 holdout cohort (pre-IL-60 only)")
    holdout_fits, holdout_skipped = fit_injured_cohort(
        conn, holdout_cohort, cache_prefix="inj_2025h",
    )

    # ── Step 3: healthy control cohort for 2025 holdout ─────────────
    injured_ids_holdout = set(holdout_cohort["pitcher_id"].tolist())
    logger.info("Fetching healthy pool for 2024-2025")
    healthy_pool = fetch_healthy_pool(
        conn, injured_ids_holdout,
        seasons=HEALTHY_CONTROL_SEASONS,
        min_pitches=HEALTHY_MIN_PITCHES,
    )
    rng = np.random.RandomState(SEED)
    rng.shuffle(healthy_pool)
    healthy_sample = healthy_pool[:n_healthy_holdout]
    logger.info("Healthy holdout sample: %d", len(healthy_sample))

    logger.info("Fitting VWR params: healthy controls 2024-2025")
    healthy_fits, healthy_skipped = fit_healthy_cohort(
        conn, healthy_sample, cache_prefix="hlt_2425",
    )

    # Training coverage
    training_coverage = {
        "train_cohort_n": int(len(train_cohort)),
        "train_fits": len(train_fits),
        "train_skipped": len(train_skipped),
        "train_skipped_reasons": (
            pd.DataFrame(train_skipped)["reason"].value_counts().to_dict()
            if train_skipped else {}
        ),
        "train_per_injury_type": (
            pd.DataFrame([{"injury_type": p["injury_type"]} for p in train_fits.values()])
              ["injury_type"].value_counts().to_dict()
            if train_fits else {}
        ),
        "train_per_year": (
            pd.DataFrame([{"yr": pd.Timestamp(p["il_date"]).year} for p in train_fits.values()])
              ["yr"].value_counts().sort_index().to_dict()
            if train_fits else {}
        ),
        "holdout_cohort_n": int(len(holdout_cohort)),
        "holdout_fits": len(holdout_fits),
        "holdout_skipped": len(holdout_skipped),
        "holdout_skipped_reasons": (
            pd.DataFrame(holdout_skipped)["reason"].value_counts().to_dict()
            if holdout_skipped else {}
        ),
        "holdout_per_injury_type": (
            pd.DataFrame([{"injury_type": p["injury_type"]} for p in holdout_fits.values()])
              ["injury_type"].value_counts().to_dict()
            if holdout_fits else {}
        ),
        "healthy_pool_n": len(healthy_pool),
        "healthy_sample_n": len(healthy_sample),
        "healthy_fits": len(healthy_fits),
        "healthy_skipped": len(healthy_skipped),
        "healthy_control_seasons": list(HEALTHY_CONTROL_SEASONS),
        "wall_fit_minutes": round((time.time() - t0) / 60, 2),
    }
    (RESULTS_DIR / "training_coverage.json").write_text(
        json.dumps(training_coverage, indent=2, default=str)
    )

    # ── Step 4: trajectories + gates for 2025 holdout ──────────────────
    logger.info("Computing holdout trajectories")
    holdout_daily, holdout_vel, healthy_daily, healthy_vel = compute_trajectories(
        conn, holdout_fits, healthy_fits,
    )
    logger.info("Holdout trajectories: %d injured, %d healthy",
                len(holdout_daily), len(healthy_daily))

    pos_df, neg_df, all_df = build_score_table(
        holdout_fits, holdout_daily, holdout_vel,
        healthy_fits, healthy_daily, healthy_vel,
    )
    all_df.to_csv(RESULTS_DIR / "roc_daily_scores.csv", index=False)

    roc_out = run_gates(all_df, pos_df)
    (RESULTS_DIR / "roc_curve.json").write_text(
        json.dumps(roc_out, indent=2, default=str)
    )

    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roc_out["vwr"]["fpr"], y=roc_out["vwr"]["tpr"], mode="lines+markers",
            name=f"VWR (AUC={roc_out['vwr']['auc']:.3f})",
        ))
        fig.add_trace(go.Scatter(
            x=roc_out["velocity_drop"]["fpr"], y=roc_out["velocity_drop"]["tpr"],
            mode="lines+markers",
            name=f"Velocity-drop (AUC={roc_out['velocity_drop']['auc']:.3f})",
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="Random",
            line=dict(dash="dash", color="grey"),
        ))
        fig.update_layout(
            title=(
                f"VWR 2025 holdout ROC "
                f"(n_pos={roc_out['vwr']['n_pos']}, n_neg={roc_out['vwr']['n_neg']})"
            ),
            xaxis_title="False-positive rate",
            yaxis_title="True-positive rate",
        )
        fig.write_html(RESULTS_DIR / "roc_curve.html")
    except Exception as exc:
        logger.warning("Plotly ROC plot failed: %s", exc)

    lead_df, lead_summary = run_lead_time(holdout_fits, holdout_daily)
    lead_df.to_csv(RESULTS_DIR / "lead_time_per_pitcher.csv", index=False)
    (RESULTS_DIR / "lead_time_distribution.json").write_text(
        json.dumps(lead_summary, indent=2)
    )
    try:
        import plotly.graph_objects as go
        fig = go.Figure()
        for th in LEAD_TIME_THRESHOLDS:
            vals = lead_df[f"lead_time_t{int(th)}"].dropna().values
            if len(vals) == 0:
                continue
            fig.add_trace(go.Histogram(x=vals, name=f"VWR>={int(th)}",
                                       nbinsx=20, opacity=0.6))
        fig.update_layout(
            barmode="overlay",
            title="VWR 2025 holdout lead-time distribution",
            xaxis_title="Lead time (days)",
            yaxis_title="Number of pitchers",
        )
        fig.write_html(RESULTS_DIR / "lead_time_distribution.html")
    except Exception as exc:
        logger.warning("Plotly lead-time failed: %s", exc)

    fpr_df, fpr_summary = run_fpr(healthy_daily)
    fpr_df.to_csv(RESULTS_DIR / "fpr_healthy_seasons.csv", index=False)
    (RESULTS_DIR / "fpr_summary.json").write_text(json.dumps(fpr_summary, indent=2))

    decision = build_decision(roc_out, lead_summary, fpr_summary)

    summary = {
        "cohort": training_coverage,
        "roc": roc_out,
        "lead_time": lead_summary,
        "fpr": fpr_summary,
        "decision_gate": decision,
        "wall_clock_minutes": round((time.time() - t0) / 60, 2),
        "meta": {
            "train_years": list(TRAIN_YEARS),
            "holdout_year": HOLDOUT_YEAR,
            "healthy_control_seasons": list(HEALTHY_CONTROL_SEASONS),
            "training_note": (
                "Expanded training cohort 2017-2024 produced per-pitcher "
                "fit checkpoints but these were NOT mixed with the 2025 "
                "holdout ROC computation; holdout fits use pre-IL-60 "
                "pitches only (no leakage). Training cohort gates are "
                "reported in the 'training_gates' section below."
            ),
        },
    }

    # ── Also run gates on the expanded TRAINING cohort for robustness ──
    # (does the residual-AUC result replicate on the 64->~1000 expansion?)
    logger.info("Building training-cohort gates (2017-2024 + same 2024-25 healthy controls)")
    train_daily, train_vel, _, _ = compute_trajectories(conn, train_fits, {})
    if train_daily:
        # Reuse the holdout healthy cohort for negatives (both sampled later
        # in the decade; avoids doubling fit cost).
        pos_df_t, neg_df_t, all_df_t = build_score_table(
            train_fits, train_daily, train_vel,
            healthy_fits, healthy_daily, healthy_vel,
        )
        roc_out_t = run_gates(all_df_t, pos_df_t)
        lead_df_t, lead_summary_t = run_lead_time(train_fits, train_daily)
        # FPR uses the same healthy pool, so reuse fpr_summary
        decision_t = build_decision(roc_out_t, lead_summary_t, fpr_summary)
        summary["training_gates"] = {
            "roc": roc_out_t,
            "lead_time": lead_summary_t,
            "fpr": fpr_summary,
            "decision_gate": decision_t,
            "n_injured_fit": len(train_fits),
        }
        lead_df_t.to_csv(RESULTS_DIR / "lead_time_per_pitcher_TRAIN.csv", index=False)
        all_df_t.to_csv(RESULTS_DIR / "roc_daily_scores_TRAIN.csv", index=False)

    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    conn.close()
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--max-train-pitchers", type=int, default=None)
    p.add_argument("--max-holdout-pitchers", type=int, default=None)
    p.add_argument("--n-healthy-holdout", type=int, default=400)
    args = p.parse_args()
    summary = run_pipeline(
        max_train_pitchers=args.max_train_pitchers,
        max_holdout_pitchers=args.max_holdout_pitchers,
        n_healthy_holdout=args.n_healthy_holdout,
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
        "n_holdout_injured": summary["cohort"]["holdout_fits"],
        "n_train_injured": summary["cohort"]["train_fits"],
        "n_healthy": summary["cohort"]["healthy_fits"],
        "wall_min": summary["wall_clock_minutes"],
        "training_verdict": summary.get("training_gates", {}).get("decision_gate", {}).get("verdict"),
        "training_residual_auc": summary.get("training_gates", {}).get("decision_gate", {}).get("auc_residual_gate", {}).get("value"),
    }, indent=2, default=str))
