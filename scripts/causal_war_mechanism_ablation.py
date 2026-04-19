#!/usr/bin/env python
"""CausalWAR Mechanism Ablation: Does the RELIEVER LEVERAGE GAP tag reflect a
causal role-awareness mechanism, or an epiphenomenon?

Strategy
--------
We mask out the single role-indicator feature in the CausalWAR confounder
matrix -- ``inning_bucket`` (1-3 / 4-6 / 7-9 / 10+) -- which is the model's
only direct proxy for reliever vs. starter context.  ``inning_bucket`` is
how the per-PA residual model "knows" that a late-inning PA belongs to a
reliever; zeroing it out strips the model's ability to residualise against
role-specific game state without touching the base-out / handedness / park
confounders.

The downstream mechanism-tag classifier is UNCHANGED.  It still reads
``position == pitcher`` and ``ip_total < 60`` from the Fangraphs WAR staging
table; those are inputs to the TAGGER, not the MODEL.  This isolates the
question: does RELIEVER LEVERAGE GAP's 78% hit rate come from the model's
role-awareness, or from the tag definition's reliever-filter acting as a
spurious sorting criterion?

Comparison to baseline
----------------------
* Baseline: ``models/causal_war/causal_war_trainsplit_2015_2022.pkl``
  trained with full 12-feature W (including ``inning_bucket``).
* Ablated: ``models/causal_war/causal_war_trainsplit_2015_2022_noRP.pkl``
  trained on the same 2015-2022 PA data with ``inning_bucket`` set to 0.

Three windows evaluated (2022->23, 2023->24, 2024->25), same hit rule as
``scripts/causal_war_contrarian_stability.py``.  Bootstrap 95% CIs over
per-year-pooled pick lists, 1000 resamples.

Outputs (``results/causal_war/mechanism_ablation/``):
    ablated_buy_low_{year}_to_{year+1}.csv       (3 files)
    ablated_over_valued_{year}_to_{year+1}.csv   (3 files)
    ablation_comparison.json
    report.md
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics.causal_war import (  # noqa: E402
    _aggregate_player_effects,
    _build_features,
    _extract_pa_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation")


# ---------------------------------------------------------------------------
# Constants (mirror contrarian_stability)
# ---------------------------------------------------------------------------

TOP_N = 25
N_BOOTSTRAP = 1000
PA_MIN = 100
IP_MIN = 20
RANDOM_STATE = 42

_DEFENSE_FIRST_POSITIONS = {"SS", "CF", "C", "2B"}
_TAG_RELIEVER = "RELIEVER LEVERAGE GAP"
_TAG_PARK = "PARK FACTOR"
_TAG_DEFENSE = "DEFENSE GAP"
_TAG_GENUINE = "GENUINE EDGE?"
_TAG_OTHER = "OTHER"

# Feature to ablate: the sole role-indicator in the CausalWAR confounder matrix.
# `_build_features` produces W with columns in this order:
#   [venue_code, platoon, on_1b, on_2b, on_3b, outs, inning_bucket,
#    if_shift, of_shift, month, stand_R, p_throws_R]
# Index 6 == inning_bucket.
ROLE_FEATURE_INDEX = 6
ROLE_FEATURE_NAME = "inning_bucket"

# Windows to evaluate
WINDOWS: list[tuple[int, int]] = [(2022, 2023), (2023, 2024), (2024, 2025)]

# Training split (must match the committed baseline)
TRAIN_START = 2015
TRAIN_END = 2022

# Nuisance model hyperparameters (match CausalWARConfig defaults)
NUISANCE_PARAMS: dict[str, Any] = dict(
    max_iter=300,
    max_depth=6,
    learning_rate=0.05,
    min_samples_leaf=20,
    l2_regularization=1.0,
    random_state=RANDOM_STATE,
)


# ---------------------------------------------------------------------------
# Role-feature ablation
# ---------------------------------------------------------------------------

def ablate_role_feature(W: np.ndarray) -> np.ndarray:
    """Zero-out the role-indicator column in the confounder matrix.

    The existing downstream code path expects a 12-column W; we keep the
    shape so nothing else has to change.  Zeroing rather than dropping also
    keeps the fitted ``HistGradientBoostingRegressor``'s ``n_features_in_``
    consistent across train/test -- it just learns that the column has no
    discriminative information.
    """
    if W.shape[1] <= ROLE_FEATURE_INDEX:
        raise ValueError(
            f"W has only {W.shape[1]} columns; expected >= {ROLE_FEATURE_INDEX+1}."
        )
    W_ablated = W.copy()
    W_ablated[:, ROLE_FEATURE_INDEX] = 0.0
    return W_ablated


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_ablated_nuisance(conn: duckdb.DuckDBPyConnection) -> tuple[Any, dict[str, Any]]:
    """Re-train the CausalWAR outcome nuisance model on 2015-2022 PAs with
    ``inning_bucket`` zeroed out.  Mirrors the ``train_test_split`` path in
    ``src/analytics/causal_war.py`` -- we fit a single full-train model so
    the artifact is drop-in comparable to the baseline pickle.
    """
    logger.info("Extracting train PAs %d-%d ...", TRAIN_START, TRAIN_END)
    train_df = _extract_pa_data(conn, year_range=(TRAIN_START, TRAIN_END))
    logger.info("  %d train PAs", len(train_df))

    W, Y, _, _ = _build_features(train_df)
    logger.info("  Baseline W shape: %s", W.shape)

    W_ablated = ablate_role_feature(W)
    logger.info(
        "  Ablated W shape: %s (zeroed column %d=%s)",
        W_ablated.shape, ROLE_FEATURE_INDEX, ROLE_FEATURE_NAME,
    )

    logger.info("Fitting ablated HistGradientBoostingRegressor ...")
    model = HistGradientBoostingRegressor(**NUISANCE_PARAMS)
    model.fit(W_ablated, Y)

    # Training diagnostics (in-sample -- only for comparability with baseline
    # artifact; true evaluation uses the window-level hit rates below).
    Y_pred = model.predict(W_ablated)
    from sklearn.metrics import mean_squared_error, r2_score
    train_r2 = float(r2_score(Y, Y_pred))
    train_rmse = float(np.sqrt(mean_squared_error(Y, Y_pred)))

    train_metrics = {
        "train_nuisance_r2": round(train_r2, 4),
        "train_rmse_residuals": round(train_rmse, 6),
        "n_observations": int(len(Y)),
        "train_start_year": TRAIN_START,
        "train_end_year": TRAIN_END,
        "ablated_feature_index": ROLE_FEATURE_INDEX,
        "ablated_feature_name": ROLE_FEATURE_NAME,
    }
    logger.info("  Ablated in-sample R2: %.4f (baseline was 0.0009)", train_r2)
    return model, train_metrics


def save_artifact(model: Any, train_metrics: dict[str, Any]) -> Path:
    out_path = ROOT / "models" / "causal_war" / "causal_war_trainsplit_2015_2022_noRP.pkl"
    artifact = {
        "nuisance_outcome": model,
        "train_split": (TRAIN_START, TRAIN_END),
        "train_metrics": train_metrics,
        "ablated_feature_index": ROLE_FEATURE_INDEX,
        "ablated_feature_name": ROLE_FEATURE_NAME,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(artifact, out_path)
    logger.info("Saved ablated artifact -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Reused helpers (copied verbatim from contrarian_stability to keep the
# downstream pipeline identical)
# ---------------------------------------------------------------------------

def load_fg_war() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data" / "fangraphs_war_staging.parquet")


def load_position_lookup(conn: duckdb.DuckDBPyConnection) -> dict[int, str]:
    try:
        rows = conn.execute(
            "SELECT player_id, position FROM players WHERE position IS NOT NULL"
        ).fetchall()
        return {int(pid): str(pos) for pid, pos in rows if pos}
    except Exception:
        return {}


def load_player_names(conn: duckdb.DuckDBPyConnection, ids: Iterable[int]) -> dict[int, str]:
    ids = list({int(p) for p in ids})
    if not ids:
        return {}
    ph = ",".join(["?"] * len(ids))
    rows = conn.execute(
        f"SELECT player_id, full_name FROM players WHERE player_id IN ({ph})", ids,
    ).fetchall()
    return {int(pid): str(name) for pid, name in rows if name}


def known_pitcher_ids(conn: duckdb.DuckDBPyConnection, season: int) -> set[int]:
    try:
        rows = conn.execute(
            "SELECT DISTINCT player_id FROM season_pitching_stats "
            "WHERE season = $1 AND ip >= 5",
            [int(season)],
        ).fetchdf()
        return set(int(p) for p in rows["player_id"].tolist())
    except Exception:
        return set()


def compute_causal_effects_for_season_ablated(
    conn: duckdb.DuckDBPyConnection,
    nuisance_model: Any,
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the *ablated* nuisance model to a single season's PAs.  We zero
    out the role-indicator column on the prediction side too so the ablated
    model never sees inning info.
    """
    pa_df = _extract_pa_data(conn, year_range=(season, season))
    W, Y, player_ids, pa_df_built = _build_features(pa_df)
    W_ablated = ablate_role_feature(W)
    Y_pred = nuisance_model.predict(W_ablated)
    Y_res = Y - Y_pred

    batter_effects = _aggregate_player_effects(Y_res, player_ids, pa_df_built, pa_min=10)
    bat_rows = [
        {"player_id": int(p), "causal_war": float(e["causal_war"]),
         "point_estimate_per_pa": float(e["point_estimate_per_pa"]),
         "pa": int(e["pa"])}
        for p, e in batter_effects.items()
    ]
    bat_df = pd.DataFrame(bat_rows)

    pitcher_ids = pa_df_built["pitcher_id"].to_numpy().astype(int)
    pitcher_effects = _aggregate_player_effects(-Y_res, pitcher_ids, pa_df_built, pa_min=50)
    pit_rows = [
        {"player_id": int(p), "causal_war": float(e["causal_war"]),
         "point_estimate_per_pa": float(e["point_estimate_per_pa"]),
         "pa": int(e["pa"])}
        for p, e in pitcher_effects.items()
    ]
    pit_df = pd.DataFrame(pit_rows)

    return bat_df, pit_df


def merge_with_bwar(
    bat_effects: pd.DataFrame,
    pit_effects: pd.DataFrame,
    fg: pd.DataFrame,
    baseline_year: int,
    pitcher_ids: set[int],
    names: dict[int, str],
) -> pd.DataFrame:
    fg_year = fg[fg["season"] == baseline_year].copy()
    fg_bat = fg_year[fg_year["position_type"] == "batter"][["player_id", "war", "pa_or_ip"]].rename(
        columns={"war": "trad_war", "pa_or_ip": "pa_total"}
    )
    fg_pit = fg_year[fg_year["position_type"] == "pitcher"][["player_id", "war", "pa_or_ip"]].rename(
        columns={"war": "trad_war", "pa_or_ip": "ip_total"}
    )

    if bat_effects.empty:
        merged_bat = pd.DataFrame()
    else:
        merged_bat = bat_effects.merge(fg_bat, on="player_id", how="left")
        merged_bat["position"] = "batter"
        merged_bat["ip_total"] = np.nan
        is_pitcher_only = (
            merged_bat["player_id"].isin(pitcher_ids)
            & (merged_bat["pa_total"].fillna(0).astype(float) < PA_MIN)
        )
        merged_bat = merged_bat[~is_pitcher_only].copy()

    if pit_effects.empty:
        merged_pit = pd.DataFrame()
    else:
        merged_pit = pit_effects.merge(fg_pit, on="player_id", how="left")
        merged_pit["position"] = "pitcher"
        merged_pit["pa_total"] = np.nan
        merged_pit = merged_pit[merged_pit["player_id"].isin(pitcher_ids)].copy()

    merged = pd.concat([merged_bat, merged_pit], ignore_index=True, sort=False)
    pa_tot = merged["pa_total"].fillna(0).astype(float)
    ip_tot = merged["ip_total"].fillna(0).astype(float)
    qualifies = (
        ((merged["position"] == "batter") & (pa_tot >= PA_MIN))
        | ((merged["position"] == "pitcher") & (ip_tot >= IP_MIN))
    )
    merged = merged[qualifies].copy()
    merged = merged.dropna(subset=["trad_war", "causal_war"]).copy()
    if merged.empty:
        return merged.reset_index(drop=True)

    merged["rank_causal"] = (
        merged.groupby("position")["causal_war"]
        .rank(ascending=False, method="min").astype(int)
    )
    merged["rank_trad"] = (
        merged.groupby("position")["trad_war"]
        .rank(ascending=False, method="min").astype(int)
    )
    merged["rank_diff"] = merged["rank_trad"] - merged["rank_causal"]
    merged["name"] = merged["player_id"].map(lambda p: names.get(int(p), f"pid_{p}"))
    merged["baseline_year"] = baseline_year
    return merged.reset_index(drop=True)


def fetch_followup_outcomes(
    conn: duckdb.DuckDBPyConnection,
    player_ids: Iterable[int],
    season: int,
) -> dict[int, dict]:
    out: dict[int, dict] = {}
    ids = list({int(p) for p in player_ids})
    if not ids:
        return out
    ph = ",".join(["?"] * len(ids))
    try:
        bat = conn.execute(
            f"SELECT player_id, war, pa, ops FROM season_batting_stats "
            f"WHERE season = {int(season)} AND player_id IN ({ph})", ids,
        ).fetchall()
        for pid, war, pa, ops in bat:
            out[int(pid)] = {
                "war": float(war) if war is not None else None,
                "pa": int(pa) if pa is not None else None,
                "ops": float(ops) if ops is not None else None,
                "ip": None, "era": None, "source": "bat",
            }
    except Exception:
        pass

    try:
        pit = conn.execute(
            f"SELECT player_id, war, ip, era FROM season_pitching_stats "
            f"WHERE season = {int(season)} AND player_id IN ({ph})", ids,
        ).fetchall()
        for pid, war, ip, era in pit:
            rec = out.get(int(pid))
            if rec is None:
                out[int(pid)] = {
                    "war": float(war) if war is not None else None,
                    "pa": None, "ops": None,
                    "ip": float(ip) if ip is not None else None,
                    "era": float(era) if era is not None else None,
                    "source": "pit",
                }
            else:
                rec["ip"] = float(ip) if ip is not None else rec.get("ip")
                rec["era"] = float(era) if era is not None else rec.get("era")
                if rec.get("war") is None and war is not None:
                    rec["war"] = float(war)
    except Exception:
        pass
    return out


def row_hit_verdict(row: pd.Series, outcomes: dict[int, dict], *, side: str) -> tuple[bool | None, str]:
    info = outcomes.get(int(row["player_id"]))
    if info is None:
        return None, "no_followup_record"
    war = info.get("war")
    baseline_war = float(row["trad_war"])
    if war is not None:
        if side == "buy_low":
            return bool(war >= baseline_war), "war_delta"
        return bool(war < baseline_war), "war_delta"
    if row["position"] == "pitcher":
        ip = info.get("ip"); era = info.get("era")
        if ip is None or era is None or ip < 30:
            return None, "insufficient_followup_pitcher"
        if side == "buy_low":
            return bool(era <= 4.00), "era_surrogate"
        return bool(era > 4.00), "era_surrogate"
    pa = info.get("pa"); ops = info.get("ops")
    if pa is None or ops is None or pa < 100:
        return None, "insufficient_followup_batter"
    if side == "buy_low":
        return bool(ops >= 0.700), "ops_surrogate"
    return bool(ops < 0.700), "ops_surrogate"


def classify_row(row: pd.Series, position_lookup: dict[int, str]) -> str:
    is_pitcher = row.get("position") == "pitcher"
    is_batter = row.get("position") == "batter"
    causal = float(row.get("causal_war", 0.0) or 0.0)
    trad = float(row.get("trad_war", 0.0) or 0.0)
    ip = row.get("ip_total"); pa = row.get("pa_total")
    ip = float(ip) if pd.notna(ip) else None
    pa = float(pa) if pd.notna(pa) else None
    fpos = position_lookup.get(int(row["player_id"]), None)
    if is_pitcher:
        if ip is not None and ip < 60 and causal > trad:
            return _TAG_RELIEVER
        if ip is not None and ip >= 60 and trad > causal:
            return _TAG_PARK
    if is_batter:
        if fpos in _DEFENSE_FIRST_POSITIONS and trad > causal:
            return _TAG_DEFENSE
        if pa is not None and pa >= 400 and causal > trad:
            return _TAG_GENUINE
    return _TAG_OTHER


def bootstrap_ci_mean(
    values: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
    ci: float = 0.95,
) -> tuple[float, float]:
    rng = np.random.RandomState(random_state)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    n = len(values)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = float(np.mean(values[rng.randint(0, n, size=n)]))
    alpha = (1 - ci) / 2
    return float(np.percentile(boots, 100 * alpha)), float(np.percentile(boots, 100 * (1 - alpha)))


def bootstrap_ci_delta(
    ablated_hits: np.ndarray,
    baseline_hits: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    random_state: int = RANDOM_STATE,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Bootstrap CI on (ablated_rate - baseline_rate) where each array is
    a 0/1 Bernoulli sample (same picks can differ between conditions).
    Resamples independently because the ablated and baseline pick lists are
    distinct.
    """
    rng = np.random.RandomState(random_state)
    if len(ablated_hits) == 0 or len(baseline_hits) == 0:
        return (float("nan"), float("nan"))
    na, nb = len(ablated_hits), len(baseline_hits)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        a = ablated_hits[rng.randint(0, na, size=na)]
        b = baseline_hits[rng.randint(0, nb, size=nb)]
        deltas[i] = float(np.mean(a)) - float(np.mean(b))
    alpha = (1 - ci) / 2
    return float(np.percentile(deltas, 100 * alpha)), float(np.percentile(deltas, 100 * (1 - alpha)))


# ---------------------------------------------------------------------------
# Per-window analysis
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    baseline_year: int
    followup_year: int
    buy_low_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    over_valued_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    buy_low_stats: dict[str, Any] = field(default_factory=dict)
    over_valued_stats: dict[str, Any] = field(default_factory=dict)
    buy_low_mechanism: dict[str, dict[str, Any]] = field(default_factory=dict)
    over_valued_mechanism: dict[str, dict[str, Any]] = field(default_factory=dict)


def analyze_window(
    conn: duckdb.DuckDBPyConnection,
    nuisance_model: Any,
    fg: pd.DataFrame,
    position_lookup: dict[int, str],
    baseline_year: int,
    followup_year: int,
) -> WindowResult:
    logger.info("Window %d -> %d", baseline_year, followup_year)

    bat_eff, pit_eff = compute_causal_effects_for_season_ablated(conn, nuisance_model, baseline_year)

    pitcher_ids = known_pitcher_ids(conn, baseline_year)
    all_ids = set(bat_eff["player_id"]) | set(pit_eff["player_id"])
    names = load_player_names(conn, all_ids)

    merged = merge_with_bwar(
        bat_eff, pit_eff, fg,
        baseline_year=baseline_year,
        pitcher_ids=pitcher_ids,
        names=names,
    )
    if merged.empty:
        return WindowResult(baseline_year, followup_year)

    buy_low = merged.sort_values("rank_diff", ascending=False).head(TOP_N).copy()
    over_valued = merged.sort_values("rank_diff", ascending=True).head(TOP_N).copy()

    buy_low["tag"] = buy_low.apply(lambda r: classify_row(r, position_lookup), axis=1)
    over_valued["tag"] = over_valued.apply(lambda r: classify_row(r, position_lookup), axis=1)

    followup_ids = set(buy_low["player_id"]) | set(over_valued["player_id"])
    outcomes = fetch_followup_outcomes(conn, followup_ids, followup_year)

    def evaluate(df: pd.DataFrame, side: str) -> pd.DataFrame:
        rows = []
        for _, r in df.iterrows():
            hit, basis = row_hit_verdict(r, outcomes, side=side)
            rows.append({**r.to_dict(), "hit": hit, "basis": basis})
        return pd.DataFrame(rows)

    buy_low_eval = evaluate(buy_low, "buy_low")
    over_valued_eval = evaluate(over_valued, "over_valued")

    def summarize(df: pd.DataFrame, side: str) -> dict[str, Any]:
        eval_df = df[df["hit"].notna()].copy()
        n = len(eval_df)
        hits = int(eval_df["hit"].sum()) if n > 0 else 0
        rate = hits / n if n > 0 else None
        if n > 0:
            lo, hi = bootstrap_ci_mean(eval_df["hit"].astype(int).to_numpy())
        else:
            lo, hi = (float("nan"), float("nan"))
        return {
            "side": side, "baseline_year": baseline_year, "followup_year": followup_year,
            "n_leaderboard": int(len(df)), "n_evaluated": int(n),
            "hits": hits, "hit_rate": rate,
            "ci_95_low": lo, "ci_95_high": hi,
        }

    def tag_breakdown(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        out = {}
        for tag, group in df.groupby("tag"):
            evald = group[group["hit"].notna()]
            n = len(evald)
            hits = int(evald["hit"].sum()) if n > 0 else 0
            out[tag] = {
                "n_in_leaderboard": int(len(group)),
                "n_evaluated": int(n),
                "hits": hits,
                "hit_rate": (hits / n) if n > 0 else None,
            }
        return out

    bl_stats = summarize(buy_low_eval, "buy_low")
    ov_stats = summarize(over_valued_eval, "over_valued")

    logger.info("  Buy-Low: %d/%d = %.3f  |  Over-Valued: %d/%d = %.3f",
                bl_stats["hits"], bl_stats["n_evaluated"], bl_stats["hit_rate"] or 0,
                ov_stats["hits"], ov_stats["n_evaluated"], ov_stats["hit_rate"] or 0)

    return WindowResult(
        baseline_year=baseline_year, followup_year=followup_year,
        buy_low_table=buy_low_eval, over_valued_table=over_valued_eval,
        buy_low_stats=bl_stats, over_valued_stats=ov_stats,
        buy_low_mechanism=tag_breakdown(buy_low_eval),
        over_valued_mechanism=tag_breakdown(over_valued_eval),
    )


# ---------------------------------------------------------------------------
# Baseline (committed) mechanism hit rates
# ---------------------------------------------------------------------------

def load_baseline_mechanism() -> list[dict[str, Any]]:
    """Load the committed 3-year baseline mechanism breakdown."""
    path = ROOT / "results" / "causal_war" / "contrarian_stability" / "mechanism_breakdown.json"
    return json.loads(path.read_text(encoding="utf-8"))


def pool_mechanism(
    per_window: list[dict[str, Any]],
    side_key: str,
    tag: str,
) -> tuple[int, int, float | None, np.ndarray]:
    """Pool per-window mechanism counts across three years and return
    (hits, n, rate, per_pick_hit_array).
    """
    hits = 0
    n = 0
    per_pick: list[int] = []
    for w in per_window:
        tags = w.get(side_key, {})
        stat = tags.get(tag, None)
        if stat is None:
            continue
        hits += int(stat["hits"])
        n += int(stat["n_evaluated"])
        # Reconstruct 0/1 vector for bootstrap
        per_pick.extend([1] * int(stat["hits"]))
        per_pick.extend([0] * (int(stat["n_evaluated"]) - int(stat["hits"])))
    rate = (hits / n) if n > 0 else None
    return hits, n, rate, np.array(per_pick, dtype=int)


def write_comparison_json(
    outdir: Path,
    baseline_mech: list[dict[str, Any]],
    results: list[WindowResult],
) -> dict[str, Any]:
    """Compute and write the ablation_comparison.json payload."""

    # ---- Ablated per-window mechanism breakdown in the same shape as ------
    ablated_mech = []
    for r in results:
        ablated_mech.append({
            "baseline_year": r.baseline_year,
            "followup_year": r.followup_year,
            "buy_low_tags": r.buy_low_mechanism,
            "over_valued_tags": r.over_valued_mechanism,
        })

    # ---- Pool per-tag hit rates across the 3 windows ----------------------
    tags_of_interest = [
        ("buy_low_tags", _TAG_RELIEVER, "Buy-Low"),
        ("over_valued_tags", _TAG_PARK, "Over-Valued"),
        ("over_valued_tags", _TAG_DEFENSE, "Over-Valued"),
    ]

    comparison = {}
    for side_key, tag, side_label in tags_of_interest:
        b_hits, b_n, b_rate, b_vec = pool_mechanism(baseline_mech, side_key, tag)
        a_hits, a_n, a_rate, a_vec = pool_mechanism(ablated_mech, side_key, tag)
        delta = (a_rate - b_rate) if (a_rate is not None and b_rate is not None) else None
        # Bootstrap CI on delta
        lo, hi = bootstrap_ci_delta(a_vec, b_vec) if (a_n > 0 and b_n > 0) else (float("nan"), float("nan"))
        comparison[f"{tag} ({side_label})"] = {
            "baseline": {"hits": b_hits, "n": b_n, "rate": b_rate},
            "ablated":  {"hits": a_hits, "n": a_n, "rate": a_rate},
            "delta": delta,
            "delta_ci_95": [lo, hi],
            "per_window_baseline": [
                {"year": w["baseline_year"],
                 **(w.get(side_key, {}).get(tag, {}))}
                for w in baseline_mech
            ],
            "per_window_ablated": [
                {"year": w["baseline_year"],
                 **(w.get(side_key, {}).get(tag, {}))}
                for w in ablated_mech
            ],
        }

    payload = {
        "ablation_description": (
            "Zeroed out the role-indicator feature (inning_bucket, index 6) in "
            "the CausalWAR outcome nuisance confounder matrix for both training "
            "(2015-2022) and inference (per-season prediction)."
        ),
        "ablated_feature_index": ROLE_FEATURE_INDEX,
        "ablated_feature_name": ROLE_FEATURE_NAME,
        "train_split": [TRAIN_START, TRAIN_END],
        "n_bootstrap": N_BOOTSTRAP,
        "random_state": RANDOM_STATE,
        "per_window_ablated": ablated_mech,
        "per_window_baseline": baseline_mech,
        "mechanism_tag_comparison": comparison,
    }
    with (outdir / "ablation_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Wrote ablation_comparison.json")
    return payload


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(outdir: Path, payload: dict[str, Any], results: list[WindowResult]) -> None:
    comp = payload["mechanism_tag_comparison"]
    lines: list[str] = []
    lines.append("# CausalWAR Mechanism Ablation: RELIEVER LEVERAGE GAP -- Causal Mechanism or Epiphenomenon?")
    lines.append("")
    lines.append("## Question")
    lines.append("")
    lines.append("The committed CausalWAR contrarian-leaderboard analysis "
                 "reports three mechanism-tag hit rates over three consecutive "
                 "year windows (2022->23, 2023->24, 2024->25):")
    lines.append("")
    lines.append("| Tag | Side | Hits / n | Rate |")
    lines.append("|---|---|---|---|")
    for label, val in comp.items():
        b = val["baseline"]
        side = "Buy-Low" if "Buy-Low" in label else "Over-Valued"
        tag = label.replace(f" ({side})", "")
        rate = f"{b['rate']*100:.1f}%" if b['rate'] is not None else "N/A"
        lines.append(f"| {tag} | {side} | {b['hits']}/{b['n']} | {rate} |")
    lines.append("")
    lines.append("The methodology paper currently phrases the claim as the tag "
                 "patterns being \"consistent with a causal driver.\" This "
                 "ablation tests the stronger claim -- that the RELIEVER "
                 "LEVERAGE GAP tag reflects a real role-awareness mechanism "
                 "in the model -- by retraining CausalWAR with the only direct "
                 "role-indicator feature (``inning_bucket``) zeroed out.")
    lines.append("")
    lines.append("## What was ablated")
    lines.append("")
    lines.append("The CausalWAR outcome nuisance model `E[Y|W]` is a "
                 "`HistGradientBoostingRegressor` fit on a 12-column confounder "
                 "matrix `W` defined in `src/analytics/causal_war._build_features`:")
    lines.append("")
    lines.append("`[venue_code, platoon, on_1b, on_2b, on_3b, outs, "
                 "inning_bucket, if_shift, of_shift, month, stand_R, p_throws_R]`")
    lines.append("")
    lines.append("Of these, only ``inning_bucket`` (values 0-3 mapping to "
                 "1-3 / 4-6 / 7-9 / 10+ innings) serves as an implicit role "
                 "indicator. Starters predominantly face batters in innings "
                 "1-6; relievers in 7+. There is no explicit ``position == "
                 "RP`` or ``role_reliever`` feature in `W`. Rate stats that "
                 "only relievers have (saves, holds) are not in `W` either -- "
                 "`W` is constructed at the PA grain, not the player grain.")
    lines.append("")
    lines.append(f"We set `W[:, {ROLE_FEATURE_INDEX}] = 0` for all training "
                 "PAs (2015-2022) and re-fit a single full-train "
                 "`HistGradientBoostingRegressor` with the same hyperparameters "
                 "as the committed baseline. The same ablation is applied at "
                 "inference time on each 2022 / 2023 / 2024 season.")
    lines.append("")
    lines.append("**The tag classifier is UNCHANGED.** It still reads "
                 "`position == \"pitcher\"` and `ip_total < 60` from the "
                 "Fangraphs WAR staging table to assign the RELIEVER LEVERAGE "
                 "GAP label. Those signals come from outside the CausalWAR "
                 "model. The ablation isolates whether the model's *internal* "
                 "residualisation depends on knowing it's looking at a reliever.")
    lines.append("")
    lines.append("Artifact: "
                 "`models/causal_war/causal_war_trainsplit_2015_2022_noRP.pkl`.")
    lines.append("")
    lines.append("## Three-year pooled comparison")
    lines.append("")
    lines.append("| Mechanism tag | Original | Ablated | Delta | 95% CI (bootstrap) |")
    lines.append("|---|---|---|---|---|")
    for label, val in comp.items():
        b = val["baseline"]; a = val["ablated"]
        br = f"{b['rate']*100:.1f}% ({b['hits']}/{b['n']})" if b['rate'] is not None else "N/A"
        ar = f"{a['rate']*100:.1f}% ({a['hits']}/{a['n']})" if a['rate'] is not None else "N/A"
        d = val["delta"]
        dr = f"{d*100:+.1f}pp" if d is not None else "N/A"
        lo, hi = val["delta_ci_95"]
        ci = (f"[{lo*100:+.1f}pp, {hi*100:+.1f}pp]"
              if not (np.isnan(lo) or np.isnan(hi)) else "N/A")
        lines.append(f"| {label} | {br} | {ar} | {dr} | {ci} |")
    lines.append("")

    # Verdict
    rel = comp.get(f"{_TAG_RELIEVER} (Buy-Low)", {})
    park = comp.get(f"{_TAG_PARK} (Over-Valued)", {})
    dfn = comp.get(f"{_TAG_DEFENSE} (Over-Valued)", {})
    rel_delta = rel.get("delta") or 0
    park_delta = park.get("delta") or 0
    dfn_delta = dfn.get("delta") or 0
    rel_ablated = rel.get("ablated", {}).get("rate") or 0
    park_ablated = park.get("ablated", {}).get("rate") or 0
    dfn_ablated = dfn.get("ablated", {}).get("rate") or 0

    # Simple rule-of-thumb verdicts
    reliever_collapsed = rel_ablated <= 0.55 and rel_delta <= -0.15
    park_held = abs(park_delta) < 0.10
    defense_held = abs(dfn_delta) < 0.10

    if reliever_collapsed and (park_held and defense_held):
        verdict = ("DEMONSTRATED -- RELIEVER LEVERAGE GAP collapsed toward the "
                   "50% baseline while PARK FACTOR and DEFENSE GAP held.")
    elif reliever_collapsed and not (park_held and defense_held):
        verdict = ("AMBIGUOUS -- RELIEVER LEVERAGE GAP collapsed, but PARK / "
                   "DEFENSE also shifted materially. Role-indicator is "
                   "implicated but not cleanly isolated.")
    elif (rel_delta <= -0.05) and (park_delta <= -0.05) and (dfn_delta <= -0.05):
        verdict = ("FALSIFIED -- all three tags collapsed together, suggesting "
                   "the tags captured an overall CausalWAR edge rather than "
                   "a mechanism-specific role signal.")
    elif abs(rel_delta) < 0.05:
        verdict = ("UNSUPPORTED -- RELIEVER LEVERAGE GAP did not collapse when "
                   "role info was removed. The tag's predictive validity does "
                   "not come from the model's role-awareness.")
    else:
        verdict = "AMBIGUOUS -- partial / wide CIs; see per-window table above."

    lines.append(f"## Verdict: **{verdict}**")
    lines.append("")
    lines.append("## Per-window detail")
    lines.append("")
    lines.append("| Window | Side | Tag | Original | Ablated |")
    lines.append("|---|---|---|---|---|")
    baseline_mech = payload["per_window_baseline"]
    ablated_mech = payload["per_window_ablated"]
    for bw, aw in zip(baseline_mech, ablated_mech):
        year = bw["baseline_year"]
        for side_key, side_label in (("buy_low_tags", "Buy-Low"), ("over_valued_tags", "Over-Valued")):
            bt = bw.get(side_key, {})
            at = aw.get(side_key, {})
            all_tags = sorted(set(bt.keys()) | set(at.keys()))
            for tag in all_tags:
                bstat = bt.get(tag, {})
                astat = at.get(tag, {})
                brate = bstat.get("hit_rate"); arate = astat.get("hit_rate")
                bhits = bstat.get("hits", 0); bn = bstat.get("n_evaluated", 0)
                ahits = astat.get("hits", 0); an = astat.get("n_evaluated", 0)
                br = f"{brate*100:.0f}% ({bhits}/{bn})" if brate is not None else "-"
                ar = f"{arate*100:.0f}% ({ahits}/{an})" if arate is not None else "-"
                lines.append(f"| {year}->{year+1} | {side_label} | {tag} | {br} | {ar} |")
    lines.append("")
    lines.append("## How to interpret the delta CIs")
    lines.append("")
    lines.append(f"Bootstrap: {N_BOOTSTRAP} resamples at the per-pick-hit level "
                 "(0/1 Bernoulli vector), drawn independently for the ablated "
                 "and baseline cohorts since the pick lists can differ. The "
                 "reported CI is the 2.5 / 97.5 percentile of "
                 "`ablated_rate - baseline_rate`. Deltas whose CI excludes 0 "
                 "are significant at the 5% level.")
    lines.append("")
    lines.append("## Ground rules followed")
    lines.append("")
    lines.append(f"- Single retraining of CausalWAR with `{ROLE_FEATURE_NAME}` masked. No architecture or hyperparameter changes.")
    lines.append("- Same 2015-2022 train cohort as the committed baseline checkpoint; no resampling.")
    lines.append("- Tag classifier is bit-identical to `scripts/causal_war_contrarian_stability.py`.")
    lines.append(f"- Bootstrap n={N_BOOTSTRAP}, random_state={RANDOM_STATE}.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `models/causal_war/causal_war_trainsplit_2015_2022_noRP.pkl`")
    lines.append(f"- `results/causal_war/mechanism_ablation/ablated_buy_low_*.csv` (3 files)")
    lines.append(f"- `results/causal_war/mechanism_ablation/ablated_over_valued_*.csv` (3 files)")
    lines.append(f"- `results/causal_war/mechanism_ablation/ablation_comparison.json`")
    lines.append(f"- `results/causal_war/mechanism_ablation/report.md` (this file)")

    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote report.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    outdir = ROOT / "results" / "causal_war" / "mechanism_ablation"
    outdir.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(str(ROOT / "data" / "baseball.duckdb"), read_only=True)
    try:
        # 1. Train ablated nuisance
        model, train_metrics = train_ablated_nuisance(conn)
        save_artifact(model, train_metrics)

        # 2. Eval each window
        fg = load_fg_war()
        position_lookup = load_position_lookup(conn)

        results: list[WindowResult] = []
        for baseline, followup in WINDOWS:
            r = analyze_window(conn, model, fg, position_lookup, baseline, followup)
            results.append(r)
            if not r.buy_low_table.empty:
                r.buy_low_table.to_csv(
                    outdir / f"ablated_buy_low_{baseline}_to_{followup}.csv", index=False,
                )
            if not r.over_valued_table.empty:
                r.over_valued_table.to_csv(
                    outdir / f"ablated_over_valued_{baseline}_to_{followup}.csv", index=False,
                )
    finally:
        conn.close()

    # 3. Load the committed baseline and build comparison
    baseline_mech = load_baseline_mechanism()
    payload = write_comparison_json(outdir, baseline_mech, results)
    write_report(outdir, payload, results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
