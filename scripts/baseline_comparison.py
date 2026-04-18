#!/usr/bin/env python
"""
CausalWAR baseline comparison vs traditional WAR.

Trains the CausalWAR DML model on a configurable year range, evaluates
per-player effects on a disjoint test window, and benchmarks the estimates
against a traditional WAR proxy derived from ``season_batting_stats`` and
``season_pitching_stats``.

Implements Ticket #2 from ``docs/models/causal_war_validation_spec.md``.

Usage
-----
    python scripts/baseline_comparison.py \
        --train-start 2015 --train-end 2022 \
        --test-start 2023 --test-end 2024 \
        --output-dir results/

Outputs
-------
    results/causal_war_baseline_comparison_{test_start}_{test_end}.csv
    results/causal_war_baseline_comparison_{test_start}_{test_end}_metrics.json
    results/causal_war_baseline_scatter.html  (if plotly is installed)

Data-quality note
-----------------
The production schema carries a ``war`` column on both season tables, but in
the current DB every row is NULL.  As a pragmatic stand-in we compute a
**proxy traditional WAR** from populated columns:

    batter_war_proxy = (ops - lg_avg_ops) * (pa / 600) * scale_batter
    pitcher_war_proxy = (lg_avg_era - era) / 9.0 * ip / _RUNS_PER_WIN

The proxy is documented and the script will automatically switch to the
real ``war`` column if it becomes populated in the future.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import duckdb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics.causal_war import CausalWARConfig, CausalWARModel  # noqa: E402
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("baseline_comparison")

if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True,
    )


# ---------------------------------------------------------------------------
# Constants: traditional-WAR proxy
# ---------------------------------------------------------------------------

# Runs per win (standard sabermetrics constant)
_RUNS_PER_WIN: float = 10.0

# Batter scaling: maps `(ops - lg_ops) * (pa / 600)` into a WAR-like unit.
# Tuned so that an .800-OPS full-time batter against a league-avg baseline
# of ~.720 lands near ~3-4 WAR, matching bWAR / fWAR scale for league-avg
# defenders.
_BATTER_WAR_SCALE: float = 45.0


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class BaselineComparisonConfig:
    """Configuration for the baseline comparison run."""

    train_start: int = 2015
    train_end: int = 2022
    test_start: int = 2023
    test_end: int = 2024
    output_dir: Path = ROOT / "results"
    n_bootstrap_train: int = 50
    n_bootstrap_ci: int = 1000
    pa_min: int = 100
    ip_min: int = 20
    random_state: int = 42


# ---------------------------------------------------------------------------
# Traditional-WAR proxy
# ---------------------------------------------------------------------------

def _fetch_batter_traditional_war(
    conn: duckdb.DuckDBPyConnection,
    test_start: int,
    test_end: int,
) -> pd.DataFrame:
    """Return per-player traditional batter WAR (averaged across the test
    window, weighted by PA).

    If the real ``war`` column is populated we use it; otherwise we fall
    back to an OPS-based proxy.
    """
    df = conn.execute(
        """
        SELECT player_id, season, pa, ab, h, doubles, triples, hr, bb, so,
               ba, obp, slg, ops, woba, wrc_plus, war, iso
        FROM season_batting_stats
        WHERE season BETWEEN $1 AND $2
        """,
        [int(test_start), int(test_end)],
    ).fetchdf()
    if df.empty:
        return df

    has_real_war = df["war"].notna().any()
    has_woba = df["woba"].notna().any()

    # League-average OPS for each season, PA-weighted over qualifying batters.
    if has_woba:
        lg = (
            df[df["pa"] >= 300]
            .groupby("season")
            .apply(lambda s: np.average(s["woba"].fillna(s["woba"].mean()), weights=s["pa"]))
            .reset_index()
            .rename(columns={0: "lg_woba"})
        )
        df = df.merge(lg, on="season", how="left")
    else:
        df["lg_woba"] = np.nan

    lg_ops = (
        df[df["pa"] >= 300]
        .groupby("season")
        .apply(lambda s: np.average(s["ops"].fillna(s["ops"].mean()), weights=s["pa"]))
        .reset_index()
        .rename(columns={0: "lg_ops"})
    )
    df = df.merge(lg_ops, on="season", how="left")

    if has_real_war:
        df["trad_war_season"] = df["war"]
        war_source = "season_batting_stats.war"
    else:
        # OPS-based proxy -- documented in the module docstring.
        df["trad_war_season"] = (
            (df["ops"].astype(float) - df["lg_ops"].astype(float))
            * (df["pa"].astype(float) / 600.0)
            * _BATTER_WAR_SCALE
        )
        war_source = "proxy_from_ops_and_pa"

    # PA-weighted average across the test years (multiple seasons -> one number).
    def _weighted_mean(g: pd.DataFrame) -> pd.Series:
        total_pa = g["pa"].sum()
        if total_pa <= 0 or g["trad_war_season"].isna().all():
            return pd.Series({"trad_war": np.nan, "pa_total": total_pa, "n_seasons": len(g)})
        valid = g.dropna(subset=["trad_war_season"])
        if valid["pa"].sum() <= 0:
            return pd.Series({"trad_war": np.nan, "pa_total": total_pa, "n_seasons": len(g)})
        return pd.Series({
            "trad_war": float(
                np.average(valid["trad_war_season"], weights=valid["pa"])
            ),
            "pa_total": int(total_pa),
            "n_seasons": int(len(g)),
        })

    out = df.groupby("player_id", group_keys=False).apply(_weighted_mean).reset_index()
    out["war_source"] = war_source
    out["position_type"] = "batter"
    return out


def _fetch_pitcher_traditional_war(
    conn: duckdb.DuckDBPyConnection,
    test_start: int,
    test_end: int,
) -> pd.DataFrame:
    """Return per-player traditional pitcher WAR (IP-weighted across test window)."""
    df = conn.execute(
        """
        SELECT player_id, season, g, gs, ip, era, whip, k_pct, bb_pct,
               hr_per_9, war
        FROM season_pitching_stats
        WHERE season BETWEEN $1 AND $2
        """,
        [int(test_start), int(test_end)],
    ).fetchdf()
    if df.empty:
        return df

    has_real_war = df["war"].notna().any()

    # League-average ERA per season, IP-weighted over qualifying pitchers.
    lg_era = (
        df[df["ip"] >= 50]
        .groupby("season")
        .apply(lambda s: np.average(s["era"].fillna(s["era"].mean()), weights=s["ip"]))
        .reset_index()
        .rename(columns={0: "lg_era"})
    )
    df = df.merge(lg_era, on="season", how="left")

    if has_real_war:
        df["trad_war_season"] = df["war"]
        war_source = "season_pitching_stats.war"
    else:
        # Classical "runs above replacement" proxy:
        #   runs_saved = (lg_era - era) / 9 * ip
        # Divide by runs-per-win to get WAR-scale output.
        df["trad_war_season"] = (
            (df["lg_era"].astype(float) - df["era"].astype(float))
            / 9.0
            * df["ip"].astype(float)
            / _RUNS_PER_WIN
        )
        war_source = "proxy_from_era_and_ip"

    def _weighted_mean(g: pd.DataFrame) -> pd.Series:
        total_ip = g["ip"].sum()
        if total_ip <= 0 or g["trad_war_season"].isna().all():
            return pd.Series({"trad_war": np.nan, "ip_total": total_ip, "n_seasons": len(g)})
        valid = g.dropna(subset=["trad_war_season"])
        if valid["ip"].sum() <= 0:
            return pd.Series({"trad_war": np.nan, "ip_total": total_ip, "n_seasons": len(g)})
        return pd.Series({
            "trad_war": float(
                np.average(valid["trad_war_season"], weights=valid["ip"])
            ),
            "ip_total": float(total_ip),
            "n_seasons": int(len(g)),
        })

    out = df.groupby("player_id", group_keys=False).apply(_weighted_mean).reset_index()
    out["war_source"] = war_source
    out["position_type"] = "pitcher"
    return out


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    causal: np.ndarray,
    trad: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compute Pearson r, Spearman rho, RMSE, MAE with bootstrap CIs.

    Returns
    -------
    dict with keys ``n``, ``pearson_r``, ``pearson_r_ci``, ``spearman_rho``,
    ``spearman_rho_ci``, ``rmse``, ``mae``.  Fields are ``None`` when the
    sample is too small to compute a meaningful correlation.
    """
    causal = np.asarray(causal, dtype=float)
    trad = np.asarray(trad, dtype=float)
    mask = np.isfinite(causal) & np.isfinite(trad)
    causal = causal[mask]
    trad = trad[mask]
    n = int(len(causal))

    result: dict[str, Any] = {
        "n": n,
        "pearson_r": None,
        "pearson_r_ci": None,
        "spearman_rho": None,
        "spearman_rho_ci": None,
        "rmse": None,
        "mae": None,
    }
    if n < 3:
        return result
    if np.std(causal) == 0 or np.std(trad) == 0:
        return result

    r = float(np.corrcoef(causal, trad)[0, 1])
    rho = float(_spearman_rho(causal, trad))
    rmse = float(np.sqrt(np.mean((causal - trad) ** 2)))
    mae = float(np.mean(np.abs(causal - trad)))

    result.update({
        "pearson_r": round(r, 4),
        "spearman_rho": round(rho, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
    })

    # Bootstrap CI on r and rho (resampling pairs).
    rng = np.random.RandomState(random_state)
    r_boots: list[float] = []
    rho_boots: list[float] = []
    for _ in range(int(n_bootstrap)):
        idx = rng.choice(n, size=n, replace=True)
        c_b = causal[idx]
        t_b = trad[idx]
        if np.std(c_b) == 0 or np.std(t_b) == 0:
            continue
        r_boots.append(float(np.corrcoef(c_b, t_b)[0, 1]))
        rho_boots.append(float(_spearman_rho(c_b, t_b)))
    if len(r_boots) >= 10:
        result["pearson_r_ci"] = [
            round(float(np.percentile(r_boots, 2.5)), 4),
            round(float(np.percentile(r_boots, 97.5)), 4),
        ]
    if len(rho_boots) >= 10:
        result["spearman_rho_ci"] = [
            round(float(np.percentile(rho_boots, 2.5)), 4),
            round(float(np.percentile(rho_boots, 97.5)), 4),
        ]
    return result


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation via pandas (avoids a hard scipy dependency)."""
    sx = pd.Series(x).rank(method="average")
    sy = pd.Series(y).rank(method="average")
    if sx.std() == 0 or sy.std() == 0:
        return float("nan")
    return float(sx.corr(sy))


# ---------------------------------------------------------------------------
# Merge + biggest movers
# ---------------------------------------------------------------------------

def merge_with_traditional(
    causal_df: pd.DataFrame,
    trad_batters: pd.DataFrame,
    trad_pitchers: pd.DataFrame,
    *,
    pa_min: int = 100,
    ip_min: int = 20,
    pitcher_ids: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """Join the CausalWAR test-set effects with traditional WAR proxies.

    Position is decided by whether the player appears in the pitcher set;
    players in both tables (two-way players) are classified by whichever
    workload is larger (PA vs 2.5 * IP).
    """
    causal_df = causal_df.copy()
    trad_batters = trad_batters.copy()
    trad_pitchers = trad_pitchers.copy()

    pitcher_set = set() if pitcher_ids is None else set(int(p) for p in pitcher_ids)

    # Combine traditional frames; prefer pitcher row for known pitchers.
    # Normalise empty frames so the rename / column-select below never KeyErrors.
    if trad_batters.empty:
        trad_batters_slim = pd.DataFrame(
            columns=["player_id", "trad_war_batter", "pa_total", "war_source_bat"]
        )
    else:
        tb = trad_batters.rename(
            columns={"pa_total": "pa_total", "trad_war": "trad_war_batter"}
        )
        if "war_source" in tb.columns:
            tb = tb.rename(columns={"war_source": "war_source_bat"})
        else:
            tb["war_source_bat"] = None
        trad_batters_slim = tb[
            ["player_id", "trad_war_batter", "pa_total", "war_source_bat"]
        ]

    if trad_pitchers.empty:
        trad_pitchers_slim = pd.DataFrame(
            columns=["player_id", "trad_war_pitcher", "ip_total", "war_source_pit"]
        )
    else:
        tp = trad_pitchers.rename(
            columns={"ip_total": "ip_total", "trad_war": "trad_war_pitcher"}
        )
        if "war_source" in tp.columns:
            tp = tp.rename(columns={"war_source": "war_source_pit"})
        else:
            tp["war_source_pit"] = None
        trad_pitchers_slim = tp[
            ["player_id", "trad_war_pitcher", "ip_total", "war_source_pit"]
        ]

    merged = causal_df.merge(
        trad_batters_slim, on="player_id", how="left",
    ).merge(
        trad_pitchers_slim, on="player_id", how="left",
    )

    # Assign position: pitcher if in pitcher_set OR has meaningful IP but little PA.
    pa_total = merged["pa_total"].fillna(0).astype(float)
    ip_total = merged["ip_total"].fillna(0).astype(float)

    is_pitcher_ids = merged["player_id"].isin(pitcher_set)
    ip_dominant = ip_total * 2.5 > pa_total
    is_pitcher = is_pitcher_ids & ip_dominant & (ip_total >= ip_min)
    # Fall back to IP-dominance for anyone not in the pitcher set but
    # clearly pitching (covers DB gaps in pitcher-ID enumeration).
    is_pitcher = is_pitcher | ((ip_total >= ip_min) & (pa_total < pa_min))

    merged["position"] = np.where(is_pitcher, "pitcher", "batter")
    merged["trad_war"] = np.where(
        is_pitcher, merged["trad_war_pitcher"], merged["trad_war_batter"]
    )
    merged["war_source"] = np.where(
        is_pitcher,
        merged["war_source_pit"].fillna("missing"),
        merged["war_source_bat"].fillna("missing"),
    )

    # Qualification: batter needs pa>=pa_min, pitcher needs ip>=ip_min.
    qualifies = (
        ((merged["position"] == "batter") & (pa_total >= pa_min))
        | ((merged["position"] == "pitcher") & (ip_total >= ip_min))
    )
    merged = merged[qualifies].copy()

    # Drop rows where trad_war couldn't be computed.
    merged = merged.dropna(subset=["trad_war", "causal_war"]).copy()

    # Ranks
    merged["rank_causal"] = merged["causal_war"].rank(ascending=False, method="min").astype(int)
    merged["rank_trad"] = merged["trad_war"].rank(ascending=False, method="min").astype(int)
    merged["rank_diff"] = merged["rank_trad"] - merged["rank_causal"]

    return merged.reset_index(drop=True)


def biggest_movers(merged: pd.DataFrame, k: int = 15) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the top-k over-valued and under-valued players.

    Over-valued:  causal rank much better than traditional rank (large negative rank_diff).
    Under-valued: traditional rank much better than causal rank (large positive rank_diff).
    """
    if merged.empty:
        return merged.iloc[0:0].copy(), merged.iloc[0:0].copy()

    cols = [
        "player_id", "name", "position", "pa_total", "ip_total",
        "causal_war", "trad_war", "rank_causal", "rank_trad", "rank_diff",
    ]
    avail_cols = [c for c in cols if c in merged.columns]

    over = merged.nsmallest(k, "rank_diff")[avail_cols].copy()
    under = merged.nlargest(k, "rank_diff")[avail_cols].copy()
    return over, under


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def write_scatter_html(merged: pd.DataFrame, output_path: Path) -> bool:
    """Write a Plotly scatter of causal vs trad WAR; return True on success."""
    try:
        import plotly.express as px  # type: ignore
        import plotly.graph_objects as go  # type: ignore
    except Exception:
        logger.warning("plotly not available; skipping scatter HTML")
        return False

    if merged.empty:
        logger.warning("No rows available for scatter; skipping HTML")
        return False

    fig = px.scatter(
        merged,
        x="trad_war",
        y="causal_war",
        color="position",
        color_discrete_map={"batter": "red", "pitcher": "blue"},
        hover_data={
            "player_id": True,
            "name": True,
            "pa_total": True,
            "ip_total": True,
            "rank_causal": True,
            "rank_trad": True,
            "causal_war": ":.2f",
            "trad_war": ":.2f",
        },
        title="CausalWAR vs Traditional WAR (test window)",
        labels={"trad_war": "Traditional WAR (proxy)", "causal_war": "CausalWAR"},
    )

    # Diagonal reference y = x
    lo = float(min(merged["trad_war"].min(), merged["causal_war"].min()))
    hi = float(max(merged["trad_war"].max(), merged["causal_war"].max()))
    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi],
            mode="lines",
            name="y = x",
            line=dict(color="grey", dash="dash"),
            showlegend=True,
        )
    )
    fig.update_layout(template="plotly_white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs="cdn")
    return True


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def _attach_player_names(
    conn: duckdb.DuckDBPyConnection,
    player_ids: Iterable[int],
) -> pd.DataFrame:
    ids = [int(p) for p in set(player_ids)]
    if not ids:
        return pd.DataFrame(columns=["player_id", "name"])
    placeholders = ", ".join(str(i) for i in ids)
    try:
        df = conn.execute(
            f"SELECT player_id, full_name AS name FROM players WHERE player_id IN ({placeholders})"
        ).fetchdf()
    except Exception:
        df = pd.DataFrame(columns=["player_id", "name"])
    return df


def _known_pitcher_ids(
    conn: duckdb.DuckDBPyConnection,
    test_start: int,
    test_end: int,
) -> set[int]:
    try:
        rows = conn.execute(
            """
            SELECT DISTINCT player_id FROM season_pitching_stats
            WHERE season BETWEEN $1 AND $2 AND ip >= 5
            """,
            [int(test_start), int(test_end)],
        ).fetchdf()
        return set(int(p) for p in rows["player_id"].tolist())
    except Exception:
        return set()


def run_comparison(cfg: BaselineComparisonConfig) -> dict[str, Any]:
    """Execute the full baseline comparison and return the metrics dict."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read-only keeps the DB concurrent-access friendly; the CausalWAR model
    # only reads from it and writes the artifact to the filesystem.
    conn = get_connection(read_only=True)

    logger.info(
        "Starting CausalWAR baseline comparison: train %d-%d, test %d-%d",
        cfg.train_start, cfg.train_end, cfg.test_start, cfg.test_end,
    )

    # ---- 1. Train CausalWAR with the reduced-bootstrap config --------------
    causal_cfg = CausalWARConfig(
        n_bootstrap=cfg.n_bootstrap_train,
        train_start_year=cfg.train_start,
        train_end_year=cfg.train_end,
        test_start_year=cfg.test_start,
        test_end_year=cfg.test_end,
        random_state=cfg.random_state,
    )
    model = CausalWARModel(config=causal_cfg)
    split = model.train_test_split(
        conn,
        train_split=(cfg.train_start, cfg.train_end),
        test_split=(cfg.test_start, cfg.test_end),
    )
    test_effects: pd.DataFrame = split["test_player_effects"]
    train_metrics = split["train_metrics"]
    test_metrics = split["test_metrics"]
    logger.info(
        "Train/test split done: train players=%d, test players=%d",
        int(train_metrics.get("n_players_estimated", 0)),
        int(test_metrics.get("n_test_players", 0)),
    )

    # ---- 2. Traditional WAR proxies ---------------------------------------
    trad_batters = _fetch_batter_traditional_war(conn, cfg.test_start, cfg.test_end)
    trad_pitchers = _fetch_pitcher_traditional_war(conn, cfg.test_start, cfg.test_end)
    pitcher_ids = _known_pitcher_ids(conn, cfg.test_start, cfg.test_end)

    logger.info(
        "Traditional WAR rows: batters=%d, pitchers=%d, pitcher_id_set=%d",
        len(trad_batters), len(trad_pitchers), len(pitcher_ids),
    )

    # ---- 3. Merge ---------------------------------------------------------
    merged = merge_with_traditional(
        test_effects,
        trad_batters,
        trad_pitchers,
        pa_min=cfg.pa_min,
        ip_min=cfg.ip_min,
        pitcher_ids=pitcher_ids,
    )

    # Attach names
    names = _attach_player_names(conn, merged["player_id"].tolist())
    merged = merged.merge(names, on="player_id", how="left")
    if "name" not in merged.columns:
        merged["name"] = None

    # ---- 4. Metrics -------------------------------------------------------
    def _metrics_for(subset: pd.DataFrame, label: str) -> dict[str, Any]:
        m = compute_metrics(
            subset["causal_war"].to_numpy(),
            subset["trad_war"].to_numpy(),
            n_bootstrap=cfg.n_bootstrap_ci,
            random_state=cfg.random_state,
        )
        m["label"] = label
        return m

    batters_subset = merged[merged["position"] == "batter"]
    pitchers_subset = merged[merged["position"] == "pitcher"]

    metrics_batters = _metrics_for(batters_subset, "batters")
    metrics_pitchers = _metrics_for(pitchers_subset, "pitchers")
    metrics_combined = _metrics_for(merged, "combined")

    # Rank-combined: compute rho on combined ranks explicitly.
    logger.info(
        "Metrics batters: r=%s rho=%s n=%s",
        metrics_batters["pearson_r"], metrics_batters["spearman_rho"], metrics_batters["n"],
    )
    logger.info(
        "Metrics pitchers: r=%s rho=%s n=%s",
        metrics_pitchers["pearson_r"], metrics_pitchers["spearman_rho"], metrics_pitchers["n"],
    )
    logger.info(
        "Metrics combined: r=%s rho=%s n=%s",
        metrics_combined["pearson_r"], metrics_combined["spearman_rho"], metrics_combined["n"],
    )

    # ---- 5. Biggest movers ------------------------------------------------
    over, under = biggest_movers(merged, k=15)

    # ---- 6. Write artifacts -----------------------------------------------
    csv_path = output_dir / (
        f"causal_war_baseline_comparison_{cfg.test_start}_{cfg.test_end}.csv"
    )
    metrics_path = output_dir / (
        f"causal_war_baseline_comparison_{cfg.test_start}_{cfg.test_end}_metrics.json"
    )
    scatter_path = output_dir / "causal_war_baseline_scatter.html"

    out_cols = [
        "player_id", "name", "position",
        "causal_war", "trad_war",
        "rank_causal", "rank_trad", "rank_diff",
        "pa", "pa_total", "ip_total",
        "ci_low", "ci_high",
        "war_source",
        "sparse",
    ]
    out_cols = [c for c in out_cols if c in merged.columns]
    merged[out_cols].to_csv(csv_path, index=False)
    logger.info("Wrote %s (%d rows)", csv_path, len(merged))

    plot_ok = write_scatter_html(merged, scatter_path)

    # Spec threshold check
    r_combined = metrics_combined["pearson_r"] or 0.0
    rho_combined = metrics_combined["spearman_rho"] or 0.0
    pass_threshold = (r_combined >= 0.5) and (rho_combined >= 0.6)

    metrics_payload: dict[str, Any] = {
        "config": {
            "train_start": cfg.train_start,
            "train_end": cfg.train_end,
            "test_start": cfg.test_start,
            "test_end": cfg.test_end,
            "pa_min": cfg.pa_min,
            "ip_min": cfg.ip_min,
            "n_bootstrap_train": cfg.n_bootstrap_train,
            "n_bootstrap_ci": cfg.n_bootstrap_ci,
            "random_state": cfg.random_state,
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "metrics_batters": metrics_batters,
        "metrics_pitchers": metrics_pitchers,
        "metrics_combined": metrics_combined,
        "spec_threshold": {
            "pearson_r_required": 0.5,
            "spearman_rho_required": 0.6,
            "pass": bool(pass_threshold),
        },
        "biggest_movers_over_valued": over.to_dict(orient="records"),
        "biggest_movers_under_valued": under.to_dict(orient="records"),
        "artifacts": {
            "csv": str(csv_path),
            "metrics_json": str(metrics_path),
            "scatter_html": str(scatter_path) if plot_ok else None,
        },
        "data_quality": {
            "batter_war_source": (
                trad_batters["war_source"].iloc[0] if len(trad_batters) else None
            ),
            "pitcher_war_source": (
                trad_pitchers["war_source"].iloc[0] if len(trad_pitchers) else None
            ),
            "n_test_effects": int(len(test_effects)),
            "n_merged_rows": int(len(merged)),
        },
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, default=_json_safe)
    logger.info("Wrote %s", metrics_path)

    conn.close()

    _print_summary(metrics_payload, over, under)

    return metrics_payload


def _json_safe(v: Any) -> Any:
    """Coerce numpy / pandas scalars to JSON-serialisable types."""
    if isinstance(v, (np.floating, np.integer)):
        if isinstance(v, np.floating) and (math.isnan(v) or math.isinf(v)):
            return None
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return str(v)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_summary(
    metrics: dict[str, Any], over: pd.DataFrame, under: pd.DataFrame
) -> None:
    mb = metrics["metrics_batters"]
    mp = metrics["metrics_pitchers"]
    mc = metrics["metrics_combined"]
    spec = metrics["spec_threshold"]
    print("\n" + "=" * 72)
    print("CausalWAR Baseline Comparison -- Summary")
    print("=" * 72)
    print(
        f"Batters:  n={mb['n']:<4d}  r={mb['pearson_r']}  rho={mb['spearman_rho']}  "
        f"RMSE={mb['rmse']}  MAE={mb['mae']}"
    )
    print(
        f"Pitchers: n={mp['n']:<4d}  r={mp['pearson_r']}  rho={mp['spearman_rho']}  "
        f"RMSE={mp['rmse']}  MAE={mp['mae']}"
    )
    print(
        f"Combined: n={mc['n']:<4d}  r={mc['pearson_r']}  rho={mc['spearman_rho']}  "
        f"RMSE={mc['rmse']}  MAE={mc['mae']}"
    )
    print("-" * 72)
    verdict = "PASS" if spec["pass"] else "FAIL"
    print(
        f"Spec threshold (r>=0.5 AND rho>=0.6): {verdict}  "
        f"(combined r={mc['pearson_r']}, rho={mc['spearman_rho']})"
    )
    print("-" * 72)
    print("Top 5 over-valued (causal rank much better than traditional):")
    for _, row in over.head(5).iterrows():
        print(
            f"  {row.get('name')!s:30s}  pos={row.get('position')}  "
            f"causal={row.get('causal_war')}  trad={row.get('trad_war')}  "
            f"rank_causal={row.get('rank_causal')}  rank_trad={row.get('rank_trad')}"
        )
    print("Top 5 under-valued (traditional rank much better than causal):")
    for _, row in under.head(5).iterrows():
        print(
            f"  {row.get('name')!s:30s}  pos={row.get('position')}  "
            f"causal={row.get('causal_war')}  trad={row.get('trad_war')}  "
            f"rank_causal={row.get('rank_causal')}  rank_trad={row.get('rank_trad')}"
        )
    print("=" * 72 + "\n")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__ or "", formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-start", type=int, default=2015)
    p.add_argument("--train-end", type=int, default=2022)
    p.add_argument("--test-start", type=int, default=2023)
    p.add_argument("--test-end", type=int, default=2024)
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument(
        "--n-bootstrap-train", type=int, default=50,
        help="Bootstrap iterations inside CausalWAR training (kept small for runtime).",
    )
    p.add_argument(
        "--n-bootstrap-ci", type=int, default=1000,
        help="Bootstrap iterations for the cheap correlation CI on final pairs.",
    )
    p.add_argument("--pa-min", type=int, default=100)
    p.add_argument("--ip-min", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = BaselineComparisonConfig(
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        output_dir=args.output_dir,
        n_bootstrap_train=args.n_bootstrap_train,
        n_bootstrap_ci=args.n_bootstrap_ci,
        pa_min=args.pa_min,
        ip_min=args.ip_min,
        random_state=args.random_state,
    )
    try:
        run_comparison(cfg)
        return 0
    except Exception as exc:
        logger.exception("baseline_comparison failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
