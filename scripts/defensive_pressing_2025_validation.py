#!/usr/bin/env python
"""
Defensive Pressing (DPI) — 2025 external-validation pass.

Answers one question: does DPI's correlation with Statcast OAA survive on
the 2025 season, which DPI's training cohort never saw?

Pipeline
--------
1. Load the frozen xOut checkpoint (train window 2015-2022). Does NOT
   retrain.
2. Score team-season DPI for 2023, 2024, 2025 (30 teams x 3 seasons).
3. Load the team-season Statcast OAA baseline
   (``data/baselines/team_defense_2023_2025.parquet``), ingested
   separately via ``scripts/ingest_team_oaa.py``.
4. Compute Pearson + Spearman correlations per year, plus 95% bootstrap
   CIs.
5. Top-5 DPI vs top-5 OAA agreement count per year.
6. 2025 disagreement analysis: which teams deviate most, and what do
   their BIP-level stats say.
7. Write artifacts to ``results/defensive_pressing/2025_validation/``.

Usage
-----
    python scripts/defensive_pressing_2025_validation.py
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics import defensive_pressing as dp  # noqa: E402
from src.analytics.defensive_pressing import (  # noqa: E402
    DPIConfig,
    calculate_team_dpi,
    load_xout_checkpoint,
    DEFAULT_XOUT_CHECKPOINT,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("dpi_2025_validation")


@dataclass
class Config:
    seasons: tuple[int, ...] = (2023, 2024, 2025)
    random_state: int = 42
    n_bootstrap_ci: int = 1000
    oaa_parquet: Path = ROOT / "data" / "baselines" / "team_defense_2023_2025.parquet"
    output_dir: Path = ROOT / "results" / "defensive_pressing" / "2025_validation"


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    sx = pd.Series(x).rank(method="average")
    sy = pd.Series(y).rank(method="average")
    if sx.std() == 0 or sy.std() == 0:
        return float("nan")
    return float(sx.corr(sy))


def _bootstrap_ci(
    x: np.ndarray, y: np.ndarray, *,
    n_boot: int, random_state: int, fn=None,
) -> tuple[float, float] | None:
    if fn is None:
        fn = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    n = int(len(x))
    if n < 3 or n_boot < 10:
        return None
    rng = np.random.RandomState(random_state)
    boots: list[float] = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        a = x[idx]
        b = y[idx]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        try:
            boots.append(float(fn(a, b)))
        except Exception:
            continue
    if len(boots) < max(10, n_boot // 4):
        return None
    return (
        round(float(np.percentile(boots, 2.5)), 4),
        round(float(np.percentile(boots, 97.5)), 4),
    )


def _pair_metrics(
    a: np.ndarray, b: np.ndarray, *,
    n_boot: int, random_state: int,
) -> dict[str, Any]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a = a[mask]
    b = b[mask]
    n = int(len(a))
    out: dict[str, Any] = {
        "n": n,
        "pearson_r": None,
        "pearson_r_ci": None,
        "spearman_rho": None,
        "spearman_rho_ci": None,
    }
    if n < 3 or np.std(a) == 0 or np.std(b) == 0:
        return out
    r = float(np.corrcoef(a, b)[0, 1])
    rho = float(_spearman(a, b))
    r_ci = _bootstrap_ci(a, b, n_boot=n_boot, random_state=random_state)
    rho_ci = _bootstrap_ci(
        a, b, n_boot=n_boot, random_state=random_state + 1, fn=_spearman,
    )
    out.update({
        "pearson_r": round(r, 4),
        "pearson_r_ci": list(r_ci) if r_ci else None,
        "spearman_rho": round(rho, 4),
        "spearman_rho_ci": list(rho_ci) if rho_ci else None,
    })
    return out


# ---------------------------------------------------------------------------
# DPI scoring
# ---------------------------------------------------------------------------

def _list_teams(conn: duckdb.DuckDBPyConnection, season: int) -> list[str]:
    df = conn.execute(
        """
        SELECT DISTINCT t FROM (
            SELECT home_team AS t FROM pitches
            WHERE EXTRACT(YEAR FROM game_date) = $1
            UNION
            SELECT away_team AS t FROM pitches
            WHERE EXTRACT(YEAR FROM game_date) = $1
        ) ORDER BY t
        """,
        [int(season)],
    ).fetchdf()
    return [str(t) for t in df["t"].tolist()]


def compute_team_dpi(
    conn: duckdb.DuckDBPyConnection, seasons: list[int],
) -> pd.DataFrame:
    config = DPIConfig()
    rows: list[dict[str, Any]] = []
    for season in seasons:
        teams = _list_teams(conn, season)
        logger.info("Scoring DPI for %d teams in %d", len(teams), season)
        t0 = time.monotonic()
        for i, team in enumerate(teams):
            profile = calculate_team_dpi(conn, team, season, config)
            if profile.get("dpi_mean") is None:
                continue
            rows.append({
                "team_id": str(team),
                "season": int(season),
                "dpi_mean": float(profile["dpi_mean"]),
                "dpi_total": float(profile["dpi_total"]),
                "dpi_std": float(profile.get("dpi_std", 0.0)),
                "consistency": float(profile["consistency"]),
                "extra_base_prevention": (
                    float(profile["extra_base_prevention"])
                    if profile["extra_base_prevention"] is not None else None
                ),
                "n_games": int(profile["n_games"]),
            })
            if (i + 1) % 10 == 0:
                rate = (i + 1) / max(time.monotonic() - t0, 1e-6)
                logger.info(
                    "  %d/%d teams (%.2f teams/s)", i + 1, len(teams), rate,
                )
        logger.info(
            "Season %d DPI complete in %.1fs", season, time.monotonic() - t0,
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Supporting proxy metrics for the disagreement section
# ---------------------------------------------------------------------------

def _team_babip_against(
    conn: duckdb.DuckDBPyConnection, seasons: list[int],
) -> pd.DataFrame:
    s_in = ", ".join(str(s) for s in seasons)
    df = conn.execute(f"""
        SELECT
            CASE WHEN inning_topbot = 'Top' THEN home_team ELSE away_team END AS team_id,
            EXTRACT(YEAR FROM game_date)::INT AS season,
            COUNT(*) AS n_bip,
            SUM(CASE WHEN events IN ('single','double','triple') THEN 1 ELSE 0 END) AS n_non_hr_hits,
            SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS n_hr
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) IN ({s_in})
          AND type = 'X'
          AND events IS NOT NULL
        GROUP BY team_id, season
    """).fetchdf()
    in_play = (df["n_bip"] - df["n_hr"]).astype(float)
    df["babip_against"] = np.where(
        in_play > 0,
        df["n_non_hr_hits"].astype(float) / in_play,
        np.nan,
    )
    return df[["team_id", "season", "babip_against", "n_bip"]]


def _team_rp_proxy(
    conn: duckdb.DuckDBPyConnection, seasons: list[int],
) -> pd.DataFrame:
    s_in = ", ".join(str(s) for s in seasons)
    df = conn.execute(f"""
        SELECT
            CASE WHEN inning_topbot = 'Top' THEN home_team ELSE away_team END AS team_id,
            EXTRACT(YEAR FROM game_date)::INT AS season,
            SUM(delta_run_exp) AS sum_delta_run_exp_on_def
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) IN ({s_in})
          AND delta_run_exp IS NOT NULL
        GROUP BY team_id, season
    """).fetchdf()
    df["rp_proxy"] = -df["sum_delta_run_exp_on_def"].astype(float)
    return df[["team_id", "season", "rp_proxy"]]


# ---------------------------------------------------------------------------
# Top-5 agreement and disagreement analysis
# ---------------------------------------------------------------------------

def _top_k_agreement(merged: pd.DataFrame, season: int, k: int = 5) -> dict:
    sub = merged[merged["season"] == season].dropna(subset=["dpi_mean", "team_oaa"])
    dpi_top = set(sub.nlargest(k, "dpi_mean")["team_id"].tolist())
    oaa_top = set(sub.nlargest(k, "team_oaa")["team_id"].tolist())
    overlap = dpi_top & oaa_top
    return {
        "season": int(season),
        "k": k,
        "dpi_top_k": sorted(dpi_top),
        "oaa_top_k": sorted(oaa_top),
        "overlap_count": int(len(overlap)),
        "overlap_teams": sorted(overlap),
    }


def _disagreement_analysis(
    merged: pd.DataFrame, season: int, top_n: int = 8,
) -> pd.DataFrame:
    sub = merged[merged["season"] == season].dropna(
        subset=["dpi_mean", "team_oaa"]
    ).copy()
    # Rank by both metrics (higher = better defense in both)
    sub["dpi_rank"] = sub["dpi_mean"].rank(ascending=False, method="min").astype(int)
    sub["oaa_rank"] = sub["team_oaa"].rank(ascending=False, method="min").astype(int)
    sub["rank_diff"] = sub["dpi_rank"] - sub["oaa_rank"]
    # Z-scores to make divergence comparable
    for c in ["dpi_mean", "team_oaa"]:
        sub[f"{c}_z"] = (sub[c] - sub[c].mean()) / sub[c].std()
    sub["z_diff"] = sub["dpi_mean_z"] - sub["team_oaa_z"]
    sub["abs_rank_diff"] = sub["rank_diff"].abs()
    sub = sub.sort_values("abs_rank_diff", ascending=False).head(top_n)
    cols = [
        "team_id", "season", "dpi_mean", "team_oaa", "team_frp", "babip_against",
        "rp_proxy", "dpi_rank", "oaa_rank", "rank_diff", "z_diff",
    ]
    cols = [c for c in cols if c in sub.columns]
    return sub[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(cfg: Config) -> dict[str, Any]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # 1) xOut checkpoint
    model, meta = load_xout_checkpoint(DEFAULT_XOUT_CHECKPOINT)
    if model is None:
        raise RuntimeError(
            f"xOut checkpoint not found at {DEFAULT_XOUT_CHECKPOINT}"
        )
    train_seasons = (meta or {}).get("train_seasons") or []
    logger.info(
        "Loaded xOut checkpoint train_seasons=%s auc=%s",
        train_seasons, (meta or {}).get("auc"),
    )
    # Sanity: train window must not include any test season
    test_seasons = list(cfg.seasons)
    overlap_train_test = sorted(set(train_seasons) & set(test_seasons))
    if overlap_train_test:
        raise RuntimeError(
            f"Leakage: xOut train_seasons overlap with test seasons: {overlap_train_test}"
        )

    # 2) Score DPI for 2023, 2024, 2025 (cache DPI CSV between runs)
    conn = get_connection(read_only=True)
    dpi_cache = cfg.output_dir / "_dpi_scores_cache.csv"
    if dpi_cache.exists():
        dpi_df = pd.read_csv(dpi_cache)
        have = set(int(s) for s in dpi_df["season"].unique().tolist())
        needed = set(test_seasons) - have
        if needed:
            logger.info("Cache miss for seasons %s; scoring", sorted(needed))
            extra = compute_team_dpi(conn, sorted(needed))
            dpi_df = pd.concat([dpi_df, extra], ignore_index=True)
            dpi_df.to_csv(dpi_cache, index=False)
        else:
            logger.info("Loaded cached DPI scores from %s", dpi_cache)
    else:
        dpi_df = compute_team_dpi(conn, test_seasons)
        dpi_df.to_csv(dpi_cache, index=False)

    # 3) Load OAA baseline
    oaa = pd.read_parquet(cfg.oaa_parquet)
    oaa = oaa.rename(columns={"team_abbr": "team_id"})

    # 4) Supporting proxies
    babip_df = _team_babip_against(conn, test_seasons)
    rp_df = _team_rp_proxy(conn, test_seasons)

    conn.close()

    merged = (
        dpi_df
        .merge(oaa[["team_id", "season", "team_oaa", "team_frp", "n_players"]],
               on=["team_id", "season"], how="left")
        .merge(babip_df, on=["team_id", "season"], how="left")
        .merge(rp_df, on=["team_id", "season"], how="left")
    )
    merged = merged.sort_values(["season", "team_id"]).reset_index(drop=True)

    # 5) Year-by-year correlations
    yearly: dict[str, Any] = {}
    for season in test_seasons:
        sub = merged[merged["season"] == season]
        mpear = _pair_metrics(
            sub["dpi_mean"].to_numpy(),
            sub["team_oaa"].to_numpy(),
            n_boot=cfg.n_bootstrap_ci, random_state=cfg.random_state + season,
        )
        mfrp = _pair_metrics(
            sub["dpi_mean"].to_numpy(),
            sub["team_frp"].to_numpy(),
            n_boot=cfg.n_bootstrap_ci, random_state=cfg.random_state + season + 100,
        )
        top5 = _top_k_agreement(merged, season, k=5)
        yearly[str(season)] = {
            "dpi_vs_oaa": mpear,
            "dpi_vs_frp": mfrp,
            "top5_agreement": top5,
        }

    # Pooled (all 3 years)
    pooled = _pair_metrics(
        merged["dpi_mean"].to_numpy(),
        merged["team_oaa"].to_numpy(),
        n_boot=cfg.n_bootstrap_ci, random_state=cfg.random_state + 999,
    )

    # 6) Disagreement analysis for 2025
    dis_2025 = _disagreement_analysis(merged, 2025, top_n=10)

    # 7) Artifacts
    rankings_path = cfg.output_dir / "team_rankings_2025.csv"
    m2025 = merged[merged["season"] == 2025].copy()
    m2025["dpi_rank"] = m2025["dpi_mean"].rank(ascending=False, method="min").astype(int)
    m2025["oaa_rank"] = m2025["team_oaa"].rank(ascending=False, method="min").astype(int)
    m2025["rank_diff"] = m2025["dpi_rank"] - m2025["oaa_rank"]
    m2025 = m2025.sort_values("dpi_rank")
    m2025.to_csv(rankings_path, index=False)

    all_rankings_path = cfg.output_dir / "team_rankings_all_years.csv"
    merged.to_csv(all_rankings_path, index=False)

    # Gate 6 replay: does 2025 pass >= 0.45?
    r_2025 = yearly["2025"]["dpi_vs_oaa"].get("pearson_r")
    ci_2025 = yearly["2025"]["dpi_vs_oaa"].get("pearson_r_ci")
    passes_2025_gate6 = bool(r_2025 is not None and r_2025 >= 0.45)

    payload = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "xout_checkpoint": {
            "path": str(DEFAULT_XOUT_CHECKPOINT),
            "train_seasons": list(train_seasons),
            "auc_on_train_holdout": (meta or {}).get("auc"),
        },
        "test_seasons": test_seasons,
        "leakage_check": {
            "train_test_overlap": overlap_train_test,
            "pass": len(overlap_train_test) == 0,
        },
        "yearly": yearly,
        "pooled_2023_2025": pooled,
        "gates": {
            "dpi_vs_oaa_2025_pearson_r_ge_0.45": {
                "threshold": 0.45,
                "measured": r_2025,
                "ci95": ci_2025,
                "pass": passes_2025_gate6,
            },
            "dpi_vs_oaa_2025_pearson_r_ge_0.40": {
                "threshold": 0.40,
                "measured": r_2025,
                "ci95": ci_2025,
                "pass": bool(r_2025 is not None and r_2025 >= 0.40),
            },
        },
        "disagreement_2025_top10": dis_2025.to_dict(orient="records"),
        "artifacts": {
            "team_rankings_2025_csv": str(rankings_path),
            "team_rankings_all_years_csv": str(all_rankings_path),
        },
    }

    json_path = cfg.output_dir / "dpi_vs_oaa_yearly.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_safe)
    logger.info("Wrote %s", json_path)

    _print_summary(payload)
    return payload


def _print_summary(payload: dict) -> None:
    print()
    print("=" * 72)
    print("DPI 2025 external-validation summary")
    print("=" * 72)
    for season, year_block in payload["yearly"].items():
        oaa = year_block["dpi_vs_oaa"]
        top5 = year_block["top5_agreement"]
        print(
            f"{season}: n={oaa['n']}  r(DPI,OAA)={oaa['pearson_r']}  "
            f"CI={oaa['pearson_r_ci']}  rho={oaa['spearman_rho']}  "
            f"top5_overlap={top5['overlap_count']}"
        )
    p = payload["pooled_2023_2025"]
    print(
        f"POOLED 2023-2025: n={p['n']}  r={p['pearson_r']}  CI={p['pearson_r_ci']}  "
        f"rho={p['spearman_rho']}"
    )
    g = payload["gates"]["dpi_vs_oaa_2025_pearson_r_ge_0.45"]
    print(
        f"Gate replay (r >= 0.45 on 2025 alone): "
        f"{'PASS' if g['pass'] else 'FAIL'}  measured={g['measured']}"
    )
    print("=" * 72)


def _json_safe(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, Path):
        return str(v)
    try:
        json.dumps(v)
        return v
    except Exception:
        return str(v)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasons", nargs="+", type=int, default=[2023, 2024, 2025])
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-bootstrap-ci", type=int, default=1000)
    p.add_argument(
        "--oaa-parquet", type=Path,
        default=ROOT / "data" / "baselines" / "team_defense_2023_2025.parquet",
    )
    p.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "results" / "defensive_pressing" / "2025_validation",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = Config(
        seasons=tuple(args.seasons),
        random_state=args.random_state,
        n_bootstrap_ci=args.n_bootstrap_ci,
        oaa_parquet=args.oaa_parquet,
        output_dir=args.output_dir,
    )
    try:
        run(cfg)
        return 0
    except Exception as exc:
        logger.exception("dpi_2025_validation failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
