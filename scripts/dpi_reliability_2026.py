"""DPI reliability quick wins -- WS3.1 of the 2026-08-10 platform improvement plan.

Remediates audit findings 2 (mislabeled stability, YoY n=1) and 9 (pooled-BIP
bootstrap CIs ignore team clustering) from docs/audits/FLAGSHIP_AUDIT_2026-08-10.md.

Three deliverables:
  1. Year-over-year stability 2023->2024 (confirm recorded r~0.59) and
     2024->2025 (never previously computed), from the recorded
     team_rankings_all_years.csv, plus a DB-recomputed all-pairs table.
  2. Split-half reliability per season 2015-2025: games split by
     game_pk % 2, per-half season DPI computed exactly as production
     (same frozen xOut model, same per-game aggregation), correlated
     across teams within season, Spearman-Brown corrected. Implied
     shrinkage coefficient + regressed-DPI column.
  3. Cluster-aware CI re-issue for Gates 2/3/6:
     (a) per-team-season CIs by resampling GAMES within team-season;
     (b) pooled/cross-team CIs by pairs cluster bootstrap (resample the
         30 team franchises) and wild cluster bootstrap-t (Rademacher
         weights on team clusters, Cameron-Gelbach-Miller 2008 /
         Cameron & Miller JHR 2015). Optional cross-check p-values via
         the `wildboottest` package when installed.

MODEL PROVENANCE (critical): the working-tree
models/defensive_pressing/xout_v1.pkl is CONTAMINATED (nightly retrain on
2015-2026, audit DPI finding 3). This script NEVER reads it. The validated
blob is extracted from git (default rev 32c7142, the commit that persisted
the validation-era checkpoint; metadata train_seasons 2015-2022, fitted
2026-04-18T22:17Z) into a temp path, its metadata is verified
(max(train_seasons) <= 2024), and its sha256 recorded in every artifact.

Read-only vs the DB (get_connection(read_only=True)); writes only to the
--out-dir results directory.

Usage:
    python scripts/dpi_reliability_2026.py                # full run
    python scripts/dpi_reliability_2026.py --reuse-scores # skip DB scoring,
        reuse <out-dir>/game_dpi_2015_2025.csv from a previous run
    python scripts/dpi_reliability_2026.py --model-path C:/path/to/clean.pkl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.db.schema import get_connection  # noqa: E402
from src.analytics import defensive_pressing as dp  # noqa: E402

DEFAULT_GIT_REV = "32c7142"
BLOB_REPO_PATH = "models/defensive_pressing/xout_v1.pkl"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "defensive_pressing" / "reliability_2026-08-10"
RANKINGS_CSV = (
    REPO_ROOT / "results" / "defensive_pressing" / "2025_validation"
    / "team_rankings_all_years.csv"
)
SEASONS = list(range(2015, 2026))  # 2015-2025 inclusive; 2026 in-season excluded
MIN_BIP_PER_GAME = dp.MIN_BIP_PER_GAME  # 5, production constant

# Recorded CIs being superseded (source strings kept verbatim for the doc).
RECORDED_CIS = {
    "gate2_rp_2023_2024": {
        "recorded_r": 0.6197, "recorded_ci": [0.4228, 0.7710],
        "source": "v1/v2 run, defensive_pressing_results.md (v1 CI shown; v2 point 0.6197)",
    },
    "gate3_babip_2023_2024": {
        "recorded_r": -0.7334, "recorded_ci": [-0.8306, -0.6230],
        "source": "v1/v2 run, defensive_pressing_results.md (v1 CI shown; v2 point -0.7334)",
    },
    "gate6_oaa_2023_2024": {
        "recorded_r": 0.5492, "recorded_ci": [0.307, 0.721],
        "source": "v2 run, defensive_pressing_results.md",
    },
    "gate6_oaa_2025": {
        "recorded_r": 0.6406, "recorded_ci": [0.4209, 0.7922],
        "source": "2025_validation/dpi_vs_oaa_yearly.json",
    },
    "gate6_oaa_pooled_2023_2025": {
        "recorded_r": 0.4869, "recorded_ci": [0.3174, 0.635],
        "source": "2025_validation/dpi_vs_oaa_yearly.json pooled_2023_2025",
    },
}


# ---------------------------------------------------------------------------
# Model provenance
# ---------------------------------------------------------------------------

def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_model(model_path: str | None, git_rev: str) -> tuple[Path, dict]:
    """Return (path, provenance) for a verified clean xOut checkpoint.

    Refuses any path inside <repo>/models (working tree is contaminated).
    """
    if model_path:
        path = Path(model_path).resolve()
        prov = {"origin": f"--model-path {model_path}"}
    else:
        scratch = Path(tempfile.gettempdir()) / "dpi_reliability_2026"
        scratch.mkdir(parents=True, exist_ok=True)
        path = scratch / f"xout_frozen_{git_rev}.pkl"
        with open(path, "wb") as fh:
            subprocess.run(
                ["git", "show", f"{git_rev}:{BLOB_REPO_PATH}"],
                cwd=str(REPO_ROOT), stdout=fh, check=True,
            )
        prov = {"origin": f"git show {git_rev}:{BLOB_REPO_PATH}"}

    models_dir = (REPO_ROOT / "models").resolve()
    if models_dir in path.parents:
        raise RuntimeError(
            f"Refusing to load {path}: working-tree models/ is contaminated "
            "(audit DPI finding 3). Pass a scratch copy or let the script "
            "extract the git blob."
        )

    import joblib
    bundle = joblib.load(path)
    meta = bundle.get("metadata", {}) if isinstance(bundle, dict) else {}
    seasons = meta.get("train_seasons") or []
    if not seasons or max(seasons) > 2024 or min(seasons) < 2015:
        raise RuntimeError(
            f"Checkpoint at {path} has train_seasons={seasons}; refusing "
            "(validated recipe must not extend past 2024)."
        )
    prov.update({
        "path": str(path),
        "sha256": sha256_of(path),
        "train_seasons": seasons,
        "fitted_at": meta.get("fitted_at"),
        "auc": meta.get("auc"),
        "feature_columns": meta.get("feature_columns"),
        "use_park": meta.get("use_park"),
        "use_weather": meta.get("use_weather", False),
    })
    return path, prov


# ---------------------------------------------------------------------------
# Production-parity game DPI scoring (vectorized)
# ---------------------------------------------------------------------------

def score_game_dpi(conn, seasons: list[int]) -> pd.DataFrame:
    """Per-game per-defending-team DPI, replicating calculate_game_dpi.

    Parity notes (vs src/analytics/defensive_pressing.py::calculate_game_dpi):
      - BIP cohort: type='X', events IS NOT NULL, defending team = home_team
        on 'Top' half rows else away_team ('Bot').
      - n_bip counts ALL such rows; games with n_bip < 5 are excluded.
      - actual_outs over all rows; expected_outs summed only over rows with
        complete (launch_speed, launch_angle, hc_x, hc_y, bb_type).
      - games with n_bip >= 5 but ZERO feature-complete rows get dpi = 0.0
        (production early-return).
      - game dpi rounded to 3 decimals (production round(dpi, 3)).
    Scoring uses dp.compute_expected_outs, i.e. the exact production
    prediction path against the loaded frozen model.
    """
    frames = []
    for season in seasons:
        df = conn.execute(
            """
            SELECT game_pk, home_team, away_team, inning_topbot,
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
            xout.loc[has_data] = dp.compute_expected_outs(df.loc[has_data])
        df["xout"] = xout

        grp = df.groupby(["game_pk", "team_id"], sort=False).agg(
            n_bip=("events", "size"),
            actual_outs=("is_out", "sum"),
            expected_outs=("xout", "sum"),
            n_feature_rows=("xout", "count"),
        ).reset_index()
        grp = grp[grp["n_bip"] >= MIN_BIP_PER_GAME].copy()
        grp["dpi"] = np.round(grp["actual_outs"] - grp["expected_outs"], 3)
        grp.loc[grp["n_feature_rows"] == 0, "dpi"] = 0.0
        grp["season"] = season
        frames.append(grp)
        print(f"  scored season {season}: {len(grp)} team-games")
    out = pd.concat(frames, ignore_index=True)
    return out[["season", "team_id", "game_pk", "n_bip", "actual_outs",
                "expected_outs", "n_feature_rows", "dpi"]]


# ---------------------------------------------------------------------------
# Bootstrap machinery
# ---------------------------------------------------------------------------

def percentile_ci(vals: np.ndarray, alpha: float = 0.05) -> list[float]:
    vals = vals[~np.isnan(vals)]
    return [float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2)))]


def pairs_cluster_bootstrap_r(
    x: np.ndarray, y: np.ndarray, clusters: np.ndarray,
    n_boot: int, rng: np.random.Generator,
) -> dict:
    """Resample whole clusters (teams) with replacement; percentile CIs for
    Pearson r and Spearman rho. With one row per cluster this reduces to the
    ordinary pairs bootstrap."""
    uniq = np.unique(clusters)
    idx_by_cluster = {c: np.flatnonzero(clusters == c) for c in uniq}
    r_vals = np.full(n_boot, np.nan)
    rho_vals = np.full(n_boot, np.nan)
    n_degenerate = 0
    for b in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in chosen])
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            n_degenerate += 1
            continue
        r_vals[b] = stats.pearsonr(xb, yb)[0]
        rho_vals[b] = stats.spearmanr(xb, yb)[0]
    return {
        "pearson_ci": percentile_ci(r_vals),
        "spearman_ci": percentile_ci(rho_vals),
        "n_boot": n_boot,
        "n_degenerate": n_degenerate,
    }


def _ols_cluster(X: np.ndarray, y: np.ndarray, cluster_ids: np.ndarray):
    """OLS slope + Liang-Zeger cluster-robust (CR1) se for X=[1, x]."""
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    G = len(np.unique(cluster_ids))
    n, k = X.shape
    meat = np.zeros((k, k))
    for g in np.unique(cluster_ids):
        m = cluster_ids == g
        s = X[m].T @ u[m]
        meat += np.outer(s, s)
    c = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 else 1.0
    V = c * (XtX_inv @ meat @ XtX_inv)
    se = float(np.sqrt(V[1, 1]))
    return beta, se, u


def wild_cluster_bootstrap_t(
    x: np.ndarray, y: np.ndarray, clusters: np.ndarray,
    n_boot: int, rng: np.random.Generator,
) -> dict:
    """Wild cluster bootstrap-t CI for the standardized-regression slope
    (= Pearson r in the original sample). Rademacher weights per cluster,
    unrestricted residuals, bootstrap-t interval (Cameron & Miller 2015 s.VI).
    """
    zx = (x - x.mean()) / x.std()
    zy = (y - y.mean()) / y.std()
    X = np.column_stack([np.ones_like(zx), zx])
    _, cluster_codes = np.unique(clusters, return_inverse=True)
    G = cluster_codes.max() + 1

    beta_hat, se_hat, u_hat = _ols_cluster(X, zy, cluster_codes)
    b_hat = float(beta_hat[1])

    t_stars = np.full(n_boot, np.nan)
    fitted = X @ beta_hat
    for b in range(n_boot):
        w = rng.choice([-1.0, 1.0], size=G)
        y_star = fitted + u_hat * w[cluster_codes]
        beta_s, se_s, _ = _ols_cluster(X, y_star, cluster_codes)
        if se_s > 0:
            t_stars[b] = (float(beta_s[1]) - b_hat) / se_s
    t_lo, t_hi = np.nanpercentile(t_stars, [2.5, 97.5])
    return {
        "slope_std": b_hat,
        "cluster_robust_se": se_hat,
        "iid_se": float(np.sqrt((1 - b_hat ** 2) / max(len(x) - 2, 1))),
        "n_clusters": int(G),
        "wcb_t_ci": [float(b_hat - t_hi * se_hat), float(b_hat - t_lo * se_hat)],
        "n_boot": n_boot,
    }


def wildboottest_crosscheck(
    x: np.ndarray, y: np.ndarray, clusters: np.ndarray,
    threshold: float, n_boot: int, seed: int,
) -> dict:
    """Optional cross-check with the `wildboottest` package: WCR-11 p-value
    for H0: slope(z-scored) = threshold. The installed API (0.3.2) only tests
    beta = 0, so we regress (zy - threshold*zx) on zx: that slope is
    (r - threshold), and its beta=0 test is exactly H0: r_slope = threshold.
    Guarded -- returns a note on failure."""
    try:
        import statsmodels.api as sm
        from wildboottest.wildboottest import wildboottest as wbt

        zx = (x - x.mean()) / x.std()
        zy = (y - y.mean()) / y.std()
        _, cluster_codes = np.unique(clusters, return_inverse=True)
        df = pd.DataFrame({"y_tilde": zy - threshold * zx, "zx": zx,
                           "cluster": cluster_codes.astype(np.int64)})
        model = sm.OLS(df["y_tilde"], sm.add_constant(df[["zx"]]))
        res = wbt(model, cluster=df["cluster"], param="zx", B=n_boot,
                  seed=seed, show=False)
        pval = float(res["p-value"].iloc[0])
        return {"available": True, "h0": f"slope_std = {threshold}",
                "p_value": pval, "B": n_boot,
                "method": "WCR-11 Rademacher, impose_null=True, "
                          "shifted-outcome reparameterization"}
    except Exception as exc:  # noqa: BLE001 -- optional dependency
        return {"available": False, "note": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def yoy_pair(df: pd.DataFrame, y0: int, y1: int, n_boot: int,
             rng: np.random.Generator, source: str) -> dict | None:
    a = df[df["season"] == y0].set_index("team_id")["dpi_mean"]
    b = df[df["season"] == y1].set_index("team_id")["dpi_mean"]
    common = a.index.intersection(b.index)
    if len(common) < 10:
        return None
    xa, xb = a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)
    r, r_p = stats.pearsonr(xa, xb)
    rho, rho_p = stats.spearmanr(xa, xb)
    boots_r, boots_rho = np.full(n_boot, np.nan), np.full(n_boot, np.nan)
    n = len(common)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.std(xa[idx]) == 0 or np.std(xb[idx]) == 0:
            continue
        boots_r[i] = stats.pearsonr(xa[idx], xb[idx])[0]
        boots_rho[i] = stats.spearmanr(xa[idx], xb[idx])[0]
    return {
        "window": f"{y0}->{y1}", "source": source, "n_teams": n,
        "pearson_r": round(float(r), 4), "pearson_p": float(r_p),
        "pearson_ci": [round(v, 4) for v in percentile_ci(boots_r)],
        "spearman_rho": round(float(rho), 4), "spearman_p": float(rho_p),
        "spearman_ci": [round(v, 4) for v in percentile_ci(boots_rho)],
    }


def split_half(game_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split games by game_pk % 2 within team-season; per-season cross-team
    correlation of half means, Spearman-Brown corrected."""
    g = game_df.copy()
    g["half"] = (g["game_pk"].astype(np.int64) % 2).map({0: "even", 1: "odd"})
    cell = g.groupby(["season", "team_id", "half"]).agg(
        dpi_half_mean=("dpi", "mean"), n_games=("dpi", "size"),
    ).reset_index()
    wide = cell.pivot_table(
        index=["season", "team_id"], columns="half",
        values=["dpi_half_mean", "n_games"],
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()

    rows = []
    for season, sub in wide.groupby("season"):
        sub = sub.dropna(subset=["dpi_half_mean_even", "dpi_half_mean_odd"])
        r_half, p = stats.pearsonr(
            sub["dpi_half_mean_even"], sub["dpi_half_mean_odd"])
        rho_half = stats.spearmanr(
            sub["dpi_half_mean_even"], sub["dpi_half_mean_odd"])[0]
        sb = 2 * r_half / (1 + r_half) if r_half > -1 else np.nan
        rows.append({
            "season": season, "n_teams": len(sub),
            "mean_games_even": round(float(sub["n_games_even"].mean()), 1),
            "mean_games_odd": round(float(sub["n_games_odd"].mean()), 1),
            "split_half_r": round(float(r_half), 4),
            "split_half_p": float(p),
            "spearman_rho_half": round(float(rho_half), 4),
            "spearman_brown_r": round(float(sb), 4),
            "meets_0707_bar": bool(sb >= 0.707),
        })
    return pd.DataFrame(rows), wide


def fisher_z_mean(rs: pd.Series) -> float:
    z = np.arctanh(np.clip(rs.astype(float), -0.999999, 0.999999))
    return float(np.tanh(z.mean()))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=None,
                    help="Clean xOut checkpoint outside models/ (default: "
                         f"extract git blob {DEFAULT_GIT_REV})")
    ap.add_argument("--git-rev", default=DEFAULT_GIT_REV)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--reuse-scores", action="store_true",
                    help="Reuse <out-dir>/game_dpi_2015_2025.csv if present")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot-team", type=int, default=2000)
    ap.add_argument("--n-boot-pairs", type=int, default=5000)
    ap.add_argument("--n-boot-wild", type=int, default=9999)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path, prov = resolve_model(args.model_path, args.git_rev)
    print(f"Model: {prov['origin']}\n  sha256={prov['sha256']}\n"
          f"  train_seasons={prov['train_seasons']} fitted_at={prov['fitted_at']}")
    model, meta = dp.load_xout_checkpoint(model_path)
    assert model is not None, "checkpoint failed to load into module state"

    # ---- score (or reuse) per-game DPI -----------------------------------
    game_csv = out_dir / "game_dpi_2015_2025.csv"
    if args.reuse_scores and game_csv.exists():
        print(f"Reusing {game_csv}")
        game_df = pd.read_csv(game_csv)
    else:
        conn = get_connection(read_only=True)
        try:
            print("Scoring per-game DPI 2015-2025 with the frozen model ...")
            game_df = score_game_dpi(conn, SEASONS)
        finally:
            conn.close()
        game_df.to_csv(game_csv, index=False)
        print(f"Wrote {game_csv} ({len(game_df)} rows)")

    # ---- season aggregation + parity check vs recorded CSV ---------------
    season_df = game_df.groupby(["season", "team_id"]).agg(
        dpi_mean=("dpi", "mean"), dpi_std=("dpi", "std"),
        n_games=("dpi", "size"),
    ).reset_index()

    recorded = pd.read_csv(RANKINGS_CSV)
    parity = season_df.merge(
        recorded[["team_id", "season", "dpi_mean"]],
        on=["team_id", "season"], suffixes=("_recomputed", "_recorded"),
    )
    parity["abs_diff"] = (
        parity["dpi_mean_recomputed"].round(3) - parity["dpi_mean_recorded"]
    ).abs()
    parity_stats = {
        "n_cells_compared": int(len(parity)),
        "max_abs_diff": float(parity["abs_diff"].max()),
        "n_cells_diff_gt_0p002": int((parity["abs_diff"] > 0.002).sum()),
    }
    print(f"Parity vs recorded 2023-2025 rankings: {parity_stats}")

    # ---- 1. YoY stability -------------------------------------------------
    yoy_rows = []
    for y0, y1 in [(2023, 2024), (2024, 2025)]:
        yoy_rows.append(yoy_pair(recorded, y0, y1, args.n_boot_pairs,
                                 np.random.default_rng(args.seed + y0),
                                 source="recorded_csv"))
    for y0 in range(2015, 2025):
        row = yoy_pair(season_df, y0, y0 + 1, args.n_boot_pairs,
                       np.random.default_rng(args.seed + 100 + y0),
                       source="recomputed_db")
        if row:
            yoy_rows.append(row)
    yoy_df = pd.DataFrame([r for r in yoy_rows if r])
    yoy_df.to_csv(out_dir / "yoy_stability.csv", index=False)

    # ---- 2. split-half reliability ---------------------------------------
    sh_df, halves_wide = split_half(game_df)
    sh_df.to_csv(out_dir / "split_half_reliability.csv", index=False)

    full_seasons = sh_df[sh_df["season"] != 2020]
    rel_all = fisher_z_mean(sh_df["spearman_brown_r"])
    rel_full = fisher_z_mean(full_seasons["spearman_brown_r"])
    shrinkage = {
        "spearman_brown_fisher_mean_all_seasons": round(rel_all, 4),
        "spearman_brown_fisher_mean_excl_2020": round(rel_full, 4),
        "reliability_bar": 0.707,
        "seasons_meeting_bar": int(sh_df["meets_0707_bar"].sum()),
        "n_seasons": int(len(sh_df)),
        "shrinkage_coefficient_full_season": round(rel_full, 4),
        "note": (
            "regressed_dpi = league_season_mean + R * (dpi_mean - "
            "league_season_mean); R = Spearman-Brown full-season reliability "
            "(Fisher-z mean over full 162-game seasons; 2020 uses its own "
            "season estimate because a 60-game season has ~30-game halves)"
        ),
    }

    season_df = season_df.merge(halves_wide, on=["season", "team_id"], how="left")
    league_mean = season_df.groupby("season")["dpi_mean"].transform("mean")
    per_season_R = sh_df.set_index("season")["spearman_brown_r"]
    R_col = season_df["season"].map(
        lambda s: per_season_R.loc[2020] if s == 2020 else rel_full
    )
    season_df["reliability_R"] = R_col.round(4)
    season_df["regressed_dpi"] = (
        league_mean + R_col * (season_df["dpi_mean"] - league_mean)
    ).round(4)

    # ---- 3a. team-season CIs (resample games within team-season) ---------
    ci_lo, ci_hi = [], []
    for _, row in season_df.iterrows():
        dpis = game_df[(game_df["season"] == row["season"])
                       & (game_df["team_id"] == row["team_id"])]["dpi"].to_numpy()
        team_hash = int(
            hashlib.md5(str(row["team_id"]).encode()).hexdigest()[:8], 16) % 997
        cell_rng = np.random.default_rng(
            args.seed + int(row["season"]) * 1000 + team_hash)
        means = cell_rng.choice(dpis, size=(args.n_boot_team, len(dpis)),
                                replace=True).mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        ci_lo.append(round(float(lo), 4))
        ci_hi.append(round(float(hi), 4))
    season_df["dpi_mean_ci_lo"] = ci_lo
    season_df["dpi_mean_ci_hi"] = ci_hi
    season_df["dpi_mean"] = season_df["dpi_mean"].round(4)
    season_df["dpi_std"] = season_df["dpi_std"].round(4)
    season_df.to_csv(out_dir / "team_season_dpi_2015_2025.csv", index=False)

    # ---- 3b. gate CI re-issue --------------------------------------------
    gates = [
        ("gate2_rp_2023_2024", "rp_proxy", [2023, 2024], 0.40, ">="),
        ("gate2_rp_pooled_2023_2025", "rp_proxy", [2023, 2024, 2025], 0.40, ">="),
        ("gate3_babip_2023_2024", "babip_against", [2023, 2024], -0.50, "<="),
        ("gate3_babip_pooled_2023_2025", "babip_against", [2023, 2024, 2025], -0.50, "<="),
        ("gate6_oaa_2023_2024", "team_oaa", [2023, 2024], 0.45, ">="),
        ("gate6_oaa_2025", "team_oaa", [2025], 0.45, ">="),
        ("gate6_oaa_pooled_2023_2025", "team_oaa", [2023, 2024, 2025], 0.45, ">="),
    ]
    gate_rows = []
    for name, ycol, window, threshold, direction in gates:
        sub = recorded[recorded["season"].isin(window)].dropna(
            subset=["dpi_mean", ycol])
        x = sub["dpi_mean"].to_numpy(float)
        y = sub[ycol].to_numpy(float)
        clusters = sub["team_id"].to_numpy()
        r = float(stats.pearsonr(x, y)[0])
        rho = float(stats.spearmanr(x, y)[0])
        pcb = pairs_cluster_bootstrap_r(
            x, y, clusters, args.n_boot_pairs,
            np.random.default_rng(args.seed))
        wcb = wild_cluster_bootstrap_t(
            x, y, clusters, args.n_boot_wild,
            np.random.default_rng(args.seed))
        xcheck = wildboottest_crosscheck(
            x, y, clusters, threshold, args.n_boot_wild, args.seed)

        if direction == ">=":
            point_pass = r >= threshold
            wcb_lower_clears = wcb["wcb_t_ci"][0] >= threshold
            pcb_lower_clears = pcb["pearson_ci"][0] >= threshold
        else:
            point_pass = r <= threshold
            wcb_lower_clears = wcb["wcb_t_ci"][1] <= threshold
            pcb_lower_clears = pcb["pearson_ci"][1] <= threshold

        rec = RECORDED_CIS.get(name, {})
        gate_rows.append({
            "gate": name, "y_metric": ycol,
            "window": "+".join(str(w) for w in window),
            "n": len(sub), "n_clusters": wcb["n_clusters"],
            "threshold": threshold, "direction": direction,
            "pearson_r": round(r, 4), "spearman_rho": round(rho, 4),
            "recorded_r": rec.get("recorded_r"),
            "recorded_ci_lo": (rec.get("recorded_ci") or [None, None])[0],
            "recorded_ci_hi": (rec.get("recorded_ci") or [None, None])[1],
            "pairs_cluster_ci_lo": round(pcb["pearson_ci"][0], 4),
            "pairs_cluster_ci_hi": round(pcb["pearson_ci"][1], 4),
            "pairs_cluster_spearman_ci_lo": round(pcb["spearman_ci"][0], 4),
            "pairs_cluster_spearman_ci_hi": round(pcb["spearman_ci"][1], 4),
            "wcb_t_ci_lo": round(wcb["wcb_t_ci"][0], 4),
            "wcb_t_ci_hi": round(wcb["wcb_t_ci"][1], 4),
            "cluster_robust_se": round(wcb["cluster_robust_se"], 4),
            "iid_se": round(wcb["iid_se"], 4),
            "se_inflation": round(wcb["cluster_robust_se"] / wcb["iid_se"], 3),
            "point_estimate_clears_threshold": point_pass,
            "wcb_ci_bound_clears_threshold": wcb_lower_clears,
            "pairs_ci_bound_clears_threshold": pcb_lower_clears,
            "wildboottest_p_at_threshold": xcheck.get("p_value"),
            "wildboottest_note": xcheck.get("note"),
        })
        print(f"{name}: r={r:.4f} pairs-cluster CI "
              f"[{pcb['pearson_ci'][0]:.4f}, {pcb['pearson_ci'][1]:.4f}] "
              f"WCB-t CI [{wcb['wcb_t_ci'][0]:.4f}, {wcb['wcb_t_ci'][1]:.4f}] "
              f"threshold {direction} {threshold} | point clears: {point_pass}, "
              f"WCB bound clears: {wcb_lower_clears}")
    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(out_dir / "gate_ci_reissue.csv", index=False)

    # ---- summary ----------------------------------------------------------
    import sklearn, duckdb, scipy  # noqa: E401
    try:
        import wildboottest as _wbt
        wbt_ver = getattr(_wbt, "__version__", "installed")
    except ImportError:
        wbt_ver = None
    summary = {
        "task": "WS3.1 DPI quick wins (YoY, split-half, cluster CIs)",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "model_provenance": prov,
        "environment": {
            "python": sys.version.split()[0],
            "sklearn": sklearn.__version__,
            "sklearn_pickle_note": (
                "checkpoint pickled under sklearn 1.8.0, loaded under "
                f"{sklearn.__version__}; fidelity verified by parity check below"
            ),
            "numpy": np.__version__, "scipy": scipy.__version__,
            "pandas": pd.__version__, "duckdb": duckdb.__version__,
            "wildboottest": wbt_ver,
        },
        "seed": args.seed,
        "bootstrap_sizes": {"team": args.n_boot_team,
                            "pairs_cluster": args.n_boot_pairs,
                            "wild_cluster": args.n_boot_wild},
        "parity_check_vs_recorded_rankings": parity_stats,
        "yoy_stability": [r for r in yoy_rows if r],
        "split_half_reliability": {
            "per_season": sh_df.to_dict(orient="records"),
            **shrinkage,
        },
        "gate_ci_reissue": gate_rows,
        "notes": [
            "Gate thresholds are from docs/models/defensive_pressing_validation_spec.md; "
            "the Gate 6 0.45 threshold was retrofitted after the first "
            "measurement (spec:175-178, audit DPI finding 4).",
            "Pooled windows cluster on the 30 team franchises "
            "(few-cluster territory, Cameron & Miller JHR 2015).",
            "Single-season windows have one row per cluster; the cluster "
            "bootstraps reduce to their ordinary iid analogues there.",
        ],
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(f"Wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
