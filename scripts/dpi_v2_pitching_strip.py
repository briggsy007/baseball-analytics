"""DPI v2 stage 1: strip pitcher contact-management -- WS3.2 of the
2026-08-10 platform improvement plan (task C1a).

Remediates audit DPI finding 7 (pitcher contact-management beyond EV/LA/spray
lands in the "defense" residual) via the Swartz two-stage adjustment
(THT: tht.fangraphs.com/adjusting-defense-efficiency-by-the-quality-of-pitching/):

  Stage 0 (input): per-game per-defending-team DPI scored with the FROZEN
      xOut checkpoint (train 2015-2022), from the Batch-A artifact
      results/defensive_pressing/reliability_2026-08-10/game_dpi_2015_2025.csv.
  Stage 1 (this script): regress game-level DPI on the defending staff's
      leakage-free pitcher-season peripherals; DPI_v2_stage1 = residual.

Peripherals (plan 3.2 -- peripherals, NOT pitcher BABIP/xwOBAcon target
encodings; contact management is y2y-unreliable, peripherals are the stable
carrier):
  * K%   = (strikeout + strikeout_double_play) / PA
  * uBB% = walk / PA            (intentional walks excluded: manager choice)
  * net GB rate = (GB - FB) / PA
  * popup rate  = PU / PA
  * SP share    = fraction of the pitcher's season appearances that were
                  starts (continuous carrier of SP/RP role)
  PA = PA-ending pitches (events IS NOT NULL) excluding the non-PA
  administrative events truncated_pa / ejection / game_advisory.

LEAKAGE RULE (pre-declared, plan 3.2): a game's staff peripheral values are
computed LEAVE-ONE-GAME-OUT (LOGO) -- season totals minus that game's own
counts -- so no outcome from the scored game enters its own adjustment.
LOGO was chosen over prior-season history because (a) it keeps the
same-season contact profile that actually confounds that season's DPI,
(b) it has no rookie/mover missingness, and (c) K/BB/GB/FB/PU rates are
stable enough within season that the strip is not noise-dominated.
Pitcher-games whose LOGO denominator falls below MIN_LOGO_PA fall back to
the league-season mean rates (count reported).

Aggregation to the team-game: BIP-weighted mean of each pitcher's LOGO
peripherals, weights = the pitcher's BIP allowed in that game (DPI sums
over BIP, so BIP share is the correct exposure).

Stage-1 regression: per-season OLS (intercept + 5 peripherals) of game DPI
on the staff aggregate.  Per-season R^2 = variance explained by the
pitching stage (quantifies audit finding 7 at game level); a team-season
level share is also reported (var of team-season mean pitching-hat vs var
of team-season mean DPI).  The fit is descriptive/in-sample by construction
(the Swartz adjustment is a decomposition, not a forecast).

K2 interim measurement (kill criterion, plan section 8): partial
r(DPI_v2_stage1, team OAA | team BABIP-against) per year 2023/2024/2025 and
pooled 2023-25, alongside the v1 baseline partials for trajectory.  The
final K2 number lands after C1b adds park + speed.

Read-only vs the DB; writes only to --out-dir.

Usage:
    python scripts/dpi_v2_pitching_strip.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.db.schema import get_connection  # noqa: E402

GAME_DPI_CSV = (
    REPO_ROOT / "results" / "defensive_pressing" / "reliability_2026-08-10"
    / "game_dpi_2015_2025.csv"
)
RANKINGS_CSV = (
    REPO_ROOT / "results" / "defensive_pressing" / "2025_validation"
    / "team_rankings_all_years.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "defensive_pressing" / "v2_2026-08"

SEASONS = list(range(2015, 2026))
K2_SEASONS = [2023, 2024, 2025]

NON_PA_EVENTS = ("truncated_pa", "ejection", "game_advisory")
K_EVENTS = ("strikeout", "strikeout_double_play")
UBB_EVENTS = ("walk",)

MIN_LOGO_PA = 20  # below this, LOGO rates fall back to league-season means
PERIPHERALS = ["k_pct", "ubb_pct", "net_gb", "popup_rate", "sp_share"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# DB pulls (read-only)
# ---------------------------------------------------------------------------

def pull_pitcher_game_counts(conn) -> pd.DataFrame:
    """Per (season, game_pk, defending team, pitcher): PA / K / uBB / GB /
    FB / PU / BIP counts.  Filters mirror the game-DPI cohort exactly
    (events IS NOT NULL, inning_topbot in Top/Bot; defending team = home
    on 'Top' rows, away on 'Bot' rows)."""
    non_pa = ", ".join(f"'{e}'" for e in NON_PA_EVENTS)
    k_ev = ", ".join(f"'{e}'" for e in K_EVENTS)
    ubb_ev = ", ".join(f"'{e}'" for e in UBB_EVENTS)
    return conn.execute(
        f"""
        SELECT
            CAST(EXTRACT(YEAR FROM game_date) AS INT) AS season,
            game_pk,
            CASE WHEN inning_topbot = 'Top' THEN home_team
                 ELSE away_team END                    AS team_id,
            pitcher_id,
            COUNT(*) FILTER (WHERE events NOT IN ({non_pa}))       AS pa,
            COUNT(*) FILTER (WHERE events IN ({k_ev}))             AS k,
            COUNT(*) FILTER (WHERE events IN ({ubb_ev}))           AS ubb,
            COUNT(*) FILTER (WHERE type = 'X' AND bb_type = 'ground_ball') AS gb,
            COUNT(*) FILTER (WHERE type = 'X' AND bb_type = 'fly_ball')    AS fb,
            COUNT(*) FILTER (WHERE type = 'X' AND bb_type = 'popup')       AS pu,
            COUNT(*) FILTER (WHERE type = 'X')                     AS bip
        FROM pitches
        WHERE events IS NOT NULL
          AND inning_topbot IN ('Top', 'Bot')
          AND EXTRACT(YEAR FROM game_date) BETWEEN {SEASONS[0]} AND {SEASONS[-1]}
        GROUP BY 1, 2, 3, 4
        """
    ).fetchdf()


def pull_starts(conn) -> pd.DataFrame:
    """Per (game_pk, defending side): the starting pitcher (first pitch of
    that side's defensive half, ordered by at_bat_number then pitch_number)."""
    return conn.execute(
        f"""
        SELECT
            CAST(EXTRACT(YEAR FROM game_date) AS INT) AS season,
            game_pk,
            CASE WHEN inning_topbot = 'Top' THEN home_team
                 ELSE away_team END AS team_id,
            pitcher_id
        FROM pitches
        WHERE inning_topbot IN ('Top', 'Bot')
          AND EXTRACT(YEAR FROM game_date) BETWEEN {SEASONS[0]} AND {SEASONS[-1]}
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY game_pk, inning_topbot
            ORDER BY at_bat_number, pitch_number
        ) = 1
        """
    ).fetchdf()


# ---------------------------------------------------------------------------
# LOGO peripherals
# ---------------------------------------------------------------------------

def build_logo_peripherals(pg: pd.DataFrame, starts: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Leave-one-game-out peripherals per pitcher-game + fallback accounting."""
    starts = starts.copy()
    starts["is_start"] = 1
    pg = pg.merge(
        starts[["season", "game_pk", "team_id", "pitcher_id", "is_start"]],
        on=["season", "game_pk", "team_id", "pitcher_id"],
        how="left",
    )
    pg["is_start"] = pg["is_start"].fillna(0).astype(int)

    count_cols = ["pa", "k", "ubb", "gb", "fb", "pu", "bip"]
    tot = pg.groupby(["season", "pitcher_id"], sort=False).agg(
        **{f"{c}_tot": (c, "sum") for c in count_cols},
        n_games=("game_pk", "size"),
        n_starts=("is_start", "sum"),
    ).reset_index()
    pg = pg.merge(tot, on=["season", "pitcher_id"], how="left")

    for c in count_cols:
        pg[f"{c}_logo"] = pg[f"{c}_tot"] - pg[c]
    pg["starts_logo"] = pg["n_starts"] - pg["is_start"]
    pg["games_logo"] = pg["n_games"] - 1

    pa = pg["pa_logo"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pg["k_pct"] = np.where(pa > 0, pg["k_logo"] / pa, np.nan)
        pg["ubb_pct"] = np.where(pa > 0, pg["ubb_logo"] / pa, np.nan)
        pg["net_gb"] = np.where(pa > 0, (pg["gb_logo"] - pg["fb_logo"]) / pa, np.nan)
        pg["popup_rate"] = np.where(pa > 0, pg["pu_logo"] / pa, np.nan)
        pg["sp_share"] = np.where(
            pg["games_logo"] > 0, pg["starts_logo"] / pg["games_logo"], np.nan
        )

    # League-season fallback means (from season TOTALS pooled over pitchers,
    # i.e. PA-weighted league rates).
    lg = pg.drop_duplicates(["season", "pitcher_id"]).groupby("season").agg(
        pa=("pa_tot", "sum"), k=("k_tot", "sum"), ubb=("ubb_tot", "sum"),
        gb=("gb_tot", "sum"), fb=("fb_tot", "sum"), pu=("pu_tot", "sum"),
        starts=("n_starts", "sum"), games=("n_games", "sum"),
    )
    lg["k_pct"] = lg["k"] / lg["pa"]
    lg["ubb_pct"] = lg["ubb"] / lg["pa"]
    lg["net_gb"] = (lg["gb"] - lg["fb"]) / lg["pa"]
    lg["popup_rate"] = lg["pu"] / lg["pa"]
    lg["sp_share"] = lg["starts"] / lg["games"]

    fallback_mask = pg["pa_logo"] < MIN_LOGO_PA
    n_fallback_rows = int(fallback_mask.sum())
    bip_w_fallback = float(pg.loc[fallback_mask, "bip"].sum())
    bip_w_total = float(pg["bip"].sum())
    for col in PERIPHERALS:
        season_means = pg["season"].map(lg[col])
        pg[col] = np.where(fallback_mask, season_means, pg[col])
        # sp_share can be NaN (games_logo==0) even when pa_logo >= MIN_LOGO_PA
        # is impossible in practice, but guard anyway:
        pg[col] = np.where(np.isnan(pg[col]), season_means, pg[col])

    accounting = {
        "min_logo_pa": MIN_LOGO_PA,
        "n_pitcher_game_rows": int(len(pg)),
        "n_rows_league_fallback": n_fallback_rows,
        "pct_rows_league_fallback": round(100.0 * n_fallback_rows / len(pg), 3),
        "pct_bip_weight_league_fallback": round(100.0 * bip_w_fallback / bip_w_total, 3),
        "league_season_means": {
            str(s): {c: round(float(lg.loc[s, c]), 5) for c in PERIPHERALS}
            for s in lg.index
        },
    }
    return pg, accounting


def aggregate_to_team_game(pg: pd.DataFrame) -> pd.DataFrame:
    """BIP-weighted staff peripherals per (season, game_pk, team_id)."""
    pg = pg[pg["bip"] > 0].copy()
    for col in PERIPHERALS:
        pg[f"w_{col}"] = pg[col] * pg["bip"]
    agg = pg.groupby(["season", "game_pk", "team_id"], sort=False).agg(
        staff_bip=("bip", "sum"),
        n_pitchers_bip=("pitcher_id", "nunique"),
        **{f"w_{c}": (f"w_{c}", "sum") for c in PERIPHERALS},
    ).reset_index()
    for col in PERIPHERALS:
        agg[f"staff_{col}"] = agg[f"w_{col}"] / agg["staff_bip"]
        agg.drop(columns=[f"w_{col}"], inplace=True)
    return agg


# ---------------------------------------------------------------------------
# Stage-1 regression
# ---------------------------------------------------------------------------

def stage1_per_season(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-season OLS of game DPI on staff peripherals; returns
    (game-level df with pitch_hat/residual, per-season regression table)."""
    feat_cols = [f"staff_{c}" for c in PERIPHERALS]
    rows = []
    out_frames = []
    for season, sub in df.groupby("season"):
        sub = sub.copy()
        X = np.column_stack(
            [np.ones(len(sub))] + [sub[c].to_numpy(float) for c in feat_cols]
        )
        y = sub["dpi"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        resid = y - yhat
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        sub["pitch_hat"] = yhat
        sub["dpi_v2_stage1"] = resid
        out_frames.append(sub)
        row = {
            "season": int(season),
            "n_games": int(len(sub)),
            "r2_game_level": round(r2, 5),
            "intercept": round(float(beta[0]), 5),
        }
        for i, c in enumerate(PERIPHERALS):
            row[f"coef_{c}"] = round(float(beta[i + 1]), 5)
        rows.append(row)
    return (
        pd.concat(out_frames, ignore_index=True),
        pd.DataFrame(rows).sort_values("season").reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Partial correlation (K2 interim)
# ---------------------------------------------------------------------------

def partial_r(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    """Partial Pearson r(x, y | z), one conditioning variable.

    CI: Fisher z with SE 1/sqrt(n-4); p: t with df = n-3.
    """
    n = len(x)
    Z = np.column_stack([np.ones(n), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    r = float(stats.pearsonr(rx, ry)[0])
    df = n - 3
    t = r * np.sqrt(df / max(1e-12, 1 - r**2))
    p = float(2 * stats.t.sf(abs(t), df))
    zr = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(n - 4)
    lo, hi = np.tanh(zr - 1.959964 * se), np.tanh(zr + 1.959964 * se)
    return {"r": round(r, 4), "n": n, "p": p,
            "fisher_ci": [round(float(lo), 4), round(float(hi), 4)]}


def partial_r_cluster_boot(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, clusters: np.ndarray,
    n_boot: int, rng: np.random.Generator,
) -> list[float]:
    """Pairs cluster bootstrap (resample teams) percentile CI for pooled
    partial r."""
    uniq = np.unique(clusters)
    idx_by = {c: np.flatnonzero(clusters == c) for c in uniq}
    vals = np.full(n_boot, np.nan)
    for b in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[c] for c in chosen])
        xb, yb, zb = x[idx], y[idx], z[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            continue
        n = len(xb)
        Z = np.column_stack([np.ones(n), zb])
        rx = xb - Z @ np.linalg.lstsq(Z, xb, rcond=None)[0]
        ry = yb - Z @ np.linalg.lstsq(Z, yb, rcond=None)[0]
        if np.std(rx) == 0 or np.std(ry) == 0:
            continue
        vals[b] = stats.pearsonr(rx, ry)[0]
    vals = vals[~np.isnan(vals)]
    return [round(float(np.percentile(vals, 2.5)), 4),
            round(float(np.percentile(vals, 97.5)), 4)]


def k2_block(
    season_df: pd.DataFrame, recorded: pd.DataFrame, dpi_col: str, label: str,
    seed: int, n_boot: int,
) -> list[dict]:
    """Per-year + pooled partial r(dpi_col, team_oaa | babip_against)."""
    merged = season_df.merge(
        recorded[["team_id", "season", "team_oaa", "babip_against"]],
        on=["team_id", "season"], how="inner",
    ).dropna(subset=[dpi_col, "team_oaa", "babip_against"])
    merged = merged[merged["season"].isin(K2_SEASONS)]
    out = []
    for season in K2_SEASONS:
        sub = merged[merged["season"] == season]
        res = partial_r(
            sub[dpi_col].to_numpy(float),
            sub["team_oaa"].to_numpy(float),
            sub["babip_against"].to_numpy(float),
        )
        out.append({"metric": label, "window": str(season), **res,
                    "cluster_boot_ci": None})
    boot_ci = partial_r_cluster_boot(
        merged[dpi_col].to_numpy(float),
        merged["team_oaa"].to_numpy(float),
        merged["babip_against"].to_numpy(float),
        merged["team_id"].to_numpy(),
        n_boot, np.random.default_rng(seed),
    )
    res = partial_r(
        merged[dpi_col].to_numpy(float),
        merged["team_oaa"].to_numpy(float),
        merged["babip_against"].to_numpy(float),
    )
    out.append({"metric": label, "window": "pooled_2023_2025", **res,
                "cluster_boot_ci": boot_ci})
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    game_df = pd.read_csv(GAME_DPI_CSV)
    game_csv_sha = sha256_of(GAME_DPI_CSV)
    print(f"Loaded {GAME_DPI_CSV.name}: {len(game_df)} team-games "
          f"(sha256 {game_csv_sha[:12]}...)")

    conn = get_connection(read_only=True)
    try:
        print("Pulling per-pitcher-game counts (read-only) ...")
        pg = pull_pitcher_game_counts(conn)
        print(f"  {len(pg)} pitcher-game rows")
        print("Pulling starter identities ...")
        starts = pull_starts(conn)
        print(f"  {len(starts)} game-side starter rows")
    finally:
        conn.close()

    pg, accounting = build_logo_peripherals(pg, starts)
    staff = aggregate_to_team_game(pg)

    # Join staff peripherals onto the frozen-model game DPI table.
    df = game_df.merge(staff, on=["season", "game_pk", "team_id"], how="left")
    n_unmatched = int(df["staff_bip"].isna().sum())
    bip_mismatch = df.dropna(subset=["staff_bip"])
    n_bip_mismatch = int((bip_mismatch["n_bip"] != bip_mismatch["staff_bip"]).sum())
    join_stats = {
        "n_team_games": int(len(df)),
        "n_unmatched_staff": n_unmatched,
        "n_bip_count_mismatch": n_bip_mismatch,
        "max_abs_bip_diff": float(
            (bip_mismatch["n_bip"] - bip_mismatch["staff_bip"]).abs().max()
        ),
    }
    print(f"Join: {join_stats}")
    df = df.dropna(subset=[f"staff_{c}" for c in PERIPHERALS]).copy()

    # Stage-1 regressions.
    df, reg_table = stage1_per_season(df)
    reg_table.to_csv(out_dir / "stage1_regressions_by_season.csv", index=False)
    keep_cols = [
        "season", "team_id", "game_pk", "n_bip", "dpi",
        "staff_bip", "n_pitchers_bip",
    ] + [f"staff_{c}" for c in PERIPHERALS] + ["pitch_hat", "dpi_v2_stage1"]
    df[keep_cols].to_csv(
        out_dir / "game_dpi_v2_stage1_2015_2025.csv", index=False
    )

    # Season aggregation (unweighted mean over games, production parity).
    season_df = df.groupby(["season", "team_id"]).agg(
        n_games=("dpi", "size"),
        dpi_mean_v1=("dpi", "mean"),
        pitch_hat_mean=("pitch_hat", "mean"),
        dpi_v2_stage1_mean=("dpi_v2_stage1", "mean"),
    ).reset_index()
    for c in ["dpi_mean_v1", "pitch_hat_mean", "dpi_v2_stage1_mean"]:
        season_df[c] = season_df[c].round(5)
    season_df.to_csv(out_dir / "team_season_dpi_v2_stage1.csv", index=False)

    # Team-season-level variance share of the pitching stage.
    var_rows = []
    for season, sub in season_df.groupby("season"):
        var_dpi = float(np.var(sub["dpi_mean_v1"], ddof=1))
        var_hat = float(np.var(sub["pitch_hat_mean"], ddof=1))
        r_hat_dpi = float(
            stats.pearsonr(sub["pitch_hat_mean"], sub["dpi_mean_v1"])[0]
        )
        var_rows.append({
            "season": int(season),
            "n_teams": int(len(sub)),
            "var_team_season_dpi_v1": round(var_dpi, 6),
            "var_team_season_pitch_hat": round(var_hat, 6),
            "pitch_hat_var_share": round(var_hat / var_dpi, 4),
            "r_pitch_hat_vs_dpi_v1": round(r_hat_dpi, 4),
        })
    var_df = pd.DataFrame(var_rows)
    var_df.to_csv(out_dir / "stage1_team_season_variance_share.csv", index=False)

    # K2 interim partials (2023-25): v1 baseline, v1 season-centered
    # (sensitivity: isolates the per-season centering that stage-1 residuals
    # get for free), and v2 stage-1.
    recorded = pd.read_csv(RANKINGS_CSV)
    league_mean = season_df.groupby("season")["dpi_mean_v1"].transform("mean")
    season_df["dpi_mean_v1_centered"] = season_df["dpi_mean_v1"] - league_mean

    k2_rows = []
    k2_rows += k2_block(season_df, recorded, "dpi_mean_v1",
                        "v1_frozen", args.seed, args.n_boot)
    k2_rows += k2_block(season_df, recorded, "dpi_mean_v1_centered",
                        "v1_frozen_season_centered", args.seed, args.n_boot)
    k2_rows += k2_block(season_df, recorded, "dpi_v2_stage1_mean",
                        "v2_stage1_pitching_strip", args.seed, args.n_boot)
    k2_df = pd.DataFrame(k2_rows)
    k2_df.to_csv(out_dir / "k2_interim_partial_r.csv", index=False)
    print("\nK2 interim partial r(DPI, OAA | BABIP-against):")
    print(k2_df.to_string(index=False))

    summary = {
        "task": "WS3.2 pitching strip (DPI v2 stage 1) -- C1a",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "game_dpi_csv": str(GAME_DPI_CSV),
            "game_dpi_csv_sha256": game_csv_sha,
            "game_dpi_model": "frozen xOut v2026.04.18 (train 2015-2022, "
                              "sha256 e689bff6ab069474...)",
            "rankings_csv": str(RANKINGS_CSV),
            "rankings_csv_sha256": sha256_of(RANKINGS_CSV),
        },
        "peripheral_definitions": {
            "pa": f"events IS NOT NULL minus {list(NON_PA_EVENTS)}",
            "k_pct": f"{list(K_EVENTS)} / PA",
            "ubb_pct": f"{list(UBB_EVENTS)} / PA (intent_walk excluded)",
            "net_gb": "(GB - FB) / PA, bb_type on type='X' rows",
            "popup_rate": "PU / PA, bb_type='popup' on type='X' rows",
            "sp_share": "LOGO share of season appearances that were starts "
                        "(start = first pitch of the side's defensive game)",
        },
        "leakage_rule": {
            "choice": "leave-one-game-out (LOGO): season totals minus the "
                      "scored game's own counts",
            "why": "keeps same-season contact profile (the actual confound), "
                   "no rookie/mover missingness; the scored game's outcomes "
                   "never enter its own adjustment",
            "fallback": f"LOGO PA < {MIN_LOGO_PA} -> league-season mean rates",
        },
        "logo_accounting": accounting,
        "join_stats": join_stats,
        "stage1_regressions_by_season": reg_table.to_dict(orient="records"),
        "team_season_variance_share": var_df.to_dict(orient="records"),
        "k2_interim_partial_r": k2_rows,
        "notes": [
            "Stage-1 fit is per-season OLS on the same season's games "
            "(descriptive decomposition, in-sample by construction; the "
            "underlying DPI for 2023-25 is OOS w.r.t. the frozen xOut "
            "train window 2015-2022).",
            "Stage-1 residuals are season-centered (OLS intercept); the "
            "v1_frozen_season_centered rows isolate how much of any pooled "
            "change comes from centering alone.",
            "Pooled partial-r rows carry both a Fisher CI (iid) and a "
            "pairs cluster bootstrap CI over the 30 team franchises.",
            "K2 adjudicates on the FINAL number after park + speed (C1b); "
            "these are interim trajectory measurements only.",
        ],
        "seed": args.seed,
        "n_boot": args.n_boot,
    }
    with open(out_dir / "pitching_strip_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, default=str)
    print(f"\nWrote {out_dir / 'pitching_strip_summary.json'}")


if __name__ == "__main__":
    main()
