"""
Volatility Surface entropy vs FIP-xFIP residual hypothesis test.

Tests whether low-entropy (predictable) MLB starters underperform their xFIP
in 2024 -- i.e. whether predictability shows up as a positive (FIP - xFIP)
residual.

H1: Pearson r(entropy, FIP - xFIP) < 0 (low entropy => positive residual,
    meaning the more predictable a pitcher is, the worse FIP runs vs xFIP).
    Equivalent threshold per spec: |r| > 0.15 with p < 0.05.
H0: |r| < 0.10.

Outputs:
- results/edges/volatility_predictability_2024.json
- results/edges/volatility_predictability_2024_scatter.png
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.analytics.volatility_surface import calculate_volatility_surface

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vol_pred_2024")

DB_PATH = r"C:\Users\hunte\projects\baseball\data\baseball.duckdb"
OUT_DIR = Path(r"C:\Users\hunte\projects\baseball\results\edges")
OUT_JSON = OUT_DIR / "volatility_predictability_2024.json"
OUT_PNG = OUT_DIR / "volatility_predictability_2024_scatter.png"

SEASON = 2024
MIN_IP = 100.0
MIN_GS_FRAC = 0.5  # at least half of appearances are starts


# ── League constants for FIP / xFIP ────────────────────────────────────────


def compute_league_constants(conn: duckdb.DuckDBPyConnection, season: int) -> dict:
    """Compute league-wide K, BB, HBP, HR, FB, IP for cFIP and HR/FB%.

    Uses pitch-level events for K/BB/HBP/HR/FB and season_pitching_stats.ip
    for total IP (the canonical source).
    """
    ev = conn.execute(
        """
        SELECT
          SUM(CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) AS K,
          SUM(CASE WHEN events IN ('walk','intent_walk') THEN 1 ELSE 0 END) AS BB,
          SUM(CASE WHEN events='hit_by_pitch' THEN 1 ELSE 0 END) AS HBP,
          SUM(CASE WHEN events='home_run' THEN 1 ELSE 0 END) AS HR
        FROM pitches WHERE EXTRACT(YEAR FROM game_date)=?
        """,
        [season],
    ).fetchone()

    fb = conn.execute(
        """
        SELECT SUM(CASE WHEN bb_type='fly_ball' THEN 1 ELSE 0 END) AS FB
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date)=? AND bb_type IS NOT NULL
        """,
        [season],
    ).fetchone()

    ip_row = conn.execute(
        "SELECT SUM(ip) FROM season_pitching_stats WHERE season=?",
        [season],
    ).fetchone()

    K, BB, HBP, HR = ev
    FB = fb[0]
    IP = ip_row[0]

    # FanGraphs-style cFIP = lgERA - (((13*HR)+(3*(BB+HBP))-(2*K)) / IP)
    # We approximate lgERA from earned runs; we don't have ER directly, so use
    # the standard convention that FIP is centered on lgERA. Use 4.08 (2024
    # MLB starter ERA proxy) as a documented constant, but the residual
    # FIP - xFIP is INVARIANT to the additive constant cFIP, so the choice
    # cancels out for our analysis.
    # We'll still compute and report cFIP for transparency.
    fip_term = ((13 * HR) + (3 * (BB + HBP)) - (2 * K)) / IP
    LGERA_2024 = 4.08
    cFIP = LGERA_2024 - fip_term

    hr_per_fb = HR / FB

    return {
        "season": season,
        "lg_K": int(K),
        "lg_BB": int(BB),
        "lg_HBP": int(HBP),
        "lg_HR": int(HR),
        "lg_FB": int(FB),
        "lg_IP": float(IP),
        "lg_HR_per_FB": float(hr_per_fb),
        "lg_ERA": LGERA_2024,
        "cFIP": float(cFIP),
    }


# ── Per-pitcher FIP / xFIP from raw pitches ──────────────────────────────


def compute_pitcher_peripherals(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: int,
) -> dict:
    """Return K, BB, HBP, HR, FB, BIP for a pitcher in a season."""
    row = conn.execute(
        """
        SELECT
          SUM(CASE WHEN events IN ('strikeout','strikeout_double_play') THEN 1 ELSE 0 END) AS K,
          SUM(CASE WHEN events IN ('walk','intent_walk') THEN 1 ELSE 0 END) AS BB,
          SUM(CASE WHEN events='hit_by_pitch' THEN 1 ELSE 0 END) AS HBP,
          SUM(CASE WHEN events='home_run' THEN 1 ELSE 0 END) AS HR,
          SUM(CASE WHEN bb_type='fly_ball' THEN 1 ELSE 0 END) AS FB
        FROM pitches
        WHERE pitcher_id=? AND EXTRACT(YEAR FROM game_date)=?
        """,
        [pitcher_id, season],
    ).fetchone()
    return {
        "K": int(row[0] or 0),
        "BB": int(row[1] or 0),
        "HBP": int(row[2] or 0),
        "HR": int(row[3] or 0),
        "FB": int(row[4] or 0),
    }


def fip_from_components(K, BB, HBP, HR, IP, cFIP):
    return ((13 * HR) + (3 * (BB + HBP)) - (2 * K)) / IP + cFIP


def xfip_from_components(K, BB, HBP, FB, IP, lg_hr_per_fb, cFIP):
    expected_hr = FB * lg_hr_per_fb
    return ((13 * expected_hr) + (3 * (BB + HBP)) - (2 * K)) / IP + cFIP


# ── Bootstrap CI ─────────────────────────────────────────────────────────


def bootstrap_corr_ci(x, y, fn, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    n_obs = len(x)
    rs = []
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        try:
            r = fn(x[idx], y[idx])
            rs.append(r)
        except Exception:
            continue
    rs = np.array(rs)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(DB_PATH, read_only=True)

    # 1. League constants
    logger.info("Computing 2024 league constants")
    lg = compute_league_constants(conn, SEASON)
    logger.info("cFIP=%.3f, HR/FB=%.4f", lg["cFIP"], lg["lg_HR_per_FB"])

    # 2. Cohort: 2024 starters with >=100 IP
    logger.info("Building cohort")
    cohort_df = conn.execute(
        """
        SELECT s.player_id, s.ip, s.gs, s.g, p.full_name
        FROM season_pitching_stats s
        LEFT JOIN players p ON p.player_id = s.player_id
        WHERE s.season=? AND s.ip>=? AND s.gs >= ? * s.g
        ORDER BY s.ip DESC
        """,
        [SEASON, MIN_IP, MIN_GS_FRAC],
    ).fetchdf()
    logger.info("Cohort size: %d pitchers", len(cohort_df))

    # 3. For each: compute entropy + FIP + xFIP
    rows = []
    for i, prow in cohort_df.iterrows():
        pid = int(prow["player_id"])
        name = prow["full_name"] or f"player_{pid}"
        ip = float(prow["ip"])
        if i % 10 == 0:
            logger.info("[%d/%d] %s (pid=%d, IP=%.1f)", i + 1, len(cohort_df), name, pid, ip)

        # Entropy
        try:
            vol = calculate_volatility_surface(conn, pid, SEASON)
            entropy = float(vol["overall_vol"])
            n_pitches = int(vol["n_pitches"])
        except Exception as exc:
            logger.warning("PIVS failed for %s (pid=%d): %s", name, pid, exc)
            continue
        if n_pitches < 300:
            logger.info("Skipping %s (only %d pitches)", name, n_pitches)
            continue

        # Peripherals -> FIP / xFIP
        per = compute_pitcher_peripherals(conn, pid, SEASON)
        if per["BB"] + per["K"] + per["HR"] == 0 or ip <= 0:
            continue
        fip = fip_from_components(per["K"], per["BB"], per["HBP"], per["HR"], ip, lg["cFIP"])
        xfip = xfip_from_components(
            per["K"], per["BB"], per["HBP"], per["FB"], ip, lg["lg_HR_per_FB"], lg["cFIP"]
        )

        rows.append(
            {
                "pitcher_id": pid,
                "name": name,
                "ip": ip,
                "n_pitches": n_pitches,
                "entropy": entropy,
                "K": per["K"],
                "BB": per["BB"],
                "HBP": per["HBP"],
                "HR": per["HR"],
                "FB": per["FB"],
                "fip": float(fip),
                "xfip": float(xfip),
                "fip_minus_xfip": float(fip - xfip),
            }
        )

    df = pd.DataFrame(rows).sort_values("entropy").reset_index(drop=True)
    logger.info("Final analytic n=%d", len(df))

    # 4. Correlations
    x = df["entropy"].values
    y = df["fip_minus_xfip"].values
    pearson_r, pearson_p = stats.pearsonr(x, y)
    spearman_rho, spearman_p = stats.spearmanr(x, y)

    pr_lo, pr_hi = bootstrap_corr_ci(
        x, y, lambda a, b: stats.pearsonr(a, b)[0], n=1000, seed=42
    )
    sr_lo, sr_hi = bootstrap_corr_ci(
        x, y, lambda a, b: stats.spearmanr(a, b)[0], n=1000, seed=42
    )

    logger.info("Pearson r = %.4f [%.4f, %.4f] p=%.4g", pearson_r, pr_lo, pr_hi, pearson_p)
    logger.info("Spearman rho = %.4f [%.4f, %.4f] p=%.4g", spearman_rho, sr_lo, sr_hi, spearman_p)

    # 5. Verdict (per spec):
    # PASS if r > 0.15 with p < 0.05 (positive relationship between entropy and fip-xfip)
    # NULL if |r| < 0.10
    # REJECTED if |r| >= 0.10 but does not meet PASS threshold (or wrong sign)
    abs_r = abs(pearson_r)
    if pearson_r > 0.15 and pearson_p < 0.05:
        verdict = "HYPOTHESIS PASS"
    elif abs_r < 0.10:
        verdict = "NULL"
    else:
        verdict = "REJECTED"
    logger.info("VERDICT: %s", verdict)

    # 6. 5 most predictable (lowest entropy) and 5 least predictable
    most_predictable = df.nsmallest(5, "entropy")[
        ["pitcher_id", "name", "ip", "entropy", "fip", "xfip", "fip_minus_xfip"]
    ].to_dict(orient="records")
    least_predictable = df.nlargest(5, "entropy")[
        ["pitcher_id", "name", "ip", "entropy", "fip", "xfip", "fip_minus_xfip"]
    ].to_dict(orient="records")

    # 7. Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.6, s=30, color="#2c7fb8", edgecolor="white", linewidth=0.5)
    # OLS regression line
    slope, intercept, *_ = stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, slope * xs + intercept, color="#d95f0e", linewidth=2,
            label=f"OLS fit: y={slope:.2f}x+{intercept:.2f}")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Volatility Surface entropy (higher = more unpredictable)")
    ax.set_ylabel("FIP - xFIP residual (positive = worse than peripherals predict)")
    ax.set_title(
        f"2024 Starters (n={len(df)}): Pitch entropy vs FIP-xFIP residual\n"
        f"Pearson r={pearson_r:+.3f} [{pr_lo:+.3f},{pr_hi:+.3f}]  p={pearson_p:.3g}  -> {verdict}"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    logger.info("Saved plot -> %s", OUT_PNG)

    # 8. Write JSON
    payload = {
        "season": SEASON,
        "cohort_filter": {
            "min_ip": MIN_IP,
            "min_gs_fraction_of_appearances": MIN_GS_FRAC,
            "min_pitches_for_entropy": 300,
        },
        "league_constants": lg,
        "n_starters_analyzed": len(df),
        "entropy_convention": "HIGH entropy = unpredictable, LOW entropy = predictable",
        "hypothesis_direction": (
            "H1: low entropy (predictable) -> positive FIP-xFIP residual; "
            "i.e. expect Pearson r > 0 if hitters can sit on patterns; "
            "spec PASS threshold r > 0.15, p < 0.05."
        ),
        "results": {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "pearson_ci95": [pr_lo, pr_hi],
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
            "spearman_ci95": [sr_lo, sr_hi],
            "verdict": verdict,
        },
        "most_predictable_5": most_predictable,
        "least_predictable_5": least_predictable,
        "per_pitcher": df.to_dict(orient="records"),
        "plot_path": str(OUT_PNG.relative_to(Path(r"C:\Users\hunte\projects\baseball"))).replace("\\", "/"),
        "wallclock_seconds": round(time.time() - t0, 1),
    }

    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    logger.info("Saved JSON -> %s", OUT_JSON)

    print()
    print("=" * 70)
    print(f"VERDICT: {verdict}")
    print(f"Pearson r = {pearson_r:+.4f} [{pr_lo:+.4f}, {pr_hi:+.4f}]  p={pearson_p:.4g}")
    print(f"Spearman rho = {spearman_rho:+.4f} [{sr_lo:+.4f}, {sr_hi:+.4f}]  p={spearman_p:.4g}")
    print(f"n = {len(df)} starters")
    print()
    print("MOST PREDICTABLE (lowest entropy):")
    for r in most_predictable:
        print(f"  {r['name']:25s}  ent={r['entropy']:.3f}  FIP={r['fip']:.2f}  xFIP={r['xfip']:.2f}  resid={r['fip_minus_xfip']:+.2f}")
    print()
    print("LEAST PREDICTABLE (highest entropy):")
    for r in least_predictable:
        print(f"  {r['name']:25s}  ent={r['entropy']:.3f}  FIP={r['fip']:.2f}  xFIP={r['xfip']:.2f}  resid={r['fip_minus_xfip']:+.2f}")
    print("=" * 70)

    conn.close()


if __name__ == "__main__":
    main()
