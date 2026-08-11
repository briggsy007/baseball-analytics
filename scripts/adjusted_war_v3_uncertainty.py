#!/usr/bin/env python
"""WS4.7 (2026-08-10): AdjustedWAR v3 uncertainty -- two never-conflated
layers, coverage-validated BEFORE any CI is allowed to ship.

PRE-REGISTERED PROTOCOL.  This docstring was written in full before the
script was executed; nothing below (construction, pool, gate, or the
labelling of primary vs diagnostic) was chosen after seeing a coverage
number.  The plan clause this implements (plan 4.7) exists because spec
Ticket 4 -- "coverage-validate before any CI ships" -- was skipped in
every prior CausalWAR/AdjustedWAR run while CIs shipped anyway.

---------------------------------------------------------------------------
The two layers (reported separately, NEVER summed into one unlabelled "the
CI" -- the DRC+ bagging category error)
---------------------------------------------------------------------------

(a) SAMPLING ERROR -- openWAR pattern: resample, with replacement, the PAs
    a batter actually took, and re-derive that batter's value from the
    resampled PA set.  Answers: "given the fitted league/opponent/park
    structure, how much of this player's number is which PAs happened to
    occur?"

    Exact algebra used (no approximation of the ridge solve itself).  For
    the ridge normal equations ``(X'X + Lambda) beta = X'y_c`` with a
    one-hot batter block, row j gives

        beta_j = sum_{i in PA(j)} r_i / (n_j + lambda),
        r_i    = y_c,i - (x_i'beta - beta_j)

    i.e. the batter coefficient IS the sum of that batter's partial
    residuals (outcome minus the fitted pitcher / park-stand / context
    contributions) shrunk by ``n_j + lambda``.  The bootstrap resamples the
    n_j partial residuals of batter j with replacement, B times, and
    recomputes ``sum(r*) / (n_j + lambda)``.  This is conditional on the
    other blocks being held at their fitted values -- stated as a
    limitation, not hidden: it is sampling error in the player's own PA
    set, which is exactly what layer (a) is defined to be.

    B = 2000 replicates (>= the 1000 floor the plan sets; openWAR's own
    reference is ~3500).  Seed 42.  Percentile interval (2.5 / 97.5).

(b) TALENT UNCERTAINTY -- the ridge posterior.  Ridge with penalty
    ``lambda`` on the identity blocks is the posterior mode of a Gaussian
    model ``y = X beta + eps``, ``eps ~ N(0, sigma^2)``, with the
    lambda-implied Gaussian prior ``beta_identity ~ N(0, sigma^2/lambda)``
    (and ``sigma^2/lambda_fe`` on the park/context blocks).  The posterior
    covariance is ``sigma^2 (X'X + Lambda)^{-1}``.

        sigma^2_hat = RSS / (n - edf),  edf = trace(G (G + Lambda)^{-1})

    Because the reported batter coefficient is PA-weight-CENTERED within
    the batter block (``coef_centered = beta_j - w'beta_bat``), the
    posterior variance of the *reported* quantity is computed exactly:

        Var(beta_j - w'beta_bat) = S_jj - 2 (S w)_j + w' S w

    with S the batter-block submatrix of the posterior covariance and w the
    PA weights (sum 1).  Gaussian interval, +/- 1.959964 sd.

---------------------------------------------------------------------------
Coverage validation (the gate)
---------------------------------------------------------------------------

Frames: the Batch C forward-eval prediction frames
``results/adjusted_war_v3/forward_eval_2026-08-10/predictions_{b}_to_{f}.csv``
for (b, f) = (2023, 2024) and (2024, 2025) -- the only fully held-out
season-forward frames on record.  Primary pool = that frame's primary
floor, follow-up PA >= 100 (identical to Batch C).  Test quantity =
whether the batter's REALIZED next-season season-aggregate wOBA falls
inside the nominal 95% interval placed around the season-b prediction
``pred = league_b + coef_centered``.

Pre-registered scope note (written before the run, because it is a
property of the design and not of the result): this is a PREDICTIVE
coverage test.  The interval nominally covers the season-b value
parameter; the realized season-(b+1) wOBA additionally contains true
talent change (aging, role, health) and the follow-up season's own
sampling noise.  A layer that under-covers here is not thereby "wrong
about season b" -- but under-coverage is exactly the failure mode that
makes a CI misleading on a product surface, which is why the plan puts
the gate here.  The two diagnostic constructions below quantify how much
of any gap is attributable to the unmodelled components.

SHIP GATE (pre-registered, applies per layer, no post-hoc softening):
    a layer's 95% CI may be rendered on ANY surface only if that layer's
    pooled empirical coverage lands in [90%, 98%].  Otherwise NO CI from
    that layer ships and the measured coverage is published in
    ``docs/models/adjusted_war_v3_2026-08.md`` with the reason.

Reported alongside, explicitly LABELLED DIAGNOSTICS and NOT shipping
candidates in this run (they are different constructions, not a
conflation of (a) and (b)):
    * ``ab_quadrature``  -- layers (a) and (b) added in quadrature.
    * ``predictive_full`` -- layer (b) plus the follow-up season's own
      sampling noise (league per-PA wOBA sd / sqrt(pa_f)) in quadrature;
      the honest *forecast* interval, shown to attribute the gap.

Also recorded: interval widths (mean/median half-width in wOBA points and
in the display AdjustedWAR scale), a reproduction check of this script's
ridge solve against the Batch C ``pred_ridge_1yr`` column, and a Wilson
95% interval on every coverage proportion.

Runtime environment: CPU only, DuckDB opened read_only=True, no DB writes,
no artifact writes.  Outputs go to
``results/adjusted_war_v3/uncertainty_2026-08-10/``.

Usage:  python scripts/adjusted_war_v3_uncertainty.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics.adjusted_war_v3 import (  # noqa: E402
    DEFAULT_LAM_FE,
    RUNS_PER_WIN,
    WOBA_SCALE,
    build_design,
    extract_ridge_pa,
)
from src.analytics.marcel import (  # noqa: E402
    league_woba_by_season,
    load_batter_season_inputs,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("awv3_uncertainty")

SCRATCH = Path(
    "C:/Users/hunte/AppData/Local/Temp/claude/"
    "C--Users-hunte-projects-baseball/"
    "5112f9f4-f7ff-4db3-bd24-90acc5fbef27/scratchpad/ridge_cache"
)
FWD_DIR = ROOT / "results" / "adjusted_war_v3" / "forward_eval_2026-08-10"
OUTDIR = ROOT / "results" / "adjusted_war_v3" / "uncertainty_2026-08-10"

# Pre-registered constants (all fixed before execution).
LAM_IDENTITY = 400.0        # lambda* from the Batch C season-forward CV
LAM_FE = DEFAULT_LAM_FE
HOLDOUTS = [(2023, 2024), (2024, 2025)]
N_BOOT = 2000
BOOT_SEED = 42
Z95 = 1.959963984540054
PRIMARY_FLOOR = 100.0
COVERAGE_GATE = (0.90, 0.98)
NOMINAL = 0.95


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson 95% interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(centre - half), float(centre + half))


def layer_a_sampling(design, fit, beta_full: np.ndarray,
                     n_boot: int, seed: int) -> pd.DataFrame:
    """Layer (a): per-PA bootstrap of each batter-season's own PA set.

    Exact ridge identity: ``beta_j = sum(partial residuals) / (n_j + lam)``.
    Resample the partial residuals of each batter with replacement.
    """
    b_sl = design.block_slices["batter"]
    n_bat = b_sl.stop - b_sl.start
    # Fitted values and per-row partial residual r_i (outcome minus every
    # block except this row's own batter coefficient).
    fitted = np.asarray(design.X @ beta_full).ravel()
    bat_col = np.asarray(
        design.X[:, b_sl].argmax(axis=1)).ravel().astype(np.int64)
    own = beta_full[b_sl][bat_col]
    partial_resid = design.y_centered - (fitted - own)

    order = np.argsort(bat_col, kind="stable")
    resid_sorted = partial_resid[order]
    counts = np.bincount(bat_col, minlength=n_bat)
    starts = np.concatenate([[0], np.cumsum(counts)])

    rng = np.random.default_rng(seed)
    lo = np.full(n_bat, np.nan)
    hi = np.full(n_bat, np.nan)
    sd = np.full(n_bat, np.nan)
    for j in range(n_bat):
        n_j = int(counts[j])
        if n_j == 0:
            continue
        r = resid_sorted[starts[j]:starts[j + 1]]
        idx = rng.integers(0, n_j, size=(n_boot, n_j))
        draws = r[idx].sum(axis=1) / (n_j + LAM_IDENTITY)
        lo[j], hi[j] = np.percentile(draws, [2.5, 97.5])
        sd[j] = float(draws.std(ddof=1))
    offset = float(fit.centering_offset_batter)
    return pd.DataFrame({
        "player_id": fit.batter["player_id"].to_numpy(),
        "pa": fit.batter["pa"].to_numpy(),
        "coef_centered": fit.batter["coef_centered"].to_numpy(),
        "a_lo": lo - offset,
        "a_hi": hi - offset,
        "a_sd": sd,
    })


def layer_b_posterior(design, fit, beta_full: np.ndarray) -> pd.DataFrame:
    """Layer (b): lambda-implied Gaussian posterior on the ridge coefficients.

    Posterior covariance ``sigma^2 (X'X + Lambda)^{-1}``; the reported
    quantity is PA-weight-centered within the batter block, so its variance
    is ``S_jj - 2 (S w)_j + w' S w`` on the batter-block submatrix S.
    """
    design._ensure_gram()
    G = design._gram
    pen = np.full(design.p, float(LAM_FE))
    pen[design.block_slices["batter"]] = LAM_IDENTITY
    pen[design.block_slices["pitcher"]] = LAM_IDENTITY
    A = G.copy()
    A[np.diag_indices_from(A)] += pen

    fitted = np.asarray(design.X @ beta_full).ravel()
    rss = float(np.sum((design.y_centered - fitted) ** 2))
    Ainv = scipy.linalg.inv(A)
    edf = float(np.einsum("ij,ji->", G, Ainv))          # trace(G A^-1)
    n = int(design.n)
    sigma2 = rss / (n - edf)

    b_sl = design.block_slices["batter"]
    S = sigma2 * Ainv[b_sl, b_sl]
    pa = fit.batter["pa"].to_numpy(dtype=float)
    w = pa / pa.sum()
    Sw = S @ w
    wSw = float(w @ Sw)
    var_centered = np.diag(S) - 2.0 * Sw + wSw
    var_centered = np.clip(var_centered, 0.0, None)
    sd = np.sqrt(var_centered)
    coef_c = fit.batter["coef_centered"].to_numpy()
    return (
        pd.DataFrame({
            "player_id": fit.batter["player_id"].to_numpy(),
            "b_sd": sd,
            "b_lo": coef_c - Z95 * sd,
            "b_hi": coef_c + Z95 * sd,
        }),
        {"sigma2": sigma2, "sigma": float(np.sqrt(sigma2)), "rss": rss,
         "edf": edf, "n_pa": n, "n_cols": int(design.p)},
    )


def coverage(lo: np.ndarray, hi: np.ndarray, actual: np.ndarray) -> dict:
    inside = (actual >= lo) & (actual <= hi)
    k, n = int(inside.sum()), int(len(inside))
    w_lo, w_hi = wilson(k, n)
    return {
        "covered": k,
        "n": n,
        "coverage": round(k / n, 4) if n else None,
        "wilson95": [round(w_lo, 4), round(w_hi, 4)],
        "mean_half_width_woba": round(float(np.mean((hi - lo) / 2.0)), 5),
        "median_half_width_woba": round(float(np.median((hi - lo) / 2.0)), 5),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    conn = get_connection(read_only=True)

    inputs = load_batter_season_inputs(conn, range(2015, 2026))
    league = league_woba_by_season(inputs)

    out: dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "pre-registered in scripts/adjusted_war_v3_uncertainty.py "
                    "docstring (written before execution)",
        "lambda_identity": LAM_IDENTITY,
        "lambda_fe": LAM_FE,
        "n_bootstrap_replicates": N_BOOT,
        "bootstrap_seed": BOOT_SEED,
        "nominal_level": NOMINAL,
        "ship_gate_coverage_interval": list(COVERAGE_GATE),
        "primary_pool_floor_followup_pa": PRIMARY_FLOOR,
        "layers": {
            "a_sampling": "openWAR-style per-PA resample of each "
                          "batter-season's own PA set; exact ridge identity "
                          "beta_j = sum(partial residuals)/(n_j+lambda); "
                          "other blocks held at fitted values",
            "b_talent": "lambda-implied Gaussian posterior "
                        "sigma^2 (X'X+Lambda)^-1, variance of the "
                        "PA-weight-centered batter coefficient",
        },
        "diagnostic_constructions": {
            "ab_quadrature": "layers (a) and (b) in quadrature -- labelled "
                             "diagnostic, not a shipping candidate",
            "predictive_full": "layer (b) plus follow-up-season sampling "
                               "noise (league per-PA wOBA sd / sqrt(pa_f)) "
                               "in quadrature -- labelled diagnostic",
        },
        "per_season": {},
        "fit_diagnostics": {},
    }

    per_row_frames = []
    for b, f in HOLDOUTS:
        logger.info("Baseline %d -> follow-up %d", b, f)
        pa_frame = extract_ridge_pa(conn, b, cache_dir=SCRATCH)
        design = build_design([pa_frame], [b])
        fit = design.solve(LAM_IDENTITY, lam_fe=LAM_FE)
        beta_full = np.concatenate([
            fit.batter["coef"].to_numpy(), fit.pitcher["coef"].to_numpy(),
            fit.parkstand["coef"].to_numpy(), fit.context["coef"].to_numpy(),
        ])

        logger.info("  layer (a): %d bootstrap replicates over %d batters ...",
                    N_BOOT, len(fit.batter))
        lay_a = layer_a_sampling(design, fit, beta_full, N_BOOT, BOOT_SEED)
        logger.info("  layer (b): posterior covariance (p=%d) ...", design.p)
        lay_b, fit_diag = layer_b_posterior(design, fit, beta_full)

        # Follow-up-season sampling-noise scale (diagnostic layer only).
        sd_pa_league = float(np.std(pa_frame["woba_value"].to_numpy(), ddof=1))
        fit_diag["league_per_pa_woba_sd"] = round(sd_pa_league, 5)
        fit_diag["lambda_identity"] = LAM_IDENTITY
        out["fit_diagnostics"][f"fit_{b}"] = fit_diag

        # Batch C eval frame (identical pool).
        pred_csv = FWD_DIR / f"predictions_{b}_to_{f}.csv"
        pool = pd.read_csv(pred_csv)
        pool = pool[pool["pa_f"] >= PRIMARY_FLOOR].copy()

        m = (pool.merge(lay_a, on="player_id", how="inner")
                 .merge(lay_b, on="player_id", how="inner"))
        lg_b = league[b]

        # Reproduction check against Batch C's pred_ridge_1yr.
        repro = float(np.max(np.abs(
            (lg_b + m["coef_centered"]) - m["pred_ridge_1yr"])))

        act = m["woba_f"].to_numpy()
        a_lo = lg_b + m["a_lo"].to_numpy()
        a_hi = lg_b + m["a_hi"].to_numpy()
        b_lo = lg_b + m["b_lo"].to_numpy()
        b_hi = lg_b + m["b_hi"].to_numpy()

        sd_a = m["a_sd"].to_numpy()
        sd_b = m["b_sd"].to_numpy()
        sd_q = np.sqrt(sd_a ** 2 + sd_b ** 2)
        centre = lg_b + m["coef_centered"].to_numpy()
        q_lo, q_hi = centre - Z95 * sd_q, centre + Z95 * sd_q

        sd_follow = sd_pa_league / np.sqrt(m["pa_f"].to_numpy())
        sd_p = np.sqrt(sd_b ** 2 + sd_follow ** 2)
        p_lo, p_hi = centre - Z95 * sd_p, centre + Z95 * sd_p

        season_out = {
            "baseline": b,
            "followup": f,
            "n_pool": int(len(m)),
            "reproduction_check_max_abs_pred_delta_vs_batchC": repro,
            "shipping_candidates": {
                "a_sampling": coverage(a_lo, a_hi, act),
                "b_talent": coverage(b_lo, b_hi, act),
            },
            "diagnostics": {
                "ab_quadrature": coverage(q_lo, q_hi, act),
                "predictive_full": coverage(p_lo, p_hi, act),
            },
        }
        out["per_season"][f] = season_out

        m["followup_season"] = f
        m["baseline_season"] = b
        m["ci_a_lo"], m["ci_a_hi"] = a_lo, a_hi
        m["ci_b_lo"], m["ci_b_hi"] = b_lo, b_hi
        m["ci_ab_lo"], m["ci_ab_hi"] = q_lo, q_hi
        m["ci_pred_lo"], m["ci_pred_hi"] = p_lo, p_hi
        per_row_frames.append(m)

        design.free_gram()

    conn.close()

    allrows = pd.concat(per_row_frames, ignore_index=True)
    allrows.to_csv(OUTDIR / "coverage_rows.csv", index=False)

    act = allrows["woba_f"].to_numpy()
    pooled = {
        "n": int(len(allrows)),
        "seasons_evaluated": len(HOLDOUTS),
        "shipping_candidates": {
            "a_sampling": coverage(allrows["ci_a_lo"].to_numpy(),
                                   allrows["ci_a_hi"].to_numpy(), act),
            "b_talent": coverage(allrows["ci_b_lo"].to_numpy(),
                                 allrows["ci_b_hi"].to_numpy(), act),
        },
        "diagnostics": {
            "ab_quadrature": coverage(allrows["ci_ab_lo"].to_numpy(),
                                      allrows["ci_ab_hi"].to_numpy(), act),
            "predictive_full": coverage(allrows["ci_pred_lo"].to_numpy(),
                                        allrows["ci_pred_hi"].to_numpy(), act),
        },
    }

    verdicts = {}
    for layer, res in pooled["shipping_candidates"].items():
        cov = res["coverage"]
        ok = cov is not None and COVERAGE_GATE[0] <= cov <= COVERAGE_GATE[1]
        verdicts[layer] = {
            "pooled_coverage": cov,
            "gate": list(COVERAGE_GATE),
            "may_ship": bool(ok),
        }
    pooled["ship_verdicts"] = verdicts
    pooled["any_ci_may_ship"] = bool(any(v["may_ship"] for v in verdicts.values()))

    # Display-scale half widths (AdjustedWAR wins) for the record.
    pa = allrows["pa"].to_numpy(dtype=float)
    for tag, sd_col in (("a_sampling", "a_sd"), ("b_talent", "b_sd")):
        hw = Z95 * allrows[sd_col].to_numpy() * pa / WOBA_SCALE / RUNS_PER_WIN
        pooled["shipping_candidates"][tag]["mean_half_width_display_war"] = round(
            float(np.mean(hw)), 4)

    out["pooled"] = pooled
    (OUTDIR / "uncertainty_coverage.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", OUTDIR / "uncertainty_coverage.json")

    print("\nWS4.7 -- AdjustedWAR v3 uncertainty, coverage as it lands")
    print("=" * 78)
    for f, so in out["per_season"].items():
        print(f"  {so['baseline']}->{f}  n={so['n_pool']}  "
              f"(repro delta {so['reproduction_check_max_abs_pred_delta_vs_batchC']:.2e})")
        for grp in ("shipping_candidates", "diagnostics"):
            for k, v in so[grp].items():
                print(f"     {grp[:4]:>4s} {k:<16s} cov={v['coverage']:.4f} "
                      f"{v['wilson95']}  half-width={v['mean_half_width_woba']:.5f}")
    print("-" * 78)
    for grp in ("shipping_candidates", "diagnostics"):
        for k, v in pooled[grp].items():
            print(f"  POOLED {grp[:4]:>4s} {k:<16s} n={v['n']} "
                  f"cov={v['coverage']:.4f} {v['wilson95']} "
                  f"half-width={v['mean_half_width_woba']:.5f}")
    print("-" * 78)
    for layer, v in verdicts.items():
        print(f"  GATE {layer:<16s} coverage {v['pooled_coverage']} in "
              f"[{COVERAGE_GATE[0]}, {COVERAGE_GATE[1]}]? "
              f"{'MAY SHIP' if v['may_ship'] else 'NO CI SHIPS'}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
