"""Unit tests for the AdjustedWAR v3 ridge module (WS4.2, 2026-08-10).

Synthetic-only: no DB, no artifacts, no models/ writes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.adjusted_war_v3 import (
    CONTEXT_COLS,
    DEFAULT_LAM_FE,
    RidgeDesign,
    build_design,
    season_forward_lambda_cv,
)


def _synth_frame(seed: int = 0, n: int = 12000, nb: int = 40, npi: int = 25,
                 npark: int = 5, noise: float = 0.45,
                 true_b: np.ndarray | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.RandomState(seed)
    bat = rng.randint(0, nb, n)
    pit = rng.randint(0, npi, n)
    park = rng.randint(0, npark, n)
    if true_b is None:
        true_b = rng.normal(0, 0.03, nb)
    true_p = rng.normal(0, 0.02, npi)
    true_pk = rng.normal(0, 0.01, npark)
    y = 0.320 + true_b[bat] + true_p[pit] + true_pk[park] + rng.normal(0, noise, n)
    df = pd.DataFrame({
        "batter_id": bat + 1000,
        "pitcher_id": pit + 2000,
        "game_pk": rng.randint(0, 400, n),
        "at_bat_number": np.arange(n),
        "stand": np.where(rng.rand(n) < 0.5, "R", "L"),
        "p_throws": np.where(rng.rand(n) < 0.5, "R", "L"),
        "park": pd.Series(park).map(lambda i: f"PK{i}"),
        "month": rng.randint(4, 10, n),
        "temp_f": np.where(rng.rand(n) < 0.3, np.nan, rng.uniform(45, 95, n)),
        "woba_value": y,
    })
    return df, true_b


def test_solver_matches_sklearn_uniform_penalty():
    """Normal-equations solve == sklearn Ridge when all penalties are equal."""
    sklearn = pytest.importorskip("sklearn.linear_model")
    df, _ = _synth_frame(seed=1, n=4000, nb=20, npi=12, npark=3)
    d = build_design([df], [2020])
    lam = 250.0
    # Uniform penalty via lam_fe == lam_identity.
    fit = d.solve(lam, lam_fe=lam)
    X = d.X.toarray()
    r = sklearn.Ridge(alpha=lam, fit_intercept=False).fit(X, d.y_centered)
    b_sl = d.block_slices["batter"]
    # Compare RAW (un-centered) batter coefficients.
    np.testing.assert_allclose(
        fit.batter["coef"].to_numpy(), r.coef_[b_sl], atol=1e-10,
    )


def test_recovery_and_shrinkage_direction():
    """Coefficients track true effects and higher lambda shrinks harder."""
    df, true_b = _synth_frame(seed=2)
    d = build_design([df], [2021])
    lo = d.solve(100.0)
    hi = d.solve(10000.0)
    m = lo.batter.copy()
    m["true"] = true_b[m["player_id"].to_numpy() - 1000]
    corr = np.corrcoef(m["coef_centered"], m["true"])[0, 1]
    assert corr > 0.5, f"batter recovery too weak: r={corr:.3f}"
    # Shrinkage: dispersion strictly decreases with lambda.
    assert hi.batter["coef_centered"].std() < lo.batter["coef_centered"].std()


def test_identity_blocks_recentered_and_ranks_stable():
    """coef_centered is PA-weighted zero-mean for both identity blocks and
    re-centering never changes within-block ordering."""
    df, _ = _synth_frame(seed=3)
    d = build_design([df], [2022])
    fit = d.solve(400.0)
    for blk in (fit.batter, fit.pitcher):
        wmean = float(np.average(blk["coef_centered"], weights=blk["pa"]))
        assert abs(wmean) < 1e-12
        raw_rank = blk["coef"].rank()
        cen_rank = blk["coef_centered"].rank()
        assert (raw_rank == cen_rank).all()


def test_design_shapes_and_block_layout():
    df, _ = _synth_frame(seed=4, n=3000, nb=15, npi=10, npark=4)
    d = build_design([df], [2019])
    n_ps = (df["park"].astype(str) + "|" + df["stand"]).nunique()
    assert d.X.shape == (3000, 15 + 10 + n_ps + len(CONTEXT_COLS))
    assert d.block_slices["batter"] == slice(0, 15)
    assert d.block_slices["pitcher"] == slice(15, 25)
    # Every row has exactly one batter, one pitcher, one park-stand hot.
    row_sums = np.asarray(
        d.X[:, : 25 + n_ps].sum(axis=1)
    ).ravel()
    np.testing.assert_allclose(row_sums, 3.0)


def test_season_forward_cv_picks_reasonable_lambda_and_never_uses_followup():
    """CV selects an interior lambda on a persistent-skill synthetic and the
    path covers exactly the requested grid."""
    rng = np.random.RandomState(7)
    true_b = rng.normal(0, 0.03, 40)
    frames = {}
    for season in (2018, 2019, 2020):
        df, _ = _synth_frame(seed=season, true_b=true_b)
        frames[season] = df
    designs = {s: build_design([f], [s]) for s, f in frames.items()}
    actual_rows = []
    for s, f in frames.items():
        g = f.groupby("batter_id")["woba_value"]
        agg = g.mean().reset_index().rename(
            columns={"batter_id": "player_id", "woba_value": "woba"})
        agg["pa"] = g.size().to_numpy()
        agg["season"] = s
        actual_rows.append(agg)
    actuals = pd.concat(actual_rows, ignore_index=True)
    league = {s: float(np.average(
        actuals[actuals["season"] == s]["woba"],
        weights=actuals[actuals["season"] == s]["pa"]))
        for s in frames}
    grid = (50.0, 400.0, 3200.0, 25600.0)
    out = season_forward_lambda_cv(
        designs, actuals, league,
        pairs=[(2018, 2019), (2019, 2020)],
        lam_grid=grid, pa_floor=50.0,
    )
    assert len(out["path"]) == len(grid)
    assert out["lambda_star"] in grid
    # With persistent skill the huge-lambda (all-league) forecast must not win.
    assert out["lambda_star"] < 25600.0
