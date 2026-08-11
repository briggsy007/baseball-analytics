"""AdjustedWAR v3 -- regularized joint estimation of player value (WS4.2).

PRODUCTION player-value model since 2026-08-10 (Batch D, user-adjudicated):
registry ``adjusted_war_v3`` production alias ``v2026.08.10``.  The product
name on live surfaces is **AdjustedWAR** -- "regularized adjustment, NOT
causal identification".  The legacy CausalWAR formulation
(``src/analytics/causal_war.py``) keeps its filename and every historical
id (DB cache keys, pick-ledger product ids, frozen board artifacts); only
display strings were renamed.

Two standing constraints on anything quoted from this model:

* **K6 (no edge claim vs Marcel).**  Board surfaces state: beats
  matched-naive (+6.5pp mean across 17 fully-OOS windows); does not beat
  the Marcel-picker (-8pp, batter channel); ties Marcel on season-forward
  forecast -- no edge claim vs Marcel.
* **No confidence intervals ship.**  WS4.7 coverage-validated both
  uncertainty layers (``scripts/adjusted_war_v3_uncertainty.py``) against
  realized next-season outcomes: sampling-error 49.6%, ridge-posterior
  71.3% empirical coverage at a nominal 95%, both outside the
  pre-registered [90%, 98%] ship gate.  Per-player AdjustedWAR values are
  displayed as point estimates only until a construction passes.

``frozen_validated`` is deliberately UNSET: no validation spec exists for
this model yet, so no gate suite has been passed.

Design (research verdict adopted in the 2026-08-10 plan: regularized joint
estimation, NOT per-player DML; precedents RAPM/Sill 2010, nflWAR,
BP DRC+):

* Unit: one row per plate appearance, outcome = per-PA wOBA value
  (``SUM(woba_value)`` over the PA's pitch rows, PA-ending rows only,
  identical to the CausalWAR extraction filter ``woba_denom > 0``).
* Sparse one-hot design, four blocks:
    - ``batter_id``      (identity block, ridge-penalized with lambda)
    - ``pitcher_id``     (identity block, ridge-penalized with lambda)
    - ``park x stand``   (``pitches.home_team`` x batter handedness;
      ``pitches.home_team`` is 100% populated 2015-2026 -- the honest park
      carrier; the ``games``-join venue is a dead 0-row table)
    - context FE: times-facing-pitcher (2nd / 3rd+), platoon advantage,
      month (May..Sep+ vs Apr-or-earlier), game temperature bucket
      (cold < 60F / hot > 75F / missing vs mild) where present
      (``game_weather`` coverage is partial; missing is its own level,
      never imputed).
  The park/context blocks carry a small fixed penalty ``lam_fe`` (default
  10.0 -- shrinkage factor n/(n+10) is negligible at the thousands of PAs
  every park-stand cell has) purely to resolve the one-hot dummy trap;
  the identity blocks carry the real ``lam_identity``.
* Outcome is centered at the fit sample's per-PA mean; the block-level
  constant migrates to the (nearly unpenalized) park/context blocks, so
  batter/pitcher coefficients are centered skill estimates in wOBA points
  per PA.  Positive batter coefficient = adds wOBA vs an average context;
  NEGATIVE pitcher coefficient = suppresses wOBA (good pitcher).
* ``lam_identity`` is chosen by SEASON-FORWARD CV (Sill 2010 pattern --
  fit season t, predict season t+1 player wOBA, PA-weighted RMSE; never
  random K-fold).  See ``season_forward_lambda_cv``.
* WAR-like scaling for board rank compatibility uses the CausalWAR
  constants (wOBA scale 1.25, 10 runs/win): value = coef * PA / 1.25 / 10
  for batters, ``-coef * PA_faced / 1.25 / 10`` for pitchers.  Ranks are
  invariant to the constants; they exist only for readability.

Solver: normal equations on the sparse Gram matrix (X'X dense ~2-4k
columns per fit, SPD solve).  The Gram is cached per design so a lambda
grid costs one Gram + cheap solves.  CPU only.

Read-only DB access throughout; creates no tables or views.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse as sp

logger = logging.getLogger(__name__)

# CausalWAR display constants (causal_war.py) -- rank-invariant scaling only.
WOBA_SCALE = 1.25
RUNS_PER_WIN = 10.0

DEFAULT_LAM_FE = 10.0

# Season-forward CV grid (log-spaced; the EB-plausible shrinkage window --
# per-PA wOBA noise variance ~0.3 over batter skill variance ~1e-3 puts the
# optimum in the low hundreds; the grid brackets it by >1 order each side).
DEFAULT_LAM_GRID: tuple[float, ...] = (
    25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0,
    6400.0, 12800.0, 25600.0, 51200.0,
)

CONTEXT_COLS: tuple[str, ...] = (
    "tto_2", "tto_3plus", "platoon",
    "month_5", "month_6", "month_7", "month_8", "month_9",
    "temp_cold", "temp_hot", "temp_missing",
)


# ---------------------------------------------------------------------------
# Extraction (read-only)
# ---------------------------------------------------------------------------

def extract_ridge_pa(
    conn,
    season: int,
    cache_dir: Path | str | None = None,
) -> pd.DataFrame:
    """One row per PA for *season* with the AdjustedWAR v3 covariates.

    PA definition identical to ``causal_war._extract_pa_data``: group the
    pitch rows by (batter, pitcher, game_pk, game_date, at_bat_number,
    stand, p_throws) with ``pitch_type IS NOT NULL``, keep PA-ending rows
    (``woba_denom > 0``).  ``home_team`` comes from the pitch rows
    themselves (100% populated), never the dead ``games`` join.

    Args:
        conn: Read-only DuckDB connection.
        season: Season year.
        cache_dir: Optional parquet cache directory (scratch); a cached
            season is loaded instead of re-queried.

    Returns:
        DataFrame [batter_id, pitcher_id, game_pk, at_bat_number, stand,
        p_throws, park, month, temp_f, woba_value].
    """
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"ridge_pa_{int(season)}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

    df = conn.execute(
        """
        SELECT
            p.batter_id,
            p.pitcher_id,
            p.game_pk,
            p.at_bat_number,
            p.stand,
            p.p_throws,
            MAX(p.home_team) AS park,
            EXTRACT(MONTH FROM p.game_date)::INTEGER AS month,
            MAX(gw.temp_f) AS temp_f,
            SUM(COALESCE(p.woba_value, 0)) AS woba_value,
            MAX(COALESCE(p.woba_denom, 0)) AS woba_denom
        FROM pitches p
        LEFT JOIN game_weather gw ON p.game_pk = gw.game_pk
        WHERE EXTRACT(YEAR FROM p.game_date) = $1
          AND p.pitch_type IS NOT NULL
        GROUP BY p.batter_id, p.pitcher_id, p.game_pk, p.at_bat_number,
                 p.stand, p.p_throws, EXTRACT(MONTH FROM p.game_date)
        HAVING MAX(COALESCE(p.woba_denom, 0)) > 0
        ORDER BY p.game_pk, p.at_bat_number
        """,
        [int(season)],
    ).fetchdf()
    df["batter_id"] = df["batter_id"].astype("int64")
    df["pitcher_id"] = df["pitcher_id"].astype("int64")
    df["woba_value"] = df["woba_value"].astype("float64")
    df = df.drop(columns=["woba_denom"])
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------

def _context_matrix(df: pd.DataFrame) -> np.ndarray:
    """Dense (n, len(CONTEXT_COLS)) context feature block."""
    n = len(df)
    out = np.zeros((n, len(CONTEXT_COLS)), dtype=np.float64)
    col = {c: i for i, c in enumerate(CONTEXT_COLS)}

    # Times facing this pitcher in this game (1st time = reference).
    order = df.sort_values(["game_pk", "at_bat_number"], kind="mergesort")
    tto = order.groupby(
        ["game_pk", "pitcher_id", "batter_id"], sort=False
    ).cumcount().to_numpy() + 1
    tto_aligned = pd.Series(tto, index=order.index).reindex(df.index).to_numpy()
    out[:, col["tto_2"]] = (tto_aligned == 2).astype(float)
    out[:, col["tto_3plus"]] = (tto_aligned >= 3).astype(float)

    out[:, col["platoon"]] = (
        df["stand"].to_numpy() != df["p_throws"].to_numpy()
    ).astype(float)

    month = pd.to_numeric(df["month"], errors="coerce").fillna(6).astype(int)
    month = month.clip(lower=4, upper=9)  # Mar->Apr bucket, Oct->Sep bucket
    for m in (5, 6, 7, 8, 9):
        out[:, col[f"month_{m}"]] = (month == m).astype(float)

    temp = pd.to_numeric(df["temp_f"], errors="coerce")
    out[:, col["temp_cold"]] = (temp < 60).fillna(False).astype(float)
    out[:, col["temp_hot"]] = (temp > 75).fillna(False).astype(float)
    out[:, col["temp_missing"]] = temp.isna().astype(float)
    return out


@dataclass
class RidgeFit:
    """Solved coefficients for one AdjustedWAR v3 fit.

    Identifiability note (pre-registered mechanics, decided before any
    evaluation ran): every PA row carries exactly one batter and one
    pitcher indicator, so the two identity-block constants trade off 1:1
    -- only within-block contrasts are identified.  Both blocks are
    therefore re-centered to PA-weighted zero (``coef_centered``); the
    level belongs to the league constant.  The raw block offsets are kept
    as the aggregate-centering-drift diagnostic the plan mandates.
    Ranks are invariant to the shift.
    """

    seasons: tuple[int, ...]
    lam_identity: float
    lam_fe: float
    intercept: float               # per-PA mean wOBA of the fit sample
    batter: pd.DataFrame           # [player_id, coef, coef_centered, pa, adjusted_war_v3]
    pitcher: pd.DataFrame          # [player_id, coef, coef_centered, pa, adjusted_war_v3]
    parkstand: pd.DataFrame        # [label, coef, n_pa]
    context: pd.DataFrame          # [label, coef]
    n_pa: int
    centering_offset_batter: float = 0.0   # PA-weighted mean raw batter coef
    centering_offset_pitcher: float = 0.0  # PA-weighted mean raw pitcher coef

    def batter_map(self) -> dict[int, float]:
        return dict(zip(self.batter["player_id"].astype(int),
                        self.batter["coef_centered"].astype(float)))


class RidgeDesign:
    """Sparse design + cached Gram for one fit dataset (>=1 seasons pooled)."""

    def __init__(self, df: pd.DataFrame, seasons: Sequence[int]):
        self.seasons = tuple(int(s) for s in sorted(set(seasons)))
        self.n = len(df)
        if self.n == 0:
            raise ValueError("Empty PA frame")

        bat_codes, self.bat_ids = pd.factorize(df["batter_id"], sort=True)
        pit_codes, self.pit_ids = pd.factorize(df["pitcher_id"], sort=True)
        ps_label = (
            df["park"].fillna("UNK").astype(str)
            + "|" + df["stand"].fillna("?").astype(str)
        )
        ps_codes, self.ps_labels = pd.factorize(ps_label, sort=True)

        n = self.n
        ones = np.ones(n, dtype=np.float64)
        rows = np.arange(n)
        X_bat = sp.csr_matrix(
            (ones, (rows, bat_codes)), shape=(n, len(self.bat_ids)))
        X_pit = sp.csr_matrix(
            (ones, (rows, pit_codes)), shape=(n, len(self.pit_ids)))
        X_ps = sp.csr_matrix(
            (ones, (rows, ps_codes)), shape=(n, len(self.ps_labels)))
        X_ctx = sp.csr_matrix(_context_matrix(df))

        self.X = sp.hstack([X_bat, X_pit, X_ps, X_ctx], format="csr")
        self.block_slices = {}
        start = 0
        for name, width in (
            ("batter", len(self.bat_ids)),
            ("pitcher", len(self.pit_ids)),
            ("parkstand", len(self.ps_labels)),
            ("context", len(CONTEXT_COLS)),
        ):
            self.block_slices[name] = slice(start, start + width)
            start += width
        self.p = start

        y = df["woba_value"].to_numpy(dtype=np.float64)
        self.ybar = float(y.mean())
        self.y_centered = y - self.ybar

        self.bat_pa = np.bincount(bat_codes, minlength=len(self.bat_ids))
        self.pit_pa = np.bincount(pit_codes, minlength=len(self.pit_ids))
        self.ps_pa = np.bincount(ps_codes, minlength=len(self.ps_labels))

        self._gram: Optional[np.ndarray] = None
        self._xty: Optional[np.ndarray] = None

    def _ensure_gram(self) -> None:
        if self._gram is None:
            logger.info("Gram for seasons %s: %d PAs x %d cols",
                        self.seasons, self.n, self.p)
            self._gram = (self.X.T @ self.X).toarray()
            self._xty = np.asarray(self.X.T @ self.y_centered).ravel()

    def free_gram(self) -> None:
        self._gram = None
        self._xty = None

    def solve(self, lam_identity: float,
              lam_fe: float = DEFAULT_LAM_FE) -> RidgeFit:
        """Ridge solve with per-block penalties (identity blocks get
        ``lam_identity``; park/context get ``lam_fe``)."""
        self._ensure_gram()
        pen = np.full(self.p, float(lam_fe))
        pen[self.block_slices["batter"]] = float(lam_identity)
        pen[self.block_slices["pitcher"]] = float(lam_identity)
        A = self._gram.copy()
        A[np.diag_indices_from(A)] += pen
        beta = scipy.linalg.solve(A, self._xty, assume_a="pos")

        b_sl = self.block_slices["batter"]
        p_sl = self.block_slices["pitcher"]
        ps_sl = self.block_slices["parkstand"]
        c_sl = self.block_slices["context"]

        bat = pd.DataFrame({
            "player_id": np.asarray(self.bat_ids, dtype=np.int64),
            "coef": beta[b_sl],
            "pa": self.bat_pa.astype(int),
        })
        bat_offset = float(np.average(bat["coef"], weights=bat["pa"]))
        bat["coef_centered"] = bat["coef"] - bat_offset
        bat["adjusted_war_v3"] = (
            bat["coef_centered"] * bat["pa"] / WOBA_SCALE / RUNS_PER_WIN
        )
        pit = pd.DataFrame({
            "player_id": np.asarray(self.pit_ids, dtype=np.int64),
            "coef": beta[p_sl],
            "pa": self.pit_pa.astype(int),
        })
        pit_offset = float(np.average(pit["coef"], weights=pit["pa"]))
        pit["coef_centered"] = pit["coef"] - pit_offset
        pit["adjusted_war_v3"] = (
            -pit["coef_centered"] * pit["pa"] / WOBA_SCALE / RUNS_PER_WIN
        )
        ps = pd.DataFrame({
            "label": list(self.ps_labels),
            "coef": beta[ps_sl],
            "n_pa": self.ps_pa.astype(int),
        })
        ctx = pd.DataFrame({"label": list(CONTEXT_COLS), "coef": beta[c_sl]})
        return RidgeFit(
            seasons=self.seasons,
            lam_identity=float(lam_identity),
            lam_fe=float(lam_fe),
            intercept=self.ybar,
            batter=bat, pitcher=pit, parkstand=ps, context=ctx,
            n_pa=self.n,
            centering_offset_batter=bat_offset,
            centering_offset_pitcher=pit_offset,
        )


def build_design(frames: Iterable[pd.DataFrame],
                 seasons: Sequence[int]) -> RidgeDesign:
    """Pool >=1 season frames (from :func:`extract_ridge_pa`) into a design."""
    frames = list(frames)
    df = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    return RidgeDesign(df.reset_index(drop=True), seasons)


# ---------------------------------------------------------------------------
# Season-forward lambda CV (Sill 2010 pattern; never random K-fold)
# ---------------------------------------------------------------------------

def season_forward_lambda_cv(
    designs: dict[int, RidgeDesign],
    actuals: pd.DataFrame,
    league: dict[int, float],
    pairs: Sequence[tuple[int, int]],
    lam_grid: Sequence[float] = DEFAULT_LAM_GRID,
    lam_fe: float = DEFAULT_LAM_FE,
    pa_floor: float = 100.0,
) -> dict[str, Any]:
    """Choose ``lam_identity`` by season-forward prediction.

    For each candidate lambda and each pair ``(b, f=b+1)``: fit the season-b
    design, predict every batter's season-f wOBA as ``league[b] + coef``,
    and score PA-weighted RMSE against actual season-f wOBA over batters
    with season-f PA >= ``pa_floor``.  The winner minimizes the mean RMSE
    across pairs.  No follow-up season ever enters a fit.

    Args:
        designs: season -> RidgeDesign (must cover every ``b`` in pairs).
        actuals: frame [player_id, season, pa, woba] (marcel loader format).
        league: season -> PA-weighted league wOBA (baseline constant).
        pairs: (baseline, follow-up) season tuples, f = b + 1.
        lam_grid: candidate penalties.
        lam_fe: fixed park/context penalty.
        pa_floor: follow-up PA floor for the scoring pool.

    Returns:
        {"lambda_star", "path": [{"lambda", "mean_rmse", "per_pair"}],
         "pairs", "pa_floor"}
    """
    from src.analytics.marcel import pa_weighted_rmse

    path = []
    for lam in lam_grid:
        per_pair = {}
        for b, f in pairs:
            fit = designs[b].solve(lam, lam_fe=lam_fe)
            pred = fit.batter[["player_id", "coef_centered"]].copy()
            pred["pred_woba"] = league[b] + pred["coef_centered"]
            act = actuals[(actuals["season"] == f)
                          & (actuals["pa"] >= pa_floor)]
            m = act.merge(pred[["player_id", "pred_woba"]],
                          on="player_id", how="inner")
            per_pair[f"{b}->{f}"] = pa_weighted_rmse(
                m["woba"].to_numpy(), m["pred_woba"].to_numpy(),
                m["pa"].to_numpy(),
            )
        vals = [v for v in per_pair.values() if v == v]
        path.append({
            "lambda": float(lam),
            "mean_rmse": float(np.mean(vals)) if vals else float("nan"),
            "per_pair": {k: round(v, 6) for k, v in per_pair.items()},
        })
    best = min(path, key=lambda r: r["mean_rmse"])
    return {
        "lambda_star": best["lambda"],
        "path": path,
        "pairs": [list(p) for p in pairs],
        "pa_floor": pa_floor,
        "lam_fe": lam_fe,
        "selection": "min mean PA-weighted RMSE across season-forward pairs",
    }
