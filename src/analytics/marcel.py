"""Literal Marcel batter projections + the pre-registered forecast scoring
protocol.

Marcel (WS4.3, plan 2026-08-10)
-------------------------------
"The Marcel the Monkey Forecasting System" (Tom Tango,
tangotiger.net/marcel) -- the mandatory floor for anything
projection-flavored on this platform ("ALL forecasting systems should be
treated as if they are nothing more than Marcel, at best").  Implemented
LITERALLY, batter wOBA flavor, per the frozen 2026 resolution spec
section 5.6 restatement:

* season weights **5/4/3** (most recent first);
* regression: **+1200 PA of league average**;
* reliability ``rel = wPA / (wPA + 1200)`` with
  ``wPA = 5*PA_y1 + 4*PA_y2 + 3*PA_y3``;
* weighted player rate ``r = sum(w_y*PA_y*woba_y) / sum(w_y*PA_y)``;
* league rate = the (5,4,3)-PA-weighted blend of the per-season
  PA-weighted league mean wOBA;
* age adjustment (age as of June 30 of the projection season):
  multiplier ``1 + 0.006*(29 - age)`` if ``age < 29`` else
  ``1 + 0.003*(29 - age)``;
* PA projection ``0.5*PA_y1 + 0.1*PA_y2 + 200``.

Where this sketch and the reference implementation
(github.com/bdilday/marcelR) disagree in any detail, the frozen spec's
[TIEBREAK] says marcelR governs **for the 2026 M1 control run** -- that pin
happens (with its dated deviations-log entry) when the M1 boards are
generated, not here.  Pitcher Marcel is likewise deferred to that pin
(spec section 6.7).

Data sources (verified 2026-08-10, all read-only):

* ``season_batting_stats.pa`` -- populated; used for PA weights.
* ``season_batting_stats.woba`` -- **100% NULL in every season** (verified
  by direct query 2026-08-10), so per-batter-season wOBA is derived from
  Statcast pitch rows instead: ``SUM(woba_value)/SUM(woba_denom)`` over
  PA-ending rows (``woba_denom > 0``) in ``pitches``.  Real data, never
  imputed; batters with no pitch-level record in a season simply have no
  rate for that season.
* Birth dates -- the ``players`` table carries none; the pre-registered
  source (frozen spec section 5.6) is the MLB Stats API
  ``/api/v1/people?personIds=...`` ``birthDate`` (``players.player_id``
  IS the MLBAM id).  Fetched once and cached to
  ``data/cache/player_birthdates.parquet``; missing birth dates leave the
  age multiplier at exactly 1.0 with ``age_known=False`` (NULL stays
  NULL -- no fabricated ages).

Scoring protocol (WS4.4, plan 2026-08-10; consumed by K3/K6 and frozen
spec section 5.6 M2)
--------------------
* PA-weighted RMSE on wOBA vs the naive constant forecast (the baseline
  season's PA-weighted league wOBA);
* head-to-head W-L vs Marcel per player on ``|error|`` with a
  **0.010-wOBA tie band**;
* PA-weighted paired t-test, superiority requires **>= 90% confidence**;
* **>= 2 resolved seasons** before any superiority claim -- a single
  season CANNOT ground one, regardless of significance.

Realistic ceiling to pre-commit (plan 4.4): top public systems beat naive
by ~.02 wOBA and each other by far less.
"""
from __future__ import annotations

import json
import logging
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
BIRTHDATE_CACHE = _PROJECT_ROOT / "data" / "cache" / "player_birthdates.parquet"

# ---------------------------------------------------------------------------
# Marcel constants (tangotiger.net/marcel; frozen spec section 5.6)
# ---------------------------------------------------------------------------

MARCEL_WEIGHTS: tuple[float, float, float] = (5.0, 4.0, 3.0)
MARCEL_REGRESSION_PA: float = 1200.0
MARCEL_AGE_PIVOT: float = 29.0
MARCEL_AGE_COEF_YOUNG: float = 0.006   # per year below 29
MARCEL_AGE_COEF_OLD: float = 0.003     # per year above 29
MARCEL_PA_PROJ_COEFS: tuple[float, float, float] = (0.5, 0.1, 200.0)


# ---------------------------------------------------------------------------
# Core projection (pure function -- unit-testable without a DB)
# ---------------------------------------------------------------------------

@dataclass
class MarcelProjection:
    """One batter's Marcel projection for a target season."""

    player_id: int
    season: int                 # projection (target) season
    proj_woba: float
    proj_woba_raw: float        # before the age multiplier
    proj_pa: float
    reliability: float
    weighted_player_rate: Optional[float]   # None when wPA == 0
    weighted_league_rate: float
    age: Optional[float]
    age_known: bool
    age_multiplier: float
    w_pa: float                 # 5*PA1 + 4*PA2 + 3*PA3

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def marcel_batter_woba(
    player_id: int,
    season: int,
    pa_by_lag: tuple[float, float, float],
    woba_by_lag: tuple[Optional[float], Optional[float], Optional[float]],
    league_woba_by_lag: tuple[float, float, float],
    age: Optional[float] = None,
) -> MarcelProjection:
    """Literal Marcel wOBA projection for one batter.

    Args:
        player_id: MLBAM id (bookkeeping only).
        season: Target (projection) season.
        pa_by_lag: ``(PA_y1, PA_y2, PA_y3)`` = seasons ``season-1``,
            ``season-2``, ``season-3``.  Missing season -> 0.
        woba_by_lag: Player wOBA per lag season; ``None`` where the player
            has no record (its PA must then be 0 to contribute nothing).
        league_woba_by_lag: PA-weighted league mean wOBA of each lag season.
        age: Age (years, may be fractional) as of June 30 of *season*, or
            ``None`` when the birth date is unknown -> multiplier 1.0,
            ``age_known=False``.

    Returns:
        :class:`MarcelProjection`.
    """
    w1, w2, w3 = MARCEL_WEIGHTS
    pa = tuple(float(p or 0.0) for p in pa_by_lag)
    woba = tuple(float(x) if x is not None else 0.0 for x in woba_by_lag)
    lg = tuple(float(x) for x in league_woba_by_lag)

    weights = (w1, w2, w3)
    w_pa = sum(w * p for w, p in zip(weights, pa))

    if w_pa > 0:
        r = sum(w * p * x for w, p, x in zip(weights, pa, woba)) / w_pa
        lg_blend = sum(w * p * g for w, p, g in zip(weights, pa, lg)) / w_pa
        weighted_player_rate: Optional[float] = r
    else:
        # No history at all: pure league-average projection.  The blend
        # degenerates; use the most recent season's league rate.
        r = lg[0]
        lg_blend = lg[0]
        weighted_player_rate = None

    reliability = w_pa / (w_pa + MARCEL_REGRESSION_PA)
    proj_raw = reliability * r + (1.0 - reliability) * lg_blend

    if age is None or (isinstance(age, float) and math.isnan(age)):
        age_mult = 1.0
        age_known = False
        age_val: Optional[float] = None
    else:
        age_val = float(age)
        delta = MARCEL_AGE_PIVOT - age_val
        coef = MARCEL_AGE_COEF_YOUNG if age_val < MARCEL_AGE_PIVOT else MARCEL_AGE_COEF_OLD
        age_mult = 1.0 + coef * delta
        age_known = True

    c1, c2, c0 = MARCEL_PA_PROJ_COEFS
    proj_pa = c1 * pa[0] + c2 * pa[1] + c0

    return MarcelProjection(
        player_id=int(player_id),
        season=int(season),
        proj_woba=proj_raw * age_mult,
        proj_woba_raw=proj_raw,
        proj_pa=proj_pa,
        reliability=reliability,
        weighted_player_rate=weighted_player_rate,
        weighted_league_rate=lg_blend,
        age=age_val,
        age_known=age_known,
        age_multiplier=age_mult,
        w_pa=w_pa,
    )


# ---------------------------------------------------------------------------
# Data loaders (read-only DuckDB -> pandas; NO tables/views are created)
# ---------------------------------------------------------------------------

def load_batter_season_inputs(
    conn,
    seasons: Iterable[int],
) -> pd.DataFrame:
    """Per-batter-season (pa, woba) inputs for Marcel.

    PA from ``season_batting_stats.pa``; wOBA derived from Statcast pitch
    rows (``SUM(woba_value)/SUM(woba_denom)`` over PA-ending rows) because
    ``season_batting_stats.woba`` is 100% NULL (verified 2026-08-10).
    Rows lacking either source are dropped (disclosed by the caller);
    nothing is imputed.

    Returns:
        DataFrame ``[player_id, season, pa, woba]``.
    """
    seasons = sorted({int(s) for s in seasons})
    ph = ",".join(str(s) for s in seasons)
    sb = conn.execute(
        f"SELECT player_id, season, pa FROM season_batting_stats "
        f"WHERE season IN ({ph}) AND pa IS NOT NULL AND pa > 0"
    ).fetchdf()
    pw = conn.execute(
        f"""
        SELECT
            batter_id AS player_id,
            EXTRACT(YEAR FROM game_date)::INTEGER AS season,
            SUM(COALESCE(woba_value, 0)) / NULLIF(SUM(COALESCE(woba_denom, 0)), 0) AS woba,
            SUM(COALESCE(woba_denom, 0)) AS woba_denom_total
        FROM pitches
        WHERE woba_denom > 0
          AND EXTRACT(YEAR FROM game_date)::INTEGER IN ({ph})
        GROUP BY 1, 2
        """
    ).fetchdf()
    df = sb.merge(pw[["player_id", "season", "woba"]], on=["player_id", "season"], how="inner")
    df = df.dropna(subset=["woba"]).copy()
    df["player_id"] = df["player_id"].astype("int64")
    df["season"] = df["season"].astype("int64")
    df["pa"] = df["pa"].astype("float64")
    return df


def league_woba_by_season(inputs: pd.DataFrame) -> dict[int, float]:
    """PA-weighted league mean wOBA per season from the Marcel input frame."""
    out: dict[int, float] = {}
    for season, g in inputs.groupby("season"):
        out[int(season)] = float(np.average(g["woba"], weights=g["pa"]))
    return out


def fetch_birthdates(
    player_ids: Iterable[int],
    cache_path: Path | str = BIRTHDATE_CACHE,
    batch_size: int = 100,
    timeout: int = 30,
) -> pd.DataFrame:
    """Birth dates from the MLB Stats API, parquet-cached.

    Pre-registered source (frozen spec section 5.6): ``/api/v1/people`` --
    this DB's ``player_id`` IS the MLBAM id.  Ids already in the cache are
    not re-fetched.  Ids the API does not return are cached with a NULL
    ``birth_date`` (NULL stays NULL; never guessed).

    Returns:
        DataFrame ``[player_id, birth_date]`` (birth_date as datetime64,
        NaT where unknown) covering the requested ids.
    """
    ids = sorted({int(p) for p in player_ids})
    cache_path = Path(cache_path)
    if cache_path.exists():
        cache = pd.read_parquet(cache_path)
    else:
        cache = pd.DataFrame({"player_id": pd.Series(dtype="int64"),
                              "birth_date": pd.Series(dtype="datetime64[ns]")})
    known = set(cache["player_id"].astype(int).tolist())
    missing = [p for p in ids if p not in known]
    if missing:
        logger.info("Fetching %d birth dates from MLB Stats API ...", len(missing))
        rows: list[dict[str, Any]] = []
        for i in range(0, len(missing), batch_size):
            chunk = missing[i:i + batch_size]
            url = (
                "https://statsapi.mlb.com/api/v1/people?personIds="
                + ",".join(str(p) for p in chunk)
                + "&fields=people,id,birthDate"
            )
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.load(resp)
            returned = {int(p["id"]): p.get("birthDate") for p in data.get("people", [])}
            for pid in chunk:
                rows.append({
                    "player_id": pid,
                    "birth_date": pd.to_datetime(returned.get(pid), errors="coerce"),
                })
        fetched = pd.DataFrame(rows)
        cache = pd.concat([cache, fetched], ignore_index=True)
        cache = cache.drop_duplicates(subset="player_id", keep="first")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.to_parquet(cache_path, index=False)
        logger.info("Birth-date cache now %d rows -> %s", len(cache), cache_path)
    out = cache[cache["player_id"].isin(ids)].copy()
    out["player_id"] = out["player_id"].astype("int64")
    return out.reset_index(drop=True)


def age_asof_june30(birth_date: pd.Timestamp | None, season: int) -> Optional[float]:
    """Age in fractional years as of June 30 of *season* (None when unknown)."""
    if birth_date is None or pd.isna(birth_date):
        return None
    asof = pd.Timestamp(year=int(season), month=6, day=30)
    return float((asof - pd.Timestamp(birth_date)).days) / 365.25


def project_batters(
    inputs: pd.DataFrame,
    season: int,
    birthdates: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Marcel-project every batter with >=1 PA in the three lag seasons.

    Args:
        inputs: Frame from :func:`load_batter_season_inputs` covering AT
            LEAST seasons ``season-3 .. season-1``.
        season: Target season.
        birthdates: Optional ``[player_id, birth_date]`` frame; when absent
            or missing a player, the age multiplier stays 1.0
            (``age_known=False``).

    Returns:
        DataFrame of :class:`MarcelProjection` fields, one row per batter.
    """
    lags = (season - 1, season - 2, season - 3)
    lg = league_woba_by_season(inputs)
    missing_lg = [y for y in lags if y not in lg]
    if missing_lg:
        raise ValueError(
            f"League wOBA unavailable for lag seasons {missing_lg}; "
            f"inputs cover {sorted(inputs['season'].unique())}."
        )
    bd_map: dict[int, Any] = {}
    if birthdates is not None and len(birthdates):
        bd_map = dict(zip(birthdates["player_id"].astype(int),
                          birthdates["birth_date"]))

    piv_pa = inputs.pivot_table(index="player_id", columns="season",
                                values="pa", aggfunc="sum")
    piv_woba = inputs.pivot_table(index="player_id", columns="season",
                                  values="woba", aggfunc="mean")
    rows = []
    for pid in piv_pa.index:
        pa_by_lag = tuple(
            float(piv_pa.at[pid, y]) if y in piv_pa.columns and pd.notna(piv_pa.at[pid, y]) else 0.0
            for y in lags
        )
        if sum(pa_by_lag) <= 0:
            continue
        woba_by_lag = tuple(
            float(piv_woba.at[pid, y]) if y in piv_woba.columns and pd.notna(piv_woba.at[pid, y]) else None
            for y in lags
        )
        age = age_asof_june30(bd_map.get(int(pid)), season)
        proj = marcel_batter_woba(
            player_id=int(pid), season=season,
            pa_by_lag=pa_by_lag, woba_by_lag=woba_by_lag,
            league_woba_by_lag=tuple(lg[y] for y in lags),
            age=age,
        )
        rows.append(proj.as_dict())
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# WS4.4 -- pre-registered forecast scoring protocol
# ---------------------------------------------------------------------------

TIE_BAND_WOBA: float = 0.010
SUPERIORITY_CONFIDENCE: float = 0.90
MIN_SEASONS_FOR_SUPERIORITY: int = 2


def pa_weighted_rmse(actual: np.ndarray, predicted: np.ndarray,
                     pa: np.ndarray) -> float:
    """PA-weighted RMSE: sqrt( sum(pa * err^2) / sum(pa) )."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    pa = np.asarray(pa, dtype=float)
    mask = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(pa) & (pa > 0)
    if mask.sum() == 0:
        return float("nan")
    err2 = (actual[mask] - predicted[mask]) ** 2
    return float(np.sqrt(np.average(err2, weights=pa[mask])))


def head_to_head_vs_marcel(
    system_pred: np.ndarray,
    marcel_pred: np.ndarray,
    actual: np.ndarray,
    tie_band: float = TIE_BAND_WOBA,
) -> dict[str, int]:
    """Per-player W-L-T on |error| with the pre-registered 0.010 tie band.

    A player is a system WIN when ``|err_system| < |err_marcel| - 0``
    beyond the tie band, i.e. ``|err_marcel| - |err_system| > tie_band``;
    a LOSS when the reverse; a TIE when the |error| gap is within the band.
    """
    s = np.abs(np.asarray(actual, dtype=float) - np.asarray(system_pred, dtype=float))
    m = np.abs(np.asarray(actual, dtype=float) - np.asarray(marcel_pred, dtype=float))
    mask = np.isfinite(s) & np.isfinite(m)
    gap = m[mask] - s[mask]
    return {
        "wins": int((gap > tie_band).sum()),
        "losses": int((gap < -tie_band).sum()),
        "ties": int((np.abs(gap) <= tie_band).sum()),
        "n": int(mask.sum()),
    }


def pa_weighted_paired_t(
    system_pred: np.ndarray,
    marcel_pred: np.ndarray,
    actual: np.ndarray,
    pa: np.ndarray,
) -> dict[str, float]:
    """PA-weighted paired t-test on per-player |error| differences.

    ``d_i = |err_marcel_i| - |err_system_i|`` (positive = system better),
    weighted by follow-up PA.  Effective sample size via Kish
    ``n_eff = (sum w)^2 / sum(w^2)``; the t statistic uses the weighted
    mean and weighted variance of ``d`` with ``n_eff`` degrees of freedom.
    One-sided confidence that the system beats Marcel.
    """
    from scipy import stats

    s = np.abs(np.asarray(actual, dtype=float) - np.asarray(system_pred, dtype=float))
    m = np.abs(np.asarray(actual, dtype=float) - np.asarray(marcel_pred, dtype=float))
    w = np.asarray(pa, dtype=float)
    mask = np.isfinite(s) & np.isfinite(m) & np.isfinite(w) & (w > 0)
    d = (m - s)[mask]
    w = w[mask]
    n = int(mask.sum())
    if n < 3:
        return {"n": n, "n_eff": float("nan"), "mean_diff": float("nan"),
                "t_stat": float("nan"), "confidence_system_better": float("nan")}
    mean_d = float(np.average(d, weights=w))
    var_d = float(np.average((d - mean_d) ** 2, weights=w))
    n_eff = float(w.sum() ** 2 / (w ** 2).sum())
    se = math.sqrt(var_d / n_eff) if var_d > 0 and n_eff > 1 else float("nan")
    if not (se and se > 0):
        return {"n": n, "n_eff": n_eff, "mean_diff": mean_d,
                "t_stat": float("nan"), "confidence_system_better": float("nan")}
    t_stat = mean_d / se
    conf = float(stats.t.cdf(t_stat, df=max(n_eff - 1.0, 1.0)))
    return {"n": n, "n_eff": round(n_eff, 1), "mean_diff": mean_d,
            "t_stat": t_stat, "confidence_system_better": conf}


def score_forecast_vs_protocol(
    *,
    system_name: str,
    actual_woba: np.ndarray,
    system_pred: np.ndarray,
    marcel_pred: np.ndarray,
    naive_pred: np.ndarray,
    followup_pa: np.ndarray,
    seasons_evaluated: int,
) -> dict[str, Any]:
    """The full WS4.4 verdict object (K6-consumable).

    ``superiority_claim_allowed`` is True ONLY when all pre-registered
    conditions hold: >= 2 resolved seasons, head-to-head wins > losses,
    and the PA-weighted paired t confidence >= 90%.  Everything else is
    reported exactly as it lands.
    """
    h2h = head_to_head_vs_marcel(system_pred, marcel_pred, actual_woba)
    t = pa_weighted_paired_t(system_pred, marcel_pred, actual_woba, followup_pa)
    verdict = {
        "system": system_name,
        "protocol": "plan 4.4 (2026-08-10): PA-weighted RMSE vs naive; "
                    "W-L vs Marcel, 0.010 tie band; PA-weighted paired t "
                    ">= 90%; >= 2 seasons before superiority",
        "n_players": int(h2h["n"]),
        "seasons_evaluated": int(seasons_evaluated),
        "rmse_system": pa_weighted_rmse(actual_woba, system_pred, followup_pa),
        "rmse_marcel": pa_weighted_rmse(actual_woba, marcel_pred, followup_pa),
        "rmse_naive": pa_weighted_rmse(actual_woba, naive_pred, followup_pa),
        "head_to_head_vs_marcel": h2h,
        "paired_t": t,
    }
    conf = t.get("confidence_system_better")
    verdict["superiority_claim_allowed"] = bool(
        seasons_evaluated >= MIN_SEASONS_FOR_SUPERIORITY
        and h2h["wins"] > h2h["losses"]
        and conf is not None
        and not (isinstance(conf, float) and math.isnan(conf))
        and conf >= SUPERIORITY_CONFIDENCE
    )
    if not verdict["superiority_claim_allowed"]:
        reasons = []
        if seasons_evaluated < MIN_SEASONS_FOR_SUPERIORITY:
            reasons.append(
                f"only {seasons_evaluated} resolved season(s); protocol "
                f"requires >= {MIN_SEASONS_FOR_SUPERIORITY}"
            )
        if h2h["wins"] <= h2h["losses"]:
            reasons.append("head-to-head does not favor the system")
        if conf is None or (isinstance(conf, float) and math.isnan(conf)) or conf < SUPERIORITY_CONFIDENCE:
            reasons.append("paired-t confidence below 90%")
        verdict["superiority_blocked_because"] = reasons
    return verdict
