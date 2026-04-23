"""
Defensive Pressing Intensity (DPI) -- soccer gegenpressing metrics applied to
baseball defense.

Measures how aggressively and effectively a team's defense converts batted balls
in play (BIP) into outs compared to expectation.  The core idea:

    In soccer, gegenpressing (counter-pressing) quantifies how quickly and
    effectively a team wins the ball back after losing possession.  DPI
    translates this to baseball defense: how often does the defense "win back"
    possession (i.e. record an out) on batted balls in play beyond what the
    batted-ball profile would suggest?

    Metrics:
        1. **Expected Out Rate** -- HistGradientBoosting model trained on
           BIP features (launch_speed, launch_angle, spray_angle, bb_type)
           predicting probability of out.
        2. **Game DPI** -- SUM(actual_out - expected_out) across all BIP in
           a game.  Positive = defence outperformed expectation.
        3. **Season DPI** -- average game DPI across the season.
        4. **Extra-base Prevention** -- fraction of hits limited to singles
           vs expected XBH rate.
        5. **Consistency (coordination proxy)** -- inverse of game-DPI
           variance; low variance = reliable defensive coordination.

    All computed from batted-ball outcome data (no fielder tracking needed).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd

from src.analytics.base import BaseAnalyticsModel
from src.analytics.features import get_latest_season

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "defensive_pressing"
DEFAULT_XOUT_CHECKPOINT = DEFAULT_MODEL_DIR / "xout_v1.pkl"
# v2 adds weather features (wind parallel/perpendicular to spray, temp_f)
# to the xOut feature matrix. v1 is preserved for reproducibility.
DEFAULT_XOUT_V2_CHECKPOINT = DEFAULT_MODEL_DIR / "xout_v2_weather.pkl"

# Events that count as a defensive out on a BIP
OUT_EVENTS: set[str] = {
    "field_out",
    "force_out",
    "grounded_into_double_play",
    "sac_fly",
    "sac_bunt",
    "fielders_choice",
    "double_play",
    "triple_play",
    "fielders_choice_out",
    "sac_fly_double_play",
    "sac_bunt_double_play",
}

# Events that count as a hit
HIT_EVENTS: set[str] = {"single", "double", "triple", "home_run"}

# Extra-base hit events
XBH_EVENTS: set[str] = {"double", "triple", "home_run"}

# Minimum thresholds
MIN_BIP_PER_GAME: int = 5  # realistic minimum (not 50 -- most games have ~30)
MIN_BIP_PER_SEASON: int = 100

# bb_type encoding map
BB_TYPE_MAP: dict[str, int] = {
    "ground_ball": 0,
    "line_drive": 1,
    "fly_ball": 2,
    "popup": 3,
}


# ---------------------------------------------------------------------------
# Stadium orientation lookup (for wind-into-spray-angle projection)
# ---------------------------------------------------------------------------
#
# Compass bearing (degrees clockwise from true north) of the "home-plate to
# center-field" axis for each MLB stadium. Public, well-documented values
# sourced from MLB park orientation references (e.g. Clem's Baseball,
# seamheads.com, aerial imagery).  Precision ~5 degrees is sufficient for a
# meaningful wind projection -- we are not modelling the exact foul-pole
# geometry, only the outward hit bearing.
#
# Convention: e.g. Citi Field CF is at compass bearing ~25 deg (NNE);
# Fenway Park CF is ~46 deg (NE); Wrigley Field CF is ~36 deg (NE).
# Values apply to the home team's current primary park; teams that moved
# parks in the 2015-2025 window (Rangers, Rays, Athletics) are approximated
# to the primary (current) venue -- the vast majority of their games use it.
TEAM_CF_BEARING_DEG: dict[str, float] = {
    "ARI": 25.0,   # Chase Field CF points NNE
    "AZ":  25.0,   # current Statcast abbreviation
    "ATL": 45.0,   # Truist Park (SunTrust pre-2021)
    "BAL": 23.0,   # Camden Yards
    "BOS": 46.0,   # Fenway Park
    "CHC": 36.0,   # Wrigley Field
    "CWS": 42.0,   # Rate/Guaranteed Rate Field
    "CIN": 30.0,   # Great American Ball Park
    "CLE": 5.0,    # Progressive Field (roughly N)
    "COL": 1.0,    # Coors Field (N)
    "DET": 155.0,  # Comerica Park (points SSE)
    "HOU": 20.0,   # Daikin / Minute Maid
    "KC":  47.0,   # Kauffman Stadium
    "LAA": 55.0,   # Angel Stadium
    "LAD": 26.0,   # Dodger Stadium
    "MIA": 48.0,   # loanDepot park / Marlins Park
    "MIL": 130.0,  # American Family Field (SE)
    "MIN": 90.0,   # Target Field (E)
    "NYM": 25.0,   # Citi Field
    "NYY": 75.0,   # Yankee Stadium (ENE)
    "ATH": 55.0,   # Sutter Health Park (Sacramento, approx)
    "OAK": 55.0,   # Oakland Coliseum (legacy abbr; similar bearing)
    "PHI": 34.0,   # Citizens Bank Park
    "PIT": 122.0,  # PNC Park (ESE, toward downtown)
    "SD":  0.0,    # Petco Park (N)
    "SF":  90.0,   # Oracle Park (E, toward bay)
    "SEA": 45.0,   # T-Mobile Park
    "STL": 30.0,   # Busch Stadium
    "TB":  46.0,   # Tropicana dome + Steinbrenner Field 2025 fallback
    "TEX": 15.0,   # Globe Life Field / Globe Life Park
    "TOR": 0.0,    # Rogers Centre (N)
    "WSH": 30.0,   # Nationals Park
}

# roof statuses where wind effects are meaningfully suppressed.
# Per task brief: dome and retractable_unknown -> wind components = 0.
# This is a stronger claim than NaN: the model sees "no wind effect"
# explicitly rather than NaN (which HistGB routes via a learned split).
WIND_SUPPRESSED_ROOFS: set[str] = {"dome", "closed", "retractable_unknown"}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DPIConfig:
    """Hyperparameters for the DPI model."""

    min_bip_per_game: int = MIN_BIP_PER_GAME
    min_bip_per_season: int = MIN_BIP_PER_SEASON
    xout_n_estimators: int = 200
    xout_max_depth: int = 6
    xout_learning_rate: float = 0.05
    random_state: int = 42


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_spray_angle(hc_x: pd.Series, hc_y: pd.Series) -> pd.Series:
    """Compute spray angle in degrees from hit coordinates.

    The Statcast coordinate system places home plate at approximately
    (125, 200).  Spray angle 0 = straight up the middle, negative = pull
    side (for RHB), positive = opposite field.

    Args:
        hc_x: Hit coordinate X.
        hc_y: Hit coordinate Y.

    Returns:
        Spray angle in degrees.
    """
    # Statcast home-plate reference point
    HP_X = 125.42
    HP_Y = 198.27

    dx = hc_x - HP_X
    dy = HP_Y - hc_y  # Y increases downward in Statcast, flip for math
    return np.degrees(np.arctan2(dx, dy))


def _encode_bb_type(bb_type_series: pd.Series) -> pd.Series:
    """Map bb_type strings to integer codes."""
    return bb_type_series.map(BB_TYPE_MAP).fillna(-1).astype(int)


def compute_wind_components(
    wind_speed_mph: pd.Series,
    wind_dir_deg: pd.Series,
    spray_angle_deg: pd.Series,
    home_team: pd.Series,
    roof_status: pd.Series | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Decompose wind into parallel/perpendicular components relative to the
    batted ball's outward bearing (Nathan/Tango framing).

    Physics:
      1. Convert meteorological wind direction (degrees wind is **from**,
         compass clockwise from N) to the direction the wind is flowing
         **toward**: ``wind_to_deg = (wind_dir_deg + 180) mod 360``.
      2. Compute the ball's compass bearing from home plate:
         ``ball_bearing = (team_cf_bearing + spray_angle_deg) mod 360`` where
         positive spray_angle is toward right field in Statcast convention
         (matches :func:`compute_spray_angle`).
      3. Parallel component (scalar projection):
         ``wind_parallel = wind_speed * cos(ball_bearing - wind_to_deg)``.
         Positive values mean the wind is flowing in the same compass
         direction as the ball is traveling -- a tailwind for that hit.
      4. Perpendicular component (magnitude):
         ``wind_perpendicular = wind_speed * |sin(ball_bearing - wind_to_deg)|``.
         A pure crosswind magnitude; sign is not physically meaningful for
         the HistGB feature since the model cannot use crosswind direction.

    Domes / retractable-unknown roof statuses are **zeroed** in both
    components (strictly stronger than NaN: the feature tree sees "no wind
    effect" as a real observed value rather than a missing-value split).

    Missing team (no entry in ``TEAM_CF_BEARING_DEG``) or missing wind
    inputs also zero the components -- safer than emitting NaN into the
    feature matrix.

    Args:
        wind_speed_mph: Per-BIP wind speed in mph (NaN-safe).
        wind_dir_deg: Meteorological wind direction (degrees wind is FROM).
        spray_angle_deg: Per-BIP Statcast spray angle, degrees from CF.
        home_team: Per-BIP home-team abbreviation (venue proxy).
        roof_status: Per-BIP roof status string (optional).

    Returns:
        Tuple of (wind_parallel_to_spray, wind_perpendicular_to_spray) as
        float Series in mph, indexed like the inputs.
    """
    idx = wind_speed_mph.index

    speed = pd.to_numeric(wind_speed_mph, errors="coerce").astype(float)
    wdir_from = pd.to_numeric(wind_dir_deg, errors="coerce").astype(float)
    spray = pd.to_numeric(spray_angle_deg, errors="coerce").astype(float)

    # Stadium CF bearing per row
    cf_bearing = home_team.astype(str).map(TEAM_CF_BEARING_DEG).astype(float)

    # Direction the wind is blowing TOWARD (compass, degrees)
    wind_to = (wdir_from + 180.0) % 360.0
    # Ball's compass bearing = CF bearing + spray (positive spray -> RF)
    ball_bearing = (cf_bearing + spray) % 360.0

    # Relative angle between wind flow and ball flight
    delta_deg = (ball_bearing - wind_to + 540.0) % 360.0 - 180.0
    delta_rad = np.radians(delta_deg)

    parallel = speed * np.cos(delta_rad)
    perpendicular = np.abs(speed * np.sin(delta_rad))

    # Zero wind effect for domes / retractable_unknown / closed and for
    # any row with missing cf_bearing / speed / direction.
    suppress_mask = pd.Series(False, index=idx)
    if roof_status is not None:
        suppress_mask = (
            roof_status.astype(str).isin(WIND_SUPPRESSED_ROOFS).fillna(False)
        )
    missing_mask = (
        speed.isna() | wdir_from.isna() | spray.isna() | cf_bearing.isna()
    )
    zero_mask = suppress_mask | missing_mask

    parallel = parallel.where(~zero_mask, 0.0).fillna(0.0)
    perpendicular = perpendicular.where(~zero_mask, 0.0).fillna(0.0)

    return (
        pd.Series(parallel.values, index=idx, name="wind_parallel_to_spray"),
        pd.Series(perpendicular.values, index=idx, name="wind_perpendicular_to_spray"),
    )


_park_target_encoding: dict[str, float] | None = None
_park_global_mean: float | None = None


def fit_park_target_encoding(
    df: pd.DataFrame,
    target: pd.Series,
    *,
    smoothing: float = 100.0,
) -> tuple[dict[str, float], float]:
    """Fit a per-park (proxy: ``home_team``) target encoding for xOut.

    Uses smoothed empirical Bayes:
        encoded(park) = (n_park * out_rate_park + smoothing * global_mean)
                        / (n_park + smoothing)

    Smoothing keeps low-sample parks (e.g. neutral-site games) from
    swinging far from the league mean. Returns the mapping plus the
    fallback global mean used for unseen parks at test time.
    """
    if "home_team" not in df.columns:
        return {}, float(target.mean()) if len(target) else 0.667
    mask = df["home_team"].notna() & target.notna()
    sub = df.loc[mask]
    tgt = target.loc[mask].astype(float)
    if len(sub) == 0:
        return {}, float(target.mean()) if len(target) else 0.667
    global_mean = float(tgt.mean())
    grouped = pd.DataFrame({
        "home_team": sub["home_team"].astype(str),
        "y": tgt.values,
    }).groupby("home_team")["y"]
    n = grouped.count()
    mu = grouped.mean()
    encoded = (n * mu + smoothing * global_mean) / (n + smoothing)
    return {str(k): float(v) for k, v in encoded.items()}, global_mean


def set_park_target_encoding(
    encoding: dict[str, float] | None, global_mean: float | None,
) -> None:
    """Store the (encoding, global_mean) tuple at module scope."""
    global _park_target_encoding, _park_global_mean
    _park_target_encoding = encoding
    _park_global_mean = global_mean


def build_bip_features(
    df: pd.DataFrame,
    *,
    include_park: bool | None = None,
    include_weather: bool = False,
) -> pd.DataFrame:
    """Build feature matrix from a BIP DataFrame.

    Args:
        df: DataFrame with launch_speed, launch_angle, hc_x, hc_y, bb_type
            (and optionally ``home_team`` if the park feature is enabled,
            and wind_speed_mph / wind_dir_deg / temp_f / roof_status columns
            if ``include_weather=True``).
        include_park: Whether to attach the per-park target-encoded feature.
            Default behaviour: include the feature iff a target encoding
            has been registered via :func:`set_park_target_encoding` (or
            fitted via :func:`fit_xout` with ``use_park=True``).
        include_weather: Whether to derive wind_parallel_to_spray,
            wind_perpendicular_to_spray, and temp_f features from the
            joined game_weather columns. Missing columns -> zeros.

    Returns:
        DataFrame with columns: launch_speed, launch_angle, spray_angle,
        bb_type_encoded, optionally park_xout_te, and optionally
        wind_parallel_to_spray, wind_perpendicular_to_spray, temp_f.
    """
    features = pd.DataFrame(index=df.index)
    features["launch_speed"] = df["launch_speed"].astype(float)
    features["launch_angle"] = df["launch_angle"].astype(float)
    features["spray_angle"] = compute_spray_angle(
        df["hc_x"].astype(float),
        df["hc_y"].astype(float),
    )
    features["bb_type_encoded"] = _encode_bb_type(df["bb_type"])

    if include_park is None:
        include_park = _park_target_encoding is not None

    if include_park and _park_target_encoding is not None:
        gm = _park_global_mean if _park_global_mean is not None else 0.667
        if "home_team" in df.columns:
            ht = df["home_team"].astype(str)
            features["park_xout_te"] = (
                ht.map(_park_target_encoding).fillna(gm).astype(float)
            )
        else:
            features["park_xout_te"] = float(gm)

    if include_weather:
        # Wind components: require wind_speed_mph, wind_dir_deg, and
        # home_team; roof_status (if present) zeroes the components for
        # dome/retractable-unknown venues. Missing columns fill with NaN,
        # which compute_wind_components treats as "zero wind effect".
        n = len(df)
        wind_speed = (
            pd.to_numeric(df["wind_speed_mph"], errors="coerce").astype(float)
            if "wind_speed_mph" in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )
        wind_dir = (
            pd.to_numeric(df["wind_dir_deg"], errors="coerce").astype(float)
            if "wind_dir_deg" in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )
        home_team = (
            df["home_team"].astype(str)
            if "home_team" in df.columns
            else pd.Series([""] * n, index=df.index, dtype=object)
        )
        roof = (
            df["roof_status"] if "roof_status" in df.columns else None
        )
        wp, wperp = compute_wind_components(
            wind_speed_mph=wind_speed,
            wind_dir_deg=wind_dir,
            spray_angle_deg=features["spray_angle"],
            home_team=home_team,
            roof_status=roof,
        )
        features["wind_parallel_to_spray"] = wp.astype(float)
        features["wind_perpendicular_to_spray"] = wperp.astype(float)

        # temp_f: HistGB handles NaN natively; leave NaN where missing
        # (e.g. fixed domes that never attached an outdoor reading).
        if "temp_f" in df.columns:
            features["temp_f"] = pd.to_numeric(
                df["temp_f"], errors="coerce"
            ).astype(float)
        else:
            features["temp_f"] = np.nan

    return features


def _is_out(events: pd.Series) -> pd.Series:
    """Return boolean Series: True if the event is a defensive out."""
    return events.isin(OUT_EVENTS).astype(int)


def _is_hit(events: pd.Series) -> pd.Series:
    """Return boolean Series: True if the event is a hit."""
    return events.isin(HIT_EVENTS).astype(int)


def _is_xbh(events: pd.Series) -> pd.Series:
    """Return boolean Series: True if the event is an extra-base hit."""
    return events.isin(XBH_EVENTS).astype(int)


# ---------------------------------------------------------------------------
# Expected Out model
# ---------------------------------------------------------------------------

_xout_model = None  # module-level cache
_xout_metadata: dict | None = None  # description of the loaded/fit model


def _xout_max_train_season(model_meta: dict | None) -> int | None:
    """Return the most-recent train season recorded in model metadata."""
    if not model_meta:
        return None
    seasons = model_meta.get("train_seasons") or []
    if not seasons:
        return None
    try:
        return int(max(seasons))
    except (TypeError, ValueError):
        return None


def _latest_pitches_season(conn: duckdb.DuckDBPyConnection) -> int | None:
    """Most-recent season present in the pitches table (None on empty)."""
    try:
        row = conn.execute(
            "SELECT MAX(EXTRACT(YEAR FROM game_date))::INT FROM pitches"
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])
    except Exception:
        return None


def load_xout_checkpoint(
    path: Path | None = None,
) -> tuple[Any, dict | None]:
    """Load a persisted xOut checkpoint (model, metadata) or (None, None)."""
    global _xout_model, _xout_metadata
    chk = Path(path) if path is not None else DEFAULT_XOUT_CHECKPOINT
    if not chk.exists():
        return None, None
    try:
        import joblib
        bundle = joblib.load(chk)
    except Exception as exc:
        logger.warning("Could not load xOut checkpoint at %s: %s", chk, exc)
        return None, None
    if isinstance(bundle, dict) and "model" in bundle:
        _xout_model = bundle["model"]
        _xout_metadata = bundle.get("metadata", {})
    else:
        _xout_model = bundle
        _xout_metadata = {}
    # Restore the park target-encoding into module state if present
    if (_xout_metadata or {}).get("use_park") and _xout_metadata.get("park_encoding"):
        set_park_target_encoding(
            _xout_metadata.get("park_encoding"),
            _xout_metadata.get("park_global_mean"),
        )
    else:
        set_park_target_encoding(None, None)
    logger.info(
        "Loaded persisted xOut model from %s (train_seasons=%s, use_park=%s)",
        chk,
        (_xout_metadata or {}).get("train_seasons"),
        (_xout_metadata or {}).get("use_park", False),
    )
    return _xout_model, _xout_metadata


def fit_xout(
    conn: duckdb.DuckDBPyConnection,
    seasons: list[int] | None = None,
    config: DPIConfig | None = None,
    persist_path: Path | None = None,
    *,
    use_park: bool = False,
    park_smoothing: float = 100.0,
    use_weather: bool = False,
) -> dict:
    """Fit the xOut classifier and optionally pickle it via joblib.

    A thin convenience wrapper around :func:`train_expected_out_model` that
    also writes the trained model + training metadata to ``persist_path``
    (default: ``DEFAULT_XOUT_CHECKPOINT``) so future calls can re-use it.

    Args:
        conn: Open DuckDB connection.
        seasons: Train seasons (default: all available).
        config: DPI configuration.
        persist_path: Where to pickle the model. ``None`` skips persistence.
        use_park: Whether to add the per-park target-encoded feature.
        park_smoothing: Smoothing for the park target encoding.
        use_weather: Whether to attach wind / temperature BIP features
            (requires ``game_weather`` table). Defaults to False.

    Returns:
        The dict returned by :func:`train_expected_out_model`, augmented
        with ``persist_path`` and ``train_seasons`` for reproducibility.
    """
    global _xout_metadata
    metrics = train_expected_out_model(
        conn,
        seasons=seasons,
        config=config,
        use_park=use_park,
        park_smoothing=park_smoothing,
        use_weather=use_weather,
    )
    metrics = dict(metrics)
    metrics["train_seasons"] = list(seasons) if seasons else None

    if persist_path is None:
        persist_path = DEFAULT_XOUT_CHECKPOINT
    persist_path = Path(persist_path)

    feature_columns = metrics.get("feature_columns") or [
        "launch_speed", "launch_angle", "spray_angle", "bb_type_encoded",
    ]
    _xout_metadata = {
        "train_seasons": metrics.get("train_seasons"),
        "n_samples": metrics.get("n_samples"),
        "auc": metrics.get("auc"),
        "train_auc": metrics.get("train_auc"),
        "out_rate": metrics.get("out_rate"),
        "fitted_at": datetime.now(timezone.utc).isoformat(),
        "feature_columns": list(feature_columns),
        "use_park": bool(use_park),
        "use_weather": bool(use_weather),
        "park_encoding": metrics.get("park_encoding"),
        "park_global_mean": metrics.get("park_global_mean"),
        "park_smoothing": metrics.get("park_smoothing"),
        "config": {
            "xout_n_estimators": (config or DPIConfig()).xout_n_estimators,
            "xout_max_depth": (config or DPIConfig()).xout_max_depth,
            "xout_learning_rate": (config or DPIConfig()).xout_learning_rate,
            "random_state": (config or DPIConfig()).random_state,
        },
    }

    if _xout_model is not None and persist_path is not None:
        try:
            import joblib
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {"model": _xout_model, "metadata": _xout_metadata},
                persist_path,
            )
            metrics["persist_path"] = str(persist_path)
            logger.info("Persisted xOut model to %s", persist_path)
        except Exception as exc:
            logger.warning("Could not persist xOut model: %s", exc)
            metrics["persist_path"] = None
    return metrics


def ensure_xout_model(
    conn: duckdb.DuckDBPyConnection,
    config: DPIConfig | None = None,
    persist_path: Path | None = None,
    force_refit: bool = False,
) -> dict:
    """Load an xOut checkpoint if available; else fit and persist a new one.

    Cache invalidation policy: if the loaded checkpoint's most-recent
    ``train_seasons`` entry is older than the most-recent season present
    in ``pitches``, refit on all seasons strictly before the current
    latest season (to keep the train cohort the same shape: trailing
    seasons up to but not including any in-progress year).

    Returns metadata describing what was done; sets module ``_xout_model``.
    """
    global _xout_model, _xout_metadata
    persist_path = Path(persist_path) if persist_path else DEFAULT_XOUT_CHECKPOINT

    # Try load first
    if not force_refit and _xout_model is None:
        load_xout_checkpoint(persist_path)

    latest_db_season = _latest_pitches_season(conn)
    chk_max_season = _xout_max_train_season(_xout_metadata)

    needs_refit = False
    refit_reason = None
    if _xout_model is None:
        needs_refit = True
        refit_reason = "no_checkpoint"
    elif force_refit:
        needs_refit = True
        refit_reason = "force_refit"
    elif (
        latest_db_season is not None
        and chk_max_season is not None
        and chk_max_season < latest_db_season - 1
    ):
        # checkpoint is more than one season behind latest pitches data
        needs_refit = True
        refit_reason = (
            f"checkpoint_stale chk_max={chk_max_season} db_max={latest_db_season}"
        )

    if needs_refit:
        logger.info("Re-fitting xOut model (%s)", refit_reason)
        # Default cohort: 2015 through the season immediately preceding the
        # latest available, to mirror the validation spec's hold-out style.
        if latest_db_season is not None:
            seasons = list(range(2015, latest_db_season))
        else:
            seasons = None
        metrics = fit_xout(
            conn, seasons=seasons, config=config, persist_path=persist_path,
        )
        metrics["loaded_from_checkpoint"] = False
        metrics["refit_reason"] = refit_reason
        return metrics

    # warm path
    return {
        "status": "loaded_checkpoint",
        "loaded_from_checkpoint": True,
        "persist_path": str(persist_path),
        "train_seasons": (_xout_metadata or {}).get("train_seasons"),
        "auc": (_xout_metadata or {}).get("auc"),
        "n_samples": (_xout_metadata or {}).get("n_samples"),
    }


def train_expected_out_model(
    conn: duckdb.DuckDBPyConnection,
    seasons: list[int] | None = None,
    config: DPIConfig | None = None,
    *,
    use_park: bool = False,
    park_smoothing: float = 100.0,
    use_weather: bool = False,
) -> dict:
    """Train a HistGradientBoosting classifier for expected out probability.

    Uses BIP data from the pitches table where type='X' and batted-ball
    features are available.

    Args:
        conn: Open DuckDB connection.
        seasons: List of seasons to train on.  If None, uses all available.
        config: DPI configuration.
        use_park: If True, fit a per-park (proxy: ``home_team``) target
            encoding on the TRAIN split only and add it as a continuous
            feature. Defaults to False (v1 baseline behaviour).
        park_smoothing: Smoothing constant for the empirical-Bayes target
            encoding (only used when ``use_park=True``).
        use_weather: If True, LEFT JOIN ``game_weather`` on ``game_pk`` and
            attach ``wind_parallel_to_spray``, ``wind_perpendicular_to_spray``,
            and ``temp_f`` to the BIP feature matrix.  Dome / retractable-
            unknown venues set wind components to 0.  Defaults to False.

    Returns:
        Dictionary of training metrics (AUC, accuracy, n_samples).
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score

    global _xout_model
    config = config or DPIConfig()

    season_filter = ""
    if seasons:
        placeholders = ", ".join(str(int(s)) for s in seasons)
        season_filter = f"AND EXTRACT(YEAR FROM game_date) IN ({placeholders})"

    # home_team is needed for use_park AND for use_weather (spray-projection
    # requires the stadium CF bearing).
    need_home_team = use_park or use_weather
    park_select = ", p.home_team" if need_home_team else ""

    if use_weather:
        weather_select = (
            ", gw.wind_speed_mph, gw.wind_dir_deg, gw.temp_f, gw.roof_status"
        )
        weather_join = (
            "LEFT JOIN game_weather gw ON p.game_pk = gw.game_pk"
        )
    else:
        weather_select = ""
        weather_join = ""

    query = f"""
        SELECT
            p.launch_speed, p.launch_angle, p.hc_x, p.hc_y,
            p.bb_type, p.events{park_select}{weather_select}
        FROM pitches p
        {weather_join}
        WHERE p.type = 'X'
          AND p.launch_speed IS NOT NULL
          AND p.launch_angle IS NOT NULL
          AND p.hc_x IS NOT NULL
          AND p.hc_y IS NOT NULL
          AND p.bb_type IS NOT NULL
          AND p.events IS NOT NULL
          {season_filter.replace('game_date', 'p.game_date')}
    """
    df = conn.execute(query).fetchdf()

    if df.empty or len(df) < 100:
        logger.warning("Not enough BIP data for training: %d rows", len(df))
        _xout_model = None
        return {"status": "insufficient_data", "n_samples": len(df)}

    target = _is_out(df["events"])

    # Build features WITHOUT park first (target encoding fitted on train split)
    set_park_target_encoding(None, None)
    features = build_bip_features(
        df, include_park=False, include_weather=use_weather,
    )

    # Drop rows with NaN in core features (and home_team if park is enabled).
    # Weather features are DESIGNED to be NaN-safe: wind components are
    # zero-filled (not NaN) in compute_wind_components, and temp_f NaN is
    # tolerated natively by HistGB.  So we exclude weather columns from the
    # "require finite" mask.
    core_cols = [
        c for c in features.columns
        if c in {
            "launch_speed", "launch_angle", "spray_angle", "bb_type_encoded",
        }
    ]
    base_mask = features[core_cols].notna().all(axis=1)
    if use_park:
        base_mask = base_mask & df["home_team"].notna()
    features = features[base_mask].reset_index(drop=True)
    target = target[base_mask].reset_index(drop=True)
    if use_park:
        df_aligned = df.loc[base_mask].reset_index(drop=True)

    if len(features) < 100:
        _xout_model = None
        return {"status": "insufficient_data_after_clean", "n_samples": len(features)}

    # Stratified split on the binary target
    train_idx, test_idx = train_test_split(
        np.arange(len(features)),
        test_size=0.2,
        random_state=config.random_state,
        stratify=target,
    )
    X_train = features.iloc[train_idx].copy()
    X_test = features.iloc[test_idx].copy()
    y_train = target.iloc[train_idx]
    y_test = target.iloc[test_idx]

    park_encoding: dict[str, float] | None = None
    park_global_mean: float | None = None
    if use_park:
        park_encoding, park_global_mean = fit_park_target_encoding(
            df_aligned.iloc[train_idx],
            y_train,
            smoothing=park_smoothing,
        )
        # Apply to BOTH splits without leaking test info
        X_train["park_xout_te"] = (
            df_aligned.iloc[train_idx]["home_team"].astype(str)
            .map(park_encoding).fillna(park_global_mean).astype(float).values
        )
        X_test["park_xout_te"] = (
            df_aligned.iloc[test_idx]["home_team"].astype(str)
            .map(park_encoding).fillna(park_global_mean).astype(float).values
        )

    model = HistGradientBoostingClassifier(
        max_iter=config.xout_n_estimators,
        max_depth=config.xout_max_depth,
        learning_rate=config.xout_learning_rate,
        random_state=config.random_state,
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = float(roc_auc_score(y_test, y_pred_proba))
    accuracy = float(accuracy_score(y_test, y_pred))

    # Train-AUC (in-sample, for diagnostics)
    train_auc = float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))

    _xout_model = model
    if use_park and park_encoding is not None:
        set_park_target_encoding(park_encoding, park_global_mean)

    metrics = {
        "status": "trained",
        "n_samples": len(features),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "auc": round(auc, 4),
        "train_auc": round(train_auc, 4),
        "accuracy": round(accuracy, 4),
        "out_rate": round(float(target.mean()), 4),
        "use_park": bool(use_park),
        "use_weather": bool(use_weather),
        "park_encoding": park_encoding,
        "park_global_mean": park_global_mean,
        "park_smoothing": park_smoothing if use_park else None,
        "feature_columns": list(X_train.columns),
    }
    logger.info(
        "xOut model trained: AUC=%.4f, accuracy=%.4f on %d BIP "
        "(use_park=%s, use_weather=%s)",
        auc, accuracy, len(features), use_park, use_weather,
    )
    return metrics


def compute_expected_outs(bip_df: pd.DataFrame) -> pd.Series:
    """Compute per-BIP expected out probability using the trained xOut model.

    Args:
        bip_df: DataFrame of batted balls in play with launch_speed,
                launch_angle, hc_x, hc_y, bb_type columns.  Weather columns
                (wind_speed_mph, wind_dir_deg, temp_f, roof_status,
                home_team) are required when the loaded model was trained
                with ``use_weather=True`` -- missing columns will zero the
                wind features and leave temp_f as NaN (both are handled
                gracefully by the downstream HistGB).

    Returns:
        Series of expected out probabilities (float in [0, 1]).
        Returns NaN if the model is not trained or features are missing.
    """
    global _xout_model

    if _xout_model is None:
        logger.warning("xOut model not trained; returning NaN")
        return pd.Series(np.nan, index=bip_df.index)

    expected_cols = (
        list((_xout_metadata or {}).get("feature_columns") or [])
        if _xout_metadata else []
    )
    use_weather_for_predict = bool(
        (_xout_metadata or {}).get("use_weather", False)
    ) or any(
        c in expected_cols for c in (
            "wind_parallel_to_spray",
            "wind_perpendicular_to_spray",
            "temp_f",
        )
    )

    # Build features using whatever the loaded model expects; the park
    # branch silently falls back to the global mean for rows without
    # ``home_team`` (covers older callers that did not select that column).
    features = build_bip_features(
        bip_df, include_weather=use_weather_for_predict,
    )

    # If the trained model expects park feature but our features lack it
    # (older calling code), backfill with global mean.
    if "park_xout_te" in expected_cols and "park_xout_te" not in features.columns:
        gm = _park_global_mean if _park_global_mean is not None else 0.667
        features["park_xout_te"] = float(gm)

    # Ensure expected weather columns are present (zero-fill wind, NaN temp)
    for wcol in (
        "wind_parallel_to_spray", "wind_perpendicular_to_spray",
    ):
        if wcol in expected_cols and wcol not in features.columns:
            features[wcol] = 0.0
    if "temp_f" in expected_cols and "temp_f" not in features.columns:
        features["temp_f"] = np.nan

    # Reorder to match the trained feature columns exactly
    if expected_cols and all(c in features.columns for c in expected_cols):
        features = features[expected_cols]

    # Determine the "finite required" columns: core BIP features must be
    # finite; weather features are already zero- or NaN-tolerated by the
    # downstream HistGB, so we don't drop rows on them.
    required_finite = [
        c for c in features.columns
        if c in {
            "launch_speed", "launch_angle", "spray_angle",
            "bb_type_encoded", "park_xout_te",
        }
    ]
    valid_mask = (
        features[required_finite].notna().all(axis=1)
        if required_finite
        else features.notna().all(axis=1)
    )
    result = pd.Series(np.nan, index=bip_df.index)

    if valid_mask.sum() == 0:
        return result

    valid_features = features[valid_mask]
    probs = _xout_model.predict_proba(valid_features)[:, 1]
    result.loc[valid_mask] = probs
    return result


# ---------------------------------------------------------------------------
# Game-level DPI
# ---------------------------------------------------------------------------

def calculate_game_dpi(
    conn: duckdb.DuckDBPyConnection,
    game_pk: int,
    team_id: str,
    config: DPIConfig | None = None,
) -> dict:
    """Calculate Defensive Pressing Intensity for one team in one game.

    DPI = sum(actual_out - expected_out) across all BIP against the team's
    pitching staff (i.e. balls hit by the opposing team's batters).

    Args:
        conn: Open DuckDB connection.
        game_pk: Game identifier.
        team_id: Three-letter team abbreviation (defensive team).
        config: DPI configuration.

    Returns:
        Dictionary with game_pk, team_id, dpi, n_bip, actual_outs,
        expected_outs, xbh_rate, xbh_expected, extra_base_prevention.
    """
    config = config or DPIConfig()

    # BIP against this team's defense = opposing batters hitting
    # If team is home, opposing batters bat in the top half (inning_topbot='Top')
    # If team is away, opposing batters bat in the bottom half (inning_topbot='Bot')
    # game_weather is LEFT-joined so weather-enabled xOut models have the
    # columns they need; non-weather models silently ignore them.  If the
    # game_weather table does not exist (e.g. in unit-test fixtures) we
    # fall back to the core-only query.
    weather_query = """
        SELECT
            p.launch_speed, p.launch_angle, p.hc_x, p.hc_y,
            p.bb_type, p.events, p.home_team,
            gw.wind_speed_mph, gw.wind_dir_deg, gw.temp_f, gw.roof_status
        FROM pitches p
        LEFT JOIN game_weather gw ON p.game_pk = gw.game_pk
        WHERE p.game_pk = $1
          AND p.type = 'X'
          AND p.events IS NOT NULL
          AND (
              (p.home_team = $2 AND p.inning_topbot = 'Top')
              OR (p.away_team = $2 AND p.inning_topbot = 'Bot')
          )
    """
    fallback_query = """
        SELECT
            launch_speed, launch_angle, hc_x, hc_y, bb_type, events, home_team
        FROM pitches
        WHERE game_pk = $1
          AND type = 'X'
          AND events IS NOT NULL
          AND (
              (home_team = $2 AND inning_topbot = 'Top')
              OR (away_team = $2 AND inning_topbot = 'Bot')
          )
    """
    try:
        bip_df = conn.execute(weather_query, [game_pk, team_id]).fetchdf()
    except (duckdb.CatalogException, duckdb.BinderException):
        bip_df = conn.execute(fallback_query, [game_pk, team_id]).fetchdf()

    n_bip = len(bip_df)
    result = {
        "game_pk": game_pk,
        "team_id": team_id,
        "dpi": None,
        "n_bip": n_bip,
        "actual_outs": 0,
        "expected_outs": 0.0,
        "xbh_count": 0,
        "hit_count": 0,
        "extra_base_prevention": None,
    }

    if n_bip < config.min_bip_per_game:
        return result

    actual_outs = int(_is_out(bip_df["events"]).sum())

    # Filter to rows with batted-ball data for xOut calculation
    has_data = (
        bip_df["launch_speed"].notna()
        & bip_df["launch_angle"].notna()
        & bip_df["hc_x"].notna()
        & bip_df["hc_y"].notna()
        & bip_df["bb_type"].notna()
    )
    bip_with_data = bip_df[has_data]

    if len(bip_with_data) == 0:
        result["actual_outs"] = actual_outs
        result["dpi"] = 0.0
        return result

    xout_probs = compute_expected_outs(bip_with_data)
    expected_outs = float(xout_probs.sum())

    # DPI = actual outs made minus expected outs
    dpi = actual_outs - expected_outs

    # Extra-base prevention
    hit_mask = _is_hit(bip_df["events"])
    hit_count = int(hit_mask.sum())
    xbh_count = int(_is_xbh(bip_df["events"]).sum())
    xbh_rate = xbh_count / hit_count if hit_count > 0 else 0.0

    result.update({
        "dpi": round(dpi, 3),
        "actual_outs": actual_outs,
        "expected_outs": round(expected_outs, 3),
        "xbh_count": xbh_count,
        "hit_count": hit_count,
        "extra_base_prevention": round(1.0 - xbh_rate, 3) if hit_count > 0 else None,
    })
    return result


# ---------------------------------------------------------------------------
# Season-level DPI
# ---------------------------------------------------------------------------

def calculate_team_dpi(
    conn: duckdb.DuckDBPyConnection,
    team_id: str,
    season: int,
    config: DPIConfig | None = None,
) -> dict:
    """Calculate season-level DPI profile for a team.

    Args:
        conn: Open DuckDB connection.
        team_id: Three-letter team abbreviation.
        season: Season year.
        config: DPI configuration.

    Returns:
        Dictionary with team_id, season, dpi_mean, dpi_total, consistency,
        extra_base_prevention, n_games, game_dpis (list).
    """
    config = config or DPIConfig()

    # Get all games this team played in this season
    query = """
        SELECT DISTINCT game_pk
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) = $1
          AND (home_team = $2 OR away_team = $2)
    """
    games = conn.execute(query, [season, team_id]).fetchdf()

    if games.empty:
        return {
            "team_id": team_id,
            "season": season,
            "dpi_mean": None,
            "dpi_total": None,
            "consistency": None,
            "extra_base_prevention": None,
            "n_games": 0,
            "game_dpis": [],
        }

    game_results = []
    for game_pk in games["game_pk"].tolist():
        game_dpi = calculate_game_dpi(conn, game_pk, team_id, config)
        if game_dpi["dpi"] is not None:
            game_results.append(game_dpi)

    if not game_results:
        return {
            "team_id": team_id,
            "season": season,
            "dpi_mean": None,
            "dpi_total": None,
            "consistency": None,
            "extra_base_prevention": None,
            "n_games": 0,
            "game_dpis": [],
        }

    dpis = [g["dpi"] for g in game_results]
    ebp_values = [g["extra_base_prevention"] for g in game_results
                  if g["extra_base_prevention"] is not None]

    dpi_mean = float(np.mean(dpis))
    dpi_total = float(np.sum(dpis))
    dpi_std = float(np.std(dpis)) if len(dpis) > 1 else 0.0

    # Consistency: inverse of coefficient of variation (higher = more consistent)
    # Use 1 / (1 + std) to keep it bounded and positive
    consistency = 1.0 / (1.0 + dpi_std)

    ebp_mean = float(np.mean(ebp_values)) if ebp_values else None

    return {
        "team_id": team_id,
        "season": season,
        "dpi_mean": round(dpi_mean, 3),
        "dpi_total": round(dpi_total, 3),
        "dpi_std": round(dpi_std, 3),
        "consistency": round(consistency, 4),
        "extra_base_prevention": round(ebp_mean, 3) if ebp_mean is not None else None,
        "n_games": len(game_results),
        "game_dpis": game_results,
    }


# ---------------------------------------------------------------------------
# Batch (all teams in a season)
# ---------------------------------------------------------------------------

def batch_calculate(
    conn: duckdb.DuckDBPyConnection,
    season: int,
    config: DPIConfig | None = None,
) -> pd.DataFrame:
    """Calculate DPI for all teams in a season and return a ranked DataFrame.

    Args:
        conn: Open DuckDB connection.
        season: Season year.
        config: DPI configuration.

    Returns:
        DataFrame with columns: team_id, season, dpi_mean, dpi_total,
        consistency, extra_base_prevention, n_games, rank, percentile.
    """
    config = config or DPIConfig()

    # Ensure the xOut model is available (load checkpoint if present;
    # fit-and-persist if not). Cache invalidation policy lives in
    # :func:`ensure_xout_model`.
    if _xout_model is None:
        logger.info(
            "xOut model not in memory; attempting checkpoint load or refit",
        )
        ensure_xout_model(conn, config=config)

    # Get all teams in the season
    query = """
        SELECT DISTINCT home_team AS team_id FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) = $1
        UNION
        SELECT DISTINCT away_team AS team_id FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) = $1
    """
    teams = conn.execute(query, [season]).fetchdf()

    if teams.empty:
        return pd.DataFrame()

    rows = []
    for team_id in teams["team_id"].tolist():
        profile = calculate_team_dpi(conn, team_id, season, config)
        if profile["dpi_mean"] is not None:
            rows.append({
                "team_id": profile["team_id"],
                "season": profile["season"],
                "dpi_mean": profile["dpi_mean"],
                "dpi_total": profile["dpi_total"],
                "dpi_std": profile["dpi_std"],
                "consistency": profile["consistency"],
                "extra_base_prevention": profile["extra_base_prevention"],
                "n_games": profile["n_games"],
            })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("dpi_mean", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    df["percentile"] = np.round(
        100 * (1 - (df["rank"] - 1) / max(len(df), 1)),
        1,
    )
    return df


# ---------------------------------------------------------------------------
# Player-level DPI proxy
# ---------------------------------------------------------------------------

def get_player_dpi(
    conn: duckdb.DuckDBPyConnection,
    player_id: int,
    season: int,
    config: DPIConfig | None = None,
) -> dict:
    """Compute a proxy per-fielder DPI.

    Since we lack fielder tracking data, we approximate by looking at BIP
    outcomes when this player is the pitcher on the mound (for pitchers)
    or by looking at all BIP in games where the player's team is fielding.

    Args:
        conn: Open DuckDB connection.
        player_id: MLB player ID (typically a pitcher).
        season: Season year.
        config: DPI configuration.

    Returns:
        Dictionary with player_id, season, dpi_mean, n_bip, actual_outs,
        expected_outs.
    """
    config = config or DPIConfig()

    # Try pitcher-based approach first
    weather_query = """
        SELECT
            p.launch_speed, p.launch_angle, p.hc_x, p.hc_y,
            p.bb_type, p.events, p.home_team,
            gw.wind_speed_mph, gw.wind_dir_deg, gw.temp_f, gw.roof_status
        FROM pitches p
        LEFT JOIN game_weather gw ON p.game_pk = gw.game_pk
        WHERE p.pitcher_id = $1
          AND EXTRACT(YEAR FROM p.game_date) = $2
          AND p.type = 'X'
          AND p.events IS NOT NULL
    """
    fallback_query = """
        SELECT
            launch_speed, launch_angle, hc_x, hc_y, bb_type, events, home_team
        FROM pitches
        WHERE pitcher_id = $1
          AND EXTRACT(YEAR FROM game_date) = $2
          AND type = 'X'
          AND events IS NOT NULL
    """
    try:
        bip_df = conn.execute(weather_query, [player_id, season]).fetchdf()
    except (duckdb.CatalogException, duckdb.BinderException):
        bip_df = conn.execute(fallback_query, [player_id, season]).fetchdf()

    result = {
        "player_id": player_id,
        "season": season,
        "dpi_mean": None,
        "n_bip": len(bip_df),
        "actual_outs": 0,
        "expected_outs": 0.0,
    }

    if len(bip_df) < config.min_bip_per_season:
        return result

    actual_outs = int(_is_out(bip_df["events"]).sum())

    has_data = (
        bip_df["launch_speed"].notna()
        & bip_df["launch_angle"].notna()
        & bip_df["hc_x"].notna()
        & bip_df["hc_y"].notna()
        & bip_df["bb_type"].notna()
    )
    bip_with_data = bip_df[has_data]

    if len(bip_with_data) == 0:
        result["actual_outs"] = actual_outs
        return result

    xout_probs = compute_expected_outs(bip_with_data)
    expected_outs = float(xout_probs.sum())
    n_valid = len(bip_with_data)

    dpi = (actual_outs - expected_outs) / n_valid if n_valid > 0 else 0.0

    result.update({
        "dpi_mean": round(dpi, 4),
        "actual_outs": actual_outs,
        "expected_outs": round(expected_outs, 3),
    })
    return result


# ---------------------------------------------------------------------------
# Model class (BaseAnalyticsModel interface)
# ---------------------------------------------------------------------------

class DefensivePressingModel(BaseAnalyticsModel):
    """Defensive Pressing Intensity model.

    Inherits from ``BaseAnalyticsModel`` for consistent lifecycle management.
    """

    def __init__(self, config: DPIConfig | None = None) -> None:
        super().__init__()
        self.config = config or DPIConfig()
        self._team_profiles: dict[str, dict] = {}
        self._is_trained = False
        self._training_season: int | None = None

    @property
    def model_name(self) -> str:
        return "defensive_pressing"

    @property
    def version(self) -> str:
        return "1.0.0"

    def train(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Train the xOut model and compute DPI for all teams.

        Args:
            conn: Open DuckDB connection.
            season: Season year (keyword argument).
            seasons: Seasons for xOut training (keyword argument).

        Returns:
            Dictionary of training metrics.
        """
        season = kwargs.get("season")
        if season is None:
            season = _get_latest_season(conn)
        training_seasons = kwargs.get("seasons")

        logger.info("Training DPI model for season %d", season)

        # Step 1: train the xOut model
        xout_metrics = train_expected_out_model(conn, training_seasons, self.config)

        # Step 2: compute team DPI across the season
        df = batch_calculate(conn, season, self.config)

        self._is_trained = True
        self._training_season = season
        for _, row in df.iterrows():
            self._team_profiles[row["team_id"]] = row.to_dict()

        metrics = {
            "xout_model": xout_metrics,
            "n_teams_scored": len(df),
            "season": season,
            "mean_dpi": round(float(df["dpi_mean"].mean()), 3) if not df.empty else None,
        }

        self.set_training_metadata(
            metrics=metrics,
            params={
                "min_bip_per_game": self.config.min_bip_per_game,
                "min_bip_per_season": self.config.min_bip_per_season,
                "xout_n_estimators": self.config.xout_n_estimators,
                "xout_max_depth": self.config.xout_max_depth,
                "xout_learning_rate": self.config.xout_learning_rate,
                "season": season,
            },
        )
        return metrics

    def predict(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict | pd.DataFrame:
        """Generate DPI predictions.

        Args:
            conn: Open DuckDB connection.
            team_id: Specific team (optional).
            season: Season year (optional).

        Returns:
            DataFrame of team DPI rankings, or dict for a single team.
        """
        team_id = kwargs.get("team_id")
        season = kwargs.get("season", self._training_season)

        if team_id is not None and season is not None:
            return calculate_team_dpi(conn, team_id, season, self.config)

        if season is not None:
            return batch_calculate(conn, season, self.config)

        return {"error": "Provide season (and optionally team_id)"}

    def evaluate(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Evaluate model quality.

        Returns:
            Dictionary with diagnostic statistics.
        """
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        profiles = [p for p in self._team_profiles.values()
                    if p.get("dpi_mean") is not None]

        if not profiles:
            return {"n_teams": 0, "status": "no_profiles"}

        dpis = [p["dpi_mean"] for p in profiles]
        arr = np.array(dpis)

        return self.validate_output({
            "n_teams": len(arr),
            "mean_dpi": round(float(np.mean(arr)), 3),
            "std_dpi": round(float(np.std(arr)), 3),
            "median_dpi": round(float(np.median(arr)), 3),
            "min_dpi": round(float(np.min(arr)), 3),
            "max_dpi": round(float(np.max(arr)), 3),
        })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_latest_season(conn: duckdb.DuckDBPyConnection) -> int:
    """Return the most recent season year in the database."""
    return get_latest_season(conn)


def get_team_game_dpi_timeline(
    conn: duckdb.DuckDBPyConnection,
    team_id: str,
    season: int,
    config: DPIConfig | None = None,
) -> pd.DataFrame:
    """Return game-by-game DPI for a team across the season.

    Args:
        conn: Open DuckDB connection.
        team_id: Three-letter team abbreviation.
        season: Season year.
        config: DPI configuration.

    Returns:
        DataFrame with game_date, game_pk, dpi, n_bip, actual_outs,
        expected_outs, extra_base_prevention columns.
    """
    config = config or DPIConfig()

    query = """
        SELECT DISTINCT game_pk, game_date
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) = $1
          AND (home_team = $2 OR away_team = $2)
        ORDER BY game_date
    """
    games = conn.execute(query, [season, team_id]).fetchdf()

    if games.empty:
        return pd.DataFrame()

    rows = []
    for _, g in games.iterrows():
        result = calculate_game_dpi(conn, int(g["game_pk"]), team_id, config)
        if result["dpi"] is not None:
            rows.append({
                "game_date": g["game_date"],
                "game_pk": int(g["game_pk"]),
                "dpi": result["dpi"],
                "n_bip": result["n_bip"],
                "actual_outs": result["actual_outs"],
                "expected_outs": result["expected_outs"],
                "extra_base_prevention": result["extra_base_prevention"],
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()
