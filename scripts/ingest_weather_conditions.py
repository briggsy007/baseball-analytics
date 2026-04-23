#!/usr/bin/env python
"""
Ingest game-time weather conditions per MLB game into
``data/staging/game_weather.parquet``.

Pipeline:
    1. Open DuckDB read-only; pull distinct game_pk from ``pitches``.
    2. Resolve each game_pk to (venue_name, game_datetime_utc) via the MLB
       StatsAPI schedule endpoint (``hydrate=venue``), batched by day.
    3. Map venue_name -> (latitude, longitude, elevation_m, roof_status)
       using a hard-coded table of every MLB park (and historical/relocation
       aliases). No live scraping.
    4. For each game, fetch NOAA hourly weather at the nearest station via
       ``meteostat.hourly`` and interpolate to game start time. Convert units
       to temp_f, wind_speed_mph, pressure_mb.
    5. Handle domed / retractable-roof stadiums explicitly:
         - Fixed dome (Tropicana Field) -> roof_status='dome', weather NULL.
         - Retractable roof with unknown open/closed at game time ->
           roof_status='retractable_unknown'. Weather is still attached so
           downstream analysts can decide whether to use it.
    6. Write parquet with the schema in the task brief.

Usage (from project root):

    python scripts/ingest_weather_conditions.py
    python scripts/ingest_weather_conditions.py --max-games 100      # smoke
    python scripts/ingest_weather_conditions.py --start-date 2024-04-01 --end-date 2024-10-01
    python scripts/ingest_weather_conditions.py --no-weather         # skip meteostat (metadata only)

STAGING ONLY — this script never writes to DuckDB.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

# Ensure the project root is on sys.path so ``src`` is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.schema import DEFAULT_DB_PATH  # noqa: E402

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_PATH: Path = PROJECT_ROOT / "data" / "staging" / "game_weather.parquet"

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
USER_AGENT = "BaseballAnalyticsPlatform/1.0 (weather-ingest)"
REQUEST_TIMEOUT = 15

# meteostat radius / limit for station lookup
STATION_RADIUS_M = 75_000
STATION_LIMIT = 5

OUTPUT_COLUMNS: list[str] = [
    "game_pk",
    "venue",
    "game_datetime_utc",
    "temp_f",
    "wind_speed_mph",
    "wind_dir_deg",
    "humidity_pct",
    "pressure_mb",
    "roof_status",
    "weather_source",
]

ROOF_OPEN = "open"
ROOF_CLOSED = "closed"
ROOF_DOME = "dome"               # fixed dome — no meaningful outdoor weather
ROOF_RETRACT_UNKNOWN = "retractable_unknown"  # retractable, status not known at ingest time

VALID_ROOF_STATUSES = {ROOF_OPEN, ROOF_CLOSED, ROOF_DOME, ROOF_RETRACT_UNKNOWN}


# ── Venue metadata ─────────────────────────────────────────────────────────
#
# Each venue maps to (latitude, longitude, elevation_m, roof_status).
# Coordinates sourced from MLB.com park-factor pages, Wikipedia, and the
# MLB StatsAPI ``venues`` endpoint (cross-checked 2026-04). Hard-coded so this
# pipeline never depends on live-scraped coordinate data.
#
# Roof policy:
#   - "dome"   -> fixed dome (Tropicana Field). Weather is NOT attached.
#   - "retractable_unknown" -> retractable, at-game state unknown from schedule
#     endpoint alone. Weather IS attached so downstream code can decide.
#   - "open"   -> open-air park. Weather attached.
#
# Retractable roofs (per task brief):
#   Rogers Centre, Tropicana Field (fixed dome treated separately),
#   Miller Park/American Family Field, Chase Field, Minute Maid Park/Daikin Park,
#   Globe Life Field, loanDepot Park, T-Mobile Park.

@dataclass(frozen=True)
class VenueInfo:
    lat: float
    lon: float
    elev_m: int
    roof_status: str  # one of VALID_ROOF_STATUSES


VENUE_REGISTRY: dict[str, VenueInfo] = {
    # ── Current (2025) venues ─────────────────────────────────────────────
    "American Family Field":           VenueInfo(43.0280, -87.9712, 190, ROOF_RETRACT_UNKNOWN),  # Brewers (retractable)
    "Angel Stadium":                   VenueInfo(33.8003, -117.8827,  48, ROOF_OPEN),
    "Busch Stadium":                   VenueInfo(38.6226, -90.1928,  141, ROOF_OPEN),
    "Chase Field":                     VenueInfo(33.4453, -112.0667, 335, ROOF_RETRACT_UNKNOWN),  # Diamondbacks (retractable)
    "Citi Field":                      VenueInfo(40.7571, -73.8458,    4, ROOF_OPEN),
    "Citizens Bank Park":              VenueInfo(39.9061, -75.1665,   15, ROOF_OPEN),
    "Comerica Park":                   VenueInfo(42.3390, -83.0485,  181, ROOF_OPEN),
    "Coors Field":                     VenueInfo(39.7559, -104.9942,1580, ROOF_OPEN),  # altitude!
    "Daikin Park":                     VenueInfo(29.7573, -95.3555,   14, ROOF_RETRACT_UNKNOWN),  # Astros (renamed 2025)
    "Dodger Stadium":                  VenueInfo(34.0739, -118.2400, 150, ROOF_OPEN),
    "Fenway Park":                     VenueInfo(42.3467, -71.0972,    6, ROOF_OPEN),
    "George M. Steinbrenner Field":    VenueInfo(27.9806, -82.5075,    6, ROOF_OPEN),  # Rays temp home 2025
    "Globe Life Field":                VenueInfo(32.7473, -97.0828,  170, ROOF_RETRACT_UNKNOWN),  # Rangers (retractable)
    "Great American Ball Park":        VenueInfo(39.0979, -84.5082,  150, ROOF_OPEN),
    "Kauffman Stadium":                VenueInfo(39.0515, -94.4803,  239, ROOF_OPEN),
    "loanDepot park":                  VenueInfo(25.7781, -80.2197,    2, ROOF_RETRACT_UNKNOWN),  # Marlins (retractable)
    "Nationals Park":                  VenueInfo(38.8730, -77.0074,    3, ROOF_OPEN),
    "Oracle Park":                     VenueInfo(37.7786, -122.3893,   5, ROOF_OPEN),
    "Oriole Park at Camden Yards":     VenueInfo(39.2838, -76.6217,   10, ROOF_OPEN),
    "Petco Park":                      VenueInfo(32.7073, -117.1566,  13, ROOF_OPEN),
    "PNC Park":                        VenueInfo(40.4469, -80.0057,  226, ROOF_OPEN),
    "Progressive Field":               VenueInfo(41.4960, -81.6852,  203, ROOF_OPEN),
    "Rate Field":                      VenueInfo(41.8300, -87.6340,  180, ROOF_OPEN),  # White Sox (renamed 2024 from Guaranteed Rate Field)
    "Rogers Centre":                   VenueInfo(43.6414, -79.3894,   91, ROOF_RETRACT_UNKNOWN),  # Blue Jays (retractable)
    "Sutter Health Park":              VenueInfo(38.5803, -121.5133,   9, ROOF_OPEN),  # Athletics temp home 2025-
    "T-Mobile Park":                   VenueInfo(47.5914, -122.3325,   4, ROOF_RETRACT_UNKNOWN),  # Mariners (retractable)
    "Target Field":                    VenueInfo(44.9817, -93.2776,  251, ROOF_OPEN),
    "Truist Park":                     VenueInfo(33.8908, -84.4678,  317, ROOF_OPEN),
    "Wrigley Field":                   VenueInfo(41.9484, -87.6553,  182, ROOF_OPEN),
    "Yankee Stadium":                  VenueInfo(40.8296, -73.9262,   16, ROOF_OPEN),

    # ── Historical / alias venue names (2015-2024) ────────────────────────
    "Angel Stadium of Anaheim":        VenueInfo(33.8003, -117.8827,  48, ROOF_OPEN),      # renamed 2016
    "AT&T Park":                       VenueInfo(37.7786, -122.3893,   5, ROOF_OPEN),      # Oracle Park (pre-2019)
    "Globe Life Park in Arlington":    VenueInfo(32.7513, -97.0830,  170, ROOF_OPEN),      # Rangers old park (pre-2020)
    "Guaranteed Rate Field":           VenueInfo(41.8300, -87.6340,  180, ROOF_OPEN),      # Rate Field pre-2024 rename
    "Marlins Park":                    VenueInfo(25.7781, -80.2197,    2, ROOF_RETRACT_UNKNOWN),  # loanDepot park (pre-2021)
    "Miller Park":                     VenueInfo(43.0280, -87.9712,  190, ROOF_RETRACT_UNKNOWN),  # American Family Field (pre-2021)
    "Minute Maid Park":                VenueInfo(29.7573, -95.3555,   14, ROOF_RETRACT_UNKNOWN),  # Daikin Park (pre-2025)
    "O.co Coliseum":                   VenueInfo(37.7516, -122.2005,  13, ROOF_OPEN),      # Oakland Coliseum (2015-2016 name)
    "Oakland Coliseum":                VenueInfo(37.7516, -122.2005,  13, ROOF_OPEN),      # A's (2017-2024)
    "Safeco Field":                    VenueInfo(47.5914, -122.3325,   4, ROOF_RETRACT_UNKNOWN),  # T-Mobile Park (pre-2019)
    "Tropicana Field":                 VenueInfo(27.7683, -82.6534,    3, ROOF_DOME),      # Fixed dome — weather not attached
    "Turner Field":                    VenueInfo(33.7350, -84.3891,  320, ROOF_OPEN),      # Braves old park (pre-2017)
    "U.S. Cellular Field":             VenueInfo(41.8300, -87.6340,  180, ROOF_OPEN),      # White Sox (2003-2016)
    "SunTrust Park":                   VenueInfo(33.8908, -84.4678,  317, ROOF_OPEN),      # Truist Park (2017-2020 name)

    # ── Neutral-site / international / specialty venues (2015-2026) ───────
    # These are rare but they DO appear in `pitches.game_pk` and we want
    # meaningful weather rather than unknown-venue NULLs for them.
    "London Stadium":                  VenueInfo(51.5386, -0.0164,    10, ROOF_OPEN),      # 2019 + 2023 Red Sox/Yankees, Cubs/Cardinals
    "Estadio de Beisbol Monterrey":    VenueInfo(25.7209, -100.3157, 540, ROOF_OPEN),      # Mexico City-area series
    "Estadio Alfredo Harp Helu":       VenueInfo(19.4043, -99.0974, 2240, ROOF_OPEN),      # Mexico City 2023 Giants/Padres
    "Hiram Bithorn Stadium":           VenueInfo(18.4249, -66.0719,   15, ROOF_OPEN),      # San Juan 2010, 2017, 2018 MLB games
    "Tokyo Dome":                      VenueInfo(35.7056, 139.7519,   20, ROOF_DOME),      # 2019, 2026 Japan series
    "Gocheok Sky Dome":                VenueInfo(37.4982, 126.8670,   45, ROOF_DOME),      # 2024 Seoul series Dodgers/Padres
    "Olympic Stadium":                 VenueInfo(45.5579, -73.5515,   10, ROOF_DOME),      # Montreal Blue Jays pre-season exhibition
    "Field of Dreams":                 VenueInfo(42.4724, -90.6479,  270, ROOF_OPEN),      # 2021 Yankees/White Sox, 2022 Cubs/Reds
    "Fort Bragg Field":                VenueInfo(35.1410, -79.0062,  170, ROOF_OPEN),      # 2016 Marlins/Braves
    "Rickwood Field":                  VenueInfo(33.5041, -86.8567,  180, ROOF_OPEN),      # 2024 Giants/Cardinals Negro Leagues tribute
    "Bristol Motor Speedway":          VenueInfo(36.5157, -82.2573,  490, ROOF_OPEN),      # 2024 Reds/Braves
    "BB&T Ballpark":                   VenueInfo(40.6487, -74.2310,   25, ROOF_OPEN),      # Charlotte — ambiguous, approx
    "TD Ameritrade Park":              VenueInfo(41.2565, -95.9258,  320, ROOF_OPEN),      # Omaha 2017 College-World-Series exhibition

    # Note: pure spring-training (Salt River, Sloan, Camelback, Publix Field,
    # Hammond, Peoria, Surprise, etc.) are NOT in this registry by design.
    # They are training facilities — weather is captured but roof_status
    # is left NULL so downstream analytics can filter them out trivially.
}


# ── MLB schedule API ───────────────────────────────────────────────────────


def _get_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": USER_AGENT})
    return sess


def fetch_schedule_batch(
    session: requests.Session,
    start_date: date,
    end_date: date,
) -> dict[int, dict]:
    """Fetch MLB schedule for a date range and return game_pk -> metadata.

    Returns a dict keyed by game_pk with fields:
        venue: venue name
        game_datetime_utc: pandas Timestamp (UTC)
        status: abstractGameState
    """
    params = {
        "sportId": 1,
        "startDate": start_date.strftime("%Y-%m-%d"),
        "endDate": end_date.strftime("%Y-%m-%d"),
        "hydrate": "venue",
    }
    for attempt in range(3):
        try:
            resp = session.get(MLB_SCHEDULE_URL, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as exc:
            if attempt == 2:
                logger.error("Schedule fetch failed for %s..%s: %s", start_date, end_date, exc)
                return {}
            time.sleep(2 ** attempt)
    else:  # pragma: no cover
        return {}

    out: dict[int, dict] = {}
    for d in data.get("dates", []):
        for g in d.get("games", []):
            gpk = g.get("gamePk")
            if not gpk:
                continue
            venue_name = (g.get("venue") or {}).get("name")
            raw_dt = g.get("gameDate")  # ISO8601 UTC, e.g. "2024-07-15T23:05:00Z"
            try:
                dt_utc = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
            except (TypeError, ValueError):
                dt_utc = None
            out[int(gpk)] = {
                "venue": venue_name,
                "game_datetime_utc": dt_utc,
                "status": (g.get("status") or {}).get("abstractGameState", ""),
            }
    return out


def resolve_game_metadata(
    game_pks: set[int],
    game_dates: dict[int, date],
) -> pd.DataFrame:
    """Resolve venue + datetime for each game_pk via MLB schedule API.

    Batches requests by month to stay under API rate limits while minimising
    round-trips. Games whose game_pk is not returned are left with NULLs.
    """
    session = _get_session()
    # Group game_pks by year-month for batched fetches.
    by_month: dict[tuple[int, int], list[int]] = {}
    for pk in game_pks:
        gd = game_dates.get(pk)
        if gd is None:
            continue
        by_month.setdefault((gd.year, gd.month), []).append(pk)

    metadata: dict[int, dict] = {}
    months = sorted(by_month.keys())
    logger.info("Fetching MLB schedule across %d months", len(months))
    for i, (yr, mo) in enumerate(months):
        start = date(yr, mo, 1)
        # month end via next-month-first - 1
        if mo == 12:
            end = date(yr + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(yr, mo + 1, 1) - timedelta(days=1)
        batch = fetch_schedule_batch(session, start, end)
        metadata.update(batch)
        if (i + 1) % 12 == 0:
            logger.info("  schedule fetched through %s (%d games resolved)", end, len(metadata))
        # polite rate limit
        time.sleep(0.10)

    rows = []
    for pk in game_pks:
        m = metadata.get(pk, {})
        rows.append({
            "game_pk": pk,
            "venue": m.get("venue"),
            "game_datetime_utc": m.get("game_datetime_utc"),
            "status": m.get("status", ""),
            "game_date": game_dates.get(pk),
        })
    df = pd.DataFrame(rows)
    return df


# ── Weather fetch via meteostat ────────────────────────────────────────────


def _get_meteostat():
    """Import meteostat lazily so missing install does not crash metadata path."""
    try:
        from meteostat import hourly, stations, Parameter, Point  # type: ignore
        return hourly, stations, Parameter, Point
    except Exception as exc:
        logger.warning("meteostat import failed (%s); weather will be NULL.", exc)
        return None, None, None, None


# Station cache: venue name -> list of station IDs (in distance order) and
# station distance (meters) to the park coordinates.
_station_cache: dict[str, list[tuple[str, float]]] = {}


def find_stations_for_venue(venue_name: str) -> list[tuple[str, float]]:
    """Return [(station_id, distance_m), ...] ordered by distance."""
    if venue_name in _station_cache:
        return _station_cache[venue_name]
    info = VENUE_REGISTRY.get(venue_name)
    if info is None:
        return []
    hourly, stations, Parameter, Point = _get_meteostat()
    if Point is None or stations is None:
        return []
    pt = Point(latitude=info.lat, longitude=info.lon, elevation=info.elev_m)
    try:
        nearby = stations.nearby(pt, radius=STATION_RADIUS_M, limit=STATION_LIMIT)
    except Exception as exc:
        logger.warning("station lookup failed for %s: %s", venue_name, exc)
        _station_cache[venue_name] = []
        return []
    if nearby is None or nearby.empty:
        _station_cache[venue_name] = []
        return []
    out: list[tuple[str, float]] = []
    for idx, row in nearby.iterrows():
        out.append((str(idx), float(row.get("distance", np.nan))))
    _station_cache[venue_name] = out
    return out


def fetch_weather_day_for_station(
    station_id: str,
    day: date,
) -> Optional[pd.DataFrame]:
    """Fetch 24-hour weather for the given station/day. Returns None on failure."""
    hourly, stations, Parameter, Point = _get_meteostat()
    if hourly is None:
        return None
    start = datetime(day.year, day.month, day.day, 0, 0)
    end = datetime(day.year, day.month, day.day, 23, 59)
    try:
        params = [
            Parameter.TEMP,
            Parameter.WSPD,
            Parameter.WDIR,
            Parameter.RHUM,
            Parameter.PRES,
        ]
        ts = hourly(station_id, start, end, parameters=params)
        if ts.empty:
            return None
        df = ts.fetch()
        if df is None or df.empty:
            return None
        # Ensure timezone-naive (meteostat returns naive UTC)
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        return df
    except Exception as exc:
        logger.debug("meteostat fetch failed (station=%s day=%s): %s", station_id, day, exc)
        return None


def interpolate_to_gametime(
    weather_df: pd.DataFrame,
    game_datetime_utc: datetime,
) -> dict[str, float]:
    """Linear-interpolate hourly weather to the game start time.

    weather_df is indexed by hourly UTC timestamps with columns:
        temp (C), rhum (%), wdir (deg), wspd (km/h), pres (hPa/mb)
    """
    if weather_df is None or weather_df.empty:
        return {k: np.nan for k in ("temp_c", "wspd_kmh", "wdir", "rhum", "pres")}
    # Ensure naive UTC for comparison
    target = game_datetime_utc
    if target.tzinfo is not None:
        target = target.astimezone(timezone.utc).replace(tzinfo=None)
    # Build an interpolation using pandas reindex + interpolate.
    # Cast to float64 first so inserting a NaN target row does not trigger
    # the pandas "concat with empty/all-NA entries" FutureWarning.
    df = weather_df.copy().astype("float64")
    df = df.sort_index()
    # Add the target timestamp, then interpolate numerically.
    if target not in df.index:
        new_row = pd.DataFrame(
            {col: [np.nan] for col in df.columns},
            index=pd.DatetimeIndex([target]),
        ).astype("float64")
        df = pd.concat([df, new_row]).sort_index()
    # Linear interp time-based for numeric columns. Wind direction should
    # really be interpolated on the unit circle — close enough for 1 hour.
    df = df.interpolate(method="time").ffill().bfill()
    row = df.loc[target]
    # If duplicate index (rare, but interpolation can double it) take first
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    out = {}
    for col in ("temp", "wspd", "wdir", "rhum", "pres"):
        v = row.get(col, np.nan)
        out[{
            "temp": "temp_c",
            "wspd": "wspd_kmh",
            "wdir": "wdir",
            "rhum": "rhum",
            "pres": "pres",
        }[col]] = float(v) if pd.notna(v) else np.nan
    return out


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0 if pd.notna(c) else np.nan


def kmh_to_mph(v: float) -> float:
    return v * 0.621371 if pd.notna(v) else np.nan


# ── Pipeline ───────────────────────────────────────────────────────────────


@dataclass
class IngestStats:
    n_games_total: int = 0
    n_games_with_venue: int = 0
    n_games_with_datetime: int = 0
    n_games_unknown_venue: int = 0
    n_games_dome_skip: int = 0
    n_games_retractable_unknown: int = 0
    n_games_weather_attached: int = 0
    n_games_weather_failed: int = 0
    unknown_venues: dict[str, int] = field(default_factory=dict)
    station_distance_m_p50: float = 0.0
    station_distance_m_p95: float = 0.0
    station_distance_m_max: float = 0.0


def read_distinct_games(db_path: Path) -> pd.DataFrame:
    """Read distinct game_pk, game_date, home_team from pitches (read-only)."""
    logger.info("Opening DuckDB (read-only): %s", db_path)
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        df = conn.execute("""
            SELECT  game_pk,
                    MIN(game_date) AS game_date,
                    ANY_VALUE(home_team) AS home_team
            FROM    pitches
            WHERE   game_pk IS NOT NULL
            GROUP BY game_pk
            ORDER BY game_pk
        """).fetchdf()
    finally:
        conn.close()
    logger.info("Distinct games in pitches: %d", len(df))
    return df


def build_weather_frame(
    games_df: pd.DataFrame,
    no_weather: bool = False,
    max_games: Optional[int] = None,
) -> tuple[pd.DataFrame, IngestStats]:
    """Build the full (game_pk, venue, datetime, weather, roof) frame."""
    stats = IngestStats()
    stats.n_games_total = len(games_df)

    # 1. Resolve metadata from MLB API
    game_dates = {int(r["game_pk"]): r["game_date"] for _, r in games_df.iterrows()}
    game_pks = set(game_dates.keys())
    if max_games is not None:
        subset = sorted(game_pks)[:max_games]
        game_pks = set(subset)
        game_dates = {pk: game_dates[pk] for pk in game_pks}
        stats.n_games_total = len(game_pks)
        logger.info("Smoke mode: limiting to %d games", max_games)

    meta_df = resolve_game_metadata(game_pks, game_dates)
    meta_df["has_venue"] = meta_df["venue"].notna()
    meta_df["has_datetime"] = meta_df["game_datetime_utc"].notna()
    stats.n_games_with_venue = int(meta_df["has_venue"].sum())
    stats.n_games_with_datetime = int(meta_df["has_datetime"].sum())

    # 2. Annotate roof_status via venue registry
    def _roof(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        info = VENUE_REGISTRY.get(v)
        return info.roof_status if info is not None else None

    meta_df["roof_status"] = meta_df["venue"].apply(_roof)
    unknown_mask = meta_df["has_venue"] & meta_df["roof_status"].isna()
    if unknown_mask.any():
        vc = meta_df.loc[unknown_mask, "venue"].value_counts().to_dict()
        stats.unknown_venues = vc
        stats.n_games_unknown_venue = int(unknown_mask.sum())
        logger.warning(
            "Unknown venue names (add to VENUE_REGISTRY): %s",
            {k: int(v) for k, v in vc.items()}
        )

    # 3. Fetch weather per game
    temps_f: list[float] = []
    winds_mph: list[float] = []
    wind_dirs: list[float] = []
    humid: list[float] = []
    press: list[float] = []
    sources: list[str] = []
    station_distances: list[float] = []

    # Cache per (venue_name, day) to avoid refetching
    day_cache: dict[tuple[str, date], pd.DataFrame] = {}

    for _, row in meta_df.iterrows():
        venue = row["venue"]
        gdt = row["game_datetime_utc"]
        roof = row["roof_status"]

        # Dome parks: explicitly skip weather (no meaningful outdoor reading)
        if roof == ROOF_DOME:
            stats.n_games_dome_skip += 1
            temps_f.append(np.nan)
            winds_mph.append(np.nan)
            wind_dirs.append(np.nan)
            humid.append(np.nan)
            press.append(np.nan)
            sources.append("dome_skipped")
            continue

        # Missing metadata or no weather mode -> NULL weather
        if no_weather or not isinstance(venue, str) or gdt is None or roof is None:
            temps_f.append(np.nan)
            winds_mph.append(np.nan)
            wind_dirs.append(np.nan)
            humid.append(np.nan)
            press.append(np.nan)
            sources.append("skipped" if no_weather else "no_metadata")
            stats.n_games_weather_failed += 1
            continue

        if roof == ROOF_RETRACT_UNKNOWN:
            stats.n_games_retractable_unknown += 1

        # Nearest station(s)
        sts = find_stations_for_venue(venue)
        if not sts:
            temps_f.append(np.nan)
            winds_mph.append(np.nan)
            wind_dirs.append(np.nan)
            humid.append(np.nan)
            press.append(np.nan)
            sources.append("no_station")
            stats.n_games_weather_failed += 1
            continue

        # Try stations in distance order; first one with data wins
        day = gdt.date() if hasattr(gdt, "date") else gdt
        weather_df = None
        chosen_station = None
        chosen_distance = np.nan
        for station_id, dist_m in sts:
            key = (station_id, day)
            if key in day_cache:
                cached = day_cache[key]
            else:
                cached = fetch_weather_day_for_station(station_id, day)
                day_cache[key] = cached
            if cached is not None and not cached.empty:
                weather_df = cached
                chosen_station = station_id
                chosen_distance = dist_m
                break

        if weather_df is None:
            temps_f.append(np.nan)
            winds_mph.append(np.nan)
            wind_dirs.append(np.nan)
            humid.append(np.nan)
            press.append(np.nan)
            sources.append("meteostat_no_data")
            stats.n_games_weather_failed += 1
            continue

        interp = interpolate_to_gametime(weather_df, gdt)
        temps_f.append(c_to_f(interp["temp_c"]))
        winds_mph.append(kmh_to_mph(interp["wspd_kmh"]))
        wind_dirs.append(interp["wdir"])
        humid.append(interp["rhum"])
        press.append(interp["pres"])
        sources.append(f"meteostat:{chosen_station}")
        station_distances.append(chosen_distance)
        stats.n_games_weather_attached += 1

    meta_df["temp_f"] = temps_f
    meta_df["wind_speed_mph"] = winds_mph
    meta_df["wind_dir_deg"] = wind_dirs
    meta_df["humidity_pct"] = humid
    meta_df["pressure_mb"] = press
    meta_df["weather_source"] = sources

    # Station distance QA
    if station_distances:
        arr = np.array([d for d in station_distances if pd.notna(d)])
        if len(arr):
            stats.station_distance_m_p50 = float(np.percentile(arr, 50))
            stats.station_distance_m_p95 = float(np.percentile(arr, 95))
            stats.station_distance_m_max = float(arr.max())

    # Canonicalise to output column order / schema
    meta_df["game_pk"] = meta_df["game_pk"].astype("Int64")
    meta_df["game_datetime_utc"] = pd.to_datetime(meta_df["game_datetime_utc"], utc=True, errors="coerce")
    # Output gets naive UTC (parquet TIMESTAMP)
    meta_df["game_datetime_utc"] = meta_df["game_datetime_utc"].dt.tz_convert("UTC").dt.tz_localize(None)

    return meta_df[OUTPUT_COLUMNS].copy(), stats


# ── Parquet writer ─────────────────────────────────────────────────────────


def write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    """Write the staged parquet with explicit pyarrow schema."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema([
        pa.field("game_pk", pa.int64()),
        pa.field("venue", pa.string()),
        pa.field("game_datetime_utc", pa.timestamp("us")),
        pa.field("temp_f", pa.float64()),
        pa.field("wind_speed_mph", pa.float64()),
        pa.field("wind_dir_deg", pa.float64()),
        pa.field("humidity_pct", pa.float64()),
        pa.field("pressure_mb", pa.float64()),
        pa.field("roof_status", pa.string()),
        pa.field("weather_source", pa.string()),
    ])
    df2 = df[OUTPUT_COLUMNS].copy()
    # Coerce numerics
    for col in ("temp_f", "wind_speed_mph", "wind_dir_deg", "humidity_pct", "pressure_mb"):
        df2[col] = pd.to_numeric(df2[col], errors="coerce").astype("float64")
    df2["game_pk"] = df2["game_pk"].astype("Int64")
    # Parquet TIMESTAMP type
    df2["game_datetime_utc"] = pd.to_datetime(df2["game_datetime_utc"], errors="coerce")
    table = pa.Table.from_pandas(df2, schema=schema, preserve_index=False)
    pq.write_table(table, str(out_path), compression="zstd")
    logger.info("Wrote %d rows -> %s (%.1f KB)",
                len(df2), out_path, out_path.stat().st_size / 1024.0)


# ── Reporting ──────────────────────────────────────────────────────────────


def print_summary(df: pd.DataFrame, stats: IngestStats, out_path: Path) -> None:
    print("\n" + "=" * 72)
    print("  GAME WEATHER INGESTION — SUMMARY")
    print("=" * 72)
    print(f"  distinct games in pitches:         {stats.n_games_total:,}")
    print(f"  games w/ venue resolved:           {stats.n_games_with_venue:,} "
          f"({100 * stats.n_games_with_venue / max(stats.n_games_total, 1):.1f}%)")
    print(f"  games w/ datetime resolved:        {stats.n_games_with_datetime:,} "
          f"({100 * stats.n_games_with_datetime / max(stats.n_games_total, 1):.1f}%)")
    print(f"  games w/ unknown venue (audit):    {stats.n_games_unknown_venue:,}")
    print(f"  games dome-skipped (Tropicana):    {stats.n_games_dome_skip:,}")
    print(f"  games retractable (unknown state): {stats.n_games_retractable_unknown:,}")
    print(f"  games with weather attached:       {stats.n_games_weather_attached:,} "
          f"({100 * stats.n_games_weather_attached / max(stats.n_games_total, 1):.1f}%)")
    print(f"  games weather failed/no-data:      {stats.n_games_weather_failed:,}")

    if stats.station_distance_m_max > 0:
        print("\n  Nearest-station distance QA (meters, games w/ weather):")
        print(f"    p50 = {stats.station_distance_m_p50:,.0f}")
        print(f"    p95 = {stats.station_distance_m_p95:,.0f}")
        print(f"    max = {stats.station_distance_m_max:,.0f}")

    if stats.unknown_venues:
        print("\n  Unknown venues encountered (add to VENUE_REGISTRY):")
        for name, count in sorted(stats.unknown_venues.items(), key=lambda x: -x[1]):
            print(f"    {name!r}: {count}")

    # Venue breakdown
    if not df.empty:
        print("\n  Rows per venue (top 32):")
        vc = df["venue"].value_counts().head(32).to_dict()
        for name, count in vc.items():
            roof = VENUE_REGISTRY.get(name)
            rs = roof.roof_status if roof else "?"
            print(f"    {str(name)[:42]:42s} {count:6,d}   roof={rs}")

        print("\n  Roof-status breakdown:")
        for rs, count in df["roof_status"].fillna("NULL").value_counts().items():
            print(f"    {rs:24s} {count:6,d}")

        print("\n  Weather source breakdown:")
        for src, count in df["weather_source"].fillna("NULL").value_counts().head(10).items():
            print(f"    {str(src)[:40]:40s} {count:6,d}")

    print(f"\n  Output parquet: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH),
                        help=f"DuckDB path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT_PATH),
                        help=f"Output parquet path (default: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Limit to first N game_pks (smoke test).")
    parser.add_argument("--start-date", default=None,
                        help="Filter pitches by game_date >= YYYY-MM-DD")
    parser.add_argument("--end-date", default=None,
                        help="Filter pitches by game_date <= YYYY-MM-DD")
    parser.add_argument("--no-weather", action="store_true",
                        help="Skip meteostat calls; stage venue + datetime + roof only.")
    parser.add_argument("--log-level", default="INFO",
                        help="Python logging level (default: INFO).")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    db_path = Path(args.db)
    out_path = Path(args.out)

    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}", file=sys.stderr)
        return 1

    games_df = read_distinct_games(db_path)
    if args.start_date:
        games_df = games_df[pd.to_datetime(games_df["game_date"])
                            >= pd.Timestamp(args.start_date)].copy()
    if args.end_date:
        games_df = games_df[pd.to_datetime(games_df["game_date"])
                            <= pd.Timestamp(args.end_date)].copy()

    df, stats = build_weather_frame(
        games_df,
        no_weather=args.no_weather,
        max_games=args.max_games,
    )
    write_parquet(df, out_path)
    print_summary(df, stats, out_path)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
