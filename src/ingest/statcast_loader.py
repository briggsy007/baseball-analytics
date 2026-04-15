"""
Statcast pitch-by-pitch data ingestion module.

Loads historical and daily Statcast data from Baseball Savant via pybaseball,
season-level batting/pitching stats from FanGraphs, and the Chadwick player ID
crosswalk. Cleans column names to match the DuckDB schema defined in
``src.db.schema`` and provides insertion helpers for each target table.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import pybaseball

# ---------------------------------------------------------------------------
# Project root (works regardless of where the module is imported from)
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column mapping: pybaseball Statcast output  ->  our DuckDB pitches schema
# ---------------------------------------------------------------------------
_STATCAST_RENAME: dict[str, str] = {
    "game_pk": "game_pk",
    "game_date": "game_date",
    "pitcher": "pitcher_id",
    "batter": "batter_id",
    "pitch_type": "pitch_type",
    "pitch_name": "pitch_name",
    "release_speed": "release_speed",
    "release_spin_rate": "release_spin_rate",
    "spin_axis": "spin_axis",
    "pfx_x": "pfx_x",
    "pfx_z": "pfx_z",
    "plate_x": "plate_x",
    "plate_z": "plate_z",
    "release_extension": "release_extension",
    "release_pos_x": "release_pos_x",
    "release_pos_y": "release_pos_y",
    "release_pos_z": "release_pos_z",
    "launch_speed": "launch_speed",
    "launch_angle": "launch_angle",
    "hit_distance_sc": "hit_distance",
    "hc_x": "hc_x",
    "hc_y": "hc_y",
    "bb_type": "bb_type",
    "estimated_ba_using_speedangle": "estimated_ba",
    "estimated_woba_using_speedangle": "estimated_woba",
    "delta_home_win_exp": "delta_home_win_exp",
    "delta_run_exp": "delta_run_exp",
    "inning": "inning",
    "inning_topbot": "inning_topbot",
    "outs_when_up": "outs_when_up",
    "balls": "balls",
    "strikes": "strikes",
    "on_1b": "on_1b",
    "on_2b": "on_2b",
    "on_3b": "on_3b",
    "stand": "stand",
    "p_throws": "p_throws",
    "at_bat_number": "at_bat_number",
    "pitch_number": "pitch_number",
    "des": "description",
    "events": "events",
    "type": "type",
    "home_team": "home_team",
    "away_team": "away_team",
    "woba_value": "woba_value",
    "woba_denom": "woba_denom",
    "babip_value": "babip_value",
    "iso_value": "iso_value",
    "zone": "zone",
    "effective_speed": "effective_speed",
    "if_fielding_alignment": "if_fielding_alignment",
    "of_fielding_alignment": "of_fielding_alignment",
    "fielder_2": "fielder_2",
}

# The exact set of columns we keep (our schema column order).
_PITCHES_COLUMNS: list[str] = [
    "game_pk",
    "game_date",
    "pitcher_id",
    "batter_id",
    "pitch_type",
    "pitch_name",
    "release_speed",
    "release_spin_rate",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "plate_x",
    "plate_z",
    "release_extension",
    "release_pos_x",
    "release_pos_y",
    "release_pos_z",
    "launch_speed",
    "launch_angle",
    "hit_distance",
    "hc_x",
    "hc_y",
    "bb_type",
    "estimated_ba",
    "estimated_woba",
    "delta_home_win_exp",
    "delta_run_exp",
    "inning",
    "inning_topbot",
    "outs_when_up",
    "balls",
    "strikes",
    "on_1b",
    "on_2b",
    "on_3b",
    "stand",
    "p_throws",
    "at_bat_number",
    "pitch_number",
    "description",
    "events",
    "type",
    "home_team",
    "away_team",
    "woba_value",
    "woba_denom",
    "babip_value",
    "iso_value",
    "zone",
    "effective_speed",
    "if_fielding_alignment",
    "of_fielding_alignment",
    "fielder_2",
]


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════


def _get_default_conn() -> duckdb.DuckDBPyConnection:
    """Open a standalone DuckDB connection at the default path."""
    db_path = ROOT / "data" / "baseball.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def _clean_statcast(df: pd.DataFrame) -> pd.DataFrame:
    """Rename and filter a raw pybaseball Statcast DataFrame to match our schema.

    - Renames columns according to ``_STATCAST_RENAME``.
    - Drops any columns that are not in ``_PITCHES_COLUMNS``.
    - Adds missing schema columns as ``NaN`` if the source omits them.
    - Casts ``game_date`` to ``datetime64[ns]``.

    Returns:
        A cleaned DataFrame with columns matching the DuckDB ``pitches`` table.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=_PITCHES_COLUMNS)

    # Rename columns that exist in the raw DataFrame.
    rename_map = {k: v for k, v in _STATCAST_RENAME.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Ensure every schema column is present (fill missing ones with NaN).
    for col in _PITCHES_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    # Keep only schema columns, in schema order.
    df = df[_PITCHES_COLUMNS].copy()

    # Normalise date column.
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # Cast integer columns that may have become float due to NaN.
    _int_cols = [
        "game_pk", "pitcher_id", "batter_id", "inning", "outs_when_up",
        "balls", "strikes", "at_bat_number", "pitch_number", "zone",
        "fielder_2",
    ]
    for col in _int_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _pct_to_float(val: object) -> Optional[float]:
    """Convert a FanGraphs percentage string like ``'25.3 %'`` to ``0.253``.

    Handles plain floats, strings with ``%``, and ``NaN`` values.
    """
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (int, float)):
        # FanGraphs sometimes returns percentages already as floats (e.g. 25.3)
        # If the value is > 1 it is almost certainly a raw percentage.
        # NOTE: values of exactly 1.0 are ambiguous (100% vs 1%) -- we treat
        # them as fractions.  In practice FanGraphs K%/BB% are always > 1.0
        # when expressed as raw percentages and < 1.0 when already fractional.
        return float(val) / 100.0 if abs(float(val)) > 1.0 else float(val)
    s = str(val).strip().replace("%", "").strip()
    try:
        v = float(s)
        return v / 100.0 if abs(v) > 1.0 else v
    except (ValueError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Validation helpers
# ═══════════════════════════════════════════════════════════════════════════

# Pitch types recognized by Statcast (plus intentional ball and pitch-out)
_VALID_PITCH_TYPES: set[str] = {
    "FF", "SI", "FC", "SL", "CU", "KC", "CS", "CH", "FS", "FO",
    "KN", "EP", "SC", "ST", "SV", "IN", "PO", "FA", "AB",
}

# Plausible release speed range (MPH)
_MIN_VELOCITY = 40.0
_MAX_VELOCITY = 110.0


def validate_pitches(df: pd.DataFrame) -> pd.DataFrame:
    """Validate critical fields in a pitch DataFrame, logging and dropping bad rows.

    Checks:
      - game_pk is not null
      - pitch_type is in the recognized set (when not null)
      - release_speed is within [40, 110] MPH (when not null)

    Returns:
        DataFrame with invalid rows removed.
    """
    if df.empty:
        return df

    initial_len = len(df)

    # --- game_pk must not be null ---
    null_gpk = df["game_pk"].isna()
    if null_gpk.any():
        logger.warning("Validation: dropping %d rows with null game_pk", null_gpk.sum())
        df = df[~null_gpk].copy()

    # --- pitch_type must be recognized (when present) ---
    if "pitch_type" in df.columns:
        has_pt = df["pitch_type"].notna() & (df["pitch_type"] != "")
        bad_pt = has_pt & ~df["pitch_type"].isin(_VALID_PITCH_TYPES)
        if bad_pt.any():
            bad_vals = df.loc[bad_pt, "pitch_type"].unique().tolist()
            logger.warning(
                "Validation: dropping %d rows with unrecognized pitch_type: %s",
                bad_pt.sum(), bad_vals[:10],
            )
            df = df[~bad_pt].copy()

    # --- release_speed must be plausible (when present) ---
    if "release_speed" in df.columns:
        has_velo = df["release_speed"].notna()
        bad_velo = has_velo & (
            (df["release_speed"] < _MIN_VELOCITY) | (df["release_speed"] > _MAX_VELOCITY)
        )
        if bad_velo.any():
            logger.warning(
                "Validation: dropping %d rows with velocity outside [%.0f, %.0f]",
                bad_velo.sum(), _MIN_VELOCITY, _MAX_VELOCITY,
            )
            df = df[~bad_velo].copy()

    dropped = initial_len - len(df)
    if dropped > 0:
        logger.info("Validation: kept %d / %d rows (%d dropped)", len(df), initial_len, dropped)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Retry helper for network calls
# ═══════════════════════════════════════════════════════════════════════════


def _retry_network_call(fn, *args, max_retries: int = 3, **kwargs):
    """Call *fn* with retry + exponential backoff on failure.

    Args:
        fn: Callable to invoke.
        max_retries: Number of attempts (default 3).

    Returns:
        The return value of *fn*.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "Retry %d/%d for %s failed (%s), waiting %ds",
                attempt + 1, max_retries, fn.__name__, exc, wait,
            )
            if attempt < max_retries - 1:
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# Data freshness helpers
# ═══════════════════════════════════════════════════════════════════════════


def update_data_freshness(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    max_game_date: Optional[str] = None,
    row_count: Optional[int] = None,
) -> None:
    """Record a freshness watermark after a successful ETL run.

    Upserts into the ``data_freshness`` table.
    """
    try:
        existing = conn.execute(
            "SELECT table_name FROM data_freshness WHERE table_name = ?",
            [table_name],
        ).fetchone()

        if existing:
            conn.execute(
                """UPDATE data_freshness
                   SET max_game_date = ?, row_count = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE table_name = ?""",
                [max_game_date, row_count, table_name],
            )
        else:
            conn.execute(
                """INSERT INTO data_freshness (table_name, max_game_date, row_count, updated_at)
                   VALUES (?, ?, ?, CURRENT_TIMESTAMP)""",
                [table_name, max_game_date, row_count],
            )
    except Exception:
        logger.warning("Could not update data_freshness for %s", table_name, exc_info=True)


def check_data_freshness(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    target_date: str,
) -> bool:
    """Return True if *table_name* already has data through *target_date*.

    This lets callers skip redundant ETL work.
    """
    try:
        row = conn.execute(
            "SELECT max_game_date FROM data_freshness WHERE table_name = ?",
            [table_name],
        ).fetchone()
        if row and row[0] is not None:
            return str(row[0]) >= target_date
    except Exception:
        pass
    return False


# ═══════════════════════════════════════════════════════════════════════════
# Public API — Statcast pitch-level loaders
# ═══════════════════════════════════════════════════════════════════════════


def load_statcast_range(
    start_date: str,
    end_date: str,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """Fetch Statcast pitch-by-pitch data for a date range.

    Args:
        start_date: ISO format start date (``"2024-04-01"``).
        end_date:   ISO format end date   (``"2024-04-30"``).
        conn:       Optional DuckDB connection. If provided, data is also
                    inserted into the ``pitches`` table.

    Returns:
        Cleaned ``pandas.DataFrame`` with columns matching the pitches schema.
    """
    logger.info("Loading Statcast data from %s to %s", start_date, end_date)
    try:
        raw = _retry_network_call(
            pybaseball.statcast, start_dt=start_date, end_dt=end_date
        )
    except Exception:
        logger.exception("Failed to fetch Statcast data for %s – %s", start_date, end_date)
        return pd.DataFrame(columns=_PITCHES_COLUMNS)

    df = _clean_statcast(raw)
    df = validate_pitches(df)
    logger.info("Fetched %d pitches for %s – %s", len(df), start_date, end_date)

    if conn is not None and not df.empty:
        insert_pitches(conn, df)

    return df


def load_statcast_pitcher(
    pitcher_id: int,
    start_date: str,
    end_date: str,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """Fetch Statcast data for a specific pitcher.

    Args:
        pitcher_id: MLB Advanced Media pitcher ID.
        start_date: ISO format start date.
        end_date:   ISO format end date.
        conn:       Optional DuckDB connection for insertion.

    Returns:
        Cleaned DataFrame matching the pitches schema.
    """
    logger.info("Loading Statcast pitcher %d (%s – %s)", pitcher_id, start_date, end_date)
    try:
        raw = _retry_network_call(
            pybaseball.statcast_pitcher,
            start_dt=start_date, end_dt=end_date, player_id=pitcher_id,
        )
    except Exception:
        logger.exception("Failed to fetch pitcher %d", pitcher_id)
        return pd.DataFrame(columns=_PITCHES_COLUMNS)

    df = _clean_statcast(raw)
    df = validate_pitches(df)

    if conn is not None and not df.empty:
        insert_pitches(conn, df)

    return df


def load_statcast_batter(
    batter_id: int,
    start_date: str,
    end_date: str,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """Fetch Statcast data for a specific batter.

    Args:
        batter_id: MLB Advanced Media batter ID.
        start_date: ISO format start date.
        end_date:   ISO format end date.
        conn:       Optional DuckDB connection for insertion.

    Returns:
        Cleaned DataFrame matching the pitches schema.
    """
    logger.info("Loading Statcast batter %d (%s – %s)", batter_id, start_date, end_date)
    try:
        raw = _retry_network_call(
            pybaseball.statcast_batter,
            start_dt=start_date, end_dt=end_date, player_id=batter_id,
        )
    except Exception:
        logger.exception("Failed to fetch batter %d", batter_id)
        return pd.DataFrame(columns=_PITCHES_COLUMNS)

    df = _clean_statcast(raw)
    df = validate_pitches(df)

    if conn is not None and not df.empty:
        insert_pitches(conn, df)

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Public API — Season-level FanGraphs loaders
# ═══════════════════════════════════════════════════════════════════════════


def _load_batting_stats_bref(season: int) -> pd.DataFrame:
    """Fallback: load batting stats from Baseball Reference when FanGraphs is unavailable.

    Returns a DataFrame with columns mapped to our schema. Baseball Reference
    provides ``mlbID`` directly, so no Chadwick register lookup is needed.
    """
    from pybaseball import batting_stats_bref

    logger.info("Falling back to Baseball Reference batting stats for %d", season)
    raw = batting_stats_bref(season)
    if raw is None or raw.empty:
        return pd.DataFrame()

    # Filter to MLB level only
    if "Lev" in raw.columns:
        raw = raw[raw["Lev"].str.contains("Maj", na=False)].copy()

    out = pd.DataFrame()
    out["player_id"] = pd.to_numeric(raw.get("mlbID"), errors="coerce")
    out["season"] = season

    # Counting stats
    out["pa"] = pd.to_numeric(raw.get("PA"), errors="coerce")
    out["ab"] = pd.to_numeric(raw.get("AB"), errors="coerce")
    out["h"] = pd.to_numeric(raw.get("H"), errors="coerce")
    out["doubles"] = pd.to_numeric(raw.get("2B"), errors="coerce")
    out["triples"] = pd.to_numeric(raw.get("3B"), errors="coerce")
    out["hr"] = pd.to_numeric(raw.get("HR"), errors="coerce")
    out["rbi"] = pd.to_numeric(raw.get("RBI"), errors="coerce")
    out["bb"] = pd.to_numeric(raw.get("BB"), errors="coerce")
    out["so"] = pd.to_numeric(raw.get("SO"), errors="coerce")
    out["sb"] = pd.to_numeric(raw.get("SB"), errors="coerce")
    out["cs"] = pd.to_numeric(raw.get("CS"), errors="coerce")

    # Rate stats
    out["ba"] = pd.to_numeric(raw.get("BA"), errors="coerce")
    out["obp"] = pd.to_numeric(raw.get("OBP"), errors="coerce")
    out["slg"] = pd.to_numeric(raw.get("SLG"), errors="coerce")
    out["ops"] = pd.to_numeric(raw.get("OPS"), errors="coerce")

    # Compute ISO from SLG - BA when available
    if "SLG" in raw.columns and "BA" in raw.columns:
        slg_vals = pd.to_numeric(raw.get("SLG"), errors="coerce")
        ba_vals = pd.to_numeric(raw.get("BA"), errors="coerce")
        out["iso"] = slg_vals - ba_vals

    # Compute K% and BB% from counting stats
    pa = pd.to_numeric(raw.get("PA"), errors="coerce")
    so = pd.to_numeric(raw.get("SO"), errors="coerce")
    bb = pd.to_numeric(raw.get("BB"), errors="coerce")
    out["k_pct"] = (so / pa).where(pa > 0)
    out["bb_pct"] = (bb / pa).where(pa > 0)

    # bref does not provide wOBA, wRC+, WAR, BABIP, xBA, xSLG, xwOBA,
    # HardHit%, Barrel% — leave as NaN
    for col in ["woba", "wrc_plus", "war", "babip", "hard_hit_pct",
                "barrel_pct", "xba", "xslg", "xwoba"]:
        out[col] = np.nan

    # Drop rows without a valid player ID
    out = out.dropna(subset=["player_id"]).copy()
    out["player_id"] = out["player_id"].astype(int)

    # Deduplicate on player_id (bref may have multiple rows for traded players)
    out = out.drop_duplicates(subset=["player_id"], keep="first")

    return out


def _load_pitching_stats_bref(season: int) -> pd.DataFrame:
    """Fallback: load pitching stats from Baseball Reference when FanGraphs is unavailable.

    Returns a DataFrame with columns mapped to our schema.
    """
    from pybaseball import pitching_stats_bref

    logger.info("Falling back to Baseball Reference pitching stats for %d", season)
    raw = pitching_stats_bref(season)
    if raw is None or raw.empty:
        return pd.DataFrame()

    # Filter to MLB level only
    if "Lev" in raw.columns:
        raw = raw[raw["Lev"].str.contains("Maj", na=False)].copy()

    out = pd.DataFrame()
    out["player_id"] = pd.to_numeric(raw.get("mlbID"), errors="coerce")
    out["season"] = season

    # Record / appearances
    out["w"] = pd.to_numeric(raw.get("W"), errors="coerce")
    out["l"] = pd.to_numeric(raw.get("L"), errors="coerce")
    out["sv"] = pd.to_numeric(raw.get("SV"), errors="coerce")
    out["g"] = pd.to_numeric(raw.get("G"), errors="coerce")
    out["gs"] = pd.to_numeric(raw.get("GS"), errors="coerce")
    out["ip"] = pd.to_numeric(raw.get("IP"), errors="coerce")

    # Run prevention
    out["era"] = pd.to_numeric(raw.get("ERA"), errors="coerce")
    out["whip"] = pd.to_numeric(raw.get("WHIP"), errors="coerce")

    # Peripheral rates from counting stats
    ip = pd.to_numeric(raw.get("IP"), errors="coerce")
    so = pd.to_numeric(raw.get("SO"), errors="coerce")
    bb = pd.to_numeric(raw.get("BB"), errors="coerce")
    hr = pd.to_numeric(raw.get("HR"), errors="coerce")
    bf = pd.to_numeric(raw.get("BF"), errors="coerce")

    out["k_per_9"] = (so * 9 / ip).where(ip > 0)
    out["bb_per_9"] = (bb * 9 / ip).where(ip > 0)
    out["hr_per_9"] = (hr * 9 / ip).where(ip > 0)
    out["k_pct"] = (so / bf).where(bf > 0)
    out["bb_pct"] = (bb / bf).where(bf > 0)

    # bref SO9 column if available
    if "SO9" in raw.columns:
        so9 = pd.to_numeric(raw.get("SO9"), errors="coerce")
        out["k_per_9"] = so9.where(so9.notna(), out["k_per_9"])

    # bref does not provide FIP, xFIP, SIERA, WAR, Stuff+, Location+,
    # Pitching+, avg_fastball_velo, avg_spin_rate, GB%, FB%, LD%,
    # xBA, xSLG, xwOBA — leave as NaN
    for col in ["fip", "xfip", "siera", "war", "stuff_plus", "location_plus",
                "pitching_plus", "avg_fastball_velo", "avg_spin_rate",
                "gb_pct", "fb_pct", "ld_pct", "xba", "xslg", "xwoba"]:
        out[col] = np.nan

    # Drop rows without a valid player ID
    out = out.dropna(subset=["player_id"]).copy()
    out["player_id"] = out["player_id"].astype(int)

    # Deduplicate on player_id
    out = out.drop_duplicates(subset=["player_id"], keep="first")

    return out


def load_season_batting_stats(
    season: int,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """Load season batting statistics from FanGraphs (or Baseball Reference as fallback).

    Fetches qualified hitters for *season* and maps columns to our
    ``season_batting_stats`` schema. FanGraphs IDs are converted to
    MLBAM IDs via the Chadwick register.

    Args:
        season: Four-digit year (e.g. ``2024``).
        conn:   Optional DuckDB connection for insertion.

    Returns:
        Cleaned DataFrame matching the ``season_batting_stats`` schema.
    """
    logger.info("Loading FanGraphs batting stats for %d", season)
    raw = None
    use_bref = False
    try:
        raw = _retry_network_call(pybaseball.batting_stats, season, season, qual=0)
    except Exception:
        logger.warning("FanGraphs batting stats unavailable for %d, trying Baseball Reference", season)
        use_bref = True

    if use_bref or raw is None or raw.empty:
        try:
            out = _load_batting_stats_bref(season)
        except Exception:
            logger.exception("Failed to fetch batting stats from both FanGraphs and Baseball Reference for %d", season)
            return pd.DataFrame()

        if out is None or out.empty:
            logger.warning("No batting stats returned for %d", season)
            return pd.DataFrame()

        # Keep only schema columns.
        schema_cols = [
            "player_id", "season", "pa", "ab", "h", "doubles", "triples", "hr",
            "rbi", "bb", "so", "sb", "cs", "ba", "obp", "slg", "ops", "woba",
            "wrc_plus", "war", "babip", "iso", "k_pct", "bb_pct", "hard_hit_pct",
            "barrel_pct", "xba", "xslg", "xwoba",
        ]
        out = out[[c for c in schema_cols if c in out.columns]]

        logger.info("Processed %d batting stat rows for %d (via Baseball Reference)", len(out), season)

        if conn is not None and not out.empty:
            insert_season_batting(conn, out)

        return out

    # --- FanGraphs path (original logic) ---
    # --- Build player ID mapping (FanGraphs IDfg -> MLBAM key_mlbam) ---
    fg_ids = raw["IDfg"].dropna().unique().tolist()
    id_map = _fg_to_mlbam_map(fg_ids, key_type="fangraphs")

    # --- Map columns ---
    out = pd.DataFrame()
    out["fg_id"] = raw["IDfg"]
    out["player_id"] = raw["IDfg"].map(id_map)
    out["season"] = season

    # Counting stats
    out["pa"] = pd.to_numeric(raw.get("PA"), errors="coerce")
    out["ab"] = pd.to_numeric(raw.get("AB"), errors="coerce")
    out["h"] = pd.to_numeric(raw.get("H"), errors="coerce")
    out["doubles"] = pd.to_numeric(raw.get("2B"), errors="coerce")
    out["triples"] = pd.to_numeric(raw.get("3B"), errors="coerce")
    out["hr"] = pd.to_numeric(raw.get("HR"), errors="coerce")
    out["rbi"] = pd.to_numeric(raw.get("RBI"), errors="coerce")
    out["bb"] = pd.to_numeric(raw.get("BB"), errors="coerce")
    out["so"] = pd.to_numeric(raw.get("SO"), errors="coerce")
    out["sb"] = pd.to_numeric(raw.get("SB"), errors="coerce")
    out["cs"] = pd.to_numeric(raw.get("CS"), errors="coerce")

    # Rate stats
    out["ba"] = pd.to_numeric(raw.get("AVG"), errors="coerce")
    out["obp"] = pd.to_numeric(raw.get("OBP"), errors="coerce")
    out["slg"] = pd.to_numeric(raw.get("SLG"), errors="coerce")
    out["ops"] = pd.to_numeric(raw.get("OPS"), errors="coerce")
    out["woba"] = pd.to_numeric(raw.get("wOBA"), errors="coerce")
    out["wrc_plus"] = pd.to_numeric(raw.get("wRC+"), errors="coerce")
    out["war"] = pd.to_numeric(raw.get("WAR"), errors="coerce")
    out["babip"] = pd.to_numeric(raw.get("BABIP"), errors="coerce")
    out["iso"] = pd.to_numeric(raw.get("ISO"), errors="coerce")

    # Percentage columns — FanGraphs returns these as floats (e.g., 25.3)
    out["k_pct"] = raw.get("K%").apply(_pct_to_float) if "K%" in raw.columns else np.nan
    out["bb_pct"] = raw.get("BB%").apply(_pct_to_float) if "BB%" in raw.columns else np.nan
    out["hard_hit_pct"] = raw.get("HardHit%").apply(_pct_to_float) if "HardHit%" in raw.columns else np.nan
    out["barrel_pct"] = raw.get("Barrel%").apply(_pct_to_float) if "Barrel%" in raw.columns else np.nan

    # Expected stats (may not be present in all FanGraphs outputs)
    out["xba"] = pd.to_numeric(raw.get("xBA"), errors="coerce") if "xBA" in raw.columns else np.nan
    out["xslg"] = pd.to_numeric(raw.get("xSLG"), errors="coerce") if "xSLG" in raw.columns else np.nan
    out["xwoba"] = pd.to_numeric(raw.get("xwOBA"), errors="coerce") if "xwOBA" in raw.columns else np.nan

    # Drop rows where we could not resolve the MLBAM player ID.
    missing_ids = out["player_id"].isna().sum()
    if missing_ids > 0:
        logger.warning(
            "%d of %d batters could not be mapped to MLBAM IDs (dropped)",
            missing_ids, len(out),
        )
    out = out.dropna(subset=["player_id"]).copy()
    out["player_id"] = out["player_id"].astype(int)

    # Keep only schema columns.
    schema_cols = [
        "player_id", "season", "pa", "ab", "h", "doubles", "triples", "hr",
        "rbi", "bb", "so", "sb", "cs", "ba", "obp", "slg", "ops", "woba",
        "wrc_plus", "war", "babip", "iso", "k_pct", "bb_pct", "hard_hit_pct",
        "barrel_pct", "xba", "xslg", "xwoba",
    ]
    out = out[[c for c in schema_cols if c in out.columns]]

    logger.info("Processed %d batting stat rows for %d", len(out), season)

    if conn is not None and not out.empty:
        insert_season_batting(conn, out)

    return out


def load_season_pitching_stats(
    season: int,
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """Load season pitching statistics from FanGraphs (or Baseball Reference as fallback).

    Args:
        season: Four-digit year (e.g. ``2024``).
        conn:   Optional DuckDB connection for insertion.

    Returns:
        Cleaned DataFrame matching the ``season_pitching_stats`` schema.
    """
    logger.info("Loading FanGraphs pitching stats for %d", season)
    raw = None
    use_bref = False
    try:
        raw = _retry_network_call(pybaseball.pitching_stats, season, season, qual=0)
    except Exception:
        logger.warning("FanGraphs pitching stats unavailable for %d, trying Baseball Reference", season)
        use_bref = True

    if use_bref or raw is None or raw.empty:
        try:
            out = _load_pitching_stats_bref(season)
        except Exception:
            logger.exception("Failed to fetch pitching stats from both FanGraphs and Baseball Reference for %d", season)
            return pd.DataFrame()

        if out is None or out.empty:
            logger.warning("No pitching stats returned for %d", season)
            return pd.DataFrame()

        schema_cols = [
            "player_id", "season", "w", "l", "sv", "g", "gs", "ip", "era", "fip",
            "xfip", "siera", "whip", "k_pct", "bb_pct", "hr_per_9", "k_per_9",
            "bb_per_9", "war", "stuff_plus", "location_plus", "pitching_plus",
            "avg_fastball_velo", "avg_spin_rate", "gb_pct", "fb_pct", "ld_pct",
            "xba", "xslg", "xwoba",
        ]
        out = out[[c for c in schema_cols if c in out.columns]]

        logger.info("Processed %d pitching stat rows for %d (via Baseball Reference)", len(out), season)

        if conn is not None and not out.empty:
            insert_season_pitching(conn, out)

        return out

    # --- FanGraphs path (original logic) ---

    # --- FG -> MLBAM ID mapping ---
    fg_ids = raw["IDfg"].dropna().unique().tolist()
    id_map = _fg_to_mlbam_map(fg_ids, key_type="fangraphs")

    out = pd.DataFrame()
    out["fg_id"] = raw["IDfg"]
    out["player_id"] = raw["IDfg"].map(id_map)
    out["season"] = season

    # Record / appearances
    out["w"] = pd.to_numeric(raw.get("W"), errors="coerce")
    out["l"] = pd.to_numeric(raw.get("L"), errors="coerce")
    out["sv"] = pd.to_numeric(raw.get("SV"), errors="coerce")
    out["g"] = pd.to_numeric(raw.get("G"), errors="coerce")
    out["gs"] = pd.to_numeric(raw.get("GS"), errors="coerce")
    out["ip"] = pd.to_numeric(raw.get("IP"), errors="coerce")

    # Run prevention
    out["era"] = pd.to_numeric(raw.get("ERA"), errors="coerce")
    out["fip"] = pd.to_numeric(raw.get("FIP"), errors="coerce")
    out["xfip"] = pd.to_numeric(raw.get("xFIP"), errors="coerce")
    out["siera"] = pd.to_numeric(raw.get("SIERA"), errors="coerce")
    out["whip"] = pd.to_numeric(raw.get("WHIP"), errors="coerce")

    # Peripheral rates (percentages)
    out["k_pct"] = raw.get("K%").apply(_pct_to_float) if "K%" in raw.columns else np.nan
    out["bb_pct"] = raw.get("BB%").apply(_pct_to_float) if "BB%" in raw.columns else np.nan
    out["hr_per_9"] = pd.to_numeric(raw.get("HR/9"), errors="coerce") if "HR/9" in raw.columns else np.nan
    out["k_per_9"] = pd.to_numeric(raw.get("K/9"), errors="coerce") if "K/9" in raw.columns else np.nan
    out["bb_per_9"] = pd.to_numeric(raw.get("BB/9"), errors="coerce") if "BB/9" in raw.columns else np.nan

    # Value / grades
    out["war"] = pd.to_numeric(raw.get("WAR"), errors="coerce")
    out["stuff_plus"] = pd.to_numeric(raw.get("Stuff+"), errors="coerce") if "Stuff+" in raw.columns else np.nan
    out["location_plus"] = pd.to_numeric(raw.get("Location+"), errors="coerce") if "Location+" in raw.columns else np.nan
    out["pitching_plus"] = pd.to_numeric(raw.get("Pitching+"), errors="coerce") if "Pitching+" in raw.columns else np.nan

    # Velocity / spin
    out["avg_fastball_velo"] = pd.to_numeric(raw.get("vFA (pi)"), errors="coerce") if "vFA (pi)" in raw.columns else np.nan
    # Fallback: pybaseball may expose the column as "FBv" or "vFA"
    if out["avg_fastball_velo"].isna().all():
        for alt in ("FBv", "vFA"):
            if alt in raw.columns:
                out["avg_fastball_velo"] = pd.to_numeric(raw.get(alt), errors="coerce")
                break

    # avg_spin_rate is not a standard FanGraphs leaderboard column; leave null.
    out["avg_spin_rate"] = np.nan

    # Batted-ball profile (percentages)
    out["gb_pct"] = raw.get("GB%").apply(_pct_to_float) if "GB%" in raw.columns else np.nan
    out["fb_pct"] = raw.get("FB%").apply(_pct_to_float) if "FB%" in raw.columns else np.nan
    out["ld_pct"] = raw.get("LD%").apply(_pct_to_float) if "LD%" in raw.columns else np.nan

    # Expected stats (may not be present)
    out["xba"] = pd.to_numeric(raw.get("xBA"), errors="coerce") if "xBA" in raw.columns else np.nan
    out["xslg"] = pd.to_numeric(raw.get("xSLG"), errors="coerce") if "xSLG" in raw.columns else np.nan
    out["xwoba"] = pd.to_numeric(raw.get("xwOBA"), errors="coerce") if "xwOBA" in raw.columns else np.nan

    # Drop unmappable rows
    missing_ids = out["player_id"].isna().sum()
    if missing_ids > 0:
        logger.warning(
            "%d of %d pitchers could not be mapped to MLBAM IDs (dropped)",
            missing_ids, len(out),
        )
    out = out.dropna(subset=["player_id"]).copy()
    out["player_id"] = out["player_id"].astype(int)

    # Keep only schema columns.
    schema_cols = [
        "player_id", "season", "w", "l", "sv", "g", "gs", "ip", "era", "fip",
        "xfip", "siera", "whip", "k_pct", "bb_pct", "hr_per_9", "k_per_9",
        "bb_per_9", "war", "stuff_plus", "location_plus", "pitching_plus",
        "avg_fastball_velo", "avg_spin_rate", "gb_pct", "fb_pct", "ld_pct",
        "xba", "xslg", "xwoba",
    ]
    out = out[[c for c in schema_cols if c in out.columns]]

    logger.info("Processed %d pitching stat rows for %d", len(out), season)

    if conn is not None and not out.empty:
        insert_season_pitching(conn, out)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Public API — Player ID crosswalk
# ═══════════════════════════════════════════════════════════════════════════


def load_player_id_map(
    conn: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """Load the Chadwick player ID register, filter to recent players, and
    return a DataFrame suitable for the ``players`` table.

    The Chadwick register includes columns such as ``key_mlbam``,
    ``key_fangraphs``, ``key_bbref``, ``name_first``, ``name_last``,
    ``mlb_played_last``, etc.

    Args:
        conn: Optional DuckDB connection. If given, inserts/replaces data
              in the ``players`` table.

    Returns:
        DataFrame with columns: ``player_id``, ``full_name``, ``fg_id``,
        ``bref_id``.
    """
    logger.info("Loading Chadwick player ID register")
    try:
        reg = _retry_network_call(pybaseball.chadwick_register)
    except Exception:
        logger.exception("Failed to load Chadwick register")
        return pd.DataFrame()

    if reg is None or reg.empty:
        return pd.DataFrame()

    # Filter to players who have appeared in an MLB game in a recent window.
    reg = reg.dropna(subset=["key_mlbam"])
    reg["key_mlbam"] = reg["key_mlbam"].astype(int)

    # Keep players active in the last ~10 years.
    if "mlb_played_last" in reg.columns:
        reg["mlb_played_last"] = pd.to_numeric(reg["mlb_played_last"], errors="coerce")
        reg = reg[reg["mlb_played_last"] >= 2015].copy()

    out = pd.DataFrame()
    out["player_id"] = reg["key_mlbam"]
    out["mlbam_id"] = reg["key_mlbam"]
    out["full_name"] = (
        reg["name_first"].fillna("").str.strip()
        + " "
        + reg["name_last"].fillna("").str.strip()
    ).str.strip()
    out["fg_id"] = reg.get("key_fangraphs").astype(str) if "key_fangraphs" in reg.columns else None
    out["bref_id"] = reg.get("key_bbref") if "key_bbref" in reg.columns else None

    # Deduplicate on player_id (keep first occurrence).
    out = out.drop_duplicates(subset=["player_id"], keep="first").reset_index(drop=True)

    logger.info("Loaded %d players from Chadwick register", len(out))

    if conn is not None and not out.empty:
        _upsert_players(conn, out)

    return out


# ═══════════════════════════════════════════════════════════════════════════
# Public API — DuckDB insertion helpers
# ═══════════════════════════════════════════════════════════════════════════


def insert_pitches(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """Insert a cleaned pitch DataFrame into the ``pitches`` table.

    Handles duplicates by filtering out rows whose ``(game_pk,
    at_bat_number, pitch_number)`` triple already exists in the table.
    Wrapped in a transaction so partial inserts are rolled back on failure.

    Args:
        conn: Open DuckDB connection.
        df:   Cleaned DataFrame (columns must match ``_PITCHES_COLUMNS``).

    Returns:
        Number of rows actually inserted.
    """
    if df.empty:
        return 0

    # Ensure the DataFrame columns are in the correct order.
    df = df[[c for c in _PITCHES_COLUMNS if c in df.columns]].copy()

    # Register the DataFrame as a temporary view for the query.
    conn.register("_stg_pitches", df)

    try:
        conn.execute("BEGIN TRANSACTION")

        # Check whether the pitches table already has data so we can deduplicate.
        try:
            existing_count = conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
        except duckdb.CatalogException:
            existing_count = 0

        if existing_count > 0:
            # Insert only rows whose natural key does not already exist.
            cols = ", ".join(c for c in _PITCHES_COLUMNS if c in df.columns)
            query = f"""
                INSERT INTO pitches ({cols})
                SELECT {cols}
                FROM _stg_pitches AS s
                WHERE NOT EXISTS (
                    SELECT 1 FROM pitches AS p
                    WHERE p.game_pk = s.game_pk
                      AND p.at_bat_number = s.at_bat_number
                      AND p.pitch_number = s.pitch_number
                )
            """
        else:
            cols = ", ".join(c for c in _PITCHES_COLUMNS if c in df.columns)
            query = f"INSERT INTO pitches ({cols}) SELECT {cols} FROM _stg_pitches"

        conn.execute(query)

        # Return count of inserted rows.
        new_count = conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
        inserted = new_count - existing_count

        conn.execute("COMMIT")

        logger.info("Inserted %d new pitch rows (total now: %d)", inserted, new_count)

        # Update data freshness after successful commit.
        try:
            max_date_row = conn.execute(
                "SELECT MAX(game_date) FROM pitches"
            ).fetchone()
            max_date = str(max_date_row[0]) if max_date_row and max_date_row[0] else None
            update_data_freshness(conn, "pitches", max_date, new_count)
        except Exception:
            logger.debug("Could not update freshness after pitch insert", exc_info=True)

        return inserted

    except Exception:
        logger.exception("Pitch insert failed — rolling back transaction")
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        try:
            conn.unregister("_stg_pitches")
        except Exception:
            pass


def insert_season_batting(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    """Insert season batting stats, replacing any existing data for the same
    ``(player_id, season)`` keys.

    Wrapped in a transaction for atomicity.

    Args:
        conn: Open DuckDB connection.
        df:   Cleaned batting stats DataFrame.
    """
    if df.empty:
        return

    seasons = df["season"].unique().tolist()
    player_ids = df["player_id"].unique().tolist()

    conn.register("_stg_batting", df)
    try:
        conn.execute("BEGIN TRANSACTION")

        # Delete existing rows for these seasons/players so we can replace.
        conn.execute(
            "DELETE FROM season_batting_stats WHERE season = ANY($1) AND player_id = ANY($2)",
            [seasons, player_ids],
        )

        cols = ", ".join(df.columns)
        conn.execute(f"INSERT INTO season_batting_stats ({cols}) SELECT {cols} FROM _stg_batting")

        conn.execute("COMMIT")
        logger.info("Upserted %d season batting rows", len(df))

        update_data_freshness(
            conn, "season_batting_stats",
            max_game_date=str(max(seasons)),
            row_count=conn.execute("SELECT COUNT(*) FROM season_batting_stats").fetchone()[0],
        )
    except Exception:
        logger.exception("Season batting insert failed — rolling back")
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        try:
            conn.unregister("_stg_batting")
        except Exception:
            pass


def insert_season_pitching(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    """Insert season pitching stats, replacing any existing data for the same
    ``(player_id, season)`` keys.

    Wrapped in a transaction for atomicity.

    Args:
        conn: Open DuckDB connection.
        df:   Cleaned pitching stats DataFrame.
    """
    if df.empty:
        return

    seasons = df["season"].unique().tolist()
    player_ids = df["player_id"].unique().tolist()

    conn.register("_stg_pitching", df)
    try:
        conn.execute("BEGIN TRANSACTION")

        conn.execute(
            "DELETE FROM season_pitching_stats WHERE season = ANY($1) AND player_id = ANY($2)",
            [seasons, player_ids],
        )

        cols = ", ".join(df.columns)
        conn.execute(f"INSERT INTO season_pitching_stats ({cols}) SELECT {cols} FROM _stg_pitching")

        conn.execute("COMMIT")
        logger.info("Upserted %d season pitching rows", len(df))

        update_data_freshness(
            conn, "season_pitching_stats",
            max_game_date=str(max(seasons)),
            row_count=conn.execute("SELECT COUNT(*) FROM season_pitching_stats").fetchone()[0],
        )
    except Exception:
        logger.exception("Season pitching insert failed — rolling back")
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        try:
            conn.unregister("_stg_pitching")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Private — ID mapping helpers
# ═══════════════════════════════════════════════════════════════════════════

# Module-level cache so we only load the Chadwick register once per process.
_CHADWICK_CACHE: Optional[pd.DataFrame] = None


def _load_chadwick_cached() -> pd.DataFrame:
    """Return the Chadwick register, cached in module memory."""
    global _CHADWICK_CACHE
    if _CHADWICK_CACHE is None:
        logger.info("Downloading Chadwick register (will be cached in-memory)")
        _CHADWICK_CACHE = pybaseball.chadwick_register()
    return _CHADWICK_CACHE


def _fg_to_mlbam_map(fg_ids: list, key_type: str = "fangraphs") -> dict:
    """Build a mapping from FanGraphs IDs to MLBAM IDs using the Chadwick register.

    Args:
        fg_ids:   List of FanGraphs ``IDfg`` values.
        key_type: Unused (reserved for future key types).

    Returns:
        Dict mapping ``IDfg`` -> ``key_mlbam`` (int).
    """
    reg = _load_chadwick_cached()
    if reg is None or reg.empty:
        logger.warning("Chadwick register unavailable; cannot map FG IDs to MLBAM")
        return {}

    # FanGraphs IDs live in key_fangraphs. They can be int or str depending on version.
    subset = reg.dropna(subset=["key_fangraphs", "key_mlbam"]).copy()
    subset["key_fangraphs_int"] = pd.to_numeric(subset["key_fangraphs"], errors="coerce")
    subset["key_mlbam"] = subset["key_mlbam"].astype(int)

    # Build both string and int-keyed lookups for robust matching.
    str_map = dict(zip(subset["key_fangraphs"].astype(str), subset["key_mlbam"]))
    int_map = dict(zip(subset["key_fangraphs_int"].dropna().astype(int), subset["key_mlbam"]))

    result: dict = {}
    for fid in fg_ids:
        if fid is None or (isinstance(fid, float) and np.isnan(fid)):
            continue
        # Try int match first, then string.
        try:
            int_fid = int(fid)
            if int_fid in int_map:
                result[fid] = int_map[int_fid]
                continue
        except (ValueError, TypeError):
            pass
        str_fid = str(fid).strip()
        if str_fid in str_map:
            result[fid] = str_map[str_fid]

    logger.info("Mapped %d / %d FanGraphs IDs to MLBAM IDs", len(result), len(fg_ids))
    return result


def _upsert_players(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> None:
    """Insert or replace rows in the ``players`` table.

    DuckDB does not support ``INSERT ... ON CONFLICT`` on all versions, so we
    delete-then-insert for the player IDs present in the DataFrame.
    """
    if df.empty:
        return

    player_ids = df["player_id"].unique().tolist()
    conn.execute("DELETE FROM players WHERE player_id = ANY($1)", [player_ids])

    # Only keep columns that exist in the players schema.
    valid_cols = ["player_id", "full_name", "team", "position", "throws", "bats",
                  "mlbam_id", "fg_id", "bref_id"]
    insert_cols = [c for c in valid_cols if c in df.columns]
    insert_df = df[insert_cols].copy()

    # Fill missing optional columns with NULL.
    for c in valid_cols:
        if c not in insert_df.columns:
            insert_df[c] = None

    insert_df = insert_df[valid_cols]

    conn.register("_stg_players", insert_df)
    cols_str = ", ".join(valid_cols)
    conn.execute(f"INSERT INTO players ({cols_str}) SELECT {cols_str} FROM _stg_players")
    conn.unregister("_stg_players")
    logger.info("Upserted %d players", len(insert_df))
