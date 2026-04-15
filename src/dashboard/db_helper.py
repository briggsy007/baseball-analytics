"""Database connection helper for the Streamlit dashboard.

Provides a cached DuckDB connection, data-availability checks, and
convenience lookup functions used by every dashboard page to decide
whether to query real analytics or fall back to mock data.
"""
from __future__ import annotations

from typing import Optional

import streamlit as st
from pathlib import Path


@st.cache_resource
def _create_db_connection():
    """Create a fresh DuckDB connection (cached by Streamlit)."""
    import logging
    logger = logging.getLogger(__name__)
    try:
        from src.db.schema import get_connection
        conn = get_connection(read_only=True)
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()
        table_names = [t[0] for t in tables]
        if "pitches" not in table_names:
            logger.warning("DB connected but 'pitches' table not found. Tables: %s", table_names)
            return None
        logger.info("DB connection established. Tables: %s", table_names)
        return conn
    except Exception as exc:
        logger.error("Failed to connect to database: %s", exc)
        return None


def get_db_connection():
    """Get a DuckDB connection with automatic reconnect on internal errors.

    Wraps the cached connection with a health check. If the connection
    is in a bad state (DuckDB InternalException) or was cached as None
    due to an earlier transient failure, clears the cache and creates a
    fresh connection.
    """
    conn = _create_db_connection()
    if conn is None:
        # A previous call may have cached None due to a transient error.
        # Clear the cache and retry once so the dashboard recovers
        # automatically instead of staying broken until a manual restart.
        _create_db_connection.clear()
        conn = _create_db_connection()
        if conn is None:
            return None
    try:
        # Health check — lightweight query to verify connection is alive
        conn.execute("SELECT 1").fetchone()
        return conn
    except Exception:
        # Connection is corrupted — clear cache and reconnect
        _create_db_connection.clear()
        return _create_db_connection()


def has_data(conn) -> bool:
    """Check if the database has any pitch data loaded."""
    if conn is None:
        return False
    try:
        count = conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
        return count > 0
    except Exception:
        return False


def get_player_id_by_name(conn, name: str) -> int | None:
    """Look up a player's MLBAM ID by name (case-insensitive substring match)."""
    if conn is None:
        return None
    try:
        result = conn.execute(
            "SELECT player_id FROM players WHERE full_name ILIKE $1 LIMIT 1",
            [f"%{name}%"],
        ).fetchone()
        return result[0] if result else None
    except Exception:
        return None


def get_all_pitchers(conn) -> list[dict]:
    """Get all pitchers in the database for dropdown selectors."""
    if conn is None:
        return []
    try:
        df = conn.execute("""
            SELECT DISTINCT p.player_id, p.full_name, p.team, p.throws
            FROM players p
            WHERE p.player_id IN (SELECT DISTINCT pitcher_id FROM pitches)
            ORDER BY p.full_name
        """).fetchdf()
        return df.to_dict("records")
    except Exception:
        return []


def get_all_batters(conn) -> list[dict]:
    """Get all batters in the database for dropdown selectors."""
    if conn is None:
        return []
    try:
        df = conn.execute("""
            SELECT DISTINCT p.player_id, p.full_name, p.team, p.bats
            FROM players p
            WHERE p.player_id IN (SELECT DISTINCT batter_id FROM pitches)
            ORDER BY p.full_name
        """).fetchdf()
        return df.to_dict("records")
    except Exception:
        return []


def get_data_status(conn) -> dict | None:
    """Return a summary of the data loaded in the database.

    Returns:
        Dict with ``pitch_count``, ``min_date``, ``max_date``, or
        ``None`` if no data is available.
    """
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS cnt, MIN(game_date) AS min_dt, MAX(game_date) AS max_dt FROM pitches"
        ).fetchone()
        if row is None or row[0] == 0:
            return None
        return {
            "pitch_count": row[0],
            "min_date": row[1],
            "max_date": row[2],
        }
    except Exception:
        return None
