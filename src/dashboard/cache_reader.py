"""Cache reader for pre-computed analytics results.

Provides thin helpers that dashboard pages use to pull leaderboard
DataFrames and individual entity results from the DuckDB cache tables
populated by ``scripts/precompute.py``.  Every function returns ``None``
when no valid (fresh-enough) cache entry exists, so callers can fall
back to on-the-fly computation.
"""
from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def get_cached_leaderboard(
    conn,
    model_name: str,
    season: int | None = None,
    max_age_hours: int = 24,
) -> pd.DataFrame | None:
    """Read a pre-computed leaderboard from leaderboard_cache.

    Returns ``None`` if no valid cache exists or the cached entry is
    older than *max_age_hours*.  When *season* is ``None``, returns
    the most recent cache entry for this model regardless of season.
    """
    try:
        if season is not None:
            row = conn.execute(
                """
                SELECT leaderboard_parquet, computed_at
                FROM leaderboard_cache
                WHERE model_name = $1 AND season = $2
                """,
                [model_name, season],
            ).fetchone()
        else:
            # No season specified — get the most recent cache entry
            row = conn.execute(
                """
                SELECT leaderboard_parquet, computed_at
                FROM leaderboard_cache
                WHERE model_name = $1
                ORDER BY computed_at DESC
                LIMIT 1
                """,
                [model_name],
            ).fetchone()

        if row is None:
            return None

        parquet_bytes, computed_at = row

        # Check age
        if computed_at.tzinfo is None:
            computed_at = computed_at.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - computed_at).total_seconds() / 3600

        if age_hours > max_age_hours:
            logger.info(
                "Cache for %s is %.1f hours old (max %d), skipping.",
                model_name,
                age_hours,
                max_age_hours,
            )
            return None

        df = pd.read_parquet(io.BytesIO(parquet_bytes))
        logger.info(
            "Cache hit for %s: %d rows, %.1f hours old.",
            model_name,
            len(df),
            age_hours,
        )
        return df

    except Exception as exc:
        logger.debug("Cache read failed for %s: %s", model_name, exc)
        return None


def get_cached_entity(
    conn,
    model_name: str,
    entity_id: int,
    season: int | None = None,
    max_age_hours: int = 24,
) -> dict | None:
    """Read a single entity's cached result from model_cache."""
    try:
        if season is not None:
            row = conn.execute(
                """
                SELECT result_json, computed_at
                FROM model_cache
                WHERE model_name = $1 AND entity_id = $2 AND season = $3
                """,
                [model_name, entity_id, season],
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT result_json, computed_at
                FROM model_cache
                WHERE model_name = $1 AND entity_id = $2
                ORDER BY computed_at DESC
                LIMIT 1
                """,
                [model_name, entity_id],
            ).fetchone()

        if row is None:
            return None

        result_json, computed_at = row

        if computed_at.tzinfo is None:
            computed_at = computed_at.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - computed_at).total_seconds() / 3600

        if age_hours > max_age_hours:
            return None

        return json.loads(result_json)

    except Exception as exc:
        logger.debug(
            "Entity cache read failed for %s/%d: %s", model_name, entity_id, exc
        )
        return None


def cache_age_display(conn, model_name: str, season: int | None = None) -> str | None:
    """Return a human-readable string of when the cache was last computed."""
    try:
        if season is not None:
            row = conn.execute(
                """
                SELECT computed_at, row_count
                FROM leaderboard_cache
                WHERE model_name = $1 AND season = $2
                """,
                [model_name, season],
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT computed_at, row_count
                FROM leaderboard_cache
                WHERE model_name = $1
                ORDER BY computed_at DESC
                LIMIT 1
                """,
                [model_name],
            ).fetchone()

        if row is None:
            return None

        computed_at, row_count = row

        if computed_at.tzinfo is None:
            computed_at = computed_at.replace(tzinfo=timezone.utc)

        age = datetime.now(timezone.utc) - computed_at
        hours = age.total_seconds() / 3600

        if hours < 1:
            age_str = f"{int(age.total_seconds() / 60)} min ago"
        elif hours < 24:
            age_str = f"{hours:.1f} hours ago"
        else:
            age_str = f"{hours / 24:.1f} days ago"

        return f"Pre-computed {age_str} ({row_count} entries)"

    except Exception:
        return None
