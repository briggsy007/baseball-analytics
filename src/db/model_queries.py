"""
Parameterised query functions for model training data.

Every function accepts an open DuckDB connection and returns a pandas
DataFrame.  Queries use ``$1``, ``$2`` … parameter placeholders to
prevent SQL injection and allow DuckDB's query planner to cache plans.
"""

from __future__ import annotations

from typing import Optional

import duckdb
import pandas as pd


def get_pitcher_game_pitches(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """Return every pitch thrown by a pitcher, optionally filtered by season.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        season: Restrict to this year (optional).

    Returns:
        DataFrame of pitch-level records ordered by game date and
        at-bat / pitch number.
    """
    if season is not None:
        query = """
            SELECT *
            FROM pitches
            WHERE pitcher_id = $1
              AND EXTRACT(YEAR FROM game_date) = $2
            ORDER BY game_date, at_bat_number, pitch_number
        """
        return conn.execute(query, [pitcher_id, season]).fetchdf()

    query = """
        SELECT *
        FROM pitches
        WHERE pitcher_id = $1
        ORDER BY game_date, at_bat_number, pitch_number
    """
    return conn.execute(query, [pitcher_id]).fetchdf()


def get_game_pitch_sequence(
    conn: duckdb.DuckDBPyConnection,
    game_pk: int,
) -> pd.DataFrame:
    """Return the full pitch sequence for a game, ordered chronologically.

    Args:
        conn: Open DuckDB connection.
        game_pk: Unique game identifier.

    Returns:
        DataFrame with every pitch in the game in thrown order.
    """
    query = """
        SELECT *
        FROM pitches
        WHERE game_pk = $1
        ORDER BY at_bat_number, pitch_number
    """
    return conn.execute(query, [game_pk]).fetchdf()


def get_batter_game_log(
    conn: duckdb.DuckDBPyConnection,
    batter_id: int,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """Return game-level aggregate statistics for a batter.

    Aggregates pitch-level data into per-game summaries including
    plate appearances, hits (approximated from events), and average
    exit velocity.

    Args:
        conn: Open DuckDB connection.
        batter_id: MLB player ID.
        season: Restrict to this year (optional).

    Returns:
        DataFrame with one row per game.
    """
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    params = [batter_id, season] if season else [batter_id]

    query = f"""
        SELECT
            game_pk,
            game_date,
            COUNT(*) AS pitches_seen,
            COUNT(DISTINCT at_bat_number) AS plate_appearances,
            SUM(CASE WHEN events IN ('single', 'double', 'triple', 'home_run')
                     THEN 1 ELSE 0 END) AS hits,
            SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS home_runs,
            SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END) AS strikeouts,
            SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS walks,
            AVG(launch_speed) AS avg_exit_velo,
            AVG(launch_angle) AS avg_launch_angle
        FROM pitches
        WHERE batter_id = $1
          {season_filter}
        GROUP BY game_pk, game_date
        ORDER BY game_date
    """
    return conn.execute(query, params).fetchdf()


def get_pitcher_appearances(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    days_back: Optional[int] = None,
) -> pd.DataFrame:
    """Return an appearance log for a pitcher with pitch counts.

    Each row is one game appearance with aggregate pitch-level stats.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        days_back: Only include appearances within this many days of
                   the most recent game (optional).

    Returns:
        DataFrame with one row per appearance.
    """
    days_filter = ""
    params = [pitcher_id]
    if days_back is not None:
        days_filter = """
            AND game_date >= (
                SELECT MAX(game_date) - INTERVAL ($2) DAY
                FROM pitches
                WHERE pitcher_id = $1
            )
        """
        params.append(days_back)

    query = f"""
        SELECT
            game_pk,
            game_date,
            COUNT(*) AS pitch_count,
            COUNT(DISTINCT at_bat_number) AS batters_faced,
            AVG(release_speed) AS avg_velo,
            SUM(CASE WHEN description IN ('swinging_strike', 'swinging_strike_blocked',
                                          'foul_tip', 'called_strike')
                     THEN 1 ELSE 0 END) AS total_strikes,
            SUM(CASE WHEN description IN ('swinging_strike', 'swinging_strike_blocked',
                                          'foul_tip')
                     THEN 1 ELSE 0 END) AS whiffs,
            MIN(inning) AS first_inning,
            MAX(inning) AS last_inning
        FROM pitches
        WHERE pitcher_id = $1
          {days_filter}
        GROUP BY game_pk, game_date
        ORDER BY game_date DESC
    """
    return conn.execute(query, params).fetchdf()


def get_team_lineup_games(
    conn: duckdb.DuckDBPyConnection,
    team_id: str,
    season: int,
) -> pd.DataFrame:
    """Return per-game lineup data and run outcomes for a team.

    Identifies the team's batters per game and aggregates offensive
    output.

    Args:
        conn: Open DuckDB connection.
        team_id: Three-letter team abbreviation (e.g. ``'PHI'``).
        season: Season year.

    Returns:
        DataFrame with one row per game.
    """
    query = """
        SELECT
            game_pk,
            game_date,
            COUNT(DISTINCT batter_id) AS unique_batters,
            COUNT(*) AS total_pitches,
            SUM(CASE WHEN events IN ('single', 'double', 'triple', 'home_run')
                     THEN 1 ELSE 0 END) AS hits,
            SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS home_runs,
            SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS walks,
            SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END) AS strikeouts,
            SUM(COALESCE(woba_value, 0)) AS total_woba_value,
            SUM(COALESCE(woba_denom, 0)) AS total_woba_denom
        FROM pitches
        WHERE (
            (inning_topbot = 'Top' AND away_team = $1)
            OR (inning_topbot = 'Bot' AND home_team = $1)
        )
        AND EXTRACT(YEAR FROM game_date) = $2
        GROUP BY game_pk, game_date
        ORDER BY game_date
    """
    return conn.execute(query, [team_id, season]).fetchdf()


def get_pitch_pairs(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
) -> pd.DataFrame:
    """Return consecutive pitch pairs for sequence analysis.

    Each row contains the current pitch and the previous pitch's key
    attributes, enabling tunnel and sequencing models to train on
    pitch-to-pitch transitions.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.

    Returns:
        DataFrame with current and previous pitch attributes.
    """
    query = """
        WITH ordered AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY game_pk, at_bat_number
                    ORDER BY pitch_number
                ) AS rn
            FROM pitches
            WHERE pitcher_id = $1
        )
        SELECT
            curr.game_pk,
            curr.game_date,
            curr.at_bat_number,
            curr.pitch_number,
            curr.pitch_type,
            curr.release_speed,
            curr.release_spin_rate,
            curr.pfx_x,
            curr.pfx_z,
            curr.plate_x,
            curr.plate_z,
            curr.description,
            prev.pitch_type   AS prev_pitch_type,
            prev.release_speed AS prev_release_speed,
            prev.release_spin_rate AS prev_release_spin_rate,
            prev.pfx_x        AS prev_pfx_x,
            prev.pfx_z        AS prev_pfx_z,
            prev.plate_x      AS prev_plate_x,
            prev.plate_z      AS prev_plate_z,
            prev.description   AS prev_description
        FROM ordered curr
        INNER JOIN ordered prev
            ON curr.game_pk = prev.game_pk
            AND curr.at_bat_number = prev.at_bat_number
            AND curr.rn = prev.rn + 1
        ORDER BY curr.game_date, curr.at_bat_number, curr.pitch_number
    """
    return conn.execute(query, [pitcher_id]).fetchdf()
