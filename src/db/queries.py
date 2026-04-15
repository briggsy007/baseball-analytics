"""
Reusable query functions for the baseball analytics platform.

Every public function accepts an open DuckDB connection as its first
argument and returns a ``pandas.DataFrame``.  Queries are parameterised
to prevent injection and enable plan caching.
"""

from datetime import date, timedelta
from typing import Optional

import duckdb
import pandas as pd


# ── Matchup & pitch-level queries ──────────────────────────────────────────


def get_matchup_history(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    batter_id: int,
) -> pd.DataFrame:
    """Return every pitch thrown between *pitcher_id* and *batter_id*.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID of the pitcher.
        batter_id: MLB player ID of the batter.

    Returns:
        DataFrame of all pitches ordered by game date and pitch number.
    """
    return conn.execute(
        """
        SELECT *
        FROM   pitches
        WHERE  pitcher_id = $1
          AND  batter_id  = $2
        ORDER  BY game_date, at_bat_number, pitch_number
        """,
        [pitcher_id, batter_id],
    ).fetchdf()


def get_pitcher_arsenal(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """Pitch-type breakdown with average velocity, spin rate, and movement.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID of the pitcher.
        season: If provided, restrict to pitches thrown in this season year.

    Returns:
        DataFrame with one row per pitch type.
    """
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    params: list = [pitcher_id]
    if season:
        params.append(season)

    return conn.execute(
        f"""
        SELECT pitch_type,
               pitch_name,
               COUNT(*)                        AS num_pitches,
               ROUND(AVG(release_speed), 1)    AS avg_velocity,
               ROUND(AVG(release_spin_rate), 0) AS avg_spin_rate,
               ROUND(AVG(pfx_x), 2)            AS avg_horz_break,
               ROUND(AVG(pfx_z), 2)            AS avg_vert_break,
               ROUND(
                   SUM(CASE WHEN description IN
                       ('swinging_strike', 'swinging_strike_blocked',
                        'foul_tip')
                       THEN 1 ELSE 0 END)
                   * 100.0 / COUNT(*), 1
               )                                AS whiff_pct
        FROM   pitches
        WHERE  pitcher_id = $1
               AND pitch_type IS NOT NULL
               {season_filter}
        GROUP  BY pitch_type, pitch_name
        ORDER  BY num_pitches DESC
        """,
        params,
    ).fetchdf()


def get_batter_zone_stats(
    conn: duckdb.DuckDBPyConnection,
    batter_id: int,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """Batting performance broken out by strike-zone region (zones 1-14).

    Args:
        conn: Open DuckDB connection.
        batter_id: MLB player ID of the batter.
        season: Optional season year filter.

    Returns:
        DataFrame with one row per zone.
    """
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    params: list = [batter_id]
    if season:
        params.append(season)

    return conn.execute(
        f"""
        SELECT zone,
               COUNT(*)                                         AS pitches_seen,
               SUM(CASE WHEN type = 'X' THEN 1 ELSE 0 END)    AS balls_in_play,
               ROUND(AVG(CASE WHEN type = 'X'
                              THEN launch_speed END), 1)        AS avg_exit_velo,
               ROUND(AVG(CASE WHEN type = 'X'
                              THEN launch_angle END), 1)        AS avg_launch_angle,
               ROUND(AVG(estimated_ba), 3)                      AS avg_xba,
               ROUND(AVG(estimated_woba), 3)                    AS avg_xwoba,
               ROUND(
                   SUM(CASE WHEN description IN
                       ('swinging_strike', 'swinging_strike_blocked',
                        'foul_tip')
                       THEN 1 ELSE 0 END)
                   * 100.0 / NULLIF(COUNT(*), 0), 1
               )                                                AS whiff_pct
        FROM   pitches
        WHERE  batter_id = $1
               AND zone IS NOT NULL
               {season_filter}
        GROUP  BY zone
        ORDER  BY zone
        """,
        params,
    ).fetchdf()


# ── Roster / player queries ───────────────────────────────────────────────


def get_team_roster(
    conn: duckdb.DuckDBPyConnection,
    team: str,
) -> pd.DataFrame:
    """Current roster for a given team abbreviation.

    Args:
        conn: Open DuckDB connection.
        team: Team abbreviation (e.g. ``'NYY'``).

    Returns:
        DataFrame of players on the team.
    """
    return conn.execute(
        """
        SELECT player_id, full_name, position, throws, bats
        FROM   players
        WHERE  team = $1
        ORDER  BY position, full_name
        """,
        [team],
    ).fetchdf()


# ── Game-log queries ──────────────────────────────────────────────────────


def get_pitcher_game_log(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    n_games: int = 10,
) -> pd.DataFrame:
    """Recent game-level pitching stats derived from the pitches table.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        n_games: Maximum number of distinct games to return.

    Returns:
        DataFrame with one row per game, most recent first.
    """
    return conn.execute(
        """
        WITH game_stats AS (
            SELECT game_pk,
                   game_date,
                   COUNT(*)                                          AS total_pitches,
                   COUNT(DISTINCT at_bat_number)                     AS batters_faced,
                   ROUND(AVG(release_speed), 1)                      AS avg_velo,
                   ROUND(
                       SUM(CASE WHEN description IN
                           ('swinging_strike', 'swinging_strike_blocked',
                            'foul_tip')
                           THEN 1 ELSE 0 END)
                       * 100.0 / COUNT(*), 1
                   )                                                 AS whiff_pct,
                   SUM(CASE WHEN events = 'strikeout' THEN 1
                            ELSE 0 END)                              AS strikeouts,
                   SUM(CASE WHEN events = 'walk' THEN 1
                            ELSE 0 END)                              AS walks,
                   SUM(CASE WHEN events = 'home_run' THEN 1
                            ELSE 0 END)                              AS home_runs,
                   ROUND(AVG(CASE WHEN type = 'X'
                                  THEN launch_speed END), 1)         AS avg_exit_velo_against
            FROM   pitches
            WHERE  pitcher_id = $1
            GROUP  BY game_pk, game_date
        )
        SELECT *
        FROM   game_stats
        ORDER  BY game_date DESC
        LIMIT  $2
        """,
        [pitcher_id, n_games],
    ).fetchdf()


def get_batter_game_log(
    conn: duckdb.DuckDBPyConnection,
    batter_id: int,
    n_games: int = 10,
) -> pd.DataFrame:
    """Recent game-level batting stats derived from the pitches table.

    Args:
        conn: Open DuckDB connection.
        batter_id: MLB player ID.
        n_games: Maximum number of distinct games to return.

    Returns:
        DataFrame with one row per game, most recent first.
    """
    return conn.execute(
        """
        WITH game_stats AS (
            SELECT game_pk,
                   game_date,
                   COUNT(DISTINCT at_bat_number)                     AS plate_appearances,
                   SUM(CASE WHEN type = 'X' THEN 1 ELSE 0 END)      AS balls_in_play,
                   ROUND(AVG(CASE WHEN type = 'X'
                                  THEN launch_speed END), 1)         AS avg_exit_velo,
                   ROUND(AVG(CASE WHEN type = 'X'
                                  THEN launch_angle END), 1)         AS avg_launch_angle,
                   SUM(CASE WHEN events = 'home_run' THEN 1
                            ELSE 0 END)                              AS home_runs,
                   SUM(CASE WHEN events = 'strikeout' THEN 1
                            ELSE 0 END)                              AS strikeouts,
                   SUM(CASE WHEN events = 'walk' THEN 1
                            ELSE 0 END)                              AS walks,
                   ROUND(AVG(estimated_woba), 3)                     AS avg_xwoba
            FROM   pitches
            WHERE  batter_id = $1
            GROUP  BY game_pk, game_date
        )
        SELECT *
        FROM   game_stats
        ORDER  BY game_date DESC
        LIMIT  $2
        """,
        [batter_id, n_games],
    ).fetchdf()


# ── Bullpen / workload ────────────────────────────────────────────────────


def get_bullpen_status(
    conn: duckdb.DuckDBPyConnection,
    team: str,
    lookback_days: int = 3,
) -> pd.DataFrame:
    """Bullpen workload summary for a team over a recent window.

    Returns each relief pitcher's pitch count, number of appearances,
    and days since their last outing within the lookback window.

    Args:
        conn: Open DuckDB connection.
        team: Team abbreviation (e.g. ``'NYY'``).
        lookback_days: Number of days to look back from today.

    Returns:
        DataFrame with one row per reliever.
    """
    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

    return conn.execute(
        """
        WITH relievers AS (
            SELECT p.pitcher_id,
                   pl.full_name,
                   p.game_date,
                   p.game_pk,
                   COUNT(*) AS pitches
            FROM   pitches p
            LEFT   JOIN players pl ON p.pitcher_id = pl.player_id
            WHERE  (p.home_team = $1 OR p.away_team = $1)
              AND  p.game_date >= CAST($2 AS DATE)
              AND  p.pitcher_id IN (
                       SELECT player_id
                       FROM   players
                       WHERE  team = $1
                         AND  position = 'RP'
                   )
            GROUP  BY p.pitcher_id, pl.full_name, p.game_date, p.game_pk
        )
        SELECT pitcher_id,
               full_name,
               SUM(pitches)                                 AS total_pitches,
               COUNT(DISTINCT game_pk)                      AS appearances,
               date_diff('day', MAX(game_date), CURRENT_DATE) AS days_since_last
        FROM   relievers
        GROUP  BY pitcher_id, full_name
        ORDER  BY total_pitches DESC
        """,
        [team, cutoff],
    ).fetchdf()


# ── Velocity trend ────────────────────────────────────────────────────────


def get_pitcher_velocity_trend(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    pitch_type: str = "FF",
    n_games: int = 10,
) -> pd.DataFrame:
    """Rolling per-game average velocity for a specific pitch type.

    Useful for detecting fatigue or injury-related velocity drops.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        pitch_type: Statcast pitch-type code (default ``'FF'`` = four-seam).
        n_games: Number of most-recent games to include.

    Returns:
        DataFrame ordered by game date (ascending) for easy plotting.
    """
    return conn.execute(
        """
        WITH game_velo AS (
            SELECT game_pk,
                   game_date,
                   COUNT(*)                       AS num_pitches,
                   ROUND(AVG(release_speed), 2)   AS avg_velocity,
                   ROUND(MIN(release_speed), 2)   AS min_velocity,
                   ROUND(MAX(release_speed), 2)   AS max_velocity,
                   ROUND(AVG(release_spin_rate), 0) AS avg_spin
            FROM   pitches
            WHERE  pitcher_id = $1
              AND  pitch_type = $2
            GROUP  BY game_pk, game_date
            ORDER  BY game_date DESC
            LIMIT  $3
        )
        SELECT *
        FROM   game_velo
        ORDER  BY game_date ASC
        """,
        [pitcher_id, pitch_type, n_games],
    ).fetchdf()


# ── Schedule ──────────────────────────────────────────────────────────────


def get_today_games(
    conn: duckdb.DuckDBPyConnection,
    date_val: Optional[date] = None,
) -> pd.DataFrame:
    """Return all games scheduled for a given date.

    Args:
        conn: Open DuckDB connection.
        date_val: Target date. Defaults to ``date.today()``.

    Returns:
        DataFrame with one row per game.
    """
    target = (date_val or date.today()).isoformat()

    return conn.execute(
        """
        SELECT game_pk, game_date, home_team, away_team,
               venue, game_type, status
        FROM   games
        WHERE  game_date = CAST($1 AS DATE)
        ORDER  BY game_pk
        """,
        [target],
    ).fetchdf()


# ── Cache refresh ─────────────────────────────────────────────────────────


def refresh_matchup_cache(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Rebuild the ``matchup_summary`` table with aggregated pitcher-vs-batter stats.

    The table is fully replaced (DELETE + INSERT) so downstream consumers
    always see a consistent snapshot.

    Args:
        conn: Open DuckDB connection.

    Returns:
        DataFrame of the newly-materialised matchup summary rows.
    """
    conn.execute("DELETE FROM matchup_summary")
    conn.execute("""
        INSERT INTO matchup_summary
        SELECT pitcher_id,
               batter_id,
               pitch_type,
               COUNT(*)                                          AS num_pitches,
               ROUND(AVG(release_speed), 1)                      AS avg_speed,
               ROUND(AVG(release_spin_rate), 0)                  AS avg_spin,
               ROUND(AVG(pfx_x), 2)                              AS avg_pfx_x,
               ROUND(AVG(pfx_z), 2)                              AS avg_pfx_z,
               ROUND(
                   SUM(CASE WHEN description IN
                       ('swinging_strike', 'swinging_strike_blocked',
                        'foul_tip')
                       THEN 1 ELSE 0 END)
                   * 100.0 / NULLIF(COUNT(*), 0), 3
               )                                                 AS whiff_rate,
               ROUND(
                   SUM(CASE WHEN events IN
                       ('single','double','triple','home_run')
                       THEN 1 ELSE 0 END)
                   * 1.0 / NULLIF(
                       SUM(CASE WHEN events IS NOT NULL
                                 AND events != 'walk'
                                 AND events != 'hit_by_pitch'
                            THEN 1 ELSE 0 END), 0), 3
               )                                                 AS ba,
               ROUND(
                   (SUM(CASE WHEN events = 'single' THEN 1
                             WHEN events = 'double' THEN 2
                             WHEN events = 'triple' THEN 3
                             WHEN events = 'home_run' THEN 4
                             ELSE 0 END)
                   * 1.0 / NULLIF(
                       SUM(CASE WHEN events IS NOT NULL
                                 AND events != 'walk'
                                 AND events != 'hit_by_pitch'
                            THEN 1 ELSE 0 END), 0)), 3
               )                                                 AS slg,
               ROUND(
                   SUM(woba_value)
                   / NULLIF(SUM(woba_denom), 0), 3
               )                                                 AS woba
        FROM   pitches
        WHERE  pitch_type IS NOT NULL
        GROUP  BY pitcher_id, batter_id, pitch_type
    """)

    return conn.execute("SELECT * FROM matchup_summary").fetchdf()
