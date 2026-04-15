"""
Database schema definition and initialization for the baseball analytics platform.

Manages the DuckDB database lifecycle: connection handling, table creation,
and schema migrations. All tables use CREATE TABLE IF NOT EXISTS for
idempotent initialization.
"""

from pathlib import Path
from typing import Optional

import duckdb

# Default database path (absolute, Windows-safe via pathlib)
DEFAULT_DB_PATH: Path = Path(r"C:\Users\hunte\projects\baseball\data\baseball.duckdb")


def get_connection(db_path: Optional[str] = None, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection, creating the parent directory if needed.

    Args:
        db_path: Filesystem path for the database file.
                 Defaults to ``data/baseball.duckdb`` inside the project root.
        read_only: If True, open in read-only mode (allows concurrent readers).

    Returns:
        An open ``duckdb.DuckDBPyConnection``.
    """
    resolved: Path = Path(db_path) if db_path else DEFAULT_DB_PATH
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(resolved), read_only=read_only)


def create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    """Create every table in the schema if it does not already exist.

    Args:
        conn: An open DuckDB connection.
    """
    _create_pitches(conn)
    _create_players(conn)
    _create_games(conn)
    _create_season_batting_stats(conn)
    _create_season_pitching_stats(conn)
    _create_transactions(conn)
    _create_matchup_summary(conn)
    _create_model_cache(conn)
    _create_leaderboard_cache(conn)
    _create_data_freshness(conn)


def init_db(db_path: Optional[str] = None) -> duckdb.DuckDBPyConnection:
    """Convenience entry-point: open a connection and ensure all tables exist.

    Args:
        db_path: Optional override for the database file location.

    Returns:
        The open connection (caller is responsible for closing).
    """
    conn = get_connection(db_path)
    create_tables(conn)
    return conn


# ── Private helpers ─────────────────────────────────────────────────────────


def _create_pitches(conn: duckdb.DuckDBPyConnection) -> None:
    """Core pitch-by-pitch Statcast table (~750 K rows per season)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pitches (
            game_pk              INTEGER,
            game_date            DATE,
            pitcher_id           INTEGER,
            batter_id            INTEGER,

            -- Pitch identification
            pitch_type           VARCHAR,
            pitch_name           VARCHAR,

            -- Release characteristics
            release_speed        FLOAT,
            release_spin_rate    FLOAT,
            spin_axis            FLOAT,

            -- Movement
            pfx_x                FLOAT,
            pfx_z                FLOAT,

            -- Location at the plate
            plate_x              FLOAT,
            plate_z              FLOAT,

            -- Release position
            release_extension    FLOAT,
            release_pos_x        FLOAT,
            release_pos_y        FLOAT,
            release_pos_z        FLOAT,

            -- Batted-ball outcomes
            launch_speed         FLOAT,
            launch_angle         FLOAT,
            hit_distance         FLOAT,
            hc_x                 FLOAT,
            hc_y                 FLOAT,
            bb_type              VARCHAR,

            -- Expected stats
            estimated_ba         FLOAT,
            estimated_woba       FLOAT,

            -- Win / run expectancy
            delta_home_win_exp   FLOAT,
            delta_run_exp        FLOAT,

            -- Game situation
            inning               INTEGER,
            inning_topbot        VARCHAR,
            outs_when_up         INTEGER,
            balls                INTEGER,
            strikes              INTEGER,
            on_1b                INTEGER,
            on_2b                INTEGER,
            on_3b                INTEGER,

            -- Batter / pitcher handedness
            stand                VARCHAR,
            p_throws             VARCHAR,

            -- Sequencing
            at_bat_number        INTEGER,
            pitch_number         INTEGER,

            -- Outcome descriptions
            description          VARCHAR,
            events               VARCHAR,
            type                 VARCHAR,

            -- Teams
            home_team            VARCHAR,
            away_team            VARCHAR,

            -- Weighting / value columns
            woba_value           FLOAT,
            woba_denom           FLOAT,
            babip_value          FLOAT,
            iso_value            FLOAT,

            -- Zone / speed
            zone                 INTEGER,
            effective_speed      FLOAT,

            -- Fielding alignment
            if_fielding_alignment VARCHAR,
            of_fielding_alignment VARCHAR,

            -- Catcher (fielder_2 from Statcast)
            fielder_2            INTEGER
        );
    """)


def _create_players(conn: duckdb.DuckDBPyConnection) -> None:
    """Player dimension table with cross-reference IDs."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS players (
            player_id   INTEGER PRIMARY KEY,
            full_name   VARCHAR,
            team        VARCHAR,
            position    VARCHAR,
            throws      VARCHAR,
            bats        VARCHAR,
            mlbam_id    INTEGER,
            fg_id       VARCHAR,
            bref_id     VARCHAR
        );
    """)


def _create_games(conn: duckdb.DuckDBPyConnection) -> None:
    """Game dimension table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_pk     INTEGER PRIMARY KEY,
            game_date   DATE,
            home_team   VARCHAR,
            away_team   VARCHAR,
            venue       VARCHAR,
            game_type   VARCHAR,
            status      VARCHAR
        );
    """)


def _create_season_batting_stats(conn: duckdb.DuckDBPyConnection) -> None:
    """Season-level batting statistics."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS season_batting_stats (
            player_id    INTEGER,
            season       INTEGER,

            -- Counting stats
            pa           INTEGER,
            ab           INTEGER,
            h            INTEGER,
            doubles      INTEGER,
            triples      INTEGER,
            hr           INTEGER,
            rbi          INTEGER,
            bb           INTEGER,
            so           INTEGER,
            sb           INTEGER,
            cs           INTEGER,

            -- Rate stats
            ba           FLOAT,
            obp          FLOAT,
            slg          FLOAT,
            ops          FLOAT,
            woba         FLOAT,
            wrc_plus     FLOAT,
            war          FLOAT,
            babip        FLOAT,
            iso          FLOAT,

            -- Percentages
            k_pct        FLOAT,
            bb_pct       FLOAT,
            hard_hit_pct FLOAT,
            barrel_pct   FLOAT,

            -- Expected stats
            xba          FLOAT,
            xslg         FLOAT,
            xwoba        FLOAT,

            PRIMARY KEY (player_id, season)
        );
    """)


def _create_season_pitching_stats(conn: duckdb.DuckDBPyConnection) -> None:
    """Season-level pitching statistics."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS season_pitching_stats (
            player_id        INTEGER,
            season           INTEGER,

            -- Record / appearances
            w                INTEGER,
            l                INTEGER,
            sv               INTEGER,
            g                INTEGER,
            gs               INTEGER,
            ip               FLOAT,

            -- Run prevention
            era              FLOAT,
            fip              FLOAT,
            xfip             FLOAT,
            siera            FLOAT,
            whip             FLOAT,

            -- Peripheral rates
            k_pct            FLOAT,
            bb_pct           FLOAT,
            hr_per_9         FLOAT,
            k_per_9          FLOAT,
            bb_per_9         FLOAT,

            -- Value / grades
            war              FLOAT,
            stuff_plus       FLOAT,
            location_plus    FLOAT,
            pitching_plus    FLOAT,

            -- Velocity / spin
            avg_fastball_velo FLOAT,
            avg_spin_rate    FLOAT,

            -- Batted-ball profile
            gb_pct           FLOAT,
            fb_pct           FLOAT,
            ld_pct           FLOAT,

            -- Expected stats (against)
            xba              FLOAT,
            xslg             FLOAT,
            xwoba            FLOAT,

            PRIMARY KEY (player_id, season)
        );
    """)


def _create_transactions(conn: duckdb.DuckDBPyConnection) -> None:
    """Roster transactions (trades, DFA, call-ups, etc.)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id   INTEGER PRIMARY KEY,
            player_id        INTEGER,
            player_name      VARCHAR,
            team             VARCHAR,
            from_team        VARCHAR,
            to_team          VARCHAR,
            transaction_type VARCHAR,
            description      VARCHAR,
            transaction_date DATE
        );
    """)


def _create_matchup_summary(conn: duckdb.DuckDBPyConnection) -> None:
    """Materialized aggregate used by ``refresh_matchup_cache``."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matchup_summary (
            pitcher_id    INTEGER,
            batter_id     INTEGER,
            pitch_type    VARCHAR,
            num_pitches   INTEGER,
            avg_speed     FLOAT,
            avg_spin      FLOAT,
            avg_pfx_x     FLOAT,
            avg_pfx_z     FLOAT,
            whiff_rate    FLOAT,
            ba            FLOAT,
            slg           FLOAT,
            woba          FLOAT,
            PRIMARY KEY (pitcher_id, batter_id, pitch_type)
        );
    """)


def _create_model_cache(conn: duckdb.DuckDBPyConnection) -> None:
    """Per-entity cache for individual model results (JSON)."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS model_cache (
            model_name      VARCHAR NOT NULL,
            entity_type     VARCHAR NOT NULL,
            entity_id       INTEGER NOT NULL,
            season          INTEGER NOT NULL,
            result_json     VARCHAR NOT NULL,
            computed_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            model_version   VARCHAR NOT NULL DEFAULT '1.0',
            compute_seconds FLOAT,
            PRIMARY KEY (model_name, entity_id, season)
        );
    """)


def _create_leaderboard_cache(conn: duckdb.DuckDBPyConnection) -> None:
    """Whole-leaderboard cache stored as Parquet bytes."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS leaderboard_cache (
            model_name      VARCHAR NOT NULL,
            season          INTEGER NOT NULL,
            leaderboard_parquet BLOB NOT NULL,
            row_count       INTEGER NOT NULL,
            computed_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            model_version   VARCHAR NOT NULL DEFAULT '1.0',
            compute_seconds FLOAT,
            PRIMARY KEY (model_name, season)
        );
    """)


def _create_data_freshness(conn: duckdb.DuckDBPyConnection) -> None:
    """Track last-known freshness of key source tables."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS data_freshness (
            table_name      VARCHAR PRIMARY KEY,
            max_game_date   DATE,
            row_count       BIGINT,
            updated_at      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)
