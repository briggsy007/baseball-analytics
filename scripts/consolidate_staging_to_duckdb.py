"""
Consolidate 7 staging parquets into DuckDB in a single write pass.

Appends the statcast gap-fill and 2010-2014 WAR parquets to existing tables,
then creates four NEW integration tables (umpire_assignments,
umpire_tendencies, game_weather, tj_surgery_dates) and loads them.

Idempotent: all inserts are anti-joined on natural keys so re-running does
not double-insert. Safe schema drift handling for 2010-2014 rows (only
7 of the 29/30 season_*_stats columns are available from bWAR).

Usage
-----
    python scripts/consolidate_staging_to_duckdb.py

Preconditions
-------------
- Streamlit dashboard STOPPED (DuckDB is single-writer).
- All 7 staging parquets present under ``data/staging/``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGING = PROJECT_ROOT / "data" / "staging"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.db.schema import create_tables, get_connection  # noqa: E402


# ── Helpers ────────────────────────────────────────────────────────────────


@dataclass
class LoadResult:
    table: str
    pre_count: int
    post_count: int
    expected_staged: int
    staged_rows: int
    inserted: int
    duplicates_skipped: int
    notes: str = ""


def _count(conn: duckdb.DuckDBPyConnection, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _staged_count(conn: duckdb.DuckDBPyConnection, parquet_path: Path) -> int:
    return conn.execute(
        f"SELECT COUNT(*) FROM read_parquet('{parquet_path.as_posix()}')"
    ).fetchone()[0]


# ── Loaders ────────────────────────────────────────────────────────────────


def load_statcast_gap_fill(conn: duckdb.DuckDBPyConnection) -> LoadResult:
    """Append 2017/2019 gap-fill pitches via anti-join on natural key."""
    parquet = STAGING / "statcast_gap_fill_2017_2019.parquet"
    pre = _count(conn, "pitches")
    staged = _staged_count(conn, parquet)

    # Pitches parquet has 53 columns in same order as the DB table.  One
    # type drift: game_date is TIMESTAMP_NS in parquet, DATE in DB — CAST.
    # All BIGINT columns in parquet auto-downcast to INTEGER, all DOUBLE to
    # FLOAT. Anti-join on (game_pk, at_bat_number, pitch_number).
    insert_sql = f"""
        INSERT INTO pitches
        SELECT
            CAST(game_pk AS INTEGER)               AS game_pk,
            CAST(game_date AS DATE)                AS game_date,
            CAST(pitcher_id AS INTEGER)            AS pitcher_id,
            CAST(batter_id AS INTEGER)             AS batter_id,
            pitch_type,
            pitch_name,
            CAST(release_speed AS FLOAT)           AS release_speed,
            CAST(release_spin_rate AS FLOAT)       AS release_spin_rate,
            CAST(spin_axis AS FLOAT)               AS spin_axis,
            CAST(pfx_x AS FLOAT)                   AS pfx_x,
            CAST(pfx_z AS FLOAT)                   AS pfx_z,
            CAST(plate_x AS FLOAT)                 AS plate_x,
            CAST(plate_z AS FLOAT)                 AS plate_z,
            CAST(release_extension AS FLOAT)       AS release_extension,
            CAST(release_pos_x AS FLOAT)           AS release_pos_x,
            CAST(release_pos_y AS FLOAT)           AS release_pos_y,
            CAST(release_pos_z AS FLOAT)           AS release_pos_z,
            CAST(launch_speed AS FLOAT)            AS launch_speed,
            CAST(launch_angle AS FLOAT)            AS launch_angle,
            CAST(hit_distance AS FLOAT)            AS hit_distance,
            CAST(hc_x AS FLOAT)                    AS hc_x,
            CAST(hc_y AS FLOAT)                    AS hc_y,
            bb_type,
            CAST(estimated_ba AS FLOAT)            AS estimated_ba,
            CAST(estimated_woba AS FLOAT)          AS estimated_woba,
            CAST(delta_home_win_exp AS FLOAT)      AS delta_home_win_exp,
            CAST(delta_run_exp AS FLOAT)           AS delta_run_exp,
            CAST(inning AS INTEGER)                AS inning,
            inning_topbot,
            CAST(outs_when_up AS INTEGER)          AS outs_when_up,
            CAST(balls AS INTEGER)                 AS balls,
            CAST(strikes AS INTEGER)               AS strikes,
            CAST(on_1b AS INTEGER)                 AS on_1b,
            CAST(on_2b AS INTEGER)                 AS on_2b,
            CAST(on_3b AS INTEGER)                 AS on_3b,
            stand,
            p_throws,
            CAST(at_bat_number AS INTEGER)         AS at_bat_number,
            CAST(pitch_number AS INTEGER)          AS pitch_number,
            description,
            events,
            type,
            home_team,
            away_team,
            CAST(woba_value AS FLOAT)              AS woba_value,
            CAST(woba_denom AS FLOAT)              AS woba_denom,
            CAST(babip_value AS FLOAT)             AS babip_value,
            CAST(iso_value AS FLOAT)               AS iso_value,
            CAST(zone AS INTEGER)                  AS zone,
            CAST(effective_speed AS FLOAT)         AS effective_speed,
            if_fielding_alignment,
            of_fielding_alignment,
            CAST(fielder_2 AS INTEGER)             AS fielder_2
        FROM read_parquet('{parquet.as_posix()}') src
        WHERE NOT EXISTS (
            SELECT 1 FROM pitches p
            WHERE p.game_pk       = CAST(src.game_pk AS INTEGER)
              AND p.at_bat_number = CAST(src.at_bat_number AS INTEGER)
              AND p.pitch_number  = CAST(src.pitch_number AS INTEGER)
        );
    """
    conn.execute(insert_sql)
    post = _count(conn, "pitches")
    inserted = post - pre
    return LoadResult(
        table="pitches",
        pre_count=pre,
        post_count=post,
        expected_staged=staged,
        staged_rows=staged,
        inserted=inserted,
        duplicates_skipped=staged - inserted,
        notes="APPEND; anti-join (game_pk, at_bat_number, pitch_number)",
    )


def load_season_war(
    conn: duckdb.DuckDBPyConnection,
    parquet_name: str,
    table: str,
) -> LoadResult:
    """Append 2010-2014 WAR rows to season_batting / season_pitching.

    Schema drift: parquet has 7 cols (player_id, player_name, season,
    position_type, war, pa_or_ip, war_source); the DuckDB table has 29/30
    cols. We insert (player_id, season, war) and — if the table has a `pa`
    / `ip` column — populate that too from `pa_or_ip`. All other advanced
    metrics stay NULL (bWAR doesn't publish them; Statcast-derived metrics
    are physically unrecoverable pre-2015).
    """
    parquet = STAGING / parquet_name
    pre = _count(conn, table)
    staged = _staged_count(conn, parquet)

    is_batting = table == "season_batting_stats"
    pa_ip_col = "pa" if is_batting else "ip"
    pa_ip_type = "INTEGER" if is_batting else "FLOAT"

    insert_sql = f"""
        INSERT INTO {table} (player_id, season, {pa_ip_col}, war)
        SELECT
            CAST(src.player_id AS INTEGER)   AS player_id,
            CAST(src.season AS INTEGER)      AS season,
            CAST(src.pa_or_ip AS {pa_ip_type}) AS {pa_ip_col},
            CAST(src.war AS FLOAT)           AS war
        FROM read_parquet('{parquet.as_posix()}') src
        WHERE NOT EXISTS (
            SELECT 1 FROM {table} t
            WHERE t.player_id = CAST(src.player_id AS INTEGER)
              AND t.season    = CAST(src.season AS INTEGER)
        );
    """
    conn.execute(insert_sql)
    post = _count(conn, table)
    inserted = post - pre
    return LoadResult(
        table=table,
        pre_count=pre,
        post_count=post,
        expected_staged=staged,
        staged_rows=staged,
        inserted=inserted,
        duplicates_skipped=staged - inserted,
        notes=(
            f"APPEND; anti-join (player_id, season); schema drift — only "
            f"(player_id, season, {pa_ip_col}, war) populated, advanced "
            f"metrics NULL (bWAR source, pre-Statcast)"
        ),
    )


def load_umpire_assignments(conn: duckdb.DuckDBPyConnection) -> LoadResult:
    """Load umpire_assignments (NEW table).

    game_pk is DOUBLE (with NULLs) in the parquet; cast to INTEGER.
    Idempotent via anti-join on (game_pk, position, umpire_name) for
    non-NULL game_pk rows, and (date, position, umpire_name) for NULL
    game_pk rows (Retrosheet unmatched).
    """
    parquet = STAGING / "umpire_assignments.parquet"
    pre = _count(conn, "umpire_assignments")
    staged = _staged_count(conn, parquet)
    insert_sql = f"""
        INSERT INTO umpire_assignments
        SELECT
            CAST(game_pk AS INTEGER)   AS game_pk,
            position,
            umpire_id,
            umpire_name,
            date,
            CAST(season AS INTEGER)    AS season,
            source
        FROM read_parquet('{parquet.as_posix()}') src
        WHERE NOT EXISTS (
            SELECT 1 FROM umpire_assignments u
            WHERE (
                (src.game_pk IS NOT NULL
                 AND u.game_pk = CAST(src.game_pk AS INTEGER)
                 AND u.position = src.position
                 AND u.umpire_name = src.umpire_name)
             OR (src.game_pk IS NULL
                 AND u.game_pk IS NULL
                 AND u.date = src.date
                 AND u.position = src.position
                 AND u.umpire_name = src.umpire_name)
            )
        );
    """
    conn.execute(insert_sql)
    post = _count(conn, "umpire_assignments")
    inserted = post - pre
    return LoadResult(
        table="umpire_assignments",
        pre_count=pre,
        post_count=post,
        expected_staged=staged,
        staged_rows=staged,
        inserted=inserted,
        duplicates_skipped=staged - inserted,
        notes="NEW table; anti-join (game_pk, position, umpire_name) or fallback on (date,...) for NULL game_pk",
    )


def load_umpire_tendencies(conn: duckdb.DuckDBPyConnection) -> LoadResult:
    parquet = STAGING / "umpire_tendencies.parquet"
    pre = _count(conn, "umpire_tendencies")
    staged = _staged_count(conn, parquet)
    insert_sql = f"""
        INSERT INTO umpire_tendencies
        SELECT
            umpire,
            CAST(season AS INTEGER)                  AS season,
            CAST(games AS INTEGER)                   AS games,
            CAST(called_pitches AS BIGINT)           AS called_pitches,
            CAST(called_correct AS BIGINT)           AS called_correct,
            CAST(called_correct_rate AS FLOAT)       AS called_correct_rate,
            CAST(overall_accuracy_wmean AS FLOAT)    AS overall_accuracy_wmean,
            CAST(x_overall_accuracy_wmean AS FLOAT)  AS x_overall_accuracy_wmean,
            CAST(accuracy_above_x_wmean AS FLOAT)    AS accuracy_above_x_wmean,
            CAST(consistency_wmean AS FLOAT)         AS consistency_wmean,
            CAST(favor_wmean AS FLOAT)               AS favor_wmean,
            CAST(total_run_impact_mean AS FLOAT)     AS total_run_impact_mean,
            CAST(batter_impact_mean AS FLOAT)        AS batter_impact_mean,
            CAST(pitcher_impact_mean AS FLOAT)       AS pitcher_impact_mean
        FROM read_parquet('{parquet.as_posix()}') src
        WHERE NOT EXISTS (
            SELECT 1 FROM umpire_tendencies t
            WHERE t.umpire = src.umpire
              AND t.season = CAST(src.season AS INTEGER)
        );
    """
    conn.execute(insert_sql)
    post = _count(conn, "umpire_tendencies")
    inserted = post - pre
    return LoadResult(
        table="umpire_tendencies",
        pre_count=pre,
        post_count=post,
        expected_staged=staged,
        staged_rows=staged,
        inserted=inserted,
        duplicates_skipped=staged - inserted,
        notes="NEW table; PK (umpire, season)",
    )


def load_game_weather(conn: duckdb.DuckDBPyConnection) -> LoadResult:
    parquet = STAGING / "game_weather.parquet"
    pre = _count(conn, "game_weather")
    staged = _staged_count(conn, parquet)
    insert_sql = f"""
        INSERT INTO game_weather
        SELECT
            CAST(game_pk AS INTEGER)         AS game_pk,
            venue,
            game_datetime_utc,
            CAST(temp_f AS FLOAT)            AS temp_f,
            CAST(wind_speed_mph AS FLOAT)    AS wind_speed_mph,
            CAST(wind_dir_deg AS FLOAT)      AS wind_dir_deg,
            CAST(humidity_pct AS FLOAT)      AS humidity_pct,
            CAST(pressure_mb AS FLOAT)       AS pressure_mb,
            roof_status,
            weather_source
        FROM read_parquet('{parquet.as_posix()}') src
        WHERE NOT EXISTS (
            SELECT 1 FROM game_weather w
            WHERE w.game_pk = CAST(src.game_pk AS INTEGER)
        );
    """
    conn.execute(insert_sql)
    post = _count(conn, "game_weather")
    inserted = post - pre
    return LoadResult(
        table="game_weather",
        pre_count=pre,
        post_count=post,
        expected_staged=staged,
        staged_rows=staged,
        inserted=inserted,
        duplicates_skipped=staged - inserted,
        notes="NEW table; PK game_pk",
    )


def load_tj_surgery_dates(conn: duckdb.DuckDBPyConnection) -> LoadResult:
    """Load tj_surgery_dates (NEW table).

    Dedup key: (mlb_id, surgery_date, source) — a pitcher may have multiple
    surgeries (revision TJ, dual-procedure) so (mlb_id) alone is not
    unique. NULL surgery_date rows (source='manual_review_needed') are
    deduped on (mlb_id, notes) to avoid re-inserting the same audit row.
    """
    parquet = STAGING / "tj_surgery_dates.parquet"
    pre = _count(conn, "tj_surgery_dates")
    staged = _staged_count(conn, parquet)
    insert_sql = f"""
        INSERT INTO tj_surgery_dates
        SELECT
            CAST(mlb_id AS BIGINT) AS mlb_id,
            player_name,
            surgery_date,
            return_date_est,
            confidence,
            source,
            notes
        FROM read_parquet('{parquet.as_posix()}') src
        WHERE NOT EXISTS (
            SELECT 1 FROM tj_surgery_dates t
            WHERE t.mlb_id IS NOT DISTINCT FROM CAST(src.mlb_id AS BIGINT)
              AND t.surgery_date IS NOT DISTINCT FROM src.surgery_date
              AND t.source IS NOT DISTINCT FROM src.source
              AND t.notes  IS NOT DISTINCT FROM src.notes
        );
    """
    conn.execute(insert_sql)
    post = _count(conn, "tj_surgery_dates")
    inserted = post - pre
    return LoadResult(
        table="tj_surgery_dates",
        pre_count=pre,
        post_count=post,
        expected_staged=staged,
        staged_rows=staged,
        inserted=inserted,
        duplicates_skipped=staged - inserted,
        notes="NEW table; anti-join (mlb_id, surgery_date, source, notes)",
    )


# ── Driver ─────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"[consolidate] Opening DuckDB write connection...")
    conn = get_connection(read_only=False)
    try:
        create_tables(conn)
        print("[consolidate] Schema ensured (CREATE TABLE IF NOT EXISTS).")
        results: list[LoadResult] = []

        # Wrap all loads in a single transaction so if any step fails we
        # rollback the whole pass and leave DuckDB in its prior state.
        conn.execute("BEGIN TRANSACTION;")
        try:
            steps: list[tuple[str, Callable[[duckdb.DuckDBPyConnection], LoadResult]]] = [
                ("statcast_gap_fill", load_statcast_gap_fill),
                (
                    "season_batting_2010_2014",
                    lambda c: load_season_war(
                        c,
                        "season_batting_stats_2010_2014.parquet",
                        "season_batting_stats",
                    ),
                ),
                (
                    "season_pitching_2010_2014",
                    lambda c: load_season_war(
                        c,
                        "season_pitching_stats_2010_2014.parquet",
                        "season_pitching_stats",
                    ),
                ),
                ("umpire_assignments", load_umpire_assignments),
                ("umpire_tendencies", load_umpire_tendencies),
                ("game_weather", load_game_weather),
                ("tj_surgery_dates", load_tj_surgery_dates),
            ]
            for name, loader in steps:
                print(f"[consolidate] Loading {name}...")
                r = loader(conn)
                results.append(r)
                print(
                    f"  -> {r.table}: pre={r.pre_count:,} post={r.post_count:,} "
                    f"inserted={r.inserted:,} dup_skipped={r.duplicates_skipped:,} "
                    f"(staged {r.staged_rows:,})"
                )
            conn.execute("COMMIT;")
            print("[consolidate] Transaction COMMITTED.")
        except Exception:
            conn.execute("ROLLBACK;")
            print("[consolidate] Transaction ROLLED BACK due to error.")
            raise

        # Print final summary table.
        print()
        print("=" * 96)
        print("CONSOLIDATION SUMMARY")
        print("=" * 96)
        header = (
            f"{'Table':<28}"
            f"{'Pre':>12}"
            f"{'Post':>14}"
            f"{'Staged':>12}"
            f"{'Inserted':>12}"
            f"{'Dup_Skip':>12}"
        )
        print(header)
        print("-" * 96)
        for r in results:
            print(
                f"{r.table:<28}"
                f"{r.pre_count:>12,}"
                f"{r.post_count:>14,}"
                f"{r.staged_rows:>12,}"
                f"{r.inserted:>12,}"
                f"{r.duplicates_skipped:>12,}"
            )
        print("=" * 96)

        # Touch data_freshness for the new tables.
        conn.execute(
            """
            INSERT OR REPLACE INTO data_freshness (table_name, max_game_date, row_count, updated_at)
            VALUES
              ('umpire_assignments', (SELECT MAX(date) FROM umpire_assignments),
                (SELECT COUNT(*) FROM umpire_assignments), CURRENT_TIMESTAMP),
              ('umpire_tendencies', NULL,
                (SELECT COUNT(*) FROM umpire_tendencies), CURRENT_TIMESTAMP),
              ('game_weather', (SELECT CAST(MAX(game_datetime_utc) AS DATE) FROM game_weather),
                (SELECT COUNT(*) FROM game_weather), CURRENT_TIMESTAMP),
              ('tj_surgery_dates', (SELECT MAX(surgery_date) FROM tj_surgery_dates),
                (SELECT COUNT(*) FROM tj_surgery_dates), CURRENT_TIMESTAMP),
              ('pitches', (SELECT MAX(game_date) FROM pitches),
                (SELECT COUNT(*) FROM pitches), CURRENT_TIMESTAMP)
            ;
            """
        )

        # Report final DB file size.
        db_path = PROJECT_ROOT / "data" / "baseball.duckdb"
        size_gb = db_path.stat().st_size / (1024 ** 3)
        print(f"[consolidate] Final DuckDB file size: {size_gb:.2f} GB")
    finally:
        conn.close()
        print("[consolidate] Connection closed; write lock released.")


if __name__ == "__main__":
    main()
