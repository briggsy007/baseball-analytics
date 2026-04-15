"""
Data Management page for the Diamond Analytics dashboard.

Displays database status (pitch count, date range, last update), and
provides buttons to trigger daily refresh and roster sync operations.
Also shows a recent data-load history derived from the pitches table.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import streamlit as st

from src.dashboard.db_helper import get_db_connection, get_data_status, has_data

logger = logging.getLogger(__name__)


def render() -> None:
    """Render the Data Management page."""
    st.header("Data Management")
    st.caption("Monitor database health and trigger data refresh operations.")
    st.info("""
**Database health and data freshness.** Trigger ETL refreshes, check data coverage, and monitor load history.

- **Pitch count** shows total Statcast data loaded — full coverage is ~700K pitches per season
- **Date range** should extend to yesterday if daily refresh is running correctly
""")

    conn = get_db_connection()

    # ── Section 1: Database Status ──────────────────────────────────────────
    st.subheader("Database Status")

    if conn is None:
        st.error(
            "Cannot connect to the database. "
            "Run `python scripts/setup_db.py` to initialise it."
        )
    else:
        status = get_data_status(conn)

        if status is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Pitches", f"{status['pitch_count']:,}")
            with col2:
                st.metric("Earliest Date", str(status["min_date"]))
            with col3:
                st.metric("Latest Date", str(status["max_date"]))

            # Additional metrics
            _render_extended_status(conn)
        else:
            st.warning(
                "No pitch data loaded. Run the backfill or daily refresh to populate."
            )

    st.markdown("---")

    # ── Section 2: Action Buttons ───────────────────────────────────────────
    st.subheader("Actions")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Daily Refresh**")
        st.caption(
            "Loads yesterday's Statcast data, refreshes season stats, "
            "syncs roster, and rebuilds the matchup cache."
        )
        if st.button("Run Daily Refresh", key="btn_daily_refresh", type="primary",
                      disabled=(conn is None)):
            _run_daily_refresh(conn)

    with col_b:
        st.markdown("**Roster Sync**")
        st.caption(
            "Updates the Phillies active roster from the MLB Stats API."
        )
        if st.button("Sync Roster", key="btn_roster_sync",
                      disabled=(conn is None)):
            _run_roster_sync(conn)

    st.markdown("---")

    # ── Section 3: Recent Data Load History ─────────────────────────────────
    st.subheader("Recent Data Loads")
    if conn is not None:
        _render_load_history(conn)
    else:
        st.info("Connect to the database to view load history.")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _render_extended_status(conn) -> None:
    """Show extra database statistics in an expander."""
    with st.expander("Detailed Status", expanded=False):
        try:
            # Player count
            player_count = conn.execute(
                "SELECT COUNT(*) FROM players"
            ).fetchone()[0]

            # Game count
            game_count = 0
            try:
                game_count = conn.execute(
                    "SELECT COUNT(*) FROM games"
                ).fetchone()[0]
            except Exception:
                pass

            # Season stats
            batting_rows = 0
            pitching_rows = 0
            try:
                batting_rows = conn.execute(
                    "SELECT COUNT(*) FROM season_batting_stats"
                ).fetchone()[0]
            except Exception:
                pass
            try:
                pitching_rows = conn.execute(
                    "SELECT COUNT(*) FROM season_pitching_stats"
                ).fetchone()[0]
            except Exception:
                pass

            # Matchup summary
            matchup_rows = 0
            try:
                matchup_rows = conn.execute(
                    "SELECT COUNT(*) FROM matchup_summary"
                ).fetchone()[0]
            except Exception:
                pass

            # Transaction count
            txn_count = 0
            try:
                txn_count = conn.execute(
                    "SELECT COUNT(*) FROM transactions"
                ).fetchone()[0]
            except Exception:
                pass

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Players", f"{player_count:,}")
                st.metric("Batting Stats", f"{batting_rows:,}")
            with c2:
                st.metric("Games", f"{game_count:,}")
                st.metric("Pitching Stats", f"{pitching_rows:,}")
            with c3:
                st.metric("Matchup Cache", f"{matchup_rows:,}")
                st.metric("Transactions", f"{txn_count:,}")

        except Exception as exc:
            st.warning(f"Could not fetch extended status: {exc}")


def _render_load_history(conn) -> None:
    """Show the number of pitches loaded per day for the most recent 14 days."""
    if not has_data(conn):
        st.info("No data to show load history.")
        return

    try:
        df = conn.execute("""
            SELECT game_date,
                   COUNT(*) AS pitch_count,
                   COUNT(DISTINCT game_pk) AS games
            FROM pitches
            WHERE game_date >= CURRENT_DATE - INTERVAL '14' DAY
            GROUP BY game_date
            ORDER BY game_date DESC
        """).fetchdf()

        if df.empty:
            st.info("No data in the last 14 days.")
            return

        # Display as a clean table
        df.columns = ["Date", "Pitches", "Games"]
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

    except Exception as exc:
        st.warning(f"Could not fetch load history: {exc}")


def _run_daily_refresh(conn) -> None:
    """Trigger the daily ETL with a progress indicator."""
    with st.spinner("Running daily ETL... This may take a few minutes."):
        try:
            from src.ingest.daily_etl import run_daily_etl
            summary = run_daily_etl(conn=conn)

            st.success(
                f"Daily refresh complete! "
                f"Pitches: {summary.get('pitches', 0):,} | "
                f"Batting: {summary.get('batting_rows', 0):,} | "
                f"Pitching: {summary.get('pitching_rows', 0):,} | "
                f"Players: {summary.get('players', 0):,}"
            )

            # Clear the cached status so it refreshes on next render
            _bust_cache()

        except Exception as exc:
            st.error(f"Daily refresh failed: {exc}")
            logger.exception("Daily refresh failed from dashboard")


def _run_roster_sync(conn) -> None:
    """Trigger a roster sync with a progress indicator."""
    with st.spinner("Syncing Phillies roster..."):
        try:
            from src.ingest.roster_tracker import sync_roster_to_db, PHILLIES_TEAM_ID
            count = sync_roster_to_db(conn, team_id=PHILLIES_TEAM_ID)
            st.success(f"Roster synced: {count} players updated.")
            _bust_cache()
        except Exception as exc:
            st.error(f"Roster sync failed: {exc}")
            logger.exception("Roster sync failed from dashboard")


def _bust_cache() -> None:
    """Attempt to clear Streamlit caches so fresh data is shown."""
    try:
        st.cache_data.clear()
    except Exception:
        pass
