"""
Phillies Hub page.

The Phillies-focused command center with team record, standings,
scouting reports, lineup matchups, transactions, and schedule.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_USE_MOCK_MATCHUP = False
try:
    from src.analytics.matchups import (
        get_matchup_stats as _real_get_matchup_stats,
        get_lineup_matchups as _real_get_lineup_matchups,
        get_pitcher_profile as _real_get_pitcher_profile,
    )
except ImportError:
    _USE_MOCK_MATCHUP = True

_USE_MOCK_ROSTER = False
try:
    from src.ingest.roster_tracker import (
        get_active_roster as _real_get_active_roster,
        get_recent_transactions as _real_get_recent_transactions,
        get_phillies_pitching_staff as _real_get_phillies_pitching_staff,
    )
except ImportError:
    _USE_MOCK_ROSTER = True

_HAS_LIVE_FEED = False
try:
    from src.ingest.live_feed import (
        get_phillies_game as _real_get_phillies_game,
        get_todays_games as _real_get_todays_games,
    )
    _HAS_LIVE_FEED = True
except ImportError:
    pass

from src.dashboard.mock_data import (
    mock_lineup,
    mock_matchup_stats,
    mock_pitcher_profile,
    mock_schedule,
    mock_standings,
    mock_transactions,
)
from src.dashboard.db_helper import (
    get_db_connection,
    has_data,
    get_player_id_by_name,
)
from src.dashboard.components.pitch_movement import create_pitch_movement_chart


# ---------------------------------------------------------------------------
# Cached wrappers (no conn argument for st.cache_data compatibility)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _cached_active_roster() -> list[dict] | None:
    """Cached active roster from the MLB API (TTL 1 hour)."""
    if _USE_MOCK_ROSTER:
        return None
    try:
        roster = _real_get_active_roster(143)
        return roster if roster else None
    except Exception:
        return None


@st.cache_data(ttl=300)
def _cached_transactions() -> list[dict] | None:
    """Cached recent transactions from the MLB API (TTL 5 min)."""
    if _USE_MOCK_ROSTER:
        return None
    try:
        txns = _real_get_recent_transactions(143)
        return txns if txns else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _cached_pitching_staff() -> dict | None:
    """Cached Phillies pitching staff view (TTL 1 hour).

    Passes the DB connection for enrichment if available.
    """
    if _USE_MOCK_ROSTER:
        return None
    try:
        conn = get_db_connection()
        return _real_get_phillies_pitching_staff(conn=conn)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _cached_player_id_by_name(name: str) -> int | None:
    """Cached player ID lookup (TTL 1 hour) -- avoids repeated DB queries per rerun."""
    conn = get_db_connection()
    return get_player_id_by_name(conn, name)


@st.cache_data(ttl=300)
def _cached_lineup_matchups(pitcher_id: int, lineup_ids: tuple[int, ...]) -> list[dict] | None:
    """Cached lineup matchup cards (TTL 5 min)."""
    conn = get_db_connection()
    if conn is None or _USE_MOCK_MATCHUP:
        return None
    try:
        return _real_get_lineup_matchups(conn, pitcher_id, list(lineup_ids))
    except Exception:
        return None


@st.cache_data(ttl=300)
def _cached_standings() -> dict[str, Any] | None:
    """Fetch real Phillies standings from the MLB Stats API (TTL 5 min).

    Returns a dict matching the mock_standings() format, or None on failure.
    """
    try:
        resp = requests.get(
            "https://statsapi.mlb.com/api/v1/standings",
            params={"leagueId": 104, "season": 2026, "hydrate": "team"},
            timeout=10,
        )
        if not resp.ok:
            return None
        data = resp.json()
        for div in data.get("records", []):
            team_records = div.get("teamRecords", [])
            phi_record = None
            for t in team_records:
                if t.get("team", {}).get("id") == 143:
                    phi_record = t
                    break
            if phi_record is None:
                continue

            wins = phi_record.get("wins", 0)
            losses = phi_record.get("losses", 0)
            total = wins + losses
            pct = f".{int(wins / total * 1000):03d}" if total > 0 else ".000"
            streak = phi_record.get("streak", {}).get("streakCode", "")
            rank = int(phi_record.get("divisionRank", 0))

            # L10
            l10 = ""
            for sr in phi_record.get("records", {}).get("splitRecords", []):
                if sr.get("type") == "lastTen":
                    l10 = f"{sr.get('wins', 0)}-{sr.get('losses', 0)}"
                    break

            # Division standings
            division_standings = []
            for t in team_records:
                t_abbr = t.get("team", {}).get("abbreviation", "")
                t_wins = t.get("wins", 0)
                t_losses = t.get("losses", 0)
                t_total = t_wins + t_losses
                t_pct = f".{int(t_wins / t_total * 1000):03d}" if t_total > 0 else ".000"
                t_gb = t.get("gamesBack", "-")
                if t_gb == "0.0" or t_gb == 0:
                    t_gb = "-"
                division_standings.append({
                    "team": t_abbr,
                    "w": t_wins,
                    "l": t_losses,
                    "pct": t_pct,
                    "gb": str(t_gb),
                })

            return {
                "team": "PHI",
                "wins": wins,
                "losses": losses,
                "pct": pct,
                "gb": str(phi_record.get("gamesBack", "-")),
                "streak": streak,
                "l10": l10,
                "division": "NL East",
                "rank": rank,
                "division_standings": division_standings,
            }
    except Exception:
        pass
    return None


@st.cache_data(ttl=1800)
def _cached_schedule() -> list[dict[str, Any]] | None:
    """Fetch real Phillies schedule from the MLB Stats API (TTL 30 min).

    Uses a single date-range API call instead of 7 sequential requests.
    Returns a list of game dicts in mock_schedule() format, or None on failure.
    """
    from datetime import datetime, timedelta
    try:
        base = datetime.now()
        start_date = base.strftime("%Y-%m-%d")
        end_date = (base + timedelta(days=6)).strftime("%Y-%m-%d")

        resp = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={
                "sportId": 1,
                "startDate": start_date,
                "endDate": end_date,
                "teamId": 143,
                "hydrate": "team,probablePitcher",
            },
            timeout=10,
        )
        if not resp.ok:
            return None

        data = resp.json()
        result = []
        for gdate in data.get("dates", []):
            for g in gdate.get("games", []):
                home_abbr = g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
                away_abbr = g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")
                home_away = "vs" if home_abbr == "PHI" else "@"
                opp = away_abbr if home_abbr == "PHI" else home_abbr

                pp_home = g.get("teams", {}).get("home", {}).get("probablePitcher", {}).get("fullName", "TBD")
                pp_away = g.get("teams", {}).get("away", {}).get("probablePitcher", {}).get("fullName", "TBD")
                phi_sp = pp_home if home_abbr == "PHI" else pp_away
                opp_sp = pp_away if home_abbr == "PHI" else pp_home

                start_time = ""
                game_dt = g.get("gameDate", "")
                if game_dt:
                    try:
                        from zoneinfo import ZoneInfo
                        dt_utc = datetime.fromisoformat(game_dt.replace("Z", "+00:00"))
                        dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
                        start_time = dt_et.strftime("%I:%M %p ET").lstrip("0")
                    except (ValueError, OSError, ImportError):
                        start_time = game_dt

                # Parse the actual game date from the API response
                raw_date = gdate.get("date", "")
                if raw_date:
                    try:
                        game_date_obj = datetime.strptime(raw_date, "%Y-%m-%d")
                    except ValueError:
                        game_date_obj = base
                else:
                    game_date_obj = base

                result.append({
                    "date": game_date_obj.strftime("%m/%d"),
                    "day": game_date_obj.strftime("%a"),
                    "opponent": opp,
                    "home_away": home_away,
                    "time": start_time,
                    "phillies_starter": phi_sp,
                    "opponent_starter": opp_sp,
                })
        return result if result else None
    except Exception:
        pass
    return None


@st.cache_data(ttl=300)
def _cached_todays_game() -> dict[str, Any] | None:
    """Fetch today's Phillies game info (TTL 5 min).

    Returns a game dict in the format expected by _render_today_game(), or None.
    """
    from datetime import datetime
    try:
        resp = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={
                "sportId": 1,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "teamId": 143,
                "hydrate": "team,probablePitcher",
            },
            timeout=10,
        )
        if not resp.ok:
            return None
        data = resp.json()
        for gdate in data.get("dates", []):
            for g in gdate.get("games", []):
                home_abbr = g.get("teams", {}).get("home", {}).get("team", {}).get("abbreviation", "")
                away_abbr = g.get("teams", {}).get("away", {}).get("team", {}).get("abbreviation", "")
                home_away = "vs" if home_abbr == "PHI" else "@"
                opp = away_abbr if home_abbr == "PHI" else home_abbr

                pp_home = g.get("teams", {}).get("home", {}).get("probablePitcher", {}).get("fullName", "TBD")
                pp_away = g.get("teams", {}).get("away", {}).get("probablePitcher", {}).get("fullName", "TBD")
                phi_sp = pp_home if home_abbr == "PHI" else pp_away
                opp_sp = pp_away if home_abbr == "PHI" else pp_home

                start_time = ""
                game_dt = g.get("gameDate", "")
                if game_dt:
                    try:
                        from zoneinfo import ZoneInfo
                        dt_utc = datetime.fromisoformat(game_dt.replace("Z", "+00:00"))
                        dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
                        start_time = dt_et.strftime("%I:%M %p ET").lstrip("0")
                    except (ValueError, OSError, ImportError):
                        start_time = game_dt

                return {
                    "opponent": opp,
                    "home_away": home_away,
                    "time": start_time,
                    "phillies_starter": phi_sp,
                    "opponent_starter": opp_sp,
                }
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Adapter functions
# ---------------------------------------------------------------------------

def _get_matchup_stats(pitcher_name: str, batter_name: str) -> dict[str, Any] | None:
    """Get matchup stats. Tries real DB first, returns None if unavailable."""
    conn = get_db_connection()
    if has_data(conn) and not _USE_MOCK_MATCHUP:
        try:
            pitcher_id = _cached_player_id_by_name(pitcher_name)
            batter_id = _cached_player_id_by_name(batter_name)
            if pitcher_id is not None and batter_id is not None:
                real = _real_get_matchup_stats(conn, pitcher_id, batter_id)
                return _adapt_matchup_stats(real, pitcher_name, batter_name)
        except Exception:
            pass
    return None


def _adapt_matchup_stats(real: dict, pitcher_name: str, batter_name: str) -> dict[str, Any]:
    """Adapt real matchup stats to dashboard format."""
    pa = real.get("plate_appearances", 0)
    pitch_breakdown: list[dict] = []
    for pt, info in real.get("pitch_type_breakdown", {}).items():
        pitch_breakdown.append({
            "pitch_type": pt,
            "label": pt,
            "count": info.get("count", 0),
            "pct": 0,
            "whiff_rate": info.get("whiff_rate", 0.0),
            "ba_against": info.get("ba_against", 0.0),
            "avg_velo": info.get("avg_velo"),
        })
    total = sum(p["count"] for p in pitch_breakdown)
    for p in pitch_breakdown:
        p["pct"] = round(p["count"] / total, 3) if total else 0

    vulnerability = ""
    if pitch_breakdown:
        weak = max(pitch_breakdown, key=lambda p: p.get("ba_against", 0) or 0)
        if weak.get("ba_against"):
            vulnerability = (
                f"Struggles against {weak['label']} "
                f"({weak['ba_against']:.3f} BA, {weak['whiff_rate']:.0%} whiff)"
            )

    return {
        "pitcher_name": pitcher_name,
        "batter_name": batter_name,
        "pa": pa,
        "ab": pa,
        "h": 0,
        "hr": 0,
        "k": 0,
        "bb": 0,
        "avg": real.get("batting_avg", 0.0),
        "slg": real.get("slug_pct", 0.0),
        "estimated_woba": real.get("woba", 0.0),
        "woba_ci_low": 0.0,
        "woba_ci_high": 0.0,
        "pitch_breakdown": pitch_breakdown,
        "vulnerability": vulnerability,
        "sample_size": pa,
    }


@st.cache_data(ttl=300)
def _get_transactions() -> list[dict[str, Any]]:
    """Return recent transactions.  Tries real API, falls back to mock."""
    real_txns = _cached_transactions()
    if real_txns is not None:
        # Adapt API output format to what dashboard expects
        adapted = []
        for txn in real_txns:
            adapted.append({
                "date": txn.get("date", ""),
                "type": txn.get("type", ""),
                "player": txn.get("player", ""),
                "detail": txn.get("description", ""),
            })
        return adapted
    st.warning("Showing demo data -- real transaction data unavailable")
    return mock_transactions()


@st.cache_data(ttl=600)
def _get_pitching_staff() -> dict[str, Any] | None:
    """Return Phillies pitching staff view. Falls back to None."""
    return _cached_pitching_staff()


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Phillies Hub page."""
    st.title("Phillies Hub")
    st.info("""
**Phillies command center** — team record, standings, scouting reports on opposing starters, lineup matchups, recent transactions, and upcoming schedule.

- **Scouting reports** show opposing starter arsenals so you know what to expect before first pitch
- **Lineup matchups** highlight which Phillies hitters have the best/worst history against today's starter
""")

    # Determine data source
    conn = get_db_connection()
    use_real = has_data(conn)

    # Fetch REAL standings from MLB API, fall back to mock only on failure
    standings = _cached_standings()
    if standings is None:
        standings = mock_standings()
        st.warning("Showing demo data -- standings API unreachable")

    # Fetch REAL schedule from MLB API, fall back to mock only on failure
    schedule = _cached_schedule()
    if schedule is None:
        schedule = mock_schedule()
        st.warning("Showing demo data -- schedule API unreachable")

    # Fetch REAL today's game info
    today_game = _cached_todays_game()
    if today_game is None and schedule:
        today_game = schedule[0]

    # Show roster from real API if available
    real_roster = _cached_active_roster()
    if real_roster is not None:
        st.info(f"Roster loaded from MLB API: {len(real_roster)} players.")
    else:
        st.caption("Roster data unavailable from MLB API.")

    # ----- Record & standings -----
    _render_record(standings)

    st.markdown("---")

    # ----- Pitching staff (from real API + DB enrichment) -----
    pitching_staff = _get_pitching_staff()
    if pitching_staff is not None:
        _render_pitching_staff(pitching_staff)
        st.markdown("---")

    # ----- Today's game info -----
    if today_game:
        _render_today_game(today_game)
        st.markdown("---")

        # ----- Opponent starter scouting -----
        opp_starter = today_game.get("opponent_starter", "")
        if opp_starter and opp_starter != "TBD":
            _render_scouting_report(opp_starter, use_real)
            st.markdown("---")

        # ----- Phillies lineup matchup breakdown -----
        if opp_starter and opp_starter != "TBD":
            _render_lineup_matchups(opp_starter, use_real)
            st.markdown("---")

    # ----- Transactions -----
    _render_transactions()

    st.markdown("---")

    # ----- Schedule -----
    _render_schedule(schedule)


# ---------------------------------------------------------------------------
# Sub-renderers
# ---------------------------------------------------------------------------

def _render_record(standings: dict[str, Any]) -> None:
    """Show team record and NL East standings."""
    st.subheader("Team Record & NL East Standings")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Record", f"{standings['wins']}-{standings['losses']}")
    c2.metric("Win %", standings["pct"])
    c3.metric("NL East", f"#{standings['rank']}")
    c4.metric("Streak", standings["streak"])
    c5.metric("Last 10", standings["l10"])

    # Division standings table
    div = standings.get("division_standings", [])
    if div:
        df = pd.DataFrame(div)
        df = df.rename(columns={
            "team": "Team", "w": "W", "l": "L", "pct": "Pct", "gb": "GB",
        })

        def _highlight_phi(row: pd.Series) -> list[str]:
            if row["Team"] == "PHI":
                return ["background-color: rgba(232, 24, 40, 0.2); font-weight: bold"] * len(row)
            return [""] * len(row)

        styled = df.style.apply(_highlight_phi, axis=1)
        st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_pitching_staff(staff: dict[str, Any]) -> None:
    """Render the pitching staff section from real API data."""
    st.subheader("Pitching Staff")

    col_rot, col_bp = st.columns(2)

    with col_rot:
        st.markdown("**Rotation**")
        rotation = staff.get("rotation", [])
        if rotation:
            df = pd.DataFrame(rotation)
            cols = ["name", "throws", "era"]
            available = [c for c in cols if c in df.columns]
            df_show = df[available].rename(columns={
                "name": "Name", "throws": "Throws", "era": "ERA",
            })
            st.dataframe(df_show, use_container_width=True, hide_index=True)
        else:
            st.info("No rotation data available.")

    with col_bp:
        st.markdown("**Bullpen**")
        bullpen_list = staff.get("bullpen", [])
        if bullpen_list:
            df = pd.DataFrame(bullpen_list)
            cols = ["name", "throws", "role", "era"]
            available = [c for c in cols if c in df.columns]
            df_show = df[available].rename(columns={
                "name": "Name", "throws": "Throws", "role": "Role", "era": "ERA",
            })
            st.dataframe(df_show, use_container_width=True, hide_index=True)
        else:
            st.info("No bullpen data available.")

    # Injured list
    il = staff.get("injured_list", [])
    if il:
        st.markdown("**Injured List**")
        df = pd.DataFrame(il)
        cols = ["name", "injury"]
        available = [c for c in cols if c in df.columns]
        df_show = df[available].rename(columns={"name": "Name", "injury": "Status"})
        st.dataframe(df_show, use_container_width=True, hide_index=True)


def _render_today_game(game: dict[str, Any]) -> None:
    """Show today's game info."""
    st.subheader("Today's Game")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Opponent", f"{game['home_away']} {game['opponent']}")
    c2.metric("First Pitch", game["time"])
    c3.metric("PHI Starter", game["phillies_starter"])
    c4.metric("Opp Starter", game["opponent_starter"])


def _render_scouting_report(pitcher_name: str, use_real: bool = False) -> None:
    """Opponent starting pitcher scouting report."""
    st.subheader(f"Scouting Report: {pitcher_name}")

    # Try real pitcher profile from DB first, fall back to mock
    profile = None
    if use_real and not _USE_MOCK_MATCHUP:
        try:
            conn = get_db_connection()
            pitcher_id = _cached_player_id_by_name(pitcher_name)
            if pitcher_id is not None:
                real_profile = _real_get_pitcher_profile(conn, pitcher_id)
                if real_profile is not None:
                    # Adapt real profile to dashboard format
                    from src.dashboard.views.matchups import _adapt_pitcher_profile
                    profile = _adapt_pitcher_profile(real_profile)
        except Exception:
            pass
    if profile is None:
        profile = mock_pitcher_profile(pitcher_name)
        st.warning("Using demo data for scouting report -- real data unavailable for this pitcher.")

    # Headline stats
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ERA", f"{profile['era']:.2f}")
    c2.metric("FIP", f"{profile['fip']:.2f}")
    c3.metric("WHIP", f"{profile['whip']:.2f}")
    c4.metric("K/9", f"{profile['k_per_9']:.1f}")
    c5.metric("BB/9", f"{profile['bb_per_9']:.1f}")

    col_chart, col_splits = st.columns([3, 2])

    with col_chart:
        # Pitch movement chart
        arsenal = profile.get("arsenal", [])
        if arsenal:
            fig = create_pitch_movement_chart(
                arsenal,
                title=f"{pitcher_name} -- Pitch Movement",
            )
            st.plotly_chart(fig, use_container_width=True, key="scout_movement")

            # Arsenal table
            df_ars = pd.DataFrame(arsenal)
            show_cols = ["label", "pct", "avg_velo", "avg_spin", "whiff_rate", "ba_against"]
            avail = [c for c in show_cols if c in df_ars.columns]
            df_show = df_ars[avail].copy()
            df_show = df_show.rename(columns={
                "label": "Pitch", "pct": "Usage", "avg_velo": "Velo",
                "avg_spin": "Spin", "whiff_rate": "Whiff %", "ba_against": "BA Agst",
            })
            if "Usage" in df_show.columns:
                df_show["Usage"] = (df_show["Usage"] * 100).round(1).astype(str) + "%"
            if "Whiff %" in df_show.columns:
                df_show["Whiff %"] = (df_show["Whiff %"] * 100).round(1).astype(str) + "%"
            if "BA Agst" in df_show.columns:
                df_show["BA Agst"] = df_show["BA Agst"].apply(lambda v: f"{v:.3f}")
            st.dataframe(df_show, use_container_width=True, hide_index=True)

    with col_splits:
        st.markdown("**Platoon Splits**")
        vs_l = profile.get("vs_left", {})
        vs_r = profile.get("vs_right", {})
        split_data = {
            "Split": ["vs LHB", "vs RHB"],
            "AVG": [f"{vs_l.get('avg', 0):.3f}", f"{vs_r.get('avg', 0):.3f}"],
            "wOBA": [f"{vs_l.get('woba', 0):.3f}", f"{vs_r.get('woba', 0):.3f}"],
            "K Rate": [f"{vs_l.get('k_rate', 0):.1%}", f"{vs_r.get('k_rate', 0):.1%}"],
        }
        st.dataframe(pd.DataFrame(split_data), use_container_width=True, hide_index=True)

        # Recent starts
        st.markdown("**Recent Starts**")
        recent = profile.get("recent_starts", [])
        if recent:
            df_recent = pd.DataFrame(recent)
            df_recent = df_recent.rename(columns={
                "date": "Date", "ip": "IP", "h": "H", "er": "ER", "k": "K", "bb": "BB",
            })
            st.dataframe(df_recent, use_container_width=True, hide_index=True)

        # Key vulnerabilities
        st.markdown("**Key Vulnerabilities**")
        weakest = None
        if arsenal:
            weakest = max(arsenal, key=lambda p: p.get("ba_against", 0))
        if weakest:
            st.warning(
                f"Weakest pitch: **{weakest.get('label', '')}** "
                f"({weakest.get('ba_against', 0):.3f} BA against, "
                f"{weakest.get('whiff_rate', 0):.1%} whiff)"
            )


def _render_lineup_matchups(opp_starter: str, use_real: bool = False) -> None:
    """Show each Phillies batter's projected matchup vs today's starter.

    When real data is available, uses ``get_lineup_matchups`` for actual
    Bayesian wOBA estimates.
    """
    st.subheader(f"Phillies Lineup vs {opp_starter}")

    conn = get_db_connection()

    # Try to get real lineup from roster API
    real_roster = _cached_active_roster()
    if real_roster is not None:
        # Build lineup from real roster (position players only)
        lineup = []
        order = 1
        for p in real_roster:
            pos = p.get("position", {}).get("abbreviation", "")
            if pos != "P" and pos != "TWP":
                lineup.append({
                    "order": order,
                    "name": p.get("fullName", p.get("name", "")),
                    "pos": pos,
                    "bats": p.get("batSide", {}).get("code", ""),
                })
                order += 1
    else:
        lineup = mock_lineup("PHI")

    # Try real lineup matchups if DB is available
    if use_real and not _USE_MOCK_MATCHUP:
        pitcher_id = _cached_player_id_by_name(opp_starter)
        if pitcher_id is not None:
            # Resolve batter IDs (cached individually -- avoids N sequential DB hits)
            lineup_ids: list[int] = []
            for batter in lineup:
                bid = _cached_player_id_by_name(batter["name"])
                if bid is not None:
                    lineup_ids.append(bid)

            if lineup_ids:
                real_cards = _cached_lineup_matchups(pitcher_id, tuple(lineup_ids))
                if real_cards is not None:
                    _render_real_lineup_matchups(real_cards, lineup)
                    return

    # No real matchup data available
    st.info("Matchup data requires historical pitch data. Run backfill.py to load Statcast data.")


def _render_real_lineup_matchups(cards: list[dict], lineup: list[dict]) -> None:
    """Render lineup matchups using real data from get_lineup_matchups."""
    rows = []
    for card in cards:
        batter_profile = card.get("batter", {})
        matchup_estimate = card.get("matchup_estimate", {})
        matchup_history = card.get("matchup_history", {})

        name = batter_profile.get("name", "Unknown")
        # Find lineup position
        order = card.get("lineup_order", 0)
        pos = ""
        bats = batter_profile.get("bats", "")
        for b in lineup:
            if b["name"] == name:
                pos = b.get("pos", "")
                break

        estimated_woba = matchup_estimate.get("estimated_woba", 0.0)
        vulnerability = matchup_estimate.get("key_pitch_vulnerability", "")

        rows.append({
            "#": order,
            "Player": name,
            "Pos": pos,
            "Bats": bats,
            "PA": matchup_history.get("plate_appearances", 0),
            "AVG": f"{matchup_history.get('batting_avg', 0):.3f}",
            "Est wOBA": f"{estimated_woba:.3f}",
            "Vulnerability": vulnerability or "",
            "Favorable": "Yes" if card.get("most_favorable_for_batter") else "",
        })

    df = pd.DataFrame(rows)

    def _color_woba(row: pd.Series) -> list[str]:
        try:
            woba = float(row["Est wOBA"])
        except (ValueError, TypeError):
            return [""] * len(row)
        if woba >= 0.370:
            return ["background-color: rgba(46, 204, 113, 0.2)"] * len(row)
        elif woba <= 0.290:
            return ["background-color: rgba(231, 76, 60, 0.2)"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_color_woba, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_transactions() -> None:
    """Show recent roster transactions."""
    st.subheader("Recent Transactions")
    txns = _get_transactions()
    if not txns:
        st.info("No recent transactions.")
        return

    df = pd.DataFrame(txns)
    df = df.rename(columns={
        "date": "Date", "type": "Type", "player": "Player", "detail": "Details",
    })
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_schedule(schedule: list[dict[str, Any]]) -> None:
    """Show the upcoming 7-game schedule."""
    st.subheader("Upcoming Schedule")
    if not schedule:
        st.info("No upcoming games.")
        return

    df = pd.DataFrame(schedule)
    df = df.rename(columns={
        "date": "Date", "day": "Day", "opponent": "Opp",
        "home_away": "H/A", "time": "Time",
        "phillies_starter": "PHI Starter", "opponent_starter": "Opp Starter",
    })
    st.dataframe(df, use_container_width=True, hide_index=True)
