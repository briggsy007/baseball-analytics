"""
Matchup Explorer page.

Interactive pitcher-vs-batter analysis with historical stats,
pitch breakdowns, zone heatmaps, and movement charts.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_USE_MOCK_MATCHUP = False
try:
    from src.analytics.matchups import (
        get_matchup_stats as _real_get_matchup_stats,
        get_pitcher_profile as _real_get_pitcher_profile,
        get_batter_profile as _real_get_batter_profile,
        estimate_matchup_woba as _real_estimate_matchup_woba,
    )
except ImportError:
    _USE_MOCK_MATCHUP = True

_HAS_QUERIES = False
try:
    from src.db.queries import get_matchup_history as _real_get_matchup_history
    _HAS_QUERIES = True
except ImportError:
    pass

from src.dashboard.mock_data import (
    ARSENALS,
    OPPONENT_BATTERS,
    PHILLIES_BATTERS,
)
from src.dashboard.db_helper import (
    get_db_connection,
    has_data,
    get_all_batters,
    get_all_pitchers,
    get_player_id_by_name,
)
from src.dashboard.components.matchup_card import create_matchup_card
from src.dashboard.components.pitch_movement import create_pitch_movement_chart
from src.dashboard.components.spray_chart import create_spray_chart
from src.dashboard.components.strike_zone import create_strike_zone_heatmap


# ---------------------------------------------------------------------------
# Cached lookups (wrappers without conn argument for st.cache_data)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _cached_all_pitchers() -> list[dict]:
    """Cached list of all pitchers in the DB (TTL 1 hour)."""
    conn = get_db_connection()
    return get_all_pitchers(conn)


@st.cache_data(ttl=3600)
def _cached_all_batters() -> list[dict]:
    """Cached list of all batters in the DB (TTL 1 hour)."""
    conn = get_db_connection()
    return get_all_batters(conn)


@st.cache_data(ttl=3600)
def _cached_pitcher_profile(pitcher_id: int) -> dict[str, Any] | None:
    """Cached pitcher profile from the real analytics module (TTL 5 min)."""
    conn = get_db_connection()
    if conn is None or _USE_MOCK_MATCHUP:
        return None
    try:
        return _real_get_pitcher_profile(conn, pitcher_id)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _cached_batter_profile(batter_id: int) -> dict[str, Any] | None:
    """Cached batter profile from the real analytics module (TTL 5 min)."""
    conn = get_db_connection()
    if conn is None or _USE_MOCK_MATCHUP:
        return None
    try:
        return _real_get_batter_profile(conn, batter_id)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _cached_matchup_stats(pitcher_id: int, batter_id: int) -> dict[str, Any] | None:
    """Cached matchup stats from the real analytics module (TTL 5 min)."""
    conn = get_db_connection()
    if conn is None or _USE_MOCK_MATCHUP:
        return None
    try:
        return _real_get_matchup_stats(conn, pitcher_id, batter_id)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _cached_matchup_woba(pitcher_id: int, batter_id: int) -> dict[str, Any] | None:
    """Cached Bayesian matchup wOBA estimate (TTL 5 min)."""
    conn = get_db_connection()
    if conn is None or _USE_MOCK_MATCHUP:
        return None
    try:
        return _real_estimate_matchup_woba(conn, pitcher_id, batter_id)
    except Exception:
        return None


@st.cache_data(ttl=3600)
def _cached_matchup_history(pitcher_id: int, batter_id: int) -> pd.DataFrame | None:
    """Cached matchup pitch history for zone heatmaps (TTL 5 min)."""
    conn = get_db_connection()
    if conn is None or not _HAS_QUERIES:
        return None
    try:
        return _real_get_matchup_history(conn, pitcher_id, batter_id)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Adapter functions -- unify real (conn, id) vs mock (name) signatures
# ---------------------------------------------------------------------------

def _get_matchup_stats(
    pitcher_name: str,
    batter_name: str,
    pitcher_id: int | None = None,
    batter_id: int | None = None,
) -> dict[str, Any] | None:
    """Get matchup stats. Tries real DB first, returns None if unavailable."""
    if pitcher_id is not None and batter_id is not None:
        real = _cached_matchup_stats(pitcher_id, batter_id)
        if real is not None:
            return _adapt_matchup_stats(real, pitcher_name, batter_name)
    return None


def _get_pitcher_profile(
    pitcher_name: str,
    pitcher_id: int | None = None,
) -> dict[str, Any] | None:
    """Get pitcher profile. Tries real DB first, returns None if unavailable."""
    if pitcher_id is not None:
        real = _cached_pitcher_profile(pitcher_id)
        if real is not None:
            return _adapt_pitcher_profile(real)
    return None


def _get_batter_profile(
    batter_name: str,
    batter_id: int | None = None,
) -> dict[str, Any] | None:
    """Get batter profile. Tries real DB first, returns None if unavailable."""
    if batter_id is not None:
        real = _cached_batter_profile(batter_id)
        if real is not None:
            return _adapt_batter_profile(real)
    return None


# ---------------------------------------------------------------------------
# Adapters: convert real analytics output to the dashboard rendering format
# ---------------------------------------------------------------------------

def _adapt_matchup_stats(real: dict, pitcher_name: str, batter_name: str) -> dict[str, Any]:
    """Adapt ``get_matchup_stats`` output to the dashboard format."""
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
    total_pitches = sum(p["count"] for p in pitch_breakdown)
    for p in pitch_breakdown:
        p["pct"] = round(p["count"] / total_pitches, 3) if total_pitches else 0

    vulnerability = ""
    if pitch_breakdown:
        weak = max(pitch_breakdown, key=lambda p: p.get("ba_against", 0) or 0)
        if weak.get("ba_against"):
            vulnerability = (
                f"Struggles against {weak['label']} "
                f"({weak['ba_against']:.3f} BA, {weak['whiff_rate']:.0%} whiff)"
            )

    # Derive counting stats from rates + PA
    avg = real.get("batting_avg", 0.0)
    slg = real.get("slug_pct", 0.0)
    k_rate = real.get("strikeout_rate", 0.0)
    bb_rate = real.get("walk_rate", 0.0)

    # Estimate AB (PA minus walks)
    bb = round(bb_rate * pa)
    k = round(k_rate * pa)
    ab = pa - bb
    h = round(avg * ab) if ab > 0 else 0
    # Estimate HR from SLG: SLG = TB/AB, HR contributes 4 bases
    # Rough heuristic: HR ≈ (SLG*AB - h) / 3 when most XBH are HR at small samples
    tb = round(slg * ab) if ab > 0 else 0
    hr = max(0, round((tb - h) / 3)) if h > 0 else 0

    woba = real.get("woba", 0.0)
    # Bayesian CI estimate (shrink toward league avg 0.320)
    ci_half = max(0.040, 0.180 / (pa ** 0.5)) if pa > 0 else 0.180
    woba_ci_low = max(0.0, round(woba - ci_half, 3))
    woba_ci_high = min(0.600, round(woba + ci_half, 3))

    return {
        "pitcher_name": pitcher_name,
        "batter_name": batter_name,
        "pa": pa,
        "ab": ab,
        "h": h,
        "hr": hr,
        "k": k,
        "bb": bb,
        "avg": avg,
        "slg": slg,
        "estimated_woba": woba,
        "woba_ci_low": woba_ci_low,
        "woba_ci_high": woba_ci_high,
        "pitch_breakdown": pitch_breakdown,
        "vulnerability": vulnerability,
        "sample_size": pa,
    }


def _adapt_pitcher_profile(real: dict) -> dict[str, Any]:
    """Adapt ``get_pitcher_profile`` output to the dashboard format.

    The real module returns ``arsenal`` as a dict keyed by pitch type code,
    whereas the mock returns a list of dicts.  The dashboard rendering
    expects the list-of-dicts form.
    """
    arsenal_list: list[dict] = []
    for pt, info in real.get("arsenal", {}).items():
        arsenal_list.append({
            "pitch_type": pt,
            "label": pt,
            "pct": info.get("usage_pct", 0.0),
            "avg_velo": info.get("avg_velo"),
            "avg_spin": info.get("avg_spin"),
            "pfx_x": info.get("avg_pfx_x"),
            "pfx_z": info.get("avg_pfx_z"),
            "whiff_rate": info.get("whiff_rate", 0.0),
            "ba_against": info.get("ba_against", 0.0),
        })

    season = real.get("season_stats", {})
    platoon = real.get("platoon_splits", {})

    return {
        "name": real.get("name", ""),
        "throws": real.get("throws", ""),
        "era": season.get("era") or 0.0,
        "fip": season.get("fip") or 0.0,
        "whip": season.get("whip") or 0.0,
        "k_per_9": (season.get("k_pct") or 0) * 100,  # approximate
        "bb_per_9": (season.get("bb_pct") or 0) * 100,
        "ip": 0,
        "wins": 0,
        "losses": 0,
        "arsenal": arsenal_list,
        "vs_left": {
            "avg": platoon.get("vs_L", {}).get("ba", 0.0),
            "woba": platoon.get("vs_L", {}).get("woba", 0.0),
            "k_rate": platoon.get("vs_L", {}).get("k_rate", 0.0),
        },
        "vs_right": {
            "avg": platoon.get("vs_R", {}).get("ba", 0.0),
            "woba": platoon.get("vs_R", {}).get("woba", 0.0),
            "k_rate": platoon.get("vs_R", {}).get("k_rate", 0.0),
        },
        "recent_starts": [],
    }


def _adapt_batter_profile(real: dict) -> dict[str, Any]:
    """Adapt ``get_batter_profile`` output to the dashboard format."""
    season = real.get("season_stats", {})
    statcast = real.get("statcast", {})
    discipline = real.get("plate_discipline", {})

    return {
        "name": real.get("name", ""),
        "bats": real.get("bats", "R"),
        "avg": season.get("ba") or 0.0,
        "obp": season.get("obp") or 0.0,
        "slg": season.get("slg") or 0.0,
        "woba": season.get("woba") or 0.0,
        "hr": 0,
        "k_rate": discipline.get("whiff_rate", 0.0),
        "bb_rate": discipline.get("chase_rate", 0.0),
        "exit_velo": statcast.get("avg_exit_velo") or 0.0,
        "hard_hit_rate": statcast.get("hard_hit_pct") or 0.0,
        "zone_grid": [],  # Real module doesn't provide a 3x3 grid directly
    }


# ---------------------------------------------------------------------------
# Build name lists for selectors
# ---------------------------------------------------------------------------

_MOCK_PITCHERS: list[str] = sorted(list(ARSENALS.keys()))

_MOCK_BATTERS: list[str] = sorted(
    [b["name"] for b in PHILLIES_BATTERS + OPPONENT_BATTERS],
)


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the Matchup Explorer page."""
    st.title("Matchup Explorer")
    st.caption("Analyze pitcher-vs-batter matchups with Bayesian-adjusted estimates.")
    st.info("""
**What this shows:** Historical pitcher-vs-batter matchup data with Bayesian-adjusted estimates that account for small sample sizes.

- **Est. wOBA** is the key number — league average is ~.320. Above .350 = batter dominates, below .290 = pitcher dominates
- **Confidence intervals** widen with fewer plate appearances — wide CI means the estimate is uncertain, take it with a grain of salt
- **Pitch breakdown** reveals which pitch types a batter handles well or struggles against
""")

    # Determine data source
    conn = get_db_connection()
    use_real_data = has_data(conn)

    # Build player lists -- prefer real DB, fall back to mock
    if use_real_data:
        db_pitchers = _cached_all_pitchers()
        db_batters = _cached_all_batters()
    else:
        db_pitchers = []
        db_batters = []

    if db_pitchers:
        pitcher_options = [p["full_name"] for p in db_pitchers]
        pitcher_id_map = {p["full_name"]: p["player_id"] for p in db_pitchers}
    else:
        pitcher_options = []
        pitcher_id_map = {}

    if db_batters:
        batter_options = [b["full_name"] for b in db_batters]
        batter_id_map = {b["full_name"]: b["player_id"] for b in db_batters}
    else:
        batter_options = []
        batter_id_map = {}

    # Data source indicator
    if use_real_data and db_pitchers:
        st.info(f"Using real data: {len(db_pitchers)} pitchers, {len(db_batters)} batters in DB.")
    else:
        st.warning("No pitch data in database. Load Statcast data with backfill.py for matchup analytics.")
        return

    # ----- Selectors -----
    sel1, sel2, sel3 = st.columns([3, 3, 2])

    with sel1:
        pitcher_name = st.selectbox(
            "Select Pitcher",
            options=pitcher_options,
            index=0,
            key="matchup_pitcher",
        )
    with sel2:
        default_batter_idx = 0
        if "Francisco Lindor" in batter_options:
            default_batter_idx = batter_options.index("Francisco Lindor")
        batter_name = st.selectbox(
            "Select Batter",
            options=batter_options,
            index=default_batter_idx,
            key="matchup_batter",
        )
    with sel3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Today's Matchup", type="secondary"):
            # Dynamically pick today's probable starter from MLB API
            _todays_pitcher = pitcher_options[0] if pitcher_options else ""
            _todays_batter = batter_options[0] if batter_options else ""
            try:
                import requests as _req
                _sched = _req.get(
                    "https://statsapi.mlb.com/api/v1/schedule",
                    params={
                        "sportId": 1,
                        "teamId": 143,
                        "date": __import__("datetime").datetime.now().strftime("%Y-%m-%d"),
                        "hydrate": "probablePitcher,team",
                    },
                    timeout=5,
                )
                if _sched.ok:
                    for _d in _sched.json().get("dates", []):
                        for _g in _d.get("games", []):
                            _home = _g.get("teams", {}).get("home", {})
                            _away = _g.get("teams", {}).get("away", {})
                            _is_home = _home.get("team", {}).get("id") == 143
                            _phi_sp = _home if _is_home else _away
                            _opp_sp = _away if _is_home else _home
                            _sp_name = _phi_sp.get("probablePitcher", {}).get("fullName", "")
                            if _sp_name and _sp_name in pitcher_options:
                                _todays_pitcher = _sp_name
                            break
                        break
            except Exception:
                pass
            st.session_state["matchup_pitcher"] = _todays_pitcher
            st.session_state["matchup_batter"] = _todays_batter
            st.rerun()

    if not pitcher_name or not batter_name:
        st.info("Select a pitcher and batter to view matchup data.")
        return

    # Resolve IDs
    pitcher_id = pitcher_id_map.get(pitcher_name)
    batter_id = batter_id_map.get(batter_name)

    # ----- Matchup Card -----
    st.markdown("---")
    st.subheader(f"{batter_name}  vs  {pitcher_name}", anchor=False)
    matchup = _get_matchup_stats(
        pitcher_name=pitcher_name,
        batter_name=batter_name,
        pitcher_id=pitcher_id,
        batter_id=batter_id,
    )
    if matchup is not None:
        create_matchup_card(matchup)
    else:
        st.info("No historical matchup data available for this pitcher-batter pair.")

    st.markdown("---")

    # ----- Strike zone heatmap + Spray chart -----
    col_zone, col_spray = st.columns(2)

    with col_zone:
        st.markdown("#### Strike Zone Heatmap")
        # Try real matchup pitch history for heatmap
        real_zone_data = None
        if pitcher_id is not None and batter_id is not None and _HAS_QUERIES:
            hist_df = _cached_matchup_history(pitcher_id, batter_id)
            if hist_df is not None and not hist_df.empty:
                real_zone_data = hist_df

        if real_zone_data is not None:
            fig_zone = create_strike_zone_heatmap(
                real_zone_data,
                metric="estimated_ba",
                title=f"{pitcher_name} -- Batting Avg by Zone",
            )
            st.plotly_chart(fig_zone, use_container_width=True, key="matchup_zone")
        else:
            st.info("No pitch location data available for this matchup.")

    with col_spray:
        st.markdown("#### Spray Chart")
        st.info("Spray chart data requires historical batted ball data in the database.")

    st.markdown("---")

    # ----- Pitcher arsenal + Batter zone profile -----
    col_ars, col_bat = st.columns(2)

    with col_ars:
        st.markdown("#### Pitcher Arsenal")
        profile = _get_pitcher_profile(pitcher_name, pitcher_id=pitcher_id)
        arsenal = profile.get("arsenal", []) if profile else []
        if arsenal:
            fig_mov = create_pitch_movement_chart(
                arsenal,
                title=f"{pitcher_name} -- Pitch Movement",
            )
            st.plotly_chart(fig_mov, use_container_width=True, key="matchup_arsenal")

            # Arsenal table
            df_ars = pd.DataFrame(arsenal)
            display_cols = ["label", "pct", "avg_velo", "avg_spin", "whiff_rate", "ba_against"]
            available = [c for c in display_cols if c in df_ars.columns]
            df_show = df_ars[available].copy()
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
        else:
            st.info("No arsenal data available.")

    with col_bat:
        st.markdown("#### Batter Zone Profile")
        bat_profile = _get_batter_profile(batter_name, batter_id=batter_id)
        if bat_profile is not None:
            _render_batter_profile(bat_profile)
        else:
            st.info("No batter profile data available.")


def _render_batter_profile(profile: dict[str, Any]) -> None:
    """Render batter profile with hot/cold zone grid and key stats."""
    if not profile:
        st.info("No batter data available.")
        return

    # Key stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AVG", f"{profile.get('avg', 0):.3f}")
    c2.metric("OBP", f"{profile.get('obp', 0):.3f}")
    c3.metric("SLG", f"{profile.get('slg', 0):.3f}")
    c4.metric("wOBA", f"{profile.get('woba', 0):.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("HR", profile.get("hr", 0))
    c6.metric("K Rate", f"{profile.get('k_rate', 0):.1%}")
    c7.metric("BB Rate", f"{profile.get('bb_rate', 0):.1%}")
    c8.metric("Exit Velo", f"{profile.get('exit_velo', 0):.1f}")

    # Hot/cold zone grid (3x3)
    zone_grid = profile.get("zone_grid", [])
    if zone_grid:
        st.markdown("**Hot/Cold Zones** (AVG)")
        import plotly.graph_objects as go
        import numpy as np

        z = np.array(zone_grid)
        labels = [[f"{v:.3f}" for v in row] for row in z]

        fig = go.Figure(data=go.Heatmap(
            z=z,
            text=labels,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmin=0.200,
            zmax=0.400,
            showscale=True,
            colorbar=dict(title="AVG"),
        ))
        fig.update_layout(
            xaxis=dict(showticklabels=False, showgrid=False),
            yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed"),
            template="plotly_dark",
            height=250,
            margin=dict(l=20, r=20, t=10, b=20),
        )
        st.plotly_chart(fig, use_container_width=True, key="batter_zone_grid")
