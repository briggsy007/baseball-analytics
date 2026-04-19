"""
Diamond Analytics -- Main Streamlit Application.

Run with:
    streamlit run src/dashboard/app.py

Multipage app using the ``st.navigation`` / ``st.Page`` pattern
(Streamlit >= 1.36).  Falls back to a simple radio-button approach
for older versions.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that ``from src.…`` imports work
# regardless of the directory from which ``streamlit run`` is invoked.
# On Windows, normalise to a consistent case/slash style for the membership
# check so that duplicate entries are avoided.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Page-level configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Diamond Analytics",
    page_icon="\u26BE",   # baseball emoji
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Lazy page imports (deferred to avoid import-time Streamlit calls)
# ---------------------------------------------------------------------------
from src.dashboard.views import live_game, matchups, bullpen, anomalies, phillies, data_management, stuff_plus, sequencing  # noqa: E402
from src.dashboard.views import causal_war, sharpe_lineup, kinetic_half_life, mesi, volatility_surface, pset  # noqa: E402
from src.dashboard.views import alpha_decay, loft, pitch_decay, allostatic_load, viscoelastic_workload, baserunner_gravity  # noqa: E402
from src.dashboard.views import chemnet_view, pitchgpt_view, mechanix_ae, defensive_pressing  # noqa: E402
from src.dashboard.views import contrarian_leaderboards  # noqa: E402
from src.dashboard.db_helper import get_db_connection, has_data, get_data_status  # noqa: E402

# ---------------------------------------------------------------------------
# Sidebar helpers
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def _get_phillies_record():
    """Get real Phillies W-L record from MLB Stats API."""
    try:
        resp = requests.get(
            "https://statsapi.mlb.com/api/v1/standings",
            params={"leagueId": 104, "season": 2026},
            timeout=10,
        )
        if resp.ok:
            for div in resp.json().get("records", []):
                for team in div.get("teamRecords", []):
                    if team.get("team", {}).get("id") == 143:  # Phillies
                        return team.get("wins", 0), team.get("losses", 0)
    except Exception:
        pass
    return None, None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> str:
    """Render the sidebar and return the selected page name."""
    with st.sidebar:
        st.markdown(
            "<h2 style='text-align:center;'>\u26BE Diamond Analytics</h2>",
            unsafe_allow_html=True,
        )
        st.caption(f"Today: {datetime.now().strftime('%A, %B %d, %Y')}")
        st.markdown("---")

        # Quick team record from the MLB Stats API
        try:
            wins, losses = _get_phillies_record()
            if wins is not None and losses is not None:
                total = wins + losses
                pct = f".{int(wins / total * 1000):03d}" if total > 0 else ".000"
                st.markdown(
                    f"**Phillies** {wins}-{losses} ({pct})"
                )
            else:
                st.caption("Phillies record unavailable")
        except Exception:
            st.caption("Phillies record unavailable")

        # Next game from live feed API
        try:
            from src.ingest.live_feed import get_phillies_game
            game = get_phillies_game()
            if game:
                home_away = "vs" if game.get("home") == "PHI" else "@"
                opp = game.get("away") if game.get("home") == "PHI" else game.get("home")
                status = game.get("status", "Preview")
                start = game.get("start_time", "")
                if status == "Live":
                    st.markdown(f"**LIVE:** {home_away} {opp}")
                else:
                    st.markdown(f"Next: {home_away} {opp} | {start}")
            else:
                st.caption("No Phillies game today")
        except Exception:
            st.caption("Schedule unavailable")

        st.markdown("---")

        # ── Data status indicator ──
        conn = get_db_connection()
        status = get_data_status(conn)
        if status is not None:
            st.success(
                f"DB: {status['pitch_count']:,} pitches "
                f"({status['min_date']} to {status['max_date']})"
            )
        else:
            st.warning("DB: No data loaded. Run backfill.py to load data.")

        st.markdown("---")

        page = st.radio(
            "Navigation",
            options=[
                "Live Game",
                "Matchup Explorer",
                "Bullpen Strategy",
                "Anomaly Alerts",
                "Stuff+",
                "Sequencing",
                "Phillies Hub",
                "Data Management",
                "───────────────",
                "CausalWAR",
                "Contrarian Leaderboards",
                "Lineup Optimizer",
                "Pitch Decay (K½)",
                "Pitch Stability (MESI)",
                "Volatility Surface",
                "Sequence Threat (PSET)",
                "───────────────",
                "Alpha Decay",
                "LOFT (Order Flow)",
                "Pitch Decay (PDR)",
                "Batting Load (ABL)",
                "Arm Stress (VWR)",
                "Runner Gravity (BGI)",
                "───────────────",
                "ChemNet (Synergy)",
                "PitchGPT",
                "MechanixAE",
                "Defensive Pressing (DPI)",
            ],
            index=0,
            key="nav_page",
        )
    return page


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Application entry point."""
    selected_page = _render_sidebar()

    # Route to the selected page
    page_map = {
        "Live Game": live_game.render,
        "Matchup Explorer": matchups.render,
        "Bullpen Strategy": bullpen.render,
        "Anomaly Alerts": anomalies.render,
        "Stuff+": stuff_plus.render,
        "Sequencing": sequencing.render,
        "Phillies Hub": phillies.render,
        "Data Management": data_management.render,
        "CausalWAR": causal_war.render,
        "Contrarian Leaderboards": contrarian_leaderboards.render,
        "Lineup Optimizer": sharpe_lineup.render,
        "Pitch Decay (K½)": kinetic_half_life.render,
        "Pitch Stability (MESI)": mesi.render,
        "Volatility Surface": volatility_surface.render,
        "Sequence Threat (PSET)": pset.render,
        "Alpha Decay": alpha_decay.render,
        "LOFT (Order Flow)": loft.render,
        "Pitch Decay (PDR)": pitch_decay.render,
        "Batting Load (ABL)": allostatic_load.render,
        "Arm Stress (VWR)": viscoelastic_workload.render,
        "Runner Gravity (BGI)": baserunner_gravity.render,
        "ChemNet (Synergy)": chemnet_view.render,
        "PitchGPT": pitchgpt_view.render,
        "MechanixAE": mechanix_ae.render,
        "Defensive Pressing (DPI)": defensive_pressing.render,
    }

    render_fn = page_map.get(selected_page)
    if render_fn:
        render_fn()
    else:
        st.error(f"Unknown page: {selected_page}")


if __name__ == "__main__":
    main()
else:
    # When Streamlit executes the script directly (not via __main__)
    main()
