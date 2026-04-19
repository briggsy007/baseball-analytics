"""
Contrarian Leaderboards -- where CausalWAR most disagrees with bWAR.

Surfaces two product-grade lists derived from the
``causal_war_baseline_comparison_2023_2024.csv`` validation artifact:

1. **Buy-Low (CausalWAR > bWAR)** -- players the model loves more than the
   public WAR market does (positive ``rank_diff``).
2. **Over-Valued (bWAR > CausalWAR)** -- players whose public WAR depends
   on glove / park / sequencing the per-PA causal model cannot see
   (negative ``rank_diff``).

Each row is tagged with a methodology heuristic (RELIEVER LEVERAGE GAP /
PARK FACTOR / DEFENSE GAP / GENUINE EDGE? / OTHER) and a 2025 follow-up
outcome pulled from ``season_batting_stats`` / ``season_pitching_stats``.

Note on direction
-----------------
``rank_diff`` in the source CSV is computed as ``rank_trad - rank_causal``.
A *positive* rank_diff means CausalWAR ranks the player higher (better)
than bWAR does -- i.e. "Buy-Low". A *negative* rank_diff means bWAR ranks
the player higher than CausalWAR -- i.e. "Over-Valued". This view sorts
accordingly (Buy-Low: descending; Over-Valued: ascending).

See also: ``docs/edges/causal_war_contrarians_2024.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.db_helper import get_db_connection

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CSV_PATH = (
    Path(__file__).resolve().parents[3]
    / "results"
    / "validate_causal_war_20260418T194415Z"
    / "causal_war_baseline_comparison_2023_2024.csv"
)

_DEFENSE_FIRST_POSITIONS = {"SS", "CF", "C", "2B"}

_TAG_RELIEVER = "RELIEVER LEVERAGE GAP"
_TAG_PARK = "PARK FACTOR"
_TAG_DEFENSE = "DEFENSE GAP"
_TAG_GENUINE = "GENUINE EDGE?"
_TAG_OTHER = "OTHER"

_TOP_N = 25

_FOLLOWUP_SEASON = 2025

_PHILLIES_RED = "#E81828"
_PHILLIES_BLUE = "#002D72"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def _load_comparison_csv() -> pd.DataFrame | None:
    """Load the CausalWAR vs bWAR baseline comparison CSV."""
    if not _CSV_PATH.exists():
        return None
    df = pd.read_csv(_CSV_PATH)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def _load_position_lookup(_conn) -> dict[int, str]:
    """Map player_id -> primary fielding position from the players table."""
    if _conn is None:
        return {}
    try:
        rows = _conn.execute(
            "SELECT player_id, position FROM players WHERE position IS NOT NULL"
        ).fetchall()
        return {int(pid): str(pos) for pid, pos in rows if pos}
    except Exception:
        return {}


@st.cache_data(ttl=3600, show_spinner=False)
def _load_2025_outcomes(_conn, player_ids: tuple[int, ...]) -> dict[int, dict]:
    """Pull 2025 follow-up stats for the given player_ids.

    The ``war`` column for season=2025 is currently un-backfilled in the
    season_*_stats tables, so this function ALSO pulls surrogate signals
    (ERA / IP for pitchers, OPS / PA for batters) so the view has something
    useful to show.

    Returns a dict keyed by player_id with sub-keys:
        ``war`` (float | None), ``ip`` (float | None), ``pa`` (int | None),
        ``era`` (float | None), ``ops`` (float | None),
        ``source`` ('bat' | 'pit' | None).
    """
    out: dict[int, dict] = {}
    if _conn is None or not player_ids:
        return out

    ids_list = list({int(p) for p in player_ids})
    placeholders = ",".join(["?"] * len(ids_list))

    # Batters
    try:
        bat = _conn.execute(
            f"""
            SELECT player_id, war, pa, ops
            FROM season_batting_stats
            WHERE season = {_FOLLOWUP_SEASON}
              AND player_id IN ({placeholders})
            """,
            ids_list,
        ).fetchall()
        for pid, war, pa, ops in bat:
            out[int(pid)] = {
                "war": float(war) if war is not None else None,
                "pa": int(pa) if pa is not None else None,
                "ops": float(ops) if ops is not None else None,
                "ip": None,
                "era": None,
                "source": "bat",
            }
    except Exception:
        pass

    # Pitchers
    try:
        pit = _conn.execute(
            f"""
            SELECT player_id, war, ip, era
            FROM season_pitching_stats
            WHERE season = {_FOLLOWUP_SEASON}
              AND player_id IN ({placeholders})
            """,
            ids_list,
        ).fetchall()
        for pid, war, ip, era in pit:
            existing = out.get(int(pid))
            if existing is None:
                out[int(pid)] = {
                    "war": float(war) if war is not None else None,
                    "pa": None,
                    "ops": None,
                    "ip": float(ip) if ip is not None else None,
                    "era": float(era) if era is not None else None,
                    "source": "pit",
                }
            else:
                # Player has both rows (rare two-way case); prefer pitcher IP/ERA
                existing["ip"] = float(ip) if ip is not None else existing.get("ip")
                existing["era"] = float(era) if era is not None else existing.get("era")
                if existing.get("war") is None and war is not None:
                    existing["war"] = float(war)

    except Exception:
        pass

    return out


# ---------------------------------------------------------------------------
# Heuristic tagging
# ---------------------------------------------------------------------------


def _classify_row(row: pd.Series, position_lookup: dict[int, str]) -> str:
    """Apply the methodology-tag heuristics from the case study."""
    is_pitcher = row.get("position") == "pitcher"
    is_batter = row.get("position") == "batter"
    causal_war = float(row.get("causal_war", 0.0) or 0.0)
    trad_war = float(row.get("trad_war", 0.0) or 0.0)
    ip_total = row.get("ip_total")
    pa_total = row.get("pa_total")
    ip_total = float(ip_total) if pd.notna(ip_total) else None
    pa_total = float(pa_total) if pd.notna(pa_total) else None
    field_pos = position_lookup.get(int(row["player_id"]), None)

    if is_pitcher:
        if ip_total is not None and ip_total < 60 and causal_war > trad_war:
            return _TAG_RELIEVER
        if ip_total is not None and ip_total >= 60 and trad_war > causal_war:
            return _TAG_PARK

    if is_batter:
        if (
            field_pos in _DEFENSE_FIRST_POSITIONS
            and trad_war > causal_war
        ):
            return _TAG_DEFENSE
        if (
            pa_total is not None
            and pa_total >= 400
            and causal_war > trad_war
        ):
            return _TAG_GENUINE

    return _TAG_OTHER


# ---------------------------------------------------------------------------
# 2025 outcome formatting
# ---------------------------------------------------------------------------


def _format_2025(row: pd.Series, outcomes: dict[int, dict]) -> str:
    """Return a short human string describing the 2025 follow-up.

    Prefers WAR delta when WAR is populated; otherwise falls back to
    ERA + IP for pitchers and OPS + PA for batters (the season WAR column
    for 2025 is not yet backfilled in the season_*_stats tables).
    """
    info = outcomes.get(int(row["player_id"]))
    if info is None:
        return "-"
    war_2025 = info.get("war")

    if war_2025 is not None:
        war_2024 = float(row.get("trad_war", 0.0) or 0.0)
        # 2024 trad_war in the CSV is the 2-year aggregate. Normalise to
        # per-year by halving so the delta read is sensible.
        war_2024_per_yr = war_2024 / 2.0
        delta = war_2025 - war_2024_per_yr
        arrow = "up" if delta >= 0 else "down"
        label = "improved" if delta >= 0.5 else "regressed" if delta <= -0.5 else "flat"
        return f"{arrow} {war_2025:+.1f} WAR ({label})"

    # Fallback: surrogate signals
    if info.get("source") == "pit" or info.get("ip") is not None:
        ip = info.get("ip")
        era = info.get("era")
        if ip is not None and era is not None:
            return f"{era:.2f} ERA / {ip:.0f} IP"
        if ip is not None:
            return f"{ip:.0f} IP"
        return "no 2025"
    pa = info.get("pa")
    ops = info.get("ops")
    if pa is not None and ops is not None:
        return f"{ops:.3f} OPS / {pa} PA"
    if pa is not None:
        return f"{pa} PA"
    return "no 2025"


# ---------------------------------------------------------------------------
# Hit-rate KPI
# ---------------------------------------------------------------------------


def _compute_hit_rate(
    buy_low_df: pd.DataFrame,
    outcomes: dict[int, dict],
) -> tuple[int, int, float | None]:
    """% of CausalWAR-loved 2024 picks who held up in 2025.

    Preferred signal: 2025 WAR >= per-year 2024 trad_war (CausalWAR's edge
    held). Fallback signals (used because the 2025 WAR column is not yet
    backfilled in season_*_stats):
      - Pitcher: 2025 ERA <= 4.00 AND IP >= 30 -> hit
      - Batter:  2025 OPS >= 0.700 AND PA >= 100 -> hit
    """
    hits = 0
    n = 0
    for _, row in buy_low_df.iterrows():
        info = outcomes.get(int(row["player_id"]))
        if info is None:
            continue

        war_2025 = info.get("war")
        if war_2025 is not None:
            n += 1
            war_2024_per_yr = float(row.get("trad_war", 0.0) or 0.0) / 2.0
            if war_2025 >= war_2024_per_yr:
                hits += 1
            continue

        # Fallback when WAR is null
        if row.get("position") == "pitcher":
            era = info.get("era")
            ip = info.get("ip")
            if era is not None and ip is not None and ip >= 30:
                n += 1
                if era <= 4.00:
                    hits += 1
        elif row.get("position") == "batter":
            ops = info.get("ops")
            pa = info.get("pa")
            if ops is not None and pa is not None and pa >= 100:
                n += 1
                if ops >= 0.700:
                    hits += 1

    if n == 0:
        return 0, 0, None
    return hits, n, hits / n


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------


def _build_display_df(
    df: pd.DataFrame,
    position_lookup: dict[int, str],
    outcomes: dict[int, dict],
) -> pd.DataFrame:
    """Build the on-screen DataFrame with derived columns."""
    df = df.copy()
    df["tag"] = df.apply(lambda r: _classify_row(r, position_lookup), axis=1)
    df["outcome_2025"] = df.apply(lambda r: _format_2025(r, outcomes), axis=1)

    # PA / IP combined display column
    def _pa_ip(row):
        if row.get("position") == "pitcher":
            ip = row.get("ip_total")
            return f"{ip:.1f} IP" if pd.notna(ip) else "-"
        pa = row.get("pa_total")
        return f"{int(pa)} PA" if pd.notna(pa) else "-"

    df["volume"] = df.apply(_pa_ip, axis=1)

    # Field position (if known)
    df["field_pos"] = df["player_id"].map(
        lambda pid: position_lookup.get(int(pid), "-")
    )

    return df


def _render_table(df: pd.DataFrame, key: str) -> None:
    """Render the standardised contrarian-leaderboard table."""
    show = df[
        [
            "name",
            "field_pos",
            "volume",
            "causal_war",
            "trad_war",
            "rank_diff",
            "tag",
            "outcome_2025",
        ]
    ].copy()
    show = show.reset_index(drop=True)
    show.index = show.index + 1
    show.index.name = "Rank"

    st.dataframe(
        show,
        use_container_width=True,
        column_config={
            "name": st.column_config.TextColumn("Player"),
            "field_pos": st.column_config.TextColumn("Pos", width="small"),
            "volume": st.column_config.TextColumn("Volume"),
            "causal_war": st.column_config.NumberColumn("CausalWAR", format="%.2f"),
            "trad_war": st.column_config.NumberColumn("bWAR", format="%.2f"),
            "rank_diff": st.column_config.NumberColumn("Rank Diff", format="%+d"),
            "tag": st.column_config.TextColumn("Methodology Tag"),
            "outcome_2025": st.column_config.TextColumn("2025 Outcome"),
        },
        height=min(900, 38 * (len(show) + 1) + 3),
        key=f"contrarian_table_{key}",
    )


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Contrarian Leaderboards page."""
    st.title("Contrarian Leaderboards")
    st.caption(
        "Where my CausalWAR most disagrees with Baseball-Reference WAR -- "
        "and what 2025 said."
    )

    df = _load_comparison_csv()
    if df is None or df.empty:
        st.error(
            f"Comparison CSV not found at `{_CSV_PATH}`. "
            "Re-run the CausalWAR validation suite first."
        )
        return

    conn = get_db_connection()
    position_lookup = _load_position_lookup(conn)

    # Position / search filters --------------------------------------------
    fcol1, fcol2 = st.columns([1, 3])
    with fcol1:
        pos_choice = st.selectbox(
            "Player type",
            options=["All", "Batters only", "Pitchers only"],
            index=0,
            key="contrarian_pos_filter",
        )
    with fcol2:
        search = st.text_input(
            "Search by name (substring, case-insensitive)",
            value="",
            key="contrarian_name_search",
        )

    filtered = df.copy()
    if pos_choice == "Batters only":
        filtered = filtered[filtered["position"] == "batter"]
    elif pos_choice == "Pitchers only":
        filtered = filtered[filtered["position"] == "pitcher"]
    if search.strip():
        needle = search.strip().lower()
        filtered = filtered[
            filtered["name"].fillna("").str.lower().str.contains(needle, regex=False)
        ]

    if filtered.empty:
        st.info("No players match the current filters.")
        return

    # Build Buy-Low and Over-Valued slices ---------------------------------
    # rank_diff > 0 -> CausalWAR ranks higher than bWAR (model loves it more)
    buy_low = filtered.sort_values("rank_diff", ascending=False).head(_TOP_N).copy()
    over_valued = filtered.sort_values("rank_diff", ascending=True).head(_TOP_N).copy()

    # 2025 outcomes (single batched lookup over both boards) ---------------
    all_ids = tuple(
        sorted(
            set(buy_low["player_id"].astype(int).tolist())
            | set(over_valued["player_id"].astype(int).tolist())
        )
    )
    outcomes = _load_2025_outcomes(conn, all_ids)

    # Decorate with tag + outcome columns ----------------------------------
    buy_low_disp = _build_display_df(buy_low, position_lookup, outcomes)
    over_valued_disp = _build_display_df(over_valued, position_lookup, outcomes)

    # KPIs -----------------------------------------------------------------
    hits, n_eval, hit_rate = _compute_hit_rate(buy_low_disp, outcomes)
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Buy-Low entries", len(buy_low_disp))
    kpi2.metric("Over-Valued entries", len(over_valued_disp))
    if hit_rate is not None:
        kpi3.metric(
            "Buy-Low 2025 hit rate",
            f"{hit_rate*100:.0f}%",
            help=(
                f"{hits}/{n_eval} CausalWAR-loved 2024 picks held up in 2025. "
                "Prefers WAR delta when populated; falls back to ERA<=4.00 "
                "(IP>=30) for pitchers and OPS>=0.700 (PA>=100) for batters "
                "since 2025 season WAR is not yet backfilled."
            ),
        )
    else:
        kpi3.metric("Buy-Low 2025 hit rate", "N/A")

    st.markdown("---")

    # Two stacked tables ---------------------------------------------------
    st.subheader("Buy-Low (CausalWAR > bWAR)")
    st.caption(
        "Top "
        f"{_TOP_N} players where my model ranks them higher than the public "
        "WAR market. Sorted by ``rank_diff`` descending (most CausalWAR-favoured "
        "first)."
    )
    _render_table(buy_low_disp, key="buylow")

    st.markdown("")

    st.subheader("Over-Valued (bWAR > CausalWAR)")
    st.caption(
        "Top "
        f"{_TOP_N} players where bWAR ranks them higher than CausalWAR -- "
        "their value likely leans on glove / park / sequencing the per-PA "
        "model cannot see. Sorted by ``rank_diff`` ascending (most bWAR-favoured "
        "first)."
    )
    _render_table(over_valued_disp, key="overvalued")

    st.markdown("---")

    # Tag distribution preview ---------------------------------------------
    with st.expander("Methodology tag distribution"):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Buy-Low tags**")
            st.dataframe(
                buy_low_disp["tag"].value_counts().rename_axis("Tag").reset_index(name="Count"),
                hide_index=True,
                use_container_width=True,
            )
        with col_b:
            st.markdown("**Over-Valued tags**")
            st.dataframe(
                over_valued_disp["tag"].value_counts().rename_axis("Tag").reset_index(name="Count"),
                hide_index=True,
                use_container_width=True,
            )

    # Methodology blurb ----------------------------------------------------
    st.markdown(
        """
**Methodology.** CausalWAR is a Double-ML offensive-run-value estimator at the
per-PA grain that residualises out venue, platoon, base-out state, and inning
context but is blind to defense, leverage, and continuous park run-environment.
bWAR includes all three, so the most extreme rank disagreements concentrate on
exactly those blind spots: the model under-rates park-friendly starters and
glove-first shortstops, and over-rates stuff-driven middle relievers whose
ERA-based value bWAR discounts via leverage chaining. Tag heuristics encode
those gaps; the deep-dive lives in
``docs/edges/causal_war_contrarians_2024.md``.
        """.strip()
    )
