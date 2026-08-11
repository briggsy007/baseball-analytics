"""
Defensive Pressing Intensity (DPI) dashboard view.

Displays team-level defensive efficiency metrics:
- Team DPI big-number with league rank, regressed DPI & consistency score
- BIP outcome chart: launch_speed vs launch_angle colored by actual vs expected
- Team leaderboard: all teams ranked by DPI (raw + regressed) with bar chart
- Extra-base prevention: team ranking on limiting advancement
- Game-by-game DPI: timeline chart across the season

2026-08-10 (WS3.6, plan docs/plans/2026-08-10_platform_improvement_plan.md):
every number quoted as evidence on this page comes from the claims registry
via ``src.claims.get_claim`` -- no hand-copied metrics, and no unregistered
external-literature figures either (a SABR BABIP variance-decomposition
parenthetical was dropped for that reason on review: it carried no registry
entry, and its fielding term read as a statement about DPI's own signal
composition). The unsourced
runs-saved impact line and the fielder-level mechanism copy (audit DPI
finding 10) were deleted; DPI is presented as a team BIP-conversion residual.
The exact deleted strings are pinned as banned in
``tests/test_defensive_pressing_view.py``.
The "pressing"/positioning name survives kill criterion K1 (which did not
fire) but ships with the positioning caveat: BIP-level alignment signal is
real, team-level positioning ranking is not reliable
(claim dpi_positioning_alignment_ab).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.claims import ClaimError, get_claim
from src.dashboard.db_helper import get_db_connection, has_data

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from src.analytics.defensive_pressing import (
    DPIConfig,
    DefensivePressingModel,
    train_expected_out_model,
    compute_expected_outs,
    calculate_game_dpi,
    calculate_team_dpi,
    batch_calculate,
    get_player_dpi,
    get_team_game_dpi_timeline,
    build_bip_features,
    compute_spray_angle,
)

_CACHE_AVAILABLE = False
try:
    from src.dashboard.cache_reader import get_cached_leaderboard, cache_age_display
    _CACHE_AVAILABLE = True
except Exception:
    pass

# Phillies red / blue palette
_PHILLIES_RED = "#E81828"
_PHILLIES_BLUE = "#002D72"
_PHILLIES_LIGHT = "#B0B7BC"
_POSITIVE_GREEN = "#2ECC71"
_NEGATIVE_RED = "#E74C3C"
_NEUTRAL_GOLD = "#FFC145"

# Batch-A (WS3.1) split-half reliability artifact: carries per-season
# reliability_R and regressed_dpi for 2015-2025 team-seasons.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RELIABILITY_CSV = (
    _REPO_ROOT / "results" / "defensive_pressing" / "reliability_2026-08-10"
    / "team_season_dpi_2015_2025.csv"
)

# Claims-registry ids rendered in the evidence panel, defensible core first.
_DPI_EVIDENCE_CLAIMS = (
    "dpi_v2_partial_r_oaa_given_babip",
    "dpi_positioning_alignment_ab",
    "dpi_split_half_reliability",
    "dpi_yoy_stability",
    "dpi_pitching_strip_variance_share",
    "dpi_oaa_2025_r",
    "dpi_gate6_pooled",
)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _cached_dpi_timeline(team_id: str, season: int) -> pd.DataFrame | None:
    """Cached DPI timeline for a team/season."""
    conn = get_db_connection()
    return get_team_game_dpi_timeline(conn, team_id, season)


def render() -> None:
    """Render the Defensive Pressing Intensity (DPI) Analysis page."""
    st.title("Defensive Pressing Intensity (DPI)")
    st.caption(
        "Team BIP-conversion residual: how many more (or fewer) balls in play "
        "a defense turns into outs than a batted-ball model expects on the "
        "same batted balls (HistGradientBoosting on exit velocity, launch "
        "angle, spray angle, batted-ball type). The 'pressing' name survives "
        "the 2026-08 positioning A/B test, with the caveat below — DPI is a "
        "team-season outcome residual, not a fielder-tracking metric."
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**DPI measures how well a defense converts batted balls into outs** compared to what the batted-ball model expects on the same batted balls.

- **Positive DPI** = more outs than the model expected on those batted balls
- **DPI near 0** = league-average conversion
- **Negative DPI** = fewer outs than expected
- **What DPI cannot tell you:** which fielder, or which mechanism. Range, positioning, transition speed, pitcher contact management, park and luck are not separated inside a team-season outcome residual. The share of it that is fielding rather than pitching or luck is not measured on this page; the one decomposition we did run is the pitching-strip variance share in Evidence & caveats.
- **Rank on regressed DPI, not raw.** Season DPI is noisy enough that raw leaderboard gaps overstate true team separation; the regressed column shrinks each team toward the league mean by the measured split-half reliability (see Evidence & caveats).
- **Consistency** is the inverse of game-DPI variance — a dispersion descriptor, not a validated skill.
- **Extra-base prevention** is the raw share of hits held to singles. There is no expected-XBH model behind it, so it is not batted-ball- or park-adjusted.
""")

    _render_evidence_panel()

    conn = get_db_connection()

    if conn is None or not has_data(conn):
        st.warning(
            "No pitch data available. Run the data backfill pipeline first "
            "(`python scripts/backfill.py`)."
        )
        return

    # ---- Sidebar controls ------------------------------------------------
    with st.sidebar:
        st.markdown("### DPI Options")

        season = st.selectbox(
            "Season",
            options=_get_available_seasons(),
            key="dpi_season",
        )

    # ---- Load leaderboard ------------------------------------------------
    leaderboard = _load_leaderboard(conn, season)

    if leaderboard is None or leaderboard.empty:
        st.info(
            "No DPI data available for this season. Ensure enough batted-ball "
            "data is loaded."
        )
        return

    _render_artifact_provenance(leaderboard, season)

    # Regressed DPI alongside raw everywhere the leaderboard renders.
    leaderboard, reliability, reliability_note = _with_regressed_dpi(
        leaderboard, season,
    )

    # ---- Team selector ---------------------------------------------------
    team_options = leaderboard["team_id"].tolist()
    default_idx = team_options.index("PHI") if "PHI" in team_options else 0
    selected_team = st.selectbox(
        "Select Team",
        options=team_options,
        index=default_idx,
        key="dpi_team_select",
    )

    # ---- Tabs ------------------------------------------------------------
    tab_overview, tab_scatter, tab_board, tab_ebp, tab_timeline = st.tabs([
        "Team DPI",
        "BIP Outcome Chart",
        "Team Leaderboard",
        "Extra-Base Prevention",
        "Game-by-Game",
    ])

    with tab_overview:
        _render_team_overview(
            conn, selected_team, season, leaderboard,
            reliability, reliability_note,
        )

    with tab_scatter:
        _render_bip_scatter(conn, selected_team, season)

    with tab_board:
        _render_leaderboard(leaderboard, season, reliability, reliability_note)

    with tab_ebp:
        _render_extra_base_prevention(leaderboard)

    with tab_timeline:
        _render_timeline(conn, selected_team, season)


# ---------------------------------------------------------------------------
# Evidence panel (WS3.6: every quoted number resolves through the registry)
# ---------------------------------------------------------------------------

def _render_evidence_panel() -> None:
    """Render the DPI claims-registry entries, defensible core first.

    Nothing here is hand-copied: values, CIs and caveats all come from
    ``docs/claims/claims.yaml`` through :func:`src.claims.get_claim`, so a
    retracted or narrowed claim cannot silently keep rendering (K6).
    """
    with st.expander("Evidence & caveats (claims registry)"):
        st.caption(
            "Values, confidence intervals and caveats are read live from "
            "`docs/claims/claims.yaml`. A retracted claim cannot render here."
        )
        for claim_id in _DPI_EVIDENCE_CLAIMS:
            try:
                claim = get_claim(claim_id)
            except ClaimError as exc:
                st.warning(f"`{claim_id}` unavailable: {exc}")
                continue
            st.markdown(f"**{claim.metric}** *(claim:{claim.id}, {claim.status})*")
            if isinstance(claim.value, dict):
                st.markdown(
                    "\n".join(f"- `{k}`: {v}" for k, v in claim.value.items())
                )
            else:
                st.markdown(f"- value: {claim.value}")
            if claim.ci:
                st.markdown(f"- CI: {claim.ci}")
            st.caption(claim.caveat)
            if claim.source_doc:
                st.caption(f"Source: `{claim.source_doc}`")
            st.markdown("---")


# ---------------------------------------------------------------------------
# Regressed DPI (WS3.1/WS3.6: rank on the shrunk value, not the raw one)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _season_reliability(season: int) -> tuple[float | None, str]:
    """Return ``(reliability_R, provenance_note)`` for one season.

    ``R`` is the Spearman-Brown split-half reliability recorded per season by
    the WS3.1 artifact ``team_season_dpi_2015_2025.csv`` (full seasons carry
    the Fisher-z mean, 2020 carries its own 60-game estimate). Seasons absent
    from that artifact -- in-season 2026, most importantly -- return
    ``(None, reason)``: applying a full-season reliability to a partial
    sample would overstate how much of the spread is signal.
    """
    if not _RELIABILITY_CSV.exists():
        return None, (
            f"reliability artifact not found at `{_RELIABILITY_CSV.name}` "
            "(run `python scripts/dpi_reliability_2026.py`)"
        )
    try:
        rel = pd.read_csv(
            _RELIABILITY_CSV, usecols=["season", "reliability_R"],
        )
    except Exception as exc:  # noqa: BLE001 -- dashboard must not hard-fail
        return None, f"could not read the reliability artifact: {exc}"
    row = rel[rel["season"] == int(season)]
    if row.empty or row["reliability_R"].isna().all():
        return None, (
            f"no published split-half reliability for {season} — the WS3.1 "
            "artifact covers full seasons 2015-2025 only, and applying a "
            "full-season reliability to a partial in-season sample would "
            "overstate true team separation"
        )
    return float(row["reliability_R"].iloc[0]), (
        f"split-half reliability for {season} from "
        f"`results/defensive_pressing/reliability_2026-08-10/"
        f"{_RELIABILITY_CSV.name}`"
    )


def _with_regressed_dpi(
    leaderboard: pd.DataFrame, season: int,
) -> tuple[pd.DataFrame, float | None, str]:
    """Attach a ``regressed_dpi`` column shrunk toward the league mean.

    ``regressed = league_mean + R * (dpi_mean - league_mean)``. The league
    mean is taken from the displayed leaderboard itself so the shrunk value
    shares a scoring artifact with the raw one (the artifact that produced
    the displayed numbers is named in the provenance caption); only ``R``
    comes from the WS3.1 split-half artifact.
    """
    reliability, note = _season_reliability(season)
    if reliability is None or "dpi_mean" not in leaderboard.columns:
        return leaderboard, None, note
    df = leaderboard.copy()
    league_mean = float(df["dpi_mean"].mean())
    df["regressed_dpi"] = (
        league_mean + reliability * (df["dpi_mean"] - league_mean)
    ).round(4)
    return df, reliability, note


def _render_regression_caption(
    reliability: float | None, note: str, season: int | None,
) -> None:
    """One-line statement of the shrinkage applied (or why it was deferred)."""
    label = str(season) if season else "this season"
    if reliability is None:
        st.caption(f"Regressed DPI not shown for {label}: {note}.")
        return
    try:
        claim = get_claim("dpi_split_half_reliability")
        caveat = claim.caveat
    except ClaimError:
        caveat = (
            "season DPI is quoted regressed because raw leaderboard gaps "
            "overstate true team separation"
        )
    st.caption(
        f"Regressed DPI shrinks each team toward the league mean by the "
        f"measured reliability R = {reliability:.3f} ({note}). {caveat}"
    )


# ---------------------------------------------------------------------------
# Artifact provenance (WS0.1: state which xOut artifact scored each season)
# ---------------------------------------------------------------------------

def _render_artifact_provenance(leaderboard: pd.DataFrame, season: int) -> None:
    """State which xOut artifact scored this season and whether in-sample.

    The frozen validated checkpoint (``xout_v1.pkl``, train 2015-2022) is
    out-of-sample for 2023+; the in-season retrain (``xout_2026_inseason.pkl``,
    train 2015-2026) is in-sample for every displayed season.
    """
    if "scoring_artifact" in leaderboard.columns:
        artifact = str(leaderboard["scoring_artifact"].iloc[0])
        seasons_label = str(
            leaderboard.get(
                "artifact_train_seasons", pd.Series(["unknown"])
            ).iloc[0]
        )
        in_sample = leaderboard.get("scored_in_sample", pd.Series([None])).iloc[0]
        if in_sample is None or pd.isna(in_sample):
            sample_note = "in-/out-of-sample status unknown"
        elif bool(in_sample):
            sample_note = "IN-SAMPLE (season inside the train window)"
        else:
            sample_note = "out-of-sample"
        st.caption(
            f"Scoring artifact: `{artifact}` (xOut train seasons "
            f"{seasons_label}) — {season} scores are **{sample_note}**. "
            f"Validation gates were scored with the frozen `xout_v1.pkl` "
            f"(train 2015-2022, OOS for 2023+), not necessarily this artifact."
        )
    else:
        st.caption(
            "Scoring artifact: unknown (cache predates provenance stamping, "
            "2026-08-10). Nightly-scored seasons 2023-2026 were produced by "
            "an in-season xOut retrain (train 2015-2026) and are IN-SAMPLE; "
            "only the frozen `xout_v1.pkl` (train 2015-2022) gives OOS scores "
            "for 2023+."
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def _get_available_seasons() -> list[int]:
    """Return list of seasons with pitch data."""
    conn = get_db_connection()
    if conn is None:
        return [2025]
    try:
        result = conn.execute(
            "SELECT DISTINCT EXTRACT(YEAR FROM game_date)::INTEGER AS season "
            "FROM pitches ORDER BY season DESC"
        ).fetchdf()
        seasons = result["season"].tolist()
        return seasons if seasons else [2025]
    except Exception:
        return [2025]


def _load_leaderboard(conn, season: int) -> pd.DataFrame | None:
    """Load DPI leaderboard, using cache then live computation."""
    # Try cache first
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "defensive_pressing", season)
            if cached is not None:
                age_info = cache_age_display(conn, "defensive_pressing", season)
                if age_info:
                    st.caption(age_info)
                return cached
        except Exception:
            pass

    try:
        with st.spinner("Computing... Run `python scripts/precompute.py` for instant loading."):
            return batch_calculate(conn, season)
    except Exception as exc:
        st.error(f"Error computing DPI leaderboard: {exc}")
        return None


# ---------------------------------------------------------------------------
# Tab: Team DPI Overview
# ---------------------------------------------------------------------------

def _render_team_overview(
    conn, team_id: str, season: int, leaderboard: pd.DataFrame,
    reliability: float | None = None, reliability_note: str = "",
) -> None:
    """Big-number DPI card with league rank, regressed DPI and consistency."""
    st.subheader(f"{team_id} Defensive Pressing Intensity")

    team_row = leaderboard[leaderboard["team_id"] == team_id]
    if team_row.empty:
        st.warning(f"No DPI data for {team_id}")
        return

    row = team_row.iloc[0]
    dpi_mean = row["dpi_mean"]
    rank = int(row["rank"])
    n_teams = len(leaderboard)
    consistency = row.get("consistency", 0)
    n_games = int(row.get("n_games", 0))
    percentile = row.get("percentile", 0)
    regressed = row.get("regressed_dpi")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        color = _POSITIVE_GREEN if dpi_mean > 0 else _NEGATIVE_RED
        st.metric(
            "DPI raw (avg per game)",
            f"{dpi_mean:+.3f}",
            delta=f"Rank #{rank} of {n_teams}",
        )

    with col2:
        st.metric(
            "DPI regressed",
            f"{regressed:+.3f}" if regressed is not None and pd.notna(regressed)
            else "n/a",
        )

    with col3:
        st.metric("Percentile", f"{percentile:.0f}th")

    with col4:
        st.metric("Consistency", f"{consistency:.3f}")

    with col5:
        st.metric("Games", str(n_games))

    _render_regression_caption(reliability, reliability_note, season)

    # Gauge chart for DPI
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=dpi_mean,
        title={"text": "DPI (Outs Above Expected / Game)"},
        delta={"reference": 0, "increasing": {"color": _POSITIVE_GREEN},
               "decreasing": {"color": _NEGATIVE_RED}},
        gauge={
            "axis": {"range": [-5, 5]},
            "bar": {"color": _PHILLIES_RED if dpi_mean > 0 else _NEGATIVE_RED},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [-5, -1], "color": "rgba(231,76,60,0.2)"},
                {"range": [-1, 1], "color": "rgba(255,193,69,0.2)"},
                {"range": [1, 5], "color": "rgba(46,204,113,0.2)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8,
                "value": 0,
            },
        },
    ))
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(t=50, b=20, l=30, r=30),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: BIP Outcome Chart
# ---------------------------------------------------------------------------

def _render_bip_scatter(conn, team_id: str, season: int) -> None:
    """Scatter of launch_speed vs launch_angle, colored by outcome vs expected."""
    st.subheader("Batted Ball Outcomes vs Expected")

    try:
        scatter_df = _load_bip_data(team_id, season)
    except Exception as exc:
        st.error(f"Error loading BIP data: {exc}")
        return

    if scatter_df is None or scatter_df.empty:
        st.info("No batted-ball data available for this team/season.")
        return

    fig = go.Figure()

    # Split by outcome
    outs = scatter_df[scatter_df["actual_out"] == 1]
    hits = scatter_df[scatter_df["actual_out"] == 0]

    fig.add_trace(go.Scatter(
        x=outs["launch_speed"],
        y=outs["launch_angle"],
        mode="markers",
        name="Out Made",
        marker=dict(
            color=outs["expected_out_prob"],
            colorscale="RdYlGn",
            cmin=0, cmax=1,
            size=6,
            opacity=0.7,
            symbol="circle",
            colorbar=dict(title="xOut Prob", x=1.05),
        ),
        text=[f"xOut: {p:.2f}" for p in outs["expected_out_prob"]],
        hovertemplate=(
            "EV: %{x:.1f} mph<br>"
            "LA: %{y:.1f} deg<br>"
            "%{text}<br>"
            "Result: Out<extra></extra>"
        ),
    ))

    fig.add_trace(go.Scatter(
        x=hits["launch_speed"],
        y=hits["launch_angle"],
        mode="markers",
        name="Hit Allowed",
        marker=dict(
            color=_NEGATIVE_RED,
            size=7,
            opacity=0.8,
            symbol="x",
        ),
        hovertemplate=(
            "EV: %{x:.1f} mph<br>"
            "LA: %{y:.1f} deg<br>"
            "Result: Hit<extra></extra>"
        ),
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Exit Velocity (mph)",
        yaxis_title="Launch Angle (deg)",
        height=500,
        margin=dict(t=30, b=40, l=50, r=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Each dot is a batted ball in play against this team's defense. "
        "Color intensity shows expected out probability (green = easy out, "
        "red = likely hit). X marks are hits allowed."
    )


@st.cache_data(ttl=3600)
def _load_bip_data(team_id: str, season: int) -> pd.DataFrame:
    """Load BIP data with expected out probabilities for the scatter chart."""
    conn = get_db_connection()
    query = """
        SELECT
            launch_speed, launch_angle, hc_x, hc_y, bb_type, events
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) = $1
          AND type = 'X'
          AND events IS NOT NULL
          AND launch_speed IS NOT NULL
          AND launch_angle IS NOT NULL
          AND hc_x IS NOT NULL
          AND hc_y IS NOT NULL
          AND bb_type IS NOT NULL
          AND (
              (home_team = $2 AND inning_topbot = 'Top')
              OR (away_team = $2 AND inning_topbot = 'Bot')
          )
    """
    df = conn.execute(query, [season, team_id]).fetchdf()

    if df.empty:
        return pd.DataFrame()

    from src.analytics.defensive_pressing import _is_out

    xout = compute_expected_outs(df)
    actual = _is_out(df["events"])

    return pd.DataFrame({
        "launch_speed": df["launch_speed"],
        "launch_angle": df["launch_angle"],
        "expected_out_prob": xout,
        "actual_out": actual,
    })


# ---------------------------------------------------------------------------
# Tab: Team Leaderboard
# ---------------------------------------------------------------------------

def _render_leaderboard(
    leaderboard: pd.DataFrame, season: int | None = None,
    reliability: float | None = None, reliability_note: str = "",
) -> None:
    """All teams ranked by DPI (raw bars + regressed markers) with a table."""
    st.subheader("Team DPI Leaderboard")

    if leaderboard.empty:
        st.info("No leaderboard data.")
        return

    # Bar chart
    sorted_df = leaderboard.sort_values("dpi_mean", ascending=True)

    colors = [
        _POSITIVE_GREEN if v > 0 else _NEGATIVE_RED
        for v in sorted_df["dpi_mean"]
    ]

    fig = go.Figure(go.Bar(
        x=sorted_df["dpi_mean"],
        y=sorted_df["team_id"],
        orientation="h",
        marker_color=colors,
        name="DPI raw",
        text=[f"{v:+.3f}" for v in sorted_df["dpi_mean"]],
        textposition="outside",
        hovertemplate=(
            "%{y}<br>"
            "DPI raw: %{x:+.3f}<br>"
            "<extra></extra>"
        ),
    ))

    if "regressed_dpi" in sorted_df.columns:
        fig.add_trace(go.Scatter(
            x=sorted_df["regressed_dpi"],
            y=sorted_df["team_id"],
            mode="markers",
            name="DPI regressed",
            marker=dict(
                color="white", symbol="diamond", size=9,
                line=dict(color=_PHILLIES_BLUE, width=1),
            ),
            hovertemplate=(
                "%{y}<br>"
                "DPI regressed: %{x:+.3f}<br>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="DPI (Outs Above Expected / Game)",
        yaxis_title="",
        height=max(400, len(sorted_df) * 28),
        margin=dict(t=30, b=40, l=60, r=60),
        xaxis=dict(zeroline=True, zerolinecolor="white", zerolinewidth=1),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    _render_regression_caption(reliability, reliability_note, season)

    # Data table
    display_cols = [
        "rank", "team_id", "dpi_mean", "regressed_dpi", "dpi_total",
        "consistency", "extra_base_prevention", "n_games", "percentile",
    ]
    available = [c for c in display_cols if c in leaderboard.columns]
    st.dataframe(
        leaderboard[available].reset_index(drop=True),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Tab: Extra-Base Prevention
# ---------------------------------------------------------------------------

def _render_extra_base_prevention(leaderboard: pd.DataFrame) -> None:
    """Team ranking on limiting advancement (extra-base prevention rate)."""
    st.subheader("Extra-Base Prevention")
    st.caption(
        "Raw fraction of hits allowed that were kept to singles "
        "(1 - XBH share). Higher = fewer doubles, triples and home runs "
        "among the hits allowed."
    )
    st.caption(
        "Descriptive only (audit DPI finding 10): there is no expected-XBH "
        "model behind this column, so it is not adjusted for batted-ball "
        "mix, park or opposing hitters, it is not validated against any "
        "external fielding metric, and it carries no claims-registry entry. "
        "Do not read it as extra-base-prevention skill or convert it to runs."
    )

    if "extra_base_prevention" not in leaderboard.columns:
        st.info("Extra-base prevention data not available.")
        return

    ebp_df = leaderboard.dropna(subset=["extra_base_prevention"]).copy()
    if ebp_df.empty:
        st.info("No extra-base prevention data.")
        return

    ebp_df = ebp_df.sort_values("extra_base_prevention", ascending=True)

    fig = go.Figure(go.Bar(
        x=ebp_df["extra_base_prevention"],
        y=ebp_df["team_id"],
        orientation="h",
        marker_color=[
            _POSITIVE_GREEN if v >= ebp_df["extra_base_prevention"].median()
            else _NEUTRAL_GOLD
            for v in ebp_df["extra_base_prevention"]
        ],
        text=[f"{v:.1%}" for v in ebp_df["extra_base_prevention"]],
        textposition="outside",
        hovertemplate=(
            "%{y}<br>"
            "EBP: %{x:.1%}<br>"
            "<extra></extra>"
        ),
    ))

    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Extra-Base Prevention Rate",
        yaxis_title="",
        height=max(400, len(ebp_df) * 28),
        margin=dict(t=30, b=40, l=60, r=80),
        xaxis=dict(tickformat=".0%"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab: Game-by-Game Timeline
# ---------------------------------------------------------------------------

def _render_timeline(conn, team_id: str, season: int) -> None:
    """Game-by-game DPI timeline across the season."""
    st.subheader(f"{team_id} DPI Timeline ({season})")

    try:
        timeline = _cached_dpi_timeline(team_id, season)
    except Exception as exc:
        st.error(f"Error loading timeline: {exc}")
        return

    if timeline is None or timeline.empty:
        st.info("No game-by-game data available.")
        return

    # DPI line chart with rolling average
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.35],
        subplot_titles=("Game DPI", "BIP per Game"),
    )

    # Game DPI bars
    colors = [
        _POSITIVE_GREEN if v > 0 else _NEGATIVE_RED
        for v in timeline["dpi"]
    ]
    fig.add_trace(
        go.Bar(
            x=timeline["game_date"],
            y=timeline["dpi"],
            marker_color=colors,
            name="Game DPI",
            opacity=0.6,
            hovertemplate="Date: %{x}<br>DPI: %{y:+.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Rolling average
    if len(timeline) >= 5:
        rolling = timeline["dpi"].rolling(window=10, min_periods=3).mean()
        fig.add_trace(
            go.Scatter(
                x=timeline["game_date"],
                y=rolling,
                mode="lines",
                name="10-Game Avg",
                line=dict(color="white", width=2),
                hovertemplate="10-game avg: %{y:+.3f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # BIP per game
    fig.add_trace(
        go.Bar(
            x=timeline["game_date"],
            y=timeline["n_bip"],
            marker_color=_PHILLIES_LIGHT,
            name="BIP",
            opacity=0.5,
            hovertemplate="BIP: %{y}<extra></extra>",
        ),
        row=2, col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        height=550,
        margin=dict(t=50, b=40, l=50, r=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        showlegend=True,
    )

    fig.update_yaxes(
        title_text="DPI",
        zeroline=True,
        zerolinecolor="rgba(255,255,255,0.3)",
        row=1, col=1,
    )
    fig.update_yaxes(title_text="BIP Count", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Season DPI Avg", f"{timeline['dpi'].mean():+.3f}")
    with col2:
        st.metric("Best Game", f"{timeline['dpi'].max():+.3f}")
    with col3:
        st.metric("Worst Game", f"{timeline['dpi'].min():+.3f}")
