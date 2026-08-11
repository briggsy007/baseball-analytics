"""
AdjustedWAR dashboard view -- context-adjusted player valuation.

RENAMED 2026-08-10 (Batch D, user-adjudicated): this page displayed
"CausalWAR" until 2026-08-10. The product is **AdjustedWAR** --
*regularized adjustment, not causal identification*. Module paths, DB
cache keys (``leaderboard_cache.model_name = 'causal_war'``) and registry
ids keep their historical names on purpose; only display strings moved.

Displays a leaderboard of context-adjusted player value, comparison
scatter plots against traditional WAR, and "Biggest Movers" analysis.

Confidence intervals do NOT render here. WS4.7 coverage-validated the two
uncertainty layers of the production ridge model against realized
next-season outcomes and both failed the pre-registered [90%, 98%] gate
(claim ``adjusted_war_v3_ci_coverage``); the legacy bootstrap intervals
this page used to plot were never coverage-validated at all. The plotting
code is retained and gated on the claim, so intervals return
automatically if a construction ever passes.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.claims import get_claim
from src.dashboard.db_helper import get_db_connection, has_data

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from src.analytics.causal_war import (
    CausalWARConfig,
    CausalWARModel,
    train,
    batch_calculate,
    calculate_causal_war,
    get_leaderboard,
)

_CACHE_AVAILABLE = False
try:
    from src.dashboard.cache_reader import get_cached_leaderboard, cache_age_display
    _CACHE_AVAILABLE = True
except Exception:
    pass

# --- Rename + uncertainty gate (Batch D, 2026-08-10) -----------------------
_CLAIM_CI = get_claim("adjusted_war_v3_ci_coverage")
#: Intervals render only if a coverage-validated construction exists.
_CI_MAY_SHIP = bool(_CLAIM_CI.value.get("any_ci_may_ship", False))

_RENAME_NOTE = (
    "AdjustedWAR -- renamed 2026-08-10; formerly CausalWAR. Regularized "
    "adjustment, not causal identification."
)
_NO_CI_NOTE = (
    "No per-player confidence interval is displayed. "
    f"{_CLAIM_CI.caveat}"
)

# K6 consequence (user-adjudicated 2026-08-10).  Rendered from the single
# claims-registry entry -- the same one the contrarian board surfaces render
# -- so this page cannot drift away from the mandated framing.
_CLAIM_K6 = get_claim("adjusted_war_boards_k6_framing")
_K6_FRAMING = f"Board evidence, in full: AdjustedWAR {_CLAIM_K6.value}."

# --- Which model produced the displayed scores (WS2.1 provenance stamps) ---
# ``scripts/precompute.py`` stamps ``scoring_model`` /
# ``scoring_artifact_version`` / ``scoring_artifact`` onto every cached row.
# Disclosure pattern is the house one:
# ``src/dashboard/views/defensive_pressing.py::_render_artifact_provenance``.
_SCORING_MODEL_LABELS = {
    "adjusted_war_v3": (
        "AdjustedWAR v3 (ridge joint estimation, "
        "`src/analytics/adjusted_war_v3.py`)"
    ),
    "causal_war_legacy": (
        "the LEGACY CausalWAR formulation (`src/analytics/causal_war.py`)"
    ),
}
_UNSTAMPED_PROVENANCE = (
    "Scored by: **the LEGACY CausalWAR formulation** "
    "(`src/analytics/causal_war.py`). This cache carries no provenance "
    "stamp, which means it was written before stamping landed (2026-08-10) "
    "and therefore before AdjustedWAR v3 (ridge) was promoted to the "
    "production scoring path -- the numbers below were NOT produced by the "
    "promoted model. They are rescored through the registry alias "
    "`adjusted_war_v3/production` on the next "
    "`python scripts/precompute.py --model causal_war` run."
)

# Phillies red / blue palette
_PHILLIES_RED = "#E81828"
_PHILLIES_BLUE = "#002D72"
_PHILLIES_LIGHT = "#B0B7BC"
_POSITIVE_GREEN = "#2ECC71"
_NEGATIVE_RED = "#E74C3C"
_CI_BAND = "rgba(232, 24, 40, 0.25)"


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the AdjustedWAR Analysis page."""
    st.title("AdjustedWAR: Context-Adjusted Player Valuation")
    st.info(f"{_RENAME_NOTE} {_NO_CI_NOTE}")
    st.caption(
        "Adjusts for park, lineup, platoon, and game-state context to "
        "estimate each player's per-PA run contribution net of the "
        "situations they hit in. An adjustment, not an identification "
        "strategy: the estimate is only as unconfounded as the covariate "
        "set, and no causal claim attaches to it."
    )
    st.warning(_K6_FRAMING)

    with st.expander("What does this mean?"):
        st.markdown(f"""
**AdjustedWAR re-prices a player's production** by statistically adjusting for park, lineup, platoon, and game situation — things traditional WAR only crudely adjusts for.

- **AdjustedWAR > traditional WAR** → the model prices the player above the public WAR market (park or context may be suppressing their raw stats)
- **AdjustedWAR < traditional WAR** → the model prices the player below it (they may be benefiting from favorable park/lineup context)
- **Biggest Movers** are the players where the adjustment is largest — candidate market disagreements, not established mispricings
- **No confidence intervals are shown.** Both uncertainty layers of the production model failed their pre-registered coverage gate (49.6% and 71.3% empirical coverage at a nominal 95%; claim:adjusted_war_v3_ci_coverage), so point estimates ship alone until a construction passes. Treat single-season values as noisy.
- **Not a forecast.** AdjustedWAR {_CLAIM_K6.value} (kill criterion K6).
""")

    conn = get_db_connection()

    if conn is None or not has_data(conn):
        st.warning(
            "No pitch data available. Run the data backfill pipeline first "
            "(`python scripts/backfill.py`)."
        )
        return

    # ---- Sidebar controls ------------------------------------------------
    with st.sidebar:
        st.markdown("### AdjustedWAR Options")

        season = st.selectbox(
            "Season",
            options=_get_available_seasons(conn),
            key="causal_war_season",
        )

        position_filter = st.radio(
            "Position Filter",
            options=["All", "Batters", "Pitchers"],
            key="causal_war_position",
        )

        position_type = {
            "All": "all",
            "Batters": "batter",
            "Pitchers": "pitcher",
        }[position_filter]

        if st.button("Train AdjustedWAR Model", type="primary"):
            _train_model_ui(conn, season)
            st.rerun()

    # ---- Load data -------------------------------------------------------
    df = _load_leaderboard(conn, season, position_type)

    if df is None or df.empty:
        st.warning("**Model not trained.** AdjustedWAR data is not available for this season.")
        st.markdown(
            "Click **Train AdjustedWAR Model** in the sidebar, or run "
            "`python scripts/precompute.py` to generate cached results."
        )
        if st.button("Train AdjustedWAR Model Now", type="primary", key="causal_train_main"):
            _train_model_ui(conn, season)
            st.rerun()
        return

    # ---- Which model produced these numbers -------------------------------
    _render_scoring_provenance(df)

    # ---- Tabs ------------------------------------------------------------
    tab_board, tab_detail, tab_compare, tab_movers = st.tabs([
        "Leaderboard",
        "Player Detail",
        "AdjustedWAR vs Traditional WAR",
        "Biggest Movers",
    ])

    with tab_board:
        _render_leaderboard(df)

    with tab_detail:
        _render_player_detail(conn, df, season)

    with tab_compare:
        _render_comparison_scatter(df)

    with tab_movers:
        _render_biggest_movers(df)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _get_available_seasons(conn) -> list[int]:
    """Return list of seasons with pitch data."""
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


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_causal_leaderboard(_conn, season: int, position_type: str) -> pd.DataFrame | None:
    """Cached wrapper for AdjustedWAR leaderboard computation."""
    return get_leaderboard(_conn, season, position_type)


def _load_leaderboard(
    conn,
    season: int,
    position_type: str,
) -> pd.DataFrame | None:
    """Load AdjustedWAR leaderboard, using cache then live computation."""
    # Try precomputed cache first
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "causal_war", season)
            if cached is not None:
                age_info = cache_age_display(conn, "causal_war", season)
                if age_info:
                    st.caption(age_info)
                return cached
        except Exception:
            pass

    try:
        with st.spinner("Computing... Run `python scripts/precompute.py` for instant loading."):
            return _cached_causal_leaderboard(conn, season, position_type)
    except RuntimeError:
        # Model not trained
        return None
    except Exception as exc:
        st.error(f"Error loading AdjustedWAR data: {exc}")
        return None


def _render_scoring_provenance(df: pd.DataFrame | None) -> None:
    """State which model produced the scores this page is displaying.

    ``scripts/precompute.py`` stamps ``scoring_model`` /
    ``scoring_artifact_version`` / ``scoring_artifact`` onto every cached
    row (WS2.1), because the AdjustedWAR cache key ``causal_war`` is served
    by two different estimators: the promoted ridge model
    (``adjusted_war_v3``) and the legacy formulation it replaced.  Same
    disclosure the DPI page makes for its xOut artifact
    (:func:`src.dashboard.views.defensive_pressing._render_artifact_provenance`).

    An UNSTAMPED frame is not "unknown": stamping and the promotion landed
    together on 2026-08-10, so a frame without the columns was necessarily
    produced by the legacy formulation, and the page says exactly that
    rather than letting promoted-model framing attach to legacy numbers.
    """
    if df is None or df.empty or "scoring_model" not in df.columns:
        st.caption(_UNSTAMPED_PROVENANCE)
        return

    model = str(df["scoring_model"].iloc[0])
    label = _SCORING_MODEL_LABELS.get(model, f"`{model}`")
    version = df.get("scoring_artifact_version", pd.Series([None])).iloc[0]
    artifact = df.get("scoring_artifact", pd.Series([None])).iloc[0]

    detail = f"Scored by: **{label}**"
    if version is not None and pd.notna(version):
        detail += f", registry version `{version}`"
    if artifact is not None and pd.notna(artifact):
        detail += f", artifact `{artifact}`"
    detail += "."
    if model != "adjusted_war_v3":
        detail += (
            " AdjustedWAR v3 (ridge) has been the production scoring model "
            "since 2026-08-10; these numbers did not come from it."
        )
    detail += (
        " Frozen contrarian boards are never rescored -- they keep the "
        "scores their picks were frozen with."
    )
    st.caption(detail)


def _train_model_ui(conn, season: int) -> None:
    """Train the AdjustedWAR model with UI feedback."""
    with st.spinner("Training AdjustedWAR model... this may take a few minutes."):
        try:
            metrics = train(conn, season=season, n_bootstrap=50, n_estimators=100)
            st.success("AdjustedWAR model trained successfully!")

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Outcome R2",
                f"{metrics.get('outcome_nuisance_r2', 0):.4f}",
            )
            col2.metric("Players Estimated", metrics.get("n_players_estimated", 0))
            col3.metric("Observations", f"{metrics.get('n_observations', 0):,}")

        except Exception as exc:
            st.error(f"Training failed: {exc}")


# ---------------------------------------------------------------------------
# Leaderboard tab
# ---------------------------------------------------------------------------


def _render_leaderboard(df: pd.DataFrame) -> None:
    """Display the AdjustedWAR leaderboard table."""
    st.subheader("AdjustedWAR Leaderboard")

    display_df = df.copy()
    display_df = display_df.reset_index(drop=True)
    display_df.index = display_df.index + 1
    display_df.index.name = "Rank"

    # Select display columns. CI columns are gated on the WS4.7 coverage
    # verdict -- with the gate failed they are withheld, not merely
    # footnoted (plan 4.7: no CI ships to any surface).
    display_cols = ["name", "causal_war"]
    if _CI_MAY_SHIP:
        display_cols += ["ci_low", "ci_high"]
    # Exactly one of the two adjusted-wOBA columns exists in any given
    # frame: `park_adj_woba` under the legacy formulation, and
    # `context_neutral_woba` under the promoted ridge path (different
    # constructions, so different labels -- see precompute_adjusted_war).
    display_cols += ["park_adj_woba", "context_neutral_woba", "raw_woba", "pa"]
    if "traditional_war" in display_df.columns:
        display_cols.append("traditional_war")

    available_cols = [c for c in display_cols if c in display_df.columns]

    # Pagination: default to top 200 rows, with a toggle to see everything.
    # Leaderboard can easily exceed 500 rows (all qualifying batters +
    # pitchers), which makes the rendered table sluggish without a cap.
    total_rows = len(display_df)
    show_all = False
    if total_rows > 200:
        show_all = st.checkbox(
            f"Show all {total_rows} players (default shows top 200)",
            value=False,
            key="causal_war_leaderboard_show_all",
        )
    rendered_df = display_df if show_all else display_df.head(200)

    if not _CI_MAY_SHIP:
        st.caption(_NO_CI_NOTE)

    st.dataframe(
        rendered_df[available_cols],
        use_container_width=True,
        height=500,
        column_config={
            "name": st.column_config.TextColumn("Player"),
            "causal_war": st.column_config.NumberColumn("AdjustedWAR", format="%.2f"),
            "ci_low": st.column_config.NumberColumn("CI Low", format="%.2f"),
            "ci_high": st.column_config.NumberColumn("CI High", format="%.2f"),
            "park_adj_woba": st.column_config.NumberColumn("Park-Adj wOBA", format="%.3f"),
            "context_neutral_woba": st.column_config.NumberColumn(
                "Ctx-Neutral wOBA", format="%.3f",
                help=(
                    "Ridge scoring path only: league mean of the fit sample "
                    "plus the player's centered batter coefficient -- the "
                    "wOBA the model attributes to the batter net of park, "
                    "lineup, platoon and base-out state. Not the same "
                    "construction as the legacy Park-Adj wOBA column."
                ),
            ),
            "raw_woba": st.column_config.NumberColumn("Raw wOBA", format="%.3f"),
            "pa": st.column_config.NumberColumn("PA", format="%d"),
            "traditional_war": st.column_config.NumberColumn("Trad. WAR", format="%.1f"),
        },
    )

    # Distribution chart (always uses the full df)
    if len(df) >= 5 and "causal_war" in df.columns:
        st.markdown("**AdjustedWAR Distribution**")
        _render_distribution(df)


def _render_distribution(df: pd.DataFrame) -> None:
    """Histogram of AdjustedWAR values."""
    values = df["causal_war"].dropna()
    if values.empty:
        return

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=values,
        nbinsx=25,
        marker_color=_PHILLIES_RED,
        opacity=0.75,
        name="AdjustedWAR",
    ))
    fig.add_vline(
        x=0,
        line=dict(color="white", width=2, dash="dash"),
        annotation_text="Replacement (0)",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis=dict(title="AdjustedWAR"),
        yaxis=dict(title="Count"),
        template="plotly_dark",
        height=350,
        margin=dict(l=50, r=30, t=30, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="causal_war_dist")


# ---------------------------------------------------------------------------
# Player detail tab
# ---------------------------------------------------------------------------


def _render_player_detail(conn, df: pd.DataFrame, season: int) -> None:
    """Show detailed AdjustedWAR breakdown for a single player."""
    st.subheader("Player Detail")

    if df.empty or "name" not in df.columns:
        st.info("No player data available.")
        return

    # Build player selector
    player_options = df["name"].dropna().tolist()
    if not player_options:
        player_options = [f"ID {pid}" for pid in df["player_id"].tolist()]

    selected = st.selectbox(
        "Select Player",
        options=player_options,
        key="causal_war_player_select",
    )

    if not selected:
        return

    # Find the player row
    if selected.startswith("ID "):
        pid = int(selected.split(" ")[1])
        player_row = df[df["player_id"] == pid]
    else:
        player_row = df[df["name"] == selected]

    if player_row.empty:
        st.warning("Player not found.")
        return

    row = player_row.iloc[0]

    # ---- Metric cards ----------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AdjustedWAR", f"{row.get('causal_war', 'N/A'):.2f}"
                if pd.notna(row.get('causal_war')) else "N/A")
    # Legacy frames carry `park_adj_woba`; ridge-scored frames carry
    # `context_neutral_woba`. Label whichever one is actually present.
    if pd.notna(row.get("park_adj_woba")):
        col2.metric("Park-Adj wOBA", f"{row.get('park_adj_woba'):.3f}")
    elif pd.notna(row.get("context_neutral_woba")):
        col2.metric("Ctx-Neutral wOBA", f"{row.get('context_neutral_woba'):.3f}")
    else:
        col2.metric("Park-Adj wOBA", "N/A")
    col3.metric("Plate Appearances", f"{int(row.get('pa', 0)):,}")

    trad_war = row.get("traditional_war")
    if pd.notna(trad_war):
        col4.metric("Traditional WAR", f"{trad_war:.1f}")
    else:
        col4.metric("Traditional WAR", "N/A")

    st.markdown("---")

    # ---- Point estimate (interval gated on the WS4.7 coverage verdict) ---
    st.markdown(
        "**Estimate**" if not _CI_MAY_SHIP else "**Confidence Interval**"
    )
    if not _CI_MAY_SHIP:
        st.caption(_NO_CI_NOTE)
    _render_single_forest_plot(row)

    st.markdown("---")

    # ---- Component breakdown ---------------------------------------------
    st.markdown("**AdjustedWAR Components**")
    _render_component_breakdown(row)


def _render_single_forest_plot(row: pd.Series) -> None:
    """Forest plot for a single player's AdjustedWAR point estimate (CI gated)."""
    causal_war = row.get("causal_war")
    ci_low = row.get("ci_low")
    ci_high = row.get("ci_high")
    name = row.get("name", "Unknown")

    if pd.isna(causal_war):
        st.caption("No AdjustedWAR estimate available for this player.")
        return

    fig = go.Figure()

    # CI error bar -- withheld while the WS4.7 coverage gate is failed.
    if _CI_MAY_SHIP and pd.notna(ci_low) and pd.notna(ci_high):
        fig.add_trace(go.Scatter(
            x=[causal_war],
            y=[name],
            error_x=dict(
                type="data",
                symmetric=False,
                array=[ci_high - causal_war],
                arrayminus=[causal_war - ci_low],
                color=_PHILLIES_RED,
                thickness=3,
                width=10,
            ),
            mode="markers",
            marker=dict(size=14, color=_PHILLIES_RED, symbol="diamond"),
            name="AdjustedWAR",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=[causal_war],
            y=[name],
            mode="markers",
            marker=dict(size=14, color=_PHILLIES_RED, symbol="diamond"),
            name="AdjustedWAR",
        ))

    fig.add_vline(x=0, line=dict(color="white", width=1, dash="dash"))
    fig.update_layout(
        xaxis=dict(title="AdjustedWAR"),
        template="plotly_dark",
        height=180,
        margin=dict(l=120, r=30, t=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="causal_war_single_forest")


def _render_component_breakdown(row: pd.Series) -> None:
    """Show waterfall of AdjustedWAR components."""
    raw_woba = row.get("raw_woba")
    park_adj = row.get("park_adj_woba")
    ctx_neutral = row.get("context_neutral_woba")
    causal_war = row.get("causal_war")
    pa = row.get("pa", 0)

    components = []
    if pd.notna(raw_woba):
        components.append(("Raw wOBA", round(float(raw_woba), 3)))
    if pd.notna(park_adj) and pd.notna(raw_woba):
        adj_delta = round(float(park_adj - raw_woba), 3)
        components.append(("Park Adjustment", adj_delta))
    elif pd.notna(ctx_neutral):
        # Ridge path: report the model's context-neutral wOBA as its own
        # row rather than differencing it against raw wOBA -- the two are
        # different constructions and their difference is not "the park
        # adjustment".
        components.append(("Ctx-Neutral wOBA (ridge)", round(float(ctx_neutral), 3)))
    if pd.notna(causal_war):
        components.append(("AdjustedWAR (wins)", round(float(causal_war), 2)))
    components.append(("Plate Appearances", int(pa)))

    if not components:
        st.caption("No component data available.")
        return

    comp_df = pd.DataFrame(components, columns=["Component", "Value"])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# AdjustedWAR vs Traditional WAR scatter
# ---------------------------------------------------------------------------


def _render_comparison_scatter(df: pd.DataFrame) -> None:
    """Scatter plot comparing AdjustedWAR with traditional WAR."""
    st.subheader("AdjustedWAR vs Traditional WAR")

    if "traditional_war" not in df.columns:
        st.info(
            "Traditional WAR data not available. "
            "Ensure season stats are loaded in the database."
        )
        return

    plot_df = df.dropna(subset=["causal_war", "traditional_war"])
    if plot_df.empty:
        st.info("No players with both AdjustedWAR and traditional WAR.")
        return

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=plot_df["traditional_war"],
        y=plot_df["causal_war"],
        mode="markers+text",
        marker=dict(
            size=10,
            color=_PHILLIES_RED,
            line=dict(width=1, color="white"),
        ),
        text=plot_df["name"].fillna(""),
        textposition="top center",
        textfont=dict(size=9),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Traditional WAR: %{x:.1f}<br>"
            "AdjustedWAR: %{y:.2f}<br>"
            "<extra></extra>"
        ),
        name="Players",
    ))

    # Perfect correlation line
    all_vals = pd.concat([plot_df["traditional_war"], plot_df["causal_war"]])
    min_val = float(all_vals.min()) - 0.5
    max_val = float(all_vals.max()) + 0.5

    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode="lines",
        line=dict(color=_PHILLIES_LIGHT, dash="dash", width=1),
        name="y = x",
    ))

    fig.update_layout(
        xaxis=dict(title="Traditional WAR"),
        yaxis=dict(title="AdjustedWAR"),
        template="plotly_dark",
        height=500,
        margin=dict(l=60, r=30, t=30, b=60),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True, key="causal_war_scatter")

    # Correlation stat
    if len(plot_df) >= 3:
        corr = plot_df["causal_war"].corr(plot_df["traditional_war"])
        st.caption(f"Pearson correlation: {corr:.3f} (n={len(plot_df)} players)")


# ---------------------------------------------------------------------------
# Biggest Movers
# ---------------------------------------------------------------------------


def _render_biggest_movers(df: pd.DataFrame) -> None:
    """Show players whose AdjustedWAR differs most from traditional metrics."""
    st.subheader("Biggest Movers: AdjustedWAR vs Traditional WAR")

    if "traditional_war" not in df.columns:
        st.info("Traditional WAR data not available for comparison.")
        return

    movers_df = df.dropna(subset=["causal_war", "traditional_war"]).copy()
    if movers_df.empty:
        st.info("No players with both metrics available.")
        return

    movers_df["war_diff"] = movers_df["causal_war"] - movers_df["traditional_war"]
    movers_df["abs_diff"] = movers_df["war_diff"].abs()
    movers_df = movers_df.sort_values("abs_diff", ascending=False).head(20)

    if movers_df.empty:
        return

    # ---- Two-column layout: gainers and losers ---------------------------
    col1, col2 = st.columns(2)

    gainers = movers_df[movers_df["war_diff"] > 0].head(10)
    losers = movers_df[movers_df["war_diff"] < 0].head(10)

    with col1:
        st.markdown("**Undervalued by Traditional WAR**")
        st.caption("AdjustedWAR > Traditional WAR")
        if not gainers.empty:
            _render_movers_table(gainers)
        else:
            st.caption("None found.")

    with col2:
        st.markdown("**Overvalued by Traditional WAR**")
        st.caption("AdjustedWAR < Traditional WAR")
        if not losers.empty:
            _render_movers_table(losers)
        else:
            st.caption("None found.")

    st.markdown("---")

    # ---- Divergence bar chart --------------------------------------------
    st.markdown("**WAR Divergence (AdjustedWAR - Traditional WAR)**")
    _render_divergence_bars(movers_df)

    # ---- Point-estimate plot for top movers ------------------------------
    st.markdown(
        "**Top Movers**" if not _CI_MAY_SHIP
        else "**Confidence Intervals for Top Movers**"
    )
    if not _CI_MAY_SHIP:
        st.caption(_NO_CI_NOTE)
    _render_forest_plot(movers_df.head(15))


def _render_movers_table(df: pd.DataFrame) -> None:
    """Display a table of biggest movers."""
    display = df[["name", "causal_war", "traditional_war", "war_diff", "pa"]].copy()
    display.columns = ["Player", "AdjustedWAR", "Trad. WAR", "Difference", "PA"]
    display = display.reset_index(drop=True)
    display.index = display.index + 1

    st.dataframe(
        display,
        use_container_width=True,
        column_config={
            "AdjustedWAR": st.column_config.NumberColumn(format="%.2f"),
            "Trad. WAR": st.column_config.NumberColumn(format="%.1f"),
            "Difference": st.column_config.NumberColumn(format="%+.2f"),
            "PA": st.column_config.NumberColumn(format="%d"),
        },
    )


def _render_divergence_bars(df: pd.DataFrame) -> None:
    """Horizontal bar chart of WAR differences."""
    plot_df = df.sort_values("war_diff").copy()

    colors = [
        _POSITIVE_GREEN if v > 0 else _NEGATIVE_RED
        for v in plot_df["war_diff"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=plot_df["name"].fillna("Unknown"),
        x=plot_df["war_diff"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}" for v in plot_df["war_diff"]],
        textposition="auto",
    ))
    fig.add_vline(x=0, line=dict(color="white", width=1))
    fig.update_layout(
        xaxis=dict(title="AdjustedWAR - Traditional WAR"),
        yaxis=dict(autorange="reversed"),
        template="plotly_dark",
        height=max(300, len(plot_df) * 30),
        margin=dict(l=120, r=30, t=20, b=50),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True, key="causal_war_divergence")


def _render_forest_plot(df: pd.DataFrame) -> None:
    """Point estimates for multiple players; CI bars gated on WS4.7."""
    if "causal_war" not in df.columns:
        st.caption("No AdjustedWAR estimates available.")
        return
    has_ci = _CI_MAY_SHIP and all(c in df.columns for c in ["ci_low", "ci_high"])

    plot_df = df.dropna(subset=["causal_war"]).copy()
    if plot_df.empty:
        return

    # Sort by AdjustedWAR descending for display
    plot_df = plot_df.sort_values("causal_war", ascending=True)

    fig = go.Figure()

    names = plot_df["name"].fillna("Unknown").tolist()
    values = plot_df["causal_war"].tolist()

    # CI bars -- only when a coverage-validated construction exists.
    if has_ci:
        ci_lows = plot_df["ci_low"].tolist()
        ci_highs = plot_df["ci_high"].tolist()
        for name, lo, hi in zip(names, ci_lows, ci_highs):
            if pd.notna(lo) and pd.notna(hi):
                fig.add_trace(go.Scatter(
                    x=[lo, hi],
                    y=[name, name],
                    mode="lines",
                    line=dict(color=_PHILLIES_RED, width=3),
                    showlegend=False,
                ))

    # Point estimates
    fig.add_trace(go.Scatter(
        x=values,
        y=names,
        mode="markers",
        marker=dict(size=10, color=_PHILLIES_RED, symbol="diamond"),
        name="AdjustedWAR",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "AdjustedWAR: %{x:.2f}<br>"
            "<extra></extra>"
        ),
    ))

    fig.add_vline(x=0, line=dict(color="white", width=1, dash="dash"))
    fig.update_layout(
        xaxis=dict(title="AdjustedWAR"),
        template="plotly_dark",
        height=max(300, len(plot_df) * 35),
        margin=dict(l=140, r=30, t=20, b=50),
    )
    st.plotly_chart(fig, use_container_width=True, key="causal_war_forest")
