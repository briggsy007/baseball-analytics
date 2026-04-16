"""
Pitch Sequencing analysis page.

Displays transition matrix heatmaps, count-specific pitch mix charts,
setup-knockout pair tables, batter vulnerability analysis, and
recommended pitch plans for specific matchups.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.db_helper import (
    get_db_connection,
    has_data,
    get_all_pitchers,
    get_all_batters,
)

# ---------------------------------------------------------------------------
# Graceful imports for analytics modules
# ---------------------------------------------------------------------------

_SEQUENCING_AVAILABLE = False
try:
    from src.analytics.pitch_sequencing import (
        analyze_sequencing_patterns,
        get_batter_pitch_type_vulnerability,
        recommend_pitch_plan,
    )
    _SEQUENCING_AVAILABLE = True
except ImportError:
    pass

_CACHE_AVAILABLE = False
try:
    from src.dashboard.cache_reader import get_cached_entity, cache_age_display
    _CACHE_AVAILABLE = True
except Exception:
    pass

# Pitch type metadata
_PITCH_LABELS: dict[str, str] = {
    "FF": "4-Seam", "SI": "Sinker", "SL": "Slider", "CU": "Curve",
    "CH": "Change", "FC": "Cutter", "FS": "Splitter", "KC": "K-Curve",
    "ST": "Sweeper", "SV": "Slurve", "IN": "Int. Ball", "PO": "Pitchout",
    "CS": "Slow Curve", "SC": "Screwball", "FA": "Fastball", "AB": "Auto Ball",
}

_PITCH_COLORS: dict[str, str] = {
    "FF": "#E81828", "SI": "#FF6B35", "SL": "#002D72", "CU": "#6A0DAD",
    "CH": "#2ECC71", "FC": "#F39C12", "FS": "#1ABC9C", "KC": "#9B59B6",
    "ST": "#E74C3C", "SV": "#3498DB",
}


# ---------------------------------------------------------------------------
# Cached computation wrappers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def _cached_sequencing_patterns(_conn, pitcher_id: int) -> dict:
    """Cached wrapper for sequencing pattern analysis."""
    return analyze_sequencing_patterns(_conn, pitcher_id)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Pitch Sequencing page."""
    st.title("Pitch Sequencing Analysis")
    st.caption(
        "Explore how pitchers construct at-bats: transition patterns, "
        "count tendencies, and matchup-specific strategies."
    )
    st.info("""
**What this shows:** How pitchers construct at-bats — which pitches follow which, how patterns change by count, and setup-knockout combinations.

- **Transition matrix** shows P(next pitch | current pitch) — bright cells mean predictable sequences batters can exploit
- **Count tendencies** reveal patterns like "always throws breaking ball on 0-2" that scouting departments target
- **Impact:** Elite sequencers (like Greg Maddux historically) get 10-15% more whiffs than their raw stuff suggests by keeping hitters off balance
""")

    conn = get_db_connection()

    if not _SEQUENCING_AVAILABLE:
        st.error(
            "The `pitch_sequencing` analytics module could not be imported. "
            "Check that all dependencies are installed."
        )
        return

    if conn is None or not has_data(conn):
        st.warning(
            "No pitch data available. Run the data backfill pipeline first."
        )
        return

    # ── Pitcher selector ─────────────────────────────────────────────────
    pitchers = get_all_pitchers(conn)
    if not pitchers:
        try:
            fallback = conn.execute("""
                SELECT DISTINCT pi.pitcher_id AS player_id,
                       COALESCE(pl.full_name, 'Pitcher ' || CAST(pi.pitcher_id AS VARCHAR)) AS full_name
                FROM (SELECT DISTINCT pitcher_id FROM pitches LIMIT 500) pi
                LEFT JOIN players pl ON pl.player_id = pi.pitcher_id
                ORDER BY full_name
            """).fetchdf()
            pitchers = fallback.to_dict("records")
        except Exception:
            st.info("No pitcher data available.")
            return

    pitcher_names = {
        p.get("full_name") or f"ID {p['player_id']}": p["player_id"]
        for p in pitchers
    }
    selected_pitcher = st.selectbox(
        "Select Pitcher",
        options=sorted(pitcher_names.keys()),
        key="seq_pitcher_select",
    )

    if not selected_pitcher:
        return

    pitcher_id = pitcher_names[selected_pitcher]

    # ── Load sequencing data ─────────────────────────────────────────────
    # Try entity cache first, then compute with Streamlit caching
    patterns = None
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_entity(conn, "sequencing", pitcher_id)
            if cached is not None:
                patterns = cached
                st.caption("Using pre-computed results")
        except Exception:
            pass

    if patterns is None:
        with st.spinner("Analysing sequencing patterns..."):
            try:
                patterns = _cached_sequencing_patterns(conn, pitcher_id)
            except Exception as exc:
                st.error(f"Error analysing patterns: {exc}")
                return

    if not patterns.get("transition_matrix") and not patterns.get("count_mix"):
        st.info("Not enough data for this pitcher's sequencing analysis.")
        return

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_trans, tab_count, tab_setup, tab_vuln, tab_plan = st.tabs([
        "Transition Matrix",
        "Count Mix",
        "Setup-Knockout",
        "Batter Vulnerability",
        "Pitch Plan",
    ])

    with tab_trans:
        _render_transition_matrix(patterns)

    with tab_count:
        _render_count_mix(patterns)

    with tab_setup:
        _render_setup_knockout(patterns)

    with tab_vuln:
        _render_batter_vulnerability(conn)

    with tab_plan:
        _render_pitch_plan(conn, pitcher_id)


# ---------------------------------------------------------------------------
# Transition matrix heatmap
# ---------------------------------------------------------------------------


def _render_transition_matrix(patterns: dict) -> None:
    """Display a heatmap of P(next_pitch | current_pitch)."""
    st.subheader("Pitch Transition Probabilities")
    st.caption(
        "Each cell shows the probability of throwing a specific pitch type "
        "given what was thrown on the previous pitch."
    )

    matrix = patterns.get("transition_matrix", {})
    if not matrix:
        st.info("No transition data available.")
        return

    # Build a full matrix with all pitch types
    all_types = sorted(set(matrix.keys()) | {
        pt for row in matrix.values() for pt in row.keys()
    })

    z_vals = []
    for curr in all_types:
        row = []
        for nxt in all_types:
            row.append(matrix.get(curr, {}).get(nxt, 0.0))
        z_vals.append(row)

    labels_x = [_PITCH_LABELS.get(pt, pt) for pt in all_types]
    labels_y = [_PITCH_LABELS.get(pt, pt) for pt in all_types]

    # Text annotations
    text_vals = [[f"{v:.0%}" if v > 0 else "" for v in row] for row in z_vals]

    fig = go.Figure(data=go.Heatmap(
        z=z_vals,
        x=labels_x,
        y=labels_y,
        text=text_vals,
        texttemplate="%{text}",
        colorscale="RdYlGn",
        reversescale=True,
        zmin=0,
        zmax=max(max(row) for row in z_vals) if z_vals else 1,
        hovertemplate=(
            "After %{y} -> %{x}<br>"
            "Probability: %{z:.1%}<extra></extra>"
        ),
    ))

    fig.update_layout(
        xaxis=dict(title="Next Pitch", side="top"),
        yaxis=dict(title="Current Pitch", autorange="reversed"),
        template="plotly_dark",
        height=max(350, len(all_types) * 60),
        margin=dict(l=100, r=30, t=80, b=30),
    )
    st.plotly_chart(fig, use_container_width=True, key="seq_transition")

    # Quick insights
    _transition_insights(matrix, all_types)


def _transition_insights(matrix: dict, all_types: list[str]) -> None:
    """Show quick text insights from the transition matrix."""
    insights: list[str] = []

    for curr in all_types:
        row = matrix.get(curr, {})
        if not row:
            continue
        most_common = max(row, key=row.get)
        pct = row[most_common]
        if pct >= 0.35:
            insights.append(
                f"After **{_PITCH_LABELS.get(curr, curr)}**, "
                f"most likely next pitch is "
                f"**{_PITCH_LABELS.get(most_common, most_common)}** ({pct:.0%})"
            )

    if insights:
        st.markdown("**Key patterns:**")
        for insight in insights[:5]:
            st.markdown(f"- {insight}")


# ---------------------------------------------------------------------------
# Count-specific pitch mix
# ---------------------------------------------------------------------------


def _render_count_mix(patterns: dict) -> None:
    """Display count-specific pitch mix as grouped bar charts."""
    st.subheader("Pitch Mix by Count")
    st.caption("How the pitcher changes their arsenal based on the count.")

    count_mix = patterns.get("count_mix", {})
    if not count_mix:
        st.info("No count-mix data available.")
        return

    # Group counts into categories
    count_groups = {
        "Pitcher Ahead": ["0-1", "0-2", "1-2"],
        "Even / Neutral": ["0-0", "1-1", "2-2"],
        "Batter Ahead": ["1-0", "2-0", "2-1", "3-0", "3-1", "3-2"],
    }

    selected_group = st.radio(
        "Count Category",
        options=list(count_groups.keys()),
        horizontal=True,
        key="seq_count_group",
    )

    counts_to_show = [c for c in count_groups[selected_group] if c in count_mix]
    if not counts_to_show:
        st.info(f"No data for {selected_group} counts.")
        return

    # Collect all pitch types across displayed counts
    all_pts = sorted({
        pt for c in counts_to_show for pt in count_mix.get(c, {}).keys()
    })

    fig = go.Figure()
    for pt in all_pts:
        values = [count_mix.get(c, {}).get(pt, 0) * 100 for c in counts_to_show]
        fig.add_trace(go.Bar(
            name=_PITCH_LABELS.get(pt, pt),
            x=counts_to_show,
            y=values,
            marker_color=_PITCH_COLORS.get(pt, "#FFFFFF"),
            hovertemplate=f"%{{x}}: %{{y:.1f}}%<extra>{_PITCH_LABELS.get(pt, pt)}</extra>",
        ))

    fig.update_layout(
        barmode="group",
        xaxis=dict(title="Count (B-S)"),
        yaxis=dict(title="Usage %", ticksuffix="%"),
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=30, t=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True, key="seq_count_mix")

    # First pitch and putaway summary
    col1, col2 = st.columns(2)
    with col1:
        fp = patterns.get("first_pitch", {})
        st.markdown("**First Pitch (0-0)**")
        if fp.get("type_dist"):
            for pt, pct in sorted(fp["type_dist"].items(), key=lambda x: -x[1]):
                st.markdown(f"- {_PITCH_LABELS.get(pt, pt)}: {pct * 100:.1f}%")
            st.metric("First-Pitch Strike Rate", f"{fp.get('strike_rate', 0) * 100:.1f}%")
        else:
            st.caption("No first-pitch data.")

    with col2:
        pa = patterns.get("putaway", {})
        st.markdown("**Putaway Pitch (2-strike counts)**")
        if pa.get("pitch"):
            st.markdown(
                f"Primary putaway: **{_PITCH_LABELS.get(pa['pitch'], pa['pitch'])}**"
            )
            st.metric("Whiff Rate (2-strike)", f"{pa.get('whiff_rate', 0) * 100:.1f}%")
            st.metric("K Rate (2-strike)", f"{pa.get('k_rate', 0) * 100:.1f}%")
        else:
            st.caption("No putaway data.")


# ---------------------------------------------------------------------------
# Setup-knockout pairs
# ---------------------------------------------------------------------------


def _render_setup_knockout(patterns: dict) -> None:
    """Display the most effective 2-pitch sequences."""
    st.subheader("Setup-Knockout Pairs")
    st.caption(
        "Two-pitch sequences and their effectiveness. "
        "'Setup' is the preceding pitch; 'Knockout' is the next pitch."
    )

    pairs = patterns.get("setup_knockout_pairs", [])
    if not pairs:
        st.info("Not enough data for setup-knockout analysis.")
        return

    rows = []
    for p in pairs:
        rows.append({
            "Setup": _PITCH_LABELS.get(p["setup"], p["setup"]),
            "Knockout": _PITCH_LABELS.get(p["knockout"], p["knockout"]),
            "Whiff Rate": f"{p['whiff_rate'] * 100:.1f}%",
            "K Rate": f"{p['k_rate'] * 100:.1f}%",
            "Count": p["count"],
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    # Highlight the best pairs
    if pairs:
        best = pairs[0]
        st.success(
            f"Best combo: **{_PITCH_LABELS.get(best['setup'], best['setup'])}** -> "
            f"**{_PITCH_LABELS.get(best['knockout'], best['knockout'])}** "
            f"({best['whiff_rate'] * 100:.1f}% whiff rate on {best['count']} sequences)"
        )

    # Tunneling scores
    tunneling = patterns.get("pitch_tunneling", {})
    if tunneling:
        st.markdown("---")
        st.markdown("**Pitch Tunneling Scores**")
        st.caption(
            "Higher score = pitches look more similar through the batter's "
            "decision point, making them harder to distinguish."
        )
        tunnel_rows = []
        for key, info in sorted(
            tunneling.items(),
            key=lambda x: x[1].get("early_tunnel_score", 0),
            reverse=True,
        ):
            parts = key.split("_to_")
            if len(parts) == 2:
                tunnel_rows.append({
                    "Pair": (
                        f"{_PITCH_LABELS.get(parts[0], parts[0])} / "
                        f"{_PITCH_LABELS.get(parts[1], parts[1])}"
                    ),
                    "Tunnel Score": f"{info['early_tunnel_score']:.3f}",
                })

        if tunnel_rows:
            st.dataframe(
                pd.DataFrame(tunnel_rows),
                use_container_width=True,
                hide_index=True,
            )


# ---------------------------------------------------------------------------
# Batter vulnerability
# ---------------------------------------------------------------------------


def _render_batter_vulnerability(conn) -> None:
    """Analyse and display a selected batter's pitch-type vulnerabilities."""
    st.subheader("Batter Vulnerability Analysis")

    batters = get_all_batters(conn)
    if not batters:
        try:
            fallback = conn.execute("""
                SELECT DISTINCT bi.batter_id AS player_id,
                       COALESCE(pl.full_name, 'Batter ' || CAST(bi.batter_id AS VARCHAR)) AS full_name
                FROM (SELECT DISTINCT batter_id FROM pitches LIMIT 500) bi
                LEFT JOIN players pl ON pl.player_id = bi.batter_id
                ORDER BY full_name
            """).fetchdf()
            batters = fallback.to_dict("records")
        except Exception:
            st.info("No batter data available.")
            return

    batter_names = {
        b.get("full_name") or f"ID {b['player_id']}": b["player_id"]
        for b in batters
    }
    selected_batter = st.selectbox(
        "Select Batter",
        options=sorted(batter_names.keys()),
        key="seq_batter_select",
    )

    if not selected_batter:
        return

    batter_id = batter_names[selected_batter]

    with st.spinner("Analysing batter vulnerability..."):
        try:
            vuln = get_batter_pitch_type_vulnerability(conn, batter_id)
        except Exception as exc:
            st.error(f"Error: {exc}")
            return

    if not vuln.get("worst_pitch_types") and not vuln.get("best_pitch_types"):
        st.info("Not enough data for this batter's vulnerability analysis.")
        return

    # ── Worst pitch types (for the batter = best for the pitcher) ────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Weaknesses** (exploit these)")
        for p in vuln.get("worst_pitch_types", []):
            woba_str = f".{int(p['woba'] * 1000):03d}" if p["woba"] is not None else "N/A"
            st.markdown(
                f"- **{_PITCH_LABELS.get(p['pitch_type'], p['pitch_type'])}**: "
                f"wOBA {woba_str}, "
                f"{p['whiff_rate'] * 100:.0f}% whiff "
                f"({p['sample']} pitches)"
            )

    with col2:
        st.markdown("**Strengths** (avoid these)")
        for p in vuln.get("best_pitch_types", []):
            woba_str = f".{int(p['woba'] * 1000):03d}" if p["woba"] is not None else "N/A"
            st.markdown(
                f"- **{_PITCH_LABELS.get(p['pitch_type'], p['pitch_type'])}**: "
                f"wOBA {woba_str}, "
                f"{p['whiff_rate'] * 100:.0f}% whiff "
                f"({p['sample']} pitches)"
            )

    # ── Vulnerable zones ─────────────────────────────────────────────────
    zones = vuln.get("vulnerable_zones", [])
    if zones:
        st.markdown("---")
        st.markdown("**Most Vulnerable Zones**")
        zone_rows = []
        for z in zones:
            row: dict[str, Any] = {
                "Zone": z["zone_desc"],
                "Whiff Rate": f"{z['whiff_rate'] * 100:.1f}%",
                "Sample": z["sample"],
            }
            if "chase_rate" in z:
                row["Chase Rate"] = f"{z['chase_rate'] * 100:.1f}%"
            zone_rows.append(row)

        st.dataframe(
            pd.DataFrame(zone_rows),
            use_container_width=True,
            hide_index=True,
        )

    # ── Recommendation ───────────────────────────────────────────────────
    rec = vuln.get("recommendation", "")
    if rec:
        st.info(f"**Recommendation:** {rec}")


# ---------------------------------------------------------------------------
# Pitch plan
# ---------------------------------------------------------------------------


def _render_pitch_plan(conn, pitcher_id: int) -> None:
    """Generate and display a recommended pitch plan for a specific matchup."""
    st.subheader("Recommended Pitch Plan")
    st.caption("Select a batter to get a matchup-specific strategy.")

    batters = get_all_batters(conn)
    if not batters:
        try:
            fallback = conn.execute("""
                SELECT DISTINCT bi.batter_id AS player_id,
                       COALESCE(pl.full_name, 'Batter ' || CAST(bi.batter_id AS VARCHAR)) AS full_name
                FROM (SELECT DISTINCT batter_id FROM pitches LIMIT 500) bi
                LEFT JOIN players pl ON pl.player_id = bi.batter_id
                ORDER BY full_name
            """).fetchdf()
            batters = fallback.to_dict("records")
        except Exception:
            st.info("No batter data available.")
            return

    batter_names = {
        b.get("full_name") or f"ID {b['player_id']}": b["player_id"]
        for b in batters
    }
    selected_batter = st.selectbox(
        "Select Batter for Pitch Plan",
        options=sorted(batter_names.keys()),
        key="seq_plan_batter_select",
    )

    if not selected_batter:
        return

    batter_id = batter_names[selected_batter]

    if st.button("Generate Pitch Plan", type="primary", key="seq_gen_plan"):
        with st.spinner("Building pitch plan..."):
            try:
                plan = recommend_pitch_plan(conn, pitcher_id, batter_id)
            except Exception as exc:
                st.error(f"Error generating plan: {exc}")
                return

        _display_pitch_plan(plan)


def _display_pitch_plan(plan: dict) -> None:
    """Render the pitch plan recommendation."""
    confidence = plan.get("confidence", "low")
    conf_color = {"high": "#2ECC71", "medium": "#F39C12", "low": "#E74C3C"}.get(
        confidence, "#FFFFFF"
    )

    st.markdown(
        f"**Confidence:** "
        f"<span style='color:{conf_color}; font-weight:bold;'>"
        f"{confidence.upper()}</span>",
        unsafe_allow_html=True,
    )

    if plan.get("platoon_advantage") is not None:
        adv = "Yes" if plan["platoon_advantage"] else "No"
        st.markdown(f"**Platoon advantage:** {adv}")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    # First pitch
    fp = plan.get("recommended_first_pitch", {})
    with col1:
        st.markdown("**First Pitch**")
        if fp.get("type"):
            label = _PITCH_LABELS.get(fp["type"], fp["type"])
            st.markdown(f"Throw: **{label}**")
            st.markdown(f"Location: {fp.get('location', 'N/A')}")
            st.caption(fp.get("reasoning", ""))
        else:
            st.caption("No recommendation (insufficient data)")

    # Putaway
    pa = plan.get("putaway_pitch", {})
    with col2:
        st.markdown("**Putaway Pitch**")
        if pa.get("type"):
            label = _PITCH_LABELS.get(pa["type"], pa["type"])
            st.markdown(f"Throw: **{label}**")
            st.markdown(f"Location: {pa.get('location', 'N/A')}")
            st.caption(pa.get("reasoning", ""))
        else:
            st.caption("No recommendation (insufficient data)")

    # Avoid
    av = plan.get("avoid", {})
    with col3:
        st.markdown("**Avoid**")
        if av.get("type"):
            label = _PITCH_LABELS.get(av["type"], av["type"])
            st.markdown(f"Pitch: **{label}**")
            st.markdown(f"Location: {av.get('location', 'N/A')}")
            st.caption(av.get("reasoning", ""))
        else:
            st.caption("No specific pitch to avoid")

    # Sequence suggestion
    seq = plan.get("sequence_suggestion", "")
    if seq:
        st.markdown("---")
        st.markdown("**Sequence Strategy**")
        st.info(seq)
