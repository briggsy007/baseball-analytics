"""
MechanixAE dashboard view -- Mechanical Drift Detection & Injury Risk.

Provides:
  - **MDI Gauge**: Current Mechanical Drift Index with color zones.
  - **Drift Timeline**: MDI over the season with game markers and changepoints.
  - **Latent Space Trajectory**: 2D projection of the pitcher's mechanical path.
  - **Feature Attribution**: Which input features drive current drift.
  - **Alert System**: Pitchers with MDI above the 80th percentile.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.dashboard.db_helper import get_db_connection, has_data, get_all_pitchers

# ---------------------------------------------------------------------------
# Graceful imports
# ---------------------------------------------------------------------------

_MECHANIX_AVAILABLE = False
try:
    from src.analytics.mechanix_ae import (
        MechanixVAE,
        FEATURE_COLS,
        N_FEATURES,
        WINDOW_SIZE,
        LATENT_DIM,
        batch_calculate,
        build_sliding_windows,
        calculate_drift_velocity,
        calculate_mdi,
        detect_changepoints,
        normalize_pitcher_pitch_type,
        prepare_features,
        train_mechanix_ae,
        _compute_reconstruction_errors,
        _fetch_pitch_data,
        _load_model,
    )
    _MECHANIX_AVAILABLE = True
except ImportError:
    pass

_PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    pass

_CACHE_AVAILABLE = False
try:
    from src.dashboard.cache_reader import get_cached_leaderboard, cache_age_display
    _CACHE_AVAILABLE = True
except Exception:
    pass

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass

# Pitch type display helpers
_PITCH_LABELS: dict[str, str] = {
    "FF": "4-Seam", "SI": "Sinker", "SL": "Slider", "CU": "Curve",
    "CH": "Change", "FC": "Cutter", "FS": "Splitter", "KC": "Knuckle-Curve",
    "ST": "Sweeper", "SV": "Slurve", "CS": "Slow Curve",
}

# MDI color zones
_MDI_COLORS = {
    "green": "#2ECC71",
    "yellow": "#F1C40F",
    "orange": "#E67E22",
    "red": "#E74C3C",
}


def _mdi_color(mdi: float) -> str:
    if mdi < 30:
        return _MDI_COLORS["green"]
    elif mdi < 60:
        return _MDI_COLORS["yellow"]
    elif mdi < 80:
        return _MDI_COLORS["orange"]
    return _MDI_COLORS["red"]


def _mdi_label(mdi: float) -> str:
    if mdi < 30:
        return "Stable"
    elif mdi < 60:
        return "Minor Drift"
    elif mdi < 80:
        return "Moderate Drift"
    return "High Drift -- Monitor"


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the MechanixAE dashboard page."""
    st.title("MechanixAE -- Mechanical Drift Detection")
    st.caption(
        "Variational Autoencoder for detecting pitching mechanical drift and "
        "predicting injury risk.  Higher MDI = greater mechanical divergence "
        "from the pitcher's healthy baseline."
    )

    with st.expander("What does this mean?"):
        st.markdown("""
**MechanixAE monitors pitcher mechanics for subtle drift** that humans can't see — using a neural network trained on each pitcher's "healthy" delivery.

- **MDI 0-40** = normal mechanical variation (all pitchers have some noise)
- **MDI 40-70** = elevated drift (mechanics are shifting — could be fatigue, adjustment, or early injury signal)
- **MDI 70+** = significant mechanical change (strong correlation with upcoming IL stints in historical data)
- **Drift Velocity** matters more than raw MDI — *accelerating* drift is the real red flag
- **Feature Attribution** shows *what* is changing (arm slot dropping? release point moving forward?) so training staff knows where to look
- **Impact:** Catching mechanical drift 2-3 starts before a pitcher hits the IL can save $10M+ in lost production and prevent minor issues from becoming major injuries
""")

    if not _MECHANIX_AVAILABLE:
        st.error(
            "The `mechanix_ae` analytics module could not be imported. "
            "Ensure PyTorch is installed:\n\n"
            "```\npip install torch --index-url https://download.pytorch.org/whl/cpu\n```"
        )
        return

    if not _PLOTLY_AVAILABLE:
        st.error("Plotly is required:\n\n```\npip install plotly\n```")
        return

    conn = get_db_connection()
    if conn is None or not has_data(conn):
        st.warning(
            "No pitch data available.  Run the data backfill pipeline first."
        )
        return

    # ── Tabs ──────────────────────────────────────────────────────────────
    tab_pitcher, tab_alerts = st.tabs(
        ["Pitcher Analysis", "Alert System"]
    )

    with tab_pitcher:
        _render_pitcher_analysis(conn)

    with tab_alerts:
        _render_alert_system(conn)


# ---------------------------------------------------------------------------
# Pitcher Analysis
# ---------------------------------------------------------------------------


def _render_pitcher_analysis(conn) -> None:
    """Full MechanixAE profile for a single pitcher."""

    pitcher_id, pitcher_name = _select_pitcher(conn, key_suffix="mechanix")
    if pitcher_id is None:
        return

    # Attempt to load model; offer to train if missing
    model = None
    checkpoint = None
    try:
        model, checkpoint = _load_model(pitcher_id)
    except FileNotFoundError:
        st.info(
            "No trained MechanixAE model found.  "
            "Click below to train a quick model on this pitcher's data."
        )
        if st.button("Train Model", key="mechanix_train_btn"):
            with st.spinner("Training MechanixAE (this may take a moment)..."):
                result = train_mechanix_ae(conn, pitcher_id=pitcher_id, epochs=15)
            if result["status"] == "trained":
                st.success(f"Model trained! Final loss: {result['final_loss']:.5f}")
                model, checkpoint = _load_model(pitcher_id)
            else:
                st.warning("Training failed -- not enough data for this pitcher.")
                return
        else:
            return

    # ── Compute MDI ───────────────────────────────────────────────────────
    with st.spinner("Computing MDI..."):
        mdi_result = calculate_mdi(
            conn, pitcher_id, model=model, checkpoint=checkpoint,
        )
        dv_result = calculate_drift_velocity(
            conn, pitcher_id, model=model, checkpoint=checkpoint,
        )

    if mdi_result["mdi"] is None:
        st.info("Not enough recent pitches to compute MDI.")
        return

    mdi_val = mdi_result["mdi"]

    # ── MDI Gauge ─────────────────────────────────────────────────────────
    st.subheader("Mechanical Drift Index")
    _render_mdi_gauge(mdi_val, pitcher_name)

    # ── Header metrics ────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MDI", f"{mdi_val:.1f}")
    col2.metric("Status", _mdi_label(mdi_val))
    dv = dv_result.get("drift_velocity")
    col3.metric("Drift Velocity", f"{dv:.4f}" if dv is not None else "N/A")
    col4.metric("Trend", dv_result.get("trend", "N/A").replace("_", " ").title())

    st.markdown("---")

    # ── Drift Timeline ────────────────────────────────────────────────────
    errors = mdi_result.get("recon_errors", [])
    if len(errors) > 1:
        _render_drift_timeline(errors, pitcher_name)
        st.markdown("---")

    # ── Latent Space Trajectory ───────────────────────────────────────────
    if model is not None and _TORCH_AVAILABLE:
        _render_latent_trajectory(conn, pitcher_id, model, pitcher_name)
        st.markdown("---")

    # ── Feature Attribution ───────────────────────────────────────────────
    if model is not None and _TORCH_AVAILABLE:
        _render_feature_attribution(conn, pitcher_id, model, pitcher_name)


# ---------------------------------------------------------------------------
# MDI Gauge
# ---------------------------------------------------------------------------


def _render_mdi_gauge(mdi: float, pitcher_name: str) -> None:
    """Plotly gauge chart for the current MDI value."""
    color = _mdi_color(mdi)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=mdi,
        title={"text": f"{pitcher_name} -- MDI"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 30], "color": "rgba(46,204,113,0.2)"},
                {"range": [30, 60], "color": "rgba(241,196,15,0.2)"},
                {"range": [60, 80], "color": "rgba(230,126,34,0.2)"},
                {"range": [80, 100], "color": "rgba(231,76,60,0.2)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": mdi,
            },
        },
    ))
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=30, r=30, t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True, key="mdi_gauge")


# ---------------------------------------------------------------------------
# Drift Timeline
# ---------------------------------------------------------------------------


def _render_drift_timeline(errors: list[float], pitcher_name: str) -> None:
    """MDI over season with changepoint markers."""
    st.subheader("Drift Timeline")

    error_arr = np.array(errors)
    # Convert to percentile scale for display
    mdi_series = []
    for i, e in enumerate(error_arr):
        pct = float(np.mean(error_arr[:i + 1] <= e) * 100.0)
        mdi_series.append(pct)

    x = list(range(len(mdi_series)))
    cps = detect_changepoints(error_arr)

    fig = go.Figure()

    # MDI line
    fig.add_trace(go.Scatter(
        x=x,
        y=mdi_series,
        mode="lines",
        name="MDI",
        line=dict(color="#3498DB", width=2),
    ))

    # Changepoint markers
    if cps:
        fig.add_trace(go.Scatter(
            x=cps,
            y=[mdi_series[i] for i in cps if i < len(mdi_series)],
            mode="markers",
            name="Changepoint",
            marker=dict(
                color="#E74C3C",
                size=12,
                symbol="diamond",
                line=dict(color="white", width=1),
            ),
        ))

    # Color zones
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(46,204,113,0.08)", line_width=0)
    fig.add_hrect(y0=30, y1=60, fillcolor="rgba(241,196,15,0.08)", line_width=0)
    fig.add_hrect(y0=60, y1=80, fillcolor="rgba(230,126,34,0.08)", line_width=0)
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(231,76,60,0.08)", line_width=0)

    fig.update_layout(
        xaxis=dict(title="Window Index"),
        yaxis=dict(title="MDI (0-100)", range=[0, 105]),
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=30, t=30, b=50),
        title=f"{pitcher_name} -- Mechanical Drift Over Time",
    )
    st.plotly_chart(fig, use_container_width=True, key="drift_timeline")

    if cps:
        st.caption(f"Detected {len(cps)} mechanical changepoint(s) at window indices: {cps}")


# ---------------------------------------------------------------------------
# Latent Space Trajectory
# ---------------------------------------------------------------------------


def _render_latent_trajectory(
    conn, pitcher_id: int, model: MechanixVAE, pitcher_name: str,
) -> None:
    """2D projection of latent space using first 2 latent dimensions."""
    st.subheader("Latent Space Trajectory")

    df = _fetch_pitch_data(conn, pitcher_id=pitcher_id)
    if df.empty or len(df) < WINDOW_SIZE:
        st.info("Not enough pitches for latent trajectory.")
        return

    df = prepare_features(df)
    df, _ = normalize_pitcher_pitch_type(df, columns=FEATURE_COLS)
    available = [c for c in FEATURE_COLS if c in df.columns]
    feat_df = df[available].fillna(0.0)
    feat_matrix = feat_df.values.astype(np.float32)

    if feat_matrix.shape[1] < N_FEATURES:
        pad_width = N_FEATURES - feat_matrix.shape[1]
        feat_matrix = np.pad(feat_matrix, ((0, 0), (0, pad_width)))

    windows = build_sliding_windows(feat_matrix, WINDOW_SIZE)
    if len(windows) == 0:
        st.info("Not enough data for latent projection.")
        return

    tensor = torch.tensor(windows, dtype=torch.float32).permute(0, 2, 1)
    with torch.no_grad():
        latent = model.get_latent(tensor).numpy()

    if latent.shape[1] < 2:
        st.info("Latent space too small for 2D projection.")
        return

    z1 = latent[:, 0]
    z2 = latent[:, 1]
    idx = np.arange(len(z1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=z1.tolist(),
        y=z2.tolist(),
        mode="lines+markers",
        marker=dict(
            size=5,
            color=idx.tolist(),
            colorscale="Viridis",
            colorbar=dict(title="Window #"),
            showscale=True,
        ),
        line=dict(color="rgba(150,150,150,0.3)", width=1),
        text=[f"Window {i}" for i in idx],
        hoverinfo="text",
    ))

    # Mark the most recent window
    fig.add_trace(go.Scatter(
        x=[z1[-1]],
        y=[z2[-1]],
        mode="markers",
        name="Current",
        marker=dict(color="#E74C3C", size=14, symbol="star"),
    ))

    fig.update_layout(
        xaxis=dict(title="Latent Dim 1"),
        yaxis=dict(title="Latent Dim 2"),
        template="plotly_dark",
        height=450,
        margin=dict(l=50, r=30, t=30, b=50),
        title=f"{pitcher_name} -- Mechanical Latent Trajectory",
    )
    st.plotly_chart(fig, use_container_width=True, key="latent_trajectory")


# ---------------------------------------------------------------------------
# Feature Attribution
# ---------------------------------------------------------------------------


def _render_feature_attribution(
    conn, pitcher_id: int, model: MechanixVAE, pitcher_name: str,
) -> None:
    """Show which input features contribute most to current drift.

    Uses simple delta analysis: compare the most recent window's
    reconstruction error per feature vs the average window.
    """
    st.subheader("Feature Attribution")

    df = _fetch_pitch_data(conn, pitcher_id=pitcher_id)
    if df.empty or len(df) < WINDOW_SIZE * 2:
        st.info("Not enough data for feature attribution.")
        return

    df = prepare_features(df)
    df, _ = normalize_pitcher_pitch_type(df, columns=FEATURE_COLS)
    available = [c for c in FEATURE_COLS if c in df.columns]
    feat_df = df[available].fillna(0.0)
    feat_matrix = feat_df.values.astype(np.float32)

    if feat_matrix.shape[1] < N_FEATURES:
        pad_width = N_FEATURES - feat_matrix.shape[1]
        feat_matrix = np.pad(feat_matrix, ((0, 0), (0, pad_width)))

    windows = build_sliding_windows(feat_matrix, WINDOW_SIZE)
    if len(windows) < 2:
        st.info("Not enough windows for attribution.")
        return

    tensor = torch.tensor(windows, dtype=torch.float32).permute(0, 2, 1)
    with torch.no_grad():
        recon, _, _ = model(tensor)

    # Per-feature reconstruction error: (batch, features, time) -> mean over time
    per_feature_error = ((tensor - recon) ** 2).mean(dim=2).numpy()

    # Latest window vs average
    latest_errors = per_feature_error[-1]
    avg_errors = per_feature_error[:-1].mean(axis=0)
    delta = latest_errors - avg_errors

    # Use FEATURE_COLS names (pad if needed)
    feature_names = FEATURE_COLS[:len(delta)]
    if len(feature_names) < len(delta):
        feature_names += [f"feat_{i}" for i in range(len(feature_names), len(delta))]

    # Sort by absolute delta
    sorted_idx = np.argsort(np.abs(delta))[::-1]

    fig = go.Figure()
    colors = ["#E74C3C" if d > 0 else "#2ECC71" for d in delta[sorted_idx]]

    fig.add_trace(go.Bar(
        x=[feature_names[i] for i in sorted_idx],
        y=[delta[i] for i in sorted_idx],
        marker_color=colors,
        text=[f"{delta[i]:+.4f}" for i in sorted_idx],
        textposition="auto",
    ))

    fig.update_layout(
        xaxis=dict(title="Feature", tickangle=-45),
        yaxis=dict(title="Error Delta (vs average)"),
        template="plotly_dark",
        height=400,
        margin=dict(l=50, r=30, t=30, b=100),
        title=f"{pitcher_name} -- Feature Contribution to Current Drift",
    )
    st.plotly_chart(fig, use_container_width=True, key="feature_attribution")

    st.caption(
        "Red bars = features with higher-than-average reconstruction error "
        "(driving current drift).  Green bars = features that are more "
        "stable than average."
    )


# ---------------------------------------------------------------------------
# Alert System
# ---------------------------------------------------------------------------


def _render_alert_system(conn) -> None:
    """List pitchers with MDI above the 80th percentile."""
    st.subheader("Mechanical Drift Alerts")
    st.caption(
        "Pitchers whose current MDI exceeds the 80th percentile "
        "of all computed MDI values.  These arms may be at elevated "
        "risk for performance decline or injury."
    )

    min_pitches = st.slider(
        "Minimum pitches to qualify", 50, 1000, 200, step=50,
        key="mechanix_alert_min",
    )

    # Try cache first
    leaderboard = None
    if _CACHE_AVAILABLE:
        try:
            cached = get_cached_leaderboard(conn, "mechanix_ae", None)
            if cached is not None:
                leaderboard = cached
                age_info = cache_age_display(conn, "mechanix_ae", None)
                if age_info:
                    st.caption(age_info)
        except Exception:
            pass

    if leaderboard is None:
        try:
            with st.spinner("Computing... Run `python scripts/precompute.py` for instant loading."):
                leaderboard = batch_calculate(conn, min_pitches=min_pitches)
        except Exception as exc:
            st.error(f"Error computing leaderboard: {exc}")
            return

    if leaderboard.empty:
        st.info("No qualifying pitchers found.")
        return

    # Filter to those with valid MDI
    valid = leaderboard.dropna(subset=["mdi"])
    if valid.empty:
        st.info("No MDI values computed.")
        return

    threshold = float(valid["mdi"].quantile(0.80))
    alerts = valid[valid["mdi"] >= threshold].copy()

    st.markdown(f"**Alert Threshold (80th pct):** MDI >= {threshold:.1f}")

    if alerts.empty:
        st.success("No pitchers currently above the alert threshold.")
    else:
        alerts = alerts.sort_values("mdi", ascending=False).reset_index(drop=True)
        alerts.index = alerts.index + 1
        alerts.index.name = "Rank"

        st.dataframe(
            alerts,
            use_container_width=True,
            column_config={
                "mdi": st.column_config.NumberColumn("MDI", format="%.1f"),
                "drift_velocity": st.column_config.NumberColumn("Drift Vel", format="%.4f"),
                "trend": st.column_config.TextColumn("Trend"),
                "name": st.column_config.TextColumn("Pitcher"),
            },
        )

        # Alert summary chart
        fig = go.Figure()
        names = alerts["name"].fillna(alerts["pitcher_id"].astype(str)).tolist()
        mdis = alerts["mdi"].tolist()
        colors = [_mdi_color(m) for m in mdis]

        fig.add_trace(go.Bar(
            x=names,
            y=mdis,
            marker_color=colors,
            text=[f"{m:.1f}" for m in mdis],
            textposition="auto",
        ))

        fig.add_hline(
            y=threshold,
            line=dict(color="white", width=2, dash="dash"),
            annotation_text=f"80th pct: {threshold:.1f}",
            annotation_position="top right",
        )

        fig.update_layout(
            xaxis=dict(title="Pitcher", tickangle=-45),
            yaxis=dict(title="MDI", range=[0, 105]),
            template="plotly_dark",
            height=400,
            margin=dict(l=50, r=30, t=30, b=100),
        )
        st.plotly_chart(fig, use_container_width=True, key="alert_chart")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _select_pitcher(conn, key_suffix: str = "") -> tuple[int | None, str]:
    """Render a pitcher selector and return (pitcher_id, name)."""
    pitchers = get_all_pitchers(conn)
    if not pitchers:
        try:
            fallback = conn.execute(
                "SELECT DISTINCT pitcher_id FROM pitches LIMIT 500"
            ).fetchdf()
            pitchers = [
                {"player_id": int(pid), "full_name": f"Pitcher {pid}"}
                for pid in fallback["pitcher_id"]
            ]
        except Exception:
            st.info("No pitcher data available.")
            return None, ""

    if not pitchers:
        st.info("No pitchers found.")
        return None, ""

    pitcher_names = {
        p.get("full_name", f"ID {p['player_id']}"): p["player_id"]
        for p in pitchers
    }
    selected_name = st.selectbox(
        "Select Pitcher",
        options=sorted(pitcher_names.keys()),
        key=f"mechanix_pitcher_select_{key_suffix}",
    )

    if not selected_name:
        return None, ""

    return pitcher_names[selected_name], selected_name
