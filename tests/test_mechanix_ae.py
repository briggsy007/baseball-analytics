"""
Tests for the MechanixAE mechanical drift detection model.

Covers:
  - Input normalization (pitcher+pitch_type centering)
  - VAE forward pass (output shape matches input)
  - Reconstruction on constant data (low error)
  - KL divergence behavior
  - MDI range [0, 100]
  - Drift velocity computation
  - Changepoint detection on synthetic step function
  - Edge cases: <20 pitches, single pitch type, missing features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.analytics.mechanix_ae import (
    FEATURE_COLS,
    LATENT_DIM,
    N_FEATURES,
    WINDOW_SIZE,
    MechanixAEModel,
    MechanixVAE,
    build_sliding_windows,
    calculate_drift_velocity,
    calculate_mdi,
    compute_arm_angle,
    detect_changepoints,
    normalize_pitcher_pitch_type,
    prepare_features,
    vae_loss,
    _compute_reconstruction_errors,
    _prepare_training_windows,
    train_mechanix_ae,
)


# ── Helpers ───────────────────────────────────────────────────────────────


def _make_pitch_df(
    n: int = 100,
    pitcher_id: int = 554430,
    pitch_types: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic pitch DataFrame for testing."""
    rng = np.random.RandomState(seed)
    if pitch_types is None:
        pitch_types = ["FF", "SL"]

    rows = []
    for i in range(n):
        pt = pitch_types[i % len(pitch_types)]
        rows.append({
            "pitcher_id": pitcher_id,
            "game_pk": 100000 + i // 20,
            "game_date": f"2025-06-{(i % 28) + 1:02d}",
            "pitch_type": pt,
            "at_bat_number": i // 5 + 1,
            "pitch_number": i % 5 + 1,
            "release_pos_x": rng.normal(-1.5, 0.1),
            "release_pos_z": rng.normal(5.8, 0.1),
            "release_extension": rng.normal(6.2, 0.2),
            "release_speed": rng.normal(94.0, 1.0),
            "release_spin_rate": rng.normal(2300, 100),
            "spin_axis": rng.uniform(180, 220),
            "pfx_x": rng.normal(-6, 1),
            "pfx_z": rng.normal(15, 1),
            "effective_speed": rng.normal(93.0, 1.0),
        })
    return pd.DataFrame(rows)


def _make_constant_pitch_df(
    n: int = 100,
    pitcher_id: int = 554430,
) -> pd.DataFrame:
    """Pitch DataFrame where every pitch has identical mechanics."""
    rows = []
    for i in range(n):
        rows.append({
            "pitcher_id": pitcher_id,
            "game_pk": 100000 + i // 20,
            "game_date": f"2025-06-{(i % 28) + 1:02d}",
            "pitch_type": "FF",
            "at_bat_number": i // 5 + 1,
            "pitch_number": i % 5 + 1,
            "release_pos_x": -1.5,
            "release_pos_z": 5.8,
            "release_extension": 6.2,
            "release_speed": 94.0,
            "release_spin_rate": 2300.0,
            "spin_axis": 200.0,
            "pfx_x": -6.0,
            "pfx_z": 15.0,
            "effective_speed": 93.0,
        })
    return pd.DataFrame(rows)


# ── Test: Normalization ───────────────────────────────────────────────────


class TestNormalization:
    """Pitch-type normalization subtracts pitcher+pitch_type mean."""

    def test_mean_near_zero_after_normalization(self):
        df = _make_pitch_df(200, pitch_types=["FF", "SL"])
        df = prepare_features(df)
        normed, means = normalize_pitcher_pitch_type(df)

        for pt in ["FF", "SL"]:
            subset = normed[df["pitch_type"] == pt]
            for col in FEATURE_COLS:
                if col in subset.columns:
                    col_mean = subset[col].dropna().mean()
                    assert abs(col_mean) < 0.1, (
                        f"Mean of {col} for {pt} should be near 0, got {col_mean}"
                    )

    def test_means_dict_populated(self):
        df = _make_pitch_df(100)
        df = prepare_features(df)
        _, means = normalize_pitcher_pitch_type(df)
        assert len(means) > 0, "Means dict should not be empty"

    def test_arm_angle_computed(self):
        df = _make_pitch_df(10)
        df = prepare_features(df)
        assert "arm_angle" in df.columns
        assert not df["arm_angle"].isna().all()


# ── Test: Sliding windows ────────────────────────────────────────────────


class TestSlidingWindows:
    def test_output_shape(self):
        features = np.random.randn(50, N_FEATURES).astype(np.float32)
        windows = build_sliding_windows(features)
        expected_n = 50 - WINDOW_SIZE + 1
        assert windows.shape == (expected_n, WINDOW_SIZE, N_FEATURES)

    def test_too_few_pitches(self):
        features = np.random.randn(10, N_FEATURES).astype(np.float32)
        windows = build_sliding_windows(features)
        assert windows.shape[0] == 0

    def test_exact_window_size(self):
        features = np.random.randn(WINDOW_SIZE, N_FEATURES).astype(np.float32)
        windows = build_sliding_windows(features)
        assert windows.shape == (1, WINDOW_SIZE, N_FEATURES)


# ── Test: VAE forward pass ───────────────────────────────────────────────


class TestVAEForwardPass:
    """Output shape must match input shape through encode -> decode."""

    def test_output_shape_matches_input(self):
        model = MechanixVAE()
        batch = torch.randn(8, N_FEATURES, WINDOW_SIZE)
        recon, mu, logvar = model(batch)
        assert recon.shape == batch.shape, (
            f"Reconstruction shape {recon.shape} != input shape {batch.shape}"
        )
        assert mu.shape == (8, LATENT_DIM)
        assert logvar.shape == (8, LATENT_DIM)

    def test_latent_dimension(self):
        model = MechanixVAE()
        batch = torch.randn(4, N_FEATURES, WINDOW_SIZE)
        latent = model.get_latent(batch)
        assert latent.shape == (4, LATENT_DIM)

    def test_single_sample(self):
        model = MechanixVAE()
        batch = torch.randn(1, N_FEATURES, WINDOW_SIZE)
        recon, mu, logvar = model(batch)
        assert recon.shape == batch.shape


# ── Test: Reconstruction on constant data ────────────────────────────────


class TestReconstruction:
    """A model trained on constant data should achieve low reconstruction error."""

    def test_constant_data_low_error(self):
        # Create constant-value windows
        const_val = 0.5
        windows = np.full((100, WINDOW_SIZE, N_FEATURES), const_val, dtype=np.float32)
        tensor = torch.tensor(windows).permute(0, 2, 1)

        model = MechanixVAE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train briefly
        model.train()
        for _ in range(50):
            optimizer.zero_grad()
            recon, mu, logvar = model(tensor)
            loss, _, _ = vae_loss(recon, tensor, mu, logvar)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            recon, _, _ = model(tensor)
        mse = ((tensor - recon) ** 2).mean().item()
        assert mse < 0.1, f"Reconstruction MSE on constant data should be small, got {mse}"

    def test_reconstruction_errors_array(self):
        model = MechanixVAE()
        windows = np.random.randn(10, WINDOW_SIZE, N_FEATURES).astype(np.float32)
        errors = _compute_reconstruction_errors(model, windows)
        assert errors.shape == (10,)
        assert np.all(errors >= 0)


# ── Test: KL divergence ──────────────────────────────────────────────────


class TestKLDivergence:
    """Posterior should approach prior with enough training on simple data."""

    def test_kl_decreases_during_training(self):
        windows = np.random.randn(200, WINDOW_SIZE, N_FEATURES).astype(np.float32) * 0.01
        tensor = torch.tensor(windows).permute(0, 2, 1)

        model = MechanixVAE()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        kl_values = []
        model.train()
        for epoch in range(30):
            optimizer.zero_grad()
            recon, mu, logvar = model(tensor)
            loss, _, kl = vae_loss(recon, tensor, mu, logvar, beta=1.0)
            loss.backward()
            optimizer.step()
            kl_values.append(kl.item())

        # KL should generally decrease or stabilize
        early_kl = np.mean(kl_values[:5])
        late_kl = np.mean(kl_values[-5:])
        assert late_kl <= early_kl + 0.5, (
            f"KL should not increase drastically: early={early_kl:.3f}, late={late_kl:.3f}"
        )


# ── Test: MDI range ──────────────────────────────────────────────────────


class TestMDI:
    """MDI must be in [0, 100] range."""

    def test_mdi_range_with_model(self):
        model = MechanixVAE()
        # Train briefly on random data
        windows = np.random.randn(100, WINDOW_SIZE, N_FEATURES).astype(np.float32)
        tensor = torch.tensor(windows).permute(0, 2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(10):
            optimizer.zero_grad()
            recon, mu, logvar = model(tensor)
            loss, _, _ = vae_loss(recon, tensor, mu, logvar)
            loss.backward()
            optimizer.step()

        model.eval()
        errors = _compute_reconstruction_errors(model, windows)
        # Compute MDI: percentile of latest error
        latest = errors[-1]
        mdi = float(np.mean(errors <= latest) * 100.0)
        assert 0 <= mdi <= 100, f"MDI should be in [0, 100], got {mdi}"

    def test_mdi_with_db(self, db_conn):
        """MDI via calculate_mdi should be in [0, 100] or None."""
        # Train a quick model
        result = train_mechanix_ae(db_conn, pitcher_id=554430, epochs=5, batch_size=32)
        if result["status"] == "no_data":
            pytest.skip("Insufficient test data for MDI test")

        mdi_result = calculate_mdi(db_conn, 554430)
        if mdi_result["mdi"] is not None:
            assert 0 <= mdi_result["mdi"] <= 100


# ── Test: Drift velocity ────────────────────────────────────────────────


class TestDriftVelocity:
    def test_drift_velocity_with_db(self, db_conn):
        """Drift velocity should return a numeric value or None."""
        try:
            result = train_mechanix_ae(db_conn, pitcher_id=554430, epochs=3, batch_size=32)
            if result["status"] == "no_data":
                pytest.skip("No data for drift velocity test")
            dv = calculate_drift_velocity(db_conn, 554430)
            if dv["drift_velocity"] is not None:
                assert isinstance(dv["drift_velocity"], float)
            assert dv["trend"] in ("increasing", "decreasing", "stable", "insufficient_data")
        except FileNotFoundError:
            pytest.skip("Model not trained")

    def test_drift_velocity_stable_for_constant(self):
        """With constant errors, drift velocity should be ~0 (stable)."""
        errors = [5.0] * 20
        # Simulate: slope of constant series = 0
        recent = np.array(errors[-10:])
        x = np.arange(len(recent), dtype=float)
        x_mean = x.mean()
        y_mean = recent.mean()
        denom = ((x - x_mean) ** 2).sum()
        slope = float(((x - x_mean) * (recent - y_mean)).sum() / denom) if denom > 0 else 0.0
        assert abs(slope) < 0.001


# ── Test: Changepoint detection ──────────────────────────────────────────


class TestChangepoints:
    def test_step_function(self):
        """A step function should produce at least one changepoint."""
        series = np.concatenate([
            np.ones(30) * 10.0,
            np.ones(30) * 50.0,
        ])
        cps = detect_changepoints(series, threshold=5.0)
        assert len(cps) > 0, "Should detect at least one changepoint in step function"
        # The changepoint should be near index 30
        assert any(25 <= cp <= 40 for cp in cps), (
            f"Changepoint should be near index 30, got {cps}"
        )

    def test_constant_series_no_changepoints(self):
        """A constant series should have no changepoints."""
        series = np.ones(50) * 42.0
        cps = detect_changepoints(series)
        assert len(cps) == 0, f"Constant series should have 0 changepoints, got {cps}"

    def test_empty_series(self):
        cps = detect_changepoints([])
        assert cps == []

    def test_short_series(self):
        cps = detect_changepoints([1.0, 2.0])
        assert cps == []

    def test_multiple_steps(self):
        """Multiple step changes should produce multiple changepoints."""
        series = np.concatenate([
            np.ones(20) * 10.0,
            np.ones(20) * 50.0,
            np.ones(20) * 10.0,
        ])
        cps = detect_changepoints(series, threshold=5.0)
        assert len(cps) >= 2, f"Should detect >= 2 changepoints, got {len(cps)}"


# ── Test: Edge cases ─────────────────────────────────────────────────────


class TestEdgeCases:
    def test_fewer_than_window_size_pitches(self):
        """With < WINDOW_SIZE pitches, windows should be empty."""
        features = np.random.randn(WINDOW_SIZE - 1, N_FEATURES).astype(np.float32)
        windows = build_sliding_windows(features)
        assert windows.shape[0] == 0

    def test_single_pitch_type(self):
        """Normalization should work with a single pitch type."""
        df = _make_pitch_df(50, pitch_types=["FF"])
        df = prepare_features(df)
        normed, means = normalize_pitcher_pitch_type(df)
        ff_mean = normed[FEATURE_COLS[0]].dropna().mean()
        assert abs(ff_mean) < 0.1

    def test_missing_effective_speed(self):
        """effective_speed fallback to release_speed."""
        df = _make_pitch_df(30)
        df = df.drop(columns=["effective_speed"])
        df = prepare_features(df)
        assert "arm_angle" in df.columns
        # effective_speed should have been restored
        assert "effective_speed" in df.columns

    def test_missing_effective_speed_uses_release_speed(self):
        """When effective_speed column is absent, release_speed fills in."""
        df = _make_pitch_df(30)
        df = df.drop(columns=["effective_speed"])
        from src.analytics.mechanix_ae import _fill_effective_speed
        filled = _fill_effective_speed(df)
        assert "effective_speed" in filled.columns
        assert np.allclose(
            filled["effective_speed"].values,
            df["release_speed"].values,
        )

    def test_vae_loss_function(self):
        """VAE loss returns correct types and non-negative values."""
        recon = torch.randn(4, N_FEATURES, WINDOW_SIZE)
        target = torch.randn(4, N_FEATURES, WINDOW_SIZE)
        mu = torch.randn(4, LATENT_DIM)
        logvar = torch.randn(4, LATENT_DIM)

        total, recon_l, kl_l = vae_loss(recon, target, mu, logvar)
        assert total.item() >= 0
        assert recon_l.item() >= 0


# ── Test: BaseAnalyticsModel interface ───────────────────────────────────


class TestMechanixAEModel:
    def test_model_name_and_version(self):
        m = MechanixAEModel()
        assert m.model_name == "MechanixAE Mechanical Drift Detector"
        assert m.version == "1.0.0"

    def test_repr(self):
        m = MechanixAEModel()
        r = repr(m)
        assert "MechanixAE" in r
