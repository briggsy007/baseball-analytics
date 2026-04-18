"""
MechanixAE -- Variational Autoencoder for Pitching Mechanical Drift Detection.

Encodes a sliding window of pitch-level biomechanical features into a compact
latent space, then measures reconstruction error to produce the **Mechanical
Drift Index (MDI)**.  Rising MDI signals that a pitcher's mechanics are
drifting from their established pattern -- an early warning for both
performance decline and injury risk.

Key outputs:
  - **MDI (0--100)**: Reconstruction-error percentile over recent 20-pitch
    windows.  Higher = more mechanical drift.
  - **Drift Velocity**: Rate of change of MDI over consecutive windows.
  - **Changepoint Detection**: CUSUM-based identification of abrupt
    mechanical shifts in the MDI time series.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.analytics.base import BaseAnalyticsModel

ScoreMode = Literal["percentile", "zscore"]

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    """Return the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Feature configuration ─────────────────────────────────────────────────

FEATURE_COLS: list[str] = [
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "release_speed",
    "release_spin_rate",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "effective_speed",
    "arm_angle",  # derived
]

RAW_FEATURE_COLS: list[str] = [
    "release_pos_x",
    "release_pos_z",
    "release_extension",
    "release_speed",
    "release_spin_rate",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "effective_speed",
]

N_FEATURES: int = 10
WINDOW_SIZE: int = 20
LATENT_DIM: int = 6
BETA_KL: float = 0.1

MODELS_DIR: Path = Path(__file__).resolve().parents[2] / "models"


# ── Feature engineering helpers ───────────────────────────────────────────


def compute_arm_angle(df: pd.DataFrame) -> pd.Series:
    """Derive arm_angle = atan2(release_pos_z - 5.5, release_pos_x).

    Returns angle in degrees.
    """
    z = df["release_pos_z"].astype(float) - 5.5
    x = df["release_pos_x"].astype(float)
    return np.degrees(np.arctan2(z, x))


def _fill_effective_speed(df: pd.DataFrame) -> pd.DataFrame:
    """Use release_speed as fallback when effective_speed is missing."""
    df = df.copy()
    if "effective_speed" not in df.columns:
        df["effective_speed"] = df["release_speed"]
    else:
        df["effective_speed"] = df["effective_speed"].fillna(df["release_speed"])
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns and return only FEATURE_COLS (with NaN rows dropped).

    The input DataFrame must contain at least RAW_FEATURE_COLS plus
    ``pitch_type`` and ``pitcher_id`` (used for normalisation upstream).
    """
    df = _fill_effective_speed(df)
    df = df.copy()
    df["arm_angle"] = compute_arm_angle(df)
    return df


def normalize_pitcher_pitch_type(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Subtract per-pitcher+pitch_type mean from *columns*.

    Returns:
        (normalised_df, means_dict) where means_dict is keyed by
        ``(pitcher_id, pitch_type)`` and valued by a dict of column means.
    """
    if columns is None:
        columns = FEATURE_COLS

    df = df.copy()
    means: dict = {}

    available_cols = [c for c in columns if c in df.columns]
    if not available_cols:
        return df, means

    group_keys = []
    if "pitcher_id" in df.columns and "pitch_type" in df.columns:
        group_keys = ["pitcher_id", "pitch_type"]
    elif "pitcher_id" in df.columns:
        group_keys = ["pitcher_id"]

    if group_keys:
        group_means = df.groupby(group_keys)[available_cols].transform("mean")
        for key, grp in df.groupby(group_keys):
            means[key] = grp[available_cols].mean().to_dict()
        df[available_cols] = df[available_cols] - group_means
    else:
        col_means = df[available_cols].mean()
        means[("global",)] = col_means.to_dict()
        df[available_cols] = df[available_cols] - col_means

    return df, means


def build_sliding_windows(
    features: np.ndarray,
    window_size: int = WINDOW_SIZE,
) -> np.ndarray:
    """Convert a (N, D) feature matrix into sliding windows of shape (M, W, D).

    Args:
        features: Array of shape (N, D).
        window_size: Number of pitches per window.

    Returns:
        Array of shape (N - window_size + 1, window_size, D).
    """
    n = len(features)
    if n < window_size:
        return np.empty((0, window_size, features.shape[1]))
    windows = np.lib.stride_tricks.sliding_window_view(
        features, window_shape=window_size, axis=0,
    )
    # sliding_window_view returns (M, D, W) -- transpose last two axes
    return windows.transpose(0, 2, 1)


# ── VAE Architecture ──────────────────────────────────────────────────────


class MechanixVAE(nn.Module):
    """Small 1-D convolutional VAE for pitch-window encoding.

    Encoder:  Conv1d(10->32, k=3) -> Conv1d(32->64, k=3) -> Flatten -> mu, logvar
    Decoder:  Linear -> Unflatten -> ConvTranspose1d(64->32) -> ConvTranspose1d(32->10)
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        window_size: int = WINDOW_SIZE,
        latent_dim: int = LATENT_DIM,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.latent_dim = latent_dim

        # ── Encoder ───────────────────────────────────────────────────
        # Input: (batch, n_features, window_size) -- channels-first
        self.enc_conv1 = nn.Conv1d(n_features, 32, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self._flat_dim = 64 * window_size  # after flatten

        self.fc_mu = nn.Linear(self._flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self._flat_dim, latent_dim)

        # ── Decoder ───────────────────────────────────────────────────
        self.fc_decode = nn.Linear(latent_dim, self._flat_dim)
        self.dec_conv1 = nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1)
        self.dec_conv2 = nn.ConvTranspose1d(32, n_features, kernel_size=3, padding=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters (mu, logvar)."""
        # x: (batch, n_features, window_size)
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to (batch, n_features, window_size)."""
        h = F.relu(self.fc_decode(z))
        h = h.view(-1, 64, self.window_size)
        h = F.relu(self.dec_conv1(h))
        return self.dec_conv2(h)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass.

        Returns:
            (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent mean vector (deterministic encoding)."""
        mu, _ = self.encode(x)
        return mu


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = BETA_KL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute beta-VAE loss = MSE_recon + beta * KL_divergence.

    Returns:
        (total_loss, recon_loss, kl_loss)
    """
    recon_loss = F.mse_loss(recon, target, reduction="mean")
    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


# ── Data loading ──────────────────────────────────────────────────────────


def _fetch_pitch_data(
    conn,
    pitcher_id: int | None = None,
    season: int | None = None,
    max_pitchers: int | None = None,
) -> pd.DataFrame:
    """Fetch pitch-level data with all required feature columns.

    Args:
        conn: DuckDB connection.
        pitcher_id: If given, fetch only this pitcher's data.
        season: If given, restrict to this season.
        max_pitchers: When pitcher_id is None (universal mode), limit to
            the top *max_pitchers* by pitch count to avoid loading the
            entire 7M+ row table.  Defaults to None (no limit) -- callers
            should set an explicit cap for training.
    """
    cols = ", ".join(RAW_FEATURE_COLS)
    conditions = ["release_speed IS NOT NULL", "pitch_type IS NOT NULL"]
    params: list = []
    param_idx = 1

    if pitcher_id is not None:
        conditions.append(f"pitcher_id = ${param_idx}")
        params.append(pitcher_id)
        param_idx += 1

    if season is not None:
        conditions.append(f"EXTRACT(YEAR FROM game_date) = ${param_idx}")
        params.append(season)
        param_idx += 1

    where = " AND ".join(conditions)

    # When training a universal model, restrict to the top N pitchers by
    # pitch count so we don't try to load all 7M+ rows at once.
    pitcher_filter = ""
    if pitcher_id is None and max_pitchers is not None:
        pitcher_filter = f"""
            AND pitcher_id IN (
                SELECT pitcher_id
                FROM pitches
                WHERE {where}
                GROUP BY pitcher_id
                ORDER BY COUNT(*) DESC
                LIMIT {int(max_pitchers)}
            )
        """

    query = f"""
        SELECT
            pitcher_id,
            game_pk,
            game_date,
            pitch_type,
            at_bat_number,
            pitch_number,
            {cols}
        FROM pitches
        WHERE {where}
            {pitcher_filter}
        ORDER BY game_date, game_pk, at_bat_number, pitch_number
    """
    return conn.execute(query, params).fetchdf()


def _fetch_il_dates(conn, pitcher_id: int) -> list[pd.Timestamp]:
    """Return dates when a pitcher was placed on the injured list."""
    query = """
        SELECT transaction_date
        FROM transactions
        WHERE player_id = $1
          AND description ILIKE '%injured list%'
          AND description NOT ILIKE '%activated%'
    """
    try:
        df = conn.execute(query, [pitcher_id]).fetchdf()
        if df.empty:
            return []
        return pd.to_datetime(df["transaction_date"]).tolist()
    except Exception:
        return []


def _exclude_pre_il_pitches(
    df: pd.DataFrame,
    il_dates: list[pd.Timestamp],
    days_before: int = 30,
) -> pd.DataFrame:
    """Remove pitches within *days_before* of any IL stint."""
    if not il_dates or df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    game_dates = pd.to_datetime(df["game_date"])
    for il_date in il_dates:
        cutoff = il_date - pd.Timedelta(days=days_before)
        mask = mask & ~((game_dates >= cutoff) & (game_dates <= il_date))
    return df[mask].reset_index(drop=True)


def _prepare_training_windows(
    conn,
    pitcher_id: int | None = None,
    exclude_pre_il: bool = True,
    max_pitchers: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Fetch data, engineer features, normalise, and build windows.

    Returns:
        (windows_array of shape (M, WINDOW_SIZE, N_FEATURES), normalisation_means)
    """
    df = _fetch_pitch_data(conn, pitcher_id=pitcher_id, max_pitchers=max_pitchers)
    if df.empty:
        return np.empty((0, WINDOW_SIZE, N_FEATURES)), {}

    # Exclude pre-IL pitches for healthy baseline training
    if exclude_pre_il and pitcher_id is not None:
        il_dates = _fetch_il_dates(conn, pitcher_id)
        df = _exclude_pre_il_pitches(df, il_dates)

    if df.empty or len(df) < WINDOW_SIZE:
        return np.empty((0, WINDOW_SIZE, N_FEATURES)), {}

    df = prepare_features(df)
    df, means = normalize_pitcher_pitch_type(df, columns=FEATURE_COLS)

    # Extract feature matrix
    available = [c for c in FEATURE_COLS if c in df.columns]
    feat_df = df[available].copy()
    # Fill remaining NaN with 0 (already mean-subtracted, so 0 = average)
    feat_df = feat_df.fillna(0.0)
    feat_matrix = feat_df.values.astype(np.float32)

    # Standardise: divide by per-column std so all features are O(1).
    # The mean-subtraction above centres each pitcher+pitch_type combo,
    # but scale varies wildly (e.g. spin_axis ~0-360, pfx ~-20-20).
    col_std = feat_matrix.std(axis=0)
    col_std[col_std < 1e-8] = 1.0  # avoid division by zero
    feat_matrix = feat_matrix / col_std
    # Clip extreme outliers to prevent NaN during training
    feat_matrix = np.clip(feat_matrix, -10.0, 10.0)

    # Pad if fewer features than expected
    if feat_matrix.shape[1] < N_FEATURES:
        pad_width = N_FEATURES - feat_matrix.shape[1]
        feat_matrix = np.pad(feat_matrix, ((0, 0), (0, pad_width)))

    windows = build_sliding_windows(feat_matrix, WINDOW_SIZE)
    return windows, means


# ── Training ──────────────────────────────────────────────────────────────


def train_mechanix_ae(
    conn,
    pitcher_id: int | None = None,
    epochs: int = 20,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    max_pitchers: int = 50,
) -> dict:
    """Train the MechanixAE model.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: If given, train a per-pitcher model (healthy baseline
                    only -- excludes 30 days before any IL stint).  If None,
                    train a universal model.
        epochs: Training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        max_pitchers: When training a universal model (pitcher_id=None),
                      limit to this many pitchers (by pitch count) to keep
                      memory and runtime manageable.  Default 50 (~200-500K
                      pitches).

    Returns:
        Training metrics dictionary.
    """
    windows, means = _prepare_training_windows(
        conn, pitcher_id=pitcher_id, max_pitchers=max_pitchers,
    )

    if len(windows) == 0:
        logger.warning(
            "No training windows available for pitcher_id=%s", pitcher_id
        )
        return {"status": "no_data", "n_windows": 0}

    logger.info(
        "Training MechanixAE: %d windows, pitcher_id=%s",
        len(windows), pitcher_id,
    )

    # Convert to tensors -- channels-first: (batch, features, window)
    device = _get_device()
    tensor_data = torch.tensor(windows, dtype=torch.float32).permute(0, 2, 1)
    dataset = TensorDataset(tensor_data, tensor_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MechanixVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    history: list[dict] = []

    for epoch in range(epochs):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_batches = 0

        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch_x)
            loss, recon_l, kl_l = vae_loss(recon, batch_x, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_l.item()
            total_kl += kl_l.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        avg_recon = total_recon / max(n_batches, 1)
        avg_kl = total_kl / max(n_batches, 1)
        history.append({
            "epoch": epoch + 1,
            "loss": round(avg_loss, 6),
            "recon_loss": round(avg_recon, 6),
            "kl_loss": round(avg_kl, 6),
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d  loss=%.5f  recon=%.5f  kl=%.5f",
                epoch + 1, epochs, avg_loss, avg_recon, avg_kl,
            )

    # Compute healthy baseline reconstruction-error stats so inference-time
    # z-score scoring (score_mode="zscore") can normalise against them.
    model.eval()
    baseline_errors = _compute_reconstruction_errors(model, windows)
    healthy_recon_mean = float(baseline_errors.mean()) if len(baseline_errors) else 0.0
    healthy_recon_std = float(baseline_errors.std()) if len(baseline_errors) else 0.0

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tag = str(pitcher_id) if pitcher_id else "universal"
    save_path = MODELS_DIR / f"mechanix_ae_{tag}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "means": means,
        "n_features": N_FEATURES,
        "window_size": WINDOW_SIZE,
        "latent_dim": LATENT_DIM,
        "healthy_recon_mean": healthy_recon_mean,
        "healthy_recon_std": healthy_recon_std,
    }, save_path)
    logger.info("Model saved to %s", save_path)

    return {
        "status": "trained",
        "n_windows": len(windows),
        "epochs": epochs,
        "final_loss": history[-1]["loss"],
        "final_recon_loss": history[-1]["recon_loss"],
        "final_kl_loss": history[-1]["kl_loss"],
        "history": history,
        "model_path": str(save_path),
    }


def _load_model(pitcher_id: int | None = None) -> tuple[MechanixVAE, dict]:
    """Load a saved model from disk.

    Falls back to the universal model if a pitcher-specific model is not found.

    Returns:
        (model, checkpoint_dict)
    """
    tag = str(pitcher_id) if pitcher_id else "universal"
    path = MODELS_DIR / f"mechanix_ae_{tag}.pt"

    if not path.exists() and pitcher_id is not None:
        path = MODELS_DIR / "mechanix_ae_universal.pt"

    if not path.exists():
        raise FileNotFoundError(
            f"No MechanixAE model found at {path}.  Train one first."
        )

    device = _get_device()
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        logger.warning(
            "weights_only=True failed for %s; falling back to weights_only=False. "
            "Only load checkpoints from trusted sources.",
            path,
        )
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = MechanixVAE(
        n_features=checkpoint.get("n_features", N_FEATURES),
        window_size=checkpoint.get("window_size", WINDOW_SIZE),
        latent_dim=checkpoint.get("latent_dim", LATENT_DIM),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


# ── Inference ─────────────────────────────────────────────────────────────


def _compute_reconstruction_errors(
    model: MechanixVAE,
    windows: np.ndarray,
) -> np.ndarray:
    """Compute per-window MSE reconstruction error.

    Args:
        model: Trained MechanixVAE (in eval mode).
        windows: Array of shape (M, window_size, n_features).

    Returns:
        1-D array of length M with per-window reconstruction errors.
    """
    if len(windows) == 0:
        return np.array([])

    device = next(model.parameters()).device
    tensor = torch.tensor(windows, dtype=torch.float32).permute(0, 2, 1).to(device)
    with torch.no_grad():
        recon, _, _ = model(tensor)
    # Per-window MSE: mean across features and time steps
    errors = ((tensor - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
    return errors


def compute_mdi_zscore(
    errors: np.ndarray,
    healthy_mean: float,
    healthy_std: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """Convert raw reconstruction errors to a magnitude-based z-score.

    ``z = (error - healthy_mean) / max(healthy_std, eps)``.  Higher z values
    indicate stronger mechanical drift relative to the pitcher's healthy
    baseline distribution.  Unlike percentile-rank MDI, z-scores are not
    bounded to [0, 100] and do not saturate at the top of each evaluation
    window.

    Args:
        errors: 1-D array of reconstruction errors.
        healthy_mean: Mean reconstruction error on the healthy training set.
        healthy_std: Std of reconstruction errors on the healthy training set.
        eps: Minimum denominator to avoid division by zero.

    Returns:
        1-D array of z-scores (same length as ``errors``).
    """
    if len(errors) == 0:
        return np.array([], dtype=np.float32)
    denom = max(float(healthy_std), eps)
    return (np.asarray(errors, dtype=float) - float(healthy_mean)) / denom


def calculate_mdi(
    conn,
    pitcher_id: int,
    game_pk: int | None = None,
    model: MechanixVAE | None = None,
    checkpoint: dict | None = None,
    score_mode: ScoreMode = "percentile",
) -> dict:
    """Calculate the Mechanical Drift Index for a pitcher.

    Two scoring modes are supported:

    - ``"percentile"`` (default, legacy): MDI is the percentile rank (0-100)
      of the most recent window's reconstruction error among all windows in
      the evaluation series.  Self-ranked — saturates at ~100 for every
      series, regardless of injury state.
    - ``"zscore"``: MDI is ``z = (error - healthy_mean) / max(healthy_std, eps)``
      against the pitcher's stored healthy baseline.  Higher z = more drift.
      Unbounded above; z=1/2/3 correspond to 1/2/3σ breaches.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        game_pk: If given, restrict to pitches from this game.
        model: Pre-loaded model (optional; loaded from disk if None).
        checkpoint: Pre-loaded checkpoint dict (optional).
        score_mode: Either "percentile" (legacy) or "zscore" (magnitude).

    Returns:
        Dictionary with ``mdi``, ``recon_errors``, ``n_windows``, and
        per-window detail.  When ``score_mode="zscore"``, ``mdi`` is the
        z-score of the latest window; thresholds z=1/2/3 are recommended
        bucket boundaries.
    """
    if model is None or checkpoint is None:
        model, checkpoint = _load_model(pitcher_id)

    means = checkpoint.get("means", {})

    # Fetch recent pitches
    df = _fetch_pitch_data(conn, pitcher_id=pitcher_id)

    if game_pk is not None:
        df = df[df["game_pk"] == game_pk].reset_index(drop=True)

    if df.empty or len(df) < WINDOW_SIZE:
        return {
            "pitcher_id": pitcher_id,
            "mdi": None,
            "n_windows": 0,
            "recon_errors": [],
            "message": "Insufficient pitches for MDI calculation.",
        }

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
        return {
            "pitcher_id": pitcher_id,
            "mdi": None,
            "n_windows": 0,
            "recon_errors": [],
            "message": "Not enough pitches for a full window.",
        }

    errors = _compute_reconstruction_errors(model, windows)
    latest_error = errors[-1]

    if score_mode == "zscore":
        healthy_mean = checkpoint.get("healthy_recon_mean")
        healthy_std = checkpoint.get("healthy_recon_std")
        if healthy_mean is None or healthy_std is None:
            # Checkpoint predates z-score era — fall back to errors themselves
            # as a self-baseline.  Callers with stored baseline stats should
            # populate the checkpoint to get the "real" z.
            healthy_mean = float(np.mean(errors))
            healthy_std = float(np.std(errors))
        z = compute_mdi_zscore(errors, healthy_mean, healthy_std)
        mdi_val = float(z[-1])
        return {
            "pitcher_id": pitcher_id,
            "mdi": round(mdi_val, 3),
            "score_mode": "zscore",
            "latest_recon_error": round(float(latest_error), 6),
            "mean_recon_error": round(float(errors.mean()), 6),
            "healthy_mean": round(float(healthy_mean), 6),
            "healthy_std": round(float(healthy_std), 6),
            "n_windows": len(windows),
            "recon_errors": [round(float(e), 6) for e in errors],
            "z_scores": [round(float(v), 3) for v in z],
        }

    # MDI = percentile rank of the most recent window's error among all windows
    mdi = float(np.mean(errors <= latest_error) * 100.0)
    mdi = float(np.clip(mdi, 0.0, 100.0))

    return {
        "pitcher_id": pitcher_id,
        "mdi": round(mdi, 1),
        "score_mode": "percentile",
        "latest_recon_error": round(float(latest_error), 6),
        "mean_recon_error": round(float(errors.mean()), 6),
        "n_windows": len(windows),
        "recon_errors": [round(float(e), 6) for e in errors],
    }


def calculate_drift_velocity(
    conn,
    pitcher_id: int,
    n_windows: int = 10,
    model: MechanixVAE | None = None,
    checkpoint: dict | None = None,
) -> dict:
    """Rate of change of MDI over recent windows.

    Computes the linear slope of reconstruction error across the last
    *n_windows* sliding windows.

    Returns:
        Dictionary with ``drift_velocity``, ``trend`` label, and detail.
    """
    if model is None or checkpoint is None:
        model, checkpoint = _load_model(pitcher_id)

    mdi_result = calculate_mdi(conn, pitcher_id, model=model, checkpoint=checkpoint)
    errors = mdi_result.get("recon_errors", [])

    if len(errors) < 2:
        return {
            "pitcher_id": pitcher_id,
            "drift_velocity": None,
            "trend": "insufficient_data",
        }

    recent = np.array(errors[-n_windows:])
    x = np.arange(len(recent), dtype=float)

    # Simple linear regression slope
    if len(recent) < 2:
        slope = 0.0
    else:
        x_mean = x.mean()
        y_mean = recent.mean()
        denom = ((x - x_mean) ** 2).sum()
        slope = float(((x - x_mean) * (recent - y_mean)).sum() / denom) if denom > 0 else 0.0

    if slope > 0.001:
        trend = "increasing"
    elif slope < -0.001:
        trend = "decreasing"
    else:
        trend = "stable"

    return {
        "pitcher_id": pitcher_id,
        "drift_velocity": round(slope, 6),
        "trend": trend,
        "n_windows_used": len(recent),
    }


def detect_changepoints(
    mdi_series: list[float] | np.ndarray,
    threshold: float = 5.0,
) -> list[int]:
    """Detect changepoints in an MDI time series using CUSUM.

    A changepoint is flagged when the cumulative sum of deviations from
    the running mean exceeds *threshold*.

    Args:
        mdi_series: Ordered sequence of MDI (or reconstruction error) values.
        threshold: Cumulative-sum threshold for flagging a changepoint.

    Returns:
        List of indices where changepoints are detected.
    """
    series = np.asarray(mdi_series, dtype=float)
    if len(series) < 3:
        return []

    mean_val = series.mean()
    cusum_pos = 0.0
    cusum_neg = 0.0
    changepoints: list[int] = []

    for i in range(len(series)):
        deviation = series[i] - mean_val
        cusum_pos = max(0.0, cusum_pos + deviation)
        cusum_neg = min(0.0, cusum_neg + deviation)

        if cusum_pos > threshold or cusum_neg < -threshold:
            changepoints.append(i)
            # Reset after detection
            cusum_pos = 0.0
            cusum_neg = 0.0

    return changepoints


def batch_calculate(
    conn,
    season: int | None = None,
    min_pitches: int = 200,
) -> pd.DataFrame:
    """Calculate current MDI for all qualifying pitchers.

    Trains a lightweight universal model on the fly (or loads a saved one),
    then evaluates all pitchers with enough data.

    Args:
        conn: Open DuckDB connection.
        season: Optional season filter.
        min_pitches: Minimum total pitches to qualify.

    Returns:
        DataFrame with pitcher_id, name, mdi, drift_velocity, trend.
    """
    season_filter = ""
    params: list = []
    if season is not None:
        season_filter = "AND EXTRACT(YEAR FROM game_date) = $1"
        params.append(season)

    query = f"""
        SELECT pitcher_id, COUNT(*) AS n_pitches
        FROM pitches
        WHERE pitch_type IS NOT NULL
          AND release_speed IS NOT NULL
          {season_filter}
        GROUP BY pitcher_id
        HAVING COUNT(*) >= {min_pitches}
    """
    pitcher_df = conn.execute(query, params).fetchdf()

    if pitcher_df.empty:
        return pd.DataFrame(
            columns=["pitcher_id", "name", "mdi", "drift_velocity", "trend"]
        )

    # Load or train universal model
    try:
        model, checkpoint = _load_model(None)
    except FileNotFoundError:
        logger.info("No universal model found; training one now...")
        train_mechanix_ae(conn, pitcher_id=None, epochs=10)
        model, checkpoint = _load_model(None)

    rows = []
    for _, row in pitcher_df.iterrows():
        pid = int(row["pitcher_id"])
        try:
            mdi_result = calculate_mdi(
                conn, pid, model=model, checkpoint=checkpoint,
            )
            dv_result = calculate_drift_velocity(
                conn, pid, model=model, checkpoint=checkpoint,
            )
            rows.append({
                "pitcher_id": pid,
                "mdi": mdi_result.get("mdi"),
                "drift_velocity": dv_result.get("drift_velocity"),
                "trend": dv_result.get("trend"),
                "n_windows": mdi_result.get("n_windows", 0),
            })
        except Exception as exc:
            logger.warning("Error for pitcher %d: %s", pid, exc)

    if not rows:
        return pd.DataFrame(
            columns=["pitcher_id", "name", "mdi", "drift_velocity", "trend"]
        )

    leaderboard = pd.DataFrame(rows)

    # Join player names
    try:
        names_df = conn.execute(
            "SELECT player_id AS pitcher_id, full_name AS name FROM players"
        ).fetchdf()
        leaderboard = leaderboard.merge(names_df, on="pitcher_id", how="left")
    except Exception:
        leaderboard["name"] = None

    front_cols = ["pitcher_id", "name", "mdi", "drift_velocity", "trend", "n_windows"]
    leaderboard = leaderboard[[c for c in front_cols if c in leaderboard.columns]]
    leaderboard = leaderboard.sort_values(
        "mdi", ascending=False, na_position="last",
    ).reset_index(drop=True)

    return leaderboard


# ── BaseAnalyticsModel wrapper ────────────────────────────────────────────


class MechanixAEModel(BaseAnalyticsModel):
    """MechanixAE model conforming to BaseAnalyticsModel interface."""

    @property
    def model_name(self) -> str:
        return "MechanixAE Mechanical Drift Detector"

    @property
    def version(self) -> str:
        return "1.0.0"

    def train(self, conn, **kwargs) -> dict:
        """Train the VAE.

        Kwargs:
            pitcher_id: Optional pitcher ID for per-pitcher training.
            epochs: Number of training epochs (default 20).
            batch_size: Mini-batch size (default 64).
        """
        pitcher_id = kwargs.get("pitcher_id")
        epochs = kwargs.get("epochs", 20)
        batch_size = kwargs.get("batch_size", 64)

        metrics = train_mechanix_ae(
            conn,
            pitcher_id=pitcher_id,
            epochs=epochs,
            batch_size=batch_size,
        )
        self.set_training_metadata(
            metrics=metrics,
            params={"pitcher_id": pitcher_id, "epochs": epochs, "batch_size": batch_size},
        )
        return metrics

    def predict(self, conn, **kwargs) -> dict:
        """Predict MDI for a pitcher.

        Kwargs:
            pitcher_id (int): Required.
            game_pk (int): Optional game filter.
        """
        pitcher_id = kwargs["pitcher_id"]
        game_pk = kwargs.get("game_pk")
        return calculate_mdi(conn, pitcher_id, game_pk=game_pk)

    def evaluate(self, conn, **kwargs) -> dict:
        """Evaluate by computing MDI across all qualifying pitchers.

        Kwargs:
            season (int): Optional season filter.
        """
        season = kwargs.get("season")
        leaderboard = batch_calculate(conn, season=season)
        n = len(leaderboard)
        if n == 0:
            return {"n_pitchers": 0, "coverage": 0.0}
        valid = leaderboard["mdi"].dropna()
        return {
            "n_pitchers": n,
            "mean_mdi": round(float(valid.mean()), 1) if len(valid) else None,
            "median_mdi": round(float(valid.median()), 1) if len(valid) else None,
            "std_mdi": round(float(valid.std()), 1) if len(valid) else None,
            "pct_above_80": round(float((valid > 80).mean() * 100), 1) if len(valid) else None,
        }
