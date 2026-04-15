"""
ChemNet -- Graph Neural Network for measuring lineup synergy and interaction effects.

Uses a Graph Attention Network (GAT) implemented in plain PyTorch (no
torch-geometric) to model batting-order protection effects.  Each lineup
is a 9-node graph where adjacent batting-order positions share an edge.
The model learns attention weights that capture pairwise synergy between
lineup neighbours and predicts total offensive output (game wOBA) for
the lineup.

Lineup Synergy Score = GNN prediction - non-graph baseline prediction.

Key public API
--------------
- ``train_chemnet(conn, ...)``       -- train the GNN + baseline
- ``calculate_synergy(conn, game_pk)``     -- synergy for one game
- ``get_protection_coefficients(conn, game_pk)``  -- attention weights
- ``batch_calculate(conn, season)``        -- per-team average synergy
- ``optimize_lineup_order(...)``           -- greedy reorder for max synergy
- ``ChemNetModel`` (BaseAnalyticsModel)   -- lifecycle wrapper
"""

from __future__ import annotations

import itertools
import logging
import math
from pathlib import Path
from typing import Any, Optional

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.analytics.base import BaseAnalyticsModel

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    """Return the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Constants ─────────────────────────────────────────────────────────────────

NUM_LINEUP_SLOTS = 9
NODE_FEATURE_DIM = 5  # rolling wOBA, K%, BB%, ISO, order_position
GAT_HIDDEN_DIM = 32
NUM_GAT_HEADS = 2
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# ── Graph construction helpers ────────────────────────────────────────────────


def _build_adjacency(n: int = NUM_LINEUP_SLOTS) -> torch.Tensor:
    """Build a band adjacency matrix for a lineup of *n* players.

    Connects slot i to slot i+1 (and vice-versa) to capture the
    protection effect between consecutive batting-order positions.

    Returns:
        (n, n) float tensor with 1s on adjacent entries, 0 elsewhere.
    """
    adj = torch.zeros(n, n)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def _get_lineup_for_game(
    conn: duckdb.DuckDBPyConnection,
    game_pk: int,
    team_side: str = "home",
) -> pd.DataFrame:
    """Extract the lineup for one team in a game from pitch data.

    Returns a DataFrame with columns ``batter_id``, ``order_position``
    (1-indexed), ``stand``, sorted by batting-order position.
    """
    if team_side == "home":
        side_filter = "inning_topbot = 'Bot'"
    else:
        side_filter = "inning_topbot = 'Top'"

    query = f"""
        SELECT
            batter_id,
            stand,
            MIN(at_bat_number) AS first_ab
        FROM pitches
        WHERE game_pk = $1
          AND {side_filter}
        GROUP BY batter_id, stand
        ORDER BY first_ab
    """
    df = conn.execute(query, [game_pk]).fetchdf()
    if df.empty:
        return df
    # Take first 9 unique batters (the starting lineup)
    df = df.head(NUM_LINEUP_SLOTS).reset_index(drop=True)
    df["order_position"] = df.index + 1
    return df


def _get_rolling_player_features(
    conn: duckdb.DuckDBPyConnection,
    batter_ids: list[int],
    before_date: str,
    window: int = 30,
) -> dict[int, np.ndarray]:
    """Compute rolling 30-game features for each batter.

    Features (per player):
        0. rolling wOBA  (SUM(woba_value) / SUM(woba_denom))
        1. K%            (strikeouts / PA)
        2. BB%           (walks / PA)
        3. ISO           (extra-base-hit power proxy from woba_value)
        4. (placeholder -- filled with order_position by caller)

    Returns:
        dict mapping batter_id -> np.array of shape (5,).
        Feature index 4 is set to 0.0 (caller fills with order position).
    """
    if not batter_ids:
        return {}

    id_list = ", ".join(str(int(b)) for b in batter_ids)

    query = f"""
        WITH game_stats AS (
            SELECT
                batter_id,
                game_pk,
                game_date,
                SUM(woba_value)  AS woba_val,
                SUM(woba_denom)  AS woba_den,
                SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END) AS so,
                SUM(CASE WHEN events = 'walk' THEN 1 ELSE 0 END) AS bb,
                SUM(CASE WHEN events IN ('double', 'triple', 'home_run') THEN 1 ELSE 0 END) AS xbh,
                COUNT(DISTINCT CASE WHEN events IS NOT NULL THEN at_bat_number END) AS pa
            FROM pitches
            WHERE batter_id IN ({id_list})
              AND game_date < '{before_date}'
            GROUP BY batter_id, game_pk, game_date
            HAVING SUM(woba_denom) > 0
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY batter_id ORDER BY game_date DESC
                   ) AS rn
            FROM game_stats
        )
        SELECT
            batter_id,
            SUM(woba_val)  AS total_woba_val,
            SUM(woba_den)  AS total_woba_den,
            SUM(so)        AS total_so,
            SUM(bb)        AS total_bb,
            SUM(xbh)       AS total_xbh,
            SUM(pa)        AS total_pa
        FROM ranked
        WHERE rn <= {window}
        GROUP BY batter_id
    """
    df = conn.execute(query).fetchdf()

    features: dict[int, np.ndarray] = {}
    for _, row in df.iterrows():
        bid = int(row["batter_id"])
        woba = (
            float(row["total_woba_val"] / row["total_woba_den"])
            if row["total_woba_den"] > 0
            else 0.320
        )
        pa = max(float(row["total_pa"]), 1.0)
        k_pct = float(row["total_so"]) / pa
        bb_pct = float(row["total_bb"]) / pa
        iso = float(row["total_xbh"]) / pa  # simple proxy
        features[bid] = np.array([woba, k_pct, bb_pct, iso, 0.0], dtype=np.float32)

    # Fill any missing players with league-average features
    for bid in batter_ids:
        if bid not in features:
            features[bid] = np.array([0.320, 0.22, 0.08, 0.04, 0.0], dtype=np.float32)

    return features


def _get_edge_features(lineup_df: pd.DataFrame) -> torch.Tensor:
    """Compute edge features for adjacent batting-order pairs.

    Currently: platoon differential encoded as -1 (LL), 0 (mixed), +1 (RR).

    Returns:
        Tensor of shape (n_edges,) with one value per undirected edge.
        For a 9-node lineup this is 8 edges.
    """
    n = len(lineup_df)
    stands = lineup_df["stand"].fillna("R").values
    platoon_map = {"L": -1.0, "R": 1.0}
    edge_feats = []
    for i in range(n - 1):
        s_i = platoon_map.get(stands[i], 0.0)
        s_j = platoon_map.get(stands[i + 1], 0.0)
        edge_feats.append(s_i * s_j)
    return torch.tensor(edge_feats, dtype=torch.float32)


def build_game_graph(
    conn: duckdb.DuckDBPyConnection,
    game_pk: int,
    team_side: str = "home",
) -> dict | None:
    """Construct a graph for one team's lineup in a specific game.

    Returns:
        Dictionary with keys ``node_features`` (9, 5), ``adj`` (9, 9),
        ``edge_features`` (8,), ``lineup`` (DataFrame), ``target``
        (scalar: total wOBA for this team in this game), or None on failure.
    """
    lineup = _get_lineup_for_game(conn, game_pk, team_side)
    if lineup.empty or len(lineup) < 2:
        return None

    n = len(lineup)
    game_date_row = conn.execute(
        "SELECT MIN(game_date) AS gd FROM pitches WHERE game_pk = $1", [game_pk]
    ).fetchone()
    if game_date_row is None:
        return None
    game_date = str(game_date_row[0])

    batter_ids = lineup["batter_id"].tolist()
    features = _get_rolling_player_features(conn, batter_ids, game_date)

    # Build node-feature matrix
    node_feats = np.zeros((n, NODE_FEATURE_DIM), dtype=np.float32)
    for i, bid in enumerate(batter_ids):
        f = features.get(bid, np.array([0.320, 0.22, 0.08, 0.04, 0.0], dtype=np.float32))
        f[4] = (i + 1) / NUM_LINEUP_SLOTS  # normalised order position
        node_feats[i] = f

    adj = _build_adjacency(n)

    # Target: total wOBA for this team in this game
    if team_side == "home":
        side_filter = "inning_topbot = 'Bot'"
    else:
        side_filter = "inning_topbot = 'Top'"

    target_row = conn.execute(
        f"""
        SELECT
            COALESCE(SUM(woba_value), 0) AS total_woba
        FROM pitches
        WHERE game_pk = $1 AND {side_filter}
          AND woba_denom IS NOT NULL AND woba_denom > 0
        """,
        [game_pk],
    ).fetchone()
    target = float(target_row[0]) if target_row else 0.0

    edge_features = _get_edge_features(lineup)

    return {
        "node_features": torch.tensor(node_feats),
        "adj": adj,
        "edge_features": edge_features,
        "lineup": lineup,
        "target": target,
        "game_pk": game_pk,
        "n_nodes": n,
    }


# ── GAT Layer (manual, no torch-geometric) ───────────────────────────────────


class GraphAttentionLayer(nn.Module):
    """Single-head Graph Attention layer.

    Implements the standard GAT formulation:
        alpha_ij = softmax_j(LeakyReLU(a^T [W h_i || W h_j]))
        h'_i = sum_j alpha_ij * W h_j

    Parameters:
        in_features:  Input feature dimensionality.
        out_features: Output feature dimensionality.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention vector: takes concatenation of [Wh_i || Wh_j]
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self, h: torch.Tensor, adj: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            h:   (N, in_features) node features.
            adj: (N, N) adjacency matrix (1 = edge, 0 = no edge).

        Returns:
            Tuple of (h', attention_weights) where h' is (N, out_features)
            and attention_weights is (N, N).
        """
        N = h.size(0)
        Wh = self.W(h)  # (N, out_features)

        # Build pairwise concatenation for every pair (i, j)
        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1)  # (N, N, F)
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1)  # (N, N, F)
        concat = torch.cat([Wh_i, Wh_j], dim=-1)  # (N, N, 2F)

        e = self.leaky_relu(self.a(concat).squeeze(-1))  # (N, N)

        # Mask non-edges with -inf so softmax ignores them.
        # Also add self-loops (a node attends to itself).
        mask = adj + torch.eye(N, device=adj.device)
        e = e.masked_fill(mask == 0, float("-inf"))

        alpha = F.softmax(e, dim=-1)  # (N, N)
        # Replace NaN from all-inf rows (isolated nodes) with 0
        alpha = alpha.nan_to_num(0.0)

        h_prime = torch.matmul(alpha, Wh)  # (N, out_features)
        return h_prime, alpha


class MultiHeadGAT(nn.Module):
    """Multi-head GAT: runs *n_heads* independent attention heads and
    concatenates (or averages) the results.

    Args:
        in_features:  Input feature dim.
        out_features: Output feature dim *per head*.
        n_heads:      Number of attention heads.
        concat:       If True, concatenate heads (output dim = out*heads).
                      If False, average heads (output dim = out).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_heads: int = NUM_GAT_HEADS,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [GraphAttentionLayer(in_features, out_features) for _ in range(n_heads)]
        )
        self.concat = concat
        self.n_heads = n_heads

    def forward(
        self, h: torch.Tensor, adj: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Returns:
            (h', [alpha_1, ..., alpha_k]) where h' has shape
            (N, out*heads) if concat else (N, out).
        """
        head_outs = []
        alphas = []
        for head in self.heads:
            out, alpha = head(h, adj)
            head_outs.append(out)
            alphas.append(alpha)

        if self.concat:
            h_prime = torch.cat(head_outs, dim=-1)
        else:
            h_prime = torch.stack(head_outs, dim=0).mean(dim=0)

        return h_prime, alphas


# ── GNN Model ────────────────────────────────────────────────────────────────


class ChemNetGNN(nn.Module):
    """Two-layer GAT that produces a single scalar prediction for a lineup.

    Architecture:
        GAT1 (5 -> 32, 2 heads, concat) -> ELU -> 64
        GAT2 (64 -> 32, 2 heads, average) -> ELU -> 32
        Global mean pool -> 32
        MLP: 32 -> 16 -> 1
    """

    def __init__(
        self,
        input_dim: int = NODE_FEATURE_DIM,
        hidden_dim: int = GAT_HIDDEN_DIM,
        n_heads: int = NUM_GAT_HEADS,
    ) -> None:
        super().__init__()
        self.gat1 = MultiHeadGAT(input_dim, hidden_dim, n_heads, concat=True)
        # After concat: hidden_dim * n_heads
        self.gat2 = MultiHeadGAT(hidden_dim * n_heads, hidden_dim, n_heads, concat=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(
        self, h: torch.Tensor, adj: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Args:
            h:   (N, input_dim) node features.
            adj: (N, N) adjacency.

        Returns:
            (prediction, attention_weights_from_layer2)
        """
        h1, _ = self.gat1(h, adj)
        h1 = F.elu(h1)
        h2, alphas = self.gat2(h1, adj)
        h2 = F.elu(h2)

        # Global mean pooling
        graph_emb = h2.mean(dim=0, keepdim=True)  # (1, hidden)
        out = self.mlp(graph_emb).squeeze()  # scalar
        return out, alphas


class BaselineMLP(nn.Module):
    """Non-graph baseline: flattens all node features and predicts runs.

    Architecture:
        Flatten 9*5=45 -> 32 -> 16 -> 1
    """

    def __init__(self, input_dim: int = NUM_LINEUP_SLOTS * NODE_FEATURE_DIM) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            h: (N, feature_dim) node features. Will be flattened.

        Returns:
            Scalar prediction.
        """
        flat = h.reshape(1, -1)
        return self.mlp(flat).squeeze()


# ── Training ──────────────────────────────────────────────────────────────────


def _collect_game_pks(
    conn: duckdb.DuckDBPyConnection,
    seasons: range,
) -> list[int]:
    """Return game_pks that have enough data to build lineup graphs."""
    season_list = ", ".join(str(s) for s in seasons)
    query = f"""
        SELECT game_pk
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) IN ({season_list})
        GROUP BY game_pk
        HAVING COUNT(DISTINCT batter_id) >= 9
        ORDER BY game_pk
    """
    df = conn.execute(query).fetchdf()
    return df["game_pk"].tolist()


def _pad_graph(graph: dict, target_n: int = NUM_LINEUP_SLOTS) -> dict:
    """Pad a graph with fewer than *target_n* nodes to the full size.

    Pads node features with zeros, and expands adjacency accordingly.
    """
    n = graph["n_nodes"]
    if n >= target_n:
        return graph

    # Pad node features
    nf = graph["node_features"]
    pad = torch.zeros(target_n - n, nf.size(1))
    graph["node_features"] = torch.cat([nf, pad], dim=0)

    # Expand adjacency
    adj = torch.zeros(target_n, target_n)
    adj[:n, :n] = graph["adj"]
    graph["adj"] = adj

    graph["n_nodes"] = target_n
    return graph


def train_chemnet(
    conn: duckdb.DuckDBPyConnection,
    seasons: range = range(2015, 2025),
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    val_seasons: range = range(2025, 2027),
    max_games: int = 2000,
) -> dict:
    """Train the ChemNet GNN and a baseline MLP.

    Builds game-level graphs for training seasons and fits both models.

    Args:
        conn:        Open DuckDB connection.
        seasons:     Training seasons.
        epochs:      Number of training epochs.
        batch_size:  Mini-batch size.
        lr:          Learning rate.
        val_seasons: Hold-out seasons for validation.
        max_games:   Cap on total training games (for speed).  Default
                     2000 (yields ~4000 lineup graphs for both sides).

    Returns:
        Dictionary of training metrics.
    """
    logger.info("Collecting training game PKs for seasons %s ...", list(seasons))
    game_pks = _collect_game_pks(conn, seasons)
    if len(game_pks) > max_games:
        rng = np.random.RandomState(42)
        game_pks = [int(x) for x in rng.choice(game_pks, size=max_games, replace=False)]
    logger.info("Building graphs for %d games ...", len(game_pks))

    graphs: list[dict] = []
    for gp in game_pks:
        for side in ("home", "away"):
            g = build_game_graph(conn, gp, side)
            if g is not None and g["n_nodes"] >= 2:
                g = _pad_graph(g)
                graphs.append(g)

    if not graphs:
        logger.warning("No valid graphs built. Aborting training.")
        return {"status": "no_data"}

    logger.info("Training on %d lineup graphs.", len(graphs))

    # Initialise models
    device = _get_device()
    gnn = ChemNetGNN().to(device)
    baseline = BaselineMLP().to(device)
    opt_gnn = torch.optim.Adam(gnn.parameters(), lr=lr)
    opt_base = torch.optim.Adam(baseline.parameters(), lr=lr)

    # Shuffle and train
    rng = np.random.RandomState(42)
    gnn_losses: list[float] = []
    base_losses: list[float] = []

    for epoch in range(epochs):
        indices = list(range(len(graphs)))
        rng.shuffle(indices)
        epoch_gnn_loss = 0.0
        epoch_base_loss = 0.0
        n_batches = 0

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            gnn_batch_loss = torch.tensor(0.0, device=device)
            base_batch_loss = torch.tensor(0.0, device=device)

            for idx in batch_idx:
                g = graphs[idx]
                nf = g["node_features"].to(device)
                adj = g["adj"].to(device)
                target = torch.tensor(g["target"], dtype=torch.float32, device=device)

                # GNN forward
                pred_gnn, _ = gnn(nf, adj)
                gnn_batch_loss = gnn_batch_loss + F.mse_loss(pred_gnn, target)

                # Baseline forward
                pred_base = baseline(nf)
                base_batch_loss = base_batch_loss + F.mse_loss(pred_base, target)

            gnn_batch_loss = gnn_batch_loss / len(batch_idx)
            base_batch_loss = base_batch_loss / len(batch_idx)

            opt_gnn.zero_grad()
            gnn_batch_loss.backward()
            opt_gnn.step()

            opt_base.zero_grad()
            base_batch_loss.backward()
            opt_base.step()

            epoch_gnn_loss += gnn_batch_loss.item()
            epoch_base_loss += base_batch_loss.item()
            n_batches += 1

        avg_gnn = epoch_gnn_loss / max(n_batches, 1)
        avg_base = epoch_base_loss / max(n_batches, 1)
        gnn_losses.append(avg_gnn)
        base_losses.append(avg_base)
        logger.info(
            "Epoch %d/%d  GNN loss=%.4f  Baseline loss=%.4f",
            epoch + 1, epochs, avg_gnn, avg_base,
        )

    # ── Validation ────────────────────────────────────────────────────────
    val_metrics: dict[str, Any] = {}
    val_pks = _collect_game_pks(conn, val_seasons)
    if val_pks:
        if len(val_pks) > 500:
            val_pks = [int(x) for x in rng.choice(val_pks, size=500, replace=False)]
        val_graphs = []
        for gp in val_pks:
            for side in ("home", "away"):
                g = build_game_graph(conn, gp, side)
                if g is not None and g["n_nodes"] >= 2:
                    g = _pad_graph(g)
                    val_graphs.append(g)

        if val_graphs:
            gnn.eval()
            baseline.eval()
            gnn_val_loss = 0.0
            base_val_loss = 0.0
            with torch.no_grad():
                for g in val_graphs:
                    nf = g["node_features"].to(device)
                    adj = g["adj"].to(device)
                    target = torch.tensor(g["target"], dtype=torch.float32, device=device)
                    pred_gnn, _ = gnn(nf, adj)
                    pred_base = baseline(nf)
                    gnn_val_loss += F.mse_loss(pred_gnn, target).item()
                    base_val_loss += F.mse_loss(pred_base, target).item()
            val_metrics["val_gnn_mse"] = round(gnn_val_loss / len(val_graphs), 4)
            val_metrics["val_baseline_mse"] = round(base_val_loss / len(val_graphs), 4)
            logger.info("Validation GNN MSE=%.4f  Baseline MSE=%.4f",
                        val_metrics["val_gnn_mse"], val_metrics["val_baseline_mse"])

    # Save models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version = "1"
    torch.save(
        {"gnn": gnn.state_dict(), "baseline": baseline.state_dict()},
        MODELS_DIR / f"chemnet_v{version}.pt",
    )
    logger.info("Models saved to %s", MODELS_DIR / f"chemnet_v{version}.pt")

    metrics = {
        "training_graphs": len(graphs),
        "epochs": epochs,
        "final_gnn_loss": round(gnn_losses[-1], 4) if gnn_losses else None,
        "final_baseline_loss": round(base_losses[-1], 4) if base_losses else None,
        **val_metrics,
    }
    return metrics


# ── Inference helpers ─────────────────────────────────────────────────────────


def _load_models(version: str = "1") -> tuple[ChemNetGNN, BaselineMLP]:
    """Load trained GNN and baseline from disk."""
    device = _get_device()
    path = MODELS_DIR / f"chemnet_v{version}.pt"
    if not path.exists():
        raise FileNotFoundError(f"ChemNet model not found at {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    gnn = ChemNetGNN()
    gnn.load_state_dict(state["gnn"])
    gnn.to(device)
    gnn.eval()
    baseline = BaselineMLP()
    baseline.load_state_dict(state["baseline"])
    baseline.to(device)
    baseline.eval()
    return gnn, baseline


def calculate_synergy(
    conn: duckdb.DuckDBPyConnection,
    game_pk: int,
    team_side: str = "home",
    version: str = "1",
) -> dict:
    """Calculate the Lineup Synergy Score for a specific game/team.

    Synergy = GNN prediction - baseline prediction.
    Positive synergy means the batting order itself is adding value
    beyond the sum of individual parts.

    Returns:
        Dict with ``synergy_score``, ``gnn_pred``, ``baseline_pred``,
        ``game_pk``, ``team_side``.
    """
    gnn, baseline = _load_models(version)

    g = build_game_graph(conn, game_pk, team_side)
    if g is None:
        return {
            "synergy_score": None,
            "gnn_pred": None,
            "baseline_pred": None,
            "game_pk": game_pk,
            "team_side": team_side,
        }

    g = _pad_graph(g)
    device = next(gnn.parameters()).device
    nf = g["node_features"].to(device)
    adj = g["adj"].to(device)

    with torch.no_grad():
        pred_gnn, _ = gnn(nf, adj)
        pred_base = baseline(nf)

    synergy = float(pred_gnn.item()) - float(pred_base.item())

    return {
        "synergy_score": round(synergy, 4),
        "gnn_pred": round(float(pred_gnn.item()), 4),
        "baseline_pred": round(float(pred_base.item()), 4),
        "game_pk": game_pk,
        "team_side": team_side,
    }


def get_protection_coefficients(
    conn: duckdb.DuckDBPyConnection,
    game_pk: int,
    team_side: str = "home",
    version: str = "1",
) -> dict:
    """Return attention weights between adjacent batting-order slots.

    These weights indicate how much the GNN considers the "protection
    effect" between consecutive hitters.

    Returns:
        Dict with ``pairs`` (list of dicts with batter pair info and weight),
        ``attention_matrix`` (list of lists), ``game_pk``.
    """
    gnn, _ = _load_models(version)

    g = build_game_graph(conn, game_pk, team_side)
    if g is None:
        return {"pairs": [], "attention_matrix": [], "game_pk": game_pk}

    g = _pad_graph(g)
    device = next(gnn.parameters()).device
    nf = g["node_features"].to(device)
    adj = g["adj"].to(device)
    lineup = g["lineup"]

    with torch.no_grad():
        _, alphas = gnn(nf, adj)

    # Average attention across heads
    avg_alpha = torch.stack(alphas, dim=0).mean(dim=0).cpu().numpy()

    pairs = []
    n = len(lineup)
    for i in range(n - 1):
        b1 = int(lineup.iloc[i]["batter_id"])
        b2 = int(lineup.iloc[i + 1]["batter_id"])
        weight_fwd = float(avg_alpha[i, i + 1])
        weight_bwd = float(avg_alpha[i + 1, i])
        pairs.append({
            "slot_from": i + 1,
            "slot_to": i + 2,
            "batter_from": b1,
            "batter_to": b2,
            "attention_fwd": round(weight_fwd, 4),
            "attention_bwd": round(weight_bwd, 4),
            "avg_attention": round((weight_fwd + weight_bwd) / 2, 4),
        })

    return {
        "pairs": pairs,
        "attention_matrix": avg_alpha.tolist(),
        "game_pk": game_pk,
    }


def batch_calculate(
    conn: duckdb.DuckDBPyConnection,
    season: int,
    version: str = "1",
    max_games: int = 500,
) -> pd.DataFrame:
    """Calculate average synergy score per team for a season.

    Returns:
        DataFrame with columns ``team``, ``avg_synergy``, ``games``,
        sorted by synergy descending.
    """
    gnn, baseline = _load_models(version)

    game_pks = _collect_game_pks(conn, range(season, season + 1))
    if not game_pks:
        return pd.DataFrame(columns=["team", "avg_synergy", "games"])

    rng = np.random.RandomState(42)
    if len(game_pks) > max_games:
        game_pks = [int(x) for x in rng.choice(game_pks, size=max_games, replace=False)]

    device = next(gnn.parameters()).device
    records: list[dict] = []
    for gp in game_pks:
        gp = int(gp)  # Ensure Python int for DuckDB
        # Determine teams
        team_row = conn.execute(
            "SELECT home_team, away_team FROM games WHERE game_pk = $1 LIMIT 1",
            [gp],
        ).fetchone()
        if team_row is None:
            continue
        for side, team in [("home", team_row[0]), ("away", team_row[1])]:
            g = build_game_graph(conn, gp, side)
            if g is None:
                continue
            g = _pad_graph(g)
            nf = g["node_features"].to(device)
            adj = g["adj"].to(device)
            with torch.no_grad():
                pred_gnn, _ = gnn(nf, adj)
                pred_base = baseline(nf)
            synergy = float(pred_gnn.item()) - float(pred_base.item())
            records.append({"team": team, "synergy": synergy})

    if not records:
        return pd.DataFrame(columns=["team", "avg_synergy", "games"])

    df = pd.DataFrame(records)
    result = (
        df.groupby("team")
        .agg(avg_synergy=("synergy", "mean"), games=("synergy", "count"))
        .reset_index()
        .sort_values("avg_synergy", ascending=False)
        .reset_index(drop=True)
    )
    result["avg_synergy"] = result["avg_synergy"].round(4)
    return result


def optimize_lineup_order(
    player_ids: list[int],
    player_features: dict[int, np.ndarray],
    version: str = "1",
) -> dict:
    """Find the batting-order permutation that maximises the GNN prediction.

    For <=7 players, tries all permutations.  For >7, uses a greedy
    swap heuristic.

    Args:
        player_ids:      List of batter IDs (up to 9).
        player_features: Mapping of batter_id -> feature array (shape 4 or 5).
        version:         ChemNet model version.

    Returns:
        Dict with ``optimal_order`` (list of batter_ids), ``gnn_score``,
        ``initial_score``, ``improvement``.
    """
    gnn, _ = _load_models(version)

    n = min(len(player_ids), NUM_LINEUP_SLOTS)
    players = player_ids[:n]

    device = next(gnn.parameters()).device

    def _score_order(order: list[int]) -> float:
        nf = np.zeros((NUM_LINEUP_SLOTS, NODE_FEATURE_DIM), dtype=np.float32)
        for i, pid in enumerate(order):
            f = player_features.get(pid, np.array([0.320, 0.22, 0.08, 0.04, 0.0], dtype=np.float32))
            if len(f) < NODE_FEATURE_DIM:
                f = np.concatenate([f, np.zeros(NODE_FEATURE_DIM - len(f))])
            f = f.copy()
            f[4] = (i + 1) / NUM_LINEUP_SLOTS
            nf[i] = f[:NODE_FEATURE_DIM]
        adj = _build_adjacency(NUM_LINEUP_SLOTS).to(device)
        with torch.no_grad():
            pred, _ = gnn(torch.tensor(nf, device=device), adj)
        return float(pred.item())

    initial_score = _score_order(players)

    if n <= 7:
        # Exhaustive search
        best_score = float("-inf")
        best_order = players[:]
        for perm in itertools.permutations(players):
            s = _score_order(list(perm))
            if s > best_score:
                best_score = s
                best_order = list(perm)
    else:
        # Greedy swap
        best_order = players[:]
        best_score = initial_score
        improved = True
        max_iter = 100
        iteration = 0
        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            for i in range(n):
                for j in range(i + 1, n):
                    candidate = best_order[:]
                    candidate[i], candidate[j] = candidate[j], candidate[i]
                    s = _score_order(candidate)
                    if s > best_score + 1e-6:
                        best_score = s
                        best_order = candidate
                        improved = True

    return {
        "optimal_order": best_order,
        "gnn_score": round(best_score, 4),
        "initial_score": round(initial_score, 4),
        "improvement": round(best_score - initial_score, 4),
    }


# ── Model class (BaseAnalyticsModel pattern) ─────────────────────────────────


class ChemNetModel(BaseAnalyticsModel):
    """ChemNet GNN model for lineup synergy — lifecycle wrapper.

    Wraps training, prediction, and evaluation into the standard
    BaseAnalyticsModel interface.
    """

    @property
    def model_name(self) -> str:
        return "chemnet"

    @property
    def version(self) -> str:
        return "1.0.0"

    def __init__(self) -> None:
        super().__init__()
        self._gnn: Optional[ChemNetGNN] = None
        self._baseline: Optional[BaselineMLP] = None
        self._leaderboard: Optional[pd.DataFrame] = None

    def train(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Train ChemNet GNN and baseline MLP.

        Keyword Args:
            seasons (range): Training season range (default 2015-2024).
            epochs (int):    Training epochs (default 10).
            batch_size (int): Mini-batch size (default 32).

        Returns:
            Training metrics dictionary.
        """
        seasons = kwargs.get("seasons", range(2015, 2025))
        epochs = kwargs.get("epochs", 10)
        batch_size = kwargs.get("batch_size", 32)
        max_games = kwargs.get("max_games", 5000)

        metrics = train_chemnet(
            conn,
            seasons=seasons,
            epochs=epochs,
            batch_size=batch_size,
            max_games=max_games,
        )

        self.set_training_metadata(metrics, params=kwargs)

        # Load trained models into instance
        try:
            self._gnn, self._baseline = _load_models()
        except FileNotFoundError:
            pass

        return metrics

    def predict(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Predict synergy for a game.

        Keyword Args:
            game_pk (int):       Game to analyse.
            team_side (str):     'home' or 'away'.

        Returns:
            Synergy result dictionary.
        """
        game_pk = kwargs.get("game_pk")
        if game_pk is None:
            raise ValueError("game_pk is required for prediction.")
        team_side = kwargs.get("team_side", "home")
        result = calculate_synergy(conn, game_pk, team_side)
        return self.validate_output(result)

    def evaluate(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Evaluate by computing team synergy leaderboard.

        Keyword Args:
            season (int): Season to evaluate.

        Returns:
            Evaluation metrics dictionary.
        """
        season = kwargs.get("season", 2024)
        df = batch_calculate(conn, season)
        self._leaderboard = df

        if df.empty:
            return {"status": "no_data", "teams": 0}

        return {
            "teams": len(df),
            "mean_synergy": round(float(df["avg_synergy"].mean()), 4),
            "max_synergy": round(float(df["avg_synergy"].max()), 4),
            "min_synergy": round(float(df["avg_synergy"].min()), 4),
            "top_team": str(df.iloc[0]["team"]),
        }
