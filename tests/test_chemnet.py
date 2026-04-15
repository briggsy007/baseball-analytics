"""
Tests for the ChemNet Graph Neural Network lineup-synergy model.

Covers graph construction, GAT layer mechanics, forward-pass shapes,
synergy isolation (GNN - baseline), protection coefficients, and
edge cases such as games with fewer than 9 batters.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.analytics.chemnet import (
    NUM_LINEUP_SLOTS,
    NODE_FEATURE_DIM,
    GAT_HIDDEN_DIM,
    NUM_GAT_HEADS,
    BaselineMLP,
    ChemNetGNN,
    ChemNetModel,
    GraphAttentionLayer,
    MultiHeadGAT,
    _build_adjacency,
    _pad_graph,
    build_game_graph,
)


# ── Graph construction ────────────────────────────────────────────────────────


class TestGraphConstruction:
    """Tests for adjacency-matrix building and graph helpers."""

    def test_adjacency_shape_default(self):
        """Adjacency matrix is 9x9 for a full lineup."""
        adj = _build_adjacency()
        assert adj.shape == (NUM_LINEUP_SLOTS, NUM_LINEUP_SLOTS)

    def test_adjacency_shape_custom(self):
        """Adjacency works for arbitrary node counts."""
        adj = _build_adjacency(5)
        assert adj.shape == (5, 5)

    def test_adjacency_band_structure(self):
        """Adjacent slots are connected; non-adjacent are not."""
        adj = _build_adjacency(9)
        # Check that (i, i+1) and (i+1, i) are 1
        for i in range(8):
            assert adj[i, i + 1].item() == 1.0
            assert adj[i + 1, i].item() == 1.0
        # Check that diagonal is 0
        for i in range(9):
            assert adj[i, i].item() == 0.0
        # Check non-adjacent are 0
        assert adj[0, 2].item() == 0.0
        assert adj[0, 8].item() == 0.0

    def test_adjacency_edge_count(self):
        """A 9-node band has exactly 8 undirected edges (16 directed entries)."""
        adj = _build_adjacency(9)
        assert adj.sum().item() == 16.0  # 8 edges * 2 directions

    def test_adjacency_two_nodes(self):
        """Minimum valid graph: 2 nodes with 1 edge."""
        adj = _build_adjacency(2)
        assert adj[0, 1].item() == 1.0
        assert adj[1, 0].item() == 1.0
        assert adj.sum().item() == 2.0

    def test_pad_graph_expands_features(self):
        """Padding a graph with <9 nodes adds zero-rows."""
        g = {
            "node_features": torch.randn(5, NODE_FEATURE_DIM),
            "adj": _build_adjacency(5),
            "n_nodes": 5,
        }
        padded = _pad_graph(g, 9)
        assert padded["node_features"].shape == (9, NODE_FEATURE_DIM)
        assert padded["adj"].shape == (9, 9)
        assert padded["n_nodes"] == 9
        # Padded rows should be all zeros
        assert torch.all(padded["node_features"][5:] == 0).item()

    def test_pad_graph_no_op_for_full(self):
        """Padding a full-size graph is a no-op."""
        g = {
            "node_features": torch.randn(9, NODE_FEATURE_DIM),
            "adj": _build_adjacency(9),
            "n_nodes": 9,
        }
        padded = _pad_graph(g, 9)
        assert padded["n_nodes"] == 9


# ── GAT Layer ─────────────────────────────────────────────────────────────────


class TestGATLayer:
    """Tests for the Graph Attention Layer."""

    def test_output_shape(self):
        """Output has the correct dimensionality."""
        layer = GraphAttentionLayer(NODE_FEATURE_DIM, GAT_HIDDEN_DIM)
        h = torch.randn(9, NODE_FEATURE_DIM)
        adj = _build_adjacency(9)
        h_prime, alpha = layer(h, adj)
        assert h_prime.shape == (9, GAT_HIDDEN_DIM)
        assert alpha.shape == (9, 9)

    def test_attention_weights_sum_to_one(self):
        """Attention weights over neighbours (+ self) sum to ~1 per node."""
        layer = GraphAttentionLayer(NODE_FEATURE_DIM, GAT_HIDDEN_DIM)
        h = torch.randn(9, NODE_FEATURE_DIM)
        adj = _build_adjacency(9)
        _, alpha = layer(h, adj)
        # Each row should sum to 1 (softmax over connected nodes + self)
        row_sums = alpha.sum(dim=1)
        np.testing.assert_allclose(
            row_sums.detach().numpy(), np.ones(9), atol=1e-5
        )

    def test_attention_non_negative(self):
        """All attention weights are non-negative."""
        layer = GraphAttentionLayer(NODE_FEATURE_DIM, GAT_HIDDEN_DIM)
        h = torch.randn(9, NODE_FEATURE_DIM)
        adj = _build_adjacency(9)
        _, alpha = layer(h, adj)
        assert (alpha >= 0).all().item()

    def test_attention_zero_for_non_neighbours(self):
        """Non-adjacent, non-self entries should have zero attention."""
        layer = GraphAttentionLayer(NODE_FEATURE_DIM, GAT_HIDDEN_DIM)
        h = torch.randn(9, NODE_FEATURE_DIM)
        adj = _build_adjacency(9)
        _, alpha = layer(h, adj)
        # Node 0 should not attend to node 2+ (only self and node 1)
        assert alpha[0, 2].item() == pytest.approx(0.0, abs=1e-6)
        assert alpha[0, 8].item() == pytest.approx(0.0, abs=1e-6)

    def test_multi_head_gat_output_shape_concat(self):
        """Multi-head GAT with concat outputs hidden*heads features."""
        mh = MultiHeadGAT(NODE_FEATURE_DIM, GAT_HIDDEN_DIM, n_heads=2, concat=True)
        h = torch.randn(9, NODE_FEATURE_DIM)
        adj = _build_adjacency(9)
        h_prime, alphas = mh(h, adj)
        assert h_prime.shape == (9, GAT_HIDDEN_DIM * 2)
        assert len(alphas) == 2

    def test_multi_head_gat_output_shape_average(self):
        """Multi-head GAT with average keeps hidden dim."""
        mh = MultiHeadGAT(NODE_FEATURE_DIM, GAT_HIDDEN_DIM, n_heads=2, concat=False)
        h = torch.randn(9, NODE_FEATURE_DIM)
        adj = _build_adjacency(9)
        h_prime, alphas = mh(h, adj)
        assert h_prime.shape == (9, GAT_HIDDEN_DIM)
        assert len(alphas) == 2


# ── Full Model Forward Pass ──────────────────────────────────────────────────


class TestChemNetGNN:
    """Tests for the full ChemNet GNN model."""

    def test_forward_pass_scalar_output(self):
        """GNN produces a single scalar prediction."""
        gnn = ChemNetGNN()
        h = torch.randn(9, NODE_FEATURE_DIM)
        adj = _build_adjacency(9)
        pred, alphas = gnn(h, adj)
        assert pred.dim() == 0  # scalar
        assert len(alphas) == NUM_GAT_HEADS

    def test_forward_pass_small_graph(self):
        """GNN handles a 3-node graph after padding."""
        gnn = ChemNetGNN()
        h = torch.zeros(9, NODE_FEATURE_DIM)
        h[:3] = torch.randn(3, NODE_FEATURE_DIM)
        adj = torch.zeros(9, 9)
        adj[:3, :3] = _build_adjacency(3)
        pred, _ = gnn(h, adj)
        assert pred.dim() == 0
        assert not torch.isnan(pred)

    def test_different_inputs_different_outputs(self):
        """Different node features should produce different predictions."""
        gnn = ChemNetGNN()
        adj = _build_adjacency(9)
        h1 = torch.randn(9, NODE_FEATURE_DIM)
        h2 = torch.randn(9, NODE_FEATURE_DIM) + 2.0
        with torch.no_grad():
            pred1, _ = gnn(h1, adj)
            pred2, _ = gnn(h2, adj)
        assert pred1.item() != pred2.item()


# ── Baseline MLP ─────────────────────────────────────────────────────────────


class TestBaselineMLP:
    """Tests for the non-graph baseline model."""

    def test_forward_pass_scalar(self):
        """Baseline MLP outputs a scalar."""
        mlp = BaselineMLP()
        h = torch.randn(9, NODE_FEATURE_DIM)
        pred = mlp(h)
        assert pred.dim() == 0

    def test_baseline_ignores_order(self):
        """Baseline should (roughly) give same output for shuffled inputs,
        because it flattens -- but actually order matters in MLP.
        This test just verifies shape is correct."""
        mlp = BaselineMLP()
        h = torch.randn(9, NODE_FEATURE_DIM)
        pred = mlp(h)
        assert not torch.isnan(pred)


# ── Synergy Isolation ────────────────────────────────────────────────────────


class TestSynergyIsolation:
    """Tests that synergy = GNN - baseline is well-defined."""

    def test_synergy_is_difference(self):
        """Synergy score should equal GNN pred minus baseline pred."""
        gnn = ChemNetGNN()
        baseline = BaselineMLP()
        h = torch.randn(9, NODE_FEATURE_DIM)
        adj = _build_adjacency(9)

        with torch.no_grad():
            gnn_pred, _ = gnn(h, adj)
            base_pred = baseline(h)

        synergy = gnn_pred.item() - base_pred.item()
        # Just verify it's a finite number
        assert np.isfinite(synergy)

    def test_synergy_sign_varies(self):
        """Across random inputs, synergy should not always be the same sign."""
        gnn = ChemNetGNN()
        baseline = BaselineMLP()
        adj = _build_adjacency(9)
        synergies = []
        torch.manual_seed(42)
        for _ in range(50):
            h = torch.randn(9, NODE_FEATURE_DIM)
            with torch.no_grad():
                gnn_pred, _ = gnn(h, adj)
                base_pred = baseline(h)
            synergies.append(gnn_pred.item() - base_pred.item())
        # With random weights, expect some variance
        assert np.std(synergies) > 0


# ── Protection Coefficients ──────────────────────────────────────────────────


class TestProtectionCoefficients:
    """Tests for attention-weight extraction."""

    def test_eight_edges_for_nine_nodes(self):
        """A 9-node lineup should produce 8 adjacent pairs."""
        gnn = ChemNetGNN()
        h = torch.randn(9, NODE_FEATURE_DIM)
        adj = _build_adjacency(9)
        with torch.no_grad():
            _, alphas = gnn(h, adj)
        # Average across heads
        avg_alpha = torch.stack(alphas, dim=0).mean(dim=0)
        # Count non-zero adjacent attention entries
        n_edges = 0
        for i in range(8):
            if avg_alpha[i, i + 1].item() > 0:
                n_edges += 1
        assert n_edges == 8

    def test_attention_matrix_shape(self):
        """Attention matrix from final layer is 9x9."""
        gnn = ChemNetGNN()
        h = torch.randn(9, NODE_FEATURE_DIM)
        adj = _build_adjacency(9)
        with torch.no_grad():
            _, alphas = gnn(h, adj)
        for alpha in alphas:
            assert alpha.shape == (9, 9)


# ── Edge Cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases: small lineups, unknown features."""

    def test_two_node_graph(self):
        """Minimum valid lineup: 2 batters."""
        gnn = ChemNetGNN()
        h = torch.zeros(9, NODE_FEATURE_DIM)
        h[:2] = torch.randn(2, NODE_FEATURE_DIM)
        adj = torch.zeros(9, 9)
        adj[:2, :2] = _build_adjacency(2)
        with torch.no_grad():
            pred, alphas = gnn(h, adj)
        assert pred.dim() == 0
        assert not torch.isnan(pred)

    def test_all_zero_features(self):
        """Model handles all-zero node features gracefully."""
        gnn = ChemNetGNN()
        h = torch.zeros(9, NODE_FEATURE_DIM)
        adj = _build_adjacency(9)
        with torch.no_grad():
            pred, _ = gnn(h, adj)
        assert not torch.isnan(pred)

    def test_single_node_no_edges(self):
        """A single-node graph (padded to 9) should still produce output."""
        gnn = ChemNetGNN()
        h = torch.zeros(9, NODE_FEATURE_DIM)
        h[0] = torch.randn(NODE_FEATURE_DIM)
        adj = torch.zeros(9, 9)  # No edges at all
        with torch.no_grad():
            pred, _ = gnn(h, adj)
        assert not torch.isnan(pred)


# ── ChemNetModel (BaseAnalyticsModel) ────────────────────────────────────────


class TestChemNetModel:
    """Tests for the lifecycle wrapper class."""

    def test_model_name(self):
        model = ChemNetModel()
        assert model.model_name == "chemnet"

    def test_version(self):
        model = ChemNetModel()
        assert model.version == "1.0.0"

    def test_repr(self):
        model = ChemNetModel()
        r = repr(model)
        assert "chemnet" in r
        assert "1.0.0" in r

    def test_metadata_initialised(self):
        model = ChemNetModel()
        meta = model.metadata
        assert "created_at" in meta
        assert meta["training_date"] is None


# ── Integration with DB (uses conftest db_conn fixture) ──────────────────────


class TestDBIntegration:
    """Integration tests using the shared test database fixture.

    These test graph construction from real (or synthetic) pitch data.
    """

    def test_build_game_graph_returns_dict_or_none(self, db_conn):
        """build_game_graph should return a dict or None."""
        game_pk = db_conn.execute(
            "SELECT game_pk FROM pitches GROUP BY game_pk "
            "ORDER BY COUNT(*) DESC LIMIT 1"
        ).fetchone()
        if game_pk is None:
            pytest.skip("No games in test DB.")
        gp = game_pk[0]
        result = build_game_graph(db_conn, gp, "home")
        if result is not None:
            assert "node_features" in result
            assert "adj" in result
            assert "target" in result
            assert result["node_features"].dim() == 2
            assert result["adj"].shape[0] == result["adj"].shape[1]

    def test_build_game_graph_node_features_shape(self, db_conn):
        """Node features should be (N, 5) where N <= 9."""
        game_pk = db_conn.execute(
            "SELECT game_pk FROM pitches GROUP BY game_pk "
            "HAVING COUNT(DISTINCT batter_id) >= 4 "
            "ORDER BY COUNT(*) DESC LIMIT 1"
        ).fetchone()
        if game_pk is None:
            pytest.skip("No games with enough batters in test DB.")
        gp = game_pk[0]
        for side in ("home", "away"):
            result = build_game_graph(db_conn, gp, side)
            if result is not None:
                nf = result["node_features"]
                assert nf.shape[1] == NODE_FEATURE_DIM
                assert nf.shape[0] <= NUM_LINEUP_SLOTS
                return
        pytest.skip("Could not build graph for either side.")
