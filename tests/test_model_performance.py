"""
Performance regression tests for trained analytics models.

These tests load production model artifacts and evaluate them against
baseline metric thresholds.  If a model's performance drops more than
``REGRESSION_TOLERANCE`` (default 5 %) below a stored baseline, the test
fails -- catching silent regressions introduced by code changes, data
shifts, or dependency upgrades.

Models with trained artifacts (Stuff+, MechanixAE, PitchGPT, ChemNet)
are evaluated via their ``evaluate()`` method or by direct inference on
test data.  Compute-only models (CausalWAR, PSET) are validated for
output value ranges and determinism.

All tests are marked ``@pytest.mark.slow`` because they load real model
files and/or run non-trivial computations.

Run with:
    pytest -m slow tests/test_model_performance.py -v
Skip slow tests in CI with:
    pytest -m "not slow"
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _PROJECT_ROOT / "models"
_BASELINES_PATH = Path(__file__).resolve().parent / "baselines.json"

# ── Load baselines ───────────────────────────────────────────────────────────

def _load_baselines() -> dict:
    """Load baseline metric thresholds from baselines.json."""
    if not _BASELINES_PATH.exists():
        pytest.skip("baselines.json not found; cannot run regression tests.")
    with open(_BASELINES_PATH) as f:
        return json.load(f)


BASELINES = _load_baselines()
REGRESSION_TOLERANCE = BASELINES.get("regression_tolerance", 0.05)


# ── Fixtures ─────────────────────────────────────────────────────────────────
# Model artifact fixtures (stuff_model_artifact, mechanix_ae_artifact,
# pitchgpt_artifact, chemnet_artifact) and db_conn are provided by
# tests/conftest.py at session scope.


@pytest.fixture(scope="module")
def baselines() -> dict:
    """Return the loaded baselines dictionary."""
    return BASELINES


# ═════════════════════════════════════════════════════════════════════════════
# Stuff+ Performance Tests
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestStuffPlusPerformance:
    """Validate Stuff+ model metrics against baselines."""

    def test_artifact_loads_successfully(self, stuff_model_artifact):
        """Model artifact should contain expected keys."""
        art = stuff_model_artifact
        assert "model" in art, "Artifact missing 'model' key"
        assert "label_encoder" in art, "Artifact missing 'label_encoder'"
        assert "league_mean" in art, "Artifact missing 'league_mean'"
        assert "league_std" in art, "Artifact missing 'league_std'"
        assert "feature_cols" in art, "Artifact missing 'feature_cols'"

    def test_league_std_positive(self, stuff_model_artifact):
        """League standard deviation must be positive (non-degenerate model)."""
        assert stuff_model_artifact["league_std"] > 0

    def test_r2_above_baseline(self, stuff_model_artifact, db_conn, baselines):
        """Retrain on test data and verify R2 meets the baseline threshold.

        This trains a fresh model on the test database and checks the
        test-split R2 against the stored baseline.  The threshold is
        intentionally lenient because the test DB has limited data.
        """
        from src.analytics.stuff_model import train_stuff_model

        try:
            metrics = train_stuff_model(db_conn)
        except ValueError as exc:
            pytest.skip(f"Insufficient data for Stuff+ training: {exc}")

        r2_threshold = baselines["stuff_plus"]["r2_test_min"]
        r2_test = metrics["r2_test"]
        assert r2_test > r2_threshold, (
            f"Stuff+ R2 test ({r2_test:.4f}) below baseline ({r2_threshold})"
        )

    def test_rmse_below_threshold(self, stuff_model_artifact, db_conn, baselines):
        """RMSE on the test split should not exceed the baseline threshold."""
        from src.analytics.stuff_model import train_stuff_model

        try:
            metrics = train_stuff_model(db_conn)
        except ValueError:
            pytest.skip("Insufficient data for Stuff+ training")

        rmse_threshold = baselines["stuff_plus"]["rmse_test_max"]
        rmse_test = metrics["rmse_test"]
        assert rmse_test < rmse_threshold, (
            f"Stuff+ RMSE test ({rmse_test:.4f}) exceeds baseline ({rmse_threshold})"
        )

    def test_stuff_plus_score_range(self, stuff_model_artifact):
        """Stuff+ scores on random input should be in a sane range (50-150)."""
        model = stuff_model_artifact["model"]
        league_mean = stuff_model_artifact["league_mean"]
        league_std = stuff_model_artifact["league_std"]

        rng = np.random.RandomState(42)
        n_features = len(stuff_model_artifact["feature_cols"])
        X_fake = rng.normal(0, 1, size=(100, n_features))
        raw_preds = model.predict(X_fake)

        stuff_scores = 100.0 + (raw_preds - league_mean) / league_std * 10.0

        # Scores should be finite and within a reasonable range
        assert np.all(np.isfinite(stuff_scores)), "Non-finite Stuff+ scores"
        assert stuff_scores.min() > 0, f"Stuff+ min ({stuff_scores.min():.1f}) too low"
        assert stuff_scores.max() < 250, f"Stuff+ max ({stuff_scores.max():.1f}) too high"

    def test_deterministic_predictions(self, stuff_model_artifact):
        """Same input should yield identical output (no randomness in predict)."""
        model = stuff_model_artifact["model"]
        rng = np.random.RandomState(99)
        n_features = len(stuff_model_artifact["feature_cols"])
        X = rng.normal(0, 1, size=(20, n_features))

        preds_1 = model.predict(X)
        preds_2 = model.predict(X)
        np.testing.assert_array_equal(preds_1, preds_2)


# ═════════════════════════════════════════════════════════════════════════════
# MechanixAE Performance Tests
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestMechanixAEPerformance:
    """Validate MechanixAE VAE metrics against baselines."""

    def test_artifact_loads_successfully(self, mechanix_ae_artifact):
        """Checkpoint should contain expected keys."""
        ckpt = mechanix_ae_artifact
        assert "model_state_dict" in ckpt, "Missing model_state_dict"

    def test_reconstruction_loss_within_threshold(
        self, mechanix_ae_artifact, baselines
    ):
        """VAE reconstruction on random noise should stay below threshold.

        This is a sanity check: the model should reconstruct near-zero
        deviation data (centred features) without blowing up.
        """
        import torch
        from src.analytics.mechanix_ae import MechanixVAE, N_FEATURES, WINDOW_SIZE

        model = MechanixVAE()
        model.load_state_dict(mechanix_ae_artifact["model_state_dict"])
        model.eval()

        # Generate centred input (near the training distribution mean)
        rng = np.random.RandomState(42)
        windows = rng.normal(0, 0.5, size=(50, WINDOW_SIZE, N_FEATURES)).astype(
            np.float32
        )
        tensor = torch.tensor(windows).permute(0, 2, 1)

        with torch.no_grad():
            recon, mu, logvar = model(tensor)
            recon_loss = torch.nn.functional.mse_loss(recon, tensor).item()

        threshold = baselines["mechanix_ae"]["final_recon_loss_max"]
        assert recon_loss < threshold, (
            f"MechanixAE recon loss ({recon_loss:.4f}) exceeds baseline ({threshold})"
        )

    def test_latent_space_reasonable_dimension(self, mechanix_ae_artifact):
        """Latent vectors should have the configured dimensionality."""
        import torch
        from src.analytics.mechanix_ae import (
            MechanixVAE,
            N_FEATURES,
            WINDOW_SIZE,
            LATENT_DIM,
        )

        model = MechanixVAE()
        model.load_state_dict(mechanix_ae_artifact["model_state_dict"])
        model.eval()

        rng = np.random.RandomState(42)
        windows = rng.normal(0, 0.5, size=(10, WINDOW_SIZE, N_FEATURES)).astype(
            np.float32
        )
        tensor = torch.tensor(windows).permute(0, 2, 1)

        with torch.no_grad():
            z = model.get_latent(tensor)

        assert z.shape == (10, LATENT_DIM)
        assert torch.all(torch.isfinite(z)), "Non-finite latent vectors"

    def test_mdi_output_range(self, mechanix_ae_artifact, db_conn):
        """MDI values should be in [0, 100] when computed on test data."""
        from src.analytics.mechanix_ae import (
            MechanixVAE,
            calculate_mdi,
        )

        model = MechanixVAE()
        model.load_state_dict(mechanix_ae_artifact["model_state_dict"])
        model.eval()
        checkpoint = mechanix_ae_artifact

        # Find a pitcher with enough data
        row = db_conn.execute(
            "SELECT pitcher_id FROM pitches "
            "GROUP BY pitcher_id "
            "HAVING COUNT(*) >= 25 "
            "ORDER BY COUNT(*) DESC LIMIT 1"
        ).fetchone()
        if row is None:
            pytest.skip("No pitcher with >= 25 pitches in test DB")

        pitcher_id = row[0]
        result = calculate_mdi(db_conn, pitcher_id, model=model, checkpoint=checkpoint)

        if result["mdi"] is not None:
            assert 0.0 <= result["mdi"] <= 100.0, (
                f"MDI ({result['mdi']}) out of [0, 100] range"
            )

    def test_deterministic_reconstruction(self, mechanix_ae_artifact):
        """Deterministic encoding path (get_latent) should be repeatable."""
        import torch
        from src.analytics.mechanix_ae import MechanixVAE, N_FEATURES, WINDOW_SIZE

        model = MechanixVAE()
        model.load_state_dict(mechanix_ae_artifact["model_state_dict"])
        model.eval()

        rng = np.random.RandomState(7)
        x = torch.tensor(
            rng.normal(0, 0.5, (5, WINDOW_SIZE, N_FEATURES)).astype(np.float32)
        ).permute(0, 2, 1)

        with torch.no_grad():
            z1 = model.get_latent(x)
            z2 = model.get_latent(x)

        torch.testing.assert_close(z1, z2)


# ═════════════════════════════════════════════════════════════════════════════
# PitchGPT Performance Tests
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestPitchGPTPerformance:
    """Validate PitchGPT transformer metrics against baselines."""

    def test_artifact_loads_successfully(self, pitchgpt_artifact):
        """Checkpoint should contain model_state_dict and config."""
        ckpt = pitchgpt_artifact
        assert "model_state_dict" in ckpt
        assert "config" in ckpt

    def test_config_dimensions_consistent(self, pitchgpt_artifact):
        """Config dimensions should be internally consistent."""
        cfg = pitchgpt_artifact["config"]
        assert cfg["d_model"] % cfg["nhead"] == 0, (
            f"d_model ({cfg['d_model']}) not divisible by nhead ({cfg['nhead']})"
        )
        assert cfg["max_seq_len"] > 0
        assert cfg["num_layers"] > 0

    def test_forward_pass_produces_logits(self, pitchgpt_artifact):
        """Forward pass should produce logits of the correct vocab size."""
        import torch
        from src.analytics.pitchgpt import (
            PitchGPTModel, TOTAL_VOCAB, VOCAB_SIZE, CONTEXT_DIM,
        )

        cfg = pitchgpt_artifact["config"]
        state_dict = pitchgpt_artifact["model_state_dict"]

        # The model architecture uses TOTAL_VOCAB (VOCAB_SIZE + 2 special
        # tokens) for the embedding, but output_head maps to VOCAB_SIZE.
        # Use the config's vocab_size for the constructor (it controls the
        # embedding size), and infer the output dim from the weights.
        model = PitchGPTModel(
            vocab_size=cfg.get("vocab_size", TOTAL_VOCAB),
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            max_seq_len=cfg["max_seq_len"],
        )
        model.load_state_dict(state_dict)
        model.eval()

        output_vocab = state_dict["output_head.weight"].shape[0]

        batch_size = 4
        seq_len = 10
        tokens = torch.randint(0, output_vocab, (batch_size, seq_len))
        ctx = torch.zeros(batch_size, seq_len, CONTEXT_DIM)

        with torch.no_grad():
            logits = model(tokens, ctx)

        assert logits.shape == (batch_size, seq_len, output_vocab), (
            f"Logits shape {logits.shape} != expected "
            f"({batch_size}, {seq_len}, {output_vocab})"
        )
        assert torch.all(torch.isfinite(logits)), "Non-finite logits"

    def test_evaluate_pps_within_range(self, pitchgpt_artifact, db_conn, baselines):
        """PPS (Pitch Predictability Score) should be within baseline range.

        Calls ``batch_calculate`` directly with ``model_version="2"`` so the
        35-dim v2 checkpoint is loaded (matches current ``CONTEXT_DIM``).
        ``PitchGPT.evaluate`` internally hardcodes v1 via its default and
        does not forward a ``model_version`` kwarg.
        """
        from src.analytics.pitchgpt import batch_calculate

        try:
            df = batch_calculate(db_conn, model_version="2")
        except FileNotFoundError:
            pytest.skip("PitchGPT v2 model not loadable for evaluate()")

        if df.empty:
            pytest.skip("No qualifying pitchers in test DB for PitchGPT")

        mean_pps = round(float(df["pps"].mean()), 4)
        pps_max = baselines["pitchgpt"]["mean_pps_max"]
        assert mean_pps <= pps_max, (
            f"PitchGPT mean PPS ({mean_pps:.4f}) exceeds baseline max ({pps_max})"
        )

    def test_perplexity_within_range(self, pitchgpt_artifact, db_conn, baselines):
        """Perplexity derived from PPS should be within a sane range.

        Uses ``batch_calculate`` directly with ``model_version="2"`` (see
        ``test_evaluate_pps_within_range`` for rationale).
        """
        from src.analytics.pitchgpt import batch_calculate

        try:
            df = batch_calculate(db_conn, model_version="2")
        except FileNotFoundError:
            pytest.skip("PitchGPT v2 model not loadable for evaluate()")

        if df.empty:
            pytest.skip("No qualifying pitchers in test DB for PitchGPT")

        mean_pps = round(float(df["pps"].mean()), 4)
        if mean_pps > 0:
            perplexity = math.exp(min(mean_pps, 20))
            ppl_max = baselines["pitchgpt"]["perplexity_max"]
            assert perplexity <= ppl_max, (
                f"PitchGPT perplexity ({perplexity:.2f}) exceeds baseline max ({ppl_max})"
            )

    def test_deterministic_forward_pass(self, pitchgpt_artifact):
        """Same input tokens should yield identical logits."""
        import torch
        from src.analytics.pitchgpt import PitchGPTModel, TOTAL_VOCAB, CONTEXT_DIM

        cfg = pitchgpt_artifact["config"]
        state_dict = pitchgpt_artifact["model_state_dict"]
        output_vocab = state_dict["output_head.weight"].shape[0]

        model = PitchGPTModel(
            vocab_size=cfg.get("vocab_size", TOTAL_VOCAB),
            d_model=cfg["d_model"],
            nhead=cfg["nhead"],
            num_layers=cfg["num_layers"],
            max_seq_len=cfg["max_seq_len"],
        )
        model.load_state_dict(state_dict)
        model.eval()

        tokens = torch.randint(0, output_vocab, (2, 8))
        ctx = torch.zeros(2, 8, CONTEXT_DIM)

        with torch.no_grad():
            out1 = model(tokens, ctx)
            out2 = model(tokens, ctx)

        torch.testing.assert_close(out1, out2)


# ═════════════════════════════════════════════════════════════════════════════
# ChemNet Performance Tests
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestChemNetPerformance:
    """Validate ChemNet GNN metrics against baselines."""

    def test_artifact_loads_successfully(self, chemnet_artifact):
        """Checkpoint should contain both GNN and baseline state dicts."""
        assert "gnn" in chemnet_artifact, "Missing 'gnn' state dict"
        assert "baseline" in chemnet_artifact, "Missing 'baseline' state dict"

    def test_gnn_produces_scalar_prediction(self, chemnet_artifact):
        """Loaded GNN should produce a scalar output for a 9-node graph."""
        import torch
        from src.analytics.chemnet import (
            ChemNetGNN,
            NODE_FEATURE_DIM,
            _build_adjacency,
        )

        gnn = ChemNetGNN()
        gnn.load_state_dict(chemnet_artifact["gnn"])
        gnn.eval()

        h = torch.randn(9, NODE_FEATURE_DIM)
        adj = _build_adjacency(9)
        with torch.no_grad():
            pred, alphas = gnn(h, adj)

        assert pred.dim() == 0, "GNN should output a scalar"
        assert torch.isfinite(pred), "GNN prediction should be finite"

    def test_synergy_within_baseline_range(self, chemnet_artifact, baselines):
        """Synergy (GNN - baseline) on random input should be bounded."""
        import torch
        from src.analytics.chemnet import (
            ChemNetGNN,
            BaselineMLP,
            NODE_FEATURE_DIM,
            _build_adjacency,
        )

        gnn = ChemNetGNN()
        gnn.load_state_dict(chemnet_artifact["gnn"])
        gnn.eval()

        baseline = BaselineMLP()
        baseline.load_state_dict(chemnet_artifact["baseline"])
        baseline.eval()

        adj = _build_adjacency(9)
        synergies = []
        torch.manual_seed(42)

        for _ in range(100):
            h = torch.randn(9, NODE_FEATURE_DIM)
            with torch.no_grad():
                gnn_pred, _ = gnn(h, adj)
                base_pred = baseline(h)
            synergies.append(gnn_pred.item() - base_pred.item())

        synergies = np.array(synergies)
        syn_min = baselines["chemnet"]["mean_synergy_min"]
        syn_max = baselines["chemnet"]["mean_synergy_max"]
        mean_syn = float(synergies.mean())

        assert syn_min <= mean_syn <= syn_max, (
            f"ChemNet mean synergy ({mean_syn:.4f}) outside "
            f"[{syn_min}, {syn_max}] baseline range"
        )

    def test_evaluate_produces_valid_output(self, chemnet_artifact, db_conn, baselines):
        """ChemNetModel.evaluate() should return valid team-level metrics."""
        from src.analytics.chemnet import ChemNetModel

        model = ChemNetModel()
        try:
            result = model.evaluate(db_conn, season=2025)
        except FileNotFoundError:
            pytest.skip("ChemNet model not loadable for evaluate()")

        if result.get("status") == "no_data" or result.get("teams", 0) == 0:
            pytest.skip("No game data for ChemNet evaluation in test DB")

        assert "mean_synergy" in result
        assert "teams" in result
        assert result["teams"] > 0
        assert np.isfinite(result["mean_synergy"])

    def test_deterministic_gnn_prediction(self, chemnet_artifact):
        """Same input should yield identical GNN output."""
        import torch
        from src.analytics.chemnet import (
            ChemNetGNN,
            NODE_FEATURE_DIM,
            _build_adjacency,
        )

        gnn = ChemNetGNN()
        gnn.load_state_dict(chemnet_artifact["gnn"])
        gnn.eval()

        h = torch.randn(9, NODE_FEATURE_DIM)
        adj = _build_adjacency(9)

        with torch.no_grad():
            pred1, _ = gnn(h, adj)
            pred2, _ = gnn(h, adj)

        torch.testing.assert_close(pred1, pred2)


# ═════════════════════════════════════════════════════════════════════════════
# CausalWAR Compute-Only Tests
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestCausalWAROutputValidation:
    """Validate CausalWAR outputs are within expected value ranges."""

    @pytest.fixture(scope="class")
    def trained_causal_war(self):
        """Train a CausalWAR model on synthetic data for testing."""
        import duckdb
        from src.db.schema import create_tables
        from src.analytics.causal_war import CausalWARConfig, CausalWARModel

        rng = np.random.RandomState(42)
        n = 600
        n_players = 8

        conn = duckdb.connect(":memory:")
        create_tables(conn)

        player_ids = rng.choice(range(100000, 100000 + n_players), size=n)
        pitcher_ids = rng.choice(range(200000, 200010), size=n)
        rows = []
        game_pk_base = 600000

        for i in range(n):
            event = rng.choice(
                [None, "single", "double", "home_run", "strikeout", "walk", "field_out"]
            )
            woba_val = round(float(rng.uniform(0, 2.0)), 3) if event else None
            woba_den = 1.0 if event else 0.0
            month = rng.randint(4, 10)
            day = rng.randint(1, 28)
            rows.append({
                "game_pk": game_pk_base + i // 10,
                "game_date": f"2025-{month:02d}-{day:02d}",
                "pitcher_id": int(pitcher_ids[i]),
                "batter_id": int(player_ids[i]),
                "pitch_type": rng.choice(["FF", "SL", "CH", "CU"]),
                "pitch_name": "FF",
                "release_speed": round(float(rng.normal(93, 2)), 1),
                "release_spin_rate": round(float(rng.normal(2300, 200)), 0),
                "spin_axis": round(float(rng.uniform(0, 360)), 1),
                "pfx_x": round(float(rng.normal(0, 5)), 1),
                "pfx_z": round(float(rng.normal(8, 4)), 1),
                "plate_x": round(float(rng.normal(0, 0.6)), 2),
                "plate_z": round(float(rng.normal(2.5, 0.6)), 2),
                "release_extension": round(float(rng.normal(6.2, 0.4)), 1),
                "release_pos_x": round(float(rng.normal(-1.5, 0.5)), 2),
                "release_pos_y": round(float(rng.normal(55, 0.5)), 2),
                "release_pos_z": round(float(rng.normal(5.8, 0.4)), 2),
                "launch_speed": round(float(rng.normal(88, 12)), 1) if event else None,
                "launch_angle": round(float(rng.normal(12, 20)), 1) if event else None,
                "hit_distance": round(float(rng.uniform(50, 420)), 0) if event else None,
                "hc_x": None,
                "hc_y": None,
                "bb_type": None,
                "estimated_ba": None,
                "estimated_woba": None,
                "delta_home_win_exp": round(float(rng.normal(0, 0.03)), 4),
                "delta_run_exp": round(float(rng.normal(0, 0.1)), 4),
                "inning": int(rng.randint(1, 10)),
                "inning_topbot": rng.choice(["Top", "Bot"]),
                "outs_when_up": int(rng.randint(0, 3)),
                "balls": int(rng.randint(0, 4)),
                "strikes": int(rng.randint(0, 3)),
                "on_1b": int(rng.choice([0, 1])),
                "on_2b": int(rng.choice([0, 1])),
                "on_3b": int(rng.choice([0, 1])),
                "stand": rng.choice(["L", "R"]),
                "p_throws": rng.choice(["L", "R"]),
                "at_bat_number": int(i // 5 + 1),
                "pitch_number": int(i % 5 + 1),
                "description": rng.choice(
                    ["called_strike", "ball", "hit_into_play", "foul"]
                ),
                "events": event,
                "type": "X" if event else "S",
                "home_team": "PHI",
                "away_team": rng.choice(["NYM", "ATL", "WSH"]),
                "woba_value": woba_val,
                "woba_denom": woba_den,
                "babip_value": None,
                "iso_value": None,
                "zone": int(rng.randint(1, 15)),
                "effective_speed": round(float(rng.normal(92, 2)), 1),
                "if_fielding_alignment": rng.choice(["Standard", "Infield shift"]),
                "of_fielding_alignment": rng.choice(["Standard", "Strategic"]),
                "fielder_2": int(rng.randint(100_000, 700_000)),
            })

        df = pd.DataFrame(rows)
        conn.execute("INSERT INTO pitches SELECT * FROM df")

        game_pks = df["game_pk"].unique()
        for gpk in game_pks:
            venue = rng.choice(
                ["Citizens Bank Park", "Yankee Stadium", "Fenway Park"]
            )
            conn.execute(
                "INSERT INTO games (game_pk, game_date, home_team, away_team, venue) "
                "VALUES ($1, '2025-06-15', 'PHI', 'NYM', $2)",
                [int(gpk), venue],
            )

        model = CausalWARModel(
            CausalWARConfig(
                n_estimators=10,
                n_bootstrap=5,
                n_splits=2,
                pa_min_qualifying=5,
            )
        )
        model.train(conn, season=2025)
        yield model, conn
        conn.close()

    def test_war_values_within_expected_range(self, trained_causal_war, baselines):
        """All individual CausalWAR values should be within [-5, 12]."""
        model, conn = trained_causal_war
        effects = model._player_effects

        if not effects:
            pytest.skip("No player effects estimated")

        war_min = baselines["causal_war"]["war_min"]
        war_max = baselines["causal_war"]["war_max"]

        for pid, effect in effects.items():
            war = effect.get("causal_war")
            if war is not None:
                assert war_min <= war <= war_max, (
                    f"Player {pid} CausalWAR ({war:.4f}) outside "
                    f"[{war_min}, {war_max}] range"
                )

    def test_mean_war_reasonable(self, trained_causal_war, baselines):
        """Mean CausalWAR across all players should be near zero."""
        model, conn = trained_causal_war
        result = model.evaluate(conn)

        if result.get("n_players", 0) == 0:
            pytest.skip("No players in evaluation")

        mean_war = result["mean_causal_war"]
        war_mean_min = baselines["causal_war"]["mean_war_min"]
        war_mean_max = baselines["causal_war"]["mean_war_max"]
        assert war_mean_min <= mean_war <= war_mean_max, (
            f"Mean CausalWAR ({mean_war:.4f}) outside "
            f"[{war_mean_min}, {war_mean_max}] range"
        )

    def test_evaluate_returns_expected_keys(self, trained_causal_war):
        """Evaluate output should contain standard diagnostic keys."""
        model, conn = trained_causal_war
        result = model.evaluate(conn)

        expected_keys = [
            "n_players",
            "mean_causal_war",
            "std_causal_war",
            "median_causal_war",
            "min_causal_war",
            "max_causal_war",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_deterministic_evaluation(self, trained_causal_war):
        """Same model state should produce the same evaluate() output."""
        model, conn = trained_causal_war
        result1 = model.evaluate(conn)
        result2 = model.evaluate(conn)

        for key in ["mean_causal_war", "std_causal_war", "median_causal_war"]:
            if key in result1 and key in result2:
                assert result1[key] == result2[key], (
                    f"Non-deterministic {key}: {result1[key]} != {result2[key]}"
                )


# ═════════════════════════════════════════════════════════════════════════════
# PSET Compute-Only Tests
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestPSETOutputValidation:
    """Validate PSET outputs are within expected value ranges."""

    def test_evaluate_returns_expected_keys(self, db_conn):
        """PSETModel.evaluate() should return standard keys."""
        from src.analytics.pset import PSETModel

        model = PSETModel()
        result = model.evaluate(db_conn)

        assert "qualifying_pitchers" in result
        assert "mean_pset_per_100" in result
        assert "std_pset_per_100" in result

    def test_pset_values_within_range(self, db_conn, baselines):
        """Mean PSET per 100 pitches should be within sane bounds."""
        from src.analytics.pset import PSETModel

        model = PSETModel()
        result = model.evaluate(db_conn)

        if result["qualifying_pitchers"] == 0:
            pytest.skip("No qualifying pitchers for PSET")

        mean_val = result["mean_pset_per_100"]
        pset_min = baselines["pset"]["pset_per_100_min"]
        pset_max = baselines["pset"]["pset_per_100_max"]
        assert pset_min <= mean_val <= pset_max, (
            f"PSET mean per 100 ({mean_val:.4f}) outside "
            f"[{pset_min}, {pset_max}] range"
        )

    def test_deterministic_evaluation(self, db_conn):
        """PSET evaluation should be deterministic."""
        from src.analytics.pset import PSETModel

        model = PSETModel()
        result1 = model.evaluate(db_conn)
        result2 = model.evaluate(db_conn)

        assert result1["qualifying_pitchers"] == result2["qualifying_pitchers"]
        assert result1["mean_pset_per_100"] == result2["mean_pset_per_100"]


# ═════════════════════════════════════════════════════════════════════════════
# Cross-Model Regression Detection
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.slow
class TestRegressionDetection:
    """Detect performance regressions by comparing current metrics to baselines.

    If any metric degrades more than REGRESSION_TOLERANCE (5%) from the
    stored baseline, the test fails.  Baselines are stored in
    ``tests/baselines.json`` and should be updated when models are
    intentionally retrained.
    """

    def test_stuff_plus_no_regression(self, stuff_model_artifact, db_conn, baselines):
        """Stuff+ R2 should not regress more than 5% below baseline."""
        from src.analytics.stuff_model import train_stuff_model

        try:
            metrics = train_stuff_model(db_conn)
        except ValueError:
            pytest.skip("Insufficient data for Stuff+ training")

        baseline_r2 = baselines["stuff_plus"]["r2_test_min"]
        actual_r2 = metrics["r2_test"]

        # Allow the tolerance margin: fail if actual < baseline * (1 - tolerance)
        floor = baseline_r2 * (1 - REGRESSION_TOLERANCE)
        assert actual_r2 >= floor, (
            f"Stuff+ R2 REGRESSION: {actual_r2:.4f} < {floor:.4f} "
            f"(baseline {baseline_r2} - {REGRESSION_TOLERANCE*100:.0f}%)"
        )

    def test_mechanix_ae_no_regression(self, mechanix_ae_artifact, baselines):
        """MechanixAE reconstruction loss should not increase beyond tolerance."""
        import torch
        from src.analytics.mechanix_ae import MechanixVAE, N_FEATURES, WINDOW_SIZE

        model = MechanixVAE()
        model.load_state_dict(mechanix_ae_artifact["model_state_dict"])
        model.eval()

        rng = np.random.RandomState(42)
        windows = rng.normal(0, 0.5, size=(50, WINDOW_SIZE, N_FEATURES)).astype(
            np.float32
        )
        tensor = torch.tensor(windows).permute(0, 2, 1)

        with torch.no_grad():
            recon, _, _ = model(tensor)
            recon_loss = torch.nn.functional.mse_loss(recon, tensor).item()

        baseline_loss = baselines["mechanix_ae"]["final_recon_loss_max"]
        ceiling = baseline_loss * (1 + REGRESSION_TOLERANCE)
        assert recon_loss <= ceiling, (
            f"MechanixAE REGRESSION: recon_loss {recon_loss:.4f} > {ceiling:.4f} "
            f"(baseline {baseline_loss} + {REGRESSION_TOLERANCE*100:.0f}%)"
        )

    def test_chemnet_synergy_no_regression(self, chemnet_artifact, baselines):
        """ChemNet synergy distribution should not shift beyond tolerance."""
        import torch
        from src.analytics.chemnet import (
            ChemNetGNN,
            BaselineMLP,
            NODE_FEATURE_DIM,
            _build_adjacency,
        )

        gnn = ChemNetGNN()
        gnn.load_state_dict(chemnet_artifact["gnn"])
        gnn.eval()
        baseline_mlp = BaselineMLP()
        baseline_mlp.load_state_dict(chemnet_artifact["baseline"])
        baseline_mlp.eval()

        adj = _build_adjacency(9)
        synergies = []
        torch.manual_seed(42)
        for _ in range(200):
            h = torch.randn(9, NODE_FEATURE_DIM)
            with torch.no_grad():
                gnn_pred, _ = gnn(h, adj)
                base_pred = baseline_mlp(h)
            synergies.append(gnn_pred.item() - base_pred.item())

        synergies = np.array(synergies)
        mean_syn = float(synergies.mean())

        syn_min = baselines["chemnet"]["mean_synergy_min"]
        syn_max = baselines["chemnet"]["mean_synergy_max"]

        # Widen the range by tolerance on each side
        adjusted_min = syn_min * (1 + REGRESSION_TOLERANCE) if syn_min < 0 else syn_min * (1 - REGRESSION_TOLERANCE)
        adjusted_max = syn_max * (1 + REGRESSION_TOLERANCE) if syn_max > 0 else syn_max * (1 - REGRESSION_TOLERANCE)

        assert adjusted_min <= mean_syn <= adjusted_max, (
            f"ChemNet REGRESSION: mean synergy {mean_syn:.4f} outside "
            f"[{adjusted_min:.4f}, {adjusted_max:.4f}]"
        )
