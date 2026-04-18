"""
PitchGPT 4-way baseline comparison — combines spec tickets #3 (LSTM),
#4 (Markov-1 / Markov-2) and #5 (heuristic) into a single runner.

This script is the canonical "award evidence" producer for the PitchGPT
novelty claim.  It trains PitchGPT and the LSTM baseline on the *same*
leakage-clean date-based split (2015-2022 train, 2023 val, 2024 test),
fits Markov-1, Markov-2, and the Heuristic baseline on the training
sequences, and reports test-set perplexity for all five models side-by-
side.

Fairness conditions (PitchGPT + LSTM only):

    * Same train/val/test DataLoader (same split, same batch size).
    * Same optimizer (AdamW), learning rate, gradient clipping.
    * Same number of epochs, same seed.
    * Same CrossEntropyLoss with ``ignore_index=PAD_TOKEN``.

Markov and Heuristic models are closed-form — they see the same
training sequences but do not use epochs.

Outputs (written to ``--output-dir``):

    pitchgpt_baselines_metrics.json     -- full comparison verdict JSON.
    pitchgpt_vs_lstm_metrics.json       -- legacy alias (same contents).
    pitchgpt_training_curves.html       -- plotly loss overlay.

Spec thresholds:

    * PitchGPT vs LSTM      ≥ 15% perplexity improvement.
    * PitchGPT vs Markov-2  ≥ 20% perplexity improvement.
    * PitchGPT vs Heuristic ≥ 25% perplexity improvement.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure project root is importable when run as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitch_lstm import PitchLSTMNetwork  # noqa: E402
from src.analytics.pitch_markov import (  # noqa: E402
    HeuristicBaseline,
    MarkovChainOrder1,
    MarkovChainOrder2,
)
from src.analytics.pitchgpt import (  # noqa: E402
    PAD_TOKEN,
    VOCAB_SIZE,
    PitchGPTModel,
    PitchSequenceDataset,
    _collate_fn,
    _get_device,
    audit_no_game_overlap,
)
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitchgpt_baselines")


# ── Hyperparameters (kept symmetric across trained models) ──────────────────
DEFAULT_BATCH = 32
DEFAULT_LR = 1e-3
GRAD_CLIP = 1.0
TRAIN_RANGE = (2015, 2022)
VAL_RANGE = (2023, 2023)
TEST_RANGE = (2024, 2024)

# Spec thresholds.
THRESH_LSTM_PCT = 15.0
THRESH_MARKOV2_PCT = 20.0
THRESH_HEURISTIC_PCT = 25.0


# ═════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═════════════════════════════════════════════════════════════════════════════

def _set_seed(seed: int) -> None:
    """Seed python, numpy, torch, and CUDA (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═════════════════════════════════════════════════════════════════════════════
# Training / evaluation helpers (for neural models)
# ═════════════════════════════════════════════════════════════════════════════

def _iter_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    output_vocab: int,
    train: bool,
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total_loss, total_tok = 0.0, 0
    ctx_mgr = torch.enable_grad() if train else torch.no_grad()
    with ctx_mgr:
        for tokens, ctx, target in loader:
            tokens = tokens.to(device)
            ctx = ctx.to(device)
            target = target.to(device)

            logits = model(tokens, ctx)
            loss = criterion(
                logits.reshape(-1, output_vocab),
                target.reshape(-1),
            )

            if train:
                assert optimizer is not None
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            n_tok = (target != PAD_TOKEN).sum().item()
            total_loss += loss.item() * n_tok
            total_tok += n_tok

    return total_loss / max(total_tok, 1)


def _train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    output_vocab: int,
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state: dict | None = None
    history: list[dict] = []

    logger.info("[%s] starting %d-epoch training", name, epochs)
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = _iter_epoch(
            model, train_loader, optimizer, criterion, device,
            output_vocab=output_vocab, train=True,
        )
        val_loss = _iter_epoch(
            model, val_loader, None, criterion, device,
            output_vocab=output_vocab, train=False,
        )
        dt = time.perf_counter() - t0

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_perplexity": round(math.exp(min(train_loss, 20)), 3),
            "val_perplexity": round(math.exp(min(val_loss, 20)), 3),
            "wall_clock_sec": round(dt, 1),
        }
        history.append(entry)
        logger.info(
            "[%s] epoch %d/%d  train=%.4f (ppl %.2f)  val=%.4f (ppl %.2f)  %.1fs",
            name, epoch, epochs,
            train_loss, entry["train_perplexity"],
            val_loss, entry["val_perplexity"],
            dt,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss = _iter_epoch(
        model, test_loader, None, criterion, device,
        output_vocab=output_vocab, train=False,
    )
    test_perplexity = math.exp(min(test_loss, 20))
    logger.info(
        "[%s] BEST epoch %d  test_loss=%.4f  test_ppl=%.3f",
        name, best_epoch, test_loss, test_perplexity,
    )

    params = sum(p.numel() for p in model.parameters())
    final = history[-1]
    return {
        "params": int(params),
        "train_loss_final": final["train_loss"],
        "val_loss_final": final["val_loss"],
        "test_loss": round(test_loss, 4),
        "test_perplexity": round(test_perplexity, 3),
        "epoch_best": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "history": history,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Closed-form baselines (Markov / Heuristic)
# ═════════════════════════════════════════════════════════════════════════════

def _evaluate_closed_form(
    name: str,
    model_ctor,
    train_ds: PitchSequenceDataset,
    val_ds: PitchSequenceDataset,
    test_ds: PitchSequenceDataset,
) -> dict:
    """Fit a closed-form baseline on ``train_ds`` and return its test
    perplexity + val perplexity + wall-clock.

    These models have no "epochs" — fit is one pass, score is another.
    """
    t0 = time.perf_counter()
    logger.info("[%s] fitting closed-form model on %d train sequences",
                name, len(train_ds))
    m = model_ctor()
    m.fit(train_ds)
    fit_sec = time.perf_counter() - t0

    t1 = time.perf_counter()
    train_ppl = m.calculate_perplexity(train_ds)
    val_ppl = m.calculate_perplexity(val_ds)
    test_ppl = m.calculate_perplexity(test_ds)
    score_sec = time.perf_counter() - t1

    wall = round(fit_sec + score_sec, 1)
    logger.info(
        "[%s] fit=%.1fs  score=%.1fs  train_ppl=%.3f  val_ppl=%.3f  test_ppl=%.3f",
        name, fit_sec, score_sec, train_ppl, val_ppl, test_ppl,
    )

    return {
        "params": 0,  # non-parametric
        "train_perplexity": round(train_ppl, 3),
        "val_perplexity": round(val_ppl, 3),
        "test_perplexity": round(test_ppl, 3),
        "train_loss": round(math.log(train_ppl), 4) if math.isfinite(train_ppl) else None,
        "val_loss": round(math.log(val_ppl), 4) if math.isfinite(val_ppl) else None,
        "test_loss": round(math.log(test_ppl), 4) if math.isfinite(test_ppl) else None,
        "wall_clock_sec": wall,
        "fit_sec": round(fit_sec, 1),
        "score_sec": round(score_sec, 1),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Plotly training-curve chart
# ═════════════════════════════════════════════════════════════════════════════

def _write_training_curves_html(
    out_path: Path,
    pitchgpt_hist: list[dict],
    lstm_hist: list[dict],
    baselines: dict | None = None,
) -> None:
    """Write standalone HTML comparing all five models.

    PitchGPT + LSTM get full train/val curves; Markov and Heuristic
    appear as horizontal lines at their test-set perplexity.
    """
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError:
        logger.warning("plotly not installed — skipping training-curves HTML")
        out_path.write_text(
            "<html><body><p>plotly not installed; no chart emitted.</p>"
            "</body></html>",
            encoding="utf-8",
        )
        return

    fig = go.Figure()
    # PitchGPT.
    fig.add_trace(go.Scatter(
        x=[h["epoch"] for h in pitchgpt_hist],
        y=[h["train_loss"] for h in pitchgpt_hist],
        mode="lines+markers", name="PitchGPT train",
        line=dict(color="#2a6df4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[h["epoch"] for h in pitchgpt_hist],
        y=[h["val_loss"] for h in pitchgpt_hist],
        mode="lines+markers", name="PitchGPT val",
        line=dict(color="#2a6df4", dash="dash", width=2),
    ))
    # LSTM.
    fig.add_trace(go.Scatter(
        x=[h["epoch"] for h in lstm_hist],
        y=[h["train_loss"] for h in lstm_hist],
        mode="lines+markers", name="LSTM train",
        line=dict(color="#e04e4e", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=[h["epoch"] for h in lstm_hist],
        y=[h["val_loss"] for h in lstm_hist],
        mode="lines+markers", name="LSTM val",
        line=dict(color="#e04e4e", dash="dash", width=2),
    ))

    # Closed-form baselines — horizontal lines over the epoch axis at
    # ``log(test_ppl)`` so they live on the same axis as train/val loss.
    epochs_axis = [h["epoch"] for h in pitchgpt_hist]
    colors = {
        "markov1": "#9467bd",
        "markov2": "#7f3e80",
        "heuristic": "#888888",
    }
    baselines = baselines or {}
    for key, color in colors.items():
        info = baselines.get(key)
        if info is None:
            continue
        test_loss = info.get("test_loss")
        if test_loss is None:
            continue
        fig.add_trace(go.Scatter(
            x=epochs_axis,
            y=[test_loss] * len(epochs_axis),
            mode="lines",
            name=f"{key} test (closed-form)",
            line=dict(color=color, dash="dot", width=2),
        ))

    fig.update_layout(
        title="PitchGPT vs LSTM vs Markov / Heuristic — Cross-Entropy",
        xaxis_title="Epoch",
        yaxis_title="Cross-entropy loss (lower is better)",
        legend=dict(x=0.02, y=0.98),
        template="plotly_white",
        width=960, height=540,
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    logger.info("wrote training curves to %s", out_path)


# ═════════════════════════════════════════════════════════════════════════════
# Verdicts
# ═════════════════════════════════════════════════════════════════════════════

def _build_pairwise_verdict(
    challenger_ppl: float,
    baseline_ppl: float,
    baseline_name: str,
    threshold_pct: float,
) -> tuple[float, bool, str]:
    """Return ``(improvement_pct, passed, verdict_str)`` for one pair."""
    if baseline_ppl <= 0 or not math.isfinite(baseline_ppl):
        return 0.0, False, f"FAIL — {baseline_name} perplexity invalid."
    imp = 100.0 * (baseline_ppl - challenger_ppl) / baseline_ppl
    imp = round(imp, 2)
    passed = imp >= threshold_pct
    prefix = "PASS" if passed else "FAIL"
    verdict = (
        f"{prefix} — PitchGPT is {imp:.1f}% better than {baseline_name} "
        f"(spec requires ≥{threshold_pct:.0f}%)."
    )
    return imp, passed, verdict


# Legacy helper kept for backward compatibility with the old test file.
def _build_verdict(improvement_pct: float) -> tuple[bool, str]:
    """Return ``(passed, verdict_str)`` against the ≥15% LSTM threshold."""
    _, passed, verdict = _build_pairwise_verdict(
        challenger_ppl=100.0 - improvement_pct,  # fake values chosen so
        baseline_ppl=100.0,                      # the math yields the
        baseline_name="LSTM",                    # same improvement_pct.
        threshold_pct=THRESH_LSTM_PCT,
    )
    return passed, verdict


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train-games", type=int, default=1000)
    parser.add_argument("--max-val-games", type=int, default=300)
    parser.add_argument("--max-test-games", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="Directory for JSON + HTML artifacts.",
    )
    parser.add_argument(
        "--db-path", type=str, default=None,
        help="Optional override for the DuckDB file.",
    )
    parser.add_argument(
        "--skip-closed-form", action="store_true",
        help="Skip Markov / Heuristic baselines (for a 2-way regression check).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)
    device = _get_device()
    logger.info("device=%s  seed=%d", device, args.seed)

    # ── Data ────────────────────────────────────────────────────────────
    conn = get_connection(args.db_path, read_only=True)
    try:
        logger.info(
            "loading datasets (train=%s val=%s test=%s, max_games %d/%d/%d)",
            TRAIN_RANGE, VAL_RANGE, TEST_RANGE,
            args.max_train_games, args.max_val_games, args.max_test_games,
        )
        t_ds0 = time.perf_counter()
        train_ds = PitchSequenceDataset(
            conn, split_mode="train",
            train_range=TRAIN_RANGE, val_range=VAL_RANGE, test_range=TEST_RANGE,
            max_games_per_split=args.max_train_games,
        )
        val_ds = PitchSequenceDataset(
            conn, split_mode="val",
            train_range=TRAIN_RANGE, val_range=VAL_RANGE, test_range=TEST_RANGE,
            max_games_per_split=args.max_val_games,
        )
        test_ds = PitchSequenceDataset(
            conn, split_mode="test",
            train_range=TRAIN_RANGE, val_range=VAL_RANGE, test_range=TEST_RANGE,
            max_games_per_split=args.max_test_games,
        )
        logger.info(
            "datasets built in %.1fs  (train=%d  val=%d  test=%d sequences)",
            time.perf_counter() - t_ds0,
            len(train_ds), len(val_ds), len(test_ds),
        )
    finally:
        conn.close()

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        logger.error(
            "Empty dataset(s): train=%d val=%d test=%d — aborting.",
            len(train_ds), len(val_ds), len(test_ds),
        )
        return 1

    audit = audit_no_game_overlap(train_ds, val_ds, test_ds)
    logger.info("leakage audit: %s", audit)
    if audit["shared_game_pks"] != 0:
        logger.error("LEAKAGE DETECTED — refusing to train.")
        return 2

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_fn,
    )

    # ── PitchGPT ────────────────────────────────────────────────────────
    _set_seed(args.seed)
    pitchgpt = PitchGPTModel().to(device)
    t_tx0 = time.perf_counter()
    pitchgpt_metrics = _train_model(
        name="PitchGPT",
        model=pitchgpt,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        output_vocab=VOCAB_SIZE,
    )
    pitchgpt_wallclock = round(time.perf_counter() - t_tx0, 1)

    # ── LSTM baseline ───────────────────────────────────────────────────
    _set_seed(args.seed)
    lstm = PitchLSTMNetwork().to(device)
    t_lstm0 = time.perf_counter()
    lstm_metrics = _train_model(
        name="LSTM",
        model=lstm,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        output_vocab=lstm.output_vocab,
    )
    lstm_wallclock = round(time.perf_counter() - t_lstm0, 1)

    # ── Closed-form baselines ───────────────────────────────────────────
    closed_form: dict[str, dict] = {}
    if not args.skip_closed_form:
        closed_form["markov1"] = _evaluate_closed_form(
            "Markov-1", MarkovChainOrder1, train_ds, val_ds, test_ds,
        )
        closed_form["markov2"] = _evaluate_closed_form(
            "Markov-2", MarkovChainOrder2, train_ds, val_ds, test_ds,
        )
        closed_form["heuristic"] = _evaluate_closed_form(
            "Heuristic", HeuristicBaseline, train_ds, val_ds, test_ds,
        )

    # ── Verdicts ────────────────────────────────────────────────────────
    tx_ppl = pitchgpt_metrics["test_perplexity"]
    ls_ppl = lstm_metrics["test_perplexity"]
    m2_ppl = closed_form.get("markov2", {}).get("test_perplexity")
    h_ppl = closed_form.get("heuristic", {}).get("test_perplexity")

    imp_lstm, pass_lstm, v_lstm = _build_pairwise_verdict(
        tx_ppl, ls_ppl, "LSTM", THRESH_LSTM_PCT,
    )
    if m2_ppl is not None:
        imp_m2, pass_m2, v_m2 = _build_pairwise_verdict(
            tx_ppl, m2_ppl, "Markov-2", THRESH_MARKOV2_PCT,
        )
    else:
        imp_m2, pass_m2, v_m2 = 0.0, False, "SKIPPED"

    if h_ppl is not None:
        imp_h, pass_h, v_h = _build_pairwise_verdict(
            tx_ppl, h_ppl, "Heuristic", THRESH_HEURISTIC_PCT,
        )
    else:
        imp_h, pass_h, v_h = 0.0, False, "SKIPPED"

    overall_pass = pass_lstm and pass_m2 and pass_h

    # ── Assemble JSON ───────────────────────────────────────────────────
    out_json = {
        "seed": args.seed,
        "device": str(device),
        "n_train_sequences": len(train_ds),
        "n_val_sequences": len(val_ds),
        "n_test_sequences": len(test_ds),
        "train_range": f"{TRAIN_RANGE[0]}-{TRAIN_RANGE[1]}"
                        if TRAIN_RANGE[0] != TRAIN_RANGE[1]
                        else f"{TRAIN_RANGE[0]}",
        "val_range": f"{VAL_RANGE[0]}-{VAL_RANGE[1]}"
                       if VAL_RANGE[0] != VAL_RANGE[1]
                       else f"{VAL_RANGE[0]}",
        "test_range": f"{TEST_RANGE[0]}-{TEST_RANGE[1]}"
                        if TEST_RANGE[0] != TEST_RANGE[1]
                        else f"{TEST_RANGE[0]}",
        "max_train_games": args.max_train_games,
        "max_val_games": args.max_val_games,
        "max_test_games": args.max_test_games,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "leakage_audit": audit,
        "pitchgpt": {
            "params": pitchgpt_metrics["params"],
            "train_loss_final": pitchgpt_metrics["train_loss_final"],
            "val_loss_final": pitchgpt_metrics["val_loss_final"],
            "test_loss": pitchgpt_metrics["test_loss"],
            "test_perplexity": pitchgpt_metrics["test_perplexity"],
            "epoch_best": pitchgpt_metrics["epoch_best"],
            "best_val_loss": pitchgpt_metrics["best_val_loss"],
            "wall_clock_sec": pitchgpt_wallclock,
            "history": pitchgpt_metrics["history"],
        },
        "lstm": {
            "params": lstm_metrics["params"],
            "train_loss_final": lstm_metrics["train_loss_final"],
            "val_loss_final": lstm_metrics["val_loss_final"],
            "test_loss": lstm_metrics["test_loss"],
            "test_perplexity": lstm_metrics["test_perplexity"],
            "epoch_best": lstm_metrics["epoch_best"],
            "best_val_loss": lstm_metrics["best_val_loss"],
            "wall_clock_sec": lstm_wallclock,
            "history": lstm_metrics["history"],
        },
        "markov1": closed_form.get("markov1"),
        "markov2": closed_form.get("markov2"),
        "heuristic": closed_form.get("heuristic"),
        "comparisons": {
            "pitchgpt_vs_lstm_pct": imp_lstm,
            "pitchgpt_vs_markov2_pct": imp_m2,
            "pitchgpt_vs_heuristic_pct": imp_h,
        },
        "thresholds": {
            "lstm_pct": THRESH_LSTM_PCT,
            "markov2_pct": THRESH_MARKOV2_PCT,
            "heuristic_pct": THRESH_HEURISTIC_PCT,
        },
        "pass": {
            "vs_lstm": bool(pass_lstm),
            "vs_markov2": bool(pass_m2),
            "vs_heuristic": bool(pass_h),
            "overall": bool(overall_pass),
        },
        "verdict": {
            "vs_lstm": v_lstm,
            "vs_markov2": v_m2,
            "vs_heuristic": v_h,
            "overall": ("PASS — PitchGPT beats every baseline above spec "
                        "threshold.") if overall_pass else (
                        "FAIL — PitchGPT does NOT beat every baseline above "
                        "spec threshold."),
        },
        # Legacy fields kept so the existing 2-way test harness doesn't
        # break.  Their values are the same as the LSTM comparison above.
        "perplexity_improvement_pct": imp_lstm,
        "spec_threshold_15pct": bool(pass_lstm),
    }

    # Write canonical + legacy JSON names.
    canonical_path = args.output_dir / "pitchgpt_baselines_metrics.json"
    canonical_path.write_text(
        json.dumps(out_json, indent=2), encoding="utf-8",
    )
    logger.info("wrote metrics to %s", canonical_path)

    legacy_path = args.output_dir / "pitchgpt_vs_lstm_metrics.json"
    legacy_path.write_text(
        json.dumps(out_json, indent=2), encoding="utf-8",
    )

    # ── Plotly training curves ──────────────────────────────────────────
    html_path = args.output_dir / "pitchgpt_training_curves.html"
    _write_training_curves_html(
        html_path, pitchgpt_metrics["history"], lstm_metrics["history"],
        closed_form,
    )

    # ── Console verdict ─────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("PitchGPT test perplexity:   %.3f", tx_ppl)
    logger.info("LSTM      test perplexity:  %.3f  (Δ %+.2f%%)", ls_ppl, imp_lstm)
    if m2_ppl is not None:
        logger.info("Markov-2 test perplexity:   %.3f  (Δ %+.2f%%)", m2_ppl, imp_m2)
    if h_ppl is not None:
        logger.info("Heuristic test perplexity:  %.3f  (Δ %+.2f%%)", h_ppl, imp_h)
    logger.info("-" * 70)
    logger.info(v_lstm)
    logger.info(v_m2)
    logger.info(v_h)
    logger.info("OVERALL: %s", "PASS" if overall_pass else "FAIL")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
