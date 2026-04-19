"""
PitchGPT calibration analysis runner — Ticket #7.

Loads a trained PitchGPT checkpoint, runs it over the 2023 validation
split (to fit the temperature) and the 2024 test split (to measure
ECE pre- and post-temperature), and writes:

    <output_dir>/pitchgpt_calibration.json
    <output_dir>/pitchgpt_reliability.html

The checkpoint path defaults to the full-variant checkpoint produced by
``scripts/pitchgpt_ablation.py`` (``results/run_5epoch/pitchgpt_full.pt``);
alternatively, ``--checkpoint models/pitchgpt_v1.pt`` can be supplied.

This is an inference-only job — no training happens here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt import (  # noqa: E402
    TOTAL_VOCAB,
    PitchGPTModel,
    PitchSequenceDataset,
    _get_device,
)
from src.analytics.pitchgpt_calibration import full_calibration_audit  # noqa: E402
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitchgpt_calibration")


TRAIN_RANGE = (2015, 2022)
VAL_RANGE = (2023, 2023)
TEST_RANGE = (2024, 2024)


def _load_checkpoint(path: Path, device: torch.device) -> PitchGPTModel:
    ck = torch.load(str(path), map_location=device, weights_only=True)
    cfg = ck["config"]
    model = PitchGPTModel(
        vocab_size=cfg.get("vocab_size", TOTAL_VOCAB),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
    )
    model.load_state_dict(ck["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _write_reliability_html(
    out_path: Path,
    audit: dict,
) -> None:
    """Write a plotly reliability diagram (pre + post temperature)."""
    try:
        import plotly.graph_objects as go  # type: ignore
    except ImportError:
        logger.warning("plotly not installed — skipping HTML chart")
        out_path.write_text(
            "<html><body><p>plotly not installed.</p></body></html>",
            encoding="utf-8",
        )
        return

    pre = audit["reliability_curve_pre"]
    post = audit["reliability_curve_post"]
    ece_pre = audit["ece_pre_temp"]
    ece_post = audit["ece_post_temp"]
    T = audit["optimal_temperature"]

    fig = go.Figure()
    # Perfect-calibration diagonal.
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="perfect calibration",
        line=dict(color="#cccccc", dash="dash"),
    ))
    # Pre-temperature reliability.
    fig.add_trace(go.Scatter(
        x=[b["mean_conf"] for b in pre],
        y=[b["empirical_acc"] for b in pre],
        mode="lines+markers",
        name=f"pre-temp (ECE={ece_pre:.3f})",
        line=dict(color="#e04e4e", width=2),
        marker=dict(size=[max(4, min(20, b['n_samples']**0.33)) for b in pre]),
        text=[f"n={b['n_samples']}" for b in pre],
        hovertemplate="conf=%{x:.3f}  acc=%{y:.3f}  %{text}",
    ))
    # Post-temperature.
    fig.add_trace(go.Scatter(
        x=[b["mean_conf"] for b in post],
        y=[b["empirical_acc"] for b in post],
        mode="lines+markers",
        name=f"post-temp T={T:.3f}  (ECE={ece_post:.3f})",
        line=dict(color="#2a6df4", width=2),
        marker=dict(size=[max(4, min(20, b['n_samples']**0.33)) for b in post]),
        text=[f"n={b['n_samples']}" for b in post],
        hovertemplate="conf=%{x:.3f}  acc=%{y:.3f}  %{text}",
    ))

    fig.update_layout(
        title=(
            f"PitchGPT reliability diagram — test set ({audit['n_test_tokens']:,} "
            f"tokens). ECE {ece_pre:.3f} → {ece_post:.3f} after T={T:.3f}."
        ),
        xaxis_title="Predicted top-1 probability",
        yaxis_title="Empirical accuracy",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        template="plotly_white",
        width=800, height=700,
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path,
        default=Path("results/run_5epoch/pitchgpt_full.pt"),
        help="PitchGPT checkpoint to calibrate.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/run_5epoch"),
    )
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument("--max-val-games", type=int, default=300)
    parser.add_argument("--max-test-games", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = _get_device()
    logger.info("device=%s", device)

    # ── Model ────────────────────────────────────────────────────────────
    if not args.checkpoint.exists():
        # Fallback: use the existing models/pitchgpt_v1.pt.
        alt = Path("models/pitchgpt_v1.pt")
        if alt.exists():
            logger.warning(
                "checkpoint %s missing — falling back to %s",
                args.checkpoint, alt,
            )
            args.checkpoint = alt
        else:
            logger.error("no PitchGPT checkpoint found at %s", args.checkpoint)
            return 1
    logger.info("loading checkpoint %s", args.checkpoint)
    model = _load_checkpoint(args.checkpoint, device)

    # ── Data ─────────────────────────────────────────────────────────────
    conn = get_connection(args.db_path, read_only=True)
    try:
        logger.info("loading val/test datasets (%s / %s)", VAL_RANGE, TEST_RANGE)
        t0 = time.perf_counter()
        # Pitcher-disjoint split (Ticket #1 hardening): exclude every
        # pitcher who appears in any train season from val/test so the
        # calibration audit doesn't measure ECE on pitchers the model
        # already saw at training time.
        train_seasons_full = list(range(TRAIN_RANGE[0], TRAIN_RANGE[1] + 1))
        train_pitcher_ids = PitchSequenceDataset.fetch_pitcher_ids_for_seasons(
            conn, train_seasons_full,
        )
        logger.info(
            "excluding %d train-cohort pitchers from val/test",
            len(train_pitcher_ids),
        )
        val_ds = PitchSequenceDataset(
            conn, split_mode="val",
            train_range=TRAIN_RANGE, val_range=VAL_RANGE, test_range=TEST_RANGE,
            max_games_per_split=args.max_val_games,
            exclude_pitcher_ids=train_pitcher_ids,
        )
        test_ds = PitchSequenceDataset(
            conn, split_mode="test",
            train_range=TRAIN_RANGE, val_range=VAL_RANGE, test_range=TEST_RANGE,
            max_games_per_split=args.max_test_games,
            exclude_pitcher_ids=train_pitcher_ids,
        )
        logger.info(
            "datasets built in %.1fs  (val=%d  test=%d sequences)",
            time.perf_counter() - t0, len(val_ds), len(test_ds),
        )
    finally:
        conn.close()

    if len(val_ds) == 0 or len(test_ds) == 0:
        logger.error("empty val or test dataset — aborting")
        return 1

    # ── Audit ────────────────────────────────────────────────────────────
    t1 = time.perf_counter()
    audit = full_calibration_audit(
        model, val_dataset=val_ds, test_dataset=test_ds,
        n_bins=args.n_bins, batch_size=args.batch_size, device=device,
    )
    audit["wall_clock_sec"] = round(time.perf_counter() - t1, 1)
    audit["checkpoint"] = str(args.checkpoint)
    logger.info(
        "calibration done in %.1fs  ECE %.4f → %.4f  T=%.4f",
        audit["wall_clock_sec"], audit["ece_pre_temp"],
        audit["ece_post_temp"], audit["optimal_temperature"],
    )

    out_json = args.output_dir / "pitchgpt_calibration.json"
    out_json.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    logger.info("wrote %s", out_json)

    out_html = args.output_dir / "pitchgpt_reliability.html"
    _write_reliability_html(out_html, audit)
    logger.info("wrote %s", out_html)

    logger.info("=" * 60)
    logger.info("CALIBRATION SUMMARY")
    logger.info("  n_test_tokens:        %d", audit["n_test_tokens"])
    logger.info("  accuracy (top-1):     %.4f", audit["accuracy"])
    logger.info("  ECE pre-temperature:  %.4f", audit["ece_pre_temp"])
    logger.info("  ECE post-temperature: %.4f", audit["ece_post_temp"])
    logger.info("  optimal T:            %.4f", audit["optimal_temperature"])
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
