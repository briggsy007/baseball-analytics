"""
Retrain PitchGPT with umpire accuracy-above-x context dim — versioned as v2.

Mirrors the pitcher-disjoint training protocol of
``results/validate_pitchgpt_20260418T150305Z/`` (the checkpoint against
which the prior 13.80%-vs-LSTM OOS delta was measured) so the old-vs-new
comparison is apples-to-apples:

    * Train = 2015-2022, pitcher-disjoint from val/test.
    * Val   = 2023.
    * Max games = 1000 / 300 / 300 (matches prior validation run).
    * 5 epochs, batch 32, AdamW lr 1e-3, grad clip 1.0, seed 42.

After training, saves to
``results/validate_pitchgpt_v2_ump_<timestamp>/pitchgpt_full.pt`` and to
``models/pitchgpt_v2.pt`` (versioned, not overwriting v1).

Smoke mode (``--smoke``) is the default: 1K games / 5 epochs.  Full
retrain (10K games) is queued as a follow-up — prior 10K run cost
~4h25m end-to-end, over the 2h RTX 3050 budget.
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

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt import (  # noqa: E402
    CONTEXT_DIM,
    PAD_TOKEN,
    TOTAL_VOCAB,
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
logger = logging.getLogger("pitchgpt_v2_ump")


TRAIN_RANGE = (2015, 2022)
VAL_RANGE = (2023, 2023)
TEST_RANGE = (2024, 2024)

DEFAULT_BATCH = 32
DEFAULT_LR = 1e-3
GRAD_CLIP = 1.0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-train-games", type=int, default=1000)
    parser.add_argument("--max-val-games", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact dir; defaults to results/validate_pitchgpt_v2_ump_<ts>/.",
    )
    parser.add_argument(
        "--version", type=str, default="2",
        help="Model version tag — saved to models/pitchgpt_v<version>.pt.",
    )
    args = parser.parse_args()

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if args.output_dir is None:
        args.output_dir = _ROOT / "results" / f"validate_pitchgpt_v2_ump_{ts}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)
    device = _get_device()
    logger.info("device=%s  seed=%d  CONTEXT_DIM=%d", device, args.seed, CONTEXT_DIM)

    # ── Load datasets (pitcher-disjoint protocol) ───────────────────────
    conn = get_connection(read_only=True)
    try:
        t0 = time.perf_counter()
        train_pitchers_all = PitchSequenceDataset.fetch_pitcher_ids_for_seasons(
            conn, range(TRAIN_RANGE[0], TRAIN_RANGE[1] + 1),
        )
        logger.info("train cohort: %d pitchers (2015-2022 full)", len(train_pitchers_all))

        train_ds = PitchSequenceDataset(
            conn,
            split_mode="train",
            train_range=TRAIN_RANGE,
            val_range=VAL_RANGE,
            test_range=TEST_RANGE,
            max_games_per_split=args.max_train_games,
        )
        val_ds = PitchSequenceDataset(
            conn,
            split_mode="val",
            train_range=TRAIN_RANGE,
            val_range=VAL_RANGE,
            test_range=TEST_RANGE,
            max_games_per_split=args.max_val_games,
            exclude_pitcher_ids=train_pitchers_all,
        )
        logger.info("datasets: train=%d  val=%d  (%.1fs)",
                    len(train_ds), len(val_ds), time.perf_counter() - t0)
    finally:
        conn.close()

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.error("Empty dataset — aborting.")
        return 1

    # Smoke-check the umpire scalar actually varies.  Sample across a
    # spread of sequence indices (not the first 5, which are all 2015
    # games with the 2014-tendencies-are-missing season-median fallback).
    idxs = sorted(set(
        int(i) for i in np.linspace(0, len(train_ds) - 1, num=min(50, len(train_ds)))
    ))
    sample_ump_vals: list[float] = []
    for i in idxs:
        _, ctx, _ = train_ds[i]
        sample_ump_vals.extend(ctx[:, -1].numpy().tolist())
    uniq = len(set(round(v, 3) for v in sample_ump_vals))
    logger.info(
        "ump-scalar smoke: n=%d unique=%d min=%.3f max=%.3f (from %d sequences)",
        len(sample_ump_vals), uniq,
        min(sample_ump_vals) if sample_ump_vals else 0.0,
        max(sample_ump_vals) if sample_ump_vals else 0.0,
        len(idxs),
    )
    if uniq < 3:
        logger.error(
            "ump scalar appears near-constant (unique=%d) — feature wiring broken "
            "or data sample unlucky.", uniq,
        )
        return 3

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_fn,
    )

    # ── Train ───────────────────────────────────────────────────────────
    _set_seed(args.seed)
    model = PitchGPTModel().to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info("PitchGPT v%s — %d params", args.version, params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    best_val_loss = float("inf")
    best_state: dict | None = None
    best_epoch = -1
    history: list[dict] = []
    train_t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total_loss, total_tok = 0.0, 0
        for tokens, ctx, target in train_loader:
            tokens = tokens.to(device)
            ctx = ctx.to(device)
            target = target.to(device)
            logits = model(tokens, ctx)
            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                target.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            n_tok = (target != PAD_TOKEN).sum().item()
            total_loss += loss.item() * n_tok
            total_tok += n_tok
        train_loss = total_loss / max(total_tok, 1)

        model.eval()
        v_loss, v_tok = 0.0, 0
        with torch.no_grad():
            for tokens, ctx, target in val_loader:
                tokens = tokens.to(device)
                ctx = ctx.to(device)
                target = target.to(device)
                logits = model(tokens, ctx)
                loss = criterion(
                    logits.reshape(-1, VOCAB_SIZE),
                    target.reshape(-1),
                )
                n_tok = (target != PAD_TOKEN).sum().item()
                v_loss += loss.item() * n_tok
                v_tok += n_tok
        val_loss = v_loss / max(v_tok, 1)

        dt = time.perf_counter() - t0
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "train_ppl": round(math.exp(min(train_loss, 20)), 3),
            "val_ppl": round(math.exp(min(val_loss, 20)), 3),
            "wall_clock_sec": round(dt, 1),
        }
        history.append(entry)
        logger.info(
            "ep %d/%d  train=%.4f (ppl %.2f)  val=%.4f (ppl %.2f)  %.1fs",
            epoch, args.epochs,
            train_loss, entry["train_ppl"],
            val_loss, entry["val_ppl"], dt,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    total_train_sec = round(time.perf_counter() - train_t0, 1)

    # ── Save ────────────────────────────────────────────────────────────
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 4,
            "max_seq_len": 256,
            "vocab_size": TOTAL_VOCAB,
            "context_dim": CONTEXT_DIM,  # 35 — new
            "context_schema_version": 2,
        },
        "version": args.version,
        "training_meta": {
            "timestamp_utc": ts,
            "train_range": TRAIN_RANGE,
            "val_range": VAL_RANGE,
            "test_range": TEST_RANGE,
            "max_train_games": args.max_train_games,
            "max_val_games": args.max_val_games,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "total_train_sec": total_train_sec,
            "pitcher_disjoint": True,
            "context_dim_added": "ump_accuracy_above_x (prior-season, "
                                 "NULL-filled with season league median)",
        },
        "history": history,
    }

    artifact_path = args.output_dir / "pitchgpt_full.pt"
    torch.save(ckpt, artifact_path)
    logger.info("wrote %s", artifact_path)

    # Also write a versioned copy under models/ so downstream callers can
    # pick it up by version tag without touching pitchgpt_v1.pt.
    models_dir = _ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    versioned_path = models_dir / f"pitchgpt_v{args.version}.pt"
    torch.save(ckpt, versioned_path)
    logger.info("wrote %s", versioned_path)

    summary_path = args.output_dir / "training_summary.json"
    summary_path.write_text(
        json.dumps({
            "version": args.version,
            "artifact": str(artifact_path),
            "versioned_copy": str(versioned_path),
            "context_dim": CONTEXT_DIM,
            "params": int(params),
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "total_train_sec": total_train_sec,
            "n_train_sequences": len(train_ds),
            "n_val_sequences": len(val_ds),
            "ump_scalar_smoke": {
                "n_samples": len(sample_ump_vals),
                "unique_rounded": uniq,
                "min": round(min(sample_ump_vals), 4) if sample_ump_vals else None,
                "max": round(max(sample_ump_vals), 4) if sample_ump_vals else None,
            },
        }, indent=2),
        encoding="utf-8",
    )
    logger.info("wrote %s", summary_path)

    logger.info("=" * 60)
    logger.info(
        "DONE — best val_loss=%.4f @ epoch %d; total train %.1fs",
        best_val_loss, best_epoch, total_train_sec,
    )
    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
