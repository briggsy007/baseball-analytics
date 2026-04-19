#!/usr/bin/env python
"""Train ChemNet v2 (opposing-pitcher graph + residual objective).

Saves the trained checkpoint to ``models/chemnet_v2.pt`` and writes the
training metrics dictionary as JSON to a sibling file (or to a path passed
via ``--metrics-out``).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analytics.chemnet import train_chemnet_v2, MODELS_DIR  # noqa: E402
from src.db.schema import get_connection  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_chemnet_v2")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-start", type=int, default=2015)
    p.add_argument("--train-end", type=int, default=2022)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-games", type=int, default=9000)
    p.add_argument("--val-frac", type=float, default=0.08)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", type=Path, default=MODELS_DIR / "chemnet_v2.pt")
    p.add_argument("--metrics-out", type=Path, default=None)
    args = p.parse_args(argv)

    conn = get_connection(read_only=True)
    seasons = range(args.train_start, args.train_end + 1)

    metrics = train_chemnet_v2(
        conn,
        seasons=seasons,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_games=args.max_games,
        val_frac=args.val_frac,
        early_stop_patience=args.patience,
        save_path=args.save_path,
        seed=args.seed,
    )
    conn.close()

    if args.metrics_out is None:
        args.metrics_out = args.save_path.with_suffix(".training_metrics.json")
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Wrote training metrics to %s", args.metrics_out)

    print(json.dumps(
        {k: v for k, v in metrics.items()
         if k not in {"train_gnn_losses", "train_base_losses",
                      "val_gnn_losses", "val_base_losses"}},
        indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
