"""
PitchGPT v3 — outcome head on the v3 factorized backbone (spec §3.5).

    "Shape is [FIXED] unchanged from the A1 winner so the only moving parts are
    the ones under test: 3-layer MLP 211 -> 128 -> 64 -> 7 with ReLU + dropout
    0.1 over concat(hidden[128], context[35], type_onehot[17], zone_onehot[26],
    velo_onehot[5]), 7 outcome classes.

    [FIXED] One deliberate change: NO class weighting."

Optimizer settings are the A1 winner's verbatim (AdamW lr 1e-3, batch 32,
5 epochs, seed 42, model selection on the 2023 slice) minus the
inverse-frequency class weights that `PHASE_0.6_DIAGNOSIS.md` §6.1 identified
as the class-marginal-bias root cause.

Two protocol points the frozen spec leaves open are pre-registered here and
recorded as deviations-log entries 4 and 5 (written BEFORE this ran):

* the training window is §5.1's (2015-2022, same 10K-game cache) with §5.2's
  2023 pitcher-disjoint slice for model selection — the same tiering the
  backbone uses, and the only choice consistent with "training data may not
  exceed 2023";
* the backbone is FROZEN while the head trains (the A1 pattern), and the head
  is trained in the **PA-scoped regime** the rollout actually consumes, so
  §4.3's "the training-time rollout is the inference-time rollout" extends to
  the outcome head.

2025/2026 are never read.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt_outcome_head import (  # noqa: E402
    NUM_OUTCOME_CLASSES,
    OUTCOME_CLASSES,
)
from src.analytics.pitchgpt_v3 import (  # noqa: E402
    SPEC_BODY_SHA256,
    SPEC_FREEZE_SHA,
    SPEC_PATH,
    V3OutcomeHead,
    fields_from_token,
    load_v3_checkpoint,
)
from src.analytics.pitchgpt_v3_data import (  # noqa: E402
    build_pa_dataset,
    load_sequence_cache,
)
from src.analytics.pitchgpt_v3_infer import pa_batch_tensors  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitchgpt_v3_outcome_head")

DEFAULT_CACHE_DIR = Path(
    r"C:\Users\hunte\AppData\Local\Temp\claude"
    r"\C--Users-hunte-projects-baseball"
    r"\5112f9f4-f7ff-4db3-bd24-90acc5fbef27\scratchpad\pitchgpt_v3_cache"
)

EPOCHS = 5
LR = 1e-3
BATCH = 32
GRAD_CLIP = 1.0
SEED = 42
HORIZON = 6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _step(model, head, pa, sl, device):
    """Frozen-backbone features -> head logits for one PA slice."""
    prepped = pa_batch_tensors(pa, sl, device, HORIZON)
    if prepped is None:
        return None
    inp, ctx, tgt, valid = prepped
    with torch.no_grad():
        hidden = model.forward_hidden(inp, ctx)
    t_s, z_s, v_s = fields_from_token(tgt)
    logits = head(hidden, ctx, t_s, z_s, v_s)
    y = torch.from_numpy(
        pa["pa_outc"][sl, : tgt.shape[1]].astype(np.int64)
    ).to(device)
    m = valid & (y >= 0)
    if not bool(m.any()):
        return None
    return logits[m], y[m]


@torch.no_grad()
def evaluate(model, head, pa, device, batch: int = 512) -> dict:
    head.eval()
    tot = 0.0
    n = 0
    conf = np.zeros(NUM_OUTCOME_CLASSES, dtype=np.int64)
    pred_mass = np.zeros(NUM_OUTCOME_CLASSES, dtype=np.float64)
    n_pa = len(pa["pa_len"])
    for start in range(0, n_pa, batch):
        out = _step(model, head, pa, slice(start, start + batch), device)
        if out is None:
            continue
        logits, y = out
        tot += float(F.cross_entropy(logits, y, reduction="sum"))
        n += int(y.numel())
        conf += np.bincount(
            y.cpu().numpy(), minlength=NUM_OUTCOME_CLASSES
        ).astype(np.int64)
        pred_mass += F.softmax(logits, -1).sum(0).cpu().numpy()
    emp = conf / max(conf.sum(), 1)
    pred = pred_mass / max(n, 1)
    return {
        "log_loss": tot / max(n, 1),
        "n": n,
        "empirical_class_share": {
            OUTCOME_CLASSES[i]: float(emp[i]) for i in range(NUM_OUTCOME_CLASSES)
        },
        "predicted_class_share": {
            OUTCOME_CLASSES[i]: float(pred[i]) for i in range(NUM_OUTCOME_CLASSES)
        },
        "max_abs_class_share_gap": float(np.max(np.abs(pred - emp))),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument(
        "--backbone", type=Path,
        default=_ROOT / "models" / "pitchgpt_v3_factorized.pt",
    )
    ap.add_argument(
        "--out-checkpoint", type=Path,
        default=_ROOT / "models" / "pitchgpt_v3_outcomehead.pt",
    )
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", type=str, default="outcome_head")
    ap.add_argument("--limit-pas", type=int, default=0, help="smoke only")
    args = ap.parse_args()

    if args.out_checkpoint.exists():
        raise SystemExit(
            f"§0.4 forbids overwriting artifacts: {args.out_checkpoint} exists."
        )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_v3_checkpoint(args.backbone, device=device)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info("frozen backbone: %s (stage=%s)", args.backbone, ck["meta"]["stage"])

    pa_train = build_pa_dataset(
        load_sequence_cache(args.cache_dir / "train.npz"), horizon=HORIZON,
    )
    pa_fit = build_pa_dataset(
        load_sequence_cache(args.cache_dir / "fit23.npz"), horizon=HORIZON,
    )
    if args.limit_pas:
        n = int(args.limit_pas)
        pa_train = {
            **{k: v[:n] for k, v in pa_train.items() if k != "meta"},
            "meta": {**pa_train["meta"], "limited_to_pas": n},
        }
    logger.info(
        "PA cohorts: train %d | fit23 %d",
        len(pa_train["pa_len"]), len(pa_fit["pa_len"]),
    )

    head = V3OutcomeHead().to(device)
    n_head = sum(p.numel() for p in head.parameters())
    logger.info("outcome head: %d params (in_dim=%d)", n_head, head.in_dim)
    optim = torch.optim.AdamW(head.parameters(), lr=args.lr)

    rng = np.random.default_rng(args.seed)
    n_pa = len(pa_train["pa_len"])
    history: list[dict] = []
    best = {"epoch": None, "log_loss": float("inf"), "state": None}
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        head.train()
        t0 = time.perf_counter()
        order = rng.permutation(n_pa)
        run = 0.0
        run_n = 0
        for bi in range(0, n_pa, args.batch_size):
            idx = np.sort(order[bi : bi + args.batch_size])
            out = _step(model, head, pa_train, idx, device)
            if out is None:
                continue
            logits, y = out
            # §3.5: UNWEIGHTED cross-entropy. No inverse-frequency weights.
            loss = F.cross_entropy(logits, y)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP)
            optim.step()
            run += float(loss.item()) * int(y.numel())
            run_n += int(y.numel())
            if (bi // args.batch_size + 1) % 4000 == 0:
                logger.info(
                    "  epoch %d  batch %d  running log-loss %.4f",
                    epoch, bi // args.batch_size + 1, run / max(run_n, 1),
                )
        ev = evaluate(model, head, pa_fit, device)
        rec = {
            "epoch": epoch,
            "train_log_loss": run / max(run_n, 1),
            "fit2023": ev,
            "sec": round(time.perf_counter() - t0, 1),
        }
        history.append(rec)
        logger.info(
            "epoch %d  train %.5f  2023 log-loss %.5f  max class-share gap %.4f  %.0fs",
            epoch, rec["train_log_loss"], ev["log_loss"],
            ev["max_abs_class_share_gap"], rec["sec"],
        )
        if ev["log_loss"] < best["log_loss"]:
            best = {
                "epoch": epoch,
                "log_loss": ev["log_loss"],
                "state": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }

    head.load_state_dict(best["state"])
    wall = time.perf_counter() - t_start
    logger.info("best epoch %d (2023 log-loss %.5f)", best["epoch"], best["log_loss"])

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _ROOT / "results" / "pitchgpt_v3" / f"train_{args.tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "stage": "outcome_head",
        "spec_path": SPEC_PATH,
        "spec_freeze_sha": SPEC_FREEZE_SHA,
        "spec_body_sha256": SPEC_BODY_SHA256,
        "git_sha": git_sha(),
        "class_weighting": "NONE (§3.5 [FIXED] deliberate change vs A1)",
        "regime": "PA-scoped (BOS + PA pitches), backbone FROZEN",
        "backbone_checkpoint": str(args.backbone),
        "backbone_stage": ck["meta"]["stage"],
        "head_params": int(n_head),
        "head_topology": "211 -> 128 -> 64 -> 7, ReLU + dropout 0.1",
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "best_epoch": best["epoch"],
        "best_fit2023_log_loss": best["log_loss"],
        "train_pa_cohort": pa_train["meta"],
        "fit_pa_cohort": pa_fit["meta"],
        "history": history,
        "wall_clock_sec": round(wall, 1),
        "gpu_seconds": round(wall, 1),
        "duckdb_read_only": True,
    }
    args.out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "config": {"d_model": 128, "context_dim": 35, "n_classes": NUM_OUTCOME_CLASSES},
            "spec_path": SPEC_PATH,
            "spec_freeze_sha": SPEC_FREEZE_SHA,
            "meta": meta,
        },
        args.out_checkpoint,
    )
    import hashlib

    h = hashlib.sha256()
    with args.out_checkpoint.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    meta["checkpoint_sha256"] = h.hexdigest()
    meta["checkpoint_path"] = str(args.out_checkpoint)
    (out_dir / "audit.json").write_text(json.dumps(meta, indent=2))
    logger.info("checkpoint -> %s (sha256 %s)", args.out_checkpoint, h.hexdigest()[:16])
    logger.info("audit -> %s", out_dir / "audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
