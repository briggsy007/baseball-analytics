"""
PitchGPT v3 — Stage A: teacher-forced pretrain of the factorized backbone.

Spec: ``docs/pitchgpt_sim_engine/PITCHGPT_V2_SPEC.md`` §4.2 (FROZEN at
``b61e05b``).  [FIXED] protocol, reproduced verbatim:

    Train seasons per §5.1 (2015-2022, 10K games); 5 epochs; AdamW lr 1e-3;
    batch 32; grad clip 1.0; seed 42; loss = the unweighted sum of the three
    field cross-entropies

        L_A = CE(H_type, t*) + CE(H_zone(·, t*), z*) + CE(H_velo(·, t*, z*), v*)

    with ground-truth t*, z* used for conditioning.  Model selection: lowest
    2023-slice total loss (§5.2), checkpoint kept per epoch, best epoch
    retained.

Also emits the §3.6 ``support_counts`` table (computed ONCE from the training
split) into the checkpoint, and the §8.2 audit JSON.

Data policy: 2025/2026 are structurally unreachable — every cohort goes
through ``pitchgpt_v3_data.assert_season_policy``.  DuckDB is never opened by
this script; it consumes the caches built by
``scripts/pitchgpt_v3_build_cache.py`` (which itself uses read_only=True).
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
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt import PAD_TOKEN, CONTEXT_DIM  # noqa: E402
from src.analytics.pitchgpt_v3 import (  # noqa: E402
    SPEC_BODY_SHA256,
    SPEC_FREEZE_SHA,
    SPEC_PATH,
    FactorizedPitchGPT,
    SupportCounts,
    fields_from_token,
    save_v3_checkpoint,
)
from src.analytics.pitchgpt_v3_data import (  # noqa: E402
    CachedSequenceDataset,
    cached_collate,
    context_indices_to_tensor,
    load_sequence_cache,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitchgpt_v3_stage_a")

DEFAULT_CACHE_DIR = Path(
    r"C:\Users\hunte\AppData\Local\Temp\claude"
    r"\C--Users-hunte-projects-baseball"
    r"\5112f9f4-f7ff-4db3-bd24-90acc5fbef27\scratchpad\pitchgpt_v3_cache"
)

# ── §4.2 [FIXED] hyperparameters ─────────────────────────────────────────────
EPOCHS = 5
LR = 1e-3
BATCH = 32
GRAD_CLIP = 1.0
SEED = 42


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
    except Exception:  # pragma: no cover - provenance best effort
        return "unknown"


def make_loader(cache: dict, batch: int, shuffle: bool, generator=None) -> DataLoader:
    ds = CachedSequenceDataset(cache)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        collate_fn=lambda b: cached_collate(b, PAD_TOKEN),
        generator=generator,
        num_workers=0,
    )


def batch_to_device(batch, device):
    """Expand the categorical context to the 35-wide one-hot on device.

    PAD positions carry an all-zero context, matching
    :func:`src.analytics.pitchgpt._collate_fn`.
    """
    toks, ctx_idx, ump, tgts = (b.to(device, non_blocking=True) for b in batch)
    ctx = context_indices_to_tensor(ctx_idx, ump, CONTEXT_DIM)
    pad_mask = toks == PAD_TOKEN
    ctx = ctx.masked_fill(pad_mask.unsqueeze(-1), 0.0)
    return toks, ctx, tgts


def field_losses(
    model: FactorizedPitchGPT,
    toks: torch.Tensor,
    ctx: torch.Tensor,
    tgts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Teacher-forced §4.2 losses, summed over valid positions.

    Returns ``(sum_type, sum_zone, sum_velo, n_valid)`` — sums, not means, so
    epoch-level per-pitch NLL is exact under variable padding.
    """
    valid = tgts != PAD_TOKEN
    t_star, z_star, v_star = fields_from_token(tgts)
    hidden, lt, lz, lv = model(toks, ctx, target_type=t_star, target_zone=z_star)

    lt_f = lt[valid]
    lz_f = lz[valid]
    lv_f = lv[valid]
    s_t = F.cross_entropy(lt_f, t_star[valid], reduction="sum")
    s_z = F.cross_entropy(lz_f, z_star[valid], reduction="sum")
    s_v = F.cross_entropy(lv_f, v_star[valid], reduction="sum")
    return s_t, s_z, s_v, int(valid.sum().item())


@torch.no_grad()
def evaluate(model: FactorizedPitchGPT, loader: DataLoader, device) -> dict:
    model.eval()
    tot_t = tot_z = tot_v = 0.0
    n = 0
    for batch in loader:
        toks, ctx, tgts = batch_to_device(batch, device)
        s_t, s_z, s_v, k = field_losses(model, toks, ctx, tgts)
        tot_t += float(s_t.item())
        tot_z += float(s_z.item())
        tot_v += float(s_v.item())
        n += k
    total = (tot_t + tot_z + tot_v) / max(n, 1)
    return {
        "nll_per_pitch": total,
        "nll_type": tot_t / max(n, 1),
        "nll_zone": tot_z / max(n, 1),
        "nll_velo": tot_v / max(n, 1),
        "perplexity": float(np.exp(total)),
        "n_pitches": n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--factorized-input", action="store_true",
        help="§3.4 ablation A-EMB (dev-only, at most one run).",
    )
    ap.add_argument(
        "--out-checkpoint", type=Path,
        default=_ROOT / "models" / "pitchgpt_v3_factorized_stage_a.pt",
    )
    ap.add_argument("--tag", type=str, default="stage_a")
    ap.add_argument("--limit-sequences", type=int, default=0, help="smoke only")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    train_cache = load_sequence_cache(args.cache_dir / "train.npz")
    fit_cache = load_sequence_cache(args.cache_dir / "fit23.npz")
    logger.info(
        "train: %s | fit23: %s",
        train_cache["meta"]["n_pitches_kept"], fit_cache["meta"]["n_pitches_kept"],
    )

    if args.limit_sequences:
        n = int(args.limit_sequences)
        ptr = train_cache["seq_ptr"][: n + 1]
        end = int(ptr[-1])
        train_cache = {
            **{
                k: v[:end]
                for k in ("tok", "ctx_idx", "ump", "abn", "outc", "pa_pos")
                for v in [train_cache[k]]
            },
            "seq_ptr": ptr,
            "game_pk": train_cache["game_pk"][:n],
            "pitcher_id": train_cache["pitcher_id"][:n],
            "meta": {**train_cache["meta"], "limited_to_sequences": n},
        }

    # ── §3.6 support table: computed ONCE from the training split ────────
    support = SupportCounts.from_tokens(train_cache["tok"])
    logger.info("support table sha256=%s", support.sha256())

    model = FactorizedPitchGPT(factorized_input=args.factorized_input).to(device)
    model.attach_support(support)
    n_out = model.output_stack_parameters()
    logger.info(
        "output stack: %d params (flat head was 285,090; ratio %.3f)",
        n_out, n_out / 285090,
    )

    gen = torch.Generator()
    gen.manual_seed(args.seed)
    train_loader = make_loader(train_cache, args.batch_size, True, gen)
    fit_loader = make_loader(fit_cache, args.batch_size, False)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: list[dict] = []
    best = {"epoch": None, "fit_nll": float("inf"), "state": None}
    t_start = time.perf_counter()
    gpu_seconds = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        run_t = run_z = run_v = 0.0
        run_n = 0
        for i, batch in enumerate(train_loader):
            toks, ctx, tgts = batch_to_device(batch, device)
            s_t, s_z, s_v, k = field_losses(model, toks, ctx, tgts)
            if k == 0:
                continue
            loss = (s_t + s_z + s_v) / k
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optim.step()
            run_t += float(s_t.item())
            run_z += float(s_z.item())
            run_v += float(s_v.item())
            run_n += k
            if (i + 1) % 400 == 0:
                logger.info(
                    "  epoch %d  batch %d  running NLL/pitch %.4f",
                    epoch, i + 1, (run_t + run_z + run_v) / max(run_n, 1),
                )
        train_sec = time.perf_counter() - t0
        gpu_seconds += train_sec

        ev = evaluate(model, fit_loader, device)
        rec = {
            "epoch": epoch,
            "train_nll_per_pitch": (run_t + run_z + run_v) / max(run_n, 1),
            "train_nll_type": run_t / max(run_n, 1),
            "train_nll_zone": run_z / max(run_n, 1),
            "train_nll_velo": run_v / max(run_n, 1),
            "train_perplexity": float(np.exp((run_t + run_z + run_v) / max(run_n, 1))),
            "fit2023": ev,
            "train_sec": round(train_sec, 1),
        }
        history.append(rec)
        logger.info(
            "epoch %d  train NLL %.4f  2023-slice NLL %.4f (ppl %.2f)  %.1fs",
            epoch, rec["train_nll_per_pitch"], ev["nll_per_pitch"],
            ev["perplexity"], train_sec,
        )
        if ev["nll_per_pitch"] < best["fit_nll"]:
            best = {
                "epoch": epoch,
                "fit_nll": ev["nll_per_pitch"],
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }

    wall = time.perf_counter() - t_start
    logger.info("best epoch = %d (2023-slice NLL %.4f)", best["epoch"], best["fit_nll"])
    model.load_state_dict(best["state"])

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _ROOT / "results" / "pitchgpt_v3" / f"train_{args.tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "stage": "A_teacher_forced_pretrain",
        "spec_path": SPEC_PATH,
        "spec_freeze_sha": SPEC_FREEZE_SHA,
        "spec_body_sha256": SPEC_BODY_SHA256,
        "git_sha": git_sha(),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "grad_clip": GRAD_CLIP,
        "best_epoch": best["epoch"],
        "best_fit2023_nll_per_pitch": best["fit_nll"],
        "train_cohort": train_cache["meta"],
        "fit_cohort": fit_cache["meta"],
        "factorized_input_ablation": bool(args.factorized_input),
        "support_counts_sha256": support.sha256(),
        "output_stack_params": n_out,
        "flat_head_params": 285090,
        "wall_clock_sec": round(wall, 1),
        "gpu_seconds": round(gpu_seconds, 1),
        "duckdb_read_only": True,
        "history": history,
    }
    sha = save_v3_checkpoint(args.out_checkpoint, model, meta=meta)
    try:
        meta["checkpoint_path"] = str(args.out_checkpoint.relative_to(_ROOT))
    except ValueError:
        meta["checkpoint_path"] = str(args.out_checkpoint)
    meta["checkpoint_sha256"] = sha
    (out_dir / "audit.json").write_text(json.dumps(meta, indent=2))
    logger.info("checkpoint -> %s (sha256 %s)", args.out_checkpoint, sha[:16])
    logger.info("audit -> %s", out_dir / "audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
