"""
PitchGPT v3 — K-v2-FIT-A: does the factorisation cost predictive quality?

Spec: ``PITCHGPT_V2_SPEC.md`` §7.1.

    After Stage A, compare teacher-forced NLL per pitch on the 2024 dev cohort
    against the frozen v2 backbone scored on the identical rows, using the same
    composite-token likelihood (v3's chain-rule product is multiplied back to a
    token probability, §3.3).

    KILL if v3's dev NLL per pitch is more than 2.0% relative WORSE than frozen
    v2's.  On kill: stop before Stage B; no rollout fine-tuning is run; publish
    the negative.

Exit codes: ``0`` = no kill, ``2`` = KILL signal.

The v1 backbone is scored too, purely as the informational third column the
§7.1 derivation cites (v1/v2 matched-scale deltas); it is **not** the kill
comparator.  2024 is the BURNED dev tier (§5.3) — these numbers carry no
validation authority and must be published as ``tier=burned-dev``.
"""

from __future__ import annotations

import argparse
import json
import logging
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

from src.analytics.pitchgpt import (  # noqa: E402
    CONTEXT_DIM,
    PAD_TOKEN,
    PitchGPTModel,
)
from src.analytics.pitchgpt_v3 import (  # noqa: E402
    SPEC_BODY_SHA256,
    SPEC_FREEZE_SHA,
    SPEC_PATH,
    fields_from_token,
    load_v3_checkpoint,
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
logger = logging.getLogger("pitchgpt_v3_dev_nll")

DEFAULT_CACHE_DIR = Path(
    r"C:\Users\hunte\AppData\Local\Temp\claude"
    r"\C--Users-hunte-projects-baseball"
    r"\5112f9f4-f7ff-4db3-bd24-90acc5fbef27\scratchpad\pitchgpt_v3_cache"
)
#: §7.1 [FIXED] kill threshold.
KILL_RELATIVE_WORSE = 0.020


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_flat_backbone(path: Path, device) -> tuple[PitchGPTModel, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    m = PitchGPTModel(
        vocab_size=cfg["vocab_size"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        context_dim=cfg.get("context_dim", CONTEXT_DIM),
    )
    m.load_state_dict(ck["model_state_dict"])
    m.to(device).eval()
    return m, cfg


def make_loader(cache: dict, batch: int) -> DataLoader:
    return DataLoader(
        CachedSequenceDataset(cache),
        batch_size=batch,
        shuffle=False,
        collate_fn=lambda b: cached_collate(b, PAD_TOKEN),
        num_workers=0,
    )


def _prep(batch, device, context_dim: int):
    toks, ctx_idx, ump, tgts = (b.to(device) for b in batch)
    ctx = context_indices_to_tensor(ctx_idx, ump, CONTEXT_DIM)
    ctx = ctx.masked_fill((toks == PAD_TOKEN).unsqueeze(-1), 0.0)
    if context_dim < CONTEXT_DIM:
        ctx = ctx[..., :context_dim]
    return toks, ctx, tgts


@torch.no_grad()
def score_v3(model, loader, device) -> dict:
    tot_t = tot_z = tot_v = 0.0
    n = 0
    for batch in loader:
        toks, ctx, tgts = _prep(batch, device, model.context_dim)
        valid = tgts != PAD_TOKEN
        t_s, z_s, v_s = fields_from_token(tgts)
        _, lt, lz, lv = model(toks, ctx, target_type=t_s, target_zone=z_s)
        tot_t += float(F.cross_entropy(lt[valid], t_s[valid], reduction="sum"))
        tot_z += float(F.cross_entropy(lz[valid], z_s[valid], reduction="sum"))
        tot_v += float(F.cross_entropy(lv[valid], v_s[valid], reduction="sum"))
        n += int(valid.sum())
    total = (tot_t + tot_z + tot_v) / n
    return {
        "nll_per_pitch": total,
        "perplexity": float(np.exp(total)),
        "nll_type": tot_t / n,
        "nll_zone": tot_z / n,
        "nll_velo": tot_v / n,
        "n_pitches": n,
    }


@torch.no_grad()
def score_flat(model, loader, device) -> dict:
    tot = 0.0
    n = 0
    for batch in loader:
        toks, ctx, tgts = _prep(batch, device, model.context_dim)
        valid = tgts != PAD_TOKEN
        logits = model(toks, ctx)
        tot += float(
            F.cross_entropy(logits[valid], tgts[valid], reduction="sum")
        )
        n += int(valid.sum())
    total = tot / n
    return {
        "nll_per_pitch": total,
        "perplexity": float(np.exp(total)),
        "n_pitches": n,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument(
        "--v3-checkpoint", type=Path,
        default=_ROOT / "models" / "pitchgpt_v3_factorized_stage_a.pt",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--tag", type=str, default="stage_a")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_cache = load_sequence_cache(args.cache_dir / "dev24.npz")
    loader = make_loader(dev_cache, args.batch_size)
    logger.info(
        "2024 dev cohort (BURNED tier): %d pitches / %d sequences",
        dev_cache["meta"]["n_pitches_kept"], dev_cache["meta"]["n_sequences"],
    )

    t0 = time.perf_counter()
    v3_model, v3_ck = load_v3_checkpoint(args.v3_checkpoint, device=device)
    v3 = score_v3(v3_model, loader, device)
    logger.info("v3  NLL/pitch %.5f  ppl %.3f", v3["nll_per_pitch"], v3["perplexity"])

    v2_path = _ROOT / "models" / "pitchgpt_v2.pt"
    v2_sha_before = sha256_file(v2_path)
    v2_model, _ = load_flat_backbone(v2_path, device)
    v2 = score_flat(v2_model, loader, device)
    logger.info("v2  NLL/pitch %.5f  ppl %.3f", v2["nll_per_pitch"], v2["perplexity"])

    v1_info = None
    v1_path = _ROOT / "models" / "pitchgpt_v1_10k.pt"
    if v1_path.exists():
        v1_model, _ = load_flat_backbone(v1_path, device)
        v1_info = score_flat(v1_model, loader, device)
        logger.info(
            "v1_10k (informational) NLL/pitch %.5f  ppl %.3f",
            v1_info["nll_per_pitch"], v1_info["perplexity"],
        )

    rel = (v3["nll_per_pitch"] - v2["nll_per_pitch"]) / v2["nll_per_pitch"]
    kill = bool(rel > KILL_RELATIVE_WORSE)
    logger.info(
        "K-v2-FIT-A: v3 is %+.3f%% relative vs frozen v2 (kill if > +%.1f%%) -> %s",
        rel * 100, KILL_RELATIVE_WORSE * 100, "KILL" if kill else "no kill",
    )

    v2_sha_after = sha256_file(v2_path)
    assert v2_sha_before == v2_sha_after, "frozen v2 checkpoint mutated!"

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _ROOT / "results" / "pitchgpt_v3" / f"dev_2024_nll_{args.tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate": "K-v2-FIT-A",
        "tier": "burned-dev",
        "spec_path": SPEC_PATH,
        "spec_freeze_sha": SPEC_FREEZE_SHA,
        "spec_body_sha256": SPEC_BODY_SHA256,
        "cohort": dev_cache["meta"],
        "v3": v3,
        "v3_checkpoint": str(args.v3_checkpoint),
        "v3_checkpoint_sha256": sha256_file(args.v3_checkpoint),
        "v3_spec_freeze_sha": v3_ck.get("spec_freeze_sha"),
        "frozen_v2": v2,
        "frozen_v2_sha256_pre": v2_sha_before,
        "frozen_v2_sha256_post": v2_sha_after,
        "v1_10k_informational": v1_info,
        "relative_nll_delta_vs_v2": rel,
        "kill_threshold_relative_worse": KILL_RELATIVE_WORSE,
        "kill": kill,
        "elapsed_sec": round(time.perf_counter() - t0, 1),
        "duckdb_read_only": True,
    }
    (out_dir / "audit.json").write_text(json.dumps(payload, indent=2))
    logger.info("audit -> %s", out_dir / "audit.json")
    return 2 if kill else 0


if __name__ == "__main__":
    raise SystemExit(main())
