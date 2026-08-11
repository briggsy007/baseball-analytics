"""
PitchGPT v3 — Stage B: curriculum rollout-aware fine-tune.

Spec: ``PITCHGPT_V2_SPEC.md`` §4.3 [FIXED], starting from the Stage A best
checkpoint, same optimizer settings except ``lr = 2e-4``:

===========  =============  ======  ==========================================
sub-stage    rollout depth  epochs  scheduled-sampling probability eps
===========  =============  ======  ==========================================
B1           2 steps        1       linear ramp 0.00 -> 0.25 across the epoch
B2           4 steps        1       linear ramp 0.25 -> 0.50 across the epoch
B3           full PA (6)    1       fixed 0.50
===========  =============  ======  ==========================================

"At each rolled position the input token is the model's own sampled token with
probability ``eps`` and the ground-truth token otherwise; the loss remains
cross-entropy against the REAL continuation at every position, so no
non-differentiable reward machinery is introduced.  Sampling inside the
curriculum uses the §3.6 masks and the §4.1 dynamic context, i.e. the
training-time rollout is the inference-time rollout."

Sequences are PA-scoped and start from ``BOS`` exactly as
``rollout_v3`` does, and the per-position ``count_state`` is the real
``_advance_count`` trajectory of the ground-truth continuation (verified
against the database's own per-pitch count; the agreement rate is recorded in
the audit JSON).

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

from src.analytics.pitchgpt import BOS_TOKEN, CONTEXT_DIM, PAD_TOKEN  # noqa: E402
from src.analytics.pitchgpt_v3 import (  # noqa: E402
    SPEC_BODY_SHA256,
    SPEC_FREEZE_SHA,
    SPEC_PATH,
    MaskStats,
    fields_from_token,
    load_v3_checkpoint,
    save_v3_checkpoint,
)
from src.analytics.pitchgpt_v3_data import (  # noqa: E402
    build_pa_dataset,
    context_indices_to_tensor,
    load_sequence_cache,
)
from src.analytics.pitchgpt_v3_rollout import pa_count_trajectory  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitchgpt_v3_stage_b")

DEFAULT_CACHE_DIR = Path(
    r"C:\Users\hunte\AppData\Local\Temp\claude"
    r"\C--Users-hunte-projects-baseball"
    r"\5112f9f4-f7ff-4db3-bd24-90acc5fbef27\scratchpad\pitchgpt_v3_cache"
)

HORIZON = 6
LR = 2e-4
BATCH = 32
GRAD_CLIP = 1.0
SEED = 42
#: §4.3 [FIXED] curriculum: (name, depth, epochs, eps_start, eps_end).
CURRICULUM = (
    ("B1", 2, 1, 0.00, 0.25),
    ("B2", 4, 1, 0.25, 0.50),
    ("B3", 6, 1, 0.50, 0.50),
)


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


def verify_count_trajectory(pa: dict, n_check: int = 20_000, seed: int = SEED) -> dict:
    """§4.1 / §8.3.1 guard on the *training* data path.

    The per-position ``count_state`` fed to Stage B must (a) not be constant
    within a PA whose count advances, and (b) equal the ``_advance_count``
    trajectory of the real outcome sequence.
    """
    rng = np.random.default_rng(seed)
    n_pa = len(pa["pa_len"])
    idx = rng.choice(n_pa, size=min(n_check, n_pa), replace=False)
    agree = total = advancing = varying = 0
    for i in idx:
        n = int(pa["pa_len"][i])
        if n < 2:
            continue
        outs = pa["pa_outc"][i, :n]
        cs = pa["pa_ctx_idx"][i, :n, 0].astype(int)
        start = (cs[0] // 3, cs[0] % 3)
        traj = pa_count_trajectory(outs, start=start)
        derived = np.array([min(b, 3) * 3 + min(s, 2) for b, s in traj])
        # Only positions after an *labelled* outcome are predictable.
        labelled = np.concatenate([[True], outs[:-1] >= 0])
        agree += int((derived[labelled] == cs[labelled]).sum())
        total += int(labelled.sum())
        if len(set(cs.tolist())) > 1:
            varying += 1
        advancing += 1
    return {
        "n_pas_checked": int(advancing),
        "count_state_positions_checked": int(total),
        "advance_count_agreement": float(agree / max(total, 1)),
        "pct_pas_with_varying_count_state": float(varying / max(advancing, 1)),
        "static_context_bug_present": bool(varying == 0),
    }


def make_batches(n: int, batch: int, rng: np.random.Generator):
    order = rng.permutation(n)
    for i in range(0, n, batch):
        yield order[i : i + batch]


def run_substage(
    model,
    pa: dict,
    *,
    depth: int,
    eps_start: float,
    eps_end: float,
    optim,
    device,
    batch: int,
    rng: np.random.Generator,
    gen: torch.Generator,
    label: str,
) -> dict:
    n_pa = len(pa["pa_len"])
    n_batches = (n_pa + batch - 1) // batch
    model.train()
    tot_loss = 0.0
    tot_n = 0
    n_substituted = 0
    n_rolled_positions = 0
    stats = MaskStats()
    t0 = time.perf_counter()

    tok_all = torch.from_numpy(pa["pa_tok"].astype(np.int64))
    ctx_all = torch.from_numpy(pa["pa_ctx_idx"].astype(np.int64))
    ump_all = torch.from_numpy(pa["pa_ump"].astype(np.float32))
    len_all = torch.from_numpy(pa["pa_len"].astype(np.int64))

    for bi, idx in enumerate(make_batches(n_pa, batch, rng)):
        eps = eps_start + (eps_end - eps_start) * (bi / max(n_batches - 1, 1))
        ii = torch.from_numpy(idx)
        tok = tok_all[ii].to(device)          # (B, H) real tokens, -1 = absent
        ctxi = ctx_all[ii].to(device)         # (B, H, 6)
        ump = ump_all[ii].to(device)          # (B,)
        lens = len_all[ii].to(device)         # (B,)
        B = tok.shape[0]
        L = int(lens.max().item())
        if L < 1:
            continue

        ctx_full = context_indices_to_tensor(
            ctxi, ump.unsqueeze(1).expand(B, HORIZON), CONTEXT_DIM,
        )  # (B, H, 35)

        # Input sequence starts at BOS; position j predicts real token j.
        cur_tokens = torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device)
        cur_context = ctx_full[:, :1, :]

        loss_sum = 0.0
        loss_terms = []
        for j in range(L):
            alive = (lens > j)
            hidden = model.forward_hidden(cur_tokens, cur_context)[:, -1, :]
            tgt = tok[:, j].clamp(min=0)
            t_s, z_s, v_s = fields_from_token(tgt)
            lt = model.type_logits(hidden)
            lz = model.zone_logits(hidden, t_s)
            lv = model.velo_logits(hidden, t_s, z_s)
            m = alive
            if m.any():
                loss_terms.append(
                    F.cross_entropy(lt[m], t_s[m], reduction="sum")
                    + F.cross_entropy(lz[m], z_s[m], reduction="sum")
                    + F.cross_entropy(lv[m], v_s[m], reduction="sum")
                )
                tot_n += int(m.sum())

            if j + 1 >= L:
                break
            # Next input token: model's own sample with probability eps, but
            # only for the first ``depth`` rolled steps (§4.3 curriculum).
            next_tok = tok[:, j].clone()
            if (j + 1) <= depth and eps > 0.0:
                with torch.no_grad():
                    sampled, _, _, _ = model.sample_tokens(
                        hidden.detach(), generator=gen, stats=stats,
                    )
                draw = torch.rand(B, device=device, generator=gen)
                use_sample = (draw < eps) & alive
                next_tok = torch.where(use_sample, sampled, next_tok)
                n_substituted += int(use_sample.sum())
                n_rolled_positions += int(alive.sum())
            next_tok = torch.where(
                alive, next_tok.clamp(min=0),
                torch.full_like(next_tok, PAD_TOKEN),
            )
            cur_tokens = torch.cat([cur_tokens, next_tok.unsqueeze(1)], dim=1)
            cur_context = torch.cat(
                [cur_context, ctx_full[:, j + 1 : j + 2, :]], dim=1,
            )

        if not loss_terms:
            continue
        total = torch.stack(loss_terms).sum()
        n_pos = int(torch.clamp(lens, max=L).sum().item())
        loss = total / max(n_pos, 1)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optim.step()
        tot_loss += float(total.item())

        if (bi + 1) % 2000 == 0:
            logger.info(
                "  [%s] batch %d/%d  eps=%.3f  NLL/pitch %.4f  (%.0fs)",
                label, bi + 1, n_batches, eps, tot_loss / max(tot_n, 1),
                time.perf_counter() - t0,
            )

    return {
        "substage": label,
        "depth": depth,
        "eps_start": eps_start,
        "eps_end": eps_end,
        "train_nll_per_pitch": tot_loss / max(tot_n, 1),
        "n_positions": tot_n,
        "n_scheduled_substitutions": n_substituted,
        "n_rollable_positions": n_rolled_positions,
        "substitution_rate": (
            n_substituted / n_rolled_positions if n_rolled_positions else 0.0
        ),
        "mask_events": stats.as_dict(),
        "sec": round(time.perf_counter() - t0, 1),
    }


@torch.no_grad()
def eval_pa_nll(model, pa: dict, device, batch: int = 256) -> dict:
    """Teacher-forced PA-scoped NLL (monitoring only, §4.3 Stage B monitoring)."""
    model.eval()
    n_pa = len(pa["pa_len"])
    tok_all = torch.from_numpy(pa["pa_tok"].astype(np.int64))
    ctx_all = torch.from_numpy(pa["pa_ctx_idx"].astype(np.int64))
    ump_all = torch.from_numpy(pa["pa_ump"].astype(np.float32))
    len_all = torch.from_numpy(pa["pa_len"].astype(np.int64))
    tot = 0.0
    n = 0
    for start in range(0, n_pa, batch):
        sl = slice(start, start + batch)
        tok = tok_all[sl].to(device)
        ctxi = ctx_all[sl].to(device)
        ump = ump_all[sl].to(device)
        lens = len_all[sl].to(device)
        B = tok.shape[0]
        L = int(lens.max().item())
        if L < 1:
            continue
        ctx_full = context_indices_to_tensor(
            ctxi, ump.unsqueeze(1).expand(B, HORIZON), CONTEXT_DIM,
        )
        inp = torch.cat(
            [
                torch.full((B, 1), BOS_TOKEN, dtype=torch.long, device=device),
                tok[:, : L - 1].clamp(min=0),
            ],
            dim=1,
        )
        pad = torch.arange(L, device=device).unsqueeze(0) >= lens.unsqueeze(1)
        inp = inp.masked_fill(pad, PAD_TOKEN)
        ctxs = ctx_full[:, :L, :].masked_fill(pad.unsqueeze(-1), 0.0)
        hidden = model.forward_hidden(inp, ctxs)
        tgt = tok[:, :L].clamp(min=0)
        t_s, z_s, v_s = fields_from_token(tgt)
        lt = model.type_logits(hidden)
        lz = model.zone_logits(hidden, t_s)
        lv = model.velo_logits(hidden, t_s, z_s)
        m = ~pad
        tot += float(
            F.cross_entropy(lt[m], t_s[m], reduction="sum")
            + F.cross_entropy(lz[m], z_s[m], reduction="sum")
            + F.cross_entropy(lv[m], v_s[m], reduction="sum")
        )
        n += int(m.sum())
    return {"pa_nll_per_pitch": tot / max(n, 1), "n_positions": n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument(
        "--stage-a-checkpoint", type=Path,
        default=_ROOT / "models" / "pitchgpt_v3_factorized_stage_a.pt",
    )
    ap.add_argument(
        "--out-checkpoint", type=Path,
        default=_ROOT / "models" / "pitchgpt_v3_factorized.pt",
    )
    ap.add_argument("--batch-size", type=int, default=BATCH)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", type=str, default="stage_b")
    ap.add_argument("--limit-pas", type=int, default=0, help="smoke only")
    ap.add_argument(
        "--resume-state", type=Path, default=None,
        help="Sub-stage resume file. Written after every sub-stage and reloaded "
             "on restart, so an infrastructure interruption costs at most one "
             "sub-stage. Not an artifact and not a protocol change: the "
             "curriculum, seeds and optimizer state carry through unchanged.",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck_a = load_v3_checkpoint(args.stage_a_checkpoint, device=device)
    model.to(device)
    logger.info(
        "loaded Stage A best epoch %s (2023-slice NLL %.5f)",
        ck_a["meta"]["best_epoch"], ck_a["meta"]["best_fit2023_nll_per_pitch"],
    )

    train_cache = load_sequence_cache(args.cache_dir / "train.npz")
    fit_cache = load_sequence_cache(args.cache_dir / "fit23.npz")
    pa_train = build_pa_dataset(train_cache, horizon=HORIZON)
    pa_fit = build_pa_dataset(fit_cache, horizon=HORIZON)
    logger.info(
        "PA cohorts: train %d PAs | fit23 %d PAs",
        pa_train["meta"]["n_pas"], pa_fit["meta"]["n_pas"],
    )

    ctx_check = verify_count_trajectory(pa_train)
    logger.info("§4.1 dynamic-context check: %s", json.dumps(ctx_check))
    if ctx_check["static_context_bug_present"]:
        raise RuntimeError(
            "§4.1 violated: no PA in the Stage B training data has a varying "
            "count_state — the static-context bug is back."
        )

    if args.limit_pas:
        n = int(args.limit_pas)
        pa_train = {
            **{k: v[:n] for k, v in pa_train.items() if k != "meta"},
            "meta": {**pa_train["meta"], "limited_to_pas": n},
        }

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device) if device.type == "cuda" else torch.Generator()
    gen.manual_seed(args.seed)

    history: list[dict] = []
    done: list[str] = []
    prior_sec = 0.0
    resume = args.resume_state
    if resume is not None and resume.exists():
        st = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(st["model_state_dict"])
        optim.load_state_dict(st["optim_state_dict"])
        rng.bit_generator.state = st["numpy_rng_state"]
        gen.set_state(st["torch_gen_state"].to(gen.get_state().device))
        history = st["history"]
        done = st["completed_substages"]
        prior_sec = float(st.get("elapsed_sec", 0.0))
        logger.info(
            "RESUMED after sub-stages %s (%.0fs of prior compute carried over)",
            done, prior_sec,
        )
    else:
        history = [{"stage": "A_baseline", **eval_pa_nll(model, pa_fit, device)}]
        logger.info("pre-Stage-B 2023 PA NLL/pitch %.5f", history[0]["pa_nll_per_pitch"])

    t_start = time.perf_counter()
    for name, depth, epochs, e0, e1 in CURRICULUM:
        if name in done:
            logger.info("%s already complete — skipping", name)
            continue
        for _ in range(epochs):
            rec = run_substage(
                model, pa_train, depth=depth, eps_start=e0, eps_end=e1,
                optim=optim, device=device, batch=args.batch_size, rng=rng,
                gen=gen, label=name,
            )
            rec.update(eval_pa_nll(model, pa_fit, device))
            history.append(rec)
            logger.info(
                "%s done: train NLL %.5f | 2023 PA NLL %.5f | subs %.1f%% | %.0fs",
                name, rec["train_nll_per_pitch"], rec["pa_nll_per_pitch"],
                100 * rec["substitution_rate"], rec["sec"],
            )
        done.append(name)
        if resume is not None:
            resume.parent.mkdir(parents=True, exist_ok=True)
            tmp = resume.with_suffix(".tmp")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optim.state_dict(),
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_gen_state": gen.get_state(),
                    "history": history,
                    "completed_substages": done,
                    "elapsed_sec": prior_sec + (time.perf_counter() - t_start),
                },
                tmp,
            )
            tmp.replace(resume)
            logger.info("resume state -> %s (after %s)", resume, name)

    wall = prior_sec + (time.perf_counter() - t_start)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _ROOT / "results" / "pitchgpt_v3" / f"train_{args.tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "stage": "B_curriculum_rollout_aware_finetune",
        "spec_path": SPEC_PATH,
        "spec_freeze_sha": SPEC_FREEZE_SHA,
        "spec_body_sha256": SPEC_BODY_SHA256,
        "git_sha": git_sha(),
        "stage_a_checkpoint": str(args.stage_a_checkpoint),
        "stage_a_meta": ck_a["meta"],
        "curriculum": [
            {"name": n, "depth": d, "epochs": e, "eps_start": a, "eps_end": b}
            for n, d, e, a, b in CURRICULUM
        ],
        "lr": args.lr,
        "batch_size": args.batch_size,
        "grad_clip": GRAD_CLIP,
        "seed": args.seed,
        "horizon": HORIZON,
        "dynamic_context_check": ctx_check,
        "train_pa_cohort": pa_train["meta"],
        "fit_pa_cohort": pa_fit["meta"],
        "history": history,
        "wall_clock_sec": round(wall, 1),
        "gpu_seconds": round(wall, 1),
        "duckdb_read_only": True,
        "two_pass_fallback_used": False,
        "resume_state_used": bool(
            args.resume_state is not None and args.resume_state.exists()
        ),
    }
    sha = save_v3_checkpoint(args.out_checkpoint, model, meta=meta)
    meta["checkpoint_sha256"] = sha
    meta["checkpoint_path"] = str(args.out_checkpoint)
    (out_dir / "audit.json").write_text(json.dumps(meta, indent=2))
    logger.info("checkpoint -> %s (sha256 %s)", args.out_checkpoint, sha[:16])
    logger.info("audit -> %s", out_dir / "audit.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
