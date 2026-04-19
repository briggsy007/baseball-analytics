"""
PitchGPT feature ablation — spec Ticket #6.

Trains four variants of PitchGPT on the canonical 2015-2022 / 2023 /
2024 date split and reports test-set perplexity for each.  The goal is
to show that every context feature pulls its weight — i.e. the full
model beats any ablation by ≥ 10%.

Variants
--------
- ``full``          : baseline PitchGPT (token + context + positional).
- ``tokens-only``   : zeros the context tensor at forward time.  The
                      context-projection weights still exist but receive
                      a zero input, so they contribute only a constant
                      bias.  This is the "no situational info" model.
- ``count-only``    : keeps only the (count_state, outs) context
                      dimensions; zeros the runner/hand/inning/score
                      dimensions.  Isolates the most obviously-useful
                      situational feature.
- ``identity-only`` : replaces the pitch-type composite token with a
                      pitcher-id token.  Model sees only *who* is
                      throwing, not *what* they're throwing.  Tests
                      whether perplexity is driven by pitcher identity
                      memorisation rather than true sequence modelling.

Hyperparameters are kept identical to ``train_pitchgpt_baselines.py``:
seed 42, 5 epochs, batch 32, lr 1e-3, 1000 train games / 300 val games /
300 test games.

Outputs
-------
``<output_dir>/pitchgpt_ablation_metrics.json`` with the four variants'
test perplexities, param counts, and the verdict against the ≥ 10%
threshold.  Optionally also saves the full-variant checkpoint at
``<output_dir>/pitchgpt_full.pt`` for downstream calibration.
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
from torch.utils.data import DataLoader, Dataset

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt import (  # noqa: E402
    CONTEXT_DIM,
    NUM_COUNT_STATES,
    NUM_OUTS,
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
logger = logging.getLogger("pitchgpt_ablation")


DEFAULT_BATCH = 32
DEFAULT_LR = 1e-3
GRAD_CLIP = 1.0
TRAIN_RANGE = (2015, 2022)
VAL_RANGE = (2023, 2023)
TEST_RANGE = (2024, 2024)

# Ablation threshold: full model must beat ANY ablation by ≥ 10%.
THRESH_ABLATION_PCT = 10.0

# Context offsets — mirrors PitchTokenizer.context_to_tensor layout:
#   [count(12), outs(3), runner(8), hand(2), inning(4), score(5)]
COUNT_END = NUM_COUNT_STATES
OUTS_END = COUNT_END + NUM_OUTS  # = 15


# ═════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ═════════════════════════════════════════════════════════════════════════════


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═════════════════════════════════════════════════════════════════════════════
# Context maskers — applied per-batch in the training loop
# ═════════════════════════════════════════════════════════════════════════════


def mask_context(ctx: torch.Tensor, mode: str) -> torch.Tensor:
    """Return a context tensor zeroed-out according to ``mode``.

    ``mode``:
      - ``full``        : no-op.
      - ``tokens_only`` : zero every context dimension.
      - ``count_only``  : zero everything except count + outs.
      - ``identity_only``: no-op (identity variant uses tokens instead;
                          full context passes through, although this
                          variant rarely benefits from it since the
                          vocabulary has collapsed).

    Works for any tensor shape ending in ``CONTEXT_DIM``.
    """
    if mode == "full":
        return ctx
    if mode == "tokens_only":
        return torch.zeros_like(ctx)
    if mode == "count_only":
        out = torch.zeros_like(ctx)
        # Keep [0, OUTS_END) — count_state + outs.
        out[..., :OUTS_END] = ctx[..., :OUTS_END]
        return out
    if mode == "identity_only":
        return ctx
    raise ValueError(f"unknown ablation mode: {mode!r}")


# ═════════════════════════════════════════════════════════════════════════════
# Identity-only dataset wrapper
# ═════════════════════════════════════════════════════════════════════════════


class IdentityTokenDataset(Dataset):
    """Wraps a ``PitchSequenceDataset`` but replaces the pitch-type
    tokens with pitcher-id tokens.

    Each sequence in the underlying dataset is a single pitcher's game,
    so the "token sequence" collapses to ``[pid, pid, pid, ...]`` — a
    degenerate signal.  This is exactly the point: the model has nothing
    to predict from except the pitcher identity, so its perplexity
    measures *only* the per-pitcher marginal distribution.

    A fixed ``pid_to_idx`` mapping is built from the first dataset
    passed in (train); val / test datasets use the same mapping with
    unseen pitchers falling back to an ``UNK`` token.
    """

    UNK_IDX = 0  # reserved slot for unknown pitcher

    def __init__(
        self,
        base_dataset: PitchSequenceDataset,
        pid_to_idx: dict[int, int] | None = None,
    ) -> None:
        self.base = base_dataset
        self.max_seq_len = base_dataset.max_seq_len

        # Build or inherit the pitcher-id vocabulary.
        if pid_to_idx is None:
            # Reserve idx 0 for UNK; assign others in sorted order so
            # the map is deterministic given the pitcher_ids set.
            pitcher_ids_sorted = sorted(base_dataset.pitcher_ids)
            self.pid_to_idx = {pid: i + 1 for i, pid in enumerate(pitcher_ids_sorted)}
        else:
            self.pid_to_idx = dict(pid_to_idx)

        # Vocab size = known pitchers + 1 UNK + 1 PAD.
        # We use the PitchGPT PAD_TOKEN-style convention: PAD is the
        # last index.  Keep it consistent with the existing collate_fn.
        self.n_known = len(self.pid_to_idx)
        self.vocab_size = self.n_known + 1  # +1 for UNK (idx 0)
        self.pad_token = self.vocab_size   # extra slot for PAD

        # Pre-compute token sequences per base-sequence.
        # The base dataset does NOT expose pitcher_id per sequence, so
        # we rebuild the sequences by re-iterating the underlying data.
        # Instead, we monkey-lookup via the base dataset's loading
        # logic: each sequence is a single (game_pk, pitcher_id) group,
        # and we stored pitcher_ids on the base dataset.  Unfortunately
        # base.sequences don't retain pid individually — so we assign
        # per-sequence pitcher ids by re-grouping.  The simplest robust
        # approach: match sequence index order to the iteration order
        # the base dataset used.
        self._build_token_sequences()

    def _build_token_sequences(self) -> None:
        """Pre-build identity-token sequences aligned with
        ``base_dataset.sequences``.

        Strategy: use the fact that each base sequence came from one
        (game_pk, pitcher_id) group; its length in tokens is equal to
        the number of pitches - 1.  Since we've lost the per-sequence
        pitcher_id on the base dataset, we recompute from raw data.
        BUT that would require a DB round-trip — instead, we take the
        pragmatic shortcut and tag each sequence with a *sampled*
        pitcher from the base dataset's pitcher_ids set based on
        sequence position, which preserves the marginal distribution
        but is not per-pitcher exact.

        To do this correctly we augment the base PitchSequenceDataset
        upstream — see ``_monkey_patch_pid_tracking`` below, called
        from ``main()`` to attach a ``.sequence_pids`` list to the
        base dataset.
        """
        if not hasattr(self.base, "sequence_pids"):
            raise RuntimeError(
                "IdentityTokenDataset requires the base dataset to have a "
                "`sequence_pids` attribute (list[int], one pitcher_id per "
                "sequence).  Call _attach_sequence_pids(ds) on the base "
                "dataset before constructing IdentityTokenDataset."
            )

        pids = self.base.sequence_pids
        if len(pids) != len(self.base.sequences):
            raise RuntimeError(
                f"sequence_pids length {len(pids)} != "
                f"n_sequences {len(self.base.sequences)}"
            )

        self._items: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for (inp_tok, ctx_tensor, tgt_tok), pid in zip(self.base.sequences, pids):
            idx = self.pid_to_idx.get(int(pid), self.UNK_IDX)
            seq_len = inp_tok.size(0)
            # Input tokens = pitcher id repeated (same length as original input).
            new_inp = torch.full((seq_len,), idx, dtype=torch.long)
            # Target tokens = same pitcher id repeated (trivial next-token task:
            # the true target is always "the same pitcher").
            new_tgt = torch.full((seq_len,), idx, dtype=torch.long)
            self._items.append((new_inp, ctx_tensor, new_tgt))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, i: int):
        return self._items[i]


def _identity_collate_factory(pad_token: int):
    """Build a collate_fn that pads using a custom PAD_TOKEN (since the
    identity-variant has its own vocabulary).
    """
    def collate(batch):
        max_len = max(item[0].size(0) for item in batch)
        tokens_batch, ctx_batch, target_batch = [], [], []

        for tokens, ctx, target in batch:
            seq_len = tokens.size(0)
            pad_len = max_len - seq_len

            tokens_padded = torch.cat(
                [tokens, torch.full((pad_len,), pad_token, dtype=torch.long)]
            )
            target_padded = torch.cat(
                [target, torch.full((pad_len,), pad_token, dtype=torch.long)]
            )
            if pad_len > 0:
                ctx_padded = torch.cat(
                    [ctx, torch.zeros(pad_len, CONTEXT_DIM)]
                )
            else:
                ctx_padded = ctx

            tokens_batch.append(tokens_padded)
            ctx_batch.append(ctx_padded)
            target_batch.append(target_padded)

        return (
            torch.stack(tokens_batch),
            torch.stack(ctx_batch),
            torch.stack(target_batch),
        )
    return collate


def _attach_sequence_pids(ds: PitchSequenceDataset, conn, seasons, max_games) -> None:
    """Re-query the DB to align a pitcher_id to every sequence in ``ds``.

    The base ``PitchSequenceDataset`` builds one sequence per
    (game_pk, pitcher_id) group, iterated in ``groupby(..., sort=False)``
    order.  We mimic that same iteration order here so indices align.

    Mutates ``ds`` by setting ``ds.sequence_pids: list[int]``.
    """
    season_filter = ""
    if seasons:
        s_str = ", ".join(str(int(s)) for s in seasons)
        season_filter = f"AND EXTRACT(YEAR FROM game_date) IN ({s_str})"

    game_filter = ""
    if max_games is not None:
        # NOTE: must match the sample used by the base dataset.  The
        # base dataset uses ``USING SAMPLE N ROWS`` which is a random
        # sample — in practice we capture the actual game_pks the
        # dataset ended up with via ds.game_pks and filter to those.
        if ds.game_pks:
            gpks = ", ".join(str(int(g)) for g in ds.game_pks)
            game_filter = f"AND game_pk IN ({gpks})"

    query = f"""
        SELECT DISTINCT game_pk, pitcher_id
        FROM pitches
        WHERE pitch_type IS NOT NULL
          {season_filter}
          {game_filter}
    """
    rows = conn.execute(query).fetchdf()
    pid_for_game = {}
    # One (game, pitcher) → one sequence, but the dataset iterates in
    # the order of the ``groupby`` on the ORDER BY output.  That order
    # is (game_pk asc, then pitcher_id by first appearance).  We can't
    # reconstruct "first appearance" without re-running the same
    # ordered query — so just do that.
    order_query = f"""
        SELECT game_pk, pitcher_id,
               MIN(at_bat_number * 1000 + pitch_number) AS _sort_key
        FROM pitches
        WHERE pitch_type IS NOT NULL
          {season_filter}
          {game_filter}
        GROUP BY game_pk, pitcher_id
        ORDER BY game_pk, _sort_key
    """
    ordered = conn.execute(order_query).fetchdf()

    # Build the expected (game_pk, pitcher_id) sequence-in order list
    ordered_pairs = [
        (int(r["game_pk"]), int(r["pitcher_id"])) for _, r in ordered.iterrows()
    ]

    # Filter to pairs that actually produced a sequence (≥ 2 pitches).
    # We don't know which pairs were dropped without replaying the
    # group_df length check — approximate by keeping pairs that actually
    # ended up in ds.game_pks and ds.pitcher_ids, and trust the 2-pitch
    # filter rarely drops (<1% of groups).  For cleanliness, we trust
    # the length equality check at the end.
    kept = [
        pid for (gpk, pid) in ordered_pairs
        if gpk in ds.game_pks and pid in ds.pitcher_ids
    ]

    if len(kept) != len(ds.sequences):
        # Fall back: if the lengths disagree, we pad/truncate.  This is
        # imperfect but we log it loudly.
        logger.warning(
            "sequence-pid alignment off: kept=%d sequences=%d — some short "
            "(<2 pitch) groups likely dropped by the dataset.  Truncating.",
            len(kept), len(ds.sequences),
        )
        # Truncate to min length.
        n = min(len(kept), len(ds.sequences))
        kept = kept[:n]
        ds.sequences = ds.sequences[:n]

    ds.sequence_pids = kept


# ═════════════════════════════════════════════════════════════════════════════
# Train / eval a single variant
# ═════════════════════════════════════════════════════════════════════════════


def _iter_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    device: torch.device,
    output_vocab: int,
    pad_token: int,
    train: bool,
    ablation_mode: str,
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

            ctx_masked = mask_context(ctx, ablation_mode)
            logits = model(tokens, ctx_masked)
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

            n_tok = (target != pad_token).sum().item()
            total_loss += loss.item() * n_tok
            total_tok += n_tok

    return total_loss / max(total_tok, 1)


def _train_variant(
    name: str,
    ablation_mode: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    vocab_size: int,
    output_vocab: int,
    pad_token: int,
    seed: int,
) -> tuple[dict, PitchGPTModel]:
    """Train one variant.  Returns (metrics_dict, trained_model)."""
    _set_seed(seed)
    model = PitchGPTModel(vocab_size=vocab_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # Use the correct PAD index for each variant (identity-only has its own).
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token)

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    history: list[dict] = []

    t_start = time.perf_counter()
    logger.info("[%s] mode=%s — %d epochs", name, ablation_mode, epochs)
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        train_loss = _iter_epoch(
            model, train_loader, optimizer, criterion, device,
            output_vocab=output_vocab, pad_token=pad_token,
            train=True, ablation_mode=ablation_mode,
        )
        val_loss = _iter_epoch(
            model, val_loader, None, criterion, device,
            output_vocab=output_vocab, pad_token=pad_token,
            train=False, ablation_mode=ablation_mode,
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
        output_vocab=output_vocab, pad_token=pad_token,
        train=False, ablation_mode=ablation_mode,
    )
    test_ppl = math.exp(min(test_loss, 20))
    wallclock = round(time.perf_counter() - t_start, 1)
    logger.info(
        "[%s] BEST epoch %d  test_loss=%.4f  test_ppl=%.3f  (%.1fs total)",
        name, best_epoch, test_loss, test_ppl, wallclock,
    )

    params = int(sum(p.numel() for p in model.parameters()))
    return {
        "mode": ablation_mode,
        "params": params,
        "train_loss_final": history[-1]["train_loss"],
        "val_loss_final": history[-1]["val_loss"],
        "test_loss": round(test_loss, 4),
        "test_perplexity": round(test_ppl, 3),
        "epoch_best": best_epoch,
        "best_val_loss": round(best_val_loss, 4),
        "wall_clock_sec": wallclock,
        "history": history,
    }, model


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
        "--output-dir", type=Path, default=Path("results/run_5epoch"),
    )
    parser.add_argument("--db-path", type=str, default=None)
    parser.add_argument(
        "--skip-identity", action="store_true",
        help="Skip the identity-only variant (for fast iteration).",
    )
    parser.add_argument(
        "--save-full-checkpoint", action="store_true", default=True,
        help="Save the full-variant checkpoint for downstream calibration.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)
    device = _get_device()
    logger.info("device=%s  seed=%d", device, args.seed)

    # ── Load datasets once, reuse across variants ────────────────────────
    conn = get_connection(args.db_path, read_only=True)
    try:
        logger.info(
            "loading datasets (train=%s val=%s test=%s)",
            TRAIN_RANGE, VAL_RANGE, TEST_RANGE,
        )
        t_ds0 = time.perf_counter()

        # Pitcher-disjoint split (Ticket #1 hardening).  Enumerate every
        # pitcher who appears in the *full* train season range first;
        # then exclude them from val and test at SQL load time.  This
        # guarantees zero pitcher overlap between train and val/test
        # regardless of how the per-split USING SAMPLE happens to land.
        # See PitchSequenceDataset.fetch_pitcher_ids_for_seasons.
        train_seasons_full = list(range(TRAIN_RANGE[0], TRAIN_RANGE[1] + 1))
        logger.info(
            "fetching train-cohort pitcher_ids for seasons %s (for pitcher-disjoint split)",
            train_seasons_full,
        )
        train_pitcher_ids = PitchSequenceDataset.fetch_pitcher_ids_for_seasons(
            conn, train_seasons_full,
        )
        logger.info(
            "train cohort has %d distinct pitchers — these are excluded from val/test",
            len(train_pitcher_ids),
        )

        # Seed BEFORE each dataset so its USING SAMPLE is reproducible.
        _set_seed(args.seed)
        train_ds = PitchSequenceDataset(
            conn, split_mode="train",
            train_range=TRAIN_RANGE, val_range=VAL_RANGE, test_range=TEST_RANGE,
            max_games_per_split=args.max_train_games,
        )
        _set_seed(args.seed + 1)
        val_ds = PitchSequenceDataset(
            conn, split_mode="val",
            train_range=TRAIN_RANGE, val_range=VAL_RANGE, test_range=TEST_RANGE,
            max_games_per_split=args.max_val_games,
            exclude_pitcher_ids=train_pitcher_ids,
        )
        _set_seed(args.seed + 2)
        test_ds = PitchSequenceDataset(
            conn, split_mode="test",
            train_range=TRAIN_RANGE, val_range=VAL_RANGE, test_range=TEST_RANGE,
            max_games_per_split=args.max_test_games,
            exclude_pitcher_ids=train_pitcher_ids,
        )
        logger.info(
            "datasets built in %.1fs (train=%d val=%d test=%d)",
            time.perf_counter() - t_ds0,
            len(train_ds), len(val_ds), len(test_ds),
        )

        audit = audit_no_game_overlap(train_ds, val_ds, test_ds)
        if audit["shared_game_pks"] != 0:
            logger.error("LEAKAGE DETECTED (game_pk overlap) — refusing to train.")
            return 2
        if audit["shared_pitcher_ids_train_test"] != 0 or audit["shared_pitcher_ids_train_val"] != 0:
            logger.error(
                "LEAKAGE DETECTED (pitcher overlap): train_test=%d train_val=%d — refusing to train.",
                audit["shared_pitcher_ids_train_test"],
                audit["shared_pitcher_ids_train_val"],
            )
            return 2

        # Attach sequence_pids for the identity-only variant.
        if not args.skip_identity:
            try:
                seasons_train = list(range(TRAIN_RANGE[0], TRAIN_RANGE[1] + 1))
                seasons_val = list(range(VAL_RANGE[0], VAL_RANGE[1] + 1))
                seasons_test = list(range(TEST_RANGE[0], TEST_RANGE[1] + 1))
                _attach_sequence_pids(train_ds, conn, seasons_train, args.max_train_games)
                _attach_sequence_pids(val_ds, conn, seasons_val, args.max_val_games)
                _attach_sequence_pids(test_ds, conn, seasons_test, args.max_test_games)
            except Exception as e:
                logger.warning(
                    "failed to attach sequence_pids — identity-only will be skipped: %s",
                    e,
                )
                args.skip_identity = True
    finally:
        conn.close()

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        logger.error("Empty datasets — aborting.")
        return 1

    # ── Standard (full / tokens_only / count_only) loaders ──────────────
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_fn,
    )

    variants: dict[str, dict] = {}

    # ── 1. full ──────────────────────────────────────────────────────────
    t_overall = time.perf_counter()
    full_metrics, full_model = _train_variant(
        name="full", ablation_mode="full",
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device, epochs=args.epochs, lr=args.lr,
        vocab_size=TOTAL_VOCAB, output_vocab=VOCAB_SIZE, pad_token=PAD_TOKEN,
        seed=args.seed,
    )
    variants["full"] = full_metrics

    if args.save_full_checkpoint:
        ckpt_path = args.output_dir / "pitchgpt_full.pt"
        torch.save(
            {
                "model_state_dict": full_model.state_dict(),
                "config": {
                    "d_model": 128,
                    "nhead": 4,
                    "num_layers": 4,
                    "max_seq_len": 256,
                    "vocab_size": TOTAL_VOCAB,
                },
                "version": "ablation_full",
                "variant": "full",
                "seed": args.seed,
                "epochs": args.epochs,
                "test_perplexity": full_metrics["test_perplexity"],
            },
            ckpt_path,
        )
        logger.info("saved full-variant checkpoint to %s", ckpt_path)

    # Free the full model before training the next (keeps RAM steady).
    del full_model

    # ── 2. tokens_only ───────────────────────────────────────────────────
    to_metrics, _ = _train_variant(
        name="tokens-only", ablation_mode="tokens_only",
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device, epochs=args.epochs, lr=args.lr,
        vocab_size=TOTAL_VOCAB, output_vocab=VOCAB_SIZE, pad_token=PAD_TOKEN,
        seed=args.seed,
    )
    variants["tokens_only"] = to_metrics

    # ── 3. count_only ────────────────────────────────────────────────────
    co_metrics, _ = _train_variant(
        name="count-only", ablation_mode="count_only",
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        device=device, epochs=args.epochs, lr=args.lr,
        vocab_size=TOTAL_VOCAB, output_vocab=VOCAB_SIZE, pad_token=PAD_TOKEN,
        seed=args.seed,
    )
    variants["count_only"] = co_metrics

    # ── 4. identity_only ─────────────────────────────────────────────────
    if not args.skip_identity:
        try:
            id_train = IdentityTokenDataset(train_ds)
            id_val = IdentityTokenDataset(val_ds, pid_to_idx=id_train.pid_to_idx)
            id_test = IdentityTokenDataset(test_ds, pid_to_idx=id_train.pid_to_idx)

            id_pad = id_train.pad_token
            id_vocab = id_train.vocab_size
            id_collate = _identity_collate_factory(id_pad)

            id_train_loader = DataLoader(
                id_train, batch_size=args.batch_size, shuffle=True,
                collate_fn=id_collate,
            )
            id_val_loader = DataLoader(
                id_val, batch_size=args.batch_size, shuffle=False,
                collate_fn=id_collate,
            )
            id_test_loader = DataLoader(
                id_test, batch_size=args.batch_size, shuffle=False,
                collate_fn=id_collate,
            )

            # Build a PitchGPTModel with a pitcher-id vocab.  The
            # output head predicts over the *same* vocabulary we're
            # tokenising with — i.e. id_vocab — so set both
            # vocab_size=id_vocab+1 (+1 for PAD) and route output
            # through a custom linear head by subclassing on the fly.
            # Simplest approach: build PitchGPTModel with
            # vocab_size = id_pad + 1, and compare logits over id_vocab.
            id_total_vocab = id_pad + 1

            # For identity-only, the PitchGPTModel has an output_head
            # that projects to VOCAB_SIZE by default — we need it to
            # project to id_vocab.  Wrap with a custom module.
            id_model = _IdentityPitchGPT(
                vocab_size=id_total_vocab,
                output_vocab=id_vocab,
                pad_token=id_pad,
            ).to(device)

            _set_seed(args.seed)
            # Re-init weights deterministically.
            for p in id_model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

            # Inline training — can't reuse _train_variant because we
            # need a custom model class.  Same loop though.
            optimizer = torch.optim.AdamW(id_model.parameters(), lr=args.lr)
            criterion = nn.CrossEntropyLoss(ignore_index=id_pad)

            best_val = float("inf")
            best_state = None
            best_epoch = -1
            history: list[dict] = []
            t_id0 = time.perf_counter()
            logger.info("[identity-only] mode=identity_only — %d epochs", args.epochs)
            for epoch in range(1, args.epochs + 1):
                t0 = time.perf_counter()
                id_model.train()
                total_l, total_n = 0.0, 0
                for tok, ctx, tgt in id_train_loader:
                    tok = tok.to(device); ctx = ctx.to(device); tgt = tgt.to(device)
                    logits = id_model(tok, ctx)  # (B,S,id_vocab)
                    loss = criterion(logits.reshape(-1, id_vocab), tgt.reshape(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(id_model.parameters(), GRAD_CLIP)
                    optimizer.step()
                    n = (tgt != id_pad).sum().item()
                    total_l += loss.item() * n
                    total_n += n
                tr_loss = total_l / max(total_n, 1)

                id_model.eval()
                v_l, v_n = 0.0, 0
                with torch.no_grad():
                    for tok, ctx, tgt in id_val_loader:
                        tok = tok.to(device); ctx = ctx.to(device); tgt = tgt.to(device)
                        logits = id_model(tok, ctx)
                        loss = criterion(logits.reshape(-1, id_vocab), tgt.reshape(-1))
                        n = (tgt != id_pad).sum().item()
                        v_l += loss.item() * n
                        v_n += n
                v_loss = v_l / max(v_n, 1)
                dt = time.perf_counter() - t0
                entry = {
                    "epoch": epoch,
                    "train_loss": round(tr_loss, 4),
                    "val_loss": round(v_loss, 4),
                    "train_perplexity": round(math.exp(min(tr_loss, 20)), 3),
                    "val_perplexity": round(math.exp(min(v_loss, 20)), 3),
                    "wall_clock_sec": round(dt, 1),
                }
                history.append(entry)
                logger.info(
                    "[identity-only] epoch %d/%d  train=%.4f (ppl %.2f)  val=%.4f (ppl %.2f)  %.1fs",
                    epoch, args.epochs, tr_loss, entry["train_perplexity"],
                    v_loss, entry["val_perplexity"], dt,
                )
                if v_loss < best_val:
                    best_val = v_loss
                    best_epoch = epoch
                    best_state = copy.deepcopy(id_model.state_dict())

            if best_state is not None:
                id_model.load_state_dict(best_state)

            id_model.eval()
            t_l, t_n = 0.0, 0
            with torch.no_grad():
                for tok, ctx, tgt in id_test_loader:
                    tok = tok.to(device); ctx = ctx.to(device); tgt = tgt.to(device)
                    logits = id_model(tok, ctx)
                    loss = criterion(logits.reshape(-1, id_vocab), tgt.reshape(-1))
                    n = (tgt != id_pad).sum().item()
                    t_l += loss.item() * n
                    t_n += n
            test_loss = t_l / max(t_n, 1)
            test_ppl = math.exp(min(test_loss, 20))
            id_wall = round(time.perf_counter() - t_id0, 1)
            logger.info(
                "[identity-only] BEST epoch %d  test_loss=%.4f  test_ppl=%.3f  (%.1fs total)",
                best_epoch, test_loss, test_ppl, id_wall,
            )

            variants["identity_only"] = {
                "mode": "identity_only",
                "params": int(sum(p.numel() for p in id_model.parameters())),
                "train_loss_final": history[-1]["train_loss"],
                "val_loss_final": history[-1]["val_loss"],
                "test_loss": round(test_loss, 4),
                "test_perplexity": round(test_ppl, 3),
                "epoch_best": best_epoch,
                "best_val_loss": round(best_val, 4),
                "wall_clock_sec": id_wall,
                "history": history,
                "vocab_size": id_vocab,
                "n_pitchers": id_train.n_known,
            }
            del id_model
        except Exception as e:
            logger.exception("identity-only variant failed: %s", e)
            variants["identity_only"] = {"error": str(e)}

    total_wall = round(time.perf_counter() - t_overall, 1)

    # ── Verdicts ─────────────────────────────────────────────────────────
    full_ppl = variants["full"]["test_perplexity"]
    ablations = {k: v for k, v in variants.items() if k != "full" and "test_perplexity" in v}
    drops = {}
    for name, m in ablations.items():
        ppl = m["test_perplexity"]
        # Improvement of full over this ablation (positive = full is better).
        pct = 100.0 * (ppl - full_ppl) / ppl if ppl > 0 else 0.0
        drops[name] = round(pct, 2)

    # "Best" ablation = the one closest to full (smallest drop).
    if drops:
        best_abl = min(drops, key=lambda k: drops[k])
        best_abl_pct = drops[best_abl]
        passed = best_abl_pct >= THRESH_ABLATION_PCT
        verdict = (
            f"{'PASS' if passed else 'FAIL'} — full model beats best ablation "
            f"({best_abl}) by {best_abl_pct:.1f}% "
            f"(spec requires ≥{THRESH_ABLATION_PCT:.0f}%)."
        )
    else:
        best_abl = None
        best_abl_pct = 0.0
        passed = False
        verdict = "FAIL — no ablation variants completed successfully."

    out = {
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "train_range": f"{TRAIN_RANGE[0]}-{TRAIN_RANGE[1]}",
        "val_range": f"{VAL_RANGE[0]}",
        "test_range": f"{TEST_RANGE[0]}",
        "n_train_sequences": len(train_ds),
        "n_val_sequences": len(val_ds),
        "n_test_sequences": len(test_ds),
        "leakage_audit": audit,
        **variants,
        "drops_vs_full_pct": drops,
        "spec_threshold_full_vs_any_ablation_pct": THRESH_ABLATION_PCT,
        "best_ablation": best_abl,
        "best_ablation_gap_pct": best_abl_pct,
        "pass": bool(passed),
        "verdict": verdict,
        "wall_clock_total_sec": total_wall,
    }
    out_path = args.output_dir / "pitchgpt_ablation_metrics.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("wrote ablation metrics to %s", out_path)

    # Console summary.
    logger.info("=" * 70)
    logger.info("Ablation summary:")
    for name, m in variants.items():
        if "test_perplexity" in m:
            drop = drops.get(name, 0.0) if name != "full" else 0.0
            logger.info(
                "  %-14s test_ppl=%8.3f  params=%8d  drop_vs_full=%+5.2f%%",
                name, m["test_perplexity"], m.get("params", 0), drop,
            )
    logger.info("-" * 70)
    logger.info(verdict)
    logger.info("=" * 70)
    return 0


# ═════════════════════════════════════════════════════════════════════════════
# Custom model for identity-only (needs configurable output vocab)
# ═════════════════════════════════════════════════════════════════════════════


class _IdentityPitchGPT(nn.Module):
    """Decoder-only transformer with a configurable pitcher-id vocab.

    Mirrors the PitchGPTModel architecture (same hidden dim, same
    number of layers/heads, same context projection) so the parameter
    count / inductive bias is comparable sans the token-embedding +
    output-head dimensions, which depend on the vocab size.
    """

    def __init__(
        self,
        vocab_size: int,
        output_vocab: int,
        pad_token: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self._pad_token = pad_token
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(
            vocab_size, d_model, padding_idx=pad_token,
        )
        self.context_proj = nn.Linear(CONTEXT_DIM, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, output_vocab)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, S = tokens.shape
        device = tokens.device
        tok_emb = self.token_embedding(tokens)
        ctx_emb = self.context_proj(context)
        pos_ids = torch.arange(S, device=device).unsqueeze(0).clamp(
            max=self.max_seq_len - 1,
        )
        pos_emb = self.pos_embedding(pos_ids)
        x = tok_emb + ctx_emb + pos_emb
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=device)
        padding_mask = (tokens == self._pad_token)
        x = self.transformer(x, mask=mask, src_key_padding_mask=padding_mask)
        return self.output_head(x)


if __name__ == "__main__":
    sys.exit(main())
