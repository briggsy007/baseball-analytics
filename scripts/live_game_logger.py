#!/usr/bin/env python
"""
Standalone live pitch-by-pitch logger (Plan B for the live-game case study).

Polls MLB StatsAPI via :class:`LiveGameTracker`, runs PitchGPT on every new
pitch (top-1 type/prob, top-3 distribution, brier vs actual, log-loss vs
actual), and appends one row per pitch to BOTH ``pitch_by_pitch.jsonl``
(line-atomic, preferred for resumability) and ``pitch_by_pitch.csv`` (the
spec format from EXECUTION_PLAN.md §4 Phase 1).

This is the belt-and-suspenders ledger for the case study — independent of
the Streamlit dashboard, so Phase 2 has data even if the dashboard crashes.

Usage
-----
    python scripts/live_game_logger.py \
        --game-pk 823475 \
        --output-dir results/live_game/2026-04-19_PHI_vs_ATL/ \
        [--poll-interval 12]

Hard rules:
    * Does NOT modify ``src/ingest/live_feed.py`` or ``src/analytics/pitchgpt.py``
    * Does NOT touch DuckDB
    * Resumable: on restart, reads existing JSONL and skips already-logged
      ``(at_bat_number, pitch_number)`` keys
    * Errors per-pitch are caught and written to ``errors.jsonl`` instead of
      crashing the logger
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import re
import signal
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ── Project root on sys.path ────────────────────────────────────────────────

ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Local project imports — read-only (do NOT modify these modules)
from src.ingest.live_feed import (  # noqa: E402
    LiveGameTracker,
    fetch_live_feed,
    parse_game_state,
)
from src.analytics.pitchgpt import (  # noqa: E402
    NUM_PITCH_TYPES,
    NUM_VELO_BUCKETS,
    NUM_ZONES,
    PITCH_TYPE_MAP,
    PitchTokenizer,
    VOCAB_SIZE,
    _load_model,
)

# Lazy torch import — keep top of file fast for --help
import numpy as np  # noqa: E402
import torch  # noqa: E402

# ── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("live_game_logger")

# Force UTF-8 stdout on Windows (cp1252 default trips on em-dashes etc.)
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True,
    )

# ── ANSI helpers (degrade gracefully) ──────────────────────────────────────

_BOLD = "\033[1m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_GREEN = "\033[92m"
_RESET = "\033[0m"


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


# ── CSV column spec (EXECUTION_PLAN.md §4 Phase 1) ─────────────────────────

CSV_COLUMNS: list[str] = [
    "timestamp_utc",
    "inning",
    "half",
    "batter_id",
    "batter_name",
    "pitcher_id",
    "pitcher_name",
    "count_balls",
    "count_strikes",
    "runners_on",
    "outs",
    "pitchgpt_top1_type",
    "pitchgpt_top1_prob",
    "pitchgpt_top3_json",
    "actual_pitch_type",
    "actual_zone",
    "actual_velo",
    "correct_top1",
    "brier",
    "log_loss",
]

# Pitches that share the same key won't be re-logged
PITCH_KEY_FIELDS = ("at_bat_number", "pitch_number")

# Default poll interval (StatsAPI processing window is 21-80s per dry-run)
DEFAULT_POLL_INTERVAL = 12

# Fail-soft NLL cap — log_loss is bounded for sanity in the CSV
_MAX_NLL = 20.0


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


_WS_RE = re.compile(r"\s+")


def _clean_name(name: Any) -> str:
    """Strip + collapse internal whitespace.

    Defensive against the StatsAPI quirk noted in the dry-run report
    where some names arrive with double spaces (e.g. ``"Hao-Yu  Lee"``).
    """
    if name is None:
        return ""
    return _WS_RE.sub(" ", str(name)).strip()


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with seconds precision."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _idx_to_pitch_type(idx: int) -> str:
    """Inverse lookup against ``PITCH_TYPE_MAP``; ``"UN"`` for unknown."""
    inv = {v: k for k, v in PITCH_TYPE_MAP.items()}
    return inv.get(idx, "UN")


def _runners_repr(on_1b: bool, on_2b: bool, on_3b: bool) -> str:
    """Compact runners string e.g. ``"1B,3B"`` or ``"empty"``."""
    bases = []
    if on_1b:
        bases.append("1B")
    if on_2b:
        bases.append("2B")
    if on_3b:
        bases.append("3B")
    return ",".join(bases) if bases else "empty"


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


# ═════════════════════════════════════════════════════════════════════════════
# Pre-pitch context extraction from a feed
# ═════════════════════════════════════════════════════════════════════════════


def _extract_pre_pitch_state(feed: dict, pitch: dict) -> dict[str, Any]:
    """Pull pre-pitch state for a pitch event from the GUMBO feed.

    Returns a dict with batter_name, pitcher_name, batter_hand, runners,
    score_diff, inning_topbot — everything PitchGPT's ``encode_context``
    needs that isn't already on the parsed pitch dict.

    Missing fields fall back to safe defaults (empty string / False / 0)
    per the dry-run finding that runners and score_diff are non-fatal
    when zeroed.
    """
    plays = feed.get("liveData", {}).get("plays", {})
    all_plays = plays.get("allPlays", [])

    at_bat_idx = pitch.get("at_bat_number")
    play: dict = {}
    for p in all_plays:
        if p.get("about", {}).get("atBatIndex") == at_bat_idx:
            play = p
            break

    matchup = play.get("matchup", {})
    about = play.get("about", {})

    batter_name = _clean_name(matchup.get("batter", {}).get("fullName", ""))
    pitcher_name = _clean_name(matchup.get("pitcher", {}).get("fullName", ""))
    batter_hand = matchup.get("batSide", {}).get("code", "R") or "R"

    # Pre-pitch runners (matchup.postOnFirst etc. is post-AB; we pull
    # current runners from linescore offense for the *currently-batting*
    # at-bat. For historical pitches we use whatever the matchup reports.)
    on_1b = bool(matchup.get("postOnFirst")) or bool(matchup.get("preOnFirst"))
    on_2b = bool(matchup.get("postOnSecond")) or bool(matchup.get("preOnSecond"))
    on_3b = bool(matchup.get("postOnThird")) or bool(matchup.get("preOnThird"))

    inning_half_raw = about.get("halfInning", "top")
    inning_topbot = "Top" if str(inning_half_raw).lower() == "top" else "Bot"

    return {
        "batter_name": batter_name,
        "pitcher_name": pitcher_name,
        "batter_hand": batter_hand,
        "on_1b": on_1b,
        "on_2b": on_2b,
        "on_3b": on_3b,
        "score_diff": 0,  # dry-run confirmed: zero is non-fatal for live inference
        "inning_topbot": inning_topbot,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PitchGPT inference: per-pitch top-K
# ═════════════════════════════════════════════════════════════════════════════


def _build_pitcher_history(feed: dict, pitcher_id: int, before_at_bat_idx: int,
                           before_pitch_idx: int) -> list[dict]:
    """Return the list of prior pitches by ``pitcher_id`` in this game,
    in pitch order, strictly before ``(before_at_bat_idx, before_pitch_idx)``.

    Each prior pitch dict has the fields needed for PitchTokenizer.encode
    (pitch_type, plate_x, plate_z, release_speed) and encode_context
    (balls, strikes, outs, on_1b/2b/3b, stand, inning, score_diff).
    """
    plays = feed.get("liveData", {}).get("plays", {})
    all_plays = plays.get("allPlays", [])

    history: list[dict] = []
    for play in all_plays:
        about = play.get("about", {})
        matchup = play.get("matchup", {})

        if matchup.get("pitcher", {}).get("id") != pitcher_id:
            continue

        ab_idx = about.get("atBatIndex", -1)
        if ab_idx > before_at_bat_idx:
            break  # plays are ordered; no more prior pitches by this pitcher

        inning_half_raw = about.get("halfInning", "top")
        inning_topbot = "Top" if str(inning_half_raw).lower() == "top" else "Bot"
        inning = about.get("inning", 1)

        bat_hand = matchup.get("batSide", {}).get("code", "R") or "R"
        on_1b = bool(matchup.get("postOnFirst")) or bool(matchup.get("preOnFirst"))
        on_2b = bool(matchup.get("postOnSecond")) or bool(matchup.get("preOnSecond"))
        on_3b = bool(matchup.get("postOnThird")) or bool(matchup.get("preOnThird"))

        for pe in play.get("playEvents", []):
            if not pe.get("isPitch", False):
                continue
            pe_idx = pe.get("index", -1)
            # Strict-before guard
            if ab_idx == before_at_bat_idx and pe_idx >= before_pitch_idx:
                continue

            details = pe.get("details", {})
            pitch_data = pe.get("pitchData", {})
            count_obj = pe.get("count", {})
            coords = pitch_data.get("coordinates", {})

            history.append({
                "pitch_type": details.get("type", {}).get("code", ""),
                "plate_x": coords.get("pX"),
                "plate_z": coords.get("pZ"),
                "release_speed": pitch_data.get("startSpeed"),
                "balls": count_obj.get("balls", 0),
                "strikes": count_obj.get("strikes", 0),
                "outs": count_obj.get("outs", 0),
                "on_1b": on_1b,
                "on_2b": on_2b,
                "on_3b": on_3b,
                "stand": bat_hand,
                "inning": inning,
                "score_diff": 0,
                "inning_topbot": inning_topbot,
            })

    return history


def _predict_top_k(
    model,
    history: list[dict],
    pre_pitch: dict,
    balls: int,
    strikes: int,
    outs: int,
    inning: int,
    k: int = 3,
) -> tuple[list[tuple[str, float]], np.ndarray]:
    """Run PitchGPT on the given history + pre-pitch context, return
    top-K (pitch_type, prob) tuples plus the full 17-dim per-type
    probability vector.

    Strategy: feed the prior pitches as the input sequence (with their
    per-pitch contexts) plus a "next-pitch context" appended at the end.
    The model's logits at the LAST position are the prediction for the
    NEXT pitch. We softmax over the 2210-dim composite vocab, then
    marginalize zone+velo to get a 17-dim pitch-type distribution.
    """
    device = next(model.parameters()).device

    # Build the input token + context tensors. We need at least one
    # sequence position; if no history exists, seed with a "zero context"
    # placeholder token so the model sees something.
    tokens: list[int] = []
    contexts: list[torch.Tensor] = []

    for prior in history:
        tok = PitchTokenizer.encode(
            prior.get("pitch_type"),
            prior.get("plate_x"),
            prior.get("plate_z"),
            prior.get("release_speed"),
        )
        tokens.append(tok)
        ctx_list = PitchTokenizer.encode_context(
            balls=int(prior.get("balls", 0)),
            strikes=int(prior.get("strikes", 0)),
            outs=int(prior.get("outs", 0)),
            on_1b=bool(prior.get("on_1b", False)),
            on_2b=bool(prior.get("on_2b", False)),
            on_3b=bool(prior.get("on_3b", False)),
            stand=str(prior.get("stand", "R")),
            inning=int(prior.get("inning", 1)),
            score_diff=int(prior.get("score_diff", 0)),
        )
        contexts.append(PitchTokenizer.context_to_tensor(ctx_list))

    # Append the "next pitch" context. The token at the last position
    # is irrelevant — we read logits at this position to predict it.
    # We use a benign placeholder (token 0) with the actual upcoming
    # pre-pitch context.
    next_ctx_list = PitchTokenizer.encode_context(
        balls=int(balls),
        strikes=int(strikes),
        outs=int(outs),
        on_1b=bool(pre_pitch.get("on_1b", False)),
        on_2b=bool(pre_pitch.get("on_2b", False)),
        on_3b=bool(pre_pitch.get("on_3b", False)),
        stand=str(pre_pitch.get("batter_hand", "R")),
        inning=int(inning),
        score_diff=int(pre_pitch.get("score_diff", 0)),
    )
    tokens.append(0)
    contexts.append(PitchTokenizer.context_to_tensor(next_ctx_list))

    # Truncate to the model's max_seq_len from the LEFT so we keep the
    # most recent pitches plus the "next" placeholder at the end.
    max_len = model.max_seq_len
    if len(tokens) > max_len:
        tokens = tokens[-max_len:]
        contexts = contexts[-max_len:]

    input_tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    context_tensor = torch.stack(contexts).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tokens, context_tensor)  # (1, S, V)
        # Take logits at the LAST position — those predict the NEXT pitch
        last_logits = logits[0, -1, :]  # (VOCAB_SIZE,)
        probs = torch.softmax(last_logits, dim=-1).cpu().numpy()  # (2210,)

    # Marginalize: composite vocab is pitch_type * (NUM_ZONES * NUM_VELO_BUCKETS)
    # + zone * NUM_VELO_BUCKETS + velo. Reshape to (NUM_PITCH_TYPES, NUM_ZONES,
    # NUM_VELO_BUCKETS) and sum over zone & velo to get (NUM_PITCH_TYPES,).
    type_probs = probs.reshape(NUM_PITCH_TYPES, NUM_ZONES, NUM_VELO_BUCKETS)
    type_probs = type_probs.sum(axis=(1, 2))  # (NUM_PITCH_TYPES,)

    # Numerical safety: re-normalize (sum should be 1 already)
    s = float(type_probs.sum())
    if s > 0:
        type_probs = type_probs / s

    # Top-K
    top_idx = np.argsort(-type_probs)[:k]
    top_k = [(_idx_to_pitch_type(int(i)), float(type_probs[int(i)])) for i in top_idx]

    return top_k, type_probs


def _brier_log_loss(
    type_probs: np.ndarray,
    actual_pitch_type: str,
) -> tuple[float, float]:
    """Compute the multiclass Brier score and log-loss against the actual
    one-hot pitch type. ``type_probs`` is a 17-vector summing to ~1.

    If the actual pitch type is OOV (not in PITCH_TYPE_MAP) we fall back
    to the unknown index (NUM_PITCH_TYPES - 1 == 16).
    """
    actual_idx = PITCH_TYPE_MAP.get(actual_pitch_type, NUM_PITCH_TYPES - 1)
    one_hot = np.zeros(NUM_PITCH_TYPES, dtype=np.float64)
    one_hot[actual_idx] = 1.0

    brier = float(np.sum((type_probs - one_hot) ** 2))

    p_actual = float(type_probs[actual_idx])
    # Clamp to avoid -inf; mirrors `min(nll, 20)` style cap from pitchgpt.py
    if p_actual <= 0:
        log_loss = _MAX_NLL
    else:
        log_loss = float(-np.log(max(p_actual, 1e-10)))
    log_loss = min(log_loss, _MAX_NLL)

    return brier, log_loss


# ═════════════════════════════════════════════════════════════════════════════
# Logger orchestration
# ═════════════════════════════════════════════════════════════════════════════


class LivePitchLogger:
    """Writes one row per new pitch to JSONL + CSV, with PitchGPT predictions."""

    def __init__(
        self,
        game_pk: int,
        output_dir: Path,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> None:
        self.game_pk = game_pk
        self.output_dir = output_dir
        self.poll_interval = poll_interval

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "pitch_by_pitch.jsonl"
        self.csv_path = self.output_dir / "pitch_by_pitch.csv"
        self.errors_path = self.output_dir / "errors.jsonl"

        # Resumability: keys already written
        self.logged_keys: set[tuple[int, int]] = set()
        self._load_existing_keys()

        # File handles — opened lazily on first write
        self._csv_fh: Optional[Any] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._jsonl_fh: Optional[Any] = None
        self._errors_fh: Optional[Any] = None

        # Rolling metrics
        self.n_total = 0
        self.n_correct = 0
        self.rolling_loss = deque(maxlen=200)

        # PitchGPT model — loaded lazily so --help is fast and load
        # failures are visible at startup
        self._model = None

        # The most recent feed (the LiveGameTracker stashes this for us)
        self.tracker: Optional[LiveGameTracker] = None
        self._stop_event = threading.Event()

    # ── Resumability helpers ────────────────────────────────────────────

    def _load_existing_keys(self) -> None:
        """Populate ``self.logged_keys`` from any existing JSONL file.

        Reads the entire file once (cheap — at most a few hundred lines).
        """
        if not self.jsonl_path.exists():
            return
        try:
            with open(self.jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ab = obj.get("at_bat_number")
                    pn = obj.get("pitch_number")
                    if ab is not None and pn is not None:
                        self.logged_keys.add((int(ab), int(pn)))
        except OSError as exc:
            logger.warning("Failed to read existing JSONL for resumability: %s", exc)
        if self.logged_keys:
            print(f"[resume] Loaded {len(self.logged_keys)} prior pitch keys "
                  f"from {self.jsonl_path}", flush=True)

    # ── File handle setup ──────────────────────────────────────────────

    def _ensure_open(self) -> None:
        if self._csv_fh is None:
            csv_exists = self.csv_path.exists() and self.csv_path.stat().st_size > 0
            self._csv_fh = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(
                self._csv_fh, fieldnames=CSV_COLUMNS, extrasaction="ignore",
            )
            if not csv_exists:
                self._csv_writer.writeheader()
                self._csv_fh.flush()
        if self._jsonl_fh is None:
            self._jsonl_fh = open(self.jsonl_path, "a", encoding="utf-8")
        if self._errors_fh is None:
            self._errors_fh = open(self.errors_path, "a", encoding="utf-8")

    def close(self) -> None:
        for fh in (self._csv_fh, self._jsonl_fh, self._errors_fh):
            if fh is not None:
                try:
                    fh.flush()
                    fh.close()
                except OSError:
                    pass
        self._csv_fh = None
        self._jsonl_fh = None
        self._errors_fh = None

    # ── PitchGPT model loading ─────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            print("[startup] Loading PitchGPT v1 checkpoint...", flush=True)
            self._model = _load_model("1")
            device = next(self._model.parameters()).device
            print(f"[startup] PitchGPT loaded on {device}, max_seq_len={self._model.max_seq_len}",
                  flush=True)
        return self._model

    # ── Main per-pitch handler ─────────────────────────────────────────

    def handle_pitch(self, pitch: dict) -> None:
        """Process a single new pitch event from the LiveGameTracker."""
        try:
            self._ensure_open()
            self._handle_pitch_inner(pitch)
        except Exception as exc:  # noqa: BLE001 — defensive top-level
            self._log_error("handle_pitch_outer", pitch, exc)

    def _handle_pitch_inner(self, pitch: dict) -> None:
        ab = pitch.get("at_bat_number")
        pn = pitch.get("pitch_number")
        if ab is None or pn is None:
            return

        key = (int(ab), int(pn))
        if key in self.logged_keys:
            return  # already logged on a prior run

        # Pull the most recent feed off the tracker (the LiveGameTracker
        # stores it as ``last_feed`` after each poll). Fall back to a
        # fresh fetch if for any reason it's not there.
        feed = (self.tracker.last_feed if self.tracker is not None else None)
        if feed is None:
            feed = fetch_live_feed(self.game_pk)

        pitcher_id = int(pitch.get("pitcher_id", 0))
        pitcher_name = _clean_name(pitch.get("pitcher_name", ""))

        pre = _extract_pre_pitch_state(feed, pitch)
        # Override pitcher_name from pitch if pre-state is empty
        if not pre.get("pitcher_name"):
            pre["pitcher_name"] = pitcher_name

        # Build pitcher history strictly BEFORE this pitch
        history = _build_pitcher_history(
            feed, pitcher_id,
            before_at_bat_idx=int(ab),
            before_pitch_idx=int(pn),
        )

        # Pre-pitch count = the count BEFORE this pitch was thrown.
        # The pitch dict's count_balls/count_strikes are post-pitch
        # (StatsAPI semantic). Reconstruct pre-pitch count from the
        # call_description: a "Ball" adds 1 to balls; a "Called/Swinging
        # Strike" or "Foul"/"Foul Tip" (under 2 strikes) adds 1 to strikes.
        # Easier and more reliable: read from the prior playEvent of the
        # same at-bat. If unavailable, fall back to the post-pitch count
        # (small bias only).
        pre_balls, pre_strikes = self._derive_pre_pitch_count(feed, ab, pn, pitch)

        # Run inference (defensive — wrap in try/except per the spec)
        top_k: list[tuple[str, float]] = []
        type_probs: np.ndarray = np.zeros(NUM_PITCH_TYPES)
        inference_ok = True
        try:
            model = self._get_model()
            top_k, type_probs = _predict_top_k(
                model=model,
                history=history,
                pre_pitch=pre,
                balls=pre_balls,
                strikes=pre_strikes,
                outs=int(pitch.get("outs", 0)),
                inning=int(pitch.get("inning", 1)),
                k=3,
            )
        except Exception as exc:  # noqa: BLE001
            inference_ok = False
            self._log_error("predict_top_k", pitch, exc)
            top_k = [("UN", 0.0), ("UN", 0.0), ("UN", 0.0)]
            type_probs = np.full(NUM_PITCH_TYPES, 1.0 / NUM_PITCH_TYPES)

        # Actual pitch type / zone / velo from the parsed event
        actual_pitch_type = str(pitch.get("pitch_type", "") or "")
        actual_zone = pitch.get("zone")
        actual_velo = _safe_float(pitch.get("start_speed"))

        # Compute brier / log-loss against the actual
        if actual_pitch_type and inference_ok:
            brier, log_loss = _brier_log_loss(type_probs, actual_pitch_type)
        else:
            brier, log_loss = float("nan"), float("nan")

        top1_type, top1_prob = top_k[0] if top_k else ("UN", 0.0)
        correct_top1 = (top1_type == actual_pitch_type) if actual_pitch_type else False

        # ── Write the row ──────────────────────────────────────────────
        runners_str = _runners_repr(pre.get("on_1b", False),
                                    pre.get("on_2b", False),
                                    pre.get("on_3b", False))
        half_str = "top" if pre.get("inning_topbot") == "Top" else "bottom"

        row = {
            "timestamp_utc": _utc_now_iso(),
            "inning": pitch.get("inning"),
            "half": half_str,
            "batter_id": pitch.get("batter_id"),
            "batter_name": pre.get("batter_name", ""),
            "pitcher_id": pitcher_id,
            "pitcher_name": pre.get("pitcher_name", pitcher_name),
            "count_balls": pre_balls,
            "count_strikes": pre_strikes,
            "runners_on": runners_str,
            "outs": pitch.get("outs", 0),
            "pitchgpt_top1_type": top1_type,
            "pitchgpt_top1_prob": round(float(top1_prob), 4),
            "pitchgpt_top3_json": json.dumps(
                [{"type": t, "prob": round(p, 4)} for t, p in top_k],
                separators=(",", ":"),
            ),
            "actual_pitch_type": actual_pitch_type,
            "actual_zone": actual_zone if actual_zone is not None else "",
            "actual_velo": round(actual_velo, 2) if actual_velo is not None else "",
            "correct_top1": bool(correct_top1),
            "brier": round(brier, 4) if not _isnan(brier) else "",
            "log_loss": round(log_loss, 4) if not _isnan(log_loss) else "",
        }

        # JSONL (line-atomic, includes raw key fields for resumability)
        json_obj = dict(row)
        json_obj["at_bat_number"] = key[0]
        json_obj["pitch_number"] = key[1]
        json_obj["game_pk"] = self.game_pk
        self._jsonl_fh.write(json.dumps(json_obj) + "\n")
        self._jsonl_fh.flush()

        # CSV
        self._csv_writer.writerow(row)
        self._csv_fh.flush()

        # Mark as logged
        self.logged_keys.add(key)

        # ── Stdout reporting ───────────────────────────────────────────
        self.n_total += 1
        if correct_top1:
            self.n_correct += 1
        if not _isnan(log_loss):
            self.rolling_loss.append(log_loss)

        # Per-pitch single-line print (compact)
        actual_str = actual_pitch_type or "?"
        pred_marker = "OK" if correct_top1 else "MISS"
        line = (
            f"  pitch {self.n_total:>3} | "
            f"inn {pitch.get('inning'):>2} {half_str:>3} | "
            f"{pre_balls}-{pre_strikes} | "
            f"{(pre.get('pitcher_name','') or '?')[:18]:<18} -> "
            f"{(pre.get('batter_name','') or '?')[:18]:<18} | "
            f"pred {top1_type} ({top1_prob:.2f}) vs {actual_str} [{pred_marker}]"
        )
        # Highlight high-confidence wrong calls
        if (not correct_top1) and (top1_prob > 0.5) and actual_pitch_type:
            line = _c(_BOLD + line, _RED)
        elif correct_top1:
            line = _c(line, _GREEN)
        print(line, flush=True)

        # Roll-up every 10 pitches
        if self.n_total > 0 and self.n_total % 10 == 0:
            acc = 100.0 * self.n_correct / max(self.n_total, 1)
            mean_loss = sum(self.rolling_loss) / max(len(self.rolling_loss), 1)
            summary = (
                f"[{self.n_total} pitches] top-1 acc: {acc:.2f}%, "
                f"rolling log-loss: {mean_loss:.2f}, "
                f"last call: {top1_type} vs {actual_str} ({correct_top1})"
            )
            print(_c(_BOLD + summary, _YELLOW), flush=True)

    def _derive_pre_pitch_count(
        self, feed: dict, at_bat_idx: int, pitch_idx: int, pitch: dict,
    ) -> tuple[int, int]:
        """Reconstruct the count BEFORE the given pitch was thrown.

        The parsed pitch dict carries the post-pitch count. To get the
        pre-pitch count we look at the immediately-prior playEvent in the
        same at-bat and read its count. If this is the first pitch of the
        at-bat, the pre-pitch count is 0-0.
        """
        if pitch_idx <= 0:
            return 0, 0

        plays = feed.get("liveData", {}).get("plays", {})
        all_plays = plays.get("allPlays", [])
        play = next(
            (p for p in all_plays if p.get("about", {}).get("atBatIndex") == at_bat_idx),
            None,
        )
        if play is None:
            return int(pitch.get("count_balls", 0)), int(pitch.get("count_strikes", 0))

        prior_pitches = [pe for pe in play.get("playEvents", [])
                          if pe.get("isPitch", False) and pe.get("index", -1) < pitch_idx]
        if not prior_pitches:
            return 0, 0
        prev = prior_pitches[-1]
        c = prev.get("count", {})
        # The count on the PRIOR pitch is the post-prior-pitch count, which
        # equals the pre-current-pitch count.
        return int(c.get("balls", 0)), int(c.get("strikes", 0))

    # ── Errors side-channel ────────────────────────────────────────────

    def _log_error(self, where: str, pitch: dict, exc: Exception) -> None:
        try:
            if self._errors_fh is None:
                self._ensure_open()
            obj = {
                "timestamp_utc": _utc_now_iso(),
                "where": where,
                "at_bat_number": pitch.get("at_bat_number"),
                "pitch_number": pitch.get("pitch_number"),
                "pitch_type": pitch.get("pitch_type"),
                "exc_type": type(exc).__name__,
                "exc_message": str(exc)[:500],
            }
            self._errors_fh.write(json.dumps(obj) + "\n")
            self._errors_fh.flush()
        except OSError:
            pass
        # Also log a one-liner to stderr so the operator knows
        print(_c(f"[error] {where} on pitch ({pitch.get('at_bat_number')},"
                 f"{pitch.get('pitch_number')}): {type(exc).__name__}: {exc}",
                 _RED),
              file=sys.stderr, flush=True)

    # ── Main loop ──────────────────────────────────────────────────────

    def run(self) -> None:
        # Pre-load the model so failures surface immediately
        self._get_model()
        self._ensure_open()

        # Set up tracker
        self.tracker = LiveGameTracker(
            game_pk=self.game_pk, poll_interval=self.poll_interval,
        )
        self.tracker.on_new_pitch(self.handle_pitch)

        # ── Backfill pitches that already happened before we started ──
        # The tracker only fires `on_new_pitch` for pitches added AFTER
        # its first poll. To capture pitches that happened before we
        # connected (e.g. mid-game restart), do a manual sweep first.
        try:
            initial_feed = fetch_live_feed(self.game_pk)
            from src.ingest.live_feed import parse_pitch_events  # noqa: E402
            existing = parse_pitch_events(initial_feed)
            self.tracker.last_feed = initial_feed
            self.tracker.pitch_count = len(existing)
            # Backfill: process any pitch we haven't already logged
            for pe in existing:
                key = (int(pe.get("at_bat_number", -1)), int(pe.get("pitch_number", -1)))
                if key not in self.logged_keys:
                    self.handle_pitch(pe)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Initial backfill failed (continuing): %s", exc)

        # ── Poll loop ─────────────────────────────────────────────────
        print(f"[live] Polling game_pk={self.game_pk} every {self.poll_interval}s. "
              f"Press Ctrl-C to stop.", flush=True)

        while not self._stop_event.is_set():
            try:
                state = self.tracker.poll_once()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Poll error (continuing): %s", exc)
                self._stop_event.wait(self.poll_interval)
                continue

            # Stop condition: detailedState Final / Game Over
            try:
                feed = self.tracker.last_feed or {}
                detailed = (feed.get("gameData", {})
                                  .get("status", {})
                                  .get("detailedState", ""))
            except Exception:
                detailed = ""

            if state.get("status") == "Final" or detailed in ("Final", "Game Over"):
                print(f"[done] Game {self.game_pk} reached terminal state "
                      f"(status={state.get('status')}, detailed={detailed!r}). "
                      "Exiting cleanly.", flush=True)
                break

            self._stop_event.wait(self.poll_interval)

        self.close()
        # Final summary
        if self.n_total > 0:
            acc = 100.0 * self.n_correct / self.n_total
            mean_loss = (sum(self.rolling_loss) / max(len(self.rolling_loss), 1)
                          if self.rolling_loss else float("nan"))
            print(_c(f"[final] {self.n_total} pitches logged | top-1 acc {acc:.2f}% "
                     f"| rolling log-loss {mean_loss:.2f}", _BOLD), flush=True)
        else:
            print("[final] No pitches logged.", flush=True)

    # ── External stop ──────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop_event.set()


def _isnan(x: Any) -> bool:
    try:
        return isinstance(x, float) and (x != x)  # NaN != NaN
    except Exception:
        return False


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone live pitch-by-pitch logger (Plan B).",
    )
    parser.add_argument(
        "--game-pk", type=int, required=True,
        help="MLB StatsAPI gamePk for the game to log.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write pitch_by_pitch.{jsonl,csv} and errors.jsonl.",
    )
    parser.add_argument(
        "--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL,
        help=f"Polling interval in seconds (default {DEFAULT_POLL_INTERVAL}).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    output_dir = Path(args.output_dir).resolve()

    print(f"[startup] game_pk={args.game_pk} | output={output_dir} | "
          f"poll_interval={args.poll_interval}s", flush=True)

    log = LivePitchLogger(
        game_pk=args.game_pk,
        output_dir=output_dir,
        poll_interval=args.poll_interval,
    )

    # Clean SIGINT handling — flush + close on Ctrl-C
    def _sigint_handler(signum, frame):  # noqa: ARG001
        print("\n[signal] SIGINT received — flushing + closing files...", flush=True)
        log.stop()

    try:
        signal.signal(signal.SIGINT, _sigint_handler)
    except (ValueError, OSError):
        # signal can only be set in main thread on some platforms
        pass

    try:
        log.run()
    except KeyboardInterrupt:
        print("\n[signal] KeyboardInterrupt — flushing + closing files...", flush=True)
        log.close()
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.exception("Fatal logger error: %s", exc)
        log.close()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
