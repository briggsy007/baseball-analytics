"""
PitchGPT -- Transformer-based pitch sequence model.

Treats each game as a "pitch sentence" and learns to predict the next
pitch token given the sequence so far plus situational context (count,
outs, runners, batter hand, inning, score differential).

Public API
----------
- ``PitchTokenizer``          -- encode/decode composite pitch tokens
- ``PitchSequenceDataset``    -- PyTorch Dataset of game pitch sequences
- ``PitchGPTModel``           -- small decoder-only transformer
- ``train_pitchgpt``          -- end-to-end training loop
- ``calculate_predictability`` -- Pitch Predictability Score (perplexity)
- ``calculate_disruption_index`` -- per-pitch surprise for a game
- ``batch_calculate``         -- PPS leaderboard for a season

Inherits from ``BaseAnalyticsModel`` for lifecycle / serialisation hooks.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Literal, Optional

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.analytics.base import BaseAnalyticsModel

logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    """Return the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _safe_int(val, default: int = 0) -> int:
    """Convert a possibly-NA/None value to int safely."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_bool(val) -> bool:
    """Convert a possibly-NA/None value to bool safely."""
    if val is None:
        return False
    try:
        return bool(int(val))
    except (TypeError, ValueError):
        return False


def _safe_str(val, default: str = "R") -> str:
    """Convert a possibly-NA/None value to str safely."""
    if val is None:
        return default
    s = str(val)
    if s in ("", "nan", "None", "<NA>"):
        return default
    return s


# ── Score reconstruction helpers ────────────────────────────────────────────
#
# The pitches table does not carry home_score / away_score / post_home_score
# columns (see schema).  We reconstruct a running score per game from the
# ``events`` column plus baserunner state and ``delta_run_exp`` as a
# tie-breaker signal.  This is an approximation (not exact Statcast runs),
# but it produces realistic, varied score differentials instead of the
# previous hard-coded ``0`` — which was the whole bug this module fixes.

# Events that reliably score runs — value is a conservative lower bound
# on runs scored on that event (runners must already be on).
_SAC_EVENTS = {"sac_fly", "sac_fly_double_play"}


def _runs_scored_on_event(
    event: str | None,
    on_1b_before: bool,
    on_2b_before: bool,
    on_3b_before: bool,
    delta_run_exp: float | None,
) -> int:
    """Estimate runs scored on the current pitch-event.

    This is a deliberately simple heuristic; exact reconstruction would
    require ``post_home_score`` / ``post_away_score`` columns we don't
    ingest.

    Rules:
      * ``home_run``: 1 (batter) + number of runners currently on base.
      * ``sac_fly`` family: 1 run (runner from 3rd).
      * Any event with ``delta_run_exp > 0.45`` and a runner on 3rd:
          assume ≥ 1 run scored (covers singles, walks-with-bases-loaded,
          fielders-choices-where-the-runner-scored, etc.).
      * Otherwise: 0.

    The threshold 0.45 is empirical — ``delta_run_exp`` is change in
    run expectancy, and a scoring play almost always produces a
    positive delta well above this cutoff.
    """
    if event is None:
        return 0
    ev = str(event).lower()
    if ev == "home_run":
        return 1 + int(on_1b_before) + int(on_2b_before) + int(on_3b_before)
    if ev in _SAC_EVENTS:
        return 1
    # General case: if a runner was on 3rd and the play clearly increased
    # run expectancy, one run likely scored.  This catches most singles,
    # bases-loaded walks, wild pitches on the pitch-level, etc.  It under-
    # counts multi-run events (e.g. bases-clearing doubles) but that is
    # acceptable for a 5-bucket score-diff context feature.
    if delta_run_exp is not None and not (isinstance(delta_run_exp, float) and np.isnan(delta_run_exp)):
        try:
            if float(delta_run_exp) > 0.45 and on_3b_before:
                # Attempt to credit additional runners who also scored on
                # doubles / triples when delta_run_exp is very high.
                bonus = 0
                if float(delta_run_exp) > 1.3 and on_2b_before:
                    bonus += 1
                if float(delta_run_exp) > 2.0 and on_1b_before:
                    bonus += 1
                return 1 + bonus
        except (TypeError, ValueError):
            pass
    return 0


def _score_diff_for_pitch(
    home_score_before: int,
    away_score_before: int,
    inning_topbot: str | None,
) -> int:
    """Compute score_diff from the perspective of the *pitching* team.

    In Statcast convention, when ``inning_topbot == 'Top'`` the home team
    is pitching (away batting); when ``'Bot'`` the away team is pitching.
    The PitchGPT context feature wants score_diff from the pitcher's POV:
    positive = pitcher's team leading.
    """
    diff = home_score_before - away_score_before
    if inning_topbot is None:
        return diff
    tb = str(inning_topbot).lower()
    if tb.startswith("bot"):
        # Away team pitching; flip sign.
        return -diff
    return diff


def _compute_per_pitch_score_diff(df: pd.DataFrame) -> list[int]:
    """Walk each game in ``df`` in pitch order and return the per-row
    score_diff (pitcher's POV) at the moment of each pitch.

    Expects columns ``game_pk``, ``inning_topbot``, ``on_1b``, ``on_2b``,
    ``on_3b``, ``events``, ``delta_run_exp``, plus an ordering that is
    monotonically increasing within a game — the caller is responsible
    for sorting by ``(at_bat_number, pitch_number)``.

    Returns a list aligned 1:1 with ``df.iterrows()`` in iteration order.
    """
    if df.empty:
        return []

    # We need to preserve the row order the caller used (iterrows order).
    # ``df.groupby(..., sort=False)`` keeps the order of first appearance
    # for each group key, and within-group order is preserved.
    result = np.zeros(len(df), dtype=np.int32)
    # Track per-game running score.
    game_scores: dict[int, tuple[int, int]] = {}

    # iterate by position to align with output array cleanly
    # df may have a non-default index; use reset_index approach
    reset = df.reset_index(drop=True)
    for pos, row in reset.iterrows():
        gpk_raw = row.get("game_pk")
        try:
            gpk = int(gpk_raw) if gpk_raw is not None else -1
        except (TypeError, ValueError):
            gpk = -1
        home_s, away_s = game_scores.get(gpk, (0, 0))

        topbot = row.get("inning_topbot")
        result[pos] = _score_diff_for_pitch(home_s, away_s, topbot)

        # Update running score AFTER recording (score_diff reflects state
        # at the moment the pitch is thrown, not after the result).
        runs = _runs_scored_on_event(
            row.get("events"),
            _safe_bool(row.get("on_1b")),
            _safe_bool(row.get("on_2b")),
            _safe_bool(row.get("on_3b")),
            row.get("delta_run_exp"),
        )
        if runs:
            tb = str(topbot).lower() if topbot is not None else ""
            if tb.startswith("top"):
                # Away team batting → away scores
                away_s += runs
            else:
                home_s += runs
            game_scores[gpk] = (home_s, away_s)
        else:
            game_scores[gpk] = (home_s, away_s)

    return result.tolist()


# ── Paths ────────────────────────────────────────────────────────────────────
_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


# ── Umpire tendency loader ───────────────────────────────────────────────────
#
# The PitchGPT context vector (added 2026-04-23) includes a per-pitch
# ``ump_accuracy_above_x`` scalar derived from the ``umpire_assignments`` +
# ``umpire_tendencies`` tables.  See ``docs/data/umpire_integration_notes.md``.
#
# Design decisions (locked in the spec doc):
#   * Use **prior-season** tendency (e.g. 2024 games use 2023 tendencies).
#     This is what the integration notes call out as the OOS-leakage-safe
#     choice — the model sees a statistic about the ump that would be
#     knowable before the game is played.
#   * NULL / missing joins (rookie umps with no prior season; 2017 / 2019
#     pitches-table gaps) are filled with the **prior-season league
#     median** of ``accuracy_above_x_wmean``, computed from
#     ``umpire_tendencies`` at load time.  A rookie ump is treated as
#     "neutral = league median" rather than 0.0 so the scalar distribution
#     stays centred on the training data.
#   * The tendency scalar is fed raw (not z-scored).  Scale across the
#     2015-2025 data ranges from roughly -1.8 to +2.5 accuracy-above-x
#     points, which is a reasonable magnitude for the transformer's
#     ``Linear(CONTEXT_DIM, d_model)`` projection to absorb.


def _fetch_prior_season_ump_tendency(
    conn: duckdb.DuckDBPyConnection,
    seasons: list[int] | range | None,
) -> dict[tuple[str, int], float]:
    """Return a mapping from ``(umpire_name, game_season)`` to
    prior-season ``accuracy_above_x_wmean``.

    The key is the *game* season, not the tendency season — so a 2024
    game lookup returns the 2023 tendency.  Build this map once per
    dataset load so the per-pitch lookup is O(1).
    """
    if seasons is None:
        season_list: list[int] | None = None
    else:
        season_list = [int(s) for s in seasons]
        if not season_list:
            return {}

    # We need tendencies from (min(seasons) - 1) through (max(seasons) - 1).
    # If seasons is None (legacy path), pull everything.
    if season_list is None:
        query = """
            SELECT umpire, season, accuracy_above_x_wmean
            FROM umpire_tendencies
            WHERE accuracy_above_x_wmean IS NOT NULL
        """
        rows = conn.execute(query).fetchdf()
    else:
        prior_seasons = sorted({s - 1 for s in season_list})
        ps_str = ", ".join(str(p) for p in prior_seasons)
        query = f"""
            SELECT umpire, season, accuracy_above_x_wmean
            FROM umpire_tendencies
            WHERE accuracy_above_x_wmean IS NOT NULL
              AND season IN ({ps_str})
        """
        rows = conn.execute(query).fetchdf()

    mapping: dict[tuple[str, int], float] = {}
    for _, row in rows.iterrows():
        ump = str(row["umpire"])
        tend_season = int(row["season"])
        val = float(row["accuracy_above_x_wmean"])
        # Key = (umpire_name, game_season) where game_season = tend_season + 1.
        mapping[(ump, tend_season + 1)] = val
    return mapping


def _fetch_season_league_median(
    conn: duckdb.DuckDBPyConnection,
    seasons: list[int] | range | None,
) -> dict[int, float]:
    """Return ``{game_season -> prior-season league-median accuracy_above_x}``.

    Used as the NULL-fill for rookie umpires and ump-assignment gaps.
    A game in 2024 gets the median of 2023 umpire tendencies.

    Edge cases:
      * If the prior season has no data (e.g. 2015 game → 2014 tendencies,
        and 2014 is absent because ``umpire_tendencies`` starts at 2015),
        fall back to the *same* season's median (so a 2015 game uses the
        2015 median instead of zero).  This is a mild lookahead but it is
        aggregate-only and strictly better than a hard zero (which would
        silently bias the ump scalar to 0 for a whole season).  The
        alternative — filling with 0 — created a constant-feature
        regression for 2015 games in v1 training.
      * If even the same-season median is missing (impossible with
        current data but safe to guard), fall back to 0.0.
    """
    if seasons is None:
        return {}
    season_list = [int(s) for s in seasons]
    if not season_list:
        return {}
    # Pull both the prior-season medians AND the current-season medians
    # in a single query so the same-season backstop is available.
    needed = sorted({s - 1 for s in season_list} | {s for s in season_list})
    ps_str = ", ".join(str(p) for p in needed)
    query = f"""
        SELECT season, MEDIAN(accuracy_above_x_wmean) AS med
        FROM umpire_tendencies
        WHERE accuracy_above_x_wmean IS NOT NULL
          AND season IN ({ps_str})
        GROUP BY season
    """
    rows = conn.execute(query).fetchdf()
    raw: dict[int, float] = {}
    for _, row in rows.iterrows():
        raw[int(row["season"])] = float(row["med"])

    medians: dict[int, float] = {}
    for s in season_list:
        # Prefer prior-season median.
        prior = raw.get(s - 1)
        if prior is not None:
            medians[s] = prior
            continue
        # Same-season backstop (only matters at the leading edge e.g. 2015).
        same = raw.get(s)
        if same is not None:
            medians[s] = same
            continue
        # Last resort — zero.
        medians[s] = 0.0
    return medians


def _fetch_hp_umpire_by_game(
    conn: duckdb.DuckDBPyConnection,
    seasons: list[int] | range | None,
) -> dict[int, str]:
    """Return ``{game_pk -> HP umpire name}`` for the requested seasons.

    Uses the ``umpire_assignments`` table restricted to ``position = 'HP'``.
    A game with no HP assignment is absent from the map and will trigger
    the NULL-fill path downstream.
    """
    if seasons is None:
        filt = ""
    else:
        season_list = [int(s) for s in seasons]
        if not season_list:
            return {}
        s_str = ", ".join(str(s) for s in season_list)
        filt = f"AND season IN ({s_str})"
    query = f"""
        SELECT game_pk, umpire_name
        FROM umpire_assignments
        WHERE position = 'HP'
          AND game_pk IS NOT NULL
          AND umpire_name IS NOT NULL
          {filt}
    """
    rows = conn.execute(query).fetchdf()
    # Deduplicate in case of repeated (game_pk, HP) rows across sources.
    out: dict[int, str] = {}
    for _, row in rows.iterrows():
        try:
            gpk = int(row["game_pk"])
        except (TypeError, ValueError):
            continue
        out[gpk] = str(row["umpire_name"])
    return out

# ── Constants ────────────────────────────────────────────────────────────────
# Pitch type vocabulary (indices 0-15 for known types, 16 = unknown)
PITCH_TYPE_MAP: dict[str, int] = {
    "FF": 0, "SI": 1, "FC": 2, "SL": 3, "CU": 4, "CH": 5,
    "FS": 6, "KC": 7, "KN": 8, "EP": 9, "CS": 10, "SV": 11,
    "ST": 12, "SC": 13, "FO": 14, "FA": 15,
}
NUM_PITCH_TYPES = 17  # 16 known + 1 unknown
NUM_ZONES = 26        # 5x5 grid (0-24) + 1 for out-of-zone
NUM_VELO_BUCKETS = 5  # <80, 80-85, 85-90, 90-95, 95+
VOCAB_SIZE = NUM_PITCH_TYPES * NUM_ZONES * NUM_VELO_BUCKETS  # 17*26*5 = 2210

# Context dimensions
NUM_COUNT_STATES = 12   # 0-0 .. 3-2
NUM_OUTS = 3            # 0, 1, 2
NUM_RUNNER_STATES = 8   # 2^3 base combinations
NUM_BATTER_HANDS = 2    # L, R
NUM_INNING_BUCKETS = 4  # early(1-3), mid(4-6), late(7-9), extra(10+)
NUM_SCORE_DIFF_BUCKETS = 5  # big deficit, small deficit, tie, small lead, big lead
# Continuous scalar added 2026-04-23: prior-season home-plate umpire accuracy
# above expectation (from ``umpire_tendencies.accuracy_above_x_wmean``).
# Positive = tighter ump (more correctly-called pitches than Statcast-expected),
# negative = looser.  Prior-season aggregate avoids lookahead leakage in OOS eval.
# Rookie umpires / NULL joins are filled with the season league median
# (computed from ``umpire_tendencies`` at load time).  Documented in
# ``docs/data/umpire_integration_notes.md``.
NUM_UMP_SCALAR = 1
CONTEXT_DIM = (
    NUM_COUNT_STATES + NUM_OUTS + NUM_RUNNER_STATES
    + NUM_BATTER_HANDS + NUM_INNING_BUCKETS + NUM_SCORE_DIFF_BUCKETS
    + NUM_UMP_SCALAR
)  # 35

# Special tokens
PAD_TOKEN = VOCAB_SIZE      # padding
BOS_TOKEN = VOCAB_SIZE + 1  # beginning of sequence
TOTAL_VOCAB = VOCAB_SIZE + 2

# Training defaults
DEFAULT_DIM = 128
DEFAULT_HEADS = 4
DEFAULT_LAYERS = 4
DEFAULT_MAX_SEQ = 256
DEFAULT_EPOCHS = 5
DEFAULT_BATCH = 32
DEFAULT_LR = 1e-3
MIN_PITCHES_LEADERBOARD = 200


# ═════════════════════════════════════════════════════════════════════════════
# Tokenizer
# ═════════════════════════════════════════════════════════════════════════════

class PitchTokenizer:
    """Encode / decode a pitch into a single composite integer token.

    token = pitch_type_idx * (NUM_ZONES * NUM_VELO_BUCKETS)
            + zone_idx * NUM_VELO_BUCKETS
            + velo_bucket
    """

    # ── Encoding helpers ─────────────────────────────────────────────────

    @staticmethod
    def pitch_type_to_idx(pitch_type: str | None) -> int:
        if pitch_type is None:
            return NUM_PITCH_TYPES - 1  # unknown
        return PITCH_TYPE_MAP.get(pitch_type, NUM_PITCH_TYPES - 1)

    @staticmethod
    def idx_to_pitch_type(idx: int) -> str:
        inv = {v: k for k, v in PITCH_TYPE_MAP.items()}
        return inv.get(idx, "UN")

    @staticmethod
    def location_to_zone(plate_x: float | None, plate_z: float | None) -> int:
        """Discretise plate location into a 5x5 grid (0-24) or 25 for missing."""
        if plate_x is None or plate_z is None or np.isnan(plate_x) or np.isnan(plate_z):
            return NUM_ZONES - 1  # out-of-zone / missing

        # Plate is roughly -1.5 to 1.5 feet wide, 1.0 to 4.0 feet tall
        x_edges = np.linspace(-1.5, 1.5, 6)
        z_edges = np.linspace(1.0, 4.0, 6)

        x_bin = int(np.clip(np.digitize(plate_x, x_edges) - 1, 0, 4))
        z_bin = int(np.clip(np.digitize(plate_z, z_edges) - 1, 0, 4))
        return z_bin * 5 + x_bin

    @staticmethod
    def velocity_to_bucket(release_speed: float | None) -> int:
        """Map velocity to one of 5 buckets."""
        if release_speed is None or np.isnan(release_speed):
            return 2  # default to middle bucket
        if release_speed < 80:
            return 0
        elif release_speed < 85:
            return 1
        elif release_speed < 90:
            return 2
        elif release_speed < 95:
            return 3
        else:
            return 4

    # ── Main encode / decode ─────────────────────────────────────────────

    @classmethod
    def encode(
        cls,
        pitch_type: str | None,
        plate_x: float | None,
        plate_z: float | None,
        release_speed: float | None,
    ) -> int:
        pt_idx = cls.pitch_type_to_idx(pitch_type)
        zone = cls.location_to_zone(plate_x, plate_z)
        velo = cls.velocity_to_bucket(release_speed)
        return pt_idx * (NUM_ZONES * NUM_VELO_BUCKETS) + zone * NUM_VELO_BUCKETS + velo

    @classmethod
    def decode(cls, token: int) -> dict[str, Any]:
        if token < 0 or token >= VOCAB_SIZE:
            return {"pitch_type": "UN", "zone": -1, "velo_bucket": -1}
        velo = token % NUM_VELO_BUCKETS
        remaining = token // NUM_VELO_BUCKETS
        zone = remaining % NUM_ZONES
        pt_idx = remaining // NUM_ZONES
        return {
            "pitch_type": cls.idx_to_pitch_type(pt_idx),
            "zone": zone,
            "velo_bucket": velo,
        }

    @classmethod
    def encode_context(
        cls,
        balls: int,
        strikes: int,
        outs: int,
        on_1b: bool,
        on_2b: bool,
        on_3b: bool,
        stand: str,
        inning: int,
        score_diff: int,
    ) -> list[int]:
        """Encode situational context as a list of categorical indices.

        Returns a list of 6 integers (one per context feature).

        Note: the umpire-accuracy scalar (added 2026-04-23) is NOT part of
        this list — it is a continuous float and enters via
        :meth:`context_to_tensor`'s ``ump_scalar`` arg.  Keeping the
        categorical encoding signature unchanged preserves backward
        compatibility for every caller that only supplies the 6 categorical
        context features.
        """
        count_state = min(balls, 3) * 3 + min(strikes, 2)
        outs_idx = min(outs, 2)
        runner_state = int(bool(on_1b)) * 4 + int(bool(on_2b)) * 2 + int(bool(on_3b))
        batter_hand = 0 if stand == "L" else 1

        if inning <= 3:
            inning_bucket = 0
        elif inning <= 6:
            inning_bucket = 1
        elif inning <= 9:
            inning_bucket = 2
        else:
            inning_bucket = 3

        if score_diff <= -4:
            score_bucket = 0
        elif score_diff < 0:
            score_bucket = 1
        elif score_diff == 0:
            score_bucket = 2
        elif score_diff <= 3:
            score_bucket = 3
        else:
            score_bucket = 4

        return [count_state, outs_idx, runner_state, batter_hand,
                inning_bucket, score_bucket]

    @classmethod
    def context_to_tensor(
        cls,
        context_list: list[int],
        ump_scalar: float = 0.0,
    ) -> torch.Tensor:
        """One-hot encode a context list plus append the umpire scalar.

        Args:
            context_list: 6 categorical indices from :meth:`encode_context`.
            ump_scalar: Prior-season home-plate umpire ``accuracy_above_x_wmean``
                for the current pitch's game.  NULL / missing must be
                resolved to the season league median at the call site before
                reaching this method (default 0.0 = neutral ump only if the
                caller has not pre-filled).

        Returns:
            Float tensor of size ``CONTEXT_DIM`` (35).  The first 34 slots
            are the categorical one-hots; the final slot is the continuous
            umpire scalar.
        """
        vec = torch.zeros(CONTEXT_DIM, dtype=torch.float32)
        offsets = [0, NUM_COUNT_STATES, NUM_COUNT_STATES + NUM_OUTS,
                   NUM_COUNT_STATES + NUM_OUTS + NUM_RUNNER_STATES,
                   NUM_COUNT_STATES + NUM_OUTS + NUM_RUNNER_STATES + NUM_BATTER_HANDS,
                   NUM_COUNT_STATES + NUM_OUTS + NUM_RUNNER_STATES + NUM_BATTER_HANDS + NUM_INNING_BUCKETS]
        for i, val in enumerate(context_list):
            vec[offsets[i] + val] = 1.0
        # Final slot: continuous umpire accuracy-above-x scalar.
        vec[CONTEXT_DIM - 1] = float(ump_scalar)
        return vec


# ═════════════════════════════════════════════════════════════════════════════
# Date-based split helpers (Ticket #1)
# ═════════════════════════════════════════════════════════════════════════════


def _range_to_years(rng: tuple[int, int]) -> set[int]:
    """Expand an inclusive (start, end) year range to a set of years."""
    a, b = rng
    if a > b:
        raise ValueError(f"Year range {rng} is inverted (start > end).")
    return set(range(int(a), int(b) + 1))


def _assert_ranges_disjoint(
    train_range: tuple[int, int],
    val_range: tuple[int, int],
    test_range: tuple[int, int],
) -> None:
    """Raise ``ValueError`` if any of train/val/test year sets overlap."""
    tr = _range_to_years(train_range)
    va = _range_to_years(val_range)
    te = _range_to_years(test_range)
    for a_name, a_set, b_name, b_set in [
        ("train", tr, "val", va),
        ("train", tr, "test", te),
        ("val", va, "test", te),
    ]:
        overlap = a_set & b_set
        if overlap:
            raise ValueError(
                f"Leakage guard: {a_name} and {b_name} year ranges overlap "
                f"on {sorted(overlap)}. train={train_range}, val={val_range}, "
                f"test={test_range}."
            )


def audit_no_game_overlap(
    train_ds: "PitchSequenceDataset",
    val_ds: "PitchSequenceDataset",
    test_ds: "PitchSequenceDataset",
) -> dict:
    """Leakage audit for the date-based split.

    Returns a dict with:
      * ``shared_game_pks``: count of ``game_pk`` values appearing in
        more than one split.  MUST be 0 — a shared game_pk is a hard
        leakage failure.
      * ``shared_pitcher_ids_train_test``: count of pitcher IDs that
        appear in both train AND test.  Per spec Ticket #1, MUST be 0
        once the pitcher-disjoint split (``exclude_pitcher_ids`` on
        :class:`PitchSequenceDataset`) is wired up upstream.  A nonzero
        value here means val/test contain pitchers the model already
        memorised in train — the identity-only ablation will dominate
        and the contextual-feature gates become uninterpretable.
      * ``shared_pitcher_ids_train_val``: same, for train vs val.
        MUST also be 0 to keep the val-set selection criterion clean.
      * Per-split ``n_game_pks`` / ``n_pitcher_ids`` counts.
    """
    train_games = set(train_ds.game_pks)
    val_games = set(val_ds.game_pks)
    test_games = set(test_ds.game_pks)

    shared_tv = train_games & val_games
    shared_tt = train_games & test_games
    shared_vt = val_games & test_games
    shared_any = shared_tv | shared_tt | shared_vt

    shared_pitchers_tt = train_ds.pitcher_ids & test_ds.pitcher_ids
    shared_pitchers_tv = train_ds.pitcher_ids & val_ds.pitcher_ids

    return {
        "shared_game_pks": len(shared_any),
        "shared_game_pks_train_val": len(shared_tv),
        "shared_game_pks_train_test": len(shared_tt),
        "shared_game_pks_val_test": len(shared_vt),
        "shared_pitcher_ids_train_test": len(shared_pitchers_tt),
        "shared_pitcher_ids_train_val": len(shared_pitchers_tv),
        "n_train_game_pks": len(train_games),
        "n_val_game_pks": len(val_games),
        "n_test_game_pks": len(test_games),
        "n_train_pitchers": len(train_ds.pitcher_ids),
        "n_val_pitchers": len(val_ds.pitcher_ids),
        "n_test_pitchers": len(test_ds.pitcher_ids),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Dataset
# ═════════════════════════════════════════════════════════════════════════════

class PitchSequenceDataset(Dataset):
    """Loads game pitch sequences from the database, tokenises them, and
    returns ``(token_seq, context_seq, target_seq)`` where target is
    token_seq shifted by 1.

    Each *game* by a single pitcher is one sequence (all at-bats
    concatenated in pitch order).
    """

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        seasons: list[int] | range | None = None,
        max_seq_len: int = DEFAULT_MAX_SEQ,
        max_games: int | None = None,
        split_mode: Optional[Literal["train", "val", "test"]] = None,
        train_range: tuple[int, int] = (2015, 2022),
        val_range: tuple[int, int] = (2023, 2023),
        test_range: tuple[int, int] = (2024, 2024),
        max_games_per_split: int | None = None,
        exclude_pitcher_ids: set[int] | None = None,
        context_dim: int = CONTEXT_DIM,
    ) -> None:
        """Loads a PyTorch Dataset of pitch sequences.

        The default (``split_mode=None``, ``seasons=...``) path is the
        backwards-compatible single-season behaviour used by dashboards
        and the existing ``train_pitchgpt`` training loop.

        When ``split_mode`` is set, the dataset resolves the season range
        from ``train_range``/``val_range``/``test_range`` (inclusive on
        both ends) and ignores ``seasons``.  A runtime guard asserts the
        three ranges are disjoint — overlap raises ``ValueError``.

        Args:
            conn: DuckDB connection.
            seasons: Legacy season filter (ignored if ``split_mode`` is set).
            max_seq_len: Truncate any single game sequence to this length.
            max_games: Legacy max-games sample (ignored if
                ``split_mode`` is set; use ``max_games_per_split`` instead).
            split_mode: One of ``"train" | "val" | "test"``.  If ``None``
                the dataset behaves exactly as before.
            train_range: (start_year, end_year) inclusive for training.
            val_range: (start_year, end_year) inclusive for validation.
            test_range: (start_year, end_year) inclusive for testing.
            max_games_per_split: Cap the number of games sampled per
                split (for quick dev runs).  Applied only when
                ``split_mode`` is set.
            exclude_pitcher_ids: Optional set of pitcher_ids to drop at
                load time.  Used by the pitcher-disjoint split: pass the
                train cohort's pitcher_ids to a val/test dataset to
                guarantee no pitcher appears in both train and val/test.
                Filter is applied at the SQL level for efficiency, so the
                ``USING SAMPLE`` reservoir picks from already-eligible
                games (see Ticket #1 in the validation spec).
        """
        self.max_seq_len = max_seq_len
        self.sequences: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        # Track which game_pks and pitcher_ids actually produced a
        # sequence — used by the leakage-audit utility.
        self.game_pks: set[int] = set()
        self.pitcher_ids: set[int] = set()
        self.split_mode = split_mode
        self.exclude_pitcher_ids: set[int] = (
            set(int(p) for p in exclude_pitcher_ids) if exclude_pitcher_ids else set()
        )
        # ``context_dim`` controls the width of the per-pitch context
        # tensor.  Default is module-level CONTEXT_DIM (35, v2 schema
        # with the umpire scalar at index 34).  Pass ``context_dim=34``
        # to reproduce the v1 schema (categoricals only, no ump) — the
        # loader drops the umpire slot at sequence-build time.
        if context_dim > CONTEXT_DIM or context_dim < 1:
            raise ValueError(
                f"context_dim={context_dim} must be in [1, {CONTEXT_DIM}]"
            )
        self.context_dim = context_dim

        if split_mode is not None:
            # Hard guard: the three ranges must be disjoint.
            _assert_ranges_disjoint(train_range, val_range, test_range)
            if split_mode == "train":
                seasons_resolved = list(range(train_range[0], train_range[1] + 1))
            elif split_mode == "val":
                seasons_resolved = list(range(val_range[0], val_range[1] + 1))
            elif split_mode == "test":
                seasons_resolved = list(range(test_range[0], test_range[1] + 1))
            else:
                raise ValueError(
                    f"split_mode must be one of 'train', 'val', 'test', "
                    f"got {split_mode!r}"
                )
            self._load(
                conn,
                seasons_resolved,
                max_games=max_games_per_split,
            )
        else:
            self._load(conn, seasons, max_games=max_games)

    # ── Class helpers ────────────────────────────────────────────────────

    @staticmethod
    def fetch_pitcher_ids_for_seasons(
        conn: duckdb.DuckDBPyConnection,
        seasons: list[int] | range,
    ) -> set[int]:
        """Return the set of distinct pitcher_ids appearing in ``seasons``.

        Used by the pitcher-disjoint split: callers fetch the train
        cohort's pitcher set with this helper, then pass it as
        ``exclude_pitcher_ids`` to the val and test datasets.

        Note: this enumerates pitchers across the *full* season range
        (no game sampling), so the exclusion is exhaustive — even if
        the train dataset later subsamples games via ``USING SAMPLE``,
        the exclusion still covers every pitcher who *could* have been
        in train.  This matches the spec: "no pitcher appearing in any
        train sequence may appear in any val or test sequence", read
        as "no pitcher in the train *cohort*", since the train sample
        is otherwise non-deterministic.
        """
        season_list = list(seasons)
        if not season_list:
            return set()
        s_str = ", ".join(str(int(s)) for s in season_list)
        query = f"""
            SELECT DISTINCT pitcher_id
            FROM pitches
            WHERE pitch_type IS NOT NULL
              AND pitcher_id IS NOT NULL
              AND EXTRACT(YEAR FROM game_date) IN ({s_str})
        """
        rows = conn.execute(query).fetchdf()
        return {int(p) for p in rows["pitcher_id"].tolist() if p is not None}

    # ── Private ──────────────────────────────────────────────────────────

    def _load(self, conn: duckdb.DuckDBPyConnection, seasons, max_games: int | None = None) -> None:
        season_filter = ""
        if seasons is not None:
            season_list = list(seasons)
            if season_list:
                s_str = ", ".join(str(int(s)) for s in season_list)
                season_filter = f"AND EXTRACT(YEAR FROM game_date) IN ({s_str})"

        # Pitcher-disjoint exclusion (Ticket #1 hardening).  Apply at the
        # SQL level so the random game sample picks only from games that
        # actually have at least one eligible pitcher.  Not perfect — a
        # game may include both eligible and excluded pitchers, but the
        # per-row WHERE filter below removes the excluded pitchers'
        # pitches from the result set, so no excluded-pitcher sequence
        # is constructed.  Combined with the in-memory guard in the
        # grouping loop, this gives belt-and-suspenders enforcement.
        pitcher_exclude_filter = ""
        if self.exclude_pitcher_ids:
            # DuckDB handles million-element IN lists but we keep this
            # safe and chunked-friendly via a temp-table approach if
            # the exclusion gets large.  For typical MLB-pitcher
            # cardinalities (~3-5K total pitchers across the era) the
            # raw IN list is fine.
            pids_str = ", ".join(str(int(p)) for p in self.exclude_pitcher_ids)
            pitcher_exclude_filter = f"AND pitcher_id NOT IN ({pids_str})"

        # When max_games is set, restrict to a random sample of game_pks
        # to avoid loading the entire 7M+ row table into memory.
        game_filter = ""
        if max_games is not None:
            game_filter = f"""
                AND game_pk IN (
                    SELECT game_pk FROM (
                        SELECT DISTINCT game_pk
                        FROM pitches
                        WHERE pitch_type IS NOT NULL {season_filter}
                          {pitcher_exclude_filter}
                    ) USING SAMPLE {int(max_games)} ROWS
                )
            """

        # NOTE: we also pull ``events``, ``delta_run_exp`` and ``inning_topbot``
        # so we can reconstruct a running score per game — the pitches
        # table does not carry home_score / away_score / post_home_score
        # columns.  See ``_compute_per_pitch_score_diff`` above.  The ORDER
        # BY must sort by ``(game_pk, at_bat_number, pitch_number)`` so
        # the reconstructor sees pitches in game order even though
        # multiple pitchers share a game_pk.
        query = f"""
            SELECT
                game_pk,
                pitcher_id,
                pitch_type,
                plate_x,
                plate_z,
                release_speed,
                balls,
                strikes,
                outs_when_up,
                on_1b,
                on_2b,
                on_3b,
                stand,
                inning,
                inning_topbot,
                events,
                delta_run_exp,
                at_bat_number,
                pitch_number,
                game_date
            FROM pitches
            WHERE pitch_type IS NOT NULL
              {season_filter}
              {pitcher_exclude_filter}
              {game_filter}
            ORDER BY game_pk, at_bat_number, pitch_number
        """
        df = conn.execute(query).fetchdf()

        if df.empty:
            logger.warning("No pitch data found for PitchGPT dataset.")
            return

        # Reconstruct per-pitch score_diff (pitcher POV) for the whole
        # frame in one sweep, then attach as a column.  Replaces the
        # previous hard-coded ``score_diff=0`` so the "situational
        # context" feature actually reflects game state.
        df = df.assign(_score_diff=_compute_per_pitch_score_diff(df))

        # Attach per-pitch umpire accuracy-above-x scalar (added 2026-04-23).
        # See ``_fetch_prior_season_ump_tendency`` above for design.
        df_seasons = sorted({int(s) for s in df["game_date"].dt.year.unique()}) \
            if "game_date" in df.columns else (
                sorted({int(y) for y in seasons}) if seasons is not None else []
            )
        ump_by_game = _fetch_hp_umpire_by_game(conn, df_seasons)
        tendency_map = _fetch_prior_season_ump_tendency(conn, df_seasons)
        median_by_season = _fetch_season_league_median(conn, df_seasons)

        def _ump_scalar(row) -> float:
            # Determine game season.
            try:
                gdate = row.get("game_date") if "game_date" in df.columns else None
                if gdate is not None and not (isinstance(gdate, float) and np.isnan(gdate)):
                    try:
                        game_season = int(pd.Timestamp(gdate).year)
                    except (TypeError, ValueError):
                        game_season = -1
                else:
                    game_season = -1
            except Exception:
                game_season = -1
            median = median_by_season.get(game_season, 0.0)
            try:
                gpk = int(row.get("game_pk")) if row.get("game_pk") is not None else None
            except (TypeError, ValueError):
                gpk = None
            if gpk is None:
                return median
            ump_name = ump_by_game.get(gpk)
            if ump_name is None:
                return median
            val = tendency_map.get((ump_name, game_season))
            if val is None:
                return median
            return float(val)

        df = df.assign(_ump_scalar=df.apply(_ump_scalar, axis=1))

        # Group by (game, pitcher) to form sequences
        grouped = df.groupby(["game_pk", "pitcher_id"], sort=False)
        for _key, game_df in grouped:
            if len(game_df) < 2:
                continue  # need at least 2 pitches

            # Belt-and-suspenders: drop any sequence whose pitcher is in
            # the exclusion set (should never trigger thanks to the SQL
            # filter above, but guards against future refactors that
            # bypass the SQL path).
            try:
                _pid = int(_key[1])
            except (TypeError, ValueError):
                _pid = -1
            if self.exclude_pitcher_ids and _pid in self.exclude_pitcher_ids:
                continue

            # Track membership for the leakage audit utility.
            try:
                self.game_pks.add(int(_key[0]))
                self.pitcher_ids.add(_pid)
            except (TypeError, ValueError):
                pass

            tokens: list[int] = []
            contexts: list[torch.Tensor] = []

            for _, row in game_df.iterrows():
                tok = PitchTokenizer.encode(
                    row.get("pitch_type"),
                    row.get("plate_x"),
                    row.get("plate_z"),
                    row.get("release_speed"),
                )
                tokens.append(tok)

                ctx_list = PitchTokenizer.encode_context(
                    balls=_safe_int(row.get("balls"), 0),
                    strikes=_safe_int(row.get("strikes"), 0),
                    outs=_safe_int(row.get("outs_when_up"), 0),
                    on_1b=_safe_bool(row.get("on_1b")),
                    on_2b=_safe_bool(row.get("on_2b")),
                    on_3b=_safe_bool(row.get("on_3b")),
                    stand=_safe_str(row.get("stand"), "R"),
                    inning=_safe_int(row.get("inning"), 1),
                    score_diff=_safe_int(row.get("_score_diff"), 0),
                )
                ump_raw = row.get("_ump_scalar")
                try:
                    ump_val = 0.0 if ump_raw is None else float(ump_raw)
                except (TypeError, ValueError):
                    ump_val = 0.0
                contexts.append(
                    PitchTokenizer.context_to_tensor(ctx_list, ump_scalar=ump_val)
                )

            # Truncate to max_seq_len
            tokens = tokens[: self.max_seq_len]
            contexts = contexts[: self.max_seq_len]

            # Input = tokens[:-1], target = tokens[1:]
            input_tokens = torch.tensor(tokens[:-1], dtype=torch.long)
            target_tokens = torch.tensor(tokens[1:], dtype=torch.long)
            context_tensor = torch.stack(contexts[:-1])  # aligned with input

            # Drop the tail (umpire scalar) if caller requested a
            # narrower schema (v1: context_dim=34).  No-op at default.
            if self.context_dim < CONTEXT_DIM:
                context_tensor = context_tensor[:, : self.context_dim]

            self.sequences.append((input_tokens, context_tensor, target_tokens))

        logger.info(
            "PitchGPT dataset: %d sequences from %d pitches.",
            len(self.sequences),
            len(df),
        )

    # ── Dataset interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx]


def _collate_fn(batch):
    """Pad variable-length sequences to the longest in the batch."""
    max_len = max(item[0].size(0) for item in batch)
    tokens_batch, ctx_batch, target_batch = [], [], []

    for tokens, ctx, target in batch:
        seq_len = tokens.size(0)
        pad_len = max_len - seq_len

        tokens_padded = torch.cat(
            [tokens, torch.full((pad_len,), PAD_TOKEN, dtype=torch.long)]
        )
        target_padded = torch.cat(
            [target, torch.full((pad_len,), PAD_TOKEN, dtype=torch.long)]
        )
        if pad_len > 0:
            # Use ctx's own last dim so collate works regardless of the
            # dataset's context_dim (v1: 34, v2: 35).
            ctx_padded = torch.cat(
                [ctx, torch.zeros(pad_len, ctx.size(-1))]
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


# ═════════════════════════════════════════════════════════════════════════════
# Model
# ═════════════════════════════════════════════════════════════════════════════

class PitchGPTModel(nn.Module):
    """Small decoder-only transformer for next-pitch prediction.

    Architecture:
        token_embedding(TOTAL_VOCAB, dim) + context_projection(CONTEXT_DIM, dim)
        + positional_encoding
        → N TransformerDecoderLayers (self-attn only, causal mask)
        → linear head → vocab_size logits
    """

    def __init__(
        self,
        vocab_size: int = TOTAL_VOCAB,
        d_model: int = DEFAULT_DIM,
        nhead: int = DEFAULT_HEADS,
        num_layers: int = DEFAULT_LAYERS,
        max_seq_len: int = DEFAULT_MAX_SEQ,
        dropout: float = 0.1,
        context_dim: int = CONTEXT_DIM,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # Store so checkpoints can round-trip the schema version.
        # Default is module-level CONTEXT_DIM (35, v2 schema); v1 checkpoints
        # were trained with 34 and the loader infers from state_dict shape.
        self.context_dim = context_dim

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_TOKEN)
        self.context_proj = nn.Linear(context_dim, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer decoder layers (self-attention only)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # Output projection
        self.output_head = nn.Linear(d_model, VOCAB_SIZE)

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        tokens: torch.Tensor,       # (B, S)
        context: torch.Tensor,      # (B, S, CONTEXT_DIM)
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Returns logits of shape (B, S, VOCAB_SIZE).
        """
        B, S = tokens.shape
        device = tokens.device

        # Embeddings
        tok_emb = self.token_embedding(tokens)               # (B, S, D)
        ctx_emb = self.context_proj(context)                  # (B, S, D)
        pos_ids = torch.arange(S, device=device).unsqueeze(0) # (1, S)
        pos_ids = pos_ids.clamp(max=self.max_seq_len - 1)    # prevent OOB
        pos_emb = self.pos_embedding(pos_ids)                 # (1, S, D)

        x = tok_emb + ctx_emb + pos_emb  # (B, S, D)

        # Causal mask: True means *masked* for nn.TransformerEncoder
        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(
                S, device=device
            )

        # Padding mask: True means *ignore*
        padding_mask = (tokens == PAD_TOKEN)  # (B, S)

        x = self.transformer(
            x,
            mask=mask,
            src_key_padding_mask=padding_mask,
        )

        logits = self.output_head(x)  # (B, S, VOCAB_SIZE)
        return logits


# ═════════════════════════════════════════════════════════════════════════════
# BaseAnalyticsModel wrapper
# ═════════════════════════════════════════════════════════════════════════════

class PitchGPT(BaseAnalyticsModel):
    """Lifecycle wrapper around :class:`PitchGPTModel`."""

    def __init__(self) -> None:
        super().__init__()
        self.model: PitchGPTModel | None = None
        self._version = "1.0.0"

    @property
    def model_name(self) -> str:
        return "PitchGPT"

    @property
    def version(self) -> str:
        return self._version

    def train(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        return train_pitchgpt(conn, **kwargs)

    def predict(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        pitcher_id = kwargs.get("pitcher_id")
        season = kwargs.get("season")
        if pitcher_id is None:
            raise ValueError("pitcher_id is required for predict().")
        return calculate_predictability(conn, pitcher_id, season)

    def evaluate(
        self,
        conn: duckdb.DuckDBPyConnection,
        model_version: str = "1",
        **kwargs,
    ) -> dict:
        """Compute a PPS leaderboard summary for a season.

        Args:
            conn: DuckDB connection.
            model_version: Checkpoint version tag to evaluate (defaults to
                ``"1"`` so v1 behaviour is preserved). Pass ``"2"`` to
                evaluate the v2_ump checkpoint once trained.
            **kwargs: Forwarded options. Recognised: ``season`` (int).
        """
        season = kwargs.get("season")
        df = batch_calculate(conn, season=season, model_version=model_version)
        if df.empty:
            return {"qualifying_pitchers": 0, "mean_pps": 0.0}
        return {
            "qualifying_pitchers": len(df),
            "mean_pps": round(float(df["pps"].mean()), 4),
            "std_pps": round(float(df["pps"].std()), 4),
        }


# ═════════════════════════════════════════════════════════════════════════════
# Training
# ═════════════════════════════════════════════════════════════════════════════

def train_pitchgpt(
    conn: duckdb.DuckDBPyConnection,
    seasons: range | list[int] | None = None,
    val_seasons: list[int] | None = None,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH,
    lr: float = DEFAULT_LR,
    d_model: int = DEFAULT_DIM,
    nhead: int = DEFAULT_HEADS,
    num_layers: int = DEFAULT_LAYERS,
    max_seq_len: int = DEFAULT_MAX_SEQ,
    version: str = "1",
    max_games: int = 3000,
) -> dict:
    """Train PitchGPT end-to-end.

    Args:
        conn: DuckDB connection.
        seasons: Training seasons (default 2015-2024).
        val_seasons: Validation seasons (default [2025, 2026]).
        epochs: Training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer layers.
        max_seq_len: Maximum sequence length.
        version: Model version tag.
        max_games: Cap on training games to sample (default 3000, yielding
                   ~750-900K pitch tokens).  Prevents loading the full 7M+
                   row table into memory.

    Returns:
        Dictionary of training metrics.
    """
    if seasons is None:
        seasons = list(range(2015, 2025))
    if val_seasons is None:
        val_seasons = [2025, 2026]

    device = _get_device()

    logger.info("Loading training data (seasons %s, max_games=%d)...", list(seasons), max_games)
    train_ds = PitchSequenceDataset(conn, seasons=list(seasons), max_seq_len=max_seq_len, max_games=max_games)
    if len(train_ds) == 0:
        logger.warning("No training sequences found.")
        return {"status": "no_data", "train_loss": float("nan")}

    val_ds = PitchSequenceDataset(conn, seasons=val_seasons, max_seq_len=max_seq_len, max_games=500)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=_collate_fn,
    ) if len(val_ds) > 0 else None

    model = PitchGPTModel(
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    best_val_loss = float("inf")
    metrics_history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0

        for tokens, ctx, target in train_loader:
            tokens, ctx, target = (
                tokens.to(device), ctx.to(device), target.to(device),
            )
            logits = model(tokens, ctx)  # (B, S, V)
            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                target.reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Count non-pad tokens
            n_tok = (target != PAD_TOKEN).sum().item()
            total_loss += loss.item() * n_tok
            total_tokens += n_tok

        train_loss = total_loss / max(total_tokens, 1)

        # Validation
        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            v_loss, v_tok = 0.0, 0
            with torch.no_grad():
                for tokens, ctx, target in val_loader:
                    tokens, ctx, target = (
                        tokens.to(device), ctx.to(device), target.to(device),
                    )
                    logits = model(tokens, ctx)
                    loss = criterion(
                        logits.reshape(-1, VOCAB_SIZE),
                        target.reshape(-1),
                    )
                    n_tok = (target != PAD_TOKEN).sum().item()
                    v_loss += loss.item() * n_tok
                    v_tok += n_tok
            val_loss = v_loss / max(v_tok, 1)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        perplexity = math.exp(min(train_loss, 20))  # cap to avoid overflow
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4) if not math.isnan(val_loss) else None,
            "perplexity": round(perplexity, 2),
        }
        metrics_history.append(epoch_metrics)
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%s  ppl=%.2f",
            epoch, epochs, train_loss,
            f"{val_loss:.4f}" if not math.isnan(val_loss) else "N/A",
            perplexity,
        )

    # Save model
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_path = _MODEL_DIR / f"pitchgpt_v{version}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "max_seq_len": max_seq_len,
                "vocab_size": TOTAL_VOCAB,
            },
            "version": version,
        },
        save_path,
    )
    logger.info("Model saved to %s", save_path)

    final_metrics = {
        "status": "trained",
        "epochs": epochs,
        "final_train_loss": metrics_history[-1]["train_loss"],
        "final_val_loss": metrics_history[-1].get("val_loss"),
        "best_val_loss": round(best_val_loss, 4) if best_val_loss < float("inf") else None,
        "final_perplexity": metrics_history[-1]["perplexity"],
        "n_train_sequences": len(train_ds),
        "n_val_sequences": len(val_ds),
        "save_path": str(save_path),
        "history": metrics_history,
    }
    return final_metrics


# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════

def _load_model(version: str = "1") -> PitchGPTModel:
    """Load a trained PitchGPT checkpoint.

    Infers ``context_dim`` from the checkpoint's
    ``context_proj.weight`` shape so v1 (34) and v2 (35) both load.
    """
    device = _get_device()
    path = _MODEL_DIR / f"pitchgpt_v{version}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained PitchGPT model at {path}. Run train_pitchgpt() first."
        )
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    ctx_weight = checkpoint["model_state_dict"]["context_proj.weight"]
    context_dim = int(ctx_weight.shape[1])
    model = PitchGPTModel(
        vocab_size=cfg.get("vocab_size", TOTAL_VOCAB),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        context_dim=context_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Inference: shared query
# ═════════════════════════════════════════════════════════════════════════════

def _get_pitcher_game_sequences(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: int | None = None,
    game_pk: int | None = None,
) -> pd.DataFrame:
    """Fetch ordered pitches for a pitcher, optionally filtered by season/game."""
    filters = ["pitcher_id = $1", "pitch_type IS NOT NULL"]
    params: list = [pitcher_id]
    idx = 2

    if season is not None:
        filters.append(f"EXTRACT(YEAR FROM game_date) = ${idx}")
        params.append(season)
        idx += 1
    if game_pk is not None:
        filters.append(f"game_pk = ${idx}")
        params.append(game_pk)
        idx += 1

    where = " AND ".join(filters)
    # Extra columns (inning_topbot, events, delta_run_exp) power the
    # per-pitch score_diff reconstruction in ``_score_sequences``.
    query = f"""
        SELECT
            game_pk, pitcher_id, pitch_type, plate_x, plate_z,
            release_speed, balls, strikes, outs_when_up,
            on_1b, on_2b, on_3b, stand, inning,
            inning_topbot, events, delta_run_exp,
            at_bat_number, pitch_number
        FROM pitches
        WHERE {where}
        ORDER BY game_pk, at_bat_number, pitch_number
    """
    return conn.execute(query, params).fetchdf()


def _score_sequences(
    model: PitchGPTModel,
    df: pd.DataFrame,
) -> list[dict]:
    """Score each game sequence and return per-pitch NLL values.

    Returns a list of dicts, one per game:
        {"game_pk": ..., "per_pitch_nll": [...], "mean_nll": ..., "pitch_types": [...]}
    """
    device = next(model.parameters()).device
    results = []

    # Attach a per-pitch score_diff column up front.  The caller (see
    # ``_get_pitcher_game_sequences``) is responsible for sorting the
    # frame by ``(game_pk, at_bat_number, pitch_number)`` so this
    # reconstruction sees pitches in game order.  If the expected
    # columns are missing (e.g. older callers / tests), fall back to
    # zeros silently — the fix is best-effort for legacy call sites.
    if "_score_diff" not in df.columns:
        required = {"inning_topbot", "events", "delta_run_exp"}
        if required.issubset(df.columns):
            df = df.assign(_score_diff=_compute_per_pitch_score_diff(df))
        else:
            df = df.assign(_score_diff=0)

    # Per-pitch umpire-scalar lookup.  Legacy call sites that don't
    # supply ``_ump_scalar`` fall back to 0.0 (neutral), matching the
    # pre-2026-04-23 behaviour exactly.  Production callers should use
    # :class:`PitchSequenceDataset` which populates this column via the
    # umpire-table join.
    if "_ump_scalar" not in df.columns:
        df = df.assign(_ump_scalar=0.0)

    for game_pk, game_df in df.groupby("game_pk", sort=False):
        if len(game_df) < 2:
            continue

        tokens_list: list[int] = []
        contexts_list: list[torch.Tensor] = []
        pitch_types: list[str] = []

        for _, row in game_df.iterrows():
            tok = PitchTokenizer.encode(
                row.get("pitch_type"),
                row.get("plate_x"),
                row.get("plate_z"),
                row.get("release_speed"),
            )
            tokens_list.append(tok)
            pitch_types.append(str(row.get("pitch_type", "UN")))

            ctx = PitchTokenizer.encode_context(
                balls=_safe_int(row.get("balls"), 0),
                strikes=_safe_int(row.get("strikes"), 0),
                outs=_safe_int(row.get("outs_when_up"), 0),
                on_1b=_safe_bool(row.get("on_1b")),
                on_2b=_safe_bool(row.get("on_2b")),
                on_3b=_safe_bool(row.get("on_3b")),
                stand=_safe_str(row.get("stand"), "R"),
                inning=_safe_int(row.get("inning"), 1),
                score_diff=_safe_int(row.get("_score_diff"), 0),
            )
            ump_raw = row.get("_ump_scalar")
            try:
                ump_val = 0.0 if ump_raw is None else float(ump_raw)
            except (TypeError, ValueError):
                ump_val = 0.0
            contexts_list.append(
                PitchTokenizer.context_to_tensor(ctx, ump_scalar=ump_val)
            )

        # Truncate to model's max sequence length
        max_len = model.max_seq_len
        tokens_list = tokens_list[:max_len]
        contexts_list = contexts_list[:max_len]
        pitch_types = pitch_types[:max_len]

        if len(tokens_list) < 2:
            continue

        input_tokens = torch.tensor(tokens_list[:-1], dtype=torch.long).unsqueeze(0).to(device)
        target_tokens = torch.tensor(tokens_list[1:], dtype=torch.long).to(device)
        context_tensor = torch.stack(contexts_list[:-1]).unsqueeze(0).to(device)

        # Match the model's context schema: v1 (34) needs the ump scalar
        # sliced off; v2+ uses the full CONTEXT_DIM width.
        m_ctx_dim = getattr(model, "context_dim", CONTEXT_DIM)
        if context_tensor.size(-1) > m_ctx_dim:
            context_tensor = context_tensor[..., :m_ctx_dim]

        with torch.no_grad():
            logits = model(input_tokens, context_tensor)  # (1, S, V)
            logits = logits.squeeze(0)  # (S, V)

            # Per-pitch cross entropy
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            per_pitch_nll = []
            for i in range(len(target_tokens)):
                t = target_tokens[i].item()
                if 0 <= t < VOCAB_SIZE:
                    nll = -log_probs[i, t].item()
                else:
                    nll = 0.0
                per_pitch_nll.append(round(nll, 4))

        results.append({
            "game_pk": int(game_pk),
            "per_pitch_nll": per_pitch_nll,
            "mean_nll": round(float(np.mean(per_pitch_nll)), 4) if per_pitch_nll else 0.0,
            "pitch_types": pitch_types[1:],  # aligned with target
        })

    return results


# ═════════════════════════════════════════════════════════════════════════════
# Public inference API
# ═════════════════════════════════════════════════════════════════════════════

def calculate_predictability(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: int | None = None,
    model_version: str = "1",
) -> dict:
    """Pitch Predictability Score (PPS): average NLL across all games.

    Lower PPS (lower perplexity) = more predictable pitcher.
    Higher PPS = less predictable / harder to model.

    Returns:
        Dict with ``pitcher_id``, ``season``, ``pps`` (mean NLL),
        ``perplexity`` (exp(pps)), ``n_games``, ``n_pitches``,
        ``game_details`` (per-game breakdown).
    """
    model = _load_model(model_version)
    df = _get_pitcher_game_sequences(conn, pitcher_id, season=season)

    if df.empty:
        return {
            "pitcher_id": pitcher_id,
            "season": season,
            "pps": 0.0,
            "perplexity": 1.0,
            "n_games": 0,
            "n_pitches": 0,
            "game_details": [],
        }

    game_results = _score_sequences(model, df)

    all_nll = []
    for g in game_results:
        all_nll.extend(g["per_pitch_nll"])

    mean_nll = float(np.mean(all_nll)) if all_nll else 0.0
    perplexity = math.exp(min(mean_nll, 20))

    return {
        "pitcher_id": pitcher_id,
        "season": season,
        "pps": round(mean_nll, 4),
        "perplexity": round(perplexity, 2),
        "n_games": len(game_results),
        "n_pitches": len(all_nll),
        "game_details": game_results,
    }


def calculate_disruption_index(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    game_pk: int,
    model_version: str = "1",
) -> dict:
    """Per-pitch surprise scores for a specific game.

    Spikes in the returned NLL values indicate unexpected / disruptive
    pitch selections.

    Returns:
        Dict with ``pitcher_id``, ``game_pk``, ``per_pitch_surprise``,
        ``pitch_types``, ``mean_surprise``, ``max_surprise``.
    """
    model = _load_model(model_version)
    df = _get_pitcher_game_sequences(conn, pitcher_id, game_pk=game_pk)

    if df.empty or len(df) < 2:
        return {
            "pitcher_id": pitcher_id,
            "game_pk": game_pk,
            "per_pitch_surprise": [],
            "pitch_types": [],
            "mean_surprise": 0.0,
            "max_surprise": 0.0,
        }

    game_results = _score_sequences(model, df)

    if not game_results:
        return {
            "pitcher_id": pitcher_id,
            "game_pk": game_pk,
            "per_pitch_surprise": [],
            "pitch_types": [],
            "mean_surprise": 0.0,
            "max_surprise": 0.0,
        }

    result = game_results[0]
    surprise = result["per_pitch_nll"]

    return {
        "pitcher_id": pitcher_id,
        "game_pk": game_pk,
        "per_pitch_surprise": surprise,
        "pitch_types": result["pitch_types"],
        "mean_surprise": round(float(np.mean(surprise)), 4) if surprise else 0.0,
        "max_surprise": round(float(np.max(surprise)), 4) if surprise else 0.0,
    }


def calculate_predictability_by_catcher(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: int | None = None,
    model_version: str = "1",
) -> list[dict]:
    """Compute PPS grouped by catcher for a pitcher.

    For each catcher that caught games for this pitcher, computes a
    separate PPS score by filtering pitch sequences to only those games.

    Args:
        conn: DuckDB connection.
        pitcher_id: MLB pitcher ID.
        season: Optional season filter.
        model_version: Model checkpoint version.

    Returns:
        List of dicts, one per catcher, each containing:
        - catcher_id: int
        - catcher_name: str (from players table, or str(id))
        - games_caught: int
        - pps: float (mean NLL for games with this catcher)
        - perplexity: float
        - game_details: list of per-game results
    """
    model = _load_model(model_version)

    # Find distinct catchers for this pitcher
    season_filter = ""
    params: list = [pitcher_id]
    idx = 2
    if season is not None:
        season_filter = f"AND EXTRACT(YEAR FROM game_date) = ${idx}"
        params.append(season)
        idx += 1

    catchers_df = conn.execute(f"""
        SELECT DISTINCT fielder_2 AS catcher_id
        FROM pitches
        WHERE pitcher_id = $1
          AND fielder_2 IS NOT NULL
          {season_filter}
    """, params).fetchdf()

    if catchers_df.empty:
        return []

    # Get catcher names from players table
    catcher_ids = catchers_df["catcher_id"].tolist()
    names_map: dict[int, str] = {}
    try:
        if catcher_ids:
            ids_str = ", ".join(str(int(c)) for c in catcher_ids)
            names_df = conn.execute(f"""
                SELECT player_id, full_name FROM players
                WHERE player_id IN ({ids_str})
            """).fetchdf()
            for _, row in names_df.iterrows():
                names_map[int(row["player_id"])] = str(row["full_name"])
    except Exception:
        pass

    # For each catcher, get games and score them
    results: list[dict] = []
    for catcher_id in catcher_ids:
        catcher_id = int(catcher_id)

        # Get game_pks where this catcher caught for this pitcher
        game_params: list = [pitcher_id, catcher_id]
        gp_idx = 3
        game_season_filter = ""
        if season is not None:
            game_season_filter = f"AND EXTRACT(YEAR FROM game_date) = ${gp_idx}"
            game_params.append(season)

        games_df = conn.execute(f"""
            SELECT DISTINCT game_pk
            FROM pitches
            WHERE pitcher_id = $1
              AND fielder_2 = $2
              {game_season_filter}
        """, game_params).fetchdf()

        if games_df.empty:
            continue

        game_pks = games_df["game_pk"].tolist()

        # Fetch all pitch sequences for this pitcher in these specific games
        pks_str = ", ".join(str(int(g)) for g in game_pks)
        pitch_query = f"""
            SELECT
                game_pk, pitcher_id, pitch_type, plate_x, plate_z,
                release_speed, balls, strikes, outs_when_up,
                on_1b, on_2b, on_3b, stand, inning,
                inning_topbot, events, delta_run_exp,
                at_bat_number, pitch_number
            FROM pitches
            WHERE pitcher_id = $1
              AND pitch_type IS NOT NULL
              AND game_pk IN ({pks_str})
            ORDER BY game_pk, at_bat_number, pitch_number
        """
        pitches_df = conn.execute(pitch_query, [pitcher_id]).fetchdf()

        if pitches_df.empty or len(pitches_df) < 2:
            continue

        game_results = _score_sequences(model, pitches_df)

        all_nll: list[float] = []
        for g in game_results:
            all_nll.extend(g["per_pitch_nll"])

        if not all_nll:
            continue

        mean_nll = float(np.mean(all_nll))
        results.append({
            "catcher_id": catcher_id,
            "catcher_name": names_map.get(catcher_id, str(catcher_id)),
            "games_caught": len(game_results),
            "pps": round(mean_nll, 4),
            "perplexity": round(math.exp(min(mean_nll, 20)), 2),
            "n_pitches": len(all_nll),
            "game_details": game_results,
        })

    # Sort by PPS ascending (most predictable first)
    results.sort(key=lambda x: x["pps"])
    return results


def batch_calculate(
    conn: duckdb.DuckDBPyConnection,
    season: int | None = None,
    min_pitches: int = MIN_PITCHES_LEADERBOARD,
    model_version: str = "1",
) -> pd.DataFrame:
    """PPS leaderboard for all qualifying pitchers.

    Args:
        conn: DuckDB connection.
        season: Season filter.
        min_pitches: Minimum pitch count to qualify.
        model_version: Checkpoint version tag.

    Returns:
        DataFrame with columns: pitcher_id, pps, perplexity, n_games, n_pitches.
    """
    cols = ["pitcher_id", "pps", "perplexity", "n_games", "n_pitches"]

    try:
        model = _load_model(model_version)
    except FileNotFoundError:
        logger.warning("PitchGPT model not found. Train first via train_pitchgpt().")
        return pd.DataFrame(columns=cols)

    season_filter = ""
    params: list = []
    if season is not None:
        season_filter = "AND EXTRACT(YEAR FROM game_date) = $1"
        params = [season]

    qualifying_query = f"""
        SELECT pitcher_id, COUNT(*) AS pitch_count
        FROM pitches
        WHERE pitch_type IS NOT NULL
          {season_filter}
        GROUP BY pitcher_id
        HAVING COUNT(*) >= {int(min_pitches)}
    """
    qualifying = conn.execute(qualifying_query, params).fetchdf()

    if qualifying.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for _, row in qualifying.iterrows():
        pid = int(row["pitcher_id"])
        try:
            df = _get_pitcher_game_sequences(conn, pid, season=season)
            if df.empty or len(df) < 2:
                continue
            game_results = _score_sequences(model, df)
            all_nll = []
            for g in game_results:
                all_nll.extend(g["per_pitch_nll"])
            if not all_nll:
                continue
            mean_nll = float(np.mean(all_nll))
            rows.append({
                "pitcher_id": pid,
                "pps": round(mean_nll, 4),
                "perplexity": round(math.exp(min(mean_nll, 20)), 2),
                "n_games": len(game_results),
                "n_pitches": len(all_nll),
            })
        except Exception as exc:
            logger.warning("PPS failed for pitcher %d: %s", pid, exc)

    if not rows:
        return pd.DataFrame(columns=cols)

    result_df = pd.DataFrame(rows)
    result_df = result_df.sort_values("pps", ascending=True).reset_index(drop=True)
    return result_df
