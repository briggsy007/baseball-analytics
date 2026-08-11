"""
PitchGPT v3 (WS5.2 factorized retrain) — cached, vectorised sequence loader.

Spec: ``docs/pitchgpt_sim_engine/PITCHGPT_V2_SPEC.md`` (FROZEN at
``b61e05b``, body sha256 ``b19b54d9…8bd6d5b``).

Why this module exists
----------------------
:class:`src.analytics.pitchgpt.PitchSequenceDataset` builds its tensors with a
per-row Python loop (``df.apply`` for the umpire scalar, ``iterrows`` for
tokenisation).  At the §3.2-fixed 10K-game budget that is ~2.9M rows and the
documented end-to-end cost of the frozen-v2 run was ~4h25m, of which only
506.8s was GPU (checkpoint ``training_meta.total_train_sec``).  The v2 program
needs the *same* rows several times (Stage A, Stage B, the 2023 fit cohort, the
2024 dev cohort), so this module:

1. issues the **identical SQL** and reuses
   :func:`~src.analytics.pitchgpt._compute_per_pitch_score_diff` verbatim,
2. vectorises the umpire-scalar join and the tokenisation/context encoding
   (pure arithmetic — bit-equivalence with the reference path is asserted by
   ``tests/test_pitchgpt_v3.py::test_cache_matches_reference_dataset``),
3. caches the result as compact integer arrays (categorical context indices,
   not the 35-wide one-hot) so a rebuild is never needed inside the program.

Nothing here mutates the v1/v2 code paths.

Data policy (spec §5, absolute)
-------------------------------
``assert_season_policy`` refuses any request touching 2025 (budgeted tier,
12/14 contacts spent) or 2026 (sealed lockbox).  Every loader call goes
through it, and the cache manifest records the seasons actually read.

Storage layout
--------------
Flat per-pitch arrays plus a sequence pointer, so a sequence *i* is
``slice(seq_ptr[i], seq_ptr[i + 1])``:

======================  =======  ==========================================
array                   dtype    meaning
======================  =======  ==========================================
``tok``                 int32    composite pitch token (0 … 2209)
``ctx_idx``             int8     (N, 6) categorical context indices
``ump``                 float32  umpire accuracy-above-x scalar
``abn``                 int32    at_bat_number (PA segmentation for Stage B)
``outc``                int8     7-class outcome, -1 = unlabelled
``pa_pos``              int8     within-PA pitch index (0-based)
``seq_ptr``             int64    (n_seq + 1,) sequence boundaries
``game_pk``             int64    (n_seq,)
``pitcher_id``          int64    (n_seq,)
======================  =======  ==========================================
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
import torch

from src.analytics.pitchgpt import (
    CONTEXT_DIM,
    NUM_COUNT_STATES,
    NUM_INNING_BUCKETS,
    NUM_OUTS,
    NUM_RUNNER_STATES,
    NUM_BATTER_HANDS,
    NUM_VELO_BUCKETS,
    NUM_ZONES,
    PITCH_TYPE_MAP,
    NUM_PITCH_TYPES,
    _compute_per_pitch_score_diff,
    _fetch_hp_umpire_by_game,
    _fetch_prior_season_ump_tendency,
    _fetch_season_league_median,
)
from src.analytics.pitchgpt_outcome_head import classify_pitch_outcome

logger = logging.getLogger(__name__)

# ── Data policy (spec §5) ────────────────────────────────────────────────────
#: Seasons this program may never read.  2025 = budgeted tier at 12/14 spent
#: with both remaining contacts reserved (§5.4); 2026 = sealed lockbox, one
#: contact at season end only (§5.5).
BUDGETED_SEASON: int = 2025
LOCKBOX_SEASON: int = 2026
FORBIDDEN_SEASONS: frozenset[int] = frozenset({BUDGETED_SEASON, LOCKBOX_SEASON})

#: §5.1 training window — [FIXED].
TRAIN_SEASONS: tuple[int, ...] = (2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022)
#: §5.2 model-selection / calibration slice.
FIT_SEASON: int = 2023
#: §5.3 burned dev tier.
DEV_SEASON: int = 2024

#: Context one-hot offsets, mirroring ``PitchTokenizer.context_to_tensor``.
_CTX_OFFSETS: tuple[int, ...] = (
    0,
    NUM_COUNT_STATES,
    NUM_COUNT_STATES + NUM_OUTS,
    NUM_COUNT_STATES + NUM_OUTS + NUM_RUNNER_STATES,
    NUM_COUNT_STATES + NUM_OUTS + NUM_RUNNER_STATES + NUM_BATTER_HANDS,
    NUM_COUNT_STATES + NUM_OUTS + NUM_RUNNER_STATES + NUM_BATTER_HANDS
    + NUM_INNING_BUCKETS,
)


class SeasonPolicyError(RuntimeError):
    """Raised when a loader call would read a forbidden season."""


def assert_season_policy(
    seasons: Iterable[int], *, allow_lockbox: bool = False
) -> list[int]:
    """Refuse 2025/2026 reads outright (spec §5.4, §5.5).

    ``allow_lockbox=True`` is reserved for the §5.5 grading run and is only
    ever passed by ``scripts/pitchgpt_v3_lockbox_2026.py``, whose entry point
    is *independently* gated by ``@holdout_access`` on the still-sealed
    ``lockbox_2026_full_season`` tier.  2025 is never permitted by this
    function under any flag — spending a 2025 contact requires its own dated
    amendment (§5.4) and is out of scope for the whole v2 program.

    Returns the sorted, de-duplicated season list on success.
    """
    resolved = sorted({int(s) for s in seasons})
    forbidden = FORBIDDEN_SEASONS - (
        frozenset({LOCKBOX_SEASON}) if allow_lockbox else frozenset()
    )
    bad = sorted(set(resolved) & forbidden)
    if bad:
        raise SeasonPolicyError(
            f"PITCHGPT_V2_SPEC §5 forbids reading {bad}: 2025 is the budgeted "
            f"tier (12/14 contacts spent, both remaining reserved) and 2026 is "
            f"the sealed lockbox (one contact at season end). Requested "
            f"seasons: {resolved}."
        )
    ceiling = LOCKBOX_SEASON if allow_lockbox else DEV_SEASON
    if any(s > ceiling for s in resolved):
        raise SeasonPolicyError(
            f"Seasons beyond {ceiling} are out of scope: {resolved}."
        )
    return resolved


# ── Vectorised tokenisation ──────────────────────────────────────────────────


def _vec_pitch_type_idx(pitch_type: pd.Series) -> np.ndarray:
    """Vectorised :meth:`PitchTokenizer.pitch_type_to_idx`."""
    mapped = pitch_type.map(PITCH_TYPE_MAP)
    return mapped.fillna(NUM_PITCH_TYPES - 1).to_numpy(dtype=np.int64)


def _vec_zone(plate_x: np.ndarray, plate_z: np.ndarray) -> np.ndarray:
    """Vectorised :meth:`PitchTokenizer.location_to_zone`."""
    x_edges = np.linspace(-1.5, 1.5, 6)
    z_edges = np.linspace(1.0, 4.0, 6)
    missing = np.isnan(plate_x) | np.isnan(plate_z)
    x_safe = np.where(missing, 0.0, plate_x)
    z_safe = np.where(missing, 0.0, plate_z)
    x_bin = np.clip(np.digitize(x_safe, x_edges) - 1, 0, 4)
    z_bin = np.clip(np.digitize(z_safe, z_edges) - 1, 0, 4)
    zone = z_bin * 5 + x_bin
    return np.where(missing, NUM_ZONES - 1, zone).astype(np.int64)


def _vec_velo_bucket(release_speed: np.ndarray) -> np.ndarray:
    """Vectorised :meth:`PitchTokenizer.velocity_to_bucket`."""
    v = release_speed
    out = np.full(v.shape, 2, dtype=np.int64)
    known = ~np.isnan(v)
    out = np.where(known & (v < 80), 0, out)
    out = np.where(known & (v >= 80) & (v < 85), 1, out)
    out = np.where(known & (v >= 85) & (v < 90), 2, out)
    out = np.where(known & (v >= 90) & (v < 95), 3, out)
    out = np.where(known & (v >= 95), 4, out)
    return out


def _vec_safe_int(series: pd.Series, default: int) -> np.ndarray:
    return (
        pd.to_numeric(series, errors="coerce")
        .fillna(default)
        .astype(np.int64)
        .to_numpy()
    )


def _vec_safe_bool(series: pd.Series) -> np.ndarray:
    """Vectorised ``_safe_bool`` — NA/0 are False, any other int is True."""
    num = pd.to_numeric(series, errors="coerce")
    return (num.notna() & (num != 0)).to_numpy()


_SAC_EVENTS_V = frozenset({"sac_fly", "sac_fly_double_play"})


def vectorised_score_diff(df: pd.DataFrame) -> np.ndarray:
    """Vectorised replica of :func:`_compute_per_pitch_score_diff`.

    The reference walks rows in order keeping a per-``game_pk`` running score.
    Runs credited by a row depend only on that row (event, pre-pitch runners,
    ``delta_run_exp``), so the running score is an *exclusive* cumulative sum
    within each game — exactly what this computes.  Equivalence with the
    reference implementation is asserted in ``tests/test_pitchgpt_v3.py``.
    """
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=np.int32)

    ev_series = df["events"]
    is_null_ev = ev_series.isna().to_numpy()
    ev = ev_series.fillna("").astype(str).str.lower().to_numpy()
    on1 = _vec_safe_bool(df["on_1b"])
    on2 = _vec_safe_bool(df["on_2b"])
    on3 = _vec_safe_bool(df["on_3b"])
    dre = pd.to_numeric(df["delta_run_exp"], errors="coerce").to_numpy(dtype=np.float64)
    dre_ok = ~np.isnan(dre)

    runs = np.zeros(n, dtype=np.int64)
    is_hr = ev == "home_run"
    runs = np.where(is_hr, 1 + on1.astype(np.int64) + on2.astype(np.int64) + on3.astype(np.int64), runs)
    is_sac = np.isin(ev, list(_SAC_EVENTS_V))
    runs = np.where(~is_hr & is_sac, 1, runs)

    general = (~is_hr) & (~is_sac) & dre_ok & (dre > 0.45) & on3
    bonus = (
        (dre_ok & (dre > 1.3) & on2).astype(np.int64)
        + (dre_ok & (dre > 2.0) & on1).astype(np.int64)
    )
    runs = np.where(general, 1 + bonus, runs)
    # Reference short-circuits on ``event is None`` before any branch.
    runs = np.where(is_null_ev, 0, runs)

    tb_series = df["inning_topbot"]
    tb_null = tb_series.isna().to_numpy()
    tb = tb_series.fillna("").astype(str).str.lower()
    is_top = (tb.str.startswith("top") & ~pd.Series(tb_null, index=tb.index)).to_numpy()
    is_bot = (tb.str.startswith("bot") & ~pd.Series(tb_null, index=tb.index)).to_numpy()

    away_runs = np.where(is_top, runs, 0)
    home_runs = np.where(is_top, 0, runs)

    gpk = pd.to_numeric(df["game_pk"], errors="coerce").fillna(-1).to_numpy(np.int64)
    frame = pd.DataFrame({"g": gpk, "h": home_runs, "a": away_runs})
    grp = frame.groupby("g", sort=False)
    home_before = (grp["h"].cumsum() - frame["h"]).to_numpy(np.int64)
    away_before = (grp["a"].cumsum() - frame["a"]).to_numpy(np.int64)

    diff = home_before - away_before
    return np.where(is_bot, -diff, diff).astype(np.int32)


def _vec_context_indices(df: pd.DataFrame) -> np.ndarray:
    """Vectorised :meth:`PitchTokenizer.encode_context` → (N, 6) int8."""
    balls = np.minimum(_vec_safe_int(df["balls"], 0), 3)
    strikes = np.minimum(_vec_safe_int(df["strikes"], 0), 2)
    count_state = balls * 3 + strikes

    outs_idx = np.minimum(_vec_safe_int(df["outs_when_up"], 0), 2)

    runner_state = (
        _vec_safe_bool(df["on_1b"]).astype(np.int64) * 4
        + _vec_safe_bool(df["on_2b"]).astype(np.int64) * 2
        + _vec_safe_bool(df["on_3b"]).astype(np.int64)
    )

    stand = df["stand"].astype("object")
    stand_str = stand.map(
        lambda v: "R" if v is None or str(v) in ("", "nan", "None", "<NA>") else str(v)
    )
    batter_hand = np.where(stand_str.to_numpy() == "L", 0, 1).astype(np.int64)

    inning = _vec_safe_int(df["inning"], 1)
    inning_bucket = np.select(
        [inning <= 3, inning <= 6, inning <= 9],
        [0, 1, 2],
        default=3,
    ).astype(np.int64)

    sd = _vec_safe_int(df["_score_diff"], 0)
    score_bucket = np.select(
        [sd <= -4, sd < 0, sd == 0, sd <= 3],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.int64)

    return np.stack(
        [count_state, outs_idx, runner_state, batter_hand, inning_bucket, score_bucket],
        axis=1,
    ).astype(np.int8)


def context_indices_to_tensor(
    ctx_idx: np.ndarray | torch.Tensor,
    ump: np.ndarray | torch.Tensor,
    context_dim: int = CONTEXT_DIM,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Expand ``(…, 6)`` categorical indices + ump scalar to the 35-wide one-hot.

    Exactly reproduces :meth:`PitchTokenizer.context_to_tensor`, batched.
    Accepts numpy arrays or (possibly CUDA-resident) torch tensors; the latter
    stay on their device, which keeps the training loops off the host round
    trip.
    """
    idx_t = (
        ctx_idx.to(torch.long)
        if isinstance(ctx_idx, torch.Tensor)
        else torch.as_tensor(np.asarray(ctx_idx), dtype=torch.long)
    )
    ump_t = (
        ump.to(torch.float32)
        if isinstance(ump, torch.Tensor)
        else torch.as_tensor(np.asarray(ump), dtype=torch.float32)
    )
    if ump_t.device != idx_t.device:
        ump_t = ump_t.to(idx_t.device)
    if device is not None:
        idx_t = idx_t.to(device)
        ump_t = ump_t.to(device)
    lead = idx_t.shape[:-1]
    out = torch.zeros(*lead, context_dim, dtype=torch.float32, device=idx_t.device)
    for j, off in enumerate(_CTX_OFFSETS):
        out.scatter_(-1, (idx_t[..., j] + off).unsqueeze(-1), 1.0)
    out[..., context_dim - 1] = ump_t
    return out


# ── Cache building ───────────────────────────────────────────────────────────


def _ump_scalar_vectorised(
    conn: duckdb.DuckDBPyConnection, df: pd.DataFrame, seasons: list[int]
) -> np.ndarray:
    """Vectorised replica of ``PitchSequenceDataset._load``'s ``_ump_scalar``."""
    ump_by_game = _fetch_hp_umpire_by_game(conn, seasons)
    tendency_map = _fetch_prior_season_ump_tendency(conn, seasons)
    median_by_season = _fetch_season_league_median(conn, seasons)

    game_season = df["game_date"].dt.year.astype("Int64")
    medians = game_season.map(median_by_season).astype(float).fillna(0.0).to_numpy()

    gpk = pd.to_numeric(df["game_pk"], errors="coerce").astype("Int64")
    ump_name = gpk.map(ump_by_game)

    key = pd.Series(list(zip(ump_name, game_season)), index=df.index)
    vals = key.map(tendency_map)
    return vals.astype(float).fillna(pd.Series(medians, index=df.index)).to_numpy(
        dtype=np.float32
    )


_SEQ_SQL = """
    SELECT
        game_pk, pitcher_id, pitch_type, plate_x, plate_z, release_speed,
        balls, strikes, outs_when_up, on_1b, on_2b, on_3b, stand, inning,
        inning_topbot, events, description, delta_run_exp,
        at_bat_number, pitch_number, game_date
    FROM pitches
    WHERE pitch_type IS NOT NULL
      {season_filter}
      {pitcher_exclude_filter}
      {game_filter}
    ORDER BY game_pk, at_bat_number, pitch_number
"""


def build_sequence_cache(
    conn: duckdb.DuckDBPyConnection,
    *,
    seasons: Iterable[int],
    max_games: int | None,
    sample_seed: int,
    exclude_pitcher_ids: set[int] | None = None,
    max_seq_len: int = 256,
    cache_path: Path | None = None,
    game_pks: Iterable[int] | None = None,
    allow_lockbox: bool = False,
) -> dict:
    """Build (or load) the compact sequence cache for ``seasons``.

    ``max_games`` uses DuckDB reservoir sampling **with an explicit seed** so
    the game draw is reproducible; the frozen-v2 run's unseeded
    ``USING SAMPLE n ROWS`` was not (deviations log §9 entry 3).
    """
    season_list = assert_season_policy(seasons, allow_lockbox=allow_lockbox)

    if cache_path is not None and Path(cache_path).exists():
        logger.info("Loading sequence cache from %s", cache_path)
        return load_sequence_cache(Path(cache_path))

    t0 = time.perf_counter()
    s_str = ", ".join(str(s) for s in season_list)
    season_filter = f"AND EXTRACT(YEAR FROM game_date) IN ({s_str})"

    pitcher_exclude_filter = ""
    if exclude_pitcher_ids:
        pids = ", ".join(str(int(p)) for p in sorted(exclude_pitcher_ids))
        pitcher_exclude_filter = f"AND pitcher_id NOT IN ({pids})"

    game_filter = ""
    if game_pks is not None:
        # Explicit game list (equivalence tests / exact replay).  Bypasses the
        # reservoir sampler entirely.
        gl = ", ".join(str(int(g)) for g in sorted(set(game_pks)))
        game_filter = f"AND game_pk IN ({gl})"
    elif max_games is not None:
        game_filter = f"""
            AND game_pk IN (
                SELECT game_pk FROM (
                    SELECT DISTINCT game_pk
                    FROM pitches
                    WHERE pitch_type IS NOT NULL {season_filter}
                      {pitcher_exclude_filter}
                ) USING SAMPLE {int(max_games)} ROWS (reservoir, {int(sample_seed)})
            )
        """

    sql = _SEQ_SQL.format(
        season_filter=season_filter,
        pitcher_exclude_filter=pitcher_exclude_filter,
        game_filter=game_filter,
    )
    df = conn.execute(sql).fetchdf()
    t_sql = time.perf_counter() - t0
    logger.info("SQL returned %d rows in %.1fs", len(df), t_sql)
    if df.empty:
        raise RuntimeError(f"No pitches for seasons={season_list}.")

    t1 = time.perf_counter()
    df = df.assign(_score_diff=vectorised_score_diff(df))
    logger.info("score_diff reconstruction: %.1fs", time.perf_counter() - t1)

    t2 = time.perf_counter()
    ump = _ump_scalar_vectorised(conn, df, season_list)
    logger.info("ump scalar join: %.1fs", time.perf_counter() - t2)

    t3 = time.perf_counter()
    pt_idx = _vec_pitch_type_idx(df["pitch_type"])
    zone = _vec_zone(
        pd.to_numeric(df["plate_x"], errors="coerce").to_numpy(dtype=np.float64),
        pd.to_numeric(df["plate_z"], errors="coerce").to_numpy(dtype=np.float64),
    )
    velo = _vec_velo_bucket(
        pd.to_numeric(df["release_speed"], errors="coerce").to_numpy(dtype=np.float64)
    )
    tok = (pt_idx * (NUM_ZONES * NUM_VELO_BUCKETS) + zone * NUM_VELO_BUCKETS + velo)
    ctx_idx = _vec_context_indices(df)
    logger.info("tokenisation: %.1fs", time.perf_counter() - t3)

    t4 = time.perf_counter()
    outc = np.fromiter(
        (
            classify_pitch_outcome(d, e)
            for d, e in zip(df["description"].to_numpy(), df["events"].to_numpy())
        ),
        dtype=np.int64,
        count=len(df),
    )
    logger.info("outcome labelling: %.1fs", time.perf_counter() - t4)

    abn = pd.to_numeric(df["at_bat_number"], errors="coerce").fillna(-1).to_numpy(
        dtype=np.int64
    )
    pn = pd.to_numeric(df["pitch_number"], errors="coerce").fillna(-1).to_numpy(
        dtype=np.int64
    )

    # ── Sequence grouping: (game_pk, pitcher_id), first-appearance order ──
    gp = pd.to_numeric(df["game_pk"], errors="coerce").fillna(-1).to_numpy(np.int64)
    pid = pd.to_numeric(df["pitcher_id"], errors="coerce").fillna(-1).to_numpy(np.int64)
    _, codes = np.unique(np.stack([gp, pid], axis=1), axis=0, return_inverse=True)
    codes = np.asarray(codes).ravel()

    order = np.argsort(codes, kind="stable")
    codes_sorted = codes[order]
    boundaries = np.flatnonzero(np.diff(codes_sorted)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(codes_sorted)]])

    keep_tok, keep_ctx, keep_ump = [], [], []
    keep_abn, keep_outc, keep_pn = [], [], []
    seq_lens, seq_game, seq_pitcher = [], [], []
    for s, e in zip(starts, ends):
        idxs = order[s:e]
        if len(idxs) < 2:
            continue  # reference path: need at least 2 pitches
        idxs = idxs[:max_seq_len]
        keep_tok.append(tok[idxs])
        keep_ctx.append(ctx_idx[idxs])
        keep_ump.append(ump[idxs])
        keep_abn.append(abn[idxs])
        keep_outc.append(outc[idxs])
        keep_pn.append(pn[idxs])
        seq_lens.append(len(idxs))
        seq_game.append(gp[idxs[0]])
        seq_pitcher.append(pid[idxs[0]])

    if not seq_lens:
        raise RuntimeError("No sequences survived the >=2-pitch filter.")

    tok_flat = np.concatenate(keep_tok).astype(np.int32)
    ctx_flat = np.concatenate(keep_ctx).astype(np.int8)
    ump_flat = np.concatenate(keep_ump).astype(np.float32)
    abn_flat = np.concatenate(keep_abn).astype(np.int32)
    outc_flat = np.concatenate(keep_outc).astype(np.int8)
    pn_flat = np.concatenate(keep_pn).astype(np.int32)
    seq_ptr = np.concatenate([[0], np.cumsum(seq_lens)]).astype(np.int64)

    # within-PA position = pitch_number - min(pitch_number) inside each
    # (sequence, at_bat_number) block.  Sequences are already in pitch order.
    pa_pos = np.zeros_like(pn_flat)
    pa_key = np.stack([np.repeat(np.arange(len(seq_lens)), seq_lens), abn_flat], axis=1)
    changed = np.ones(len(pa_key), dtype=bool)
    changed[1:] = np.any(pa_key[1:] != pa_key[:-1], axis=1)
    grp_start_idx = np.maximum.accumulate(np.where(changed, np.arange(len(pa_key)), 0))
    pa_pos = (pn_flat - pn_flat[grp_start_idx]).astype(np.int8)

    payload = {
        "tok": tok_flat,
        "ctx_idx": ctx_flat,
        "ump": ump_flat,
        "abn": abn_flat,
        "outc": outc_flat,
        "pa_pos": pa_pos,
        "seq_ptr": seq_ptr,
        "game_pk": np.asarray(seq_game, dtype=np.int64),
        "pitcher_id": np.asarray(seq_pitcher, dtype=np.int64),
        "meta": {
            "seasons": season_list,
            "max_games": max_games,
            "sample_seed": int(sample_seed),
            "n_excluded_pitchers": len(exclude_pitcher_ids or ()),
            "max_seq_len": int(max_seq_len),
            "n_rows_sql": int(len(df)),
            "n_sequences": int(len(seq_lens)),
            "n_pitches_kept": int(len(tok_flat)),
            "n_games": int(len(set(seq_game))),
            "n_pitchers": int(len(set(seq_pitcher))),
            "build_sec": round(time.perf_counter() - t0, 1),
            "game_pk_sha256": hashlib.sha256(
                ",".join(str(int(g)) for g in sorted(set(seq_game))).encode()
            ).hexdigest(),
        },
    }
    logger.info(
        "cache built: %d sequences / %d pitches / %d games in %.1fs",
        payload["meta"]["n_sequences"],
        payload["meta"]["n_pitches_kept"],
        payload["meta"]["n_games"],
        payload["meta"]["build_sec"],
    )
    if cache_path is not None:
        save_sequence_cache(payload, Path(cache_path))
    return payload


def save_sequence_cache(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in payload.items() if k != "meta"}
    np.savez_compressed(path, **arrays, meta_json=np.array(json.dumps(payload["meta"])))
    logger.info("cache saved to %s", path)


def load_sequence_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files if k != "meta_json"}
        out["meta"] = json.loads(str(z["meta_json"]))
    return out


_PA_TERMINAL_SQL = """
    WITH season_rows AS (
        SELECT game_pk, at_bat_number, pitcher_id, batter_id, pitch_number,
               events, woba_value, description
        FROM pitches
        WHERE EXTRACT(YEAR FROM game_date) = {season}
          AND pitch_type IS NOT NULL
          AND pitcher_id IS NOT NULL
          AND batter_id IS NOT NULL
          AND at_bat_number IS NOT NULL
          AND pitch_number IS NOT NULL
          {pitcher_exclude_filter}
    ),
    pa_groups AS (
        SELECT game_pk, at_bat_number, pitcher_id, batter_id,
               MIN(pitch_number) AS first_pn,
               MAX(pitch_number) AS last_pn,
               COUNT(*)          AS pa_pitch_count
        FROM season_rows
        GROUP BY game_pk, at_bat_number, pitcher_id, batter_id
    )
    SELECT g.game_pk, g.at_bat_number, g.pitcher_id, g.batter_id,
           g.pa_pitch_count,
           p.events     AS terminal_event,
           p.woba_value AS terminal_woba,
           p.description AS terminal_description
    FROM pa_groups g
    JOIN season_rows p
      ON p.game_pk = g.game_pk
     AND p.at_bat_number = g.at_bat_number
     AND p.pitcher_id = g.pitcher_id
     AND p.batter_id = g.batter_id
     AND p.pitch_number = g.last_pn
    WHERE p.events IS NOT NULL AND p.events != ''
"""


def fetch_pa_terminal_facts(
    conn: duckdb.DuckDBPyConnection,
    season: int,
    exclude_pitcher_ids: set[int] | None = None,
    *,
    allow_lockbox: bool = False,
) -> pd.DataFrame:
    """One row per PA with its terminal event / wOBA / pitch count.

    Same SQL shape as ``PHASE_0.6_PLAN`` §3.3 (via
    ``scripts/pitchgpt_rollout_sanity_2025.py::_fetch_pitcher_disjoint_pa_starts``),
    parameterised by season and gated by :func:`assert_season_policy`.  Used to
    build the empirical K%/BB%/HR%/HBP%/hit% / wOBA / PA-length baselines the
    §6.4 marginal arm is graded against.
    """
    assert_season_policy([season], allow_lockbox=allow_lockbox)
    excl = ""
    if exclude_pitcher_ids:
        pids = ", ".join(str(int(p)) for p in sorted(exclude_pitcher_ids))
        excl = f"AND pitcher_id NOT IN ({pids})"
    return conn.execute(
        _PA_TERMINAL_SQL.format(season=int(season), pitcher_exclude_filter=excl)
    ).fetchdf()


def fetch_train_pitcher_ids(
    conn: duckdb.DuckDBPyConnection, seasons: Iterable[int] = TRAIN_SEASONS
) -> set[int]:
    """Distinct pitcher_ids in ``seasons`` — the pitcher-disjoint exclusion set."""
    season_list = assert_season_policy(seasons)
    s_str = ", ".join(str(s) for s in season_list)
    rows = conn.execute(
        f"""
        SELECT DISTINCT pitcher_id FROM pitches
        WHERE pitch_type IS NOT NULL AND pitcher_id IS NOT NULL
          AND EXTRACT(YEAR FROM game_date) IN ({s_str})
        """
    ).fetchdf()
    return {int(p) for p in rows["pitcher_id"].tolist() if p is not None}


# ── Torch dataset over the cache ─────────────────────────────────────────────


class CachedSequenceDataset(torch.utils.data.Dataset):
    """``(input_tokens, context_idx, ump, target_tokens)`` per game-sequence.

    Mirrors :class:`PitchSequenceDataset`'s contract (input = ``tokens[:-1]``,
    target = ``tokens[1:]``, context aligned with the input) but keeps the
    context categorical until collate time.
    """

    def __init__(self, cache: dict) -> None:
        self.cache = cache
        self.seq_ptr = cache["seq_ptr"]
        self.n = len(self.seq_ptr) - 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        a, b = int(self.seq_ptr[i]), int(self.seq_ptr[i + 1])
        tok = torch.from_numpy(self.cache["tok"][a:b].astype(np.int64))
        ctx = torch.from_numpy(self.cache["ctx_idx"][a:b].astype(np.int64))
        ump = torch.from_numpy(self.cache["ump"][a:b].astype(np.float32))
        return tok[:-1], ctx[:-1], ump[:-1], tok[1:]


def build_pa_dataset(cache: dict, horizon: int = 6) -> dict:
    """Slice the game-sequence cache into PA-scoped tensors for Stage B.

    Inference (``pitchgpt_sim.rollout`` / ``rollout_v3``) starts a PA from
    ``BOS`` with the PA-start context and mutates **only** the ``count_state``
    one-hot as the count advances; every other context field is treated as
    PA-invariant.  Stage B mirrors that construction exactly so "the
    training-time rollout is the inference-time rollout" (spec §4.3): the
    PA-start row supplies the invariant fields and each position's real
    ``count_state`` supplies the mutating one.

    Returns padded ``(n_pa, horizon)`` arrays plus the per-PA length.  PAs are
    truncated at ``horizon`` (§4.3.1); PAs whose first pitch is missing from the
    cache (sequence-boundary truncation) are dropped.
    """
    seq_ptr = cache["seq_ptr"]
    n_seq = len(seq_ptr) - 1
    seq_idx = np.repeat(np.arange(n_seq), np.diff(seq_ptr))
    abn = cache["abn"]
    pa_pos = cache["pa_pos"]

    key = np.stack([seq_idx, abn], axis=1)
    new_block = np.ones(len(abn), dtype=bool)
    new_block[1:] = np.any(key[1:] != key[:-1], axis=1)
    starts = np.flatnonzero(new_block)
    ends = np.concatenate([starts[1:], [len(abn)]])

    tok = cache["tok"]
    ctx_idx = cache["ctx_idx"]
    ump = cache["ump"]
    outc = cache["outc"]

    keep = pa_pos[starts] == 0
    starts, ends = starts[keep], ends[keep]
    n_pa = len(starts)

    pa_tok = np.full((n_pa, horizon), -1, dtype=np.int32)
    pa_ctx = np.zeros((n_pa, horizon, 6), dtype=np.int8)
    pa_outc = np.full((n_pa, horizon), -1, dtype=np.int8)
    pa_len = np.zeros(n_pa, dtype=np.int8)
    pa_ump = np.zeros(n_pa, dtype=np.float32)
    pa_seq = np.zeros(n_pa, dtype=np.int64)
    pa_abn = np.zeros(n_pa, dtype=np.int64)

    for i, (a, b) in enumerate(zip(starts, ends)):
        n = min(int(b - a), horizon)
        sl = slice(a, a + n)
        pa_tok[i, :n] = tok[sl]
        # PA-invariant fields from the PA-start row; count_state per position.
        pa_ctx[i, :n] = ctx_idx[a]
        pa_ctx[i, :n, 0] = ctx_idx[sl, 0]
        pa_outc[i, :n] = outc[sl]
        pa_len[i] = n
        pa_ump[i] = ump[a]
        pa_seq[i] = seq_idx[a]
        pa_abn[i] = abn[a]

    return {
        "pa_tok": pa_tok,
        "pa_ctx_idx": pa_ctx,
        "pa_outc": pa_outc,
        "pa_len": pa_len,
        "pa_ump": pa_ump,
        "pa_seq": pa_seq,
        "pa_abn": pa_abn,
        # Join key back to the database: one row per (game_pk, at_bat_number).
        "pa_game_pk": np.asarray(cache["game_pk"])[pa_seq],
        "pa_pitcher_id": np.asarray(cache["pitcher_id"])[pa_seq],
        "meta": {
            **cache["meta"],
            "n_pas": int(n_pa),
            "horizon": int(horizon),
            "pa_scope": (
                "PA-start invariant context + per-position count_state, "
                "matching the rollout engine (spec §4.3)"
            ),
        },
    }


def cached_collate(batch, pad_token: int):
    """Pad to the batch's longest sequence; PAD tokens carry zero context."""
    max_len = max(item[0].size(0) for item in batch)
    toks, ctxs, umps, tgts = [], [], [], []
    for tok, ctx, ump, tgt in batch:
        pad = max_len - tok.size(0)
        toks.append(
            torch.cat([tok, torch.full((pad,), pad_token, dtype=torch.long)])
        )
        tgts.append(
            torch.cat([tgt, torch.full((pad,), pad_token, dtype=torch.long)])
        )
        ctxs.append(torch.cat([ctx, torch.zeros(pad, ctx.size(-1), dtype=torch.long)]))
        umps.append(torch.cat([ump, torch.zeros(pad, dtype=torch.float32)]))
    return (
        torch.stack(toks),
        torch.stack(ctxs),
        torch.stack(umps),
        torch.stack(tgts),
    )
