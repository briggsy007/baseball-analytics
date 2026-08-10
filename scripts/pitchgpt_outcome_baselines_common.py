"""
Shared cohort + feature-extraction utilities for Plan B baselines (A3/A4/A5).

Provides a deterministic, pitcher-disjoint cohort builder over the same
splits used by Phase 0.3 (10K-game train 2015-2022, 2023 val pitcher-disjoint,
2025 test pitcher-disjoint), engineered features per Plan B §3:

    pitch_type, zone, release_speed, count, outs, on_1b/2b/3b,
    batter_stand, pitcher_throws, inning, score_diff,
    umpire_scalar, pitcher_id, batter_id

Outputs feature DataFrames + 7-class outcome labels using the same
``classify_pitch_outcome`` rule set as Phase 0.3 (locked label semantics).

Read-only DuckDB; does NOT touch any model checkpoint.
"""

from __future__ import annotations

import logging
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.pitchgpt import _compute_per_pitch_score_diff  # noqa: E402
from src.analytics.pitchgpt_outcome_head import (  # noqa: E402
    NUM_OUTCOME_CLASSES,
    OUTCOME_CLASSES,
    OUTCOME_TO_IDX,
    OUTCOME_UNK,
    classify_pitch_outcome,
)
from src.db.schema import get_connection  # noqa: E402

logger = logging.getLogger("pitchgpt_outcome_baselines_common")

TRAIN_SEASONS = list(range(2015, 2023))   # 2015..2022 inclusive
VAL_SEASONS = [2023]
TEST_SEASONS = [2025]  # holdout-contact: historical contacts registered in docs/holdout_ledger.jsonl; any NEW run must append a new contact via src.holdout (@holdout_access / record_contact)

# Plan B cohort scale: 10K-game train.  Val + test sample sizes are
# matched to phase 0.3 (val 2K games) and to the 2025 holdout cohort
# already defined elsewhere in the codebase.
DEFAULT_TRAIN_GAMES = 10_000
DEFAULT_VAL_GAMES = 2_000
DEFAULT_TEST_GAMES = 2_000

DEFAULT_SEED = 42

# Engineered feature columns (Plan B §3).
# Categorical features one-hot or label-encoded; continuous features kept raw.
FEATURE_COLUMNS_NUMERIC = [
    "release_speed",       # continuous
    "umpire_scalar",       # continuous (called_correct_rate proxy)
    "score_diff",          # continuous (signed)
]
FEATURE_COLUMNS_CATEGORICAL = [
    "pitch_type",
    "zone",
    "balls",
    "strikes",
    "outs_when_up",
    "on_1b_bool",
    "on_2b_bool",
    "on_3b_bool",
    "stand",
    "p_throws",
    "inning_bucket",       # binned 1-3 / 4-6 / 7+ / extras
    "pitcher_id",
    "batter_id",
]


def fetch_pitcher_ids(conn: Any, seasons: Iterable[int]) -> set[int]:
    s_list = ", ".join(str(int(s)) for s in seasons)
    if not s_list:
        return set()
    rows = conn.execute(f"""
        SELECT DISTINCT pitcher_id
        FROM pitches
        WHERE pitch_type IS NOT NULL
          AND pitcher_id IS NOT NULL
          AND EXTRACT(YEAR FROM game_date) IN ({s_list})
    """).fetchdf()
    return {int(p) for p in rows["pitcher_id"].tolist() if p is not None}


def sample_game_pks(
    conn: Any,
    seasons: Iterable[int],
    max_games: int,
    exclude_pitcher_ids: set[int] | None = None,
    seed: int = DEFAULT_SEED,
) -> list[int]:
    """Deterministically sample game_pks for the requested seasons,
    optionally restricted to games containing at least one non-excluded
    pitcher.

    Determinism: we fetch the full eligible game_pk list, sort it, and
    take a numpy seeded sample.  ``DuckDB USING SAMPLE`` is not
    deterministic across runs even with seeds in some versions, so we
    do client-side sampling.
    """
    s_list = ", ".join(str(int(s)) for s in seasons)
    if not s_list:
        return []
    rng = np.random.default_rng(seed)

    # Fetch eligible games.
    if exclude_pitcher_ids:
        # Game must contain at least one pitch from a NON-excluded pitcher.
        # Build a temp filter — large IN lists are fine for DuckDB.
        excl_sample_size = len(exclude_pitcher_ids)
        # If exclusion list is very large, push through a temp table.
        if excl_sample_size > 200_000:
            # Edge case; in practice ~3-5K pitchers max.
            excl = list(exclude_pitcher_ids)
            # Use temp table.
            tmp_name = f"_baselines_excl_{seed}"
            conn.execute(f"DROP TABLE IF EXISTS {tmp_name}")
            conn.execute(
                f"CREATE TEMP TABLE {tmp_name} (pitcher_id BIGINT)"
            )
            conn.executemany(
                f"INSERT INTO {tmp_name} VALUES (?)",
                [(int(p),) for p in excl],
            )
            game_q = f"""
                SELECT DISTINCT game_pk
                FROM pitches
                WHERE pitch_type IS NOT NULL
                  AND pitcher_id IS NOT NULL
                  AND pitcher_id NOT IN (SELECT pitcher_id FROM {tmp_name})
                  AND EXTRACT(YEAR FROM game_date) IN ({s_list})
            """
            df = conn.execute(game_q).fetchdf()
            conn.execute(f"DROP TABLE IF EXISTS {tmp_name}")
        else:
            pids_str = ", ".join(str(int(p)) for p in exclude_pitcher_ids)
            game_q = f"""
                SELECT DISTINCT game_pk
                FROM pitches
                WHERE pitch_type IS NOT NULL
                  AND pitcher_id IS NOT NULL
                  AND pitcher_id NOT IN ({pids_str})
                  AND EXTRACT(YEAR FROM game_date) IN ({s_list})
            """
            df = conn.execute(game_q).fetchdf()
    else:
        df = conn.execute(f"""
            SELECT DISTINCT game_pk
            FROM pitches
            WHERE pitch_type IS NOT NULL
              AND pitcher_id IS NOT NULL
              AND EXTRACT(YEAR FROM game_date) IN ({s_list})
        """).fetchdf()

    games = sorted(int(g) for g in df["game_pk"].tolist())
    if not games:
        return []
    if len(games) <= max_games:
        return games
    idx = rng.choice(len(games), size=max_games, replace=False)
    sampled = sorted([games[i] for i in idx])
    return sampled


def fetch_umpire_scalar(conn: Any) -> dict[tuple[int, str], float]:
    """Build a (season, umpire_id) -> called_correct_rate lookup.

    The umpire-feature is a per-game season-level called-correct-rate
    proxy — robust to noisy single-game data.  We average over all
    rows for the season.  Falls back to the global mean for missing
    seasons / umpires.
    """
    df = conn.execute("""
        SELECT season, umpire, called_correct_rate
        FROM umpire_tendencies
        WHERE called_correct_rate IS NOT NULL
    """).fetchdf()
    if df.empty:
        return {}
    out: dict[tuple[int, str], float] = {}
    for _, row in df.iterrows():
        s = int(row["season"]) if row["season"] is not None else None
        u = str(row["umpire"]) if row["umpire"] is not None else None
        r = float(row["called_correct_rate"])
        if s is not None and u is not None and not np.isnan(r):
            out[(s, u)] = r
    return out


def _inning_bucket(inning: int | None) -> int:
    if inning is None or pd.isna(inning):
        return 0
    i = int(inning)
    if i <= 3:
        return 1
    if i <= 6:
        return 2
    if i <= 9:
        return 3
    return 4  # extras


def fetch_features_for_games(
    conn: Any,
    game_pks: list[int],
    umpire_scalar_lookup: dict[tuple[int, str], float],
    exclude_pitcher_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Fetch a feature DataFrame for the requested ``game_pks``.

    Joins the home-plate umpire (position='HP') for the umpire scalar.
    Filters to rows with a non-null pitch_type and a labellable description.

    Returns
    -------
    pd.DataFrame
        columns: features + ``outcome_label`` (int) + ``pitcher_id``,
        ``batter_id`` (int).  Rows with OUTCOME_UNK label are dropped.
    """
    if not game_pks:
        return pd.DataFrame()

    # Chunk the IN list to avoid SQL parser limits on very large lists.
    chunks: list[pd.DataFrame] = []
    chunk_size = 5_000
    for start in range(0, len(game_pks), chunk_size):
        end = min(start + chunk_size, len(game_pks))
        gpks_str = ", ".join(str(int(g)) for g in game_pks[start:end])

        excl_filter = ""
        if exclude_pitcher_ids:
            pids_str = ", ".join(str(int(p)) for p in exclude_pitcher_ids)
            excl_filter = f"AND p.pitcher_id NOT IN ({pids_str})"

        # Home-plate umpire join (position='HP').  LEFT JOIN so missing-ump
        # rows still come through with NULL — we impute below.
        q = f"""
            SELECT
                p.game_pk,
                p.pitcher_id,
                p.batter_id,
                p.pitch_type,
                COALESCE(p.zone, -1) AS zone,
                p.release_speed,
                p.balls,
                p.strikes,
                p.outs_when_up,
                p.on_1b,
                p.on_2b,
                p.on_3b,
                p.stand,
                p.p_throws,
                p.inning,
                p.inning_topbot,
                p.delta_run_exp,
                p.description,
                p.events,
                p.at_bat_number,
                p.pitch_number,
                EXTRACT(YEAR FROM p.game_date)::INTEGER AS season,
                ua.umpire_id AS hp_umpire_id
            FROM pitches p
            LEFT JOIN umpire_assignments ua
              ON p.game_pk = ua.game_pk AND ua.position = 'HP'
            WHERE p.pitch_type IS NOT NULL
              AND p.game_pk IN ({gpks_str})
              {excl_filter}
            ORDER BY p.game_pk, p.at_bat_number, p.pitch_number
        """
        df = conn.execute(q).fetchdf()
        if not df.empty:
            chunks.append(df)
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)

    # 7-class label.
    labels = [
        classify_pitch_outcome(d, e)
        for d, e in zip(df["description"].tolist(), df["events"].tolist())
    ]
    df["outcome_label"] = labels
    df = df[df["outcome_label"] != OUTCOME_UNK].reset_index(drop=True)

    # Engineered features.
    df["on_1b_bool"] = df["on_1b"].notna().astype(np.int8)
    df["on_2b_bool"] = df["on_2b"].notna().astype(np.int8)
    df["on_3b_bool"] = df["on_3b"].notna().astype(np.int8)
    df["inning_bucket"] = df["inning"].map(_inning_bucket).astype(np.int16)

    # Score diff: pre-pitch home-vs-away score from the pitcher's POV.
    # IMPORTANT: do NOT use ``delta_run_exp`` — that is a POST-PITCH outcome
    # signal (a home run sets delta_run_exp ≈ +1.5; a called strike ≈ -0.07)
    # and would directly leak the pitch outcome into features.  Reconstruct
    # the pre-pitch running score using the same helper PitchGPT uses for
    # its context tensor.  Requires the rows to be in (game_pk, at_bat,
    # pitch_number) order which our SQL already guarantees.
    score_diffs = _compute_per_pitch_score_diff(df)
    df["score_diff"] = np.asarray(score_diffs, dtype=np.float32)

    # Umpire scalar.
    if umpire_scalar_lookup:
        global_mean = float(np.mean(list(umpire_scalar_lookup.values())))
    else:
        global_mean = 0.92  # league-average ump accuracy fallback
    def _ump_scalar(row):
        ump = row.get("hp_umpire_id")
        season = row.get("season")
        if ump is None or season is None or pd.isna(ump):
            return global_mean
        try:
            return umpire_scalar_lookup.get((int(season), str(ump)), global_mean)
        except (ValueError, TypeError):
            return global_mean
    df["umpire_scalar"] = df.apply(_ump_scalar, axis=1).astype(np.float32)

    # Fill NULLs in numerics.
    df["release_speed"] = df["release_speed"].fillna(0.0).astype(np.float32)
    df["balls"] = df["balls"].fillna(0).astype(np.int8)
    df["strikes"] = df["strikes"].fillna(0).astype(np.int8)
    df["outs_when_up"] = df["outs_when_up"].fillna(0).astype(np.int8)
    df["zone"] = df["zone"].fillna(-1).astype(np.int16)
    df["pitch_type"] = df["pitch_type"].fillna("UNK").astype(str)
    df["stand"] = df["stand"].fillna("UNK").astype(str)
    df["p_throws"] = df["p_throws"].fillna("UNK").astype(str)

    df["pitcher_id"] = df["pitcher_id"].astype(np.int64)
    df["batter_id"] = df["batter_id"].fillna(-1).astype(np.int64)

    # Reorder for consistent output.
    keep_cols = (
        FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL + ["outcome_label"]
    )
    out = df[keep_cols].copy()
    return out


def build_cohort(
    train_games: int = DEFAULT_TRAIN_GAMES,
    val_games: int = DEFAULT_VAL_GAMES,
    test_games: int = DEFAULT_TEST_GAMES,
    seed: int = DEFAULT_SEED,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build train / val / test feature DataFrames.

    Pitcher-disjoint discipline:
        - Train pitchers: union over 2015-2022 (full season).
        - Val: 2023 games whose pitchers are NOT in train.
        - Test: 2025 games whose pitchers are NOT in train.

    Optional ``cache_dir``: if set, caches the featurised DataFrames to
    parquet keyed on (train_games, val_games, test_games, seed) so
    subsequent runs reuse the same SQL output (saves ~7 min on full
    scale).  Cache is a strict (gp1, gp2, gp3, seed) match — different
    args → cache miss.
    """
    cache_key = f"cohort_t{train_games}_v{val_games}_te{test_games}_s{seed}"
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_p = cache_dir / f"{cache_key}_train.parquet"
        val_p = cache_dir / f"{cache_key}_val.parquet"
        test_p = cache_dir / f"{cache_key}_test.parquet"
        if train_p.exists() and val_p.exists() and test_p.exists():
            logger.info("Cache hit: loading cohort from %s", cache_dir)
            train_df = pd.read_parquet(train_p)
            val_df = pd.read_parquet(val_p)
            test_df = pd.read_parquet(test_p)
            logger.info("Loaded train=%d val=%d test=%d from cache",
                        len(train_df), len(val_df), len(test_df))
            return train_df, val_df, test_df

    conn = get_connection(read_only=True)
    try:
        logger.info("Fetching train pitcher cohort (2015-2022)...")
        train_pitchers = fetch_pitcher_ids(conn, TRAIN_SEASONS)
        logger.info("Train pitcher cohort: %d unique pitchers", len(train_pitchers))

        logger.info("Sampling %d train games from 2015-2022...", train_games)
        train_pks = sample_game_pks(
            conn, TRAIN_SEASONS, max_games=train_games, seed=seed,
        )
        logger.info("Train games selected: %d", len(train_pks))

        logger.info(
            "Sampling %d val games from 2023 (pitcher-disjoint)...", val_games,
        )
        val_pks = sample_game_pks(
            conn, VAL_SEASONS, max_games=val_games,
            exclude_pitcher_ids=train_pitchers, seed=seed + 1,
        )
        logger.info("Val games selected: %d", len(val_pks))

        logger.info(
            "Sampling %d test games from 2025 (pitcher-disjoint)...", test_games,
        )
        test_pks = sample_game_pks(
            conn, TEST_SEASONS, max_games=test_games,
            exclude_pitcher_ids=train_pitchers, seed=seed + 2,
        )
        logger.info("Test games selected: %d", len(test_pks))

        logger.info("Building umpire-scalar lookup...")
        ump_lookup = fetch_umpire_scalar(conn)
        logger.info("Umpire-scalar entries: %d", len(ump_lookup))

        logger.info("Fetching train features...")
        train_df = fetch_features_for_games(
            conn, train_pks, ump_lookup,
        )
        logger.info("Train rows (post-UNK-filter): %d", len(train_df))

        logger.info("Fetching val features (drop train pitchers)...")
        val_df = fetch_features_for_games(
            conn, val_pks, ump_lookup, exclude_pitcher_ids=train_pitchers,
        )
        logger.info("Val rows: %d", len(val_df))

        logger.info("Fetching test features (drop train pitchers)...")
        test_df = fetch_features_for_games(
            conn, test_pks, ump_lookup, exclude_pitcher_ids=train_pitchers,
        )
        logger.info("Test rows: %d", len(test_df))
    finally:
        conn.close()

    if cache_dir is not None:
        logger.info("Writing cohort to cache %s", cache_dir)
        train_df.to_parquet(train_p)
        val_df.to_parquet(val_p)
        test_df.to_parquet(test_p)
    return train_df, val_df, test_df


# ═════════════════════════════════════════════════════════════════════════════
# Metrics
# ═════════════════════════════════════════════════════════════════════════════

def class_frequency_prior(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=NUM_OUTCOME_CLASSES).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return np.full(NUM_OUTCOME_CLASSES, 1.0 / NUM_OUTCOME_CLASSES)
    return counts / total


def log_loss_from_probs(probs: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    return float(-np.log(p[np.arange(len(y)), y]).mean())


def per_class_log_loss(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for c, name in enumerate(OUTCOME_CLASSES):
        mask = y == c
        if mask.sum() == 0:
            out[name] = float("nan")
            continue
        out[name] = log_loss_from_probs(probs[mask], y[mask])
    return out


def top1_accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float((probs.argmax(axis=1) == y).mean())


def ece_10bin(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Top-1 expected calibration error (10-bin)."""
    if len(y) == 0:
        return float("nan")
    top1_p = probs.max(axis=1)
    top1_idx = probs.argmax(axis=1)
    correct = (top1_idx == y).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_total = top1_p.shape[0]
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (top1_p >= lo) & (top1_p <= hi)
        else:
            mask = (top1_p >= lo) & (top1_p < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        ece += (n / n_total) * abs(top1_p[mask].mean() - correct[mask].mean())
    return float(ece)


def fit_temperature(
    logits: np.ndarray, y: np.ndarray, lr: float = 0.05, n_iter: int = 300,
) -> float:
    """Fit a single scalar temperature via numerical optimisation (LBFGS-like).

    Pure-numpy gradient descent on T (a single positive scalar); converges in
    <300 iterations for typical 7-class outcome logits.
    """
    if len(y) == 0:
        return 1.0
    # Use scipy's minimize_scalar for robustness.
    from scipy.optimize import minimize_scalar

    def nll(T: float) -> float:
        T = float(T)
        if T <= 1e-3:
            return 1e6
        scaled = logits / T
        # numerically stable log-softmax
        m = scaled.max(axis=1, keepdims=True)
        log_z = m.squeeze(1) + np.log(np.exp(scaled - m).sum(axis=1))
        log_probs = scaled - log_z[:, None]
        return float(-log_probs[np.arange(len(y)), y].mean())

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


def softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=1, keepdims=True)
    e = np.exp(z - m)
    return e / e.sum(axis=1, keepdims=True)


def bootstrap_log_loss(
    probs: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 42,
) -> tuple[float, float, float]:
    """Returns (point, ci_lo, ci_hi) for log-loss via row resampling."""
    rng = np.random.default_rng(seed)
    n = len(y)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    point = log_loss_from_probs(probs, y)
    samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[b] = log_loss_from_probs(probs[idx], y[idx])
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return point, lo, hi


def per_pitcher_log_loss(
    probs: np.ndarray, y: np.ndarray, pitcher_ids: np.ndarray, top_k: int = 50,
) -> list[dict]:
    """Compute log-loss per top-k most-frequent pitcher (by row count)."""
    if len(y) == 0:
        return []
    df = pd.DataFrame({"pid": pitcher_ids, "_idx": np.arange(len(y))})
    counts = df.groupby("pid").size().sort_values(ascending=False)
    top = counts.head(top_k).index.tolist()
    out: list[dict] = []
    for pid in top:
        mask = pitcher_ids == pid
        n = int(mask.sum())
        if n < 30:  # too small
            continue
        ll = log_loss_from_probs(probs[mask], y[mask])
        out.append({"pitcher_id": int(pid), "n": n, "log_loss": ll})
    return out


def freq_prior_log_loss_on(y: np.ndarray, prior: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(prior, eps, 1.0)
    return float(-np.log(p[y]).mean())
