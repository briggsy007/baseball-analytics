#!/usr/bin/env python
"""Ingest umpire identity + zone tendencies for 2015-2026.

Stages two parquet files under ``data/staging/`` (this script does **not**
write to DuckDB — schema extension is planned in
``docs/data/umpire_integration_notes.md``):

* ``umpire_assignments.parquet`` — one row per (game_pk, position,
  umpire_id, umpire_name). Position is one of ``HP``, ``1B``, ``2B``,
  ``3B``, ``LF``, ``RF`` (LF/RF only present on postseason / ASG games).

* ``umpire_tendencies.parquet`` — one row per (umpire_name, season) with
  called-strike-rate / accuracy / run-impact fields aggregated from
  umpscorecards.com per-game data.

Sources
-------
1. **Retrosheet game logs** (https://www.retrosheet.org/gamelogs/gl{YEAR}.zip)
   — provides HP + base umpire IDs and names per game, keyed on
   (date, home_team, game_num). Joined to MLBAM ``game_pk`` via the
   ``pitches`` table using a Retrosheet-team-code -> MLBAM-team-code map.
   Postings lag the season, so 2026 is expected to be absent (404).

2. **Umpire Scorecards** (https://umpscorecards.com/api/games) — game-keyed
   on MLBAM ``game_pk`` directly. Provides HP umpire name + per-game
   accuracy / favor / run-impact. Aggregated to season-level in the
   tendencies parquet. Covers 2015-present (including current season
   partial).

Idempotency
-----------
Retrosheet zips are cached under ``data/staging/retrosheet_cache/`` so
repeat runs only hit the network when a year is missing locally. The
umpscorecards dump is refreshed on every run (it changes nightly). Pass
``--refresh`` to force a full re-download. Pass ``--no-write`` for a
dry run.

Usage (from project root):

    python scripts/ingest_umpire_data.py
    python scripts/ingest_umpire_data.py --seasons 2023,2024,2025
    python scripts/ingest_umpire_data.py --refresh
"""

from __future__ import annotations

import argparse
import io
import logging
import sys
import zipfile
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.db.schema import DEFAULT_DB_PATH  # noqa: E402

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

STAGING_DIR = PROJECT_ROOT / "data" / "staging"
CACHE_DIR = STAGING_DIR / "retrosheet_cache"

ASSIGNMENTS_PATH = STAGING_DIR / "umpire_assignments.parquet"
TENDENCIES_PATH = STAGING_DIR / "umpire_tendencies.parquet"

DEFAULT_START = 2015
DEFAULT_END = 2026

RETROSHEET_URL = "https://www.retrosheet.org/gamelogs/gl{year}.zip"
UMPSCORECARDS_URL = "https://umpscorecards.com/api/games"

REQUEST_TIMEOUT = 90

# Retrosheet ships 161 columns; we only read the ones we need. The full
# column order is taken from pybaseball.retrosheet.gamelog_columns, which
# mirrors the Retrosheet DTD (`glfields.html`).
try:
    from pybaseball.retrosheet import gamelog_columns as _PYB_GL_COLS
    GAMELOG_COLUMNS: list[str] = list(_PYB_GL_COLS)
except Exception:  # pragma: no cover — pybaseball optional safety net
    GAMELOG_COLUMNS = []

# Retrosheet -> MLBAM team-code translation. Keys on the Retrosheet home_team
# field; values are the codes present in the project's ``pitches.home_team``
# column. Codes identical across the two systems (ATL, BOS, CIN, CLE, COL,
# DET, HOU, MIL, MIN, PHI, PIT, SEA, TEX, TOR, BAL, MIA) pass through.
RETROSHEET_TO_MLBAM: dict[str, str] = {
    "ANA": "LAA",
    "ARI": "AZ",
    "CHA": "CWS",
    "CHN": "CHC",
    "KCA": "KC",
    "LAN": "LAD",
    "NYA": "NYY",
    "NYN": "NYM",
    "OAK": "ATH",  # Athletics rebrand — MLBAM uses ATH from 2025
    "SDN": "SD",
    "SFN": "SF",
    "SLN": "STL",
    "TBA": "TB",
    "WAS": "WSH",
}

# Positions emitted per game. Retrosheet gives us up to six; most regular-
# season games only have four (HP / 1B / 2B / 3B).
UMPIRE_POSITIONS: tuple[str, ...] = ("HP", "1B", "2B", "3B", "LF", "RF")

# Tendency columns kept from the per-game umpscorecards payload.
# Aggregation strategy: weight rate-stats by called_pitches, sum counts.
_UMPSCORECARDS_GAME_COLS = (
    "game_pk",
    "date",
    "type",
    "umpire",
    "home_team",
    "away_team",
    "called_pitches",
    "called_correct",
    "called_wrong",
    "overall_accuracy",
    "x_overall_accuracy",
    "accuracy_above_x",
    "correct_calls_above_x",
    "consistency",
    "favor",
    "home_batter_impact",
    "home_pitcher_impact",
    "away_batter_impact",
    "away_pitcher_impact",
    "total_run_impact",
)


# ── Retrosheet ingest ──────────────────────────────────────────────────────


def _retrosheet_zip_path(year: int) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"gl{year}.zip"


def _download_retrosheet(year: int, refresh: bool = False) -> Optional[Path]:
    """Download ``gl{YEAR}.zip`` if not already cached. Return path or None.

    Returns None when the file doesn't exist on retrosheet.org (e.g. current
    in-progress season before year-end posting).
    """
    dest = _retrosheet_zip_path(year)
    if dest.exists() and not refresh:
        logger.info("retrosheet %d: cached at %s", year, dest)
        return dest

    url = RETROSHEET_URL.format(year=year)
    logger.info("retrosheet %d: GET %s", year, url)
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as exc:
        logger.warning("retrosheet %d: network error %s", year, exc)
        return None
    if r.status_code == 404:
        logger.info("retrosheet %d: not yet published (404)", year)
        return None
    if r.status_code != 200:
        logger.warning("retrosheet %d: HTTP %d", year, r.status_code)
        return None
    dest.write_bytes(r.content)
    logger.info("retrosheet %d: wrote %d bytes to cache", year, len(r.content))
    return dest


def _read_retrosheet_gamelog(path: Path) -> pd.DataFrame:
    """Return the gamelog txt inside ``path`` as a typed DataFrame."""
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            df = pd.read_csv(
                f,
                header=None,
                names=GAMELOG_COLUMNS,
                low_memory=False,
                dtype={"date": str, "game_num": "Int64"},
            )
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce").dt.date
    df["home_team_mlbam"] = df["home_team"].map(RETROSHEET_TO_MLBAM).fillna(df["home_team"])
    return df


def ingest_retrosheet(
    seasons: list[int],
    refresh: bool = False,
) -> pd.DataFrame:
    """Return long-form (date, home_team_mlbam, game_num, position, umpire_id,
    umpire_name) frame across ``seasons``. Missing years are skipped."""
    frames: list[pd.DataFrame] = []
    for year in seasons:
        path = _download_retrosheet(year, refresh=refresh)
        if path is None:
            continue
        gl = _read_retrosheet_gamelog(path)

        # Reshape wide -> long: one row per position
        records: list[dict] = []
        pos_fields = [
            ("HP", "ump_home_id", "ump_home_name"),
            ("1B", "ump_first_id", "ump_first_name"),
            ("2B", "ump_second_id", "ump_second_name"),
            ("3B", "ump_third_id", "ump_third_name"),
            ("LF", "ump_lf_id", "ump_lf_name"),
            ("RF", "ump_rf_id", "ump_rf_name"),
        ]
        for pos, id_col, name_col in pos_fields:
            if id_col not in gl.columns:
                continue
            block = gl[["date", "home_team_mlbam", "game_num", id_col, name_col]].copy()
            block.columns = ["date", "home_team_mlbam", "game_num", "umpire_id", "umpire_name"]
            block["position"] = pos
            records.append(block)
        long_df = pd.concat(records, ignore_index=True)
        # Drop rows with no umpire assigned. Retrosheet uses the literal
        # "(none)" (and similar variants) for games without an LF/RF umpire,
        # which is the norm outside of postseason / All-Star games.
        _sentinel = {"", "(none)", "(unknown)", "unknown"}
        long_df = long_df[long_df["umpire_name"].notna()]
        long_df = long_df[~long_df["umpire_name"].astype(str).str.strip().str.lower().isin(
            {s.lower() for s in _sentinel}
        )]
        long_df["season"] = year
        frames.append(long_df)
        logger.info("retrosheet %d: %d umpire-assignment rows", year, len(long_df))
    if not frames:
        return pd.DataFrame(
            columns=["date", "home_team_mlbam", "game_num", "umpire_id",
                     "umpire_name", "position", "season"]
        )
    return pd.concat(frames, ignore_index=True)


# ── game_pk join ───────────────────────────────────────────────────────────


def _load_game_pk_map(db_path: Path) -> pd.DataFrame:
    """Return a DataFrame of distinct (game_pk, game_date, home_team) from
    ``pitches``. Doubleheaders are resolved via ``game_num`` later by
    ordering game_pks within a (date, home_team) cluster."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(
            """
            SELECT
                game_pk,
                game_date,
                home_team
            FROM (
                SELECT game_pk, game_date, home_team,
                       ROW_NUMBER() OVER (PARTITION BY game_pk ORDER BY game_date) AS rn
                FROM pitches
                WHERE game_pk IS NOT NULL AND game_date IS NOT NULL
            )
            WHERE rn = 1
            """
        ).fetchdf()
    finally:
        con.close()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def _attach_game_pk(assignments: pd.DataFrame, game_map: pd.DataFrame) -> pd.DataFrame:
    """Left-join retrosheet assignments onto MLBAM game_pks.

    Handles doubleheaders. Retrosheet's ``game_num`` is 0 for single
    games, 1/2 for doubleheader halves. We normalize to a 1-indexed slot
    (single games -> 1, DH game 1 -> 1, DH game 2 -> 2), then rank MLBAM
    game_pks the same way within each (date, home_team) cluster.
    """
    if assignments.empty:
        return assignments.assign(game_pk=pd.Series([], dtype="Int64"))

    assignments = assignments.copy()
    assignments["game_slot"] = assignments["game_num"].fillna(0).astype(int)
    # Retrosheet single games use 0 — treat as slot 1
    assignments.loc[assignments["game_slot"] == 0, "game_slot"] = 1

    gm = game_map.copy()
    gm = gm.sort_values(["game_date", "home_team", "game_pk"])
    gm["game_slot"] = gm.groupby(["game_date", "home_team"]).cumcount() + 1

    merged = assignments.merge(
        gm.rename(columns={
            "game_date": "date",
            "home_team": "home_team_mlbam",
        }),
        on=["date", "home_team_mlbam", "game_slot"],
        how="left",
    )
    return merged


# ── umpscorecards ingest ───────────────────────────────────────────────────


def fetch_umpscorecards_games() -> pd.DataFrame:
    """Return the full /api/games table (all seasons, HP-umpire only)."""
    logger.info("umpscorecards: GET %s", UMPSCORECARDS_URL)
    r = requests.get(UMPSCORECARDS_URL, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    rows = r.json().get("rows", [])
    df = pd.DataFrame(rows)
    logger.info("umpscorecards: %d game rows", len(df))
    return df


def build_tendencies(games: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-game umpscorecards rows to (umpire, season) tendencies.

    Called-strike-rate is not directly exposed by umpscorecards; the
    closest analogue is the strike-vs-ball *accuracy* metric together with
    the ``favor`` sign (+=batter-favoring, -=pitcher-favoring in the
    umpscorecards convention). We surface:

    * called_pitches (sum)
    * called_correct_rate (called_correct / called_pitches, weighted)
    * overall_accuracy_wmean (avg accuracy weighted by called_pitches)
    * accuracy_above_x_wmean (delta vs expected, weighted by called_pitches)
    * consistency_wmean (weighted)
    * favor_wmean (weighted; negative = pitcher-favoring, positive = batter-favoring)
    * total_run_impact_mean (per-game)
    * batter_impact_mean, pitcher_impact_mean (home/away averaged per-game)
    """
    if games.empty:
        return pd.DataFrame()

    keep_cols = [c for c in _UMPSCORECARDS_GAME_COLS if c in games.columns]
    df = games[keep_cols].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["date"].dt.year

    # Numeric coercion (some rows have nulls)
    numeric_cols = [
        "called_pitches", "called_correct", "called_wrong",
        "overall_accuracy", "x_overall_accuracy", "accuracy_above_x",
        "correct_calls_above_x", "consistency", "favor",
        "home_batter_impact", "home_pitcher_impact",
        "away_batter_impact", "away_pitcher_impact",
        "total_run_impact",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop rows without a weight (can't weight-average)
    df = df[df["called_pitches"].fillna(0) > 0].copy()

    # Combined batter/pitcher impact (mean across home & away)
    df["batter_impact_combined"] = (
        df[["home_batter_impact", "away_batter_impact"]].mean(axis=1, skipna=True)
    )
    df["pitcher_impact_combined"] = (
        df[["home_pitcher_impact", "away_pitcher_impact"]].mean(axis=1, skipna=True)
    )

    def _wmean(group: pd.DataFrame, col: str) -> float:
        w = group["called_pitches"].fillna(0).astype(float).values
        v = group[col].astype(float).values
        mask = (~pd.isna(v)) & (w > 0)
        if not mask.any():
            return float("nan")
        return float((v[mask] * w[mask]).sum() / w[mask].sum())

    grouped = df.groupby(["umpire", "season"], dropna=False)

    def _agg(group: pd.DataFrame) -> pd.Series:
        called = float(group["called_pitches"].fillna(0).sum())
        correct = float(group["called_correct"].fillna(0).sum())
        return pd.Series({
            "games": int(len(group)),
            "called_pitches": int(called),
            "called_correct": int(correct),
            "called_correct_rate": (correct / called) if called > 0 else float("nan"),
            "overall_accuracy_wmean": _wmean(group, "overall_accuracy"),
            "x_overall_accuracy_wmean": _wmean(group, "x_overall_accuracy"),
            "accuracy_above_x_wmean": _wmean(group, "accuracy_above_x"),
            "consistency_wmean": _wmean(group, "consistency"),
            "favor_wmean": _wmean(group, "favor"),
            "total_run_impact_mean": float(group["total_run_impact"].mean(skipna=True)),
            "batter_impact_mean": float(group["batter_impact_combined"].mean(skipna=True)),
            "pitcher_impact_mean": float(group["pitcher_impact_combined"].mean(skipna=True)),
        })

    tendencies = grouped.apply(_agg, include_groups=False).reset_index()
    # Order rows for reproducibility
    tendencies = tendencies.sort_values(["season", "umpire"]).reset_index(drop=True)
    return tendencies


# ── Assemble assignments parquet ───────────────────────────────────────────


def build_assignments(
    retrosheet_long: pd.DataFrame,
    umpscorecards_games: pd.DataFrame,
    game_map: pd.DataFrame,
) -> pd.DataFrame:
    """Produce final (game_pk, position, umpire_id, umpire_name) frame.

    Priority rules:
    * HP rows from umpscorecards (keyed directly on ``game_pk``) take
      precedence over retrosheet HP rows when the same game_pk is present —
      umpscorecards uses MLBAM names verbatim which is what downstream joins
      expect.
    * Base-umpire rows (1B/2B/3B/LF/RF) are retrosheet-only. Retrosheet IDs
      are kept in ``umpire_id`` because no MLBAM equivalent exists.
    """
    frames: list[pd.DataFrame] = []

    # Retrosheet → joined to game_pk
    retro_with_pk = _attach_game_pk(retrosheet_long, game_map)
    retro_df = retro_with_pk[[
        "game_pk", "position", "umpire_id", "umpire_name",
        "date", "season",
    ]].copy()
    retro_df["source"] = "retrosheet"
    frames.append(retro_df)

    # umpscorecards HP → already keyed on game_pk
    if not umpscorecards_games.empty:
        usc = umpscorecards_games[["game_pk", "umpire", "date"]].copy()
        usc = usc[usc["umpire"].notna() & (usc["umpire"].astype(str) != "")]
        usc["date"] = pd.to_datetime(usc["date"]).dt.date
        usc["season"] = pd.to_datetime(usc["date"]).dt.year
        usc["position"] = "HP"
        usc["umpire_id"] = None  # umpscorecards has no stable id
        usc["umpire_name"] = usc["umpire"].astype(str)
        usc["source"] = "umpscorecards"
        frames.append(usc[[
            "game_pk", "position", "umpire_id", "umpire_name",
            "date", "season", "source",
        ]])

    combined = pd.concat(frames, ignore_index=True, sort=False)

    # De-dupe HP rows only (they come from both sources); preserve all
    # base-umpire rows and any unmatched retrosheet rows. Within a matched
    # (game_pk, position="HP") pair, prefer umpscorecards.
    source_priority = {"umpscorecards": 0, "retrosheet": 1}
    combined["_prio"] = combined["source"].map(source_priority).fillna(99).astype(int)

    hp_matched = combined[(combined["position"] == "HP") & combined["game_pk"].notna()].copy()
    hp_matched = (
        hp_matched.sort_values(["game_pk", "_prio"])
        .drop_duplicates(["game_pk", "position"], keep="first")
    )
    other = combined[~((combined["position"] == "HP") & combined["game_pk"].notna())]
    combined = pd.concat([hp_matched, other], ignore_index=True).drop(columns=["_prio"])

    # Keep rows without game_pk too — they represent retrosheet games we
    # failed to match to MLBAM. Downstream users can use (date, team, game_num)
    # as a backup join key. Leave game_pk as NULL rather than fabricating.
    combined = combined.sort_values(
        ["season", "date", "game_pk", "position"],
        na_position="last",
    ).reset_index(drop=True)
    return combined


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=str, default=None,
                        help="Comma-separated seasons (default 2015-2026).")
    parser.add_argument("--refresh", action="store_true",
                        help="Force re-download of cached Retrosheet zips.")
    parser.add_argument("--no-write", action="store_true",
                        help="Dry run — compute but don't write parquet files.")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH),
                        help="DuckDB path (read-only; used only for game_pk map).")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not GAMELOG_COLUMNS:
        logger.error("pybaseball.retrosheet.gamelog_columns unavailable — aborting.")
        return 2

    if args.seasons:
        seasons = [int(s) for s in args.seasons.split(",") if s.strip()]
    else:
        seasons = list(range(DEFAULT_START, DEFAULT_END + 1))

    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    # ── Retrosheet ────────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Retrosheet game-log ingest — seasons %s", seasons)
    logger.info("=" * 70)
    retro_long = ingest_retrosheet(seasons, refresh=args.refresh)
    logger.info("retrosheet: %d total umpire-assignment rows", len(retro_long))
    covered = sorted(retro_long["season"].unique().tolist()) if not retro_long.empty else []
    missing = [s for s in seasons if s not in covered]
    if missing:
        logger.info("retrosheet: seasons without local data (expected for in-progress years): %s", missing)

    # ── umpscorecards ─────────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("Umpire Scorecards per-game ingest")
    logger.info("=" * 70)
    try:
        usc_games = fetch_umpscorecards_games()
    except requests.RequestException as exc:
        logger.warning("umpscorecards: fetch failed (%s) — continuing without it", exc)
        usc_games = pd.DataFrame()

    # Filter to seasons in scope
    if not usc_games.empty:
        usc_games["date"] = pd.to_datetime(usc_games["date"])
        usc_games = usc_games[usc_games["date"].dt.year.isin(seasons)].copy()
        logger.info("umpscorecards: %d rows in seasons %s", len(usc_games), seasons)

    # ── game_pk map (from pitches table) ──────────────────────────────────
    logger.info("loading game_pk map from pitches table")
    db_path = Path(args.db)
    game_map = _load_game_pk_map(db_path)
    logger.info("game_map: %d distinct game_pks", len(game_map))

    # ── Build outputs ─────────────────────────────────────────────────────
    assignments = build_assignments(retro_long, usc_games, game_map)
    tendencies = build_tendencies(usc_games)

    # Coverage diagnostics
    n_with_pk = assignments["game_pk"].notna().sum()
    logger.info(
        "assignments: %d rows (%d with game_pk, %d NULL)",
        len(assignments),
        n_with_pk,
        len(assignments) - n_with_pk,
    )
    logger.info("  by position:\n%s", assignments["position"].value_counts().to_string())
    logger.info("  by source:\n%s", assignments["source"].value_counts().to_string())
    logger.info("tendencies: %d (umpire, season) rows", len(tendencies))

    if args.no_write:
        logger.info("--no-write set: skipping parquet writes.")
        return 0

    assignments.to_parquet(ASSIGNMENTS_PATH, index=False)
    logger.info("wrote %s (%d rows)", ASSIGNMENTS_PATH, len(assignments))

    if not tendencies.empty:
        tendencies.to_parquet(TENDENCIES_PATH, index=False)
        logger.info("wrote %s (%d rows)", TENDENCIES_PATH, len(tendencies))
    else:
        logger.warning("tendencies frame empty — not writing parquet")

    return 0


if __name__ == "__main__":
    sys.exit(main())
