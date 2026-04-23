#!/usr/bin/env python
"""
Stage pre-2015 season-level WAR (2010-2014) to parquet for CausalWAR CI tightening.

Scope (2026-04-23)
------------------
The CausalWAR contrarian backtest's Buy-Low hit-rate 68% CI is noisy because
2015-2025 only yields ~11 season-pairs of transitions.  Extending back to
2010 adds 5 additional season-pairs, which back-of-envelope should shrink
the 68% CI from ~+/-4.5 pct pts to ~+/-3.3 pct pts under an n^(-1/2) scaling.

This script stages the underlying season stats ONLY -- it does NOT merge
into DuckDB.  The DB merge is a separate concern (gated by model validation
on the expanded 2010-2014 window).

Data source
-----------
The user's original ask was ``pybaseball.fg_batting_data(2010, 2014)`` /
``pybaseball.fg_pitching_data(2010, 2014)``.  As documented in
``scripts/backfill_fwar.py`` (2026-04-16), the FanGraphs leaders endpoint
returns HTTP 403 for unauthenticated requests from this host; every
``pybaseball`` FanGraphs wrapper hits the same 403.  We therefore use the
same documented fallback the existing 2015-2024 backfill uses:
``pybaseball.bwar_bat(return_all=True)`` and
``pybaseball.bwar_pitch(return_all=True)``.  This keeps the pre-2015
extension *schema-compatible* with the production 2015-2024 WAR column
(``war_source = "bref_bwar"`` for every row).

Outputs
-------
- ``data/staging/season_batting_stats_2010_2014.parquet``
- ``data/staging/season_pitching_stats_2010_2014.parquet``
- ``data/staging/season_war_2010_2014_audit.json``

Each staging parquet mirrors the long-format schema used by
``backfill_fwar.py``:
    player_id, player_name, season, position_type, war, pa_or_ip, war_source

DuckDB is NOT touched.  To merge, an operator runs ``backfill_fwar.py``
(or a follow-up script) after reviewing the audit JSON.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_season_war_2010_2014")

STAGING_DIR = ROOT / "data" / "staging"
STAGING_BAT = STAGING_DIR / "season_batting_stats_2010_2014.parquet"
STAGING_PIT = STAGING_DIR / "season_pitching_stats_2010_2014.parquet"
AUDIT_PATH = STAGING_DIR / "season_war_2010_2014_audit.json"

DEFAULT_YEARS = tuple(range(2010, 2015))  # 2010..2014 inclusive


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def _retry_fetch(fn, *, label: str, max_tries: int = 5) -> pd.DataFrame:
    """Simple exponential-ish backoff around transient fetch errors."""
    last_exc: Exception | None = None
    for attempt in range(1, max_tries + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - network path
            last_exc = exc
            wait = 30 * attempt
            logger.warning(
                "%s fetch failed (attempt %d/%d): %s -- sleeping %ds",
                label, attempt, max_tries, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(f"{label} fetch exhausted retries: {last_exc!r}")


def fetch_war_for_years(years: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download bWAR batting + pitching, filter to ``years``, aggregate stints.

    Returns (batters_df, pitchers_df), both long-format with schema:
        player_id (MLBAM, int64)
        player_name (str)
        season (int)
        position_type ('batter' or 'pitcher')
        war (float, nullable)
        pa_or_ip (float; PA for batters, IP for pitchers)
        war_source (str, always 'bref_bwar')
    """
    import pybaseball  # local import so tests can mock the module

    years_set = set(int(y) for y in years)

    logger.info("Downloading bWAR batting (filter to %s)", sorted(years_set))
    bwar_bat = _retry_fetch(lambda: pybaseball.bwar_bat(return_all=True), label="bwar_bat")
    logger.info("bwar_bat raw rows: %d (year range %s-%s)",
                len(bwar_bat), bwar_bat["year_ID"].min(), bwar_bat["year_ID"].max())

    logger.info("Downloading bWAR pitching (filter to %s)", sorted(years_set))
    bwar_pit = _retry_fetch(lambda: pybaseball.bwar_pitch(return_all=True), label="bwar_pitch")
    logger.info("bwar_pit raw rows: %d", len(bwar_pit))

    # -------- Batter aggregate --------
    bat = bwar_bat[bwar_bat["year_ID"].isin(years_set)].copy()
    bat["PA"] = pd.to_numeric(bat["PA"], errors="coerce").fillna(0.0)
    bat["WAR"] = pd.to_numeric(bat["WAR"], errors="coerce")
    bat["mlb_ID"] = pd.to_numeric(bat["mlb_ID"], errors="coerce")
    bat = bat.dropna(subset=["mlb_ID"]).copy()
    bat["mlb_ID"] = bat["mlb_ID"].astype("int64")

    bat_agg = (
        bat.groupby(["mlb_ID", "year_ID"], as_index=False)
        .agg(
            war=("WAR", "sum"),
            pa=("PA", "sum"),
            name=("name_common", "first"),
            war_any_nonnull=("WAR", lambda s: s.notna().any()),
        )
    )
    bat_agg = bat_agg[bat_agg["pa"] > 0].copy()
    bat_agg.loc[~bat_agg["war_any_nonnull"], "war"] = np.nan
    bat_agg = bat_agg.drop(columns=["war_any_nonnull"])
    bat_agg["position_type"] = "batter"
    bat_agg = bat_agg.rename(columns={
        "mlb_ID": "player_id",
        "year_ID": "season",
        "name": "player_name",
        "pa": "pa_or_ip",
    })
    bat_agg = bat_agg[["player_id", "player_name", "season", "position_type", "war", "pa_or_ip"]]
    bat_agg["season"] = bat_agg["season"].astype(int)
    bat_agg["war_source"] = "bref_bwar"

    # -------- Pitcher aggregate --------
    pit = bwar_pit[bwar_pit["year_ID"].isin(years_set)].copy()
    pit["IPouts"] = pd.to_numeric(pit["IPouts"], errors="coerce").fillna(0.0)
    pit["IP"] = pit["IPouts"] / 3.0
    pit["WAR"] = pd.to_numeric(pit["WAR"], errors="coerce")
    pit["mlb_ID"] = pd.to_numeric(pit["mlb_ID"], errors="coerce")
    pit = pit.dropna(subset=["mlb_ID"]).copy()
    pit["mlb_ID"] = pit["mlb_ID"].astype("int64")

    pit_agg = (
        pit.groupby(["mlb_ID", "year_ID"], as_index=False)
        .agg(
            war=("WAR", "sum"),
            ip=("IP", "sum"),
            name=("name_common", "first"),
            war_any_nonnull=("WAR", lambda s: s.notna().any()),
        )
    )
    pit_agg = pit_agg[pit_agg["ip"] > 0].copy()
    pit_agg.loc[~pit_agg["war_any_nonnull"], "war"] = np.nan
    pit_agg = pit_agg.drop(columns=["war_any_nonnull"])
    pit_agg["position_type"] = "pitcher"
    pit_agg = pit_agg.rename(columns={
        "mlb_ID": "player_id",
        "year_ID": "season",
        "name": "player_name",
        "ip": "pa_or_ip",
    })
    pit_agg = pit_agg[["player_id", "player_name", "season", "position_type", "war", "pa_or_ip"]]
    pit_agg["season"] = pit_agg["season"].astype(int)
    pit_agg["war_source"] = "bref_bwar"

    return bat_agg, pit_agg


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit(bat: pd.DataFrame, pit: pd.DataFrame) -> dict[str, Any]:
    """Summarise staged rows for the audit JSON."""
    summary: dict[str, Any] = {
        "n_batter_rows": int(len(bat)),
        "n_pitcher_rows": int(len(pit)),
        "batter_nonnull_war": int(bat["war"].notna().sum()),
        "pitcher_nonnull_war": int(pit["war"].notna().sum()),
    }
    per_year = []
    years = sorted(set(bat["season"].tolist()) | set(pit["season"].tolist()))
    for y in years:
        by = bat[bat["season"] == y]
        py = pit[pit["season"] == y]
        per_year.append({
            "season": int(y),
            "batters": int(len(by)),
            "pitchers": int(len(py)),
            "batters_nonnull_war": int(by["war"].notna().sum()),
            "pitchers_nonnull_war": int(py["war"].notna().sum()),
            "war_mean_bat": float(by["war"].mean()) if len(by) else None,
            "war_std_bat": float(by["war"].std()) if len(by) else None,
            "war_mean_pit": float(py["war"].mean()) if len(py) else None,
            "war_std_pit": float(py["war"].std()) if len(py) else None,
        })
    summary["per_year"] = per_year
    if len(bat):
        summary["batter_war_p01"] = float(np.nanpercentile(bat["war"], 1))
        summary["batter_war_p99"] = float(np.nanpercentile(bat["war"], 99))
    if len(pit):
        summary["pitcher_war_p01"] = float(np.nanpercentile(pit["war"], 1))
        summary["pitcher_war_p99"] = float(np.nanpercentile(pit["war"], 99))
    return summary


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def run(
    years: tuple[int, ...] = DEFAULT_YEARS,
    *,
    staging_bat: Path = STAGING_BAT,
    staging_pit: Path = STAGING_PIT,
    audit_path: Path = AUDIT_PATH,
) -> dict[str, Any]:
    """Stage pre-2015 season WAR to parquet. Does NOT touch DuckDB."""
    t0 = time.time()
    bat, pit = fetch_war_for_years(years)

    staging_bat.parent.mkdir(parents=True, exist_ok=True)
    bat.to_parquet(staging_bat, index=False)
    pit.to_parquet(staging_pit, index=False)
    logger.info("Wrote staging parquets:\n  %s (%d rows)\n  %s (%d rows)",
                staging_bat, len(bat), staging_pit, len(pit))

    summary = audit(bat, pit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote audit JSON: %s", audit_path)

    elapsed = time.time() - t0
    logger.info("Total wall-clock: %.1fs", elapsed)
    return {
        "audit": summary,
        "elapsed_seconds": elapsed,
        "staging_bat": str(staging_bat),
        "staging_pit": str(staging_pit),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--start-year", type=int, default=2010)
    p.add_argument("--end-year", type=int, default=2014)
    p.add_argument("--staging-bat", type=Path, default=STAGING_BAT)
    p.add_argument("--staging-pit", type=Path, default=STAGING_PIT)
    p.add_argument("--audit-path", type=Path, default=AUDIT_PATH)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    years = tuple(range(args.start_year, args.end_year + 1))
    try:
        run(
            years,
            staging_bat=args.staging_bat,
            staging_pit=args.staging_pit,
            audit_path=args.audit_path,
        )
        return 0
    except Exception as exc:  # pragma: no cover
        logger.exception("backfill_season_war_2010_2014 failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
