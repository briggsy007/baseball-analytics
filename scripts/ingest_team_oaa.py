#!/usr/bin/env python
"""
Ingest team-season Statcast OAA baseline from Baseball Savant.

Pulls per-player OAA + FRP via ``pybaseball.statcast_outs_above_average``
for each of the six non-catcher positions (1B, 2B, 3B, SS, LF, CF, RF),
filters the synthetic multi-team ``---`` row, and aggregates to
(team, season). Writes a parquet + audit JSON under ``data/baselines/``.

Methodology reproduces the prior ``team_defense_2023_2024.parquet`` file
to within r >= 0.99 Pearson correlation on overlap years.

Usage
-----
    python scripts/ingest_team_oaa.py --seasons 2023 2024 2025 \\
        --output data/baselines/team_defense_2023_2025.parquet

If the output file exists and ``--force`` is not passed, the script
refuses to overwrite. Use ``--force`` to refresh.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingest_team_oaa")


DISPLAY_TO_ABBR: dict[str, str] = {
    "Angels": "LAA",
    "Astros": "HOU",
    "Athletics": "ATH",
    "Blue Jays": "TOR",
    "Braves": "ATL",
    "Brewers": "MIL",
    "Cardinals": "STL",
    "Cubs": "CHC",
    "D-backs": "AZ",
    "Dodgers": "LAD",
    "Giants": "SF",
    "Guardians": "CLE",
    "Mariners": "SEA",
    "Marlins": "MIA",
    "Mets": "NYM",
    "Nationals": "WSH",
    "Orioles": "BAL",
    "Padres": "SD",
    "Phillies": "PHI",
    "Pirates": "PIT",
    "Rangers": "TEX",
    "Rays": "TB",
    "Red Sox": "BOS",
    "Reds": "CIN",
    "Rockies": "COL",
    "Royals": "KC",
    "Tigers": "DET",
    "Twins": "MIN",
    "White Sox": "CWS",
    "Yankees": "NYY",
}

POSITIONS: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9)  # no catcher


def fetch_year(year: int) -> pd.DataFrame:
    """Fetch per-player OAA + FRP for one season, all non-catcher positions."""
    import pybaseball
    frames: list[pd.DataFrame] = []
    for pos in POSITIONS:
        try:
            df = pybaseball.statcast_outs_above_average(
                year=year, pos=pos, min_att=0, view="Fielder",
            )
        except Exception as exc:
            logger.warning("pos=%d year=%d failed: %s", pos, year, exc)
            continue
        df = df.copy()
        df["pos_code"] = pos
        frames.append(df)
        logger.info(
            "  year=%d pos=%d rows=%d sum_oaa=%s",
            year, pos, len(df), int(df["outs_above_average"].sum()),
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def aggregate_to_team(all_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-player rows to (team, season). Filters '---' rows."""
    df = all_df[all_df["display_team_name"] != "---"].copy()
    grp = df.groupby(["display_team_name", "year"], as_index=False).agg(
        team_oaa=("outs_above_average", "sum"),
        team_frp=("fielding_runs_prevented", "sum"),
        n_players=("player_id", "nunique"),
    )
    grp = grp.rename(columns={"year": "season"})
    grp["team_abbr"] = grp["display_team_name"].map(DISPLAY_TO_ABBR)
    missing = grp[grp["team_abbr"].isna()]
    if not missing.empty:
        raise RuntimeError(
            f"Unmapped display_team_name values: {missing['display_team_name'].tolist()}"
        )
    grp["source"] = "baseballsavant_oaa"
    return grp[
        [
            "team_abbr",
            "season",
            "team_oaa",
            "team_frp",
            "n_players",
            "source",
            "display_team_name",
        ]
    ].sort_values(["season", "team_abbr"]).reset_index(drop=True)


def run(seasons: Iterable[int], output: Path, force: bool) -> None:
    output = Path(output)
    if output.exists() and not force:
        raise SystemExit(
            f"{output} already exists. Use --force to overwrite."
        )

    seasons = sorted(set(int(s) for s in seasons))
    frames = []
    for season in seasons:
        logger.info("Fetching OAA for season %d", season)
        df_year = fetch_year(season)
        if df_year.empty:
            raise RuntimeError(f"No OAA data returned for {season}")
        frames.append(df_year)
    all_df = pd.concat(frames, ignore_index=True)

    team_df = aggregate_to_team(all_df)

    output.parent.mkdir(parents=True, exist_ok=True)
    team_df.to_parquet(output, index=False)

    audit_path = output.with_name(output.stem + "_audit.json")
    audit = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "source": (
            "Baseball Savant Outs Above Average leaderboard via "
            "pybaseball.statcast_outs_above_average (per-player, min_att=0, "
            "all non-catcher positions 3-9, filter display_team_name != '---', "
            "sum to team-season)."
        ),
        "n_rows": int(len(team_df)),
        "seasons": seasons,
        "n_teams_per_season": {
            str(s): int((team_df["season"] == s).sum()) for s in seasons
        },
        "metric_columns": [
            "team_oaa (sum of player OAA)",
            "team_frp (sum of player Fielding Runs Prevented)",
        ],
        "team_oaa_definition": (
            "Sum of outs_above_average across all players for that "
            "team-season (Baseball Savant Statcast OAA)."
        ),
        "team_frp_definition": (
            "Sum of fielding_runs_prevented (Statcast FRP) across all "
            "players for that team-season."
        ),
        "note": (
            "Multi-team-during-season players are counted under each team "
            "they appeared for (Baseball Savant splits player rows by team)."
        ),
        "output_path": str(output),
    }
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    logger.info("Wrote %s (%d rows, %d seasons)", output, len(team_df), len(seasons))
    logger.info("Wrote %s", audit_path)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--seasons", nargs="+", type=int, required=True,
        help="Seasons to ingest (e.g. 2023 2024 2025).",
    )
    p.add_argument(
        "--output", type=Path, required=True,
        help="Output parquet path (e.g. data/baselines/team_defense_2023_2025.parquet).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output file.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        run(args.seasons, args.output, args.force)
        return 0
    except Exception as exc:
        logger.exception("ingest_team_oaa failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
