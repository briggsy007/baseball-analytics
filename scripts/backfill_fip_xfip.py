#!/usr/bin/env python
"""
Compute FIP and xFIP per (pitcher_id, season) from raw ``pitches`` data and
write back to ``season_pitching_stats.fip`` / ``.xfip``.

Formulas
--------
    uncFIP = ((13*HR) + (3*(BB+HBP)) - (2*K)) / IP
    cFIP   = league_ERA - league_uncFIP            (per season)
    FIP    = uncFIP + cFIP

    lgHRFB = sum(HR) / sum(HR + FB)               (per season)
    xHR    = (FB + HR) * lgHRFB                   (per pitcher-season)
    uncxFIP = ((13*xHR) + (3*(BB+HBP)) - (2*K)) / IP
    xFIP    = uncxFIP + cFIP

Event mapping (pitches.events)
------------------------------
    BB  : 'walk' OR 'intent_walk'        (canonical FIP includes IBB)
    HBP : 'hit_by_pitch'
    K   : 'strikeout' OR 'strikeout_double_play'
    HR  : 'home_run'
    FB  : pitches.bb_type == 'fly_ball' (excludes popup, per FG xFIP convention)

IP source
---------
``season_pitching_stats.ip`` (already populated for every row) — but it is
encoded in baseball "X.1 = X+1/3, X.2 = X+2/3" notation, so we convert to
true thirds before dividing.

Idempotent: re-running overwrites with the same values.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db.schema import DEFAULT_DB_PATH  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backfill_fip_xfip")


def _ip_to_true(ip_baseball: pd.Series) -> pd.Series:
    """Convert baseball-notation IP (e.g. 50.1 = 50 1/3) to true decimal IP."""
    whole = np.floor(ip_baseball)
    frac = (ip_baseball - whole).round(2)
    # 0.1 -> 1/3, 0.2 -> 2/3, 0.0 -> 0
    third = np.where(np.isclose(frac, 0.1), 1 / 3,
             np.where(np.isclose(frac, 0.2), 2 / 3, 0.0))
    return whole + third


def aggregate_pitches(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per-(pitcher_id, season) BB, HBP, K, HR, FB, plus league HR/FB rate."""
    sql = """
    WITH last_pitch AS (
        -- one row per plate appearance (the resolution pitch)
        SELECT pitcher_id,
               EXTRACT(YEAR FROM game_date)::INT AS season,
               events,
               bb_type
        FROM pitches
        WHERE events IS NOT NULL
    )
    SELECT pitcher_id,
           season,
           SUM(CASE WHEN events IN ('walk', 'intent_walk') THEN 1 ELSE 0 END) AS bb,
           SUM(CASE WHEN events = 'hit_by_pitch' THEN 1 ELSE 0 END) AS hbp,
           SUM(CASE WHEN events IN ('strikeout', 'strikeout_double_play') THEN 1 ELSE 0 END) AS k,
           SUM(CASE WHEN events = 'home_run' THEN 1 ELSE 0 END) AS hr,
           SUM(CASE WHEN bb_type = 'fly_ball' THEN 1 ELSE 0 END) AS fb
    FROM last_pitch
    GROUP BY pitcher_id, season
    """
    return conn.execute(sql).fetchdf()


def compute_fip_xfip(
    pitch_agg: pd.DataFrame,
    season_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_pitcher_df, per_season_constants_df) with FIP/xFIP filled."""
    # Join IP and ERA from season_pitching_stats
    pitch_agg = pitch_agg.rename(columns={"pitcher_id": "player_id"})
    df = season_stats.merge(
        pitch_agg, on=["player_id", "season"], how="left",
        suffixes=("", "_pa"),
    ).rename(columns={"player_id": "pitcher_id"})
    # Fill missing event aggregates with 0 (pitcher has season row but no
    # statcast events, e.g. pre-2015 or pitchers without resolution events)
    for col in ("bb", "hbp", "k", "hr", "fb"):
        df[col] = df[col].fillna(0).astype(int)

    df["ip_true"] = _ip_to_true(df["ip"])

    # League aggregates per season
    league_rows = []
    for season, grp in df.groupby("season"):
        ip_total = grp["ip_true"].sum()
        if ip_total <= 0:
            continue
        bb = grp["bb"].sum()
        hbp = grp["hbp"].sum()
        k = grp["k"].sum()
        hr = grp["hr"].sum()
        fb = grp["fb"].sum()
        # League uncFIP
        uncfip_lg = ((13 * hr) + (3 * (bb + hbp)) - (2 * k)) / ip_total
        # League ERA from raw data: no ER column; fall back to mean(weighted by IP) of season_pitching_stats.era
        # When all rows share a season, weighted-by-IP ERA is the league ERA proxy
        valid = grp.dropna(subset=["era"])
        if len(valid):
            lg_era = float((valid["era"] * valid["ip_true"]).sum() / valid["ip_true"].sum())
        else:
            lg_era = np.nan
        cfip = lg_era - uncfip_lg if np.isfinite(lg_era) else np.nan
        # League HR/FB
        denom = hr + fb
        lg_hrfb = (hr / denom) if denom > 0 else np.nan
        league_rows.append({
            "season": int(season),
            "lg_era": lg_era,
            "lg_uncfip": uncfip_lg,
            "cfip": cfip,
            "lg_hrfb": lg_hrfb,
            "ip_total": ip_total,
            "hr_total": int(hr),
            "fb_total": int(fb),
            "k_total": int(k),
            "bb_total": int(bb),
            "hbp_total": int(hbp),
        })
    league = pd.DataFrame(league_rows)

    # Per-pitcher FIP / xFIP
    df = df.merge(league[["season", "cfip", "lg_hrfb"]], on="season", how="left")
    # FIP: skip rows with IP <= 0
    mask = df["ip_true"] > 0
    df["uncfip"] = np.where(
        mask,
        ((13 * df["hr"]) + (3 * (df["bb"] + df["hbp"])) - (2 * df["k"])) / df["ip_true"].replace(0, np.nan),
        np.nan,
    )
    df["fip_calc"] = df["uncfip"] + df["cfip"]

    # xFIP uses expected HR = (FB + HR) * lgHRFB
    df["xhr"] = (df["fb"] + df["hr"]) * df["lg_hrfb"]
    df["uncxfip"] = np.where(
        mask,
        ((13 * df["xhr"]) + (3 * (df["bb"] + df["hbp"])) - (2 * df["k"])) / df["ip_true"].replace(0, np.nan),
        np.nan,
    )
    df["xfip_calc"] = df["uncxfip"] + df["cfip"]

    # If a pitcher had zero events recorded (no statcast resolution rows for
    # that season -- can happen for very early seasons or rare bullpens),
    # FIP will be cFIP itself (uncfip=0).  That's noisy; mark as NaN if
    # the pitcher recorded literally zero of the four FIP events.
    no_events = (df["bb"] + df["hbp"] + df["k"] + df["hr"]) == 0
    df.loc[no_events, ["fip_calc", "xfip_calc"]] = np.nan

    out = df[["pitcher_id", "season", "fip_calc", "xfip_calc"]].rename(
        columns={"pitcher_id": "player_id", "fip_calc": "fip", "xfip_calc": "xfip"}
    )
    return out, league


def _connect_write(db_path: Path, max_wait_seconds: int = 15 * 60) -> duckdb.DuckDBPyConnection:
    start = time.time()
    backoff = 30
    attempts = 0
    while True:
        attempts += 1
        try:
            return duckdb.connect(str(db_path), read_only=False)
        except duckdb.IOException as exc:
            elapsed = time.time() - start
            if elapsed > max_wait_seconds:
                raise RuntimeError(
                    f"Could not acquire DB write lock after {elapsed:.0f}s "
                    f"({attempts} attempts): {exc}"
                ) from exc
            logger.warning(
                "DB write lock busy (attempt %d, %.0fs elapsed): %s -- sleeping %ds",
                attempts, elapsed, exc, backoff,
            )
            time.sleep(backoff)


def run(db_path: Path = DEFAULT_DB_PATH) -> dict:
    with duckdb.connect(str(db_path), read_only=True) as conn:
        before_fip, before_xfip, total = conn.execute(
            "SELECT SUM(CASE WHEN fip IS NOT NULL THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN xfip IS NOT NULL THEN 1 ELSE 0 END), COUNT(*) "
            "FROM season_pitching_stats"
        ).fetchone()
        season_stats = conn.execute(
            "SELECT player_id, season, ip, era FROM season_pitching_stats"
        ).fetchdf()
        logger.info("aggregating pitches (this may take ~30-60s for 7M rows)")
        pitch_agg = aggregate_pitches(conn)
    logger.info("pitch_agg shape: %s", pitch_agg.shape)
    logger.info(
        "before: total=%d with_fip=%s with_xfip=%s",
        total, before_fip, before_xfip,
    )

    out, league = compute_fip_xfip(pitch_agg, season_stats)
    logger.info("league constants per season:\n%s", league.to_string(index=False))

    # Filter to rows we can actually write
    out_valid = out.dropna(subset=["fip"]).copy()
    out_valid["player_id"] = out_valid["player_id"].astype("int64")
    out_valid["season"] = out_valid["season"].astype("int64")
    out_valid["fip"] = out_valid["fip"].astype(float)
    out_valid["xfip"] = out_valid["xfip"].astype(float)

    conn = _connect_write(db_path)
    try:
        conn.register("fip_updates", out_valid)
        conn.execute("BEGIN TRANSACTION")
        conn.execute(
            """
            UPDATE season_pitching_stats AS s
            SET fip = u.fip,
                xfip = u.xfip
            FROM fip_updates AS u
            WHERE s.player_id = u.player_id AND s.season = u.season
            """
        )
        conn.execute("COMMIT")
        conn.unregister("fip_updates")
        after_fip, after_xfip = conn.execute(
            "SELECT SUM(CASE WHEN fip IS NOT NULL THEN 1 ELSE 0 END), "
            "SUM(CASE WHEN xfip IS NOT NULL THEN 1 ELSE 0 END) "
            "FROM season_pitching_stats"
        ).fetchone()
    finally:
        conn.close()

    logger.info("after: with_fip=%s with_xfip=%s (delta_fip=%s)",
                after_fip, after_xfip, (after_fip or 0) - (before_fip or 0))
    return {
        "before_fip": int(before_fip or 0),
        "after_fip": int(after_fip or 0),
        "before_xfip": int(before_xfip or 0),
        "after_xfip": int(after_xfip or 0),
        "league": league.to_dict(orient="records"),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    try:
        run(db_path=args.db_path)
        return 0
    except Exception as exc:
        logger.exception("backfill_fip_xfip failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
