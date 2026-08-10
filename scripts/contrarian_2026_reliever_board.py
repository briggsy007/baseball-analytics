#!/usr/bin/env python
"""
Generate + freeze the 2026 Contrarian RELIEVER Board (plan WS1.3, spec section 6).

What this is
------------
The pitcher-side companion to the batter-only 2026 mid-season contrarian
board. Its criterion was pre-registered BEFORE this board existed, in the
FROZEN resolution spec ``docs/models/contrarian_2026_resolution_spec.md``
section 6 (sha256-at-freeze
``1a27cd0e2d9b7d08c69c5a8a5944602585931121d9396be922e1e519557c760e``,
commit ``912ede6``). This script follows section 6 mechanically -- it does
not re-derive any criterion:

* CausalWAR pitcher effects for 2026 from the frozen 2015-2022 nuisance
  checkpoint ``models/causal_war/causal_war_trainsplit_2015_2022.pkl``,
  pitcher aggregation ``pa_min=50`` (stability-script convention) (s6.1);
* progress constant ``p_f`` = median team games through the baseline as-of
  date / 162, 4 dp, from the spec s4.4 query (s6.2);
* reliever band: ``IP >= round(20 * p_f)`` AND ``IP < 60 * p_f`` (upper
  bound full precision) (s6.3);
* ranks WITHIN the qualified reliever pool, ``method="min"`` descending;
  Buy-Low requires ``rank_diff > 0``, Over-Valued ``rank_diff < 0``; up to
  25 per side; boundary ties broken by larger ``|causal_war - trad_war|``
  then lower ``player_id`` (s6.4).

Baseline basis (declared deviation, spec s8 entry 2)
----------------------------------------------------
Section 6.1 asks for a *fresh* season-to-date bWAR staging fetch at freeze.
This freeze runs under Batch-B constraints (read-only DB, no network
fetches), so the baseline basis is the closest surviving dated fetch --
artifact A3, ``war_2026_staging_asof_2026-08-09.parquet`` (byte-identical
to the live ``data/staging/war_2026_staging.parquet`` at freeze time; this
script verifies the sha256 and aborts on mismatch). ``p_f`` is measured
through the SAME as-of date (2026-08-09) so the parquet is both the pick
baseline and the neighbor baseline with no basis offset -- exactly the
single-basis rule s6.6 requires.

What this is NOT
----------------
NOT a validated hit-rate claim. The reliever-leverage cohort is the only
historical cohort that cleared its within-filter naive base rate
(25/32 = 78.1% vs 56.9%), but n=32 is small, the rate aggregates three
windows, and the hit rule inherits the post-hoc criterion (2026-08-10
audit section 2). That historical rate does NOT transfer to these picks.
These picks are UNRESOLVED; resolution happens once, mechanically, per the
frozen spec (hypothesis H4; K4 publication rule: a miss publishes as
prominently as a win).

Outputs
-------
* ``results/edges/contrarian_2026_midseason/2026-08-10/reliever_board.csv``
  and ``reliever_summary.md`` (dated dir, alongside the batter board);
* freeze archives + sha256s under
  ``results/edges/contrarian_2026_midseason/frozen_2026_resolution_inputs/``;
* one prospective pick per board row appended to ``predictions/picks.jsonl``
  via ``src.pick_ledger.emit_pick`` (append-only; re-runs skip existing ids);
* ``latest.json`` gains reliever pointer keys (atomic replace).

Reads the DB strictly ``read_only=True``. Zero DB writes.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db.schema import get_connection  # noqa: E402
from src.pick_ledger import emit_pick, load_ledger, LedgerError  # noqa: E402
from src.analytics.causal_war import (  # noqa: E402
    _aggregate_player_effects,
    _build_features,
    _extract_pa_data,
)
from src.dashboard.views.contrarian_leaderboards import (  # noqa: E402
    _classify_row as classify_row,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("contrarian_2026_reliever")

SEASON = 2026
FREEZE_DATE = "2026-08-10"          # the dated-dir / archive date
BASELINE_ASOF = "2026-08-09"        # A3 staging-fetch as-of date (declared, spec s4.4 tiebreak)
TOP_N = 25
PITCHER_PA_MIN = 50                 # s6.1: stability-script pitcher aggregation convention

# Frozen resolution spec (s5.7: rule_hash = sha256 of the spec at the freeze commit).
SPEC_PATH = "docs/models/contrarian_2026_resolution_spec.md"
SPEC_SHA256 = "1a27cd0e2d9b7d08c69c5a8a5944602585931121d9396be922e1e519557c760e"
SPEC_COMMIT = "912ede6"

CHECKPOINT = ROOT / "models" / "causal_war" / "causal_war_trainsplit_2015_2022.pkl"

OUT_DIR = ROOT / "results" / "edges" / "contrarian_2026_midseason"
DATED_DIR = OUT_DIR / FREEZE_DATE
FROZEN_DIR = OUT_DIR / "frozen_2026_resolution_inputs"
STAGING_PARQUET = ROOT / "data" / "staging" / "war_2026_staging.parquet"
A3_PARQUET = FROZEN_DIR / f"war_2026_staging_asof_{BASELINE_ASOF}.parquet"
# Spec s4.1 A3 sha256 (upper-case hex as printed in the spec).
A3_SHA256 = "0DF755CF2C3AC30BEE4C4EFD2FF1CA9ABEF47C755E4F36DD9986D8B839249EDD"
LATEST_POINTER = OUT_DIR / "latest.json"

RESOLVE_BY = "2026-10-11"  # calendar estimate; the spec s7.1 formula governs

_PHI = "PHI"

# Detached-claims banner: same language family as the batter board
# (scripts/contrarian_2026_midseason.py::_BANNER), reliever-side specifics
# and the n=32 caveat added. Registry entry: causal_war_reliever_tag.
_BANNER = (
    "APPLICATION of the 2023-2025 CausalWAR contrarian methodology to "
    "PARTIAL-SEASON 2026 data -- pitcher/reliever side. NOT a validated "
    "edge. The reliever-leverage cohort is the only historical cohort that "
    "cleared its within-filter naive base rate (Buy-Low 25/32 = 78.1% vs "
    "56.9% within-filter naive), but n=32 is SMALL, the rate aggregates "
    "three windows, and the hit rule inherits the post-hoc criterion "
    "(2026-08-10 audit). That historical rate does NOT transfer to these "
    "picks. The batter-side headline (68.4%, 13/19) is likewise NOT a "
    "validated edge (post-hoc hit criterion; 95% CI [0.474, 0.843] "
    "includes chance; intention-to-treat is 13/25 = 52%; matched-naive "
    "baselines score 66.5-73% on the same pools). 2026 picks here are "
    "UNRESOLVED. Resolution is PRE-REGISTERED and mechanical: frozen spec "
    f"`{SPEC_PATH}` section 6 (hypothesis H4) -- ITT scoring with the "
    "section-5.3 exit asymmetry, WITHIN-FILTER matched-naive control, "
    "pitcher-Marcel RA9 control, one resolution event at end of 2026 "
    "regular season + 7 days; a miss publishes as prominently as a win "
    f"(K4). Spec sha256 at freeze: `{SPEC_SHA256}` (commit `{SPEC_COMMIT}`)."
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


# ---------------------------------------------------------------------------
# Frozen-basis guards + progress constant
# ---------------------------------------------------------------------------


def verify_frozen_basis() -> dict[str, str]:
    """Abort unless the staging parquet is byte-identical to archived A3."""
    if not A3_PARQUET.exists():
        raise RuntimeError(f"Frozen baseline A3 missing: {A3_PARQUET}")
    a3 = _sha256(A3_PARQUET)
    if a3 != A3_SHA256:
        raise RuntimeError(
            f"A3 sha256 mismatch: {a3} != spec-registered {A3_SHA256}"
        )
    live = _sha256(STAGING_PARQUET) if STAGING_PARQUET.exists() else "MISSING"
    if live != A3_SHA256:
        raise RuntimeError(
            "data/staging/war_2026_staging.parquet no longer matches frozen "
            f"A3 ({live} != {A3_SHA256}). The reliever board must be built "
            "on the frozen A3 basis; refusing to mix bases (spec s6.6)."
        )
    logger.info("Frozen basis verified: staging parquet == A3 (%s)", A3_SHA256)
    return {"a3_sha256": a3, "staging_sha256": live}


def measure_p_f(conn) -> tuple[float, int, str]:
    """Spec s4.4 progress query at d = BASELINE_ASOF; returns (p_f, median_games, max_game_date)."""
    q = f"""
        WITH per_team AS (
            SELECT team, COUNT(DISTINCT game_pk) AS g FROM (
                SELECT home_team AS team, game_pk FROM pitches
                WHERE game_date BETWEEN DATE '{SEASON}-03-01' AND DATE '{BASELINE_ASOF}'
                UNION ALL
                SELECT away_team AS team, game_pk FROM pitches
                WHERE game_date BETWEEN DATE '{SEASON}-03-01' AND DATE '{BASELINE_ASOF}'
            ) GROUP BY team
        )
        SELECT MEDIAN(g), COUNT(*) FROM per_team
    """
    med, n_teams = conn.execute(q).fetchone()
    if n_teams != 30:
        raise RuntimeError(f"Expected 30 teams in progress query, got {n_teams}")
    p_f = round(float(med) / 162.0, 4)
    max_date = conn.execute(
        f"SELECT MAX(game_date) FROM pitches WHERE game_date >= DATE '{SEASON}-01-01'"
    ).fetchone()[0]
    return p_f, int(med), str(max_date)


# ---------------------------------------------------------------------------
# CausalWAR pitcher effects (s6.1: frozen checkpoint, pa_min=50)
# ---------------------------------------------------------------------------


def compute_pitcher_effects(conn) -> pd.DataFrame:
    art = joblib.load(CHECKPOINT)
    nuisance = art["nuisance_outcome"]
    logger.info("Loaded frozen nuisance checkpoint %s", CHECKPOINT)

    logger.info("Extracting %d PA data (read-only) ...", SEASON)
    pa_df = _extract_pa_data(conn, year_range=(SEASON, SEASON))
    logger.info("  %d PAs", len(pa_df))

    # v1 feature layout: the frozen 2015-2022 checkpoint was trained on the
    # 12 LEGACY confounders only. ``_build_features`` appends the optional
    # ump/weather block whenever those columns exist in the frame (added to
    # the extractor for v2), so drop them first to reproduce the v1 path
    # verbatim (12-column W, same order).
    _V2_ONLY_COLS = [
        "ump_accuracy_above_x", "ump_favor", "temp_f", "wind_speed_mph",
        "wind_dir_deg", "humidity_pct", "pressure_mb", "roof_status",
    ]
    pa_df = pa_df.drop(columns=[c for c in _V2_ONLY_COLS if c in pa_df.columns])

    W, Y, _batter_ids, pa_df_built = _build_features(pa_df)
    if W.shape[1] != 12:
        raise RuntimeError(
            f"v1 checkpoint expects 12 confounders, built {W.shape[1]} -- "
            "feature layout drifted; refusing to score."
        )
    Y_res = Y - nuisance.predict(W)

    pitcher_ids = pa_df_built["pitcher_id"].to_numpy().astype(int)
    effects = _aggregate_player_effects(
        -Y_res, pitcher_ids, pa_df_built, pa_min=PITCHER_PA_MIN
    )
    rows = [
        {
            "player_id": int(pid),
            "causal_war": float(e["causal_war"]),
            "point_estimate_per_pa": float(e["point_estimate_per_pa"]),
            "bf_total": int(e["pa"]),
        }
        for pid, e in effects.items()
    ]
    df = pd.DataFrame(rows)
    logger.info("  %d pitcher effects at pa_min=%d", len(df), PITCHER_PA_MIN)
    return df


# ---------------------------------------------------------------------------
# Names / teams (read-only convenience lookups)
# ---------------------------------------------------------------------------


def load_names(conn, ids: list[int]) -> dict[int, str]:
    if not ids:
        return {}
    ph = ",".join(["?"] * len(ids))
    return {
        int(pid): str(name)
        for pid, name in conn.execute(
            f"SELECT player_id, full_name FROM players WHERE player_id IN ({ph})", ids
        ).fetchall()
        if name
    }


def derive_pitcher_teams(conn, ids: list[int]) -> dict[int, str]:
    """Modal 2026 team per pitcher: home team pitches the Top half, away the Bot."""
    if not ids:
        return {}
    ph = ",".join(["?"] * len(ids))
    q = f"""
        WITH b AS (
            SELECT pitcher_id AS player_id,
                   CASE WHEN inning_topbot = 'Top' THEN home_team ELSE away_team END AS team,
                   COUNT(*) AS n
            FROM pitches
            WHERE pitcher_id IN ({ph}) AND game_date >= DATE '{SEASON}-01-01'
            GROUP BY 1, 2
        ),
        ranked AS (
            SELECT player_id, team, n,
                   ROW_NUMBER() OVER (PARTITION BY player_id ORDER BY n DESC) AS rn
            FROM b
        )
        SELECT player_id, team FROM ranked WHERE rn = 1
    """
    try:
        return {
            int(pid): str(team)
            for pid, team in conn.execute(q, ids).fetchall()
            if team
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Board construction (s6.3 band, s6.4 ranks/sides/tiebreaks)
# ---------------------------------------------------------------------------


def build_board(conn, p_f: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    effects = compute_pitcher_effects(conn)

    fg = pd.read_parquet(A3_PARQUET)
    pit = fg[fg["position_type"] == "pitcher"][
        ["player_id", "player_name", "war", "pa_or_ip"]
    ].rename(columns={"war": "trad_war", "pa_or_ip": "ip_total"})
    pit["player_id"] = pit["player_id"].astype("int64")
    n_parquet_pitchers = len(pit)

    ip_lower = round(20 * p_f)          # s6.3: round(20 * p_f)
    ip_upper = 60 * p_f                 # s6.3: full precision, no rounding
    band = pit[(pit["ip_total"] >= ip_lower) & (pit["ip_total"] < ip_upper)].copy()
    band = band.dropna(subset=["trad_war"])
    n_band_parquet = len(band)

    merged = band.merge(effects, on="player_id", how="inner")
    merged = merged.dropna(subset=["trad_war", "causal_war"]).copy()
    n_ranked_pool = len(merged)

    meta: dict[str, Any] = {
        "season": SEASON,
        "p_f": p_f,
        "ip_lower": ip_lower,
        "ip_upper_full_precision": ip_upper,
        "pitcher_pa_min": PITCHER_PA_MIN,
        "top_n": TOP_N,
        "n_parquet_pitchers": n_parquet_pitchers,
        "n_ip_band_parquet": n_band_parquet,
        "n_ranked_pool": n_ranked_pool,
    }
    if merged.empty:
        return merged, meta

    # Ranks WITHIN the qualified reliever pool (s6.4).
    merged["rank_causal"] = (
        merged["causal_war"].rank(ascending=False, method="min").astype(int)
    )
    merged["rank_trad"] = (
        merged["trad_war"].rank(ascending=False, method="min").astype(int)
    )
    merged["rank_diff"] = merged["rank_trad"] - merged["rank_causal"]
    merged["abs_gap"] = (merged["causal_war"] - merged["trad_war"]).abs()

    ids = merged["player_id"].astype(int).tolist()
    names = load_names(conn, ids)
    teams = derive_pitcher_teams(conn, ids)
    merged["name"] = merged.apply(
        lambda r: names.get(int(r["player_id"]), str(r["player_name"])), axis=1
    )
    merged["team"] = merged["player_id"].map(lambda p: teams.get(int(p)))
    merged["position"] = "pitcher"
    merged["pa_total"] = np.nan  # for classify_row signature parity

    merged["tag"] = merged.apply(lambda row: classify_row(row, {}), axis=1)

    # Sides: sign REQUIRED (s6.4). Tiebreaks: larger |causal-trad|, lower player_id.
    bl_pool = merged[merged["rank_diff"] > 0].sort_values(
        by=["rank_diff", "abs_gap", "player_id"],
        ascending=[False, False, True],
    )
    ov_pool = merged[merged["rank_diff"] < 0].sort_values(
        by=["rank_diff", "abs_gap", "player_id"],
        ascending=[True, False, True],
    )
    buy_low = bl_pool.head(TOP_N).copy()
    buy_low["board"] = "reliever_buy_low"
    buy_low["side_rank"] = np.arange(1, len(buy_low) + 1)
    over_valued = ov_pool.head(TOP_N).copy()
    over_valued["board"] = "reliever_over_valued"
    over_valued["side_rank"] = np.arange(1, len(over_valued) + 1)

    meta["n_rank_diff_pos"] = int(len(bl_pool))
    meta["n_rank_diff_neg"] = int(len(ov_pool))
    meta["n_rank_diff_zero"] = int((merged["rank_diff"] == 0).sum())

    board = pd.concat([buy_low, over_valued], ignore_index=True)
    cols = [
        "board", "side_rank", "player_id", "name", "team", "position",
        "causal_war", "trad_war", "rank_causal", "rank_trad", "rank_diff",
        "ip_total", "bf_total", "tag",
    ]
    return board[cols], meta


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def _table_md(df: pd.DataFrame, n: int = 10) -> str:
    if df.empty:
        return "_(none)_"
    lines = [
        "| # | Player | Team | CausalWAR | bWAR | RankDiff | IP | Tag |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in df.head(n).itertuples(index=False):
        star = " (PHI)" if r.team == _PHI else ""
        lines.append(
            f"| {r.side_rank} | {r.name}{star} | {r.team or '-'} | "
            f"{r.causal_war:.2f} | {r.trad_war:.2f} | {r.rank_diff:+d} | "
            f"{r.ip_total:.1f} | {r.tag} |"
        )
    return "\n".join(lines)


def write_summary(
    board: pd.DataFrame, meta: dict[str, Any], out_path: Path,
) -> None:
    buy_low = board[board["board"] == "reliever_buy_low"]
    over_valued = board[board["board"] == "reliever_over_valued"]
    p_f = meta["p_f"]
    ip_all = board["ip_total"].astype(float)

    lines: list[str] = []
    lines.append("# Contrarian RELIEVER Board -- 2026 (frozen 2026-08-10)")
    lines.append("")
    lines.append(f"> **{_BANNER}**")
    lines.append("")
    lines.append(
        "_Pitcher-side CausalWAR (frozen 2015-2022 nuisance checkpoint "
        "`models/causal_war/causal_war_trainsplit_2015_2022.pkl`, pitcher "
        f"aggregation >= {meta['pitcher_pa_min']} batters faced) vs 2026 "
        "season-to-date Baseball-Reference bWAR (frozen staging parquet "
        f"as-of {BASELINE_ASOF}, spec artifact A3). DB pitch data through "
        f"{meta['db_max_game_date']}._"
    )
    lines.append("")
    lines.append("## Pre-registered criterion (spec section 6 -- fixed before this board existed)")
    lines.append("")
    lines.append(f"- Progress constant `p_f` = **{p_f:.4f}** "
                 f"(median {meta['median_games']} team games through {BASELINE_ASOF} / 162; spec s6.2).")
    lines.append(f"- Reliever band (s6.3): **IP >= {meta['ip_lower']}** (= round(20 x p_f)) AND "
                 f"**IP < {meta['ip_upper_full_precision']:.4f}** (= 60 x p_f, full precision).")
    lines.append(f"- Ranks within the qualified reliever pool; Buy-Low requires rank_diff > 0, "
                 f"Over-Valued rank_diff < 0; up to {meta['top_n']} per side; ties by larger "
                 "|causal_war - trad_war| then lower player_id (s6.4).")
    lines.append(f"- Scoring (s6.5): pace target T = snapshot bWAR / {p_f:.4f}; exit floor "
                 f"RoS_IP < 30 x (1 - p_f) = {30 * (1 - p_f):.4f}; ERA surrogate "
                 "(IP >= 30, Buy-Low HIT iff ERA <= 4.00); ITT with the s5.3 exit asymmetry.")
    lines.append(f"- Matched-naive (s6.6): WITHIN-FILTER -- neighbors only from the qualified "
                 f"reliever pool, window +/-0.3 x p_f = +/-{0.3 * p_f:.4f}, >= 3 neighbors, "
                 "single-basis (this frozen parquet is both pick and neighbor baseline).")
    lines.append("- Marcel-pitcher control (s6.7): projected RA9, parameters pinned by a dated "
                 "s8 entry before season end; miss the deadline -> VOID-M.")
    lines.append("- Hypothesis H4 (s2): reliever Buy-Low ITT vs within-filter matched-naive ITT. "
                 "K4 publication rule applies identically.")
    lines.append("")
    lines.append("## Pool composition (honest accounting)")
    lines.append("")
    lines.append(f"- Staging-parquet pitcher rows (as-of {BASELINE_ASOF}): {meta['n_parquet_pitchers']}")
    lines.append(f"- In the IP band on the parquet basis: {meta['n_ip_band_parquet']} "
                 "(this is the s6.6 neighbor pool)")
    lines.append(f"- Rankable (band AND >= {meta['pitcher_pa_min']} BF CausalWAR effects): "
                 f"**{meta['n_ranked_pool']}**")
    lines.append(f"- rank_diff > 0: {meta['n_rank_diff_pos']} | rank_diff < 0: "
                 f"{meta['n_rank_diff_neg']} | rank_diff = 0 (excluded by the s6.4 sign "
                 f"requirement): {meta['n_rank_diff_zero']}")
    lines.append(f"- Board sides: Buy-Low **{len(buy_low)}** / Over-Valued **{len(over_valued)}** "
                 f"(spec allows up to {meta['top_n']}; shortfalls are the spec's own branch, "
                 "not stretched)")
    lines.append(f"- Board IP distribution: min {ip_all.min():.1f} / median {ip_all.median():.1f} "
                 f"/ max {ip_all.max():.1f}")

    def _sign_mix(s: pd.Series) -> str:
        return (f"spans {s.min():+.2f} to {s.max():+.2f} "
                f"({int((s < 0).sum())} negative / {int((s == 0).sum())} zero / "
                f"{int((s > 0).sum())} positive)")

    lines.append("- Baseline-bWAR sign mix (direction-of-effect disclosure, cf. spec s4.5): "
                 f"Buy-Low snapshot bWAR {_sign_mix(buy_low['trad_war'].astype(float))}; "
                 f"Over-Valued {_sign_mix(over_valued['trad_war'].astype(float))}. "
                 "Under the frozen pace rule a NEGATIVE baseline makes the bullish target "
                 "lenient (final bWAR >= a negative number) and the bearish target strict; "
                 "most Buy-Low baselines here are negative. This is the pre-registered "
                 "rule's own behavior -- disclosed, not adjustable after the fact.")
    lines.append("")
    lines.append("## Buy-Low (CausalWAR > bWAR), top 10")
    lines.append("")
    lines.append(_table_md(buy_low))
    lines.append("")
    lines.append("## Over-Valued (bWAR > CausalWAR), top 10")
    lines.append("")
    lines.append(_table_md(over_valued))
    lines.append("")
    lines.append("## Freeze provenance")
    lines.append("")
    lines.append(f"- Board CSV sha256: `{meta['board_sha256']}`")
    lines.append(f"- Baseline parquet (A3) sha256: `{A3_SHA256}` (as-of {BASELINE_ASOF})")
    lines.append(f"- Nuisance checkpoint sha256: `{meta['checkpoint_sha256']}`")
    lines.append(f"- Frozen spec: `{SPEC_PATH}` sha256-at-freeze `{SPEC_SHA256}` (commit `{SPEC_COMMIT}`)")
    lines.append(f"- Picks appended to `predictions/picks.jsonl` "
                 f"({meta['n_picks_emitted']} emitted, {meta['n_picks_skipped']} already present); "
                 "evidence_class `prospective`.")
    lines.append("- Deviation record: spec s8 entry 2 (no fresh fetch under Batch-B "
                 "read-only constraints; A3 reused as the freeze fetch; p_f measured "
                 "through the A3 as-of date for single-basis consistency with s6.6).")
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Ledger emission (s5.7 / s6; one prospective pick per row)
# ---------------------------------------------------------------------------


def _resolution_rule_text(p_f: float) -> str:
    exit_floor = 30 * (1 - p_f)
    return (
        f"Per the FROZEN spec {SPEC_PATH} (sha256-at-freeze {SPEC_SHA256}, "
        f"commit {SPEC_COMMIT}), section 6 (reliever board): the section-5.2 "
        "algorithm with pitcher variables -- global/player voids -> exit "
        f"check RoS_IP < 30 x (1 - p_f) = {exit_floor:.4f} -> WAR branch "
        f"with pace target T = snapshot bWAR / {p_f:.4f}: buy_low HIT iff "
        "final 2026 bWAR >= T, over_valued HIT iff < T -> ERA surrogate "
        "branch (eligibility IP >= 30; buy_low HIT iff ERA <= 4.00, "
        "over_valued HIT iff ERA > 4.00; WAR NULL and surrogate-ineligible "
        "= exit). Section 5.3 ITT accounting (buy_low exit = MISS; "
        "over_valued exit = VOID-EXIT); section 6.6 WITHIN-FILTER "
        "matched-naive (neighbors only from the qualified reliever pool, "
        f"window +/-0.3 x p_f = +/-{0.3 * p_f:.4f}, >= 3 neighbors, "
        "single-basis on the frozen as-of-2026-08-09 parquet); section 6.7 "
        "pitcher-Marcel RA9 control; section 7 resolution procedure (one "
        "resolution event at R = last 2026 regular-season game + 7 days)."
    )


def emit_board_picks(
    board: pd.DataFrame, p_f: float, board_sha256: str, checkpoint_sha256: str,
) -> tuple[int, int]:
    existing = {p.get("pick_id") for p in load_ledger()["picks"]}
    rule_text = _resolution_rule_text(p_f)
    artifact_version = (
        "CausalWAR pitcher effects: frozen nuisance checkpoint "
        f"models/causal_war/causal_war_trainsplit_2015_2022.pkl (sha256 "
        f"{checkpoint_sha256[:16].lower()}...), pa_min={PITCHER_PA_MIN}, 2026 PAs from "
        "DB pitches; baselines: war_2026_staging as-of "
        f"{BASELINE_ASOF} (spec artifact A3)"
    )
    n_emitted = n_skipped = 0
    for r in board.itertuples(index=False):
        pick_id = f"contrarian2026-{r.board}-{int(r.side_rank)}-{int(r.player_id)}"
        if pick_id in existing:
            logger.warning("Pick %s already in ledger; skipping (append-only).", pick_id)
            n_skipped += 1
            continue
        side = "buy_low" if r.board == "reliever_buy_low" else "over_valued"
        wb = float(r.trad_war)
        target = wb / p_f
        if side == "buy_low":
            direction = "bullish"
            claim = (
                f"Better than the market number: 2026 full-season bWAR >= "
                f"{target:.4f} (snapshot pitcher bWAR {wb:.4f} through "
                f"{BASELINE_ASOF}, pace divisor {p_f:.4f}; full float "
                "precision governs at resolution)"
            )
        else:
            direction = "bearish"
            claim = (
                f"Worse than the market number: 2026 full-season bWAR < "
                f"{target:.4f} (snapshot pitcher bWAR {wb:.4f} through "
                f"{BASELINE_ASOF}, pace divisor {p_f:.4f}; full float "
                "precision governs at resolution)"
            )
        pick = {
            "pick_id": pick_id,
            "product": "contrarian_board_2026_midseason_reliever",
            "model": "causal_war",
            "artifact_version": artifact_version,
            "subject": {
                "name": str(r.name),
                "player_id": int(r.player_id),
                "position": "pitcher",
                "team": r.team if isinstance(r.team, str) else None,
            },
            "market_or_claim": claim,
            "p_or_direction": {"direction": direction, "side": side},
            "resolution_rule": rule_text,
            "resolution_source": (
                "Baseball-Reference bWAR (war_daily) fetched via "
                "scripts/backfill_2026_war.py; the dated resolution parquet "
                "under results/edges/contrarian_2026_midseason/resolution/"
                "<date>/ governs (spec section 4.2)."
            ),
            "resolve_by": RESOLVE_BY,
            "resolve_by_note": (
                "Formula governs: last 2026 MLB regular-season game + 7 days "
                "(spec section 7.1); the date here is a calendar estimate "
                "for resolver scheduling only."
            ),
            "rule_hash": SPEC_SHA256,
            "evidence_class": "prospective",
            "board_row": {
                "board_csv_sha256": board_sha256,
                "causal_war": round(float(r.causal_war), 4),
                "trad_war": wb,
                "rank_causal": int(r.rank_causal),
                "rank_trad": int(r.rank_trad),
                "rank_diff": int(r.rank_diff),
                "ip_total": round(float(r.ip_total), 4),
                "bf_total": int(r.bf_total),
                "p_f": p_f,
            },
        }
        try:
            emit_pick(pick)
            n_emitted += 1
        except LedgerError as exc:  # defensive: never crash the freeze on one row
            logger.error("Ledger refused %s: %s", pick_id, exc)
            n_skipped += 1
    return n_emitted, n_skipped


# ---------------------------------------------------------------------------
# latest.json pointer (preserve existing keys; atomic replace)
# ---------------------------------------------------------------------------


def update_latest_pointer() -> None:
    try:
        payload = json.loads(LATEST_POINTER.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    payload["reliever_board_csv"] = f"{FREEZE_DATE}/reliever_board.csv"
    payload["reliever_summary_md"] = f"{FREEZE_DATE}/reliever_summary.md"
    payload["reliever_note"] = (
        "Reliever board is a ONE-TIME freeze (2026-08-10, spec s6). Later "
        "batter-board runs may rewrite this pointer without these keys; the "
        "dashboard falls back to the fixed frozen path "
        f"{FREEZE_DATE}/reliever_board.csv, which never changes."
    )
    payload["written_utc"] = datetime.now(timezone.utc).isoformat()
    tmp = LATEST_POINTER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, LATEST_POINTER)
    logger.info("Updated latest pointer with reliever keys -> %s", LATEST_POINTER)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate() -> int:
    hashes = verify_frozen_basis()
    checkpoint_sha256 = _sha256(CHECKPOINT)

    conn = get_connection(read_only=True)
    try:
        p_f, median_games, db_max_game_date = measure_p_f(conn)
        logger.info(
            "p_f = %.4f (median %d team games through %s; DB pitches through %s)",
            p_f, median_games, BASELINE_ASOF, db_max_game_date,
        )
        board, meta = build_board(conn, p_f)
    finally:
        conn.close()

    if board.empty:
        logger.error("Reliever board is EMPTY after s6 gating (meta=%s). "
                     "Nothing written -- the spec's own branch governs.", meta)
        return 1

    meta["median_games"] = median_games
    meta["db_max_game_date"] = db_max_game_date
    meta["checkpoint_sha256"] = checkpoint_sha256

    DATED_DIR.mkdir(parents=True, exist_ok=True)
    board_csv = DATED_DIR / "reliever_board.csv"
    board.to_csv(board_csv, index=False)
    board_sha256 = _sha256(board_csv)
    meta["board_sha256"] = board_sha256
    logger.info("Wrote %s (%d rows, sha256 %s)", board_csv, len(board), board_sha256)

    # Freeze archives (s6.1): dated board CSV copy + freeze record with hashes.
    FROZEN_DIR.mkdir(parents=True, exist_ok=True)
    frozen_csv = FROZEN_DIR / f"reliever_board_asof_{FREEZE_DATE}.csv"
    shutil.copyfile(board_csv, frozen_csv)
    assert _sha256(frozen_csv) == board_sha256

    # Ledger picks (one per row, prospective).
    n_emitted, n_skipped = emit_board_picks(board, p_f, board_sha256, checkpoint_sha256)
    meta["n_picks_emitted"] = n_emitted
    meta["n_picks_skipped"] = n_skipped
    logger.info("Ledger: %d picks emitted, %d skipped (already present)", n_emitted, n_skipped)

    freeze_record = {
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "spec": {"path": SPEC_PATH, "sha256_at_freeze": SPEC_SHA256, "commit": SPEC_COMMIT},
        "p_f": p_f,
        "p_f_derivation": (
            f"median {median_games} team games through {BASELINE_ASOF} / 162, "
            "4 dp, spec s4.4 query (COUNT(DISTINCT game_pk) per team from "
            "pitches home/away)"
        ),
        "baseline_asof": BASELINE_ASOF,
        "baseline_parquet": {
            "path": str(A3_PARQUET.relative_to(ROOT)).replace("\\", "/"),
            "sha256": hashes["a3_sha256"],
            "note": (
                "Spec artifact A3 reused as the s6.1 freeze fetch (no fresh "
                "network fetch under Batch-B read-only constraints; "
                "live data/staging parquet verified byte-identical at freeze). "
                "Deviations Log entry 2 records this."
            ),
        },
        "nuisance_checkpoint": {
            "path": "models/causal_war/causal_war_trainsplit_2015_2022.pkl",
            "sha256": checkpoint_sha256,
            "pitcher_pa_min": PITCHER_PA_MIN,
        },
        "db_pitches_max_game_date_2026": db_max_game_date,
        "board_csv": {
            "dated": str(board_csv.relative_to(ROOT)).replace("\\", "/"),
            "frozen_copy": str(frozen_csv.relative_to(ROOT)).replace("\\", "/"),
            "sha256": board_sha256,
            "n_rows": int(len(board)),
            "n_buy_low": int((board["board"] == "reliever_buy_low").sum()),
            "n_over_valued": int((board["board"] == "reliever_over_valued").sum()),
        },
        "constants": {
            "ip_lower": meta["ip_lower"],
            "ip_upper_full_precision": meta["ip_upper_full_precision"],
            "pace_divisor": p_f,
            "exit_floor_ros_ip": 30 * (1 - p_f),
            "naive_window": 0.3 * p_f,
            "surrogate": "IP >= 30; buy_low HIT iff ERA <= 4.00 (s6.5)",
        },
        "pool": {
            "n_parquet_pitchers": meta["n_parquet_pitchers"],
            "n_ip_band_parquet": meta["n_ip_band_parquet"],
            "n_ranked_pool": meta["n_ranked_pool"],
            "n_rank_diff_pos": meta["n_rank_diff_pos"],
            "n_rank_diff_neg": meta["n_rank_diff_neg"],
            "n_rank_diff_zero": meta["n_rank_diff_zero"],
        },
        "picks": {
            "product": "contrarian_board_2026_midseason_reliever",
            "pick_id_scheme": "contrarian2026-<board>-<side_rank>-<player_id>",
            "n_emitted": n_emitted,
            "n_skipped_existing": n_skipped,
            "evidence_class": "prospective",
        },
    }
    record_path = FROZEN_DIR / f"reliever_board_freeze_record_{FREEZE_DATE}.json"
    if record_path.exists():
        # Write-once: the freeze record documents the ORIGINAL freeze event
        # (including how many picks were emitted then); re-runs must not
        # rewrite it with post-hoc counts.
        logger.warning("Freeze record already exists; NOT rewriting %s", record_path)
    else:
        record_path.write_text(json.dumps(freeze_record, indent=2), encoding="utf-8")
        logger.info("Wrote freeze record %s", record_path)

    write_summary(board, meta, DATED_DIR / "reliever_summary.md")
    update_latest_pointer()

    logger.info(
        "Done. Buy-Low=%d Over-Valued=%d (ranked pool=%d of %d in band; "
        "p_f=%.4f, IP band [%d, %.4f))",
        freeze_record["board_csv"]["n_buy_low"],
        freeze_record["board_csv"]["n_over_valued"],
        meta["n_ranked_pool"], meta["n_ip_band_parquet"],
        p_f, meta["ip_lower"], meta["ip_upper_full_precision"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(generate())
