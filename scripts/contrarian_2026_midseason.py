#!/usr/bin/env python
"""
Generate the 2026 mid-season Contrarian Leaderboards (Buy-Low / Over-Valued).

What this is
------------
An **application** of the 2023-2025 CausalWAR-vs-bWAR contrarian
methodology to *partial-season* 2026 data.  It joins the retrained CausalWAR
2026 leaderboard (from ``leaderboard_cache``) against 2026 season-to-date
Baseball-Reference WAR (backfilled by ``scripts/backfill_2026_war.py``),
ranks players by the two-directional rank disagreement, tags each row with the
dashboard mechanism-tag heuristic, and writes a board + summary.

What this is NOT
---------------
This is NOT a validated hit-rate claim.  The headline Buy-Low figure
(68.4%, 13/19) is NOT a validated edge (2026-08-10 audit §2): its hit
criterion was defined post-hoc, the 95% CI [0.474, 0.843] includes chance,
intention-to-treat scoring (exits count against the pick) is 13/25 = 52%,
and matched-naive mean-reversion baselines score 66.5-73% on the same
pools.  It was measured on FULL-SEASON 2023-24 picks resolved against 2025
outcomes.  Mid-season 2026 picks are UNRESOLVED -- they can only be scored
after a future season plays out.  The generated ``summary.md`` states this
explicitly.

Cohort gates (mid-season, pro-rated)
-----------------------------------
The validated full-season inclusion gates are PA>=300 (batters) / IP>=50
(pitchers).  Through Aug 3 2026 teams have played ~110 of 162 games
(~68% of the season), so those gates are pro-rated by 0.68:

    * batters: PA >= 204   (round(0.68 * 300))
    * pitchers: IP >= 34    (round(0.68 * 50))

Note: the CausalWAR 2026 leaderboard is a *batter* leaderboard (entity=
"batter"), so in practice only the batter gate binds; the pitcher gate is
carried for completeness / future pitcher boards.

Data honesty
------------
Rows whose 2026 bWAR is NULL are DROPPED (never imputed).  If 2026 WAR has
not been backfilled yet, the script refuses to run (use ``--check`` to test
the precondition).  NULL stays NULL.

Reads the DB strictly ``read_only=True``.  Writes only result files under
``results/edges/contrarian_2026_midseason/``.

Usage
-----
    # Precondition gate (exit 0 = ready, non-zero = not ready)::
    python scripts/contrarian_2026_midseason.py --check

    # Generate the board + summary (after 2026 WAR is backfilled)::
    python scripts/contrarian_2026_midseason.py

FROZEN (2026-08-10): generation is permanently refused — the board is a
frozen resolution basis (see the guard above ``generate()`` and the spec's
section 8 Deviations Log). ``--check`` still works read-only.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db.schema import get_connection  # noqa: E402

# Reuse the EXACT validated mechanism-tag heuristic the dashboard ships.
# (An identical copy lives in scripts/causal_war_contrarian_stability.py::
#  classify_row; the canonical home is the view below.)
from src.dashboard.views.contrarian_leaderboards import (  # noqa: E402
    _classify_row as classify_row,
    _TAG_RELIEVER,
    _TAG_PARK,
    _TAG_DEFENSE,
    _TAG_GENUINE,
    _TAG_OTHER,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("contrarian_2026_midseason")

SEASON = 2026

# Season progress through Aug 3 2026 (~110 of 162 games).  Used to pro-rate the
# validated full-season cohort gates.
SEASON_PROGRESS = 0.68
FULL_PA_GATE = 300  # validated full-season batter inclusion gate
FULL_IP_GATE = 50   # validated full-season pitcher inclusion gate
PA_MIN = round(FULL_PA_GATE * SEASON_PROGRESS)  # 204
IP_MIN = round(FULL_IP_GATE * SEASON_PROGRESS)  # 34

TOP_N = 25

OUT_DIR = ROOT / "results" / "edges" / "contrarian_2026_midseason"
# WS0.4 (2026-08-10): the script writes DATED copies under
# OUT_DIR/YYYY-MM-DD/{board.csv,summary.md} plus an atomically-replaced
# ``latest.json`` pointer. The original top-level board.csv / summary.md
# (first generation, 2026-08) are frozen legacy artifacts — never
# overwritten in place again; their content is preserved as the 2026-08-10
# dated copy.
LEGACY_BOARD_CSV = OUT_DIR / "board.csv"
LEGACY_SUMMARY_MD = OUT_DIR / "summary.md"
LATEST_POINTER = OUT_DIR / "latest.json"


def _dated_dir(run_date: str) -> Path:
    return OUT_DIR / run_date


def _write_latest_pointer(run_date: str) -> None:
    """Atomically update ``latest.json`` to name the newest dated dir.

    Written via temp file + ``os.replace`` (atomic on the same volume) so a
    concurrent reader never sees a partial pointer.
    """
    payload = {
        "latest": run_date,
        "board_csv": f"{run_date}/board.csv",
        "summary_md": f"{run_date}/summary.md",
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp = LATEST_POINTER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, LATEST_POINTER)

_PHI = "PHI"

# Detached language per WS0.3 (2026-08-10 audit §2): no "validated" attached
# to 68.4%. Wording mirrors the dashboard's _BANNER_2026
# (src/dashboard/views/contrarian_leaderboards.py).
_BANNER = (
    "APPLICATION of the 2023-2025 CausalWAR contrarian methodology to "
    "PARTIAL-SEASON 2026 data. The headline Buy-Low hit rate (68.4%, 13/19) "
    "is NOT a validated edge: its hit criterion was defined post-hoc, the "
    "95% CI [0.474, 0.843] includes chance, intention-to-treat scoring "
    "(exits count against the pick) is 13/25 = 52%, and matched-naive "
    "mean-reversion baselines score 66.5-73% on the same pools. It "
    "was measured on FULL-SEASON 2023-24 picks resolved "
    "against 2025 outcomes and does NOT transfer to these mid-season boards. "
    "2026 picks here are UNRESOLVED and can only be scored after future seasons."
)


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

def check_preconditions(conn) -> tuple[bool, dict[str, Any]]:
    """Verify 2026 WAR is backfilled and the CausalWAR leaderboard exists."""
    n_bat = conn.execute(
        "SELECT COUNT(*) FROM season_batting_stats WHERE season = ? AND war IS NOT NULL",
        [SEASON],
    ).fetchone()[0]
    n_pit = conn.execute(
        "SELECT COUNT(*) FROM season_pitching_stats WHERE season = ? AND war IS NOT NULL",
        [SEASON],
    ).fetchone()[0]
    lb_row = conn.execute(
        "SELECT row_count FROM leaderboard_cache WHERE model_name = 'causal_war' AND season = ?",
        [SEASON],
    ).fetchone()
    lb_rows = int(lb_row[0]) if lb_row else 0

    ready = (n_bat > 0) and (lb_rows > 0)
    info = {
        "batting_war_nonnull": int(n_bat),
        "pitching_war_nonnull": int(n_pit),
        "causal_war_leaderboard_rows": lb_rows,
        "ready": ready,
    }
    return ready, info


# ---------------------------------------------------------------------------
# Data loading (all read-only)
# ---------------------------------------------------------------------------

def load_causal_leaderboard(conn) -> pd.DataFrame:
    row = conn.execute(
        "SELECT leaderboard_parquet FROM leaderboard_cache "
        "WHERE model_name = 'causal_war' AND season = ?",
        [SEASON],
    ).fetchone()
    if row is None:
        raise RuntimeError(
            f"No CausalWAR leaderboard cached for season {SEASON}. "
            "Run the CausalWAR precompute first."
        )
    df = pd.read_parquet(io.BytesIO(row[0]))
    df["player_id"] = df["player_id"].astype("int64")
    return df


def load_bwar(conn, ids: list[int]) -> tuple[dict[int, dict], dict[int, dict]]:
    """Return (batting, pitching) 2026 stat dicts keyed by player_id."""
    ph = ",".join(["?"] * len(ids))
    bat = {}
    for pid, war, pa, ops in conn.execute(
        f"SELECT player_id, war, pa, ops FROM season_batting_stats "
        f"WHERE season = {SEASON} AND player_id IN ({ph})", ids
    ).fetchall():
        bat[int(pid)] = {
            "war": float(war) if war is not None else None,
            "pa": int(pa) if pa is not None else None,
            "ops": float(ops) if ops is not None else None,
        }
    pit = {}
    for pid, war, ip, era in conn.execute(
        f"SELECT player_id, war, ip, era FROM season_pitching_stats "
        f"WHERE season = {SEASON} AND player_id IN ({ph})", ids
    ).fetchall():
        pit[int(pid)] = {
            "war": float(war) if war is not None else None,
            "ip": float(ip) if ip is not None else None,
            "era": float(era) if era is not None else None,
        }
    return bat, pit


def load_names(conn, ids: list[int]) -> dict[int, str]:
    ph = ",".join(["?"] * len(ids))
    return {
        int(pid): str(name)
        for pid, name in conn.execute(
            f"SELECT player_id, full_name FROM players WHERE player_id IN ({ph})", ids
        ).fetchall()
        if name
    }


def load_position_lookup(conn) -> dict[int, str]:
    """player_id -> primary fielding position (sparse; used for DEFENSE tag)."""
    try:
        return {
            int(pid): str(pos)
            for pid, pos in conn.execute(
                "SELECT player_id, position FROM players WHERE position IS NOT NULL"
            ).fetchall()
            if pos
        }
    except Exception:
        return {}


def derive_teams(conn, ids: list[int]) -> dict[int, str]:
    """Modal 2026 team per batter, derived from pitch-level home/away context.

    A batter bats for the home team in the bottom half and the away team in the
    top half of an inning.  Real data -- no fabrication.
    """
    ph = ",".join(["?"] * len(ids))
    q = f"""
        WITH b AS (
            SELECT batter_id AS player_id,
                   CASE WHEN inning_topbot = 'Bot' THEN home_team ELSE away_team END AS team,
                   COUNT(*) AS n
            FROM pitches
            WHERE batter_id IN ({ph}) AND game_date >= DATE '{SEASON}-01-01'
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
# Board construction
# ---------------------------------------------------------------------------

def build_board(conn) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return (board_df, meta).  board_df has a ``board`` column per side."""
    lb = load_causal_leaderboard(conn)
    ids = lb["player_id"].astype(int).tolist()

    bat, pit = load_bwar(conn, ids)
    names = load_names(conn, ids)
    position_lookup = load_position_lookup(conn)
    teams = derive_teams(conn, ids)

    rows = []
    for r in lb.itertuples(index=False):
        pid = int(r.player_id)
        b = bat.get(pid, {})
        p = pit.get(pid, {})
        # CausalWAR 2026 is a batter leaderboard: offensive causal value.
        rows.append({
            "player_id": pid,
            "name": names.get(pid, str(getattr(r, "name", pid))),
            "team": teams.get(pid, None),
            "position": "batter",
            "field_pos": position_lookup.get(pid, None),
            "causal_war": float(r.causal_war),
            "trad_war": b.get("war"),
            "pa_total": b.get("pa") if b.get("pa") is not None else int(getattr(r, "pa", 0) or 0),
            "ip_total": p.get("ip"),
            "ops": b.get("ops"),
            "era": p.get("era"),
        })
    merged = pd.DataFrame(rows)

    n_pool_raw = len(merged)
    # Real-data discipline: cannot rank a player whose 2026 bWAR is NULL.
    merged = merged.dropna(subset=["trad_war"]).copy()
    n_with_bwar = len(merged)

    # Mid-season cohort gate (pro-rated).
    pa_tot = merged["pa_total"].fillna(0).astype(float)
    ip_tot = merged["ip_total"].fillna(0).astype(float)
    qualifies = (
        ((merged["position"] == "batter") & (pa_tot >= PA_MIN))
        | ((merged["position"] == "pitcher") & (ip_tot >= IP_MIN))
    )
    merged = merged[qualifies].copy()
    n_qualified = len(merged)

    meta: dict[str, Any] = {
        "season": SEASON,
        "season_progress": SEASON_PROGRESS,
        "pa_min": PA_MIN,
        "ip_min": IP_MIN,
        "top_n": TOP_N,
        "n_leaderboard": n_pool_raw,
        "n_with_bwar": n_with_bwar,
        "n_qualified_pool": n_qualified,
        "n_phillies_in_pool": int((merged["team"] == _PHI).sum()),
    }

    if merged.empty:
        return merged, meta

    # Rank within position (matches the validated stability convention).
    merged["rank_causal"] = (
        merged.groupby("position")["causal_war"].rank(ascending=False, method="min").astype(int)
    )
    merged["rank_trad"] = (
        merged.groupby("position")["trad_war"].rank(ascending=False, method="min").astype(int)
    )
    # rank_diff = rank_trad - rank_causal.  Positive -> CausalWAR ranks player
    # higher (better) than bWAR => Buy-Low.  Negative => Over-Valued.
    merged["rank_diff"] = merged["rank_trad"] - merged["rank_causal"]

    merged["tag"] = merged.apply(lambda row: classify_row(row, position_lookup), axis=1)

    buy_low = merged.sort_values("rank_diff", ascending=False).head(TOP_N).copy()
    buy_low["board"] = "buy_low"
    buy_low["side_rank"] = np.arange(1, len(buy_low) + 1)

    over_valued = merged.sort_values("rank_diff", ascending=True).head(TOP_N).copy()
    over_valued["board"] = "over_valued"
    over_valued["side_rank"] = np.arange(1, len(over_valued) + 1)

    board = pd.concat([buy_low, over_valued], ignore_index=True)

    cols = [
        "board", "side_rank", "player_id", "name", "team", "position", "field_pos",
        "causal_war", "trad_war", "rank_causal", "rank_trad", "rank_diff",
        "pa_total", "ip_total", "tag",
    ]
    board = board[cols]
    return board, meta


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def _tag_counts_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(none)_"
    vc = df["tag"].value_counts()
    return "\n".join(f"- `{tag}`: {int(n)}" for tag, n in vc.items())


def _top_table_md(df: pd.DataFrame, n: int = 10) -> str:
    if df.empty:
        return "_(none)_"
    lines = ["| # | Player | Team | CausalWAR | bWAR | RankDiff | PA | Tag |",
             "|---|---|---|---|---|---|---|---|"]
    for r in df.head(n).itertuples(index=False):
        star = " (PHI)" if r.team == _PHI else ""
        pa = int(r.pa_total) if pd.notna(r.pa_total) else "-"
        lines.append(
            f"| {r.side_rank} | {r.name}{star} | {r.team or '-'} | "
            f"{r.causal_war:.2f} | {r.trad_war:.2f} | {r.rank_diff:+d} | {pa} | {r.tag} |"
        )
    return "\n".join(lines)


def write_summary(
    board: pd.DataFrame, meta: dict[str, Any], out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buy_low = board[board["board"] == "buy_low"]
    over_valued = board[board["board"] == "over_valued"]

    phi_bl = buy_low[buy_low["team"] == _PHI]["name"].tolist()
    phi_ov = over_valued[over_valued["team"] == _PHI]["name"].tolist()

    lines: list[str] = []
    lines.append("# Contrarian Leaderboards -- 2026 Mid-Season (live)")
    lines.append("")
    lines.append(f"> **{_BANNER}**")
    lines.append("")
    lines.append(f"_Generated from CausalWAR-2026 (leaderboard_cache) vs 2026 "
                 f"season-to-date Baseball-Reference WAR. Season snapshot through Aug 3, 2026._")
    lines.append("")
    lines.append("## Cohort gates (mid-season, pro-rated)")
    lines.append("")
    lines.append(f"- Season progress applied: **{meta['season_progress']:.0%}** of a 162-game season (~110 games through Aug 3).")
    lines.append(f"- Batters: **PA >= {meta['pa_min']}** (round({meta['season_progress']} x {FULL_PA_GATE})).")
    lines.append(f"- Pitchers: **IP >= {meta['ip_min']}** (round({meta['season_progress']} x {FULL_IP_GATE})) "
                 "-- carried for completeness; the 2026 CausalWAR leaderboard is batter-only.")
    lines.append("")
    lines.append("## Pool")
    lines.append("")
    lines.append(f"- CausalWAR-2026 leaderboard rows: {meta['n_leaderboard']}")
    lines.append(f"- With non-null 2026 bWAR (rankable): {meta['n_with_bwar']}")
    lines.append(f"- Passing the mid-season cohort gate: {meta['n_qualified_pool']}")
    lines.append(f"- Phillies in qualified pool: {meta['n_phillies_in_pool']}")
    lines.append("")
    lines.append("## Buy-Low (CausalWAR > bWAR)")
    lines.append("")
    lines.append("Players my model ranks higher than the public WAR market at the 2026 "
                 "mid-season snapshot (positive rank_diff). Sorted by rank_diff descending.")
    lines.append("")
    lines.append(_top_table_md(buy_low, n=10))
    lines.append("")
    lines.append("**Buy-Low mechanism tags:**")
    lines.append("")
    lines.append(_tag_counts_md(buy_low))
    lines.append("")
    if phi_bl:
        lines.append(f"**Phillies on the Buy-Low board:** {', '.join(phi_bl)}")
        lines.append("")
    lines.append("## Over-Valued (bWAR > CausalWAR)")
    lines.append("")
    lines.append("Players bWAR ranks higher than CausalWAR -- value likely leaning on "
                 "glove / park / sequencing the per-PA model cannot see (negative rank_diff). "
                 "Sorted by rank_diff ascending.")
    lines.append("")
    lines.append(_top_table_md(over_valued, n=10))
    lines.append("")
    lines.append("**Over-Valued mechanism tags:**")
    lines.append("")
    lines.append(_tag_counts_md(over_valued))
    lines.append("")
    if phi_ov:
        lines.append(f"**Phillies on the Over-Valued board:** {', '.join(phi_ov)}")
        lines.append("")
    lines.append("## Methodology + honest caveats")
    lines.append("")
    lines.append("- **Tags** reuse the dashboard mechanism-tag heuristic "
                 "(`src/dashboard/views/contrarian_leaderboards.py::_classify_row`): "
                 f"`{_TAG_RELIEVER}`, `{_TAG_PARK}`, `{_TAG_DEFENSE}`, `{_TAG_GENUINE}`, `{_TAG_OTHER}`. "
                 "Internal tag thresholds (e.g. GENUINE EDGE needs PA>=400) are the "
                 "FULL-SEASON values from the 2023-24 evidence run applied as-is, "
                 "so fewer rows earn GENUINE at mid-season.")
    lines.append("- The 2026 CausalWAR leaderboard is **batter-only**, so RELIEVER LEVERAGE GAP "
                 "and PARK FACTOR (pitcher tags) do not appear on this board.")
    lines.append("- **DEFENSE GAP** depends on `players.position`, which is currently NULL for the "
                 "2026 leaderboard players, so that tag rarely fires; it will activate automatically "
                 "once fielding positions are populated.")
    lines.append("- Rows with NULL 2026 bWAR are dropped, never imputed. NULL stays NULL.")
    lines.append("- **These picks are unresolved.** The headline 68.4% Buy-Low hit rate is a "
                 "full-season 2023-24 -> 2025 result and is NOT a validated edge (post-hoc hit "
                 "criterion; 95% CI [0.474, 0.843] includes chance; intention-to-treat is "
                 "13/25 = 52%; matched-naive baselines score 66.5-73% on the same pools). "
                 "It does not transfer here; resolution requires a future season.")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# WS4 carry-over (2026-08-10): refuse-regeneration guard.
#
# The 2026 mid-season resolution basis is FROZEN: the 50 batter picks are
# fixed by docs/models/contrarian_2026_resolution_spec.md section 3.1 (board.csv
# sha256 EF89356E..., commit 0928baf) and registered in the append-only pick
# ledger (predictions/picks.jsonl). Regenerating this board post-freeze would
# mint lookalike board.csv/summary.md artifacts from a drifted data snapshot
# that CANNOT join the frozen resolution basis, inviting exactly the
# artifact-identity confusion the audit calls failure class F-A. The guard is
# unconditional (no override flag): a new season's board is a new script/spec.
# Recorded as a dated entry in the spec's section 8 Deviations Log.
# ---------------------------------------------------------------------------
_FREEZE_GUARD_MSG = (
    "REFUSED: the 2026 mid-season contrarian board is a FROZEN resolution "
    "basis (docs/models/contrarian_2026_resolution_spec.md section 3.1, spec sha256 "
    "1a27cd0e..., commit 912ede6; picks registered in predictions/picks.jsonl). "
    "Regeneration would create artifacts that cannot join the frozen basis. "
    "Read the frozen board from results/edges/contrarian_2026_midseason/ "
    "(dated dirs + latest.json). Resolution happens once, per spec section 7."
)


def generate() -> int:
    logger.error(_FREEZE_GUARD_MSG)
    return 2


def _generate_prefreeze_disabled() -> int:  # pragma: no cover - frozen path
    """Original generator, disabled 2026-08-10 by the freeze guard above."""
    conn = get_connection(read_only=True)
    try:
        ready, info = check_preconditions(conn)
        logger.info("Preconditions: %s", info)
        if not ready:
            logger.error(
                "Preconditions NOT met (2026 bWAR non-null=%d, leaderboard rows=%d). "
                "Run scripts/backfill_2026_war.py first.",
                info["batting_war_nonnull"], info["causal_war_leaderboard_rows"],
            )
            return 1

        board, meta = build_board(conn)
        if board.empty:
            logger.error("Board is empty after cohort gating (meta=%s). Nothing written.", meta)
            return 1

        # WS0.4: dated output dir + atomically-replaced latest pointer.
        # The legacy top-level board.csv / summary.md are never rewritten.
        run_date = date.today().isoformat()
        dated = _dated_dir(run_date)
        dated.mkdir(parents=True, exist_ok=True)
        board_csv = dated / "board.csv"
        board.to_csv(board_csv, index=False)
        logger.info("Wrote %s (%d rows)", board_csv, len(board))
        write_summary(board, meta, dated / "summary.md")
        _write_latest_pointer(run_date)
        logger.info("Updated latest pointer -> %s", LATEST_POINTER)
        logger.info(
            "Done. Buy-Low=%d Over-Valued=%d (pool=%d, thresholds PA>=%d/IP>=%d)",
            int((board["board"] == "buy_low").sum()),
            int((board["board"] == "over_valued").sum()),
            meta["n_qualified_pool"], PA_MIN, IP_MIN,
        )
        return 0
    finally:
        conn.close()


def check() -> int:
    conn = get_connection(read_only=True)
    try:
        ready, info = check_preconditions(conn)
        status = "READY" if ready else "NOT READY"
        logger.info(
            "[%s] 2026 batting WAR non-null=%d, pitching WAR non-null=%d, "
            "CausalWAR leaderboard rows=%d",
            status, info["batting_war_nonnull"], info["pitching_war_nonnull"],
            info["causal_war_leaderboard_rows"],
        )
        if not ready:
            logger.info("Run scripts/backfill_2026_war.py to backfill 2026 WAR, then re-run.")
        return 0 if ready else 1
    finally:
        conn.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--check", action="store_true",
        help="Verify preconditions (2026 WAR backfilled + leaderboard present) and exit. "
             "Exit 0 = ready, non-zero = not ready.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.check:
        return check()
    return generate()


if __name__ == "__main__":
    sys.exit(main())
