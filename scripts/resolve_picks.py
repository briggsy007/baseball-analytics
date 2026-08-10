#!/usr/bin/env python
"""Morning resolver for the append-only pick ledger (plan WS1.2).

For every pick in ``predictions/picks.jsonl`` with no matching line in
``predictions/resolutions.jsonl`` whose resolvable date has passed, fetch the
outcome and append a resolution. Numbers are reported exactly as they land —
misses with the same prominence as hits (kill criterion K4).

Per product:

* ``hit_parlay`` legs — resolved from the DuckDB ``pitches`` table
  (READ-ONLY; the single-writer rule is never touched) per the frozen leg
  rule wording in ``src/pick_ledger.py``. Resolvable once the DB's ingested
  watermark covers the pick date.
* ``hit_parlay`` parlay picks — resolved from their legs' recorded
  resolutions (never from the DB directly).
* ``contrarian_board_2026_midseason`` / ``..._reliever`` — NOT resolvable before its
  ``resolve_by`` (last 2026 regular-season game + 7 days, spec §7.1). The
  resolver only reports the external source it will use
  (Baseball-Reference bWAR via ``scripts/backfill_2026_war.py`` per the
  frozen spec) and skips.
* anything else — reported as requiring manual/external resolution.

Usage
-----
    python scripts/resolve_picks.py             # resolve everything due
    python scripts/resolve_picks.py --dry-run   # show what would resolve
    python scripts/resolve_picks.py --as-of 2026-08-10
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.db.schema import get_connection  # noqa: E402
from src.pick_ledger import (  # noqa: E402
    LedgerError,
    compute_track_record,
    load_ledger,
    resolve_pick,
)

# AB-excluded PA-ending events (mirrors hit_parlay_today.py's AB filter).
_NON_AB_EVENTS = (
    "'walk','hit_by_pitch','sac_fly','sac_bunt','catcher_interf',"
    "'intent_walk','sac_fly_double_play','truncated_pa'"
)
_HIT_EVENTS = "'single','double','triple','home_run'"


def _db_watermark(conn) -> date | None:
    row = conn.execute("SELECT MAX(game_date) FROM pitches").fetchone()
    return row[0] if row else None


def resolve_hit_parlay_leg(conn, pick: dict) -> tuple[str, dict]:
    """Return (outcome, score_fields) for one leg per the frozen rule."""
    subject = pick.get("subject") or {}
    pid = subject.get("player_id")
    day = subject.get("date")
    if pid is None or day is None:
        return "void", {
            "note": "subject lacks player_id/date; cannot resolve mechanically",
            "resolution_branch": "void-missing-subject",
        }
    row = conn.execute(
        f"""
        SELECT
          SUM(CASE WHEN events IN ({_HIT_EVENTS}) THEN 1 ELSE 0 END) AS hits,
          SUM(CASE WHEN events IS NOT NULL
                    AND events NOT IN ({_NON_AB_EVENTS}) THEN 1 ELSE 0 END) AS ab
        FROM pitches
        WHERE batter_id = ? AND game_date = ?
        """,
        [pid, day],
    ).fetchone()
    hits = int(row[0] or 0)
    ab = int(row[1] or 0)
    p = (pick.get("p_or_direction") or {}).get("p")
    if hits >= 1:
        outcome = "yes"
    elif ab >= 1:
        outcome = "no"
    else:
        outcome = "void"
    fields: dict = {
        "hits": hits,
        "ab": ab,
        "source": "duckdb pitches (read_only)",
        "resolution_branch": outcome if outcome != "void" else "void-no-ab",
    }
    if p is not None and outcome in ("yes", "no"):
        y = 1.0 if outcome == "yes" else 0.0
        fields["p"] = p
        fields["y"] = y
        fields["brier"] = round((p - y) ** 2, 6)
    return outcome, fields


def resolve_hit_parlay_parlay(pick: dict, resolutions_by_id: dict) -> tuple[str, dict] | None:
    """Resolve a parlay from its legs' resolutions; None when legs pending."""
    leg_ids = (pick.get("subject") or {}).get("legs") or []
    leg_outcomes = {}
    for lid in leg_ids:
        r = resolutions_by_id.get(lid)
        if r is None:
            return None  # legs not all resolved yet
        leg_outcomes[lid] = r["outcome"]
    if any(o == "no" for o in leg_outcomes.values()):
        outcome = "no"
    elif all(o == "yes" for o in leg_outcomes.values()):
        outcome = "yes"
    else:
        outcome = "void"
    p = (pick.get("p_or_direction") or {}).get("p")
    fields: dict = {"legs": leg_outcomes, "source": "leg resolutions (ledger)"}
    if p is not None and outcome in ("yes", "no"):
        y = 1.0 if outcome == "yes" else 0.0
        fields["p"] = p
        fields["y"] = y
        fields["brier"] = round((p - y) ** 2, 6)
    return outcome, fields


def main() -> int:
    ap = argparse.ArgumentParser(description="Resolve due picks in the ledger.")
    ap.add_argument("--as-of", default=date.today().isoformat(),
                    help="Treat this date as 'today' (YYYY-MM-DD).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would resolve; append nothing.")
    args = ap.parse_args()
    as_of = date.fromisoformat(args.as_of)

    ledger = load_ledger()
    picks = ledger["picks"]
    resolutions_by_id = {r["pick_id"]: r for r in ledger["resolutions"]}
    unresolved = [p for p in picks if p["pick_id"] not in resolutions_by_id]

    print(f"resolve_picks | as_of={as_of} | picks={len(picks)} "
          f"resolved={len(resolutions_by_id)} unresolved={len(unresolved)}")

    if not unresolved:
        print("Nothing to do.")
        return 0

    conn = None
    watermark: date | None = None
    appended = 0
    skipped: list[str] = []
    parlays_pending: list[dict] = []

    def _ensure_conn():
        nonlocal conn, watermark
        if conn is None:
            conn = get_connection(read_only=True)
            watermark = _db_watermark(conn)
        return conn

    for pick in unresolved:
        pid = pick["pick_id"]
        product = pick.get("product")

        if product == "hit_parlay":
            subject = pick.get("subject") or {}
            if "legs" in subject:
                parlays_pending.append(pick)  # after legs
                continue
            leg_day = subject.get("date")
            if leg_day and date.fromisoformat(leg_day) >= as_of:
                skipped.append(f"{pid}: game day not complete (as_of {as_of})")
                continue
            c = _ensure_conn()
            if leg_day and watermark and date.fromisoformat(leg_day) > watermark:
                skipped.append(
                    f"{pid}: DB watermark {watermark} has not ingested "
                    f"{leg_day} yet"
                )
                continue
            outcome, fields = resolve_hit_parlay_leg(c, pick)
            if args.dry_run:
                print(f"  DRY-RUN would resolve {pid}: {outcome} {fields}")
            else:
                resolve_pick(pid, outcome, score_fields=fields)
                resolutions_by_id[pid] = {"pick_id": pid, "outcome": outcome}
                appended += 1
                print(f"  resolved {pid}: {outcome.upper()} {fields}")

        elif product in ("contrarian_board_2026_midseason",
                         "contrarian_board_2026_midseason_reliever"):
            rb = pick.get("resolve_by")
            if rb and date.fromisoformat(rb) > as_of:
                skipped.append(
                    f"{pid}: not resolvable before {rb} (frozen spec: last "
                    "2026 regular-season game + 7 days; source = B-Ref bWAR "
                    "via scripts/backfill_2026_war.py, spec section 7)"
                )
            else:
                skipped.append(
                    f"{pid}: past resolve_by — run the spec section 7 "
                    "procedure (backfill_2026_war.py + dated resolution "
                    "parquet), then resolve via the resolution table, not "
                    "this ad-hoc path"
                )

        else:
            skipped.append(
                f"{pid}: product {product!r} has no mechanical resolver; "
                f"external source: {pick.get('resolution_source')}"
            )

    # Parlays after their legs.
    for pick in parlays_pending:
        pid = pick["pick_id"]
        result = resolve_hit_parlay_parlay(pick, resolutions_by_id)
        if result is None:
            skipped.append(f"{pid}: legs not all resolved yet")
            continue
        outcome, fields = result
        if args.dry_run:
            print(f"  DRY-RUN would resolve {pid}: {outcome} {fields}")
        else:
            resolve_pick(pid, outcome, score_fields=fields)
            appended += 1
            print(f"  resolved {pid}: {outcome.upper()} {fields}")

    if conn is not None:
        conn.close()

    for s in skipped:
        print(f"  skipped {s}")

    # Honest summary, exactly as it lands (K4: losses at full prominence).
    if appended and not args.dry_run:
        ledger = load_ledger()
        tr = compute_track_record(ledger["picks"], ledger["resolutions"])
        s = tr["summary"]
        print(
            f"\nTRACK RECORD (all products, ledger-only): "
            f"{s['hits']} hits / {s['misses']} misses / {s['voids']} voids "
            f"of {s['resolved']} resolved; ITT hit rate "
            f"{s['hit_rate_itt']}; mean Brier {s['mean_brier']} "
            f"(n={s['n_brier_scored']}; reference: 0.25 = coin flip at p=0.5)"
        )
        for name, d in tr["by_product"].items():
            print(
                f"  {name}: {d['yes']}W-{d['no']}L-{d['void']}V "
                f"(hit rate {d['hit_rate_itt']}, worst-case "
                f"{d['hit_rate_worst_case']}, Brier {d['mean_brier']})"
            )

    print(f"\nAppended {appended} resolution(s). Skipped {len(skipped)}.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LedgerError as exc:
        print(f"LEDGER ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
