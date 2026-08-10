"""Append-only pick ledger (plan WS1.2, 2026-08-10).

Two JSONL files under ``predictions/`` (tracked in git) form the platform's
prospective track record:

``predictions/picks.jsonl``
    One line per frozen pick. Required fields::

        pick_id, frozen_utc, product, model, artifact_version, git_sha,
        subject, market_or_claim, p_or_direction, resolution_rule,
        resolution_source, resolve_by, rule_hash, evidence_class

``predictions/resolutions.jsonl``
    One line per resolved pick::

        pick_id, resolved_utc, outcome (yes|no|void), score fields

Append-only contract (audit failure classes F-B / F-D):

* a ``pick_id`` is frozen the moment it is appended — :func:`emit_pick`
  refuses to re-emit an existing ``pick_id`` and there is no update/delete
  API of any kind;
* resolution never touches the pick line — it is a separate append via
  :func:`resolve_pick`, refused when the pick is unknown or already resolved;
* the track-record computation (:func:`compute_track_record`) reads ONLY the
  two ledger files — never model artifacts, never dashboards, never docs.

Evidence classes:

* ``"prospective"`` — pick frozen through this module before its outcome
  was knowable (the only class new emissions should use);
* ``"prospective-backfilled_<date>"`` — pick provably frozen (committed /
  file-dated) before its outcome, but registered into the ledger later;
* anything else is displayed with a warning by the track-record view.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT: Path = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR: Path = ROOT / "predictions"
PICKS_PATH: Path = PREDICTIONS_DIR / "picks.jsonl"
RESOLUTIONS_PATH: Path = PREDICTIONS_DIR / "resolutions.jsonl"

VALID_OUTCOMES = ("yes", "no", "void")

REQUIRED_PICK_FIELDS = (
    "pick_id",
    "product",
    "model",
    "subject",
    "market_or_claim",
    "p_or_direction",
    "resolution_rule",
    "resolution_source",
    "resolve_by",
    "rule_hash",
    "evidence_class",
)


class LedgerError(RuntimeError):
    """Raised on any attempt to violate the append-only ledger contract."""


# ---------------------------------------------------------------------------
# Low-level JSONL helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise LedgerError(
                    f"{path} line {lineno} is not valid JSON: {exc}"
                ) from exc
    return entries


def _append_jsonl(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def current_git_sha() -> str | None:
    """HEAD sha for provenance stamps; None when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=15,
        )
        sha = (out.stdout or "").strip()
        return sha if out.returncode == 0 and sha else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_ledger(
    picks_path: Path | str | None = None,
    resolutions_path: Path | str | None = None,
) -> dict[str, list[dict]]:
    """Load both ledger files. Returns ``{"picks": [...], "resolutions": [...]}``."""
    picks = _read_jsonl(Path(picks_path) if picks_path else PICKS_PATH)
    resolutions = _read_jsonl(
        Path(resolutions_path) if resolutions_path else RESOLUTIONS_PATH
    )
    return {"picks": picks, "resolutions": resolutions}


def emit_pick(pick: dict, picks_path: Path | str | None = None) -> dict:
    """Append one pick. Refuses to rewrite an existing ``pick_id``.

    ``frozen_utc``, ``git_sha`` and ``artifact_version`` are auto-filled when
    absent (now-UTC / current HEAD / ``"unversioned"``). All other
    :data:`REQUIRED_PICK_FIELDS` must be present and non-empty.
    """
    missing = [
        f for f in REQUIRED_PICK_FIELDS
        if f not in pick or pick[f] in (None, "")
    ]
    if missing:
        raise LedgerError(f"pick is missing required fields: {missing}")

    path = Path(picks_path) if picks_path else PICKS_PATH
    existing_ids = {p.get("pick_id") for p in _read_jsonl(path)}
    if pick["pick_id"] in existing_ids:
        raise LedgerError(
            f"pick_id {pick['pick_id']!r} already exists in {path}; the pick "
            "ledger is append-only — existing picks are never rewritten. "
            "If the pick must change, emit a NEW pick_id and resolve the old "
            "one as void with a documented reason."
        )

    entry = dict(pick)
    entry.setdefault("frozen_utc", _utcnow())
    entry.setdefault("git_sha", current_git_sha())
    entry.setdefault("artifact_version", "unversioned")
    _append_jsonl(path, entry)
    return entry


def resolve_pick(
    pick_id: str,
    outcome: str,
    score_fields: dict[str, Any] | None = None,
    resolutions_path: Path | str | None = None,
    picks_path: Path | str | None = None,
    resolved_utc: str | None = None,
) -> dict:
    """Append one resolution. Never edits the pick line.

    Refuses when ``outcome`` is not yes/no/void, when the pick_id is unknown,
    or when the pick is already resolved (a resolution is final; a wrong
    resolution is corrected by a NEW void+re-emit cycle, documented in the
    score fields — never by rewriting history).
    """
    if outcome not in VALID_OUTCOMES:
        raise LedgerError(
            f"outcome must be one of {VALID_OUTCOMES}, got {outcome!r}"
        )

    p_path = Path(picks_path) if picks_path else PICKS_PATH
    r_path = Path(resolutions_path) if resolutions_path else RESOLUTIONS_PATH

    known_ids = {p.get("pick_id") for p in _read_jsonl(p_path)}
    if pick_id not in known_ids:
        raise LedgerError(
            f"cannot resolve unknown pick_id {pick_id!r} (not in {p_path})"
        )
    already = {r.get("pick_id") for r in _read_jsonl(r_path)}
    if pick_id in already:
        raise LedgerError(
            f"pick_id {pick_id!r} is already resolved; resolutions are "
            "append-only and final."
        )

    entry: dict[str, Any] = {
        "pick_id": pick_id,
        "resolved_utc": resolved_utc or _utcnow(),
        "outcome": outcome,
    }
    if score_fields:
        entry.update(
            {k: v for k, v in score_fields.items() if k not in entry}
        )
    _append_jsonl(r_path, entry)
    return entry


# ---------------------------------------------------------------------------
# Track-record computation (ledger files are the ONLY input)
# ---------------------------------------------------------------------------


def _pick_probability(pick: dict) -> float | None:
    """Extract the frozen probability from ``p_or_direction`` if present."""
    pod = pick.get("p_or_direction")
    if isinstance(pod, dict) and isinstance(pod.get("p"), (int, float)):
        return float(pod["p"])
    if isinstance(pod, (int, float)):
        return float(pod)
    return None


def _pick_direction(pick: dict) -> str | None:
    pod = pick.get("p_or_direction")
    if isinstance(pod, dict):
        d = pod.get("direction")
        return str(d) if d else None
    return None


def compute_track_record(
    picks: Iterable[dict],
    resolutions: Iterable[dict],
    calibration_bucket_edges: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 1.0),
) -> dict:
    """Compute the full track record from the two ledgers and nothing else.

    Returns a dict with:

    * ``summary`` — resolved/unresolved counts, hits, misses, voids, hit
      rates (ITT-style ``yes/(yes+no)`` and worst-case ``yes/(yes+no+void)``),
      mean Brier over probabilistic picks;
    * ``rolling_brier`` — per-resolution cumulative-mean Brier in
      ``resolved_utc`` order (probabilistic non-void picks only);
    * ``calibration`` — buckets of frozen probability vs observed hit rate;
    * ``by_product`` — the same win/loss accounting per product (losses get
      identical treatment to wins — no field exists to hide a miss);
    * ``rows`` — one joined row per pick (resolved or not) for tabular
      display.
    """
    picks = list(picks)
    resolutions = list(resolutions)
    res_by_id = {r["pick_id"]: r for r in resolutions if "pick_id" in r}

    rows: list[dict] = []
    for p in picks:
        r = res_by_id.get(p.get("pick_id"))
        prob = _pick_probability(p)
        row = {
            "pick_id": p.get("pick_id"),
            "product": p.get("product"),
            "model": p.get("model"),
            "subject": p.get("subject"),
            "market_or_claim": p.get("market_or_claim"),
            "p": prob,
            "direction": _pick_direction(p),
            "frozen_utc": p.get("frozen_utc"),
            "resolve_by": p.get("resolve_by"),
            "evidence_class": p.get("evidence_class"),
            "outcome": r.get("outcome") if r else None,
            "resolved_utc": r.get("resolved_utc") if r else None,
            "brier": None,
        }
        if r and prob is not None and r.get("outcome") in ("yes", "no"):
            y = 1.0 if r["outcome"] == "yes" else 0.0
            row["brier"] = round((prob - y) ** 2, 6)
        rows.append(row)

    def _bucket(prob: float) -> str:
        e = calibration_bucket_edges
        for lo, hi in zip(e[:-1], e[1:]):
            if lo <= prob < hi or (hi == e[-1] and prob == hi):
                return f"[{lo:.2f}, {hi:.2f})"
        return "out-of-range"

    resolved = [row for row in rows if row["outcome"] is not None]
    yes = sum(1 for row in resolved if row["outcome"] == "yes")
    no = sum(1 for row in resolved if row["outcome"] == "no")
    void = sum(1 for row in resolved if row["outcome"] == "void")
    briers = [row["brier"] for row in resolved if row["brier"] is not None]

    # Rolling (cumulative-mean) Brier in resolution order.
    scored = sorted(
        (row for row in resolved if row["brier"] is not None),
        key=lambda row: (row["resolved_utc"] or "", row["pick_id"] or ""),
    )
    rolling: list[dict] = []
    running = 0.0
    for i, row in enumerate(scored, 1):
        running += row["brier"]
        rolling.append(
            {
                "n": i,
                "resolved_utc": row["resolved_utc"],
                "pick_id": row["pick_id"],
                "brier": row["brier"],
                "cumulative_mean_brier": round(running / i, 6),
            }
        )

    # Calibration buckets over probabilistic scored picks.
    buckets: dict[str, dict] = {}
    for row in scored:
        b = _bucket(row["p"])
        d = buckets.setdefault(
            b, {"bucket": b, "n": 0, "sum_p": 0.0, "hits": 0}
        )
        d["n"] += 1
        d["sum_p"] += row["p"]
        d["hits"] += 1 if row["outcome"] == "yes" else 0
    calibration = [
        {
            "bucket": b["bucket"],
            "n": b["n"],
            "mean_predicted": round(b["sum_p"] / b["n"], 4),
            "observed_rate": round(b["hits"] / b["n"], 4),
            "gap": round(b["hits"] / b["n"] - b["sum_p"] / b["n"], 4),
        }
        for b in sorted(buckets.values(), key=lambda x: x["bucket"])
    ]

    by_product: dict[str, dict] = {}
    for row in rows:
        d = by_product.setdefault(
            row["product"] or "unknown",
            {"picks": 0, "resolved": 0, "yes": 0, "no": 0, "void": 0,
             "briers": []},
        )
        d["picks"] += 1
        if row["outcome"] is not None:
            d["resolved"] += 1
            d[row["outcome"]] += 1
        if row["brier"] is not None:
            d["briers"].append(row["brier"])
    for name, d in by_product.items():
        denom = d["yes"] + d["no"]
        d["hit_rate_itt"] = round(d["yes"] / denom, 4) if denom else None
        wc = d["yes"] + d["no"] + d["void"]
        d["hit_rate_worst_case"] = round(d["yes"] / wc, 4) if wc else None
        d["mean_brier"] = (
            round(sum(d["briers"]) / len(d["briers"]), 6) if d["briers"] else None
        )
        d.pop("briers")

    denom = yes + no
    summary = {
        "picks_total": len(rows),
        "resolved": len(resolved),
        "unresolved": len(rows) - len(resolved),
        "hits": yes,
        "misses": no,
        "voids": void,
        "hit_rate_itt": round(yes / denom, 4) if denom else None,
        "hit_rate_worst_case": (
            round(yes / (yes + no + void), 4) if (yes + no + void) else None
        ),
        "mean_brier": round(sum(briers) / len(briers), 6) if briers else None,
        "n_brier_scored": len(briers),
    }

    return {
        "summary": summary,
        "rolling_brier": rolling,
        "calibration": calibration,
        "by_product": by_product,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Product rule texts + pick builders (shared by emitter, backfill, resolver)
# ---------------------------------------------------------------------------

#: Frozen wording of the hit-parlay leg resolution rule. The rule_hash of a
#: leg pick is the SHA-256 of exactly this string — changing the wording
#: changes the hash, which makes silent rule drift visible in the ledger.
HIT_PARLAY_LEG_RULE = (
    "YES if the named player records >= 1 official hit (single, double, "
    "triple, or home run) in the named game on the named date; NO if he "
    "completes >= 1 official at-bat in that game without recording a hit; "
    "VOID if he records no official at-bat that day (did not play, or the "
    "game was postponed/cancelled). Walks, HBP, and sacrifices are not "
    "at-bats but do not void the pick when >= 1 AB exists. Primary "
    "resolution source: the local Statcast `pitches` table (events in "
    "{single, double, triple, home_run}; AB = PA-ending events excluding "
    "walk/HBP/sacrifice/catcher-interference), publicly cross-checkable "
    "against the MLB Stats API box score for the game."
)

#: Frozen wording of the hit-parlay combined-parlay resolution rule.
HIT_PARLAY_PARLAY_RULE = (
    "NO if any leg resolves no; otherwise YES if every leg resolves yes; "
    "otherwise VOID (>= 1 leg void and none no — the frozen combined "
    "probability no longer describes the reduced parlay). Legs resolve per "
    "the hit-parlay leg rule."
)

HIT_PARLAY_RESOLUTION_SOURCE = (
    "DuckDB `pitches` table (Statcast ingest, read-only); public "
    "cross-check: MLB Stats API boxscore "
    "(https://statsapi.mlb.com/api/v1.1/game/<gamePk>/feed/live)"
)


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slug(name: str) -> str:
    import re
    import unicodedata

    ascii_name = (
        unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    )
    return re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")


def build_hit_parlay_leg_pick(
    day: str,
    leg_index: int,
    leg: dict,
    evidence_class: str = "prospective",
    frozen_utc: str | None = None,
    git_sha: str | None = None,
    player_id: int | None = None,
) -> dict:
    """Build (not emit) a ledger pick dict for one hit-parlay leg."""
    name = leg["name"]
    pick: dict[str, Any] = {
        "pick_id": f"hitparlay-{day}-leg{leg_index}-{_slug(name)}",
        "product": "hit_parlay",
        "model": "hit_parlay_heuristic",
        "artifact_version": "scripts/hit_parlay_today.py (explicitly non-flagship, unvalidated heuristic)",
        "subject": {
            "name": name,
            "player_id": player_id if player_id is not None else leg.get("player_id"),
            "team": leg.get("team"),
            "game": leg.get("game"),
            "date": day,
            "opp_sp": leg.get("opp_sp"),
            "lineup_slot": leg.get("slot"),
            "lineup_projected": leg.get("projected"),
        },
        "market_or_claim": (
            f"{name} records >= 1 hit in {leg.get('game')} on {day}"
        ),
        "p_or_direction": {"p": leg.get("p_hit_game")},
        "resolution_rule": HIT_PARLAY_LEG_RULE,
        "resolution_source": HIT_PARLAY_RESOLUTION_SOURCE,
        "resolve_by": _next_day(day),
        "rule_hash": _sha256_text(HIT_PARLAY_LEG_RULE),
        "evidence_class": evidence_class,
    }
    if frozen_utc:
        pick["frozen_utc"] = frozen_utc
    if git_sha is not None:
        pick["git_sha"] = git_sha
    return pick


def build_hit_parlay_parlay_pick(
    day: str,
    leg_pick_ids: list[str],
    combined_prob: float,
    evidence_class: str = "prospective",
    frozen_utc: str | None = None,
    git_sha: str | None = None,
) -> dict:
    """Build (not emit) the combined-parlay ledger pick dict."""
    pick: dict[str, Any] = {
        "pick_id": f"hitparlay-{day}-parlay",
        "product": "hit_parlay",
        "model": "hit_parlay_heuristic",
        "artifact_version": "scripts/hit_parlay_today.py (explicitly non-flagship, unvalidated heuristic)",
        "subject": {"date": day, "legs": leg_pick_ids},
        "market_or_claim": (
            f"All {len(leg_pick_ids)} hit-parlay legs record >= 1 hit on {day}"
        ),
        "p_or_direction": {"p": combined_prob},
        "resolution_rule": HIT_PARLAY_PARLAY_RULE,
        "resolution_source": HIT_PARLAY_RESOLUTION_SOURCE,
        "resolve_by": _next_day(day),
        "rule_hash": _sha256_text(HIT_PARLAY_PARLAY_RULE),
        "evidence_class": evidence_class,
    }
    if frozen_utc:
        pick["frozen_utc"] = frozen_utc
    if git_sha is not None:
        pick["git_sha"] = git_sha
    return pick


def _next_day(day: str) -> str:
    from datetime import date as _date, timedelta

    y, m, d = (int(x) for x in day.split("-"))
    return (_date(y, m, d) + timedelta(days=1)).isoformat()
