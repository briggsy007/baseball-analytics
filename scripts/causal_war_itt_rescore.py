#!/usr/bin/env python
"""WS4.6 (2026-08-10): intention-to-treat rescoring of EVERY historical
contrarian board.

The audit (section 2.5) found the marquee 68.4% excluded 6 of 25 Buy-Low
picks (survivorship-trimmed denominator, exclusions asymmetric in the
model's favor).  This script rescores every existing board artifact under
the frozen resolution spec's ITT convention (spec section 5.3, the
direction-symmetric statement):

* **Buy-Low pick with no resolvable follow-up record (exit)** -> MISS.
  A bullish pick asserts delivery; non-delivery is the claimed outcome
  failing.  ITT denominator = all 25 picks.
* **Over-Valued exit** -> VOID (excluded, disclosed, never a HIT).
  Crediting the bearish side for injuries/roster exits is exactly the
  inflation the audit flagged.  ITT denominator = 25 - voids; the
  worst-case rate (hits/25) is published alongside.
* Per-protocol (hits / evaluated) is reported next to ITT everywhere --
  continuity with the historical artifacts, never as the headline.

Boards rescored (all existing evidence artifacts):
  1. Single-season windows 2022->23, 2023->24, 2024->25 from
     ``results/causal_war/contrarian_stability/{buy_low,over_valued}_*.csv``
     (frozen 2015-2022 nuisance; the 2022 baseline is lightly in-sample
     for the nuisance, carried as the artifact's own caveat).
  2. The 2-yr-aggregate config (the 68.4% marquee) from
     ``results/causal_war/contrarian_stability/hit_rates_reproduction.json``
     (row-level reproduction of artifact A7, v1 pin).

In the stability CSVs an unresolved row is ``hit = NaN`` with basis
``no_2026_record`` (no follow-up record at all) or
``insufficient_followup_*`` (record below the surrogate floors PA>=100 /
IP>=30).  Both are "no resolvable performance record" under section 5.3 -> exit.

Read-only everywhere; writes ``results/causal_war/itt_rescoring_2026-08-10/``.

Usage:  python scripts/causal_war_itt_rescore.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("itt_rescore")

STAB = ROOT / "results" / "causal_war" / "contrarian_stability"
OUTDIR = ROOT / "results" / "causal_war" / "itt_rescoring_2026-08-10"
WINDOWS = [(2022, 2023), (2023, 2024), (2024, 2025)]

# 1000-resample seed-42 percentile bootstrap -- verbatim convention from
# causal_war_contrarian_stability.py::bootstrap_ci / frozen spec section 5.7.
N_BOOT, SEED = 1000, 42


def bootstrap_ci(scores: np.ndarray) -> tuple[float, float]:
    rng = np.random.RandomState(SEED)
    if len(scores) == 0:
        return float("nan"), float("nan")
    n = len(scores)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        boots[i] = float(np.mean(scores[rng.randint(0, n, size=n)]))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _rate(h: int, n: int) -> float | None:
    return round(h / n, 4) if n > 0 else None


def score_side(hits: list[bool | None], side: str, n_leaderboard: int) -> dict[str, Any]:
    """ITT / worst-case / per-protocol accounting for one board side."""
    n_hit = sum(1 for h in hits if h is True)
    n_miss = sum(1 for h in hits if h is False)
    n_unresolved = sum(1 for h in hits if h is None)
    assert n_hit + n_miss + n_unresolved == n_leaderboard

    if side == "buy_low":
        itt_n = n_leaderboard              # exits are misses
        itt_scores = np.array(
            [1 if h is True else 0 for h in hits], dtype=int
        )
        voids = 0
    else:  # over_valued: exits are voids, excluded and disclosed
        itt_n = n_leaderboard - n_unresolved
        itt_scores = np.array(
            [1 if h is True else 0 for h in hits if h is not None], dtype=int
        )
        voids = n_unresolved

    lo, hi = bootstrap_ci(itt_scores)
    pp_n = n_hit + n_miss
    return {
        "side": side,
        "n_leaderboard": n_leaderboard,
        "hits": n_hit,
        "resolved_misses": n_miss,
        "exits_unresolved": n_unresolved,
        "itt": {
            "rate": _rate(n_hit, itt_n),
            "numerator": n_hit,
            "denominator": itt_n,
            "voids_excluded": voids,
            "ci95_bootstrap": [round(lo, 4), round(hi, 4)],
            "convention": ("exit=MISS (denominator 25)" if side == "buy_low"
                           else "exit=VOID (excluded, disclosed)"),
        },
        "worst_case_rate": _rate(n_hit, n_leaderboard),
        "per_protocol": {
            "rate": _rate(n_hit, pp_n),
            "numerator": n_hit,
            "denominator": pp_n,
        },
    }


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "convention": "frozen resolution spec section 5.3 (2026-08-10, commit "
                      "912ede6): Buy-Low exit = MISS; Over-Valued exit = VOID "
                      "(never HIT, never MISS); per-protocol reported "
                      "alongside, never as headline",
        "windows": [],
    }

    # ---- 1. Single-season stability windows ------------------------------
    for b, f in WINDOWS:
        win: dict[str, Any] = {
            "config": "single_season",
            "baseline_year": b, "followup_year": f,
            "nuisance": "frozen 2015-2022 checkpoint"
                        + (" (2022 baseline lightly in-sample)" if b == 2022 else " (fully OOS)"),
        }
        for side, fname in (("buy_low", f"buy_low_{b}_to_{f}.csv"),
                            ("over_valued", f"over_valued_{b}_to_{f}.csv")):
            df = pd.read_csv(STAB / fname)
            hits: list[bool | None] = [
                None if pd.isna(h) else bool(h) for h in df["hit"]
            ]
            s = score_side(hits, side, len(df))
            s["exit_basis_breakdown"] = (
                df.loc[df["hit"].isna(), "basis"].value_counts().to_dict()
            )
            s["exit_names"] = df.loc[df["hit"].isna(), "name"].tolist()
            win[side] = s
        out["windows"].append(win)

    # ---- 2. The 2-yr-aggregate config (68.4% marquee, artifact A7/A9) ----
    repro = json.loads((STAB / "hit_rates_reproduction.json").read_text(encoding="utf-8"))
    win2: dict[str, Any] = {
        "config": "two_year_aggregate_2023_24_to_2025",
        "baseline_years": [2023, 2024], "followup_year": 2025,
        "source": "hit_rates_reproduction.json (row-level reproduction of "
                  "causal_war_baseline_comparison_2023_2024.csv, v1 pin)",
        "nuisance": "frozen 2015-2022 checkpoint (fully OOS for 2023-24 baselines)",
    }
    for side in ("buy_low", "over_valued"):
        rows = repro[side]["rows"]
        hits = [r["hit"] for r in rows]  # true/false/None
        s = score_side(
            [bool(h) if h is not None else None for h in hits], side, len(rows)
        )
        s["exit_names"] = [r["player"] for r in rows if r["hit"] is None]
        s["marquee_per_protocol_check"] = {
            "expected": {"buy_low": "13/19 = 68.4%",
                         "over_valued": "14/23 = 60.9%"}[side],
            "recomputed": f"{s['per_protocol']['numerator']}/"
                          f"{s['per_protocol']['denominator']}",
        }
        win2[side] = s
    out["windows"].append(win2)

    (OUTDIR / "itt_summary.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8",
    )
    logger.info("Wrote %s", OUTDIR / "itt_summary.json")

    # ---- Console table, exactly as it lands ------------------------------
    print("\nITT rescoring -- every historical board (spec 5.3 convention)")
    print("=" * 96)
    print(f"{'window':38s} {'side':12s} {'ITT':>16s} {'worst-case':>11s} "
          f"{'per-protocol':>13s} {'exits':>6s}")
    print("-" * 96)
    for w in out["windows"]:
        label = (f"{w.get('baseline_year', w.get('baseline_years'))}"
                 f"->{w['followup_year']} [{w['config']}]")
        for side in ("buy_low", "over_valued"):
            s = w[side]
            itt = s["itt"]
            print(
                f"{label:38s} {side:12s} "
                f"{itt['numerator']:>3d}/{itt['denominator']:<3d}"
                f"={itt['rate'] * 100 if itt['rate'] is not None else float('nan'):5.1f}% "
                f"{s['worst_case_rate'] * 100:9.1f}% "
                f"{s['per_protocol']['numerator']:>4d}/{s['per_protocol']['denominator']:<3d}"
                f"={s['per_protocol']['rate'] * 100:5.1f}% "
                f"{s['exits_unresolved']:>5d}"
            )
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
