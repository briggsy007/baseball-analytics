#!/usr/bin/env python
"""Reproduce the 68% Buy-Low hit rate from the dashboard `contrarian_leaderboards`
view EXACTLY, using the same CSV + same hit rule + same follow-up signals.

Companion to `causal_war_contrarian_stability.py`. Writes
`hit_rates_reproduction.json` to
`results/causal_war/contrarian_stability/`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

_CSV_PATH = (
    ROOT / "results" / "validate_causal_war_20260418T194415Z"
    / "causal_war_baseline_comparison_2023_2024.csv"
)

TOP_N = 25
_FOLLOWUP_SEASON = 2025
N_BOOTSTRAP = 1000
RANDOM_STATE = 42


def bootstrap_ci(hits_array: np.ndarray, n_boot: int = N_BOOTSTRAP,
                 random_state: int = RANDOM_STATE, ci: float = 0.95) -> tuple[float, float]:
    if len(hits_array) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(random_state)
    n = len(hits_array)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = float(np.mean(hits_array[rng.randint(0, n, size=n)]))
    alpha = (1.0 - ci) / 2.0
    return (float(np.percentile(boots, 100 * alpha)),
            float(np.percentile(boots, 100 * (1 - alpha))))


def _row_hit(row: pd.Series, info: dict, side: str) -> tuple[bool | None, str]:
    """Apply the dashboard's exact hit rule, symmetrised for both sides."""
    war_followup = info.get("war")
    war_baseline_per_yr = float(row.get("trad_war", 0.0) or 0.0) / 2.0  # 2-yr avg per dashboard

    if war_followup is not None:
        if side == "buy_low":
            return bool(war_followup >= war_baseline_per_yr), "war_delta"
        return bool(war_followup < war_baseline_per_yr), "war_delta"

    if row["position"] == "pitcher":
        ip = info.get("ip")
        era = info.get("era")
        if ip is None or era is None or ip < 30:
            return None, "insufficient_followup"
        if side == "buy_low":
            return bool(era <= 4.00), "era_surrogate"
        return bool(era > 4.00), "era_surrogate"
    else:
        pa = info.get("pa")
        ops = info.get("ops")
        if pa is None or ops is None or pa < 100:
            return None, "insufficient_followup"
        if side == "buy_low":
            return bool(ops >= 0.700), "ops_surrogate"
        return bool(ops < 0.700), "ops_surrogate"


def main() -> int:
    if not _CSV_PATH.exists():
        print(f"Missing CSV: {_CSV_PATH}", file=sys.stderr)
        return 1

    df = pd.read_csv(_CSV_PATH)
    buy_low = df.sort_values("rank_diff", ascending=False).head(TOP_N).copy()
    over_valued = df.sort_values("rank_diff", ascending=True).head(TOP_N).copy()

    conn = duckdb.connect(str(ROOT / "data" / "baseball.duckdb"), read_only=True)
    try:
        ids = list(set(buy_low["player_id"]) | set(over_valued["player_id"]))
        ph = ",".join(["?"] * len(ids))

        bat = conn.execute(
            f"SELECT player_id, war, pa, ops FROM season_batting_stats "
            f"WHERE season={_FOLLOWUP_SEASON} AND player_id IN ({ph})", ids,
        ).fetchall()
        outcomes: dict[int, dict] = {}
        for pid, war, pa, ops in bat:
            outcomes[int(pid)] = {
                "war": float(war) if war is not None else None,
                "pa": int(pa) if pa is not None else None,
                "ops": float(ops) if ops is not None else None,
                "ip": None, "era": None,
            }
        pit = conn.execute(
            f"SELECT player_id, war, ip, era FROM season_pitching_stats "
            f"WHERE season={_FOLLOWUP_SEASON} AND player_id IN ({ph})", ids,
        ).fetchall()
        for pid, war, ip, era in pit:
            rec = outcomes.get(int(pid))
            if rec is None:
                outcomes[int(pid)] = {
                    "war": float(war) if war is not None else None,
                    "pa": None, "ops": None,
                    "ip": float(ip) if ip is not None else None,
                    "era": float(era) if era is not None else None,
                }
            else:
                rec["ip"] = float(ip) if ip is not None else rec.get("ip")
                rec["era"] = float(era) if era is not None else rec.get("era")
                if rec.get("war") is None and war is not None:
                    rec["war"] = float(war)
    finally:
        conn.close()

    def evaluate(df_side: pd.DataFrame, side: str) -> dict:
        hits_list = []
        rows = []
        for _, r in df_side.iterrows():
            info = outcomes.get(int(r["player_id"]))
            if info is None:
                rows.append({"player": r["name"], "hit": None, "basis": "no_record"})
                continue
            hit, basis = _row_hit(r, info, side)
            rows.append({
                "player": r["name"], "position": r["position"],
                "baseline_war": float(r["trad_war"]),
                "followup_war": info.get("war"),
                "hit": hit, "basis": basis,
            })
            if hit is not None:
                hits_list.append(int(bool(hit)))
        n = len(hits_list)
        hits = sum(hits_list)
        rate = hits / n if n > 0 else None
        ci = bootstrap_ci(np.asarray(hits_list, dtype=int))
        return {
            "side": side,
            "hits": hits,
            "n_evaluated": n,
            "hit_rate": rate,
            "ci_95_low": ci[0],
            "ci_95_high": ci[1],
            "rows": rows,
        }

    bl = evaluate(buy_low, "buy_low")
    ov = evaluate(over_valued, "over_valued")

    print(f"Buy-Low (2023-24 -> 2025):    {bl['hits']}/{bl['n_evaluated']} = "
          f"{(bl['hit_rate'] or 0)*100:.1f}%  95% CI [{bl['ci_95_low']:.3f}, {bl['ci_95_high']:.3f}]")
    print(f"Over-Valued (2023-24 -> 2025): {ov['hits']}/{ov['n_evaluated']} = "
          f"{(ov['hit_rate'] or 0)*100:.1f}%  95% CI [{ov['ci_95_low']:.3f}, {ov['ci_95_high']:.3f}]")

    outdir = ROOT / "results" / "causal_war" / "contrarian_stability"
    outdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_csv": str(_CSV_PATH),
        "followup_season": _FOLLOWUP_SEASON,
        "top_n": TOP_N,
        "buy_low": bl,
        "over_valued": ov,
    }
    (outdir / "hit_rates_reproduction.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8",
    )
    print(f"Wrote {outdir / 'hit_rates_reproduction.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
