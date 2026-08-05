#!/usr/bin/env python
"""Retrain the 3 active flagships that learn from 2026 data, then refresh
their dashboard leaderboard caches.

Covers: Stuff+, Defensive Pressing (DPI), CausalWAR.

NOT covered (by design): PitchGPT — a frozen fixed-cohort research artifact
(train 2015-2022 / test 2025) whose backbone SHA256 is load-bearing for the
outcome head + calibrations. Retraining it would neither ingest 2026 nor be
safe; its integrity is verified separately via tests/test_pitchgpt_sim.py.

Lock discipline: Phase 1 retrains model artifacts over a READ-ONLY DuckDB
connection (writes only to models/), closes it, THEN Phase 2 opens the
single writer for cache refresh. The two phases never overlap, so no
DuckDB lock conflict.

Usage
-----
    python scripts/retrain_active_2026.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STATUS_PATH = Path(
    r"C:\Users\hunte\AppData\Local\Temp\claude"
    r"\C--Users-hunte-projects-baseball"
    r"\1284a947-3226-496c-b90a-c86e975dce67\scratchpad\retrain_status.json"
)
SEASON = 2026
status: dict = {"started": datetime.now().isoformat(), "steps": {}, "done": False}


def _save():
    try:
        STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATUS_PATH.write_text(json.dumps(status, indent=2, default=str))
    except Exception:
        pass


def _mark(step: str, **kv):
    status["steps"][step] = {**kv, "at": datetime.now().isoformat()}
    _save()
    print(f"::STEP {step} :: {kv}", flush=True)


def main() -> None:
    from src.db.schema import get_connection

    t0 = time.time()

    # ---- Phase 1: artifact retrains (READ-ONLY DB; writes to models/) --------
    print("=== PHASE 1: artifact retrains (read-only DB) ===", flush=True)
    conn_ro = get_connection(read_only=True)

    # 1a. Stuff+
    try:
        from src.analytics.stuff_model import train_stuff_model, DEFAULT_MODEL_PATH
        m = train_stuff_model(conn_ro)
        _mark("stuff_plus_artifact", ok=True,
              r2_test=m.get("r2_test"), rmse_test=m.get("rmse_test"),
              path=str(DEFAULT_MODEL_PATH))
        print("PHASE1_STUFF_DONE", flush=True)
    except Exception as exc:
        _mark("stuff_plus_artifact", ok=False, error=str(exc))
        print(f"PHASE1_STUFF_FAIL: {exc}", flush=True)

    # 1b. DPI xOut (persist checkpoint the live dashboard loads, incl. 2026 BIP)
    try:
        import src.analytics.defensive_pressing as dp
        res = dp.fit_xout(conn_ro, seasons=list(range(2015, SEASON + 1)))
        _mark("dpi_xout_artifact", ok=True,
              detail={k: res.get(k) for k in list(res)[:6]} if isinstance(res, dict) else str(res))
        print("PHASE1_DPI_DONE", flush=True)
    except Exception as exc:
        _mark("dpi_xout_artifact", ok=False, error=str(exc))
        print(f"PHASE1_DPI_FAIL: {exc}", flush=True)

    conn_ro.close()  # release read lock BEFORE any writer opens
    print("=== PHASE 1 complete, read lock released ===", flush=True)

    # ---- Phase 2: leaderboard cache refresh (single WRITER, sequential) ------
    print("=== PHASE 2: dashboard cache refresh (read-write DB) ===", flush=True)
    from scripts.precompute import run_precompute

    for model in ("stuff_plus", "defensive_pressing", "causal_war"):
        try:
            print(f"--- precompute {model} ---", flush=True)
            results = run_precompute(season=SEASON, model_filter=model, force=True)
            r = results[0] if results else {"status": "no_result", "rows": 0}
            _mark(f"precompute_{model}", ok=(r.get("status") == "ok"),
                  rows=r.get("rows"), seconds=r.get("seconds"), st=r.get("status"))
            print(f"PRECOMPUTE_{model}_DONE status={r.get('status')} rows={r.get('rows')}", flush=True)
        except Exception as exc:
            _mark(f"precompute_{model}", ok=False, error=str(exc))
            print(f"PRECOMPUTE_{model}_FAIL: {exc}", flush=True)

    status["done"] = True
    status["elapsed_sec"] = round(time.time() - t0, 1)
    status["finished"] = datetime.now().isoformat()
    _save()
    print(f"RETRAIN_COMPLETE elapsed={status['elapsed_sec']}s", flush=True)


if __name__ == "__main__":
    main()
