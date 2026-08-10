#!/usr/bin/env python
"""Nightly automation wrapper for the baseball analytics platform.

Orchestrates the three scripts that must run, in order, every night so the
dashboard's data and leaderboard caches stay fresh:

    1. scripts/daily_refresh.py                              (required)
       yesterday's Statcast ETL + roster/txn sync + matchup cache + pregame
    2. scripts/precompute.py --season <S> --tier 1 --force   (required)
       rebuilds tier-1 leaderboard caches (Stuff+, DPI, etc.) -- daily_refresh
       does NOT do this, so caches go stale without it
    3. scripts/hit_parlay_today.py                           (non-fatal)
       needs live lineups; tolerated to fail early in the day

This is a *wrapper only*. It does not modify daily_refresh.py / precompute.py
and it does not open a long-lived DuckDB writer -- each wrapped script opens
its own connection. The wrapper only:

  * refuses to start if the Streamlit dashboard is running or the DuckDB
    writer lock is held (DuckDB is single-writer; a running dashboard holds a
    lock and both wrapped writers would fail);
  * runs each step as a subprocess with per-step logging + timing;
  * tolerates the known "exit 127 at teardown after the COMMIT already landed"
    quirk by verifying each step's actual effect (data watermark / cache
    freshness / output file) instead of trusting the exit code blindly;
  * writes a machine-readable status.json for the run.

Usage
-----
    python scripts/nightly_refresh.py                 # full nightly run
    python scripts/nightly_refresh.py --dry-run       # plan + preflight only
    python scripts/nightly_refresh.py --season 2026   # override derived season

Logs land under  logs/nightly/YYYY-MM-DD/  :
    nightly.log            orchestrator timeline
    01_daily_refresh.log   subprocess stdout+stderr
    02_precompute.log
    03_hit_parlay.log
    status.json            final machine-readable summary
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent
DB_PATH: Path = ROOT / "data" / "baseball.duckdb"
LOG_ROOT: Path = ROOT / "logs" / "nightly"
PY: str = sys.executable  # same interpreter that launched us

TASK_NAME = "BaseballNightlyRefresh"

# Per-step wall-clock budgets (seconds). Generous; a hang is better caught than
# left to block the machine forever.
TIMEOUT_DAILY = 3600      # ETL can be slow on a big Statcast day
TIMEOUT_PRECOMPUTE = 3600  # tier-1 trains Stuff+ and xOut models
TIMEOUT_PARLAY = 900      # network-bound (MLB StatsAPI)


# ---------------------------------------------------------------------------
# Season derivation
# ---------------------------------------------------------------------------

def default_season(today: date | None = None) -> int:
    """Derive the MLB season from the calendar date.

    Feb-Nov is the live season and maps to the current year. December still
    reports the year that just concluded. January (deep offseason) rolls back
    to the prior completed season.
    """
    today = today or date.today()
    return today.year - 1 if today.month == 1 else today.year


# ---------------------------------------------------------------------------
# Tiny logger (console + nightly.log), timestamped
# ---------------------------------------------------------------------------

class Log:
    def __init__(self, path: Path):
        self.path = path
        self._fh = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str, level: str = "INFO") -> None:
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {level:<5} | {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dashboard / process detection
# ---------------------------------------------------------------------------

import re

# Matches the actual launch form -- `streamlit run`, `streamlit.exe run`, or
# `-m streamlit` -- NOT an incidental mention of the word "streamlit" in some
# unrelated shell/editor/grep command line.
_STREAMLIT_LAUNCH = re.compile(r"""streamlit(\.exe)?["']?\s+run\b|-m\s+streamlit""",
                               re.IGNORECASE)


def _looks_like_dashboard(cmdline: str) -> bool:
    """True if a command line looks like `streamlit run .../dashboard/app.py`.

    Requires the streamlit *launch pattern* AND an app/dashboard token, so a
    shell that merely mentions the word 'streamlit' (e.g. this orchestrator's
    own invocation, a grep, or an editor buffer) is not mistaken for a running
    dashboard. The DuckDB writer-lock probe is the authoritative gate regardless.
    """
    cl = cmdline.lower()
    if not _STREAMLIT_LAUNCH.search(cl):
        return False
    return ("app.py" in cl) or ("dashboard" in cl)


def find_dashboard_processes() -> dict:
    """Return {'matches': [str, ...], 'method': str}.

    Tries psutil, then a PowerShell CIM query (reliable on Win11, gives full
    command lines), then a name-only tasklist for streamlit.exe as a last
    resort. Never raises -- inability to enumerate is reported as method
    'unavailable' and lets the DuckDB writer-lock probe act as the real gate.
    """
    self_pid = os.getpid()

    # 1) psutil (not installed today, but future-proof and cheapest)
    try:
        import psutil  # type: ignore

        matches = []
        for p in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                pid = p.info["pid"]
                if pid == self_pid:
                    continue
                cl = " ".join(p.info.get("cmdline") or [])
                if _looks_like_dashboard(cl):
                    matches.append(f"pid={pid} :: {cl}")
            except Exception:
                continue
        return {"matches": matches, "method": "psutil"}
    except ImportError:
        pass
    except Exception:
        pass

    # 2) PowerShell CIM (full command lines)
    try:
        ps = (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -and $_.CommandLine -match 'streamlit' } | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, timeout=30,
        )
        raw = (out.stdout or "").strip()
        if raw:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                parsed = [parsed]
            matches = []
            for item in parsed:
                pid = item.get("ProcessId")
                cl = item.get("CommandLine") or ""
                if pid == self_pid:
                    continue
                if _looks_like_dashboard(cl):
                    matches.append(f"pid={pid} :: {cl.strip()[:200]}")
            return {"matches": matches, "method": "powershell-cim"}
    except Exception:
        pass

    # 3) tasklist name-only (only catches a distinct streamlit.exe launcher)
    try:
        out = subprocess.run(
            ["tasklist", "/fi", "imagename eq streamlit.exe", "/fo", "csv", "/nh"],
            capture_output=True, text=True, timeout=30,
        )
        if "streamlit.exe" in (out.stdout or "").lower():
            return {"matches": ["streamlit.exe (via tasklist)"], "method": "tasklist"}
        return {"matches": [], "method": "tasklist"}
    except Exception:
        return {"matches": [], "method": "unavailable"}


# ---------------------------------------------------------------------------
# DuckDB probes (read-only connectivity + writer-lock availability)
# ---------------------------------------------------------------------------

def probe_read_only() -> tuple[bool, str | None]:
    """Open a brief read_only connection. Fails if another process holds the
    read-write (exclusive) lock, which is the clearest signal the DB is busy."""
    try:
        import duckdb
        c = duckdb.connect(str(DB_PATH), read_only=True)
        c.execute("SELECT 1").fetchone()
        c.close()
        return True, None
    except Exception as exc:
        return False, str(exc)


def probe_writer() -> tuple[bool, str | None]:
    """Open a writer connection and immediately close it.

    This is the authoritative single-writer gate: it fails if ANY other process
    holds the DB open (a reader-only dashboard blocks a new writer too). The
    connection is released immediately so the wrapped scripts can acquire it.
    Only called in a real run, right before the first writing step.
    """
    try:
        import duckdb
        c = duckdb.connect(str(DB_PATH), read_only=False)
        c.close()
        return True, None
    except Exception as exc:
        return False, str(exc)


def read_watermark() -> dict:
    """Cheap read_only snapshot used for effect-verification of the steps."""
    snap: dict = {}
    try:
        import duckdb
        c = duckdb.connect(str(DB_PATH), read_only=True)
        try:
            pmax = c.execute("SELECT MAX(game_date) FROM pitches").fetchone()[0]
            snap["pitches_max_game_date"] = str(pmax) if pmax is not None else None
        except Exception as exc:
            snap["pitches_max_game_date"] = None
            snap["pitches_error"] = str(exc)
        try:
            row = c.execute(
                "SELECT updated_at FROM data_freshness WHERE table_name='pitches'"
            ).fetchone()
            snap["data_freshness_pitches_updated_at"] = (
                row[0].isoformat() if row and row[0] else None
            )
        except Exception:
            snap["data_freshness_pitches_updated_at"] = None
        try:
            row = c.execute("SELECT MAX(computed_at) FROM leaderboard_cache").fetchone()
            snap["leaderboard_cache_max_computed_at"] = (
                row[0].isoformat() if row and row[0] else None
            )
        except Exception:
            snap["leaderboard_cache_max_computed_at"] = None
        c.close()
    except Exception as exc:
        snap["error"] = str(exc)
    return snap


def _leaderboard_max_computed_at() -> datetime | None:
    try:
        import duckdb
        c = duckdb.connect(str(DB_PATH), read_only=True)
        row = c.execute("SELECT MAX(computed_at) FROM leaderboard_cache").fetchone()
        c.close()
        return row[0] if row and row[0] else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def _subprocess_env() -> dict:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_step(log: Log, step_dir: Path, index: int, name: str,
             cmd: list[str], timeout: int, required: bool) -> dict:
    """Run one wrapped script as a subprocess, teeing stdout+stderr to a file."""
    step_log = step_dir / f"{index:02d}_{name}.log"
    started = datetime.now(timezone.utc)
    log(f"STEP {index} [{name}] start  ->  {' '.join(cmd)}")
    log(f"STEP {index} [{name}] log    ->  {step_log}")

    rc: int | None = None
    timed_out = False
    t0 = time.time()
    try:
        with open(step_log, "w", encoding="utf-8") as lf:
            lf.write(f"# step {index}: {name}\n# started: {started.isoformat()}\n")
            lf.write(f"# cmd: {' '.join(cmd)}\n\n")
            lf.flush()
            proc = subprocess.run(
                cmd, cwd=str(ROOT), stdout=lf, stderr=subprocess.STDOUT,
                env=_subprocess_env(), timeout=timeout,
            )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        rc = None
        log(f"STEP {index} [{name}] TIMEOUT after {timeout}s", "ERROR")
    except Exception as exc:
        log(f"STEP {index} [{name}] launch error: {exc}", "ERROR")
        rc = None

    elapsed = round(time.time() - t0, 1)
    log(f"STEP {index} [{name}] finished rc={rc} in {elapsed}s")
    return {
        "index": index,
        "name": name,
        "cmd": cmd,
        "required": required,
        "log": str(step_log.relative_to(ROOT)).replace("\\", "/"),
        "returncode": rc,
        "timed_out": timed_out,
        "seconds": elapsed,
        "started_at": started.isoformat(),
    }


# ---------------------------------------------------------------------------
# Effect verification (tolerates exit-127-after-commit)
# ---------------------------------------------------------------------------

def _classify(rc: int | None, effect_ok: bool, timed_out: bool) -> str:
    """Map (exit code, verified effect) to a status.

    ok           - clean exit 0
    ok_verified  - nonzero/None exit but the effect check confirms it worked
                   (covers the documented exit-127-at-teardown quirk)
    fail         - nonzero exit and effect could not be confirmed
    """
    if timed_out:
        return "fail"
    if rc == 0:
        return "ok"
    if effect_ok:
        return "ok_verified"
    return "fail"


def verify_daily_refresh(step: dict, before: dict) -> dict:
    """Effect check: pitches watermark present and not regressed."""
    after = read_watermark()
    before_max = before.get("pitches_max_game_date")
    after_max = after.get("pitches_max_game_date")
    effect_ok = after_max is not None and (before_max is None or after_max >= before_max)
    step["verify"] = {
        "check": "pitches_max_game_date present and >= pre-run",
        "before": before_max,
        "after": after_max,
        "effect_ok": effect_ok,
    }
    step["status"] = _classify(step["returncode"], effect_ok, step["timed_out"])
    return step


def verify_precompute(step: dict) -> dict:
    """Effect check: leaderboard_cache has an entry computed at/after step start."""
    step_start = datetime.fromisoformat(step["started_at"]).replace(tzinfo=None)
    lb_max = _leaderboard_max_computed_at()
    effect_ok = lb_max is not None and lb_max >= step_start
    step["verify"] = {
        "check": "leaderboard_cache.max(computed_at) >= step start",
        "step_start_utc": step_start.isoformat(),
        "leaderboard_max_computed_at": lb_max.isoformat() if lb_max else None,
        "effect_ok": effect_ok,
    }
    step["status"] = _classify(step["returncode"], effect_ok, step["timed_out"])
    return step


def verify_hit_parlay(step: dict, run_day: str, run_start: float) -> dict:
    """Effect check: results/hit_parlay/<day>.json written this run.

    Non-fatal step: never yields 'fail' at the overall level. A missing output
    (no lineups posted yet) is reported as 'warn'.
    """
    out = ROOT / "results" / "hit_parlay" / f"{run_day}.json"
    fresh = out.exists() and out.stat().st_mtime >= run_start
    step["verify"] = {
        "check": f"results/hit_parlay/{run_day}.json written this run",
        "path": str(out.relative_to(ROOT)).replace("\\", "/"),
        "exists": out.exists(),
        "fresh": fresh,
    }
    if step["returncode"] == 0:
        step["status"] = "ok"
    elif fresh:
        step["status"] = "ok_verified"
    else:
        step["status"] = "warn"
    return step


# ---------------------------------------------------------------------------
# Status writer
# ---------------------------------------------------------------------------

def write_status(step_dir: Path, status: dict) -> Path:
    path = step_dir / "status.json"
    path.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Single-instance lock (WS0.4)
# ---------------------------------------------------------------------------
# schtasks' IgnoreNew policy only governs *scheduler-triggered* starts; a
# manual launch can still overlap a running chain. A process-level file lock
# is the real mutual-exclusion gate. Uses the `filelock` package (installed);
# falls back to an O_EXCL lockfile if it is ever missing.

LOCK_PATH = LOG_ROOT / "nightly.lock"
_LOCK_BLOCKED_EXIT = 4


class _OExclLock:
    """Minimal O_EXCL lockfile fallback when `filelock` is unavailable.

    NOT reentrant and leaves a stale file if the process is killed hard;
    `filelock` (which recovers stale locks) is strongly preferred.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fd: int | None = None

    def __enter__(self):
        try:
            self._fd = os.open(
                str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
            os.write(self._fd, str(os.getpid()).encode())
        except FileExistsError as exc:
            raise TimeoutError(f"lock already held: {self.path}") from exc
        return self

    def __exit__(self, *exc_info):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            self.path.unlink()
        except OSError:
            pass
        return False


def _acquire_nightly_lock():
    """Return an acquired, non-blocking lock context or raise Timeout."""
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        from filelock import FileLock, Timeout  # type: ignore

        lock = FileLock(str(LOCK_PATH))
        try:
            lock.acquire(timeout=0)
        except Timeout as exc:
            raise TimeoutError(
                f"another nightly run holds {LOCK_PATH}"
            ) from exc
        return lock
    except ImportError:
        return _OExclLock(LOCK_PATH).__enter__()


def _release_nightly_lock(lock) -> None:
    try:
        if hasattr(lock, "release"):
            lock.release()
        else:
            lock.__exit__(None, None, None)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Acquire the single-instance nightly lock, then run the chain."""
    try:
        lock = _acquire_nightly_lock()
    except TimeoutError as exc:
        print(
            f"REFUSED: nightly chain already running ({exc}). "
            f"Exit {_LOCK_BLOCKED_EXIT}.",
            flush=True,
        )
        return _LOCK_BLOCKED_EXIT
    try:
        return _run()
    finally:
        _release_nightly_lock(lock)


def _run() -> int:
    ap = argparse.ArgumentParser(description="Nightly refresh orchestration wrapper.")
    ap.add_argument("--season", type=int, default=None,
                    help="MLB season for precompute (default: derived from today's date).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and run preflight checks without executing any step.")
    args = ap.parse_args()

    season = args.season if args.season is not None else default_season()
    run_day = date.today().strftime("%Y-%m-%d")
    step_dir = LOG_ROOT / run_day
    step_dir.mkdir(parents=True, exist_ok=True)

    log = Log(step_dir / "nightly.log")
    started_utc = datetime.now(timezone.utc)
    run_start_epoch = time.time()

    # Plan: (name, cmd, timeout, required)
    plan = [
        ("daily_refresh",
         [PY, "-u", str(ROOT / "scripts" / "daily_refresh.py")],
         TIMEOUT_DAILY, True),
        ("precompute",
         [PY, "-u", str(ROOT / "scripts" / "precompute.py"),
          "--season", str(season), "--tier", "1", "--force"],
         TIMEOUT_PRECOMPUTE, True),
        ("hit_parlay",
         [PY, "-u", str(ROOT / "scripts" / "hit_parlay_today.py")],
         TIMEOUT_PARLAY, False),
    ]

    log("=" * 64)
    log(f"NIGHTLY REFRESH  run_day={run_day}  season={season}  dry_run={args.dry_run}")
    log(f"repo   : {ROOT}")
    log(f"db     : {DB_PATH}")
    log(f"python : {PY}")
    log("Planned steps:")
    for i, (n, c, to, req) in enumerate(plan, 1):
        tag = "required" if req else "non-fatal"
        log(f"  {i}. {n:<14} [{tag}] timeout={to}s :: {' '.join(c)}")

    # --- Preflight: dashboard process + read-only connectivity ---------------
    dash = find_dashboard_processes()
    ro_ok, ro_err = probe_read_only()
    log(f"Preflight: dashboard scan via {dash['method']} -> "
        f"{len(dash['matches'])} match(es)")
    for m in dash["matches"]:
        log(f"           MATCH: {m}", "WARN")
    log(f"Preflight: read_only DB probe -> {'OK' if ro_ok else 'BLOCKED'}"
        + ("" if ro_ok else f" ({ro_err})"))

    blocked: list[str] = []
    if dash["matches"]:
        blocked.append(f"dashboard/streamlit process running: {dash['matches']}")
    if not ro_ok:
        blocked.append(f"read_only DB probe failed (writer lock likely held): {ro_err}")

    status: dict = {
        "run_day": run_day,
        "season": season,
        "started_at": started_utc.isoformat(),
        "dry_run": args.dry_run,
        "preflight": {
            "dashboard_scan_method": dash["method"],
            "dashboard_matches": dash["matches"],
            "read_only_probe": "ok" if ro_ok else f"blocked: {ro_err}",
            "writer_probe": "not-run",
        },
        "steps": [],
        "overall_status": None,
        "refused_reason": None,
        "post_run": {},
    }

    # --- Dry run: report and stop (never opens a writer) ---------------------
    if args.dry_run:
        log("DRY RUN: writer lock is only probed at write-time in a real run; "
            "not opening a writer now.")
        would_block = bool(blocked)
        for b in blocked:
            log(f"DRY RUN would BLOCK on: {b}", "WARN")
        log(f"DRY RUN result: {'WOULD BE BLOCKED' if would_block else 'WOULD PROCEED'}")
        status["overall_status"] = "dry_run"
        status["would_proceed"] = not would_block
        status["blocked_on"] = blocked
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        sp = write_status(step_dir, status)
        log(f"Wrote {sp}")
        log("=" * 64)
        log.close()
        return 3 if would_block else 0

    # --- Real run: refuse on preflight block --------------------------------
    if blocked:
        for b in blocked:
            log(f"REFUSED: {b}", "ERROR")
        status["overall_status"] = "refused"
        status["refused_reason"] = blocked
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        sp = write_status(step_dir, status)
        log(f"Wrote {sp}")
        log.close()
        return 2

    # --- Writer-lock probe (authoritative single-writer gate) ---------------
    w_ok, w_err = probe_writer()
    log(f"Preflight: writer lock probe -> {'OK' if w_ok else 'BLOCKED'}"
        + ("" if w_ok else f" ({w_err})"))
    status["preflight"]["writer_probe"] = "ok" if w_ok else f"blocked: {w_err}"
    if not w_ok:
        log(f"REFUSED: cannot acquire DuckDB writer lock: {w_err}", "ERROR")
        status["overall_status"] = "refused"
        status["refused_reason"] = [f"writer lock unavailable: {w_err}"]
        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        sp = write_status(step_dir, status)
        log(f"Wrote {sp}")
        log.close()
        return 2

    # --- Execute steps ------------------------------------------------------
    before = read_watermark()
    log(f"Pre-run watermark: pitches_max={before.get('pitches_max_game_date')}")

    for i, (name, cmd, to, req) in enumerate(plan, 1):
        step = run_step(log, step_dir, i, name, cmd, to, req)
        if name == "daily_refresh":
            step = verify_daily_refresh(step, before)
        elif name == "precompute":
            step = verify_precompute(step)
        elif name == "hit_parlay":
            step = verify_hit_parlay(step, run_day, run_start_epoch)
        note = ""
        if step["status"] == "ok_verified":
            note = " (nonzero exit tolerated: effect verified -- see exit-127 quirk)"
        log(f"STEP {i} [{name}] status={step['status']}{note}")
        status["steps"].append(step)
        # persist incrementally so a crash still leaves a partial status.json
        write_status(step_dir, status)

    # --- Post-run summary ---------------------------------------------------
    status["post_run"] = read_watermark()

    required_fail = any(s["required"] and s["status"] == "fail" for s in status["steps"])
    has_warnings = any(
        s["status"] in ("ok_verified", "warn", "fail") for s in status["steps"]
    )
    if required_fail:
        overall = "failed"
    elif has_warnings:
        overall = "ok_with_warnings"
    else:
        overall = "ok"
    status["overall_status"] = overall
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    status["total_seconds"] = round(time.time() - run_start_epoch, 1)

    sp = write_status(step_dir, status)
    log(f"Post-run watermark: pitches_max={status['post_run'].get('pitches_max_game_date')}")
    log(f"OVERALL: {overall}  ({status['total_seconds']}s total)")
    log(f"Wrote {sp}")
    log("=" * 64)
    log.close()

    return 1 if overall == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
