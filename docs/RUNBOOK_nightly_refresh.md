# Runbook: Nightly Refresh

Automated nightly chain that keeps the dashboard's data and leaderboard caches
fresh. Wrapper script: `scripts/nightly_refresh.py`. Scheduled via Windows Task
Scheduler at **06:30 local**.

## What it runs (in order)

| # | Step | Command | Required? |
|---|------|---------|-----------|
| 1 | Artifact registry gate (pre) | `python scripts/verify_artifacts.py` | yes — a tampered frozen artifact ABORTS the chain |
| 2 | Daily data refresh | `python scripts/daily_refresh.py` | yes |
| 3 | Leaderboard precompute | `python scripts/precompute.py --season <S> --tier 1 --force` | yes |
| 4 | Hit-parlay board | `python scripts/hit_parlay_today.py` | no (needs live lineups) |
| 5 | Morning pick resolver | `python scripts/resolve_picks.py` | no (read-only DB + JSONL appends) |
| 6 | Artifact registry gate (post) | `python scripts/verify_artifacts.py` | yes — proves the run modified no pinned artifact |

Step 3 is the reason this wrapper exists: `daily_refresh.py` does **not** call
`precompute.py`, so leaderboard caches (Stuff+, DPI, etc.) go stale without it.
Steps 1 and 6 are the WS2.1 frozen-evidence integrity gates.

`<S>` (season) is derived from the calendar date — Feb–Dec map to that year,
January rolls back to the prior completed season. Override with `--season`.

## Pre-flight safety (why a run may refuse to start)

DuckDB is single-writer. Before writing, the wrapper refuses (clear log +
non-zero exit `2`) if either is true:

1. A **Streamlit dashboard** process is running (detected by the `streamlit run …
   dashboard/app.py` launch pattern via psutil → PowerShell CIM → tasklist).
2. The **DuckDB writer lock** can't be acquired — probed first with a brief
   `read_only=True` connection, then (real runs only) a momentary writer open
   that is released immediately before step 1 starts.

**If a run refuses:** stop the dashboard (close the `streamlit run` terminal),
confirm no other writer is open, then re-run manually (below).

## Register the scheduled task

Run this **once** in an elevated (or normal) `cmd`/PowerShell — creates a daily
06:30 task under the current user (runs only while logged in; the simplest,
password-free, reliable form). `/f` overwrites an existing task of the same name.

```
schtasks /create /tn "BaseballNightlyRefresh" /sc daily /st 06:30 /tr "cmd /c cd /d C:\Users\hunte\projects\baseball && C:\Users\hunte\AppData\Local\Programs\Python\Python312\python.exe scripts\nightly_refresh.py" /f
```

The script manages its own logs, so no output redirection is needed in `/tr`.

> **This is a cmd/PowerShell command and does NOT survive a paste into Git Bash.**
> MSYS path conversion rewrites the `/create` flag into `C:/Program Files/Git/create`
> and the command fails with `ERROR: Invalid argument/option`. Hit for real on
> 2026-08-16. In Git Bash, use:
>
> ```
> MSYS_NO_PATHCONV=1 schtasks /create /tn "BaseballNightlyRefresh" /sc daily /st 06:30 /tr 'cmd /c cd /d C:\Users\hunte\projects\baseball && C:\Users\hunte\AppData\Local\Programs\Python\Python312\python.exe scripts\nightly_refresh.py' /f
> ```
>
> Single-quote the `/tr` value so bash leaves the backslashes and `&&` alone.
> (Doubling the slashes — `//create`, `//tn` — also works.)

## Check status

Logs land under `logs/nightly/YYYY-MM-DD/`:

- `status.json` — machine-readable summary (steps, ok/fail, durations, pre/post
  `pitches` max game_date). Check `overall_status`:
  - `ok` — all clean
  - `ok_with_warnings` — a required step exited non-zero but its effect was
    verified (see exit-127 caveat), and/or the non-fatal hit-parlay step warned
  - `failed` — a required step failed and its effect could not be verified
  - `refused` — pre-flight blocked the run (see `refused_reason`)
- `nightly.log` — orchestrator timeline.
- `NN_<step>.log` — each step's full stdout+stderr.

Quick check (PowerShell):
```
Get-Content "C:\Users\hunte\projects\baseball\logs\nightly\$(Get-Date -Format yyyy-MM-dd)\status.json" | ConvertFrom-Json | Select overall_status, season
```

Last scheduled run result:
```
schtasks /query /tn "BaseballNightlyRefresh" /v /fo list
```

## Run manually / preview

```
# full run now
C:\Users\hunte\AppData\Local\Programs\Python\Python312\python.exe scripts\nightly_refresh.py

# dry run: prints the plan + runs the dashboard/lock pre-flight, executes nothing
C:\Users\hunte\AppData\Local\Programs\Python\Python312\python.exe scripts\nightly_refresh.py --dry-run

# override season
C:\Users\hunte\AppData\Local\Programs\Python\Python312\python.exe scripts\nightly_refresh.py --season 2026
```

Exit codes: `0` ok / ok_with_warnings · `1` a required step failed · `2` refused
(pre-flight block) · `3` dry-run would be blocked · `4` another nightly run
already holds the single-instance lock.

## Addendum 2026-08-10 (WS0.1 / WS0.4)

- **Single-instance lock.** The wrapper now acquires `logs/nightly/nightly.lock`
  (via `filelock`, non-blocking) before doing anything. A second concurrent
  launch — manual or scheduled — refuses immediately with exit `4`. schtasks'
  IgnoreNew policy only governs scheduler triggers; the lock is the real gate.
- **Task registration status.** ~~As of 2026-08-10 no `BaseballNightlyRefresh`
  task is registered.~~ **REGISTERED 2026-08-16**: daily 06:30, Status `Ready`,
  Run As User `hunte`, first scheduled fire 2026-08-17 06:30. Between 2026-08-10
  and 2026-08-16 nothing ran automatically, which is what produced a five-day
  Statcast gap (Aug 11–15, 19,684 pitches, backfilled 2026-08-16) and let the
  matchup-cache abort below go unnoticed.
- **Frozen vs in-season model artifacts.** Retrains inside the chain
  (`daily_refresh --full`, tier-1 precompute first-run, `retrain_active_2026`)
  write ONLY the in-season artifacts `models/stuff_model_2026_inseason.pkl` /
  `models/defensive_pressing/xout_2026_inseason.pkl` (gitignored). The frozen
  validated artifacts `models/stuff_model.pkl` / `models/defensive_pressing/xout_v1.pkl`
  are never written by any production path; overwriting them requires an
  explicit `allow_frozen_overwrite=True` / `--allow-checkpoint-overwrite`.
- **`retrain_active_2026` status JSON** now lands at
  `logs/nightly/retrain_status.json` (was a dead hard-coded scratchpad path).
- **Contrarian mid-season boards** are written as dated copies under
  `results/edges/contrarian_2026_midseason/YYYY-MM-DD/` plus an atomically
  replaced `latest.json` pointer; the top-level `board.csv` / `summary.md` are
  frozen legacy artifacts (first generation, preserved as the `2026-08-10`
  dated copy).

## Disable / enable / delete the task

```
schtasks /change /tn "BaseballNightlyRefresh" /disable   # pause
schtasks /change /tn "BaseballNightlyRefresh" /enable    # resume
schtasks /run    /tn "BaseballNightlyRefresh"            # fire once now
schtasks /delete /tn "BaseballNightlyRefresh" /f         # remove
```

## Caveats

- **Nonzero-exit-after-commit quirk — and its dangerous edge (corrected 2026-08-16).**
  Large DuckDB write transactions *can* exit non-zero at process teardown after
  the COMMIT genuinely landed, so the wrapper does not trust exit codes blindly:
  it verifies each step's real effect and marks a verified step `ok_verified`.
  That tolerance is still correct — but the previous wording asserted the COMMIT
  had landed **unconditionally**, and that was false in at least one case.

  On 2026-08-11 the `daily_refresh` step recorded **rc 3221226505 (0xC0000409,
  Windows fast-fail abort** — bash renders it as "127"**)** and the COMMIT had
  **not** landed: the process was aborting *inside* the `matchup_summary`
  rebuild, and the transaction rolled back. Reproduced three times. Because the
  `pitches` watermark was written early in the ETL and was therefore healthy,
  the step still classified `ok_verified` and the chain carried on — for months.
  Everything downstream of the rebuild inside `daily_refresh.py` (roster sync,
  transaction sync, the step-4 cache rebuild, the pregame report) silently never
  ran, and the cache drifted 923,797 pitches stale.

  **A nonzero exit is now only excusable if the effect check independently
  proves the work landed.** The `daily_refresh` check was accordingly tightened
  to require the `pitches` watermark AND matchup-cache sync
  (`SUM(matchup_summary.num_pitches) == COUNT(pitches WHERE pitch_type IS NOT NULL)`),
  the latter being the *tail* of the ETL and therefore real evidence it ran to
  completion. Both numbers are recorded in `status.json`. The root cause — the
  cache table's `PRIMARY KEY` and its ART index — was removed; the rebuild now
  replaces the table wholesale (`DROP` + PK-less `CREATE` + `INSERT`), which
  also migrates any existing PK'd copy. See `src/db/queries.py::refresh_matchup_cache`.
- **Logged-in requirement.** The task as registered runs only while the user is
  logged in. To run when logged off, re-create it with `/ru <user> /rp <pass>`
  (not required for the intended single-user desktop setup).
- **Dashboard must be stopped.** The nightly run and a live dashboard cannot
  both hold the writer; the run will refuse rather than corrupt state. Keep the
  dashboard closed overnight, or expect a `refused` status if it's left open.
- **Season boundary.** January maps to the prior year's season; if you truly
  want the upcoming season in deep offseason, pass `--season` explicitly.
- **FanGraphs 403.** `daily_refresh` season-stat fetches may 403 on FanGraphs;
  the underlying code already falls back to Baseball-Reference. Harmless.
