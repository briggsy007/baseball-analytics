# Runbook: Nightly Refresh

Automated nightly chain that keeps the dashboard's data and leaderboard caches
fresh. Wrapper script: `scripts/nightly_refresh.py`. Scheduled via Windows Task
Scheduler at **06:30 local**.

## What it runs (in order)

| # | Step | Command | Required? |
|---|------|---------|-----------|
| 1 | Daily data refresh | `python scripts/daily_refresh.py` | yes |
| 2 | Leaderboard precompute | `python scripts/precompute.py --season <S> --tier 1 --force` | yes |
| 3 | Hit-parlay board | `python scripts/hit_parlay_today.py` | no (needs live lineups) |

Step 2 is the reason this wrapper exists: `daily_refresh.py` does **not** call
`precompute.py`, so leaderboard caches (Stuff+, DPI, etc.) go stale without it.

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
(pre-flight block) · `3` dry-run would be blocked.

## Disable / enable / delete the task

```
schtasks /change /tn "BaseballNightlyRefresh" /disable   # pause
schtasks /change /tn "BaseballNightlyRefresh" /enable    # resume
schtasks /run    /tn "BaseballNightlyRefresh"            # fire once now
schtasks /delete /tn "BaseballNightlyRefresh" /f         # remove
```

## Caveats

- **Exit-127-after-commit quirk.** Large DuckDB write transactions can exit
  non-zero at process teardown even though the COMMIT landed. The wrapper does
  not trust exit codes blindly: it verifies each step's real effect (step 1 →
  `pitches` watermark not regressed; step 2 → `leaderboard_cache` computed_at
  advanced; step 3 → today's hit-parlay JSON written). A verified step is marked
  `ok_verified` and does not fail the run.
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
