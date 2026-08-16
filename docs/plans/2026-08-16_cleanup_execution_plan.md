# 2026-08-16 Cleanup Execution Plan

Single-day plan. Executes to completion today. Phase: post-audit remediation (batches A–D done);
this closes the operational debt found on 2026-08-16 plus the small open items from
`docs/audits/REMEDIATION_EXECUTION_2026-08-11.md` §5 that fit in a day.

---

## 0. Execution mode — AUTONOMOUS RUN

This doc is executed by an autonomous multi-agent run. All five decisions are **DECIDED** (§4);
no decision-shaped question remains open, and none may be re-litigated mid-run.

**Decisions of record (user-adjudicated 2026-08-16):**

| # | Decision | RECORDED ANSWER |
|---|---|---|
| D1 | Coverage gate | **Ratchet `fail_under` 90 → 50**, keep `--cov` |
| D2 | Production-path ECE contact | **DEFER to season end** — record the deferral, spend nothing |
| D3 | v3 lockbox contact | **FORFEIT** — adjudicate today via deviations-log append |
| D4 | Fire `schtasks /run` today | **Yes** — final verification gate |
| D5 | Drop `matchup_summary` PRIMARY KEY | **Yes** — GROUP BY guarantees uniqueness; the ART index is the crash site |

**The two USER gates** (the permission classifier denies scheduled-task commands to Claude —
these are the *only* steps a human must perform):

- **GATE 1 — before the run starts:** register the task (P2). Must be done first; the autonomous
  run verifies registration in P0 and proceeds regardless, but "automation registered" is the
  day's headline outcome.
- **GATE 2 — after P8 completes:** fire `!schtasks /run /tn "BaseballNightlyRefresh"` as the
  final proof of scheduler-context invocation. **Moved to the end** (was P5) so the autonomous
  run is never blocked mid-flight waiting on a human.

**Resequenced order for autonomous execution:**
`P0 → P1 → P3 → P4 → P6 → P7 → P8 → [GATE 2: user fires /run] → P5-verify`

P2 is GATE 1 (done before the run). P5's *archive-logs* substep moves to just before GATE 2; its
verification becomes the closing step. Everything between P0 and P8 runs without human input.

**Standing rules for the run:** the §3 kill criterion is binding and pre-registered — do not
extend the budget. Do not touch frozen artifacts, and never pass `--allow-checkpoint-overwrite` /
`allow_frozen_overwrite=True`. Ledgers are append-only (§6). No `claims.yaml` edits. If
`verify_artifacts` goes red at any point, STOP the entire run.

---

## 1. End state

By end of day: the nightly chain is **registered in Windows Task Scheduler and proven under
scheduler invocation**, the `matchup_summary` cache is **in sync with `pitches` and its rebuild no
longer hard-aborts** (with a nightly effect-check that can actually see it if it ever regresses),
leaderboard caches are **computed today**, the three user adjudications (coverage gate, ECE
contact, v3 lockbox contact) are **decided and recorded**, every doc that described the old broken
state (runbook, validate-model roster, remediation open-items) **tells the truth**, and the tree is
**clean and pushed** with the suite exiting 0 under the decided coverage convention. We know we got
there when the Final Checklist (§7) — nine commands, each with a pass condition — is all green.

### Corrections to the session findings (verified today, plan is built on these)

- The 8/11 nightly's `daily_refresh` step recorded **rc=3221226505 (0xC0000409, Windows fast-fail
  abort)** in `logs/nightly/2026-08-11/status.json`, classified `ok_verified`. "Exit 127" is the
  bash rendering; the doc trail should cite 0xC0000409.
- **Blast radius is bigger than a stale cache.** `02_daily_refresh.log` ends mid-line at
  "Refreshing matchup_summary cache" — the process died inside `run_daily_etl` step 5, so
  **roster sync, transaction sync, the step-4 rebuild, and the pregame report never ran** on 8/11
  (and every night this pattern held). Fixing the rebuild un-breaks four downstream steps.
- There are **two rebuild implementations**: `src/ingest/daily_etl.py::_refresh_matchup_cache`
  (txn-wrapped; the one that dies) and `src/db/queries.py::refresh_matchup_cache` (no explicit
  txn; called by `scripts/daily_refresh.py::_step_matchup_cache`, currently unreachable because
  the process is already dead). The fix must unify them.
- Rebuild sizes today: DELETE would remove **1,671,256** rows; the fresh aggregate is
  **1,851,623** rows / 8,161,237 pitches. Cache currently holds 7,237,440 pitches → **923,797
  stale** (matches the session finding exactly).
- The freshness noise is at `src/ingest/statcast_loader.py:1091` and `:1141` —
  `max_game_date=str(max(seasons))` puts `"2026"` into a DATE column; `check_data_freshness` is
  only ever consulted for `pitches`, so passing `None` is safe.
- New minor finding (not in session notes): `transactions.transaction_date` has garbage
  (`MAX = 2925-11-26`). Filed in §6, not today.

---

## 2. Ordered execution sequence

Serialization rule: steps P3c → P4 → P5 each open a DuckDB **writer** and must not overlap each
other or a running dashboard. Dashboard stays closed until P8 is done.

### P0 — Preflight (CLAUDE, 5 min)
- **Do:** `python scripts/nightly_refresh.py --dry-run` from repo root.
- **Verify:** exit 0 and log line `DRY RUN result: WOULD PROCEED`; verify_artifacts step `ok`
  (baseline re-confirmed today: `ok=24 warn=0 fail=0`).
- **If it fails:** a dashboard/writer is open — close it and re-run. If the artifact registry is
  red, STOP the whole plan (frozen-evidence integrity comes first).

### P1 — Commit and push this morning's session work (CLAUDE, 10 min)
- **Do:** `git add docs/NORTH_STAR_CURRENT.md scripts/backfill_2026_05_to_07_gap.py
  docs/plans/2026-08-16_cleanup_execution_plan.md` then `/commit` (project style, no co-author),
  then `git push`. (This plan doc is currently untracked — commit it with the session work so the
  tree is genuinely clean before code edits begin.)
- **Why first:** clean tree before today's edits; keeps the backfill + NORTH_STAR regeneration as
  its own reviewable unit. (Confirmed: no unpushed commits exist; these are the only two dirty files.)
- **Verify:** `git status --short` empty; `git log origin/main..main --oneline` empty after push.
- **If it fails:** push failure = network/auth; fix and retry, do not proceed to code edits with a
  dirty tree.

### P2 — Register the scheduled task (**USER GATE 1 — do this BEFORE starting the autonomous run**, 2 min)
- **Do (user pastes with `!` prefix, exact text from `docs/RUNBOOK_nightly_refresh.md:42`):**
  ```
  schtasks /create /tn "BaseballNightlyRefresh" /sc daily /st 06:30 /tr "cmd /c cd /d C:\Users\hunte\projects\baseball && C:\Users\hunte\AppData\Local\Programs\Python\Python312\python.exe scripts\nightly_refresh.py" /f
  ```
- **Verify (CLAUDE):** `schtasks /query /tn "BaseballNightlyRefresh" /v /fo list` shows status
  Ready, Next Run Time 06:30 tomorrow.
- **Do NOT `/run` it yet** — the chain would die at the matchup rebuild exactly as on 8/11. The
  scheduler-fired proof run is P5, after the fix.
- **If it fails:** run from an elevated prompt; if still refused, this is the one item that slips —
  everything else proceeds, and "automation registered" moves to the top of tomorrow's list.

### P3 — Matchup cache fix (CLAUDE, 60–90 min total; kill criterion in §3)

**P3a. Validate the hypothesis on a copy — no live risk (15–25 min).**
- Copy `data/baseball.duckdb` (2.1 GB; 168 GB free) to the session scratchpad.
- On the **copy**: (1) reproduce the old path once (`_refresh_matchup_cache` against the copy) —
  expect the 0xC0000409 abort, for the record; (2) run the staging-swap form:
  `CREATE OR REPLACE TABLE matchup_summary AS SELECT <exact aggregation from queries.py> …` —
  single atomic statement, no DELETE, no PK/ART index to merge at COMMIT.
- **Verify:** step (2) exits 0 and on the copy
  `SUM(num_pitches) == COUNT(*) FROM pitches WHERE pitch_type IS NOT NULL` (= 8,161,237) with
  1,851,623 rows.
- **If the swap also aborts on the copy:** hypothesis refuted — try variant v2 (explicit staging
  table + `DROP` + `ALTER TABLE … RENAME`), then v3 (PK-less table + batched DELETE/INSERT).
  These count against the §3 attempt budget.

**P3b. Implement (30–40 min). Files:**
- `src/db/queries.py::refresh_matchup_cache` → staging-swap implementation (keep the ROUNDed
  SELECT this version already has — it is the dashboard-facing one); return the row count, not a
  1.85M-row DataFrame; adjust the `len(df)` call at `scripts/daily_refresh.py:163`.
- `src/ingest/daily_etl.py::_refresh_matchup_cache` → delegate to the queries.py implementation
  (one implementation, two call sites), keep the CatalogException skip-if-missing behavior.
- `src/db/schema.py::_create_matchup_summary` → **drop the `PRIMARY KEY(pitcher_id, batter_id,
  pitch_type)`** with a comment: uniqueness is by GROUP BY construction; no consumer upserts into
  this table (verified — no `ON CONFLICT` anywhere); the PK's ART index is what the abort dies in.
  (Flagged as decision D5, recommend yes.)
- `scripts/nightly_refresh.py::verify_daily_refresh` (+`read_watermark`) → effect-check now
  requires **watermark not regressed AND cache sync**:
  `SUM(matchup_summary.num_pitches) == COUNT(pitches WHERE pitch_type IS NOT NULL)`. This is the
  change that makes a recurrence *visible*: a mid-ETL death can never again classify `ok_verified`.
- `src/ingest/statcast_loader.py:1091` and `:1141` → `max_game_date=None` (kills the
  ConversionException noise honestly — NULL where no date exists, per hard rules).
- New `tests/test_matchup_refresh.py` → temp-DB unit test: rebuild produces sync, is idempotent,
  tolerates a missing table.
- **Verify:** `python -m pytest tests/test_matchup_refresh.py --no-cov -q` green.

**P3c. Live rebuild through the fixed path (5 min; writer window).**
- **Do:** one-off invocation of `refresh_matchup_cache` via `src.db.schema.get_connection()`
  (dashboard closed; nothing else writing).
- **Verify:**
  ```
  python -c "import duckdb; c=duckdb.connect('data/baseball.duckdb', read_only=True); m=c.execute('SELECT SUM(num_pitches) FROM matchup_summary').fetchone()[0]; e=c.execute('SELECT COUNT(*) FROM pitches WHERE pitch_type IS NOT NULL').fetchone()[0]; print(m, e, 'SYNC' if m==e else 'STALE')"
  ```
  Must print `SYNC` (8,161,237 both sides today) and the invoking process must exit 0. The copy
  test was the smoke; **this is the gate.**
- **If it fails:** consumes an attempt from the §3 budget.

### P4 — Full nightly chain, manual (CLAUDE, 10–15 min; writer window)
- **Do:** `python scripts/nightly_refresh.py`
- This is the day's integration gate: clears the stale leaderboards (last computed 8/11 20:43),
  and — for the first time since the abort began — runs roster sync, transaction sync, and the
  pregame report to completion. ETL step 1 will skip (pitches already fresh through 8/15 —
  correct). `hit_parlay` may warn pre-lineups (non-fatal by design); `resolve_picks` appends
  resolutions for the 8/11 picks (normal, append-only).
- **Verify:** in `logs/nightly/2026-08-16/status.json`: `daily_refresh` step has `returncode: 0`
  and `status: "ok"` (rc=0 is the proof the abort is gone — `ok_verified` is no longer good
  enough); `precompute` ok; overall `ok` or `ok_with_warnings` only from `hit_parlay`. Then:
  leaderboard freshness — `SELECT MAX(computed_at) FROM leaderboard_cache` is today; no
  `ConversionException` in `02_daily_refresh.log`; the P3c SYNC one-liner still prints `SYNC`.
- **If it fails:** `daily_refresh` dying again = §3 budget; `precompute` failure = read
  `03_precompute.log`, frozen artifacts are guarded by the post-verify step regardless;
  `verify_artifacts_post` fail = STOP EVERYTHING and diff `models/registry.json` state before any
  further step.

### P5 — Scheduler-fired proof run (**USER GATE 2 — MOVED TO AFTER P8**, 15 min) — D4 = yes

> **Resequenced for autonomous execution.** This step no longer sits mid-plan. The autonomous run
> performs the log-archive substep at the very end of P8, then STOPS and hands off. The user fires
> the one command; Claude then runs the verification as the closing action of the day.
- **Do (CLAUDE first):** archive the manual run's evidence:
  `Move-Item logs/nightly/2026-08-16 logs/nightly/2026-08-16_manual` (the wrapper recreates the
  dated dir; step logs are opened `"w"` and would otherwise be overwritten).
- **Do (USER):** `!schtasks /run /tn "BaseballNightlyRefresh"`
- **Verify (CLAUDE):** fresh `logs/nightly/2026-08-16/status.json` with `overall_status` ok/
  ok_with_warnings(hit_parlay only); `schtasks /query` shows Last Run Time = today, Last Result 0.
  This settles "automation works" *today* instead of hoping about tomorrow 06:30.
- **If it fails:** task context problem (working dir / user) — fix the `/tr` string per runbook and
  re-register; the manual-run evidence from P4 still stands.

### P6 — Small fixes + documentation truth-sync (CLAUDE, 45 min; no DB writes — can start while P4/P5 run)
1. **validate-model roster** (`.claude/skills/validate-model/SKILL.md`): remove retracted VWR from
   the `all`/`flagships` bulk roster (line 22), the Step-7 example roster line (~338), and the
   frontmatter "four flagships" phrasing → the three active flagships (`causal_war`, `pitchgpt`,
   `defensive_pressing`). VWR stays individually invocable (§4d untouched) — history is not
   deleted, it just leaves the flagship bulk run.
   **Verify:** `grep -n viscoelastic .claude/skills/validate-model/SKILL.md` shows it only in
   individual dispatch + §4d, not in any bulk-roster line.
2. **Runbook** (`docs/RUNBOOK_nightly_refresh.md`): "What it runs" table 3 → 6 steps (verify pre,
   daily, precompute, hit_parlay, resolve_picks, verify post); addendum "no task registered" →
   registered 2026-08-16 (after P2 actually succeeds, not before); rewrite the exit-127 caveat
   (line ~124): the observed code is 0xC0000409 fast-fail and for the matchup rebuild the COMMIT
   did **not** land — the caveat's "COMMIT already landed" claim was false for this step; document
   the staging-swap fix and the new cache-sync effect-check.
   **Verify:** re-read; no remaining "3 steps" / "no task registered" / unconditional
   "COMMIT already landed" text.
3. **Remediation report** (`docs/audits/REMEDIATION_EXECUTION_2026-08-11.md`): append a dated
   "Addendum 2026-08-16" under §5 recording what closed today (task registered; matchup abort
   root-caused + fixed; VWR roster fixed; coverage convention decided; lockbox adjudication if
   decided; ECE deferral if decided). Append, don't rewrite the table — it's an audit record.
   **Verify:** `git diff` shows additions only in that file.

### P7 — Implement the user's D1–D3 decisions (CLAUDE, 20 min; user can decide these while P3 runs)
- D1 coverage: edit `pyproject.toml` per decision.
  **Verify:** `python -m pytest -q -m "not slow"` (or full suite per convention) exits **0**.
- D3 lockbox: append a dated deviations-log entry (the PITCHGPT_V2_SPEC §9 log — same home as
  entry 1, the freeze SHA) recording the adjudication. Never touch `docs/holdout_ledger.jsonl`
  by hand.
- D2 ECE: record the decision (spend or defer) in the P6.3 addendum; if "spend", that work is
  explicitly **not today** (needs a written dated amendment first — see §5).

### P8 — Regression gate + final commit (CLAUDE, 30 min)
- **Do:** full `python -m pytest -q` under the new convention (expect ≥1318 passed + the new
  matchup tests, exit 0); `python -m pytest tests/test_claims_drift_guard.py --no-cov -q` (30/30);
  `python scripts/verify_artifacts.py` (ok=24, only the two *advisory* in-season drift lines);
  the SYNC one-liner. If any parallel agent batches were used during execution (3+), run the
  `validation-agent` skill — non-negotiable per standing rule.
- **Ledger integrity check before committing:** `git diff predictions/ docs/holdout_ledger.jsonl`
  must show **additions at end-of-file only** (resolve_picks appends from P4/P5). Any modified or
  deleted line in a ledger = do not commit, investigate.
- **Do:** commit in two logical units — (1) matchup fix + nightly effect-check + freshness fix +
  tests; (2) docs + roster + coverage + adjudication records — then `git push` (standing rule:
  always push after commit).
- **Verify:** `git status --short` empty; `git log origin/main..main` empty.

---

## 3. Kill criterion — matchup cache work (pre-registered, per project rule)

**Budget: 3 live-path attempts OR 90 minutes wall-clock on P3 (whichever hits first), attempts
being materially distinct variants** (v1 `CREATE OR REPLACE TABLE … AS SELECT` atomic swap; v2
explicit staging table + DROP + RENAME; v3 PK-less table + batched DELETE/INSERT). Success =
P3c's gate: live `SUM(num_pitches)` equals the eligible-pitch count AND the invoking process
exits 0, then re-confirmed inside the P4 chain run with `daily_refresh` rc=0.

**If the budget is exhausted without a committed in-sync cache: STOP.**
1. `git checkout` the rebuild-implementation edits (revert to committed behavior) — **except** the
   `verify_daily_refresh` cache-sync assertion, which lands regardless: if the cache must stay
   stale, the nightly must *say so* (`fail`, not `ok_verified`). Visible staleness is the honest
   fallback; silent staleness is what got us here.
2. Gate the in-ETL rebuild behind an env flag defaulting to skip-with-loud-WARN so the chain can
   complete end-to-end (leaderboards must still refresh tonight).
3. File it: P6.3 addendum entry + runbook caveat ("matchup_summary stale as of 2026-08-16,
   rebuild disabled pending DuckDB upgrade test / incremental-refresh design"), and add a
   "DuckDB version bump + retest" item to the backlog.
4. Proceed to P4 — the rest of the day does not burn with it.

Smokes are not gates: a green run on the DB **copy** (P3a) proves nothing about the live gate and
does not extend the budget.

---

## 4. Decision points — ALL DECIDED 2026-08-16

**Status: closed.** The user adjudicated D1–D5 on 2026-08-16; every recommendation below was
accepted as written. The table is retained as the rationale of record — it is not an open
question list, and the autonomous run must not reopen it. See §0 for the decision summary.

| # | Decision | Recommendation → **ACCEPTED** | Tradeoff in one line |
|---|---|---|---|
| D1 | Coverage gate: `fail_under = 90` vs ~54% actual makes every green suite exit non-zero | **Ratchet: set `fail_under = 50` now** (below current floor, so exit code means "tests passed and coverage didn't regress"); revisit upward later. Alternative: drop `--cov` from default addopts (faster runs, coverage becomes opt-in) | Ratchet keeps coverage regression-guarded on every run; removing `--cov` is faster but makes coverage invisible until someone asks |
| D2 | Production-path ECE: dated amendment + spend 1 of the **2 remaining** 2025 holdout contacts | **DEFER to season end.** Record the deferral today; bundle the contact with the K4-era season-end work when the question it answers is sharpest | Spending now puts a production calibration number on the board; deferring preserves half the remaining pre-registered budget for decisions that may matter more |
| D3 | v3 lockbox contact: does the Stage-B fit kill forfeit the one K5-granted contact? | **Adjudicate FORFEIT today** (5-min deviations-log append; spends nothing). A killed candidate's evaluation rights end with the kill — that's the plain reading of the pre-registration culture | Leaving it open preserves optionality for a v3 successor, but ambiguity is how holdout budgets creep; an explicit forfeit is the conservative, self-binding read |
| D4 | Fire `schtasks /run` once today after the manual chain (P5)? | **Yes** — it is the only way to prove scheduler-context invocation *today* rather than discovering a broken `/tr` string at 06:30 tomorrow | Costs ~10 min runtime + one archived log dir; skipping means automation stays unproven until tomorrow |
| D5 | Drop the `PRIMARY KEY` from `matchup_summary` DDL (part of P3b) | **Yes** — uniqueness is guaranteed by GROUP BY construction, nothing upserts into it, and the PK's ART index is the crash site; a 1.85M-row cache table needs no index for dashboard reads | Keeping the PK preserves declared uniqueness but keeps the crash mechanism in play on every variant that must rebuild the index |

~~D1–D3 can be decided while P3 runs; D4 is needed before P5; D5 before P3b.~~
**Superseded 2026-08-16:** all five decided up front (§0) precisely so the run never blocks.

---

## 5. NOT TODAY (deliberate deferrals)

| Item | Why not today |
|---|---|
| WS6 (Checking-Our-Work page, versioned write-ups, WAR-disagreement board, uncertainty-native UX, annual self-review) | Large designed build; content sessions of its own — cramming it into a cleanup day produces exactly the shallow work the audit culture exists to prevent |
| K4 season-end prep (pitcher-Marcel pin per resolution spec §6.7, season-end sprint_speed refresh) | Trigger is season end, not today; doing it early risks pinning against an incomplete season |
| ECE amendment + holdout contact spend | Per D2 recommendation; and even if the user decides "spend", the dated amendment must be *written and reviewed* before any contact — that is not a same-day tack-on |
| `adjusted_war_v3` validation spec | Pre-registration-first task with its own gates; explicitly out of scope per the validate-model banner |
| WS0.4 two-dated-dirs acceptance | Physically requires two real scheduled mornings; check on 2026-08-18 |
| `transactions.transaction_date` garbage (`MAX = 2925-11-26`) | New finding, needs a source-data audit; harmless to today's goals; never fabricate a "fix" for bad source dates |
| Umpire / game_weather / tj_surgery refresh cadence (stale to 8/8–8/9; not in the nightly chain) | Separate loaders, separate decision — and umpire work is gated by the standing ABS-era drift-check rule anyway |
| Incremental matchup refresh (delta-merge instead of full rebuild) | Only relevant if the staging swap fails; if it does, it gets *filed* (kill criterion step 3), not built at 6 PM |

---

## 6. Risks — and the crown jewels

**Crown jewels: `predictions/picks.jsonl`, `predictions/resolutions.jsonl`,
`docs/holdout_ledger.jsonl` (APPEND-ONLY — never rewritten, never "cleaned up"), the frozen
artifacts in `models/registry.json`, and `docs/claims/claims.yaml`.**

1. **Frozen artifacts.** P4/P5 retrain only the gitignored `*_2026_inseason.pkl` siblings; the
   frozen `stuff_model.pkl` / `xout_v1.pkl` / pitchgpt checkpoints are never written by any
   production path. Guard: `verify_artifacts` pre+post inside every chain run + the P8 final run
   must show the same 24 OK with only the two advisory in-season drift lines. Never pass
   `--allow-checkpoint-overwrite` / `allow_frozen_overwrite=True` today, for anything.
2. **Ledgers.** Only `hit_parlay_today.py` / `resolve_picks.py` may touch `predictions/*.jsonl`,
   and only by appending. P8's diff review is the gate: any non-append change blocks the commit.
   The D3 adjudication is a *deviations-log* append — it does not touch `holdout_ledger.jsonl`.
   No `sed -i`/`jq` over any ledger, ever.
3. **Claims registry.** No `claims.yaml` edits are planned today. Doc edits in P6 could break a
   claim citation; `tests/test_claims_drift_guard.py` at P8 is the guard — if it fails, fix the
   doc, never the registry entry.
4. **DB safety.** Single writer: P3c → P4 → P5 strictly serialized, dashboard closed throughout.
   The P3a scratchpad copy (2.1 GB) doubles as a same-day restore point — keep it until the Final
   Checklist is green, then delete. A crash mid-swap is transactional in DuckDB (old table
   survives; WAL rolls back on next open) — after any crash, re-run the SYNC probe before deciding
   anything.
5. **Chain-level.** Tomorrow 06:30 fires with whatever state we leave: if the dashboard is left
   open overnight the run refuses (exit 2) by design — that is honesty, not breakage. The P5
   re-run overwrites the dated log dir, hence the archive-first step. `hit_parlay` at 06:30
   pre-lineups warns — known, non-fatal.
6. **Scope creep.** The matchup work has a pre-registered budget (§3). The moment it's exhausted,
   the fallback is *visible staleness + a filed issue*, and the rest of the plan proceeds. The
   day's win condition is the §1 end state, not a perfect cache.

---

## 7. Final checklist (all must pass)

| # | Command | Pass condition |
|---|---|---|
| 1 | `schtasks /query /tn "BaseballNightlyRefresh" /v /fo list` | exists; Last Result 0 (today); Next Run 06:30 |
| 2 | SYNC one-liner (P3c) | prints `SYNC` |
| 3 | `logs/nightly/2026-08-16/status.json` | `daily_refresh` rc=0 status ok; overall ok / ok_with_warnings(hit_parlay only) |
| 4 | `SELECT MAX(computed_at) FROM leaderboard_cache` (read-only) | today's date |
| 5 | `python scripts/verify_artifacts.py` | `ok=24 warn=0 fail=0`, exit 0 |
| 6 | `python -m pytest -q` | exit 0 under the decided convention |
| 7 | `python -m pytest tests/test_claims_drift_guard.py --no-cov -q` | 30/30 pass |
| 8 | `git diff predictions/ docs/holdout_ledger.jsonl` (pre-commit) | appends only |
| 9 | `git status --short` + `git log origin/main..main` | both empty (committed AND pushed) |

D1–D3 decisions recorded in writing (P6.3 addendum + deviations log) — a decision that isn't
written down didn't happen.
