---
name: validation-agent
description: Post-batch regression check. Runs pytest, import spot-checks modules touched in the session, git-diff sanity, and reports pass/fail + any regressions. Use after 3+ parallel agent batches. [trigger keywords: "validate this", "run validation agent", "check for regressions", after any multi-agent batch]
---

# validation-agent

You are executing a standardized post-batch regression check after a multi-agent workload (3+ parallel agents modifying code/config). This skill is non-negotiable per user memory: always run after multi-agent batches to catch regressions. **Report only — never modify files.** Work from repo root `C:/Users/hunte/projects/baseball`. Keep total runtime under ~2 min.

## When to invoke

- User says: "validate this", "run validation agent", "check for regressions", "post-batch check".
- Automatically after any session where 3+ parallel agents modified `src/`, `tests/`, `scripts/`, or `configs/`.
- Before reporting a multi-agent batch as complete.

Do NOT invoke for single-agent edits or doc-only changes (`docs/`, `results/` markdown only).

## Step 1 — Scope detection

Identify what changed this session:
- `git status` (no `-uall`) — list modified + untracked.
- `git diff --stat HEAD` — per-file line counts.
- Cross-reference against the session transcript for modules the agents touched.

Classify each changed path into buckets:
- **code**: `src/**`, `scripts/**`, `tests/**`, `configs/**`
- **docs-only**: `docs/**/*.md`, `results/**` (non-code)
- **data**: `data/**`, `models/**` (binary/parquet/pkl — do not diff)

If **only docs-only** changed: skip Steps 3 and 4; jump to Step 5 diff sanity and report PASS-WITH-NOTES ("docs-only batch, test run skipped").

## Step 2 — Test collection sanity

From repo root:
```
pytest tests/ --collect-only -q
```
Record total tests collected. **FAIL immediately** if collection errors (import failures, syntax errors in test files). Show the first error and stop — nothing else matters until collection is clean.

## Step 3 — Full test run (conditional)

Trigger if any of these changed: `src/analytics/**`, `src/db/**`, `src/dashboard/views/**`, `tests/**`.

```
pytest tests/ -x --tb=short
```

`-x` stops at first fail; `--tb=short` keeps traceback scannable. Record passed/failed/skipped counts. If anything fails, capture the first 20 lines of the failing output verbatim.

## Step 4 — Import spot-check

For each changed module under `src/**` (convert path to dotted module, e.g. `src/analytics/causal_war.py` → `src.analytics.causal_war`):

```
python -c "import src.analytics.causal_war"
```

Run sequentially, one per touched module. Record any `ImportError`, `SyntaxError`, `ModuleNotFoundError`. This catches modules the test suite does not exercise (common blind spot for dashboard views, new scripts).

Skip `__init__.py`-only changes and non-Python files.

## Step 5 — Git diff sanity

Run `git diff --stat HEAD` and scan for red flags:
- **Unexpectedly large diffs** — any single file > 500 lines changed without clear reason.
- **Deleted tests** — any line in `tests/` with `-` prefix on a `def test_` line.
- **New `TODO` / `FIXME` / `XXX`** introduced this batch (`git diff HEAD | grep -E '^\+.*(TODO|FIXME|XXX)'`).
- **Sensitive files staged** — `.env`, `credentials.json`, `*.key`, `*secret*`, any `*.pkl`/`*.pt` over 100MB that is not already tracked.
- **Accidental commits of `data/**` binaries** outside expected parquet refreshes.

Record each flag as a note (not an automatic FAIL unless sensitive files are staged — that IS a FAIL).

## Step 6 — Report

Output a single table and verdict. Use plain ASCII (no emoji).

```
validation-agent report — <ISO timestamp>

| Check                  | Status      | Detail                                  |
|------------------------|-------------|-----------------------------------------|
| Scope detection        | OK          | 7 files changed (5 code, 2 docs)        |
| Test collection        | PASS        | 142 tests collected                     |
| Full test run          | PASS        | 142 passed, 0 failed, 3 skipped         |
| Import spot-check      | PASS        | 5/5 modules imported cleanly            |
| Git diff sanity        | PASS-NOTES  | 1 new TODO in src/analytics/chemnet.py  |

Overall: PASS / PASS-WITH-NOTES / FAIL
Regressions: <list gate names, or "none">
```

If **FAIL**: show first 20 lines of the failing step's error output and a recommended fix (e.g., "ImportError in src.analytics.foo — missing `import numpy as np` on line 12; add the import and re-run").

## Constraints

- **Read-only.** Never modify files, never stage, never commit.
- **Windows bash.** Forward slashes only. Scratch files go to `C:/Users/hunte/AppData/Local/Temp/` (do not write to repo).
- **No long-running commands.** Do not invoke training scripts, `scripts/pitchgpt_ablation.py`, or any `validate-model` flow — those are separate skills.
- **No network.** No `pip install`, no data downloads.
- **Budget ~2 min total.** If `pytest tests/` runs long, note it and proceed; do not kill the run.

## Example triggers

- "I just finished a 4-agent refactor of `src/analytics/*` — run validation agent."
- "Check for regressions after the batch."
- After the assistant completes a 3-way parallel edit, proactively invoke before declaring done.
