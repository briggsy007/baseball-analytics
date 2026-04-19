---
description: Reconstruct project state after a break and report a concise sitrep
---

You are running the `/sitrep` orientation flow. Tell the user: "Running sitrep — reconstructing project state." Then execute the 5 steps below in order, briefly, and finish with a one-paragraph summary.

## Step 1 — North Star (source of truth)
Read `docs/NORTH_STAR.md`. Note the current phase/priority and any "what's next" markers.

## Step 2 — Git state
Run these in parallel:
- `git log --oneline -10`
- `git status`

Identify the last commit's intent and whether the working tree has meaningful uncommitted work (vs routine artifact churn).

## Step 3 — Latest validation runs
List the five newest validation output dirs:
```bash
ls -1td results/validate_*/ 2>/dev/null | head -5
```
For the newest 1-2, peek at their `summary.json` or top-level result file to note verdict (pass/fail) and model.

## Step 4 — Running Python processes
Check for in-flight work (Windows):
```bash
tasklist //FI "IMAGENAME eq python.exe" 2>/dev/null | head -20
wmic process where "name='python.exe'" get commandline,processid 2>/dev/null | head -20
```
If either returns commandlines, note what script is running. If nothing, say "no python running."

## Step 5 — Report
Write ONE paragraph (4-7 sentences) covering:
- **What's live:** any python processes currently running
- **What finished:** most recent commit + most recent validation verdict
- **What's next:** the next action per NORTH_STAR.md

Keep it tight. No bullet lists, no headers in the final summary — just the paragraph so the user can re-enter flow fast.
