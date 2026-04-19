---
allowed-tools: Bash(git status:*), Bash(git diff:*), Bash(git log:*), Bash(git add:*), Bash(git commit:*), Bash(git rev-parse:*)
description: Commit staged changes with project-style message (no Claude co-author)
argument-hint: [optional message hint]
---

## Context

- Status: !`git status`
- Staged diff: !`git diff --staged`
- Unstaged diff: !`git diff`
- Recent commits (style reference): !`git log -5 --oneline`
- Message hint from user: $ARGUMENTS

## Your task

Create a single git commit following this project's conventions.

### Message style (match recent commits)
- Short imperative subject line, punchy and technical
- Em-dash, colon, and lowercase are fine (e.g. "PitchGPT 10K ablation: gap widened 7.8x, context alive but bloated")
- Focus on WHY not WHAT — the diff shows the what
- 1–2 sentences total; no body unless the change truly needs it
- If `$ARGUMENTS` is non-empty, treat it as a hint for the subject — do not quote it verbatim unless it already reads well

### Staging rules
- Stage ONLY files relevant to this logical change — inspect the diff first
- NEVER run `git add -A` or `git add .` (risk of committing secrets or large binaries)
- Refuse to stage files matching `.env*`, `*credentials*`, `*.key`, `*.pem`, or anything >10MB unless the user explicitly confirms
- If secret-looking files are already staged, warn the user and stop

### Commit rules
- Pass the message via HEREDOC for correct multi-line formatting
- Do NOT include a `Co-Authored-By: Claude` trailer — this project's history does not use one
- Do NOT use `--no-verify`; if a pre-commit hook fails, fix the underlying issue and create a NEW commit (never `--amend`)

### Commit template

```bash
git commit -m "$(cat <<'EOF'
<subject line>

<optional 1-sentence body>
EOF
)"
```

### After committing
- Run `git status` to confirm a clean tree (or note remaining unstaged work)
- Print one line: `committed as <short SHA>` using `git rev-parse --short HEAD`
