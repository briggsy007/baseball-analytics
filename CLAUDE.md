# Baseball Analytics Platform — Claude Context

## Source of truth
- `docs/NORTH_STAR.md` is the current-state doc — read first every session. Current strategy: edge surfacing over gate completion (Path 2, 2026-04-18 reset).

## Flagships
- Flagships (post-evidence consolidation 2026-04-18 late evening): **DPI, CausalWAR, PitchGPT** — see NORTH_STAR "Post-evidence consolidation" section for claim details and caveats.
- Retracted / retired / demoted — **do not re-litigate**: VWR (retracted, small-sample artifact), MechanixAE (demoted to descriptive profiling), ChemNet v1+v2 (retired null), volatility_surface (retired null), Allostatic Load (non-flagship, 1/4 gates pass).

## Project Claude commands & skills
- `/validate-model <name>` (`.claude/skills/validate-model/`) — runs the validation playbook for a flagship. Accepts `causal_war`, `pitchgpt`, `defensive_pressing`, `mechanix_ae`, `viscoelastic_workload`, `allostatic_load`, or `all` / `flagships`.
- `/commit` (`.claude/commands/commit.md`) — project-style commit message, no Claude co-author.
- `/sitrep` (`.claude/commands/sitrep.md`) — 5-step orientation (NORTH_STAR + git + results + procs + summary) when resuming after a break.
- `validation-agent` skill (`.claude/skills/validation-agent/`) — post-batch regression check; triggered automatically after 3+ parallel agent batches.

## Testing & validation
- Test command: `pytest tests/` (configured via `[tool.pytest.ini_options]` in `pyproject.toml`) — runs ~35 test files.
- Model validation: use `/validate-model <name>` skill. Specs at `docs/models/<model>_validation_spec.md`; results at `docs/models/<model>_results.md`. Never invent thresholds — read the spec.
- After any 3+ parallel agent batch, launch a validation agent (pytest + import spot-check + regression report). Non-negotiable.

## Project structure
- Analytics modules in `src/analytics/`: `stuff_model.py`, `causal_war.py`, `pitchgpt.py`, `mechanix_ae.py`, `viscoelastic_workload.py`, `defensive_pressing.py`, `allostatic_load.py`.
- Dashboard in `src/dashboard/` using radio-button nav via `views/`. Never create `src/dashboard/pages/` — Streamlit would auto-discover and break routing.
- Scripts in `scripts/`; backfill scripts match `scripts/backfill_*.py`.

## Data rules
- DuckDB is single-writer. Canonical opener: `src/db/schema.py::get_connection(read_only=False)`; dashboard reads via `src/dashboard/db_helper.py` with `read_only=True`. Parallel readers must pass `read_only=True`. Streamlit holds a write lock — stop the dashboard before any backfill/retrain.
- Always backfill NULL/missing data before iterating on architecture. Never fabricate — leave NULL where source data doesn't exist.

## Environment
- Windows 11 + bash. Use forward slashes; scratch files go to `C:/Users/hunte/AppData/Local/Temp/` (not `/tmp`).
- Torch installed with CUDA support (verify exact version with `pip show torch` when needed). GPU helps PitchGPT and MechanixAE; CausalWAR/VWR/DPI/ABL are CPU-only sklearn.

## Tool discipline
- Prefer Grep/Read/Glob tools over `grep`/`head`/`cat`/`find` via Bash — session logs show ~112 bash anti-pattern calls that the native tools would handle better and without prompts.
