# Baseball Analytics Platform

## Fresh-session protocol
Rely on this file + `MEMORY.md` only. Do **not** auto-read `docs/NORTH_STAR.md` (~25KB) — on demand (strategy, current phase, what's next, `/sitrep`), fetch `docs/NORTH_STAR_CURRENT.md` first (current-state snapshot, claims-registry-governed); `docs/NORTH_STAR.md` is the full history.

## Flagships
- Active (3): **DPI, AdjustedWAR (formerly CausalWAR), PitchGPT**. Stats/caveats in `project_flagships.md` memory.
- Retracted / retired — do not re-litigate: VWR, MechanixAE (descriptive only), ChemNet v1+v2, volatility_surface, Allostatic Load.

## Hard rules
- **Data:** DuckDB single-writer. Open via `src/db/schema.py::get_connection`; readers pass `read_only=True`. Stop dashboard before any backfill/retrain. Never fabricate — leave NULL where source data is missing, and backfill before iterating on architecture.
- **Dashboard:** Views go in `src/dashboard/views/`. Never create `src/dashboard/pages/` (Streamlit auto-discovery breaks routing).
- **Validation:** Specs/results at `docs/models/<model>_validation_spec.md` and `_results.md`. Never invent thresholds — read the spec.
- **Regression:** After 3+ parallel agent batches, run the `validation-agent` skill. Non-negotiable.
- **Tools:** Prefer Grep / Read / Glob over `grep` / `head` / `cat` / `find` via Bash.
- **Env:** Windows 11 + bash. Forward slashes; scratch files to `C:/Users/hunte/AppData/Local/Temp/`.

## Project commands
- `/sitrep` — 5-step orientation (NORTH_STAR + git + results + procs + summary) when resuming after a break.
- `/validate-model <name>` — flagship validation playbook. Names: `causal_war`, `pitchgpt`, `defensive_pressing`, `mechanix_ae`, `viscoelastic_workload`, `allostatic_load`, `all`, `flagships`.
- `/commit` — project-style commit message (no Claude co-author).

## Module map
Analytics: `src/analytics/{stuff_model,causal_war,pitchgpt,mechanix_ae,viscoelastic_workload,defensive_pressing,allostatic_load}.py`. Tests: `pytest tests/`. Backfills: `scripts/backfill_*.py`. DB connection helper name varies (`_get_conn` vs `get_db_connection`) — grep before importing.
