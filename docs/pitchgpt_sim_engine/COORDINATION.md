# PitchGPT Sim Engine — Cross-Session Coordination

Purpose: keep concurrent Claude sessions aligned. Read this first. Append a session-close entry at the bottom before ending a session. If something here conflicts with what you're about to do, stop and surface it to the user.

## Phase status (authoritative snapshot)

| Phase | Status | Owner | Notes |
|---|---|---|---|
| 0.1 Sampling-fidelity vs LSTM | **FAIL** per §3.1 gate | Session B | PG won metrics 1/2/3, lost metric 4 (2-gram Frobenius). Claim narrows to "calibrated rollout engine, matches marginals." |
| 0.2 Outcome-head design | **COMPLETE** | Session A | Frozen backbone + 2-layer MLP (128→64→7). Joint route blew calibration budget 2.5×. |
| 0.3 Train outcome head | **In progress** 2026-04-24 | Session A (this session) | Background agent running. Frozen-backbone + 2-layer MLP on 2015–2022 pitcher-disjoint, HBP class-weighting. Will load current `models/pitchgpt_v2.pt` (10K-retrained) as frozen backbone; pre/post SHA256 assertion. Writing to `models/pitchgpt_v2_outcomehead.pt` (new). |
| 0.4 Outcome-head OOS validation | Not started | Session A (claimed) | Will fire automatically when 0.3 completes. |
| 0.5 Rollout harness `pitchgpt_sim.py` | Not started | unassigned | Depends on 0.3. |
| 0.6 Rollout sanity check | Not started | unassigned | Depends on 0.5. |

## Scale-verify results (2026-04-24)

Original 13.80% LSTM delta was 1K vs 1K. At matched 10K scale:
- v1 (context_dim=34, no ump): **+2.57%** (CI +1.68/+3.43)
- v2 (context_dim=35, with ump): **+3.13%** (CI +2.19/+4.05)
- Both FAIL 15% perplexity gate. Ump feature adds +0.56pp (within noise).
- Calibration survives: ECE post-temp 0.009 (v1) / 0.0075 (v2).
- VWR scale-verify lesson recurring. Do NOT re-promote on small-sample LSTM deltas.

## File ownership

| File | Owner | Rule |
|---|---|---|
| `src/analytics/pitchgpt.py` | shared | CONTEXT_DIM now parameterized (default=35). Backwards-compat with frozen-backbone outcome head. Don't break. |
| `src/analytics/pitchgpt_calibration.py` | shared | +8 lines context slicing by B; outcome head may extend. |
| `src/analytics/pitchgpt_outcome_head.py` | A | Stable; don't refactor without A's sign-off. |
| `scripts/pitchgpt_sampling_fidelity.py` | B | Done; don't modify. |
| `scripts/pitchgpt_outcome_head_smoke.py` | A | Stable. |
| `scripts/pitchgpt_2025_holdout.py` | B | Loader infers context_dim from checkpoint. Safe to extend. |
| `scripts/train_pitchgpt_v2_ump.py` | B | Added `--context-dim`, `--no-ump`, `--model-filename` flags. |
| `docs/pitchgpt_sim_engine/EXECUTION_PLAN.md` | A (primary) | B may append phase-status updates inline; do NOT rewrite sections. |
| `docs/pitchgpt_sim_engine/pa_outcome_head_design.md` | A | LOCKED 2026-04-24 per §3.1 decision log. |

## Checkpoints — LOCKED (do NOT overwrite)

- `models/pitchgpt_v1.pt` — legacy v1, committed Apr 9. Paper reference.
- `models/pitchgpt_v2.pt` — current flagship backbone. Paper reference. Outcome head must FREEZE it.
- `models/pitchgpt_v1_10k.pt` — matched-scale v1 retrain artifact.
- `models/pitch_lstm_10k.pt` — matched LSTM baseline for sampling-fidelity and future 0.4 baseline comparisons.
- `models/pitchgpt_v2_smoke.pt` — preserved smoke (pre-10K retrain) for reference.

New checkpoint path for Phase 0.3: `models/pitchgpt_v2_outcomehead.pt` (not yet created).

## Pending decisions

- 4-commit consolidation (Session B): P0.1 scale-verify / sim-engine scaffold / P1 edges / NORTH_STAR update. Awaiting user greenlight 2026-04-24.
- 4/19 dashboard drift (14 files under `src/dashboard/views/`) — user-gated; not owned by either session.
- Whether to append a "scale-verify update" section to the Phase 0.2 design doc referencing the 10K v1/v2 numbers — would strengthen the "no cheap LSTM rescue" decision rationale.

## Locked decisions (user-greenlit 2026-04-24)

- Phase 0.1 gate FAIL accepted. Sim-engine claim narrowed in EXECUTION_PLAN §3 to "calibrated rollout engine" (CI-backed) plus narrowed positive "matches empirical marginals with calibrated uncertainty." No claim of sequence-structure superiority over LSTM.
- A4 (Deceptiveness leaderboard) DEMOTED from Tier A to Tier C. Revival path: (a) regression of per-pitcher NLL on SwStr% outperforming a pitch-mix-entropy baseline, OR (b) longer-horizon sequencing study (H≥12) that restores a transformer 2-gram/3-gram win. Until then A4 does not ship.
- Phase 0.3 greenlit with frozen-backbone + 2-layer MLP route.

## Rules of engagement

1. Before touching a file in the ownership table, check git status for concurrent mods.
2. Do NOT re-litigate §3.1 decision-log items (metric-4-mandatory gate, 7-class outcome target, B3 seasonal-residual gate).
3. Do NOT overwrite LOCKED checkpoints. Save new artifacts under new names.
4. After a 3+ parallel agent batch, run a validation agent (pytest + import spot-check) before closing the session.
5. End every session with an entry appended to the log below (under ~10 lines).

## Session log (append only)

### 2026-04-24 — Session A close
- Drafted `EXECUTION_PLAN.md` (57KB) + `pa_outcome_head_design.md` (32KB).
- Phase 0.2 smoke: frozen MLP route chosen; joint blew ECE budget by 2.5×.
- Built `src/analytics/pitchgpt_outcome_head.py` (`JointOutcomeHead` + `FrozenOutcomeHead`).
- Smoke harness at `scripts/pitchgpt_outcome_head_smoke.py`.
- Smoke checkpoints: `models/pitchgpt_outcome_smoke_{joint,frozen}.pt`.
- Plan locked: 7 outcome classes, metric-4 mandatory for 0.1 PASS, B3 seasonal-residual gate.
- Left uncommitted for user review.

### 2026-04-24 — Session B close (in progress)
- Landed P0.1 scale-verify: v1@10K +2.57%, v2@10K +3.13% vs LSTM. Both FAIL 15% gate. Calibration clean.
- Built CONTEXT_DIM parameterization across `pitchgpt.py` / `pitchgpt_calibration.py` / `pitchgpt_2025_holdout.py` / `train_pitchgpt_v2_ump.py` (backwards-compat, default=35).
- Ran sampling-fidelity eval → updated `EXECUTION_PLAN.md` Phase 0.1 status to FAIL per §3.1.
- New artifacts: `pitchgpt_v1_10k.pt`, `pitch_lstm_10k.pt`, `pitchgpt_v2_smoke.pt` (preserved), sampling-fidelity results, v1/v2 holdout reports.
- Landed P1.1 weather edge doc + P1.3 TJ-into-projections wiring (unrelated to sim engine).
- Preparing 4-commit consolidation. Awaiting user greenlight.
- **For next session:** Phase 0.3 (train outcome head) is the critical-path unblocker. Use `pitchgpt_v2.pt` as frozen backbone; design locked in `pa_outcome_head_design.md`. Reference new `pitch_lstm_10k.pt` as matched baseline for Phase 0.4 comparisons.
