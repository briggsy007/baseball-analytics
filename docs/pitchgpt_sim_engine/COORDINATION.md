# PitchGPT Sim Engine — Cross-Session Coordination

Purpose: keep concurrent Claude sessions aligned. Read this first. Append a session-close entry at the bottom before ending a session. If something here conflicts with what you're about to do, stop and surface it to the user.

## Phase status (authoritative snapshot)

| Phase | Status | Owner | Notes |
|---|---|---|---|
| 0.1 Sampling-fidelity vs LSTM | **FAIL** per §3.1 gate | Session B | PG won metrics 1/2/3, lost metric 4 (2-gram Frobenius). Claim narrows to "calibrated rollout engine, matches marginals." |
| 0.2 Outcome-head design | **COMPLETE** | Session A | Frozen backbone + 2-layer MLP (128→64→7). Joint route blew calibration budget 2.5×. |
| 0.3 Train outcome head | **COMPLETE 2026-04-26 — A1 ships** | Session A | Plan B closed. A1 (frozen v2 + concat-input 3-layer MLP head, 211→128→64→7) lifts +18.31% over freq prior (CI [+18.10%, +18.53%]); ECE post-T 0.0114; HBP log-loss 3.02 (first variant under PASS <4.0). A1−A3 paired delta +2.48pp (CI [+2.24, +2.72]). Backbone byte-identity verified. Verdict: WEAKER PASS (clears <2.5 in_play_hit, misses full <2.0). See `results/pitchgpt_sim/outcome_baselines_2026_04_25/SUMMARY.md`. |
| 0.4 Outcome-head OOS validation | Not started, unblocked | unassigned | A1's per-pitcher variance + per-class log-loss already measured during Step 2 (mean ll 1.346, var 0.0010, range [1.27, 1.40] across top-50 pitchers; see `a1_concat/report.md`). The full 0.4 ticket as scoped (per-class confusion + per-class reliability diagrams) remains open as a follow-up but is no longer blocking. |
| 0.5 Rollout harness `pitchgpt_sim.py` | Not started, unblocked | unassigned | Production OutcomePredictor: PGConcatHeadPredictor at `models/pitchgpt_v2_outcomehead_a1.pt`. See `PHASE_0.5_PLAN.md` (forthcoming). |
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
- `models/pitchgpt_v2.pt` — current flagship backbone. Paper reference. Outcome head must FREEZE it. SHA256 `6f952054…62883c` (verified unchanged through Plan B Step 2).
- `models/pitchgpt_v1_10k.pt` — matched-scale v1 retrain artifact.
- `models/pitch_lstm_10k.pt` — matched LSTM baseline for sampling-fidelity and future 0.4 baseline comparisons.
- `models/pitchgpt_v2_smoke.pt` — preserved smoke (pre-10K retrain) for reference.
- `models/pitchgpt_v2_outcomehead.pt` — Phase 0.3 frozen-MLP-on-hidden-state checkpoint (the −5.34% FAIL artifact). Preserved for replay; NOT production. SHA256 `6b47f97d…cbb54a0`.
- `models/pitchgpt_v2_outcomehead_a1.pt` — **Plan B winner (2026-04-26).** Backs `PGConcatHeadPredictor`. 3-layer MLP head 211→128→64→7 over concat(hidden[128] + context[35] + pitch_type_oh[17] + zone_oh[26] + velo_oh[5]). Calibration T=0.8003. Do NOT overwrite.

## Pending decisions

- 4/19 dashboard drift (14 files under `src/dashboard/views/`) — user-gated; not owned by either session.
- Whether to append a "scale-verify update" section to the Phase 0.2 design doc referencing the 10K v1/v2 numbers — would strengthen the "no cheap LSTM rescue" decision rationale.
- Phase 0.5 implementation (build `src/analytics/pitchgpt_sim.py`) — handed off to next session per `docs/pitchgpt_sim_engine/PHASE_0.5_PLAN.md`.

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

## Plan B verdict (2026-04-26) — A1 ships

Plan B was the recovery plan for Phase 0.3's −5.34% FAIL. Step 1 ran three baselines (A3 XGBoost, A4 logistic, A5 empirical) on the same 7-class outcome target; Step 2 added A1 (frozen v2 backbone + concat-input 3-layer MLP head). All evaluated on 2025 pitcher-disjoint holdout (~204K rows).

| variant | log-loss | lift vs prior | 95% CI on lift | ECE post-T | hit ll | hbp ll | verdict |
|---|---:|---:|---:|---:|---:|---:|:--:|
| **A1 frozen+concat** | **1.3507** | **+18.31%** | **[+18.10%, +18.53%]** | **0.0114** | 2.34 | **3.02** | **WEAKER PASS — SHIP** |
| A4 logistic | 1.3650 | +17.35% | [+17.14%, +17.57%] | 0.0264 | 2.37 | 4.91 | WEAKER PASS |
| A3 XGBoost | 1.3853 | +16.12% | [+15.87%, +16.37%] | 0.0181 | 2.31 | 3.57 | WEAKER PASS |
| A5 empirical | 1.5800 | +4.33% | [+4.24%, +4.44%] | 0.0015 | 2.86 | 5.54 | FAIL (kill criterion check) |

A1 − A3 paired bootstrap (204,513 rows): **+2.48 pp** lift delta (CI [+2.24, +2.72]) — clears the +1pp ship-A1 threshold by ~2×. CI excludes zero by ~22 SE. The PG v2 backbone DOES carry outcome-discriminative information; Phase 0.3's failure was head-capacity (128→64→7 over hidden state alone) not backbone information.

A1 is the FIRST variant to clear the HBP <4.0 PASS threshold (3.02). Misses full PASS only on `in_play_hit` (2.34, threshold 2.0; clears WEAKER threshold <2.5). The `in_play_hit` ceiling is structural — it depends on launch_speed/launch_angle, post-pitch features no architecture in this study has access to. Tier-A consumers inherit and must disclose.

Phase 0.3 narrative correction: the −5.34% FAIL was head-capacity, not backbone information. Same frozen v2 + larger-head + concatenated context features → +18.31% lift. The prior diagnosis was wrong.

Backbone integrity: `models/pitchgpt_v2.pt` SHA256 verified unchanged through both runs.

Artifacts:
- `results/pitchgpt_sim/outcome_baselines_2026_04_25/SUMMARY.md` — full Plan B summary
- `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/{metrics.json, report.md, train.log}`
- `models/pitchgpt_v2_outcomehead_a1.pt` — production OutcomePredictor checkpoint

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

### 2026-04-26 — Session close (Plan B)
- Launched Plan B Step 1 (3 baselines): A3 XGBoost, A4 logistic, A5 empirical. All landed; A4 led Step 1.
- Launched Plan C: SIM_ENGINE_API.md formal spec drafted from RESEARCH_PLAN_sim_engine_api.md.
- Validation agent run after multi-agent batch — clean.
- Launched Plan B Step 2: A1 (frozen v2 + concat-input 3-layer MLP head). Beat A3 by +2.48pp paired (CI [+2.24, +2.72]) — A1 ships.
- Phase 0.3 reopened and resolved — narrative correction logged in EXECUTION_PLAN §6.0.3.
- New checkpoint: `models/pitchgpt_v2_outcomehead_a1.pt`. v2.pt byte-identity verified.
- Doc updates locked in this session: COORDINATION (Plan B verdict + checkpoint registry + session log), EXECUTION_PLAN §6.0.3/0.4/0.5/0.6 + §11, SIM_ENGINE_API (PGConcatHeadPredictor as 4th impl, PGFrozenHeadPredictor deprecated), NORTH_STAR (Plan B closes section appended).
- **For next session:** Phase 0.5 — implement `src/analytics/pitchgpt_sim.py` per `SIM_ENGINE_API.md` and the new `PHASE_0.5_PLAN.md`. Production OutcomePredictor: PGConcatHeadPredictor. Do NOT overwrite v2.pt or v2_outcomehead_a1.pt.
