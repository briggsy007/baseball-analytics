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
| 0.6 Rollout sanity check | **CLOSED as FAIL 2026-08-10** — K%/BB%/HR% FAIL; wOBA + PA-length PASSes [TAINTED and now PERMANENTLY UNEARNED for v2-era PitchGPT: 0.6.2 KILLED 2026-08-10 at the fit-convergence gate, so the clean re-evaluation never ran] | — | Clean-provenance re-run (HEAD + pos-0 calibration). Static-context drift diagnosed as ~6pp of the +11.6pp K% bias. Kill record: §"2026-08-10 — Phase 0.6.2 executed" below + `docs/models/pitchgpt_phase062_results.md`. |
| 0.6.1 Mid-PA context mutation | **IMPLEMENTED 2026-08-04, pending sanity re-run** | this session | `count_state` one-hot re-emitted per position from the running (balls,strikes) via `_advance_count`; flows through BOTH the pitch-token backbone forward and the outcome-head forward (shared `cur_context`). Verified functionally correct (integration test). Added `PGConcatHeadPredictor(disable_class_calibration=…)` + `PITCHGPT_DISABLE_CLASS_CALIBRATION` env A/B switch (pos-0 recalibration always retained). Orchestrator to re-run `scripts/pitchgpt_rollout_sanity_2025.py` with/without the switch. |

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

### 2026-08-04 — Session close (Phase 0.6.1 mid-PA context mutation)
- **Finding:** the mid-PA `count_state` mutation was already committed (6111cd6) but its docstrings still said "context held constant" — which misled the prior sanity-run author (ee29462) into logging "mutation remains the required fix". Verified empirically (GPU probe) the mutation IS functionally correct: the outcome head receives count_state 0 at pos0 → deepening to 3-2 by pos5, and the pitch-token backbone forward shares the same `cur_context`.
- Refactored the count transition into a pure, unit-tested `_advance_count(balls, strikes, outcome_class)` helper (ball / CS / SS / foul-capped-at-2 / terminal-unchanged); rewired the rollout loop to call it. Behavior byte-equivalent (existing walk/K/in-play termination tests still pass).
- Fixed the 3 stale "context held constant" docstrings to describe the 0.6.1 mutation.
- Added `PGConcatHeadPredictor(disable_class_calibration=…)` (default `None` ⇒ reads `PITCHGPT_DISABLE_CLASS_CALIBRATION` env). Drops the all-position `class_calibration` (fit under the static-context bug) while KEEPING the pos-0 recalibration. Lets the orchestrator A/B the sanity run with zero script edits.
- **Empirical (single 0-0 PA, directional not gate):** per-position CS marginals stay ~0.28-0.33 calibrated even though context mutates — the model's count-conditioning is weak. class_calibration INFLATES PA-level K% (raw 0.269 → calibrated 0.402; target 0.218). So disabling class_calibration is a real lever, but it CRASHES BB% (0.114 → 0.033) — not a clean win alone.
- Tests: `tests/test_pitchgpt_sim.py` +19 (state machine, non-constant-context integration, disable flag + env). Full file 49 passed.
- **For orchestrator:** re-run `scripts/pitchgpt_rollout_sanity_2025.py` A/B (with and without `PITCHGPT_DISABLE_CLASS_CALIBRATION=1`). KILL CRITERION: if K%/BB%/HR% still FAIL, the ONE principled follow-up is refitting `class_calibration` on the mutated rollout (old fit is bug-contaminated). Beyond that, stop and keep the narrowed claim — no hack-tuning.

### 2026-08-04 — Phase 0.6.1 sanity A/B verdict

Post-mutation-verification A/B on the 2025 pitcher-disjoint cohort (10K PA x 100 samples, seed 42):

| variant | K% (emp 0.218) | BB% (emp 0.088) | HR% (emp 0.032) | wOBA | PA len | overall |
|---|---:|---:|---:|:--:|:--:|:--:|
| class_calibration ON (default) | 0.3339 FAIL | 0.1177 FAIL | 0.0242 FAIL | PASS | PASS | **FAIL** |
| class_calibration OFF (`PITCHGPT_DISABLE_CLASS_CALIBRATION=1`) | 0.3651 FAIL | 0.0376 FAIL | 0.0261 FAIL | FAIL | PASS | **FAIL (worse)** |

NOTE (2026-08-10, superseded same day — see the amendment immediately below): the wOBA /
PA-length PASS cells above are TAINTED - pending Phase 0.6.2 re-evaluation (pos-0 calibration
was fit on the eval cohort). Both A/B variants retained the pos-0 recalibration, so these
PASSes are partly bought with holdout-fitted weights and must be re-earned under the clean
2023-only fit (`PHASE_0.6.2_PLAN.md` §2, §5).

AMENDED NOTE (2026-08-10, Batch D / K5 consequence): "pending" is now false. **Phase 0.6.2 was
KILLED 2026-08-10 at the fit-convergence gate** (§6 first disjunct; iteration 1 = 4.418pp,
iteration 2 = 2.625pp vs the 1.0pp threshold, 2023 fit cohort), so the clean 2023-only
re-evaluation of these cells NEVER RAN and none is authorized under that protocol (2025 was
never read; holdout contact #13 unspent). **The wOBA / PA-length PASSes above are permanently
unearned for v2-era PitchGPT** and may not be quoted as PASS anywhere. Claim ids:
`pitchgpt_phase062_kill` (the kill), `pitchgpt_woba_pa_pass_pre062` (retracted PASSes).

Verdict: keep class_calibration ON as production default. Root cause of residual K% surplus is the
model's flat per-position strike response in the self-generated rollout regime (CS ~0.28-0.33 across
positions vs empirical collapse to 0.045 by pos 5) — the teacher-forced class_calibration amplifies
it. NOTE: class_calibration was fit teacher-forced on true 2023 val contexts; it is NOT contaminated
by the (already-fixed-in-6111cd6) static-context bug, correcting the 2026-08-04 morning assumption.

Next (designed follow-up, own kill criterion required before starting): refit class-marginal
re-weighting ON ROLLOUT OUTPUTS over the 2023 val cohort (per-position generalization of the pos-0
npz); PASS only if it transfers to the untouched 2025 gates. Until then the flagship claim stays
narrowed: per-pitch calibration (ECE 0.0114) intact; PA-level K/BB/HR marginals biased — Tier-A
rank/differential products usable with disclosure, absolute-rate products blocked.

Plan for the refit: `PHASE_0.6.2_PLAN.md` (2026-08-04) — includes remediation of the pos-0
npz 2025 fit-on-holdout taint discovered during planning.

### 2026-08-10 — Phase 0.6.2 executed: KILL at the §4 fit stage (Phase 0.6 closes as FAIL)

- Ran `scripts/pitchgpt_fit_rollout_calibration.py` (2023-only W fit, seed 42, 10K PAs x 100
  samples, horizon 6). The pre-registered §6 kill criterion fired on its FIRST disjunct: the
  fit did not converge within 2 fixed-point iterations — max per-position class-marginal
  |delta| vs empirical: raw-T 16.37pp → iteration 1 (W1) **4.418pp** → iteration 2 (W2)
  **2.625pp**, threshold 1.0pp. Persistent residual: ball-marginal deficit at positions 0–4
  (reweighting is partially undone by the count-trajectory feedback loop — the exposure-bias
  signature; the WS5 research verdict called this in advance).
- **NO artifact shipped**: `models/calibration_rollout_perpos.npz` does not exist. The
  non-converged W is quarantined at
  `results/pitchgpt/rollout_calibration_fit_2023/W_FAILED_FIT_quarantine.npz` (sha256
  395e6fcd...) with full provenance sidecar. §10.A6 registration: n/a.
- **The §5 single 2025 evaluation NEVER RAN** (§6: "stop permanently"); holdout contact #13
  was NOT spent — 2025 budget stands at 12/14, documented in a `docs/holdout_ledger.jsonl`
  note entry. The §10.A2 production-path ECE rode that contact and is therefore UNMEASURED
  (needs a new dated amendment + ledger authorization if wanted). Attribution diagnostic
  (#14) not triggered.
- Integrity: v2.pt + a1.pt SHA256 byte-identical pre/post (v2 = registry-pinned 6f952054...);
  DuckDB read-only throughout; `tests/test_pitchgpt_sim.py` + `tests/test_holdout_ledger.py`
  72 passed / 1 skipped (the skip = shipped-W provenance test, correctly absent);
  `scripts/verify_artifacts.py` ok=19 fail=0.
- Verdict doc: `docs/models/pitchgpt_phase062_results.md`. Per §6 the flagship claim stays
  permanently narrowed to "per-pitch calibrated rollout engine" (post-T ECE 0.0114 claim
  untouched by construction); PA-level absolute-rate products drop from Tier-A scope; the
  0.6.1 wOBA/PA-length TAINTED PASSes are now permanently unresolved for v2-era PitchGPT.
  Claims-registry/product-scope execution + any WS5.2 retrain go/no-go = Batch D
  (orchestrator + user), per the 2026-08-10 plan §8 K5.

### 2026-08-10 — Batch D: K5 consequences executed + WS5.2/5.3 v2 spec pre-registered

User adjudication of 2026-08-10: **K5 FIRED. Phase 0.6.2 is KILLED**; consequences execute now,
and the WS5.2 v2 retrain is green-lit *with its spec pre-registered BEFORE any training* (no
training in this batch; the spec freezes by the orchestrator's commit).

Consequences applied (all documentation/surface work — no model, no GPU, no DB write):

- **Claims registry** (`docs/claims/claims.yaml`): new active entry
  `pitchgpt_phase062_kill` (iteration 1 = 4.418pp, iteration 2 = 2.625pp vs the 1.0pp
  threshold, 2023 fit cohort, contact #13 unspent, no artifact shipped). Caveats updated on
  `pitchgpt_per_pitch_ece` (production-path ECE now UNMEASURED **and stranded** — it rode
  contact #13; a standalone measurement needs a new dated amendment plus one of the 2
  remaining budgeted 2025 contacts, NOT authorized), `pitchgpt_pa_rates_fail` (FAIL is now the
  permanent PA-level position) and `pitchgpt_woba_pa_pass_pre062` (PASSes permanently
  unearned).
- **Dashboard**: `views/matchup_sim.py` (A3) now WITHHOLDS every simulated wOBA quantity —
  level, p05/p25/p50/p75/p95 bands, histogram, mean, in-play-hit share, K%/BB%/HR% — and
  publishes only the pair's ordinal position in the loaded cohort behind a scope banner.
  Median-centring was evaluated as an alternative to withholding and REJECTED: PA-terminal
  wOBA is zero for every PA ending in an out, so the simulated median is exactly zero and a
  median-centred display renders byte-identical numbers under a "delta" label.
  `views/pitch_call_grades.py` (A1) keeps its
  grades — a rank/differential product §6 explicitly permits — behind a required marginal-bias
  disclosure that names the kill.
- **TAINTED-pending-0.6.2 markers retired repo-wide** (they now cite the kill): this file's
  phase table + the 0.6.1 A/B note, `PHASE_0.6_DIAGNOSIS.md` §9.3, `docs/NORTH_STAR.md` audit
  note, `docs/NORTH_STAR_CURRENT.md` §3.3, `docs/awards/methodology_paper_pitchgpt.md` note (3),
  `docs/awards/methodology_paper_pitchgpt_v2.md` (new correction notice — its "calibrated
  PA-level distribution generator" framing may not be written into prose),
  `docs/models/pitchgpt_validation_spec.md` foreknowledge amendment.
  `PHASE_0.6.2_PLAN.md` §§1–8 were deliberately NOT edited (frozen verbatim; §11 already
  records the kill).
- **New pre-registration**: `docs/pitchgpt_sim_engine/PITCHGPT_V2_SPEC.md` — chain-rule
  factorized heads (pitch_type → zone|type → velo|type,zone), rollout-aware curriculum
  fine-tuning, data policy (train ≤2023, dev = 2024 burned tier, 2025 budgeted tier NOT
  touched, sealed-2026 lockbox = exactly ONE contact at season end per K5), a gate suite that
  can fail (classwise-ECE/TACE, KCE hypothesis test, PIT/marginal calibration, per-count-state
  binned calibration, decision calibration) with numeric thresholds fixed in advance, and
  fit-stage + gate-stage kill criteria. **No training may start before that spec's freeze
  commit exists**; its deviations-log entry 1 is the freeze SHA, added by the orchestrator.
