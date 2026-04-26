# PitchGPT Sim Engine — Phase 0.5 Execution Plan

**Date:** 2026-04-25
**Status:** drafted, not started. Phase 0.5/0.6 prerequisites all green (Plan B closed 2026-04-26 with A1 ship verdict).
**Purpose of this document:** self-contained, navigational. The next session reading only this file plus `docs/NORTH_STAR.md`, `docs/pitchgpt_sim_engine/SIM_ENGINE_API.md`, and `docs/pitchgpt_sim_engine/COORDINATION.md` should be able to execute end-to-end through Phase 0.6 with no further context. Match the rigor of `EXECUTION_PLAN.md`.

---

## 1. Start here

**Purpose.** Implement Phase 0.5 (rollout harness `src/analytics/pitchgpt_sim.py` per the SIM_ENGINE_API contract) AND Phase 0.6 (10K rollout sanity check on the 2025 cohort). These are the last two Phase-0 items — after they land, Tier-A items (A1 grades, A2 projections, A3 matchup sim per `EXECUTION_PLAN.md` §6) are unblocked.

**Role.** PM-level coordinator per `feedback_pm_role.md` — delegate implementation to subagents, do NOT write code directly. Per `feedback_validation_agent.md`, run a validation agent after Agents 1+2 complete (a 2-agent batch counts as a 3+ batch when paired with the validation agent itself; rule is non-negotiable here because the contract surface is large).

**Goal.** `src/analytics/pitchgpt_sim.py` callable in <5s per 10-PA batch on RTX 3050, samples calibrated under SIM_ENGINE_API §3.5, marginals match 2025 league rates within ±10% relative or ±1pp absolute, mean wOBA within ±0.015. The API contract surface (rollout entry, dataclasses, OutcomePredictor protocol with four implementations, aggregation utilities, calibration validity) is fully specified — Phase 0.5 is execution against it.

**Locked context (do NOT re-litigate).**
- **A1 ships as the production OutcomePredictor.** `models/pitchgpt_v2_outcomehead_a1.pt` (151KB, 3-layer MLP head 211→128→64→7 over concat(hidden + context + pitch_type_oh + zone_oh + velo_oh), T=0.8003, ECE post-T 0.0114, +18.31% lift over freq prior). See `COORDINATION.md` "Locked decisions" + Phase 0.3 status row.
- **Backbone v2 frozen.** `models/pitchgpt_v2.pt` SHA256 `6f952054…62883c`. Byte-identity verified through Plan B Step 2. Do NOT alter, retrain, or refactor `src/analytics/pitchgpt.py` during Phase 0.5.
- **Plan B verdict closed.** A1 is production. A3 (XGBoost) is the Plan B fallback the API still wires (per SIM_ENGINE_API §4.2) but is not the default. PG-frozen-head (Phase 0.3 FAIL) is preserved for replay only (per SIM_ENGINE_API §4.1). Empirical lookup table is the kill-path fallback (per SIM_ENGINE_API §4.3) and is OPTIONAL for Phase 0.5 if its lookup parquet does not yet exist on disk — register a stub that raises a clear `KeyError` instead of silently failing.
- **Sampling-fidelity narrowed claim.** "Calibrated rollout engine" is the public claim per `EXECUTION_PLAN.md` §3 — the Phase 0.6 sanity check measures marginals, not LSTM-superiority. Do NOT regress on retracted framings.

**How this document is used.** §2 critical path. §3 Phase 0.6 spec. §4 sequencing. §5 risks. §6 binding gates. §7 next-session pickup. §8 paste-ready agent prompts. §9 explicit do-NOT list. §10 file appendix.

---

## 2. Critical path — Phase 0.5 ticket breakdown

The implementation is decomposed into six tickets, sequenced for two parallel implementing agents + one validation agent (§4). Every ticket cross-references SIM_ENGINE_API by section — DO NOT restate API content in implementation; refer to the spec.

### 0.5.1 — Skeleton + dataclasses + `rollout()` entry point

- **Owner.** Agent 1.
- **Spec.** SIM_ENGINE_API §3 (`rollout()` signature, `PAContext`, `RolloutResult`).
- **Deliverable.** `src/analytics/pitchgpt_sim.py` with:
  - `PAContext` (frozen dataclass, fields per §3.2 table). Include `PAContext.from_pitches_row(row, ump_scalar)` factory.
  - `RolloutResult` (frozen dataclass, fields per §3.3 table). Includes pad-with-NaN convention on `pitch_probs`/`outcome_probs`.
  - `ROLLOUT_PAD_PITCH = 2210` (= `PAD_TOKEN`), `ROLLOUT_PAD_OUTCOME = 7` constants.
  - `rollout()` entry point with the §3.1 signature, dispatching to a backbone-loaded singleton + the supplied `OutcomePredictor`. Implements:
    - All edge cases from §3.1 (`n_samples == 0`, `horizon == 0`, `horizon > 12` warning, `temperature != 1.0` warning + force calibration_valid=False, `prefix_pitch_tokens` length check).
    - PA-termination logic per §3.3 (outcome-driven → walk → strikeout → horizon exhaustion, first match wins).
    - `sampling_metadata` schema per §3.4 (every required key present).
- **Implementation references.**
  - Backbone forward pass: `src/analytics/pitchgpt.py::PitchGPTModel.forward` (line 1095).
  - Context vector construction: `PitchTokenizer.encode_context` (line 528) + `context_to_tensor` (line 580). The rollout's per-position context vector is *currently constant within a PA* — counterfactual mid-PA context mutation is Phase 1 (per §5 risk 2).
  - Token sampling: existing `PitchSequenceDataset` data path (line ~803) shows the prefix-token shape; rollout sampling reuses the backbone's softmax + `torch.multinomial`.
- **Out of scope.** OutcomePredictor implementations (0.5.2). Aggregation utilities (0.5.3). Calibration gate (0.5.4).
- **Effort.** M (~0.5 day).

### 0.5.2 — `OutcomePredictor` Protocol + four concrete implementations

- **Owner.** Agent 1 (continues after 0.5.1).
- **Spec.** SIM_ENGINE_API §4 (Protocol + 4.1–4.4 implementations).
- **Deliverables (in `src/analytics/pitchgpt_sim.py` or split into a new `src/analytics/pitchgpt_outcome_predictors.py` — implementer's call):**
  - `OutcomePredictor` Protocol matching §4 (runtime-checkable, three required attributes + `predict_outcome_probs` method).
  - **`PGConcatHeadPredictor` (PRODUCTION).** Wraps `FrozenOutcomeHeadConcat` from `src/analytics/pitchgpt_outcome_head.py`. Loads `models/pitchgpt_v2_outcomehead_a1.pt`. Consumes `backbone_hidden`, `context_vec`, AND a token-decomposed pitch_token (decompose via `(token // (NUM_ZONES * NUM_VELO_BUCKETS), ...)` — see `pitchgpt_outcome_a1_concat.py` for the canonical decomposition). Applies T=0.8003 from its calibration.json.
    - `name = "pg_concat_head"`. **Note:** this name is NEW relative to SIM_ENGINE_API §3.4 enumeration — explicitly add `"pg_concat_head"` to the allowed `outcome_predictor` metadata literals. The API spec was drafted before A1 was named the winner; this is a clarifying extension, not a contract violation.
  - **`PGFrozenHeadPredictor` (DEPRECATED, kept for replay).** Wraps `FrozenOutcomeHead` (the Phase 0.3 FAIL artifact). Loads `models/pitchgpt_v2_outcomehead.pt`. Per SIM_ENGINE_API §4.1 — kept registered for backtest replay, NOT used by default consumers.
  - **`XGBoostOutcomePredictor` (Plan B fallback).** Per §4.2. Loads `models/pitchgpt_outcome_xgb.bin` IF PRESENT. If the checkpoint is missing on disk (it is, as of 2026-04-25 — A3 lost to A1 and was not promoted), the registry registration MUST raise `KeyError` cleanly when consumers call `OutcomePredictorRegistry.get("xgboost")` — don't crash at module import.
  - **`EmpiricalPATerminalLookup` (kill-path fallback).** Per §4.3. Loads `models/pitchgpt_outcome_empirical_lookup.parquet` IF PRESENT. Same conditional-registration pattern as XGBoost.
  - `OutcomePredictorRegistry` per §8.2 (`get`, `register`, `list_registered`).
- **Critical implementation note.** `PGConcatHeadPredictor` must expose backbone hidden states without mutating the backbone. The A1 training script (`scripts/pitchgpt_outcome_a1_concat.py`) has the canonical pattern: `extract_backbone_hidden_states` from `pitchgpt_outcome_head.py` operates on `backbone.eval()` mode and runs the embedding+transformer stack but stops before `output_head`. Reuse it as-is. The predictor wraps both the backbone forward (for hidden states) AND the head forward (for outcome probs); the rollout loop calls the predictor once per position with already-extracted hidden states + freshly-sampled pitch tokens.
- **Out of scope.** Training new predictors. Refactoring `pitchgpt_outcome_head.py`.
- **Effort.** M (~0.5 day).

### 0.5.3 — Aggregation utilities

- **Owner.** Agent 1 (continues after 0.5.2; final slice of Agent 1's work).
- **Spec.** SIM_ENGINE_API §5.1–§5.4.
- **Deliverables.**
  - `pa_woba_distribution(rollout, woba_lookup)` — §5.1. **For 0.5/0.6: use a default constant `WObaTable.default()` lookup keyed on the 7-class outcome only (a 7-element scalar table: ball/CS/SS/foul/PA-incomplete = 0; in_play_out = 0; in_play_hit = 0.892 league-avg-on-base wOBA-per-hit; HBP = 0.708).** This is a **TBD-resolution-for-0.5**: SIM_ENGINE_API §5.1 punts the question of "scalar table vs full Statcast (outcome × pitch_type) table." The recommended Phase-1 follow-up is the full Statcast-empirical table. Document the 7-element default values in the docstring + cite this doc.
  - `percentile_of_actual_outcome(rollout, actual_outcome, actual_pitch_token, woba_lookup)` — §5.2.
  - `outcome_marginal(rollout, outcome_class)` — §5.3.
  - `pitch_token_marginal(rollout, position)` — §5.4.
- **NaN handling rule** (load-bearing, per SIM_ENGINE_API §3.3): truncated-position rows in `pitch_probs` and `outcome_probs` are NaN, NEVER zero. All four utilities use `np.nanmean` / `np.nanpercentile`. Aggregation against silent-zero would bias KLs and Wassersteins toward zero — this is the bug that the §9.5 unit test explicitly catches.
- **Out of scope.** Per-pitcher / per-game / per-season aggregations (those live in Phase-1 consumer modules, NOT in `pitchgpt_sim.py`).
- **Effort.** S (~3 hours).

### 0.5.4 — Calibration validity gate + per-feature CDFs

- **Owner.** Agent 2 (parallel with Agent 1, but depends on 0.5.1 dataclass shapes).
- **Spec.** SIM_ENGINE_API §6 (4 conditions + 9 enumerated invalid-reason strings).
- **Deliverables.**
  - `_check_calibration_valid(starting_context, backbone_calibration, predictor_calibration, temperature, horizon) -> tuple[bool, list[str]]` — internal to `pitchgpt_sim.py`. Implements all 4 conditions of §6 + emits the §6 enumerated reason strings.
  - **Build `models/calibration_feature_cdfs.npz`** for the v2 backbone (per SIM_ENGINE_API §6 condition 4 + §7.2 step 5). The file stores empirical CDFs over the calibration cohort (2025 pitcher-disjoint holdout, same cohort as `scripts/pitchgpt_2025_holdout.py`) for: `count_state` ∈ [0, 11], `outs` ∈ [0, 2], `score_diff` (continuous → bucketed per `encode_context`'s 5 levels), `inning_bucket` ∈ [0, 3], plus a `runner_state` ∈ [0, 7] occurrence-mask (for the `"context_runner_state_unseen"` reason).
  - **Reusable build script: `scripts/pitchgpt_build_calibration_cdfs.py`** (NEW, single-purpose). Reads the 2025 cohort using the same loader as `pitchgpt_2025_holdout.py`; emits the npz file. Registered in `pitchgpt_sim.py`'s checkpoint discovery so the calibration-feature-CDFs path is sibling to the backbone checkpoint per SIM_ENGINE_API §6 condition 4.
- **Out of scope.** Re-fitting backbone temperature (it's already done — see existing `models/pitchgpt_v2.pt` calibration material). Outcome-predictor calibration (handled in 0.5.5).
- **Effort.** M (~0.5 day, dominated by the CDFs script which has to load + bin the 2025 cohort).

### 0.5.5 — `calibration.json` for A1 outcome predictor

- **Owner.** Agent 2 (continues after 0.5.4).
- **Spec.** SIM_ENGINE_API §7.3 (calibration.json schema).
- **Deliverables.**
  - **Write `models/calibration_pitchgpt_v2_outcomehead_a1.json`** — adjacent to the A1 checkpoint per the §7.3 schema. The values are already known from `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/metrics.json`:
    - `T = 0.8003096499977166`
    - `ECE_pre = 0.0614403106926628` (test set, `ece_pre_temp`)
    - `ECE_post = 0.011409712028199842` (test set, `ece_post_temp`)
    - `holdout_season = 2025`
    - `holdout_n_pitches = 204513`
    - `fit_date = "2026-04-25"`
    - `checkpoint_sha256` = recompute SHA256 of `models/pitchgpt_v2_outcomehead_a1.pt` at write time, embed.
    - `predictor_kind = "pg_concat_head"`
  - The implementer MUST NOT invent values. If a value differs from the metrics.json, halt and surface to PM.
  - The file is read by `PGConcatHeadPredictor.__init__` at predictor-construction time. Per §7.3 refusal behavior: missing file, stale `fit_date` (>12 months), out-of-budget `ECE_post` (>0.05 for outcome predictors per §7.3), or mismatched `checkpoint_sha256` raises `CalibrationError`.
  - **Verify the same backbone calibration assets exist already** for `models/pitchgpt_v2.pt`. If not, the implementer surfaces a blocker — do NOT fit a new backbone calibration during 0.5; that's the calibration-author's job (Plan B context: the v2 backbone calibration is in `results/pitchgpt/2025_holdout/...` already; if the JSON adjacent to the .pt is missing, Agent 2 writes one assembled from existing artifacts).
- **Out of scope.** Re-running temperature scaling (use the locked T from training). Calibrating XGBoost/empirical predictors (those checkpoints don't exist; their calibration.json files are out-of-scope for Phase 0.5).
- **Effort.** S (~2 hours, mostly mechanical).

### 0.5.6 — Unit tests

- **Owner.** Agent 2 (continues after 0.5.5).
- **Spec.** SIM_ENGINE_API §9 (validation gates the API itself must pass).
- **Deliverable.** `tests/test_pitchgpt_sim.py` with ≥10 tests, covering at minimum:

| # | Test name | What it validates | Source spec |
|---|---|---|---|
| 1 | `test_horizon_one_token_marginal_matches_softmax` | `pitch_token_marginal(rollout, position=0)` over 10K samples matches the backbone's direct next-token softmax in TVD < 0.01 (per the §6.4 binding gate). Stricter than KL ≤ 0.005 nats but easier to reason about. | API §9 + §6.4 |
| 2 | `test_pad_token_semantics` | After PA termination, `pitch_tokens` rows past the terminator are filled with `ROLLOUT_PAD_PITCH`; `pitch_probs` rows past are NaN, NOT zero. | API §3.3 |
| 3 | `test_none_predictor_degraded_path` | `outcome_predictor=None` → `outcomes`, `outcome_probs`, `pa_outcome` are all `None` in the result. PA-termination only fires on count-driven (walk/strikeout) heuristics or horizon exhaustion. `sampling_metadata["outcome_predictor"] == "none"`. | API §4.4 |
| 4 | `test_calibration_valid_temperature_branch` | `temperature != 1.0` forces `calibration_valid=False`, appends `"temperature_not_unity"` to reasons. | API §6 condition 1 |
| 5 | `test_calibration_valid_horizon_branch` | `horizon=13` forces `calibration_valid=False`, appends `"horizon_exceeds_validated"`. | API §3.1 + §6 |
| 6 | `test_calibration_valid_context_band_branch` | `PAContext` with `outs=99` (out of band) appends `"context_outs_out_of_band"`. Construct deliberately-out-of-band contexts for each of 4 numeric features + 1 categorical (`runner_state_unseen`) = 5 sub-tests OR a single parameterized test with 5 cases. | API §6 condition 4 |
| 7 | `test_pa_termination_walk_with_predictor` | A rollout where the outcome predictor is forced to emit "ball" 4 times in a row terminates exactly at position 3 with `final_count == (4, 0)`, `pa_outcome == ROLLOUT_PAD_OUTCOME` (walk has no terminal-outcome class — it terminates via count). | API §3.3 condition 2 |
| 8 | `test_pa_termination_in_play_with_predictor` | Forcing predictor to emit "in_play_hit" terminates at position 0 with `pa_outcome == 5`. | API §3.3 condition 1 |
| 9 | `test_aggregation_nan_mask_correctness` | Per API §9 pad-NaN test: 50 truncated samples out of 100 must produce identical `np.nanmean(...)` and "manually filter `pa_terminated`" results to within float epsilon. | API §9 |
| 10 | `test_latency_10pa_batch` | 10-PA batch with `n_samples=100`, `horizon=6`, A1 predictor must complete in <5s on RTX 3050 (or <30s on CPU as a CI-friendly fallback — gate the assertion on `torch.cuda.is_available()`). FAIL → marks the latency-budget bug per API §9 (`return_probs=False` becomes default). | API §9 + EXECUTION_PLAN.md §6.0.5 |
| 11+ | Optional: `test_outcome_marginal_full_grid_shape`, `test_pitch_token_marginal_position_none_shape`, `test_metadata_required_keys_present` | Smoke shape + metadata coverage. | API §3.4 + §5 |

- **Note on test 10.** The latency assertion is the HARD constraint per the goal in §1. If the first implementation misses 5s on RTX 3050, the next iteration must instrument GPU/CPU profiling BEFORE adding optimizations — measure, then optimize, never optimize blind. Document this in a code comment.
- **Effort.** M (~0.5 day).

---

## 3. Phase 0.6 — Rollout sanity check on 2025

### 3.1 Scope

Run the rollout harness against 10K randomly-sampled 2025 PA starting contexts. Compare aggregate marginals (K%, BB%, HR%, mean wOBA, mean PA-length-in-pitches) to the empirical 2025 league rates. Per `EXECUTION_PLAN.md` §6.0.6 — the rollout-sanity success criterion is the gate-of-record for Phase 0 exit.

### 3.2 Methodology

- **Cohort.** 10K PA starts sampled from 2025 pitcher-disjoint holdout cohort (same as `scripts/pitchgpt_2025_holdout.py`'s test split). Use deterministic `np.random.default_rng(seed=42)`.
- **For each PA start:**
  - Construct `PAContext` from the row using `PAContext.from_pitches_row(row, ump_scalar=...)`. Resolve `ump_scalar` to season-2025-league-median if NULL upstream.
  - Run `rollout(context, outcome_predictor=PGConcatHeadPredictor(), n_samples=100, horizon=6, temperature=1.0, seed=<deterministic per row>)`.
  - Aggregate the 100 samples to a per-PA outcome distribution.
- **Empirical baseline.** 2025 league rates. Source: direct aggregation of the 2025 `pitches` cohort (NOT `season_pitching_stats`, which is per-pitcher, not per-PA league rate). Compute league K% = strikeouts / PAs, BB% = walks / PAs, HR% = home_runs / PAs, mean PA wOBA = mean(woba_value on PA-terminal rows). Filter to the same pitcher-disjoint test pitchers used by the cohort.
- **Outputs:**
  - `results/pitchgpt/rollout_sanity_2025/metrics.json` — sampled vs empirical for K%, BB%, HR%, mean wOBA, mean PA length, with 95% bootstrap CIs on the sampled side.
  - `results/pitchgpt/rollout_sanity_2025/report.md` — markdown report with PASS/FAIL per gate.

### 3.3 Honest-caveat (must be in the report)

The A1 outcome predictor's `in_play_hit` test log-loss is 2.34 (per `metrics.json::test_metrics.per_class_log_loss.in_play_hit`), which clears the WEAKER PASS gate (<2.5) but misses the full PASS gate (<2.0). This means the hit-vs-out marginal is noisier than the league average, and **mean-wOBA aggregation will be noisier than K%/BB% because it depends on the hit-vs-out resolution**. The report MUST surface this — `EXECUTION_PLAN.md` §3 calls out that the locked claim is "calibrated rollout engine" with calibration-as-load-bearing; we honor that by disclosing where the rollout sub-distributions are weakest. Do NOT paper over it.

### 3.4 Phase 0.6 secondary run — None-predictor bias surfacing

**Run the sanity check TWICE:**
1. With `outcome_predictor=PGConcatHeadPredictor()` — primary, gates Phase 0.6 PASS.
2. With `outcome_predictor=None` — secondary, **NOT gated**. Per SIM_ENGINE_API §4.4 + §5 risk 3 of this plan, the count-only fallback heuristic biases K%/BB% high (every in-play and foul is misclassified as a strike). The secondary run measures the bias magnitude empirically so consumers know what to expect when the predictor is unavailable.

The secondary run's K%/BB%/HR% deltas vs the primary go into the report as a "None-predictor degraded mode" section. Report writes "expected bias confirmed at +Xpp on K%, +Ypp on BB%" — does NOT trip the Phase 0.6 PASS/FAIL gate.

### 3.5 Phase 0.6 success criteria (binding)

Per `EXECUTION_PLAN.md` §6.0.6 + SIM_ENGINE_API §9:
- K%, BB%, HR% within ±10% relative or ±1pp absolute (whichever is tighter) of 2025 league empirical rates.
- Mean wOBA within ±0.015 absolute.
- Mean PA-length-in-pitches within ±0.5 pitches of empirical 2025 average (~3.9).
- `calibration_valid=True` on at least 95% of the 10K rollouts (per-PA flag from `sampling_metadata`). If <95%, surface why — the most likely cause is the percentile-band gate from SIM_ENGINE_API §6 condition 4 over-rejecting.

FAIL on any of K%/BB%/HR%/mean-wOBA → API marked "uncalibrated rollout" per SIM_ENGINE_API §9; do NOT ship the harness. Diagnose root cause first (most likely: PA-termination logic bug, hit-vs-out marginal noise, or context CDF mis-build).

---

## 4. Sequencing within the next session

**Single session, single GPU, single user.** No parallel rollout training (every checkpoint already exists; nothing trains in Phase 0.5/0.6).

**Recommended subagent batch (optimized for ~1 session of wall-clock):**

| Stage | Agent | Tickets | Wall-clock | Depends on |
|---|---|---|---|---|
| 1 | Agent 1 | 0.5.1 + 0.5.2 + 0.5.3 (API contract, predictors, aggregations) | ~1.5 day | nothing |
| 1 (parallel) | Agent 2 | 0.5.4 + 0.5.5 + 0.5.6 (calibration gate, calibration.json, unit tests) | ~1.5 day | Agent 1's dataclass shapes — coordinate via the API doc, not via git |
| 2 | Validation agent | pytest run, import spot-check, regression report | ~10 min | Agents 1+2 PASS |
| 3 | Agent 3 | Phase 0.6 sanity check (primary + secondary runs, report) | ~half day | Validation agent PASS |

**Coordination rule for Agents 1 & 2.** Agent 2's calibration-validity gate (0.5.4) reads `RolloutResult` and `PAContext` shapes — Agent 1's 0.5.1 freezes those. Solution: Agent 2 starts on 0.5.4 ONLY after Agent 1 commits 0.5.1's dataclass definitions to a feature branch (or pushes the dataclasses to the implementation file before moving to 0.5.2). Agent 2's prompt enforces this: "Read `src/analytics/pitchgpt_sim.py`'s dataclass section before starting."

**Latency-test gate (5s on 10-PA batch) is HARD.** If Agent 2's test 10 fails on the first implementation:
- Do NOT ask Agent 1 to "make it faster" without measurement.
- Agent 3 (or a fresh perf agent) runs `torch.profiler` on the 10-PA batch, identifies the bottleneck (likely: too-frequent CPU-GPU transfers, or `return_probs=True` at scale memory cost), reports findings.
- Agent 1 then optimizes against the profile, NOT against intuition.

**Validation-agent-after-batch is non-negotiable** per `feedback_validation_agent.md`. It runs pytest, does an import spot-check on every module touched (`pitchgpt_sim.py`, `pitchgpt_outcome_head.py` if touched, `pitchgpt_calibration.py` if touched), and emits a regression report. Cannot be skipped.

**Commit cadence.** Per `feedback_always_push_clean_workspace.md`, push after each ticket-batch commit. Per `feedback_pm_role.md`, the PM session ends with no uncommitted local work.

---

## 5. Risks + open questions

### 5.1 Hidden-state shape coupling between backbone and head

**Risk.** `PGConcatHeadPredictor` has to expose backbone hidden states without retraining. The backbone's `d_model=128` is hard-coded, and the A1 head expects `211 = 128 + 35 + 17 + 26 + 5`. If a future Plan A swaps the backbone (per SIM_ENGINE_API §10 compatibility matrix), the head input shape changes and the predictor's checkpoint isn't backward-compatible.

**Mitigation for 0.5.** Predictor wraps both backbone forward + head forward into a single object. The d_model coupling is documented in the predictor's docstring + module-level constants. **DO NOT re-engineer for backbone-agnostic heads in 0.5/0.6** — that's a Phase-1+ refactor.

### 5.2 Counterfactual context-mutation is TBD-pending-Phase-1

**Risk.** SIM_ENGINE_API §6.4.3 (and `EXECUTION_PLAN.md` §3.3) describe counterfactual context mutation — sample prefix unchanged, mutate the context vector for continuation positions. The current plan keeps context constant within a PA. A1 (counterfactual pitch-call grade) eventually needs mid-PA context mutation.

**Mitigation for 0.5.** **Not blocking 0.5/0.6.** A1's counterfactual setup uses `prefix_pitch_tokens` to freeze the prefix, then re-rolls with the unchanged starting context. Mid-PA situational-context mutation (count, outs, etc.) is a Phase-1 extension. **Flag the future Tier-A agents** that the context-vector is constant within a PA in the current rollout — A1's spec needs to handle that limitation.

### 5.3 None-predictor PA-termination heuristic biases K%/BB%

**Risk.** Per SIM_ENGINE_API §4.4, the count-only fallback heuristic (zone in-strike-zone → +1 strike, otherwise +1 ball) misclassifies all in-play, foul, and HBP outcomes as strikes. K%/BB% rates from None-predictor rollouts will be biased high.

**Mitigation for 0.5.** Documented contract per SIM_ENGINE_API §4.4. **Phase 0.6 measures the bias magnitude** (per §3.4 of this plan) — secondary run quantifies "expected bias = +Xpp" so consumers know what they get when the predictor is unavailable. Does NOT gate 0.6 PASS.

### 5.4 Open question — `pa_woba_distribution` lookup table choice

**Question.** Should `pa_woba_distribution` (SIM_ENGINE_API §5.1) use:
- **(a)** A simple 7-element scalar table keyed on outcome class only.
- **(b)** A full Statcast empirical (outcome × pitch_type) table — wOBA-per-outcome-per-pitch-type, weighted by 2015–2022 cohort.

SIM_ENGINE_API §5.1 punts on this.

**Recommendation for 0.5/0.6.** Default to **(a)** — the 7-element scalar table per ticket 0.5.3 spec. Phase-1 follow-up: when A1 ships, the full Statcast table is the obvious unlock. Document in `pa_woba_distribution`'s docstring: "default uses scalar table; pass `WObaTable.from_statcast(...)` when full per-pitch-type wOBA is needed."

**TBD that I'm flagging, not resolving.** The recommendation above is a default, not a contract. A future PM may choose (b) for 0.5 if Phase 0.6's mean-wOBA gate is failing borderline and (b) tightens it.

### 5.5 Risk — Phase 0.6 mean-wOBA gate marginal due to A1's hit-vs-out noise

**Risk.** A1's `in_play_hit` log-loss (2.34) is the noisiest class. Phase 0.6's ±0.015 mean-wOBA gate is tight; if A1's hit-vs-out marginal is biased even slightly, the rollout's mean wOBA could miss the gate.

**Mitigation.** Phase 0.6 report MUST decompose mean-wOBA into per-component contributions (in_play_out × wOBA_out=0, in_play_hit × wOBA_hit=0.892, HBP × wOBA_HBP=0.708, ball-in-walk × wOBA_walk=0.690) and surface where the largest gap is. If the gate fails on hit-vs-out specifically, the diagnosis is "A1 outcome predictor needs improvement"; if it fails on count-driven (walks/Ks), the diagnosis is "PA-termination logic bug." These are different fixes.

### 5.6 Risk — DuckDB single-writer

**Risk.** Per `CLAUDE.md` data rules, DuckDB is single-writer. Phase 0.6 reads the 2025 cohort. If the dashboard is running, the read might still succeed (dashboard holds a write lock; reads pass `read_only=True`). But if any Phase 0.5/0.6 step accidentally opens a write connection, it will deadlock.

**Mitigation.** Every DuckDB call in Phase 0.5/0.6 uses `read_only=True`. Validation agent grep-checks for this in modified files. If the user's dashboard is running, do NOT close it — the read-only contract handles concurrency.

---

## 6. Validation gates (binding — copy these into agent prompts)

Per `EXECUTION_PLAN.md` §6.0.5/§6.0.6 + SIM_ENGINE_API §9. PASS-FAIL with CIs in the gate report.

| § | Gate | Threshold | Source spec |
|---|---|---|---|
| 6.1 | **Unit-test count.** | ≥10 tests covering all major API paths (per the table in 0.5.6). | this plan §0.5.6 |
| 6.2 | **Latency budget.** | <5s per 10-PA batch (`n_samples=100`, `horizon=6`, A1 predictor) on RTX 3050. Larger-batch (100, 1K) measured but NOT gated. | EXECUTION_PLAN.md §6.0.5 + API §9 |
| 6.3 | **Marginal sanity (K%/BB%/HR%).** | Within ±10% relative or ±1pp absolute of 2025 empirical (whichever is tighter). 95% bootstrap CIs reported. | EXECUTION_PLAN.md §6.0.6 + API §9 |
| 6.4 | **Mean wOBA.** | Within ±0.015 absolute of 2025 empirical mean wOBA. | EXECUTION_PLAN.md §6.0.6 |
| 6.5 | **Calibration regression at H=1.** | Sampled `pitch_token_marginal(rollout, position=0)` over 10K samples must match backbone's direct next-token softmax in TVD < 0.01 (or KL ≤ 0.005 nats — pick one in the test). | API §9 |
| 6.6 | **Calibration-valid coverage.** | ≥95% of Phase 0.6's 10K rollouts have `calibration_valid=True`. | API §6 |
| 6.7 | **Pad-NaN convention.** | `np.nanmean` and "manually filter `pa_terminated`" produce identical results within float epsilon on a 50%-truncated test fixture. | API §9 |
| 6.8 | **Backbone byte-identity.** | `models/pitchgpt_v2.pt` SHA256 = `6f952054…62883c` pre and post any 0.5/0.6 work. | COORDINATION.md "Checkpoints — LOCKED" |
| 6.9 | **A1 checkpoint untouched.** | `models/pitchgpt_v2_outcomehead_a1.pt` byte-identity verified (recompute SHA256 at 0.5/0.6 close, compare to start). | COORDINATION.md "Checkpoints — LOCKED" |

**No commit until validation agent + Phase 0.6 PASS.** Per `feedback_pm_role.md` and §9 of this plan.

---

## 7. Where to pick up next session

**First action.** Read this plan + `SIM_ENGINE_API.md` + `COORDINATION.md` + the latest `NORTH_STAR.md` update (currently 2026-04-25, narrows PitchGPT to "calibrated rollout engine"). Then launch Agent 1 with the §8.1 prompt.

**Second action.** As soon as Agent 1's 0.5.1 dataclasses land in `src/analytics/pitchgpt_sim.py`, launch Agent 2 in parallel with the §8.2 prompt.

**Third action.** Validation agent (§8.3 prompt) after Agents 1+2 both report PASS.

**Fourth action.** Agent 3 (§8.4 prompt) for Phase 0.6, after validation agent PASS.

**Fifth action.** Commit the harness + tests + Phase 0.6 report in a single commit (per `/commit` style — no Claude co-author). Push. Update `COORDINATION.md` Phase 0.5/0.6 status rows from "Not started" to "COMPLETE." Update `EXECUTION_PLAN.md` §6.0.5 + §6.0.6 status with PASS/FAIL.

**If anything fails a gate.** Halt. Surface to PM. The default is "diagnose root cause, do NOT paper over." Per `feedback_pm_role.md`, the PM does NOT decide the fix — escalate to the user with diagnostic options.

---

## 8. Agent prompt templates (paste-ready)

### 8.1 Agent 1 — API contract + predictors + aggregations

```
You are Agent 1 for Phase 0.5 of the PitchGPT sim engine at `C:\Users\hunte\projects\baseball`. Build the rollout harness API contract per the locked spec. Do NOT retrain any model. Do NOT modify the v2 backbone or the A1 head.

Read first:
1. `docs/pitchgpt_sim_engine/PHASE_0.5_PLAN.md` §1, §2 (tickets 0.5.1, 0.5.2, 0.5.3), §5, §6.
2. `docs/pitchgpt_sim_engine/SIM_ENGINE_API.md` §3 (rollout, PAContext, RolloutResult), §4 (OutcomePredictor + four implementations), §5 (aggregation utilities), §8 (registry).
3. `docs/pitchgpt_sim_engine/COORDINATION.md` (file ownership, locked checkpoints).
4. `src/analytics/pitchgpt.py::PitchGPTModel` (~line 1095) and `PitchTokenizer.encode_context`/`context_to_tensor` (lines 528–610).
5. `src/analytics/pitchgpt_outcome_head.py` (`FrozenOutcomeHeadConcat` is the A1 head you wrap).
6. `scripts/pitchgpt_outcome_a1_concat.py` (canonical A1 token-decomposition + hidden-state extraction pattern).

Execute (in order):
1. Create `src/analytics/pitchgpt_sim.py`. Implement ticket 0.5.1: `PAContext`, `RolloutResult`, `ROLLOUT_PAD_*` constants, and the `rollout()` entry per SIM_ENGINE_API §3.1–§3.4. Include all edge cases. Apply the PA-termination logic per §3.3. Populate `sampling_metadata` per §3.4.
2. Implement ticket 0.5.2: `OutcomePredictor` Protocol, `PGConcatHeadPredictor` (production, wraps A1 head + v2 backbone hidden states), `PGFrozenHeadPredictor` (deprecated, per §4.1), `XGBoostOutcomePredictor` (conditionally registered if `models/pitchgpt_outcome_xgb.bin` exists), `EmpiricalPATerminalLookup` (conditionally registered if its parquet exists). `OutcomePredictorRegistry` per §8.2. The new name `"pg_concat_head"` must be added to allowed `outcome_predictor` literals — note this in the docstring.
3. Implement ticket 0.5.3: aggregation utilities `pa_woba_distribution`, `percentile_of_actual_outcome`, `outcome_marginal`, `pitch_token_marginal`. Use a 7-element scalar `WObaTable.default()` for wOBA (the values: ball/CS/SS/foul/in_play_out = 0; in_play_hit = 0.892; HBP = 0.708; per PHASE_0.5_PLAN §2.0.5.3). NaN-mask truncated positions — silent zero is forbidden.

Artifacts to create:
- `src/analytics/pitchgpt_sim.py` (NEW)

Guardrails:
- Do NOT touch `models/pitchgpt_v2.pt`, `models/pitchgpt_v2_outcomehead.pt`, or `models/pitchgpt_v2_outcomehead_a1.pt`.
- Do NOT modify `src/analytics/pitchgpt.py` or `src/analytics/pitchgpt_outcome_head.py`.
- DuckDB calls must pass `read_only=True`.
- Do NOT write any unit tests — that is Agent 2's job.
- Do NOT commit. Leave unstaged.

Return a 200-word summary:
- Files created + line counts.
- Protocol implementations registered (which ones; which are conditional).
- Any spec ambiguity you resolved + how you flagged it in code comments.
- Backbone + A1 SHA256 verified unchanged (yes/no).
```

### 8.2 Agent 2 — calibration gate + calibration.json + unit tests

```
You are Agent 2 for Phase 0.5 of the PitchGPT sim engine at `C:\Users\hunte\projects\baseball`. Build the calibration validity gate, write A1's calibration.json, and write the unit-test suite. Depends on Agent 1's dataclass shapes — read the dataclass section of `src/analytics/pitchgpt_sim.py` BEFORE starting.

Read first:
1. `docs/pitchgpt_sim_engine/PHASE_0.5_PLAN.md` §2 (tickets 0.5.4, 0.5.5, 0.5.6), §6 (gates table).
2. `docs/pitchgpt_sim_engine/SIM_ENGINE_API.md` §6 (4 conditions + 9 invalid-reason strings), §7 (calibration.json schema + refusal behavior), §9 (validation gates).
3. `src/analytics/pitchgpt_sim.py` (Agent 1's dataclasses — do NOT mutate, only read).
4. `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/metrics.json` (source of A1 calibration values).
5. `scripts/pitchgpt_2025_holdout.py` (cohort loader pattern for the calibration-feature-CDFs build).

Execute (in order):
1. Implement ticket 0.5.4: add `_check_calibration_valid()` to `pitchgpt_sim.py` per SIM_ENGINE_API §6's 4 conditions + 9 enumerated reason strings. Wire into `rollout()` so `sampling_metadata["calibration_valid"]` and `calibration_invalid_reasons` are populated.
2. Build `scripts/pitchgpt_build_calibration_cdfs.py` (NEW). Reads 2025 holdout cohort (read-only DuckDB), computes empirical CDFs over `count_state` ∈ [0,11], `outs` ∈ [0,2], `score_diff_bucket` ∈ [0,4], `inning_bucket` ∈ [0,3], `runner_state` ∈ [0,7]. Saves to `models/calibration_feature_cdfs.npz`.
3. Implement ticket 0.5.5: write `models/calibration_pitchgpt_v2_outcomehead_a1.json` with values copied from the metrics.json (T=0.8003096499977166, ECE_pre=0.0614403106926628, ECE_post=0.011409712028199842, holdout_season=2025, holdout_n_pitches=204513, fit_date="2026-04-25", checkpoint_sha256=recompute, predictor_kind="pg_concat_head"). Verify backbone calibration.json exists at `models/pitchgpt_v2_calibration.json` (or equivalent); if missing, halt and surface as a blocker.
4. Implement ticket 0.5.6: `tests/test_pitchgpt_sim.py` with the 10+ tests per PHASE_0.5_PLAN §2.0.5.6 table. Test 10 (latency) gates on `torch.cuda.is_available()`: 5s on GPU, 30s on CPU.

Artifacts to create:
- `scripts/pitchgpt_build_calibration_cdfs.py` (NEW)
- `models/calibration_feature_cdfs.npz` (NEW)
- `models/calibration_pitchgpt_v2_outcomehead_a1.json` (NEW)
- `tests/test_pitchgpt_sim.py` (NEW)
- `src/analytics/pitchgpt_sim.py` (EXTEND with `_check_calibration_valid` only; do NOT touch Agent 1's code)

Guardrails:
- Do NOT touch `models/pitchgpt_v2.pt`, `models/pitchgpt_v2_outcomehead.pt`, or `models/pitchgpt_v2_outcomehead_a1.pt`.
- Do NOT invent calibration values — copy from the metrics.json. If a value differs, halt.
- Do NOT modify Agent 1's dataclasses or `rollout()` body — only add `_check_calibration_valid` and wire it from the existing entry point.
- DuckDB read-only.
- Do NOT commit. Leave unstaged.

Return a 200-word summary:
- Files created + line counts.
- All 10+ tests names + pass/fail status.
- A1 calibration.json values match metrics.json (yes/no).
- Calibration-feature-CDFs artifact size + the 5 features + their CDF lengths.
- Any blocker surfaced.
```

### 8.3 Validation agent — post-batch regression check

```
You are the validation agent for Phase 0.5 of the PitchGPT sim engine at `C:\Users\hunte\projects\baseball`. Agents 1 and 2 just completed. Run the regression battery and report.

Execute:
1. `pytest tests/` — full suite. Report PASS/FAIL counts + any new failures introduced since `git log -n 5`.
2. Import spot-check: `python -c "from src.analytics.pitchgpt_sim import rollout, PAContext, RolloutResult, OutcomePredictor, PGConcatHeadPredictor, OutcomePredictorRegistry"`. Must import cleanly.
3. Import spot-check: every module touched in Agents 1+2's diff. Use `git diff --name-only` to identify; for each `.py`, run `python -c "import <module>"`.
4. Verify `models/pitchgpt_v2.pt` SHA256 == `6f952054d14ac6f918f3eb9502b496b70bc0c87dfc65dc50d98ee7244a62883c`. Verify `models/pitchgpt_v2_outcomehead_a1.pt` byte-size == 151289 (per metrics.json::checkpoint.size_bytes).
5. Spot-check: any new DuckDB call uses `read_only=True`. Use grep on diff: any `get_connection()` without `read_only=True` is a regression.
6. Run `tests/test_pitchgpt_sim.py::test_latency_10pa_batch` separately and report wall-clock + GPU/CPU device.

Return a 200-word summary:
- pytest verdict (PASS/FAIL, count of new vs existing failures).
- Import spot-check verdict.
- SHA256/size verification verdict.
- DuckDB read-only verdict.
- Latency verdict (wall-clock + device + PASS/FAIL on the 5s/30s gate).
- Any regressions introduced. If any, name them by file + reason. If clean, say "clean."

Guardrails: do NOT commit anything. Do NOT fix regressions yourself — surface them.
```

### 8.4 Agent 3 — Phase 0.6 sanity check

```
You are Agent 3 for Phase 0.6 of the PitchGPT sim engine at `C:\Users\hunte\projects\baseball`. Run the rollout sanity check on 2025. Depends on validation agent PASS.

Read first:
1. `docs/pitchgpt_sim_engine/PHASE_0.5_PLAN.md` §3 (Phase 0.6 spec — read all subsections), §6 (gates).
2. `docs/pitchgpt_sim_engine/SIM_ENGINE_API.md` §9 (validation gates the API itself must pass).
3. `src/analytics/pitchgpt_sim.py` (the harness Agent 1 + 2 just shipped).
4. `scripts/pitchgpt_2025_holdout.py` (cohort splits — same 2025 pitcher-disjoint test set).

Execute (in order):
1. Build a script `scripts/pitchgpt_rollout_sanity_2025.py` (NEW). Sample 10K PA starts from 2025 pitcher-disjoint test cohort with `np.random.default_rng(seed=42)`. For each, construct `PAContext.from_pitches_row(row, ump_scalar=...)`. Resolve NULL ump_scalar to 2025 league-median.
2. **PRIMARY run.** For each PA start, call `rollout(context, outcome_predictor=PGConcatHeadPredictor(), n_samples=100, horizon=6, temperature=1.0, seed=<deterministic per row>)`. Aggregate to per-PA outcome distribution. Compute K%, BB%, HR%, mean wOBA, mean PA-length-in-pitches with 95% bootstrap CIs.
3. **EMPIRICAL baseline.** Direct aggregation of the 2025 `pitches` test cohort (read_only=True). Compute the same five metrics over the actual data.
4. **SECONDARY run.** Same as PRIMARY but with `outcome_predictor=None`. Compute the same five metrics. Use this to surface bias magnitude per PHASE_0.5_PLAN §3.4. Do NOT gate on this run.
5. Write `results/pitchgpt/rollout_sanity_2025/metrics.json` (PRIMARY + SECONDARY + empirical, plus bootstrap CIs and PASS/FAIL per gate).
6. Write `results/pitchgpt/rollout_sanity_2025/report.md` with:
    - Gates table (per PHASE_0.5_PLAN §3.5 thresholds — 6.3 K%/BB%/HR%, 6.4 mean wOBA, 6.6 calibration_valid coverage).
    - Honest caveat per PHASE_0.5_PLAN §3.3 — A1's `in_play_hit` log-loss 2.34, mean-wOBA noisier than K%/BB%.
    - Decomposition of mean-wOBA per PHASE_0.5_PLAN §5.5 (component contributions if mean wOBA gate fails).
    - None-predictor bias section.
    - Backbone + A1 SHA256 verification.

Artifacts to create:
- `scripts/pitchgpt_rollout_sanity_2025.py` (NEW)
- `results/pitchgpt/rollout_sanity_2025/metrics.json` (NEW)
- `results/pitchgpt/rollout_sanity_2025/report.md` (NEW)

Guardrails:
- Do NOT touch any model checkpoint.
- Do NOT modify `src/analytics/pitchgpt_sim.py` (it's now committed surface).
- DuckDB read_only=True.
- Do NOT commit yet — PM will commit after reviewing the report.
- If any gate FAILs, do NOT paper over. Surface clearly in the report; PM decides next step.

Return a 200-word summary:
- Wall-clock for the 10K-rollout primary run.
- 5 metrics: PRIMARY sampled vs empirical, with deltas, with PASS/FAIL per gate.
- 5 metrics: SECONDARY sampled vs empirical, deltas (the bias measurement).
- Calibration_valid coverage % (gate 6.6).
- Any gate FAIL + your diagnosis hypothesis (PA-termination bug / hit-vs-out noise / context CDF mis-build).
```

---

## 9. What we explicitly do NOT do in Phase 0.5/0.6

- **Do NOT retrain any model.** Backbone v2 frozen. A1 head frozen. No new training jobs.
- **Do NOT modify the A1 head architecture.** `FrozenOutcomeHeadConcat` is locked. The predictor wraps it; it does not extend it.
- **Do NOT build any Tier-A consumer.** A1 grades, A2 projections, A3 matchup sim are Phase 1, NOT 0.5/0.6. Even if the harness lands cleanly with time to spare, do NOT start A1.
- **Do NOT skip the calibration_valid contract.** Per SIM_ENGINE_API §6, the API rejects rollouts that don't qualify. There is NO override flag.
- **Do NOT commit until validation agent PASSes.** Per §6 + `feedback_pm_role.md`.
- **Do NOT relitigate Plan B verdict.** A1 is production. Closed 2026-04-26 per `COORDINATION.md`.
- **Do NOT relitigate Phase 0.1 sampling-fidelity narrowing.** Locked. Do NOT add LSTM-superiority claims to docs.
- **Do NOT relitigate the 7-class outcome target.** Locked per `EXECUTION_PLAN.md` §3.1 decision 2.
- **Do NOT touch the dashboard.** No `src/dashboard/views/` work in Phase 0.5/0.6.
- **Do NOT extend the rollout to mid-PA context mutation.** Phase 1 risk per §5.2.
- **Do NOT build Tier-A consumer dossiers** (A1/A2/A3) — those are Phase 1 dossiers in `EXECUTION_PLAN.md` §6 and require a fresh planning pass.

---

## 10. Appendix

### 10.1 Files to create (Phase 0.5/0.6)

| Path | Owner | Created in ticket |
|---|---|---|
| `src/analytics/pitchgpt_sim.py` | Agent 1 (primary), Agent 2 (extend `_check_calibration_valid` only) | 0.5.1 / 0.5.2 / 0.5.3 / 0.5.4 |
| `tests/test_pitchgpt_sim.py` | Agent 2 | 0.5.6 |
| `scripts/pitchgpt_build_calibration_cdfs.py` | Agent 2 | 0.5.4 |
| `models/calibration_feature_cdfs.npz` | Agent 2 (output of build script) | 0.5.4 |
| `models/calibration_pitchgpt_v2_outcomehead_a1.json` | Agent 2 | 0.5.5 |
| `scripts/pitchgpt_rollout_sanity_2025.py` | Agent 3 | Phase 0.6 |
| `results/pitchgpt/rollout_sanity_2025/report.md` | Agent 3 | Phase 0.6 |
| `results/pitchgpt/rollout_sanity_2025/metrics.json` | Agent 3 | Phase 0.6 |

Optional (only if missing): `models/pitchgpt_v2_calibration.json` (assembled from existing `results/pitchgpt/2025_holdout/...` artifacts; Agent 2 writes if absent — do NOT re-fit).

### 10.2 Files this plan MUST NOT touch

| Path | Reason |
|---|---|
| `models/pitchgpt_v2.pt` | LOCKED. Backbone byte-identity invariant. SHA256 `6f952054…62883c`. |
| `models/pitchgpt_v2_outcomehead.pt` | LOCKED. Phase 0.3 FAIL artifact preserved for replay. |
| `models/pitchgpt_v2_outcomehead_a1.pt` | LOCKED. Production A1 checkpoint. Read-only. |
| `models/pitchgpt_v1.pt`, `models/pitchgpt_v1_10k.pt`, `models/pitch_lstm_10k.pt`, `models/pitchgpt_v2_smoke.pt` | LOCKED. Reference checkpoints. |
| `src/analytics/pitchgpt.py` | Frozen for 0.5. CONTEXT_DIM, PAD_TOKEN, VOCAB_SIZE, encode_context, context_to_tensor, PitchGPTModel — all referenced by the harness, none modified. |
| `src/analytics/pitchgpt_outcome_head.py` | Frozen for 0.5 per COORDINATION.md ownership table. |
| `src/analytics/pitchgpt_calibration.py` | Frozen for 0.5 (existing utilities sufficient). |
| `scripts/pitchgpt_sampling_fidelity.py`, `scripts/pitchgpt_outcome_a1_concat.py`, `scripts/pitchgpt_outcome_head_smoke.py`, `scripts/pitchgpt_2025_holdout.py` | Read-only references for the harness implementation. |
| `docs/pitchgpt_sim_engine/SIM_ENGINE_API.md`, `docs/pitchgpt_sim_engine/EXECUTION_PLAN.md`, `docs/pitchgpt_sim_engine/COORDINATION.md`, `docs/pitchgpt_sim_engine/pa_outcome_head_design.md` | Spec docs. The doc-updates agent owns post-Phase-0.6 status updates; Phase 0.5/0.6 implementing agents do NOT modify them. |
| `docs/NORTH_STAR.md` | Strategy doc. Updated only after Phase 0.6 PASSes by the doc-updates agent. |

### 10.3 Cross-referenced specs

- **API surface:** `docs/pitchgpt_sim_engine/SIM_ENGINE_API.md` (1.0).
- **Phase scope:** `docs/pitchgpt_sim_engine/EXECUTION_PLAN.md` §6.0.5, §6.0.6, §9.
- **Cross-session state:** `docs/pitchgpt_sim_engine/COORDINATION.md` (ownership + locked checkpoints).
- **Outcome-head design rationale:** `docs/pitchgpt_sim_engine/pa_outcome_head_design.md`.
- **A1 metrics:** `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/metrics.json`.
- **A1 verdict summary:** `results/pitchgpt_sim/outcome_baselines_2026_04_25/SUMMARY.md`.
- **Sampling-fidelity narrowing:** `EXECUTION_PLAN.md` §3, §6.0.1, §7.5.

---

*Document author: Claude (session 2026-04-25, planning agent). Plan-only — no code written. Awaits user review before the Phase 0.5 implementing agents fire. Leave unstaged.*
