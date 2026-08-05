# Phase 0.6.2 — Rollout-regime per-position class-marginal calibration

**Status:** PLANNED (next session). Written 2026-08-04 after the Phase 0.6.1 A/B verdict.
**Owner:** 1 implementation agent + orchestrator for GPU/DB runs.
**Prereq reading:** `COORDINATION.md` (2026-08-04 entries), `results/pitchgpt/rollout_sanity_2025{,_nocc}/report.md`, `scripts/pitchgpt_build_pos0_calibration.py`, `PGConcatHeadPredictor` in `src/analytics/pitchgpt_sim.py`.

## 1. Objective and the claim at stake

Close the Phase 0.6 PA-level marginal-fidelity gates (K%, BB%, HR%) on the 2025 pitcher-disjoint
holdout, **or** terminate the effort permanently under the kill criterion. The flagship claim at
stake: "calibrated rollout engine matches empirical PA-level outcome marginals." Per-pitch top-1
ECE (0.0114) is NOT at stake and must not be perturbed (see §4 mode-scoping).

## 2. Current state (post 0.6.1, 2026-08-04)

- Mid-PA count mutation is live and tested (verified in `6111cd6`, tests added in `ed1d4dd`).
- Sanity A/B on 2025 (10K PA × 100 samples, seed 42):
  class_calibration ON → K% +11.6pp / BB% +3.0pp / HR% −0.8pp FAIL, wOBA + PA-length PASS.
  class_calibration OFF → strictly worse (BB% crashes to 0.038, wOBA flips FAIL).
- Root cause of residual K% surplus: flat per-position strike response in the self-generated
  rollout regime (CS ~0.28–0.33 across positions vs empirical collapse 0.290 → 0.045 by pos 5).
  The teacher-forced `class_calibration` amplifies this when applied inside rollouts.
- **Discovered taint (remediation required):** `calibration_class_marginal_pos0.npz` was fit ON
  the 2025 pitcher-disjoint TEST cohort (see its docstring, "Cohort" section). The current 2025
  wOBA/PA-length PASSes were therefore partly bought with holdout-fitted weights. This violates
  the platform's no-fit-on-holdout rule (the VWR lesson). Phase 0.6.2 must replace it with a
  2023-fitted equivalent and re-establish ALL 2025 gate results clean.

## 3. Design

Replace the stacked corrections (`class_calibration` all-positions + `pos0` vector) **in rollout
mode only** with a single per-position table:

```
W ∈ R^{H×7},  H = 6 (rollout horizon), 7 outcome classes
p_i(pos) ← softmax_T(z)_i · W[pos, i] / Σ_j softmax_T(z)_j · W[pos, j]
```

- Position 0's row generalizes (and supersedes) the pos-0 vector; later rows correct the
  per-position drift the diagnostics quantified (KL rises monotonically with position, r=0.822).
- **Mode-scoping:** teacher-forced / per-pitch scoring paths keep the existing JSON
  `class_calibration` unchanged — the per-pitch marginal-bias fix and the ECE 0.0114 claim are
  untouched by construction. Only `rollout()` opts into W via an explicit predictor flag
  (default preserves current behavior until the orchestrator flips it for evaluation).
- Old artifacts (`calibration_class_marginal_pos0.npz`, JSON `class_calibration`) stay on disk
  for replay; nothing is deleted or overwritten. New artifact:
  `models/calibration_rollout_perpos.npz` (**must be committed with its producing script in the
  same commit** — lesson of the pos-0 dangling-reference incident).

## 4. Fit procedure (2023 only — 2025 is never read)

1. **Cohort:** 2023 pitcher-disjoint validation cohort, same recipe as A1's temperature fit
   (pitchers in the 2015–2022 train split EXCLUDED; reuse the
   `scripts/pitchgpt_outcome_a1_concat` cohort machinery). Subsample 10K PAs, seed 42.
2. **Empirical target:** per-position class marginals from real 2023 PA sequences
   (within-PA pitch index 1..6 → positions 0..5, truncated at horizon).
3. **Rollout marginals:** run `rollout()` on the 2023 cohort with raw T-softmax
   (no class_calibration, no pos0 — the W fit must act on the uncorrected distribution).
4. **Fit:** `W[pos, c] = empirical[pos, c] / rollout[pos, c]`, per-row geometric-mean
   normalized, floor/cap ratios at [0.2, 5.0] to prevent degenerate reweighting of rare classes
   (HBP). Guard: any class with < 500 rollout observations at a position inherits W = 1.0.
5. **Fixed-point iteration (feedback correction):** applying W changes sampled outcomes → count
   trajectories → marginals. Re-roll the 2023 cohort WITH W applied, re-measure, update
   `W ← W · (empirical / rollout_W)` once. **Maximum 2 iterations total.** Converged when every
   2023 per-position class marginal is within 1pp of empirical; if iteration 2 does not converge
   on 2023, that itself is a kill signal (§6) — do not add iterations.

## 5. Evaluation (the only 2025 contact)

- Single production sanity run: `scripts/pitchgpt_rollout_sanity_2025.py` with W enabled
  (10K PA × 100 samples, seed 42, defaults otherwise). One run. No peeking-and-refitting.
- **PASS requires all five gates green:** K%, BB%, HR% within their CI bands AND wOBA +
  PA-length staying PASS *under the clean fit* (they must be re-earned without the 2025-fitted
  pos0 vector — if they regress, that is reported as-is).
- Also run the sanity harness on 2023 (fit-regime sanity) so the results doc can show
  fit-cohort vs holdout transfer explicitly.
- Regression checks: full `tests/test_pitchgpt_sim.py`; SHA256 asserts on `pitchgpt_v2.pt` and
  `pitchgpt_v2_outcomehead_a1.pt` pre/post (no checkpoint mutation, ever).

## 6. Kill criterion (hard — pre-registered)

Stop **permanently** and close Phase 0.6 as FAIL if ANY of:
- The 2023 fit does not converge within 2 fixed-point iterations, or
- The single 2025 evaluation run leaves any of K%/BB%/HR% outside its gate band, or
- wOBA or PA-length regress to FAIL under the clean fit and cannot be attributed to removing
  the tainted pos-0 vector (attribution = one diagnostic comparison run, no refitting).

On kill: the flagship claim stays permanently narrowed to "per-pitch calibrated rollout engine"
(ECE-based); PA-level absolute-rate products (A3 matchup K%/BB% displays) are dropped from
Tier-A scope; rank/differential products (A1 grades, A2 projection *distribution shapes*)
proceed with the marginal-bias disclosure. No third calibration layer, no backbone/head
retraining, no capacity increase — those are Plan-A-shaped moves that were already retired.

## 7. Tickets

| # | Ticket | Est | Depends |
|---|---|---|---|
| 1 | Parameterize cohort season in the sanity/rollout harness (extract `HOLDOUT_SEASON=2025` hard-coding into a `--season` arg + cohort builder reuse for 2023) | 0.5 d | — |
| 2 | Fit script `scripts/pitchgpt_fit_rollout_calibration.py` (§4; `--dry-run`, audit JSON, read-only DB) | 0.5 d | 1 |
| 3 | `PGConcatHeadPredictor` per-position W path + rollout-mode flag + tests (state-machine untouched) | 0.5 d | — |
| 4 | GPU runs: fit (≤2 × ~7 min) + 2023 sanity + single 2025 evaluation (~10 min) — orchestrator only | ~0.5 h GPU | 1–3 |
| 5 | Results doc + COORDINATION update + NORTH_STAR claim update (either direction) + commit artifacts | 0.25 d | 4 |

Single-writer discipline: every step is read-only on DuckDB; stop the dashboard anyway during
GPU runs per hard rule. All scripts runnable end-to-end by the orchestrator; the implementation
agent never opens a writer and never runs the 2025 evaluation.

## 8. Explicitly out of scope

- Retraining A1 or the backbone with different class weights (the "fix it at the source" option
  — rejected: SHA-locked artifacts anchor every published number).
- Fitting on 2024 or 2025 in any form. 2024 stays untouched as a potential future second holdout.
- Iterating past 2 fixed-point rounds or adding per-count / per-pitcher calibration dimensions
  (unbounded knob space = the hack-tuning this plan exists to prevent).
