# Phase 0.6.2 — Rollout-regime per-position class-marginal calibration

**Status:** EXECUTED 2026-08-10 — **KILLED at the §4 fit stage** (§6 first disjunct: 2023 fit
did not converge within 2 fixed-point iterations). See §11 execution record and
`docs/models/pitchgpt_phase062_results.md`. Originally written 2026-08-04 after the
Phase 0.6.1 A/B verdict.
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
  class_calibration ON → K% +11.6pp / BB% +3.0pp / HR% −0.8pp FAIL, wOBA + PA-length PASS
  [TAINTED - pending Phase 0.6.2 re-evaluation (pos-0 calibration was fit on the eval cohort);
  see the taint bullet below].
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

---

## 9. Foreknowledge (appended 2026-08-10, WS2.3 — append-only; sections 1–8 unmodified)

Holdout-contact accounting per `docs/holdout_ledger.jsonl` (created
2026-08-10):

- At this plan's authoring (2026-08-04) the 2025 pitcher-disjoint cohort had
  **12 recorded evaluation contacts** (ledger contacts 1–12; contacts 8–12
  landed on 2026-08-04 itself, including the pos-0 fit-on-holdout this plan
  remediates). Every rollout-family contact used the same seed-42 10K-PA
  subsample.
- The section 5 evaluation run is **contact #13 of a budget of 14** and MUST
  be appended to the ledger via `src/holdout.py` (`@holdout_access` or
  `record_contact`) at run time, before the results doc is committed (see the
  ledger header's `todo` field). The section 6 attribution diagnostic, if
  triggered, is contact #14. The budget then closes: any further 2025 contact
  requires a logged override plus a dated spec amendment.
- 2024 remains the burned dev tier (see plan section 0 corrections in
  `docs/plans/2026-08-10_platform_improvement_plan.md`: three committed
  scripts used `TEST_RANGE=(2024,2024)`, so section 8's "potential future
  second holdout" framing is superseded — the sealed holdout is 2026).
- The 2026 full-season lockbox stays sealed; nothing in Phase 0.6.2 touches it.

---

## 10. Amendments — 2026-08-10 (WS5.0, pre-execution; append-only deviations log)

Recorded BEFORE any Phase 0.6.2 fit or evaluation run, per
`docs/plans/2026-08-10_platform_improvement_plan.md` §5.0. Sections 1–8 remain
verbatim; where an amendment touches a section-5 parameter it says so
explicitly. The section-6 kill criterion is NOT amended — it stands as written.

**A1 — Full-cohort evaluation (amends §5 sample size only).** The single 2025
evaluation runs on the FULL 2025 pitcher-disjoint PA-start cohort
(**64,460 PAs**, all eligible), not the seed-42 10K subsample — every one of
the 12 historical rollout-family contacts reused that same subsample (audit
F-C), so the verdict number must come from the full cohort. All other §5
parameters unchanged: 100 samples/PA, horizon 6, T=1.0, seed 42, secondary
None-predictor bias run included. Gate bands unchanged (empirical baseline was
always full-cohort). Measured cost basis: 10K PA = 404 s primary + 202 s
secondary on the RTX 3050 ⇒ ~65 min expected.

**A2 — Production-path ECE measured in the same run.** The shipped-probability
per-pitch top-1 ECE has never been measured post-stack (audit §3 finding 3).
The evaluation run additionally computes teacher-forced 10-bin top-1 ECE on
the locked A1 test-cohort recipe (2025 pitcher-disjoint, 2,000 games, seed
44 = DEFAULT_SEED+2, `FixedGamesSequenceDataset` + `collect_logits`) under
four stacks: (a) post-T only — must reproduce the locked 0.0114 within noise;
(b) post-T + `class_calibration` (teacher-forced production stack);
(c) post-T + `class_calibration` + tainted pos-0 vector (the stack actually
shipped in rollout mode pre-0.6.2 — recorded for the audit record only);
(d) rollout-regime stack post-T + W[pos] on within-PA positions 0–5 (the
0.6.2 replacement as shipped in rollout mode). Reported with bootstrap CIs;
no new gate is created by this amendment — the numbers publish either way.

**A3 — Holdout-ledger registration is part of the run.** The §5 evaluation is
**contact #13** for `pitchgpt_2025_pitcher_disjoint` and is appended via
`src.holdout` wired into the harness itself: budget enforced BEFORE the eval
executes, contact entry appended at completion, before any results doc is
written. The A2 ECE measurement rides the SAME contact (one run, one contact)
with its metrics listed in `metrics_revealed`. The §6 attribution diagnostic,
if triggered, is contact #14. No other 2025 contact is authorized.

**A4 — Test remediation + provenance guard (K5 enforcement).** The test
enshrining the tainted pos-0 npz as a required production artifact
(`tests/test_pitchgpt_sim.py::test_disable_class_calibration_flag_drops_class_cal_keeps_pos0`)
is replaced. New provenance-guard tests enforce: calibration artifacts must
declare their fit cohort (sidecar schema extension: the W npz carries
`fit_cohort_season`, `fit_seed`, `fit_n_pas`, `n_iterations`, `converged`,
`produced_by`), and `PGConcatHeadPredictor` structurally REFUSES to load a
rollout-regime W artifact whose declared `fit_cohort_season` is 2025 or 2026
(the budgeted/lockbox eval tiers). The tainted
`calibration_class_marginal_pos0.npz` (declares `cohort_season=2025`) stays on
disk for replay per §3 but is bypassed by construction in the rollout-regime W
path.

**A5 — Legacy landmine removal.** `train_pitchgpt` default
`val_seasons=[2025, 2026]` (`src/analytics/pitchgpt.py`) changes to the
dev-safe `[2024]`; passing 2025 or 2026 now requires an explicit
`allow_holdout_val=True` opt-in (and remains subject to the ledger). This is a
default-safety change only; no training runs in Phase 0.6.2.

**A6 — Registry registration.** The PitchGPT calibration artifacts are
registered as write-once registry versions (pointer form, no alias changes):
`models/pitchgpt_v2_calibration.json` (backbone T),
`models/calibration_pitchgpt_v2_outcomehead_a1.json` (A1 head T +
class_calibration, fit 2023 val), and the new
`models/calibration_rollout_perpos.npz` once produced.

**A7 — §4 fixed-point measurement semantics (operationalization, not a
change).** "Iteration" counts W updates: iteration 1 = the initial ratio fit
from the raw-T roll; iteration 2 = the single feedback update. Convergence of
a W is adjudicated on a measurement re-roll WITH that W applied. GPU passes
are therefore: roll-0 (raw T) → W₁; roll-1 (W₁) → if every 2023 per-position
class marginal is within 1pp of empirical, converged at iteration 1; else
W₂ = W₁·(empirical/roll-1), roll-2 (W₂) → within 1pp ⇒ converged at iteration
2, otherwise the §6 kill trigger fires. No third update, no third re-roll.
Ratio guard order per §4: <500-observation guard (W=1.0), then floor/cap
[0.2, 5.0], then per-row geometric-mean normalization (cosmetic — the
renormalize-after-multiply application is scale-invariant); the iteration-2
update factor is capped the same way and the combined W re-capped.

**A8 — Output isolation.** The committed 0.6.1 artifacts under
`results/pitchgpt/rollout_sanity_2025/` are never overwritten. Phase 0.6.2
writes: fit audit → `results/pitchgpt/rollout_calibration_fit_2023/`;
2023 fit-regime sanity → `results/pitchgpt/rollout_sanity_2023_phase062/`;
the single 2025 evaluation → `results/pitchgpt/rollout_sanity_2025_phase062/`.
The harness gains a `--season` argument (Ticket 1); for season 2025 the
HR-given-hit constant keeps its locked source
(`rollout_sanity_2025/empirical_baselines_2025.json`, hit% 0.22177); for the
2023 fit-regime sanity the analogous hit% is computed in-run from the 2023
cohort (that file is a 2025-only referent).

---

## 11. Execution record — 2026-08-10 (append-only; §§1–8 remain verbatim)

Executed per §4 + §10.A7 by `scripts/pitchgpt_fit_rollout_calibration.py` (2023
pitcher-disjoint cohort, 19,625 eligible PAs, 10K sampled seed 42, 100 samples/PA,
horizon 6, raw-T roll-0 → W₁ → roll-1 → W₂ → roll-2).

**Outcome: KILL (§6, first disjunct).** Convergence required every 2023 per-position
class marginal within 1.0pp of empirical within 2 iterations; measured max |delta|:
roll-0 (raw T) 16.37pp → iteration 1 (W₁) 4.418pp → iteration 2 (W₂) 2.625pp. Phase 0.6
closes as FAIL. No artifact shipped to `models/`; the non-converged W is quarantined at
`results/pitchgpt/rollout_calibration_fit_2023/W_FAILED_FIT_quarantine.npz`
(sha256 `395e6fcd16b188f58a9fc124c5ac33fded15fb8946e137a95310c5e931b27d12`).

Consequences applied exactly as pre-registered: the §5 single 2025 evaluation never ran;
holdout contact #13 was not spent (ledger note, 2026-08-10; 2025 budget stands 12/14);
the §10.A2 production-path ECE is therefore unmeasured; the §6 attribution diagnostic
(#14) was not triggered; §10.A6 W registration is n/a. Checkpoint SHAs byte-identical
pre/post; DuckDB read-only throughout. Full audit:
`results/pitchgpt/rollout_calibration_fit_2023/fit_audit.json`; verdict doc:
`docs/models/pitchgpt_phase062_results.md`.
