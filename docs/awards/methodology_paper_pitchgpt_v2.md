# PitchGPT v2: A Calibrated Rollout Engine for MLB Plate Appearances

**Author:** Hunter Briggs
**Status:** v2 OUTLINE - Phase 1 in progress.
**Source of v1:** `docs/awards/methodology_paper_pitchgpt.md` (calibrated next-pitch claim).
**Source of architectural delta:** `docs/pitchgpt_sim_engine/SIM_ENGINE_API.md`,
`docs/pitchgpt_sim_engine/EXECUTION_PLAN.md`.

> **CORRECTION NOTICE (2026-08-10) — read before expanding any stub.** This
> outline's framing "we ship a calibrated **PA-level** distribution generator"
> (§1) does not survive the evidence and may not be written into prose as-is.
> Phase 0.6 closed as **FAIL** and Phase 0.6.2 — the pre-registered attempt to
> make rollout PA-level marginals honest — was **KILLED 2026-08-10 at its
> fit-convergence gate** (`PHASE_0.6.2_PLAN.md` §6 first disjunct: iteration 1
> = 4.418pp, iteration 2 = 2.625pp vs a 1.0pp threshold on the 2023 fit
> cohort; verdict doc `docs/models/pitchgpt_phase062_results.md`; claim
> `pitchgpt_phase062_kill`). The defensible v2 thesis is narrower: a
> **per-pitch** calibrated engine whose PA-level absolute rates FAIL their
> fidelity gates (K%/BB%/HR%) and whose wOBA/PA-length PASSes are permanently
> unearned. Products built on it may be rank/differential only (A1 grades,
> distribution shapes), each carrying the marginal-bias disclosure; no
> absolute PA-level rate may be quoted. The successor design and its failable
> gate suite are pre-registered in
> `docs/pitchgpt_sim_engine/PITCHGPT_V2_SPEC.md`.

---

## Outline orientation

This document is the v2 paper **outline**. Each section carries a 3-5 line
"what goes here" stub for a future paper-finishing agent to expand into
prose. v1 sections that don't change are referenced, not restated.

The v2 paper **narrows** the v1 thesis from "calibrated next-pitch
prediction" to **"a calibrated rollout engine on which pitch-call grades,
matchup sims, and probabilistic projections are built."** The narrowing was
forced by two empirical results (see §3.6 sampling fidelity and §3.7 outcome
head); both are surfaced honestly here, not buried.

---

## §1 - Reframed claim

**STUB.** Restate the v2 thesis in 1 paragraph: PitchGPT v2 is a
**calibrated rollout engine** - not a sequence-superiority claim. It emits
calibrated next-token + 7-class outcome distributions with N-sample CIs on
every emitted quantity. Reference `NORTH_STAR.md` 2026-04-24 narrowing
("PitchGPT 10K scale-verify + sampling-fidelity: LSTM delta collapses, claim
narrows") and the 2026-04-25 commit that locked the production sim engine to
`(PG-v2 backbone) x (PGConcatHeadPredictor)`. The pitch a public observer
can give: "Public next-pitch work is uncalibrated and point-estimate; we
ship a calibrated PA-level distribution generator."

What this paper does NOT claim (also load-bearing):
- NOT "beats LSTM by spec margin on perplexity" (retracted; v1 §6).
- NOT "sequence-aware sampling superior to LSTM" (retracted at 10K scale;
  see §3.6).
- NOT "full PASS on PA-outcome head" - WEAKER PASS only; see §3.7.

---

## §2 - Methodology recap (delta vs v1)

**STUB.** Cover what changed since v1 in <1 page:
1. **Pitcher-disjoint discipline** (already in v1; reaffirm as the
   foundation of all 2025 holdout numbers).
2. **10K-game scale-verify** (new): the 1K-game training that produced v1's
   13.80% LSTM-perplexity-delta did NOT replicate at 10K games (delta
   collapses to a CI that includes zero - see §3.6). This is the
   load-bearing reason for the v1 -> v2 narrowing.
3. **PA-outcome head** (new): a 7-class PA-outcome predictor trained on
   frozen v2 hidden states + concatenated context + pitch-token features
   (the A1 winner of Plan B; see §3.7).
4. **Rollout engine + API** (new): `pitchgpt_sim.rollout()` (Phase 0.5)
   wraps backbone + outcome predictor behind a pluggable
   `OutcomePredictor` protocol. See §4.

Everything from v1 §2 (token vocabulary, 34/35-dim context, training
hyperparameters, calibration via temperature scaling) carries unchanged
into v2.

---

## §3.1 - §3.5 - Inherited from v1

**Reference, do not restate.** v1 paper (`docs/awards/methodology_paper_pitchgpt.md`)
covers:
- §3.1 Spec gate table (5 gates pass; 1 fails honestly).
- §3.2 Calibration result (ECE 0.0098 post-temperature on 2025 holdout).
- §3.3 Per-bin reliability + over/under-confidence diagnosis.
- §3.4 Pitcher-disjoint validation (334 holdout pitchers, no shared IDs).
- §3.5 Token-vocabulary sanity (top-1 vs uniform-prior; honest about the
  2210-class structural-low-accuracy framing).

v2 inherits all of these. The citations in the v2 prose should point at v1
section anchors rather than re-proving the same claims.

---

## §3.6 - Sampling fidelity (Phase 0.1 result)

**STUB.** Honest negative result on sampling fidelity at 10K-game scale.

What goes here:
- Setup: rolled out PG and LSTM 10K samples each at horizon=6 from a
  matched 2025 PA cohort; computed marginal token frequencies (1-gram),
  next-pitch-given-prefix transitions (2-gram, 3-gram), and 2-gram
  Frobenius distance to the empirical distribution.
- Result: PG **wins** the 1-gram, 2-gram (raw token), and 3-gram marginal
  vs the empirical, with 95% bootstrap CIs **excluding zero** (PG <
  empirical-distance < LSTM by ~0.01 to 0.04 nats).
- Result: PG **loses** the 2-gram Frobenius distance to the empirical,
  with the LSTM minus PG delta CI = [0.008, 0.250] - excludes zero in the
  LSTM's favor. This was the regression that forced the v1 -> v2
  narrowing.
- Interpretation: PG's per-token marginals are honest; its tighter
  sequence-pair structure does NOT survive the scale-up. The "sequence
  modeling > LSTM" framing is retracted.

Source artifacts: `results/pitchgpt/scale_verify_10k_2026_04_18/` (or
the 2026-04-23 reference commit `9330bb7` per git log).

This is presented as an **honest negative** - the paper publishes the
delta-CI rather than burying it.

---

## §3.7 - PA-outcome head (Plan B story arc)

**STUB.** This is the new section that wasn't in v1. The narrative arc is
non-trivial and must be told honestly.

**Phase 0.3 - frozen MLP on hidden state alone (FAIL).**
- Architecture: `128 (hidden) -> 64 -> 7`, frozen v2 backbone.
- Result: -5.34% lift vs frequency prior on 2023 val. Worse than the
  prior; head failed to extract signal.
- Misdiagnosis (initial): "the backbone's hidden state lacks outcome
  information." This was wrong - see Plan B Step 2.
- Source: `models/pitchgpt_v2_outcomehead.pt` (preserved untouched for
  replay-ability of the FAIL artifact); SUMMARY.md "Comparison vs Phase
  0.3 PG-frozen-MLP failure" section.

**Plan B baselines (Step 1).**
- A3 XGBoost on engineered features (no backbone): **+16.12% lift**, ECE
  0.0181, hbp ll 3.57. WEAKER PASS.
- A4 logistic regression on engineered features: **+17.35% lift**, ECE
  0.0264, hbp ll 4.91. WEAKER PASS.
- A5 empirical PA-terminal lookup (kill-criterion safe-harbor): **+4.33%
  lift**. FAIL (below the +5% kill threshold).
- Source:
  `results/pitchgpt_sim/outcome_baselines_2026_04_25/{a3_xgboost,a4_logistic,a5_empirical}/`.

**Plan B Step 2 - A1 (frozen v2 backbone + concat-input MLP head).**
- Architecture: `concat(hidden[128] + context[35] + pitch_type_oh[17] +
  zone_oh[26] + velo_oh[5]) = 211d -> 128 -> 64 -> 7` MLP, ReLU + dropout
  0.1, weighted CE (inverse-frequency capped at 10x), 5 epochs (best at
  epoch 3), ~2 min on RTX 3050.
- Result: **+18.31% lift** (CI [+18.10%, +18.53%]) on 2025 pitcher-disjoint
  holdout (n=204,513). Log-loss 1.3507. ECE post-T 0.0114 (well under
  the 0.05 outcome-predictor budget).
- Per-class log-loss: A1 **wins 5 of 7 classes** (ball, swinging_strike,
  foul, in_play_out, hbp). HBP gap is the largest per-class win (3.02 vs
  A3's 3.57); first variant in study under the 4.0 PASS threshold.
- Per-pitcher stability (top-50): A1 mean 1.346, var 0.0010, range
  [1.27, 1.40]. Best in study; tightens A3's 1.91 outlier.

**A1 vs A3 paired bootstrap (locked headline question).**
- Paired on 204,513-row intersection (matched via game_pk + at_bat +
  pitch_number).
- A1 - A3 lift delta: **+2.48 pp** (CI [+2.24, +2.72]) - clears the +1pp
  ship threshold by ~2x; CI excludes zero by ~22 SE.
- Decision: SHIP A1.

**Narrative correction.** The Phase 0.3 -5.34% diagnosis as
"head-capacity issue, not backbone-information issue" is the load-bearing
correction. With the same frozen backbone, an extra MLP layer and the
just-sampled pitch token + context as concat inputs, lift swings from
-5.34% to +18.31% - a >23pp swing on the same backbone hidden states.
The backbone DOES carry outcome-discriminative information; the original
head simply lacked the capacity (and the pitch-token features) to
decode it.

**Honest caveat - WEAKER PASS, not full PASS.**
- A1 misses one PASS gate: `in_play_hit` log-loss is **2.34 > 2.0** (the
  spec floor). It clears WEAKER PASS at <2.5.
- Root cause: hit-vs-out distinction at pitch-decision time has a
  **structural ceiling** because `launch_speed` and `launch_angle` (the
  exit-velocity / launch-angle features that decide hit-vs-out) are
  observable only **after** contact - no architecture in this study has
  access to them.
- Inheritance: **all Tier-A consumers (A1 grades, A2 projections, A3
  matchup sims) inherit this ceiling.** The disclosure is non-negotiable
  per `SIM_ENGINE_API.md` §4 ("`in_play_hit` ceiling - load-bearing
  disclosure"). Tier-A artifacts MUST flag any wOBA / PA-outcome
  aggregation that depends on this distinction.

**Source artifacts.**
- `results/pitchgpt_sim/outcome_baselines_2026_04_25/SUMMARY.md` (Plan B
  Final ship verdict).
- `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/{metrics.json,
  report.md, train.log}`.
- `models/pitchgpt_v2_outcomehead_a1.pt` (~38 KB, 28K head params).
- `scripts/pitchgpt_outcome_a1_concat.py` (training script).

---

## §4 - Sim engine architecture

**STUB - reference, then 1-2 paragraphs of new content.**

Reference `docs/pitchgpt_sim_engine/SIM_ENGINE_API.md` §3 for the locked
contract: `rollout()` entry point, `PAContext` + `RolloutResult`
dataclasses, `OutcomePredictor` protocol with 4 concrete implementations
(`PGConcatHeadPredictor` (PRODUCTION), `XGBoostOutcomePredictor`,
`PGFrozenHeadPredictor` (DEPRECATED), `EmpiricalPATerminalLookup` (kill
safe-harbor)).

What's worth a paragraph here that isn't in the API doc:
- **Why a pluggable protocol matters for the v2 paper claim.** The
  protocol decouples backbone research from outcome-predictor research
  from consumer code. v2's "calibrated rollout engine" claim is robust
  even if Plan A re-opens (e.g., a wider backbone wins) - the consumer
  code doesn't change, only the registered predictor swaps. See API §10
  (the 12-cell Plan A x Plan B compatibility matrix).
- **Aggregation utilities live in `pitchgpt_sim.py`, not consumers.**
  Per-rollout reductions (`pa_woba_distribution`,
  `percentile_of_actual_outcome`, `outcome_marginal`,
  `pitch_token_marginal`) live in the engine; per-pitcher / per-game /
  per-season aggregations live in consumer modules (
  `pitchgpt_projection.py`, `pitchgpt_matchup.py`, etc.). This boundary
  is what keeps the engine claim narrow and consumer claims separately
  validatable.

---

## §5 - Calibration validity contract

**STUB - reference + 1-paragraph of pitch.**

Reference `SIM_ENGINE_API.md` §6: `calibration_valid` flag is True iff all
four conditions hold (T = 1.0; backbone calibration.json current; outcome
predictor calibration.json current; starting context in-distribution per
the per-feature 1st-99th percentile gate).

The pitch for the public reader: **PitchGPT v2 never silently emits an
uncalibrated probability.** Every rollout carries a metadata flag stating
whether the calibration regime is honored. Downstream artifacts (Tier-A
products, the dashboard, future API consumers) MUST surface this flag and
degrade gracefully when it is False. This is what makes the v2 claim
defensible in adversarial review - a reviewer cannot point at any Tier-A
output and ask "is this calibration claim load-bearing here?" without
getting a yes/no answer in metadata.

---

## §6 - Tier-A results (placeholder)

**STUB - Phase 1 in progress.**

Each of the three Tier-A dossiers will populate one sub-section:
- §6.1 A1 - Counterfactual pitch-call grades (status: cohort frozen
  2026-04-26 at `results/a1_eval_cohort_2026_04_26/`; Phase 1 grader not
  yet run).
- §6.2 A2 - Probabilistic pitcher projections with CIs (status: scaffolding
  not yet started; depends on A1 and A3 landing first).
- §6.3 A3 - Matchup sim with CIs (status: cohort frozen 2026-04-26 at
  `results/a3_matchup_cohort_2026_04_26/`; Phase 1 simulator not yet run.
  Note schema-mismatch in cohort manifest: 2025-only ge50 band is empty,
  multi-season aggregation needed for Gate 2 backtest).

When Phase 1 results land, each dossier section reports:
1. Cohort details (n pitchers / pairs / pitches).
2. Headline metric (rank correlation, calibration scatter, etc.) with
   95% CI.
3. Verdict (PASS / WEAKER PASS / FAIL) per `EXECUTION_PLAN.md` §6 gate
   spec.
4. Honest caveats (e.g., A1 grade reliability for in-play balls is
   bounded by the §3.7 ceiling; A3 ge50 band only populated at
   multi-season scale).

---

## §7 - Limitations

**STUB.** Three load-bearing limitations:

1. **`in_play_hit` ceiling.** Detailed in §3.7. All Tier-A consumers
   inherit. Surfaced in the dashboard via the `in_play_hit_share`
   metric on matchup sim and an explicit warning on pitch-call grades
   that hinge on hit/out distinction.

2. **ABS-era umpire drift.** PitchGPT v2 conditions on a continuous
   `umpire_scalar` (mean accuracy_above_x_wmean per HP umpire). Any
   downstream consumer that emits an **edge product** keyed on this
   feature is gated on an ABS drift check
   (`feedback_no_umpire_edges_until_abs_drift_check.md`) - 2026+
   automated-ball-strike rule changes invalidate historical umpire
   tendencies. The engine itself accepts the scalar; the gate lives in
   consumer code.

3. **OOS extrapolation beyond 2025.** Calibration is fit on a 2-year-prior
   pitcher-disjoint slice and measured on the most-recently-completed
   season. Re-measurement protocol (`SIM_ENGINE_API.md` §7) fires
   annually every January. Any rollout for a season strictly later than
   the calibration's `holdout_season + 1` is automatically marked
   `calibration_valid = False`. This is intentional - the paper's
   "calibrated rollout" claim is bounded to the calibrated regime.

---

## §8 - Reproducibility

**STUB.** Single-paragraph map of what's reproducible and where:

- **Backbone checkpoint:** `models/pitchgpt_v2.pt`, SHA256
  `6f952054d14ac6f918f3eb9502b496b70bc0c87dfc65dc50d98ee7244a62883c`.
  Frozen; never overwritten. Re-built via `scripts/pitchgpt_train.py`.
- **Outcome predictor (A1) checkpoint:** `models/pitchgpt_v2_outcomehead_a1.pt`,
  ~38 KB, 28K head params. Trained via
  `scripts/pitchgpt_outcome_a1_concat.py`.
- **Calibration JSON:** schema in `SIM_ENGINE_API.md` §7.3. Files:
  - `models/pitchgpt_v2_calibration.json` (backbone, T = 1.1201, ECE_post
    = 0.0098 on 2025).
  - `models/pitchgpt_v2_outcomehead_a1_calibration.json` (A1 head, T =
    0.8003, ECE_post = 0.0114 on 2025).
- **Eval-cohort manifests** (this paper's load-bearing artifacts):
  - A1: `results/a1_eval_cohort_2026_04_26/manifest.json` - 2025 H1
    grade-computation cohort + 2025 H2 calibration-check cohort.
  - A3: `results/a3_matchup_cohort_2026_04_26/manifest.json` - 2025
    matchup-pair bands (ge50 / 20-49 / <20) per Gate 2/3.
- **Tier-A consumer code:**
  - `src/dashboard/views/pitch_call_grades.py` (A1 view; uses
    `src/dashboard/fixtures/pitch_call_grades_fixture.py` until Phase 0.5
    lands).
  - `src/dashboard/views/matchup_sim.py` (A3 view; uses
    `src/dashboard/fixtures/matchup_sim_fixture.py`).

A reviewer can replay the whole pipeline by:
1. Re-load `models/pitchgpt_v2.pt` + the A1 head + calibration JSONs.
2. Re-fetch the cohort row-IDs from the manifest parquets.
3. Run `pitchgpt_sim.rollout(...)` (Phase 0.5) on those rows.
4. Aggregate per the §5 utilities.

---

*Outline author: Claude (session 2026-04-25 scaffolding agent). Awaits a
Phase-1 paper-finishing agent to fill prose. Leave unstaged.*
