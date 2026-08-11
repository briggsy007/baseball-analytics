# PitchGPT v2 Retrain — Architecture, Training and Gate Pre-Registration (WS5.2 + WS5.3)

**Status: FROZEN ON COMMIT.** The git commit that introduces this file is the tamper-evident
freeze. After that commit, §0–§8 of this document are immutable: corrections, clarifications,
threshold changes and protocol amendments happen exclusively as dated, append-only entries in the
Deviations Log (§9). **No training run of any kind may start before that freeze commit exists**, and
every run must cite the freeze SHA in its audit JSON.

**Authored:** 2026-08-10, per `docs/plans/2026-08-10_platform_improvement_plan.md` §5.2 + §5.3,
after the user adjudication of the same date ("K5 FIRED; the WS5.2 v2 retrain is green-lit with its
spec pre-registered BEFORE any training").
**Supersedes as the forward path:** `PHASE_0.6.2_PLAN.md` (executed and KILLED 2026-08-10; §§1–8 of
that document remain frozen verbatim and are not edited by this one).
**Governs:** kill criterion **K5** — *"A v2 retrain gets ONE lockbox contact against the sealed 2026
holdout per its pre-registered spec; failure locks the per-pitch-only claim for the season. No
calibration vector may ever be fit on a cohort that any gate is evaluated on (provenance-guard test
enforces)."* This document is that pre-registered spec.

**Structure** follows the OSF pre-registration template used by
`docs/models/contrarian_2026_resolution_spec.md`: Foreknowledge (§1), Hypotheses (§2), then the
executable protocol (§3 architecture, §4 training, §5 data policy), the analysis/gate plan (§6),
kill criteria (§7), artifacts and provenance (§8), Deviations Log (§9).

**Executability standard.** A person with the DuckDB database, this document, and the artifacts it
names must be able to build, train, calibrate and grade v2 with **no judgment calls**. Every place a
judgment call was unavoidable at authoring time, the choice is pre-registered inline and marked
**[FIXED]** with its derivation. Where a number is inherited from an existing document it is quoted
with its source; **no threshold in this spec was invented without a stated anchor**, and none may be
softened after any result exists.

---

## 0. Scope, and what this spec does NOT authorize

In scope: one factorized-head backbone retrain, one outcome-head retrain on that backbone, the
calibrator fits that ride them, a dev-tier evaluation program on 2024, and **exactly one** sealed-2026
lockbox evaluation graded by §6.

Explicitly NOT authorized by this document:

1. **Any contact with the 2025 pitcher-disjoint budgeted tier.** 12 of 14 contacts are used
   (`docs/holdout_ledger.jsonl`); the 2 remaining are reserved for the purposes named in §5.4 and
   require their own dated amendment. v2 work never touches 2025.
2. **Any post-hoc output-reweighting layer** (per-position W tables, class-marginal vectors,
   per-count calibration matrices). That family is retired: it is what Phase 0.6/0.6.1/0.6.2 tried
   and it is what the research verdict rejected in advance (plan WS5 header; arXiv 2411.02988 on
   post-hoc multiclass calibration overfitting calibration sets at high class counts). Temperature
   scaling per head (a single scalar) is the ONLY post-hoc calibration permitted, and only under §4.4.
3. **A second lockbox contact.** One per frozen spec version, per K5 and the ledger tier policy.
4. **Overwriting any existing artifact.** `models/pitchgpt_v2.pt` (sha256 `6f952054...62883c`) and
   `models/pitchgpt_v2_outcomehead_a1.pt` (sha256 `37b50e87...bb25e5`) are registry-pinned and
   byte-frozen; v2 writes new paths only (§8.1).
5. **Any claim on a dashboard or in a doc without a claims-registry entry** (K6).

---

## 1. Foreknowledge declaration

Stated plainly, per OSF norms. The author of this spec is not naive about the outcomes it will be
graded on.

### 1.1 The 0.6.2 kill is known at authoring

Phase 0.6.2 was executed and **KILLED on 2026-08-10**, hours before this spec was written, at its
pre-registered fit-convergence gate (`PHASE_0.6.2_PLAN.md` §6, first disjunct). Measured: max
per-position class-marginal |delta| vs empirical on the 2023 fit cohort = **16.37pp** (raw-T
reference roll) -> **4.418pp** (iteration 1) -> **2.625pp** (iteration 2), against a **1.0pp**
threshold. Verdict doc `docs/models/pitchgpt_phase062_results.md`; claim `pitchgpt_phase062_kill`.
This spec exists because that kill fired, and §6/§7 deliberately reuse the 1.0pp quantity as a gate
so that v2 is graded on the exact number its predecessor died on.

### 1.2 Holdout-contact history

`pitchgpt_2025_pitcher_disjoint` carries **12 recorded contacts** (ledger contacts 1-12: v1/v2 at 1K
and 10K game scale, sampling fidelity, the A1 outcome-head evaluation, the initial Phase 0.5/0.6
rollout sanity run, the D1 rollout diagnostics, the pos-0 calibration **fit on the holdout itself**,
the clean-provenance sanity re-run, and both arms of the 0.6.1 A/B). Every rollout-family contact
reused the same seed-42 10K-PA subsample (audit F-C). Budget is 14; contact #13 was reserved for the
0.6.2 evaluation and was **never spent** because the fit gate killed the phase first; #14 (the
attribution diagnostic) was never triggered. 2024 is the **BURNED dev tier** — three committed
scripts used `TEST_RANGE=(2024,2024)` — and therefore carries no validation authority and no budget.
2026 full season is the **sealed lockbox**, unsealed only by an appended `entry_type=unseal` line
after the final 2026 regular-season game.

### 1.3 The author has seen every 2025 diagnostic

Non-exhaustive list of results already in hand at authoring, all of which could bias a "clever"
design toward the known failure modes:

- PA-level rollout marginals on 2025 (10K PA x 100 samples, seed 42, class_calibration ON, the
  production default): K% 0.3339 vs empirical 0.218 FAIL; BB% 0.1177 vs 0.0876 FAIL; HR% 0.0242 vs
  0.0321 FAIL (claim `pitchgpt_pa_rates_fail`). With class_calibration OFF: strictly worse
  (BB% collapses to 0.0376, wOBA flips to FAIL).
- wOBA and PA-length "PASSes" from those runs are retracted as tainted and, post-kill, permanently
  unearned (claim `pitchgpt_woba_pa_pass_pre062`).
- Per-pitch post-temperature top-1 ECE 0.0090 (backbone) / 0.0114 (A1 outcome head, T=0.8003), with
  the production-path ECE never measured and now stranded (claim `pitchgpt_per_pitch_ece`).
- Root-cause diagnostics: the A1 head's inverse-frequency CE class weights (cap 10x,
  `scripts/pitchgpt_outcome_a1_concat.py:583-596`) pull mass out of `ball` — predicted per-pitch
  ball share 24.5% vs 36.1% empirical (-11.6pp), strike-class share 36.5% vs 26.7% (+9.8pp)
  (`PHASE_0.6_DIAGNOSIS.md` §6.1). Called-strike rate is flat across rollout positions (~0.28-0.33)
  where the empirical rate collapses 0.290 -> 0.045 by position 5. Per-position KL from empirical
  rises monotonically with within-PA position, r = 0.822.
- The 0.6.2 residual after two reweighting iterations is a **ball-marginal deficit at positions
  0-4**; the reweighting is partially undone by the count-trajectory feedback loop. This is the
  exposure-bias signature, and it is the specific thing §4.3 exists to attack at the source.
- Empirical 2025 baselines are locked (`PHASE_0.6_PLAN.md` §3.1, n=64,460 PAs): K% 21.80%,
  BB% 8.76%, HR% 3.21%, HBP% 1.15%, hit% 22.18%, mean wOBA 0.3302, mean PA length 3.886.

### 1.4 Mitigations, not cures

The design below is anchored to prior art chosen by the research track **before** any v2 result
exists (§3.1, §4.3 citations), the gate thresholds are derived mechanically from numbers that
predate this spec (§6.6 derivation table), and the anti-unfailability rule (§6.7) tightens any gate
that the FROZEN v2 stack already passes on dev — computed before the lockbox contact, so a gate
cannot be quietly set where it cannot fail. The author's 2025 knowledge is a reason for those
mechanisms, not a substitute for them.

---

## 2. Hypotheses

- **H1 (representation).** Factorizing the flat 2,210-way composite softmax into the chain
  pitch_type -> zone|type -> velo|type,zone does not cost next-pitch predictive quality
  (teacher-forced NLL) and makes per-head calibration estimable and correctable at class counts
  where calibration statistics are actually well-defined.
- **H2 (exposure bias).** The PA-level marginal failure is a train/inference mismatch, not an
  output-scaling problem. Curriculum rollout-aware fine-tuning (train on the model's own samples)
  reduces the per-position class-marginal drift that per-position output reweighting could not close.
- **H3 (product).** If H1 and H2 hold jointly and the §6 gate suite passes on sealed 2026, PA-level
  absolute-rate products become quotable again for the v3-factorized artifact only — never
  retroactively for the v2-era stack.

Each hypothesis has a paired kill in §7. H3 has no fallback path: failure locks the per-pitch-only
claim for the season (K5).

---

## 3. Architecture (WS5.2, part 1) — chain-rule factorized heads

### 3.1 Prior art the design is copied from

- Sequential sub-token decoding beats both a flat joint vocabulary and parallel independent heads on
  NLL at comparable parameter budgets — Nested Music Transformer, arXiv 2408.01180.
- Chain-rule factorized event heads with per-field legality masks are the working pattern for
  sports-event language models — soccer Large Events Model, arXiv 2402.06820.
- Post-hoc multiclass calibration at high C overfits its calibration set — arXiv 2411.02988. This is
  why the fix is in the head structure, not in another calibration layer.

### 3.2 What stays identical to the frozen v2 backbone

So that the comparison is architecture-vs-architecture and not scale-vs-scale, the following are
**[FIXED]** at the frozen v2 values (`src/analytics/pitchgpt.py` defaults, verified at authoring):
`d_model = 128`, `n_layers = 4`, `n_heads = 4`, `max_seq_len = 256`, causal upper-triangular mask,
`CONTEXT_DIM = 35` (12 count states + 3 outs + 8 runner states + 2 batter hands + 4 inning buckets
+ 5 score-diff buckets + 1 umpire scalar), context projection into `d_model`, `PAD_TOKEN` /
`BOS_TOKEN` handling, optimizer family (AdamW), `lr = 1e-3`, `batch_size = 32`, `grad_clip = 1.0`,
`seed = 42`, and the games budget of the training run that produced `models/pitchgpt_v2.pt`
(`scripts/train_pitchgpt_v2_ump.py`, 10K games).

### 3.3 The factorization

Let `h in R^128` be the backbone hidden state at a position (unchanged). The composite token keeps
its existing arithmetic identity, `token_id = type_idx * 130 + zone_idx * 5 + velo_idx`
(`PitchTokenizer`, `NUM_PITCH_TYPES = 17`, `NUM_ZONES = 26`, `NUM_VELO_BUCKETS = 5`,
`VOCAB_SIZE = 2210`), so any v3 output can be projected back onto the v2 token space for
comparison. The output distribution is decomposed exactly:

```
p(token) = p_type(t | h) * p_zone(z | h, t) * p_velo(v | h, t, z)
```

**[FIXED] head shapes** (no tuning is authorized; a change requires a §9 amendment before the run):

| Head | Inputs | Layers | C | Parameters |
|---|---|---|---:|---:|
| `H_type` | `h` (128) | `Linear(128 -> 17)` | 17 | 2,193 |
| `H_zone` | `concat(h, E_type[t])` (128+32) | `Linear(160 -> 64)` + GELU + `Linear(64 -> 26)` | 26 | 12,538 (incl. `E_type` 17x32 = 544) |
| `H_velo` | `concat(h, E_type[t], E_zone[z])` (128+32+32) | `Linear(192 -> 64)` + GELU + `Linear(64 -> 5)` | 5 | 13,509 (incl. `E_zone` 26x32 = 832) |
| **Total output stack** | | | | **28,240** |

Compare the flat head being replaced: `nn.Linear(128, 2210)` = 282,880 matrix weights + 2,210 bias
= **285,090** parameters. The output stack therefore **drops by ~10x** (285,090 -> 28,240), inside
the plan's stated expectation of "283K output matrix -> ~40K" (§5.2). **[FIXED] hard ceiling:** the
build asserts `output_stack_params < 45,000` and `< 0.25 * 285,090`; a build that violates either
assertion aborts.

Per-head class counts are 17 / 26 / 5, plus 7 for the outcome head (§3.5) — the regime in which
classwise calibration statistics are estimable, versus C = 2,210 with mean top-1 confidence ~5%
where the 0.10 ECE gate was near-unfailable. (The plan's "per-head C ~ 10-26" is the approximate
range; the exact counts are 17/26/5/7 and those are what §6 gates.)

### 3.4 Input embedding: primary and one named ablation

**[FIXED] Primary (the run that ships unless the ablation wins under the §4.5 rule):** the INPUT
token embedding stays flat — `nn.Embedding(TOTAL_VOCAB = 2212, 128)` — so the v3 backbone is
initialization-compatible with `models/pitchgpt_v2.pt` and the two are comparable layer-for-layer.
Only the OUTPUT side factorizes.

**Ablation A-EMB (pre-registered, optional, dev-only):** factorize the input as the sum of three
field embeddings, `E_in_type(17x128) + E_in_zone(26x128) + E_in_velo(5x128)` = 6,144 parameters,
replacing 283,136. It may be run at most once, is judged **only** on 2024 dev teacher-forced NLL,
and may replace the primary only if it improves dev NLL per pitch by >= 1.0% relative. Any other
outcome and the primary ships. No other architectural ablation is authorized.

### 3.5 Outcome head

Shape is **[FIXED] unchanged** from the A1 winner so the only moving parts are the ones under test:
3-layer MLP `211 -> 128 -> 64 -> 7` with ReLU + dropout 0.1 over
`concat(hidden[128], context[35], type_onehot[17], zone_onehot[26], velo_onehot[5])`, 7 outcome
classes `(ball, called_strike, swinging_strike, foul, in_play_out, in_play_hit, hbp)`.

**[FIXED] One deliberate change: NO class weighting.** The A1 head was trained with
inverse-frequency CE class weights capped at 10x, and `PHASE_0.6_DIAGNOSIS.md` §6.1 identifies
exactly that as the class-marginal-bias root cause (predicted `ball` share 24.5% vs 36.1%
empirical), with §9.4 naming "retraining without inverse-frequency weights" as one of the two
principled fixes. v2 trains every head with **unweighted** cross-entropy. Correcting the marginal at
the source is the point; the post-hoc correction family is banned by §0.2.

### 3.6 Per-field sampling masks

**[FIXED]** At generation time, before each field is sampled:

1. `H_type`: mask index 16 (`unknown` pitch type — a data-coverage artifact, never a legal pitch to
   emit). Renormalize over the remaining 16.
2. `H_zone`: mask any `zone` whose training support given the sampled `type` is `< 10` occurrences.
3. `H_velo`: mask any `velo` bucket whose training support given the sampled `(type, zone)` is
   `< 10` occurrences.
4. If masking would empty a field's support, fall back one conditioning level (drop the `zone`
   condition for velo; drop the `type` condition for zone) and record the event in the run's audit
   JSON under `mask_fallbacks`. If the unconditional support is also empty, sample the field
   uniformly over its legal range and count it in `mask_degenerate`.

The support table is computed ONCE from the training split (§5.1), stored inside the checkpoint as
`support_counts`, and hashed into the manifest. Masking is applied identically in training-time
rollout sampling (§4.3) and at inference, so the fine-tuning distribution equals the inference
distribution (this is the whole point of §4.3 and is asserted by a test, §8.3).

---

## 4. Training protocol (WS5.2, part 2)

### 4.1 Data path requirement (non-negotiable)

The **dynamic mid-PA context fix of commit `6111cd6`** stays in the data path for every stage: the
`count_state` one-hot is re-emitted at every within-PA position from the running `(balls, strikes)`
via `_advance_count`, flowing through BOTH the pitch-token backbone forward and the outcome-head
forward via the shared `cur_context`. A static-context training path silently reintroduces the bug
that the 0.6.x saga spent two phases diagnosing. §8.3 registers the test that fails the run if the
context is constant across positions.

### 4.2 Stage A — teacher-forced pretrain

**[FIXED]** Train seasons per §5.1; 5 epochs; AdamW `lr = 1e-3`; batch 32; grad clip 1.0; seed 42;
loss = the unweighted sum of the three field cross-entropies

```
L_A = CE(H_type, t*) + CE(H_zone(., t*), z*) + CE(H_velo(., t*, z*), v*)
```

with ground-truth `t*, z*` used for conditioning (teacher forcing on the fields as well as on the
sequence). Model selection: lowest 2023-slice total loss (§5.2), checkpoint kept per epoch, best
epoch retained. Reference cost: the plan measures a backbone retrain at 8.5 minutes (§5.2), while
`scripts/train_pitchgpt_v2_ump.py` documents the prior 10K-game run at ~4h25m end-to-end; the
discrepancy is not resolved at authoring, so the wall clock actually observed is recorded in §9 and
the compute stop rule of §7.3 governs.

### 4.3 Stage B — curriculum rollout-aware fine-tune

Prior art: multi-step autoregressive fine-tuning over the rollout horizon is the documented recipe
for exactly this compounding-error signature (weather model, arXiv 2604.01215). Two-pass scheduled
sampling (Mihaylova & Martins, ACL 2019) is the **named cheaper fallback** invoked only under §4.3.4.

**[FIXED] curriculum**, starting from the Stage A best checkpoint, same optimizer settings except
`lr = 2e-4` (one fifth of Stage A — a fine-tune, not a retrain):

| Sub-stage | Rollout depth | Epochs | Scheduled-sampling probability eps |
|---|---:|---:|---|
| B1 | 2 steps | 1 | linear ramp 0.00 -> 0.25 across the epoch |
| B2 | 4 steps | 1 | linear ramp 0.25 -> 0.50 across the epoch |
| B3 | full PA (horizon 6) | 1 | fixed 0.50 |

At each rolled position the input token is the model's own sampled token with probability `eps` and
the ground-truth token otherwise; the loss remains cross-entropy against the REAL continuation at
every position, so no non-differentiable reward machinery is introduced. Sampling inside the
curriculum uses the §3.6 masks and the §4.1 dynamic context, i.e. the training-time rollout is the
inference-time rollout.

4.3.1 The PA horizon, termination logic and PAD conventions are the existing ones
(`SIM_ENGINE_API.md` §3.3, horizon 6).
4.3.2 Seeds: cohort sampling seed 42; per-PA rollout seed `42 + pa_index * 1000` (`PHASE_0.6_PLAN.md`
§4.4 convention).
4.3.3 The 2023 fit-cohort diagnostic of §7.2 is computed after B3 and is a kill gate, not a tuning
signal: it may be looked at exactly once per curriculum run.
4.3.4 **Fallback [FIXED]:** if Stage B cannot be run at the specified depth for a mechanical reason
(GPU memory on the 4GB card, or wall clock exceeding §7.3), substitute two-pass scheduled sampling
(a second forward pass over the model's own first-pass predictions) at B3 depth only, record the
substitution in §9 BEFORE the run continues, and grade identically. The fallback is a cost
concession, never a response to a bad number.

### 4.4 Calibration (the only post-hoc step permitted)

**[FIXED]** One temperature scalar per head (`T_type`, `T_zone`, `T_velo`, `T_outcome`), each fit by
NLL minimization on the **2023 pitcher-disjoint slice only** (§5.2). No vectors, no matrices, no
per-position or per-count tables — those are banned by §0.2. Each temperature is written into a
sidecar carrying the provenance schema already enforced by the guard
(`fit_cohort_season`, `fit_seed`, `fit_n_pas`, `produced_by`), and `fit_cohort_season` must be 2023.
The predictor structurally refuses any calibration artifact declaring 2025 or 2026 (K5 second
clause, enforced by `PGConcatHeadPredictor` and `tests/test_pitchgpt_sim.py`).

### 4.5 Run budget for the whole program

**[FIXED]** Stage A: at most 2 attempts (one initial, one repair pass containing exactly one
documented change). Stage B: at most 2 curriculum runs, same rule. Ablation A-EMB: at most 1 run.
Beyond that the program stops (§7.3). Every attempt appends a §9 entry naming the single change.

---

## 5. Data policy and holdout tiers

### 5.1 Training split

**[FIXED]** Seasons **2015-2022**, all pitchers, the same window and games budget as the frozen v2
artifact (`models/pitchgpt/v2026.04.23/manifest.json`: "2015-2022 train / 2023 val / 2024 test
(burned)"). Training data may not exceed 2023 in any case; the primary run does not include 2023
rows at all, so the 2023 slice stays a clean model-selection and calibration cohort.

Pre-registered variant **A-2023 (optional, at most one run):** add the 2023 rows whose `pitcher_id`
IS in the 2015-2022 train set (i.e. explicitly NOT the pitcher-disjoint slice, which stays reserved)
to the training data. Judged only on 2024 dev NLL; may replace the primary only on a >= 1.0%
relative dev-NLL improvement. This is the full extent of the plan's "train <= 2023" latitude.

### 5.2 Model-selection and calibration slice: 2023 pitcher-disjoint

The 2023 cohort built by the existing machinery (pitchers present in the 2015-2022 train split
EXCLUDED; `scripts/pitchgpt_outcome_a1_concat` cohort builder; 19,625 eligible PAs as measured by
the 0.6.2 fit). Used for: Stage A epoch selection, Stage B monitoring, all temperature fits (§4.4),
and the §7.2 fit-stage kill measurement. It is a fit cohort, never a gate cohort.

### 5.3 Dev tier: 2024 (BURNED)

2024 carries no validation authority (three committed scripts trained/tested on it) and no budget.
It is therefore the correct place for unlimited iteration: NLL comparisons, the ablations of §3.4
and §5.1, the anti-unfailability measurement of §6.7, and the KCE power probe of §6.3. Dev-tier
contacts need not be logged. **No 2024 number may be published as a validation result** — dev
numbers appear in the results doc explicitly labelled `tier=burned-dev`.

### 5.4 Budgeted tier: 2025 — NOT TOUCHED

`pitchgpt_2025_pitcher_disjoint` stands at **12 of 14** contacts used. v2 work does not read 2025 in
any form. The 2 remaining contacts are **reserved, and this spec spends neither**:

- **Reserved contact A (would be #13): the stranded production-path per-pitch ECE** of the shipped
  v2-era stack (audit §3 finding 3; 0.6.2 amendment §10.A2, which died with the phase). Spending it
  requires a new dated amendment to `docs/models/pitchgpt_validation_spec.md` naming the exact
  stacks measured, plus a ledger entry. It is a v2-era audit measurement and is not part of the v3
  program.
- **Reserved contact B (would be #14): a single integrity-failure fallback.** If and only if the
  2026 lockbox cohort turns out to be unusable for a data-integrity reason discovered during the one
  contact (e.g. a coverage collapse in a required column that makes the §6 gates uncomputable, not
  merely unfavourable), the platform may fall back to ONE 2025 evaluation of the same §6 gate suite,
  after a dated §9 entry that states the integrity defect and BEFORE any v3 number is looked at
  again. An unfavourable result is never an integrity defect.

Any 2025 contact past 14 requires a logged override plus a dated spec amendment (ledger header rule).

### 5.5 Lockbox: 2026 full season — exactly ONE contact

**[FIXED]** Per K5 and the ledger tier policy: sealed until an `entry_type=unseal` line is appended
after the final 2026 regular-season game, then hash-versioned, then **one** pre-registered contact
per frozen spec version. That contact is the §6 grading run and it is atomic:

- Cohort: **all** eligible 2026 pitcher-disjoint PA starts (pitchers absent from the training split
  of §5.1), full cohort — **no subsample**. The seed-42 10K-PA subsample is banned here: reusing it
  is audit finding F-C and it is precisely what amendment A1 of 0.6.2 removed.
- The run computes the 2026 empirical baselines (same SQL shape as `PHASE_0.6_PLAN.md` §3.3, same
  outcome-class mapping from `classify_pitch_outcome`) and all §6 statistics in a single pass, so
  baseline computation and grading share the one contact.
- It is registered via `src/holdout.py` (`@holdout_access` / `record_contact`) with budget enforced
  BEFORE execution and the contact appended at completion, before any results doc is written.
- Rollout parameters: 100 samples per PA, horizon 6, temperature 1.0 (the per-head T of §4.4 is
  already baked into the probabilities), seeds per §4.3.2.
- **One run. No peeking-and-refitting.** Nothing may be fit, tuned, selected or re-thresholded after
  any 2026 number is observed.

---

## 6. Gate suite (WS5.3) — a suite that can fail

Graded on the §5.5 lockbox cohort in the one contact. All five gates are binding; the verdict logic
is in §6.8. Every statistic is also computed on 2024 dev BEFORE the contact (free tier) — both to
sanity-check the implementation and to run the anti-unfailability rule of §6.7.

### 6.1 G1 — classwise-ECE and TACE per factor head

Statistic: classwise-ECE (Nixon et al. 2019, "Measuring Calibration in Deep Learning" — the
static-binning classwise variant, explicitly recommended for high class counts), 15 equal-mass bins,
computed per head over its own conditional distribution; plus TACE (thresholded adaptive calibration
error) with threshold `1e-3` and 15 adaptive bins.

**[FIXED] thresholds**, derived by the single rule `threshold(C) = max(0.010, 0.010 * sqrt(C / 7))`
rounded to the nearest 0.005 (derivation in §6.6):

| Head | C | cwECE gate | TACE gate |
|---|---:|---:|---:|
| `H_type` | 17 | <= 0.015 | <= 0.015 |
| `H_zone` (conditional on sampled type) | 26 | <= 0.020 | <= 0.020 |
| `H_velo` (conditional on type, zone) | 5 | <= 0.010 | <= 0.010 |
| outcome head | 7 | <= 0.010 | <= 0.010 |

Classes with fewer than 100 lockbox observations are excluded from the head's classwise mean and
reported separately with their counts (a mean over 3-observation classes is noise, not calibration);
the exclusion list is part of the audit JSON.

### 6.2 G1 is comparable to the frozen v2 stack

The same four statistics are computed for the FROZEN v2 stack by marginalizing its flat 2,210-way
softmax onto the three fields (`p_type(t) = sum_{z,v} p(token)`, and the conditionals by division),
and for its A1 outcome head directly. Both sets are reported side by side in the results doc. This
comparison is what makes §6.7 possible.

### 6.3 G2 — KCE hypothesis test (calibration as a testable claim)

Statistic: the kernel calibration error of Widmann, Lindsten & Zachariah (NeurIPS 2019), unbiased
SKCE estimator, Laplacian kernel on the probability simplex with bandwidth set by the median
heuristic on the dev cohort (fixed before the contact and recorded), p-value by 1,000-resample
bootstrap, computed per head.

**[FIXED] gate:** FAIL if `p < 0.01` for any head — i.e. the gate fires when calibration is
*rejected* at the 1% level.

**[FIXED] power probe (mandatory, dev tier, before the contact):** re-run the same test on the dev
cohort after injecting a known miscalibration (multiply each head's logits by 1.10, a ~2pp-scale
distortion of the top-1 probability). If the test does not reject that injected distortion at
`p < 0.01`, G2 is **uninformative and is recorded as VOID** for this spec version (not as a pass);
the verdict then rests on G1/G3/G4/G5, and the void is stated in the results doc and the claims
entry. A test that cannot detect a planted error may not be counted as evidence of calibration.

### 6.4 G3 — PA-level PIT and marginal calibration

**PIT (Gneiting, Balabdaoui & Raftery 2007).** Randomized probability integral transform of the
PA-terminal wOBA under the model's simulated PA-outcome distribution, 20 equal-width bins on [0,1].
**[FIXED] gate:** Kolmogorov-Smirnov distance from U(0,1) `<= 0.03`, AND no bin's mass outside
`[0.67 x 0.05, 1.5 x 0.05]` = `[0.0335, 0.0750]`. (Effect-size gates, deliberately not a
significance test: at n ~ 60K a chi-square rejects on trivial deviations and would make the gate
unfailable in the opposite direction.)

**Marginal calibration.** The Phase 0.6 gate definitions are reused verbatim — no new tolerance is
invented (`PHASE_0.6_PLAN.md` §5, "K%/BB%/HR% within +/-10% relative OR +/-1pp absolute, whichever is
tighter; mean wOBA within +/-0.015 absolute; mean PA length within +/-0.5 pitches"), recomputed
against the **2026** empirical baselines measured in the same contact, and extended to the two rates
Phase 0.6 measured but did not gate:

| Quantity | Tolerance rule |
|---|---|
| K% | tighter of +/-10% relative and +/-1.00pp absolute |
| BB% | tighter of +/-10% relative and +/-1.00pp absolute |
| HR% | tighter of +/-10% relative and +/-1.00pp absolute |
| HBP% | tighter of +/-10% relative and +/-1.00pp absolute |
| hit% (1B+2B+3B+HR) | tighter of +/-10% relative and +/-1.00pp absolute |
| mean wOBA | +/-0.015 absolute |
| mean PA length (pitches) | +/-0.5 pitches |
| `calibration_valid` coverage | >= 95% of rollouts (`PHASE_0.6_PLAN.md` §5.2) |

### 6.5 G4 — binned calibration per count-state, and the position gate

**Per-count-state reliability** (nflfastR-style binned calibration plots, published as plots and as
numbers): for each of the 12 count states, top-1 ECE of the outcome head over 10 equal-mass bins.
**[FIXED] gate:** `<= 0.02` for every count state with `>= 500` lockbox observations; states below
500 are reported, plotted, and not gated.

**Position gate [FIXED]:** max over within-PA positions 0-5 and over the 7 outcome classes of
`|rollout marginal - empirical marginal|` `<= 1.0pp`. This is the identical quantity and threshold
that Phase 0.6.2 died on (§1.1), which is the point: v2-era raw-T scored 16.37pp and two rounds of
reweighting only reached 2.625pp, so this gate is decisively failable and is the direct test of H2.
Secondary, reported but not gated: Spearman rho between position index and per-position KL from
empirical (v2-era value r = 0.822 — the drift signature).

### 6.6 G5 — decision calibration against the consumer's decision functions

Framing: Zhao, Ma & Ermon (NeurIPS 2021) — full distribution calibration at high C is infeasible;
calibration against the specific decision functions a consumer applies is not. The sim's consumers
compute K%, BB% and HR% from PA distributions, so those three are the decision functions.

For each `d in {K, BB, HR}`: partition lockbox PAs into deciles of the model's predicted `P(d)`.
**[FIXED] gate:** `|mean predicted - empirical|` `<= 1.0pp` in every decile with `n >= 500`, AND the
n-weighted mean absolute decile gap `<= 0.5pp`. Deciles with `n < 500` are reported, not gated.

**Derivation of the 1.0pp family** (so no threshold here is free-floating): 1.0pp is (a) the absolute
arm of the Phase 0.6 K%/BB%/HR% tolerance rule, locked 2026-04-26 before any of these results
existed, and (b) the 0.6.2 fixed-point convergence threshold. The 0.010 ECE floor is the same 1pp in
probability units. The classwise ladder of §6.1 scales that floor by `sqrt(C / 7)` — the standard
`1/sqrt(n_per_class)` estimation-noise scaling at fixed cohort size, anchored at the 7-class outcome
head where 1pp is known to be measurable.

### 6.7 Anti-unfailability rule (executed on dev, BEFORE the lockbox contact)

The audit's core PitchGPT finding was a gate that could not fail (top-1 ECE 0.10 at C = 2,210, mean
confidence ~5%, measured 0.0090-0.0114 — roughly an order of magnitude inside its own line). To
prevent a repeat:

For every threshold in G1 and G4's per-count-state arm, compute the corresponding statistic for the
**FROZEN v2 stack on the 2024 dev cohort** (§6.2 marginalization for the field heads). **[FIXED]
rule:** if the frozen v2 stack already satisfies a threshold on dev, that threshold is tightened to
`0.6 x (frozen-v2 dev value)`, floored at `0.005`, and the tightened value is what v3 is graded on.
Tightening is executed once, recorded in §9 with both numbers, and happens strictly before the
unseal. Thresholds are never loosened by this rule — if frozen v2 fails a threshold on dev, the
threshold stands as written.

G3's marginal arm is exempt (its tolerances are inherited verbatim from a pre-existing locked spec
and the v2 stack fails them decisively — K% 0.3339 vs 0.218). G2 has its own power probe (§6.3).
G5 is exempt for the same reason as G3 (v2-era K%/BB%/HR% biases are ~11pp and ~3pp, far outside
1.0pp).

### 6.8 Verdict logic

**PASS = G1 AND G3 AND G4 AND G5 all green, and G2 green-or-VOID.** Any red gate = FAIL. There is no
partial pass, no "directionally favourable" verdict, and no gate may be dropped after the numbers
land. The verdict publishes either way (plan §5.1: "Either way, publish the verdict"), with a claims
registry entry, and the results doc volunteers the weakest number unprompted (the FanGraphs-Lab
fidelity-report norm, plan §5.5).

---

## 7. Pre-registered kill criteria for v2

Mirroring the discipline of `PHASE_0.6.2_PLAN.md` §6: hard, numeric, and dated before any result.

### 7.1 K-v2-FIT-A — the factorization must not cost predictive quality (dev tier)

After Stage A, compare teacher-forced NLL per pitch on the **2024 dev** cohort against the frozen v2
backbone scored on the identical rows, using the same composite-token likelihood (v3's chain-rule
product is multiplied back to a token probability, §3.3).

**KILL if v3's dev NLL per pitch is more than 2.0% relative WORSE than frozen v2's.** Derivation of
2.0%: the platform's own matched-scale architecture deltas on this data are +2.57% (v1) and +3.13%
(v2) perplexity vs LSTM — architecture-level differences here live in the low single digits, so a
>2% regression means the factorization is paying real predictive cost, and H1 is false. On kill: stop
before Stage B; no rollout fine-tuning is run; publish the negative.

### 7.2 K-v2-FIT-B — exposure bias must actually close on the fit cohort

After Stage B3, on the **2023 fit cohort** (§5.2), measure the same quantity Phase 0.6.2 died on:
max over within-PA positions 0-5 and outcome classes of `|rollout marginal - empirical marginal|`.

**KILL if that maximum is > 1.0pp.** Reference points: 16.37pp for the v2-era stack under raw T,
2.625pp after two rounds of the (now-banned) output reweighting. At most **2 curriculum runs**
total, the second containing exactly one documented change (§4.5); a third is forbidden, exactly as
"no third update, no third re-roll" was in 0.6.2 §10.A7. **No post-hoc reweighting layer may be added
to rescue this number** (§0.2) — that move is what this whole program replaces.

### 7.3 K-v2-COMPUTE — the program has a budget

**KILL (pause, and require a dated §9 amendment to continue) if cumulative GPU time across Stage A,
Stage B, ablations and dev evaluation exceeds 12 GPU-hours**, or if any single run exceeds 6 hours
wall clock. Anchors: the plan's 8.5-minute backbone-retrain figure and the training script's
documented ~4h25m 10K-game run bracket the uncertainty; 12 GPU-hours covers the pessimistic bracket
with room for the two permitted attempts, and the cap exists so an unbounded retry loop cannot
develop quietly on one RTX 3050.

### 7.4 K-v2-GATE — the lockbox verdict is final

Per K5: the retrain gets **ONE** contact against sealed 2026, graded by §6.8. **On FAIL: the
per-pitch-only claim locks for the season**; PA-level absolute-rate products stay dropped from
Tier-A scope; no second lockbox contact, no re-fit, no re-thresholding, no subgroup rescue (the
banned-rescue clause of K3 applies here by analogy and is adopted explicitly). The failure is
published with the same prominence a pass would get. On PASS: PA-level absolute-rate products
unblock **for the v3-factorized artifact only**, each with its own claims-registry entry; nothing
about the v2-era stack is re-earned retroactively.

### 7.5 Provenance guard (K5 second clause, standing)

No calibration artifact may be fit on any cohort a gate is evaluated on. Enforced structurally:
every calibration sidecar declares `fit_cohort_season`, and the predictor refuses artifacts
declaring 2025 or 2026. A run that would violate this aborts rather than warns.

---

## 8. Artifacts, provenance, reproducibility

### 8.1 Naming (write-once; nothing existing is overwritten)

`models/pitchgpt_v2.pt` is a locked, SHA-pinned, paper-referenced checkpoint, so the WS5.2 "v2
retrain" ships under a NEW artifact name. **[FIXED]:**

| Artifact | Path |
|---|---|
| Factorized backbone + heads | `models/pitchgpt_v3_factorized.pt` |
| Outcome head on the v3 backbone | `models/pitchgpt_v3_outcomehead.pt` |
| Per-head temperatures | `models/calibration_pitchgpt_v3.json` |
| Fit / training audit | `results/pitchgpt_v3/train_<stage>_<UTC>/` |
| 2024 dev evaluation | `results/pitchgpt_v3/dev_2024_<UTC>/` |
| 2023 fit-cohort kill measurement | `results/pitchgpt_v3/fit_2023_<UTC>/` |
| Sealed-2026 lockbox grading | `results/pitchgpt_v3/lockbox_2026_<UTC>/` |

Naming note: the program is "WS5.2 v2" but the artifacts are `v3_*` because the v2 filenames are
taken by frozen blobs. Registry versions are `pitchgpt/v<YYYY.MM.DD>-factorized` etc., registered
write-once with `manifest.json` per WS2.1; **no alias change** (`production` / `frozen_validated`
stay pointed at `v2026.04.23`) unless and until §6.8 returns PASS, at which point the alias move is
its own reviewed step.

### 8.2 Every run writes an audit JSON containing, at minimum

`spec_path` and `spec_freeze_sha` (the §9 entry-1 SHA), `git_sha`, `stage`, seeds, cohort definition
and row counts, per-head parameter counts and the §3.3 assertions' results, `support_counts` hash,
mask fallback/degenerate counts, wall clock and GPU seconds, checkpoint SHA256 pre/post for every
frozen artifact touched, and the DuckDB access mode (must be `read_only=True`).

### 8.3 Tests required before any training run

1. Dynamic-context assertion: the context vector fed at within-PA positions 0..5 is NOT constant
   across positions for a PA whose count advances (guards the `6111cd6` fix, §4.1).
2. Chain-rule identity: for a random batch, `p_type * p_zone * p_velo` reconstructed onto the 2,210
   token space sums to 1 within `1e-5`, and the argmax token matches sequential-greedy decode.
3. Parameter-budget assertion (§3.3) fails the build if the output stack exceeds its ceiling.
4. Mask legality: no sampled token ever has `type_idx == 16`, and every sampled `(type, zone, velo)`
   has training support `>= 10` or a recorded fallback.
5. Provenance guard extension: a v3 calibration sidecar declaring `fit_cohort_season in {2025, 2026}`
   is refused on load (§7.5).
6. Ledger guard: the lockbox evaluation entry point is decorated with `@holdout_access` and refuses
   to run while `lockbox_2026_full_season.unsealed == false`.

### 8.4 Single-writer and hardware rules

DuckDB `read_only=True` for every step of this program; the dashboard is stopped during GPU runs;
GPU work serializes with any other lane's GPU work (one RTX 3050).

---

## 9. Deviations Log (append-only — the ONLY mutable section)

Every entry: UTC date, what changed, why, and who. Entries are added; nothing is edited or removed.
Sections 0-8 are immutable after the freeze commit.

| # | Date (UTC) | Entry |
|---|---|---|
| 1 | _pending_ | **Freeze commit SHA: `<to be filled by the orchestrator in the commit that introduces this file, or the immediately following commit>`.** This spec's body (§0-§8) is frozen as of that commit. Authored 2026-08-10 by the Batch-D executor for task D3a under the 2026-08-10 platform improvement plan; no training run had been executed against it at freeze time. |
