# Research Plan B — PA-Outcome Prediction

**Date:** 2026-04-24
**Status:** drafted, not greenlit. No execution before user sign-off.
**Discipline:** Research → Plan → Execute (per `feedback_research_plan_execute.md`).
**Scope:** find the best-available calibrated 7-class pitch-outcome predictor for the sim engine. Distinct from Plan A (close PitchGPT's perplexity gap) and Plan C (sim engine API).

---

## 1. Mission statement

**Primary question.** Given a per-pitch context (sequence prefix, count, runners, batter handedness, pitcher, etc.), what's the best calibrated 7-class outcome predictor we can build, and is it good enough to feed the sim engine's PA-outcome rollouts?

**Why the sim engine needs this.** Per `docs/pitchgpt_sim_engine/EXECUTION_PLAN.md` §4.1: every Tier-A item (counterfactual pitch-call grade A1, probabilistic projections A2, matchup sims A3) reduces a rollout to a plate-appearance outcome. Without per-pitch outcome probabilities — `ball / called_strike / swinging_strike / foul / in_play_out / in_play_hit / hbp` — the sim cannot terminate or aggregate to PA-level outcomes. This is the critical-path unblocker for Phase 1 of the sim-engine plan.

**What the Phase 0.3 FAIL told us.** Source: `results/pitchgpt/outcome_head_train_2026_04_24/report.md`.

- The frozen v2 backbone hidden state + 2-layer MLP head (128 → 64 → 7) was trained on the 2015–2022 pitcher-disjoint cohort (10K games, 2.79M valid outcomes). 5 epochs, early-stopped at epoch 3.
- 7-class log-loss on 2023 val (post-temp): 1.7367. Frequency-prior baseline: 1.6487. **Lift: −5.34% — the model is WORSE than constant frequency prior.**
- Per-class diagnosis: `ball/called_strike/swinging_strike/foul/in_play_out` all approximately match prior (~1.66–1.74 nats). `in_play_hit` 2.58, `hbp` 5.96 — the head catastrophically fails on rare classes. Inverse-frequency capped-at-10× weights were insufficient.
- ECE post-temp 0.0213 (PASS <0.05 budget). Backbone byte-identity verified pre/post (frozen).
- The Phase 0.2 smoke at 500 games landed log-loss 1.6337 (within sampling noise of 1.6487 prior). The smoke was misleading: it suggested barely-positive lift, the full-scale run revealed a true negative lift.

**What this implies (the reframe).** Stop assuming the PitchGPT v2 backbone hidden state is the right substrate for outcome prediction. The Phase 0.3 result is direct evidence that, for this target, the backbone's representation does NOT carry the discriminative information needed beyond what a constant prior provides. The right question becomes: **"what's the best predictor of pitch outcome given the available features?"** PitchGPT's hidden state is one candidate; engineered features into XGBoost are another; logistic regression is a lower-bound interpretable baseline; per-pitcher × per-count × per-pitch-type empirical priors are a non-parametric baseline.

The PitchGPT-frozen-MLP head failed before we ever ran the obvious baselines. **Plan B starts with those baselines.**

**Success.** A predictor (any architecture A1–A6 below) achieves on the 2025 pitcher-disjoint holdout:
- 7-class log-loss lift ≥ +10% vs frequency prior, AND
- ECE post-temperature < 0.05 (10-bin), AND
- Per-class log-loss for `in_play_hit` < 2.0 and `hbp` < 4.0 (so rare-class discrimination is at least minimally usable).

**Failure (and what we DO with it).** No architecture clears +5% lift on the 2025 holdout. Outcome: declare the pitch-outcome problem fundamentally hard at the 7-class granularity, sim engine pivots to PA-level rollouts using existing Statcast PA-terminal events directly (no per-pitch outcome head). Tier-A items are reframed accordingly. Stop trying to build an outcome head. **Do not cycle.**

**This plan is NOT about.** Building the sim engine itself (Plan C). Re-architecting PitchGPT (Plan A). Re-litigating the calibration claim. Predicting wOBA / xwOBA / launch-angle as separate problems (those are downstream of any successful outcome head; we treat them as future work).

---

## 2. Reframe — stop privileging the PitchGPT backbone

The Phase 0.3 design (`docs/pitchgpt_sim_engine/pa_outcome_head_design.md` §9) and the EXECUTION_PLAN §4.1 framing both privilege PitchGPT's hidden state as THE substrate for the outcome head. Two routes were considered (joint vs frozen); both used the backbone as the base. Phase 0.3 fail tells us this is the wrong frame.

### 2.1 What the PitchGPT backbone optimizes for

`PitchGPTModel.forward` (line 1152 of `src/analytics/pitchgpt.py`) is trained with cross-entropy over the 2,210-token composite vocab. That objective is "predict the joint distribution of `(pitch_type, zone, velocity_bucket)` for the NEXT pitch." It is NOT "predict the outcome of the NEXT pitch given the current pitch." The hidden state at position `j` is optimized to encode whatever features minimize CE on the next-pitch token — primarily pitcher style, count-state preferences, and recent pitch-mix tendencies.

Pitch outcome (ball / strike / contact / etc.) depends on:
- The pitch's actual location, velocity, and movement (the model knows *what kind* of pitch is coming but not exactly where it ends up).
- The umpire's judgment (partial-ABS in 2025).
- The batter's swing decision and contact quality.
- Game-state factors the backbone consumes (count, runners) but in a representation tuned for next-pitch *prediction*, not outcome inference.

The backbone hidden state does encode some of these factors (they're predictive of the next-pitch token, so they leak in), but it's not optimized to surface them in a way usable by a downstream classifier. That's a generic problem with frozen-backbone transfer; the Phase 0.3 result confirms it empirically here.

### 2.2 What features actually matter for pitch outcome

The known industry baseline for this target uses engineered features:
- pitch type (categorical, 17 levels)
- zone / location (categorical 26 or continuous plate_x/plate_z)
- velocity bucket OR continuous release_speed
- count (12 states)
- outs (3)
- runner state (8)
- batter hand × pitcher hand (4)
- inning (4 bucket)
- score diff (5 bucket)
- umpire scalar (1, optional)
- pitcher identity (categorical, ~3K levels)
- batter identity (categorical, ~3K levels)

Those features, fed into a properly-tuned XGBoost, are the obvious baseline that Phase 0.3 SKIPPED. The Plan B priority order below puts XGBoost first — if it fails, we know the problem is data/features, not architecture; if it succeeds, we have the floor every transformer-based variant must beat.

---

## 3. Hypotheses to test

Order is by expected information-per-cost. Cheapest informative tests first; expensive backbone variants last.

### A3 — XGBoost on engineered features (run FIRST)

**Mechanism.** Standard tabular gradient-boosted-trees classifier on the 12-feature set above (pitch_type, zone, velocity, count, outs, runners, hands, inning, score_diff, ump scalar, pitcher_id, batter_id). 7-class softmax objective; class weights inverse-frequency capped at 10. Tune via `xgboost.cv` on 2023 val.

**Why it might work.** This is the standard industry baseline. It uses pitch_type, zone, and velocity DIRECTLY (not as a 2,210-class softmax target the backbone has to disentangle), it includes pitcher and batter identity (which PitchGPT explicitly does NOT have), and XGBoost handles class imbalance reasonably with appropriate weighting. The Phase 0.3 fail shows the PG backbone hidden state lacks something XGBoost would capture trivially.

**Why it might NOT work.** 7-class log-loss is intrinsically hard if the conditional distribution is high-entropy — a fastball middle-middle on count 0-0 with no runners might genuinely be ~30% ball, ~20% called strike, ~10% swinging strike, ~25% foul, ~15% in_play (out / hit) regardless of pitcher / batter. If the residual entropy IS the answer, no model will beat the frequency prior by much.

**Expected effect size.** +10 to +20% lift over frequency prior. This is a strong prior — XGBoost on a comparable target (Statcast-style "swing decision" classification) typically lands in this range in published baselines. If A3 lands in this range, it pushes through the +10% PASS gate decisively.

**Test.**
1. Build feature matrix from the same 2015–2022 pitcher-disjoint train cohort, 2023 val, 2025 pitcher-disjoint holdout used by every other PitchGPT experiment. Use `pitches.description` + `pitches.events` mapped to 7 classes per the locked rules in `pa_outcome_head_design.md` §2.2.
2. Train XGBoost with: `objective=multi:softprob`, `num_class=7`, `max_depth=6`, `learning_rate=0.08`, `n_estimators=400` with early-stopping at 20 rounds on 2023 val log-loss, `min_child_weight=3`, scale_pos_weight via inverse-frequency vector.
3. Hyperparameter tune via 5-fold CV on the train cohort (NOT touching 2023 val for tuning). Grid: `max_depth ∈ {4, 6, 8}`, `learning_rate ∈ {0.05, 0.08, 0.12}`.
4. Apply temperature scaling on 2023 val, evaluate on 2025 holdout.

**Cost.** Wall-clock: feature build ~10 min, train ~30 min on CPU (XGBoost is fast), tune ~3 hours total. **~4 hours wall-clock, no GPU.** Should be the FIRST experiment.

**Decision after A3.**
- If A3 ≥ +10%: A3 is a candidate for the production outcome predictor. Proceed to A1 only to test whether the PG backbone adds anything; if A1 doesn't beat A3, ship A3.
- If A3 in [+5%, +10%): proceed to A1 + A2 to see if architectural lift on top of XGBoost helps.
- If A3 < +5%: HARD STOP. The problem is data/features, not architecture. Per kill criteria in §7, sim engine pivots to PA-level rollouts. Do NOT run A1, A2, A6.

### A4 — Logistic regression on engineered features (interpretable lower bound)

**Mechanism.** Multinomial logistic regression with the same feature set as A3 (one-hot encode all categoricals, leave continuous features as-is). L2 regularization, tuned on 2023 val.

**Why it might work.** A linear predictor on well-engineered features is often surprisingly competitive on tabular data; if it lands close to A3, that's strong evidence the problem is approximately additive in the features, no need for XGBoost.

**Why it might NOT work.** Pitch outcomes have known interactions (a fastball middle-middle vs a curveball middle-middle have very different swing rates; the count × pitch-type interaction is large). Pure linear modeling captures the marginals but not the interactions.

**Expected effect size.** +3 to +12% lift over frequency prior. Most likely outperforms PG-frozen-MLP (since A3 should), underperforms XGBoost.

**Test.**
1. Same cohort as A3.
2. Train sklearn `LogisticRegression(multi_class='multinomial', solver='saga', max_iter=2000, C ∈ tuned grid)`.
3. Apply temperature scaling on 2023 val, evaluate on 2025 holdout.

**Cost.** ~30 min total. CPU-only. Run alongside A3 in parallel.

**Why include this.** Provides an interpretable lower bound. If logistic regression gets +8% and XGBoost gets +10%, the marginal value of XGBoost is small and we'd consider shipping logistic regression for transparency/auditability.

### A5 — Per-pitcher × per-count × per-pitch-type empirical priors (Bayesian non-parametric)

**Mechanism.** Build a 4-way table indexed by `(pitcher_id, count, pitch_type, batter_hand)` with smoothed empirical outcome distributions. Use additive (Dirichlet) smoothing with α = 1; for unseen pitcher × bucket combinations, fall back to (count, pitch_type, batter_hand) marginal; further fall back to overall frequency prior.

**Why it might work.** Pure no-model baseline. Pitcher tendencies in specific counts on specific pitch types are real and large — Aaron Nola throws a curveball on 0-2 to a left-handed batter and the outcome distribution is tightly known. Empirical lookup may capture most of the signal a model would.

**Why it might NOT work.** Pitcher-disjoint holdout (2025 debuts) cannot have pitcher-specific empirical priors — every test pitcher has zero training data. The fallback to (count, pitch_type, batter_hand) is what would actually drive predictions on holdout. So this hypothesis is really about whether "context-specific frequency tables" beat the global frequency prior — which they should, but how much?

**Expected effect size.** +2 to +8% lift over global frequency prior on the 2025 holdout (limited because pitcher-specific lookups are unavailable for unseen pitchers).

**Test.**
1. Build the lookup table on 2015–2022 train cohort.
2. Predict on 2023 val, then 2025 holdout. For each test row, look up the deepest available bucket; if missing, fall up the hierarchy.
3. Evaluate log-loss + ECE.

**Cost.** ~15 min total. No model training. Run alongside A3 + A4.

**Why include this.** Sets the floor for "structured frequency lookups beat marginal frequency prior by X%." If A5 gets +5% and A3 gets +6%, the XGBoost is barely doing better than a smart lookup — that's a useful diagnostic.

### A1 — Frozen PG backbone + larger MLP head + raw context vector concatenated at head input

**Mechanism.** Take the same frozen v2 backbone as Phase 0.3, but the head input is `concat(backbone_hidden_state, raw_context_vector_35d, pitch_type_onehot_17d, zone_idx_onehot_26d, velocity_bucket_onehot_5d) → 128 + 83 = 211d`. Larger 3-layer MLP: 211 → 128 → 64 → 7. Same training protocol as Phase 0.3 (5 epochs, AdamW lr 1e-3, inverse-frequency class weights cap 10).

**Why it might work.** Phase 0.3 diagnosis: "backbone hidden state alone is insufficient for outcome discrimination." The raw context vector and the actual pitch token components encode information the backbone may have compressed away in service of next-token prediction. Concatenating them gives the head the same features XGBoost has, with the addition of whatever sequence-prefix information the backbone genuinely captures. If the backbone adds anything beyond what engineered features provide, A1 should beat A3.

**Why it might NOT work.** If the backbone hidden state truly carries no marginal information beyond engineered features (which is what Phase 0.3's −5.34% lift suggests), A1 will land at approximately A3's performance with extra parameters and slower training. Worse: if the backbone encodes irrelevant noise that the head must learn to ignore, A1 underperforms A3.

**Expected effect size.** +0 to +3 pp ABOVE A3's lift. If A3 is +12%, A1 is +12% to +15%. If A3 is +6%, A1 is +6% to +9%. The marginal value of the backbone is what's being measured.

**Test.**
1. Same cohort + holdout as Phase 0.3 / A3.
2. Modify `src/analytics/pitchgpt_outcome_head.py::FrozenOutcomeHead` to accept the concatenated input.
3. Train head only (backbone still frozen) for 5 epochs.
4. Apply temperature scaling on 2023 val, evaluate on 2025 holdout.
5. Report A1 log-loss − A3 log-loss with bootstrap CI on the delta.

**Cost.** Code change ~0.5 day. Training ~3 min on RTX 3050 (head-only, 5 epochs). **~0.5 day + ~10 min GPU.**

**Sequencing note.** Run A1 ONLY after A3 lands. If A3 fails the kill criterion, A1 also dies (the backbone can't add information that engineered features don't already capture).

### A2 — Joint-trained head (revisits Phase 0.2 decision)

**Mechanism.** Same architecture as the original Phase 0.2 joint route — full backbone unfrozen, training with `CE_token + λ * CE_outcome` joint loss. λ tuned on val (0.5 was the unverified default). Backbone allowed to adapt for the outcome task at the cost of next-token PPL and calibration drift.

**Why it might work.** Joint training lets the backbone learn outcome-discriminative features the frozen route can't access. Phase 0.2 smoke (500-game scale) showed joint barely underperformed frozen on log-loss BUT blew the calibration budget by 2.5×. At full 10K scale, joint may give meaningfully more lift if the backbone has room to adapt without catastrophic calibration loss.

**Why it might NOT work.** Phase 0.2 already documented that joint at smoke scale violated the +0.005 ECE budget by 2.5×. Scaling to 10K may amplify rather than fix the problem. Also: joint training breaks the locked v2 paper checkpoint — to retain the calibration claim, we'd save a SEPARATE jointly-trained checkpoint and never overwrite v2.

**Expected effect size.** +0 to +5 pp above frozen, with high probability of exceeding the ECE budget. Conditional value: only useful if (a) it materially beats A1 on log-loss AND (b) backbone calibration on the new joint checkpoint stays within the ECE budget.

**Test.**
1. Same cohort.
2. Modify training loop: backbone unfrozen, joint loss with λ ∈ {0.1, 0.3, 0.5, 1.0} grid (4 runs).
3. Save jointly-trained checkpoint to `models/pitchgpt_v3_joint_outcomehead.pt` (NEVER overwrite v2).
4. Measure outcome log-loss + outcome ECE + backbone token ECE for each λ.
5. PASS criterion: joint outcome log-loss > A1 log-loss by ≥0.01 nats AND backbone token ECE on jointly-trained checkpoint < 0.0125 (the +0.005 budget).

**Cost.** 4 full backbone retrains at 10K scale ~13 min each + outcome head training time = ~80 min GPU per λ × 4 = ~5 hours GPU. Plus all the calibration re-measurement. **~5 hours GPU + 1 day analysis.**

**Sequencing note.** Run A2 ONLY if A1 lands ≥ +10% AND user wants to push for the maximum-lift configuration. If A3 already wins decisively (≥ +15%) skip A2 — the simplest model that works wins.

### A6 — Two-stage head (HBP-vs-rest then 6-class)

**Mechanism.** A1's per-class diagnostic showed `hbp` log-loss 5.96 dragged the overall mean below the frequency prior. A6 splits the problem: stage 1 binary classifier `hbp vs not-hbp`, stage 2 6-class softmax over `{ball, called_strike, swinging_strike, foul, in_play_out, in_play_hit}` conditional on not-hbp. Each stage gets appropriate class weights for its own class imbalance.

**Why it might work.** HBP is a 0.3% class. The 7-class softmax forces the model to allocate logit budget to a class it almost never sees, which can destabilize the other classes. Splitting isolates the rare-class problem: stage 1 has its own loss landscape, stage 2 sees a more balanced problem.

**Why it might NOT work.** Two-stage classifiers are notoriously calibration-tricky — the joint probability `p(hbp) × p(class | not-hbp)` doesn't naturally calibrate via single-stage temperature scaling. ECE may degrade.

**Expected effect size.** +1 to +3 pp on 7-class log-loss IF A1 is HBP-bottlenecked. If the bottleneck is `in_play_hit` (also 5%-ish), A6 doesn't help much.

**Test.**
1. Reuse A1's substrate (frozen backbone + larger MLP).
2. Split into 2 heads.
3. Train each independently with appropriate class weighting.
4. Combine probabilities, apply temperature scaling at the joint output.
5. Compare to A1.

**Cost.** Code change ~0.5 day. Training same as A1.

**Sequencing note.** Run A6 ONLY if A1's per-class diagnostic confirms HBP and `in_play_hit` are the bottleneck. If they're not, A6 doesn't help.

---

## 4. Baselines

Every variant from A1–A6 must report:

1. **Frequency prior** (the published Phase 0.3 baseline = 1.6487 on 2023 val). The bar to beat for ANY claim of utility.
2. **Per-class uniform prior** (log(7) ≈ 1.9459). The trivial floor.
3. **Per-count empirical prior** (count-conditional frequency lookup, no model). Catches "is the model doing more than count-conditional averaging?"
4. **Phase 0.3 frozen-MLP baseline** (the failing baseline we want to beat by definition).

Headline claim format: "Variant X beats frequency prior by Y% (95% CI [Y_lo, Y_hi]) AND beats the Phase 0.3 frozen-MLP baseline by Z% (95% CI [Z_lo, Z_hi]) on 2025 pitcher-disjoint holdout."

---

## 5. Evaluation cohort

**Train.** 2015–2022 pitcher-disjoint, 10K-game subset (matches Phase 0.3 cohort, matches the canonical PitchGPT splits). 2.79M valid outcome labels.

**Val.** 2023 pitcher-disjoint slice. 75,384 valid outcome labels (per Phase 0.3 numbers). Used for hyperparameter tuning, temperature scaling, early stopping. NEVER for headline claim measurement.

**Test.** 2025 pitcher-disjoint holdout. The same 469-pitcher cohort the methodology paper, the matched-LSTM 10K runs, and the sampling-fidelity report all use. **All headline numbers come from this set.**

**Metrics.**
- 7-class log-loss (primary)
- Per-class log-loss (diagnostic — to detect rare-class collapse like Phase 0.3's HBP)
- ECE (10-bin, both pre and post temperature scaling)
- Per-class confusion (7×7 matrix)
- Top-1 accuracy (sanity, low-bar)
- Per-pitcher log-loss variance across the 50 highest-frequency 2025 pitchers (drift sanity — same as Phase 0.4 spec from EXECUTION_PLAN §6.0.4)

**Statistics.** Bootstrap 1,000 outcome-label-level resamples for 95% CIs on log-loss. Pairwise bootstrap on the variant-vs-baseline DELTA (paired by sampling indices to preserve correlation structure).

**Multiple-comparison.** ~6 architectures × multiple sub-experiments → ~12 comparisons. Apply Bonferroni at α/12 = 0.0042 for HEADLINE claims; per-experiment 95% CIs for diagnostic purposes.

---

## 6. Gate criteria — explicit numbers

| Verdict | 7-class log-loss lift vs frequency prior (point) | 95% CI lower bound | ECE post-temp | Per-class `in_play_hit` log-loss | Per-class `hbp` log-loss |
|---|---|---|---|---|---|
| **PASS (full)** | ≥ +10% | ≥ +5% | < 0.05 | < 2.0 | < 4.0 |
| **WEAKER PASS** | ≥ +5% | ≥ +2% | < 0.05 | < 2.5 | < 5.0 |
| **FAIL** | < +5% | OR < +2% | OR ≥ 0.05 | OR ≥ 2.5 | OR ≥ 5.0 |

WEAKER PASS is usable for the sim engine but flagged honestly: "the outcome head provides a +5–10% lift over a frequency prior; downstream sim items inherit that ceiling." Tier-A items downstream MUST report the underlying outcome head's lift in their own validation tables — no laundering.

PASS is what the sim engine wants — meaningful lift with calibration intact. FAIL triggers §7 kill criterion.

---

## 7. Kill criteria — explicit and binding

After A3, A4, A5 are evaluated (the cheap baseline triad), check the score:

- **If max delta from A3+A4+A5 < +5%**: HARD STOP. Per-pitch outcome prediction at 7-class granularity is fundamentally hard. Sim engine pivots to **PA-level rollouts** — instead of per-pitch outcome rollouts, sample directly from the empirical PA-terminal-event distribution conditioned on (pitch type sequence, count progression, runner state, batter hand). This is a meaningful product reframe, see §7.1 below. Do NOT run A1, A2, A6.
- **If A3 ≥ +10% AND A1 doesn't add ≥+1pp lift**: SHIP A3 (XGBoost on engineered features). The PitchGPT backbone provides no marginal value for this target. Calibration A3 with temperature scaling, integrate into sim engine. Stop.
- **If A1 ≥ A3 by +1 to +3 pp**: SHIP A1 (frozen PG + larger head + concatenated context). Do NOT run A2 unless user wants to push for A2's marginal lift over A1.
- **If A1 ≥ +15% AND A2 (joint) breaks calibration budget**: SHIP A1. Calibration is the load-bearing flagship claim; we don't sacrifice it for marginal log-loss gains.

**Hard cycle limit.** No architecture is re-tried with tweaked hyperparameters more than ONCE. If A3 fails, "A3 with deeper trees" is not a separate hypothesis.

**Calibration kill.** Same as Plan A: ANY variant that posts a log-loss lift but breaks the ECE budget (≥0.05) is REJECTED.

### 7.1 The PA-level rollout reframe (kill-path product spec)

If §7's hard stop fires (no architecture clears +5% lift), the sim engine reframes as follows. This is documented here so the stop is clean — not a failure mode but a planned alternative:

- **Replace per-pitch outcome head with empirical PA-terminal lookup.** For each rolled-out PA (a sequence of sampled pitch tokens), use the EMPIRICAL distribution of PA-terminal `events` from training, conditioned on: terminal pitch type, terminal zone, terminal velocity bucket, count at termination, batter hand, pitcher hand. This is non-parametric, requires no training, and is calibrated by construction (it IS the empirical distribution).
- **Cost.** wOBA rollups become noisier (the PA-distribution conditioning is coarse), but Tier-A items (A1 grades, A2 projections, A3 matchup sims) still ship — with WIDER CIs and a "PA-empirical-sampling" footnote rather than "outcome-head-conditional-sampling."
- **What gets dropped from sim-engine scope.** The fine-grained per-pitch outcome distribution — useful for, e.g., "what's the probability this specific pitch is called a strike given context?" — is unavailable. A1 (counterfactual pitch-call grade) loses some of its sharpness but still works at PA level (rollout-percentile-of-actual-outcome).
- **Methodology paper consequence.** v2 paper would document the negative result honestly ("we attempted a learned 7-class outcome head; it failed to beat a frequency prior; we pivoted to empirical PA-terminal lookup"). Per Path 2 philosophy, this is fine.

---

## 8. Sequencing

### 8.1 Independent vs sequential

| Hypothesis | Depends on | Parallelizable? |
|---|---|---|
| A3 | nothing | yes (CPU-only, can run alongside any GPU work) |
| A4 | nothing | yes (CPU-only, alongside A3) |
| A5 | nothing | yes (no training, alongside A3+A4) |
| A1 | A3 (kill check) | sequential after A3 |
| A2 | A1 (kill check) | sequential after A1 |
| A6 | A1 (per-class diagnostic) | conditional on A1 outcome |

### 8.2 Wall-clock estimates

**Sequential.**
- A3: ~4 hours (CPU)
- A4: ~30 min (CPU, parallel with A3)
- A5: ~15 min (CPU, parallel with A3)
- A1: ~0.5 day code + ~10 min GPU
- A2: ~5 hours GPU + 1 day analysis
- A6: ~0.5 day code + ~10 min GPU

**Expected case (A3 wins decisively, ship A3).** ~4 hours wall-clock. Stop.

**Expected case (A3 + A1 needed).** ~4 hours + 0.5 day + 10 min GPU = ~1 work day.

**Worst case (A3 + A1 + A2 + A6 all run).** ~2 work days.

**Hardest case (A3 fails, kill criterion fires).** ~4 hours wall-clock. Then 1 day to write the PA-empirical-lookup pivot doc. Total ~1.5 days.

### 8.3 Recommended sequence

1. **A3 + A4 + A5 in parallel** (all CPU, all cheap). Decision after A3 per §7.
2. **A1 second** (only if A3 ≥ +5%). Decision after A1 per §7.
3. **A6 conditional** on A1's per-class diagnostic.
4. **A2 last** (only if A1 ≥ +10% AND user explicitly wants to push for max lift).

---

## 9. What this plan does NOT do

- **Does not build the sim engine.** Plan C (or EXECUTION_PLAN §6.0.5) owns the rollout harness. Plan B's deliverable is a calibrated outcome predictor with measured log-loss + ECE on 2025 holdout; downstream consumption is not in scope.
- **Does not re-architect PitchGPT.** Plan A owns the next-token PPL gap. Plan B treats the backbone as a fixed (or, in A2, augmented) feature extractor.
- **Does not re-litigate the calibration claim.** ECE is a guardrail (must not break) but not a target.
- **Does not chase the +15% gate without bound.** §7's kill criteria are binding.
- **Does not predict launch_speed, launch_angle, exit-velocity bucket, or wOBA directly.** These are downstream of any outcome head; they are future work, not Plan B scope.

---

*Document author: Claude (session 2026-04-24). Doc-only research plan; awaits user greenlight before any execution agent fires. Leave unstaged.*
