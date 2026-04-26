# Research Plan A — PitchGPT Accuracy Gap

**Date:** 2026-04-24
**Status:** drafted, not greenlit. No execution before user sign-off.
**Discipline:** Research → Plan → Execute (per `feedback_research_plan_execute.md`).
**Scope:** model work to close the PitchGPT-vs-LSTM perplexity gap on the canonical 2025 pitcher-disjoint holdout. Distinct from Plan B (build a sim-engine product on the existing PitchGPT) and Plan C (sim-engine API).

---

## 1. Mission statement

**Primary question.** Can PitchGPT's perplexity gap over a matched LSTM at 10K-game scale on the canonical 2025 pitcher-disjoint holdout be closed from the current ~3% to ≥15% (the locked spec gate from `docs/models/pitchgpt_validation_spec.md`) without breaking the calibration claim that anchors the flagship today?

**Why it matters relative to the existing flagship.** PitchGPT is currently a flagship by virtue of *calibration*, not accuracy. Per `docs/NORTH_STAR.md` "Post-evidence consolidation — 2026-04-18 late evening" and the 2026-04-24 update: the flagship-allowed claim is "calibrated rollout engine, matches empirical marginals, beats naive baselines by wide margins." The ≥15% LSTM gate is RETIRED as a live claim (the 13.80% number from the methodology paper §3.1 was a 1K-vs-1K artifact; at matched 10K scale v2 lands at +3.13%, see `results/pitchgpt/2025_holdout/report.md`).

This plan asks: *if we can close the gap cheaply, do we?* If yes, the methodology paper gains a clean "transformer beats LSTM by spec margin" pillar alongside calibration, and the flagship narrative thickens. If no, the narrowed claim is permanent, the LSTM-gate language is removed from the spec, and we move on. Either outcome is publishable; what is NOT acceptable is endless re-litigation of the gap with no stop condition.

**Success.**
- A configuration of PitchGPT (any combination of H1–H8 below) achieves ≥15% perplexity improvement over a matched-scale LSTM on 2025 pitcher-disjoint holdout, point estimate, with 95% CI lower bound ≥ 10%.
- Calibration survives: post-temperature ECE on the 2025 holdout < 0.02 (current v2: 0.0075; budget +0.0125).
- Markov-2 and heuristic baselines remain PASS with wide margins (sanity).

**Failure (and what we DO with the failure).**
- After H1–H8 evaluated in priority order, max LSTM-improvement < 10% (or improves accuracy but breaks calibration). Outcome: declare the narrowed flagship claim permanent, edit the validation spec to drop the ≥15% LSTM gate, write a short "negative result" addendum to the methodology paper, stop. **Do not cycle.**

**This plan is NOT about.** Building products (Plan B). Building a new model (NORTH_STAR Path 2 still applies — only architecture-equivalent variants of PitchGPT are in scope; no Mamba, SSM, diffusion, or alternative families). Retro-fitting outcome prediction (that's Plan B). Re-litigating the locked decision that the calibration claim alone is sufficient for the current paper.

---

## 2. Current state — exact numbers

All numbers below cite source files; do not edit them in flight.

### 2.1 Canonical 2025 holdout, matched 10K scale (the binding result)

Source: `results/pitchgpt/2025_holdout/report.md` (v2 ump, generated 2026-04-24T03:53Z).

| Model | Params | Holdout PPL | 95% CI | N pitches |
|---|---|---|---|---|
| PitchGPT v2 | 1,398,690 | 118.645 | 117.903 / 119.430 | 202,923 |
| LSTM (matched 10K) | 837,282 | 122.483 | 121.608 / 123.294 | 202,923 |
| Markov-2 | 0 | 344.347 | 341.929 / 346.705 | 202,923 |
| Heuristic | 0 | 469.899 | 467.290 / 472.550 | 202,923 |

Gates (binding): PitchGPT vs LSTM **+3.13%** (CI +2.19 / +4.05) — **FAIL** ≥15% spec. Markov +65.54% PASS. Heuristic +74.75% PASS.

### 2.2 v1 at 10K scale — independent replication of the small effect

Source: `results/pitchgpt/2025_holdout_v1_10k/report.md` (v1 no-ump, 2026-04-24T15:19Z).

- PitchGPT v1 PPL 119.829 (CI 119.091 / 120.601), LSTM 122.990 (CI 122.208 / 123.803).
- Gain vs LSTM: **+2.57%** (CI +1.68 / +3.43). Umpire feature is worth ~+0.56pp at 10K scale (within noise).

### 2.3 The historical 1K-vs-1K number (RETIRED, do not re-quote)

Source: `docs/awards/methodology_paper_pitchgpt.md` §3.1.

- PitchGPT 152.187 vs LSTM 176.554, **+13.80%** (CI +12.22 / +15.51) at 1,000-game train scale.
- This was retired 2026-04-24 as a small-sample artifact. The matched-10K runs cut the delta from ~14% to ~3%. Per `feedback_scale_verify_before_flagship.md`, this is the same small-sample collapse VWR underwent — NORTH_STAR locks both findings.

### 2.4 Hypothesis-relevant trajectory: 1K → 10K shrunk the gap

The gap COMPRESSED from ~14% (1K games) to ~3% (10K games). Two interpretations, both consistent with the data, opposite implications:

- **Interpretation A (LSTM caught up to PG with more data).** LSTM at 1K games is undertrained on 2,210-token vocab; LSTM at 10K closed most of its undertraining gap. PG architecture is not delivering structurally — it just looked good at 1K because the LSTM was data-starved.
- **Interpretation B (PG is also still undertrained at 10K).** Both architectures benefit from scale; PG plateaus later; full ~30K (7.3M-pitch) corpus would re-open the gap. This is the optimistic story for H3.

Plan must cleanly distinguish these interpretations — see H3.

### 2.5 Current architecture (`src/analytics/pitchgpt.py`)

- 4 layers, 4 heads, d_model=128, max_seq_len=256, dropout 0.1, total params 1,398,690.
- Composite token vocab 2,210 = 17 pitch types × 26 zones × 5 velocity buckets (`PitchGPTModel`, line 1095).
- 35-dim context vector: 12 count states + 3 outs + 8 runner states + 2 batter hands + 4 inning buckets + 5 score-diff buckets + 1 umpire scalar.
- **Pitcher identity is NOT an input feature** — by design, to enforce pitcher-disjoint generalization. Confirmed by inspection of `encode_context` and `context_to_tensor` (lines 528–610 of `pitchgpt.py`).
- Training: AdamW lr=1e-3, batch=32, grad-clip=1.0, 5 epochs, no LR schedule, no label smoothing.
- 10K-game training run wall-clock: 763 sec (~13 min) on RTX 3050 (`results/validate_pitchgpt_v1_10k_20260424T091314Z/training_summary.json`).

### 2.6 Training-curve evidence at 10K (v1)

From `training_summary.json`: best_val_loss 4.7247 at epoch 5 (final). Epoch 4 val_loss was 4.7447 (still improving), epoch 5 4.7247. **Val loss did not plateau within the 5-epoch budget** — the model may benefit from more epochs. This is direct evidence for H4 below.

---

## 3. Known vs assumed vs unknown

**Known (load-bearing facts, from cited results).**

- The 1K→10K compression is real and replicates across v1 and v2 (§2.1, §2.2).
- Calibration is stable across pre/post temperature scaling, across 2024 and 2025 holdouts, and across v1 vs v2 (ECE post-temp 0.0075–0.0098).
- Pitcher-disjointness is enforced by construction (`PitchSequenceDataset.fetch_pitcher_ids_for_seasons` + `exclude_pitcher_ids`, lines 803–836).
- The 5-epoch training schedule does not plateau val loss at 10K scale (§2.6).
- Markov-2 and heuristic gates pass with very wide margins. The architecture-vs-architecture (PG-vs-LSTM) competition is the only unresolved gate.

**Assumed (informed but not measured).**

- A matched-architecture LSTM is the "right" baseline. It captures recurrence without attention; it shares vocab + context. This was the spec author's choice (`docs/models/pitchgpt_validation_spec.md` Ticket #3) and we keep it.
- The 7.3M-pitch corpus from 2015–2024 would, at full scale, retain the same pitcher-disjoint structure (i.e. the holdout pitchers in 2025 are absent from any training subset we draw). This is structurally true given the 2025-only holdout cohort, so the assumption is essentially mechanical.
- Single seed (seed=42) results are representative of the underlying architecture. Multi-seed variance is unmeasured for the 10K runs. We will measure it as part of H1's first ablation.

**Unknown (would require measurement).**

- How much of the 1K→10K compression came from LSTM catching up vs PG saturating (§2.4 interpretations A vs B).
- Whether longer training (10+ epochs, with LR schedule) closes any of the current gap.
- Whether wider context (longer max_seq_len than 256) gives the transformer's attention something to actually use at PA scale.
- Whether adding a per-pitcher learned embedding (post-leakage-fix, this is allowed if we exclude held-out pitcher IDs from the embedding lookup) gives the transformer more signal than the LSTM can match through recurrence.
- Whether the composite-token vocabulary (2,210) is too coarse for fine-grained discrimination, or too fine for sample efficiency.

---

## 4. Hypotheses to test

Each hypothesis carries: stated mechanism, why it might work, why it might NOT, expected effect size, test design, expected wall-clock + GPU compute. Order is by expected information-per-cost — cheapest informative tests first.

### H1 — Multi-seed variance baseline + longer training schedule

**Mechanism.** Two interventions packaged together because they're cheap and they unblock interpretation of every later hypothesis.

(a) Re-run v2 at 10K with seeds {42, 7, 13} to measure single-seed noise. The current ~3% number is from one seed.
(b) Re-train v2 at 10K with epochs=15, AdamW + cosine LR schedule from 1e-3 to 1e-5, otherwise unchanged. Measure val loss curve and final 2025-holdout PPL.

**Why it might work.** Val loss did not plateau at epoch 5 (§2.6). Cosine schedule is the standard transformer protocol; we never used one. If 5-epoch undertraining is a meaningful chunk of the architecture-vs-architecture story, a 15-epoch run gets us closer to PG's "true" PPL on this corpus.

**Why it might NOT work.** Longer training on the *same* data could equally help LSTM (we'd need to symmetrize). LSTM also benefits from cosine schedules. If both improve proportionally the gap doesn't move. Also possible: PG starts overfitting at epoch 8-10 and val loss creeps back up.

**Expected effect size.** +1 to +3 pp on the LSTM gap if the transformer is data-undertrained relative to the LSTM at this scale; ~0 if both saturate together.

**Test.**
1. Three-seed v2 retrains at 10K, current 5-epoch schedule. Report mean ± 1σ on 2025-holdout PPL.
2. Single 15-epoch retrain with cosine schedule, seed=42. Compare to (1)'s mean.
3. Symmetrize: re-train LSTM at 15 epochs + cosine. Compare PG-vs-LSTM at the new schedule.

**Cost.** Per 10K-train: ~13 min (5 epochs) → ~40 min (15 epochs). Total: 3 × 13 + 1 × 40 + 1 × 40 = ~120 min wall-clock GPU. **Cheapest informative experiment.** Run first.

**Decision after H1.**
- If H1 alone closes to ≥10%: STOP, declare PARTIAL pass, decide whether to push for ≥15%.
- If H1 closes by <2pp: proceed with H1's longer schedule as the new baseline for H2–H8 (so they're not cheating by adding compute on top of an undertrained baseline).

### H2 — Wider model (d_model 128 → 256)

**Mechanism.** Doubling the hidden dimension quadruples FFN capacity (4× d_model) and doubles attention head dimensionality, giving the transformer more capacity to represent the joint distribution over the 2,210-token vocab.

**Why it might work.** A 1.4M-parameter transformer is small for 2,210-class softmax over 8M+ pitches in the 10K cohort. The output_head alone is 128 × 2,210 = 283K parameters; widening to 256 makes it 567K. The model is plausibly capacity-limited at d_model=128.

**Why it might NOT work.** The LSTM baseline is FIXED at 837K params; widening PG to 256 inflates it to ~5.5M, which is an unfair architecture comparison. Two ways to handle: (i) report PG-256 vs LSTM-837K (current spec's matched-vocabulary baseline; capacity-asymmetric), (ii) widen both. If the LSTM widens with PG, both gain. The structural-architecture-edge claim depends on PG winning at matched capacity, which H2 alone does not test.

**Expected effect size.** +2 to +4 pp on PPL gap if capacity is the bottleneck. 0 if not. Could go negative if 5-epoch budget is too short for the wider model to converge (interacts with H1).

**Test.**
1. Retrain v2 at 10K with d_model=256 (other dims unchanged), under H1's chosen schedule.
2. Retrain matched-LSTM at d_model=256 (also under H1's schedule).
3. Report both PG-128 vs LSTM-128 and PG-256 vs LSTM-256 deltas. The "fair" architecture-edge number is the 256-vs-256 delta.

**Cost.** Wider model is roughly 1.5–2× per-epoch cost. With H1's schedule: ~80 min PG + ~80 min LSTM. **Total ~160 min GPU.**

### H3 — Full corpus training (10K → ~30K games / 7.3M pitches)

**Mechanism.** Distinguish §2.4 interpretations A vs B. If gap re-opens at full scale, PG was never saturated — the architecture has more headroom. If gap stays at ~3%, PG-vs-LSTM is structurally bounded at that delta on this dataset.

**Why it might work.** 1K→10K shrunk gap from ~14% to ~3%. A naïve linear extrapolation says 10K→30K could shrink further, NOT widen. But: transformers historically scale BETTER than LSTMs with data (their capacity is more compute-bound, less data-bound). So the directional question is genuinely open.

**Why it might NOT work.** The 1K→10K trajectory was monotone-shrinking. If 30K continues that trend, gap goes to ~1% or 0. That would be a strong falsification of the "scale will save us" story.

**Expected effect size.** Either +5 pp (if H3 reverses the trend; transformer-saturates-later story) or -1 pp (if it continues; PG-LSTM-converge story). The sign is what matters; either is informative.

**Test.**
1. Pull all 2015–2022 pitcher-disjoint games, filtered to non-2025-holdout pitchers. Estimated cohort size: 2015–2022 has ~12.5M pitches across all pitchers; pitcher-disjoint exclusion removes ~10–15% (2025 debut pitchers are a small fraction). Net training corpus: ~10M pitches across ~25K–30K games.
2. Train PG and LSTM at full corpus, H1 schedule.
3. Report holdout PPL deltas.

**Cost.** ~3× the 10K cost = ~120 min PG + ~120 min LSTM under 5-epoch schedule, ~360 min under 15-epoch schedule. **Total 4–6 hours GPU.** Most expensive single hypothesis. Run AFTER H1 and H2 (which cheaply rule in or rule out simpler explanations).

### H4 — Longer max_seq_len (256 → 512 or 1024)

**Mechanism.** A transformer's attention can in principle leverage every prior pitch in the sequence. Current max_seq_len=256 truncates ~5% of training sequences (typical pitcher-game has 80–120 pitches; truncation hits multi-game contiguous slices, not single games). At 256, the attention is rarely binding — but if the model sees 512 or 1024, it can attend across multiple games for the same pitcher within a sequence.

**Why it might work.** Cross-game memory is a transformer advantage that LSTM cannot match within a fixed hidden state without specialized architectural tricks. If pitcher-style is auto-recovered from the prior game's pitches in-context, this looks like an unlearnable feature for the LSTM.

**Why it might NOT work.** Each "sequence" in `PitchSequenceDataset` is one (game, pitcher) tuple — they don't span multiple games (line 970 of `pitchgpt.py`: `df.groupby(["game_pk", "pitcher_id"], sort=False)`). max_seq_len=256 already covers the longest single (game, pitcher) sequence; doubling it adds no new context unless we change the grouping. **This hypothesis requires a code change to fuse multiple games per pitcher into a single longer sequence.**

**Expected effect size.** +2 to +5 pp IF the data pipeline change is implemented AND cross-game pitcher style is informative. 0 otherwise.

**Test.**
1. Code change: extend `PitchSequenceDataset._load` to optionally group by `pitcher_id` only (concatenate all that pitcher's games in date order), or to concatenate the prior-N games as context prefix.
2. Train PG with max_seq_len=1024 on the new pipeline. Train LSTM same way.
3. Compare deltas.

**Cost.** Code change ~1 day. Training ~2× per-epoch cost (longer sequences). With H1 schedule: ~80 min PG + ~80 min LSTM. **Total 1 day code + ~160 min GPU.**

### H5 — Per-pitcher learned embedding (auxiliary input)

**Mechanism.** Add a `pitcher_embedding(pitcher_id) → d_model` lookup; sum into the input embeddings alongside token + context + position. The model can learn pitcher-specific style profiles directly. Equivalent for the LSTM.

**Why it might work.** PG currently has zero pitcher identity feature. The 469 holdout pitchers in 2025 are unseen, so no embedding for them — but that's exactly the right behavior at OOS. For pitchers seen in training, the embedding could provide a strong pitcher-style anchor that the model otherwise has to reconstruct from sequence context.

**Why it might NOT work.** The 2025 holdout is pitcher-disjoint by spec — every test pitcher has an unseen embedding. If we use a default-value (zero or mean-embedding) at test time, the embedding contributes nothing to test PPL. The win, if any, would manifest as a *training-time* speedup (pitchers seen in training reach lower train loss faster) which doesn't necessarily help the spec gate. Worse: if embedding overfits to seen pitchers, train PPL drops but val/holdout PPL doesn't move, AND we may HURT calibration on the unseen-pitcher slice.

**Expected effect size.** +0 to +2 pp on the 2025 holdout. Could be the wrong axis to invest in — a per-pitcher embedding fundamentally bets on the in-training-population, while the spec gate measures out-of-training-population.

**Test.**
1. Add `nn.Embedding(num_pitchers, d_model)` to PG; pitcher_id input is mean-pooled at OOS time (use per-pitcher embedding mean across train cohort as the default).
2. Train PG-with-embedding at 10K, H1 schedule.
3. Train matched LSTM-with-embedding.
4. Report deltas. Also report calibration (we expect this could move ECE).

**Cost.** Code change 0.5 day. Training same as H1 (~80 min for both models). **Total 0.5 day + ~160 min GPU.**

### H6 — Tokenization variants

**Mechanism.** Current 2,210 = 17 × 26 × 5. Possibilities to test (in priority order):

(a) Drop velocity bucket: 17 × 26 = 442 tokens. Lower-resolution but better-populated.
(b) Coarser zone: 17 × 9 × 5 = 765 (3×3 grid + out-of-bound) or 17 × 5 × 5 = 425 (very coarse).
(c) Separate heads (factorized softmax): 17-class pitch_type head + 26-class zone head + 5-class velocity head, joint loss. Lower per-step compute, more sample-efficient because each axis sees more data per token.

**Why it might work.** The 2,210-class softmax has long tails — many tokens see <50 training examples. Sample efficiency is the bottleneck. Coarser tokens or factorized heads spread the same training data over fewer effective classes per axis, improving each.

**Why it might NOT work.** If the joint distribution structure (e.g., specific pitch-type-zone combinations like "FF up-and-in" that have outsized predictive value for the next pitch) is informative, factorizing kills it. Coarser zones lose calibration granularity at the velocity-zone level.

**Expected effect size.** Factorized softmax: +1 to +3 pp on PPL with possibly improved calibration. Coarser tokens: 0 to +2 pp at the cost of representational fidelity. The 2,210-class softmax is also where calibration was measured; changing the head may invalidate the existing temperature scalar.

**Test.**
1. Reuse same 10K cohort.
2. Train PG-factorized (3 separate output heads, joint cross-entropy with equal weights) at H1 schedule.
3. Train matched LSTM-factorized.
4. Report PPL on the equivalent 2,210-token unfactorized space (multiply per-axis probabilities to get joint).
5. Report calibration on the unfactorized space.

**Cost.** Code change ~1 day. Training same as H1. **Total 1 day + ~160 min GPU.** Lower-priority than H1–H4 because it changes the architecture so much that the calibration-claim transfer is questionable.

### H7 — Regularization tuning

**Mechanism.** Inspect train vs val curves under H1's longer schedule. If train loss dives below val loss meaningfully, increase dropout (currently 0.1) and/or add weight decay (currently 0). Lower-priority hypothesis because §2.6 evidence is that the model is UNDER-trained, not over-trained.

**Why it might work.** If H1's longer schedule shows late-epoch val regression, regularization buys back the late-epoch gain.

**Why it might NOT work.** Current evidence is undertraining, not overfitting. Regularization on an undertrained model HURTS.

**Expected effect size.** -1 pp to +1 pp. Diagnostic, not load-bearing.

**Test.** Read H1's epoch-by-epoch val loss curve. If overfitting symptoms appear, run a small grid (dropout in {0.1, 0.2, 0.3}, weight_decay in {0, 1e-5, 1e-4}). Otherwise SKIP H7.

**Cost.** Conditional on H1 outcome. If skipped, 0. If run, ~9 small training runs at H1 schedule = ~6 hours GPU.

### H8 — Score-diff and umpire-feature ablation under H1's longer schedule

**Mechanism.** Re-run the existing context ablation (`docs/models/pitchgpt_calibration_ablation_results.md`) at H1's longer schedule. The known result is that context features add only 1–7% lift over tokens-only at 1K and 10K games / 5 epochs. With longer training, does the context-feature lift grow? If so, that's evidence the model's representational capacity for context is bottlenecked by training time, not architecture.

**Why it might work.** Confirms H1's "undertrained" hypothesis from a different angle. Doesn't directly close the LSTM gap, but tells us where the next investment should go.

**Why it might NOT work.** Diagnostic only — no PPL improvement directly.

**Expected effect size.** 0 pp on the LSTM gate; informative for future architecture decisions.

**Test.** Re-run `scripts/pitchgpt_ablation.py` (per `docs/models/pitchgpt_calibration_ablation_results.md`) at H1's longer schedule.

**Cost.** ~1 day of agent time; covered by H1's compute budget if scheduled together.

---

## 5. Baselines

Every variant from H1–H8 must be evaluated against **all five** of:

1. **Matched-scale LSTM** at the same training scale and schedule (the binding head-to-head). For H2 (wider), this means LSTM also widened. For H3 (full corpus), LSTM also at full corpus. For H4 (longer seq), LSTM same.
2. **Markov-2** on the same training cohort (sanity floor).
3. **Heuristic** (frequency baseline) on the same training cohort (deeper sanity).
4. **Current pitchgpt_v2.pt at matched 10K scale** (the current locked checkpoint — the bar to beat to declare any improvement real).
5. **Multi-seed v2 baseline from H1(a)** (so single-seed noise doesn't masquerade as a delta).

A hypothesis is considered to "work" only if it beats all five with non-overlapping CIs. A win against just (4) and (5) but not (1) is uninformative for the spec gate.

---

## 6. Evaluation cohort

**Test set.** 2025 pitcher-disjoint holdout per `scripts/pitchgpt_2025_holdout.py`. 5,915 sequences from 469 pitchers debut-2025 (or first-rostered-2025 with no 2015–2022 appearance). 202,923 non-PAD pitch tokens.

**Why this and not other splits.**
- This is the canonical published holdout. Methodology paper's calibration claim, the matched-10K LSTM gap, the 2025_holdout reports, and the sampling-fidelity report all use this cohort.
- Pitcher-disjointness is enforced via `PitchSequenceDataset.fetch_pitcher_ids_for_seasons` + `exclude_pitcher_ids` (lines 803–836 of `pitchgpt.py`).
- Drift from train (2015–2022) to test (2025) is meaningful — a 2-year gap, a partial-ABS season, and a fresh pitcher cohort.

**Statistics.** Bootstrap 1,000 pitch-level resamples for 95% CIs on perplexity. Use the same `bootstrap_perplexity_ci` pattern from `scripts/pitchgpt_2025_holdout.py`.

**Multiple-comparison.** Up to 8 hypotheses × multiple sub-experiments → ~30 comparisons. Apply Bonferroni at α/30 = 0.0017 → 99.83% CIs for any HEADLINE claim ("PG-with-X beats LSTM by Y%"). Per-experiment 95% CIs are still reported for diagnostic purposes; the claim threshold is the corrected one.

**Calibration re-check.** For every variant that lands a PPL improvement, re-run temperature scaling on a 2023 val slice and re-measure ECE on 2025. Variants with post-temp ECE > 0.0125 are disqualified (calibration claim takes priority).

---

## 7. Gate criteria — explicit numbers

| Verdict | LSTM perplexity improvement on 2025 holdout (point) | 95% CI lower bound | Calibration ECE post-temp |
|---|---|---|---|
| **PASS (full)** | ≥ +15% | ≥ +10% | < 0.0125 |
| **PARTIAL** | ≥ +10% | ≥ +5% | < 0.0125 |
| **FAIL** | < +10% | OR < +5% | OR ≥ 0.0125 |

PARTIAL is meaningful: a 10–15% gap closes most of the spec shortfall and is publishable as a substantial improvement. The flagship narrative would update from "calibrated, narrowly beats LSTM" to "calibrated, materially beats LSTM" — a real strengthening, but not a "beats every baseline by spec margin" rescue.

PASS rebuilds the original spec claim (the methodology paper's §3.1 line "the LSTM upper CI bound (15.51%) does reach the spec, but the point estimate and lower bound do not" becomes obsolete). FAIL locks the narrowed claim permanently.

---

## 8. Kill criteria — explicit and binding

After H1, H2, H3 are evaluated (the high-information triad), check the score:

- **If max delta from H1 alone ≥ 10%**: STOP. Declare PARTIAL pass, write up. Do NOT chase the extra 5% with H4–H8 unless the user explicitly requests a follow-up plan.
- **If max delta from H1+H2+H3 < 5%**: STOP after H3. Do not run H4–H8. The LSTM gap is structural at this corpus scale and architecture family. Declare narrowed-claim permanent. Write a 2-page negative-result addendum to the methodology paper. Edit the validation spec to drop the ≥15% LSTM gate.
- **If max delta from H1+H2+H3 in [5%, 10%)**: ONE more pass. Run the cheapest of H4–H8 still expected to be most informative (likely H4 max_seq_len if the H1–H3 epoch curves suggest cross-game context is the bottleneck, otherwise H5 pitcher embedding). After ONE more, regardless of outcome: STOP. Write up.

**Hard cycle limit.** No hypothesis is re-tried with tweaked hyperparameters more than ONCE. If H1's longer schedule doesn't help, "longer with cosine-warmup-100" is not a separate hypothesis — it's giving up.

**Calibration kill.** ANY variant that posts a PPL improvement but breaks the calibration ECE budget (>0.0125) is REJECTED. The flagship is calibration-anchored; we do NOT trade calibration for accuracy.

---

## 9. Sequencing

### 9.1 Independent vs sequential

| Hypothesis | Depends on | Parallelizable? |
|---|---|---|
| H1 | nothing | yes (3 seeds + 1 schedule = 4 parallel runs if GPU available) |
| H2 | H1 chosen schedule | yes (PG-256 + LSTM-256 in parallel) |
| H3 | H1 chosen schedule | yes (PG-full + LSTM-full in parallel) |
| H4 | H1, code change | sequential (depends on data-pipeline change) |
| H5 | H1, code change | sequential (depends on embedding-layer change) |
| H6 | H1, code change | sequential (depends on factorization change) |
| H7 | H1 (conditional on overfit symptoms) | conditional |
| H8 | H1 schedule | yes (parallel with H2/H3) |

GPU constraint: RTX 3050 has 8GB; widening to d_model=256 increases memory ~2.5×. Cannot run two 256-dim trainings in parallel. ONE training at a time on GPU.

### 9.2 Wall-clock estimates

**Sequential (one experiment at a time on the single GPU).**
- H1: ~120 min
- H2: ~160 min
- H3: ~360 min
- H4 (if triggered): code change 1 day + ~160 min GPU
- H5 (if triggered): code change 0.5 day + ~160 min GPU
- H6 (if triggered): code change 1 day + ~160 min GPU
- H7 (if triggered): ~6 hours
- H8: covered by H1's compute

Worst case (PASS or PARTIAL achieved at H3): **~640 min GPU = ~11 hours on the GPU**, spread over 2-3 days of session time including report-writing.

Worst case (FAIL after all 8 hypotheses): **~640 min GPU + 3 days code work** = ~1 work week.

Expected case: H1 + H3 either win or lose decisively → 480 min GPU = ~8 hours, 1 work day.

### 9.3 Recommended sequence

1. **H1 first** (cheapest, unblocks everything else). Decision after H1 per §8.
2. **H3 second** (most informative single experiment — distinguishes the 1K→10K interpretation A vs B). Run in parallel with H8 (ablation re-run) on next GPU slot.
3. **H2 third** (capacity question, unaffected by H1/H3 outcomes).
4. Conditional H4/H5/H6/H7 per §8.

---

## 10. What this plan does NOT do

- **Does not build products.** That's Plan B (outcome head + sim-engine wrapping) and Plan C (sim-engine API).
- **Does not introduce new model families.** Mamba, S4/S5, retentive networks, diffusion language models are out of scope. Per NORTH_STAR Path 2 and `feedback_validation_over_models.md`.
- **Does not retrofit outcome prediction.** Plan B owns that work; this plan is purely about closing the next-token PPL gap with pitch-token-only architectures.
- **Does not relitigate the calibration claim.** ECE is a guardrail (must not break) but not a target (we don't try to make it better). The flagship claim is anchored at ECE 0.0075 and any variant that doesn't preserve that is rejected on the spot.
- **Does not chase the spec gate to the death.** The kill criteria in §8 are non-negotiable. After ≤8 hypotheses, this work ends — either with an updated paper claim or with a narrowed-claim-locked-in addendum.

---

*Document author: Claude (session 2026-04-24). Doc-only research plan; awaits user greenlight before any execution agent fires. Leave unstaged.*
