# PitchGPT: A Calibrated Decoder-Only Transformer for MLB Pitch Sequences

**Author:** Hunter Briggs

---

## Abstract

PitchGPT is a decoder-only transformer (4 layers, 4 heads, 128 hidden dimensions, 2,210-token composite vocabulary) trained on Statcast pitch sequences with integrated situational context. To our knowledge this is the first public application of causal-masked language-modeling to MLB pitch-type-by-zone-by-velocity composite tokens. We evaluate on a strict pitcher-disjoint 2025 out-of-sample holdout (334 pitchers new to 2025, zero shared game_pks or pitcher_ids with training), after a 409-pitcher leakage bug was identified and fixed. PitchGPT achieves ECE = 0.0098 post-temperature-scaling (T = 1.1201) — calibration survives the temporal shift from the 2023 validation slice into the 2025 holdout (ECE pre-temp 0.0201). On 53,723 scored pitches, PitchGPT's perplexity of 152.19 beats a fresh 2-layer LSTM baseline by 13.80% (95% CI 12.22 / 15.51), Markov-2 by 56.85%, and a frequency-heuristic baseline by 67.75%. The LSTM gate (spec 15%) fails by 1.2 percentage points, and a downstream pitch-outcome-bucket utility test returns a statistical tie with the LSTM (log-loss delta 0.00007, CI spans zero). We present PitchGPT as a calibrated sequence-modeling achievement, not a decision-utility replacement.

---

## 1. Edge: Calibrated pitch-distribution modeling

Public next-pitch prediction work in baseball analytics is typically uncalibrated — it optimizes top-1 accuracy or raw cross-entropy and reports nothing about whether the predicted probability vector over the pitch space is honest. This is a material gap. Any downstream consumer of pitch-distribution probabilities — a batter-facing expected-outcome model, a game-theoretic pitch-sequencing analysis, a pitcher-development workload tool that weights "expected pitch type given count" — implicitly assumes the probabilities are calibrated. When they are not, downstream log-losses, Brier scores, and policy decisions all inherit miscalibration as a silent bias.

PitchGPT targets that gap. The edge claim is *not* "best accuracy on next-pitch" — the 2,210-token composite vocabulary makes absolute top-1 accuracy structurally low (3.4% on the 2025 holdout, against a uniform-prior rate of 0.045%). The edge is that PitchGPT's top-1 probability estimates land within empirical accuracy to within 0.01 in Expected Calibration Error on a strict out-of-sample holdout season drawn from 334 pitchers that never appeared in training. In the bin where 97.7% of holdout pitches live (mean confidence 0.042, n = 52,476), empirical accuracy is 0.0337 post-temperature-scaling — inside 0.01 of the predicted mean. The model is honest about what it knows.

Three features of the calibration result matter:

1. **Temporal robustness.** The temperature scalar (T = 1.1201) was fit on the 2023 validation slice; applying it to 2025 holdout logits produced ECE 0.0098. Compared to the 2024 internal-test calibration from the validation run (ECE pre-temp 0.0203, post-temp 0.0096 at T = 1.1246), the temperature migrates cleanly across a two-year temporal gap. This is the signal that PitchGPT's miscalibration is a stable architectural property, not a season-specific artifact.

2. **Pitcher-disjointness.** The 334 holdout pitchers are new to 2025 — none appear in the 2015-2022 training cohort (1,426 pitchers) or the 2023 temperature-fit slice (115 pitchers). Calibration is therefore measured on a population the model cannot have memorized.

3. **Honest about downstream.** Section 4 below shows a downstream pitch-outcome-bucket test returns a tie with the LSTM. The calibration narrative stands on its own; we are not smuggling in a utility claim.

The thesis of this paper is a bounded one: PitchGPT is a calibrated sequence-modeling artifact on which future pitch-distribution consumers can be built, not a finished decision-making tool.

---

## 2. Methodology

### 2.1 Architecture

PitchGPT is a decoder-only transformer defined in `src/analytics/pitchgpt.py` (`class PitchGPTModel`, line 432). The network has 4 transformer decoder layers, 4 attention heads per layer, hidden dimension d_model = 128, and max sequence length = 256. Causal self-attention is enforced via a standard upper-triangular mask so that predictions at position t depend only on positions ≤ t. Total parameters: 1,398,562.

### 2.2 Token vocabulary

The core methodological novelty is the composite-token vocabulary. Rather than predicting pitch type and pitch location independently, PitchGPT tokenizes each pitch as a single integer ID in a 2,210-token space constructed as:

```
token_id = pitch_type_idx * 130 + zone_idx * 5 + velocity_bucket_idx
```

where `pitch_type_idx` ranges over 17 values (16 MLB pitch types plus unknown), `zone_idx` ranges over 26 values (a 5×5 plate grid plus an out-of-bounds bucket), and `velocity_bucket_idx` ranges over 5 velocity bins. The encoding is deterministic and invertible. `PAD_TOKEN = 2210` pads short sequences to a batch-constant length. This composite-token framing lets a single softmax predict a joint distribution over (pitch type, zone, velocity bucket) without needing separate heads or a product-of-experts factorization.

### 2.3 Situational context

Alongside the sequence of composite tokens, PitchGPT consumes a 34-dimensional one-hot situational context vector at each position: count (12 states), outs (3), base-runner layout (8), batter handedness (2), inning bucket (4), and score differential bucket (5). The context is projected to d_model = 128 and added to the token embedding and positional embedding before the first transformer layer. An earlier ablation study (see `docs/models/pitchgpt_calibration_ablation_results.md`) established that at the 10,000-training-game scale the full 34-dim context adds +6.85% perplexity over a tokens-only variant but only +4.14% over a count-and-outs-only variant; the full context is therefore retained as an input feature but the narrative weight of the "rich situational-awareness" claim is bounded.

### 2.4 Pitcher-disjoint split (corrected)

The first validation run (`validate_pitchgpt_20260418T132155Z`) enforced game_pk-disjointness across train, validation, and test but did not enforce pitcher-disjointness — 409 pitchers appeared in both the 2015-2022 train set and the 2024 test set. This leak manifested most severely in the `identity_only` ablation variant, which achieved test perplexity 52.99 (against the full model's 148.47) by memorizing pitcher identity. The `PitchSequenceDataset._load()` method was modified to enforce pitcher-disjoint splits after date-based allocation. The corrected run (`validate_pitchgpt_20260418T150305Z`) reports 0 shared pitchers train-test and the `identity_only` variant collapsed to test perplexity 2583.32 (+94.06% worse than full) — the expected honest degradation. All numbers in this paper derive from post-fix checkpoints.

The 2025 holdout split used in `scripts/pitchgpt_2025_holdout.py` inherits this discipline: the 334 holdout pitchers are pitchers who made their first appearance in 2025 and are therefore mechanically absent from the 2015-2022 training window.

### 2.5 Baselines

Three baselines are evaluated against PitchGPT on the same 2025 holdout pitches:

- **LSTM (`src/analytics/pitch_lstm.py`)**: 2-layer LSTM, 128 hidden units, same 2,210-token vocabulary and situational-context inputs as PitchGPT, trained from scratch on the same 2015-2022 pitcher-disjoint cohort. 837,154 parameters. 5 epochs, AdamW lr 1e-3, early-stopped on 2023 validation loss.
- **Markov-2 (`src/analytics/pitch_markov.py`)**: pitch-type-only second-order n-gram fit on the same training sequences. 0 trained parameters (closed-form transition table).
- **Heuristic (`src/analytics/pitch_markov.py`, `HeuristicBaseline`)**: fixed pitch-type distribution (fastball ~0.50, breaking ~0.30, offspeed ~0.20) derived from training-cohort frequencies. 0 parameters.

### 2.6 Calibration via temperature scaling

Post-hoc calibration uses a single-scalar temperature fit via LBFGS on the 2023 validation logits (`src/analytics/pitchgpt_calibration.py`), then applied unchanged to the 2025 holdout. The optimal temperature T = 1.1201 is very close to the 2024 calibration fit (T = 1.1246), consistent with mild over-confidence being a stable architectural property.

### 2.7 Training hyperparameters

Seed 42, AdamW lr 1e-3, batch size 32, gradient clip 1.0, 5 epochs on the 2015-2022 training slice (1,000 games, 8,412 sequences after pitcher-disjointness). No learning rate schedule, no label smoothing (the model is under-confident, not over-confident — see §3.3). Checkpoint: `results/validate_pitchgpt_20260418T150305Z/pitchgpt_full.pt`. Hardware: RTX 3050 (CUDA) for training; CPU inference works but is ~10× slower.

---

## 3. Validation

### 3.1 Spec gate table

| Gate | Spec threshold | Measured (2025 holdout) | Verdict |
|---|---|---|---|
| Leakage — shared game_pks | = 0 | 0 | PASS |
| Leakage — shared pitcher_ids (train-holdout) | = 0 | 0 | PASS |
| Perplexity — PitchGPT vs LSTM | ≥ 15% | +13.80% (CI +12.22 / +15.51) | **FAIL** (1.2pp short of point) |
| Perplexity — PitchGPT vs Markov-2 | ≥ 20% | +56.85% (CI +55.96 / +57.64) | PASS |
| Perplexity — PitchGPT vs Heuristic | ≥ 25% | +67.75% (CI +67.18 / +68.29) | PASS |
| Calibration ECE pre-temperature | < 0.10 | 0.0201 | PASS |
| Calibration ECE post-temperature | < 0.10 | 0.0098 (T = 1.1201) | PASS |

Overall: fails the LSTM gate by 1.2 percentage points on the point estimate. The 95% bootstrap upper bound (15.51%) does reach the spec, but the point estimate and lower bound do not, and the standard for the gate is the point estimate. This is reported honestly — no spin.

### 3.2 Out-of-sample perplexity comparison (2025)

Holdout: 500 game_pks drawn from 2025, yielding 1,535 sequences and 53,723 non-PAD pitch tokens across 334 pitchers who are disjoint from the training pitcher cohort. Bootstrap CIs from 1,000 pitch-level resamples.

| Model | Parameters | Holdout perplexity | 95% CI | Δ vs PitchGPT |
|---|---|---|---|---|
| **PitchGPT** | 1,398,562 | **152.187** | 150.086 / 154.265 | — |
| LSTM | 837,154 | 176.554 | 174.058 / 179.036 | +13.80% (CI +12.22 / +15.51) |
| Markov-2 | 0 | 352.657 | 348.082 / 357.483 | +56.85% (CI +55.96 / +57.64) |
| Heuristic | 0 | 471.856 | 466.807 / 476.669 | +67.75% (CI +67.18 / +68.29) |

All three baseline comparisons have tight, non-overlapping CIs. The heuristic and Markov-2 results are clean wins with large margins and no CI ambiguity. The LSTM result is the load-bearing edge claim — and it falls short of the 15% spec by a non-trivial margin.

### 3.3 Calibration on 2025 out-of-sample

53,723 holdout pitches scored. ECE computed with 10 equal-width confidence bins.

| Metric | Pre-temperature | Post-temperature (T = 1.1201) |
|---|---|---|
| Expected Calibration Error (10 bins) | 0.0201 | **0.0098** |
| Top-1 accuracy | 0.0341 | 0.0341 (invariant under monotonic rescaling) |
| Mass in bin [0.0, 0.1) | 50,813 / 53,723 (94.6%) | 52,476 / 53,723 (97.7%) |

The reliability diagram (`results/pitchgpt/2025_holdout/reliability_2025.html`) shows the post-temperature predicted distribution is diffuse: 97.7% of 2025 pitches receive a top-1 probability below 0.1, with mean predicted confidence 0.042 against an empirical accuracy of 0.034 in that bin. Temperature T = 1.1201 flattens a handful of over-committed predictions in the [0.1, 0.3) bins. Because the predicted distribution is already close-to-diagonal pre-temperature, temperature scaling buys another 0.01 ECE — not an order-of-magnitude correction.

Cross-season comparison:

| Temporal window | ECE pre-temp | ECE post-temp | T |
|---|---|---|---|
| 2024 internal test | 0.0203 | 0.0096 | 1.1246 |
| 2025 held-out OOS | 0.0201 | 0.0098 | 1.1201 |

The near-identical numbers across two non-overlapping seasons (both pitcher-disjoint) are the strongest single piece of evidence that PitchGPT's calibration is a stable architectural property, not a selection artifact of any single test window.

### 3.4 Downstream pitch-outcome-bucket utility

The downstream utility test (`scripts/pitchgpt_downstream_utility.py`) asks whether PitchGPT's calibrated next-pitch distribution drives better predictions of actual pitch outcomes — a 5-class target: `called_strike`, `ball`, `foul`, `in_play`, `swinging_strike`. Three XGBoost classifiers (400 trees, depth 6, lr 0.08, early-stopping on a 10% train slice) are trained with identical hyperparameters and scored on the same 2025 holdout (51,925 targets):

| Variant | Feature dim | Log-loss (95% CI) | Brier (95% CI) | Accuracy (95% CI) |
|---|---|---|---|---|
| null_situational_only | 34 | 1.5207 (1.5173 / 1.5245) | 0.7616 (0.7598 / 0.7633) | 0.3591 (0.3549 / 0.3632) |
| pitchgpt_plus_situational | 85 | 1.5202 (1.5164 / 1.5239) | 0.7615 (0.7597 / 0.7631) | 0.3592 (0.3550 / 0.3634) |
| lstm_plus_situational | 85 | 1.5202 (1.5165 / 1.5239) | 0.7614 (0.7599 / 0.7630) | 0.3594 (0.3556 / 0.3637) |

Pairwise deltas (500 paired bootstrap resamples):

| Comparison (A vs B) | Δ Log-loss (95% CI) | Δ Brier (95% CI) | Δ Accuracy (95% CI) |
|---|---|---|---|
| pitchgpt_vs_lstm | +0.00007 (-0.00051 / +0.00074) | +0.00011 (-0.00018 / +0.00040) | -0.00015 (-0.00081 / +0.00052) |
| pitchgpt_vs_null | -0.00042 (-0.00118 / +0.00040) | -0.00010 (-0.00039 / +0.00020) | +0.00014 (-0.00045 / +0.00069) |
| lstm_vs_null | -0.00048 (-0.00134 / +0.00037) | -0.00021 (-0.00054 / +0.00012) | +0.00029 (-0.00031 / +0.00104) |

Every CI brackets zero on every metric. The downstream pitch-outcome-bucket task is a statistical tie — PitchGPT's calibrated distribution does not beat either the LSTM's distribution or a situational-features-only null. The majority-class (`ball`) baseline is 35.91% accuracy; all three trained models land at 35.92-35.94%. The downstream ceiling for this target is essentially at majority class.

High-confidence subsets do not rescue the picture. At a top-1 cutoff of 0.10 (4.6% coverage, 2,389 pitches), the PitchGPT-vs-LSTM log-loss delta actually flips in LSTM's favor (+0.0017, CI [-0.0019, +0.0059]); the top-10%-by-top-1 subset (5,193 pitches, threshold 0.0819) shows the same directionless tie. PitchGPT is not systematically more useful on pitches where it is more confident.

### 3.5 Leakage audit

| Pair | Shared game_pks | Shared pitcher_ids |
|---|---|---|
| train ↔ val | 0 | 0 |
| train ↔ holdout (2025) | 0 | 0 |
| val ↔ holdout | 0 | 0 |

Cohort sizes: train 1,426 pitchers / 1,000 game_pks / 8,412 sequences; val 115 pitchers / 300 game_pks / 469 sequences; holdout 334 pitchers / 500 game_pks / 1,535 sequences. The 334 holdout pitchers are those who debuted in 2025 or were rostered in 2025 without appearing in 2015-2022; by construction they could not have been seen during training.

---

## 4. Limitations

**LSTM gate fails by 1.2 percentage points.** The spec required PitchGPT to beat the LSTM baseline by ≥ 15% on 2025 holdout perplexity. The observed delta is 13.80% with 95% CI [12.22, 15.51]. The upper CI bound reaches the spec; the point and lower bound do not. This is a thin, honest miss and is reported as such. The margin is real (CIs non-overlapping between models) and the direction is correct — PitchGPT does beat the LSTM — but the claim "beats LSTM by ≥ 15%" is not supported.

**Downstream pitch-outcome-bucket prediction is a tie.** The calibrated pitch distribution from PitchGPT does not translate into better 5-class pitch-outcome predictions compared with either the LSTM baseline or a situational-features-only null. Log-loss and Brier CIs span zero on every pairwise comparison. Three non-exclusive explanations:

1. *Target is too coarse.* A 5-class bucket (called_strike, ball, foul, in_play, swinging_strike) may be too high-level for pitch-type-zone-velocity distinctions to matter — e.g., a well-located fastball and a well-located slider both produce "ball" roughly 35% of the time in this dataset.
2. *Situational context saturates.* The 34-dim one-hot count-outs-runners-handedness-inning-score-diff feature set is already a powerful predictor of outcome class distribution. The XGB's marginal accuracy over a majority-class predictor is only 0.3% — there is very little headroom for a pitch-distribution feature to fill.
3. *Ceiling is low.* Majority-class prediction is 35.91%. Even a perfect pitch-distribution oracle would not produce a large log-loss improvement on this particular target.

**High-confidence subset does not rescue.** At the 0.10 cutoff, the PitchGPT-vs-LSTM log-loss margin reverses in LSTM's favor (though CIs still bracket zero). PitchGPT is not producing a differentially useful signal on the pitches it is most confident about — at least not for this downstream target.

**Thin architectural margin against a 2-layer LSTM.** 13.80% is real but thin. A vanilla 2-layer LSTM with 128 hidden units captures most of the available next-pitch sequence structure at this dataset scale. The transformer architecture may not be earning its 1.7× parameter count over the LSTM — at least not in the 1,000-training-game regime this paper evaluates. Prior ablation work (`pitchgpt_calibration_ablation_results.md`) at 10,000-game scale widened the transformer's edge over a tokens-only variant from +0.88% to +6.85%, which is qualitatively consistent with the hypothesis that more training data would widen the LSTM gap too — but that is speculative until demonstrated.

**Single 2025 OOS window.** Calibration robustness rests on the 2024 vs 2025 temperature-agreement numbers. A second full-season holdout in 2026 will widen the evidence base for the claim "calibration survives temporal shift." Until then the claim is directionally supported but not multiply-confirmed.

---

## 5. Reproducibility

### 5.1 Code

All code is in this repository at commit-time (`main` branch).

- Model: `src/analytics/pitchgpt.py`
- LSTM baseline: `src/analytics/pitch_lstm.py`
- Markov / heuristic baselines: `src/analytics/pitch_markov.py`
- Temperature scaling: `src/analytics/pitchgpt_calibration.py`
- Feature ablation run: `scripts/pitchgpt_ablation.py`
- 2025 holdout evaluation: `scripts/pitchgpt_2025_holdout.py`
- Downstream utility: `scripts/pitchgpt_downstream_utility.py`
- Calibration reliability analysis: `scripts/pitchgpt_calibration_analysis.py`

### 5.2 Commands to reproduce each table

Table 2 (perplexity) and Table 3 (calibration):

```
python scripts/pitchgpt_2025_holdout.py \
  --checkpoint results/validate_pitchgpt_20260418T150305Z/pitchgpt_full.pt \
  --max-train-games 1000 --max-val-games 300 --max-holdout-games 500 \
  --seed 42
```

Writes `results/pitchgpt/2025_holdout/perplexity_comparison.json`, `calibration_2025.json`, `reliability_2025.html`, `report.md`.

Table 4 (downstream utility):

```
python scripts/pitchgpt_downstream_utility.py \
  --checkpoint results/validate_pitchgpt_20260418T150305Z/pitchgpt_full.pt \
  --max-train-games 1000 --max-val-games 300 --max-holdout-games 500 \
  --seed 42
```

Writes `results/pitchgpt/downstream_utility/downstream_comparison.json`, `calibration_utility_subset.json`, `report.md`.

### 5.3 Checkpoints

- PitchGPT full checkpoint: `results/validate_pitchgpt_20260418T150305Z/pitchgpt_full.pt` (5.3 MB). Trained 2015-2022 with pitcher-disjoint splits enforced via the corrected `PitchSequenceDataset._load()`.
- LSTM baseline: retrained from scratch in each run of `pitchgpt_2025_holdout.py`; no persistent checkpoint is committed (5-epoch training takes ~30 seconds on RTX 3050).

### 5.4 Hardware and determinism

Training and evaluation used a single NVIDIA RTX 3050 with CUDA 12.1. Seed 42 is propagated to numpy, torch, and torch.cuda. CPU-only inference works but runs ~10× slower on the 2025 holdout. CPU vs CUDA bit-for-bit parity has not been validated; numerical agreement to within ~1e-5 is expected but is not a tested guarantee.

---

## 6. Discussion

PitchGPT lands as a calibrated-sequence-modeling artifact, not a finished decision-making tool. The ECE = 0.0098 result on 2025 out-of-sample data, combined with the clean migration of the temperature scalar from 2024 to 2025 (both within 0.01 of each other), establishes that the composite-token decoder-only architecture produces probabilities that can be treated as honest estimates by downstream consumers. This matters mainly for future work, not for the current paper.

Three classes of downstream consumer would benefit:

1. **Game-theoretic pitch-sequencing analyses.** Prior work on pitcher strategy typically treats pitch-type choice as an independent categorical; PitchGPT supplies a joint distribution over pitch type, zone, and velocity bucket that preserves the correlations among them. Honest calibration is required for those joint probabilities to feed any Bayes-optimal batter-response policy.

2. **Batter-facing expected-outcome models.** Any model that predicts contact quality, barrel probability, or run-expectancy change *conditional on pitch characteristics* can consume PitchGPT's per-pitch distribution as a feature. The downstream utility test in §3.4 shows this does not work on a coarse 5-class pitch-outcome bucket target — but it leaves open whether a more granular target (exit-velocity bucket, barrel classification, plate-appearance outcome) would surface the calibration value.

3. **Pitcher-development workload tools.** Expected-pitch-mix given count-state is a natural feature for workload-risk models, injury-precursor detection, and pitch-design evaluation. The situational-context branch of PitchGPT is the direct supplier of those expectations.

### Future work

- **Re-run downstream utility on a different target.** Plate-appearance outcome (5 classes: strikeout, walk, single, extra-base hit, out-in-play) is a finer-grained target than pitch-level outcome and may surface the calibration value. Exit-velocity bucket or launch-angle bucket, conditional on in-play, is another natural candidate.
- **Longer training run to test LSTM-gap asymptotics.** The 1K vs 10K prior ablation showed the transformer's edge over tokens-only widening 7.8× with 10× data. If the LSTM gap widens similarly with 10K-game training, the spec gate would likely pass cleanly at that scale.
- **Transformer ensembles.** Two independent seeds, averaged at the logit level and re-temperature-scaled, would test whether the calibration result is already near-optimal for this architecture or whether there is residual diversity to exploit.
- **A second full-season OOS window in 2026.** This is the cleanest strengthening of the calibration-survives-temporal-shift claim.

The paper's strength is its honesty. The LSTM gate misses by 1.2 percentage points; the downstream utility test returns a tie; the transformer architecture does not clearly dominate a 2-layer LSTM on this dataset. What remains is the calibration result and the baseline comparisons against Markov-2 and the frequency heuristic, both of which pass decisively. That is enough to publish PitchGPT as a calibrated sequence-modeling contribution — and not enough to publish it as anything more.
