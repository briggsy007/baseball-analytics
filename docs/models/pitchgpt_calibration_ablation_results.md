# PitchGPT — Calibration and Ablation Results

**Tickets:** spec #6 (feature ablation), spec #7 (calibration audit).
**Run:** 2026-04-17 — canonical 5-epoch, seed=42, 2015-2022 train / 2023 val / 2024 test; 1000/300/300 games per split.
**Artifacts:** `results/run_5epoch/pitchgpt_calibration.json`, `results/run_5epoch/pitchgpt_reliability.html`, `results/run_5epoch/pitchgpt_ablation_metrics.json`.

## Methodology

We trained four PitchGPT variants on an identical date-based leakage-clean split, recorded test perplexity for each, and ran a post-hoc calibration audit on the full variant's checkpoint. Calibration computes top-1 reliability on the 2024 test pitches and fits a single temperature scalar on the 2023 validation logits via LBFGS. All hyperparameters (seed 42, 5 epochs, batch 32, AdamW lr 1e-3, grad clip 1.0) match the canonical baselines run so ablation drops and calibration numbers are directly comparable to the PitchGPT-vs-LSTM headline result.

## Calibration

The full-variant checkpoint (test perplexity 145.5, 85,083 test pitch tokens) is **already well-calibrated** — no temperature scaling required.

| Metric | Pre-temperature | Post-temperature (T=1.072) |
|---|---|---|
| Expected Calibration Error (ECE, 10 bins) | **0.0173** | **0.0107** |
| Top-1 accuracy | 3.61% | 3.61% (invariant under monotonic rescaling) |

Spec threshold was ECE < 0.10 — we hit **ECE = 0.017** without any post-hoc correction, 5.8x inside the spec ceiling. Optimal temperature T ≈ 1.07 indicates very mild overconfidence, not the dangerous overconfidence the calibration audit was designed to catch.

**Interpretation.** The reliability diagram (`pitchgpt_reliability.html`) shows the predicted distribution is extremely flat: 80,276 of 85,083 test tokens (94%) land in the lowest-confidence bin [0.0, 0.1), with mean top-1 confidence of just 0.048. The model rarely commits to any single pitch — it spreads probability thinly over the 2210-token vocabulary. In these diffuse-prediction regions empirical accuracy (3.4%) tracks mean confidence (4.8%) to within 1.4 percentage points, which is what drives the low ECE. Reliability drifts off the diagonal only in sparsely-populated 0.2-0.5 bins (71-326 samples each) where the model over-commits to a specific pitch and misses. Temperature scaling (T=1.072) broadens those spikes slightly and drops ECE another 0.007.

## Ablation

Four variants, identical hyperparameters and splits, same 1000 training games. Each variant's test perplexity was measured on the 2535-sequence 2024 test split.

| Variant | Params | Test perplexity | Drop vs full (% higher) | Wall clock |
|---|---|---|---|---|
| full | 1,398,562 | **145.459** | — | 668.3 s |
| tokens_only | 1,398,562 | 146.745 | +0.88% | 526.8 s |
| count_only | 1,398,562 | 148.349 | +1.95% | 596.0 s |
| identity_only (degenerate) | 1,204,656 | 85.904 | **not directly comparable** | 557.4 s |

**Note on `tokens_only` parameter count.** The context-projection layer is still present; its input is forced to zero at forward time, so the weights exist but contribute only a constant bias. This keeps the parameter count identical across full / tokens_only / count_only so the comparison isolates *information content*, not capacity.

**Note on `identity_only`.** This variant swaps both the input token vocabulary AND the output prediction target to pitcher-ids (1455 known pitchers + 1 UNK). Because a sequence is a single pitcher's game, the prediction target is trivially "the same pitcher-id as the input" — a degenerate task. Its test perplexity of 85.9 reflects the smaller, nearly-deterministic target space, not superior sequence modelling. The identity variant answers a *different* question ("is the pitcher-id predictable from itself?") than the other three and its cross-entropy is not on the same axis. It is reported here because spec ticket #6 required it and because the reviewer question "is the model just memorising pitcher identity?" needs an answer — but it does NOT enter the pass/fail calculation.

### Headline verdict

**FAIL — full model beats the best non-degenerate ablation (tokens_only) by only +0.88%, far below the spec's ≥10% threshold.**

Ignoring identity_only (different task), the gap between the full contextual model and a pitch-tokens-only variant is <1%. Even the count_only variant — which does see the most obviously-useful situational features — is within 2% of full. At the current 1000-game / 5-epoch training scale, **the 34-dim context vector is adding essentially zero signal**.

## Surprising finding

The biggest surprise was **how little context matters at this training scale**. The `tokens_only` model — which sees nothing about count, runners, inning, score, or batter handedness — lands at test perplexity 146.7 vs the full model's 145.5, a gap of under one percent. For a "contextual" sequence model headlining *situational awareness* as a differentiator, this is a weak signal that the context branch is currently decorative.

Two non-mutually-exclusive explanations:

1. **Insufficient data.** With 1000 games (~300K pitches), the model may simply not see enough count-conditional examples to distinguish signal from noise. The context projection is a single 34→128 linear layer; at full 10,000+ game scale with more epochs, it may start to pay off.
2. **Context is mostly redundant with pitch-type sequence.** A pitcher's prior-pitch sequence already encodes count implicitly (e.g. three balls in a row strongly implies a 3-ball count is live), so explicitly surfacing the count may double-count information the model was already extracting.

The calibration finding reinforces point 2: the model produces a very flat predictive distribution because *no* feature — context or token — reliably disambiguates the next pitch from a pool of 2210 composite tokens.

## Wall-clock breakdown

| Phase | Wall clock |
|---|---|
| Dataset loading (3 splits, single pass) | 123 s |
| PitchGPT `full` training (5 epochs, saved checkpoint) | 668 s |
| PitchGPT `tokens_only` training | 527 s |
| PitchGPT `count_only` training | 596 s |
| PitchGPT `identity_only` training | 557 s |
| Calibration (val + test inference, LBFGS temperature fit) | 80 s |
| **Total** | **~43 min** |

Training ran significantly slower than the reference ~90 s / epoch because two other agents were sharing CPU (fWAR backfill + LSTM baseline). On an idle machine this job would complete in ~28 min. No identity-only skip was needed.

## Recommendations for the scale-up rerun

1. **Skip temperature scaling.** ECE = 0.017 without correction is already well under the 0.10 spec. Adding T=1.07 buys another 0.007 ECE — noise level. Do not add calibration logic to the production path.
2. **Skip label smoothing.** The model is **under-confident** (diffuse predictions across the 2210-token vocabulary), not overconfident. Label smoothing would flatten the distribution further and likely hurt perplexity.
3. **Before deciding to drop context, re-ablate at full scale.** The 1000-game ablation shows tokens_only ≈ full, but this may reverse with 10× more data. Re-run this script with `--max-train-games 10000 --epochs 20`. If the gap is still <5% at scale, **remove the context-projection layer** to save inference latency; if the gap grows past 10%, keep it and consider expanding the context feature set (finer buckets, handedness matchup, pitcher fatigue).
4. **Consider architectural changes to reduce output diffusion.** The model over-spreads probability, which is why ECE is so low and perplexity is so high. Options: (a) deeper stacks (d_model 192, 6 layers) to sharpen predictions; (b) output-space factorisation (separate heads for pitch_type / zone / velocity) to collapse the 2210-token vocabulary into three smaller prediction tasks where the model can be more confident. (b) is especially promising — current vocabulary is an engineered product, not a natural label space.
5. **Do not lean on the identity-only result in the award submission.** It answers the reviewer question "is the model just learning pitcher identity?" in the right direction (its perplexity isn't directly comparable, so it can't be used to prove or disprove pitcher-memorisation), but framing it as part of the pass/fail ablation is misleading.

## Next ticket

Spec ticket #9 (reproducibility — `scripts/train_pitchgpt_deterministic.py`, 5 runs with <0.1% test-perplexity variance) is ready to execute. All required infrastructure (seeded training loop, leakage-audited split, deterministic sampling) already exists in `train_pitchgpt_baselines.py` and in this ablation script.

## Files modified / added

**Added:**
- `src/analytics/pitchgpt_calibration.py` — reliability curve, ECE, temperature scaling utilities.
- `scripts/pitchgpt_calibration_analysis.py` — end-to-end calibration runner for a checkpoint.
- `scripts/pitchgpt_ablation.py` — 4-variant ablation trainer, with context-masking helpers and an identity-only dataset wrapper.
- `tests/test_pitchgpt_calibration.py` — 11 unit tests (ECE on perfectly calibrated / maximally miscalibrated distributions, temperature scaling behaviour on known-overconfident and well-calibrated synthetic models).
- `tests/test_pitchgpt_ablation.py` — 10 unit tests (context masking modes, identity-dataset vocabulary swap, identity model output dimensions).
- `results/run_5epoch/pitchgpt_full.pt` — saved full-variant checkpoint for downstream use.
- `results/run_5epoch/pitchgpt_calibration.json`, `pitchgpt_reliability.html`, `pitchgpt_ablation_metrics.json`.
- `docs/models/pitchgpt_calibration_ablation_results.md` — this document.

**No changes** to `src/analytics/pitchgpt.py` — the calibration and ablation modules import and reuse `PitchGPT`, `PitchSequenceDataset`, `PitchTokenizer` as-is per the task constraint.
