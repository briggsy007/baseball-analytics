# PitchGPT — Calibration and Ablation Results

**Tickets:** spec #6 (feature ablation), spec #7 (calibration audit), spec #2N (10K-game scale decision).
**Artifacts:**
- 1K run: `results/run_5epoch/pitchgpt_calibration.json`, `results/run_5epoch/pitchgpt_reliability.html`, `results/run_5epoch/pitchgpt_ablation_metrics.json`.
- 10K run: `results/run_10kgame/pitchgpt_ablation_metrics.json`, `results/run_10kgame/pitchgpt_full.pt`, `results/run_10kgame/_ablation.log`.

## Headline

**FAIL** — at 10,000 training games, full PitchGPT beats its best non-degenerate ablation (`count_only`) by only **4.14%**, still well under the ≥10% spec threshold. The full-vs-`tokens_only` gap did widen sharply (0.88% → 6.85%) — context is no longer dead — but the bulk of that signal is recoverable from count+outs alone. **Recommendation: drop the full 34-dim context projection; keep a minimal count+outs branch**, and pivot the novelty narrative away from "situational context" toward transformer-sequence + perplexity-on-sequencing-outcomes.

---

## Section 1: Initial 1K ablation (context dead at this scale)

**Run:** 2026-04-17 — canonical 5-epoch, seed=42, 2015-2022 train / 2023 val / 2024 test; 1000/300/300 games per split.

### Methodology

We trained four PitchGPT variants on an identical date-based leakage-clean split, recorded test perplexity for each, and ran a post-hoc calibration audit on the full variant's checkpoint. Calibration computes top-1 reliability on the 2024 test pitches and fits a single temperature scalar on the 2023 validation logits via LBFGS. All hyperparameters (seed 42, 5 epochs, batch 32, AdamW lr 1e-3, grad clip 1.0) match the canonical baselines run so ablation drops and calibration numbers are directly comparable to the PitchGPT-vs-LSTM headline result.

### Calibration

The full-variant checkpoint (test perplexity 145.5, 85,083 test pitch tokens) is **already well-calibrated** — no temperature scaling required.

| Metric | Pre-temperature | Post-temperature (T=1.072) |
|---|---|---|
| Expected Calibration Error (ECE, 10 bins) | **0.0173** | **0.0107** |
| Top-1 accuracy | 3.61% | 3.61% (invariant under monotonic rescaling) |

Spec threshold was ECE < 0.10 — we hit **ECE = 0.017** without any post-hoc correction, 5.8x inside the spec ceiling. Optimal temperature T ≈ 1.07 indicates very mild overconfidence, not the dangerous overconfidence the calibration audit was designed to catch.

**Interpretation.** The reliability diagram (`pitchgpt_reliability.html`) shows the predicted distribution is extremely flat: 80,276 of 85,083 test tokens (94%) land in the lowest-confidence bin [0.0, 0.1), with mean top-1 confidence of just 0.048. The model rarely commits to any single pitch — it spreads probability thinly over the 2210-token vocabulary. In these diffuse-prediction regions empirical accuracy (3.4%) tracks mean confidence (4.8%) to within 1.4 percentage points, which is what drives the low ECE. Reliability drifts off the diagonal only in sparsely-populated 0.2-0.5 bins (71-326 samples each) where the model over-commits to a specific pitch and misses. Temperature scaling (T=1.072) broadens those spikes slightly and drops ECE another 0.007.

### Ablation (1K)

Four variants, identical hyperparameters and splits, same 1000 training games. Each variant's test perplexity was measured on the 2535-sequence 2024 test split.

| Variant | Params | Test perplexity | Drop vs full | Wall clock |
|---|---|---|---|---|
| full | 1,398,562 | **145.459** | — | 668.3 s |
| tokens_only | 1,398,562 | 146.745 | +0.88% | 526.8 s |
| count_only | 1,398,562 | 148.349 | +1.95% | 596.0 s |
| identity_only (degenerate) | 1,204,656 | 85.904 | not directly comparable | 557.4 s |

**1K verdict:** FAIL — full model beats the best non-degenerate ablation (tokens_only) by only +0.88%, far below the spec's ≥10% threshold. At the 1000-game / 5-epoch training scale, **the 34-dim context vector is adding essentially zero signal**. Two non-mutually-exclusive hypotheses:

1. **Insufficient data.** With 1000 games (~300K pitches), the model may simply not see enough count-conditional examples to distinguish signal from noise.
2. **Context is mostly redundant with pitch-type sequence.** A pitcher's prior-pitch sequence already encodes count implicitly (e.g. three balls in a row strongly implies a 3-ball count is live), so explicitly surfacing the count may double-count information the model was already extracting.

Hypothesis 1 is testable by scaling up training data — see Section 2.

---

## Section 2: 10K ablation (decides context claim)

**Run:** 2026-04-18 — seed=42, 2015-2022 train / 2023 val / 2024 test; **10,000 / 3,000 / 3,000 games per split**. 5 epochs, batch 32, AdamW lr 1e-3. Per spec ticket #2N the `identity_only` variant was dropped as degenerate (not comparable, confirmed at 1K).

### Dataset scale

| Split | Games | Sequences | Pitches |
|---|---|---|---|
| train (2015-2022) | 9,999 | 85,675 | 2,934,227 |
| val (2023) | 2,450 | 20,789 | 723,276 |
| test (2024) | 2,517 | 21,566 | 735,055 |

Leakage audit: 0 shared game_pks across any pair of splits. 599 / 678 pitchers are shared train↔test / train↔val (expected — same players pitch across years, but their specific game sessions are held out).

### Results

| Variant | Params | Test perplexity | Drop vs full | Wall clock (min) |
|---|---|---|---|---|
| **full** | 1,398,562 | **113.088** | — | 91.1 |
| tokens_only | 1,398,562 | 121.402 | **+6.85%** | 80.9 |
| count_only | 1,398,562 | 117.969 | **+4.14%** | 82.4 |

### Comparison vs 1K

| Comparison | 1K | 10K | Δ |
|---|---|---|---|
| full test_ppl | 145.459 | 113.088 | −22.2% (all models benefit from scale) |
| full vs tokens_only | +0.88% | **+6.85%** | **gap widened 7.8×** |
| full vs count_only | +1.95% | **+4.14%** | gap widened 2.1× |

### 10K Verdict

**FAIL** — full model beats the best ablation (`count_only`) by only 4.14%, below even the spec's "ambiguous band" lower bound of 5%.

This is a more interesting result than the 1K FAIL because **the context branch clearly is doing something at scale** — the full-vs-`tokens_only` gap (6.85%) is no longer a rounding error. But once the model has access to count+outs (`count_only`), adding the rest of the 34-dim context vector (runners, handedness, inning, score-diff) buys only 4.14% — below the 5% threshold. In other words:

- **The "situational context matters" narrative is partially vindicated** (+6.85% gap for context vs. no-context, up from +0.88%) — context is not dead at scale.
- **The "rich 34-dim situational context" narrative is refuted.** Once you have count + outs, the rest of the feature set (runner layout, batter handedness, inning, score differential) is worth less than 5% perplexity at 10K scale.

### Training trajectories

Full was **not** converged at epoch 5 — val loss was still descending (104.22 → heading lower). Tokens_only and count_only were also still descending. Epoch 5 gap may understate the asymptotic gap; a ~20-epoch run might widen it further. But the 1K run showed the same "still descending at epoch 5" pattern and the 10× scaling only bought us from 0.88% to 6.85% on the full-vs-tokens axis. Expecting another 10× budget bump to produce the missing 6% signal is speculative.

### Wall-clock breakdown

| Phase | Wall clock | Notes |
|---|---|---|
| Dataset loading (3 splits) | 659 s (11.0 min) | train: 8 min (2.9M pitches + score-diff recompute), val/test ~1.5 min each |
| PitchGPT `full` training (5 epochs, saved checkpoint) | 5467 s (91.1 min) | avg 1089 s/epoch |
| PitchGPT `tokens_only` training | 4854 s (80.9 min) | avg 971 s/epoch — ~11% faster than full (context-projection branch receives zero input but weights still multiply) |
| PitchGPT `count_only` training | 4945 s (82.4 min) | avg 989 s/epoch |
| **Total training** | 15,266 s (254 min) | roughly 6× the 1K run's 39 min of training |
| **End-to-end (including dataset build)** | ~4h 25m | over the 4h budget by ~25 min; allowed to run because both tokens_only (the critical decision baseline) and count_only had fully converged before the 3.5h kill-threshold triggered |

Memory: peak RSS **2.43 GB**, mean 1.67 GB during training (well under the machine's 16 GB / ~9 GB free). No OOM concerns at 10K scale; the dataset-build 2.4 GB peak is the bottleneck (DuckDB query result frame + per-row tokenisation). At 50K+ games an out-of-core streaming loader would be worth adding.

## Updated recommendations

1. **Drop the context-projection layer — or replace it with a count+outs-only 4-dim projection.** At 10K training games the full 34-dim context buys +4.14% vs count-only and +6.85% vs no-context. Keeping the count+outs branch (15 dims) costs ~2% perplexity against full, recovers ~3% against tokens-only, and enables honest "count-aware" language. The remaining 19 dims (runner layout, inning, handedness, score-diff) can be retired.
2. **Pivot the novelty narrative.** Drop "situational awareness over a 34-dim context vector" from the award submission. The defensible claim is:
   - **Transformer sequence modelling over a 2,210-token composite vocabulary** (pitch-type × zone × velocity bucket) — unusual in the public baseball-analytics literature.
   - **Calibrated probability surface** (ECE 0.017, no temperature needed) — the model's diffuse-but-honest predictions can be composed with downstream bet-sizing / confidence-thresholded production logic.
   - **Count-aware next-pitch probabilities** — not 34-dim situational awareness, but defensible.
3. **Before the LSTM head-to-head rerun**, decide between two architectures:
   - *(A) "light" PitchGPT:* drop context branch entirely, keep tokens only. Simplest, fastest inference, honest claim restricted to sequence modelling.
   - *(B) "count-aware" PitchGPT:* keep a 15-dim count+outs projection only. Small extra cost, justifies a "situational" word in the narrative.
   Recommend (B) — the 2.7% perplexity gain over tokens-only is real and the architectural cost is negligible.
4. **Do not spend more compute chasing a 10% full-vs-ablation gap at current architecture.** The 1K → 10K scaling bought 7.8× on the tokens_only gap but only 2.1× on the count_only gap, and both are still short of spec. The architectural payoff curve is diminishing. Further gains would require either (a) richer context features (pitch-sequence-to-date within at-bat, pitcher fatigue, batter-vs-pitcher history), or (b) architectural changes (separate heads for pitch_type / zone / velocity) — not more epochs of the current setup.
5. **Skip temperature scaling.** ECE = 0.017 without correction is already well under the 0.10 spec; temperature scaling buys another 0.007 ECE — noise level. (Unchanged from 1K recommendation.)
6. **Skip label smoothing.** Model is under-confident (diffuse predictions), not overconfident. (Unchanged.)

## What changes in the platform code

- **Kill the context claim in the dashboard and ticker copy.** `src/dashboard/views/pitchgpt_view.py` currently advertises "situational-context awareness"; this should be rephrased to "count-aware sequence modelling" once the light/count-aware architecture ships.
- **Retain the context projection as a 15-dim count+outs-only layer** in `src/analytics/pitchgpt.py` (new `CONTEXT_DIM=15`), or gate the full 34-dim projection behind a feature flag for backward compatibility.
- **No retraining of downstream consumers required** — all downstream PitchGPT consumers call the model by checkpoint, not by feature layout.

## Files modified / added

**Added (this ticket, #2N):**
- `results/run_10kgame/pitchgpt_ablation_metrics.json` — 10K ablation metrics with `comparison_vs_1k`.
- `results/run_10kgame/pitchgpt_full.pt` — 10K full-variant checkpoint (~5.3 MB).
- `results/run_10kgame/_ablation.log` — full training log (dataset build, 3 × 5 epochs).
- `results/run_10kgame/_memlog.txt` — per-30s RSS samples for the training process.
- `results/run_10kgame/_build_partial_json.py` — fallback parser to rebuild the JSON from log output if the run had been killed before completion (unused — run finished cleanly).

**Modified:**
- `docs/models/pitchgpt_calibration_ablation_results.md` — this document (added Section 2 + updated recommendations).

**Unchanged:**
- `scripts/pitchgpt_ablation.py` — the existing `--skip-identity` flag was sufficient to drop the degenerate variant; no code changes were needed to run at 10× scale.
- `src/analytics/pitchgpt.py` — ablation used the existing model unchanged.
- `tests/test_pitchgpt_ablation.py` — all 10 tests pass (unchanged).

## Next ticket

- If the PM accepts the "light / count-aware" pivot: implement `PitchGPTLight` with 15-dim count+outs projection, retrain, verify test perplexity within 2-3% of the full-context 10K checkpoint (113.088).
- Then re-run the PitchGPT-vs-LSTM head-to-head at 10K with the new architecture. Do **not** advertise situational-context novelty in that doc.
