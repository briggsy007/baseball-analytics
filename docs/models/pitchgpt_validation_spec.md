# PitchGPT Validation Spec

**Source:** `src/analytics/pitchgpt.py`. Network: `class PitchGPTModel` (line 432). Lifecycle wrapper: `class PitchGPT` (line 528). Tokenizer: `class PitchTokenizer` (line 124). Dataset: `class PitchSequenceDataset` (line 266).

## A. Current state

**Architecture.** Decoder-only transformer, 4 layers, 4 heads, 128 hidden dim.

- **Context embedding.** Six categorical features one-hot encoded (34-dim total): count (12), outs (3), runner state (8), batter hand (2), inning bucket (4), score diff (5).
- **Token embedding.** Composite pitch tokens with vocabulary size 2210, built from (pitch_type × 17) × (zone × 26) × (velocity_bucket × 5). Deterministic invertible encoding: `pt_idx * 130 + zone * 5 + velo`.
- **Positional embedding.** Dense, 128-dim, `max_seq_len = 256`.
- **Attention.** Causal self-attention. `PAD_TOKEN = 2210`.

**Tokenization.** Pitch type: 16 MLB types plus unknown. Zone: 5x5 plate grid plus OOB. Velocity: 5 buckets.

**Training.** Cross-entropy loss with PAD ignored. AdamW, `lr = 1e-3`. Gradient clip 1.0. CUDA if available. 5 epochs default, batch 32, max 3000 games per run. Checkpoints saved to `models/pitchgpt_v{version}.pt`. No LR schedule.

**Tested.** Tokenizer round-trip, token range, context shape and one-hot encoding, score-diff bucket map, forward-pass shape, single-token pass, causal mask enforcement (the key correctness test), gradient flow, training returns metrics, inference smoke test, edge cases.

**Not tested.**
- Date-based train/val/test split. Currently groups by `game_pk` / `pitcher_id` only — no temporal enforcement.
- LSTM / Markov / heuristic baselines.
- Feature ablation.
- Calibration (reliability diagram, ECE).
- Disruption index → outcome regression.
- Reproducibility seeding.
- Hyperparameter sensitivity.
- CPU vs CUDA consistency.

**Critical gap.** `score_diff` is hard-coded to 0 at line 365 (also at line 840). The "contextual" claim in model marketing is currently false until fixed.

## B. Validation tickets

Tiered: Tier 1-2 blocking, Tier 3 validation, Tier 4 productization, Tier 5 bug fixes.

### Tier 1 — Data integrity (blocking)

#### Ticket 1. Date-based train/val/test split (M)
- **Goal.** Enforce temporal honesty: 2015-2022 train, 2023 val, 2024 test. No game crosses splits; no pitcher appears in both train and test.
- **Artifacts.** Modify `PitchSequenceDataset._load()` (line 266) to accept a `split_mode` parameter. Add unit test `test_date_based_split_no_leakage`.
- **Success.** No shared `game_pk` across splits; no pitcher in both train and test; test set consists exclusively of 2024 regular season.
- **Effort.** M.

#### Ticket 2. Fix score_diff broken context (S)
- **Goal.** Populate the `score_diff` context feature from actual game state.
- **Artifacts.** Join to `games` table; compute `home_score - away_score` at pitch time. Replace hard-coded `score_diff=0` at line 365 and line 840.
- **Success.** Distribution of `score_diff` spans all 5 buckets on real data; unit test asserts non-zero bucket-1 and bucket-5 frequency.
- **Effort.** S.

### Tier 2 — Baselines (novelty validation)

#### Ticket 3. LSTM baseline (M)
- **Goal.** Beat the obvious sequence-model alternative.
- **Artifacts.** `src/analytics/pitch_lstm.py`. 2-layer LSTM, 128 hidden. Same tokenizer and splits as PitchGPT.
- **Success.** Transformer perplexity < LSTM perplexity by ≥ 15% on the 2024 test set.
- **Effort.** M.

#### Ticket 4. Markov 1st/2nd order baseline (S)
- **Goal.** Rule out a trivial counting model.
- **Artifacts.** `src/analytics/pitch_markov.py`.
- **Success.** Transformer perplexity < Markov-2 perplexity by ≥ 20%.
- **Effort.** S.

#### Ticket 5. Heuristic baseline (S)
- **Goal.** Rule out a naive prior.
- **Artifacts.** Fixed distribution: fastball 50, breaking 30, offspeed 20.
- **Success.** Transformer perplexity < heuristic perplexity by ≥ 25%.
- **Effort.** S.

### Tier 3 — Model validation

#### Ticket 6. Feature ablation (M)
- **Goal.** Show every input feature pulls its weight.
- **Artifacts.** Training variants: full / tokens-only / count-only / pitcher-identity-only.
- **Success.** Full model beats any ablation by ≥ 10% on test perplexity.
- **Effort.** M.

#### Ticket 7. Calibration audit (M)
- **Goal.** Next-pitch probabilities must be well-calibrated, not just low-loss.
- **Artifacts.** `src/analytics/pitchgpt_calibration.py` producing reliability diagram and ECE.
- **Success.** Expected Calibration Error < 0.10.
- **Effort.** M.

#### Ticket 8. Disruption index → outcome regression (M)
- **Goal.** Validate that high per-pitch NLL ("disruption") is orthogonal to contact quality, so it indexes *expectation violation* rather than simply bad pitches.
- **Artifacts.** `src/analytics/pitchgpt_disruption_validation.py`. Regress per-pitch NLL on `woba_value` and `launch_speed`.
- **Success.** R² < 0.05 (disruption orthogonal to contact quality). Note: this hypothesis framing is subtle — reviewers may push back that disruption *should* correlate with outcomes. Be ready to justify the orthogonality framing.
- **Effort.** M.

#### Ticket 9. Reproducibility script (M)
- **Goal.** Bit-for-bit reproducibility across runs.
- **Artifacts.** `scripts/train_pitchgpt_deterministic.py`. Seeds numpy, torch, torch.cuda.
- **Success.** 5 runs produce < 0.1% test-perplexity variance.
- **Effort.** M.

#### Ticket 10. Hyperparameter grid (L)
- **Goal.** Justify or replace defaults.
- **Artifacts.** Grid over `d_model × nhead × num_layers`.
- **Success.** Best config identified OR defaults confirmed near-optimal.
- **Effort.** L.

### Tier 4 — Productization

#### Ticket 11. Checkpoint metadata JSON (S)
- **Goal.** Every checkpoint must be self-describing.
- **Artifacts.** `models/pitchgpt_v{v}_metadata.json` alongside each `.pt` file, containing train-date ranges, final train/val/test losses, test perplexity, architecture, parameter count, baseline numbers.
- **Success.** A new Claude session can identify which checkpoint beat which baseline without reading source.
- **Effort.** S.

#### Ticket 12. Results report (M)
- **Goal.** Publishable write-up.
- **Artifacts.** `docs/models/pitchgpt_results.md`. Sections: architecture, leakage audit, results table vs baselines, ablation table, calibration plot, disruption validation, reproducibility note, limitations.
- **Success.** Self-contained, ~1500-2000 words.
- **Effort.** M.

### Tier 5 — Bug fixes

#### Ticket 13. score_diff hard-coded
- Covered above (Ticket 2).

#### Ticket 14. Device parity test (S)
- **Goal.** Identical outputs on CPU and CUDA.
- **Artifacts.** `test_model_device_parity`.
- **Success.** Max abs diff between CPU and CUDA forward passes < 1e-5.
- **Effort.** S.

#### Ticket 15. Warm-start LR scheduler (S)
- **Goal.** Reduce validation loss via standard transformer-training hygiene.
- **Artifacts.** CosineAnnealing or linear warmup scheduler added to training loop.
- **Success.** Expected ~5% val-loss improvement vs fixed LR.
- **Effort.** S.

## C. Award-readiness checklist

- First transformer sequence model on Statcast data with integrated situational context.
- Causal-masked next-pitch prediction enables real-time disruption scoring.
- Validated disruption index correlating appropriately with outcomes.
- Beats LSTM, Markov-2, and heuristic baselines on test perplexity.

**Disqualifiers.** Sequence leakage. Weak baselines. Incomplete context (`score_diff` still 0). No reproducibility. No calibration. Undocumented distribution shift on test.

**Likely reviewer questions.**
- "Why not fine-tune GPT-2?"
- "How do you know it's not just learning pitcher identity?" → answered by the identity-only ablation (Ticket 6).
- "Does high disruption correlate with better outcomes?" → answered by disruption validation (Ticket 8).
- "How do you rule out train/test contamination?" → answered by the leakage audit unit test (Ticket 1).

## D. Risk flags

- **Priority 1 — data leakage.** No explicit date split. Same pitcher appears in train and val. `score_diff = 0`.
- **Priority 2 — training stability.** No LR schedule, no early stopping, gradient clip may be too tight, no CPU/CUDA consistency test.
- **Priority 3 — capacity vs overfit.** 3.2M params over ~750K-900K pitch tokens gives ~250 tokens per parameter (recommend ≥ 100; current ratio is acceptable but tight).
- **Priority 4 — inference latency.** No timing benchmark for real-time in-game use.
- **Priority 5 — artifacts.** Ad-hoc versioning, no metadata JSON.

---

Status: Specified 2026-04-16. Phase 2 pending. First ticket: #2 (score_diff fix — cheapest unblocker).
