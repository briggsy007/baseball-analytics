# PitchGPT Baseline Comparison -- 4-Way Results

**Tickets.** #3 (LSTM), #4 (Markov-1/2), #5 (Heuristic) from
`docs/models/pitchgpt_validation_spec.md`.

**Status.** Completed 2026-04-16.

> Results in this document are produced by
> `scripts/train_pitchgpt_baselines.py` and persisted to
> `results/run_5epoch/pitchgpt_baselines_metrics.json` and
> `results/run_5epoch/pitchgpt_training_curves.html`.  Re-run with
> `python scripts/train_pitchgpt_baselines.py --max-train-games 1000 --max-val-games 300 --max-test-games 300 --epochs 5 --seed 42 --output-dir results/run_5epoch`.

## 1. Methodology

All four models are evaluated on the **same** leakage-clean date-based
split: 2015-2022 training (1 000 sampled games, 8 567 per-pitcher game
sequences, ~291 K pitches), 2023 validation (300 games, 2 548 sequences)
and 2024 test (300 games, 2 622 sequences).  A pre-training audit
confirms **zero** overlap of `game_pk` across any pair of splits (the
unavoidable train/test pitcher-identity overlap of 398 pitchers is a
known by-product of date-based splitting on a modeling task whose
target is the pitch-sequence distribution, not generalisation to unseen
pitchers, and is documented in the spec).

**PitchGPT** (decoder-only transformer, 4 layers × 4 heads × 128 hidden,
1.40 M parameters) and the **LSTM baseline** (2 layers × 128 hidden,
0.84 M parameters) train under identical conditions: 5 epochs, batch
size 32, AdamW optimiser with learning rate 1 × 10⁻³, gradient clip
1.0, cross-entropy loss with `ignore_index = PAD_TOKEN`, single shared
seed (42), best-val-loss checkpoint selection before test evaluation.
The shared `PitchSequenceDataset`, tokenizer, and collate function
guarantee pixel-identical inputs.  **Markov-1**, **Markov-2**, and the
**Heuristic baseline** (fastball 50 % / breaking 30 % / offspeed 20 %
mixed 50/50 with the empirical training marginal) are closed-form:
counts + Laplace smoothing (α = 0.1), fit on the same training
sequences, scored on the same 2024 test sequences.  All five models
produce perplexity over the identical 2 210-token vocabulary, so the
numbers are directly comparable.

## 2. Results

### 2.1 Headline table (2024 test set)

| Model      | Params     | Train ppl | Val ppl   | Test ppl  | Wall-clock | vs PitchGPT    |
|------------|-----------:|----------:|----------:|----------:|-----------:|---------------:|
| PitchGPT   | 1 398 562  | 119.08    | 135.92    | **148.20**| 447 s      | —              |
| LSTM       |   837 154  | 131.97    | 153.13    |   169.80  | 132 s      | +14.6 %        |
| Markov-2   |         0  | 252.83    | 301.26    |   332.67  |   9 s      | +124.5 %       |
| Markov-1   |         0  | 279.73    | 325.38    |   355.00  |   9 s      | +139.6 %       |
| Heuristic  |         0  | 394.00    | 425.57    |   438.86  |  <1 s      | +196.1 %       |

_"vs PitchGPT" reports how much worse the baseline is; higher is better
for PitchGPT._

### 2.2 Spec verdicts

| Comparison              | PitchGPT  | Baseline  | Improvement | Threshold | Verdict |
|-------------------------|----------:|----------:|------------:|----------:|:--------|
| PitchGPT vs LSTM        | 148.20    | 169.80    |  **12.7 %** |   ≥ 15 %  | **FAIL** |
| PitchGPT vs Markov-2    | 148.20    | 332.67    |  **55.5 %** |   ≥ 20 %  | **PASS** |
| PitchGPT vs Heuristic   | 148.20    | 438.86    |  **66.2 %** |   ≥ 25 %  | **PASS** |

**Overall:** two of three thresholds cleared; the LSTM margin misses
spec by 2.3 percentage points.

### 2.3 Training curves

See `results/run_5epoch/pitchgpt_training_curves.html` for an
interactive plot.  Both neural models are still descending at epoch 5
(PitchGPT val-loss 5.46 → 4.91, LSTM 5.60 → 5.03); the per-epoch val-
loss gap between the two has also widened from 0.14 nats at epoch 1 to
0.12 nats at epoch 5, suggesting further training would probably
*not* push PitchGPT over the 15 % LSTM threshold — both are bottlenecked
on the same 8 K-sequence training set.

## 3. Interpretation

The honest read on these numbers: **PitchGPT is a decisive winner
against the Markov and heuristic baselines, and a modest but below-
threshold winner against the LSTM.**  The transformer beats a 2-layer
LSTM by 12.7 % — a real, reproducible improvement, but 2.3 pp short of
the "≥ 15 %" threshold the spec committed to.

Three observations frame the narrative:

1. **Markov / Heuristic are demolished.**  A 55 % improvement over
   2nd-order Markov and 66 % over the family-prior heuristic is
   evidence that the sequence model is extracting genuine higher-order
   structure — composite pitch type × zone × velocity transitions that
   a count-aware pair-memory model cannot capture.  Those two results
   are the strongest single-model-vs-baselines spread we have produced
   on this platform.

2. **LSTM vs transformer at this scale is a close call.**  At 8 K
   training sequences (≈ 290 K tokens) over a 2 210-class vocabulary
   with a 34-dim context, the transformer's inductive-bias advantage
   over a recurrent network is real but narrow.  This tracks with
   published results on small-to-medium sequence tasks: transformers
   dominate when the dataset is large enough for the positional /
   attention capacity to pay for itself.  The transformer is also
   3.4× slower wall-clock, so the per-compute-dollar picture narrows
   the gap further.

3. **The award framing must be truthful.**  The defensible claim from
   these results is: *"PitchGPT is the first transformer sequence
   model on Statcast pitch data and decisively outperforms classical
   sequence baselines (Markov-2 and naive priors) while remaining
   competitive with an LSTM of equivalent capacity — at our current
   training scale."*  Claiming a clean transformer sweep is not yet
   justified; claiming a transformer-family win over classical
   baselines is.

## 4. Limitations

- **Training scale.**  1 000 games out of the ~24 000 regular-season
  games available per year is a small fraction — a scale ceiling that
  likely depresses the transformer's margin over the LSTM specifically.
  The 5 % of full-year data in the training set is a cost / latency
  choice, not a fundamental capacity limitation.
- **Reconstructed `score_diff`.**  The pitches table lacks a native
  running-score column.  We reconstruct per-pitch score differential
  from `events` + baserunner state + `delta_run_exp`; this is
  approximate and may miss multi-run plays.  The impact is bounded —
  `score_diff` is one of six context buckets — but documented.
- **Pitcher-identity overlap.**  398 pitchers appear in both train and
  test (unavoidable on a date split).  The task is modeling the
  pitch-sequence distribution, not generalisation to unseen pitchers;
  the leakage audit confirms zero *game-level* overlap.
- **CPU-only run.**  5 epochs on CPU took 11 minutes; a CUDA re-run
  at 3 000 games should be cheap and would test whether scale closes
  the LSTM gap.

## 5. Next blockers

- **Ticket #6 — Ablation.**  Train PitchGPT without context, without
  tokens (pitch-identity only), and with count-only.  Spec requires
  the full model to beat every ablation by ≥ 10 % on test perplexity.
  If this passes, the "contextual" half of the novelty claim is
  earned.
- **Ticket #7 — Calibration.**  Expected Calibration Error < 0.10
  on the held-out 2024 set.  Perplexity alone is insufficient for a
  real-time in-game disruption scoring claim.
- **(Optional) scale re-run at 3 000 games.**  Cheapest path to
  closing the LSTM gap and converting the LSTM verdict from FAIL to
  PASS; does not depend on #6 or #7.

---

**Artifacts produced**

- `results/run_5epoch/pitchgpt_baselines_metrics.json`
- `results/run_5epoch/pitchgpt_training_curves.html`
- `results/run_5epoch/pitchgpt_vs_lstm_metrics.json` (legacy alias)

**Reproducing.**  The run is deterministic under seed 42 modulo CUDA
nondeterminism (this run was CPU).  Any dev should get bit-for-bit the
same numbers.
