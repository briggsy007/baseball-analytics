# PitchGPT 2025 Holdout Report

Generated: 2026-04-24T03:53:27Z

Checkpoint: `models\pitchgpt_v2.pt`

Train: 2015-2022 (85621 sequences)

Val:   2023 (1504 sequences, used only for temperature scaling)

Holdout: 2025 (5915 sequences, pitcher-disjoint from train)


## Leakage audit

- shared game_pks across splits: **0**
- shared pitchers train/holdout: **0**
- train pitchers: 2094  val: 161  holdout: 469

## Holdout perplexity (lower is better)

| Model | Params | Holdout PPL | 95% CI | N pitches |
|---|---|---|---|---|
| pitchgpt | 1,398,690 | 118.645 | 117.903 – 119.43 | 202923 |
| lstm | 837,282 | 122.483 | 121.608 – 123.294 | 202923 |
| markov2 | 0 | 344.347 | 341.929 – 346.705 | 202923 |
| heuristic | 0 | 469.899 | 467.29 – 472.55 | 202923 |

## Gates

| Comparison | Spec | Point | 95% CI | Point verdict | CI-lower verdict |
|---|---|---|---|---|---|
| PitchGPT vs LSTM | ≥15% | +3.13% | +2.19 / +4.05 | FAIL | FAIL |
| PitchGPT vs Markov-2 | ≥20% | +65.54% | +65.23 / +65.87 | PASS | PASS |
| PitchGPT vs Heuristic | ≥25% | +74.75% | +74.55 / +74.96 | PASS | PASS |

**Overall (point)**: FAIL
**Overall (CI lower bound)**: FAIL

## Calibration on 2025 (out-of-sample)

- ECE pre-temperature: **0.0124** (gate <0.10: PASS)
- ECE post-temperature (T=1.055): **0.0075** (gate <0.10: PASS)
- Top-1 accuracy: 0.0423
- Non-PAD pitches scored: 202923

## Data provenance

- PitchGPT checkpoint: trained 2015-2022 (pitcher-disjoint) per validate_pitchgpt_20260418T150305Z.
- LSTM: freshly trained on the same 2015-2022 pitcher-disjoint cohort.
- Markov-2, Heuristic: fit on the same training sequences.
- 2025 was NOT observed during training or validation of any model.