# PitchGPT 2025 Holdout Report

Generated: 2026-04-19T00:28:44Z

Checkpoint: `C:\Users\hunte\projects\baseball\results\validate_pitchgpt_20260418T150305Z\pitchgpt_full.pt`

Train: 2015-2022 (8412 sequences)

Val:   2023 (469 sequences, used only for temperature scaling)

Holdout: 2025 (1535 sequences, pitcher-disjoint from train)


## Leakage audit

- shared game_pks across splits: **0**
- shared pitchers train/holdout: **0**
- train pitchers: 1426  val: 115  holdout: 334

## Holdout perplexity (lower is better)

| Model | Params | Holdout PPL | 95% CI | N pitches |
|---|---|---|---|---|
| pitchgpt | 1,398,562 | 152.187 | 150.086 – 154.265 | 53723 |
| lstm | 837,154 | 176.554 | 174.058 – 179.036 | 53723 |
| markov2 | 0 | 352.657 | 348.082 – 357.483 | 53723 |
| heuristic | 0 | 471.856 | 466.807 – 476.669 | 53723 |

## Gates

| Comparison | Spec | Point | 95% CI | Point verdict | CI-lower verdict |
|---|---|---|---|---|---|
| PitchGPT vs LSTM | ≥15% | +13.80% | +12.22 / +15.51 | FAIL | FAIL |
| PitchGPT vs Markov-2 | ≥20% | +56.85% | +55.96 / +57.64 | PASS | PASS |
| PitchGPT vs Heuristic | ≥25% | +67.75% | +67.18 / +68.29 | PASS | PASS |

**Overall (point)**: FAIL
**Overall (CI lower bound)**: FAIL

## Calibration on 2025 (out-of-sample)

- ECE pre-temperature: **0.0201** (gate <0.10: PASS)
- ECE post-temperature (T=1.1201): **0.0098** (gate <0.10: PASS)
- Top-1 accuracy: 0.0341
- Non-PAD pitches scored: 53723

## Data provenance

- PitchGPT checkpoint: trained 2015-2022 (pitcher-disjoint) per validate_pitchgpt_20260418T150305Z.
- LSTM: freshly trained on the same 2015-2022 pitcher-disjoint cohort.
- Markov-2, Heuristic: fit on the same training sequences.
- 2025 was NOT observed during training or validation of any model.