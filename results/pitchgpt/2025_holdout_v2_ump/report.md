# PitchGPT 2025 Holdout Report

Generated: 2026-04-23T06:22:14Z

Checkpoint: `results\validate_pitchgpt_v2_ump_20260423T061634Z\pitchgpt_full.pt`

Train: 2015-2022 (8574 sequences)

Val:   2023 (452 sequences, used only for temperature scaling)

Holdout: 2025 (1489 sequences, pitcher-disjoint from train)


## Leakage audit

- shared game_pks across splits: **0**
- shared pitchers train/holdout: **0**
- train pitchers: 1457  val: 132  holdout: 325

## Holdout perplexity (lower is better)

| Model | Params | Holdout PPL | 95% CI | N pitches |
|---|---|---|---|---|
| pitchgpt | 1,398,690 | 159.142 | 156.967 – 161.278 | 51090 |
| lstm | 837,282 | 180.416 | 177.878 – 183.318 | 51090 |
| markov2 | 0 | 350.84 | 346.309 – 356.028 | 51090 |
| heuristic | 0 | 470.942 | 465.979 – 476.065 | 51090 |

## Gates

| Comparison | Spec | Point | 95% CI | Point verdict | CI-lower verdict |
|---|---|---|---|---|---|
| PitchGPT vs LSTM | ≥15% | +11.79% | +10.15 / +13.61 | FAIL | FAIL |
| PitchGPT vs Markov-2 | ≥20% | +54.64% | +53.68 / +55.53 | PASS | PASS |
| PitchGPT vs Heuristic | ≥25% | +66.21% | +65.61 / +66.84 | PASS | PASS |

**Overall (point)**: FAIL
**Overall (CI lower bound)**: FAIL

## Calibration on 2025 (out-of-sample)

- ECE pre-temperature: **0.0106** (gate <0.10: PASS)
- ECE post-temperature (T=1.101): **0.003** (gate <0.10: PASS)
- Top-1 accuracy: 0.0366
- Non-PAD pitches scored: 51090

## Data provenance

- PitchGPT checkpoint: trained 2015-2022 (pitcher-disjoint) per validate_pitchgpt_20260418T150305Z.
- LSTM: freshly trained on the same 2015-2022 pitcher-disjoint cohort.
- Markov-2, Heuristic: fit on the same training sequences.
- 2025 was NOT observed during training or validation of any model.