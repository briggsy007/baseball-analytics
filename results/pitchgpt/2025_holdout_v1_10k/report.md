# PitchGPT 2025 Holdout Report

Generated: 2026-04-24T15:19:45Z

Checkpoint: `models\pitchgpt_v1_10k.pt`

Train: 2015-2022 (86054 sequences)

Val:   2023 (1507 sequences, used only for temperature scaling)

Holdout: 2025 (5958 sequences, pitcher-disjoint from train)


## Leakage audit

- shared game_pks across splits: **0**
- shared pitchers train/holdout: **0**
- train pitchers: 2062  val: 160  holdout: 477

## Holdout perplexity (lower is better)

| Model | Params | Holdout PPL | 95% CI | N pitches |
|---|---|---|---|---|
| pitchgpt | 1,398,562 | 119.829 | 119.091 – 120.601 | 204403 |
| lstm | 837,154 | 122.99 | 122.208 – 123.803 | 204403 |
| markov2 | 0 | 344.072 | 341.682 – 346.505 | 204403 |
| heuristic | 0 | 467.144 | 464.605 – 469.591 | 204403 |

## Gates

| Comparison | Spec | Point | 95% CI | Point verdict | CI-lower verdict |
|---|---|---|---|---|---|
| PitchGPT vs LSTM | ≥15% | +2.57% | +1.68 / +3.43 | FAIL | FAIL |
| PitchGPT vs Markov-2 | ≥20% | +65.17% | +64.84 / +65.50 | PASS | PASS |
| PitchGPT vs Heuristic | ≥25% | +74.35% | +74.14 / +74.57 | PASS | PASS |

**Overall (point)**: FAIL
**Overall (CI lower bound)**: FAIL

## Calibration on 2025 (out-of-sample)

- ECE pre-temperature: **0.0131** (gate <0.10: PASS)
- ECE post-temperature (T=1.0476): **0.009** (gate <0.10: PASS)
- Top-1 accuracy: 0.0408
- Non-PAD pitches scored: 204403

## Data provenance

- PitchGPT checkpoint: trained 2015-2022 (pitcher-disjoint) per validate_pitchgpt_20260418T150305Z.
- LSTM: freshly trained on the same 2015-2022 pitcher-disjoint cohort.
- Markov-2, Heuristic: fit on the same training sequences.
- 2025 was NOT observed during training or validation of any model.