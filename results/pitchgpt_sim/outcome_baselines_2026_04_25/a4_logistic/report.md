# A4 — Multinomial Logistic Regression on Engineered Features

**Verdict:** **WEAKER PASS**

## Headline (2025 holdout, post-temperature)

- 7-class log-loss: **1.3650**  (freq prior: 1.6516)
- Lift vs frequency prior: **+17.35%** (95% CI [+17.14%, +17.57%])
- 10-bin ECE post-temp: **0.0264**
- Top-1 accuracy: 0.4897
- Best C: 1.0  Temperature: 0.7914

## Cohort

- Train rows: 2,879,316
- Val rows (2023 pitcher-disjoint): 77,281
- Test rows (2025 pitcher-disjoint): 210,482
- Feature width: 71

## Tune history (val log-loss per C)

| C | val log-loss | fit (s) |
|---|---:|---:|
| 0.3 | 1.3669 | 25.2 |
| 1 | 1.3652 | 21.4 |
| 3 | 1.3675 | 24.2 |

## Per-class log-loss (test, post-temp)

| class | log-loss |
|-------|---------:|
| ball | 0.9582 |
| called_strike | 1.2824 |
| swinging_strike | 1.5619 |
| foul | 1.6047 |
| in_play_out | 1.5888 |
| in_play_hit | 2.3699 |
| hbp | 4.9133 |

## Val metrics (used for hyperparam + T)

- Log-loss pre/post: 1.3652 / 1.3504
- ECE pre/post: 0.0868 / 0.0282
- Lift vs freq prior: +18.00%

## Per-pitcher stability (test)

- Top-50 pitchers: n=50
- Log-loss mean / var / range: 1.3569 / 0.0010 / [1.2828, 1.4176]

## Wall clock

- Cohort: 1.2s
- Featurise: 14.5s
- Final fit: 21.5s
- Total: 196.1s
