# A3 — XGBoost on Engineered Features

**Verdict:** **WEAKER PASS**

## Headline (2025 holdout, post-temperature)

- 7-class log-loss: **1.3853**  (freq prior: 1.6516)
- Lift vs frequency prior: **+16.12%** (95% CI [+15.87%, +16.37%])
- 10-bin ECE post-temp: **0.0181**
- Top-1 accuracy: 0.4755
- Best params: max_depth=8, lr=0.08, best_iteration=185
- Temperature: 0.7988

## Cohort

- Train rows: 2,879,316
- Val rows (2023 pitcher-disjoint): 77,281
- Test rows (2025 pitcher-disjoint): 210,482
- Feature width: 16

## CV history (mean per (max_depth, learning_rate))

| max_depth | learning_rate | CV log-loss mean | std |
|---:|---:|---:|---:|
| 6 | 0.08 | 1.3859 | 0.0009 |
| 6 | 0.12 | 1.3868 | 0.0009 |
| 8 | 0.08 | 1.3845 | 0.0015 |
| 8 | 0.12 | 1.3856 | 0.0008 |

## Per-class log-loss (test, post-temp)

| class | log-loss |
|-------|---------:|
| ball | 1.0062 |
| called_strike | 1.2337 |
| swinging_strike | 1.5433 |
| foul | 1.6764 |
| in_play_out | 1.6418 |
| in_play_hit | 2.3113 |
| hbp | 3.5708 |

## Val metrics (used for early-stopping + temperature)

- Log-loss pre/post: 1.3186 / 1.3042
- ECE pre/post: 0.0750 / 0.0213
- Lift vs freq prior: +20.81%

## Per-pitcher stability (test)

- Top-50 pitchers (n>=30 rows each): n=50
- Log-loss mean / var / range: 1.3689 / 0.0105 / [1.2729, 1.9102]

## Wall clock

- Cohort: 1.1s
- Featurise: 34.2s
- Final fit: 315.2s
- Total: 540.1s
