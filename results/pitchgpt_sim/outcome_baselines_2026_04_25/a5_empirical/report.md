# A5 — Per-Pitcher × Per-Count × Per-Pitch-Type Empirical Priors

**Verdict:** **FAIL**

## Headline (2025 holdout, post-temperature)

- 7-class log-loss: **1.5800**  (freq prior: 1.6516)
- Lift vs frequency prior: **+4.33%** (95% CI [+4.24%, +4.44%])
- 10-bin ECE post-temp: **0.0015**
- Top-1 accuracy: 0.3674
- Temperature: 1.0331 (fit on 2023 val)

## Cohort

- Train rows: 2,879,316
- Val rows (2023 pitcher-disjoint): 77,281
- Test rows (2025 pitcher-disjoint): 210,482

## Per-class log-loss (test, post-temp)

| class | log-loss |
|-------|---------:|
| ball | 1.0150 |
| called_strike | 1.5362 |
| swinging_strike | 2.1094 |
| foul | 1.6047 |
| in_play_out | 2.1325 |
| in_play_hit | 2.8585 |
| hbp | 5.5437 |

## Val metrics (2023 pitcher-disjoint, used for hyperparam + T)

- Log-loss pre/post temp: 1.5764 / 1.5762
- ECE pre/post temp: 0.0042 / 0.0075
- Lift vs freq prior: +4.29%

## Per-pitcher stability (test)

- Top-50 pitchers (n>=30 rows each): n=50
- Log-loss mean / var / range: 1.5780 / 0.0006 / [1.5304, 1.6281]

## Lookup utilisation (test)

- level_0: 0
- level_1: 210,474
- level_2: 4
- level_3: 4
- fallback_global: 0

## Hierarchy levels

- Level 0: ['pitcher_id', 'balls', 'strikes', 'pitch_type', 'stand']
- Level 1: ['balls', 'strikes', 'pitch_type', 'stand']
- Level 2: ['pitch_type', 'stand']
- Level 3: []

## Gate criteria

- PASS: lift >= 10% AND CI lower >= 5% AND ECE < 0.05 AND hit ll < 2.0 AND hbp ll < 4.0
- WEAKER PASS: lift >= 5% AND CI lower >= 2% AND ECE < 0.05 AND hit ll < 2.5 AND hbp ll < 5.0
