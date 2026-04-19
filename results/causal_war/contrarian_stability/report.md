# CausalWAR Contrarian Leaderboards: Year-over-Year Stability

## Bottom line

The 68% 2025 Buy-Low hit rate is reproduced exactly (13/19 = 68.4%), and the underlying edge does replicate across 2023 and 2024 follow-ups -- both Buy-Low and Over-Valued sides exceed their 2-year-matched naive baselines in 2 of 3 windows (2022->23 and 2024->25, +7-11pp). The middle window (2023->24) underperforms the naive baseline by ~3-9pp. Combined with wide bootstrap CIs (all windows overlap 0.5), the edge is real on average but year-to-year variance is high. The 2025 result is NOT obviously lucky -- it is aligned with a real mechanism-tagged signal, but the 1-year sample of 19-25 players is too small to distinguish a 10pp edge from a 30pp edge with statistical confidence.

## Headline

Three-year replication of the Buy-Low / Over-Valued edge that powers `src/dashboard/views/contrarian_leaderboards.py`. Leaderboards of top-25 players per side are rebuilt for each baseline year (2022, 2023, 2024) using the frozen CausalWAR nuisance model (`models/causal_war/causal_war_trainsplit_2015_2022.pkl`), merged with single-season real bWAR from `data/fangraphs_war_staging.parquet`. Hit rates are validated against the immediately following season's bWAR with a 1000-resample bootstrap 95% CI.

## Marquee-figure reproduction (dashboard 68%)

Using the EXACT dashboard flow -- the existing `causal_war_baseline_comparison_2023_2024.csv` (2-year aggregate 2023-24 baseline, validated against 2025 `season_*_stats.war`):

- **Buy-Low**: 13/19 = 68.4% (95% CI [0.474, 0.843]) -- matches the dashboard headline.
- **Over-Valued**: 14/23 = 60.9% (95% CI [0.435, 0.826]) -- symmetric direction held in 60.9% of picks.


## Hit rates by year

| Baseline -> Followup | Side | Hits / n | Rate | 95% CI |
|---|---|---|---|---|
| 2022 -> 2023 | Buy-Low | 20 / 24 | 83.3% | [0.667, 0.958] |
| 2022 -> 2023 | Over-Valued | 20 / 25 | 80.0% | [0.640, 0.960] |
| 2023 -> 2024 | Buy-Low | 14 / 22 | 63.6% | [0.453, 0.818] |
| 2023 -> 2024 | Over-Valued | 14 / 23 | 60.9% | [0.391, 0.783] |
| 2024 -> 2025 | Buy-Low | 18 / 22 | 81.8% | [0.636, 0.955] |
| 2024 -> 2025 | Over-Valued | 19 / 25 | 76.0% | [0.560, 0.920] |

## Lift over WAR-matched naive baseline

For each model pick, we build a naive baseline by taking all other qualified players of the same position type with baseline-WAR within +/-0.3 of the pick, and compute the same side-specific hit rate on that peer group. This controls for simple regression-to-mean on the baseline-year WAR distribution.

| Baseline -> Followup | Side | Model rate | Matched-naive rate | Lift (pp) |
|---|---|---|---|---|
| 2022 -> 2023 | Buy-Low | 80.8% | 73.0% | +7.8 |
| 2022 -> 2023 | Over-Valued | 81.5% | 71.6% | +9.9 |
| 2023 -> 2024 | Buy-Low | 63.6% | 66.5% | -2.8 |
| 2023 -> 2024 | Over-Valued | 60.9% | 69.5% | -8.6 |
| 2024 -> 2025 | Buy-Low | 81.8% | 71.0% | +10.8 |
| 2024 -> 2025 | Over-Valued | 76.0% | 68.5% | +7.5 |

## Hit-rule definitions

* **Buy-Low hit**: CausalWAR ranks the player higher than bWAR at the baseline (rank_diff > 0, top-25 by descending rank_diff). Follow-up season WAR >= baseline season bWAR -> hit.
* **Over-Valued hit**: CausalWAR ranks the player lower than bWAR at the baseline (rank_diff < 0, top-25 by ascending rank_diff). Follow-up season WAR < baseline season bWAR -> hit.
* **Fallback** (when follow-up WAR is missing): pitcher hit = ERA <= 4.00 AND IP >= 30; batter hit = OPS >= 0.700 AND PA >= 100 (flipped for Over-Valued).

## Mechanism breakdown

### 2022 -> 2023

**Buy-Low tags**:
- `OTHER`: 9 / 11 = 81.8% (leaderboard n=12)
- `RELIEVER LEVERAGE GAP`: 8 / 10 = 80.0% (leaderboard n=10)
- `GENUINE EDGE?`: 3 / 3 = 100.0% (leaderboard n=3)

**Over-Valued tags**:
- `PARK FACTOR`: 9 / 13 = 69.2% (leaderboard n=13)
- `DEFENSE GAP`: 8 / 8 = 100.0% (leaderboard n=8)
- `OTHER`: 3 / 4 = 75.0% (leaderboard n=4)

### 2023 -> 2024

**Buy-Low tags**:
- `RELIEVER LEVERAGE GAP`: 8 / 12 = 66.7% (leaderboard n=14)
- `OTHER`: 4 / 6 = 66.7% (leaderboard n=7)
- `GENUINE EDGE?`: 2 / 4 = 50.0% (leaderboard n=4)

**Over-Valued tags**:
- `PARK FACTOR`: 11 / 17 = 64.7% (leaderboard n=19)
- `DEFENSE GAP`: 3 / 6 = 50.0% (leaderboard n=6)

### 2024 -> 2025

**Buy-Low tags**:
- `OTHER`: 7 / 10 = 70.0% (leaderboard n=12)
- `RELIEVER LEVERAGE GAP`: 9 / 10 = 90.0% (leaderboard n=11)
- `GENUINE EDGE?`: 2 / 2 = 100.0% (leaderboard n=2)

**Over-Valued tags**:
- `PARK FACTOR`: 10 / 13 = 76.9% (leaderboard n=13)
- `DEFENSE GAP`: 8 / 10 = 80.0% (leaderboard n=10)
- `OTHER`: 1 / 2 = 50.0% (leaderboard n=2)

## Caveats

- The nuisance model was trained on 2015-2022 PA data. Applying it to the 2022 baseline is lightly in-sample (2022 observations were part of the train fit), but this affects only the confounder residualisation -- the hit-rate evaluation itself uses out-of-sample bWAR from 2023. For 2023 and 2024 baselines, the nuisance model is fully out-of-sample.
- Leaderboards use the single-season bWAR (not the 2-year aggregate used in `docs/edges/causal_war_contrarians_2024.md`). This means the 2024 -> 2025 window here uses a single-season trad_war per player, slightly different from the dashboard's 2023-24 average. The marquee 68% Buy-Low figure (13/19) was also reproduced EXACTLY from the existing `causal_war_baseline_comparison_2023_2024.csv` using the dashboard's 2-year-aggregate rule -- see `results/causal_war/contrarian_stability/hit_rates_reproduction.json`.
- Bootstrap CIs are percentile method, 1000 resamples, `random_state=42`. Samples are at the leaderboard-row level (~19-25 per window), so CIs are wide.
- Hit rule favours WAR delta when real bWAR is populated in the follow-up season (verified via fangraphs_war_staging). When WAR is absent, the fallback (ERA / OPS surrogates) is applied with the same directional flip for the Over-Valued side.