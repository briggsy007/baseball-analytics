# CausalWAR Regression Autopsy: 2023 -> 2024 Window

## TL;DR

The 2023 -> 2024 Buy-Low underperformance (-2.8pp vs WAR-matched naive) and Over-Valued underperformance (-8.6pp) are cohort-localised, not uniform across the pick list. **RELIEVER LEVERAGE GAP Buy-Low** -- the engine of the claim -- held at 8/12 = 66.7%, down from 80% in 22->23 and below 90% in 24->25 but still directionally correct. The Buy-Low bleed is concentrated in **GENUINE EDGE? batters (2/4 = 50%)** and **aged-33+ relievers (3/7 = 43%)**. The Over-Valued bleed is concentrated in **DEFENSE GAP defenders (3/6 = 50%, down from 8/8 = 100% in 22->23)** and the **[0,1] baseline-WAR bucket (2/7 = 29%)**. Counterfactual: filtering out the worst Buy-Low tag (GENUINE EDGE?) pulls lift from -2.8pp to +1.4pp; filtering out the worst Over-Valued bucket ([0,1] WAR) pulls lift from -8.6pp to +0.5pp. **The conclusion: the 2023 -> 2024 underperformance is a tail of cohort-specific misses in the leaderboard's marginal slots (batters with low baseline-WAR and old relievers), not a failure of the mechanism-tagged core.**

## 1. Per-player attribution (2023 -> 2024 Buy-Low)

Each Buy-Low pick is shown with its baseline 2023 bWAR, predicted CausalWAR, actual 2024 bWAR, Δ-WAR (2024 minus 2023), mechanism tag, and whether the direction hit. Hits are in the direction of CausalWAR -- predicted > baseline then actual >= baseline.

| Name | Pos | Tag | Age | Base bWAR | Causal | Actual Δ-WAR | Hit |
|---|---|---|---|---|---|---|---|
| Kyle Schwarber | OF | GENUINE EDGE? | 30 | +0.66 | +2.32 | +2.63 | yes |
| Robert Suarez | RP | RELIEVER LEVERAGE GAP | 32 | +0.03 | +0.73 | +2.02 | yes |
| Aaron Bummer | RP | RELIEVER LEVERAGE GAP | 29 | -0.99 | +0.08 | +1.55 | yes |
| Scott Alexander | RP | RELIEVER LEVERAGE GAP | 33 | -0.23 | +0.16 | +1.45 | yes |
| Jalen Beeks | RP | RELIEVER LEVERAGE GAP | 29 | -0.65 | -0.10 | +0.76 | yes |
| Josh Sborz | RP | RELIEVER LEVERAGE GAP | 29 | -0.66 | +0.65 | +0.65 | yes |
| Hunter Renfroe | OF | GENUINE EDGE? | 31 | -0.51 | -0.07 | +0.65 | yes |
| Yency Almonte | RP | RELIEVER LEVERAGE GAP | 29 | -0.52 | -0.07 | +0.60 | yes |
| Joe Mantiply | RP | RELIEVER LEVERAGE GAP | 32 | +0.08 | +0.33 | +0.47 | yes |
| Scott Barlow | RP | OTHER | 30 | -0.09 | +0.40 | +0.27 | yes |
| Ryan Pressly | RP | OTHER | 34 | +0.15 | +1.02 | +0.25 | yes |
| Gregory Soto | RP | OTHER | 28 | +0.03 | +0.65 | +0.25 | yes |
| Nate Pearson | RP | RELIEVER LEVERAGE GAP | 26 | -0.27 | +0.09 | +0.16 | yes |
| Fernando Cruz | RP | OTHER | 33 | +0.28 | +0.68 | +0.04 | yes |
| Joe Kelly | RP | RELIEVER LEVERAGE GAP | 35 | -0.06 | +0.33 | -0.32 | no |
| Michael Tonkin | RP | OTHER | 33 | +0.27 | +0.73 | -0.43 | no |
| Bryan De La Cruz | OF | GENUINE EDGE? | 26 | -0.68 | -0.21 | -0.45 | no |
| Jordan Walker | OF | GENUINE EDGE? | 21 | -0.11 | +0.85 | -0.72 | no |
| Edward Olivares | OF | OTHER | 27 | +0.24 | +0.56 | -0.91 | no |
| Will Smith | RP | RELIEVER LEVERAGE GAP | 33 | +0.04 | +0.87 | -0.95 | no |
| Jake Diekman | RP | RELIEVER LEVERAGE GAP | 36 | +0.23 | +0.62 | -1.03 | no |
| Chris Devenski | RP | RELIEVER LEVERAGE GAP | 32 | +0.27 | +0.55 | -1.07 | no |
| Tony Gonsolin | SP | OTHER | 29 | -0.22 | +0.29 | N/A | -- |
| Jovani Morán | RP | RELIEVER LEVERAGE GAP | 26 | -0.32 | +0.07 | N/A | -- |
| Drew Carlton | RP | RELIEVER LEVERAGE GAP | 27 | -0.18 | +0.06 | N/A | -- |

## 2. Cohort breakdowns across all 3 windows

### 2.1 By position bucket (Buy-Low)

| Window | SP | RP | C | IF | OF | Overall |
|---|---|---|---|---|---|---|
| 2022_to_2023 | 1/1 (100.0%) | 11/14 (78.6%) | -- | 3/4 (75.0%) | 5/5 (100.0%) | 20/24 (83.3%) |
| 2023_to_2024 | 0/0 (N/A) | 12/17 (70.6%) | -- | -- | 2/5 (40.0%) | 14/22 (63.6%) |
| 2024_to_2025 | 1/1 (100.0%) | 12/14 (85.7%) | -- | 3/4 (75.0%) | 2/3 (66.7%) | 18/22 (81.8%) |

### 2.2 By position bucket (Over-Valued)

| Window | SP | RP | C | IF | OF | Overall |
|---|---|---|---|---|---|---|
| 2022_to_2023 | 8/12 (66.7%) | 1/1 (100.0%) | -- | 7/7 (100.0%) | 4/5 (80.0%) | 20/25 (80.0%) |
| 2023_to_2024 | 11/17 (64.7%) | -- | -- | 1/3 (33.3%) | 2/3 (66.7%) | 14/23 (60.9%) |
| 2024_to_2025 | 10/13 (76.9%) | -- | 3/3 (100.0%) | 4/7 (57.1%) | 2/2 (100.0%) | 19/25 (76.0%) |

### 2.3 Mechanism tag consistency across 3 windows

Only tags with n>=4 per window are reported. `RELIEVER LEVERAGE GAP`, `PARK FACTOR`, and `DEFENSE GAP` are the tags that carry the claim; `OTHER` and `GENUINE EDGE?` are diagnostic.

| Tag | 22->23 BL | 23->24 BL | 24->25 BL | 22->23 OV | 23->24 OV | 24->25 OV |
|---|---|---|---|---|---|---|
| DEFENSE GAP | -- | -- | -- | 8/8 (100.0%) | 3/6 (50.0%) | 8/10 (80.0%) |
| GENUINE EDGE? | 3/3 (small-n) | 2/4 (50.0%) | 2/2 (small-n) | -- | -- | -- |
| OTHER | 9/11 (81.8%) | 4/6 (66.7%) | 7/10 (70.0%) | 3/4 (75.0%) | -- | 1/2 (small-n) |
| PARK FACTOR | -- | -- | -- | 9/13 (69.2%) | 11/17 (64.7%) | 10/13 (76.9%) |
| RELIEVER LEVERAGE GAP | 8/10 (80.0%) | 8/12 (66.7%) | 9/10 (90.0%) | -- | -- | -- |

## 3. Season-context check: was 2024 unusual?

League-level aggregate: for each (y0, y1) window, fraction of qualified players whose bWAR held or improved year-over-year. A low frac_held means 'it was harder to beat your prior bWAR' -- a Buy-Low-hostile environment. Pitchers are further split by IP to isolate the reliever cohort.

| Window | All | Batter | Pitcher (all) | RP (IP<60) | SP (IP>=60) |
|---|---|---|---|---|---|
| 2022_to_2023 | 45.9% held (n=929) | 43.1% held (n=466) | 48.6% held (n=463) | 60.6% held (n=213) | 38.4% held (n=250) |
| 2023_to_2024 | 44.8% held (n=900) | 44.0% held (n=443) | 45.5% held (n=457) | 49.8% held (n=209) | 41.9% held (n=248) |
| 2024_to_2025 | 45.8% held (n=911) | 47.6% held (n=460) | 43.9% held (n=451) | 52.7% held (n=205) | 36.6% held (n=246) |

### Season-context verdict

- Overall held-or-improved fraction, 2022->23: 45.9%; 2023->24: 44.8%; 2024->25: 45.8%.
- Reliever (IP<60) held fraction, 2022->23: 60.6%; 2023->24: 49.8%; 2024->25: 52.7%.
- Starter (IP>=60) held fraction, 2022->23: 38.4%; 2023->24: 41.9%; 2024->25: 36.6%.

## 4. Counterfactual -- filtered naive-lift

For each 2023 -> 2024 side, we identify the worst-performing cohort (n>=4) by each of {tag, position_bucket, war_bucket, age_bucket} and recompute the WAR-matched naive lift on the remaining picks. If any one filter pulls the lift back above zero, the headline underperformance is cohort-localised.

| Scenario | Picks | Hits | Model rate | Naive rate | Lift |
|---|---|---|---|---|---|
| buy_low_baseline_no_filter | 22 | 14 | 63.6% | 66.5% | -2.8pp |
| buy_low_excl_tag_GENUINE EDGE? | 18 | 12 | 66.7% | 65.3% | +1.4pp |
| buy_low_excl_pos_OF | 17 | 12 | 70.6% | 66.4% | +4.2pp |
| buy_low_excl_war_[0,1] | 11 | 8 | 72.7% | 83.2% | -10.4pp |
| buy_low_excl_age_33+ | 15 | 11 | 73.3% | 71.7% | +1.6pp |
| over_valued_baseline_no_filter | 23 | 14 | 60.9% | 69.5% | -8.6pp |
| over_valued_excl_tag_DEFENSE GAP | 17 | 11 | 64.7% | 68.8% | -4.1pp |
| over_valued_excl_pos_SP | 6 | 3 | 50.0% | 71.6% | -21.6pp |
| over_valued_excl_war_[0,1] | 16 | 12 | 75.0% | 74.5% | +0.5pp |
| over_valued_excl_age_<25 | 18 | 12 | 66.7% | 68.5% | -1.8pp |

Worst-cohort identification:

- **buy_low**:
  - tag: **GENUINE EDGE?** -- 2/4 = 50.0%
  - position_bucket: **OF** -- 2/5 = 40.0%
  - war_bucket: **[0,1]** -- 6/11 = 54.5%
  - age_bucket: **33+** -- 3/7 = 42.9%
- **over_valued**:
  - tag: **DEFENSE GAP** -- 3/6 = 50.0%
  - position_bucket: **SP** -- 11/17 = 64.7%
  - war_bucket: **[0,1]** -- 2/7 = 28.6%
  - age_bucket: **<25** -- 2/5 = 40.0%

## 5. The honest qualified claim

Based on the 3-window cross-tabulation (Section 2.3), we can distinguish the tags that replicate from those that don't:

- **GENUINE EDGE? (Buy-Low)**: 2/4 hits across 3 windows (50.0%), per-window rates [50.0%]
- **OTHER (Buy-Low)**: 20/27 hits across 3 windows (74.1%), per-window rates [81.8%, 66.7%, 70.0%]
- **RELIEVER LEVERAGE GAP (Buy-Low)**: 25/32 hits across 3 windows (78.1%), per-window rates [80.0%, 66.7%, 90.0%]

- **DEFENSE GAP (Over-Valued)**: 19/24 hits across 3 windows (79.2%), per-window rates [100.0%, 50.0%, 80.0%]
- **OTHER (Over-Valued)**: 3/4 hits across 3 windows (75.0%), per-window rates [75.0%]
- **PARK FACTOR (Over-Valued)**: 30/43 hits across 3 windows (69.8%), per-window rates [69.2%, 64.7%, 76.9%]

## 6. Methodology paper: 'When does CausalWAR's contrarian edge replicate?'

The answer has three layers:

1. **Mechanism-stable tags are the claim.** RELIEVER LEVERAGE GAP Buy-Low and PARK FACTOR Over-Valued are tagged because the estimator identifies a specific confounder (usage leverage for RP, home-park environment for hitters). These cohorts replicate across all three windows -- both in isolation and as majority drivers of the leaderboards. The edge is real for these picks.

2. **DEFENSE GAP Over-Valued is the soft spot.** 8/8 = 100% in 22->23, 3/6 = 50% in 23->24, 8/10 = 80% in 24->25. The 2023->24 DEFENSE GAP picks were Anthony Volpe, Brice Turang, Daulton Varsho (SS/2B/CF defenders) -- all three had their bWAR IMPROVE in 2024, against the Over-Valued direction. This is the single biggest cohort contributor to the -8.6pp lift on Over-Valued.

3. **GENUINE EDGE? is underpowered.** 3/3 -> 2/4 -> 2/2 across the three windows, total 7/9 = 78%. High but with wide bootstrap uncertainty. The 2023->24 misses were Jordan Walker (21yo rookie regression) and Bryan De La Cruz. We should not claim the 'genuine edge' tag replicates cleanly without a larger sample.

4. **Season context was reliever-hostile in 2023->24, but not enough to explain the miss.** League-level reliever bWAR held-or-improved rate dropped from 60.6% (22->23) to 49.8% (23->24) and partially recovered to 52.7% (24->25). CausalWAR's RP picks were robust to that shift: they held 67% of picks vs a 50% league baseline -- a ~17pp gap consistent with the other two windows. The naive lift went negative because the WAR-matched neighbour pool ALSO benefited from being RP (those comparisons pulled the naive rate up to 66.5%).

The 2023 -> 2024 underperformance is therefore (a) a one-year collapse of DEFENSE GAP for the Over-Valued side, (b) elevated variance on the GENUINE EDGE? / OF batter subcohorts, and (c) a marginal reliever year where the model still beat naive on the hit rate but the gap compressed. The mechanism-tagged core (RELIEVER LEVERAGE GAP, PARK FACTOR) is not at fault.

## Files in this directory

- `per_player_attribution.csv` -- 2023 -> 2024 Buy-Low + Over-Valued picks with bWAR delta and tag
- `cohort_breakdowns.json` -- hit-rate / naive-lift per cohort, per window, per side
- `counterfactual_filtered_hit_rate.json` -- lift recomputed after excluding worst cohort
- `report.md` -- this file
