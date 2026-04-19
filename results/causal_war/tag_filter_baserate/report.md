# Does the RELIEVER LEVERAGE GAP tag filter explain the contrarian edge?

## TL;DR

At IP<60 (WAR filter = 'any', the MOST honest comparator because CausalWAR picks span WAR in [-1.0, 0.3]), using the dashboard Buy-Low rule, pooled across 2022->23, 2023->24, 2024->25:

| Group | Description | n | Hit rate | 95% CI |
|---|---|---|---|---|
| A | CausalWAR picks | 32 | 78.1% | [0.625, 0.906] |
| B | Tag-filter universe | 1159 | 56.9% | [0.542, 0.597] |
| C | Random-within-filter (mean) | n_sample=32 | 56.9% | [0.406, 0.750] |
| D | Top-N by year-N bWAR | 30 | 10.0% | [0.000, 0.233] |

## Reading the comparison

- **If Group B is close to 50%** then the tag filter does not produce its own survivorship bias -- CausalWAR's selection is doing real work.
- **If Group B ~= 78%** then the tag filter's structural properties (short-IP reliever + positive bWAR) already regress to create the headline rate, independent of CausalWAR.
- **If Group D ~= Group A** then a trivial 'top-N by bWAR' baseline reproduces the edge -- CausalWAR would be ornamental.
- **If Group A >> Group C mean** then CausalWAR adds value vs. random picks from the same universe.

## Per-window hit rates (dashboard rule, war_filter='any')

### IP < 50

| Window | A (CausalWAR) | B (universe) | C (random mean) | D (top-N bWAR) |
|---|---|---|---|---|
| 2022->2023 | 80.0% (8/10) | 59.4% (189/318) | 59.1% (n_sample=10, universe=318) | 12.5% (1/8) |
| 2023->2024 | 66.7% (8/12) | 58.0% (184/317) | 58.7% (n_sample=12, universe=317) | 20.0% (2/10) |
| 2024->2025 | 90.0% (9/10) | 56.3% (178/316) | 56.3% (n_sample=10, universe=316) | 11.1% (1/9) |

### IP < 60

| Window | A (CausalWAR) | B (universe) | C (random mean) | D (top-N bWAR) |
|---|---|---|---|---|
| 2022->2023 | 80.0% (8/10) | 59.6% (229/384) | 59.4% (n_sample=10, universe=384) | 11.1% (1/9) |
| 2023->2024 | 66.7% (8/12) | 55.6% (215/387) | 55.9% (n_sample=12, universe=387) | 8.3% (1/12) |
| 2024->2025 | 90.0% (9/10) | 55.4% (215/388) | 55.5% (n_sample=10, universe=388) | 11.1% (1/9) |

### IP < 70

| Window | A (CausalWAR) | B (universe) | C (random mean) | D (top-N bWAR) |
|---|---|---|---|---|
| 2022->2023 | 80.0% (8/10) | 56.7% (259/457) | 56.8% (n_sample=10, universe=457) | 0.0% (0/9) |
| 2023->2024 | 66.7% (8/12) | 53.5% (242/452) | 52.8% (n_sample=12, universe=452) | 10.0% (1/10) |
| 2024->2025 | 90.0% (9/10) | 53.2% (238/447) | 53.2% (n_sample=10, universe=447) | 10.0% (1/10) |

## Per-window hit rates (literal 'bWAR dropped' rule, war_filter='any')

### IP < 50

| Window | A (CausalWAR) | B (universe) | C (random mean) | D (top-N bWAR) |
|---|---|---|---|---|
| 2022->2023 | 20.0% (2/10) | 40.6% (129/318) | 40.9% (n_sample=10, universe=318) | 87.5% (7/8) |
| 2023->2024 | 33.3% (4/12) | 42.0% (133/317) | 41.3% (n_sample=12, universe=317) | 80.0% (8/10) |
| 2024->2025 | 10.0% (1/10) | 43.7% (138/316) | 43.7% (n_sample=10, universe=316) | 88.9% (8/9) |

### IP < 60

| Window | A (CausalWAR) | B (universe) | C (random mean) | D (top-N bWAR) |
|---|---|---|---|---|
| 2022->2023 | 20.0% (2/10) | 40.4% (155/384) | 40.6% (n_sample=10, universe=384) | 88.9% (8/9) |
| 2023->2024 | 33.3% (4/12) | 44.4% (172/387) | 44.1% (n_sample=12, universe=387) | 91.7% (11/12) |
| 2024->2025 | 10.0% (1/10) | 44.6% (173/388) | 44.5% (n_sample=10, universe=388) | 88.9% (8/9) |

### IP < 70

| Window | A (CausalWAR) | B (universe) | C (random mean) | D (top-N bWAR) |
|---|---|---|---|---|
| 2022->2023 | 20.0% (2/10) | 43.3% (198/457) | 43.2% (n_sample=10, universe=457) | 100.0% (9/9) |
| 2023->2024 | 33.3% (4/12) | 46.5% (210/452) | 47.2% (n_sample=12, universe=452) | 90.0% (9/10) |
| 2024->2025 | 10.0% (1/10) | 46.8% (209/447) | 46.8% (n_sample=10, universe=447) | 90.0% (9/10) |

## Pooled across 3 windows (dashboard rule)

| IP< | WAR filter | A | B | C mean | D |
|---|---|---|---|---|---|
| 50 | positive | 78.1% (25/32) | 39.6% (197/497) | 39.2% (n_sample=32) | 14.3% (4/28) |
| 60 | positive | 78.1% (25/32) | 38.7% (247/638) | 38.9% (n_sample=32) | 10.0% (3/30) |
| 70 | positive | 78.1% (25/32) | 37.2% (299/803) | 37.5% (n_sample=32) | 6.9% (2/29) |
| 50 | any | 78.1% (25/32) | 57.9% (551/951) | 57.8% (n_sample=32) | 14.8% (4/27) |
| 60 | any | 78.1% (25/32) | 56.9% (659/1159) | 56.9% (n_sample=32) | 10.0% (3/30) |
| 70 | any | 78.1% (25/32) | 54.5% (739/1356) | 54.4% (n_sample=32) | 6.9% (2/29) |
| 50 | match_group_a | 78.1% (25/32) | 63.0% (509/808) | 63.5% (n_sample=32) | 20.0% (4/20) |
| 60 | match_group_a | 78.1% (25/32) | 63.7% (586/920) | 63.8% (n_sample=32) | 16.7% (4/24) |
| 70 | match_group_a | 78.1% (25/32) | 64.4% (635/986) | 64.1% (n_sample=32) | 29.2% (7/24) |

## Pooled across 3 windows (drop rule)

| IP< | WAR filter | A | B | C mean | D |
|---|---|---|---|---|---|
| 50 | positive | 21.9% (7/32) | 60.4% (300/497) | 60.8% (n_sample=32) | 85.7% (24/28) |
| 60 | positive | 21.9% (7/32) | 61.3% (391/638) | 61.1% (n_sample=32) | 90.0% (27/30) |
| 70 | positive | 21.9% (7/32) | 62.8% (504/803) | 62.5% (n_sample=32) | 93.1% (27/29) |
| 50 | any | 21.9% (7/32) | 42.1% (400/951) | 42.2% (n_sample=32) | 85.2% (23/27) |
| 60 | any | 21.9% (7/32) | 43.1% (500/1159) | 43.1% (n_sample=32) | 90.0% (27/30) |
| 70 | any | 21.9% (7/32) | 45.5% (617/1356) | 45.6% (n_sample=32) | 93.1% (27/29) |
| 50 | match_group_a | 21.9% (7/32) | 37.0% (299/808) | 36.5% (n_sample=32) | 80.0% (16/20) |
| 60 | match_group_a | 21.9% (7/32) | 36.3% (334/920) | 36.2% (n_sample=32) | 83.3% (20/24) |
| 70 | match_group_a | 21.9% (7/32) | 35.6% (351/986) | 35.9% (n_sample=32) | 70.8% (17/24) |

## Verdict

- **Group B (tag-filter universe) base rate is 56.9%** (not 78%). The tag filter does NOT create the 78% hit rate on its own.
- **Group C (random-within-filter) mean is 56.9%**, with 95% CI [0.406, 0.750] -- Group A's 78.1% is +21.2pp above C's mean and clearly outside C's CI.
- **Group D (top-N by bWAR) is 10.0%**, CI [0.000, 0.233] -- far BELOW Group A; CIs do NOT overlap. 'Take the best short-IP relievers by bWAR' is the WORST strategy, not the best. CausalWAR is NOT ornamental.
- **Bottom line: the contrarian edge story survives.** CausalWAR's DML residual is selecting *which* short-IP relievers are Buy-Low, and the selection substantially outperforms both random picks and high-bWAR picks from the same filter universe.

### Note on the 'drop' rule
The dashboard's Buy-Low hit rule is `WAR_{N+1} >= WAR_N` (the model's bullish call held). Under that rule, the 78% headline is a real edge. If instead the question is 'did bWAR literally drop?' (the framing in the tag-filter-reinterpretation hypothesis), Group A actually drops LESS often than the universe (~21% vs ~61%). The top-bWAR baseline (Group D) drops 90%+ of the time -- confirming high-bWAR short-IP relievers DO regress, but that is precisely the pool from which CausalWAR *avoids* picking winners.

## Method

- Tag-filter universe rebuilt from `data/fangraphs_war_staging.parquet` per year using the same IP threshold as `src/analytics/causal_war.classify_row` (IP<60).
- Group A reads the historical CausalWAR Buy-Low picks from `results/causal_war/contrarian_stability/buy_low_*.csv` and filters `tag == 'RELIEVER LEVERAGE GAP'`.
- Group A's 'hit' column is the dashboard's hit (year-N+1 WAR >= baseline WAR). 'hit_drop' is rederived by joining follow-up bWAR.
- Hit evaluation uses real bWAR from fangraphs_war_staging (season 2023, 2024, 2025 already backfilled).
- Bootstrap CIs: 1000 resamples, random_state=42.