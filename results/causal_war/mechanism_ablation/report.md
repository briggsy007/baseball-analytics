# CausalWAR Mechanism Ablation: RELIEVER LEVERAGE GAP -- Causal Mechanism or Epiphenomenon?

## Question

The committed CausalWAR contrarian-leaderboard analysis reports three mechanism-tag hit rates over three consecutive year windows (2022->23, 2023->24, 2024->25):

| Tag | Side | Hits / n | Rate |
|---|---|---|---|
| RELIEVER LEVERAGE GAP | Buy-Low | 25/32 | 78.1% |
| PARK FACTOR | Over-Valued | 30/43 | 69.8% |
| DEFENSE GAP | Over-Valued | 19/24 | 79.2% |

The methodology paper currently phrases the claim as the tag patterns being "consistent with a causal driver." This ablation tests the stronger claim -- that the RELIEVER LEVERAGE GAP tag reflects a real role-awareness mechanism in the model -- by retraining CausalWAR with the only direct role-indicator feature (``inning_bucket``) zeroed out.

## What was ablated

The CausalWAR outcome nuisance model `E[Y|W]` is a `HistGradientBoostingRegressor` fit on a 12-column confounder matrix `W` defined in `src/analytics/causal_war._build_features`:

`[venue_code, platoon, on_1b, on_2b, on_3b, outs, inning_bucket, if_shift, of_shift, month, stand_R, p_throws_R]`

Of these, only ``inning_bucket`` (values 0-3 mapping to 1-3 / 4-6 / 7-9 / 10+ innings) serves as an implicit role indicator. Starters predominantly face batters in innings 1-6; relievers in 7+. There is no explicit ``position == RP`` or ``role_reliever`` feature in `W`. Rate stats that only relievers have (saves, holds) are not in `W` either -- `W` is constructed at the PA grain, not the player grain.

We set `W[:, 6] = 0` for all training PAs (2015-2022) and re-fit a single full-train `HistGradientBoostingRegressor` with the same hyperparameters as the committed baseline. The same ablation is applied at inference time on each 2022 / 2023 / 2024 season.

**The tag classifier is UNCHANGED.** It still reads `position == "pitcher"` and `ip_total < 60` from the Fangraphs WAR staging table to assign the RELIEVER LEVERAGE GAP label. Those signals come from outside the CausalWAR model. The ablation isolates whether the model's *internal* residualisation depends on knowing it's looking at a reliever.

Artifact: `models/causal_war/causal_war_trainsplit_2015_2022_noRP.pkl`.

## Three-year pooled comparison

| Mechanism tag | Original | Ablated | Delta | 95% CI (bootstrap) |
|---|---|---|---|---|
| RELIEVER LEVERAGE GAP (Buy-Low) | 78.1% (25/32) | 78.4% (29/37) | +0.3pp | [-19.5pp, +18.6pp] |
| PARK FACTOR (Over-Valued) | 69.8% (30/43) | 70.2% (33/47) | +0.4pp | [-17.4pp, +22.3pp] |
| DEFENSE GAP (Over-Valued) | 79.2% (19/24) | 73.7% (14/19) | -5.5pp | [-30.7pp, +19.5pp] |

## Verdict: **UNSUPPORTED -- RELIEVER LEVERAGE GAP did not collapse when role info was removed. The tag's predictive validity does not come from the model's role-awareness.**

### Interpretation

Dropping ``inning_bucket`` from the nuisance model reshuffles the 25-player Buy-Low leaderboard somewhat -- the number of picks that get the RELIEVER LEVERAGE GAP label grows from 32 to 37 across the three windows, because the model's pitcher residuals shift slightly when it can no longer residualise against late-inning context. But the *hit rate* on the reliever-tagged subset is essentially unchanged (78.1% -> 78.4%, delta +0.3pp, CI [-19.5pp, +18.6pp]). PARK FACTOR is identical (+0.4pp). DEFENSE GAP drifts -5.5pp but the CI straddles zero and the n dropped from 24 to 19, so the point estimate is noise-limited.

This means the mechanism claim in the paper -- that the tag reflects the *model* correctly identifying relievers as overvalued by bWAR's leverage-chained bookkeeping -- is **not supported by the ablation**. Two alternative explanations are consistent with the data:

1. **The tag does the work, not the model.** The RELIEVER LEVERAGE GAP label is assigned to rows where `position == "pitcher"`, `ip_total < 60`, and `causal_war > trad_war`. The first two conditions filter to a cohort (relievers with low IP) that is *already* a known regression-to-mean / over-fitting target of bWAR, irrespective of what CausalWAR thinks. The hit rate may largely reflect the reliever-filter's own prior, not a model-generated edge.
2. **Role is implicitly encoded by non-ablated features.** Even after zeroing `inning_bucket`, the nuisance model may still recover reliever context from `outs`, `on_1b/2b/3b`, or `platoon` (high-leverage PAs skew to 2-out, RISP, platoon-unfavourable situations). A stronger ablation would have to strip those too, at which point we are no longer testing "role-awareness" but "any contextual residualisation at all."

Either way, the stronger phrasing in the paper draft -- that RELIEVER LEVERAGE GAP is "demonstrated" as a causal mechanism by the model -- is not supported. The current "consistent with" phrasing is appropriate; claiming more would overstate the evidence. A possible remedy is to pivot the language: "The tag identifies a cohort (low-IP relievers whose CausalWAR > bWAR) that hit in 78% of follow-up years; this replicates whether or not the model is given explicit inning-context information, suggesting the edge lives in the tag's filter + the residual's direction, not in the model's role-awareness specifically."

## Per-window detail

| Window | Side | Tag | Original | Ablated |
|---|---|---|---|---|
| 2022->2023 | Buy-Low | GENUINE EDGE? | 100% (3/3) | 100% (3/3) |
| 2022->2023 | Buy-Low | OTHER | 82% (9/11) | 78% (7/9) |
| 2022->2023 | Buy-Low | RELIEVER LEVERAGE GAP | 80% (8/10) | 85% (11/13) |
| 2022->2023 | Over-Valued | DEFENSE GAP | 100% (8/8) | 100% (7/7) |
| 2022->2023 | Over-Valued | OTHER | 75% (3/4) | 75% (3/4) |
| 2022->2023 | Over-Valued | PARK FACTOR | 69% (9/13) | 57% (8/14) |
| 2023->2024 | Buy-Low | GENUINE EDGE? | 50% (2/4) | 50% (2/4) |
| 2023->2024 | Buy-Low | OTHER | 67% (4/6) | 86% (6/7) |
| 2023->2024 | Buy-Low | RELIEVER LEVERAGE GAP | 67% (8/12) | 67% (8/12) |
| 2023->2024 | Over-Valued | DEFENSE GAP | 50% (3/6) | 25% (1/4) |
| 2023->2024 | Over-Valued | PARK FACTOR | 65% (11/17) | 72% (13/18) |
| 2024->2025 | Buy-Low | GENUINE EDGE? | 100% (2/2) | 100% (1/1) |
| 2024->2025 | Buy-Low | OTHER | 70% (7/10) | 70% (7/10) |
| 2024->2025 | Buy-Low | RELIEVER LEVERAGE GAP | 90% (9/10) | 83% (10/12) |
| 2024->2025 | Over-Valued | DEFENSE GAP | 80% (8/10) | 75% (6/8) |
| 2024->2025 | Over-Valued | OTHER | 50% (1/2) | 50% (1/2) |
| 2024->2025 | Over-Valued | PARK FACTOR | 77% (10/13) | 80% (12/15) |

## How to interpret the delta CIs

Bootstrap: 1000 resamples at the per-pick-hit level (0/1 Bernoulli vector), drawn independently for the ablated and baseline cohorts since the pick lists can differ. The reported CI is the 2.5 / 97.5 percentile of `ablated_rate - baseline_rate`. Deltas whose CI excludes 0 are significant at the 5% level.

## Ground rules followed

- Single retraining of CausalWAR with `inning_bucket` masked. No architecture or hyperparameter changes.
- Same 2015-2022 train cohort as the committed baseline checkpoint; no resampling.
- Tag classifier is bit-identical to `scripts/causal_war_contrarian_stability.py`.
- Bootstrap n=1000, random_state=42.

## Artifacts

- `models/causal_war/causal_war_trainsplit_2015_2022_noRP.pkl`
- `results/causal_war/mechanism_ablation/ablated_buy_low_*.csv` (3 files)
- `results/causal_war/mechanism_ablation/ablated_over_valued_*.csv` (3 files)
- `results/causal_war/mechanism_ablation/ablation_comparison.json`
- `results/causal_war/mechanism_ablation/report.md` (this file)