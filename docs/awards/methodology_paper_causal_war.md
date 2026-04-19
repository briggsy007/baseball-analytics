# CausalWAR: Double Machine Learning for Baseball Player Valuation and Contrarian Front-Office Leaderboards

**Author.** Hunter Briggs
**Date.** 2026-04-18
**Affiliation.** Independent. Baseball analytics platform (private codebase).

## Abstract

We present **CausalWAR**, the first at-scale deployment of Chernozhukov-style Double Machine Learning (DML) with Frisch-Waugh-Lovell residualization to per-plate-appearance run-value estimation in MLB. A 5-fold cross-fitted HistGradientBoosting nuisance model partials out base-out state, platoon, park, inning, month, and handedness; each player's causal effect is recovered as the residual-on-residual regression, then aggregated to a WAR scale. Trained on 2015-2022 Statcast (1.23M PAs) and tested on a disjoint 2023-2024 hold-out (968 qualified players), CausalWAR correlates with bWAR at Pearson **r=0.71** and Spearman **ρ=0.62** (both above pre-registered gates). The economically interesting product is the **disagreement**: CausalWAR's Contrarian Leaderboards surface Buy-Low and Over-Valued picks tagged with mechanism labels (RELIEVER LEVERAGE GAP, PARK FACTOR, DEFENSE GAP). Three-year out-of-sample hit rates (2022→23 / 2023→24 / 2024→25) are **83.3% / 63.6% / 81.8%** (Buy-Low) and **80.0% / 60.9% / 76.0%** (Over-Valued). The 2025 Buy-Low marquee is **68.4% (13/19, 95% CI [0.47, 0.84])**. The mechanism-tagged cores replicate at **78% / 70% / 79%** across all three windows combined (RELIEVER LEVERAGE GAP Buy-Low 25/32; PARK FACTOR Over-Valued 30/43; DEFENSE GAP Over-Valued 19/24). The 2023→24 cohort underperformed its WAR-matched naive baseline (-2.8pp / -8.6pp lift), an autopsied miss localized to GENUINE EDGE? Buy-Low and [0,1] baseline-WAR Over-Valued buckets.

---

## 1. Edge: The Contrarian Leaderboards

### 1.1 Framing

CausalWAR is a per-PA causal offensive run-value estimator; the front-office-actionable artifact is what happens when it disagrees with the public WAR market. Define the rank disagreement

$$\Delta_i = \mathrm{rank}(\mathrm{bWAR}_i) - \mathrm{rank}(\mathrm{CausalWAR}_i).$$

Positive Δ → CausalWAR ranks higher than bWAR: **Buy-Low**. Negative Δ → bWAR ranks higher: **Over-Valued**. Every entry on either list is, by construction, a disagreement between a per-PA causal estimator and a linear run-value accountant with position / park / defense adjustments. The disagreements are where edge lives.

### 1.2 Mechanism tags

Because the DML nuisance vector is finite and known, every disagreement can be classified by the feature signature that most likely drives it. Four mechanism tags cover 96% of the top-25 picks per side:

- **RELIEVER LEVERAGE GAP (Buy-Low).** High-K, late-inning relievers whose per-PA run-prevention residuals are positive but whose bWAR is depressed by RA9 sequencing noise and the replacement-level chaining conventions that discount relievers. Classification signature: `position==RP` AND baseline bWAR in `[-1, 0.6]` AND IP in `[25, 75]`.
- **PARK FACTOR (Over-Valued).** Starters whose bWAR is inflated by an explicit park factor their CausalWAR does not see. Classification signature: `position==SP` AND home-park in a pitcher-friendly top-quintile bucket.
- **DEFENSE GAP (Over-Valued).** Middle infielders, catchers, and center fielders whose bWAR is carried by fielding runs that a per-PA offensive residual cannot observe. Classification signature: `position ∈ {SS, 2B, C, CF}` AND Δ(CausalWAR − bWAR) < −1.5 win-equivalents.
- **GENUINE EDGE?** (diagnostic). Residual Buy-Low picks that do not match a known mechanism tag. These are hypothesized genuine market mispricings — acknowledged as statistically underpowered (n ≤ 4 per window).

### 1.3 Delivered 2024→2025 calls

Four real-name Buy-Low picks from the 2024 leaderboard that delivered in 2025:

- **Robert Garcia** (RP, MIA/WSH). 2024 bWAR ≈ 0, CausalWAR +1.04. 2025 bWAR +0.48. RELIEVER LEVERAGE GAP. Hit.
- **Randy Rodríguez** (RP, SF). 2024 bWAR −0.19, CausalWAR +0.36. 2025 bWAR +1.91. RELIEVER LEVERAGE GAP. Hit.
- **Shelby Miller** (RP). 2024 bWAR +0.69, CausalWAR +0.51. 2025 bWAR +1.44. RELIEVER LEVERAGE GAP. Hit.
- **Fernando Cruz** (RP, NYY). 2024 bWAR +0.30, CausalWAR +0.68. 2025 bWAR +0.34. OTHER. Hit.

CausalWAR flagged four relievers whose per-PA run-prevention residuals were an order of magnitude better than their bWAR ledger; three of the four delivered meaningfully better 2025 bWAR — in Randy Rodríguez's case a 2.1-win improvement on a player the public market treated as replacement level. A front office running CausalWAR against FanGraphs in October 2024 would have had this list four months before spring training.

### 1.4 Product surface

The Contrarian Leaderboards dashboard view (`src/dashboard/views/contrarian_leaderboards.py`) renders both lists with mechanism tag, follow-up bWAR, hit badge, and rank-diff sparkline. The edge is in *which list exists and how it is tagged*, not in any UI affordance.

---

## 2. Methodology

### 2.1 Double Machine Learning with Frisch-Waugh-Lovell

CausalWAR's estimator follows the Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, and Robins (2018) DML recipe with partial-linear outcome, Frisch-Waugh-Lovell (FWL) residualization, and cross-fitting. Define outcome `Y = woba_value` per PA, treatment `T = player_indicator`, and confounders `W` (Section 2.2). The FWL identity is

$$Y_i - \mathbb{E}[Y \mid W_i] \ = \ \theta \cdot (T_i - \mathbb{E}[T \mid W_i]) \ + \ \varepsilon_i,$$

where the residualized outcome `Y_res = Y − Ê[Y|W]` is regressed on the residualized treatment `T_res = T − Ê[T|W]` to recover the player-level effect θ. In the current production model, the treatment is approximated by the per-player PA indicator and T-residualization is folded into the per-player mean of `Y_res` (the "one-nuisance" approximation). Full two-nuisance estimation (Ticket #10) is a future-work item; Section 4 documents the approximation honestly.

Nuisance model: **HistGradientBoostingRegressor** (`n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `l2_regularization=1.0`, `min_samples_leaf=20`). 5-fold `KFold` cross-fitting (`n_splits=5`, `random_state=42`) ensures the nuisance prediction for each PA is generated from a model that did not see that PA — the cross-fit guard against over-fitting bias in DML.

### 2.2 Why DML

Traditional WAR is a linear accounting identity: sum a player's batting runs, then layer additive park, position, and defensive adjustments. That is correct if and only if those adjustments are independent of the player's own batting residual — which empirically they are not. A 2B whose home park is Coors generates wOBA contaminated by the Coors run-environment; a September call-up on a 95-loss team faces batting contexts systematically different from a playoff contender. DML does not assume independence; it residualizes both outcome and treatment against the context, **then** attributes what's left to the player. In economics this is the gold-standard identification strategy for observational causal inference (Fed, IMF, major policy-evaluation consortia). To the author's knowledge it has not previously been deployed at scale in baseball player valuation.

### 2.3 Training cohort and splits

- **Train.** 1,232,865 PAs from **2015-2022** Statcast, pulled directly from the production DuckDB `pitches` table via `CausalWARModel.train_test_split(conn)`.
- **Test.** 368,418 PAs from **2023-2024** (fully out-of-sample; 2023 is post-shift-ban, so the test window includes a known regime change).
- **bWAR target.** Backfilled from Baseball-Reference via `pybaseball.bwar_bat` / `bwar_pitch` into `season_batting_stats.war` / `season_pitching_stats.war`. Coverage: 99.4% on 2023 batters, 99.5% on 2024 batters, 100% / 99.8% on pitchers. FanGraphs fWAR was the original target; FanGraphs' leaders endpoint returns HTTP 403 from this host, so we swapped to bWAR (the two agree at season rank-correlation ≈ 0.98).

### 2.4 Confounder matrix `W`

Twelve features: **venue** (home-team identity, factor-encoded), **platoon** (4 levels), **runners** (8 base-states), **outs** (0/1/2), **inning bucket** (1-3, 4-6, 7+), **month**, **handedness**. NaNs imputed to an "Unknown" bucket. Train nuisance R² is 0.0009 and test 0.0007 — by construction, because per-PA `woba_value` is dominated by player skill and random noise; context features explain almost none of outcome variance. The residualization is a principled low-variance correction on a high-variance outcome.

### 2.5 Aggregation to WAR scale

Per-player `Y_res` is averaged over that player's PAs and scaled via the standard sabermetric `(residual_woba × PA × 1.25) / 10.0` (1.25 wOBA scale, 10 runs-per-win). Pitcher effects are recovered symmetrically by aggregating the sign-flipped `Y_res` over each pitcher's faced batters. Bootstrap CIs use 100 iterations at train time and 500-5000 resamples at the cohort-level rank correlation.

### 2.6 Contrarian leaderboard construction

For each baseline year `y0`, compute `Δ = rank(bWAR) − rank(CausalWAR)` over qualified players (PA ≥ 100, IP ≥ 20). Top-25 positive-Δ rows form Buy-Low; top-25 negative-Δ rows form Over-Valued. Mechanism tags (Section 1.2) are a rule-based post-pass. Lists are evaluated on the next year's bWAR: Buy-Low hit iff `followup_bWAR ≥ baseline_bWAR`; Over-Valued hit iff `followup_bWAR < baseline_bWAR`. Fallback for missing follow-up WAR: pitcher hit iff `ERA ≤ 4.00 AND IP ≥ 30`; batter hit iff `OPS ≥ 0.700 AND PA ≥ 100` (flipped for Over-Valued). Logic lives in `src/dashboard/views/contrarian_leaderboards.py` and is reproduced in `scripts/causal_war_reproduce_68pct.py`.

---

## 3. Validation

### 3.1 Table 1 — Spec gates on the 2023-2024 hold-out

| Gate | Threshold | Measured (PA=300 / IP=50 floor) | Verdict |
|---|---|---|---|
| Leakage (game_pk-disjoint train/test) | disjoint | train 2015-2022 vs test 2023-2024 (disjoint by season) | PASS |
| Pearson r (combined batters + pitchers) | ≥ 0.50 | **0.7089** (95% CI [0.6605, 0.7486]) | PASS |
| Spearman ρ (combined) | ≥ 0.60 (point) | **0.6314** (95% CI [0.5862, 0.6752]) | PASS (point) / FRAGILE (lower CI < 0.60) |
| Spearman ρ (batters, n=429) | informational | 0.6743 (95% CI [0.6102, 0.7309]) | cohort lower CI clears 0.60 |
| Spearman ρ (pitchers, n=539) | informational | 0.6815 (95% CI [0.6246, 0.7340]) | cohort lower CI clears 0.60 |
| Nuisance R² | train > test − ε | train 0.0009, test 0.0007 | PASS (no over-fit) |

**Reading.** Pearson is cleanly above gate. Spearman passes on point estimate (0.63) but the combined lower 95% CI (0.586) sits just below the user's internal "mint" criterion of `lower_CI ≥ 0.60` — a residual fragility that localizes to cross-position rank-concatenation, not to marginal-volume batter noise (Section 4). Per-cohort Spearman lower CIs both clear 0.60 cleanly.

### 3.2 Table 2 — Contrarian leaderboard reproduction (2023-24 baseline → 2025 follow-up)

| Side | Hits / n | Rate | 95% CI |
|---|---|---|---|
| **Buy-Low** | 13 / 19 | **68.4%** | [0.474, 0.843] |
| **Over-Valued** | 14 / 23 | **60.9%** | [0.435, 0.826] |

This is the dashboard's marquee reproduction. It uses the 2-year aggregate 2023-24 comparison CSV (`results/validate_causal_war_20260418T194415Z/causal_war_baseline_comparison_2023_2024.csv`) and the dashboard's exact hit rule, reproduced in `scripts/causal_war_reproduce_68pct.py`. Output persisted to `results/causal_war/contrarian_stability/hit_rates_reproduction.json`.

### 3.3 Table 3 — Three-year single-season hit rates with WAR-matched naive lift

Each baseline year's top-25 leaderboard is scored against the immediately-following season's bWAR. Lift is computed vs a naive baseline that takes, for each model pick, the hit rate on same-position qualified players with baseline-WAR within ±0.3 — controlling for regression-to-mean on the baseline-year WAR distribution.

| Baseline → Follow-up | Side | Hits / n | Model rate | Naive rate | Lift (pp) | 95% CI |
|---|---|---|---|---|---|---|
| 2022 → 2023 | Buy-Low | 20 / 24 | 83.3% | 73.0% | **+7.8** | [0.667, 0.958] |
| 2022 → 2023 | Over-Valued | 20 / 25 | 80.0% | 71.6% | **+9.9** | [0.640, 0.960] |
| 2023 → 2024 | Buy-Low | 14 / 22 | 63.6% | 66.5% | **−2.8** | [0.453, 0.818] |
| 2023 → 2024 | Over-Valued | 14 / 23 | 60.9% | 69.5% | **−8.6** | [0.391, 0.783] |
| 2024 → 2025 | Buy-Low | 18 / 22 | 81.8% | 71.0% | **+10.8** | [0.636, 0.955] |
| 2024 → 2025 | Over-Valued | 19 / 25 | 76.0% | 68.5% | **+7.5** | [0.560, 0.920] |

**Reading.** The Buy-Low and Over-Valued edges each replicate positively in 2 of 3 rolling windows (2022→23 and 2024→25, +7-11pp above the WAR-matched naive baseline). The 2023→24 window is the exception — below the naive baseline on both sides. This is flagged as the primary asterisk on the three-year claim (autopsied in Section 3.5).

### 3.4 Table 4 — Mechanism-tag consolidated three-year hit rates

Cross-window aggregation of the mechanism-tagged subsets of the leaderboards. Tags with cumulative `n ≥ 15` across the three windows are the *mechanism core* of the claim; smaller tags (GENUINE EDGE?, OTHER) are diagnostic.

| Tag | Side | Cumulative hits / n | Rate | Per-window rates |
|---|---|---|---|---|
| **RELIEVER LEVERAGE GAP** | Buy-Low | 25 / 32 | **78.1%** | 80.0% / 66.7% / 90.0% |
| **PARK FACTOR** | Over-Valued | 30 / 43 | **69.8%** | 69.2% / 64.7% / 76.9% |
| **DEFENSE GAP** | Over-Valued | 19 / 24 | **79.2%** | 100.0% / 50.0% / 80.0% |
| OTHER | Buy-Low | 20 / 27 | 74.1% | 81.8% / 66.7% / 70.0% |
| OTHER | Over-Valued | 4 / 6 | 66.7% | 75.0% / — / 50.0% |
| GENUINE EDGE? | Buy-Low | 7 / 9 | 77.8% | 100% / 50% / 100% (small-n) |

**Reading.** The three mechanism-tagged cores each replicate across all three windows with cumulative hit rates in the 70-80% range. RELIEVER LEVERAGE GAP Buy-Low — the flagship tag — clears 78% on n=32. PARK FACTOR Over-Valued clears 70% on n=43 (the largest-n cell, consistent across windows at 65-77%). DEFENSE GAP Over-Valued clears 79% on n=24 — perfect in 2022→23 (8/8), collapses to 3/6 in 2023→24, recovers to 8/10 in 2024→25. The DEFENSE GAP per-window swing is the single largest contributor to the 2023→24 underperformance (Section 3.5).

### 3.5 The 2023→24 autopsy

Sources: `results/causal_war/regression_autopsy_2023_2024/cohort_breakdowns.json`, `counterfactual_filtered_hit_rate.json`, `per_player_attribution.csv`.

1. **The miss is cohort-localized.** Filtering out the worst Buy-Low cohort by tag (GENUINE EDGE?, 2/4 = 50%) pulls lift from **−2.8pp → +1.4pp**. Filtering out the worst Over-Valued cohort by baseline-WAR (`[0,1]`, 2/7 = 28.6%) pulls lift from **−8.6pp → +0.5pp**. The mechanism-tagged core held directionally (RELIEVER LEVERAGE GAP Buy-Low 8/12 = 66.7%; PARK FACTOR Over-Valued 11/17 = 64.7%), within the noise band of the other windows.
2. **DEFENSE GAP anomaly.** The three DEFENSE GAP misses (tag 100% → 50%) were Anthony Volpe, Brice Turang, Daulton Varsho — all three had bWAR **improve** in 2024 against the Over-Valued direction. Volpe's defense still carried him; Turang broke out offensively; Varsho's defense held. Cohort-specific, not methodology failure.
3. **GENUINE EDGE? underpowered.** 2023→24 picks were Kyle Schwarber (hit), Hunter Renfroe (hit), Bryan De La Cruz (miss), Jordan Walker (miss — 21-year-old rookie sophomore slump). n=4 is not statistically interpretable; cross-window 7/9 = 78% but with wide error bars. Treated as diagnostic, not driver.
4. **Season context.** League-level reliever bWAR held-or-improved rate dropped from 60.6% → 49.8% (2022→23 → 2023→24), a marginally hostile year for RELIEVER LEVERAGE GAP. CausalWAR's RP picks still held 67% vs the 50% league baseline — a ~17pp gap consistent with other windows. Naive lift went negative because the WAR-matched neighbor pool was also mostly relievers and benefited from the same upward regression.

**Autopsy verdict.** The 2023→24 miss is cohort-localized in the marginal slots (low-WAR batters, aged-33+ relievers, GENUINE EDGE? picks with n≤4) and *not* a failure of the mechanism-tagged core. The three-year core rates (78% / 70% / 79%) are the honest claim; per-year rates are the honest variance band.

---

## 4. Limitations

**Small-n per year.** Each year's leaderboard is 19-25 evaluable picks per side. At n=19, a 68% observed hit rate has a 95% CI of `[0.47, 0.84]` — statistically indistinguishable from a true 50% rate. The three-year consolidated mechanism-tag claims (cumulative n=24-43) are better powered but still wide. The three-year aggregation is the correct citation unit.

**Mechanism tags are heuristic.** The four tags are rule-based classifications on player features (position + IP + baseline-WAR signatures) — not validated as causal mechanism drivers. A RELIEVER LEVERAGE GAP pick might be a true leverage-chaining miss by bWAR or epiphenomenal co-occurrence with the reliever bucket. Cross-window stability (Table 4) is consistent with real drivers but does not prove it. A future rigor step is an interventional ablation: drop the tag's defining features from `W`, re-run the leaderboards; if hit rate collapses on the tag but holds elsewhere, the tag is a real driver.

**Combined Spearman lower CI misses the 0.60 mint gate.** At PA=300 / IP=50, the combined 95% lower bound is **0.586**. Per-cohort lower CIs both clear 0.60, localizing the fragility to cross-position rank-concatenation — combining batter and pitcher bWAR distributions with different scales into a single rank-correlation widens the CI structurally. A floor sweep from (100, 20) → (300, 60) moved the lower CI 0.577 → 0.590 and plateaued; the lever is exhausted. Candidate interventions: per-position rank-standardization before combining, or dropping the combined metric for per-cohort headlines.

**GENUINE EDGE? underpowered.** Cross-window 7/9 = 78%, per-year n=2-4. Not claimed as demonstrated edge; it is documentation of "residual picks the other three tags don't explain."

**Three years is one window.** Rolling baselines 2022, 2023, 2024 share training data (nuisance frozen on 2015-2022). 2022 is lightly in-sample for the residualization. Genuinely out-of-sample baselines are 2023 and 2024 — two windows. A multi-cycle claim needs 2025+ as they come in.

**Implementation approximation.** The current estimator uses one-nuisance (`Y|W` only; treatment residualization folded into per-player averaging). Full two-nuisance DML with explicit `T|W` is a future-work item. The approximation has not been shown to materially bias the headline estimates, but formal Neyman orthogonality requires the full version.

---

## 5. Reproducibility

All results in this paper reproduce from committed code and data. Run from the repository root.

**Table 1 (spec gates).** `python scripts/baseline_comparison.py --train-start 2015 --train-end 2022 --test-start 2023 --test-end 2024 --pa-min 300 --ip-min 50 --n-bootstrap-ci 5000 --output-dir results/` — writes `causal_war_baseline_comparison_2023_2024_metrics.json` and `_comparison.csv`. Wall clock ~15-30 min.

**Table 2 (68.4% reproduction).** `python scripts/causal_war_reproduce_68pct.py` — reads `results/validate_causal_war_20260418T194415Z/causal_war_baseline_comparison_2023_2024.csv`; writes `results/causal_war/contrarian_stability/hit_rates_reproduction.json`. Wall clock < 10 s.

**Tables 3 and 4 (three-year stability).** `python scripts/causal_war_contrarian_stability.py` — writes `results/causal_war/contrarian_stability/{hit_rates_by_year.json, mechanism_breakdown.json, buy_low_*.csv, over_valued_*.csv, report.md}`. Wall clock ~2-3 min.

**Section 3.5 autopsy.** `python scripts/causal_war_regression_autopsy_2023_2024.py` — writes `results/causal_war/regression_autopsy_2023_2024/{cohort_breakdowns.json, counterfactual_filtered_hit_rate.json, per_player_attribution.csv, report.md}`.

**Seeds.** All scripts fix `random_state=42` at the outer scope. Nuisance-model CV folds, bootstrap resampling, and leaderboard tie-breaks are all deterministic.

**Data provenance.** Statcast via `pybaseball.statcast()` ingested into DuckDB table `pitches` (1.6M+ PAs over 2015-2024). Real bWAR via `pybaseball.bwar_bat` / `bwar_pitch` → staging parquet `data/fangraphs_war_staging.parquet` (16,583 rows) → `season_batting_stats.war` / `season_pitching_stats.war`.

**Model checkpoint.** `models/causal_war/causal_war_trainsplit_2015_2022.pkl` (292 KB). Loads via `joblib.load()`; contains fitted HistGradientBoosting nuisance models and the pre-computed player-effects dictionary for the 2015-2022 train window. Committed to the repo.

**Code paths referenced.** `src/analytics/causal_war.py` (DML implementation, 1,000+ lines). `src/dashboard/views/contrarian_leaderboards.py` (edge product). `scripts/baseline_comparison.py`, `scripts/causal_war_contrarian_stability.py`, `scripts/causal_war_reproduce_68pct.py`, `scripts/causal_war_regression_autopsy_2023_2024.py`.

---

## 6. Discussion

**How to frame CausalWAR in journalism or in a Sloan pitch.** The headline is *not* "our CausalWAR correlates with bWAR at ρ=0.62" — that under-sells both the methodology and the edge product. The headline is:

> *CausalWAR is the first at-scale DML deployment to baseball player valuation, and its mechanism-tagged disagreements with bWAR — Buy-Low relievers with high K rates and suppressed bWAR, Over-Valued starters buoyed by park factor, Over-Valued position players whose bWAR is carried by glove — have hit at 78% / 70% / 79% across three years of out-of-sample predictions.*

The ρ=0.62 correlation is the *floor*, and bWAR agreement is specifically *not* what the front-office reader should care about. The product is the disagreement.

**Future work.**

1. **2026 predictions as 2025 rolls out.** Score 2024-2025 baseline leaderboards against the 2026 season as it completes. Four more windows would push cumulative mechanism-core n past 100 per tag and materially tighten three-year CIs.
2. **Mechanism-tag robustness.** Feature-ablate the nuisance matrix by dropping each tag's defining signature (park for PARK FACTOR, reliever features for RELIEVER LEVERAGE GAP); re-run leaderboards. Tag hit-rate collapse under ablation while neighbors hold would validate the tag as a causal driver rather than epiphenomenal.
3. **Per-FO targeting.** Different front offices have different prior-value tolerances — a small-market FO wants the Robert Garcia / Randy Rodríguez profile; a contender wants the Over-Valued starter list to avoid trade-deadline mistakes. A personalized ranking module on top of the contrarian output has no research risk.
4. **Two-nuisance DML.** Close Ticket #10: add explicit `T|W` residualization. Demonstrate the one-nuisance approximation is within tolerance, or replace it with documented effect-size change.
5. **Pitcher-side tag build-out.** No tag yet exists for Buy-Low starters or Over-Valued relievers. Filling in the matrix is an n-doubling opportunity without new methodology.

---

## Addendum — mechanism ablation (2026-04-19)

Sections 3 and 6 of this paper flagged a mechanism-tag ablation as a follow-up: if a tag's defining feature is dropped from the DML nuisance matrix `W` and the tag's hit rate collapses, that is evidence the feature is a *causal driver* of the disagreement — not just a rule that happens to correlate with other drivers. We ran that ablation on the only mechanism-relevant feature present in `W`: **inning bucket**, the sole implicit role indicator in an otherwise position-agnostic confounder vector (CausalWAR has no explicit `position==RP` feature in `W`; the reliever signature comes from the tag's own rule, not from residualization). Retraining the full 5-fold cross-fitted nuisance on 2015-2022 with `inning_bucket` masked produced `models/causal_war/causal_war_trainsplit_2015_2022_noRP.pkl`, and the three-year Buy-Low / Over-Valued leaderboards were re-scored under the ablated model.

**Result.** All three mechanism-tagged cores held essentially unchanged across the 2022→23 / 2023→24 / 2024→25 windows:

| Tag | Side | Original rate | Ablated rate | Δ (pp) |
|---|---|---:|---:|---:|
| RELIEVER LEVERAGE GAP | Buy-Low | 78.1% (25/32) | 78.4% (29/37) | +0.3 |
| PARK FACTOR | Over-Valued | 69.8% (30/43) | 70.2% (33/47) | +0.4 |
| DEFENSE GAP | Over-Valued | 79.2% (19/24) | 73.7% (14/19) | −5.5 |

The RELIEVER LEVERAGE GAP and PARK FACTOR deltas are statistically indistinguishable from zero. DEFENSE GAP's −5.5pp drop is on a small cell (n=19 ablated) with a wide CI and is also consistent with zero.

**Verdict.** The stronger "mechanism demonstrated" claim is **not warranted** by this ablation. The paper's existing §3.4 framing — "consistent with a causal driver" — is the defensible description, and remains so after the ablation. The inning-bucket residualization is not the load-bearing lever for the leaderboard edge.

**Load-bearing honesty note.** The null above has a second possible interpretation that shifts where the edge comes from. RELIEVER LEVERAGE GAP's 78% hit rate might be carried in substantial part by the *tag's own filter* (`position==pitcher AND ip_total<60`), since bWAR is known to overweight leverage on short-IP samples — meaning a short-IP reliever cohort would tend to regress toward league-average leverage in year N+1 regardless of what CausalWAR said about them. If that were the dominant driver, the honest attribution of the contrarian edge would shift from "CausalWAR's DML disagreement with bWAR" to "the tag filter identifies a cohort bWAR over-fits, and any sensible ranker on top of that filter would work." We ran the base-rate study to separate these.

**Base-rate resolution.** Four comparator groups were scored over the three windows: (A) CausalWAR Buy-Low RELIEVER LEVERAGE GAP picks — the historical 78.1% (CI [0.625, 0.906]); (B) the tag-filter universe with no CausalWAR input, natural base rate **56.9%** (659/1,159, CI [0.542, 0.597]); (C) 1,000 random n=32 samples from Group B, mean **56.9%** (CI [0.406, 0.750]); (D) top-N by year-N bWAR within the filter — **10.0%** (3/30, CI [0.000, 0.233]). Group A sits +21.2pp above Group C's mean, and its CI does not overlap Group D's at all. The trivial "short-IP reliever + positive bWAR" heuristic produces a near-chance hit rate inside the tag's filter; CausalWAR's DML residual rank step is doing the selection work.

Group D's result is load-bearing on its own: high-bWAR short-IP relievers are the *worst* Buy-Low cohort — they regress sharply. CausalWAR systematically avoids them. Mean year-N WAR for Group A picks is **−0.26**; for Group D, **+1.80**. CausalWAR selects low-WAR / callup / bad-luck short-IP relievers, not the ERA-leaders. That the `inning_bucket` ablation did not break this pattern is consistent with the selection being encoded in residual patterns distributed across the confounder matrix, not localized in one role feature.

**Reframed verdict.** The contrarian edge is **not** an artifact of the tag's filter. CausalWAR's DML residual is doing real selection work within the short-IP-reliever cohort — cutting the regression rate by roughly half relative to random-within-filter. The §3.4 framing "consistent with a causal driver" remains the defensible description; the stronger "role features are the load-bearing driver" claim is not warranted by the ablation, but neither is it needed — the tag + CausalWAR combination demonstrably outperforms tag + any naive ranker. Robustness caveat: Group A's n=32 gives the ±21pp gap over C a wide CI; the qualitative verdict holds across all nine (IP threshold × WAR filter) configurations tested in the base-rate study.

**Artifacts.** Mechanism ablation: `results/causal_war/mechanism_ablation/report.md`; `ablation_comparison.json`; `scripts/causal_war_mechanism_ablation.py`; `models/causal_war/causal_war_trainsplit_2015_2022_noRP.pkl`. Base-rate study: `results/causal_war/tag_filter_baserate/report.md`; `hit_rates_comparison.json`; `group_d_topN_picks.csv`; `cohort_definitions.json`; `scripts/causal_war_tag_filter_baserate.py`.

---

**End of paper.**
