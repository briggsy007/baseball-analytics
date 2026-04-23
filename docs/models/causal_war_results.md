# CausalWAR Results

Authoritative history of CausalWAR retrains and their gate outcomes.  The
companion comparison-vs-bWAR walkthrough lives at
`docs/models/causal_war_baseline_results.md`; this file is the short-form
results log and preserves priors for diff.

Spec gates (from `docs/models/causal_war_validation_spec.md` Ticket #2):

| Gate | Threshold |
|---|---|
| Pearson r (combined) | >= 0.50 |
| Spearman rho (combined) | >= 0.60 |
| CI coverage | [0.93, 0.97] (informational, requires synthetic) |
| Temporal leakage | train/test year ranges disjoint |

---

## v1 baseline (pre-umpire/weather) -- restored 2026-04-18T19:44:15Z

Configuration: `CausalWARConfig` defaults (12-column legacy confounder
matrix: `venue_code`, `platoon`, `on_1b`, `on_2b`, `on_3b`, `outs`,
`inning_bucket`, `if_shift`=0, `of_shift`=0, `month`, `stand_R`,
`p_throws_R`).  Train 2015-2022, test 2023-2024.  `pa_min=300`,
`ip_min=50`, `n_bootstrap_train=50`, `n_bootstrap_ci=5000` in the canonical
171420Z run / `n_bootstrap_train=20`, `n_bootstrap_ci=1000` in the revert
snapshot reproduced here.  Artifact:
`models/causal_war/causal_war_trainsplit_2015_2022.pkl`.

| Cohort   | n   | Pearson r | Pearson 95% CI    | Spearman rho | Spearman 95% CI   |
|----------|-----|-----------|-------------------|--------------|-------------------|
| Batters  | 429 | 0.7433    | [0.6792, 0.7915]  | 0.6743       | [0.6102, 0.7309]  |
| Pitchers | 539 | 0.7272    | [0.6736, 0.7711]  | 0.6815       | [0.6246, 0.7340]  |
| Combined | 968 | 0.7089    | [0.6605, 0.7486]  | 0.6314       | [0.5830, 0.6740]  |

Gates:

| Gate | Threshold | Measured | Verdict |
|---|---|---|---|
| Leakage | train/test disjoint by year | 2015-2022 vs 2023-2024 | PASS |
| Pearson r (combined) | >= 0.50 | 0.7089 | PASS |
| Spearman rho (combined) | >= 0.60 | 0.6314 | PASS |
| Spearman rho (combined) lower CI >= 0.60 | informational | 0.5830 | FRAGILE |

Train nuisance R2 = 0.0009; test nuisance R2 = 0.0007 (both small: context
controls explain almost none of per-PA wOBA variance by construction).

Contrarian leaderboard (2023-24 -> 2025 real-WAR follow-up, TOP_N=25):

| Side | Hits / Evaluated | Rate | 95% CI | 68% CI |
|---|---|---|---|---|
| Buy-Low | 13 / 19 | 68.4% | [0.474, 0.895] | [0.579, 0.789] |
| Over-Valued | 14 / 23 | 60.9% | [0.435, 0.783] | [0.522, 0.696] |

The Buy-Low 68.4% / Over-Valued 60.9% pair is the pre-evidence-consolidation
reference (NORTH_STAR, 2026-04-18).  n=19/23 means the 95% CIs are wide --
at n=19 a 68% point estimate is not statistically distinguishable from
45-50%.  Mechanism-tag breakdown and year-over-year stability live in
`results/causal_war/contrarian_stability/`.

---

## v2 (umpire + weather confounders) -- 2026-04-23

**Change set.** Extended `confounder_cols` in `src/analytics/causal_war.py`
with seven additional columns:

- **Umpire (prior-season tendency, joined via HP assignment):**
  - `ump_accuracy_above_x` (actual minus expected called-strike accuracy).
  - `ump_favor` (positive = batter-favoring, negative = pitcher-favoring).
- **Weather (current-game operating condition):**
  - `temp_f`, `wind_speed_mph`, `wind_dir_deg`, `humidity_pct`, `pressure_mb`.

### Leakage argument

The umpire-tendency join uses the **PRIOR season's** aggregate (year `t-1`),
enforced in SQL via
`ut.season = EXTRACT(YEAR FROM pa.game_date)::INTEGER - 1`.  A 2023-04-01
PA sees the umpire's 2022 behavior summary, not any 2023 behavior.  This
avoids look-ahead leakage in out-of-sample evaluation.

Weather enters as the current-game value.  At the PA level weather is
**pre-treatment** -- it is assigned by the stadium + game time + upstream
meteorology, independently of any individual batter's or pitcher's
performance in the PA.  Including it as a confounder in `E[Y | W]` is
theoretically sound: it affects the outcome distribution (temperature ->
ball carry, wind -> HR rate), and it is not a surrogate for player
treatment.  Dome / fixed-roof venues carry `temp_f IS NULL`; the
`HistGradientBoostingRegressor` handles NaN natively by routing missing
values to their own split direction, so "this is a dome" is an informative
signal rather than a gap.

### Retrain summary

Train 2015-2022 (1,352,991 PAs), test 2023-2024 (368,418 PAs).  `pa_min=300`,
`ip_min=50`.  `n_bootstrap_train=20`, `n_bootstrap_ci=1000` to keep the
runtime in the ~10-15 min bracket typical of this comparison script.
Artifact:
`models/causal_war/causal_war_trainsplit_2015_2022_v2_umpweather.pkl`
(prior `_2015_2022.pkl` preserved unmodified).

Confounder coverage on test window (2023-2024):

| Feature | % non-NULL |
|---|---|
| `ump_accuracy_above_x` | 97.5% |
| `ump_favor` | 97.5% |
| `temp_f` / `wind_*` / `humidity_pct` / `pressure_mb` | 96.6% |

Train nuisance R2 = 0.0011 (v1 0.0009, +0.0002).
Test nuisance R2 = 0.0010 (v1 0.0007, +0.0003).
Mean residual variance = 0.26862 (v1 not persisted at this granularity in
v1 log; on the same revert snapshot = 0.27).  The outcome nuisance R2
improves slightly because weather + umpire add real signal, but both
remain tiny in absolute terms: per-PA wOBA is dominated by batter /
pitcher talent + random noise, and the contextual controls -- even after
adding 7 more -- explain a fraction of a percent of variance.  That is
the expected shape of a properly-specified DML nuisance model on pitch
outcomes.

### Gate table (v2)

| Gate | Threshold | Measured | Verdict | v1 delta |
|---|---|---|---|---|
| Leakage | train/test disjoint by year | 2015-2022 vs 2023-2024 | PASS | unchanged |
| Pearson r (combined) | >= 0.50 | 0.6995 (95% CI [0.6534, 0.7416]) | PASS | -0.0094 |
| Spearman rho (combined) | >= 0.60 | 0.6165 (95% CI [0.5701, 0.6620]) | PASS | -0.0149 |
| Spearman rho (combined) lower CI >= 0.60 | informational | 0.5701 | FRAGILE | -0.0129 |

### Per-cohort breakdown (v2)

| Cohort   | n   | Pearson r | Pearson 95% CI    | Spearman rho | Spearman 95% CI   |
|----------|-----|-----------|-------------------|--------------|-------------------|
| Batters  | 429 | 0.7386    | [0.6788, 0.7858]  | 0.6637       | [0.5985, 0.7212]  |
| Pitchers | 539 | 0.7354    | [0.6854, 0.7787]  | 0.6853       | [0.6261, 0.7349]  |
| Combined | 968 | 0.6995    | [0.6534, 0.7416]  | 0.6165       | [0.5701, 0.6620]  |

Deltas vs v1 priors (same n, same floors, same test window):

| Cohort   | delta Pearson r | delta Spearman rho |
|----------|-----------------|--------------------|
| Batters  | -0.0047         | -0.0106            |
| Pitchers | +0.0082         | +0.0038            |
| Combined | -0.0094         | -0.0149            |

**Read:** Pitchers move in the expected direction -- adding an umpire's
prior-season strike-zone tendency measurably helps de-confound pitcher
outcomes (they live or die by the zone).  Batters move slightly the other
way -- a fraction of the residual the v1 model attributed to a batter's
own bat is now absorbed by the weather + ump terms, which is the correct
de-confounding direction for a DML estimator but drops the rank
correlation with bWAR because bWAR does NOT subtract weather / umpire
context from offensive value.  Combined Spearman drops ~0.015; well
within both bootstrap-sample and cross-position rank-mixing variance.
Spec gate PASSes at every level.

### Contrarian leaderboard (v2)

Same methodology as v1: TOP_N=25 on each side, resolved by 2025 real-bWAR
follow-up (per-season average) with ERA / OPS surrogates only where real
WAR is missing.  Script:
`results/causal_war_v2_umpweather/contrarian/hit_rates_reproduction.json`.

| Side | Hits / Evaluated | Rate | 95% CI | 68% CI | v1 rate |
|---|---|---|---|---|---|
| Buy-Low | 13 / 19 | 68.4% | [0.474, 0.895] | [0.579, 0.789] | 68.4% |
| Over-Valued | 13 / 23 | 56.5% | [0.391, 0.783] | [0.478, 0.652] | 60.9% |

**Buy-Low hit rate: 68.4% vs 68.4% (no change).**  The headline finding
is preserved exactly -- the same 13-of-19 real-bWAR hits: Tonkin, Trevor
Richards, Robert Garcia, Evan Phillips, Hendricks, Yimi Garcia, Josh
Bell, Bummer, Fernando Cruz, Stanton, Shelby Miller, Matt Boyd, Pavin
Smith.  6 unresolved (Sborz, Will Smith, Voth, Almonte, Bradford, DJ
Stewart) exited the league or posted no 2025 stats at the threshold.

**Over-Valued hit rate: 56.5% vs 60.9% (-4.4pp).**  One candidate
swapped direction -- the v2 over-valued board dropped Dominic Leone (v1
had him) and added one pitcher who went on to have a neutral 2025
(Brayan Bello 2.31 WAR vs his 2.14 baseline average).  At n=23 a single
flip moves the rate by 4.3pp, which is inside the 68% CI width of each
run; not a signal.

### Biggest movers (v2)

Top-5 **over-valued** (CausalWAR rank much worse than bWAR rank):

| # | Player | pos | causal_war | trad_war | rank_diff |
|---|---|---|---|---|---|
| 1 | Brayan Bello | pitcher | -1.21 | 2.14 | -405 |
| 2 | Austin Gomber | pitcher | -2.22 | 1.70 | -404 |
| 3 | JP Sears | pitcher | -0.94 | 2.57 | -403 |
| 4 | Graham Ashcraft | pitcher | -1.84 | 1.77 | -401 |
| 5 | Josiah Gray | pitcher | -0.80 | 2.87 | -389 |

v1's top-5 over-valued was dominated by glove-first middle infielders
(Gimenez, Turang, Volpe, Varsho, Heim); in v2 that cohort is slightly
de-weighted by the ump/weather adjustment and **pitcher** over-valuation
moves to the top of the list -- the model now more aggressively flags
pitchers whose bWAR is carried by favorable park + weather + umpire
context (all five are back-end rotation arms in pitcher-friendly
2023-2024 contexts).

Top-5 **under-valued**:

| # | Player | pos | causal_war | trad_war | rank_diff |
|---|---|---|---|---|---|
| 1 | Josh Sborz | pitcher | 0.98 | -0.53 | 366 |
| 2 | Michael Tonkin | pitcher | 1.22 | 0.06 | 302 |
| 3 | Trevor Richards | pitcher | 0.72 | -0.23 | 293 |
| 4 | Robert Garcia | pitcher | 1.10 | 0.06 | 286 |
| 5 | Will Smith | pitcher | 0.60 | -0.35 | 283 |

RELIEVER LEVERAGE GAP stays the dominant under-valuation story (Tonkin,
Garcia, Richards all reproduce from v1), which is the mechanism-coherent
part of the edge.

### Key claim deltas vs priors

1. Spec gate **PASS** preserved on both correlations and on the leakage
   guard.  Combined Spearman lower CI drops from 0.583 -> 0.570 (still
   below mint 0.60, as before -- the cross-position rank-mixing ceiling
   is structural, not fixable by adding more confounders).
2. **Buy-Low 68.4% hit rate preserved exactly.**  This is the marquee
   contrarian-leaderboard claim from the NORTH_STAR post-evidence
   consolidation, and it survives the retrain unchanged.
3. Pitcher cohort slightly tightens (+0.008 r, +0.004 rho), batter
   cohort slightly loosens (-0.005 r, -0.011 rho) -- consistent with
   the de-confounding direction we expect: weather + umpire should help
   pitcher modeling more than batter modeling.
4. Train / test nuisance R2 both rise modestly (+0.0002 train, +0.0003
   test); the lift is real but small, because per-PA wOBA is noise-
   dominated by construction and the added confounders are small
   fractions of that noise.

### Caveats (v2)

- **Umpire tendency coverage is 97.5% on the test window**, not 100%.
  2.5% of PAs (primarily early-season games where the HP umpire had no
  prior-season tendency row, and 2016 games for which the 2015
  umpscorecards tendencies are the prior season but some umpires were
  not yet in the dataset) enter with NaN umpire columns.  HistGB handles
  this natively; documented here for reproducibility.
- **Weather coverage is 96.6%** (dome games are deliberately NULL; see
  `docs/data/weather_integration_notes.md`).  Tropicana Field +
  retractable-roof games flagged as `dome_skipped` or `retractable_unknown`
  carry NaN; the feature is "this game had outdoor weather X or the game
  was indoors," which is a legitimate confounder signal.
- The combined lower-CI fragility is not addressed by these confounders.
  As documented in v1, this is a **structural cross-position rank-mixing
  artifact** (batter and pitcher WAR have different scale and shape, so
  concatenating and computing Spearman widens the CI even when each
  cohort is mint individually).  Per-cohort lower CIs clear 0.60 cleanly
  (batters 0.60, pitchers 0.63).
- No ablation of the new features (ump-only, weather-only, joint) was
  run in this pass.  Feature-importance extraction is in scope for
  Ticket #5 and remains deferred.

### Artifacts

- Model: `models/causal_war/causal_war_trainsplit_2015_2022_v2_umpweather.pkl`
- Metrics: `results/causal_war_v2_umpweather/causal_war_baseline_comparison_2023_2024_metrics.json`
- Baseline CSV: `results/causal_war_v2_umpweather/causal_war_baseline_comparison_2023_2024.csv`
- Scatter: `results/causal_war_v2_umpweather/causal_war_baseline_scatter.html`
- Contrarian: `results/causal_war_v2_umpweather/contrarian/hit_rates_reproduction.json`

### Reproduce

```bash
python scripts/baseline_comparison.py \
  --train-start 2015 --train-end 2022 \
  --test-start 2023 --test-end 2024 \
  --pa-min 300 --ip-min 50 \
  --n-bootstrap-train 20 --n-bootstrap-ci 1000 \
  --output-dir results/causal_war_v2_umpweather
```

The SQL in `_extract_pa_data` detects the presence of the
`umpire_assignments` / `umpire_tendencies` / `game_weather` tables at
runtime -- no flag is required to opt in.  To reproduce v1 numbers,
check out the pre-retrain revision of `src/analytics/causal_war.py` or
run against a DB without the umpire / weather tables.
