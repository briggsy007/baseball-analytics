# Allostatic Load (ABL) Validation Results

**Date:** 2026-04-18.
**Spec:** ABL validation gate -- AUC >= 0.60, median lead-time >= 14 days,
FPR <= 30%, delta_AUC vs games-played baseline >= 0.05, no seasonal artifact.
**Pipeline:** [`scripts/allostatic_load_roc_analysis.py`](../../scripts/allostatic_load_roc_analysis.py).
**Spec doc:** [`docs/models/allostatic_load_validation_spec.md`](allostatic_load_validation_spec.md).

## Headline verdict

**NOT FLAGSHIP.**  ABL passes 1 of 4 hard decision-gate criteria on its
raw 30-day pre-IL AUC.  It clears the lead-time gate by a wide margin
(42-day median) but **fails catastrophically on the FPR gate (77.5%)**
and falls short on AUC (0.581 vs 0.60), delta_AUC vs the games-played
baseline (+0.014 vs +0.05), and the seasonal-monotonicity residual
(0.560, just barely above the 0.55 floor for "no artifact").  The
fundamental issue: the ABL score is **per-season z-score normalised**,
so every healthy season eventually hits 100 (saturation 77.5% at
threshold 75; **median healthy max-ABL is 100.0**).  Threshold-based
alerts are therefore non-discriminative as currently specified.

| Gate                                                  | Result                          | Pass? |
|-------------------------------------------------------|---------------------------------|-------|
| AUC >= 0.60 at 30-day pre-IL window                   | **0.581** (95% CI 0.559 - 0.603) | FAIL |
| Median lead-time >= 14 days at threshold ABL=75       | **42 days** (27/31 batters breach) | PASS |
| FPR at operating threshold ABL=75 <= 30 %             | **77.5 %**                      | FAIL |
| delta_AUC vs games-played baseline >= 0.05            | **+0.014**                      | FAIL |
| *(informational)* Seasonal-monotonicity residual AUC  | **0.560** (no full collapse)    | borderline |

Recommendation: **do not promote ABL to flagship slot**.  The model has
a real but small biomechanical-load signal (per-injury-token AUCs of
0.66-0.72 for wrist, quad, groin, knee), but the current scoring
formulation (per-season z-score rescaled to 0-100) makes operating
thresholds meaningless on the healthy population.  Two concrete
retrofits could rescue it: (1) anchor ABL percentiles to a
cross-season pooled baseline distribution (so 75 means something
absolute), and (2) report ABL relative to a per-batter career mean
(deviation rather than level).

---

## Methodology

### Cohort construction (Step 1)

- **Injured cohort:** 1,498 raw `non_arm` IL stints from
  `data/injury_labels.parquet` (2015-2016, dedup on player + earliest IL).
  After eligibility filtering (>=15 distinct games as a batter before
  `il_date - 14 days` in the IL season), **31 batters retained**.
  The 1,467 dropped were primarily minor-league transactions or players
  who hit IL early-season before accruing a meaningful pre-IL workload.

- **Healthy control cohort:** 200 batter-seasons sampled uniformly at
  random (seed 42) from 636 candidates with >=60 games in 2015 OR 2016
  AND no IL placement of any type, any year, in the labels parquet.
  All 200 fit successfully.

The "pitcher_id" column in `injury_labels.parquet` is misleadingly
named -- it actually stores MLB player IDs.  We use these as
`batter_id` for the hitter-side ABL pipeline.

### ABL trajectory (Step 2)

For each batter, we call the production `calculate_abl()` function in
`src/analytics/allostatic_load.py` over the entire IL season (or
season of interest for healthy controls).  The function:

1. Pulls per-game stressor totals (pitches seen, borderline pitches,
   swings, schedule density, travel) from the `pitches` table.
2. Applies a leaky integrator per channel
   (`L_t = alpha * L_{t-1} + stressor_t`, `alpha=0.85`).
3. Z-scores each channel across the season.
4. Sums positive z-scores to a "raw composite" then **rescales to
   0-100 by dividing by the season max**.

That last step is the key methodological choice that drives the FPR
result -- by construction, the season max is always 100 for every
batter who plays a full season.

### ROC / AUC (Step 3)

- Positive class: per-game ABL where `game_date in [il_date - 30 days,
  il_date]` for an injured batter -> **154 positive (player, day)
  rows**.
- Negative class: per-game ABL from healthy 90-day random windows
  (seed 42) -> **17,061 raw**, capped at 10:1 to **1,540 sampled**.
- Trapezoid AUC, 1,000-bootstrap 95% CI, seed 42.

### Games-played (workload) baseline (Step 4)

For each (batter, game_date) tuple we count games in the trailing 30
days and scale to 0-100 (max=30 games).  Run the identical ROC pipeline
and report `delta_AUC = AUC(ABL) - AUC(workload)`.

### Seasonal-monotonicity control (Step 5)

Mirrors VWR exactly: linear-regress ABL on `season_day`, recompute AUC
of residuals; joint logistic with both predictors.  If residual AUC
collapses below 0.55, flag a calendar-time artifact.

### Per-injury-token breakdown (Step 6)

Tokenize `injury_description_raw` against a fixed list of body-part
strings (oblique, hamstring, knee, wrist, etc).  Report per-token AUC
where >=5 positive day-labels exist.

---

## Headline metrics

| Metric                                     | ABL              | Games-played baseline |
|--------------------------------------------|------------------|------------------------|
| AUC (30-day pre-IL)                        | 0.581            | 0.567                 |
| AUC 95% CI (1k-bootstrap)                  | [0.559, 0.603]   | not reported          |
| delta_AUC                                  | **+0.014**       | --                    |
| Median lead-time at threshold 60 / 75 / 85 | 44 / **42** / 38 days | --              |
| Fraction of injured batters breaching 75   | 27 / 31 (87.1 %) | --                    |
| Fraction of healthy seasons breaching 75   | **155 / 200 (77.5 %)** | --              |
| Healthy max-ABL median                     | **100.0**        | --                    |
| Saturation % at ABL >= 75                  | 77.5 %           | --                    |
| Saturation % at ABL >= 85                  | 69.5 %           | --                    |

## Seasonal-monotonicity control

| Probe                                       | AUC   | Reading |
|---------------------------------------------|-------|---------|
| ABL alone                                   | 0.581 | weak |
| season_day alone                            | 0.289 | inverted -- negatives sampled later in season |
| ABL residual (after partialling season_day) | 0.560 | small lift; model survives the artifact check |
| Joint logistic (ABL + season_day)           | 0.716 | season_day is the dominant predictor here |
| Logistic beta(ABL)                          | +0.0073 | higher ABL -> slightly higher IL risk |
| Logistic beta(season_day)                   | -0.022 | later in season -> lower IL risk in this sample |

The story is similar to VWR: positives sample earlier in season than
negatives (mean DOY ~ 150 vs ~ 200), so `season_day` has a strong
**inverted** standalone AUC.  Unlike VWR though -- where partialling
out season_day **boosted** the residual AUC to 0.75 -- partialling out
season_day for ABL only nudges it from 0.581 down to 0.560.  The
biomechanical signal is real but small; calendar time is doing
relatively little of the lifting.

## Per-injury-token breakdown

| Body part   | n positive day-labels | AUC   |
|-------------|------------------------|-------|
| **wrist**   | 27                     | **0.719** |
| **groin**   | 25                     | **0.714** |
| **quad**    | 25                     | **0.665** |
| **knee**    | 25                     | 0.647 |
| **thumb**   | 106                    | 0.620 |
| **back**    | 60                     | 0.618 |
| **hamstring** | 152                  | 0.596 |
| **concussion** | 40                  | 0.583 |
| **foot**    | 73                     | 0.566 |
| unspecified | 101                    | 0.434 |
| **ankle**   | 15                     | 0.232 (n too small / acute) |

ABL does best on **wrist, groin, quad, and knee** -- soft-tissue,
overuse-flavoured failure modes where cumulative cognitive +
biomechanical load is plausibly causal.  It is anti-predictive on
**ankle** (n=15, likely acute / traumatic) and on the
**unspecified** bucket (residual transactions noise).  These per-token
numbers suggest the underlying signal is real but the pooled headline
AUC is dragged down by label noise.

## Saturation -- the make-or-break problem

`calculate_abl()` rescales the per-season composite by the season max
to fit a 0-100 display range.  Consequently:

- **100% of full-season healthy batters reach ABL = 100** at some
  point in their season.
- **77.5% of healthy seasons** hit ABL >= 75, and **69.5%** hit ABL
  >= 85.

Compare to VWR's 12.1% saturation at threshold 85.  ABL's
operating-threshold-based alerts are essentially non-discriminative on
healthy data.  This is **NOT** a flaw in the leaky-integrator
mechanics -- it is a flaw in how the composite is presented.  The raw
sum-of-positive-z-scores carries discriminative information (residual
AUC 0.56), but the per-season normalisation washes it out at any
fixed numeric threshold.

## Limitations

1. **Cohort size (CRITICAL).**  31 trained injured batters is small.
   Bootstrap CI on the headline AUC is +/- 0.02; per-injury AUCs at
   n < 30 should be read with caution.
2. **2015-2016 only.**  Position-player Statcast coverage and
   injury-label fidelity are stable in this window; newer seasons are
   either unlabelled or have different IL classification rules.
3. **Label noise (CRITICAL).**  `non_arm` is a residual bucket
   bundling oblique strains with paternity leave, illness, and freak
   collisions.  Hard cap on AUC ceiling; wrist 0.72 vs unspecified
   0.43 is the clearest evidence.
4. **Per-season z-score rescaling.**  Detailed above -- this is the
   model's biggest validation-time pain point.  Fixing it requires
   either a cross-season pool or per-batter-career anchoring.
5. **Healthy 90-day window sampling.**  Random within-season window
   per healthy player; could shift the season_day distribution and
   change the seasonal-control numbers.  Not run with multiple seeds
   here.
6. **Causal direction.**  ABL captures cognitive + decision load.
   Many position-player IL stints are physical (oblique, hamstring,
   acute trauma) where the load -> injury arrow is longer and noisier
   than the connective-tissue cumulative-strain story for VWR.

## Why ABL is not flagship -- and what would close the gap

The diagnostics are mixed but mostly pessimistic:

- AUC 0.58 below the 0.60 spec gate (close, not catastrophic).
- delta_AUC vs trivial games-played baseline only +0.014 -- the
  five-channel machinery barely beats a count of recent games.
- Lead-time signal is real (median 42 days, 87 % of injured batters
  cross threshold 75).
- FPR catastrophe (77.5 %) traces directly to per-season rescaling.
- Per-injury sub-AUCs of 0.65-0.72 on overuse-style failure modes
  suggest a real but specific signal.

Three concrete retrofits, in priority order:

1. **Cross-season pooled-percentile anchoring.**  Compute strain
   percentiles against a stable reference distribution (e.g. all
   batter-game ABL composite values from the prior season) instead of
   per-season max rescaling.  Plausibly drops healthy saturation from
   77.5% toward 10-15% and unlocks a usable operating threshold.
   ~1 day of work.
2. **Stratified-season-day ROC.**  Compute AUC within calendar-month
   buckets and pool.  Same idea as VWR's stratified probe.  Expected
   lift +0.05 to +0.10 AUC.  ~half day.
3. **Bigger and more recent cohort.**  Adding 2017-2019 non_arm
   labels would roughly triple `n_injured` and likely tighten the
   per-token AUCs.  Blocked on label-ingestion work.

## Decision gate verdict

**NOT FLAGSHIP.**  Three of four hard gates fail (AUC, FPR,
delta_AUC).  Lead-time passes by a wide margin.  Seasonal artifact
check is borderline (residual AUC 0.560, just above the 0.55 floor
that would have flagged "calendar artifact").

**Action:**

1. Keep ABL in the analytics suite as a **decision-fatigue indicator**
   with the honest dashboard caveat: the 0-100 score is per-season
   relative, not absolute, so threshold-based alerts are unreliable
   without retrofit (1) above.
2. Do **NOT** promote ABL to flagship slot 3.  Keep MechanixAE
   demoted; treat the third flagship slot as open pending the
   pooled-percentile retrofit run on either ABL or a recovered VWR
   variant.
3. Schedule the cross-season anchoring retrofit as a follow-up
   sprint.  If the retrofitted ABL reaches AUC >= 0.60 AND FPR <= 30%,
   reopen the flagship decision.

---

## Reproducibility

- Seeds fixed to 42 (`random`, `numpy`).
- Per-batter checkpoints: `models/allostatic_load/per_batter/inj_{pid}_{season}.pkl`
  (31 files) and `models/allostatic_load/per_batter/hlt_{pid}_{season}.pkl`
  (200 files).
- Run: `python scripts/allostatic_load_roc_analysis.py --n-healthy 200`.
  Reuse cached trajectories on rerun (cache hit by default).
- Raw outputs: `results/allostatic_load/`
  - `summary.json`, `roc_curve.{json,html}`
  - `lead_time_distribution.{json,html}`, `lead_time_per_pitcher.csv`
  - `fpr_summary.json`, `fpr_healthy_seasons.csv`
  - `roc_daily_scores.csv` (1,694 rows = the full ROC input)
  - `training_coverage.json`

## Wall clock

- Full run: **1.7 minutes** (31 injured + 200 healthy ABL trajectories,
  ROC + bootstrap, seasonal partial-out, FPR, per-token AUCs).
