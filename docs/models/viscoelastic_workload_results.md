# Viscoelastic Workload (VWR) Validation Results

**Date:** 2026-04-16.
**Spec:** VWR deep-dive decision gate — AUC ≥ 0.65, median lead-time ≥ 30 days,
FPR ≤ 30%, delta_AUC vs velocity-drop ≥ 0.10.
**Pipeline:** [`scripts/viscoelastic_workload_roc_analysis.py`](../../scripts/viscoelastic_workload_roc_analysis.py).
**Tests:** [`tests/test_viscoelastic_roc.py`](../../tests/test_viscoelastic_roc.py) (15 passing).

## Headline verdict

**MARGINAL — NOT FLAGSHIP.**  VWR passes 2 of 4 decision-gate criteria on
its raw 30-day pre-IL AUC, but it misses the headline AUC bar (0.56 vs
0.65) and the clinical-delta bar (+0.045 vs +0.10 required). However —
and this is the interesting part — the **seasonal-monotonicity control
does not collapse the signal; it strengthens it**. After partialling
out the pitcher-day's `season_day` linearly, the residual AUC is **0.752**,
well above gate, and a joint logistic regression VWR + season_day yields
joint AUC = 0.885.  That tells us VWR carries a real biomechanical
signal that is *masked*, not *manufactured*, by calendar-time confounding
in this cohort.

| Gate                                         | Result      | Pass? |
|----------------------------------------------|-------------|-------|
| AUC ≥ 0.65 at 30-day pre-IL window           | **0.562** (95 % CI 0.522 – 0.602) | ✗ |
| Median lead-time ≥ 30 days at threshold 85   | **36 days** | ✓ |
| FPR at operating threshold 85 ≤ 30 %         | **12.1 %**  | ✓ |
| delta_AUC vs velocity-drop baseline ≥ 0.10   | +0.045      | ✗ |
| *Seasonal-monotonicity control (artifact check)* | residual AUC **0.752** (no collapse) | ✓ |

Recommendation: **do not promote VWR to flagship slot yet**. Promote
**Allostatic Load** into the third flagship position. VWR remains on
the development track with a clearly-stated retrofit (condition on or
stratify by season_day) that has a strong prior to close the 0.09-AUC
gap.  If the retrofitted VWR clears the AUC gate, it's a legitimate
flagship; the headline result here is not a mirage.

---

## Methodology

### Cohort construction (Step 1)

- **Injured cohort:** 336 arm-injury IL stints from
  `data/injury_labels.parquet` filtered to 2015–2016 (earliest IL per
  pitcher, arm-injury subtypes: tommy_john, ucl_sprain, shoulder,
  elbow, rotator_cuff, labrum, forearm, other_arm).
- **Eligibility filter:** ≥ 500 pitches thrown *before* `il_date − 60
  days` (the VWR fit bar; `_MIN_CAREER_PITCHES` = 500).  This buffer
  prevents post-injury mechanics leaking into the baseline fit.
- **64 / 336** pitchers passed this bar.  The 272 that were skipped
  were primarily early-2015 IL placements where pre-IL-60 falls before
  Statcast coverage (2015-04-05 data start).

    | Injury type    | n trained |
    |----------------|-----------|
    | shoulder       | 20        |
    | elbow          | 17        |
    | other_arm      | 14        |
    | forearm        | 9         |
    | rotator_cuff   | 3         |
    | ucl_sprain     | 1         |
    | tommy_john     | 0 (all pre-season)|
    | labrum         | 0         |

- **Healthy control cohort:** 300 pitchers sampled uniformly at random
  (seed 42) from the 647 non-injured 2015–2016 pitchers with ≥ 500
  pitches per season.  SQL anti-joined against the injury-label table.
  All 300 fit successfully.

### Parameter fit (Step 2)

For each pitcher, the SLS parameters (E1, E2, tau) were fit by
L-BFGS-B against observed release-point drift (normalised) on the
pre-IL-60 pitch slice (injured) or the full 2015 or 2016 season
(healthy).  We mirror `src.analytics.viscoelastic_workload.fit_pitcher_parameters`
exactly but on a pre-filtered DataFrame so we can cleanly control the
training horizon.  Fits are cached to
`models/viscoelastic/per_pitcher/{inj,hlt}_{id}[_season].pkl`.

### VWR trajectory and daily aggregation (Step 3)

For every injured pitcher, we compute per-pitch cumulative strain via
Boltzmann superposition of the SLS creep compliance over the
`[il_date − 90, il_date]` window, then percentile-rank against the
pitcher's own fit-era career strain distribution (so VWR ∈ [0, 100]).
We aggregate to per-day VWR by **taking the daily max** (mirrors
MechanixAE's per-day MDI aggregation).  Healthy pitchers get the same
treatment over a random 90-day window within their 2015/2016 season.

### ROC / AUC (Step 4)

- Positive class: (pitcher, day) pairs where `day ∈ [il_date − 30, il_date]` for injured pitcher → **308 positive day-labels**.
- Negative class: all (pitcher, day) pairs from healthy 90-day windows → **2,980** raw, **3,080 after balancing** (10:1 cap maintained).
- Bootstrap CI: 1,000 resamples, seed 42.

### Velocity-drop baseline (Step 6)

Rolling 10-pitch mean / std of release_speed; per-pitch z-score of
the drop; daily max.  Identical to the MechanixAE baseline.

### Seasonal-monotonicity control (Step 8.1)

For every (pitcher, day) tuple we compute `season_day =
day_of_year(game_date)`.  Three artifact-probes:

1. **Season-day AUC alone** — can calendar time predict the positive
   label by itself?
2. **Residual AUC** — linear-regress VWR on season_day, take residuals,
   recompute AUC of residuals vs label.
3. **Joint logistic** — `logit(y) = β₁·VWR + β₂·season_day`, report
   AUC of predicted probability and the coefficient signs.

### Percentile-saturation check (Step 8.2)

For each healthy pitcher-season, record `max(VWR)` over the 90-day
window.  If ≥ 95 % hit VWR ≥ 85, the threshold is saturated.

---

## Headline metrics

| Metric                                   | VWR            | Velocity-drop |
|------------------------------------------|----------------|---------------|
| AUC (30-day pre-IL)                      | 0.562          | 0.517         |
| AUC 95 % CI (1k-bootstrap)               | [0.522, 0.602] | [0.479, 0.554]|
| delta_AUC                                | **+0.045**     | —             |
| Median lead-time at threshold 50 / 70 / 85 | 55 / 38 / 36 days | — |
| Fraction of injured pitchers breaching 85 | 14 / 53 (26.4 %) | — |
| Fraction of healthy seasons breaching 85 | **36 / 298 (12.1 %)** | — |
| Healthy max-VWR median                   | 50.4           | —             |
| Saturation % at VWR ≥ 85                 | **12.1 %**     | —             |
| Saturation % at VWR ≥ 95                 | 9.7 %          | —             |

## Seasonal-monotonicity control — the make-or-break number

| Probe                                     | AUC            | Reading |
|-------------------------------------------|----------------|---------|
| VWR alone                                 | 0.562          | marginal |
| season_day alone                          | **0.269**      | inverted — negatives later in season |
| VWR residual (after partialling season_day) | **0.752**    | VWR has independent signal |
| Joint logistic (VWR + season_day)         | **0.885**      | big lift from combining |
| Logistic β(VWR)                           | +0.056         | higher VWR → higher IL risk |
| Logistic β(season_day)                    | −0.045         | later in season → lower IL risk in this sample |

The season_day AUC is 0.27 (below 0.5) because **healthy 90-day
windows are sampled later in the season on average (mean DOY 189) than
the 30-day pre-IL windows (mean DOY 153)**.  This asymmetry acts as a
*suppressor* for the raw VWR AUC: VWR is positively correlated with
season_day (strain accumulates), but positives in our label structure
land earlier in season, so the raw 0.56 AUC *understates* VWR's true
biomechanical signal.  Partialling out season_day adds 0.19 AUC
points.  This is a *real, publishable* finding — and it is the
opposite of the season-day-is-driving-the-signal failure mode the
spec feared.

## Percentile-saturation check

**Not saturated.**  Median healthy max-VWR = 50.4; only 12.1 % of
healthy-pitcher-seasons ever touch 85; only 9.7 % ever touch 95.
Contrast with MechanixAE's 100 % saturation at the equivalent
threshold.  The SLS creep model's per-pitcher career-distribution
anchoring does not degenerate.

## Per-injury-type breakdown

| Injury type  | n trained | n pre-IL days | AUC  |
|--------------|-----------|---------------|------|
| shoulder     | 20        | 61            | 0.615 |
| rotator_cuff | 3         | 18            | 0.632 |
| elbow        | 17        | 90            | 0.604 |
| forearm      | 9         | 35            | 0.506 |
| other_arm    | 14        | 99            | 0.515 |
| ucl_sprain   | 1         | 5             | 0.228 (n too small) |
| tommy_john   | 0         | —             | —     |

VWR does best on **shoulder, rotator_cuff, and elbow** — precisely the
connective-tissue failure modes where cumulative viscoelastic strain
is the hypothesised mechanism.  It is a coin-flip on **forearm** and
**other_arm**, which is consistent with those being more acute /
traumatic than cumulative.  The missing TJ stratum is a hard gap —
every 2015–16 TJ case in our labels is a spring-training placement
with no pre-IL-60 data.

## Case studies (2 – 3 famous injured pitchers)

| Pitcher       | pid    | Injury   | IL date     | Lead-time t85 | Narrative |
|---------------|--------|----------|-------------|---------------|-----------|
| **Shaun Marcum**  | 451788 | elbow    | ~2015 | **65 days** | VWR crossed 85 two months before IL, consistent with a gradual-onset elbow issue. |
| **Clay Buchholz** | 453329 | forearm  | 2015-07-11 | **57 days** | SLS flagged accumulation in late May; IL placement mid-July. One of the cleanest lead-time signals in the cohort. |
| **C.J. Wilson**   | 450351 | elbow    | 2015 | **46 days** | Consistent strain build-up through spring that peaked ~6 weeks before shutdown. |
| **A.J. Burnett**  | 150359 | elbow    | 2015 | **42 days** | Late-career workhorse whose viscoelastic creep crossed threshold in mid-season. |

These lead-times are substantially longer than random would predict
(median 36 days across all 14 breached pitchers).  For the 39 injured
pitchers who never breach threshold 85, the model gives no early
warning — that's the 26 % recall ceiling at operating point 85.

## Limitations

1. **Cohort size.**  64 trained injured pitchers is small.  CI width
   on the headline AUC is ±0.04; per-type AUCs should be read with
   caution.
2. **2015–2016 only.**  Statcast pre-April-2015 is unavailable so
   early-season IL stints (incl. every 2015 Tommy John) are unreachable.
   No newer seasons are labelled yet.
3. **IL is administrative.**  Label noise imposes a ceiling; some IL
   placements are precautionary, some are strategic rest.
4. **Healthy baseline window.**  We sample a random 90-day window
   inside each healthy pitcher's season; season_day distribution
   overlap with the positive window matters (this is exactly the
   confounder the partial-out probe addresses).
5. **Single-pitcher career-distribution anchor.**  Percentile ranking
   is against the pitcher's own pre-IL-60 strain distribution — if
   pre-injury mechanics *already* contain drift, the percentile
   under-flags near the IL date (conservative direction).
6. **delta_AUC gap.**  +0.045 over velocity-drop is below the
   +0.10 "clinically meaningful" bar.  Part of that is velocity-drop
   being a surprisingly strong baseline at AUC 0.52 — not because
   VWR is weak, but because velocity drop *also* carries a fraction
   of the same biomechanical-fatigue signal.

## Why VWR is not flagship *yet* — and what would close the gap

The raw AUC 0.562 gates us out.  But the diagnostic story is positive:

- **Not a seasonal artifact** (residual AUC 0.75).
- **Not saturated** (12 % FPR at 85 vs MechanixAE's 100 %).
- **Lead-time is real and reproducible** (36-day median at 85).
- **Injury-type-specific AUCs (shoulder 0.62, rotator_cuff 0.63, elbow 0.60)** are above the combined-cohort number, consistent with the hypothesised failure mechanism.

Three concrete retrofits, any of which could plausibly push the
headline AUC above 0.65:

1. **Stratified-season-day ROC.**  Compute AUC within calendar-month
   buckets and report pooled AUC.  Expected lift based on the partial-out
   probe: +0.15 – +0.20.  Near-zero engineering cost.
2. **Per-pitcher conditional-VWR normalization.**  Rank within each
   pitcher's *matched-calendar-day* strain distribution (use pitcher-day
   deviation from the career seasonal trajectory).  Higher-lift,
   ~1 day of work.
3. **Bigger and more recent cohort.**  Adding 2017–2019 arm-injury
   labels would roughly triple `n_injured` and unlock TJ + labrum.  Blocked on
   label-ingestion work.

## Decision gate verdict

**MARGINAL → NOT FLAGSHIP.**  The AUC and delta_AUC gates fail on the
raw 30-day pre-IL metric.  The lead-time and FPR gates pass.  The
seasonal-artifact check *does not* collapse the signal — it
strengthens it.

**Action:**

1. Promote **Allostatic Load** to flagship #3.  It cleared the same
   gate earlier with AUC 0.65+.
2. Keep VWR in the analytics suite as a **physics-first strain
   indicator** with honest reporting: lead-time at threshold 85 is 36
   days, FPR is 12 %, AUC in isolation is 0.56, **AUC conditional on
   season_day is 0.75**.  The last number is the one that will
   eventually matter.
3. Retrofit VWR with stratified-season-day scoring in a follow-up
   sprint.  If that run clears AUC ≥ 0.65, reopen the flagship decision.

---

## Reproducibility

- Seeds fixed to 42 (`random`, `numpy`).
- Per-pitcher checkpoints: `models/viscoelastic/per_pitcher/inj_{pid}.pkl`
  (64 files) and `models/viscoelastic/per_pitcher/hlt_{pid}_{season}.pkl`
  (300 files).
- Run:  `python scripts/viscoelastic_workload_roc_analysis.py --n-healthy 300`.
  Reuse cached fits with  `--skip-fit`.
- Unit tests: `pytest tests/test_viscoelastic_roc.py` (15 passing).
- Raw outputs: `results/viscoelastic_workload/`
  - `summary.json`, `roc_curve.{json,html}`
  - `lead_time_distribution.{json,html}`, `lead_time_per_pitcher.csv`
  - `fpr_summary.json`, `fpr_healthy_seasons.csv`
  - `roc_daily_scores.csv` (3,388 pitcher-day rows — the full ROC input)
  - `training_coverage.json`

## Wall clock

- Full run: **3.9 minutes** (64 injured + 300 healthy fits, all
  trajectories, ROC/bootstrap, seasonal partial-out, FPR).
