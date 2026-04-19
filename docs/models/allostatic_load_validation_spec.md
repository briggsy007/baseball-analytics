# Allostatic Load (ABL) Validation Spec

**Source:** `src/analytics/allostatic_load.py` (`class AllostaticLoadModel` at line 684; config `class ABLConfig` at line 71).

## A. Current state

**Architecture.** Multi-channel cumulative decision-fatigue model for hitters.
Five independent leaky-integrator channels:

1. Pitch Processing Load (pitches seen per game)
2. Decision Conflict Load (borderline pitches near zone edge)
3. Swing Exertion (total swings per game)
4. Temporal Demand (game density in trailing 7 days)
5. Travel Stress (consecutive away games / home-away switches)

Each channel is a leaky integrator `L(d) = alpha * L(d-1) + stressor(d)` with
`alpha = 0.85` (game-day) and `alpha^2` decay on off-days.  Channel loads are
z-scored across the season; the composite ABL is the sum of positive z-scores
scaled to 0–100.  **Higher ABL = more cumulative cognitive/physical load.**

**Implemented.**
- Per-batter, per-season composite ABL with full per-game timeline.
- Five channel decompositions and recovery-day projection helper.
- Internal "convergent validity" check vs O-Swing% and Z-Contact% per
  game (`validate_against_outcomes`).
- Dashboard: timeline, gauges, fatigue-vs-performance scatter, rest
  optimisation, leaderboard (`src/dashboard/views/allostatic_load.py`).

**Tested.** Unit tests on the leaky integrator, channel computation,
batch leaderboard.  No injury-label external validation prior to this
spec.

**Not implemented before this spec.**
- ROC / AUC against IL placements.
- Lead-time analysis.
- FPR audit on healthy player-seasons.
- Comparison vs a naïve "games-played" or "pitches-seen" workload baseline.
- Seasonal-monotonicity / season-day artifact control.
- Per-injury-type breakdown.

## B. Decision gates

This spec mirrors the validation surface of the Viscoelastic Workload
(VWR) deep-dive — the canonical pitcher-load template
(`docs/models/viscoelastic_workload_results.md`).  Allostatic Load is the
hitter analogue of that pipeline: cumulative load → injury early warning.

The gates are **calibrated to the smaller hitter cohort and the
weaker label channel** (position-player IL placements are less arm-
specific and more heterogeneous than pitcher arm injuries).  Each
threshold is documented with rationale.

| Gate | Threshold | Operator | Rationale |
|------|-----------|----------|-----------|
| **AUC** at 30-day pre-IL window      | 0.60 | ≥ | Hitter-injury labels are noisier than pitcher arm injuries (DL placements bundle hand/wrist/leg/oblique/illness). The 0.05 reduction vs VWR's 0.65 reflects the label-noise penalty; still well above chance and clinically actionable. |
| **Median lead-time** at the operating threshold | 14 days | ≥ | Hitter cumulative load builds and recovers faster than viscoelastic pitcher strain; two weeks of advance warning is enough to schedule a rest day or DH-only game and still beat reactive IL placement. |
| **FPR** at the operating threshold (% of healthy seasons that breach) | 30 % | ≤ | Same false-positive ceiling as VWR — anything more frequent than 1-in-3 healthy players false-flagging makes the alert non-actionable for a manager. |
| **delta_AUC** vs games-played (workload) baseline | 0.05 | ≥ | The ABL machinery (leaky integration, multi-channel z-scoring, off-day decay) must beat a brain-dead "games in trailing 30 days" count.  +0.05 AUC is the "earns its complexity" floor; +0.10 would be flagship-strong. |
| *(informational)* Seasonal-monotonicity residual AUC | n/a (no collapse) | qual. | If partialling out `season_day` collapses ABL AUC toward 0.5, the headline is a calendar-time artifact.  We require the residual-AUC probe to *not* collapse — i.e. residual AUC stays above 0.55. |

**Operating threshold:** ABL ≥ 75 (out of 100).  The dashboard expander
calls 60+ "high load" but 75+ is the upper third of the season-scaled
range and the most defensible point for an alert (matches the VWR
operating threshold of 85th percentile).

**Verdict bands:**

- **FLAGSHIP** — all four hard gates pass AND no seasonal artifact.
- **MARGINAL** — at least two of {AUC, lead-time, FPR} pass; flagship
  re-eligibility blocked on the failing gate.
- **NOT FLAGSHIP** — fewer than two pass, or seasonal artifact collapses
  the residual AUC.

## C. Methodology requirements

### C.1 Cohort construction

- **Injured cohort.** From `data/injury_labels.parquet`, take rows
  where `injury_type == "non_arm"` AND `il_date in [2015-01-01,
  2016-12-31]`.  These are the position-player DL placements.
  Deduplicate on `pitcher_id` (the column actually stores MLB
  player IDs, used as `batter_id` for hitter analyses), keeping the
  earliest IL stint.
- **Eligibility filter.** ≥ **15 distinct games as a batter** in the
  season of the IL stint AND **≥ 14 days of pitches BEFORE the IL
  date** (the FIT_BUFFER for hitters; analogous to VWR's 60-day
  buffer but compressed because hitter fatigue cycles are shorter).
- **Healthy control cohort.** From the universe of batters with ≥ 60
  distinct games in 2015 OR 2016, anti-join the union of all IDs
  appearing anywhere in `injury_labels.parquet`.  Sample 200 at random
  (seed 42).

### C.2 ABL computation

- For each injured batter, compute ABL on **all season pitches** via
  `calculate_abl()` (the production function in
  `src/analytics/allostatic_load.py`).  Do NOT reimplement.
- Extract the per-game `composite_abl` series from `result["timeline"]`.
- Re-key into `(player_id, game_date, abl)` tuples for downstream ROC.
- For each healthy batter-season, compute ABL the same way over the
  full season.

### C.3 ROC / AUC at 30-day pre-IL window

- Positive class: per-game ABL values where
  `game_date in [il_date - 30 days, il_date]` for an injured batter.
- Negative class: per-game ABL values from healthy batter-seasons,
  taking a randomly-selected 90-day window per season (seed 42).
- 10:1 cap on negatives via random down-sampling to control class
  imbalance.
- AUC computed with the trapezoid rule on the full ROC curve.
- 95 % CI from 1,000 bootstrap resamples (seed 42).

### C.4 Games-played (workload) baseline

For each (player, game_date) tuple in the ROC frame, compute a naive
workload score:

```
workload_score = number of games played in trailing 30 days
```

Run the identical ROC pipeline and report `delta_AUC = AUC(ABL) - AUC(workload)`.

### C.5 Lead-time analysis

For each injured batter, find the first day in the 90-day pre-IL
window where ABL ≥ 75; record `(il_date - first_breach_date).days`.
Report median, mean, p25, p75 across the cohort.  Repeat at
thresholds 60 and 85 for sensitivity context.

### C.6 FPR on healthy seasons

For each healthy batter-season, record `max(ABL)` over a random 90-day
window AND the fraction of days at which ABL ≥ {60, 75, 85}.  Headline
metric: **percentage of healthy seasons that ever breach ABL = 75**.

### C.7 Seasonal-monotonicity control

Mirror the VWR pipeline exactly (`partial_out_auc()` in
`scripts/viscoelastic_workload_roc_analysis.py`):

1. AUC of `season_day` alone.
2. Linear-regress ABL on `season_day`; residual AUC.
3. Joint logistic `logit(y) = beta1*ABL + beta2*season_day`.

If residual AUC < 0.55 OR (raw AUC − residual AUC) > 0.08, flag
"seasonal artifact" in the verdict.

### C.8 Per-injury-type breakdown

Even though we filter to `non_arm`, the `injury_description_raw`
field contains body-part hints (oblique, hamstring, hand, knee,
illness).  Tokenize the raw description and report per-tokenized
sub-injury AUC where n ≥ 5 positive day-labels — no hard gate, just
sanity check whether ABL is universally weak or whether it shines on
specific failure modes (e.g. oblique strains, where chronic over-
swinging is plausibly causal).

### C.9 Reproducibility

- Seeds fixed to 42.
- Read-only DuckDB.
- One-command run:
  `python scripts/allostatic_load_roc_analysis.py --n-healthy 200`.
- Outputs land in `results/allostatic_load/` with the same artifact
  set the validate-model skill expects (see Step 5 of skill).

## D. Risk flags

- **Label noise (HIGH).**  `non_arm` is a residual bucket; lots of
  unrelated DL placements (illness, paternity leave, freak collisions)
  contaminate the positive class.  Hard cap on AUC ceiling.
- **Cohort size (MEDIUM).**  ~30 eligible injured batters with the
  proposed eligibility filter.  Bootstrap CIs will be wide
  (likely ±0.07).  Per-injury breakdowns will mostly be n < 5.
- **Two-season window (MEDIUM).**  2015–2016 only because Statcast
  data and labels are stable there.  Cross-validation across additional
  seasons is a follow-up.
- **Causality direction (MEDIUM).**  ABL captures cumulative *cognitive*
  load (decision conflict, pitch processing).  Many hitter injuries
  are *physical* (oblique, hamstring) — the load → injury arrow is
  longer and noisier than for pitchers.
- **Off-season decay normalization (LOW).**  ABL z-scores within a
  season; cross-season comparison is not meaningful.  This script
  evaluates within-season trajectories only.
- **The "pitcher_id" column name in the labels file is misleading**
  — it stores MLB player IDs, used here as `batter_id`.  Fixed in
  the script's documentation.

## E. Award-readiness checklist

- ROC AUC ≥ 0.60 at 30-day pre-IL window with bootstrap CI.
- Median lead-time ≥ 14 days at the operating threshold.
- FPR on healthy seasons ≤ 30 % at the operating threshold.
- Beats games-played baseline by ≥ 0.05 AUC.
- Seasonal-monotonicity residual AUC ≥ 0.55 (no calendar artifact).
- Reproducible run script with deterministic seeds.
- Honest per-injury-type breakdown.
- Limitations section in `docs/models/allostatic_load_results.md`.

**Disqualifiers.** Saturation (>50 % of healthy seasons breach the
operating threshold).  AUC < 0.55 (below "barely better than a coin").
delta_AUC < 0 vs the games-played baseline.  Seasonal-artifact
collapse.

---

Status: Specified 2026-04-18.  First validation run scheduled to land
in `results/validate_allostatic_load_<UTC-timestamp>/`.
