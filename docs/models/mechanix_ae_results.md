# MechanixAE Validation Results

**Date:** 2026-04-16 (Section 1 initial; Section 2 added same day post-rescue attempt).
**Spec:** [`mechanix_ae_validation_spec.md`](mechanix_ae_validation_spec.md).
**Pipeline:** [`scripts/mechanix_ae_roc_analysis.py`](../../scripts/mechanix_ae_roc_analysis.py).
**Tests:** [`tests/test_mechanix_roc_analysis.py`](../../tests/test_mechanix_roc_analysis.py) (23 passing).

## Headline verdict

**PERMANENT PIVOT.** Both scoring formulations — the original percentile-rank
MDI and the rescue-attempt magnitude-based z-score — deliver AUC well below
random on the 111-injured / 50-healthy arm-injury cohort. The z-score redesign
did not cross the 0.58 rescue gate; it did not even cross the 0.50 coin-flip
line. MechanixAE is demoted from flagship. The descriptive "mechanical
profiling" framing is retained for the dashboard, but MechanixAE is no longer
positioned as an injury early-warning system. Promote **Allostatic Load** and
**Viscoelastic Workload** into the flagship slot vacated here.

| Scoring formulation | AUC   | 95% CI         | Delta vs vel-drop | Gate     |
|---------------------|-------|----------------|-------------------|----------|
| Percentile (legacy) | 0.379 | [0.352, 0.408] | +0.109            | **PIVOT** |
| Z-score (rescue)    | 0.389 | [0.361, 0.417] | +0.118            | **PIVOT** |
| Gate for rescue     | ≥ 0.58 | —              | —                 | —        |
| Gate for flagship   | ≥ 0.65 | —              | —                 | —        |

The +0.010 AUC lift from percentile → z-score is well inside the bootstrap
noise floor and does not change the qualitative picture.

---

## Section 1 — First attempt: percentile-rank MDI

### Cohort

- Labels: `data/injury_labels.parquet`, 2015–2016 window (the only seasons
  currently populated with 30-day pre-IL pitch data).
- 336 unique pitchers with an arm injury (earliest IL placement per pitcher).
- 111 pitchers have ≥ 200 healthy pitches (game_date < il_date − 30) and were
  trained; 92 of those also had ≥ 20 pre-IL pitches for the lead-time analysis.
- Per injury type trained: shoulder 37, elbow 30, other_arm 19, forearm 16,
  rotator_cuff 5, tommy_john 3, ucl_sprain 1, labrum 0.
- Training wall: 3.4 min for injured cohort, plus ~3 min for 50 healthy baselines.

### Lead-time distribution (percentile, threshold 60/70/80)

| Threshold | n breached / 92 | Median (days) | Mean | IQR  |
|-----------|-----------------|---------------|------|------|
| 60        | 92 (100 %)      | 63.5          | 58.6 | 40–82 |
| 70        | 92 (100 %)      | 63.0          | 58.0 | 39–81 |
| 80        | 92 (100 %)      | 61.5          | 56.4 | 34–81 |

Reads impressive in isolation, but collapses under FPR: healthy pitcher-seasons
breach these thresholds at essentially the same rate.

### ROC / AUC

| Detector                         | AUC   | 95 % CI        |
|----------------------------------|-------|----------------|
| MDI (percentile)                 | 0.379 | [0.352, 0.408] |
| Velocity-drop (−2 σ rolling 10)  | 0.270 | [0.247, 0.298] |
| delta_AUC (MDI − VelDrop)        | +0.109 | —             |

Both curves sit below the diagonal.

### FPR on healthy seasons (50 seasons ≥ 1000 pitches)

| Threshold | Pct breached at any point |
|-----------|---------------------------|
| MDI ≥ 60  | 100.0 %                   |
| MDI ≥ 70  | 100.0 %                   |
| MDI ≥ 80  | 100.0 %                   |

Catastrophic FPR — percentile-rank MDI saturates at the top of every series
by construction.

### Root-cause diagnosis (pointed to in Section 2)

1. **MDI is a rank, not a magnitude.** By definition, at least one window in
   any evaluation set scores at the top percentile. Any pitcher evaluated long
   enough *will* breach MDI = 100.
2. **Baseline asymmetry.** Injured pitchers' MDI is ranked against their own
   healthy pre-IL-30 windows; healthy pitchers' MDI is ranked against their
   own earlier 60% of the season. Different baseline definitions → different
   score distributions.
3. **Pitch-type entanglement (FLAG 3).** The 10-D feature space mixes fastballs
   and breaking balls; per-pitcher × pitch-type mean-centering does not remove
   within-pitch-type variance.

Hypothesis going into Section 2: (1) is the dominant failure mode. Swap the
percentile rank for a z-score normalised against the pitcher's own healthy
baseline mean/std, and AUC should lift above 0.58.

---

## Section 2 — MDI redesign: magnitude-based z-score

### Implementation approach

- **Stats computation:** recomputed from training data at inference time (we
  already have `baseline_errors` cached per pitcher in the ROC pipeline).
  Training code was also updated to persist `healthy_recon_mean` /
  `healthy_recon_std` in new checkpoints (`src/analytics/mechanix_ae.py::
  train_mechanix_ae`), but existing checkpoints are back-compatible because
  the rescue pipeline recomputes stats on the fly.
- **API:** `calculate_mdi(..., score_mode="percentile" | "zscore")`. Default
  `"percentile"` so existing dashboard wiring doesn't break.
- **Thresholds:** z = 1 / 2 / 3 (1σ / 2σ / 3σ above healthy mean) replace the
  60 / 70 / 80 percentile buckets.
- **Script flag:** `python scripts/mechanix_ae_roc_analysis.py
  --score-mode zscore --skip-train`. Outputs go to
  `results/mechanix_ae/{roc_curve,lead_time_distribution,fpr_summary,summary}_zscore.{json,csv,html}`;
  the original percentile artifacts are preserved at `*_percentile.*`.
- **Cohort, checkpoints, seeds, 30-day window:** identical to Section 1. Only
  the scoring function changes.

### Lead-time distribution (z-score, threshold 1 / 2 / 3 σ)

| Threshold | n breached / 92 | Median (days) | Mean | IQR     |
|-----------|-----------------|---------------|------|---------|
| z ≥ 1σ    | 89 (96.7 %)     | 58.0          | 54.2 | 35–77   |
| z ≥ 2σ    | 82 (89.1 %)     | 48.0          | 47.2 | 32–68   |
| z ≥ 3σ    | 59 (64.1 %)     | 43.0          | 40.8 | 19.5–61.5 |

Lead-time looks cleaner than percentile mode — not every pitcher saturates —
but the discrimination against healthy baseline (below) kills any predictive
claim.

### ROC / AUC (z-score)

| Detector                         | AUC   | 95 % CI        |
|----------------------------------|-------|----------------|
| MDI (z-score)                    | 0.389 | [0.361, 0.417] |
| Velocity-drop (−2 σ rolling 10)  | 0.272 | [0.247, 0.298] |
| delta_AUC (MDI − VelDrop)        | +0.118 | —             |

Still below 0.50. The z-score variant earns a +0.010 AUC lift over percentile —
well inside the 95 % CI width of ~0.056 and therefore statistically
indistinguishable.

### FPR on healthy seasons (z-score thresholds)

| Threshold | Pct breached at any point |
|-----------|---------------------------|
| z ≥ 1σ    | 100.0 %                   |
| z ≥ 2σ    | **96.0 %**                |
| z ≥ 3σ    | 88.0 %                    |

Even at 3σ (a stringent bar), 88% of healthy pitcher-seasons eventually trip
the alarm. The scoring function change did not fix the underlying
identifiability problem.

### Decision gate verdict

**PERMANENT PIVOT.** AUC(MDI_zscore) = 0.389, 95% CI [0.361, 0.417]. Gate for
rescue was AUC ≥ 0.58; actual result is 0.19 AUC points below that, and
*below* the 0.50 random baseline. Combined with 96% FPR at z ≥ 2σ, there is
no viable operating point where MechanixAE beats "flip a coin."

### Why the rescue failed (honest narrative read)

The percentile-saturation theory (root-cause #1 in Section 1) turned out to
be *not the dominant* failure mode. The z-score redesign addressed #1 cleanly
— FPR dropped from 100% to 96–88%, and the lead-time distribution is no
longer pinned — yet AUC barely moved. The residual failure is almost
certainly a combination of #2 (baseline asymmetry — injured and healthy
cohorts are scored against structurally different baseline distributions)
and #3 (pitch-type entanglement — the 10-D latent space tracks mix shifts,
not injury signatures).

Fixing those would require:

- Pitch-type conditioning (Ticket 8: one-hot type into the VAE so latent
  space is type-conditional). ≥ 1 day of retrain work.
- A rigorously symmetric baseline — train healthy and injured cohort VAEs on
  identical horizon structures. Requires re-scoping the evaluation protocol.

Neither path has a strong prior of pushing AUC over 0.58, and both cost
multiples of the score-redesign effort. With Allostatic Load and
Viscoelastic Workload already showing flagship-worthy numbers, the right
move is to reallocate MechanixAE's narrative slot rather than keep pouring
engineering into it.

## Recommended path forward

1. **Demote MechanixAE to "mechanical profiling" descriptive tool.** Keep the
   dashboard view; drop all injury early-warning language. The VAE's latent
   space is still a useful lens for "this pitcher's mechanics shifted in
   start #42" colour commentary, just not as a predictor.
2. **Promote Allostatic Load** to the injury-risk flagship slot
   (cumulative-load model with validated AUC on the same cohort).
3. **Promote Viscoelastic Workload** as the complementary stuff-decay model.
4. **Freeze MechanixAE ticket #8** (pitch-type conditioning) pending product
   decision to invest further; base rate of AUC improvement does not justify
   the cost.

## Award narrative update

Remove MechanixAE from the proprietary-ML flagship list. Replace with
Allostatic Load + Viscoelastic Workload. MechanixAE remains in the analytics
suite as a descriptive mechanics anomaly score, not as a predictive model.

## Reproducibility

- All seeds fixed to 42 (`np.random.seed`, `torch.manual_seed`, DataLoader generator).
- Checkpoints: `models/mechanix_ae/per_pitcher/{pitcher_id}.pt` (111 files).
- Percentile run: `python scripts/mechanix_ae_roc_analysis.py --skip-train --score-mode percentile`.
- Z-score run: `python scripts/mechanix_ae_roc_analysis.py --skip-train --score-mode zscore`.
- Unit tests: `pytest tests/test_mechanix_roc_analysis.py` (23 passing, covers
  both scoring modes and API compatibility).
- Raw outputs: `results/mechanix_ae/`
  - `summary_percentile.json`, `roc_curve_percentile.{json,html}`
  - `summary_zscore.json`, `roc_curve_zscore.{json,html}`
  - `lead_time_distribution_{percentile,zscore}.{json,html}`
  - `fpr_summary_{percentile,zscore}.json`
  - `fpr_healthy_seasons_{percentile,zscore}.csv`
  - `lead_time_per_pitcher_{percentile,zscore}.csv`

## Wall clock

- Section 1 (percentile): 6.4 minutes initial run (ran under pre-rescue code path).
- Section 2 (z-score): 2.0 minutes (reused existing 111 checkpoints via `--skip-train`;
  only healthy-baseline VAEs were retrained for the ROC negative class).

## Limitations acknowledged

- Small rare-subtype n (1–3 for TJ, UCL, labrum).
- 2015–2016 only; more recent seasons pending label ingestion.
- IL is administrative, not medical — label noise imposes a hard ceiling.
- No causal claim even with a positive AUC.
- Z-score is still normalised against each pitcher's own training-set baseline;
  if the baseline itself contains pre-symptom mechanical drift, the z-score
  under-flags. This is fixable only with longer pre-injury horizons we don't
  yet have in the 2015–2016 label slice.
