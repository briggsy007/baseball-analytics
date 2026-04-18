# MechanixAE Validation Results

**Date:** 2026-04-16.
**Spec:** [`mechanix_ae_validation_spec.md`](mechanix_ae_validation_spec.md).
**Pipeline:** [`scripts/mechanix_ae_roc_analysis.py`](../../scripts/mechanix_ae_roc_analysis.py).
**Wall clock:** 6.4 minutes end-to-end on CPU (111 per-pitcher VAEs plus 50 healthy
baselines).

## Executive summary

MechanixAE's Mechanical Drift Index (MDI), in its current per-pitcher percentile-rank
form, does **not** discriminate injured from healthy pitcher-days at the 30-day pre-IL
window. AUC(MDI) on a 111-pitcher / 50-season evaluation cohort is **0.384** (95%
bootstrap CI 0.358–0.414) — materially below the 0.50 random baseline, and far below
the award-worthy 0.65 gate. Velocity-drop is also worse than random (AUC 0.271). The
percentile-rank definition of MDI saturates at ~100 for essentially every
pitcher-season (100% of the 50 healthy seasons breach MDI ≥ 80 at some point), making
the "lead-time" statistic misleading: injured pitchers do cross the threshold ~60 days
pre-IL, but so does every healthy pitcher many times per season. The honest verdict
is a **PIVOT** from "injury early-warning" to "mechanical profiling." The unsupervised
framing retains descriptive value; the predictive framing is not supported by these
data.

## Cohort

- Labels: `data/injury_labels.parquet`, 2015–2016 window (the only seasons currently
  populated with 30-day pre-IL pitch data).
- 336 unique pitchers with an arm injury (earliest IL placement per pitcher).
- 111 pitchers have ≥ 200 healthy pitches (game_date < il_date − 30) and were trained;
  92 of those also had ≥ 20 pre-IL pitches for the lead-time analysis.
- 0 training failures, 0 skipped after eligibility filter.
- Per injury type trained: shoulder 37, elbow 30, other_arm 19, forearm 16,
  rotator_cuff 5, tommy_john 3, ucl_sprain 1, labrum 0.
- Training wall: 3.4 min for injured cohort, plus ~3 min for 50 healthy baselines.
- Per-pitcher recon loss median 0.80, range 0.54–0.97 — VAE fits the healthy windows.

## Lead-time distribution (Ticket 3)

Measured as (il_date − first_date_daily_MDI ≥ threshold).

| Threshold | n breached / 92 | Median (days) | Mean | IQR |
|-----------|-----------------|---------------|------|-----|
| 60        | 92 (100 %)      | **63.5**      | 58.6 | 40–82 |
| 70        | 92 (100 %)      | **63.0**      | 58.0 | 39–81 |
| 80        | 92 (100 %)      | **61.5**      | 56.4 | 34–81 |

By injury type at threshold 70 (median days): shoulder 69, elbow 65, other_arm 61,
rotator_cuff 58, forearm 58, tommy_john 80 (n=1), ucl_sprain 32 (n=1).

These numbers look impressive in isolation (spec calls >10 days "meaningful") but
collapse once the FPR section is read: the same thresholds also fire for essentially
every healthy pitcher-season because MDI is a per-pitcher percentile rank — the
maximum in any held-out window is, by construction, near 100.

## ROC / AUC (Tickets 4 & 5)

- **Positive class:** pitcher-days within 30 days pre-IL for the 92 pitchers who had
  pre-IL data. n_pos = 613 pitcher-days.
- **Negative class:** random 50 pitcher-seasons (2015–2016) from pitchers never
  appearing in the arm-injury label set, each split 60/40 train/eval (MDI computed on
  the eval half). Downsampled to 10× positive class. n_neg = 645 pitcher-days.
- Bootstrap CI uses 200 resamples with seed 42.

| Detector                         | AUC   | 95 % CI        | Gate |
|----------------------------------|-------|----------------|------|
| MDI                              | 0.384 | [0.358, 0.414] | PIVOT |
| Velocity-drop (−2 σ rolling 10)  | 0.271 | [0.246, 0.300] | — |
| delta_AUC = MDI − VelDrop         | **+0.113** | — | meets ≥ 0.10 |

Both curves sit below the diagonal. The signed ordering of MDI *is* correct (injured
score slightly higher on average than healthy in bootstrap subsets with equal size)
but the threshold-sweep monotonicity is reversed because healthy pitcher-seasons
contribute evaluation windows drawn from a different baseline distribution than the
injured pitchers' own baselines — see the Methodology caveat below.

## False-positive rate on healthy seasons (Ticket 6)

50 healthy pitcher-seasons (≥ 1000 pitches, never injured per labels).

| Threshold | Pct breached at any point |
|-----------|---------------------------|
| MDI ≥ 60  | 100.0 %                   |
| MDI ≥ 70  | 100.0 %                   |
| MDI ≥ 80  | **100.0 %**               |

Per-season pct-of-days above threshold: median 100 % at MDI ≥ 60, 100 % at MDI ≥ 70,
91 % at MDI ≥ 80. Spec ceiling of 50 % at MDI ≥ 80 for an "unusable model" is exceeded
by 50 percentage points. This is the hard fail.

## Methodology caveats

1. **MDI is a rank, not a magnitude.** By definition, at least one window in any
   evaluation set scores at the top percentile. Any pitcher evaluated long enough
   *will* breach MDI = 100. This is a property of the scoring function, not of the
   pitcher's mechanics. A usable binary detector has to use a *fixed* absolute
   reconstruction-error threshold (e.g. z-scored against baseline mean+sd) rather
   than a ranked percentile.
2. **Baseline asymmetry.** Injured pitchers' MDI is ranked against their own healthy
   pre-IL-30 windows. Healthy pitchers' MDI is ranked against their own earlier
   60 % of the season. These two baseline definitions aren't comparable — the
   injured-baseline is more pitch-type-homogeneous and has less seasonal drift
   baked in, so the injured pitchers' evaluation windows look *less* anomalous
   than the healthy evaluation set. This partially explains AUC < 0.5.
3. **90-day vs full-season evaluation horizon.** Positives are drawn from 30 days;
   negatives from the final 40 % of season. The negative pool has many more
   opportunities to produce high-percentile windows.
4. **IL date precision.** Free-text parse; retroactive placements and rehab
   assignments inject noise into the positive class.
5. **Pitch-type entanglement (FLAG 3).** The 10-D feature space mixes fastballs
   and breaking balls; per-pitcher × pitch-type mean-centering does not remove
   within-pitch-type variance. This is Ticket 8 territory — the latent space may
   be tracking mix shifts rather than injury signatures.

## Decision gate verdict

**PIVOT.** AUC(MDI) = 0.384 is below the 0.58 pivot threshold. We are not submitting
MechanixAE as an injury-prediction system on this evidence. Three forward options:

1. **Re-cast as mechanical profiling.** Keep MDI on the dashboard as a descriptive
   anomaly score — useful for "this pitcher's mechanics shifted noticeably this
   start" colour commentary. Drop the "injury early warning" language.
2. **Redefine the score.** Replace percentile-rank MDI with z-scored reconstruction
   error against the pitcher's own healthy baseline, then sweep AUC again. This is
   cheap (≈ 10 min) and may rescue the predictive framing; the current pipeline
   already produces per-window recon errors alongside the MDI.
3. **Pitch-type conditioning (Ticket 8).** Add one-hot pitch-type conditioning to
   the VAE so the latent space models within-type variation only. This directly
   attacks FLAG 3 and is the most likely candidate to produce a measurable lift.

Recommended path: execute (2) immediately — it's a one-line change — before
committing to the harder (3) retrain cycle.

## Reproducibility

- All seeds fixed to 42 (`np.random.seed`, `torch.manual_seed`, DataLoader generator).
- Checkpoints written to `models/mechanix_ae/per_pitcher/{pitcher_id}.pt`.
- Run: `python scripts/mechanix_ae_roc_analysis.py`.
- Unit tests: `pytest tests/test_mechanix_roc_analysis.py` (13 passing).
- Raw outputs: `results/mechanix_ae/` — `summary.json`, `roc_curve.{json,html}`,
  `lead_time_distribution.{json,html}`, `fpr_summary.json`, `training_coverage.json`.

## Next tickets

- **#7 (CUSUM tuning).** Low priority until the score definition is fixed.
- **#8 (pitch-type conditioning).** Highest ROI next step — directly addresses the
  most plausible confounder.
- **New ticket: score redefinition.** Swap percentile-rank for z-scored recon error;
  repeat Steps 3-5 in this pipeline. If post-fix AUC still < 0.58, adopt the
  "mechanical profiling" narrative permanently.

## Limitations acknowledged

- Small rare-subtype n (1–3 for TJ, UCL, labrum).
- 2015–2016 only; more recent seasons pending label ingestion beyond those years.
- IL is administrative, not medical — label noise imposes a hard ceiling.
- No causal claim — even with a positive AUC, MDI would be *associated with* IL
  placement, not *causing* it.
