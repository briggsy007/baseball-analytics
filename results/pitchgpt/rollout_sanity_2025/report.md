# PitchGPT Phase 0.6 Rollout Sanity 2025 -- Report

Generated: 2026-04-26T15:20:17Z

Spec: `docs/pitchgpt_sim_engine/PHASE_0.5_PLAN.md §3 + §6.3 / §6.4 / §6.6`


## Configuration

- n_pa = **10000**, n_samples = **100**, horizon = **6**, T = **1.000**, seed = **42**
- Cohort: 2025 pitcher-disjoint from train (2015-2022 excluded 2247 pitchers)
- Total eligible PAs in cohort: 64460; sampled: 10000
- League-median ump scalar (2024 prior season): 0.525333
- Device: cuda

## Wall clock

- PRIMARY (PGConcatHeadPredictor, 10K x 100 samples x H=6): **406.0s**
- SECONDARY (None predictor): 191.4s
- Empirical baseline aggregation: 1.2s

## Phase 0.6 binding gates (per PHASE_0.5_PLAN §3.5 + §6.3 / §6.4 / §6.6)

| Gate | Sampled | Empirical | Delta | Tolerance | PASS band | Verdict |
|---|---|---|---|---|---|---|
| k_pct | 0.3068 | 0.2180 | +0.0888 | +/-0.0100 (min(rel*emp, abs)) | [0.2080, 0.2280] | **FAIL** |
| bb_pct | 0.0430 | 0.0876 | -0.0446 | +/-0.0088 (min(rel*emp, abs)) | [0.0789, 0.0964] | **FAIL** |
| hr_pct | 0.0276 | 0.0321 | -0.0044 | +/-0.0032 (min(rel*emp, abs)) | [0.0288, 0.0353] | **FAIL** |
| mean_woba | 0.2939 | 0.3302 | -0.0363 | +/-0.015 (abs) | [0.3152, 0.3452] | **FAIL** |
| mean_pa_length_pitches | 3.2259 | 3.8858 | -0.6599 | +/-0.5 (abs) | [3.3858, 4.3858] | **FAIL** |
| calibration_valid_coverage (§6.6) | 1.0000 | >= 0.95 | -- | -- | -- | **PASS** |

**Overall**: **FAIL**


## PRIMARY -- 5 metrics with 95% bootstrap CIs

| Metric | Sampled | 95% CI | Empirical | 95% CI |
|---|---|---|---|---|
| K% | 0.3068 | [0.3058, 0.3078] | 0.2180 | [0.2147, 0.2210] |
| BB% | 0.0430 | [0.0426, 0.0434] | 0.0876 | [0.0855, 0.0898] |
| HR% | 0.0276 | [0.0275, 0.0277] | 0.0321 | [0.0307, 0.0335] |
| HBP% | 0.0184 | [0.0182, 0.0187] | -- | -- |
| in_play_hit% | 0.1912 | [0.1904, 0.1920] | -- | -- |
| mean wOBA | 0.2939 | [0.2929, 0.2949] | 0.3302 | [0.3262, 0.3344] |
| mean PA length | 3.2259 | [3.2229, 3.2290] | 3.8858 | [3.8716, 3.9006] |
| p_truncated | 0.0257 | [0.0254, 0.0260] | -- | -- |

PRIMARY n_pa = 10000, calibration_valid_coverage = 1.0000

## SECONDARY -- None-predictor degraded mode (NOT gated)

Per PHASE_0.5_PLAN §3.4 + §5.3: with `outcome_predictor=None`, PA-termination falls back to a count-only heuristic that misclassifies in-play / foul / HBP outcomes as strikes -- biases K% / BB% high by construction.  Run measures the bias magnitude.

| Metric | Sampled (None) | Sampled (PRIMARY) | Empirical | Bias vs PRIMARY | Bias vs Empirical |
|---|---|---|---|---|---|
| K% | 0.5905 | 0.3068 | 0.2180 | +0.2837 | +0.3726 |
| BB% | 0.4095 | 0.0430 | 0.0876 | +0.3665 | +0.3218 |
| mean PA length | 4.9035 | 3.2259 | 3.8858 | +1.6776 | +1.0177 |

NOTE: HR%, mean wOBA are unavailable in SECONDARY (no outcome predictor -> `pa_outcome=None`).  Per the API contract, consumers MUST NOT silently emit wOBA aggregations when the outcome predictor is missing.

## Honest caveat -- A1 hit-vs-out noise (per PHASE_0.5_PLAN §3.3)

The A1 outcome predictor's `in_play_hit` test log-loss is **2.34** (per `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/metrics.json::test_metrics.per_class_log_loss.in_play_hit`).  This clears the WEAKER PASS gate (<2.5) but misses the full PASS gate (<2.0).  Consequence: the hit-vs-out marginal is structurally noisier than the league average -- A1 cannot condition on post-pitch `launch_speed` / `launch_angle`, so hit-vs-out is decided downstream of the model's information set.  **Mean-wOBA aggregation inherits this noise and is the noisiest of the five gated metrics.**  K% / BB% terminate via count-only and are insulated from this ceiling.

Per `EXECUTION_PLAN.md` §3 the locked claim is "calibrated rollout engine" with calibration as load-bearing -- this report honors that by surfacing where rollout sub-distributions are weakest.

## Mean-wOBA decomposition (per PHASE_0.5_PLAN §5.5)

| Component | Value |
|---|---|
| in_play_hit_pct (sampled) | 0.1912 |
| HBP%   (sampled)          | 0.0184 |
| in_play_hit contribution (* 0.892)  | 0.1705 |
| HBP contribution (* 0.708)          | 0.0131 |
| **Sum (computed mean wOBA)**        | **0.1836** |
| Walk attribution if added (BB% * 0.690) | 0.0297 |

Notes:
- `WObaTable.default()` is the 7-element scalar wOBA table per PHASE_0.5_PLAN §2.0.5.3.
- Walks (count-driven, `pa_outcome=ROLLOUT_PAD_OUTCOME`) get **0** from the default table.  The empirical 2025 mean wOBA includes walks at the league wOBA-on-walk (~0.69).  This is the single largest known under-attribution.
- If `mean wOBA` gate FAILs, decomposition above identifies whether the gap is in (a) hit-vs-out marginal noise (A1 ceiling) or (b) walk attribution (table choice).  Phase-1 follow-up: extend `WObaTable` to attribute walks correctly OR switch to the full Statcast empirical (outcome x pitch_type) table.

**Mean wOBA gate FAIL diagnosis path** (per §5.5):

- Total mean wOBA gap (sampled - empirical): -0.0363
- Expected walk under-attribution gap (if we DID attribute walks at 0.69): -0.0297
- Residual after walk-correction (= hit-vs-out / table effects): -0.0066

Diagnosis hypothesis ranking:
1. If `|residual|` < 0.005 -- the mean-wOBA failure is a known under-attribution from the scalar wOBA table not crediting walks.  Phase-1 fix: extend `WObaTable` to handle walks.  Not a rollout-engine bug.
2. If `|residual|` >= 0.005 -- the gap is real.  Likely sources (most -> least): (a) A1 hit-vs-out marginal noise (test log-loss 2.34); (b) PA-termination logic bug in `rollout()`; (c) calibration-feature-CDF mis-build in `pitchgpt_build_calibration_cdfs.py`.

## None-predictor bias (per PHASE_0.5_PLAN §3.4)

Expected K% bias (None vs PRIMARY): **+0.2837** (+28.37pp)
Expected BB% bias (None vs PRIMARY): **+0.3665** (+36.65pp)

Per SIM_ENGINE_API §4.4 + PHASE_0.5_PLAN §5.3: None-predictor falls back to a zone-based count heuristic (in-zone token -> +1 strike, otherwise +1 ball).  Misclassifies in-play / foul / HBP as strikes.  Bias magnitude here is what consumers should expect when the predictor is unavailable -- DOES NOT trip the Phase 0.6 PASS/FAIL gate.

## Backbone + A1 SHA256 verification (per PHASE_0.5_PLAN §6.8 / §6.9)

| Asset | Path | SHA256 (recomputed) | Match (locked) | Match (runtime) | Size |
|---|---|---|---|---|---|
| backbone v2 | `C:\Users\hunte\projects\baseball\models\pitchgpt_v2.pt` | `6f952054d14ac6f918f3eb9502b496b7...` | **YES** | YES | -- |
| A1 head | `C:\Users\hunte\projects\baseball\models\pitchgpt_v2_outcomehead_a1.pt` | `37b50e87599013c281560c9f63286fe5...` | -- | **YES** | 151289 bytes (expected 151289: YES) |

Backbone locked SHA: `6f952054d14ac6f918f3eb9502b496b70bc0c87dfc65dc50d98ee7244a62883c`
Backbone recomputed SHA: `6f952054d14ac6f918f3eb9502b496b70bc0c87dfc65dc50d98ee7244a62883c`
A1 runtime SHA: `37b50e87599013c281560c9f63286fe5b7645297d0042694d907287417bb25e5`

## Cross-references

- API spec: `docs/pitchgpt_sim_engine/SIM_ENGINE_API.md` §3, §4, §5, §6, §9.
- Phase plan: `docs/pitchgpt_sim_engine/PHASE_0.5_PLAN.md` §3, §6.3 / §6.4 / §6.6 / §6.8 / §6.9.
- A1 metrics: `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/metrics.json`.
- Empirical baseline (prep): `results/pitchgpt/rollout_sanity_2025/empirical_baselines_2025.json`.