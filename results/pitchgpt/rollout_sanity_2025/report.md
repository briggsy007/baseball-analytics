# PitchGPT Phase 0.6 Rollout Sanity 2025 -- Report

Generated: 2026-08-05T01:23:57Z

Spec: `docs/pitchgpt_sim_engine/PHASE_0.5_PLAN.md §3 + §6.3 / §6.4 / §6.6`


## Configuration

- n_pa = **10000**, n_samples = **100**, horizon = **6**, T = **1.000**, seed = **42**
- Cohort: 2025 pitcher-disjoint from train (2015-2022 excluded 2247 pitchers)
- Total eligible PAs in cohort: 64460; sampled: 10000
- League-median ump scalar (2024 prior season): 0.525333
- Device: cuda

## Wall clock

- PRIMARY (PGConcatHeadPredictor, 10K x 100 samples x H=6): **544.6s**
- SECONDARY (None predictor): 269.0s
- Empirical baseline aggregation: 1.2s

## Phase 0.6 binding gates (per PHASE_0.5_PLAN §3.5 + §6.3 / §6.4 / §6.6)

| Gate | Sampled | Empirical | Delta | Tolerance | PASS band | Verdict |
|---|---|---|---|---|---|---|
| k_pct | 0.3339 | 0.2180 | +0.1159 | +/-0.0100 (min(rel*emp, abs)) | [0.2080, 0.2280] | **FAIL** |
| bb_pct | 0.1177 | 0.0876 | +0.0301 | +/-0.0088 (min(rel*emp, abs)) | [0.0789, 0.0964] | **FAIL** |
| hr_pct | 0.0242 | 0.0321 | -0.0079 | +/-0.0032 (min(rel*emp, abs)) | [0.0288, 0.0353] | **FAIL** |
| mean_woba | 0.3188 | 0.3302 | -0.0114 | +/-0.015 (abs) | [0.3152, 0.3452] | **PASS** |
| mean_pa_length_pitches | 3.9219 | 3.8858 | +0.0362 | +/-0.5 (abs) | [3.3858, 4.3858] | **PASS** |
| calibration_valid_coverage (§6.6) | 1.0000 | >= 0.95 | -- | -- | -- | **PASS** |

**Overall**: **FAIL**


## PRIMARY -- 5 metrics with 95% bootstrap CIs

| Metric | Sampled | 95% CI | Empirical | 95% CI |
|---|---|---|---|---|
| K% | 0.3339 | [0.3329, 0.3349] | 0.2180 | [0.2146, 0.2212] |
| BB% | 0.1177 | [0.1171, 0.1184] | 0.0876 | [0.0855, 0.0898] |
| HR% | 0.0242 | [0.0241, 0.0243] | 0.0321 | [0.0307, 0.0334] |
| HBP% | 0.0114 | [0.0112, 0.0117] | -- | -- |
| in_play_hit% | 0.1672 | [0.1665, 0.1679] | -- | -- |
| mean wOBA | 0.3188 | [0.3176, 0.3199] | 0.3302 | [0.3262, 0.3343] |
| mean PA length | 3.9219 | [3.9184, 3.9253] | 3.8858 | [3.8708, 3.9006] |
| p_truncated | 0.0552 | [0.0547, 0.0556] | -- | -- |

PRIMARY n_pa = 10000, calibration_valid_coverage = 1.0000

## SECONDARY -- None-predictor degraded mode (NOT gated)

Per PHASE_0.5_PLAN §3.4 + §5.3: with `outcome_predictor=None`, PA-termination falls back to a count-only heuristic that misclassifies in-play / foul / HBP outcomes as strikes -- biases K% / BB% high by construction.  Run measures the bias magnitude.

| Metric | Sampled (None) | Sampled (PRIMARY) | Empirical | Bias vs PRIMARY | Bias vs Empirical |
|---|---|---|---|---|---|
| K% | 0.6445 | 0.3339 | 0.2180 | +0.3107 | +0.4266 |
| BB% | 0.3554 | 0.1177 | 0.0876 | +0.2377 | +0.2678 |
| mean PA length | 4.9662 | 3.9219 | 3.8858 | +1.0443 | +1.0804 |

NOTE: HR%, mean wOBA are unavailable in SECONDARY (no outcome predictor -> `pa_outcome=None`).  Per the API contract, consumers MUST NOT silently emit wOBA aggregations when the outcome predictor is missing.

## Honest caveat -- A1 hit-vs-out noise (per PHASE_0.5_PLAN §3.3)

The A1 outcome predictor's `in_play_hit` test log-loss is **2.34** (per `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/metrics.json::test_metrics.per_class_log_loss.in_play_hit`).  This clears the WEAKER PASS gate (<2.5) but misses the full PASS gate (<2.0).  Consequence: the hit-vs-out marginal is structurally noisier than the league average -- A1 cannot condition on post-pitch `launch_speed` / `launch_angle`, so hit-vs-out is decided downstream of the model's information set.  **Mean-wOBA aggregation inherits this noise and is the noisiest of the five gated metrics.**  K% / BB% terminate via count-only and are insulated from this ceiling.

Per `EXECUTION_PLAN.md` §3 the locked claim is "calibrated rollout engine" with calibration as load-bearing -- this report honors that by surfacing where rollout sub-distributions are weakest.

## Mean-wOBA decomposition (per PHASE_0.5_PLAN §5.5)

| Component | Value |
|---|---|
| in_play_hit_pct (sampled) | 0.1672 |
| HBP%   (sampled)          | 0.0114 |
| in_play_hit contribution (* 0.892)  | 0.1491 |
| HBP contribution (* 0.708)          | 0.0081 |
| **Sum (computed mean wOBA)**        | **0.1572** |
| Walk attribution if added (BB% * 0.690) | 0.0812 |

Notes:
- `WObaTable.default()` is the 7-element scalar wOBA table per PHASE_0.5_PLAN §2.0.5.3.
- Walks (count-driven, `pa_outcome=ROLLOUT_PAD_OUTCOME`) get **0** from the default table.  The empirical 2025 mean wOBA includes walks at the league wOBA-on-walk (~0.69).  This is the single largest known under-attribution.
- If `mean wOBA` gate FAILs, decomposition above identifies whether the gap is in (a) hit-vs-out marginal noise (A1 ceiling) or (b) walk attribution (table choice).  Phase-1 follow-up: extend `WObaTable` to attribute walks correctly OR switch to the full Statcast empirical (outcome x pitch_type) table.


## None-predictor bias (per PHASE_0.5_PLAN §3.4)

Expected K% bias (None vs PRIMARY): **+0.3107** (+31.07pp)
Expected BB% bias (None vs PRIMARY): **+0.2377** (+23.77pp)

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