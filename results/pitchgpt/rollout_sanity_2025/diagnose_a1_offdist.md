# Phase 0.6 PitchGPT Diagnostic -- A1 Off-Distribution at Counterfactually-Sampled Pitches

Generated: 2026-04-26T16:56:25Z

Verdict: **FALSIFIED**


FALSIFIED (direction-opposite): TVD=0.0406, KL=0.0049.  Counterfactual A1 outcome distribution drifts in the OPPOSITE direction from the Phase 0.6 K%/BB% bias: ball: +2.63pp (OVER-predicts ball), called_strike: +1.22pp, swinging_strike: -3.21pp (net strike-marginal goes DOWN under counterfactual sampling, not UP).  Whatever drift exists in A1 under counterfactual sampling is in the WRONG direction to explain the +8.9pp K% / -4.5pp BB% gate FAIL.  The bias must come from elsewhere -- pitch-token sampling, PA-termination logic, or static-context (parallel agents).


## Configuration

- n_pa requested: **5000**, processed: **5000**
- n_samples per PA (rollout): **20**, horizon: **6**, seed: 42
- Device: cuda
- Backbone v2 SHA: `6f952054d14ac6f9...`
- A1 head SHA: `37b50e87599013c2...`
- A1 locked temperature: 0.8003

## Wall clock

- Pass A (in-distribution): 68.6s
- Pass B (counterfactual rollout): 393.2s
- Total: 478.4s

## Eval set sizes

- In-distribution pitches scored: 19,528
- Counterfactual sampled pitches scored: 320,939

## A1 outcome marginal -- in-distribution vs counterfactual

| Class | In-distribution | Counterfactual | Delta (pp) |
|---|---|---|---|
| ball | 0.2685 | 0.2948 | +2.6272 |
| called_strike | 0.1943 | 0.2065 | +1.2184 |
| swinging_strike | 0.2026 | 0.1705 | -3.2093 |
| foul | 0.1426 | 0.1341 | -0.8536 |
| in_play_out | 0.1284 | 0.1289 | +0.0529 |
| in_play_hit | 0.0589 | 0.0595 | +0.0553 |
| hbp | 0.0047 | 0.0057 | +0.1017 |

## Distributional distances

- TVD(in_dist, counterfactual) = **0.0406** (threshold 0.0500)
- KL(in || cf) = **0.0049** (threshold 0.0500)
- KL(cf || in) = 0.0046
- JS divergence = 0.0012

## Rare-token off-distribution analysis

- Train rare threshold: freq < 0.0010
- Rollout common threshold: freq >= 0.0100
- N rare tokens (low train, high rollout): **0**
- N counterfactual eval rows on rare tokens: 0
- N counterfactual eval rows on common tokens: 320,939
- TVD(rare_marg, common_marg) = **None**

Top 15 over-sampled tokens (rollout - train, with rollout_freq >= 0.005):

| Token | Train freq | Rollout freq | Delta |
|---|---|---|---|
| 168 | 0.008143 | 0.010304 | +0.002161 |
| 173 | 0.006285 | 0.008301 | +0.002016 |
| 218 | 0.004919 | 0.006525 | +0.001606 |
| 53 | 0.007999 | 0.009326 | +0.001326 |
| 198 | 0.006572 | 0.007877 | +0.001305 |
| 138 | 0.004532 | 0.005836 | +0.001304 |
| 193 | 0.008434 | 0.009563 | +0.001129 |
| 13 | 0.005276 | 0.006319 | +0.001043 |
| 143 | 0.005673 | 0.006599 | +0.000927 |
| 28 | 0.005208 | 0.006001 | +0.000793 |
| 18 | 0.004928 | 0.005680 | +0.000752 |
| 163 | 0.007424 | 0.008123 | +0.000699 |
| 158 | 0.005091 | 0.005718 | +0.000626 |
| 188 | 0.008215 | 0.008603 | +0.000388 |
| 78 | 0.007184 | 0.007534 | +0.000350 |

## Calibration drift (in-distribution ECE with vs without temperature)

- ECE in-distribution, T=0.8003 (production): **0.0206**
- ECE in-distribution, T=1.0 (no scaling):    **0.0383**

Counterfactual ECE cannot be computed without ground-truth labels for sampled pitches.  We report in-distribution ECE with T=0.8003 (production) vs T=1.0 (no scaling).  If the T=0.8003 ECE is much smaller than T=1.0 ECE in-distribution AND the marginal TVD(in, cf) > 0.05, the temperature is over-fit in-distribution and contributes to the off-dist bias under counterfactual sampling.

## K% attribution estimate (rough)

- In-distribution strike marginal (CS+SS, per-pitch): 0.3969
- Counterfactual strike marginal (CS+SS, per-pitch):  0.3770
- Delta strike marginal: **-1.9908pp**
- Naive K% pp attribution (upper-bound order-of-magnitude): -1.9908pp
- Phase 0.6 observed K% gap: +8.88pp
- Naive fraction of observed gap explained: -0.2242

Naive attribution: per-pitch strike-marginal shift in pp is an upper-bound estimate of per-PA K% shift contribution.  Compounding across PA-length distribution may amplify (strike-per-pitch -> K-per-PA is a positive-feedback loop).  Use this value as a rough order-of-magnitude only.

## Cross-references

- Hypothesis source: PHASE_0.6_DIAGNOSIS or comparable diagnostic plan.
- A1 head: `src/analytics/pitchgpt_outcome_head.py::FrozenOutcomeHeadConcat`
- Rollout: `src/analytics/pitchgpt_sim.py::rollout`
- Phase 0.6 gate-fail metrics: `results/pitchgpt/rollout_sanity_2025/metrics.json`