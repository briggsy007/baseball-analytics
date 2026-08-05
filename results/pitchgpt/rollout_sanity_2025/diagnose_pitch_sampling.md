# Phase 0.6 PitchGPT Rollout Sanity FAIL -- Diagnostic #1: Pitch-Token Sampling

Generated: 2026-04-26T16:38:24Z
Wall clock: 141.9s

## Hypothesis under test

T=1.0 pitch-token sampling over-emits in-strike-zone tokens at H=0 (first pitch of PA), which accumulates to strikes faster than actual MLB pitchers and explains the +8.9pp K% bias / -4.5pp BB% under-bias / -0.66 mean-PA-length deficit observed in `metrics.json`.

## Verdict: **FALSIFIED**


## Headline numbers

| Metric | Value |
|---|---|
| Observed K% bias (PRIMARY vs empirical, from metrics.json) | **+8.88 pp** |
| Observed BB% bias (PRIMARY vs empirical, from metrics.json) | **-4.46 pp** |
| In-zone delta sampled@T=1.0 vs empirical (tokenizer 5x5 inner-3x3) | **-6.29 pp** |
| Heuristic-implied K% gap (sampled - empirical, H=6 i.i.d.) | **-11.59 pp** |
| Heuristic-implied BB% gap (sampled - empirical, H=6 i.i.d.) | **+11.59 pp** |
| Attributable K% bias to pitch-token sampling | **+0.00 pp** (0.0% of observed +8.88pp) |
| Attributable BB% bias to pitch-token sampling | **+0.00 pp** |

## Convention note (load-bearing)

Statcast's `pitches.zone` column uses a **non-tokenizer** scheme:
- `zone in {1..9}` = inner 3x3 strike zone (Statcast's strict zones)
- `zone in {11..14}` = outer four (gap / low / etc.)

The PitchTokenizer rebuckets `(plate_x, plate_z)` into a 5x5 grid (zones 0..24) plus zone 25 = missing.  Inside the rollout, the count-only fallback (`outcome_predictor=None`) treats the **inner-3x3 of the 5x5 grid** = zones `{6, 7, 8, 11, 12, 13, 16, 17, 18}` as in-zone (`_IN_ZONE_INDICES`).  We compute the in-zone rate under BOTH conventions; **the tokenizer convention is the apples-to-apples comparison** for the rollout.

## Empirical 2025 first-pitch distribution

Source: `pitches WHERE pitch_number=1 AND year=2025`, full season (no pitcher-disjoint filter -- it does not change first-pitch zone marginals materially).

| Convention | in-zone | out-of-zone | missing | n |
|---|---|---|---|---|
| Statcast (`zone in {1..9}` vs `{11..14}`) | 0.5495 | 0.4505 | 0.0000 | 190155 |
| Tokenizer (5x5 inner-3x3 vs rest) | 0.5390 | 0.4610 | 0.0000 | 190155 |

## Sampled (T=1.0) H=0 distribution

- n_pa = 10000, n_samples = 100, n_total_pitches = 1000000
- Token TVD vs empirical: **0.2133**
- Token KL(sampled || empirical): 0.2522
- Token KL(empirical || sampled): 0.1632
- Zone TVD vs empirical: **0.0708**
- Zone KL(sampled || empirical): 0.0244
- Pitch-type TVD vs empirical: 0.1089
- Velo-bucket TVD vs empirical: 0.0839
- In-zone (tokenizer convention): **0.4761** (empirical 0.5390; delta -0.0629; vs uniform +0.1299)
- Wall clock for MAIN run: 89.8s

## Heuristic-implied K%/BB% (count-only None-predictor, H=6 i.i.d.)

This is the closed-form K% / BB% under the rollout's None-predictor fallback (in-zone -> +1 strike, otherwise +1 ball) assuming each of 6 pitches is i.i.d. drawn at the H=0 marginal in-zone rate.  ISOLATES the pitch-token sampling effect from the outcome predictor.

| In-zone source | p_in_zone | implied K% | implied BB% | implied truncated% |
|---|---|---|---|---|
| Empirical (tokenizer) | 0.5390 | 0.7263 | 0.2737 | 0.0000 |
| Empirical (Statcast)  | 0.5495 | 0.7440 | 0.2560 | 0.0000 |
| Sampled @ T=1.0       | 0.4761 | 0.6104 | 0.3896 | 0.0000 |

Note: this i.i.d. heuristic does NOT mirror the rollout's outcome predictor.  It is a stress test that tells us: "given JUST the pitch-token sampling, what K%/BB% would the count-only fallback emit?"  The implied K% delta between the empirical-in-zone and the sampled-in-zone rate is the **maximum K% bias attributable to pitch-token sampling alone**.

## Temperature sweep

| T | n_pa | wall (s) | TVD vs emp (token) | TVD vs emp (zone) | in-zone | delta vs emp |
|---|---|---|---|---|---|---|
| **1.0** (main) | 10000 | 89.8 | 0.2133 | 0.0708 | 0.4761 | -0.0629 |
| 0.8 | 2000 | 16.0 | 0.2613 | 0.0531 | 0.5091 | -0.0299 |
| 1.2 | 2000 | 21.8 | 0.2244 | 0.0909 | 0.4538 | -0.0853 |

Hypothesis check on temperature: if T=1.0 is the issue, T=1.2 (more entropy) should reduce the in-zone bias — i.e., its `delta vs emp` should be smaller (closer to 0) than T=1.0's.  T=0.8 (more concentrated) should increase the bias.

## Decision matrix used to determine verdict

- **CONFIRMED**: `delta_in_zone @T=1.0 > +0.02` AND heuristic-implied K% gap >= +0.005 (0.5pp).
- **FALSIFIED**: `delta_in_zone @T=1.0 <= +0.005` (essentially no in-zone over-emission).
- **INDETERMINATE**: in-zone bias non-trivial but K% impact below 0.5pp.

## Key quantitative claim

Estimate: **+0.00 pp** of the observed +8.88pp K% bias is attributable to pitch-token sampling alone (= 0.0%).  The remainder (~+8.88pp) must come from elsewhere -- candidates: (a) outcome-predictor calibration off-distribution from PA terminal classes; (b) static-context (constant within PA) interacting with the predictor; (c) PA-termination logic.

## Cross-references

- Source rollout-sanity: `scripts/pitchgpt_rollout_sanity_2025.py`
- Source metrics: `results/pitchgpt/rollout_sanity_2025/metrics.json`
- Tokenizer: `src/analytics/pitchgpt.py` :: `PitchTokenizer.encode/decode`
- In-zone heuristic: `src/analytics/pitchgpt_sim.py` :: `_IN_ZONE_INDICES` / `_zone_is_in_strike_zone`