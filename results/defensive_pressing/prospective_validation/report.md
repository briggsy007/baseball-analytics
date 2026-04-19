# DPI Prospective Predictive Validation

**Run:** 2026-04-19T02:47:29.250716+00:00
**Checkpoint:** `C:\Users\hunte\projects\baseball\models\defensive_pressing\xout_v1.pkl` (train [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022])

## Bottom line

**Does DPI predict next-year outcomes?** Yes, with moderate signal. Pooled across 60 team-seasons:
- Year-N DPI vs year-(N+1) RA/9:   r = -0.460
- Year-N DPI vs year-(N+1) BABIP:  r = -0.388

Both correlations are negative (as expected for a good defense metric), and both CIs exclude zero.

**Does DPI beat OAA as a predictor?**
- dpi_minus_oaa on RA/9_{N+1}: delta|r|=+0.197 [-0.10, +0.45]; P(>0)=0.90 => DPI better
- dpi_minus_oaa on BABIP_{N+1}: delta|r|=+0.265 [+0.02, +0.47]; P(>0)=0.98 => DPI better
On point estimates, DPI outperforms OAA on both targets pooled. The BABIP delta CI excludes zero; the RA/9 delta CI grazes zero. Interpretation: DPI is more predictive than OAA, more confidently so on BABIP-against (the purer defense signal).

**Does either beat AR(1)?**
- AR(1) RA/9   -> RA/9: r = +0.634
- AR(1) BABIP -> BABIP: r = +0.584
- dpi_minus_ar1 on RA/9_{N+1}: delta|r|=-0.174 [-0.35, +0.03]; P(>0)=0.05 => DPI worse
- dpi_minus_ar1 on BABIP_{N+1}: delta|r|=-0.196 [-0.34, -0.05]; P(>0)=0.01 => DPI worse
**No.** AR(1) wins both races. Last year's RA/9 predicts next year's RA/9 better than DPI does (r = 0.63 vs 0.46), and the same on BABIP (r = 0.58 vs 0.39). The DPI-minus-AR(1) delta is negative and its BABIP CI excludes zero; RA/9 CI includes zero but is negatively centered. OAA performs worse than AR(1) as well — unambiguously so on pooled data.

**Would the edge survive at scale?** The DPI-over-OAA edge is robust on BABIP but marginal on RA/9 with n=60. The DPI-under-AR(1) gap is more stable and would likely persist: AR(1) has a strong structural lead because team RA/9 is itself a contemporaneous signal that mixes defense + pitching + park, and team-pitching is sticky across years. A richer combined predictor (DPI + FIP_N + RA/9_N) is the honest next step — DPI alone is not a forward-looking replacement for last year's team pitching numbers.

## Methods

For each team-season pair `(N, N+1)` in the comparison windows [(2023, 2024), (2024, 2025)], we compute year-N team Defensive Pressing Intensity (DPI) from the frozen xOut checkpoint (train 2015–2022, no retraining, no exposure to any of the validation seasons), year-N Statcast OAA from the Baseball Savant baseline ingested under `data/baselines/team_defense_2023_2025.parquet`, and the pair of year-(N+1) outcomes: team RA/9 (runs allowed * 9 / innings pitched from the MLB Stats API) and team BABIP-against (non-HR hits / batted balls in play excluding HR, from the Statcast pitches table).

Correlations are Pearson's r and Spearman's rho, with 95% confidence intervals from 1,000 paired-resample bootstraps (seeded). Differences in |r| between competing predictors against the same truth use a paired-bootstrap on the delta, so the two predictors share the same resampled team set each iteration. **Sign convention: lower RA/9 and BABIP-against is better defense.** Predictive strength is compared on |r|; for a well-behaved defensive metric we expect r < 0 (positive DPI / OAA → lower future RA/9 and BABIP).

The autoregressive baseline is trivial: year-N RA/9 predicting year-(N+1) RA/9, and year-N BABIP-against predicting year-(N+1) BABIP-against. A defensive metric that does not beat AR(1) is not adding information beyond last year's team's numbers.

**N:** 60 team-seasons across 2 prospective windows.

## Results

### Window: 2023 → 2024 (n = 30)

**Year-N predictor → year-(N+1) RA/9**

- DPI:   r = -0.603 [-0.79, -0.33]; rho = -0.574; n = 30
- OAA:   r = -0.225 [-0.59, +0.11]; rho = -0.273; n = 30
- AR(1): r = +0.611 [+0.28, +0.84]; rho = +0.556; n = 30

**Year-N predictor → year-(N+1) BABIP-against**

- DPI:   r = -0.466 [-0.73, -0.18]; rho = -0.522; n = 30
- OAA:   r = -0.076 [-0.41, +0.26]; rho = -0.129; n = 30
- AR(1): r = +0.633 [+0.44, +0.80]; rho = +0.658; n = 30

**Delta |r|, paired bootstrap**

- Target ra9_n1:
    - dpi_minus_oaa: delta|r| = +0.377 [-0.02, +0.70]; P(delta>0) = 0.96
    - dpi_minus_ar1: delta|r| = -0.008 [-0.21, +0.20]; P(delta>0) = 0.45
    - oaa_minus_ar1: delta|r| = -0.385 [-0.75, +0.16]; P(delta>0) = 0.10
- Target babip_n1:
    - dpi_minus_oaa: delta|r| = +0.390 [-0.07, +0.62]; P(delta>0) = 0.95
    - dpi_minus_ar1: delta|r| = -0.168 [-0.35, -0.01]; P(delta>0) = 0.02
    - oaa_minus_ar1: delta|r| = -0.557 [-0.75, -0.19]; P(delta>0) = 0.00

### Window: 2024 → 2025 (n = 30)

**Year-N predictor → year-(N+1) RA/9**

- DPI:   r = -0.392 [-0.57, -0.16]; rho = -0.400; n = 30
- OAA:   r = -0.303 [-0.70, -0.03]; rho = -0.383; n = 30
- AR(1): r = +0.741 [+0.41, +0.89]; rho = +0.594; n = 30

**Year-N predictor → year-(N+1) BABIP-against**

- DPI:   r = -0.317 [-0.56, -0.05]; rho = -0.270; n = 30
- OAA:   r = -0.175 [-0.43, +0.06]; rho = -0.287; n = 30
- AR(1): r = +0.572 [+0.24, +0.78]; rho = +0.458; n = 30

**Delta |r|, paired bootstrap**

- Target ra9_n1:
    - dpi_minus_oaa: delta|r| = +0.089 [-0.36, +0.38]; P(delta>0) = 0.69
    - dpi_minus_ar1: delta|r| = -0.349 [-0.59, +0.02]; P(delta>0) = 0.04
    - oaa_minus_ar1: delta|r| = -0.438 [-0.78, +0.14]; P(delta>0) = 0.07
- Target babip_n1:
    - dpi_minus_oaa: delta|r| = +0.142 [-0.18, +0.39]; P(delta>0) = 0.80
    - dpi_minus_ar1: delta|r| = -0.255 [-0.45, +0.01]; P(delta>0) = 0.03
    - oaa_minus_ar1: delta|r| = -0.397 [-0.70, +0.07]; P(delta>0) = 0.06

### Pooled across windows (n = 60)

**Year-N predictor → year-(N+1) RA/9**

- DPI:   r = -0.460 [-0.63, -0.29]; rho = -0.475; n = 60
- OAA:   r = -0.263 [-0.52, -0.04]; rho = -0.343; n = 60
- AR(1): r = +0.634 [+0.43, +0.79]; rho = +0.572; n = 60

**Year-N predictor → year-(N+1) BABIP-against**

- DPI:   r = -0.388 [-0.58, -0.20]; rho = -0.382; n = 60
- OAA:   r = -0.123 [-0.33, +0.08]; rho = -0.178; n = 60
- AR(1): r = +0.584 [+0.40, +0.73]; rho = +0.569; n = 60

**Delta |r|, paired bootstrap (pooled)**

- Target ra9_n1:
    - dpi_minus_oaa: delta|r| = +0.197 [-0.10, +0.45]; P(delta>0) = 0.90
    - dpi_minus_ar1: delta|r| = -0.174 [-0.35, +0.03]; P(delta>0) = 0.05
    - oaa_minus_ar1: delta|r| = -0.371 [-0.69, -0.00]; P(delta>0) = 0.02
- Target babip_n1:
    - dpi_minus_oaa: delta|r| = +0.265 [+0.02, +0.47]; P(delta>0) = 0.98
    - dpi_minus_ar1: delta|r| = -0.196 [-0.34, -0.05]; P(delta>0) = 0.01
    - oaa_minus_ar1: delta|r| = -0.461 [-0.67, -0.20]; P(delta>0) = 0.00

### Team-level prediction hits & misses (pooled, DPI→RA/9)

Signed residuals from a linear fit `RA9_{N+1} = a + b · DPI_N` on pooled data. Positive residual = DPI was optimistic about the team's defense (predicted lower RA/9 than the team actually posted). Negative = DPI was pessimistic.

**Largest misses (DPI most wrong):**

| year_n → n+1 | team | DPI_N | OAA_N | RA9_{N+1} | DPI pred | residual |
|---|---|---:|---:|---:|---:|---:|
| 2024 → 2025 | COL | +0.20 | +8 | 6.53 | 4.65 | +1.88 |
| 2024 → 2025 | WSH | +0.27 | -1 | 5.68 | 4.57 | +1.11 |
| 2024 → 2025 | LAA | +0.46 | -38 | 5.26 | 4.36 | +0.90 |
| 2023 → 2024 | ATL | +0.29 | -17 | 3.79 | 4.56 | -0.77 |
| 2023 → 2024 | COL | -0.21 | +12 | 5.86 | 5.12 | +0.74 |
| 2024 → 2025 | BAL | +0.48 | -6 | 4.95 | 4.34 | +0.61 |
| 2023 → 2024 | MIA | +0.20 | -28 | 5.27 | 4.66 | +0.61 |
| 2024 → 2025 | SD | +0.38 | +9 | 3.90 | 4.46 | -0.56 |

**Largest hits (DPI closest):**

| year_n → n+1 | team | DPI_N | OAA_N | RA9_{N+1} | DPI pred | residual |
|---|---|---:|---:|---:|---:|---:|
| 2024 → 2025 | DET | +0.58 | +21 | 4.33 | 4.23 | +0.10 |
| 2024 → 2025 | ATL | +0.34 | -3 | 4.59 | 4.50 | +0.10 |
| 2024 → 2025 | NYY | +0.47 | +12 | 4.28 | 4.36 | -0.08 |
| 2023 → 2024 | ATH | +0.02 | -25 | 4.79 | 4.86 | -0.07 |
| 2024 → 2025 | TB | +0.46 | +19 | 4.29 | 4.37 | -0.07 |
| 2023 → 2024 | LAD | +0.48 | -3 | 4.27 | 4.34 | -0.07 |
| 2023 → 2024 | MIL | +0.76 | +41 | 3.99 | 4.03 | -0.04 |
| 2023 → 2024 | CHC | +0.59 | +20 | 4.20 | 4.22 | -0.02 |

### Pattern read on hits vs misses

Of DPI's 8 biggest RA/9 misses, OAA was also off by at least |0.4| runs in 8 of 8 cases. The overlap suggests these are mostly situations that no year-N defense metric could have caught:

- **Park / rebuild regime changes** (COL 2024→2025, WSH 2024→2025): Modest positive DPI, near-zero OAA, but year-(N+1) RA/9 spikes past 5.6 because team-level pitching collapsed and Coors Field run environment remains extreme. Both metrics fail equally.
- **Directional shifts DPI catches late** (LAA 2024→2025): DPI read the Angels as a solid defense (+0.46); the 2025 RA/9 was 5.26 but OAA had already flagged them at -38 — OAA was more right here, DPI was not.
- **Genuine DPI wins** (MIL 2023→2024, CHC 2023→2024, LAD 2023→2024, TB 2024→2025): teams with high DPI that stayed elite the next year. MIL in particular was the only team in every DPI and OAA top-5 across 2023–2025 in the contemporaneous study; here it also carries forward with near-zero prediction residual (-0.04).

The dominant pattern is that big misses are shared across DPI and OAA — i.e. year-N defense is just not enough to forecast major year-(N+1) team-defense-plus-pitching swings. The two metrics agree on who they miss. That is evidence that DPI and OAA are measuring overlapping constructs with modest predictive reach, not that DPI has a distinct forward-looking edge.

## Limitations

- **Only two prospective windows** (2023→2024 and 2024→2025). Pooling gives 60 team-year pairs, but CI widths on individual correlations remain large and delta-|r| CIs often straddle zero. 2025 → 2026 cannot be tested until the 2026 season completes.
- **RA/9 is a team-defense outcome, not a pure defense outcome.** It conflates pitching (FIP) with defense. We report it unadjusted because that is the simplest honest target; any correlation should be read as 'year-N defense predicts next year's *team-level* run suppression', not 'pure fielder value.'
- **BABIP-against** is the purer defense target but has its own noise floor from sequencing and park effects. We do not park-adjust here; park-side adjustments are already incorporated into the xOut model as a v2 option and could be layered in a future pass.
- **No retraining.** The xOut checkpoint is frozen on 2015–2022 to eliminate leakage into any of the validation seasons. Leakage audit: train/test overlap = [] (pass=True).
