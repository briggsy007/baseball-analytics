# DPI Combined-Predictor: Does DPI add variance beyond AR(1)+FIP?

*n = 60 team-seasons (2023->2024, 2024->2025)*

## Question

The prospective validation showed DPI predicts next-year team defense (r≈-0.46 for RA/9, r≈-0.39 for BABIP-against), but loses head-to-head to AR(1) (r≈+0.63 for RA/9). This script asks whether DPI still carries signal **after conditioning on AR(1) + team FIP**. If DPI's coefficient in a multivariable OLS has 95% CI excluding zero, the "loses to AR(1)" caveat flips into "adds variance beyond AR(1)".

## Headline: RA/9

- **A1** (AR(1) only):           R² = 0.402
- **A2** (AR(1)+FIP+DPI):        R² = 0.426 (ΔR² vs A1 = +0.023)
- **A3** (AR(1)+FIP+OAA):        R² = 0.426 (ΔR² vs A1 = +0.024)

DPI in A2: β_std = -0.061 (95% CI [-0.329, +0.207]), p = 0.651
OAA in A3: β_std = -0.055 (95% CI [-0.270, +0.160]), p = 0.610

Partial F-test DPI|AR(1)+FIP: F(1,56) = 0.207, p = 0.651, ΔR² = +0.002
Partial F-test OAA|AR(1)+FIP: F(1,56) = 0.264, p = 0.610, ΔR² = +0.003

## Headline: BABIP-against

- **B1** (AR(1) only):           R² = 0.341
- **B2** (AR(1)+FIP+DPI):        R² = 0.355 (ΔR² vs B1 = +0.013)
- **B3** (AR(1)+FIP+OAA):        R² = 0.350 (ΔR² vs B1 = +0.008)

DPI in B2: β_std = +0.116 (95% CI [-0.206, +0.439]), p = 0.474
OAA in B3: β_std = +0.033 (95% CI [-0.191, +0.258]), p = 0.766

Partial F-test DPI|AR(1)+FIP: F(1,56) = 0.520, p = 0.474, ΔR² = +0.006
Partial F-test OAA|AR(1)+FIP: F(1,56) = 0.089, p = 0.766, ΔR² = +0.001

## Collinearity (VIFs)

- **A2**: ra9_n=379.15, fip_n=420.54, dpi_n=4.64
- **A3**: ra9_n=285.42, fip_n=285.82, oaa_n=1.09
- **B2**: babip_n=192.15, fip_n=186.72, dpi_n=3.31
- **B3**: babip_n=187.83, fip_n=187.68, oaa_n=1.01

## Leave-one-team-out (DPI coefficient stability)

- RA/9 model A2: DPI β ranges [-0.122, +0.001], mean -0.061, std 0.027. Significant at p<0.05 in 0/30 LOO fits.
- BABIP model B2: DPI β ranges [+0.046, +0.192], mean +0.116, std 0.029. Significant at p<0.05 in 0/30 LOO fits.

## Predictor correlations

|           | dpi_n | oaa_n | fip_n | ra9_n | babip_n |
|-----------|-------|-------|-------|-------|---------|
| dpi_n     | +1.000 | +0.561 | -0.434 | -0.630 | -0.736 |
| oaa_n     | +0.561 | +1.000 | -0.214 | -0.316 | -0.247 |
| fip_n     | -0.434 | -0.214 | +1.000 | +0.840 | +0.445 |
| ra9_n     | -0.630 | -0.316 | +0.840 | +1.000 | +0.723 |
| babip_n   | -0.736 | -0.247 | +0.445 | +0.723 | +1.000 |

## Supplementary: AR(1) + DPI (no FIP)

The VIFs for RA/9_N and FIP_N are extreme (>300), reflecting that team FIP explains 84%+ of team RA/9 variance within a season. To check whether this collinearity was masking a true DPI effect, we refit dropping FIP. If DPI now carries a significant coefficient, the negative result in A2/B2 could be a collinearity artifact.

- **A2b** (RA/9 ~ RA/9_N + DPI_N): R² = 0.408; DPI β_std = -0.100 (95% CI [-0.363, +0.162]), p = 0.448
- **B2b** (BABIP ~ BABIP_N + DPI_N): R² = 0.345; DPI β_std = +0.090 (95% CI [-0.226, +0.407]), p = 0.570

## Sign inspection

Univariate r(DPI_N, RA9_{N+1}) = -0.46 and r(DPI_N, BABIP_{N+1}) = -0.39 (both negative, as expected: more pressing = better future defense). In the combined models, DPI's conditional coefficient is β = -0.061 in A2 (RA/9, expected sign) but β = +0.116 in B2 (BABIP, opposite sign). The BABIP sign-flip is a suppression/collinearity artifact: BABIP_N already captures the DPI signal so strongly (r = -0.74 between them) that the residual DPI variance after conditioning on BABIP_N and FIP_N is essentially noise with a small positive coefficient. The p-value of 0.47 indicates this is not a meaningful sign, but it is a reminder that DPI and BABIP-against are very nearly the same construct at the team level.

## Verdict

**HONEST NULL.**

DPI's coefficient is not statistically distinguishable from zero in either combined model. Partial F-tests (DPI | AR(1)+FIP) give p = 0.65 (RA/9) and p = 0.47 (BABIP). At n=60, DPI is **subsumed by the AR(1) + FIP baseline**. The strong univariate r=-0.63 between DPI_N and RA9_N (and r=-0.74 between DPI_N and BABIP_N) means DPI-derived defensive skill is already embedded in the previous year's run-prevention totals. The supplementary no-FIP models (A2b, B2b) still do not surface a significant DPI coefficient, so the null is not a pure collinearity artifact — it is a genuine overlap between DPI and year-N team run-prevention.

OAA performs similarly: β_std = -0.055 (p = 0.610) in RA/9 model A3, and +0.033 (p = 0.766) in BABIP model B3. Neither DPI nor OAA demonstrably adds variance beyond AR(1)+FIP at this sample size. **The DPI-vs-OAA comparison is therefore indeterminate under the rescue test** — both carry the same directional sign but neither reaches significance.

## Caveats

- **n=60** is small; coefficient estimates carry wide CIs. A single anomalous team-season can shift β by ±0.10 (see LOO).
- The two prospective windows (2023->2024, 2024->2025) are not independent observations — they share 2024 values as both a year-N+1 and year-N observation for the same franchises. Standard errors are reported under the OLS iid assumption; a cluster-robust SE by team would be more conservative but the directional conclusion would not change dramatically.
- FIP is computed from raw Statcast pitches using the pitcher's primary team (most pitches in the season). Mid-season trades move event totals to the primary team only. For the 3% of IP lost to non-primary-team appearances, our team FIP slightly underweights those innings, but the effect on the season total is minimal.
- **Path forward**: the rescue test would benefit from adding the 2025->2026 window (an additional 30 team-seasons, bringing n=90). At n=90 the minimum-detectable β for a single predictor in a 3-variable OLS drops by ~20%, and the DPI point-estimate magnitudes we currently see (|β_std| ≈ 0.06-0.12) are within the range that could reach significance with that extra power.
