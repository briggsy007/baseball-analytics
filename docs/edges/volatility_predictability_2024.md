# Volatility Surface Predictability Edge -- 2024 Test

## Headline

We tested whether low-entropy ("predictable") MLB starters underperform their xFIP in 2024 -- the hypothesis being that hitters can sit on patterned location/outcome distributions and beat the pitcher relative to peripherals. Across **n = 124** qualified 2024 starters (>=100 IP, >=50% appearances as GS), Pearson correlation between Volatility Surface entropy and (FIP - xFIP) residual was **r = -0.013** (95% bootstrap CI [-0.168, +0.156], p = 0.888). Spearman rho = **-0.029** (95% CI [-0.195, +0.145], p = 0.748). The point estimate is essentially zero and the wrong sign for the H1 prediction. **Verdict: NULL.** The hypothesis is cleanly rejected; there is no detectable correlation between the entropy metric this module produces and FIP-vs-xFIP residual at the 2024 starter level.

## What the entropy actually measures

`src/analytics/volatility_surface.py` builds a **(26 zone x 12 count) Shannon entropy surface** per pitcher: 5x5 in-zone grid plus one out-of-zone bucket, crossed with all (balls, strikes) states. Each cell's entropy is computed over the **distribution of seven outcome categories** (called_strike, swinging_strike, foul, ball, weak/medium/hard contact). Cells with fewer than 5 pitches are NaN and filled by RBF interpolation. The reported `overall_vol` is the **pitch-count-weighted mean entropy** across all cells. **High entropy = the pitcher's outcome distribution is uniform / unpredictable; low entropy = the pitcher is concentrated in a few outcomes (hitters can sit on it).** So H1 (predictability hurts FIP-xFIP) means: expect Pearson r > 0 between entropy and (FIP - xFIP)... wait -- low entropy => positive residual means **negative correlation** between entropy and the residual. Per spec the PASS threshold was framed as r > 0.15; we report and interpret both signs honestly below.

## Cohort + sample size

124 MLB pitchers with IP >= 100 in 2024 and games-started >= 50% of total appearances. IP range: 102.2 to 216.0; pitch counts (for the entropy estimate) range ~1.5K to ~3K. League-wide 2024 derivation constants used for FIP/xFIP: cFIP = 3.130, lg HR/FB = 0.1026 (computed from in-DB pitch events). FIP and xFIP were computed from raw pitch events (`strikeout`, `walk`, `intent_walk`, `hit_by_pitch`, `home_run`, `bb_type='fly_ball'`) because `season_pitching_stats.fip`/`.xfip` are NULL in this build of the database. Residual = FIP - xFIP (positive = worse than peripherals predict).

## Results

| Statistic | Value | 95% bootstrap CI | p-value |
|---|---|---|---|
| Pearson r(entropy, FIP-xFIP) | -0.013 | [-0.168, +0.156] | 0.888 |
| Spearman rho(entropy, FIP-xFIP) | -0.029 | [-0.195, +0.145] | 0.748 |
| n | 124 | -- | -- |

Per the spec's pre-registered thresholds (PASS: r > 0.15 AND p < 0.05; NULL: |r| < 0.10): the absolute Pearson correlation 0.013 sits **deep in the NULL band**, the CI is symmetric across zero, and p > 0.7 on both tests. The data do not support the predictability-tax hypothesis, in either direction.

## Scatter plot

`results/edges/volatility_predictability_2024_scatter.png`

The cloud is a featureless blob: entropy ranges from ~1.01 (Blake Snell, Michael Lorenzen, Lance Lynn) to ~1.19 (Tarik Skubal, Kutter Crawford, George Kirby), but residuals span from roughly -0.6 to +0.7 with no visible slope. The OLS line is essentially flat.

## Honest interpretation

Both extremes contain success and failure stories simultaneously, which is the cleanest possible refutation of the "predictability tax" framing:

**5 most predictable (lowest entropy):** Blake Snell (1.008, residual -0.52 -- *outperformed* xFIP), Michael Lorenzen (1.016, +0.07), Andre Pallante (1.016, -0.23), Jordan Hicks (1.016, 0.00), Lance Lynn (1.016, -0.12). Mean residual ~ -0.16 -- if anything the most "predictable" group **outperformed** xFIP.

**5 least predictable (highest entropy):** Tarik Skubal (1.189, -0.37 -- the AL Cy Young winner), Kutter Crawford (1.183, +0.43), George Kirby (1.182, -0.25), Miles Mikolas (1.176, +0.12), Aaron Nola (1.174, +0.47). Mean residual ~ +0.08 -- the high-entropy group is also a mix.

Critically, Blake Snell at the most-predictable extreme had the *lowest* FIP-xFIP residual in the cohort (he beat xFIP by 0.52 runs), while Tarik Skubal at the most-unpredictable extreme had a similar performance (-0.37). The metric is uncorrelated with the residual at both the linear and rank levels. Whatever the entropy surface measures, it is not "the outcome distribution that hitters can exploit relative to peripherals" -- at least not at the seasonal level for qualified starters.

## Caveats

- **Single season (2024).** Larger panels (multi-year, 250+ pitcher-seasons) might surface a small effect, but the 95% CI here spans -0.17 to +0.16, so even doubling the sample is unlikely to recover an r > 0.15 effect.
- **FIP/xFIP recomputed from pitch events** rather than read from FanGraphs; absolute scaling differs by ~0.1 run vs FG values, but FIP - xFIP is invariant to the additive cFIP constant, so the residual we correlate is robust.
- **Outcome categories are coarse** (7 buckets). A finer outcome alphabet (e.g. EV deciles, xwOBA buckets) might produce a more discriminating entropy surface. That is a re-design, not a re-validation.
- **Entropy is averaged across all (zone, count) cells uniformly** with smoothing. A starter with very few cells covered by enough pitches gets RBF-interpolated to the global mean, which compresses cross-pitcher variance.
- **The 26x12=312 cell space is over-parameterized** for ~2K-pitch starter samples (avg ~6.5 pitches/cell pre-filter). The MIN_PITCHES_PER_CELL=5 threshold + RBF smoothing collapses most variation into the global mean; this is consistent with the narrow entropy range observed (1.01 to 1.19, ~9% spread).

## What this suggests next

**Retire the volatility surface as an edge candidate** at the seasonal pitcher-evaluation level. The hypothesis was clean, the test is statistically conclusive, and the marketing story ("predictable pitchers underperform peripherals") does not hold in 2024 data. The dashboard view (`src/dashboard/views/volatility_surface.py`) can remain as a descriptive visualization for individual pitcher analysis, but it should not be promoted as a flagship edge model in the audit's HIGH-priority slot. Free that slot for the next-best HIGH candidate from `docs/edges/platform_audit.md` -- candidates by audit priority are **chemnet** (lineup synergy out-of-sample), **defensive_pressing** (DPI vs DRS/OAA), **alpha_decay** (in-game whiff decay), or **pset** (pitch-pair theory vs SIERA/xFIP residual). Of those, chemnet was already recommended as #1 in the audit and remains the highest-leverage validation to run next.

A possible salvage path -- not recommended in this scope -- would be to redefine entropy on a **finer outcome alphabet** (xwOBA-bucketed) and test at the **per-game** level rather than seasonal, where the variance has more headroom. But that is a model rebuild, not a hypothesis re-test.
