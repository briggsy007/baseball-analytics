# DPI Reliability & Cluster-Honest CIs — 2026-08-10 (WS3.1)

**Task:** Platform improvement plan WS3.1 (`docs/plans/2026-08-10_platform_improvement_plan.md`),
remediating audit DPI findings 2 (mislabeled stability, YoY n=1) and 9
(bootstrap CIs ignore team clustering) from `docs/audits/FLAGSHIP_AUDIT_2026-08-10.md`.
**Analysis code:** `scripts/dpi_reliability_2026.py` (re-runnable; seeded).
**Numeric artifacts:** `results/defensive_pressing/reliability_2026-08-10/`
(`game_dpi_2015_2025.csv`, `team_season_dpi_2015_2025.csv`, `yoy_stability.csv`,
`split_half_reliability.csv`, `gate_ci_reissue.csv`, `summary.json`).
**This document does not edit `defensive_pressing_results.md`** — errata for the
two false CI statements there (audit finding 5) are owned by task A3. This doc
supersedes the *inference*, not the point estimates.

## Headline results (reported exactly as they landed)

1. **YoY stability is confirmed for 2023→24 (r = 0.5898) but roughly halves in
   the newly computed 2024→25 window: r = 0.3699, 95% CI [0.076, 0.640].**
   The "stability ≈ 0.59" talking point was a single high window. Across all
   ten adjacent-season windows 2015→16 … 2024→25 the Fisher-z mean is
   **0.4414** (range 0.225–0.594).
2. **DPI does not meet the FanGraphs 0.707 split-half reliability bar in any
   full 162-game season** (10 of 10 full seasons below the bar; Spearman-Brown
   corrected r = 0.495–0.635). Full-season reliability is **0.584** →
   the implied regression toward the league mean is **~42%** (UZR-style
   "regress ~50%" territory). A `regressed_dpi` column now ships alongside raw.
3. **Cluster-honest CIs: every Gate 2 and Gate 6 estimate still clears its
   threshold as a point estimate, but no Gate 2/Gate 6 CI is separated from
   its threshold any more.** Only Gate 3 (DPI vs BABIP-against — the gate the
   audit calls near-tautological) remains CI-separated. The pooled Gate 6
   (r = 0.4869 vs the retrofitted 0.45 threshold, +0.037 margin) has a wild
   cluster bootstrap p-value of **0.707** against H0: r = 0.45 — the marquee
   pooled external-validation number is statistically indistinguishable from
   its own gate line.

## Model provenance

The working-tree `models/defensive_pressing/xout_v1.pkl` is contaminated
(nightly retrain on 2015–2026; audit DPI finding 3, quarantined by task A1) and
was **not** used. All scoring used the validation-era blob extracted from git:

- **Source:** `git show 32c7142:models/defensive_pressing/xout_v1.pkl`
  (commit "DPI flagship reinforcement: persisted xOut checkpoint + park
  target-encoding", 2026-04-18)
- **sha256:** `e689bff6ab069474c57df6950ba3ed7d376de8b0a3a7a71861d2a96dc3d3bb39`
- **Metadata:** `train_seasons = [2015..2022]`, `fitted_at =
  2026-04-18T22:17:15Z`, `use_park = false`, features
  `[launch_speed, launch_angle, spray_angle, bb_type_encoded]`, internal AUC
  0.8936 — exactly the recipe in spec §B (train 2015–2022, test 2023–2024) and
  the same checkpoint the 2025 external validation recorded
  (`2025_validation/dpi_vs_oaa_yearly.json` → identical train_seasons and AUC).
  Note: the WS0.1 plan text describes the frozen recipe as "2015–2024 per
  `ensure_xout_model` policy"; the actual committed validation blob is
  2015–2022. 2015–2022 is what every published gate number was produced with,
  so it is the correct artifact for re-issuing their CIs.
- **Fidelity check:** recomputed season DPI for all 90 team-seasons 2023–2025
  matches `2025_validation/team_rankings_all_years.csv` with
  **max |diff| = 0.000** (90/90 cells exact at the recorded 3-decimal
  resolution), despite the checkpoint being pickled under sklearn 1.8.0 and
  loaded under 1.6.1. The parity check is recorded in `summary.json`.

Per-game DPI was recomputed for 2015–2025 with the exact production semantics
(`calculate_game_dpi` parity: same BIP cohort filter, n_bip ≥ 5 per game,
actual−expected with expected summed over feature-complete rows, game DPI
rounded to 3 decimals, season = mean of game DPIs) via the production
prediction path (`defensive_pressing.compute_expected_outs` against the loaded
frozen model). 52,498 team-games scored, read-only DB access via
`src/db/schema.py::get_connection(read_only=True)`.

## 1. Year-over-year stability (audit finding 2)

The results doc's "three-year stability 0.58/0.56/0.64" claim was mislabeled —
those are per-year DPI-vs-OAA *cross-metric* correlations. True YoY stability
had been measured once (2023→24). Now measured properly:

| Window | Source | n | Pearson r | 95% CI | Spearman ρ | 95% CI |
|---|---|---:|---:|---|---:|---|
| 2023→2024 | recorded rankings CSV | 30 | **0.5898** | [0.393, 0.742] | 0.5818 | [0.307, 0.752] |
| 2024→2025 | recorded rankings CSV | 30 | **0.3699** | [0.076, 0.640] | 0.3651 | [0.002, 0.701] |

The 2023→24 figure confirms the recorded r ≈ 0.59 (Gate 4 recorded 0.588–0.595
across v1/v2/v3 runs; tiny differences are fresh-scoring-run variance).
The 2024→25 window — never previously computed — is barely above Gate 4's 0.30
threshold and its CI reaches down to 0.08.

Full history (DB-recomputed with the frozen model, ordinary paired bootstrap,
n = 30 teams each):

| Window | r | 95% CI | | Window | r | 95% CI |
|---|---:|---|---|---|---:|---|
| 2015→16 | 0.2986 | [−0.170, 0.594] | | 2020→21 | 0.5939 | [0.209, 0.792] |
| 2016→17 | 0.2411 | [−0.181, 0.534] | | 2021→22 | 0.5049 | [0.152, 0.736] |
| 2017→18 | 0.5658 | [0.332, 0.784] | | 2022→23 | 0.4350 | [0.136, 0.672] |
| 2018→19 | 0.4966 | [0.292, 0.667] | | 2023→24 | 0.5895 | [0.388, 0.739] |
| 2019→20 | 0.2252 | [−0.213, 0.553] | | 2024→25 | 0.3695 | [0.078, 0.637] |

Fisher-z mean over the ten windows: **0.4414**. 2023→24 was the single
strongest window in a decade; quoting it as "the" stability is selection.
An honest stability claim is "YoY r ≈ 0.44 on average, ranging 0.23–0.59"
(three of ten windows — 2015→16, 2016→17, 2019→20 — have CIs including zero;
only the last involves the 2020 short season).

## 2. Split-half reliability (Spearman-Brown)

Method: within each team-season, games split by `game_pk % 2` (~82 games per
half in full seasons; ~31 in 2020); DPI computed per half exactly as production
computes seasons; the two half-values correlated across the 30 teams within
each season; Spearman-Brown corrected to full-season length
(r_full = 2r/(1+r)). Convention: FanGraphs' Cronbach-α 0.707 "reliable" bar
(α for a two-part split equals the Spearman-Brown corrected split-half r).

| Season | split-half r | SB-corrected r_full | ≥ 0.707? |
|---:|---:|---:|:---:|
| 2015 | 0.4279 | 0.5993 | no |
| 2016 | 0.4601 | 0.6303 | no |
| 2017 | 0.4084 | 0.5799 | no |
| 2018 | 0.3920 | 0.5633 | no |
| 2019 | 0.4472 | 0.6181 | no |
| 2020* | 0.5773 | 0.7320 | (yes)* |
| 2021 | 0.3291 | 0.4953 | no |
| 2022 | 0.4517 | 0.6223 | no |
| 2023 | 0.4654 | 0.6352 | no |
| 2024 | 0.3501 | 0.5186 | no |
| 2025 | 0.3892 | 0.5603 | no |

\* 2020 is a 60-game season (~31-game halves) — its Spearman-Brown correction
extrapolates to a 60-game "full season", not 162 games, and the estimate is a
single n=30 draw; it is not evidence the metric is reliable at 162 games.

**Full-season reliability (Fisher-z mean over the ten 162-game seasons):
0.584.** All ten full seasons fall below the 0.707 bar. Implication: a
full-season team DPI is ~58% signal / ~42% noise on the between-team axis.

**Shrinkage.** The implied regression coefficient toward the league mean is
R = 0.584 (regress ~42%, comparable to the UZR "regress ~50%" norm):

```
regressed_dpi = league_season_mean + 0.584 * (dpi_mean - league_season_mean)
```

`team_season_dpi_2015_2025.csv` now carries `reliability_R` and
`regressed_dpi` for every team-season (2020 uses its own season estimate,
0.732). Any consumer quoting season DPI rankings should quote the regressed
column; raw leaderboard gaps overstate true team separation by ~1/0.584 ≈ 1.7×.

## 3. Cluster-aware CI re-issue for Gates 2, 3, 6 (audit finding 9)

What is being replaced: the recorded CIs were 1,000-draw percentile bootstraps
resampling team-season *rows* as iid (`defensive_pressing_validation.py::_bootstrap_ci`).
In the 60-cell (2023–24) and 90-cell (2023–25) windows each franchise
contributes 2–3 rows whose DPI correlates ~0.4–0.6 year-to-year (§1), so the
iid-row bootstrap understates uncertainty. Replacements (all seeded, in
`scripts/dpi_reliability_2026.py`):

- **Team-season DPI CIs:** percentile bootstrap resampling *games within
  team-season* (B = 2,000) — per-cell CIs in `team_season_dpi_2015_2025.csv`.
- **Pairs cluster bootstrap** (B = 5,000): resample the 30 franchises with
  replacement, keep all seasons of a drawn franchise; percentile CI on
  Pearson r (and Spearman ρ).
- **Wild cluster bootstrap-t** (B = 9,999, Rademacher weights on the 30
  franchise clusters, CR1 cluster-robust se, bootstrap-t interval; Cameron &
  Miller JHR 2015 — 30 clusters is few-cluster territory): CI for the
  standardized-regression slope, which equals Pearson r in the original
  sample. Caveat: bootstrap slope replicates are not bounded to [−1, 1], so
  WCB-t bounds can exceed 1 (see 2025 row).
- **Cross-check:** `wildboottest` 0.3.2 (pip-installed; the package
  self-reports `__version__` 0.0.0) WCR-11 p-values for H0: slope = threshold
  via a shifted-outcome reparameterization (regress zy − threshold·zx on zx and
  test that slope = 0), B = 9,999, null imposed.
- Single-season windows have one row per cluster, so both cluster schemes
  reduce to their ordinary iid analogues there (re-issued for completeness).

Thresholds are from `docs/models/defensive_pressing_validation_spec.md` —
Gate 2 ≥ 0.40 (RP proxy), Gate 3 ≤ −0.50 (BABIP-against), Gate 6 ≥ 0.45 (OAA).
**The Gate 6 threshold was retrofitted**: the spec itself records that 0.45 was
set *after* the first measurement (spec:175-178; audit DPI finding 4), and a
fallback ≥ 0.40 gate is pre-positioned in `dpi_vs_oaa_yearly.json`. The gates
as specified bind on point estimates, not CI bounds; both are reported.

### Re-issued CIs — exactly as they landed

Point estimates recomputed from the published team-season table
(`team_rankings_all_years.csv`); ±0.002-scale differences vs the v2 run's
recorded r (0.6197/−0.7334) are fresh-scoring-run variance in that run, not in
this analysis.

| Gate (window) | r | threshold | old iid CI | pairs-cluster CI | WCB-t CI | se infl. | point clears? | CI separated from threshold? | WCB p @ threshold |
|---|---:|---:|---|---|---|---:|:---:|:---:|---:|
| G2 RP (2023–24, n=60) | 0.6162 | ≥ 0.40 | [0.423, 0.771] | [0.393, 0.761] | [0.306, 0.929] | 1.28× | PASS | **NO** | 0.161 |
| G2 RP (pooled 23–25, n=90) | 0.5826 | ≥ 0.40 | — | [0.369, 0.724] | [0.262, 0.915] | 1.72× | PASS | **NO** | 0.354 |
| G3 BABIP (2023–24, n=60) | −0.7355 | ≤ −0.50 | [−0.831, −0.623] | [−0.826, −0.615] | [−0.956, −0.513] | 1.09× | PASS | yes | 0.039 |
| G3 BABIP (pooled 23–25, n=90) | −0.7180 | ≤ −0.50 | — | [−0.803, −0.613] | [−0.932, −0.501] | 1.19× | PASS | yes | 0.013 |
| G6 OAA (2023–24, n=60) | 0.5612 | ≥ 0.45 | [0.307, 0.721] | [0.308, 0.754] | [0.235, 0.886] | 1.31× | PASS | **NO** | 0.470 |
| G6 OAA (2025, n=30) | 0.6406 | ≥ 0.45 | [0.421, 0.792] | [0.424, 0.797] | [0.270, 1.006] | 0.93× | PASS | **NO** | 0.162 |
| G6 OAA (pooled 23–25, n=90) | 0.4869 | ≥ 0.45 | [0.317, 0.635] | [0.317, 0.627] | [0.273, 0.700] | 1.01× | PASS | **NO** | **0.707** |

("se infl." = cluster-robust se / iid se. "CI separated" = the 95% cluster-aware
CI lies entirely on the passing side of the threshold, under both cluster
methods.)

### Gate re-issue verdicts

- **Gates 2, 3, 6 all still PASS as specified** — every point estimate clears
  its threshold, including pooled windows not covered by the original spec.
- **The margin story changes.** Under cluster-honest CIs, *no* Gate 2 or
  Gate 6 estimate is separated from its threshold. The audit's prediction
  (finding 9: pooled CIs too narrow) is confirmed in direction for five of
  seven windows (se inflation up to 1.72×); the pooled Gate 6 CI itself barely
  moves (1.01×) — the honest headline there was already visible in the old CI
  and simply was not stated: **0.4869 clears the retrofitted 0.45 line by
  0.037, with a threshold-test p of 0.707.** Passing a threshold the data
  cannot distinguish from the estimate is not evidence of margin.
- **The only CI-separated gate is Gate 3**, which the audit (findings 1, 6)
  classifies as near-tautological (DPI shares half to two-thirds of its team
  variance with BABIP-against by construction). The gate whose passing means
  least is the only one that passes with statistical room.
- The two previously published claims that CI lower bounds cleared thresholds
  ("bottom of the CI [0.307] is well above the 0.45 threshold";
  "2025's CI lower bound (0.42) is above ... 0.45") were false when written
  and remain false under the re-issued CIs (0.31/0.23 and 0.42/0.27 by
  method). Errata in `defensive_pressing_results.md` are task A3's deliverable.

## What this changes about the DPI story

1. **Retire the "stability 0.59" claim.** Replace with: mean YoY r ≈ 0.44
   (range 0.23–0.59 over ten windows); 2024→25 = 0.37 [0.08, 0.64]. Gate 4's
   0.30 threshold clears in only **7 of 10** historical windows — 2015→16
   (0.299), 2016→17 (0.241), and 2019→20 (0.225, COVID-adjacent) fall below
   it. Had Gate 4 been evaluated on a different adjacent-season pair, it could
   have FAILed as specified.
2. **Season DPI should ship regressed.** Full-season reliability 0.584 means
   raw ranking gaps overstate true separation ~1.7×; the regressed column
   exists now. No full season meets the 0.707 convention.
3. **Gate margins were an artifact of understated uncertainty plus a
   retrofitted threshold.** The defensible statement after this re-issue:
   "DPI-vs-OAA r ≈ 0.49–0.64 by window; cluster-aware CIs span roughly
   0.24–0.89; the pre-registered-after-measurement 0.45 threshold is passed on
   the point estimate in every window but is inside every CI." The stronger
   defensible core remains the audit's partial-r finding (WS3.6 owns the
   claims rewrite).

## Reproduction

```
python scripts/dpi_reliability_2026.py            # extracts the git blob itself
python scripts/dpi_reliability_2026.py --reuse-scores   # analysis only, ~2 min
```

Environment at run time: Python 3.12.0, sklearn 1.6.1, numpy 2.2.6,
scipy 1.17.1, pandas 2.2.3, duckdb 1.2.2, wildboottest 0.3.2; seed 42;
B = 2,000 (team) / 5,000 (pairs-cluster) / 9,999 (wild-cluster).
Artifact sha256 `e689bff6...d3bb39` is embedded in `summary.json`.
