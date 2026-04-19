# Projections Validation Spec

**Source:** `src/analytics/projections.py` (`class ProjectionModel`); CLI entry
points `scripts/build_projections.py` and `scripts/validate_projections.py`.

## A. Current state

**Architecture.** Marcel-style weighted-average baseline (5/4/3 prior-season
weights, regression-to-mean PA/IP prior of 1200 PA / 134 IP, age curve at
27-31 peak with +/- 0.5 WAR/yr slopes off peak) plus a Statcast forward-
indicator overlay capped at +/- 0.5 WAR. Batter overlay = (xwOBA - wOBA) >=
0.015 over the prior 2 seasons; pitcher overlay = (wOBA-against -
xwOBA-against) >= 0.015 over the prior 2 seasons (xFIP/FIP columns are
0% filled in the current DB, so the FIP/xFIP test was substituted by an
equivalent wOBA-vs-xwOBA test on the pitcher's PAs from the `pitches`
table). Per-player ages come from a cached `data/player_ages.parquet`
built from Baseball-Reference `bwar_bat` / `bwar_pitch` (Chadwick mlbam
keyed; 115,483 player-season rows, 100% join to `players.player_id`).

**Implemented.**
- `ProjectionModel.fit(conn, train_seasons)` -- loads season-level WAR,
  league baselines, and per-PA xwOBA from `pitches`.
- `ProjectionModel.project(conn, target_season, players=None)` -- emits
  `(player_id, name, position, projected_war, marcel_war,
  statcast_adjustment, age, prior_3yr_war, prior_3yr_pa_or_ip)`.
- `save(path)` / `load(path)` via the `BaseAnalyticsModel` joblib path.
- CLI: `scripts/build_projections.py --train-seasons ... --target-season
  ... --output-path ...` emits `projections_{target}.csv` +
  `model_metadata.json`.
- Backtest harness: `scripts/validate_projections.py --train-seasons ...
  --target-season ... --run-dir ...` produces `validation_summary.json`
  matching the `/validate-model` skill schema.

**Tested.** Smoke test on 2021-2023 -> 2024 produces 1,520 player
projections (695 batter, 825 pitcher) with sensible top-of-leaderboard
ordering (Soto, Judge, Tatis, Acuna, Betts, Olson, Riley, Tucker,
Henderson, Alcantara, etc.).

**Not implemented.**
- Multi-year backtest (only 2024 measured).
- Full park-adjusted volume projection (currently uses weighted-avg prior
  PA/IP as the next-season volume, which under-projects breakouts and
  over-projects health risks).
- Position-specific aging curves (single curve applied to both batters
  and pitchers; literature suggests pitcher peak is ~26 not 27-31).
- Calibration plot (only point hit-rate at >= 3 WAR is reported).
- Pitcher overlay using the canonical xFIP - FIP signal (currently the
  wOBA-against / xwOBA-against proxy because both `xfip` and `fip` are
  100% NULL in `season_pitching_stats`).

## B. Validation tickets

### Ticket 1. Backtest 2024 vs 2021-2023 (M)
- **Goal.** Project 2024 from 2021-2023 inputs only, compare to actual 2024 WAR.
- **Artifacts.** `scripts/validate_projections.py`; `results/validate_projections_<UTC-ts>/{projections_2024.csv,backtest_pairs.csv,backtest_metrics.json,top_movers.csv,validation_summary.json}`.
- **Success.** Combined RMSE <= 1.5, Pearson r >= 0.55, Spearman rho >= 0.50, RMSE delta vs Marcel <= 0.0, leakage check pass.
- **Effort.** M.

### Ticket 2. Multi-year backtest stability (M)
- **Goal.** Repeat the backtest for 2022 (from 2019-2021) and 2023 (from 2020-2022) to test reproducibility.
- **Artifacts.** Three `validate_projections_<...>` runs; aggregated metrics table.
- **Success.** Spearman rho stable to within +/- 0.05 across the three target years; no individual year beats /lags by more than 0.10 RMSE.
- **Effort.** M.

### Ticket 3. Position-cohort breakdown (S)
- **Goal.** Pitcher cohort underperforms batter cohort by a wide margin in v1.
- **Artifacts.** Documented batter / pitcher / combined cohort metrics.
- **Success.** Cohort metrics published; root cause for any cohort gap explained (volatility, sample-size, missing FIP/xFIP).
- **Effort.** S. Already done in this run.

### Ticket 4. Statcast overlay ablation (M)
- **Goal.** Quantify the lift the overlay adds over Marcel-only.
- **Artifacts.** `rmse_delta_vs_marcel` reported in metrics for combined / batters / pitchers; honest report of whether overlay helps each cohort.
- **Success.** Combined RMSE delta <= 0; per-cohort breakdown published.
- **Effort.** M. Already done.

### Ticket 5. Calibration plot (S)
- **Goal.** Beyond point-hit-rate at 3 WAR, characterise calibration across the full WAR range.
- **Artifacts.** `scripts/projections_calibration_plot.py` producing reliability diagrams (binned projected vs actual mean).
- **Success.** Calibration error (mean abs deviation across bins) < 0.5 WAR.
- **Effort.** S.

### Ticket 6. Comparison to ZiPS / Steamer / public Marcel (L)
- **Goal.** Most defensible result: head-to-head against canonical projections.
- **Artifacts.** Pull public ZiPS / Steamer 2024 projections from FanGraphs (or pre-season download); comparison table on shared player IDs.
- **Success.** Within +/- 0.10 RMSE of public Marcel; within +/- 0.20 RMSE of ZiPS/Steamer (which use proprietary data and aging curves we do not).
- **Effort.** L.

### Ticket 7. Position-specific aging curves (M)
- **Goal.** Calibrate the +/- 0.5 WAR/yr slope on actual data per position.
- **Artifacts.** `scripts/age_curve_calibration.py` fitting age-vs-WAR-delta on 2015-2023 batter and pitcher cohorts; updated `ProjectionConfig` defaults.
- **Success.** Calibrated slopes within +/- 0.2 of canonical literature (~+0.3 WAR/yr below 27 for batters, ~-0.4 WAR/yr above 31).
- **Effort.** M.

### Ticket 8. Volume / playing-time projection (L)
- **Goal.** The current volume estimator is weighted-avg prior PA/IP, which over-projects injury-prone players and under-projects breakouts. Need a proper next-season-volume model.
- **Artifacts.** Logistic / GBR model on (age, prior_pa, injury_history, position) -> projected_pa.
- **Success.** Volume RMSE < 150 PA / 25 IP on a held-out year.
- **Effort.** L.

### Ticket 9. Pitcher overlay using real xFIP/FIP (S)
- **Goal.** Replace the xwOBA-against proxy with canonical xFIP - FIP once those columns are backfilled.
- **Artifacts.** Conditional branch in `_pitcher_overlay`.
- **Success.** Pitcher cohort Pearson r increases by >= 0.05.
- **Effort.** S, blocked on data ingestion.

### Ticket 10. Production refresh schedule (S)
- **Goal.** Annual retrain at season-end; mid-season "rest-of-season" projections are out of scope for v1.
- **Artifacts.** `scripts/refresh_projections_annual.py` + cron entry.
- **Success.** Idempotent; deterministic.
- **Effort.** S.

## C. Headline gates

- **`leakage_check`** -- all train_seasons strictly < target_season. Required for any release.
- **`rmse_combined`** <= 1.5 WAR. Marcel achieves ~1.6 in literature; this gate is "modestly better than published Marcel."
- **`pearson_r_combined`** >= 0.55. Roughly equivalent to "explains 30% of next-year WAR variance."
- **`spearman_rho_combined`** >= 0.50. Rank ordering at the GM-decision-relevant level.
- **`rmse_delta_vs_marcel`** <= 0.0. Statcast overlay must not hurt; a negative delta means the overlay improves over Marcel-only.

**Per-cohort informational gates.** Batter and pitcher metrics are reported
separately. The pitcher cohort is structurally noisier than the batter
cohort (year-over-year pitcher WAR correlation is ~0.4 in the literature
vs ~0.6 for batters), so the combined gates are weighted by the larger
pitcher cohort and may FAIL even when batters PASS cleanly.

## D. Risk flags

- **Single-year backtest only.** 2024 is one realisation; need three-year
  multi-target backtest (Ticket 2) before publication.
- **Pitcher year-over-year volatility.** Pitcher WAR has well-documented
  low autocorrelation (injury, role changes from SP -> RP, BABIP swings).
  v1 will under-perform on the pitcher cohort relative to the batter
  cohort; this is structural, not a bug.
- **Volume projection naive.** Weighted-avg prior PA/IP is the simplest
  defensible estimator but over-projects injury-prone full-time players
  who lose half a season and under-projects breakouts who jump from 200
  PA -> 600 PA.
- **xFIP / FIP not in DB.** Pitcher overlay uses xwOBA-against vs
  wOBA-against from the `pitches` table as a substitute. The canonical
  xFIP - FIP signal is preferred; Ticket 9.
- **Aging curve uncalibrated.** Defaults are literature-canonical (peak
  27-31, +/- 0.5 WAR/yr off peak) but not fit on this DB. Could push
  young breakouts higher and older players lower than the data supports.
- **No park / regression-to-mean differentiation by position.** Catchers
  and middle infielders should regress more aggressively than corner
  outfielders / DHs in canonical Marcel; v1 uses a single 1200 PA prior.
- **Single 1200 PA / 134 IP prior** is the canonical Marcel default; not
  re-fit on this DB.
- **Statcast overlay scale.** Conversion of (xwOBA - wOBA) -> WAR uses a
  literature-derived ~0.5 WAR per 30-wOBA-point gap. Not empirically
  calibrated on this DB.

---

Status: Specified 2026-04-18 with the v1 model release. First validation
ticket (#1, 2024 backtest) executed in the same run -- see
`docs/models/projections_results.md`.

---

## E. v2 features (2026-04-18, opt-in via `--v2` CLI flag)

Three structural features added behind config flags. Default OFF; the
v1.1 backtest is bit-identical when all flags are False.

### E.1 Tommy John dampener (`enable_tj_flag`)

Loads TJ surgery dates from `data/injury_labels.parquet` (subtype ==
`tommy_john`) and applies a -0.75 WAR adjustment to any pitcher with a
TJ surgery within `tj_window_months` (default 24) before the target
season's start (April 1).

**Coverage limitation (CRITICAL).** `data/injury_labels.parquet` is
ingested by `scripts/ingest_injury_labels.py` from the `transactions`
table, which currently only contains rows for 2015-2016. The TJ
dampener therefore only fires for target seasons in [2015 + 1y,
2016 + 24mo] = [2016, 2018]. For target seasons 2019 or later (including
the 2024 backtest), the feature is **operationally dormant**: the
infrastructure is wired and tested, but no pitcher in the projection
set has a recent-enough TJ on file. This is documented honestly; the
unblock is to backfill `transactions` (2017-2024 IL stints) and re-run
`scripts/ingest_injury_labels.py`.

### E.2 Empirically-calibrated age curve (`enable_calibrated_age_curve`)

Replaces the canonical "+0.5 / 0 / -0.5" piecewise-linear curve with a
data-fit per-position curve. Uses every consecutive-season WAR pair from
`age_curve_train_year_lo` to `age_curve_train_year_hi` (default
2015-2023, must be strictly < target_season for leakage safety). Group
by integer age, compute mean WAR-delta from age N to N+1, smooth with a
centered 3-age rolling mean. Ages with sample size below
`age_curve_min_sample` (default 30) fall back to the canonical curve.

**Diagnostics persisted in fit metrics** at
`v2_age_curve_diagnostics.batter_curve` / `pitcher_curve`. As of 2026-04-18,
batter peak is age 22-24 (+0.13 to +0.53 WAR/yr); pitcher peak is age
23-24 (+0.24 to +0.35 WAR/yr). Both curves are flatter on the old side
and earlier-peaking than the canonical 27-31 plateau.

### E.3 Role-change feature (`enable_role_change`)

For each pitcher, classify each prior season as SP (`gs >= 0.5 * g`) or
RP. If the most recent two prior seasons differ, apply a -0.30 WAR
adjustment (capped at +/- 0.50 WAR). Captures the typical performance
degradation of SP -> RP and RP -> SP transitions.

### E.4 Combined v2 verdict

See the v2 section in `docs/models/projections_results.md`. v2 lifts
combined RMSE from 1.55 -> 1.48 (PASSES the 1.5 gate for the first
time), Pearson 0.496 -> 0.511, Spearman 0.394 -> 0.408. Pitcher cohort
Pearson 0.314 -> 0.328 (+0.015) -- the targeted +0.07 to clear 0.40 was
not hit; the TJ feature is dormant per coverage caveat.
