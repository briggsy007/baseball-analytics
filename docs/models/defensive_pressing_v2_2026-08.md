# DPI v2 — Methods & Results (2026-08 remediation series)

**Status:** COMPLETE through C1b — C1a (WS3.2 pitching strip + WS3.3
sprint-speed feature) + C1b (WS3.4 park jointly estimated, WS3.5 positioning
thesis, final K1/K2 measurement, §5). Nothing here changes flagship status,
aliases, or claims — kill-criteria adjudication is Batch D. This document
reports numbers only; it contains **no adjudication language**.

**Provenance anchor:** every DPI number in this doc descends from the frozen
validated xOut checkpoint `models/defensive_pressing/xout_v1.pkl`
(sha256 `e689bff6ab069474…`, train 2015–2022, recorded holdout AUC 0.8936),
via the Batch-A game-level scoring artifact
`results/defensive_pressing/reliability_2026-08-10/game_dpi_2015_2025.csv`
(sha256 `6ce88a267300…`, 52,498 team-games). Output artifacts live in
`results/defensive_pressing/v2_2026-08/`.

**Pre-committed expectation (plan WS3 header):** the win is bias removal and
honest CIs, NOT a higher headline correlation. Per "Solving DIPS", single-season
BABIP variance ≈ luck 44% / pitcher 28% / fielding 17% / park 11% — DPI v1's
residual is mostly not fielding, and stripping the pitcher share should
*reduce* shared variance with everything, including OAA.

---

## 1. WS3.2 — Stage 1: strip pitcher contact-management

### 1.1 Method (Swartz two-stage, THT)

Game-level DPI (frozen model, Batch-A CSV) is regressed on the defending
staff's pitcher-season peripherals; **DPI_v2_stage1 = the residual**. Script:
`scripts/dpi_v2_pitching_strip.py` (read-only vs DuckDB).

Peripherals (plan-mandated: peripherals, NOT pitcher BABIP/xwOBAcon target
encodings — contact management is y2y-unreliable, peripherals are the stable
carrier). PA = PA-ending pitches (`events IS NOT NULL`) excluding
`truncated_pa`/`ejection`/`game_advisory`:

| feature | definition |
|---|---|
| `k_pct` | (strikeout + strikeout_double_play) / PA |
| `ubb_pct` | walk / PA (intent_walk excluded: manager choice, not pitcher skill) |
| `net_gb` | (GB − FB) / PA (bb_type on `type='X'` rows) |
| `popup_rate` | popups / PA |
| `sp_share` | share of the pitcher's season appearances that were starts (start = threw the side's first pitch, ordered by at_bat_number, pitch_number) |

**Leakage rule (pre-declared choice: leave-one-game-out).** A game's staff
peripheral values are season totals **minus that game's own counts**, so no
outcome from the scored game enters its own adjustment. LOGO was chosen over
prior-season history because it (a) keeps the same-season contact profile —
the actual confound in that season's DPI, (b) has no rookie/mover missingness,
and (c) K/BB/GB/FB/PU are stable enough within season that the strip is not
noise-dominated. Pitcher-games with LOGO PA < 20 fall back to league-season
mean rates: 4,685 / 227,267 pitcher-game rows (2.061%), carrying 1.39% of BIP
weight.

**Aggregation:** BIP-weighted mean of each pitcher's LOGO peripherals per
(game, defending team); weights = the pitcher's BIP allowed in that game
(DPI sums over BIP, so BIP share is the correct exposure). Join parity vs the
Batch-A CSV is exact: 52,498/52,498 team-games matched, 0 BIP-count
mismatches.

**Stage-1 regression:** per-season OLS (intercept + 5 peripherals) of game
DPI on the staff aggregate; residual = DPI_v2_stage1. The fit is a
descriptive decomposition — in-sample per season by construction (the
underlying 2023–25 DPI is OOS w.r.t. the frozen xOut's 2015–2022 train
window). Residuals are season-centered via the OLS intercept; comparisons
below account for this explicitly.

### 1.2 Variance explained by the pitching stage (audit finding 7, quantified)

Game level (R² of the per-season stage-1 OLS; game DPI is luck-dominated at
this grain, so small R² is expected):

| season | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| R² (game) | .0319 | .0355 | .0374 | .0348 | .0243 | .0264 | .0295 | .0354 | .0299 | .0250 | .0333 |

Team-season level (where DPI claims operate): variance of the team-season
mean pitching-hat as a share of team-season DPI variance, and its
correlation with team-season DPI v1:

| season | var share | r(pitch_hat, DPI v1) |
|---|---|---|
| 2015 | 0.0785 | −0.2805 |
| 2016 | 0.1356 | 0.2668 |
| 2017 | 0.1018 | 0.2752 |
| 2018 | 0.1295 | 0.0490 |
| 2019 | 0.0939 | 0.2888 |
| 2020 | 0.0478 | 0.0262 |
| 2021 | 0.0559 | 0.1762 |
| 2022 | 0.1487 | 0.5221 |
| 2023 | 0.1346 | 0.5236 |
| 2024 | 0.1639 | 0.3005 |
| 2025 | 0.1045 | 0.2201 |

Read: in the K2 window (2023–25), the pitching stage carries 10–16% of
team-season DPI variance and correlates up to r = 0.52 with the v1 ranking —
audit finding 7's confound is real and material at the team-season level.
(All values in-sample descriptive decomposition; dataset = 52,498 team-games
2015–2025, 30 teams/season.)

### 1.3 K2 interim — partial r(DPI, OAA | BABIP-against), 2023–25

Kill criterion K2 (verbatim, plan §8): *"if after pitching-strip + park +
speed the partial r(DPI_v2, OAA | BABIP) pooled 2023–25 falls below 0.30,
DPI's defense claim narrows to descriptive BIP-outcome residual; flagship
status reviewed."* The number below is the **interim trajectory after the
pitching strip alone** — the adjudicating number lands after C1b adds
park + speed.

Covariates from `results/defensive_pressing/2025_validation/team_rankings_all_years.csv`
(team OAA = Savant sum over non-catcher positions; BABIP-against from the
same recorded table). n = 30 teams/year, 90 pooled. CIs: Fisher-z
(SE 1/√(n−4)); pooled rows also carry a pairs cluster bootstrap over the 30
franchises (5,000 draws, seed 42).

| metric | 2023 | 2024 | 2025 | pooled 2023–25 |
|---|---|---|---|---|
| v1 (frozen, audit baseline) | 0.4206 [0.064, 0.682] | 0.6850 [0.425, 0.841] | 0.5387 [0.215, 0.756] | 0.4064 [0.217, 0.567]; cluster [0.221, 0.541] |
| v1 season-centered (sensitivity) | 0.4206 | 0.6850 | 0.5387 | 0.5725 [0.414, 0.698]; cluster [0.411, 0.696] |
| **v2 stage-1 (pitching strip)** | **0.3647** [−0.002, 0.645] | **0.5226** [0.193, 0.746] | **0.4039** [0.044, 0.671] | **0.4531** [0.270, 0.604]; cluster [0.287, 0.593] |

Reading the trajectory honestly:

- The strip **lowers every per-year partial** (0.421→0.365, 0.685→0.523,
  0.539→0.404): part of DPI v1's OAA-aligned variance was being carried by
  staff peripherals, exactly as audit finding 7 predicted. This is the bias
  removal working, not signal loss to be argued away — but it also means the
  defensible core shrinks as the confound is removed.
- The pooled raw-v1 comparison (0.4064 → 0.4531) is **not** a like-for-like
  read: stage-1 residuals are season-centered for free, and centering alone
  moves pooled v1 to 0.5725. Against the centered baseline the strip lowers
  pooled partial r 0.5725 → 0.4531. Per-year columns are unaffected by
  centering and are the clean comparison.
- Interim K2 position: pooled 0.4531 is above the 0.30 line; the closest
  per-year value is 2023 at 0.3647 (p = 0.052, n = 30). Both remaining
  adjustments (park, speed) are expected to remove further shared variance;
  headroom above the line is 0.15 pooled.

All partial-r values are descriptive on 2023–25 team-seasons; DPI scores are
OOS w.r.t. the frozen xOut train window; the stage-1 fit itself is
within-season in-sample (decomposition, not forecast).

Artifacts: `k2_interim_partial_r.csv`, `stage1_regressions_by_season.csv`,
`stage1_team_season_variance_share.csv`, `game_dpi_v2_stage1_2015_2025.csv`,
`team_season_dpi_v2_stage1.csv`, `pitching_strip_summary.json`.

---

## 2. WS3.3 — Sprint speed on topped/weak contact

### 2.1 Method

Canonical script: `scripts/dpi_v2_speed_xout_det.py` (deterministic total
row order `ORDER BY game_pk, at_bat_number`, uniqueness asserted; test-set
predictions persisted to
`speed_xout_det_test_predictions.parquet` so every number below is
recomputable without retraining — recomputation verified).
Attempt #1 (`scripts/dpi_v2_speed_xout.py`, shared helpers) had no ORDER BY
in its pull; DuckDB scan order is connection-dependent, so its 80/20 split
is not reproducible across processes. Its numbers are internally valid and
are reported as corroboration, not as the record. A follow-up subset-CI
script rebuilt the split in a fresh process, landed a different one, and
scored the persisted variant on rows overlapping its own training rows —
those subset deltas are leakage-inflated, recorded as INVALID in
`speed_xout_summary.json.first_attempt_status`, and must never be quoted;
the script was deleted.
Source: new `sprint_speed` DuckDB table (C0 ingest, Savant leaderboard via
`pybaseball.statcast_sprint_speed`, min_opp=10, seasons 2015–2026; ingest log
`results/ingest/sprint_speed_ingest_20260810T154343Z.json`; same-season
batter join coverage vs BIP rows 96.3–99.5% per season).

- **Feature** (`sprint_speed_gbweak`, module helpers
  `defensive_pressing.sprint_speed_applicable_mask` /
  `build_sprint_speed_feature` — NEW functions, production code untouched):
  batter's same-season sprint speed, active **only** on grounders
  (`bb_type='ground_ball'`) or weak contact (EV < 59 mph — Statcast
  "weakly hit" ceiling; approximation of Savant's topped/weak xBA cohort).
  Non-applicable rows and unmatched batters get the **measured** league mean.
- **League mean, measured not assumed:** 27.1015 ft/s (BIP-weighted mean over
  matched applicable rows, 2015–2022 train window). Per-season player-level
  means run 26.812 (2020) – 27.232 (2022). One global value everywhere, so
  the feature encodes zero season-level information. Known limitation:
  unmatched batters are disproportionately pitchers-batting / cup-of-coffee
  players who are slower than league mean; league-mean imputation
  overestimates them (documented, per-plan recipe).
- **Coverage** (train window): 44.1–46.9% of feature-complete BIP rows are
  applicable; 95.75–98.54% of applicable rows match a same-season speed row
  (`speed_feature_coverage_by_season.csv`).
- **Monotonic constraint:** `monotonic_cst=[0,0,0,0,-1]` — a faster runner
  can never raise P(out). Verified empirically on 2,000 applicable holdout
  rows × 6-point speed grid (23→30.5 ft/s): PASS.
- **Training:** frozen recipe (2015–2022, HistGB 200 iter / depth 6 /
  lr 0.05 / seed 42, stratified 80/20). CPU only, `OMP_NUM_THREADS=6`.

**Data-snapshot caveat (measured, material):** today's DB yields 918,292
feature-complete 2015–2022 BIP rows for the identical frozen-recipe query
that yielded 837,571 at freeze time (2026-04-18 validation log). Interim
backfills added ~80.7K rows. All comparisons below are internally consistent
(all models scored on the same current-snapshot holdout of 183,659 rows),
but the frozen model's recorded 0.8936 AUC belongs to the April snapshot.

**Comparison design:** the frozen checkpoint's own April split indices are
not reconstructible, so the frozen model scored on today's holdout may
overlap its own train rows (inflating its AUC — a bias *against* the speed
variant). A 4-feature control retrained on the identical rows/seed is the
clean apples-to-apples baseline; the frozen row ships as reference.

### 2.2 Results — holdout AUC (2015–2022 internal 20% holdout, not future-season OOS)

Canonical (deterministic run, `speed_xout_det_summary.json`; n_test =
183,659 of 918,292 feature-complete rows; dataset = pitches 2015–2022 +
sprint_speed table):

| model | overall AUC (n=183,659) | GB/weak subset (n=83,042) | non-applicable (n=100,617) |
|---|---|---|---|
| frozen xOut (as loaded, reference) | 0.89539 | 0.80407 | 0.93386 |
| control, 4 features, same split | 0.89526 | 0.80304 | 0.93399 |
| **speed variant, 5 features** | **0.89509** | **0.80559** | **0.93296** |

Deltas, variant − control unless stated (paired bootstrap over test rows,
1,000 draws, seed 42; same-split pairing, so CIs are for the delta itself):

- **Overall: −0.00018, 95% CI [−0.00033, −0.00001].** The sprint-speed
  feature does NOT add overall holdout AUC; the point estimate is
  marginally negative and the CI excludes zero.
- **GB/weak applicable subset: +0.00256, CI [+0.00199, +0.00315]** — a
  real but small gain exactly where the Savant recipe says speed matters.
- Non-applicable subset: −0.00103, CI [−0.00116, −0.00091] — the gain is
  paid back (and slightly more) on rows where the feature is a neutral
  constant, a capacity/regularization effect at the frozen recipe's fixed
  200 iterations.
- Overall, variant − frozen-as-loaded: −0.00031, CI [−0.00051, −0.00010]
  (reference only; frozen may overlap its own April train split).

Attempt #1 corroborates (same cohort, different split): overall −0.00000
[−0.00016, +0.00016]; subset point deltas +0.00153 / −0.00047 (no valid
subset CIs — see 2.1).

Reported exactly as landed: the Savant-recipe prior (GB xBA r² 0.46→0.57
with speed) did **not** translate into an overall xOut AUC gain in this
design. The +0.0026 subset gain is real (CI excludes zero) but confined to
the ~45% applicable slice and nets out slightly negative overall. For scale
only: K1's materiality bar for the 3.5 alignment features is 0.002 overall
holdout AUC — the speed feature's overall delta (−0.00018) is an order of
magnitude below that bar and of the wrong sign (K1 does not govern this
feature).

### 2.3 Artifacts

- **Canonical:** `models/defensive_pressing/xout_v2_speed_det_2026_08_10.pkl`
  (sha256 `31af3e3ea5b1dc2e…`), registered as **experimental** registry
  version `defensive_pressing/v2026.08.10-speed.det` (write-once dir,
  hash_policy pinned, git_sha d3dbe23).
- Attempt #1: `models/defensive_pressing/xout_v2_speed_2026_08_10.pkl`
  (sha256 `f6a523e7f70d1d31…`), version `v2026.08.10-speed` — kept in the
  registry history as the superseded nondeterministic attempt.

Production and frozen_validated aliases untouched;
`scripts/verify_artifacts.py` green (ok=15, warn=0, fail=0) after both
registrations. No existing artifact was modified; frozen `xout_v1.pkl`
hash re-verified byte-identical (`e689bff6ab069474…`) before every use.

---

## 3. WS3.5 — The positioning thesis test (xOut-A vs xOut-B) — C1b

Audit DPI finding 8: the "pressing"/positioning framing was never tested
against the `if_fielding_alignment` / `of_fielding_alignment` columns.
DRS-PART blueprint (SIS 2020: Positioning = team credit, Range = player
credit). Script: `scripts/dpi_v2_alignment_ab.py`; all design choices were
pre-declared in its docstring before any result was computed.

### 3.1 Method

- **xOut-A** — ball features only (the frozen recipe's 4:
  `launch_speed, launch_angle, spray_angle, bb_type_encoded`).
- **xOut-B** — A + an 11-column alignment block: `stand_R`; IF one-hots
  `if_shift / if_shade / if_strategic` (vs Standard; NaN where the column
  is NULL — HistGB routes missingness); OF one-hots `of_strategic /
  of_other` ("4th outfielder" + "Extreme outfield shift", 742 rows
  league-wide); `pull_spray` = spray_angle signed toward the batter's pull
  side (the stand × spray interaction); and four alignment × pull_spray
  interactions.
- **Era split EXACTLY at 2023-03-30** (shift ban). DB check: the earliest
  2023 game on record is 2023-04-01, so pre-ban = 2015–2022 (918,292
  core-complete BIP rows), post-ban = 2023–2025 (382,184). 2026 (in-season
  lockbox) excluded. Alignment coverage 97.97–99.86% per season
  (IF-missing: 0.805% of pre-ban rows, 0.369% of post-ban rows); `stand`
  100%. Post-ban has zero "Infield shift" rows (banned), as expected —
  shade/strategic remain.
- **Identical protocol per era for A and B:** deterministic total order
  (`ORDER BY game_pk, at_bat_number`, uniqueness asserted), stratified
  80/20 split seed 42, frozen-recipe HistGB (200/6/0.05), CPU
  (`OMP_NUM_THREADS=6`). Consistency check: pre-ban xOut-A holdout AUC
  0.89526 equals C1a's 4-feature control on the same cohort/split exactly.
- **Positioning value** = mean(P_B − P_A) per defending team-season, over
  ALL era rows scored with the era models (train rows included —
  descriptive, partially in-sample; documented). Split-half: `game_pk % 2`
  halves within team-season, per-season cross-team Pearson r,
  Spearman-Brown corrected; **era alpha = Fisher-z mean of the per-season
  Spearman-Brown values** (the WS3.1/A5 convention). Sensitivity:
  era-pooled season-centered variant.

### 3.2 Results — holdout AUC, B minus A (per-era internal 20% holdout, not future-season OOS)

| era | n_test | AUC A | AUC B | **delta B−A** | 95% CI (paired bootstrap, 1,000 draws, seed 42) |
|---|---|---|---|---|---|
| pre-ban (2015–2022) | 183,659 | 0.89526 | 0.90716 | **+0.0119** | [+0.0113, +0.0125] |
| post-ban (2023–2025) | 76,437 | 0.89434 | 0.90782 | **+0.0135** | [+0.0126, +0.0144] |

The alignment block adds ~0.012–0.013 holdout AUC in BOTH eras — an order
of magnitude above K1's 0.002 materiality line, and it adds MORE after the
shift ban than before (shade/strategic positioning still carries
information). Dataset: `pitches` 2015–2025 core-complete BIP; test
predictions persisted per era
(`alignment_ab_test_predictions_{pre,post}_ban.parquet`).

### 3.3 Results — per-team positioning value and its split-half reliability

Positioning value scale (per-BIP P(out) units, team-season level):
pre-ban mean +0.00034, sd 0.00324, range [−0.0077, +0.0106]; post-ban mean
+0.00009, sd 0.00163, range [−0.0042, +0.0045]. The shift ban compressed
the cross-team spread by ~half — directionally consistent with positioning
freedom being constrained.

Split-half (game_pk % 2, Spearman-Brown corrected, cross-team r per season):

| era | per-season Spearman-Brown | **era alpha (Fisher mean)** | pooled season-centered SB (sensitivity) |
|---|---|---|---|
| pre-ban | 0.62, 0.64, 0.45, 0.53, 0.53, 0.32 (2020, 60-game), 0.42, 0.39 | **0.4958** | 0.4617 (n=240 team-seasons) |
| post-ban | 0.46, −0.17 (2024), 0.55 | **0.3088** | 0.3557 (n=90 team-seasons) |

Both era alphas are below 0.5. In-sample/OOS status: positioning values are
descriptive quantities computed with within-era models (train rows
included); the split-half correlation itself is an internal-consistency
measure, not a forecast. Artifacts: `positioning_value_team_season.csv`,
`positioning_half_aggregates.csv`, `positioning_split_half.csv`,
`alignment_ab_summary.json`.

### 3.4 Artifact

The four era models (pre/post × A/B) are bundled in ONE new artifact
`models/defensive_pressing/xout_v2_alignment_ab_2026_08_10.pkl` (sha256
`d6fab7444f191f5d…`), registered write-once as experimental version
`defensive_pressing/v2026.08.10-alignment-ab` (pinned). Aliases untouched.

---

## 4. WS3.4 — Park, jointly estimated (MixedLM) + the final DPI_v2 pipeline — C1b

*(section written after the pre-declared script `scripts/dpi_v2_park_mixedlm.py`
ran; the primary-number pre-commitment below was declared in the script
docstring BEFORE any fit was run)*

### 4.1 Method

Plan 3.4's preferred design, adopted: mixed model on xOut residuals,
`residual ~ (1|home_park) + (1|fielding_team_season)`, empirical-Bayes
shrinkage via statsmodels MixedLM (crossed REs via the single-group
variance-components trick; statsmodels 0.14.6). **DPI_v2 = the team-season
BLUP; park falls out as its own estimand.** Park is NOT stacked
independently on batted-ball mix (plan-cited 0.696 confound): the response
already conditions on EV/LA/spray/bb_type through the xOut model, and
park + team are estimated jointly.

- **Grain:** game-level stage-1 residuals (52,498 team-games), not 1.2M
  BIP rows — the plan's "if intractable, subsample" branch resolved by
  aggregation (DPI is additive over BIP, so the team/park information is
  preserved; a BIP-grain dense VC design would be ~1.2M × 360).
  Unweighted game rows (production parity: team-season DPI is an
  unweighted mean over games).
- **Park proxy** = the game's `home_team` (100% populated; production's
  own proxy). Franchises that moved parks in-window (TEX/ATL/OAK-ATH)
  collapse to one level — documented limitation.
- **Optimizer validated before the real fit:** on synthetic data with
  known effects, lbfgs stopped short of the REML optimum
  (converged=False, worse llf); bfgs/powell/cg agreed on the optimum and
  recovered the realized truth (park BLUP vs truth r = 0.993). bfgs used.
- **Two pipelines:**
  - **P1 "strip+park" (frozen):** C1a's stage-1 residuals (frozen xOut)
    → MixedLM → team BLUP. Isolates the park increment on the C1a
    trajectory.
  - **P2 "strip+park+speed" (FINAL):** game DPI re-scored 2015–2025 with
    the registered speed variant `v2026.08.10-speed.det` (hash-verified
    against its manifest; production-parity aggregation identical to
    A5's `score_game_dpi` — n_bip≥5, round-3, zero-feature-row → 0),
    staff peripherals joined from the C1a CSV (52,498/52,498 matched,
    0 dropped), per-season stage-1 OLS refit, MixedLM → team BLUP
    = **DPI_v2_final**.
- **PRE-COMMITTED PRIMARY K2 NUMBER** (declared in the script docstring
  before any fit ran): partial r(P2 team BLUP, OAA | BABIP-against),
  pooled 2023–25, from the FULL-WINDOW (2015–2025) MixedLM fit.
  Sensitivity (reported, not primary): MixedLM refit on 2023–25 rows
  only.

### 4.2 P2 re-score with the speed variant

Game DPI from the speed variant correlates r = 0.99705 with the
frozen-model game DPI over all 52,498 team-games (the WS3.3 finding —
the speed feature barely moves the expectation — carries through to game
grain). Stage-1 per-season R² on the speed-scored DPI: 0.0244–0.0367,
mirroring C1a's 0.0243–0.0374. Artifact:
`game_dpi_speed_2015_2025.csv`. In-sample/OOS: 2023–25 scores are OOS
w.r.t. the variant's 2015–2022 train window; 2015–2022 scores overlap
its own train rows (they inform the park RE and their own seasons'
BLUPs only).

### 4.3 MixedLM fits (all converged)

| fit | var(park) | var(team_season) | var(residual) | game-level shares (park / team / resid) |
|---|---|---|---|---|
| P1 strip+park (frozen), 2015–25 | 0.0410 | 0.0259 | 3.5031 | 1.15% / 0.73% / 98.12% |
| **P2 final strip+park+speed, 2015–25** | 0.0399 | 0.0256 | 3.4842 | 1.13% / 0.72% / 98.15% |
| P2 sensitivity, 2023–25 only | 0.0421 | 0.0093 | 3.2406 | 1.28% / 0.28% / 98.44% |

Game-level DPI is luck-dominated (~98% residual — expected; cf. the
"Solving DIPS" 44%-luck figure at season grain). At the effect level,
**park variance exceeds team-season defensive variance** (0.040 vs 0.026
pooled window) — park was a material un-modeled confound in DPI v1, as
audit finding 7 anticipated. The 2023–25-only fit shows much smaller
team_season variance (0.0093; its optimizer emitted a
boundary-of-parameter-space note — reported as-is), consistent with the
positioning-freedom compression seen in §3.3.

**Park estimand** (BLUP, DPI-per-game units, P2 fit): sd 0.194 across the
30 parks; most negative COL −0.689 (Coors converts fewer BIP to outs than
the ball profile predicts — expected sign), most positive STL +0.267.
Full table: `park_effects_mixedlm.csv`. Team-season BLUPs:
`team_season_dpi_v2_blup.csv` (2023–25 BLUP sd 0.105 DPI/game,
EB-shrunk).

### 4.4 K2 measurement — partial r(DPI_v2, OAA | BABIP-against)

Covariates from `2025_validation/team_rankings_all_years.csv` (identical
to C1a §1.3); n = 30 teams/year, 90 pooled; Fisher-z CIs, pooled rows
also carry the pairs cluster bootstrap over the 30 franchises (5,000
draws, seed 42). Full table: `k2_final_partial_r.csv`.

| pipeline | 2023 | 2024 | 2025 | pooled 2023–25 |
|---|---|---|---|---|
| P1 strip+park (frozen) | 0.4112 [0.053, 0.676] | 0.5362 [0.211, 0.754] | 0.4346 [0.081, 0.691] | 0.4805 [0.303, 0.626]; cluster [0.300, 0.607] |
| **P2 FINAL strip+park+speed** | **0.3959** [0.034, 0.666] | **0.5309** [0.204, 0.751] | **0.4213** [0.065, 0.682] | **0.4698** [0.290, 0.618]; cluster [0.285, 0.600] |
| P2 sensitivity (2023–25-only fit) | 0.3457 [−0.024, 0.632] | 0.4266 [0.071, 0.686] | 0.4441 [0.093, 0.697] | 0.4299 [0.244, 0.586]; cluster [0.278, 0.542] |

Trajectory across the full v2 pipeline (pooled 2023–25, like-for-like
season-centered basis):

| stage | pooled partial r |
|---|---|
| v1 season-centered (C1a baseline) | 0.5725 |
| + pitching strip (C1a stage 1) | 0.4531 |
| + park, jointly estimated (P1) | 0.4805 |
| **+ speed variant expectation (P2 = DPI_v2_final)** | **0.4698** |

The pitching strip removes shared variance (0.5725 → 0.4531); the park
adjustment + EB shrinkage then *raises* every per-year partial
(0.3647/0.5226/0.4039 → 0.4112/0.5362/0.4346) — removing park noise that
was unrelated to recorded defense cleans the OAA alignment rather than
eroding it. The speed-variant expectation trims ~0.011 off the pooled
value. All partials are descriptive on 2023–25 team-seasons; the
underlying 2023–25 DPI scores are OOS w.r.t. both expectation models'
train windows; stage-1 OLS and the MixedLM are within-window
decompositions, not forecasts.

---

## 5. K1 / K2 kill-criterion quantities (measurement only — Batch D adjudicates)

This section lists the exact quantities the pre-registered criteria (plan
§8) adjudicate on, exactly as they landed. No adjudication language
appears here.

### 5.1 K1 quantities (WS3.5, §3)

K1 (verbatim): *"if xOut-B (alignment features) adds < 0.002 holdout AUC
over xOut-A in BOTH eras AND per-team positioning value fails split-half
reliability (α < 0.5), the 'pressing/positioning' thesis is dead …"*

| quantity | pre-ban (2015–2022) | post-ban (2023–2025) |
|---|---|---|
| holdout AUC delta, B − A | **+0.0119** (0.01189), CI95 [+0.0113, +0.0125], n_test 183,659 | **+0.0135** (0.01348), CI95 [+0.0126, +0.0144], n_test 76,437 |
| K1's referenced AUC line | 0.002 | 0.002 |
| positioning-value split-half **alpha** (Fisher mean of per-season Spearman-Brown) | **0.4958** (8 seasons: 0.62/0.64/0.45/0.53/0.53/0.32/0.42/0.39) | **0.3088** (3 seasons: 0.46/−0.17/0.55) |
| alpha, pooled season-centered sensitivity | 0.4617 (n=240) | 0.3557 (n=90) |
| K1's referenced alpha line | 0.5 | 0.5 |

Measured relations, stated numerically: the AUC deltas are above the
0.002 line in both eras (each ≈6–7× the line, CIs excluding 0.002); the
split-half alphas are below the 0.5 line in both eras (pre-ban by 0.004,
post-ban by 0.19). Dataset: `pitches` 2015–2025 core-complete BIP; AUC on
per-era internal 20% holdouts (not future-season OOS); positioning values
descriptive (in-era models, train rows included).

### 5.2 K2 quantities (WS3.2–3.4, §1 + §4)

K2 (verbatim): *"if after pitching-strip + park + speed the partial
r(DPI_v2, OAA | BABIP) pooled 2023–25 falls below 0.30, DPI's 'defense'
claim narrows to 'descriptive BIP-outcome residual'; flagship status
reviewed."*

- **PRIMARY (pre-committed before fitting): DPI_v2_final = strip + park +
  speed (P2, full-window fit): pooled 2023–25 partial r = 0.4698**;
  Fisher CI [0.2900, 0.6177]; team-cluster bootstrap CI [0.2847, 0.5998];
  n = 90 team-seasons. Per-year: 2023 = 0.3959 [0.034, 0.666];
  2024 = 0.5309 [0.204, 0.751]; 2025 = 0.4213 [0.065, 0.682] (n=30 each).
- K2's referenced line: 0.30. The pooled point estimate is 0.17 above the
  line; the Fisher lower bound sits 0.010 below it and the cluster lower
  bound 0.015 below it (the criterion's wording references the pooled
  partial r itself).
- Sensitivity (2023–25-only MixedLM fit): pooled 0.4299 [Fisher 0.2435,
  0.5857; cluster 0.278, 0.542].
- Context (strip+park, frozen expectation, no speed): pooled 0.4805
  [Fisher 0.3026, 0.6261; cluster 0.3004, 0.6070].
- Dataset: DPI_v2 team-season BLUPs vs recorded team OAA / BABIP-against
  (`team_rankings_all_years.csv`); 2023–25 DPI scores OOS w.r.t. the
  expectation models' 2015–2022 train windows; stage-1 and MixedLM are
  within-window descriptive decompositions.

---

## 6. Registry ledger for this document (C1a + C1b)

| version | artifact | hash_policy | role |
|---|---|---|---|
| `v2026.04.18` | `xout_v1.pkl` (sha256 `e689bff6ab069474…`) | pinned | frozen_validated alias — untouched |
| `v2026.08.10` | `xout_2026_inseason.pkl` | advisory | production alias — untouched |
| `v2026.04.23-weather` | `xout_v2_weather.pkl` (sha256 `76bdb28dd44b54ad…`) | pinned | **RETIRED** experimental weather variant, registered by C1b to close the Batch-B critic note. Fitted 2026-04-23 on the frozen recipe (2015–2022); recorded holdout AUC 0.8925 vs frozen 0.8936 — no improvement, never promoted, no alias ever pointed at it. Retirement stated in its manifest notes. |
| `v2026.08.10-speed` | `xout_v2_speed_2026_08_10.pkl` | pinned | superseded nondeterministic WS3.3 attempt (C1a) |
| `v2026.08.10-speed.det` | `xout_v2_speed_det_2026_08_10.pkl` (sha256 `31af3e3ea5b1dc2e…`) | pinned | canonical WS3.3 speed variant (C1a); the P2 expectation model in §4 |
| `v2026.08.10-alignment-ab` | `xout_v2_alignment_ab_2026_08_10.pkl` (sha256 `d6fab7444f191f5d…`) | pinned | WS3.5 A/B probe bundle (C1b) |

`scripts/verify_artifacts.py` after all C1b registrations: **ok=18 warn=0
fail=0**. No existing artifact modified; production / frozen_validated
aliases untouched throughout.
