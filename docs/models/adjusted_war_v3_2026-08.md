# AdjustedWAR v3 — regularized joint estimation (WS4.2), 2026-08-10

**Status (updated 2026-08-10, Batch D): PRODUCTION.** The user adjudicated
the naming and promotion calls: the "Causal" brand is retired for player
value — the product is **AdjustedWAR** on every live surface — and
AdjustedWAR v3 (ridge) is the production player-value model. Sections 1–7
below are the original measurement report, unchanged except for one
corrected word in §6 (see §10). §8 records the K3 verdict, §9 the WS4.7
uncertainty work, §10 the corrections log.

Artifact: `models/adjusted_war_v3/adjusted_war_v3_2026_08_10.pkl`, registered
as `adjusted_war_v3 v2026.08.10` (sha256 `60a62686ebcaf4…`, hash pinned).
Since 2026-08-10 the registry `production` alias points at this version;
`frozen_validated` remains **unset** — no validation spec exists for this
model, so there is no gate suite it could have passed. Code:
`src/analytics/adjusted_war_v3.py` (+ `tests/test_adjusted_war_v3.py`, 5
tests, solver verified against sklearn Ridge to ~1e-10).
Evaluation scripts (protocols pre-registered in their docstrings, written
before any result existed): `scripts/adjusted_war_v3_forward_eval.py`,
`scripts/adjusted_war_v3_boards.py`.

## 1. Design

Per-PA outcome (wOBA value; PA definition and `woba_denom > 0` filter
identical to the CausalWAR extractor) on a sparse one-hot design:

| block | contents | penalty |
|---|---|---|
| batter_id | one column per batter | λ (identity) |
| pitcher_id | one column per pitcher | λ (identity) |
| park × stand | `pitches.home_team` (100% populated; the dead `games` join is never used) × batter handedness, ~60 cols | λ_fe = 10 (dummy-trap resolution only) |
| context FE | times-facing-pitcher (2nd / 3rd+), platoon, month (May–Sep+), temperature bucket (cold <60°F / hot >75°F / **missing as its own level**, never imputed) | λ_fe = 10 |

Solved by normal equations on the sparse Gram (~1.6–2.4k columns per fit;
CPU, seconds per fit). Outcome centered at the fit sample's per-PA mean.
**Identifiability (pre-registered mechanics, decided before evaluation):**
each row carries exactly one batter and one pitcher indicator, so the two
block constants trade off 1:1; both identity blocks are re-centered to
PA-weighted zero (`coef_centered`). Raw offsets are kept as the
centering-drift diagnostic (§4). Ranks are invariant to the shift.
Opponent quality, park, and shrinkage are handled structurally in one
estimation — the Sill 2010 / nflWAR / DRC+ pattern the plan mandated.
WAR-like display scaling reuses the CausalWAR constants (÷1.25 wOBA/run,
÷10 runs/win); ranks are scale-invariant.

## 2. λ selection (season-forward CV; never random K-fold)

Pairs (fit season b → predict season b+1 batter wOBA), PA-weighted RMSE,
follow-up PA ≥ 100. For the forward evaluation: pairs 2015→2016 … 2022→2023
(follow-ups strictly before the held-out seasons). Path (mean RMSE across
pairs):

| λ | 25 | 50 | 100 | 200 | **400** | 800 | 1600 | 3200 | 6400 | 12800 | 25600 | 51200 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RMSE | .04477 | .04225 | .03975 | .03774 | **.03683** | .03715 | .03817 | .03923 | .04000 | .04047 | .04073 | .04087 |

**λ\* = 400** — clean interior optimum. For the board rebuild (§5) λ was
re-selected per window under an expanding rule (only pairs whose follow-up
≤ the board's baseline year); **every one of the 17 windows selected λ = 400**.

## 3. K3 half 1 — season-forward prediction, ridge vs current formulation

Protocol: WS4.4 (`src/analytics/marcel.py`) — PA-weighted RMSE, W-L vs
Marcel with 0.010-wOBA tie band, PA-weighted paired t, ≥ 2 seasons.
**Held-out seasons: 2024 and 2025** (baselines 2023, 2024). OOS status:
`current_v1` = frozen 2015–2022 v1 nuisance (fully OOS for both baselines);
ridge fits use baseline-season PAs only; λ*/C* selected on follow-ups
≤ 2023. Neither held-out season entered any selection step. Pool =
batters with follow-up PA ≥ 100 and predictions from ridge, current, and
Marcel (intersection; baseline PA ≥ 10 via current_v1). Dataset: Statcast
pitch-derived wOBA (`season_batting_stats.woba` is 100% NULL); weights =
follow-up PA.

PA-weighted RMSE (follow-up PA ≥ 100):

| season | n | ridge_1yr | current_v1 | ridge_3yr | shrunk_raw | Marcel | naive |
|---|---|---|---|---|---|---|---|
| 2023→2024 | 409 | .03382 | .04742 | .03361 | .03439 | .03399 | .03797 |
| 2024→2025 | 403 | .03141 | .04383 | .03172 | .03130 | .03175 | .03698 |
| **pooled** | **812** | **.03265** | **.04567** | .03268 | .03289 | .03290 | .03748 |

Direct ridge_1yr vs current_v1 (pooled, the K3 half-1 quantity):
* RMSE delta **−0.013028** (ridge better), in both seasons individually.
* Head-to-head |error| W-L-T: **321-143-348** (n=812).
* PA-weighted paired t: mean |error| improvement +0.00813 wOBA,
  t = 8.58, n_eff ≈ 670, confidence ridge better **≈ 1.0** (1 − 7e-17).

Versus Marcel (pooled, floor 100):
* **ridge_1yr: 178W-133L-501T**, RMSE .03265 vs .03290; paired-t
  confidence 0.567 → **`superiority_claim_allowed = false`** (below the
  90% bar; wins > losses is not enough).
* **current_v1: 175W-320L-317T**, RMSE .04567 vs .03290; paired-t
  confidence ≈ 0 (t = −7.9) — current_v1 loses to Marcel decisively, and
  is worse than the naive constant (.04567 vs .03748).

Sensitivity floors (pooled pattern holds per season; see
`forward_eval.json`): at PA ≥ 300 and ≥ 502 Marcel is slightly better
than ridge (e.g. 2024 floor-502: Marcel .02668 vs ridge .02770); at
floor 100 and no-floor ridge is slightly better. Ridge ≈ Marcel overall;
the decisive gap is ridge vs current, not ridge vs Marcel.

**Shrunk-to-shrunk (DRC+ lesson, diagnostic):** `shrunk_raw` =
league + (raw − league)·PA/(PA+C*), C* = 450 by the same CV. Pooled
.03289 vs ridge .03265 — most of ridge's win over *raw* wOBA is
shrinkage; the adjustment layer adds only ~0.0002–0.0007 RMSE beyond
identically-shrunk raw wOBA. The honest headline is ridge vs current,
not ridge vs raw.

## 4. Diagnostics (fit-level)

| diagnostic | fit 2023 | fit 2024 |
|---|---|---|
| raw centering offset, batter block (pre-recentring) | +0.00626 | +0.00596 |
| raw centering offset, pitcher block | −0.00367 | −0.00417 |
| max |per-park mean fit residual| | 0.00013 | 0.00020 |
| park-skill corr: ridge coef_centered | −0.026 | −0.023 |
| park-skill corr: current_v1 point estimate | +0.047 | +0.087 |
| park-skill corr: raw wOBA (reference) | +0.034 | +0.070 |

(park-skill corr = correlation over batters between the skill estimate and
the batter's primary park's raw scoring environment; ridge removes the
park association, current_v1 retains slightly more of it than raw wOBA.)

## 5. K3 half 2 — board lifts across the WS4.5 backfilled windows

The WS4.5 run (`results/causal_war/backfill_windows_2026-08-10/`,
executed this session from C2a's pre-registered script) produced all 17
planned fully-OOS windows: 9 single-season (2016→17 … 2024→25, honest
per-window nuisance ladder T(2015)…T(2021) + frozen T(2022)) and 8
two-yr-aggregate ({2016,17}→18 … {2023,24}→25). The 2-yr b=2016 window is
structurally unavailable (would need pre-2015 pitch data), as
pre-declared. Ridge boards (`results/adjusted_war_v3/boards_2026-08-10/`)
reuse the **identical pools (0 rows lost in all 17 windows), hit rules,
ITT accounting (spec 5.3), matched-naive construction, and Marcel
boards**; only the ranking score differs. Continuity check: the WS4.5
2023→24 buy-low ITT 0.560 equals the WS4.6 rescoring of the April
artifact (frozen-nuisance path reproduces despite the
sklearn-1.8→1.6 unpickle warning).

Mean fully-OOS ITT lift, percentage points, with 95% t-intervals across
windows (primary mean includes COVID-touched windows as pre-declared;
COVID-excluded is a labeled sensitivity, never the headline):

**vs ITT-consistent matched-naive (full boards, batters + pitchers):**

| config / side | current formulation | ridge (AdjustedWAR v3) |
|---|---|---|
| single-season buy-low (n=9) | **+11.05** [+4.58, +17.51] | **+8.28** [+2.34, +14.22] |
| single-season over-valued (n=9) | **+1.82** [−6.60, +10.25] | **+2.79** [−5.34, +10.91] |
| 2-yr-aggregate buy-low (n=8) | **+8.22** [+0.93, +15.51] | **+8.76** [−1.52, +19.04] |
| 2-yr-aggregate over-valued (n=8) | **+4.72** [−0.71, +10.15] | **+7.29** [+3.34, +11.24] |
| unweighted mean of the four | **+6.45** | **+6.78** |
| COVID-excluded sensitivity | +11.68 / +2.52 / +8.31 / +2.78 | +8.95 / +4.55 / +6.23 / +7.31 |

**vs Marcel (batter channel only — see coverage gap below):**

| config / side | current formulation | ridge (AdjustedWAR v3) |
|---|---|---|
| single-season buy-low (n=9) | **−8.00** [−14.34, −1.66] | **−10.22** [−18.23, −2.22] |
| single-season over-valued (n=9) | **−13.37** [−20.65, −6.10] | **−7.37** [−14.35, −0.39] |
| 2-yr-aggregate buy-low (n=8) | **−5.00** [−12.53, +2.53] | **−5.00** [−17.22, +7.22] |
| 2-yr-aggregate over-valued (n=8) | **−7.83** [−13.31, −2.35] | **−9.83** [−18.70, −0.97] |
| unweighted mean of the four | **−8.55** | **−8.11** |
| COVID-excluded sensitivity | −7.43 / −13.53 / −4.80 / −5.14 | −9.14 / −8.10 / −4.80 / −9.20 |

Per-window values, board CSVs, and per-board bootstrap CIs are in the two
summary JSONs. Every window is fully OOS with respect to its follow-up
season for both formulations; the value scores themselves are estimated
on the baseline season(s) being ranked (true of both formulations by
construction).

**Measured-subset labels (no extrapolation):**
* The Marcel control exists on the **batter channel only** (pitcher
  Marcel is deferred to the marcelR pin, frozen spec §6.7): model and
  Marcel each pick top-25/side from the identical qualified,
  Marcel-projectable batter pool. A full-board (batter+pitcher) Marcel
  control is structurally unavailable this session.
* Window 2016→2017's Marcel uses truncated 2-season history (no
  pitch-level wOBA before 2015), as pre-declared in the WS4.5 protocol.

## 6. K3 block (kill-criterion inputs — measurements only, no adjudication)

> K3 verbatim (plan §8): "if the ridge formulation does not beat the
> current formulation on season-forward prediction AND mean fully-OOS
> board lift (vs matched-naive AND Marcel) across the backfilled windows
> is ≤ 0, contrarian boards lose the edge label permanently and ship as
> descriptive divergence viewers. No post-hoc subgroup rescues."

Measured quantities, exactly as they landed:

1. **Season-forward (held-out 2024 + 2025, pooled n=812, PA-weighted):**
   ridge RMSE .03265 vs current .04567 (delta −0.013028; h2h 321-143-348;
   paired-t confidence ≈ 1.0) — **the ridge formulation beats the current
   formulation**, in both held-out seasons individually. Marcel W-L:
   ridge 178-133 (paired-t conf 0.567, no superiority claim permitted);
   current 175-320.
2. **Mean fully-OOS board lift across the backfilled windows (all 17
   windows measured; 9 single + 8 two-yr):**
   * vs matched-naive: **positive in all four config-sides for both
     formulations** (current: +11.05/+1.82/+8.22/+4.72, mean +6.45;
     ridge: +8.28/+2.79/+8.76/+7.29, mean +6.78) — with four of eight
     config-side t-intervals crossing 0 (see §5 tables).
   * vs Marcel (batter channel only): **negative in all four
     config-sides for both formulations** (current: −8.00/−13.37/−5.00/
     −7.83, mean −8.55; ridge: −10.22/−7.37/−5.00/−9.83, mean −8.11).
   * Windows measurable: all 17 pre-registered windows for the naive
     control; all 17 for the batter-channel Marcel control (2016→17 with
     truncated Marcel history). Full-board Marcel control: not measurable
     this session (pitcher Marcel deferred), labeled above.

Adjudication of these numbers against K3 belongs to Batch D with the user.

## 7. Limitations

* **Ridge ≈ shrunk raw wOBA on forecasting** (.03265 vs .03289): the
  adjustment layer's forecasting edge beyond identical shrinkage is
  small. Its measurable non-forecast advantages are park cleanliness
  (§4) and structural opponent adjustment.
* **Marcel board control is batter-channel only**; pitcher-side board
  lift vs Marcel is unmeasured until the marcelR pitcher pin (frozen
  spec §6.7).
* The board-window λ rule is expanding and leakage-free, but λ's
  stability (400 everywhere) means the rule was never stressed.
* Ridge value scores for a board are estimated on the baseline season(s)
  being ranked — a valuation convention shared with the current
  formulation, not a train/test violation of the follow-up claim.
* Talent-uncertainty and sampling-error layers (WS4.7) are not built
  here; no CIs ship on per-player AdjustedWAR values. **They were built and
  coverage-tested in Batch D — see §9. Both failed the gate, so the "no CIs
  ship" position stands, now as a measured result rather than an omission.**
* The frozen v1 checkpoint unpickles with a sklearn 1.8→1.6
  InconsistentVersionWarning; the scoring path reproduces the WS4.6
  rescoring numbers exactly (2023→24 ITT 0.560), so results are
  unaffected, but the environment pin should be fixed before any
  re-freeze.
* 2020-touched windows carry the pre-declared COVID structural
  distortion; they are included in the primary means (pre-registered)
  with labeled exclusion sensitivities.
* No claims-registry entries are created here (Batch D); nothing in this
  document may be quoted on a dashboard surface until then (K6).
  **Superseded by §8: the Batch D entries `adjusted_war_v3_forward_rmse`,
  `adjusted_war_v3_vs_marcel_forward`, `adjusted_war_v3_naive_lift_17w`,
  `adjusted_war_v3_marcel_lift_17w`, `adjusted_war_boards_k6_framing` and
  `adjusted_war_v3_ci_coverage` now exist in `docs/claims/claims.yaml`, and
  only those numbers may render.**

---

## 8. K3 verdict (2026-08-10, user-adjudicated) — DOES NOT FIRE

**Criterion, quoted verbatim from the platform improvement plan §8:**

> **K3 (CausalWAR pivot, 4.2/4.5):** if the ridge formulation does not beat
> the current formulation on season-forward prediction AND mean fully-OOS
> board lift (vs matched-naive AND Marcel) across the backfilled windows is
> ≤ 0, contrarian boards lose the "edge" label permanently and ship as
> descriptive divergence viewers. No post-hoc subgroup rescues (the −2.8pp
> autopsy pattern is banned by this clause).

**Adjudication.** The criterion is a conjunction; both limbs must hold for
it to fire.

| Limb | Measured | Fires? |
|---|---|---|
| "ridge does not beat the current formulation on season-forward prediction" | Ridge **beats** current: pooled PA-weighted RMSE .03265 vs .04567 (Δ −0.013028), better in *both* held-out seasons individually, h2h 321-143-348, paired-t confidence ≈ 1.0 | **NO** |
| "mean fully-OOS board lift (vs matched-naive AND Marcel) ≤ 0" | vs matched-naive: **positive** in all four config-sides for both formulations (ridge mean +6.78pp, legacy +6.45pp). vs Marcel: negative (ridge −8.11pp, legacy −8.55pp), batter channel only | **NO** (the naive limb is positive, so the AND-conjunction over both controls is not ≤ 0) |

**Verdict: K3 DOES NOT FIRE.** The contrarian boards keep shipping and do
not become descriptive-only viewers.

**What K3 not firing does NOT license.** The Marcel half of the second limb
is unambiguously negative, and the pre-registered WS4.4 protocol denies a
forecasting superiority claim (paired-t confidence 0.567 against a 0.90
bar). The **K6 consequence therefore binds**: every board surface states,
verbatim-equivalent,

> beats matched-naive (+6.5pp mean across 17 fully-OOS windows); does not
> beat the Marcel-picker (−8pp, batter channel); ties Marcel on
> season-forward forecast — no edge claim vs Marcel

registered as claim `adjusted_war_boards_k6_framing`. Exact figures behind
the rounding: naive-lift unweighted means +6.45pp (legacy) / +6.78pp
(ridge); Marcel-lift unweighted means −8.55pp (legacy) / −8.11pp (ridge),
batter channel only. Standing caveats that travel with the verdict: four of
eight config-side t-intervals cross zero (§5), the Marcel control has no
pitcher channel, and ridge's forecasting win over *identically shrunk* raw
wOBA is ~0.0002–0.0007 RMSE (§3) — the promotion rests on beating the
legacy formulation and on structural park/opponent cleanliness, not on
out-forecasting the field.

**Promotion executed on this verdict:** registry alias
`adjusted_war_v3/production = v2026.08.10` (the only alias touched;
`frozen_validated` deliberately unset, with the reason recorded in the
alias history entry in `models/registry.json` and, because manifests are
write-once, in the sidecar
`models/adjusted_war_v3/v2026.08.10/AMENDMENTS.md` — the manifest's own
`notes` field still says "No production/frozen_validated alias set", which
was true at registration and is superseded by that amendment).
`scripts/precompute.py::precompute_adjusted_war` resolves the scoring model
through the alias, and every cached row carries `scoring_model` /
`scoring_artifact_version` / `scoring_artifact`.

**The disclosure is rendered, not merely stamped.**
`src/dashboard/views/causal_war.py::_render_scoring_provenance` reads those
columns back and states on the page which model produced the displayed
numbers — the same pattern as
`views/defensive_pressing.py::_render_artifact_provenance`. A frame with no
stamp is reported as legacy-produced rather than as "unknown": stamping and
the promotion landed on the same day, so an unstamped cache necessarily
predates both. This matters right now — the live `leaderboard_cache` row for
`causal_war` was computed 2026-08-09 by the legacy formulation, and without
this the page would render legacy numbers under a promoted-model banner.

**Column parity across the two scoring paths.** The ridge fit produces
neither the legacy `park_adj_woba` (`raw_woba` + mean residual) nor
`traditional_war`. The ridge branch therefore emits `context_neutral_woba`
(fit-sample league mean + PA-weighted-centered batter coefficient — the
module's own forward predictor) under its **own** label rather than
overloading `park_adj_woba` with a different construction, and joins
season-stat WAR via `_traditional_war_frame` so the comparison-scatter and
Biggest-Movers tabs keep working. The view labels whichever adjusted-wOBA
column is present.

The frozen 2026 boards and every pick already in `predictions/picks.jsonl`
are **not** rescored — they resolve under the frozen resolution spec against
the legacy scores they were frozen with.

---

## 9. WS4.7 — uncertainty done right (two layers, coverage-gated)

Script: `scripts/adjusted_war_v3_uncertainty.py` (protocol pre-registered in
its docstring, written before execution). Run 2026-08-10, CPU only, DuckDB
`read_only=True`, no artifact writes. Outputs:
`results/adjusted_war_v3/uncertainty_2026-08-10/{uncertainty_coverage.json,
coverage_rows.csv}` (json sha256
`e4b8a30f443a8f9d274186250b614e5eedd2766801015cf41ef0f4fbd89be4ef`).

### 9.1 The two layers (never conflated)

**(a) Sampling error — openWAR pattern.** Resample, with replacement, the
PAs a batter actually took. The ridge normal equations make this exact
rather than approximate: for a one-hot batter block, row *j* of
`(X'X + Λ)β = X'y_c` reduces to

    β_j = Σ_{i ∈ PA(j)} r_i / (n_j + λ),
    r_i = y_c,i − (x_i'β − β_j)      (partial residual)

i.e. the batter coefficient *is* the sum of that batter's partial residuals
(outcome minus the fitted pitcher / park×stand / context contributions),
shrunk by `n_j + λ`. The bootstrap resamples those n_j partial residuals
and recomputes the ratio. **B = 2000 replicates**, seed 42, percentile
(2.5 / 97.5) interval. Conditional on the other blocks being held at their
fitted values — stated, not hidden: that is precisely what "sampling error
in the player's own PA set" means.

**(b) Talent uncertainty — the λ-implied ridge posterior.** Ridge is the
posterior mode of `y = Xβ + ε`, `ε ~ N(0, σ²)`, with the λ-implied Gaussian
prior `β_identity ~ N(0, σ²/λ)`; posterior covariance `σ²(X'X + Λ)⁻¹` with
`σ̂² = RSS/(n − edf)`, `edf = trace(G(G+Λ)⁻¹)`. Because the *reported*
coefficient is PA-weight-centered within the batter block, its posterior
variance is computed exactly as `S_jj − 2(Sw)_j + w'Sw`. Gaussian
±1.959964 sd.

Fit diagnostics: 2023 — n = 184,177 PA, p = 1,589 cols, σ̂ = 0.5175,
edf = 538.1; 2024 — n = 184,241 PA, p = 1,577 cols, σ̂ = 0.5102,
edf = 535.8.

The two layers are reported side by side and are never summed into one
unlabelled "the CI" (the DRC+ bagging category error).

### 9.2 Coverage validation (the gate, pre-registered)

Frames: the Batch C forward-eval prediction frames
(`predictions_2023_to_2024.csv`, `predictions_2024_to_2025.csv`), primary
pool follow-up PA ≥ 100 — identical to §3. Test: does the batter's realized
**next-season season-aggregate wOBA** fall inside the nominal 95% interval
placed around the season-*b* prediction `league_b + coef_centered`?
Reproduction check: this script's ridge solve reproduces Batch C's
`pred_ridge_1yr` to 5.6e-17 / 1.1e-16 max absolute difference.

**Ship gate, set before the run:** a layer's 95% interval may render on any
surface only if that layer's pooled empirical coverage lands in
**[90%, 98%]**.

| construction | 2023→24 (n=409) | 2024→25 (n=403) | pooled (n=812) | Wilson 95% | mean half-width (wOBA) | ships? |
|---|---|---|---|---|---|---|
| **(a) sampling error** | 0.4743 | 0.5186 | **0.4963** | [0.462, 0.531] | 0.02308 | **NO** |
| **(b) ridge posterior** | 0.7017 | 0.7246 | **0.7131** | [0.681, 0.743] | 0.03738 | **NO** |
| *diag:* (a)⊕(b) quadrature | 0.7702 | 0.8164 | 0.7931 | [0.764, 0.820] | 0.04436 | not a candidate |
| *diag:* posterior ⊕ follow-up sampling noise | 0.9242 | 0.9479 | 0.9360 | [0.917, 0.951] | 0.06754 | not a candidate |

**Verdict: NO CONFIDENCE INTERVAL SHIPS.** Both shipping candidates miss the
gate decisively — layer (a) covers half of what it claims, layer (b) about
three quarters. On the display scale the withheld intervals were ±0.75 WAR
(sampling) and ±1.06 WAR (posterior) on average, so this is not a cosmetic
omission: shipping them would have implied roughly 2× more precision than
the data supports.

**Where the gap lives.** The only construction that reaches nominal is the
labelled diagnostic that *also* carries the follow-up season's own sampling
noise (93.6%, inside [90%, 98%]). That locates most of the shortfall in
unmodelled next-season variation rather than in a mis-specified posterior —
but it was pre-registered as a diagnostic, not a shipping candidate, and it
does not become one by having scored well. Two honest limits stated with
it: (i) this is a *predictive* coverage test, while the intervals nominally
cover the season-*b* value parameter, so under-coverage here is evidence
that the intervals mislead as forecasts, not proof that the season-*b*
estimate is wrong; (ii) a forecast interval that only works because it adds
next-season noise would need its own pre-registration and its own held-out
test before it could ship.

**Consequences executed now.** `src/dashboard/views/causal_war.py` gates
every interval on the claim (`_CI_MAY_SHIP`, currently `False`): the CI Low
/ CI High leaderboard columns, the single-player error bars and the forest
CI bars are all withheld and replaced by the measured-coverage note. The
plotting code is retained and flips back automatically if a construction
ever passes. `precompute_adjusted_war` emits no `ci_low` / `ci_high`
columns. Legacy CausalWAR bootstrap intervals are covered by the same
position — they were never coverage-validated at all.

This closes spec **Ticket 4**, the "coverage-validate before any CI ships"
item that had been skipped in every prior run while CIs shipped anyway.

---

## 10. Corrections log

* **2026-08-10 (Batch D).** §6 item 2 read "three of eight config-side
  t-intervals crossing 0"; the correct count is **four of eight** — the
  crossing intervals are legacy single-season over-valued [−6.60, +10.25],
  legacy 2-yr over-valued [−0.71, +10.15], ridge single-season over-valued
  [−5.34, +10.91] and ridge 2-yr buy-low [−1.52, +19.04] (§5). Corrected in
  place; the §5 tables the count summarizes were always right.
* **2026-08-10 (Batch D).** Status header updated from EXPERIMENTAL to
  PRODUCTION and the "working name" language removed, per the user's
  adjudicated naming + promotion decision. §§1–7 measurements are
  otherwise unchanged.
* **2026-08-11 (Batch D review fix).** §8 previously described the
  provenance stamp as sufficient ("every cached row carries `scoring_model`
  … so a surface can state which model produced the number"). The stamp was
  written but no surface read it: the AdjustedWAR page rendered
  legacy-produced cache rows with no statement of origin on the same day the
  ridge promotion was announced on that page.
  `_render_scoring_provenance` now renders it, with a legacy fallback for
  unstamped caches, and §8 says so. Same pass: the ridge scoring branch had
  silently dropped `park_adj_woba` and never produced `traditional_war`,
  which would have hollowed out three of the page's four tabs on the first
  real precompute run; it now emits `context_neutral_woba` (its own,
  correctly-labelled construction) and joins season-stat WAR. No measured
  number in §§1–7 or §9 changed.
