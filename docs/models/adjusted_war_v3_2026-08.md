# AdjustedWAR v3 — regularized joint estimation (WS4.2), 2026-08-10

**Status: EXPERIMENTAL.** Working name only. The naming decision for the
CausalWAR product surface ("Causal" brand retirement vs methods-note) is
pre-registered as the **user's call in Batch D** (plan 4.2) and is NOT made
here. This document reports measurements; K3 adjudication language is
deliberately absent — adjudication is Batch D's job.

Artifact: `models/adjusted_war_v3/adjusted_war_v3_2026_08_10.pkl`, registered
as `adjusted_war_v3 v2026.08.10` (sha256 `60a62686ebcaf4…`, hash pinned,
**no production / frozen_validated alias set**). Code:
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
     ridge: +8.28/+2.79/+8.76/+7.29, mean +6.78) — with three of eight
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
  here; no CIs ship on per-player AdjustedWAR values.
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
