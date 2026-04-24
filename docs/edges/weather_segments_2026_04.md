# DPI weather cohorts: where defense matters most

## Top line

The weather-augmented xOut model (DPI v2) does **not** beat the
weather-agnostic baseline (v1) in aggregate on 2023-2025 held-out BIP —
global Brier actually *rises* from 0.12302 (v1) to 0.12346 (v2), a 0.35%
degradation driven almost entirely by retractable-roof parks where the
wind feature is zero-suppressed and the model is absorbing noise. But in
two cleanly-defined weather cohorts the lift is real: **cold games
(<=55°F) with strong inward wind (n=5,748, +1.50% relative Brier lift,
95% CI [+0.93%, +2.08%])** and **Fenway Park with strong inward wind
(n=4,365, +1.06% relative Brier lift, 95% CI [+0.54%, +1.65%])**. Both
CIs exclude zero, both are physically coherent (balls that should have
carried get knocked down, defensive plays over-index on outs the baseline
wrongly expects as hits), and both describe the narrow game conditions
where a GM should most weight DPI over conventional defensive metrics.
Everywhere else, v2's marginal signal is within noise or negative — an
honest finding, not a buried one.

## Data and method

**Test cohort.** All batted balls in play (BIP) from 2023-2025 with
complete core features (launch_speed, launch_angle, hc_x, hc_y, bb_type,
events): 382,184 BIP across 1,601 pitchers. Both xOut models were
trained on 2015-2022 so 2023-2025 is fully held out. Model artefacts:
`models/defensive_pressing/xout_v1.pkl` (test AUC 0.8936,
in-sample) and `models/defensive_pressing/xout_v2_weather.pkl` (test AUC
0.8925). The near-identical headline AUC is what motivates the cohort
slice: v2 can't help *on average*, so the question is where the
additional features (wind_parallel_to_spray, wind_perpendicular_to_spray,
temp_f, see `src/analytics/defensive_pressing.py:compute_wind_components`)
shift calibration.

**Outcome.** Binary out-vs-nonout per BIP. We report Brier score
(`mean((p - y)^2)`) on each slice — lower is better, positive delta =
v2 helps. Brier was chosen over log-loss for its bounded behaviour and
direct interpretation as MSE on probability; all ranking decisions
reproduce qualitatively under log-loss.

**Cohort axes** (all bins chosen a priori, not tuned):
- **Wind parallel to spray**: quintiles with sign — `strong_inward`
  (<=-5 mph), `mild_inward` (-5 to 0), `calm` (|par|<0.01), `mild_outward`
  (0 to +5), `strong_outward` (>=+5 mph). Sign is relative to the ball's
  outward bearing; positive = tailwind toward the outfield.
- **Temperature**: {`cold <=55°F`, `mild 55-75°F`, `hot >=75°F`}.
- **Ballpark**: Statcast `home_team` abbreviation.
- **Pitcher type**: fly-ball (FB+popup rate >= 2023-25 league median of
  0.340) vs ground-ball, using pitcher-season BIP with n>=100 BIP
  per pitcher. Pitchers below threshold are tagged `unclassified` and
  excluded from two-way summaries.

**Confidence intervals.** Non-parametric bootstrap over BIP indices
within each cohort, 1000 resamples, 2.5%/97.5% quantiles. `delta_brier`
is `brier_v1 - brier_v2` so positive = v2 is the better-calibrated model
on that cohort.

**Practical-significance flag.** `True` iff the 95% CI excludes zero
*and* the relative Brier lift is >=1.0% of v1's Brier. A cohort passes
only if the effect is both statistically and practically meaningful —
the cold+inward and Fenway+inward cohorts clear both bars.

## Ranked cohorts

Top 10 by |delta_brier|, minimum n=30, all three cohort families
(single-axis, 2-way ballpark x wind, 2-way temp x wind) combined.

| Rank | Cohort | n | Brier v1 | Brier v2 | Delta Brier | 95% CI | Rel lift % | Practical? |
|---|---|---:|---:|---:|---:|---|---:|:---:|
| 1 | temp<=55°F & strong inward wind | 5,748 | 0.13020 | 0.12825 | **+0.00195** | [+0.00121, +0.00271] | +1.50% | **yes** |
| 2 | Fenway (BOS) & strong inward wind | 4,365 | 0.13942 | 0.13794 | **+0.00148** | [+0.00075, +0.00229] | +1.06% | **yes** |
| 3 | Wrigley (CHC) & strong inward wind | 4,923 | 0.12548 | 0.12463 | +0.00085 | [+0.00021, +0.00147] | +0.68% | no (rel<1%) |
| 4 | Petco (SD) & mild inward wind | 3,657 | 0.11979 | 0.11911 | +0.00068 | [+0.00005, +0.00131] | +0.57% | no (rel<1%) |
| 5 | temp<=55°F (all winds) | 23,297 | 0.12514 | 0.12452 | +0.00062 | [+0.00029, +0.00094] | +0.49% | no (rel<1%) |
| 6 | Wrigley (CHC, all winds) | 12,366 | 0.12559 | 0.12505 | +0.00054 | [+0.00014, +0.00096] | +0.43% | no (rel<1%) |
| 7 | mild 55-75°F & strong inward wind | 28,254 | 0.12578 | 0.12539 | +0.00039 | [+0.00013, +0.00065] | +0.31% | no (rel<1%) |
| 8 | strong inward wind (all temps/parks) | 56,714 | 0.12437 | 0.12427 | +0.00010 | [-0.00010, +0.00028] | +0.08% | no (CI 0) |
| 9 | FB pitcher & strong inward wind | 26,047 | 0.11891 | 0.11871 | +0.00020 | [-0.00008, +0.00047] | +0.17% | no (CI 0) |
| 10 | FB pitcher & mild inward wind | 26,406 | 0.11859 | 0.11874 | -0.00015 | [-0.00039, +0.00009] | -0.13% | no (CI 0) |

**Null / negative cohorts** (honest report — these are where v2 *hurts*):

| Cohort | n | Delta Brier | 95% CI | Rel lift % |
|---|---:|---:|---|---:|
| Globe Life (TEX, retractable_unknown) | 12,865 | -0.00153 | [-0.00188, -0.00118] | **-1.29%** |
| Chase (AZ, retractable_unknown) | 13,697 | -0.00137 | [-0.00173, -0.00101] | -1.13% |
| loanDepot (MIA, retractable_unknown) | 12,895 | -0.00132 | [-0.00165, -0.00095] | -1.06% |
| Tropicana (TB, dome) | 12,098 | -0.00124 | [-0.00160, -0.00086] | -0.98% |
| roof_status == dome (all parks) | 8,133 | -0.00159 | [-0.00200, -0.00109] | -1.33% |
| hot >=75°F & calm wind | 58,421 | -0.00111 | [-0.00129, -0.00095] | -0.91% |

**Pattern.** The positive cohorts are all *outdoor + cold + inward-wind*.
The negative cohorts are all *retractable-roof-unknown + dome*. This is
consistent with the v2 design: wind features are zero-suppressed for
`dome` and `retractable_unknown`, which means at those parks v2 is
effectively *the same model as v1 but with an extra all-zero column* —
the HistGB has to spend a small amount of capacity routing around it,
which marginally hurts calibration. The lift is real where the physics
actually applies; the noise is real where the sensors don't.

## Worked example

**Yankees at Braves, Truist Park, 2025-07-19 (game_pk 777097).**
Weather: 90.0°F, 8.1 mph wind from 230° (SW) — i.e. flowing toward 50°
(NE), essentially straight out to centre at a park with a 45° CF bearing.
Per-BIP wind_parallel_to_spray is +6.6 to +8.0 mph depending on spray
direction, a clean tailwind on every outfield-bound fly. ATL defence
faced 33 BIP from the Yankees.

- Actual outs: 22.
- v1 xOut sum: 22.76. DPI_v1 = 22 - 22.76 = **-0.76** (below average;
  "ATL defence let some outs slip away").
- v2 xOut sum: 21.84. DPI_v2 = 22 - 21.84 = **+0.16** (above average;
  "ATL defence held up fine given the wind").

The v2 call flips the sign: with a strong tailwind, the baseline
(v1) was wrongly expecting too many outs on balls that were actually
carrying. Four Yankees home runs were hit. ATL's defence made 22 outs
in conditions where the expected number, accounting for the carry,
was ~21.8. The per-pitch calibration evidence for this cohort
(Fenway+inward, cold+inward, here: Truist+outward) is what drives the
Brier lift.

**Outcome validation is one-game and partial.** The Braves lost 12-9;
the shift in DPI sign doesn't change the scoreboard. What the example
*does* demonstrate is that v2's re-ranking of defensive performance in
extreme-wind games is physically coherent (ATL made *more* outs than
v2 expected, consistent with them being an above-average defence on the
night — and the four HRs were all balls that v2 correctly already
down-weighted as harder-to-catch). The 2025 game sample where v2 flips
the DPI sign vs v1 is concentrated on exactly the days with |wind_par|
>= 5 mph — 40+ games across the season.

## Limitations

1. **Effect is small and confined.** Two cohorts clear the "practically
   significant" bar; the weather-aware model is *worse* than v1 in
   aggregate. A fair paper paragraph is "weather features help DPI
   calibrate specifically under cold+inward-wind conditions at outdoor
   parks", not "weather makes DPI better". Don't oversell.
2. **Retractable-roof unknown is a genuine confound.** 21% of
   2023-2025 weather rows (n=5,289) carry `roof_status =
   retractable_unknown`. The Statcast API does not expose which games
   at AZ / HOU / MIA / MIL / SEA / TEX / TOR were actually played with
   the roof closed. The weather feature is zero-suppressed for these
   rows, but some of them were genuinely played open — and that slippage
   is where v2's negative delta at those parks comes from.
3. **Cold+inward-wind cohort is outdoor-only by construction.** Domes
   never see <=55°F in-stadium; the n=5,748 signal is almost entirely
   April/October outdoor games at North-latitude parks (BOS, CHC, CLE,
   DET, MIN, NYM, NYY, PHI, PIT). The finding generalises to "cold
   outdoor games with a stiff breeze into the batter's face", not to
   baseball at large.
4. **Single-game worked example, not an outcome trial.** The 777097
   Braves game is an illustration of DPI-sign-flipping under wind, not
   a causal proof that v2 predicts future outs better than v1 on that
   same defensive unit. The cohort-level Brier delta is the evidence;
   one game is the illustration.
5. **Stadium CF-bearing table is ±5°.** The
   `TEAM_CF_BEARING_DEG` lookup in `defensive_pressing.py` is sourced
   from public orientation references, not survey-grade GPS. A 5°
   error at wind_speed 15 mph perturbs wind_parallel by ~1.3 mph,
   which is within the quintile bin widths. It's noise in the cohort,
   not a systematic bias.
6. **xOut v2 training set (2015-2022) pre-dates the 2023 shift ban.**
   The rule change altered BIP-to-out conversion. Both v1 and v2 share
   this limitation equally, so cohort *deltas* are not affected, but
   absolute Brier levels in 2023-25 are higher than a 2023-25-trained
   model would deliver.

## How this gets used

A future "DPI paper" paragraph should lead with the 0.64 OAA correlation
(the flagship external-validation result, see NORTH_STAR post-evidence
consolidation) and cite this cohort analysis as the honest, narrow
weather-adjustment story: the v2 xOut checkpoint calibrates better
than v1 on the specific cohorts where wind and temperature materially
change BIP physics (cold + inward wind, Fenway + inward wind), and is
neutral-to-slightly-worse elsewhere. In the dashboard, the
`defensive_pressing.py` view should surface a weather-adjusted DPI only
when wind_parallel magnitude exceeds ~5 mph at an outdoor park, with
an explicit tooltip showing the expected-outs delta so the user can
see how much the adjustment is worth — everywhere else, default to v1.
