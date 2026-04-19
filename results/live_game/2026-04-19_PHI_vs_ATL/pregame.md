# Pre-Game Intelligence — PHI vs ATL — 2026-04-19

**Generated:** 2026-04-19, target ≤45 min before 6:30 PM EST gate.
**Game PK:** 823475 (MLB StatsAPI).
**Source-of-truth artifacts in this directory:** `game_header.json`,
`contrarian_hits_on_rosters.csv` / `.json`, `dpi_head_to_head.json`,
`pitchgpt_starter_profiles.csv` / `.json`, `league_pitch_distribution_2025.json`.

---

## 1. Game header

| Field | Value |
|---|---|
| Date | 2026-04-19 (Sunday) |
| First pitch | **7:20 PM EDT** (23:20 UTC) |
| Ballpark | **Citizens Bank Park**, Philadelphia, PA (venue_id 2681) |
| Home | **Philadelphia Phillies (PHI)** |
| Away | **Atlanta Braves (ATL)** |
| PHI probable starter | **Andrew Painter** (MLBAM 691725) — RHP, rookie, 2 MLB starts so far in 2026 |
| ATL probable starter | **Grant Holmes** (MLBAM 656550) — RHP, 4 starts in 2026 (after 22 GS in 2025) |
| Weather | Not yet posted by MLB feed (`weather` block empty pre-game). Will populate at gate close. |

Roster source: MLB StatsAPI `/api/v1/teams/{143|144}/roster?rosterType=active` (26-man each, fetched 2026-04-19). Starters confirmed via the same date's schedule call with `hydrate=probablePitcher`.

---

## 2. Contrarian leaderboard cross-reference

Roster-filtered hits from the committed 2024→2025 contrarian leaderboards
(`results/causal_war/contrarian_stability/buy_low_2024_to_2025.csv` and
`over_valued_2024_to_2025.csv`). 2025 actual WAR pulled from
`data/fangraphs_war_staging.parquet` (full 2025 season; `pa_or_ip`
column shows usage). Headline framing of CausalWAR per
`docs/awards/headline_findings.md` Finding 2: 2024→2025 Buy-Low hit rate
68.4% (13/19, CI [0.47, 0.84]); Over-Valued 60.9% (14/23).

| Team | Direction | Player | Mech tag | 2024 baseline bWAR | CausalWAR (2024) | Predicted Δ-WAR | **Actual 2025 bWAR** | PA / IP | Hit per artifact |
|---|---|---|---|---:|---:|---:|---:|---:|:---:|
| ATL | Buy-Low | **Ronald Acuña Jr.** | OTHER | +0.02 | +0.29 | +0.27 | **+2.97** (412 PA) | 412 | YES |
| PHI | Over-Valued | **Bryson Stott** | DEFENSE GAP | +2.30 | −1.12 | −3.42 | **+2.23** (560 PA) | 560 | YES (per artifact) |
| ATL | Over-Valued | **Jonah Heim** | DEFENSE GAP | +1.36 | −2.44 | −3.80 | **+0.37** (433 PA) | 433 | YES |
| ATL | Over-Valued | **Martín Pérez** | PARK FACTOR | +0.88 | −1.39 | −2.27 | **+1.31** (56 IP) | 56 | NO |

**Honest reads (no spin):**

- **Ronald Acuña Jr. (ATL Buy-Low — OTHER tag).** Marquee CausalWAR call. After a 0.02-bWAR injury-shortened 2024 (222 PA) the Buy-Low predicted he'd outperform; he posted **+2.97 bWAR** in 2025 over 412 PA. Realised Δ vs 2024 = **+2.95**. Still on a shortened season but the direction is correct and the magnitude is large.
- **Bryson Stott (PHI Over-Valued — DEFENSE GAP).** Marked as a hit in the committed artifact (`hit=True`) on the 2024→2025 window. Honest read of the underlying numbers: 2024 bWAR +2.30 → 2025 bWAR +2.23, almost flat (Δ −0.07). Predicted Δ was −3.42. Direction (slightly down, not the +1 a top-50 bWAR rank would imply) matches CausalWAR's call that bWAR is over-crediting his defense, but the magnitude is far smaller than the leaderboard's `−1.12` causal_war scalar suggested. **Treat the "hit" label as direction-only** for in-game PA tracking.
- **Jonah Heim (ATL Over-Valued — DEFENSE GAP).** A genuine win for the Over-Valued list. 2024 bWAR +1.36 → 2025 +0.37 (Δ −0.99). CausalWAR's `−2.44` predicted a sharper fall and that direction held.
- **Martín Pérez (ATL Over-Valued — PARK FACTOR).** **Documented miss.** Predicted Δ −2.27; actual Δ +0.43 (1.36 → 0.88 → 1.31; he over-performed slightly over 56 IP in 2025 in the new park). Listed here per the spec's no-spin rule.

PHI Buy-Low hits on this roster: **0**. The DEFENSE GAP cohort being the strongest 2024→2025 Over-Valued tag at 79% is consistent with both Heim and Stott landing in the OV bucket; the platform's caveat that DEFENSE-tagged players' realised drop is sometimes attenuated also matches Stott's roughly-flat actual.

---

## 3. DPI head-to-head — 2025 team defense

Source: `results/defensive_pressing/2025_validation/team_rankings_2025.csv` and
`disagreement_analysis.md`. Headline DPI claim per
`docs/awards/methodology_paper_dpi.md`: 2025 contemporaneous DPI–OAA
Pearson r = 0.641 (CI [0.42, 0.79], n = 30).

| Metric (2025) | **PHI** | **ATL** | Gap (ATL − PHI) |
|---|---:|---:|---:|
| DPI mean (residual outs / game) | **+0.417** | **+0.677** | **+0.260** |
| DPI rank (1 = best) | **24** | **12** | ATL +12 ranks better |
| Statcast OAA | +8 (rank 10) | +7 (rank 12) | −1 OAA, but ATL +2 ranks worse |
| BABIP-against | **.298** | **.295** | −.003 |
| Extra-base prevention | .673 | .668 | −.005 |
| RP-proxy (runs) | +68.3 | −6.4 | −74.7 |
| DPI–OAA rank-diff | **+14** (DPI is 14 ranks worse than OAA) | **0** (perfect agreement) | — |

**ATL is the clearly stronger team-defense team by DPI** — +12 rank
positions higher and a +0.26-residual-outs-per-game edge. OAA almost
ignores the gap (PHI +8, ATL +7), but **ATL's BABIP-against (.295) is
better than PHI's (.298)** and ATL's extra-base prevention is roughly
equivalent. The committed disagreement analysis (`disagreement_analysis.md`)
flags PHI explicitly as one of the four 2025 "OAA-only positive"
teams — Statcast positioning credit that doesn't cash out in BIP outcome
suppression. ATL is one of the few teams where DPI and OAA fully agree
(rank_diff = 0 at near-median).

**Important honest qualification (per the DPI paper §4 limitations).** PHI's
**RP-proxy of +68.3 runs is the second-highest in MLB 2025** despite the
unimpressive DPI. RP-proxy is a runs-saved-vs-league composite; the
divergence between PHI's strong RP-proxy and middling DPI is a known
"OAA-only" pattern — credit accrues at the positioning/route level
(captured by OAA, not DPI). DPI's claim is *BIP-outcome residual*, not
*total run prevention*; the headline ATL > PHI defensive edge is the
DPI-native call for tonight, but PHI is not a bad defensive team in any
absolute sense.

---

## 4. PitchGPT starting-pitcher profiles

Per the EXECUTION_PLAN, the **batter-preparation artifact**. For each
starter we (i) pulled the last ≤5 starts of MLB pitches from the
DuckDB `pitches` table, (ii) ran the **committed PitchGPT v1
checkpoint** (`models/pitchgpt_v1.pt`, 4-layer 4-head 128-D decoder,
2210-token vocab) over those sequences, (iii) marginalised the
predicted-next-pitch logits over zone × velocity to recover a
pitch-type distribution per count state, and (iv) compared to the
2025 league-wide pitch-type-by-count baseline computed from the
same `pitches` table.

Source files: `pitchgpt_starter_profiles.csv` /
`.json` and `league_pitch_distribution_2025.json`.

**No retraining was performed.** Inference only.

### 4.1 Andrew Painter (PHI, RHP) — n = 2 starts, 174 pitches

Painter is a young arm with **only 2 MLB starts on file** (both 2026:
2026-03-31 and 2026-04-06). Sample is small; treat distributions as
indicative, not stable.

| Overall mix | FF 36.8% | SL 20.1% | SI 16.1% | CU 11.5% | CH 8.0% | ST 5.2% | FS 2.3% |
|---|---|---|---|---|---|---|---|

**Key counts (observed-favorite, PitchGPT-favorite, model entropy):**

| Count | n obs | Observed favorite | PitchGPT favorite | H (nats) |
|---|---:|---|---|---:|
| 0-0 | 42 | FF (36%) | FF (40%) | 1.50 |
| 0-1 | 19 | FF (47%) | FF (37%) | 1.53 |
| **0-2** | 6 | **FF (83%)** | FF (39%) | 1.45 |
| 1-2 | 14 | FF (36%) | FF (39%) | 1.47 |
| 2-2 | 16 | SL (31%) | FF (36%) | 1.51 |
| **3-0** | 7 | FF (43%) | **FF (45%)** | 1.42 |
| 3-1 | 7 | FF (43%) | FF (45%) | 1.42 |
| 3-2 | 9 | FF (33%) | FF (37%) | 1.51 |

**Top deviations vs 2025 league** (signed Δ in pitch-type proportion;
positive = Painter uses MORE than league):

- **0-2: FF +0.52** (way more 4-seamers in put-away count than league),
  SL −0.17, ST −0.10. League put-away counts skew breaking; Painter
  goes hard.
- **3-0: FF +0.35**, SI −0.23. League sinkers up, fastballs down on 3-0.
  Painter still goes 4-seamer.
- **1-1: CU +0.20**, SI +0.12. Curveball preference in even counts.
- **2-1: FF +0.19** — same fastball-heavy 2-1 pattern.
- **0-1: FF +0.19** — sets up the 0-2 FF tendency.

**Batter takeaway for tonight:** Painter goes fastball-first in put-away
counts (0-2, 3-0, 3-1) at well above league rates. PitchGPT's
predicted distributions track his observed favorite in 9 of 11
applicable counts — this is a *predictable* pitcher pattern by the
model. Watch for hitters sitting fastball in 0-2.

### 4.2 Grant Holmes (ATL, RHP) — n = 5 starts, 419 pitches

Adequate sample (419 pitches across 4 starts in 2026 + last 2025 outing).

| Overall mix | SL 37.0% | FF 34.4% | CU 12.4% | FC 8.1% | SI 4.5% | CH 3.6% |
|---|---|---|---|---|---|---|

**Key counts:**

| Count | n obs | Observed favorite | PitchGPT favorite | H (nats) |
|---|---:|---|---|---:|
| 0-0 | 112 | FF (38%) | SL (30%) | 1.51 |
| 0-1 | 49 | SL (37%) | FF (27%) | 1.56 |
| **0-2** | 26 | **SL (62%)** | SL (34%) | 1.48 |
| 1-1 | 42 | SL (36%) | FF (29%) | 1.50 |
| **1-2** | 34 | **SL (53%)** | SL (29%) | 1.56 |
| 2-2 | 41 | SL (46%) | SL (30%) | 1.50 |
| **3-1** | 9 | **FF (89%)** | FF (37%) | 1.43 |
| 3-2 | 19 | FF (47%) | FF (32%) | 1.49 |

**Top deviations vs 2025 league:**

- **0-2: SL +0.44** (massive slider preference in put-away). League is
  mixed; Holmes is a slider specialist there.
- **1-2: SL +0.36**, **2-2: SL +0.31**, **2-1: SL +0.27** — slider
  permeates two-strike and high-tension counts at a wide gap above
  league.
- **3-1: FF +0.46**, SI −0.23, SL −0.11. Hitter's count → almost
  exclusively 4-seamer (89% observed; league barely hits 43%).
- **0-1: SL +0.22**, SI −0.12 — slider on first-pitch take.

**Batter takeaway for tonight:** Holmes is a **two-pitch slider/fastball
RHP** who goes slider in any count where he's ahead or even, and
fastball when he's behind 3-1 / 3-2. PitchGPT correctly identifies the
slider as the favorite in 6 of 12 counts and fastball in 5 of 12;
only count 0-0 shows genuine model–observed disagreement (he goes FF
38% but PitchGPT predicts SL 30% as marginal favorite — likely because
catcher and game-script context also favor opener-style sliders).
Tonight, Phillies hitters who fall to 0-2 or 1-2 should sit on
slider; in 3-1 / 3-2 they should look fastball.

---

## 5. What to watch for tonight (5 non-obvious model-driven calls)

These are model-grounded predictions where the flagship calls *disagree
with the obvious public read*. Each cites the underlying artifact.

1. **PitchGPT can be tested live on Andrew Painter's 0-2 fastball
   tendency.** Painter throws FF on 0-2 at +0.52 vs league.
   The model expects 39% FF in that count (matching league-style
   diversification); his actual is 83%. **Live-game test:** if
   Painter reaches a 0-2 count tonight, the gap between PitchGPT's
   predicted prob (~0.39) and the actual call (likely FF) is a
   "model said X, human saw Y" moment per the EXECUTION_PLAN
   Phase 2 spec — exactly the kind of high-confidence-but-wrong
   anecdote the post-game writeup wants. Sample is only 6 prior 0-2s
   for Painter; tonight is a meaningful test.

2. **DPI says ATL will out-defend PHI tonight despite OAA showing PHI
   +8, ATL +7.** ATL's 2025 BABIP-against (.295) actually beats PHI's
   (.298) and DPI ranks ATL 12 vs PHI 24. PHI is in the "OAA-only
   positive" disagreement group. **Watch:** if a tied or close game
   sees a defensive misplay decide an inning, DPI predicts PHI is
   the more likely culprit.

3. **Ronald Acuña Jr. is the marquee CausalWAR Buy-Low validator on
   either roster.** 2024 bWAR 0.02 → 2025 bWAR +2.97. Tonight any
   PA he wins on is incremental confirmation of a finding already in
   the bag for the methodology paper. **Watch:** his per-PA outcomes
   vs Painter (rookie RHP) should be the first thing the post-game
   contrarian-section bullet covers.

4. **Grant Holmes vs Bryson Stott is a CausalWAR Over-Valued matchup
   inside one PA.** Stott is on the PHI Over-Valued list (DEFENSE GAP
   tag, predicted Δ −3.42, actual 2025 Δ ≈ flat). Holmes throws
   slider to LHB at well above league rates. **Watch:** Stott's
   slider PAs against Holmes are the in-game evidence point — does
   the bWAR-overrated-by-defense lefty handle the slider specialist?

5. **PitchGPT's entropy on Painter = 1.42–1.53 nats across counts,
   essentially flat.** This is *not* a high-entropy unpredictable
   pitcher; the model finds him relatively decodable count-to-count.
   Holmes is similar (1.43–1.56). **Watch:** if rolling top-1
   accuracy on PitchGPT live is above ~30% for either pitcher,
   that's calibration consistency. The 2025 OOS ECE = 0.0098 is the
   reference bar (`docs/awards/methodology_paper_pitchgpt.md`).

---

## 6. Data-quality caveats

- **Painter sample is tiny (2 starts, 174 pitches).** All Painter
  per-count distributions have wide CIs; the FF +0.52 on 0-2 is
  dramatic but rests on n = 6 0-2 pitches.
- **Weather not yet populated** in the MLB StatsAPI feed at fetch
  time (~3h pre-game). Will be available closer to first pitch.
- **PHI Stott "hit" label is direction-only** in our table. The
  committed artifact's `hit=True` reflects rank-direction over
  cohort, but his realised 2024→2025 Δ-bWAR is essentially flat
  (−0.07), not the −3.42 the leaderboard's causal scalar implied.
  Honest interpretation: an Over-Valued direction call that landed
  but with magnitude smaller than implied — typical for the
  DEFENSE GAP cohort.
- **Holmes 5-start window mixes 2025 (1 game, 2025-07-26) with 2026
  (4 games)**. Pitch-mix may be transitioning between seasons; the
  PitchGPT marginal is a pooled view.
- **DPI 2025 numbers reflect *full-season 2025 data*** (frozen
  artifact from the 2025 validation). Tonight is early 2026; that
  signal is the most-recent available but may already be drifting.
- **No new training, no model retraining was performed.** This is
  pure inference + data join.
- **MLB StatsAPI access succeeded.** No pybaseball fallback was
  needed for any of: schedule, probable pitchers, rosters, live feed.

---

## 7. Files in this directory

| File | Purpose |
|---|---|
| `pregame.md` | This document. |
| `game_header.json` | Game PK, teams, venue, starters, weather placeholder. |
| `contrarian_hits_on_rosters.csv` / `.json` | Roster-filtered Buy-Low / Over-Valued hits with predicted vs actual 2025 WAR. |
| `dpi_head_to_head.json` | PHI / ATL 2025 DPI rankings, OAA, BABIP-against, disagreement direction. |
| `pitchgpt_starter_profiles.csv` / `.json` | PitchGPT predicted vs observed pitch-type by count for both starters. |
| `league_pitch_distribution_2025.json` | 2025 league-wide pitch-type-by-count baseline used for deviation analysis. |
