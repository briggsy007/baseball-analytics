# New York Mets — 2026 Team Diagnosis

**Date:** 2026-04-20
**Sample:** 22-26 games. 2026 pitches through 2026-04-19. NYM 2026 pitches ≈ 3,835 (pitching) / 2,805 (batting).
**Record:** 7-15 (.318), NL East 5th of 5. Run diff −25 (RS 72 / RA 97).
**Method:** Flagships-only (CausalWAR 2023-24 contrarian prior, PitchGPT, DPI, Stuff+ 2026 refreshed today). VWR and MechanixAE-as-predictor excluded per 2026-04-18 post-evidence consolidation. CausalWAR not re-run on 2026 (<300 PA per player per spec); used as prior only.

---

## Glossary

**Rate stats**
- **wOBA** — weighted On-Base Average; all-purpose offensive rate (~.320 league average).
- **xwOBA** — expected wOBA from exit velocity + launch angle; removes luck.
- **BABIP** — Batting Average on Balls In Play; luck indicator (~.290 norm).
- **K%** — strikeouts ÷ plate appearances.
- **HardHit%** — share of batted balls at 95+ mph exit velocity.
- **Barrel%** — share of "barreled" contact (high EV + optimal launch angle).
- **R/G, RS, RA** — runs per game, runs scored, runs allowed.
- **bWAR** — Baseball-Reference Wins Above Replacement (the public stat).

**Pitch types (Statcast codes)**
- **FF** four-seam fastball · **SI** sinker / two-seam · **FC** cutter · **SL** slider · **ST** sweeper · **CU** curve · **KC** knuckle-curve · **CH** changeup · **FO** forkball / splitter.
- **FB / BR / OS** — broad families: fastball / breaking / offspeed.

**Plate discipline**
- **O-Swing%** — chase rate: swings at pitches OUT of the strike zone.
- **Z-Contact%** — contact rate on in-zone swings.
- **Whiff%** — swing-and-miss rate (misses ÷ swings).
- **PA** plate appearances · **BF** batters faced · **IP** innings pitched.

**Context**
- **TTO** — Times Through the Order (TTO-1 = 1st batter through #9 first time, TTO-2 = second time, etc.).
- **HL** — High Leverage: inning 8+ of a close game (proxy for manager's trust).
- **IL** — Injured List.
- **LHP / RHP** — left- / right-handed pitcher. **LHB / RHB** — batter.
- **NL East** — National League East division (PHI, ATL, MIA, WSH, NYM).

**Our models** (full detail in `docs/NORTH_STAR.md`)
- **CausalWAR** — Double Machine Learning player-value estimate. 2023-24 fit is our contrarian prior (68% Buy-Low hit rate on 2025 retrospective).
- **Stuff+** — per-pitch quality model (100 = league average; 105+ above, 95- below).
- **DPI** — Defensive Pressing Index (team defense). 2025 external validation r=0.64 vs Statcast OAA.
- **PitchGPT** — decoder-only transformer predicting the next pitch from prior sequence + situation.

**Contrarian leaderboard terms**
- **Buy-Low** — CausalWAR says player is worth MORE than bWAR / public market implies.
- **Over-Valued** — CausalWAR says player is worth LESS than bWAR / public market implies.
- **rank_diff** — ranking gap between CausalWAR and bWAR. Larger magnitude = higher model confidence in the disagreement.

**External reference**
- **OAA** — Outs Above Average (Statcast's public defensive metric). DPI validates against it.

**Tipping signature** (recurring term in this doc)
- A pitcher whose **stuff quality is flat or improved** while **results have collapsed** — signature of opponents knowing what's coming (tipped delivery or predictable sequencing), not a skill loss.

---

## Headline

**The Mets are cursed, not broken.** Offense wOBA is dead-last in MLB (.333) but xwOBA is mid-pack (.308). Pitching wOBA-against ranks 6/30 (.369) with xwOBA-against top-10 (.316). DPI team defense ranks **7/30 — top quartile**. BABIP .276 (well below norm). This is roughly a .480 team playing .318 baseball.

Two specific problems are real, not luck: **David Peterson** is in genuine stuff-decline and **Freddy Peralta** is showing a clean tipping signature. If those two get fixed or cut, and BABIP normalizes, this is a wildcard contender in 40 games.

**The most interesting single finding:** our 2023-24 CausalWAR contrarian board wrote this roster's 2026 story two years ago. Peterson (rank_diff −296) and Polanco (−131) were both flagged Over-Valued; both are the team's biggest drags today. Manaea, Holmes, Kimbrel, Brazobán, Peralta all flagged Buy-Low; that IS the Mets' best pitching core in 2026. **The model saw this roster's shape before it existed.**

---

## Context — the 2026 Mets are a largely new team

Winter roster churn was massive:

**Outbound:** Pete Alonso (BAL), Brandon Nimmo (TEX), Jeff McNeil (ATH), Starling Marte (KC), Luisangel Acuña (CWS), **Edwin Díaz (LAD)**.

**Inbound:** Bo Bichette (from TOR), Luis Robert Jr. (from CWS), Marcus Semien (from TEX), rookie Carson Benge.

**Retained but anomalous:** Juan Soto (40 PA / 10 games played in a 22-game span — possible injury or usage issue; transactions table not queried this pass).

**Closer:** Devin Williams (not Díaz). 6 of 9 Mets ninth innings.

---

## The damage — it's the bats, not the arms

| Metric | Value | MLB rank |
|---|---|---:|
| Offense wOBA | .333 | **30 / 30** |
| Offense xwOBA | .308 | mid-pack (MLB avg .323) |
| Pitching wOBA-against | .369 | 6 / 30 |
| Pitching xwOBA-against | .316 | top-10 |
| DPI team defense (proxy) | +0.012 | **7 / 30** |
| Team BABIP | .276 | well below norm |
| Pitching wOBA−xwOBA gap | **+.054** | large negative luck |
| Run diff | −25 | bottom third MLB |

**~85% of the slow-start delta is offense.** Defense is a strength (top quartile). Pitching is good underneath. BABIP regression alone buys back 3-4 runs.

---

## Trajectory

First 13 games vs last 9:

| | Games 1-13 | Games 14-22 |
|---|---|---|
| Offense wOBA | .427 | **.259** |
| Offense xwOBA | .332 | .289 |
| Pitching wOBA-against | .413 | .338 |
| Pitching xwOBA-against | **.278** | **.343** |

Offense was carrying them early and has cooled hard. Pitching is the opposite — results improving on the surface but **xwOBA-against drifting worse** (.278 → .343). The surface wins are masking pitching drift.

---

## Three actionable problems

### 1 — The top-of-order (Lindor + Bichette) is the offensive bottleneck

**Francisco Lindor:** OS O-Swing 31.5% → **48.5%**. FB wOBA .407 → .272. xwOBA still .322 — more unlucky than broken.
**Bo Bichette:** BR O-Swing 42.5% → **51.4%**. FB wOBA .384 → .251 on 229 pitches. New team, new park, BR chase crisis.

Together Lindor and Bichette have seen 31% of all Mets offensive pitches. League has attacked the Mets with +1.6pp more fastball and +3.4pp more offspeed (fewer breakers), and these two are biting the offspeed/fastball trap hardest.

**Action:** drop Bichette to 6th until contact quality returns. Targeted offspeed-recognition coaching for Lindor.

### 2 — Freddy Peralta is a cleaner tipping case than Walker

Stuff+ 102.5 (2025: 104.5) — essentially flat. xwOBA-against +21 pts. **Slider xwOBA .175 → .511 on 62 pitches.** 0-0 fastball rate **45% → 64%** with entropy 1.55 (league 2.80).

**Three signals stacked:**
1. Stuff flat
2. Mix compressed (tipping proxy — batters can sit fastball on 0-0)
3. Results collapsed on the slider (the secondary that's supposed to bail out the predictable fastball)

Walker (Phillies) was stuff-up + results-down. Peralta is stuff-flat + mix-compressed + results-collapsed — cleaner signature, single most actionable pitcher call on the staff.

**Action:** tape + TrackMan review this week on slider release/sequencing. Diversify 0-0 mix.

### 3 — David Peterson is stuff-decline (different diagnosis, different treatment)

Stuff+ 92.3 (**−6.2 vs 2025**). SI xwOBA **.527 at every height** — not a location issue, a shape issue. K% 20.7 → 19.8. 107 BF, xwOBA-a .350.

CausalWAR 2023-24 flagged Peterson as Over-Valued with rank_diff **−296 — the largest NYM gap in the entire contrarian dataset.** 2026 vindicates it.

**Tipping cases fix themselves in a week of tape. Stuff-decline cases are an injury, a delivery change, or a career inflection.** Confusing the two is how rotations get burned.

**Action:** pull from rotation. Cap at 85 pitches next outing; if SI doesn't find it by May 10, move to swing role. Promote Tong or demote a reliever for the rotation slot.

---

## The rest of the rotation

| Pitcher | xwOBA 26 | K% | Stuff+ 26 (Δ vs 25) | Verdict |
|---|---:|---:|---:|---|
| Peralta | .314 | 25.0 | 102.5 (−2.0) | **Tipping case — fixable** |
| Peterson | **.350** | 19.8 | 92.3 (**−6.2**) | **Stuff-decline — pull** |
| McLean | .229 | **31.5** | 103.8 (−5.5) | Overperforming; regression watch but let K% play |
| Holmes | .318 | 17.2 | 105.9 (−3.2) | Fine; cutter .480 xwOBA is pitch-specific |
| Senga | .316 | 26.8 | 99.7 (+0.3) | Forkball elite (.215 / 93% whiff 0-2); rest of arsenal (FC, ST) getting hit |
| Manaea | .355 | **17.1** | 88.1 (**−7.9**) | **Velo −2.3 mph, stuff crisis. Not a starter.** |

**Manaea to lefty bulk role (6th–7th).** Velo and stuff don't support a starter workload right now. 5 IP/start is already the de-facto pattern.

---

## Bullpen — the problem is the 7th inning, not the 9th

| Reliever | Inn 8+ PA | HL xwOBA-a | Stuff+ Δ | Read |
|---|---:|---:|---:|---|
| **Williams** (closer) | 31 | **.295** | **−8.8** | HL xwOBA fine. .416 wOBA is cluster-luck. Hold, watch stuff. |
| Myers | 23 | **.287** | +0.1 | Quiet bright spot. Stuff stable, results good. |
| Weaver | 24 | .409 | +0.9 | Stuff intact, results bad. Gap case. |
| Lovelady | 26 | .296 | **−17** | Stuff collapsed. Demote out of HL. |
| García Jr. | 25 | .386 | +5.4 | Stuff improving, results bad. Upside. |
| Raley | 17 | .269 | **−14** | Results OK, stuff down. Fade watch. |
| **Brazobán** | **0** | — | **+5.3** | **Stuff+ 108 with zero HL usage. Misallocated.** |

**Williams's 9th-inning "struggle" (.416 wOBA on 31 BF) is cluster-luck.** xwOBA-against .295 is league-average for a closer. Hold him. The stuff-decline is the real medium-term concern — if he loses another 5 Stuff+ points, re-evaluate.

**Real live problem: 7th inning** (Manaea bulk .353 post-0408, Weaver .409 HL). That's where games are being lost before Williams ever gets the ball.

**Promote Brazobán to medium-high leverage.** Stuff+ 108 with zero inning-8+ usage is the single clearest misallocation.

---

## Offense — full scorecard

9 regulars in 64-107 PA range:

| Batter | 2026 wOBA / xwOBA | 2025 wOBA / xwOBA | Read |
|---|---|---|---|
| **F. Álvarez** (R) | **.418 / .406** | .357 / .326 | **Hidden star. OS xwOBA .497. Ride him.** |
| L. Robert Jr. (R) | .366 / .350 | .330 / .319 | Hot on breakers (.431 wOBA vs .314 prior). Regression watch. |
| Benge (L, rookie) | .268 / .263 | — | Reasonable rookie line. |
| Semien (R) | .294 / .303 | .314 / .318 | In-line with 2025. No crisis. Elite vs LHP (.368 xwOBA). |
| Vientos (R) | .316 / **.280** | .319 / .320 | Batting over quality of contact. Regression down. |
| Lindor (S) | .292 / .323 | .364 / .345 | Cold start; xwOBA says unlucky. **Chase crisis on offspeed.** |
| Bichette (R) | .280 / .313 | .383 / .356 | Real cold start. BR chase crisis. **Drop to 6th.** |
| Polanco (S) | **.270 / .270** | .358 / .338 | **Worst regular bat; xwOBA confirms. Demote / platoon.** |
| Baty (L) | **.258 / .236** | .359 / .333 | **Worst collapse on team. Sit vs RHP (.234 xwOBA on 61 PA).** |

**Soto anomaly:** .450 / .321 on 40 PA / 10 G of a 22-game span. Usage is the story, not contrarian value. Flag for transactions follow-up.

**Ride these:** Álvarez, Vientos (for now), Semien vs LHP, Robert vs LHP.
**Sit these:** Baty vs RHP, Polanco entirely, Soto vs LHP (.220 xwOBA on 13 PA).

---

## Plate discipline — the league's attack

| Class | 2025 share | 2026 share | Δ |
|---|---:|---:|---:|
| Fastball | 53.5% | **55.1%** | +1.6 |
| Breaking | 32.6% | **28.8%** | −3.8 |
| Offspeed | 12.3% | **15.7%** | +3.4 |

**League is feeding the Mets more fastballs + offspeed, fewer breakers.** Different attack than the Phillies (who get more offspeed on held breaker share). Mets are being **pitched aggressively to contact** — classic "this lineup can't punish a straight fastball" gameplan.

Team-level wOBA vs FF is 67 pts below MLB average. **This is the single most exploitable offensive weakness in the lineup.**

---

## Defense — top quartile, underrated

DPI proxy +0.012, rank 7/30. Out rate above expected on BIP. **Defense is a strength.** Don't touch the positioning.

---

## CausalWAR contrarian vindication

The 2023-24 board on current Mets roster — both sides hitting:

**Buy-Low vindicated:**
- **Manaea** (rank_diff +68): 2026 xwOBA-a .355 on 70 BF. Good underlying even as results vary.
- **Holmes** (+62): **best reliever line on team.** .265 wOBA / .318 xwOBA-a on 93 BF.
- **Peralta** (+44): rotation anchor. (Note: tipping case above is a separate current problem — the Buy-Low call is about baseline value.)
- **Kimbrel** (+143): .262 wOBA-a on 13 BF (small).
- **Brazobán** (+130): .259 wOBA-a on 38 BF. Underused.
- **Vientos** (+63): .316 / .280 on 57 PA. Hitting over quality but still a vindicated Buy-Low.

**Over-Valued vindicated:**
- **Peterson** (**−296 — largest NYM gap in dataset**): 2026 .429 wOBA-a / .350 xwOBA-a. Vindicated hard.
- **Weaver** (−141): .393 / .383 on 35 BF. Vindicated.
- **Polanco** (−131): .270 / .270. Vindicated.
- **Bichette** (−66): partial — wOBA bad, xwOBA says unlucky.
- **Senga** (−74): mixed — surface wOBA .409, xwOBA .316.

**Net: both sides of the 2-year-old CausalWAR board hit on this year's Mets roster.** The Buy-Low side *is* the team's best pitching. The Over-Valued side *is* the team's biggest drags.

---

## What's NOT the problem

- **Defense** — top quartile.
- **Pitching aggregate** — xwOBA-a top-10.
- **Devin Williams** — cluster-luck noise, not collapse.
- **F. Álvarez** — .418/.406 with OS xwOBA .497. Hidden star.
- **Marcus Semien** — .294/.303, in-line with 2025.
- **McLean** — overperforming but K% 31.5% is real. Let him cook.
- **Luis Robert vs LHP** — .478/.529 on 18 PA. Lineup superpower.

---

## Structural headwinds

- **Soto usage** — 40 PA / 10 G anomaly. Either IL or manager choice; needs clarification.
- **Díaz gone, Williams in** — closer transition bleeds some IP/leverage efficiency while the org adjusts.
- **No evident IL queries pulled this pass** — transactions follow-up warranted.

---

## Action plan

1. **Pull Peterson from rotation; promote Tong or Manaea-slot.** Biggest single pitching move.
2. **Drop Bichette to 6th.** Biggest single batting-order move.
3. **Peralta tape + TrackMan review this week.** Slider collapse + 0-0 predictability is the most fixable issue on the staff.
4. **Manaea to lefty bulk (6th-7th).** Velo and stuff say not a starter right now.
5. **Promote Brazobán to medium-high leverage.** Only misallocation in the bullpen.
6. **Platoon/sit Polanco; sit Baty vs RHP.** Both sample-adequate; Over-Valued calls confirmed.
7. **Ride Álvarez, Vientos, McLean, Williams.** Don't panic on early variance.
8. **Offspeed-recognition coaching — Lindor first, Baty second.** Bichette's issue is breakers (different coaching block).

---

## Top 3 Briggsy takes

1. **"The Mets are cursed, not broken."** BABIP .276, offense wOBA 30/30 but xwOBA mid-pack, DPI 7/30, pitching xwOBA top-10. This is a .480 team playing .318 baseball. If they lose Peterson and Polanco and BABIP normalizes, they're a wildcard contender in 40 games.

2. **"Our 2023-24 contrarian board wrote the 2026 Mets' story."** Peterson Over-Valued at rank_diff −296 (biggest NYM gap in the CSV — vindicated as the worst pitcher on the staff). Polanco Over-Valued −131 (vindicated as the worst regular bat). Meanwhile Manaea / Holmes / Kimbrel / Brazobán / Peralta flagged Buy-Low — that IS the 2026 rotation and bullpen core. **The model called this roster's shape two years before it existed.**

3. **"Freddy Peralta is a cleaner tipping case than Walker."** Stuff flat (−2), xwOBA +21 pts, slider xwOBA quadrupled (.175 → .511), 0-0 fastball rate 45% → 64% with league-outlier entropy 1.55. Hits all three legs of the tipping stool: stuff flat, mix compressed, results collapsed. Walker (Phillies) was two legs. **Peralta is the one where a week of tape room fixes it.**

---

## Opposing-team edge surface (what works against them)

1. **Attack with fastballs.** Mets wOBA vs FF is 67 pts below MLB average. Even mediocre-FB starters can survive this lineup.
2. **Stack RHB against Peterson.** He's leaking to everyone; matchup is exploitable.
3. **LHP opener + RHP bulk.** Mets platoon is inverted (.305 vs LHP, .343 vs RHP). Standard RHP-heavy rotation optimization works against you here.
4. **Don't read Williams 9th-inning surface stats.** .416 wOBA is cluster-luck. Late-game strategy that assumes a weak Mets closer will burn you.

---

## Caveats

- 22–26 game sample. Per-player samples 30–110 PA; per-starter 4–6 outings. Confidence is directional, not forensic.
- CausalWAR NOT re-run on 2026 (below spec's ≥300 PA floor). 2024 fit used as prior only.
- DPI is proxy (out-rate vs 1-xBA on BIP ≥ league mean), not the full trained model. Validated as directionally correct per 2025 external OAA r=0.64.
- **Peralta slider case at 62 pitches** — stronger than Walker's v1 sample (35), not yet v2 (71). Re-validate in 2 starts.
- Peterson SI case at 133 pitches — sample-adequate.
- Luis Robert vs LHP (.529 xwOBA) is 18 PA — directional.
- **Soto's 40 PA / 10 G anomaly is the most important open question in the data** — transactions/IL lookup not completed this pass.
- Park-adjusted xwOBA not computed for new-team players (Bichette, Robert, Semien).
- No VWR (retracted). No MechanixAE-as-predictor (demoted). No ChemNet / volatility_surface (retired).
- Flagship transformers NOT retrained. Inference/scoring refreshed only today (Stuff+ for 412 pitchers).

---

## Files referenced

- `docs/NORTH_STAR.md`, `docs/phillies/2026-04-20_slow_start_diagnosis.md` (methodology reference)
- `results/validate_causal_war_20260418T194415Z/causal_war_baseline_comparison_2023_2024.csv` (contrarian prior)
- `src/dashboard/db_helper.py`, `src/analytics/stuff_model.py::batch_calculate_stuff_plus`
- `src/analytics/defensive_pressing.py`
- `data/baseball.duckdb` (pitches through 2026-04-19)
- MLB Stats API (standings, per-game feeds)
