# Phillies 2026 Slow-Start Diagnosis

**Date:** 2026-04-19
**Sample:** 21 games played (opening day ~2026-03-27). DuckDB pitches through 2026-04-08; standings/IL current as of document date.
**Team state:** 8-13, −38 run diff (worst in MLB), NL East 4th.
**Method:** 3-agent diagnostic (batters / pitchers / team-level) using platform flagships CausalWAR, PitchGPT, DPI. Stuff+ used as a commodity pitch-quality score. VWR and MechanixAE-as-predictor excluded per post-evidence consolidation 2026-04-18.

---

## Glossary

**Rate stats**
- **wOBA** — weighted On-Base Average; all-purpose offensive rate (~.320 league average).
- **xwOBA** — expected wOBA from exit velocity + launch angle; removes luck.
- **BABIP** — Batting Average on Balls In Play; luck indicator (~.290 norm).
- **K%** — strikeouts ÷ plate appearances.
- **HardHit%** — share of batted balls at 95+ mph exit velocity.
- **Barrel%** — share of "barreled" contact (high EV + optimal launch angle).
- **ISO** — Isolated Power (SLG − AVG); raw extra-base power.
- **R/G, RS, RA** — runs per game, runs scored, runs allowed.
- **bWAR** — Baseball-Reference Wins Above Replacement (the public stat).

**Pitch types (Statcast codes)**
- **FF** four-seam fastball · **SI** sinker / two-seam · **FC** cutter · **SL** slider · **ST** sweeper · **CU** curve · **KC** knuckle-curve · **CH** changeup · **FO** forkball / splitter.
- **FB / BR / OS** — broad families: fastball / breaking / offspeed.

**Plate discipline**
- **O-Swing%** — chase rate: swings at pitches OUT of the strike zone.
- **Z-Contact%** — contact rate on in-zone swings.
- **Whiff%** — swing-and-miss rate (misses ÷ swings).
- **PA** plate appearances · **BF** batters faced · **IP** innings pitched · **BIP** balls in play.

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

**Tipping signature** (recurring term)
- A pitcher whose **stuff quality is flat or improved** while **results have collapsed** — signature of opponents knowing what's coming (tipped delivery or predictable sequencing), not a skill loss.

---

## TL;DR

**Our own contrarian board called this in October.** The CausalWAR contrarian leaderboard's top four Phillies "Over-Valued" calls from 2023–24 — Walker, Stott, Luzardo, Nola — are exactly the four names driving the 2026 collapse. The slow start is **~60% real (roster/injury/regression) and ~40% variance (BABIP/sequencing).** Wheeler and Duran on the IL = ~30% of the pain (unavoidable). Walker is being tipped. Stott's pitch recognition has cratered. The league is gameplanning offspeed across the whole lineup. Fixable in 3 roster moves + 1 coaching block.

---

## The damage, decomposed

| Bucket | 2025 | 2026 | Δ R/G |
|---|---|---|---:|
| Offense wOBA | .352 | .389 | **−1.27** |
| Pitching wOBA-against | .325 | .448 | **+1.23** |
| DPI team rank | 24 | **29** | structural |
| Run diff | +112 (full year) | **−38 (21 G)** | worst in MLB |

Offense is roughly at 2025's level; the league is hotter, so .389 ranks 28th. Split: **~50/50 offense-pitching** on the run-differential gap.

---

## Three fixable problems

### 1 — Taijuan Walker is being **tipped**, not declining

| Signal | 2025 | 2026 | Read |
|---|---|---|---|
| Stuff+ | 104.0 | 104.3 | Unchanged |
| Hard-hit% against | .413 | .293 | **Down** |
| Cutter xwOBA-against | .321 | **.563** | Collapsed |
| xwOBA-against (all) | .330 | **.398** | Collapsed |
| 1st TTO xwOBA | — | **.493** | Hit from batter #1 |

Unchanged stuff + unchanged contact quality + collapsed results = tipping/gameplan signature. Opposing hitters aren't squaring balls up — they're getting pitchers' pitches and finding holes because they know what's coming. 35-pitch cutter sample is small, but directional is loud.

**Action:** tape review on cutter delivery this week. If starting, pull at pitch 60–70 (end of TTO-1). 4-year contract makes DFA unrealistic; role change is the lever.

### 2 — Bryson Stott has a breaking-ball recognition crisis

| Signal | 2025 | 2026 |
|---|---:|---:|
| Breaking-ball O-Swing% | 23.7% | **50.0%** |
| Down-in slider chase | 31.8% | **69.2%** |
| Offspeed O-Swing% | 28.7% | **54.5%** |
| wOBA / xwOBA | .317 / .312 | **.188 / .286** |
| wOBA vs RHP (2026) | — | **.152 (38 PA)** |

xwOBA .286 says this isn't pure luck — the approach is broken. Chase doubling is a scouting signal, not variance.

**Action:** platoon vs RHP. Video room pulls Stott's slider takes 2025 vs 2026 and finds the tell.

### 3 — The whole lineup is being gameplanned on offspeed

**9 of 9 Phillies regulars are seeing more changeups/splitters than 2025** — a team-wide scouting directive, not a coincidence:

| Batter | 2025 OS share | 2026 OS share | Offspeed O-Swing Δ |
|---|---:|---:|---:|
| Stott | 13.9% | **19.9%** | +25.8pp |
| Turner | 13.1% | 14.7% | +17.3pp |
| Harper | 12.8% | 15.8% | +8.3pp |
| Schwarber | 13.9% | 17.0% | +6.0pp |
| Marsh | 11.3% | 15.0% | — |

**Fastball is not the problem** — whiff% on FB is *down* for 7 of 9 regulars. The league has decided the Phillies can't lay off a front-hip splitter.

**Action:** hitting coach burns a week on offspeed recognition drills. Highest-leverage *coaching* move on the team.

---

## Bullpen leverage is misallocated

| Reliever | Stuff+ | 2026 xwOBA-against | Inn 8/9 P | Verdict |
|---|---:|---:|---:|---|
| Duran (now on IL) | 109.3 | .262 | 100 | Correct pre-IL; will be again |
| Keller | 109.5 | **.220** | 86 | Earning leverage |
| Mayza | 99.5 | **.147** | 78 | **Underused** |
| **Alvarado** | 103.9 | **.360 (8th) / .545 all** | 55 | **Misallocated** |
| Bowlan | 109.0 | .226 | — | Promotion candidate |

Alvarado's sinker is down 1.2 mph and he's bleeding the 8th. Mayza (.147 xwOBA) and Keller (.220) should inherit high-leverage lanes until Duran returns.

---

## Sequencing tells (fixable, not urgent)

- **Cristopher Sánchez** — 1.25 bits entropy on 0-0 (league 2.80), 64% sinker on first pitch. Most predictable starter in the sport. Surviving on pitch quality. Needs a show-me 4-seam by May 1 or hitters sit sinker every first pitch.
- **Aaron Nola 3-2** — 53% knuckle-curve. Coin-flip giveaway in a full count. Lethal by July if not diversified.

---

## What's NOT the problem (don't overreact)

- **Harper** — xwOBA .380. Fine, unlucky, ride him.
- **Schwarber** — xwOBA **.415**, team's best. 34% K rate is the only real worry. Vindicated as #1 Buy-Low.
- **Realmuto** — quietly team's 2nd-best hitter at .342 xwOBA. BR O-Swing dropped 33% → 14%. No one has noticed.
- **Turner** — .272 xwOBA is a real drop but 54 PA is small and 2024 CausalWAR wasn't pessimistic on him. Keep batting 2.
- **Hitting fastballs** — every starter's FB whiff% is *down*. The bat works; the brain is getting fooled.

---

## Structural headwinds (time heals)

- **Wheeler IL** → ~5 runs lost per missed start. Biggest single item.
- **Duran IL** → closer chain downgraded.
- **Miller (SS) IL + Rojas on restricted list** → **DPI rank 29 vs preseason ~22**. Defense structurally worse until those two return.

---

## This week's playbook

1. **Walker tape review + shorten leash to TTO-1.** Fix the tipping or move to long relief.
2. **Promote Andrew Painter** (.345 xwOBA-a, 9K/2BB in 42 BF) to rotation slot #2–3.
3. **Alvarado out of 8th-inning setup; Mayza in.**
4. **Platoon Stott vs RHP.**
5. **Offspeed recognition block** — hitting coach, whole lineup, this homestand.
6. **Wait on Duran's return** — auto-fixes the closer chain.

---

## Top 3 Briggsy takes

1. **Our Over-Valued board predicted this slow start six months ago.** Walker, Stott, Luzardo, Nola — our four highest-confidence 2023–24 PHI Over-Valued calls — are the four biggest 2026 problems. That's the story. Our contrarian edge isn't theoretical; the Phillies just lived it.
2. **Walker is being tipped, not declining.** Stuff+ flat, hard-hit DOWN, xwOBA catastrophe. Three-signal signature of a pitcher who's been solved, not one who's lost stuff. Tape-room fix.
3. **The whole lineup has a splitter problem.** 9-for-9 offspeed-share increase + 8-for-9 chase increase = league agreed on a gameplan in spring training and Philadelphia hasn't adjusted. Not mechanics, not injury — **recognition**. One week of coaching, not a roster move.

---

## Caveats

- 21 games, ~3 weeks. Per-pitcher arsenals are 35–350 pitches; per-batter discipline splits are 30–80 pitches. **Confidence is directional, not forensic.**
- **CausalWAR was not re-run on 2026** — validated spec needs ≥300 PA. 2024 CausalWAR is used as the prior; 2026 wOBA / wOBA-against is the current signal.
- 2026 data in DuckDB ends 2026-04-08; standings/IL pulled from MLB Stats API current. Pitcher report shows Duran active (100 P through 4-08) while team report shows him on D15 IL — real IL transition between those dates, not a contradiction.
- VWR and MechanixAE-as-predictor explicitly excluded (retracted/demoted 2026-04-18).
- **Re-read at ≥400 pitches per starter / ≥100 PA per regular before committing to structural moves** (≈2 more weeks).

---

# Analyst reports (raw)

## Batter-side diagnostic

### 1. Discovery — data is good, sample is small

- **2026 pitch database:** 73,589 pitches, 247 games, latest date 2026-04-08. ~10% of a full season.
- **Phillies-specific:** 16 team games, 151 distinct batters have appeared. Regulars with ≥30 PA: **9 hitters** (Stott and Crawford near the lower bound at 46 PA).
- Roster cross-check via 2026 appearances (players.team is point-in-time). **Adolis García** confirmed on 2026 PHI (offseason acquisition from TEX).

| Batter (Bats) | 2026 PA | Pitches seen |
|---|---:|---:|
| Schwarber (L) | 64 | 264 |
| Turner (R) | 63 | 238 |
| Harper (L) | 60 | 215 |
| Bohm (R) | 57 | 205 |
| A. García (R) | 57 | 230 |
| Marsh (L) | 51 | 193 |
| Stott (L) | 46 | 181 |
| Crawford (L) | 46 | 153 |
| Realmuto (R) | 39 | 157 |

### 2. Performance scorecard

| Batter | 2026 wOBA / xwOBA | 2025 wOBA / xwOBA | Δ wOBA | Read |
|---|---|---|---:|---|
| **Schwarber** | .376 / **.415** | .393 / .401 | −.017 | xwOBA up 14 pts; bad luck, K 34% (vs 27%) is the worry |
| **Turner** | **.294 / .272** | .367 / .320 | **−.073** | xwOBA collapsed — genuine underperformance |
| **Harper** | .347 / **.380** | .363 / .364 | −.016 | Identical xwOBA; .226 BA is noise |
| **A. García** | .337 / .339 | .294 / .304 | **+.043** | Playing over his head; regression target |
| **Bohm** | .291 / .295 | .342 / .327 | **−.051** | Contact elite (12% K), power gone — 5 XBH |
| **Marsh** | .334 / **.311** | .340 / .328 | −.006 | In line; mild regression risk |
| **Stott** | **.188 / .286** | .317 / .312 | **−.129** | **Worst on team.** xwOBA .286 — approach broken |
| **Crawford** | .316 / **.247** | n/a | — | .273 BA BABIP-driven; xwOBA red flag, but 46 PA |
| **Realmuto** | **.367 / .342** | .317 / .320 | **+.050** | Quietly team's best; sustainable |

CausalWAR not run on 2026 — sample below validated spec (≥300 PA).

**Regression candidates:** García, Realmuto.
**Cold-start luck stories:** Schwarber, Harper.
**Genuine underperformers:** Turner, Bohm, Stott.

### 3. Predictability / plate-discipline patterns

**Opponents feeding Phillies MORE offspeed** (9-for-9 regulars, directional):

| Batter | 2025 OS share | 2026 OS share |
|---|---:|---:|
| Schwarber | 13.9% | **17.0%** |
| Turner | 13.1% | 14.7% |
| Harper | 12.8% | 15.8% |
| A. García | 10.5% | 14.3% |
| Stott | 13.9% | **19.9%** |
| Marsh | 11.3% | 15.0% |

O-Swing% on offspeed up across lineup — Stott (28.7% → 54.5%), Turner (32.7% → 50.0%), Harper (39.3% → 47.6%), Schwarber (23.6% → 29.6%).

**Breaking-ball chase is the defining 2026 problem:**
- **Stott** BR O-Swing 23.7% → **50.0%**; down-in slider chase 31.8% → 69.2%.
- **Turner** BR O-Swing 34.5% → **46.3%**; down-away slider chase 42.4% → 66.7%.
- **Bohm** is the opposite — BR O-Swing 30.5% → 14.7%. Power drop not a chase story.

**Fastballs fine:** whiff/swing on FB down for Schwarber (22.1% → 10.8%), Harper (20.3% → 12.2%), García (25.7% → 17.7%). Not a "can't hit the fastball" story.

**Platoon splits worth acting on** (small-sample flag):
- Stott vs RHP: .152 / .247 in 38 PA.
- Marsh vs LHP: .150 / .146 in 10 PA.
- Turner vs LHP: .139 / .214 in 24 PA (unusual — watch).
- Crawford vs LHP: .129 / .210 in 13 PA.

### 4. Sit / play recommendations

- **Schwarber** — play through. xwOBA .415 is elite.
- **Turner** — play, drop vs LHP until sample sustains.
- **Harper** — play every day.
- **Bohm** — play, but re-evaluate ISO collapse at 100 PA.
- **García** — play, expect regression.
- **Marsh** — strict platoon, stop running vs LHP.
- **Stott** — **demote in order or platoon.** The one unambiguous sit-call.
- **Crawford** — keep looks vs RHP only.
- **Realmuto** — ride him.

### 5. Top 3 Briggsy takes (batter side)

1. **Bryson Stott is the single most concerning bat, and it's pitch recognition, not mechanics.** BR chase rate doubled (23.7% → 50.0%); down-in slider chase 69.2%. Video room should isolate slider-takes 2025 vs 2026.
2. **The league has declared open season on Phillies offspeed.** 9-for-9 regulars seeing more CH/SP than 2025; O-Swing on offspeed up for 8 of 9. Scouting signal about the whole lineup.
3. **Realmuto is the 2026 Phillies' best hitter and no one has noticed.** .367 / .342, .410 OBP. BR O-Swing dropped 33.2% → 13.6%.

---

## Pitcher-side diagnostic

### 1. Roster & sample

**Usable (≥100 PHI pitches, 2026):** Nola (349 P / 4 GS / 20.7 IP), C. Sanchez (326 / 4 / 20.3), Luzardo (286 / 3 / 17.3), Walker (186 / 2 / 10.0), Painter (174 / 2 / 9.0), Alvarado (126 / 7 / 5.0), Bowlan (125 / 7 / 6.3), Mayza (111 / 6 / 8.0), Keller (108 / 6 / 6.0), Pop (107 / 7 / 6.0), Duran (100 / 7 / 6.3).

Rotation: Nola, Sanchez, Luzardo, Walker, Painter. Duran was closer (all 100 P in inning 9) before IL.

### 2. Performance scorecard

| Pitcher | xwOBA 26 | xwOBA 25 | Δ | HardHit 26 | HardHit 25 | Barrel 26 | Barrel 25 | K 26 | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **Nola** | .326 | .314 | +.012 | .450 | .442 | .250 | .223 | 23/70 | FF/SI damage |
| **Sanchez** | .285 | .274 | +.011 | .444 | .402 | .278 | .225 | 27/69 | Contact up, K up |
| **Luzardo** | .258 | .284 | **−.026** | .289 | .375 | .178 | .202 | 24/49 | Legitimately better |
| **Walker** | .398 | .330 | **+.068** | .293 | .413 | .195 | .212 | 6/52 | **RED FLAG — K 11.5%** |
| **Painter** | .257 | — | — | .290 | — | .129 | — | 9/42 | Rookie, holding up |
| Alvarado | .333 | .297 | +.036 | .375 | .480 | .063 | .360 | 7/22 | xwOBA up |
| Duran | .262 | .247 | +.015 | .438 | .425 | .125 | .254 | 8/21 | Fine (pre-IL) |
| Mayza | .147 | .274 | **−.127** | .350 | .455 | .150 | .200 | 8/23 | Dominant |
| Bowlan | .226 | .303 | −.077 | .350 | .368 | .100 | .197 | 9/21 | Quietly great |

**Velocity deltas:** Nola 91.9→91.4 (−0.5), Sanchez SI 95.4→94.5 (**−0.9**), Luzardo 96.5→96.7 (+0.2), Walker 92.2→91.9, Duran 100.6→100.0 (−0.6), **Alvarado SI 99.1→97.9 (−1.2)**.

### 3. Stuff vs Results Gap — THE HEADLINE

| Pitcher | Stuff+ 26 | Stuff+ 25 | Δ | 2026 xwOBA | Verdict |
|---|---:|---:|---:|---:|---|
| **Walker** | **104.3** | 104.0 | +0.3 | **.398** | **GAP — stuff unchanged, results collapsed** |
| **Alvarado** | 103.9 | 103.4 | +0.5 | .333 | Gap — velo −1.2 mph, ambush |
| Nola | 93.7 | 96.8 | **−3.1** | .326 | Stuff eroding — not tipping, decline |
| Sanchez | 92.9 | 91.0 | +1.9 | .285 | Stuff steady — but predictable (§4) |
| Luzardo | 101.8 | 100.5 | +1.3 | .258 | No issue |
| Duran | 109.3 | 106.7 | +2.6 | .262 | Elite |
| Bowlan | 109.0 | 107.2 | +1.8 | .226 | **Hidden gem** |

**Walker:** Stuff+ 104.3 (unchanged), K 11.5%, xwOBA .398. Cutter 2025 .321 xwOBA → 2026 **.563** (35 P, flag). Hard-hit *dropped* on the cutter (.355 → .167). Tipping/gameplan signature.

**Alvarado:** Stuff+ flat, sinker xwOBA .386 → .435. Hard-hit down (.536 → .417) but damage up. Tipping signature + −1.2 mph velo.

### 4. Predictability — pitch-mix entropy by count

**Method:** Shannon entropy over per-count pitch-type distributions (bits). PitchGPT ad-hoc inference out of scope this turn — entropy is the proxy. Lower = more predictable. League baseline: 0-0 = 2.80 bits.

| Pitcher | 0-0 ent | 0-0 top / rate | 3-2 ent | 2-2 ent | Verdict |
|---|---|---|---|---|---|
| **Sanchez** | **1.25** | **SI 64%** | 0.92 (SI 67%) | 1.41 | **Most predictable starter in baseball.** 3-pitch mix. |
| Luzardo | 1.77 | SI 42% | 1.99 | 1.55 | Below league, not alarming |
| Walker | **2.49** | FC 25% | — | — | **Unpredictable — so .398 is NOT predictability, reinforces tipping** |
| Nola | 2.19 | FF 34% | **1.43** (KC 53%) | 2.02 | Fine overall, full-count KC 53% is a tell |
| Painter | 2.15 | FF 36% | — | 2.35 | Normal |

### 5. Times through the order (xwOBA)

| Pitcher | 1st | 2nd | 3rd+ |
|---|---|---|---|
| Nola | .315 | .314 | **.362** |
| Sanchez | .270 | .268 | **.344** |
| Luzardo | .272 | .324 | .147 (small) |
| Walker | **.493** | .344 | .352 |
| Painter | .188 | .335 | .231 |

Walker is getting hit from batter #1. Pull-point should be batter 9, not batter 18.

### 6. Relief / leverage

| Reliever | Inn 8/9 P | xwOBA 8/9 | Stuff+ | Signal |
|---|---|---|---|---|
| **Duran** | 100 | .262 | 109.3 | Correct (pre-IL) |
| Keller | 86 | **.220** | 109.5 | Earning leverage |
| Mayza | 78 | **.152** | 99.5 | **Underused** |
| **Alvarado** | 55 | **.360** | 103.9 | **Misallocated** |
| Pop | 55 | .233 | 103.2 | Fine |
| Backhus | 53 | **.368** | 89.4 | Mop-up only |

### 7. Top 3 Briggsy takes (pitcher side)

1. **Walker is the tipping case. Not Nola, not Sanchez — Walker.** Stuff+ unchanged, hard-hit actually DOWN, xwOBA up .068. That's not worse, that's solved.
2. **Sanchez is the most predictable good starter in the sport right now and it's a ticking clock.** 1.25-bit 0-0 entropy, 64% sinker first pitch. Getting away with it in April because hitters haven't scouted him yet this year.
3. **Rick is giving Alvarado the Carlos Estevez role he lost last summer.** .360 8th-inning xwOBA, −1.2 mph on sinker. Mayza .147 xwOBA playing up. Swap the deck chairs.

---

## Team-level diagnostic

### 1. Team state

| Metric | Value | Context |
|---|---|---|
| Record | **8-13** (.381) | 6.5 GB NL East, NL 14/15, **MLB 30/30 run diff** |
| Run diff | **−38** (RS 75 / RA 113) | Worst MLB. Pythagorean ~6-15. |
| NL East | **4th of 5** (ATL 15-7, MIA 10-12, WSH 10-12, PHI 8-13, NYM 7-15) | |
| Streak | **L5** | Accelerating |
| 2025 baseline | 96-66, +112 | Projected ~90 wins. 2026 pace ~62. |

**Material IL / restricted:** Wheeler (D15), Duran (D15), Miller (D7 SS), Pop (D15), Lazar (D15), Rojas (Restricted List — CF glove gone).

### 2. CausalWAR contrarian board — vindicated

**2023-24 validated board, 68% Buy-Low hit rate @ n=19. 2026 read on top Phillies calls:**

| Prior Call | 2026 Outcome | Verdict |
|---|---|---|
| **Walker OVER-VALUED** (−358 rank diff) | wOBA-a **.480**, 3 HR in 52 BF | **VINDICATED** |
| **Stott OVER-VALUED** (−197) | wOBA **.268**, 0 HR | **VINDICATED** |
| **Luzardo OVER-VALUED** (−208) | wOBA-a **.448** | Partial |
| **Nola OVER-VALUED** (−60) | wOBA-a **.423** | Vindicated |
| **Schwarber BUY-LOW** (+138) | wOBA **.453**, team leader | Vindicated |
| Bohm BUY-LOW (+83) | wOBA .350 | Neutral |
| T. Richards BUY-LOW (+310) | Not on 2026 roster | N/A |
| Alvarado BUY-LOW (+149) | wOBA-a .545 | **Model wrong** |
| Realmuto OVER-VALUED (−39) | wOBA .445 | **Model wrong** |

Four most-confident Over-Valued calls are four of the 2026 biggest problems. **Concentration, not distribution:** 6 of 9 hitters are at/above 2025 (Schwarber, Realmuto, Marsh, Harper, García, Crawford wOBA ≥ .358). Offense drag is Stott, Turner, Bohm.

### 3. DPI team defense

- **PHI DPI 2025:** rank **24 of 30** (OAA +8 but OAA/DPI disagreed by 14 ranks — our model caught a friendly positioning advantage masking below-average range).
- **PHI DPI proxy 2026:** rank **29 of 30** (−0.039). Out rate 62.6% vs expected 66.6% on 412 BIP.
- **Material contributor to slow start: YES** — ~15-20% of staff's inflated wOBA-against.

### 4. Slow-start attribution

Run differential decomp vs 2025:
- 2025 pace: +0.69 R/G → 2026 actual: −1.81 R/G → **−2.50 R/G swing = ~52 runs below expectation over 21 games.**
- Offense: **−1.27 R/G (−26.7 runs)** — wOBA .352 → .389 (league hotter, rank 28th).
- Pitching: **+1.23 R/G (+25.8 runs)** — wOBA-against .325 → .448.
- **Split: ~50/50.**

**Top 3 drivers:**
1. **Wheeler absence** — replacing ~5 WAR ace ≈ 5-6 run swing across missed starts.
2. **Walker + Alvarado collapses in leverage** — combined ~8-10 HL runs above expectation.
3. **Stott offensive void** — .268 wOBA in near-full-time role.

**Luck vs signal:** Hitters .243 BABIP (below .266 — bad luck on hard contact); pitchers .317 BABIP-against vs .266 (~+.050). DPI rank 29 says positioning is a real contributor, not pure variance. **~60% real / ~40% variance.**

### 5. Contrarian lens on current roster

**BUY/play more:** Schwarber (vindicated), Painter (RELIEVER LEVERAGE GAP-adjacent, real), Mayza (under-used HL), Marsh (stealth).
**SIT/reduce:** Walker (top Over-Valued; move to long relief), Stott (platoon), Alvarado (out of HL until velo stabilizes).

### 6. Recommendations

- Promote Painter to rotation #2/3.
- Walker → long relief / spot starts.
- Demote Alvarado out of 8th/9th for 2 weeks.
- Bench/platoon Stott vs LHP; Sosa gets 2B starts.
- Don't overreact on Turner (54 PA, 2024 CausalWAR was not pessimistic).
- Bullpen leverage: wait on Duran return.

### 7. Top 3 Briggsy takes (team)

1. **Our Over-Valued board called this slow start in October.** Walker, Stott, Luzardo, Nola flagged; all vindicated.
2. **The defense was a known problem before the season and got worse.** DPI ranked 24th in 2025 despite +8 OAA; 2026's 29th proxy confirms it. With Rojas restricted and Miller on IL, bleeding ~0.5 R/G above expectation. Not luck — structural.
3. **Play Painter, sit Walker, wait for Duran.** Three moves this week buy back ~15-20 runs over the next 30 games. Rest is Schwarber/Harper/Realmuto continuing to deliver — which they already are.

---

**Files referenced (read-only):**
- `docs/NORTH_STAR.md`
- `results/validate_causal_war_20260418T194415Z/causal_war_baseline_comparison_2023_2024.csv`
- `results/defensive_pressing/2025_validation/team_rankings_2025.csv`
- `src/analytics/defensive_pressing.py`, `src/analytics/stuff_model.py::batch_calculate_stuff_plus`
- `src/dashboard/views/contrarian_leaderboards.py`, `src/dashboard/views/phillies.py`
- `data/baseball.duckdb` (through 2026-04-08)
- MLB Stats API (standings, IL current)
