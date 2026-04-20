# Phillies 2026 Slow-Start Diagnosis — v2

**Date:** 2026-04-20
**Sample:** 21 regular-season games. DuckDB `pitches` through 2026-04-19 (+43,664 rows backfilled since v1). PHI 2026 pitches 7,505 (was 4,792). Stuff+ 2026 refreshed (412 pitchers). Matchup cache rebuilt.
**Split used for trend analysis:** first-12 (≤ 2026-04-08, v1 window) vs last-9 (2026-04-10 → 2026-04-19).
**Method:** same as v1. CausalWAR contrarian prior (2024 fit) + 2026 wOBA/xwOBA/Stuff+/DPI-proxy as current signal. VWR and MechanixAE-as-predictor excluded per 2026-04-18 post-evidence consolidation.
**v1 baseline:** `docs/phillies/2026-04-19_slow_start_diagnosis.md`.

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
- **first-12 vs last-9** — the 21-game window split used for trend analysis (first-12 = through 2026-04-08 / v1 window; last-9 = 2026-04-10 → 2026-04-19).

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

## What changed since yesterday — 5 inversions + 1 new finding

The story isn't what held — it's what flipped.

| Yesterday's call | v2 verdict | Driver |
|---|---|---|
| **Stott** BR chase doubled (50% O-Swing) — platoon vs RHP | **INVERTED** | Adjusted. Full-season BR O-Swing 41.3% (last-9 37.5%). Last-9 wOBA .314 / xwOBA .357. vs RHP last-9 .284/.339. 69.2% down-away slider chase was a 13-pitch artifact. **Keep starting.** |
| **Realmuto** quietly best xwOBA — ride him | **INVERTED** | BR O-Swing 13.8% → **47.4%** over last 9. xwOBA dropped .342 → .312. New regression concern. |
| **Mayza** underused, promote to high leverage | **INVERTED** | Post-0408 xwOBA-a **.317** (was .147). Slippage. Out of HL rotation. |
| **Alvarado** out of 8th inning | **REVERSES** | Last-9 xwOBA-a **.224**. SI velo 97.9 → 98.8 (recovering). Return to HL. |
| Slow start ~60% real / 40% variance, ~50/50 offense-pitching | **DEEPENED** | Last-9 R/G −2.56 vs first-12 −1.25. Offense flat .305 → .304; pitching collapsed .316 → .365. **New split: ~70% real / 30% variance, ~30/70 offense / pitching-defense.** |

**Three holds that strengthened:**

| Yesterday's call | v2 verdict | Driver |
|---|---|---|
| **Walker** tipping case | **STRONGER** | Stuff+ 97.3 → **105.8** (+8.5 vs 2025). Cutter xwOBA .518 on **71 pitches** (2x sample). Hard-hit on cutter **0.080** (top-decile elite). Signature intact on bigger N. |
| Whole-lineup offspeed gameplan (9-for-9) | **HELD** | Zero opponent pullback. But batter response has forked (see action plan). |
| CausalWAR Over-Valued board vindicated | **STRENGTHENED** | 3 of 4 still vindicated; **Luzardo moves from "partial" to "fully vindicated"**; Stott softening. |

**New finding (single biggest update):**

> **Jesús Luzardo is the second tipping case.** Last-9 xwOBA-against **.350** (first-12: .258). Stuff+ 101.7 (−2.8 vs 2025 — mild, not crisis). Same stuff-vs-results gap signature as Walker. The rotation now has **two tipping candidates**, not one. **Painter promotion is urgent, not optional.**

---

## TL;DR v2

**The slow start is accelerating, not stabilizing.** 2-7 in the 9 new games, −23 run differential, pace moved from −1.25 R/G to −2.56 R/G. Offense is flat (.305 wOBA both windows); the collapse is **pitching + defense**. Luzardo joined Walker as a tipping candidate. The bullpen reshuffled completely (Kerkering is the 8th-inning answer, not Mayza). Stott adjusted himself off yesterday's sit-call; Realmuto regressed off yesterday's hidden-star call. DPI rank went 29 → **30 of 30**. BABIP luck unwound against Philadelphia, not toward them. **70/30 real/variance now.**

---

## The damage — 21 games, deepening

| Metric | First-12 (≤ 4/08) | **Last-9 (4/10–4/19)** | Δ |
|---|---:|---:|---|
| Record | 6-6 | **2-7** | collapse |
| Run diff | −15 | **−23** | worse |
| R/G | −1.25 | **−2.56** | deepening |
| PHI offense wOBA | .305 | **.304** | flat |
| PHI pitching wOBA-against | .316 | **.365** | **+.049** |
| DPI proxy | −0.046 | **−0.078** | worsening (MLB 30/30) |

Offense unchanged. Pitching wOBA-against +49 points in last-9. Defense rank dropped 29 → 30. BABIP-against moved from .317 → ~.335 — luck unwound AGAINST Philadelphia on the pitching side. **The collapse is real, not regressing favorably.**

---

## Three fixable problems v2

### 1 — Two tipping candidates, not one: Walker + Luzardo

**Walker (stronger):**
- Stuff+ 97.3 (2025) → **105.8 (2026)** — stuff IMPROVED. Yesterday called 104.3 flat; refreshed 2025 baseline now says 2026 stuff is actually *better*.
- Cutter xwOBA 2025 .321 → 2026 .518 on **71 pitches** (doubled from yesterday's 35).
- Hard-hit on cutter **0.080** (top-decile elite contact suppression) → xwOBA .518 gap is pure result-quality divergence.
- TTO-1 .479 xwOBA; *gets better* each TTO — classic solved-arsenal signature.
- **Single highest-confidence pitcher call on the team.**

**Luzardo (new):**
- First-12 xwOBA-a .258 → **Last-9 xwOBA-a .350** (+.092).
- Stuff+ 101.7 — mild erosion (−2.8 vs 2025), not collapse.
- Sweeper xwOBA .521 on 21 BBE (flag sample).
- Same stuff-vs-results gap signature as Walker.
- Action: tape review on cutter (Walker) and sweeper sequencing (Luzardo) **this week**.

### 2 — Bohm is the new worst PHI bat (Stott already adjusted)

**Bohm:**
- Last-9 wOBA **.128**, xwOBA **.217**. Full-season .261 / .264 (worsened from .291 / .295 yesterday).
- Now sample-adequate (80 PA) to act on.
- Hard-hit ticked up (31.4%) but zero slug — GB/FB mix issue, not contact quality.
- **Demote in the order.** Not fixable by approach alone.

**Stott (for the record — call inverted):**
- Last-9 wOBA .314, xwOBA .357, HardHit 41.4%.
- vs RHP last-9: .284 / .339 in 22 PA.
- BR O-Swing 50% → **41.3% full / 37.5% last-9**.
- Yesterday's platoon call was a 38-PA artifact; current data says keep him starting vs RHP.

### 3 — Whole-lineup offspeed gameplan still live, but targets have shifted

**Opponent side:** 9-for-9 OS-share uptick UNCHANGED. No pullback in the 9 new games.

**Batter response has forked:**
- **Adjusting down (good):** Stott OS O-Swing 54.5% → 45.7%, Bohm 35.3% → 26.1%, Crawford 87.5% → 56.2%.
- **Chasing harder (work these):** **Harper OS O-Swing 47.6% → 62.1%**, Marsh up, **Realmuto BR O-Swing 13.8% → 47.4% last-9**.

**Coaching-priority rotation shifts from Stott to Harper + Realmuto.**

---

## Bullpen v2 — reshuffled

| Role | v1 pick | **v2 pick** | Driver |
|---|---|---|---|
| Closer | Duran (IL) | Duran on return | No change |
| 8th setup | Mayza (.147 xwOBA-a) | **Kerkering** | Activated 4/7. Last-9 xwOBA-a .212. Stuff+ 103. Overtook everyone. |
| Medium leverage | Alvarado (demote call) | **Alvarado (rehabilitated) + Mayza** | Alvarado SI velo 97.9 → 98.8; last-9 xwOBA-a .224. Mayza post-0408 .317. |
| Backend | Keller, Pop | Keller (slipping to .335 post), Pop (steady) | Keller going the wrong way |
| Mop-up | Backhus | Backhus | No change |

**Kerkering wasn't on yesterday's table.** He's now the highest-ranked healthy HL option.

---

## CausalWAR contrarian board v2

Prior from 2023–24 board (68% Buy-Low hit rate validated). 2026 check:

| Prior call | v1 verdict | v2 verdict | Driver |
|---|---|---|---|
| **Walker OVER-VALUED** (−358 rank diff) | Vindicated (.480 wOBA-a) | **Vindicated ×2** | Last-9 wOBA-a .433 |
| **Luzardo OVER-VALUED** (−208) | Partial (.268) | **Now vindicated** | Last-9 wOBA-a .420, xwOBA-a .350 |
| **Nola OVER-VALUED** (−60) | Vindicated | Holds | Last-9 wOBA-a .404; xwOBA stable |
| **Stott OVER-VALUED** (−197) | Vindicated (.188 wOBA) | **Weakening** | Last-9 wOBA .314 / xwOBA .357 |
| **Schwarber BUY-LOW** (+138) | Vindicated (team leader) | **Vindicated ×2** | Full season .442 wOBA |
| **Alvarado BUY-LOW** (+149) | Model wrong (.545) | **Flipping back** | Last-9 xwOBA-a .224 |
| **Realmuto OVER-VALUED** (−39) | Model wrong (.367) | **Cooling toward prior** | Last-9 xwOBA .275 |

**Net:** model's top over-valued cohort (Walker + Luzardo + Nola + Stott) is 3-for-4 still vindicated, with Stott softening. Schwarber Buy-Low reinforced. The two "model wrong" calls (Alvarado, Realmuto) are both moving toward the model in the new 9 games.

---

## What's NOT the problem (updated)

- **Schwarber** — .442 / .411, team leader. Vindicated Buy-Low.
- **Harper** — heating up (+.027 wOBA since 4-08), xwOBA .422. Chase rate is a July watch; bat is working now.
- **Crawford** — xwOBA climbed to .287. Legit for RHP starts.
- **Turner** — LHP scare fully regressed (.303 xwOBA vs LHP). Keep batting 2.
- **Stott** — inverted from v1's #1 concern. Keep starting vs RHP.
- **Aggregate offense** — wOBA flat first-12 vs last-9. Not where the collapse is.

---

## Structural (time heals / doesn't)

- **Wheeler** still on IL — no 2026 rehab activation transaction logged as of 4/19.
- **Duran** likely still on IL — 3 PA post-0408 then absent.
- **Miller** (SS IL) + **Rojas** (restricted list) still out — defense rank **30 of 30**.
- **Kerkering** activated 4/7 — one reinforcement arrived and is already earning HL innings.

---

## This week's playbook v2

1. **Walker + Luzardo tape review this week.** Both show the tipping signature. Walker is the stronger and older case (cutter); Luzardo is fresher and may be more fixable (sweeper sequencing). Highest-leverage single action on the roster.
2. **Promote Painter to rotation — now urgent.** Two tipping candidates + Luzardo's last-9 collapse + Wheeler's absence = rotation is a 5-deep liability. Cap Painter at 5 innings / 75 pitches (TTO-2 .360 xwOBA-a).
3. **Bullpen reshuffle: Kerkering → 8th vs RHB. Alvarado returns to HL. Mayza out of HL.** Yesterday's Mayza-promotion call is dead.
4. **Demote Bohm in the order.** Last-9 wOBA .128 / xwOBA .217 at 80 PA is sample-adequate.
5. **Pivot offspeed-recognition coaching from Stott to Harper + Realmuto.** Stott adjusted; Harper (62.1% OS chase) and Realmuto (47.4% BR chase last-9) did not.
6. **Wait on Wheeler/Duran returns.** No activation logged.

---

## Top 3 Briggsy takes v2

1. **The slow start isn't stabilizing — it's deepening, and it's pitching, not offense.** 2-7 in 9 new games, −2.56 R/G pace. Offense wOBA flat. Pitching wOBA-against +49 points. DPI rank 30 of 30. BABIP luck unwound AGAINST us. Attribution moves from 60/40 to **70/30 real**, and from 50/50 offense-pitching to **30/70 offense-pitching/defense**. This is not a regression story — it's a trajectory story.
2. **Walker has a twin. Luzardo is tipping case #2.** Last-9 xwOBA-a .350 on a flat arsenal. Same signature. Yesterday's "Walker is the tipping case" has become "the rotation has two tipping candidates and Painter promotion is urgent." Single biggest update since v1.
3. **Yesterday's report was right about the shape; wrong about five specific players.** Stott adjusted, Realmuto regressed, Mayza collapsed, Alvarado recovered, Kerkering emerged. The 8th-inning handoff map got completely redrawn. Offspeed-recognition targets shifted from Stott to Harper + Realmuto. **Lesson: a 16-game sample surfaces the shape of a team's problems; a 25-game sample surfaces the names.**

---

## Caveats

- 21 regular-season games; v1's same 21 games but richer pitch sample (7,505 vs 4,792 PHI pitches). MLB Stats API standings endpoint lagged at query time; per-game feeds confirm the 21-game window is self-consistent.
- Per-pitcher last-9 samples are 15–55 PA; per-batter last-9 are 19–25 PA. Confidence still directional.
- **CausalWAR NOT re-run on 2026** (below spec's ≥300 PA per pitcher / batter). 2024 CausalWAR used as prior.
- **DPI proxy** uses out-rate vs 1-xBA on BIP ≥ league mean; not the full trained DPI model. Validated in v1 as directionally correct.
- **Walker cutter case at 71 pitches** (doubled from yesterday) — stronger, not forensic.
- **Luzardo tipping case at 9-game sample** — new, should be re-validated in 2 more starts.
- Wheeler/Duran IL status inferred from absent pitch data + (for Wheeler) transactions table; no explicit 2026 activation in either case.
- VWR and MechanixAE-as-predictor explicitly excluded per post-evidence consolidation 2026-04-18.
- **Flagship transformers NOT retrained** (12 days of data is below the threshold for retraining models fit on 10+ years). Inference/scoring refreshed only (Stuff+ 412 pitchers, matchup_summary full rebuild). If full retraining is desired, flag as a separate task — hours of GPU time.

---

# Raw analyst reports (v2)

## Batter-side diagnostic v2

### Sample growth

All 9 regulars now in the 57–93 PA band (target hit vs v1's 40–65). Still pre-100 for structural moves but above the "directional only" floor.

| Batter | 2026 PA (4-08 → 4-19) | Pitches |
|---|---|---:|
| Schwarber | 54 → **93** | 414 |
| Turner | 54 → **92** | 391 |
| Harper | 52 → **88** | 344 |
| A. García | 49 → **85** | 396 |
| Bohm | 48 → **80** | 320 |
| Marsh | 43 → **72** | 293 |
| Crawford | 38 → **65** | 256 |
| Stott | 38 → **63** | 274 |
| Realmuto | 32 → **57** | 258 |

### Performance scorecard (diff-first)

| Batter | 2025 wOBA / xwOBA | v1 (4-08) | **v2 (4-19)** | Δ wOBA vs 4-08 | Read |
|---|---|---|---|---:|---|
| Schwarber | .406 / .401 | .453 / .415 | **.442 / .411** | −.011 | Elite. Stable. |
| Turner | .371 / .320 | .324 / .272 | **.317 / .282** | −.007 | xwOBA ticked up .010. |
| Harper | .368 / .364 | .388 / .380 | **.415 / .422** | **+.027** | **Heating up.** |
| A. García | .301 / .304 | .369 / .339 | **.321 / .324** | **−.048** | Regressing. |
| Bohm | .354 / .327 | .350 / .295 | **.261 / .264** | **−.089** | **Free-fall.** Last-9 wOBA .128. |
| Marsh | .349 / .327 | .392 / .311 | **.351 / .323** | −.041 | Settling. |
| Stott | .325 / .312 | .268 / .286 | **.287 / .314** | **+.019** | **xwOBA at 2025 level.** |
| Crawford | n/a | .358 / .247 | **.354 / .287** | −.004 | xwOBA +.040. |
| Realmuto | .320 / .320 | .445 / .342 | **.382 / .312** | **−.063** | **Regressing.** |

### Yesterday's findings — did they hold?

- **Stott BR chase (v1 headline):** WEAKENED. BR O-Swing 25.5% (25) → 41.3% full (was 50.0%). Last-9 37.5%. Down-away slider chase 69.2% was a 13-pitch artifact. Adjusted.
- **Whole-lineup offspeed uptick:** HELD. 9-for-9 intact. No opponent pullback.
- **Batter OS O-Swing response:** FORKED. Stott 54.5% → 45.7%, Harper 47.6% → 62.1%, Bohm 35.3% → 26.1%, Crawford 87.5% → 56.2%.
- **Fastball whiff% down:** WEAKENED (7 of 9 → 5 of 9). Bohm and Realmuto flipped UP.
- **Platoon scares:** Turner vs LHP .214 → .303 (noise). Stott vs RHP .247 → .285 (league-avg). Marsh vs LHP still a real problem (12 PA).

### New findings from added sample

- **Realmuto BR O-Swing 13.8% → 47.4% over last 9.** Last-9 xwOBA .275. New regression concern.
- **Bohm last-9 wOBA .128 / xwOBA .217.** Now sample-adequate (80 PA).
- **Harper +.027 wOBA, OS O-Swing 62.1%.** Mashing anyway; July problem.
- **Crawford xwOBA climbed .287.** Legit for RHP starts.
- **Stott last-9 HardHit 41.4%.** Approach coming back online.

### Batter top 3 Briggsy takes

1. **Stott recognition-crisis call INVERTS.** BR chase 50% → 41.3%. Last-9 .314/.357. The 69.2% down-away chase was 13 pitches. Platoon call off.
2. **Realmuto hidden-star call WEAKENS.** BR chase 13.8% → 47.4% last-9. xwOBA fell .342 → .312. The approach-discipline edge evaporated.
3. **Offspeed gameplan holds; targets shifted.** 9-for-9 opponent uptick unchanged, but batter response forked. Work Harper + Realmuto first, not Stott.

---

## Pitcher-side diagnostic v2

### Scorecard + Δ since 4-08

| Pitcher | xwOBA 25 | xwOBA 26 full | pre-0408 | post-0408 | Δ post | K% 26 | HardHit 26 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Walker** | .330 | **.400** | .398 | **.403** | flat | 17.2% | .213 |
| **Sanchez** | .274 | **.267** | .285 | **.245** | **−.040** | 35.0% | .279 |
| **Nola** | .314 | .323 | .326 | **.316** | −.010 | 30.1% | .267 |
| **Luzardo** | .284 | .304 | .258 | **.350** | **+.092** | 36.7% | .211 |
| **Painter** | — | .275 | .257 | **.294** | +.037 | 25.3% | .189 |

**Velocity:** Sanchez SI recovered (95.4 → 94.5 → 95.2). Alvarado SI partial recovery (99.1 → 97.9 → 98.8). Luzardo FF up. Nola FF up. Velocity is not the story for any starter.

### Stuff vs Results Gap v2

| Pitcher | Stuff+ 25 | Stuff+ 26 | Δ | 2026 xwOBA | Verdict |
|---|---:|---:|---:|---:|---|
| **Walker** | 97.3 | **105.8** | **+8.5** | .400 | **Gap WIDENED.** Refreshed 2025 baseline shows 2026 stuff is *better*. Tipping case stronger. |
| Nola | 99.7 | 93.5 | **−6.2** | .323 | Decline holds and deepens. |
| Sanchez | 107.8 | 92.9 | **−14.9** | .267 | New caveat. Results improved despite Stuff+ drop (likely refit anomaly). |
| Alvarado | 103.9 | 103.6 | −0.3 | .297 | Flat stuff, xwOBA easing. |
| Luzardo | 104.5 | 101.7 | −2.8 | .304 | Slight erosion + last-9 collapse to .350. **New concern.** |
| Painter | — | 106.4 | — | .275 | Legit. |

**Walker cutter:** 2025 .321 → 2026 full **.518 xwOBA on 71 pitches** (was 35). Hard-hit on cutter **0.080** (top-decile). Stuff better + contact softer + results worse = tipping signature confirmed at 2x sample.

### Predictability (Shannon entropy, 2026)

League baseline 2026: 0-0 = 2.80 bits, 3-2 = 2.70.

| Pitcher | 0-0 ent | 0-0 top / rate | 3-2 ent | Verdict |
|---|---:|---|---:|---|
| **Sanchez** | 1.35 | SI 59% | 1.39 | Predictable call HOLDS, weakens. Post-0408 0-0 = 1.47 / 52%. Drifting toward league. |
| Nola | 2.19 | FF 33% | **1.69** (KC 48%) | 3-2 KC tell still exists, less loud. |
| Walker | **2.50** | FC 24% | 2.09 | **Unpredictable.** So .400 is NOT a mix problem. Tipping-not-patterning reinforced. |
| Luzardo | 1.79 | SI 43% | 1.96 | Normal. |
| Painter | 2.20 | FF 40% | 2.32 | Diverse and holding. |

### TTO xwOBA

| Pitcher | TTO-1 | TTO-2 | TTO-3 |
|---|---:|---:|---:|
| **Walker** | **.479** | .391 | .294 | gets *better* each TTO = solved signature |
| Sanchez | .283 | .241 | .281 | |
| Nola | .319 | .309 | .354 | |
| Luzardo | .248 | .346 | .323 | |
| Painter | **.183** | .360 | .312 | rookie arsenal-reveal; 5-inning cap |

### Relief / leverage v2

| Reliever | 2026 Inn8+ PA | xwOBA 8+ | pre-0408 | post-0408 | Verdict |
|---|---:|---:|---:|---:|---|
| **Mayza** | 17 | .198 | .152 | **.317** | **COLLAPSED post-0408.** |
| **Kerkering** | 23 | **.222** | .253 | **.212** | **NEW — activated 4/7. Earning 8th.** |
| Duran | 24 | .237 | .262 | — | IL |
| Pop | 13 | .233 | same | same | Steady. |
| Keller | 24 | .263 | .220 | **.335** | **Slipping.** |
| **Alvarado** | 18 | .328 | **.360** | **.246** | **EASED.** Velo recovered. |
| Bowlan | 4 | .206 | .226 | — | Usage shifted. |
| Backhus | 12 | .368 | same | — | Low-leverage only. |

### Pitcher top 3 Briggsy takes

1. **Walker tipping case got LOUDER, not quieter.** Stuff+ 105.8 (+8.5), cutter .518 on 71 P (2x sample), hard-hit cutter .080. Highest-confidence pitcher call on team.
2. **Sanchez predictability call WEAKENS but doesn't invert.** 0-0 entropy 1.25 → 1.35 full (1.47 post). SI share 64% → 59%. Diversifying. Downgrade from "ticking clock" to "watch by May 15."
3. **Bullpen reshuffled: Kerkering is the post-Duran 8th.** 91 P, .212 post-0408 xwOBA. Alvarado demote call softens (.246 post). Mayza out. Nine games changed the closer chain.

---

## Team-level diagnostic v2

### Team state (first-12 vs last-9 split)

| Metric | v1 (through 4-08) | **v2 (through 4-19)** |
|---|---|---|
| Record | 8-13 | 8-13 (API lag); per-game-feed-reconstructed RS=75 / RA=113 |
| Run diff | −38 | **−38** |
| NL East rank | 4/5 | 4/5 |
| MLB run-diff rank | 30/30 | **30/30** |
| **Last-9 (4/10–4/19)** | n/a | **2-7, −23, −2.56 R/G** |
| First-12 | — | 6-6, −15, −1.25 R/G |

### Team aggregates (pitch-DB events-based)

| Slice | off wOBA | off xwOBA | def wOBA-a | def xwOBA-a | DPI proxy |
|---|---:|---:|---:|---:|---:|
| Season-to-date | .305 | .317 | .338 | .298 | **−0.060 (30/30)** |
| First 12 | .305 | — | .316 | — | −0.046 |
| Last 9 | **.304 (flat)** | — | **.365** | — | **−0.078 (worse)** |

**Offense flat. Pitching collapsed. Defense worsened. BABIP unwound AGAINST us.**

### Slow-start attribution v2

- First-12: R/G −1.25, offense 3.50, defense-pitching 4.75.
- Last-9: R/G **−2.56**, offense 3.67 (slight improve), defense-pitching **6.22** (+1.47).
- **v1 50/50 attribution → v2 30/70** (offense / pitching-defense).
- **v1 60% real / 40% variance → v2 ~70% / 30%.** BABIP-against .317 → ~.335 in last-9.

### CausalWAR contrarian board v2

See synthesis table above. **3 of 4 top Over-Valued calls still vindicated; Luzardo moved partial → fully vindicated; Stott weakening; Schwarber Buy-Low reinforced; Alvarado + Realmuto both moving back toward model.**

### DPI v2

542 BIP (v1: 412). Out rate 60.5% vs expected 66.5%. DPI proxy **−0.060** (v1: −0.039). **Rank 30 of 30** (v1: 29). Last 9: −0.078. Miller IL + Rojas restricted unchanged. Structural.

### Recommendations v2

| v1 call | v2 verdict |
|---|---|
| Promote Painter | **STILL HOLDS** (urgent) |
| Sit/relief-role Walker | **STILL HOLDS, stronger** |
| Demote Alvarado out of 8th | **REVERSE** (velo recovered, last-9 xwOBA-a .224) |
| Platoon Stott vs RHP | **SOFTEN** — keep in lineup |
| Offspeed-recognition coaching block | **STILL HOLDS**; pivot targets to Harper + Realmuto |

**New calls:**
- **Kerkering → 8th vs RHB.** 91 P, .212 post-0408 xwOBA, Stuff+ 103.
- **Luzardo diagnostic required.** The second Walker-scenario. Pull xwOBA by pitch type + tipping candidates.
- **Bohm is now the worst PHI bat.** Demote in order until contact normalizes.

### Team top 3 Briggsy takes

1. **v1 60%/40% real/variance → v2 70%/30%, tilted toward pitching-defense. Collapse is deepening.**
2. **Walker has a companion: Luzardo is tipping case #2. Painter promotion now urgent.**
3. **Alvarado Buy-Low flipping back; Kerkering return invalidates the 8th-inning recommendation.** Mayza out. Reorder: Kerkering → Alvarado → Keller.

---

## Files referenced (read-only except as noted)

- `docs/NORTH_STAR.md`, `docs/phillies/2026-04-19_slow_start_diagnosis.md`
- `scripts/backfill_2026_04_20_gap.py` (new, written this session)
- `src/ingest/statcast_loader.py::load_statcast_range`
- `src/ingest/daily_etl.py::_refresh_matchup_cache`
- `src/analytics/stuff_model.py::batch_calculate_stuff_plus`
- `src/analytics/defensive_pressing.py`
- `src/dashboard/views/contrarian_leaderboards.py`
- `results/validate_causal_war_20260418T194415Z/causal_war_baseline_comparison_2023_2024.csv`
- `data/baseball.duckdb` (pitches through 2026-04-19; `data_freshness` watermark updated)
- MLB Stats API (per-game `feed/live`, transactions, standings)
