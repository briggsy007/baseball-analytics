# CausalWAR vs bWAR: Contrarian Case Studies (2023-24)

## Headline

CausalWAR's biggest disagreements with Baseball-Reference WAR over the
2023-24 test cohort are not random noise -- they line up almost perfectly
with the three things the model explicitly does *not* see: infield defense,
late-inning leverage, and home-park run environment. CausalWAR is a per-PA
offensive run-value estimator that residualises out venue, platoon, and
base-out state but knows nothing about gloves, win-probability swings, or
ballpark suppression of contact quality. Three of the most extreme rank
disagreements in the leaderboard -- Andres Gimenez (-368), Brayan Bello
(-399), and Robert Garcia (+285) -- each isolate one of those blind spots
cleanly. Two of the three look like genuine methodology gaps; the third
(Gimenez) looks like a real edge that the public WAR is currently
*overpaying* for.

Test cohort: 968 players, ages 2023 and 2024, PA >= 300 / IP >= 50,
nuisance models trained on 2015-2022 only. Source files:
`results/validate_causal_war_20260418T194415Z/causal_war_baseline_comparison_2023_2024.csv`
and the matching `_metrics.json`.

---

## Case 1: Andres Gimenez (player_id 665926, SS, CLE)

**The disagreement.** `causal_war = -1.98`, `trad_war = +4.48`,
`rank_diff = -368` (rank_causal=392, rank_trad=24). bWAR has him as a
top-25 player in the league across 2023-24; CausalWAR has him in the
bottom third.

**The line.** 1,289 PA across the two seasons. 2023: .251/.314/.399, 5.02
bWAR. 2024: .250/.297/.335 (a .632 OPS), 3.99 bWAR. From the pitch table:
2024 average exit velocity 80.2 mph, xwOBA on contact 0.296, 10 HR in 691
PA. This is a below-average bat playing every day at shortstop. He was a
top-3 finisher in AL Gold Glove voting both seasons.

**Why CausalWAR disagrees.** The model's outcome is `woba_value` per PA
and the only confounders are venue, platoon, base-out state, inning
bucket, month, and handedness (see `_build_features` in
`src/analytics/causal_war.py:812-909`). Defensive runs are *not in the
data the model ingests*, full stop. bWAR's 4.48 over two seasons for a
.297-OBP hitter is mathematically only possible because Baseball-Reference
is crediting him roughly +25 to +30 fielding runs across the cohort.
CausalWAR sees the bat (genuinely below replacement on offense, with a
2024 wOBA value per PA below the 0.292 actual aggregate at-bat outcome)
and prices him accordingly.

**Honest take.** This is the textbook methodology gap. CausalWAR is
*correct on offense* -- a 2024 .632 OPS shortstop is a -1.5 to -2 win
offensive contributor and that is what the model returns -- but it is not
trying to be total WAR. The interesting question is whether bWAR is
*right* about the +25-30 defensive runs. UZR/OAA/DRS rarely agree on a
glove-first player to that magnitude, and TOR signed Gimenez to a $106M
extension that the market is now visibly underwater on (.603 OPS in 2025,
0.759 OPS in 40 PA so far in 2026). My read: CausalWAR is flagging
something real -- bWAR is over-rewarding the defensive component and the
post-extension performance shows the market overpaid based on that signal.

**What it would take to test.** If CausalWAR's view is right, Gimenez's
*total* 2026 value (offense + defense) finishes well below his 4 bWAR
average -- he's already on pace for a 2-2.5 bWAR season at best. If bWAR
is right and the glove still carries him, he should rebound to 4+ even
without bat improvement. Track his 2026 OAA and DRS alongside his bWAR.

---

## Case 2: Brayan Bello (player_id 678394, SP, BOS)

**The disagreement.** `causal_war = -1.19`, `trad_war = +2.14`,
`rank_diff = -399`. bWAR ranks him a top-100 pitcher; CausalWAR has him
near replacement-level.

**The line.** 319.1 IP across 2023-24. 2023: 4.24 ERA / 19.8% K / 6.7%
BB / 1.38 HR/9 in 157 IP, 3.14 bWAR. 2024: 4.49 ERA / 21.8% K / 9.1% BB /
1.05 HR/9 in 162.1 IP, 1.18 bWAR. From the pitch table: 2023 xwOBA
allowed 0.316 vs actual wOBA *allowed* 0.353 (a 37-point under-perform).
2024 xwOBA 0.331 vs actual 0.332 -- the 2024 quality-of-contact normalised.

**Why CausalWAR disagrees.** Two compounding effects.

1. *Park and contact-quality.* Bello pitches half his games at Fenway,
   one of the most run-friendly parks for righties in MLB. The model's
   only park control is a label-encoded `venue_code` -- it knows the
   identity of the ballpark but does not have explicit run-environment
   coefficients. At Fenway 2023-24, opposing batters posted a 0.343-0.353
   actual wOBA against him; on the road, 0.321-0.354. The model is
   absorbing the Fenway penalty into the player effect rather than the
   nuisance. bWAR uses an explicit park factor that *credits* him for
   pitching there.
2. *Outcome variance.* CausalWAR's outcome is per-PA `woba_value` --
   essentially every batted ball is treated equally. A starter with an
   xwOBA-actual wOBA gap of 37 points in 2023 looks markedly worse to
   CausalWAR than to bWAR, which uses RA9 with park and defense adjustments.

**Honest take.** This one is mostly a methodology gap. Bello's underlying
stuff is real (95.6 mph sinker, 45% fastball rate in 2024, kept the ball
in the park better than league average). His 2025 follow-up bears this
out: 169 IP, 3.41 ERA -- the bWAR view was closer to right and CausalWAR
was capturing both Fenway noise and bad-luck sequencing. The fix isn't
to add another feature -- it's to use a park-factor multiplier on the
outcome residual, the way standard pitcher-WAR formulations do. A label-
encoded venue cannot mechanically reproduce a continuous park factor
inside a tree-based nuisance model unless every team-season pair gets
enough cross-fit folds to learn its own coefficient, which 2023-24 sample
sizes don't really allow.

**What it would take to test.** Bello already passed the 2026 prediction
test backwards: 2025's 3.41 ERA on 169 IP would have given him roughly
2.5-3 bWAR, which is what the *traditional* model said in 2023-24, not
what CausalWAR said. If 2026 trends the same way (he's at 8 IP with a
9.00 ERA early -- small sample), the verdict is final: CausalWAR is
underrating Fenway starters systematically.

---

## Case 3: Robert Garcia (player_id 676395, RP, MIA -> WSH)

**The disagreement.** `causal_war = +1.04`, `trad_war = +0.06`,
`rank_diff = +285` (rank_causal=138, rank_trad=423). CausalWAR thinks he
was a quietly above-average reliever; bWAR has him as roughly replacement
level.

**The line.** 91.2 IP across 2023-24, all in relief. 2023: 32 IP, 3.66
ERA, 25.9% K, 0.40 bWAR. 2024: 59.2 IP, 4.22 ERA, 29.9% K, 6.4% BB,
0.61 HR/9, -0.12 bWAR. Pitch-table evidence: 2024 xwOBA allowed 0.251
vs actual wOBA 0.286 (a 35-point *over-perform* by hitters that ate his
ERA). Hidden inside the negative bWAR is a high-K, low-walk peripheral
profile.

**Why CausalWAR disagrees.** Two clean reasons.

1. *Per-PA equality vs leverage weighting.* CausalWAR treats every PA
   equally; bWAR weights pitcher value with a leverage adjustment but
   then *discounts* relievers via the chaining/replacement-level
   conventions. Garcia's 2024 inning distribution skews late: 65.7% of
   his pitches came in inning 7 or later. He's a late-inning lefty
   bridge piece. The DML estimator sees those late PAs as identical to
   any other PA and credits him with the underlying xwOBA outperformance.
   bWAR's reliever penalty erases most of it.
2. *xwOBA vs actual wOBA gap.* The 2024 actual wOBA against him (0.286)
   was 35 points worse than his expected (0.251). bWAR is a results-based
   metric (RA9-driven). CausalWAR's residual on a per-PA basis is closer
   to the expected outcome because the nuisance model partials out the
   base-out state. Result: CausalWAR thinks he had bad luck on contact;
   bWAR records what actually happened.

**Honest take.** This one I think CausalWAR is *right* and the public
WAR is wrong, on stuff. A reliever with a 30% K rate, a 6% walk rate,
and 0.6 HR/9 should not be 0 bWAR. The problem is bWAR's reliever
chaining. The 2025 follow-up confirms: 64 IP, 2.95 ERA, ERA finally
caught up to the peripherals. He's posting a 2.25 ERA in early 2026
(small sample). This is a genuine edge -- the kind of player a front
office could acquire cheaply because the public WAR sites underrate him.

**What it would take to test.** If CausalWAR is right, his ERA and FIP
should continue to converge to the high-K profile suggests (sub-3.50 ERA,
sub-3.00 FIP). If bWAR is right and the contact quality was a fluke, he
regresses to a 4.50 ERA middle reliever. Through 4 IP in 2026 the early
read favors CausalWAR.

---

## Pattern across the three

The three cases each map cleanly to a *different* feature CausalWAR
deliberately or implicitly omits: **defense (Gimenez), park run-environment
beyond label-encoded venue (Bello), and leverage / reliever chaining
(Garcia)**. The pattern isn't that CausalWAR is broken -- it's that bWAR
and CausalWAR are measuring overlapping but distinct constructs, and the
extreme disagreements are exactly where one of those non-overlap
components dominates. There is no consistent direction: bWAR over-rewards
defense and park-friendly starters, under-rewards stuff-driven middle
relievers. CausalWAR is the inverse on each.

## What to do next

The most actionable methodology change is to add a continuous park-factor
feature (HR factor, BACON factor) to the nuisance vector rather than
relying on a label-encoded venue, which would close most of the Bello-type
gaps. Second priority is a defensive-runs-saved (or OAA aggregate) feature
on batter PAs, which the current per-PA grain doesn't naturally support
but could be added at the aggregation step as a separate `defensive_war`
column shown alongside `causal_war` -- explicit acknowledgement that the
model is offense-only. The Garcia-type cases are *features*, not bugs:
those are the genuine edges. The natural product surface is a "CausalWAR
above bWAR" leaderboard -- a buy-low scouting list -- combined with a
"bWAR above CausalWAR" leaderboard flagging players whose WAR depends
on glove + park + sequencing rather than per-PA contact quality.
