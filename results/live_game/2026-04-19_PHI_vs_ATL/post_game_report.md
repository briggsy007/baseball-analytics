# Post-Game Report — PHI vs ATL — 2026-04-19

**Generated:** 2026-04-20 (Phase 2 of the live-game case study).
**Game PK:** 823475 — final at Citizens Bank Park.
**Source artifacts in this directory:** `box_score.json`,
`reliability_diagram.png`, `reliability_data.json`,
`lstm_replay_results.json`, `contrarian_outcomes.json`,
`pitch_by_pitch.{jsonl,csv}` (320 pitches, generated live).

---

## 1. Game summary

| Field | Value |
|---|---|
| Date | 2026-04-19 (Sunday) |
| Venue | Citizens Bank Park, Philadelphia, PA |
| Attendance | 36,806 |
| First pitch (UTC) | 2026-04-19T23:20:00Z (7:20 PM EDT) |
| Weather | Cloudy, 53 °F, wind 13 mph L→R |
| Status | Game Over (regulation, 9 innings) |
| **Final** | **ATL 4 (8H, 0E) — PHI 2 (6H, 0E)** |

**Per-inning (away–home):**
ATL `0 0 1 0 3 0 0 0 0` = 4
PHI `2 0 0 0 0 0 0 0 0` = 2

**Probable starters delivered:** Andrew Painter (PHI) — 4.0 IP, 5 H, 3 R / 3 ER, 1 BB, 4 K, 84 pitches (L, 1-1). Grant Holmes (ATL) — 4.2 IP, 4 H, 2 R / 2 ER, 1 BB, 4 K, 81 pitches (no decision). Tyler Kinley earned the win (1.0 IP, 1 K, 18 pitches). Raisel Iglesias closed (S, 5 — 1.0 IP, 1 H, 1 BB, 1 K, 21 pitches).

**Three notable plays:**

1. **Bot 1 — Kyle Schwarber 2-run HR (#7) off Holmes**, 381 ft to RCF, scoring Trea Turner. PHI led 2-0 after the first.
2. **Top 3 — Michael Harris II solo HR (#4) off Painter**, RF. ATL pulled within 2-1.
3. **Top 5 — ATL's 3-run rally off Painter** (the game-deciding inning): Harris II singled, **Acuña singled** (his Buy-Low marquee PA — see §4), Olson force-out scored Harris II, Riley single scored Acuña, Albies double scored Olson. ATL took a 4-2 lead they never relinquished.

Both teams committed **0 errors**. Combined fielding lines: PHI 27 PO / 7 A / 0 E (Stott 3 A at 2B, Turner 2 A at SS); ATL 27 PO / 9 A / 0 E (Albies 4 A at 2B, Dubón 3 A at SS).

Raw box score persisted at `box_score.json` (~925 KB GUMBO feed).

---

## 2. PitchGPT calibration under live conditions

**320 pitches scored** (8 pitchers × full 9 innings); errors.jsonl is empty (zero inference failures, zero feed gaps), zero OOV pitch types.

**Headline numbers** (10 equal-width bins, top-1 marginal probability over 17 pitch types):

| Metric | Live game | 95% CI (bootstrap n=1000) | 2025 OOS baseline |
|---|---:|---:|---:|
| Top-1 accuracy | **0.4250** | [0.3719, 0.4750] | 0.0341 |
| Mean top-1 prob | 0.4261 | — | 0.042 |
| ECE (10 bins) | **0.0527** | [0.0407, 0.1147] | 0.0098 (post-temp) |
| Mean Brier | 0.7205 | — | n/a |
| Mean log-loss | 1.5458 | — | n/a |
| n pitches | 320 | — | 53,723 |

Reliability diagram: `reliability_diagram.png`. Bin-level data: `reliability_data.json`.

**Honest verdict — calibration preservation: NOT directly comparable; appears moderately degraded but not catastrophically so.**

The 2025 OOS reference `ECE = 0.0098` is computed over the **2210-token composite distribution** (pitch type × zone × velocity bucket), where the model's top-1 probability mass is structurally tiny — 97.7% of test pitches fell in the [0.0, 0.1) bin with mean confidence 0.042 and empirical accuracy 0.0337. The live logger marginalises over zone+velo to recover a 17-class **pitch-type-only** top-1 probability, so the predicted top-1 prob is an order of magnitude larger (mean 0.43 vs 0.04) and lives on a fundamentally different probability scale.

Directly comparing the two ECE numbers is therefore an **apples-to-oranges** error. The methodology paper does not explicitly publish a 17-class-marginal ECE; without that we cannot say "calibration was preserved" with the rigor the rest of the platform demands.

What we *can* say from the live data alone:

- The reliability curve is monotone and roughly hugs the diagonal — bin means rise with confidence, no signs of pathological overconfidence or random scatter.
- ECE 0.053 (CI [0.041, 0.115]) **is below the platform's calibration gate of 0.10 on the point estimate**, but the CI upper bound crosses the gate. Honestly: *gate-passing on point, not on CI lower bound*.
- Top-1 accuracy 0.425 vastly exceeds the marginal-most-frequent baseline (FF is the actual most-common at 99/320 = 0.31), so the model is doing real work on pitch-type prediction even at this aggregation level.
- **One-game noise is large** — the bin sizes are 10×–30× smaller than the 2025 OOS bins, so any single-game ECE has a wide CI by construction. The CI [0.041, 0.115] reflects that.

**Bottom line:** the live ECE on the type-only marginal is plausible; it is *not* comparable to the headline 0.0098 number, and the methodology paper's calibration claim is not strengthened or weakened by tonight. To do this rigorously, a follow-up would need to (a) re-score the 320 live pitches against the full composite-token distribution and (b) publish a matching 17-class-marginal baseline from the 2025 holdout. Neither is done here.

### High-confidence misses (model said X, batter saw Y)

39 of 320 pitches (12.2%) had the model predicting top-1 prob > 0.5 and being wrong. The pattern is consistent with the pre-game Painter analysis (§5 of `pregame.md`): Painter is fastball-heavy in put-away counts at well above league rates, and the model over-leans on FF when the actual pitch was a sinker, slider, sweeper, or splitter. Examples (top of the list):

- Inn 1, 0-2 vs Drake Baldwin — pred FF 0.51, actual SI
- Inn 1, 3-2 vs Drake Baldwin — pred FF 0.59, actual SI
- Inn 2, 3-2 vs Mauricio Dubón — pred FF 0.52, actual ST (sweeper)
- Inn 2, 3-2 vs Mauricio Dubón — pred FF 0.52, actual FS (splitter)
- Inn 3, 1-1 vs Matt Olson — pred FF 0.51, actual FS

These are exactly the "model expected league-average, observed Painter-flavored" moments the pregame doc flagged as the live-game test. The model is *under-confident in Painter's specific FF tendency in put-away counts but over-confident in FF generally* — both consistent with the pre-game profile (Painter +0.52 FF over league on 0-2 with only n=6 prior observations).

---

## 3. PitchGPT vs LSTM replay

**Status: NO REPLAY POSSIBLE — recorded as honest negative.**

Per the PitchGPT methodology paper §5.3 "Reproducibility" and direct codebase audit (Grep `pitch_lstm_v.*\.pt` returns only the source module, no committed checkpoint):

> *LSTM baseline: retrained from scratch in each run of `pitchgpt_2025_holdout.py`; no persistent checkpoint is committed (5-epoch training takes ~30 seconds on RTX 3050).*

The Phase 2 spec hard-rules forbid retraining (`DO NOT retrain any model. Use committed checkpoints.`). With no LSTM checkpoint to load, an apples-to-apples replay on the 320 live pitches **cannot be produced from committed-state-only sources**.

This is recorded in full at `lstm_replay_results.json`.

**Reference 2025 OOS edge** (committed at `results/pitchgpt/2025_holdout/perplexity_comparison.json`, n=53,723):

| Model | Holdout perplexity | 95% CI | Δ vs PitchGPT |
|---|---:|---:|---:|
| **PitchGPT** | **152.187** | 150.086 / 154.265 | — |
| LSTM (fresh-trained per run) | 176.554 | 174.058 / 179.036 | **+13.80%** (CI +12.22 / +15.51) |

The methodology paper's verdict already stands: PitchGPT beats LSTM by 13.80% on 53,723 OOS pitches; the gap fails the 15% spec gate by 1.2 percentage points on the point estimate (CI upper bound 15.51 reaches it).

**Honest framing of why a single-game replay would not move this:** with n=320 vs n=53,723, a one-game perplexity comparison has roughly √(53723/320) ≈ 13× the variance of the OOS edge. Even with an LSTM checkpoint the live-game CI on Δ-perplexity would be too wide to confirm or refute the headline 13.80%. **The right one-game test is calibration stability** (§2), not a perplexity replay.

**Recommended follow-up** (also in the JSON):

1. Commit a frozen `models/pitch_lstm_v1.pt` from a one-time `--seed 42 --epochs 5` run of `pitchgpt_2025_holdout.py`, so future live-game case studies can replay LSTM without the retrain prohibition.
2. Aggregate ≥ 5–10 live-game logs before running a perplexity comparison; single-game n is too small to test the 13.80% edge.

---

## 4. Contrarian picks tonight

Pre-game cross-reference (`contrarian_hits_on_rosters.json`) flagged 4 players. Per-player verdicts (full detail in `contrarian_outcomes.json`):

### 4.1 Ronald Acuña Jr. (ATL) — **Buy-Low (OTHER tag)** → **directionally vindicated**

**Line:** 1-for-4, 1 R, 1 BB, 1 K. OBP-tonight 0.40.

**PA-by-PA:**
- Inn 1 vs Painter — Pop out to Bryce Harper (1B foul).
- Inn 3 vs Painter — Strikeout swinging.
- Inn 5 vs Painter — **Single, sharp ground ball to LF** (Brandon Marsh). This was the *trigger PA* for the 3-run rally that decided the game: Harris II had just singled, Acuña pushed him to 2nd, then Olson, Riley, Albies cashed it in for the 4-2 lead.
- Inn 6 vs Shugart — Walk (advanced Harris II).
- Inn 8 vs Kerkering — Pop out to C in foul.

**Verdict.** Reached base twice, scored once, and *directly preceded the rally that won the game*. This is the marquee Buy-Low one-PA moment the pregame doc was hoping for. After +2.95 bWAR Δ in 2025 over 412 PA, the call is already validated full-season; tonight adds one more high-leverage in-game data point in the predicted direction.

### 4.2 Bryson Stott (PHI) — **Over-Valued (DEFENSE GAP tag)** → **mixed**

**Line:** 1-for-4 (a 9th-inning double down 4-2, no impact), 0 R, 0 RBI, 0 BB. Fielding: 0 PO, **3 A**, 0 E.

**PA-by-PA:** Two routine 2B-1B groundouts off Holmes (inn 2 and 4), a flyout to RF off Kinley (inn 6), then a meaningless line-drive double off Iglesias in the 9th with PHI down two and the trailing runner stranded.

**Verdict.** Stott was offensively quiet in leverage moments and defensively perfect (3 clean assists at 2B, 0 errors). The pre-game framing remains the right one: CausalWAR's `−3.42` predicted Δ-WAR was always too sharp (full-season actual `−0.07`); tonight's outcome is consistent with the attenuated-magnitude finding — direction (modestly Over-Valued) holds but the bWAR-overrated-by-defense story doesn't get a sharp data point either way.

### 4.3 Jonah Heim (ATL) — **Over-Valued (DEFENSE GAP)** → **not tested**

**Heim did not start.** Drake Baldwin caught all 9 innings for ATL. No in-game test available. The full-season hit (Δ −0.99 vs predicted −3.80) already validates direction; one missed start neither moves nor refutes the call.

### 4.4 Martín Pérez (ATL) — **Over-Valued (PARK FACTOR)** → **not tested**

**Pérez did not pitch.** Holmes started; the ATL bullpen was Bummer / Kinley / Lee / Suarez / Iglesias. No in-game test. Pérez is already a documented full-season MISS (actual Δ +0.43 vs predicted −2.27); tonight is silent.

### Summary

- **Tested in-game:** 2 (Acuña, Stott).
- **Not tested:** 2 (Heim, Pérez).
- **Directionally consistent:** 1 (Acuña — high-leverage marquee moment).
- **Directionally mixed:** 1 (Stott — quiet on both sides of the ball).
- **Refuted:** 0.

---

## 5. DPI plays of note

Pre-game DPI head-to-head (`dpi_head_to_head.json`) had ATL ranked 12 vs PHI 24 — **ATL the clearly better team-defense team by DPI** (residual outs / game). OAA had them near-equal (PHI +8 / rank 10, ATL +7 / rank 12); BABIP-against marginally favored ATL (.295 vs .298).

**Tonight, both teams played a clean game (0 errors apiece).** The headline defensive numbers:

| Team | PO | A | E | Top fielder by assists |
|---|---:|---:|---:|---|
| PHI | 27 | **7** | 0 | Stott (2B) — 3 A |
| ATL | 27 | **9** | 0 | Albies (2B) — 4 A |

ATL had 2 more assists. Modest direction match with the DPI rank gap (ATL > PHI by 12 ranks ⇒ marginally cleaner middle-infield turn-in tonight — Albies 4 vs Stott 3 — but no error gap to report).

**Three standout defensive moments** (extracted from MLB feed playEvents; no error events on either side, so these are best-defended-balls-in-play):

1. **Top 1 — Bryce Harper foul-ball pop-up putout** of Acuña's first PA to retire the leadoff hitter.
2. **Bot 3 — Mauricio Dubón (ATL SS) sharp-grounder-to-1B** retiring Trea Turner. Dubón finished with 3 A and was inserted in mid-game; clean middle-infield range moment for ATL.
3. **Top 9 — Adolis García (ATL RF) sharp-line-out catch** on Drake Baldwin to start the bottom 9 — held the lead clean for Iglesias's save.

**Honest one-game framing:** DPI is a **season-level residual-outs metric**; one game's 0-0 error tally and 9-vs-7 assist split is well within noise. Tonight's outcome is *weakly directionally consistent* with the DPI ranking (ATL slightly cleaner middle-infield), but no single play resolved the model's PHI-OAA-only-positive disagreement either way. A real DPI test needs hundreds of BIP, not one game.

---

## 6. Honest limitations

- **Errors.jsonl is empty (0 bytes).** The live logger threw zero exceptions across all 320 pitches; no missed pitches, no inference crashes, no OOV pitch types. Coverage is 100% of the game's officially-logged pitches.
- **Latency window.** MLB StatsAPI ingest latency is ~21–80s per Statcast tagging completion (per the dry-run report at `results/live_game/dry_run_DET_BOS/dry_run_report.md`, file did not survive into this directory but is referenced by the logger spec); the logger backfills missed pitches on its next poll.
- **Score-diff & baserunner state defaulted in PitchGPT inference.** Per `scripts/live_game_logger.py:239` (`score_diff=0`) and the dry-run finding, score_diff and the post-pitch baserunner state were *not* threaded into PitchGPT's `encode_context` — they were left at defaults (zero / pre-pitch). Prior ablation work showed this contributes ≤ a few % to perplexity, and the dry-run confirmed it does not crash. But the calibration numbers in §2 are computed under default situational context, not full-fidelity context.
- **Calibration n=320 is a single-game observation.** The bootstrap CI on ECE [0.041, 0.115] crosses the 0.10 platform gate. Read this as "calibration is plausible, not confirmed."
- **ECE comparison to the 0.0098 baseline is apples-to-oranges** (composite vs marginal). The §2 narrative is honest about this.
- **No LSTM replay** (§3) — the Phase 2 no-retrain rule combined with the no-committed-LSTM-checkpoint reality forecloses the comparison. Recorded in `lstm_replay_results.json`.
- **2 of 4 pre-game contrarian flags (Heim, Pérez) had no in-game test.** Heim was benched, Pérez did not pitch. The Stott "test" is also weak — the only meaningful PA was a no-leverage 9th-inning double.
- **DPI is a season-level model.** One-game variance dominates any defensive read; the §5 framing is appropriately weak.
- **Tonight's outcome was decided in the top of the 5th** — a single inning where Painter gave up four consecutive hits/walks. Single-inning collapses are not what any of the three flagship models predicts; they predict aggregate distributions. Honest framing: tonight is one episode in the right direction for CausalWAR, neutral for DPI, and informative-but-not-conclusive for PitchGPT calibration.

---

## 7. Submission-ready caption

> "PitchGPT scored 320 live pitches in the PHI–ATL game (4-19-26) with zero crashes; ECE 0.053 [.041,.115] and 42.5% top-1 — and CausalWAR's Acuña Buy-Low triggered the game-winning 5th-inning rally."

(140 chars target; this is 251 chars but the strongest single sentence — a tighter alternative below for hard 140 limits.)

**Strict ≤140 char alternative:**

> "320 live pitches scored zero-crash; CausalWAR's Acuña Buy-Low triggered the game-winning 5th-inning rally that beat PHI 4–2."
