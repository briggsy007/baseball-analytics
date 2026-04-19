# NORTH STAR

## Mission

Build a baseball analytics platform that sees what the best MLB front offices see — surfacing edges the public analytics (FanGraphs, Baseball Savant) miss — and validates each edge with world-class data-science rigor. The award is downstream: top-tier methodology + tweetable real findings is what wins it. This document is the authoritative record of the strategy, the current state of the platform, and the sequence of work required to cross the finish line. Future Claude sessions should treat it as the single source of truth and begin each session by reading it.

**Strategy reset 2026-04-18 (Path 2).** The original NORTH_STAR (2026-04-16) framed the mission as pure validation rigor for three flagships — finish all tickets, then write papers, then live-game case study. Today's work demonstrated that **the most pitchable findings come from edge surfacing, not from completing more rigor tickets in isolation.** The Buy-Low contrarian leaderboard validating at 68% in 2025 is more convincing to an award judge than a CausalWAR ablation study would be. Going forward: edge-defining over gate-passing. Missing ticket-level rigor work is a follow-up to strengthen papers, not a blocker. See "Strategy reset" section below.

## Current state (as of 2026-04-16)

- 16 proprietary ML models built on 7.3M Statcast pitches (2015-2025).
- 25 dashboard views, all model-backed (no decorative mocks).
- 784 tests passing.
- Phase 1 validation groundwork landed in five commits: commit sweep, `tests/baselines.json` scaffold, three flagship validation specs, regression verification.

The platform is model-rich and validation-poor. The reason no innovation award is in hand yet is simple: zero flagship has a published, peer-reviewable backtest. Award committees read rigor, not breadth. That gap is what Phase 2 is designed to close.

## Strategic pivot

Stop adding models. Start validating them. Three flagships have been chosen for rigorous validation; the remaining 13 are held back — intact, tested, usable in the dashboard, but deliberately not featured in the award submission. Post-award they become the second-wave expansion story. This pivot is non-negotiable: any new-model work before Phase 2 completes is scope creep and weakens the submission.

## The three flagships

**CausalWAR** (`src/analytics/causal_war.py`, `class CausalWARModel`). Double Machine Learning player-value estimation with Frisch-Waugh-Lovell residualization and cross-fitted `HistGradientBoostingRegressor` nuisance models. The innovation is applying Chernozhukov-style DML to player valuation — a methodology mainstream causal-inference economists would recognize but that baseball analytics has not previously deployed at scale. Biggest validation risk: the model currently has no temporal train/test split, so every reported effect is in-sample and CIs are optimistic. Win condition: Spearman rank correlation versus FanGraphs/B-Ref WAR ≥ 0.6 on a 2023-2024 held-out test set with validated 95% CI coverage in [93%, 97%].

**PitchGPT** (`src/analytics/pitchgpt.py`, wrapper `class PitchGPT`, network `class PitchGPTModel`). A 4-layer, 4-head, 128-dim decoder-only transformer predicting the next pitch from composite (type × zone × velocity) tokens conditioned on game situation. Clearest "first of its kind" narrative — no prior public work applies a causal-masked transformer to Statcast pitch sequences with integrated situational context. Biggest validation risk: no date-based train/test split (splits are by `game_pk`/`pitcher_id` only, so pitcher identity and era bleed across splits) and the `score_diff` context is hard-coded to 0 at line 365, so the contextual claim is currently false. Win condition: transformer perplexity beats LSTM by ≥15%, Markov-2 by ≥20%, heuristic by ≥25% on a 2024 temporal holdout, with calibration ECE < 0.10.

**MechanixAE** (`src/analytics/mechanix_ae.py`, network `class MechanixVAE`, wrapper `class MechanixAEModel`). A β-VAE over 20-pitch windows of 10D biomechanics features producing a Mechanical Drift Index (MDI), intended as an injury early-warning signal. Highest-impact headline if validated — "unsupervised model predicts pitcher injury 10+ days early" is a press-ready story. Biggest validation risk: the model is fully unsupervised and has never been compared to actual IL stints; there is no evidence MDI predicts injury. Win condition: AUC ≥ 0.65 on 30-day pre-IL classification, with delta AUC ≥ 0.10 over a velocity-drop baseline, median lead-time ≥ 10 days, and false-positive rate at MDI ≥ 80 ≤ 20%.

## Phase plan

- **Phase 1 (week 1, complete).** Commit sweep, `tests/baselines.json` scaffold, three validation specs produced and written to disk. Five commits landed. Regression check clean.
- **Phase 2 (weeks 2-6).** Execute validation tickets. MechanixAE injury-label ingestion first (fastest derisk — the entire spec is blocked until labels exist). In parallel: CausalWAR temporal split + FanGraphs/B-Ref baseline comparison; PitchGPT `score_diff` fix + LSTM baseline.
- **Phase 3 (weeks 7-9).** Write 5-page methodology papers per flagship plus a reproducibility guide. Each paper is written to be submissible without further editing.
- **Phase 4 (weeks 10-12).** Live-game case study — a concrete, time-stamped award-submission artifact showing the three flagships operating together on a real game.

## The 13 held-back models

Held intact for post-award expansion. Listed for inventory completeness (count is approximate — several are consolidation candidates):

- **Stuff+** — per-pitch quality score vs league.
- **ChemNet** — team-chemistry graph-neural-network.
- **Allostatic Load** — cumulative pitcher-stress index.
- **Viscoelastic Workload** — time-decayed workload accumulator.
- **MESI** — multi-event situational index.
- **LOFT** — lineup optimization / fatigue tracking.
- **Baserunner Gravity** — baserunner disruption field.
- **PSET** — pitcher-state estimation toolkit.
- **Alpha Decay** — edge-persistence decay over rolling windows.
- **Volatility Surface** — outcome-distribution volatility term structure.
- **Sharpe Lineup** — risk-adjusted lineup construction.
- **Kinetic Half-Life** — arm-kinetics decay tracking.
- **Pitch Decay** — per-pitch-type degradation curves.
- **Defensive Pressing** — defensive-positioning pressure index.
- **Anomaly Detection** — generic outlier flagging.
- **Bullpen** — reliever usage and leverage.
- **Matchups** — batter-vs-pitcher projection.
- **Win Probability** — live WP estimator.
- **Sequencing** — pitch-sequence analytics (overlaps with PitchGPT).

Consolidation candidates to resolve later: Kinetic Half-Life vs Pitch Decay; Alpha Decay vs PSET vs PitchGPT.

## Where to pick up next session

Start by reading `docs/models/*_validation_spec.md` for ticket lists.

**First action:** Phase 2, MechanixAE injury-label ingestion — see `docs/models/mechanix_ae_validation_spec.md`, Ticket #2. This is the single highest-priority unblocker: every downstream MechanixAE ticket depends on `data/injury_labels.parquet` existing.

## Links

- `docs/models/causal_war_validation_spec.md`
- `docs/models/pitchgpt_validation_spec.md`
- `docs/models/mechanix_ae_validation_spec.md`
- `tests/BASELINES_README.md`

---

# Strategy reset — 2026-04-18 evening (Path 2)

## What changed

The 2026-04-16 NORTH_STAR framed success as completing per-flagship validation tickets, then writing papers, then a live-game case study, with new-model work explicitly defined as scope creep. The 2026-04-18 working session demonstrated three things that rebalanced the strategy:

1. **MechanixAE PIVOT was confirmed empirically** (AUC 0.387 < random). The injury-EWS framing is dead; descriptive-profiling reframe leaves a model with 2/5 gates passing. Cannot lean on it as a flagship.
2. **Held-back models contained genuine flagship-quality findings.** VWR PASSed 5/5 under a seasonal-residual reframe (residual AUC 0.768 — the raw 0.56 was a calendar artifact). Defensive Pressing PASSed 6/6 with external Statcast OAA validation (r=0.55). Both are now full flagships.
3. **Edge surfacing produced a more pitchable headline than ticket completion would have.** The contrarian leaderboard (CausalWAR vs bWAR, where my model most disagrees) shows a 68% Buy-Low hit rate in 2025 — Robert Garcia, Randy Rodríguez, Shelby Miller, Fernando Cruz all delivered. That single number is more compelling to an award judge than a CausalWAR ablation study or a hyperparameter grid would be.

## Current flagship four (replaces "three flagships" section above)

- **CausalWAR** (`src/analytics/causal_war.py`). DML for player run-value. PASSes spec gates (Pearson r=0.71, Spearman ρ=0.62 vs real bWAR). Per-cohort lower CIs both clear 0.60 at pa=300/ip=50. Combined Spearman lower CI=0.58 is a documented, well-explained methodological finding (cross-position rank-mixing irreducibility), not a defect.
- **PitchGPT** (`src/analytics/pitchgpt.py`). Decoder-only transformer for pitch sequences. Pitcher-disjoint splits enforced after fixing 409-pitcher leak today. Calibration mint (ECE 0.020 → 0.010 with temperature). Ablation gate FAILs because contextual features add only 1.0–1.4% lift over tokens-only at 1K games / 5 epochs — an honest scaling-study question, not a defect.
- **Viscoelastic Workload (VWR)** (`src/analytics/viscoelastic_workload.py`). Per-pitcher load model. PASS 5/5 under seasonal-residual gate: residual AUC 0.768, delta vs vel-drop +0.213, lead-time 36 days, FPR 12.4%, sanity check passes. The "raw AUC misleads, residual AUC reveals real biomechanical signal masked by season-day confounding" is the strongest single methodology narrative.
- **Defensive Pressing (DPI)** (`src/analytics/defensive_pressing.py`). HistGB xOut model + team-aggregated DPI. PASS 6/6 with external validation (DPI vs Statcast OAA r=0.549). Persisted xOut checkpoint. Only team-defense metric in the platform.

**Demoted:**
- **MechanixAE**: pivoted to descriptive profiling. New profiling spec at 2/5 gates. Stays in platform with experimental dashboard banner; not flagship.

**Documented non-flagship:**
- **Allostatic Load**: built validation surface today; 1/4 gates pass; per-season normalization is the structural blocker.

**Permanently retired from flagship-candidate list:**
- **ChemNet v1 + v2**: r=0.089 (v1) and 0.155 (v2 with opposing-pitcher node + residual objective). Architecture genuinely doesn't learn lineup synergy at this scale.
- **volatility_surface**: clean null on the predictability-tax hypothesis (r=-0.013, p=0.89). Kept as descriptive viz only.

## Live edge products (new — not in original NORTH_STAR)

- **Contrarian Leaderboards** (`src/dashboard/views/contrarian_leaderboards.py`). Two boards: "Buy-Low (CausalWAR > bWAR)" and "Over-Valued (bWAR > CausalWAR)". Methodology-tag heuristic (RELIEVER LEVERAGE GAP / PARK FACTOR / DEFENSE GAP / GENUINE EDGE?). 2025 follow-up column with real WAR (post-backfill). **Buy-Low hit rate 68.4% (13/19) using real Δ-WAR.** This is the marquee edge product.

## Phase scorecard (updated)

- **Phase 2 (validation tickets).** Original target: complete 8 + 14 + ~10 tickets across CausalWAR / PitchGPT / MechanixAE. **Reality:** CausalWAR 2/8, PitchGPT 5/14, MechanixAE pivoted. **Plus** new validation surface for VWR (PASS), DPI (PASS), Allostatic Load (FAIL), Projections (PARTIAL), ChemNet (FAIL), volatility_surface (NULL), MechanixAE-profiling (PARTIAL). The original tickets are not abandoned — they become **rigor follow-ups** that strengthen the eventual papers but no longer block paper-writing.
- **Phase 3 (methodology papers).** Original target: 5-page paper per flagship, reviewable without source code. **Reframed:** each paper now leads with the **edge story** (what does it tell a GM that public analytics doesn't?) and follows with methodology + validation. Four flagships → four papers. Plus one cross-flagship "headline findings" doc.
- **Phase 4 (live-game case study).** Original target stands — one Phillies game with all flagships running. **Augmented:** the contrarian leaderboard refresh + at-risk-pitcher board (VWR scores for active MLB pitchers, refreshed weekly) + DPI live team rankings should also be live by Phase 4.

## Held-back model status (updated from 13 to current)

Promoted to flagship (no longer held back): **VWR, Defensive Pressing.**
Validated and retired: **ChemNet, volatility_surface.**
Built non-flagship validation surface: **Allostatic Load, Projections, MechanixAE-profiling.**
Still genuinely untouched (next promotion candidates per platform audit at `docs/edges/platform_audit.md`):
- **Stuff+** — flagged as commodity, low edge potential
- **MESI, LOFT, Baserunner Gravity, PSET, Alpha Decay, Sharpe Lineup, Kinetic Half-Life, Pitch Decay, Anomaly Detection, Bullpen, Matchups, Win Probability, Pitch Sequencing**

## Phase 5 — Next-session execution plan (Path 2)

**Order is by leverage on the award narrative, not by complexity.**

### 5A. Methodology papers (parallelizable, ~1 agent each)

Write four `docs/awards/methodology_paper_<flagship>.md` documents (5 pages each), structured as: Edge → Methodology → Validation → Limitations → Reproducibility. Pulls from existing spec + results docs + the contrarian leaderboards / OAA-vs-DPI / seasonal-residual artifacts. Reviewer-grade self-contained.

Plus one `docs/awards/headline_findings.md` — the tweetable summary doc. One paragraph per flagship + one paragraph on the contrarian leaderboards + one paragraph on documented negatives (MechanixAE pivot, ChemNet retirement). This is the doc you can hand a journalist.

### 5B. Edge product expansion (sequential, each ~1 agent)

1. **At-risk pitcher board** — VWR scores for active 2024-2026 MLB pitchers, dashboard view, weekly refresh. Will require extending VWR pipeline coverage past 2015-2016 (only 64 fits exist currently).
2. **DPI live team rankings** — defensive_pressing scores for current MLB teams, dashboard view, weekly refresh.

### 5C. Data depth (always backfill)

1. **TJ surgery dates 2017-2025** — unblocks the dormant Projections v2 TJ flag (Walker Buehler example: model has the feature wired but the data is empty for his 2022 surgery). Likely source: `pybaseball` IL-stint data with `surgery_type` filter, or a curated source.
2. **Season-by-season player roster** — current `players.team` is point-in-time only. Tracking roster history would unlock per-season cohort assembly for several models.

### 5D. Rigor follow-ups (lower priority, but include in papers when complete)

- CausalWAR Tickets 3 (DML ablation), 4 (CI coverage), 5 (feature importance), 6 (residual diagnostics), 7 (reproducibility script). These will strengthen the CausalWAR paper but are not blockers for it.
- PitchGPT Tickets 4 (Markov baseline), 5 (heuristic baseline), 8 (disruption→outcome regression), 12 (results report). Markov + heuristic baselines especially — they're cheap and they substantiate the "transformer beats simpler baselines" claim.

### 5E. Phase 4 — Live-game case study

Once Phase 5A-5C are landed, run a real-time integration test on a Phillies game: contrarian leaderboards, at-risk pitcher board, DPI live, PitchGPT calibrated next-pitch predictions. Time-stamped artifact = the award submission's demo piece.

## What we explicitly do NOT do under Path 2

- Build new models from scratch (per the original NORTH_STAR clause — still applies; today's Projections + MechanixAE-profiling were edge-case exceptions and they should be the last new-model builds before papers ship).
- Chase made-up "lower-CI ≥ gate" mint criteria when the spec gate is already met. Honest CI reporting is the bar.
- Re-litigate decided pivots (MechanixAE → descriptive; ChemNet → retired; volatility_surface → retired).

---

# Post-evidence consolidation — 2026-04-18 late evening

## Context

The Path 2 reset above posited four flagships (CausalWAR, PitchGPT, VWR, DPI) based largely on in-distribution validation. Before writing papers, we ran a four-agent evidence-strengthening pass in parallel: each flagship tested against 2025 out-of-sample data and, for VWR, against an expanded-coverage cohort. Results changed the flagship count and narrowed two claims.

## Flagship status (post-evidence)

- **DPI — reinforced.** 2025 Pearson r=0.641 vs Statcast OAA (CI [0.42, 0.79]) — the highest of three years (0.58 / 0.56 / 0.64). DPI-to-BABIP-against r=-0.80 vs OAA's -0.44: DPI is the tighter BIP-outcome residual metric. Disagreements with OAA are interpretable as genuine edges (DPI captures outcome pressure OAA removes as "positioning credit"). Top-5 overlap with OAA is 2/5 across all three years, consistent with r≈0.6. **This is now the platform's strongest external-validation result, and the single most defensible flagship claim.**

- **CausalWAR — real edge, narrowed claim.** 2025 68.4% Buy-Low hit rate reproduced exactly (13/19, CI [0.47, 0.84]). Symmetric Over-Valued side validates at 60.9% (14/23). Year-over-year Buy-Low lift over a WAR-matched naive baseline: **+7.8pp (2022→23), -2.8pp (2023→24), +10.8pp (2024→25)**. Mechanism-tag breakdown: RELIEVER LEVERAGE GAP is the durable Buy-Low driver (67–90% across three windows); PARK FACTOR and DEFENSE GAP drive Over-Valued. The edge is real and mechanism-coherent but **not monotonically stable** — the 2023→24 underperformance vs naive is a honesty-note, not a fatal flaw. At n=19, 68% is statistically indistinguishable from 45–50%.

- **PitchGPT — calibration clean, LSTM gate fails by 1.2pp.** Pitcher-disjoint 2025 holdout (334 pitchers new to 2025, 53,723 tokens, 0 leakage): PitchGPT perplexity 152.19 vs LSTM 176.55 = **13.80% improvement, CI [12.22, 15.51] — FAIL vs 15% spec gate**. Markov-2 (56.85%) and heuristic (67.75%) gates pass with tight CIs. Calibration survives OOS: ECE 0.0098 post-temperature at T=1.120. The LSTM gap is real but thin, matching the prior 12.7% 2024 finding and the ablation story (context adds 1–7% lift). **Defensible claim narrows to: "calibrated transformer sequence model that survives temporal shift and beats naive baselines by wide margins."** Drop the "beats every baseline by spec margin" framing.

- **VWR — retracted from flagship.** Expanded cohort (n=563 injured fits, 2017–2024) residual AUC **0.438 (CI 0.425–0.452) — below chance**. 2025 holdout residual AUC 0.493 (coin flip), median lead-time 12.5d (vs 30d gate), delta vs vel-drop +0.002. Per-injury AUCs collapsed (shoulder 0.62→0.37, elbow 0.60→0.45; only rotator_cuff n=11 hints at signal). The 0.77 residual AUC on the original 64 fits was a **small-sample artifact of 2015–2016 label-calendar structure** — season_day-alone AUC moves from 0.27 (2015–16) to 0.45 (expanded), so partial-out stops lifting raw AUC. The "seasonal-residual reveals buried biomechanical signal" methodology finding does not replicate. Model retained as a descriptive workload-tracking tool (fits converge, FPR calibrates at 9.8%, saturation well-behaved), but **permanently off the flagship-candidate list** — do not re-promote.

## Updated flagship three

1. **DPI** — strongest external validation (r=0.641 vs OAA on 2025, three-year stable).
2. **CausalWAR** — contrarian leaderboards edge, mechanism-coherent, year-to-year variable.
3. **PitchGPT** — calibrated sequence model, OOS-stable calibration, sub-spec edge over LSTM.

## Phase 5 plan (post-evidence update)

- **5A methodology papers** — three papers now (no VWR paper). Each leads with the narrowed claim above. DPI paper is the lead document.
- **5B edge-product expansion.** At-risk pitcher board (originally VWR-based) is **cancelled or radically reframed** — VWR residuals cannot drive it. Consider a velocity-drop + pitch-count workload tracker as a simpler substitute, or drop from the award artifact. DPI live team rankings **proceed — now the lead edge product**. Contrarian leaderboard refresh proceeds, with honest CI and the 2023→24 asterisk visible.
- **5C data depth** (unchanged): TJ dates 2017–2025, season-by-season rosters.
- **5D rigor follow-ups.** PitchGPT 1.2pp LSTM gap: accept and narrow the claim (defensible under Path 2) rather than rescue with more compute. CausalWAR 2023→24 underperformance: one-time cohort analysis to see whether a specific pitcher cohort or injury season drove the regression.
- **5E live-game case study** — leaner, three flagships not four.

## What we explicitly do NOT do (post-evidence)

- Re-promote VWR. The small-sample-artifact lesson is locked in.
- Spin sub-spec numbers. PitchGPT's 13.80% LSTM delta is the number; the -2.8pp 2023→24 CausalWAR naive-lift is the number.
- Build new models to replace VWR in the flagship slot. Three strong flagships beats four with one fake.
