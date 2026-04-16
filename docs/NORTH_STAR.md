# NORTH STAR

## Mission

Win an innovation award on a predictive baseball model. The bar is world-class data-science rigor: methodological novelty, reproducible results, temporally honest validation, and published baselines. This document is the authoritative record of the strategy, the current state of the platform, and the sequence of work required to cross the finish line. Future Claude sessions should treat it as the single source of truth and begin each session by reading it.

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
