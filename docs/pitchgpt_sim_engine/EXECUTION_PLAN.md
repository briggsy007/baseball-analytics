# PitchGPT Sim Engine — Execution Plan

**Date:** 2026-04-24
**Status:** drafted, not started. Phase 0 prerequisites unstarted.
**Purpose of this document:** self-contained multi-session navigational doc. A future Claude session reading only this file plus `docs/NORTH_STAR.md` should be able to pick up and execute. Match the rigor of `docs/live_game/2026-04-19_PHI_vs_ATL/EXECUTION_PLAN.md`.

---

## 1. Start here

**Role.** You are the PM-level assistant coordinating a multi-session build of a PitchGPT-powered **simulation engine**. Per `feedback_pm_role.md`, act as PM — delegate implementation to subagents, do not write code directly. Per `feedback_validation_agent.md`, after any 3+ parallel agent batch run a validation agent (pytest + import spot-check + regression report) before declaring a phase done. Per `feedback_always_backfill.md`, if a phase surfaces a data gap, queue the backfill proactively rather than treating the gap as out-of-scope.

**Goal.** Promote PitchGPT from a calibrated next-token predictor to a calibrated **rollout engine** that produces joint distributions over PA outcomes, matchups, innings, and games — with honest CIs on every emitted quantity. Use the rollouts to power five Tier-A edge products (counterfactual pitch-call grade, probabilistic pitcher projections with CIs, matchup sim, deceptiveness leaderboard) and four Tier-B live decision-support tools. Tier-C items exist behind explicit gates.

**Strategy context.** Per `docs/NORTH_STAR.md` "Post-evidence consolidation — 2026-04-18 late evening":
- Flagships are three: DPI, CausalWAR, **PitchGPT**.
- PitchGPT's current narrowed claim is "calibrated sequence-modeling achievement, not a decision-utility replacement." See `docs/awards/methodology_paper_pitchgpt.md` §1 — the 13.80% LSTM delta (CI [12.22, 15.51]) misses the 15% spec gate by 1.2pp; the downstream pitch-outcome-bucket log-loss delta vs LSTM was 0.00007 (CI brackets zero, a statistical tie). That framing is **locked in** — the sim-engine plan lives inside those constraints, it does not relitigate them.
- Path 2 philosophy: edge surfacing over gate completion. Honest-negatives reporting is the standard.
- Retracted / demoted / retired (do NOT re-litigate): VWR (retracted 2026-04-18 late evening as small-sample artifact), MechanixAE (demoted to descriptive profiling), ChemNet v1+v2 (retired null), volatility_surface (retired null).

**How this document is used.** Section 5 is the phase plan. Section 6 is the per-item dossier (A1–A4, B1–B4, C1–C4) — consult when launching agents. Section 11 is the pickup point each session. Sections 7 and 9 are the risk + validation-gate contracts every item must satisfy before it's allowed to claim an edge in public-facing artifacts (dashboard, papers, leaderboards).

---

## 2. Framing — what "sim engine" means here

PitchGPT as a sim engine rests on **three capabilities** the checkpoint already possesses (from `docs/awards/methodology_paper_pitchgpt.md`, §1–3):

1. **Sequence-aware sampling.** Rollouts respect 2-gram+ dependencies across pitches within a PA (tunneling, set-up pitches, count-driven sequence adaptation). The transformer attention over the prior pitch history is the mechanism. No public product — Savant, FanGraphs, Stathead — exposes calibrated sequence-conditioned sampling at the pitch level. This is a structural edge, not a tuning edge.
2. **Calibrated joint distributions.** Post-temperature ECE = 0.0075 on the 2025 holdout (see `results/pitchgpt/2025_holdout/report.md` most recent regeneration; the methodology paper shows ECE 0.0098 at an earlier measurement window — the two numbers are concordant within measurement noise on a single holdout). Because the 2,210-token composite vocabulary is `pitch_type × zone × velocity_bucket` in one softmax, the model's marginals over each axis are derivable directly from the joint — no product-of-experts factorization is needed. CIs on everything the rollout produces.
3. **Counterfactual conditioning.** The situational context (count / outs / runners / handedness / inning / score_diff / umpire tendency) is wired into `PitchGPTModel.forward` and added to the token + positional embedding before the first transformer layer. See `src/analytics/pitchgpt.py::PitchGPTModel` (line 1095) for the integration and §2.3 of the methodology paper for the feature set. **Re-sampling under counterfactual context is a direct consequence of this — sample the prefix unchanged, then mutate the context vector for the continuation positions.**

What the sim engine produces: for any PA starting context drawn from real 2025+ data, **N rollout samples of length H pitches**, each with per-pitch probability vectors and (post-Phase-0) per-pitch PA-outcome-head probabilities. Aggregations — per-PA wOBA distribution, per-start ERA distribution, per-game WP distribution — are downstream reductions of this N×H tensor.

---

## 3. Flagship-claim constraints

What the sim engine **is allowed** to claim publicly (in dashboards, methodology paper v2, leaderboards):
- **"Calibrated rollout engine"** — the sim emits CI-backed joint distributions with ECE <0.02 post-temperature on 2025 pitcher-disjoint rollouts. This is the primary, unconditional claim. ~~"Calibrated rollout distributions match empirical 2025 joint distributions better than LSTM"~~ was narrowed after Phase 0.1 2026-04-24 — PG beat LSTM on marginals (pitch type, zone, velocity KL / Wasserstein, CIs excluding zero) but LOST on 2-gram Frobenius (CI [0.008, 0.250]; thin under Bonferroni correction but a loss under the locked 95%-uncorrected gate). See §7.5 and §6 item 0.1 status. Allowed language is now "matches empirical marginals with calibrated uncertainty" — do NOT claim sequence-structure superiority over LSTM.
- "PitchGPT rollouts match empirical 2025 *marginal* pitch type / zone / velocity distributions with tighter fit than a matched-architecture LSTM" — supported by Phase 0.1 metrics 1/2/3 results (all CIs exclude zero). This is a narrowed positive claim, not the full sampling-fidelity headline originally scoped.
- "Counterfactual re-sampling preserves calibration" — if and only if an OOS calibration re-check on counterfactual-mutated contexts holds ECE < 0.02. Threshold is **TBD — measured during Phase 1**. If it drifts we narrow the claim.
- "N-sample CIs on any downstream quantity" — CI emission is a core property, not a claim; we emit it unconditionally.

What the sim engine is **not allowed** to claim:
- "Beats every baseline by spec margin." Retired at 2026-04-18 evidence consolidation. The narrowed claim in `docs/awards/methodology_paper_pitchgpt.md` stands.
- "Replaces downstream decision utility." The §3.4 downstream utility tie vs LSTM on the pitch-outcome-bucket target is a locked finding. We may separately show that a **different** downstream target (PA outcome, exit-velocity bucket) unlocks utility — but that is a **new result**, not a contradiction of the old one, and it needs its own CI-backed table.
- "Flagship for umpire-centric edges" — per `feedback_no_umpire_edges_until_abs_drift_check.md`, do not ship umpire-conditioned strategy products without an ABS-era drift study. See Item C1.

Everything in this plan is measured or labeled "TBD — measured during Phase X." If a measurement later contradicts this document, update this document — do not let the dashboard or papers drift ahead of the doc.

### 3.1 Decision log (locked 2026-04-24)

The following three decisions were flagged as open by the drafting agent and were resolved by the user before Phase 0 was green-lit. **Do not re-litigate without explicit user sign-off.**

1. **Phase 0.1 sampling-fidelity exit bar = ≥3 of 5 metrics, with metric 4 (2-gram Frobenius) mandatory among the wins.** Rationale: beating on marginals alone (metrics 1–3) is a weak flagship story — any competent sequence model should land close on base rates. Metric 4 tests the transformer's actual sequence-awareness edge (tunneling, set-up pitches, count-driven sequencing). Winning on marginals while losing on 2-gram does not support the "sequence-aware sampling" pillar of the sim engine framing; if that happens, halt and narrow the claim per §7.5. This is stricter than a vanilla ≥3/5.
2. **PA-outcome head target = 7 classes** (`ball`, `called_strike`, `swinging_strike`, `foul`, `in_play_out`, `in_play_hit`, `hbp`). The richer label space is what unlocks counterfactual pitch-call grading (A1) and matchup sims (A3). Divergence from the methodology paper's 5-class downstream utility test is intentional — that was a post-hoc utility wrapper; this is the primary head.
3. **B3 (anomalous-outing detector) retains its seasonal-residual gate** pre-emptively, modeled on the VWR retraction at 2026-04-18. Raw log-likelihood anomalies are known to confound with season_day; we require the flag-rate-by-season_day F-test to be flat (p > 0.05) before B3 publishes.

---

## 4. Critical path / prerequisites

### 4.1 The single biggest unlock: a PA-outcome head

**All Tier-A items depend on the model producing per-pitch outcomes, not just pitch tokens.** PitchGPT currently predicts the next pitch's `(type, zone, velo_bucket)` composite token. The 2025 holdout work does not produce a ball / called-strike / swinging-strike / foul / in-play-out / in-play-hit / HBP distribution — only the post-hoc XGBoost utility wrapper in `scripts/pitchgpt_downstream_utility.py` does (and that XGB was a tie with LSTM per §3.4).

To sample a rollout that terminates in a PA outcome, we need per-pitch-in-rollout outcome probabilities. Two options, to be evaluated in Phase 0:

- **(a) Auxiliary head trained jointly.** Add a 7-class outcome head to `PitchGPTModel`, train with a weighted cross-entropy loss alongside the next-token loss on the existing 2015–2022 pitcher-disjoint cohort. Outcome labels derive from `pitches.description` and `pitches.events` — both columns exist in the schema (see Appendix §12).
- **(b) Post-hoc head trained frozen.** Freeze the current checkpoint's final hidden state, train a small MLP head on top. Preserves the flagship checkpoint (methodology paper PPL numbers stay valid). Slower training convergence expected because gradients cannot flow into the backbone.

Decision between (a) and (b) happens in Phase 0. **Criterion:** whichever route yields lower OOS log-loss on the 7-class target on a 2025 pitcher-disjoint evaluation window, subject to the constraint that the backbone's next-token OOS ECE does **not** degrade by more than +0.005 absolute (i.e., from ECE 0.0075 to <0.0125). If joint training hurts calibration, fall back to the post-hoc head.

This is **critical path** — no Tier-A item starts until the PA-outcome head lands. Flag it in every agent prompt.

### 4.2 Data requirements

All required columns exist in the current schema (`src/db/schema.py`, pitches DDL lines 119–211). The PA-outcome head and Tier-A downstream items consume:

| Column | Purpose | Coverage status |
|---|---|---|
| `pitches.pitch_type` | token construction + label | populated (filters applied throughout) |
| `pitches.plate_x`, `pitches.plate_z` | zone token construction | populated |
| `pitches.release_speed` | velocity-bucket token | populated |
| `pitches.description` | PA-outcome-head class (ball / called-strike / swinging / foul / HBP) | populated |
| `pitches.events` | PA-terminal outcome (single, double, home_run, strikeout, walk, etc.) | populated |
| `pitches.woba_value` | outcome-grade roll-up (Tier A1) | populated but sparse — only non-null on terminal pitches |
| `pitches.launch_speed` | contact-quality conditioning (Tier A3, B4 extensions) | populated on in-play only — expected sparsity |
| `pitches.delta_run_exp` | outcome-delta ground truth for A1 counterfactual grading | populated but approximate (see `_runs_scored_on_event` in `src/analytics/pitchgpt.py`) |
| `pitches.balls`, `pitches.strikes`, `pitches.outs_when_up`, `pitches.on_1b/2b/3b`, `pitches.stand`, `pitches.inning`, `pitches.inning_topbot` | situational context feature reconstruction | populated |
| `umpire_assignments`, `umpire_tendencies` | umpire context (already wired in v2) | 103,192 rows 2015–2026, 93.6% game_pk-matched |
| `game_weather` | contextual conditioning (DPI v3 wiring, not PitchGPT's direct context) | 25,050 rows, 94.2% coverage |

No backfill blockers for Phase 0. For Tier-B4 (optimal pitch-mix for upcoming opponent lineup), season-by-season roster history is currently incomplete (see post-ingestion next-phase plan §5C item 2). This is flagged as a backfill prerequisite for B4 and **only** for B4 — the other items can proceed on point-in-time roster.

### 4.3 Compute

User memory (`project_environment.md`, `.claude/CLAUDE.md`): Windows 11 + bash, RTX 3050 with CUDA support, torch installed. PitchGPT training is GPU-accelerated. A 10K-game retrain took ~90 min wall-clock (post-ingestion next-phase plan P0.1). The PA-outcome-head training should be materially cheaper than a full backbone retrain — 1–3 epochs on top of a frozen backbone, or at most a double-pass over training data if we go with joint training. **DuckDB is single-writer** — stop the dashboard (`streamlit`) before any training run per `CLAUDE.md` data rules.

### 4.4 Prior art to build on

- `scripts/pitchgpt_sampling_fidelity.py` — already drafted. Read the docstring and first ~400 lines. It reframes PitchGPT's claim around sampling/simulation fidelity rather than next-token PPL. Its output at `results/pitchgpt/sampling_fidelity_2026_04_24/` is Phase 0's first deliverable. The sim-engine plan treats this script as the validating experiment for the "PitchGPT rollouts match empirical 2025 joint distributions better than LSTM rollouts" hypothesis. If the script lands a null result (PitchGPT == LSTM), Section 7 spells out the narrowed claim.
- `src/analytics/pitchgpt.py::PitchGPTModel` (line 1095) — architecture is reusable as-is. The checkpoint at `models/pitchgpt_v2.pt` (committed, working) is the baseline for any head work.
- `scripts/pitchgpt_2025_holdout.py` — produces the holdout splits and calibration temperature scaling that all Phase-0 work piggybacks on.
- `scripts/pitchgpt_downstream_utility.py` — XGBoost-based utility harness. For Tier-A items, the harness is superseded by a model-native outcome head, but the evaluation scaffolding (pitcher-disjoint splits, bootstrap CIs) is reusable.

---

## 5. Phase plan

The phase ordering is by dependency, not by hype. Phase 0 gates everything.

### Phase 0 — Prerequisites (1–2 weeks)

Unblocks all Tier-A items. Run agents in parallel where possible.

| # | Item | Prereq | Effort | Agent-parallelizable? |
|---|---|---|---|---|
| 0.1 | Land `scripts/pitchgpt_sampling_fidelity.py` run — marginals / 2-grams / Wasserstein on 2025 pitcher-disjoint PAs | drafted script | M (~1 day wall-clock, includes training matched LSTM at 10K) | yes, one dedicated agent |
| 0.2 | PA-outcome head design doc — compare joint vs frozen-backbone routes, pick one | 0.1 landed (so we know the sampler is trustworthy) | S (~half day) | yes |
| 0.3 | Train PA-outcome head | 0.2 | M (~1 day wall-clock GPU) | no — GPU-bound, run serially |
| 0.4 | OOS validation of outcome head on 2025 (log-loss, calibration, per-class confusion, per-pitcher stability) | 0.3 | S | yes |
| 0.5 | End-to-end rollout harness: `src/analytics/pitchgpt_sim.py` with a `rollout(context, n_samples, horizon)` API | 0.3 | M | yes |
| 0.6 | Sanity-check rollouts vs empirical PA outcomes on 2025 — marginal K/BB/HR rates, mean wOBA, inning-length distribution | 0.5 | S | yes |

**Phase 0 exit criteria:**
- Sampling-fidelity comparison vs LSTM posted (PASS: PitchGPT better or tied on ≥3 of 5 metrics). **If PitchGPT loses on ≥3 of 5**, halt and narrow to "calibrated but not differentiated" — Section 7.
- PA-outcome head OOS log-loss and ECE measured and < TBD thresholds (set in 0.2 design doc, not invented here).
- Rollout harness produces marginals consistent with 2025 league rates (K%, BB%, HR% within ±10% relative or ±1pp absolute, whichever is tighter).

### Phase 1 — Tier A award-narrative edges (3–5 weeks)

Items are pitchable to a journalist or award judge. Each has its own Section 6 dossier.

| # | Name | Depends on |
|---|---|---|
| A1 | Counterfactual pitch-call grade | Phase 0 |
| A2 | Probabilistic pitcher projections with CIs | Phase 0, A1 scaffolding |
| A3 | Matchup sim with CIs (batter × pitcher dashboard view) | Phase 0 |
| ~~A4~~ | ~~Deceptiveness leaderboard~~ — DEMOTED to Tier C 2026-04-24 post-0.1 | parked |

Run A1 first — it's the most pitchable and it doubles as stress-test for the rollout harness. A2 and A3 are infrastructure-parallel, A4 is last because it benefits from whatever post-sim cleanup A1–A3 surface.

### Phase 2 — Tier B decision-support tools (2–3 weeks, gated on Phase 1 landing at least 2 items)

| # | Name | Depends on |
|---|---|---|
| B1 | Live-game WP upgrade (rollout-based, replaces base-state RE lookup) | A3 |
| B2 | Reliever-change decision support | A2 scaffolding, B1 |
| B3 | Anomalous-outing detector (log-likelihood under own distribution) | Phase 0 |
| B4 | Optimal pitch-mix recommendation for upcoming opponent | A3, roster history backfill |

B3 is the cleanest replacement for the retracted-VWR anomalous-outing use case (see `docs/NORTH_STAR.md`, "post-evidence consolidation" — VWR residual AUC 0.438 on expanded cohort was below chance; PitchGPT own-distribution log-likelihood is a fundamentally different signal and is not burdened by the season-day artifact that killed VWR). **This should NOT be called "VWR successor" in public artifacts** — it's a distinct methodology; frame it as "sequence-likelihood anomaly detection."

### Phase 3 — Tier C (flag-dependent, lower leverage)

| # | Name | Gate |
|---|---|---|
| C1 | Umpire-conditioned strategy | **Hard gate: ABS-era drift check per `feedback_no_umpire_edges_until_abs_drift_check.md`.** Until that drift check lands, C1 is not built, not shipped, not dashboarded. |
| C2 | Pre-AB pitch prediction for advance scouting | No hard gate; low novelty publicly. Ship after C1 drift check if we have time. |
| C3 | Batter-weakness leaderboard (simulated xwOBA per batter × pitch type vs league-avg pitcher) | Phase 2 complete. |
| C4 | Stability / signal-to-noise floor (methodology contribution) | Phase 2 complete. Methodology-paper appendix material. |

Phase 3 is optional. A cleaner submission leads with Phase 1 + Phase 2 and files Phase 3 as "future work" in the v2 methodology paper.

---

## 6. Item dossier

Each dossier follows: Goal / Edge claim / Methodology / Artifacts / Success criteria / Dependencies / Effort / Status.

### Phase 0 items

#### 0.1 — Sampling-fidelity validation (PitchGPT vs LSTM rollouts)

- **Goal.** Confirm PitchGPT rollouts match empirical 2025 marginal + joint distributions better than a matched LSTM's rollouts do.
- **Edge claim (gated on outcome).** "Calibrated transformer rollouts reproduce empirical 2025 pitch-sequence distributions at tighter error than an LSTM." Claim only fires if ≥3 of 5 metrics (marginal pitch-type KL, marginal zone KL, velocity Wasserstein, 2-gram Frobenius, PA-outcome-rate L1) favor PitchGPT with CIs excluding zero.
- **Methodology.** Already drafted — see `scripts/pitchgpt_sampling_fidelity.py` docstring and §1. 2025 pitcher-disjoint PA starts (pitchers not in 2015–2022 train), auto-regressive sampling at T=1.0, N=10 samples per PA start, horizon H=6.
- **Artifacts.** `results/pitchgpt/sampling_fidelity_2026_04_24/metrics.json` + `report.md`. Commit only if PASS; if null, commit with explicit "null result" annotation.
- **Success criteria.** PitchGPT wins or ties on ≥3 of 5 metrics at 95% bootstrap CI **AND metric 4 (2-gram Frobenius) must be one of those wins** — see §3.1 decision log for rationale. PA-outcome-rate L1 must be ≤0.05 on both models (sanity — neither model catastrophically mis-samples).
- **Dependencies.** None — the script is drafted, needs one execution pass plus a matched LSTM train.
- **Effort.** M (~1 day wall-clock, dominated by LSTM training).
- **Status. FAIL — 2026-04-24.** Wall-clock 12 min (much faster than estimated — dataset build 8.6 min, LSTM train 2.5 min, sampling 3 sec each, bootstrap 39 sec). Results at `results/pitchgpt/sampling_fidelity_2026_04_24/`. LSTM checkpoint: `models/pitch_lstm_10k.pt` (val_ppl 117.88).
  - PG wins metrics 1/2/3 (pitch_type_kl, zone_kl, velocity_wasserstein) with tight CIs excluding zero.
  - PG **loses** metric 4 (transition_frobenius): Δ +0.1623, CI [0.0078, 0.2497]. Gate FAIL per §3.1 lock.
  - Metric 5 (outcome_chi2) degenerate — both models emit 100% strikeouts via the zone-heuristic proxy. Becomes informative after Phase 0.3 outcome head lands.
  - Bonferroni caveat: 5 tests → α=0.01 → 99% CI would likely include zero on metric 4 (95% lower bound is 0.008, razor-thin). Under multiple-comparison correction the result is closer to a tie than a loss. The gate as locked is 95% uncorrected, so it FAILs; honest readers should know the margin is slim.
  - Per §7.5 the sim-engine claim narrows to "calibrated rollout engine" (CI-backed) rather than "rollouts match 2025 joint distributions better than LSTM." Tier-A dossiers using the "PG better than LSTM on sampling fidelity" framing must be revised. A4 (deceptiveness leaderboard) is most tightly bound to the sequence-edge claim and is a candidate for Tier-C demotion pending a different validation path.

#### 0.2 — PA-outcome head design doc

- **Goal.** Decide joint vs frozen-backbone architecture for the outcome head.
- **Methodology.** Short design doc at `docs/pitchgpt_sim_engine/pa_outcome_head_design.md` (path reserved; do NOT create until this item runs). Compare: joint CE loss with backbone unfrozen (option a) vs MLP-on-frozen-features (option b). Run a small smoke experiment on 500 games for each to estimate convergence rate and calibration impact.
- **Artifacts.** Design doc + smoke-experiment results.
- **Success criteria.** Design choice made with empirical justification — not a vibe call.
- **Effort.** S (~half day). Actual: ~15 min wall-clock.
- **Status. COMPLETE — 2026-04-24. Route chosen: FROZEN backbone + 2-layer MLP head (d_model=128 → 64 → 7).** Smoke results (500 train / 100 eval, 2023 pitcher-disjoint, n=5,617):
  - Frozen: 7-class log-loss 1.6337, 7-class ECE 0.0103, backbone token ECE 0.0093 (Δ +0.0000 vs reference).
  - Joint: 7-class log-loss 1.6445, 7-class ECE 0.0171, backbone token ECE 0.0219 (Δ +0.0126 — **blows the +0.005 budget by 2.5×**).
  - Backbone-ECE budget disqualifies joint decisively; frozen also beats joint on the objective.
  - HBP class (0.3% freq) had high per-class NLL — flagged for Phase 0.3 class-weighting.
  - Artifacts: `docs/pitchgpt_sim_engine/pa_outcome_head_design.md`, `src/analytics/pitchgpt_outcome_head.py`, `scripts/pitchgpt_outcome_head_smoke.py`, `results/pitchgpt/outcome_head_smoke_2026_04_24/metrics.json`, smoke checkpoints `models/pitchgpt_outcome_smoke_{joint,frozen}.pt`. `models/pitchgpt_v2.pt` untouched.

#### 0.3 — PA-outcome head training

- **Goal.** Produce an outcome-conditional probability at every rollout position.
- **Methodology.** Train per 0.2's decision on 2015–2022 pitcher-disjoint cohort. 7-class target: `ball`, `called_strike`, `swinging_strike`, `foul`, `in_play_out`, `in_play_hit`, `hbp`. Labels derived from `pitches.description` + `pitches.events` join (see `src/analytics/pitchgpt.py` for existing description-parsing patterns).
- **Artifacts.** New checkpoint at `models/pitchgpt_v2_outcomehead.pt`. Does NOT overwrite `models/pitchgpt_v2.pt` — the paper checkpoint stays frozen.
- **Success criteria.** 7-class log-loss lower than a frequency-prior baseline by ≥15% on 2025 pitcher-disjoint holdout. ECE (10-bin) < 0.05 post-temperature. Backbone next-token ECE un-degraded by more than +0.005 (see §4.1).
- **Dependencies.** 0.2.
- **Effort.** M (~1 day GPU).
- **Status.** not started.

#### 0.4 — Outcome-head OOS validation

- **Goal.** Confirm the outcome head is trustworthy before any Tier-A item consumes it.
- **Methodology.** Pitcher-disjoint 2025 holdout. Per-class confusion matrix. Per-pitcher log-loss stability (variance across N=50 top-frequency 2025 pitchers). Reliability diagram per class.
- **Artifacts.** `results/pitchgpt/outcome_head_2025/` directory with `metrics.json`, `report.md`, `reliability_by_class.html`.
- **Success criteria.** Matches 0.3 thresholds on fresh 2025 data; per-pitcher log-loss 5th–95th percentile band width < 0.5 nats (sanity — pitcher-specific drift is bounded).
- **Dependencies.** 0.3.
- **Effort.** S.
- **Status.** not started.

#### 0.5 — Rollout harness `src/analytics/pitchgpt_sim.py`

- **Goal.** Single `rollout(starting_context, n_samples=100, horizon=6)` API that every downstream Tier-A item calls.
- **Methodology.** Load backbone + outcome head, autoregressively sample. Expose: `rollout_pitch_tokens` (n_samples × horizon), `rollout_outcomes` (n_samples × horizon), `per_pitch_prob_vectors` (n_samples × horizon × 2210). PA-termination behavior: stop at first in-play or walk/strikeout outcome; pad with NaN to `horizon`.
- **Artifacts.** `src/analytics/pitchgpt_sim.py`, unit tests at `tests/test_pitchgpt_sim.py`.
- **Success criteria.** API is callable on a 10-PA batch in <5s on RTX 3050. Samples match single-pitch distribution when horizon=1 (regression check against next-token sampling).
- **Dependencies.** 0.3.
- **Effort.** M.
- **Status.** not started.

#### 0.6 — Rollout sanity check on 2025

- **Goal.** Confirm rolled-out PA outcomes match 2025 empirical rates.
- **Methodology.** Sample 10K rollouts from 2025 PA starts. Compute K%, BB%, HR%, mean wOBA, inning-length proxies. Compare to 2025 empirical.
- **Artifacts.** `results/pitchgpt/rollout_sanity_2025/report.md`.
- **Success criteria.** Sampled K%, BB%, HR% within ±10% relative or ±1pp absolute of 2025 league. Mean wOBA within ±0.015 of 2025 league.
- **Dependencies.** 0.5.
- **Effort.** S.
- **Status.** not started.

### Phase 1 — Tier A dossiers

#### A1 — Counterfactual pitch-call grade

- **Goal.** For each real PA in 2025, score the actual pitch call against the rollout's recommended alternatives — "Nola threw FF middle-middle; model's top rollout was CU low-away (+0.08 xwOBA delta). Pitch grade: D."
- **Edge claim (tweetable).** "These 5 pitches cost the Phillies 0.4 wins last week." Measured in simulated xwOBA delta, aggregated to per-pitch and per-pitcher grades.
- **Methodology.** For each real pitch in a sampled 2025 cohort:
  1. Freeze the PA prefix up to (but not including) that pitch.
  2. Sample N=100 rollouts from PitchGPT for the remaining horizon.
  3. For each rollout, compute expected-wOBA using `woba_value`-conditional lookups from the outcome head + empirical wOBA-per-outcome rates.
  4. The "call grade" is the percentile rank of the actual pitch's simulated wOBA outcome within the rollout distribution (low percentile = pitcher gave up more than expected given the model's best rollouts).
- **Artifacts.**
  - `src/dashboard/views/pitch_call_grades.py` — new view.
  - `results/pitchgpt_sim/pitch_call_grades_2025/leaderboard.csv` — pitcher × week grades.
  - Methodology paper v2 addendum section.
- **Success criteria.**
  - Gate 1 (sampling-fidelity): pass Phase 0.
  - Gate 2 (OOS calibration): rollout wOBA distribution must be calibrated against actual wOBA within ECE_wOBA < 0.04 (**TBD — set after Phase 0.6 measures the baseline**).
  - Gate 3 (held-out evaluation cohort): split 2025 into first-half (grade computation) and second-half (calibration check). Grades on first-half must predict second-half pitcher-level xwOBA allowed with rank correlation ≥ 0.3 (TBD — set after first empirical run).
- **Dependencies.** Phase 0.
- **Effort.** M.
- **Status.** not started.

#### A2 — Probabilistic pitcher projections with CIs

- **Goal.** Sample 30-start seasons for every MLB pitcher under their current pitch distribution, producing ERA / FIP / K% / wOBA-allowed histograms with real 90% and 95% bands.
- **Edge claim.** "Public analytics give point projections; we give GMs confidence intervals." Differentiates from Steamer, ZiPS, ATC — those are point estimates with coarse error from their authors.
- **Methodology.** For each MLB pitcher with ≥300 pitches in the last 365 days:
  1. Condition the rollout on the pitcher's recent PA starting contexts (from their last 365 days of PAs).
  2. Sample 30 full "starts" per pitcher, where a start = N=25 PAs drawn from opponent-league-average lineup structure.
  3. Aggregate to ERA / FIP / K% / wOBA. Produce 90% and 95% bands from the N_sims distribution.
- **Artifacts.**
  - `src/analytics/pitchgpt_projection.py` — new module wrapping the rollout harness.
  - `src/dashboard/views/projections.py` — extend existing file with PitchGPT CIs column.
  - `results/pitchgpt_sim/projections_2026/projections.csv` with columns `player_id, player_name, era_p5, era_p50, era_p95, fip_p5, ..., n_sims`.
- **Success criteria.**
  - Gate 1: Phase 0 PASS.
  - Gate 2: empirical 90% CI coverage on a 2024 backtest — draw projections as-of 2024-03-01, compare to actual 2024 season ERA / FIP, compute CI coverage. Must land in [85%, 95%].
  - Gate 3: point estimate (median) rank-correlates with actual 2024 ERA at ρ ≥ 0.35 (TBD floor — set after first backtest; Steamer-at-launch is typically ρ ≈ 0.25–0.40).
  - Honest-note: small-sample pitchers (n < 300 pitches) get `NaN` projections, not spurious narrow bands. Guard with min-sample-rule checks.
- **Dependencies.** Phase 0.
- **Effort.** L (significant backtest tooling, season-long simulation).
- **Status.** not started.

#### A3 — Matchup sim with CIs

- **Goal.** Trea Turner vs Nola: sample 10,000 PAs → wOBA distribution histogram (not point estimate). A new batter × pitcher dashboard view.
- **Edge claim.** Public matchup stats are thin counts over small samples ("Turner is 3-for-12 against Nola"). Our matchup panel produces a full distribution conditioned on the 2,210-token + context-aware sequence model.
- **Methodology.** For each (batter, pitcher) pair:
  1. Sample 10K PA rollouts conditioned on `batter.stand` × `pitcher.p_throws` matchup and the pitcher's recent distribution.
  2. Per rollout, compute PA-outcome + wOBA contribution.
  3. Emit histogram with p05/p25/p50/p75/p95.
- **Artifacts.**
  - `src/dashboard/views/matchup_sim.py`.
  - `src/analytics/pitchgpt_matchup.py`.
- **Success criteria.**
  - Gate 1: Phase 0 PASS.
  - Gate 2: observed-vs-simulated wOBA on 2025 matchup pairs with ≥50 real PAs — calibration scatter plot, Pearson r ≥ 0.4 (TBD — measured after first run).
  - Gate 3: guard against small-sample artifacts — any pair with fewer than 20 real matchup PAs is marked "rollout only, no empirical cross-check" in the UI.
- **Dependencies.** Phase 0.
- **Effort.** M.
- **Status.** not started.

#### A4 — Deceptiveness leaderboard — **DEMOTED to Tier C 2026-04-24**

**Demotion note (2026-04-24, post-Phase-0.1).** Phase 0.1 showed PitchGPT loses metric 4 (2-gram Frobenius) to LSTM with CI [0.008, 0.250]. That is precisely the signal A4 depends on — if the transformer's sequence structure is not tighter than LSTM's, the "sequence-likelihood deceptiveness" framing does not earn its claim over a simpler LSTM-based or entropy-based deception index. A4 is parked until one of: (a) a different validation angle is found (e.g., regress per-pitcher NLL on swinging-strike rate and show the regression is stronger than a pitch-mix-entropy baseline's), or (b) a longer-horizon sequencing study (horizon ≥12 instead of 6) restores a transformer 2-gram/3-gram win. Until then, A4 does NOT ship as a Tier-A edge product. Ranking below is preserved for future revisit.

---

- **Goal.** Per-pitcher: average negative log-likelihood under a league-prior rollout minus NLL under the pitcher's own rollout. High surprise under league prior = deceptive.
- **Edge claim.** Ties cleanly to the disruption-orthogonality result from the prior PitchGPT validation spec (`docs/models/pitchgpt_validation_spec.md`). Publicly, "deception" in baseball is measured by tunneling heuristics and pitcher-level pitch-mix entropy. Ours is a sequence-likelihood quantity that captures both pitch-type choice and in-count sequencing.
- **Methodology.** For each pitcher in the 2025 cohort with ≥200 pitches:
  1. Compute NLL of the pitcher's actual 2025 pitches under their own PitchGPT-conditioned rollout distribution.
  2. Compute NLL of the same pitches under a league-prior distribution (marginal of PitchGPT over 2015–2022 training cohort).
  3. Deceptiveness index = league_NLL − own_NLL. High = the league prior is surprised by this pitcher, but their own distribution is not.
- **Artifacts.**
  - `src/dashboard/views/deceptiveness_leaderboard.py`.
  - `results/pitchgpt_sim/deceptiveness_2025/leaderboard.csv`.
- **Success criteria.**
  - Gate 1: Phase 0.
  - Gate 2: index is stable within a pitcher across months (first-half vs second-half rank correlation ≥ 0.5 on pitchers with ≥500 pitches).
  - Gate 3: index correlates with external deception proxies where they exist (e.g., swinging-strike rate, chase rate) at ρ ≥ 0.2 — this is a sanity floor, not a validation (a high index that has zero proxy correlation is suspicious).
- **Dependencies.** Phase 0.
- **Effort.** M.
- **Status.** not started.

### Phase 2 — Tier B dossiers

#### B1 — Live-game WP upgrade

- **Goal.** Replace base-state run-expectancy lookup (the standard WP table) with a rollout-based WP that uses the actual pitcher's distribution.
- **Methodology.** At each live game state (inning, outs, runners, batter, pitcher), roll out rest-of-inning under the pitcher's distribution N=500 times. WP = P(home_team_wins_from_here | simulated rest of game).
- **Artifacts.**
  - Extend `src/dashboard/views/live_game.py` with a "PitchGPT WP" channel alongside the existing base-state WP.
  - `src/analytics/pitchgpt_wp.py`.
- **Success criteria.**
  - Gate 1: Phase 0 + A3 landed (A3 is where the rollout-to-wOBA reduction is validated).
  - Gate 2: live WP latency < 1s per request on RTX 3050 (precompute N=500 rollouts offline, reuse on dashboard).
  - Gate 3: calibration — across 2025 late-innings (9th+), simulated WP must bin-calibrate within ECE_WP < 0.05 vs actual game outcomes.
- **Dependencies.** Phase 0, A3.
- **Effort.** M.
- **Status.** not started.

#### B2 — Reliever-change decision support

- **Goal.** Under candidate reliever A vs B, rollout rest-of-game, compare expected run distribution.
- **Methodology.** Rest-of-game rollout under pitcher A vs pitcher B with same game state as input. Emit run-distribution delta with CI.
- **Artifacts.**
  - `src/dashboard/views/reliever_decision.py`.
  - Per-game artifact when live-game view is active.
- **Success criteria.**
  - Gate 1: Phase 0, B1.
  - Gate 2: recommendation-direction must agree with leverage-index-weighted-wOBA recommendations in ≥70% of 2024 late-inning cases (sanity — if the model disagrees with LI-based reasoning 50% of the time, something is off).
- **Dependencies.** A2 scaffolding, B1.
- **Effort.** M.
- **Status.** not started.

#### B3 — Anomalous-outing detector

- **Goal.** Log-likelihood of a pitcher's actual outing under their own distribution. Low likelihood = fatigue / grip change / early injury signal.
- **Edge claim.** A **distinct-methodology** replacement for the retracted-VWR anomalous-outing use case. Frame as "sequence-likelihood anomaly detection," not "VWR successor."
- **Methodology.**
  1. For each pitcher outing in 2025: compute NLL of actual pitch sequence under the pitcher's own distribution (estimated from the prior 30 days).
  2. Threshold: NLL > mean + 2×std over prior 30 days → anomalous-outing flag.
- **Artifacts.**
  - `src/dashboard/views/anomalous_outings.py`.
  - `results/pitchgpt_sim/anomalous_outings_2025/flags.csv`.
- **Success criteria.**
  - Gate 1: Phase 0.
  - Gate 2 (honest-retracted-VWR check): flagged outings must not be explainable by season-day alone. Run the same seasonal-residual check that retracted VWR — if the flag-rate as a function of season_day is approximately flat (F-test p > 0.05 after partialing), PASS. This is exactly the gate that VWR failed at expanded-cohort scale, so we apply it pre-emptively.
  - Gate 3 (external-validation, weak): spot-check flagged outings against public IL-stint announcements within ±30 days. This is **not a required gate** (injury causation is noisy and low n for any given pitcher) but is run as a honest-caveat appendix.
- **Dependencies.** Phase 0.
- **Effort.** M.
- **Status.** not started.

#### B4 — Optimal pitch-mix recommendation

- **Goal.** Grid-search the mix that minimizes simulated wOBA vs upcoming opponent lineup. Per-start game-plan artifact.
- **Methodology.** Over a grid of pitch-mix perturbations (±5pp in each of top-3 pitch types), sample 1K rollouts per lineup. Recommend the mix minimizing median wOBA allowed, with CI on the delta vs current mix.
- **Artifacts.**
  - `src/dashboard/views/pitch_mix_gameplan.py`.
  - Per-game plan.
- **Success criteria.**
  - Gate 1: Phase 0, A3.
  - Gate 2: recommended mix must stay within physical-feasibility bounds — no recommendation to throw 50% changeups if the pitcher's career max usage is 18%. Enforce with a min-distance-to-historical-mix constraint.
  - Gate 3: 2024 backtest — when we recommend mix changes, do pitchers who (for other reasons) shifted toward the recommended mix actually see wOBA allowed improve? Rank correlation ≥ 0.15 (TBD — this is a very weak predictive floor; the real edge is decision support, not prediction).
- **Dependencies.** A3, season-by-season roster history backfill.
- **Effort.** L.
- **Status.** not started.

### Phase 3 — Tier C dossiers

#### C1 — Umpire-conditioned strategy

- **Goal.** "If ump X is calling the strike zone tight, what should pitcher Y throw?"
- **HARD GATE.** Per `feedback_no_umpire_edges_until_abs_drift_check.md`: do NOT ship without ABS-era drift check. ABS (MLB spring 2025, regular-season 2026 challenge) distorts the tail where an umpire-edge would live. Historical `umpire_tendencies` is a stale prior.
- **Methodology (gate first).**
  1. Run ABS-era drift analysis on `umpire_tendencies.accuracy_above_x` — compare 2023 (pre-ABS) vs 2025 (partial-ABS) distributions via KS test + tail-rate comparison. Block the product until KS p > 0.05 or cohort-restricted.
  2. Only then: condition rollouts on a range of umpire-tendency inputs and emit pitch-recommendation deltas.
- **Artifacts (gated).** TBD. If the gate opens, produce `src/dashboard/views/umpire_strategy.py`.
- **Success criteria.**
  - Gate 0 (NEW — the ABS-drift gate): documented at `docs/edges/abs_drift_check_2026.md`. This is a prerequisite paper / doc, not a product.
  - Gate 1+: per Tier A items.
- **Dependencies.** ABS-drift analysis.
- **Effort.** M for the drift check; M more for the product if the gate opens.
- **Status.** not started, gated.

#### C2 — Pre-AB pitch prediction for advance scouting

- **Goal.** "What's the likely first pitch from this pitcher given this count, this batter?"
- **Edge.** Low novelty publicly (Savant has next-pitch probabilities for individual pitchers). Still Phillies-useful for internal game-prep.
- **Methodology.** Rollout one pitch at count 0-0 under the pitcher's distribution + batter-hand conditioning. Emit top-3 predicted pitch types with probabilities.
- **Artifacts.** Extend existing `src/dashboard/views/matchups.py` — add a "pre-AB prediction" column rather than build a new view.
- **Success criteria.** Low-bar sanity; Phase 0 is sufficient.
- **Effort.** S.
- **Status.** not started.

#### C3 — Batter-weakness leaderboard

- **Goal.** Simulated xwOBA per batter × pitch type vs league-average pitcher. A public batter-vulnerability map.
- **Methodology.** For each batter with ≥300 PAs in 2024–2025: sample 2K PAs against a league-average pitcher distribution but fix the pitcher's next-pitch to each of the top-6 pitch types. Emit xwOBA heatmap.
- **Artifacts.** `src/dashboard/views/batter_weakness.py`.
- **Success criteria.**
  - Gate 1: Phase 0.
  - Gate 2: batter weaknesses must stabilize within a season (first-half vs second-half rank correlation ≥ 0.5).
- **Effort.** M.
- **Status.** not started.

#### C4 — Stability / signal-to-noise floor

- **Goal.** How much does a pitcher's sim-output change under small input perturbations? Methodology contribution for the paper.
- **Methodology.** Perturb input context vectors by ±1 bucket in each dimension independently; measure distributional shift (1-Wasserstein) in rollout outcomes. Report mean + 95% CI per-input-dimension.
- **Artifacts.** `results/pitchgpt_sim/stability_2025/report.md`.
- **Success criteria.** All perturbation-induced shifts must be < the across-pitcher natural variance (sanity — if perturbing 1 context bucket moves the distribution more than switching pitchers does, the sim is not signal-dominated).
- **Effort.** M.
- **Status.** not started.

---

## 7. Risks and honest-caveats

### 7.1 Compounding-error accumulation under long rollouts

PitchGPT is trained on next-token cross-entropy, not multi-step sequence likelihood. Exposure bias (teacher-forcing at train, sampled-prefix at inference) is a known failure mode. At rollout horizon H=6 we expect mild drift; beyond H≈10 we expect noticeable divergence of sampled-prefix distributions from teacher-forced distributions. **Phase 0.6's sanity check on K%/BB%/HR% rates is the first line of defense.** If the rates drift significantly between H=3 and H=8 rollouts, horizon must be capped and the docs must reflect that cap. **This measurement is TBD — done during Phase 0.6.**

### 7.2 Calibration drift across eras

ECE is measured on 2025 pitcher-disjoint holdouts. Sim engine extrapolates to 2026. The methodology paper §3.3 shows ECE drifted +0.02 pre-temperature from 2024 to 2025 (0.0203 → 0.0201), and +0.00 post-temperature — i.e., calibration was stable, but the measurement only covers two years. Any application beyond single-season OOS (e.g., A2's 30-start-season projections for 2026) will require re-measuring ECE on early-2026 data and updating the temperature scalar.

### 7.3 ABS-era umpire drift (C1 hard gate)

Per `feedback_no_umpire_edges_until_abs_drift_check.md`. The current PitchGPT v2 checkpoint uses `umpire_tendencies.accuracy_above_x` as 1 context dim (see post-ingestion next-phase plan). The sim engine's umpire-axis rollouts inherit this prior and are therefore potentially stale for 2026 predictions. The fix: for any umpire-touching sim (C1 explicitly, and any Tier-B item that conditions on ump), either (a) run the ABS-drift gate and proceed only under a pass, or (b) sample rollouts with the ump dim **zeroed** (ignoring the umpire signal) and disclose in the artifact.

### 7.4 Per-pitcher matchup small-sample artifacts

A3 produces batter × pitcher matchup distributions. Public observers will want to cross-check rollout wOBA against actual historical matchup wOBA. At typical n=3 to n=12 matchup PAs, sampling noise is enormous. **Guard with minimum-sample-rules.** Any matchup pair with fewer than 20 real PAs is labeled "rollout only, no empirical cross-check." Apply at the UI level, not as a filter that hides pairs.

### 7.5 Phase 0 sampling-fidelity result may be null

The script `scripts/pitchgpt_sampling_fidelity.py` may land PitchGPT ≈ LSTM on the 5 metrics. **If so:** the sim-engine claim narrows to "calibrated, not differentiated" — meaning we can still ship CIs and counterfactual resampling as a PitchGPT product (the calibration, per the methodology paper, is the load-bearing claim), but the **marketing** of the sim-engine becomes "calibrated rollout engine," not "rollouts match 2025 joint distributions better than LSTM." This is an honest-negative outcome and is publishable as a v2-paper addendum.

**Under the null outcome:**
- A1, A2, A3, B1–B4 still ship, but each item's public-facing language drops claims that rely on PitchGPT-better-than-LSTM sampling.
- The methodology paper v2 gains a "sampling fidelity null" section. Per Path 2 philosophy, this is honest-negative content, which is **fine** — not a failure mode.
- Retire A4 if it also turns up null (A4 is the item most tightly bound to the sampling-edge claim).

### 7.6 Outcome-head drift

The 7-class outcome target has known confounding with pitch type itself (e.g., a fastball's outcome distribution differs from a curveball's). The head must be a joint-distribution over `(pitch_token, outcome_class)`, not marginal over outcome. Verify in Phase 0.2 design doc.

### 7.7 LSTM-parity risk on downstream items

The methodology paper §3.4 already reports a downstream utility tie with LSTM on the 5-class pitch-outcome-bucket target. If the 7-class PA-outcome head similarly ties with an LSTM equivalent, Tier-A items lose their "transformer advantage" framing. **Mitigation:** always train a matched LSTM outcome-head as a baseline for every Tier-A item's success criterion; report the delta with CI. If the delta is a tie, reframe the item's claim around "calibration + CIs," which the LSTM does not automatically provide.

---

## 8. Integration points

### 8.1 Methodology paper

Current paper at `docs/awards/methodology_paper_pitchgpt.md`. Phase-1 landings produce a **v2 paper** at `docs/awards/methodology_paper_pitchgpt_v2.md` (path reserved; do not create until Phase 1 lands at least 2 items). V2 adds:
- §3.6 Sampling fidelity (from Phase 0.1 results).
- §3.7 PA-outcome head (from Phase 0.3–0.4).
- §7 Sim-engine applications (dossier-by-dossier summary of Phase 1 items with their validation-gate tables).
- V2 does **not** remove v1 sections; it adds, so reviewers can trace the narrowing and then the extension.

### 8.2 Dashboard

All Tier-A and Tier-B items that land a UI go in `src/dashboard/views/`. Per `CLAUDE.md`, never create `src/dashboard/pages/`. Expected new views:
- `src/dashboard/views/pitch_call_grades.py` (A1)
- `src/dashboard/views/matchup_sim.py` (A3)
- `src/dashboard/views/deceptiveness_leaderboard.py` (A4)
- `src/dashboard/views/reliever_decision.py` (B2)
- `src/dashboard/views/anomalous_outings.py` (B3)
- `src/dashboard/views/pitch_mix_gameplan.py` (B4)
- `src/dashboard/views/batter_weakness.py` (C3)

A2 extends the existing `src/dashboard/views/projections.py`; B1 extends `src/dashboard/views/live_game.py`; C2 extends `src/dashboard/views/matchups.py`.

### 8.3 Contrarian leaderboards

CausalWAR's contrarian leaderboards are downstream consumers of any pitcher-projection shift. Once A2 (probabilistic projections) lands, the Buy-Low / Over-Valued rankings should be re-run conditioning on PitchGPT projection CIs alongside the CausalWAR residual. Flag as a cross-flagship integration — **tracked as a Phase 2 follow-up, not a blocker.**

### 8.4 Live-game case study

The 2026-04-19 PHI-vs-ATL case study used PitchGPT for next-pitch calibration only. A new live case study after Phase 2 lands B1 (WP) + B2 (reliever) would showcase the sim engine in-situ. Queue this as "future case study" — do not schedule until B1 and B2 land.

---

## 9. Validation gates

Every Tier-A item must pass, before being allowed to publish a public-facing edge claim, the following three-gate sequence. Per-item Section 6 dossiers already list these; Section 9 codifies the contract.

**Gate (a) — Sampling-fidelity validation.** Phase 0.1 PASS condition: PitchGPT wins or ties vs LSTM on ≥3 of 5 metrics (marginal pitch-type KL, marginal zone KL, velocity Wasserstein, 2-gram Frobenius, PA-outcome-rate L1), with 95% bootstrap CI excluding zero on at least 2 of the 3 wins. **Metric 4 (2-gram Frobenius) is mandatory among the wins — see §3.1 decision log.** If Phase 0.1 fails this gate, all Tier-A items lose their "PitchGPT better than LSTM" framing — see §7.5.

**Gate (b) — OOS calibration re-check on 2025.** The item-specific calibration criterion (see each A-dossier, "Gate 2") must be measured on the 2025 pitcher-disjoint holdout and must PASS. Pre-temperature ECE is acceptable; post-temperature preferred. Calibration measurements use 10 bins unless the item explicitly justifies otherwise.

**Gate (c) — Held-out evaluation cohort.** Each A item specifies a hold-out split (e.g., first-half vs second-half 2025 for A1; 2024 backtest for A2; empirical-matchup cross-check for A3; first-half vs second-half for A4). The cross-check must PASS at its dossier-specified threshold with a 95% CI excluding the null direction.

**For Tier B:** gates (a)–(c) are required when the item publishes a claim; not required for internal decision-support tooling (B1 live-game WP can ship without gate (c) if the item is labeled "operational, not published").

**For Tier C:** gate (a) + gate (b) are required. C1 has an **additional** ABS-era drift gate (§7.3).

**Reporting.** Every item's final report lives at `results/pitchgpt_sim/<item>/report.md` and includes a Gates table with PASS/FAIL per gate plus CI. Follows the existing pattern from `results/pitchgpt/2025_holdout/report.md`.

---

## 10. What we do NOT do

- **Do not build a full end-to-end MLB game simulator from scratch.** Low marginal value over focused PA-level tools (A1, A2, A3). High compute. Less defensible as an award claim than counterfactual pitch-call grading. Explicitly deprioritized.
- **Do not retrain PitchGPT to close the 1.2pp LSTM perplexity gap.** The narrowed claim already accommodates the shortfall. Per `docs/NORTH_STAR.md` "post-evidence consolidation," this is locked in. Do NOT relitigate.
- **Do not re-promote VWR, MechanixAE as injury EWS, ChemNet, or volatility_surface.** Retracted / demoted / retired stays. B3 (anomalous outings) is **not** a VWR successor — it uses a fundamentally different methodology (sequence likelihood, not biomechanical residual).
- **Do not ship umpire-conditioned edge products (C1) without an ABS-era drift check.** Per `feedback_no_umpire_edges_until_abs_drift_check.md`.
- **Do not fabricate thresholds.** Any criterion labeled TBD in this document stays TBD until measured. Do not substitute a "looks reasonable" number pulled from the air.
- **Do not build new models.** Per NORTH_STAR Path 2 + post-ingestion next-phase plan §2. The sim engine is a wrapping of the existing PitchGPT checkpoint plus an outcome head. It is NOT a new model.
- **Do not skip the validation agent** after any 3+ parallel-agent batch. Per `feedback_validation_agent.md`.
- **Do not commit retries or checkpoints to main** without validation-agent PASS.

---

## 11. Where to pick up next session

**First action:** Execute Phase 0.1 — the sampling-fidelity script has been drafted as `scripts/pitchgpt_sampling_fidelity.py`. Run it end-to-end, commit results to `results/pitchgpt/sampling_fidelity_2026_04_24/`, and update this document's Phase 0 exit-criteria section with the PASS/FAIL verdict.

**Exact agent prompt (paste into a fresh session):**

> You are the Phase 0.1 execution agent for the PitchGPT sim engine at `C:\Users\hunte\projects\baseball`. Your task is to land the sampling-fidelity comparison vs a matched LSTM on 2025 pitcher-disjoint PA starts.
>
> Read first:
> 1. `docs/pitchgpt_sim_engine/EXECUTION_PLAN.md` §5 Phase 0 and §9 Validation gates.
> 2. `scripts/pitchgpt_sampling_fidelity.py` (already drafted — complete it if needed).
> 3. `docs/awards/methodology_paper_pitchgpt.md` §1–3 (for claim constraints).
>
> Execute:
> 1. Complete the script if it is not end-to-end runnable.
> 2. Train the matched LSTM at the 10K-game scale per the script's docstring contract.
> 3. Run the 5-metric comparison: marginal pitch-type KL, marginal zone KL, velocity Wasserstein, 2-gram Frobenius, PA-outcome-rate L1. Bootstrap CIs at 95%.
> 4. Write `results/pitchgpt/sampling_fidelity_2026_04_24/metrics.json` + `report.md`.
> 5. Commit ONLY the results JSON + report; do NOT commit model checkpoint files that are already committed-in-place, do NOT overwrite `models/pitchgpt_v2.pt`.
> 6. Update `docs/pitchgpt_sim_engine/EXECUTION_PLAN.md` §5 Phase 0 exit criteria with the PASS/FAIL verdict of Phase 0.1.
>
> Guardrails:
> - Do NOT touch the PitchGPT v2 checkpoint. Only train the LSTM.
> - Use read-only DuckDB connections (`read_only=True`) — user may have dashboard open.
> - Per `feedback_pm_role.md` equivalents at the agent level: ONE agent does this, linearly; do not fan out.
> - If the result is a null (PitchGPT loses on ≥3 of 5 metrics), commit the null result honestly; per §7.5 of the EXECUTION_PLAN it is scientifically valuable.
> - Return a 200-word summary: verdict, key numbers, gate PASS/FAIL, any unexpected findings.

**Second action after 0.1 lands:** Launch Phase 0.2 agent (PA-outcome head design doc). Smoke experiment — joint vs frozen-backbone — at 500 games per option. Final choice drives 0.3's full training run.

**Downstream:** After Phase 0 passes its exit criteria, launch A1 + A3 in parallel as the Phase 1 opener (A1 is the most pitchable; A3 unblocks B1). Run a validation agent after the A1/A3 batch completes per `feedback_validation_agent.md`.

---

## 12. Appendix — data/column requirements

Cross-checked against `src/db/schema.py` lines 119–211 (pitches table DDL). Every column below exists in the current schema; Phase 0 has **no backfill blockers**. B4 has a backfill prerequisite — see below.

### 12.1 pitches table — required for Phase 0 and all Tier A items

| Column | Type | Sim-engine use | Coverage |
|---|---|---|---|
| `pitch_type` | VARCHAR | token composite | populated; small-fraction nulls filtered at query time |
| `plate_x`, `plate_z` | FLOAT | zone token | populated on all tagged pitches |
| `release_speed` | FLOAT | velocity bucket | populated |
| `description` | VARCHAR | outcome-head label component (called_strike / ball / swinging_strike / foul / hit_by_pitch) | populated |
| `events` | VARCHAR | outcome-head label component on PA-terminal pitches (strikeout / walk / single / double / triple / home_run / etc.) | populated only on PA-terminal pitches — expected null elsewhere |
| `woba_value` | FLOAT | A1 outcome grading, A2 wOBA agg | populated on PA-terminal pitches only — expected sparsity |
| `launch_speed` | FLOAT | contact-quality conditioning (future extension) | populated on in-play only |
| `delta_run_exp` | FLOAT | A1 counterfactual ground truth | populated |
| `balls`, `strikes`, `outs_when_up` | INTEGER | context | populated |
| `on_1b`, `on_2b`, `on_3b` | INTEGER (bool-ish) | context | populated |
| `stand`, `p_throws` | VARCHAR | context | populated |
| `inning`, `inning_topbot` | INTEGER, VARCHAR | context | populated |
| `at_bat_number`, `pitch_number` | INTEGER | PA grouping | populated |
| `game_pk`, `game_date` | INTEGER, DATE | cohort splits | populated |
| `zone` | INTEGER | supplementary zone features (if ever needed beyond composite token) | populated |

### 12.2 Supplementary tables

| Table | Rows | Use | Backfill status |
|---|---|---|---|
| `umpire_assignments` | 103,192 | C1 umpire context — gated per §7.3 | 93.6% game_pk matched; treat unmatched as "no ump ctx" |
| `umpire_tendencies` | 1,104 | C1 umpire context | umpire × season; sufficient for C1 (gated) |
| `game_weather` | 25,050 | NOT directly needed for PitchGPT sim (weather is a DPI signal, not a PitchGPT context) | 94.2% coverage |
| `tj_surgery_dates` | 302 | B3 anomalous-outing external-proxy check (weak, caveat-level) | 224 pitchers |
| `season_pitching_stats` | ... + 2010–2014 backfilled | A2 projection-backtest ground truth (actual 2024 ERA, etc.) | populated |
| `players` | — | A3 matchup view (name lookup) | populated |
| **roster history by season** | does NOT exist as a clean table | B4 opponent-lineup sampling | **BACKFILL PREREQUISITE for B4** — flagged in post-ingestion next-phase plan §5C item 2 |

### 12.3 Model artifacts

| Artifact | Path | Role |
|---|---|---|
| PitchGPT v2 backbone checkpoint | `models/pitchgpt_v2.pt` | current flagship — do NOT overwrite |
| PitchGPT v1 (legacy) | `models/pitchgpt_v1_10k.pt` | fallback per P0.3 decision in post-ingestion plan |
| PitchGPT outcome-head checkpoint | `models/pitchgpt_v2_outcomehead.pt` | TO BE CREATED in Phase 0.3 |
| LSTM baseline | freshly trained per run (no committed checkpoint) | Phase 0.1 requires a matched-scale train |
| Rollout harness | `src/analytics/pitchgpt_sim.py` | TO BE CREATED in Phase 0.5 |

---

*Document author: Claude (session 2026-04-24). Not yet committed per user instruction — leave unstaged for user review.*
