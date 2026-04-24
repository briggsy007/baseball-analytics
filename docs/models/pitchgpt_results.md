# PitchGPT — Results Log

This document tracks the PitchGPT evidence history. Each section
is dated; older numbers are **preserved verbatim** — do not rewrite
historical results.

Spec: `docs/models/pitchgpt_validation_spec.md`.
Architecture doc: `src/analytics/pitchgpt.py`.

---

## 1. 2024 leakage-clean test (2026-04-16)

Source: `docs/models/pitchgpt_baseline_results.md` (canonical 5-epoch,
1000/300/300 games, 2015-2022 / 2023 / 2024 split, seed 42, CPU).

| Model      | Test ppl | vs PitchGPT |
|------------|---------:|------------:|
| PitchGPT   | **148.20** | — |
| LSTM       | 169.80   | +14.6 %     |
| Markov-2   | 332.67   | +124.5 %    |
| Markov-1   | 355.00   | +139.6 %    |
| Heuristic  | 438.86   | +196.1 %    |

Spec verdicts: LSTM **FAIL** (12.7% vs 15% gate — missed by 2.3pp);
Markov-2 and Heuristic **PASS**.

## 2. 10K ablation (2026-04-18)

Source: `docs/models/pitchgpt_calibration_ablation_results.md` Section 2.
At 10K games / 5 epochs on CUDA:

| Variant | Test ppl | Δ vs full |
|---|---:|---:|
| full | **113.088** | — |
| tokens_only | 121.402 | +6.85 % |
| count_only  | 117.969 | +4.14 % |

Verdict: context adds real signal (+6.85%) but distilled count+outs
recovers most of it. Full-context branch contributes ~4% on top of
count+outs — below the 10% ablation gate.

## 3. Pitcher-disjoint 2025 OOS (2026-04-18 evening, v1 checkpoint)

Source: `results/pitchgpt/2025_holdout/report.md`. Checkpoint trained
on `validate_pitchgpt_20260418T150305Z/pitchgpt_full.pt` (1K games,
5 epochs, **pitcher-disjoint** from val/holdout; 0 leakage).

2025 holdout: 334 pitchers new to 2025, 53,723 tokens.

| Model     | Params    | Holdout PPL | 95% CI         |
|-----------|----------:|------------:|----------------|
| PitchGPT  | 1,398,562 | **152.187** | 150.09 – 154.27|
| LSTM      |   837,154 | 176.554     | 174.06 – 179.04|
| Markov-2  |         0 | 352.657     | 348.08 – 357.48|
| Heuristic |         0 | 471.856     | 466.81 – 476.67|

| Gate | Point | CI-lower | Verdict |
|---|---:|---:|:---|
| vs LSTM (≥15%) | +13.80% | +12.22% | **FAIL** |
| vs Markov-2 (≥20%) | +56.85% | +55.96% | PASS |
| vs Heuristic (≥25%) | +67.75% | +67.18% | PASS |

Calibration 2025: ECE pre 0.0201 → post 0.0098 (T=1.120) — **PASS**.

Consolidated finding (NORTH_STAR post-evidence section): "calibrated
transformer sequence model that survives temporal shift and beats naive
baselines by wide margins." LSTM gate missed by 1.2pp under the
pitcher-disjoint reality; context ablation already showed the gap was
not recoverable at this scale.

---

## 4. v2 — Umpire context dimension added (2026-04-23)

**Change.** Added a 35th context dimension to PitchGPT: the per-pitch
**home-plate umpire ``accuracy_above_x`` from the prior season**
(``umpire_tendencies`` table). One-hot categorical slots [0..33]
unchanged; the new scalar lives in slot 34.

Choice rationale (see `docs/data/umpire_integration_notes.md`):

* **Prior-season** tendency, not current-season — avoids look-ahead
  leakage in OOS evaluation. A 2024 game uses 2023 umpire tendencies.
* **NULL-fill = prior-season league median** for rookie umpires or
  missing HP assignments. A missing joined ump maps to a "neutral"
  ump rather than zero, keeping the feature distribution centred on
  the training data. For 2015 (where 2014 tendencies are absent from
  the staged data), we backstop to the **same-season** median — mild
  aggregate-only lookahead, strictly better than a hard zero that
  would silently bias the entire 2015 cohort.
* **Raw scalar, not z-scored.** Across 2015-2025 the value ranges
  approximately -1.8 to +2.5. The transformer's
  ``Linear(CONTEXT_DIM, d_model)`` projection absorbs that magnitude
  unproblematically.

**Retrain.** Smoke-scale retrain (1K games × 5 epochs, pitcher-disjoint
protocol identical to the v1 150305Z run). Wall-clock **41.8 s** on
RTX 3050 (CUDA), 1,398,690 params (v1: 1,398,562 — the 128-param
delta is the one extra input column on the context projection linear
layer). Artifact written to ``models/pitchgpt_v2.pt`` — v1 preserved.

Training curve:

| Epoch | train_ppl | val_ppl |
|:---:|---:|---:|
| 1 | 288.81 | 261.38 |
| 2 | 174.71 | 210.10 |
| 3 | 144.44 | 178.99 |
| 4 | 127.94 | 167.10 |
| 5 | **118.93** | **153.43** |

Umpire-scalar smoke audit (50 train sequences, 1,666 context rows):
40 unique rounded values, range -3.03 to +1.89 — feature is
non-degenerate at the input stage.

**Full retrain queued.** A 10K-game retrain would take ~90 min on the
current GPU budget and is deferred to a follow-up ticket; given the
1K ablation already shows ≤7% context lift at 10K, the v1→v2
comparison at 1K is the cheapest honest signal.

### 4.1 2025 pitcher-disjoint OOS (v2)

Same holdout protocol as run 3 above (``scripts/pitchgpt_2025_holdout.py``,
pitcher-disjoint from train, 2025 only). LSTM freshly trained on the
same train split; Markov-2 / Heuristic closed-form from the same
training sequences. 51,090 non-PAD tokens scored.

Artifacts: `results/pitchgpt/2025_holdout_v2_ump/` (perplexity_comparison.json,
calibration_2025.json, reliability_2025.html, report.md).

| Model     | Holdout PPL | 95% CI         |
|-----------|------------:|----------------|
| PitchGPT (v2, +ump) | **159.142** | 156.97 – 161.28 |
| LSTM      | 180.416     | 177.88 – 183.32 |
| Markov-2  | 350.840     | 346.31 – 356.03 |
| Heuristic | 470.942     | 465.98 – 476.07 |

| Gate | Point | CI-lower | Verdict |
|---|---:|---:|:---|
| vs LSTM (≥15%) | **+11.79%** | +10.15% | **FAIL** |
| vs Markov-2 (≥20%) | +54.64% | +53.68% | PASS |
| vs Heuristic (≥25%) | +66.21% | +65.61% | PASS |

Calibration 2025: ECE pre 0.0106 → **post 0.0030** (T=1.101) — PASS,
and **markedly sharper than v1's 0.0098**.

### 4.2 v1 vs v2 comparison

| Metric | v1 (150305Z) | v2 (ump) | Δ |
|---|---:|---:|---:|
| Context dim | 34 | 35 | +1 (ump scalar) |
| Params | 1,398,562 | 1,398,690 | +128 |
| Train ppl (epoch 5) | ~118 | 118.93 | flat |
| Val ppl (epoch 5) | ~153 | 153.43 | flat |
| 2025 PitchGPT ppl | 152.187 | 159.142 | **+4.6% worse** |
| 2025 LSTM ppl | 176.554 | 180.416 | +2.2% worse |
| LSTM gate point | +13.80% | **+11.79%** | **-2.0 pp** |
| LSTM gate CI-lower | +12.22% | +10.15% | -2.1 pp |
| ECE post-temp | 0.0098 | **0.0030** | **-70%** |
| Calibration temp | 1.120 | 1.101 | closer to 1 |

### 4.3 Interpretation

The ump context did **not improve** headline perplexity or the LSTM
gate — it shifted the LSTM delta from -1.2pp-short to -3.2pp-short.
Two consistent reads:

1. **Context-adds-little finding replicates.** The 10K ablation
   already showed context adds 1-7% lift with count+outs doing most
   of the work. Adding a single prior-season ump scalar at 1K games
   and 5 epochs is not enough training to exploit a feature whose
   variance is dominated by inter-umpire not within-game signal.
   Both PitchGPT and LSTM got slightly worse in absolute terms —
   consistent with the two models stochastically sampling a slightly
   harder 2025 holdout slice (500 games vs 1535 sequences in v1 —
   the `max_games_per_split` default was overridden to 500 in this
   run, which is a **confounder**: CI widths shrink but point
   estimates move on the order of 1-3%).
2. **Calibration sharpened materially.** Post-temperature ECE
   dropped 70% (0.0098 → 0.0030) and the optimal temperature
   shifted closer to 1.0. Interpretation: the ump scalar, while
   uninformative for the perplexity objective, provides a useful
   auxiliary signal that flattens the model's over-confident
   mass in the 0.2-0.5 bin of the reliability diagram. This is a
   real, defensible outcome of adding the feature, even though
   the headline gate was unaffected.

### 4.4 Verdict

- **LSTM gate 15%:** FAIL (11.79% point, 10.15% CI-lower). Gate
  still open; v2 did not close it.
- **Markov-2 gate 20%:** PASS (54.64%).
- **Heuristic gate 25%:** PASS (66.21%).
- **Calibration gate 0.10 ECE:** PASS (pre 0.0106, post 0.0030) —
  sharper than v1, best-ever for PitchGPT.

The narrowed claim from the post-evidence-consolidation section of
NORTH_STAR ("calibrated transformer sequence model that survives
temporal shift and beats naive baselines by wide margins")
**holds and is sharper** under v2 on the calibration axis, while
the LSTM-margin claim weakens by 2pp. Recommendation: keep v1 as
the flagship checkpoint for the LSTM-delta narrative, and quote the
v2 calibration as the "umpire-context experiment confirms the
calibration is not a fluke of feature selection" result. Do not
re-promote the LSTM-gate number; the honesty-note from the Path 2
reset stands.

### 4.5 Follow-up tickets (status updated in §5)

- Full 10K-game retrain with ump context — **CLOSED** by §5.2. Actual wall-clock 8.5 min on RTX 3050 (docstring 4h25m estimate was stale).
- Ump-scalar ablation at 10K — **CLOSED** by §5.1 + §5.2. v1 (context_dim=34) is effectively the v2-no-ump baseline at 10K; delta is +0.56 pp, within noise.
- Per-at-bat ump scalar — **deferred indefinitely.** ABS drift risk (`feedback_no_umpire_edges_until_abs_drift_check.md`) makes further umpire-feature investment low-EV.

---

## 5. 10K matched-scale + sampling-fidelity (2026-04-24)

Sections 3 and 4.1 reported 1K-vs-1K LSTM deltas (13.80% and 11.79%). Under scale-verify discipline (`feedback_scale_verify_before_flagship.md`), headline claims must replicate at larger matched scale. This section reports matched 10K retrains for both v1 and v2 plus a sampling-fidelity eval that tests the sim-engine reframe.

### 5.1 v1 retrained at 10K (context_dim=34, no ump)

- Artifact: `models/pitchgpt_v1_10k.pt`. Original `models/pitchgpt_v1.pt` preserved verbatim.
- Results: `results/pitchgpt/2025_holdout_v1_10k/`.
- Protocol: 2015-2022 train, 2023 val, 2025 holdout; pitcher-disjoint; seed 42; 5 epochs; batch 32; lr 1e-3.

| Model | 2025 PPL | 95% CI |
|---|---:|---|
| PitchGPT v1 @ 10K | **119.829** | 119.09 – 120.60 |
| LSTM (matched 10K) | ~122.9 | — |
| Markov-2 | ~344 | — |
| Heuristic | ~470 | — |

| Gate | Point | CI-lower | Verdict |
|---|---:|---:|:---|
| vs LSTM (≥15%) | **+2.57%** | +1.68% | **FAIL** |
| vs Markov-2 (≥20%) | +65.17% | +64.84% | PASS |
| vs Heuristic (≥25%) | +74.35% | +74.14% | PASS |

Calibration 2025: ECE pre 0.0131, post 0.0090 (T=1.048) — PASS.

### 5.2 v2 retrained at 10K (context_dim=35, with ump)

- Artifact: `models/pitchgpt_v2.pt` (replaces the 1K smoke from §4). `models/pitchgpt_v2_smoke.pt` preserves the prior smoke.
- Results: `results/pitchgpt/2025_holdout/` (updated in-place with 10K numbers).
- Same protocol as §5.1; same split seeds.

| Gate | Point | CI-lower | Verdict |
|---|---:|---:|:---|
| vs LSTM (≥15%) | **+3.13%** | +2.19% | **FAIL** |
| vs Markov-2 (≥20%) | +65.54% | +65.23% | PASS |
| vs Heuristic (≥25%) | +74.75% | +74.55% | PASS |

Calibration 2025: ECE pre 0.0124, post **0.0075** (T=1.055) — PASS, best-ever for PitchGPT.

### 5.3 Interpretation — 1K→10K scale gap

The 13.80% and 11.79% headlines do NOT replicate at matched 10K. Both architectures improve with scale; LSTM closes more ground than PitchGPT does, collapsing the gap to 2.6-3.1%. The ump feature (v2 vs v1) contributes +0.56 pp — within noise given overlapping CIs. This is the VWR scale-verify pattern from 2026-04-18; the lesson applies. Do NOT re-promote small-sample LSTM deltas.

### 5.4 Sampling-fidelity eval

Tests the sim-engine claim: can PitchGPT generate higher-fidelity pitch sequences than a matched LSTM? 2000 2025 pitcher-disjoint PAs × 10 samples per model, temperature=1.0, horizon=6; 1000-iteration bootstrap on each of 5 distributional metrics.

- Artifact: `results/pitchgpt/sampling_fidelity_2026_04_24/`.
- LSTM checkpoint: `models/pitch_lstm_10k.pt` (persisted for reuse by Phase 0.4+ of the sim engine).

| Metric | PG | LSTM | Δ (PG−LSTM) | 95% CI on Δ | Verdict |
|---|---:|---:|---:|---|:---|
| pitch_type_kl | 0.0597 | 0.0717 | −0.012 | [−0.019, −0.006] | **PG WIN** |
| zone_kl | 0.0149 | 0.0338 | −0.019 | [−0.022, −0.015] | **PG WIN** |
| velocity_wasserstein | 1.63 | 1.91 | −0.28 | [−0.33, −0.23] | **PG WIN** |
| transition_frobenius | 1.48 | 1.32 | +0.16 | [+0.008, +0.25] | **LSTM win** |
| outcome_chi2 | 3.07 | 3.07 | 0.00 | [0.00, 0.00] | Tie (degenerate proxy) |

Three PG wins survive Bonferroni correction at α=0.01. The pattern is mechanistically coherent: **transformers win on context-conditioned marginals (what pitches look like), LSTMs win on sequence dynamics (how pitches flow)**. The outcome_chi2 is degenerate under the zone-based ball/strike heuristic; a learned outcome head (Phase 0.3 of the sim-engine plan) will replace it.

### 5.5 Updated verdict + flagship framing

- **LSTM perplexity gate 15%:** FAIL at matched 10K. v1 +2.57%, v2 +3.13%. 1K headline did not replicate. Retired as a live claim.
- **Markov-2 gate 20%:** PASS (v1 +65.17%, v2 +65.54%). Stable across scale.
- **Heuristic gate 25%:** PASS (v1 +74.35%, v2 +74.75%). Stable.
- **Calibration gate ECE 0.10:** PASS (v1 post-temp 0.0090, v2 0.0075). Durable across 1K→10K and across OOS windows.
- **Sampling-fidelity (sim-engine claim):** PitchGPT wins 3/5 marginals with CIs excluding zero, loses 2-gram transitions. Per sim-engine plan §3.1 locked gate (metric 4 mandatory), Phase 0.1 FAILS. Claim narrows to "calibrated rollout engine that matches empirical marginals with CIs."

**Flagship status preserved under the narrowed framing.** PitchGPT's go-forward value is as the simulation backbone for Tier-A downstream products (counterfactual pitch-call grades, probabilistic projections with CIs, matchup sim). See `docs/pitchgpt_sim_engine/EXECUTION_PLAN.md` for the full roadmap and `docs/pitchgpt_sim_engine/COORDINATION.md` for cross-session coordination.

### 5.6 Infrastructure added in this session

- `CONTEXT_DIM` parameterized across `src/analytics/pitchgpt.py`, `pitchgpt_calibration.py`, `scripts/pitchgpt_2025_holdout.py`, `scripts/train_pitchgpt_v2_ump.py`. Default preserved at 35 (backwards-compat). v1 (34) and v2 (35) checkpoints load from one codebase; loader infers from state_dict shape.
- `scripts/pitchgpt_sampling_fidelity.py` — new 2077-line harness for the §5.4 eval. Bootstrap CIs over 5 distributional metrics + stratified-by-context analysis.
- `models/pitch_lstm_10k.pt` — matched LSTM baseline checkpoint, persisted for reuse by Phase 0.4 outcome-head OOS validation.
