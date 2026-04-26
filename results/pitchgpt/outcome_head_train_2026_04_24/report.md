# PitchGPT PA-Outcome Head — Phase 0.3 Training Report

**Date:** 2026-04-24
**Spec:** `docs/pitchgpt_sim_engine/EXECUTION_PLAN.md` §6.0.3
**Design:** `docs/pitchgpt_sim_engine/pa_outcome_head_design.md` §9

## TL;DR

**Gate status:** **FAIL**

- 7-class log-loss lift vs frequency prior (post-temp): **-5.34%** (threshold 15.00%) — **FAIL**
- 10-bin ECE post-temperature: **0.0213** (threshold <0.05) — **PASS**
- Backbone next-token ECE degradation: **0.0000** (budget +0.005) — **PASS** (frozen by construction; SHA256 byte-identity verified)

## Training

- Cohort: 2015–2022 pitcher-disjoint vs 2023 val
- Train games: 10000  Val games: 2000
- Train sequences: 85,797  Val sequences: 1,890
- Train valid outcomes: 2,789,122  Val valid outcomes: 75,384
- Val unique pitchers: 168  (pitcher-disjoint from 2015-2022)
- Hyperparameters: AdamW lr=0.001, batch=32, epochs≤10 (patience=2), seed=42, grad-clip=1.0, class weights inv-freq cap=10.0
- Architecture: **FROZEN v2 backbone** + `Linear(128→64) → GELU → Dropout(0.1) → Linear(64→7)`
- Wall-clock: **159.0s**  Best epoch: **3** (of 5 run)

### Epoch history

| epoch | train loss (wtd) | val log-loss | seconds |
|-------|------------------|-------------:|--------:|
| 1 | 1.7892 | 1.7536 | 28.7 |
| 2 | 1.7755 | 1.7509 | 28.6 |
| 3 | 1.7742 | 1.7424 | 28.9 |  **←best**
| 4 | 1.7735 | 1.7497 | 29.5 |
| 5 | 1.7730 | 1.7509 | 40.8 |

## Val metrics (2023 pitcher-disjoint)

- Valid outcome labels evaluated: **75,384**
- Temperature scalar: **0.7488**

| metric                    | pre-temp | post-temp |
|---------------------------|---------:|----------:|
| 7-class log-loss          | 1.7424   | 1.7367    |
| 10-bin ECE                | 0.0054   | 0.0213    |
| Top-1 accuracy            | 0.2228   | (same — argmax-invariant) |
| Frequency-prior log-loss  | 1.6487   | — |
| Lift vs frequency prior   | -5.68%   | -5.34% |

### Per-class breakdown

| class | n val | pre-T ll | post-T ll | train freq | weight |
|-------|------:|---------:|----------:|-----------:|-------:|
| ball | 27,531 | 1.6952 | 1.6737 | 0.3649 | 0.37 |
| called_strike | 12,054 | 1.6859 | 1.6647 | 0.1635 | 0.82 |
| swinging_strike | 8,088 | 1.6542 | 1.6179 | 0.1083 | 1.24 |
| foul | 14,372 | 1.6889 | 1.6640 | 0.1932 | 0.70 |
| in_play_out | 8,740 | 1.7447 | 1.7386 | 0.1131 | 1.19 |
| in_play_hit | 4,382 | 2.3714 | 2.5759 | 0.0543 | 1.34 |
| hbp | 217 | 4.9042 | 5.9580 | 0.0027 | 1.34 |

### Backbone token (sanity — frozen, should be identical to pre-head v2)

- Backbone next-token ECE on val: **0.0111**
- Backbone next-token accuracy on val: **0.0447**
- Backbone param-level SHA256 pre:  `c9b79869f0dc6da75821f0de3a0d3b32e920ef128c0fc19c7f127a375529fb12`
- Backbone param-level SHA256 post: `c9b79869f0dc6da75821f0de3a0d3b32e920ef128c0fc19c7f127a375529fb12`
- `models/pitchgpt_v2.pt` SHA256 pre:  `6f952054d14ac6f918f3eb9502b496b70bc0c87dfc65dc50d98ee7244a62883c`
- `models/pitchgpt_v2.pt` SHA256 post: `6f952054d14ac6f918f3eb9502b496b70bc0c87dfc65dc50d98ee7244a62883c`
- **Byte-identity verified:** YES

## Baselines

- Uniform (1/7) log-loss: **1.9459**
- Frequency-prior log-loss on val: **1.6487**
- Model post-temp log-loss on val: **1.7367**  (**-5.34%** lift)

## Checkpoint

- Path: `C:\Users\hunte\projects\baseball\models\pitchgpt_v2_outcomehead.pt`
- Size: 39,869 bytes
- Includes: head state_dict, architecture config, frozen-backbone SHA256, class-weight vector, val_2023 metrics, temperature, 2025-holdout placeholder.
- **Flagship `models/pitchgpt_v2.pt` untouched.**

## Gate table

| gate | value | threshold | PASS/FAIL |
|------|-------|-----------|----------|
| 7-class log-loss lift vs freq prior | -5.34% | >= 15.00% | **FAIL** |
| 10-bin ECE (post-temp) | 0.0213 | < 0.050 | **PASS** |
| Backbone ECE degradation | +0.0000 | <= +0.005 | **PASS** (frozen backbone — 0.0 by construction; param SHA verified) |

## Phase 0.4 handoff

- Checkpoint ready for 2025 pitcher-disjoint OOS validation (`holdout_2025` field in the checkpoint is a placeholder).
- Temperature scalar fitted on 2023 val; re-fit on 2025 val slice in Phase 0.4 to catch era drift.
- Per-pitcher log-loss stability measurement and per-class confusion diagram are Phase 0.4 deliverables (EXECUTION_PLAN §6.0.4).
