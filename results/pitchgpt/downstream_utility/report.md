# PitchGPT Downstream Utility Report

Generated: 2026-04-19T03:41:41Z

Checkpoint: `C:\Users\hunte\projects\baseball\results\validate_pitchgpt_20260418T150305Z\pitchgpt_full.pt`

Device: cuda  Seed: 42


## Question
Does PitchGPT's calibrated per-pitch next-token distribution drive better downstream *pitch-outcome* prediction than the LSTM baseline's distribution — even though PitchGPT's raw perplexity edge is only 13.80% (below the 15% spec gate)?

## Methodology
- Same pitcher-disjoint splits as `scripts/pitchgpt_2025_holdout.py`: train 1000 games (2015-2022), val 300 games (2023, for LSTM early-stop), holdout 500 games (2025, pitchers never seen in train).
- Both models score the same sequences. For every non-PAD target pitch we softmax over the 2,210-token vocab and summarise on-device into a 51-dim vector: 17 pitch-type marginals + 26 zone marginals + 5 velocity-bucket marginals + top-1 prob + top-1 log-prob + entropy.
- Situational context (34-d one-hot: count, outs, runners, batter-hand, inning, score-diff) is identical to the model's input context.
- Downstream model: XGBoost `multi:softprob`, 400 trees, depth 6, lr 0.08, early-stopping on a 10% slice of the train-year targets (the 2025 holdout is never seen during training).
- Outcome target (5 classes): called_strike, ball, foul, in_play, swinging_strike.  Low-count descriptions are folded in (`blocked_ball`→`ball`, `foul_tip`/`foul_bunt`/`bunt_foul_tip`→`foul`, `swinging_strike_blocked`/`missed_bunt`→`swinging_strike`, `hit_by_pitch`/`pitchout`→`ball`).
- Bootstrap CIs: 500 resamples with replacement; paired resamples for pairwise deltas.

## Leakage audit
- shared `game_pk` across splits: **0**
- shared pitchers train/holdout: **0**
- train pitchers=1444  val pitchers=121  holdout pitchers=310

## Dataset sizes
- Train targets (post-subsample): 283,617
- Holdout targets: 51,925
- Train outcome counts: `called_strike`=45,996, `ball`=102,209, `foul`=53,363, `in_play`=50,884, `swinging_strike`=31,165
- Holdout outcome counts: `in_play`=9,287, `foul`=10,056, `called_strike`=8,096, `ball`=18,648, `swinging_strike`=5,838

## Headline results (2025 holdout)

| Variant | Feature dim | Log-loss (95% CI) | Brier (95% CI) | Accuracy (95% CI) |
|---|---|---|---|---|
| null_situational_only | 34 | 1.5207 (1.5173, 1.5245) | 0.7616 (0.7598, 0.7633) | 0.3591 (0.3549, 0.3632) |
| pitchgpt_plus_situational | 85 | 1.5202 (1.5164, 1.5239) | 0.7615 (0.7597, 0.7631) | 0.3592 (0.3550, 0.3634) |
| lstm_plus_situational | 85 | 1.5202 (1.5165, 1.5239) | 0.7614 (0.7599, 0.7630) | 0.3594 (0.3556, 0.3637) |

## Pairwise deltas (A − B)

Negative log-loss/Brier delta ⇒ A better. Positive accuracy delta ⇒ A better.

| Comparison (A vs B) | Δ Log-loss (95% CI) | Δ Brier (95% CI) | Δ Accuracy (95% CI) |
|---|---|---|---|
| pitchgpt_vs_lstm | 0.0001 (-0.0005, 0.0007) | 0.0001 (-0.0002, 0.0004) | -0.0002 (-0.0008, 0.0005) |
| pitchgpt_vs_null | -0.0004 (-0.0012, 0.0004) | -0.0001 (-0.0004, 0.0002) | 0.0001 (-0.0005, 0.0007) |
| lstm_vs_null | -0.0005 (-0.0013, 0.0004) | -0.0002 (-0.0005, 0.0001) | 0.0003 (-0.0003, 0.0010) |

## High-confidence subsets

PitchGPT top-1 probability on the 2025 holdout ranges from 0.0081 to 0.5612 (mean 0.0530, p50 0.0473, p90 0.0819, p99 0.1450). The wide 2,210-token vocabulary keeps absolute confidences low; we therefore report several cutoffs: the requested absolute 0.5 threshold, a sweep at 0.10/0.20/0.30, and a top-10% percentile cutoff (0.0819).

| Subset | Threshold | n (coverage) | Δ log-loss PGpT-LSTM (95% CI) | Δ log-loss PGpT-Null (95% CI) |
|---|---|---|---|---|
| abs_threshold_0.5 | 0.5000 | 1 (0.0%) | — (n ≤ 100) | — (n ≤ 100) |
| abs_threshold_0.10 | 0.1000 | 2,389 (4.6%) | 0.0017 (-0.0019, 0.0059) | -0.0010 (-0.0068, 0.0048) |
| abs_threshold_0.20 | 0.2000 | 100 (0.2%) | — (n ≤ 100) | — (n ≤ 100) |
| abs_threshold_0.30 | 0.3000 | 6 (0.0%) | — (n ≤ 100) | — (n ≤ 100) |
| top_10pct_by_top1 | 0.0819 | 5,193 (10.0%) | 0.0007 (-0.0016, 0.0032) | -0.0009 (-0.0042, 0.0022) |

## Chance-rate context

- Majority class = `ball` (18,648 / 51,925 = 35.9% of holdout pitches).
- Majority-class predictor accuracy: **0.3591**.
- Class-frequency-prior log-loss: **1.529**.
- All three trained XGBs land at accuracy ≈ 0.359 (identical to the majority predictor) and log-loss ≈ 1.520 — so the downstream model is essentially predicting the majority class everywhere and earning only a 0.009-nat log-loss improvement over the constant prior.
- **Consequence**: the downstream task has almost no signal-to-noise left after controlling for situational context, so neither PitchGPT nor LSTM can plausibly demonstrate a downstream advantage.

## Confidence context

- PitchGPT top-1 prob mean = 0.053 (max 0.561); LSTM top-1 prob mean = 0.044 (max ≈ 0.33).
- PitchGPT > LSTM top-1 on **63.1%** of holdout pitches; PitchGPT mean entropy 4.80 vs LSTM 4.98 — consistent with its calibration edge.
- So PitchGPT IS the more confident / calibrated model. It just isn't *useful for this downstream target*.

## Interpretation

- PitchGPT vs LSTM is a statistical tie downstream (CI spans zero on every metric: log-loss, Brier, accuracy). The 13.80% perplexity edge and the ECE 0.0098 calibration do **not** translate into downstream outcome-prediction gains.
- Adding PitchGPT's distribution over the situational-only null does NOT significantly reduce log-loss (point estimate favours PGpT by 0.0004, CI [-0.0012, +0.0004]).
- The LSTM distribution shows the same null-result pattern (point -0.0005, CI [-0.0013, +0.0004]).
- On the high-confidence subset (PGpT top-1 > 0.10, 4.6% coverage) the PGpT-vs-LSTM gap actually **widens in LSTM's favour** (+0.0017 log-loss, CI [-0.0019, +0.0059]) — confidence-tightening does NOT rescue PitchGPT.
- **Verdict**: downstream-utility does NOT bulletproof the flagship claim. It shows that PitchGPT's sub-spec perplexity edge is methodology-only, not decision-value. The calibrated-transformer narrative remains defensible as a methodology artifact (tight ECE, survives temporal shift, beats naive baselines by wide margins) — but the "drives better downstream decisions" claim would be unsupported.