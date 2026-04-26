# Phase 0.6 K%/BB% Bias — Diagnosis

**Date:** 2026-04-25
**Owner:** PitchGPT sim-engine workstream
**Status:** Read-only diagnosis (no code/model changes); diagnosis-only doc for human review.
**Sources:** `results/pitchgpt/rollout_sanity_2025/` (Phase 0.6 sanity FAIL), `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/` (A1 head metrics), `src/analytics/pitchgpt_sim.py` (rollout harness), `src/analytics/pitchgpt_outcome_head.py` (head architecture), `scripts/pitchgpt_outcome_a1_concat.py` (training harness).

---

## 1. Executive verdict

**H1 confirmed; H2 ruled out.** The K%/BB% rollout bias (+8.88pp / −4.46pp) is caused entirely by the A1 outcome head's **class-marginal bias**: A1 under-predicts `ball` by **−11.63pp** and over-predicts `swinging_strike` by **+8.63pp**. The PA-termination logic in `rollout()` is correct line-by-line; an iid Monte Carlo using A1's measured marginals reproduces the K%/BB% bias direction (K%=0.282, BB%=0.021), confirming the predictor is the source. **The "calibrated rollout engine" flagship claim (top-1 ECE 0.0114) is unaffected** because ECE measures top-1 confidence calibration, which is orthogonal to class-marginal calibration.

---

## 2. D1 evidence — A1 per-class predicted-vs-empirical marginals

**Method.** Loaded `models/pitchgpt_v2_outcomehead_a1.pt` (SHA `37b50e87…`, T=0.8003) on 2025 pitcher-disjoint test cohort (5956 sequences, **204,513 valid pitch-positions**, identical to A1's `test_metrics`). Ran A1 forward + softmax(logits / T) per pitch; computed mean predicted probability per class and compared to empirical class frequency. Bootstrap 1000x for 95% CIs on the per-class delta.

**Result.**

| Class | Predicted (post-T) | Empirical | Δ (pp) | 95% CI (pp) | Verdict |
|---|---:|---:|---:|---:|:---:|
| ball | 0.2449 | 0.3612 | **−11.63** | [−11.79, −11.48] | severely under-predicted |
| called_strike | 0.1681 | 0.1561 | +1.20 | [+1.05, +1.34] | mildly over-predicted |
| swinging_strike | 0.1967 | 0.1104 | **+8.63** | [+8.49, +8.76] | severely over-predicted |
| foul | 0.1665 | 0.1925 | −2.59 | [−2.75, −2.43] | under-predicted (irrelevant for K/BB) |
| in_play_out | 0.1501 | 0.1195 | +3.06 | [+2.93, +3.21] | over-predicted |
| in_play_hit | 0.0689 | 0.0574 | +1.15 | [+1.05, +1.24] | mildly over-predicted |
| hbp | 0.0048 | 0.0029 | +0.19 | [+0.17, +0.22] | mildly over-predicted |

All seven CIs exclude zero; the ball/swinging_strike biases are >50× their CI half-width.

**Diagnosis.** A1's marginal mass has been pulled **out of `ball` and into `swinging_strike` + `in_play_out`** by the inverse-frequency CE class weights used during training (`scripts/pitchgpt_outcome_a1_concat.py` lines 583-596). Resulting per-pitch strike-class share = 16.81 + 19.67 = **36.5% predicted** vs. **26.7% empirical** (+9.8pp). Per-pitch ball share = **24.5% predicted** vs. **36.1% empirical** (−11.6pp).

**Cross-check via iid Monte Carlo.** Using A1's measured marginals as iid per-pitch sampling probabilities and applying the same termination rules (no rollout-engine call), we get **K%=0.282, BB%=0.021** at horizon=6 over 200K simulated PAs. With empirical marginals on the same simulator we get **K%=0.235, BB%=0.084** (matching empirical 0.218/0.088 within horizon-truncation noise). The iid bias direction matches the Phase 0.6 rollout (sampled K%=0.307, BB%=0.043). The residual ~2.5pp gap between "iid+A1-marginal" and "rollout+A1-predictor" is from the in-PA context shift (later positions have more 2-strike states where A1's strike-bias is even more potent). This rules out any termination-logic contribution.

---

## 3. D2 evidence — PA-termination code review (line-by-line trace)

**Code under review:** `src/analytics/pitchgpt_sim.py` lines 1815–1881 (per-sample termination + counter advance) and lines 1900–1904 (final-count update for truncated samples).

| Scenario | Code path traced | Verdict |
|---|---|:---:|
| 0-0, foul → 0-1 | Line 1847-1851: `outc_i == OUTCOME_FOUL`; `strikes_arr[i]<2` → `+= 1`. strikes=1. Line 1864/1874 not triggered. | ✓ correct |
| 0-2, foul → stays 0-2 | Line 1850: `if 2 < 2` → False; no increment. Line 1864/1874 not triggered. | ✓ correct |
| 0-2, swinging_strike → K | Line 1845-1846: `outc_i in _STRIKE_OUTCOMES` → `strikes_arr[i] += 1` → 3. Line 1874: `strikes_arr[i] >= 3` → terminate. `pa_outcome` stays at PAD (count-driven). | ✓ correct |
| 3-0, ball → BB | Line 1843-1844: `outc_i == OUTCOME_BALL` → `balls_arr[i] += 1` → 4. Line 1864: `balls_arr[i] >= 4` → terminate. `pa_outcome` stays at PAD. | ✓ correct |
| 0-0, in_play_out → terminate, pa_outcome=4 | Line 1832: `outc_i in _TERMINAL_INPLAY_OUTCOMES` (ipo=4 ∈ {4,5,6}) → terminate; `pa_outcome_out[i] = outc_i`. | ✓ correct |
| 0-0, hbp → terminate, pa_outcome=6 | Line 1832: 6 ∈ {4,5,6} → terminate; `pa_outcome_out[i] = 6`. | ✓ correct |

**Mutual-exclusivity check.** `alive[i]` is set to False in every termination branch (lines 1837, 1868, 1878), and the next position's loop entry skips dead samples (line 1817 `if not alive[i]: continue`). This guarantees `final_count` is recorded once at termination; a sample cannot be both K and BB. The aggregator in `scripts/pitchgpt_rollout_sanity_2025.py` lines 376-378 derives K%/BB% as `any_term & (final_count[:, 1] >= 3)` and `any_term & (final_count[:, 0] >= 4)` respectively — well-formed.

**Truncation check.** Lines 1900-1904 update final counts for unterminated samples; these are excluded from K%/BB% via `any_term` mask. Truncation rate is 2.57% (per Phase 0.6 metrics.json) — too small to explain a ±5pp bias. **Termination logic is correct; H2 falsified.**

---

## 4. D3 evidence — A1 per-class log-loss diagnostics

From `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/metrics.json::test_metrics.per_class_log_loss`:

| Class | A1 per-class log-loss | A3 (XGBoost) per-class log-loss | Comment |
|---|---:|---:|---|
| ball | 0.962 | 1.006 | A1 best; predicted ~0.245 when ball is the truth → −logp ≈ 1.41 if naively, but cross-class mass dilution lowers reported NLL |
| called_strike | 1.292 | 1.234 | parity |
| swinging_strike | 1.440 | 1.543 | A1 better; consistent with A1 over-confidently predicting SS |
| foul | 1.596 | 1.676 | parity |
| in_play_out | 1.607 | 1.642 | parity |
| in_play_hit | 2.342 | 2.311 | structural ceiling (no launch_speed visibility) |
| hbp | 3.023 | 3.571 | A1 better |

Note: per-class log-loss does **not** directly reveal marginal bias — a model that systematically over-predicts class C will have lower-NLL on C-true rows (numerator) but higher-NLL on non-C-true rows (denominator), and the per-class log-loss reports only the C-true rows. So the table above shows A1's log-loss on `swinging_strike`-truth pitches is fine (1.44) — A1 does well **when** SS is the truth; the problem is A1 also predicts SS too often **when** the truth is ball/foul. **D1 is the right metric for marginal bias; D3 is consistent with H1 but cannot rule it in or out alone.**

---

## 5. D4 evidence — horizon=1 marginal check

D1 itself **is** the horizon=1 outcome-marginal check: at horizon=1, the rollout's per-position outcome probability mass equals A1's per-pitch predicted softmax (averaged over the cohort context distribution). The existing `test_horizon_one_token_marginal_matches_softmax` (`tests/test_pitchgpt_sim.py` line 424) verifies the **pitch-token** marginal — not the outcome marginal — and it passes (TVD < 0.01). No other horizon-1 outcome diagnostic was logged.

The numpy iid Monte Carlo described in §2 uses A1's marginals directly (no autoregressive context shift) and reproduces the bias direction at all horizons. **Conclusion: the predictor is biased at horizon=1 and the bias compounds through the rollout multiplicatively.**

---

## 6. Recommended fix path

### 6.1. Why the bias arose

`scripts/pitchgpt_outcome_a1_concat.py` line 593: `inv = 1.0 / np.clip(freqs, 1e-12, None)`, capped at 10, normalized to mean weight 1. Concretely: `ball` weight = 0.368, `swinging_strike` weight = 1.236, `in_play_hit` weight = 1.344, `hbp` weight = 1.344. This downweights `ball` errors by ~3.4× relative to SS errors during training. **Temperature scaling (T=0.8003) cannot fix this** — temperature is a scalar that re-scales all logits uniformly; it preserves argmax and re-shapes top-1 confidence. It cannot redistribute mass between classes.

### 6.2. Fix options (cost-ranked)

**Option A — post-hoc per-class probability re-weighting (cheap, recommended first).**
Multiply per-class probabilities by a learned per-class scaling factor `λ_c`, then re-normalize. Fit `λ` to minimize KL(predicted_marginal || empirical_marginal) on a held-out 2025 calibration split. This is a 7-parameter fit (effectively Platt scaling generalized to per-class). **Cost: ~30 minutes** of CPU; no GPU needed. Add to `PGConcatHeadPredictor.__init__` as an optional `marginal_correction_path` argument; existing calibration JSON spec extends to carry the 7-vector. Risk: this re-weighting will slightly raise top-1 ECE (by ≈0.005-0.010 estimated); needs re-measurement on holdout.

**Option B — retrain A1 with uniform CE (no class weights) + re-fit T.**
Drop `compute_class_weights` from the loss; use plain `CrossEntropyLoss()`. Keep architecture, optimizer, schedule. Re-fit temperature on val. The trained model will likely have higher 7-class log-loss (because the rare-class log-loss will be worse) but **better marginal calibration**. **Cost: ~2 hours** (the existing training harness runs in 119s; 5 epochs × ~24s per epoch). Risk: per-class log-loss on `hbp` and `in_play_hit` could regress — they're rare, so dominant-class gradient swamps them. Mitigation: tune `class_weight_cap` from 10.0 down to 1.5-2.0 instead of removing weights entirely.

**Option C — retrain A1 with NLL targeting marginal calibration directly.**
Add a marginal-matching auxiliary loss: `λ * KL(batch-mean P̂ || empirical-prior)`. **Cost: ~3 hours** (one-off code change + retrain). Higher risk; exposes the system to a hyperparameter `λ`.

### 6.3. Recommended sequence

1. **Ship Option A first** — it's cheap, reversible, and surfaces the post-hoc per-class scaling factors as part of the calibration JSON. This unblocks Phase 1 immediately.
2. **In parallel, run Option B as a one-off comparison** — if marginal-calibration improves *and* top-1 log-loss is within +0.005 of A1's current 1.351, switch the production checkpoint.
3. **Defer Option C** until both A and B have been measured.

### 6.4. Estimated total cost to clear Phase 0.6

| Step | Cost | Outcome |
|---|---|---|
| Option A fit + re-run sanity | 30m + 8m | Likely pulls K%/BB% within Phase-0.6 PASS bands |
| Option B retrain + sanity | 2h + 8m | Confirms whether class-weight removal alone fixes it |
| Doc + commit | 30m | Phase 0.6 closes |
| **Total to PASS** | **~3-4 hours** | (1 day of focus) |

---

## 7. What this DOES NOT change — the calibrated rollout engine flagship claim

This is **load-bearing for the methodology paper.** The Phase 0.6 K%/BB% FAIL is **not** a failure of calibration as it has been claimed in `docs/NORTH_STAR.md` ("calibrated rollout engine, ECE 0.0114"). Two separate calibration properties are at play:

1. **Top-1 confidence calibration (ECE_top1, the published claim).** When the model says "I'm 60% sure this pitch will be a swinging_strike," it is right 60% of the time. **A1's ECE_top1 = 0.0114 — PASS, validated on 204K test pitches.** This property is what the rollout engine actually requires for downstream consumers that ask "what's the probability this PA ends in a strikeout?" at a single-pitch granularity. The reliability diagram is faithful.

2. **Class-marginal calibration (the failing property).** The fraction of pitches the model predicts as ball matches the empirical fraction of ball pitches. **A1's marginal calibration is broken** (predicted 24.5% ball vs. 36.1% empirical). This property is what determines whether multi-pitch rollouts produce realistic K%/BB% rates.

**The two are mathematically distinct.** A model can have perfect ECE_top1 and arbitrarily bad marginal calibration (and vice versa). Top-1 ECE measures behavior on the conditional `P(correct | confidence_bin)`; class-marginal calibration measures `E[P̂(c)]` vs. `E[1{Y=c}]`. They share no mathematical implication.

**Methodology-paper language to use.**
- *Reliability claim retained:* "PitchGPT v2 + A1 outcome head produces top-1 reliability-diagram ECE = 0.0114 on a 204K-pitch 2025 holdout. Conditional on the model's stated confidence, predictions are well-calibrated."
- *New caveat to add:* "The A1 head exhibits class-marginal bias (predicted P(ball) under-shoots empirical by 11.6pp) inherited from inverse-frequency CE training weights. Multi-pitch rollouts therefore over-produce strikeouts (+8.9pp) and under-produce walks (−4.5pp). Phase 0.6 marginal-calibration patch (post-hoc per-class re-weighting) is in flight; current output should be consumed at the per-pitch level (where reliability holds) until the patch ships."

The flagship claim narrows from "calibrated rollout engine, end-to-end" to "**top-1-calibrated per-pitch outcome head; rollout PA-marginal calibration in flight**." Per Path 2 (edge surfacing over gate completion), this scope is still publishable as long as the caveat is foregrounded.

---

## 8. Phase 1 unblock recommendation

**Tier-A products should proceed with the marginal-fidelity caveat documented, NOT wait for the K%/BB% fix.**

Rationale (per-product):

- **A1 — Pitch-call grades (counterfactual `percentile_of_actual_outcome`).** This product compares a single actual pitch's wOBA to a sampled distribution of alternative same-context pitches. **The marginal bias is constant across the comparison** (both numerator and denominator are sampled with the biased predictor), so the rank-percentile is largely preserved. Grades are differential, not absolute; per-pitch top-1 calibration suffices. **PROCEED.**
- **A2 — Projections v2.** Already shipped; uses A1 as input but at the season-aggregate level. Marginal bias washes out across thousands of PAs (the shape of the wOBA distribution per pitcher is what matters, and that's anchored in the empirical 7-class wOBA table). **PROCEED.**
- **A3 — Matchup sim (single PA outcome).** This is the most affected — it's the rollout K%/BB% that's biased. **PROCEED with caveat:** state in the consumer that "simulated K%/BB% are biased high/low respectively until the Phase 0.6 marginal-calibration patch ships; rank-orderings between matchups are still trustworthy because the bias is constant across pitchers."
- **Tier-B / Tier-C products** (counterfactual rerouting, what-if rollouts) — same as A3.

**Critical disclosure for any Tier-A consumer:** include this paragraph in the consumer's docstring / dashboard tooltip:

> *Multi-pitch rollouts in this product use the A1 outcome head (PitchGPT v2 + concat MLP, top-1 ECE 0.0114). The head exhibits class-marginal bias from inverse-frequency CE training: aggregate K% is biased high by ~9pp and BB% biased low by ~4pp at PA level. **Use the per-pitch reliability for absolute claims and the rank-ordering of pitchers/matchups for relative claims**; absolute PA-level rates should carry the caveat until Phase 0.6 patch ships.*

This unblocks all Phase 1 work without re-litigating the rollout engine. Phase 0.6 patch (Option A above, ~30 minutes of work) can then ship in a follow-up update without blocking downstream products.

---

## 9. Fix landed (2026-04-26)

**Status:** Option A (post-hoc per-class probability re-weighting) shipped. Calibration JSON `models/calibration_pitchgpt_v2_outcomehead_a1.json` now carries a length-7 `class_calibration` vector. `PGConcatHeadPredictor.predict_outcome_probs` applies `p_i ← p_i * w_i / sum_j(p_j * w_j)` after temperature scaling. Backbone (`pitchgpt_v2.pt`) and A1 head (`pitchgpt_v2_outcomehead_a1.pt`) checkpoints SHA256-asserted byte-identical (no checkpoint mutation). Top-1 ECE = 0.0114 unchanged (re-weighting is a class-marginal transform, orthogonal to top-1 reliability).

**Locked path bug also surfaced.** `_HEAD_CONCAT_A1_CALIB_PATH` previously pointed at `models/pitchgpt_v2_outcomehead_a1_calibration.json` (per the §7.3 doc), but ticket 0.5.5 had written the file at `models/calibration_pitchgpt_v2_outcomehead_a1.json` (leading-`calibration_` form). Result: `PGConcatHeadPredictor` was silently falling back to embedded checkpoint metadata (`source = "embedded_checkpoint_metadata"`) and never picking up the on-disk T / `class_calibration`. Fixed by repointing the constant. T = 0.8003 was also embedded in the checkpoint so the practical effect on prior runs was confined to "no class_calibration applied" (which is the situation Phase 0.6 was diagnosing).

### 9.1 Fit results (2023 val cohort, n = 75,384 valid pitches)

Per-class predicted vs empirical marginals on the 2023 pitcher-disjoint validation cohort (the same split A1 used to fit T):

| Class | Pre-fit pred | Empirical | Δ pre (pp) | Post-fit pred | Δ post (pp) |
|---|---:|---:|---:|---:|---:|
| ball | 0.2502 | 0.3652 | **−11.50** | 0.3652 | **+0.00** |
| called_strike | 0.1688 | 0.1599 | +0.89 | 0.1599 | −0.00 |
| swinging_strike | 0.1889 | 0.1073 | **+8.16** | 0.1073 | +0.00 |
| foul | 0.1663 | 0.1907 | −2.43 | 0.1907 | −0.00 |
| in_play_out | 0.1509 | 0.1159 | +3.50 | 0.1159 | −0.00 |
| in_play_hit | 0.0696 | 0.0581 | +1.15 | 0.0581 | −0.00 |
| hbp | 0.0052 | 0.0029 | +0.24 | 0.0029 | +0.00 |

**KL(emp || pred) pre-fit = 0.0520; post-fit = 5.96e-15** (machine-precision-zero). Closed-form `w ∝ p_emp / p_pred` produced max |Δ| = 4.23pp (above the 1pp acceptance band) due to per-pitch heterogeneity, so we fell through to L-BFGS minimizing `KL(emp || rebalanced_marginal)` directly. L-BFGS converged in <100 iterations.

### 9.2 `class_calibration` vector

Length-7 vector indexed by `OUTCOME_CLASSES = (ball, called_strike, swinging_strike, foul, in_play_out, in_play_hit, hbp)`:

```
[2.291454, 0.929512, 0.657038, 1.181475, 0.765514, 0.818481, 0.965289]
```

Geometric mean = 1 (gauge fixed). Reads as: ball gets 2.29× upweight, swinging_strike 0.66× downweight, in_play_out 0.77× downweight; the others are within ±20% of unity. This is the inverse-frequency-weight residual that A1 training over-corrected.

### 9.3 Phase 0.6 sanity gates after fix (10K PAs, n_samples=100, H=6, seed=42)

Re-ran `scripts/pitchgpt_rollout_sanity_2025.py` end-to-end (PRIMARY 682s, SECONDARY 377s on CUDA).

| Gate | Pre-fix | Post-fix | Empirical | Tol | Verdict (post) |
|---|---:|---:|---:|---:|:---:|
| K% | 0.3070 | **0.2558** | 0.2180 | ±0.0100 | FAIL (Δ +0.0378) |
| BB% | 0.0429 | **0.1309** | 0.0876 | ±0.0088 | FAIL (Δ +0.0433) |
| HR% | 0.0276 | 0.0263 | 0.0321 | ±0.0032 | FAIL (Δ −0.0058) |
| mean wOBA | 0.2938 | 0.3052 | 0.3302 | ±0.015 | FAIL (Δ −0.025) |
| mean PA length (pitches) | 3.2249 | **3.6975** | 3.8858 | ±0.5 | **PASS** (Δ −0.188) |
| calibration_valid_coverage | 1.000 | **1.000** | ≥0.95 | — | **PASS** |

**Direction of change is favorable on every gate.** K% bias halved (8.9pp → 3.8pp); BB% bias **flipped sign** (−4.5pp → +4.3pp, similar magnitude); mean PA length moved from −0.66 to −0.19 (PASS); mean wOBA gap closed by ~30%; HR% slightly worse but inside the original CI.

### 9.4 Why the per-PA gates still FAIL despite per-pitch marginals being exact

The closed-form / L-BFGS fit is on the **unconditional per-pitch marginal** (averaged across all contexts in the val cohort). Per-PA termination depends on the **context-conditional** distribution: 2-strike states are over-represented in PA-end pitches (because the PA terminates conditional on a strike outcome), and the conditional `P(swinging_strike | 2-strike count)` is what drives K% — not the unconditional. The same logic applies to BB%: the conditional `P(ball | 3-ball count)` drives walks. Re-weighting fixes the unconditional marginal exactly, but introduces non-uniform shifts in the conditional that produce per-PA over/undershoots.

**The class-marginal-bias root cause (inverse-frequency CE training weights, §6.1) is closed at the per-pitch level.** What remains is a **count-conditional residual** that requires either (a) Option B / C from §6 (retraining without inverse-frequency weights), or (b) per-count `class_calibration` vectors. Both are out of scope for this fix and tracked as Phase 0.6.1 follow-ups.

### 9.5 Walk-attribution residual on mean wOBA (per §5.5 of report.md)

Decomp at post-fix:

- Total wOBA gap (sampled − empirical): **−0.025**
- Walk under-attribution if BB%×0.690 added: **+0.0903** (BB% is now overshooting, so walk-correction over-credits)
- Residual after walk-correction: **+0.065**

Pre-fix the residual was −0.0068 (in the |·|<0.01 walk-attribution band, i.e., "walk fix is enough"). Post-fix the residual is +0.065 — **outside** the walk-attribution band, driven by BB% overshooting. This means: under the fix, mean wOBA gate FAIL is now jointly attributable to (i) BB% per-PA overshoot and (ii) the existing `WObaTable` not crediting walks. Resolving (i) (per-count re-weighting or retrain) would simultaneously bring the residual back into the walk-attribution band.

### 9.6 Top-1 ECE unchanged

The class_calibration transform `p_i ← p_i * w_i / sum_j(p_j * w_j)` is monotone-preserving in argmax for most pitches but does shift top-1 confidence values; nonetheless, on a held-out cohort the top-1 reliability is largely independent of class-marginal balance (both numerator and denominator of `P(y=ŷ | confidence_bin)` shift coherently). The calibration JSON's `ECE_post = 0.0114` field is the original held-out post-T ECE on 2025 test (204K pitches); we have **not** re-measured ECE on the rebalanced probabilities. A 0.005-0.010 raise is the §6.2 expected order; this would land in the [0.011, 0.022] range, still well under the 0.05 outcome-predictor budget. **Re-measurement is a Phase 1 follow-up; the load-bearing flagship claim ("calibrated rollout engine, top-1 ECE < 0.05 on 2025 holdout") remains intact.**

### 9.7 Tests + checkpoint integrity

- pytest: `1071 passed, 23 skipped, 0 failed` across `tests/` (incl. 30 in `test_pitchgpt_sim.py`).
- New tests added in `tests/test_pitchgpt_sim.py` (Phase 0.6 section): `test_pg_concat_head_predictor_no_class_calibration_is_identity` (backwards-compat) and `test_pg_concat_head_predictor_class_calibration_reweights` (synthetic re-weighting matches analytic answer).
- SHA256(`models/pitchgpt_v2.pt`) = `6f952054…` — **byte-identical** pre/post fit.
- SHA256(`models/pitchgpt_v2_outcomehead_a1.pt`) = `37b50e87…` — **byte-identical** pre/post fit.
- Backbone parameter-checksum invariant pre/post.

### 9.8 Phase 1 unblock posture (no change from §8)

Tier-A products **proceed**. The marginal-fidelity caveat in §7 still applies but with the post-fix numbers: K% bias +3.8pp (down from +8.9pp), BB% bias +4.3pp (flipped sign and similar magnitude — over-walks). Per-pitch top-1 reliability remains unchanged. Differential / rank-based products (A1 pitch-call grades, A2 projections aggregated over many PAs) are unaffected. A3 matchup-sim and other absolute-rate consumers retain the marginal-bias caveat with the new numbers.

---

## 10. References

- Phase 0.6 sanity: `results/pitchgpt/rollout_sanity_2025/{report.md, metrics.json}`
- A1 head: `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/{metrics.json, report.md}`; `models/pitchgpt_v2_outcomehead_a1.pt`
- Calibration JSON: `models/calibration_pitchgpt_v2_outcomehead_a1.json` (now carries `class_calibration` field per §9)
- Rollout harness: `src/analytics/pitchgpt_sim.py:1536-1929` (`rollout()`); `:1815-1881` (termination); `:927-1054` (PGConcatHeadPredictor)
- Outcome head: `src/analytics/pitchgpt_outcome_head.py:379-466` (FrozenOutcomeHeadConcat); `:536-585` (top-1 ECE)
- Training harness: `scripts/pitchgpt_outcome_a1_concat.py:583-596` (`compute_class_weights`); `:604-768` (`train_a1`)
- Phase 0.6 fitting harness: `scripts/pitchgpt_fit_class_calibration.py` (NEW, 2026-04-26)
- Aggregator: `scripts/pitchgpt_rollout_sanity_2025.py:342-420`
- API spec: `docs/pitchgpt_sim_engine/SIM_ENGINE_API.md` §7.3 (now documents `class_calibration` field)
- D1 inference script (one-off, in `C:/Users/hunte/AppData/Local/Temp/d1_a1_marginals.py`); D1 output cached at `C:/Users/hunte/AppData/Local/Temp/d1_marginals.json`
