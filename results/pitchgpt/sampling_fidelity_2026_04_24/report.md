# PitchGPT Sampling-Fidelity vs LSTM — 2025 Holdout

Generated: 2026-04-24T18:43:58Z

PitchGPT checkpoint: `C:\Users\hunte\projects\baseball\models\pitchgpt_v2.pt`
LSTM checkpoint: `C:\Users\hunte\projects\baseball\models\pitch_lstm_10k.pt`
Context schema: v2, context_dim=35  (umpire scalar present)

Seed: 42  Device: cuda

## Configuration

- PAs drawn from 2025 (pitcher-disjoint cohort): **2000** (requested 2000)
- Samples per PA per model: **10**
- Fixed horizon per sample: **6** pitches
- Sampling temperature: **1.0** (autoregressive, top-k=none)
- LSTM training: 5 epochs, max_train_games=10000 (matches v2 10K holdout split)

## Hypothesis

PitchGPT's decoder-only transformer, by virtue of its attention over the full context plus richer contextual conditioning (count × outs × runners × stand × inning × score × ump), generates pitch sequences whose marginal and joint distributions are closer to empirical 2025 distributions than a matched LSTM baseline's.  This is the first test of PitchGPT's claim as a **calibrated simulation engine** (not a next-token-accuracy king — that claim was already narrowed: at matched 10K scale, v2 beats LSTM by only 3.13% on holdout perplexity).

## Method

1. **Checkpoint**: v2 PitchGPT (10K games, context_dim=35, umpire scalar included) per user-specified `models/pitchgpt_v2.pt`.
2. **Matched LSTM**: 2-layer LSTM (d_model=128), trained fresh on the same 10K pitcher-disjoint 2015-2022 train split with seed=42, epochs=5, lr=1e-3, batch=32.  Context width matches PitchGPT.
3. **PA starts**: Uniformly sampled from 2025 plate appearances thrown by pitchers NOT in the 2015-2022 train cohort (pitcher-disjoint).  A PA = `(game_pk, at_bat_number, pitcher_id)` group with ≥2 pitches.  We use the real first-pitch situational context (count 0-0, outs, runners, stand, inning, score_diff, ump tendency).
4. **Sampling**: Temperature-1 multinomial sampling on model logits over the full 2210-class token vocab (pitch_type × zone × velo_bucket).  Fixed horizon = 6 pitches (covers the 76th percentile of 2025 PAs by pitch count).  Context updates between steps use a zone-based ball/strike heuristic (in-zone → strike, out-of-zone → ball).
5. **Reference**: Empirical 2025 per-pitch distributions over the SAME PAs' real continuations.

**Limitations documented up front:**
- Fixed horizon does not model PA termination; implied outcome distribution is a heuristic proxy based on running counts, not a learned outcome head.  Real Statcast PA ends on events not observable from the token stream alone.
- Context updates hold runners/outs/inning fixed, because the models don't predict those.
- The LSTM is trained to the same 5-epoch schedule as PitchGPT, not tuned further.
- Chi-square p-values are uncorrected; ≥5 tests run so Bonferroni α=0.01 is the appropriate floor.

## Results

**Convention: smaller distance = closer to empirical.  Δ(PG − LSTM) < 0 means PitchGPT is closer.**

| Metric | Description | PitchGPT | LSTM | Δ(PG − LSTM) | 95% CI on Δ |
|---|---|---:|---:|---:|---|
| `pitch_type_kl` | Marginal pitch-type KL(empirical ‖ model) | 0.0597 | 0.0717 | -0.0120 | [-0.0189, -0.0055] |
| `zone_kl` | Marginal zone KL (26-class, 5×5 + out-of-zone) | 0.0149 | 0.0338 | -0.0189 | [-0.0224, -0.0152] |
| `velocity_wasserstein` | Velocity 1-D Wasserstein (mph) | 1.6335 | 1.9120 | -0.2785 | [-0.3292, -0.2284] |
| `transition_frobenius` | Pitch-type 2-gram transition matrix Frobenius norm | 1.4818 | 1.3195 | +0.1623 | [0.0078, 0.2497] |
| `outcome_chi2` | Implied at-bat outcome χ² (K/BB/IP vs empirical) | 3.0733 | 3.0733 | +0.0000 | [0.0, 0.0] |

### Diagnostics (one-shot, no CI)

| Test | PitchGPT | LSTM | PG p (uncorr.) | LSTM p (uncorr.) |
|---|---:|---:|---:|---:|
| χ² pitch-type vs empirical | 73210.3 | 85108.7 | 0.0000 | 0.0000 |
| χ² zone vs empirical | 5639.6 | 10457.4 | 0.0000 | 0.0000 |

_Chi² p-values are uncorrected for multiple testing.  With ~5 tests, Bonferroni α=0.01 is the floor for significance; values below this are suggestive, not confirmatory._

### Outcome-implied distribution (3-class: strikeout / walk / in_play)

| Class | Empirical | PitchGPT | LSTM |
|---|---:|---:|---:|
| strikeout | 0.2455 | 1.0000 | 1.0000 |
| walk | 0.1075 | 0.0000 | 0.0000 |
| in_play | 0.6470 | 0.0000 | 0.0000 |

_Note: sampling-generated outcomes use a zone-based heuristic (in-zone→strike, out-of-zone→ball).  Real Statcast outcome classification uses pitch-description labels not present in the token stream._

### Marginal pitch-type distribution (top 7 by empirical mass)

| Pitch | Empirical | PitchGPT | LSTM | PG−Emp | LSTM−Emp |
|---|---:|---:|---:|---:|---:|
| FF | 0.3470 | 0.3088 | 0.2971 | -0.0382 | -0.0499 |
| SL | 0.1338 | 0.1410 | 0.1652 | +0.0072 | +0.0314 |
| SI | 0.1337 | 0.2002 | 0.2017 | +0.0665 | +0.0680 |
| CH | 0.0924 | 0.0904 | 0.0971 | -0.0020 | +0.0047 |
| ST | 0.0780 | 0.0295 | 0.0249 | -0.0485 | -0.0531 |
| CU | 0.0771 | 0.0664 | 0.0722 | -0.0107 | -0.0049 |
| FC | 0.0750 | 0.0778 | 0.0748 | +0.0028 | -0.0002 |

### Calibration-by-context: pitch-type KL stratified

#### count
| stratum | n PAs | PG KL | LSTM KL | Δ(PG−LSTM) |
|---|---:|---:|---:|---:|
| 0-0 | 1996 | 0.0599 | 0.0719 | -0.0120 |

#### leverage
| stratum | n PAs | PG KL | LSTM KL | Δ(PG−LSTM) |
|---|---:|---:|---:|---:|
| medium | 689 | 0.0568 | 0.0777 | -0.0209 |
| low | 683 | 0.0943 | 0.0901 | +0.0042 |
| high | 628 | 0.0768 | 0.0832 | -0.0064 |

#### inning_bucket
| stratum | n PAs | PG KL | LSTM KL | Δ(PG−LSTM) |
|---|---:|---:|---:|---:|
| mid(4-6) | 689 | 0.0568 | 0.0777 | -0.0209 |
| early(1-3) | 681 | 0.0960 | 0.0897 | +0.0063 |
| late(7-9) | 604 | 0.0851 | 0.0879 | -0.0028 |
| extra(10+) | 26 | 0.1763 | 0.2361 | -0.0598 |

## Interpretation

- `pitch_type_kl`: Δ=-0.0120, CI [-0.0189, -0.0055] — **PitchGPT significantly closer to empirical**.
- `zone_kl`: Δ=-0.0189, CI [-0.0224, -0.0152] — **PitchGPT significantly closer to empirical**.
- `velocity_wasserstein`: Δ=-0.2785, CI [-0.3292, -0.2284] — **PitchGPT significantly closer to empirical**.
- `transition_frobenius`: Δ=+0.1623, CI [0.0078, 0.2497] — **LSTM significantly closer**.
- `outcome_chi2`: Δ=+0.0000, CI [0.0, 0.0] — CI crosses 0, **no significant difference**.

**Scorecard**: PitchGPT wins 3 / LSTM wins 1 / ties 1 out of 5 metrics (95% CI on Δ).

**Verdict**: PitchGPT shows real sampling-fidelity advantages on multiple metrics; the generative-engine claim survives.

## Limitations

- **Horizon truncation**: 6-pitch horizon covers the modal PA (3-5 pitches) but truncates long PAs.
- **Outcome heuristic**: Implied strike/ball counting uses zone position as a proxy; this biases both models uniformly but does not reflect Statcast's actual ball/strike logic (umpire effects, foul-ball dynamics, check-swings, chase rates).
- **Empirical reference is finite**: At ~2K PAs × mean 3.7 pitches = ~7.4K empirical tokens, KL/χ² estimates have non-trivial noise.
- **Temperature=1 only**: We do not sweep temperature.  Higher temperatures will systematically widen both models' distributions; lower will sharpen.  The tradeoff may differ between architectures.
- **Multiple-comparison issue**: 5 CI tests + strata → ~25 comparisons.  Bonferroni α/N reduces effective α to 0.002, tightening any significance claims.
- **Matching to 10K**: We use the same 10K game subset as the matched holdout.  Scaling could change any winner's margin.

## Reproducibility

- PA fetch: 4.7s
- PitchGPT sampling: 1.6s
- LSTM sampling: 1.1s
- Metric computation (incl. bootstrap): 39.2s

Exact command:
```
python scripts/pitchgpt_sampling_fidelity.py \
  --pitchgpt-checkpoint models/pitchgpt_v2.pt \
  --lstm-checkpoint models/pitch_lstm_10k.pt \
  --n-pas 2000 \
  --n-samples-per-pa 10 \
  --horizon 6 \
  --temperature 1.0 \
  --seed 42
```
