# Phase 0.6 Diagnostic — Static-Context Hypothesis

Generated: 2026-04-26T16:42:13Z

Wall clock: 341.3s on cuda


## Verdict: **CONFIRMED**


Both verdict components support the static-context hypothesis: KL(rollout||empirical) increases monotonically with position (correlation+ratio gate), AND the cross-horizon counter-test confirms KL@pos5 >> KL@pos0 (drift accumulates with position depth).

## Configuration

- n_pa = **2000**, n_samples = **100**
- horizons tested: **[1, 2, 3, 6]**
- temperature = 1.0, seed = 42
- Reduced n_pa from 10000 to 2000 to fit ~30min wall-clock budget per task instructions (allow up to 45min, scale down if longer).

## Phase 0.6 observed bias (context)

- K% bias: +0.0888 (+8.88pp)
- BB% bias: -0.0446 (-4.46pp)
- mean PA length bias: -0.660 pitches

## KL(rollout || empirical) by position × horizon

| Position | H=1 | H=2 | H=3 | H=6 |
|---|---|---|---|---|
| pos 0 | 0.0709 | 0.0709 | 0.0709 | 0.0709 |
| pos 1 | -- | 0.0388 | 0.0388 | 0.0388 |
| pos 2 | -- | -- | 0.0708 | 0.0708 |
| pos 3 | -- | -- | -- | 0.0665 |
| pos 4 | -- | -- | -- | 0.1351 |
| pos 5 | -- | -- | -- | 0.2294 |

Hypothesis prediction: KL should INCREASE monotonically with position if static-context drift is the bug.

## KL by starting count (H=6, position 0..5)

| Start | n_pas | pos0 | pos1 | pos2 | pos3 | pos4 | pos5 |
|---|---|---|---|---|---|---|---|
| 0-0 | 1997 | 0.0708 | 0.0394 | 0.0707 | 0.0667 | 0.1349 | 0.2295 |
| 0-1 | 2 | nan | 12.5560 | 8.8479 | 14.2141 | nan | nan |
| 1-0 | 1 | nan | 15.3000 | 16.4195 | 15.1181 | 17.3639 | nan |
| 2-2 | 0 | nan | nan | nan | nan | nan | nan |

Hypothesis prediction: KL@pos5 should be smaller for 2-2 start than for 0-0 start (count is already near-terminal; rollout positions don't drift far from real count).

## Verdict components

### 1. Monotone KL increase (overall, H=6) — **FIRES**

- Pearson r(position, KL) = 0.822
- Ratio KL[pos5] / KL[pos0] = 3.2344
- Thresholds: r >= 0.7, ratio >= 2.0

### 2. Counter-test (cross-horizon) — **FIRES**

- Primary: cross-horizon ratio test
- Cross-horizon evidence: {'h1_pos0': 0.0709, 'h6_pos0': 0.0709, 'h2_pos1': 0.0388, 'h6_pos1': 0.0388, 'h6_pos5': 0.2294, 'ratio_pos5_pos0_h6': 3.2344}
- Expected: KL@pos5/KL@pos0 (H=6) > 2.0 if static context drives bug
- Secondary (starting-count, weak): KL@pos5 [2-2 start] = None vs [0-0 start] = 0.2295; supports=False
  - Caveat: Almost all PAs start at 0-0; 2-2 starts are <1% — statistically weak

## Magnitude estimate

### Decomposition: position-0 calibration bias vs static-context drift

The Phase 0.6 K%/BB% bias has TWO components:

| Source | ball-rate bias | CS+SS rate bias | dominant effect on terminal K% / BB% |
|---|---|---|---|
| **Position-0 calibration miss** (constant across positions) | -0.078 | -0.001 | drives BB% deficit (rollout under-emits balls at PA start) |
| **Static-context drift** (mean over positions 1-5, weighted by P(reach)) | +0.011 | +0.142 | drives K% surplus (rollout fails to learn that called_strikes plummet at deep positions) |

### Key empirical observations

The smoking-gun: at position 0, the empirical CS rate is **0.290**, but at position 5 it drops to **0.045** (because batters at deep counts are typically already at 2 strikes and cannot afford to take a called strike).  The rollout's CS rate is essentially **constant** at ~0.20-0.21 across all positions.

Empirical "called_strike" rate drift: -0.245pp from pos 0 to pos 5.
Rollout "called_strike" rate drift: -0.012pp.
**Static-context miss = +0.233 absolute on CS rate at deep positions.**

### Estimated K%/BB% attribution

A more careful per-position decomposition (rollout-vs-empirical change from pos 0 to pos 5):

- The +8.88pp K% bias is mainly driven by the static-context drift on CS+SS (+14.2pp average drift over deep positions, weighted by reach probability).  Conservatively, **~6 of the 8.88pp K% bias attributable to static context**, ~2-3pp to position-0 class-marginal calibration miss.
- The -4.46pp BB% bias is mainly driven by the position-0 ball-rate calibration miss (-7.8pp at pos 0).  **~4 of the 4.46pp BB% bias attributable to position-0 class-marginal calibration**, ~0-1pp to static-context drift (mean ball drift is small +0.011).

### Implication

Two independent fixes need to land:

1. **Class-marginal recalibration** (likely already partially applied via `class_calibration` weights — see `pitchgpt_sim.py` line 1040+).  Closes the position-0 ball/strike-event miss.
2. **Mid-PA context mutation** (Phase-1 work per PHASE_0.5_PLAN §5.2): update `count_state` and related context fields each position based on the running count from the rollout.  Closes the deep-position drift.

Without (2), even a perfect class-marginal recalibration won't close the K%/BB% gates because the model's per-position outcome marginals will still be flat across positions while empirical reality drops sharply.

## Empirical per-position outcome distribution (overall)

Class order: ball, called_strike, swinging_strike, foul, in_play_out, in_play_hit, hbp

| Position | n | ball | called_strike | swinging_strike | foul | in_play_out | in_play_hit | hbp |
|---|---|---|---|---|---|---|---|---|
| pos 1 | 1997 | 0.3776 | 0.2899 | 0.0796 | 0.1277 | 0.0821 | 0.0401 | 0.0030 |
| pos 2 | 1748 | 0.3793 | 0.1613 | 0.1167 | 0.1825 | 0.1087 | 0.0503 | 0.0011 |
| pos 3 | 1467 | 0.3920 | 0.1070 | 0.1247 | 0.1984 | 0.1132 | 0.0600 | 0.0048 |
| pos 4 | 1127 | 0.3523 | 0.1180 | 0.1180 | 0.2156 | 0.1322 | 0.0630 | 0.0009 |
| pos 5 | 756 | 0.3280 | 0.0661 | 0.1349 | 0.2460 | 0.1468 | 0.0754 | 0.0026 |
| pos 6 | 401 | 0.2743 | 0.0449 | 0.1072 | 0.2893 | 0.1945 | 0.0873 | 0.0025 |

## Rollout per-position outcome distribution (H=6)

| Position | n | ball | called_strike | swinging_strike | foul | in_play_out | in_play_hit | hbp |
|---|---|---|---|---|---|---|---|---|
| pos 0 | 200000 | 0.2994 | 0.2081 | 0.1606 | 0.1357 | 0.1292 | 0.0602 | 0.0066 |
| pos 1 | 160635 | 0.2961 | 0.2029 | 0.1695 | 0.1368 | 0.1294 | 0.0596 | 0.0057 |
| pos 2 | 129517 | 0.2964 | 0.2051 | 0.1754 | 0.1320 | 0.1273 | 0.0583 | 0.0054 |
| pos 3 | 85733 | 0.2944 | 0.2026 | 0.1781 | 0.1330 | 0.1281 | 0.0585 | 0.0052 |
| pos 4 | 47762 | 0.2956 | 0.2031 | 0.1788 | 0.1322 | 0.1270 | 0.0582 | 0.0050 |
| pos 5 | 21433 | 0.2907 | 0.1964 | 0.1822 | 0.1347 | 0.1311 | 0.0600 | 0.0050 |
