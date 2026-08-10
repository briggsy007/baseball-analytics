# Phase 0.6.2 rollout-regime W fit — 2023 (audit report)

Fit date: 2026-08-10.  Spec: PHASE_0.6.2_PLAN.md §4 + §10.A7.

Cohort: 2023 pitcher-disjoint, 10000 PAs (seed 42), 100 samples/PA, horizon 6.

**Iterations used: 2.  Converged: NO — KILL SIGNAL (§6).**

| iteration | roll | max |delta| (pp) | converged |
|---|---|---|---|
| 1 | roll-0 raw-T | 4.418 | False |
| 2 | roll-2 W2 | 2.625 | False |

Final artifact: `C:\Users\hunte\projects\baseball\results\pitchgpt\rollout_calibration_fit_2023\W_FAILED_FIT_quarantine.npz` (sha256 `395e6fcd16b188f58a9fc124c5ac33fded15fb8946e137a95310c5e931b27d12`)

Per-position final W rows (classes: ball, called_strike, swinging_strike, foul, in_play_out, in_play_hit, hbp):

- pos 0: [1.9519, 1.8907, 0.5625, 1.1239, 0.6948, 0.7685, 0.8027]
- pos 1: [1.8947, 1.1355, 0.8242, 1.3951, 0.9138, 0.9523, 0.4645]
- pos 2: [1.8895, 0.5872, 0.85  , 1.6395, 0.9593, 1.1323, 0.5955]
- pos 3: [1.9427, 0.6526, 0.8606, 1.4447, 0.9763, 0.9158, 0.7094]
- pos 4: [1.6304, 0.4261, 0.8758, 1.6441, 1.084 , 1.0543, 0.8749]
- pos 5: [1.396 , 0.2582, 0.9651, 2.1621, 1.4411, 1.3192, 0.6993]