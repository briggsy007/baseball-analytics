# A1 — Frozen v2 Backbone + Concat MLP Head

**Verdict (A1 standalone):** **WEAKER PASS**
**Ship verdict (A1 vs A3 paired):** **SHIP A1**

## TL;DR

- 7-class log-loss (post-T): **1.3507**  (freq prior: 1.6535)
- Lift vs prior: **+18.31%** (95% CI [+18.10%, +18.53%])
- 10-bin ECE post-T: **0.0114**
- Top-1 accuracy: 0.4671
- Temperature: 0.8003
- Best epoch: 3  Wall-clock: 119.3s

### A1 vs A3 paired-bootstrap delta

- Paired rows: **204,513** (A1=204,513, A3=210,482)
- Lift delta (A1 - A3): **+2.4768%** 95% CI [+2.2380%, +2.7189%]
- Log-loss delta (A1 - A3): **-0.0410** 95% CI [-0.0450, -0.0370]  (negative = A1 wins)
- A1 paired lift: +18.34%, A3 paired lift: +15.86%

## Architecture

- **Backbone:** `models/pitchgpt_v2.pt` — FROZEN by construction (byte-identity verified pre/post).
- **Head input:** concat(hidden[128] + context[35] + pitch_type_oh[17] + zone_oh[26] + velo_oh[5]) = 211d
- **Head:** MLP `211 -> 128 -> 64 -> 7` (ReLU + dropout 0.1).
- **Loss:** weighted CE, inverse-frequency class weights cap 10.

## Cohort

- Train sequences: 85,879  valid outcomes: 2,793,715
- Val sequences: 1,890  valid outcomes: 75,384
- Test sequences: 5,956  valid outcomes: 204,513
- Test unique pitchers: 473

## Training history

| epoch | train loss (wtd) | val log-loss | seconds |
|-------|------------------|-------------:|--------:|
| 1 | 1.5543 | 1.3750 | 23.2 |
| 2 | 1.5068 | 1.3850 | 23.5 |
| 3 | 1.4966 | 1.3555 | 24.2 |  **<-best**
| 4 | 1.4921 | 1.3698 | 23.8 |
| 5 | 1.4900 | 1.3737 | 23.7 |

## Per-class log-loss (test, post-temp)

| class | log-loss |
|-------|---------:|
| ball | 0.9624 |
| called_strike | 1.2917 |
| swinging_strike | 1.4399 |
| foul | 1.5962 |
| in_play_out | 1.6070 |
| in_play_hit | 2.3422 |
| hbp | 3.0234 |

## Val metrics

- Log-loss pre/post: 1.3555 / 1.3439
- ECE pre/post: 0.0676 / 0.0172
- Freq prior log-loss: 1.6488
- Lift post-T: +18.49%

## Per-pitcher stability (top-50 most-frequent test pitchers)

- n pitchers (n>=30 rows each): 50
- Log-loss mean / var / range: 1.3455 / 0.0010 / [1.2685, 1.4014]

## Backbone byte-identity verification

- v2.pt SHA256 pre:  `6f952054d14ac6f918f3eb9502b496b70bc0c87dfc65dc50d98ee7244a62883c`
- v2.pt SHA256 post: `6f952054d14ac6f918f3eb9502b496b70bc0c87dfc65dc50d98ee7244a62883c`
- Backbone param-SHA pre:  `c9b79869f0dc6da75821f0de3a0d3b32e920ef128c0fc19c7f127a375529fb12`
- Backbone param-SHA post: `c9b79869f0dc6da75821f0de3a0d3b32e920ef128c0fc19c7f127a375529fb12`
- **Byte-identity verified:** YES
- Phase 0.3 ckpt present (untouched): YES
- Phase 0.3 ckpt SHA256: `6b47f97dd69604355f598bd3fc52bc1b34a41ea9854f5cd23060f6c64cbb54a0`

## Final ship recommendation

**SHIP A1**

A1 beat A3 by >= +1pp lift on 2025 holdout with paired CI lo > 0. The PG v2 backbone adds marginal value beyond engineered features. Ship A1.
