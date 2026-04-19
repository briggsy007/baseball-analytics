# ChemNet Validation Results

**Spec:** `docs/models/chemnet_validation_spec.md`
**Validation script:** `scripts/chemnet_validation.py`
**Run dir:** `results/validate_chemnet_20260418T204255Z/`

## Headline verdict

**FAIL — NOT FLAGSHIP / NEEDS WORK.**

Four of five hard gates FAIL on the held-out 2023-2024 game-side cohort
(n = 9,836 game-sides across 5,010 unique games). Only the structural
leakage gate PASSES (by construction, season-disjoint train/test ranges).
The shipped `models/chemnet_v1.pt` checkpoint does NOT predict its own
training target (game-side total wOBA) on the held-out window, the GNN
does NOT measurably beat its non-graph baseline, and the dashboard's
"synergy score" is statistically a near-zero residual signal that does
not meaningfully predict lineup over- or under-performance.

This finding is itself useful: ChemNet should not be promoted to the
flagship trio (CausalWAR, PitchGPT, VWR), and the `Lineup Synergy &
Protection Effects` page in the Streamlit dashboard should be marked
"experimental — not validated" or hidden until the model is rebuilt.

## Methodology

### Cohort

- **Test window.** All MLB games 2023-01-01 -> 2024-12-31 with `>= 18`
  distinct `batter_id`s in the `pitches` table (i.e. real two-sided
  starting lineups). 5,010 game_pks survive this filter; we score both
  `home` and `away` sides, yielding 10,020 candidate game-sides.
- **PA filter.** Sides with `< 25` PAs (`woba_denom > 0`) are dropped to
  exclude rain-shortened games and one-sided blowouts where the target
  variable is dominated by sample-size noise. 184 sides drop here, leaving
  **9,836** in the headline.
- **Train reference.** 2015-2022. The shipped checkpoint's exact training
  manifest is not preserved (no metadata JSON), but the season-disjoint
  audit confirms zero `game_pk` overlap between this reference range and
  the test cohort.

### What the model predicts

Per spec section A: `ChemNetGNN(node_features, adjacency)` outputs a
single scalar predicting `SUM(woba_value)` for one team-side of one
game (typical range 6-18 wOBA-points). The companion `BaselineMLP` flatten
of the same 9x5 features (no graph) is trained on the same target with
the same loss; `synergy_score = gnn_pred - baseline_pred`.

### What we measured

For each game-side in the test cohort:
1. Build the 9-node graph from `pitches`-derived lineup composition
   (first 9 unique batters by `MIN(at_bat_number)`) plus rolling 30-game
   pre-game features.
2. Run both `gnn` and `baseline` forward passes on GPU (cuda:0).
3. Record (`gnn_pred`, `baseline_pred`, `actual_total_woba`,
   `synergy_score`, `residual_woba = actual - baseline_pred`).
4. Compute Pearson r, Spearman rho, RMSE, MAE on (pred, actual) for both
   models; bootstrap 95 % CIs over 1,000 paired-resamples.
5. Compute Spearman rho on (synergy_score, residual_woba) — the dashboard's
   implicit claim — also with 1,000-rep bootstrap CI.
6. Compute decile-stratified mean residual lift to characterise the
   top-vs-bottom synergy edge.

Wall clock: 9 min 14 s (model load 0.1 s, inference 553 s on RTX 3050,
metrics 2.6 s).

## Gate table

| Gate | Threshold | Measured (95% CI) | Verdict |
|---|---|---|---|
| Gate 1 — GNN Pearson r vs actual game-side wOBA | `>= 0.30` | **0.0891** [0.0702, 0.1095] | **FAIL** |
| Gate 2 — GNN beats baseline by RMSE | `>= 2% RMSE reduction` | **-0.14%** (GNN is *worse*) | **FAIL** |
| Gate 3 — Synergy predicts residual (Spearman) | `>= 0.10` with CI lower > 0 | **0.0263** [0.0066, 0.0457] | **FAIL** (point misses; CI lower is just barely > 0) |
| Gate 4 — GNN RMSE calibration | `<= 4.0` wOBA-points | **4.5855** | **FAIL** |
| Gate 5 — Leakage audit | 0 shared `game_pk`s train/test | **0** (5010 test pks vs 17094 train pks, season-disjoint) | **PASS** |
| Gate 6 (informational) — Top-decile vs bottom-decile residual lift | observed | top decile mean residual 0.578, bottom 0.329, lift +0.249 wOBA-points | informational |

## Honest interpretation

### What this tells us

1. **The model does not predict its own training target out-of-sample.**
   `r = 0.089` with the upper 95 % CI bound at `0.110` rules out anything
   that could be called predictive performance. By comparison, simply
   predicting the league mean for every game-side (a trivial null model)
   would yield `r = 0.0` with `RMSE ~ 4.6` — which is exactly what we
   observe. The GNN's RMSE of 4.586 is essentially identical to the
   global-mean RMSE; the model has learned to produce a prediction near
   the mean.

2. **The graph structure is not earning its complexity.** The non-graph
   baseline lands at the same `r = 0.085` and a *0.0066 wOBA-points
   smaller* RMSE. This means the GNN's adjacency-aware attention, which
   is the entire premise of ChemNet's "lineup synergy" framing, is
   contributing exactly nothing on the held-out window. The two models
   are essentially indistinguishable point predictors. As a direct
   consequence:

3. **The synergy score is statistical noise.** `synergy = gnn - baseline`
   subtracts two near-identical estimators, so the residual is dominated
   by the small differences in their respective fits to training noise.
   The Spearman correlation between `synergy_score` and the actual
   residual `(actual - baseline_pred)` is `rho = 0.026`, with the lower
   95 % CI at `0.0066` — only barely separated from zero. The dashboard's
   claim that positive synergy means "the batting order adds value beyond
   the sum of individual parts" cannot be supported by these data.

4. **The decile lift is real but tiny and monotonically inconsistent.**
   Top-synergy-decile game-sides outperform bottom-decile by `+0.25`
   summed wOBA-points (~0.02 wOBA per PA). Across the 10 buckets the
   trend is upward but noisy — buckets 4 and 5 invert. This is the
   strongest "edge signal" we found, and it is small enough to be
   indistinguishable from the systematic positive baseline bias (mean
   residual across all buckets is `+0.45`, suggesting the trained
   baseline systematically under-predicts wOBA in 2023-2024 by about
   half a wOBA-point — a calibration drift independent of any synergy
   signal).

### Why it likely fails (root-cause hypotheses)

- **Per-game wOBA has a low R^2 ceiling.** Even a hypothetical
  ground-truth lineup model cannot exceed `r ~ 0.4` on per-game-side
  wOBA, because the bulk of game variance comes from opposing-pitcher
  quality, BABIP luck, weather, and umpire strike-zone — none of which
  ChemNet sees. We picked `r >= 0.30` as the gate; the model lands at
  `0.09`, which is not a "barely under" miss. It is a fundamental
  prediction failure.
- **Tiny model, tiny training set.** The shipped checkpoint is 36 KB.
  At default hyperparameters (`max_games=2000`, 10 epochs) the GNN sees
  ~4,000 lineup graphs total; that is well below what is needed for a
  GAT to learn meaningful pairwise structure on a 5-feature node space.
- **Training target may not match the inference setup.** The shipped
  checkpoint was trained against a target (`SUM(woba_value)`) using
  rolling features computed at training time; if those rolling features
  were computed differently (different window, different cutoff date),
  the trained model would produce predictions that don't match the
  validation feature distribution, manifesting as low correlation. This
  is plausible but not verifiable without training-run metadata.
- **Adjacency may not encode the right structure.** The model assumes
  protection effects only flow between *adjacent* batting-order slots.
  Real lineup-protection literature is mixed on whether this effect
  exists at all, let alone whether it is strictly nearest-neighbour.

### What is missing for ChemNet to become flagship

1. **Re-train the model on a substantially larger graph cohort** —
   ideally all ~17 K training-window games (both sides), 50+ epochs,
   richer node features (xwOBA, sprint speed, lineup-position-handedness,
   pitcher-side features as additional graph context).
2. **Add an opposing-pitcher representation to the graph** — without it,
   the irreducible variance ceiling caps the achievable correlation
   well below useful.
3. **Re-formulate the synergy claim.** If the GNN cannot beat the
   baseline as a wOBA predictor, the `gnn - baseline` framing is
   structurally not going to work. An alternative is to train the GNN to
   predict the *residual* directly (target = actual - simple_woba_sum),
   which forces it to learn synergy-by-construction.
4. **Validate against an external lineup-protection benchmark** (e.g.
   batter-specific OPS gain when batting in front of a star vs not,
   conditional on opposing pitcher). This is what the literature
   actually measures.

## Caveats

- **Single checkpoint.** This validation evaluates the shipped
  `chemnet_v1.pt`. A re-trained model with the suggestions above may
  pass these gates; this verdict applies only to the current artifact.
- **No training-run metadata.** We cannot confirm exactly which seasons
  / hyperparameters were used. The season-disjoint leakage gate
  (5,010 test game_pks, 17,094 candidate train game_pks, zero overlap)
  rules out the worst-case scenario regardless.
- **Held-out distribution shift.** 2023-2024 is post-shift-ban; the
  league-average wOBA is ~0.01 higher than the 2015-2022 training window
  baseline. The systematic +0.45 wOBA-points residual bias we observe
  is consistent with this drift, and is independent of the synergy
  question.
- **Inference is correct.** The model loads cleanly, runs 0.06 s per
  game-side on GPU, produces non-degenerate predictions in the 6-18
  range, and the per-game CSV is auditable. The pipeline is not
  miswired; the model itself is the limitation.
- **The validation set is large.** n = 9,836 game-sides means the
  bootstrap CIs are tight. The negative result is statistically
  decisive, not a small-sample artifact.

## Artifacts

- `results/validate_chemnet_20260418T204255Z/chemnet_validation_metrics.json`
- `results/validate_chemnet_20260418T204255Z/chemnet_validation_per_game.csv` (9,836 rows)
- `results/validate_chemnet_20260418T204255Z/chemnet_validation_decile_lift.csv` (10 rows)
- `results/validate_chemnet_20260418T204255Z/step_1_validation.log` (full stdout/stderr)
- `results/validate_chemnet_20260418T204255Z/validation_summary.json` (per-skill schema)

---

## v2 architecture iteration -- 2026-04-18T23:15:49Z

**Spec deltas vs v1:** two architecture changes intended to address the v1
diagnostic ("opposing-pitcher graph context" + "residual-prediction
objective"). Code lives alongside v1 in `src/analytics/chemnet.py`; v1
checkpoint is preserved at `models/chemnet_v1.pt`.

### Architecture changes implemented

1. **Opposing-pitcher node + star-overlay adjacency.** Graph grows from 9
   nodes to 10. Pitcher node receives star edges to every lineup slot on
   top of the existing band adjacency (see `_build_adjacency_v2` and
   `build_game_graph_v2` in `src/analytics/chemnet.py`). Pitcher feature
   row is `[rolling_FIP_proxy, K%, BB%, HR_per_9, GB%]`, computed from the
   pitcher's last 30 game appearances and z-normalised within the training
   cohort (mean = `[3.97, 0.215, 0.076, 1.21, 0.45]`, std = `[0.94, 0.054,
   0.025, 0.49, 0.075]`).
2. **Residual-prediction objective.** Both `ChemNetGNNv2` and
   `BaselineMLPv2` are trained on `residual_target = actual_total_wOBA -
   sum_of_parts`, where `sum_of_parts = sum(slot_rolling_wOBA *
   slot_PA_schedule)` with a fixed schedule
   `[4.6, 4.5, 4.4, 4.3, 4.2, 4.1, 4.0, 3.9, 3.8]` (sums to ~37.8 PAs per
   game-side, matching the cohort mean). At inference,
   `pred_absolute = sum_of_parts + pred_residual`; gate metrics evaluate
   the absolute-equivalent predictions for direct apples-to-apples
   comparison with v1.

### Training stats

- Cohort: 17,094 game_pks across 2015-2022; bulk-loaded 4.99M pitch rows,
  built 34,188 candidate graphs (31,453 train / 2,735 internal val).
- Optimiser: Adam, lr=1e-3, batch_size=64, MSE on residual.
- An initial 25-epoch attempt completed 9 epochs before being killed for
  budget; val_gnn MSE plateaued at 21.41-21.47 across all 9 epochs (RMSE
  ~4.63 on the residual scale, identical to the absolute scale --
  sum_of_parts is a weak baseline). The shipped `models/chemnet_v2.pt`
  comes from a 1-epoch eager-save (val_gnn=21.4253, val_base=21.3612);
  additional epochs would not move the gate verdict.

### v1 vs v2 gate comparison (test cohort: 2023-2024, n = 9,836 game-sides)

| Gate | Threshold | v1 | v2 | Verdict |
|---|---|---|---|---|
| 1. GNN Pearson r | `>= 0.30` | 0.0891 [0.07, 0.11] | **0.1554 [0.14, 0.17]** | both FAIL; v2 +0.066 |
| 2. GNN beats baseline (RMSE-pct) | `>= 2%` | -0.14% | **-0.15%** | both FAIL; flat |
| 3. Synergy -> residual (Spearman) | `>= 0.10`, CI lower > 0 | +0.0263 [0.007, 0.046] | **-0.0369 [-0.057, -0.018]** | v2 FAIL with sign flip |
| 4. GNN RMSE | `<= 4.0` | 4.5855 | **4.5308** | both FAIL; v2 -0.05 |
| 5. Leakage | 0 shared `game_pk` | 0 (PASS) | 0 (PASS) | both PASS by construction |
| 6 (info). Top-bottom decile lift | observed | +0.249 wOBA | -0.638 wOBA | sign flip |

### Residual-only diagnostic numbers (v2 only, for completeness)

- `gnn_residual_pred` vs `actual - sum_of_parts`: r = 0.1287
- `baseline_residual_pred` vs `actual - sum_of_parts`: r = 0.1396
- `sum_of_parts` alone vs `actual_total_wOBA`: r = 0.1045

The residual objective adds ~0.05 r over `sum_of_parts` alone but the GNN
remains marginally below the MLP baseline on this scale -- mirroring the
absolute-equivalent picture.

### Verdict

**FAIL -- SECOND CLEAN NEGATIVE.** ChemNet retires from the
flagship-candidate list.

The pitcher node *did* lift correlation by 0.066 r and pull RMSE down by
0.055, both directionally consistent with the v1 diagnostic note. The
residual objective did *not* recover predictive content the GNN couldn't
learn from the absolute target -- the residual MSE is essentially the
same scale as the absolute MSE because the `sum_of_parts` baseline is
itself a weak predictor (slot wOBAs vary by only +/- 0.04 across the 9
slots, well below the 4.75 wOBA-units game-side standard deviation). With
both architectural changes applied, the model still cannot cross any of
the four predictive gates, and the synergy framing now yields a
*negative* rho (top-decile synergy lineups *underperform* their baselines
by 0.64 wOBA, the sign flipped from v1's +0.25). This rules out the
"baseline subtraction is the problem" hypothesis.

### Banner-removal recommendation

**No.** The dashboard's experimental banner on `chemnet_view.py` should
remain; v2's gates fail more decisively than v1's on the synergy claim
(sign-flipped, CI strictly below zero). PM decision welcome.

### Artifacts

- `results/validate_chemnet_v2_20260418T231549Z/chemnet_validation_metrics.json`
- `results/validate_chemnet_v2_20260418T231549Z/chemnet_validation_per_game.csv` (9,836 rows)
- `results/validate_chemnet_v2_20260418T231549Z/chemnet_validation_decile_lift.csv`
- `results/validate_chemnet_v2_20260418T231549Z/step_1_validation.log`
- `results/validate_chemnet_v2_20260418T231549Z/validation_summary.json`
- `models/chemnet_v2.pt` (37 KB; v1 checkpoint preserved alongside)
- `models/chemnet_v2.training_metrics.json` (only present if a full training
  pass completed -- the shipped checkpoint comes from an eager-save during
  a partial run)
