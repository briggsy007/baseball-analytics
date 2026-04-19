# ChemNet Validation Spec

**Source:** `src/analytics/chemnet.py`
(`class ChemNetGNN` at line 395; `class BaselineMLP` at line 444;
`class ChemNetModel(BaseAnalyticsModel)` at line 948.)

## A. Model description

ChemNet is a Graph Attention Network (GAT) lineup-synergy model. For one
team-side of one game it constructs a 9-node graph where each node is a
batting-order slot, edges connect adjacent slots (i, i+1), and node features
are a 5-vector of pre-game rolling 30-game batter stats:

`[rolling_wOBA, K%, BB%, ISO_proxy, normalised_order_position]`.

Two models are trained jointly on the SAME target — `total_wOBA` summed
over the team's plate appearances in that game — using `MSE`:

- `ChemNetGNN`: two GAT layers (5 -> 32x2 heads concat = 64 -> 32x2 heads
  averaged = 32) -> global mean pool -> MLP[32 -> 16 -> 1]. The graph
  structure forces information to flow only between adjacent batting-order
  slots, so any predictive lift over the baseline is attributable to
  pairwise (or higher-order via stacking) batting-order interactions.
- `BaselineMLP`: flattens the same 9x5 = 45 features and fits MLP[45 -> 32
  -> 16 -> 1]. By construction this can use individual slot quality and
  fixed slot positions but cannot reason about pairwise neighbour
  interactions (no graph structure).

**Predicted quantity:** game-side `SUM(woba_value)` (sum of weighted-on-base
contributions across that team-side's PAs in the game). On the wOBA scale,
so units are wOBA-points-summed (typical full game ~10-15 for a 9-inning
batting line).

**Synergy score:** `synergy = gnn_pred - baseline_pred`. Positive synergy
means the GNN, with access to adjacency, predicts more offence than the
baseline — interpreted as a lineup whose batting-order arrangement adds
value over the sum-of-parts. Negative synergy means the baseline outpredicts
the GNN, i.e. the order is *worse* than the parts.

**Inputs.**
- DuckDB `pitches` table: derives lineup composition (first 9 unique
  batter_ids by `MIN(at_bat_number)` per side); rolling 30-game batter
  features computed from prior PAs (`game_date < game_date_of_target`);
  `stand` for edge platoon features; target `SUM(woba_value)` over PAs
  with `woba_denom > 0`.
- The model checkpoint at `models/chemnet_v1.pt` contains a `gnn` and
  `baseline` state_dict pair; ~36 KB total.

**Training.** The default in `train_chemnet()` samples up to 2000 game
seeds across a season range with `np.random.RandomState(42)`, builds both
home and away graphs, and runs `epochs=10`, `batch_size=32`, `lr=1e-3` on
both models in lockstep. The shipped `chemnet_v1.pt` was trained by an
unspecified prior run (no metadata file); for the purposes of this spec we
treat the checkpoint as a fixed artifact and validate it on a held-out
window.

## B. Train/test split

- **Train assumption.** 2015-2022 (mirrors CausalWAR for cross-model
  consistency). The shipped checkpoint pre-dates this spec; we audit the
  HELD-OUT 2023-2024 window for predictive validity, treating any pre-2023
  game as inside the model's possible training distribution. The model has
  no ability to memorise specific 2023-2024 game targets unless the prior
  training set leaked into them — see the leakage gate below.
- **Test.** 2023-2024 game-sides. Eligibility filter: side must have
  `>= 9` distinct batters (so a real starting lineup graph exists) and
  total `woba_denom > 0`.
- **Leakage gate.** The validation script enforces an explicit
  season-disjoint test cohort. Because chemnet inference re-extracts node
  features per call from `pitches` filtered by `game_date < target_date`,
  and the rolling-30 window for a 2023-04-01 target uses 2022-09-XX
  features (which are inside the training distribution), this is a
  forward-looking inference setup, not target leakage. The hard leakage
  audit is: **0 game_pks from the 2023-2024 test cohort appear in the
  2015-2022 reference range.** This is true by construction (season-disjoint
  ranges) and the script asserts it.

## C. Headline gates

The pre-registered gates below are picked to test what the model actually
claims to do, not to be trivially passable. The model targets game-side
total wOBA on a noisy ~10-15 unit scale; per-game R^2 of any honest
lineup model is bounded by the irreducible game-day variance (pitcher
matchup, weather, BABIP luck), and is unlikely to clear 0.4 even with a
ground-truth model. We set thresholds against that ceiling.

### Gate 1 — Out-of-sample predictive correlation (HEADLINE)

- **Threshold:** `Pearson r >= 0.30` between `gnn_pred` and actual
  `SUM(woba_value)` across all qualifying 2023-2024 game-sides.
- **Source field:** `metrics.gnn.pearson_r` in
  `chemnet_validation_metrics.json`.
- **Rationale.** A lineup model that beats r = 0.30 on per-game wOBA is
  doing real work given the noise floor. By comparison the per-game
  baseline (sum of rolling wOBAs * PA estimate) typically lands around
  r = 0.25-0.35 on similar windows in published projection-system audits.
  We require the GNN to clear 0.30 to be considered useful. r < 0.20 is a
  hard FAIL.

### Gate 2 — GNN must beat the baseline (synergy edge claim)

- **Threshold:** `gnn_test_rmse < baseline_test_rmse` AND
  `(baseline_rmse - gnn_rmse) / baseline_rmse >= 0.02` (>= 2% RMSE
  reduction).
- **Source field:** `metrics.gnn.rmse`, `metrics.baseline.rmse`,
  `metrics.synergy_lift.rmse_pct_improvement`.
- **Rationale.** This is the most important honest gate. ChemNet's whole
  thesis is that the graph structure (adjacency between batting-order
  slots) extracts pairwise interaction signal beyond what an
  identical-feature non-graph MLP can. If the GNN does not measurably beat
  the baseline on the held-out window, the synergy score
  `gnn - baseline` is structurally noise: subtracting two equally-good
  estimators of the same target gives a residual whose distribution
  centres on zero with no predictive content. 2% is intentionally
  modest; failing 2% means the architectural choice contributes nothing.

### Gate 3 — Synergy score predicts residual lineup performance

- **Threshold:** Spearman `rho(synergy_score, residual_wOBA) >= 0.10`,
  where `residual_wOBA = actual_total_wOBA - baseline_pred`. CI lower
  bound from a 1000-rep bootstrap must be `> 0` (synergy must explain
  *some* residual variance with statistical confidence).
- **Source field:** `metrics.synergy_residual.spearman_rho`,
  `metrics.synergy_residual.spearman_rho_ci`.
- **Rationale.** The audit doc explicitly frames the test as "synergy
  score predicts next-game wOBA out-of-sample". Operationalised: when
  ChemNet says lineup X has positive synergy, does that lineup actually
  outperform what its individual parts predict? rho >= 0.10 with a CI
  strictly above zero is the minimum for a directionally-correct edge
  signal. rho near zero with a CI straddling zero would mean the
  attention weights and synergy headline number on the dashboard are
  uninterpretable noise.

### Gate 4 — Calibration (RMSE on the wOBA scale)

- **Threshold:** GNN test `RMSE <= 4.0` on the held-out 2023-2024 cohort
  (units: summed wOBA per game-side; typical games range 6-18 with
  league mean ~11). Equivalently the model must explain meaningful
  variance, not merely correlate.
- **Source field:** `metrics.gnn.rmse`.
- **Rationale.** A naive global-mean predictor (predict the train-window
  mean for every game) would land near `RMSE ~ 3.5-4.0` (the
  population standard deviation of game-side total wOBA). The GNN must
  clear that floor to be useful as a point predictor, not just a
  correlator. Without this gate, a model could hit Gate 1's r = 0.30
  while being miscalibrated by a constant offset and still "pass".

### Gate 5 — Leakage audit (hard prerequisite)

- **Threshold:** Zero `game_pk` overlap between the 2015-2022 reference
  range and the 2023-2024 test cohort.
- **Source field:** `leakage_audit.shared_game_pks_train_test`,
  `leakage_audit.train_seasons`, `leakage_audit.test_seasons`.
- **Rationale.** Disjoint by season range, asserted in code. PASS by
  construction; the gate exists so the JSON is auditable.

### Gate 6 (informational, not a hard gate) — Top-decile synergy lift

- **Measured:** mean `actual_total_wOBA - baseline_pred` for game-sides
  in the top 10% of `synergy_score`, vs the bottom 10%. Reported but not
  gated. Useful for the dashboard claim "high-chemistry lineups outperform
  their sum-of-parts by N runs."
- **Source field:** `metrics.synergy_top_decile_lift`,
  `metrics.synergy_bottom_decile_lift`.

## D. Verdict bands

- **FLAGSHIP.** All five hard gates (1-5) PASS, AND Gate 6 shows the
  top-decile synergy cohort outperforms the bottom-decile by `>= 0.5`
  wOBA-units per game (~0.04 wOBA per PA at ~12 PAs/game, a real edge).
- **VALIDATED, NOT FLAGSHIP.** Gates 1, 4, 5 PASS but Gate 2 (synergy
  edge) or Gate 3 (synergy predicts residual) FAILS. Model is a useful
  lineup-wOBA predictor but the *synergy* claim — the dashboard's headline
  story — is not supported and should be removed from the dashboard
  copy.
- **NOT FLAGSHIP / NEEDS WORK.** Gate 1 or Gate 4 FAIL (the model does
  not predict its own training target out-of-sample). Recommend
  re-training with more data and / or richer node features before any
  promotion claim.

## E. Methodology notes

- **Inference cost.** Per-game-side: ~0.06 s once models are warm
  (DB query for lineup + rolling features dominates; GNN forward is
  microseconds). Full 2023-2024 sweep is ~5000 game-sides at < 10 min.
- **Rolling-features warm-up.** Games very early in the 2023 season
  (e.g. 2023-03-30) have node features computed from end-of-2022 PAs.
  This is BY DESIGN — the model's claim is that it can predict 2023-2024
  outcomes given the prior 30 games of context, which necessarily uses
  late-2022 data. We do NOT exclude these; instead we report the
  earliest-test-date metric breakdown for transparency.
- **Pad / sparse lineups.** Game-sides with `< 9` unique batters in the
  starting cohort (rare in 2023-2024 — would require an opener / bullpen
  game scenario) are excluded from the headline metrics but counted in
  `data_quality.dropped_for_short_lineup`.
- **Determinism.** ChemNet is a forward-pass-only inference at validation
  time; no training, no sampling, fully deterministic given the same
  checkpoint and the same DB.

## F. Risk flags

- **Single-game wOBA is a high-variance target.** The R^2 ceiling on a
  per-game-side basis is structurally low; even a ground-truth model
  cannot exceed ~0.4 (literature on Steamer / ZiPS one-game projections).
  Gate 1's r = 0.30 is therefore an honest threshold, not a soft pass.
- **Synergy = GNN - baseline subtracts two correlated estimators.** If
  the GNN and baseline track each other almost perfectly, synergy will be
  numerically tiny and noisy. Gate 2 is the decisive gate here.
- **Order-of-arrival lineup detection.** `_get_lineup_for_game` infers
  the batting order from `MIN(at_bat_number)`. A pinch-hit at slot 7 in
  the 8th inning that ALSO catches its first PA before the original slot 7
  does (impossible in MLB) is the only way this is wrong; it is robust.
- **No park / pitcher-quality control.** ChemNet sees only the 9 batters'
  rolling stats, not the opposing pitcher or the venue. So the same
  lineup against Coors/Kershaw will get the same prediction; actual wOBA
  will differ enormously. This is BY DESIGN (chemnet's edge is supposed
  to be order-only) but it caps Gate 1's achievable correlation.
- **No metadata on training history.** The checkpoint has no
  `training_metadata.json`. We assume 2015-2022 was used (or earlier);
  the season-disjoint leakage gate covers the worst case.

---

Status: Specified 2026-04-18 by `validate-model` flagship-candidate
promotion review. First validation run: see
`docs/models/chemnet_results.md`.
