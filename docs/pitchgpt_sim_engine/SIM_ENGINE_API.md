# PitchGPT Sim Engine — API Specification

**Date:** 2026-04-25
**Status:** formal spec, doc-only. Extends `docs/pitchgpt_sim_engine/RESEARCH_PLAN_sim_engine_api.md` (Plan C).
**Audience:** the implementing agent for Phase 0.5 (`src/analytics/pitchgpt_sim.py`) and the Tier-A consumer agents (A1, A2, A3).
**Implementation file (TBC):** `src/analytics/pitchgpt_sim.py` (not yet created — `EXECUTION_PLAN.md` §6.0.5).
**Style guarantee:** every signature in this doc is a stable contract. Implementation may add private helpers; it MUST NOT alter the public types or semantics without a doc revision.

---

## 1. Purpose & scope

This spec defines the API surface that the PitchGPT-powered simulation engine exposes to its downstream consumers (Tier-A products A1/A2/A3 in `EXECUTION_PLAN.md` §6, plus future Tier-B items B1–B4). The contract decouples three concurrent workstreams: (a) Plan A research that may swap the backbone (PG-v2 vs a wider variant vs a pitcher-embedding variant), (b) Plan B research that picks the outcome predictor (PG-frozen-head, XGBoost, or PA-empirical fallback), and (c) the Tier-A consumer code that grades pitches, projects pitchers, and simulates matchups. With this contract in place, consumers can be scaffolded, validated, and dashboarded against the API while research continues; only the registered backbone or outcome-predictor object swaps when research completes. The API is **model-agnostic by design** — the backbone and outcome predictor are pluggable behind protocols, not hard-wired.

**In scope:** the `rollout()` entry point, the `PAContext` and `RolloutResult` dataclasses, the `OutcomePredictor` protocol with its three concrete implementations, the in-module aggregation utilities, the calibration validity contract, and the checkpoint-discovery + versioning rules.

**Out of scope:** dashboard layouts (each Tier-A dossier owns its UI), per-Tier-A backtest protocols (in dossier gates), per-consumer aggregation (per-game / per-season aggregations live in `pitchgpt_projection.py`, `pitchgpt_matchup.py`, etc.), training of the underlying models (this API consumes trained checkpoints only), and any non-PitchGPT simulation (no full game-state simulator — PA-level only).

---

## 2. Locked context

These are not API parameters; they are constraints on how the API is built and used. Sourced from `EXECUTION_PLAN.md` §3, `pa_outcome_head_design.md` §9, `COORDINATION.md` "Locked decisions," and `NORTH_STAR.md` 2026-04-24 update.

- **Backbone checkpoint.** `models/pitchgpt_v2.pt` — frozen, never overwrite. Discovery rule in §8. The API rejects any backbone load that mutates this file.
- **Outcome predictor.** **LOCKED 2026-04-26 — `PGConcatHeadPredictor` (A1 from Plan B Step 2, see §4.4).** Backed by `models/pitchgpt_v2_outcomehead_a1.pt`. Plan B closed: A1 beat A3 XGBoost by +2.48pp paired (CI [+2.24, +2.72]) on 2025 holdout. WEAKER PASS verdict (clears 4 of 5 PASS gates; misses `in_play_hit < 2.0` due to launch-feature ceiling). The protocol still supports XGBoost / empirical-fallback for backstop and replay; production rollouts use `PGConcatHeadPredictor`.
- **7-class outcome target.** `{ball, called_strike, swinging_strike, foul, in_play_out, in_play_hit, hbp}` indexed 0–6. Locked per `EXECUTION_PLAN.md` §3.1 decision 2; mapping rules at `pa_outcome_head_design.md` §2.2.
- **Calibration is load-bearing.** Post-temperature ECE < 0.02 on the 2025 pitcher-disjoint holdout is the flagship claim. The API never silently emits an uncalibrated probability — `sampling_metadata["calibration_valid"]` is the gate (§6).
- **Allowed claims downstream of this API.** "Calibrated rollout engine," "matches empirical marginals with calibrated uncertainty," "N-sample CIs on every emitted quantity." NOT allowed: "beats LSTM by spec margin on perplexity," "sequence-aware sampling superior to LSTM" (retracted; see `EXECUTION_PLAN.md` §3 and `NORTH_STAR.md` 2026-04-24). Tier-A artifacts that derive from `rollout()` MUST inherit these constraints.
- **ABS-era umpire gate.** Any consumer that conditions on `umpire_scalar` for an edge-product claim is gated on the ABS drift check (`feedback_no_umpire_edges_until_abs_drift_check.md`). The API itself accepts the scalar; the gate lives in consumer code.
- **PitchGPT model constants.** `VOCAB_SIZE = 2210` (= 17 pitch_types × 26 zones × 5 velo buckets), `PAD_TOKEN = 2210`, `TOTAL_VOCAB = 2212` (PAD + END), `CONTEXT_DIM = 35` (34 categorical one-hots + 1 ump scalar) for v2, `CONTEXT_DIM = 34` for v1 legacy. These are referenced from `src/analytics/pitchgpt.py`; the API does NOT redefine them.

---

## 3. Primary API

### 3.1 `rollout()`

```python
def rollout(
    starting_context: PAContext,
    *,
    backbone_version: str = "v2",
    outcome_predictor: OutcomePredictor | None = None,
    n_samples: int = 100,
    horizon: int = 6,
    temperature: float = 1.0,
    seed: int | None = None,
    return_probs: bool = True,
) -> RolloutResult:
    ...
```

**Semantics.** Sample `n_samples` independent autoregressive rollouts from the PA's starting state, each up to `horizon` pitch positions. At each position: (a) the backbone produces a 2210-class softmax over the next pitch token conditioned on the prefix tokens + the (possibly time-varying) context vector; (b) a token is sampled with `temperature`; (c) if `outcome_predictor is not None`, the predictor produces a 7-class outcome distribution and a sample; (d) PA-termination logic checks whether the rollout has terminated (see §3.3).

**Edge cases.**
- `n_samples == 0` raises `ValueError`. Zero samples is meaningless.
- `horizon == 0` returns an empty `RolloutResult` with all arrays shape `(n_samples, 0)` — useful only for testing.
- `horizon > 12` triggers a WARNING. Exposure-bias drift is unmeasured beyond H=6 (see `EXECUTION_PLAN.md` §7.1) and the calibration-valid flag is forced to False.
- `temperature != 1.0` triggers a WARNING and forces `calibration_valid = False`. Calibration is fit at T=1.0; sampling at any other T invalidates it.
- `seed is None` uses `np.random.default_rng()` non-deterministically. Pass a `seed: int` for reproducibility.
- `starting_context.prefix_pitch_tokens` longer than `horizon` raises `ValueError` (the prefix already consumes the budget).
- A starting context whose feature values fall outside the calibration cohort's 1st–99th percentile band on any feature triggers `calibration_valid = False` (§6).

### 3.2 `PAContext` dataclass

Captures the PA's starting state — the inputs to the first rollout step, before any sampling. Field types match the source columns in `pitches` schema (`src/db/schema.py` lines 168–179) where applicable.

```python
@dataclass(frozen=True)
class PAContext:
    pitcher_id: int
    batter_id: int
    count: tuple[int, int]
    outs: int
    runners: tuple[bool, bool, bool]
    batter_stand: Literal["L", "R"]
    pitcher_throws: Literal["L", "R"]
    inning: int
    inning_topbot: Literal["Top", "Bot"]
    score_diff: int
    umpire_scalar: float = 0.0
    prefix_pitch_tokens: tuple[int, ...] = ()
```

| field | type | source column | valid range | semantics |
|---|---|---|---|---|
| `pitcher_id` | `int` | `pitches.pitcher_id` | MLBAM positive int | NOT consumed by v2 backbone forward pass; carried for output joining and for `OutcomePredictor` variants that condition on pitcher (XGBoost). |
| `batter_id` | `int` | `pitches.batter` | MLBAM positive int | Same as `pitcher_id`. |
| `count` | `tuple[int, int]` | `pitches.balls`, `pitches.strikes` | balls ∈ [0,3], strikes ∈ [0,2] | `(balls, strikes)`. The v2 `encode_context` collapses this to `count_state = balls*3 + min(strikes, 2)` ∈ [0, 11]; values outside the valid range raise `ValueError`. |
| `outs` | `int` | `pitches.outs_when_up` | [0, 2] | Standard. |
| `runners` | `tuple[bool, bool, bool]` | `pitches.on_1b/on_2b/on_3b` | each ∈ {True, False} | `(on_1b, on_2b, on_3b)`. Encoded to `runner_state = 4*on_1b + 2*on_2b + on_3b` ∈ [0, 7]. |
| `batter_stand` | `Literal["L", "R"]` | `pitches.stand` | "L" or "R" | Switch-hitter resolution is the caller's responsibility (use the at-bat's resolved stand). |
| `pitcher_throws` | `Literal["L", "R"]` | `pitches.p_throws` | "L" or "R" | NOT directly consumed by v2 backbone (no separate input); kept for symmetry with the L/R matchup convention and for future predictors. |
| `inning` | `int` | `pitches.inning` | [1, 30] | Bucketed by `encode_context` into `{1–3, 4–6, 7–9, 10+}`. |
| `inning_topbot` | `Literal["Top", "Bot"]` | `pitches.inning_topbot` | "Top" or "Bot" | Carried for output joining; v2 backbone does not directly consume. |
| `score_diff` | `int` | derived from `pitches` running score | [-30, 30] practical | Pitcher's POV. Bucketed by `encode_context` to 5 levels. |
| `umpire_scalar` | `float` | `umpire_tendencies.accuracy_above_x_wmean` for the game's HP umpire | typically [-0.05, 0.05] | Continuous. NULL/missing must be resolved to season-league-median upstream; default 0.0 means "neutral / unknown ump." |
| `prefix_pitch_tokens` | `tuple[int, ...]` | derived from prior pitches in the PA | each ∈ [0, 2209] | Pitches already thrown in the PA. Empty for a 0-0 query; partial for a counterfactual mid-PA rollout (e.g., A1's "what if pitch 3 had been different" — caller passes pitches 1–2 as prefix and re-rolls from position 3). PAD/END tokens (2210, 2211) are forbidden in the prefix and raise `ValueError`. |

**Round-trip helper.** A factory `PAContext.from_pitches_row(row, ump_scalar=...)` is provided for convenience but is NOT part of the rollout contract.

### 3.3 `RolloutResult` dataclass

Captures the full rollout output. Every array field has a fixed shape and dtype; pad-token conventions are explicit.

```python
@dataclass(frozen=True)
class RolloutResult:
    pitch_tokens: np.ndarray           # (n_samples, horizon), int64
    pitch_probs: np.ndarray | None     # (n_samples, horizon, 2210), float32
    outcomes: np.ndarray | None        # (n_samples, horizon), int64
    outcome_probs: np.ndarray | None   # (n_samples, horizon, 7), float32
    pa_terminated: np.ndarray          # (n_samples, horizon), bool
    pa_outcome: np.ndarray | None      # (n_samples,), int64
    final_count: np.ndarray            # (n_samples, 2), int64 (balls, strikes)
    sampling_metadata: dict
```

| field | shape | dtype | None when | pad convention | semantics |
|---|---|---|---|---|---|
| `pitch_tokens` | `(n_samples, horizon)` | int64 | never | `ROLLOUT_PAD_PITCH = 2210` (PAD_TOKEN) for positions past PA termination | Sampled tokens. Always present. |
| `pitch_probs` | `(n_samples, horizon, 2210)` | float32 | when `return_probs=False` | rows past termination are filled with NaN (NOT zeros — silent zeros would propagate to KL/Wasserstein aggregates) | Per-position, post-temperature, calibrated next-token probabilities. Memory cost: `n_samples * horizon * 2210 * 4 bytes`; for `n_samples=10_000, horizon=6` ≈ 530 MB. Use `return_probs=False` when only token marginals are needed. |
| `outcomes` | `(n_samples, horizon)` | int64 | `outcome_predictor is None` | `ROLLOUT_PAD_OUTCOME = 7` for positions past PA termination | Sampled 7-class outcomes (0–6). |
| `outcome_probs` | `(n_samples, horizon, 7)` | float32 | `outcome_predictor is None` | NaN past termination | Per-position post-temperature outcome distribution. |
| `pa_terminated` | `(n_samples, horizon)` | bool | never | False past termination, True at the terminating position, False before | Marks the position at which each PA ended. Exactly one True per sample if termination occurred; all False otherwise (truncated). |
| `pa_outcome` | `(n_samples,)` | int64 | `outcome_predictor is None` | `ROLLOUT_PAD_OUTCOME = 7` for samples that did not terminate within horizon | The terminal-position 7-class outcome for each sample. |
| `final_count` | `(n_samples, 2)` | int64 | never | clamped to (4, 3) at termination | `(balls, strikes)` at the terminating position; (4, X) means walk, (X, 3) means K. Available even when `outcome_predictor is None`. |
| `sampling_metadata` | dict | — | never | — | See §3.4. |

**PA-termination logic** (applied per-sample, position-by-position; first match wins):

1. **Outcome-driven termination** (only when `outcome_predictor is not None`): if the sampled `outcome ∈ {in_play_out, in_play_hit, hbp}` (indices 4, 5, 6), the PA terminates at this position.
2. **Walk** (always evaluated): the running `(balls, strikes)` updated by the sampled token + outcome reaches `balls == 4`. With outcome predictor present, "ball" outcomes increment the ball counter; without, a heuristic mapping from token zone is used (see §4 `EmpiricalPATerminalLookup` notes — fallback is approximate).
3. **Strikeout** (always evaluated): running `strikes` reaches 3. With outcome predictor present, swinging-strike, called-strike, and (on 0–1 strikes) foul outcomes increment strikes per MLB rules; foul on 2 strikes does NOT advance to 3. Without outcome predictor, the heuristic-from-zone fallback applies and must be interpreted with caution.
4. **Horizon exhaustion** (always evaluated): if no termination by position `horizon - 1`, the sample is marked truncated (`pa_terminated` all False, `pa_outcome = ROLLOUT_PAD_OUTCOME`).

After termination, all subsequent positions for that sample are padded per the table above. The `pad-with-NaN` convention on `pitch_probs` and `outcome_probs` is non-negotiable: aggregation utilities (§5) rely on it to correctly skip truncated positions via NaN-masking rather than risk a silent zero biasing KLs and Wassersteins toward zero.

### 3.4 `sampling_metadata` schema

Free-form `dict` with these required keys:

```python
{
    "temperature": float,
    "seed": int | None,
    "n_samples": int,
    "horizon": int,
    "backbone_version": str,                    # e.g., "v2"
    "backbone_checkpoint_sha256": str,          # hex digest of the loaded .pt file
    "outcome_predictor": str,                   # "none" | "pg_concat_head" | "pg_frozen_head" | "xgboost" | "empirical_pa_terminal"
    "outcome_predictor_checkpoint_sha256": str | None,
    "calibration_valid": bool,                  # see §6
    "calibration_invalid_reasons": list[str],   # empty if valid; see §6 for enumerated reasons
    "rollout_engine_version": str,              # semantic version of pitchgpt_sim.py
    "n_truncated": int,                         # how many of the n_samples did not terminate within horizon
}
```

Implementations MAY add keys but MUST NOT remove or rename these. Consumers MAY rely on any required key being present.

---

## 4. `OutcomePredictor` protocol

The plug-in interface. Any object implementing this protocol can be passed as `outcome_predictor` to `rollout()`.

```python
@runtime_checkable
class OutcomePredictor(Protocol):
    name: str  # "pg_concat_head" | "pg_frozen_head" | "xgboost" | "empirical_pa_terminal"
    checkpoint_sha256: str
    calibration: dict  # loaded from calibration.json (see §7)

    def predict_outcome_probs(
        self,
        backbone_hidden: torch.Tensor,   # (B, S, d_model=128) — the backbone's per-position hidden state
        context_vec: torch.Tensor,       # (B, S, CONTEXT_DIM=35)
        pitch_token: torch.Tensor,       # (B, S) — the just-sampled pitch token for each position
    ) -> torch.Tensor:                   # (B, S, 7) — calibrated, post-temperature probabilities (sum=1)
        ...
```

Implementations MUST:
- Apply their own internal temperature scaling before returning.
- Return probabilities (rows sum to 1 along the last dim), NOT logits.
- Be deterministic given fixed inputs (no internal sampling — sampling happens in the rollout loop).
- Carry `calibration` metadata loaded from a `calibration.json` adjacent to the checkpoint (§7).

Implementations MAY:
- Ignore any of the three input tensors. The protocol takes all three for uniformity.
- Run on CPU or GPU; the rollout loop adapts to the predictor's device.

**`in_play_hit` ceiling (load-bearing disclosure).** Any predictor that does not condition on post-pitch `launch_speed` / `launch_angle` inherits a structural ceiling at `in_play_hit` log-loss ≈ 2.3 — hit-vs-out is decided by exit velocity and launch angle, neither of which is observable at pitch-decision time. Plan B's best variant (A1) reaches 2.34; the kill-criterion fallback A5 sits at 2.86. **Tier-A consumers (A1 grades, A2 projections, A3 matchup sims) MUST disclose this ceiling in any wOBA / PA-outcome aggregation derived from `rollout()`.** The disclosure is non-negotiable per the Plan B WEAKER PASS verdict (see `EXECUTION_PLAN.md` §6.0.3 UPDATE 2026-04-26).

### 4.1 `PGFrozenHeadPredictor` — **DEPRECATED 2026-04-26**

**Status: deprecated.** Phase 0.3 baseline that lost to A1 (`PGConcatHeadPredictor`) by +2.48pp paired bootstrap (CI [+2.24, +2.72]) on the 2025 pitcher-disjoint holdout. Kept in this API for replay-ability of the Phase 0.3 −5.34% FAIL artifact, NOT production. Production is §4.4 `PGConcatHeadPredictor`.

The Phase 0.3 / Plan B candidate (`src/analytics/pitchgpt_outcome_head.py::FrozenOutcomeHead`).

- **Inputs used:** `backbone_hidden` (the only signal). `context_vec` and `pitch_token` are ignored.
- **Inputs ignored:** `context_vec`, `pitch_token`. (See `pa_outcome_head_design.md` §7.4 for why this matters for token-conditioned counterfactuals — this predictor does NOT produce `p(outcome | token, prefix)`, only `p(outcome | prefix)`.)
- **Checkpoint location:** `models/pitchgpt_v2_outcomehead.pt`.
- **Calibration JSON path:** `models/pitchgpt_v2_outcomehead_calibration.json`.
- **Status (2026-04-25):** Phase 0.3 PARTIAL FAIL. Log-loss is below frequency prior at 10K scale. Plan B candidates A1 (concat-context-and-pitch-token) or A6 (HBP-vs-rest two-stage) are the live alternatives; if either passes, the Plan B winner re-uses this class with a different backbone-head config or a sibling class.
- **Calibration JSON required keys:** see §7. `T` is the temperature scalar; the head outputs raw logits that the predictor divides by `T` before softmax.

### 4.2 `XGBoostOutcomePredictor`

Plan B's A3 candidate — engineered features into a tabular gradient-boosted-trees classifier (`RESEARCH_PLAN_outcome_prediction.md` §3 hypothesis A3).

- **Inputs used:** `context_vec` (the 35-dim vector — count, outs, runners, hands, inning bucket, score bucket, ump scalar) AND `pitch_token` (decomposed back into pitch_type / zone / velocity_bucket via the canonical `(token // (NUM_ZONES * NUM_VELO_BUCKETS), ...)` decomposition). Plus `pitcher_id` and `batter_id` from the surrounding `PAContext` (passed via the rollout-loop closure; the predictor receives them from the API at construction time by way of a per-call `set_ids(pitcher_id, batter_id)` setter — implementation note, not protocol-level).
- **Inputs ignored:** `backbone_hidden`. The PG backbone hidden state adds no value beyond engineered features for this predictor (see `RESEARCH_PLAN_outcome_prediction.md` §2.1).
- **Checkpoint location:** `models/pitchgpt_outcome_xgb.bin` (XGBoost `Booster.save_model` format).
- **Calibration JSON path:** `models/pitchgpt_outcome_xgb_calibration.json`.
- **Calibration JSON required keys:** §7. XGBoost's `multi:softprob` is poorly calibrated by default; temperature scaling on the val slice is mandatory.

### 4.3 `EmpiricalPATerminalLookup`

Plan B's §7.1 fallback — non-parametric lookup. Fires only if the kill criterion in `RESEARCH_PLAN_outcome_prediction.md` §7 triggers (no architecture clears +5% lift).

- **Inputs used:** `pitch_token` (decomposed) and the `(balls, strikes, batter_stand)` derived from `context_vec` and rollout state.
- **Inputs ignored:** `backbone_hidden`. The lookup is a frequency table — no learned representation enters.
- **Lookup table location:** `models/pitchgpt_outcome_empirical_lookup.parquet` — a Parquet file keyed on `(pitch_type, zone, velocity_bucket, count_state, batter_stand)` with smoothed empirical 7-class outcome distributions. Built from the 2015–2022 cohort with Dirichlet smoothing α=1; falls back through the (count_state, pitch_type, batter_stand) marginal then the global frequency prior for unseen buckets.
- **Calibration JSON path:** `models/pitchgpt_outcome_empirical_lookup_calibration.json` (degenerate — `T=1.0` always, since the table is empirical-by-construction).
- **Calibration JSON required keys:** §7. ECE_pre and ECE_post are equal (no temperature fit needed); `holdout_n_pitches` and `holdout_season` document the holdout the table was *measured against*, even though no fitting was done.

### 4.4 `PGConcatHeadPredictor` — **PRODUCTION (Plan B winner, 2026-04-26)**

Plan B Step 2 winner per `RESEARCH_PLAN_outcome_prediction.md` §7 ship-criterion. Backs the production sim engine. Beats all other Plan B variants (A3 XGBoost, A4 logistic, A5 empirical) on log-loss, ECE, HBP discrimination, and per-pitcher stability.

- **Inputs used:** `concat(backbone_hidden[128] + context_vec[35] + pitch_type_oh[17] + zone_oh[26] + velocity_oh[5])` = 211d. The pitch-token one-hots are decomposed from the just-sampled `pitch_token` via the canonical decomposition (token decomposes to pitch_type ∈ [0,16], zone ∈ [0,25], velocity_bucket ∈ [0,4]).
- **Inputs ignored:** none — every protocol input is consumed.
- **Architecture:** 3-layer MLP `211 → 128 → 64 → 7` with ReLU + dropout 0.1. Trained with weighted CE (inverse-frequency class weights capped at 10×) for 5 epochs (best at epoch 3); training wall-clock ~2 min on RTX 3050.
- **Checkpoint location:** `models/pitchgpt_v2_outcomehead_a1.pt` (~38 KB, 28K head params).
- **Calibration JSON path:** `models/pitchgpt_v2_outcomehead_a1_calibration.json`. Temperature scaled on 2023 pitcher-disjoint validation slice; T = 0.8003. ECE post-T on 2025 pitcher-disjoint holdout (204,513 rows): **0.0114** (well under the 0.05 outcome-predictor budget).
- **2025 holdout headline metrics:** log-loss 1.3507; lift vs frequency prior **+18.31%** (CI [+18.10%, +18.53%]); top-1 accuracy 0.4671; HBP per-class log-loss **3.02** (first variant under <4.0 PASS); `in_play_hit` per-class log-loss 2.34 (misses full <2.0 PASS, clears WEAKER <2.5).
- **Verdict:** WEAKER PASS — clears 4 of 5 PASS gates. Misses `in_play_hit` < 2.0 because hit-vs-out depends on launch_speed/launch_angle (post-pitch features). See the in_play_hit ceiling note in §4 — Tier-A consumers MUST disclose.
- **A1 vs A3 paired bootstrap (204,513-row intersection):** A1 − A3 lift delta +2.48 pp (CI [+2.24, +2.72]) — clears the +1pp ship threshold by ~2×.
- **Per-pitcher stability:** top-50 most-frequent test pitchers, mean log-loss 1.346, var 0.0010, range [1.27, 1.40]. Best in study; tightens upper end vs A3's 1.91 outlier.
- **Backbone integrity:** verified — `models/pitchgpt_v2.pt` SHA256 unchanged pre/post training.
- **Calibration JSON required keys:** §7. `predictor_kind = "pg_concat_head"`. `T = 0.8003`. `ECE_post = 0.0114`. `holdout_season = 2025`.
- **Artifacts:** `results/pitchgpt_sim/outcome_baselines_2026_04_25/SUMMARY.md`, `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/{metrics.json, report.md, train.log}`. Training script: `scripts/pitchgpt_outcome_a1_concat.py`.

### 4.5 Behavior when `outcome_predictor=None`

Fully specified graceful-degrade contract. When `rollout()` is invoked with `outcome_predictor=None`:

- `outcomes = None`, `outcome_probs = None`, `pa_outcome = None` in the returned `RolloutResult`.
- PA-termination logic falls back to **count-only**: the running ball/strike counter advances by a heuristic mapping from the sampled token's zone:
  - Zone in {0, 1, 2, 3} (the in-zone 4 corners and middle): heuristic "called strike if not swung at" — counts as +1 strike on the running counter.
  - Zone outside the strike zone (zones 11–13 and the synthetic out-of-zone slot): counts as +1 ball.
  - This is **explicitly approximate** — without an outcome head we cannot distinguish ball / called-strike / swinging-strike / foul / in-play, so K%/BB% from a None-predictor rollout will be biased high (every in-play and foul is misclassified as a strike).
- Walk and strikeout terminations still fire on the heuristic counters; in-play and HBP terminations are unavailable, so PAs end only on count-driven outcomes or horizon exhaustion.
- `sampling_metadata["outcome_predictor"] = "none"`. Consumers MUST check this and either degrade their own emission (e.g., A1 reverts to a token-marginal-percentile rather than wOBA-percentile) or raise. They MUST NOT silently emit outcome aggregations from missing data.
- `calibration_valid` is independent of the outcome predictor's presence; the backbone's calibration is what matters for the pitch-token side. `calibration_valid` may still be True even with `outcome_predictor=None`.

This contract is what makes Plan B's WEAKER-PASS or FAIL outcomes safe to ship — products fall back to "pitch-token-only rollouts" with explicit consumer-side disclosure rather than silently going wrong.

---

## 5. Aggregation utilities (live in `pitchgpt_sim.py`, NOT in consumers)

Per-rollout and per-PA reductions belong here. Per-pitcher, per-game, per-season aggregations live in consumer modules (`pitchgpt_projection.py`, `pitchgpt_matchup.py`, etc.).

### 5.1 `pa_woba_distribution`

```python
def pa_woba_distribution(
    rollout: RolloutResult,
    woba_lookup: WObaTable | None = None,
) -> np.ndarray:  # (n_samples,), float32
    ...
```

For each of the `n_samples` rollouts, compute the per-rollout PA-level wOBA contribution:

- If the rollout terminated AND `rollout.outcomes` is not None: look up wOBA in `woba_lookup` keyed on `(pa_outcome[i], terminal_pitch_token[i])`. `WObaTable` is a thin wrapper over the empirical 2015–2022 wOBA-per-(7class-outcome × pitch-type) Statcast values; default lookup is provided via `WObaTable.default()`.
- If the rollout terminated BUT `rollout.outcomes is None`: fall back to PA-empirical lookup (per `RESEARCH_PLAN_outcome_prediction.md` §7.1) keyed on `(terminal_pitch_token, final_count)`. This path produces noisier wOBA estimates and is flagged via the metadata field `pa_woba_distribution_used_empirical_fallback = True` on the returned ndarray's `__array_metadata__` attribute (numpy doesn't natively support this — implementation may switch to a small wrapper dataclass; the contract is "the caller can detect fallback was used").
- If the rollout did NOT terminate within horizon: emit `np.nan` at that index. **Silent zero would bias aggregates downward**; consumers must NaN-mask.

**Returns** shape `(n_samples,)`, float32. Truncated samples are NaN; consumers compute `np.nanmean / np.nanpercentile` to aggregate.

### 5.2 `percentile_of_actual_outcome`

Used by A1 (counterfactual pitch-call grade).

```python
def percentile_of_actual_outcome(
    rollout: RolloutResult,
    actual_outcome: int,                    # 0..6
    actual_pitch_token: int | None = None,  # 0..2209 if known
    woba_lookup: WObaTable | None = None,
) -> float:
    ...
```

Returns the percentile rank of the actual pitch's wOBA outcome within the rollout's wOBA distribution. Low percentile = pitcher gave up MORE than expected given the model's best rollouts (worse call); high percentile = pitcher gave up LESS than expected (better call).

**Edge cases.**
- If `rollout.outcomes is None`: percentile is computed against PA-empirical wOBA (§5.1's fallback path). The return value's metadata flags this so the consumer can disclose "empirical-fallback percentile" in artifacts.
- If `actual_pitch_token` is provided and differs from the modal sampled-prefix token, log a WARNING. The rollout was conditioned on the prefix WITHOUT the actual pitch — this IS the counterfactual setup, not an error. The warning exists so consumers don't confuse the modes.
- If all `n_samples` samples are truncated, return `np.nan`.

### 5.3 `outcome_marginal`

```python
def outcome_marginal(
    rollout: RolloutResult,
    outcome_class: int | None = None,  # 0..6, or None for full vector
) -> np.ndarray:
    ...
```

Frequency-aggregation over the `n_samples` rollouts.

- If `outcome_class` is provided: returns shape `(horizon,)` — per-position frequency of that class across the `n_samples` rollouts, NaN-masked over truncated positions.
- If `outcome_class is None`: returns shape `(horizon, 7)` — full per-position 7-class marginal.

**NaN handling.** Truncated positions (where `pa_terminated` is False past a sample's actual termination) are excluded via NaN-masking on `outcome_probs` before averaging; the returned values are always in [0, 1] regardless of how many samples truncated at each position.

**Use case.** A2 projection aggregation (per-position K% over a 25-PA start), B1 live-game WP construction (per-PA outcome distribution over rest of inning).

### 5.4 `pitch_token_marginal`

```python
def pitch_token_marginal(
    rollout: RolloutResult,
    position: int | None = None,    # 0..horizon-1, or None for full grid
) -> np.ndarray:
    ...
```

Frequency-aggregation over sampled pitch tokens.

- If `position` is provided: shape `(2210,)` — frequency of each pitch token at that position.
- If `position is None`: shape `(horizon, 2210)`.

Critical for sampling-fidelity downstream checks: the H=1 marginal MUST match the next-token softmax distribution as a regression check (`EXECUTION_PLAN.md` §6.0.5 success criterion).

**NaN handling.** Same pattern — truncated positions excluded via NaN mask on `pitch_probs`. Returned values are in [0, 1].

---

## 6. Calibration validity contract

`sampling_metadata["calibration_valid"]` is True iff ALL FOUR conditions hold (per `RESEARCH_PLAN_sim_engine_api.md` §3.5). When False, downstream artifacts MUST flag "uncalibrated rollout" — this is non-overrideable.

1. **Temperature is exactly 1.0.** Sampling at the calibrated temperature is the only regime under which post-temperature ECE measurements transfer.
2. **Backbone's calibration.json is current.** The backbone checkpoint's `calibration.json` lists `holdout_season`; the rollout's `starting_context.inning` and `score_diff` ranges are within the calibration cohort. If the rollout is for a season strictly later than the calibration's `holdout_season + 1` (e.g., calibration is on 2025, rollout context is 2027), `calibration_valid = False` until re-measured.
3. **Outcome predictor's calibration.json is current.** Same rule, applied to the predictor (if present). With `outcome_predictor=None`, this clause is vacuously satisfied.
4. **Starting context is in-distribution.** Per-feature percentile gate: for each numeric feature in `(count_state, outs, score_diff, inning_bucket)`, the starting context's value must fall within the [1st, 99th] percentile band of the calibration cohort's empirical CDF. The calibration cohort's per-feature CDFs are stored alongside the backbone's `calibration.json` as `calibration_feature_cdfs.npz` (see §7 schema).

**`calibration_invalid_reasons`** enumerates which conditions failed, drawn from this set:
- `"temperature_not_unity"`
- `"horizon_exceeds_validated"`
- `"backbone_calibration_stale"`
- `"outcome_predictor_calibration_stale"`
- `"context_count_state_out_of_band"`
- `"context_outs_out_of_band"`
- `"context_score_diff_out_of_band"`
- `"context_inning_out_of_band"`
- `"context_runner_state_unseen"` (the 8-state runner combination not present in calibration cohort — exceedingly rare)

Multiple reasons accumulate; the list is empty iff `calibration_valid is True`.

---

## 7. Calibration re-check protocol

When and how to re-measure calibration. The API enforces these contracts; consumers don't get a vote.

### 7.1 When to re-measure

- **Annually.** Every January (against the most-recently-completed full season). Procedure §7.2. If the fitted T differs from the stored T by > ±0.05, log WARNING and update; if by > ±0.15, the API HALTS rollout emission until the change is reviewed and a new `calibration.json` is written.
- **On context drift.** Detected per-call via condition 4 in §6. Per-call only; no re-fit.
- **On checkpoint change.** Any new backbone or outcome-predictor checkpoint MUST ship with a `calibration.json` written by §7.2 BEFORE it is registered (§8). The API rejects checkpoints lacking valid `calibration.json` files.

### 7.2 Re-measurement procedure (6-step)

1. Load the checkpoint to be calibrated (backbone or outcome-predictor).
2. Score it on the most-recent full-season pitcher-disjoint holdout cohort (e.g., 2025 if the current year is 2026). Use the same cohort builder pattern as `scripts/pitchgpt_2025_holdout.py` so results are comparable.
3. Fit temperature via LBFGS on a 2-year-prior pitcher-disjoint validation slice (e.g., 2023 if calibrating against 2025). Reuse `src/analytics/pitchgpt_calibration.py::temperature_scale`.
4. Measure 10-bin ECE pre and post temperature on the holdout.
5. Compute the calibration-feature CDFs over the same holdout: `count_state`, `outs`, `score_diff`, `inning_bucket` (and `runner_state` for the categorical case). Save as `calibration_feature_cdfs.npz`.
6. Write `calibration.json` (schema below) adjacent to the checkpoint. Sim engine reads this on checkpoint load; if missing or `fit_date` older than 12 months, the engine refuses to use the checkpoint.

### 7.3 `calibration.json` schema

```json
{
  "T": 1.0234,
  "ECE_pre": 0.0203,
  "ECE_post": 0.0098,
  "holdout_season": 2025,
  "holdout_n_pitches": 412877,
  "fit_date": "2026-01-15",
  "checkpoint_sha256": "1a2b3c4d...",
  "predictor_kind": "backbone",
  "class_calibration": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
}
```

- `T`: temperature scalar to divide logits by before softmax.
- `ECE_pre`: 10-bin pre-temperature expected calibration error on the holdout.
- `ECE_post`: 10-bin post-temperature ECE. MUST be < 0.02 for the backbone (load-bearing claim); MUST be < 0.05 for any outcome predictor (per `EXECUTION_PLAN.md` §6.0.3 outcome-head success criterion). Out-of-budget calibration files cause the API to refuse the checkpoint.
- `holdout_season`: the year used for the post-temp ECE measurement (drives the staleness check in §6 condition 2/3).
- `holdout_n_pitches`: pitch-level n; for outcome predictors, this is also the n outcome labels.
- `fit_date`: ISO-8601 date of the fit. > 12 months stale is auto-rejected.
- `checkpoint_sha256`: hex digest of the checkpoint binary, recomputed at load time and asserted to match.
- `predictor_kind`: `"backbone"` | `"pg_concat_head"` | `"pg_frozen_head"` | `"xgboost"` | `"empirical_pa_terminal"`.
- `class_calibration` *(optional, outcome predictors only, added Phase 0.6 2026-04-26)*: length-7 list of positive floats indexed by `OUTCOME_CLASSES = (ball, called_strike, swinging_strike, foul, in_play_out, in_play_hit, hbp)`. Applied AFTER temperature scaling and softmax, BEFORE return: `p_i ← p_i * w_i / sum_j(p_j * w_j)`. This is a post-hoc per-class probability re-weighting that closes class-marginal bias (PHASE_0.6_DIAGNOSIS.md). Does not affect top-1 ECE (top-1 reliability is independent of class-marginal balance). Missing or `null` ⇒ identity (no re-weighting), backwards-compatible.

**Refusal behavior.** If `calibration.json` is missing, has a stale `fit_date`, has an out-of-budget `ECE_post`, has a mismatched `checkpoint_sha256`, or is missing `calibration_feature_cdfs.npz`, the API raises `CalibrationError` at checkpoint-load time. `class_calibration`, when present, must be length-7 with all-positive finite floats (else `CalibrationError`). There is no override flag; bad calibration must be corrected upstream.

---

## 8. Versioning & checkpoint discovery

### 8.1 Backbone discovery

Path rule: `models/pitchgpt_<version>.pt`.
- `version="v1"` → `models/pitchgpt_v1.pt` (legacy, CONTEXT_DIM=34, no umpire scalar).
- `version="v2"` → `models/pitchgpt_v2.pt` (current flagship, CONTEXT_DIM=35).
- `version="v2_10k"` → `models/pitchgpt_v1_10k.pt` (matched-scale retrain artifact; preserved for backtest replication).
- Future versions follow the same pattern.

The loader infers `CONTEXT_DIM` from the checkpoint's `state_dict` shape (`context_proj.weight.shape[1]`), so the API does not need to know in advance. v1 checkpoints are loaded with the `umpire_scalar` field of `PAContext` ignored (no error; warning if non-zero is supplied).

### 8.2 Outcome-predictor registry

Discovery is via factory, not path-rule, because the three concrete predictors have different file formats (`.pt`, `.bin`, `.parquet`).

```python
class OutcomePredictorRegistry:
    @classmethod
    def get(cls, name: str) -> OutcomePredictor: ...

    @classmethod
    def register(cls, name: str, factory: Callable[[], OutcomePredictor]) -> None: ...

    @classmethod
    def list_registered(cls) -> list[str]: ...
```

Default registrations (loaded at module import):
- `"pg_concat_head"` → `PGConcatHeadPredictor()` — checkpoint `models/pitchgpt_v2_outcomehead_a1.pt`. **PRODUCTION (Plan B winner).**
- `"pg_frozen_head"` → `PGFrozenHeadPredictor()` — checkpoint `models/pitchgpt_v2_outcomehead.pt`. **DEPRECATED**; preserved for replay only.
- `"xgboost"` → `XGBoostOutcomePredictor()` — checkpoint `models/pitchgpt_outcome_xgb.bin`. Backstop, not production.
- `"empirical_pa_terminal"` → `EmpiricalPATerminalLookup()` — table `models/pitchgpt_outcome_empirical_lookup.parquet`. Kill-criterion safe-harbor only.

`OutcomePredictorRegistry.get("pg_concat_head")` returns the registered predictor or raises `KeyError` if the checkpoint is missing or its `calibration.json` is invalid.

### 8.3 Adding a new outcome predictor (4-step)

1. **Implement the protocol.** New class in `src/analytics/pitchgpt_outcome_predictors/<name>.py` exposing `predict_outcome_probs`, `name`, `checkpoint_sha256`, `calibration`.
2. **Calibrate.** Run §7.2's 6-step on the new predictor's checkpoint. ECE_post must be in budget.
3. **Write `calibration.json` and `calibration_feature_cdfs.npz`** adjacent to the checkpoint.
4. **Register at module import** via `OutcomePredictorRegistry.register("<name>", lambda: <Class>(checkpoint_path=...))`.

After step 4, consumers can pass `outcome_predictor=OutcomePredictorRegistry.get("<name>")` to `rollout()` with no other code changes.

---

## 9. Validation gates the API itself must pass

The API has its own gates separate from the Tier-A consumer gates. These are validation gates on the rollout engine, not on edge claims that consume it.

- **Phase 0.6 sanity (per `EXECUTION_PLAN.md` §6.0.6).** On 10K rollouts from 2025 PA starts, the rollout-derived K%, BB%, HR%, and mean wOBA must match 2025 empirical league rates within ±10% relative or ±1pp absolute, whichever is tighter. Mean wOBA within ±0.015 absolute. FAIL → API marked "uncalibrated rollout" until cause is found and fixed.
- **Pitch-token marginal regression at horizon=1.** With `horizon=1`, `pitch_token_marginal(rollout, position=0)` averaged over `n_samples=10_000` MUST match the backbone's direct next-token softmax (computed on the same starting context) within KL divergence ≤ 0.005 nats. This validates that the rollout sampler doesn't introduce sampling-bias artifacts.
- **Latency budget (per `EXECUTION_PLAN.md` §6.0.5 success criterion).** A 10-PA batch with `n_samples=100`, `horizon=6`, `outcome_predictor != None` must complete in < 5 seconds on RTX 3050. FAIL → `pitch_probs` becomes opt-in (`return_probs=False` becomes default for batch operations) and a tracked perf-regression issue is filed.
- **Calibration-validity flag round-trip.** A unit test must verify that each of the 9 enumerated invalidation reasons (§6) actually trips `calibration_valid = False` and appends the right string to `calibration_invalid_reasons`.
- **Pad-NaN convention.** A unit test must verify that aggregation utilities (§5) NaN-mask correctly: a rollout with 50 truncated samples out of 100 must produce identical means via `np.nanmean(...)` and via "manually filter pa_terminated then mean," to within float epsilon.

---

## 10. Compatibility matrix (Plans A × Plan B)

**Production cell as of 2026-04-26:** `(PG-v2 backbone × PGConcatHeadPredictor)`. Plan B Step 2 closed; A1 wins. Cells below preserved for forward-research planning if Plan A re-opens.

How the API survives the joint research outcome. Plan A (backbone) × Plan B (outcome predictor): 3 backbone × 4 predictor = 12 cells. For each cell: (a) which `OutcomePredictor` plugs in, (b) what (if anything) changes downstream, (c) what calibration re-check is required.

| Plan A outcome → / Plan B outcome ↓ | **Plan A: PG-v2 stays (PRODUCTION)** | **Plan A: PG-wider wins** (v3, larger d_model) | **Plan A: PG-pitcher-embedding wins** (v3-emb) |
|---|---|---|---|
| **Plan B: PGConcatHead** (`pg_concat_head`) — **PRODUCTION** | **PRODUCTION CELL.** A1 from Plan B Step 2; head input concat(hidden[128] + context[35] + pitch_oh[17] + zone_oh[26] + velo_oh[5]) = 211d, 3-layer MLP `211→128→64→7`. Checkpoint `models/pitchgpt_v2_outcomehead_a1.pt`. Calibration T=0.8003, ECE post-T 0.0114 on 2025 holdout. WEAKER PASS verdict — `in_play_hit` ceiling inherited. | Concat-head retrained against v3-wider's hidden state. Predictor checkpoint regenerated at `models/pitchgpt_v3_wider_outcomehead_a1.pt`. Both backbone and predictor `calibration.json` re-fit. The 211d input becomes `(d_model_v3 + 35 + 48)`. | Concat-head retrained against v3-emb's hidden state (which now includes pitcher embedding). The head's input shape becomes `(d_model + emb_dim + 35 + 48)`. **FLAG FOR USER REVIEW:** as in the frozen-head cell — decide whether to ship two predictor checkpoints (one per backbone) or to deprecate v2 once v3-emb wins. |
| **Plan B: XGBoost** (`xgboost`) | A3 baseline from Plan B Step 1. WEAKER PASS at +16.12% lift (vs A1's +18.31%). Available as a backstop predictor; not production. Backbone unchanged. | XGBoost predictor; backbone v3 swap — re-fit ENTIRE backbone temperature on the matched holdout, regenerate `calibration_feature_cdfs.npz`. XGBoost re-uses its existing calibration since it does not consume backbone hidden state (the backbone change is invisible to XGBoost). | Same as the wider-wins cell (XGBoost ignores backbone hidden state). Pitcher-embedding may add a `pitcher_emb_id` field; this is invisible to XGBoost. **FLAG FOR USER REVIEW:** if pitcher-embedding becomes a useful XGBoost feature, the predictor's input contract changes — schedule an API revision. |
| **Plan B: PG-frozen-head** (`pg_frozen_head`) — **DEPRECATED** | Phase 0.3 baseline that lost to A1 by +2.48pp paired (CI excludes zero by ~22 SE). Preserved for replay-ability of Phase 0.3 −5.34% FAIL artifact only; NOT production. | Would require retraining; not pursued — the concat-head dominates this design. | Would require retraining; not pursued. |
| **Plan B: PA-empirical-fallback** (`empirical_pa_terminal`) — kill-criterion safe-harbor | A5 baseline from Plan B Step 1. FAIL kill-criterion at +4.33% lift. Available as the "predictor unavailable" fallback. Backbone unchanged. Lookup table re-built only on annual season-data refresh. | Lookup table unchanged (does not consume backbone). Backbone calibration re-fit (§7.2). | Lookup table unchanged. Backbone calibration re-fit. **No predictor coupling to pitcher embedding** — the table keys are `(pitch_type, zone, velocity_bucket, count_state, batter_stand)`, which are all derivable from `pitch_token` + `context_vec` regardless of backbone variant. |

**Cells flagged for user review:**
- `(PG-pitcher-embedding, PGConcatHead)` and `(PG-pitcher-embedding, PG-frozen-head)` — non-trivial: the predictor's architecture must accept variable hidden-state shapes, or two predictors must be maintained. Decide before Plan A v3-emb is greenlit.
- `(PG-pitcher-embedding, XGBoost)` — only if pitcher embedding becomes an XGBoost feature; else trivial.

**Default cell** (PRODUCTION as of 2026-04-26): `(PG-v2 stays, PGConcatHead)`. Plan B Step 2 closed decisively in favor of A1. PG-frozen-head deprecated; XGBoost retained as documented backstop; empirical-fallback retained as kill-criterion safe-harbor.

---

## 11. What this API does NOT do

- **No game-state simulator.** The API is PA-level. Inning, game, and season aggregations are consumer responsibilities (`pitchgpt_projection.py` for season; `pitchgpt_wp.py` (B1) for inning/game).
- **No backtesting harness.** Each Tier-A dossier (`EXECUTION_PLAN.md` §6) owns its own backtest. The API provides the rollout primitive only.
- **No dashboard rendering.** All UI lives under `src/dashboard/views/`. The API has no Streamlit dependency.
- **No model training.** The API consumes trained checkpoints registered via §8. Training scripts live under `scripts/` (e.g., `pitchgpt_outcome_head_train.py`).
- **No claim-narrowing.** The "calibrated rollout engine" claim is what consumers can claim. Any further narrowing or broadening requires a doc revision (this doc + `EXECUTION_PLAN.md` §3) and user sign-off.
- **No silent fallback.** Every degraded mode (no outcome predictor, calibration-invalid context, truncated rollout) is surfaced via `sampling_metadata` or via NaN in arrays. Consumers cannot accidentally consume degraded output as if it were full-fidelity.
- **No automatic checkpoint discovery beyond §8.** New backbone variants must follow the path rule; new outcome predictors must follow the registry rule. The API will not introspect random `.pt` files.
- **No locking the surface forever.** This doc is versioned. Updates require user sign-off. The current version is `1.0` and matches `rollout_engine_version` in `sampling_metadata`.

---

*Document author: Claude (session 2026-04-25). Spec extends Plan C draft; awaits user review before the Phase 0.5 implementing agent fires. Leave unstaged.*
