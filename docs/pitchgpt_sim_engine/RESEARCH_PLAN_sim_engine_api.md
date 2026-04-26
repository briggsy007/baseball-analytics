# Research Plan C — Sim Engine API

**Date:** 2026-04-24
**Status:** drafted, not greenlit. No execution before user sign-off.
**Discipline:** Research → Plan → Execute (per `feedback_research_plan_execute.md`).
**Scope:** define the sim engine's API surface so product work (A1, A2, A3 from `EXECUTION_PLAN.md`) can proceed independent of which models from Plans A and B end up feeding it.

---

## 1. Mission statement

**Primary question.** Can we specify a model-agnostic sim-engine API surface that downstream Tier-A products (counterfactual pitch-call grade, probabilistic projections, matchup sims) target, before Plans A and B finish? If yes, product work parallelizes against research work — Tier-A integrations get scaffolded, validated, and dashboarded against the API contract while the underlying models swap in or out.

**Why this matters.** Three risks the API doc mitigates:
1. **Rework risk.** Without an API, every model variant from Plans A/B forces every downstream consumer to refactor. With an API, consumers code once.
2. **Honest-emission risk.** If the outcome predictor (Plan B) lands at WEAKER PASS or fails entirely, the API needs a CLEAR way to emit "no outcome distribution available — use PA-empirical fallback." Without that contract, downstream products silently produce garbage. The API formalizes the fallback path.
3. **CI propagation risk.** The flagship claim is calibrated CIs on every emitted quantity. The API is where that contract is enforced — every output carries CIs, no exceptions.

**Success.** A short, clear API doc that:
- Specifies the rollout function signature, input shapes, output shapes.
- Specifies the model-version selector contract (PG-with-outcome-head, PG-without, PG + external outcome model).
- Specifies the CI emission contract (every output carries CIs, no exceptions).
- Specifies the calibration re-check contract (when do we trust temperature; when do we re-fit).
- Specifies the fallback contract for the WEAKER-PASS / FAIL outcomes of Plan B.

**Failure.** N/A — this is a doc-only plan; the doc either exists and is reviewed, or it doesn't. There's no experiment to fail.

**This plan is NOT about.** Implementing the API (that's Phase 0.5 of `EXECUTION_PLAN.md`). Implementing Tier-A consumers (Phase 1 of EXECUTION_PLAN). Closing the PG perplexity gap (Plan A). Building the outcome head (Plan B).

---

## 2. Background — what's locked

The sim engine API has to live within these locked decisions:

- **Backbone.** PitchGPT v2 (`models/pitchgpt_v2.pt`, 35-dim context, ump scalar). Frozen — never overwrite. (`docs/pitchgpt_sim_engine/COORDINATION.md`).
- **Outcome head decision (locked, may be reopened by Plan B).** Frozen-backbone + 2-layer MLP head (`docs/pitchgpt_sim_engine/pa_outcome_head_design.md` §9). Phase 0.3 trained checkpoint at `models/pitchgpt_v2_outcomehead.pt` is currently FAIL gate per `results/pitchgpt/outcome_head_train_2026_04_24/report.md`. Plan B tests alternatives.
- **7-class outcome target.** `ball / called_strike / swinging_strike / foul / in_play_out / in_play_hit / hbp` per EXECUTION_PLAN §3.1 decision 2.
- **Calibration is the load-bearing flagship claim.** Post-temperature ECE < 0.02 on 2025 pitcher-disjoint holdout. The API never emits an uncalibrated probability silently.
- **Flagship-allowed claims.** "Calibrated rollout engine," "matches empirical marginals with calibrated uncertainty," "beats naive baselines by wide margins." NOT allowed: "beats LSTM by spec margin on perplexity," "sequence-aware sampling superior to LSTM."

---

## 3. API surface

Implementation target: `src/analytics/pitchgpt_sim.py` (per EXECUTION_PLAN §6.0.5; not yet created).

### 3.1 Primary entry point

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
) -> RolloutResult:
    ...
```

**Inputs.**

- `starting_context: PAContext` — a typed object capturing the PA's starting state. Fields:
  - `pitcher_id: int` (NOT used by backbone for inference; kept for output joining and pitcher-specific outcome model variants)
  - `batter_id: int` (same)
  - `count: tuple[int, int]` (balls, strikes)
  - `outs: int`
  - `runners: tuple[bool, bool, bool]` (on_1b, on_2b, on_3b)
  - `batter_stand: Literal["L", "R"]`
  - `pitcher_throws: Literal["L", "R"]`
  - `inning: int`
  - `inning_topbot: Literal["Top", "Bot"]`
  - `score_diff: int` (pitcher's POV)
  - `umpire_scalar: float` (prior-season ump accuracy_above_x; 0.0 default)
  - `prefix_pitch_tokens: list[int]` (any pitches already thrown in this PA; empty for a 0-0 start, partial for a counterfactual mid-PA query)

- `backbone_version: str` — default "v2"; "v1" supported via the legacy 34-dim context loader. Future versions resolve via `models/pitchgpt_<version>.pt` discovery. Backbone determines the next-pitch token distribution.

- `outcome_predictor: OutcomePredictor | None` — pluggable outcome model. Three concrete variants:
  - `PGFrozenHeadPredictor(checkpoint_path)` — Phase 0.3 frozen-MLP head. Currently FAIL-gate per Plan B; usable when Plan B replaces it.
  - `XGBoostOutcomePredictor(checkpoint_path)` — Plan B's A3 candidate (engineered features + XGBoost).
  - `EmpiricalPATerminalLookup(lookup_table_path)` — Plan B's §7.1 fallback (no model; empirical PA-terminal distribution).
  - `None` — sim returns pitch-token rollouts only, no outcomes; downstream consumers must handle absent outcomes (see §3.4 below).

- `n_samples: int` — number of independent rollout samples. Default 100 (covers most CI uses; A3 matchup sim wants 10K, A2 projections wants 30 starts × 25 PAs × 100 = 75K).

- `horizon: int` — max pitch positions per rollout. Default 6 (covers ~76th percentile of 2025 PA lengths). PA-termination behavior: stop at first sampled outcome in `{in_play_out, in_play_hit, hbp}` OR when the sampled count reaches walk (4 balls) or strikeout (3 strikes); pad remaining horizon positions with `RolloutPAD`.

- `temperature: float` — sampling temperature on the backbone's softmax. Default 1.0. Values <1 sharpen; >1 widen. Calibration is measured at T=1.0 — sampling at T≠1 INVALIDATES the calibration claim and the API logs a WARNING.

- `seed: int | None` — for reproducibility. None = nondeterministic.

**Output: `RolloutResult`** — a typed dataclass with fields:

- `pitch_tokens: np.ndarray[int]` shape `(n_samples, horizon)`. Sampled pitch tokens; `RolloutPAD = 2210` for positions past PA termination.
- `pitch_probs: np.ndarray[float]` shape `(n_samples, horizon, 2210)`. Per-sample, per-position probability vectors over the full token vocab. Calibrated (post-temperature). Shape large; can be omitted via `return_probs=False` for memory-constrained calls.
- `outcomes: np.ndarray[int] | None` shape `(n_samples, horizon)`. Sampled 7-class outcomes; `OutcomePAD = 7` for past-termination positions. **`None` if `outcome_predictor` is None.**
- `outcome_probs: np.ndarray[float] | None` shape `(n_samples, horizon, 7)`. Per-sample per-position outcome distribution. Calibrated post-temperature. **`None` if `outcome_predictor` is None.**
- `pa_terminated: np.ndarray[bool]` shape `(n_samples, horizon)`. True at the position where the PA ended.
- `pa_outcome: np.ndarray[int] | None` shape `(n_samples,)`. The terminal-position 7-class outcome (or `OutcomePAD` if PA didn't terminate within horizon). **`None` if `outcome_predictor` is None.**
- `sampling_metadata: dict` — temperature used, seed, backbone_version, outcome_predictor type + checkpoint hash, calibration validity flag (True iff T=1.0 and the calibration measurement is current, see §3.5).

### 3.2 Aggregations

Downstream consumers (A1, A2, A3) need PA-level and pitcher-level aggregations. Per-PA and per-rollout aggregations live in `pitchgpt_sim.py` as utility functions; per-pitcher and per-game aggregations live in the consumer modules (`pitchgpt_projection.py`, `pitchgpt_matchup.py`).

```python
def pa_woba_distribution(
    rollout: RolloutResult,
    woba_lookup: WObaTable | None = None,
) -> np.ndarray[float]:
    """
    Compute per-rollout-sample expected wOBA contribution.

    For each of n_samples rollouts:
      - If rollout terminated, look up wOBA for the terminal 7-class outcome
        × terminal pitch context. wOBA per outcome class is read from
        ``woba_lookup`` (Statcast empirical wOBA values per (outcome, pitch_type)).
      - If outcome_predictor was None, fall back to PA-empirical lookup (per
        Plan B §7.1), conditioning on terminal pitch token + count.
      - If rollout did not terminate within horizon, emit np.nan
        (NOT zero — silently dropped truncations would bias aggregates).

    Returns shape (n_samples,) with np.nan for truncated samples.
    """
```

```python
def percentile_of_actual_outcome(
    rollout: RolloutResult,
    actual_outcome: int,
    actual_pitch_token: int | None = None,
) -> float:
    """
    For A1 counterfactual pitch-call grading.

    Returns the percentile rank of the actual pitch's wOBA outcome
    within the rollout's wOBA distribution.

    Edge cases:
      - If outcome_predictor was None, the percentile is computed against
        PA-empirical outcomes; flag this in the returned metadata.
      - If actual_pitch_token differs from the rollout's predicted distribution,
        log a WARNING (the rollout was conditioned on the prefix WITHOUT the
        actual pitch — this is the counterfactual setup).
    """
```

### 3.3 Model-agnostic plug-in interface

The `OutcomePredictor` protocol:

```python
class OutcomePredictor(Protocol):
    def predict_outcome_probs(
        self,
        backbone_hidden_state: torch.Tensor,  # (B, S, d_model=128)
        context_vector: torch.Tensor,          # (B, S, 35)
        pitch_token: torch.Tensor,             # (B, S) — the sampled pitch
    ) -> torch.Tensor:                         # (B, S, 7)
        """
        Return calibrated 7-class outcome probabilities.

        Implementations MUST:
          - Apply their own internal temperature scaling if relevant.
          - Return probabilities (sum to 1 along the last dim), not logits.

        Implementations MAY:
          - Ignore backbone_hidden_state (e.g., XGBoost on engineered features
            doesn't need it). The signature accepts it for protocol uniformity.
        """
```

Concrete implementations:

- **`PGFrozenHeadPredictor`** uses backbone_hidden_state directly; ignores pitch_token (the head was trained without pitch_token as input).
- **`XGBoostOutcomePredictor`** ignores backbone_hidden_state; uses context_vector + pitch_token as features.
- **`EmpiricalPATerminalLookup`** ignores backbone_hidden_state; uses (pitch_token, count, batter_hand) as the lookup key.

This interface lets Plans A/B swap models without touching the rollout loop or any downstream consumer.

### 3.4 Behavior when `outcome_predictor` is None

The API MUST gracefully degrade. If no outcome predictor is supplied or available:

- `outcomes = None`, `outcome_probs = None`, `pa_outcome = None`.
- PA-termination logic falls back to count-based: walk if 4 balls, strikeout if 3 strikes, otherwise truncate at horizon. The `in_play_out` / `in_play_hit` / `hbp` axis is unavailable.
- Downstream consumers MUST check for `None` and either degrade their own emission OR raise. NEVER emit nonsense outcome aggregations from missing data.
- `sampling_metadata["outcome_predictor"] = "none"` so downstream artifacts can document the limitation.

This is the contract that makes Plan B's WEAKER-PASS or FAIL outcomes safe to ship — products fall back to "pitch-token-only rollouts" with explicit consumer-side disclosure rather than silently going wrong.

### 3.5 Calibration validity flag

`sampling_metadata["calibration_valid"]` is True iff ALL of:

1. `temperature == 1.0` (sampling at the calibrated temperature).
2. The backbone checkpoint's last calibration measurement is for the current (or a more recent) season — explicit metadata field on the checkpoint. Re-measure if the rollout is for a season later than the calibration window.
3. The outcome_predictor (if present) has its own calibration metadata current.
4. The starting_context falls within the in-distribution range for the calibration measurement (no extreme score_diff, no extra-inning extrapolation beyond the calibration's extrema).

If False, downstream consumers MUST flag the result as "uncalibrated rollout" in any artifact (dashboard, paper, leaderboard). The flag is non-overrideable.

---

## 4. Calibration re-check contract

Calibration is the flagship claim. The API must enforce when calibration is trusted vs when it must be re-measured.

### 4.1 When to re-measure

- **Annually.** Every January, re-fit temperature on a fresh validation slice (most-recent full season). If T changes by more than ±0.05, log a WARNING and update the saved temperature; if it changes by more than ±0.15, halt all sim-engine emission until the change is reviewed.
- **On context drift.** If the API is asked to roll out a context whose `(score_diff, inning, runner_state, etc.)` cluster is not represented in the calibration cohort, mark `calibration_valid = False` for that emission.
- **On checkpoint change.** Any new backbone or outcome_head checkpoint MUST be calibrated (temperature fit + ECE measurement on a held-out slice) before being usable in the API. The API rejects checkpoints lacking calibration metadata.

### 4.2 Re-measurement procedure

1. Load the checkpoint.
2. Score it on the most-recent full-season pitcher-disjoint holdout (e.g., 2025 if we're in 2026; 2026 if we're in 2027).
3. Fit temperature via LBFGS on a 2-year-prior validation slice.
4. Measure 10-bin ECE pre and post temperature.
5. Write `calibration.json` adjacent to the checkpoint with `{T, ECE_pre, ECE_post, holdout_season, holdout_n_pitches, fit_date}`.
6. Sim engine reads `calibration.json` on checkpoint load; if missing or older than 12 months, refuse to use the checkpoint.

### 4.3 Calibration on counterfactual contexts

EXECUTION_PLAN §3 flags this as TBD. The API surface CAN make it concrete: any rollout whose `starting_context` differs from the empirical 2025 distribution by more than X (TBD; concrete suggestion below) is marked `calibration_valid = False`.

**Concrete X: per-feature percentile distance.** For each context dimension (count, outs, score_diff, inning), compute the holdout's empirical CDF. A starting_context with any feature value below the 1st percentile or above the 99th percentile of the calibration cohort's distribution triggers `calibration_valid = False`.

This is conservative — it flags sim-engine queries on edge-case contexts that the calibration measurement didn't see — but it's safe.

---

## 5. Implementation sequencing

This plan is doc-only. Implementation lives in EXECUTION_PLAN Phase 0.5 (rollout harness) and Phase 0.6 (rollout sanity check).

When Plans A and B finish, this API doc tells the implementing agent:

1. Load whichever backbone Plan A leaves (v2 if A fails, v3-wider if H2 wins, etc.) at the locked path.
2. Load whichever outcome predictor Plan B leaves (PGFrozenHeadPredictor if A1 wins, XGBoostOutcomePredictor if A3 wins, EmpiricalPATerminalLookup if Plan B kill-criterion fires).
3. Implement `pitchgpt_sim.py` per §3 above.
4. Implement Tier-A consumers per EXECUTION_PLAN §6 dossiers, calling `rollout()` and respecting the calibration-validity flag.
5. The validation gate for the API itself is EXECUTION_PLAN §6.0.6 (rollout sanity check on 2025: K%, BB%, HR%, mean wOBA within ±10% relative or ±1pp absolute of empirical 2025).

---

## 6. What this plan does NOT do

- Does not implement anything. Implementation is EXECUTION_PLAN Phase 0.5/0.6 territory.
- Does not specify the dashboard view layouts. Each Tier-A item dossier in EXECUTION_PLAN §6 owns its UI.
- Does not specify backtest protocols for Tier-A items. Each item's gate criteria are in its dossier.
- Does not lock the API surface forever — Plan B's outcome shifting may motivate small additions (e.g., a `predict_outcome_probs_batched` method for XGBoost batch efficiency). Updates to this doc require re-review.

---

*Document author: Claude (session 2026-04-24). Doc-only spec; awaits user greenlight before any implementation agent fires. Leave unstaged.*
