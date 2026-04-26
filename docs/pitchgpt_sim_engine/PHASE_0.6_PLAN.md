# PitchGPT Sim Engine — Phase 0.6 Execution Plan

**Date:** 2026-04-26 (planning pass)
**Status:** drafted, not started. Phase 0.5 is **in flight in another Claude session** — this plan presupposes 0.5 PASS-of-record before any 0.6 step fires.
**Purpose of this document:** self-contained execution-ready dossier for Phase 0.6 (rollout sanity check on 2025). The next session reading only this file plus `docs/pitchgpt_sim_engine/PHASE_0.5_PLAN.md`, `docs/pitchgpt_sim_engine/SIM_ENGINE_API.md`, `docs/pitchgpt_sim_engine/COORDINATION.md`, and the locked baselines at `results/pitchgpt/rollout_sanity_2025/empirical_baselines_2025.json` should be able to run Phase 0.6 end-to-end with no further context.

---

## 1. Start here

**Purpose.** Run the Phase 0.5 rollout harness against 10K randomly-sampled 2025 PA starting contexts and verify that aggregate marginals (K%, BB%, HR%, mean wOBA, mean PA-length-in-pitches) match empirical 2025 league rates within the binding gate bands. PASS = Phase 0 closes; Tier-A items (A1 grades, A2 projections, A3 matchup sim per `EXECUTION_PLAN.md` §6) are unblocked.

**Prerequisites (HARD — verify before firing).**
- Phase 0.5 COMPLETE. `src/analytics/pitchgpt_sim.py` shipped with `rollout()` callable end-to-end. `tests/test_pitchgpt_sim.py` 100% green. Validation agent PASS-of-record.
- Backbone byte-identity: `models/pitchgpt_v2.pt` SHA256 = `6f952054d14ac6f918f3eb9502b496b70bc0c87dfc65dc50d98ee7244a62883c` (pre AND post 0.5).
- A1 head byte-identity: `models/pitchgpt_v2_outcomehead_a1.pt` size 151,289 bytes; recompute SHA256 at 0.6 close, compare to start.
- A1 calibration JSON: `models/calibration_pitchgpt_v2_outcomehead_a1.json` exists with `T = 0.8003096499977166`, `ECE_post = 0.0114097...`, `predictor_kind = "pg_concat_head"`. Per SIM_ENGINE_API §7.3 the API refuses to load A1 if this file is missing or out-of-budget (`ECE_post < 0.05`).
- Calibration feature CDFs: `models/calibration_feature_cdfs.npz` exists (built by Phase 0.5 ticket 0.5.4).
- Empirical baselines (this plan): `results/pitchgpt/rollout_sanity_2025/empirical_baselines_2025.json` exists and matches the gate numbers in §5 below.

**Strategic context.** Phase 0.6 PASS = Phase 0 exit gate per `EXECUTION_PLAN.md` §5. After 0.6 PASS, A1 (counterfactual pitch-call grades), A2 (probabilistic pitcher projections with CIs), A3 (matchup sims) can be scaffolded against the API. After Phase 0.6 FAIL, Phase 0 stalls until root cause is diagnosed; the rollout-engine claim narrows (see §9 kill criteria).

**Role.** PM-level coordinator per `feedback_pm_role.md` — delegate Phase 0.6 execution to a single subagent (Agent 3 per PHASE_0.5_PLAN §8.4). After 0.6 returns, run a validation agent before commit. PM does NOT write code directly.

**How this document is used.** §2 inheritance from PHASE_0.5_PLAN §3. §3 empirical baselines. §4 cohort spec. §5 binding gate numbers. §6 aggregation pseudo-spec. §7 caveats. §8 None-predictor secondary run. §9 kill criteria. §10 wall-clock + compute. §11 pickup checklist. §12 explicit out-of-scope.

---

## 2. Inherited spec from PHASE_0.5_PLAN §3

This document **REFINES PHASE_0.5_PLAN §3 with concrete empirical numbers**. On any conflict between this doc and PHASE_0.5_PLAN §3, **this doc wins** for Phase 0.6 execution. PHASE_0.5_PLAN §3 remains the parent design contract; if a parent revision is needed (cohort changed, gate redefined, methodology shifted), update PHASE_0.5_PLAN §3 first then re-derive this doc.

Concretely, this doc supplies:
- The locked empirical 2025 league rates with 95% bootstrap CIs (PHASE_0.5_PLAN §3.5 left these as variables; this doc fills them).
- The eligible-PA-starts count for the 2025 pitcher-disjoint cohort (PHASE_0.5_PLAN §3.2 assumed ≥10K; this doc confirms 64,460 eligible).
- The aggregation pseudo-spec (PHASE_0.5_PLAN §3.2 sketched the methodology; this doc disambiguates outcome → K/BB/HR mapping and wOBA aggregation).
- Kill criteria specifically for the 5 binding gates (PHASE_0.5_PLAN §3.3, §3.5 + §5.5 sketched the diagnosis tree; this doc enumerates concrete next-step diagnoses per gate).

Read PHASE_0.5_PLAN §3.1 (scope), §3.2 (methodology), §3.3 (honest caveat), §3.4 (None-predictor secondary), §3.5 (success criteria) BEFORE reading the rest of this document.

---

## 3. Empirical 2025 baselines (LOCKED)

Computed 2026-04-26 on the **same pitcher-disjoint cohort** that `scripts/pitchgpt_2025_holdout.py` and `scripts/pitchgpt_outcome_a1_concat.py` use: 2025 pitches whose `pitcher_id` is NOT in the 2015-2022 train cohort (2,247 train pitchers excluded; 499 eligible 2025 pitchers post-exclusion, 64,460 eligible PAs with terminal events).

Note: `results/pitchgpt/2025_holdout/report.md` was regenerated at an earlier date and reports 469 holdout pitchers; the 30-pitcher delta reflects 2025-season rows added by ingestion since that regeneration. The cohort *definition* (2025 ∧ NOT IN 2015-2022 train pitchers) is unchanged. A1's concat-head metrics file (`a1_concat/metrics.json`) reports 473 unique test pitchers because it required ≥2 pitches per pitcher-PA-sequence (Plan B Step 2 sequence-based loader); the empirical baseline is row-flat over all PAs and so legitimately differs.

### 3.1 League-level rates (point + 95% bootstrap CI, n=1000 iterations, seed=42, percentile method)

| Metric | Value | 95% CI | n PAs |
|---|---|---|---|
| K% | **21.80%** | [21.47%, 22.12%] | 64,460 |
| BB% | **8.76%** | [8.55%, 8.97%] | 64,460 |
| HR% | **3.21%** | [3.07%, 3.35%] | 64,460 |
| HBP% | 1.15% | [1.08%, 1.23%] | 64,460 |
| Hit% (1B+2B+3B+HR) | 22.18% | [21.86%, 22.50%] | 64,460 |
| Mean wOBA | **0.3302** | [0.3260, 0.3342] | 64,460 |
| Mean PA length (pitches) | **3.886** | [3.872, 3.900] | 64,460 |

### 3.2 By-inning bucket (point estimates only; for diagnostic context only — NOT gated)

| Bucket | n PAs | K% | BB% | HR% | mean wOBA |
|---|---|---|---|---|---|
| 1-3 | 31,034 | 22.07% | 8.21% | 3.04% | 0.3242 |
| 4-6 | 22,236 | 22.00% | 8.93% | 3.34% | 0.3349 |
| 7-9 | 10,431 | 21.04% | 9.91% | 3.34% | 0.3334 |
| 10+ | 759 | 18.45% | 11.59% | 3.95% | 0.3608 |

### 3.3 SQL used (one read pass per CLAUDE.md compute rules)

```sql
-- Train pitcher set (2015-2022)
SELECT DISTINCT pitcher_id
FROM pitches
WHERE pitch_type IS NOT NULL
  AND pitcher_id IS NOT NULL
  AND EXTRACT(YEAR FROM game_date) IN (2015..2022);

-- 2025 pitcher-disjoint pitches (NOT IN train cohort)
SELECT game_pk, pitcher_id, batter_id, at_bat_number, pitch_number,
       events, woba_value, inning
FROM pitches
WHERE EXTRACT(YEAR FROM game_date) = 2025
  AND pitch_type IS NOT NULL
  AND pitcher_id IS NOT NULL
  AND batter_id IS NOT NULL
  AND at_bat_number IS NOT NULL
  AND pitch_number IS NOT NULL
  AND pitcher_id NOT IN (<train pitchers>);

-- Per-PA aggregation:
--   group by (game_pk, at_bat_number, pitcher_id, batter_id)
--   terminal_event = events on row with MAX(pitch_number)
--   terminal_woba  = woba_value on the same terminal row
```

PA-terminal events are taken from the **same outcome-class set** as `src/analytics/pitchgpt_outcome_head.py::classify_pitch_outcome` (do NOT invent a new mapping):

- **K** = `events ∈ {strikeout, strikeout_double_play}`
- **BB** = `events == 'walk'`
- **HR** = `events == 'home_run'`
- **HBP** = `events == 'hit_by_pitch'`
- **Hit** = `events ∈ {single, double, triple, home_run}`
- **wOBA** = `pitches.woba_value` on the terminal pitch (NULL drop rate: 0.00% — every PA in the cohort has a non-null `woba_value` on its terminal row)

### 3.4 Data anomalies

- **119 PAs dropped** for missing terminal `events` (incomplete tracking). 0.18% drop rate; documented; not surfaced as a backfill blocker.
- **0 PAs dropped** for NULL `woba_value` on the terminal row — Statcast has 100% wOBA-on-terminal coverage in 2025 for this cohort, which is unusually clean.
- **Cohort-size delta vs report.md** (469 → 499 pitchers, 30 added): explained by post-report ingestion. Same SQL filter; freshly counted. The empirical baseline is the source of truth for Phase 0.6, not the report.md.

---

## 4. 10K PA-start cohort spec

### 4.1 Sampling rule

```python
import numpy as np
rng = np.random.default_rng(seed=42)
# pa_starts is a sorted DataFrame keyed on (game_pk, at_bat_number, pitcher_id, batter_id)
# with pitch_number == 1 only (the PA's starting context).
# n_eligible = 64,460 per §3.
n_eligible = 64_460
n_target = 10_000
sampled_idx = rng.choice(n_eligible, size=n_target, replace=False)
sampled_pas = pa_starts.iloc[sampled_idx].reset_index(drop=True)
```

Eligible PA starts = 64,460 (matches `n_pa_starts_eligible_for_rollout` in the JSON). 10K is comfortably achievable; **no upscale or `min(10K, available)` fallback needed**.

### 4.2 Per-PA context construction

For each sampled PA start:
1. Load the row corresponding to `(game_pk, at_bat_number, pitcher_id, batter_id, pitch_number == 1)` from `pitches`.
2. Construct `PAContext.from_pitches_row(row, ump_scalar=...)` per SIM_ENGINE_API §3.2.
3. Resolve `umpire_scalar`: lookup `umpire_assignments` join `umpire_tendencies` for `game_pk` → use HP umpire's `accuracy_above_x_wmean` for season=2025; if NULL or missing, fall back to **2025 league-median** (compute once over all 2025 umpire-tendency rows, cache).
4. Per-PA `prefix_pitch_tokens` = `()` (empty — Phase 0.6 rollouts start from 0-0 count, not mid-PA).

### 4.3 Edge cases

- **Multi-PA games.** A single game contributes multiple PAs; sampling is at the PA level not the game level, so multi-PA games are correctly over-represented in proportion to their PA density. This is the intended behavior — league rates are PA-weighted, not game-weighted.
- **Pinch hitters.** Treated identically to other batters; the `(game_pk, at_bat_number, pitcher_id, batter_id)` PA key uniquely identifies them. The batter's `stand` field on pitch 1 of the PA captures their handedness (resolved by Statcast against the at-bat's pitcher).
- **Ghost runner / extra innings.** PAs in the 10+ inning bucket inherit ghost-runner context (`on_2b == 1` from the ghost). This is encoded into `PAContext.runners` as expected. The 759 PAs in the 10+ bucket are real; they bias `runner_state` distributions away from the regular-inning median, which is fine for the per-PA rollout but flagged here for diagnostic clarity.
- **PAs in games where the pitcher was both starter and reliever** — duplicated `pitcher_id` is fine; PAs are uniquely keyed on `at_bat_number` per game.
- **Rosterless PAs.** `pitcher_id IS NULL` already filtered upstream. The remaining cohort has fully-resolved IDs.

### 4.4 Determinism rule

- The 10K sampling is deterministic in `seed=42`.
- For each sampled PA, the per-rollout seed passed to `rollout(..., seed=...)` is **`seed = 42 + pa_index * 1000`** (where `pa_index ∈ [0, 9999]`). This makes per-PA reruns independently reproducible without seed collision across PAs (the `* 1000` stride accommodates `n_samples=100` worth of internal RNG draws per PA without collision).
- The `np.random.default_rng(seed=42)` applies to BOTH the cohort sampling AND the bootstrap CIs over the 10K rollouts.

---

## 5. Phase 0.6 binding gates (LOCKED with concrete numbers)

These supersede the placeholder thresholds in PHASE_0.5_PLAN §3.5. Per `EXECUTION_PLAN.md` §6.0.6 + SIM_ENGINE_API §9: K%/BB%/HR% within ±10% relative OR ±1pp absolute (whichever is **tighter**); mean wOBA within ±0.015 absolute; mean PA length within ±0.5 pitches.

### 5.1 Binding gate table

| Gate | Empirical target (2025) | Tolerance (used) | PASS band | FAIL diagnosis hook (§9) |
|---|---:|---:|---:|---|
| **K%** | 21.80% (CI [21.47%, 22.12%]) | ±1.00 pp (abs is tighter than 10% rel = ±2.18 pp) | **[20.80%, 22.80%]** | strikeout count-driven termination logic OR per-class swinging-strike + called-strike marginals |
| **BB%** | 8.76% (CI [8.55%, 8.97%]) | ±0.876 pp (10% rel is tighter than ±1pp) | **[7.89%, 9.64%]** | walk count-driven termination logic OR ball marginal bias |
| **HR%** | 3.21% (CI [3.07%, 3.35%]) | ±0.321 pp (10% rel is tighter than ±1pp) | **[2.88%, 3.53%]** | `in_play_hit` predictor head (per §7 caveat) OR HR/non-HR sub-class within in-play-hit (current scalar wOBA table assigns HR a flat 0.892, masking the underlying class imbalance) |
| **Mean wOBA** | 0.3302 (CI [0.3260, 0.3342]) | ±0.015 abs | **[0.3152, 0.3452]** | `in_play_hit` ll=2.34 noise (most likely; see §7) OR scalar WObaTable values mismatched to 2025 league |
| **Mean PA length** | 3.886 (CI [3.872, 3.900]) | ±0.5 pitches | **[3.386, 4.386]** | termination logic too aggressive (foul/2-strike handling) OR horizon-truncation rate too high (samples should hit termination before horizon=6) |

### 5.2 Calibration-validity coverage gate (also binding)

Per SIM_ENGINE_API §6 + PHASE_0.5_PLAN §3.5: ≥95% of the 10K rollouts must report `sampling_metadata["calibration_valid"] == True`. If <95%, surface the dominant `calibration_invalid_reasons` from the bag of 9 enumerated strings — likely culprits: `context_score_diff_out_of_band` (extra-innings games), `context_runner_state_unseen` (rare runner combos), `context_outs_out_of_band` (defensive-fielding edge cases).

### 5.3 PASS verdict logic

**ALL of {K%, BB%, HR%, mean wOBA, mean PA length} ∈ PASS band AND calibration_valid coverage ≥95%** ⇒ Phase 0.6 PASS ⇒ Phase 0 exit ⇒ Tier-A unblocked.

**ANY gate FAIL** ⇒ Phase 0.6 FAIL ⇒ kill criterion §9 fires ⇒ diagnose, do NOT iterate the harness blind.

### 5.4 Reporting requirement

The Phase 0.6 report (`results/pitchgpt/rollout_sanity_2025/report.md`) MUST surface:
1. Sampled value + 95% bootstrap CI on every gate.
2. PASS/FAIL verdict per gate with the empirical target and PASS band shown alongside.
3. Calibration-valid coverage % + top-3 invalidation reasons (if <100%).
4. Honest caveat per §7.

---

## 6. Aggregation pseudo-spec

The actual `scripts/pitchgpt_rollout_sanity_2025.py` is written by the Phase 0.6 executing agent (per PHASE_0.5_PLAN §8.4). The pseudo-code below disambiguates the aggregation logic so the executing agent has zero ambiguity. Nothing here is implementation-final; the agent is free to refactor as long as the semantics match.

### 6.1 7-class outcome → K/BB/HR/in-play count mapping

```python
# Per SIM_ENGINE_API §3.3: 7 classes indexed 0-6.
OUTCOME_BALL = 0
OUTCOME_CALLED_STRIKE = 1
OUTCOME_SWINGING_STRIKE = 2
OUTCOME_FOUL = 3
OUTCOME_IN_PLAY_OUT = 4
OUTCOME_IN_PLAY_HIT = 5
OUTCOME_HBP = 6

def map_pa_outcome_to_marginal(rollout_result) -> dict:
    """For ONE rollout result (n_samples=100 samples for a single PA),
    return per-PA marginal counts.

    Returns
    -------
    dict with keys:
        k_count: int  — samples that ended in strikeout (terminated via
                        running_strikes == 3 path; pa_outcome may be ROLLOUT_PAD
                        because "K" is count-driven, not an in-play-class outcome)
        bb_count: int — samples that ended in walk (running_balls == 4)
        hr_count: int — samples where pa_outcome == OUTCOME_IN_PLAY_HIT AND the
                        HR resolution probability assigned by the WObaTable was a
                        HR (see §6.4 — using the scalar table this collapses to
                        in_play_hit; for HR-specific aggregation use the per-PA
                        HR-vs-other split documented in §6.4)
        hbp_count: int — samples where pa_outcome == OUTCOME_HBP
        in_play_count: int — samples where pa_outcome ∈
                             {OUTCOME_IN_PLAY_OUT, OUTCOME_IN_PLAY_HIT, OUTCOME_HBP}
        truncated_count: int — samples where pa_terminated.any(axis=1) == False
                               (PA did not end within horizon=6)
        n_samples: int
    """
    # final_count is shape (n_samples, 2) - (balls, strikes) at termination
    n_samples = rollout_result.final_count.shape[0]
    bb_count = int((rollout_result.final_count[:, 0] >= 4).sum())
    k_count = int((rollout_result.final_count[:, 1] >= 3).sum())
    pa_term = rollout_result.pa_terminated.any(axis=1)
    truncated_count = int((~pa_term).sum())
    if rollout_result.pa_outcome is None:
        # None-predictor mode: in-play counts are unavailable
        return dict(k_count=k_count, bb_count=bb_count, hr_count=0, hbp_count=0,
                    in_play_count=0, truncated_count=truncated_count,
                    n_samples=n_samples)
    pa_outcome = rollout_result.pa_outcome  # (n_samples,), ROLLOUT_PAD_OUTCOME=7 for non-terminating
    in_play_out_count = int((pa_outcome == OUTCOME_IN_PLAY_OUT).sum())
    in_play_hit_count = int((pa_outcome == OUTCOME_IN_PLAY_HIT).sum())
    hbp_count = int((pa_outcome == OUTCOME_HBP).sum())
    in_play_count = in_play_out_count + in_play_hit_count + hbp_count
    # HR is a sub-class of in_play_hit; with the scalar WObaTable
    # (no per-pitch-type info) we cannot decompose HR vs non-HR hits at the
    # outcome-class level. Two options for HR%:
    #   (a) Use the in_play_hit count multiplied by the empirical HR-fraction-among-hits
    #       (= 3.21% / 22.18% = 14.5% from §3.1), which assumes the rollout
    #       reproduces the 2025 hit-mix at scale.
    #   (b) When the full Statcast WObaTable lands (Phase-1 follow-up per
    #       SIM_ENGINE_API §5.1), HR resolution becomes type-conditioned and
    #       direct.
    # For Phase 0.6, use (a). Document in the report.
    HR_FRACTION_OF_HITS_2025 = 0.1446  # = HR% / Hit% from §3.1 (3.21 / 22.18)
    hr_count = int(round(in_play_hit_count * HR_FRACTION_OF_HITS_2025))
    return dict(k_count=k_count, bb_count=bb_count, hr_count=hr_count,
                hbp_count=hbp_count, in_play_count=in_play_count,
                truncated_count=truncated_count, n_samples=n_samples)
```

**Rationale for HR-via-fraction.** The A1 outcome head emits 7 classes; HR is not one of them. The scalar WObaTable (PHASE_0.5_PLAN §0.5.3) doesn't disambiguate HR from other in-play-hit. The Phase-1 follow-up (full Statcast wOBA table) decomposes HR via the per-pitch-type wOBA grid; until then, the `HR_FRACTION_OF_HITS_2025` proxy is the only honest aggregation. The report MUST disclose this.

### 6.2 Per-PA → per-rollout aggregation (n=100 samples)

```python
def per_pa_marginals(rollout_result) -> dict:
    counts = map_pa_outcome_to_marginal(rollout_result)
    # Valid samples = those that terminated within horizon
    n_valid = counts["n_samples"] - counts["truncated_count"]
    if n_valid == 0:
        return {"k_pct": np.nan, "bb_pct": np.nan, "hr_pct": np.nan,
                "hbp_pct": np.nan, "in_play_pct": np.nan,
                "truncation_rate": 1.0,
                "mean_pa_length": np.nan,
                "mean_woba": np.nan}
    # Per-PA fractions are over ALL n_samples (truncated rollouts count as
    # "did not terminate", which is itself a real outcome state — this matches
    # the empirical-side definition where every PA has SOME outcome).
    per_pa = {
        "k_pct": counts["k_count"] / counts["n_samples"],
        "bb_pct": counts["bb_count"] / counts["n_samples"],
        "hr_pct": counts["hr_count"] / counts["n_samples"],
        "hbp_pct": counts["hbp_count"] / counts["n_samples"],
        "in_play_pct": counts["in_play_count"] / counts["n_samples"],
        "truncation_rate": counts["truncated_count"] / counts["n_samples"],
    }
    # Mean PA length = mean position-of-termination + 1 (positions are 0-indexed)
    # over only the terminated samples; truncated samples contribute horizon (6)
    # as a worst-case lower bound on their length.
    pa_term = rollout_result.pa_terminated  # (n_samples, horizon)
    term_pos = np.where(pa_term.any(axis=1),
                        pa_term.argmax(axis=1) + 1,
                        rollout_result.pa_terminated.shape[1])  # horizon
    per_pa["mean_pa_length"] = float(term_pos.mean())
    # Mean wOBA per PA: aggregate via pitchgpt_sim.pa_woba_distribution
    # which returns shape (n_samples,) with NaN on truncated. Use np.nanmean.
    from src.analytics.pitchgpt_sim import pa_woba_distribution
    woba_per_sample = pa_woba_distribution(rollout_result)  # (n_samples,)
    per_pa["mean_woba"] = float(np.nanmean(woba_per_sample))
    return per_pa
```

### 6.3 League-level rollup (10K PAs)

```python
def league_rollup(per_pa_records: list[dict]) -> dict:
    """Aggregate per-PA marginals to league level.

    Each per_pa_records[i] is a dict from per_pa_marginals() above.
    """
    df = pd.DataFrame(per_pa_records)
    # Each PA contributes equal weight (PA-weighted league rate, matching
    # the empirical-side definition).
    league = {
        "k_pct": float(np.nanmean(df["k_pct"])),
        "bb_pct": float(np.nanmean(df["bb_pct"])),
        "hr_pct": float(np.nanmean(df["hr_pct"])),
        "hbp_pct": float(np.nanmean(df["hbp_pct"])),
        "mean_woba": float(np.nanmean(df["mean_woba"])),
        "mean_pa_length": float(np.nanmean(df["mean_pa_length"])),
        "truncation_rate": float(np.nanmean(df["truncation_rate"])),
    }
    # 95% bootstrap CIs on each: resample PAs (not samples-within-PA) with
    # replacement N=1000 times, recompute mean.
    rng = np.random.default_rng(seed=42)
    for metric in list(league.keys()):
        vals = df[metric].dropna().values
        n = len(vals)
        boot = np.array([
            vals[rng.integers(0, n, size=n)].mean() for _ in range(1000)
        ])
        league[f"{metric}_ci_lo"] = float(np.percentile(boot, 2.5))
        league[f"{metric}_ci_hi"] = float(np.percentile(boot, 97.5))
    return league
```

### 6.4 Mean-wOBA from 7-class outcomes — using the locked default WObaTable

Per PHASE_0.5_PLAN §0.5.3 the default `WObaTable.default()` is a **7-element scalar table** keyed on outcome class only:

```
ball              = 0.000
called_strike     = 0.000
swinging_strike   = 0.000
foul              = 0.000
in_play_out       = 0.000
in_play_hit       = 0.892   (league-avg wOBA on a hit, all hit types weighted)
hbp               = 0.708   (the canonical wOBA-per-HBP value)
```

For a sample whose PA terminates with `pa_outcome ∈ {OUTCOME_BALL ... OUTCOME_FOUL}` (count-driven, not in-play): the wOBA contribution depends on the count-driven termination — walks contribute **0.690** (canonical wOBA-per-walk) and Ks contribute **0.000**. Strict implementation in `pa_woba_distribution()` derives walk/K from `final_count` (balls≥4 → walk, strikes≥3 → K) and applies the wOBA values; the exact per-class mapping the agent uses must mirror `WObaTable.default()` from `pitchgpt_sim.py` (Phase 0.5 ticket 0.5.3).

**If the implemented `WObaTable.default()` differs from the values above, this doc loses. Re-derive the empirical target accordingly. The values above are the 2025 canonical wOBA constants and should match. If they do not, surface to PM before running the rollout.**

### 6.5 Decomposition for FAIL-diagnosis (per PHASE_0.5_PLAN §5.5)

If mean wOBA fails, decompose into:

```
mean_wOBA_predicted = K%_pred * 0.000
                    + BB%_pred * 0.690
                    + HBP%_pred * 0.708
                    + in_play_hit%_pred * 0.892
                    + in_play_out%_pred * 0.000
```

vs

```
mean_wOBA_empirical = K%_emp  * 0.000
                    + BB%_emp  * 0.690 = 0.0876 * 0.690 = 0.0604
                    + HBP%_emp * 0.708 = 0.0115 * 0.708 = 0.0081
                    + Hit%_emp * 0.892 = 0.2218 * 0.892 = 0.1979
                    + InPlayOut%_emp * 0 = 0
                    = 0.2664 (decomposed)
```

There is a 0.064 gap between this scalar-table reconstruction (0.2664) and the empirical mean wOBA from the woba_value column (0.3302) — this gap reflects that the scalar table understates wOBA for non-HR hits (1B = 0.870, 2B = 1.244, 3B = 1.572, HR = 2.000 in canonical 2025 wOBA weights). The PHASE_0.5_PLAN §0.5.3 default uses 0.892 = league-avg wOBA-per-hit which is appropriate for **average** behavior but creates a known systematic under-estimate vs the empirical-side `pitches.woba_value`.

**Implication for the wOBA gate.** The empirical target 0.3302 in §3.1 uses `pitches.woba_value` (full per-event Statcast wOBA). The rollout's mean wOBA uses the scalar 7-class WObaTable. If the predicted-vs-empirical wOBA gap is approximately 0.064 (the systematic gap above), the FAIL is a **WObaTable-spec issue, not a predictor issue** — the executing agent surfaces this and the report distinguishes the two. The right Phase-0.6 PASS judgment is "rollout reproduces empirical when measured under matched aggregation" — see §9 for the kill-criterion narrowing.

**Recommended Phase 0.6 implementation:** ALSO compute "rollout-aggregated wOBA under empirical-side aggregation" — use the rollout's K/BB/HR/Hit/HBP marginals plug in the canonical wOBA-per-event weights to reconstruct what the empirical-side aggregation would yield. Compare BOTH to 0.3302 and the 0.2664 decomposition target. If the rollout matches one but not the other, the diagnosis is unambiguous.

---

## 7. Honest caveats (must be in the Phase 0.6 report when 0.6 runs)

These are inherited disclosure obligations per `EXECUTION_PLAN.md` §3 + `COORDINATION.md` "Plan B verdict" + SIM_ENGINE_API §4 ceiling note. The Phase 0.6 executing agent MUST surface them in `report.md`.

### 7.1 A1 `in_play_hit` ceiling

Source: `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/metrics.json` field `test_metrics.per_class_log_loss.in_play_hit = 2.342`.

Disclosure text the report MUST contain (paraphrased acceptable, key facts mandatory): *"The A1 outcome predictor's `in_play_hit` test log-loss is 2.34 (clears WEAKER PASS <2.5; misses full PASS <2.0). This is a structural ceiling — hit-vs-out resolution depends on launch_speed and launch_angle, post-pitch features no architecture in this study has access to. Mean wOBA aggregation is therefore noisier than K%/BB%/HR% because it depends on the hit-vs-out marginal."*

### 7.2 HR-via-fraction proxy

Per §6.1 the rollout's HR% is computed as `in_play_hit% × 0.1446`, where 0.1446 = empirical HR-fraction-among-hits in the 2025 cohort. This **assumes the rollout reproduces the 2025 hit-mix**. If the rollout systematically over- or under-samples HR among hits (which the model has no direct signal for, since the outcome head treats all hits as `in_play_hit`), the HR% gate will reflect that bias. Phase-1 follow-up (full Statcast WObaTable) resolves this; for Phase 0.6 it is a documented limitation.

### 7.3 Scalar WObaTable default

Per §6.4 the wOBA scalar table is a deliberate Phase-0.5 default (PHASE_0.5_PLAN §0.5.3 + §5.4). It uses `in_play_hit = 0.892` (league-avg wOBA-per-hit) which is correct on average but creates a known systematic gap vs `pitches.woba_value` aggregation. The report MUST distinguish "rollout-aggregated wOBA under matched aggregation" from "rollout-aggregated wOBA vs empirical full-Statcast wOBA" — they are NOT the same number.

### 7.4 Calibration_valid is a hard gate

If <95% of rollouts report `calibration_valid == True`, Phase 0.6 stalls — the harness is rejecting too many contexts as out-of-distribution. Most likely culprits are §6 condition 4 percentile-band gates over-rejecting on extra-innings or rare runner combos. Surface the dominant invalidation reason; PM decides whether to widen the percentile band (1st-99th → 0.5th-99.5th, say) or accept the narrowed cohort.

### 7.5 Bonferroni / multiple-comparison note (recommended, not required)

The 5 binding gates are tested simultaneously. Strict 95% Bonferroni-corrected α per gate would be 0.01 (= 0.05 / 5). The CIs in §3.1 are 95% uncorrected; the gate bands in §5.1 are 95% percentile bands too. If FAIL borderline (one gate FAIL with the FAIL-side CI bracket including PASS-band), the report should note: *"FAIL is uncorrected; under Bonferroni this is closer to a tie than a clean FAIL"* — same convention used for Phase 0.1 metric 4 (see `EXECUTION_PLAN.md` §6 item 0.1 status).

---

## 8. None-predictor secondary run (NOT GATED)

Per PHASE_0.5_PLAN §3.4 + SIM_ENGINE_API §4.4 + §5 risk 3.

### 8.1 Methodology

Same 10K cohort as the primary run. Same per-PA seeds. Only difference: `outcome_predictor=None` instead of `PGConcatHeadPredictor()`.

### 8.2 Expected behavior

Per SIM_ENGINE_API §4.4: PA-termination falls back to count-only heuristic — zone in-zone → +1 strike, zone out-of-zone → +1 ball. **Every in-play, foul, and HBP outcome is misclassified as a strike** at the termination logic. K%/BB% will be biased high. In-play and HBP marginals are 0% by construction (no outcome head means no in-play class). Mean wOBA is undefined (no outcome → no wOBA assignment); report should record N/A or compute via `pa_woba_distribution`'s degraded path (returns the §5.1 fallback if the rollout's `outcomes is None`).

### 8.3 Reporting

The Phase 0.6 report's "None-predictor degraded mode" section MUST include:

| Metric | Empirical | Primary (A1) | Secondary (None) | Bias (None − Empirical) |
|---|---|---|---|---|
| K% | 21.80% | <run> | <run> | <delta> pp |
| BB% | 8.76% | <run> | <run> | <delta> pp |
| HR% | 3.21% | <run> | <run> (likely 0%) | <delta> pp |
| Mean wOBA | 0.3302 | <run> | <run> (N/A or fallback) | <delta> |

The "Bias" column is the **point estimate of None-mode bias magnitude**. This is the secondary's deliverable — it tells consumers what they get when the predictor is unavailable. The report writes "expected bias confirmed at +Xpp on K%, +Ypp on BB%" — this number is informational, NOT a Phase 0.6 PASS/FAIL gate input.

### 8.4 Conflation guard

The 5 binding gates in §5.1 use the **PRIMARY (A1) results only**. Do NOT include the None-predictor numbers in the PASS/FAIL judgment. Do NOT average across primary + secondary. The two runs are computed and reported independently; the secondary exists solely to surface bias magnitude per SIM_ENGINE_API §5 risk 3.

---

## 9. Kill criteria (explicit, per `feedback_research_plan_execute.md` discipline)

Per memory `feedback_research_plan_execute.md`: every plan has a kill criterion. Phase 0.6's:

### 9.1 Per-gate FAIL diagnosis (mandatory before iteration)

If FAIL on K%/BB%/HR%/wOBA/PA-length, do **NOT** iterate on the harness without first diagnosing root cause. Diagnoses:

1. **K% FAIL diagnosis tree.** (a) Inspect rollout-emitted swinging_strike + called_strike marginals against empirical 16.4% + 16.8% (from A1 metrics file). (b) Verify the 2-strike foul handling — fouls on 2 strikes do NOT advance to 3 per MLB rules; check the harness's PA-termination logic implements this. (c) Verify the count counter's order-of-operations: is the strike-counter incremented BEFORE or AFTER the outcome predictor fires? Off-by-one here biases K% high.

2. **BB% FAIL diagnosis tree.** (a) Inspect rollout-emitted ball marginal against empirical 36.5% (from A1 metrics file `train_freq_prior.ball`). (b) Verify intent_ball / pitchout aren't being double-counted (they should map to OUTCOME_BALL via the SIM_ENGINE_API §3.3 contract). (c) Verify the ball-counter wrap to walk: `running_balls == 4` triggers walk; `>=4` is also OK but `==` is the spec.

3. **HR% FAIL diagnosis tree.** (a) Most likely the HR-via-fraction proxy (§7.2). Compute the rollout's `in_play_hit%` directly and compare to empirical 22.18% — if `in_play_hit%` is on-target, the HR-fraction proxy is the issue. (b) If `in_play_hit%` itself is off, the diagnosis is on the A1 head's hit-vs-out marginal (per §7.1).

4. **mean wOBA FAIL diagnosis tree.** (a) Compute the §6.5 decomposition reconstructed-vs-empirical to check whether the gap is a WObaTable-spec issue or a predictor issue. (b) If the predictor's class marginals are on-target but mean wOBA is off, the WObaTable scalar is the issue → narrow per §9.2.

5. **mean PA length FAIL diagnosis tree.** (a) Inspect `truncation_rate` — if >5% of rollouts are truncating (PA does not terminate within horizon=6), the harness is under-emitting K/BB/in-play terminators or the horizon is too short for the empirical 3.886-pitch mean. (b) Check the foul-on-2-strike logic per (1c) — over-aggressive foul-strikes-out biases PA length down.

### 9.2 If diagnosis confirms a structural FAIL, the rollout claim narrows

Per `feedback_pm_role.md` and PHASE_0.5_PLAN §3.5 + §5.5: if the FAIL persists after diagnosis, the rollout claim narrows from **"calibrated rollout engine on PA-level marginals"** (which is what Phase 0.6 is meant to demonstrate) to **"calibrated rollout engine on token-level marginals only"** (which is what Phase 0.1 demonstrated, and what the existing `EXECUTION_PLAN.md` §3 narrowing already covers).

This is a **real product narrowing** — surface to user, do NOT paper over. The `EXECUTION_PLAN.md` §3 allowed-claims list and `NORTH_STAR.md` PitchGPT row update to drop the "PA-level marginals" sub-claim. Tier-A items (A1, A2, A3) lose one of their gates and pick up an "uncalibrated PA-level rollouts" disclosure.

### 9.3 Do NOT iterate the harness blind

Per PHASE_0.5_PLAN §4 latency-test discipline: if the first-pass result FAILs, do **NOT** ask Agent 3 (or a new agent) to "make it pass" without measurement. Run the diagnosis tree first, measure, then iterate ONLY on the specific subsystem the diagnosis pinpoints.

### 9.4 Calibration-valid coverage <95% kill

If the calibration-validity gate (§5.2) FAILs at <95% coverage, halt before evaluating §5.1 gates. Surface the dominant invalidation reason. PM decides:
- (a) Widen the percentile band (e.g., 1st-99th → 0.5th-99.5th) — accepts more cohort in.
- (b) Restrict the cohort to in-band PAs only — re-runs Phase 0.6 on the smaller cohort.
- (c) Accept narrowed coverage and ship with a "<X% rollouts uncalibrated" disclosure.

The decision is the user's, NOT the agent's — escalate per `feedback_pm_role.md`.

---

## 10. Wall-clock estimate + compute budget

### 10.1 Latency budget per PHASE_0.5_PLAN §0.5.6

10-PA batch with `n_samples=100`, `horizon=6`, A1 predictor → <5s on RTX 3050.

### 10.2 10K rollout extrapolation

10,000 PAs × 100 samples × 6 horizon = 6M rollout-positions. At 5s per 10-PA batch that's **5,000s ≈ 1h 23min** wall-clock pure rollout time on RTX 3050. Add SQL fetch (~1 min for the 10K cohort + 64K-PA empirical baseline cross-check), per-PA `PAContext` construction (~5 min Python overhead), aggregation + bootstrap CIs (~2 min). **Estimated end-to-end Phase 0.6 wall-clock: ~1h 35min on RTX 3050 GPU.**

CPU fallback (no CUDA): ~6× slower per PHASE_0.5_PLAN §0.5.6 30s-on-CPU budget vs 5s-on-GPU → ~9h 20min total. Recommend GPU only.

### 10.3 Secondary (None-predictor) run

Without A1 head forward, latency is ~2× faster (no predictor forward in the inner loop). Estimated **~45 min on RTX 3050**.

### 10.4 Total wall-clock estimate

Primary + secondary + report = **~2h 30min single-session**. Fits within a typical PM single-session budget.

### 10.5 GPU/CPU split

- **GPU.** Backbone forward (per-position transformer + softmax) + A1 head forward (concat + 3-layer MLP). 100% of the inner rollout loop.
- **CPU.** SQL fetch, PAContext construction, aggregation, bootstrap CIs, JSON/markdown emission.
- **Memory.** With `return_probs=True` (default per SIM_ENGINE_API §3.1): `n_samples=100 * horizon=6 * 2210 vocab * 4 bytes = 5.3 MB per PA × 10K PAs = ~53 GB`. **Exceeds RTX 3050 VRAM AND system RAM.** Use `return_probs=False` for the 10K rollout — Phase 0.6 only needs marginals, not per-position probabilities. With probs disabled, memory drops to `(n_samples * horizon * 8 bytes) ≈ 5 KB per PA = ~50 MB total`. **The Phase 0.6 agent MUST pass `return_probs=False`.** Surface this in the report.

---

## 11. Pickup checklist for the 0.6 executing agent

Run these 5 checks BEFORE firing `scripts/pitchgpt_rollout_sanity_2025.py`. Any FAIL halts and surfaces to PM.

- [ ] **(i) Verify Phase 0.5 PASS-of-record.** `pytest tests/test_pitchgpt_sim.py` → all green. `git log -n 3` shows the Phase 0.5 commit. Validation agent's report.md (most recent) reports clean.
- [ ] **(ii) Verify A1 checkpoint SHA matches COORDINATION.md.** `python -c "import hashlib; print(hashlib.sha256(open('models/pitchgpt_v2_outcomehead_a1.pt','rb').read()).hexdigest())"` → compare to the value in `models/calibration_pitchgpt_v2_outcomehead_a1.json`'s `checkpoint_sha256` field.
- [ ] **(iii) Verify backbone v2.pt SHA still `6f952054…62883c`.** Same SHA256 routine on `models/pitchgpt_v2.pt`. Must match `6f952054d14ac6f918f3eb9502b496b70bc0c87dfc65dc50d98ee7244a62883c`.
- [ ] **(iv) Verify `models/calibration_pitchgpt_v2_outcomehead_a1.json` exists AND `ECE_post < 0.05`.** `python -c "import json; d=json.load(open('models/calibration_pitchgpt_v2_outcomehead_a1.json')); print(d['ECE_post'], d['ECE_post'] < 0.05)"` → expect `0.011409712028199842 True`.
- [ ] **(v) Read THIS doc + `results/pitchgpt/rollout_sanity_2025/empirical_baselines_2025.json`.** Cross-check that the `gates_for_phase_0_6` block in the JSON matches §5.1 of this doc (no drift; the JSON is the machine-readable mirror of §5.1's table).

Bonus (recommended but not gating):
- [ ] **(vi)** Verify `models/calibration_feature_cdfs.npz` exists (built by Phase 0.5 ticket 0.5.4). `python -c "import numpy as np; d = np.load('models/calibration_feature_cdfs.npz'); print(list(d.keys()))"` → expect at least `count_state`, `outs`, `score_diff`, `inning_bucket`, `runner_state` keys.
- [ ] **(vii)** Spot-check that `from src.analytics.pitchgpt_sim import rollout, PAContext, PGConcatHeadPredictor, pa_woba_distribution` imports cleanly.

---

## 12. What this plan does NOT do

- **No harness implementation.** `src/analytics/pitchgpt_sim.py` is owned by Phase 0.5.
- **No collision with Phase 0.5.** Zero edits to any file in PHASE_0.5_PLAN's ticket scope (`pitchgpt_sim.py`, `pitchgpt_outcome_predictors/*`, `pitchgpt_build_calibration_cdfs.py`, `models/calibration_*`, `tests/test_pitchgpt_sim.py`).
- **No COORDINATION.md edits.** The Phase 0.5 session may be writing it; let them close first.
- **No model edits.** All checkpoints LOCKED per COORDINATION.md "Checkpoints — LOCKED."
- **No Tier-A consumer scaffolding.** A1 grades, A2 projections, A3 matchup sim are Phase 1 dossiers (post-Phase-0); PHASE_0.6_PLAN does not pre-empt their planning passes.
- **No XGBoost or empirical-fallback evaluation.** Per SIM_ENGINE_API §8.2 those predictors are conditionally registered (XGBoost checkpoint missing on disk; empirical lookup parquet missing) but NOT gated by Phase 0.6. The PRIMARY run uses `PGConcatHeadPredictor` exclusively.
- **No commits during planning.** Empirical baselines + this doc are unstaged. PM decides commit timing post-Phase-0.6 (likely as a single Phase 0.5 + 0.6 + planning consolidation commit).
- **No agent firing during planning.** No model training. No rollout sampling. Phase 0.6 EXECUTION (which fires the agent and runs the rollouts) is a separate session, gated on Phase 0.5 PASS-of-record.
- **No re-litigation of Plan B verdict** (A1 ships per 2026-04-26).
- **No relaxation of PASS bands** without surfacing to user — §5.1 numbers are LOCKED.

---

## 13. Cross-references

- **Parent design contract:** `docs/pitchgpt_sim_engine/PHASE_0.5_PLAN.md` §3 (Phase 0.6 inherited spec).
- **API surface:** `docs/pitchgpt_sim_engine/SIM_ENGINE_API.md` §3 (`rollout()`), §4 (`OutcomePredictor` Protocol + `PGConcatHeadPredictor`), §5 (aggregation), §6 (calibration validity), §9 (gates the API itself must pass).
- **Phase scope:** `docs/pitchgpt_sim_engine/EXECUTION_PLAN.md` §6.0.6 (original dossier), §6.0.5 (latency dependency), §3 (allowed-claims constraints).
- **Cross-session state:** `docs/pitchgpt_sim_engine/COORDINATION.md` (Plan B verdict, A1 ship details, file ownership, locked checkpoints).
- **A1 metrics (load-bearing for §7 caveats):** `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/metrics.json`.
- **A1 verdict summary:** `results/pitchgpt_sim/outcome_baselines_2026_04_25/SUMMARY.md`.
- **Cohort definition (canonical):** `scripts/pitchgpt_2025_holdout.py` (function `fetch_pitcher_ids_for_seasons` + the train/val/test split convention).
- **Outcome class mapping (canonical):** `src/analytics/pitchgpt_outcome_head.py::classify_pitch_outcome` + the `_HIT_EVENTS` / `_IN_PLAY_OUT_EVENTS` frozensets.
- **Empirical baselines (machine-readable):** `results/pitchgpt/rollout_sanity_2025/empirical_baselines_2025.json` (this plan's mirror file).
- **Empirical baselines (human-readable):** `results/pitchgpt/rollout_sanity_2025/empirical_baselines_2025.md`.

---

*Document author: Claude (session 2026-04-26, planning agent). Plan-only — no code written, no agents fired, no commits. Awaits user review and Phase 0.5 PASS-of-record before the Phase 0.6 executing agent fires. Leave unstaged.*
