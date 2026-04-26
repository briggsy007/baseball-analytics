# Empirical 2025 League Baselines (Pitcher-Disjoint Cohort)

Generated: 2026-04-26T14:36:13+00:00  
Computation wall-clock: 7.6s  
Bootstrap: n=1000, seed=42, percentile method  

## Purpose

Locked empirical 2025 league rates for the Phase 0.6 rollout sanity check. Computed on the **same pitcher-disjoint test cohort** that `scripts/pitchgpt_2025_holdout.py` and `scripts/pitchgpt_outcome_a1_concat.py` use (2025 pitches whose `pitcher_id` is NOT in the 2015-2022 train cohort).

## Cohort

- Season: **2025**  
- Pitcher-disjoint exclusion: 2015-2022 train cohort (2,247 pitchers excluded)  
- Eligible 2025 pitchers (post-exclusion): **499**  
- Eligible PAs (with terminal event): **64,460**  
- Total pitches scored: **250,476**  
- 10K rollout cohort achievable: **YES** (eligible PA starts: 64,460)  
- PAs dropped (no terminal event in `events`): 119  
- PAs with NULL `woba_value` (excluded from mean wOBA only): 0 (0.00%)  

## League rates with 95% bootstrap CIs

| Metric | Value | 95% CI | n |
|---|---|---|---|
| K% | 21.80% | [21.47%, 22.12%] | 64,460 |
| BB% | 8.76% | [8.55%, 8.97%] | 64,460 |
| HR% | 3.21% | [3.07%, 3.35%] | 64,460 |
| HBP% | 1.15% | [1.08%, 1.23%] | 64,460 |
| Hit% (1B+2B+3B+HR) | 22.18% | [21.86%, 22.50%] | 64,460 |
| Mean wOBA | 0.3302 | [0.3260, 0.3342] | 64,460 |
| Mean PA length (pitches) | 3.886 | [3.872, 3.900] | 64,460 |

## By-inning bucket (point estimates only)

| Bucket | n PAs | K% | BB% | HR% | mean wOBA |
|---|---|---|---|---|---|
| 1-3 | 21,642 | 22.42% | 8.73% | 3.50% | 0.3301 |
| 4-6 | 22,144 | 20.84% | 8.50% | 3.25% | 0.3358 |
| 7-9 | 19,961 | 22.25% | 9.04% | 2.87% | 0.3245 |
| 10+ | 713 | 19.63% | 10.10% | 2.24% | 0.3157 |

## Phase 0.6 binding gates (locked)

Per PHASE_0.5_PLAN §3.5 + EXECUTION_PLAN §6.0.6: K%/BB%/HR% within ±10% relative OR ±1pp absolute (whichever is **tighter**); mean wOBA within ±0.015 absolute; mean PA length within ±0.5 pitches.

| Metric | Target | Tolerance (used) | PASS band |
|---|---|---|---|
| K% | 21.80% | ±1.00pp | [20.80%, 22.80%] |
| BB% | 8.76% | ±0.88pp | [7.89%, 9.64%] |
| HR% | 3.21% | ±0.32pp | [2.88%, 3.53%] |
| Mean wOBA | 0.3302 | ±0.015 | [0.3152, 0.3452] |
| Mean PA length | 3.886 | ±0.5 | [3.386, 4.386] |

## SQL used (key filter)

```sql
-- Train pitcher set (2015-2022)
SELECT DISTINCT pitcher_id
FROM pitches
WHERE pitch_type IS NOT NULL
  AND pitcher_id IS NOT NULL
  AND EXTRACT(YEAR FROM game_date) IN (2015..2022);

-- 2025 pitcher-disjoint pitches
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
-- group by (game_pk, at_bat_number, pitcher_id, batter_id)
-- terminal_event = events on row with MAX(pitch_number)
-- terminal_woba = woba_value on the same terminal row
```

## Outcome label mapping

Strict mirror of `src/analytics/pitchgpt_outcome_head.py` `HIT_EVENTS` / `IN_PLAY_OUT_EVENTS` / `WALK_EVENTS` / `K_EVENTS` / `HBP_EVENTS`:

- **K** ← `events ∈ {strikeout, strikeout_double_play}`
- **BB** ← `events ∈ {walk}` (intent_walk on `events` is rare and merged into walk by Statcast as of 2018)
- **HR** ← `events == 'home_run'`
- **HBP** ← `events == 'hit_by_pitch'`
- **Hit** ← `events ∈ {single, double, triple, home_run}`
- **wOBA** ← `pitches.woba_value` on terminal pitch (NULL drop rate reported above: 0.00%)

## Cross-reference

- A1 outcome predictor metrics (same cohort): `results/pitchgpt_sim/outcome_baselines_2026_04_25/a1_concat/metrics.json`  
- Holdout report (cohort definition): `results/pitchgpt/2025_holdout/report.md` (reports 469 holdout pitchers; this run finds 499 — any difference reflects added rows since the report was regenerated)  
- Phase 0.6 plan: `docs/pitchgpt_sim_engine/PHASE_0.6_PLAN.md`  