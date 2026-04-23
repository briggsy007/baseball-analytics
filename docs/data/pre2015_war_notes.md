# Pre-2015 Season WAR Backfill Notes (2010-2014)

**Date:** 2026-04-23
**Script:** `scripts/backfill_season_war_2010_2014.py`
**Staging:** `data/staging/season_batting_stats_2010_2014.parquet`, `data/staging/season_pitching_stats_2010_2014.parquet`
**DuckDB:** *NOT touched.* Stage-only; merge gated by separate review.

## Motivation

The CausalWAR contrarian backtest's Buy-Low hit rate 68% CI is noisy with only
~11 season-pairs (2015→2016 through 2024→2025). Adding 2010-2014 yields 4
additional season-pairs (2010→2011, 2011→2012, 2012→2013, 2013→2014) plus
one boundary pair (2014→2015) if we join to the existing 2015 row.

## Data source decision (bWAR, not FanGraphs)

The ticket asked for `pybaseball.fg_batting_data(2010, 2014)` /
`pybaseball.fg_pitching_data(2010, 2014)`. Both wrappers — and every other
FanGraphs wrapper in `pybaseball` 2.2.7 — hit
`https://www.fangraphs.com/leaders-legacy.aspx` and return **HTTP 403** from
this host (verified 2026-04-23; same issue already documented in
`scripts/backfill_fwar.py` from 2026-04-16).

We therefore use the same fallback the production 2015-2024 WAR column uses:
`pybaseball.bwar_bat(return_all=True)` and `pybaseball.bwar_pitch(return_all=True)`
(Baseball-Reference WAR, which ships as a CSV and is not 403-blocked). Every
staged row is tagged `war_source = "bref_bwar"` — identical to the 2015-2024
production WAR dialect in `season_*_stats.war`, so the join across the boundary
is apples-to-apples.

If FanGraphs becomes reachable, swap the source in `fetch_war_for_years()` —
the rest of the pipeline is dialect-agnostic.

## Row counts per year

### Batters (pa_or_ip = PA > 0)

| Season | Rows | Non-null WAR | Mean WAR | Std WAR |
| ------ | ---- | ------------ | -------- | ------- |
| 2010   | 944  | 944          | 0.62     | 1.51    |
| 2011   | 934  | 934          | 0.63     | 1.52    |
| 2012   | 961  | 961          | 0.61     | 1.50    |
| 2013   | 948  | 948          | 0.62     | 1.52    |
| 2014   | 947  | 947          | 0.62     | 1.51    |
| **Total** | **4,734** | **4,734 (100%)** | | |

### Pitchers (pa_or_ip = IP > 0)

| Season | Rows | Non-null WAR | Mean WAR | Std WAR |
| ------ | ---- | ------------ | -------- | ------- |
| 2010   | 635  | 635          | 0.65     | 1.46    |
| 2011   | 662  | 662          | 0.62     | 1.39    |
| 2012   | 661  | 661          | 0.62     | 1.31    |
| 2013   | 678  | 678          | 0.60     | 1.40    |
| 2014   | 691  | 691          | 0.59     | 1.35    |
| **Total** | **3,327** | **3,327 (100%)** | | |

**No thin or missing years.** Batter counts (~940/yr) and pitcher counts
(~665/yr) are uniform across 2010-2014 and consistent with the 2015-2019
range in production (~960-990 batters, ~735-830 pitchers per year — the
pitcher count trend is real, a league-wide effect from rising staff sizes,
not a data artifact).

WAR distributions are stable: batter mean ~0.62 WAR (std ~1.51), pitcher
mean ~0.61 WAR (std ~1.39), centered near zero-ish as expected for a
replacement-level baseline across every player with non-trivial PA/IP.

## Schema drift: bWAR source vs DuckDB `season_*_stats`

The staged parquets use the **long-format projection** already in use by
`scripts/backfill_fwar.py`: `(player_id, player_name, season, position_type,
war, pa_or_ip, war_source)` — 7 columns. They intentionally do NOT mirror the
29-column `season_batting_stats` / 30-column `season_pitching_stats` wide
schemas, because:

1. The purpose here is to extend `war` only (plus the keys needed to merge).
2. bWAR does not publish the advanced FanGraphs metrics (`wrc_plus`, `xwoba`,
   `stuff_plus`, `xfip`, `siera`, Statcast quality-of-contact stats, etc.).
3. The DuckDB tables use NULL-tolerant columns; a WAR-only merge is fine.

### Columns present in bWAR source but not in DuckDB season tables

bWAR carries 49 batter columns / 43 pitcher columns. Beyond what we project,
these would be lost (by design) if we ever widened the merge:

- **Batter bWAR extras:** `runs_bat`, `runs_br`, `runs_dp`, `runs_field`,
  `runs_infield`, `runs_outfield`, `runs_catcher`, `runs_good_plays`,
  `runs_defense`, `runs_position`, `runs_replacement`, `WAA`, `WAR_def`,
  `WAR_off`, `WAR_rep`, `teamRpG`, `oppRpG`, `waa_win_perc`, `OPS_plus`,
  `TOB_lg`, `TB_lg` (bWAR decomposition — useful if we ever want position-
  adjusted defensive WAR or want to reconcile fWAR vs bWAR disagreement).
- **Pitcher bWAR extras:** `IPouts_start`, `IPouts_relief`, `xRA`,
  `xRA_sprp_adj`, `xRA_def_pitcher`, `PPF`, `BIP_perc`, `RpO_replacement`,
  `GR_leverage_index_avg`, `ERA_plus` (relief-vs-start split would be
  especially useful for bullpen-adjusted CausalWAR).

### Columns present in DuckDB season tables but NOT available from bWAR

These are the metrics we could NOT backfill from bWAR, period. Any CausalWAR
feature built on these will have NULL features for 2010-2014 rows:

- **Batting (DuckDB wide schema):** `wrc_plus`, `woba`, `babip`, `iso`,
  `k_pct`, `bb_pct`, `hard_hit_pct`, `barrel_pct`, `xba`, `xslg`, `xwoba`.
- **Pitching (DuckDB wide schema):** `fip`, `xfip`, `siera`, `stuff_plus`,
  `location_plus`, `pitching_plus`, `avg_fastball_velo`, `avg_spin_rate`,
  `gb_pct`, `fb_pct`, `ld_pct`, `xba`, `xslg`, `xwoba`.

**Statcast features (`xwoba`, `xba`, `xslg`, `avg_fastball_velo`, `avg_spin_rate`,
`barrel_pct`, `hard_hit_pct`) are not recoverable pre-2015** — Statcast data
collection began mid-2015. This is a hard physical limit, not a scraping
limit. CausalWAR's proxy features would need to fall back to traditional
metrics (BA/OBP/SLG + ERA/WHIP) for 2010-2014 rows.

The standard FanGraphs advanced metrics (`woba`, `wrc_plus`, `fip`, `xfip`,
`siera`, `k_pct`, `bb_pct`, `babip`, `iso`) **are** available from FanGraphs
for 2010-2014 — but only if the 403 block is resolved (e.g., via a
non-`pybaseball` fetch or a FanGraphs key). They are NOT needed for the
immediate CausalWAR CI-tightening objective, which only consumes `war`.

## Back-of-envelope CausalWAR CI impact

Current state (2015-2025): 11 season-pairs → Buy-Low hit rate 68% CI of
~±4.5 pct pts on the contrarian bucket.

Extended state (2010-2025, after merge): 15 season-pairs — 4 new fully-
interior pairs (2010→11, 2011→12, 2012→13, 2013→14) plus the 2014→2015
boundary pair. Under i.i.d. bootstrap assumptions, CI width scales as
`n_pairs^(-1/2)`:

- **Before:** sqrt(1/11) ≈ 0.302 → CI half-width ∝ 0.302
- **After:**  sqrt(1/15) ≈ 0.258 → CI half-width ∝ 0.258
- **Ratio:** 0.258 / 0.302 ≈ **0.854** — CI shrinks by ~14.6%

So ±4.5 pct pts → **~±3.8 pct pts** expected half-width. Slightly less
than the naive ±3.3 pct pts in the scope note because the added pairs are
4, not 5 (2014→2015 is already in the 2015-2025 window).

**Caveats:**

1. **Regime non-stationarity.** The Statcast / juiced-ball / 3-batter-rule
   regime shifts (2015+, 2019, 2020) make 2010-2014 pairs partially
   out-of-distribution. If the Buy-Low mechanism is regime-conditional,
   adding pre-regime pairs could either tighten the CI or reveal a
   mechanism break — the empirical result is the useful answer. See
   `scripts/causal_war_contrarian_stability.py` for the per-year breakdown
   currently used to diagnose this.
2. **Feature NULL-ing.** If CausalWAR features beyond `war` (e.g., `xwoba`)
   are gated-required, the 2010-2014 rows will be dropped from the model
   fit, and the CI won't tighten. Verify the model's feature set vs the
   "columns NOT available from bWAR" list above before merging.
3. **bWAR vs fWAR replacement-level drift.** The 2015-2024 production WAR
   is `bref_bwar`, and so is this staging, so no dialect break.

## Next steps (gated, NOT yet authorized)

1. Review audit JSON: `data/staging/season_war_2010_2014_audit.json`.
2. Confirm CausalWAR is robust to pre-Statcast feature NULL-ing (or decide
   which features to impute / drop for the 2010-2014 rows).
3. Backfill `players.mlbam_id` / `player_id` coverage for 2010-2014 players
   not in the current DB (pre-backfill check will surface the unmatched set,
   same pattern as `backfill_fwar.match_to_db_players`).
4. Merge via a small adapter script (pattern: `backfill_fwar.merge_into_db`)
   that UPDATEs `season_*_stats.war` for `season IN (2010..2014)`.
5. Re-run `scripts/causal_war_contrarian_stability.py` on the widened
   window and compare the 68% CI.
