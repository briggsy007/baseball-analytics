# Umpire data integration notes

Source: `scripts/ingest_umpire_data.py` (2026-04-23).

This document covers the proposed DuckDB schema addition for the two
parquet files staged under `data/staging/`, the downstream wiring plan for
CausalWAR (confounder addition) and PitchGPT (context dimension
addition), and the known integration risks.

## Staged artifacts

| File | Rows | Grain |
|------|------|-------|
| `data/staging/umpire_assignments.parquet` | 103,192 | (game_pk, position, umpire_id, umpire_name) |
| `data/staging/umpire_tendencies.parquet`  | 1,104 | (umpire, season) |

### `umpire_assignments.parquet` columns

| Column | Type | Notes |
|--------|------|-------|
| `game_pk` | INT64 (nullable) | MLBAM game primary key. NULL for Retrosheet games that don't appear in `pitches` (see "Coverage gaps" below). |
| `position` | VARCHAR | One of `HP`, `1B`, `2B`, `3B`, `LF`, `RF`. LF/RF only present on postseason / All-Star games. |
| `umpire_id` | VARCHAR | Retrosheet umpire ID (e.g. `kulpr901`). NULL for umpscorecards HP rows. |
| `umpire_name` | VARCHAR | Canonical display name (e.g. `Ron Kulpa`). |
| `date` | DATE | Game date. |
| `season` | INT32 | Calendar year of `date`. |
| `source` | VARCHAR | `retrosheet` or `umpscorecards`. HP assignments prefer umpscorecards when both exist. |

Coverage by season (HP position only, games matched to `pitches.game_pk`):

| Season | Pitches games | HP matched | Coverage |
|--------|---------------|------------|----------|
| 2015   | 2,464 | 2,465 | 100% |
| 2016   | 2,462 | 2,463 | 100% |
| 2017   | 1,683 | 2,468 | data gap in pitches* |
| 2018   | 2,464 | 2,464 | 100% |
| 2019   | 1,756 | 2,466 | data gap in pitches* |
| 2020   | 951   | 951   | 100% |
| 2021   | 2,612 | 2,466 | 94%   |
| 2022   | 2,702 | 2,470 | 91%   |
| 2023   | 2,450 | 2,471 | 100% |
| 2024   | 2,560 | 2,473 | 97%   |
| 2025   | 2,553 | 2,476 | 97%   |
| 2026   |   393 |   353 | 90% (in progress) |

\* Shortfall relative to umpscorecards in 2017 / 2019 reflects missing
Statcast rows in the project's `pitches` table, not a gap in umpire data.
These rows currently have NULL `game_pk` but are joinable on `(date,
home_team, game_slot)` if `pitches` is later backfilled.

### `umpire_tendencies.parquet` columns

| Column | Type | Notes |
|--------|------|-------|
| `umpire` | VARCHAR | Display name. |
| `season` | INT32 | |
| `games` | INT32 | Number of HP-ump games aggregated. |
| `called_pitches` | INT64 | Sum across games. |
| `called_correct` | INT64 | Sum across games. |
| `called_correct_rate` | FLOAT | `called_correct / called_pitches`. |
| `overall_accuracy_wmean` | FLOAT | Weighted by called_pitches. |
| `x_overall_accuracy_wmean` | FLOAT | Expected accuracy given Statcast pitch locations. |
| `accuracy_above_x_wmean` | FLOAT | Actual minus expected accuracy. Positive = above-average ump. |
| `consistency_wmean` | FLOAT | Weighted by called_pitches. |
| `favor_wmean` | FLOAT | Weighted. Positive = batter-favoring, negative = pitcher-favoring. |
| `total_run_impact_mean` | FLOAT | Per-game average absolute run impact of blown calls. |
| `batter_impact_mean` | FLOAT | Per-game mean batter-run impact (home+away averaged). |
| `pitcher_impact_mean` | FLOAT | Per-game mean pitcher-run impact. |

### Zone-boundary data — not staged

**Decision:** per-umpire zone-shape deltas (e.g. high strike bias, inside
corner bias) were not staged in this pass. The umpscorecards `/api/games`
endpoint exposes abstract accuracy / favor metrics but not the raw called
strike vs. called ball geometry needed to rebuild a zone map. Rebuilding
that at the platform level is feasible from our own `pitches` table
(joining `zone`, `plate_x`, `plate_z`, `description` against the new
`umpire_name`) — but it belongs in a follow-up analytics module, not in
this ingest. See "Follow-up work" below.

## Proposed DuckDB schema addition

When ready to land in DuckDB (single-writer rule — stop the dashboard
first), add two tables to `src/db/schema.py` via new private helpers:

```python
def _create_umpire_assignments(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS umpire_assignments (
            game_pk      INTEGER,
            position     VARCHAR,            -- HP | 1B | 2B | 3B | LF | RF
            umpire_id    VARCHAR,            -- Retrosheet ID (nullable)
            umpire_name  VARCHAR,
            date         DATE,
            season       INTEGER,
            source       VARCHAR,            -- retrosheet | umpscorecards
            PRIMARY KEY (game_pk, position, umpire_name)
        );
    """)


def _create_umpire_tendencies(conn) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS umpire_tendencies (
            umpire                     VARCHAR,
            season                     INTEGER,
            games                      INTEGER,
            called_pitches             BIGINT,
            called_correct             BIGINT,
            called_correct_rate        FLOAT,
            overall_accuracy_wmean     FLOAT,
            x_overall_accuracy_wmean   FLOAT,
            accuracy_above_x_wmean     FLOAT,
            consistency_wmean          FLOAT,
            favor_wmean                FLOAT,
            total_run_impact_mean      FLOAT,
            batter_impact_mean         FLOAT,
            pitcher_impact_mean        FLOAT,
            PRIMARY KEY (umpire, season)
        );
    """)
```

Plus an index on `umpire_assignments(game_pk)` and
`(umpire_name, season)` so that per-pitch joins are cheap.

The loader (call it `scripts/load_umpire_data.py` or add to
`src/ingest/umpire_loader.py`) should:

1. Stop the dashboard (user-initiated — document prominently).
2. Open `get_connection(read_only=False)`.
3. Read the two parquets from `data/staging/`.
4. `INSERT OR REPLACE` into the tables (DuckDB syntax:
   `INSERT INTO ... ON CONFLICT REPLACE` — or `DELETE + INSERT` inside a
   transaction).
5. Emit a freshness row into `data_freshness`.

## Downstream wiring plans (not executed in this ingest)

### CausalWAR confounder addition

`src/analytics/causal_war.py` lines 881-895 define the `confounder_cols`
list that feeds the FWL residualization. The known risk flag the spec
cites — *"missing umpire identity, time-of-season drift"* — is exactly
what the assignments + tendencies tables address.

**Proposed edit (do not execute in this scope):**

1. Add a join in the data-build step that attaches
   `umpire_tendencies.accuracy_above_x_wmean` and `favor_wmean` per pitch
   via `(game_pk -> umpire_assignments WHERE position='HP') -> umpire ->
   umpire_tendencies (season)`.
2. Add the two scalars to the `confounder_cols` list so they enter the
   `W` matrix. No umpire identity one-hot — season-level tendencies
   capture what matters without adding ~100 dummy columns.
3. Re-run the held-out CausalWAR backtest (existing spec gates at
   `docs/models/causal_war_validation_spec.md`). If bWAR-correlation
   improves materially after adding umpire confounders, that's a clean
   methodology paper claim; if not, the result itself is a publishable
   honesty-note ("umpire variation does not meaningfully confound
   player-level run values at season aggregation").

**Integration risk:** Umpire assignment is *pre-treatment* with respect
to a player's batting performance (the umpire does not depend on the
batter), so including it as a confounder is theoretically sound. The
concern is **time-of-season drift** — umpire assignments cluster by
month (e.g. umpires who work only spring / summer), so per-umpire
season-level tendencies can absorb a month effect that CausalWAR already
captures via `month`. Mitigation: keep `month` in the confounder list
*and* add umpire tendencies, let cross-validation detect collinearity.

### PitchGPT context dimension addition

`src/analytics/pitchgpt.py` token/context construction (around line 365,
where `score_diff` was previously hard-coded to 0 and has since been
fixed) is where a new scalar/categorical context feature would enter.

**Proposed edit (do not execute in this scope):**

1. Add a per-pitch `ump_accuracy_above_x` float via the same join chain
   as CausalWAR (game_pk -> HP umpire -> umpire_tendencies by season).
2. Bin or z-score the value and concatenate into the existing context
   vector so the transformer can condition next-pitch predictions on
   umpire tightness (pitchers on a tighter ump may throw more zone-edge
   pitches; batters may swing earlier).
3. Re-run the 2025 OOS perplexity evaluation. If umpire context adds
   ≥1% perplexity improvement over the current tokens+context baseline,
   log it and narrow the calibration claim accordingly. If not, accept
   the honest null — consistent with the existing "context adds 1–7%
   lift" Path-2 finding.

**Integration risk:** The `umpire_tendencies` season aggregation means
early-season pitches effectively use *prior-season* umpire behavior
(because we haven't observed enough current-season calls to aggregate
yet). This is a minor leakage concern only if the model is evaluated on
the same season as training; for the pitcher-disjoint 2025 holdout it's
clean. Document in any paper: "umpire tendencies are the umpire's
**prior-season** aggregate, not current-season — this avoids
look-ahead leakage in out-of-sample evaluation."

## Coverage gaps and known limitations

1. **2026 Retrosheet unpublished.** Retrosheet posts `gl{YEAR}.zip` after
   the season ends. 2026 returns 404. Base umpires (1B/2B/3B) are
   therefore NULL for 2026 games — HP umpire (umpscorecards) is present
   for the ~353 games played through 2026-04-19. This will resolve after
   the 2026 season concludes; the script is idempotent and will back-fill
   on next run post-publication.
2. **2017 / 2019 pitches-table data gaps.** 782 2017 and 710 2019 games
   are in umpire data but not in `pitches`. The rows ship with NULL
   `game_pk` and a `date + home_team_mlbam` fallback for future joins.
3. **Umpscorecards has no stable umpire ID.** The service returns only
   display names. We preserve the Retrosheet ID where possible via the
   Retrosheet HP row, but umpscorecards-only rows (e.g. 2017 where
   Retrosheet didn't match) have `umpire_id=NULL`. Name-based joins to
   tendencies work — name strings are stable across the two sources
   (verified spot-check: "Ron Kulpa" appears identically).
4. **Doubleheaders resolved by chronological order.** The (date, home_team,
   game_slot) join assumes MLBAM and Retrosheet order doubleheader games
   the same way. Spot-check on 2023-04-18 CHA vs PHI confirms. Edge case:
   suspended / resumed games that MLBAM numbers as two game_pks but
   Retrosheet treats as one — these may mis-match; impact is <5 games per
   season.
5. **Umpscorecards season aggregates include postseason.** The
   `umpire_tendencies` rows mix regular-season and postseason games for
   the HP umpire; if regular-season-only tendencies are needed, filter on
   `umpire_assignments.game_pk -> games.game_type='R'` after loading to
   DuckDB.

## Follow-up work

1. **Per-pitch zone-boundary reconstruction from `pitches`.** Using
   `plate_x`, `plate_z`, `zone`, `description`, join umpire_name via HP
   assignment, then fit a per-umpire called-strike probability surface.
   This produces true zone-tendency maps (not just accuracy aggregates)
   and is a stronger CausalWAR confounder than the current scalars.
   Estimated effort: 1 analytics agent batch.
2. **Retrosheet backfill for 2017 / 2019 pitches gap.** The `pitches`
   table is short ~780 + ~710 games for those seasons; fill from the
   pybaseball Statcast weekly archives, then re-run
   `ingest_umpire_data.py` to raise coverage to 100%.
3. **Crew / crew-chief attribution.** Retrosheet's `ump_home_id` is
   sometimes the crew chief, sometimes not; adding a crew-chief column
   from a curated MLB source would allow crew-level effects (some crews
   call tighter strike zones as a unit).

## Reproducibility

Re-running the ingest is idempotent: Retrosheet zips are cached under
`data/staging/retrosheet_cache/` and umpscorecards (a small JSON dump,
~28 MB) is refreshed each run. Total runtime ≈ 15 seconds on a warm
cache, ~1 minute cold. Pass `--refresh` to force re-download.
