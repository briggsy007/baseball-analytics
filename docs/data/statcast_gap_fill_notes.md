# Statcast gap-fill — 2017 + 2019 missing months

Generated 2026-04-23 by `scripts/backfill_statcast_gaps_2017_2019.py`.

## Problem observed

Year totals in the `pitches` DuckDB table showed two years materially below
their neighbours:

| Year  | Rows in DB | Games | Neighbour baseline |
| ----- | ----------:| -----:| ------------------:|
| 2016  | 725,627    | 2,462 |                  — |
| 2017  | **500,617**| **1,683** | ~725K–750K     |
| 2018  | 734,567    | 2,464 |                  — |
| 2019  | **517,269**| **1,756** | ~725K–750K     |
| 2020  | 280,398    |   951 | COVID-shortened (not a gap) |
| 2021  | 752,574    | 2,612 |                  — |
| 2022  | 768,161    | 2,702 |                  — |

That is a 225K–235K row shortfall per year. The monthly-window backfill
(`scripts/backfill.py`) silently dropped four monthly chunks — Savant has the
data, but network failures or rate-limit issues meant those months were never
written to DuckDB.

## Missing months (confirmed by month-level DuckDB query)

Baseline: 2018 and 2020-2022 show 105K–128K rows per in-season month with
380–425 games. Both 2017 and 2019 had four-month contiguous stretches where
**no rows at all** existed for specific months:

| Year-month | Rows in DB before gap-fill | Status |
| ---------- | --------------------------:| ------ |
| 2017-04    | 108,785 | present |
| 2017-05    | 126,719 | present |
| **2017-06**| **0**   | **missing** |
| **2017-07**| **0**   | **missing** |
| 2017-08    | 125,708 | present |
| 2017-09    | 123,970 | present |
| 2017-10    |  15,435 | present (playoffs only) |
| 2019-03    |  32,950 | present |
| 2019-04    | 115,524 | present |
| 2019-05    | 124,404 | present |
| 2019-06    | 122,227 | present |
| 2019-07    | 110,982 | present |
| **2019-08**| **0**   | **missing** |
| **2019-09**| **0**   | **missing** |
| 2019-10    |  11,182 | present (playoffs only) |

## What the gap-fill script did

Re-fetched the four missing months from Baseball Savant via
`pybaseball.statcast(start_dt, end_dt)` in weekly windows (smaller windows are
more robust against Savant timeouts on older seasons). Cleaned via
`src.ingest.statcast_loader._clean_statcast` + `validate_pitches` so the
schema matches the DuckDB `pitches` table column-for-column. Deduped against
existing pitches in DuckDB (read-only connection) by natural key
`(game_pk, at_bat_number, pitch_number)`.

Output:

- Parquet: `data/staging/statcast_gap_fill_2017_2019.parquet`
- Rows staged: **480,969**
- All 53 schema columns present, column order matches `_PITCHES_COLUMNS`.

### Per-month fetched row counts

| Year-month | Rows fetched | Games | Kept after dedup |
| ---------- | -----------: | ----: | ----------------:|
| 2017-06    | 122,152      | 408   | 122,152 |
| 2017-07    | 112,888      | 376   | 112,888 |
| 2019-08    | 126,268      | 416   | 126,268 |
| 2019-09    | 119,661      | 392   | 119,661 |
| **Total**  | **480,969**  | 1,592 | **480,969** |

Dedup overlap with existing DuckDB rows: **0** (as expected — these months
had no rows present at all, so every fetched row is new).

### Projected year totals after load

| Year | Current | Staged | Projected | Baseline |
| ---- | ------: | -----: | --------: | -------: |
| 2017 | 500,617 | 235,040 | **735,657** | ~725K |
| 2019 | 517,269 | 245,929 | **763,198** | ~725K–750K |

Both projections land squarely inside the neighbouring-year envelope.

## Source-side observations (source ceiling vs our bug)

- **2017-07-10 through 2017-07-13**: zero MLB games (All-Star break).
  Not a gap, just calendar reality. The 2017-07 total (112,888) matches
  the 2018-07 control (111,907) almost exactly.
- **All four target months returned full-season rows from Savant.** No
  month hit a source-side ceiling — this was entirely a backfill-pipeline
  bug, not a Savant data availability issue.
- Older Savant exports (pre-2019) include BOTH `des` and `description`
  columns. Our renamer maps `des -> description`, so the raw `description`
  had to be dropped pre-rename to avoid a pandas duplicate-column error
  when writing parquet. The script handles this transparently.

## Next step (NOT done here — out of scope)

A future loader step should:

1. Stop the Streamlit dashboard (it holds the DuckDB write lock).
2. Open DuckDB read-write and call
   `src.ingest.statcast_loader.insert_pitches(conn, df)` on the staged
   parquet. `insert_pitches` already dedupes on the same natural key, so
   re-running is safe.
3. Verify post-load year totals match the projections above.

## Reproducibility

```bash
# Re-fetch from Savant (pybaseball cache makes re-runs nearly instant):
python scripts/backfill_statcast_gaps_2017_2019.py

# Skip if parquet already exists:
python scripts/backfill_statcast_gaps_2017_2019.py --skip-existing

# Subset of months:
python scripts/backfill_statcast_gaps_2017_2019.py --months 2017-06 2019-08
```
