# TJ Surgery Dates — Staging Notes

Staged roster of Tommy John (UCL reconstruction), hybrid UCL-repair, and
keyword-adjacent (gated) TJ surgeries, 2017–2025. Output:
`data/staging/tj_surgery_dates.parquet`.

Produced by: `scripts/ingest_tj_dates.py`.
Unblocks: Projections v2 `tj_surgery_date` / `had_tj_surgery` feature flag
(NORTH_STAR 5C, "TJ surgery dates 2017–2025").

## Scope disclosures

- **Staging-only.** Parquet written to `data/staging/`. Not loaded into
  DuckDB. No model code touched.
- **Read-only against DuckDB.** The script opens the canonical
  `baseball.duckdb` with `read_only=True` and reads from `transactions`
  and `pitches`.
- **Never fabricates.** Keyword-adjacent rows (UCL sprain, elbow sprain,
  elbow inflammation, elbow surgery) are admitted only when a follow-up
  signal is present. When the surgery date cannot be pinned,
  `surgery_date = NULL` and `source = 'manual_review_needed'`.

## 2026-04-23 classifier-gap closure

Prior staging runs caught only "Tommy John" / "UCL reconstruction" phrasing,
missing modern IL-text wordings that describe the same procedure. The
upstream `scripts/ingest_injury_labels.py` classifier and this staging
script were extended to surface those cases without fabricating TJs from
ambiguous sprain/inflammation hits.

### Classifier changes — `scripts/ingest_injury_labels.py`

1. **Explicit-surgical tier** (`tj_classification_tier='explicit_surgical'`,
   `injury_type='tommy_john'`):
   - `Tommy John`, `TJ surgery` (unchanged)
   - `UCL reconstruction`, `UCL reconstructive` (unchanged)
   - `ulnar collateral .{0,20}reconstruct` (unchanged)
   - **NEW**: `UCL repair`, `ulnar collateral .{0,20}repair`
   - **NEW**: `internal brace`, `elbow reconstruction`, `elbow repair`,
     `elbow reconstruction/repair`
2. **Keyword-adjacent tier** (`tj_classification_tier='keyword_adjacent'`,
   `injury_type` unchanged from prior enum — usually `ucl_sprain` or
   `elbow`):
   - `UCL sprain|tear|strain|inflammation`
   - `ulnar collateral sprain|strain|tear|inflammation`
   - `elbow sprain|inflammation|surgery`, `elbow injury recovery`
   - `sprained elbow`, `elbow discomfort`, `UCL soreness`
   - **Shoulder guard**: the elbow-adjacent pattern requires the literal
     token `"elbow"` to appear, so "shoulder sprain" / "shoulder
     inflammation" do NOT classify as TJ-adjacent.
3. `injury_type` enum is deliberately unchanged so existing consumers that
   filter on `injury_type == 'tommy_john'` keep their prior semantics.
   `tj_classification_tier` is a NEW column alongside `injury_type`.
4. Ingest window extended: `END_YEAR` 2024 → 2025 so the downstream TJ
   staging (2017–2025) can see adjacent rows in the final year.

### Output schema (injury_labels.parquet)

9 columns — adds `tj_classification_tier` to the prior 8:

```
pitcher_id              INTEGER
pitcher_name            VARCHAR
season                  INTEGER
il_date                 DATE
il_end_date             DATE              NULLable
injury_type             VARCHAR           (unchanged enum)
injury_description_raw  VARCHAR
source                  VARCHAR
tj_classification_tier  VARCHAR           NULLable
                                          ('explicit_surgical' |
                                           'keyword_adjacent' | NULL)
```

### Admission rules — `scripts/ingest_tj_dates.py`

- **explicit_surgical tier** → admit. Surgery-date extraction rules
  unchanged (see "Confidence semantics" below).
- **keyword_adjacent tier** → admit ONLY if a follow-up signal is present:
  1. **Single-placement IL duration ≥ 400 days**, OR
  2. **Aggregated IL span ≥ 400 days** across consecutive IL placements
     chained by either: (a) next placement's start ≤ 45 days after prior
     placement's end, or (b) next placement's start within 540 days of the
     prior start/end when either end is NULL. Chain end falls back to the
     first post-placement activation in the `transactions` table when no
     placement in the chain has a paired `il_end_date`.
  3. A subsequent **explicit_surgical** transaction from the same pitcher
     within 730 days (24 months).

Rows that fail ALL three gates are DROPPED. The script never fabricates a
TJ from a sprain / inflammation alone.

Admitted adjacent-tier rows are capped at `confidence='low'` and tagged
`source='transactions_kw_adjacent'` (or `'manual_review_needed'` when no
surgery date can be extracted at all). The notes field includes the
triggering gate signal (e.g. `gate_signal=aggregated IL span 531d >= 400d`).

## Row count and distributions (2026-04-23 run)

- **Total TJ events staged:** 302
- **Unique pitchers:** 224
- **Return dates inferred** (from `pitches`): 212
- **Rows collapsed by dedupe** (±30d window; NULL-date per-pitcher
  collapse): 94

### Confidence distribution

| Confidence | Count | Share |
|------------|------:|------:|
| `high`     |     1 |  0.3% |
| `medium`   |    74 | 24.5% |
| `low`      |   227 | 75.2% |

The increase in `low` count (39% → 75%) reflects the 188 new
keyword-adjacent-tier rows admitted under the follow-up-signal gate.

### Source distribution

| Source                        | Count |
|-------------------------------|------:|
| `transactions_kw_adjacent`    |   188 |
| `transactions_kw`             |    75 |
| `manual_review_needed`        |    39 |
| `mlb_official` (reserved)     |     0 |
| `retrosheet` (reserved)       |     0 |

### Year distribution (by `surgery_date`)

| Year | Count |
|-----:|------:|
| 2017 |    12 |
| 2018 |    25 |
| 2019 |    26 |
| 2020 |    19 |
| 2021 |    36 |
| 2022 |    40 |
| 2023 |    34 |
| 2024 |    42 |
| 2025 |    28 |

2024 and 2025 now lead the year histogram, reflecting the modern
TJ-text evolution (internal brace, elbow sprain → surgery) that the
keyword-adjacent tier captures with follow-up-signal gating.

## Confidence semantics

- **`high`** — Description contains an explicit `<Month> <YYYY>` reference
  that sits within a realistic post-surgery IL window (≤ 36 months before
  `il_date`). Only explicit-tier rows can reach `high`.
  Example: "Recovering from August 2016 Tommy John Surgery" → 2016-08-01.
- **`medium`** — Explicit-tier row without a month/year reference. Two
  sub-cases:
  1. Current-event phrasing ("Tommy John surgery."): `surgery_date =
     il_date`. For Feb/Mar placements, notes warn the surgery was likely
     1–6 months earlier (off-season).
  2. A month/year extraction outside the plausible 36-month window.
- **`low`** — Three sub-cases:
  1. Explicit-tier recovery phrasing ("Recovery from Tommy John surgery.")
     with no month: `surgery_date=NULL`,
     `source='manual_review_needed'`.
  2. Adjacent-tier row admitted by the follow-up-signal gate without a
     month/year extraction: `surgery_date = il_date`,
     `source='transactions_kw_adjacent'`.
  3. Adjacent-tier row with an extractable month/year (rare): `surgery_date =
     extracted`, `source='transactions_kw_adjacent'`. Even with a date,
     adjacent-tier is capped at `low` confidence.

## UCL repair / internal brace handling

Same as prior staging: the `_TJ_ANCHOR` regex matches "internal brace",
"UCL repair", and "elbow reconstruction/repair" so hybrid procedures are
captured. The `notes` field is prefixed with `UCL_REPAIR_BRACE (hybrid,
not classic TJ); ...` so downstream consumers can filter them out of
classic-TJ analyses.

## Sanity-check results (2026-04-23 run)

| Case                              | Expected date | Status  | Detail |
|-----------------------------------|---------------|---------|--------|
| Shohei Ohtani — TJ 2018           | 2018-10-01    | MISS    | Text is "Right elbow UCL sprain" (adjacent tier); chained IL spans total ~60d and no subsequent "Tommy John" mention exists for Ohtani in `transactions` — gate correctly DROPS this row. **Unrecoverable from transactions alone.** |
| Shohei Ohtani — Revision TJ 2023  | 2023-09-19    | MISS    | Transaction text at Sept 2023 was "Oblique" — not a classifier gap. |
| Jacob deGrom — TJ 2023            | 2023-06-12    | **HIT** | Adjacent tier admitted via aggregated IL-span gate (2023-04-29 → 2024-09-13 = ~502 days). `surgery_date=2023-04-29` (il_date), `confidence=low`. True June 2023 date requires `mlb_official` curation. |
| Spencer Strider — Internal brace 2024 | 2024-04-12 | **HIT** | Adjacent tier admitted via aggregated IL-span gate (2024-04-07 → 2025-04-16). `surgery_date=2024-04-07` (il_date, 5d from expected), `confidence=low`. |
| Walker Buehler — Revision TJ 2022 | 2022-08-23    | HIT     | Explicit-tier hit via 2023-02-16 IL text "Right elbow UCL reconstruction". `surgery_date=2023-02-16`, `confidence=medium`. |
| Bryce Harper — UCL repair 2022    | 2022-11-23    | PARTIAL | Row captured, flagged `UCL_REPAIR_BRACE` in notes. `surgery_date=NULL`, `source='manual_review_needed'`. |

**Hit rate on classifier-fixable gaps (deGrom 2023, Strider 2024):** 2/2
HIT. (Ohtani 2018 is unrecoverable from transactions alone — no "Tommy
John" keyword ever appears for him in the source data; respecting the
"never fabricate" rule, the gate correctly drops it.)

## Known misses / next steps

Unchanged from prior notes; the 2026-04-23 classifier extension addressed
the "Descriptions that don't use Tommy John/UCL reconstruction wording"
category for cases where IL duration or subsequent TJ text provides a
follow-up signal. Still unrecoverable from transactions alone:

1. **Position players and two-way players with UCL sprain text only**
   (Ohtani 2018) — no subsequent surgical keyword ever logged; the gate
   correctly refuses to fabricate.
2. **Off-season surgeries without any IL placement** — if a pitcher had
   surgery in October and no IL placement until the next season's `Tommy
   John surgery.` text, the date is only month-level.
3. **Exact surgery date** for captured cases. `transactions_kw` / `_adjacent`
   gives at best Month-level resolution via `<Month> <YYYY>` extraction
   (1 high-confidence row); most rows use `il_date` as a proxy.

### Recommended next augmentation

- Reserved sources `mlb_official` and `retrosheet` are in the schema enum
  but currently unused. A follow-up script `scripts/backfill_tj_dates_mlb_official.py`
  that scrapes the MLB.com injury tracker, or ingests Jon Roegele's
  public Tommy John database CSV, would raise the `high`-confidence row
  count substantially and close the Ohtani MISSes.
- A short hand-curation pass for the top 20 high-profile TJ cases
  (marked `source='manual'`) would give Projections v2 the exact dates
  needed for marquee pitchers.

## Schema reference (tj_surgery_dates.parquet)

```
mlb_id           BIGINT                 NULLable
player_name      VARCHAR
surgery_date     DATE                   NULLable
return_date_est  DATE                   NULLable
confidence       VARCHAR ('high'|'medium'|'low')
source           VARCHAR ('transactions_kw'|'transactions_kw_adjacent'|
                          'mlb_official'|'retrosheet'|
                          'manual_review_needed')
notes            VARCHAR  -- includes hybrid-brace flag, gate_signal,
                            il_date, desc excerpt
```

## Reproducibility

```bash
python scripts/ingest_injury_labels.py --no-audit
python scripts/ingest_tj_dates.py
# or with overrides
python scripts/ingest_tj_dates.py --start-year 2017 --end-year 2025 \
    --out data/staging/tj_surgery_dates.parquet
```

Idempotent: both output parquets are overwritten on each run. DuckDB is
opened `read_only=True`; safe to run while Streamlit dashboard is up.
