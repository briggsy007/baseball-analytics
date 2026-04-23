# Game-Time Weather Integration Notes

Staging pipeline: `scripts/ingest_weather_conditions.py` -> `data/staging/game_weather.parquet`.

Task brief (2026-04-23): ingest NOAA historical weather per game so CausalWAR
can add weather as a confounder and DPI can add wind as a fly-ball feature.
This file documents coverage, integration plan, and caveats.

## TL;DR

- One row per distinct `game_pk` in `pitches` (N ~= 25k games, 2015-2026-04).
- Fields: `temp_f`, `wind_speed_mph`, `wind_dir_deg`, `humidity_pct`,
  `pressure_mb`, interpolated to scheduled game start time (UTC).
- `roof_status` is one of `open` / `closed` / `dome` / `retractable_unknown`.
- Source: `meteostat` hourly NOAA reanalysis, nearest station to park coords.
- **Staging only — never written to DuckDB by this script.**

## Coverage (full ingest, 2026-04-23)

Ran against 25,050 distinct `game_pk` values in `pitches` (2015-04-05 to
2026-04-19).

| Metric | Value |
|---|---|
| Distinct games in `pitches` | 25,050 |
| Venue resolved via MLB StatsAPI | 25,050 (100.0%) |
| `game_datetime_utc` resolved | 25,050 (100.0%) |
| Weather attached (temp/wind/etc populated) | 23,607 (94.2%) |
| Dome-skipped (Tropicana, fixed roof) | 733 |
| Retractable-unknown (weather still attached) | 5,289 |
| Unknown venue (spring-training / exhibition, NULL roof) | 710 |
| Station distance p50 | 7.2 km |
| Station distance p95 | 20.0 km |
| Station distance max | 29.3 km |
| temp_f distribution (mean / std) | 74.1 / 12.4 °F |
| temp_f range | 26 - 116 °F |

**Unknown-venue audit (spring-training venues, not in registry by design):**
Sahlen Field, Salt River Fields, Goodyear Ballpark, Peoria Stadium, TD
Ballpark, Surprise Stadium, Sloan Park, Camelback Ranch, Publix Field (ex-
Joker Marchant), Scottsdale Stadium, Clover Park, LECOM Park, BayCare
Ballpark, Hohokam Stadium, JetBlue Park, Ed Smith Stadium, CoolToday Park,
Charlotte Sports Park, Ballpark of the Palm Beaches (FITTEAM / CACTI aliases),
Tempe Diablo Stadium, UNIQLO Field at Dodger Stadium (exhibition), Hammond
Stadium, Lee Health Sports Complex, Spectrum Field, CenturyLink Sports
Complex, First Data Field, Dunedin Stadium, and similar. These rows have
`roof_status=NULL`, `weather_source=no_metadata`. Regular-season models
should filter on `roof_status IS NOT NULL` or on `venue IN VENUE_REGISTRY`
to exclude them.

## Dome / retractable-roof handling

Per task brief, these are handled explicitly:

| Venue | Current team | Roof | `roof_status` in output |
|---|---|---|---|
| Tropicana Field | Rays (pre-2025) | Fixed dome | `dome` (weather NULL) |
| Rogers Centre | Blue Jays | Retractable | `retractable_unknown` (weather attached) |
| American Family Field | Brewers (2021-) | Retractable | `retractable_unknown` |
| Miller Park | Brewers (2015-2020 alias) | Retractable | `retractable_unknown` |
| Chase Field | Diamondbacks | Retractable | `retractable_unknown` |
| Minute Maid Park | Astros (pre-2025 alias) | Retractable | `retractable_unknown` |
| Daikin Park | Astros (2025-) | Retractable | `retractable_unknown` |
| Globe Life Field | Rangers (2020-) | Retractable | `retractable_unknown` |
| loanDepot park | Marlins (2021-) | Retractable | `retractable_unknown` |
| Marlins Park | Marlins (2015-2020 alias) | Retractable | `retractable_unknown` |
| T-Mobile Park | Mariners (2019-) | Retractable | `retractable_unknown` |
| Safeco Field | Mariners (2015-2018 alias) | Retractable | `retractable_unknown` |

For fixed-dome Tropicana, weather is **deliberately NULL** — outdoor temp/wind/
humidity readings have no meaning inside a sealed dome. `weather_source` is
set to `dome_skipped` to distinguish from failed fetches.

For retractable-roof parks, per the brief we do **not** guess open/closed
state from the schedule endpoint alone. `roof_status='retractable_unknown'`
and the outdoor weather **is** attached so downstream analysts can either:

1. Use the weather as-is (simplest; wind/temp still informative for wind
   decisions and player conditioning).
2. Join later to a roof-status source (MLB GUMBO `gameData.weather.condition`
   sometimes carries "Roof Closed") and selectively zero-out weather for
   closed games. This is a follow-up ticket, not done in this pass.

## Venue coverage (30 current + aliases)

Hard-coded in `VENUE_REGISTRY` in `scripts/ingest_weather_conditions.py`.
Includes two 2025 relocations:

- **Athletics** -> `Sutter Health Park` (West Sacramento, 38.5803, -121.5133),
  pre-2025 venue was `Oakland Coliseum` / `O.co Coliseum` (37.7516, -122.2005).
- **Rays** -> `George M. Steinbrenner Field` (Tampa spring-training site,
  27.9806, -82.5075) for 2025 season after Tropicana roof damage from
  Hurricane Milton. Classified as `open` air since Steinbrenner Field has
  no roof.

Historical alias set covers: `AT&T Park` (Oracle pre-2019), `Globe Life Park
in Arlington` (pre-2020 Rangers), `Angel Stadium of Anaheim` (pre-2016),
`Guaranteed Rate Field` (Rate Field pre-2024), `Marlins Park` (loanDepot
pre-2021), `Miller Park` (American Family pre-2021), `Minute Maid Park`
(Daikin pre-2025), `O.co Coliseum`/`Oakland Coliseum`, `Safeco Field`
(T-Mobile pre-2019), `Turner Field` (Braves pre-2017), `U.S. Cellular Field`
(White Sox 2003-2016). Spring-training / neutral-site games (Seoul dome,
London Stadium, Sydney Cricket Ground, Estadio Monterrey, etc.) will surface
as unknown-venue rows — see the audit section of the summary print.

## Proposed CausalWAR confounder addition

Current CausalWAR confounder set (`src/analytics/causal_war.py` L882):

```python
confounder_cols = [
    "venue_code",      # venue as label-encoded int
    "inning",
    "outs_when_up",
    "stand_code",
    "p_throws_code",
    # ...
]
```

Proposed addition (PR follow-up, not in this pass since brief says no model
changes):

```python
# Weather block. Joined from game_weather.parquet via game_pk -> game_weather.
# Venues without weather (dome) carry NULL; NULL is handled by HistGB natively.
confounder_cols.extend([
    "temp_f",           # Park-adjusted wOBA should drop with extreme heat
                         # (pitcher fatigue) and cold (ball doesn't carry)
    "wind_speed_mph",   # Outfield carry, HR rate
    "wind_dir_deg",     # Out-to-center winds increase HR; in-winds suppress
    "humidity_pct",     # Low humidity + high temp = ball carries further
    "pressure_mb",      # Lower pressure (altitude + storms) = more carry;
                         # Coors effect largely captured by venue_code but
                         # day-to-day barometric variation is net new.
])
```

Cross-fitted `HistGradientBoostingRegressor` is the nuisance model and it
handles NaN natively (treats NaN as its own split direction). No imputation
needed on the DML side. The retractable_unknown rows enter as informative
features for Y|W and T|W, and the venue_code already partitions by park, so
double-counting is not a concern — the model is residualising.

**Expected lift.** Modest. Existing `venue_code` absorbs the park-constant
component; weather adds day-to-day variance around the park mean. The
interesting residual signal is wind × park (e.g. Wrigley Field out-blowing
days) which the current LinearRegression second stage will not capture by
itself. A follow-up ticket should consider allowing the causal model a
venue × wind interaction term.

## Proposed DPI xOut feature

`src/analytics/defensive_pressing.py` fits a HistGB xOut model on batted
balls. Wind materially affects fly-ball carry and should enter the feature
matrix:

```python
# DPI feature additions (batted-ball subset).
# Join game_weather via game_pk.
extra_features = [
    "wind_speed_mph",   # Fly-ball carry distance (primary signal)
    "wind_dir_deg",     # Direction relative to spray angle (derived below)
    "temp_f",           # Air density proxy; cold air -> more drag
    # Derived:
    "wind_parallel_mph",   # cos(wind_dir - spray_angle) * wind_speed_mph
    "wind_perpendicular_mph", # sin(...) * wind_speed_mph
]
```

The derived parallel / perpendicular wind components are the physically
meaningful quantities — a 15 mph wind out-to-center inflates HR rate only
in the 'parallel' direction of the batted ball's spray angle. Without this
decomposition the model sees only a scalar wind-speed correlate.

**Expected lift.** Batted-ball wind effects are well-documented in public
Statcast research (Alan Nathan, Tom Tango). Adding wind to DPI's xOut
predictor should tighten the within-park residual error, which is exactly
the channel DPI aggregates to its defensive-pressure index. A targeted
ablation (xOut AUC with vs without wind) is the validation gate for this
feature.

## Station-distance QA

The nearest `meteostat` station is used per game. In the smoke run all
stations were within 25 km of the park (p95 ~20 km). Note the following
edge cases to watch for in the full run:

- **Coors Field** (Denver): high-altitude station coverage is good; DEN
  airport is ~15 km from Coors.
- **Rocky-Mountain weather variability** within 15 km is real; the reported
  temperature is a close proxy but not a measurement at the ballpark.
- **Coastal parks** (Oracle, Petco, T-Mobile, Marlins) may have microclimate
  differences between an airport station and the waterfront. Station
  distance will flag the worst of these.

A future refinement would be to co-register an official NOAA station at
each ballpark (MLB teams publish in-stadium weather on `mlb.com/weather` but
that is not a bulk-downloadable source without scraping).

## Weather-source attribution

`weather_source` column values:

- `meteostat:<station_id>` — interpolated from that NOAA station's hourly
  record.
- `dome_skipped` — Tropicana Field. Weather NULL by design.
- `no_metadata` — MLB schedule endpoint did not return venue / datetime.
- `no_station` — venue in registry but no meteostat station within 75 km.
- `meteostat_no_data` — nearest station(s) have no hourly data for the game
  date (historical gap).
- `skipped` — `--no-weather` CLI flag set (metadata-only mode).

## Regenerating

```bash
# Smoke test:
python scripts/ingest_weather_conditions.py --max-games 100

# Metadata-only (no meteostat calls):
python scripts/ingest_weather_conditions.py --no-weather

# Date-filtered:
python scripts/ingest_weather_conditions.py \
    --start-date 2024-04-01 --end-date 2024-10-31

# Full run (~40 min for 25k games):
python scripts/ingest_weather_conditions.py
```

## Open questions / follow-ups

1. **Retractable-roof open/closed resolution.** GUMBO live-feed JSON carries
   `gameData.weather.condition` which sometimes includes "Roof Closed".
   A second-pass script could hydrate roof state for retractable-unknown
   rows.
2. **Wind direction interpolation.** Currently linear; the correct
   interpolation is on the unit circle. Error is < 5 degrees over a 1-hour
   window so acceptable for now.
3. **Rain / precipitation column.** `meteostat.Parameter.PRCP` exists; not
   ingested in this pass. Add if DPI needs wet-ball detection.
4. **Neutral-site games.** Spring-training, international series (Seoul,
   London, Sydney, Mexico City, etc.) will appear as unknown venues; the
   summary print surfaces them for targeted backfill.
