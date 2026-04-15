# Baseball Analytics Platform — Architecture Plan

## Vision
A real-time baseball analytics dashboard that surfaces anomalies, pitcher-batter matchups, bullpen strategy insights, and deep Statcast metrics — giving you an edge over TV announcers during Phillies games.

---

## Phase 1: Data Foundation (MVP)

### Data Sources (All Free)

| Source | What We Get | Access Method | Freshness |
|--------|-------------|---------------|-----------|
| **MLB Stats API (GUMBO)** | Live pitch-by-pitch, play-by-play, rosters, lineups, schedules, transactions | REST polling (no auth) | Real-time (every pitch) |
| **Baseball Savant (Statcast)** | Spin rate, exit velo, launch angle, xBA, xwOBA, barrel%, pitch movement (92 columns/pitch) | `pybaseball` | Next-day |
| **FanGraphs** | WAR, wRC+, FIP, xFIP, SIERA, Stuff+, projections | `pybaseball` | Daily |
| **Baseball Reference** | Historical splits, game logs, bWAR | `pybaseball` | Daily |
| **Chadwick Register** | Player ID crosswalk (joins all sources) | `pybaseball` | Periodic |

### Tech Stack (MVP)

```
┌─────────────────────────────────────────────────┐
│                   Streamlit UI                   │
│  (Plotly charts, strike zone heatmaps, cards)   │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────┐
│              Python Analytics Engine             │
│  pandas · scipy · pymc5 · scikit-learn           │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────┬───────────┴────────────┬─────────────┐
│  Redis   │     DuckDB             │  pybaseball  │
│  (live   │  (pitch history,       │  + MLB Stats │
│  game    │   matchups,            │  API client  │
│  state)  │   analytics)           │  (ingest)    │
└──────────┴────────────────────────┴─────────────┘
```

| Layer | Technology | Why |
|-------|-----------|-----|
| **Ingestion** | `pybaseball` + `MLB-StatsAPI` + custom poller | Free, Python-native, well-maintained |
| **Live Game Feed** | Poll GUMBO endpoint every 10s | Simple, sufficient for dashboard |
| **Database** | DuckDB (single file, ~500MB/season) | Zero config, 1500x faster than Postgres on analytics |
| **Cache** | Redis (or in-memory dict) | Live game state, pre-computed matchups |
| **Analytics** | pandas + scipy + pymc5 | Bayesian matchups, anomaly detection |
| **Frontend** | Streamlit + Plotly | Fastest path to interactive viz |

### Database Schema (DuckDB)

```sql
-- Core pitch-by-pitch table (~750K rows/season, ~7.5M total for 2015-2025)
CREATE TABLE pitches (
    game_pk           INTEGER,
    game_date         DATE,
    pitcher_id        INTEGER,
    batter_id         INTEGER,
    pitch_type        VARCHAR,
    pitch_name        VARCHAR,
    release_speed     FLOAT,
    release_spin_rate FLOAT,
    spin_axis         FLOAT,
    pfx_x             FLOAT,    -- horizontal movement (inches)
    pfx_z             FLOAT,    -- vertical movement (inches)
    plate_x           FLOAT,    -- horizontal plate location
    plate_z           FLOAT,    -- vertical plate location
    release_extension FLOAT,
    release_pos_x     FLOAT,
    release_pos_y     FLOAT,
    release_pos_z     FLOAT,
    launch_speed      FLOAT,    -- exit velocity
    launch_angle      FLOAT,
    hit_distance      FLOAT,
    hc_x              FLOAT,    -- hit coordinate x
    hc_y              FLOAT,    -- hit coordinate y
    bb_type           VARCHAR,  -- ground_ball/line_drive/fly_ball/popup
    estimated_ba      FLOAT,    -- xBA
    estimated_woba    FLOAT,    -- xwOBA
    delta_home_win_exp FLOAT,
    delta_run_exp     FLOAT,
    inning            INTEGER,
    inning_topbot     VARCHAR,
    outs_when_up      INTEGER,
    balls             INTEGER,
    strikes           INTEGER,
    on_1b             INTEGER,
    on_2b             INTEGER,
    on_3b             INTEGER,
    stand             VARCHAR,  -- batter L/R
    p_throws          VARCHAR,  -- pitcher L/R
    at_bat_number     INTEGER,
    pitch_number      INTEGER,
    description       VARCHAR,  -- called_strike, swinging_strike, ball, foul, hit_into_play
    events            VARCHAR,  -- strikeout, single, home_run, etc.
    type              VARCHAR   -- B/S/X (ball/strike/in-play)
);

CREATE TABLE players (
    player_id    INTEGER PRIMARY KEY,
    full_name    VARCHAR,
    team         VARCHAR,
    position     VARCHAR,
    throws       VARCHAR,
    bats         VARCHAR,
    mlbam_id     INTEGER,
    fg_id        VARCHAR,
    bref_id      VARCHAR
);

CREATE TABLE games (
    game_pk      INTEGER PRIMARY KEY,
    game_date    DATE,
    home_team    VARCHAR,
    away_team    VARCHAR,
    venue        VARCHAR,
    game_type    VARCHAR
);

CREATE TABLE season_stats (
    player_id    INTEGER,
    season       INTEGER,
    stat_type    VARCHAR,  -- 'batting' or 'pitching'
    -- Batting
    pa INTEGER, ab INTEGER, h INTEGER, hr INTEGER, bb INTEGER, so INTEGER,
    ba FLOAT, obp FLOAT, slg FLOAT, ops FLOAT,
    woba FLOAT, wrc_plus FLOAT, war FLOAT,
    -- Pitching
    ip FLOAT, era FLOAT, fip FLOAT, xfip FLOAT, siera FLOAT,
    k_pct FLOAT, bb_pct FLOAT, whip FLOAT,
    stuff_plus FLOAT, location_plus FLOAT, pitching_plus FLOAT,
    PRIMARY KEY (player_id, season, stat_type)
);

-- Materialized views for fast matchup queries
CREATE TABLE matchup_cache AS
SELECT
    pitcher_id, batter_id,
    COUNT(*) as pitches,
    COUNT(DISTINCT game_pk) as games,
    AVG(CASE WHEN type = 'X' THEN launch_speed END) as avg_exit_velo,
    AVG(CASE WHEN type = 'X' THEN estimated_ba END) as avg_xba,
    SUM(CASE WHEN events IS NOT NULL AND events NOT IN ('strikeout','walk','hit_by_pitch')
        THEN 1 ELSE 0 END) as balls_in_play,
    SUM(CASE WHEN events IN ('single','double','triple','home_run') THEN 1 ELSE 0 END) as hits,
    SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END) as strikeouts
FROM pitches
GROUP BY pitcher_id, batter_id;
```

---

## Phase 2: Analytics Modules

### Module 1: Live Game Dashboard
- **Real-time game state** from GUMBO feed (score, count, runners, outs)
- **Win probability graph** updated every pitch (lookup table: inning × outs × base state × score diff)
- **Current matchup card**: pitcher vs batter with historical stats + Bayesian prediction
- **Pitch-by-pitch log** with velocity, type, location, result
- **Run expectancy** for current base-out state

### Module 2: Pitcher-Batter Matchups
- **Bayesian hierarchical log5 model** (PyMC5) to handle small samples
  - Shrinks toward population priors when matchup data is sparse
  - Full posterior distributions → uncertainty quantification
  - Factors: batter wOBA, pitcher FIP, platoon splits, pitch type vulnerability
- **Pitch type vulnerability matrix**: batter's run value vs each pitch type
- **Zone heatmaps**: where the pitcher locates vs where the batter is dangerous
- **Historical results** with context (date, count, pitch sequence → result)

### Module 3: Bullpen Strategy
- **Available relievers** with current fatigue state:
  - Days rest, pitches in last 72 hours, consecutive appearance count
  - Color-coded: green (fresh) / yellow (moderate) / red (fatigued)
- **Matchup grid**: each available reliever vs upcoming batters
  - Platoon advantage indicator
  - Pitch arsenal vs batter weakness alignment
  - Bayesian expected wOBA for each matchup
- **Leverage Index** of current situation → recommended reliever deployment
- **Upcoming batters queue** (next 3-6 hitters with key stats)

### Module 4: Anomaly Detection
- **In-game alerts** (real-time during Phillies games):
  - Velocity drop: rolling 5-pitch avg drops >1.5 mph below game baseline
  - Spin rate change: >5% deviation from baseline
  - Release point drift: RMSE of last 10 pitches vs game average exceeds threshold
  - Pitch mix shift: unusual pitch type frequency for the count/situation
- **Cross-game anomalies** (updated daily):
  - Sprint speed decline (potential injury)
  - Exit velocity trend changes (CUSUM change point detection)
  - Batting approach shifts (chase rate, swing rate deviations)
- **Methods**: Z-scores, EWMA, CUSUM, Isolation Forest for multivariate anomalies

### Module 5: Phillies Focus
- **Pre-game report**: today's matchup analysis, starting pitcher scouting
- **Opponent pitcher breakdown**: arsenal, tendencies, weaknesses
- **Phillies batter cards**: how each batter fares vs today's pitch types
- **Live roster tracking**: transactions, IL moves, call-ups (via MLB Stats API `/transactions`)
- **Season dashboard**: Phillies standings, upcoming schedule, rotation matchups

---

## Phase 3: Advanced Analytics (Nobel Prize Territory)

### Stuff+ Model (Custom)
- Build our own pitch quality model using XGBoost regression
- Features: velocity, spin rate, spin axis, horizontal/vertical break, extension, release point
- Target: run value per pitch
- Train on full Statcast history (2015-present)
- Compare against FanGraphs' published Stuff+ for validation

### Pitch Sequencing Model
- Directed graph embeddings of pitch sequences (Gaussian Mixture Model clustering)
- Identify "setup" and "knockout" patterns
- Detect when pitchers deviate from their normal sequencing → anomaly signal

### Swing Decision Model (Bat Tracking)
- Integrate 2024-2025 bat tracking data (bat speed, swing length, attack angle)
- Model optimal vs actual swing decisions
- Identify batters whose process (swing quality) exceeds results → buy-low candidates

### Injury Risk Model
- Ensemble of Gradient Boosting + Random Forest + Logistic Regression
- Features: velocity trend, spin rate trend, release point variability, workload metrics
- Published research achieves AUC 0.81 for Tommy John prediction
- Alert when a pitcher's biomechanical signature shifts toward injury profiles

### Win Probability Model (Enhanced)
- Standard: historical lookup table (inning, outs, bases, score)
- Enhanced: incorporate pitcher quality remaining, bullpen depth, specific matchup projections
- Monte Carlo simulation of remaining game from current state

---

## Data Pipeline Architecture

### Daily ETL (runs at 6 AM ET)
```
1. pybaseball.statcast(yesterday) → DuckDB pitches table
2. pybaseball.batting_stats(current_year) → DuckDB season_stats
3. pybaseball.pitching_stats(current_year) → DuckDB season_stats
4. MLB Stats API /transactions → check for roster moves
5. Rebuild matchup_cache table
6. Update anomaly baselines (rolling averages, std devs)
7. Re-run Bayesian matchup models for today's Phillies game
```

### Live Game Loop (during Phillies games)
```
Every 10 seconds:
1. Poll GUMBO: statsapi.mlb.com/api/v1.1/game/{gamePk}/feed/live
2. Diff against cached state
3. On new pitch:
   a. Update live pitch log
   b. Recalculate win probability
   c. Check anomaly triggers (velocity, spin, release point)
   d. Update current matchup display
4. On pitching change:
   a. Load new pitcher's arsenal data
   b. Generate matchup cards for upcoming batters
   c. Update bullpen availability
5. Push state to Redis → SSE to Streamlit
```

### Historical Backfill (one-time)
```
For each year 2015-2025:
  pybaseball.statcast(start_dt=f'{year}-03-01', end_dt=f'{year}-11-30')
  → ~750K pitches/year → DuckDB
Total: ~7.5M pitches, ~1.2GB compressed
```

---

## Project Structure

```
baseball/
├── PLAN.md                     # This file
├── requirements.txt            # Python dependencies
├── data/
│   ├── baseball.duckdb         # Main database
│   └── cache/                  # pybaseball cache directory
├── src/
│   ├── ingest/
│   │   ├── statcast_loader.py  # Historical Statcast data loader
│   │   ├── live_feed.py        # GUMBO live game poller
│   │   ├── roster_tracker.py   # Roster/transaction monitor
│   │   └── daily_etl.py        # Daily stats refresh
│   ├── analytics/
│   │   ├── matchups.py         # Bayesian pitcher-batter matchup model
│   │   ├── anomaly.py          # Anomaly detection (velocity, spin, release)
│   │   ├── bullpen.py          # Bullpen fatigue + matchup optimizer
│   │   ├── win_probability.py  # Win expectancy model
│   │   └── stuff_model.py      # Custom Stuff+ pitch quality model
│   ├── db/
│   │   ├── schema.py           # DuckDB schema creation
│   │   └── queries.py          # Common analytical queries
│   └── dashboard/
│       ├── app.py              # Streamlit main app
│       ├── pages/
│       │   ├── live_game.py    # Live game view
│       │   ├── matchups.py     # Pitcher-batter matchup explorer
│       │   ├── bullpen.py      # Bullpen strategy view
│       │   ├── anomalies.py    # Anomaly alerts view
│       │   └── phillies.py     # Phillies-focused dashboard
│       └── components/
│           ├── strike_zone.py  # Strike zone heatmap component
│           ├── spray_chart.py  # Spray chart component
│           ├── pitch_movement.py # Pitch movement scatter
│           ├── matchup_card.py # Matchup comparison card
│           └── win_prob.py     # Win probability line chart
├── models/
│   └── matchup_model.pkl       # Trained Bayesian model cache
└── scripts/
    ├── backfill.py             # One-time historical data load
    └── setup_db.py             # Initialize database
```

---

## Python Dependencies

```
# Data acquisition
pybaseball>=2.2.7
MLB-StatsAPI>=1.9.0
requests

# Database
duckdb>=1.1.0

# Analytics
pandas>=2.0
numpy
scipy
scikit-learn
pymc>=5.0
arviz

# Visualization / Dashboard
streamlit>=1.35
plotly>=5.0

# Caching / Live state
redis

# Utilities
python-dateutil
tqdm
```

---

## Implementation Order

1. **Setup** — project structure, dependencies, DuckDB schema
2. **Historical backfill** — load 2015-2025 Statcast data (~7.5M pitches)
3. **Season stats loader** — FanGraphs batting/pitching stats
4. **Player ID crosswalk** — Chadwick register for joining sources
5. **Basic dashboard shell** — Streamlit with navigation
6. **Matchup explorer** — query historical pitcher vs batter data + visualize
7. **Live game feed** — GUMBO poller + live game dashboard page
8. **Anomaly detection** — velocity/spin/release point monitoring
9. **Bayesian matchup model** — PyMC5 hierarchical model
10. **Bullpen strategy module** — fatigue tracking + matchup grid
11. **Phillies focus page** — pre-game reports + roster tracking
12. **Advanced models** — Stuff+, pitch sequencing, win probability
