# Post-Ingestion Next Phase Plan

**Date:** 2026-04-23
**Prior milestone commits:** `fcb47ee` (ingestion) + `b1e374c` (flagship wiring)
**Strategy anchor:** NORTH_STAR.md Path 2 — edge surfacing over gate completion

---

## 1. Where we are

Two commits landed today that closed a long-open data-depth gap:

**Ingestion expansion (fcb47ee)** — five new data dimensions staged to parquet and consolidated to DuckDB in a single transaction. DB grew 1.86 → 2.00 GB:

| Dataset | Rows | Coverage |
|---|---:|---|
| `umpire_assignments` | 103,192 | 2015-2026, 93.6% matched to `game_pk` |
| `umpire_tendencies` | 1,104 | umpire × season, accuracy_above_x + favor |
| `game_weather` | 25,050 | 94.2% with weather (meteostat/NOAA); dome handling spec'd |
| `tj_surgery_dates` | 302 | 224 pitchers, classifier now catches UCL sprain / elbow sprain / internal brace |
| `pitches` (append) | +480,969 | 2017-06/07 and 2019-08/09 were 100% absent in prior monthly-window backfill |
| `season_batting_stats` (append) | +4,734 | 2010-2014 bWAR |
| `season_pitching_stats` (append) | +3,327 | 2010-2014 bWAR |

**Flagship wiring (b1e374c)** — v2/v3 artifacts, v1s preserved:

| Model | Features added | Key result |
|---|---|---|
| CausalWAR v2_umpweather | 2 umpire (prior-season) + 5 weather (current-game) | Buy-Low **68.4% UNCHANGED** — marquee claim preserved; pitchers tighten, batters loosen as expected |
| PitchGPT v2_ump | 1 context dim (ump accuracy_above_x) | LSTM delta **regressed** 13.8→11.8% at smoke scale; calibration ECE **-70%** (best ever) |
| DPI v3_weather | wind_parallel, wind_perpendicular, temp_f | All 6 gates PASS; external OAA **+0.0081** consistent across 3 windows |

Tests: 1004 passed / 0 failed / 23 skipped.

---

## 2. Phase goals

1. **Resolve the PitchGPT verdict** — is the LSTM regression a smoke-scale artifact or real? Decision gates which checkpoint is flagship.
2. **Convert staged data into actionable edges** — weather-sensitive segments, umpire-sensitive pitchers, TJ-aware projections.
3. **Tighten CausalWAR contrarian CI** by wiring pre-2015 WAR into the backtest (n=11 → n=15 season-pairs; ~16% CI shrink).
4. **Close the follow-up debt** — one-line production blockers + pre-existing session drift.

Out of scope: new model builds (NORTH_STAR Path 2 constraint), MiLB Statcast (research rejected), Trackman raw biomechanics (inaccessible), MechanixAE/VWR re-litigation.

---

## 3. Work items

### P0 — PitchGPT verdict (1-2 days, blocking)

| # | Item | Effort | Deliverable |
|---|---|---|---|
| P0.1 | Run full 10K-game PitchGPT v2 retrain | ~90 min wall-clock | Updated `docs/models/pitchgpt_results.md` with full-scale LSTM delta |
| P0.2 | Fix `PitchGPT.evaluate()` to forward `model_version` kwarg | ~30 min | `src/analytics/pitchgpt.py` signature + downstream callers |
| P0.3 | Dashboard checkpoint routing decision + implementation | ~1 hour | `src/dashboard/views/pitchgpt_view.py` loads the chosen flagship |

**Decision gate after P0.1:**
- If LSTM delta ≥15% at full scale → v2 becomes flagship, cite calibration as bonus
- If LSTM delta stays <15% → v1 remains flagship, v2 shipped as "calibration variant" with narrative framing
- If the dual-narrative feels contrived → revert CONTEXT_DIM in main branch, keep v2 in a feature branch

### P1 — Edge surfacing from new data (1 week)

| # | Item | Effort | Deliverable |
|---|---|---|---|
| P1.1 | Weather-sensitive sub-segment analysis | 2-3 days | `docs/edges/weather_segments_2026_04.md` — ranked cohorts where DPI lift concentrates (fly-ball pitchers × wind_parallel × cold-weather) |
| P1.2 | Umpire-sensitive pitcher board | 2 days | `docs/edges/umpire_sensitive_pitchers.md` — pitchers whose projection shifts most under specific ump tendencies |
| P1.3 | Wire TJ dates into Projections v2 flag | 1 day | `scripts/projections.py` consumes `tj_surgery_dates`; verify Buehler/deGrom/Strider now flagged; dashboard surfacing |

### P2 — Infrastructure + second-order wins (1-2 weeks)

| # | Item | Effort | Deliverable |
|---|---|---|---|
| P2.1 | Extend CausalWAR contrarian backtest with 2010-2014 WAR | 2-3 days | n=15 season-pairs, CI tightened ~16%; caveat pre-Statcast NULL feature handling |
| P2.2 | Hydrate retractable-roof status via GUMBO API | 2 days | ~5,300 games move from `retractable_unknown` → proper `open`/`closed`; DPI wind signal recovers |
| P2.3 | Pre-existing dashboard drift triage | User decision | 13 files from 4/19 — commit / revert / selective |

---

## 4. Dependencies + parallelism

```
P0.1 (10K retrain, background) ─┬─→ P0.3 (routing)
P0.2 (evaluate kwarg fix) ──────┘

P1.1, P1.2 independent — can run parallel with P0.1
P1.3 needs P0 complete (projections may load pitchgpt)

P2.1 needs idle CausalWAR training slot (after P1.x)
P2.2 independent
P2.3 user-gated
```

**Suggested execution:**
- Day 1: kick off P0.1 in background, do P0.2 in foreground, launch P1.1 + P1.2 in parallel
- Day 2: P0.3 after P0.1 lands, P1.3 starts
- Week 1 close: P1.x edges drafted, P0 verdict committed
- Week 2: P2.1 + P2.2

---

## 5. Open decisions

1. **PitchGPT flagship** — v1 (LSTM narrative) vs v2 (calibration narrative) vs both. Decide after P0.1.
2. **Dashboard drift** — commit pre-existing 4/19 modifications, revert, or selectively stage?
3. **Edge output format** — live dashboard view (interactive) vs weekly markdown reports (shareable). Hybrid possible.
4. **Weather lift "decisive" threshold** — current DPI +0.0081 is within CI noise. What n or effect-size would upgrade it from "suggestive" to "claim"?

---

## 6. Success criteria

Phase closes when:

- [ ] PitchGPT flagship verdict locked + dashboard routes correctly
- [ ] At least one actionable edge surfaced from weather OR umpire data (signed markdown, linked from NORTH_STAR)
- [ ] TJ flag live in Projections v2 with ≥3 canonical cases verified
- [ ] CausalWAR contrarian CI either tightened with pre-2015 data OR explicit hold-at-n=11 decision documented
- [ ] Workspace clean: no self-generated uncommitted changes; 4/19 drift resolved

---

## 7. Deferred / not in this phase

| Item | Why deferred |
|---|---|
| MiLB Statcast ingestion | Research agent rejected — doesn't fit flagship architecture; scope creep toward new model |
| Trackman / Hawk-Eye raw biomechanics | Inaccessible (private license); would only aid demoted MechanixAE |
| Full Baseball Prospectus injury DB | VWR retracted + MechanixAE demoted; TJ dates are the only injury signal worth the work |
| New model builds | NORTH_STAR Path 2 explicit constraint |
| Methodology papers (NORTH_STAR 5A) | Wait until at least one edge from P1 validates at scale |

---

## 8. Honest read on today's milestone

Ingestion was a clean technical win — every dataset staged, consolidated, validated, and committed idempotently. The TJ classifier fix was a genuine upgrade (123 → 302 events with a real admission gate, no fabrication).

Flagship wiring was mixed. CausalWAR preserved the marquee Buy-Low claim exactly and the cohort splits behaved as physics predicted. DPI caught a real positive signal (+0.0081 external OAA, consistent across three windows, within CI noise on any single one). PitchGPT's ump context did not deliver on perplexity — but it unexpectedly sharpened calibration 70%, which is its own finding worth citing.

The phase ahead is about converting this foundation into actionable output. The data is landed; now it has to earn its rent.
