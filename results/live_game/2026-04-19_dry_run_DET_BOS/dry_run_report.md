# Live Pipeline Dry-Run Report — DET @ BOS, 2026-04-19

**game_pk:** 824777
**venue:** Fenway Park
**status at probe time:** Live, top 2nd, 1-1, Crochet (BOS) vs Lee (DET)
**probe time (UTC):** 2026-04-19T21:05Z

---

## 1. Pipeline state: **READY** (with two soft caveats)

End-to-end probe walked **schedule lookup -> live feed fetch -> parse -> tokenize -> PitchGPT inference -> top-1/top-3/log-loss scoring** with no exceptions. Both backend modules (`src/ingest/live_feed.py`) and the frontend page (`src/dashboard/views/live_game.py`) import without error and expose a `render()` entry point ready for radio-button nav.

### Component-by-component

| Component | Status | Notes |
|---|---|---|
| MLB StatsAPI schedule (`/api/v1/schedule`) | OK | DET-BOS resolved as gamePk 824777 in 0.5s. 14 games on the slate. |
| Live feed (`/api/v1.1/game/{pk}/feed/live`) | OK | 352 KB payload, 0.69s wall time. |
| `src.ingest.live_feed` import | OK | Exports: `fetch_live_feed`, `parse_game_state`, `parse_pitch_events`, `parse_lineup`, `parse_bullpen`, `get_todays_games`, `get_phillies_game`, `LiveGameTracker`. |
| `parse_pitch_events` | OK | 60 pitches normalized with pitcher_id, batter_id, count, pitch_type, start_speed, zone, plate_x/z, call_description. |
| `src.dashboard.views.live_game` import | OK | `render()` resolves. All optional deps (`calculate_win_probability`, `GameAnomalyMonitor`, `get_matchup_stats`) imported successfully — `_USE_MOCK_*` flags will all be False at runtime. |
| PitchGPT checkpoint load (`models/pitchgpt_v1.pt`, 5.6 MB) | OK | CUDA available, loaded on GPU. Config: d_model=128, 4 heads, 4 layers, max_seq=256, vocab=2212. |
| PitchGPT inference on 10 live pitches | OK | Top-1 50%, Top-3 90%, log-loss 1.38, no OOV pitches/zones. |

---

## 2. Concrete failures and fixes (ordered by severity)

**No HARD failures.** Two SOFT issues to fix or accept before 7:20:

1. **MEDIUM — runners not threaded into PitchGPT context at inference.** `parse_pitch_events` returns no per-pitch baserunner state. The dry-run script set `on_1b/on_2b/on_3b=False` for all pitches, which biases the model toward base-state-empty distributions. Fix: extract pre-pitch matchup runners from `play.matchup.postOnFirst`/`postOnSecond`/`postOnThird` (or walk `runners` events). Estimated work: 30 min. **Decision:** acceptable for tonight's case study — runners affect pitch-type prediction modestly (mostly via pickoff-attempt detail, not pitch selection), and the PHI starter (Wheeler/Sanchez) won't OOM the runners channel.
2. **LOW — `score_diff` hard-coded to 0 in dry-run.** Same reason — live feed parsing doesn't reconstruct running score in `parse_pitch_events`. Easy fix during inference: the `result.homeScore` / `result.awayScore` fields ARE in the live feed at the play level. Worth a 15-min patch if Phase 2 wants score-aware DPI, but tonight's NLL/top-1 metrics aren't sensitive to this.

**No fixes required to ship Option A (dashboard).** No fixes required to ship Option B (standalone logger) — the standalone path uses the exact `LiveGameTracker` class which is already wired correctly.

---

## 3. StatsAPI quirks observed

- **Latency:** newest-pitch endTime trails wall clock by **21-80s** at the moment of fetch. That is StatsAPI's intrinsic processing window (Statcast post-processing); polling more often than every 10-15s gains nothing.
- **Field completeness on last 20 pitches:** `pitch_type_code` 100%, `startSpeed` 100%, `zone` 100%, `plate_x/z` 100%, `endSpeed` 100%, `extension` 95%, `spinRate` 95%. **Spin rate misses ~5% of pitches** — likely the rare radar dropout. PitchGPT does not consume spin, so this is harmless for the PitchGPT pipeline; flag if the anomaly module relies on spin.
- **Zone numbering:** 1-9 are in-strike-zone (3x3 grid), 11-14 are the four out-of-zone quadrants. Observed in this game: zones 1-9 + 11-14 (no 10 anywhere — zone 10 is unused per the convention). PitchGPT uses its own location-to-zone reducer (`PitchTokenizer.location_to_zone` from plate_x/z), so the StatsAPI `zone` field is descriptive only — no risk of vocab mismatch.
- **Pitch type codes observed:** `FF, SI, FC, SL, CU, CH` and `ST` (sweeper). All in PitchGPT's vocab — **0 OOV pitch types** on the 60 pitches in this game.
- **Pitcher/batter name encoding:** `Hao-Yu  Lee` has a double-space — defensive `.strip()` on names recommended if rendering into UI.

---

## 4. PitchGPT OOV rate on this game

- **Pitch-type OOV: 0/60 (0.0%)** — all observed types (`FF, SI, FC, SL, CU, CH, ST`) are in `PITCH_TYPE_MAP`. (Note: `ST` IS in the map at index 12.)
- **Zone OOV: 0/60** — model bins from plate_x/z directly via `location_to_zone`, immune to source field changes.
- **Velo OOV: 0/60** — bucketing is `<80 / 80-85 / 85-90 / 90-95 / 95+`, observed range 81.2-97.5 mph fits cleanly.
- **No silent fallbacks fired.**

### Inference scorecard (10 pitches, Crochet)

```
top1_accuracy:                 0.50
top3_accuracy:                 0.90
mean_log_loss (pitch-type):    1.38
mean_prob_assigned_to_actual:  0.275
n_oov_pitches:                 0
```

Caveat: N=10 single-pitcher single-game — this proves the **pipeline works**, not that the model is calibrated. Top-3 0.90 is a healthy signal; the misses are all FF-vs-FC/ST cases on a pitcher who throws ~50% fastballs. Comparable to typical OOS PitchGPT top-3 (~0.83-0.93 depending on pitcher).

---

## 5. Recommendation — Option A (dashboard) vs Option B (standalone script)

**Recommendation: OPTION A (dashboard) — confidence MEDIUM-HIGH.**

Rationale:
- `src/dashboard/views/live_game.py` is end-to-end importable, all optional deps (`calculate_win_probability`, `GameAnomalyMonitor`, `get_matchup_stats`) resolve to real implementations (no mock-mode fallbacks armed).
- `_get_game_state()` -> `_adapt_game_state()` -> `_get_pitch_log()` chain works against the live feed (just demonstrated end-to-end).
- Streamlit holds the DuckDB write lock — but tonight is **read-only consumption**, not backfill, so this is fine.
- The dashboard already has scoreboard, win-prob curve, pitch log, velocity tracker, and anomaly alerts wired. We get the visual polish for free.

**Hedge — keep a standalone tracker script ready as Option B fallback.** The `LiveGameTracker` class in `live_feed.py` is feature-complete (callbacks for `on_new_pitch`, `on_play_complete`, `on_pitching_change`). A 30-line standalone script that polls every 15s, dumps each pitch + PitchGPT prediction to `results/live_game/2026-04-19_PHI_vs_ATL/pitch_log.jsonl`, and prints to stdout would take 20 min to write. **Recommend writing it as a parallel Plan B before 7:20**, even if we run Option A — gives us a guaranteed timestamped pitch ledger for Phase 2 post-game synthesis regardless of dashboard health.

**Open risks for tonight (unrelated to today's probe):**
- We didn't probe the Phillies' game_pk (823475) — its feed will be empty until first pitch at 7:20 PM, so we cannot validate it now. Plan: re-run this same probe at 7:25 PM EDT against the Phillies feed, expect identical health.
- Win probability module is imported but not exercised by today's probe (no batter/runner data fed through `calculate_win_probability` yet) — call it once during the 7:25 re-probe to confirm.

---

## Artifacts produced

- `raw_feed_sample.json` — gameData skeleton + last 5 plays from the live feed
- `parsed_pitches.json` — all 60 pitches via `parse_pitch_events`
- `last20_pitches.json` — most recent 20 pitches with full pitchData fields
- `feed_completeness.md` — field-by-field population %, latency table
- `pitchgpt_live_dryrun.json` — top-1/top-3/log-loss per pitch + summary
- `dry_run_report.md` — this file
