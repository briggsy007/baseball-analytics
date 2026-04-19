# Live Game Case Study — 2026-04-19 PHI vs ATL — Session Handover & Execution Plan

**First pitch:** 7:20 PM EST, 2026-04-19.
**Teams:** Philadelphia Phillies vs Atlanta Braves (confirm home/away from MLB feed at game time).
**Purpose of this document:** self-contained handover — a fresh Claude session with no prior context should be able to read this plus `docs/NORTH_STAR.md` and execute end-to-end.

---

## 1. Start here (read this section first)

**Role.** You are the PM-level assistant coordinating a live-game case study for Hunter Briggs's baseball analytics platform. User's feedback memory confirms: act as PM, delegate implementation to agents, never write code directly, always push after committing, always run a validation agent after multi-agent batches.

**Goal of tonight's case study.** Produce a time-stamped submission-ready artifact showing the three flagship models (DPI, CausalWAR, PitchGPT) operating on a real live 2026 MLB game. Output lives in this directory: `docs/live_game/2026-04-19_PHI_vs_ATL/` and `results/live_game/2026-04-19_PHI_vs_ATL/`.

**Strategy context.** Per `docs/NORTH_STAR.md` "Post-evidence consolidation" (most recent section), the flagship three are **DPI, CausalWAR, PitchGPT**. VWR was retracted. MechanixAE demoted. ChemNet retired. Path 2 philosophy = edge surfacing over gate completion. Every committed number has a CI. Honest-negative reporting is the standard.

**Parallel session.** A separate Claude session is working on frontend enhancements — dashboard live-game view, `.streamlit/config.toml` tweaks, and `src/ingest/live_feed.py` (MLB StatsAPI client). Coordinate via the shared git workspace; their uncommitted changes should be left alone in your working tree. If push fails with "remote has advanced" it's their work — fetch + rebase + push.

---

## 2. Project state at handover

**Git state:** `origin/main` is fully synced. Most recent commits:
- `6ade5b0` Add headline_findings.md: 5-minute award-committee synthesis
- `af1cb68` Tighten summary.md: honest CausalWAR framing + DPI AR(1) caveat + retired-models section
- `e405e7b` Paper strengtheners: CausalWAR mechanism ablation + base-rate + DPI combined predictor
- `c6d8eb9` Flagship methodology papers: DPI + CausalWAR + PitchGPT (Phase 5A)
- `807bbd9` Flagship bulletproofing: DPI prospective + CausalWAR autopsy + PitchGPT downstream utility
- `4a5aace` OOS evidence sweep — VWR retracted, DPI reinforced

**Docs portfolio (committed):**
- `docs/awards/summary.md` — recruiter landing
- `docs/awards/headline_findings.md` — 5-min award committee
- `docs/awards/methodology_paper_dpi.md` — reviewer-grade DPI
- `docs/awards/methodology_paper_causal_war.md` — reviewer-grade CausalWAR (incl. mechanism-ablation + base-rate addenda)
- `docs/awards/methodology_paper_pitchgpt.md` — reviewer-grade PitchGPT

**Key validation artifacts committed:**
- `results/defensive_pressing/{2025_validation, prospective_validation, combined_predictor}/`
- `results/causal_war/{contrarian_stability, tag_filter_baserate, regression_autopsy_2023_2024, mechanism_ablation}/`
- `results/pitchgpt/{2025_holdout, downstream_utility}/`

---

## 3. Flagship summary — use these exact framings

### DPI (Defensive Pressing Index)
- **Paper claim:** "Best BIP-outcome residual metric in the platform. 2025 r=0.641 vs Statcast OAA (CI [0.42, 0.79]). Beats OAA on next-year BABIP (CI excludes zero). Loses to AR(1) on persistence forecasting — measurement tool, not standalone forecaster."
- **Lead flagship** in the award submission.
- **Live-game role:** pre-game team-defense framing; per-pitch DPI pressure is NOT a thing the model does (aggregates season-level).

### CausalWAR + Contrarian Leaderboards
- **Paper claim:** "First DML-at-scale deployment to baseball player valuation. 2025 Buy-Low 68.4% (13/19, CI [0.47, 0.84]). Three-year mechanism cores: RELIEVER LEVERAGE GAP 78%, PARK FACTOR 70%, DEFENSE GAP 79%. Base-rate study (committed) confirms CausalWAR's DML residual is doing real selection work — tag filter alone is 56.9%, top-N-by-bWAR is 10%, CausalWAR is 78.1%."
- **Live-game role:** pre-game check — any Phillies or Braves on tonight's roster tagged Buy-Low or Over-Valued? If yes, track their line during the game; is the in-game performance consistent with CausalWAR's prediction?

### PitchGPT
- **Paper claim:** "Calibrated decoder-only transformer. 2025 OOS ECE 0.0098 post-temperature (gate <0.10). Beats LSTM 13.80% (CI [12.22, 15.51]) — **1.2pp below 15% spec gate, honest shortfall**. Beats Markov-2 56.85%, heuristic 67.75%. Downstream utility on pitch-outcome-bucket returns a tie — calibrated, not decision-utility-replacement."
- **Live-game role:** THE marquee in-game test. Run PitchGPT against live pitches, compare predicted distribution to actual. Log calibration under real conditions.

### Honest negatives (brief context for anyone reading)
- VWR retracted (scale-verified null). MechanixAE demoted (AUC below random). ChemNet retired (r=0.089/0.155). volatility_surface retired (clean null). See paper portfolio for full receipts.

---

## 4. Execution plan (Phase 0 → 1 → 2)

### Phase 0 — Pre-game intelligence package (deadline: 6:30 PM EST)

**Launch this as soon as the new session starts.** Target completion ≤45 min. One background agent, subagent_type `general-purpose`.

**Output:** `results/live_game/2026-04-19_PHI_vs_ATL/pregame.md` + supporting JSON/CSVs in the same directory.

**Sections the pregame doc should produce:**
1. **Game header** — date, first pitch, ballpark, confirmed home/away, starting pitchers (use MLB StatsAPI probable pitchers endpoint), weather if available.
2. **Contrarian leaderboard cross-reference** — filter `results/causal_war/contrarian_stability/buy_low_2024_to_2025.csv` and `over_valued_2024_to_2025.csv` to players currently on the PHI or ATL 26-man roster (join on `player_id` or name — roster from MLB StatsAPI team endpoint). For any hits: list name, Buy-Low or Over-Valued, mechanism tag, predicted Δ-WAR, actual 2025 WAR so far.
3. **DPI head-to-head** — 2025 team DPI rankings for PHI and ATL from `results/defensive_pressing/2025_validation/team_rankings_2025.csv`, plus 2025 BABIP-against, plus the DPI–OAA disagreement direction per team (from `disagreement_analysis.md`).
4. **PitchGPT starting-pitcher profiles** — for each confirmed starter, pull their last 3-5 starts from Statcast and run PitchGPT over those sequences to produce: predicted pitch-type distribution per count state, entropy (predictability), favorite pitch per count, where they most deviate from league-average calls. This is the **batter-preparation artifact**.
5. **What to watch for tonight** — 3-5 bullet points: specific matchups, counts, or situations where the flagship models predict something non-obvious will happen.

**Agent prompt skeleton for Phase 0:**

> You are the pre-game-intelligence agent for tonight's Phillies vs Braves case study at `C:\Users\hunte\projects\baseball`. First pitch 7:20 PM EST 2026-04-19. Produce `results/live_game/2026-04-19_PHI_vs_ATL/pregame.md` following §4 Phase 0 of `docs/live_game/2026-04-19_PHI_vs_ATL/EXECUTION_PLAN.md`. Read `docs/awards/headline_findings.md` and the three methodology papers for the exact claim-framings. Use real bWAR already backfilled at `data/fangraphs_war_staging.parquet`. Do not retrain any model. Do not spin; if a Buy-Low player has already under-performed in 2025 YTD, say so. Return a 200-word summary.

### Phase 1 — Live pitch logger (7:20 PM → game end, ~3h)

**Runs on user's machine, NOT as an agent** — live work is too long and lossy for a long-running agent. The logger either:
- **Option A (preferred if ready):** dashboard live-game view (the frontend session's work). User opens `streamlit run src/dashboard/app.py`, navigates to Live Game view, watches PitchGPT predictions roll in. Dashboard writes to `results/live_game/2026-04-19_PHI_vs_ATL/pitch_by_pitch.csv`.
- **Option B (fallback):** standalone script `scripts/live_game_logger.py` that polls MLB StatsAPI `/game/{game_pk}/playByPlay` every 10-15s. **This script does not exist yet.** If Option A is not usable by 6:45 PM, a new session should delegate writing this script to a quick agent.

**CSV columns required:** `timestamp_utc, inning, half, batter_id, batter_name, pitcher_id, pitcher_name, count_balls, count_strikes, runners_on, outs, pitchgpt_top1_type, pitchgpt_top1_prob, pitchgpt_top3_json, actual_pitch_type, actual_zone, actual_velo, correct_top1, brier, log_loss`.

**Running metrics during the game:**
- Rolling top-1 accuracy (PitchGPT's stated probability vs actual)
- Rolling ECE on pitches so far
- Flag every high-confidence-but-wrong call (prob >0.5 and wrong) — these are "model said X, human saw Y" moments, the interesting anecdotes
- If Option A dashboard is live, these should update on screen. If Option B script, print to stdout every N pitches.

**Latency note.** Baseball Savant pitch-level lags 5-10 min. MLB StatsAPI play-by-play is ~1-5s but lacks Statcast `pitch_type` occasionally (defaults to NULL until tagging). First decision once live: which feed. StatsAPI is probably the right one for live; you can retroactively enrich with Savant post-game.

### Phase 2 — Post-game synthesis (tomorrow morning, 2026-04-20)

**One agent, subagent_type `general-purpose`, ~30-45 min.** Reads the committed pitch-by-pitch CSV and produces `results/live_game/2026-04-19_PHI_vs_ATL/post_game_report.md`.

**Required sections:**
1. **Game summary** — final score, notable plays, box score reference.
2. **PitchGPT calibration under live conditions** — reliability diagram on the actual pitches, ECE computed live, compare to 2025 OOS ECE=0.0098 (is calibration preserved under live latency + noise?).
3. **PitchGPT vs LSTM replay** — re-run the LSTM baseline on the same logged pitches, compute the log-loss delta with CI. Does the 13.80% OOS edge hold for one game? (Single-game n may be too small for a stable number — report anyway with the caveat.)
4. **Contrarian picks tonight** — for each pre-game-flagged player who played: their pitch-level / PA-level outcomes. Did the game line match CausalWAR's direction? If a Buy-Low Phillie had 3 hits and an RBI, that's a marquee sentence.
5. **DPI plays of note** — any standout defensive plays. Was the team with higher DPI actually the better defense tonight? (One-game noise is high — frame honestly.)
6. **Honest limitations for the writeup** — data gaps, latency missed pitches, wrong calls, any live-feed failures.
7. **Submission-ready caption** — one 140-character-style summary line usable on the public-facing portfolio.

---

## 5. Known dependencies

| Dependency | Status | Owner | Fallback if broken |
|---|---|---|---|
| `src/ingest/live_feed.py` (MLB StatsAPI wrapper) | In progress, frontend session | Frontend session | Write a standalone `scripts/live_game_logger.py` via a quick agent. |
| `src/dashboard/views/live_game.py` | Exists (extent unknown); frontend session may be extending | Frontend session | Use Option B standalone script. |
| MLB StatsAPI uptime | External | MLB | pybaseball scrape fallback (slower, higher latency). |
| PitchGPT checkpoint | Committed, verified working | — | None needed. |
| Committed artifacts for DPI + CausalWAR pre-game | Done | — | None needed. |
| User's DuckDB (single-writer) | — | User | **IMPORTANT:** if the dashboard is open, no other process can write to DuckDB. Stop the dashboard before running backfill scripts or retrains. |

---

## 6. Exact commands (copy-paste ready)

```bash
# Verify origin is synced at session start
cd C:/Users/hunte/projects/baseball && git fetch origin && git status

# Check if live_feed.py exists and is importable (tests frontend session progress)
python -c "from src.ingest.live_feed import get_phillies_game" 2>&1 | head -5

# Get tonight's game_pk (run ~1-2h before first pitch so lineups are posted)
python -c "from pybaseball import schedule_and_record; import pandas as pd; sched = schedule_and_record(2026, 'PHI'); print(sched[sched['Date'].str.contains('Sat, Apr 19', na=False)])" 2>&1 | head

# Once game_pk known, tap play-by-play (use actual game_pk, this is the API shape)
python -c "import requests; r = requests.get('https://statsapi.mlb.com/api/v1.1/game/{GAME_PK}/feed/live'); print(r.json().get('liveData', {}).get('plays', {}).get('allPlays', [])[:2])"

# At game end, commit + push the pitch-by-pitch CSV
cd C:/Users/hunte/projects/baseball && git add results/live_game/2026-04-19_PHI_vs_ATL/ && git commit -m "Live game case study: pitch-by-pitch log + pregame intel (PHI vs ATL)" && git push origin main
```

---

## 7. Decision points (in order of when they come up)

1. **At session start (now):** Launch Phase 0 pre-game agent. Background mode so context isn't blocked. (Prompt skeleton in §4 Phase 0.)
2. **At 6:00 PM EST:** Check Phase 0 output. If missing or incomplete, escalate — either re-launch or produce a minimal version inline.
3. **At 6:30-6:45 PM EST:** Check whether `src/ingest/live_feed.py` and the dashboard live-game view are usable. If yes → Option A. If no → launch a quick agent to write `scripts/live_game_logger.py` as Option B fallback.
4. **At 7:20 PM EST:** First pitch. Logger running.
5. **Throughout game:** Monitor the logger output. If a Buy-Low or Over-Valued player from Phase 0's roster cross-reference has a big moment, note it for Phase 2. If PitchGPT makes a high-confidence wrong call on a memorable pitch, flag it.
6. **At game end:** Commit the raw CSV. Don't launch Phase 2 tonight — it can wait for morning. User may want to watch / rest.
7. **Tomorrow morning 2026-04-20:** Launch Phase 2 post-game synthesis agent.
8. **After Phase 2 lands:** Commit post-game report. Run validation-agent (`C:\Users\hunte\projects\baseball\.claude\skills\validation-agent`) since this will be a doc-heavy batch.

---

## 8. Honest risks

- **Live feed may never come up tonight.** Frontend session's `live_feed.py` is in-progress; it may not be ready or may have bugs. Plan B is a standalone script; Plan C is post-game-only on recorded data (still submission-valid, just less "live").
- **Latency-driven coverage gaps.** Even in Option B, MLB StatsAPI might drop or lag pitches. Any gaps must be called out in Phase 2 honestly.
- **PitchGPT composite-token mapping on live data.** Live data may not have the exact `pitch_type × zone × velocity-bucket` composite token from training. If tonight's pitcher throws a pitch type that maps to an out-of-vocabulary token, PitchGPT returns UNK probability — log it, don't crash.
- **Single-game n.** Live metrics from one game have wide CIs. Phase 2 must frame this as "does calibration survive live conditions?" not "is PitchGPT better than LSTM on 2026 data?" — the latter needs many games.
- **Roster uncertainty.** Relievers are usually unknown pre-game. Phase 0 can flag them when they enter (agent-free — this is logger work).

---

## 9. What success looks like

- Phase 0 pre-game doc committed before first pitch with concrete player-level picks.
- Phase 1 CSV with ≥80% of game's pitches logged (targets total: ~280-300 pitches; acceptable: ~225+).
- Phase 2 post-game report with reliability diagram + LSTM replay + ≥1 real-world Buy-Low/Over-Valued data point + one honest-limitation section.
- Validation-agent PASS after Phase 2.
- Everything committed and pushed.

---

## 10. If something goes wrong and you need to ask the user

The user cleared context deliberately — they don't want to re-explain. Default to the plan above. If you truly need user input (e.g., "which Buy-Low player do you want to highlight?"), state the question tersely and offer a recommended default. Don't ask permission for routine things like "should I push?" — per `feedback_always_push_clean_workspace.md`, the answer is always yes.

---

*Document author: Hunter's prior Claude session (pre-context-clear). Committed on 2026-04-19 in preparation for tonight's live study.*
