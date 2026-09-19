# Next Steps — Ranked Recommendations (2026-08-16)

**Prepared:** 2026-08-16, evening, after the cleanup-day commits (`3e6143c` → `9c88ee7`).
**Basis:** direct exploration of the repo, the DuckDB database (read-only), the claims registry
(40 entries verified: 23 active / 6 narrowed / 3 superseded / 8 retracted), the ledgers, all 29
view modules, the nightly chain and its logs, and the frozen 2026 resolution spec. Every number
here was read from a cited file or queried from `data/baseball.duckdb` today; where something
is reported from code-reading rather than direct execution, I say so.

**Two-sentence diagnosis.** The dashboard currently *under-displays the platform's validated
results and over-displays its unvalidated ones* — the exact inversion of the mission. And
today's headline fix is partially disarmed on arrival: I verified in
`logs/nightly/2026-08-16/status.json` that the scheduled run scored
`precompute: rc=0, effect_ok=false, status="ok"` — the new effect-checks are never consulted
when a step exits 0, and the two required work steps (`daily_refresh.py`, `precompute.py`)
*always* exit 0.

---

## RANKED RECOMMENDATIONS

Timing key: **[NOW]** = this week · **[PRE-SEASON-END]** = must complete before the last 2026
regular-season game (spec: "late September 2026") · **[CAN WAIT]** = offseason.

---

### R1. [NOW] The Surfacing & Truth Sprint — make the dashboard say what the registry says (2–3 days)

**What.** One dashboard-focused batch, seven concrete edits, all content already existing:

1. **Route the two orphaned views.** `src/dashboard/views/matchup_sim.py` and
   `src/dashboard/views/pitch_call_grades.py` are finished, tested
   (`tests/test_matchup_sim_view.py`, `tests/test_pitch_call_grades_view.py`), fixture-fed,
   claim-wired — and absent from `src/dashboard/app.py`'s `page_map`. They are the ONLY
   places in `src/` that call `get_claim()` on any PitchGPT claim; because they are
   unreachable, **zero PitchGPT claims render anywhere on the dashboard** for an active
   flagship. Minutes of work each.
2. **Render AdjustedWAR's one decisive positive.** `claim:adjusted_war_v3_forward_rmse`
   (ridge .03265 vs legacy .04567, h2h 321-143-348, paired-t conf ≈ 1.0, n=812) is active in
   the registry and rendered **nowhere** — `views/causal_war.py` shows only the two negative
   claims (no-CI banner, K6 no-edge framing). Ship it with the mandated ties-Marcel framing
   attached. The page is currently all caveat, no result.
3. **Give the PitchGPT page an evidence panel.** `views/pitchgpt_view.py` is an active
   flagship view with *zero* claims wiring that still asserts "~5% more whiffs and ~10 fewer
   runs allowed per season" — an unsourced impact claim on the very model that fired two
   pre-registered kills. Replace with the registered margins (+65.17/+65.54% vs Markov-2
   PASS `[claim:pitchgpt_vs_markov2]`, +74.35/+74.75% vs frequency PASS
   `[claim:pitchgpt_vs_heuristic]`, +2.57/+3.13% vs LSTM FAIL `[claim:pitchgpt_vs_lstm_10k]`)
   plus the 0.6.2-kill and PA-rates disclosures. `views/defensive_pressing.py:82-92` is the
   template.
4. **Sweep the unvalidated index views.** `sharpe_lineup.py` ("10-20 runs over a season"),
   `kinetic_half_life.py` ("2-3 wins"), `pset.py` ("1.5-2 WAR"), `baserunner_gravity.py`
   ("0.5+ WAR/season"), `bullpen.py` ("3-5 more games per season"), plus `mesi.py`,
   `loft.py`, `alpha_decay.py`, `pitch_decay.py` — confident unsourced impact claims with no
   registry entry. This violates K6's spirit today, and the drift guard cannot see it:
   `tests/test_claims_drift_guard.py:318` bans only six specific retracted strings. Strip or
   caveat the copy AND extend the drift guard with an impact-claim pattern test so the class
   stays dead.
5. **Fix or unroute the Projections view.** `views/projections.py` renders
   `results/projections/projections_2024.csv` (target season 2024) in August 2026 via
   `sorted(glob(...))[-1]`, with no staleness warning and no disclosure that the model
   **failed its validation gates** (`docs/models/projections_results.md` §2.1: RMSE 1.541 vs
   ≤1.5 FAIL; r 0.497 vs ≥0.55 FAIL; ρ 0.394 vs ≥0.50 FAIL — overall FAIL). Retired models
   got banners for less. Either banner it honestly (the batter cohort *did* pass both
   correlation gates and the overlay beats Marcel-only by −0.017 RMSE — say that too) or
   pull it from nav.
6. **Put tonight's products on a surface.** `scripts/hit_parlay_today.py` writes
   `results/hit_parlay/2026-08-16.json` (today: 3 picks, combined_prob 0.4603) every night
   as chain step 4 — and **no view renders it**; it reaches the UI only as an aggregate row
   on Track Record. `scripts/pregame_report.py` (770 lines, runs nightly) prints to a
   terminal nobody reads, writes **no file**, and is referenced nowhere in `src/`. Add a
   "Tonight" panel to the Phillies Hub: today's parlay picks with their honest non-flagship
   label (already written in `src/pick_ledger.py`), and refactor the pregame report to emit
   an artifact the Hub can render (which also gives the nightly an effect-checkable output —
   see R2).
7. **Restructure the nav.** `app.py:136-172` is a flat 31-option radio (including 3
   selectable divider strings that route to `st.error`). DPI — the best-governed page in the
   app — is dead last at position 28; Track Record (the credibility asset) is buried at 12;
   five dead models sit scattered with the same visual weight as Phillies Hub. Labeled
   sections: **Tonight / Flagships & Evidence / Indices (unvalidated) / Retired**.

**Why it beats the alternatives.** This is Path 2 executed literally: surfacing over gate
completion, zero new modeling. It fixes the credibility inversion *and* delivers the product
goal (a Phillies fan gets tonight's parlay, tonight's scouting, and honest flagship evidence).
Everything is packaging of already-validated content; the effort-to-value ratio is the best in
the repo.

**Effort.** 2–3 days. Items 1–2 are hours; item 4's drift-guard extension is the only subtle
part.

**Unblocks / de-risks.** WS6.1 becomes mostly done (R5); K6 exposure closed; the nightly
parlay product stops being invisible; two orphan views stop rotting.

**Honest case against.** It is UI work, and the standing preference is validation depth over
building. Counter: this *is* claims discipline — K6 is a standing rule and roughly nine routed
pages currently violate its spirit; no number gets invented, several get deleted.

---

### R2. [NOW] Re-arm the effect checks — finish the fix that shipped this morning (0.5–1 day)

**What.** Today's session added a cache-sync assertion so "a mid-run death can never again
score ok_verified." True for crash-shaped deaths — but I verified the more common
failure shape sails through:

- **`scripts/nightly_refresh.py:375-389` (`_classify`)**: `if rc == 0: return "ok"` —
  `effect_ok` is consulted only on the *nonzero*-exit path.
- **`scripts/daily_refresh.py`**: `main() -> None`, no `sys.exit` anywhere; every pipeline
  step is wrapped in `except Exception` (lines 121–206). It exits 0 no matter what fails —
  and steps 1–3 (ETL, roster, transactions) increment `steps_completed` unconditionally, so
  the human-readable log prints "All steps completed successfully" even when all three threw.
  `scripts/precompute.py` likewise never exits nonzero (per-model failures caught at 900/1023).
- **Live proof, today's scheduled run** (`logs/nightly/2026-08-16/status.json`):
  `precompute: rc=0, effect_ok=false, status="ok"`, `overall_status: "ok"`.
- Why precompute's check is false: **a UTC/local timestamp bug**. `precompute.py` writes
  tz-aware UTC into a DuckDB `TIMESTAMP`, which stores it converted to local time
  (verified: rows stamped `14:31` for work done at `18:25Z`); `verify_precompute`
  (`nightly_refresh.py:440-452`) compares that local value against UTC step-start, so during
  EDT the check is effectively **permanently false** — and permanently ignored by the
  classifier. Two bugs masking each other. The same tz skew inflates every
  `src/dashboard/cache_reader.py` age calculation by ~4–5 h (the 24 h freshness window is
  really ~20 h; views then silently fall back to live recompute).

**The fix set (small, surgical):**
1. `_classify`: a required step with `rc == 0` and `effect_ok is False` must be **fail** — a
   clean exit with a failed effect check is worse than a dirty exit, not better.
2. Real exit codes in `daily_refresh.py` / `precompute.py` (nonzero when any step failed),
   and honest `steps_failed` accounting for ETL steps 1–3.
3. Fix the timestamp convention (store naive UTC or read back with `AT TIME ZONE`), plus a
   pinning test.
4. **A heartbeat.** Nothing anywhere reads `status.json` — no dashboard tile, no test, no
   alert. The failure class that actually cost five days (task never fired) produces *no
   artifact at all*, which no in-chain assertion can catch. Add a Data Management tile +
   pytest check: today-or-yesterday's `status.json` exists, is < 30 h old, and is
   ok/ok_with_warnings.
5. While in the file: fix the hit-parlay emission order (`scripts/hit_parlay_today.py:345-370`
   appends leg ids to the parlay *before* `emit_pick` succeeds, inside a swallowed
   `except` — a partially-emitted parlay would sit "unresolvable" in the ledger forever,
   invisibly), and archive rather than truncate the dated log dir on re-run/dry-run
   (`nightly_refresh.py:339,510-513` — a noon `--dry-run` currently overwrites the 06:30
   run's `status.json`).

**Why #2.** It re-arms every assertion written today and closes the exact bug-shape the user
asked to hunt. Everything else in this doc sits on this substrate. It is also the cheapest
item on the list relative to what it protects.

**Honest case against.** It's plumbing with no user-visible feature. But the platform's
operating principle is that the correction loop is the asset — an ops layer that scores its
own failed checks "ok" is a correction loop with a hole in it.

---

### R3. [NOW] Revive the live next-pitch product — the only "before the announcers" capability (1–2 days)

**What.** `scripts/live_game_logger.py` polls MLB StatsAPI, runs PitchGPT on every pitch
(top-1/top-3 next-pitch distribution, per-pitch Brier and log-loss vs actual), appends
JSONL/CSV, and feeds the "PitchGPT Live" strip in `views/live_game.py`. It is resumable,
touches no DuckDB — and has run twice ever (`results/live_game/2026-04-19_PHI_vs_ATL/` plus a
dry run), then never again. Build the thin missing piece: auto-resolve tonight's Phillies
`game_pk` from the schedule and start/stop the logger automatically (a second scheduled task
or a dashboard toggle), so Live Game has next-pitch probabilities during every remaining
Phillies game.

**Why.** Next-pitch prediction is PitchGPT's one *validated* strength (+65% vs Markov-2, +74%
vs frequency, pitcher-disjoint holdout) and the literal embodiment of "insights before the TV
announcers": a top-3 pitch distribution on screen before the pitch, with a running
realized-accuracy line keeping it honest (the logger already computes per-pitch Brier and
log-loss, so the display is self-auditing). ~6 weeks of season remain; this capability is
worthless in a repo and only valuable live.

**Effort.** 1–2 days.

**Honest case against.** Machine must be on during games; absolute top-1 accuracy is modest;
the narrowed ECE claim describes the pre-class-calibration stack, so the display must show
realized accuracy rather than assert calibration (the logger's design already supports exactly
that). And it serves one fan — but that fan is the user, and this is the bar they set.

---

### R4. [PRE-SEASON-END — the only hard external deadline] K4 resolution readiness (2–3 days, start by Sept 1)

**What.** Three sub-items, in deadline order:

1. **Marcel control boards — the unmovable one.** Resolution spec §5.6/M1 (batter) and §6.7
   (pitcher): the Marcel control boards "must be generated from pre-2026 inputs and
   **committed before the final 2026 regular-season game**, else `VOID-M`" — and §5.4's V-M
   clause says the miss "is published as a **process miss**." This is implementation work,
   not a paragraph: the pitcher-Marcel variant does not exist (`src/analytics/marcel.py` is
   batter-only; marcelR governs all pitcher details per the spec's TIEBREAK), and §5.6 notes
   the `players` table has **no birth-date column** — the pre-registered fix is an MLB Stats
   API `birthDate` backfill. H3 adjudication depends on these boards existing. Requires a
   dated §8 deviations entry when pinned.
2. **Resolution-path rehearsal.** On or after R, K4 resolves via a full
   `scripts/backfill_2026_war.py` run (fetch → stage → crosswalk → merge) with a daily retry
   ladder through R+14; exhaustion = **V1, the entire resolution voids**. Last exercised for
   the 2026-08-09 staging parquet. Rehearse mid-September against a DB copy: does the B-Ref
   fetch still parse, does the crosswalk still match all 100 board players (absence from
   staging = V2 void per §7.4)? Finding a source-format break in September costs a fix;
   finding it inside the R..R+14 window risks voiding the platform's flagship credibility
   event.
3. **Write the season-end runbook now.** One page listing, with owners and triggers: the
   sprint_speed refresh (2026 currently 520 rows vs 579 for full-2025 — a partial-season
   snapshot), the sealed-2026 lockbox hash-versioning (PITCHGPT_V2_SPEC §5.4), the ECE
   amendment drafting (D2 deferred the *spend* to season end, but the dated amendment must
   be written and reviewed first — draft it in September), and K4 publication into the Track
   Record page by R+14.

**Why.** K4 is the platform's largest bet on its own correction loop: 104 public picks,
frozen spec, miss-published-like-a-win. Everything else in this doc degrades gracefully if
late; this item has two void classes (VOID-M, V1/V2) wired directly to calendar and
operational readiness.

**Effort.** Marcel boards 1–1.5 days (incl. birth-date backfill + marcelR reconciliation +
dated §8 entry); rehearsal 0.5 day; runbook an hour.

**Honest case against.** "The trigger is season end, not today" — today's cleanup plan
correctly deferred it. Agreed for the *execution*; but the Marcel boards are legal to build
NOW ("Marcel's inputs are 2023–2025 only, so it is outcome-blind by construction even if run
later") and the deadline is enforced "for process discipline" — exactly the discipline this
platform sells. Start by Sept 1; on that date this becomes recommendation #1.

---

### R5. [SPLIT — part pre-season-end, part offseason] WS6: build 6.1 before K4 publishes; defer the rest. I disagree with treating WS6 as the monolithic next thing.

**What / position.** The plan of record (`docs/plans/2026-08-10_platform_improvement_plan.md`
WS6, the largest open item) bundles five deliverables. Split them:

- **6.1 "Checking Our Work" page — build by late September** (~2 days after R1, which does
  most of it: Track Record promoted, retired models honestly grouped, evidence panels on all
  three flagships). The genuine increment is one narrative page: the kill record (K5 + the
  v3 successor's own Stage-B kill — "two kills in a row on the same hypothesis family is the
  correction loop working," NORTH_STAR_CURRENT §3.4), the retraction roster, calibration
  plots from the ledgers. **It must exist before K4 publishes (~R+14, mid-October)** so the
  resolution — win or miss — lands into a page built for it, not a scramble.
- **6.5 annual self-review — skeleton only now**, filled at season end from the pick ledger.
- **6.2 versioned write-ups, 6.3 WAR-disagreement living board, 6.4 uncertainty-native UX —
  offseason.** None is season-coupled; 6.4 (HOPs, error-budget bands) is real design work
  that would eat scarce in-season weeks; 6.3 must be built under the K6 framing (descriptive
  divergence, no edge claim), which deserves unhurried care.

**Why this beats "do WS6 next as designed."** WS6's content climax *is* K4 resolution. The
scarce resource between now and late September is live-season time — surfacing (R1/R3) and
deadline work (R4) spend it better. WS6's offseason half loses nothing by waiting; its
pre-K4 half is small once R1 lands.

**Honest case against my split.** WS6 was designed as "content as the moat," and momentum
matters; splitting risks 6.2–6.4 never happening. Mitigation: the season-end runbook (R4.3)
lists them with dates.

---

### R6. [SOON — 1–1.5 days, schedulable anytime] Data-integrity batch: the holes a green check still hides

**What.** Verified today unless marked otherwise:

1. **2026 transactions are ~99% missing.** `transactions` holds ~62–68k rows/year for
   2021–25 but only **94 rows for all of 2026**, with **May–July completely empty**
   (by month: Feb 2, Mar 50, Apr 30, Aug 11). Root cause: the historical bulk load
   (`scripts/ingest_transactions.py`) was league-wide, but the nightly sync
   (`src/ingest/roster_tracker.py::sync_transactions_to_db`) is **Phillies-only with a
   7-day lookback** — and it was dead from April until today's restoration, so the hole is
   permanent until backfilled. Also 7 garbage-dated rows (years 23, 2924, 2925) and a
   future-dated 2026-11-18 row. Backfill Mar–Aug 2026 league-wide, or record a scope
   decision that the table is Phillies-only going forward and banner the Hub panel.
2. **Roster rows never expire** (agent-verified code path,
   `src/ingest/roster_tracker.py:290-350`): `sync_roster_to_db` marks current Phillies but
   never clears departed players, so ex-Phillies stay `team='PHI'` forever — quietly
   corrupting `get_bullpen_status`, the pregame bullpen section, and `hit_parlay_today`'s
   projected lineups. Add a clear-then-set or last_seen sweep + a roster-size sanity check.
3. **Per-model leaderboard staleness.** The nightly runs `precompute.py --tier 1` only.
   Verified: `causal_war` (flagship) cache rows date to **2026-08-11**; ten models to
   **April 12**; retired `volatility_surface` — tier 1 — refreshed today. When cache is
   stale, `cache_reader` returns `None` and views **silently recompute live** with no
   staleness banner, so four-month-old precompute is invisible from the UI. Fix: per-model
   freshness assertions in the nightly effect-check, computed-at stamps on cache-fed views,
   `causal_war` to tier 1, retired models out of the precompute table.
4. **`games` table has 0 rows** with live consumers: `views/data_management.py` displays the
   count; legacy `causal_war.py:992` LEFT JOINs it (venue/home_team/away_team NULL —
   consistent with the audit's "park confounders are dead in the legacy implementation");
   `get_today_games` returns empty silently. Populate or document-and-remove.
5. **The dashboard's "Run Daily Refresh" button cannot work and says it did** (verified:
   `src/dashboard/db_helper.py` caches `get_connection(read_only=True)`;
   `views/data_management.py::_run_daily_refresh` pushes that read-only conn into
   `run_daily_etl`, whose steps swallow the write errors) — it renders
   `st.success("Daily refresh complete! Pitches: 0 | ...")`. This is the button an operator
   reaches for after noticing stale data. Make it spawn the real script or remove it.
6. **`daily_refresh --date` is a no-op** (agent-verified path: `_step_etl` accepts the date,
   prints it, then calls `run_daily_etl(conn=conn)` which hard-codes yesterday). The
   documented gap-patch tool does nothing while reporting success — this is also the
   already-memorialized refresh-gap quirk; fix or delete the flag.
7. **Stale side tables with no cadence owner:** `umpire_assignments` (→ 8/8), `game_weather`
   (→ 8/9), `tj_surgery_dates` (→ 2025-09-20), `sprint_speed` — none in the nightly chain,
   none with watermark assertions. Decide cadence per table or stamp "stale as of" on
   consuming surfaces. `game_weather` becomes load-bearing if R7's weather panel ships.
8. **Governance consistency:** `stuff_model` carries `frozen_validated = v2026.08.10` in
   `models/registry.json` with **no validation spec, no results doc, no claims entry, no
   artifacts under `results/`** — while `adjusted_war_v3` deliberately left the alias unset
   for exactly that reason. Write the one-page spec or unset the alias. Related
   (agent-verified): `verify_artifacts` treats advisory-hash versions as `ok` on mismatch
   and `warn` on missing file, and warns don't fail the gate — so a deleted in-season
   scoring artifact passes `ok=24` while `stuff_model.py` silently falls back to the frozen
   2015-25 model with nothing stamping which artifact scored the board.
9. **Registry-vs-results contradiction:** `claims.yaml` (`causal_war_v2_correlation_gates`
   caveat) says no bWAR-correlation gate "has been measured for the ridge" — but
   `results/causal_war/v3_gates_2026-08-10/v3_gates_metrics.json` measured it (r 0.6932 /
   ρ 0.6052, pass true, Spearman lower CI 0.5572 below the 0.60 bar — fragile like v2's).
   Register with the fragility caveat or correct the registry text.

**Why.** Today's lesson generalized: the platform found one silent failure behind a green
check and fixed that instance. Items 1–6 are more live instances of the same class, three of
them product-facing (bullpen/lineup correctness, the refresh button, flagship cache staleness).

**Honest case against.** None of this blocks a user-visible feature this week — which is why
it ranks below R1–R3 — but items 2 and 3 directly affect the correctness of surfaces R1 and
R3 promote, so at minimum ride those two along.

---

### R7. [SEPTEMBER] Two cheap validation deepeners on data already in hand (1–2 days)

**What.**

1. **Base-rate-test the two untested contrarian mechanism tags.**
   `results/causal_war/mechanism_ablation/report.md` shows pooled hit rates PARK FACTOR
   30/43 = 69.8% and DEFENSE GAP 19/24 = 79.2% (Over-Valued cohorts) — neither has the
   within-filter naive control that RELIEVER LEVERAGE (25/32 = 78.1% vs 56.9%) already
   cleared, and the harness **already exists** (`results/causal_war/tag_filter_baserate/`).
   Run it. Either outcome — one or two new base-rate-cleared cohorts, or two kills — is a
   win under this platform's rules.
2. **Register and conditionally surface the DPI weather cohorts.**
   `docs/edges/weather_segments_2026_04.md`: cold (≤55°F) + strong-inward-wind BIP, n=5,748,
   +1.50% relative Brier lift, 95% CI [+0.93%, +2.08%]; Fenway + inward, n=4,365, +1.06%
   [+0.54%, +1.65%] — a-priori bins, CIs exclude zero, honest global bound already written
   (v2 is *worse* in aggregate, 0.12302 → 0.12346). Zero references anywhere in `src/`. Add
   the two claims to the registry and implement the doc's own shipping rule: a
   weather-adjusted DPI panel only when |wind_parallel| ≥ 5 mph at an outdoor park.
   **Timing honesty:** the cold cohort is an April/late-September phenomenon — value
   concentrates in the season's final two weeks and next April. Depends on R6.7 (weather
   loader into the nightly).

**Why.** "Deepen validation over building models," executed literally on harnesses already
paid for. The tag tests are the single cheapest chance in the repo to mint a new defensible
edge.

**Honest case against.** Both tag cohorts inherit the same post-hoc-criterion caveat the
reliever tag records; a pass licenses only the same narrow phrasing. Expected value is one
new narrow claim, not a headline.

---

### R8. [DECIDE NOW, EXECUTE MOSTLY OFFSEASON] Prune the fleet — and park PitchGPT's remaining budget. *(The recommendation I expect you to disagree with.)*

**What.** Two linked calls:

1. **Cut the primary nav to ~13 pages.** Archive (secondary section, not deletion): the five
   dead models (VWR, ABL, MechanixAE, ChemNet, Volatility Surface) and the weakest
   unvalidated indices (LOFT, Alpha Decay, Pitch Decay PDR, Baserunner Gravity, MESI at
   minimum). Remove retired models from `scripts/precompute.py`'s table — tier-1 nightly
   compute currently refreshes retired `volatility_surface` while flagship `causal_war` sits
   at tier 3, stale since 8/11. Roughly **19 of 27 routed pages are not something a fan
   should act on tonight**; they carry the same visual weight as the flagships and dilute
   the one thing the platform actually owns — the trustworthiness of what it *does* show.
2. **Do not spend the PitchGPT §4.5 second-curriculum run in-season; treat the sim program
   as closed for 2026.** Two consecutive pre-registered kills on the same hypothesis family
   (0.6.2 fit-convergence; v3 Stage-B 1.8852pp vs 1.0pp). The v3 build proved the
   factorization thesis (Stage-A NO KILL, −0.65% NLL at a 10× smaller head) and still
   couldn't clear the rollout-marginal line. The lockbox contact is FORFEIT as of today
   (deviations entry 18); the curriculum budget requires a user-authorized dated deviations
   entry anyway. Spending scarce in-season weeks on attempt #3 — when the per-pitch product
   (R3) already works and needs no retrain — is the lowest-EV use of time on this list.
   Revisit in the offseason with the full sealed-2026 season as the honest evaluation
   cohort.

**Why I expect disagreement, and why I'm making it anyway.** (1) You built these views, and
the culture here is "history is never deleted" — but that norm governs ledgers, claims, and
docs, not *navigation prominence*; git and an Archive section preserve every byte. (2) The v3
build's genuinely exciting sub-result (outcome head 6.7×/29.6× better calibrated than frozen
v2) makes attempt #3 tempting — and a promising sub-metric rescuing a killed program is
precisely the pattern the kill-criteria culture exists to resist. Kills must cost something,
or they aren't kills.

**Honest case against.** Archiving unvalidated-but-live indices removes exploratory surface a
future edge might come from — the compromise is that the Archive stays one click away. And if
the offseason produces a well-motivated single-change curriculum spec, §4.5 is explicitly
still available — parked is not killed.

---

## Offseason queue (explicitly CAN WAIT — listed so nothing gets lost)

| Item | Note |
|---|---|
| ABS-era umpire drift check | Standing rule blocks all umpire-edge products until done; the platform holds 104,599 `umpire_assignments` rows + 1,108 `umpire_tendencies` it is forbidden to use. Best done season-end with the complete first-ABS-challenge-season sample. Unblocks a lane for 2027. |
| `adjusted_war_v3` validation spec | Production model with `frozen_validated` deliberately unset; a pre-registered spec (forward-RMSE + the M2 protocol) would let 2027 boards run gated. Pre-registration-first; offseason-shaped. |
| Production-path ECE contact | Per D2: dated amendment first (draft in September under R4.3), spend one of the 2 remaining 2025 contacts at season end. |
| WS6.2 / 6.3 / 6.4 | Versioned write-ups, WAR-disagreement living board (K6 framing mandatory), uncertainty-native UX. |
| PitchGPT curriculum decision | User decision, dated deviations entry required before any run (R8.2: offseason at earliest). |
| Umpire / weather / TJ / sprint_speed cadence | Decision item from R6.7. |
| Resolver completeness hardening | `scripts/resolve_picks.py` gates only on global `MAX(game_date)` — one ingested game advances the watermark for the whole slate, and a player's missing game resolves `no`/`void` permanently (resolutions are final by design, `src/pick_ledger.py:190-199`). Add a per-game completeness check (the leg's `game_pk` present with plausible PA counts) before resolving. Affects hit-parlay legs only — K4 boards resolve from the B-Ref parquet, not `pitches`. |

---

## Season-end calendar (deadlines as verifiable facts)

| When | What | Source |
|---|---|---|
| **Before final 2026 regular-season game** (~late Sept) | Marcel control boards (batter M1 + pitcher §6.7) generated from pre-2026 inputs and **committed**; else VOID-M, published as a process miss | `docs/models/contrarian_2026_resolution_spec.md` §5.6, §5.4 V-M |
| Mid-September (self-imposed) | `backfill_2026_war.py` rehearsal on a DB copy; crosswalk coverage over all 100 board picks | spec §7.2–7.4 (V1/V2 void classes) |
| Season end | sprint_speed full-season refresh (2026: 520 rows); sealed-2026 lockbox hash-versioning; ECE amendment finalize + contact decision (D2); WS6.5 self-review | remediation report §5; PITCHGPT_V2_SPEC §5.4 |
| R = last game + 7 days | Resolution run per frozen spec (stop dashboard, single writer); retry ladder to R+14 | spec §7 |
| R + 14 | Publication deadline — results doc, ledger appends, Track Record rendering, K4 prominence rule. **WS6.1 page should exist before this date** (R5) | spec §7.7 |

---

# SUPPORTING ANALYSIS

## Q1. What is the platform actually good at right now — and what does it credibly offer tonight?

**Credible and useful tonight** (all verified live-data paths):

- **Matchup Explorer** — 1,851,623-row `matchup_summary` cache, in sync as of today (pitches
  through 2026-08-15), Bayesian-shrunk matchup wOBA with CIs. Genuinely announcer-beating
  for any specific PA.
- **Phillies Hub** — live standings/schedule/roster + opposing-starter scouting from the DB.
  The best pregame surface in the product.
- **Bullpen Strategy, Live Game, Anomaly Alerts** — live feeds; the Live Game PitchGPT strip
  is built but starved (logger dormant since 2026-04-19). Caveat: the bullpen/lineup logic
  sits on the never-expiring roster rows (R6.2).
- **Stuff+** — nightly-retrained leaderboard with honest artifact-provenance captioning
  (`stuff_plus.py:127-139`) — but no validation record anywhere (R6.8).
- **DPI page** — the best-governed page in the app (seven registry claims in an evidence
  expander, provenance caption, an extra-base panel that discloses its own non-registration).
- **Contrarian boards + Track Record** — frozen 2026 batter/reliever boards rendering from
  frozen artifacts with the K6 framing and the resolution-spec SHA pinned; 112 ledgered
  picks, 8 resolutions (3 yes / 5 no), losses as prominent as wins.
- **The nightly hit parlay** — honest, explicitly non-flagship, ledgered every night (today:
  combined p 0.4603; record: legs 3/6, parlays 0/2) — and invisible (R1.6).

**The honest "less than the dashboard implies":** yes, materially. 29 view modules, 27
routed, roughly **19 not actionable tonight** — five carry death banners, eight assert
unmeasured impact with zero claims wiring, one (Projections) renders a failed model's
two-season-stale CSV with no disclosure. Meanwhile the genuinely validated positives —
forward-RMSE, the next-pitch margins, the weather cohorts, the v3 outcome-head calibration
result — render nowhere. The platform's real, defensible core tonight is: fresh matchup data,
Stuff+, DPI-with-caveats, the frozen boards awaiting K4, the reliever-leverage tag (the only
base-rate-cleared cohort), the hit parlay, and a dormant live next-pitch engine. That is a
smaller but much more honest product than the nav implies — R1/R8 make the appearance match
the reality in both directions.

## Q2. Highest-value unexploited edge (data already in hand)

Ranked by validated-signal-per-unit-effort:

1. **PitchGPT's next-pitch head, live** (R3) — the strongest validated margins in the repo
   (+65%/+74%), currently exercised zero times per week.
2. **The mechanism-tag cohorts** (R7.1) — 69.8% and 79.2% pooled hit rates sitting untested
   next to an already-built base-rate harness.
3. **DPI weather cohorts** (R7.2) — CI-backed, a-priori, honest-bounded, unregistered,
   unsurfaced. Seasonal — worth having live before late September.
4. **Umpire tendencies** (offseason) — 104,599 assignments + 1,108 tendency rows, blocked by
   the ABS drift-check rule; the check itself is the unlock and wants full-2026 data.
5. **Weather × park generally** — `game_weather` (26,483 games) is consumed almost nowhere
   outside DPI v2's zero-suppressed features; no pre-registered hypothesis exists, so under
   this platform's rules it is a research candidate, not an edge.
6. **Sprint speed / transactions / alignment** — alignment's BIP-level signal is already
   inside DPI's claims (the team-ranking route is closed by K1's α failure); sprint speed's
   validated role is inside the xOut expectation model; transactions are an ops table and
   currently a data-quality problem (R6.1).

## Q3. Validated-but-unsurfaced (the Path 2 gap, itemized)

Confirmed by grep across `src/` — none of these renders on any routed surface:

| Result | Where it lives | Status |
|---|---|---|
| `adjusted_war_v3_forward_rmse` (.03265 vs .04567, conf ≈ 1.0) | `claims.yaml` active; `results/adjusted_war_v3/forward_eval_2026-08-10/` | Registered, unrendered |
| `pitchgpt_vs_markov2` +65.17/+65.54% PASS; `pitchgpt_vs_heuristic` +74.35/+74.75% PASS; `pitchgpt_vs_lstm_10k` FAIL-but-positive | `results/pitchgpt/2025_holdout_v1_10k/` | Registered, unrendered |
| `pitchgpt_per_pitch_ece`, `pitchgpt_outcome_head_in_play_hit`, `pitchgpt_pa_rates_fail`, `pitchgpt_phase062_kill` | wired into the two **unrouted** views only | Registered, unreachable |
| DPI weather cohorts (+1.50% / +1.06% Brier lift, CIs exclude 0) | `docs/edges/weather_segments_2026_04.md` | Unregistered, unrendered |
| PARK FACTOR 30/43, DEFENSE GAP 19/24 tag cohorts | `results/causal_war/mechanism_ablation/report.md` | Untested vs base rate |
| v3 bWAR gates for the ridge (r 0.6932, pass, fragile ρ CI) | `results/causal_war/v3_gates_2026-08-10/v3_gates_metrics.json` | Contradicted by registry text (R6.9) |
| DPI prospective validation (DPI_N → RA/9_{N+1} r = −0.460; beats OAA, loses to AR(1)) | `results/defensive_pressing/prospective_validation/report.md` | v1-era; needs a v2 re-run before any promotion |
| Daily hit-parlay picks; pregame report | `results/hit_parlay/<date>.json`; terminal stdout only | Products with no surface |
| PitchGPT sampling fidelity (3 of 5 distributional metrics Bonferroni-surviving wins) | `docs/models/pitchgpt_results.md` §5.4 | Unregistered |

And the inverse — **surfaced but unsupported**: the eight index views' impact copy,
`pitchgpt_view.py`'s whiff/runs claims, `bullpen.py`'s "3-5 games," Projections' undisclosed
FAIL, Stuff+'s absent validation record. Per NORTH_STAR_CURRENT §6, only the three flagships
carry claims — these pages assert value that has never been measured.

## Q4. Fragility — where the next matchup-cache-shaped bug is hiding

Same shape = a check that passes for a reason unrelated to what it guards, or a path with no
check at all. Items 1–4 I verified directly today; 5–12 are code-path findings from a
dedicated read of the chain (file:line cited, spot-checked where load-bearing):

1. **The classifier ignores effect checks on exit 0** (`nightly_refresh.py:375-389`), and
   `daily_refresh.py`/`precompute.py` always exit 0. Live proof in today's scheduled run:
   `precompute rc=0, effect_ok=false, status="ok"`. → R2.
2. **UTC/local timestamp skew** makes `verify_precompute` permanently false during EDT and
   inflates all dashboard cache ages ~4–5 h. → R2.
3. **Per-model cache staleness invisible**: aggregate `MAX(computed_at)` today, `causal_war`
   8/11, ten models April 12; stale cache silently falls back to live recompute in every
   consuming view. → R6.3.
4. **Transactions dead-sync masked by historical bulk** (534,831 total rows look healthy;
   2026 has 94). A row-count check would have been green all year. → R6.1.
5. **"All steps completed successfully" with a dead ETL**: `daily_refresh` steps 1–3
   increment `steps_completed` unconditionally; failures never touch `steps_failed`
   (`daily_refresh.py:267-333`). → R2.
6. **The cache-sync assertion is vacuous when the ETL inserts nothing** — network failure or
   an off-day look identical (watermark unmoved, cache trivially in sync); nothing asserts
   "rows exist for a date on which games were played" (`nightly_refresh.py:392-437`). A
   games-played-vs-pitches gap query exists nowhere; the Aug 11–15 gap was found by a human.
7. **No heartbeat**: nothing reads `status.json`; the task is registered
   run-only-when-logged-in; a never-fired task leaves no artifact for any in-chain check.
   → R2.4.
8. **Hit-parlay ledger emission is the unverified tail** — the effect check asserts the JSON
   (written first); the `emit_pick` block that makes picks count is a swallowed `except`
   with a leg-id ordering bug that can strand a parlay unresolvable forever
   (`hit_parlay_today.py:329-372`). → R2.5.
9. **Resolver finality on incomplete data**: global-watermark gate + append-only-final
   resolutions; a missing game resolves `no`/`void` permanently, and the promised
   box-score cross-check is never performed (`resolve_picks.py:57-103`,
   `pick_ledger.py:190-199` — finality verified directly). → offseason queue.
10. **`verify_artifacts` green with the scoring artifact gone**: advisory-hash versions
    report `ok` on mismatch / `warn` on missing, warns don't fail; `stuff_model.py` then
    silently falls back to the frozen model with nothing stamping provenance on the board.
    → R6.8.
11. **The read-only "Run Daily Refresh" button** succeeds with zeros (verified). → R6.5.
    **`daily_refresh --date` no-op** (matches the memorialized refresh-gap quirk). → R6.6.
12. **Log/evidence clobbering**: `--dry-run` or a re-run overwrites the day's `status.json`
    and step logs (today's tree already shows the manual workaround:
    `logs/nightly/2026-08-16_manual`). Contrarian `latest.json` resolvers fail *open* to the
    legacy frozen artifact on any exception, silently (`contrarian_leaderboards.py:133-197`).
    Test-suite gap behind all of it: `scripts/nightly_refresh.py` has **zero tests**, and
    `pyproject.toml` excludes `src/ingest/*` and `src/dashboard/*` from coverage — the two
    layers where every one of these lives.

## Q5. What season end makes impossible or much harder

1. **The Marcel control boards' committed-before-final-game deadline** — after it, VOID-M is
   permanent and publishes as a process miss (R4.1). Nothing else on this list has that
   property.
2. **Live product demonstrations** — R1.6/R3 are worth ~6 more weeks this season, then
   nothing until April. Every week of delay costs ~1/6 of the remaining in-season value.
3. **Weather-cohort surfacing** — the cold+inward cohort effectively reappears in late
   September and then not until April 2027 (R7.2).
4. **Season-end-triggered work that needs prep now** — resolution rehearsal, ECE amendment
   draft, WS6.1 page, sprint_speed refresh planning (R4.3, R5).
5. **The 2026 lockbox seals its value at season end** — hash-versioning, and the offseason
   decision about any future PitchGPT evaluation, both key off that moment.

## Verified-numbers appendix (all measured today, read-only)

- DB: 15 tables; `pitches` 8,258,584 rows, 2015-04-05 → **2026-08-15**; `matchup_summary`
  1,851,623 rows (in sync); `games` **0**; `transactions` 534,831 (2026: **94**;
  2026 by month Feb 2 / Mar 50 / Apr 30 / Aug 11, May–Jul 0); `sprint_speed` 6,624
  (2026: 520); `game_weather` 26,483 (→ 8/9); `umpire_assignments` 104,599 (→ 8/8);
  `tj_surgery_dates` 302 (→ 2025-09-20).
- `leaderboard_cache` by model: five models stamped 2026-08-16 (14:26–14:31 local);
  `causal_war` 2026-08-11 20:43; ten models 2026-04-12.
- Today's scheduled nightly (`logs/nightly/2026-08-16/status.json`): all six steps rc=0,
  overall `ok`; `precompute.verify.effect_ok = false`
  (`leaderboard_max_computed_at 2026-08-16T14:31:09` vs `step_start_utc 18:25:40` — the tz
  bug), classified `ok` anyway (the classifier bug).
- Ledgers: `predictions/picks.jsonl` 112 lines (50 batter + 50 reliever + 12 hit-parlay);
  `resolutions.jsonl` 8 (3 yes / 5 no); `docs/holdout_ledger.jsonl` 14 lines (2025 tier
  12/14; 2026 lockbox 0 contacts).
- Claims registry: 40 entries — 23 active / 6 narrowed / 3 superseded / 8 retracted (counted
  from `docs/claims/claims.yaml`; matches NORTH_STAR_CURRENT exactly).
- Nightly history: `logs/nightly/` contains only 2026-08-04, -10, -11, -16, -16_manual —
  automation genuinely never ran before this month. Task registered; first scheduled fire
  2026-08-17 06:30.
- Today's parlay (`results/hit_parlay/2026-08-16.json`): 3 picks, combined_prob 0.4603.
