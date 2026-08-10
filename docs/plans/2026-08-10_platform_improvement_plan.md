# Platform Improvement Plan — 2026-08-10

**Status:** APPROVED FOR EXECUTION (pending user go). Built from the 2026-08-10 adversarial audit (`docs/audits/FLAGSHIP_AUDIT_2026-08-10.md`) plus six research tracks (defensive-metrics prior art, valuation/causal prior art, sequence-model prior art, provenance infrastructure, platform landscape survey, repo feasibility scout). Research citations are inlined; execution agents should not need to re-research.

**How to use this doc:** Workstreams are ordered by priority. Each task has: goal, method (with source), key files, acceptance criteria, effort (S ≤ 2h, M ≤ 1 day, L > 1 day). The fan-out map (§9) says what runs in parallel. Kill criteria (§8) are pre-registered — read them before starting the task they govern, and do not soften them after seeing results.

---

## 0. Strategic frame (why this plan, in this order)

The landscape survey found four open lanes this platform is uniquely positioned to own:

1. **A validated pitch-level simulator with a public fidelity report.** FanGraphs Lab shipped a PA-level sim (Apr 2026) with a published fidelity audit; pitch-level work is scattered papers and dead hobby apps. Nobody has published a validated pitch→PA rollout engine — PitchGPT's failing gates are themselves the research contribution.
2. **Prospective, pre-registered validation as the brand.** The honest-methodology audience is proven (FiveThirtyEight "Checking Our Work", BP's Stuff+ team-switcher takedown, Driveline's "Revisiting Stuff+", nflfastR calibration plots) and the niche is vacant in baseball. This platform already practices it internally — it just doesn't publish it.
3. **Team-defense decomposition** (DPI's lane, unowned publicly).
4. **WAR-disagreement as a living product** (no comparable public board exists).

Corollary from the audit: the platform's credibility asset is its correction loop, not its marquee numbers. This plan removes the inflated marquees, fixes the biases behind them, and builds the infrastructure that makes honesty automatic rather than heroic. Explicitly out of scope: any new model. The stuff-model niche is saturated (aStuff+, tjStuff+, PLV, PitchingBot); differentiation is validation depth.

**Corrections to assumptions made before the feasibility scout (do not plan against the old beliefs):**
- **2024 is NOT a virgin PitchGPT holdout.** Burned by `scripts/pitchgpt_ablation.py:87`, `scripts/pitchgpt_calibration_analysis.py:52`, `scripts/train_pitchgpt_v2_ump.py:65-67` (all `TEST_RANGE=(2024,2024)`). The sealed holdout must be 2026 (post-season-end).
- **Park effects need no `games` backfill:** `pitches.home_team` is 100% populated 2015–2026 (30 values); `game_weather.venue` covers 94.3% of game_pks as a cross-check.
- **A model registry already exists:** `src/analytics/registry.py::ModelRegistry` (versioned `_meta.json` sidecars, tested). The nightly writers bypass it. Adoption, not invention.
- Fielding-alignment coverage is 95.3–99.6% per season — usable now. Sprint speed is absent from the DB but `pybaseball.statcast_sprint_speed` imports cleanly (backfill S).

---## WS0 — Stop the bleed (ops integrity; first, before any model work)

### 0.1 Quarantine the DPI checkpoint overwrite — S, CRITICAL
The nightly chain overwrites the frozen validated `models/defensive_pressing/xout_v1.pkl` with a 2015–2026-trained model (`scripts/retrain_active_2026.py:80`; also `src/dashboard/precompute.py:372-376`, and `precompute.py:234-241` for `models/stuff_model.pkl`).
- Retrain a clean frozen artifact on the validated recipe (2015–2024, per `ensure_xout_model` policy `defensive_pressing.py:642-647`) OR restore from git history (`git show` the last committed blob — verify its pickle metadata `train_seasons` before trusting it).
- Move in-season retrains to versioned paths (see 2.1); production/dashboard resolves via registry alias, validation resolves `frozen_validated`.
- Until 2.1 lands, minimal fix: retrain writes `xout_2026_inseason.pkl`; hardcode dashboard to it; leave `xout_v1.pkl` frozen.
- Do NOT `git add` the currently-dirty `models/*.pkl` / `*.pt` binaries — they are the contamination.
- Same decision for `stuff_model.pkl` (non-flagship but same pattern).
**Accept:** frozen artifact hash matches validation-era blob; nightly run no longer modifies it; dashboard states which artifact scored each season (in-sample vs OOS).

### 0.2 Stop tests writing into models/ — S
`tests/test_pitchgpt.py:314` writes `models/pitchgpt_vtest.pt`; `src/analytics/mechanix_ae.py:527` writes `models/mechanix_ae_{tag}.pt` during tests (source of the dirty `mechanix_ae_554430.pt`). Redirect to `tmp_path` fixtures; add a conftest guard that fails any test touching `models/`.
**Accept:** full suite leaves `git status` clean.

### 0.3 Outward-surface compliance sweep — M
Bring every user-facing surface into line with the platform's own retraction rules (audit §4):
- `src/dashboard/views/mechanix_ae.py:143-157`: add demotion banner (copy `chemnet_view.py:92` pattern); delete injury-prediction and "$10M+" claims (model scored AUC 0.387, below random). Same disclosure pass for `viscoelastic_workload.py`, `volatility_surface.py`, `allostatic_load.py`.
- `docs/awards/headline_findings.md:31`: remove the banned 13.80% LSTM claim (NORTH_STAR:246 ban; locked number 2.57–3.13%).
- `docs/awards/summary.md:15,24-26`: replace "70–80%" and "perfect mathematical calibration" with claims-registry language.
- `docs/models/defensive_pressing_results.md:214-216, 343-346`: correct the two false CI statements (0.307 < 0.45; 0.42 < 0.45) — append a dated erratum, don't silently edit.
- `src/dashboard/views/contrarian_leaderboards.py:77,402-408`: detach "validated" from 68.4%; state the CI includes chance and ITT is 13/25; pin the evidence tab to the artifact matching production model version (v2) or label the v1 pin explicitly.
- Results docs: mark the wOBA/PA-length PASSes as TAINTED-pending-0.6.2 wherever quoted.
**Accept:** grep for each banned/inflated number returns only docs that bound it properly; every demoted model's view carries a banner.

### 0.4 Nightly hygiene quick fixes — S
- `retrain_active_2026.py:33-37`: status JSON writes to a hard-coded dead scratchpad path — move to `logs/nightly/`.
- `scripts/contrarian_2026_midseason.py:100-102,448,475`: overwrites `board.csv`/`summary.md` in place — write dated copies (`results/edges/contrarian_2026_midseason/2026-08-10/...`) + `latest` pointer. Prerequisite for WS1 pick ledger.
- Wrap the nightly chain in `filelock` (`nightly.lock`) — schtasks IgnoreNew only governs triggers.
**Accept:** two consecutive nightly runs produce two dated dirs and no in-place mutations outside registry pointers.

---

## WS1 — Pre-register 2026 (time-critical: freeze before the season resolves)

### 1.1 2026 board resolution spec — M, DEADLINE Sept 1
Write `docs/models/contrarian_2026_resolution_spec.md`, committed (commit = tamper-evident freeze), containing, per OSF pre-registration structure (hypotheses / foreknowledge / sampling & stopping / variables / analysis plan / deviations log):
- Exact hit rule per pick type, with formulas. Fix the audit's holes: **ITT accounting — a pick with no resolvable record counts as a MISS on the bullish side and a HIT on the bearish side is NOT allowed either; exits score against the pick's direction** (Buy-Low exit = miss; Over-Valued exit = unresolvable-void, stated explicitly).
- **Matched-naive control** (same position, baseline WAR ±0.3) computed for the SAME window/config as the headline — the 2-yr-aggregate config currently lacks its control (`causal_war_contrarian_stability.py:447-507`).
- **Marcel baseline** (see 4.3) scored on the same pools.
- Resolution date (end of 2026 regular season + 7 days), authoritative source (bWAR via the `backfill_2026_war.py` pipeline), VOID branches, and a Metaculus-style "resolves ambiguous → scores voided, published anyway" clause.
- Foreknowledge declaration: 2026 mid-season boards were generated 2026-08 without a frozen criterion; this spec is written before resolution but after board publication — say so.
**Accept:** spec committed before Sept 1; hash referenced from the dashboard banner.

### 1.2 Pick ledger + resolver — M
Per infra research (bet-tracker schema, Metaculus resolution norms):
- `predictions/picks.jsonl` (append-only, in git): `{pick_id, frozen_utc, product, model, artifact_version, git_sha, subject, market/claim, p or direction, resolution_rule, resolution_source, resolve_by, rule_hash, evidence_class: "prospective"}`.
- `predictions/resolutions.jsonl`: `{pick_id, resolved_utc, outcome: yes|no|void, score fields}`. Never edit picks; resolution is a separate append.
- Backfill-register the existing frozen artifacts: 2026 mid-season boards (from dated copies, 0.4) and `results/hit_parlay/{date}.json` (already dated — good).
- Nightly: parlay + any board emission writes picks; a morning resolver step fills resolutions from the Stats API / DB.
- New dashboard view `src/dashboard/views/track_record.py` (in `views/`, never `pages/`): rolling Brier, calibration buckets, ITT hit rates, computed ONLY from these two files; losses rendered as prominently as wins (Scotty's-Edge grammar).
**Accept:** every nightly pick lands in the ledger with a resolution rule that names a public source and a resolve_by; track-record view renders from ledger only.

### 1.3 Reliever board for 2026 — S/M
The only base-rate-cleared CausalWAR cohort (reliever leverage tag: 78.1% vs 56.9% within-filter naive, n=32) structurally cannot appear on the batter-only 2026 board. Generate the pitcher-side 2026 board (relievers IP<60 equivalent pro-rated), freeze it through 1.1/1.2 with its own pre-registered criterion.
**Accept:** reliever board frozen in the pick ledger before Sept 1.

---

## WS2 — Provenance & claims infrastructure (~25h total per infra research)

### 2.1 Model manifest + registry adoption — M
Extend `src/analytics/registry.py::ModelRegistry` (exists, tested) rather than building new:
- Layout: `models/<name>/v<YYYY.MM.DD>/artifact + manifest.json` (write-once; creator refuses if dir exists); `models/registry.json` = the ONLY mutable file: `{name: {production: vX, frozen_validated: vY, history: []}}` (MLflow immutable-versions + mutable-aliases pattern; GTO layers on later if wanted).
- `manifest.json`: `{artifact, version, sha256, created_utc, git_sha, train_window, data_snapshot: {tables, row_counts}, training_script, spec_version, validation_results_ref}`. The PitchGPT sim stack already does sha256 sidecars (`models/pitchgpt_v2_calibration.json`) — unify the pattern.
- `scripts/verify_artifacts.py`: recompute hashes, fail on mismatch. Nightly runs it first AND last. Also a pytest.
- Migrate loaders: `defensive_pressing.py:49`, `stuff_model.py:33`, dashboard precompute, validation runners → resolve via registry alias.
- Atomic pointer updates: temp file same dir + `os.replace` (same-volume caveat on Windows).
**Accept:** training code is structurally unable to modify a `frozen_validated` artifact; `verify_artifacts.py` green in nightly and pytest.

### 2.2 Claims registry — M
- `docs/claims/claims.yaml`, one entry per reportable number (HF model-index-inspired): `{id, model, artifact_version, metric, value, ci, dataset+hash, source_doc, status: active|narrowed|retracted|superseded, superseded_by, effective, caveat}`.
- `src/claims.py::get_claim(id)` — raises on retracted; returns value + mandatory caveat string.
- Migrate the 7 hand-copied sites found by the scout: `contrarian_leaderboards.py:77,407,632`, `pitch_call_grades.py:70,81`, `matchup_sim.py:71-72`, `chemnet_view.py:92-93`.
- Seed the registry with the audit-corrected claims (defensible cores per audit doc), including retracted entries for: 13.80%, the −0.80-as-corroboration framing, "68.4% validated", pre-0.6.2 wOBA/PA-length PASSes.
- Drift-guard pytest: grep views/ for numeric literals adjacent to metric keywords; fail unless line has `# claim:<id>` or uses `get_claim`.
**Accept:** a retracted claim cannot render on the dashboard; drift-guard test green.

### 2.3 Holdout ledger — S/M
- `docs/holdout_ledger.jsonl` (append-only) + `@holdout_access(dataset, purpose, budget)` decorator: verifies dataset hash, appends `{ts, dataset, sha256, purpose, git_sha, metrics_revealed, contact_number}`, refuses past budget without a logged `--override`.
- Tier policy, declared in the ledger's header entry: **2025 pitcher-disjoint cohort = budgeted validation tier, ~13 contacts already on record (backfill the known contacts from the audit list); 2024 = BURNED (dev tier); 2026 full season = lockbox, sealed until regular season ends, one pre-registered contact per spec version.** Optional Ladder rule for the budgeted tier: only record a metric that beats prior best by preset epsilon (Blum & Hardt 2015).
- Validation specs gain a "Foreknowledge" section (OSF pattern): contact count at spec-freeze.
**Accept:** ledger backfilled with historical 2025 contacts; the 0.6.2 run (5.1) appears in it; any eval script touching a registered holdout without the decorator fails CI.

### 2.4 Palimpsest cleanup — S (optional, low priority)
NORTH_STAR is five dated strategy strata. Write a current-state `docs/NORTH_STAR_CURRENT.md` (or top section) generated per current claims registry; keep history below. Fix inventory drift (28 views, count of models).

---

## WS3 — DPI v2: bias removal + reliability (research: DPI prior-art track)

Context anchor: "Solving DIPS" decomposition of single-season BABIP variance — **luck 44%, pitcher 28%, fielding 17%, park 11%** (SABR; sabr.org/journal/article/the-many-flavors-of-dips-a-history-and-an-overview/). DPI's residual is currently mostly not fielding. Expectation to pre-commit: the win is bias removal and honest CIs, NOT a higher headline correlation — outcome-residual methods are already near their historical ceiling vs tracking metrics.

### 3.1 Quick wins (do first, independent) — S
- Compute 2024→2025 YoY stability from `results/defensive_pressing/2025_validation/team_rankings_all_years.csv` (10-line join; the "stability" claim currently has n=1).
- Split-half reliability: odd/even `game_pk % 2` halves per team-season, Spearman-Brown corrected, per season 2015–2025; adopt the FanGraphs Cronbach-α 0.707 convention. Publish the implied shrinkage coefficient; report regressed DPI alongside raw (UZR "regress ~50%" norm).
- Replace pooled BIP bootstrap with cluster bootstrap: resample games within team-season for team CIs; wild cluster bootstrap (30 clusters = few-cluster territory; `wildboottest` package; Cameron-Miller JHR 2015) for pooled/cross-team CIs. Re-issue Gate 2/3/6 CIs.

### 3.2 Strip pitcher contact-management (highest-impact bias fix) — M
Swartz two-stage (THT: tht.fangraphs.com/adjusting-defense-efficiency-by-the-quality-of-pitching/): build per-pitcher-season peripherals in DuckDB (K%, BB%, net GB rate (GB−FB)/PA, popup rate, SP/RP role); regress game-level DPI residuals on opposing-pitcher peripherals; DPI_v2 = second-stage residual. Use peripherals, NOT pitcher BABIP/xwOBAcon target encodings (contact management is y2y-unreliable; peripherals are the stable carrier). Leakage rule: peripherals from other games only (prior-history or leave-one-game-out windows).

### 3.3 Sprint speed on topped/weak contact — S backfill + S feature
Backfill `pybaseball.statcast_sprint_speed` per season into a new table (pattern: `scripts/ingest_team_oaa.py`); LEFT JOIN on batter+season; apply only on grounders/weak contact per Savant's own recipe (MLB Tech Blog: augmenting xBA with sprint speed; GB xBA r² 0.46→0.57 with speed, fantasy.fangraphs.com/a-sprint-speed-adjustment-for-xba/); impute league mean ~27 ft/s; enforce `monotonic_cst` (faster runner never raises out probability).

### 3.4 Park, jointly estimated — M
`pitches.home_team` is 100% populated — no backfill blocker. Preferred: mixed model on xOut-GBM residuals, `residual ~ (1|home_park) + (1|fielding_team_season)` with empirical-Bayes shrinkage (2026 arXiv 2603.21163 does exactly this simultaneous park+defense estimation; statsmodels MixedLM). DPI_v2 = team BLUP; park falls out as its own estimand. Fallback: sklearn `TargetEncoder` (cross-fits internally) on venue inside the GBM. Do not stack park on top of batted-ball mix independently — they confound (BABIP park factors are 0.696-explained by batted-ball mix).

### 3.5 Test the positioning thesis (the model's name is on the line) — M
DRS-PART blueprint (SIS 2020: Positioning = team credit, Range = player credit): fit xOut-A (ball features only) vs xOut-B (A + `if_fielding_alignment`, `of_fielding_alignment`, batter stand, × spray interactions; coverage 95–99.6%/season). Positioning value = mean(P_B − P_A) per team-season; split eras at 2023-03-30 (shift ban; The American Statistician 2025 DiD estimate: ban ≈ +9pts LHB BABIP). Kill criterion in §8.

### 3.6 Claims rewrite — S (after 3.1–3.5)
Registry entries: lead with partial r(DPI, OAA | BABIP) — the defensible core; retire the −0.80 talking point (circular) and the mislabeled stability claim; add split-half reliability + cluster CIs. Optional later: multinomial total-bases xTB variant (intrinsic-value-of-a-batted-ball template, arXiv 1603.00050) to make extra-base prevention honest — L, do not start until 3.1–3.5 land.

---

## WS4 — CausalWAR → honest formulation (research: valuation track)

Research verdict, adopted: **pivot to regularized joint estimation; do not build per-player DML.** Player identity is the unit, not the treatment; every field that solved this shape (RAPM/Sill 2010, nflWAR multilevel, BP DRC+ mixed models) converged on ridge/mixed-effects with opponent indicators estimated simultaneously. DML is reserved for genuine low-dimensional interventions later (park/team switches).

### 4.1 Immediate: opponent-quality covariate in the current model — S
`pitches.pitcher_id` has 0 NULLs. Compute season-lagged pitcher wOBA-against in-DB (GROUP BY pitcher_id, season over `woba_value/woba_denom` PA-ending rows); join into `_extract_pa_data` as a W feature. (`season_pitching_stats.xwoba` and `.stuff_plus` are 100% NULL — do not plan on them.) Also un-dead the park confounder via `home_team` (the honest version of the reverted 2026-04-18 fix — and accept the correlation gates land where they land; gate-metric optimization over correctness is the audit finding, not a practice to repeat).

### 4.2 AdjustedWAR v3: ridge/mixed-effects joint model — L
Per-PA outcome (wOBA value, or the DRC+ 8-outcome decomposition if ambitious) on sparse one-hot design: [batter_id | pitcher_id | home_park×batter-stand | context FE (TTO, platoon, month, temperature)]. `Ridge`/`RidgeCV` penalizing identity blocks only; λ by season-forward CV (Sill 2010). Batter coefficients → linear weights → runs. This structurally fixes opponent quality, park, and shrinkage in one move, with three published precedents. **Naming decision (user call, pre-registered here): the "Causal" brand is retired for player value — rename (e.g., AdjustedWAR / jWAR) or keep the product name with an explicit "regularized adjustment, not causal identification" methods note. The paper's "gold-standard identification" language goes away either way.**
Evaluation frame (DRC+ lessons): compare shrunk-to-shrunk (regress the bWAR/wOBA baseline identically or the reliability win is an artifact — Hareeb's DRC+ critique); check residual park correlation; check aggregate centering drift; evaluate at season-aggregate level, not per-PA R² (per-PA outcomes are near-Bernoulli noise; the ~0.001 nuisance R² was a wrong diagnostic, not only a wrong model).

### 4.3 Marcel baseline — S
Literal Marcel (tangotiger.net/marcel/): 5/4/3 season weights; +1200 PA league average; reliability = wPA/(wPA+1200); age ±0.006/0.003 around 29; PA projection 0.5·y1 + 0.1·y2 + 200. Reference implementation: github.com/bdilday/marcelR. This is the mandatory floor: "ALL forecasting systems should be treated as if they are nothing more than Marcel, at best" (Tango).

### 4.4 Scoring protocol (pre-registered, applies to all boards) — S
PA-weighted RMSE on wOBA vs naive league-average; head-to-head W-L vs Marcel with .010-wOBA tie band; PA-weighted paired t-test ≥90% confidence; ≥2 seasons before any superiority claim (single-season rankings churn — BtBS gradings). Realistic ceiling to pre-commit: top public systems beat naive by ~.02 wOBA and each other by far less.

### 4.5 Backfill contrarian windows 2015→2025 — M
`data/fangraphs_war_staging.parquet` (actually bref bWAR) is complete 2015–2025. `causal_war_contrarian_stability.py` windows are a module constant (lines 80-84) — mechanically S, but honest pre-2022 windows require per-window nuisance retrains (the script reuses the 2015–2022 checkpoint; any baseline ≤2022 is in-sample, docstring-acknowledged). Target: ~8 fully-OOS windows with matched-naive + Marcel controls for EVERY config including the 2-yr-aggregate.

### 4.6 ITT board scoring + survivorship — S
Every historical and future board scores every name: exits count against the pick direction (or via Lichtman-style imputation of dropout performance from the player's own projection — BP survivor-bias framework); report ITT rate alongside PA-weighted rate. No public buy-low system does this — it is itself a differentiator.

### 4.7 Uncertainty done right — M
Two layers, never conflated (the DRC+ bagging category error): (a) sampling error via openWAR-style per-PA resampling (~3500 replicates); (b) talent uncertainty from the ridge posterior. Coverage-validate on held-out seasons (do nominal 95% CIs cover ~95%?) BEFORE any CI ships — this was spec Ticket 4, skipped in every run while CIs shipped anyway.

---

## WS5 — PitchGPT: through 0.6.2 and beyond (research: sequence-model track)

Research verdict, adopted: **no more patch-stacking.** Post-hoc multiclass calibration at C=2210 is documented to overfit calibration sets (arXiv 2411.02988); exposure bias is a train/inference mismatch no output reweighting removes. If 0.6.2 kills, the path is a retrain.

### 5.0 Amend 0.6.2 BEFORE running it — S
Documented spec amendments (dated, pre-execution, in the deviations log — not silent):
- Evaluate on the FULL 64,460-PA cohort, not the seed-42 10K subsample the whole saga was tuned on. Cost measured: ≈65–70 min on the RTX 3050 (`--n-pa 64460` already exists; 10K logged at ~10 min).
- Add production-path ECE measurement (full stack: T + class_cal + whatever pos-0 replacement ships) to the same run — the shipped-probabilities number has never been measured.
- Register the run in the holdout ledger (2.3) as a budgeted 2025 contact.
- Remove/replace the test that enshrines the tainted pos-0 artifact (`tests/test_pitchgpt_sim.py:1100`) as part of the remediation, and add a provenance-guard test (calibration artifacts must declare fit-cohort ≠ eval-cohort).

### 5.1 Run 0.6.2 and honor the kill criterion — S (compute-bound)
As pre-registered in `PHASE_0.6.2_PLAN.md` + 5.0 amendments. On PASS: PA-level rate products unblock with claims-registry entries. On KILL: per-pitch-only claim locks permanently for v2-era PitchGPT; PA-level absolute-rate products stay dead until 5.2 passes its own gates. Either way, publish the verdict.

### 5.2 v2 retrain design (only on 0.6.2 kill, or as scheduled successor) — L
Two changes, both demonstrated feasible at this scale on a 4GB card:
- **Chain-rule factorized heads**: replace flat 2210 vocab with sequential sub-token decode pitch_type → zone|type → velo|type,zone, per-field sampling masks (soccer LEM pattern, arXiv 2402.06820; Nested Music Transformer shows sequential sub-tokens beat flat AND parallel heads on NLL, arXiv 2408.01180). Parameter count DROPS (283K output matrix → ~40K). Per-head C≈10–26 makes classwise-ECE estimable and single post-hoc calibrators fittable without overfitting.
- **Rollout-aware fine-tuning**: teacher-forced pretrain, then curriculum multi-step autoregressive fine-tune over the PA horizon (2 steps → full PA; the weather-model recipe for exactly this compounding signature, arXiv 2604.01215; two-pass scheduled sampling as cheaper fallback, Mihaylova & Martins ACL 2019). Backbone retrain measured at 8.5 min; the real cost is the downstream chain (outcome head, calibrations, gates — budget M).
- Keep the dynamic mid-PA context fix (6111cd6) in the training data path. Pre-register the v2 spec + gates BEFORE the first training run.

### 5.3 Gate suite that can fail — M
Replace top-1 ECE as the load-bearing gate: classwise-ECE / TACE per factor head (Nixon et al. 2019, explicitly for high class counts), KCE hypothesis test with p-values (Widmann et al. 2019) so "calibrated" is testable, PIT/marginal-calibration for PA-level (Gneiting 2007), nflfastR-style binned calibration plots per count-state. Decision-calibration framing (Zhao et al. 2021) for the sim consumer: calibrate against the K%/BB%/HR%-computing decision functions, which is feasible where full distribution calibration at C=2210 is not.

### 5.4 Holdout scheme — S (policy, with 2.3)
2025 = budgeted tier (exhausted; day-to-day iteration on 2023-fit/2024-dev only). **2026 full season = the lockbox**: sealed now, hash-versioned at season end, one contact per pre-registered spec. All 5.2 iteration happens without touching it.

### 5.5 Public fidelity report — M (after gates pass; WS6 flagship content)
FanGraphs Lab sim validation triad: recreate known WP/RE tables, aggregate-stat R² vs actuals, and volunteer the weakest number unprompted. This artifact — a validated pitch-level sim with published calibration — has no competition (landscape survey).

---

## WS6 — Publish the honesty (after WS0–2 land; content as the moat)

- 6.1 **"Checking Our Work" page**: public page (dashboard tab or repo) with calibration plots + the graded pick ledger + the retraction record — FiveThirtyEight/nflfastR pattern. The 16-built/3-survived ledger becomes front-page content, not buried docs.
- 6.2 **Versioned model write-ups** (TJStats v1/v2/v3 pattern) with explicit train/test years; one shareable graphic per insight, methodology one click deep (UmpScorecards pattern).
- 6.3 **WAR-disagreement living board** as a public product (unowned lane), with severe-testing splits (team-switchers — the BP Stuff+ critique pattern) in every write-up.
- 6.4 **Uncertainty-native UX**: range-band dots and hypothetical-outcome plots (sim rollouts are natively HOP-able) instead of error bars; ZiPS-style pre-committed error budgets ("~20% of players will bust their bands") printed at publication time.
- 6.5 Annual self-review ("Booms & Busts" pattern) scheduled for season end, fed by the pick ledger.

---

## 7. Explicit non-goals
- No new models (stuff-model niche saturated; validation depth is the differentiator).
- No re-litigating retired models (VWR, ChemNet, volatility_surface, MechanixAE-as-EWS, Allostatic Load).
- No umpire-edge products without the ABS-era drift check (standing rule).
- No rescue attempts on the 68.4% or 13.80% numbers; the narrowed claims are the position.
- No betting-stake sizing/bankroll features — the parlay stays a logged heuristic; the deliverable is the honest track record, not wagering advice.

## 8. Pre-registered kill criteria (set now, before results exist)
- **K1 (DPI positioning, 3.5):** if xOut-B (alignment features) adds < 0.002 holdout AUC over xOut-A in BOTH eras AND per-team positioning value fails split-half reliability (α < 0.5), the "pressing/positioning" thesis is dead: rename/reframe DPI as a BIP-conversion residual metric within one session of the result. No third feature-set attempt.
- **K2 (DPI attribution, 3.2–3.4):** if after pitching-strip + park + speed the partial r(DPI_v2, OAA | BABIP) pooled 2023–25 falls below 0.30, DPI's "defense" claim narrows to "descriptive BIP-outcome residual"; flagship status reviewed.
- **K3 (CausalWAR pivot, 4.2/4.5):** if the ridge formulation does not beat the current formulation on season-forward prediction AND mean fully-OOS board lift (vs matched-naive AND Marcel) across the backfilled windows is ≤ 0, contrarian boards lose the "edge" label permanently and ship as descriptive divergence viewers. No post-hoc subgroup rescues (the −2.8pp autopsy pattern is banned by this clause).
- **K4 (2026 boards, WS1):** resolution strictly per the frozen 1.1 spec. If ITT hit rate ≤ matched-naive, the miss is published on the track-record page with the same prominence a win would get.
- **K5 (PitchGPT, 5.1/5.2):** 0.6.2 verdict stands as pre-registered (+5.0 amendments). A v2 retrain gets ONE lockbox contact against the sealed 2026 holdout per its pre-registered spec; failure locks the per-pitch-only claim for the season. No calibration vector may ever be fit on a cohort that any gate is evaluated on (provenance-guard test enforces).
- **K6 (global):** no claim ships to dashboard/docs without a claims-registry entry; no projection-flavored claim without beating Marcel per 4.4.

## 9. Next-session fan-out map
Run `validation-agent` after every 3 parallel batches (standing rule). GPU tasks serialize (one RTX 3050). Suggested batches:

**Batch A — parallel, no interdependencies (agents: 4-5):**
- A1: WS0.1 checkpoint quarantine + 0.4 nightly hygiene (one agent; touches nightly chain)
- A2: WS0.2 test-writes fix
- A3: WS0.3 outward-surface compliance sweep
- A4: WS1.1 resolution spec draft (writes docs only)
- A5: WS3.1 DPI quick wins (YoY, split-half, cluster bootstrap — read-only vs models, writes results/docs)

**Batch B — after A (infra; agents: 3):**
- B1: WS2.1 manifest/registry adoption (depends on A1's layout decision)
- B2: WS2.2 claims registry + view migration + seeding (depends on A3 wording)
- B3: WS2.3 holdout ledger + 2.4 + WS1.2 pick ledger & resolver (schema work shares files)

**Batch C — model lanes, parallel (worktree isolation recommended; agents: 3):**
- C1: WS3.2–3.5 DPI v2 (sequential within lane; sprint-speed backfill first — DB single-writer: stop dashboard, coordinate with any other DB-writing lane so only one writes at a time)
- C2: WS4.1, 4.3, 4.4, 4.6 CausalWAR quick set, then 4.5 backfill (nuisance retrains are CPU/GPU-light), then 4.2 ridge pivot
- C3: WS5.0 amendments + provenance-guard tests (no GPU); then 5.1 run (GPU, serialize with any C1/C2 GPU use)

**Batch D — after C results:** WS3.6 + 4.7 + kill-criteria adjudication + claims-registry updates; then WS5.2/5.3 if triggered; WS6 content as follow-on sessions.

Single-writer DB rule applies throughout: stop the dashboard before backfills/retrains; only one agent writes to DuckDB at a time.

## 10. Effort roll-up
WS0 ≈ 1 day. WS1 ≈ 1 day (deadline-bound). WS2 ≈ 3 days. WS3 ≈ 2–3 days. WS4 ≈ 3–4 days (4.2 is the L). WS5 ≈ 0.5 day through 5.1; +2–3 days if 5.2 triggers. WS6 ≈ ongoing content cadence. Total core (WS0–5): roughly two working weeks of agent-executed effort.
