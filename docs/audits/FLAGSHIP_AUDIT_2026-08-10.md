# Flagship Adversarial Audit — 2026-08-10

**Method:** Four independent read-only audit agents (one per flagship + one platform/narrative pass), 2026-08-10. Marked items (†) were independently recomputed from raw artifacts or verified against the live DuckDB, not just read from docs. This document is the evidence base for `docs/plans/2026-08-10_platform_improvement_plan.md`.

**Scope note:** This audit does not retract any flagship. It separates each model's **defensible core** (real, worth defending) from its **inflated marquee** (the number that travels but shouldn't).

---

## 1. DPI (Defensive Pressing)

### Verified
- 2025 team DPI vs Statcast OAA r = 0.6406, CI [0.42, 0.79], n=30. † (recomputed from `results/defensive_pressing/2025_validation/team_rankings_all_years.csv`)
- Per-year DPI-vs-OAA: 2023 = 0.5798, 2024 = 0.5567, 2025 = 0.6406; pooled 2023–25 (n=90) = 0.4869.
- xOut holdout AUC 0.894 (Gate 1), YoY stability 2023→24 r ≈ 0.59 (Gate 4).

### Findings (severity order)
1. **BABIP-against correlation (−0.80) is majority circularity.** DPI's numerator *is* the team BIP out-rate minus an expectation; BABIP-against is the same out-rate inverted. † R²(DPI ~ BABIP-against) = 0.63 / 0.43 / 0.65 by year — DPI shares half to two-thirds of its team variance with raw BABIP by construction. Any residual-on-outcome metric wins this comparison against an externally-measured metric (OAA −0.44). Gate 3 is near-tautological; results doc's "not tautological" claim (`defensive_pressing_results.md:519-528`) materially understates this. **Mitigating (†, not in any doc):** partial r(DPI, OAA | BABIP-against) = 0.42 / 0.69 / 0.54 by year (0.41 pooled) — DPI carries genuine OAA-aligned signal beyond raw outcomes. This is the defensible core.
2. **"Three-year stability 0.58/0.56/0.64" is mislabeled.** Those are per-year DPI-vs-OAA cross-metric correlations. Actual YoY stability was measured once (2023→24, r≈0.59); 2024→25 never computed.
3. **PRODUCTION LEAKAGE (live as of this audit):** `scripts/retrain_active_2026.py:80` trains xOut on 2015–**2026** and overwrites `models/defensive_pressing/xout_v1.pkl` — the frozen validated checkpoint — via the nightly chain. † Pickle inspected: `train_seasons: [2015..2026]`, `fitted_at: 2026-08-10T02:57Z`. Dashboard DPI for 2023–2026 is now scored in-sample; "byte-identical v1" claims (`defensive_pressing_results.md:185-188`, `defensive_pressing.py:50-51`) are false. Validation runner (`scripts/defensive_pressing_validation.py:127-173`) would silently refit-and-overwrite it back.
4. **Gate 6 threshold retrofitted — admitted in spec:** "threshold was set after the first measurement (r = 0.557...) gave a defensible margin" (`defensive_pressing_validation_spec.md:175-178`). Spec and v1 run share the same date (2026-04-18). A fallback threshold (≥0.40) is pre-positioned in `dpi_vs_oaa_yearly.json:194-213`.
5. **Two factually false CI statements, both favorable:** "bottom of the CI [0.307] is well above the 0.45 threshold" (`defensive_pressing_results.md:343-346`); "2025's CI lower bound (0.42) is above ... 0.45" (lines 214-216).
6. **Gate 2 (run-prevention proxy) shares outcome draws with DPI** — positive correlation expected under a pure-luck null; spec's algebraic-identity defense (spec:73-78) doesn't address shared-variance circularity. Only Gates 4 and 6 are non-circular.
7. **Attribution confound:** pitcher contact-management beyond EV/LA/spray, batter speed, and park all land in the "defense" residual. `disagreement_analysis.md:40-46` concedes the alternative reading for SEA/PIT/NYY/CIN, then labels it an "edge claim."
8. **Untested thesis:** "pressing"/positioning framing never tested against `if_fielding_alignment`/`of_fielding_alignment` (present in `pitches` table, unused).
9. **Bootstrap CIs ignore team clustering** (same team's seasons correlate ~0.59); pooled CIs too narrow; pooled r 0.4869 clears the retrofitted 0.45 gate by 0.04.
10. Minor: `extra_base_prevention` is raw XBH rate (no expected-XBH model, contradicts module docstring `defensive_pressing.py:21-22` vs `:1050`); dashboard copy asserts fielder-level mechanisms + unsourced "30-50 runs" (`views/defensive_pressing.py:79-87`) against the spec's own risk flag (spec:234-237); tests are synthetic/directional only, nothing pins gate values or protects the frozen checkpoint.

---

## 2. CausalWAR

### Verified
- Correlation gates pass and were pre-registered: v1 r=0.7089 / ρ=0.6314; v2 (production) r=0.6995 / ρ=0.6165 (n=968). Combined Spearman CI-lower persistently ~0.583–0.590, labeled STILL FRAGILE.
- Buy-Low 13/19 = 68.4%, CI [0.474, 0.843] confirmed in artifacts. † (CI upper bound inconsistent across docs: 0.895 in `causal_war_results.md:52,166` vs 0.843 in JSON artifacts/paper.)
- Naive-lift by window: +7.75 / −2.84 / +10.80pp (Buy-Low); +9.9 / −8.6 / +7.5pp (Over-Valued). † The +7.8pp window (2022→23) is partially in-sample (2022 inside train range); only two fully-OOS windows exist.

### Findings (severity order)
1. **Not DML.** `_fit_dml` (`causal_war.py:592-609`): one nuisance E[Y|W], per-player mean residuals, no treatment model, no residual-on-residual. Spec flags it (spec:105); paper discloses as "one-nuisance approximation" but headlines "first at-scale DML deployment" and "gold-standard identification."
2. **Park and alignment confounders are dead in production.** † Live DB check: `games` table has **0 rows** → `venue_code` constant 0 (`causal_war.py:864, 904-908`); shift flags hardcoded 0 (`:925-926`). Known and chosen: a correct 2026-04-18 fix dropped Spearman 0.63→0.49 and was **reverted to restore the passing gate** (`causal_war_baseline_results.md:453-523`). Paper §2.4 still describes venue as a live confounder; Bello case study (`causal_war_contrarians_2024.md:83-90`) attributes behavior to park knowledge the model doesn't have.
3. **Adjustment is nearly inert:** nuisance R² ≈ 0.001 train / 0.0007–0.0010 test. CausalWAR ≈ wOBA-above-expectation; the contrarian edge is substantially "offense-only per-PA metric vs total-value metric" (conceded at `causal_war_baseline_results.md:128-164`).
4. **68.4% marquee ≈ naive mean-reversion base rate.** Platform's own matched-naive controls: 66.5–73.0% across windows (`hit_rates_by_year.json`). The 2-yr-aggregate 68.4% configuration is the one variant that never received a matched-naive control (`causal_war_contrarian_stability.py:447-507` runs single-season only).
5. **Survivorship-trimmed denominator:** 6 of 25 Buy-Low picks excluded (no 2025 record / below surrogate floors). Intention-to-treat = 13/25 = **52%**. Exclusion asymmetric in model's favor (6 bullish would-be misses vs 2 bearish would-be hits).
6. **Hit criterion is post-hoc:** no hit-rate gate exists in the spec; the rule (WAR ≥ half of 2-yr baseline; ERA≤4.00/IP≥30; OPS≥0.700/PA≥100) originates in dashboard product code (`contrarian_leaderboards.py:280-325`), retro-formalized in the stability script.
7. **"Same 13-of-19 preserved" (v1→v2) is false at player level:** 10/13 overlap (v1: Randy Rodríguez/Gonsolin/Barlow; v2: Bummer/Stanton/Boyd). † Identical 13/19 is coincidence. Paper's delivered-calls exhibit features a v1-only pick; dashboard evidence tab pinned to v1 CSV while production default is v2 (`contrarian_leaderboards.py:43-48`). Production v2 Over-Valued is 13/23 = 56.5%, docs cite v1's 60.9%.
8. **−2.8pp middle window reported honestly, then argued away post-hoc:** autopsy removes worst-performing cohort after seeing outcomes to flip lift positive (`regression_autopsy_2023_2024/report.md:5, 89-100`); paper promotes post-hoc subgroup aggregation ("78/70/79%") to the abstract; DEFENSE GAP called replicating despite a 3/6 middle window.
9. **Paper's mechanism-tag definitions don't match code** (paper §1.2 vs `contrarian_leaderboards.py:196-227`): PARK FACTOR consults no park data; DEFENSE GAP has no −1.5 threshold.
10. **Bootstrap CIs never coverage-validated** (Ticket 4 SKIPPED in every run) yet ship on dashboard and docs; bootstrap refits in-sample while point estimate is cross-fitted (`causal_war.py:697-706`).
11. **2026 mid-season board:** banner discipline is good, but board is batter-only → RELIEVER LEVERAGE GAP (the only base-rate-cleared cohort, 78.1% vs 56.9% within-filter naive, n=32) structurally cannot appear; all 50 rows tag OTHER; pro-rated gates (PA≥204) unvalidated. "Application of the validated methodology" label sits on the cohort where validated evidence is absent.
12. Defensible core: correlation gates; reliever-tag subset; two fully-OOS windows averaging ~+4pp with a sign flip. Tests cover mechanics only — nothing tests hit rule, boards, naive baselines; synthetic fixtures supply venue strings so the dead join can't be caught (`test_causal_war.py:40-43,69`).

---

## 3. PitchGPT

### Verified / reconciled
- Two gate families. **Family A (backbone, next-pitch):** vs LSTM ≥15% **FAIL** (1K +13.80% collapsed to +2.57/+3.13% at matched 10K; retired); vs Markov-2 and heuristic PASS by wide margins; ablation ≥10% FAIL at every scale (best 4.14%); ECE < 0.10 PASS on paper (0.0090–0.0114). **Family B (Phase 0.6 PA-level rollout marginals):** K% 0.3339 vs 0.218 FAIL, BB% 0.1177 vs 0.0876 FAIL, HR% 0.0242 vs 0.0321 FAIL, wOBA/PA-length PASS-but-tainted → overall **FAIL** (`COORDINATION.md:133-141`).
- "ECE passes" and "gates remain FAIL" are both true — different families, mathematically independent (`PHASE_0.6_DIAGNOSIS.md:119-133`).

### Findings (severity order)
1. **Pos-0 calibration taint:** `calibration_class_marginal_pos0.npz` was fit ON the 2025 pitcher-disjoint test cohort (`scripts/pitchgpt_build_pos0_calibration.py:31-38`; same recipe/seed as the eval cohort) and was active when wOBA/PA-length flipped to PASS (commit ee29462). Self-reported in `PHASE_0.6.2_PLAN.md:23-27` ("partly bought with holdout-fitted weights"). Those two PASSes are unearned until re-established clean. A test enshrines the tainted artifact as expected behavior (`tests/test_pitchgpt_sim.py:1100`).
2. **2025 holdout is exhausted:** ~13 substantive evaluation contacts (v1/v2 × 1K/10K, sampling-fidelity, A1 head, D1 diagnostics, sanity runs, 0.6.1 A/B ×2, pos-0 fit itself), **all on the same seed-42 10K-PA subsample**. The 0.6.2 correction's functional form was chosen after observing 2025 failures — fit-on-2023 does not cure that. 2024 is the reserved virgin holdout.
3. **Production-path ECE is unmeasured.** Headline ECE describes probabilities before class_calibration + pos-0 corrections; post-stack ECE never re-measured (est. +0.005–0.010 drift, deferred — `PHASE_0.6_DIAGNOSIS.md:221`). The load-bearing per-pitch claim is stale by two transforms.
4. **The ECE gate is near-unfailable as specified:** threshold 0.10 for a 2210-way predictor with mean confidence 4.8% and 94% of tokens in the [0, 0.1) bin. Passing carries little information; it is the most-quoted gate.
5. **LSTM gate:** pre-registered against 2024 test, evaluated on 2025 (documented but unregistered protocol swap); 1K CI [12.22, 15.51] includes 15 (inconclusive, not near-miss); honest retirement of the number, but response to failure was re-scoping the flagship (sim-engine reframe), not killing it. Claim narrowed serially: context transformer → beats-baselines → calibrated rollout engine → per-pitch-calibrated engine with known-biased PA rates.
6. Minor: `train_pitchgpt` defaults `val_seasons=[2025, 2026]` (`pitchgpt.py:1287-1291`) — legacy landmine; results table quotes both +14.6%/12.7% denominators; v2 1K silently changed `max_games_per_split`; HR% gate flows through a fixed 14.46% HR-fraction proxy the model can't influence; zone-25 semantics mislabeled (missing-data bucket, `pitchgpt.py:470-481`).
7. Genuine credits: 409-pitcher leak self-caught with before/after published (identity-only ppl 53→2583); FAILs recorded as FAILs; static mid-PA context root-caused and fixed (6111cd6); Phase 0.6.2 plan (fit 2023 only, single 2025 eval, pre-registered kill criterion, 2024 virgin) is the most disciplined document in the repo.

---

## 4. Platform / narrative

1. **Two-tier honesty:** internal docs are honest and self-critical; outward surfaces carry pre-narrowing claims:
   - `docs/awards/headline_findings.md:31` still headlines the 13.80% LSTM number that `NORTH_STAR.md:246` (2026-04-24) explicitly banned (locked: 2.57–3.13%). Untouched since 2026-04-19.
   - `src/dashboard/views/mechanix_ae.py:143-157` still sells injury prediction ("MDI 70+ ... strong correlation with upcoming IL stints", "$10M+") for a model that scored AUC 0.387 (below random) on that task; no demotion banner. `chemnet_view.py:92` shows the correct pattern. `viscoelastic_workload.py`, `volatility_surface.py`, `allostatic_load.py` also lack retraction/null disclosure.
   - `docs/awards/summary.md:15,24-26` inflates ("70–80%", "perfect mathematical calibration") what NORTH_STAR carefully bounds ("statistically indistinguishable from 45–50%").
2. **Marquee-number selection pattern:** each flagship's most-quoted number is its least sound (DPI −0.80 = circularity; CausalWAR 68.4% = base rate + survivorship; PitchGPT ECE = unfailable gate on a stale stack), while defensible results (DPI partial-r, reliever-tag lift, per-pitch calibration vs real baselines) go unquoted.
3. **"Edge over gates" pivot (2026-04-18)** was justified partly by VWR's 5/5 PASS, retracted hours later; the pivot itself never revisited.
4. **Banner-gated shipping** is the emergent idiom letting unvalidated applications coexist with the rigor narrative (2026 boards "UNRESOLVED"; `hit_parlay_today.py` "NOT a validated flagship model" yet in the nightly chain).
5. **Trajectory:** award framing quiet since April; August work is live personal-edge consumption (nightly retrains, mid-season boards, parlay, Phillies-first UX). The validation culture persists strongest in PitchGPT 0.6.2.
6. Inventory drift: "25 dashboard views" vs 28 files; "16 models" vs 17–20 modules.

---

## 5. Cross-cutting failure classes (what infrastructure must prevent)

- **F-A. Artifact identity confusion:** one file path serving both frozen-validation and nightly-production roles (DPI xOut; also the pos-0 npz committed as a dangling reference).
- **F-B. Claims drift:** headline numbers hand-copied into dashboards/docs/memory, surviving their own retractions.
- **F-C. Holdout exhaustion:** no ledger of holdout contacts; fixed eval subsample reused across refit iterations.
- **F-D. Post-hoc criteria:** success rules defined after outcomes (buy-low hit rule, retrofitted Gate 6 threshold, autopsy subgroup rescues).
- **F-E. Silent scope substitution:** "validated" labels applied to cohorts/configs the validation never covered (2026 batter-only board; v1 numbers on v2 production).
