# NORTH STAR — CURRENT STATE

**Snapshot date:** 2026-08-10 (post-audit remediation, **regenerated after Batch D** —
kill-criteria adjudication, AdjustedWAR rename + promotion, DPI v2 claims rewrite, PitchGPT
0.6.2 kill consequences).
**Generated from:** `docs/claims/claims.yaml` (the claims registry — the ONLY sanctioned
source for headline numbers, kill criterion K6). Every number below carries its claim id
inline as `[claim:<id>]`; the registry entry's caveat is part of the claim, not optional. All
**37** registry entries are cited somewhere in this file.
**Relationship to `docs/NORTH_STAR.md`:** that file is a layered historical strategy record
(five dated strata, 2026-04-16 → 2026-08). It is preserved unedited for provenance. THIS
file is the current state. When they disagree, this file and the claims registry win.
**Maintenance rule:** regenerate this snapshot whenever `claims.yaml` changes status on any
claim. Do not add numbers here that lack a registry entry.

---

## 1. Mission (unchanged) and current posture

Surface baseball edges the public analytics miss, validated with pre-registered,
adversarially-audited rigor. The 2026-08-10 audit's corollary is now the operating
principle: **the platform's credibility asset is its correction loop, not its marquee
numbers.** Marquee inflation has been stripped from all outward surfaces (Batch A, commit
`912ede6`); honesty infrastructure (claims registry, ledgers, frozen specs) makes the
corrections structural rather than heroic. No new models — validation depth is the
differentiator (plan §7 non-goals).

## 2. Current phase

**Post-audit remediation**, executing `docs/plans/2026-08-10_platform_improvement_plan.md`
(built from `docs/audits/FLAGSHIP_AUDIT_2026-08-10.md`). Kill criteria K1–K6 were
pre-registered in plan §8 before any Batch-C result existed and may not be softened after the
fact.

- **Batch A — DONE** (commit `912ede6`): frozen checkpoints quarantined, outward-surface
  compliance sweep, 2026 resolution spec frozen, DPI reliability quick wins.
- **Batch B — DONE** (commit `d3dbe23`): claims registry + view migration, model
  manifest/registry adoption, holdout ledger, pick ledger + resolver + track-record view,
  reliever board frozen before the Sept-1 deadline, first NORTH_STAR_CURRENT generation.
- **Batch C — DONE** (commit `aab12a4`): DPI v2 bias fixes (WS3.2–3.5), AdjustedWAR honest
  reformulation (WS4.1–4.6), PitchGPT Phase 0.6.2 run (WS5.1). Numbers landed exactly as
  measured; no adjudication language was written into the results docs.
- **Batch D — THIS SESSION:** kill-criteria adjudication (§3.4 below and the dedicated
  verdicts doc), DPI claims rewrite (WS3.6), AdjustedWAR uncertainty + rename + ridge
  promotion (WS4.7), PitchGPT kill consequences + v2 spec pre-registration (WS5.2/5.3 spec
  only — **no training run**), this regeneration.
- **Next deadlines:** K4 resolves after the 2026 regular season (resolution date R = last
  2026 regular-season game + 7 days; publication by R + 14). The WS5.2 training run may start
  only after the `PITCHGPT_V2_SPEC.md` freeze commit exists.

## 3. Active flagships (3) — narrowed claims only

**DPI · AdjustedWAR · PitchGPT.** (AdjustedWAR was named CausalWAR until 2026-08-10; see
§3.2.)

### 3.1 DPI (Defensive Pressing) — `src/analytics/defensive_pressing.py`

Frozen validated expectation model `models/defensive_pressing/xout_v1.pkl` (train 2015–2022,
sha256 `e689bff6ab069474c57df6950ba3ed7d376de8b0a3a7a71861d2a96dc3d3bb39`), registry alias
`defensive_pressing/frozen_validated = v2026.04.18`.

- **Defensible core (DPI v2):** partial r(DPI_v2_final, OAA | BABIP-against) pooled 2023–25 =
  **0.4698** (n=90 team-seasons; per-year 0.3959 / 0.5309 / 0.4213), after the pitching strip,
  jointly-estimated park and the sprint-speed expectation model. Fisher-z 95% [0.2900,
  0.6177]; team-cluster bootstrap [0.2847, 0.5998]. **Both CI floors sit below the 0.30 K2
  line**, and the stage trajectory declines as confounds are removed (0.5725 → 0.4531 →
  0.4805 → 0.4698) — two standing caveats that travel with the number.
  `[claim:dpi_v2_partial_r_oaa_given_babip]`
- **Positioning thesis, tested:** the fielding-alignment block adds **+0.0119** (pre-ban) /
  **+0.0135** (post-ban) holdout AUC over ball-features-only — an order of magnitude above the
  0.002 materiality line in both eras. But per-team positioning value fails split-half
  reliability (α 0.4958 / 0.3088). Honest split: BIP-level positioning signal is real,
  team-level positioning *ranking* is noise. **No team positioning leaderboard, ranking, or
  per-team positioning-runs number may ship.** `[claim:dpi_positioning_alignment_ab]`
- **The confound, quantified:** 10.5–16.4% of team-season DPI v1 variance in 2023–25 is
  opposing-staff contact management, correlating up to r = 0.52 with the v1 ranking. Every DPI
  v1 number carries it. `[claim:dpi_pitching_strip_variance_share]`
- 2025 team DPI vs OAA r = 0.6406, n=30; wild-cluster bootstrap-t CI [0.270, 1.006]; clears
  the Gate 6 line (0.45, retrofitted) on the point estimate only (wild-cluster p = 0.162).
  Raw DPI v1 basis — inherits the stripped confounds. `[claim:dpi_oaa_2025_r]`
- Pooled 2023–25 r = 0.4869 (n=90) clears the RETROFITTED 0.45 threshold by 0.037, wild-cluster
  p = 0.707 against H0 r=0.45 — statistically indistinguishable from its own gate line. Same
  raw-v1 basis caveat. `[claim:dpi_gate6_pooled]`
- The v1 partial r (0.42 / 0.69 / 0.54; 0.41 pooled) is **SUPERSEDED** by the v2 entry above —
  retained, never deleted. `[claim:dpi_partial_r_oaa_given_babip — SUPERSEDED]`
- YoY stability (NARROWED): 2024→25 r = 0.3699 [0.076, 0.640]; Fisher-z mean over ten
  adjacent-season windows 0.4414. Replaces the mislabeled "0.58/0.56/0.64 stability" talking
  point (those were cross-metric DPI-vs-OAA correlations). `[claim:dpi_yoy_stability]`
- Split-half reliability (Spearman-Brown, Fisher-z mean) 0.584; **no 162-game season clears
  the 0.707 bar** — quote season DPI regressed ~42% toward the league mean; raw leaderboard
  gaps overstate true separation ~1.7×. `[claim:dpi_split_half_reliability]`
- **RETRACTED talking point:** the −0.80 DPI-vs-BABIP-against "corroboration" is majority
  circular (shared team variance R² 0.63/0.43/0.65 by construction).
  `[claim:dpi_babip_corroboration — RETRACTED]`
- **Kill verdicts:** K1 and K2 both **DO NOT FIRE** (2026-08-10). The positioning name
  survives; DPI remains a flagship. See §3.4 and
  `docs/models/kill_criteria_verdicts_2026-08.md`.
- Not claimed: no causal or fielder-level attribution, no team positioning ranking, no
  run-value conversion of DPI or of extra-base prevention, and no claim that v2 improves on v1
  *as a number* (the lower v2 partial r is the intended outcome of bias removal).

### 3.2 AdjustedWAR — `src/analytics/adjusted_war_v3.py` (production) + `src/analytics/causal_war.py` (legacy)

**Renamed 2026-08-10** (user-adjudicated, Batch D): the product is **AdjustedWAR** —
*regularized adjustment, not causal identification*. Live surfaces say AdjustedWAR; module
paths, DB cache keys, registry ids and pick-ledger product ids keep the historical
`causal_war` spelling (append-only history is not rewritten). The claim ids below that
begin `causal_war_` are historical registry ids, unaffected by the rename.

**Production model since 2026-08-10:** `adjusted_war_v3` (ridge), registry alias
`production = v2026.08.10`; `frozen_validated` deliberately **unset** — no validation spec
exists for this model, so there is no gate suite it could have passed.

- **Season-forward prediction (the promotion evidence):** PA-weighted RMSE on next-season
  batter wOBA, held-out 2024 + 2025 pooled (n=812) — ridge **.03265** vs the legacy
  formulation **.04567** (Δ −0.013028; h2h 321-143-348; paired-t confidence ≈ 1.0). Marcel
  .03290 and identically-shrunk raw wOBA .03289 are both **ties** with ridge; the legacy
  formulation is worse than a naive league constant (.03748).
  `[claim:adjusted_war_v3_forward_rmse]`
- **Versus Marcel, head-to-head:** ridge 178-133-501 with paired-t confidence **0.567** against
  the pre-registered 0.90 bar → `superiority_claim_allowed = false`. AdjustedWAR **ties**
  Marcel on season-forward forecasting. `[claim:adjusted_war_v3_vs_marcel_forward]`
- **Board lift vs matched-naive**, 17 fully-OOS backfilled windows: ridge +6.78pp / legacy
  +6.45pp unweighted mean, positive in all four config-sides for both formulations — but four
  of eight config-side t-intervals cross zero. `[claim:adjusted_war_v3_naive_lift_17w]`
- **Board lift vs the Marcel-picker**, same windows, **batter channel only**: ridge −8.11pp /
  legacy −8.55pp, negative in all four config-sides. The binding negative result.
  `[claim:adjusted_war_v3_marcel_lift_17w]`
- **Board evidence, in full (K6 — mandatory framing on every board surface):** AdjustedWAR
  beats matched-naive (+6.5pp mean across 17 fully-OOS windows); does not beat the
  Marcel-picker (−8pp, batter channel); ties Marcel on season-forward forecast — no edge claim
  vs Marcel. `[claim:adjusted_war_boards_k6_framing]`
- **No per-player confidence interval ships.** WS4.7 coverage-validated both uncertainty
  layers against realized next-season outcomes: 49.6% (sampling error) and 71.3% (ridge
  posterior) at a nominal 95%, against a pre-registered [90%, 98%] gate. Legacy CausalWAR
  bootstrap intervals were never coverage-validated at all.
  `[claim:adjusted_war_v3_ci_coverage]`
- The v2 bWAR correlation gates (r = 0.6995 / ρ = 0.6165, test 2023–24, n=968; Spearman lower
  CI 0.5701 below the 0.60 line, recorded FRAGILE) are **SUPERSEDED**: they describe the v2
  formulation, not production, and no bWAR-correlation gate has been measured for the ridge.
  They still govern the frozen boards the v2 artifact produced. Correlation with bWAR was
  always agreement, never validation of an edge. `[claim:causal_war_v2_correlation_gates —
  SUPERSEDED]`
- Buy-Low 2023/24→2025: 13/19 = 68.4% survivor-evaluated (NARROWED — **not a validated
  edge**): hit criterion was post-hoc; intention-to-treat is 13/25 = 52%; matched-naive
  mean-reversion controls score 66.5–73.0% on the same pools; the 95% CI includes chance.
  Full-season picks only — does not transfer to mid-season boards.
  `[claim:causal_war_buy_low_68_4]`
- Over-Valued (v2): 13/23 = 56.5%, CI includes chance; do not quote v1's 60.9% against it.
  `[claim:causal_war_v2_over_valued]`
- Reliever leverage tag: 25/32 = 78.1% vs 56.9% within-filter naive — the ONLY
  base-rate-cleared cohort; n=32, hit rule inherits the post-hoc criterion; structurally
  absent from the batter-only 2026 batter board (the 2026 *reliever* board is now frozen —
  §5). `[claim:causal_war_reliever_tag]`
- The old two-window fully-OOS reading (−2.8pp / +10.8pp) is **SUPERSEDED** by the 17-window
  backfill. `[claim:causal_war_oos_windows — SUPERSEDED]`
- **RETRACTED framings:** "68.4% validated" `[claim:causal_war_buy_low_validated —
  RETRACTED]` and the awards-page "70–80% hit rates" `[claim:contrarian_70_80_pct —
  RETRACTED]`.
- **K3 did NOT fire** (2026-08-10): ridge beats the legacy formulation on season-forward
  prediction, and naive-lift is positive — so neither conjunct holds. Consequence executed:
  ridge promoted to production. What it does **not** license: any edge claim against Marcel
  (see the K6 framing claim above). §3.4 and
  `docs/models/kill_criteria_verdicts_2026-08.md`.
- Method status (audit §2.1): the LEGACY implementation is a one-nuisance approximation,
  **not DML**; park/alignment confounders are dead in it. It remains the frozen historical
  formulation behind the 2023–25 evidence boards and every pick already in
  `predictions/picks.jsonl`, which are never rescored.

### 3.3 PitchGPT — `src/analytics/pitchgpt.py` + sim stack

- Per-pitch margins over naive baselines (2025 pitcher-disjoint holdout, matched 10K scale):
  vs 2nd-order Markov +65.17% (v1) / +65.54% (v2), PASS ≥20% gate
  `[claim:pitchgpt_vs_markov2]`; vs frequency heuristic +74.35% / +74.75%, PASS ≥25% gate
  `[claim:pitchgpt_vs_heuristic]`. Next-pitch prediction margins only — not evidence of
  PA-level simulation fidelity.
- vs matched LSTM: +2.57% (v1) / +3.13% (v2) — **FAIL against the pre-registered ≥15% gate,
  decisively.** Direction correct, CIs exclude zero. `[claim:pitchgpt_vs_lstm_10k]` The 1K
  +13.80% headline is retracted and banned. `[claim:lstm_13_80 — RETRACTED]`
- Per-pitch ECE 0.0090–0.0114 (NARROWED/SCOPED): describes the pre-class-calibration stack;
  the production-path ECE has **never been measured**, and the 0.10 top-1 gate is
  near-unfailable at C=2210. It is now also **STRANDED** — the measurement rode Phase 0.6.2's
  holdout contact #13, which the kill voided; measuring it needs a new dated amendment plus
  one of the 2 remaining budgeted 2025 contacts, not authorized.
  `[claim:pitchgpt_per_pitch_ece]` "Perfect mathematical calibration" is retracted.
  `[claim:pitchgpt_perfect_calibration — RETRACTED]`
- PA-level rollout marginal rates FAIL their fidelity gates (K% 0.3339 vs 0.218; BB% 0.1177
  vs 0.0876; HR% 0.0242 vs 0.0321) — no product may quote absolute K%/BB%/HR% from
  rollouts. `[claim:pitchgpt_pa_rates_fail]` The wOBA/PA-length PASSes are retracted as
  tainted (pos-0 calibration fit on the eval cohort) and, after the 0.6.2 kill, are
  **permanently unearned** for v2-era PitchGPT — the clean re-evaluation never ran.
  `[claim:pitchgpt_woba_pa_pass_pre062 — RETRACTED]`
- Outcome head in_play_hit log-loss 2.34 — weaker pass only (< 2.5, misses < 2.0);
  hit-vs-out at pitch time has a structural ceiling.
  `[claim:pitchgpt_outcome_head_in_play_hit]`
- **Phase 0.6.2 = KILLED 2026-08-10 at the pre-registered fit-convergence gate** (2023 fit
  cohort; max per-position class-marginal |delta| 4.418pp after iteration 1, 2.625pp after
  iteration 2, threshold 1.0pp). Phase 0.6 closes as FAIL; no artifact shipped; 2025 was
  never read and contact #13 was never spent. `[claim:pitchgpt_phase062_kill]` Standing
  product scope: flagship claim permanently narrowed to "per-pitch calibrated rollout
  engine"; PA-level absolute-rate products dropped from Tier-A scope; rank/differential
  products (A1 grades, distribution shapes) ship only with the marginal-bias disclosure.
- **K5 FIRED** (2026-08-10) — the only criterion that did. Consequences executed; the
  successor is the **WS5.2/5.3 v2 retrain**, pre-registered in
  `docs/pitchgpt_sim_engine/PITCHGPT_V2_SPEC.md` (chain-rule factorized heads +
  rollout-aware fine-tuning + a failable gate suite). Spec freezes by commit BEFORE any
  training run; K5 grants it ONE lockbox contact against sealed 2026. Holdout tiers: 2025 =
  budgeted, **12 of 14 contacts used** (`docs/holdout_ledger.jsonl`), 2024 = burned dev
  tier, **2026 full season = sealed lockbox** until regular season ends.

### 3.4 Kill-criteria verdicts (adjudicated 2026-08-10, user-approved)

Full text, measured quantities and executed consequences:
**`docs/models/kill_criteria_verdicts_2026-08.md`**.

| Criterion | Governs | Verdict | Headline reason |
|---|---|---|---|
| **K1** | DPI positioning thesis | **does not fire** | AUC conjunct decisively false (+0.0119 / +0.0135 vs a 0.002 line, both eras); α conjunct true but an AND needs both |
| **K2** | DPI attribution | **does not fire** | Pooled partial r 0.4698 vs the 0.30 line, on the point estimate as K2 is worded; CI floors 0.2900 / 0.2847 and the declining stage trajectory recorded as standing caveats |
| **K3** | AdjustedWAR ridge pivot | **does not fire** | Ridge beats legacy on season-forward prediction (conf ≈ 1.0) and naive-lift is positive; the K6 no-edge-vs-Marcel constraint binds |
| **K4** | 2026 boards | **PENDING** | Resolves only after the 2026 regular season, strictly per the frozen spec |
| **K5** | PitchGPT | **FIRED — KILL** | 2023 fit did not converge in 2 permitted iterations (2.625pp vs 1.0pp) |
| **K6** | Global claims discipline | **standing rule** | Not a one-time verdict; enforced by the claims registry, the drift guard and the Marcel protocol |

## 4. Retired / retracted roster — do not re-litigate

| Model | One-line retraction fact | Registry |
|---|---|---|
| **VWR** (viscoelastic workload) | Promotion-day residual AUC 0.768 (64 fits) collapsed same-day to 0.438 OOS (n=563, 2017–24; 2025 holdout 0.493) — permanently off the flagship-candidate list. | `[claim:vwr_flagship_promotion — RETRACTED]` |
| **MechanixAE as injury EWS** | ROC AUC 0.387 — below random — on 30-day pre-IL classification; demoted to descriptive pitcher profiling; no injury or dollar-value claim may render. | `[claim:mechanix_ae_injury_prediction — RETRACTED]` |
| **ChemNet v1+v2** | v1 r = 0.09 vs gate ≥ 0.30 (4 of 5 hard gates FAIL, n=9,836 game-sides); v2 r = 0.155 also failed — synergy scores are statistical noise. | `[claim:chemnet_v1_validation_fail]` (honest negative, active) |
| **Volatility Surface** | Predictability-tax hypothesis is a clean null (r = −0.013, p = 0.89); descriptive visualization only, no edge claim may attach. | `[claim:volatility_surface_null]` (honest negative, active) |
| **Allostatic Load** | 3 of 4 gates FAIL (AUC 0.581 vs ≥0.60; FPR 77.5% vs ≤30%); descriptive workload telemetry, NOT an injury predictor. | `[claim:allostatic_load_ews_fail]` (honest negative, active) |

All four retired-model dashboard views carry demotion banners (Batch A, WS0.3).

## 5. Pre-registration and provenance infrastructure

- **Frozen 2026 resolution spec:** `docs/models/contrarian_2026_resolution_spec.md`, frozen
  at commit `912ede6`, sha256-at-freeze
  `1a27cd0e2d9b7d08c69c5a8a5944602585931121d9396be922e1e519557c760e`. §0–§7 are immutable;
  §8 (deviations log) is append-only — entries 1–3 on record, so the file's current hash
  legitimately differs from the freeze hash. Governs K4: 2026 boards resolve strictly per this
  spec; a miss publishes with the same prominence as a win. The spec and the pick-ledger
  product ids deliberately keep the `causal_war` / `CausalWAR-2026` spelling — the frozen 2026
  boards resolve under the frozen spec regardless of the live-surface rename.
- **Pre-registered PitchGPT successor:** `docs/pitchgpt_sim_engine/PITCHGPT_V2_SPEC.md`,
  frozen on commit, §0–§8 immutable + append-only deviations log. **No training run of any
  kind may start before that freeze commit exists**; none has been performed.
- **Kill-criteria verdicts:** `docs/models/kill_criteria_verdicts_2026-08.md` (K1–K6,
  adjudicated 2026-08-10).
- **Claims registry:** `docs/claims/claims.yaml` + `src/claims.py::get_claim(id)` (raises
  on retracted claims) — **37 entries: 20 active, 6 narrowed, 3 superseded, 8 retracted.**
  K6: no claim ships to dashboard/docs without a registry entry. Enforced by
  `tests/test_claims_drift_guard.py` (view literal scan, banned-string list, and the
  citation check over this file).
- **Marcel floor:** `src/analytics/marcel.py` (literal Marcel) + the WS4.4 scoring protocol
  (PA-weighted RMSE, .010-wOBA tie band, paired-t ≥90%, ≥2 seasons). K6 clause 2: no
  projection-flavored claim without beating it. Currently batter-channel only (pitcher Marcel
  deferred to the marcelR pin, resolution spec §6.7).
- **Pick ledger:** `predictions/picks.jsonl` (append-only; **104 picks** — 50 contrarian 2026
  mid-season batter board, 50 contrarian 2026 mid-season **reliever** board frozen 2026-08-10
  per spec §6 ahead of the Sept-1 deadline, 4 hit-parlay) and
  `predictions/resolutions.jsonl` (4 resolutions: 2 yes / 2 no). Track-record view:
  `src/dashboard/views/track_record.py`, rendering ONLY from these two files, losses as
  prominent as wins.
- **Holdout ledger:** `docs/holdout_ledger.jsonl` — tier-policy header + 12 backfilled 2025
  contacts + the 2026-08-10 note voiding Phase 0.6.2's contact #13. 2025 = budgeted (12/14
  used), 2024 = burned, 2026 = lockbox.
- **Model registry** (`models/registry.json`, the only mutable index; versioned dirs are
  write-once): `defensive_pressing` production `v2026.08.10` / frozen_validated
  `v2026.04.18`; `stuff_model` production `v2026.08.10-inseason` / frozen_validated
  `v2026.08.10`; `pitchgpt` production = frozen_validated `v2026.04.23`; `adjusted_war_v3`
  production `v2026.08.10`, frozen_validated **unset** by design. Frozen artifacts:
  `models/defensive_pressing/xout_v1.pkl` (sha256 `e689bff6…`) and `models/stuff_model.pkl`
  (sha256 `3d8672ec…`). Nightly in-season retrains write only gitignored
  `*_2026_inseason.pkl` siblings; training code raises on frozen-path writes;
  `scripts/verify_artifacts.py` runs first and last in the nightly chain.
- **Contrarian boards:** resolve via `results/edges/contrarian_2026_midseason/latest.json`
  → dated dirs; the mid-season generator now refuses regeneration (deviations-log entry 3)
  because the pick basis is frozen; nightly chain filelock-wrapped; no scheduled task
  currently registered.

## 6. Inventory (measured 2026-08-10, post-Batch-C/D)

The historical "16 models / 25 dashboard views" figures (NORTH_STAR "Current state as of
2026-04-16") no longer match the repo. Measured counts:

- **Dashboard views:** **29** view modules in `src/dashboard/views/` (30 `.py` files minus
  `__init__.py`). Newest: `track_record.py` (Batch B). No new view in Batch C/D — the
  AdjustedWAR page is still the file `views/causal_war.py`, renamed only in its display
  strings and app registration.
- **Analytics modules:** **34** modules in `src/analytics/` (35 `.py` files minus
  `__init__.py`) — up 2 from Batch B (`adjusted_war_v3.py`, `marcel.py`). Breakdown: **24**
  implement a distinct model/index (the 3 flagships across 4 modules — `defensive_pressing`,
  `adjusted_war_v3` + legacy `causal_war`, `pitchgpt` — plus stuff_model, mechanix_ae,
  viscoelastic_workload, allostatic_load, chemnet, volatility_surface, mesi, loft,
  baserunner_gravity, pset, alpha_decay, sharpe_lineup, kinetic_half_life, pitch_decay,
  pitch_sequencing, anomaly, bullpen, matchups, win_probability, projections); **3** are
  PitchGPT sim-stack sub-modules (`pitchgpt_calibration`, `pitchgpt_outcome_head`,
  `pitchgpt_sim`); **3** are baseline comparators (`pitch_lstm`, `pitch_markov`, `marcel`);
  **4** are infrastructure (`base`, `features`, `validation`, `registry`).

These are file counts, not endorsements: only the 3 active flagships carry claims; the rest
are held-back or retired per §3–§4.

## 7. Standing rules (unchanged, load-bearing)

- Research → Plan → Execute; every plan needs a kill criterion; smokes ≠ gates.
- No flagship promotion on small samples (the VWR lesson).
- No umpire-edge products without the ABS-era drift check.
- No rescue attempts on 68.4% or 13.80%; the narrowed claims are the position.
- No post-hoc subgroup rescues after a board result (K3's banned-pattern clause).
- DuckDB single-writer; views in `src/dashboard/views/`, never `pages/`; validation-agent
  after 3+ parallel batches.
