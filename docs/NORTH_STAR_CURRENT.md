# NORTH STAR — CURRENT STATE

**Snapshot date:** 2026-08-10 (post-audit remediation, Batch B).
**Generated from:** `docs/claims/claims.yaml` (the claims registry — the ONLY sanctioned
source for headline numbers, kill criterion K6). Every number below carries its claim id
inline as `[claim:<id>]`; the registry entry's caveat is part of the claim, not optional.
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
(built from `docs/audits/FLAGSHIP_AUDIT_2026-08-10.md`). Kill criteria K1–K6 are
pre-registered in plan §8 and may not be softened after results exist.

- **Batch A — DONE** (commit `912ede6`, 2026-08-10): frozen checkpoints quarantined,
  outward-surface compliance sweep, 2026 resolution spec frozen, DPI reliability quick wins.
- **Batch B — IN FLIGHT** (this session): claims registry + view migration, model
  manifest/registry adoption, holdout ledger, pick ledger + resolver + track-record view,
  this palimpsest cleanup (WS2.4).
- **Batch C — PENDING**: DPI v2 bias fixes (WS3.2–3.5), CausalWAR honest reformulation
  (WS4), PitchGPT 0.6.2 run (WS5). GPU tasks serialize; DB single-writer rule throughout.
- **Deadline:** 2026 reliever board must be generated and frozen into the pick ledger
  **before Sept 1** (WS1.3; its criterion is already pre-registered in the resolution
  spec §6, but the board itself is not yet in `predictions/picks.jsonl`).

## 3. Active flagships (3) — narrowed claims only

### 3.1 DPI (Defensive Pressing) — `src/analytics/defensive_pressing.py`

Scored by the frozen `models/defensive_pressing/xout_v1.pkl` (train 2015–2022, sha256
`e689bff6ab069474c57df6950ba3ed7d376de8b0a3a7a71861d2a96dc3d3bb39`).

- **Defensible core:** partial r(DPI, OAA | BABIP-against) = 0.42 / 0.69 / 0.54 by year,
  0.41 pooled 2023–25 — OAA-aligned signal beyond raw outcomes. Audit-computed; carries no
  pre-registered threshold. `[claim:dpi_partial_r_oaa_given_babip]`
- 2025 team DPI vs OAA r = 0.6406, n=30; wild-cluster bootstrap-t CI [0.270, 1.006]. Clears
  the Gate 6 line (0.45, set after the first measurement) on the point estimate only —
  wild-cluster p = 0.162 vs 0.45. Quote with the CI, never as a bare r.
  `[claim:dpi_oaa_2025_r]`
- Pooled 2023–25 r = 0.4869 (n=90 team-seasons) clears the RETROFITTED 0.45 threshold by
  0.037, wild-cluster bootstrap p = 0.707 against H0 r=0.45 — statistically
  indistinguishable from its own gate line. `[claim:dpi_gate6_pooled]`
- YoY stability (NARROWED): 2024→25 r = 0.3699 [0.076, 0.640]; Fisher-z mean over ten
  adjacent-season windows 0.4414. Replaces the mislabeled "0.58/0.56/0.64 stability"
  talking point (those were cross-metric DPI-vs-OAA correlations).
  `[claim:dpi_yoy_stability]`
- Split-half reliability (Spearman-Brown, Fisher-z mean) 0.584; **no 162-game season clears
  the 0.707 bar** — quote season DPI regressed ~42% toward the league mean; raw leaderboard
  gaps overstate true separation ~1.7x. `[claim:dpi_split_half_reliability]`
- **RETIRED talking point:** the −0.80 DPI-vs-BABIP-against "corroboration" is retracted as
  majority circular (shared team variance R² 0.63/0.43/0.65 by construction).
  `[claim:dpi_babip_corroboration — RETRACTED]`
- Pending: WS3 bias fixes (pitcher contact-management strip, park, sprint speed, positioning
  test). Kill criteria K1 (positioning thesis) and K2 (attribution: pooled partial r < 0.30
  after fixes narrows the claim to "descriptive BIP-outcome residual") govern.

### 3.2 CausalWAR — `src/analytics/causal_war.py`

- Correlation gates (pre-registered) PASS: v2 production r = 0.6995 / ρ = 0.6165 vs bWAR,
  test 2023–24, n=968; Spearman lower CI 0.5701 sits below the 0.60 line — recorded
  FRAGILE. Agreement with bWAR, not validation of any contrarian edge.
  `[claim:causal_war_v2_correlation_gates]`
- Buy-Low 2023/24→2025: 13/19 = 68.4% survivor-evaluated (NARROWED — **not a validated
  edge**): hit criterion was post-hoc; intention-to-treat is 13/25 = 52%; matched-naive
  mean-reversion controls score 66.5–73.0% on the same pools; the 95% CI includes chance.
  Full-season picks only — does not transfer to mid-season boards.
  `[claim:causal_war_buy_low_68_4]`
- **RETRACTED framings:** "68.4% validated" `[claim:causal_war_buy_low_validated —
  RETRACTED]` and the awards-page "70–80% hit rates" `[claim:contrarian_70_80_pct —
  RETRACTED]`.
- Over-Valued (v2 production): 13/23 = 56.5%, CI includes chance; do not quote v1's 60.9%
  against v2 production. `[claim:causal_war_v2_over_valued]`
- Reliever leverage tag: 25/32 = 78.1% vs 56.9% within-filter naive — the ONLY
  base-rate-cleared cohort; n=32, hit rule inherits the post-hoc criterion; structurally
  absent from the batter-only 2026 board. `[claim:causal_war_reliever_tag]`
- Fully-OOS window lift vs matched-naive: −2.8pp (2023→24) / +10.8pp (2024→25) — two
  windows, signs flip; no post-hoc subgroup rescues (K3). `[claim:causal_war_oos_windows]`
- Method status (audit §2.1): the implementation is a one-nuisance approximation, **not
  DML**; park/alignment confounders are dead in production. WS4 pivots to regularized joint
  estimation (ridge/mixed-effects), and per plan §4.2 the "Causal" brand is retired for
  player value (rename pending user call). K3 governs: no ridge win + no OOS board lift ⇒
  boards permanently lose the "edge" label.

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
  the production-path ECE (after class_calibration + pos-0 corrections) has **never been
  measured**, and the 0.10 top-1 gate is near-unfailable at C=2210.
  `[claim:pitchgpt_per_pitch_ece]` "Perfect mathematical calibration" is retracted.
  `[claim:pitchgpt_perfect_calibration — RETRACTED]`
- PA-level rollout marginal rates FAIL their fidelity gates (K% 0.3339 vs 0.218; BB% 0.1177
  vs 0.0876; HR% 0.0242 vs 0.0321) — no product may quote absolute K%/BB%/HR% from
  rollouts. `[claim:pitchgpt_pa_rates_fail]` The wOBA/PA-length PASSes are retracted as
  tainted (pos-0 calibration fit on the eval cohort) pending the pre-registered 0.6.2
  re-evaluation. `[claim:pitchgpt_woba_pa_pass_pre062 — RETRACTED]`
- Outcome head in_play_hit log-loss 2.34 — weaker pass only (< 2.5, misses < 2.0);
  hit-vs-out at pitch time has a structural ceiling.
  `[claim:pitchgpt_outcome_head_in_play_hit]`
- Status: Phase 0.6.2 (fit on 2023 only, single 2025 eval, pre-registered kill criterion)
  is the next contact — amended per plan §5.0, governed by K5. Holdout tiers: 2025 =
  budgeted (~13 contacts on record in `docs/holdout_ledger.jsonl`), 2024 = burned dev tier,
  **2026 full season = sealed lockbox** until regular season ends.

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
  §8 (deviations log) is append-only — entry 1 (commit `9253303`) records the freeze SHA,
  so the file's current hash legitimately differs from the freeze hash. Governs K4: 2026
  boards resolve strictly per this spec; a miss publishes with the same prominence as a win.
- **Claims registry:** `docs/claims/claims.yaml` + `src/claims.py::get_claim(id)` (raises
  on retracted claims). K6: no claim ships to dashboard/docs without a registry entry.
- **Pick ledger:** `predictions/picks.jsonl` (append-only; 54 picks as of this snapshot:
  50 contrarian 2026 mid-season board + 4 hit-parlay) and `predictions/resolutions.jsonl`
  (4 resolutions: 2 yes / 2 no). Track-record view: `src/dashboard/views/track_record.py`,
  rendering ONLY from these two files, losses as prominent as wins.
- **Holdout ledger:** `docs/holdout_ledger.jsonl` — tier-policy header + 12 backfilled 2025
  contacts. 2025 = budgeted, 2024 = burned, 2026 = lockbox.
- **Frozen model artifacts:** `models/defensive_pressing/xout_v1.pkl` (sha256 `e689bff6…`,
  above) and `models/stuff_model.pkl` (re-frozen 2026-08-10, train 2015–2025, sha256
  `3d8672ec9a272a04e1794a06aa2f50fbabf6684c6ace92c29f71538b4da58592`). Nightly in-season
  retrains write only the gitignored `*_2026_inseason.pkl` siblings; training code raises on
  frozen-path writes.
- **Contrarian boards:** resolve via `results/edges/contrarian_2026_midseason/latest.json`
  → dated dirs; nightly chain filelock-wrapped; no scheduled task currently registered.

## 6. Inventory (measured 2026-08-10 — corrects NORTH_STAR drift, audit §4.6)

The historical "16 models / 25 dashboard views" figures (NORTH_STAR "Current state as of
2026-04-16") no longer match the repo. Measured counts:

- **Dashboard views:** **29** view modules in `src/dashboard/views/` (30 `.py` files minus
  `__init__.py`), not 25. Newest: `track_record.py` (Batch B).
- **Analytics modules:** **32** modules in `src/analytics/` (33 `.py` files minus
  `__init__.py`), not 16–20. Breakdown: **23** implement a distinct model/index
  (the 3 flagships + stuff_model, mechanix_ae, viscoelastic_workload, allostatic_load,
  chemnet, volatility_surface, mesi, loft, baserunner_gravity, pset, alpha_decay,
  sharpe_lineup, kinetic_half_life, pitch_decay, pitch_sequencing, anomaly, bullpen,
  matchups, win_probability, projections); **3** are PitchGPT sim-stack sub-modules
  (`pitchgpt_calibration`, `pitchgpt_outcome_head`, `pitchgpt_sim`); **2** are baseline
  comparators (`pitch_lstm`, `pitch_markov`); **4** are infrastructure (`base`, `features`,
  `validation`, `registry`).

These are file counts, not endorsements: only the 3 active flagships carry claims; the rest
are held-back or retired per §3–§4.

## 7. Standing rules (unchanged, load-bearing)

- Research → Plan → Execute; every plan needs a kill criterion; smokes ≠ gates.
- No flagship promotion on small samples (the VWR lesson).
- No umpire-edge products without the ABS-era drift check.
- No rescue attempts on 68.4% or 13.80%; the narrowed claims are the position.
- DuckDB single-writer; views in `src/dashboard/views/`, never `pages/`; validation-agent
  after 3+ parallel batches.
