# Kill-Criteria Verdicts — K1 through K6 (2026-08 remediation series)

**Adjudication date: 2026-08-10.** All readings below were **approved by the user** on that
date and are **binding**: they are recorded here, not re-litigated. Where a reading was a
judgment call between two defensible constructions (K2), the user's chosen reading is stated
as such, together with the quantities that argue the other way — recorded as *standing
caveats*, never as verdict-changers.

**What this document is.** The criteria K1–K6 were pre-registered in
`docs/plans/2026-08-10_platform_improvement_plan.md` §8 **before any Batch-C result existed**
and may not be softened after the fact (plan §5, §8 header). This file is the single place
where each criterion's verbatim text, the quantities it adjudicates, the verdict, and the
consequences that executed are recorded together.

**No new numbers appear here.** Every quantity is quoted from an artifact or results document
that already carries it; each is file-referenced at the point of use. Where a number is
already a claims-registry entry, the entry id is given — that entry, with its mandatory
caveat, is what may be quoted outward (K6).

**Verdict summary**

| Criterion | Governs | Verdict (2026-08-10) | Consequence status |
|---|---|---|---|
| **K1** | DPI positioning thesis (WS3.5) | **DOES NOT FIRE** | Name survives; team-level positioning ranking banned |
| **K2** | DPI attribution (WS3.2–3.4) | **DOES NOT FIRE** (point-estimate reading) | DPI stays a flagship; two standing caveats registered |
| **K3** | AdjustedWAR ridge pivot (WS4.2/4.5) | **DOES NOT FIRE** | Boards keep shipping; K6 no-edge-vs-Marcel framing binds |
| **K4** | 2026 boards (WS1) | **PENDING** — resolves after the 2026 regular season | Frozen spec + 104 ledgered picks in place |
| **K5** | PitchGPT (WS5.1/5.2) | **FIRED — Phase 0.6.2 KILLED** | Consequences executed; v2 retrain spec pre-registered |
| **K6** | Global claims discipline | **STANDING RULE** (not a one-time verdict) | Enforced by claims registry + drift guard + Marcel protocol |

---

## K1 — DPI positioning thesis

### Criterion, verbatim (plan §8)

> **K1 (DPI positioning, 3.5):** if xOut-B (alignment features) adds < 0.002 holdout AUC over
> xOut-A in BOTH eras AND per-team positioning value fails split-half reliability (α < 0.5),
> the "pressing/positioning" thesis is dead: rename/reframe DPI as a BIP-conversion residual
> metric within one session of the result. No third feature-set attempt.

### Measured quantities

Source: `docs/models/defensive_pressing_v2_2026-08.md` §3 (method) and §5.1 (the K1 block,
written with no adjudication language). Artifacts:
`results/defensive_pressing/v2_2026-08/alignment_ab_summary.json`,
`positioning_split_half.csv`, `alignment_ab_test_predictions_{pre,post}_ban.parquet`; model
bundle `models/defensive_pressing/xout_v2_alignment_ab_2026_08_10.pkl`, registry version
`defensive_pressing/v2026.08.10-alignment-ab`. Registered as
`[claim:dpi_positioning_alignment_ab]`.

| Quantity | Pre-ban (2015–2022) | Post-ban (2023–2025) | Criterion line |
|---|---|---|---|
| holdout AUC delta, xOut-B − xOut-A | **+0.0119**, 95% CI [+0.0113, +0.0125], n_test 183,659 | **+0.0135**, 95% CI [+0.0126, +0.0144], n_test 76,437 | < 0.002 in BOTH eras |
| positioning-value split-half α (Fisher-z mean of per-season Spearman-Brown) | **0.4958** (8 seasons) | **0.3088** (3 seasons) | α < 0.5 |
| α, pooled season-centered sensitivity | 0.4617 (n=240) | 0.3557 (n=90) | — |

Era split exactly at the 2023-03-30 shift ban; alignment coverage 97.97–99.86% per season;
AUC measured on per-era internal 20% holdouts (not future-season OOS); positioning values are
descriptive (in-era models scored on all era rows, train rows included).

### Verdict: **DOES NOT FIRE**

K1 is a conjunction and the AUC conjunct is **decisively false**: the alignment block adds
roughly 6–7× the 0.002 materiality line in **both** eras, with paired-bootstrap CIs excluding
the line, and it adds *more* after the shift ban than before. The α conjunct is satisfied in
both eras, but one true conjunct does not fire an AND.

Consequences: the "pressing"/positioning name **survives**; no rename or reframe is triggered;
and per the criterion's final sentence there is **no third feature-set attempt**.

### Standing caveat that ships with the surviving name (not a verdict-changer)

Positioning signal is real at the **BIP level** and unreliable at the **team level** — α
0.4958 / 0.3088, both under the criterion's 0.5 bar and far under the FanGraphs 0.707
convention, with a negative season in the post-ban series (2024, −0.167). Enforced
consequence, carried in the mandatory caveat of `[claim:dpi_positioning_alignment_ab]`: **no
team positioning leaderboard, no team positioning ranking, and no per-team positioning-runs
number may ship.**

Verdict also recorded at `docs/models/defensive_pressing_v2_2026-08.md` §7.1.

---

## K2 — DPI attribution

### Criterion, verbatim (plan §8)

> **K2 (DPI attribution, 3.2–3.4):** if after pitching-strip + park + speed the partial
> r(DPI_v2, OAA | BABIP) pooled 2023–25 falls below 0.30, DPI's "defense" claim narrows to
> "descriptive BIP-outcome residual"; flagship status reviewed.

### Measured quantities

Source: `docs/models/defensive_pressing_v2_2026-08.md` §1.3, §4.4 and §5.2 (the K2 block).
Artifacts: `results/defensive_pressing/v2_2026-08/k2_final_partial_r.csv`,
`team_season_dpi_v2_blup.csv`, `park_effects_mixedlm.csv`; covariates from
`results/defensive_pressing/2025_validation/team_rankings_all_years.csv`. Registered as
`[claim:dpi_v2_partial_r_oaa_given_babip]`.

The **primary** number was pre-committed in the docstring of
`scripts/dpi_v2_park_mixedlm.py` *before any fit ran*: partial r(P2 team BLUP, OAA |
BABIP-against), pooled 2023–25, from the full-window (2015–2025) MixedLM fit.

| Quantity | Value |
|---|---|
| **PRIMARY: partial r(DPI_v2_final, OAA \| BABIP), pooled 2023–25** | **0.4698** (n = 90 team-seasons) |
| criterion line | 0.30 |
| Fisher-z 95% CI | [0.2900, 0.6177] |
| team-cluster pairs bootstrap 95% CI (30 franchises, 5,000 draws) | [0.2847, 0.5998] |
| per-year | 2023 0.3959 · 2024 0.5309 · 2025 0.4213 |
| sensitivity (2023–25-only MixedLM fit) | 0.4299 [Fisher 0.2435, 0.5857; cluster 0.2780, 0.5420] |
| context (strip + park, no speed) | 0.4805 [Fisher 0.3026, 0.6261; cluster 0.3004, 0.6070] |

Stage trajectory on the like-for-like season-centered basis (§4.4): v1 season-centered
**0.5725** → + pitching strip **0.4531** → + park jointly estimated **0.4805** → + speed
variant expectation **0.4698**.

### Verdict: **DOES NOT FIRE** — adjudicated on the point estimate, as the criterion is worded

The criterion names "the partial r … falls below 0.30". The pre-committed pooled point
estimate 0.4698 sits 0.17 **above** the line. DPI's "defense" claim is therefore **not**
narrowed to "descriptive BIP-outcome residual" by K2, and the flagship-status review clause is
not triggered — **DPI remains a flagship**.

This was the user's adjudicated reading (2026-08-10) of a criterion whose wording does not
specify point estimate vs interval. The quantities arguing the other way are recorded below in
full and were not treated as changing the verdict.

### Two standing caveats (recorded, may not be dropped)

1. **Both CI floors sit below the line.** Fisher 0.2900 and team-cluster 0.2847 are 0.010 and
   0.015 *below* 0.30 — the estimate is **not statistically separated from its own criterion**.
   The verdict rests on the point estimate because that is the quantity K2 names.
2. **The stage trajectory declines** as confounds are removed (0.5725 → 0.4531 → 0.4805 →
   0.4698). That was the pre-committed WS3 expectation (bias removal, not a higher headline
   correlation), but it means a further bias-removal stage could carry the pooled value under
   the line. **Any future stage re-adjudicates K2 on this same pre-registered wording.**

Both caveats are carried in the mandatory caveat of
`[claim:dpi_v2_partial_r_oaa_given_babip]` and restated at
`docs/models/defensive_pressing_v2_2026-08.md` §7.2.

### Related consequences executed under this verdict

The v1 partial-r entry `[claim:dpi_partial_r_oaa_given_babip]` is **superseded** (never
deleted) by the v2 entry; `[claim:dpi_oaa_2025_r]` and `[claim:dpi_gate6_pooled]` gained dated
pointers noting they are raw DPI v1 correlations carrying the stripped confounds; the
opposing-staff confound is itself now quotable as
`[claim:dpi_pitching_strip_variance_share]` (10.5–16.4% of team-season DPI v1 variance,
r up to 0.52 with the v1 ranking). Full consequence list:
`docs/models/defensive_pressing_v2_2026-08.md` §7.3–§7.4.

---

## K3 — AdjustedWAR (formerly CausalWAR) ridge pivot

### Criterion, verbatim (plan §8)

> **K3 (CausalWAR pivot, 4.2/4.5):** if the ridge formulation does not beat the current
> formulation on season-forward prediction AND mean fully-OOS board lift (vs matched-naive AND
> Marcel) across the backfilled windows is ≤ 0, contrarian boards lose the "edge" label
> permanently and ship as descriptive divergence viewers. No post-hoc subgroup rescues (the
> −2.8pp autopsy pattern is banned by this clause).

### Measured quantities

Source: `docs/models/adjusted_war_v3_2026-08.md` §3, §5 and §6 (the K3 block, measurements
only). Artifacts: `results/adjusted_war_v3/forward_eval_2026-08-10/forward_eval.json`,
`results/adjusted_war_v3/boards_2026-08-10/ridge_boards_summary.json`,
`results/causal_war/backfill_windows_2026-08-10/windows_summary.json`. Registered as
`[claim:adjusted_war_v3_forward_rmse]`, `[claim:adjusted_war_v3_vs_marcel_forward]`,
`[claim:adjusted_war_v3_naive_lift_17w]`, `[claim:adjusted_war_v3_marcel_lift_17w]`.

**Limb 1 — season-forward prediction** (held-out 2024 + 2025, pooled n=812, PA-weighted RMSE,
follow-up PA ≥ 100; WS4.4 protocol, `src/analytics/marcel.py`):

| Predictor | pooled RMSE |
|---|---|
| ridge (AdjustedWAR v3, λ*=400) | **0.03265** |
| legacy current_v1 | **0.04567** |
| Marcel | 0.03290 |
| identically-shrunk raw wOBA (diagnostic) | 0.03289 |
| naive league constant | 0.03748 |

Ridge − current delta **−0.013028**; head-to-head |error| **321-143-348**; PA-weighted
paired t = 8.58 (n_eff ≈ 670), confidence ridge better **≈ 1.0**. Ridge better in *both*
held-out seasons individually. Versus Marcel: ridge **178-133-501** with paired-t confidence
**0.567** against the pre-registered 0.90 bar → `superiority_claim_allowed = false`.

**Limb 2 — mean fully-OOS board lift across the 17 backfilled windows** (9 single-season +
8 two-year-aggregate; identical pools, hit rules, ITT accounting):

| Control | legacy unweighted mean | ridge unweighted mean |
|---|---|---|
| vs ITT-consistent matched-naive | **+6.45pp** (positive in all four config-sides) | **+6.78pp** (positive in all four config-sides) |
| vs Marcel-picker (batter channel only) | **−8.55pp** (negative in all four) | **−8.11pp** (negative in all four) |

Four of the eight config-side t-intervals cross zero (§5 tables).

### Verdict: **DOES NOT FIRE**

Both limbs must hold for a conjunction to fire; **neither** does.

| Limb | Measured | Fires? |
|---|---|---|
| "ridge does not beat the current formulation on season-forward prediction" | Ridge beats current decisively (Δ RMSE −0.013028, confidence ≈ 1.0, both seasons) | **NO** |
| "mean fully-OOS board lift (vs matched-naive AND Marcel) ≤ 0" | vs matched-naive **positive** in all four config-sides for both formulations; vs Marcel negative | **NO** — the naive limb is positive, so the AND over both controls is not ≤ 0 |

The contrarian boards keep shipping and do **not** become descriptive-only viewers.

### Binding constraint: the K6 no-edge-vs-Marcel consequence

K3 not firing licenses nothing about Marcel. The Marcel half of limb 2 is unambiguously
negative and the pre-registered WS4.4 protocol denies a forecasting superiority claim
(paired-t confidence 0.567 vs the 0.90 bar). **Every board surface therefore states,
verbatim-equivalent:**

> beats matched-naive (+6.5pp mean across 17 fully-OOS windows); does not beat the
> Marcel-picker (−8pp, batter channel); ties Marcel on season-forward forecast — no edge claim
> vs Marcel

registered as `[claim:adjusted_war_boards_k6_framing]` and rendered through
`src.claims.get_claim` on the board surfaces (`src/dashboard/views/contrarian_leaderboards.py`).
Standing caveats travelling with the verdict: four of eight config-side t-intervals cross
zero; the Marcel control has **no pitcher channel** (deferred to the marcelR pin, frozen
resolution spec §6.7); ridge's forecasting win over *identically shrunk* raw wOBA is only
~0.0002–0.0007 RMSE. The "edge" wording is spent against Marcel, permanently, unless a
Marcel-beating result is measured under WS4.4.

### Consequences executed under this verdict

- **Rename** (user-adjudicated 2026-08-10): the "Causal" brand is retired for player value —
  the product is **AdjustedWAR** on every live surface. Module paths, DB cache keys, registry
  ids and pick-ledger product ids keep the historical `causal_war` spelling; append-only
  history is not rewritten. Registry ids beginning `causal_war_` are historical ids,
  unaffected by the rename.
- **Promotion**: AdjustedWAR v3 (ridge) is the production player-value model — `models/registry.json`
  alias `adjusted_war_v3/production = v2026.08.10`; `frozen_validated` deliberately **unset**
  (no validation spec exists for it, so there is no gate suite it could have passed).
  `[claim:causal_war_v2_correlation_gates]` is **superseded** in consequence — it describes the
  v2 formulation and may not be quoted as a property of production; no bWAR-correlation gate
  has been measured for the ridge.
- **No per-player CI ships** (WS4.7, a separate pre-registered gate, not K3): empirical
  coverage 49.6% (sampling error) and 71.3% (ridge posterior) at a nominal 95% against a
  [90%, 98%] gate. `[claim:adjusted_war_v3_ci_coverage]`.
- The frozen 2026 boards and every pick already in `predictions/picks.jsonl` are **not**
  rescored; they resolve under the frozen resolution spec against the legacy scores they were
  frozen with.

Verdict also recorded at `docs/models/adjusted_war_v3_2026-08.md` §8.

---

## K4 — 2026 contrarian boards

### Criterion, verbatim (plan §8)

> **K4 (2026 boards, WS1):** resolution strictly per the frozen 1.1 spec. If ITT hit rate ≤
> matched-naive, the miss is published on the track-record page with the same prominence a win
> would get.

### Verdict: **PENDING** — not adjudicable in 2026-08

K4 adjudicates on 2026 full-season outcomes that do not exist yet. Per
`docs/models/contrarian_2026_resolution_spec.md` §7.1, the resolution date **R** = the date of
the last 2026 MLB regular-season game actually played **+ 7 days** (formula, not a calendar
guess), with publication due by **R + 14**. Nothing in Batch C or Batch D touches K4, and no
2026 board number may be adjudicated before R.

### Pre-registration state as of 2026-08-10 (what is already locked)

- **Frozen spec:** `docs/models/contrarian_2026_resolution_spec.md`, freeze commit
  `912ede6a8a179284ffcc5c1e4039c9c59078c24c`, file sha256 at freeze
  `1a27cd0e2d9b7d08c69c5a8a5944602585931121d9396be922e1e519557c760e` (= the `rule_hash`
  carried by every 2026 pick). §0–§7 are immutable; §8 is an append-only deviations log
  (entries 1–3 on record), so the working-tree hash legitimately differs from the freeze hash.
- **Picks frozen in the ledger:** `predictions/picks.jsonl` carries **104** append-only picks —
  50 `contrarian_board_2026_midseason` (batter), 50
  `contrarian_board_2026_midseason_reliever` (frozen 2026-08-10, deviations-log entry 2,
  meeting the plan's Sept-1 deadline), and 4 `hit_parlay`. `predictions/resolutions.jsonl`
  holds 4 resolutions (2 yes / 2 no), all hit-parlay.
- **Scoring machinery pre-registered:** ITT accounting with the exit asymmetry (spec §5.3),
  matched-naive controls on the same window/config (§5.5), the Marcel-picker control (§5.6,
  M1/M2), exhaustive VOID branches and the ambiguity clause (§5.4), and the reliever criterion
  (§6).
- **Prominence rule already implemented:** `src/dashboard/views/track_record.py` renders from
  the two ledger files only, with losses at the same prominence as wins.

**Rename policy note:** the frozen spec §0–§7 and the pick-ledger product ids retain the
`causal_war` / `CausalWAR-2026` spelling deliberately. The 2026 boards resolve under the frozen
spec regardless of the live-surface rename to AdjustedWAR.

---

## K5 — PitchGPT

### Criterion, verbatim (plan §8)

> **K5 (PitchGPT, 5.1/5.2):** 0.6.2 verdict stands as pre-registered (+5.0 amendments). A v2
> retrain gets ONE lockbox contact against the sealed 2026 holdout per its pre-registered
> spec; failure locks the per-pitch-only claim for the season. No calibration vector may ever
> be fit on a cohort that any gate is evaluated on (provenance-guard test enforces).

### Measured quantities

Source: `docs/models/pitchgpt_phase062_results.md` §1 and §6. Pre-registered protocol:
`docs/pitchgpt_sim_engine/PHASE_0.6.2_PLAN.md` §§1–8 (frozen 2026-08-04) + §10 amendments
A1–A8 (recorded 2026-08-10, pre-execution); the §6 kill criterion was never amended. Executed
2026-08-10 by `scripts/pitchgpt_fit_rollout_calibration.py` (exit code 2 = kill-signal branch);
audit trail `results/pitchgpt/rollout_calibration_fit_2023/{fit_audit.json, report.md}`.
Registered as `[claim:pitchgpt_phase062_kill]`.

Fit cohort: 2023 pitcher-disjoint (2,247 train-split pitchers excluded), 19,625 eligible PAs,
10,000 sampled seed 42, 100 rollout samples/PA, horizon 6, T=1.0. **2025 was never read.**

| Iteration | max per-position class-marginal \|delta\| vs empirical | Converged (≤ 1.0pp)? |
|---|---:|---|
| — (pre-fit reference, raw T-softmax) | 16.37pp | n/a |
| 1 (W1) | **4.418pp** | NO |
| 2 (W2) | **2.625pp** | **NO → KILL** |

Pre-registered threshold **1.0pp**; two fixed-point iterations permitted, no third. Terminal
delta is ~2.6× the threshold — not a borderline read. The spec defines no CI for the kill
quantity (deterministic threshold on measured marginals).

### Verdict: **FIRED — Phase 0.6.2 KILLED; Phase 0.6 closes as FAIL**

Kill under `PHASE_0.6.2_PLAN.md` §6, first disjunct ("The 2023 fit does not converge within 2
fixed-point iterations"). Integrity: both checkpoint SHAs byte-identical pre/post
(`models/pitchgpt_v2.pt` `6f952054…62883c`; `models/pitchgpt_v2_outcomehead_a1.pt`
`37b50e87…bb25e5`); DuckDB `read_only=True` throughout; **no artifact shipped to `models/`** —
the non-converged W is quarantined at
`results/pitchgpt/rollout_calibration_fit_2023/W_FAILED_FIT_quarantine.npz` (sha256
`395e6fcd…31b27d12`) with a full provenance sidecar.

### Consequences executed (2026-08-10, documentation and surface work only)

- **Flagship claim permanently narrowed** for v2-era PitchGPT to "per-pitch calibrated rollout
  engine" — and that per-pitch ECE claim is itself scoped, with its production-path number now
  **STRANDED**: the measurement rode holdout contact #13, which the kill voided; measuring it
  needs a new dated amendment plus one of the 2 remaining budgeted 2025 contacts, **not
  authorized**. `[claim:pitchgpt_per_pitch_ece]`.
- **PA-level absolute-rate products dropped from Tier-A scope.** The PA-level marginal-rate
  FAIL is now the standing, permanent position `[claim:pitchgpt_pa_rates_fail]`;
  `src/dashboard/views/matchup_sim.py` withholds absolute simulated wOBA levels and the
  absolute in-play-hit rate behind a scope banner.
- **The 0.6.1 wOBA / PA-length PASSes are permanently unearned** for v2-era PitchGPT — the
  clean re-evaluation that would have re-earned them never ran and never will under that
  protocol. `[claim:pitchgpt_woba_pa_pass_pre062 — RETRACTED]`. TAINTED-pending-0.6.2 markers
  were retired repo-wide, each now citing the kill.
- **Rank/differential products retained** with the marginal-bias disclosure
  (`src/dashboard/views/pitch_call_grades.py`).
- **Holdout budget preserved:** contact #13 never spent (2025 tier stands at 12 of 14 used,
  `docs/holdout_ledger.jsonl` note entry 2026-08-10); the §6 attribution diagnostic (#14) was
  not triggered; the **2026 lockbox is untouched and sealed**.
- **Provenance-guard clause green** (K5 sentence 3): no calibration vector was fit on any
  gate-evaluation cohort in this phase; `PGConcatHeadPredictor` structurally refuses W
  artifacts declaring 2025/2026, enforced by tests.

### Green-lit successor (user-adjudicated, same date)

The **WS5.2 v2 retrain is green-lit with its spec pre-registered BEFORE any training**:
`docs/pitchgpt_sim_engine/PITCHGPT_V2_SPEC.md` (chain-rule factorized heads + rollout-aware
fine-tuning + the WS5.3 failable gate suite), frozen on commit, §0–§8 immutable with an
append-only deviations log. **No training run may start before that freeze commit exists**, and
training happens in a later workflow — none was authorized or performed in this batch. Per K5
the retrain gets **exactly one** lockbox contact against sealed 2026 under that spec; failure
locks the per-pitch-only claim for the season.

Verdict also recorded at `docs/models/pitchgpt_phase062_results.md` §7.

---

## K6 — Global claims discipline

### Criterion, verbatim (plan §8)

> **K6 (global):** no claim ships to dashboard/docs without a claims-registry entry; no
> projection-flavored claim without beating Marcel per 4.4.

### Nature of the criterion

K6 is a **standing rule**, not a one-time verdict: it has no measured quantity and cannot
"fire" or "not fire" on a result. It is adjudicated continuously, by enforcement. As of
2026-08-10 that enforcement is mechanical rather than editorial.

### Enforcement in place (clause 1 — registry-gated claims)

- **Registry:** `docs/claims/claims.yaml` — 37 entries (20 active, 6 narrowed, 3 superseded,
  8 retracted). Every entry carries a **mandatory caveat**; schema validity is unit-tested.
- **API:** `src/claims.py::get_claim(id)` **raises** `ClaimRetracted` on retracted entries — a
  retracted number is structurally unable to render.
- **Drift guard:** `tests/test_claims_drift_guard.py` scans `src/dashboard/views/` for numeric
  literals adjacent to metric keywords and fails CI unless the line renders through
  `get_claim` or carries a `# claim:<id>` annotation resolving to a real id. Companion tests
  ban the retracted strings outright (`13.80`, `70–80%`, "perfect mathematical calibration",
  "68.4% validated") and require the bounded retracted numbers (`68.4`, `0.387`, `0.768`) to
  appear only on claim-annotated lines.
- **Strategy doc gated too:** every `[claim:<id>]` citation in `docs/NORTH_STAR_CURRENT.md`
  must resolve against the registry (`test_north_star_current_claim_citations_resolve`).

### Enforcement in place (clause 2 — the Marcel floor)

- **Reference implementation:** `src/analytics/marcel.py` (literal Marcel: 5/4/3 weights,
  +1200 PA league average, reliability wPA/(wPA+1200), age ±0.006/0.003 around 29, PA
  projection 0.5·y1 + 0.1·y2 + 200), unit-tested in `tests/test_marcel.py`.
- **Protocol:** WS4.4 — PA-weighted RMSE vs naive, head-to-head W-L with a .010-wOBA tie band,
  PA-weighted paired t-test at ≥90% confidence, **≥2 seasons before any superiority claim**.
- **First binding application:** AdjustedWAR v3 measured paired-t confidence **0.567** against
  the 0.90 bar → `superiority_claim_allowed = false`
  (`[claim:adjusted_war_v3_vs_marcel_forward]`), and the board channel loses to the
  Marcel-picker by −8.11pp (ridge) / −8.55pp (legacy)
  (`[claim:adjusted_war_v3_marcel_lift_17w]`). The mandatory board-surface framing sentence
  `[claim:adjusted_war_boards_k6_framing]` is the K6 consequence in rendered form. See K3
  above.
- **Forward-looking:** the frozen 2026 resolution spec §5.6 pre-registers M1 (Marcel as picker)
  and M2 (forecast-quality protocol) on the same pools, so K4's resolution carries its Marcel
  control by construction.

### Known limit of the second clause, recorded

The Marcel control currently exists on the **batter channel only** — pitcher Marcel is deferred
to the marcelR pin (frozen resolution spec §6.7). Until that lands, "does not beat Marcel" is a
batter-channel statement and no full-board Marcel comparison exists. This limit is carried in
the caveats of `[claim:adjusted_war_v3_marcel_lift_17w]` and
`[claim:adjusted_war_boards_k6_framing]`.

---

## Provenance of this document

- Criterion texts: `docs/plans/2026-08-10_platform_improvement_plan.md` §8, quoted verbatim.
- Measured quantities: `docs/models/defensive_pressing_v2_2026-08.md` (K1, K2),
  `docs/models/adjusted_war_v3_2026-08.md` (K3), `docs/models/contrarian_2026_resolution_spec.md`
  (K4), `docs/models/pitchgpt_phase062_results.md` (K5), `docs/claims/claims.yaml` +
  `tests/test_claims_drift_guard.py` + `src/analytics/marcel.py` (K6).
- Adjudication: user, 2026-08-10. K1/K2 also at `defensive_pressing_v2_2026-08.md` §7; K3 at
  `adjusted_war_v3_2026-08.md` §8; K5 at `pitchgpt_phase062_results.md` §7.
- No number in this file was computed here; each is quoted from the artifact or document cited
  beside it.
