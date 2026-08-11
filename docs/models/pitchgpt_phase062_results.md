# PitchGPT Phase 0.6.2 — Results & Verdict (2026-08-10)

**Verdict: KILL. Phase 0.6 closes as FAIL under the pre-registered kill criterion**
(`docs/pitchgpt_sim_engine/PHASE_0.6.2_PLAN.md` §6, first disjunct: *"The 2023 fit does
not converge within 2 fixed-point iterations"*).

- **Protocol:** `PHASE_0.6.2_PLAN.md` §§1–8 (frozen 2026-08-04) + §10 amendments A1–A8
  (recorded 2026-08-10, pre-execution). The §6 kill criterion was never amended.
- **Executed:** 2026-08-10, `scripts/pitchgpt_fit_rollout_calibration.py` (exit code 2 =
  kill-signal branch). Full log-audit trail:
  `results/pitchgpt/rollout_calibration_fit_2023/{fit_audit.json, report.md}`.
- **Plan-level frame:** platform improvement plan §8 **K5** — "0.6.2 verdict stands as
  pre-registered (+5.0 amendments)." This document reports the measured quantities exactly
  as they landed. Adjudication *consequences* (claim-registry updates, product-scope
  changes) are executed by the orchestrator in Batch D, not here.

---

## 1. The measured kill quantities

Fit cohort (§4): **2023** pitcher-disjoint (2,247 pitchers from the 2015–2022 train split
excluded), 19,625 eligible PAs, 10,000 sampled with seed 42, 100 rollout samples/PA,
horizon 6, T=1.0, league-median ump scalar 0.360814. **2025 was never read.** All numbers
below are 2023 fit-cohort measurements (in-sample for the fit by design); no OOS number
was produced by this phase.

Convergence rule (§4 step 5 + §10.A7): every per-position class marginal within **1.0pp**
of empirical, adjudicated on a measurement re-roll with the candidate W applied; maximum
2 W-updates, no third roll.

| Iteration | Roll measured | max abs delta vs empirical | Converged (<= 1.0pp)? |
|---|---|---:|---|
| — (pre-fit reference) | roll-0, raw T-softmax | 16.37pp (pos 5, called_strike) | n/a |
| 1 (W1 = guarded ratio from roll-0) | roll-1 with W1 | **4.418pp** | NO |
| 2 (W2 = W1 · guarded update from roll-1) | roll-2 with W2 | **2.625pp** | **NO → KILL** |

Per-position max abs delta at the final (iteration-2) measurement, in pp:
pos 0 = 1.64, pos 1 = 2.05, pos 2 = 2.62, pos 3 = 2.36, pos 4 = 1.42, pos 5 = 0.75 —
the worst cell at every position 0–4 is the **ball** marginal (rollout under-produces
balls even after reweighting); largest residuals: pos-2 ball (emp 0.3765 vs 0.3503,
2.62pp), pos-3 ball (2.36pp), pos-1 ball (2.05pp), pos-2 called_strike (1.76pp).

Guard audit: 0 cells hit the <500-observation guard; 0 ratios were floor/cap clipped in
either update (all movement was inside [0.2, 5.0]).

Uncertainty note: the convergence rule is a pre-registered deterministic threshold on the
measured marginals (10K PAs × 100 samples ≈ up to 1M sampled outcomes at pos 0, ~204K at
pos 5); no CI is defined by the spec for the kill quantity itself. The 2.625pp terminal
delta is ~2.6× the threshold — not a borderline read.

## 2. Integrity checks (all clean)

- `models/pitchgpt_v2.pt` SHA256 byte-identical pre/post
  (`6f952054d14ac6f918f3eb9502b496b70bc0c87dfc65dc50d98ee7244a62883c` = the registry-pinned
  frozen_validated blob).
- `models/pitchgpt_v2_outcomehead_a1.pt` SHA256 byte-identical pre/post
  (`37b50e87599013c281560c9f63286fe5b7645297d0042694d907287417bb25e5`).
- DuckDB opened `read_only=True` throughout; no tables/views created.
- **No artifact shipped to `models/`** — the §6 kill branch quarantined the non-converged
  W to `results/pitchgpt/rollout_calibration_fit_2023/W_FAILED_FIT_quarantine.npz`
  (sha256 `395e6fcd16b188f58a9fc124c5ac33fded15fb8946e137a95310c5e931b27d12`, with full
  provenance sidecar: fit_cohort_season=2023, n_iterations=2, converged=False).
- Regression suite: `tests/test_pitchgpt_sim.py` + `tests/test_holdout_ledger.py`
  → 72 passed, 1 skipped (the skip is
  `test_shipped_rollout_perpos_artifact_provenance_if_present`, correctly skipping because
  no `calibration_rollout_perpos.npz` exists). `scripts/verify_artifacts.py` → ok=19,
  warn=0, fail=0.

## 3. What did NOT run, and the clause that forbids it

| Step | Status | Governing text |
|---|---|---|
| §5 single 2025 full-cohort evaluation (64,460 PAs, A1 amendment) | **NEVER RAN** | §6: "Stop **permanently** and close Phase 0.6 as FAIL if ANY of: The 2023 fit does not converge within 2 fixed-point iterations, …". The §5 run is defined "with W enabled"; no converged W exists. |
| Holdout-ledger contact #13 | **NOT SPENT** — 2025 budget stands at 12/14 used | §10.A3 authorized #13 only for the §5 evaluation run. A ledger `note` entry (2026-08-10) documents the void; the header `todo` was conditional on the run occurring. |
| §10.A2 production-path per-pitch ECE (stacks a–d) | **UNMEASURED** | §10.A3: the ECE measurement "rides the SAME contact (one run, one contact) … No other 2025 contact is authorized." Standalone measurement would need a new dated amendment + ledger authorization — a Batch D / orchestrator decision. Stack (d) is moot regardless (no W). |
| 2023 fit-regime sanity harness run (§5 second bullet, §10.A8 output dir `rollout_sanity_2023_phase062/`) | **NOT RUN** | It existed to show fit-cohort vs holdout transfer for a converged W in the results doc; there is no W and no holdout number to transfer to. The fit-regime behavior is fully documented in `fit_audit.json` (roll-0/1/2 marginals vs empirical). |
| §6 attribution diagnostic (contact #14) | **NOT TRIGGERED** | Only the third kill disjunct (wOBA/PA-length regression attribution) triggers it; the kill was disjunct 1. |
| §10.A6 registry registration of `models/calibration_rollout_perpos.npz` | **N/A** | A6 registers the npz "once produced"; it was never produced. No registry writes were made by this phase. |
| Third fixed-point iteration / refit variants | **FORBIDDEN** | §4 step 5, §10.A7 ("No third update, no third re-roll"), §8 (no added calibration dimensions). |

## 4. Pre-registered consequences on kill (§6, verbatim — execution is Batch D)

> "On kill: the flagship claim stays permanently narrowed to 'per-pitch calibrated rollout
> engine' (ECE-based); PA-level absolute-rate products (A3 matchup K%/BB% displays) are
> dropped from Tier-A scope; rank/differential products (A1 grades, A2 projection
> *distribution shapes*) proceed with the marginal-bias disclosure. No third calibration
> layer, no backbone/head retraining, no capacity increase — those are Plan-A-shaped moves
> that were already retired."

Additional consequences that follow mechanically from the kill landing at the *fit* stage:

- The 0.6.1 wOBA + PA-length PASSes remain **TAINTED — now permanently unresolved for
  v2-era PitchGPT**: they were to be re-earned under the clean fit in the §5 run, which is
  dead; no further 2025 contact is authorized under this protocol.
- The shipped-probability (production-path) ECE remains unmeasured (audit §3 finding 3
  stays open). The locked per-pitch claim (post-T ECE 0.0114) is unchanged by this phase —
  the §3 mode-scoping kept teacher-forced paths untouched by construction, and both
  checkpoint SHAs are byte-identical.
- Per plan §5.1: "On KILL: per-pitch-only claim locks permanently for v2-era PitchGPT;
  PA-level absolute-rate products stay dead until 5.2 passes its own gates." The
  designed successor is the WS5.2 v2 retrain (chain-rule factorized heads +
  rollout-aware fine-tuning), which per K5 gets ONE lockbox contact against sealed 2026
  under its own pre-registered spec. Whether to start 5.2 is a Batch D decision.

## 5. Diagnosis (descriptive only — not a rescue argument)

The fixed-point map was contracting (16.37 → 4.42 → 2.63pp, ratio ≈ 0.6/iteration) but
far too slowly to pass a 1.0pp threshold in the 2 permitted iterations. The persistent
residual is a **ball-marginal deficit at positions 0–4** that per-position outcome
reweighting cannot close: applying W changes sampled outcomes → count trajectories →
the backbone's pitch-token distribution at later positions, which partially undoes the
marginal correction (the §4 feedback loop). This is the exposure-bias signature the
research track pre-identified — "post-hoc multiclass calibration … is documented to
overfit calibration sets; exposure bias is a train/inference mismatch no output
reweighting removes. If 0.6.2 kills, the path is a retrain" (plan WS5 header). The
2-iteration cap firing on a slowly-contracting sequence is the protocol working as
designed: more iterations = the unbounded knob-tuning §8 exists to prevent.

## 6. K5 adjudication data (for Batch D)

- **Kill quantity:** 2023 fit convergence after 2 fixed-point iterations — max
  per-position class-marginal |delta| = **2.625pp** (iteration 1: 4.418pp) vs
  pre-registered threshold **1.0pp**. Dataset: 2023 pitcher-disjoint fit cohort, 10K PAs
  seed 42 (in-sample for the fit; no OOS measurement exists). → §6 disjunct 1 = TRUE →
  **Phase 0.6 = FAIL, verdict stands as pre-registered.**
- Provenance-guard status (K5 second clause): enforced and green — no calibration vector
  was fit on any gate-evaluation cohort in this phase (fit cohort 2023; 2025/2026
  untouched; `PGConcatHeadPredictor` structurally refuses W artifacts declaring 2025/2026;
  tests green).
- Holdout ledger: `pitchgpt_2025_pitcher_disjoint` = 12/14 contacts used, #13 void
  (documented in the ledger note, 2026-08-10), #14 not triggered. 2026 lockbox untouched
  and sealed.

---

## 7. Consequence execution record (appended 2026-08-10, Batch D — §§1–6 unmodified)

The user adjudicated K5 as FIRED on 2026-08-10; the §4/§6 consequences were executed the same
day (documentation and surface work only — no model run, no GPU, no DB write). Where they landed:

| Consequence | Landed in |
|---|---|
| Kill registered as a quotable claim (iteration 1 = 4.418pp, iteration 2 = 2.625pp vs 1.0pp, 2023 fit cohort, contact #13 unspent) | `docs/claims/claims.yaml` → new entry `pitchgpt_phase062_kill` (dataset `results/pitchgpt/rollout_calibration_fit_2023/fit_audit.json`, sha256 `d8df38a1...411b0a`) |
| Production-path ECE recorded as UNMEASURED **and stranded** | caveat of `pitchgpt_per_pitch_ece`; measuring it needs a new dated amendment + one of the 2 remaining budgeted 2025 contacts — NOT authorized in this batch |
| PA-level FAIL made the permanent position; wOBA/PA-length PASSes made permanently unearned | caveats of `pitchgpt_pa_rates_fail` and `pitchgpt_woba_pa_pass_pre062` |
| PA-level absolute-rate products dropped | `src/dashboard/views/matchup_sim.py` (A3) withholds every simulated wOBA quantity (level, p05/p25/p50/p75/p95 bands, histogram, mean, in-play-hit share, K%/BB%/HR%); publishes only the pair's ordinal position in the loaded cohort behind a scope banner; median-centring evaluated and rejected |
| Rank/differential products retained with the marginal-bias disclosure | `src/dashboard/views/pitch_call_grades.py` (A1) |
| TAINTED-pending-0.6.2 markers retired repo-wide (each now cites the kill; originals preserved with dated pointers) | `COORDINATION.md`, `PHASE_0.6_DIAGNOSIS.md` §9.3, `docs/NORTH_STAR.md`, `docs/NORTH_STAR_CURRENT.md`, `docs/awards/methodology_paper_pitchgpt.md`, `docs/awards/methodology_paper_pitchgpt_v2.md`, `docs/models/pitchgpt_validation_spec.md` |
| Successor pre-registered (no training authorized before its freeze commit) | `docs/pitchgpt_sim_engine/PITCHGPT_V2_SPEC.md` |

`PHASE_0.6.2_PLAN.md` §§1–8 were deliberately NOT edited (frozen verbatim; §11 already carries the
execution record).
