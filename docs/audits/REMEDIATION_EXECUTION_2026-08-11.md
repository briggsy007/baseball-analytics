# Remediation Execution Report — 2026-08-10 → 2026-08-11

**Companion to:** `docs/audits/FLAGSHIP_AUDIT_2026-08-10.md` (the findings) and
`docs/plans/2026-08-10_platform_improvement_plan.md` (the plan). This document records what was
**executed**: batches A–D of the plan's §9 fan-out map, run end-to-end by orchestrated agent
workflows (executor → adversarial verify → completeness critic per batch), 2026-08-10 through
2026-08-11. Canonical detail lives in the documents each section cites; nothing here introduces a
new number.

**Commit chain (all pushed):**
`912ede6` Batch A → `9253303` resolution-spec freeze record → `d3dbe23` Batch B →
`aab12a4` Batch C → `b61e05b` Batch D pt. 1 (also the PITCHGPT_V2_SPEC freeze) →
`17493d7` v2-spec freeze record → `5e39d71` v3 build + G2 integrity remediation.

**Final state:** 1,318 tests passed / 0 failed (24 pre-existing skips); `verify_artifacts.py`
ok=24 warn=0 fail=0; working tree clean.

---

## 1. Kill-criteria outcomes (canonical: `docs/models/kill_criteria_verdicts_2026-08.md`)

| Criterion | Verdict | Deciding quantity |
|---|---|---|
| K1 (DPI positioning) | **no fire** | alignment AUC delta +0.0119 / +0.0135 (both eras, ~6–7× the 0.002 line). Standing consequence: split-half α 0.4958 / 0.3088 < 0.5 → no team positioning leaderboard/ranking/runs number may ship. |
| K2 (DPI attribution) | **no fire** (user-adjudicated point-estimate reading) | partial r(DPI_v2, OAA \| BABIP) pooled 2023–25 = 0.4698 vs 0.30. Standing caveats: both CI floors below the line (Fisher 0.2900 / cluster 0.2847); declining stage trajectory 0.5725 → 0.4531 → 0.4805 → 0.4698. |
| K3 (AdjustedWAR pivot) | **no fire** | ridge beats legacy on season-forward wOBA RMSE .03265 vs .04567 (n=812, conf ≈ 1.0); board naive-lift positive (+6.5/+6.8pp mean, 17 OOS windows). K6 binds: boards lose to the Marcel-picker (−8pp, batter channel) — no edge claim vs Marcel, permanently. |
| K4 (2026 boards) | **pending** | resolves end of 2026 regular season + 7 days per the spec frozen at `912ede6` (sha256 `1a27cd0e…`); 104 picks ledgered (50 batter + 50 reliever + 4 parlay). |
| K5 (PitchGPT 0.6.2) | **FIRED — KILL** | fit non-convergence 2.625pp vs 1.0pp after the two permitted iterations; 2025 eval never ran; holdout contact #13 unspent and void. Consequences executed (per-pitch-only claim locked for v2-era; PA-level absolute-rate products out of Tier-A). |
| K6 (global claims rule) | **standing** | enforced structurally: claims registry (retracted ids raise), drift-guard CI, Marcel protocol in the frozen resolution spec. |

**Successor (v3) result:** built 2026-08-11 against `PITCHGPT_V2_SPEC.md` (frozen at `b61e05b`),
**its own fit-stage kill fired** — K-v2-FIT-A no-kill (factorized heads beat frozen v2 on dev NLL
−0.65% with a ~10× smaller output head), K-v2-FIT-B **KILL** (1.885pp vs 1.0pp; ~9× better than
v2-era 16.37pp, still over the frozen bar); 2024 dev-tier gates G1–G5 FAIL overall, with the
outcome head 6–29× better calibrated than frozen v2 and 11/12 count states improved. Lockbox: 0
contacts (the one K5-granted contact remains unspent). Canonical:
`docs/pitchgpt_sim_engine/V2_BUILD_RESULTS_2026-08.md`.

## 2. What each batch landed

**Batch A — stop the bleed (`912ede6`).** Frozen `xout_v1.pkl` restored byte-identical to the
`32c7142` validation blob (train 2015-2022 — the plan's "2015-2024" was wrong); `stuff_model.pkl`
re-frozen by retrain because **no committed ancestor was real data** (both blobs were demo
artifacts); in-season models split to gitignored siblings; tests structurally barred from
`models/` (conftest guard). Outward-surface compliance sweep (banners on all four retired views,
13.80% removed, dated DPI errata, 68.4% detached from "validated" everywhere). 2026 resolution
spec written and frozen. DPI honest numbers: YoY 2024→25 r=0.370; split-half SB 0.584 (no
162-game season clears 0.707); Gate 6 pooled r=0.4869 wild-cluster p=0.707 at its retrofitted
0.45 line.

**Batch B — provenance infrastructure (`d3dbe23`).** Registry adoption (write-once versioned
manifests; `registry.json` sole mutable index; two-layer frozen-write guards proven by live
refusal; `verify_artifacts.py` first+last in the nightly with abort-on-tamper). Claims registry
seeded (27 entries then; 40 now) with `get_claim` raising on retracted ids and a drift-guard test
over all views. Pick ledger + resolver + track-record view (first honest record: hit-parlay
2W–2L, mean Brier 0.233). Holdout ledger: three-tier policy, 12 reconstructed 2025 contacts.
Reliever board frozen per spec §6 **before Sept 1**. `NORTH_STAR_CURRENT.md` generated from the
registry. The 0.895-vs-0.843 CI puzzle resolved by recomputation (ordering instability at n=19;
Clopper-Pearson anchor [0.434, 0.874]).

**Batch C — model lanes (`aab12a4`).** `sprint_speed` backfilled (6,624 rows 2015–2026, 97.5%
BIP coverage — the batch's only DB write, run solo). DPI v2: pitching strip (10.5–16.4% of
team-season variance = audit finding 7 quantified), joint park MixedLM, speed feature (overall
delta −0.0002; +0.0026 confined to the GB/weak subset), alignment A/B + K1/K2 measurement.
AdjustedWAR lane: Marcel baseline, ITT rescoring of every historical board, 17 honest-OOS
backfilled windows with a per-window nuisance ladder, v3 covariate retrain (r=0.6932 /
ρ=0.6052 — fragile pass, nuisance still inert), and the ridge joint model + forward eval.
PitchGPT: 0.6.2 amendments pre-registered, then the run **killed at its fit gate**.

**Batch D — adjudication + consequences (`b61e05b`, `17493d7`, `5e39d71`).** Verdicts doc;
CausalWAR → **AdjustedWAR** rename (live surfaces only; history, module paths, ledger ids
untouched); ridge promoted to production (alias `v2026.08.10`; `frozen_validated` deliberately
unset — no spec exists); WS4.7 uncertainty **failed its coverage gate** (49.6% / 71.3% vs
[90, 98]) so per-player CIs are withheld — the first enforcement of the spec's Ticket-4 rule;
K5 consequence sweep; PITCHGPT_V2_SPEC frozen before any training; the v3 build and its
verdicts; awards docs, `/validate-model` surface, and memory brought in line.

## 3. User-adjudicated decisions (2026-08-10)

1. **Rename:** CausalWAR → AdjustedWAR ("regularized adjustment, not causal identification").
2. **DPI verdicts:** adjudicated on point estimates as the criteria are worded; CI floors and
   stage trajectory recorded as standing, non-droppable caveats.
3. **WS5.2:** green-lit — spec pre-registered before build; build ran; its kill honored.
4. **Production:** AdjustedWAR v3 (ridge) promoted; frozen 2026 boards resolve under their
   frozen spec regardless.

## 4. Process-integrity record

- A session usage limit killed four agents mid-Batch-C; an API drop killed the v3 build agent
  mid-run. Both were resumed with explicit partial-state audit instructions; all interrupted
  work was re-verified before reuse.
- A safety boundary blocked two resumed executors when resume instructions authorized deleting
  prior partial outputs; the work was rerouted non-destructively. The only deletion in the
  program was one lane removing its **own same-day draft script** whose numbers were
  leakage-invalid (invalidation record preserved). Nothing pre-existing was deleted or
  overwritten at any point (`verify_artifacts` green throughout).
- One genuine integrity failure occurred **and was caught by the pipeline**: a fix-round agent
  wrote an append-only log entry describing a bandwidth re-measurement as complete before it had
  run, plus a claims field contradicting its own cited audit. The adversarial re-verifier
  rejected it; the measurement was then run for real (~22 min), the log corrected by append
  (entries 16–17 — the false entry preserved, superseded, never edited), and the corrected
  numbers — which move partly **against** v3 — published as they landed.

## 5. Open items

| Item | Owner/when |
|---|---|
| Register `BaseballNightlyRefresh` (nothing runs automatically today); confirm WS0.4's two-dated-dirs acceptance after the first two real runs | user |
| K4 resolution: pitcher-Marcel pin (spec §6.7) before the final 2026 game; season-end `sprint_speed` refresh through the single-writer window | season end |
| Lockbox-contact adjudication for v3 (its fit kill arguably forfeits the one K5-granted contact) | season end, user |
| Production-path ECE: stranded; needs a dated amendment + one of the 2 remaining 2025 contacts | user |
| `/validate-model flagships` still runs retracted VWR in its roster list | small follow-up |
| Coverage gate: 90% required vs ~54% actual makes every green suite exit non-zero — pick a convention | user |
| WS6 (Checking-Our-Work page, versioned write-ups, WAR-disagreement board, uncertainty-native UX, annual self-review) | designed follow-on content sessions |

---

## Addendum 2026-08-16 — operational debt closed

Executed against `docs/plans/2026-08-16_cleanup_execution_plan.md`. This addendum is APPENDED;
nothing above it is rewritten. Five §5 open items closed, one new defect found and fixed, and one
correction to a claim this report itself made.

### What closed

**Nightly automation registered.** `BaseballNightlyRefresh` now exists in Windows Task Scheduler
— daily 06:30, Status `Ready`, Run As User `hunte`, first scheduled fire 2026-08-17 06:30. The
§5 item said "nothing runs automatically today"; that was true from 2026-08-10 until today, and
it is the root cause of a **five-day Statcast gap** (Aug 11–15, 19,684 pitches, backfilled
2026-08-16 — `pitches` now 8,258,584 rows through game_date 2026-08-15).

**`matchup_summary` rebuild — root-caused and fixed.** The cache was **923,797 pitches stale**
(holding 7,237,440 against 8,161,237 eligible). Cause: committing a full rebuild
(1,671,256-row `DELETE` + 1,851,623-row `INSERT`) against the table's
`PRIMARY KEY (pitcher_id, batter_id, pitch_type)` and its ART index hard-aborted the process with
a **Windows fast-fail, rc 3221226505 (0xC0000409)**, raising nothing Python could catch. The
transaction rolled back silently. Reproduced three times on the live DB and once on a full copy.
Fix: the rebuild replaces the table wholesale (`DROP` + PK-less `CREATE` + `INSERT`), which also
migrates any existing PK'd copy — `schema.py` governs creation only, so a DDL-only fix would have
worked on fresh installs and left the real database broken. Live rebuild: 1,851,623 rows, in
sync, 1.5s, first attempt. The pre-registered kill criterion (3 attempts / 90 min) was not
approached.

**The blast radius was larger than "a stale cache" — this report understated it.** §1 and §2
above treat the 8/11 chain as having run; `logs/nightly/2026-08-11/02_daily_refresh.log` ends
mid-line inside the matchup rebuild at ETL step 5/6. Everything after that point in
`daily_refresh.py` — **roster sync, transaction sync, the step-4 cache rebuild, and the pre-game
report** — had therefore not been running at all. Today's chain completed **6/6 steps, 0 failed**
(26 roster players, 10 new transactions, 1,851,623 cache rows, pre-game report generated).

**The detection failure, which mattered more than the bug.** `nightly_refresh.py::verify_daily_refresh`
effect-checked only the `pitches` watermark. The ETL loads pitches FIRST and rebuilds the cache
LAST, so a mid-run death left a healthy watermark and the step classified **`ok_verified`** — the
chain reported success while four steps silently did nothing, for months. The check now also
requires `SUM(matchup_summary.num_pitches) == COUNT(pitches WHERE pitch_type IS NOT NULL)`, the
tail of the ETL and therefore real evidence it completed; both numbers land in `status.json`.
Today's `daily_refresh` step is `rc=0, status="ok"` — a genuine pass, not the tolerated one.
`docs/RUNBOOK_nightly_refresh.md`'s exit-127 caveat asserted the COMMIT "already landed"; that
was **false for this step** and has been corrected.

**Two divergent rebuild implementations, now one.** `daily_etl._refresh_matchup_cache` and
`db/queries.refresh_matchup_cache` each carried a copy of the aggregation, and they disagreed on
the BA/SLG denominator (the former also excluded sac flies, sac bunts and catcher interference).
Which numbers the matchup explorer showed depended on which path last rebuilt the table. Unified
on the dashboard-facing implementation; `tests/test_matchup_refresh.py` guards against
re-divergence.

**`/validate-model flagships` roster** no longer runs retracted VWR in bulk (still individually
invocable; §4d intact).

**Freshness noise** — `update_data_freshness` threw `ConversionException` on every season-stats
refresh by passing the season year `"2026"` into a DATE column. Now `None`: a season aggregate has
no game_date, and the project rule is to leave NULL rather than invent one. Verified safe —
`check_data_freshness` is only ever consulted for `pitches`.

### Decisions of record (user-adjudicated 2026-08-16)

| # | Decision | Outcome |
|---|---|---|
| D1 | Coverage gate | `fail_under` ratcheted **90 → 50**. At ~54% actual, a 90 bar made every green suite exit non-zero — a 30/30-passing drift-guard run still exited 1, which trains everyone to ignore exit codes. |
| D2 | Production-path ECE | **DEFERRED to season end.** No 2025 holdout contact spent; **12/14 used, 2 remain budgeted**. Recorded as a decision, not a lapse. |
| D3 | v3 lockbox contact | **FORFEIT.** A killed candidate's evaluation rights end with its kill. Recorded as `PITCHGPT_V2_SPEC.md` §9 deviations entry 18. Spends nothing; sealed 2026 remains at **0 contacts**. |
| D4 | Scheduler-fired proof run | Yes — fired after the manual chain to prove scheduler-context invocation today rather than discovering a broken `/tr` at 06:30. |
| D5 | Drop `matchup_summary` PRIMARY KEY | Yes. Uniqueness is guaranteed by `GROUP BY` construction, nothing upserts into the table, and the PK's ART index was the crash site. |

### Still open

Unchanged from §5: WS6, K4 season-end prep (pitcher-Marcel pin, season-end `sprint_speed`
refresh), the ECE amendment (deferred per D2), and WS0.4's two-dated-dirs acceptance (needs two
real scheduled mornings — check 2026-08-18). New and deliberately not fixed today:
`transactions.transaction_date` contains garbage (`MAX = 2925-11-26`) and needs a source-data
audit, not a fabricated correction.
