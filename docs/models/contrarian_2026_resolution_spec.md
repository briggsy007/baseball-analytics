# 2026 Contrarian Board Resolution Specification (Pre-Registration)

**Status:** FROZEN ON COMMIT. The git commit that introduces this file is the tamper-evident freeze
(plan WS1.1: "commit = tamper-evident freeze"). After that commit, the body of this document
(§0–§7) is immutable. Any change of any kind — corrections, clarifications, protocol amendments —
happens exclusively as dated, append-only entries in the Deviations Log (§8).

**Authored:** 2026-08-10, per `docs/plans/2026-08-10_platform_improvement_plan.md` §1.1 (Task A4).
**Remediates:** `docs/audits/FLAGSHIP_AUDIT_2026-08-10.md` §2.4 (missing 2-yr-aggregate
matched-naive control), §2.5 (survivorship-trimmed denominator), §2.6 (post-hoc hit criterion),
§5 F-D (post-hoc criteria) and F-E (silent scope substitution).
**Governs:** kill criterion **K4** — *"resolution strictly per the frozen 1.1 spec. If ITT hit rate
≤ matched-naive, the miss is published on the track-record page with the same prominence a win
would get."* This document is that spec. No threshold in it may be softened after results exist.

**Structure** follows the OSF pre-registration template: Foreknowledge (§1), Hypotheses (§2),
Sampling & Stopping (§3), Variables (§4), Analysis Plan (§5), plus the reliever-board
pre-registration (§6), the resolution procedure (§7), and the Deviations Log (§8).

**Executability standard:** a person with the DuckDB database, the artifacts named in §4, and this
document must be able to score every pick with **no judgment calls**. Every place where a judgment
call was unavoidable at authoring time, the tiebreak is pre-registered inline and marked
**[TIEBREAK]**.

---

## 1. Foreknowledge declaration

Stated plainly, per OSF norms:

1. **The 2026 mid-season boards were generated WITHOUT a frozen resolution criterion.** The board
   generator (`scripts/contrarian_2026_midseason.py`, commit `0928baf`, 2026-08-04, data snapshot
   through 2026-08-03) froze the *picks* but not the *scoring rule*. At generation time the only
   hit rule in existence was post-hoc dashboard product code
   (`src/dashboard/views/contrarian_leaderboards.py:280-325`), retro-formalized in
   `scripts/causal_war_contrarian_stability.py::row_hit_verdict`. The audit (§2.6) correctly
   identifies this as a post-hoc criterion. This spec is the fix: it freezes the criterion
   **before resolution but AFTER board publication**.
2. **The author has partial outcome knowledge.** This spec is written 2026-08-10 — seven days of
   post-snapshot games have been played and are publicly visible, and the author has general
   knowledge of 2026 standings and player performance to date. Mitigation, not cure: every formula
   below is anchored to machinery that predates the 2026 board (the WAR-delta rule, the surrogate
   floors, the ±0.3 matched-naive window, the 1000-resample seed-42 bootstrap all exist verbatim
   in `causal_war_contrarian_stability.py`, committed before the 2026 board was built), and every
   new constant is derived mechanically from frozen conventions with its derivation shown (§5.1).
   No constant in this spec was chosen by looking at how any 2026 pick is performing.
3. **The boards' own banner already says application-not-validated.** Quoting the shipped banner:
   the validated 68.4% figure "was measured on FULL-SEASON 2023-24 picks resolved against 2025
   outcomes and does NOT transfer to these mid-season boards. 2026 picks here are UNRESOLVED."
   That labeling stays until resolution. Nothing in this spec upgrades the 2026 boards to
   "validated" — see §2 interpretation bounds.
4. **The 2-yr-aggregate retrospective control (§5.5.3) is computed with outcomes fully known.**
   2025 outcomes resolved long ago; the author knows the marquee numbers (13/19 = 68.4%
   per-protocol; 13/25 = 52.0% ITT). That computation is a *disclosure obligation* (audit §2.4:
   the one config that never received its matched-naive control), not a prospective test, and it
   is labeled as such wherever published. It cannot rescue, upgrade, or re-validate the 68.4%
   claim; the plan's non-goals (§7: "No rescue attempts on the 68.4%...; the narrowed claims are
   the position") apply in full.
5. **Holdout/contact accounting:** resolving the 2026 boards consumes end-of-season 2026 bWAR — a
   public outcome series, not a modeling holdout. It does not touch the sealed 2026 pitch-level
   lockbox (plan 2.3/5.4). The resolution run will be registered in the pick ledger
   (`predictions/resolutions.jsonl`, WS1.2) when that infrastructure lands.

---

## 2. Hypotheses

Primary and secondary hypotheses, with pre-committed interpretation bounds. All hit rates are
**intention-to-treat (ITT)** as defined in §5.3, computed on the control basis defined in §5.5.1.

- **H1 (primary; K4 adjudicates on this):** the Buy-Low board's ITT hit rate exceeds its
  matched-naive control's ITT hit rate (same pool, same window, same scoring rules; §5.5).
- **H2:** the Over-Valued board's ITT hit rate exceeds its matched-naive control's ITT hit rate.
- **H3:** each board side's ITT hit rate exceeds the Marcel-picker control's ITT hit rate for the
  same side (§5.6, M1).
- **H4 (reliever board, §6):** the reliever Buy-Low board's ITT hit rate exceeds its
  **within-filter** matched-naive ITT hit rate (neighbors drawn only from the qualified reliever
  pool — the audit's 78.1%-vs-56.9% within-filter convention).
- **Non-hypothesis disclosure computation:** the 2-yr-aggregate 2023-24 → 2025 matched-naive
  control (§5.5.3). Outcomes already known; reported for completeness, never as a test.

**Pre-committed context (written before any 2026 outcome exists):**

- Historical matched-naive Buy-Low rates in this platform's own artifacts run **66.5–73.0%**
  (`results/causal_war/contrarian_stability/hit_rates_by_year.json`: 73.0% / 66.5% / 71.0% across
  the 2022→23, 2023→24, 2024→25 windows). Naive mean-reversion is a high bar; that is the point.
- Only two fully-OOS historical windows exist, averaging roughly +4pp with a sign flip in the
  middle window (−2.8pp). A 25-pick side cannot statistically distinguish a 10pp edge from a 30pp
  edge. This resolution is **one more window**, feeding K3's mean-lift average — it is not, by
  itself, a validation.
- **A pass does NOT confer "validated."** Per plan 4.4 and K6, no projection-flavored claim ships
  without beating Marcel at ≥90% confidence over ≥2 resolved seasons. One resolved window cannot
  meet that. The dashboard banner language may state the resolved 2026 result only alongside its
  matched-naive and Marcel controls and its void counts.
- **A miss publishes at full prominence** (K4, verbatim above). No post-hoc subgroup rescue of a
  failed board is permitted (K3: "the −2.8pp autopsy pattern is banned by this clause").

---

## 3. Sampling & stopping

1. **The picks are fixed.** Exactly the 50 rows of the frozen board artifact
   `results/edges/contrarian_2026_midseason/board.csv` (25 `buy_low` + 25 `over_valued`, all
   `position = "batter"`), committed in `0928baf` (2026-08-04), SHA-256
   `EF89356EFABBD2AD81C8C5DE331ED4727CAED496016CEE60D78A7D8AF31442F1`. No pick may be added,
   removed, or substituted. The reliever board (§6) adds its own frozen rows before 2026-09-01
   under this same spec; nothing else joins.
2. **No survivorship trimming.** Every named pick appears in the published resolution table with
   exactly one outcome from {HIT, MISS, VOID-*}. Denominators are stated with void counts. The
   audit's 6-of-25-excluded pattern (§2.5) is structurally impossible under §5.3.
3. **One resolution event.** Scoring happens once, at the resolution date defined in §7 (end of
   2026 regular season + 7 days). No interim hit rates are published as resolution numbers; the
   live dashboard may continue to display descriptive current-season data, but no number derived
   from this spec's scoring rules ships before the resolution date.
4. **No optional stopping.** The resolution date is fixed by formula (§7), not by looking at
   results. There is no early-stop branch and no extension branch except the enumerated retry/VOID
   ladder (§5.4, §7).
5. **No new configurations.** Scoring rules, pools, windows, and controls are exhaustively
   enumerated here. Running any additional configuration against these picks and reporting it
   alongside the pre-registered numbers is prohibited; sensitivity analyses are limited to the
   ones pre-listed (§5.5.4) and are explicitly non-adjudicating.

---

## 4. Variables and data sources

### 4.1 Artifact register (frozen inputs, SHA-256)

| # | Artifact | Role | SHA-256 |
|---|----------|------|---------|
| A1 | `results/edges/contrarian_2026_midseason/board.csv` (git `0928baf`) | The 50 picks; board-basis baselines | `EF89356EFABBD2AD81C8C5DE331ED4727CAED496016CEE60D78A7D8AF31442F1` |
| A2 | `results/edges/contrarian_2026_midseason/summary.md` | Board provenance (pool 278/405, gates PA≥204/IP≥34, snapshot 2026-08-03) | `10C960061C803E5D605E181AA1EFCF5255707AE47605B5DEC23E38A398BEB6F0` |
| A3 | `results/edges/contrarian_2026_midseason/frozen_2026_resolution_inputs/war_2026_staging_asof_2026-08-09.parquet` | Control-basis snapshot: full-universe 2026 season-to-date bWAR, fetched 2026-08-09 23:04 local. 1,401 rows = 613 batters + 788 pitchers (pre-crosswalk; 5 pitcher rows unmatched → A5, leaving 1,396 matched: 613 + 783 per A4) | `0DF755CF2C3AC30BEE4C4EFD2FF1CA9ABEF47C755E4F36DD9986D8B839249EDD` |
| A4 | `.../frozen_2026_resolution_inputs/war_2026_staging_audit_asof_2026-08-09.json` | Fetch audit for A3 (n=1401, match rate 0.9964) | `9A744C38D29A08A08782CEB200359E419A8CF54F608CEDBF93DD791FD508D818` |
| A5 | `.../frozen_2026_resolution_inputs/war_2026_unmatched_asof_2026-08-09.csv` | The 5 unmatched rows from the A3 fetch | `57A42C770414859B878F5C1A2E1306B789B7A3743545E2B53D14FCB5B0FA1FF9` |
| A6 | `.../frozen_2026_resolution_inputs/causal_war_2026_leaderboard_cache_asof_extraction_2026-08-10.parquet` | Causal-side pool snapshot (411 rows; `leaderboard_cache` blob, `computed_at` 2026-08-09 23:02:51, extracted read-only 2026-08-10) | `92D6658034F67D20B2274F5CFAC788B8B2ABA1BEAE9A85F025108978919306AD` |
| A7 | `results/validate_causal_war_20260418T194415Z/causal_war_baseline_comparison_2023_2024.csv` | 2-yr-aggregate config artifact (the 68.4% marquee's source; v1 pin per audit §2.7) | `FFAEC4DBCEAB781B36F04777FE49EC376CB51A653B1B5CEA4423F7636FA4CB25` |
| A8 | `data/fangraphs_war_staging.parquet` | Historical bWAR 2015–2025 (misnamed "fangraphs"; content is B-Ref bWAR). Universe for the 2-yr-aggregate control | `DB652CA11966EDC93D1BA54F3E2BDC1E721AB490FD3892FB5EAEE414D325B977` |
| A9 | `results/causal_war/contrarian_stability/hit_rates_reproduction.json` | Row-level reproduction of the 2-yr config's per-pick verdicts (13/19 BL, 14/23 OV) | (in-repo; row-level content quoted in §5.5.3) |

**Why A3/A6 exist (declared candidly):** `data/staging/war_2026_staging.parquet` and the
`leaderboard_cache` DB row are **mutable** — both were overwritten by the nightly chain on
2026-08-09, six days after the board's 2026-08-03 snapshot, and the Aug-3 full-universe values are
unrecoverable (B-Ref's `war_daily` files are current-state only; `data/staging/` is gitignored).
A3/A6 are dated, hash-pinned copies of the closest surviving state, archived 2026-08-10 so that
the control pool (§5.5) has a frozen basis. The 6-day offset between the board snapshot
(2026-08-03) and the control basis (2026-08-09) is a declared limitation of the control, handled
by computing the control **entirely** on the A3 basis (§5.5.1) rather than mixing bases.

### 4.2 Resolution data source (authoritative)

- **Pipeline:** `scripts/backfill_2026_war.py` — pybaseball `bwar_bat` / `bwar_pitch` (Baseball-
  Reference `war_daily` files), stint-aggregated by `scripts/backfill_fwar.py::fetch_war_for_years`,
  MLBAM-id crosswalked via `match_to_db_players` (in this DB, `players.player_id` IS the MLBAM id).
- **Primary resolution artifact [TIEBREAK]:** the dated staging parquet written by the resolution
  run (§7 step 2), columns `player_id, player_name, season, position_type, war, pa_or_ip,
  war_source`. If the DuckDB merge target and this parquet ever disagree (partial merge, later
  overwrite), **the dated resolution parquet governs**.
- **DB merge target (convenience/secondary):** `season_batting_stats.war`, `.pa`, `.ops` and
  `season_pitching_stats.war`, `.ip`, `.era` at `season = 2026` (schema:
  `src/db/schema.py`). The OPS/ERA surrogate branch (§5.2 step 5) reads from these DB columns
  because the parquet does not carry OPS/ERA.
- **bWAR is the authoritative WAR flavor.** No substitution of fWAR or any proxy under any branch
  (the pipeline's own kill criterion already enforces "NULL stays NULL").

### 4.3 Variable definitions (per pick *i* on the batter board)

| Variable | Definition | Source column |
|---|---|---|
| `side_i` | `buy_low` or `over_valued` | A1 `board` |
| `wb_i` | Baseline bWAR at board snapshot (season-to-date through 2026-08-03) | A1 `trad_war`, full float precision as stored — **[TIEBREAK]** no rounding before any comparison |
| `pab_i` | Baseline PA at board snapshot | A1 `pa_total` |
| `wf_i` | Final full-season 2026 bWAR | resolution parquet `war` (batter row) |
| `paf_i` | Final full-season 2026 PA | resolution parquet `pa_or_ip` (batter row) |
| `ops_f_i` | Final full-season 2026 OPS (surrogate branch only) | `season_batting_stats.ops`, season 2026 |
| `RoS_PA_i` | Rest-of-season plate appearances | `paf_i − pab_i` |
| `wb'_i`, `pab'_i` | Control-basis baseline bWAR / PA (through 2026-08-09) | A3 `war`, `pa_or_ip` (batter row, matched by `player_id`) |

**[TIEBREAK] Two-way players:** all 50 picks carry `position = "batter"`; the batter row of every
source governs. (For §6 reliever picks, the pitcher row governs.) If a player has two batter rows
in a source (should be impossible post stint-aggregation), the row with larger `pa_or_ip` governs.

### 4.4 Frozen constants (derivations shown; chosen 2026-08-10, before any outcome exists)

Season-progress is **measured**, not estimated: `p(d)` = median over the 30 teams of regular-season
games played through date `d`, divided by 162; games counted as
`COUNT(DISTINCT game_pk)` per team from the `pitches` table (`home_team`/`away_team`, `game_date`
between 2026-03-01 and `d`; the DB ingests regular-season games only — team spread 115–119 on
Aug 3 confirms no spring-training contamination).

| Constant | Value | Derivation |
|---|---|---|
| `p_snap` | **0.7222** | median 117 games through 2026-08-03 / 162 (measured 2026-08-10, read-only query above) |
| `p_ctrl` | **0.7531** | median 122 games through 2026-08-09 / 162 (same query) |
| Pace divisor, board basis | `p_snap` = 0.7222 | §4.5 discussion |
| Pace divisor, control basis | `p_ctrl` = 0.7531 | A3 fetched 2026-08-09 |
| Exit floor, board basis | `RoS_PA < 28` | round(100 × (1 − p_snap)) = round(27.78); 100 = the validated full-season follow-up surrogate floor (PA ≥ 100) scaled to the remaining season fraction |
| Exit floor, control basis | `RoS_PA < 25` | round(100 × (1 − p_ctrl)) = round(24.69) |
| Matched-naive window, control basis | **±0.2259** | 0.3 × p_ctrl — the historical ±0.3 window is defined per full season of baseline WAR span (§5.5.2 convention); baseline span here is 0.7531 seasons |
| Control pool PA gate | **PA ≥ 226**, n = **262** | round(300 × p_ctrl) = round(225.9); 300 = validated full-season inclusion gate; count frozen from A3 |
| Bootstrap | 1000 resamples, percentile, `numpy.random.RandomState(42)`, pick-level | verbatim from `causal_war_contrarian_stability.py::bootstrap_ci` |
| Surrogate thresholds | batter OPS ≥ 0.700, PA ≥ 100; pitcher ERA ≤ 4.00, IP ≥ 30 | verbatim from the pre-existing rule (`contrarian_leaderboards.py:280-325`) |

**[TIEBREAK] Snapshot-date ambiguity:** B-Ref daily files may lag one day; the snapshot dates are
fixed by declaration as 2026-08-03 (board) and 2026-08-09 (A3). Worst-case divisor error < 1%,
disclosed, not adjustable later.

### 4.5 Declared divergence: the generator's 0.68 vs measured 0.7222

`contrarian_2026_midseason.py` hardcoded `SEASON_PROGRESS = 0.68` ("~110 of 162 games through
Aug 3"). The DB says the median team had played **117** games through 2026-08-03 → 0.7222. The
0.68 estimate was simply wrong at generation time.

- The **PA gate stays 204** — it defined the actual frozen pool and is not revisable.
- The **pace divisor uses measured 0.7222**, because the hit rule extrapolates the actual season
  fraction elapsed.
- This choice is not a soften. Direction of effect of 0.7222 vs 0.68: for a Buy-Low pick with
  **positive** baseline WAR the measured divisor *lowers* the pace target (more lenient); with
  **negative** baseline WAR it *raises* it (stricter); Over-Valued is the mirror image. The board
  contains both signs on both sides (Buy-Low `wb` spans −0.54 to +1.66; Over-Valued +0.87 to
  +4.15). A sensitivity run at divisor 0.68 is published, non-adjudicating (§5.5.4).

---

## 5. Analysis plan

### 5.1 Hit rule — plain statement

A **Buy-Low** pick claims the player is better than the market number; it **hits** if the player's
full-season 2026 bWAR reaches the pace his snapshot bWAR implied, i.e. the player at minimum *kept
up the pace the market number already conceded* (and therefore the market number rose toward the
model's view or the player out-earned it). An **Over-Valued** pick claims the player's market
number overstates him; it **hits** if the player falls short of his snapshot pace. This is the
within-season analog of the historical rule (follow-up WAR ≥ baseline WAR), which compares
realized value against the baseline the market had already banked.

### 5.2 Pick-scoring algorithm (exact; evaluation order is normative)

For each of the 50 picks, evaluate **in this order** and stop at the first branch that fires:

1. **Global voids.** If V1 or V4 (§5.4) is in force → outcome `VOID-V1` / `VOID-V4` for all picks.
2. **Player-level data voids.** If the player is absent from the resolution parquet's batter rows
   AND absent from the pre-crosswalk staged frame by MLBAM `player_id` (§7 fallback) → `VOID-V2`.
   If present but both `war` and `pa_or_ip` are NULL → `VOID-V3`.
3. **Exit check.** With `RoS_PA_i = paf_i − pab_i`: if `RoS_PA_i < 28` →
   - `buy_low`: **MISS** (branch `exit`);
   - `over_valued`: **VOID-EXIT** (branch `exit`).
   Justification of the asymmetry: §5.3. **[TIEBREAK]** if `paf_i` is NULL while `wf_i` is
   non-NULL (should be impossible in B-Ref daily files), skip the exit check, score on step 4,
   and flag the row in `notes`.
4. **WAR branch (primary).** If `wf_i` is non-NULL, with pace target `T_i = wb_i / 0.7222`:
   - `buy_low`: **HIT** iff `wf_i >= T_i`, else MISS.
   - `over_valued`: **HIT** iff `wf_i < T_i`, else MISS.
   Operators are exact: `>=` on the bullish side (ties favor the pick, matching the historical
   `war_followup >= war_baseline` rule), strict `<` on the bearish side. Full float precision.
5. **Surrogate branch.** If `wf_i` is NULL but `paf_i` is present (note: a non-exit pick has
   `paf_i ≥ pab_i + 28 ≥ 232 > 100`, so the historical PA ≥ 100 surrogate floor is automatically
   satisfied): read `ops_f_i` from `season_batting_stats`;
   - `buy_low`: **HIT** iff `ops_f_i >= 0.700`, else MISS;
   - `over_valued`: **HIT** iff `ops_f_i < 0.700`, else MISS;
   - `ops_f_i` NULL → `VOID-V3`.
   Surrogate values are full-season (pre-snapshot-contaminated) — a declared coarseness inherited
   from the historical rule; expected to fire ~never (B-Ref publishes final WAR for everyone).

Every pick's row in the resolution table records which branch fired (`war` | `surrogate` | `exit` |
`void-*`).

### 5.3 ITT accounting and the exit asymmetry (direction-symmetric statement)

**Principle:** a pick with no resolvable performance record scores **against the pick's direction
where that is coherent** — and is voided where it is not:

- **Buy-Low exit = MISS.** A bullish pick asserts the player will deliver value. Non-delivery —
  injury, demotion, release, retirement, benching — *is the claimed outcome failing*. Counting it
  as a miss is truthful, not merely conservative.
- **Over-Valued exit = VOID (never HIT, never MISS).** Symmetric treatment would score a bearish
  pick's exit as a HIT ("he indeed didn't deliver"), but that awards the model credit for
  injuries and roster transactions it has no mechanism to predict — exactly the inflation the
  audit flagged (§2.5: exclusions asymmetric in the model's favor). Scoring it MISS would be
  scoring against the observed truth (the player genuinely delivered nothing). VOID is the only
  non-distorting branch. **The asymmetry is therefore intentional and conservative against the
  model:** bullish exits count against it; bearish exits never count for it.

**Denominators (all published together, always):**

| Quantity | Buy-Low | Over-Valued |
|---|---|---|
| ITT rate (headline; K4 adjudicates on the control-basis version) | hits / (25 − #VOID-data), exits in denominator as misses | hits / (25 − #VOID-all), exits excluded as voids |
| Worst-case rate | hits / 25 | hits / 25 |
| Per-protocol rate (continuity with historical artifacts) | hits / #(war- or surrogate-branch picks) | same |
| Void disclosure | count + named list of every VOID pick | same |

No pick is ever silently dropped: the resolution table contains all 50 named rows regardless of
branch (§3.2).

### 5.4 VOID branches (exhaustive) and the ambiguity clause

- **V1 (global):** B-Ref publishes no usable final 2026 bWAR by resolution date + 14 days
  (`fetch_war_for_years` retry ladder exhausted on every attempt) → the entire board resolves
  AMBIGUOUS → every pick `VOID-V1`, the table is **published anyway** with the void reason.
- **V2 (player):** pick absent from resolution data after the pre-crosswalk fallback (§7) →
  `VOID-V2`.
- **V3 (player):** present but WAR NULL and surrogate inputs NULL → `VOID-V3`.
- **V4 (global):** the 2026 regular season ends with median team games < 140 (strike, pandemic,
  or other shortening) → the pace-extrapolation rule is invalid → entire board `VOID-V4`,
  published anyway.
- **V-EXIT (Over-Valued only):** §5.2 step 3.
- **V-M (controls only):** the Marcel control boards are not committed before the final 2026
  regular-season game (§5.6) → Marcel comparisons resolve `VOID-M` and the failure to run the
  control is published as a **process miss**; model-pick scoring is unaffected.
- **Ambiguity clause (Metaculus norm, verbatim requirement of plan 1.1):** any condition this spec
  failed to anticipate that prevents mechanical scoring of a pick → that pick resolves
  **AMBIGUOUS → VOID, scores voided, published anyway**, with a dated Deviations Log entry
  describing the condition. Voids are never dropped from the table and every published rate states
  its void count. Resolution-time discretion is limited to *classifying* a pick into the branches
  above; it may never *re-score* one.
- **Explicit non-void [TIEBREAK]:** if Baseball-Reference revises its WAR methodology between
  snapshot and resolution (component updates happen routinely), scores are computed from bWAR as
  published at the resolution fetch and **stand**; the methodology change is noted in the
  Deviations Log. Frozen snapshot baselines (`board.csv trad_war`) are never restated.

### 5.5 Matched-naive controls

**Purpose:** the audit's core finding (§2.4) is that the marquee hit rate ≈ the naive
mean-reversion base rate. Every headline number this spec produces therefore ships **with** a
matched-naive control computed for the **same window/config** — including the 2-yr-aggregate
config that has never had one.

#### 5.5.1 2026 mid-season config (prospective; adjudicates K4)

All quantities on the **control basis** = artifact A3 (baselines through 2026-08-09) + the
resolution parquet (outcomes). Single-basis computation avoids mixing the 2026-08-03 board
baselines with 2026-08-09 neighbor baselines inside one matching rule.

- **Universe `U`:** batter rows of A3 with `pa_or_ip >= 226` → **n = 262** (frozen §4.4). (All 50
  picks appear in A3's batter rows — verified 2026-08-10; and all 262 gated batters appear in the
  A6 causal leaderboard snapshot, so `U` is also the model's menu.)
- **Neighbors of pick *i*:** `N_i = { j ∈ U : j ≠ i, |war_j − war_i| <= 0.2259 }` using A3 `war`
  for both, same position type (all batters here). If `|N_i| < 3`, pick *i* contributes nothing to
  the naive rate (verbatim historical convention).
- **Neighbor scoring:** each `j ∈ N_i` is scored with the **full §5.2 algorithm** under pick *i*'s
  direction, on the control basis: pace target `war_j / 0.7531`, exit floor `RoS_PA_j < 25`
  (`RoS_PA_j` = resolution `pa_or_ip` − A3 `pa_or_ip`), same ITT accounting (Buy-Low direction:
  neighbor exit = miss; Over-Valued direction: neighbor exit = excluded void).
- **Naive rate** (per side) = mean over contributing picks of mean neighbor score — the historical
  average-of-per-pick-neighbor-rates convention.
- **Model rate, control basis** (per side) = the 25 picks scored with the same §5.2 algorithm but
  with A3 baselines (`T'_i = war'_i / 0.7531`, exit floor 25). This is the apples-to-apples
  number.
- **K4 adjudication [TIEBREAK]:** per side, compare **model ITT rate (control basis)** vs
  **matched-naive ITT rate (control basis)**. The board-basis model rate (§5.2, the headline
  pick-scoring) is published alongside; if board-basis and control-basis model rates disagree in
  sign relative to the naive rate, both are published and the **control basis adjudicates**,
  because it is the internally consistent comparison. This tiebreak is fixed now, before outcomes.

#### 5.5.2 Window convention (declared, applies everywhere)

The historical ±0.3 matching window was defined on single-full-season baseline WAR. This spec
generalizes it as **±0.3 per season of baseline-WAR span**: ±0.3 × 0.7531 = ±0.2259 for the
partial-2026 basis; ±0.3 × 2 = ±0.6 for the 2-yr aggregate; ±0.3 × p_f for the reliever board.
Rationale: baseline WAR dispersion scales with the span it accumulates over; a fixed ±0.3 would be
effectively looser matching on partial seasons and effectively stricter on aggregates. The raw
±0.3 variant is published as a sensitivity everywhere (§5.5.4), non-adjudicating.

#### 5.5.3 2-yr-aggregate config (retrospective disclosure; audit §2.4 closure)

Foreknowledge: outcomes fully known (§1.4). Protocol is mechanical so no result-shopping is
possible:

- **Model side:** the frozen top-25-per-side of artifact A7, per-pick verdicts as already
  reproduced row-level in A9: Buy-Low 13 hits / 19 evaluated / 6 no-record
  (Sborz, W. Smith, Voth, Almonte, Bradford, V. González); Over-Valued 14 / 23 / 2 no-record
  (J. Gray, Cobb). ITT restatement under §5.3: **Buy-Low ITT = 13/25 = 52.0%** (6 exits = misses);
  **Over-Valued ITT = 14/23 = 60.9%** with 2 voids disclosed (worst-case 14/25 = 56.0%).
- **Universe `U2`:** from A8 — per `player_id`: `war_agg` = Σ `war` over rows with
  `season ∈ {2023, 2024}` (≥1 season present); `vol_agg` = Σ `pa_or_ip` over the same rows;
  position from `position_type`. **[TIEBREAK]** if a player has both batter and pitcher rows, the
  type with larger `vol_agg` governs; exact tie → batter. Qualification: batter `vol_agg >= 200`,
  pitcher `vol_agg >= 40` (2 × the single-season 100/20 stability-pool floors).
- **Follow-up:** `war_2025` from A8 `season = 2025` rows. Neighbor with no 2025 row = exit
  (Buy-Low-direction miss / Over-Valued-direction excluded void, inherited per §5.3).
- **Matching:** same position type, `|war_agg_j − war_agg_i| <= 0.6`, `j ≠ i`, ≥3 neighbors else
  the pick contributes nothing.
- **Neighbor hit:** Buy-Low direction `war_2025_j >= war_agg_j / 2`; Over-Valued `< `. (The `/2`
  per-year normalization is the frozen dashboard rule.)
- **Output:** model ITT vs naive ITT for both sides, published wherever the 68.4% is bounded
  (claims registry per WS2.2, results docs, dashboard evidence tab). Labeled
  **RETROSPECTIVE — outcomes were known when this control was computed.**
- **Scope note:** A7 is the v1-pinned artifact (audit §2.7). If WS0.3/WS2 re-pins the evidence
  surface to the v2 comparison artifact, this identical protocol runs against that artifact too
  and both results are published; neither replaces the other silently.

#### 5.5.4 Pre-registered sensitivities (published, never adjudicating)

1. Matching window raw ±0.3 (all configs).
2. Control-pool gate at PA ≥ 204 (n = 278) and PA ≥ 213 (n = 272) for the 2026 config.
3. Pace divisor 0.68 (the generator's estimate) for the 2026 config.
No other variant may be computed or reported.

### 5.6 Marcel baseline (scored on the same pools)

Marcel is the mandatory floor for anything projection-flavored (plan 4.3/4.4; Tango: "ALL
forecasting systems should be treated as if they are nothing more than Marcel, at best").
Implementation lands in a later batch (plan 4.3); **this section freezes the scoring protocol
now** so the implementation has no degrees of freedom that touch these boards.

**Parameters (plan 4.3, restated as formulas):** for 2026 batter projections from seasons y1=2025,
y2=2024, y3=2023 with weights (5, 4, 3):

- `wPA = 5·PA_y1 + 4·PA_y2 + 3·PA_y3`; reliability `rel = wPA / (wPA + 1200)`.
- Weighted player rate `r = Σ(w_y · PA_y · woba_y) / Σ(w_y · PA_y)` (from
  `season_batting_stats.woba`, `.pa`).
- League rate `lg_y` = PA-weighted league mean wOBA per season, batters, from
  `season_batting_stats`; regression target = the (5,4,3)-PA-weighted blend of `lg_y1,lg_y2,lg_y3`.
- `proj_woba_raw = rel·r + (1−rel)·lg`.
- Age adjustment, age as of 2026-06-30: multiplier `1 + 0.006·(29 − age)` if `age < 29`, else
  `1 + 0.003·(29 − age)` (≤ 1 for age > 29).
- PA projection: `0.5·PA_y1 + 0.1·PA_y2 + 200`.
- **[TIEBREAK]** Where this sketch and the reference implementation
  (github.com/bdilday/marcelR) disagree in any detail, **marcelR governs**, and the disagreement
  is recorded as a dated Deviations Log entry at implementation time — before resolution.
- **Birth dates:** the `players` table has no birth-date column (verified against
  `src/db/schema.py`). Pre-registered source: MLB Stats API `/api/v1/people?personIds=…`
  `birthDate` (this DB's `player_id` IS the MLBAM id). The Marcel batch backfills it.

**M1 — Marcel as picker (control for H3):** on the same control pool `U` (n = 262):
`rank_marcel` = rank of Marcel-projected 2026 wOBA, descending, `method="min"`;
`rank_trad'` = rank of A3 `war`, descending, `method="min"`;
`mdiff = rank_trad' − rank_marcel`. Marcel-Buy-Low = top-25 by `mdiff` descending; Marcel-Over-
Valued = top-25 ascending (no sign requirement — mirroring the frozen board's construction
exactly). **[TIEBREAK]** boundary ties: lower `player_id` (deterministic, arbitrary, declared).
Both Marcel boards are scored with the **identical** control-basis §5.2 algorithm + §5.3 ITT,
including the identical matched-naive treatment. Deadline: the Marcel boards must be generated
from pre-2026 inputs and **committed before the final 2026 regular-season game**, else `VOID-M`
(§5.4). (Marcel's inputs are 2023–2025 only, so it is outcome-blind by construction even if run
later; the deadline is enforced anyway for process discipline.)

**M2 — forecast-quality protocol (plan 4.4, defined here, runs when Marcel lands):** over all
non-void members of `U` at resolution: PA-weighted RMSE of projected vs actual full-season 2026
wOBA; comparisons: (a) vs the naive constant forecast = 2025 PA-weighted league wOBA; (b)
head-to-head W-L vs Marcel per player on `|error|` with a **0.010-wOBA tie band**; (c) PA-weighted
paired t-test at **≥90% confidence**; (d) **≥2 resolved seasons before any superiority claim** —
2026 alone cannot ground one (K6: no projection-flavored claim without beating Marcel per 4.4).
CausalWAR participates in M2 only if/when WS4 gives it an explicit forecast head; until then its
board performance is measured through M1.

### 5.7 Statistical reporting and publication requirements

- **Bootstrap CIs:** 1000 percentile resamples, `RandomState(42)`, resampling pick-level ITT
  scores (voids excluded from the array; Buy-Low exits included as 0). Same machinery for naive
  and Marcel rates. CIs are descriptive; K4 compares point rates (pre-registered, small-n).
- **Resolution table (mandatory schema, one row per pick, all 50 + reliever rows):**
  `pick_id` (`contrarian2026-<board>-<side_rank>-<player_id>`), `board`, `side_rank`, `player_id`,
  `name`, `wb` (board basis), `pab`, `pace_target`, `wb_ctrl`, `wf`, `paf`, `ros_pa`, `branch`,
  `outcome` (HIT/MISS/VOID-*), `notes`.
- **Published files:** `results/edges/contrarian_2026_midseason/resolution/<date>/`
  (resolution table CSV + dated resolution parquet + audit JSON + hashes) and
  `docs/models/contrarian_2026_resolution_results.md` (numbers exactly as they land, per side:
  ITT / worst-case / per-protocol rates, void lists, matched-naive with lift in pp, Marcel M1
  comparison, sensitivities, K4 verdict line).
- **Ledger:** picks backfill-registered into `predictions/picks.jsonl` and outcomes appended to
  `predictions/resolutions.jsonl` (WS1.2), `rule_hash` = SHA-256 of this file at the freeze
  commit; the dashboard track-record view renders from the ledger only.
- **Claims registry (K6):** every number that ships to dashboard/docs gets a claims-registry entry
  (WS2.2) carrying the mandatory caveat string (single window; control-adjusted; void counts).
- **K4, verbatim, again, so it cannot be missed at publication time:** if ITT hit rate ≤
  matched-naive, the miss is published on the track-record page with the same prominence a win
  would get.

---

## 6. Reliever board (plan 1.3) — criterion pre-registered here

Plan 1.3 will freeze a pitcher-side reliever board before 2026-09-01 (the only base-rate-cleared
historical cohort — reliever leverage tag, 78.1% vs 56.9% **within-filter** naive, n=32 — is
structurally absent from the batter-only board). Its criterion is fixed **now**, before that board
exists:

1. **Generation (constraints on the 1.3 task):** pitcher-side CausalWAR effects for 2026 computed
   with the frozen 2015–2022 nuisance checkpoint
   (`models/causal_war/causal_war_trainsplit_2015_2022.pkl`), pitcher aggregation `pa_min = 50`
   (the stability-script convention), merged against a **fresh** 2026 season-to-date bWAR staging
   fetch. At freeze, the task must archive with SHA-256s, in
   `results/edges/contrarian_2026_midseason/frozen_2026_resolution_inputs/`: the dated board CSV,
   the dated staging parquet from the same fetch, and the freeze-date progress constant `p_f`.
   These are appended to §8 as a dated entry — the append-only log is the registration channel.
2. **Progress constant:** `p_f` = median team games through the freeze date / 162 (4 dp), from the
   §4.4 query, recorded at freeze.
3. **Reliever band (pro-rated from the full-season tag definition `ip_total < 60` and the
   stability pool floor `IP ≥ 20`):** qualify iff
   `IP_snapshot >= round(20 · p_f)` AND `IP_snapshot < 60 · p_f` (upper bound compared at full
   precision, no rounding — **[TIEBREAK]**).
4. **Board construction:** ranks computed **within the qualified reliever pool** (`rank_causal`,
   `rank_trad` descending, `method="min"`; `rank_diff = rank_trad − rank_causal`). Buy-Low
   requires `rank_diff > 0`; Over-Valued requires `rank_diff < 0` (sign requirement declared —
   unlike the batter board — because the reliever pool may be small). Sides take up to 25 by
   `rank_diff` descending / ascending. **[TIEBREAK]** boundary ties: larger
   `|causal_war − trad_war|`; then lower `player_id`.
5. **Scoring:** the §5.2 algorithm with pitcher variables: `wb` = snapshot pitcher bWAR, `wf` /
   `ipf` from the resolution parquet's pitcher rows, pace target `T = wb / p_f`, exit floor
   `RoS_IP < 30 · (1 − p_f)` (full precision; 30 = the validated full-season pitcher surrogate
   floor), surrogate branch (WAR NULL): eligibility `ipf >= 30`, Buy-Low HIT iff
   `era_f <= 4.00`, Over-Valued HIT iff `era_f > 4.00` (full-season values, declared coarseness).
   **[TIEBREAK]** WAR NULL and surrogate-ineligible (`ipf < 30` or `era_f` NULL) → treated as an
   exit (Buy-Low MISS / Over-Valued VOID-EXIT). ITT and the exit asymmetry exactly per §5.3.
6. **Matched-naive (within-filter, per the audit's own convention):** neighbors drawn **only**
   from the qualified reliever pool, window ±`0.3 · p_f` (§5.5.2 convention), ≥3 neighbors, same
   single-basis rule (the freeze-fetch parquet is both the pick baseline and the neighbor
   baseline — no basis offset, because §6.1 archives the parquet at freeze).
7. **Marcel-pitcher control:** marcelR's pitcher variant, ranked by projected RA9 (ascending =
   better); parameters pinned by a dated §8 entry at implementation time, before season end
   (**[TIEBREAK]**: marcelR governs all pitcher-Marcel details); board construction and scoring
   identical to M1. Missing the deadline → `VOID-M` for the reliever Marcel comparison.
8. **Hypothesis H4** (§2) adjudicates reliever Buy-Low ITT vs within-filter matched-naive ITT.
   K4's publication rule applies identically.

---

## 7. Resolution procedure, dates, and finality

1. **Resolution date `R`** = the date of the last 2026 MLB regular-season game actually played
   (per the official MLB.com schedule; any tiebreaker game counts as regular season, matching
   B-Ref's convention) **+ 7 days**. The currently scheduled end is late September 2026; `R` is
   defined by the formula, not a calendar guess.
2. **On or after `R`:** stop the dashboard (single-writer rule); run
   `python scripts/backfill_2026_war.py` (full run — fetch, stage, crosswalk, merge). Immediately
   archive the run's staging parquet, audit JSON, and unmatched CSV as dated copies under
   `results/edges/contrarian_2026_midseason/resolution/<date>/` with SHA-256s. That dated parquet
   is the authoritative resolution artifact (§4.2 precedence).
3. **Retry ladder:** if the source is unreachable or returns no usable 2026 rows (the script's
   own `KillCriterion`), retry daily through `R + 14`. Exhausted → **V1** (§5.4).
4. **Crosswalk fallback [TIEBREAK]:** a pick absent from the *matched* output is looked up by
   MLBAM `player_id` in the *pre-crosswalk staged frame*; if present there, that row's `war` /
   `pa_or_ip` govern scoring (the crosswalk failed, not the source); if absent entirely → V2.
5. **Scoring:** apply §5 / §6 mechanically. No human re-classification beyond the enumerated
   branches; anything else → ambiguity clause.
6. **Finality:** scores computed from the `R`-window fetch are final. Later B-Ref WAR revisions do
   not reopen resolved picks. Frozen baselines are never restated.
7. **Publication:** per §5.7 — results doc, resolution table with all named picks, ledger appends,
   claims-registry entries, dashboard track-record rendering, K4 prominence rule. Deadline for
   publication: `R + 14`.

---

## 8. Deviations Log (append-only)

**Instructions (normative):** This log is the ONLY part of this document that may change after the
freeze commit. Additions are **append-only** — never edit or delete an existing entry, never edit
§0–§7. Every entry must carry: an ISO date, the author/agent, what changed or was observed, why,
and whether it touches scoring (entries that would alter the scoring of any already-resolved pick
are prohibited; entries that classify an unanticipated condition under the §5.4 ambiguity clause
are the expected use). Required future entries include: the freeze-commit SHA of this file
(entry 1, added by the committing orchestrator); the reliever-board freeze record (§6.1); the
Marcel parameter pin (§5.6/§6.7); any B-Ref methodology-change observation (§5.4); any ambiguity-
clause invocation.

| # | Date | Author | Entry |
|---|------|--------|-------|
| — | — | — | *(empty at freeze)* |
| 1 | 2026-08-10 | orchestrator (Batch A) | Freeze commit of this spec: `912ede6a8a179284ffcc5c1e4039c9c59078c24c`. File sha256 at freeze: `1a27cd0e2d9b7d08c69c5a8a5944602585931121d9396be922e1e519557c760e` (= §5.7 rule_hash, referenced from the dashboard 2026-board banner). Verify against the frozen blob via `git show 912ede6:docs/models/contrarian_2026_resolution_spec.md` — the working-tree file legitimately diverges from the frozen hash from this entry onward. Does not touch scoring. |
| 2 | 2026-08-10 | executor agent B4a (plan WS1.3) | **Reliever-board freeze record (§6.1 registration).** Board generated by `scripts/contrarian_2026_reliever_board.py` and frozen 2026-08-10. Board CSV `results/edges/contrarian_2026_midseason/2026-08-10/reliever_board.csv` (byte-identical archive `frozen_2026_resolution_inputs/reliever_board_asof_2026-08-10.csv`), sha256 `B4C4F51702FD308581CB6D141C1CCA3175709D9C2F3606A36444B3A7D545C556`, **50 rows** (25 `reliever_buy_low` / 25 `reliever_over_valued`; qualified reliever pool n = 244; rank_diff > 0: 124, < 0: 117, = 0: 3 excluded by the §6.4 sign rule). Progress constant **p_f = 0.7531** (median 122 team games through 2026-08-09 / 162, §4.4 query); reliever band **IP ≥ 15 AND IP < 45.1860** (60 × p_f, full precision); pace divisor p_f; exit floor RoS_IP < 7.4070; naive window ±0.2259. Nuisance checkpoint `models/causal_war/causal_war_trainsplit_2015_2022.pkl` sha256 `5D13242CC3C3D7F0F8A33D52AAAA6BF337B9F1F576C0722484BE11AAC0B97672` (v1 12-confounder layout), pitcher aggregation pa_min = 50; DB pitch data through 2026-08-08. **Declared deviation from §6.1's "fresh … staging fetch":** the freeze ran under Batch-B constraints (DB read-only, no network fetches), so the baseline basis is artifact A3 (`war_2026_staging_asof_2026-08-09.parquet`, sha256 `0DF755CF2C3AC30BEE4C4EFD2FF1CA9ABEF47C755E4F36DD9986D8B839249EDD`), verified byte-identical to the live `data/staging/war_2026_staging.parquet` at freeze time; the as-of date is 2026-08-09 by the §4.4 declared-date convention, and p_f is measured through that same as-of date so the frozen parquet is both the pick baseline and the neighbor baseline with no basis offset (the §6.6 single-basis rule). All 50 picks appended to `predictions/picks.jsonl`: product `contrarian_board_2026_midseason_reliever`, pick_id scheme `contrarian2026-<board>-<side_rank>-<player_id>` with board ∈ {`reliever_buy_low`, `reliever_over_valued`} (the §5.7 scheme; board token disambiguates from the batter picks), evidence_class `prospective`, rule_hash = this spec's freeze sha256. Machine-readable record: `frozen_2026_resolution_inputs/reliever_board_freeze_record_2026-08-10.json`. **Does not touch scoring:** no pick is resolved and no scoring rule is altered; all §6 constants instantiate mechanically from p_f. |
| 3 | 2026-08-10 | executor agent C2a (plan WS4 carry-over) | **Regeneration guard added** to `scripts/contrarian_2026_midseason.py`: `generate()` now unconditionally refuses (exit 2) because the §3.1 pick basis is frozen and post-freeze regeneration would mint lookalike board artifacts that cannot join the resolution basis (audit failure class F-A); `--check` remains functional read-only. Does not touch scoring: no pick, rule, constant, or artifact is altered. |
