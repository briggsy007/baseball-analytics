# PitchGPT v3-factorized — Build + DEV-TIER Results (2026-08-11)

> ## ⚠ THESE ARE 2024 BURNED-DEV NUMBERS. THIS IS NOT THE LOCKBOX VALIDATION.
>
> 2024 is the **BURNED dev tier** (`PITCHGPT_V2_SPEC.md` §5.3, `docs/holdout_ledger.jsonl`):
> three committed scripts trained/tested on it, so it carries **no validation authority and
> no budget**. Everything below is labelled `tier=burned-dev` and **may not be quoted as a
> validation result**.
>
> **The one graded contact is against the sealed 2026 full season, at season end, per §5.5 —
> and it was NOT made.** `lockbox_2026_full_season.unsealed` is still `false`.
> **2025 was never read**: `pitchgpt_2025_pitcher_disjoint` still stands at **12/14** contacts
> used and this program spent none of them.
>
> **A pre-registered kill criterion fired at the fit stage (K-v2-FIT-B, §7.2).** While it
> stands, the §5.5 lockbox contact is not authorized.

**Protocol:** `docs/pitchgpt_sim_engine/PITCHGPT_V2_SPEC.md`, FROZEN at commit
`b61e05b3729aca2fa4609fbdadf4e533c1cd814f`, body sha256
`b19b54d96b496c5ffbff3f9af3070c180609003ec3d5d0aae71bfa84bb8d6d5b`
(verify: `git show b61e05b:docs/pitchgpt_sim_engine/PITCHGPT_V2_SPEC.md | sha256sum`).
§0–§8 immutable; every deviation is a dated §9 entry, all of which were written **before** the
run they govern except entries 11 and 12, which record measured outcomes.

**Executed:** 2026-08-11 (UTC), single RTX 3050 4GB, DuckDB `read_only=True` throughout.

---

## 1. Verdict summary

| Gate / criterion | Cohort | Measured | Threshold | Verdict |
|---|---|---:|---:|---|
| **K-v2-FIT-A** (§7.1) dev NLL vs frozen v2 | 2024 dev | **−0.650% relative** (better) | kill if > +2.0% worse | **NO KILL** |
| **K-v2-FIT-B** (§7.2) max per-position class-marginal \|Δ\| | 2023 fit | **1.8852pp** | ≤ 1.00pp | **KILL** |
| **K-v2-COMPUTE** (§7.3) cumulative GPU | — | **4.09 GPU-h**; largest run 2.16h | ≤ 12 GPU-h; ≤ 6h/run | **NO KILL** |
| **G1** classwise-ECE + TACE per head | 2024 dev | 2 of 8 sub-thresholds met | see §5 | **FAIL** |
| **G2** KCE hypothesis test | 2024 dev, **n = 2,000 of 149,949/head** (§9-13), pinned kernel (§9-16) | 3 of 4 heads reject calibration at p<0.01 | no head may reject | **FAIL** (probe informative, **not VOID**) |
| **G3** PIT + PA marginals | 2024 dev | PIT KS 0.14447; K%/BB%/HBP% out of tolerance | KS ≤ 0.03 + all marginals | **FAIL** |
| **G4** per-count-state ECE + position gate | 2024 dev | position gap 2.5408pp; 8/12 count states pass | ≤ 1.0pp; ≤ 0.02 | **FAIL** |
| **G5** decision calibration (K/BB/HR) | 2024 dev | K weighted gap 5.8107pp | ≤ 0.5pp weighted, ≤ 1.0pp/decile | **FAIL** |
| **§6.8 dev-tier verdict** | 2024 dev | — | G1∧G3∧G4∧G5 ∧ (G2 green-or-VOID) | **FAIL** |

**H1 (representation) — SUPPORTED.** The factorization does not cost predictive quality: it
*gains* 0.650% dev NLL at a **10.1× smaller output stack** (28,240 vs 285,090 parameters).

**H2 (exposure bias) — NOT SUPPORTED at the pre-registered threshold, but the mechanism moved
a long way.** The quantity Phase 0.6.2 died on went **16.37pp → 1.8852pp** (8.7×) and beat the
2.625pp that two rounds of the now-banned output reweighting reached — while remaining 1.9× the
1.0pp line. On the same fit cohort the monotone drift signature reads ρ(position, KL) = **0.6**
(p = 0.208, n = 6 positions) against the v2-era 0.822 — a weakening, but at n = 6 it is not
statistically distinguishable from either 0.822 or 0. Do not quote it as a result on its own; the
gated quantity is the 1.8852pp above. (The 2024 dev cohort's separate value is 0.2571, p = 0.6228;
see §G4. The two are different cohorts and are not interchangeable.)

**H3 (product) — BLOCKED.** PA-level absolute-rate products stay dropped. Nothing here re-earns
anything for the v2-era stack, and nothing here earns anything for v3 either: a dev-tier read is
not a validation.

### The weakest number, volunteered (§6.8, FanGraphs-Lab norm)

**G5 K-decile calibration: 11.1155pp max gap, 5.8107pp n-weighted.** Worse than the headline
level error, the *decile structure* shows the model's per-PA P(K) carries almost no
discriminative signal on this cohort: predicted P(K) spans 0.0991 → 0.2398 across deciles while
the empirical rate is nearly flat and non-monotone (0.2098 → 0.2338, range 2.4pp).

| Decile | n | mean predicted P(K) | empirical K% | gap |
|---:|---:|---:|---:|---:|
| 1 | 3,999 | 0.0991 | 0.2103 | +11.116pp |
| 2 | 3,999 | 0.1247 | 0.2141 | +8.935pp |
| 3 | 3,999 | 0.1376 | 0.2206 | +8.296pp |
| 4 | 3,999 | 0.1491 | 0.2246 | +7.550pp |
| 5 | 3,999 | 0.1596 | 0.2116 | +5.194pp |
| 6 | 3,999 | 0.1698 | 0.2323 | +6.252pp |
| 7 | 3,999 | 0.1812 | 0.2338 | +5.259pp |
| 8 | 3,999 | 0.1930 | 0.2311 | +3.802pp |
| 9 | 3,999 | 0.2080 | 0.2098 | +0.183pp |
| 10 | 3,998 | 0.2398 | 0.2246 | +1.522pp |

Part of the level offset is the horizon-6 truncation (8.74% of rollouts never terminate, so
their K is never realized) measured against uncapped SQL terminal events. The *flatness of the
empirical column* is not explained by truncation and is the substantive finding.

---

## 2. Architecture, as built (§3)

`src/analytics/pitchgpt_v3.py`. Backbone identical to frozen v2 (§3.2: d_model 128, 4 layers,
4 heads, max_seq 256, CONTEXT_DIM 35, causal mask, AdamW, lr 1e-3, batch 32, grad_clip 1.0,
seed 42, 10K-game budget). Output side replaced by the exact chain rule
`p(token) = p_type(t|h) · p_zone(z|h,t) · p_velo(v|h,t,z)` over the unchanged token identity
`token_id = type·130 + zone·5 + velo`.

| Head | Layers | C | Params (spec) | Params (built) |
|---|---|---:|---:|---:|
| `H_type` | `Linear(128→17)` | 17 | 2,193 | 2,193 |
| `H_zone` | `E_type(17×32)` + `Lin(160→64)`+GELU+`Lin(64→26)` | 26 | 12,538 | 12,538 |
| `H_velo` | `E_zone(26×32)` + `Lin(192→64)`+GELU+`Lin(64→5)` | 5 | 13,509 | 13,509 |
| **Total output stack** | | | **28,240** | **28,240** |
| Flat head replaced | `Linear(128→2210)` | 2,210 | 285,090 | — |

**10.1× drop**, inside the plan's "283K → ~40K" expectation and both §3.3 build assertions
(< 45,000 and < 0.25 × 285,090), which are asserted at construction and unit-tested.
Input embedding stays flat (§3.4 primary). **Ablations A-EMB and A-2023 were NOT run** — both are
optional and judged only on dev NLL, and the §7.2 kill made a dev-NLL-selected variant moot.

Per-field sampling masks (§3.6) are live in training-time rollout and at inference from the same
`support_counts` table (sha256 `9f4a8a138b6bf988…`), computed once from the training split.
Across the full 2023 fit-cohort roll: 901 zone fallbacks, 1,382 velo fallbacks, **0 degenerate
(uniform) draws**; type index 16 (`unknown`) was never emitted.

---

## 3. Training (§4)

### Stage A — teacher-forced pretrain (§4.2)

2015–2022, 9,999 games / 2,942,098 pitches (seeded reservoir draw, §9 entry 3), 5 epochs,
seed 42. **427.3s GPU** — matching the plan's 8.5-minute figure, not the training script's
documented ~4h25m, because the v3 data path is a cached vectorised loader whose
bit-equivalence with `PitchSequenceDataset` is asserted by test.

| Epoch | train NLL/pitch | 2023-slice NLL/pitch | 2023 ppl |
|---:|---:|---:|---:|
| 1 | 4.97174 | 4.83985 | 126.45 |
| 2 | 4.67865 | 4.77393 | 118.38 |
| 3 | 4.63936 | 4.73664 | 114.05 |
| 4 | 4.61684 | 4.70118 | 110.08 |
| **5 (best)** | 4.60103 | **4.69032** | **108.89** |

Frozen v2's best 2023 val loss was 4.7276 on the same slice definition.

### K-v2-FIT-A (§7.1) — the factorization must not cost predictive quality

Teacher-forced NLL per composite token, 2024 dev, 151,619 pitches, identical rows, v3's chain
rule multiplied back to a token probability.

| Stack | NLL/pitch | Perplexity |
|---|---:|---:|
| **v3 factorized (Stage A)** | **4.70643** | **110.657** |
| frozen v2 (`pitchgpt_v2.pt`) | 4.73721 | 114.115 |
| v1 @10K (informational) | 4.74006 | 114.441 |

v3 field decomposition: type 1.36673 + zone 2.90868 + velo 0.43102.
**Relative delta vs frozen v2: −0.650% (better). Kill threshold: +2.0% worse. NO KILL.**
`models/pitchgpt_v2.pt` sha256 byte-identical pre/post.

### Stage B — curriculum rollout-aware fine-tune (§4.3)

Full curriculum ran as pre-registered; **the §4.3.4 two-pass scheduled-sampling fallback was NOT
needed** (`two_pass_fallback_used: false`). lr 2e-4, batch 32, seed 42, horizon 6,
756,204 PAs/sub-stage. Total **7,765.3s** (2.16h) — under the §7.3 6h single-run cap.

| Sub-stage | depth | ε | substitution rate | train NLL | 2023 PA NLL | sec |
|---|---:|---|---:|---:|---:|---:|
| (pre-B baseline) | — | — | — | — | 5.42019 | — |
| B1 | 2 | 0.00→0.25 | 12.5% | 5.24148 | **5.39036** | 1,988 |
| B2 | 4 | 0.25→0.50 | 37.5% | 5.40791 | 5.43111 | 2,853 |
| B3 | full PA (6) | 0.50 | 50.0% | 5.49010 | 5.44927 | 2,923 |

PA-scoped teacher-forced NLL rises across B2/B3 — the expected trade of likelihood for
distributional fidelity under scheduled sampling. It is monitoring only, never a gate.

**§4.1 dynamic mid-PA context (commit `6111cd6`) verified on the training data path**, not
assumed: 99.9466% of checked positions match the `_advance_count` trajectory of the real outcome
sequence, 99.9944% of PAs have a varying `count_state`, `static_context_bug_present: false`. The
run aborts if no PA varies.

Stage B attempt 1 was killed by the executor harness at ~58 min with no artifact and no gate
number; the re-run is byte-identical in protocol and reproduced identical NLLs at identical batch
indices. See §9 entry 10a. **§4.5 budget: Stage A 1/2, Stage B 1/2 — the repair pass is unspent.**

### Outcome head (§3.5) — the one deliberate change earns its keep

`211→128→64→7`, A1 topology unchanged, **no inverse-frequency class weighting**, frozen v3
backbone, PA-scoped regime (§9 entries 4–5). Best epoch 2, 2023 log-loss 1.21391, 35,847 params.

`PHASE_0.6_DIAGNOSIS.md` §6.1 identified the A1 head's class weights as the class-marginal-bias
root cause. Removing them fixed it at the source, on the 2023 fit cohort:

| Class | v3 predicted share | empirical | Δ | A1 head (2025, from §6.1) |
|---|---:|---:|---:|---:|
| `ball` | 0.3689 | 0.3705 | **−0.16pp** | 24.5% vs 36.1% = **−11.6pp** |
| `called_strike` | 0.1515 | 0.1678 | −1.63pp | strike-class share +9.8pp |
| `swinging_strike` | 0.1136 | 0.1055 | +0.81pp | |
| `foul` | 0.1884 | 0.1842 | +0.42pp | |
| `in_play_out` | 0.1209 | 0.1126 | +0.83pp | |
| `in_play_hit` | 0.0547 | 0.0564 | −0.17pp | |
| `hbp` | 0.0020 | 0.0030 | −0.10pp | |

Max |Δ| = 1.63pp, versus 11.6pp for the head this replaces.

### Calibration (§4.4) — the only post-hoc step permitted

Four scalars, NLL-minimized on the **2023 pitcher-disjoint slice only** (19,653 PAs / 74,501
pitch rows). No vectors, no matrices, no per-position or per-count tables (§0.2).

| Head | T | NLL @ T=1 | NLL @ T |
|---|---:|---:|---:|
| `T_type` | 1.0906 | 1.76216 | 1.75921 |
| `T_zone` | 0.9920 | 2.96652 | 2.96651 |
| `T_velo` | 0.8720 | 0.72059 | 0.71586 |
| `T_outcome` | 0.9857 | 1.21391 | 1.21384 |

Sidecar `models/calibration_pitchgpt_v3.json` declares `fit_cohort_season: 2023`, `fit_seed: 42`,
`fit_n_pas: 19653`, `produced_by`. The loader **structurally refuses** any sidecar declaring 2025
or 2026 (§7.5); the check runs before the artifact is written, so a violating run aborts.
Every gate cohort (2024 dev; sealed 2026) is disjoint from the fit cohort.

---

## 4. K-v2-FIT-B (§7.2) — the kill that fired

Full **19,653-PA** 2023 pitcher-disjoint cohort, **no subsample** (the seed-42 10K-PA subsample is
audit finding F-C and was not reused), 100 samples/PA, horizon 6, per-PA seeds `42 + i·1000`.
Looked at **exactly once** (§4.3.3).

| Configuration | max per-position class-marginal \|Δ\| | Worst cell |
|---|---:|---|
| **v3, shipped per-head T (gated)** | **1.8852pp** | pos 1, `called_strike` |
| v3, raw T = 1.0 | 1.8405pp | pos 1, `called_strike` |
| *v2-era, raw T (0.6.2 roll-0)* | *16.37pp* | *pos 5, `called_strike`* |
| *v2-era, after 2 rounds of banned reweighting* | *2.625pp* | *pos 2, `ball`* |
| **Threshold** | **1.00pp** | |

Per-position maxima (shipped T, pp): 0.792 / **1.885** / 1.573 / 1.166 / 1.678 / 1.114.
Secondary (reported, not gated), **measured on this 2023 fit cohort**: ρ(position, per-position
KL) = **0.6**, p = 0.208, n = 6 positions, vs the v2-era reference 0.822. At n = 6 the rank
correlation is not significant and its confidence interval spans essentially the whole range, so
the honest reading is "no longer clearly monotone", not "3× weaker". The 2024 dev cohort gives a
different value on a different cohort (0.2571, p = 0.6228, §G4); the two must not be swapped.

**→ KILL.** No post-hoc reweighting layer was added to rescue it (§0.2 forbids it; that family is
what this program replaces). The §4.5 second curriculum run is **deliberately unspent**: choosing
its single change after seeing 1.8852pp is exactly the knob-tuning §4.5/§7.2 bound, so it is an
orchestrator/user decision requiring a dated §9 entry naming the change *before* it runs.

PA terminal shares on the fit cohort — the level the v2-era stack got badly wrong:

| Terminal | v3 model | empirical | Δ |
|---|---:|---:|---:|
| K | 0.1664 | 0.1749 | −0.85pp |
| BB | 0.0702 | 0.0703 | −0.01pp |
| HBP | 0.0068 | 0.0112 | −0.44pp |
| in_play_hit | 0.2084 | 0.2138 | −0.54pp |
| in_play_out | 0.4607 | 0.4270 | +3.37pp |
| truncated (horizon 6) | 0.0875 | 0.1028 | −1.53pp |

For scale: the v2-era stack produced K% 0.3339 against a 0.218 empirical (+11.6pp).

---

## 5. The §6 gate suite on 2024 BURNED DEV (`tier=burned-dev`)

Cohort: full 2024 pitcher-disjoint season, **40,042 PAs / 149,949 scored pitch rows**
(2,247 training pitchers excluded). 100 samples/PA, horizon 6, per-PA seeds `42 + i·1000`.
Frozen `pitchgpt_v2.pt` and `pitchgpt_v2_outcomehead_a1.pt` sha256 byte-identical pre/post.

### G1 — classwise-ECE and TACE per factor head (§6.1, §6.2, §6.7) → FAIL

15 equal-mass bins; TACE threshold 1e-3, 15 adaptive bins; classes with < 100 obs excluded from
the classwise mean and reported (`H_type`: classes 10, 13, 14, 16 with 23 / 0 / 39 / 13 obs).

| Head | C | v3 cwECE | frozen v2 cwECE | eff. threshold | | v3 TACE | frozen v2 TACE | eff. threshold | |
|---|---:|---:|---:|---:|:--|---:|---:|---:|:--|
| `H_type` | 17 | 0.02127 | 0.01674 | 0.01500 *(as written)* | **FAIL** | 0.02414 | 0.01727 | 0.01500 | **FAIL** |
| `H_zone` | 26 | 0.00404 | 0.00416 | **0.00500 *(tightened)*** | PASS | 0.00709 | 0.00462 | **0.00500 *(tightened)*** | **FAIL** |
| `H_velo` | 5 | 0.02281 | 0.01989 | 0.01000 *(as written)* | **FAIL** | 0.03504 | 0.02690 | 0.01000 | **FAIL** |
| outcome | 7 | **0.00603** | 0.04036 | 0.01000 *(as written)* | **PASS** | **0.00784** | 0.04318 | 0.01000 | **PASS** |

**§6.7 anti-unfailability executed before any contact**, on dev, per the rule
`0.6 × frozen-v2 dev value`, floored at 0.005, never loosened: it **tightened the `H_zone`
thresholds from 0.020 to 0.005 (4× harder)** and left `H_type`/`H_velo`/outcome as written
because frozen v2 already fails those lines. The zone TACE FAIL is a direct consequence of the
tightening — the mechanism worked exactly as designed, and is recorded here with both numbers.

The honest read: **v3's outcome head is 6.7× better calibrated than the frozen A1 head**
(0.00603 vs 0.04036) and clears its line; **v3's three field heads are slightly *worse* than
frozen v2's marginalised equivalents** and miss theirs. Factorizing made per-head calibration
*measurable* (H1's second clause) without making it *good*.

### G2 — KCE hypothesis test (§6.3) → FAIL, and the probe says the test is informative

Unbiased SKCE, Laplacian kernel on the simplex, 1,000-resample bootstrap. The gate fires when
calibration is *rejected* at p < 0.01.

**Denominator, stated in the section body (§9 entry 13):** every SKCE and p-value below is
computed on a fixed, seeded draw of **n = 2,000 of the 149,949 available scored pitch rows —
98.7% of the cohort discarded** (149,946 for the outcome head), seed 42, for the v3 statistic,
the power probe and the frozen-v2 reference alike. §6.3 authorizes no subsample; the unbiased
SKCE is an O(n²) U-statistic whose 1,000 bootstrap replicates are O(n²) again, so the full-n
kernel matrix (~1.7 TB for one head) is not computable here. No G2 number may be quoted without
that `n_used`.

**One kernel per head (§9 entries 14 and 16).** The bandwidth is the §6.3 median heuristic
fitted once on v3's 2024 dev probabilities, written write-once to
`models/kce_bandwidths_pitchgpt_v3_dev2024.json` (sha256
`d78903da10873ed2569c7320860fd8e59cceb44467e5bf0819cb884c61fef450`) and replayed for **all
three** measurements of that head. Two SKCE values are comparable only under one shared kernel.

| Head | pinned bandwidth | v3 SKCE | v3 p | frozen v2 SKCE | v2 p | v2 ÷ v3 | rejects v3? |
|---|---:|---:|---:|---:|---:|---:|---|
| `H_type` | 0.528886 | 2.6936e-03 | 0.0010 | 1.8710e-03 | 0.0010 | 0.69× | **YES** |
| `H_zone` | 0.591714 | 2.6104e-04 | 0.0010 | 2.0644e-04 | 0.0040 | 0.79× | **YES** |
| `H_velo` | 1.515810 | 2.3504e-03 | 0.0010 | 1.4909e-03 | 0.0010 | 0.63× | **YES** |
| outcome | 0.969238 | **3.6241e-04** | **0.0130** | 1.0722e-02 | 0.0010 | **29.6×** | no |

**SUPERSEDED — the frozen-v2 column as first published, measured under v2's *own* refitted
bandwidth (a different kernel from v3's, so not comparable to it).** Retained because it is what
the earlier version of this section printed while showing only v3's bandwidth column:

| Head | v2's own bandwidth | frozen v2 SKCE | v2 p | v2 ÷ v3 (cross-kernel, invalid) |
|---|---:|---:|---:|---:|
| `H_type` | 0.628343 | 2.1435e-03 | 0.0010 | 0.80× |
| `H_zone` | 0.528486 | 2.0142e-04 | 0.0030 | 0.77× |
| `H_velo` | 1.430578 | 1.4764e-03 | 0.0010 | 0.63× |
| outcome | 0.949141 | 1.0592e-02 | 0.0010 | 29.2× |

**Verdict, measured under the pinned kernel: G2 = FAIL, unchanged.** v3's own SKCEs and p-values
are bit-for-bit identical to the first run (asserted in-run; they were always computed at v3's
own bandwidth, which is the value now pinned), so the gate could not have moved: `H_type`,
`H_zone` and `H_velo` are all rejected at p = 0.0010 < 0.01. Only the frozen-v2 *reference* — which
§6.2 reports and never gates on — was recomputed.

**Power probe (mandatory, §6.3): all four heads reject a planted ×1.10 logit distortion at
p = 0.0010, under the same pinned kernel. G2 is therefore INFORMATIVE and is recorded as a genuine
FAIL, not VOID.**

**What the correction did to the comparison, in both directions.** The outcome head is still the
one head the test does not reject, and under the shared kernel its SKCE is **29.6× smaller** than
frozen v2's (29.2× under the old cross-kernel figures) — so that reading survives and is
marginally stronger. The three field heads move the *other* way: `H_type`'s frozen-v2 reference
drops from 2.1435e-03 to **1.8710e-03**, widening v3's deficit from 1.26× to **1.44×** worse than
the incumbent. The other two barely move and both move slightly in v3's favour: `H_velo` narrows
1.59× → 1.58×, `H_zone` narrows 1.30× → 1.26×. Net: the like-for-like comparison makes v3's
`H_type` look materially worse than the published cross-kernel number did, and leaves the other
two essentially where they were. The frozen-v2 `H_zone` p-value also moves 0.0030 → 0.0040, still a
rejection at the 1% level. Three of four *frozen v2* heads and three of four *v3* heads are
rejected, so G2 as implemented is a gate almost nothing passes — a property of the pre-registered
gate, flagged rather than fixed (§9 entry 13).

Re-measurement audit: `results/pitchgpt_v3/dev_2024_g2_bandwidth_fix_20260811T074725Z/audit.json`
(sha256 `cac232b303cee54057864bde5e26d6bff4e2cebd324ae9f7961a44e4ba06e183`), which retains both
tables above plus the bit-for-bit reproduction deltas against the first run; the pinned artifact's
sha256 is recorded beside it in `pin_record.json`.

### G3 — PA-level PIT and marginal calibration (§6.4) → FAIL

**PIT** (randomized, 20 equal-width bins, n = 35,982): **KS = 0.14447** vs ≤ 0.03 → FAIL; bin
masses outside `[0.0335, 0.0750]` → FAIL. The bin profile localizes the defect:

`0.051 0.052 0.051 0.050 0.050 0.050 0.049 0.050 0.053 0.051 0.048 0.050 0.046 0.046` **`0.087 0.134 0.074`** `0.009 0.000 0.000`

Bins 1–14 sit essentially at the 0.05 ideal; the entire failure is the pile-up at bins 15–17 and
the vacuum at 18–20 — the signature of a discrete 6-value PA-terminal wOBA support plus horizon-6
truncation, not a diffuse miscalibration.

**Marginals.** Both comparators are computed and **both must pass** (§9 entry 7): **A** replays
the empirical PAs through the production `_advance_count` and truncates at horizon 6, so the same
estimator scores both sides; **B** is the spec-literal PHASE_0.6 §3.3 SQL baseline (uncapped).

| Quantity | v3 model | A empirical | Δ vs A | tol | | B empirical (SQL) | Δ vs B | |
|---|---:|---:|---:|---:|:--|---:|---:|:--|
| K% | 0.16620 | 0.18228 | −1.608pp | 1.000pp | **FAIL** | 0.22121 | −5.501pp | **FAIL** |
| BB% | 0.07076 | 0.05974 | +1.102pp | 0.597pp | **FAIL** | 0.08113 | −1.037pp | **FAIL** |
| HR% | 0.03094 | 0.03110 | **−0.016pp** | 0.311pp | **PASS** | 0.03135 | **−0.041pp** | **PASS** |
| HBP% | 0.00676 | 0.01081 | −0.405pp | 0.108pp | **FAIL** | 0.01150 | −0.474pp | **FAIL** |
| hit% | 0.20772 | 0.20881 | −0.108pp | 1.000pp | **PASS** | 0.22058 | −1.286pp | **FAIL** |
| mean wOBA | 0.26179 | 0.26166 | **+0.00013** | 0.015 | **PASS** | 0.32383 *(DB `woba_value`)* | −0.06205 | **FAIL** |
| mean PA length | 3.7649 | 3.7448 | +0.020 | 0.5 | **PASS** | 3.8833 | −0.118 | **PASS** |
| `calibration_valid` coverage | **0.9804** | — | — | ≥ 0.95 | **PASS** | — | — | — |

HR% is derived as `P(in_play_hit) × HR|hit`, with `HR|hit = 0.1489` measured on the **2023 fit
cohort** — never on a gate cohort (§9 entry 7c); the undecomposed `P(in_play_hit) = 0.20772` is
the structural upper bound, since the 7-class vocabulary has one hit channel. `calibration_valid`
uses the v3 analogue defined in §9 entry 8. **8.74%** of rollouts never terminated within
horizon 6 — the main driver of the K%/hit% deficits against comparator B.

### G4 — per-count-state calibration and the position gate (§6.5) → FAIL

**Position gate: max \|rollout − empirical\| = 2.5408pp** vs ≤ 1.00pp at position 1, class
`ball` → **FAIL**. Per-position maxima (pp): 1.680 / **2.541** / 1.051 / 1.745 / 2.136 / 1.126.
Secondary, **measured on this 2024 dev cohort** (not the 2023 fit cohort, whose value is 0.6 —
see §4): ρ(position, KL) = **0.2571**, p = 0.6228, n = 6 positions, vs the v2-era 0.822. The
monotone within-PA drift that was the exposure-bias fingerprint is no longer detectable here, but
with n = 6 and p = 0.62 this is an absence of evidence, not evidence of absence.

**Per-count-state top-1 ECE of the outcome head** (10 equal-mass bins, gated where n ≥ 500;
§6.7 tightening applied per state): **8 of 12 states PASS**. v3's ECE is lower than frozen v2's in
**11 of 12** states (the sole exception is count 1-0: v3 0.01990 vs v2 0.01488).

| Count | n | v3 ECE | frozen v2 ECE | eff. threshold | Verdict |
|---|---:|---:|---:|---:|---|
| 0-0 | 39,972 | 0.00941 | 0.06368 | 0.02000 | PASS |
| 0-1 | 20,158 | 0.01104 | 0.03431 | 0.02000 | PASS |
| 0-2 | 10,632 | 0.01618 | 0.03894 | 0.02000 | PASS |
| 1-0 | 15,209 | 0.01990 | 0.01488 | **0.00893** *(tightened)* | **FAIL** |
| 1-1 | 15,708 | 0.01229 | 0.01837 | **0.01102** *(tightened)* | **FAIL** |
| 1-2 | 15,075 | 0.01827 | 0.04494 | 0.02000 | PASS |
| 2-0 | 4,993 | 0.01512 | 0.20120 | 0.02000 | PASS |
| 2-1 | 8,025 | 0.01395 | 0.02150 | 0.02000 | PASS |
| 2-2 | 11,235 | 0.02626 | 0.05008 | 0.02000 | **FAIL** |
| 3-0 | 1,541 | 0.04002 | 0.23183 | 0.02000 | **FAIL** |
| 3-1 | 3,304 | 0.01814 | 0.02708 | 0.02000 | PASS |
| 3-2 | 4,094 | 0.01869 | 0.15349 | 0.02000 | PASS |

Two of the four failures (1-0, 1-1) are failures against a §6.7-tightened line that frozen v2
set; on the spec-as-written 0.02 line both would pass. That is the anti-unfailability rule doing
its job, and it is stated rather than quietly absorbed. 3-0 (0.04002) is a real miss on the
spec's own line, on the thinnest gated state.

### G5 — decision calibration against K / BB / HR (§6.6) → FAIL

Deciles of the model's predicted P(d); gated where n ≥ 500 (all 10 deciles gated for all three
decisions, `insufficient_data: false`).

| Decision | max gated decile gap | ≤ 1.0pp | n-weighted mean gap | ≤ 0.5pp | Verdict |
|---|---:|---|---:|---|---|
| K | **11.1155pp** | FAIL | **5.8107pp** | FAIL | **FAIL** |
| BB | 4.7017pp | FAIL | 2.3170pp | FAIL | **FAIL** |
| HR | 1.5727pp | FAIL | 0.5131pp | FAIL | **FAIL** |

HR is the near-miss: 8 of 10 deciles pass and the weighted gap misses 0.5pp by 0.013pp. The K
decile table is in §1 above.

### §6.8 verdict logic applied

G1 FAIL ∧ G2 FAIL (not VOID) ∧ G3 FAIL ∧ G4 FAIL ∧ G5 FAIL → **DEV-TIER VERDICT: FAIL.**
No partial pass, no gate dropped, no threshold moved after the numbers landed.

---

## 6. Compute budget (§7.3)

| Run | GPU / wall seconds |
|---|---:|
| Stage A pretrain | 427.3 |
| K-v2-FIT-A dev NLL | 4.7 |
| Stage B curriculum (completed) | 7,765.3 |
| Stage B attempt 1 (harness-killed, §9 10a) | ~3,480 |
| Outcome head | 819.3 |
| Calibration | 2.0 |
| K-v2-FIT-B (2023, two rolls) | 206.6 |
| 2024 dev gate suite | 1,121.1 |
| Smoke / plumbing runs | ~900 |
| **Total** | **~14,726s = 4.09 GPU-h** |

Cap 12 GPU-h; largest single run 2.16h against a 6h cap. **NO KILL.**

---

## 7. Artifacts, provenance, integrity

| Artifact | Path | sha256 (16) | Registry version |
|---|---|---|---|
| Factorized backbone + heads | `models/pitchgpt_v3_factorized.pt` | `8c11f1c9219dc932` | `pitchgpt/v2026.08.11-factorized` |
| Outcome head | `models/pitchgpt_v3_outcomehead.pt` | `4d83c4a989d6fc52` | `pitchgpt/v2026.08.11-factorized-outcomehead` |
| Per-head temperatures | `models/calibration_pitchgpt_v3.json` | `5055946cee57f49d` | `pitchgpt/v2026.08.11-factorized-calibration` |
| §6.3 pinned KCE bandwidths | `models/kce_bandwidths_pitchgpt_v3_dev2024.json` | `d78903da10873ed2` | `pitchgpt/v2026.08.11-factorized-kce-bandwidths` |
| Stage A checkpoint (intermediate) | `models/pitchgpt_v3_factorized_stage_a.pt` | `c0e73ba8c15d67a1` | not registered |

All four registered **write-once** with WS2.1 manifests, `hash_policy: pinned`. Re-running the
registration raises `FileExistsError`. `scripts/verify_artifacts.py` → **ok=24, warn=0, fail=0**.

**No alias changed.** `pitchgpt/production` and `pitchgpt/frozen_validated` both still resolve to
`v2026.04.23` (§8.1). The alias move is a separate reviewed step available only after a §6.8 PASS
on the sealed-2026 contact — which has not happened and, while the §7.2 kill stands, is not
authorized.

Audit JSONs (each carries `spec_path`, `spec_freeze_sha`, `git_sha`, seeds, cohort definition and
row counts, per-head parameter counts and the §3.3 assertion results, `support_counts` hash, mask
fallback/degenerate counts, wall clock and GPU seconds, frozen-checkpoint sha256 pre/post, and
`duckdb_read_only: true`):

```
results/pitchgpt_v3/train_stage_a_20260811T022833Z/audit.json
results/pitchgpt_v3/dev_2024_nll_stage_a_20260811T022934Z/audit.json
results/pitchgpt_v3/train_stage_b_20260811T054847Z/audit.json
results/pitchgpt_v3/train_outcome_head_20260811T060528Z/audit.json
results/pitchgpt_v3/fit_2023_calibration_20260811T060636Z/audit.json
results/pitchgpt_v3/fit_2023_killB_20260811T061012Z/audit.json
results/pitchgpt_v3/dev_2024_gates_20260811T062956Z/audit.json
results/pitchgpt_v3/dev_2024_g2_bandwidth_fix_20260811T074725Z/audit.json   (+ pin_record.json)
```

**Integrity, all clean:** `models/pitchgpt_v2.pt`
(`6f952054d14ac6f918f3eb9502b496b70bc0c87dfc65dc50d98ee7244a62883c`) and
`models/pitchgpt_v2_outcomehead_a1.pt`
(`37b50e87599013c281560c9f63286fe5b7645297d0042694d907287417bb25e5`) byte-identical pre/post in
every run that touched them; nothing existing overwritten; DuckDB `read_only=True` throughout; no
tables or views created.

### Data policy, as executed

| Tier | Season | Status |
|---|---|---|
| Train | 2015–2022 | read (9,999 games, seeded reservoir) |
| Fit / calibration / §7.2 kill | 2023 pitcher-disjoint | read (19,653 PAs) |
| **Dev (BURNED)** | **2024 pitcher-disjoint** | **read (40,042 PAs) — no validation authority** |
| **Budgeted** | **2025** | **NEVER READ. 12/14 contacts stand, both remaining reserved** |
| **Lockbox** | **2026 full season** | **NEVER READ. Sealed. NOT CONTACTED** |

Enforced structurally, not by discipline: `pitchgpt_v3_data.assert_season_policy` refuses 2025
under every flag and refuses 2026 unless `allow_lockbox=True`, whose only call site in the repo
is inside `scripts/pitchgpt_v3_lockbox_2026.py::run_lockbox_2026_grading` — itself decorated
`@holdout_access(dataset="lockbox_2026_full_season", budget=1)`, which raises
`HoldoutSealedError` before the body runs while the ledger says `unsealed: false`, and whose body
is a documented `NotImplementedError` stub. Its `main()` refuses outright. An AST-based test
fails the build if any v3 executable statement embeds 2025 or 2026 as a literal outside the
policy constants and that one ledger-gated door.

### Sim integration

`src/analytics/pitchgpt_sim.py` is **unmodified**; a test asserts the string `pitchgpt_v3` does
not appear in it and that no v3 predictor is in `OutcomePredictorRegistry`. The only sim-shaped
door is `pitchgpt_v3_rollout.rollout_v3_optin`, which raises `V3SimOptInError` unless the caller
passes `enable_v3=True` or sets `PITCHGPT_V3_SIM_OPTIN=1`. Production sim behaviour is unchanged.

---

## 8. What this does and does not authorize

**Does:**
- H1 is supported on dev: chain-rule factorization is free (−0.650% NLL) at 10.1× fewer output
  parameters, and it makes per-head calibration estimable at C = 17/26/5/7.
- §3.5's no-class-weighting change is vindicated: the A1 head's −11.6pp `ball` deficit became
  −0.16pp, and outcome-head cwECE improved 6.7× over frozen v2.
- The exposure-bias attack works directionally: 16.37pp → 1.8852pp, better than the banned
  reweighting's 2.625pp. The drift signature ρ(position, KL) reads 0.6 (p = 0.208) on that fit
  cohort against the v2-era 0.822 — directionally weaker, not significant at n = 6, and not a
  claim on its own.

**Does not:**
- No PA-level absolute-rate product unblocks. H3 is blocked; the per-pitch-only claim stands.
- No number here is a validation. 2024 is burned. `tier=burned-dev` on every figure.
- No claim about v3 on sealed 2026 exists, because no 2026 datum has been read.
- Nothing is re-earned for the v2-era stack (claims `pitchgpt_pa_rates_fail`,
  `pitchgpt_woba_pa_pass_pre062` unchanged).
- The §4.5 Stage-B repair pass is available but unspent, and spending it requires a dated §9
  entry naming the single change *before* the run.

**Open decision for the orchestrator / user (not the executor's to make):** whether to spend the
one remaining §4.5 curriculum run on a pre-registered single change, or to close WS5.2 at the
§7.2 kill with this negative published. Either way the §5.5 lockbox stays sealed until a
pre-registered spec version is ready to spend its one contact at season end.
