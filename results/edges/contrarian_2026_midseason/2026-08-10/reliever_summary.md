# Contrarian RELIEVER Board -- 2026 (frozen 2026-08-10)

> **APPLICATION of the 2023-2025 CausalWAR contrarian methodology to PARTIAL-SEASON 2026 data -- pitcher/reliever side. NOT a validated edge. The reliever-leverage cohort is the only historical cohort that cleared its within-filter naive base rate (Buy-Low 25/32 = 78.1% vs 56.9% within-filter naive), but n=32 is SMALL, the rate aggregates three windows, and the hit rule inherits the post-hoc criterion (2026-08-10 audit). That historical rate does NOT transfer to these picks. The batter-side headline (68.4%, 13/19) is likewise NOT a validated edge (post-hoc hit criterion; 95% CI [0.474, 0.843] includes chance; intention-to-treat is 13/25 = 52%; matched-naive baselines score 66.5-73% on the same pools). 2026 picks here are UNRESOLVED. Resolution is PRE-REGISTERED and mechanical: frozen spec `docs/models/contrarian_2026_resolution_spec.md` section 6 (hypothesis H4) -- ITT scoring with the section-5.3 exit asymmetry, WITHIN-FILTER matched-naive control, pitcher-Marcel RA9 control, one resolution event at end of 2026 regular season + 7 days; a miss publishes as prominently as a win (K4). Spec sha256 at freeze: `1a27cd0e2d9b7d08c69c5a8a5944602585931121d9396be922e1e519557c760e` (commit `912ede6`).**

_Pitcher-side CausalWAR (frozen 2015-2022 nuisance checkpoint `models/causal_war/causal_war_trainsplit_2015_2022.pkl`, pitcher aggregation >= 50 batters faced) vs 2026 season-to-date Baseball-Reference bWAR (frozen staging parquet as-of 2026-08-09, spec artifact A3). DB pitch data through 2026-08-08._

## Pre-registered criterion (spec section 6 -- fixed before this board existed)

- Progress constant `p_f` = **0.7531** (median 122 team games through 2026-08-09 / 162; spec s6.2).
- Reliever band (s6.3): **IP >= 15** (= round(20 x p_f)) AND **IP < 45.1860** (= 60 x p_f, full precision).
- Ranks within the qualified reliever pool; Buy-Low requires rank_diff > 0, Over-Valued rank_diff < 0; up to 25 per side; ties by larger |causal_war - trad_war| then lower player_id (s6.4).
- Scoring (s6.5): pace target T = snapshot bWAR / 0.7531; exit floor RoS_IP < 30 x (1 - p_f) = 7.4070; ERA surrogate (IP >= 30, Buy-Low HIT iff ERA <= 4.00); ITT with the s5.3 exit asymmetry.
- Matched-naive (s6.6): WITHIN-FILTER -- neighbors only from the qualified reliever pool, window +/-0.3 x p_f = +/-0.2259, >= 3 neighbors, single-basis (this frozen parquet is both pick and neighbor baseline).
- Marcel-pitcher control (s6.7): projected RA9, parameters pinned by a dated s8 entry before season end; miss the deadline -> VOID-M.
- Hypothesis H4 (s2): reliever Buy-Low ITT vs within-filter matched-naive ITT. K4 publication rule applies identically.

## Pool composition (honest accounting)

- Staging-parquet pitcher rows (as-of 2026-08-09): 788
- In the IP band on the parquet basis: 244 (this is the s6.6 neighbor pool)
- Rankable (band AND >= 50 BF CausalWAR effects): **244**
- rank_diff > 0: 124 | rank_diff < 0: 117 | rank_diff = 0 (excluded by the s6.4 sign requirement): 3
- Board sides: Buy-Low **25** / Over-Valued **25** (spec allows up to 25; shortfalls are the spec's own branch, not stretched)
- Board IP distribution: min 15.7 / median 29.5 / max 45.0
- Baseline-bWAR sign mix (direction-of-effect disclosure, cf. spec s4.5): Buy-Low snapshot bWAR spans -1.12 to +0.22 (20 negative / 1 zero / 4 positive); Over-Valued spans -0.11 to +1.01 (3 negative / 1 zero / 21 positive). Under the frozen pace rule a NEGATIVE baseline makes the bullish target lenient (final bWAR >= a negative number) and the bearish target strict; most Buy-Low baselines here are negative. This is the pre-registered rule's own behavior -- disclosed, not adjustable after the fact.

## Buy-Low (CausalWAR > bWAR), top 10

| # | Player | Team | CausalWAR | bWAR | RankDiff | IP | Tag |
|---|---|---|---|---|---|---|---|
| 1 | Hunter Bigge | TB | 0.11 | -0.84 | +155 | 19.3 | RELIEVER LEVERAGE GAP |
| 2 | Joe Boyle | TB | 0.20 | -0.20 | +116 | 15.7 | RELIEVER LEVERAGE GAP |
| 3 | Camilo Doval | NYY | -0.10 | -0.74 | +111 | 42.7 | RELIEVER LEVERAGE GAP |
| 4 | Will Vest | DET | -0.18 | -0.90 | +99 | 26.7 | RELIEVER LEVERAGE GAP |
| 5 | Justin Slaten | BOS | -0.15 | -0.64 | +96 | 30.3 | RELIEVER LEVERAGE GAP |
| 6 | Wilber Dotel | PIT | 0.21 | -0.07 | +95 | 26.7 | RELIEVER LEVERAGE GAP |
| 7 | Alex Hoppe | SEA | 0.00 | -0.23 | +91 | 28.7 | RELIEVER LEVERAGE GAP |
| 8 | Hunter Greene | CIN | -0.06 | -0.36 | +87 | 27.7 | RELIEVER LEVERAGE GAP |
| 9 | Pete Fairbanks | MIA | -0.27 | -1.08 | +86 | 36.3 | RELIEVER LEVERAGE GAP |
| 10 | Riley Cornelio | WSH | -0.10 | -0.45 | +85 | 20.0 | RELIEVER LEVERAGE GAP |

## Over-Valued (bWAR > CausalWAR), top 10

| # | Player | Team | CausalWAR | bWAR | RankDiff | IP | Tag |
|---|---|---|---|---|---|---|---|
| 1 | Richard Lovelady | WSH | -0.70 | 0.41 | -158 | 30.3 | OTHER |
| 2 | Jose Quintana | COL | -0.67 | 0.30 | -141 | 41.0 | OTHER |
| 3 | Blas Castano | COL | -0.41 | 0.40 | -119 | 20.7 | OTHER |
| 4 | Shane Bieber | TOR | -0.54 | 0.23 | -114 | 37.0 | OTHER |
| 5 | Jesse Scholtens | TB | -0.34 | 0.30 | -99 | 37.7 | OTHER |
| 6 | Taylor Rogers | MIN | -0.50 | 0.14 | -95 | 40.7 | OTHER |
| 7 | Seth Halvorsen | COL | -0.47 | 0.19 | -94 | 19.0 | OTHER |
| 8 | Chase Shugart (PHI) | PHI | -0.31 | 0.31 | -93 | 38.3 | OTHER |
| 9 | Patrick Sandoval | BOS | -0.37 | 0.24 | -93 | 24.0 | OTHER |
| 10 | John Schreiber | KC | -0.32 | 0.28 | -86 | 42.7 | OTHER |

## Freeze provenance

- Board CSV sha256: `B4C4F51702FD308581CB6D141C1CCA3175709D9C2F3606A36444B3A7D545C556`
- Baseline parquet (A3) sha256: `0DF755CF2C3AC30BEE4C4EFD2FF1CA9ABEF47C755E4F36DD9986D8B839249EDD` (as-of 2026-08-09)
- Nuisance checkpoint sha256: `5D13242CC3C3D7F0F8A33D52AAAA6BF337B9F1F576C0722484BE11AAC0B97672`
- Frozen spec: `docs/models/contrarian_2026_resolution_spec.md` sha256-at-freeze `1a27cd0e2d9b7d08c69c5a8a5944602585931121d9396be922e1e519557c760e` (commit `912ede6`)
- Picks appended to `predictions/picks.jsonl` (50 emitted, 0 already present); evidence_class `prospective`.
- Deviation record: spec s8 entry 2 (no fresh fetch under Batch-B read-only constraints; A3 reused as the freeze fetch; p_f measured through the A3 as-of date for single-basis consistency with s6.6).
