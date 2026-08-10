# Contrarian Leaderboards -- 2026 Mid-Season (live)

> **APPLICATION of the 2023-2025 CausalWAR contrarian methodology to PARTIAL-SEASON 2026 data. The headline Buy-Low hit rate (68.4%, 13/19) is NOT a validated edge: its hit criterion was defined post-hoc, the 95% CI [0.474, 0.843] includes chance, intention-to-treat scoring (exits count against the pick) is 13/25 = 52%, and matched-naive mean-reversion baselines score 66.5-73% on the same pools. It was measured on FULL-SEASON 2023-24 picks resolved against 2025 outcomes and does NOT transfer to these mid-season boards. 2026 picks here are UNRESOLVED and can only be scored after future seasons.**

_Generated from CausalWAR-2026 (leaderboard_cache) vs 2026 season-to-date Baseball-Reference WAR. Season snapshot through Aug 3, 2026._

## Cohort gates (mid-season, pro-rated)

- Season progress applied: **68%** of a 162-game season (~110 games through Aug 3).
- Batters: **PA >= 204** (round(0.68 x 300)).
- Pitchers: **IP >= 34** (round(0.68 x 50)) -- carried for completeness; the 2026 CausalWAR leaderboard is batter-only.

## Pool

- CausalWAR-2026 leaderboard rows: 405
- With non-null 2026 bWAR (rankable): 405
- Passing the mid-season cohort gate: 278
- Phillies in qualified pool: 9

## Buy-Low (CausalWAR > bWAR)

Players my model ranks higher than the public WAR market at the 2026 mid-season snapshot (positive rank_diff). Sorted by rank_diff descending.

| # | Player | Team | CausalWAR | bWAR | RankDiff | PA | Tag |
|---|---|---|---|---|---|---|---|
| 1 | Nathaniel Lowe | CIN | 0.82 | 0.64 | +124 | 276 | OTHER |
| 2 | Ryan O'Hearn | PIT | 0.64 | 0.41 | +121 | 398 | OTHER |
| 3 | George Springer | TOR | 0.45 | 0.26 | +121 | 360 | OTHER |
| 4 | Spencer Steer | CIN | 0.41 | 0.24 | +120 | 384 | OTHER |
| 5 | Carlos Cortes | ATH | 0.04 | -0.15 | +108 | 260 | OTHER |
| 6 | Jesús Sánchez | TOR | -0.02 | -0.35 | +102 | 240 | OTHER |
| 7 | Ronald Acuña | ATL | 0.67 | 0.74 | +100 | 266 | OTHER |
| 8 | Francisco Álvarez | NYM | 0.74 | 0.89 | +96 | 278 | OTHER |
| 9 | Spencer Horwitz | PIT | 0.96 | 1.09 | +93 | 303 | OTHER |
| 10 | Oneil Cruz | PIT | 1.13 | 1.20 | +92 | 283 | OTHER |

**Buy-Low mechanism tags:**

- `OTHER`: 25

## Over-Valued (bWAR > CausalWAR)

Players bWAR ranks higher than CausalWAR -- value likely leaning on glove / park / sequencing the per-PA model cannot see (negative rank_diff). Sorted by rank_diff ascending.

| # | Player | Team | CausalWAR | bWAR | RankDiff | PA | Tag |
|---|---|---|---|---|---|---|---|
| 1 | Taylor Walls | TB | -1.03 | 2.49 | -205 | 310 | OTHER |
| 2 | Nasim Nuñez | WSH | -0.99 | 2.27 | -190 | 368 | OTHER |
| 3 | Evan Carter | TEX | -0.70 | 2.15 | -159 | 317 | OTHER |
| 4 | Masyn Winn | STL | -0.82 | 1.81 | -146 | 430 | OTHER |
| 5 | Jackson Merrill | SD | -0.71 | 1.64 | -125 | 461 | OTHER |
| 6 | Brayan Rocchio | CLE | -0.29 | 2.24 | -124 | 427 | OTHER |
| 7 | Matt Chapman | SF | -0.07 | 2.88 | -120 | 352 | OTHER |
| 8 | Nico Hoerner | CHC | -0.47 | 1.82 | -117 | 490 | OTHER |
| 9 | José Caballero | NYY | -0.26 | 2.12 | -114 | 348 | OTHER |
| 10 | Caleb Durbin | BOS | -0.05 | 2.70 | -113 | 388 | OTHER |

**Over-Valued mechanism tags:**

- `OTHER`: 25

## Methodology + honest caveats

- **Tags** reuse the dashboard mechanism-tag heuristic (`src/dashboard/views/contrarian_leaderboards.py::_classify_row`): `RELIEVER LEVERAGE GAP`, `PARK FACTOR`, `DEFENSE GAP`, `GENUINE EDGE?`, `OTHER`. Internal tag thresholds (e.g. GENUINE EDGE needs PA>=400) are the FULL-SEASON values from the 2023-24 evidence run applied as-is, so fewer rows earn GENUINE at mid-season.
- The 2026 CausalWAR leaderboard is **batter-only**, so RELIEVER LEVERAGE GAP and PARK FACTOR (pitcher tags) do not appear on this board.
- **DEFENSE GAP** depends on `players.position`, which is currently NULL for the 2026 leaderboard players, so that tag rarely fires; it will activate automatically once fielding positions are populated.
- Rows with NULL 2026 bWAR are dropped, never imputed. NULL stays NULL.
- **These picks are unresolved.** The headline 68.4% Buy-Low hit rate is a full-season 2023-24 -> 2025 result and is NOT a validated edge (post-hoc hit criterion; 95% CI [0.474, 0.843] includes chance; intention-to-treat is 13/25 = 52%; matched-naive baselines score 66.5-73% on the same pools). It does not transfer here; resolution requires a future season.
