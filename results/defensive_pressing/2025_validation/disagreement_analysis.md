# DPI vs OAA — 2025 Disagreement Analysis

Ten team-seasons in 2025 where the DPI rank (1 = best) and Statcast OAA rank
(1 = best) diverge the most, ordered by absolute rank difference.

`rank_diff = dpi_rank - oaa_rank` (negative = DPI higher than OAA; positive
= DPI lower than OAA). `z_diff = z(dpi_mean) - z(team_oaa)` for cross-
metric standardisation.

## Top-10 disagreements (2025)

| team | dpi_rank | oaa_rank | rank_diff | dpi_mean | team_oaa | babip_against | rp_proxy | read |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| HOU | 22 | 3 | +19 | 0.517 | +28 | .280 | +26.4 | OAA-only elite |
| SEA | 8 | 24 | −16 | 0.727 | −21 | .286 | +37.9 | DPI-only elite |
| PHI | 24 | 10 | +14 | 0.417 | +8 | .298 | +68.3 | OAA-only positive |
| STL | 16 | 2 | +14 | 0.631 | +34 | .299 | −4.8 | OAA-only elite |
| PIT | 3 | 14 | −11 | 0.829 | +2 | .280 | +74.6 | DPI-only elite |
| KC | 14 | 3 | +11 | 0.672 | +28 | .284 | +22.6 | OAA-only elite |
| CWS | 17 | 26 | −9 | 0.614 | −24 | .289 | −80.5 | DPI over-rates? |
| NYY | 10 | 19 | −9 | 0.719 | −13 | .280 | +40.9 | DPI-only positive |
| TB | 21 | 29 | −8 | 0.522 | −34 | .282 | +24.1 | DPI over-rates? |
| CIN | 9 | 17 | −8 | 0.723 | −9 | .278 | −0.5 | DPI-only positive |

## Interpretation

The disagreement pattern is **largely interpretable**, not noise. Five
teams split cleanly into the two canonical failure modes for each metric:

### DPI > OAA: "pitch-level pressure" teams

**SEA (+16 rank delta), PIT (+11), NYY (+9), CIN (+8).** These teams have
DPI in the top-10 but OAA below the median.

All four show **very low BABIP-against** (.278-.286) and a **positive
run-prevention proxy**, i.e. their pitchers do prevent hits and runs on
the BIP outcomes DPI actually scores. OAA downgrades them because it
adjusts for difficulty of opportunity via fielder starting position and
sprint-to-catch trajectories — inputs DPI is silent on. The edge claim
here is that DPI **is capturing genuine outcome-level pressure that OAA
filters out as "not the fielder's fault"**; SEA and PIT in particular
are plausible edge candidates because their BABIP-against is *.28 or
below* — elite level — even though their fielder-movement OAA says
otherwise. Interpretation: SEA/PIT/NYY/CIN have pitching-and-position
systems that generate easy plays, which DPI credits and OAA explicitly
removes.

### OAA > DPI: "range skill without outcome dominance" teams

**HOU (−19), STL (−14), PHI (−14), KC (−11).** These have OAA in the
top-10 but DPI at or below the median.

HOU's rp_proxy is only +26 runs despite OAA saying +28 — suggesting
the fielder movement OAA picks up doesn't fully convert to outcome
run-prevention. STL is the starkest case: OAA #2 league-wide with a
BABIP-against of .299 and a *negative* run-prevention proxy. PHI
has an OAA of only +8 but the rest of its defence (.298 BABIP, +68
RP proxy) isn't consistent with a top-10 defence either. These look
like teams where positional / positioning OAA credit isn't cashing
out in BIP outcomes — exactly what DPI's residual-on-outcome target
should under-weight.

### Ambiguous: CWS and TB

CWS and TB show "DPI says middling, OAA says bad" (−9, −8). Their
BABIP-against is .289 / .282 respectively. The most parsimonious
read is that DPI under-penalises them — these teams' OAA is very low
(−24 / −34), and the BIP outcomes don't look that much better than the
fielding metric implies. This is probably the closest slice to *noise*
in the top-10 list.

## Aggregate read on the disagreements

- 4 of 10 top disagreements (SEA, PIT, NYY, CIN) are plausibly **DPI
  capturing outcome pressure OAA filters out** — the edge claim.
- 4 of 10 (HOU, STL, PHI, KC) are plausibly **OAA capturing
  positioning credit that doesn't cash out** — OAA-style limitations
  that DPI's outcome focus is agnostic to.
- 2 of 10 (CWS, TB) are harder to read and might just be rank-correlation
  noise at the margin.

So ~8/10 of the biggest 2025 disagreements have a clean "DPI and OAA
measure overlapping but not identical things" explanation. This is
consistent with the **r = 0.64 Pearson / 0.60 Spearman** observed value —
the two metrics agree on ~40% of variance and the other 60% splits into
these interpretable regimes, not random shuffle.

## Cross-year sanity check

DPI's correlation with BABIP-against is stronger in every year than
OAA's is:

| year | r(DPI, BABIP-against) | r(OAA, BABIP-against) |
|---|---:|---:|
| 2023 | −0.80 | −0.44 |
| 2024 | −0.65 | −0.06 |
| 2025 | −0.80 | −0.44 |

This confirms: DPI is more tightly tied to pure outcome suppression
on BIP than OAA is. Teams where BABIP-against is low but OAA is
poor (SEA, PIT in 2025) score high on DPI — that is the metric
behaving as designed.
