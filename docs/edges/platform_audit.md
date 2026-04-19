# Platform Edge Audit

## Headline

The platform contains 28 modules under `src/analytics/`. Five are flagship-validated this session (CausalWAR, PitchGPT, MechanixAE, VWR, Allostatic Load) and three are infrastructure (`base.py`, `features.py`, `registry.py`, `validation.py`). That leaves 20 unvalidated analytics modules, 17 of which are independent models (PitchGPT has three baseline siblings — `pitch_lstm`, `pitch_markov`, plus a calibration helper — that are scaffolding, not standalone edges). Of those 17 independent models, 15 already have a corresponding `src/dashboard/views/` page, meaning the platform surfaces a lot of unvalidated machinery to the user.

## Inventory Table (Unvalidated Models)

| Model | What it computes (1 sentence) | Dashboard view? | Tested? | Trained artifacts? | Edge-potential | Recommended next step |
|---|---|---|---|---|---|---|
| alpha_decay | Quant-trading "alpha decay" applied to pitch sequencing: how fast a pitch-pair's whiff uplift erodes with repetition (within game / series / season). | yes (`alpha_decay.py`) | yes (`test_alpha_decay.py`) | none | HIGH | Validate in-game decay predictions vs out-of-sample whiffs — H1 hr. |
| anomaly | Live-game velocity / spin / release-point / pitch-mix anomaly detection vs pitcher baseline. | yes (`anomalies.py`) | none | none (baselines computed at runtime) | MEDIUM | Add a regression test fixture; otherwise wire a Phillies-only live alert. |
| baserunner_gravity | NBA-style "gravity" applied to baserunners: how a runner distorts pitch selection, location, and outcome on the batter. | yes (`baserunner_gravity.py`) | yes (`test_baserunner_gravity.py`) | dir exists, empty | HIGH | Validate against known burner cohorts (Acuna, Witt) — H1 hr. |
| bullpen | Reliever fatigue tracking + Leverage-Index-weighted matchup recommender. | yes (`bullpen.py`) | none | none | MEDIUM | Backtest reliever recommendations vs actual late-inning outcomes. |
| chemnet | GAT-based lineup synergy / protection model; predicts game wOBA from order graph and attention-weights pairwise effects. | yes (`chemnet_view.py`) | yes (`test_chemnet.py`) | `chemnet_v1.pt` (35 KB) | HIGH | Validate synergy score predicts next-game wOBA out-of-sample — half day. |
| defensive_pressing | Soccer gegenpressing applied to defense: HistGB expected-out model -> game/season DPI vs expectation. | yes (`defensive_pressing.py`) | yes (`test_defensive_pressing.py`) | dir exists | HIGH | Validate DPI rankings correlate with DRS/OAA (Statcast). H1 day. |
| kinetic_half_life | Pharmacokinetic exponential-decay fit on per-game stuff quality -> pitcher stamina half-life K1/2. | yes (`kinetic_half_life.py`) | yes (`test_kinetic_half_life.py`) | none (per-game fit) | MEDIUM | Compare K1/2 vs TTO penalty / pitch-count survival curves. |
| loft | VPIN-style "order flow toxicity" of pitches: bucketed buy/sell imbalance with EWMA + alert sigma. | yes (`loft.py`) | yes (`test_loft.py`) | none | HIGH | Validate alert breach predicts adverse next-inning xwOBA — H1 hr. |
| matchups | Bayesian Beta-Binomial pitcher-vs-batter wOBA estimator with shrinkage to platoon priors. | yes (`matchups.py`) | none | none | MEDIUM | Add tests; benchmark against Steamer matchup projections. |
| mesi | Motor Engram Stability: SNR of execution vectors x context stability x learning-curve fit. | yes (`mesi.py`) | yes (`test_mesi.py`) | none | MEDIUM | MESI is conceptually adjacent to MechanixAE — clarify positioning vs MechanixAE before validating. |
| pitch_decay | F1-style per-pitch-type fatigue cliff via piecewise linear regression -> "first to die" pitch type. | yes (`pitch_decay.py`) | yes (`test_pitch_decay.py`) | none | HIGH | Validate cliff prediction vs actual per-type quality drop in held-out games. |
| pitch_lstm | LSTM baseline for PitchGPT (Ticket #3 foil). | no (correct — baseline) | yes (`test_pitch_lstm.py`) | trained alongside PitchGPT (`scripts/train_pitchgpt_vs_lstm.py`) | n/a | Keep as baseline; do not promote. |
| pitch_markov | Markov + heuristic baselines for PitchGPT (Tickets #4 / #5). | no (correct — baseline) | yes (`test_pitch_markov.py`) | none (fit at runtime) | n/a | Keep as baseline; do not promote. |
| pitch_sequencing | Heuristic sequencing engine: transition matrices, count-specific tendencies, setup-knockout pairs, tunneling. | yes (`sequencing.py`) | none | none | MEDIUM | Largely overlaps with PitchGPT and alpha_decay. Decide whether to retire or keep as the descriptive surface. |
| pset | Tennis-serve game-theory PSET: run_value + tunnel_bonus - predictability_penalty per pitch pair. | yes (`pset.py`) | yes (`test_pset.py`) | none | HIGH | Validate PSET-per-100 correlates with pitcher SIERA / xFIP residual. H1 day. |
| sharpe_lineup | Markowitz portfolio theory on batter game-wOBA: PSR + efficient-frontier lineup optimizer. | yes (`sharpe_lineup.py`) | yes (`test_sharpe_lineup.py`) | none | MEDIUM | Validate frontier-optimal lineups outperform actual lineups in a backtest. Cute, but optimizing wOBA without LI weighting limits real-world relevance. |
| stuff_model | Stuff+: HistGB on physical features predicting pitch run value, scaled to 100-centered grade. | yes (`stuff_plus.py`) | none | `stuff_model.pkl` (314 KB) | LOW | This is a commodity reproduction of FanGraphs Stuff+; keep but don't pitch as edge. |
| volatility_surface | Options-style implied vol surface: Shannon entropy of outcomes across 26 zones x 12 counts. | yes (`volatility_surface.py`) | yes (`test_volatility_surface.py`) | none | HIGH | Validate that low-entropy pitchers underperform xFIP (predictability penalty). H1 day. |
| win_probability | Standard 24-state run-expectancy matrix + sigmoid-of-score-diff WPA calculator. | none directly (used by `live_game.py`) | none | none (closed-form) | LOW | Commodity — used as infrastructure. Leave alone. |

## Three to Promote Next

1. **chemnet** — already trained (`chemnet_v1.pt`), dashboarded, and tested. The lineup-synergy / protection-attention output is genuinely not on FanGraphs or Baseball Savant. Promotion cost: a single validation pass (synergy score predicts next-game team wOBA out-of-sample). Highest reward-to-effort ratio in the audit.
2. **volatility_surface** — Shannon-entropy zone x count surface is unique, has tests, and a dashboard view. Validation is one notebook: do low-entropy pitchers get hit harder than their stuff predicts? If yes, this is a publishable edge metric and a clean differentiator from Stuff+.
3. **defensive_pressing** — DPI is the only team-defense metric in the platform and methodology (HistGB expected outs vs actual) is sound. Validating against DRS/OAA is straightforward and gives the platform a defensive answer to complement CausalWAR's hitting/pitching focus.

## Dead Weight to Delete

- **stuff_model** — this is Stuff+, which FanGraphs already publishes for free. It's not edge, the methodology is conventional HistGB, and keeping it positioned as a flagship dilutes the "see what others don't" framing. Recommend: keep it as an internal feature for other models (it feeds nothing currently, which is itself a smell), but remove `stuff_plus.py` from the user-visible dashboard nav.
- **win_probability** — pure commodity sigmoid-of-score-diff. Keep as infrastructure for `live_game.py`, but it's not a "model" worth surfacing as such.
- **pitch_sequencing** — heavily overlaps with PitchGPT (sequence learning) and alpha_decay (sequence effectiveness over time). Has no tests and no clear edge story versus those two. Either retire or repurpose as the descriptive/visual layer for the two model-driven sequencing pages.

Nothing else looks like vaporware — the unvalidated set is mostly real, working code that has just never been put through validation. That is the platform's true bottleneck, not module count.

## Capability Gaps

A top-tier MLB analytics platform would have, and we are missing:

- **Pitch framing / catcher receiving model.** Zero coverage. Statcast publishes Catcher Framing Runs; we have nothing that scores catcher receiving or stealing strikes, despite having the pitch-by-pitch zone and called-strike data to do it.
- **Baserunning / fielding tracking integration (Statcast Sprint Speed, OAA).** `baserunner_gravity` proxies threat from SB attempts, but the platform never pulls Statcast running splits or OAA. This is a pulled punch.
- **Park / weather / batted-ball xstats adjustment layer.** No park factors, no temperature/wind adjustment, no spray-chart-aware xwOBA correction. Many of the existing models would tighten significantly with park-adjusted residuals as input.
- **Plate discipline / swing-decision model.** Nothing scores batter swing decisions independently of outcome (the public-analytics frontier — see SEAGER, Robert Orr's work). With pitch-level zone + count + result data, we have the inputs; we just haven't built it.
- **Projection system.** No Steamer/ZiPS-style multi-year projection that combines our descriptives into next-season forecasts. This is the form publishers (FG / BR) sell as their headline product, and would be a natural roof for CausalWAR + chemnet + Allostatic Load to feed into.
