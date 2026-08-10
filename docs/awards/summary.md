# Hunter Briggs: Flagship Baseball Analytics Portfolio

Welcome to the central hub for my flagship baseball machine learning models. 

The public market for baseball analytics is saturated with metrics that describe what happened. The edge for a modern front office lies in causal isolation, outcome-based residuals, and mathematically honest forecasting. Combining causal inference and machine learning with production-grade data engineering, this repository houses three pipelines designed to identify market mispricings and build calibrated foundations for game-theory strategy. 

The unifying philosophy across all three models is transparency: edge is found exactly where these models rigorously disagree with public consensus.

---

## ⚾ 1. CausalWAR: Exploiting the Value Inefficiency
Traditional WAR is an accounting ledger that easily gets fooled by park factors and sequencing noise. **CausalWAR** is a DML-inspired model (a one-nuisance approximation, not full residual-on-residual DML) trained on over 1.2 million plate appearances that residualizes observable context (inning, base-state, platoon) to isolate per-PA offensive skill.

* **The Edge:** The model automatically generates "Contrarian Leaderboards" that highlight strict disagreements with traditional bWAR. It identifies **Buy-Low** targets (short-sample / late-callup relievers whose small-IP bWAR is artificially depressed by leverage weighting) and **Over-Valued** players (starters buoyed by pitcher-friendly parks, fielders credited by favorable defensive positioning).
* **The Proof (bounded):** The headline Buy-Low hit rate is 68.4% (13/19) under a criterion defined post-hoc; its 95% CI [0.474, 0.843] includes chance, intention-to-treat scoring is 13/25 = 52%, and matched-naive mean-reversion baselines score 66.5–73% on the same pools — at n=19 the rate is statistically indistinguishable from the naive base rate. The defensible, base-rate-cleared result is the reliever-tag subset (78.1% vs 56.9% within-filter naive, n=32), plus two fully-OOS windows (−2.8pp / +10.8pp vs naive). Treat the boards as a divergence-surfacing shopping list, not a validated edge.

## 🧤 2. Defensive Pressing Intensity (DPI): The System Illusion
Public defensive metrics like Outs Above Average (OAA) measure individual athleticism and range. They explicitly penalize teams that position their players perfectly, creating a blind spot where highly efficient defensive *systems* are graded poorly. **DPI** measures outcome suppression.

* **The Edge:** DPI uses a league-trained expected-out classifier to determine if a defense actually converted a batted ball into an out, regardless of how far the fielder had to run. It isolates teams that look unathletic on paper but secretly suffocate hits, and teams that make flashy plays but bleed runs overall.
* **The Proof:** DPI adds distinct information to public defensive metrics like OAA — particularly on next-year BABIP-against, where DPI's advantage over OAA is statistically significant (CI excludes zero). It is the missing half of the defensive evaluation equation — a measurement tool rather than a standalone forecaster, best used alongside prior-year run-prevention totals.

## 🤖 3. PitchGPT: The Calibrated Sequence Engine
Public pitch prediction models often chase raw accuracy, generating overconfident forecasts that poison downstream strategy simulators. **PitchGPT** is a decoder-only Transformer model that predicts the next pitch sequence (pitch type, zone, and velocity) with measured, bounded calibration rather than raw-accuracy bravado.

* **The Edge:** PitchGPT reports honest probabilities. Its measured top-1 expected calibration error is low (ECE 0.0090–0.0114 post-temperature across evaluations) — though note that a 0.10 ECE gate for a 2,210-class predictor is easy to pass, so this is evidence of calibration, not proof of it.
* **The Proof (scoped):** Evaluated on a strict, out-of-sample holdout of pitchers the model had never seen (334 pitchers new to 2025, zero leakage), PitchGPT scores top-1 ECE = 0.0098 post-temperature. That number describes the pre-class-calibration per-pitch stack; the production-path ECE (after the class-calibration corrections production applies) has not been measured, and PA-level rollout marginals (K%/BB%/HR%) currently FAIL their fidelity gates. It is a per-pitch-calibrated sequence engine and a foundation under active validation — not yet a validated simulator for absolute PA-level rates.

---

### What Didn't Make It

Sixteen models were built; three passed the validation gauntlet. That ratio is the credibility signal — validated negatives matter as much as the wins.

* **VWR (Viscoelastic Workload Model):** Promoted to flagship on a 64-pitcher 2015–2016 validation showing residual AUC 0.768. Scale-verified out-of-sample (n=563) collapsed to 0.438, below chance. Retracted the same day the larger run finished. The small-sample finding was a label-calendar artifact.
* **MechanixAE:** A β-VAE over pitching biomechanics intended as an injury early-warning signal. First ROC analysis returned AUC 0.387 — below random. Demoted to descriptive pitcher-profiling; kept behind an experimental dashboard banner.
* **ChemNet (v1 + v2):** Graph neural network for lineup synergy / team chemistry. r = 0.089 (v1) and 0.155 (v2 with opposing-pitcher node + residual objective). Retired — the architecture does not learn this target at this scale.
* **Volatility Surface:** Tested a predictability-tax hypothesis. Clean null (r = -0.013, p = 0.89). Kept as descriptive visualization only.

Each documented in full with committed code, validation artifacts, and an explicit retraction / demotion rationale. An honest research portfolio shows both the successes and the stopped experiments.

---

### Technical Infrastructure
All models in this repository are designed for scale and reproducibility:
* **Data Pipelines:** Automated ingestion of Statcast and FanGraphs data into DuckDB.
* **Machine Learning:** PyTorch (Transformers), scikit-learn (HistGradientBoosting), and Chernozhukov-style DML integrations.
* **Validation:** Strict, temporally-disjoint holdout testing with bootstrap confidence intervals and documented edge-case autopsies.

---

### Further Reading

For full methodology, validation gates, and honest limitations of each flagship:

* [`methodology_paper_dpi.md`](methodology_paper_dpi.md) — Defensive Pressing Index: methodology, validation, AR(1) honest limitation, post-submission bulletproofing addendum.
* [`methodology_paper_causal_war.md`](methodology_paper_causal_war.md) — CausalWAR: Double Machine Learning player value, Contrarian Leaderboards edge product, mechanism ablation + base-rate study addenda.
* [`methodology_paper_pitchgpt.md`](methodology_paper_pitchgpt.md) — PitchGPT: calibrated transformer for pitch sequences, honest LSTM-gate shortfall, downstream utility test.

Primary validation and bulletproofing artifacts live under `results/` (committed); reproducibility drivers under `scripts/`.
