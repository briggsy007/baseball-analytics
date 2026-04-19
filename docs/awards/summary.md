# Hunter Briggs: Flagship Baseball Analytics Portfolio

Welcome to the central hub for my flagship baseball machine learning models. 

The public market for baseball analytics is saturated with metrics that describe what happened. The edge for a modern front office lies in causal isolation, outcome-based residuals, and mathematically honest forecasting. Combining causal inference and machine learning with production-grade data engineering, this repository houses three pipelines designed to identify market mispricings and build calibrated foundations for game-theory strategy. 

The unifying philosophy across all three models is transparency: edge is found exactly where these models rigorously disagree with public consensus.

---

## ⚾ 1. CausalWAR: Exploiting the Value Inefficiency
Traditional WAR is an accounting ledger that easily gets fooled by park factors and sequencing noise. **CausalWAR** is a Double Machine Learning (DML) model trained on over 1.2 million plate appearances that strips away context (inning, stadium, base-state) to isolate pure, causal player skill.

* **The Edge:** The model automatically generates "Contrarian Leaderboards" that highlight strict disagreements with traditional bWAR. It identifies **Buy-Low** targets (e.g., elite relievers penalized by traditional leverage formulas) and **Over-Valued** players (e.g., starters buoyed by pitcher-friendly parks).
* **The Proof:** Over a three-year out-of-sample window, players flagged on these contrarian leaderboards regress in the predicted direction **70% to 80%** of the time the following season. It serves as a data-backed trade deadline and off-season shopping list.

## 🧤 2. Defensive Pressing Intensity (DPI): The System Illusion
Public defensive metrics like Outs Above Average (OAA) measure individual athleticism and range. They explicitly penalize teams that position their players perfectly, creating a blind spot where highly efficient defensive *systems* are graded poorly. **DPI** measures outcome suppression.

* **The Edge:** DPI uses a league-trained expected-out classifier to determine if a defense actually converted a batted ball into an out, regardless of how far the fielder had to run. It isolates teams that look unathletic on paper but secretly suffocate hits, and teams that make flashy plays but bleed runs overall.
* **The Proof:** DPI significantly outperforms OAA in predicting a team’s BABIP-against (hits allowed on balls in play) for the following season. It is the missing half of the defensive evaluation equation.

## 🤖 3. PitchGPT: The Calibrated Sequence Engine
Public pitch prediction models often chase raw accuracy, generating overconfident forecasts that poison downstream strategy simulators. **PitchGPT** is a decoder-only Transformer model that predicts the next pitch sequence (pitch type, zone, and velocity) not with a crystal ball, but with perfect mathematical calibration.

* **The Edge:** PitchGPT is honest about what it knows. If it predicts a 3.4% chance of a slider low-and-away, that exact scenario happens exactly 3.4% of the time. 
* **The Proof:** Validated on a strict, out-of-sample holdout of pitchers the model had never seen, PitchGPT achieves near-perfect Expected Calibration Error (ECE = 0.0098). While it is not designed as a standalone dugout tool to guess the next pitch, it is the mathematically sound sequence engine required to power advanced batter-preparation simulations and pitcher workload-management systems.

---

### Technical Infrastructure
All models in this repository are designed for scale and reproducibility:
* **Data Pipelines:** Automated ingestion of Statcast and FanGraphs data into DuckDB.
* **Machine Learning:** PyTorch (Transformers), scikit-learn (HistGradientBoosting), and Chernozhukov-style DML integrations.
* **Validation:** Strict, temporally-disjoint holdout testing with bootstrap confidence intervals and documented edge-case autopsies.

---

Would you like me to generate a secondary Markdown file that organizes the specific commands needed to run your validation scripts, so anyone cloning the repo has a quick-start guide?
