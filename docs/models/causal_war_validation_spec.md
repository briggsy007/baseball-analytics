# CausalWAR Validation Spec

**Source:** `src/analytics/causal_war.py` (`class CausalWARModel` at line 89; config `class CausalWARConfig` at line 62).

## A. Current state

**Architecture.** Double Machine Learning (DML) with Frisch-Waugh-Lovell residualization. Nuisance models are `HistGradientBoostingRegressor` (300 estimators, `max_depth=6`, `l2_regularization=1.0`) that predict `E[Y|W]`, where `Y = woba_value` and `W = {venue, platoon, runners, outs, inning_bucket, month, handedness}`. Per-player residual averaging gives point estimates. Bootstrap (100 iterations, 5-fold cross-fit CV) provides 95% confidence intervals.

**Implemented.**
- PA-level data extraction query (`SUM(COALESCE(p.woba_value, 0))` at line 529).
- Feature engineering: 12 confounders with NaN imputation.
- Cross-fitted DML pipeline.
- Per-player residual averaging.
- Bootstrap CI construction.
- Leaderboard with rank/filter.
- Dashboard views.

**Tested.** Data extraction correctness, feature engineering robustness, DML fitting on small synthetic data, output schema, leaderboard rank/filter, edge cases.

**Not implemented.**
- Temporal train/test split (single-season only).
- FanGraphs / Baseball-Reference WAR baseline comparisons.
- Ablation (naive OLS vs partial DML vs full DML).
- Bootstrap CI coverage validation.
- Feature importance (SHAP / permutation).
- Residual diagnostics (QQ, orthogonality).
- Reproducibility script.
- Results document.
- The model saves no artifacts to `models/causal_war/`.

## B. Validation tickets

### Ticket 1. Temporal train/test split (M)
- **Goal.** Split 2015-2022 train, 2023-2024 test. No in-sample leakage.
- **Artifacts.** New `train_test_split()` method in `src/analytics/causal_war.py` returning `(train_df, test_df, train_metrics, test_metrics)`.
- **Success.** Test R² reported alongside train R²; CI coverage check passes on held-out season.
- **Effort.** M.

### Ticket 2. Baseline comparison vs FanGraphs / B-Ref WAR (M)
- **Goal.** Validate the CausalWAR point estimates against traditional WAR.
- **Artifacts.** `scripts/baseline_comparison.py`; output CSV with Pearson r, Spearman rho, RMSE, MAE; plus a leaderboard of the biggest over- and under-valued players.
- **Success.** Pearson r ≥ 0.5 AND Spearman rho_rank ≥ 0.6.
- **Effort.** M.

### Ticket 3. Ablation study (M)
- **Goal.** Show the DML machinery earns its complexity.
- **Artifacts.** Helper methods for naive-OLS and partial-DML variants in `causal_war.py`; `scripts/ablation_study.py` producing a comparison table.
- **Success.** Full DML has lower bias than naive OLS on synthetic ground-truth data; the Frisch-Waugh residualization step is material (>10% bias reduction vs partial DML).
- **Effort.** M.

### Ticket 4. Bootstrap CI coverage (S-M)
- **Goal.** Empirically validate the 95% CIs.
- **Artifacts.** `scripts/ci_coverage_test.py` using synthetic data with known effects.
- **Success.** Overall coverage in [93%, 97%]; coverage > 90% even for sparse players (low PA counts).
- **Effort.** S-M.

### Ticket 5. Feature importance (S)
- **Goal.** Explain what drives the nuisance predictions.
- **Artifacts.** `get_confounder_importance()` method extracting `feature_importances_` from the HistGBR nuisance models; `scripts/feature_importance.py` producing a Plotly bar chart.
- **Success.** Importances exported to `results/causal_war_feature_importance.csv` and visualized.
- **Effort.** S.

### Ticket 6. Residual diagnostics (M)
- **Goal.** Verify DML assumptions hold on real data.
- **Artifacts.** `scripts/residual_diagnostics.py`; `results/causal_war_diagnostics/` containing QQ plot, residual-vs-fitted plot, `E[resid | confounder_j]` orthogonality checks, Durbin-Watson statistic.
- **Success.** Residuals approximately normal; orthogonality holds per-confounder; no autocorrelation red flags.
- **Effort.** M.

### Ticket 7. Reproducibility script (M)
- **Goal.** One-command retrain with full determinism.
- **Artifacts.** `scripts/train_causal_war.py` with flags `--season`, `--train-split`, `--test-split`, `--n-bootstrap`, `--output-dir`. Saves model binary + metadata JSON + results CSV. Seed-deterministic across numpy, sklearn, and the bootstrap RNG.
- **Success.** Two identical invocations produce byte-identical output CSVs.
- **Effort.** M.

### Ticket 8. Results report (M)
- **Goal.** Publishable methodology write-up.
- **Artifacts.** `docs/models/causal_war_results.md`, ~1500-2000 words. Sections: methodology, train/test metrics, baseline comparison, ablation table, feature importance, residual diagnostics, biggest movers vs traditional WAR, limitations.
- **Success.** Self-contained; a reviewer can evaluate the model without reading source code.
- **Effort.** M.

### Ticket 9. (Optional) Hyperparameter grid search (L)
- **Goal.** Justify the 300 / 6 / 1.0 defaults.
- **Artifacts.** Grid over `n_estimators × learning_rate × l2_regularization` trained on 2015-2022, evaluated on 2023-2024. Results table.
- **Success.** Best config identified OR defaults confirmed near-optimal; train-test gap < 10% to rule out overfit.
- **Effort.** L.

### Ticket 10. (Optional) Full two-nuisance DML (L)
- **Goal.** Close the Neyman orthogonality gap.
- **Artifacts.** Explicit treatment residualization `T | W` added to the pipeline; orthogonality check comparing current (Y | W only) and full (Y | W and T | W) estimators.
- **Success.** Either the current approximation is shown to be within tolerance, or the full estimator replaces it with documented effect-size change.
- **Effort.** L.

## C. Award-readiness checklist

- Rigorous DML with Neyman orthogonality, cross-fitting, and bootstrap CIs.
- Temporal holdout evaluation rules out leakage.
- Spearman rho_rank ≥ 0.6 vs traditional WAR.
- Validated CI coverage in [93%, 97%].
- Deterministic retrain script and published code.

**Disqualifiers.** Data leakage via in-season testing. No baseline comparison. Unvalidated CIs. Unjustified hyperparameters. Opaque confounder selection. Missing limitations discussion.

## D. Risk flags

- **Incomplete treatment residualization** — currently `Y | W` only, not `T | W`. Violates full Neyman orthogonality; approximate but not formally correct.
- **Hardcoded hyperparameters** without cross-validation justification.
- **Bootstrap resampling correctness** requires empirical coverage validation (Ticket 4) before any CI claim can be published.
- **Only 12 confounders** — missing pitcher quality, batter form, team strength, umpire identity, time-of-season drift.
- **No temporal split currently** → CIs are optimistic.
- **NULL `woba_value` filled with 0** (line 581, `df["woba_value"].fillna(0)`). If more than 5% of PAs are NULL (walks, errors, HBP), the estimator is biased downward. Impute with league average instead.
- **Venue label-encoded via `pd.factorize()`** (line 589). Not stable across CV folds. Use one-hot encoding or a fixed ordinal mapping saved with the model.

---

Status: Specified 2026-04-16. Phase 2 pending. First ticket: #1 (temporal split).
