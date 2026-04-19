# MechanixAE Validation Spec

**Source:** `src/analytics/mechanix_ae.py`. VAE network: `class MechanixVAE` (line 181). Lifecycle wrapper: `class MechanixAEModel` (line 955). MDI scoring: `calculate_mdi()` (line 648). Dashboard consumption: `src/dashboard/views/mechanix_ae.py`.

**Status:** Reframed 2026-04-18 after the formal pivot documented in `docs/models/mechanix_ae_results.md` (Sections 1 and 2 — both supervised injury-validation attempts returned AUC < 0.50). MechanixAE is no longer positioned as an injury early-warning system. The gates below validate its **demonstrated** capability: per-pitcher mechanical profiling for the dashboard.

## A. Current state

**Input features (10D).** `release_pos_x`, `release_pos_z`, `release_extension`, `release_speed`, `release_spin_rate`, `spin_axis`, `pfx_x`, `pfx_z`, `effective_speed`, `arm_angle` (derived as `atan2(release_z - 5.5, release_x)` in degrees).

**Normalization.** Per-pitcher × pitch-type centering plus column standard-deviation scaling (builds a pitcher-specific mechanical baseline). IL dates are excluded 30 days pre-stint from training data.

**VAE architecture.**
- Encoder: `Conv1d(10 → 32, k=3)` → `Conv1d(32 → 64, k=3)` → flatten → linear to 6D latent.
- Decoder: symmetric.
- Window size: 20 pitches.
- Beta-VAE weighting: β = 0.1.
- Loss: MSE reconstruction + 0.1 × KL divergence.
- Healthy-baseline reconstruction `mean` and `std` persisted to checkpoint for z-score scoring.

**Drift scoring (MDI — Mechanical Deviation Index).** Per-window reconstruction error, exposed in two modes:
- `percentile`: percentile rank (0–100) of latest window vs the pitcher's recent history (legacy default for the dashboard gauge).
- `zscore`: `(error − healthy_mean) / max(healthy_std, eps)` against the pitcher's stored healthy baseline.
CUSUM changepoint detection (`detect_changepoints`, threshold 5.0) flags abrupt mechanical shifts. `calculate_drift_velocity` reports the rolling-window slope.

**Dashboard surfaces** (`src/dashboard/views/mechanix_ae.py`):
- MDI gauge with four colour zones (Stable / Minor / Moderate / High Drift).
- Drift timeline with changepoint markers.
- 2-D latent-space trajectory (first two latent dims).
- Feature-attribution bar chart: latest-window vs average per-feature reconstruction error.
- Alert tab listing pitchers above the 80th percentile of cohort MDI.

## B. Reframing rationale (read this before B1–B5)

Two prior validation rounds — percentile-rank MDI and z-score MDI — measured discrimination of arm-injury labels on the 2015–2016 cohort and returned AUC = 0.379 and 0.389 respectively (both < random; see `docs/models/mechanix_ae_results.md` Sections 1 and 2). The prior spec encoded injury-detection gates (AUC ≥ 0.65, lead-time ≥ 10 days, FPR ≤ 20%) that the model demonstrably cannot meet. Faithfully running those gates produces a misleading "HARD FAIL" verdict — the model is not broken, it is being asked the wrong question.

This spec validates what MechanixAE actually delivers to the dashboard: **per-pitcher mechanical profiling** — does the VAE learn each pitcher's signature, does the MDI score behave as a usable deviation measurement, and does coverage extend to the qualified-pitcher universe? All injury-discrimination language (AUC, FPR, lead-time, PIVOT band) is removed.

## C. Validation gates (descriptive profiling)

Gates apply to the cohort of pitchers with per-pitcher checkpoints under `models/mechanix_ae/per_pitcher/*.pt` (currently 111). Source-of-truth artifacts and thresholds below are what the `validate-model` skill should compare against.

### Gate 1. Per-pitcher VAE reconstruction fit
- **Threshold.** ≥ 80% of per-pitcher checkpoints achieve `final_recon_loss ≤ 0.50` (mean MSE on the standardized 10-D feature space, where unit-variance features are clipped to [−10, 10] — see `_prepare_training_windows`).
- **Source.** Per-pitcher training history persisted via `train_mechanix_ae` return dict (`final_recon_loss` field). For the existing 111-pitcher cohort, derive from `models/mechanix_ae/per_pitcher/<pid>.pt` by recomputing reconstruction error on each pitcher's own healthy windows (no retrain — load checkpoint, run forward pass, take MSE).
- **Output artifact.** `results/validate_mechanix_ae_<ts>/per_pitcher_fit.csv` columns `pitcher_id, n_windows, recon_mse, pass`.

### Gate 2. MDI distribution well-formedness
- **Threshold.** ≥ 90% of per-pitcher MDI distributions (computed in `zscore` mode over each pitcher's full available history) satisfy all of: `std(MDI) > 0.10` (non-degenerate); `|skew(MDI)| ≤ 5` (no pathological tail); at least one window with `|z| ≥ 1` AND at least one window with `|z| ≤ 0.5` (range exercised).
- **Source.** `calculate_mdi(..., score_mode="zscore")` over each pitcher's full pitch history. Per-pitcher z-score arrays already returned in the result dict's `z_scores` field.
- **Output artifact.** `results/validate_mechanix_ae_<ts>/mdi_distribution.csv` columns `pitcher_id, std, skew, frac_high_z, frac_low_z, pass`.

### Gate 3. Coverage of qualified pitchers
- **Threshold.** ≥ 75% of pitchers with ≥ 200 healthy pitches (the `min_pitches=200` qualification used by `batch_calculate`) have a usable per-pitcher checkpoint at `models/mechanix_ae/per_pitcher/<pitcher_id>.pt` that loads without error and produces a non-`None` MDI.
- **Source.** Cross-reference `SELECT pitcher_id FROM pitches GROUP BY pitcher_id HAVING COUNT(*) >= 200` against the directory listing of `models/mechanix_ae/per_pitcher/`. Load each checkpoint via `_load_model(pitcher_id)` and call `calculate_mdi`; record load/inference success.
- **Output artifact.** `results/validate_mechanix_ae_<ts>/coverage.json` with fields `n_qualified, n_with_checkpoint, n_loadable, n_produces_mdi, coverage_pct`.

### Gate 4. Intra-pitcher MDI stability (same-day reliability)
- **Threshold.** Pearson correlation between MDI values of overlapping adjacent windows within the same `game_pk` ≥ 0.70 in the median across pitchers with ≥ 2 multi-window starts. Operationalises "the score doesn't oscillate randomly between two windows captured 5 pitches apart."
- **Source.** For each pitcher, group windows by `game_pk`, compute MDI for every window in the start, correlate `MDI[t]` with `MDI[t+1]` across all consecutive within-game window pairs.
- **Output artifact.** `results/validate_mechanix_ae_<ts>/intra_pitcher_stability.csv` columns `pitcher_id, n_pairs, pearson_r, pass`. Cohort summary in `results/validate_mechanix_ae_<ts>/stability_summary.json`.

### Gate 5. Feature-attribution interpretability proxy
- **Threshold.** For the top-10% highest-MDI windows per pitcher, the dominant attributed feature (largest positive entry of `latest_errors − avg_errors` from `_render_feature_attribution`) is one of `{release_pos_x, release_pos_z, arm_angle, release_extension, spin_axis}` (the five mechanically-meaningful inputs) in ≥ 60% of cases. This is a sanity check that high-MDI windows surface mechanically-plausible drivers, not random feature noise.
- **Source.** Replicate the per-feature MSE computation from `src/dashboard/views/mechanix_ae.py::_render_feature_attribution` (lines 442–501) over each pitcher's window series; tabulate which feature wins on the top-decile windows.
- **Output artifact.** `results/validate_mechanix_ae_<ts>/feature_attribution_top_decile.csv` columns `pitcher_id, window_idx, mdi, top_feature, in_mechanical_set`. Cohort summary in `results/validate_mechanix_ae_<ts>/attribution_summary.json` with `cohort_pct_in_mechanical_set`.

## D. Award-readiness checklist (descriptive-profiling framing)

- All 5 gates pass on the 111-pitcher cohort.
- Dashboard view (`src/dashboard/views/mechanix_ae.py`) renders without error for any pitcher meeting Gate 3 coverage.
- Reproducibility: `np.random.seed(42)`, `torch.manual_seed(42)` set at training start.
- Documented limitations include the prior injury-discrimination failure (Sections 1 and 2 of the results doc) so reviewers see the full evidence trail.

**No PIVOT band.** No injury-discrimination claim is made; the model is positioned as a descriptive mechanical-profiling tool only. There is therefore no "AUC < 0.58 ⇒ pivot" branch in the verdict logic — gates are pass/fail against the descriptive thresholds above.

## E. Risk flags

- **FLAG A.** Per-pitcher data sparsity — pitchers with < 200 healthy pitches are excluded from Gate 3's denominator. Coverage figures should be reported alongside the cohort size, not as a bare percentage.
- **FLAG B.** Pitch-type entanglement persists in the latent space (root-cause #3 in the results doc). For descriptive profiling this is acceptable — feature attribution still surfaces real mechanical changes — but reviewers should be told the latent space mixes pitch types.
- **FLAG C.** No medical / coaching ground truth for "is this drift real?" Gate 5 uses a structural proxy (mechanical-feature membership) rather than ground-truth attribution.
- **FLAG D.** MDI ≥ 80 alert tab in the dashboard is a *cohort-relative ranking* (see `_render_alert_system`, threshold = `valid["mdi"].quantile(0.80)`), not an injury risk score. Wording in the dashboard view should reflect this — outside the scope of this spec.
- **FLAG E.** Checkpoint freshness — per-pitcher VAEs need refit when a pitcher materially adjusts their delivery. Not gated here; tracked operationally.

## F. Historical context

For the prior injury-detection framing and the supervised-validation evidence that motivated this reframe, see `docs/models/mechanix_ae_results.md`:
- Section 1 — Percentile-rank MDI: AUC = 0.379, FPR = 100%.
- Section 2 — Z-score MDI rescue attempt: AUC = 0.389, FPR = 96–100%.
- "Spec reframe 2026-04-18" — the change-of-framing narrative.

---

Status: Reframed 2026-04-18 (descriptive-profiling gates). Prior injury-detection spec (AUC / lead-time / FPR / PIVOT band) is superseded.
