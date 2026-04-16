# MechanixAE Validation Spec

**Source:** `src/analytics/mechanix_ae.py`. VAE network: `class MechanixVAE` (line 179). Lifecycle wrapper: `class MechanixAEModel` (line 878). IL-date ingestion helper: `_fetch_il_dates()` (line 338).

## A. Current state

**Input features (10D).** `release_pos_x`, `release_pos_z`, `release_extension`, `release_speed`, `release_spin_rate`, `spin_axis`, `pfx_x`, `pfx_z`, `effective_speed`, `arm_angle` (derived as `atan2(release_z - 5.5, release_x)` in degrees).

**Normalization.** Per-pitcher × pitch-type centering plus column standard-deviation scaling. This is the correct approach — it builds a pitcher-specific mechanical baseline. IL dates are excluded 30 days pre-stint from the training data.

**VAE architecture.**
- Encoder: `Conv1d(10 → 32, k=3)` → `Conv1d(32 → 64, k=3)` → flatten → linear to 6D latent.
- Decoder: symmetric.
- Window size: 20 pitches.
- Beta-VAE weighting: β = 0.1.
- Loss: MSE reconstruction + 0.1 × KL divergence.

**Drift scoring (MDI — Mechanical Drift Index).** Per-window reconstruction error → percentile rank (0-100) versus recent history. CUSUM changepoint detection on the percentile series, threshold = 5.0.

**Tested.** Normalization centering, VAE forward pass, reconstruction error on constant input, KL decay, MDI range, drift-velocity slope, changepoint detection on synthetic steps, edge cases.

**Not tested — existential gap.** Injury-label validation. **The model is fully unsupervised.** There is no comparison of MDI to actual IL stints, no ROC, no AUC, no baseline-vs-velocity-drop analysis, no per-pitch-type conditioning (the VAE may conflate pitch-type variance with injury drift), no reproducibility seeding, no false-positive-rate measurement on healthy seasons.

**Label data status.** The `transactions` table has a free-text `description` field. `_fetch_il_dates()` (line 338) reads it, but the output is used only to *exclude* training windows — never for supervised validation. No `data/injury_labels.parquet` exists.

## B. Validation tickets

### Ticket 1. Per-pitcher training default (S)
- **Goal.** Confirm production uses per-pitcher models, not a universal model.
- **Artifacts.** Config check in `MechanixAEModel` init path. Flip default to per-pitcher for all pitchers with ≥ 200 pitches if currently universal.
- **Success.** Every qualifying pitcher has a per-pitcher checkpoint; universal fallback only used below the threshold.
- **Effort.** S.

### Ticket 2. Injury label ingestion (M) — CRITICAL UNBLOCKER
- **Goal.** Build the supervised label set that unblocks every downstream ticket.
- **Artifacts.** Query `transactions` table 2015-2024. Classify injury type: TJ, UCL, shoulder, elbow, rotator cuff, labrum. Exclude non-arm injuries. Cross-check against Chadwick Bureau and Retrosheet. Save to `data/injury_labels.parquet`. Manual audit of ~50 records to validate the text-parsing regexes.
- **Success.** Parquet file exists with `pitcher_id`, `il_start_date`, `injury_type`, `source` columns. Manual audit agreement ≥ 90%.
- **Effort.** M.

### Ticket 3. Lead-time analysis (M)
- **Goal.** Measure how early MDI detects drift before an IL stint.
- **Artifacts.** For each injured pitcher: backfill 90 days pre-IL of pitch data, train VAE on healthy windows only, compute MDI daily, log the first threshold breach at the 60th / 70th / 80th percentile. Output histogram of lead-times to `results/mechanix_ae_lead_times.png`.
- **Success.** Decision gate — median lead-time < 5 days = model fires too late, > 30 days = potentially useful. Report all three threshold curves.
- **Effort.** M.

### Ticket 4. ROC / AUC (M)
- **Goal.** Headline predictive metric.
- **Artifacts.** Positive class = 30-day pre-IL windows. Negative class = matched healthy pitcher-seasons. Threshold sweep → ROC curve.
- **Success.** Award-worthy if AUC > 0.65. Random baseline ≈ 0.55 given class imbalance.
- **Effort.** M.

### Ticket 5. Velocity-drop baseline (M)
- **Goal.** Show the VAE beats a trivial heuristic.
- **Artifacts.** Rolling 10-pitch mean velocity; flag on -2σ deviation. Same window definition and labels as Ticket 4. Compute AUC.
- **Success.** `delta_AUC = AUC(MDI) - AUC(VelDrop) ≥ 0.10`. This is the clinically-meaningful-improvement threshold.
- **Effort.** M.

### Ticket 6. False-positive rate on healthy seasons (S)
- **Goal.** Usability check.
- **Artifacts.** Compute percentage of healthy pitcher-seasons that exceed MDI ≥ 80, ≥ 70, and ≥ 60 at any point.
- **Success.** FPR at MDI ≥ 80 ≤ 20%. If > 50% at MDI = 80, the model is unusable regardless of AUC.
- **Effort.** S.

### Ticket 7. CUSUM threshold tuning (S)
- **Goal.** Optimize the changepoint operating point.
- **Artifacts.** Sweep threshold ∈ {1, 3, 5, 10, 20} on a mixed injured + healthy cohort.
- **Success.** Identify threshold that maximizes TPR subject to FPR < 20%.
- **Effort.** S.

### Ticket 8. Pitch-type conditioning (M)
- **Goal.** Rule out the null hypothesis that MDI is a pitch-mix detector, not an injury detector.
- **Artifacts.** Check latent-space separation by pitch type (PCA / UMAP of 6D latent, colored by type). If the first 2 latent dims cluster by type, add type conditioning — either embed the type and concatenate with the latent, or condition the Conv layers.
- **Success.** Retrained model shows within-type reconstruction variance drops without increasing between-type confusion.
- **Effort.** M.

### Ticket 9. Reproducibility seeding (S)
- **Goal.** Deterministic retrain.
- **Artifacts.** Add `np.random.seed(42)` and `torch.manual_seed(42)` at training start.
- **Success.** Two independent runs produce identical checkpoint weights (max abs diff < 1e-6).
- **Effort.** S.

### Ticket 10. Results report (M)
- **Goal.** Publishable validation document.
- **Artifacts.** `docs/models/mechanix_ae_results.md`. Sections: lead-time distribution, ROC curves (MDI vs VelDrop), false-positive rate, CUSUM operating point, case studies (Jacob deGrom 2021 pre-injury as flagship positive case, one high-MDI healthy pitcher as calibration anchor, one documented false positive), limitations.
- **Success.** Self-contained and submissible.
- **Effort.** M.

## C. Award-readiness checklist

- Lead-time median ≥ 10 days.
- AUC(MDI) ≥ 0.65.
- delta_AUC ≥ 0.10 over the velocity-drop baseline.
- False-positive rate at MDI ≥ 80 ≤ 20%.
- Sample size ≥ 50 arm injuries with 90+ days of pre-IL pitch data.
- Deterministic reproducibility.

**Peer-review attack surface.**
- Small n of injuries.
- Confounders: workload, rest, age.
- Lookback bias in training-window construction.
- Overfit to high-profile cases (deGrom, Ohtani).
- Survivor bias — pitchers detected early and rested never hit IL, appearing as false positives in the evaluation set.

## D. Risk flags

- **FLAG 1 (existential).** Model is fully unsupervised. There is no evidence MDI predicts injury. Tickets 2-4 must complete before ANY award submission is drafted.
- **FLAG 2.** Per-pitcher data sparsity. Some pitchers have fewer than 50 pitches in window. Enforce a minimum-pitch threshold and report coverage.
- **FLAG 3.** Pitch-type entanglement. The 10D feature space mixes fastballs and curveballs; per-pitcher × pitch-type normalization may not be sufficient. Ticket 8 addresses this.
- **FLAG 4.** IL-date precision. Free-text parsing is fragile — phantom IL placements, rehab assignments, and retroactive placements all contaminate the label set.
- **FLAG 5.** No medical ground truth. IL is an administrative construct, not a medical one. Inherent label noise is a ceiling on achievable AUC.
- **FLAG 6.** Model drift. Universal models go stale; per-pitcher models need automatic retrain on roughly a 60-day cadence.

**DECISION GATE.** If AUC(MDI) < 0.58 after Phase 2 validation, PIVOT the narrative from "injury early-warning" to "mechanical profiling." The unsupervised framing retains descriptive value (identifying mechanical outliers, benchmarking return-from-injury pitchers) even if the predictive framing fails. Do not pull the model from the platform; reframe its claimed utility.

---

Status: Specified 2026-04-16. Phase 2 pending. First ticket: #2 (injury label ingestion — the entire spec depends on this).
