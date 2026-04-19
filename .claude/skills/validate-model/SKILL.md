---
name: validate-model
description: Run the full validation playbook for a flagship baseball model (CausalWAR, PitchGPT, VWR, Defensive Pressing, MechanixAE, or Allostatic Load). Use when the user asks to validate a model, check if a model passes its spec gates, run the validation suite, or invokes `/validate-model <name>`. Accepts model name as argument; `all` or `flagships` runs the four flagships sequentially.
---

# validate-model

You are executing the repo's canonical validation playbook for one of six baseball models. This skill orchestrates existing scripts, checks outputs against spec-defined gates, writes a JSON summary, and appends a verdict to the results doc. Do NOT write new Python, retrain models, or invent new thresholds — read the spec, run the script, compare, report.

## Step 1 — Dispatch on model argument

Accepted arguments (case-insensitive, accept hyphens/underscores):
- `causalwar` | `causal_war` | `causal-war` → `causal_war`
- `pitchgpt` | `pitch-gpt` → `pitchgpt`
- `mechanixae` | `mechanix_ae` | `mechanix-ae` → `mechanix_ae`
- `vwr` | `viscoelastic` | `viscoelastic_workload` | `viscoelastic-workload` → `viscoelastic_workload`
- `abl` | `allostatic` | `allostatic_load` | `allostatic-load` → `allostatic_load`
- `dpi` | `defensive_pressing` | `defensive-pressing` → `defensive_pressing`
- `all` | `flagships` → run `causal_war`, `pitchgpt`, `viscoelastic_workload`, `defensive_pressing` sequentially (the current flagship four; MechanixAE and Allostatic Load are validated but not flagship and are excluded from bulk runs)

If argument is missing or unrecognized: ask the user which model. Do not guess.

Normalize to one of `{causal_war, pitchgpt, mechanix_ae, viscoelastic_workload, allostatic_load, defensive_pressing}` — this is the `<model>` token used below.

## Step 2 — Load the spec and extract gates

Read `docs/models/<model>_validation_spec.md`. Extract the pass/fail thresholds below. These are the authoritative gates this skill checks — if the spec has been updated and thresholds drift, re-read and use the current values.

### CausalWAR gates (source: `docs/models/causal_war_validation_spec.md`)
- **Baseline comparison (Ticket 2):** `Pearson r ≥ 0.5` AND `Spearman rho_rank ≥ 0.6` vs traditional WAR proxy.
- **Leakage check (Ticket 1 / Award checklist):** no overlap in `game_pk` between train split (2015-2022) and test split (2023-2024); test R² reported alongside train R².
- **CI coverage (Ticket 4, award checklist):** overall coverage in `[93%, 97%]` (informational — only check if `results/ci_coverage*.json` exists; do not rerun).

### PitchGPT gates (source: `docs/models/pitchgpt_validation_spec.md`)
- **Leakage (Ticket 1):** no shared `game_pk` across train/val/test splits; **no pitcher in both train and test** (post-2026-04-18 fix: `PitchSequenceDataset` now enforces pitcher-disjoint splits).
- **Calibration (Ticket 7):** `ECE < 0.10` on the 2024 test split, both pre- and post-temperature.
- **Ablation (Ticket 6):** full model beats every ablation variant (`tokens_only`, `count_only`, `identity_only`) by `≥ 10%` on test perplexity.
- **Baselines (Tier 2, informational only — only check if metrics exist):** perplexity vs LSTM `≥ 15%`, vs Markov-2 `≥ 20%`, vs heuristic `≥ 25%`.

### MechanixAE gates (source: `docs/models/mechanix_ae_validation_spec.md` — descriptive-profiling reframe 2026-04-18; injury-EWS gates are defunct)
- **Per-pitcher VAE fit (Gate 1):** `pct_pitchers_recon_le_0_50 ≥ 0.80` on standardized features. Source: `per_pitcher_fit.csv`.
- **MDI distribution well-formedness (Gate 2):** `pct_well_formed ≥ 0.90` where well-formed = `std > 0.10` AND `|skew| ≤ 5` AND has at least one `|z| ≥ 1` and one `|z| ≤ 0.5`. Source: `mdi_distribution.csv`.
- **Coverage (Gate 3):** `coverage_pct ≥ 0.75` over pitchers with `≥ 200` healthy pitches. Source: `coverage.json`.
- **Intra-pitcher MDI stability (Gate 4):** `median_pearson_r ≥ 0.70` across consecutive within-game windows. Source: `intra_pitcher_stability.csv` / `stability_summary.json`.
- **Feature-attribution interpretability (Gate 5):** `cohort_pct_in_mechanical_set ≥ 0.60` for top-decile MDI windows where the mechanical set is `{release_pos_x, release_pos_z, arm_angle, release_extension, spin_axis}`. Source: `feature_attribution_top_decile.csv` / `attribution_summary.json`.
- No PIVOT band, no AUC, no lead-time, no FPR — those gates are defunct under the descriptive-profiling reframe.

### Viscoelastic Workload (VWR) gates (source: `docs/models/viscoelastic_workload_validation_spec.md` — flagship under seasonal-residual framing)
- **AUC seasonal-residual (HEADLINE) (Gate 1):** `≥ 0.65`. Source field: `roc.seasonal_control.residual_auc` in `summary.json`. The raw 30-day-pre-IL AUC is informational only — do not gate on it.
- **delta_AUC vs velocity-drop, residual space (Gate 2):** `≥ 0.10`. Source: `roc.seasonal_control.delta_residual_auc`.
- **Median lead-time at VWR ≥ 85 (Gate 3):** `≥ 30 days`. Source: `lead_time.lead_time_t85.median_days`.
- **FPR at VWR ≥ 85 on healthy seasons (Gate 4):** `pct_breach_85 ≤ 30%`. Source: `fpr.pct_breach_85`.
- **Seasonal-control sanity (Gate 5):** `residual_auc ≥ raw_auc` (residual must NOT collapse below raw — sanity that the signal is not a calendar artifact). Source: `roc.seasonal_control.residual_auc` vs `roc.vwr.auc`.

### Allostatic Load (ABL) gates (source: `docs/models/allostatic_load_validation_spec.md` — non-flagship; documented for completeness)
- **AUC at 30-day pre-IL window (Gate 1):** `≥ 0.60`. Source: `roc.abl.auc`.
- **Median lead-time at ABL ≥ 75 (Gate 2):** `≥ 14 days`. Source: `lead_time.lead_time_t75.median_days`.
- **FPR at ABL ≥ 75 on healthy seasons (Gate 3):** `pct_breach_75 ≤ 30%`. Source: `fpr.pct_breach_75`.
- **delta_AUC vs games-played baseline (Gate 4):** `≥ 0.05`. Source: `roc.delta_auc`.
- **Seasonal-control sanity (informational, NOT a hard gate):** `residual_auc ≥ 0.55` AND `(raw_auc - residual_auc) ≤ 0.08` (spec defines "seasonal artifact" as failing either condition). Source: `roc.seasonal_control.residual_auc`.
- **Verdict bands** (record in `notes`, not hard gate): FLAGSHIP / MARGINAL / NOT FLAGSHIP per the spec's verdict table.

### Defensive Pressing (DPI) gates (source: `docs/models/defensive_pressing_validation_spec.md` — flagship promoted 2026-04-18, PASS 5/5)
- **xOut model AUC (Gate 1):** `≥ 0.70` on test split. Source: `metrics.xout.test_auc` in `defensive_pressing_validation_metrics.json`.
- **DPI vs run-prevention proxy (Gate 2):** `Pearson r ≥ 0.40` against team-level `-SUM(delta_run_exp)` on defense (proxy; no DRS/OAA staged yet). Source: `metrics.consumer_claim.dpi_vs_rp_proxy.pearson_r`.
- **DPI vs BABIP-against (Gate 3):** `Pearson r ≤ -0.50` (negative direction — high DPI should suppress BABIP-against). Source: `metrics.consumer_claim.dpi_vs_babip_against.pearson_r`. NOTE: operator is `<=`, not `>=`.
- **YoY stability (Gate 4):** `Pearson r ≥ 0.30` between consecutive-season DPI per team. Source: `metrics.stability.yoy_pearson_r`.
- **Leakage (Gate 5):** `shared_game_pks_train_test == 0` between 2015-2022 train and 2023-2024 test. Source: `leakage_audit.shared_game_pks_train_test`.

## Step 3 — Prepare the run directory

```
results/validate_<model>_<UTC-timestamp>/
```

Use `date -u +%Y%m%dT%H%M%SZ` (or equivalent) for the timestamp. Create it at the start. All JSON summaries, verdict snippets, and copied artifacts land here. Record wall-clock per step using `time.monotonic()` in your shell or simple `date +%s` diffs.

## Step 4 — Run the validation steps

Work from the repo root `C:\Users\hunte\projects\baseball`. All scripts assume that cwd. Use `python scripts/<script>.py ...`. Stream output to a log file in the run directory (`step_<n>.log`). If a step fails (non-zero exit), record the failure in the summary JSON with the stderr tail, mark overall verdict FAIL, and continue to the remaining independent steps.

### 4a — CausalWAR

1. **Leakage check.** Before running anything, confirm the model file exists at `models/causal_war/causal_war_trainsplit_2015_2022.pkl`. The split is encoded in the filename — train = 2015-2022. The baseline script re-extracts PA-level data for train-start/end and test-start/end and performs its own in-memory split; confirm `--train-start 2015 --train-end 2022 --test-start 2023 --test-end 2024` in the invocation so no overlap is possible. Record `leakage_check: PASS` with the `game_pk`-disjointness justification (separate season ranges) in the summary JSON.

2. **Baseline comparison.** Invoke:
   ```
   python scripts/baseline_comparison.py \
     --train-start 2015 --train-end 2022 \
     --test-start 2023 --test-end 2024 \
     --output-dir results/validate_causal_war_<ts>/
   ```
   The script emits `causal_war_baseline_comparison_2023_2024_metrics.json` (Pearson r, Spearman rho, RMSE, MAE, bootstrap 95% CIs) and a CSV leaderboard. As of 2026-04-18, both batter AND pitcher cohorts are populated (bug fix to DML residual aggregation in `train_test_split()`).

3. **Gate check.** Parse the metrics JSON. Compare:
   - `pearson_r >= 0.5` → PASS/FAIL
   - `spearman_rho >= 0.6` → PASS/FAIL
   - Record measured values, 95% CIs, and biggest-movers CSV path. Note in `notes` if the lower CI bound dips below the threshold (known fragility on combined Spearman).

### 4b — PitchGPT

1. **Leakage check.** Confirm checkpoint at `models/pitchgpt_v1.pt`. The pitchgpt scripts use fixed splits: train 2015-2022, val 2023, test 2024 (see `TRAIN_RANGE / VAL_RANGE / TEST_RANGE` constants in `scripts/pitchgpt_calibration_analysis.py`). As of 2026-04-18, `PitchSequenceDataset` enforces pitcher-disjoint splits (no pitcher in any two of train/val/test). Verify `shared_pitcher_ids_train_test == 0` and `shared_pitcher_ids_train_val == 0` in the leakage_audit emitted by the ablation script. If either is non-zero, FAIL the leakage gate.

2. **Ablation.** Invoke:
   ```
   python scripts/pitchgpt_ablation.py \
     --output-dir results/validate_pitchgpt_<ts>/ \
     --epochs 5 --seed 42
   ```
   This is expensive (four training runs). On RTX 3050 GPU (auto-detected via `torch.cuda.is_available()`), expect ~5 min per variant at default 1K train games. If the user requests a faster dry-run, pass `--max-train-games 200 --max-val-games 100 --max-test-games 100` and mark the result "smoke" in the summary. Output: `pitchgpt_ablation_metrics.json` with full/tokens_only/count_only/identity_only test perplexities; script also drops `pitchgpt_full.pt` for downstream calibration.

3. **Calibration.** Invoke (after ablation so `pitchgpt_full.pt` exists; otherwise point at `models/pitchgpt_v1.pt`):
   ```
   python scripts/pitchgpt_calibration_analysis.py \
     --checkpoint results/validate_pitchgpt_<ts>/pitchgpt_full.pt \
     --output-dir results/validate_pitchgpt_<ts>/
   ```
   The script has no fallback logic — its argparse default is `Path("results/run_5epoch/pitchgpt_full.pt")`. If ablation was run in this session, pass `--checkpoint results/validate_pitchgpt_<ts>/pitchgpt_full.pt` (as shown above). If ablation was skipped, pass `--checkpoint models/pitchgpt_v1.pt` explicitly. Output: `pitchgpt_calibration.json` (ECE pre/post temperature, optimal T) and `pitchgpt_reliability.html`.

4. **Gate checks.**
   - Calibration: `ece_pre_temperature < 0.10` AND `ece_post_temperature < 0.10` → PASS/FAIL.
   - Ablation: for each variant in {tokens_only, count_only, identity_only}, `(variant_ppl - full_ppl) / full_ppl >= 0.10` → PASS/FAIL per variant. Overall ablation gate PASSes only if all three do.

### 4c — MechanixAE (descriptive profiling)

**STATUS as of 2026-04-18: producer script TBD.** The spec was reframed away from injury early-warning to descriptive mechanical profiling; the legacy `scripts/mechanix_ae_roc_analysis.py` produces the wrong artifacts under the new gates. A new `scripts/mechanix_ae_profiling_analysis.py` has not yet been written. Until it exists:

1. **Leakage check.** Per-pitcher VAE checkpoints live under `models/mechanix_ae/per_pitcher/*.pt`. Training excludes pitches within 30 days pre-IL (spec section A). `--skip-train` is the default behavior when per-pitcher checkpoints already exist. Record `leakage_check: PASS`.

2. **Profiling pipeline.** Run a script (when written) that produces:
   - `per_pitcher_fit.csv` — VAE reconstruction MSE per pitcher
   - `mdi_distribution.csv` — per-pitcher MDI distribution stats (std, skew, range)
   - `coverage.json` — fraction of qualified pitchers (≥200 healthy pitches) with usable checkpoints
   - `intra_pitcher_stability.csv` + `stability_summary.json` — within-game adjacent-window MDI Pearson r
   - `feature_attribution_top_decile.csv` + `attribution_summary.json` — per-feature MSE replication for top-decile MDI windows

3. **Gate checks.** Parse the artifacts above per Step 2 thresholds.

4. **If the producer script does not exist**: write `validation_summary.json` with `overall_pass: null`, gates list with `measured: "PENDING — scripts/mechanix_ae_profiling_analysis.py not yet implemented"` for each gate, and a clear note in `notes` that this is a documented follow-up. Do NOT fall back to the old ROC script — its artifacts do not match the new gates.

### 4d — Viscoelastic Workload (VWR)

1. **Leakage check.** Per-pitcher VWR fits under `models/viscoelastic/per_pitcher/`. The 60-day pre-IL eligibility filter prevents post-injury mechanics leaking into baseline fit (`_MIN_CAREER_PITCHES = 500` enforced before `il_date - 60d`). Healthy-control 90-day windows are sampled with a fixed seed for reproducibility. Record `leakage_check: PASS` with the eligibility-filter justification.

2. **ROC / lead-time / FPR / seasonal-residual analysis.** Invoke:
   ```
   python scripts/viscoelastic_workload_roc_analysis.py
   ```
   The script writes into `results/viscoelastic_workload/` (hardcoded — do not pass a custom output dir). Outputs include `summary.json` with the full `roc.seasonal_control` block (raw_auc, residual_auc, delta_residual_auc, joint_auc), `roc_curve.json`, `lead_time_per_pitcher.csv`, `lead_time_distribution.json`, `fpr_summary.json`, `fpr_healthy_seasons.csv`. After the script finishes, copy these into `results/validate_viscoelastic_workload_<ts>/`.

3. **Gate checks.** Parse `summary.json`:
   - `roc.seasonal_control.residual_auc >= 0.65` → PASS/FAIL (HEADLINE)
   - `roc.seasonal_control.delta_residual_auc >= 0.10` → PASS/FAIL
   - `lead_time.lead_time_t85.median_days >= 30` → PASS/FAIL
   - `fpr.pct_breach_85 <= 0.30` → PASS/FAIL
   - `roc.seasonal_control.residual_auc >= roc.vwr.auc` → PASS/FAIL (sanity)
   Record raw_auc as informational; do NOT gate on it (it is contaminated by season-day confounding).

### 4e — Allostatic Load (ABL)

1. **Leakage check.** Cohort built from `data/injury_labels.parquet` (2015-2016 arm and non-arm IL stints; ABL is a hitter model and uses `non_arm` IL labels as positives). Eligibility filter: ≥500 PAs before `il_date - 60d`. Healthy controls sampled with fixed seed. Record `leakage_check: PASS`.

2. **ROC / lead-time / FPR analysis.** Invoke:
   ```
   python scripts/allostatic_load_roc_analysis.py
   ```
   Script writes to `results/allostatic_load/` (hardcoded). Outputs: `summary.json`, `roc_curve.json`, `lead_time_per_player.csv`, `lead_time_distribution.json`, `fpr_summary.json`, `fpr_healthy_seasons.csv`. Copy artifacts into `results/validate_allostatic_load_<ts>/`.

3. **Gate checks.** Parse `summary.json`:
   - `roc.abl.auc >= 0.60` → PASS/FAIL
   - `lead_time.lead_time_t75.median_days >= 14` → PASS/FAIL
   - `fpr.pct_breach_75 <= 0.30` → PASS/FAIL
   - `roc.delta_auc >= 0.05` → PASS/FAIL
   Informational seasonal-control: `residual_auc >= 0.55` AND `(raw_auc - residual_auc) <= 0.08`. If either fails, note "potential seasonal artifact" in `notes`.

4. **Verdict band.** Per the ABL spec's verdict table, record in `notes` whether the result falls in FLAGSHIP / MARGINAL / NOT FLAGSHIP. ABL's first-pass result (2026-04-18) was NOT FLAGSHIP (1/4 hard gates pass) — this is documented as expected, not a regression.

### 4f — Defensive Pressing (DPI)

1. **Leakage check.** The validation script performs in-memory split on `pitches` table by season range — train 2015-2022 / test 2023-2024 are disjoint by construction. Confirm `shared_game_pks_train_test == 0` in the emitted `leakage_audit` block. Record `leakage_check: PASS` with the season-disjointness justification.

2. **xOut model fit + DPI computation.** Invoke:
   ```
   python scripts/defensive_pressing_validation.py --output-dir results/validate_defensive_pressing_<ts>/
   ```
   The script: (a) fits a `HistGradientBoostingClassifier` xOut model on per-BIP features `[launch_speed, launch_angle, spray_angle, bb_type]` from 2015-2022 (train) BIP rows, (b) holds out a 50K test sample for AUC, (c) aggregates team-season DPI = `SUM(actual_out - expected_out_prob)` over 2023-2024 BIP, (d) computes consumer-claim correlations against run-prevention proxy and BABIP-against, (e) computes YoY stability across the test window. Emits `defensive_pressing_validation_metrics.json`, `team_seasons.csv`, `xout_holdout_sample.csv`, `step_<n>.log`.

3. **Gate checks.** Parse `defensive_pressing_validation_metrics.json`:
   - `metrics.xout.test_auc >= 0.70` → PASS/FAIL
   - `metrics.consumer_claim.dpi_vs_rp_proxy.pearson_r >= 0.40` → PASS/FAIL
   - `metrics.consumer_claim.dpi_vs_babip_against.pearson_r <= -0.50` → PASS/FAIL (negative direction)
   - `metrics.stability.yoy_pearson_r >= 0.30` → PASS/FAIL
   - `leakage_audit.shared_game_pks_train_test == 0` → PASS/FAIL

   Known limitations to record in `notes` (do not block PASS): no external DRS/OAA baseline staged (consumer-claim gates use proxies derived from same `pitches` table); no park feature in xOut (Coors/Fenway bias); xOut model is currently re-trained on every run (not pickled — see Operational notes); DPI silent on K-suppression staffs (no BIP = no credit).

## Step 5 — Consolidate into a single JSON summary

Write `results/validate_<model>_<ts>/validation_summary.json`:

```json
{
  "model": "<causal_war|pitchgpt|mechanix_ae|viscoelastic_workload|allostatic_load|defensive_pressing>",
  "invocation": "/validate-model <model>",
  "timestamp_utc": "<ISO-8601>",
  "spec_path": "docs/models/<model>_validation_spec.md",
  "gates": [
    {"name": "...", "threshold": "...", "measured": "...", "operator": "...", "pass": true, "source": "..."},
    ...
  ],
  "overall_pass": true,
  "artifacts": ["results/.../metrics.json", "..."],
  "wall_clock_seconds": {"...": ...},
  "notes": "any soft fails, fragility callouts, verdict bands, smoke-mode flags"
}
```

`overall_pass` is `true` iff every hard gate is `true`. Soft flags (CausalWAR Spearman lower-CI fragility, ABL seasonal-control informational, ABL verdict band) do not flip it — record them in `notes`. For MechanixAE when the profiling script is not yet implemented, set `overall_pass: null` and explain in `notes`.

## Step 6 — Append verdict to the results doc

Append (do not overwrite) a new section to the appropriate doc:
- `causal_war` → `docs/models/causal_war_baseline_results.md`
- `pitchgpt` → `docs/models/pitchgpt_calibration_ablation_results.md`
- `mechanix_ae` → `docs/models/mechanix_ae_results.md`
- `viscoelastic_workload` → `docs/models/viscoelastic_workload_results.md`
- `allostatic_load` → `docs/models/allostatic_load_results.md`
- `defensive_pressing` → `docs/models/defensive_pressing_results.md`

Section template:

```markdown

---

## Validation run <ISO timestamp>

**Invocation:** `/validate-model <model>`
**Summary JSON:** `results/validate_<model>_<ts>/validation_summary.json`

| Gate | Threshold | Measured | Verdict |
|---|---|---|---|
| ... | ... | ... | PASS/FAIL |

**Overall:** PASS / FAIL

**Failed gates:** (list, or "none")

**Artifacts:**
- `results/validate_<model>_<ts>/...`

**Notes:** <soft flags, fragility callouts, verdict bands, smoke-mode disclaimers>
```

Use plain ASCII in the verdict (no emoji — matches repo convention).

## Step 7 — Terse final summary to user

Print (do not write to disk — this is the assistant reply):

```
<Model>: <PASS|FAIL|PENDING>
Gates passed: X / Y
Failed: <comma-separated gate names, or "none">
Report: results/validate_<model>_<ts>/validation_summary.json
Verdict appended to: docs/models/<results_doc>.md
```

For `all` / `flagships`, print one block per model in the order they ran, then a one-line roster summary at the end (e.g., "Flagship roster: 3/4 PASS — CausalWAR PASS, PitchGPT FAIL (ablation), VWR PASS, DPI PASS").

Keep each block to those 5 lines unless the user asked for detail.

## Operational notes

- **Hardware.** GPU is available — `torch 2.11.0+cu126`, RTX 3050 Laptop, 4GB VRAM, cuDNN 9.10.02. Scripts auto-detect via `_get_device()` / `torch.cuda.is_available()`. Per-model:
  - **PitchGPT**: GPU strongly preferred (~2× faster than CPU at small model size, ~800 MB peak VRAM).
  - **MechanixAE**: GPU eval (light); training would be GPU-heavy but `--skip-train` is default.
  - **CausalWAR**: CPU only (sklearn / DML).
  - **VWR / Allostatic Load**: CPU only (sklearn LinearRegression / LogisticRegression + numpy).
  - **Defensive Pressing (DPI)**: CPU only (sklearn HistGradientBoostingClassifier + numpy). Known limitation: xOut model is re-trained on every validation run (not pickled). v2 hardening ticket: persist xOut checkpoint to `models/defensive_pressing/xout_v1.pkl`.
- **Never modify** existing scripts, specs, results docs above the appended verdict line, or model checkpoints.
- **Never retrain** the deployed models — the scripts here either are inference-only (calibration, ROC) or do isolated ablation training that writes to the run directory only.
- **DB access is read-only.** Scripts open DuckDB via `src.db.schema.get_connection`; the DB file is at the repo default. Do not pass a custom `--db-path` unless the user explicitly asks.
- **Windows paths.** The repo runs on Windows. Use forward slashes in argument values — Python's `pathlib` handles conversion. The bash shell is available; prefer `python scripts/...` invocations.
- **Long runs.** PitchGPT ablation (four training runs) takes ~5 min on GPU at default 1K train games. Warn the user up-front if invoking full ablation; offer smoke mode as a fallback.
- **Partial failures.** If one script fails, continue with independent steps (e.g., if PitchGPT calibration fails, still report ablation results). Overall verdict is FAIL but the report is still useful.
- **Idempotent re-run.** Each run gets its own timestamped directory — never overwrite a prior validation run.
- **Bulk runs (`all` / `flagships`).** Run sequentially, not in parallel — VRAM is shared and DuckDB has a single-writer lock for any rare write paths. Stop and report on first hard error rather than continuing blindly.
