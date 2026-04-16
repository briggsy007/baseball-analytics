# tests/baselines.json

## Purpose

`baselines.json` stores metric thresholds used by `tests/test_model_performance.py`
to detect silent regressions in the 6 models that expose quantitative evaluation
output: Stuff+, MechanixAE, PitchGPT, ChemNet, CausalWAR, and PSET. If a trained
model's metric drifts past its stored bound (or past the bound widened by
`regression_tolerance`, default 5%), the corresponding test fails loudly instead
of silently passing.

The other 10 analytics models (alpha_decay, baserunner_gravity, loft, mesi,
pitch_decay, sharpe_lineup, viscoelastic_workload, volatility_surface,
kinetic_half_life, defensive_pressing, allostatic_load) do not yet have
regression assertions in `test_model_performance.py` and are therefore not
represented here.

## How to update a baseline

1. Run the relevant training / evaluation script (e.g. `scripts/train_stuff_model.py`,
   `scripts/train_mechanix_ae.py`, `scripts/train_pitchgpt.py`,
   `scripts/train_chemnet.py`, or call `model.evaluate(conn)` directly for
   compute-only models).
2. Read the metric returned by `evaluate()` (e.g. `r2_test`, `final_recon_loss`,
   `mean_pps`, `mean_synergy`, `mean_causal_war`, `mean_pset_per_100`).
3. Replace the threshold:
   - For lower bounds (`*_min`): store `best_value * (1 - regression_tolerance)`.
   - For upper bounds (`*_max`): store `best_value * (1 + regression_tolerance)`.
   - For two-sided ranges, widen both sides by tolerance.
4. Remove the corresponding `_TODO` key once the real value is in place.
5. Commit the updated `baselines.json` in the same change as the training-run
   artifact it was derived from, so the baseline and the model weights stay
   in sync.

## Currently stubbed (needs first-run numbers)

All six model entries are presently stubbed with deliberately loose
placeholders and carry a `_TODO` note:

- `stuff_plus` — run `scripts/train_stuff_model.py`
- `mechanix_ae` — run `scripts/train_mechanix_ae.py`
- `pitchgpt` — run `scripts/train_pitchgpt.py`
- `chemnet` — run `scripts/train_chemnet.py`
- `causal_war` — run `CausalWARModel.train(conn, season=...)` then `evaluate`
- `pset` — run `PSETModel().evaluate(conn)` against a full season

## Warning

Do **not** loosen these thresholds to make a failing regression test pass
unless you have independently confirmed that the model's performance
regression is intentional and accepted (e.g. retraining on different data,
an architecture change with known trade-offs). The whole point of this file
is to catch silent quality loss; editing the number to silence the alarm
defeats the purpose.
