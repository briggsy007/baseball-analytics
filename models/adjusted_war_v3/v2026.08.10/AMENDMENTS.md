# Manifest amendments — `adjusted_war_v3` `v2026.08.10`

`manifest.json` in this directory is **write-once and immutable**
(`src/analytics/registry.py`: the WS2.1 layout note, "manifest.json
(write-once, immutable)", and the `set_alias` docstring, "manifests and
version dirs are write-once"). It is never edited after registration, even
when a statement inside it stops being true. Corrections are appended here
instead, newest first. Read this file alongside the manifest.

## 2026-08-11 — the `notes` field is STALE on the alias question

`manifest.json` `notes` reads:

> EXPERIMENTAL ridge joint-estimation model (working name AdjustedWAR v3).
> No production/frozen_validated alias set: naming + flagship decisions are
> pre-registered as the user call in Batch D.

True at registration (`created_utc` 2026-08-10T22:26:31Z). No longer true:

- **`production` IS set** to `v2026.08.10`, at 2026-08-11T00:03:29Z, per the
  user-adjudicated Batch D decision (K3 does not fire; AdjustedWAR v3 ridge
  becomes the production player-value model). The authoritative record is the
  `alias` history entry for `adjusted_war_v3` in `models/registry.json` — the
  only mutable file in the registry — which carries the full promotion note.
- **`frozen_validated` remains unset, deliberately.** No validation spec
  exists for `adjusted_war_v3`
  (`docs/models/adjusted_war_v3_validation_spec.md` does not exist), so there
  is no pre-registered gate suite this artifact could have passed. The
  promotion rests on the pre-registered K3 measurement set in
  `docs/models/adjusted_war_v3_2026-08.md` (season-forward RMSE vs the legacy
  formulation; 17 fully-OOS board windows vs matched-naive and vs Marcel) —
  **not** on a validation-spec pass. `frozen_validated` must stay unset until
  a spec is written, pre-registered, and passed.
- The "working name" wording is superseded: **AdjustedWAR** is the product
  name on live surfaces as of 2026-08-10 (formerly CausalWAR). File names,
  module paths, registry ids and pick-ledger product ids keep their
  historical `causal_war` spellings on purpose.
- **K6 binds** on anything scored by this artifact: beats matched-naive, does
  not beat the Marcel-picker, ties Marcel on season-forward forecast — no
  edge claim vs Marcel (claim `adjusted_war_boards_k6_framing`).
- Per-player confidence intervals from this artifact **do not ship**: WS4.7
  measured 49.6% / 71.3% empirical coverage at a nominal 95% against a
  pre-registered [90%, 98%] gate (claim `adjusted_war_v3_ci_coverage`).

Nothing else in the manifest changed: `sha256`, `train_window`,
`training_script` and `validation_results_ref` stand as registered.
