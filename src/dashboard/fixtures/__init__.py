"""Hand-built fixtures used by dashboard view stubs while real engines land.

Each fixture mirrors the shape of an upstream artifact (e.g. a
``RolloutResult`` from ``pitchgpt_sim.rollout()``) without importing the
upstream module, so views can render in dev environments where the upstream
implementation has not yet been merged.

Fixtures live here (not under ``tests/``) because they are also consumed by
the dashboard at runtime as a "Phase 0.5 not yet available" placeholder.
"""
