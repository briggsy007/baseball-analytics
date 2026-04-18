"""
Deprecated alias for :mod:`scripts.train_pitchgpt_baselines`.

The 2-way PitchGPT-vs-LSTM runner was extended to a 4-way comparison
(PitchGPT vs LSTM vs Markov-1/2 vs Heuristic) in the Markov baseline
ticket.  This module re-exports the new module's public surface so any
callers that still import ``train_pitchgpt_vs_lstm`` (e.g. the legacy
test harness in ``tests/test_pitchgpt_vs_lstm.py``) continue to work.

For new code, import ``scripts.train_pitchgpt_baselines`` directly.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Make the project root importable when run directly.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

# Re-export the public API of the canonical 4-way module.
from train_pitchgpt_baselines import (  # type: ignore  # noqa: F401,E402
    DEFAULT_BATCH,
    DEFAULT_LR,
    GRAD_CLIP,
    TEST_RANGE,
    TRAIN_RANGE,
    VAL_RANGE,
    _build_pairwise_verdict,
    _build_verdict,
    _evaluate_closed_form,
    _iter_epoch,
    _set_seed,
    _train_model,
    _write_training_curves_html,
    main,
)

# Legacy constant name expected by older tests / callers.
SPEC_THRESHOLD_PCT = 15.0


if __name__ == "__main__":
    warnings.warn(
        "scripts/train_pitchgpt_vs_lstm.py is deprecated — use "
        "scripts/train_pitchgpt_baselines.py instead.  This shim will "
        "be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    sys.exit(main())
