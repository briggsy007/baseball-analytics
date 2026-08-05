"""
PitchGPT sim engine -- Phase 0.5 rollout harness.

This module is the API surface specified in
``docs/pitchgpt_sim_engine/SIM_ENGINE_API.md``.  It is the single dependency
that downstream Tier-A consumers (A1 grades, A2 projections, A3 matchup sim,
plus Tier-B B1-B4) hit; the contract is locked.

Phase 0.5 is split across two parallel agents (per
``docs/pitchgpt_sim_engine/PHASE_0.5_PLAN.md`` §4):

* **Agent 1** (tickets 0.5.1 / 0.5.2 / 0.5.3): the rollout API contract,
  `OutcomePredictor` protocol with four implementations, aggregation
  utilities.  Owns the public surface.
* **Agent 2** (tickets 0.5.4 / 0.5.5 / 0.5.6): calibration validity gate,
  calibration.json artifacts, unit-test suite.

This file currently holds:

* Locked module constants ``ROLLOUT_PAD_PITCH``, ``ROLLOUT_PAD_OUTCOME``,
  ``ROLLOUT_ENGINE_VERSION``, the four condition reason strings (§6).
* Frozen ``PAContext`` and ``RolloutResult`` dataclass *shapes* per
  SIM_ENGINE_API §3.2 / §3.3 (so Agent 2's calibration gate and unit tests
  have stable shapes to reference).  Agent 1 may extend these dataclasses
  with helpers (``PAContext.from_pitches_row``) but MUST NOT mutate field
  layout or types.
* The full ``_check_calibration_valid`` helper (Agent 2 ticket 0.5.4).
* A ``CalibrationError`` exception type (§7.3).

What it does NOT yet hold (waiting on Agent 1):

* ``rollout()`` entry point and PA-termination logic (ticket 0.5.1).
* ``OutcomePredictor`` Protocol + four concrete implementations
  (ticket 0.5.2).
* Aggregation utilities ``pa_woba_distribution``,
  ``percentile_of_actual_outcome``, ``outcome_marginal``,
  ``pitch_token_marginal`` (ticket 0.5.3).

Once Agent 1 lands its tickets, the only required wiring is to call
``_check_calibration_valid`` from ``rollout()`` and put the (bool, list[str])
result into ``sampling_metadata["calibration_valid"]`` and
``sampling_metadata["calibration_invalid_reasons"]``.

DO NOT modify ``models/pitchgpt_v2.pt``,
``models/pitchgpt_v2_outcomehead.pt`` or
``models/pitchgpt_v2_outcomehead_a1.pt``.  DuckDB is read-only here.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# Module-level constants  (locked per SIM_ENGINE_API §3 and §6)
# ═════════════════════════════════════════════════════════════════════════════

#: Pad token for ``RolloutResult.pitch_tokens`` past PA termination
#: (= ``PAD_TOKEN`` in ``src/analytics/pitchgpt.py``).
ROLLOUT_PAD_PITCH: int = 2210

#: Pad sentinel for ``RolloutResult.outcomes`` and ``pa_outcome`` past PA
#: termination (out-of-range for the 7-class index).
ROLLOUT_PAD_OUTCOME: int = 7

#: Semantic version of this rollout engine surface.  Bump on contract change.
ROLLOUT_ENGINE_VERSION: str = "1.0"

#: ECE budgets per SIM_ENGINE_API §7.3 ("MUST be < 0.02 for the backbone";
#: "MUST be < 0.05 for any outcome predictor").
_ECE_BUDGET_BACKBONE: float = 0.02
_ECE_BUDGET_OUTCOME_PREDICTOR: float = 0.05

#: Calibration `fit_date` staleness threshold in days (12 months per §7.3).
_CALIBRATION_STALE_DAYS: int = 365

#: Horizon validity threshold; horizons > this trigger
#: ``"horizon_exceeds_validated"`` per SIM_ENGINE_API §3.1 / §6.
_HORIZON_VALIDATED_MAX: int = 12

#: Calibration-feature CDF percentile band (1st / 99th).  See SIM_ENGINE_API
#: §6 condition 4.  A starting context's value at percentile <0.01 or >0.99
#: of the calibration cohort triggers the corresponding "out_of_band" reason.
_CDF_BAND_LO: float = 0.01
_CDF_BAND_HI: float = 0.99

#: Default location of the calibration-feature CDFs npz.  The build script
#: ``scripts/pitchgpt_build_calibration_cdfs.py`` writes here.
DEFAULT_CALIBRATION_CDFS_PATH: Path = (
    Path(__file__).resolve().parents[2] / "models" / "calibration_feature_cdfs.npz"
)

#: Enumerated invalid-reason strings (§6).  Order is presentation-only.
CALIBRATION_INVALID_REASONS: tuple[str, ...] = (
    "temperature_not_unity",
    "horizon_exceeds_validated",
    "backbone_calibration_stale",
    "outcome_predictor_calibration_stale",
    "context_count_state_out_of_band",
    "context_outs_out_of_band",
    "context_score_diff_out_of_band",
    "context_inning_out_of_band",
    "context_runner_state_unseen",
)


class CalibrationError(RuntimeError):
    """Raised when a calibration.json is missing, stale, out-of-budget, or
    has a mismatched checkpoint_sha256.  See SIM_ENGINE_API §7.3 refusal
    behavior.  No override flag.
    """


# ═════════════════════════════════════════════════════════════════════════════
# Dataclasses (SIM_ENGINE_API §3.2 / §3.3 -- frozen contract surface)
# ═════════════════════════════════════════════════════════════════════════════
#
# Agent 1 owns the rollout() implementation but the dataclass *shapes* are
# locked here so Agent 2's calibration gate and unit tests can reference them
# without coordination friction.  Field types and order MUST match
# SIM_ENGINE_API §3.2 / §3.3.  Agent 1 MAY add factory classmethods (e.g.,
# ``PAContext.from_pitches_row``) but MUST NOT change field layout.


@dataclass(frozen=True)
class PAContext:
    """Starting state of a plate appearance.  See SIM_ENGINE_API §3.2."""

    pitcher_id: int
    batter_id: int
    count: tuple[int, int]
    outs: int
    runners: tuple[bool, bool, bool]
    batter_stand: Literal["L", "R"]
    pitcher_throws: Literal["L", "R"]
    inning: int
    inning_topbot: Literal["Top", "Bot"]
    score_diff: int
    umpire_scalar: float = 0.0
    prefix_pitch_tokens: tuple[int, ...] = ()

    @property
    def count_state(self) -> int:
        """Encoded count state ∈ [0, 11], matching ``encode_context``.

        Formula ``min(balls, 3) * 3 + min(strikes, 2)``.
        """
        balls, strikes = self.count
        return min(max(balls, 0), 3) * 3 + min(max(strikes, 0), 2)

    @property
    def runner_state(self) -> int:
        """Encoded runner state ∈ [0, 7], matching ``encode_context``."""
        on_1b, on_2b, on_3b = self.runners
        return (4 if on_1b else 0) + (2 if on_2b else 0) + (1 if on_3b else 0)

    @classmethod
    def from_pitches_row(
        cls,
        row: Any,
        ump_scalar: float | None = None,
    ) -> "PAContext":
        """Build a :class:`PAContext` from a single ``pitches`` table row.

        ``row`` may be a pandas Series, a dict-like, or any object that
        supports ``__getitem__`` / ``.get`` for column access.  ``ump_scalar``
        is optional; if omitted, defaults to 0.0 ("neutral / unknown ump")
        per SIM_ENGINE_API §3.2.

        This factory is *not* part of the rollout contract (per §3.2);
        it is provided as a convenience.  Callers in Phase 0.6 use it
        to construct contexts from ``pitches`` cohort rows.
        """

        def _g(key: str, default=None):
            if hasattr(row, "get"):
                return row.get(key, default)
            try:
                return row[key]
            except (KeyError, IndexError, TypeError):
                return default

        def _safe_int(val, default=0):
            if val is None:
                return default
            try:
                if isinstance(val, float) and np.isnan(val):
                    return default
                return int(val)
            except (TypeError, ValueError):
                return default

        def _safe_bool(val):
            if val is None:
                return False
            try:
                if isinstance(val, float) and np.isnan(val):
                    return False
                return bool(int(val))
            except (TypeError, ValueError):
                return bool(val)

        balls = _safe_int(_g("balls"), 0)
        strikes = _safe_int(_g("strikes"), 0)
        outs = _safe_int(_g("outs_when_up"), 0)
        runners = (
            _safe_bool(_g("on_1b")),
            _safe_bool(_g("on_2b")),
            _safe_bool(_g("on_3b")),
        )
        stand = str(_g("stand", "R") or "R")
        if stand not in ("L", "R"):
            stand = "R"
        p_throws = str(_g("p_throws", "R") or "R")
        if p_throws not in ("L", "R"):
            p_throws = "R"
        inning = max(1, _safe_int(_g("inning"), 1))
        topbot = str(_g("inning_topbot", "Top") or "Top")
        if topbot not in ("Top", "Bot"):
            topbot = "Top"
        score_diff = _safe_int(_g("score_diff"), 0)
        pitcher_id = _safe_int(_g("pitcher_id"), 0)
        batter_id = _safe_int(_g("batter"), 0)
        ump = float(ump_scalar) if ump_scalar is not None else 0.0

        return cls(
            pitcher_id=pitcher_id,
            batter_id=batter_id,
            count=(min(balls, 3), min(strikes, 2)),
            outs=min(outs, 2),
            runners=runners,
            batter_stand=stand,  # type: ignore[arg-type]
            pitcher_throws=p_throws,  # type: ignore[arg-type]
            inning=min(inning, 30),
            inning_topbot=topbot,  # type: ignore[arg-type]
            score_diff=score_diff,
            umpire_scalar=ump,
            prefix_pitch_tokens=(),
        )


@dataclass(frozen=True)
class RolloutResult:
    """Output of ``rollout()``.  See SIM_ENGINE_API §3.3."""

    pitch_tokens: np.ndarray  # (n_samples, horizon), int64
    pitch_probs: np.ndarray | None  # (n_samples, horizon, 2210), float32
    outcomes: np.ndarray | None  # (n_samples, horizon), int64
    outcome_probs: np.ndarray | None  # (n_samples, horizon, 7), float32
    pa_terminated: np.ndarray  # (n_samples, horizon), bool
    pa_outcome: np.ndarray | None  # (n_samples,), int64
    final_count: np.ndarray  # (n_samples, 2), int64 (balls, strikes)
    sampling_metadata: dict = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# Calibration validity helpers
# ═════════════════════════════════════════════════════════════════════════════


def _bucket_score_diff(score_diff: int) -> int:
    """Match ``PitchTokenizer.encode_context``'s 5-level bucketing."""
    if score_diff <= -4:
        return 0
    if score_diff < 0:
        return 1
    if score_diff == 0:
        return 2
    if score_diff <= 3:
        return 3
    return 4


def _bucket_inning(inning: int) -> int:
    """Match ``PitchTokenizer.encode_context``'s 4-bucket inning split."""
    if inning <= 3:
        return 0
    if inning <= 6:
        return 1
    if inning <= 9:
        return 2
    return 3


def _calibration_is_stale(
    calibration: dict | None,
    *,
    today: date | None = None,
    stale_days: int = _CALIBRATION_STALE_DAYS,
) -> bool:
    """Return True if a calibration's `fit_date` is older than `stale_days`.

    Missing or unparseable `fit_date` is treated as stale (conservative).
    """
    if not calibration:
        return True
    fit_date_str = calibration.get("fit_date")
    if not fit_date_str:
        return True
    try:
        fit = date.fromisoformat(str(fit_date_str))
    except (TypeError, ValueError):
        return True
    today = today or date.today()
    return (today - fit).days > stale_days


def _cdf_value_in_band(
    cdf: np.ndarray | None,
    value: int,
    *,
    band_lo: float = _CDF_BAND_LO,
    band_hi: float = _CDF_BAND_HI,
) -> bool:
    """Test whether `value` (a categorical bucket index) lies within the
    [band_lo, band_hi] percentile band of `cdf`.

    Per SIM_ENGINE_API §6 condition 4: each numeric/categorical feature's
    starting value must fall within the [1st, 99th] percentile of the
    calibration-cohort's empirical CDF.  A value `v` is in-band iff:

        cdf[v-1] <= band_hi   (i.e., value isn't past the high tail)
        AND
        cdf[v]    >= band_lo  (i.e., value isn't entirely below the low tail)

    The first row (v=0) uses an implicit `cdf[-1] = 0.0`.

    If `cdf` is None (CDFs file not loaded), we conservatively return True --
    the staleness/missing-cdfs check happens elsewhere.  If `value` is outside
    the cdf's bucket range, we return False.
    """
    if cdf is None:
        return True
    if value < 0 or value >= cdf.shape[0]:
        return False

    # Lower edge of bucket `v`
    lo_edge = float(cdf[value - 1]) if value > 0 else 0.0
    # Upper edge of bucket `v`
    hi_edge = float(cdf[value])

    # In-band if the bucket spans some portion of [band_lo, band_hi].
    return (hi_edge >= band_lo) and (lo_edge <= band_hi)


def _runner_state_seen(runner_mask: np.ndarray | None, value: int) -> bool:
    """Return True if `runner_mask[value]` is True (calibration cohort
    observed this 8-state runner combination).  None mask = conservatively
    True; out-of-range = False.
    """
    if runner_mask is None:
        return True
    if value < 0 or value >= runner_mask.shape[0]:
        return False
    return bool(runner_mask[value])


def load_calibration_feature_cdfs(
    path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Load the calibration-feature CDFs npz.  Returns None if missing.

    The build script ``scripts/pitchgpt_build_calibration_cdfs.py`` produces
    this file.  Keys: count_state_cdf, outs_cdf, score_diff_bucket_cdf,
    inning_bucket_cdf, runner_state_mask, holdout_season,
    holdout_n_pitches, fit_date.
    """
    p = Path(path) if path else DEFAULT_CALIBRATION_CDFS_PATH
    if not p.exists():
        logger.warning("calibration_feature_cdfs.npz missing at %s", p)
        return None
    try:
        with np.load(p, allow_pickle=False) as data:
            return {
                "count_state_cdf": np.asarray(data["count_state_cdf"], dtype=np.float32),
                "outs_cdf": np.asarray(data["outs_cdf"], dtype=np.float32),
                "score_diff_bucket_cdf": np.asarray(
                    data["score_diff_bucket_cdf"], dtype=np.float32,
                ),
                "inning_bucket_cdf": np.asarray(data["inning_bucket_cdf"], dtype=np.float32),
                "runner_state_mask": np.asarray(data["runner_state_mask"], dtype=np.bool_),
                "holdout_season": int(data["holdout_season"]),
                "holdout_n_pitches": int(data["holdout_n_pitches"]),
                "fit_date": str(data["fit_date"]),
            }
    except (OSError, ValueError, KeyError) as exc:
        logger.warning("calibration_feature_cdfs.npz unreadable at %s: %s", p, exc)
        return None


def _check_calibration_valid(
    starting_context: PAContext,
    backbone_calibration: dict | None,
    predictor_calibration: dict | None,
    temperature: float,
    horizon: int,
    *,
    feature_cdfs: dict[str, Any] | None = None,
    today: date | None = None,
) -> tuple[bool, list[str]]:
    """Implement the four-condition calibration validity gate.

    Per SIM_ENGINE_API §6, ``calibration_valid`` is True iff ALL FOUR
    conditions hold:

        1. ``temperature == 1.0``  (calibration is fit at T=1.0).
        2. Backbone's calibration.json is current.
        3. Outcome predictor's calibration.json is current (vacuously True
           when ``predictor_calibration is None``, i.e., no predictor).
        4. Starting context is in-distribution: per-feature percentile
           band on (count_state, outs, score_diff_bucket, inning_bucket)
           plus the categorical "runner_state must be observed" check.

    Per SIM_ENGINE_API §3.1, ``horizon > 12`` ALSO forces
    ``calibration_valid = False`` with reason ``"horizon_exceeds_validated"``;
    this is folded into condition 1's bucket (sampling-regime invalidation).

    Returns (is_valid, reasons).  ``reasons`` is empty iff is_valid; otherwise
    a list of strings drawn from CALIBRATION_INVALID_REASONS.

    Args:
        starting_context: The PA's starting state (see ``PAContext``).
        backbone_calibration: Loaded backbone calibration.json dict, or None.
            None ⇒ "backbone_calibration_stale" (condition 2 fails).
        predictor_calibration: Loaded predictor calibration.json dict, or None.
            None means "no outcome predictor in this rollout" -- condition 3
            is vacuously True (NOT a fail).  If the predictor IS present but
            its calibration is malformed, the caller should pass the loaded
            dict and let staleness drive the verdict.
        temperature: Sampling temperature.
        horizon: Rollout horizon length.
        feature_cdfs: Optional preloaded calibration-feature CDFs (saves
            re-reading the npz per call).  Defaults to lazy-load.
        today: Override for today's date (testing).

    Notes:
        * The split between "no predictor present" (condition 3 vacuously
          True) and "predictor present, calibration stale" (condition 3
          fails) is decided by the *caller*: callers without a predictor
          should pass ``predictor_calibration=None``.  Callers with a
          predictor should pass its calibration dict regardless of staleness
          so this function can decide.
    """
    reasons: list[str] = []

    # ── Condition 1 + horizon hint: sampling regime ──────────────────────
    if not _isclose(temperature, 1.0):
        reasons.append("temperature_not_unity")
    if horizon > _HORIZON_VALIDATED_MAX:
        reasons.append("horizon_exceeds_validated")

    # ── Condition 2: backbone calibration current ────────────────────────
    if _calibration_is_stale(backbone_calibration, today=today):
        reasons.append("backbone_calibration_stale")

    # ── Condition 3: outcome predictor calibration current ───────────────
    # Vacuously True when no predictor is in the rollout.
    if predictor_calibration is not None:
        if _calibration_is_stale(predictor_calibration, today=today):
            reasons.append("outcome_predictor_calibration_stale")

    # ── Condition 4: starting context in-distribution ────────────────────
    if feature_cdfs is None:
        feature_cdfs = load_calibration_feature_cdfs()

    if feature_cdfs is not None:
        # Use PAContext properties so the bucket math is in one place.
        count_state = starting_context.count_state
        runner_state = starting_context.runner_state
        outs = starting_context.outs
        score_diff_bucket = _bucket_score_diff(starting_context.score_diff)
        inning_bucket = _bucket_inning(starting_context.inning)

        if not _cdf_value_in_band(feature_cdfs.get("count_state_cdf"), count_state):
            reasons.append("context_count_state_out_of_band")
        if not _cdf_value_in_band(feature_cdfs.get("outs_cdf"), outs):
            reasons.append("context_outs_out_of_band")
        if not _cdf_value_in_band(
            feature_cdfs.get("score_diff_bucket_cdf"),
            score_diff_bucket,
        ):
            reasons.append("context_score_diff_out_of_band")
        if not _cdf_value_in_band(feature_cdfs.get("inning_bucket_cdf"), inning_bucket):
            reasons.append("context_inning_out_of_band")
        if not _runner_state_seen(feature_cdfs.get("runner_state_mask"), runner_state):
            reasons.append("context_runner_state_unseen")
    else:
        # CDFs missing -> we cannot verify condition 4; conservatively flag
        # all numeric features by way of the count_state reason (the most
        # common gating cue) and let the consumer surface "context CDFs
        # missing" via the metadata.  We pick `context_count_state_out_of_band`
        # because it's the principal numeric feature; alternative is to
        # introduce a new reason string but the spec enumeration is closed.
        reasons.append("context_count_state_out_of_band")

    return (len(reasons) == 0, reasons)


def _isclose(a: float, b: float, *, atol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= atol


def _env_truthy(name: str) -> bool:
    """Return True if environment variable ``name`` is set to a truthy value.

    Truthy = one of ``{"1", "true", "yes", "on"}`` (case-insensitive).  Used
    for the Phase 0.6.1 ``PITCHGPT_DISABLE_CLASS_CALIBRATION`` A/B switch so
    the orchestrator can flip the all-position class_calibration off for the
    sanity re-run WITHOUT editing any script.
    """
    import os

    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# ═════════════════════════════════════════════════════════════════════════════
# Calibration JSON loading (for §7.3 refusal behavior wiring)
# ═════════════════════════════════════════════════════════════════════════════


def load_calibration_json(
    path: Path | str,
    *,
    expected_predictor_kind: str | None = None,
    ece_budget: float | None = None,
) -> dict:
    """Load a calibration.json with refusal behavior per SIM_ENGINE_API §7.3.

    Raises ``CalibrationError`` on:
        - file missing
        - missing required keys
        - stale ``fit_date``  (>12 months per §7.3)
        - out-of-budget ``ECE_post``  (caller passes the appropriate budget)
        - mismatched ``predictor_kind`` (when caller specifies one)

    Returns the parsed dict on success.

    Note:
        ``checkpoint_sha256`` mismatches are checked at predictor-construction
        time by the predictor itself (it can recompute the hash of the .pt it
        is loading).  This loader does NOT recompute hashes so it stays cheap
        to call from the calibration-validity gate.
    """
    p = Path(path)
    if not p.exists():
        raise CalibrationError(
            f"calibration.json missing at {p} -- per SIM_ENGINE_API §7.3 the "
            "API refuses to use a checkpoint without a calibration file.",
        )
    try:
        with p.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"calibration.json unreadable at {p}: {exc}") from exc

    required = {
        "T",
        "ECE_pre",
        "ECE_post",
        "holdout_season",
        "holdout_n_pitches",
        "fit_date",
        "checkpoint_sha256",
        "predictor_kind",
    }
    missing = required.difference(data.keys())
    if missing:
        raise CalibrationError(
            f"calibration.json at {p} is missing required keys: {sorted(missing)}",
        )

    if expected_predictor_kind is not None and data["predictor_kind"] != expected_predictor_kind:
        raise CalibrationError(
            f"calibration.json at {p} has predictor_kind={data['predictor_kind']!r}; "
            f"expected {expected_predictor_kind!r}",
        )

    if _calibration_is_stale(data):
        raise CalibrationError(
            f"calibration.json at {p} is stale: fit_date={data['fit_date']} is more than "
            f"{_CALIBRATION_STALE_DAYS} days old.  Re-run §7.2's 6-step procedure.",
        )

    if ece_budget is not None and float(data["ECE_post"]) > float(ece_budget):
        raise CalibrationError(
            f"calibration.json at {p} reports ECE_post={data['ECE_post']:.4f} which "
            f"exceeds the budget {ece_budget:.4f} for {data['predictor_kind']!r}.  "
            "Re-fit temperature on a more-recent holdout.",
        )

    return dict(data)


# ═════════════════════════════════════════════════════════════════════════════
# AGENT 1 CONTRIBUTION GOES BELOW THIS LINE
# ═════════════════════════════════════════════════════════════════════════════
#
# Agent 1 owns:
#   - rollout(starting_context, *, backbone_version, outcome_predictor,
#             n_samples, horizon, temperature, seed, return_probs)
#       implementation + PA-termination logic + sampling_metadata population
#       per SIM_ENGINE_API §3.1-§3.4.
#   - OutcomePredictor Protocol + PGConcatHeadPredictor /
#     PGFrozenHeadPredictor / XGBoostOutcomePredictor /
#     EmpiricalPATerminalLookup implementations per §4.
#   - OutcomePredictorRegistry per §8.2.
#   - Aggregation utilities pa_woba_distribution /
#     percentile_of_actual_outcome / outcome_marginal /
#     pitch_token_marginal per §5.
#
# The wire-up that Agent 1 must add inside rollout() once it lands:
#
#       backbone_cal = load_calibration_json(
#           Path("models") / f"pitchgpt_{backbone_version}_calibration.json",
#           expected_predictor_kind="backbone",
#           ece_budget=_ECE_BUDGET_BACKBONE,
#       )
#       pred_cal = (
#           outcome_predictor.calibration if outcome_predictor is not None else None
#       )
#       cal_valid, cal_reasons = _check_calibration_valid(
#           starting_context, backbone_cal, pred_cal, temperature, horizon,
#       )
#       sampling_metadata["calibration_valid"] = cal_valid
#       sampling_metadata["calibration_invalid_reasons"] = cal_reasons
#
# (Agent 2 has done the rest of ticket 0.5.4 — the helper, the CDFs file, and
# the unit tests at tests/test_pitchgpt_sim.py.)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 CONTRIBUTION — rollout, predictors, registry, aggregations
# ─────────────────────────────────────────────────────────────────────────────
#
# Tickets 0.5.1 / 0.5.2 / 0.5.3 land below.  This is everything Agent 1 owns
# per PHASE_0.5_PLAN.md §2:
#
#   - rollout()                                      (ticket 0.5.1)
#   - PAContext.from_pitches_row factory             (ticket 0.5.1; above)
#   - OutcomePredictor Protocol + 4 implementations  (ticket 0.5.2)
#   - OutcomePredictorRegistry                       (ticket 0.5.2)
#   - Aggregation utilities                          (ticket 0.5.3)
#
# Spec ambiguities resolved (flagged inline near each):
#
# (1) Predictor name "pg_concat_head" is NEW relative to SIM_ENGINE_API §3.4's
#     original literal enumeration.  Per PHASE_0.5_PLAN §2.0.5.2, A1 wins
#     Plan B and ships under this name.  Recorded in
#     ALLOWED_OUTCOME_PREDICTOR_NAMES below + surfaced in rollout() docstring.
#
# (2) WObaTable.default() uses the 7-element scalar table per
#     PHASE_0.5_PLAN §2.0.5.3 (final paragraph): ball/CS/SS/foul/in_play_out=0;
#     in_play_hit=0.892; HBP=0.708.  Phase-1 follow-up: full Statcast table.
#
# (3) The "in-zone" heuristic for outcome_predictor=None uses Statcast's 5x5
#     grid inner 3x3 indices.  Documented in _zone_is_in_strike_zone.
#
# (4) Mid-PA context mutation (Phase 0.6.1, 2026-08-04): the rollout seeds
#     the starting context at position 0 and then re-emits the ``count_state``
#     one-hot for each subsequent position from the running (balls, strikes)
#     per sample (see ``_advance_count`` + the sampling loop).  The
#     mutated context flows through the SAME encode path (count_state one-hot
#     at offset 0 of CONTEXT_DIM) into BOTH the pitch-token backbone forward
#     and the outcome-head forward, since they share ``cur_context``.  All
#     other context fields (outs / runners / hand / inning / score / ump) are
#     PA-invariant and stay constant, which is correct.  This closes the
#     static-context drift documented in
#     ``results/pitchgpt/rollout_sanity_2025/diagnose_static_context.md``.


import hashlib
import warnings
from typing import Callable, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.analytics.pitchgpt import (
    BOS_TOKEN,
    CONTEXT_DIM,
    NUM_COUNT_STATES,
    NUM_PITCH_TYPES,
    NUM_VELO_BUCKETS,
    NUM_ZONES,
    PAD_TOKEN,
    TOTAL_VOCAB,
    VOCAB_SIZE,
    PitchGPTModel,
    PitchTokenizer,
    _get_device,
)
from src.analytics.pitchgpt_outcome_head import (
    NUM_OUTCOME_CLASSES,
    OUTCOME_CLASSES,
    FrozenOutcomeHead,
    FrozenOutcomeHeadConcat,
    extract_backbone_hidden_states,
    freeze_backbone,
)


# ─────────────────────────────────────────────────────────────────────────────
# Outcome class indices (mirrors OUTCOME_CLASSES; module-level for readability)
# ─────────────────────────────────────────────────────────────────────────────

OUTCOME_BALL: int = 0
OUTCOME_CALLED_STRIKE: int = 1
OUTCOME_SWINGING_STRIKE: int = 2
OUTCOME_FOUL: int = 3
OUTCOME_IN_PLAY_OUT: int = 4
OUTCOME_IN_PLAY_HIT: int = 5
OUTCOME_HBP: int = 6

_STRIKE_OUTCOMES: frozenset[int] = frozenset(
    {OUTCOME_CALLED_STRIKE, OUTCOME_SWINGING_STRIKE}
)
_TERMINAL_INPLAY_OUTCOMES: frozenset[int] = frozenset(
    {OUTCOME_IN_PLAY_OUT, OUTCOME_IN_PLAY_HIT, OUTCOME_HBP}
)

# Inner 3x3 of the 5x5 zone grid.  Flat index = z_bin * 5 + x_bin.  Used by
# the count-only fallback to map zone-in-zone to "+1 strike".
_IN_ZONE_INDICES: frozenset[int] = frozenset(
    z_bin * 5 + x_bin
    for z_bin in (1, 2, 3)
    for x_bin in (1, 2, 3)
)

# ── Count-state mutation thresholds (Phase 0.6.1 mid-PA context mutation) ──
#: Ball count that ends the PA in a walk.
_WALK_BALLS: int = 4
#: Strike count that ends the PA in a strikeout.
_STRIKEOUT_STRIKES: int = 3
#: Foul balls never advance the strike count at or beyond this many strikes.
_FOUL_STRIKE_CAP: int = 2


def _advance_count(balls: int, strikes: int, outcome_class: int) -> tuple[int, int]:
    """Pure per-pitch count transition — Phase 0.6.1 mid-PA state machine.

    Given the pre-pitch ``(balls, strikes)`` and the sampled 7-class
    ``outcome_class``, return the post-pitch ``(balls, strikes)`` following
    baseball rules:

        * ``ball``                          → ``balls + 1``
        * ``called_strike`` / ``swinging_strike`` → ``strikes + 1``
        * ``foul``                          → ``strikes + 1`` **unless** already
          at ``_FOUL_STRIKE_CAP`` (2) strikes, in which case the count is
          unchanged (a foul with two strikes prolongs the PA without
          advancing).
        * ``in_play_out`` / ``in_play_hit`` / ``hbp`` (terminal) and any pad /
          out-of-range value → count **unchanged** (the caller ends the PA
          before ever advancing the count for these).

    This function does NOT decide termination.  The caller detects a walk via
    ``balls >= _WALK_BALLS`` and a strikeout via ``strikes >= _STRIKEOUT_STRIKES``
    on the returned count, and handles terminal in-play outcomes separately.

    Pure (no side effects); unit-tested directly in
    ``tests/test_pitchgpt_sim.py``.
    """
    if outcome_class == OUTCOME_BALL:
        return balls + 1, strikes
    if outcome_class in _STRIKE_OUTCOMES:
        return balls, strikes + 1
    if outcome_class == OUTCOME_FOUL:
        if strikes < _FOUL_STRIKE_CAP:
            return balls, strikes + 1
        return balls, strikes
    # Terminal in-play / HBP / pad / unknown: count unchanged.
    return balls, strikes


# Allowed outcome_predictor names.  NEW name "pg_concat_head" is added per
# PHASE_0.5_PLAN §2.0.5.2 — clarifying extension, not a contract violation.
ALLOWED_OUTCOME_PREDICTOR_NAMES: frozenset[str] = frozenset(
    {
        "none",
        "pg_concat_head",   # NEW — Plan B winner (A1)
        "pg_frozen_head",
        "xgboost",
        "empirical_pa_terminal",
    }
)


# Default model paths.
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_MODELS_DIR: Path = _PROJECT_ROOT / "models"
_BACKBONE_V2_PATH: Path = _MODELS_DIR / "pitchgpt_v2.pt"
_HEAD_CONCAT_A1_PATH: Path = _MODELS_DIR / "pitchgpt_v2_outcomehead_a1.pt"
_HEAD_FROZEN_DEPRECATED_PATH: Path = _MODELS_DIR / "pitchgpt_v2_outcomehead.pt"
_XGB_PATH: Path = _MODELS_DIR / "pitchgpt_outcome_xgb.bin"
_EMPIRICAL_LOOKUP_PATH: Path = (
    _MODELS_DIR / "pitchgpt_outcome_empirical_lookup.parquet"
)
_HEAD_CONCAT_A1_CALIB_PATH: Path = (
    # Phase 0.5 ticket 0.5.5 wrote this file with a leading
    # `calibration_` prefix; the §7.3 doc lists it with a
    # `_calibration` suffix.  Use the actual on-disk name so
    # PGConcatHeadPredictor picks up class_calibration / T from disk
    # (Phase 0.6, 2026-04-26).
    _MODELS_DIR / "calibration_pitchgpt_v2_outcomehead_a1.json"
)
_HEAD_FROZEN_DEPRECATED_CALIB_PATH: Path = (
    _MODELS_DIR / "pitchgpt_v2_outcomehead_calibration.json"
)
#: Position-0 class-marginal recalibration vector.  Saved by
#: ``scripts/pitchgpt_build_pos0_calibration.py`` as a 7-vector under the
#: ``class_calibration_pos0`` key.  Phase 0.6 fix (2026-04-26) -- closes
#: the +7.8pp `ball` under-emission at PA position 0 that the
#: static-context diagnostic surfaced as the dominant driver of the
#: Phase 0.6 BB% deficit.  Applied AFTER the existing JSON
#: ``class_calibration`` (which is fit on the OVERALL training-cohort
#: marginal -- this second-order vector targets pos-0 specifically).
#: Optional: missing file ⇒ identity (backwards-compat).
_HEAD_CONCAT_A1_POS0_CALIB_PATH: Path = (
    _MODELS_DIR / "calibration_class_marginal_pos0.npz"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _sha256_file(path: Path) -> str:
    """Stream-hash a file and return the hex SHA256 digest.

    Used to populate ``sampling_metadata['backbone_checkpoint_sha256']`` and
    ``...['outcome_predictor_checkpoint_sha256']``.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _decompose_pitch_token(
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompose a pitch token into (pitch_type_idx, zone_idx, velo_idx).

    Mirrors the canonical pattern in
    ``scripts/pitchgpt_outcome_a1_concat.py``: out-of-range tokens
    (pad/BOS) collapse to (0, 0, 0).
    """
    safe = torch.where(
        (tokens >= 0) & (tokens < VOCAB_SIZE),
        tokens,
        torch.zeros_like(tokens),
    )
    velo = safe % NUM_VELO_BUCKETS
    rem = safe // NUM_VELO_BUCKETS
    zone = rem % NUM_ZONES
    pt = rem // NUM_ZONES
    return pt, zone, velo


def _zone_is_in_strike_zone(zone_idx: int) -> bool:
    """True if the zone index is within the 3x3 inner strike zone.

    Used only by the count-only fallback (outcome_predictor=None).
    Per SIM_ENGINE_API §4.5 — zone-in-zone → +1 strike, else +1 ball.
    Approximate; consumers must surface that.
    """
    return int(zone_idx) in _IN_ZONE_INDICES


# ─────────────────────────────────────────────────────────────────────────────
# WObaTable
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class WObaTable:
    """Per-outcome wOBA scalar table.

    Phase 0.5 default is a 7-element table keyed on the 7-class outcome
    only (per PHASE_0.5_PLAN §2.0.5.3 + §5.4):

        ball/CS/SS/foul/in_play_out -> 0.0
        in_play_hit                 -> 0.892
        hbp                         -> 0.708

    Phase-1 follow-up: full Statcast empirical (outcome × pitch_type)
    table can be plugged in via ``WObaTable.from_statcast(...)``.  Phase
    0.5 ships the scalar default only.
    """

    values: tuple[float, ...]

    @classmethod
    def default(cls) -> "WObaTable":
        """The Phase 0.5 default 7-element scalar table.

        Per PHASE_0.5_PLAN §2.0.5.3 (final paragraph): values are
        league-average wOBA-per-outcome with hits collapsing to a
        single 0.892 (the hit-weighted league wOBA) and HBP at 0.708.
        Index follows ``OUTCOME_CLASSES`` order.
        """
        return cls(
            values=(
                0.0,    # ball
                0.0,    # called_strike
                0.0,    # swinging_strike
                0.0,    # foul
                0.0,    # in_play_out
                0.892,  # in_play_hit
                0.708,  # hbp
            ),
        )

    def lookup(self, outcome_class: int) -> float:
        """Return the wOBA value for a 7-class outcome, NaN for the pad sentinel."""
        if 0 <= int(outcome_class) < NUM_OUTCOME_CLASSES:
            return float(self.values[int(outcome_class)])
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# OutcomePredictor protocol
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class OutcomePredictor(Protocol):
    """Plug-in interface for 7-class outcome heads (SIM_ENGINE_API §4).

    Implementations MUST apply their own internal temperature scaling
    and return calibrated probabilities (not logits); MUST be
    deterministic given fixed inputs.
    """

    name: str
    checkpoint_sha256: str
    calibration: dict

    def predict_outcome_probs(
        self,
        backbone_hidden: torch.Tensor,
        context_vec: torch.Tensor,
        pitch_token: torch.Tensor,
    ) -> torch.Tensor:
        """Return calibrated 7-class probabilities, shape ``(B, S, 7)``."""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Backbone loader (singleton-cached)
# ─────────────────────────────────────────────────────────────────────────────


_BACKBONE_CACHE: dict[str, tuple[PitchGPTModel, str]] = {}


def _load_backbone(version: str = "v2") -> tuple[PitchGPTModel, str]:
    """Load a PitchGPT backbone checkpoint and return (model, sha256).

    Caches by version string.  Path rule per SIM_ENGINE_API §8.1:
    ``models/pitchgpt_<version>.pt``.  ``CONTEXT_DIM`` is inferred
    from the checkpoint's ``context_proj.weight`` shape.
    """
    if version in _BACKBONE_CACHE:
        return _BACKBONE_CACHE[version]

    path = _MODELS_DIR / f"pitchgpt_{version}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"Backbone checkpoint not found at {path}.  Per SIM_ENGINE_API "
            f"§8.1 the path rule is models/pitchgpt_<version>.pt."
        )
    sha = _sha256_file(path)
    device = _get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    cfg = checkpoint["config"]
    ctx_weight = checkpoint["model_state_dict"]["context_proj.weight"]
    context_dim = int(ctx_weight.shape[1])
    model = PitchGPTModel(
        vocab_size=cfg.get("vocab_size", TOTAL_VOCAB),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        max_seq_len=cfg["max_seq_len"],
        context_dim=context_dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    freeze_backbone(model)
    _BACKBONE_CACHE[version] = (model, sha)
    return model, sha


# ─────────────────────────────────────────────────────────────────────────────
# Predictor implementations
# ─────────────────────────────────────────────────────────────────────────────


class PGConcatHeadPredictor(nn.Module):
    """**PRODUCTION** outcome predictor — Plan B winner (A1, 2026-04-26).

    Wraps :class:`FrozenOutcomeHeadConcat` from
    ``src.analytics.pitchgpt_outcome_head``.  Input concat = backbone
    hidden state (128) + context vector (35) + pitch_type one-hot (17)
    + zone one-hot (26) + velocity one-hot (5) = 211d.

    Loads ``models/pitchgpt_v2_outcomehead_a1.pt`` and applies the
    locked temperature ``T = 0.8003``.

    Per SIM_ENGINE_API §4.4: ``predictor_kind = "pg_concat_head"``,
    ``T = 0.8003``, ``ECE_post = 0.0114``, ``holdout_season = 2025``.
    Note: ``"pg_concat_head"`` is NEW relative to the original
    SIM_ENGINE_API §3.4 enumeration -- a clarifying extension per
    PHASE_0.5_PLAN §2.0.5.2.

    Hot-path note: rollout loop calls
    :func:`extract_backbone_hidden_states` once per position and passes
    the per-position hidden state plus the just-sampled token; this
    predictor wraps the head forward + temperature scaling.
    """

    name: str = "pg_concat_head"

    def __init__(
        self,
        checkpoint_path: Path | str = _HEAD_CONCAT_A1_PATH,
        calibration_path: Path | str | None = None,
        require_calibration: bool = False,
        pos0_calibration_path: Path | str | None = None,
        disable_class_calibration: bool | None = None,
    ):
        """
        Parameters
        ----------
        checkpoint_path
            A1 head checkpoint (``models/pitchgpt_v2_outcomehead_a1.pt``).
        calibration_path
            JSON calibration (T + per-class re-weighting `class_calibration`,
            length-7).  Optional; default = sibling
            ``calibration_pitchgpt_v2_outcomehead_a1.json``.
        require_calibration
            If True, raise on missing/invalid calibration JSON; else fall
            back to embedded checkpoint metadata.
        disable_class_calibration
            **NEW (Phase 0.6.1, 2026-08-04).**  A/B switch for the all-position
            ``class_calibration`` re-weighting.  This vector was fit on the
            pre-mutation (static-context) rollout, so after the mid-PA
            count-state mutation lands it may over-correct the per-pitch
            marginal.  When True, the loaded ``class_calibration`` is dropped
            (``_class_calibration = None``) so the head returns the pure
            temperature-scaled softmax.  The position-0 recalibration
            (``pos0_calibration_path``) is INTENTIONALLY retained — it was fit
            on the always-correct position-0 context and stays valid.
            Default ``None`` reads the ``PITCHGPT_DISABLE_CLASS_CALIBRATION``
            environment variable (unset/false ⇒ existing behavior preserved),
            letting the orchestrator A/B the sanity run without editing any
            script.
        pos0_calibration_path
            **NEW (Phase 0.6, 2026-04-26).**  Optional npz path containing
            a ``class_calibration_pos0`` length-7 vector that recalibrates
            the position-0 class marginal.  This calibration is applied
            in ``predict_outcome_probs`` AFTER the JSON
            ``class_calibration``: ``p_i ← p_i * w_pos0_i / sum(p * w_pos0)``.
            Built by ``scripts/pitchgpt_build_pos0_calibration.py`` from the
            2025 pitcher-disjoint cohort's empirical pos-0 marginal divided
            by the rollout's H=1 pos-0 marginal (post-existing-calibration).
            Default ``None`` resolves to the sibling
            ``calibration_class_marginal_pos0.npz``; missing file silently
            no-ops (identity).  Closes the Phase 0.6 BB% deficit by
            restoring the +7.8pp `ball` under-emission at PA-start.
        """
        super().__init__()
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"PGConcatHeadPredictor checkpoint missing at {path}."
            )
        self.checkpoint_path = path
        self.checkpoint_sha256 = _sha256_file(path)

        device = _get_device()
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        self.head = FrozenOutcomeHeadConcat(
            d_model=cfg["d_model"],
            context_dim=cfg["context_dim"],
            n_pitch_types=cfg["n_pitch_types"],
            n_zones=cfg["n_zones"],
            n_velo_buckets=cfg["n_velo_buckets"],
            hidden_dims=tuple(cfg["hidden_dims"]),
            dropout=cfg["dropout"],
            n_classes=cfg["n_classes"],
        )
        self.head.load_state_dict(ckpt["head_state_dict"])
        self.head.to(device)
        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad = False

        # Locked temperature from the A1 checkpoint metadata (= 0.8003).
        self.temperature = float(ckpt["temperature"])
        self._device = device
        self._n_pitch_types = cfg["n_pitch_types"]
        self._n_zones = cfg["n_zones"]
        self._n_velo_buckets = cfg["n_velo_buckets"]

        # Phase 0.5: do NOT require the JSON — Agent 2 writes it in
        # ticket 0.5.5.  Fall back to embedded checkpoint metadata so
        # downstream consumers always see post-T values.
        self.calibration: dict = {
            "T": self.temperature,
            "ECE_pre": float(
                ckpt["test_metrics"].get("ece_pre_temp", float("nan"))
            ),
            "ECE_post": float(
                ckpt["test_metrics"].get("ece_post_temp", float("nan"))
            ),
            "holdout_season": 2025,
            "holdout_n_pitches": int(
                ckpt.get("paired_vs_a3", {}).get("n_a1", 0)
            ) or 204513,
            "fit_date": "2026-04-25",
            "checkpoint_sha256": self.checkpoint_sha256,
            "predictor_kind": "pg_concat_head",
            "source": "embedded_checkpoint_metadata",
        }
        cpath = (
            Path(calibration_path)
            if calibration_path is not None
            else _HEAD_CONCAT_A1_CALIB_PATH
        )
        if cpath.exists():
            try:
                self.calibration = load_calibration_json(
                    cpath,
                    expected_predictor_kind="pg_concat_head",
                    ece_budget=_ECE_BUDGET_OUTCOME_PREDICTOR,
                )
            except CalibrationError:
                if require_calibration:
                    raise
                logger.warning(
                    "PGConcatHeadPredictor: calibration JSON at %s "
                    "exists but is invalid; falling back to embedded "
                    "checkpoint metadata.",
                    cpath,
                )
        elif require_calibration:
            raise CalibrationError(
                f"PGConcatHeadPredictor: calibration JSON missing at "
                f"{cpath} and require_calibration=True."
            )

        # Phase 0.6 (2026-04-26): post-hoc per-class probability
        # re-weighting to close A1's class-marginal bias (under-predicts
        # `ball` by 11.6pp, over-predicts `swinging_strike` by 8.6pp;
        # see PHASE_0.6_DIAGNOSIS.md).  Optional: missing/None ⇒
        # identity (backwards-compat with prior calibration JSONs).
        # Length-7 vector indexed by OUTCOME_CLASSES order:
        # [ball, called_strike, swinging_strike, foul, in_play_out,
        #  in_play_hit, hbp].  Applied AFTER temperature scaling and
        # softmax, BEFORE return: p_i = p_i * w_i / sum_j(p_j * w_j).
        # Top-1 ECE is unaffected (top-1 reliability is independent of
        # class-marginal balance); per-PA K%/BB% are restored.
        cc = self.calibration.get("class_calibration", None)
        if cc is None:
            self._class_calibration: torch.Tensor | None = None
        else:
            cc_arr = np.asarray(cc, dtype=np.float32)
            if cc_arr.shape != (7,):
                raise CalibrationError(
                    f"PGConcatHeadPredictor: class_calibration has "
                    f"shape {cc_arr.shape}; expected (7,) matching "
                    f"OUTCOME_CLASSES."
                )
            if not np.all(np.isfinite(cc_arr)) or np.any(cc_arr <= 0):
                raise CalibrationError(
                    f"PGConcatHeadPredictor: class_calibration must "
                    f"be all-positive finite floats; got {cc_arr.tolist()}"
                )
            self._class_calibration = torch.from_numpy(cc_arr).to(device)

        # Phase 0.6.1 (2026-08-04): A/B switch to DISABLE the all-position
        # class_calibration.  It was fit on the pre-mutation static-context
        # rollout; after the mid-PA count-state mutation it may over-correct
        # the per-pitch marginal (empirically it inflates PA-level K%).
        # Dropping it here yields the pure temperature-scaled softmax while
        # KEEPING the position-0 recalibration (which was fit on the
        # always-correct pos-0 context).  Default None ⇒ read the
        # PITCHGPT_DISABLE_CLASS_CALIBRATION env var so the orchestrator can
        # A/B the sanity run with no script edits.
        if disable_class_calibration is None:
            disable_class_calibration = _env_truthy(
                "PITCHGPT_DISABLE_CLASS_CALIBRATION"
            )
        self.class_calibration_disabled: bool = bool(disable_class_calibration)
        if self.class_calibration_disabled:
            self._class_calibration = None
            self.calibration = dict(self.calibration)
            self.calibration["class_calibration_disabled"] = True
            logger.info(
                "PGConcatHeadPredictor: class_calibration DISABLED "
                "(Phase 0.6.1 A/B); pos-0 recalibration retained."
            )

        # Phase 0.6 (2026-04-26): position-0 class-marginal
        # recalibration.  Applied AFTER the JSON `class_calibration` to
        # close the +7.8pp `ball` under-emission at PA position 0
        # (the dominant driver of the Phase 0.6 BB% gate FAIL per
        # `results/pitchgpt/rollout_sanity_2025/diagnose_static_context.md`).
        # Missing npz ⇒ identity (backwards-compat: prior calibration
        # JSONs without a pos-0 npz still work).
        ppath = (
            Path(pos0_calibration_path)
            if pos0_calibration_path is not None
            else _HEAD_CONCAT_A1_POS0_CALIB_PATH
        )
        if ppath.exists():
            try:
                npz = np.load(ppath, allow_pickle=False)
                pos0 = np.asarray(
                    npz["class_calibration_pos0"], dtype=np.float32
                )
                if pos0.shape != (7,):
                    raise CalibrationError(
                        f"PGConcatHeadPredictor: class_calibration_pos0 "
                        f"in {ppath} has shape {pos0.shape}; expected "
                        f"(7,) matching OUTCOME_CLASSES."
                    )
                if not np.all(np.isfinite(pos0)) or np.any(pos0 <= 0):
                    raise CalibrationError(
                        f"PGConcatHeadPredictor: class_calibration_pos0 "
                        f"must be all-positive finite floats; got "
                        f"{pos0.tolist()}"
                    )
                self._class_calibration_pos0: torch.Tensor | None = (
                    torch.from_numpy(pos0).to(device)
                )
                logger.info(
                    "PGConcatHeadPredictor: loaded pos-0 class calibration "
                    "from %s (vector=%s)",
                    ppath, np.array2string(pos0, precision=4),
                )
            except CalibrationError:
                raise
            except Exception as exc:  # corrupt npz / missing key
                logger.warning(
                    "PGConcatHeadPredictor: pos-0 class-calibration npz at "
                    "%s exists but failed to load (%s); using identity.",
                    ppath, exc,
                )
                self._class_calibration_pos0 = None
        else:
            self._class_calibration_pos0 = None

    @torch.no_grad()
    def predict_outcome_probs(
        self,
        backbone_hidden: torch.Tensor,
        context_vec: torch.Tensor,
        pitch_token: torch.Tensor,
    ) -> torch.Tensor:
        """Return calibrated 7-class probabilities, shape ``(B, S, 7)``.

        Calibration application order (Phase 0.6, 2026-04-26):
          1. Logits / T  (T = locked checkpoint temperature)
          2. Softmax
          3. ``class_calibration`` (overall training-cohort marginal
             match, fit on 2023 val)
          4. ``class_calibration_pos0`` (PA-start pos-0 marginal match,
             fit on 2025 val) -- closes +7.8pp `ball` under-emission at
             PA position 0.

        Each multiplicative step is followed by per-row renormalization
        on the last axis.  Missing pos-0 calibration ⇒ identity.
        """
        pt_idx, zone_idx, velo_idx = _decompose_pitch_token(pitch_token)
        pt_oh = F.one_hot(pt_idx, num_classes=self._n_pitch_types).float()
        zn_oh = F.one_hot(zone_idx, num_classes=self._n_zones).float()
        vl_oh = F.one_hot(velo_idx, num_classes=self._n_velo_buckets).float()
        logits = self.head(backbone_hidden, context_vec, pt_oh, zn_oh, vl_oh)
        scaled = logits / max(self.temperature, 1e-8)
        probs = F.softmax(scaled, dim=-1)
        if self._class_calibration is not None:
            # Element-wise re-weight + renormalize on the last (class)
            # axis.  Broadcasts (..., 7) * (7,) -> (..., 7).
            weighted = probs * self._class_calibration
            denom = weighted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            probs = weighted / denom
        if self._class_calibration_pos0 is not None:
            # Phase 0.6 pos-0 recalibration: same broadcast pattern as
            # the JSON class_calibration.  Stacks multiplicatively;
            # mathematically equivalent to a single combined vector
            # (kept separate so they can be retired/refit
            # independently).
            weighted = probs * self._class_calibration_pos0
            denom = weighted.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            probs = weighted / denom
        return probs


class PGFrozenHeadPredictor(nn.Module):
    """**DEPRECATED** outcome predictor — Phase 0.3 frozen-head MLP.

    Per SIM_ENGINE_API §4.1.  Kept registered for backtest replay of
    the Phase 0.3 −5.34% FAIL artifact, NOT for production.  Wraps
    :class:`FrozenOutcomeHead` and loads
    ``models/pitchgpt_v2_outcomehead.pt``.

    Inputs ignored: ``context_vec``, ``pitch_token`` -- this head
    consumes only ``backbone_hidden``.  Hence it produces
    ``p(outcome | prefix)`` rather than ``p(outcome | token, prefix)``
    and is unsuitable for token-conditioned counterfactuals.
    """

    name: str = "pg_frozen_head"

    def __init__(
        self,
        checkpoint_path: Path | str = _HEAD_FROZEN_DEPRECATED_PATH,
        calibration_path: Path | str | None = None,
        require_calibration: bool = False,
    ):
        super().__init__()
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"PGFrozenHeadPredictor checkpoint missing at {path}."
            )
        self.checkpoint_path = path
        self.checkpoint_sha256 = _sha256_file(path)
        device = _get_device()

        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
        d_model = cfg.get("d_model", 128)
        hidden_dim = cfg.get("hidden_dim", 64)
        dropout = cfg.get("dropout", 0.1)
        n_classes = cfg.get("n_classes", NUM_OUTCOME_CLASSES)
        self.head = FrozenOutcomeHead(
            d_model=d_model,
            hidden_dim=hidden_dim,
            dropout=dropout,
            n_classes=n_classes,
        )
        sd = (
            ckpt.get("head_state_dict")
            or ckpt.get("model_state_dict")
            or ckpt.get("state_dict")
        )
        if sd is None:
            raise RuntimeError(
                f"PGFrozenHeadPredictor: no recognised state_dict key in "
                f"{path}; found keys {list(ckpt.keys())}"
            )
        self.head.load_state_dict(sd)
        self.head.to(device)
        self.head.eval()
        for p in self.head.parameters():
            p.requires_grad = False
        self.temperature = float(ckpt.get("temperature", 1.0))
        self._device = device

        self.calibration: dict = {
            "T": self.temperature,
            "predictor_kind": "pg_frozen_head",
            "source": "embedded_checkpoint_metadata",
            "deprecated": True,
        }
        cpath = (
            Path(calibration_path)
            if calibration_path is not None
            else _HEAD_FROZEN_DEPRECATED_CALIB_PATH
        )
        if cpath.exists():
            try:
                self.calibration = load_calibration_json(
                    cpath,
                    expected_predictor_kind="pg_frozen_head",
                    ece_budget=_ECE_BUDGET_OUTCOME_PREDICTOR,
                )
            except CalibrationError:
                if require_calibration:
                    raise
                logger.warning(
                    "PGFrozenHeadPredictor: calibration JSON at %s "
                    "invalid; using embedded metadata (deprecated path).",
                    cpath,
                )

    @torch.no_grad()
    def predict_outcome_probs(
        self,
        backbone_hidden: torch.Tensor,
        context_vec: torch.Tensor,
        pitch_token: torch.Tensor,
    ) -> torch.Tensor:
        # Frozen-head ignores context_vec and pitch_token.
        logits = self.head(backbone_hidden)
        scaled = logits / max(self.temperature, 1e-8)
        return F.softmax(scaled, dim=-1)


class XGBoostOutcomePredictor:
    """**BACKSTOP** Plan B A3 candidate.  Conditionally registered only.

    Per SIM_ENGINE_API §4.2.  Loads
    ``models/pitchgpt_outcome_xgb.bin`` IF PRESENT.  As of 2026-04-26
    the checkpoint does not exist (A3 lost to A1 by +2.48pp paired and
    was not promoted).  The registry registration check fires at
    module-import time -- consumers requesting this predictor get a
    clean ``KeyError`` if the checkpoint is missing.

    Inputs used: ``context_vec`` and the decomposed ``pitch_token``
    components; ``backbone_hidden`` is ignored.
    """

    name: str = "xgboost"

    def __init__(
        self,
        checkpoint_path: Path | str = _XGB_PATH,
        calibration_path: Path | str | None = None,
        require_calibration: bool = False,
    ):
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(
                f"XGBoostOutcomePredictor: checkpoint missing at {path}."
            )
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "XGBoostOutcomePredictor requires xgboost; install it "
                "or use a different predictor."
            ) from exc
        self.checkpoint_path = path
        self.checkpoint_sha256 = _sha256_file(path)
        self._booster = xgb.Booster()
        self._booster.load_model(str(path))
        self.temperature = 1.0

        self.calibration: dict = {
            "T": 1.0,
            "predictor_kind": "xgboost",
            "source": "stub_default",
        }
        cpath = (
            Path(calibration_path) if calibration_path is not None
            else _MODELS_DIR / "pitchgpt_outcome_xgb_calibration.json"
        )
        if cpath.exists():
            try:
                self.calibration = load_calibration_json(
                    cpath,
                    expected_predictor_kind="xgboost",
                    ece_budget=_ECE_BUDGET_OUTCOME_PREDICTOR,
                )
                self.temperature = float(self.calibration["T"])
            except CalibrationError:
                if require_calibration:
                    raise

        self._pitcher_id: int = 0
        self._batter_id: int = 0

    def set_ids(self, pitcher_id: int, batter_id: int) -> None:
        """Set per-rollout pitcher/batter IDs (per SIM_ENGINE_API §4.2)."""
        self._pitcher_id = int(pitcher_id)
        self._batter_id = int(batter_id)

    def predict_outcome_probs(
        self,
        backbone_hidden: torch.Tensor,
        context_vec: torch.Tensor,
        pitch_token: torch.Tensor,
    ) -> torch.Tensor:
        pt_idx, zone_idx, velo_idx = _decompose_pitch_token(pitch_token)
        B, S = pitch_token.shape
        ctx_np = context_vec.detach().cpu().numpy().reshape(B * S, -1)
        pt_np = pt_idx.detach().cpu().numpy().reshape(-1, 1)
        zn_np = zone_idx.detach().cpu().numpy().reshape(-1, 1)
        vl_np = velo_idx.detach().cpu().numpy().reshape(-1, 1)
        ids_np = np.full(
            (B * S, 2),
            [self._pitcher_id, self._batter_id],
            dtype=np.float32,
        )
        feats = np.concatenate([ctx_np, pt_np, zn_np, vl_np, ids_np], axis=1)
        import xgboost as xgb
        dmat = xgb.DMatrix(feats)
        probs = self._booster.predict(dmat)
        if self.temperature != 1.0:
            logp = np.log(np.clip(probs, 1e-12, 1.0))
            scaled = logp / max(self.temperature, 1e-8)
            ex = np.exp(scaled - scaled.max(axis=1, keepdims=True))
            probs = ex / ex.sum(axis=1, keepdims=True)
        out = torch.from_numpy(probs.reshape(B, S, 7)).to(
            device=pitch_token.device, dtype=torch.float32,
        )
        return out


class EmpiricalPATerminalLookup:
    """**KILL-CRIT FALLBACK** non-parametric Plan B A5 baseline.

    Per SIM_ENGINE_API §4.3.  Loads
    ``models/pitchgpt_outcome_empirical_lookup.parquet`` IF PRESENT.
    As of 2026-04-26 the table does not exist on disk; the registry
    registration check fires at module-import time -- consumers
    requesting this predictor get a ``KeyError`` if the parquet is
    missing.

    Phase 0.5 ships construction-only; the actual lookup with smoothed
    fallback chain is Phase-1 work.  Calling
    :meth:`predict_outcome_probs` raises ``NotImplementedError`` so
    consumers cannot silently consume garbage.
    """

    name: str = "empirical_pa_terminal"

    def __init__(
        self,
        lookup_path: Path | str = _EMPIRICAL_LOOKUP_PATH,
        calibration_path: Path | str | None = None,
        require_calibration: bool = False,
    ):
        path = Path(lookup_path)
        if not path.exists():
            raise FileNotFoundError(
                f"EmpiricalPATerminalLookup: parquet missing at {path}."
            )
        self.lookup_path = path
        self.checkpoint_sha256 = _sha256_file(path)
        try:
            import pyarrow.parquet as pq  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "EmpiricalPATerminalLookup requires pyarrow."
            ) from exc
        import pandas as pd
        self._table = pd.read_parquet(path)
        self.temperature = 1.0
        self.calibration: dict = {
            "T": 1.0,
            "predictor_kind": "empirical_pa_terminal",
            "source": "stub_default",
        }
        cpath = (
            Path(calibration_path) if calibration_path is not None
            else _MODELS_DIR / "pitchgpt_outcome_empirical_lookup_calibration.json"
        )
        if cpath.exists():
            try:
                self.calibration = load_calibration_json(
                    cpath,
                    expected_predictor_kind="empirical_pa_terminal",
                    ece_budget=_ECE_BUDGET_OUTCOME_PREDICTOR,
                )
            except CalibrationError:
                if require_calibration:
                    raise

    def predict_outcome_probs(
        self,
        backbone_hidden: torch.Tensor,
        context_vec: torch.Tensor,
        pitch_token: torch.Tensor,
    ) -> torch.Tensor:
        # Phase 0.5: registration-only stub.  Smoothed-lookup-with-
        # fallback-chain implementation lands alongside the actual
        # parquet build in the kill-criterion path.
        raise NotImplementedError(
            "EmpiricalPATerminalLookup.predict_outcome_probs is a "
            "Phase-1 implementation deliverable.  The Phase 0.5 stub "
            "validates registry conditional-registration only."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Predictor registry
# ─────────────────────────────────────────────────────────────────────────────


class OutcomePredictorRegistry:
    """Name → factory dispatch for outcome predictors (SIM_ENGINE_API §8.2).

    Default registrations are loaded at module import.  Conditional
    registrations (XGBoost, Empirical) ONLY register if the checkpoint
    exists on disk; consumers requesting an unregistered predictor get
    a clean ``KeyError`` from :meth:`get`.
    """

    _factories: dict[str, Callable[[], "OutcomePredictor"]] = {}

    @classmethod
    def get(cls, name: str) -> "OutcomePredictor":
        if name not in cls._factories:
            raise KeyError(
                f"OutcomePredictorRegistry: no predictor named {name!r}.  "
                f"Registered: {cls.list_registered()}"
            )
        return cls._factories[name]()

    @classmethod
    def register(
        cls,
        name: str,
        factory: Callable[[], "OutcomePredictor"],
    ) -> None:
        cls._factories[name] = factory

    @classmethod
    def list_registered(cls) -> list[str]:
        return sorted(cls._factories.keys())

    @classmethod
    def deregister(cls, name: str) -> None:
        cls._factories.pop(name, None)


def _register_default_predictors() -> None:
    """Register the 4 default predictors per SIM_ENGINE_API §8.2.

    Always-registered (production / replay):
      - ``"pg_concat_head"`` (PRODUCTION)
      - ``"pg_frozen_head"`` (DEPRECATED replay)

    Conditionally-registered (only if checkpoint exists on disk):
      - ``"xgboost"``
      - ``"empirical_pa_terminal"``
    """
    if _HEAD_CONCAT_A1_PATH.exists():
        OutcomePredictorRegistry.register(
            "pg_concat_head",
            lambda: PGConcatHeadPredictor(
                checkpoint_path=_HEAD_CONCAT_A1_PATH,
            ),
        )
    if _HEAD_FROZEN_DEPRECATED_PATH.exists():
        OutcomePredictorRegistry.register(
            "pg_frozen_head",
            lambda: PGFrozenHeadPredictor(
                checkpoint_path=_HEAD_FROZEN_DEPRECATED_PATH,
            ),
        )
    if _XGB_PATH.exists():
        OutcomePredictorRegistry.register(
            "xgboost",
            lambda: XGBoostOutcomePredictor(checkpoint_path=_XGB_PATH),
        )
    if _EMPIRICAL_LOOKUP_PATH.exists():
        OutcomePredictorRegistry.register(
            "empirical_pa_terminal",
            lambda: EmpiricalPATerminalLookup(
                lookup_path=_EMPIRICAL_LOOKUP_PATH,
            ),
        )


_register_default_predictors()


# ─────────────────────────────────────────────────────────────────────────────
# Rollout entry point  (ticket 0.5.1)
# ─────────────────────────────────────────────────────────────────────────────


def _build_starting_tokens_and_context(
    starting_context: PAContext,
    backbone_context_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the (initial tokens, initial context, prefix_len) tensors.

    Decoder-only backbone with a BOS token at position 0; prefix tokens
    follow.  This builds only the position-0 (starting) context; the rollout
    loop then mutates the ``count_state`` one-hot per subsequent position from
    the running count (Phase 0.6.1 mid-PA context mutation).  The remaining
    context fields (outs / runners / hand / inning / score / ump) are
    PA-invariant and are correctly broadcast unchanged.

    Returns
    -------
    tokens : (1, prefix_len + 1) int64 — BOS followed by prefix tokens.
    context : (1, prefix_len + 1, CONTEXT_DIM) float32.
    prefix_len : int — number of pitches already in the PA before
        rollout begins.
    """
    balls, strikes = starting_context.count
    on_1b, on_2b, on_3b = starting_context.runners

    ctx_list = PitchTokenizer.encode_context(
        balls=balls,
        strikes=strikes,
        outs=starting_context.outs,
        on_1b=on_1b, on_2b=on_2b, on_3b=on_3b,
        stand=starting_context.batter_stand,
        inning=starting_context.inning,
        score_diff=starting_context.score_diff,
    )
    ctx_vec = PitchTokenizer.context_to_tensor(
        ctx_list, ump_scalar=starting_context.umpire_scalar,
    )
    if backbone_context_dim < ctx_vec.shape[-1]:
        ctx_vec = ctx_vec[: backbone_context_dim]
    elif backbone_context_dim > ctx_vec.shape[-1]:
        pad = torch.zeros(backbone_context_dim - ctx_vec.shape[-1])
        ctx_vec = torch.cat([ctx_vec, pad])

    prefix = list(starting_context.prefix_pitch_tokens)
    # Validate prefix tokens (PAD / BOS / out-of-range forbidden per §3.2).
    for tok in prefix:
        if not (0 <= int(tok) < VOCAB_SIZE):
            raise ValueError(
                f"PAContext.prefix_pitch_tokens contains out-of-range or "
                f"PAD/BOS token {tok}; valid range is [0, {VOCAB_SIZE})"
            )
    tokens = [BOS_TOKEN] + prefix
    tokens_t = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    context_t = ctx_vec.unsqueeze(0).unsqueeze(0).expand(
        1, tokens_t.shape[1], -1
    ).contiguous()
    return tokens_t, context_t, len(prefix)


def _build_metadata(
    *,
    temperature: float,
    seed: int | None,
    n_samples: int,
    horizon: int,
    backbone_version: str,
    backbone_sha: str,
    outcome_predictor_name: str,
    outcome_predictor_sha: str | None,
    calibration_valid: bool,
    calibration_invalid_reasons: list[str],
    n_truncated: int,
) -> dict:
    """Build the ``sampling_metadata`` dict per SIM_ENGINE_API §3.4.

    Every required key MUST be present.
    """
    return {
        "temperature": float(temperature),
        "seed": (int(seed) if seed is not None else None),
        "n_samples": int(n_samples),
        "horizon": int(horizon),
        "backbone_version": str(backbone_version),
        "backbone_checkpoint_sha256": str(backbone_sha),
        "outcome_predictor": str(outcome_predictor_name),
        "outcome_predictor_checkpoint_sha256": (
            str(outcome_predictor_sha)
            if outcome_predictor_sha is not None else None
        ),
        "calibration_valid": bool(calibration_valid),
        "calibration_invalid_reasons": list(calibration_invalid_reasons),
        "rollout_engine_version": ROLLOUT_ENGINE_VERSION,
        "n_truncated": int(n_truncated),
    }


def _load_backbone_calibration(version: str) -> dict | None:
    """Best-effort load of ``models/pitchgpt_<version>_calibration.json``.

    Returns None on missing/unreadable.  The calibration validity gate
    treats None as "stale" via :func:`_calibration_is_stale`.  This is
    distinct from the predictor-construction-time refusal in
    :func:`load_calibration_json`: here we accept missing because Agent
    2's Phase-0.5 work to write that file may be in progress.
    """
    p = _MODELS_DIR / f"pitchgpt_{version}_calibration.json"
    if not p.exists():
        return None
    try:
        with p.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def rollout(
    starting_context: PAContext,
    *,
    backbone_version: str = "v2",
    outcome_predictor: OutcomePredictor | None = None,
    n_samples: int = 100,
    horizon: int = 6,
    temperature: float = 1.0,
    seed: int | None = None,
    return_probs: bool = True,
) -> RolloutResult:
    """Sample ``n_samples`` independent PA continuations from a starting state.

    See SIM_ENGINE_API §3.1 for full semantics.  Edge cases enforced:

    - ``n_samples == 0`` → ``ValueError``.
    - ``horizon == 0`` → empty :class:`RolloutResult` (shape (n_samples, 0)).
    - ``horizon > 12`` → WARNING + ``calibration_valid = False`` with
      reason ``"horizon_exceeds_validated"``.
    - ``temperature != 1.0`` → WARNING + ``calibration_valid = False``
      with reason ``"temperature_not_unity"``.
    - ``starting_context.prefix_pitch_tokens`` longer than ``horizon``
      → ``ValueError`` (the prefix already consumes the budget).
    - ``starting_context.prefix_pitch_tokens`` containing PAD/BOS/
      out-of-range tokens → ``ValueError``.

    PA-termination logic (per §3.3) — first match wins per position:
        1. Outcome-driven (only when predictor is not None): if outcome
           ∈ {in_play_out, in_play_hit, hbp}, terminate at this position.
        2. Walk: balls reaches 4.
        3. Strikeout: strikes reaches 3.  Foul on 0–1 strikes counts;
           foul on 2 strikes does not advance.
        4. Horizon exhaustion at position ``horizon - 1`` → truncated.

    Allowed ``outcome_predictor`` literal names (sampling_metadata):
        ``"none" | "pg_concat_head" | "pg_frozen_head" | "xgboost" |
        "empirical_pa_terminal"``

    Note: ``"pg_concat_head"`` is a NEW name relative to the original
    SIM_ENGINE_API §3.4 enumeration; see PHASE_0.5_PLAN §2.0.5.2.  It
    is the production A1 head as of 2026-04-26.

    Note (Phase 0.6.1, 2026-08-04): the rollout MUTATES the ``count_state``
    one-hot of the context vector at each position from the running
    (balls, strikes) of each sample (per ``_advance_count``), so both the
    pitch-token backbone forward and the outcome-head forward see the updated
    count at position t+1.  All other context fields are PA-invariant and stay
    constant.  This closes the static-context drift diagnosed in
    ``results/pitchgpt/rollout_sanity_2025/diagnose_static_context.md``.
    """
    # ── Edge cases (§3.1) ────────────────────────────────────────────────
    if int(n_samples) <= 0:
        raise ValueError(
            f"rollout(): n_samples must be > 0, got {n_samples}"
        )
    n_samples = int(n_samples)
    horizon = int(horizon)
    if horizon < 0:
        raise ValueError(f"rollout(): horizon must be >= 0, got {horizon}")
    if len(starting_context.prefix_pitch_tokens) > horizon:
        raise ValueError(
            f"rollout(): prefix_pitch_tokens length "
            f"{len(starting_context.prefix_pitch_tokens)} exceeds horizon "
            f"{horizon}; the prefix already consumes the budget."
        )

    if abs(float(temperature) - 1.0) > 1e-9:
        warnings.warn(
            f"rollout(): temperature={temperature} != 1.0 — calibration "
            f"is fit at T=1.0 so calibration_valid will be False.",
            UserWarning,
            stacklevel=2,
        )
    if horizon > _HORIZON_VALIDATED_MAX:
        warnings.warn(
            f"rollout(): horizon={horizon} > {_HORIZON_VALIDATED_MAX} — "
            f"exposure-bias drift is unmeasured beyond H=6 and "
            f"calibration_valid will be False.",
            UserWarning,
            stacklevel=2,
        )

    # ── Backbone load + SHA ─────────────────────────────────────────────
    backbone, backbone_sha = _load_backbone(backbone_version)
    device = next(backbone.parameters()).device

    # ── Predictor metadata ─────────────────────────────────────────────
    pred_name = "none"
    pred_sha: str | None = None
    pred_calibration: dict | None = None
    if outcome_predictor is not None:
        pred_name = getattr(outcome_predictor, "name", "unknown")
        pred_sha = getattr(outcome_predictor, "checkpoint_sha256", None)
        pred_calibration = getattr(outcome_predictor, "calibration", None)
        if pred_name not in ALLOWED_OUTCOME_PREDICTOR_NAMES:
            warnings.warn(
                f"rollout(): outcome_predictor.name={pred_name!r} is not "
                f"in the allowed-names enumeration "
                f"{sorted(ALLOWED_OUTCOME_PREDICTOR_NAMES)}; treating as "
                f"custom.",
                UserWarning,
                stacklevel=2,
            )

    # ── Calibration-valid gate (Agent 2 ticket 0.5.4 helper) ────────────
    backbone_cal = _load_backbone_calibration(backbone_version)
    cal_valid, cal_reasons = _check_calibration_valid(
        starting_context=starting_context,
        backbone_calibration=backbone_cal,
        predictor_calibration=pred_calibration,
        temperature=temperature,
        horizon=horizon,
    )

    # ── Empty-horizon edge case ────────────────────────────────────────
    if horizon == 0:
        empty_pitch = np.empty((n_samples, 0), dtype=np.int64)
        empty_probs_pitch = (
            np.empty((n_samples, 0, VOCAB_SIZE), dtype=np.float32)
            if return_probs else None
        )
        empty_outc = (
            np.empty((n_samples, 0), dtype=np.int64)
            if outcome_predictor is not None else None
        )
        empty_outc_probs = (
            np.empty((n_samples, 0, NUM_OUTCOME_CLASSES), dtype=np.float32)
            if (outcome_predictor is not None and return_probs) else None
        )
        empty_term = np.zeros((n_samples, 0), dtype=bool)
        empty_pa_outc = (
            np.full((n_samples,), ROLLOUT_PAD_OUTCOME, dtype=np.int64)
            if outcome_predictor is not None else None
        )
        empty_count = np.tile(
            np.array(starting_context.count, dtype=np.int64),
            (n_samples, 1),
        )
        meta = _build_metadata(
            temperature=temperature,
            seed=seed,
            n_samples=n_samples,
            horizon=horizon,
            backbone_version=backbone_version,
            backbone_sha=backbone_sha,
            outcome_predictor_name=pred_name,
            outcome_predictor_sha=pred_sha,
            calibration_valid=cal_valid,
            calibration_invalid_reasons=cal_reasons,
            n_truncated=n_samples,
        )
        return RolloutResult(
            pitch_tokens=empty_pitch,
            pitch_probs=empty_probs_pitch,
            outcomes=empty_outc,
            outcome_probs=empty_outc_probs,
            pa_terminated=empty_term,
            pa_outcome=empty_pa_outc,
            final_count=empty_count,
            sampling_metadata=meta,
        )

    # ── RNG ────────────────────────────────────────────────────────────
    if seed is None:
        torch_gen = None
    else:
        if device.type == "cuda":
            torch_gen = torch.Generator(device=device)
        else:
            torch_gen = torch.Generator()
        torch_gen.manual_seed(int(seed) & 0xFFFFFFFFFFFFFFFF)

    # ── Allocate output buffers ────────────────────────────────────────
    pitch_tokens_out = np.full(
        (n_samples, horizon), ROLLOUT_PAD_PITCH, dtype=np.int64,
    )
    pitch_probs_out: np.ndarray | None
    if return_probs:
        pitch_probs_out = np.full(
            (n_samples, horizon, VOCAB_SIZE), np.nan, dtype=np.float32,
        )
    else:
        pitch_probs_out = None

    if outcome_predictor is not None:
        outcomes_out: np.ndarray | None = np.full(
            (n_samples, horizon), ROLLOUT_PAD_OUTCOME, dtype=np.int64,
        )
        if return_probs:
            outcome_probs_out: np.ndarray | None = np.full(
                (n_samples, horizon, NUM_OUTCOME_CLASSES),
                np.nan, dtype=np.float32,
            )
        else:
            outcome_probs_out = None
        pa_outcome_out: np.ndarray | None = np.full(
            (n_samples,), ROLLOUT_PAD_OUTCOME, dtype=np.int64,
        )
    else:
        outcomes_out = None
        outcome_probs_out = None
        pa_outcome_out = None

    pa_terminated_out = np.zeros((n_samples, horizon), dtype=bool)
    final_count_out = np.tile(
        np.array(starting_context.count, dtype=np.int64), (n_samples, 1)
    )

    # ── Build initial token + context tensors ──────────────────────────
    init_tokens, init_context, prefix_len = _build_starting_tokens_and_context(
        starting_context, backbone_context_dim=backbone.context_dim,
    )
    init_tokens = init_tokens.to(device)
    init_context = init_context.to(device)
    cur_tokens = init_tokens.expand(n_samples, -1).contiguous()
    cur_context = init_context.expand(n_samples, -1, -1).contiguous()

    alive = np.ones((n_samples,), dtype=bool)
    balls_arr = np.full((n_samples,), starting_context.count[0], dtype=np.int64)
    strikes_arr = np.full((n_samples,), starting_context.count[1], dtype=np.int64)

    # ── Phase 0.6 fix (2026-04-26): mid-PA context mutation ────────────
    # Per PHASE_0.5_PLAN §5.2 risk 2 + the Phase 0.6 static-context
    # diagnostic (`results/pitchgpt/rollout_sanity_2025/diagnose_static_context.md`),
    # holding the context vector CONSTANT within a PA causes the
    # per-position outcome distribution to flatten across positions
    # (rollout's CS rate ~0.20 across all positions vs empirical 0.29
    # at pos 0 dropping to 0.045 at pos 5).  Result: K% over-emits by
    # +6pp.  Fix: re-emit the count_state one-hot of the context vector
    # for each next position based on the running (balls, strikes)
    # per sample.  The runner_state, outs, score_diff, inning_bucket,
    # and ump_scalar fields stay constant within a PA -- only
    # count_state mutates.  Encoding follows
    # PitchTokenizer.encode_context (count_state = balls*3 + strikes,
    # capped at balls<=3, strikes<=2 since walk/K terminate first).
    # The count_state one-hot occupies the FIRST NUM_COUNT_STATES (=12)
    # slots of the CONTEXT_DIM vector (offset 0); see
    # context_to_tensor.

    # Per-sample running context for the NEXT position.  We seed it
    # from the starting context and mutate the count_state one-hot
    # in-place each position based on the new (balls, strikes) per
    # sample, after PA-termination has been resolved for that position.
    running_context = init_context[0, 0, :].clone().unsqueeze(0).expand(
        n_samples, -1,
    ).contiguous()  # shape (n_samples, CONTEXT_DIM)

    n_truncated = n_samples

    # ── Sampling loop ──────────────────────────────────────────────────
    for pos in range(horizon):
        if not alive.any():
            break

        with torch.no_grad():
            logits = backbone(cur_tokens, cur_context)  # (N, S, V)
            next_logits = logits[:, -1, :]
            scaled = next_logits / max(float(temperature), 1e-8)
            probs = F.softmax(scaled, dim=-1)
            if torch_gen is not None:
                next_tok = torch.multinomial(
                    probs, num_samples=1, generator=torch_gen,
                ).squeeze(1)
            else:
                next_tok = torch.multinomial(
                    probs, num_samples=1,
                ).squeeze(1)
        next_tok_np = next_tok.detach().cpu().numpy().astype(np.int64)
        probs_np = (
            probs.detach().cpu().numpy().astype(np.float32)
            if return_probs else None
        )

        outcome_probs_pos = None
        outcome_idx_np = None
        if outcome_predictor is not None:
            with torch.no_grad():
                hidden = extract_backbone_hidden_states(
                    backbone, cur_tokens, cur_context,
                )
                last_hidden = hidden[:, -1:, :]
                last_ctx = cur_context[:, -1:, :]
                last_tok = next_tok.unsqueeze(1)

                outc_probs = outcome_predictor.predict_outcome_probs(
                    last_hidden, last_ctx, last_tok,
                )  # (N, 1, 7)
                outc_probs = outc_probs.squeeze(1)
                # Defensive: predictors may return CPU tensors regardless of rollout device; coerce before generator-bound multinomial.
                outc_probs = outc_probs.to(device)
                if torch_gen is not None:
                    sampled_outc = torch.multinomial(
                        outc_probs, num_samples=1, generator=torch_gen,
                    ).squeeze(1)
                else:
                    sampled_outc = torch.multinomial(
                        outc_probs, num_samples=1,
                    ).squeeze(1)
                outcome_idx_np = sampled_outc.detach().cpu().numpy().astype(
                    np.int64
                )
                outcome_probs_pos = (
                    outc_probs.detach().cpu().numpy().astype(np.float32)
                )

        # Per-sample PA-termination + counter advance.
        for i in range(n_samples):
            if not alive[i]:
                continue
            tok_i = int(next_tok_np[i])
            pitch_tokens_out[i, pos] = tok_i
            if return_probs:
                pitch_probs_out[i, pos, :] = probs_np[i]
            terminated_here = False

            if outcome_predictor is not None:
                outc_i = int(outcome_idx_np[i])
                outcomes_out[i, pos] = outc_i
                if return_probs:
                    outcome_probs_out[i, pos, :] = outcome_probs_pos[i]

                # Condition 1: outcome-driven terminal.
                if outc_i in _TERMINAL_INPLAY_OUTCOMES:
                    pa_terminated_out[i, pos] = True
                    pa_outcome_out[i] = outc_i
                    final_count_out[i, 0] = balls_arr[i]
                    final_count_out[i, 1] = strikes_arr[i]
                    alive[i] = False
                    n_truncated -= 1
                    terminated_here = True
                    continue

                # Counter advance per outcome class (Phase 0.6.1 state
                # machine).  ball -> +1 ball; called/swinging strike -> +1
                # strike; foul -> +1 strike capped at 2.  Terminal in-play
                # outcomes never reach here (handled by the `continue` above).
                new_b, new_s = _advance_count(
                    int(balls_arr[i]), int(strikes_arr[i]), outc_i
                )
                balls_arr[i] = new_b
                strikes_arr[i] = new_s

            else:
                # outcome_predictor is None -> count-only fallback.
                decoded = PitchTokenizer.decode(tok_i)
                zone_idx = int(decoded.get("zone", -1))
                if _zone_is_in_strike_zone(zone_idx):
                    strikes_arr[i] += 1
                else:
                    balls_arr[i] += 1

            # Conditions 2 & 3: walk / K via counter checks.
            if not terminated_here:
                if balls_arr[i] >= 4:
                    pa_terminated_out[i, pos] = True
                    final_count_out[i, 0] = min(int(balls_arr[i]), 4)
                    final_count_out[i, 1] = min(int(strikes_arr[i]), 3)
                    alive[i] = False
                    n_truncated -= 1
                    # pa_outcome stays at ROLLOUT_PAD_OUTCOME -- walks
                    # have no terminal-outcome class (per §3.3 +
                    # PHASE_0.5_PLAN §2.0.5.6 test 7).
                    continue
                if strikes_arr[i] >= 3:
                    pa_terminated_out[i, pos] = True
                    final_count_out[i, 0] = min(int(balls_arr[i]), 4)
                    final_count_out[i, 1] = min(int(strikes_arr[i]), 3)
                    alive[i] = False
                    n_truncated -= 1
                    # K terminates via count; no terminal-outcome class.
                    continue

        # Append sampled token + per-sample mutated context for the
        # next-position forward pass.  Dead samples get PAD_TOKEN so
        # the backbone's padding-mask ignores them.
        next_tok_full = next_tok.clone()
        if not alive.all():
            dead_mask = torch.from_numpy(~alive).to(device)
            next_tok_full = torch.where(
                dead_mask,
                torch.full_like(next_tok_full, PAD_TOKEN),
                next_tok_full,
            )
        cur_tokens = torch.cat(
            [cur_tokens, next_tok_full.unsqueeze(1)], dim=1
        )

        # Phase 0.6 fix: re-emit per-sample count_state one-hot for the
        # NEXT position based on running (balls, strikes).  All other
        # context fields (outs, runner_state, batter_hand, inning,
        # score_diff, ump_scalar) are PA-invariant and remain whatever
        # was in the seeded `running_context`.  Walks (4 balls) and
        # strikeouts (3 strikes) terminate the PA at the current
        # position, so the next-position context is only consumed for
        # samples whose new count is in the 12-bucket grid (balls 0..3,
        # strikes 0..2).  We still cap defensively in case of any edge.
        new_count_state = (
            np.minimum(balls_arr, 3) * 3 + np.minimum(strikes_arr, 2)
        ).astype(np.int64)  # (n_samples,)
        # Zero the count_state one-hot slots [0, NUM_COUNT_STATES) and
        # set the new bucket.  This is a vectorised in-place update.
        running_context[:, :NUM_COUNT_STATES] = 0.0
        new_count_idx_t = torch.from_numpy(new_count_state).to(
            running_context.device,
        )
        running_context.scatter_(
            1,
            new_count_idx_t.unsqueeze(1),
            torch.ones((n_samples, 1), device=running_context.device),
        )
        ctx_step = running_context.unsqueeze(1)  # (n_samples, 1, CONTEXT_DIM)
        cur_context = torch.cat([cur_context, ctx_step], dim=1)

    # Update final counters for unterminated (truncated) samples.
    for i in range(n_samples):
        if alive[i]:
            final_count_out[i, 0] = min(int(balls_arr[i]), 4)
            final_count_out[i, 1] = min(int(strikes_arr[i]), 3)

    meta = _build_metadata(
        temperature=temperature,
        seed=seed,
        n_samples=n_samples,
        horizon=horizon,
        backbone_version=backbone_version,
        backbone_sha=backbone_sha,
        outcome_predictor_name=pred_name,
        outcome_predictor_sha=pred_sha,
        calibration_valid=cal_valid,
        calibration_invalid_reasons=cal_reasons,
        n_truncated=int(n_truncated),
    )

    return RolloutResult(
        pitch_tokens=pitch_tokens_out,
        pitch_probs=pitch_probs_out,
        outcomes=outcomes_out,
        outcome_probs=outcome_probs_out,
        pa_terminated=pa_terminated_out,
        pa_outcome=pa_outcome_out,
        final_count=final_count_out,
        sampling_metadata=meta,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation utilities (ticket 0.5.3, SIM_ENGINE_API §5)
# ─────────────────────────────────────────────────────────────────────────────


def pa_woba_distribution(
    rollout_result: RolloutResult,
    woba_lookup: WObaTable | None = None,
) -> np.ndarray:
    """Per-rollout PA-level wOBA contribution.  See SIM_ENGINE_API §5.1.

    For each of the ``n_samples`` rollouts:
      * Terminated AND ``outcomes is not None``: lookup wOBA via the
        7-class outcome (Phase 0.5 default scalar table per
        :meth:`WObaTable.default`).
      * Terminated BUT ``outcomes is None`` (no predictor): NaN
        (Phase-1 follow-up: PA-empirical-fallback).
      * Did NOT terminate: NaN.

    Returns shape ``(n_samples,)``, float32.  Truncated samples are
    NaN; consumers compute ``np.nanmean / np.nanpercentile``.  Silent
    zero is forbidden per SIM_ENGINE_API §3.3.
    """
    table = woba_lookup or WObaTable.default()
    n_samples = rollout_result.pitch_tokens.shape[0]
    out = np.full((n_samples,), np.nan, dtype=np.float32)
    if rollout_result.pa_outcome is None:
        return out
    pa_outc = rollout_result.pa_outcome
    for i in range(n_samples):
        oc = int(pa_outc[i])
        if oc < 0 or oc >= NUM_OUTCOME_CLASSES:
            continue  # PAD / truncated -> NaN
        out[i] = table.lookup(oc)
    return out


def _modal_token_at_position(
    rollout_result: RolloutResult, position: int,
) -> int | None:
    """Return the most-frequent sampled token at ``position`` (or None)."""
    if position < 0 or position >= rollout_result.pitch_tokens.shape[1]:
        return None
    col = rollout_result.pitch_tokens[:, position]
    valid = col != ROLLOUT_PAD_PITCH
    if not valid.any():
        return None
    vals, counts = np.unique(col[valid], return_counts=True)
    return int(vals[np.argmax(counts)])


def percentile_of_actual_outcome(
    rollout_result: RolloutResult,
    actual_outcome: int,
    actual_pitch_token: int | None = None,
    woba_lookup: WObaTable | None = None,
) -> float:
    """Percentile rank of the actual pitch's wOBA within the rollout.

    See SIM_ENGINE_API §5.2.  Used by A1 (counterfactual pitch-call
    grade).  Low percentile = pitcher gave up MORE than expected
    (worse call); high = better.

    Edge cases:
      * ``rollout.outcomes is None`` -> NaN (Phase-1 follow-up:
        PA-empirical-fallback percentile).
      * ``actual_pitch_token`` provided AND differs from modal sampled
        token at position 0 -> log a WARNING (this IS the
        counterfactual setup, not an error).
      * All ``n_samples`` samples truncated -> NaN.
    """
    if rollout_result.outcomes is None:
        logger.warning(
            "percentile_of_actual_outcome: rollout has no outcome "
            "predictor -- returning NaN.",
        )
        return float("nan")

    table = woba_lookup or WObaTable.default()
    actual_woba = table.lookup(int(actual_outcome))
    if not np.isfinite(actual_woba):
        return float("nan")

    sampled = pa_woba_distribution(rollout_result, woba_lookup=table)
    valid = ~np.isnan(sampled)
    if not valid.any():
        return float("nan")

    if actual_pitch_token is not None:
        modal_tok = _modal_token_at_position(rollout_result, position=0)
        if modal_tok is not None and int(actual_pitch_token) != int(modal_tok):
            logger.warning(
                "percentile_of_actual_outcome: actual_pitch_token=%d "
                "differs from modal sampled token=%s -- this IS the "
                "counterfactual setup, not an error.",
                int(actual_pitch_token), str(modal_tok),
            )

    valid_woba = sampled[valid]
    return float(np.mean(valid_woba <= actual_woba))


def outcome_marginal(
    rollout_result: RolloutResult,
    outcome_class: int | None = None,
) -> np.ndarray:
    """Per-position 7-class outcome frequency, NaN-masked.

    See SIM_ENGINE_API §5.3.
      * ``outcome_class`` provided: shape ``(horizon,)``.
      * ``outcome_class is None``: shape ``(horizon, 7)``.

    Truncated positions (NaN past sample termination) are excluded via
    ``np.nanmean``.  Silent zero would bias aggregates per §3.3.
    """
    if rollout_result.outcome_probs is None:
        n_samples, horizon = rollout_result.pitch_tokens.shape
        if outcome_class is None:
            return np.full(
                (horizon, NUM_OUTCOME_CLASSES), np.nan, dtype=np.float32,
            )
        return np.full((horizon,), np.nan, dtype=np.float32)
    probs = rollout_result.outcome_probs  # (N, H, 7) NaN past term
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        marg = np.nanmean(probs, axis=0)  # (H, 7)
    if outcome_class is None:
        return marg.astype(np.float32)
    if not (0 <= int(outcome_class) < NUM_OUTCOME_CLASSES):
        raise ValueError(
            f"outcome_marginal: outcome_class must be in [0, "
            f"{NUM_OUTCOME_CLASSES}), got {outcome_class}"
        )
    return marg[:, int(outcome_class)].astype(np.float32)


def pitch_token_marginal(
    rollout_result: RolloutResult,
    position: int | None = None,
) -> np.ndarray:
    """Per-position pitch-token frequency, NaN-masked.

    See SIM_ENGINE_API §5.4.
      * ``position`` provided: shape ``(VOCAB_SIZE,)``.
      * ``position is None``: shape ``(horizon, VOCAB_SIZE)``.

    Critical for sampling-fidelity: H=1 marginal MUST match the
    backbone's direct next-token softmax (SIM_ENGINE_API §9 binding
    gate; PHASE_0.5_PLAN §6.5).
    """
    if rollout_result.pitch_probs is None:
        n_samples, horizon = rollout_result.pitch_tokens.shape
        if position is None:
            return np.full(
                (horizon, VOCAB_SIZE), np.nan, dtype=np.float32,
            )
        return np.full((VOCAB_SIZE,), np.nan, dtype=np.float32)
    probs = rollout_result.pitch_probs  # (N, H, V) NaN past term
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        marg = np.nanmean(probs, axis=0)
    if position is None:
        return marg.astype(np.float32)
    h = rollout_result.pitch_tokens.shape[1]
    if not (0 <= int(position) < h):
        raise ValueError(
            f"pitch_token_marginal: position must be in [0, {h}), got "
            f"{position}"
        )
    return marg[int(position), :].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Public API surface
# ─────────────────────────────────────────────────────────────────────────────


__all__ = [
    # Constants
    "ROLLOUT_PAD_PITCH",
    "ROLLOUT_PAD_OUTCOME",
    "ROLLOUT_ENGINE_VERSION",
    "CALIBRATION_INVALID_REASONS",
    "ALLOWED_OUTCOME_PREDICTOR_NAMES",
    "OUTCOME_BALL",
    "OUTCOME_CALLED_STRIKE",
    "OUTCOME_SWINGING_STRIKE",
    "OUTCOME_FOUL",
    "OUTCOME_IN_PLAY_OUT",
    "OUTCOME_IN_PLAY_HIT",
    "OUTCOME_HBP",
    # Errors
    "CalibrationError",
    # Dataclasses
    "PAContext",
    "RolloutResult",
    "WObaTable",
    # Protocol
    "OutcomePredictor",
    # Predictors
    "PGConcatHeadPredictor",
    "PGFrozenHeadPredictor",
    "XGBoostOutcomePredictor",
    "EmpiricalPATerminalLookup",
    # Registry
    "OutcomePredictorRegistry",
    # Entry point
    "rollout",
    # Calibration helpers (Agent 2 ticket 0.5.4)
    "_check_calibration_valid",
    "load_calibration_feature_cdfs",
    "load_calibration_json",
    # Aggregation utilities (ticket 0.5.3)
    "pa_woba_distribution",
    "percentile_of_actual_outcome",
    "outcome_marginal",
    "pitch_token_marginal",
]
