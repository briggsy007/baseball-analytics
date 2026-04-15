#!/usr/bin/env python
"""
Automated model retraining pipeline for the baseball analytics platform.

Iterates through all BaseAnalyticsModel subclasses, evaluates the current
production model (if any), trains a new version, compares metrics, and
optionally promotes the new model if it improves over the baseline.

All results are appended to ``logs/training_log.jsonl`` for auditability.

Usage
-----
    python scripts/retrain.py --model all                          # retrain all models
    python scripts/retrain.py --model causal_war                   # retrain one model
    python scripts/retrain.py --model all --dry-run                # evaluate only
    python scripts/retrain.py --model all --promote                # auto-promote winners
    python scripts/retrain.py --model all --since 2025-06-01       # data range filter
"""
from __future__ import annotations

import argparse
import importlib
import io
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Project root on sys.path
# ---------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retrain")

# Ensure stdout handles Unicode on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True,
    )


# ---------------------------------------------------------------------------
# ANSI formatting (consistent with precompute.py / daily_refresh.py)
# ---------------------------------------------------------------------------
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_RESET = "\033[0m"

DIVIDER = "\033[90m" + "-" * 60 + _RESET


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


def _ok(msg: str) -> None:
    print(f"  {_c('[OK]', _GREEN + _BOLD)}  {msg}")


def _warn(msg: str) -> None:
    print(f"  {_c('[WARN]', _YELLOW + _BOLD)} {msg}")


def _fail(msg: str) -> None:
    print(f"  {_c('[FAIL]', _RED + _BOLD)} {msg}")


def _info(msg: str) -> None:
    print(f"  {_c('[..]', _CYAN)}  {msg}")


def _skip(msg: str) -> None:
    print(f"  {_c('[SKIP]', _DIM + _BOLD)} {msg}")


# ---------------------------------------------------------------------------
# Model registry — maps the precompute "name" to the BaseAnalyticsModel class
# ---------------------------------------------------------------------------
# Each entry: (precompute name, module path, class name)
# "stuff_plus" is special — it uses a standalone train function, not a
# BaseAnalyticsModel subclass, so it is excluded from the class-based pipeline.

MODEL_CLASSES: list[dict[str, str]] = [
    {"name": "stuff_plus",          "module": "src.analytics.stuff_model",            "class": None,                         "train_fn": "train_stuff_model"},
    {"name": "volatility_surface",  "module": "src.analytics.volatility_surface",     "class": "PitchVolatilitySurfaceModel"},
    {"name": "pset",                "module": "src.analytics.pset",                   "class": "PSETModel"},
    {"name": "sharpe_lineup",       "module": "src.analytics.sharpe_lineup",          "class": "SharpeLineupModel"},
    {"name": "defensive_pressing",  "module": "src.analytics.defensive_pressing",     "class": "DefensivePressingModel"},
    {"name": "mesi",                "module": "src.analytics.mesi",                   "class": "MESIModel"},
    {"name": "kinetic_half_life",   "module": "src.analytics.kinetic_half_life",      "class": "KineticHalfLifeModel"},
    {"name": "alpha_decay",         "module": "src.analytics.alpha_decay",            "class": "AlphaDecayModel"},
    {"name": "allostatic_load",     "module": "src.analytics.allostatic_load",        "class": "AllostaticLoadModel"},
    {"name": "loft",                "module": "src.analytics.loft",                   "class": "LOFTModel"},
    {"name": "baserunner_gravity",  "module": "src.analytics.baserunner_gravity",     "class": "BaserunnerGravityModel"},
    {"name": "pitch_decay",         "module": "src.analytics.pitch_decay",            "class": "PitchDecayRateModel"},
    {"name": "viscoelastic_workload", "module": "src.analytics.viscoelastic_workload", "class": "ViscoelasticWorkloadModel"},
    {"name": "causal_war",          "module": "src.analytics.causal_war",             "class": "CausalWARModel"},
    {"name": "pitchgpt",            "module": "src.analytics.pitchgpt",               "class": "PitchGPT"},
    {"name": "mechanix_ae",         "module": "src.analytics.mechanix_ae",            "class": "MechanixAEModel"},
    {"name": "chemnet",             "module": "src.analytics.chemnet",                "class": "ChemNetModel"},
]


# ---------------------------------------------------------------------------
# Training log
# ---------------------------------------------------------------------------

LOGS_DIR: Path = ROOT / "logs"
TRAINING_LOG_PATH: Path = LOGS_DIR / "training_log.jsonl"


def _get_conn():
    """Open a DuckDB connection for the retraining pipeline."""
    from src.db.schema import get_connection
    return get_connection(read_only=False)


def _get_registry():
    """Return a fresh ModelRegistry instance (patchable for tests)."""
    from src.analytics.registry import ModelRegistry
    return ModelRegistry()


def _is_base_analytics_model(obj) -> bool:
    """Check if *obj* is an instance of BaseAnalyticsModel (patchable)."""
    from src.analytics.base import BaseAnalyticsModel
    return isinstance(obj, BaseAnalyticsModel)


def _append_training_log(entry: dict[str, Any]) -> None:
    """Append a single JSON line to the training log file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRAINING_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


# ---------------------------------------------------------------------------
# Metric comparison
# ---------------------------------------------------------------------------

# Keys that should be *higher* for a better model.
_HIGHER_IS_BETTER: set[str] = {
    "r2_train", "r2_test", "r2",
    "n_players", "n_pitchers", "coverage",
    "mean_mesi", "mean_causal_war",
}

# Keys that should be *lower* for a better model.
_LOWER_IS_BETTER: set[str] = {
    "rmse_train", "rmse_test", "rmse",
    "mae", "mse",
    "std_causal_war",
}


def compare_metrics(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[bool, str]:
    """Compare two metric dicts and decide whether the candidate improves.

    Strategy:
    1.  For known keys in ``_HIGHER_IS_BETTER`` / ``_LOWER_IS_BETTER``, tally
        how many improved vs. degraded.
    2.  If improvements > degradations, candidate wins.
    3.  If no known keys overlap, accept the candidate (no baseline to beat).

    Returns:
        (improved: bool, reason: str)
    """
    improvements = 0
    degradations = 0
    details: list[str] = []

    shared_keys = set(baseline.keys()) & set(candidate.keys())

    for key in shared_keys:
        bval = baseline.get(key)
        cval = candidate.get(key)

        # Only compare numeric values
        if not isinstance(bval, (int, float)) or not isinstance(cval, (int, float)):
            continue

        if key in _HIGHER_IS_BETTER:
            if cval > bval:
                improvements += 1
                details.append(f"{key}: {bval} -> {cval} (improved)")
            elif cval < bval:
                degradations += 1
                details.append(f"{key}: {bval} -> {cval} (degraded)")
        elif key in _LOWER_IS_BETTER:
            if cval < bval:
                improvements += 1
                details.append(f"{key}: {bval} -> {cval} (improved)")
            elif cval > bval:
                degradations += 1
                details.append(f"{key}: {bval} -> {cval} (degraded)")

    if improvements == 0 and degradations == 0:
        return True, "No comparable baseline metrics; accepting new model."

    improved = improvements > degradations
    summary = (
        f"{improvements} improved, {degradations} degraded"
        + (f" [{'; '.join(details)}]" if details else "")
    )
    return improved, summary


# ---------------------------------------------------------------------------
# Stuff+ special-case handler
# ---------------------------------------------------------------------------

def _retrain_stuff_plus(
    conn,
    *,
    dry_run: bool,
    promote: bool,
    since: Optional[str],
) -> dict[str, Any]:
    """Handle Stuff+ retraining, which uses a standalone function."""
    result: dict[str, Any] = {
        "model": "stuff_plus",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_range": {"since": since} if since else None,
        "status": "pending",
    }

    if dry_run:
        _info("DRY RUN: would retrain stuff_plus via train_stuff_model()")
        result["status"] = "dry_run"
        result["decision"] = "skipped (dry run)"
        return result

    try:
        from src.analytics.stuff_model import train_stuff_model

        _info("Training stuff_plus model...")
        metrics = train_stuff_model(conn)
        result["metrics_new"] = metrics
        result["status"] = "trained"
        result["decision"] = "saved (standalone model, always saves)"
        _ok(f"stuff_plus trained: R2_test={metrics.get('r2_test', 'N/A')}")
    except Exception as exc:
        _fail(f"stuff_plus training failed: {exc}")
        logger.exception("stuff_plus training failure")
        result["status"] = "error"
        result["error"] = str(exc)
        result["decision"] = "failed"

    return result


# ---------------------------------------------------------------------------
# Core retraining loop
# ---------------------------------------------------------------------------

def _retrain_single_model(
    model_def: dict[str, str],
    conn,
    *,
    dry_run: bool,
    promote: bool,
    since: Optional[str],
) -> dict[str, Any]:
    """Retrain a single BaseAnalyticsModel subclass.

    Returns a result dict for the training log.
    """
    name = model_def["name"]
    module_path = model_def["module"]
    class_name = model_def["class"]

    result: dict[str, Any] = {
        "model": name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data_range": {"since": since} if since else None,
        "status": "pending",
    }

    # -- Import the model class ------------------------------------------------
    try:
        mod = importlib.import_module(module_path)
        model_cls = getattr(mod, class_name)
    except (ImportError, AttributeError) as exc:
        _fail(f"Could not import {module_path}.{class_name}: {exc}")
        result["status"] = "import_error"
        result["error"] = str(exc)
        result["decision"] = "skipped (import error)"
        return result

    # -- Load existing production model and get baseline metrics ----------------
    registry = _get_registry()
    baseline_metrics: dict[str, Any] | None = None

    try:
        existing_model = registry.load_model(name)
        if _is_base_analytics_model(existing_model):
            _info(f"Evaluating existing production model for baseline...")
            baseline_metrics = existing_model.evaluate(conn)
            result["metrics_baseline"] = baseline_metrics
            _info(f"Baseline metrics: {baseline_metrics}")
        else:
            _info("Existing model is not a BaseAnalyticsModel; skipping baseline eval.")
    except FileNotFoundError:
        _info("No existing model found in registry; training from scratch.")
    except Exception as exc:
        _warn(f"Could not evaluate existing model: {exc}")

    # -- Dry run: evaluate only, don't train -----------------------------------
    if dry_run:
        _info(f"DRY RUN: would retrain {name}")
        result["status"] = "dry_run"
        result["decision"] = "skipped (dry run)"
        return result

    # -- Train a new model -----------------------------------------------------
    try:
        _info(f"Instantiating {class_name}...")
        new_model = model_cls()

        train_kwargs: dict[str, Any] = {}
        if since is not None:
            train_kwargs["since"] = since

        _info(f"Training {name}...")
        t0 = time.time()
        training_metrics = new_model.train(conn, **train_kwargs)
        train_seconds = round(time.time() - t0, 2)

        result["train_seconds"] = train_seconds
        _ok(f"Training complete in {train_seconds}s")
    except Exception as exc:
        _fail(f"Training failed for {name}: {exc}")
        logger.exception("Training failure for %s", name)
        result["status"] = "error"
        result["error"] = str(exc)
        result["decision"] = "failed (training error)"
        return result

    # -- Evaluate the new model ------------------------------------------------
    try:
        _info(f"Evaluating new {name} model...")
        new_metrics = new_model.evaluate(conn)
        result["metrics_new"] = new_metrics
        _info(f"New metrics: {new_metrics}")
    except Exception as exc:
        _warn(f"Evaluation failed for new {name}: {exc}")
        new_metrics = training_metrics if training_metrics else {}
        result["metrics_new"] = new_metrics

    # -- Compare and decide ----------------------------------------------------
    if baseline_metrics is not None:
        improved, reason = compare_metrics(baseline_metrics, new_metrics)
        result["comparison"] = reason
    else:
        improved = True
        reason = "No baseline model; accepting new model."
        result["comparison"] = reason

    if improved:
        _ok(f"New model improves over baseline: {reason}")
    else:
        _warn(f"New model does NOT improve: {reason}")

    # -- Save (always as new version, never overwrite) -------------------------
    if improved:
        stage = "production" if promote else "dev"
        try:
            save_info = registry.save_model(
                model=new_model,
                name=name,
                metadata={
                    "training_metrics": training_metrics,
                    "eval_metrics": new_metrics,
                    "baseline_metrics": baseline_metrics,
                    "comparison": reason,
                    "data_range": {"since": since} if since else "all",
                },
                stage=stage,
            )
            result["status"] = "promoted" if promote else "saved"
            result["decision"] = f"promoted to {stage}" if promote else "saved as dev"
            result["version"] = save_info["version"]
            result["model_path"] = save_info["model_path"]
            _ok(f"Saved as {save_info['version']} (stage={stage})")
        except Exception as exc:
            _fail(f"Could not save model {name}: {exc}")
            result["status"] = "save_error"
            result["error"] = str(exc)
            result["decision"] = "failed (save error)"
    else:
        result["status"] = "rejected"
        result["decision"] = "rejected (no improvement)"
        _info("Model not saved (no improvement over baseline).")

    return result


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_retrain(
    model_filter: str = "all",
    since: Optional[str] = None,
    dry_run: bool = False,
    promote: bool = False,
) -> list[dict[str, Any]]:
    """Execute the retraining pipeline.

    Args:
        model_filter: Specific model name or ``"all"``.
        since: Only use data since this date (YYYY-MM-DD).
        dry_run: If True, evaluate only; don't train or save.
        promote: If True, auto-promote improved models to production.

    Returns:
        List of result dicts, one per model.
    """
    start_time = time.time()

    print()
    print(_c("=" * 60, _BOLD))
    print(_c("  MODEL RETRAINING PIPELINE", _BOLD + _CYAN))
    print(_c("=" * 60, _BOLD))
    print(f"  Model:      {model_filter}")
    print(f"  Since:      {since or 'all available data'}")
    print(f"  Dry run:    {dry_run}")
    print(f"  Promote:    {promote}")
    print(_c("=" * 60, _BOLD))
    print()

    # Filter model list
    if model_filter == "all":
        targets = MODEL_CLASSES
    else:
        targets = [m for m in MODEL_CLASSES if m["name"] == model_filter]
        if not targets:
            _fail(f"Unknown model: {model_filter}")
            available = ", ".join(m["name"] for m in MODEL_CLASSES)
            print(f"  Available models: {available}")
            return []

    # Open database connection
    conn = _get_conn()

    results: list[dict[str, Any]] = []

    for idx, model_def in enumerate(targets, 1):
        name = model_def["name"]

        print(DIVIDER)
        print(_c(f"  [{idx}/{len(targets)}] {name}", _BOLD))
        print(DIVIDER)

        # Stuff+ is a special case — it uses a standalone function
        if model_def.get("class") is None:
            train_fn_name = model_def.get("train_fn")
            if train_fn_name == "train_stuff_model":
                result = _retrain_stuff_plus(
                    conn,
                    dry_run=dry_run,
                    promote=promote,
                    since=since,
                )
            else:
                _skip(f"{name} has no model class; skipping.")
                result = {
                    "model": name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": "skipped",
                    "decision": "no BaseAnalyticsModel class",
                }
        else:
            result = _retrain_single_model(
                model_def,
                conn,
                dry_run=dry_run,
                promote=promote,
                since=since,
            )

        results.append(result)

        # Append to training log
        _append_training_log(result)

        print()

    try:
        conn.close()
    except Exception:
        pass

    # Print summary
    total_elapsed = time.time() - start_time
    _print_summary(results, total_elapsed)

    return results


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _print_summary(results: list[dict], elapsed: float) -> None:
    """Print a formatted summary table of the retraining run."""
    print()
    print(_c("=" * 60, _BOLD))
    print(_c("  RETRAINING SUMMARY", _BOLD + _CYAN))
    print(_c("=" * 60, _BOLD))
    print()

    if not results:
        print("  No models were processed.")
        print(_c("=" * 60, _BOLD))
        print()
        return

    # Header
    header = f"  {'Model':<28s} {'Status':<15s} {'Decision'}"
    print(_c(header, _BOLD))
    print("  " + "-" * 56)

    counts: dict[str, int] = {}

    for r in results:
        name = r.get("model", "?")
        status = r.get("status", "?")
        decision = r.get("decision", "")

        counts[status] = counts.get(status, 0) + 1

        if status in ("promoted", "saved"):
            status_display = _c(status.upper(), _GREEN)
        elif status == "rejected":
            status_display = _c("REJECTED", _YELLOW)
        elif status == "dry_run":
            status_display = _c("DRY RUN", _DIM)
        elif status == "skipped":
            status_display = _c("SKIPPED", _DIM)
        elif status in ("error", "import_error", "save_error"):
            status_display = _c("ERROR", _RED)
        else:
            status_display = status

        print(f"  {name:<28s} {status_display:<24s} {decision}")

    print("  " + "-" * 56)

    parts = [f"{v} {k}" for k, v in sorted(counts.items())]
    print(f"  {' | '.join(parts)}")
    print(f"  Total elapsed: {elapsed:.1f}s")
    print(f"  Training log: {TRAINING_LOG_PATH}")

    error_count = sum(v for k, v in counts.items() if "error" in k)
    if error_count == 0:
        print(f"\n  {_c('Pipeline complete.', _GREEN + _BOLD)}")
    else:
        print(f"\n  {_c(f'{error_count} model(s) had errors.', _YELLOW + _BOLD)} Check logs.")

    print(_c("=" * 60, _BOLD))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the retraining pipeline."""
    parser = argparse.ArgumentParser(
        description="Automated model retraining pipeline.",
    )
    parser.add_argument(
        "--model",
        default="all",
        help='Model name to retrain, or "all" (default: all).',
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Only use data since this date (YYYY-MM-DD). Default: all available.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate existing models only; do not train or save.",
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Auto-promote new models to production if metrics improve.",
    )
    args = parser.parse_args()

    run_retrain(
        model_filter=args.model,
        since=args.since,
        dry_run=args.dry_run,
        promote=args.promote,
    )


if __name__ == "__main__":
    main()
