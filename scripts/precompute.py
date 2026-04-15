#!/usr/bin/env python
"""
Pre-compute analytics models and cache results in DuckDB.

Iterates through the model registry, calls each model's batch function,
serialises the resulting DataFrame to Parquet, and stores it in the
``leaderboard_cache`` table for instant dashboard reads.

Usage
-----
    python scripts/precompute.py --season 2025               # all models
    python scripts/precompute.py --season 2025 --tier 1      # fast models only
    python scripts/precompute.py --model stuff_plus --season 2025
    python scripts/precompute.py --dry-run --season 2025     # show plan only
    python scripts/precompute.py --force --season 2025       # ignore existing cache
"""
from __future__ import annotations

import argparse
import importlib
import io
import json
import logging
import sys
import time

import pandas as pd
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
logger = logging.getLogger("precompute")

# Ensure stdout can handle Unicode on Windows (cp1252 fallback otherwise)
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True,
    )


# ---------------------------------------------------------------------------
# ANSI formatting (matches daily_refresh.py style)
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


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS: list[dict[str, Any]] = [
    # Tier 1: Fast (< 2 min)
    {"name": "stuff_plus",         "module": "src.analytics.stuff_model",            "fn": "batch_calculate_stuff_plus", "entity": "pitcher", "tier": 1},
    {"name": "volatility_surface", "module": "src.analytics.volatility_surface",     "fn": "batch_calculate",            "entity": "pitcher", "tier": 1},
    {"name": "pset",               "module": "src.analytics.pset",                   "fn": "batch_calculate",            "entity": "pitcher", "tier": 1},
    {"name": "sharpe_lineup",      "module": "src.analytics.sharpe_lineup",          "fn": "batch_player_sharpe",        "entity": "batter",  "tier": 1},
    {"name": "defensive_pressing", "module": "src.analytics.defensive_pressing",     "fn": "batch_calculate",            "entity": "team",    "tier": 1},
    # Tier 2: Medium (2-10 min)
    {"name": "mesi",               "module": "src.analytics.mesi",                   "fn": "batch_calculate",            "entity": "pitcher", "tier": 2},
    {"name": "kinetic_half_life",  "module": "src.analytics.kinetic_half_life",      "fn": "batch_calculate",            "entity": "pitcher", "tier": 2},
    {"name": "alpha_decay",        "module": "src.analytics.alpha_decay",            "fn": "batch_calculate",            "entity": "pitcher", "tier": 2},
    {"name": "allostatic_load",    "module": "src.analytics.allostatic_load",        "fn": "batch_calculate",            "entity": "batter",  "tier": 2},
    {"name": "loft",               "module": "src.analytics.loft",                   "fn": "batch_game_analysis",        "entity": "pitcher", "tier": 2},
    {"name": "baserunner_gravity", "module": "src.analytics.baserunner_gravity",     "fn": "batch_calculate",            "entity": "batter",  "tier": 2},
    # Tier 3: Slow (10+ min)
    {"name": "pitch_decay",        "module": "src.analytics.pitch_decay",            "fn": "batch_calculate",            "entity": "pitcher", "tier": 3},
    {"name": "viscoelastic_workload", "module": "src.analytics.viscoelastic_workload", "fn": "batch_calculate",          "entity": "pitcher", "tier": 3},
    {"name": "causal_war",         "module": "src.analytics.causal_war",             "fn": "batch_calculate",            "entity": "batter",  "tier": 3, "train_fn": "train"},
    {"name": "pitchgpt",           "module": "src.analytics.pitchgpt",               "fn": "batch_calculate",            "entity": "pitcher", "tier": 3},
    {"name": "mechanix_ae",        "module": "src.analytics.mechanix_ae",            "fn": "batch_calculate",            "entity": "pitcher", "tier": 3},
    {"name": "chemnet",            "module": "src.analytics.chemnet",                "fn": "batch_calculate",            "entity": "team",    "tier": 3},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_to_parquet(df) -> bytes:
    """Serialize a DataFrame to Parquet bytes, falling back to pickle."""
    try:
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow")
        return buf.getvalue()
    except Exception:
        import pickle
        logger.warning("pyarrow unavailable, falling back to pickle serialization.")
        buf = io.BytesIO()
        pickle.dump(df, buf)
        return buf.getvalue()


def _call_batch_fn(fn, conn, season: int):
    """Call a batch function, handling different signature styles.

    Tries keyword arg first, then positional, to accommodate the various
    function signatures across analytics modules.
    """
    try:
        return fn(conn, season=season)
    except TypeError:
        # Some functions require season as a positional arg
        return fn(conn, season)


def _ensure_cache_tables(conn) -> None:
    """Create the three cache tables if they don't already exist."""
    from src.db.schema import (
        _create_model_cache,
        _create_leaderboard_cache,
        _create_data_freshness,
    )
    _create_model_cache(conn)
    _create_leaderboard_cache(conn)
    _create_data_freshness(conn)


def _update_data_freshness(conn) -> None:
    """Refresh the data_freshness table with current source-table stats."""
    source_tables = [
        ("pitches", "game_date"),
        ("season_batting_stats", None),
        ("season_pitching_stats", None),
        ("players", None),
        ("games", "game_date"),
    ]
    for table_name, date_col in source_tables:
        try:
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            if date_col:
                max_date = conn.execute(
                    f"SELECT MAX({date_col}) FROM {table_name}"
                ).fetchone()[0]
            else:
                max_date = None

            conn.execute(
                """
                INSERT OR REPLACE INTO data_freshness
                    (table_name, max_game_date, row_count, updated_at)
                VALUES ($1, $2, $3, $4)
                """,
                [table_name, max_date, row_count, datetime.now(timezone.utc)],
            )
        except Exception as exc:
            logger.debug("Could not update freshness for %s: %s", table_name, exc)


def _check_existing_cache(conn, model_name: str, season: int) -> bool:
    """Return True if a valid cache entry already exists (< 24 hours old)."""
    try:
        row = conn.execute(
            """
            SELECT computed_at FROM leaderboard_cache
            WHERE model_name = $1 AND season = $2
            """,
            [model_name, season],
        ).fetchone()
        if row is None:
            return False
        computed_at = row[0]
        if computed_at.tzinfo is None:
            computed_at = computed_at.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - computed_at).total_seconds() / 3600
        return age_hours < 24
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Core precompute loop
# ---------------------------------------------------------------------------

def run_precompute(
    season: int,
    model_filter: Optional[str] = None,
    tier_filter: Optional[int] = None,
    force: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Run the pre-computation pipeline.

    Args:
        season: MLB season year to compute.
        model_filter: If set, only run this specific model name.
        tier_filter: If set, only run models at this tier or lower.
        force: If True, recompute even if a fresh cache entry exists.
        dry_run: If True, print the plan but don't execute anything.

    Returns:
        A list of result dicts, one per model attempted.
    """
    start_time = time.time()

    print()
    print(_c("=" * 60, _BOLD))
    print(_c("  ANALYTICS PRE-COMPUTATION", _BOLD + _CYAN))
    print(_c("=" * 60, _BOLD))
    print(f"  Season:     {season}")
    print(f"  Model:      {model_filter or 'all'}")
    print(f"  Tier:       {'<= ' + str(tier_filter) if tier_filter else 'all'}")
    print(f"  Force:      {force}")
    print(f"  Dry run:    {dry_run}")
    print(_c("=" * 60, _BOLD))
    print()

    # Filter the registry
    targets = MODELS
    if model_filter:
        targets = [m for m in targets if m["name"] == model_filter]
        if not targets:
            _fail(f"Unknown model: {model_filter}")
            print(f"  Available: {', '.join(m['name'] for m in MODELS)}")
            return []
    if tier_filter:
        targets = [m for m in targets if m["tier"] <= tier_filter]

    # Sort by tier (fastest first)
    targets = sorted(targets, key=lambda m: (m["tier"], m["name"]))

    if dry_run:
        print(_c("  DRY RUN -- no models will be executed", _YELLOW + _BOLD))
        print()
        for i, m in enumerate(targets, 1):
            print(f"  {i:2d}. [{m['tier']}] {m['name']:<25s}  ({m['module']}.{m['fn']})")
        print()
        print(f"  Total: {len(targets)} model(s) would be computed.")
        print(_c("=" * 60, _BOLD))
        print()
        return []

    # Open database connection (NOT read_only -- we need to write cache)
    from src.db.schema import get_connection
    conn = get_connection(read_only=False)

    # Ensure cache tables exist
    _ensure_cache_tables(conn)

    results: list[dict] = []

    for idx, model_def in enumerate(targets, 1):
        name = model_def["name"]
        module_path = model_def["module"]
        fn_name = model_def["fn"]
        entity_type = model_def["entity"]
        tier = model_def["tier"]

        print(DIVIDER)
        print(_c(f"  [{idx}/{len(targets)}] {name}  (tier {tier}, {entity_type})", _BOLD))
        print(DIVIDER)

        result: dict[str, Any] = {
            "name": name,
            "tier": tier,
            "entity": entity_type,
            "status": "pending",
            "rows": 0,
            "seconds": 0.0,
        }

        # Skip if fresh cache exists (unless --force)
        if not force and _check_existing_cache(conn, name, season):
            _info("Fresh cache exists, skipping (use --force to override).")
            result["status"] = "cached"
            results.append(result)
            print()
            continue

        # Dynamic import
        try:
            _info(f"Importing {module_path}.{fn_name}...")
            mod = importlib.import_module(module_path)
            batch_fn = getattr(mod, fn_name)
        except (ImportError, AttributeError) as exc:
            _fail(f"Could not import {module_path}.{fn_name}: {exc}")
            logger.exception("Import failure for %s", name)
            result["status"] = "import_error"
            results.append(result)
            print()
            continue

        # If model requires training first, run the train function
        train_fn_name = model_def.get("train_fn")
        if train_fn_name:
            try:
                _info(f"Training {name} before batch compute...")
                train_fn = getattr(mod, train_fn_name)
                train_fn(conn)
                _info("Training complete.")
            except Exception as exc:
                _warn(f"Training failed: {exc}. Attempting batch anyway...")

        # Execute the batch function
        try:
            _info("Computing...")
            t0 = time.time()
            df = _call_batch_fn(batch_fn, conn, season)
            elapsed = time.time() - t0
            result["seconds"] = round(elapsed, 2)

            if df is None or (hasattr(df, "empty") and df.empty):
                _warn(f"Model returned empty result ({elapsed:.1f}s).")
                result["status"] = "empty"
                results.append(result)
                print()
                continue

            row_count = len(df)
            result["rows"] = row_count

            # Serialize to Parquet bytes
            _info(f"Serializing {row_count} rows to Parquet...")
            parquet_bytes = _serialize_to_parquet(df)

            # Upsert into leaderboard_cache
            _info("Writing to leaderboard_cache...")
            conn.execute(
                """
                INSERT OR REPLACE INTO leaderboard_cache
                    (model_name, season, leaderboard_parquet, row_count,
                     computed_at, model_version, compute_seconds)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                [
                    name,
                    season,
                    parquet_bytes,
                    row_count,
                    datetime.now(timezone.utc),
                    "1.0",
                    round(elapsed, 2),
                ],
            )

            # Also store per-entity results in model_cache for deep-dive lookups
            id_col = None
            for candidate in ("pitcher_id", "batter_id", "team_id", "player_id", "runner_id"):
                if candidate in df.columns:
                    id_col = candidate
                    break

            if id_col is not None:
                entity_count = 0
                for _, erow in df.iterrows():
                    eid = erow.get(id_col)
                    if eid is None or (hasattr(eid, '__class__') and eid.__class__.__name__ == 'NAType'):
                        continue
                    try:
                        row_dict = {}
                        for col in df.columns:
                            val = erow[col]
                            if hasattr(val, 'item'):
                                val = val.item()
                            if pd.isna(val):
                                val = None
                            row_dict[col] = val
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO model_cache
                                (model_name, entity_type, entity_id, season, result_json,
                                 computed_at, model_version, compute_seconds)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            """,
                            [
                                name,
                                entity_type,
                                int(eid),
                                season,
                                json.dumps(row_dict, default=str),
                                datetime.now(timezone.utc),
                                "1.0",
                                round(elapsed / max(row_count, 1), 4),
                            ],
                        )
                        entity_count += 1
                    except Exception:
                        continue
                _info(f"Cached {entity_count} entity records in model_cache.")

            _ok(f"{row_count} rows cached in {elapsed:.1f}s ({len(parquet_bytes) / 1024:.0f} KB)")
            result["status"] = "ok"

        except Exception as exc:
            elapsed = time.time() - t0 if "t0" in dir() else 0.0
            _fail(f"Computation failed after {elapsed:.1f}s: {exc}")
            logger.exception("Computation failure for %s", name)
            result["status"] = "error"
            result["error"] = str(exc)

        results.append(result)
        print()

    # Update data freshness
    _info("Updating data_freshness table...")
    try:
        _update_data_freshness(conn)
        _ok("Data freshness updated.")
    except Exception as exc:
        _warn(f"Could not update data freshness: {exc}")

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
    """Print a formatted summary table of the precompute run."""
    print()
    print(_c("=" * 60, _BOLD))
    print(_c("  PRE-COMPUTATION SUMMARY", _BOLD + _CYAN))
    print(_c("=" * 60, _BOLD))
    print()

    if not results:
        print("  No models were executed.")
        print(_c("=" * 60, _BOLD))
        print()
        return

    # Header
    header = f"  {'Model':<25s} {'Tier':>4s} {'Status':<13s} {'Rows':>7s} {'Time':>8s}"
    print(_c(header, _BOLD))
    print("  " + "-" * 57)

    ok_count = 0
    err_count = 0
    skip_count = 0
    total_rows = 0

    for r in results:
        name = r["name"]
        tier = str(r["tier"])
        status = r["status"]
        rows = r["rows"]
        secs = r["seconds"]

        if status == "ok":
            status_display = _c("OK", _GREEN)
            ok_count += 1
            total_rows += rows
        elif status == "cached":
            status_display = _c("CACHED", _DIM)
            skip_count += 1
        elif status == "empty":
            status_display = _c("EMPTY", _YELLOW)
            skip_count += 1
        elif status == "import_error":
            status_display = _c("IMPORT ERR", _RED)
            err_count += 1
        else:
            status_display = _c("ERROR", _RED)
            err_count += 1

        row_str = str(rows) if rows else "-"
        time_str = f"{secs:.1f}s" if secs else "-"
        print(f"  {name:<25s} {tier:>4s} {status_display:<22s} {row_str:>7s} {time_str:>8s}")

    print("  " + "-" * 57)
    print(
        f"  OK: {ok_count}  |  Cached: {skip_count}  |  "
        f"Errors: {err_count}  |  Total rows: {total_rows:,}"
    )
    print(f"  Total elapsed: {elapsed:.1f}s")

    if err_count == 0:
        print(f"\n  {_c('All models completed successfully.', _GREEN + _BOLD)}")
    else:
        print(f"\n  {_c(f'{err_count} model(s) had errors.', _YELLOW + _BOLD)} Check logs for details.")

    print(_c("=" * 60, _BOLD))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and run the precompute pipeline."""
    parser = argparse.ArgumentParser(
        description="Pre-compute analytics models and cache results.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Run only this model (e.g. stuff_plus, mesi).",
    )
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help="MLB season year (e.g. 2025).",
    )
    parser.add_argument(
        "--tier",
        type=int,
        default=None,
        help="Only run models at this tier or lower (1=fast, 2=medium, 3=slow).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if a fresh cache entry exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be computed without executing.",
    )
    args = parser.parse_args()

    run_precompute(
        season=args.season,
        model_filter=args.model,
        tier_filter=args.tier,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
