"""Regression tests for the nightly wrapper's effect checks (2026-08-16).

Two defects shipped together and hid each other:

  1. ``_classify`` returned "ok" whenever ``rc == 0``, discarding ``effect_ok``
     entirely. The wrapped scripts trap their own sub-step failures and exit 0
     regardless, so the exit code is the WEAKER signal -- a step could exit
     cleanly having done nothing and be indistinguishable from one that worked.
  2. ``verify_precompute`` compared a timezone-aware UTC step start against a
     naive LOCAL timestamp from DuckDB, making its check false for any run
     shorter than the UTC offset -- i.e. permanently.

(2) was invisible because of (1): the 2026-08-16 scheduled run recorded
``precompute rc=0, effect_ok=false, status="ok"`` for a precompute that had
demonstrably just rewritten the cache.

These tests load the wrapper by path because ``scripts/`` is not a package.
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_WRAPPER = Path(__file__).resolve().parents[1] / "scripts" / "nightly_refresh.py"


@pytest.fixture(scope="module")
def nr():
    spec = importlib.util.spec_from_file_location("nightly_refresh_under_test", _WRAPPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# _classify
# --------------------------------------------------------------------------


def test_clean_exit_with_unverified_effect_is_a_failure(nr):
    """THE regression. A step that exits 0 while doing nothing must not pass."""
    assert nr._classify(0, False, False) == "fail"


def test_clean_exit_with_verified_effect_is_ok(nr):
    assert nr._classify(0, True, False) == "ok"


@pytest.mark.parametrize("rc", [1, 127, 3221226505, None])
def test_nonzero_exit_with_verified_effect_stays_tolerated(nr, rc):
    """The exit-nonzero-after-commit tolerance is legitimate and must survive.

    A large DuckDB write can genuinely exit non-zero at teardown after the
    COMMIT landed; when the effect check independently confirms the work, that
    is still ``ok_verified`` and must not fail the chain.
    """
    assert nr._classify(rc, True, False) == "ok_verified"


@pytest.mark.parametrize("rc", [1, 127, 3221226505, None])
def test_nonzero_exit_without_verified_effect_fails(nr, rc):
    assert nr._classify(rc, False, False) == "fail"


def test_timeout_always_fails(nr):
    assert nr._classify(0, True, True) == "fail"


# --------------------------------------------------------------------------
# verify_precompute timezone handling
# --------------------------------------------------------------------------


def test_precompute_check_compares_in_one_clock(nr, monkeypatch):
    """A cache written moments after the step started must verify.

    Before the fix this was false whenever the local offset exceeded the step
    duration, which for a UTC-4 machine meant always.
    """
    started = datetime.now(timezone.utc) - timedelta(minutes=5)
    # DuckDB hands back a naive LOCAL timestamp.
    wrote_at = datetime.now().replace(tzinfo=None) - timedelta(minutes=1)
    monkeypatch.setattr(nr, "_leaderboard_max_computed_at", lambda: wrote_at)

    step = {"started_at": started.isoformat(), "returncode": 0, "timed_out": False}
    out = nr.verify_precompute(step)

    assert out["verify"]["effect_ok"] is True
    assert out["status"] == "ok"


def test_precompute_check_still_fails_on_a_genuinely_stale_cache(nr, monkeypatch):
    """The check must keep its teeth: a cache older than the step start fails."""
    started = datetime.now(timezone.utc)
    stale = datetime.now().replace(tzinfo=None) - timedelta(days=5)
    monkeypatch.setattr(nr, "_leaderboard_max_computed_at", lambda: stale)

    step = {"started_at": started.isoformat(), "returncode": 0, "timed_out": False}
    out = nr.verify_precompute(step)

    assert out["verify"]["effect_ok"] is False
    assert out["status"] == "fail"


def test_precompute_check_fails_when_cache_is_unreadable(nr, monkeypatch):
    monkeypatch.setattr(nr, "_leaderboard_max_computed_at", lambda: None)

    step = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "returncode": 0,
        "timed_out": False,
    }
    out = nr.verify_precompute(step)

    assert out["verify"]["effect_ok"] is False
    assert out["status"] == "fail"
