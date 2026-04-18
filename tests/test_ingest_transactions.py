"""
Tests for scripts/ingest_transactions.py.

Covers:
    1. map_transaction() on 5+ canned / real StatsAPI payloads.
    2. map_transactions() deduplication + bad-id drop.
    3. iter_month_windows() tiling edge cases.
    4. upsert_rows() idempotency: inserting the same batch twice yields the
       same final row count.
    5. Schema match: every mapped row has the columns the transactions table
       expects, and the DB accepts them.
    6. _fetch_window() retries transient failures (mocked, NO real network).
    7. HTTP 4xx → empty list (mocked).

No real network calls.
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import pytest
import requests

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.ingest_transactions import (  # noqa: E402
    TRANSACTION_COLUMNS,
    _fetch_window,
    _parse_iso_date,
    iter_month_windows,
    map_transaction,
    map_transactions,
    upsert_rows,
)
from src.db.schema import create_tables  # noqa: E402


_FIXTURE_PATH = (
    _PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "mlb_statsapi"
    / "transactions_2023_03.json"
)


# ── Canned payloads ────────────────────────────────────────────────────────

# 1. Trade with fromTeam + toTeam.
_TRADE = {
    "id": 100001,
    "person": {"id": 500001, "fullName": "Trade Guy"},
    "fromTeam": {"id": 120, "name": "Chicago White Sox"},
    "toTeam": {"id": 118, "name": "Kansas City Royals"},
    "date": "2023-07-31",
    "effectiveDate": "2023-07-31",
    "resolutionDate": "2023-07-31",
    "typeCode": "TR",
    "typeDesc": "Trade",
    "description": "Chicago White Sox traded RHP Trade Guy to Kansas City Royals.",
}

# 2. IL placement (Status Change).
_IL_PLACEMENT = {
    "id": 100002,
    "person": {"id": 500002, "fullName": "Injured Arm"},
    "team": {"id": 143, "name": "Philadelphia Phillies"},
    "date": "2023-05-15",
    "effectiveDate": "2023-05-10",
    "resolutionDate": "2023-05-15",
    "typeCode": "SC",
    "typeDesc": "Status Change",
    "description": "Philadelphia Phillies placed RHP Injured Arm on the 15-day injured list retroactive to May 10, 2023. Right UCL sprain.",
}

# 3. DFA (team only, no from/to).
_DFA = {
    "id": 100003,
    "person": {"id": 500003, "fullName": "Cut Player"},
    "team": {"id": 119, "name": "Los Angeles Dodgers"},
    "date": "2023-06-01",
    "typeCode": "DFA",
    "typeDesc": "Designated for Assignment",
    "description": "Los Angeles Dodgers designated OF Cut Player for assignment.",
}

# 4. Activation / reinstate (has both from and to in some payloads, only team in others).
_ACTIVATION = {
    "id": 100004,
    "person": {"id": 500002, "fullName": "Injured Arm"},
    "team": {"id": 143, "name": "Philadelphia Phillies"},
    "date": "2023-06-20",
    "effectiveDate": "2023-06-20",
    "typeCode": "SC",
    "typeDesc": "Status Change",
    "description": "Philadelphia Phillies activated RHP Injured Arm from the 15-day injured list.",
}

# 5. Bad / missing id → should be dropped.
_BAD = {
    "person": {"id": 500005, "fullName": "Ghost"},
    "date": "2023-04-01",
    "typeDesc": "Status Change",
    "description": "no id here",
}


# ── map_transaction() ──────────────────────────────────────────────────────


def test_map_trade():
    row = map_transaction(_TRADE)
    assert row is not None
    assert row["transaction_id"] == 100001
    assert row["player_id"] == 500001
    assert row["player_name"] == "Trade Guy"
    assert row["from_team"] == "Chicago White Sox"
    assert row["to_team"] == "Kansas City Royals"
    # team falls through to toTeam (preferred) when no generic team key.
    assert row["team"] == "Kansas City Royals"
    assert row["transaction_type"] == "Trade"
    assert row["transaction_date"] == date(2023, 7, 31)


def test_map_il_placement_uses_effective_date():
    # IL placement carries both ``date`` (announced) and ``effectiveDate``
    # (retroactive). We prefer effectiveDate for the DB column.
    row = map_transaction(_IL_PLACEMENT)
    assert row is not None
    assert row["transaction_date"] == date(2023, 5, 10)
    assert row["transaction_type"] == "Status Change"
    assert "UCL sprain" in row["description"]


def test_map_dfa_no_from_to():
    row = map_transaction(_DFA)
    assert row is not None
    assert row["from_team"] == ""
    assert row["to_team"] == ""
    assert row["team"] == "Los Angeles Dodgers"


def test_map_activation():
    row = map_transaction(_ACTIVATION)
    assert row is not None
    assert row["transaction_type"] == "Status Change"
    assert "activated" in row["description"].lower()


def test_map_dropped_when_missing_id():
    assert map_transaction(_BAD) is None


def test_map_transaction_handles_non_int_id():
    txn = {"id": "not-an-int", "person": {"id": 1, "fullName": "x"}}
    assert map_transaction(txn) is None


def test_map_transaction_tolerates_missing_fields():
    # Minimum viable payload.
    row = map_transaction({"id": 1})
    assert row is not None
    assert row["transaction_id"] == 1
    assert row["player_id"] is None
    assert row["player_name"] is None
    assert row["from_team"] == ""
    assert row["to_team"] == ""
    assert row["team"] == ""
    assert row["transaction_date"] is None


def test_map_transactions_dedups_by_id():
    dupes = [_TRADE, _IL_PLACEMENT, {**_TRADE, "description": "updated"}]
    out = map_transactions(dupes)
    assert len(out) == 2  # two unique ids
    # Last-wins: the updated description should survive.
    by_id = {r["transaction_id"]: r for r in out}
    assert by_id[100001]["description"] == "updated"


# ── Real fixture (431 rows from March 1-3, 2023) ───────────────────────────


@pytest.fixture(scope="module")
def real_fixture() -> list[dict]:
    if not _FIXTURE_PATH.exists():
        pytest.skip(f"fixture not present: {_FIXTURE_PATH}")
    with _FIXTURE_PATH.open() as f:
        return json.load(f).get("transactions", [])


def test_real_fixture_parses_without_errors(real_fixture):
    # Every row the StatsAPI actually ships for a 3-day spring-training window
    # must either map to a full row or drop cleanly via the ``id`` guard.
    mapped = map_transactions(real_fixture)
    assert len(mapped) > 0, "fixture yielded zero mapped rows"
    # Every mapped row must have the full schema.
    for r in mapped:
        assert set(r.keys()) >= set(TRANSACTION_COLUMNS)
        assert isinstance(r["transaction_id"], int)


def test_real_fixture_dates_all_in_march_2023(real_fixture):
    mapped = map_transactions(real_fixture)
    dates = [r["transaction_date"] for r in mapped if r["transaction_date"]]
    assert dates, "no dates parsed"
    for d in dates:
        assert d.year == 2023 and d.month == 3


# ── Date-window tiling ─────────────────────────────────────────────────────


def test_iter_month_windows_full_calendar_month():
    wins = list(iter_month_windows(date(2023, 1, 1), date(2023, 3, 31)))
    assert wins == [
        (date(2023, 1, 1), date(2023, 1, 31)),
        (date(2023, 2, 1), date(2023, 2, 28)),
        (date(2023, 3, 1), date(2023, 3, 31)),
    ]


def test_iter_month_windows_partial_ends():
    wins = list(iter_month_windows(date(2023, 3, 15), date(2023, 5, 10)))
    assert wins == [
        (date(2023, 3, 15), date(2023, 3, 31)),
        (date(2023, 4, 1), date(2023, 4, 30)),
        (date(2023, 5, 1), date(2023, 5, 10)),
    ]


def test_iter_month_windows_year_boundary():
    wins = list(iter_month_windows(date(2023, 12, 20), date(2024, 1, 5)))
    assert wins == [
        (date(2023, 12, 20), date(2023, 12, 31)),
        (date(2024, 1, 1), date(2024, 1, 5)),
    ]


def test_iter_month_windows_single_day():
    wins = list(iter_month_windows(date(2023, 6, 15), date(2023, 6, 15)))
    assert wins == [(date(2023, 6, 15), date(2023, 6, 15))]


# ── _parse_iso_date ────────────────────────────────────────────────────────


def test_parse_iso_date_variants():
    assert _parse_iso_date("2023-05-10") == date(2023, 5, 10)
    assert _parse_iso_date("2023-05-10T00:00:00Z") == date(2023, 5, 10)
    assert _parse_iso_date(None) is None
    assert _parse_iso_date("") is None
    assert _parse_iso_date("not-a-date") is None


# ── upsert_rows() idempotency + schema match ───────────────────────────────


@pytest.fixture()
def mem_conn():
    conn = duckdb.connect(":memory:")
    create_tables(conn)
    yield conn
    conn.close()


def test_upsert_round_trip_matches_schema(mem_conn):
    rows = map_transactions([_TRADE, _IL_PLACEMENT, _DFA, _ACTIVATION])
    assert len(rows) == 4
    upsert_rows(mem_conn, rows)
    stored = mem_conn.execute(
        f"SELECT {', '.join(TRANSACTION_COLUMNS)} FROM transactions ORDER BY transaction_id"
    ).fetchdf()
    assert len(stored) == 4
    ids = stored["transaction_id"].astype(int).tolist()
    assert ids == [100001, 100002, 100003, 100004]


def test_upsert_is_idempotent(mem_conn):
    # Insert the same batch twice — row count must not double.
    rows = map_transactions([_TRADE, _IL_PLACEMENT])
    upsert_rows(mem_conn, rows)
    upsert_rows(mem_conn, rows)
    n = mem_conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    assert n == 2


def test_upsert_replaces_existing_row(mem_conn):
    # Initial batch writes description "v1"; second batch writes "v2".
    v1 = [
        {
            "transaction_id": 999,
            "player_id": 1,
            "player_name": "X",
            "team": "PHI",
            "from_team": "",
            "to_team": "",
            "transaction_type": "Trade",
            "description": "v1",
            "transaction_date": date(2023, 7, 1),
        }
    ]
    v2 = [{**v1[0], "description": "v2"}]
    upsert_rows(mem_conn, v1)
    upsert_rows(mem_conn, v2)
    desc = mem_conn.execute(
        "SELECT description FROM transactions WHERE transaction_id = 999"
    ).fetchone()[0]
    assert desc == "v2"


def test_upsert_handles_empty_batch(mem_conn):
    assert upsert_rows(mem_conn, []) == 0
    n = mem_conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    assert n == 0


def test_fixture_end_to_end_insert(mem_conn, real_fixture):
    # The real StatsAPI fixture must insert cleanly without any schema errors.
    rows = map_transactions(real_fixture)
    assert len(rows) > 100  # sanity: at least 100 rows in a 3-day fixture
    upsert_rows(mem_conn, rows)
    n = mem_conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    assert n == len(rows)
    # Idempotent rerun.
    upsert_rows(mem_conn, rows)
    n2 = mem_conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    assert n2 == n


# ── HTTP mocking ───────────────────────────────────────────────────────────


def _mock_response(status: int, body: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.json.return_value = body or {"transactions": []}
    if status >= 500:
        resp.raise_for_status.side_effect = requests.HTTPError(
            f"{status} Server Error", response=resp
        )
    elif status >= 400:
        # _fetch_window short-circuits on 4xx before raise_for_status.
        resp.raise_for_status.side_effect = requests.HTTPError(
            f"{status} Client Error", response=resp
        )
    return resp


def test_fetch_window_happy_path():
    session = MagicMock()
    session.get.return_value = _mock_response(
        200, {"transactions": [_TRADE, _IL_PLACEMENT]}
    )
    out = _fetch_window(session, date(2023, 3, 1), date(2023, 3, 31))
    assert len(out) == 2
    assert session.get.call_count == 1


def test_fetch_window_4xx_returns_empty():
    session = MagicMock()
    session.get.return_value = _mock_response(422, {"transactions": []})
    out = _fetch_window(session, date(2023, 3, 1), date(2023, 3, 31))
    assert out == []
    # 4xx short-circuits retries.
    assert session.get.call_count == 1


def test_fetch_window_retries_on_timeout():
    session = MagicMock()
    session.get.side_effect = [
        requests.Timeout("slow"),
        _mock_response(200, {"transactions": [_TRADE]}),
    ]
    with patch("scripts.ingest_transactions.time.sleep"):
        out = _fetch_window(session, date(2023, 3, 1), date(2023, 3, 31))
    assert len(out) == 1
    assert session.get.call_count == 2


def test_fetch_window_gives_up_after_max_retries():
    session = MagicMock()
    session.get.side_effect = requests.ConnectionError("down")
    with patch("scripts.ingest_transactions.time.sleep"):
        out = _fetch_window(session, date(2023, 3, 1), date(2023, 3, 31))
    assert out == []
    # max_retries=2 → 3 total attempts (initial + 2 retries).
    assert session.get.call_count == 3
