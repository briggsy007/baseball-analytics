"""
Tests for scripts/ingest_injury_labels.py.

Covers:
    1. classify_injury() on 10 known description strings (pure unit test).
    2. is_placement() / is_activation() discriminators.
    3. End-to-end build_labels() on an in-memory DuckDB with 5+ fake transactions.
    4. write_labels() round-trip through parquet.
    5. Idempotency: running the script twice yields the same row count with
       no duplicates.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import pytest

# Ensure the project root is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.ingest_injury_labels import (  # noqa: E402
    _match_activations,
    build_labels,
    classify_injury,
    is_activation,
    is_placement,
    write_labels,
)
from src.db.schema import create_tables  # noqa: E402


# ── classify_injury() ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("description", "expected"),
    [
        # 1. Tommy John surgery — clean keyword hit.
        ("Jacob deGrom underwent Tommy John surgery.", "tommy_john"),
        # 2. Alt wording for TJ.
        ("Pitcher X had UCL reconstruction surgery.", "tommy_john"),
        # 3. UCL sprain, non-surgical.
        ("placed on the 60-day IL with a UCL sprain.", "ucl_sprain"),
        # 4. Rotator cuff.
        ("placed on IL with rotator cuff inflammation.", "rotator_cuff"),
        # 5. Labrum / SLAP tear.
        ("diagnosed with a SLAP tear in the shoulder labrum.", "labrum"),
        # 6. Generic shoulder injury.
        ("placed on the 15-day injured list with right shoulder impingement.", "shoulder"),
        # 7. Forearm / flexor strain.
        ("placed on IL with right forearm flexor strain.", "forearm"),
        # 8. Elbow (non-UCL, non-TJ).
        ("placed on IL with elbow soreness.", "elbow"),
        # 9. Other arm (biceps).
        ("placed on IL with right biceps tendinitis.", "other_arm"),
        # 10. Non-arm (hamstring, oblique, covid, paternity, bereavement).
        ("placed on IL with a left hamstring strain.", "non_arm"),
        # 11. NULL / empty -> non_arm.
        (None, "non_arm"),
        ("", "non_arm"),
        # 12. Shoulder blade (scapula) should NOT classify as shoulder arm injury.
        ("fractured right shoulder blade.", "non_arm"),
    ],
)
def test_classify_injury(description: str | None, expected: str) -> None:
    assert classify_injury(description) == expected


# ── is_placement() / is_activation() ───────────────────────────────────────


def test_is_placement_true_on_il_placement() -> None:
    assert is_placement(
        "Philadelphia Phillies placed RHP Zack Wheeler on the 15-day injured list."
    )


def test_is_placement_false_on_activation() -> None:
    assert not is_placement(
        "Philadelphia Phillies activated RHP Orion Kerkering from the 15-day injured list."
    )


def test_is_placement_false_on_paternity_list() -> None:
    assert not is_placement("placed on the paternity list.")
    assert not is_placement("placed on the bereavement list.")
    assert not is_placement("placed on the restricted list.")


def test_is_activation_true() -> None:
    assert is_activation("activated RHP X from the 15-day injured list.")
    assert is_activation("reinstated RHP X from the 10-day injured list.")


def test_is_activation_false_on_placement() -> None:
    assert not is_activation("placed RHP X on the 15-day injured list.")


# ── End-to-end build_labels() against in-memory DuckDB ─────────────────────


@pytest.fixture
def mini_db() -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with 5 fake IL transactions + matched activations + pitches."""
    conn = duckdb.connect(":memory:")
    create_tables(conn)

    # Seed the pitches table so pitcher_id whitelisting passes.  Use minimal
    # rows -- only pitcher_id and game_date matter for the whitelist.
    pitches_rows = [
        {"game_pk": 1, "game_date": date(2020, 5, 1), "pitcher_id": 100, "batter_id": 999,
         "pitch_type": "FF", "pitch_name": "Four-Seam",
         "release_speed": 95.0, "release_spin_rate": 2300, "spin_axis": 180.0,
         "pfx_x": -6.0, "pfx_z": 15.0, "plate_x": 0.0, "plate_z": 2.5,
         "release_extension": 6.2, "release_pos_x": -1.5, "release_pos_y": 55.0,
         "release_pos_z": 5.8, "launch_speed": None, "launch_angle": None,
         "hit_distance": None, "hc_x": None, "hc_y": None, "bb_type": None,
         "estimated_ba": None, "estimated_woba": None,
         "delta_home_win_exp": 0.0, "delta_run_exp": 0.0,
         "inning": 1, "inning_topbot": "Top", "outs_when_up": 0,
         "balls": 0, "strikes": 0, "on_1b": 0, "on_2b": 0, "on_3b": 0,
         "stand": "R", "p_throws": "R",
         "at_bat_number": 1, "pitch_number": 1,
         "description": "called_strike", "events": None, "type": "S",
         "home_team": "PHI", "away_team": "NYM",
         "woba_value": None, "woba_denom": 0.0, "babip_value": None, "iso_value": None,
         "zone": 5, "effective_speed": 95.0,
         "if_fielding_alignment": "Standard", "of_fielding_alignment": "Standard",
         "fielder_2": 500},
        # Pitcher 200 (TJ case)
        {**{k: v for k, v in {}.items()}, "game_pk": 2, "game_date": date(2019, 6, 15),
         "pitcher_id": 200, "batter_id": 998, "pitch_type": "SL", "pitch_name": "Slider",
         "release_speed": 87.0, "release_spin_rate": 2500, "spin_axis": 100.0,
         "pfx_x": 3.0, "pfx_z": 2.0, "plate_x": 0.3, "plate_z": 2.0,
         "release_extension": 6.0, "release_pos_x": -1.2, "release_pos_y": 55.0,
         "release_pos_z": 5.5, "launch_speed": None, "launch_angle": None,
         "hit_distance": None, "hc_x": None, "hc_y": None, "bb_type": None,
         "estimated_ba": None, "estimated_woba": None,
         "delta_home_win_exp": 0.0, "delta_run_exp": 0.0,
         "inning": 2, "inning_topbot": "Bot", "outs_when_up": 1,
         "balls": 1, "strikes": 1, "on_1b": 0, "on_2b": 0, "on_3b": 0,
         "stand": "L", "p_throws": "R",
         "at_bat_number": 3, "pitch_number": 2,
         "description": "ball", "events": None, "type": "B",
         "home_team": "ATL", "away_team": "PHI",
         "woba_value": None, "woba_denom": 0.0, "babip_value": None, "iso_value": None,
         "zone": 4, "effective_speed": 87.0,
         "if_fielding_alignment": "Standard", "of_fielding_alignment": "Standard",
         "fielder_2": 501},
        # Pitcher 300 (shoulder case)
        {"game_pk": 3, "game_date": date(2022, 8, 1), "pitcher_id": 300, "batter_id": 997,
         "pitch_type": "FF", "pitch_name": "Four-Seam",
         "release_speed": 96.0, "release_spin_rate": 2400, "spin_axis": 190.0,
         "pfx_x": -5.0, "pfx_z": 16.0, "plate_x": -0.2, "plate_z": 2.8,
         "release_extension": 6.3, "release_pos_x": -1.7, "release_pos_y": 55.0,
         "release_pos_z": 5.9, "launch_speed": None, "launch_angle": None,
         "hit_distance": None, "hc_x": None, "hc_y": None, "bb_type": None,
         "estimated_ba": None, "estimated_woba": None,
         "delta_home_win_exp": 0.0, "delta_run_exp": 0.0,
         "inning": 3, "inning_topbot": "Top", "outs_when_up": 2,
         "balls": 0, "strikes": 2, "on_1b": 0, "on_2b": 0, "on_3b": 0,
         "stand": "R", "p_throws": "R",
         "at_bat_number": 5, "pitch_number": 3,
         "description": "swinging_strike", "events": None, "type": "S",
         "home_team": "LAD", "away_team": "SFG",
         "woba_value": None, "woba_denom": 0.0, "babip_value": None, "iso_value": None,
         "zone": 6, "effective_speed": 96.0,
         "if_fielding_alignment": "Standard", "of_fielding_alignment": "Standard",
         "fielder_2": 502},
    ]
    pitches_df = pd.DataFrame(pitches_rows)  # noqa: F841  (consumed by DuckDB below)
    conn.execute("INSERT INTO pitches SELECT * FROM pitches_df")

    # Seed the transactions table with 6 rows:
    # - 3 pitcher IL placements (TJ, UCL, shoulder)
    # - 1 non-arm pitcher placement (hamstring)
    # - 1 non-pitcher placement (position player; must be filtered out)
    # - 1 activation paired to the UCL placement (provides il_end_date)
    # - 1 paternity-list placement (must be dropped)
    # NOTE: keep pitcher names free of arm/shoulder/elbow substrings -- the
    # classifier scans the full description, which includes the name.
    tx_rows = [
        # TJ surgery -- pitcher 200, joins to pitches.
        (1001, 200, "Alex Pitcher", "LAD", None, "LAD", "Status Change",
         "Los Angeles Dodgers placed RHP Alex Pitcher on the 60-day injured list. Tommy John surgery.",
         date(2019, 7, 1)),
        # UCL sprain -- pitcher 100, joins to pitches.  Activation 30 days later.
        (1002, 100, "Ben Wheeler", "PHI", None, "PHI", "Status Change",
         "Philadelphia Phillies placed RHP Ben Wheeler on the 15-day injured list. UCL sprain.",
         date(2020, 5, 10)),
        (1003, 100, "Ben Wheeler", "PHI", None, "PHI", "Status Change",
         "Philadelphia Phillies activated RHP Ben Wheeler from the 15-day injured list.",
         date(2020, 6, 10)),
        # Shoulder -- pitcher 300, joins to pitches.
        (1004, 300, "Carl Steady", "LAD", None, "LAD", "Status Change",
         "Los Angeles Dodgers placed RHP Carl Steady on the 15-day injured list. Right shoulder inflammation.",
         date(2022, 8, 15)),
        # Non-arm pitcher placement -- pitcher 200.
        (1005, 200, "Alex Pitcher", "LAD", None, "LAD", "Status Change",
         "Los Angeles Dodgers placed RHP Alex Pitcher on the 10-day injured list. Right hamstring strain.",
         date(2018, 4, 15)),
        # Non-pitcher (position player) placement -- player_id 999, NOT in pitches.
        # The description does not mention RHP/LHP, so the fallback filter excludes too.
        (1006, 999, "Johnny Bat", "NYY", None, "NYY", "Status Change",
         "New York Yankees placed SS Johnny Bat on the 10-day injured list. Left oblique strain.",
         date(2021, 6, 1)),
        # Paternity list -- must be dropped entirely.
        (1007, 100, "Ben Wheeler", "PHI", None, "PHI", "Status Change",
         "Philadelphia Phillies placed RHP Ben Wheeler on the paternity list.",
         date(2021, 8, 1)),
    ]
    for row in tx_rows:
        conn.execute(
            "INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            list(row),
        )
    return conn


def test_build_labels_end_to_end(mini_db: duckdb.DuckDBPyConnection) -> None:
    labels, stats = build_labels(mini_db)

    # Expected: 4 IL stints (TJ, UCL-sprain, shoulder, hamstring).
    # Paternity-list row is dropped.  Position-player is filtered out.
    assert len(labels) == 4, f"Got {len(labels)} rows: {labels[['pitcher_name','injury_type']]}"

    # Injury-type assignments.  Two rows for "Alex Pitcher": TJ + non_arm.
    pairs = set(zip(labels["pitcher_name"], labels["injury_type"]))
    assert ("Alex Pitcher", "tommy_john") in pairs
    assert ("Alex Pitcher", "non_arm") in pairs
    assert ("Ben Wheeler", "ucl_sprain") in pairs
    assert ("Carl Steady", "shoulder") in pairs

    # Activation pairing: Ben Wheeler's il_end_date should be 2020-06-10.
    bw = labels[
        (labels["pitcher_name"] == "Ben Wheeler") & (labels["injury_type"] == "ucl_sprain")
    ].iloc[0]
    assert bw["il_end_date"] == date(2020, 6, 10)

    # Usable sample size: all 3 distinct pitchers join to pitches.
    assert stats.usable_sample_size == 4  # 4 stints, all joinable
    assert stats.joinable_arm_injuries == 3  # TJ + UCL + shoulder (not hamstring)

    # Activations matched: exactly 1 (Big Wheel).
    assert stats.activations_matched == 1

    # All rows flagged as source='transactions'.
    assert (labels["source"] == "transactions").all()


def test_build_labels_idempotent_write(mini_db: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
    """Writing twice must leave the same dataset on disk -- no duplicates."""
    labels, _ = build_labels(mini_db)
    out = tmp_path / "injury_labels.parquet"

    write_labels(labels, out)
    first = pq.read_table(str(out)).to_pandas()

    write_labels(labels, out)
    second = pq.read_table(str(out)).to_pandas()

    assert len(first) == len(second)
    pd.testing.assert_frame_equal(
        first.sort_values(["pitcher_id", "il_date"]).reset_index(drop=True),
        second.sort_values(["pitcher_id", "il_date"]).reset_index(drop=True),
    )


def test_build_labels_empty_transactions_errors() -> None:
    """build_labels() must raise a clear RuntimeError with remediation when empty."""
    conn = duckdb.connect(":memory:")
    create_tables(conn)
    # Transactions table exists but is empty.
    with pytest.raises(RuntimeError, match="(?i)empty|remediation"):
        build_labels(conn)


def test_output_parquet_schema(mini_db: duckdb.DuckDBPyConnection, tmp_path: Path) -> None:
    """The written parquet must have exactly the 8 documented columns."""
    labels, _ = build_labels(mini_db)
    out = tmp_path / "injury_labels.parquet"
    write_labels(labels, out)

    table = pq.read_table(str(out))
    expected = {
        "pitcher_id", "pitcher_name", "season", "il_date",
        "il_end_date", "injury_type", "injury_description_raw", "source",
    }
    assert set(table.column_names) == expected


def test_match_activations_skips_null_player_id() -> None:
    """Regression: activations with NULL player_id must not crash int() coercion.

    Not every transaction row involves a player — team-level "Status Change"
    rows have NULL player_id.  Those rows must be silently skipped rather
    than crash the ingest with ``TypeError: int() argument must be ... not
    'NAType'``.
    """
    placements = pd.DataFrame(
        {
            "player_id": pd.array([100], dtype="Int64"),
            "il_date": [pd.Timestamp("2020-05-10")],
        }
    )
    # One activation for player 100 (valid) + one with NULL player_id.
    activations = pd.DataFrame(
        {
            "player_id": pd.array([100, pd.NA], dtype="Int64"),
            "transaction_date": [
                pd.Timestamp("2020-06-10"),
                pd.Timestamp("2020-06-15"),
            ],
        }
    )

    # Must not raise.
    result = _match_activations(placements, activations)

    # Output is aligned to the placements index (length 1).
    assert len(result) == 1
    # The valid placement/activation pairing is preserved.
    assert pd.Timestamp(result.iloc[0]) == pd.Timestamp("2020-06-10")


def test_match_activations_skips_null_placement_player_id() -> None:
    """A placement with NULL player_id must not crash; it returns NaT."""
    placements = pd.DataFrame(
        {
            "player_id": pd.array([pd.NA, 100], dtype="Int64"),
            "il_date": [
                pd.Timestamp("2020-04-01"),
                pd.Timestamp("2020-05-10"),
            ],
        }
    )
    activations = pd.DataFrame(
        {
            "player_id": pd.array([100], dtype="Int64"),
            "transaction_date": [pd.Timestamp("2020-06-10")],
        }
    )

    # Must not raise.
    result = _match_activations(placements, activations)

    # Series aligned to the 2-row placements index.
    assert len(result) == 2
    # NULL-player row gets NaT; valid row gets its matching activation.
    assert pd.isna(result.iloc[0])
    assert pd.Timestamp(result.iloc[1]) == pd.Timestamp("2020-06-10")


def test_output_parquet_empty_labels_still_writes(tmp_path: Path) -> None:
    """An empty label DataFrame should still produce a valid parquet file."""
    empty = pd.DataFrame(
        columns=[
            "pitcher_id", "pitcher_name", "season", "il_date",
            "il_end_date", "injury_type", "injury_description_raw", "source",
        ]
    )
    out = tmp_path / "empty.parquet"
    write_labels(empty, out)
    assert out.exists()
    table = pq.read_table(str(out))
    assert table.num_rows == 0
    assert set(table.column_names) == {
        "pitcher_id", "pitcher_name", "season", "il_date",
        "il_end_date", "injury_type", "injury_description_raw", "source",
    }
