"""
Tests for ``scripts/baseline_comparison.py``.

Covers:
- Correlation math on a constructed fixture with known effects.
- Biggest-movers selection logic.
- Join-key handling when a player is in one table but not the other.
- Proxy-WAR fetch against an in-memory DuckDB fixture populated with
  synthetic season stats.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.baseline_comparison import (  # noqa: E402
    _fetch_batter_traditional_war,
    _fetch_pitcher_traditional_war,
    biggest_movers,
    compute_metrics,
    merge_with_traditional,
)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_correlation(self):
        causal = np.linspace(-2, 5, 10)
        trad = causal.copy()
        m = compute_metrics(causal, trad, n_bootstrap=50)
        assert m["n"] == 10
        assert m["pearson_r"] == pytest.approx(1.0, abs=1e-6)
        assert m["spearman_rho"] == pytest.approx(1.0, abs=1e-6)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-6)
        assert m["mae"] == pytest.approx(0.0, abs=1e-6)
        lo, hi = m["pearson_r_ci"]
        assert lo <= 1.0 + 1e-6 and hi <= 1.0 + 1e-6

    def test_negative_correlation(self):
        causal = np.linspace(0, 5, 10)
        trad = -causal
        m = compute_metrics(causal, trad, n_bootstrap=50)
        assert m["pearson_r"] == pytest.approx(-1.0, abs=1e-6)
        assert m["spearman_rho"] == pytest.approx(-1.0, abs=1e-6)

    def test_zero_correlation_noise(self):
        rng = np.random.RandomState(0)
        causal = rng.normal(size=500)
        trad = rng.normal(size=500)
        m = compute_metrics(causal, trad, n_bootstrap=100)
        assert abs(m["pearson_r"]) < 0.2
        assert m["pearson_r_ci"] is not None
        lo, hi = m["pearson_r_ci"]
        assert lo < 0 < hi or abs(m["pearson_r"]) < 0.1

    def test_nan_handling(self):
        causal = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        trad = np.array([1.1, 2.2, 3.3, np.nan, 4.9])
        m = compute_metrics(causal, trad, n_bootstrap=25)
        assert m["n"] == 3
        assert m["pearson_r"] is not None

    def test_too_few_points(self):
        m = compute_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert m["n"] == 2
        assert m["pearson_r"] is None

    def test_constant_causal_returns_none(self):
        causal = np.ones(10)
        trad = np.arange(10, dtype=float)
        m = compute_metrics(causal, trad)
        assert m["pearson_r"] is None

    def test_known_fixture_ten_players(self):
        """Constructed 10-player fixture with a known r around +0.85."""
        causal = np.array([3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5])
        trad = np.array([3.5, 2.0, 2.2, 1.6, 0.8, 0.9, 0.1, -0.3, -1.2, -0.9])
        m = compute_metrics(causal, trad, n_bootstrap=200, random_state=7)
        assert m["n"] == 10
        assert 0.8 < m["pearson_r"] <= 1.0
        assert 0.8 < m["spearman_rho"] <= 1.0
        assert m["pearson_r_ci"] is not None
        lo, hi = m["pearson_r_ci"]
        assert lo <= m["pearson_r"] <= hi


# ---------------------------------------------------------------------------
# merge_with_traditional + biggest_movers
# ---------------------------------------------------------------------------

def _make_causal_df(players: list[dict]) -> pd.DataFrame:
    rows = []
    for p in players:
        rows.append({
            "player_id": p["player_id"],
            "causal_war": p["causal_war"],
            "pa": p.get("pa", 400),
            "ci_low": p.get("ci_low"),
            "ci_high": p.get("ci_high"),
            "sparse": p.get("sparse", False),
        })
    return pd.DataFrame(rows)


def _make_trad_batters(players: list[dict]) -> pd.DataFrame:
    rows = []
    for p in players:
        rows.append({
            "player_id": p["player_id"],
            "trad_war": p["trad_war"],
            "pa_total": p.get("pa_total", 400),
            "n_seasons": p.get("n_seasons", 1),
            "war_source": p.get("war_source", "proxy_from_ops_and_pa"),
            "position_type": "batter",
        })
    return pd.DataFrame(rows)


def _make_trad_pitchers(players: list[dict]) -> pd.DataFrame:
    rows = []
    for p in players:
        rows.append({
            "player_id": p["player_id"],
            "trad_war": p["trad_war"],
            "ip_total": p.get("ip_total", 100.0),
            "n_seasons": p.get("n_seasons", 1),
            "war_source": p.get("war_source", "proxy_from_era_and_ip"),
            "position_type": "pitcher",
        })
    return pd.DataFrame(rows)


class TestMergeWithTraditional:
    def test_basic_batter_merge(self):
        causal = _make_causal_df([
            {"player_id": 1, "causal_war": 3.0},
            {"player_id": 2, "causal_war": 1.0},
        ])
        trad_b = _make_trad_batters([
            {"player_id": 1, "trad_war": 2.8},
            {"player_id": 2, "trad_war": 0.9},
        ])
        trad_p = _make_trad_pitchers([])
        merged = merge_with_traditional(causal, trad_b, trad_p, pa_min=100)
        assert set(merged["player_id"]) == {1, 2}
        assert (merged["position"] == "batter").all()
        assert merged["rank_diff"].abs().sum() == 0  # ranks align

    def test_pitcher_classification_via_pitcher_ids(self):
        causal = _make_causal_df([
            {"player_id": 10, "causal_war": 2.5, "pa": 5},
            {"player_id": 20, "causal_war": 3.0},
        ])
        trad_b = _make_trad_batters([
            {"player_id": 20, "trad_war": 2.9},
        ])
        trad_p = _make_trad_pitchers([
            {"player_id": 10, "trad_war": 2.4, "ip_total": 120.0},
        ])
        merged = merge_with_traditional(
            causal, trad_b, trad_p, pitcher_ids={10}, pa_min=100, ip_min=20,
        )
        got = merged.set_index("player_id")
        assert got.loc[10, "position"] == "pitcher"
        assert got.loc[20, "position"] == "batter"
        assert got.loc[10, "trad_war"] == pytest.approx(2.4)
        assert got.loc[20, "trad_war"] == pytest.approx(2.9)

    def test_player_in_only_one_table_is_dropped_if_no_war(self):
        """A player with no trad WAR at all should drop out of the merge."""
        causal = _make_causal_df([
            {"player_id": 1, "causal_war": 3.0},
            {"player_id": 99, "causal_war": 2.0},  # not in either trad table
        ])
        trad_b = _make_trad_batters([{"player_id": 1, "trad_war": 2.8}])
        trad_p = _make_trad_pitchers([])
        merged = merge_with_traditional(causal, trad_b, trad_p, pa_min=100)
        assert 99 not in set(merged["player_id"])
        assert 1 in set(merged["player_id"])

    def test_below_pa_min_is_dropped(self):
        causal = _make_causal_df([
            {"player_id": 1, "causal_war": 0.2, "pa": 400},
            {"player_id": 2, "causal_war": 0.1, "pa": 400},
        ])
        trad_b = _make_trad_batters([
            {"player_id": 1, "trad_war": 0.3, "pa_total": 50},   # below pa_min
            {"player_id": 2, "trad_war": 0.1, "pa_total": 400},
        ])
        trad_p = _make_trad_pitchers([])
        merged = merge_with_traditional(causal, trad_b, trad_p, pa_min=100)
        assert set(merged["player_id"]) == {2}

    def test_ranks_are_computed(self):
        causal = _make_causal_df([
            {"player_id": i, "causal_war": float(i)} for i in [1, 2, 3, 4, 5]
        ])
        trad_b = _make_trad_batters([
            {"player_id": i, "trad_war": float(6 - i)} for i in [1, 2, 3, 4, 5]
        ])
        trad_p = _make_trad_pitchers([])
        merged = merge_with_traditional(causal, trad_b, trad_p, pa_min=100)
        # rank_causal: 1->5, 5->1; rank_trad: 1->1, 5->5
        got = merged.set_index("player_id")
        assert got.loc[5, "rank_causal"] == 1
        assert got.loc[5, "rank_trad"] == 5
        assert got.loc[1, "rank_causal"] == 5
        assert got.loc[1, "rank_trad"] == 1


class TestBiggestMovers:
    def test_selects_top_k(self):
        # Build a merged frame where player 1 is massively over-valued
        # (high causal, low trad) and player 10 massively under-valued.
        rows = []
        for i in range(1, 11):
            rows.append({
                "player_id": i,
                "name": f"P{i}",
                "position": "batter",
                "causal_war": 11 - i,  # descending: 10, 9, ..., 1
                "trad_war": i,         # ascending: 1, 2, ..., 10
                "pa_total": 500,
                "ip_total": 0,
            })
        merged = pd.DataFrame(rows)
        merged["rank_causal"] = merged["causal_war"].rank(ascending=False, method="min").astype(int)
        merged["rank_trad"] = merged["trad_war"].rank(ascending=False, method="min").astype(int)
        merged["rank_diff"] = merged["rank_trad"] - merged["rank_causal"]
        over, under = biggest_movers(merged, k=3)
        # Over-valued: rank_causal much better than rank_trad --> rank_diff most negative.
        # Player 1 has rank_causal=1, rank_trad=10 --> diff=9 (NOT negative).
        # In this construction, player 10 has rank_causal=10, rank_trad=1 --> diff=-9 (most negative).
        # So player 10 is the largest "over-valued" per rank_diff sign convention
        # documented in biggest_movers (over = most-negative diff).
        assert 10 in over["player_id"].tolist()
        assert 1 in under["player_id"].tolist()
        assert len(over) == 3
        assert len(under) == 3

    def test_empty_input(self):
        empty = pd.DataFrame(columns=["player_id", "rank_diff"])
        over, under = biggest_movers(empty, k=5)
        assert over.empty and under.empty


# ---------------------------------------------------------------------------
# Proxy WAR fetch against in-memory DB
# ---------------------------------------------------------------------------

@pytest.fixture
def db_with_seasons():
    import duckdb
    from src.db.schema import create_tables

    conn = duckdb.connect(":memory:")
    create_tables(conn)

    # Insert a handful of synthetic batting rows for 2023-2024.
    batting_rows = [
        # player_id, season, pa, ops
        (101, 2023, 600, 0.900),  # great batter
        (101, 2024, 600, 0.880),
        (102, 2023, 550, 0.720),  # avg batter
        (102, 2024, 500, 0.700),
        (103, 2023, 300, 0.600),  # poor batter
        (103, 2024, 150, 0.580),
        (104, 2023, 400, 0.950),  # star
    ]
    for pid, season, pa, ops in batting_rows:
        conn.execute(
            """
            INSERT INTO season_batting_stats
            (player_id, season, pa, ab, h, doubles, triples, hr, rbi, bb, so,
             sb, cs, ba, obp, slg, ops, iso, k_pct, bb_pct)
            VALUES ($1,$2,$3,$4,$5,0,0,0,0,0,0,0,0,$6,$7,$8,$9,0.15,0.22,0.08)
            """,
            [pid, season, pa, pa - 50, int(pa * 0.25), ops / 3, ops / 3, ops / 3, ops],
        )

    # Synthetic pitching rows
    pitching_rows = [
        # player_id, season, ip, era
        (201, 2023, 180.0, 2.80),  # ace
        (201, 2024, 190.0, 3.00),
        (202, 2023, 150.0, 4.20),  # avg
        (202, 2024, 140.0, 4.40),
        (203, 2023, 60.0, 6.10),   # bad
        (203, 2024, 40.0, 6.50),
    ]
    for pid, season, ip, era in pitching_rows:
        conn.execute(
            """
            INSERT INTO season_pitching_stats
            (player_id, season, w, l, sv, g, gs, ip, era, whip,
             k_pct, bb_pct, hr_per_9, k_per_9, bb_per_9)
            VALUES ($1,$2,10,5,0,30,30,$3,$4,1.20,0.25,0.08,1.0,9.0,3.0)
            """,
            [pid, season, ip, era],
        )
    yield conn
    conn.close()


class TestTraditionalWarFetch:
    def test_batter_proxy_ordering(self, db_with_seasons):
        df = _fetch_batter_traditional_war(db_with_seasons, 2023, 2024)
        assert set(df["player_id"]) == {101, 102, 103, 104}
        df = df.set_index("player_id")
        # Star with .900+ OPS should score higher than avg .700 OPS batter.
        assert df.loc[101, "trad_war"] > df.loc[102, "trad_war"]
        assert df.loc[104, "trad_war"] > df.loc[103, "trad_war"]
        # No real war data -> proxy label
        assert (df["war_source"] == "proxy_from_ops_and_pa").all()

    def test_pitcher_proxy_ordering(self, db_with_seasons):
        df = _fetch_pitcher_traditional_war(db_with_seasons, 2023, 2024)
        assert set(df["player_id"]) == {201, 202, 203}
        df = df.set_index("player_id")
        assert df.loc[201, "trad_war"] > df.loc[202, "trad_war"]
        assert df.loc[202, "trad_war"] > df.loc[203, "trad_war"]
        assert (df["war_source"] == "proxy_from_era_and_ip").all()

    def test_empty_window_returns_empty(self, db_with_seasons):
        df = _fetch_batter_traditional_war(db_with_seasons, 2099, 2100)
        assert df.empty
