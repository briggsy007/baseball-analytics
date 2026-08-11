"""Cluster-honest CI machinery in the DPI validation harness (WS3.6).

Recurrence guard for audit finding 9 ("pooled bootstrap CIs ignore team
clustering"). ``scripts/defensive_pressing_validation.py`` used to publish an
iid-row percentile bootstrap for every gate CI; a window that carries the same
franchise in several seasons then gets an interval that is too narrow, because
the resample treats correlated rows as independent draws.

These tests pin the replacement:

  * ``_pair_metrics`` REQUIRES cluster labels -- omitting them is a TypeError,
    not a silent fallback to an iid interval;
  * the machinery is the SAME implementation the WS3.1 reliability run uses
    (imported from ``scripts/dpi_reliability_2026.py``, not copied);
  * on data with real within-cluster duplication the cluster interval is wider
    than the iid interval the harness used to publish, and the cluster-robust
    SE exceeds the iid SE;
  * mask/cluster alignment, determinism, and degenerate-input handling.

No DB access, no model fitting: everything runs on synthetic arrays.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

dpv = pytest.importorskip("defensive_pressing_validation")
dpi_rel = pytest.importorskip("dpi_reliability_2026")


# ---------------------------------------------------------------------------
# Fixtures: a two-season window where each franchise contributes two rows
# ---------------------------------------------------------------------------

def _clustered_window(n_teams: int = 30, rho: float = 0.6, seed: int = 7):
    """Two seasons x ``n_teams`` franchises with a franchise-level effect.

    Each franchise has a persistent (x, y) offset, so its two rows are
    positively correlated -- exactly the structure a pooled 2023-24 or
    2023-25 gate window has, and exactly what the iid bootstrap ignores.
    """
    rng = np.random.default_rng(seed)
    team_x = rng.normal(size=n_teams)
    team_y = rho * team_x + np.sqrt(1 - rho ** 2) * rng.normal(size=n_teams)
    xs, ys, clusters = [], [], []
    for season in (2023, 2024):
        noise_scale = 0.35
        xs.append(team_x + noise_scale * rng.normal(size=n_teams))
        ys.append(team_y + noise_scale * rng.normal(size=n_teams))
        clusters.append(np.array([f"T{i:02d}" for i in range(n_teams)]))
        del season
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(clusters),
    )


def _iid_percentile_ci(x, y, n_boot=1000, seed=42):
    """The interval the harness used to publish (iid rows, percentile)."""
    rng = np.random.RandomState(seed)
    n = len(x)
    boots = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        a, b = x[idx], y[idx]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        boots.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ---------------------------------------------------------------------------
# 1. The iid path is structurally unavailable
# ---------------------------------------------------------------------------

def test_iid_bootstrap_helper_is_gone():
    """The old iid-row bootstrap must not exist as a callable at all."""
    assert not hasattr(dpv, "_bootstrap_ci"), (
        "_bootstrap_ci (iid-row percentile bootstrap) is back -- audit "
        "finding 9 recurrence"
    )


def test_pair_metrics_requires_cluster_labels():
    x, y, _ = _clustered_window()
    with pytest.raises(TypeError):
        dpv._pair_metrics(x, y, n_boot=50, random_state=1)  # no clusters


def test_pair_metrics_rejects_misaligned_clusters():
    x, y, clusters = _clustered_window()
    with pytest.raises(ValueError, match="clusters length"):
        dpv._pair_metrics(
            x, y, clusters[:-1], n_boot=50, random_state=1, n_boot_wild=50,
        )


def test_machinery_is_imported_not_reimplemented():
    """Both DPI CI publishers must share one implementation."""
    assert dpv.pairs_cluster_bootstrap_r is dpi_rel.pairs_cluster_bootstrap_r
    assert dpv.wild_cluster_bootstrap_t is dpi_rel.wild_cluster_bootstrap_t


# ---------------------------------------------------------------------------
# 2. The cluster interval is honestly wider
# ---------------------------------------------------------------------------

def test_cluster_ci_is_wider_than_the_iid_ci_it_replaced():
    x, y, clusters = _clustered_window()
    out = dpv._pair_metrics(
        x, y, clusters, n_boot=2000, random_state=42, n_boot_wild=499,
    )
    lo, hi = out["pearson_r_ci"]
    iid_lo, iid_hi = _iid_percentile_ci(x, y, n_boot=2000, seed=42)
    assert (hi - lo) > (iid_hi - iid_lo), (
        f"cluster CI [{lo}, {hi}] is not wider than the iid CI "
        f"[{iid_lo:.4f}, {iid_hi:.4f}] on clustered data"
    )
    assert lo < out["pearson_r"] < hi


def test_cluster_robust_se_exceeds_iid_se():
    x, y, clusters = _clustered_window()
    out = dpv._pair_metrics(
        x, y, clusters, n_boot=500, random_state=42, n_boot_wild=499,
    )
    wild = out["wild_cluster"]
    assert wild is not None
    assert wild["n_clusters"] == 30
    assert wild["cluster_robust_se"] > wild["iid_se"]
    assert wild["se_inflation_vs_iid"] > 1.0
    lo, hi = wild["wild_cluster_t_ci"]
    assert lo < hi


def test_ci_method_string_is_recorded_and_names_the_cluster_bootstrap():
    x, y, clusters = _clustered_window()
    out = dpv._pair_metrics(
        x, y, clusters, n_boot=200, random_state=42, n_boot_wild=199,
    )
    assert "cluster" in out["ci_method"].lower()
    assert "iid-row bootstrap removed" in out["ci_method"]
    assert out["n_clusters"] == 30
    assert out["n"] == 60


# ---------------------------------------------------------------------------
# 3. Alignment, determinism, degenerate inputs
# ---------------------------------------------------------------------------

def test_non_finite_rows_drop_their_cluster_labels_in_lockstep():
    x, y, clusters = _clustered_window()
    y = y.copy()
    # Kill both rows of one franchise; its cluster must disappear too.
    doomed = clusters == "T00"
    y[doomed] = np.nan
    out = dpv._pair_metrics(
        x, y, clusters, n_boot=200, random_state=42, n_boot_wild=199,
    )
    assert out["n"] == 58
    assert out["n_clusters"] == 29


def test_results_are_deterministic_for_a_fixed_seed():
    x, y, clusters = _clustered_window()
    kwargs = dict(n_boot=300, random_state=99, n_boot_wild=199)
    first = dpv._pair_metrics(x, y, clusters, **kwargs)
    second = dpv._pair_metrics(x, y, clusters, **kwargs)
    assert first["pearson_r_ci"] == second["pearson_r_ci"]
    assert first["spearman_rho_ci"] == second["spearman_rho_ci"]
    assert (
        first["wild_cluster"]["wild_cluster_t_ci"]
        == second["wild_cluster"]["wild_cluster_t_ci"]
    )


def test_single_row_per_cluster_still_produces_an_interval():
    """Single-season windows have one row per franchise; the cluster
    bootstrap reduces to the ordinary pairs bootstrap there."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=30)
    y = 0.5 * x + rng.normal(size=30)
    clusters = np.array([f"T{i:02d}" for i in range(30)])
    out = dpv._pair_metrics(
        x, y, clusters, n_boot=500, random_state=42, n_boot_wild=299,
    )
    lo, hi = out["pearson_r_ci"]
    assert lo < out["pearson_r"] < hi
    assert out["n_clusters"] == out["n"] == 30


def test_degenerate_inputs_return_none_without_raising():
    empty = dpv._pair_metrics(
        np.array([]), np.array([]), np.array([]),
        n_boot=100, random_state=1, n_boot_wild=99,
    )
    assert empty["n"] == 0
    assert empty["pearson_r"] is None and empty["pearson_r_ci"] is None

    constant = dpv._pair_metrics(
        np.ones(20), np.arange(20.0), np.array([f"T{i%5}" for i in range(20)]),
        n_boot=100, random_state=1, n_boot_wild=99,
    )
    assert constant["pearson_r"] is None
    assert constant["wild_cluster"] is None


# ---------------------------------------------------------------------------
# 4. Per-team-season CIs resample GAMES, not BIP rows
# ---------------------------------------------------------------------------

def test_games_within_team_season_ci_brackets_the_mean():
    rng = np.random.default_rng(11)
    games = list(rng.normal(loc=0.4, scale=2.0, size=162))
    lo, hi = dpv._games_within_team_season_ci(
        games, n_boot=2000, random_state=42,
    )
    mean = float(np.mean(games))
    assert lo is not None and hi is not None
    assert lo < mean < hi
    # A 162-game season mean of a ~2.0-sd quantity has a tight but non-zero
    # interval: roughly +/- 2 * 2.0/sqrt(162) ~ 0.31 wide on each side.
    assert 0.05 < (hi - lo) < 1.5


def test_games_within_team_season_ci_accepts_per_game_result_dicts():
    """calculate_team_dpi returns game dicts, not bare floats."""
    rng = np.random.default_rng(12)
    dict_games = [{"dpi": float(v)} for v in rng.normal(size=80)]
    lo, hi = dpv._games_within_team_season_ci(
        dict_games, n_boot=500, random_state=42,
    )
    assert lo is not None and hi is not None and lo < hi

    mixed = dict_games[:40] + [{"dpi": None}] * 5
    lo2, hi2 = dpv._games_within_team_season_ci(
        mixed, n_boot=500, random_state=42,
    )
    assert lo2 is not None and hi2 is not None


def test_games_within_team_season_ci_declines_tiny_samples():
    assert dpv._games_within_team_season_ci(
        [0.1, -0.2, 0.3], n_boot=500, random_state=1,
    ) == (None, None)
    assert dpv._games_within_team_season_ci(
        [], n_boot=500, random_state=1,
    ) == (None, None)


# ---------------------------------------------------------------------------
# 5. The published policy travels with the numbers
# ---------------------------------------------------------------------------

def test_ci_policy_constant_documents_the_recurrence_guard():
    assert "cluster" in dpv.CI_METHOD
    assert "audit finding 9" in dpv.CI_METHOD


# ---------------------------------------------------------------------------
# 6. Per-team-season seeds are reproducible ACROSS PROCESSES
# ---------------------------------------------------------------------------
# The published dpi_mean_ci_lo / dpi_mean_ci_hi columns are only meaningful if
# a re-run at the same --random-state reproduces them byte-for-byte: the
# metrics payload advertises random_state + a ci_policy block, and claims.yaml
# pins a sha256 of the CSV. A str-hash-derived seed breaks that silently,
# because PYTHONHASHSEED is randomized per process and is pinned nowhere in
# this repo.

def test_team_season_seed_pinned_values():
    """crc32-derived seeds; pinned so a formula change is a visible break."""
    base = 42
    assert dpv._team_season_seed(base, 2023, "PHI") == 42 + 2023 * 1000 + 119
    assert dpv._team_season_seed(base, 2024, "PHI") == 42 + 2024 * 1000 + 119
    assert dpv._team_season_seed(base, 2023, "LAD") == 42 + 2023 * 1000 + 375
    assert dpv._team_season_seed(base, 2023, "NYY") == 42 + 2023 * 1000 + 534
    # the real MLB franchise codes (incl. both OAK and ATH) land on 31
    # distinct offsets mod 997 -- no team-seed collisions in a live window
    teams = (
        "ARI ATL BAL BOS CHC CWS CIN CLE COL DET HOU KC LAA LAD MIA MIL MIN "
        "NYM NYY OAK ATH PHI PIT SD SF SEA STL TB TEX TOR WSH"
    ).split()
    seeds = {dpv._team_season_seed(base, 2025, t) for t in teams}
    assert len(seeds) == len(teams) == 31
    # ...and seasons never alias onto each other
    assert dpv._team_season_seed(base, 2023, "PHI") != dpv._team_season_seed(
        base, 2024, "PHI",
    )


def test_team_season_seed_does_not_use_python_str_hash():
    """Recurrence guard: `hash(` must not reappear in the seed derivation."""
    import inspect

    for fn in (dpv._team_season_seed, dpv._compute_team_season_dpi):
        src = inspect.getsource(fn)
        assert "abs(hash(" not in src and "hash(str(" not in src, (
            f"{fn.__name__} derives a seed from PYTHONHASHSEED-randomized "
            "str hashing -- the published CI columns stop being reproducible"
        )
    # ...and the caller really routes through the helper.
    assert "_team_season_seed(" in inspect.getsource(dpv._compute_team_season_dpi)


@pytest.mark.parametrize("hashseed", ["0", "1"])
def test_team_season_seed_is_stable_under_pythonhashseed(hashseed):
    """Same seed in a fresh interpreter with an explicit PYTHONHASHSEED."""
    import json
    import os
    import subprocess

    cases = [(42, 2023, "PHI"), (42, 2025, "ATH"), (7, 2024, "SF")]
    expected = [dpv._team_season_seed(*c) for c in cases]

    code = (
        "import sys, json;"
        f"sys.path.insert(0, {str(_ROOT)!r});"
        f"sys.path.insert(0, {str(_ROOT / 'scripts')!r});"
        "import defensive_pressing_validation as d;"
        f"print(json.dumps([d._team_season_seed(*c) for c in {cases!r}]))"
    )
    env = dict(os.environ, PYTHONHASHSEED=hashseed)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, env=env, cwd=str(_ROOT), timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert json.loads(out.stdout.strip().splitlines()[-1]) == expected


def test_games_within_team_season_ci_is_reproducible_for_a_fixed_seed():
    rng = np.random.default_rng(21)
    games = [{"dpi": float(v)} for v in rng.normal(size=140)]
    seed = dpv._team_season_seed(42, 2023, "PHI")
    first = dpv._games_within_team_season_ci(games, n_boot=800, random_state=seed)
    second = dpv._games_within_team_season_ci(games, n_boot=800, random_state=seed)
    other = dpv._games_within_team_season_ci(
        games, n_boot=800, random_state=dpv._team_season_seed(42, 2023, "LAD"),
    )
    assert first == second
    assert first[0] is not None and other[0] is not None


def test_ci_policy_records_the_seed_rule():
    """The published payload documents the reproducibility contract."""
    import inspect

    src = inspect.getsource(dpv)
    assert "per_team_season_seed" in src
    assert "crc32" in src


def test_validation_config_exposes_the_three_bootstrap_knobs():
    cfg = dpv.DPIValidationConfig()
    assert cfg.n_bootstrap_ci >= 1000
    assert cfg.n_bootstrap_wild_cluster >= 99
    assert cfg.n_bootstrap_team_season >= 500

    args = dpv.build_argparser().parse_args([])
    assert args.n_bootstrap_wild_cluster == cfg.n_bootstrap_wild_cluster
    assert args.n_bootstrap_team_season == cfg.n_bootstrap_team_season
