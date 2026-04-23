"""
Shared pytest fixtures for the baseball analytics test suite.

Fixtures create an in-memory DuckDB database with the full schema and load
realistic synthetic data.  By default, all tests run against synthetic data
only -- no production database is required.

Pass ``--use-prod-db`` to opt into tests that read from the production DB.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from src.db.schema import create_tables

logger = logging.getLogger(__name__)

_PROD_DB_PATH = Path(r"C:\Users\hunte\projects\baseball\data\baseball.duckdb")

# Known pitcher IDs for targeted sampling
ZACK_WHEELER_ID = 554430
KNOWN_PITCHER_IDS = [ZACK_WHEELER_ID]


# ---------------------------------------------------------------------------
# pytest plugin hooks
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the ``--use-prod-db`` command-line flag."""
    parser.addoption(
        "--use-prod-db",
        action="store_true",
        default=False,
        help="Enable tests that require the production DuckDB database.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``prod_db`` marker so ``--strict-markers`` won't fail."""
    config.addinivalue_line(
        "markers",
        "prod_db: marks tests that require the production database (requires --use-prod-db flag)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip ``@pytest.mark.prod_db`` tests unless ``--use-prod-db`` is given."""
    if config.getoption("--use-prod-db"):
        return
    skip_prod = pytest.mark.skip(reason="requires --use-prod-db to run")
    for item in items:
        if "prod_db" in item.keywords:
            item.add_marker(skip_prod)


# ---------------------------------------------------------------------------
# Production data loader (only used when --use-prod-db is active)
# ---------------------------------------------------------------------------


def _try_load_real_data(mem_conn: duckdb.DuckDBPyConnection) -> bool:
    """Attempt to sample real data from the production database.

    Loads ~100 pitches per season and all pitches for known pitcher IDs.
    Returns True if successful, False otherwise.
    """
    try:
        prod_conn = duckdb.connect(str(_PROD_DB_PATH), read_only=True)
    except Exception as exc:
        logger.warning("Cannot open production DB: %s", exc)
        return False

    try:
        # Sample 100 pitches per season using window function
        sample_query = """
            SELECT *
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY EXTRACT(YEAR FROM game_date)
                           ORDER BY random()
                       ) AS _rn
                FROM pitches
            ) sub
            WHERE _rn <= 100
        """
        sample_df = prod_conn.execute(sample_query).fetchdf()
        sample_df = sample_df.drop(columns=["_rn"], errors="ignore")

        # Also load all pitches for known pitchers (for targeted tests)
        pitcher_ids_str = ", ".join(str(pid) for pid in KNOWN_PITCHER_IDS)
        pitcher_query = f"""
            SELECT * FROM pitches
            WHERE pitcher_id IN ({pitcher_ids_str})
        """
        pitcher_df = prod_conn.execute(pitcher_query).fetchdf()

        # Combine and deduplicate
        combined = pd.concat([sample_df, pitcher_df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["game_pk", "at_bat_number", "pitch_number"],
            keep="first",
        )

        # Load into the in-memory DB
        mem_conn.execute("INSERT INTO pitches SELECT * FROM combined")

        # Also load players if the table exists
        try:
            players_df = prod_conn.execute("SELECT * FROM players").fetchdf()
            if not players_df.empty:
                mem_conn.execute("INSERT INTO players SELECT * FROM players_df")
        except Exception:
            pass

        # Load games
        try:
            game_pks = combined["game_pk"].unique().tolist()
            if game_pks:
                gp_str = ", ".join(str(g) for g in game_pks[:500])
                games_df = prod_conn.execute(
                    f"SELECT * FROM games WHERE game_pk IN ({gp_str})"
                ).fetchdf()
                if not games_df.empty:
                    mem_conn.execute("INSERT INTO games SELECT * FROM games_df")
        except Exception:
            pass

        prod_conn.close()
        row_count = mem_conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
        logger.info("Loaded %d real pitches into test DB.", row_count)
        return row_count > 0

    except Exception as exc:
        logger.warning("Failed to sample real data: %s", exc)
        try:
            prod_conn.close()
        except Exception:
            pass
        return False


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

# Pitch-type physical profiles: (mean_velo, std_velo, mean_spin, std_spin,
#                                 mean_pfx_x, std_pfx_x, mean_pfx_z, std_pfx_z)
_PITCH_PROFILES: dict[str, tuple[float, ...]] = {
    "FF": (94.5, 2.0, 2300, 200, -6, 3, 15, 2),
    "SI": (93.0, 2.0, 2100, 200, -10, 3, 8, 2),
    "SL": (85.0, 2.5, 2500, 250, 3, 2, 2, 2),
    "CU": (79.0, 3.0, 2700, 300, 5, 3, -8, 3),
    "CH": (85.0, 2.5, 1700, 200, -8, 3, 4, 2),
    "FC": (88.0, 2.0, 2400, 200, -2, 2, 10, 2),
    "FS": (82.0, 2.5, 1400, 200, -7, 3, 1, 2),
    "KC": (78.0, 3.0, 2600, 300, 4, 3, -10, 3),
}

_PITCH_TYPES = list(_PITCH_PROFILES.keys())

_DESCRIPTIONS = [
    "called_strike",
    "swinging_strike",
    "ball",
    "foul",
    "hit_into_play",
    "hit_into_play_no_out",
    "swinging_strike_blocked",
    "blocked_ball",
]

_EVENTS = [
    None, None, None, None,  # Most pitches have no event
    "single", "double", "triple", "home_run",
    "strikeout", "walk", "field_out",
    "grounded_into_double_play", "force_out",
    "sac_fly", "fielders_choice",
]

_BB_TYPES = ["fly_ball", "ground_ball", "line_drive", "popup"]
_TEAMS = [
    "PHI", "NYM", "ATL", "WSH", "MIA", "NYY", "BOS", "TBR", "TOR", "BAL",
    "CHC", "MIL", "STL", "CIN", "PIT", "LAD", "SFG", "SDP", "ARI", "COL",
    "HOU", "TEX", "SEA", "LAA", "OAK", "MIN", "CLE", "CHW", "DET", "KCR",
]


def _generate_synthetic_pitches(
    rng: np.random.RandomState,
    n_pitches: int = 1200,
    seasons: tuple[int, ...] = tuple(range(2015, 2027)),
) -> pd.DataFrame:
    """Build a DataFrame of synthetic Statcast pitches with realistic distributions.

    Generates *at least* ``n_pitches`` rows (rounded up to fill the last
    season evenly).  Zack Wheeler (554430) is assigned ~20 % of each season's
    pitches so that pitcher-specific tests always have data.

    The returned DataFrame matches every column in ``src.db.schema._create_pitches``.
    """
    per_season = max(n_pitches // len(seasons), 100)
    rows: list[dict] = []
    game_pk_counter = 100_000

    for season in seasons:
        for i in range(per_season):
            # Wheeler gets ~20% of pitches
            pitcher_id = ZACK_WHEELER_ID if i < (per_season // 5) else rng.randint(100_000, 700_000)
            batter_id = rng.randint(100_000, 700_000)
            game_pk = game_pk_counter + (i // 6)  # ~6 pitches per game_pk
            pt = rng.choice(_PITCH_TYPES)

            # Physical profile for this pitch type
            prof = _PITCH_PROFILES[pt]
            velo = rng.normal(prof[0], prof[1])
            spin = rng.normal(prof[2], prof[3])
            pfx_x_val = rng.normal(prof[4], prof[5])
            pfx_z_val = rng.normal(prof[6], prof[7])

            desc = rng.choice(_DESCRIPTIONS)
            is_bip = desc.startswith("hit_into")
            event = rng.choice(_EVENTS) if is_bip else None
            woba_val = round(float(rng.uniform(0, 2.0)), 3) if event else None
            woba_den = 1.0 if event else 0.0

            month = int(rng.randint(3, 11))  # March-October
            day = int(rng.randint(1, 29))

            # Determine pitch result type code
            if "strike" in desc or desc == "foul":
                type_code = "S"
            elif desc == "ball" or desc == "blocked_ball":
                type_code = "B"
            else:
                type_code = "X"

            home = rng.choice(_TEAMS)
            away = rng.choice([t for t in _TEAMS if t != home])

            row = {
                "game_pk": game_pk,
                "game_date": f"{season}-{month:02d}-{day:02d}",
                "pitcher_id": pitcher_id,
                "batter_id": batter_id,
                "pitch_type": pt,
                "pitch_name": pt,
                "release_speed": round(float(np.clip(velo, 55, 105)), 1),
                "release_spin_rate": round(float(np.clip(spin, 100, 3800)), 0),
                "spin_axis": round(float(rng.uniform(0, 360)), 1),
                "pfx_x": round(float(np.clip(pfx_x_val, -20, 20)), 1),
                "pfx_z": round(float(np.clip(pfx_z_val, -20, 20)), 1),
                "plate_x": round(float(rng.normal(0, 0.6)), 2),
                "plate_z": round(float(np.clip(rng.normal(2.5, 0.6), 0.5, 5.0)), 2),
                "release_extension": round(float(rng.normal(6.2, 0.5)), 1),
                "release_pos_x": round(float(rng.normal(-1.5, 0.5)), 2),
                "release_pos_y": round(float(rng.normal(55, 0.5)), 2),
                "release_pos_z": round(float(rng.normal(5.8, 0.4)), 2),
                "launch_speed": round(float(rng.normal(88, 12)), 1) if event else None,
                "launch_angle": round(float(rng.normal(12, 20)), 1) if event else None,
                "hit_distance": round(float(rng.uniform(50, 420)), 0) if event else None,
                "hc_x": round(float(rng.uniform(20, 230)), 1) if event else None,
                "hc_y": round(float(rng.uniform(20, 220)), 1) if event else None,
                "bb_type": rng.choice(_BB_TYPES) if event else None,
                "estimated_ba": round(float(rng.uniform(0, 1)), 3) if event else None,
                "estimated_woba": round(float(rng.uniform(0, 2)), 3) if event else None,
                "delta_home_win_exp": round(float(rng.normal(0, 0.03)), 4),
                "delta_run_exp": round(float(rng.normal(0, 0.1)), 4),
                "inning": int(rng.randint(1, 10)),
                "inning_topbot": rng.choice(["Top", "Bot"]),
                "outs_when_up": int(rng.randint(0, 3)),
                "balls": int(rng.randint(0, 4)),
                "strikes": int(rng.randint(0, 3)),
                "on_1b": int(rng.choice([0, 1])),
                "on_2b": int(rng.choice([0, 1])),
                "on_3b": int(rng.choice([0, 1])),
                "stand": rng.choice(["L", "R"]),
                "p_throws": rng.choice(["L", "R"]),
                "at_bat_number": int(i // 5 + 1),
                "pitch_number": int(i % 5 + 1),
                "description": desc,
                "events": event,
                "type": type_code,
                "home_team": home,
                "away_team": away,
                "woba_value": woba_val,
                "woba_denom": woba_den,
                "babip_value": round(float(rng.uniform(0, 1)), 3) if event else None,
                "iso_value": round(float(rng.uniform(0, 1)), 3) if event else None,
                "zone": int(rng.randint(1, 15)),
                "effective_speed": round(float(np.clip(velo - rng.uniform(0, 3), 55, 105)), 1),
                "if_fielding_alignment": rng.choice(["Standard", "Infield shift", "Strategic"]),
                "of_fielding_alignment": rng.choice(["Standard", "Strategic", "4th outfielder"]),
                "fielder_2": int(rng.randint(100_000, 700_000)),
            }
            rows.append(row)

        game_pk_counter += 200

    return pd.DataFrame(rows)


def _generate_synthetic_players(rng: np.random.RandomState) -> pd.DataFrame:
    """Generate a small set of synthetic player records."""
    players = [
        {
            "player_id": ZACK_WHEELER_ID,
            "full_name": "Zack Wheeler",
            "team": "PHI",
            "position": "SP",
            "throws": "R",
            "bats": "L",
            "mlbam_id": ZACK_WHEELER_ID,
            "fg_id": "11122",
            "bref_id": "wheelza01",
        },
    ]
    # Add a handful of synthetic players
    for i in range(20):
        pid = 600_000 + i
        players.append(
            {
                "player_id": pid,
                "full_name": f"Test Player {i}",
                "team": rng.choice(list(_TEAMS)),
                "position": rng.choice(["SP", "RP", "C", "1B", "SS", "CF", "RF", "LF"]),
                "throws": rng.choice(["L", "R"]),
                "bats": rng.choice(["L", "R", "S"]),
                "mlbam_id": pid,
                "fg_id": str(20_000 + i),
                "bref_id": f"testpl{i:02d}",
            }
        )
    return pd.DataFrame(players)


def _generate_synthetic_games(
    rng: np.random.RandomState,
    pitches_df: pd.DataFrame,
) -> pd.DataFrame:
    """Generate game records for every unique game_pk in the pitch data."""
    game_pks = pitches_df[["game_pk", "game_date", "home_team", "away_team"]].drop_duplicates(
        subset=["game_pk"]
    )
    games = []
    venues = ["Citizens Bank Park", "Citi Field", "Truist Park", "Nationals Park", "loanDepot park"]
    for _, row in game_pks.iterrows():
        games.append(
            {
                "game_pk": int(row["game_pk"]),
                "game_date": row["game_date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "venue": rng.choice(venues),
                "game_type": "R",
                "status": "Final",
            }
        )
    return pd.DataFrame(games)


def _load_synthetic_data(conn: duckdb.DuckDBPyConnection) -> None:
    """Populate all core tables in the in-memory DB with synthetic data."""
    rng = np.random.RandomState(42)

    pitches_df = _generate_synthetic_pitches(rng, n_pitches=1200)
    conn.execute("INSERT INTO pitches SELECT * FROM pitches_df")

    players_df = _generate_synthetic_players(rng)
    conn.execute("INSERT INTO players SELECT * FROM players_df")

    games_df = _generate_synthetic_games(rng, pitches_df)
    conn.execute("INSERT INTO games SELECT * FROM games_df")

    count = conn.execute("SELECT COUNT(*) FROM pitches").fetchone()[0]
    logger.info("Generated %d synthetic pitches for testing.", count)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def db_conn(request: pytest.FixtureRequest):
    """In-memory DuckDB with the full schema and test data.

    By default only synthetic data is loaded (fast, deterministic, no external
    dependencies).  Pass ``--use-prod-db`` to attempt loading a sample of real
    Statcast data from the production database instead.
    """
    conn = duckdb.connect(":memory:")
    create_tables(conn)

    use_prod = request.config.getoption("--use-prod-db", default=False)

    loaded = False
    if use_prod:
        loaded = _try_load_real_data(conn)
        if not loaded:
            logger.warning(
                "Production DB requested but unavailable; falling back to synthetic data."
            )

    if not loaded:
        _load_synthetic_data(conn)

    yield conn
    conn.close()


@pytest.fixture(scope="session")
def sample_pitches_df(db_conn) -> pd.DataFrame:
    """DataFrame of all test pitches from the test DB."""
    return db_conn.execute("SELECT * FROM pitches").fetchdf()


@pytest.fixture(scope="session")
def sample_game_pitches(db_conn) -> pd.DataFrame:
    """All pitches from one specific game (the game with the most pitches)."""
    game_pk = db_conn.execute("""
        SELECT game_pk
        FROM pitches
        GROUP BY game_pk
        ORDER BY COUNT(*) DESC
        LIMIT 1
    """).fetchone()[0]
    return db_conn.execute(
        "SELECT * FROM pitches WHERE game_pk = $1 ORDER BY at_bat_number, pitch_number",
        [game_pk],
    ).fetchdf()


@pytest.fixture(scope="session")
def wheeler_pitches_df(db_conn) -> pd.DataFrame:
    """All test pitches thrown by Zack Wheeler (pitcher_id = 554430)."""
    return db_conn.execute(
        "SELECT * FROM pitches WHERE pitcher_id = $1 ORDER BY game_date, at_bat_number, pitch_number",
        [ZACK_WHEELER_ID],
    ).fetchdf()


@pytest.fixture(scope="session")
def synthetic_pitches_df() -> pd.DataFrame:
    """A standalone comprehensive synthetic pitch dataset (1000+ rows).

    This fixture is independent of the DB -- useful for unit tests that only
    need a DataFrame and should never touch any database.  It always returns
    the same deterministic dataset (seed=99).
    """
    rng = np.random.RandomState(99)
    return _generate_synthetic_pitches(rng, n_pitches=1200)


@pytest.fixture(scope="session")
def sample_players_df(db_conn) -> pd.DataFrame:
    """DataFrame of all players in the test DB."""
    return db_conn.execute("SELECT * FROM players").fetchdf()


# ---------------------------------------------------------------------------
# Model artifact fixtures (skip gracefully when artifacts are missing)
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODELS_DIR = _PROJECT_ROOT / "models"


@pytest.fixture(scope="session")
def stuff_model_artifact():
    """Load the Stuff+ model artifact; skip if not available."""
    import joblib

    path = _MODELS_DIR / "stuff_model.pkl"
    if not path.exists():
        pytest.skip(f"Stuff+ model artifact not found at {path}")
    return joblib.load(path)


@pytest.fixture(scope="session")
def mechanix_ae_artifact():
    """Load the MechanixAE universal model; skip if not available."""
    import torch

    path = _MODELS_DIR / "mechanix_ae_universal.pt"
    if not path.exists():
        pytest.skip(f"MechanixAE artifact not found at {path}")
    try:
        checkpoint = torch.load(path, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, weights_only=False)
    return checkpoint


@pytest.fixture(scope="session")
def pitchgpt_artifact():
    """Load the PitchGPT v2 model (35-dim CONTEXT_DIM); skip if not available.

    v2 matches the current ``CONTEXT_DIM`` in ``src/analytics/pitchgpt.py``.
    v1 (``pitchgpt_v1.pt``) uses the old 34-dim context and is still
    retained for the LSTM-delta narrative but is not loaded by these tests.
    """
    import torch

    path = _MODELS_DIR / "pitchgpt_v2.pt"
    if not path.exists():
        pytest.skip(f"PitchGPT artifact not found at {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


@pytest.fixture(scope="session")
def chemnet_artifact():
    """Load the ChemNet GNN+baseline; skip if not available."""
    import torch

    path = _MODELS_DIR / "chemnet_v1.pt"
    if not path.exists():
        pytest.skip(f"ChemNet artifact not found at {path}")
    return torch.load(path, map_location="cpu", weights_only=True)
