"""
Stuff+ pitch quality model.

Evaluates the physical "nastiness" of each pitch by training a gradient boosting
model on pitch characteristics (velocity, spin, movement, release point) to
predict run value.  Predictions are then scaled to a 100-centered metric where
higher values indicate better "stuff" for the pitcher.

The model uses scikit-learn's HistGradientBoostingRegressor for speed and
native missing-value handling.  Trained artefacts are persisted with joblib.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "stuff_model.pkl"

# ── Feature columns ─────────────────────────────────────────────────────────
_PHYSICAL_FEATURES: list[str] = [
    "release_speed",
    "release_spin_rate",
    "spin_axis",
    "pfx_x",
    "pfx_z",
    "release_extension",
    "release_pos_x",
    "release_pos_z",
]

# Categorical features that require encoding
_CAT_FEATURES: list[str] = [
    "p_throws",
    "stand",
    "pitch_type",
]

# Minimum number of pitches required for meaningful training
_MIN_TRAINING_PITCHES: int = 200


# ─────────────────────────────────────────────────────────────────────────────
# Training data preparation
# ─────────────────────────────────────────────────────────────────────────────


def prepare_training_data(
    conn: duckdb.DuckDBPyConnection,
) -> tuple[pd.DataFrame, pd.Series]:
    """Query the pitches table and prepare features / target for model training.

    Selects pitches where ``woba_denom > 0`` (plate-appearance-ending pitches)
    with complete physical characteristics.  The target is the *negative*
    ``woba_value`` so that a *higher* prediction corresponds to better stuff
    (lower run value allowed).

    Args:
        conn: Open DuckDB connection with a populated ``pitches`` table.

    Returns:
        Tuple of ``(features_df, target_series)`` ready for model fitting.

    Raises:
        ValueError: If fewer than ``_MIN_TRAINING_PITCHES`` usable rows exist.
    """
    query = f"""
        SELECT
            release_speed,
            release_spin_rate,
            spin_axis,
            pfx_x,
            pfx_z,
            release_extension,
            release_pos_x,
            release_pos_z,
            p_throws,
            stand,
            pitch_type,
            woba_value,
            woba_denom,
            game_date
        FROM pitches
        WHERE woba_denom > 0
          AND release_speed IS NOT NULL
          AND release_spin_rate IS NOT NULL
          AND pfx_x IS NOT NULL
          AND pfx_z IS NOT NULL
          AND pitch_type IS NOT NULL
    """
    df: pd.DataFrame = conn.execute(query).fetchdf()

    if len(df) < _MIN_TRAINING_PITCHES:
        raise ValueError(
            f"Only {len(df)} usable pitches found; need at least "
            f"{_MIN_TRAINING_PITCHES} for training."
        )

    # ── Encode categoricals ──────────────────────────────────────────────
    df["p_throws_enc"] = (df["p_throws"] == "R").astype(int)
    df["stand_enc"] = (df["stand"] == "R").astype(int)

    # Label-encode pitch_type (HistGBR handles ordinal encoding fine for
    # tree-based models; one-hot is unnecessary and inflates dimensionality).
    le = LabelEncoder()
    df["pitch_type_enc"] = le.fit_transform(df["pitch_type"].astype(str))

    feature_cols = _PHYSICAL_FEATURES + ["p_throws_enc", "stand_enc", "pitch_type_enc"]
    X = df[feature_cols].copy()
    y = -df["woba_value"].astype(float)  # negative: lower wOBA = better stuff

    return X, y


# ─────────────────────────────────────────────────────────────────────────────
# Model training
# ─────────────────────────────────────────────────────────────────────────────


def train_stuff_model(
    conn: duckdb.DuckDBPyConnection,
    model_path: Optional[str] = None,
) -> dict:
    """Train the Stuff+ gradient boosting model and persist it to disk.

    Uses a *date-based* 80/20 split (earliest 80 % of games for training,
    latest 20 % for evaluation) to avoid temporal leakage.

    Args:
        conn: Open DuckDB connection.
        model_path: Override for the saved model location.  Defaults to
                    ``models/stuff_model.pkl``.

    Returns:
        Dictionary of training metrics::

            {
                "r2_train": float,
                "r2_test": float,
                "rmse_train": float,
                "rmse_test": float,
                "n_train": int,
                "n_test": int,
                "feature_importances": dict[str, float],
                "model_path": str,
                "league_mean": float,
                "league_std": float,
                "pitch_type_classes": list[str],
            }
    """
    save_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Prepare data ─────────────────────────────────────────────────────
    # We need game_date for the temporal split, so we query separately.
    query = """
        SELECT
            release_speed,
            release_spin_rate,
            spin_axis,
            pfx_x,
            pfx_z,
            release_extension,
            release_pos_x,
            release_pos_z,
            p_throws,
            stand,
            pitch_type,
            woba_value,
            woba_denom,
            game_date
        FROM pitches
        WHERE woba_denom > 0
          AND release_speed IS NOT NULL
          AND release_spin_rate IS NOT NULL
          AND pfx_x IS NOT NULL
          AND pfx_z IS NOT NULL
          AND pitch_type IS NOT NULL
    """
    df: pd.DataFrame = conn.execute(query).fetchdf()

    if len(df) < _MIN_TRAINING_PITCHES:
        raise ValueError(
            f"Only {len(df)} usable pitches; need >= {_MIN_TRAINING_PITCHES}."
        )

    # ── Encode ───────────────────────────────────────────────────────────
    df["p_throws_enc"] = (df["p_throws"] == "R").astype(int)
    df["stand_enc"] = (df["stand"] == "R").astype(int)

    le = LabelEncoder()
    df["pitch_type_enc"] = le.fit_transform(df["pitch_type"].astype(str))

    feature_cols = _PHYSICAL_FEATURES + ["p_throws_enc", "stand_enc", "pitch_type_enc"]
    X = df[feature_cols]
    y = -df["woba_value"].astype(float)

    # ── Temporal train / test split ──────────────────────────────────────
    df["game_date"] = pd.to_datetime(df["game_date"])
    sorted_dates = df["game_date"].sort_values().unique()
    cutoff_idx = int(len(sorted_dates) * 0.8)
    cutoff_date = sorted_dates[cutoff_idx]

    train_mask = df["game_date"] < cutoff_date
    test_mask = ~train_mask

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    # ── Train ────────────────────────────────────────────────────────────
    model = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # ── Evaluate ─────────────────────────────────────────────────────────
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    r2_train = float(r2_score(y_train, pred_train))
    r2_test = float(r2_score(y_test, pred_test))
    rmse_train = float(np.sqrt(mean_squared_error(y_train, pred_train)))
    rmse_test = float(np.sqrt(mean_squared_error(y_test, pred_test)))

    # ── League mean/std for Stuff+ scaling ───────────────────────────────
    all_preds = model.predict(X)
    league_mean = float(np.mean(all_preds))
    league_std = float(np.std(all_preds))
    if league_std == 0:
        league_std = 1.0  # guard against degenerate data

    # ── Feature importances (permutation-based from the model internals) ─
    # HistGBR does not have .feature_importances_ in older sklearn; fall back.
    try:
        raw_imp = model.feature_importances_
    except AttributeError:
        raw_imp = np.zeros(len(feature_cols))

    importances = {col: round(float(imp), 4) for col, imp in zip(feature_cols, raw_imp)}

    # ── Persist ──────────────────────────────────────────────────────────
    artefact = {
        "model": model,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "league_mean": league_mean,
        "league_std": league_std,
        "pitch_type_classes": list(le.classes_),
    }
    joblib.dump(artefact, save_path)
    logger.info("Stuff+ model saved to %s", save_path)

    return {
        "r2_train": round(r2_train, 4),
        "r2_test": round(r2_test, 4),
        "rmse_train": round(rmse_train, 4),
        "rmse_test": round(rmse_test, 4),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "feature_importances": importances,
        "model_path": str(save_path),
        "league_mean": round(league_mean, 6),
        "league_std": round(league_std, 6),
        "pitch_type_classes": list(le.classes_),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stuff+ calculation
# ─────────────────────────────────────────────────────────────────────────────


def _load_model(model_path: Optional[str] = None) -> dict:
    """Load the persisted Stuff+ model artefact.

    Returns:
        Dictionary containing the model, encoders, and scaling parameters.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Stuff+ model not found at {path}. "
            "Run train_stuff_model() first."
        )
    return joblib.load(path)


def calculate_stuff_plus(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    season: Optional[int] = None,
    pitch_type: Optional[str] = None,
    model_path: Optional[str] = None,
) -> dict:
    """Calculate Stuff+ for a specific pitcher.

    Loads the trained model, retrieves the pitcher's pitches from the database,
    scores each pitch, and scales to a 100-centered metric.

    Args:
        conn: Open DuckDB connection.
        pitcher_id: MLB player ID.
        season: Restrict to this season year (optional).
        pitch_type: Restrict to this pitch type code (optional).
        model_path: Override for the model artefact path.

    Returns:
        Dictionary with overall Stuff+, per-pitch-type breakdown,
        feature contributions, and percentile rank.
    """
    artefact = _load_model(model_path)
    model = artefact["model"]
    le: LabelEncoder = artefact["label_encoder"]
    league_mean: float = artefact["league_mean"]
    league_std: float = artefact["league_std"]

    # ── Query pitcher data ───────────────────────────────────────────────
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    pitch_filter = "AND pitch_type = $3" if pitch_type else ""

    params: list = [pitcher_id]
    if season:
        params.append(season)
    if pitch_type:
        params.append(pitch_type)

    query = f"""
        SELECT
            release_speed,
            release_spin_rate,
            spin_axis,
            pfx_x,
            pfx_z,
            release_extension,
            release_pos_x,
            release_pos_z,
            p_throws,
            stand,
            pitch_type
        FROM pitches
        WHERE pitcher_id = $1
          AND release_speed IS NOT NULL
          AND release_spin_rate IS NOT NULL
          AND pfx_x IS NOT NULL
          AND pfx_z IS NOT NULL
          AND pitch_type IS NOT NULL
          {season_filter}
          {pitch_filter}
    """
    df: pd.DataFrame = conn.execute(query, params).fetchdf()

    if df.empty:
        return {
            "pitcher_id": pitcher_id,
            "overall_stuff_plus": None,
            "by_pitch_type": {},
            "feature_contributions": {},
            "percentile_rank": None,
        }

    # ── Encode ───────────────────────────────────────────────────────────
    df["p_throws_enc"] = (df["p_throws"] == "R").astype(int)
    df["stand_enc"] = (df["stand"] == "R").astype(int)

    # Handle unseen pitch types gracefully
    known_classes = set(le.classes_)
    df["pitch_type_safe"] = df["pitch_type"].apply(
        lambda x: x if x in known_classes else le.classes_[0]
    )
    df["pitch_type_enc"] = le.transform(df["pitch_type_safe"].astype(str))

    feature_cols = _PHYSICAL_FEATURES + ["p_throws_enc", "stand_enc", "pitch_type_enc"]
    X = df[feature_cols]

    # ── Predict ──────────────────────────────────────────────────────────
    raw_preds = model.predict(X)

    # Scale to 100-centered: stuff_plus = 100 + (pred - league_mean) / league_std * 10
    # The factor of 10 maps +-1 std to +-10 Stuff+ points.
    scaling_factor = 10.0
    stuff_scores = 100.0 + (raw_preds - league_mean) / league_std * scaling_factor
    df["stuff_plus"] = stuff_scores
    df["raw_pred"] = raw_preds

    overall_stuff = float(np.mean(stuff_scores))

    # ── Per-pitch-type breakdown ─────────────────────────────────────────
    by_pitch_type: dict = {}
    for pt, grp in df.groupby("pitch_type"):
        by_pitch_type[pt] = {
            "stuff_plus": round(float(grp["stuff_plus"].mean()), 1),
            "count": int(len(grp)),
            "avg_velo": round(float(grp["release_speed"].mean()), 1),
            "avg_spin": round(float(grp["release_spin_rate"].mean()), 0),
            "avg_pfx_x": round(float(grp["pfx_x"].mean()), 1),
            "avg_pfx_z": round(float(grp["pfx_z"].mean()), 1),
        }

    # ── Feature contributions ────────────────────────────────────────────
    feature_contributions = _compute_feature_contributions(
        model, X, feature_cols, league_mean, league_std, scaling_factor
    )

    # ── Percentile rank among all pitchers ───────────────────────────────
    percentile = _compute_percentile(conn, pitcher_id, overall_stuff, season, model_path)

    return {
        "pitcher_id": pitcher_id,
        "overall_stuff_plus": round(overall_stuff, 1),
        "by_pitch_type": by_pitch_type,
        "feature_contributions": feature_contributions,
        "percentile_rank": percentile,
    }


def _compute_feature_contributions(
    model,
    X: pd.DataFrame,
    feature_cols: list[str],
    league_mean: float,
    league_std: float,
    scaling_factor: float,
) -> dict[str, float]:
    """Estimate how much each feature group contributes to a pitcher's Stuff+.

    Uses a simple "leave-group-out" approach: replace each feature group with
    its column mean (league average) and measure the drop in predicted Stuff+.
    """
    base_pred = float(np.mean(model.predict(X)))

    groups = {
        "velocity": ["release_speed"],
        "movement": ["pfx_x", "pfx_z"],
        "spin": ["release_spin_rate", "spin_axis"],
        "extension": ["release_extension"],
        "release_point": ["release_pos_x", "release_pos_z"],
    }

    contributions: dict[str, float] = {}
    for group_name, cols in groups.items():
        available_cols = [c for c in cols if c in X.columns]
        if not available_cols:
            contributions[group_name] = 0.0
            continue

        X_modified = X.copy()
        for c in available_cols:
            X_modified[c] = X_modified[c].mean()

        modified_pred = float(np.mean(model.predict(X_modified)))
        # Contribution = how much Stuff+ drops when we neutralise this feature
        delta = (base_pred - modified_pred) / league_std * scaling_factor
        contributions[group_name] = round(delta, 1)

    return contributions


def _compute_percentile(
    conn: duckdb.DuckDBPyConnection,
    pitcher_id: int,
    stuff_plus: float,
    season: Optional[int] = None,
    model_path: Optional[str] = None,
) -> float:
    """Compute percentile rank by checking how many pitchers have lower Stuff+.

    For efficiency, we sample up to 200 pitchers from the database rather than
    scoring every single pitcher.
    """
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $1" if season else ""
    params: list = []
    if season:
        params.append(season)

    query = f"""
        SELECT DISTINCT pitcher_id
        FROM pitches
        WHERE pitch_type IS NOT NULL
              AND release_speed IS NOT NULL
              {season_filter}
        GROUP BY pitcher_id
        HAVING COUNT(*) >= 100
    """
    pitcher_df = conn.execute(query, params).fetchdf()

    if pitcher_df.empty or len(pitcher_df) < 2:
        return 50.0  # not enough data for ranking

    # Sample up to 200 pitchers for efficiency
    pitcher_ids = pitcher_df["pitcher_id"].tolist()
    if len(pitcher_ids) > 200:
        rng = np.random.RandomState(42)
        pitcher_ids = rng.choice(pitcher_ids, size=200, replace=False).tolist()
        # Ensure the target pitcher is included
        if pitcher_id not in pitcher_ids:
            pitcher_ids[-1] = pitcher_id

    # For each pitcher, compute a quick average Stuff+ score
    artefact = _load_model(model_path)
    model = artefact["model"]
    le = artefact["label_encoder"]
    league_mean_val = artefact["league_mean"]
    league_std_val = artefact["league_std"]

    scores: list[float] = []

    id_list_str = ", ".join(str(int(pid)) for pid in pitcher_ids)
    season_filter2 = f"AND EXTRACT(YEAR FROM game_date) = {int(season)}" if season else ""

    bulk_query = f"""
        SELECT
            pitcher_id,
            release_speed,
            release_spin_rate,
            spin_axis,
            pfx_x,
            pfx_z,
            release_extension,
            release_pos_x,
            release_pos_z,
            p_throws,
            stand,
            pitch_type
        FROM pitches
        WHERE pitcher_id IN ({id_list_str})
          AND release_speed IS NOT NULL
          AND release_spin_rate IS NOT NULL
          AND pfx_x IS NOT NULL
          AND pfx_z IS NOT NULL
          AND pitch_type IS NOT NULL
          {season_filter2}
    """
    bulk_df = conn.execute(bulk_query).fetchdf()

    if bulk_df.empty:
        return 50.0

    # Encode
    bulk_df["p_throws_enc"] = (bulk_df["p_throws"] == "R").astype(int)
    bulk_df["stand_enc"] = (bulk_df["stand"] == "R").astype(int)
    known_classes = set(le.classes_)
    bulk_df["pitch_type_safe"] = bulk_df["pitch_type"].apply(
        lambda x: x if x in known_classes else le.classes_[0]
    )
    bulk_df["pitch_type_enc"] = le.transform(bulk_df["pitch_type_safe"].astype(str))

    feature_cols = _PHYSICAL_FEATURES + ["p_throws_enc", "stand_enc", "pitch_type_enc"]
    bulk_df["raw_pred"] = model.predict(bulk_df[feature_cols])
    bulk_df["stuff_plus"] = (
        100.0 + (bulk_df["raw_pred"] - league_mean_val) / league_std_val * 10.0
    )

    pitcher_avgs = bulk_df.groupby("pitcher_id")["stuff_plus"].mean()
    n_below = int((pitcher_avgs < stuff_plus).sum())
    percentile = round(n_below / len(pitcher_avgs) * 100, 1)

    return percentile


# ─────────────────────────────────────────────────────────────────────────────
# Grade mapping
# ─────────────────────────────────────────────────────────────────────────────


def get_pitch_grade(stuff_plus: float) -> str:
    """Map a Stuff+ value to a scouting grade string.

    Args:
        stuff_plus: The 100-centered Stuff+ metric.

    Returns:
        One of "Elite", "Plus-Plus", "Above Average", "Average",
        "Below Average", or "Poor".
    """
    if stuff_plus >= 130:
        return "Elite"
    if stuff_plus >= 115:
        return "Plus-Plus"
    if stuff_plus >= 105:
        return "Above Average"
    if stuff_plus >= 95:
        return "Average"
    if stuff_plus >= 85:
        return "Below Average"
    return "Poor"


# ─────────────────────────────────────────────────────────────────────────────
# Batch leaderboard
# ─────────────────────────────────────────────────────────────────────────────


def batch_calculate_stuff_plus(
    conn: duckdb.DuckDBPyConnection,
    season: Optional[int] = None,
    min_pitches: int = 100,
    model_path: Optional[str] = None,
) -> pd.DataFrame:
    """Calculate Stuff+ for all qualifying pitchers and return a leaderboard.

    Args:
        conn: Open DuckDB connection.
        season: Optional season filter.
        min_pitches: Minimum pitches to qualify for the leaderboard.
        model_path: Override for the model artefact path.

    Returns:
        DataFrame with columns: pitcher_id, name, overall_stuff_plus, grade,
        and per-pitch-type Stuff+ columns (FF_stuff, SL_stuff, etc.).
    """
    artefact = _load_model(model_path)
    model = artefact["model"]
    le: LabelEncoder = artefact["label_encoder"]
    league_mean: float = artefact["league_mean"]
    league_std: float = artefact["league_std"]

    season_filter = "AND EXTRACT(YEAR FROM game_date) = $1" if season else ""
    params: list = []
    if season:
        params.append(season)

    query = f"""
        SELECT
            pitcher_id,
            release_speed,
            release_spin_rate,
            spin_axis,
            pfx_x,
            pfx_z,
            release_extension,
            release_pos_x,
            release_pos_z,
            p_throws,
            stand,
            pitch_type
        FROM pitches
        WHERE release_speed IS NOT NULL
          AND release_spin_rate IS NOT NULL
          AND pfx_x IS NOT NULL
          AND pfx_z IS NOT NULL
          AND pitch_type IS NOT NULL
          {season_filter}
    """
    df = conn.execute(query, params).fetchdf()

    if df.empty:
        return pd.DataFrame(columns=["pitcher_id", "name", "overall_stuff_plus", "grade"])

    # ── Encode ───────────────────────────────────────────────────────────
    df["p_throws_enc"] = (df["p_throws"] == "R").astype(int)
    df["stand_enc"] = (df["stand"] == "R").astype(int)
    known_classes = set(le.classes_)
    df["pitch_type_safe"] = df["pitch_type"].apply(
        lambda x: x if x in known_classes else le.classes_[0]
    )
    df["pitch_type_enc"] = le.transform(df["pitch_type_safe"].astype(str))

    feature_cols = _PHYSICAL_FEATURES + ["p_throws_enc", "stand_enc", "pitch_type_enc"]
    df["raw_pred"] = model.predict(df[feature_cols])
    df["stuff_plus"] = 100.0 + (df["raw_pred"] - league_mean) / league_std * 10.0

    # ── Filter by minimum pitch count ────────────────────────────────────
    pitcher_counts = df.groupby("pitcher_id").size()
    qualifying = pitcher_counts[pitcher_counts >= min_pitches].index
    df = df[df["pitcher_id"].isin(qualifying)]

    if df.empty:
        return pd.DataFrame(columns=["pitcher_id", "name", "overall_stuff_plus", "grade"])

    # ── Aggregate ────────────────────────────────────────────────────────
    overall = (
        df.groupby("pitcher_id")["stuff_plus"]
        .mean()
        .reset_index()
        .rename(columns={"stuff_plus": "overall_stuff_plus"})
    )
    overall["overall_stuff_plus"] = overall["overall_stuff_plus"].round(1)
    overall["grade"] = overall["overall_stuff_plus"].apply(get_pitch_grade)

    # Per-pitch-type aggregates
    pt_agg = (
        df.groupby(["pitcher_id", "pitch_type"])["stuff_plus"]
        .mean()
        .reset_index()
    )
    pt_pivot = pt_agg.pivot(
        index="pitcher_id", columns="pitch_type", values="stuff_plus"
    ).reset_index()
    # Rename columns to "<PT>_stuff"
    pt_pivot.columns = [
        f"{c}_stuff" if c != "pitcher_id" else c for c in pt_pivot.columns
    ]
    for col in pt_pivot.columns:
        if col.endswith("_stuff"):
            pt_pivot[col] = pt_pivot[col].round(1)

    result = overall.merge(pt_pivot, on="pitcher_id", how="left")

    # ── Join player names ────────────────────────────────────────────────
    try:
        names_df = conn.execute(
            "SELECT player_id AS pitcher_id, full_name AS name FROM players"
        ).fetchdf()
        result = result.merge(names_df, on="pitcher_id", how="left")
    except Exception:
        result["name"] = None

    # Reorder columns: pitcher_id, name first
    front_cols = ["pitcher_id", "name", "overall_stuff_plus", "grade"]
    other_cols = [c for c in result.columns if c not in front_cols]
    result = result[front_cols + sorted(other_cols)]

    result = result.sort_values("overall_stuff_plus", ascending=False).reset_index(drop=True)

    return result
