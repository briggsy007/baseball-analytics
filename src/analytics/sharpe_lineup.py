"""
Player Sharpe Ratio & Efficient Lineup Frontier model.

Applies Markowitz portfolio theory to baseball lineups: each batter is an
"asset" whose game-level wOBA is the return stream.  The Player Sharpe Ratio
(PSR) measures risk-adjusted offensive consistency, and the efficient frontier
identifies lineup combinations that maximise expected wOBA for a given
variance budget.

Key concepts:
- **Player Sharpe Ratio (PSR)**: (mean_game_wOBA - league_avg) / std_game_wOBA
- **Correlation matrix**: pairwise game-level wOBA correlation among teammates
- **Efficient frontier**: Markowitz optimisation via scipy.optimize.minimize
"""

from __future__ import annotations

import logging
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.analytics.base import BaseAnalyticsModel

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

LEAGUE_AVG_WOBA: float = 0.320
MIN_GAMES_DEFAULT: int = 40
ROLLING_WINDOW_DEFAULT: int = 40


# ── Helper: game-level wOBA aggregation ───────────────────────────────────────


def _game_level_woba(
    conn: duckdb.DuckDBPyConnection,
    batter_id: int,
    season: Optional[int] = None,
) -> pd.DataFrame:
    """Aggregate pitch-level wOBA to game-level for a single batter.

    Computes ``SUM(woba_value) / SUM(woba_denom)`` per game, keeping only
    games with at least one PA-ending event (``woba_denom > 0``).

    Args:
        conn: Open DuckDB connection.
        batter_id: MLB player ID.
        season: Optional season filter.

    Returns:
        DataFrame with columns ``game_pk``, ``game_date``, ``game_woba``,
        ``woba_value_sum``, ``woba_denom_sum``.
    """
    season_filter = "AND EXTRACT(YEAR FROM game_date) = $2" if season else ""
    params: list = [batter_id]
    if season:
        params.append(season)

    query = f"""
        SELECT
            game_pk,
            game_date,
            SUM(woba_value)  AS woba_value_sum,
            SUM(woba_denom)  AS woba_denom_sum
        FROM pitches
        WHERE batter_id = $1
          AND woba_denom IS NOT NULL
          {season_filter}
        GROUP BY game_pk, game_date
        HAVING SUM(woba_denom) > 0
        ORDER BY game_date
    """
    df = conn.execute(query, params).fetchdf()
    if not df.empty:
        df["game_woba"] = df["woba_value_sum"] / df["woba_denom_sum"]
    return df


def _batch_game_level_woba(
    conn: duckdb.DuckDBPyConnection,
    player_ids: list[int],
    season: Optional[int] = None,
) -> pd.DataFrame:
    """Aggregate pitch-level wOBA to game-level for multiple batters at once.

    Args:
        conn: Open DuckDB connection.
        player_ids: List of MLB player IDs.
        season: Optional season filter.

    Returns:
        DataFrame with columns ``batter_id``, ``game_pk``, ``game_date``,
        ``game_woba``.
    """
    if not player_ids:
        return pd.DataFrame(columns=["batter_id", "game_pk", "game_date", "game_woba"])

    # Safety: int() cast on each pid prevents SQL injection (non-numeric
    # values will raise ValueError before reaching the query).
    id_list = ", ".join(str(int(pid)) for pid in player_ids)
    season_filter = f"AND EXTRACT(YEAR FROM game_date) = {int(season)}" if season else ""

    query = f"""
        SELECT
            batter_id,
            game_pk,
            game_date,
            SUM(woba_value)  AS woba_value_sum,
            SUM(woba_denom)  AS woba_denom_sum
        FROM pitches
        WHERE batter_id IN ({id_list})
          AND woba_denom IS NOT NULL
          {season_filter}
        GROUP BY batter_id, game_pk, game_date
        HAVING SUM(woba_denom) > 0
        ORDER BY batter_id, game_date
    """
    df = conn.execute(query).fetchdf()
    if not df.empty:
        df["game_woba"] = df["woba_value_sum"] / df["woba_denom_sum"]
    return df


# ── Core analytics functions ──────────────────────────────────────────────────


def calculate_player_sharpe(
    conn: duckdb.DuckDBPyConnection,
    batter_id: int,
    season: Optional[int] = None,
    window: int = ROLLING_WINDOW_DEFAULT,
    league_avg: float = LEAGUE_AVG_WOBA,
) -> dict:
    """Compute the Player Sharpe Ratio for a single batter.

    PSR = (mean_game_wOBA - league_avg) / std_game_wOBA

    Uses the most recent *window* games if the batter has more.

    Args:
        conn: Open DuckDB connection.
        batter_id: MLB player ID.
        season: Optional season filter.
        window: Rolling window of games to consider.
        league_avg: League average wOBA (the "risk-free rate").

    Returns:
        Dictionary with keys ``batter_id``, ``psr``, ``mean_woba``,
        ``std_woba``, ``games``, ``game_woba_values``.
    """
    df = _game_level_woba(conn, batter_id, season)

    if df.empty:
        logger.warning("No game-level wOBA data for batter %d.", batter_id)
        return {
            "batter_id": batter_id,
            "psr": None,
            "mean_woba": None,
            "std_woba": None,
            "games": 0,
            "game_woba_values": [],
        }

    # Take the most recent `window` games
    recent = df.tail(window)
    game_wobas = recent["game_woba"].values

    mean_woba = float(np.mean(game_wobas))
    std_woba = float(np.std(game_wobas, ddof=1)) if len(game_wobas) > 1 else 0.0

    if std_woba == 0.0:
        psr = None  # undefined when there's no variance
    else:
        psr = (mean_woba - league_avg) / std_woba

    return {
        "batter_id": batter_id,
        "psr": round(psr, 3) if psr is not None else None,
        "mean_woba": round(mean_woba, 4),
        "std_woba": round(std_woba, 4),
        "games": len(recent),
        "game_woba_values": [round(float(v), 4) for v in game_wobas],
    }


def batch_player_sharpe(
    conn: duckdb.DuckDBPyConnection,
    team_id: Optional[str] = None,
    season: Optional[int] = None,
    min_games: int = MIN_GAMES_DEFAULT,
    league_avg: float = LEAGUE_AVG_WOBA,
) -> pd.DataFrame:
    """Compute PSR for every qualifying batter on a team (or league-wide).

    Args:
        conn: Open DuckDB connection.
        team_id: Three-letter team abbreviation (e.g. ``'PHI'``).
                 If None, returns league-wide results.
        season: Season year filter.
        min_games: Minimum number of games to qualify.
        league_avg: League average wOBA ("risk-free rate").

    Returns:
        DataFrame with columns ``batter_id``, ``name``, ``psr``,
        ``mean_woba``, ``std_woba``, ``games``, sorted by PSR descending.
    """
    # Identify qualifying batters
    season_filter = f"AND EXTRACT(YEAR FROM p.game_date) = {int(season)}" if season else ""

    if team_id:
        # Filter by team: join players table to get batter's team
        batter_query = f"""
            SELECT p.batter_id, COUNT(DISTINCT p.game_pk) AS games
            FROM pitches p
            INNER JOIN players pl ON p.batter_id = pl.player_id
            WHERE p.woba_denom IS NOT NULL
              AND p.woba_denom > 0
              AND pl.team = '{team_id}'
              {season_filter}
            GROUP BY p.batter_id
            HAVING COUNT(DISTINCT p.game_pk) >= {min_games}
        """
    else:
        batter_query = f"""
            SELECT p.batter_id, COUNT(DISTINCT p.game_pk) AS games
            FROM pitches p
            WHERE p.woba_denom IS NOT NULL
              AND p.woba_denom > 0
              {season_filter}
            GROUP BY p.batter_id
            HAVING COUNT(DISTINCT p.game_pk) >= {min_games}
        """
    batter_df = conn.execute(batter_query).fetchdf()

    if batter_df.empty:
        return pd.DataFrame(columns=["batter_id", "name", "psr", "mean_woba", "std_woba", "games"])

    player_ids = batter_df["batter_id"].tolist()
    game_data = _batch_game_level_woba(conn, player_ids, season)

    rows = []
    for pid in player_ids:
        player_games = game_data[game_data["batter_id"] == pid]
        if player_games.empty:
            continue

        game_wobas = player_games["game_woba"].values
        mean_woba = float(np.mean(game_wobas))
        std_woba = float(np.std(game_wobas, ddof=1)) if len(game_wobas) > 1 else 0.0

        if std_woba == 0.0:
            psr = None
        else:
            psr = (mean_woba - league_avg) / std_woba

        rows.append({
            "batter_id": int(pid),
            "psr": round(psr, 3) if psr is not None else None,
            "mean_woba": round(mean_woba, 4),
            "std_woba": round(std_woba, 4),
            "games": len(player_games),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["batter_id", "name", "psr", "mean_woba", "std_woba", "games"])

    # Attach player names
    try:
        names_df = conn.execute(
            "SELECT player_id AS batter_id, full_name AS name FROM players"
        ).fetchdf()
        result = result.merge(names_df, on="batter_id", how="left")
    except Exception:
        result["name"] = None

    # Reorder and sort
    front = ["batter_id", "name", "psr", "mean_woba", "std_woba", "games"]
    result = result[[c for c in front if c in result.columns]]
    result = result.sort_values("psr", ascending=False, na_position="last").reset_index(drop=True)
    return result


def compute_correlation_matrix(
    conn: duckdb.DuckDBPyConnection,
    player_ids: list[int],
    season: Optional[int] = None,
) -> pd.DataFrame:
    """Compute pairwise game-level wOBA correlation between players.

    Only games where both players participated are used for each pair.

    Args:
        conn: Open DuckDB connection.
        player_ids: List of batter IDs (typically from the same team).
        season: Optional season filter.

    Returns:
        Square DataFrame indexed and columned by ``batter_id`` with
        Pearson correlations. Diagonal is 1.0.
    """
    if len(player_ids) < 2:
        idx = pd.Index(player_ids, name="batter_id")
        return pd.DataFrame(
            np.eye(len(player_ids)),
            index=idx,
            columns=player_ids,
        )

    game_data = _batch_game_level_woba(conn, player_ids, season)

    if game_data.empty:
        idx = pd.Index(player_ids, name="batter_id")
        return pd.DataFrame(
            np.eye(len(player_ids)),
            index=idx,
            columns=player_ids,
        )

    # Pivot: rows=game_pk, columns=batter_id, values=game_woba
    pivot = game_data.pivot_table(
        index="game_pk",
        columns="batter_id",
        values="game_woba",
        aggfunc="first",
    )

    # Compute correlation (min_periods=5 to avoid spurious correlations)
    corr = pivot.corr(min_periods=5)

    # Ensure all requested players appear (fill missing with 0 correlation, 1 on diag)
    for pid in player_ids:
        if pid not in corr.columns:
            corr[pid] = 0.0
            corr.loc[pid] = 0.0
            corr.at[pid, pid] = 1.0

    # Reorder to match input order and fill NaN with 0 (except diagonal)
    corr = corr.reindex(index=player_ids, columns=player_ids)
    np.fill_diagonal(corr.values, 1.0)
    corr = corr.fillna(0.0)

    return corr


def optimize_lineup(
    player_stats: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    n_players: int = 9,
    risk_budget: Optional[float] = None,
) -> dict:
    """Find the optimal lineup using Markowitz mean-variance optimisation.

    Maximises expected portfolio wOBA minus a risk penalty:

        max  w^T * mu  -  (risk_budget / 2) * w^T * Sigma * w

    Subject to:
    - weights sum to 1
    - all weights >= 0
    - at most ``n_players`` players selected (non-zero weight)

    When ``risk_budget`` is None, a balanced default of 1.0 is used.

    Args:
        player_stats: DataFrame with columns ``batter_id``, ``mean_woba``,
                      ``std_woba``. Must include all players in the
                      correlation matrix.
        correlation_matrix: Square correlation DataFrame from
                           ``compute_correlation_matrix``.
        n_players: Number of players in the lineup (default 9).
        risk_budget: Risk aversion parameter. Higher = more conservative.

    Returns:
        Dictionary with:
        - ``weights``: dict mapping batter_id -> weight
        - ``expected_woba``: portfolio expected wOBA
        - ``portfolio_variance``: portfolio variance
        - ``portfolio_std``: portfolio standard deviation
        - ``sharpe``: portfolio Sharpe ratio
        - ``selected_players``: list of batter_ids with non-zero weight
    """
    if risk_budget is None:
        risk_budget = 1.0

    stats = player_stats.copy()
    if "batter_id" in stats.columns:
        stats = stats.set_index("batter_id")

    # Align player IDs between stats and correlation matrix
    common_ids = sorted(
        set(stats.index) & set(correlation_matrix.columns)
    )
    if len(common_ids) == 0:
        raise ValueError("No overlapping player IDs between stats and correlation matrix.")

    stats = stats.loc[common_ids]
    mu = stats["mean_woba"].values.astype(float)
    sigma_vec = stats["std_woba"].values.astype(float)

    # Build covariance matrix from correlation + individual std devs
    corr = correlation_matrix.loc[common_ids, common_ids].values.astype(float)
    cov = np.outer(sigma_vec, sigma_vec) * corr

    n = len(common_ids)

    # If we have fewer players than requested, use all of them
    effective_n = min(n_players, n)

    def neg_objective(w):
        """Negative of (expected return - risk penalty)."""
        ret = w @ mu
        var = w @ cov @ w
        return -(ret - (risk_budget / 2.0) * var)

    # Constraints and bounds
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    ]
    bounds = [(0.0, 1.0)] * n

    # Initial guess: equal weight on top-n_players by mean_woba
    w0 = np.zeros(n)
    top_indices = np.argsort(-mu)[:effective_n]
    w0[top_indices] = 1.0 / effective_n

    result = minimize(
        neg_objective,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-12},
    )

    weights = result.x
    # Zero out very small weights (numerical noise)
    weights[weights < 1e-4] = 0.0
    # Renormalise
    if weights.sum() > 0:
        weights = weights / weights.sum()

    expected_woba = float(weights @ mu)
    portfolio_var = float(weights @ cov @ weights)
    portfolio_std = float(np.sqrt(max(portfolio_var, 0.0)))

    if portfolio_std > 0:
        portfolio_sharpe = (expected_woba - LEAGUE_AVG_WOBA) / portfolio_std
    else:
        portfolio_sharpe = None

    weight_dict = {
        int(pid): round(float(w), 4)
        for pid, w in zip(common_ids, weights)
    }
    selected = [pid for pid, w in weight_dict.items() if w > 0]

    return {
        "weights": weight_dict,
        "expected_woba": round(expected_woba, 4),
        "portfolio_variance": round(portfolio_var, 6),
        "portfolio_std": round(portfolio_std, 4),
        "sharpe": round(portfolio_sharpe, 3) if portfolio_sharpe is not None else None,
        "selected_players": selected,
    }


def efficient_frontier(
    player_stats: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    n_points: int = 50,
    n_players: int = 9,
) -> list[dict]:
    """Trace the efficient frontier across a range of risk budgets.

    Each point on the frontier is the optimal portfolio for a different
    risk aversion parameter (from very aggressive to very conservative).

    Args:
        player_stats: DataFrame with ``batter_id``, ``mean_woba``, ``std_woba``.
        correlation_matrix: Square correlation DataFrame.
        n_points: Number of points to compute along the frontier.
        n_players: Lineup size constraint.

    Returns:
        List of dicts, each with ``expected_woba``, ``variance``,
        ``std``, ``risk_budget``, ``weights``.
    """
    # Risk budgets from very low (aggressive) to high (conservative)
    risk_budgets = np.logspace(-2, 2, n_points)

    frontier_points = []
    for rb in risk_budgets:
        try:
            result = optimize_lineup(
                player_stats,
                correlation_matrix,
                n_players=n_players,
                risk_budget=float(rb),
            )
            frontier_points.append({
                "expected_woba": result["expected_woba"],
                "variance": result["portfolio_variance"],
                "std": result["portfolio_std"],
                "risk_budget": round(float(rb), 4),
                "weights": result["weights"],
            })
        except Exception as exc:
            logger.debug("Skipping risk_budget=%.4f: %s", rb, exc)
            continue

    return frontier_points


def get_sharpe_leaderboard(
    conn: duckdb.DuckDBPyConnection,
    season: Optional[int] = None,
    min_games: int = MIN_GAMES_DEFAULT,
    league_avg: float = LEAGUE_AVG_WOBA,
) -> pd.DataFrame:
    """Produce a league-wide leaderboard of batters ranked by PSR.

    Args:
        conn: Open DuckDB connection.
        season: Season year filter.
        min_games: Minimum games to qualify.
        league_avg: League average wOBA.

    Returns:
        DataFrame with columns ``batter_id``, ``name``, ``psr``,
        ``mean_woba``, ``std_woba``, ``games``, ``rank``.
    """
    df = batch_player_sharpe(
        conn, team_id=None, season=season,
        min_games=min_games, league_avg=league_avg,
    )
    if df.empty:
        return df

    df = df.dropna(subset=["psr"])
    df = df.sort_values("psr", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# ── Model class (follows BaseAnalyticsModel pattern) ─────────────────────────


class SharpeLineupModel(BaseAnalyticsModel):
    """Markowitz portfolio model applied to baseball lineups.

    Wraps the functional API into the standard BaseAnalyticsModel lifecycle.
    """

    @property
    def model_name(self) -> str:
        return "sharpe_lineup"

    @property
    def version(self) -> str:
        return "1.0.0"

    def __init__(self) -> None:
        super().__init__()
        self._leaderboard: Optional[pd.DataFrame] = None
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._frontier: Optional[list[dict]] = None

    def train(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Compute the full Sharpe leaderboard and cache it.

        Keyword Args:
            season (int): Season year filter.
            min_games (int): Minimum games to qualify (default 40).
            team_id (str): Optional team filter.

        Returns:
            Training metrics including number of qualifying batters.
        """
        season = kwargs.get("season")
        min_games = kwargs.get("min_games", MIN_GAMES_DEFAULT)
        team_id = kwargs.get("team_id")

        self._leaderboard = batch_player_sharpe(
            conn, team_id=team_id, season=season, min_games=min_games,
        )

        metrics = {
            "qualifying_batters": len(self._leaderboard),
            "season": season,
            "min_games": min_games,
        }

        if not self._leaderboard.empty and "psr" in self._leaderboard.columns:
            valid_psr = self._leaderboard["psr"].dropna()
            if not valid_psr.empty:
                metrics["mean_psr"] = round(float(valid_psr.mean()), 3)
                metrics["max_psr"] = round(float(valid_psr.max()), 3)
                metrics["min_psr"] = round(float(valid_psr.min()), 3)

        self.set_training_metadata(metrics, params=kwargs)
        return metrics

    def predict(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Optimise a lineup from the cached player data.

        Keyword Args:
            player_ids (list[int]): Specific player IDs to consider.
            season (int): Season filter.
            risk_budget (float): Risk aversion parameter.

        Returns:
            Optimal lineup dictionary.
        """
        player_ids = kwargs.get("player_ids")
        season = kwargs.get("season")
        risk_budget = kwargs.get("risk_budget")

        if player_ids is None and self._leaderboard is not None:
            player_ids = self._leaderboard["batter_id"].tolist()

        if not player_ids:
            raise ValueError("No player IDs provided and no leaderboard cached.")

        stats = batch_player_sharpe(conn, season=season, min_games=1)
        stats = stats[stats["batter_id"].isin(player_ids)]

        if stats.empty:
            raise ValueError("No qualifying players found.")

        corr = compute_correlation_matrix(conn, stats["batter_id"].tolist(), season)

        result = optimize_lineup(stats, corr, risk_budget=risk_budget)
        return self.validate_output(result)

    def evaluate(self, conn: duckdb.DuckDBPyConnection, **kwargs) -> dict:
        """Evaluate by computing PSR distribution statistics.

        Returns:
            Dictionary of distribution metrics.
        """
        season = kwargs.get("season")
        lb = get_sharpe_leaderboard(conn, season=season)

        if lb.empty:
            return {"qualifying_batters": 0}

        valid = lb["psr"].dropna()
        return {
            "qualifying_batters": len(lb),
            "mean_psr": round(float(valid.mean()), 3) if not valid.empty else None,
            "median_psr": round(float(valid.median()), 3) if not valid.empty else None,
            "std_psr": round(float(valid.std()), 3) if not valid.empty else None,
            "pct_positive_psr": round(
                float((valid > 0).sum() / len(valid) * 100), 1
            ) if not valid.empty else None,
        }
