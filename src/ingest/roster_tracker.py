"""
Roster tracking and transaction ingestion for the baseball analytics platform.

Fetches active / 40-man roster data and recent transactions from the MLB
Stats API, synchronises them into the DuckDB database, and provides a
structured view of the Phillies pitching staff.

No authentication is required for the MLB Stats API.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

# ── MLB Stats API endpoints ───────────────────────────────────────────────

BASE_URL = "https://statsapi.mlb.com"
ROSTER_URL = f"{BASE_URL}/api/v1/teams/{{team_id}}/roster"
TRANSACTIONS_URL = f"{BASE_URL}/api/v1/transactions"
PERSON_URL = f"{BASE_URL}/api/v1/people/{{player_id}}"
TEAM_STATS_URL = f"{BASE_URL}/api/v1/teams/{{team_id}}/stats"

USER_AGENT = "BaseballAnalyticsPlatform/1.0 (contact: analytics@example.com)"
REQUEST_TIMEOUT = 10  # seconds

# Phillies constants
PHILLIES_TEAM_ID = 143
PHILLIES_ABBREV = "PHI"

# ── HTTP session ──────────────────────────────────────────────────────────

_session: Optional[requests.Session] = None
_session_lock = threading.Lock()


def _get_session() -> requests.Session:
    """Return a module-level requests.Session for connection pooling.

    Thread-safe: uses a lock to prevent duplicate Session creation.
    """
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = requests.Session()
                _session.headers.update({"User-Agent": USER_AGENT})
    return _session


def _api_get(url: str, params: Optional[dict] = None, retries: int = 3) -> dict:
    """Execute a GET request with exponential backoff on transient failures.

    Args:
        url: Fully-qualified URL to fetch.
        params: Optional query-string parameters.
        retries: Maximum retry attempts for 5xx / connection errors.

    Returns:
        Parsed JSON as a dict.

    Raises:
        requests.HTTPError: On non-retryable HTTP errors (4xx).
        requests.ConnectionError: After all retries exhausted.
    """
    session = _get_session()
    last_exc: Optional[Exception] = None

    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "Transient error on %s (attempt %d/%d), retrying in %ds: %s",
                url, attempt + 1, retries, wait, exc,
            )
            time.sleep(wait)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code >= 500:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "Server error %s on %s (attempt %d/%d), retrying in %ds",
                    exc.response.status_code, url, attempt + 1, retries, wait,
                )
                time.sleep(wait)
            else:
                raise

    raise requests.ConnectionError(
        f"Failed after {retries} attempts on {url}"
    ) from last_exc


# ── Roster functions ──────────────────────────────────────────────────────


def get_active_roster(team_id: int = PHILLIES_TEAM_ID) -> list[dict]:
    """Fetch the current active (26-man) roster for a team.

    Uses ``hydrate=person(stats(...))`` to pull bats/throws and current-season
    stats in a single API call.

    Args:
        team_id: MLB team ID (default: 143, Phillies).

    Returns:
        List of player dicts::

            [{"id": 123, "name": "...", "position": "SP", "throws": "R",
              "bats": "L", "jersey": "27", "status": "Active",
              "season_stats": {...}}, ...]

        The ``position`` field for pitchers is refined to ``"SP"`` or ``"RP"``
        based on season games-started ratio when stats are available.
    """
    url = ROSTER_URL.format(team_id=team_id)
    current_year = date.today().year
    hydrate = f"person(stats(type=season,season={current_year}))"
    data = _api_get(url, params={"rosterType": "active", "hydrate": hydrate})
    return _parse_roster_response(data)


def get_40_man_roster(team_id: int = PHILLIES_TEAM_ID) -> list[dict]:
    """Fetch the 40-man roster for a team (superset of active).

    Args:
        team_id: MLB team ID (default: 143, Phillies).

    Returns:
        List of player dicts (same shape as ``get_active_roster``).
    """
    url = ROSTER_URL.format(team_id=team_id)
    current_year = date.today().year
    hydrate = f"person(stats(type=season,season={current_year}))"
    data = _api_get(url, params={"rosterType": "40Man", "hydrate": hydrate})
    return _parse_roster_response(data)


def _parse_roster_response(data: dict) -> list[dict]:
    """Normalise the MLB Stats API roster response into clean dicts.

    Extracts player bio (bats/throws) from hydrated ``person`` and
    season pitching stats (if present) for downstream classification.

    Args:
        data: Raw JSON from the roster endpoint.

    Returns:
        List of player dicts.  Each dict includes a ``season_stats`` sub-dict
        for pitchers if the data was available (keys: era, g, gs, sv, ip).
    """
    roster: list[dict] = []
    for entry in data.get("roster", []):
        person = entry.get("person", {})
        position = entry.get("position", {})
        status = entry.get("status", {})

        pos_abbrev = position.get("abbreviation", "")
        throws = person.get("pitchHand", {}).get("code", "")
        bats = person.get("batSide", {}).get("code", "")

        # Extract season pitching stats from hydrated person data
        season_stats: dict[str, Any] = {}
        for stat_group in person.get("stats", []):
            splits = stat_group.get("splits", [])
            if splits:
                stat = splits[0].get("stat", {})
                # Only capture pitching stats (they have 'era')
                if "era" in stat:
                    era_str = stat.get("era", "0.00")
                    try:
                        era_val = float(era_str)
                    except (ValueError, TypeError):
                        era_val = 0.0
                    season_stats = {
                        "era": era_val,
                        "g": stat.get("gamesPlayed", 0),
                        "gs": stat.get("gamesStarted", 0),
                        "sv": stat.get("saves", 0),
                        "ip": stat.get("inningsPitched", "0"),
                        "w": stat.get("wins", 0),
                        "l": stat.get("losses", 0),
                    }

        # Refine generic "P" to "SP" or "RP" based on season stats
        if pos_abbrev == "P" and season_stats:
            gs = season_stats.get("gs", 0) or 0
            g = season_stats.get("g", 0) or 0
            if gs > 0 and (g == 0 or gs / g > 0.5):
                pos_abbrev = "SP"
            elif g > 0:
                pos_abbrev = "RP"

        roster.append({
            "id": person.get("id", 0),
            "name": person.get("fullName", ""),
            "position": pos_abbrev,
            "throws": throws,
            "bats": bats,
            "jersey": entry.get("jerseyNumber", ""),
            "status": status.get("description", "Active"),
            "season_stats": season_stats,
        })

    return roster


# ── Transaction functions ─────────────────────────────────────────────────


def get_recent_transactions(
    team_id: int = PHILLIES_TEAM_ID,
    days: int = 7,
) -> list[dict]:
    """Fetch recent transactions from the MLB Stats API.

    Args:
        team_id: MLB team ID to filter on.
        days: Number of days to look back.

    Returns:
        List of transaction dicts::

            [{"date": "2026-04-06", "player": "...",
              "type": "Placed on IL", "description": "...",
              "player_id": 123, "transaction_id": 456}, ...]
    """
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    data = _api_get(
        TRANSACTIONS_URL,
        params={
            "startDate": start_date.strftime("%m/%d/%Y"),
            "endDate": end_date.strftime("%m/%d/%Y"),
        },
    )

    transactions: list[dict] = []
    for txn in data.get("transactions", []):
        # Filter to the requested team
        txn_team_id = txn.get("team", {}).get("id", 0)
        # Also check toTeam / fromTeam for trades
        to_team_id = txn.get("toTeam", {}).get("id", 0)
        from_team_id = txn.get("fromTeam", {}).get("id", 0)

        if team_id not in (txn_team_id, to_team_id, from_team_id):
            continue

        player = txn.get("person", {})
        txn_date_str = txn.get("date", "")
        txn_date = ""
        if txn_date_str:
            try:
                txn_date = datetime.fromisoformat(
                    txn_date_str.replace("Z", "+00:00")
                ).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                txn_date = txn_date_str

        transactions.append({
            "transaction_id": txn.get("id", 0),
            "date": txn_date,
            "player": player.get("fullName", ""),
            "player_id": player.get("id", 0),
            "type": txn.get("typeDesc", ""),
            "description": txn.get("description", ""),
            "from_team": txn.get("fromTeam", {}).get("name", ""),
            "to_team": txn.get("toTeam", {}).get("name", ""),
        })

    return transactions


# ── Database sync functions ───────────────────────────────────────────────


def sync_roster_to_db(conn: Any, team_id: int = PHILLIES_TEAM_ID) -> int:
    """Fetch the active roster and upsert into the ``players`` table.

    For each player: if the ``player_id`` already exists, update team and
    position; otherwise insert a new row.  Wrapped in a transaction.

    Args:
        conn: An open DuckDB connection.
        team_id: MLB team ID.

    Returns:
        Number of players processed.
    """
    roster = get_active_roster(team_id)

    # Resolve team abbreviation for the DB
    team_abbrev = _team_id_to_abbrev(team_id)

    upserted = 0
    conn.execute("BEGIN TRANSACTION")
    try:
        for player in roster:
            pid = player["id"]
            # Check if player exists
            existing = conn.execute(
                "SELECT player_id FROM players WHERE player_id = ?", [pid]
            ).fetchone()

            if existing:
                conn.execute(
                    """
                    UPDATE players
                    SET team = ?, position = ?, throws = ?, bats = ?, full_name = ?
                    WHERE player_id = ?
                    """,
                    [team_abbrev, player["position"], player["throws"],
                     player["bats"], player["name"], pid],
                )
            else:
                conn.execute(
                    """
                    INSERT INTO players (player_id, full_name, team, position,
                                         throws, bats, mlbam_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [pid, player["name"], team_abbrev, player["position"],
                     player["throws"], player["bats"], pid],
                )
            upserted += 1

        conn.execute("COMMIT")
        logger.info("Synced %d players for team_id=%d", upserted, team_id)
    except Exception:
        logger.exception("Roster sync failed — rolling back")
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise

    return upserted


def sync_transactions_to_db(conn: Any, days: int = 7) -> int:
    """Fetch recent Phillies transactions and insert new ones into the DB.

    Skips rows whose ``transaction_id`` already exists in the table to avoid
    duplicates.  Wrapped in a transaction for atomicity.

    Args:
        conn: An open DuckDB connection.
        days: Number of days to look back.

    Returns:
        Number of new transactions inserted.
    """
    transactions = get_recent_transactions(team_id=PHILLIES_TEAM_ID, days=days)
    inserted = 0

    conn.execute("BEGIN TRANSACTION")
    try:
        for txn in transactions:
            tid = txn["transaction_id"]
            if tid == 0:
                # Skip transactions with no ID — can't deduplicate
                continue

            existing = conn.execute(
                "SELECT transaction_id FROM transactions WHERE transaction_id = ?",
                [tid],
            ).fetchone()

            if existing:
                continue

            # Parse date string into a date object for DuckDB
            txn_date = None
            if txn["date"]:
                try:
                    txn_date = datetime.strptime(txn["date"], "%Y-%m-%d").date()
                except (ValueError, TypeError):
                    txn_date = None

            conn.execute(
                """
                INSERT INTO transactions (
                    transaction_id, player_id, player_name, team,
                    from_team, to_team, transaction_type, description,
                    transaction_date
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    tid,
                    txn["player_id"],
                    txn["player"],
                    PHILLIES_ABBREV,
                    txn.get("from_team", ""),
                    txn.get("to_team", ""),
                    txn["type"],
                    txn["description"],
                    txn_date,
                ],
            )
            inserted += 1

        conn.execute("COMMIT")
        logger.info("Inserted %d new transactions (looked back %d days)", inserted, days)
    except Exception:
        logger.exception("Transaction sync failed — rolling back")
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        raise

    return inserted


# ── Pitching staff view ───────────────────────────────────────────────────


def get_phillies_pitching_staff(conn: Any = None) -> dict:
    """Return a structured view of the Phillies pitching staff.

    Combines roster data from the API with season pitching stats from the
    database (if a connection is provided) to build rotation, bullpen, and
    injured-list views.

    Args:
        conn: Optional open DuckDB connection.  If provided, ERA and role
              information is enriched from the ``season_pitching_stats``
              table.

    Returns:
        Dict with keys ``rotation``, ``bullpen``, ``injured_list``::

            {
                "rotation": [{"id": ..., "name": ..., "throws": ...,
                              "era": ...}, ...],
                "bullpen":  [{"id": ..., "name": ..., "throws": ...,
                              "era": ..., "role": "Closer"}, ...],
                "injured_list": [{"id": ..., "name": ..., "injury": ...,
                                  "return_date": ...}, ...]
            }
    """
    # Get both active and 40-man rosters to catch IL players.
    # These calls already hydrate season stats from the API.
    active_roster = get_active_roster(PHILLIES_TEAM_ID)
    forty_man = get_40_man_roster(PHILLIES_TEAM_ID)

    # Build a stats lookup — prefer the DB (richer data) if available,
    # fall back to the API-hydrated season_stats from the roster response.
    stats_lookup: dict[int, dict] = {}

    # Seed from roster API stats (available for everyone)
    for p in active_roster + forty_man:
        if p.get("season_stats"):
            stats_lookup.setdefault(p["id"], p["season_stats"])

    # Overlay with DB stats if a connection is provided
    if conn is not None:
        try:
            current_year = date.today().year
            rows = conn.execute(
                """
                SELECT player_id, era, gs, sv, g, ip, w, l, k_per_9, bb_per_9
                FROM season_pitching_stats
                WHERE season = ?
                """,
                [current_year],
            ).fetchall()
            for row in rows:
                stats_lookup[row[0]] = {
                    "era": row[1],
                    "gs": row[2],
                    "sv": row[3],
                    "g": row[4],
                    "ip": row[5],
                    "w": row[6],
                    "l": row[7],
                    "k_per_9": row[8],
                    "bb_per_9": row[9],
                }
        except Exception:
            logger.warning("Could not fetch season pitching stats from DB", exc_info=True)

    # Active pitchers (on the 26-man) — position already refined to SP/RP
    active_pitchers = [
        p for p in active_roster if p["position"] in ("SP", "RP", "P")
    ]

    # 40-man pitchers NOT on the active roster — likely IL
    active_ids = {p["id"] for p in active_roster}
    il_candidates = [
        p for p in forty_man
        if p["id"] not in active_ids
        and p["position"] in ("SP", "RP", "P")
    ]

    rotation: list[dict] = []
    bullpen: list[dict] = []
    injured_list: list[dict] = []

    for p in active_pitchers:
        pid = p["id"]
        stats = stats_lookup.get(pid, {})
        era = stats.get("era", 0.0)
        gs = stats.get("gs", 0) or 0
        sv = stats.get("sv", 0) or 0
        g = stats.get("g", 0) or 0

        entry = {
            "id": pid,
            "name": p["name"],
            "throws": p["throws"],
            "era": era if era is not None else 0.0,
        }

        # Use the refined position from roster parsing
        if p["position"] == "SP":
            rotation.append(entry)
        elif p["position"] == "RP":
            role = _classify_bullpen_role(sv, g, stats)
            entry["role"] = role
            bullpen.append(entry)
        else:
            # Still generic "P" (no stats available) — classify by stats or
            # default to bullpen
            if gs > 0 and (g == 0 or gs / g > 0.5):
                rotation.append(entry)
            else:
                role = _classify_bullpen_role(sv, g, stats)
                entry["role"] = role
                bullpen.append(entry)

    for p in il_candidates:
        injured_list.append({
            "id": p["id"],
            "name": p["name"],
            "injury": p.get("status", "Unknown"),
            "return_date": None,
        })

    return {
        "rotation": rotation,
        "bullpen": bullpen,
        "injured_list": injured_list,
    }


# ── Internal helpers ──────────────────────────────────────────────────────


def _classify_bullpen_role(
    sv: Optional[int],
    g: Optional[int],
    stats: dict,
) -> str:
    """Heuristic to classify a reliever's role.

    Args:
        sv: Saves this season.
        g: Games appeared.
        stats: Full stats dict from the DB.

    Returns:
        ``"Closer"``, ``"Setup"``, or ``"Middle"``.
    """
    sv = sv or 0
    g = g or 0

    if sv >= 5 or (sv >= 2 and g > 0 and sv / g > 0.3):
        return "Closer"

    # Setup men tend to have many games and few saves
    if g >= 15:
        return "Setup"

    return "Middle"


_TEAM_ABBREV_CACHE: dict[int, str] = {
    143: "PHI",
}


def _team_id_to_abbrev(team_id: int) -> str:
    """Map a team ID to its abbreviation, caching results.

    Falls back to an API call if the team is not already cached.

    Args:
        team_id: MLB team ID.

    Returns:
        Team abbreviation string (e.g. ``"PHI"``).
    """
    if team_id in _TEAM_ABBREV_CACHE:
        return _TEAM_ABBREV_CACHE[team_id]

    try:
        url = f"{BASE_URL}/api/v1/teams/{team_id}"
        data = _api_get(url)
        teams = data.get("teams", [])
        if teams:
            abbrev = teams[0].get("abbreviation", str(team_id))
            _TEAM_ABBREV_CACHE[team_id] = abbrev
            return abbrev
    except Exception:
        logger.warning("Could not resolve abbreviation for team_id=%d", team_id)

    return str(team_id)
