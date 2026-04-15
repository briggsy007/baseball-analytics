"""
Real-time MLB game feed ingestion via the MLB Stats API (GUMBO feed).

Polls the public statsapi.mlb.com endpoints for live game data, parses the
deeply-nested GUMBO JSON into clean domain objects, and provides a
LiveGameTracker class for event-driven consumption of pitch-level updates.

No authentication is required for the MLB Stats API.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import date, datetime
from typing import Any, Callable, Optional

import requests

logger = logging.getLogger(__name__)

# ── MLB Stats API endpoints ───────────────────────────────────────────────

BASE_URL = "https://statsapi.mlb.com"
SCHEDULE_URL = f"{BASE_URL}/api/v1/schedule"
LIVE_FEED_URL = f"{BASE_URL}/api/v1.1/game/{{game_pk}}/feed/live"
DIFF_PATCH_URL = (
    f"{BASE_URL}/api/v1.1/game/{{game_pk}}/feed/live/diffPatch"
    "?startTimecode={timecode}"
)

USER_AGENT = "BaseballAnalyticsPlatform/1.0 (contact: analytics@example.com)"
REQUEST_TIMEOUT = 10  # seconds

# Phillies abbreviation used for filtering
PHILLIES_ABBREV = "PHI"

# ── HTTP session ──────────────────────────────────────────────────────────

_session: Optional[requests.Session] = None
_session_lock = threading.Lock()


def _get_session() -> requests.Session:
    """Return a module-level requests.Session for connection pooling.

    Thread-safe: uses a lock to prevent duplicate Session creation
    when multiple threads call this concurrently.
    """
    global _session
    if _session is None:
        with _session_lock:
            # Double-check inside lock to avoid race condition
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


# ── Public functions: schedule ────────────────────────────────────────────


def get_todays_games(date_str: Optional[str] = None) -> list[dict]:
    """Fetch the MLB schedule for a given date.

    Args:
        date_str: Date in ``YYYY-MM-DD`` format.  Defaults to today (UTC).

    Returns:
        List of game dicts::

            [{"game_pk": 12345, "home": "PHI", "away": "NYM",
              "status": "Live", "start_time": "7:05 PM"}, ...]
    """
    if date_str is None:
        date_str = date.today().strftime("%Y-%m-%d")

    data = _api_get(
        SCHEDULE_URL,
        params={"sportId": 1, "date": date_str, "hydrate": "team"},
    )
    games: list[dict] = []

    for game_date in data.get("dates", []):
        for g in game_date.get("games", []):
            game_pk = g.get("gamePk")
            teams = g.get("teams", {})
            home = teams.get("home", {}).get("team", {}).get("abbreviation", "")
            away = teams.get("away", {}).get("team", {}).get("abbreviation", "")

            status_obj = g.get("status", {})
            abstract_state = status_obj.get("abstractGameState", "Preview")

            # Parse the scheduled start time (convert UTC to Eastern Time)
            game_dt_str = g.get("gameDate", "")
            start_time = ""
            if game_dt_str:
                try:
                    from zoneinfo import ZoneInfo
                    dt_utc = datetime.fromisoformat(game_dt_str.replace("Z", "+00:00"))
                    dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
                    # Use %I (zero-padded) and strip the leading zero for
                    # cross-platform compatibility (%-I is Linux-only).
                    start_time = dt_et.strftime("%I:%M %p ET").lstrip("0")
                except (ValueError, OSError, ImportError):
                    start_time = game_dt_str

            games.append({
                "game_pk": game_pk,
                "home": home,
                "away": away,
                "status": abstract_state,
                "start_time": start_time,
            })

    return games


def get_phillies_game(date_str: Optional[str] = None) -> Optional[dict]:
    """Return today's Phillies game, or ``None`` if they have no game.

    Args:
        date_str: Date in ``YYYY-MM-DD`` format.  Defaults to today.

    Returns:
        A game dict (same shape as ``get_todays_games`` entries) or ``None``.
    """
    games = get_todays_games(date_str)
    for g in games:
        if g.get("home") == PHILLIES_ABBREV or g.get("away") == PHILLIES_ABBREV:
            return g
    return None


# ── Public functions: live feed ───────────────────────────────────────────


def fetch_live_feed(game_pk: int) -> dict:
    """Fetch the full GUMBO live feed for a game.

    Args:
        game_pk: The MLB game primary key (integer).

    Returns:
        Raw GUMBO JSON dict.
    """
    url = LIVE_FEED_URL.format(game_pk=game_pk)
    return _api_get(url)


# ── Public functions: parsing ─────────────────────────────────────────────


def parse_game_state(feed: dict) -> dict:
    """Extract a clean game-state summary from a GUMBO feed.

    Args:
        feed: Raw GUMBO JSON dict from ``fetch_live_feed``.

    Returns:
        A normalised game state dict with keys: game_pk, status, inning,
        inning_half, outs, home_team, away_team, home_score, away_score,
        runners, current_batter, current_pitcher, count, innings,
        scoring_plays, weather, venue.
    """
    game_data = feed.get("gameData", {})
    live_data = feed.get("liveData", {})

    # Teams
    teams = game_data.get("teams", {})
    home_abbr = teams.get("home", {}).get("abbreviation", "")
    away_abbr = teams.get("away", {}).get("abbreviation", "")

    # Status
    status = game_data.get("status", {}).get("abstractGameState", "Preview")

    # Linescore
    linescore = live_data.get("linescore", {})
    inning = linescore.get("currentInning", 0)
    inning_half = linescore.get("inningHalf", "Top")
    outs = linescore.get("outs", 0)

    ls_teams = linescore.get("teams", {})
    home_score = ls_teams.get("home", {}).get("runs", 0)
    away_score = ls_teams.get("away", {}).get("runs", 0)

    # Innings breakdown
    innings_raw = linescore.get("innings", [])
    innings = []
    for inn in innings_raw:
        innings.append({
            "num": inn.get("num"),
            "home_runs": inn.get("home", {}).get("runs", 0),
            "away_runs": inn.get("away", {}).get("runs", 0),
        })

    # Runners (from linescore offense block)
    offense = linescore.get("offense", {})
    runners = {
        "first": "first" in offense,
        "second": "second" in offense,
        "third": "third" in offense,
    }

    # Current play
    plays = live_data.get("plays", {})
    current_play = plays.get("currentPlay", {})
    matchup = current_play.get("matchup", {})

    batter = matchup.get("batter", {})
    pitcher = matchup.get("pitcher", {})
    current_batter = {
        "id": batter.get("id", 0),
        "name": batter.get("fullName", ""),
        "side": matchup.get("batSide", {}).get("code", ""),
    }
    current_pitcher = {
        "id": pitcher.get("id", 0),
        "name": pitcher.get("fullName", ""),
        "hand": matchup.get("pitchHand", {}).get("code", ""),
    }

    # Count — from the last playEvent of the current play, or the play-level count
    count = {"balls": 0, "strikes": 0}
    play_events = current_play.get("playEvents", [])
    if play_events:
        last_event_count = play_events[-1].get("count", {})
        count["balls"] = last_event_count.get("balls", 0)
        count["strikes"] = last_event_count.get("strikes", 0)

    # Scoring plays
    scoring_indices = plays.get("scoringPlays", [])
    all_plays = plays.get("allPlays", [])
    scoring_descs: list[str] = []
    for idx in scoring_indices:
        if 0 <= idx < len(all_plays):
            desc = all_plays[idx].get("result", {}).get("description", "")
            if desc:
                scoring_descs.append(desc)

    # Weather & venue
    weather = game_data.get("weather", {})
    venue = game_data.get("venue", {}).get("name", "")

    # game_pk
    game_pk = game_data.get("game", {}).get("pk", feed.get("gamePk", 0))

    return {
        "game_pk": game_pk,
        "status": status,
        "inning": inning,
        "inning_half": inning_half,
        "outs": outs,
        "home_team": home_abbr,
        "away_team": away_abbr,
        "home_score": home_score,
        "away_score": away_score,
        "runners": runners,
        "current_batter": current_batter,
        "current_pitcher": current_pitcher,
        "count": count,
        "innings": innings,
        "scoring_plays": scoring_descs,
        "weather": weather,
        "venue": venue,
    }


def parse_pitch_events(feed: dict) -> list[dict]:
    """Extract every pitch event from the GUMBO feed.

    Args:
        feed: Raw GUMBO JSON dict.

    Returns:
        List of normalised pitch dicts with keys: pitcher_id, batter_id,
        inning, inning_topbot, pitch_type, pitch_name, start_speed, zone,
        px, pz, call_description, count_balls, count_strikes, outs,
        at_bat_number, pitch_number, event, is_current.
    """
    live_data = feed.get("liveData", {})
    plays = live_data.get("plays", {})
    all_plays = plays.get("allPlays", [])
    current_play = plays.get("currentPlay", {})
    current_ab_index = current_play.get("about", {}).get("atBatIndex", -1)

    pitches: list[dict] = []

    for play in all_plays:
        about = play.get("about", {})
        matchup = play.get("matchup", {})
        result = play.get("result", {})

        pitcher_id = matchup.get("pitcher", {}).get("id", 0)
        pitcher_name = matchup.get("pitcher", {}).get("fullName", "")
        batter_id = matchup.get("batter", {}).get("id", 0)
        inning = about.get("inning", 0)
        inning_topbot = "Top" if about.get("halfInning", "top") == "top" else "Bot"
        at_bat_index = about.get("atBatIndex", -1)
        is_complete = about.get("isComplete", False)
        ab_event = result.get("event", "") if is_complete else ""

        play_events = play.get("playEvents", [])
        pitch_events = [pe for pe in play_events if pe.get("isPitch", False)]

        for i, pe in enumerate(pitch_events):
            details = pe.get("details", {})
            pitch_data = pe.get("pitchData", {})
            count_obj = pe.get("count", {})
            coords = pitch_data.get("coordinates", {})
            call_desc = details.get("call", {}).get("description", "")
            pitch_type_info = details.get("type", {})

            # Event: attach only to the *last* pitch of a completed at-bat
            event = ""
            if is_complete and i == len(pitch_events) - 1:
                event = ab_event

            pitches.append({
                "pitcher_id": pitcher_id,
                "pitcher_name": pitcher_name,
                "batter_id": batter_id,
                "inning": inning,
                "inning_topbot": inning_topbot,
                "pitch_type": pitch_type_info.get("code", ""),
                "pitch_name": pitch_type_info.get("description", ""),
                "start_speed": pitch_data.get("startSpeed"),
                "zone": pitch_data.get("zone"),
                "px": coords.get("pX"),
                "pz": coords.get("pZ"),
                "call_description": call_desc,
                "count_balls": count_obj.get("balls", 0),
                "count_strikes": count_obj.get("strikes", 0),
                "outs": count_obj.get("outs", 0),
                "at_bat_number": at_bat_index,
                "pitch_number": pe.get("index", i),
                "event": event,
                "is_current": at_bat_index == current_ab_index,
            })

    return pitches


def parse_lineup(feed: dict, team: str = "home") -> list[dict]:
    """Extract the batting order for a team from the boxscore.

    Args:
        feed: Raw GUMBO JSON dict.
        team: ``"home"`` or ``"away"``.

    Returns:
        Ordered list of lineup dicts::

            [{"order": 1, "id": 123, "name": "...", "position": "SS",
              "bats": "R"}, ...]
    """
    live_data = feed.get("liveData", {})
    boxscore = live_data.get("boxscore", {})
    team_data = boxscore.get("teams", {}).get(team, {})
    players_map = team_data.get("players", {})
    batting_order = team_data.get("battingOrder", [])

    game_players = feed.get("gameData", {}).get("players", {})

    lineup: list[dict] = []
    for order_num, player_id in enumerate(batting_order, start=1):
        # The boxscore keys are like "ID123456"
        player_key = f"ID{player_id}"
        box_player = players_map.get(player_key, {})

        # Get position from boxscore
        position = box_player.get("position", {}).get("abbreviation", "")

        # Get name and bats from gameData players
        gd_player = game_players.get(player_key, {})
        name = gd_player.get("fullName", box_player.get("person", {}).get("fullName", ""))
        bats = gd_player.get("batSide", {}).get("code", "")

        lineup.append({
            "order": order_num,
            "id": player_id,
            "name": name,
            "position": position,
            "bats": bats,
        })

    return lineup


def parse_bullpen(feed: dict, team: str = "home") -> list[dict]:
    """Extract available bullpen pitchers for a team.

    Bullpen = pitchers on the roster who are NOT the starting pitcher and
    have not yet appeared in the game, plus relievers who appeared.

    Args:
        feed: Raw GUMBO JSON dict.
        team: ``"home"`` or ``"away"``.

    Returns:
        List of bullpen dicts::

            [{"id": 456, "name": "...", "throws": "R", "era": 3.21}, ...]
    """
    live_data = feed.get("liveData", {})
    boxscore = live_data.get("boxscore", {})
    team_data = boxscore.get("teams", {}).get(team, {})
    players_map = team_data.get("players", {})

    game_players = feed.get("gameData", {}).get("players", {})

    # Identify who is in the batting order (starters, not pitchers unless DH rule)
    batting_order_ids = set(team_data.get("battingOrder", []))

    # Identify bullpen IDs listed by the boxscore
    bullpen_ids = team_data.get("bullpen", [])
    pitchers_ids = team_data.get("pitchers", [])

    # Combine bullpen + pitchers minus anyone already starting
    # The "bullpen" key typically lists relievers; "pitchers" lists all who pitched
    candidate_ids = set(bullpen_ids) | set(pitchers_ids)

    bullpen: list[dict] = []
    for pid in candidate_ids:
        player_key = f"ID{pid}"
        box_player = players_map.get(player_key, {})
        gd_player = game_players.get(player_key, {})

        name = gd_player.get("fullName", box_player.get("person", {}).get("fullName", ""))
        throws = gd_player.get("pitchHand", {}).get("code", "")

        # Try to get ERA from the boxscore season stats
        stats = box_player.get("seasonStats", {}).get("pitching", {})
        era_str = stats.get("era", "0.00")
        try:
            era = float(era_str)
        except (ValueError, TypeError):
            era = 0.0

        bullpen.append({
            "id": pid,
            "name": name,
            "throws": throws,
            "era": era,
        })

    return bullpen


# ── LiveGameTracker class ─────────────────────────────────────────────────


class LiveGameTracker:
    """Event-driven poller for live MLB game data.

    Periodically fetches the GUMBO feed, diffs against the previous state,
    and fires registered callbacks when new pitches, completed at-bats, or
    pitching changes are detected.

    Usage::

        tracker = LiveGameTracker(game_pk=717421, poll_interval=10)
        tracker.on_new_pitch(lambda pitch: print(pitch))
        tracker.on_play_complete(lambda play: print(play))
        tracker.run()  # blocks until game ends

    Args:
        game_pk: MLB game primary key.
        poll_interval: Seconds between polls (default 10).
    """

    def __init__(self, game_pk: int, poll_interval: int = 10) -> None:
        self.game_pk: int = game_pk
        self.poll_interval: int = poll_interval
        self.last_feed: Optional[dict] = None
        self.pitch_count: int = 0
        self.callbacks: dict[str, list[Callable]] = {
            "new_pitch": [],
            "play_complete": [],
            "pitching_change": [],
        }
        self._last_play_count: int = 0
        self._last_pitcher_ids: dict[str, int] = {"home": 0, "away": 0}
        self._lock = threading.Lock()

    # ── Callback registration ─────────────────────────────────────────

    def on_new_pitch(self, callback: Callable[[dict], Any]) -> None:
        """Register a callback invoked for each new pitch event.

        The callback receives a single pitch dict (same shape as
        ``parse_pitch_events`` entries).
        """
        self.callbacks["new_pitch"].append(callback)

    def on_play_complete(self, callback: Callable[[dict], Any]) -> None:
        """Register a callback invoked when an at-bat completes.

        The callback receives the play dict from the GUMBO ``allPlays``
        array.
        """
        self.callbacks["play_complete"].append(callback)

    def on_pitching_change(self, callback: Callable[[dict], Any]) -> None:
        """Register a callback invoked on a pitching change.

        The callback receives a dict with keys ``team`` (``"home"`` or
        ``"away"``), ``old_pitcher_id``, ``new_pitcher_id``,
        ``new_pitcher_name``.
        """
        self.callbacks["pitching_change"].append(callback)

    # ── Polling ───────────────────────────────────────────────────────

    def poll_once(self) -> dict:
        """Fetch the live feed, diff against previous state, fire callbacks.

        Thread-safe: holds a lock while reading/writing shared state so
        that concurrent ``on_*`` registrations and external reads of
        ``pitch_count`` / ``last_feed`` are safe.

        Returns:
            The current game state dict (from ``parse_game_state``).
        """
        feed = fetch_live_feed(self.game_pk)
        all_pitches = parse_pitch_events(feed)
        current_total_pitches = len(all_pitches)

        plays = feed.get("liveData", {}).get("plays", {})
        all_plays = plays.get("allPlays", [])
        current_play_count = len(all_plays)

        with self._lock:
            # --- Detect new pitches ---
            if self.last_feed is not None and current_total_pitches > self.pitch_count:
                new_pitches = all_pitches[self.pitch_count:]
                for pitch in new_pitches:
                    self._fire("new_pitch", pitch)

            # --- Detect completed at-bats ---
            if self.last_feed is not None and current_play_count > self._last_play_count:
                for idx in range(self._last_play_count, current_play_count):
                    play = all_plays[idx]
                    if play.get("about", {}).get("isComplete", False):
                        self._fire("play_complete", play)

            # --- Detect pitching changes ---
            self._check_pitching_change(feed)

            # Update state
            self.pitch_count = current_total_pitches
            self._last_play_count = current_play_count
            self.last_feed = feed

        return parse_game_state(feed)

    def run(self, stop_event: Optional[threading.Event] = None) -> None:
        """Poll in a loop until the game reaches ``Final`` or *stop_event* is set.

        This is a blocking call.  To run in the background, wrap in a thread::

            stop = threading.Event()
            t = threading.Thread(target=tracker.run, args=(stop,))
            t.start()
            # ... later ...
            stop.set()

        Args:
            stop_event: Optional threading.Event; if set externally the loop
                exits after the current poll cycle.
        """
        logger.info("Starting live tracking for game_pk=%d (interval=%ds)",
                     self.game_pk, self.poll_interval)

        while True:
            if stop_event is not None and stop_event.is_set():
                logger.info("Stop event received, exiting tracker loop.")
                break

            try:
                state = self.poll_once()
            except Exception:
                logger.exception("Error polling game_pk=%d", self.game_pk)
                # Still sleep and retry on next cycle
                time.sleep(self.poll_interval)
                continue

            if state.get("status") == "Final":
                logger.info("Game %d is Final. Stopping tracker.", self.game_pk)
                break

            time.sleep(self.poll_interval)

    # ── Internal helpers ──────────────────────────────────────────────

    def _fire(self, event_type: str, data: Any) -> None:
        """Invoke all callbacks registered for *event_type*."""
        for cb in self.callbacks.get(event_type, []):
            try:
                cb(data)
            except Exception:
                logger.exception("Callback error for %s", event_type)

    def _check_pitching_change(self, feed: dict) -> None:
        """Compare current pitcher to last-seen pitcher for each team side."""
        if self.last_feed is None:
            # First poll — just record the current pitchers, no event to fire
            self._record_current_pitchers(feed)
            return

        live = feed.get("liveData", {})
        plays = live.get("plays", {})
        current_play = plays.get("currentPlay", {})
        about = current_play.get("about", {})
        matchup = current_play.get("matchup", {})

        half = about.get("halfInning", "top")
        # When it's the top of the inning the *away* team bats, so the
        # *home* team is pitching, and vice versa.
        pitching_side = "home" if half == "top" else "away"

        new_pitcher_id = matchup.get("pitcher", {}).get("id", 0)
        old_pitcher_id = self._last_pitcher_ids.get(pitching_side, 0)

        if old_pitcher_id != 0 and new_pitcher_id != 0 and old_pitcher_id != new_pitcher_id:
            new_pitcher_name = matchup.get("pitcher", {}).get("fullName", "")
            self._fire("pitching_change", {
                "team": pitching_side,
                "old_pitcher_id": old_pitcher_id,
                "new_pitcher_id": new_pitcher_id,
                "new_pitcher_name": new_pitcher_name,
            })

        # Always update the record
        self._record_current_pitchers(feed)

    def _record_current_pitchers(self, feed: dict) -> None:
        """Snapshot the current pitcher for the side that is pitching."""
        current_play = (
            feed.get("liveData", {})
            .get("plays", {})
            .get("currentPlay", {})
        )
        about = current_play.get("about", {})
        matchup = current_play.get("matchup", {})
        half = about.get("halfInning", "top")
        pitching_side = "home" if half == "top" else "away"
        pitcher_id = matchup.get("pitcher", {}).get("id", 0)
        if pitcher_id:
            self._last_pitcher_ids[pitching_side] = pitcher_id
