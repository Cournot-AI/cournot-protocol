"""
FotMob match stat extractor.

Extracts structured match statistics from fotmob.com match pages by
parsing the __NEXT_DATA__ JSON embedded in the HTML (Next.js SSR data).

This avoids the Cloudflare Turnstile that blocks the /api/matchDetails
endpoint, and captures ALL stats including those behind the "All stats"
accordion that JS-rendering tools (Gemini UrlContext, Jina) cannot see.

Usage:
    from agents.collector.fotmob import fetch_match_stats, find_stat, summarize_for_llm

    data = fetch_match_stats(url, http_client)
    stat = find_stat(data, "Shots outside box")
    if stat:
        print(f"{data.away_team}: {stat.away_value}")
    summary = summarize_for_llm(data)  # compact text for LLM consumption
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class FotMobExtractionError(Exception):
    """Raised when stat extraction from a FotMob page fails."""


@dataclass
class FotMobStat:
    """A single match statistic with values for both teams."""
    title: str
    key: str
    home_value: Any
    away_value: Any
    category: str


@dataclass
class FotMobShot:
    """A single shot from the shotmap."""
    event_type: str          # e.g. "Goal", "AttemptSaved", "Miss"
    team_id: int | None
    player_name: str
    situation: str           # e.g. "FromCorner", "OpenPlay", "SetPiece"
    shot_type: str           # e.g. "RightFoot", "LeftFoot", "Head"
    minute: int | None
    expected_goals: float | None
    is_home: bool
    on_target: bool


@dataclass
class FotMobEvent:
    """A single match event (goal, card, VAR, substitution, etc.)."""
    type: str                # e.g. "Goal", "Card", "Substitution", "AddedTime"
    time: int | None
    overload_time: int | None
    is_home: bool
    player_name: str
    player_id: int | None = None
    card_type: str | None = None        # "Yellow", "Red", "SecondYellow"
    is_own_goal: bool = False
    assist_player: str | None = None
    var_decision: str | None = None     # e.g. "Goal confirmed", "Penalty cancelled"
    raw: dict = field(default_factory=dict)


@dataclass
class FotMobMatchData:
    """Parsed match data from a FotMob page."""
    match_id: str
    home_team: str
    away_team: str
    home_team_id: int | None = None
    away_team_id: int | None = None
    url: str = ""
    stats_by_category: dict[str, list[FotMobStat]] = field(default_factory=dict)
    shots: list[FotMobShot] = field(default_factory=list)
    events: list[FotMobEvent] = field(default_factory=list)
    event_types: list[str] = field(default_factory=list)
    home_score: int | None = None
    away_score: int | None = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEXT_DATA_RE = re.compile(
    r'<script\s+id="__NEXT_DATA__"\s+type="application/json">(.*?)</script>',
    re.DOTALL,
)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_match_stats(url: str, http_client: Any) -> FotMobMatchData:
    """Fetch a FotMob match page and extract all stats.

    Args:
        url: Full FotMob match URL (e.g.
             ``https://www.fotmob.com/matches/osasuna-vs-real-madrid/2e2ylz``).
        http_client: An httpx-compatible client with a ``.get()`` method.

    Returns:
        Parsed ``FotMobMatchData`` with all stat categories.

    Raises:
        FotMobExtractionError: On HTTP errors or unexpected page structure.
    """
    resp = http_client.get(url, headers=_BROWSER_HEADERS, timeout=30.0)
    if resp.status_code != 200:
        raise FotMobExtractionError(
            f"HTTP {resp.status_code} fetching {url}"
        )

    html = resp.text
    m = _NEXT_DATA_RE.search(html)
    if not m:
        raise FotMobExtractionError(
            f"No __NEXT_DATA__ script tag found in {url}"
        )

    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        raise FotMobExtractionError(f"Invalid JSON in __NEXT_DATA__: {e}") from e

    return _parse_next_data(data, url)


def find_stat(
    match_data: FotMobMatchData,
    stat_title: str,
) -> FotMobStat | None:
    """Find a stat by title across all categories.

    Searches by case-insensitive title match first, then by normalized key.
    Skips ``type: "title"`` header entries.

    Args:
        match_data: Parsed match data from ``fetch_match_stats``.
        stat_title: The stat to find (e.g. ``"Shots outside box"``).

    Returns:
        The matching ``FotMobStat``, or ``None`` if not found.
    """
    title_lower = stat_title.lower().strip()
    key_normalized = title_lower.replace(" ", "_")

    for _cat_key, stats in match_data.stats_by_category.items():
        for stat in stats:
            if stat.title.lower().strip() == title_lower:
                return stat
            if stat.key.lower() == key_normalized:
                return stat
    return None


def match_team(
    match_data: FotMobMatchData,
    target_entity: str,
) -> str | None:
    """Determine whether target_entity is the home or away team.

    Returns ``"home"``, ``"away"``, or ``None`` if no match.
    """
    entity_lower = target_entity.lower().strip()
    home_lower = match_data.home_team.lower().strip()
    away_lower = match_data.away_team.lower().strip()

    # Exact match
    if entity_lower == home_lower:
        return "home"
    if entity_lower == away_lower:
        return "away"

    # Substring match
    if entity_lower in home_lower or home_lower in entity_lower:
        return "home"
    if entity_lower in away_lower or away_lower in entity_lower:
        return "away"

    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_next_data(data: dict, url: str) -> FotMobMatchData:
    """Parse the __NEXT_DATA__ JSON into FotMobMatchData."""
    try:
        page_props = data["props"]["pageProps"]
        general = page_props["general"]
    except (KeyError, TypeError) as e:
        raise FotMobExtractionError(
            f"Unexpected __NEXT_DATA__ structure — missing general path: {e}"
        ) from e

    match_id = str(general.get("matchId", ""))
    home = general.get("homeTeam", {})
    away = general.get("awayTeam", {})

    # --- Stats (resilient — may be absent) ---
    stats_by_category: dict[str, list[FotMobStat]] = {}
    try:
        stats_periods = page_props["content"]["stats"]["Periods"]["All"]["stats"]
        for category in stats_periods:
            cat_key = category.get("key", "unknown")
            cat_stats: list[FotMobStat] = []
            for stat in category.get("stats", []):
                if stat.get("type") == "title":
                    continue
                values = stat.get("stats", [])
                if not isinstance(values, list) or len(values) != 2:
                    continue
                cat_stats.append(FotMobStat(
                    title=stat.get("title", ""),
                    key=stat.get("key", ""),
                    home_value=values[0],
                    away_value=values[1],
                    category=cat_key,
                ))
            stats_by_category[cat_key] = cat_stats
    except (KeyError, TypeError):
        pass  # stats not available

    # --- Shotmap ---
    shots = _parse_shotmap(page_props)

    # --- Events + score ---
    events, event_types, ev_home_score, ev_away_score = _parse_events(page_props)

    # --- Score fallback from header ---
    home_score, away_score = _parse_score(page_props)
    if home_score is None:
        home_score = ev_home_score
    if away_score is None:
        away_score = ev_away_score

    # Raise only if ALL data sources are empty
    if not stats_by_category and not shots and not events:
        raise FotMobExtractionError(
            "No stats, shots, or events found in __NEXT_DATA__"
        )

    return FotMobMatchData(
        match_id=match_id,
        home_team=home.get("name", ""),
        away_team=away.get("name", ""),
        home_team_id=home.get("id"),
        away_team_id=away.get("id"),
        url=url,
        stats_by_category=stats_by_category,
        shots=shots,
        events=events,
        event_types=event_types,
        home_score=home_score,
        away_score=away_score,
    )


def _parse_shotmap(page_props: dict) -> list[FotMobShot]:
    """Parse shotmap data from ``content.shotmap.shots``."""
    try:
        raw_shots = page_props["content"]["shotmap"]["shots"]
    except (KeyError, TypeError):
        return []

    if not isinstance(raw_shots, list):
        return []

    shots: list[FotMobShot] = []
    for s in raw_shots:
        shots.append(FotMobShot(
            event_type=s.get("eventType", ""),
            team_id=s.get("teamId"),
            player_name=s.get("playerName", ""),
            situation=s.get("situation", ""),
            shot_type=s.get("shotType", ""),
            minute=s.get("min"),
            expected_goals=s.get("expectedGoals"),
            is_home=bool(s.get("isHome", False)),
            on_target=bool(s.get("onTarget", False)),
        ))
    return shots


def _parse_events(
    page_props: dict,
) -> tuple[list[FotMobEvent], list[str], int | None, int | None]:
    """Parse match events from ``content.matchFacts.events``.

    Returns ``(events, event_types, home_score, away_score)``.
    """
    try:
        events_data = page_props["content"]["matchFacts"]["events"]
    except (KeyError, TypeError):
        return [], [], None, None

    if not isinstance(events_data, dict):
        return [], [], None, None

    events: list[FotMobEvent] = []
    type_set: set[str] = set()
    home_score: int | None = None
    away_score: int | None = None

    for raw_event in events_data.get("events", []):
        ev_type = raw_event.get("type", "")
        type_set.add(ev_type)

        time_val = raw_event.get("time")
        overload = raw_event.get("overloadTime")
        is_home = bool(raw_event.get("isHome", False))

        # Player info
        player = raw_event.get("player", {}) or {}
        player_name = player.get("name", "")
        player_id = player.get("id")

        # Card info
        card_obj = raw_event.get("card", None)
        card_type = None
        if isinstance(card_obj, str):
            card_type = card_obj
        elif isinstance(card_obj, dict):
            card_type = card_obj.get("type")

        # Goal details
        is_own_goal = bool(raw_event.get("ownGoal", False))
        assist_obj = raw_event.get("assist", {}) or {}
        assist_player = assist_obj.get("name")

        # VAR decision
        var_decision = None
        var_obj = raw_event.get("VAR", None)
        if isinstance(var_obj, dict):
            var_decision = var_obj.get("decision") or var_obj.get("result")
        elif isinstance(var_obj, str):
            var_decision = var_obj

        # Track score from goal events
        new_score = raw_event.get("newScore")
        if isinstance(new_score, list) and len(new_score) == 2:
            try:
                home_score = int(new_score[0])
                away_score = int(new_score[1])
            except (ValueError, TypeError):
                pass

        events.append(FotMobEvent(
            type=ev_type,
            time=time_val,
            overload_time=overload,
            is_home=is_home,
            player_name=player_name,
            player_id=player_id,
            card_type=card_type,
            is_own_goal=is_own_goal,
            assist_player=assist_player,
            var_decision=var_decision,
            raw=raw_event,
        ))

    return events, sorted(type_set), home_score, away_score


def _parse_score(page_props: dict) -> tuple[int | None, int | None]:
    """Parse final score from ``header.teams`` as fallback."""
    try:
        teams = page_props["header"]["teams"]
        home_score = int(teams[0]["score"])
        away_score = int(teams[1]["score"])
        return home_score, away_score
    except (KeyError, TypeError, IndexError, ValueError):
        return None, None


def summarize_for_llm(match_data: FotMobMatchData) -> str:
    """Produce a compact text summary of match data for LLM consumption.

    Sections: match info, stats, shotmap, events.
    """
    lines: list[str] = []

    # --- Match info ---
    score_str = ""
    if match_data.home_score is not None and match_data.away_score is not None:
        score_str = f" (Final score: {match_data.home_score}-{match_data.away_score})"
    lines.append(f"Match: {match_data.home_team} vs {match_data.away_team}{score_str}")
    if match_data.url:
        lines.append(f"Source: {match_data.url}")
    lines.append("")

    # --- Stats ---
    if match_data.stats_by_category:
        lines.append("=== STATS ===")
        for cat_key, stats in match_data.stats_by_category.items():
            for stat in stats:
                lines.append(
                    f"  {stat.title}: {match_data.home_team} {stat.home_value}, "
                    f"{match_data.away_team} {stat.away_value}"
                )
        lines.append("")

    # --- Shotmap ---
    if match_data.shots:
        lines.append("=== SHOTMAP ===")
        for shot in match_data.shots:
            team = match_data.home_team if shot.is_home else match_data.away_team
            xg_str = f", xG={shot.expected_goals:.2f}" if shot.expected_goals is not None else ""
            on_target_str = " [on target]" if shot.on_target else ""
            lines.append(
                f"  {shot.minute}' {shot.player_name} ({team}) — "
                f"{shot.event_type}, {shot.situation}, {shot.shot_type}"
                f"{xg_str}{on_target_str}"
            )
        lines.append("")

    # --- Events ---
    if match_data.events:
        lines.append("=== EVENTS ===")
        for ev in match_data.events:
            time_str = f"{ev.time}'" if ev.time is not None else "?"
            if ev.overload_time:
                time_str += f"+{ev.overload_time}"
            side = match_data.home_team if ev.is_home else match_data.away_team

            detail_parts: list[str] = []
            if ev.player_name:
                detail_parts.append(ev.player_name)
            if ev.card_type:
                detail_parts.append(f"card={ev.card_type}")
            if ev.is_own_goal:
                detail_parts.append("OWN GOAL")
            if ev.assist_player:
                detail_parts.append(f"assist={ev.assist_player}")
            if ev.var_decision:
                detail_parts.append(f"VAR: {ev.var_decision}")
            detail = ", ".join(detail_parts)

            lines.append(f"  {time_str} [{ev.type}] ({side}) {detail}")
        lines.append("")

    return "\n".join(lines)
