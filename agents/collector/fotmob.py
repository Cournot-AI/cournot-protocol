"""
FotMob match stat extractor.

Extracts structured match statistics from fotmob.com match pages by
parsing the __NEXT_DATA__ JSON embedded in the HTML (Next.js SSR data).

This avoids the Cloudflare Turnstile that blocks the /api/matchDetails
endpoint, and captures ALL stats including those behind the "All stats"
accordion that JS-rendering tools (Gemini UrlContext, Jina) cannot see.

Usage:
    from agents.collector.fotmob import fetch_match_stats, find_stat

    data = fetch_match_stats(url, http_client)
    stat = find_stat(data, "Shots outside box")
    if stat:
        print(f"{data.away_team}: {stat.away_value}")
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
class FotMobMatchData:
    """Parsed match data from a FotMob page."""
    match_id: str
    home_team: str
    away_team: str
    home_team_id: int | None = None
    away_team_id: int | None = None
    url: str = ""
    stats_by_category: dict[str, list[FotMobStat]] = field(default_factory=dict)


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
        stats_periods = page_props["content"]["stats"]["Periods"]["All"]["stats"]
    except (KeyError, TypeError) as e:
        raise FotMobExtractionError(
            f"Unexpected __NEXT_DATA__ structure — missing stats path: {e}"
        ) from e

    match_id = str(general.get("matchId", ""))
    home = general.get("homeTeam", {})
    away = general.get("awayTeam", {})

    stats_by_category: dict[str, list[FotMobStat]] = {}
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

    return FotMobMatchData(
        match_id=match_id,
        home_team=home.get("name", ""),
        away_team=away.get("name", ""),
        home_team_id=home.get("id"),
        away_team_id=away.get("id"),
        url=url,
        stats_by_category=stats_by_category,
    )
