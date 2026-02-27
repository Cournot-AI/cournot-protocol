"""FBRef site extractor.

Uses Gemini with UrlContext to read FBRef match pages (which are
behind Cloudflare JS challenges that block direct HTTP scraping),
then extracts structured match data into a text summary.
"""
from __future__ import annotations

import concurrent.futures
import re
from typing import Any

from .base import ExtractionError, SiteExtractor


def _parse_game_id(url: str) -> str | None:
    """Extract the FBRef game_id (hex/numeric ID) from a match URL.

    URL pattern: https://fbref.com/en/matches/{game_id}/...
    """
    m = re.search(r"fbref\.com/en/matches/([a-f0-9]+)", url)
    return m.group(1) if m else None


def _infer_league_season(url: str) -> tuple[str, str]:
    """Infer league and season from the FBRef match URL slug.

    Returns (league_string, season_string).
    """
    _LEAGUE_MAP: dict[str, str] = {
        "Premier-League": "ENG-Premier League",
        "La-Liga": "ESP-La Liga",
        "Serie-A": "ITA-Serie A",
        "Bundesliga": "GER-Bundesliga",
        "Ligue-1": "FRA-Ligue 1",
    }
    _DEFAULT_LEAGUE = "ENG-Premier League"

    league = _DEFAULT_LEAGUE
    slug = url.split("/matches/")[1] if "/matches/" in url else ""
    for url_slug, league_str in _LEAGUE_MAP.items():
        if url_slug in slug:
            league = league_str
            break

    year_match = re.search(r"-(\d{4})-", slug)
    if year_match:
        year = int(year_match.group(1))
        month_match = re.search(
            r"-(January|February|March|April|May|June|July|August|September|October|November|December)-",
            slug,
        )
        if month_match:
            early_months = {"January", "February", "March", "April", "May", "June", "July"}
            if month_match.group(1) in early_months:
                start_year = year - 1
            else:
                start_year = year
        else:
            start_year = year
        season = f"{start_year}-{start_year + 1}"
    else:
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        start_year = now.year - 1 if now.month <= 7 else now.year
        season = f"{start_year}-{start_year + 1}"

    return league, season


_EXTRACTION_PROMPT = """\
You are reading a FBRef.com match report page. Extract ALL match data \
into a structured text summary. Include:

1. **Score line**: "HomeTeam X - Y AwayTeam"
2. **Player stats table**: For each team, list every player with their \
key stats (minutes, goals, assists, shots, shots on target, passes, \
tackles, interceptions, fouls, cards).
3. **Shot events**: If a shot/goal log is visible, list each shot with \
minute, player, team, outcome (Goal/Saved/Blocked/Off Target), and \
body part.
4. **Team totals**: Possession, total shots, shots on target, corners, \
fouls, offsides, cards — for both teams.

Format the output as plain text sections. Do NOT summarize or omit data \
— include every player row and every stat column you can see on the page.

You MUST respond with ONLY the extracted data, no commentary.
"""


def _call_gemini_extract(
    client: Any, model: str, url: str,
) -> str:
    """Call Gemini with UrlContext to read the FBRef page and extract data.

    Separated for easy mocking in tests.
    """
    from google.genai import types

    tools = [
        types.Tool(url_context=types.UrlContext()),
    ]

    prompt = f"Read this FBRef match report page: {url}\n\n{_EXTRACTION_PROMPT}"

    def _do_call() -> Any:
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                tools=tools,
            ),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_do_call)
        response = future.result(timeout=60)

    # Extract text from response
    texts: list[str] = []
    for candidate in getattr(response, "candidates", []):
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []):
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
    return "\n".join(texts)


class FBRefExtractor(SiteExtractor):
    """Extracts match data from fbref.com using Gemini UrlContext.

    FBRef uses Cloudflare JS challenges that block direct HTTP scraping.
    This extractor leverages Gemini's UrlContext tool to read the page
    and extract structured match data into a text summary.

    Requires ``gemini_client`` and ``gemini_model`` kwargs.
    """

    source_id = "fbref"

    def can_handle(self, url: str) -> bool:
        return "fbref.com/en/matches/" in url

    def extract_and_summarize(
        self, url: str, http_client: Any, **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        game_id = _parse_game_id(url)
        if not game_id:
            raise ExtractionError(f"Could not extract game_id from URL: {url}")

        gemini_client = kwargs.get("gemini_client")
        gemini_model = kwargs.get("gemini_model", "gemini-2.5-flash")

        if gemini_client is None:
            raise ExtractionError(
                "FBRefExtractor requires gemini_client kwarg "
                "(FBRef blocks direct HTTP scraping via Cloudflare)"
            )

        league, season = _infer_league_season(url)

        try:
            data_summary = _call_gemini_extract(gemini_client, gemini_model, url)
        except Exception as e:
            raise ExtractionError(
                f"Gemini UrlContext failed to read FBRef page: {e}"
            ) from e

        if not data_summary or len(data_summary.strip()) < 50:
            raise ExtractionError(
                f"No meaningful data extracted from FBRef for game_id={game_id}"
            )

        # Prepend context header
        sections: list[str] = [
            f"Match data from: {url}",
            f"League: {league}, Season: {season}",
            "",
            data_summary,
        ]
        summary = "\n".join(sections)

        metadata = {
            "source_url": url,
            "fbref_game_id": game_id,
            "fbref_league": league,
            "fbref_season": season,
            "extraction_method": "gemini_url_context",
        }

        return summary, metadata
