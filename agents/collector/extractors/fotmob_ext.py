"""FotMob site extractor.

Wraps the existing fotmob.py module (fetch_match_stats, summarize_for_llm)
into the SiteExtractor interface.
"""
from __future__ import annotations

from typing import Any

from .base import ExtractionError, SiteExtractor
from ..fotmob import (
    FotMobExtractionError,
    fetch_match_stats as fotmob_fetch_match_stats,
    summarize_for_llm as fotmob_summarize_for_llm,
)


class FotMobExtractor(SiteExtractor):
    """Extracts structured match data from fotmob.com match pages."""

    source_id = "fotmob"

    def can_handle(self, url: str) -> bool:
        return "fotmob.com/matches/" in url

    def extract_and_summarize(
        self, url: str, http_client: Any, **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        try:
            match_data = fotmob_fetch_match_stats(url, http_client)
        except FotMobExtractionError as e:
            raise ExtractionError(str(e)) from e

        summary = fotmob_summarize_for_llm(match_data)

        metadata = {
            "fotmob_url": url,
            "fotmob_match_id": match_data.match_id,
            "fotmob_home_team": match_data.home_team,
            "fotmob_away_team": match_data.away_team,
            "fotmob_home_score": match_data.home_score,
            "fotmob_away_score": match_data.away_score,
            "fotmob_stats_categories": list(match_data.stats_by_category.keys()),
            "fotmob_shots_count": len(match_data.shots),
            "fotmob_events_count": len(match_data.events),
        }

        return summary, metadata
