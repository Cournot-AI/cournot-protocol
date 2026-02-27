"""Tests for FBRef site extractor."""
import sys
import pytest
from unittest.mock import MagicMock, patch

# Ensure google.genai can be imported even when the package isn't installed.
if "google.genai" not in sys.modules:
    _genai_mock = MagicMock()
    sys.modules.setdefault("google", MagicMock())
    sys.modules["google.genai"] = _genai_mock
    sys.modules["google.genai.types"] = _genai_mock.types

from agents.collector.extractors.fbref_ext import FBRefExtractor, _parse_game_id, _infer_league_season
from agents.collector.extractors.base import ExtractionError


class TestParseGameId:
    def test_extracts_from_standard_url(self):
        url = "https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal-January-1-2025-Premier-League"
        assert _parse_game_id(url) == "7e6892e4"

    def test_extracts_from_url_with_trailing_slash(self):
        url = "https://fbref.com/en/matches/abc123de/Some-Match/"
        assert _parse_game_id(url) == "abc123de"

    def test_returns_none_for_non_match_url(self):
        url = "https://fbref.com/en/squads/18bb7c10/Arsenal"
        assert _parse_game_id(url) is None


class TestInferLeagueSeason:
    def test_infers_premier_league(self):
        url = "https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal-January-1-2025-Premier-League"
        league, season = _infer_league_season(url)
        assert league == "ENG-Premier League"

    def test_infers_la_liga(self):
        url = "https://fbref.com/en/matches/abc123/Osasuna-Real-Madrid-February-21-2026-La-Liga"
        league, season = _infer_league_season(url)
        assert league == "ESP-La Liga"

    def test_defaults_to_premier_league(self):
        url = "https://fbref.com/en/matches/abc123/Some-Match"
        league, season = _infer_league_season(url)
        assert league == "ENG-Premier League"


_SAMPLE_EXTRACTED_DATA = """\
Everton 0 - 1 Manchester United

=== TEAM STATS ===
Possession: Everton 45%, Manchester United 55%
Total Shots: Everton 8, Manchester United 15
Shots on Target: Everton 2, Manchester United 6
Corners: Everton 3, Manchester United 7
Fouls: Everton 12, Manchester United 9

=== PLAYER STATS ===

--- Everton ---
  Pickford: min=90 saves=5 GA=1
  Calvert-Lewin: min=90 goals=0 shots=3 SoT=1

--- Manchester United ---
  Lammens: min=90 saves=2 GA=0
  Rashford: min=78 goals=1 assists=0 shots=4 SoT=2
  Fernandes: min=90 goals=0 assists=1 shots=3 SoT=1

=== SHOT EVENTS ===
  23' Rashford (Manchester United) — Goal Right Foot dist=14
  45' Calvert-Lewin (Everton) — Saved Header dist=8
  67' Fernandes (Manchester United) — Off Target Left Foot dist=22
"""


class TestFBRefExtractor:
    def test_can_handle_match_url(self):
        ext = FBRefExtractor()
        assert ext.can_handle("https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal")

    def test_cannot_handle_squad_url(self):
        ext = FBRefExtractor()
        assert not ext.can_handle("https://fbref.com/en/squads/18bb7c10/Arsenal")

    def test_cannot_handle_non_fbref(self):
        ext = FBRefExtractor()
        assert not ext.can_handle("https://fotmob.com/matches/abc123")

    def test_extract_calls_gemini_and_returns_summary(self):
        ext = FBRefExtractor()
        mock_http = MagicMock()
        mock_client = MagicMock()

        with patch("agents.collector.extractors.fbref_ext._call_gemini_extract",
                   return_value=_SAMPLE_EXTRACTED_DATA) as mock_call:
            summary, meta = ext.extract_and_summarize(
                "https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal-February-12-2026-Premier-League",
                mock_http,
                gemini_client=mock_client,
                gemini_model="gemini-2.5-flash",
            )

        # Verify Gemini was called with the URL
        mock_call.assert_called_once_with(
            mock_client, "gemini-2.5-flash",
            "https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal-February-12-2026-Premier-League",
        )

        assert "Manchester United" in summary
        assert "Rashford" in summary
        assert meta["fbref_game_id"] == "7e6892e4"
        assert meta["source_url"] == "https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal-February-12-2026-Premier-League"
        assert meta["extraction_method"] == "gemini_url_context"
        assert meta["fbref_league"] == "ENG-Premier League"

    def test_extract_raises_on_empty_data(self):
        ext = FBRefExtractor()
        mock_client = MagicMock()

        with patch("agents.collector.extractors.fbref_ext._call_gemini_extract",
                   return_value="  "):
            with pytest.raises(ExtractionError, match="No meaningful data"):
                ext.extract_and_summarize(
                    "https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal-2026-Premier-League",
                    MagicMock(),
                    gemini_client=mock_client,
                )

    def test_extract_raises_on_gemini_failure(self):
        ext = FBRefExtractor()
        mock_client = MagicMock()

        with patch("agents.collector.extractors.fbref_ext._call_gemini_extract",
                   side_effect=RuntimeError("Gemini timeout")):
            with pytest.raises(ExtractionError, match="Gemini UrlContext failed"):
                ext.extract_and_summarize(
                    "https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal-2026-Premier-League",
                    MagicMock(),
                    gemini_client=mock_client,
                )

    def test_extract_raises_on_bad_url(self):
        ext = FBRefExtractor()
        with pytest.raises(ExtractionError, match="game_id"):
            ext.extract_and_summarize(
                "https://fbref.com/en/squads/foo",
                MagicMock(),
                gemini_client=MagicMock(),
            )

    def test_extract_raises_without_gemini_client(self):
        ext = FBRefExtractor()
        with pytest.raises(ExtractionError, match="requires gemini_client"):
            ext.extract_and_summarize(
                "https://fbref.com/en/matches/7e6892e4/Brentford-Arsenal",
                MagicMock(),
            )
