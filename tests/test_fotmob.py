"""Tests for agents.collector.fotmob — FotMob stat extraction."""

import json
import pytest
from unittest.mock import MagicMock

from agents.collector.fotmob import (
    FotMobExtractionError,
    FotMobMatchData,
    FotMobShot,
    FotMobEvent,
    FotMobStat,
    fetch_match_stats,
    find_stat,
    match_team,
    summarize_for_llm,
)


# ---------------------------------------------------------------------------
# Fixture: minimal __NEXT_DATA__ JSON embedded in HTML
# ---------------------------------------------------------------------------

_SAMPLE_STATS = {
    "props": {
        "pageProps": {
            "general": {
                "matchId": "4837357",
                "homeTeam": {"name": "Osasuna", "id": 8371},
                "awayTeam": {"name": "Real Madrid", "id": 8633},
            },
            "header": {
                "teams": [
                    {"name": "Osasuna", "score": 0, "id": 8371},
                    {"name": "Real Madrid", "score": 4, "id": 8633},
                ],
            },
            "content": {
                "stats": {
                    "Periods": {
                        "All": {
                            "stats": [
                                {
                                    "title": "Top stats",
                                    "key": "top_stats",
                                    "stats": [
                                        {
                                            "title": "Total shots",
                                            "key": "total_shots",
                                            "stats": [13, 15],
                                            "type": "text",
                                        },
                                    ],
                                },
                                {
                                    "title": "Shots",
                                    "key": "shots",
                                    "stats": [
                                        {
                                            "title": "Shots",
                                            "key": "shots",
                                            "stats": [None, None],
                                            "type": "title",
                                        },
                                        {
                                            "title": "Total shots",
                                            "key": "total_shots",
                                            "stats": [13, 15],
                                            "type": "text",
                                        },
                                        {
                                            "title": "Shots on target",
                                            "key": "ShotsOnTarget",
                                            "stats": [2, 5],
                                            "type": "text",
                                        },
                                        {
                                            "title": "Shots inside box",
                                            "key": "shots_inside_box",
                                            "stats": [11, 7],
                                            "type": "text",
                                        },
                                        {
                                            "title": "Shots outside box",
                                            "key": "shots_outside_box",
                                            "stats": [2, 8],
                                            "type": "text",
                                        },
                                    ],
                                },
                                {
                                    "title": "Passes",
                                    "key": "passes",
                                    "stats": [
                                        {
                                            "title": "Accurate passes",
                                            "key": "accurate_passes",
                                            "stats": ["293 (83%)", "511 (90%)"],
                                            "type": "text",
                                        },
                                    ],
                                },
                            ],
                        },
                    },
                },
                "shotmap": {
                    "shots": [
                        {
                            "eventType": "Goal",
                            "teamId": 8633,
                            "playerName": "Vinicius Jr",
                            "situation": "OpenPlay",
                            "shotType": "LeftFoot",
                            "min": 12,
                            "expectedGoals": 0.45,
                            "isHome": False,
                            "onTarget": True,
                        },
                        {
                            "eventType": "AttemptSaved",
                            "teamId": 8371,
                            "playerName": "Budimir",
                            "situation": "FromCorner",
                            "shotType": "Head",
                            "min": 25,
                            "expectedGoals": 0.12,
                            "isHome": True,
                            "onTarget": True,
                        },
                        {
                            "eventType": "Miss",
                            "teamId": 8633,
                            "playerName": "Bellingham",
                            "situation": "OpenPlay",
                            "shotType": "RightFoot",
                            "min": 33,
                            "expectedGoals": 0.08,
                            "isHome": False,
                            "onTarget": False,
                        },
                    ],
                },
                "matchFacts": {
                    "events": {
                        "events": [
                            {
                                "type": "Goal",
                                "time": 12,
                                "isHome": False,
                                "player": {"name": "Vinicius Jr", "id": 961995},
                                "assist": {"name": "Bellingham"},
                                "newScore": [0, 1],
                            },
                            {
                                "type": "Card",
                                "time": 38,
                                "isHome": True,
                                "player": {"name": "Moncayola", "id": 773152},
                                "card": "Yellow",
                            },
                            {
                                "type": "Goal",
                                "time": 55,
                                "isHome": False,
                                "player": {"name": "Bellingham", "id": 839498},
                                "newScore": [0, 2],
                            },
                            {
                                "type": "Substitution",
                                "time": 70,
                                "isHome": True,
                                "player": {"name": "Budimir", "id": 339344},
                            },
                        ],
                    },
                },
            },
        },
    },
}


def _make_html(next_data: dict) -> str:
    """Wrap a __NEXT_DATA__ dict in minimal HTML."""
    return (
        '<!DOCTYPE html><html><head></head><body>'
        f'<script id="__NEXT_DATA__" type="application/json">'
        f'{json.dumps(next_data)}'
        f'</script></body></html>'
    )


SAMPLE_HTML = _make_html(_SAMPLE_STATS)


# ---------------------------------------------------------------------------
# Tests: fetch_match_stats
# ---------------------------------------------------------------------------

class TestFetchMatchStats:

    def test_parses_valid_html(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_HTML

        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp

        data = fetch_match_stats(
            "https://www.fotmob.com/matches/osasuna-vs-real-madrid/2e2ylz",
            mock_http,
        )
        assert isinstance(data, FotMobMatchData)
        assert data.match_id == "4837357"
        assert data.home_team == "Osasuna"
        assert data.away_team == "Real Madrid"
        assert "shots" in data.stats_by_category
        assert "top_stats" in data.stats_by_category
        assert data.home_score == 0
        assert data.away_score == 4
        assert len(data.shots) == 3
        assert len(data.events) == 4

    def test_raises_on_non_200(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp

        with pytest.raises(FotMobExtractionError, match="HTTP 403"):
            fetch_match_stats("https://www.fotmob.com/matches/x", mock_http)

    def test_raises_on_missing_next_data(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "<html><body>No data here</body></html>"

        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp

        with pytest.raises(FotMobExtractionError, match="__NEXT_DATA__"):
            fetch_match_stats("https://www.fotmob.com/matches/x", mock_http)

    def test_raises_on_empty_data(self):
        """When no stats, shots, or events are present, should raise."""
        bad_data = {"props": {"pageProps": {"general": {"matchId": "1"}, "content": {}}}}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _make_html(bad_data)

        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp

        with pytest.raises(FotMobExtractionError, match="No stats, shots, or events"):
            fetch_match_stats("https://www.fotmob.com/matches/x", mock_http)

    def test_sends_browser_user_agent(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_HTML

        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp

        fetch_match_stats("https://www.fotmob.com/matches/x", mock_http)

        call_kwargs = mock_http.get.call_args
        headers = call_kwargs.kwargs.get("headers", {})
        assert "User-Agent" in headers
        assert "Mozilla" in headers["User-Agent"]


# ---------------------------------------------------------------------------
# Tests: find_stat
# ---------------------------------------------------------------------------

class TestFindStat:

    def _make_data(self) -> FotMobMatchData:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_HTML
        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp
        return fetch_match_stats("https://www.fotmob.com/matches/x", mock_http)

    def test_finds_by_exact_title(self):
        data = self._make_data()
        stat = find_stat(data, "Shots outside box")
        assert stat is not None
        assert stat.home_value == 2
        assert stat.away_value == 8
        assert stat.category == "shots"

    def test_finds_by_title_case_insensitive(self):
        data = self._make_data()
        stat = find_stat(data, "shots OUTSIDE BOX")
        assert stat is not None
        assert stat.away_value == 8

    def test_finds_by_key_fallback(self):
        data = self._make_data()
        stat = find_stat(data, "shots_outside_box")
        assert stat is not None
        assert stat.away_value == 8

    def test_finds_stat_in_different_category(self):
        data = self._make_data()
        stat = find_stat(data, "Accurate passes")
        assert stat is not None
        assert stat.category == "passes"

    def test_returns_none_for_missing_stat(self):
        data = self._make_data()
        stat = find_stat(data, "Nonexistent stat")
        assert stat is None

    def test_skips_title_type_entries(self):
        """The 'Shots' title entry (type=title) should not be returned."""
        data = self._make_data()
        stat = find_stat(data, "Shots")
        # Should find the 'top_stats' category "Total shots" or similar,
        # NOT the title-type "Shots" entry
        if stat is not None:
            assert stat.home_value is not None  # not None like title entries


# ---------------------------------------------------------------------------
# Tests: match_team
# ---------------------------------------------------------------------------

class TestMatchTeam:

    def _make_data(self) -> FotMobMatchData:
        return FotMobMatchData(
            match_id="123",
            home_team="Osasuna",
            away_team="Real Madrid",
        )

    def test_exact_match_home(self):
        assert match_team(self._make_data(), "Osasuna") == "home"

    def test_exact_match_away(self):
        assert match_team(self._make_data(), "Real Madrid") == "away"

    def test_case_insensitive(self):
        assert match_team(self._make_data(), "real madrid") == "away"
        assert match_team(self._make_data(), "OSASUNA") == "home"

    def test_substring_match(self):
        assert match_team(self._make_data(), "Real") == "away"
        assert match_team(self._make_data(), "Madrid") == "away"

    def test_no_match(self):
        assert match_team(self._make_data(), "Barcelona") is None


# ---------------------------------------------------------------------------
# Tests: shotmap parsing
# ---------------------------------------------------------------------------

class TestFotMobShotmapParsing:

    def _make_data(self) -> FotMobMatchData:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_HTML
        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp
        return fetch_match_stats("https://www.fotmob.com/matches/x", mock_http)

    def test_parses_shot_count(self):
        data = self._make_data()
        assert len(data.shots) == 3

    def test_parses_shot_fields(self):
        data = self._make_data()
        goal_shot = data.shots[0]
        assert goal_shot.event_type == "Goal"
        assert goal_shot.player_name == "Vinicius Jr"
        assert goal_shot.team_id == 8633
        assert goal_shot.situation == "OpenPlay"
        assert goal_shot.shot_type == "LeftFoot"
        assert goal_shot.minute == 12
        assert goal_shot.expected_goals == 0.45
        assert goal_shot.is_home is False
        assert goal_shot.on_target is True

    def test_parses_corner_shot(self):
        data = self._make_data()
        corner_shot = data.shots[1]
        assert corner_shot.situation == "FromCorner"
        assert corner_shot.shot_type == "Head"
        assert corner_shot.is_home is True

    def test_parses_miss(self):
        data = self._make_data()
        miss = data.shots[2]
        assert miss.event_type == "Miss"
        assert miss.on_target is False


# ---------------------------------------------------------------------------
# Tests: event parsing
# ---------------------------------------------------------------------------

class TestFotMobEventParsing:

    def _make_data(self) -> FotMobMatchData:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = SAMPLE_HTML
        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp
        return fetch_match_stats("https://www.fotmob.com/matches/x", mock_http)

    def test_parses_event_count(self):
        data = self._make_data()
        assert len(data.events) == 4

    def test_parses_goal_event(self):
        data = self._make_data()
        goal = data.events[0]
        assert goal.type == "Goal"
        assert goal.time == 12
        assert goal.player_name == "Vinicius Jr"
        assert goal.is_home is False
        assert goal.assist_player == "Bellingham"

    def test_parses_card_event(self):
        data = self._make_data()
        card = data.events[1]
        assert card.type == "Card"
        assert card.card_type == "Yellow"
        assert card.player_name == "Moncayola"
        assert card.is_home is True

    def test_parses_event_types(self):
        data = self._make_data()
        assert "Goal" in data.event_types
        assert "Card" in data.event_types
        assert "Substitution" in data.event_types

    def test_score_from_header(self):
        data = self._make_data()
        assert data.home_score == 0
        assert data.away_score == 4


# ---------------------------------------------------------------------------
# Tests: summarize_for_llm
# ---------------------------------------------------------------------------

class TestSummarizeForLlm:

    def test_includes_match_info(self):
        data = FotMobMatchData(
            match_id="123",
            home_team="Arsenal",
            away_team="Tottenham",
            home_score=2,
            away_score=1,
            url="https://www.fotmob.com/matches/test",
        )
        summary = summarize_for_llm(data)
        assert "Arsenal vs Tottenham" in summary
        assert "2-1" in summary
        assert "fotmob.com/matches/test" in summary

    def test_includes_stats(self):
        data = FotMobMatchData(
            match_id="123",
            home_team="Arsenal",
            away_team="Tottenham",
            stats_by_category={
                "top_stats": [
                    FotMobStat(title="Corners", key="corners",
                               home_value=5, away_value=2, category="top_stats"),
                ],
            },
        )
        summary = summarize_for_llm(data)
        assert "STATS" in summary
        assert "Corners: Arsenal 5, Tottenham 2" in summary

    def test_includes_shots(self):
        data = FotMobMatchData(
            match_id="123",
            home_team="Arsenal",
            away_team="Tottenham",
            shots=[
                FotMobShot(
                    event_type="Goal", team_id=1, player_name="Saka",
                    situation="FromCorner", shot_type="Head", minute=15,
                    expected_goals=0.35, is_home=True, on_target=True,
                ),
            ],
        )
        summary = summarize_for_llm(data)
        assert "SHOTMAP" in summary
        assert "Saka" in summary
        assert "Arsenal" in summary
        assert "FromCorner" in summary
        assert "xG=0.35" in summary

    def test_includes_events(self):
        data = FotMobMatchData(
            match_id="123",
            home_team="Arsenal",
            away_team="Tottenham",
            events=[
                FotMobEvent(
                    type="Goal", time=15, overload_time=None,
                    is_home=True, player_name="Saka",
                    assist_player="Odegaard",
                ),
                FotMobEvent(
                    type="Card", time=30, overload_time=None,
                    is_home=False, player_name="Son",
                    card_type="Yellow",
                ),
            ],
        )
        summary = summarize_for_llm(data)
        assert "EVENTS" in summary
        assert "Goal" in summary
        assert "Saka" in summary
        assert "assist=Odegaard" in summary
        assert "card=Yellow" in summary

    def test_empty_sections_omitted(self):
        data = FotMobMatchData(
            match_id="123",
            home_team="Arsenal",
            away_team="Tottenham",
            stats_by_category={
                "top_stats": [
                    FotMobStat(title="Corners", key="corners",
                               home_value=5, away_value=2, category="top_stats"),
                ],
            },
        )
        summary = summarize_for_llm(data)
        assert "STATS" in summary
        assert "SHOTMAP" not in summary
        assert "EVENTS" not in summary


# ---------------------------------------------------------------------------
# Tests: stats-only data (backward compatibility)
# ---------------------------------------------------------------------------

class TestNoEventsFallback:

    def test_stats_only_data_works(self):
        """Data with only stats (no shotmap/events) should still parse."""
        stats_only = {
            "props": {
                "pageProps": {
                    "general": {
                        "matchId": "999",
                        "homeTeam": {"name": "TeamA", "id": 1},
                        "awayTeam": {"name": "TeamB", "id": 2},
                    },
                    "content": {
                        "stats": {
                            "Periods": {
                                "All": {
                                    "stats": [
                                        {
                                            "title": "Top stats",
                                            "key": "top_stats",
                                            "stats": [
                                                {
                                                    "title": "Corners",
                                                    "key": "corners",
                                                    "stats": [5, 3],
                                                    "type": "text",
                                                },
                                            ],
                                        },
                                    ],
                                },
                            },
                        },
                    },
                },
            },
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = _make_html(stats_only)
        mock_http = MagicMock()
        mock_http.get.return_value = mock_resp

        data = fetch_match_stats("https://www.fotmob.com/matches/x", mock_http)
        assert data.match_id == "999"
        assert len(data.stats_by_category) > 0
        assert data.shots == []
        assert data.events == []
        assert data.home_score is None
        assert data.away_score is None
