"""
tests/test_llm_client.py
────────────────────────
Unit tests for src/prompt_builder.py and src/llm_client.py.

All Groq API calls are mocked — no real API key is needed.

Covers:
  - build_prompt() structure and content
  - _parse_llm_response() happy path and error cases
  - get_recommendations() retry + backoff logic
  - get_recommendations() heuristic fallback on total failure
  - Fallback metadata propagation
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.filters import FilterResult
from src.llm_client import _heuristic_fallback, _parse_llm_response, get_recommendations
from src.models import RecommendationItem, RecommendationResponse, UserPreferenceInput
from src.prompt_builder import build_prompt, _format_candidates_table, _format_preferences


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Spice Garden", "Pasta Palace", "Udupi Corner"],
            "location": ["Koramangala", "Koramangala", "Indiranagar"],
            "cuisines": ["north indian", "italian", "south indian"],
            "average_cost_for_two": [600, 1200, 250],
            "aggregate_rating": [4.2, 3.9, 4.5],
            "votes": [500, 120, 800],
            "has_online_delivery": [True, False, True],
            "has_table_booking": [True, True, False],
            "budget": ["medium", "medium", "low"],
        }
    )


@pytest.fixture()
def sample_prefs() -> UserPreferenceInput:
    return UserPreferenceInput(
        location="Koramangala",
        budget="medium",
        cuisine="italian",
        min_rating=3.5,
        additional_preferences="romantic dinner",
    )


@pytest.fixture()
def filter_result(sample_candidates) -> FilterResult:
    return FilterResult(candidates=sample_candidates, fallback_used=False, fallback_note="")


@pytest.fixture()
def valid_llm_json(sample_candidates) -> str:
    """A valid JSON string as the LLM would return it."""
    recommendations = []
    for _, row in sample_candidates.head(3).iterrows():
        recommendations.append(
            {
                "name": row["name"],
                "location": row["location"],
                "cuisines": row["cuisines"],
                "aggregate_rating": row["aggregate_rating"],
                "average_cost_for_two": int(row["average_cost_for_two"]),
                "has_online_delivery": bool(row["has_online_delivery"]),
                "has_table_booking": bool(row["has_table_booking"]),
                "explanation": f"{row['name']} is a great choice for you.",
            }
        )
    return json.dumps({"recommendations": recommendations, "summary": "Great options!"})


# ── prompt_builder tests ──────────────────────────────────────────────────────


class TestFormatPreferences:
    def test_all_fields(self, sample_prefs):
        text = _format_preferences(sample_prefs)
        assert "Koramangala" in text
        assert "medium" in text.lower() or "Medium" in text
        assert "italian" in text.lower() or "Italian" in text
        assert "3.5" in text
        assert "romantic dinner" in text

    def test_empty_prefs(self):
        text = _format_preferences(UserPreferenceInput())
        assert "No specific constraints" in text

    def test_partial_prefs(self):
        prefs = UserPreferenceInput(location="Whitefield", min_rating=4.0)
        text = _format_preferences(prefs)
        assert "Whitefield" in text
        assert "4.0" in text
        # budget / cuisine lines should not appear
        assert "Budget" not in text or "None" not in text


class TestFormatCandidatesTable:
    def test_header_present(self, sample_candidates):
        table = _format_candidates_table(sample_candidates)
        assert "Name" in table
        assert "Rating" in table

    def test_all_restaurants_listed(self, sample_candidates):
        table = _format_candidates_table(sample_candidates)
        for name in sample_candidates["name"]:
            assert name in table

    def test_row_count(self, sample_candidates):
        table = _format_candidates_table(sample_candidates)
        # header + separator + N data rows
        lines = [l for l in table.splitlines() if l.strip()]
        assert len(lines) == len(sample_candidates) + 2


class TestBuildPrompt:
    def test_returns_two_strings(self, sample_prefs, sample_candidates):
        sys_p, user_p = build_prompt(sample_prefs, sample_candidates)
        assert isinstance(sys_p, str)
        assert isinstance(user_p, str)

    def test_system_prompt_has_json_instruction(self, sample_prefs, sample_candidates):
        sys_p, _ = build_prompt(sample_prefs, sample_candidates)
        assert "json" in sys_p.lower() or "JSON" in sys_p

    def test_user_prompt_contains_candidate_count(self, sample_prefs, sample_candidates):
        _, user_p = build_prompt(sample_prefs, sample_candidates)
        assert str(len(sample_candidates)) in user_p

    def test_user_prompt_contains_preference_info(self, sample_prefs, sample_candidates):
        _, user_p = build_prompt(sample_prefs, sample_candidates)
        assert "Koramangala" in user_p


# ── _parse_llm_response tests ─────────────────────────────────────────────────


class TestParseLlmResponse:
    def test_valid_json(self, valid_llm_json):
        result = _parse_llm_response(valid_llm_json)
        assert isinstance(result, RecommendationResponse)
        assert len(result.recommendations) == 3

    def test_strips_markdown_fences(self, valid_llm_json):
        fenced = f"```json\n{valid_llm_json}\n```"
        result = _parse_llm_response(fenced)
        assert isinstance(result, RecommendationResponse)

    def test_strips_plain_fences(self, valid_llm_json):
        fenced = f"```\n{valid_llm_json}\n```"
        result = _parse_llm_response(fenced)
        assert isinstance(result, RecommendationResponse)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="invalid JSON"):
            _parse_llm_response("not json at all {{")

    def test_wrong_schema_raises(self):
        # Missing required "recommendations" key
        bad = json.dumps({"answer": "hello"})
        with pytest.raises(ValueError, match="schema"):
            _parse_llm_response(bad)


# ── _heuristic_fallback tests ─────────────────────────────────────────────────


class TestHeuristicFallback:
    def test_returns_response(self, sample_candidates):
        resp = _heuristic_fallback(sample_candidates, False, "", "timeout")
        assert isinstance(resp, RecommendationResponse)
        assert resp.fallback_used is True
        assert len(resp.recommendations) == 3  # only 3 rows in fixture

    def test_max_five_items(self):
        big = pd.DataFrame(
            {
                "name": [f"R{i}" for i in range(10)],
                "location": ["Loc"] * 10,
                "cuisines": ["c"] * 10,
                "average_cost_for_two": [500] * 10,
                "aggregate_rating": [4.0] * 10,
                "votes": [100] * 10,
                "has_online_delivery": [True] * 10,
                "has_table_booking": [False] * 10,
                "budget": ["medium"] * 10,
            }
        )
        resp = _heuristic_fallback(big, False, "", "timeout")
        assert len(resp.recommendations) <= 5

    def test_fallback_note_appended(self, sample_candidates):
        resp = _heuristic_fallback(sample_candidates, True, "rating removed", "timeout")
        assert "rating removed" in resp.fallback_note
        assert "heuristic" in resp.fallback_note.lower()


# ── get_recommendations integration tests (mocked) ────────────────────────────


class TestGetRecommendations:
    def _make_mock_response(self, json_str: str) -> MagicMock:
        """Build a mock that looks like a Groq ChatCompletion response."""
        msg = MagicMock()
        msg.content = json_str
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    def test_happy_path(self, sample_prefs, filter_result, valid_llm_json):
        mock_resp = self._make_mock_response(valid_llm_json)

        with patch("src.llm_client.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = mock_resp
            result = get_recommendations(
                sample_prefs, filter_result,
                groq_api_key="test-key",
            )

        assert isinstance(result, RecommendationResponse)
        assert len(result.recommendations) >= 1
        assert result.fallback_used is False

    def test_retries_on_rate_limit(self, sample_prefs, filter_result, valid_llm_json):
        from groq import RateLimitError

        mock_resp = self._make_mock_response(valid_llm_json)

        # Fail twice, succeed on the third attempt
        side_effects = [
            RateLimitError("rate limit", response=MagicMock(status_code=429), body={}),
            RateLimitError("rate limit", response=MagicMock(status_code=429), body={}),
            mock_resp,
        ]

        with patch("src.llm_client.Groq") as MockGroq:
            with patch("src.llm_client.time.sleep"):  # skip actual sleep
                MockGroq.return_value.chat.completions.create.side_effect = side_effects
                result = get_recommendations(
                    sample_prefs, filter_result,
                    groq_api_key="test-key",
                    max_retries=3,
                )

        assert len(result.recommendations) >= 1

    def test_falls_back_after_all_retries_fail(self, sample_prefs, filter_result):
        import httpx
        from groq import APIConnectionError

        mock_request = httpx.Request("POST", "https://api.groq.com/")

        with patch("src.llm_client.Groq") as MockGroq:
            with patch("src.llm_client.time.sleep"):
                MockGroq.return_value.chat.completions.create.side_effect = (
                    APIConnectionError(message="connection refused", request=mock_request)
                )
                result = get_recommendations(
                    sample_prefs, filter_result,
                    groq_api_key="test-key",
                    max_retries=2,
                )

        assert result.fallback_used is True
        assert "heuristic" in result.fallback_note.lower()
        assert len(result.recommendations) > 0

    def test_empty_candidates_returns_early(self, sample_prefs):
        empty_result = FilterResult(
            candidates=pd.DataFrame(columns=["name","location","cuisines",
                "average_cost_for_two","aggregate_rating","votes",
                "has_online_delivery","has_table_booking","budget"]),
            fallback_used=True,
            fallback_note="all filters relaxed",
        )
        with patch("src.llm_client.Groq"):
            result = get_recommendations(
                sample_prefs, empty_result,
                groq_api_key="test-key",
            )
        assert result.recommendations == []
        assert "No restaurants" in result.summary

    def test_filter_fallback_metadata_propagated(self, sample_prefs, filter_result, valid_llm_json):
        filter_result.fallback_used = True
        filter_result.fallback_note = "rating filter removed"
        mock_resp = self._make_mock_response(valid_llm_json)

        with patch("src.llm_client.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = mock_resp
            result = get_recommendations(
                sample_prefs, filter_result,
                groq_api_key="test-key",
            )

        assert result.fallback_used is True
        assert "rating filter removed" in (result.fallback_note or "")
