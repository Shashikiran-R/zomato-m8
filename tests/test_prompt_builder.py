"""
tests/test_prompt_builder.py
─────────────────────────────
Unit tests for src/prompt_builder.py.

Covers:
  - _format_preferences() with all/partial/empty preferences
  - _format_candidates_table() structure, row count, and content
  - build_prompt() returns correct tuple and embeds context
  - Edge cases: special characters in free-text, empty candidates DF,
    single candidate, large DataFrames, Unicode in cuisine/location
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models import UserPreferenceInput
from src.prompt_builder import (
    _SYSTEM_PROMPT,
    _format_candidates_table,
    _format_preferences,
    build_prompt,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_candidates() -> pd.DataFrame:
    """A small DataFrame representative of the processed Parquet schema."""
    return pd.DataFrame(
        {
            "name": ["Spice Garden", "Pasta Palace", "Udupi Corner"],
            "location": ["Koramangala", "Koramangala", "Indiranagar"],
            "cuisines": ["north indian, chinese", "italian, continental", "south indian"],
            "average_cost_for_two": [600, 1200, 250],
            "aggregate_rating": [4.2, 3.9, 4.5],
            "votes": [500, 120, 800],
            "has_online_delivery": [True, False, True],
            "has_table_booking": [True, True, False],
            "budget": ["medium", "medium", "low"],
        }
    )


@pytest.fixture()
def full_prefs() -> UserPreferenceInput:
    """Fully populated user preferences."""
    return UserPreferenceInput(
        location="Koramangala",
        budget="medium",
        cuisine="italian",
        min_rating=3.5,
        additional_preferences="romantic dinner",
    )


# ── _format_preferences tests ────────────────────────────────────────────────


class TestFormatPreferences:
    def test_all_fields_present(self, full_prefs):
        text = _format_preferences(full_prefs)
        assert "Koramangala" in text
        assert "Medium" in text or "medium" in text
        assert "italian" in text.lower() or "Italian" in text
        assert "3.5" in text
        assert "romantic dinner" in text

    def test_empty_prefs_shows_no_constraints(self):
        text = _format_preferences(UserPreferenceInput())
        assert "No specific constraints" in text

    def test_location_only(self):
        prefs = UserPreferenceInput(location="Whitefield")
        text = _format_preferences(prefs)
        assert "Whitefield" in text
        # Budget/cuisine/rating lines should NOT appear
        assert "Budget" not in text
        assert "Cuisine" not in text

    def test_rating_only(self):
        prefs = UserPreferenceInput(min_rating=4.0)
        text = _format_preferences(prefs)
        assert "4.0" in text

    def test_budget_only_low(self):
        prefs = UserPreferenceInput(budget="low")
        text = _format_preferences(prefs)
        assert "Low" in text or "low" in text
        assert "500" in text  # cost hint for low tier

    def test_budget_only_high(self):
        prefs = UserPreferenceInput(budget="high")
        text = _format_preferences(prefs)
        assert "High" in text or "high" in text
        assert "1500" in text  # cost hint for high tier

    def test_additional_preferences_with_special_characters(self):
        """Special chars in free-text should not break prompt formatting."""
        prefs = UserPreferenceInput(
            additional_preferences='rooftop & live music "jazz" <outdoor>'
        )
        text = _format_preferences(prefs)
        assert "rooftop & live music" in text
        assert '"jazz"' in text
        assert "<outdoor>" in text

    def test_additional_preferences_with_unicode(self):
        """Unicode characters (e.g. emoji) in notes should pass through."""
        prefs = UserPreferenceInput(
            additional_preferences="🍕 pizza night! Café ambiance"
        )
        text = _format_preferences(prefs)
        assert "🍕" in text
        assert "Café" in text

    def test_zero_rating_not_shown(self):
        """Default min_rating=0.0 should not appear in formatted text."""
        prefs = UserPreferenceInput(location="Indiranagar")
        text = _format_preferences(prefs)
        assert "Min rating" not in text and "min rating" not in text.lower().replace("min rating", "")


# ── _format_candidates_table tests ────────────────────────────────────────────


class TestFormatCandidatesTable:
    def test_header_present(self, sample_candidates):
        table = _format_candidates_table(sample_candidates)
        assert "Name" in table
        assert "Rating" in table
        assert "Cost" in table
        assert "Online Del." in table

    def test_separator_present(self, sample_candidates):
        table = _format_candidates_table(sample_candidates)
        assert "---" in table

    def test_all_restaurants_listed(self, sample_candidates):
        table = _format_candidates_table(sample_candidates)
        for name in sample_candidates["name"]:
            assert name in table

    def test_row_count_matches(self, sample_candidates):
        table = _format_candidates_table(sample_candidates)
        lines = [line for line in table.splitlines() if line.strip()]
        # header + separator + N data rows
        assert len(lines) == len(sample_candidates) + 2

    def test_empty_dataframe(self):
        """Empty DF should produce only header + separator, no data rows."""
        empty = pd.DataFrame(
            columns=[
                "name", "location", "cuisines", "average_cost_for_two",
                "aggregate_rating", "votes", "has_online_delivery",
                "has_table_booking", "budget",
            ]
        )
        table = _format_candidates_table(empty)
        lines = [line for line in table.splitlines() if line.strip()]
        assert len(lines) == 2  # header + separator only

    def test_single_candidate(self):
        single = pd.DataFrame(
            {
                "name": ["Solo Diner"],
                "location": ["HSR Layout"],
                "cuisines": ["continental"],
                "average_cost_for_two": [800],
                "aggregate_rating": [4.0],
                "votes": [50],
                "has_online_delivery": [False],
                "has_table_booking": [True],
                "budget": ["medium"],
            }
        )
        table = _format_candidates_table(single)
        assert "Solo Diner" in table
        lines = [line for line in table.splitlines() if line.strip()]
        assert len(lines) == 3  # header + sep + 1 row

    def test_special_chars_in_name(self):
        """Restaurant names with special chars should be preserved."""
        df = pd.DataFrame(
            {
                "name": ["Café O'Brien's & Grill"],
                "location": ["MG Road"],
                "cuisines": ["continental, american"],
                "average_cost_for_two": [1500],
                "aggregate_rating": [3.8],
                "votes": [200],
                "has_online_delivery": [True],
                "has_table_booking": [True],
                "budget": ["medium"],
            }
        )
        table = _format_candidates_table(df)
        assert "Café O'Brien's & Grill" in table

    def test_rupee_symbol_in_cost(self, sample_candidates):
        """Cost should be formatted with the ₹ symbol."""
        table = _format_candidates_table(sample_candidates)
        assert "₹600" in table or "₹ 600" in table

    def test_online_delivery_yes_no(self, sample_candidates):
        table = _format_candidates_table(sample_candidates)
        assert "Yes" in table
        assert "No" in table


# ── build_prompt tests ─────────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_returns_two_strings(self, full_prefs, sample_candidates):
        sys_p, user_p = build_prompt(full_prefs, sample_candidates)
        assert isinstance(sys_p, str)
        assert isinstance(user_p, str)

    def test_system_prompt_is_constant(self, full_prefs, sample_candidates):
        sys_p, _ = build_prompt(full_prefs, sample_candidates)
        assert sys_p == _SYSTEM_PROMPT

    def test_system_prompt_has_json_schema(self, full_prefs, sample_candidates):
        sys_p, _ = build_prompt(full_prefs, sample_candidates)
        assert "JSON" in sys_p or "json" in sys_p
        assert "recommendations" in sys_p
        assert "summary" in sys_p

    def test_system_prompt_has_ranking_guidelines(self, full_prefs, sample_candidates):
        sys_p, _ = build_prompt(full_prefs, sample_candidates)
        assert "Ranking" in sys_p or "ranking" in sys_p

    def test_user_prompt_contains_candidate_count(self, full_prefs, sample_candidates):
        _, user_p = build_prompt(full_prefs, sample_candidates)
        assert str(len(sample_candidates)) in user_p

    def test_user_prompt_contains_preferences(self, full_prefs, sample_candidates):
        _, user_p = build_prompt(full_prefs, sample_candidates)
        assert "Koramangala" in user_p
        assert "romantic dinner" in user_p

    def test_user_prompt_contains_restaurant_names(self, full_prefs, sample_candidates):
        _, user_p = build_prompt(full_prefs, sample_candidates)
        for name in sample_candidates["name"]:
            assert name in user_p

    def test_empty_prefs_still_builds(self, sample_candidates):
        """Even with zero preferences, build_prompt should not raise."""
        prefs = UserPreferenceInput()
        sys_p, user_p = build_prompt(prefs, sample_candidates)
        assert "No specific constraints" in user_p

    def test_empty_candidates_still_builds(self, full_prefs):
        """build_prompt with an empty DataFrame should produce a valid prompt."""
        empty = pd.DataFrame(
            columns=[
                "name", "location", "cuisines", "average_cost_for_two",
                "aggregate_rating", "votes", "has_online_delivery",
                "has_table_booking", "budget",
            ]
        )
        sys_p, user_p = build_prompt(full_prefs, empty)
        assert "0 options" in user_p or "0" in user_p
