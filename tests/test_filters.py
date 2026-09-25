"""
tests/test_filters.py
─────────────────────
Unit tests for src/filters.py and src/models.py.

Covers:
  - Individual filter functions (location, cuisine, budget, rating)
  - Full shortlist() pipeline with single and multi-criteria inputs
  - Ranking and pruning logic
  - Fallback ladder (zero-result strict filters)
  - Edge cases: empty DF, None prefs, special characters, boundary ratings
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.filters import (
    FilterResult,
    _filter_budget,
    _filter_cuisine,
    _filter_location,
    _filter_rating,
    _rank_and_prune,
    shortlist,
)
from src.models import BudgetTier, RecommendationItem, RecommendationResponse, UserPreferenceInput


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """A small DataFrame representative of the processed Parquet schema."""
    return pd.DataFrame(
        {
            "name": [
                "Spice Garden",
                "Pasta Palace",
                "Udupi Corner",
                "Beijing Bites",
                "Tandoor Express",
                "Cafe Latte",
                "Biryani Hub",
            ],
            "location": [
                "Koramangala",
                "Koramangala",
                "Indiranagar",
                "Indiranagar",
                "Whitefield",
                "Koramangala",
                "Whitefield",
            ],
            "cuisines": [
                "north indian, chinese",
                "italian, continental",
                "south indian",
                "chinese, thai",
                "north indian, mughlai",
                "cafe, beverages",
                "north indian, biryani",
            ],
            "average_cost_for_two": [600, 1200, 250, 900, 400, 350, 700],
            "aggregate_rating": [4.2, 3.9, 4.5, 3.5, 4.1, 3.8, 4.3],
            "votes": [500, 120, 800, 200, 350, 90, 620],
            "has_online_delivery": [True, False, True, True, True, False, True],
            "has_table_booking": [True, True, False, False, True, False, False],
            "budget": ["medium", "medium", "low", "medium", "low", "low", "medium"],
        }
    )


# ── UserPreferenceInput validation ─────────────────────────────────────────────


class TestUserPreferenceInput:
    def test_defaults_are_none(self):
        p = UserPreferenceInput()
        assert p.location is None
        assert p.cuisine is None
        assert p.budget is None
        assert p.min_rating == 0.0
        assert p.additional_preferences is None

    def test_empty_string_becomes_none(self):
        p = UserPreferenceInput(location="  ", cuisine="")
        assert p.location is None
        assert p.cuisine is None

    def test_budget_enum_coercion(self):
        p = UserPreferenceInput(budget="medium")
        assert p.budget == "medium"

    def test_invalid_budget_raises(self):
        with pytest.raises(Exception):
            UserPreferenceInput(budget="ultra")

    def test_min_rating_clamp_validation(self):
        with pytest.raises(Exception):
            UserPreferenceInput(min_rating=6.0)

    def test_additional_preferences_stripped(self):
        p = UserPreferenceInput(additional_preferences="  romantic dinner  ")
        assert p.additional_preferences == "romantic dinner"


# ── Location filter ───────────────────────────────────────────────────────────


class TestFilterLocation:
    def test_exact_match(self, sample_df):
        result = _filter_location(sample_df, "Koramangala")
        assert set(result["name"]) == {"Spice Garden", "Pasta Palace", "Cafe Latte"}

    def test_case_insensitive(self, sample_df):
        result = _filter_location(sample_df, "koramangala")
        assert len(result) == 3

    def test_partial_match(self, sample_df):
        # "Indira" should match "Indiranagar"
        result = _filter_location(sample_df, "Indira")
        assert set(result["name"]) == {"Udupi Corner", "Beijing Bites"}

    def test_none_returns_all(self, sample_df):
        result = _filter_location(sample_df, None)
        assert len(result) == len(sample_df)

    def test_empty_string_returns_all(self, sample_df):
        result = _filter_location(sample_df, "")
        assert len(result) == len(sample_df)

    def test_no_match_returns_empty(self, sample_df):
        result = _filter_location(sample_df, "Atlantis")
        assert len(result) == 0


# ── Cuisine filter ────────────────────────────────────────────────────────────


class TestFilterCuisine:
    def test_single_tag(self, sample_df):
        result = _filter_cuisine(sample_df, "italian")
        assert set(result["name"]) == {"Pasta Palace"}

    def test_case_insensitive(self, sample_df):
        result = _filter_cuisine(sample_df, "CHINESE")
        assert "Spice Garden" in result["name"].values
        assert "Beijing Bites" in result["name"].values

    def test_partial_cuisine(self, sample_df):
        # "indian" matches "north indian" and "south indian"
        result = _filter_cuisine(sample_df, "indian")
        assert len(result) >= 4

    def test_none_returns_all(self, sample_df):
        result = _filter_cuisine(sample_df, None)
        assert len(result) == len(sample_df)

    def test_no_match(self, sample_df):
        result = _filter_cuisine(sample_df, "sushi")
        assert len(result) == 0


# ── Budget filter ─────────────────────────────────────────────────────────────


class TestFilterBudget:
    def test_low(self, sample_df):
        result = _filter_budget(sample_df, "low")
        assert set(result["name"]) == {"Udupi Corner", "Tandoor Express", "Cafe Latte"}

    def test_medium(self, sample_df):
        result = _filter_budget(sample_df, "medium")
        assert set(result["name"]) == {"Spice Garden", "Pasta Palace", "Beijing Bites", "Biryani Hub"}

    def test_high_empty(self, sample_df):
        result = _filter_budget(sample_df, "high")
        assert len(result) == 0

    def test_budget_tier_enum(self, sample_df):
        result = _filter_budget(sample_df, BudgetTier.LOW)
        assert len(result) == 3

    def test_none_returns_all(self, sample_df):
        result = _filter_budget(sample_df, None)
        assert len(result) == len(sample_df)


# ── Rating filter ─────────────────────────────────────────────────────────────


class TestFilterRating:
    def test_threshold_inclusive(self, sample_df):
        # Udupi Corner has 4.5 — should be included at threshold 4.5
        result = _filter_rating(sample_df, 4.5)
        assert "Udupi Corner" in result["name"].values
        assert len(result) == 1

    def test_zero_returns_all(self, sample_df):
        result = _filter_rating(sample_df, 0.0)
        assert len(result) == len(sample_df)

    def test_high_threshold_empty(self, sample_df):
        result = _filter_rating(sample_df, 5.0)
        assert len(result) == 0

    def test_mid_threshold(self, sample_df):
        result = _filter_rating(sample_df, 4.0)
        # Spice Garden(4.2), Udupi Corner(4.5), Tandoor Express(4.1), Biryani Hub(4.3)
        assert len(result) == 4


# ── Ranking and pruning ───────────────────────────────────────────────────────


class TestRankAndPrune:
    def test_sorted_by_rating_desc(self, sample_df):
        result = _rank_and_prune(sample_df, max_candidates=7)
        ratings = result["aggregate_rating"].tolist()
        assert ratings == sorted(ratings, reverse=True)

    def test_ties_broken_by_votes(self, sample_df):
        # Create a tie in rating
        tie_df = pd.DataFrame(
            {
                "name": ["A", "B"],
                "location": ["X", "X"],
                "cuisines": ["c", "c"],
                "average_cost_for_two": [100, 100],
                "aggregate_rating": [4.0, 4.0],
                "votes": [50, 200],
                "has_online_delivery": [True, True],
                "has_table_booking": [False, False],
                "budget": ["low", "low"],
            }
        )
        result = _rank_and_prune(tie_df, max_candidates=2)
        assert result.iloc[0]["name"] == "B"  # higher votes wins

    def test_prune_to_max(self, sample_df):
        result = _rank_and_prune(sample_df, max_candidates=3)
        assert len(result) == 3

    def test_index_reset(self, sample_df):
        result = _rank_and_prune(sample_df, max_candidates=5)
        assert list(result.index) == list(range(len(result)))


# ── Full shortlist pipeline ───────────────────────────────────────────────────


class TestShortlist:
    def test_no_filters(self, sample_df):
        prefs = UserPreferenceInput()
        result = shortlist(sample_df, prefs, max_candidates=15)
        assert isinstance(result, FilterResult)
        assert len(result.candidates) == len(sample_df)
        assert not result.fallback_used

    def test_location_only(self, sample_df):
        prefs = UserPreferenceInput(location="Whitefield")
        result = shortlist(sample_df, prefs)
        assert set(result.candidates["name"]) == {"Tandoor Express", "Biryani Hub"}
        assert not result.fallback_used

    def test_multi_criteria(self, sample_df):
        prefs = UserPreferenceInput(
            location="Koramangala",
            budget="medium",
            cuisine="chinese",
            min_rating=4.0,
        )
        result = shortlist(sample_df, prefs)
        # Only Spice Garden: Koramangala + medium + chinese + rating 4.2
        assert len(result.candidates) == 1
        assert result.candidates.iloc[0]["name"] == "Spice Garden"
        assert not result.fallback_used

    def test_result_is_ranked(self, sample_df):
        prefs = UserPreferenceInput(location="Koramangala")
        result = shortlist(sample_df, prefs)
        ratings = result.candidates["aggregate_rating"].tolist()
        assert ratings == sorted(ratings, reverse=True)

    def test_max_candidates_respected(self, sample_df):
        prefs = UserPreferenceInput()
        result = shortlist(sample_df, prefs, max_candidates=3)
        assert len(result.candidates) <= 3

    def test_fallback_on_zero_results(self, sample_df):
        prefs = UserPreferenceInput(
            location="Atlantis",  # no such location
            cuisine="sushi",
            budget="high",
            min_rating=5.0,
        )
        result = shortlist(sample_df, prefs)
        assert result.fallback_used
        assert len(result.candidates) > 0
        assert result.fallback_note != ""

    def test_empty_dataframe_fallback(self):
        empty_df = pd.DataFrame(
            columns=[
                "name", "location", "cuisines", "average_cost_for_two",
                "aggregate_rating", "votes", "has_online_delivery",
                "has_table_booking", "budget",
            ]
        )
        prefs = UserPreferenceInput(location="Koramangala")
        result = shortlist(empty_df, prefs)
        assert result.fallback_used
        assert len(result.candidates) == 0  # even fallback finds nothing in empty DF


# ── Model instantiation ───────────────────────────────────────────────────────


class TestRecommendationModels:
    def test_recommendation_item(self):
        item = RecommendationItem(
            name="Spice Garden",
            location="Koramangala",
            cuisines="north indian",
            aggregate_rating=4.2,
            average_cost_for_two=600,
            has_online_delivery=True,
            has_table_booking=False,
            explanation="Great food and value.",
        )
        assert item.name == "Spice Garden"
        assert item.aggregate_rating == 4.2

    def test_recommendation_response_defaults(self):
        resp = RecommendationResponse(recommendations=[])
        assert resp.fallback_used is False
        assert resp.summary is None
        assert resp.fallback_note is None

    def test_recommendation_response_with_items(self):
        item = RecommendationItem(
            name="Test", location="Loc", cuisines="c",
            aggregate_rating=4.0, average_cost_for_two=500,
            has_online_delivery=True, has_table_booking=False,
            explanation="Good.",
        )
        resp = RecommendationResponse(
            recommendations=[item],
            summary="Top pick.",
            fallback_used=True,
            fallback_note="rating filter removed",
        )
        assert len(resp.recommendations) == 1
        assert resp.fallback_used is True
