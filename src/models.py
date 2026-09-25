"""
src/models.py
─────────────
Pydantic data models shared across the entire application.

Defines:
  - UserPreferenceInput   Input schema from the user (API / UI).
  - RestaurantRecord      A single cleaned restaurant row.
  - RecommendationItem    One LLM-ranked recommendation with explanation.
  - RecommendationResponse  Full LLM response returned to the caller.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Enumerations ─────────────────────────────────────────────────────────────


class BudgetTier(str, Enum):
    """Budget category based on approximate cost for two people."""
    LOW = "low"       # cost < 500
    MEDIUM = "medium" # 500 <= cost <= 1500
    HIGH = "high"     # cost > 1500


# ── Input model ───────────────────────────────────────────────────────────────


class UserPreferenceInput(BaseModel):
    """
    User-supplied preferences that drive restaurant filtering and ranking.

    All fields are optional except none are truly required — the system
    degrades gracefully when fields are omitted (treats them as "any").

    Examples
    --------
    >>> pref = UserPreferenceInput(
    ...     location="Koramangala",
    ...     budget="medium",
    ...     cuisine="italian",
    ...     min_rating=3.5,
    ...     additional_preferences="rooftop with live music",
    ... )
    """

    location: Optional[str] = Field(
        default=None,
        description="Area or neighbourhood to search in (case-insensitive substring match).",
        examples=["Koramangala", "Indiranagar"],
    )
    budget: Optional[BudgetTier] = Field(
        default=None,
        description="Spend tier: low (<500), medium (500-1500), high (>1500) for two people.",
    )
    cuisine: Optional[str] = Field(
        default=None,
        description="Preferred cuisine type (case-insensitive substring match).",
        examples=["Italian", "South Indian", "Chinese"],
    )
    min_rating: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Minimum aggregate rating threshold (inclusive).",
    )
    additional_preferences: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Free-text notes passed to the LLM (e.g. 'romantic', 'pet-friendly').",
    )

    @field_validator("location", "cuisine", mode="before")
    @classmethod
    def strip_and_none_if_empty(cls, v: object) -> Optional[str]:
        """Coerce empty/whitespace strings to None."""
        if v is None:
            return None
        stripped = str(v).strip()
        return stripped if stripped else None

    @field_validator("additional_preferences", mode="before")
    @classmethod
    def strip_additional(cls, v: object) -> Optional[str]:
        """Coerce empty/whitespace strings to None."""
        if v is None:
            return None
        stripped = str(v).strip()
        return stripped if stripped else None

    model_config = {"use_enum_values": True}


# ── Data model ────────────────────────────────────────────────────────────────


class RestaurantRecord(BaseModel):
    """
    Structured representation of a single restaurant row from the dataset.

    Used to pass candidates to the LLM prompt builder as typed objects
    rather than raw DataFrame rows.
    """

    name: str = Field(description="Restaurant display name.")
    location: str = Field(description="Area / neighbourhood.")
    cuisines: str = Field(description="Comma-separated, lowercased cuisine tags.")
    average_cost_for_two: int = Field(
        ge=0, description="Approximate cost in INR for two people."
    )
    aggregate_rating: float = Field(
        ge=0.0, le=5.0, description="Aggregate customer rating out of 5."
    )
    votes: int = Field(ge=0, description="Total number of customer votes/reviews.")
    has_online_delivery: bool = Field(description="Whether online delivery is available.")
    has_table_booking: bool = Field(description="Whether table booking is accepted.")
    budget: str = Field(description="Budget tier: low | medium | high.")

    model_config = {"from_attributes": True}


# ── LLM output models ─────────────────────────────────────────────────────────


class RecommendationItem(BaseModel):
    """
    A single restaurant recommendation produced by the LLM.

    The LLM is instructed to output a JSON array of these objects.
    """

    name: str = Field(description="Restaurant name exactly as it appears in the dataset.")
    location: str = Field(description="Neighbourhood / area.")
    cuisines: str = Field(description="Comma-separated cuisine tags.")
    aggregate_rating: float = Field(
        ge=0.0, le=5.0, description="Aggregate rating out of 5."
    )
    average_cost_for_two: int = Field(
        ge=0, description="Approximate cost in INR for two people."
    )
    has_online_delivery: bool = Field(description="Online delivery available.")
    has_table_booking: bool = Field(description="Table booking available.")
    explanation: str = Field(
        description=(
            "Human-friendly, 2-3 sentence explanation of why this restaurant "
            "matches the user's specific preferences."
        )
    )


class RecommendationResponse(BaseModel):
    """
    Top-level response object returned by the recommendation endpoint.

    Contains the ranked list of restaurants plus an optional AI summary
    paragraph that contextualises the selections.
    """

    recommendations: list[RecommendationItem] = Field(
        description="Ranked list of recommended restaurants (best match first).",
        max_length=10,
    )
    summary: Optional[str] = Field(
        default=None,
        description="One-paragraph AI summary of the overall recommendation set.",
    )
    fallback_used: bool = Field(
        default=False,
        description=(
            "True when strict filters returned no results and the engine "
            "automatically relaxed one or more constraints to find candidates."
        ),
    )
    fallback_note: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of which constraint was relaxed, if any.",
    )
