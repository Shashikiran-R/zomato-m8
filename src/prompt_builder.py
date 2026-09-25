"""
src/prompt_builder.py
─────────────────────
Constructs the system + user prompt that is sent to the Groq LLM.

The prompt gives the model:
  1. A system role with strict JSON output instructions.
  2. A summary of the user's preferences.
  3. A compact table of the filtered candidate restaurants.

The model is asked to return a JSON object with two keys:
  - "recommendations": list[RecommendationItem]   (up to 5)
  - "summary":         str                        (one-paragraph overview)

Public API
----------
    build_prompt(prefs, candidates) -> tuple[str, str]
        Returns (system_prompt, user_prompt).
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import pandas as pd

from src.models import UserPreferenceInput

logger = logging.getLogger(__name__)

# ── JSON schema that the LLM must conform to ─────────────────────────────────

_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "required": ["recommendations", "summary"],
    "properties": {
        "recommendations": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "required": [
                    "name", "location", "cuisines",
                    "aggregate_rating", "average_cost_for_two",
                    "has_online_delivery", "has_table_booking",
                    "explanation",
                ],
                "properties": {
                    "name":                 {"type": "string"},
                    "location":             {"type": "string"},
                    "cuisines":             {"type": "string"},
                    "aggregate_rating":     {"type": "number", "minimum": 0, "maximum": 5},
                    "average_cost_for_two": {"type": "integer", "minimum": 0},
                    "has_online_delivery":  {"type": "boolean"},
                    "has_table_booking":    {"type": "boolean"},
                    "explanation":          {"type": "string"},
                },
            },
        },
        "summary": {"type": "string"},
    },
}

_SCHEMA_STR: str = json.dumps(_OUTPUT_SCHEMA, indent=2)


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT: str = f"""You are an expert restaurant recommendation assistant with deep knowledge of dining experiences and local cuisine.

## Your Task
Analyse the candidate restaurants provided by the user and select the **top 5** that best match the stated preferences. Return your response as a **single valid JSON object** — no markdown, no code fences, no extra commentary.

## Response Schema
{_SCHEMA_STR}

## Ranking Guidelines
- Prioritise restaurants that closely match the user's cuisine preference and budget.
- Among equal matches, favour higher `aggregate_rating` then higher `votes`.
- Write each `explanation` in 2-3 warm, personable sentences that clearly state *why* the restaurant is a good fit for *this specific user*.
- The `summary` should be 1 paragraph (3-5 sentences) synthesising the overall set of recommendations and mentioning any notable highlights.

## Strict Rules
- Output ONLY the JSON object — nothing else.
- Do NOT invent or modify restaurant data; use values exactly as provided.
- If fewer than 5 candidates are supplied, return all of them.
- Preserve the exact `name` and `location` strings from the candidate table.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _format_preferences(prefs: UserPreferenceInput) -> str:
    """Render the user's preferences as a neat bullet list."""
    lines: list[str] = []
    if prefs.location:
        lines.append(f"- Location      : {prefs.location}")
    if prefs.budget:
        tier = prefs.budget if isinstance(prefs.budget, str) else prefs.budget.value
        cost_hint = {"low": "< ₹500 for two", "medium": "₹500–₹1500 for two", "high": "> ₹1500 for two"}.get(tier, tier)
        lines.append(f"- Budget        : {tier.capitalize()} ({cost_hint})")
    if prefs.cuisine:
        lines.append(f"- Cuisine       : {prefs.cuisine}")
    if prefs.min_rating and prefs.min_rating > 0:
        lines.append(f"- Min rating    : {prefs.min_rating:.1f} / 5.0")
    if prefs.additional_preferences:
        lines.append(f"- Special notes : {prefs.additional_preferences}")
    return "\n".join(lines) if lines else "- No specific constraints (surprise me!)"


def _format_candidates_table(candidates: pd.DataFrame) -> str:
    """
    Render the candidate restaurants as a compact markdown-style table
    so the LLM can easily scan all candidates in one pass.
    """
    rows: list[str] = []
    header = (
        "| # | Name | Location | Cuisines | Rating | Cost/2 | "
        "Online Del. | Table Booking |"
    )
    sep = "|---|---|---|---|---|---|---|---|"
    rows.append(header)
    rows.append(sep)

    for i, (_, row) in enumerate(candidates.iterrows(), start=1):
        name     = str(row.get("name", ""))
        location = str(row.get("location", ""))
        cuisines = str(row.get("cuisines", ""))
        rating   = f"{float(row.get('aggregate_rating', 0)):.1f}"
        cost     = f"₹{int(row.get('average_cost_for_two', 0))}"
        online   = "Yes" if row.get("has_online_delivery") else "No"
        booking  = "Yes" if row.get("has_table_booking") else "No"
        rows.append(f"| {i} | {name} | {location} | {cuisines} | {rating} | {cost} | {online} | {booking} |")

    return "\n".join(rows)


# ── Public API ─────────────────────────────────────────────────────────────────


def build_prompt(
    prefs: UserPreferenceInput,
    candidates: pd.DataFrame,
) -> tuple[str, str]:
    """
    Build the (system_prompt, user_prompt) pair for the Groq chat completion.

    Parameters
    ----------
    prefs:
        The user's filtering preferences (from the API / UI).
    candidates:
        Filtered and ranked candidate restaurants (output of :func:`src.filters.shortlist`).

    Returns
    -------
    tuple[str, str]
        ``(system_prompt, user_prompt)`` ready to be passed to the Groq client.
    """
    pref_block   = _format_preferences(prefs)
    table_block  = _format_candidates_table(candidates)

    user_prompt = f"""## User Preferences
{pref_block}

## Candidate Restaurants ({len(candidates)} options)
{table_block}

Please select the best 5 and return the JSON object as instructed."""

    logger.debug(
        "Prompt built: %d candidates, user_prompt length=%d chars",
        len(candidates), len(user_prompt),
    )
    return _SYSTEM_PROMPT, user_prompt
