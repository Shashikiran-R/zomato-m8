"""
src/llm_client.py
─────────────────
Groq API client for the restaurant recommendation pipeline.

Responsibilities
----------------
1. Call the Groq chat-completions endpoint with the prompt from
   :mod:`src.prompt_builder`.
2. Parse and validate the JSON response into a
   :class:`~src.models.RecommendationResponse`.
3. Enforce rate limits (30 RPM, 1K RPD, 8K TPM, 200K TPD) via
   :mod:`src.rate_limiter` before every API call.
4. Retry on transient failures with exponential backoff.
5. Fall back to heuristic recommendations when the LLM is unavailable
   or rate limits are exhausted.

Public API
----------
    get_recommendations(prefs, candidates, settings) -> RecommendationResponse
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import pandas as pd
from groq import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    Groq,
    RateLimitError,
)
from pydantic import ValidationError

from src.filters import FilterResult
from src.models import (
    RecommendationItem,
    RecommendationResponse,
    UserPreferenceInput,
)
from src.prompt_builder import build_prompt
from src.rate_limiter import RateLimitExceeded, rate_limiter

logger = logging.getLogger(__name__)

# ── Retry configuration ───────────────────────────────────────────────────────

_RETRYABLE_EXCEPTIONS = (RateLimitError, APIConnectionError, APITimeoutError)
_INITIAL_BACKOFF: float = 1.0   # seconds
_BACKOFF_FACTOR:  float = 2.0   # exponential multiplier

# Rough token estimation: ~1 token per 4 characters (conservative).
_CHARS_PER_TOKEN: int = 4
_MAX_COMPLETION_TOKENS: int = 2048


# ── Fallback helpers ──────────────────────────────────────────────────────────


def _heuristic_fallback(
    candidates: pd.DataFrame,
    fallback_used: bool,
    fallback_note: str,
    error_reason: str,
) -> RecommendationResponse:
    """
    Build a :class:`RecommendationResponse` from the top candidates without
    calling the LLM.  Called when every retry attempt has failed.

    Each item gets a templated explanation based on its rating and cuisines.
    """
    logger.warning("Using heuristic fallback. Reason: %s", error_reason)
    items: list[RecommendationItem] = []

    for _, row in candidates.head(5).iterrows():
        name     = str(row.get("name", "Unknown"))
        rating   = float(row.get("aggregate_rating", 0.0))
        cuisines = str(row.get("cuisines", "various cuisines"))
        cost     = int(row.get("average_cost_for_two", 0))
        location = str(row.get("location", ""))

        explanation = (
            f"{name} is a top-rated restaurant in {location} serving {cuisines}. "
            f"It holds an impressive rating of {rating:.1f}/5 and costs approximately "
            f"₹{cost} for two people. A reliable choice based on community ratings."
        )
        items.append(
            RecommendationItem(
                name=name,
                location=location,
                cuisines=cuisines,
                aggregate_rating=rating,
                average_cost_for_two=cost,
                has_online_delivery=bool(row.get("has_online_delivery", False)),
                has_table_booking=bool(row.get("has_table_booking", False)),
                explanation=explanation,
            )
        )

    return RecommendationResponse(
        recommendations=items,
        summary=(
            f"AI recommendations are temporarily unavailable ({error_reason}). "
            "Showing the top-rated restaurants based on community ratings."
        ),
        fallback_used=True,
        fallback_note=(
            fallback_note + ("; LLM unavailable — heuristic fallback applied" if fallback_note
                             else "LLM unavailable — heuristic fallback applied")
        ),
    )


def _parse_llm_response(raw_content: str) -> RecommendationResponse:
    """
    Parse the raw JSON string from the LLM into a validated
    :class:`RecommendationResponse`.

    Raises
    ------
    ValueError
        When the JSON is malformed or fails Pydantic validation.
    """
    # Strip accidental markdown fences if the model adds them
    text = raw_content.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

    try:
        return RecommendationResponse(**data)
    except (ValidationError, TypeError) as exc:
        raise ValueError(f"LLM JSON did not match expected schema: {exc}") from exc


# ── Main client ───────────────────────────────────────────────────────────────


def get_recommendations(
    prefs: UserPreferenceInput,
    filter_result: FilterResult,
    *,
    groq_api_key: str,
    llm_model: str = "openai/gpt-oss-120b",
    max_retries: int = 3,
    timeout: float = 30.0,
) -> RecommendationResponse:
    """
    Call the Groq API to rank candidates and return a
    :class:`RecommendationResponse`.

    Parameters
    ----------
    prefs:
        The user's input preferences (used for prompt context).
    filter_result:
        Output of :func:`src.filters.shortlist`, containing the candidate
        DataFrame and any fallback metadata.
    groq_api_key:
        Groq API key string.
    llm_model:
        Groq model name to use.  Defaults to ``openai/gpt-oss-120b``.
    max_retries:
        Number of retry attempts for transient failures.
    timeout:
        Seconds to wait for a Groq API response.

    Returns
    -------
    RecommendationResponse
        Validated recommendations with per-item explanations and a summary.
        On total failure, returns a heuristic-based response with
        ``fallback_used=True``.
    """
    candidates = filter_result.candidates

    if candidates.empty:
        logger.warning("No candidates to rank; returning empty response.")
        return RecommendationResponse(
            recommendations=[],
            summary="No restaurants matched your criteria. Try broadening your search.",
            fallback_used=filter_result.fallback_used,
            fallback_note=filter_result.fallback_note or None,
        )

    system_prompt, user_prompt = build_prompt(prefs, candidates)
    client = Groq(api_key=groq_api_key, timeout=timeout)

    # Estimate prompt tokens for rate-limit pre-check
    prompt_chars = len(system_prompt) + len(user_prompt)
    estimated_tokens = (prompt_chars // _CHARS_PER_TOKEN) + _MAX_COMPLETION_TOKENS

    last_error: str = "unknown error"
    backoff = _INITIAL_BACKOFF
    # Some models (e.g. openai/gpt-oss-120b) don't support response_format=json_object.
    # We try with it first; if the API returns a 400 json_validate_failed we switch to
    # prompt-only JSON mode for all subsequent attempts.
    use_json_format: bool = True

    for attempt in range(1, max_retries + 1):
        try:
            # ── Rate-limit gate ───────────────────────────────────────────
            try:
                rate_limiter.acquire(estimated_tokens=estimated_tokens)
            except RateLimitExceeded as exc:
                last_error = str(exc)
                logger.warning("Rate limit exhausted: %s", last_error)
                break  # daily cap — fall through to heuristic fallback

            logger.debug(
                "Groq API call — attempt %d/%d, model=%s, json_mode=%s",
                attempt, max_retries, llm_model, use_json_format,
            )

            call_kwargs: dict = dict(
                model=llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=_MAX_COMPLETION_TOKENS,
            )
            if use_json_format:
                call_kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**call_kwargs)

            raw = response.choices[0].message.content or ""
            logger.debug("Groq raw response length: %d chars", len(raw))

            # Record actual token usage from the API response
            usage = getattr(response, "usage", None)
            if usage:
                actual_tokens = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
                rate_limiter.record_tokens(actual_tokens)
                logger.debug("Actual token usage: %d (prompt=%s, completion=%s)",
                             actual_tokens, usage.prompt_tokens, usage.completion_tokens)

            recommendation = _parse_llm_response(raw)

            # Merge filter-level fallback metadata with the LLM response
            recommendation.fallback_used = filter_result.fallback_used or recommendation.fallback_used
            if filter_result.fallback_note and not recommendation.fallback_note:
                recommendation.fallback_note = filter_result.fallback_note

            logger.info(
                "LLM returned %d recommendations (attempt %d/%d).",
                len(recommendation.recommendations), attempt, max_retries,
            )
            return recommendation

        except _RETRYABLE_EXCEPTIONS as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Transient Groq error on attempt %d/%d — backing off %.1f s. %s",
                attempt, max_retries, backoff, last_error,
            )
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= _BACKOFF_FACTOR
            continue

        except APIStatusError as exc:
            # 400 json_validate_failed means this model doesn't support response_format.
            # Disable json_object mode and retry immediately (don't count as a full retry).
            if exc.status_code == 400 and use_json_format:
                logger.warning(
                    "Model '%s' does not support response_format=json_object. "
                    "Retrying with prompt-only JSON mode.",
                    llm_model,
                )
                use_json_format = False
                continue  # retry same attempt without json_object
            # Other 4xx errors (bad model name, invalid key) — don't retry
            last_error = f"API error {exc.status_code}: {exc.message}"
            logger.error("Non-retryable Groq API error: %s", last_error)
            break

        except ValueError as exc:
            # JSON parse / Pydantic validation failure — retry with same prompt
            last_error = str(exc)
            logger.warning(
                "Parse failure on attempt %d/%d: %s",
                attempt, max_retries, last_error,
            )
            if attempt < max_retries:
                time.sleep(backoff)
                backoff *= _BACKOFF_FACTOR
            continue

        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Unexpected error calling Groq API: %s", last_error)
            break

    # ── All attempts exhausted ── fall back to heuristic rankings ────────────
    return _heuristic_fallback(
        candidates=candidates,
        fallback_used=filter_result.fallback_used,
        fallback_note=filter_result.fallback_note or "",
        error_reason=last_error,
    )
