"""
src/filters.py
──────────────
Deterministic filtering and shortlisting layer.

Takes the full restaurant DataFrame and a UserPreferenceInput, applies
sequential filters (location → cuisine → budget → rating), ranks
survivors by rating + votes, and returns the top-N candidates.

If strict filtering returns zero results a progressive fallback ladder
relaxes one constraint at a time until at least one candidate is found.

Public API
----------
    shortlist(df, prefs, max_candidates) -> FilterResult
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.models import BudgetTier, UserPreferenceInput

logger = logging.getLogger(__name__)

_DEFAULT_MAX: int = 15


# ── Result container ─────────────────────────────────────────────────────────


@dataclass
class FilterResult:
    """
    Returned by :func:.

    Attributes
    ----------
    candidates : pd.DataFrame
        Top-N filtered and ranked restaurant rows.
    fallback_used : bool
        True when at least one constraint was relaxed to find results.
    fallback_note : str
        Human-readable description of which constraint(s) were relaxed.
    """
    candidates: pd.DataFrame
    fallback_used: bool = False
    fallback_note: str = ""


# ── Individual filter helpers ────────────────────────────────────────────────


def _filter_location(df: pd.DataFrame, location: str | None) -> pd.DataFrame:
    """
    Retain rows whose location column contains *location* as a
    case-insensitive substring or shares at least one whitespace-delimited
    token with it.

    Returns the full DataFrame unchanged when *location* is None/empty.
    """
    if not location:
        return df

    needle = location.strip().lower()
    # Substring match on the normalised location string
    mask = df["location"].str.lower().str.contains(needle, regex=False, na=False)

    if mask.sum() == 0:
        # Fallback: token overlap (handles "Koramangala 5th Block" vs "Koramangala")
        tokens = set(needle.split())
        mask = df["location"].str.lower().apply(
            lambda loc: bool(tokens & set(str(loc).split()))
        )

    return df[mask]


def _filter_cuisine(df: pd.DataFrame, cuisine: str | None) -> pd.DataFrame:
    """
    Retain rows whose cuisines field contains *cuisine* as a
    case-insensitive substring.

    The cuisines column is already normalised to lowercase comma-separated
    tags, so a simple str.contains is sufficient.

    Returns the full DataFrame unchanged when *cuisine* is None/empty.
    """
    if not cuisine:
        return df

    needle = cuisine.strip().lower()
    mask = df["cuisines"].str.contains(needle, regex=False, na=False)
    return df[mask]


def _filter_budget(df: pd.DataFrame, budget: str | BudgetTier | None) -> pd.DataFrame:
    """
    Retain rows matching the requested budget tier.

    The budget column (low / medium / high) was computed during
    ingestion and maps directly to the :class: enum.

    Returns the full DataFrame unchanged when *budget* is None.
    """
    if not budget:
        return df

    tier = budget.value if isinstance(budget, BudgetTier) else str(budget).lower().strip()
    return df[df["budget"] == tier]


def _filter_rating(df: pd.DataFrame, min_rating: float) -> pd.DataFrame:
    """
    Retain rows where aggregate_rating >= min_rating.

    A *min_rating* of 0.0 effectively disables this filter.
    """
    if min_rating <= 0.0:
        return df
    return df[df["aggregate_rating"] >= min_rating]


def _rank_and_prune(df: pd.DataFrame, max_candidates: int) -> pd.DataFrame:
    """
    Sort by aggregate_rating (desc) then votes (desc) and return
    the top *max_candidates* rows, reset to a clean 0-based index.
    """
    return (
        df.sort_values(
            by=["aggregate_rating", "votes"],
            ascending=[False, False],
        )
        .drop_duplicates(subset=["name"], keep="first")
        .head(max_candidates)
        .reset_index(drop=True)
    )


# ── Fallback ladder ──────────────────────────────────────────────────────────


def _apply_fallback(
    df_full: pd.DataFrame,
    prefs: UserPreferenceInput,
    max_candidates: int,
) -> FilterResult:
    """
    Progressive fallback: relax one constraint at a time until at least one
    candidate is found.

    Ladder (in order of relaxation):
      1. Relax rating  → drop rating filter.
      2. Relax cuisine  → remove cuisine filter.
      3. Relax budget   → remove budget filter.
      4. Relax location  → remove location filter (return top-N globally).
    """
    relaxations: list[str] = []

    # ── Step 1: relax rating ─────────────────────────────────────────────────
    attempt = _filter_location(df_full, prefs.location)
    attempt = _filter_cuisine(attempt, prefs.cuisine)
    attempt = _filter_budget(attempt, prefs.budget)
    if len(attempt) > 0:
        relaxations.append("minimum rating constraint removed")
        return FilterResult(
            candidates=_rank_and_prune(attempt, max_candidates),
            fallback_used=True,
            fallback_note="; ".join(relaxations),
        )

    # ── Step 2: relax cuisine ────────────────────────────────────────────────
    attempt = _filter_location(df_full, prefs.location)
    attempt = _filter_budget(attempt, prefs.budget)
    if len(attempt) > 0:
        relaxations.append("minimum rating constraint removed")
        relaxations.append("cuisine filter broadened")
        return FilterResult(
            candidates=_rank_and_prune(attempt, max_candidates),
            fallback_used=True,
            fallback_note="; ".join(relaxations),
        )

    # ── Step 3: relax budget ─────────────────────────────────────────────────
    attempt = _filter_location(df_full, prefs.location)
    if len(attempt) > 0:
        relaxations.append("minimum rating constraint removed")
        relaxations.append("cuisine filter broadened")
        relaxations.append("budget tier filter removed")
        return FilterResult(
            candidates=_rank_and_prune(attempt, max_candidates),
            fallback_used=True,
            fallback_note="; ".join(relaxations),
        )

    # ── Step 4: relax location (last resort — return global top-N) ───────────
    relaxations.append("minimum rating constraint removed")
    relaxations.append("cuisine filter broadened")
    relaxations.append("budget tier filter removed")
    relaxations.append("location filter removed (showing top results from all areas)")
    return FilterResult(
        candidates=_rank_and_prune(df_full, max_candidates),
        fallback_used=True,
        fallback_note="; ".join(relaxations),
    )


# ── Public entry point ───────────────────────────────────────────────────────


def shortlist(
    df: pd.DataFrame,
    prefs: UserPreferenceInput,
    max_candidates: int = _DEFAULT_MAX,
) -> FilterResult:
    """
    Apply all active filters, rank survivors, and return the top-N candidates.

    Parameters
    ----------
    df:
        The full restaurants DataFrame (output of :func:).
    prefs:
        User preference constraints. None fields skip the corresponding filter.
    max_candidates:
        Maximum number of candidates to return. Defaults to 15.

    Returns
    -------
    FilterResult
        A dataclass with .candidates (DataFrame), .fallback_used
        (bool), and .fallback_note (str).

    Notes
    -----
    Filter application order matters for performance:
        location  →  cuisine  →  budget  →  rating
    Location typically shrinks the pool the most aggressively.
    """
    logger.debug(
        "shortlist called: location=%r cuisine=%r budget=%r min_rating=%s max_candidates=%d",
        prefs.location, prefs.cuisine, prefs.budget, prefs.min_rating, max_candidates,
    )

    # ── Strict filtering ─────────────────────────────────────────────────────
    filtered = _filter_location(df, prefs.location)
    logger.debug("After location filter: %d rows", len(filtered))

    filtered = _filter_cuisine(filtered, prefs.cuisine)
    logger.debug("After cuisine filter: %d rows", len(filtered))

    filtered = _filter_budget(filtered, prefs.budget)
    logger.debug("After budget filter: %d rows", len(filtered))

    filtered = _filter_rating(filtered, prefs.min_rating)
    logger.debug("After rating filter: %d rows", len(filtered))

    if len(filtered) == 0:
        logger.info(
            "Strict filters returned 0 results; engaging fallback ladder. "
            "(location=%r, cuisine=%r, budget=%r, min_rating=%s)",
            prefs.location, prefs.cuisine, prefs.budget, prefs.min_rating,
        )
        return _apply_fallback(df, prefs, max_candidates)

    candidates = _rank_and_prune(filtered, max_candidates)
    logger.info(
        "Shortlisted %d candidates from %d matches.",
        len(candidates), len(filtered),
    )
    return FilterResult(candidates=candidates)
