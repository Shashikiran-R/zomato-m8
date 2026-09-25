"""
tests/conftest.py
──────────────────
Shared fixtures and configuration for the test suite.

Ensures:
  - The rate limiter is reset between tests to avoid cross-contamination.
  - Common DataFrame fixtures are available to all test modules.
"""

from __future__ import annotations

import pytest

from src.rate_limiter import RateLimiter, rate_limiter


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """Reset the module-level rate limiter before each test to ensure isolation."""
    rate_limiter._minute_window.clear()
    rate_limiter._day_window.clear()
    yield
    rate_limiter._minute_window.clear()
    rate_limiter._day_window.clear()
