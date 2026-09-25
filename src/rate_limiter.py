"""
src/rate_limiter.py
───────────────────
Thread-safe, sliding-window rate limiter for the Groq LLM API.

Enforces the following limits for openai/gpt-oss-120b:
    - 30  requests  / minute
    - 1 000  requests  / day
    - 8 000  tokens   / minute
    - 200 000  tokens   / day

The limiter tracks timestamps of past requests and cumulative token counts
within each window.  Before every API call the caller invokes
:meth:`RateLimiter.acquire` which either:
    • returns immediately (budget available),
    • sleeps until the oldest entry expires and budget opens up, or
    • raises :class:`RateLimitExceeded` when the *daily* cap is hit
      (sleeping would be unreasonably long).

After a successful API response the caller reports actual token usage via
:meth:`RateLimiter.record_tokens` so that the token-per-minute /
token-per-day windows stay accurate.

Public API
----------
    rate_limiter          Module-level singleton instance.
    RateLimitExceeded     Exception raised when daily limits are exhausted.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Exception ────────────────────────────────────────────────────────────────


class RateLimitExceeded(Exception):
    """Raised when a rate-limit window is exhausted and waiting is impractical."""


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RateLimits:
    """Default limits for openai/gpt-oss-120b on Groq."""
    requests_per_minute: int = 30
    requests_per_day: int = 1_000
    tokens_per_minute: int = 8_000
    tokens_per_day: int = 200_000


# ── Limiter ──────────────────────────────────────────────────────────────────


class RateLimiter:
    """
    Thread-safe sliding-window rate limiter.

    Maintains two deques of ``(timestamp, token_count)`` entries — one for the
    1-minute window and one for the 24-hour window — and enforces both request
    count and token count limits within each window.
    """

    _MINUTE: float = 60.0
    _DAY: float = 86_400.0

    def __init__(self, limits: RateLimits | None = None) -> None:
        self._limits = limits or RateLimits()
        self._lock = threading.Lock()

        # Each entry is (timestamp, token_count).
        # token_count is 0 at acquire-time and updated via record_tokens().
        self._minute_window: deque[list] = deque()   # mutable lists so we can patch token counts
        self._day_window: deque[list] = deque()

    # ── Internal helpers ─────────────────────────────────────────────────

    def _purge(self, now: float) -> None:
        """Remove entries that have fallen outside their respective windows."""
        minute_cutoff = now - self._MINUTE
        day_cutoff = now - self._DAY

        while self._minute_window and self._minute_window[0][0] < minute_cutoff:
            self._minute_window.popleft()
        while self._day_window and self._day_window[0][0] < day_cutoff:
            self._day_window.popleft()

    def _minute_requests(self) -> int:
        return len(self._minute_window)

    def _day_requests(self) -> int:
        return len(self._day_window)

    def _minute_tokens(self) -> int:
        return sum(entry[1] for entry in self._minute_window)

    def _day_tokens(self) -> int:
        return sum(entry[1] for entry in self._day_window)

    # ── Public API ───────────────────────────────────────────────────────

    def acquire(self, estimated_tokens: int = 0) -> None:
        """
        Block until budget is available in all windows, or raise
        :class:`RateLimitExceeded` if the daily cap is hit.

        Parameters
        ----------
        estimated_tokens:
            Rough token estimate for the upcoming request (prompt + max
            completion).  Used to pre-check the token budget.  The actual
            count is recorded later via :meth:`record_tokens`.
        """
        with self._lock:
            now = time.time()
            self._purge(now)

            # ── Daily caps (non-waitable — would sleep for hours) ────────
            if self._day_requests() >= self._limits.requests_per_day:
                raise RateLimitExceeded(
                    f"Daily request limit reached ({self._limits.requests_per_day}/day). "
                    "Try again tomorrow."
                )
            if self._day_tokens() + estimated_tokens > self._limits.tokens_per_day:
                raise RateLimitExceeded(
                    f"Daily token limit approaching ({self._day_tokens()}/{self._limits.tokens_per_day} "
                    f"used, need ~{estimated_tokens} more). Try again tomorrow."
                )

            # ── Per-minute request cap (waitable) ────────────────────────
            if self._minute_requests() >= self._limits.requests_per_minute:
                oldest_ts = self._minute_window[0][0]
                wait = (oldest_ts + self._MINUTE) - now + 0.1  # small buffer
                if wait > 0:
                    logger.info(
                        "Rate limit: %d/%d RPM used. Sleeping %.1f s.",
                        self._minute_requests(),
                        self._limits.requests_per_minute,
                        wait,
                    )
                    self._lock.release()
                    time.sleep(wait)
                    self._lock.acquire()
                    now = time.time()
                    self._purge(now)

            # ── Per-minute token cap (waitable) ──────────────────────────
            if self._minute_tokens() + estimated_tokens > self._limits.tokens_per_minute:
                oldest_ts = self._minute_window[0][0] if self._minute_window else now
                wait = (oldest_ts + self._MINUTE) - now + 0.1
                if wait > 0:
                    logger.info(
                        "Rate limit: %d/%d TPM used, need ~%d. Sleeping %.1f s.",
                        self._minute_tokens(),
                        self._limits.tokens_per_minute,
                        estimated_tokens,
                        wait,
                    )
                    self._lock.release()
                    time.sleep(wait)
                    self._lock.acquire()
                    now = time.time()
                    self._purge(now)

            # ── Record the request (tokens updated later) ────────────────
            entry = [now, 0]  # mutable so record_tokens() can patch
            self._minute_window.append(entry)
            self._day_window.append(entry)

            logger.debug(
                "Rate limiter: acquired slot. RPM=%d/%d, RPD=%d/%d, TPM=%d/%d, TPD=%d/%d",
                self._minute_requests(), self._limits.requests_per_minute,
                self._day_requests(), self._limits.requests_per_day,
                self._minute_tokens(), self._limits.tokens_per_minute,
                self._day_tokens(), self._limits.tokens_per_day,
            )

    def record_tokens(self, actual_tokens: int) -> None:
        """
        Update the most recent entry with the actual token count reported
        by the API response.

        Call this immediately after a successful API response.
        """
        with self._lock:
            if self._minute_window:
                # Both deques share the same list object for the latest entry
                self._minute_window[-1][1] = actual_tokens
            logger.debug(
                "Rate limiter: recorded %d tokens. TPM=%d/%d, TPD=%d/%d",
                actual_tokens,
                self._minute_tokens(), self._limits.tokens_per_minute,
                self._day_tokens(), self._limits.tokens_per_day,
            )

    def status(self) -> dict:
        """Return a snapshot of current usage for observability."""
        with self._lock:
            self._purge(time.time())
            return {
                "requests_per_minute": f"{self._minute_requests()}/{self._limits.requests_per_minute}",
                "requests_per_day": f"{self._day_requests()}/{self._limits.requests_per_day}",
                "tokens_per_minute": f"{self._minute_tokens()}/{self._limits.tokens_per_minute}",
                "tokens_per_day": f"{self._day_tokens()}/{self._limits.tokens_per_day}",
            }


# ── Module-level singleton ───────────────────────────────────────────────────

rate_limiter = RateLimiter()
