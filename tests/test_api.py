"""
tests/test_api.py
─────────────────
Integration tests for the FastAPI backend (api/main.py).

Uses FastAPI's TestClient to test all endpoints without starting a real server.
All LLM calls are mocked — no real API key is needed.

Covers:
  - GET /health: readiness probe with dataset loaded/unloaded
  - GET /locations: distinct sorted location list
  - GET /cuisines: distinct sorted cuisine tags
  - POST /recommend: full recommendation pipeline (happy path)
  - POST /recommend: 404 when no restaurants match
  - POST /recommend: 422 for invalid input schema
  - POST /recommend: 500 on LLM upstream failure
  - Edge cases: special characters in additional_preferences, empty body
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

# We need to mock settings and data loading BEFORE importing the app,
# so we set up the environment with a monkeypatch-compatible approach.


# ── Sample data ──────────────────────────────────────────────────────────────


def _make_sample_df() -> pd.DataFrame:
    """Create a small test DataFrame matching the processed Parquet schema."""
    return pd.DataFrame(
        {
            "name": [
                "Spice Garden",
                "Pasta Palace",
                "Udupi Corner",
                "Beijing Bites",
                "Tandoor Express",
            ],
            "location": [
                "Koramangala",
                "Koramangala",
                "Indiranagar",
                "Indiranagar",
                "Whitefield",
            ],
            "cuisines": [
                "north indian, chinese",
                "italian, continental",
                "south indian",
                "chinese, thai",
                "north indian, mughlai",
            ],
            "average_cost_for_two": [600, 1200, 250, 900, 400],
            "aggregate_rating": [4.2, 3.9, 4.5, 3.5, 4.1],
            "votes": [500, 120, 800, 200, 350],
            "has_online_delivery": [True, False, True, True, True],
            "has_table_booking": [True, True, False, False, True],
            "budget": ["medium", "medium", "low", "medium", "low"],
        }
    )


def _make_valid_llm_json(candidates: pd.DataFrame) -> str:
    """Build a valid LLM response JSON from the first few candidates."""
    recs = []
    for _, row in candidates.head(3).iterrows():
        recs.append(
            {
                "name": str(row["name"]),
                "location": str(row["location"]),
                "cuisines": str(row["cuisines"]),
                "aggregate_rating": float(row["aggregate_rating"]),
                "average_cost_for_two": int(row["average_cost_for_two"]),
                "has_online_delivery": bool(row["has_online_delivery"]),
                "has_table_booking": bool(row["has_table_booking"]),
                "explanation": f"{row['name']} is an excellent choice.",
            }
        )
    return json.dumps({"recommendations": recs, "summary": "Great options for you!"})


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_df():
    return _make_sample_df()


@pytest.fixture()
def client(sample_df):
    """
    Create a TestClient with the app state pre-populated (bypassing lifespan).
    This avoids needing a real Parquet file on disk.
    """
    from contextlib import asynccontextmanager
    from api.main import app, app_state

    # Replace the real lifespan with a no-op so it doesn't read from disk
    orig_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def _noop_lifespan(a):
        yield

    app.router.lifespan_context = _noop_lifespan

    # Manually set app state as the lifespan would
    app_state.df = sample_df

    raw_locs = sample_df["location"].dropna().unique()
    app_state.locations = sorted(
        list(set(str(loc).strip() for loc in raw_locs if str(loc).strip()))
    )

    all_cuisines = set()
    for c_list in sample_df["cuisines"].dropna():
        tags = [c.strip() for c in c_list.split(",")]
        all_cuisines.update(tags)
    app_state.cuisines = sorted(list(c for c in all_cuisines if c))

    with TestClient(app, raise_server_exceptions=False) as tc:
        yield tc

    # Cleanup
    app_state.df = None
    app_state.locations = []
    app_state.cuisines = []
    app.router.lifespan_context = orig_lifespan


# ── GET /health ──────────────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_ok_when_data_loaded(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["dataset_loaded"] is True

    def test_health_degraded_when_no_data(self):
        from contextlib import asynccontextmanager
        from api.main import app, app_state

        # Save original state and lifespan
        orig_df = app_state.df
        orig_locs = app_state.locations
        orig_cuisines = app_state.cuisines
        orig_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def _noop_lifespan(a):
            yield

        try:
            app_state.df = pd.DataFrame()  # empty → degraded
            app_state.locations = []
            app_state.cuisines = []
            app.router.lifespan_context = _noop_lifespan

            with TestClient(app, raise_server_exceptions=False) as tc:
                resp = tc.get("/health")
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "degraded"
                assert data["dataset_loaded"] is False
        finally:
            app_state.df = orig_df
            app_state.locations = orig_locs
            app_state.cuisines = orig_cuisines
            app.router.lifespan_context = orig_lifespan

    def test_health_contains_rate_limits(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "rate_limits" in data
        assert "requests_per_minute" in data["rate_limits"]


# ── GET /locations ───────────────────────────────────────────────────────────


class TestLocationsEndpoint:
    def test_returns_list(self, client):
        resp = client.get("/locations")
        assert resp.status_code == 200
        locs = resp.json()
        assert isinstance(locs, list)
        assert len(locs) > 0

    def test_locations_sorted(self, client):
        resp = client.get("/locations")
        locs = resp.json()
        assert locs == sorted(locs)

    def test_locations_are_distinct(self, client):
        resp = client.get("/locations")
        locs = resp.json()
        assert len(locs) == len(set(locs))

    def test_expected_locations_present(self, client):
        resp = client.get("/locations")
        locs = resp.json()
        assert "Koramangala" in locs
        assert "Indiranagar" in locs
        assert "Whitefield" in locs


# ── GET /cuisines ────────────────────────────────────────────────────────────


class TestCuisinesEndpoint:
    def test_returns_list(self, client):
        resp = client.get("/cuisines")
        assert resp.status_code == 200
        cuisines = resp.json()
        assert isinstance(cuisines, list)
        assert len(cuisines) > 0

    def test_cuisines_sorted(self, client):
        resp = client.get("/cuisines")
        cuisines = resp.json()
        assert cuisines == sorted(cuisines)

    def test_cuisines_are_distinct(self, client):
        resp = client.get("/cuisines")
        cuisines = resp.json()
        assert len(cuisines) == len(set(cuisines))

    def test_expected_cuisines_present(self, client):
        resp = client.get("/cuisines")
        cuisines = resp.json()
        assert "italian" in cuisines or "Italian" in cuisines or any(
            "italian" in c.lower() for c in cuisines
        )


# ── POST /recommend ──────────────────────────────────────────────────────────


class TestRecommendEndpoint:
    def _mock_groq_response(self, json_str: str) -> MagicMock:
        """Build a mock Groq ChatCompletion response."""
        msg = MagicMock()
        msg.content = json_str
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = None
        return resp

    def test_happy_path(self, client, sample_df):
        llm_json = _make_valid_llm_json(sample_df)
        mock_resp = self._mock_groq_response(llm_json)

        with patch("src.llm_client.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = mock_resp
            resp = client.post(
                "/recommend",
                json={
                    "location": "Koramangala",
                    "budget": "medium",
                    "cuisine": "italian",
                    "min_rating": 3.5,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "recommendations" in data
        assert len(data["recommendations"]) >= 1

    def test_minimal_body(self, client, sample_df):
        """POST with empty body (all defaults) should still work."""
        llm_json = _make_valid_llm_json(sample_df)
        mock_resp = self._mock_groq_response(llm_json)

        with patch("src.llm_client.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = mock_resp
            resp = client.post("/recommend", json={})

        assert resp.status_code == 200

    def test_with_additional_preferences(self, client, sample_df):
        """Free-text additional preferences should not break the endpoint."""
        llm_json = _make_valid_llm_json(sample_df)
        mock_resp = self._mock_groq_response(llm_json)

        with patch("src.llm_client.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = mock_resp
            resp = client.post(
                "/recommend",
                json={
                    "location": "Koramangala",
                    "additional_preferences": "romantic & candle-lit 🕯️",
                },
            )

        assert resp.status_code == 200

    def test_special_characters_in_free_text(self, client, sample_df):
        """Special chars like <, >, &, quotes should not cause errors."""
        llm_json = _make_valid_llm_json(sample_df)
        mock_resp = self._mock_groq_response(llm_json)

        with patch("src.llm_client.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = mock_resp
            resp = client.post(
                "/recommend",
                json={
                    "additional_preferences": '<script>alert("xss")</script> & "quotes"',
                },
            )

        assert resp.status_code == 200

    def test_response_schema(self, client, sample_df):
        """Response should match the RecommendationResponse schema."""
        llm_json = _make_valid_llm_json(sample_df)
        mock_resp = self._mock_groq_response(llm_json)

        with patch("src.llm_client.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = mock_resp
            resp = client.post(
                "/recommend",
                json={"location": "Koramangala"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "recommendations" in data
        assert "summary" in data
        assert "fallback_used" in data
        assert "fallback_note" in data

        # Each recommendation should have required fields
        for rec in data["recommendations"]:
            assert "name" in rec
            assert "location" in rec
            assert "cuisines" in rec
            assert "aggregate_rating" in rec
            assert "average_cost_for_two" in rec
            assert "explanation" in rec


# ── POST /recommend — error paths ────────────────────────────────────────────


class TestRecommendErrors:
    def test_422_invalid_budget(self, client):
        """Invalid budget tier should return 422."""
        resp = client.post(
            "/recommend",
            json={"budget": "ultra_premium"},
        )
        assert resp.status_code == 422

    def test_422_invalid_rating_too_high(self, client):
        """Rating above 5.0 should return 422."""
        resp = client.post(
            "/recommend",
            json={"min_rating": 10.0},
        )
        assert resp.status_code == 422

    def test_422_invalid_rating_negative(self, client):
        """Negative rating should return 422."""
        resp = client.post(
            "/recommend",
            json={"min_rating": -1.0},
        )
        assert resp.status_code == 422

    def test_500_when_dataset_unavailable(self):
        """If dataset is not loaded, /recommend should return 500."""
        from contextlib import asynccontextmanager
        from api.main import app, app_state

        orig_df = app_state.df
        orig_locs = app_state.locations
        orig_cuisines = app_state.cuisines
        orig_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def _noop_lifespan(a):
            yield

        try:
            app_state.df = None
            app_state.locations = []
            app_state.cuisines = []
            app.router.lifespan_context = _noop_lifespan

            with TestClient(app, raise_server_exceptions=False) as tc:
                resp = tc.post(
                    "/recommend",
                    json={"location": "Koramangala"},
                )
                assert resp.status_code == 500
                assert "unavailable" in resp.json()["detail"].lower()
        finally:
            app_state.df = orig_df
            app_state.locations = orig_locs
            app_state.cuisines = orig_cuisines
            app.router.lifespan_context = orig_lifespan

    def test_llm_timeout_returns_fallback(self, client, sample_df):
        """When the LLM times out, the API should still return a response
        (heuristic fallback) rather than a 500 error.
        """
        import httpx
        from groq import APIConnectionError

        mock_request = httpx.Request("POST", "https://api.groq.com/")

        with patch("src.llm_client.Groq") as MockGroq:
            with patch("src.llm_client.time.sleep"):
                MockGroq.return_value.chat.completions.create.side_effect = (
                    APIConnectionError(
                        message="Connection timed out", request=mock_request
                    )
                )
                resp = client.post(
                    "/recommend",
                    json={"location": "Koramangala"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["fallback_used"] is True
        assert len(data["recommendations"]) > 0


# ── POST /recommend — edge cases ─────────────────────────────────────────────


class TestRecommendEdgeCases:
    def _mock_groq_response(self, json_str: str) -> MagicMock:
        msg = MagicMock()
        msg.content = json_str
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        resp.usage = None
        return resp

    def test_no_match_extreme_filters(self, client):
        """Extreme filter criteria that match nothing — should use fallback
        (since the fallback ladder eventually returns global top-N).
        """
        # The fallback ladder in filters.py always returns something from
        # the global top-N, so we expect a 200 with fallback_used=True.
        df = _make_sample_df()
        llm_json = _make_valid_llm_json(df)
        mock_resp = self._mock_groq_response(llm_json)

        with patch("src.llm_client.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = mock_resp
            resp = client.post(
                "/recommend",
                json={
                    "location": "Atlantis",
                    "budget": "high",
                    "cuisine": "martian",
                    "min_rating": 5.0,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["fallback_used"] is True

    def test_whitespace_only_location(self, client, sample_df):
        """Whitespace-only location should be treated as None (no filter)."""
        llm_json = _make_valid_llm_json(sample_df)
        mock_resp = self._mock_groq_response(llm_json)

        with patch("src.llm_client.Groq") as MockGroq:
            MockGroq.return_value.chat.completions.create.return_value = mock_resp
            resp = client.post(
                "/recommend",
                json={"location": "   "},
            )

        assert resp.status_code == 200
