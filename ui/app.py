"""
ui/app.py
─────────
Streamlit interactive frontend for the AI-Powered Restaurant Recommendation
System.  Communicates with the FastAPI backend to fetch metadata and request
AI-driven recommendations.

Run:
    streamlit run ui/app.py --server.port 8501
"""

import os
import sys

import httpx
import streamlit as st

# ── Resolve API base URL ─────────────────────────────────────────────────────
# Try to load from config; fall back to env var or hard default.
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NextLeap Eats",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Global ────────────────────────────────────────────────────────── */
    .main .block-container { padding-top: 2rem; }

    /* ── Summary banner ────────────────────────────────────────────────── */
    .ai-summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #ffffff;
        padding: 1.2rem 1.6rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        font-size: 1.05rem;
        line-height: 1.6;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.35);
    }

    /* ── Restaurant card ───────────────────────────────────────────────── */
    .restaurant-card {
        background: #ffffff;
        border: 1px solid #e8ecf1;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .restaurant-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.10);
    }
    .restaurant-card h3 {
        margin: 0 0 0.3rem 0;
        color: #1a1a2e;
        font-size: 1.25rem;
    }
    .restaurant-card .meta {
        color: #5a6275;
        font-size: 0.9rem;
        margin-bottom: 0.6rem;
    }
    .restaurant-card .explanation {
        color: #374151;
        font-size: 0.95rem;
        line-height: 1.55;
        border-left: 3px solid #667eea;
        padding-left: 0.8rem;
        margin-top: 0.5rem;
    }

    /* ── Cuisine badge ─────────────────────────────────────────────────── */
    .cuisine-badge {
        display: inline-block;
        background: #eef2ff;
        color: #4338ca;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 4px;
        margin-top: 4px;
    }

    /* ── Feature pills ─────────────────────────────────────────────────── */
    .feature-pill {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 500;
        margin-right: 4px;
        margin-top: 4px;
    }
    .feature-pill.yes {
        background: #dcfce7;
        color: #166534;
    }
    .feature-pill.no {
        background: #fee2e2;
        color: #991b1b;
    }

    /* ── Fallback alert ────────────────────────────────────────────────── */
    .fallback-alert {
        background: #fef3c7;
        border: 1px solid #f59e0b;
        color: #92400e;
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        font-size: 0.92rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper: fetch metadata from backend ──────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def _fetch_metadata(endpoint: str) -> list[str]:
    """Fetch a list of strings from a backend metadata endpoint."""
    try:
        resp = httpx.get(f"{API_BASE_URL}{endpoint}", timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def _fetch_locations() -> list[str]:
    return _fetch_metadata("/locations")


def _fetch_cuisines() -> list[str]:
    return _fetch_metadata("/cuisines")


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.image(
    "https://img.icons8.com/color/96/restaurant.png",
    width=64,
)
st.sidebar.title("🍽️ Preferences")
st.sidebar.caption("Tell us what you're looking for and our AI will find the best matches.")

# Location
locations = _fetch_locations()
if locations:
    location_choice = st.sidebar.selectbox(
        "📍 Location",
        options=["Any"] + locations,
        index=0,
        help="Choose a neighbourhood or area.",
    )
    selected_location = None if location_choice == "Any" else location_choice
else:
    selected_location = st.sidebar.text_input(
        "📍 Location",
        placeholder="e.g. Koramangala",
        help="Type a location (backend unreachable for dropdown data).",
    ) or None

# Budget
budget_choice = st.sidebar.radio(
    "💰 Budget",
    options=["Any", "Low (< ₹500)", "Medium (₹500–₹1500)", "High (> ₹1500)"],
    index=0,
    horizontal=True,
)
_budget_map = {
    "Any": None,
    "Low (< ₹500)": "low",
    "Medium (₹500–₹1500)": "medium",
    "High (> ₹1500)": "high",
}
selected_budget = _budget_map[budget_choice]

# Cuisine
cuisines = _fetch_cuisines()
if cuisines:
    cuisine_choice = st.sidebar.selectbox(
        "🍜 Cuisine",
        options=["Any"] + cuisines,
        index=0,
        help="Pick a cuisine type.",
    )
    selected_cuisine = None if cuisine_choice == "Any" else cuisine_choice
else:
    selected_cuisine = st.sidebar.text_input(
        "🍜 Cuisine",
        placeholder="e.g. Italian",
        help="Type a cuisine (backend unreachable for dropdown data).",
    ) or None

# Minimum rating slider
selected_rating = st.sidebar.slider(
    "⭐ Minimum Rating",
    min_value=0.0,
    max_value=5.0,
    value=0.0,
    step=0.5,
    help="Filter restaurants below this aggregate rating.",
)

# Additional preferences
additional_prefs = st.sidebar.text_area(
    "📝 Additional Notes",
    placeholder="e.g. romantic dinner, pet-friendly, quick bites …",
    max_chars=500,
    height=80,
)

# Action button
find_button = st.sidebar.button(
    "🔍 Find Recommendations",
    use_container_width=True,
    type="primary",
)


# ── Main Display ─────────────────────────────────────────────────────────────

st.title("🍽️ NextLeap Eats")
st.markdown("Discover your next favourite restaurant — powered by AI.")

if find_button:
    # Build request payload
    payload = {
        "location": selected_location,
        "budget": selected_budget,
        "cuisine": selected_cuisine,
        "min_rating": selected_rating,
        "additional_preferences": additional_prefs.strip() if additional_prefs else None,
    }

    with st.spinner("🤖  Our AI is analysing restaurants for you …"):
        try:
            resp = httpx.post(
                f"{API_BASE_URL}/recommend",
                json=payload,
                timeout=60.0,
            )

            if resp.status_code == 404:
                st.warning(
                    "🔎 No restaurants matched your criteria. "
                    "Try broadening your filters — for example, pick 'Any' for location or cuisine."
                )
            elif resp.status_code == 422:
                detail = resp.json().get("detail", "Invalid input.")
                st.error(f"⚠️ Validation error: {detail}")
            elif resp.status_code >= 500:
                detail = resp.json().get("detail", "Server error.")
                st.error(f"🚨 Server error: {detail}")
            elif resp.status_code == 200:
                data = resp.json()
                recommendations = data.get("recommendations", [])
                summary = data.get("summary")
                fallback_used = data.get("fallback_used", False)
                fallback_note = data.get("fallback_note")

                # ── Fallback warning ─────────────────────────────────────
                if fallback_used and fallback_note:
                    st.markdown(
                        f'<div class="fallback-alert">⚠️ <strong>Filters relaxed:</strong> {fallback_note}</div>',
                        unsafe_allow_html=True,
                    )

                # ── AI summary banner ────────────────────────────────────
                if summary:
                    st.markdown(
                        f'<div class="ai-summary">🤖 <strong>AI Summary:</strong> {summary}</div>',
                        unsafe_allow_html=True,
                    )

                # ── Recommendation cards ─────────────────────────────────
                if not recommendations:
                    st.info("No recommendations returned. Try different preferences.")
                else:
                    for idx, rec in enumerate(recommendations, start=1):
                        # Build cuisine badges
                        cuisine_tags = [c.strip() for c in rec.get("cuisines", "").split(",") if c.strip()]
                        badges_html = " ".join(
                            f'<span class="cuisine-badge">{tag}</span>' for tag in cuisine_tags[:5]
                        )

                        # Feature pills
                        delivery_cls = "yes" if rec.get("has_online_delivery") else "no"
                        delivery_lbl = "🚚 Delivery" if rec.get("has_online_delivery") else "No Delivery"
                        booking_cls = "yes" if rec.get("has_table_booking") else "no"
                        booking_lbl = "📅 Booking" if rec.get("has_table_booking") else "No Booking"

                        rating = rec.get("aggregate_rating", 0)
                        stars = "⭐" * int(round(rating))

                        card_html = f"""
                        <div class="restaurant-card">
                            <h3>#{idx} {rec.get("name", "Unknown")}</h3>
                            <div class="meta">
                                📍 {rec.get("location", "")} &nbsp;|&nbsp;
                                {stars} {rating:.1f}/5 &nbsp;|&nbsp;
                                💰 ₹{rec.get("average_cost_for_two", 0)} for two
                            </div>
                            <div>{badges_html}</div>
                            <div style="margin-top:6px;">
                                <span class="feature-pill {delivery_cls}">{delivery_lbl}</span>
                                <span class="feature-pill {booking_cls}">{booking_lbl}</span>
                            </div>
                            <div class="explanation">💬 {rec.get("explanation", "")}</div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)

            else:
                st.error(f"Unexpected status code {resp.status_code} from the backend.")

        except httpx.ConnectError:
            st.error(
                "🔌 **Cannot reach the backend API.** "
                f"Make sure the FastAPI server is running at `{API_BASE_URL}`. "
                "Start it with: `uvicorn api.main:app --reload`"
            )
        except httpx.TimeoutException:
            st.warning(
                "⏳ The request timed out. The AI may need more time — please try again."
            )
        except Exception as exc:
            st.error(f"An unexpected error occurred: {exc}")

else:
    # Landing state
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🎯 Smart Filters")
        st.markdown("Narrow down by location, budget, cuisine, and rating.")
    with col2:
        st.markdown("### 🤖 AI-Powered")
        st.markdown("Our AI ranks and explains why each restaurant suits you.")
    with col3:
        st.markdown("### ⚡ Instant Results")
        st.markdown("Get personalised picks from 40,000+ restaurants in seconds.")
    st.markdown("---")
    st.caption("👈 Use the sidebar to set your preferences, then click **Find Recommendations**.")
