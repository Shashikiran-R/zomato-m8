import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import get_settings
from src.filters import shortlist
from src.llm_client import get_recommendations
from src.models import RecommendationResponse, UserPreferenceInput
from src.rate_limiter import rate_limiter

logger = logging.getLogger(__name__)


# ── Global State ─────────────────────────────────────────────────────────────

class AppState:
    df: pd.DataFrame = None
    locations: list[str] = []
    cuisines: list[str] = []

app_state = AppState()


# ── Lifespan Events ──────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup event: Load the preprocessed Parquet file into memory and extract
    distinct locations and cuisines for dropdowns.
    """
    settings = get_settings()
    logger.info("Loading dataset from %s", settings.data_path)
    try:
        df = pd.read_parquet(settings.data_path)
        app_state.df = df
        
        # Precompute locations (sorted, distinct, non-empty)
        raw_locs = df["location"].dropna().unique()
        app_state.locations = sorted(
            list(set(str(loc).strip() for loc in raw_locs if str(loc).strip()))
        )
        
        # Precompute cuisines (sorted, distinct, non-empty tags)
        all_cuisines = set()
        for c_list in df["cuisines"].dropna():
            tags = [c.strip() for c in c_list.split(",")]
            all_cuisines.update(tags)
        app_state.cuisines = sorted(list(c for c in all_cuisines if c))
        
        logger.info(
            "Dataset loaded successfully: %d rows, %d locations, %d cuisines.",
            len(df), len(app_state.locations), len(app_state.cuisines)
        )
    except Exception as exc:
        logger.error("Failed to load dataset during startup: %s", exc)
        app_state.df = pd.DataFrame()
        
    yield
    # Shutdown event: clear memory
    app_state.df = None


# ── Application Setup ────────────────────────────────────────────────────────

app = FastAPI(
    title="Zomato AI Recommendation API",
    description="RESTful endpoints for the AI-Powered Restaurant Recommendation System.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Error Handlers ───────────────────────────────────────────────────────────

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """Custom 422 handler for validation errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation error", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Custom 500 handler for unexpected failures (e.g. upstream LLM crashes)."""
    logger.exception("Unhandled server error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error due to unexpected failure."},
    )


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check():
    """Health and readiness probe."""
    is_ready = app_state.df is not None and not app_state.df.empty
    return {
        "status": "ok" if is_ready else "degraded",
        "dataset_loaded": is_ready,
        "rate_limits": rate_limiter.status(),
    }


@app.get("/locations", tags=["Metadata"])
async def get_locations() -> list[str]:
    """Return distinct sorted locations available in the dataset for dropdowns."""
    return app_state.locations


@app.get("/cuisines", tags=["Metadata"])
async def get_cuisines() -> list[str]:
    """Return distinct sorted cuisine tags for dropdowns."""
    return app_state.cuisines


@app.post("/recommend", response_model=RecommendationResponse, tags=["Core"])
async def recommend_restaurants(prefs: UserPreferenceInput):
    """
    Accept UserPreferenceInput, run filter pipeline, call LLM, and return
    RecommendationResponse.
    """
    if app_state.df is None or app_state.df.empty:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Restaurant dataset is currently unavailable."
        )
        
    settings = get_settings()
    
    # 1. Deterministic Filtering
    try:
        filter_result = shortlist(app_state.df, prefs, max_candidates=settings.max_candidates)
    except Exception as exc:
        logger.error("Error during filtering: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing restaurant filters."
        )

    # Return 404 if no matching restaurants (even after fallback)
    if filter_result.candidates.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No restaurants found matching the given preferences."
        )
        
    # 2. LLM Integration
    try:
        response = get_recommendations(
            prefs=prefs,
            filter_result=filter_result,
            groq_api_key=settings.groq_api_key,
            llm_model=settings.llm_model,
            max_retries=settings.llm_max_retries,
            timeout=settings.llm_timeout_seconds,
        )
        return response
    except Exception as exc:
        logger.error("Upstream LLM failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendations from the upstream LLM."
        )
