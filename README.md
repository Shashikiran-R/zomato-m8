## 🧠 System Prompt
Our recommendation engine uses a highly structured system prompt to guide the LLM. It establishes the persona of an "expert restaurant recommendation assistant" and explicitly instructs the model to return a single valid JSON object. The prompt provides ranking guidelines (prioritizing matches based on cuisine, budget, rating, and votes) and explicitly commands the model to avoid generating markdown wrappers, code fences, or extraneous commentary.

## 📄 Response Schema
The system enforces a strict JSON output structure expected by the FastAPI backend. The output must conform to a Pydantic-validated schema consisting of:
- `recommendations`: An array (max 5 items) of restaurant objects containing keys for `name`, `location`, `cuisines`, `aggregate_rating`, `average_cost_for_two`, `has_online_delivery`, `has_table_booking`, and a 2-3 sentence personalized `explanation` of why it's a good fit.
- `summary`: A one-paragraph (3-5 sentences) overall synthesis of the generated recommendations. 

## 🔄 What Changed Across Prompt Versions & Why
During development (Phase 4), we transitioned from a conversational, free-form prompt (originally targeting the Gemini API) to a highly deterministic, constraint-heavy prompt targeting the Groq API (`openai/gpt-oss-120b`). 
* **Why:** Open-weights models often wrap their output in markdown (` ```json ... ``` `), which breaks strict JSON parsing in backend pipelines. The new prompt explicitly forbids markdown and code fences, forcing raw JSON. We also added stricter instructions to prioritize `aggregate_rating` and `votes` to improve the relevance of the recommendations.

## 📏 How the Scope Limit is Enforced
To prevent context window overflow, hallucination, and excessive token usage, the scope is strictly limited in two phases:
1. **Deterministic Pre-filtering:** Before the LLM is ever called, the `src/filters.py` layer aggressively filters the Zomato dataset by location, budget, cuisine, and rating.
2. **Hard Capping:** The filtered results are sorted by rating/votes and forcefully pruned to a maximum of **15 candidate restaurants** (`MAX_CANDIDATES = 15`). This ensures a compact, highly relevant payload is sent to the LLM, from which it is instructed to return only the top 5.

## 🛠️ Tech Stack
* **Language:** Python 3.11+
* **Backend:** FastAPI, Uvicorn, Pydantic (validation)
* **Frontend:** Custom HTML/JS with Tailwind CSS 
* **Data Processing:** Pandas, PyArrow (Parquet caching)
* **Dataset Engine:** Hugging Face `datasets`
* **LLM Integration:** Groq API (`openai/gpt-oss-120b` via the `groq` SDK)
