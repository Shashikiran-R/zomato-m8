# Deployment Guide: NextLeap Eats

This guide explains how to deploy the entire **NextLeap Eats Recommendation Engine** into production using **Railway** for the FastAPI backend and **Vercel** for the static HTML/JS frontend.

---

## Part 1: Deploying the Backend on Railway

Railway is an excellent platform for deploying Python/FastAPI applications. It will automatically detect your `requirements.txt` and build your environment.

### Steps:
1. **Push your code to GitHub:** Ensure your latest codebase, including `requirements.txt` and the `data/` folder, is pushed to a GitHub repository.
2. **Create a Railway Project:**
   - Go to [Railway.app](https://railway.app/) and log in.
   - Click **New Project** > **Deploy from GitHub repo**.
   - Select your NextLeap Eats repository.
3. **Configure Environment Variables:**
   - Go to the **Variables** tab in your new Railway service.
   - Add your necessary secrets, primarily:
     - `GROQ_API_KEY`: `your_groq_api_key_here`
4. **Deploy & Get URL:**
   - Railway will automatically start building and deploying your app using `uvicorn`.
   - Once deployed, go to the **Settings** tab.
   - Click **Generate Domain** (if it hasn't generated one already).
   - Copy this Public URL (e.g., `https://your-backend-app.up.railway.app`). You will need this for the frontend!

*(Note: The backend has `allow_origins=["*"]` configured in `api/main.py`, so CORS is already handled and won't block your Vercel frontend).*

---

## Part 2: Deploying the Frontend on Vercel

Vercel is the perfect host for our static HTML, CSS, and JS frontend.

### Steps:
1. **Update the API URL in your Code:**
   - Before deploying (or by pushing a new commit), open `frontend/script.js`.
   - Find the first line: 
     ```javascript
     const API_BASE_URL = 'http://localhost:8000';
     ```
   - Change it to your new Railway backend URL from Part 1. *Do not include a trailing slash.*
     ```javascript
     const API_BASE_URL = 'https://your-backend-app.up.railway.app';
     ```
   - Commit and push this change to GitHub.
2. **Create a Vercel Project:**
   - Go to [Vercel.com](https://vercel.com/) and log in.
   - Click **Add New...** > **Project**.
   - Import your NextLeap Eats repository.
3. **Configure the Project Root:**
   - Before clicking deploy, look for the **Root Directory** setting.
   - Click **Edit** and select the `frontend` folder. (This tells Vercel to serve `frontend/index.html` as the main page).
4. **Deploy:**
   - Click **Deploy**.
   - Vercel will instantly deploy your static assets. 
   - Click **Visit** to see your live "Editorial Epicure" UI connected to your production backend!

---

## Maintenance & Troubleshooting

- **Dataset Updates:** If you update the `zomato.parquet` file in the `data/` folder, just push the changes to GitHub. Railway will automatically redeploy and load the new data into memory on startup.
- **Backend Errors:** If the frontend shows a "Cannot reach the backend API" error in production, check your Railway deployment logs to ensure the server started correctly and the `GROQ_API_KEY` is valid.
