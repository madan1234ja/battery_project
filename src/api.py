# src/api.py
from typing import Optional
from fastapi import FastAPI, Request, Header, HTTPException, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import pandas as pd
import time
import logging
import os
from collections import Counter

# ----------------------------
# Configuration & utilities
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("battery_api")

# Optional API key: set API_KEY in environment on Render to require it
API_KEY = os.getenv("API_KEY")


def check_api_key(x_api_key: Optional[str]):
    """Raise HTTPException 401 if API_KEY is set and header is incorrect/missing."""
    if API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API Key")


# ----------------------------
# Load model (at module import)
# ----------------------------
# model file expected at models/battery_life_model.pkl
# If this fails in production check requirements/scikit-learn version compatibility.
model = joblib.load("models/battery_life_model.pkl")

# ----------------------------
# FastAPI app + CORS
# ----------------------------
app = FastAPI(title="Battery Life Prediction API")

# In production prefer an exact origin list instead of "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://madan1234ja.github.io"],  # frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Simple in-memory metrics
# ----------------------------
_metrics = Counter()


# ----------------------------
# Request logging middleware
# ----------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception(f"Error handling request {request.method} {request.url.path}")
        raise
    elapsed_ms = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} status={response.status_code} time_ms={elapsed_ms:.1f}")
    return response


# ----------------------------
# Short-term OPTIONS preflight fallback (safe to remove later)
# ----------------------------
# This is a safety fallback for browsers that send OPTIONS preflight.
# CORSMiddleware should normally handle preflight; remove this handler once CORS is confirmed.
@app.options("/predict")
def predict_options():
    headers = {
        "Access-Control-Allow-Origin": "https://madan1234ja.github.io",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-KEY",
        "Access-Control-Allow-Credentials": "true",
    }
    return Response(status_code=204, headers=headers)


# ----------------------------
# Pydantic input model
# ----------------------------
class BatteryInput(BaseModel):
    battery_percent: float
    cpu_pct: float
    screen_on: int
    background_factor: float
    drain_per_min: float
    roll_cpu_5: float
    roll_drain_5: float
    session_encoded: int


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health_check():
    """Basic health check and presence of the API."""
    return {"status": "ok", "message": "Battery API is running"}


@app.get("/metrics")
def get_metrics():
    """Return a simple JSON metric; replace with prometheus_client for production scraping."""
    return {"predict_calls": _metrics["predict_calls"]}


@app.post("/predict")
def predict_battery_life(data: BatteryInput, x_api_key: Optional[str] = Header(None)):
    """Predict remaining minutes of battery life from the incoming features.

    If API_KEY environment variable is set, the request must include header `x-api-key`.
    """
    # API key check (no-op if API_KEY is not configured)
    check_api_key(x_api_key)

    # Increment in-process counter
    _metrics["predict_calls"] += 1

    # Build DataFrame exactly as the model expects
    df = pd.DataFrame([[
        data.battery_percent,
        data.cpu_pct,
        data.screen_on,
        data.background_factor,
        data.drain_per_min,
        data.roll_cpu_5,
        data.roll_drain_5,
        data.session_encoded
    ]], columns=[
        "battery_percent",
        "cpu_pct",
        "screen_on",
        "background_factor",
        "drain_per_min",
        "roll_cpu_5",
        "roll_drain_5",
        "session_encoded"
    ])

    # Model inference
    pred = model.predict(df)[0]
    return {
        "predicted_minutes_remaining": round(float(pred), 2),
        "input": data.dict()
    }


# ----------------------------
# Local development runner
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
