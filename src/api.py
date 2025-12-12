# src/api.py
from fastapi import FastAPI, Request, Header, HTTPException, Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import pandas as pd
import time
import logging
import os
from collections import Counter

# --- logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("battery_api")

# --- load model ----------
model = joblib.load("models/battery_life_model.pkl")

# --- app + CORS ----------
app = FastAPI(title="Battery Life Prediction API")

# During testing you may use ["*"], but in production use exact origin list
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://madan1234ja.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- simple metrics ---
_metrics = Counter()
@app.get("/metrics")
def get_metrics():
    return {"predict_calls": _metrics["predict_calls"]}


# --- request logging middleware ---
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

# --- OPTIONS preflight fallback (short-term) ---
@app.options("/predict")
def predict_options():
    headers = {
        "Access-Control-Allow-Origin": "https://madan1234ja.github.io",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, X-API-KEY",
        "Access-Control-Allow-Credentials": "true",
    }
    return Response(status_code=204, headers=headers)

# --- input model ----------
class BatteryInput(BaseModel):
    battery_percent: float
    cpu_pct: float
    screen_on: int
    background_factor: float
    drain_per_min: float
    roll_cpu_5: float
    roll_drain_5: float
    session_encoded: int

# --- health endpoint ----------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Battery API is running"}

# --- metrics endpoint ----------
@app.get("/metrics")
def get_metrics():
    return {"predict_calls": _metrics["predict_calls"]}

# --- optional API key protection (reads API_KEY env var) ---
API_KEY = os.getenv("API_KEY")

def check_api_key(x_api_key: str | None):
    if API_KEY:
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API Key")

# --- predict endpoint ----------
@app.post("/predict")
def predict_battery_life(data: BatteryInput, x_api_key: str | None = Header(None)):
    # verify API key if configured
    check_api_key(x_api_key)

    # increment simple counter
    _metrics["predict_calls"] += 1

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

    pred = model.predict(df)[0]
    return {
        "predicted_minutes_remaining": round(float(pred), 2),
        "input": data.dict()
    }

# --- local run (for developer) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
