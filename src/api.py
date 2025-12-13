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

# Prometheus
from prometheus_client import Counter as PromCounter
from prometheus_client import Histogram, generate_latest, CONTENT_TYPE_LATEST

# ------------------ CONFIG ------------------
MODEL_NAME = os.getenv("MODEL_NAME", "battery_life_model")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
API_KEY = os.getenv("API_KEY")

# ------------------ LOGGING ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("battery_api")

# ------------------ LOAD MODEL ------------------
model = joblib.load("models/battery_life_model.pkl")

# ------------------ APP ------------------
app = FastAPI(title="Battery Life Prediction API")

# ------------------ CORS ------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://madan1234ja.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ SIMPLE JSON METRICS ------------------
_simple_metrics = Counter()

@app.get("/metrics")
def json_metrics():
    return {"predict_calls": _simple_metrics["predict_calls"]}

# ------------------ PROMETHEUS METRICS ------------------
PREDICT_COUNTER = PromCounter(
    "predict_requests_total",
    "Total number of prediction requests"
)

PREDICT_LATENCY = Histogram(
    "predict_request_latency_seconds",
    "Prediction request latency in seconds"
)

@app.get("/prometheus")
def prometheus_metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# ------------------ MIDDLEWARE ------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed_ms = (time.time() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} time_ms={elapsed_ms:.2f}"
    )
    return response

# ------------------ AUTH ------------------
def check_api_key(x_api_key: str | None):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

# ------------------ SCHEMA ------------------
class BatteryInput(BaseModel):
    battery_percent: float
    cpu_pct: float
    screen_on: int
    background_factor: float
    drain_per_min: float
    roll_cpu_5: float
    roll_drain_5: float
    session_encoded: int

# ------------------ ROUTES ------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": {
            "name": MODEL_NAME,
            "version": MODEL_VERSION
        }
    }

@app.post("/predict")
def predict_battery_life(
    data: BatteryInput,
    x_api_key: str | None = Header(None)
):
    check_api_key(x_api_key)

    start_time = time.time()

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

    prediction = model.predict(df)[0]

    # ---- metrics ----
    _simple_metrics["predict_calls"] += 1
    PREDICT_COUNTER.inc()
    PREDICT_LATENCY.observe(time.time() - start_time)

    return {
        "predicted_minutes_remaining": round(float(prediction), 2),
        "input": data.dict()
    }

# ------------------ LOCAL RUN ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000)
