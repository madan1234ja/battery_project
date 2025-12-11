from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import joblib
import pandas as pd

# Load model at startup
model = joblib.load("models/battery_life_model.pkl")

app = FastAPI(title="Battery Life Prediction API")
# allow cross-origin requests from your GitHub Pages frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://madan1234ja.github.io"],  # <- exact origin of your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BatteryInput(BaseModel):
    battery_percent: float
    cpu_pct: float
    screen_on: int
    background_factor: float
    drain_per_min: float
    roll_cpu_5: float
    roll_drain_5: float
    session_encoded: int

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Battery API is running"}

@app.post("/predict")
def predict_battery_life(data: BatteryInput):
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
