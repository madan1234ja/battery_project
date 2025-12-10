import joblib
import pandas as pd

model = joblib.load("../models/battery_life_model.pkl")

def predict_remaining_minutes(
    battery_percent,
    cpu_pct,
    screen_on,
    background_factor,
    drain_per_min,
    roll_cpu_5,
    roll_drain_5,
    session_encoded
):
    data = pd.DataFrame([[
        battery_percent, cpu_pct, screen_on, background_factor,
        drain_per_min, roll_cpu_5, roll_drain_5, session_encoded
    ]], columns=[
        "battery_percent", "cpu_pct", "screen_on", "background_factor",
        "drain_per_min", "roll_cpu_5", "roll_drain_5", "session_encoded"
    ])

    pred = model.predict(data)[0]
    return round(pred, 2)

if __name__ == "__main__":
    print("Example prediction:", predict_remaining_minutes(
        50, 20, 1, 1.0, 0.30, 22, 0.28, 1
    ))
