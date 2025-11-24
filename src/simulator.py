from typing import Optional
import pandas as pd
import numpy as np
import joblib

def _count_in_window(times: pd.Series, pivot: pd.Timestamp, window_min: int = 30) -> int:
    start = pivot - pd.Timedelta(minutes=window_min/2)
    end = pivot + pd.Timedelta(minutes=window_min/2)
    return int(((times >= start) & (times <= end)).sum())

def simulate_shift(df: pd.DataFrame, model_path: str, flight_number: str, shift_minutes: int = 15,
                   by: str = "departure", window_min: int = 30) -> dict:
    """Shift a flight's scheduled time and estimate new delay risk using a trained model.
    """
    model = joblib.load(model_path)
    if by == "departure":
        tcol = "scheduled_departure_dt"
        hour_col = "dep_hour"
        dow_col = "dep_dow"
        delay_col = "dep_delay_min"
    else:
        tcol = "scheduled_arrival_dt"
        hour_col = "arr_hour"
        dow_col = "arr_dow"
        delay_col = "arr_delay_min"

    s = df.copy()
    target = s[s.get("flight_number","").astype(str) == str(flight_number)].head(1)
    if target.empty:
        return {"error": f"Flight {flight_number} not found."}

    original_time = target.iloc[0][tcol]
    new_time = original_time + pd.Timedelta(minutes=shift_minutes)

    # Estimate congestion: number of flights within window
    times = s[tcol].dropna().sort_values()
    orig_cong = _count_in_window(times, original_time, window_min=window_min)
    new_cong = _count_in_window(times, new_time, window_min=window_min)

    # Build feature row for model
    def _row(time, cong):
        return pd.DataFrame({
            "hour": [time.hour],
            "dow": [time.dayofweek],
            "slot_window_flights": [cong],
            "airline": [target.get("airline", pd.Series(["UNK"])).iloc[0] if "airline" in target else "UNK"],
            "destination": [target.get("destination", pd.Series(["UNK"])).iloc[0] if "destination" in target else "UNK"],
            "origin": [target.get("origin", pd.Series(["UNK"])).iloc[0] if "origin" in target else "UNK"],
        })

    orig_prob = float(model.predict_proba(_row(original_time, orig_cong))[:,1][0])
    new_prob = float(model.predict_proba(_row(new_time, new_cong))[:,1][0])

    return {
        "flight_number": str(flight_number),
        "original_time": str(original_time),
        "shift_minutes": int(shift_minutes),
        "new_time": str(new_time),
        "orig_window_flights": int(orig_cong),
        "new_window_flights": int(new_cong),
        "orig_delay_prob": orig_prob,
        "new_delay_prob": new_prob,
        "delta_prob": new_prob - orig_prob
    }
