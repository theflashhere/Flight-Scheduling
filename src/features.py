import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer departure for "takeoff" analysis; arrival for "landing"
    if "scheduled_departure_dt" in df:
        df["dep_hour"] = df["scheduled_departure_dt"].dt.hour
        df["dep_dow"] = df["scheduled_departure_dt"].dt.dayofweek  # Monday=0
        df["dep_slot_15m"] = (df["scheduled_departure_dt"].dt.floor("15min"))
    if "scheduled_arrival_dt" in df:
        df["arr_hour"] = df["scheduled_arrival_dt"].dt.hour
        df["arr_dow"] = df["scheduled_arrival_dt"].dt.dayofweek
        df["arr_slot_15m"] = (df["scheduled_arrival_dt"].dt.floor("15min"))
    return df

def compute_congestion(df: pd.DataFrame, slot_col: str, within_minutes: int = 15, kind: str = "departure") -> pd.DataFrame:
    # Count flights per slot (rolling window)
    if slot_col not in df:
        return df
    sdf = df.sort_values(slot_col).copy()
    sdf["slot_count"] = 1
    # Group by slot
    counts = sdf.groupby(slot_col)["slot_count"].sum().rename("slot_flights")
    sdf = sdf.merge(counts, left_on=slot_col, right_index=True, how="left")

    # Rolling within +- within_minutes/2 window using resample
    # First resample to 1-minute frequency counts
    ts = counts.resample("1min").sum().fillna(0)
    window = within_minutes
    rolling = ts.rolling(f"{window}min", center=True).sum().rename("slot_window_flights")
    sdf = sdf.merge(rolling, left_on=slot_col, right_index=True, how="left")
    return sdf

def mark_peak_hours(df: pd.DataFrame, ref_col: str = "dep_hour") -> pd.DataFrame:
    if ref_col in df:
        # Peak if in [7-10] or [18-22] (heuristic; can be learned from data)
        df["is_peak"] = df[ref_col].between(7, 10) | df[ref_col].between(18, 22)
    return df
