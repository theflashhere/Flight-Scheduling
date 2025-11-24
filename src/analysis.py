import pandas as pd
import numpy as np

def busiest_slots(df: pd.DataFrame, by: str = "departure") -> pd.DataFrame:
    if by == "departure":
        slot = "dep_slot_15m"
        delay = "dep_delay_min"
    else:
        slot = "arr_slot_15m"
        delay = "arr_delay_min"

    keep = [c for c in [slot, delay] if c in df]
    if not keep:
        return pd.DataFrame()

    g = df.groupby(slot).agg(
        flights=("flight_number", "count") if "flight_number" in df else (slot, "size"),
        avg_delay=(delay, "mean") if delay in df else (slot, "size")
    ).reset_index().sort_values(["flights", "avg_delay"], ascending=[False, False])
    return g

def best_time_windows(df: pd.DataFrame, by: str = "departure", window_minutes: int = 60) -> pd.DataFrame:
    if by == "departure":
        tcol = "scheduled_departure_dt"
        delay = "dep_delay_min"
    else:
        tcol = "scheduled_arrival_dt"
        delay = "arr_delay_min"

    if tcol not in df or delay not in df:
        return pd.DataFrame()

    s = df.set_index(tcol).sort_index()
    # Resample to windows and compute stats
    stats = s[delay].resample(f"{window_minutes}min").agg(["count", "mean", "median"])
    stats = stats.rename(columns={"count":"flights","mean":"avg_delay","median":"median_delay"}).reset_index()
    stats = stats.sort_values(["avg_delay","flights"], ascending=[True, False])
    return stats

def runway_utilization(df: pd.DataFrame, by: str = "departure", capacity_per_15m: int = 20) -> pd.DataFrame:
    slot_col = "dep_slot_15m" if by == "departure" else "arr_slot_15m"
    if slot_col not in df:
        return pd.DataFrame()
    counts = df.groupby(slot_col).size().rename("flights").reset_index()
    counts["capacity"] = capacity_per_15m
    counts["utilization"] = counts["flights"] / counts["capacity"]
    return counts.sort_values("utilization", ascending=False)
