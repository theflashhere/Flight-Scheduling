from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from dateutil import tz

# Common column aliases
ALIASES = {
    "flight_number": ["flight_number", "flight", "flightno", "flt_no", "number"],
    "date": ["date", "flight_date", "dt"],
    "scheduled_departure": ["scheduled_departure", "std", "sched_dep_time", "schedule_departure", "dep_sched"],
    "actual_departure": ["actual_departure", "atd", "dep_time", "actual_dep_time", "departure_actual"],
    "scheduled_arrival": ["scheduled_arrival", "sta", "sched_arr_time", "schedule_arrival", "arr_sched"],
    "actual_arrival": ["actual_arrival", "ata", "arr_time", "actual_arr_time", "arrival_actual"],
    "origin": ["origin", "from", "dep_airport", "origin_airport"],
    "destination": ["destination", "to", "arr_airport", "destination_airport"],
    "airline": ["airline", "carrier", "operator"],
    "registration": ["registration", "tail", "aircraft", "reg"]
}

def _first_present(dcols: List[str], candidates: List[str]):
    for c in candidates:
        if c in dcols:
            return c
    return None

def standardize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    dcols = [c.strip().lower() for c in df.columns]
    mapping = {}
    for std, aliases in ALIASES.items():
        found = _first_present(dcols, aliases)
        if found:
            mapping[std] = df.columns[dcols.index(found)]
    # Rename to standardized
    sdf = df.rename(columns={v: k for k, v in mapping.items()})
    return sdf, mapping

def parse_times(df: pd.DataFrame, local_tz: str = "Asia/Kolkata") -> pd.DataFrame:
    # Merge date column with time columns if times are HH:MM strings
    tzinfo = tz.gettz(local_tz)

    def _combine(date_col, time_col):
        if date_col in df and time_col in df:
            # If time is already datetime, keep; else combine
            if not np.issubdtype(df[time_col].dtype, np.datetime64):
                return pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str), errors="coerce").dt.tz_localize(tzinfo, nonexistent='NaT', ambiguous='NaT')
            else:
                x = pd.to_datetime(df[time_col], errors="coerce")
                if x.dt.tz is None:
                    x = x.dt.tz_localize(tzinfo, nonexistent='NaT', ambiguous='NaT')
                return x
        return None

    # Try departure times
    dep_sched = _combine("date", "scheduled_departure") or (pd.to_datetime(df.get("scheduled_departure"), errors="coerce"))
    dep_act = _combine("date", "actual_departure") or (pd.to_datetime(df.get("actual_departure"), errors="coerce"))
    arr_sched = _combine("date", "scheduled_arrival") or (pd.to_datetime(df.get("scheduled_arrival"), errors="coerce"))
    arr_act = _combine("date", "actual_arrival") or (pd.to_datetime(df.get("actual_arrival"), errors="coerce"))

    # Localize if not tz-aware
    def _ensure_tz(x):
        if x is None:
            return None
        x = pd.to_datetime(x, errors="coerce")
        try:
            if x.dt.tz is None:
                x = x.dt.tz_localize(tzinfo, nonexistent='NaT', ambiguous='NaT')
        except Exception:
            try:
                x = x.dt.tz_convert(tzinfo)
            except Exception:
                pass
        return x

    dep_sched = _ensure_tz(dep_sched)
    dep_act = _ensure_tz(dep_act)
    arr_sched = _ensure_tz(arr_sched)
    arr_act = _ensure_tz(arr_act)

    if dep_sched is not None:
        df["scheduled_departure_dt"] = dep_sched
    if dep_act is not None:
        df["actual_departure_dt"] = dep_act
    if arr_sched is not None:
        df["scheduled_arrival_dt"] = arr_sched
    if arr_act is not None:
        df["actual_arrival_dt"] = arr_act

    return df

def load_flights(path: str, local_tz: str = "Asia/Kolkata") -> pd.DataFrame:
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df, mapping = standardize_columns(df)
    df = parse_times(df, local_tz=local_tz)

    # Compute delays in minutes if possible
    if "scheduled_departure_dt" in df and "actual_departure_dt" in df:
        df["dep_delay_min"] = (df["actual_departure_dt"] - df["scheduled_departure_dt"]).dt.total_seconds() / 60.0
    if "scheduled_arrival_dt" in df and "actual_arrival_dt" in df:
        df["arr_delay_min"] = (df["actual_arrival_dt"] - df["scheduled_arrival_dt"]).dt.total_seconds() / 60.0

    # Clean up negatives (early departures/arrivals) allowed but cap extreme outliers
    for c in ["dep_delay_min", "arr_delay_min"]:
        if c in df:
            df[c] = df[c].clip(lower=-120, upper=600)

    # Airport normalization
    for c in ["origin", "destination", "airline", "flight_number", "registration"]:
        if c in df:
            df[c] = df[c].astype(str).str.strip().str.upper()

    return df
