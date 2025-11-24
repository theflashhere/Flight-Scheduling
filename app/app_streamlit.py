import streamlit as st
import pandas as pd
import numpy as np
import os

from src.data_loader import load_flights
from src.features import add_time_features, compute_congestion, mark_peak_hours
from src.analysis import busiest_slots, best_time_windows, runway_utilization
from src.model_delay import train_delay_model
from src.cascade import link_rotations, cascade_scores
from src.simulator import simulate_shift
from app.nlp import intent_and_params

st.set_page_config(page_title="Flight Scheduling AI", layout="wide")

st.title("✈️ Flight Scheduling & Delay Insights")

st.sidebar.header("Upload / Sample Data")
uploaded = st.sidebar.file_uploader("Upload CSV/XLSX", type=["csv","xlsx"])
use_sample = st.sidebar.checkbox("Use sample data", value=True)

if uploaded:
    tmp_path = os.path.join("data", uploaded.name)
    with open(tmp_path, "wb") as f:
        f.write(uploaded.getbuffer())
    data_path = tmp_path
elif use_sample and os.path.exists("data/Flight_Data.xlsx"):
    data_path = "data/Flight_Data.xlsx"
else:
    data_path = None

if not data_path:
    st.info("Upload a dataset or enable 'Use sample data' to proceed.")
    st.stop()

df = load_flights(data_path)
df = add_time_features(df)

# Choose analysis mode
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview","Busiest Slots","Best Times","Simulator","Cascade Impact"])

with tab1:
    st.subheader("Overview")
    st.write("Rows:", len(df))
    st.dataframe(df.head(50))

    # Simple KPIs
    kpi_cols = st.columns(4)
    if "dep_delay_min" in df:
        kpi_cols[0].metric("Avg Dep Delay (min)", f"{df['dep_delay_min'].mean():.1f}")
    if "arr_delay_min" in df:
        kpi_cols[1].metric("Avg Arr Delay (min)", f"{df['arr_delay_min'].mean():.1f}")
    if "dep_slot_15m" in df:
        kpi_cols[2].metric("Unique Dep Slots", df["dep_slot_15m"].nunique())
    if "arr_slot_15m" in df:
        kpi_cols[3].metric("Unique Arr Slots", df["arr_slot_15m"].nunique())

with tab2:
    st.subheader("Busiest Slots")
    by = st.selectbox("By", ["departure","arrival"], index=0)
    slot_col = "dep_slot_15m" if by=="departure" else "arr_slot_15m"
    dfC = compute_congestion(df, slot_col=slot_col, within_minutes=30, kind=by)
    busy = busiest_slots(dfC, by=by)
    st.dataframe(busy.head(50))
    util = runway_utilization(dfC, by=by, capacity_per_15m=st.number_input("Capacity per 15-min", 1, 60, 20))
    st.bar_chart(util.set_index(util.columns[0])["utilization"])

with tab3:
    st.subheader("Best Time Windows (Least Delay)")
    by = st.selectbox("Optimize for", ["departure","arrival"], index=0, key="best_by")
    window = st.slider("Window (minutes)", 30, 180, 60, step=15)
    best = best_time_windows(df, by=by, window_minutes=window)
    st.dataframe(best.head(50))
    # show top-10 as bar chart
    if not best.empty:
        top = best.nsmallest(10, "avg_delay")
        st.bar_chart(top.set_index(top.columns[0])["avg_delay"])

with tab4:
    st.subheader("What-if Simulator")
    by = st.selectbox("Mode", ["departure","arrival"], index=0, key="sim_by")
    delay_thr = st.slider("Delay threshold (min)", 5, 60, 15, step=5)
    metrics = train_delay_model(df, by=by, delay_threshold=delay_thr, model_path="reports/delay_model.joblib")
    st.code(metrics["report"])
    st.write("ROC-AUC:", metrics["roc_auc"])

    flight = st.text_input("Flight number to shift (e.g., AI101)", "")
    shift = st.slider("Shift minutes", -120, 120, 15, step=5)
    if st.button("Simulate Shift") and flight:
        res = simulate_shift(df, "reports/delay_model.joblib", flight, shift_minutes=shift, by=by)
        st.json(res)

with tab5:
    st.subheader("Cascade Impact")
    apt = st.text_input("Airport ICAO/IATA (arrivals to / departures from)", "BOM")
    edges = link_rotations(df, airport=apt)
    st.write("Rotation edges:", len(edges))
    st.dataframe(edges.head(50))
    scores = cascade_scores(df, edges)
    st.subheader("Top Cascade Flights")
    st.dataframe(scores.head(50))

st.markdown("---")
st.subheader("NLP Query")
q = st.text_input("Ask a question, e.g., 'best time to land' or 'simulate flight AI101 shift 20 min'")
if q:
    intent = intent_and_params(q)
    st.write("Intent:", intent)
    if intent["intent"] == "best_departure_time":
        best = best_time_windows(df, by="departure", window_minutes=60)
        st.dataframe(best.head(10))
    elif intent["intent"] == "best_arrival_time":
        best = best_time_windows(df, by="arrival", window_minutes=60)
        st.dataframe(best.head(10))
    elif intent["intent"] == "busiest_slots":
        dfC = compute_congestion(df, slot_col="dep_slot_15m" if "dep_slot_15m" in df else "arr_slot_15m", within_minutes=30)
        st.dataframe(busiest_slots(dfC))
    elif intent["intent"] == "simulate":
        if intent.get("flight"):
            st.json(simulate_shift(df, "reports/delay_model.joblib", intent["flight"], shift_minutes=intent.get("shift", 15)))
        else:
            st.warning("Please specify a flight number, e.g., 'simulate flight AI101 shift 20 min'")
    elif intent["intent"] == "cascade":
        edges = link_rotations(df, airport="BOM")
        st.dataframe(cascade_scores(df, edges).head(20))
    else:
        st.info("Sorry, I couldn't understand. Try asking about 'best time', 'busiest slots', 'simulate', or 'cascade'.")
