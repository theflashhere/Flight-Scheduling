# Flight Scheduling & Delay Insights (BOM/DEL)

An AI-driven toolkit to analyze 1-week flight data (e.g., from Flightradar24/FlightAware) for **Mumbai (BOM)** and **Delhi (DEL)** and assist controllers/operators with:
- Best time to takeoff/land (based on scheduled vs. actual delays)
- Busiest time slots to avoid
- A "what-if" **scheduling simulator** (shift a flight time, estimate congestion & delay impact)
- **Cascading impact**: identify flights with highest potential to propagate delays
- **NLP query interface** to ask natural questions like _"When is the least congested landing slot?"_

> ⚙️ **Sample data**: `data/Flight_Data.xlsx`. You can replace it with your own 1-week export from Flightradar24.

---

## Quickstart

```bash
# 1) Create & activate a virtual environment (optional)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Inspect/standardize data
python -m scripts.preview_data --path data/Flight_Data.xlsx

# 4) Run the Streamlit app
streamlit run app/app_streamlit.py
```

Open the local URL (usually http://localhost:8501) to use the UI.

---

## Data Expectations

The toolkit tries to auto-detect column names. It works best if your dataset includes some of the following (any reasonable variations are handled):

- **Flight identifiers**: `flight_number` / `flight` / `flightno`
- **Dates**: `date`
- **Scheduled times**: `scheduled_departure`, `scheduled_arrival` (or `std`, `sta`, `sched_dep_time`, `sched_arr_time`)
- **Actual times**: `actual_departure`, `actual_arrival` (or `atd`, `ata`, `dep_time`, `arr_time`)
- **Airports**: `origin`, `destination` (or `from`, `to`, `dep_airport`, `arr_airport`)
- **Airline**: `airline`
- **Registration / Tail number**: `registration` / `aircraft` (optional, improves cascade analysis)

Timestamps are parsed to timezone-aware `datetime64[ns]` where possible. Delays are computed in **minutes**.

---

## Major Components

- **src/data_loader.py** — Robust loader to standardize schema and parse times.
- **src/features.py** — Feature engineering (delays, slots, congestion, peaks).
- **src/analysis.py** — Busiest slots, best times to schedule.
- **src/model_delay.py** — Delay classifier/regressor to estimate delay risk.
- **src/cascade.py** — Graph-based cascade potential (links sequential rotations).
- **src/simulator.py** — What-if shift simulations using model + simple queuing approximation.
- **app/app_streamlit.py** — Interactive UI + **NLP query**.
- **scripts/generate_report.py** — Creates a PDF summary (charts + key insights).

---

## NLP Interface

A lightweight rules-based intent router (no closed-source APIs), with optional HuggingFace models if you want to add semantic search later. See `app/nlp.py` to customize intents/utterances.

---

## Submission

- Use **scripts/generate_report.py** to render a concise PDF with the required insights.
- Export figures from the app (PNG) and attach to your PDF if needed.
- Include this README + your PDF when submitting.
