import pandas as pd
import numpy as np
import networkx as nx

def link_rotations(df: pd.DataFrame, airport: str = "BOM", max_turn_minutes: int = 240) -> pd.DataFrame:
    """Link arrivals into airport to next departure of same registration (best) or airline within a window.
    Returns edges: arrival_flight -> departure_flight
    """
    sdf = df.copy()
    # Identify arrivals into airport and subsequent departures from airport
    arrivals = sdf[(sdf.get("destination") == airport) & ("actual_arrival_dt" in sdf)].dropna(subset=["actual_arrival_dt"])
    departures = sdf[(sdf.get("origin") == airport) & ("scheduled_departure_dt" in sdf)].dropna(subset=["scheduled_departure_dt"])

    # Priority 1: registration
    edges = []
    if "registration" in sdf:
        for reg, a_group in arrivals.groupby("registration"):
            if reg not in departures.get("registration", pd.Series(dtype=str)).unique():
                continue
            d_group = departures[departures["registration"] == reg]
            for _, a in a_group.iterrows():
                # Next departure of same reg after arrival
                candidate = d_group[d_group["scheduled_departure_dt"] >= a["actual_arrival_dt"]].sort_values("scheduled_departure_dt").head(1)
                if not candidate.empty:
                    d = candidate.iloc[0]
                    dt = (d["scheduled_departure_dt"] - a["actual_arrival_dt"]).total_seconds() / 60.0
                    if 0 <= dt <= max_turn_minutes:
                        edges.append((str(a.get("flight_number", f"A{_}")), str(d.get("flight_number", f"D{d.name}")), {"turn_minutes": dt, "reg": reg}))

    # Fallback 2: airline-based linking (approximate)
    if not edges and "airline" in sdf:
        for al, a_group in arrivals.groupby("airline"):
            d_group = departures[departures["airline"] == al]
            for _, a in a_group.iterrows():
                candidate = d_group[d_group["scheduled_departure_dt"] >= a["actual_arrival_dt"]].sort_values("scheduled_departure_dt").head(1)
                if not candidate.empty:
                    d = candidate.iloc[0]
                    dt = (d["scheduled_departure_dt"] - a["actual_arrival_dt"]).total_seconds() / 60.0
                    if 0 <= dt <= max_turn_minutes:
                        edges.append((str(a.get("flight_number", f"A{_}")), str(d.get("flight_number", f"D{d.name}")), {"turn_minutes": dt, "airline": al}))

    return pd.DataFrame([{"src": e[0], "dst": e[1], **e[2]} for e in edges])

def cascade_scores(df: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Compute centrality metrics to identify flights with high cascade potential."""
    if edges.empty:
        return pd.DataFrame()
    G = nx.DiGraph()
    G.add_nodes_from(df.get("flight_number", pd.Series([f"F{i}" for i in range(len(df))])).astype(str).tolist())
    for _, e in edges.iterrows():
        G.add_edge(e["src"], e["dst"], weight=max(1.0, 240 - e.get("turn_minutes", 60)))  # tighter turns => higher weight
    # Centrality
    btw = nx.betweenness_centrality(G, weight="weight", normalized=True)
    pr = nx.pagerank(G, weight="weight")
    outdeg = dict(G.out_degree())
    indeg = dict(G.in_degree())

    res = pd.DataFrame({
        "flight_number": list(btw.keys()),
        "betweenness": list(btw.values()),
        "pagerank": [pr.get(k, 0) for k in btw.keys()],
        "out_degree": [outdeg.get(k, 0) for k in btw.keys()],
        "in_degree": [indeg.get(k, 0) for k in btw.keys()],
    }).sort_values(["betweenness","pagerank","out_degree"], ascending=False)
    return res
