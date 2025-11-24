from typing import Dict, Any
import re

def intent_and_params(query: str) -> Dict[str, Any]:
    q = query.lower().strip()

    # Simple intents
    if re.search(r"\b(best|least|low(est)?)\b.*\b(time|slot)\b.*\b(takeoff|depart|departure)", q):
        return {"intent": "best_departure_time", "by": "departure"}
    if re.search(r"\b(best|least|low(est)?)\b.*\b(time|slot)\b.*\b(land|arrival|arrive)", q):
        return {"intent": "best_arrival_time", "by": "arrival"}
    if re.search(r"\b(busiest|avoid)\b.*\b(slot|time)", q):
        return {"intent": "busiest_slots"}
    if re.search(r"\b(simulate|what if|shift)\b", q):
        # try to extract flight number and minutes
        m1 = re.search(r"flight\s+([A-Za-z0-9-]+)", q)
        m2 = re.search(r"shift.*?(\-?\d+)\s*min", q)
        return {"intent": "simulate", "flight": m1.group(1) if m1 else None, "shift": int(m2.group(1)) if m2 else 15}
    if re.search(r"\b(cascade|cascading|propagate)\b", q):
        return {"intent": "cascade"}
    return {"intent": "unknown"}
