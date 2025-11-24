from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

def prepare_training_table(df: pd.DataFrame, by: str = "departure", delay_threshold: int = 15) -> Tuple[pd.DataFrame, pd.Series]:
    if by == "departure":
        delay = "dep_delay_min"
        hour = "dep_hour"
        slot_window = "slot_window_flights"
    else:
        delay = "arr_delay_min"
        hour = "arr_hour"
        slot_window = "slot_window_flights"

    # Filter rows with delay info
    sdf = df.dropna(subset=[delay]).copy()
    sdf["label_delayed"] = (sdf[delay] >= delay_threshold).astype(int)

    # Features
    X = pd.DataFrame({
        "hour": sdf[hour] if hour in sdf else np.nan,
        "dow": sdf["dep_dow"] if "dep_dow" in sdf else (sdf["arr_dow"] if "arr_dow" in sdf else np.nan),
        "slot_window_flights": sdf[slot_window] if slot_window in sdf else 0,
        "airline": sdf.get("airline", "UNK"),
        "destination": sdf.get("destination", "UNK"),
        "origin": sdf.get("origin", "UNK"),
    })
    y = sdf["label_delayed"]
    return X, y

def train_delay_model(df: pd.DataFrame, by: str = "departure", delay_threshold: int = 15, model_path: str = "reports/delay_model.joblib") -> dict:
    X, y = prepare_training_table(df, by=by, delay_threshold=delay_threshold)

    num_features = ["hour", "dow", "slot_window_flights"]
    cat_features = ["airline", "destination", "origin"]

    pre = ColumnTransformer([
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    clf = LogisticRegression(max_iter=1000, n_jobs=None)
    pipe = Pipeline([("pre", pre), ("clf", clf)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:,1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "report": classification_report(y_test, y_pred, output_dict=False)
    }

    joblib.dump(pipe, model_path)
    return metrics
