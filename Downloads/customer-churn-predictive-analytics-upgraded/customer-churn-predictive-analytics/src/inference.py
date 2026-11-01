from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/best_model.joblib"

def load_bundle(path=MODEL_PATH):
    return joblib.load(path)

def predict_records(bundle, records):
    frame = pd.DataFrame(records)
    missing = [c for c in bundle["feature_columns"] if c not in frame.columns]
    if missing:
        raise ValueError("Missing required input fields: " + ", ".join(missing))
    frame = frame[bundle["feature_columns"]]
    probabilities = bundle["pipeline"].predict_proba(__import__("src.pipeline", fromlist=["add_features"]).add_features(frame))[:, 1]
    threshold = float(bundle["threshold"])
    return [{
        "prediction": "Yes" if p >= threshold else "No",
        "churn_probability": round(float(p), 6),
        "retention_probability": round(float(1-p), 6),
        "threshold": threshold
    } for p in probabilities]
