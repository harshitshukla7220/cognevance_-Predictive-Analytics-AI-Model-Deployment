from pathlib import Path
import pandas as pd
from src.pipeline import add_features
from src.inference import load_bundle, predict_records, MODEL_PATH

ROOT=Path(__file__).resolve().parents[1]

def test_dataset_exists_and_schema():
    df=pd.read_csv(ROOT/"data/customer_churn.csv")
    assert len(df)>7000 and "Churn" in df.columns

def test_feature_engineering():
    df=pd.read_csv(ROOT/"data/customer_churn.csv").drop(columns=["Churn"]).head(10)
    out=add_features(df)
    assert "customerID" not in out
    assert "tenure_years" in out and "number_of_services" in out

def test_trained_bundle_predicts():
    assert MODEL_PATH.exists()
    bundle=load_bundle()
    df=pd.read_csv(ROOT/"data/customer_churn.csv").drop(columns=["Churn"]).head(1)
    row=df.drop(columns=["customerID"]).iloc[0].to_dict()
    result=predict_records(bundle,[row])[0]
    assert result["prediction"] in ["Yes","No"]
    assert 0 <= result["churn_probability"] <= 1


def test_fastapi_endpoints():
    from fastapi.testclient import TestClient
    from src.app import app
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        info = client.get("/model-info").json()
        assert "customerID" not in info["features"]
        sample = pd.read_csv(ROOT/"data/customer_churn.csv").drop(columns=["customerID","Churn"]).iloc[0].to_dict()
        response = client.post("/predict", json={"features": sample})
        assert response.status_code == 200
        assert response.json()["prediction"] in ["Yes", "No"]
