from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.inference import load_bundle, predict_records, MODEL_PATH

bundle = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    if MODEL_PATH.exists():
        bundle = load_bundle()
    yield
    bundle = None

app = FastAPI(title="Customer Churn Prediction API", version="1.0.0", lifespan=lifespan)

class PredictionRequest(BaseModel):
    features: dict

class BatchRequest(BaseModel):
    records: list[dict] = Field(min_length=1, max_length=500)

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bundle is not None}

@app.get("/model-info")
def model_info():
    if bundle is None:
        raise HTTPException(503, "Model not trained. Run python -m src.train first.")
    return {"model": bundle["model_name"], "features": bundle["feature_columns"], "threshold": bundle["threshold"]}

@app.post("/predict")
def predict(req: PredictionRequest):
    if bundle is None:
        raise HTTPException(503, "Model not trained. Run python -m src.train first.")
    try:
        return predict_records(bundle, [req.features])[0]
    except ValueError as e:
        raise HTTPException(422, str(e))

@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    if bundle is None:
        raise HTTPException(503, "Model not trained. Run python -m src.train first.")
    try:
        return {"count": len(req.records), "predictions": predict_records(bundle, req.records)}
    except ValueError as e:
        raise HTTPException(422, str(e))
