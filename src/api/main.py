from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI
import pandas as pd

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    PredictionResponse,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_DIR / "models" / "hgb_churn_model_v1.joblib"

model_artifact = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_artifact.update(joblib.load(MODEL_PATH))
    yield
    model_artifact.clear()


app = FastAPI(
    title="KKBox Churn Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "kkbox-churn-prediction",
        "model_loaded": "model" in model_artifact,
    }

def get_risk_action(probability: float) -> tuple[str, str]:
    if probability < 0.05:
        return "0-5%", "no_discount"

    if probability < 0.13:
        return "5-13%", "no_discount_under_current_threshold"

    if probability < 0.30:
        return "13-30%", "low_cost_reminder"

    if probability < 0.60:
        return "30-60%", "moderate_retention_offer"

    return "60-100%", "strongest_retention_action"

def build_prediction_response(
    probability: float,
    threshold: float,
) -> PredictionResponse:
    predicted_churn = int(probability >= threshold)
    risk_band, suggested_action = get_risk_action(probability)

    return PredictionResponse(
        churn_probability=probability,
        prediction_threshold=threshold,
        predicted_churn=predicted_churn,
        risk_band=risk_band,
        suggested_action=suggested_action,
    )

@app.post("/predict", response_model=PredictionResponse)
def predict_churn(features: CustomerFeatures):
    model = model_artifact["model"]
    threshold = float(model_artifact["threshold"])

    input_data = pd.DataFrame([features.model_dump()])

    probability = float(model.predict_proba(input_data)[:, 1][0])

    return build_prediction_response(
        probability=probability,
        threshold=threshold,
    )

@app.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
)
def predict_churn_batch(
    request: BatchPredictionRequest,
):
    model = model_artifact["model"]
    threshold = float(model_artifact["threshold"])

    input_data = pd.DataFrame(
        [
            customer.model_dump()
            for customer in request.customers
        ]
    )

    probabilities = model.predict_proba(input_data)[:, 1]

    predictions = [
        build_prediction_response(
            probability=float(probability),
            threshold=threshold,
        )
        for probability in probabilities
    ]

    return BatchPredictionResponse(
        count=len(predictions),
        predictions=predictions,
    )