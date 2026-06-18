import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


PROJECT_DIR = Path(__file__).resolve().parents[1]
SAMPLE_REQUEST_PATH = PROJECT_DIR / "examples" / "sample_request.json"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_customer():
    return json.loads(
        SAMPLE_REQUEST_PATH.read_text(encoding="utf-8")
    )


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_valid_prediction(client, valid_customer):
    response = client.post(
        "/predict",
        json=valid_customer,
    )

    body = response.json()

    assert response.status_code == 200
    assert 0 <= body["churn_probability"] <= 1
    assert body["prediction_threshold"] == 0.13
    assert body["predicted_churn"] in [0, 1]
    assert "risk_band" in body
    assert "suggested_action" in body


def test_negative_transaction_count_is_rejected(
    client,
    valid_customer,
):
    invalid_customer = valid_customer.copy()
    invalid_customer["transaction_count"] = -5

    response = client.post(
        "/predict",
        json=invalid_customer,
    )

    assert response.status_code == 422

def test_batch_prediction(client, valid_customer):
    response = client.post(
        "/predict-batch",
        json={
            "customers": [
                valid_customer,
                valid_customer,
            ]
        },
    )

    body = response.json()

    assert response.status_code == 200
    assert body["count"] == 2
    assert len(body["predictions"]) == 2

    for prediction in body["predictions"]:
        assert 0 <= prediction["churn_probability"] <= 1
        assert prediction["prediction_threshold"] == 0.13
        assert prediction["predicted_churn"] in [0, 1]


def test_empty_batch_is_rejected(client):
    response = client.post(
        "/predict-batch",
        json={"customers": []},
    )

    assert response.status_code == 422

def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "online"
    assert response.json()["documentation"] == "/docs"