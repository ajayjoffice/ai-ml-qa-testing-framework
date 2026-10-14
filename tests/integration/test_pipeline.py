import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.model import load_model
from app.preprocessing import build_features
from validation.data_validation import validate_customer_data

client = TestClient(app)

VALID_RECORD = {
    "call_failure": 8,
    "complains": 0,
    "subscription_length": 38,
    "charge_amount": 0,
    "seconds_of_use": 4370,
    "frequency_of_use": 71,
    "frequency_of_sms": 5,
    "distinct_called_numbers": 17,
    "age_group": 3,
    "tariff_plan": 1,
    "status": 1,
    "age": 30,
    "customer_value": 197.64,
}


def test_data_validation_preprocessing_and_model_work_as_one_pipeline():
    input_frame = pd.DataFrame([VALID_RECORD])
    validation = validate_customer_data(input_frame)
    model_features = build_features(input_frame.to_dict(orient="records"))
    predictions = load_model().predict(model_features)

    assert validation.is_valid, validation.errors
    assert model_features.shape == (1, 13)
    assert int(predictions[0]) in (0, 1)


def test_api_prediction_matches_direct_model_prediction():
    response = client.post("/predict", json=VALID_RECORD)
    direct = int(load_model().predict(build_features([VALID_RECORD]))[0])

    assert response.status_code == 200
    assert response.json()["prediction"] == direct
    assert response.json()["label"] == ("churn" if direct == 1 else "stay")


@pytest.mark.parametrize(
    "invalid_record",
    [
        {**VALID_RECORD, "call_failure": -1},
        {**VALID_RECORD, "subscription_length": 1.5},
        {**VALID_RECORD, "complains": 2},
    ],
)
def test_invalid_business_values_are_rejected_before_inference(invalid_record):
    response = client.post("/predict", json=invalid_record)

    assert response.status_code == 422


def test_empty_json_request_is_rejected():
    response = client.post("/predict", json={})

    assert response.status_code == 422
    assert "detail" in response.json()


def test_non_json_body_is_rejected():
    response = client.post("/predict", content="not-json", headers={"Content-Type": "application/json"})

    assert response.status_code == 422


def test_unavailable_model_fails_health_and_prediction_safely(monkeypatch):
    def unavailable_model():
        raise FileNotFoundError("model missing")

    monkeypatch.setattr("app.main.load_model", unavailable_model)

    health = client.get("/health")
    prediction = client.post("/predict", json=VALID_RECORD)

    assert health.status_code == 503
    assert health.json()["detail"] == "Model is unavailable"
    assert prediction.status_code == 503
    assert prediction.json()["detail"] == "Model is unavailable"


def test_unavailable_model_is_not_called_for_invalid_api_input(monkeypatch):
    def unexpected_model_call():
        raise AssertionError("model should not be loaded for a schema-invalid request")

    monkeypatch.setattr("app.main.load_model", unexpected_model_call)
    response = client.post("/predict", json={**VALID_RECORD, "customer_value": "unknown"})

    assert response.status_code == 422
