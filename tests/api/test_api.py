from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_REQUEST = {
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


def test_health_endpoint_reports_healthy_when_model_is_available():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model_info_endpoint_has_expected_fields():
    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json() == {
        "model_name": "UCI Telecom Churn Classifier",
        "model_version": "0.2.0",
        "model_type": "LogisticRegression",
    }


def test_predict_endpoint_returns_prediction_and_probability():
    response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"prediction", "label", "churn_probability"}
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("stay", "churn")
    assert 0 <= body["churn_probability"] <= 1


def test_predict_endpoint_rejects_missing_field():
    payload = {key: value for key, value in VALID_REQUEST.items() if key != "call_failure"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "call_failure" in response.text


def test_predict_endpoint_rejects_invalid_field_types():
    payload = {**VALID_REQUEST, "call_failure": "a lot"}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_endpoint_rejects_out_of_range_value():
    payload = {**VALID_REQUEST, "call_failure": -1}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_binary_value():
    payload = {**VALID_REQUEST, "complains": 2}

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
