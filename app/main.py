from fastapi import FastAPI, HTTPException

from app.model import load_model
from app.preprocessing import build_features
from app.schemas import CustomerFeatures, PredictionResponse
from validation.data_validation import validate_customer_data

app = FastAPI(title="AI/ML QA Testing Framework", version="0.2.0")
MODEL_VERSION = "0.2.0"


@app.get("/health")
def health() -> dict[str, str]:
    try:
        load_model()
    except (FileNotFoundError, OSError):
        raise HTTPException(status_code=503, detail="Model is unavailable")
    return {"status": "healthy"}


@app.get("/model-info")
def model_info() -> dict[str, str]:
    return {"model_name": "UCI Telecom Churn Classifier", "model_version": MODEL_VERSION, "model_type": "LogisticRegression"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    try:
        model = load_model()
        frame = build_features([customer.model_dump()])
        validation = validate_customer_data(frame)
        if not validation.is_valid:
            raise HTTPException(status_code=422, detail=validation.errors)
        prediction = int(model.predict(frame)[0])
        probability = float(model.predict_proba(frame)[0][1])
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(status_code=503, detail="Model is unavailable") from exc
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Prediction input could not be processed: {exc}") from exc
    return PredictionResponse(
        prediction=prediction,
        label="churn" if prediction == 1 else "stay",
        churn_probability=probability,
    )
