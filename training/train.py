"""Train a simple classifier on the locally saved UCI churn dataset."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.preprocessing import FEATURES
from validation.data_validation import validate_customer_data

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "Customer Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
TARGET = "Churn"
SOURCE_TO_FEATURE = {
    "Call  Failure": "call_failure",
    "Complains": "complains",
    "Subscription  Length": "subscription_length",
    "Charge  Amount": "charge_amount",
    "Seconds of Use": "seconds_of_use",
    "Frequency of use": "frequency_of_use",
    "Frequency of SMS": "frequency_of_sms",
    "Distinct Called Numbers": "distinct_called_numbers",
    "Age Group": "age_group",
    "Tariff Plan": "tariff_plan",
    "Status": "status",
    "Age": "age",
    "Customer Value": "customer_value",
}


def make_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Load and validate the UCI CSV, returning ordered features and labels."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    expected = list(SOURCE_TO_FEATURE) + [TARGET]
    missing = [column for column in expected if column not in raw.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    X = raw[list(SOURCE_TO_FEATURE)].rename(columns=SOURCE_TO_FEATURE)
    X = X[FEATURES]
    validation = validate_customer_data(X, check_duplicates=False)
    if not validation.is_valid:
        raise ValueError("Dataset validation failed: " + "; ".join(validation.errors))
    y = pd.to_numeric(raw[TARGET], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("Churn target must contain only 0 and 1")
    return X, y.astype(int)


def train_pipeline() -> Pipeline:
    """Build a fresh, unfitted preprocessing and classifier pipeline."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )


def train_model() -> dict:
    X, y = make_dataset()
    model = train_pipeline()
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return {"model": model, "rows": len(X), "path": MODEL_PATH}


if __name__ == "__main__":
    result = train_model()
    print(f"Trained LogisticRegression on {result['rows']} UCI churn rows; saved {result['path']}")
