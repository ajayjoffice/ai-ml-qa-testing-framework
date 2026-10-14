"""Quality checks for a trained churn model."""
import os
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from app.preprocessing import FEATURES
from training.train import MODEL_PATH, make_dataset


@dataclass
class ModelValidationResult:
    is_valid: bool
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"is_valid": self.is_valid, "metrics": self.metrics, "errors": self.errors}


def configured_minimum_f1(value: float | None = None) -> float:
    """Return the F1 threshold from an argument, environment, or default."""
    if value is not None:
        threshold = float(value)
    else:
        threshold = float(os.getenv("MINIMUM_F1", "0.60"))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Minimum F1 threshold must be between 0 and 1")
    return threshold


def evaluate_model(model, minimum_f1: float | None = None) -> ModelValidationResult:
    """Check feature compatibility, inference outputs, and held-out F1."""
    threshold = configured_minimum_f1(minimum_f1)
    errors: list[str] = []
    metrics: dict[str, float] = {}
    try:
        model_features = list(model.feature_names_in_)
    except AttributeError:
        model_features = []
        errors.append("Model does not expose expected input feature names")
    if model_features and model_features != FEATURES:
        errors.append(f"Model feature mismatch: expected {FEATURES}, received {model_features}")

    X, y = make_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    # The persisted model was fit on the full deterministic dataset. A fresh
    # evaluation model is fit only on the training split to avoid leakage.
    from training.train import train_pipeline

    evaluation_model = train_pipeline()
    evaluation_model.fit(X_train, y_train)
    predictions = np.asarray(evaluation_model.predict(X_test))
    probabilities = np.asarray(evaluation_model.predict_proba(X_test))
    if predictions.ndim != 1 or not np.isin(predictions, [0, 1]).all():
        errors.append("Predictions must be a one-dimensional array containing only 0 or 1")
    if probabilities.shape != (len(X_test), 2):
        errors.append("Probability output must have two class probabilities per row")
    elif not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        errors.append("Probability outputs must be finite values between 0 and 1")
    else:
        metrics["f1"] = float(f1_score(y_test, predictions))
        metrics["minimum_f1"] = threshold
        if metrics["f1"] < threshold:
            errors.append(f"F1 score {metrics['f1']:.3f} is below minimum {threshold:.3f}")
    return ModelValidationResult(not errors, metrics, errors)


def validate_model_file(path: str | Path = MODEL_PATH, minimum_f1: float | None = None) -> ModelValidationResult:
    """Load a persisted model and run quality checks, reporting load failures."""
    try:
        model = joblib.load(path)
    except Exception as exc:
        return ModelValidationResult(False, errors=[f"Model could not be loaded: {exc}"])
    return evaluate_model(model, minimum_f1)
