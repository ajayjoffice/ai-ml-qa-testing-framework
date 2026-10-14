"""Compare candidate model quality against a saved evaluation baseline."""
import json
import os
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from training.train import make_dataset, train_pipeline

ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "reports" / "baseline_metrics.json"
METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc")


def configured_regression_tolerance(value: float | None = None) -> float:
    tolerance = float(value if value is not None else os.getenv("MAX_F1_DROP", "0.02"))
    if not 0 <= tolerance <= 1:
        raise ValueError("Maximum F1 drop must be between 0 and 1")
    return tolerance


def evaluation_data():
    X, y = make_dataset()
    return train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


def calculate_metrics(model) -> dict[str, float]:
    _, X_test, _, y_test = evaluation_data()
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
    }


def train_candidate_model():
    """Fit a fresh model using only the deterministic training split."""
    X_train, _, y_train, _ = evaluation_data()
    model = train_pipeline()
    model.fit(X_train, y_train)
    return model


def write_baseline(path: str | Path = BASELINE_PATH, model=None) -> dict[str, float]:
    """Create a baseline on the fixed holdout from a model or fresh candidate."""
    path = Path(path)
    if model is None:
        model = train_candidate_model()
    metrics = calculate_metrics(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model_version": "0.2.0", "metrics": metrics}, indent=2) + "\n")
    return metrics


def compare_with_baseline(
    candidate_model,
    baseline_path: str | Path = BASELINE_PATH,
    maximum_f1_drop: float | None = None,
    minimum_f1: float | None = None,
) -> dict:
    """Return candidate and baseline metrics with pass/fail and explanations."""
    path = Path(baseline_path)
    if not path.exists():
        return {"is_valid": False, "errors": [f"Baseline file not found: {path}"]}
    try:
        baseline = json.loads(path.read_text())["metrics"]
        if any(name not in baseline for name in METRIC_NAMES):
            raise ValueError("Baseline is missing one or more required metrics")
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {"is_valid": False, "errors": [f"Baseline file is invalid: {exc}"]}

    tolerance = configured_regression_tolerance(maximum_f1_drop)
    floor = float(minimum_f1 if minimum_f1 is not None else os.getenv("MINIMUM_F1", "0.60"))
    if not 0 <= floor <= 1:
        raise ValueError("Minimum F1 threshold must be between 0 and 1")
    candidate = calculate_metrics(candidate_model)
    errors = []
    allowed_f1 = max(floor, float(baseline["f1"]) - tolerance)
    if candidate["f1"] < allowed_f1:
        errors.append(
            f"Candidate F1 {candidate['f1']:.3f} is below acceptance threshold {allowed_f1:.3f} "
            f"(baseline {baseline['f1']:.3f}, maximum drop {tolerance:.3f}, minimum {floor:.3f})"
        )
    return {
        "is_valid": not errors,
        "baseline": {name: float(baseline[name]) for name in METRIC_NAMES},
        "candidate": candidate,
        "maximum_f1_drop": tolerance,
        "minimum_f1": floor,
        "errors": errors,
    }
