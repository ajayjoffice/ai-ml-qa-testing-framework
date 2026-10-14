from pathlib import Path

import pytest
from sklearn.dummy import DummyClassifier

from app.model import load_model
from training.train import make_dataset
from validation.model_validation import (
    configured_minimum_f1,
    evaluate_model,
    validate_model_file,
)


def test_saved_model_passes_quality_validation():
    result = validate_model_file(minimum_f1=0.60)

    assert result.is_valid, result.errors
    assert result.metrics["f1"] >= result.metrics["minimum_f1"]
    assert 0 <= result.metrics["f1"] <= 1


def test_model_has_expected_feature_names_and_predicts():
    model = load_model()
    features, _ = make_dataset()
    features = features.head(4)

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    assert list(model.feature_names_in_) == make_dataset()[0].columns.tolist()
    assert len(predictions) == 4
    assert probabilities.shape == (4, 2)


def test_model_below_configured_f1_threshold_fails():
    features, target = make_dataset()
    weak_model = DummyClassifier(strategy="most_frequent").fit(features, target)

    result = evaluate_model(weak_model, minimum_f1=0.99)

    assert not result.is_valid
    assert any("below minimum" in error for error in result.errors)


def test_corrupt_model_file_returns_failure(tmp_path: Path):
    corrupt_file = tmp_path / "corrupt-model.joblib"
    corrupt_file.write_text("not a model")

    result = validate_model_file(corrupt_file)

    assert not result.is_valid
    assert any("could not be loaded" in error for error in result.errors)


@pytest.mark.parametrize("threshold", [-0.01, 1.01])
def test_threshold_outside_zero_to_one_is_rejected(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        configured_minimum_f1(threshold)
