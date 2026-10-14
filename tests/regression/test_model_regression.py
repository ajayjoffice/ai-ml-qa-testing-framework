import json

from sklearn.dummy import DummyClassifier

from training.train import make_dataset
from validation.regression_validation import (
    METRIC_NAMES,
    compare_with_baseline,
    configured_regression_tolerance,
    train_candidate_model,
    write_baseline,
)


def test_candidate_model_passes_regression_against_generated_baseline(tmp_path):
    candidate = train_candidate_model()
    baseline_path = tmp_path / "baseline.json"
    baseline_metrics = write_baseline(baseline_path, model=candidate)

    result = compare_with_baseline(candidate, baseline_path)

    assert result["is_valid"], result["errors"]
    assert set(baseline_metrics) == set(METRIC_NAMES)
    assert result["candidate"]["f1"] >= result["baseline"]["f1"] - result["maximum_f1_drop"]


def test_degraded_candidate_fails_f1_regression_gate(tmp_path):
    good_model = train_candidate_model()
    baseline_path = tmp_path / "baseline.json"
    write_baseline(baseline_path, model=good_model)

    X, y = make_dataset()
    degraded_model = DummyClassifier(strategy="most_frequent").fit(X, y)
    result = compare_with_baseline(
        degraded_model,
        baseline_path,
        maximum_f1_drop=0.02,
        minimum_f1=0.0,
    )

    assert not result["is_valid"]
    assert any("below acceptance threshold" in error for error in result["errors"])


def test_minimum_f1_floor_cannot_be_bypassed_by_regression_tolerance(tmp_path):
    model = train_candidate_model()
    baseline_path = tmp_path / "baseline.json"
    write_baseline(baseline_path, model=model)

    result = compare_with_baseline(
        model, baseline_path, maximum_f1_drop=1.0, minimum_f1=0.99
    )

    assert not result["is_valid"]
    assert "0.990" in result["errors"][0]


def test_missing_baseline_fails_with_clear_message(tmp_path):
    result = compare_with_baseline(train_candidate_model(), tmp_path / "absent.json")

    assert not result["is_valid"]
    assert "Baseline file not found" in result["errors"][0]


def test_malformed_baseline_fails_with_clear_message(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"metrics": {"f1": 0.7}}))

    result = compare_with_baseline(train_candidate_model(), path)

    assert not result["is_valid"]
    assert "Baseline file is invalid" in result["errors"][0]


def test_regression_tolerance_range_is_checked():
    assert configured_regression_tolerance(0.1) == 0.1
