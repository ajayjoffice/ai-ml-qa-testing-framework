import pytest

from app.model import load_model
from training.train import make_dataset


def test_model_loads_and_has_expected_features():
    model = load_model()

    assert list(model.feature_names_in_) == make_dataset()[0].columns.tolist()


def test_prediction_is_binary_and_probability_is_valid():
    model = load_model()
    features, _ = make_dataset()
    prediction = model.predict(features.head(1))
    probabilities = model.predict_proba(features.head(1))

    assert prediction.shape == (1,)
    assert int(prediction[0]) in (0, 1)
    assert probabilities.shape == (1, 2)
    assert probabilities[0].sum() == pytest.approx(1.0)
    assert all(0.0 <= probability <= 1.0 for probability in probabilities[0])


def test_real_dataset_load_is_reproducible_and_has_expected_shape():
    first_features, first_target = make_dataset()
    second_features, second_target = make_dataset()

    assert first_features.equals(second_features)
    assert first_target.equals(second_target)
    assert first_features.shape == (3150, 13)
    assert first_target.isin([0, 1]).all()
