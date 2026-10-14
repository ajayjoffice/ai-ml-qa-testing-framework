import pandas as pd

from app.preprocessing import FEATURES, build_features

SAMPLE = {
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


def test_build_features_preserves_model_feature_order():
    frame = build_features([SAMPLE])

    assert list(frame.columns) == FEATURES
    assert frame.iloc[0].to_dict() == SAMPLE


def test_build_features_accepts_multiple_records():
    frame = build_features([SAMPLE, {**SAMPLE, "subscription_length": 39}])

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape == (2, len(FEATURES))
