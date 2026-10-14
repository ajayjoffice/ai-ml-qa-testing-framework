import pandas as pd
import pytest

from training.train import make_dataset
from validation.data_validation import validate_customer_data, validate_customer_data_or_raise

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


@pytest.fixture
def valid_data():
    return pd.DataFrame([SAMPLE, {**SAMPLE, "subscription_length": 39}])


def test_uci_customer_data_passes_schema_and_value_checks():
    features, _ = make_dataset()

    result = validate_customer_data(features, check_duplicates=False)

    assert result.is_valid, result.errors


def test_valid_customer_data_passes(valid_data):
    result = validate_customer_data(valid_data)
    assert result.is_valid
    assert result.errors == []


def test_missing_required_column_fails(valid_data):
    result = validate_customer_data(valid_data.drop(columns=["status"]))
    assert not result.is_valid
    assert any("Missing required columns: status" in error for error in result.errors)


def test_unexpected_column_fails(valid_data):
    valid_data["extra"] = 1
    result = validate_customer_data(valid_data)
    assert not result.is_valid
    assert any("Unexpected columns: extra" in error for error in result.errors)


def test_null_value_fails(valid_data):
    valid_data.loc[0, "customer_value"] = None
    result = validate_customer_data(valid_data)
    assert not result.is_valid
    assert any("Null values found in: customer_value" in error for error in result.errors)


def test_duplicate_rows_fail(valid_data):
    duplicate = pd.concat([valid_data, valid_data.iloc[[0]]], ignore_index=True)
    result = validate_customer_data(duplicate)
    assert not result.is_valid
    assert "Duplicate rows found" in result.errors


@pytest.mark.parametrize(
    "column,value,expected_error",
    [
        ("call_failure", -1, "call_failure values must be between 0 and 100"),
        ("age_group", 6, "age_group values must be between 1 and 5"),
        ("subscription_length", 1.5, "subscription_length must contain whole numbers"),
        ("status", "active", "status must contain numeric values"),
    ],
)
def test_invalid_numeric_value_fails(valid_data, column, value, expected_error):
    valid_data[column] = valid_data[column].astype(object)
    valid_data.loc[0, column] = value
    result = validate_customer_data(valid_data)
    assert not result.is_valid
    assert expected_error in result.errors


def test_raise_helper_includes_validation_details(valid_data):
    valid_data.loc[0, "age"] = 12
    with pytest.raises(ValueError, match="age values must be between 15 and 100"):
        validate_customer_data_or_raise(valid_data)
