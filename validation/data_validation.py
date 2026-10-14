"""Validation rules for the UCI churn feature data."""
from dataclasses import dataclass, field

import pandas as pd

from app.preprocessing import FEATURES

NUMERIC_RANGES = {
    "call_failure": (0, 100),
    "complains": (0, 1),
    "subscription_length": (1, 100),
    "charge_amount": (0, 20),
    "seconds_of_use": (0, 30000),
    "frequency_of_use": (0, 500),
    "frequency_of_sms": (0, 1000),
    "distinct_called_numbers": (0, 200),
    "age_group": (1, 5),
    "tariff_plan": (1, 2),
    "status": (1, 2),
    "age": (15, 100),
    "customer_value": (0, 10000),
}
INTEGER_COLUMNS = set(FEATURES) - {"customer_value"}


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"is_valid": self.is_valid, "errors": self.errors}


def validate_customer_data(data: pd.DataFrame, check_duplicates: bool = True) -> ValidationResult:
    """Check feature columns, missing values, types, ranges, and duplicates.

    The source CSV has repeated feature profiles but no customer ID. They may
    represent different customers, so the training loader skips the duplicate
    profile check while still checking schema and values.
    """
    errors: list[str] = []
    missing_columns = [column for column in FEATURES if column not in data.columns]
    extra_columns = [column for column in data.columns if column not in FEATURES]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    if extra_columns:
        errors.append(f"Unexpected columns: {', '.join(extra_columns)}")
    if missing_columns or extra_columns:
        return ValidationResult(False, errors)

    null_columns = data.columns[data.isna().any()].tolist()
    if null_columns:
        errors.append(f"Null values found in: {', '.join(null_columns)}")
    if check_duplicates and data.duplicated().any():
        errors.append("Duplicate rows found")

    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        numeric = pd.to_numeric(data[column], errors="coerce")
        if (numeric.isna() & data[column].notna()).any():
            errors.append(f"{column} must contain numeric values")
            continue
        if column in INTEGER_COLUMNS and ((numeric % 1) != 0).any():
            errors.append(f"{column} must contain whole numbers")
        if ((numeric < minimum) | (numeric > maximum)).any():
            errors.append(f"{column} values must be between {minimum} and {maximum}")

    return ValidationResult(not errors, errors)


def validate_customer_data_or_raise(data: pd.DataFrame) -> None:
    result = validate_customer_data(data)
    if not result.is_valid:
        raise ValueError("Customer data validation failed: " + "; ".join(result.errors))
