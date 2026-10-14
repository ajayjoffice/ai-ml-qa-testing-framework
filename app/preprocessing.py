"""Feature names and lightweight input-to-dataframe conversion."""
import pandas as pd

FEATURES = [
    "call_failure",
    "complains",
    "subscription_length",
    "charge_amount",
    "seconds_of_use",
    "frequency_of_use",
    "frequency_of_sms",
    "distinct_called_numbers",
    "age_group",
    "tariff_plan",
    "status",
    "age",
    "customer_value",
]


def build_features(records: list[dict]) -> pd.DataFrame:
    """Convert API records into the fixed, ordered model feature frame."""
    return pd.DataFrame(records, columns=FEATURES)
