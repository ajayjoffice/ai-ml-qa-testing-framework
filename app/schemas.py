from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """Input fields from the UCI Iranian Churn dataset."""

    call_failure: int = Field(ge=0, le=100)
    complains: int = Field(ge=0, le=1)
    subscription_length: int = Field(ge=1, le=100)
    charge_amount: int = Field(ge=0, le=20)
    seconds_of_use: int = Field(ge=0, le=30000)
    frequency_of_use: int = Field(ge=0, le=500)
    frequency_of_sms: int = Field(ge=0, le=1000)
    distinct_called_numbers: int = Field(ge=0, le=200)
    age_group: int = Field(ge=1, le=5)
    tariff_plan: int = Field(ge=1, le=2)
    status: int = Field(ge=1, le=2)
    age: int = Field(ge=15, le=100)
    customer_value: float = Field(ge=0, le=10000)


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    churn_probability: float = Field(ge=0, le=1)
