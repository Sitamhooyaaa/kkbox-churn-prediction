from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    has_member_data: Literal[0, 1]
    age_clean: float | None = Field(default=None, ge=13, le=80)
    age_missing_or_invalid: Literal[0, 1]
    gender: Literal["male", "female", "unknown", "no_member_data"]
    city: int
    registered_via: int
    registration_tenure_days: float | None = Field(default=None, ge=0)
    registration_after_cutoff_flag: Literal[0, 1]

    transaction_count: int = Field(ge=0)
    cancel_count: int = Field(ge=0)
    auto_renew_count: int = Field(ge=0)
    zero_payment_or_plan_count: int = Field(ge=0)
    unique_payment_method_count: int = Field(ge=0)
    total_actual_amount_paid: float = Field(ge=0)
    mean_actual_amount_paid: float = Field(ge=0)
    max_actual_amount_paid: float = Field(ge=0)
    mean_plan_list_price: float = Field(ge=0)
    mean_payment_plan_days: float = Field(ge=0)
    discount_transaction_count: int = Field(ge=0)
    manual_renew_count: int = Field(ge=0)

    latest_payment_method_id: int
    latest_payment_plan_days: float
    latest_plan_list_price: float
    latest_actual_amount_paid: float
    latest_is_auto_renew: Literal[-999, 0, 1]
    latest_is_cancel: Literal[-999, 0, 1]
    latest_discount_amount: float
    latest_transaction_days_before_cutoff: float
    latest_membership_expire_days_after_cutoff: float
    has_transaction_data: Literal[0, 1]

class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1)
    prediction_threshold: float = Field(ge=0, le=1)
    predicted_churn: Literal[0, 1]
    risk_band: str
    suggested_action: str

class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(
        min_length=1,
        max_length=1000,
    )


class BatchPredictionResponse(BaseModel):
    count: int = Field(ge=1)
    predictions: list[PredictionResponse]