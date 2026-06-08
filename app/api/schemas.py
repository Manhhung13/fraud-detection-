from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class FeatureReadyTransaction(BaseModel):
    """
    Request body cho API predict.

    API hiện tại nhận feature-ready input, tức là các feature đã được xử lý
    giống như dữ liệu trong model_train_features.parquet.

    Ví dụ:
    {
        "features": {
            "mcc_entropy_30d": 0.4,
            "night_ratio_30d": 0.1,
            "log_amount_to_max_30d": 0.8,
            ...
        }
    }
    """

    features: dict[str, Any] = Field(
        ...,
        description="Dictionary chứa các feature model cần để predict.",
    )


class FraudPredictionResponse(BaseModel):
    fraud_probability: float
    prediction: int
    threshold: float
    risk_level: str
    decision: str
    reason: str


class HealthResponse(BaseModel):
    status: str
    model: str | None
    threshold: float