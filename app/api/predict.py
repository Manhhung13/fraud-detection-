from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Request

from app.api.schemas import FeatureReadyTransaction, FraudPredictionResponse


router = APIRouter(
    tags=["Prediction"],
)


@router.post("/predict", response_model=FraudPredictionResponse)
def predict(payload: FeatureReadyTransaction, request: Request):
    """
    Predict fraud probability từ feature-ready transaction.

    Lưu ý:
    - Endpoint này chưa nhận raw transaction.
    - Input phải là các feature đã được xử lý đúng như lúc train model.
    """
    predictor = request.app.state.predictor

    input_df = pd.DataFrame([payload.features])
    result = predictor.predict(input_df).iloc[0].to_dict()

    return result