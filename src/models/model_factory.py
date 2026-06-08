from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.models.logistic_regression import build_logistic_regression_pipeline
from src.models.random_forest import build_random_forest_pipeline


def create_model(
    model_name: str,
    numeric_features: list[str],
    categorical_features: list[str],
    binary_features: list[str],
    interaction_features: list[str],
    **kwargs,
) -> Pipeline:
    """
    Factory tạo model.

    Hỗ trợ:
    - logistic_regression
    - random_forest
    - lightgbm
    - xgboost
    - catboost
    """
    model_name = model_name.lower().strip()

    if model_name in ["logistic_regression", "logreg", "lr"]:
        return build_logistic_regression_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            binary_features=binary_features,
            interaction_features=interaction_features,
            **kwargs,
        )

    if model_name in ["random_forest", "rf"]:
        return build_random_forest_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            binary_features=binary_features,
            interaction_features=interaction_features,
            **kwargs,
        )

    if model_name in ["lightgbm", "lgbm"]:
        from src.models.lightgbm_model import build_lightgbm_pipeline

        return build_lightgbm_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            binary_features=binary_features,
            interaction_features=interaction_features,
            **kwargs,
        )

    if model_name in ["xgboost", "xgb"]:
        from src.models.xgboost_model import build_xgboost_pipeline

        return build_xgboost_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            binary_features=binary_features,
            interaction_features=interaction_features,
            **kwargs,
        )

    if model_name in ["catboost", "cat"]:
        from src.models.catboost_model import build_catboost_pipeline

        return build_catboost_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            binary_features=binary_features,
            interaction_features=interaction_features,
            **kwargs,
        )

    raise ValueError(f"Model chưa được hỗ trợ: {model_name}")