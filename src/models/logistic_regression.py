from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.pipelines.preprocessing_pipeline import build_preprocessor


def build_logistic_regression_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    binary_features: list[str],
    interaction_features: list[str],
    class_weight: str | dict | None = "balanced",
    C: float = 1.0,
    max_iter: int = 3000,
    random_state: int = 42,
    use_rare_category_grouping: bool = True,
    min_frequency: int | None = 50,
) -> Pipeline:
    """
    Build Logistic Regression baseline.

    class_weight='balanced':
    - phù hợp dữ liệu fraud bị mất cân bằng

    OneHotEncoder:
    - use_rare_category_grouping=True
    - min_frequency=50
    để gom mcc/categorical hiếm.
    """
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        binary_features=binary_features,
        interaction_features=interaction_features,
        use_rare_category_grouping=use_rare_category_grouping,
        min_frequency=min_frequency,
    )

    model = LogisticRegression(
        penalty="l2",
        C=C,
        class_weight=class_weight,
        solver="liblinear",
        max_iter=max_iter,
        random_state=random_state,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline