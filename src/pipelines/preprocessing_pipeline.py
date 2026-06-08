from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def make_one_hot_encoder(
    use_rare_category_grouping: bool = True,
    min_frequency: int | None = 50,
) -> OneHotEncoder:
    """
    Tạo OneHotEncoder.

    Với dataset của bạn, mcc có nhiều category hiếm.
    Vì vậy nên dùng:
    - handle_unknown='infrequent_if_exist'
    - min_frequency=50

    Nếu sklearn version cũ không hỗ trợ sparse_output,
    fallback sang sparse=True.
    """
    params: dict[str, Any] = {}

    try:
        OneHotEncoder(sparse_output=True)
        params["sparse_output"] = True
    except TypeError:
        params["sparse"] = True

    if use_rare_category_grouping and min_frequency is not None:
        params["handle_unknown"] = "infrequent_if_exist"
        params["min_frequency"] = min_frequency
    else:
        params["handle_unknown"] = "ignore"

    return OneHotEncoder(**params)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    binary_features: list[str],
    interaction_features: list[str],
    use_rare_category_grouping: bool = True,
    min_frequency: int | None = 50,
) -> ColumnTransformer:
    """
    Build preprocessing pipeline cho Logistic Regression.

    numeric:
    - fill median
    - StandardScaler

    categorical:
    - fill missing
    - OneHotEncoder
    - gom rare categories nếu bật use_rare_category_grouping

    binary + interaction:
    - fill 0
    - passthrough
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                make_one_hot_encoder(
                    use_rare_category_grouping=use_rare_category_grouping,
                    min_frequency=min_frequency,
                ),
            ),
        ]
    )

    binary_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    if binary_features:
        transformers.append(("binary", binary_pipeline, binary_features))

    if interaction_features:
        transformers.append(("interaction", binary_pipeline, interaction_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )

    return preprocessor


def get_preprocessed_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Lấy tên feature sau preprocessing.
    """
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []