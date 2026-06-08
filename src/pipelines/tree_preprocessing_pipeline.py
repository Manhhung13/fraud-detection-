from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def make_tree_one_hot_encoder(
    use_rare_category_grouping: bool = True,
    min_frequency: int | None = 100,
) -> OneHotEncoder:
    """
    OneHotEncoder cho tree models.

    Với dataset hiện tại:
    - mcc có cardinality cao
    - nhiều category hiếm
    => nên dùng rare category grouping.

    min_frequency=100 là mức hợp lý hơn 50 dựa trên experiment Logistic Regression.
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


def build_tree_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    binary_features: list[str],
    interaction_features: list[str],
    use_rare_category_grouping: bool = True,
    min_frequency: int | None = 100,
) -> ColumnTransformer:
    """
    Preprocessing cho Random Forest / LightGBM dùng sklearn pipeline.

    numeric:
    - median imputer
    - không scale

    categorical:
    - fill missing
    - one-hot có rare grouping

    binary / interaction:
    - fill 0
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            (
                "onehot",
                make_tree_one_hot_encoder(
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

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=True,
    )