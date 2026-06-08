from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.pipelines.tree_preprocessing_pipeline import build_tree_preprocessor


def build_xgboost_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    binary_features: list[str],
    interaction_features: list[str],
    n_estimators: int = 500,
    learning_rate: float = 0.03,
    max_depth: int = 5,
    min_child_weight: float = 5.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    scale_pos_weight: float | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
    use_rare_category_grouping: bool = True,
    min_frequency: int | None = 100,
    use_gpu: bool = True,
) -> Pipeline:
    """
    XGBoost GPU pipeline.

    Với XGBoost version mới:
    - tree_method='hist'
    - device='cuda'

    Nếu GPU lỗi, set use_gpu=False để chạy CPU.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "Bạn chưa cài xgboost. Chạy: pip install xgboost"
        ) from exc

    preprocessor = build_tree_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        binary_features=binary_features,
        interaction_features=interaction_features,
        use_rare_category_grouping=use_rare_category_grouping,
        min_frequency=min_frequency,
    )

    gpu_params = {}
    if use_gpu:
        gpu_params["tree_method"] = "hist"
        gpu_params["device"] = "cuda"
    else:
        gpu_params["tree_method"] = "hist"
        gpu_params["device"] = "cpu"

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=1,
        **gpu_params,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )