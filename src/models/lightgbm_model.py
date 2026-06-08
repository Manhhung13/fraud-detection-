from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.pipelines.tree_preprocessing_pipeline import build_tree_preprocessor


def build_lightgbm_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    binary_features: list[str],
    interaction_features: list[str],
    n_estimators: int = 500,
    learning_rate: float = 0.03,
    num_leaves: int = 31,
    max_depth: int = -1,
    min_child_samples: int = 50,
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
    gpu_device_type: str = "gpu",
) -> Pipeline:
    """
    LightGBM GPU pipeline.

    gpu_device_type:
    - "gpu": OpenCL GPU backend
    - "cuda": CUDA backend nếu bản LightGBM cài đặt hỗ trợ

    Nếu LightGBM GPU lỗi trên Windows, thử:
    - use_gpu=False
    - hoặc gpu_device_type="cuda"
    """
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:
        raise ImportError(
            "Bạn chưa cài lightgbm. Chạy: pip install lightgbm"
        ) from exc

    preprocessor = build_tree_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        binary_features=binary_features,
        interaction_features=interaction_features,
        use_rare_category_grouping=use_rare_category_grouping,
        min_frequency=min_frequency,
    )

    extra_params = {}
    if use_gpu:
        extra_params["device_type"] = gpu_device_type
        extra_params["gpu_use_dp"] = False

    model = LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=-1,
        **extra_params,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )