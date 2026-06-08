from __future__ import annotations

from sklearn.pipeline import Pipeline

from src.pipelines.tree_preprocessing_pipeline import build_tree_preprocessor


def build_catboost_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    binary_features: list[str],
    interaction_features: list[str],
    iterations: int = 500,
    learning_rate: float = 0.03,
    depth: int = 6,
    l2_leaf_reg: float = 5.0,
    scale_pos_weight: float | None = None,
    random_state: int = 42,
    use_rare_category_grouping: bool = True,
    min_frequency: int | None = 100,
    use_gpu: bool = True,
    gpu_devices: str = "0",
) -> Pipeline:
    """
    CatBoost GPU pipeline.

    Lưu ý:
    - Bản này vẫn dùng OneHotEncoder trước CatBoost để giữ kiến trúc chung.
    - Nếu muốn CatBoost native categorical, nên làm một pipeline riêng.
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError(
            "Bạn chưa cài catboost. Chạy: pip install catboost"
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
        gpu_params["task_type"] = "GPU"
        gpu_params["devices"] = gpu_devices
    else:
        gpu_params["task_type"] = "CPU"

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        scale_pos_weight=scale_pos_weight,
        random_seed=random_state,
        verbose=100,
        allow_writing_files=False,
        **gpu_params,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )