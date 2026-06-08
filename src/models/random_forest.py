from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.pipelines.tree_preprocessing_pipeline import build_tree_preprocessor


def build_random_forest_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    binary_features: list[str],
    interaction_features: list[str],
    n_estimators: int = 300,
    max_depth: int | None = 12,
    min_samples_leaf: int = 20,
    max_features: str | float | None = "sqrt",
    class_weight: str | dict | None = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
    use_rare_category_grouping: bool = True,
    min_frequency: int | None = 100,
) -> Pipeline:
    """
    Random Forest baseline cho fraud detection.

    Thiết kế chống overfit:
    - max_depth giới hạn
    - min_samples_leaf tương đối lớn
    - class_weight='balanced'
    - rare category grouping cho categorical
    """
    preprocessor = build_tree_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        binary_features=binary_features,
        interaction_features=interaction_features,
        use_rare_category_grouping=use_rare_category_grouping,
        min_frequency=min_frequency,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )