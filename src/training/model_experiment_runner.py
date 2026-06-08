from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from src.models.model_factory import create_model
from src.pipelines.preprocessing_pipeline import get_preprocessed_feature_names
from src.training.evaluate import (
    confusion_matrix_dataframe,
    evaluate_binary_classifier,
    prediction_dataframe,
)
from src.training.threshold import search_thresholds, select_best_threshold


@dataclass
class ModelExperimentConfig:
    experiment_name: str
    model_name: str
    description: str

    drop_categorical_features: list[str] | None = None
    drop_interaction_features: bool = False

    use_rare_category_grouping: bool = True
    min_frequency: int | None = 100

    threshold_metric: str = "f2"
    max_review_rate: float | None = 0.05

    model_params: dict | None = None


def load_feature_metadata(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy feature metadata: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def filter_metadata(
    metadata: dict,
    config: ModelExperimentConfig,
) -> dict:
    metadata = metadata.copy()

    drop_cats = set(config.drop_categorical_features or [])

    metadata["categorical_features"] = [
        col for col in metadata["categorical_features"]
        if col not in drop_cats
    ]

    if config.drop_interaction_features:
        metadata["interaction_features"] = []

    metadata["all_model_features"] = (
        metadata["numeric_features"]
        + metadata["categorical_features"]
        + metadata["binary_features"]
        + metadata["interaction_features"]
    )

    return metadata


def split_xy(
    df: pd.DataFrame,
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy target column: {target_col}")

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    return X, y


def get_positive_class_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def get_feature_importance(model) -> pd.DataFrame:
    """
    Lấy feature importance nếu model hỗ trợ.

    RandomForest / LightGBM / XGBoost / CatBoost đều có thể có feature_importances_.
    """
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    feature_names = get_preprocessed_feature_names(preprocessor)

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    else:
        return pd.DataFrame()

    if len(feature_names) != len(importances):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    return importance_df.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)


def compute_scale_pos_weight(train_df: pd.DataFrame, target_col: str) -> float:
    """
    Tính scale_pos_weight = số mẫu âm / số mẫu dương.

    Dùng cho các boosting models:
    - LightGBM
    - XGBoost
    - CatBoost

    Với fraud detection mất cân bằng, giá trị này giúp model chú ý hơn vào lớp fraud.
    """
    y = train_df[target_col].astype(int)

    positive = int((y == 1).sum())
    negative = int((y == 0).sum())

    if positive == 0:
        return 1.0

    return float(negative / positive)


def run_model_experiment(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_metadata: dict,
    config: ModelExperimentConfig,
    model_output_root: str | Path,
    report_output_root: str | Path,
) -> dict:
    print("=" * 80)
    print(f"[EXPERIMENT] {config.experiment_name}")
    print(f"[MODEL] {config.model_name}")
    print(config.description)
    print("=" * 80)

    start_time = time.time()

    model_output_dir = Path(model_output_root) / config.experiment_name
    report_output_dir = Path(report_output_root) / config.experiment_name

    model_output_dir.mkdir(parents=True, exist_ok=True)
    report_output_dir.mkdir(parents=True, exist_ok=True)

    metadata = filter_metadata(base_metadata, config)

    target_col = metadata["target_col"]

    X_train, y_train = split_xy(train_df, target_col)
    X_valid, y_valid = split_xy(validation_df, target_col)
    X_test, y_test = split_xy(test_df, target_col)

    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]
    binary_features = metadata["binary_features"]
    interaction_features = metadata["interaction_features"]

    print(f"Numeric features:      {len(numeric_features)}")
    print(f"Categorical features:  {len(categorical_features)}")
    print(f"Binary features:       {len(binary_features)}")
    print(f"Interaction features:  {len(interaction_features)}")

    model_params = dict(config.model_params or {})

    # Boosting models imbalance handling.
    # Convert scale_pos_weight="auto" thành số float.
    # XGBoost không chấp nhận string "auto", nên bắt buộc phải convert trước khi fit.
    model_name_lower = config.model_name.lower().strip()

    if model_name_lower in [
        "lightgbm",
        "lgbm",
        "xgboost",
        "xgb",
        "catboost",
        "cat",
    ]:
        if model_params.get("scale_pos_weight") == "auto":
            model_params["scale_pos_weight"] = compute_scale_pos_weight(
                train_df=train_df,
                target_col=target_col,
            )
            print(f"[INFO] scale_pos_weight={model_params['scale_pos_weight']:.4f}")

    model = create_model(
        model_name=config.model_name,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        binary_features=binary_features,
        interaction_features=interaction_features,
        use_rare_category_grouping=config.use_rare_category_grouping,
        min_frequency=config.min_frequency,
        **model_params,
    )

    print("[TRAIN] Fitting model...")
    model.fit(X_train, y_train)

    preprocessor = model.named_steps["preprocessor"]
    feature_names = get_preprocessed_feature_names(preprocessor)
    n_preprocessed_features = len(feature_names)

    print(f"[INFO] Features after preprocessing: {n_preprocessed_features:,}")

    print("[PREDICT] Predicting probabilities...")
    train_proba = get_positive_class_proba(model, X_train)
    valid_proba = get_positive_class_proba(model, X_valid)
    test_proba = get_positive_class_proba(model, X_test)

    train_metrics_05 = evaluate_binary_classifier(
        y_true=y_train,
        y_proba=train_proba,
        threshold=0.5,
        split_name="train_threshold_05",
    )

    valid_metrics_05 = evaluate_binary_classifier(
        y_true=y_valid,
        y_proba=valid_proba,
        threshold=0.5,
        split_name="validation_threshold_05",
    )

    print("[THRESHOLD] Searching threshold on validation...")
    threshold_report = search_thresholds(
        y_true=y_valid,
        y_proba=valid_proba,
        split_name="validation",
    )

    best_threshold_row = select_best_threshold(
        threshold_report=threshold_report,
        metric=config.threshold_metric,
        max_review_rate=config.max_review_rate,
    )

    best_threshold = float(best_threshold_row["threshold"])
    print(f"[THRESHOLD] Selected threshold: {best_threshold}")

    valid_best_metrics = evaluate_binary_classifier(
        y_true=y_valid,
        y_proba=valid_proba,
        threshold=best_threshold,
        split_name="validation_best_threshold",
    )

    test_metrics = evaluate_binary_classifier(
        y_true=y_test,
        y_proba=test_proba,
        threshold=best_threshold,
        split_name="test",
    )

    print("[SAVE] Saving artifacts...")

    model_path = model_output_dir / "pipeline.joblib"
    joblib.dump(model, model_path)

    save_dataframe(
        pd.DataFrame([train_metrics_05]),
        report_output_dir / "metrics_train_threshold_05.csv",
    )

    save_dataframe(
        pd.DataFrame([valid_metrics_05]),
        report_output_dir / "metrics_validation_threshold_05.csv",
    )

    save_dataframe(
        pd.DataFrame([valid_best_metrics]),
        report_output_dir / "metrics_validation_best_threshold.csv",
    )

    save_dataframe(
        pd.DataFrame([test_metrics]),
        report_output_dir / "metrics_test.csv",
    )

    save_dataframe(
        threshold_report,
        report_output_dir / "threshold_search_validation.csv",
    )

    confusion_matrix_dataframe(
        y_true=y_valid,
        y_proba=valid_proba,
        threshold=best_threshold,
    ).to_csv(
        report_output_dir / "confusion_matrix_validation.csv",
        encoding="utf-8-sig",
    )

    confusion_matrix_dataframe(
        y_true=y_test,
        y_proba=test_proba,
        threshold=best_threshold,
    ).to_csv(
        report_output_dir / "confusion_matrix_test.csv",
        encoding="utf-8-sig",
    )

    prediction_dataframe(
        y_true=y_valid,
        y_proba=valid_proba,
        threshold=best_threshold,
    ).to_csv(
        report_output_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction_dataframe(
        y_true=y_test,
        y_proba=test_proba,
        threshold=best_threshold,
    ).to_csv(
        report_output_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    importance_df = get_feature_importance(model)
    if not importance_df.empty:
        save_dataframe(
            importance_df,
            report_output_dir / "feature_importance.csv",
        )

    elapsed_seconds = time.time() - start_time

    experiment_metadata = {
        "experiment_config": asdict(config),
        "filtered_metadata": metadata,
        "model_path": str(model_path),
        "n_preprocessed_features": n_preprocessed_features,
        "selected_threshold": best_threshold,
        "elapsed_seconds": elapsed_seconds,
        "train_metrics_threshold_05": train_metrics_05,
        "validation_metrics_threshold_05": valid_metrics_05,
        "validation_metrics_best_threshold": valid_best_metrics,
        "test_metrics": test_metrics,
    }

    save_json(
        experiment_metadata,
        model_output_dir / "experiment_metadata.json",
    )

    summary = {
        "experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "description": config.description,
        "dropped_categorical": ",".join(config.drop_categorical_features or []),
        "drop_interaction_features": config.drop_interaction_features,
        "use_rare_category_grouping": config.use_rare_category_grouping,
        "min_frequency": config.min_frequency,
        "selected_threshold": best_threshold,
        "max_review_rate": config.max_review_rate,
        "n_preprocessed_features": n_preprocessed_features,
        "elapsed_seconds": elapsed_seconds,

        "validation_roc_auc": valid_best_metrics["roc_auc"],
        "validation_pr_auc": valid_best_metrics["pr_auc"],
        "validation_precision": valid_best_metrics["precision"],
        "validation_recall": valid_best_metrics["recall"],
        "validation_f1": valid_best_metrics["f1"],
        "validation_f2": valid_best_metrics["f2"],
        "validation_review_rate": valid_best_metrics["review_rate"],

        "test_roc_auc": test_metrics["roc_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_f2": test_metrics["f2"],
        "test_review_rate": test_metrics["review_rate"],
        "test_tp": test_metrics["tp"],
        "test_fp": test_metrics["fp"],
        "test_fn": test_metrics["fn"],
        "test_tn": test_metrics["tn"],
    }

    print(
        f"[DONE] {config.experiment_name}: "
        f"PR-AUC={summary['test_pr_auc']:.4f}, "
        f"F2={summary['test_f2']:.4f}, "
        f"Precision={summary['test_precision']:.4f}, "
        f"Recall={summary['test_recall']:.4f}, "
        f"Review={summary['test_review_rate']:.4f}"
    )

    return summary


def generate_model_comparison_report(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    show_cols = [
        "experiment_name",
        "model_name",
        "dropped_categorical",
        "drop_interaction_features",
        "min_frequency",
        "selected_threshold",
        "n_preprocessed_features",
        "test_pr_auc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_f2",
        "test_review_rate",
        "test_tp",
        "test_fp",
        "test_fn",
        "test_tn",
        "elapsed_seconds",
    ]

    show_cols = [col for col in show_cols if col in comparison_df.columns]

    lines = []

    lines.append("# Tree / GPU Boosting Model Experiment Comparison Report\n")

    lines.append("## 1. Mục tiêu\n")
    lines.append(
        """
So sánh các mô hình mở rộng với Logistic Regression baseline:

- Random Forest.
- LightGBM.
- XGBoost.
- CatBoost.

Tất cả model dùng cùng train/validation/test split, cùng metric, cùng threshold search trên validation.
"""
    )

    lines.append("\n## 2. Bảng so sánh\n")
    lines.append(
        comparison_df[show_cols]
        .sort_values(["test_f2", "test_pr_auc"], ascending=False)
        .to_markdown(index=False)
    )

    lines.append("\n## 3. Cách chọn model\n")
    lines.append(
        """
Ưu tiên model có:
1. PR-AUC cao.
2. F2 cao vì fraud detection ưu tiên recall.
3. Review rate phù hợp năng lực kiểm tra.
4. Precision không quá thấp.
5. Số chiều không quá lớn.
6. Không tăng độ phức tạp quá nhiều nếu hiệu năng chỉ cải thiện rất nhỏ.
"""
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")