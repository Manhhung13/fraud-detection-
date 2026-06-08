from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, asdict
import json

import joblib
import numpy as np
import pandas as pd

from src.models.model_factory import create_model
from src.pipelines.preprocessing_pipeline import get_preprocessed_feature_names
from src.training.evaluate import (
    evaluate_binary_classifier,
    confusion_matrix_dataframe,
    prediction_dataframe,
)
from src.training.threshold import search_thresholds, select_best_threshold


@dataclass
class ExperimentConfig:
    experiment_name: str
    description: str
    class_weight: str | dict | None = "balanced"
    C: float = 1.0
    use_rare_category_grouping: bool = True
    min_frequency: int | None = 50
    drop_categorical_features: list[str] | None = None
    threshold_metric: str = "f2"
    max_review_rate: float | None = 0.05


def load_feature_metadata(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy feature metadata: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def filter_metadata_for_experiment(
    metadata: dict,
    config: ExperimentConfig,
) -> dict:
    """
    Tạo metadata riêng cho từng experiment.

    Ví dụ:
    - drop mcc khỏi categorical_features
    - giữ nguyên các nhóm feature khác
    """
    metadata = metadata.copy()

    drop_cols = set(config.drop_categorical_features or [])

    categorical_features = [
        col for col in metadata["categorical_features"]
        if col not in drop_cols
    ]

    metadata["categorical_features"] = categorical_features

    all_model_features = (
        metadata["numeric_features"]
        + metadata["categorical_features"]
        + metadata["binary_features"]
        + metadata["interaction_features"]
    )

    metadata["all_model_features"] = all_model_features

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


def extract_logistic_regression_coefficients(model) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    feature_names = get_preprocessed_feature_names(preprocessor)

    if len(feature_names) == 0:
        feature_names = [f"feature_{i}" for i in range(classifier.coef_.shape[1])]

    coefficients = classifier.coef_[0]

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
            "direction": np.where(
                coefficients >= 0,
                "increase_risk",
                "decrease_risk",
            ),
        }
    )

    return coef_df.sort_values(
        "abs_coefficient",
        ascending=False,
    ).reset_index(drop=True)


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def run_single_experiment(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_metadata: dict,
    config: ExperimentConfig,
    model_output_root: str | Path,
    report_output_root: str | Path,
) -> dict:
    """
    Chạy một experiment Logistic Regression.

    Output mỗi experiment được lưu riêng:
    - models/experiments/<experiment_name>/
    - data/reports/model/experiments/<experiment_name>/
    """
    print("=" * 80)
    print(f"[EXPERIMENT] {config.experiment_name}")
    print(config.description)
    print("=" * 80)

    model_output_dir = Path(model_output_root) / config.experiment_name
    report_output_dir = Path(report_output_root) / config.experiment_name

    model_output_dir.mkdir(parents=True, exist_ok=True)
    report_output_dir.mkdir(parents=True, exist_ok=True)

    metadata = filter_metadata_for_experiment(
        metadata=base_metadata,
        config=config,
    )

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

    print("[MODEL] Building pipeline...")
    model = create_model(
        model_name="logistic_regression",
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        binary_features=binary_features,
        interaction_features=interaction_features,
        class_weight=config.class_weight,
        C=config.C,
        use_rare_category_grouping=config.use_rare_category_grouping,
        min_frequency=config.min_frequency,
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

    print("[EVAL] Threshold 0.5 metrics...")
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

    print("[COEF] Extracting coefficients...")
    coef_df = extract_logistic_regression_coefficients(model)

    coef_no_mcc_df = coef_df[
        ~coef_df["feature"].astype(str).str.startswith("categorical__mcc_")
    ].copy()

    print("[SAVE] Saving experiment artifacts...")

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

    save_dataframe(
        coef_df,
        report_output_dir / "coefficients.csv",
    )

    save_dataframe(
        coef_no_mcc_df,
        report_output_dir / "coefficients_excluding_mcc.csv",
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

    experiment_metadata = {
        "experiment_config": asdict(config),
        "filtered_metadata": metadata,
        "model_path": str(model_path),
        "n_preprocessed_features": n_preprocessed_features,
        "selected_threshold": best_threshold,
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
        "description": config.description,
        "class_weight": config.class_weight,
        "C": config.C,
        "use_rare_category_grouping": config.use_rare_category_grouping,
        "min_frequency": config.min_frequency,
        "dropped_categorical": ",".join(config.drop_categorical_features or []),
        "threshold_metric": config.threshold_metric,
        "max_review_rate": config.max_review_rate,
        "selected_threshold": best_threshold,
        "n_preprocessed_features": n_preprocessed_features,

        "validation_roc_auc": valid_best_metrics["roc_auc"],
        "validation_pr_auc": valid_best_metrics["pr_auc"],
        "validation_precision": valid_best_metrics["precision"],
        "validation_recall": valid_best_metrics["recall"],
        "validation_f1": valid_best_metrics["f1"],
        "validation_f2": valid_best_metrics["f2"],
        "validation_review_rate": valid_best_metrics["review_rate"],
        "validation_tp": valid_best_metrics["tp"],
        "validation_fp": valid_best_metrics["fp"],
        "validation_fn": valid_best_metrics["fn"],
        "validation_tn": valid_best_metrics["tn"],

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

    print("[DONE] Experiment completed.")
    print(
        f"Test PR-AUC={summary['test_pr_auc']:.4f}, "
        f"Precision={summary['test_precision']:.4f}, "
        f"Recall={summary['test_recall']:.4f}, "
        f"Review rate={summary['test_review_rate']:.4f}"
    )

    return summary


def generate_experiment_comparison_report(
    comparison_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    lines.append("# Fraud Detection - Experiment Comparison Report\n")

    lines.append("## 1. Mục tiêu\n")
    lines.append(
        """
So sánh các cấu hình Logistic Regression để kiểm tra độ ổn định của feature `mcc`:

- Giữ `mcc` với rare category grouping min_frequency=50.
- Bỏ `mcc`.
- Giữ `mcc` nhưng tăng min_frequency=100.
- Giữ `mcc` nhưng tăng min_frequency=200.

Mục tiêu là xem model có phụ thuộc quá mạnh vào `mcc` hay không.
"""
    )

    lines.append("\n## 2. Bảng so sánh tổng thể\n")

    show_cols = [
        "experiment_name",
        "min_frequency",
        "dropped_categorical",
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
    ]

    show_cols = [col for col in show_cols if col in comparison_df.columns]

    lines.append(
        comparison_df[show_cols]
        .sort_values("test_f2", ascending=False)
        .to_markdown(index=False)
    )

    lines.append("\n## 3. Cách đọc kết quả\n")
    lines.append(
        """
- Nếu bỏ `mcc` làm PR-AUC/F2 giảm mạnh, nghĩa là `mcc` đang mang tín hiệu quan trọng.
- Nếu bỏ `mcc` mà kết quả gần tương đương, nên cân nhắc bỏ `mcc` để model dễ generalize hơn.
- Nếu min_frequency=100 hoặc 200 giữ hiệu năng gần bằng min_frequency=50 nhưng giảm số chiều, nên ưu tiên bản gọn hơn.
- `review_rate` cho biết tỷ lệ giao dịch bị đẩy sang kiểm tra thủ công.
- Test set chỉ dùng để đánh giá cuối cùng, không dùng để chọn threshold.
"""
    )

    lines.append("\n## 4. Khuyến nghị chọn model\n")
    lines.append(
        """
Ưu tiên model có:
1. PR-AUC tốt.
2. F2 tốt vì fraud detection ưu tiên recall.
3. Review rate phù hợp năng lực kiểm tra nghiệp vụ.
4. Số chiều sau preprocessing không quá lớn.
5. Không phụ thuộc quá cực đoan vào một nhóm category hiếm.
"""
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")