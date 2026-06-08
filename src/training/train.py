from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.models.base_model import save_model
from src.models.model_factory import create_model
from src.pipelines.preprocessing_pipeline import get_preprocessed_feature_names
from src.training.evaluate import (
    confusion_matrix_dataframe,
    evaluate_binary_classifier,
    prediction_dataframe,
)
from src.training.threshold import search_thresholds, select_best_threshold


def load_feature_metadata(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy feature metadata: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    """
    Extract coefficient sau khi pipeline đã fit.

    Output giúp giải thích model.
    """
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
            "direction": np.where(coefficients >= 0, "increase_risk", "decrease_risk"),
        }
    )

    coef_df = coef_df.sort_values(
        "abs_coefficient",
        ascending=False,
    ).reset_index(drop=True)

    return coef_df


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_model_markdown_report(
    output_path: str | Path,
    model_config: dict,
    train_metrics_05: dict,
    validation_metrics_05: dict,
    validation_best_metrics: dict,
    test_metrics: dict,
    coefficient_df: pd.DataFrame,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    lines.append("# Fraud Detection - Logistic Regression Baseline Report\n")

    lines.append("## 1. Model configuration\n")
    for key, value in model_config.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.append("\n## 2. Metrics at threshold 0.5\n")
    metrics_05 = pd.DataFrame([train_metrics_05, validation_metrics_05])
    lines.append(metrics_05.to_markdown(index=False))

    lines.append("\n## 3. Best threshold selected on validation\n")
    lines.append(pd.DataFrame([validation_best_metrics]).to_markdown(index=False))

    lines.append("\n## 4. Final test metrics using selected validation threshold\n")
    lines.append(pd.DataFrame([test_metrics]).to_markdown(index=False))

    lines.append("\n## 5. Top positive coefficients: increase fraud risk\n")
    top_positive = (
        coefficient_df[coefficient_df["coefficient"] > 0]
        .sort_values("coefficient", ascending=False)
        .head(20)
    )
    lines.append(top_positive.to_markdown(index=False))

    lines.append("\n## 6. Top negative coefficients: decrease fraud risk\n")
    top_negative = (
        coefficient_df[coefficient_df["coefficient"] < 0]
        .sort_values("coefficient", ascending=True)
        .head(20)
    )
    lines.append(top_negative.to_markdown(index=False))

    lines.append("\n## 7. Business interpretation\n")
    lines.append(
        """
- PR-AUC quan trọng hơn accuracy vì fraud là lớp thiểu số.
- Recall cho biết model bắt được bao nhiêu fraud thật.
- Precision cho biết trong các giao dịch bị cảnh báo, bao nhiêu là fraud thật.
- Review rate cho biết tỷ lệ giao dịch bị đẩy sang kiểm tra thủ công.
- Threshold được chọn trên validation, sau đó áp dụng nguyên sang test.
- Test set không được dùng để chọn threshold.
"""
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def train_logistic_regression_baseline(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metadata: dict,
    output_dir: str | Path,
    report_dir: str | Path,
    model_name: str = "logistic_regression",
    class_weight: str | dict | None = "balanced",
    C: float = 1.0,
    use_rare_category_grouping: bool = True,
    min_frequency: int | None = 50,
    threshold_metric: str = "f2",
) -> dict:
    """
    Train full Logistic Regression baseline.
    """
    output_dir = Path(output_dir)
    report_dir = Path(report_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    target_col = metadata["target_col"]

    numeric_features = metadata["numeric_features"]
    categorical_features = metadata["categorical_features"]
    binary_features = metadata["binary_features"]
    interaction_features = metadata["interaction_features"]

    X_train, y_train = split_xy(train_df, target_col)
    X_valid, y_valid = split_xy(validation_df, target_col)
    X_test, y_test = split_xy(test_df, target_col)

    model_config = {
        "model_name": model_name,
        "class_weight": class_weight,
        "C": C,
        "use_rare_category_grouping": use_rare_category_grouping,
        "min_frequency": min_frequency,
        "threshold_metric": threshold_metric,
        "numeric_features": len(numeric_features),
        "categorical_features": len(categorical_features),
        "binary_features": len(binary_features),
        "interaction_features": len(interaction_features),
    }

    model = create_model(
        model_name=model_name,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        binary_features=binary_features,
        interaction_features=interaction_features,
        class_weight=class_weight,
        C=C,
        use_rare_category_grouping=use_rare_category_grouping,
        min_frequency=min_frequency,
    )

    print("[TRAIN] Fitting model...")
    model.fit(X_train, y_train)

    print("[PREDICT] Predicting probabilities...")
    train_proba = get_positive_class_proba(model, X_train)
    valid_proba = get_positive_class_proba(model, X_valid)
    test_proba = get_positive_class_proba(model, X_test)

    print("[EVAL] Metrics at threshold 0.5...")
    train_metrics_05 = evaluate_binary_classifier(
        y_true=y_train,
        y_proba=train_proba,
        threshold=0.5,
        split_name="train",
    )

    validation_metrics_05 = evaluate_binary_classifier(
        y_true=y_valid,
        y_proba=valid_proba,
        threshold=0.5,
        split_name="validation",
    )

    print("[THRESHOLD] Searching best threshold on validation...")
    threshold_report = search_thresholds(
        y_true=y_valid,
        y_proba=valid_proba,
        split_name="validation",
    )

    best_threshold_row = select_best_threshold(
        threshold_report=threshold_report,
        metric=threshold_metric,
        max_review_rate=0.20,
    )

    best_threshold = float(best_threshold_row["threshold"])

    print(f"[THRESHOLD] Best threshold by {threshold_metric}: {best_threshold}")

    validation_best_metrics = evaluate_binary_classifier(
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

    print("[SAVE] Saving model...")
    model_path = output_dir / "logistic_regression_pipeline.joblib"
    save_model(model, model_path)

    print("[SAVE] Saving reports...")
    pd.DataFrame([train_metrics_05]).to_csv(
        report_dir / "model_metrics_train_threshold_05.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([validation_metrics_05]).to_csv(
        report_dir / "model_metrics_validation_threshold_05.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([validation_best_metrics]).to_csv(
        report_dir / "model_metrics_validation_best_threshold.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([test_metrics]).to_csv(
        report_dir / "model_metrics_test.csv",
        index=False,
        encoding="utf-8-sig",
    )

    threshold_report.to_csv(
        report_dir / "threshold_search_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )

    confusion_matrix_dataframe(
        y_true=y_valid,
        y_proba=valid_proba,
        threshold=best_threshold,
    ).to_csv(
        report_dir / "confusion_matrix_validation.csv",
        encoding="utf-8-sig",
    )

    confusion_matrix_dataframe(
        y_true=y_test,
        y_proba=test_proba,
        threshold=best_threshold,
    ).to_csv(
        report_dir / "confusion_matrix_test.csv",
        encoding="utf-8-sig",
    )

    prediction_dataframe(
        y_true=y_valid,
        y_proba=valid_proba,
        threshold=best_threshold,
    ).to_csv(
        report_dir / "validation_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction_dataframe(
        y_true=y_test,
        y_proba=test_proba,
        threshold=best_threshold,
    ).to_csv(
        report_dir / "test_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    coefficient_df = extract_logistic_regression_coefficients(model)
    coefficient_df.to_csv(
        report_dir / "logistic_regression_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model_metadata = {
        "model_path": str(model_path),
        "selected_threshold": best_threshold,
        "selected_threshold_metric": threshold_metric,
        "model_config": model_config,
        "train_metrics_threshold_05": train_metrics_05,
        "validation_metrics_threshold_05": validation_metrics_05,
        "validation_metrics_best_threshold": validation_best_metrics,
        "test_metrics": test_metrics,
    }

    save_json(
        model_metadata,
        output_dir / "logistic_regression_metadata.json",
    )

    generate_model_markdown_report(
        output_path=report_dir.parent / "model_report.md",
        model_config=model_config,
        train_metrics_05=train_metrics_05,
        validation_metrics_05=validation_metrics_05,
        validation_best_metrics=validation_best_metrics,
        test_metrics=test_metrics,
        coefficient_df=coefficient_df,
    )

    return model_metadata