from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.evaluate import evaluate_binary_classifier


def search_thresholds(
    y_true,
    y_proba,
    thresholds: np.ndarray | None = None,
    split_name: str = "validation",
) -> pd.DataFrame:
    """
    Search threshold trên validation.

    Không search threshold trên test.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)

    rows = []

    for threshold in thresholds:
        metrics = evaluate_binary_classifier(
            y_true=y_true,
            y_proba=y_proba,
            threshold=float(threshold),
            split_name=split_name,
        )
        rows.append(metrics)

    return pd.DataFrame(rows)


def select_best_threshold(
    threshold_report: pd.DataFrame,
    metric: str = "f2",
    min_precision: float | None = None,
    min_recall: float | None = None,
    max_review_rate: float | None = None,
) -> dict:
    """
    Chọn threshold tốt nhất từ validation.

    metric:
    - f1: cân bằng precision/recall
    - f2: ưu tiên recall hơn precision, phù hợp fraud detection
    - recall: bắt fraud nhiều nhất
    - precision: ít false positive nhất
    """
    if threshold_report.empty:
        raise ValueError("threshold_report rỗng.")

    result = threshold_report.copy()

    if min_precision is not None:
        result = result[result["precision"] >= min_precision]

    if min_recall is not None:
        result = result[result["recall"] >= min_recall]

    if max_review_rate is not None:
        result = result[result["review_rate"] <= max_review_rate]

    if result.empty:
        result = threshold_report.copy()

    if metric not in result.columns:
        raise ValueError(f"Metric không tồn tại trong threshold_report: {metric}")

    result = result.sort_values(
        [metric, "recall", "precision"],
        ascending=[False, False, False],
    )

    return result.iloc[0].to_dict()