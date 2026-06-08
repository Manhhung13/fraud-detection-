from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def predict_label_from_proba(
    y_proba: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return (y_proba >= threshold).astype(int)


def evaluate_binary_classifier(
    y_true,
    y_proba,
    threshold: float = 0.5,
    split_name: str = "validation",
) -> dict:
    """
    Đánh giá binary classifier cho fraud detection.

    Không dùng accuracy làm metric chính.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    y_pred = predict_label_from_proba(y_proba, threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc_auc = np.nan

    try:
        pr_auc = average_precision_score(y_true, y_proba)
    except ValueError:
        pr_auc = np.nan

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    total = len(y_true)
    predicted_positive = int(y_pred.sum())
    actual_positive = int(y_true.sum())

    review_rate = predicted_positive / total if total > 0 else 0.0
    fraud_rate = actual_positive / total if total > 0 else 0.0

    return {
        "split": split_name,
        "threshold": threshold,
        "rows": total,
        "fraud_count": actual_positive,
        "fraud_rate": fraud_rate,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "balanced_accuracy": balanced_acc,
        "review_rate": review_rate,
        "predicted_positive": predicted_positive,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def confusion_matrix_dataframe(
    y_true,
    y_proba,
    threshold: float,
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_pred = predict_label_from_proba(y_proba, threshold)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return pd.DataFrame(
        cm,
        index=["actual_non_fraud", "actual_fraud"],
        columns=["pred_non_fraud", "pred_fraud"],
    )


def prediction_dataframe(
    y_true,
    y_proba,
    threshold: float,
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    y_pred = predict_label_from_proba(y_proba, threshold)

    return pd.DataFrame(
        {
            "y_true": y_true,
            "fraud_probability": y_proba,
            "y_pred": y_pred,
            "threshold": threshold,
        }
    )