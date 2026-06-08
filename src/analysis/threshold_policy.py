from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.training.evaluate import evaluate_binary_classifier


def choose_threshold_by_capacity(
    threshold_df: pd.DataFrame,
    max_review_rate: float,
    metric: str = "f2",
) -> dict:
    valid = threshold_df[threshold_df["review_rate"] <= max_review_rate].copy()

    if valid.empty:
        valid = threshold_df.copy()

    valid = valid.sort_values(
        [metric, "recall", "precision"],
        ascending=[False, False, False],
    )

    return valid.iloc[0].to_dict()


def generate_threshold_policy(
    project_root: str | Path,
    experiment_name: str = "cat_tune_depth7_l2_8_lr003_iter700",
    capacities: list[float] | None = None,
) -> None:
    project_root = Path(project_root)

    if capacities is None:
        capacities = [0.03, 0.05, 0.10]

    report_dir = project_root / "data" / "reports" / "catboost_tuning" / experiment_name

    threshold_path = report_dir / "threshold_search_validation.csv"
    test_pred_path = report_dir / "test_predictions.csv"

    if not threshold_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {threshold_path}")

    if not test_pred_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {test_pred_path}")

    threshold_df = pd.read_csv(threshold_path)
    test_preds = pd.read_csv(test_pred_path)

    rows = []

    for cap in capacities:
        selected = choose_threshold_by_capacity(
            threshold_df=threshold_df,
            max_review_rate=cap,
            metric="f2",
        )

        threshold = float(selected["threshold"])

        test_metrics = evaluate_binary_classifier(
            y_true=test_preds["y_true"],
            y_proba=test_preds["fraud_probability"],
            threshold=threshold,
            split_name=f"test_policy_review_{int(cap * 100)}pct",
        )

        row = {
            "review_capacity": cap,
            "selected_threshold_from_validation": threshold,
            "validation_precision": selected["precision"],
            "validation_recall": selected["recall"],
            "validation_f2": selected["f2"],
            "validation_review_rate": selected["review_rate"],
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

        rows.append(row)

    policy_df = pd.DataFrame(rows)

    output_dir = project_root / "data" / "reports" / "threshold_policy"
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_df.to_csv(output_dir / "threshold_policy_review_capacity.csv", index=False, encoding="utf-8-sig")

    report_path = project_root / "reports" / "threshold_policy_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    display_df = policy_df.copy()
    for col in [
        "review_capacity",
        "validation_precision",
        "validation_recall",
        "validation_f2",
        "validation_review_rate",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_f2",
        "test_review_rate",
    ]:
        if col in display_df.columns:
            display_df[col] = display_df[col].astype(float).round(4)

    lines = []
    lines.append("# Threshold Policy Report\n")
    lines.append("## 1. Review capacity scenarios\n")
    lines.append(display_df.to_markdown(index=False))

    lines.append(
        """
## 2. Cách đọc

- `review_capacity`: năng lực review tối đa giả định của đội vận hành.
- Threshold được chọn trên validation, sau đó áp dụng sang test.
- Nếu review capacity tăng, recall thường tăng nhưng precision có thể giảm.
- Chọn policy phụ thuộc vào chi phí bỏ sót fraud và chi phí review nhầm.
"""
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved threshold policy: {output_dir / 'threshold_policy_review_capacity.csv'}")
    print(f"Saved report: {report_path}")