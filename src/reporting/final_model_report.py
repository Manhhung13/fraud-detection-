from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_one_row_csv(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"File rỗng: {path}")

    return df.iloc[0].to_dict()


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def _fmt_float(value, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):.{digits}f}"


def _fmt_pct(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value) * 100:.{digits}f}%"


def _build_model_row(
    model_label: str,
    model_role: str,
    metrics: dict,
    n_features: int | None = None,
    note: str = "",
) -> dict:
    return {
        "model_label": model_label,
        "model_role": model_role,
        "threshold": metrics.get("threshold"),
        "n_features": n_features,
        "pr_auc": metrics.get("pr_auc"),
        "roc_auc": metrics.get("roc_auc"),
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "f2": metrics.get("f2"),
        "review_rate": metrics.get("review_rate"),
        "tp": metrics.get("tp"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
        "tn": metrics.get("tn"),
        "note": note,
    }


def _load_metadata_n_features(path: Path) -> int | None:
    if not path.exists():
        return None

    try:
        import json

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return int(data.get("n_preprocessed_features"))
    except Exception:
        return None


def _load_feature_importance(path: Path, top_n: int = 20) -> pd.DataFrame:
    df = _read_csv_if_exists(path)

    if df.empty:
        return df

    if "importance" not in df.columns:
        return df.head(top_n)

    return df.sort_values("importance", ascending=False).head(top_n)


def _write_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "Không có dữ liệu."

    return df.to_markdown(index=False)


def generate_final_model_report(
    project_root: str | Path,
    output_path: str | Path | None = None,
) -> None:
    """
    Generate final model report.

    So sánh:
    - Logistic Regression baseline: lr_no_mcc
    - CatBoost có interaction: cat_gpu_mcc_minfreq_100
    - CatBoost bỏ interaction: cat_gpu_mcc_minfreq_100_no_interactions
    """
    project_root = Path(project_root)

    if output_path is None:
        output_path = project_root / "reports" / "final_model_comparison_report.md"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Logistic Regression baseline
    lr_dir = project_root / "data" / "reports" / "model" / "experiments" / "lr_no_mcc"
    lr_meta_path = (
        project_root
        / "models"
        / "experiments"
        / "lr_no_mcc"
        / "experiment_metadata.json"
    )

    # CatBoost with interactions
    cat_inter_dir = (
        project_root
        / "data"
        / "reports"
        / "gpu_boosting_models"
        / "cat_gpu_mcc_minfreq_100"
    )
    cat_inter_meta_path = (
        project_root
        / "models"
        / "gpu_boosting_experiments"
        / "cat_gpu_mcc_minfreq_100"
        / "experiment_metadata.json"
    )

    # CatBoost without manual interactions
    cat_no_inter_dir = (
        project_root
        / "data"
        / "reports"
        / "gpu_boosting_models"
        / "cat_gpu_mcc_minfreq_100_no_interactions"
    )
    cat_no_inter_meta_path = (
        project_root
        / "models"
        / "gpu_boosting_experiments"
        / "cat_gpu_mcc_minfreq_100_no_interactions"
        / "experiment_metadata.json"
    )

    lr_test = _read_one_row_csv(lr_dir / "metrics_test.csv")
    cat_inter_test = _read_one_row_csv(cat_inter_dir / "metrics_test.csv")
    cat_no_inter_test = _read_one_row_csv(cat_no_inter_dir / "metrics_test.csv")

    lr_validation = _read_one_row_csv(lr_dir / "metrics_validation_best_threshold.csv")
    cat_inter_validation = _read_one_row_csv(
        cat_inter_dir / "metrics_validation_best_threshold.csv"
    )
    cat_no_inter_validation = _read_one_row_csv(
        cat_no_inter_dir / "metrics_validation_best_threshold.csv"
    )

    cat_no_inter_train = _read_one_row_csv(
        cat_no_inter_dir / "metrics_train_threshold_05.csv"
    )

    comparison_rows = [
        _build_model_row(
            model_label="lr_no_mcc",
            model_role="Baseline - dễ giải thích",
            metrics=lr_test,
            n_features=_load_metadata_n_features(lr_meta_path),
            note="Logistic Regression baseline, không dùng MCC.",
        ),
        _build_model_row(
            model_label="cat_gpu_mcc_minfreq_100",
            model_role="Advanced - có manual interactions",
            metrics=cat_inter_test,
            n_features=_load_metadata_n_features(cat_inter_meta_path),
            note="CatBoost GPU, dùng MCC rare grouping, có interaction thủ công.",
        ),
        _build_model_row(
            model_label="cat_gpu_mcc_minfreq_100_no_interactions",
            model_role="Final advanced - bỏ manual interactions",
            metrics=cat_no_inter_test,
            n_features=_load_metadata_n_features(cat_no_inter_meta_path),
            note="CatBoost GPU, dùng MCC rare grouping, bỏ interaction thủ công.",
        ),
    ]

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_display = comparison_df.copy()
    for col in ["pr_auc", "roc_auc", "precision", "recall", "f1", "f2", "review_rate"]:
        if col in comparison_display.columns:
            if col in ["precision", "recall", "review_rate"]:
                comparison_display[col] = comparison_display[col].apply(_fmt_pct)
            else:
                comparison_display[col] = comparison_display[col].apply(_fmt_float)

    gap_df = pd.DataFrame(
        [
            {
                "split": "train_threshold_0.5",
                "threshold": cat_no_inter_train.get("threshold"),
                "pr_auc": cat_no_inter_train.get("pr_auc"),
                "precision": cat_no_inter_train.get("precision"),
                "recall": cat_no_inter_train.get("recall"),
                "f1": cat_no_inter_train.get("f1"),
                "f2": cat_no_inter_train.get("f2"),
                "review_rate": cat_no_inter_train.get("review_rate"),
            },
            {
                "split": "validation_best_threshold",
                "threshold": cat_no_inter_validation.get("threshold"),
                "pr_auc": cat_no_inter_validation.get("pr_auc"),
                "precision": cat_no_inter_validation.get("precision"),
                "recall": cat_no_inter_validation.get("recall"),
                "f1": cat_no_inter_validation.get("f1"),
                "f2": cat_no_inter_validation.get("f2"),
                "review_rate": cat_no_inter_validation.get("review_rate"),
            },
            {
                "split": "test",
                "threshold": cat_no_inter_test.get("threshold"),
                "pr_auc": cat_no_inter_test.get("pr_auc"),
                "precision": cat_no_inter_test.get("precision"),
                "recall": cat_no_inter_test.get("recall"),
                "f1": cat_no_inter_test.get("f1"),
                "f2": cat_no_inter_test.get("f2"),
                "review_rate": cat_no_inter_test.get("review_rate"),
            },
        ]
    )

    gap_display = gap_df.copy()
    for col in ["pr_auc", "precision", "recall", "f1", "f2", "review_rate"]:
        if col in gap_display.columns:
            if col in ["precision", "recall", "review_rate"]:
                gap_display[col] = gap_display[col].apply(_fmt_pct)
            else:
                gap_display[col] = gap_display[col].apply(_fmt_float)

    fi_df = _load_feature_importance(
        cat_no_inter_dir / "feature_importance.csv",
        top_n=25,
    )

    fi_display = fi_df.copy()
    if not fi_display.empty and "importance" in fi_display.columns:
        fi_display["importance"] = fi_display["importance"].apply(
            lambda x: _fmt_float(x, digits=4)
        )

    # Business confusion matrix from final model
    tp = int(cat_no_inter_test.get("tp", 0))
    fp = int(cat_no_inter_test.get("fp", 0))
    fn = int(cat_no_inter_test.get("fn", 0))
    tn = int(cat_no_inter_test.get("tn", 0))
    rows = int(cat_no_inter_test.get("rows", tp + fp + fn + tn))
    fraud_count = int(cat_no_inter_test.get("fraud_count", tp + fn))
    predicted_positive = int(cat_no_inter_test.get("predicted_positive", tp + fp))

    lines = []

    lines.append("# Final Fraud Detection Model Report\n")

    lines.append("## 1. Mục tiêu báo cáo\n")
    lines.append(
        """
Báo cáo này tổng hợp kết quả cuối cùng của project fraud detection, so sánh:

1. Logistic Regression baseline.
2. CatBoost GPU có manual interaction features.
3. CatBoost GPU bỏ manual interaction features.

Mục tiêu là chọn model cuối cùng dựa trên PR-AUC, F2, precision, recall, review rate và khả năng giải thích bằng insight nghiệp vụ.
"""
    )

    lines.append("\n## 2. Bảng so sánh model\n")
    lines.append(_write_markdown_table(comparison_display))

    lines.append("\n## 3. Model được chọn cuối cùng\n")
    lines.append(
        """
Model được chọn cuối cùng là:

**`cat_gpu_mcc_minfreq_100_no_interactions`**

Lý do:
- Có PR-AUC và F2 tốt nhất hoặc gần tốt nhất.
- Precision cao hơn bản có interaction.
- Review rate thấp hơn.
- Số feature ít hơn vì bỏ manual interaction features.
- Feature importance vẫn khớp với insight giai đoạn 3.
- CatBoost có khả năng tự học quan hệ phi tuyến và interaction từ feature gốc.
"""
    )

    lines.append("\n## 4. Diễn giải nghiệp vụ trên tập test\n")
    lines.append(
        f"""
Trên tập test có **{rows:,}** giao dịch, trong đó có **{fraud_count:,}** fraud thật.

Với threshold **{cat_no_inter_test.get("threshold")}**, model:
- Cảnh báo **{predicted_positive:,}** giao dịch cần review.
- Bắt đúng **{tp:,}** fraud.
- Báo nhầm **{fp:,}** giao dịch thường.
- Bỏ sót **{fn:,}** fraud.
- Xác định đúng **{tn:,}** giao dịch thường.
- Review rate: **{_fmt_pct(cat_no_inter_test.get("review_rate"))}**.
- Precision: **{_fmt_pct(cat_no_inter_test.get("precision"))}**.
- Recall: **{_fmt_pct(cat_no_inter_test.get("recall"))}**.
"""
    )

    lines.append("\n## 5. Kiểm tra overfit: train vs validation vs test\n")
    lines.append(_write_markdown_table(gap_display))

    lines.append(
        """
Nhận xét:
- Train PR-AUC cao hơn validation/test là bình thường với boosting model.
- Validation và test không bị sụp; test còn tốt hơn validation.
- Threshold chọn trên validation áp dụng sang test vẫn cho kết quả tốt.
- Không có dấu hiệu overfit nghiêm trọng.
"""
    )

    lines.append("\n## 6. Top feature importance của final model\n")
    lines.append(_write_markdown_table(fi_display))

    lines.append("\n## 7. Giải thích bằng insight nghiệp vụ\n")
    lines.append(
        """
Các feature quan trọng nhất của final CatBoost model tập trung vào các nhóm insight đã khai phá:

- **MCC behavior**: `mcc_entropy_30d` là feature quan trọng nhất. Model học hành vi đa dạng ngành hàng thay vì học thuộc mã MCC cụ thể.
- **Night behavior**: `night_ratio_30d` đứng top đầu, phản ánh thói quen giao dịch theo thời gian.
- **Amount anomaly**: `log_amount_to_max_30d`, `amount_z_30d` cho thấy số tiền hiện tại lệch khỏi lịch sử user là tín hiệu mạnh.
- **Spending trend**: `spending_trend` có importance cao, khớp với insight spending trend bất thường.
- **Country behavior**: `distinct_countries_30d` thể hiện lịch sử đa dạng quốc gia của user.
- **Technical/risk history**: `decline_rate_30d`, `device_diversity_30d`, `credit_util_today`, `ip_score` hỗ trợ thêm cho model.

Việc bỏ manual interaction features không làm giảm performance vì CatBoost có thể tự học interaction thông qua các split nhiều tầng.
"""
    )

    lines.append("\n## 8. Kết luận cuối\n")
    lines.append(
        """
Logistic Regression được giữ làm baseline vì dễ giải thích bằng coefficient và giúp kiểm chứng feature engineering.
Tuy nhiên, CatBoost GPU với MCC rare grouping `min_frequency=100` và không dùng manual interaction features được chọn làm final advanced model vì đạt hiệu năng tốt hơn rõ rệt, review rate thấp hơn và vẫn học đúng các insight nghiệp vụ chính.
"""
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved final model report: {output_path}")