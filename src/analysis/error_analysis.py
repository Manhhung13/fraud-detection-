from __future__ import annotations

from pathlib import Path

import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return pd.read_csv(path)


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    return pd.read_parquet(path)


def summarize_numeric_by_group(
    df: pd.DataFrame,
    group_col: str,
    numeric_cols: list[str],
) -> pd.DataFrame:
    available_cols = [c for c in numeric_cols if c in df.columns]

    if not available_cols:
        return pd.DataFrame()

    rows = []

    for group_value, g in df.groupby(group_col):
        row = {
            group_col: group_value,
            "rows": len(g),
        }

        for col in available_cols:
            x = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = x.mean()
            row[f"{col}_median"] = x.median()
            row[f"{col}_p90"] = x.quantile(0.90)

        rows.append(row)

    return pd.DataFrame(rows)


def summarize_categorical_by_error_type(
    df: pd.DataFrame,
    categorical_cols: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    rows = []

    for col in categorical_cols:
        if col not in df.columns:
            continue

        for error_type, g in df.groupby("error_type"):
            vc = (
                g[col]
                .astype(str)
                .fillna("missing")
                .value_counts(normalize=True)
                .head(top_n)
            )

            for value, ratio in vc.items():
                rows.append(
                    {
                        "error_type": error_type,
                        "feature": col,
                        "value": value,
                        "ratio": ratio,
                        "count": int((g[col].astype(str) == value).sum()),
                    }
                )

    return pd.DataFrame(rows)


def run_error_analysis(
    project_root: str | Path,
    experiment_name: str = "cat_tune_depth7_l2_8_lr003_iter700",
) -> None:
    project_root = Path(project_root)

    report_dir = project_root / "data" / "reports" / "catboost_tuning" / experiment_name
    processed_dir = project_root / "data" / "processed"

    test_pred_path = report_dir / "test_predictions.csv"
    test_features_path = processed_dir / "model_test_features.parquet"

    output_dir = project_root / "data" / "reports" / "final_error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    preds = _safe_read_csv(test_pred_path)
    test_df = _safe_read_parquet(test_features_path).reset_index(drop=True)

    if len(preds) != len(test_df):
        raise ValueError(
            f"Số dòng predictions ({len(preds)}) khác test features ({len(test_df)})."
        )

    df = pd.concat([test_df, preds], axis=1)

    df["error_type"] = "TN"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 1), "error_type"] = "TP"
    df.loc[(df["y_true"] == 0) & (df["y_pred"] == 1), "error_type"] = "FP"
    df.loc[(df["y_true"] == 1) & (df["y_pred"] == 0), "error_type"] = "FN"

    df.to_csv(output_dir / "test_predictions_with_features.csv", index=False, encoding="utf-8-sig")

    df[df["error_type"] == "FP"].to_csv(output_dir / "false_positives.csv", index=False, encoding="utf-8-sig")
    df[df["error_type"] == "FN"].to_csv(output_dir / "false_negatives.csv", index=False, encoding="utf-8-sig")
    df[df["error_type"] == "TP"].to_csv(output_dir / "true_positives.csv", index=False, encoding="utf-8-sig")

    numeric_cols = [
        "mcc_entropy_30d",
        "night_ratio_30d",
        "log_amount_to_max_30d",
        "spending_trend",
        "distinct_countries_30d",
        "amount_z_30d",
        "decline_rate_30d",
        "device_diversity_30d",
        "credit_util_today",
        "ip_score",
        "txn_count_ratio_7d_30d",
    ]

    categorical_cols = [
        "payment_channel",
        "merchant_country",
        "card_entry_mode",
        "auth_result",
        "term_location",
        "time_period",
        "day_name",
        "amount_to_max_group",
        "night_ratio_group",
        "credit_util_risk_group",
    ]

    numeric_summary = summarize_numeric_by_group(
        df=df,
        group_col="error_type",
        numeric_cols=numeric_cols,
    )

    categorical_summary = summarize_categorical_by_error_type(
        df=df,
        categorical_cols=categorical_cols,
        top_n=10,
    )

    numeric_summary.to_csv(output_dir / "error_numeric_summary.csv", index=False, encoding="utf-8-sig")
    categorical_summary.to_csv(output_dir / "error_categorical_summary.csv", index=False, encoding="utf-8-sig")

    counts = (
        df["error_type"]
        .value_counts()
        .rename_axis("error_type")
        .reset_index(name="count")
    )
    counts["ratio"] = counts["count"] / len(df)
    counts.to_csv(output_dir / "error_type_counts.csv", index=False, encoding="utf-8-sig")

    report_path = project_root / "reports" / "error_analysis_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# Error Analysis Report\n")
    lines.append("## 1. Error type counts\n")
    lines.append(counts.to_markdown(index=False))

    lines.append("\n## 2. Numeric summary by error type\n")
    lines.append(numeric_summary.to_markdown(index=False))

    lines.append("\n## 3. Top categorical patterns by error type\n")
    if categorical_summary.empty:
        lines.append("Không có categorical summary.")
    else:
        lines.append(categorical_summary.head(80).to_markdown(index=False))

    lines.append(
        """
## 4. Cách đọc báo cáo

- **FP**: model cảnh báo fraud nhưng thực tế không fraud. Nhóm này gây tải review.
- **FN**: model bỏ sót fraud. Nhóm này nguy hiểm nhất về nghiệp vụ.
- Cần xem FN có đặc điểm gì: amount thấp hơn, entropy bình thường hơn, hoặc thiếu tín hiệu bất thường.
- Cần xem FP có tập trung vào kênh/thời điểm/quốc gia nào không để cân nhắc rule hoặc threshold phụ.
"""
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved error analysis to: {output_dir}")
    print(f"Saved report to: {report_path}")