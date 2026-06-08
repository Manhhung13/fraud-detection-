from pathlib import Path
from typing import Optional

import pandas as pd


def get_dataset_overview(df: pd.DataFrame, target_col: str = "fraud") -> pd.DataFrame:
    """
    Create high-level dataset overview.
    """
    overview = {
        "num_rows": len(df),
        "num_columns": df.shape[1],
        "duplicate_rows": df.duplicated().sum(),
        "duplicate_rate": df.duplicated().mean(),
        "total_missing_values": df.isna().sum().sum(),
        "total_missing_rate": df.isna().sum().sum() / (df.shape[0] * df.shape[1]),
    }

    if target_col in df.columns:
        overview["target_column"] = target_col
        overview["fraud_count"] = int(df[target_col].sum())
        overview["non_fraud_count"] = int((df[target_col] == 0).sum())
        overview["fraud_rate"] = float(df[target_col].mean())

    return pd.DataFrame([overview])


def get_target_distribution(df: pd.DataFrame, target_col: str = "fraud") -> pd.DataFrame:
    """
    Analyze target distribution.
    """
    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy cột target: {target_col}")

    result = (
        df[target_col]
        .value_counts(dropna=False)
        .reset_index()
    )

    result.columns = [target_col, "count"]
    result["rate"] = result["count"] / len(df)

    return result


def get_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create missing value report.
    """
    result = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().values,
            "missing_rate": df.isna().mean().values,
            "non_missing_count": df.notna().sum().values,
        }
    )

    result = result.sort_values("missing_rate", ascending=False)

    return result


def get_column_overview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create overview for each column.
    """
    rows = []

    for col in df.columns:
        series = df[col]

        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "non_null_count": int(series.notna().sum()),
                "null_count": int(series.isna().sum()),
                "null_rate": float(series.isna().mean()),
                "unique_count": int(series.nunique(dropna=True)),
                "unique_rate": float(series.nunique(dropna=True) / len(df)),
                "sample_values": ", ".join(series.dropna().astype(str).unique()[:5]),
            }
        )

    result = pd.DataFrame(rows)
    result = result.sort_values("unique_count", ascending=False)

    return result


def get_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary statistics for numeric columns.
    """
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        return pd.DataFrame()

    summary = numeric_df.describe().T.reset_index()
    summary = summary.rename(columns={"index": "column"})

    summary["missing_count"] = numeric_df.isna().sum().values
    summary["missing_rate"] = numeric_df.isna().mean().values
    summary["skewness"] = numeric_df.skew(numeric_only=True).values

    return summary


def get_categorical_summary(df: pd.DataFrame, max_unique_display: int = 10) -> pd.DataFrame:
    """
    Summary for categorical/object columns.
    """
    categorical_df = df.select_dtypes(exclude=["number"])

    rows = []

    for col in categorical_df.columns:
        value_counts = categorical_df[col].value_counts(dropna=False).head(max_unique_display)

        top_values = "; ".join(
            [f"{idx}: {cnt}" for idx, cnt in value_counts.items()]
        )

        rows.append(
            {
                "column": col,
                "dtype": str(categorical_df[col].dtype),
                "unique_count": int(categorical_df[col].nunique(dropna=True)),
                "missing_count": int(categorical_df[col].isna().sum()),
                "missing_rate": float(categorical_df[col].isna().mean()),
                "top_values": top_values,
            }
        )

    return pd.DataFrame(rows)


def get_high_cardinality_report(
    df: pd.DataFrame,
    threshold: int = 1000,
    target_col: str = "fraud",
) -> pd.DataFrame:
    """
    Detect high-cardinality columns.
    """
    rows = []

    for col in df.columns:
        if col == target_col:
            continue

        unique_count = df[col].nunique(dropna=True)

        if unique_count >= threshold:
            rows.append(
                {
                    "column": col,
                    "dtype": str(df[col].dtype),
                    "unique_count": int(unique_count),
                    "unique_rate": float(unique_count / len(df)),
                    "note": "High cardinality - không nên one-hot trực tiếp",
                }
            )

    return pd.DataFrame(rows)


def get_fraud_rate_by_categorical(
    df: pd.DataFrame,
    feature: str,
    target_col: str = "fraud",
    min_count: int = 30,
) -> pd.DataFrame:
    """
    Calculate fraud rate by categorical feature.
    """
    if feature not in df.columns:
        raise ValueError(f"Không tìm thấy feature: {feature}")

    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy target: {target_col}")

    result = (
        df.groupby(feature, dropna=False)
        .agg(
            count=(target_col, "size"),
            fraud_count=(target_col, "sum"),
            fraud_rate=(target_col, "mean"),
        )
        .reset_index()
    )

    result = result[result["count"] >= min_count]
    result = result.sort_values("fraud_rate", ascending=False)

    return result


def generate_markdown_report(
    dataset_overview: pd.DataFrame,
    target_distribution: pd.DataFrame,
    column_overview: pd.DataFrame,
    high_cardinality_report: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Generate simple markdown report for data understanding stage.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    overview_row = dataset_overview.iloc[0].to_dict()

    lines = []

    lines.append("# Fraud Detection - Data Understanding Report\n")

    lines.append("## 1. Dataset Overview\n")
    lines.append(f"- Số dòng: **{overview_row.get('num_rows', 'N/A')}**")
    lines.append(f"- Số cột: **{overview_row.get('num_columns', 'N/A')}**")
    lines.append(f"- Số dòng duplicate: **{overview_row.get('duplicate_rows', 'N/A')}**")
    lines.append(f"- Tổng missing values: **{overview_row.get('total_missing_values', 'N/A')}**")

    if "fraud_rate" in overview_row:
        fraud_rate = overview_row["fraud_rate"]
        lines.append(f"- Fraud rate: **{fraud_rate:.4f} ({fraud_rate * 100:.2f}%)**")

    lines.append("\n## 2. Target Distribution\n")
    lines.append(target_distribution.to_markdown(index=False))

    lines.append("\n## 3. Column Overview\n")
    lines.append(column_overview.head(30).to_markdown(index=False))

    lines.append("\n## 4. High Cardinality Columns\n")

    if high_cardinality_report.empty:
        lines.append("Không phát hiện cột high-cardinality theo threshold hiện tại.")
    else:
        lines.append(high_cardinality_report.to_markdown(index=False))

    lines.append("\n## 5. Nhận xét ban đầu\n")
    lines.append(
        """
- Dataset cần được kiểm tra mất cân bằng nhãn vì fraud thường chiếm tỷ lệ nhỏ.
- Các cột có cardinality cao như ID, device, merchant hoặc IP không nên one-hot trực tiếp.
- Các feature dạng timestamp nên được tách thành hour, day_of_week, is_weekend, is_night.
- Các feature lịch sử hành vi như mean_amount_30d, max_amount_30d, mcc_entropy_30d, night_ratio_30d cần được khai phá kỹ.
- Cần kiểm tra leakage với các biến được tính theo lịch sử 30 ngày/365 ngày.
"""
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_overview_outputs(
    df: pd.DataFrame,
    output_dir: str | Path,
    report_dir: Optional[str | Path] = None,
    target_col: str = "fraud",
) -> None:
    """
    Save all overview reports to CSV and markdown.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if report_dir is None:
        report_dir = output_dir

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset_overview = get_dataset_overview(df, target_col=target_col)
    target_distribution = get_target_distribution(df, target_col=target_col)
    missing_report = get_missing_report(df)
    column_overview = get_column_overview(df)
    numeric_summary = get_numeric_summary(df)
    categorical_summary = get_categorical_summary(df)
    high_cardinality_report = get_high_cardinality_report(df, target_col=target_col)

    dataset_overview.to_csv(output_dir / "dataset_overview.csv", index=False)
    target_distribution.to_csv(output_dir / "target_distribution.csv", index=False)
    missing_report.to_csv(output_dir / "missing_report.csv", index=False)
    column_overview.to_csv(output_dir / "column_overview.csv", index=False)
    numeric_summary.to_csv(output_dir / "numeric_summary.csv", index=False)
    categorical_summary.to_csv(output_dir / "categorical_summary.csv", index=False)
    high_cardinality_report.to_csv(output_dir / "high_cardinality_report.csv", index=False)

    generate_markdown_report(
        dataset_overview=dataset_overview,
        target_distribution=target_distribution,
        column_overview=column_overview,
        high_cardinality_report=high_cardinality_report,
        output_path=report_dir / "eda_report.md",
    )