from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.mining.fraud_rate_analysis import fraud_rate_by_numeric_bin


def create_numeric_bin_edges_from_train(
    train_df: pd.DataFrame,
    features: Iterable[str],
    bins: int = 10,
) -> dict[str, list[float]]:
    """
    Tạo bin edges từ train để apply giống hệt sang validation.

    Nếu dùng qcut riêng cho train và validation thì bin khác nhau,
    không so sánh stability được.
    """
    edges: dict[str, list[float]] = {}

    for feature in features:
        if feature not in train_df.columns:
            continue

        x = pd.to_numeric(train_df[feature], errors="coerce").dropna()

        if x.empty or x.nunique() < 2:
            continue

        try:
            _, bin_edges = pd.qcut(
                x,
                q=bins,
                retbins=True,
                duplicates="drop",
            )
        except ValueError:
            _, bin_edges = pd.cut(
                x,
                bins=bins,
                retbins=True,
                duplicates="drop",
            )

        bin_edges = list(sorted(set(float(v) for v in bin_edges)))

        if len(bin_edges) < 2:
            continue

        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        edges[feature] = bin_edges

    return edges


def analyze_numeric_features_with_train_bins(
    df: pd.DataFrame,
    bin_edges: dict[str, list[float]],
    target_col: str = "fraud",
    min_count: int | None = None,
    min_count_ratio: float = 0.005,
    default_min_count: int = 100,
) -> pd.DataFrame:
    """
    Analyze numeric features bằng bin edges đã fit từ train.
    """
    outputs = []

    for feature, edges in bin_edges.items():
        if feature not in df.columns:
            continue

        try:
            report = fraud_rate_by_numeric_bin(
                df=df,
                feature=feature,
                target_col=target_col,
                bins=edges,
                min_count=min_count,
                min_count_ratio=min_count_ratio,
                default_min_count=default_min_count,
                method="custom",
            )

            if not report.empty:
                outputs.append(report)

        except Exception as exc:
            print(f"Skip numeric stability feature {feature}: {exc}")

    if not outputs:
        return pd.DataFrame()

    return pd.concat(outputs, ignore_index=True)


def compare_pattern_stability(
    train_report: pd.DataFrame,
    valid_report: pd.DataFrame,
    key_cols: list[str],
    suffix_train: str = "_train",
    suffix_valid: str = "_valid",
) -> pd.DataFrame:
    """
    So sánh pattern giữa train và validation.

    Ví dụ:
    - numeric: key_cols = ["feature", "bin"]
    - categorical: key_cols = ["feature", "value"]
    - interaction: key_cols = ["rule"]
    """
    if train_report is None or valid_report is None:
        return pd.DataFrame()

    if train_report.empty or valid_report.empty:
        return pd.DataFrame()

    metric_cols = [
        "count",
        "support",
        "fraud_count",
        "fraud_rate",
        "global_fraud_rate",
        "fraud_rate_diff",
        "lift",
    ]

    for col in key_cols:
        if col not in train_report.columns or col not in valid_report.columns:
            return pd.DataFrame()

    train_cols = key_cols + [col for col in metric_cols if col in train_report.columns]
    valid_cols = key_cols + [col for col in metric_cols if col in valid_report.columns]

    merged = train_report[train_cols].merge(
        valid_report[valid_cols],
        on=key_cols,
        how="inner",
        suffixes=(suffix_train, suffix_valid),
    )

    if merged.empty:
        return pd.DataFrame()

    train_lift_col = f"lift{suffix_train}"
    valid_lift_col = f"lift{suffix_valid}"

    train_fr_col = f"fraud_rate{suffix_train}"
    valid_fr_col = f"fraud_rate{suffix_valid}"

    if train_lift_col in merged.columns and valid_lift_col in merged.columns:
        merged["lift_diff_valid_train"] = (
            merged[valid_lift_col] - merged[train_lift_col]
        )

        merged["lift_ratio_valid_train"] = (
            merged[valid_lift_col] / (merged[train_lift_col] + 1e-9)
        )

    if train_fr_col in merged.columns and valid_fr_col in merged.columns:
        merged["fraud_rate_diff_valid_train"] = (
            merged[valid_fr_col] - merged[train_fr_col]
        )

    return merged.sort_values(
        by=[valid_lift_col, train_lift_col],
        ascending=False,
    ).reset_index(drop=True)


def filter_stable_patterns(
    stability_report: pd.DataFrame,
    min_lift_train: float = 1.5,
    min_lift_valid: float = 1.5,
    min_support_train: int = 50,
    min_support_valid: int = 30,
    max_lift_ratio_gap: float = 0.75,
) -> pd.DataFrame:
    """
    Lọc pattern ổn định giữa train và validation.

    max_lift_ratio_gap = 0.75 nghĩa là:
    lift_valid / lift_train nằm trong khoảng [0.25, 1.75].
    """
    if stability_report is None or stability_report.empty:
        return pd.DataFrame()

    result = stability_report.copy()

    required_cols = [
        "lift_train",
        "lift_valid",
        "lift_ratio_valid_train",
    ]

    for col in required_cols:
        if col not in result.columns:
            return pd.DataFrame()

    support_train_col = "support_train" if "support_train" in result.columns else "count_train"
    support_valid_col = "support_valid" if "support_valid" in result.columns else "count_valid"

    if support_train_col not in result.columns or support_valid_col not in result.columns:
        return pd.DataFrame()

    lower = max(0, 1 - max_lift_ratio_gap)
    upper = 1 + max_lift_ratio_gap

    result = result[
        (result["lift_train"] >= min_lift_train)
        & (result["lift_valid"] >= min_lift_valid)
        & (result[support_train_col] >= min_support_train)
        & (result[support_valid_col] >= min_support_valid)
        & (result["lift_ratio_valid_train"] >= lower)
        & (result["lift_ratio_valid_train"] <= upper)
    ]

    if result.empty:
        return pd.DataFrame()

    return result.sort_values(
        ["lift_valid", "lift_train"],
        ascending=False,
    ).reset_index(drop=True)


def generate_stability_markdown_report(
    output_path: str | Path,
    split_summary: pd.DataFrame,
    stable_numeric: pd.DataFrame,
    stable_categorical: pd.DataFrame,
    stable_priority_interactions: pd.DataFrame,
    stable_boolean_interactions: pd.DataFrame,
    stable_categorical_pairs: pd.DataFrame,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def to_md(df: pd.DataFrame, top_n: int = 20) -> str:
        if df is None or df.empty:
            return "Không có pattern ổn định theo ngưỡng hiện tại."
        return df.head(top_n).to_markdown(index=False)

    lines = []

    lines.append("# Fraud Insight Stability Check Report\n")

    lines.append("## 1. Split summary\n")
    lines.append(split_summary.to_markdown(index=False))

    lines.append("\n## 2. Stable numeric patterns\n")
    lines.append(to_md(stable_numeric, top_n=20))

    lines.append("\n## 3. Stable categorical patterns\n")
    lines.append(to_md(stable_categorical, top_n=20))

    lines.append("\n## 4. Stable priority business interactions\n")
    lines.append(to_md(stable_priority_interactions, top_n=20))

    lines.append("\n## 5. Stable boolean interactions\n")
    lines.append(to_md(stable_boolean_interactions, top_n=20))

    lines.append("\n## 6. Stable categorical pair interactions\n")
    lines.append(to_md(stable_categorical_pairs, top_n=20))

    lines.append("\n## 7. Kết luận\n")
    lines.append(
        """
Các insight nên được ưu tiên đưa sang giai đoạn Feature Engineering là các pattern:
- Có lift cao trên train.
- Vẫn giữ lift cao trên validation.
- Support đủ lớn ở cả train và validation.
- Có ý nghĩa nghiệp vụ rõ ràng.

Không nên chọn rule chỉ mạnh trên train nhưng yếu trên validation, vì đó có thể là overfit insight.
"""
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")