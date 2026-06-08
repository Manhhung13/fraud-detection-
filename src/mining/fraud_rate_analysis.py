from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd


def calculate_global_fraud_rate(
    df: pd.DataFrame,
    target_col: str = "fraud",
) -> float:
    """
    Tính fraud rate toàn bộ dataset.

    Vì fraud thường là:
    - 0: non-fraud
    - 1: fraud

    Nên mean(fraud) chính là fraud rate.
    """
    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy target column: {target_col}")

    target = pd.to_numeric(df[target_col], errors="coerce")

    if target.dropna().empty:
        raise ValueError(f"Target column {target_col} không có dữ liệu hợp lệ.")

    return float(target.mean())


def resolve_min_count(
    df: pd.DataFrame,
    min_count: int | None = None,
    min_count_ratio: float = 0.005,
    default_min_count: int = 100,
) -> int:
    """
    Tự động chọn min_count để tránh pattern quá ít mẫu.

    Ví dụ:
    - len(df) = 100,000
    - min_count_ratio = 0.005
    => min_count = 500

    Nếu truyền min_count cụ thể thì ưu tiên min_count đó.
    """
    if min_count is not None:
        return int(min_count)

    return max(default_min_count, int(len(df) * min_count_ratio))


def _safe_lift(
    fraud_rate: pd.Series,
    global_rate: float,
) -> pd.Series:
    """
    Tính lift an toàn, tránh chia cho 0.
    """
    if global_rate == 0 or np.isnan(global_rate):
        return pd.Series(np.nan, index=fraud_rate.index)

    return fraud_rate / global_rate


def add_risk_metrics(
    result: pd.DataFrame,
    global_rate: float,
) -> pd.DataFrame:
    """
    Thêm metric giúp đọc insight:
    - global_fraud_rate
    - lift
    - fraud_rate_diff
    """
    result = result.copy()

    result["global_fraud_rate"] = global_rate
    result["lift"] = _safe_lift(result["fraud_rate"], global_rate)
    result["fraud_rate_diff"] = result["fraud_rate"] - global_rate

    return result


def fraud_rate_by_category(
    df: pd.DataFrame,
    feature: str,
    target_col: str = "fraud",
    min_count: int | None = None,
    min_count_ratio: float = 0.005,
    default_min_count: int = 100,
) -> pd.DataFrame:
    """
    Tính fraud rate theo biến categorical.

    Output:
    - feature
    - value
    - count
    - fraud_count
    - fraud_rate
    - global_fraud_rate
    - fraud_rate_diff
    - lift
    """
    if feature not in df.columns:
        raise ValueError(f"Không tìm thấy feature: {feature}")

    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy target: {target_col}")

    used_min_count = resolve_min_count(
        df=df,
        min_count=min_count,
        min_count_ratio=min_count_ratio,
        default_min_count=default_min_count,
    )

    global_rate = calculate_global_fraud_rate(df, target_col)

    tmp = df[[feature, target_col]].copy()
    tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")
    tmp = tmp.dropna(subset=[target_col])

    result = (
        tmp.groupby(feature, dropna=False)
        .agg(
            count=(target_col, "size"),
            fraud_count=(target_col, "sum"),
            fraud_rate=(target_col, "mean"),
        )
        .reset_index()
        .rename(columns={feature: "value"})
    )

    result["feature"] = feature
    result = add_risk_metrics(result, global_rate)
    result = result[result["count"] >= used_min_count]

    result = result[
        [
            "feature",
            "value",
            "count",
            "fraud_count",
            "fraud_rate",
            "global_fraud_rate",
            "fraud_rate_diff",
            "lift",
        ]
    ].sort_values(
        ["lift", "fraud_rate", "count"],
        ascending=[False, False, False],
    )

    return result.reset_index(drop=True)


def create_numeric_bins(
    series: pd.Series,
    bins: int | Sequence[float] = 10,
    method: str = "quantile",
) -> pd.Series:
    """
    Tạo bin cho biến numeric.

    method:
    - quantile: chia theo phân vị
    - equal_width: chia khoảng đều
    - custom: chia theo ngưỡng nghiệp vụ
    """
    x = pd.to_numeric(series, errors="coerce")

    if method == "quantile":
        try:
            return pd.qcut(
                x,
                q=int(bins),
                duplicates="drop",
            )
        except ValueError:
            return pd.cut(
                x,
                bins=int(bins),
                duplicates="drop",
            )

    if method == "equal_width":
        return pd.cut(
            x,
            bins=int(bins),
            duplicates="drop",
        )

    if method == "custom":
        if not isinstance(bins, Sequence):
            raise ValueError("Khi method='custom', bins phải là list ngưỡng.")

        return pd.cut(
            x,
            bins=list(bins),
            duplicates="drop",
            include_lowest=True,
        )

    raise ValueError("method chỉ nhận: 'quantile', 'equal_width', hoặc 'custom'")


def fraud_rate_by_numeric_bin(
    df: pd.DataFrame,
    feature: str,
    target_col: str = "fraud",
    bins: int | Sequence[float] = 10,
    min_count: int | None = None,
    min_count_ratio: float = 0.005,
    default_min_count: int = 100,
    method: str = "quantile",
) -> pd.DataFrame:
    """
    Tính fraud rate theo bin của biến numeric.
    """
    if feature not in df.columns:
        raise ValueError(f"Không tìm thấy feature: {feature}")

    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy target: {target_col}")

    used_min_count = resolve_min_count(
        df=df,
        min_count=min_count,
        min_count_ratio=min_count_ratio,
        default_min_count=default_min_count,
    )

    tmp = df[[feature, target_col]].copy()
    tmp[feature] = pd.to_numeric(tmp[feature], errors="coerce")
    tmp[target_col] = pd.to_numeric(tmp[target_col], errors="coerce")

    tmp = tmp.dropna(subset=[feature, target_col])

    if tmp.empty:
        return pd.DataFrame()

    if tmp[feature].nunique() < 2:
        return pd.DataFrame()

    tmp["bin"] = create_numeric_bins(
        series=tmp[feature],
        bins=bins,
        method=method,
    )

    tmp = tmp.dropna(subset=["bin"])

    if tmp.empty:
        return pd.DataFrame()

    global_rate = calculate_global_fraud_rate(df, target_col)

    result = (
        tmp.groupby("bin", observed=False)
        .agg(
            count=(target_col, "size"),
            fraud_count=(target_col, "sum"),
            fraud_rate=(target_col, "mean"),
            min_value=(feature, "min"),
            max_value=(feature, "max"),
            mean_value=(feature, "mean"),
        )
        .reset_index()
    )

    result["feature"] = feature
    result["bin"] = result["bin"].astype(str)
    result = add_risk_metrics(result, global_rate)
    result = result[result["count"] >= used_min_count]

    result = result[
        [
            "feature",
            "bin",
            "count",
            "fraud_count",
            "fraud_rate",
            "global_fraud_rate",
            "fraud_rate_diff",
            "lift",
            "min_value",
            "max_value",
            "mean_value",
        ]
    ].sort_values(
        ["lift", "fraud_rate", "count"],
        ascending=[False, False, False],
    )

    return result.reset_index(drop=True)


def analyze_many_categorical_features(
    df: pd.DataFrame,
    features: Iterable[str],
    target_col: str = "fraud",
    min_count: int | None = None,
    min_count_ratio: float = 0.005,
    default_min_count: int = 100,
) -> pd.DataFrame:
    """
    Chạy fraud rate analysis cho nhiều biến categorical.
    """
    outputs = []

    for feature in features:
        if feature not in df.columns:
            continue

        try:
            result = fraud_rate_by_category(
                df=df,
                feature=feature,
                target_col=target_col,
                min_count=min_count,
                min_count_ratio=min_count_ratio,
                default_min_count=default_min_count,
            )

            if not result.empty:
                outputs.append(result)

        except Exception as exc:
            print(f"Skip categorical feature {feature}: {exc}")

    if not outputs:
        return pd.DataFrame()

    return pd.concat(outputs, ignore_index=True)


def analyze_many_numeric_features(
    df: pd.DataFrame,
    features: Iterable[str],
    target_col: str = "fraud",
    bins: int = 10,
    min_count: int | None = None,
    min_count_ratio: float = 0.005,
    default_min_count: int = 100,
    method: str = "quantile",
) -> pd.DataFrame:
    """
    Chạy fraud rate analysis cho nhiều biến numeric.
    """
    outputs = []

    for feature in features:
        if feature not in df.columns:
            continue

        try:
            result = fraud_rate_by_numeric_bin(
                df=df,
                feature=feature,
                target_col=target_col,
                bins=bins,
                min_count=min_count,
                min_count_ratio=min_count_ratio,
                default_min_count=default_min_count,
                method=method,
            )

            if not result.empty:
                outputs.append(result)

        except Exception as exc:
            print(f"Skip numeric feature {feature}: {exc}")

    if not outputs:
        return pd.DataFrame()

    return pd.concat(outputs, ignore_index=True)


def detect_strong_numeric_patterns(
    numeric_bin_report: pd.DataFrame,
    min_lift: float = 2.0,
    min_fraud_rate: float | None = None,
    min_count: int | None = None,
) -> pd.DataFrame:
    """
    Lọc ra các bin numeric có tín hiệu fraud mạnh.
    """
    if numeric_bin_report.empty:
        return pd.DataFrame()

    result = numeric_bin_report.copy()
    result = result[result["lift"] >= min_lift]

    if min_fraud_rate is not None:
        result = result[result["fraud_rate"] >= min_fraud_rate]

    if min_count is not None:
        result = result[result["count"] >= min_count]

    return result.sort_values(
        ["lift", "fraud_rate", "count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def detect_strong_categorical_patterns(
    categorical_report: pd.DataFrame,
    min_lift: float = 1.5,
    min_count: int = 100,
    min_fraud_rate: float | None = None,
) -> pd.DataFrame:
    """
    Lọc ra các value categorical có tín hiệu fraud mạnh.
    """
    if categorical_report.empty:
        return pd.DataFrame()

    result = categorical_report.copy()

    result = result[
        (result["lift"] >= min_lift)
        & (result["count"] >= min_count)
    ]

    if min_fraud_rate is not None:
        result = result[result["fraud_rate"] >= min_fraud_rate]

    return result.sort_values(
        ["lift", "fraud_rate", "count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)