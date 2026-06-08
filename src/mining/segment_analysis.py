from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class SegmentRule:
    """
    Định nghĩa một segment nghiệp vụ để phân tích fraud.
    """
    name: str
    description: str
    condition_func: Callable[[pd.DataFrame], pd.Series]


def _empty_condition(df: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=df.index)


def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    return pd.to_numeric(df[col], errors="coerce")


def q_low(df: pd.DataFrame, col: str, q: float = 0.1) -> pd.Series:
    """
    Segment giá trị thấp theo quantile.
    """
    if col not in df.columns:
        return _empty_condition(df)

    x = _to_numeric(df, col)

    if x.dropna().empty:
        return _empty_condition(df)

    threshold = x.quantile(q)
    return x <= threshold


def q_high(df: pd.DataFrame, col: str, q: float = 0.9) -> pd.Series:
    """
    Segment giá trị cao theo quantile.
    """
    if col not in df.columns:
        return _empty_condition(df)

    x = _to_numeric(df, col)

    if x.dropna().empty:
        return _empty_condition(df)

    threshold = x.quantile(q)
    return x >= threshold


def equals_value(df: pd.DataFrame, col: str, value) -> pd.Series:
    if col not in df.columns:
        return _empty_condition(df)

    return df[col] == value


def bool_true(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Chuẩn hóa các kiểu True/False, 1/0, yes/no.
    """
    if col not in df.columns:
        return _empty_condition(df)

    return (
        df[col]
        .astype(str)
        .str.lower()
        .str.strip()
        .isin(["true", "1", "yes", "y"])
    )


def bool_false(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Chuẩn hóa các kiểu True/False, 1/0, yes/no.
    """
    if col not in df.columns:
        return _empty_condition(df)

    return (
        df[col]
        .astype(str)
        .str.lower()
        .str.strip()
        .isin(["false", "0", "no", "n"])
    )


def evaluate_segment(
    df: pd.DataFrame,
    condition: pd.Series,
    segment_name: str,
    description: str = "",
    target_col: str = "fraud",
) -> dict:
    """
    Đánh giá một segment:
    - support
    - support_rate
    - fraud_count
    - fraud_rate
    - lift
    """
    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy target column: {target_col}")

    target = pd.to_numeric(df[target_col], errors="coerce")

    valid_mask = target.notna()
    condition = condition.reindex(df.index).fillna(False) & valid_mask

    total_count = int(valid_mask.sum())
    global_fraud_rate = float(target[valid_mask].mean()) if total_count > 0 else 0.0

    segment_target = target[condition]
    support = int(condition.sum())

    if support == 0:
        fraud_count = 0
        fraud_rate = 0.0
        lift = 0.0
        support_rate = 0.0
    else:
        fraud_count = int(segment_target.sum())
        fraud_rate = float(segment_target.mean())
        lift = float(fraud_rate / global_fraud_rate) if global_fraud_rate > 0 else np.nan
        support_rate = float(support / total_count) if total_count > 0 else 0.0

    return {
        "segment_name": segment_name,
        "description": description,
        "support": support,
        "support_rate": support_rate,
        "fraud_count": fraud_count,
        "fraud_rate": fraud_rate,
        "global_fraud_rate": global_fraud_rate,
        "fraud_rate_diff": fraud_rate - global_fraud_rate,
        "lift": lift,
    }


def build_business_segments() -> list[SegmentRule]:
    """
    Các segment nghiệp vụ quan trọng.

    File này chỉ phân tích segment, không tạo feature flag.
    Feature flag đã nằm trong:
    - src/features/behavior_features.py
    """
    return [
        SegmentRule(
            name="low_mcc_entropy",
            description=(
                "User có lịch sử MCC ít đa dạng. Đây là tín hiệu behavioral anomaly "
                "vì giao dịch mới có thể lệch khỏi hành vi ngành hàng quen thuộc."
            ),
            condition_func=lambda df: q_low(df, "mcc_entropy_30d", 0.1),
        ),
        SegmentRule(
            name="very_low_mcc_entropy",
            description="Nhóm cực thấp về mcc_entropy_30d, thường cần soi kỹ hơn.",
            condition_func=lambda df: q_low(df, "mcc_entropy_30d", 0.05),
        ),
        SegmentRule(
            name="low_night_ratio",
            description="User gần như không có thói quen giao dịch ban đêm.",
            condition_func=lambda df: q_low(df, "night_ratio_30d", 0.1),
        ),
        SegmentRule(
            name="night_transaction",
            description="Giao dịch hiện tại xảy ra ban đêm.",
            condition_func=lambda df: equals_value(df, "is_night_txn", 1),
        ),
        SegmentRule(
            name="night_unusual",
            description=(
                "Giao dịch ban đêm nhưng user có lịch sử night_ratio_30d thấp. "
                "Đây là tín hiệu bất thường về thời gian."
            ),
            condition_func=lambda df: (
                equals_value(df, "is_night_txn", 1)
                & q_low(df, "night_ratio_30d", 0.1)
            ),
        ),
        SegmentRule(
            name="high_max_amount_30d",
            description="User có max_amount_30d thuộc nhóm rất cao.",
            condition_func=lambda df: q_high(df, "max_amount_30d", 0.9),
        ),
        SegmentRule(
            name="high_current_amount",
            description="Giao dịch hiện tại có amount thuộc nhóm cao.",
            condition_func=lambda df: q_high(df, "amount", 0.9),
        ),
        SegmentRule(
            name="low_spending_trend",
            description="Xu hướng chi tiêu 30 ngày thấp hoặc giảm.",
            condition_func=lambda df: q_low(df, "spending_trend_30d", 0.1),
        ),
        SegmentRule(
            name="high_decline_rate",
            description="User có tỷ lệ decline cao trong 30 ngày.",
            condition_func=lambda df: q_high(df, "decline_rate_30d", 0.9),
        ),
        SegmentRule(
            name="has_chargeback_history",
            description="User có lịch sử chargeback trong 365 ngày.",
            condition_func=lambda df: _to_numeric(df, "chargebacks_365d").fillna(0) > 0,
        ),
        SegmentRule(
            name="low_country_diversity",
            description="User có lịch sử giao dịch ở ít quốc gia.",
            condition_func=lambda df: q_low(df, "distinct_countries_30d", 0.1),
        ),
        SegmentRule(
            name="high_device_diversity",
            description=(
                "User dùng nhiều thiết bị trong 30 ngày. Có thể liên quan account takeover "
                "hoặc account sharing."
            ),
            condition_func=lambda df: q_high(df, "device_diversity_30d", 0.9),
        ),
        SegmentRule(
            name="card_not_present",
            description="Giao dịch không có thẻ vật lý, thường rủi ro hơn trong fraud online.",
            condition_func=lambda df: bool_false(df, "card_present"),
        ),
        SegmentRule(
            name="cross_border",
            description="Giao dịch xuyên biên giới.",
            condition_func=lambda df: bool_true(df, "cross_border"),
        ),
        SegmentRule(
            name="high_ip_score",
            description="IP risk score thuộc nhóm cao.",
            condition_func=lambda df: q_high(df, "ip_score", 0.9),
        ),
        SegmentRule(
            name="tokenised_transaction",
            description="Giao dịch tokenised.",
            condition_func=lambda df: bool_true(df, "tokenised"),
        ),
        SegmentRule(
            name="high_amount_to_mean",
            description=(
                "Amount hiện tại cao hơn nhiều so với mean_amount_30d. "
                "Đây là tín hiệu amount anomaly."
            ),
            condition_func=lambda df: q_high(df, "amount_to_mean_30d", 0.9),
        ),
        SegmentRule(
            name="high_amount_to_max",
            description=(
                "Amount hiện tại tiệm cận hoặc vượt max_amount_30d. "
                "Cần kiểm tra vì có thể là giao dịch bất thường."
            ),
            condition_func=lambda df: q_high(df, "amount_to_max_30d", 0.9),
        ),
    ]


def analyze_business_segments(
    df: pd.DataFrame,
    target_col: str = "fraud",
) -> pd.DataFrame:
    """
    Chạy toàn bộ segment nghiệp vụ.
    """
    rows = []

    for segment in build_business_segments():
        try:
            condition = segment.condition_func(df)
            row = evaluate_segment(
                df=df,
                condition=condition,
                segment_name=segment.name,
                description=segment.description,
                target_col=target_col,
            )
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "segment_name": segment.name,
                    "description": segment.description,
                    "support": 0,
                    "support_rate": 0.0,
                    "fraud_count": 0,
                    "fraud_rate": 0.0,
                    "global_fraud_rate": np.nan,
                    "fraud_rate_diff": np.nan,
                    "lift": 0.0,
                    "error": str(exc),
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        ["lift", "fraud_rate", "support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)