from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.features.interaction_features import create_fraud_insight_features
from src.mining.fraud_rate_analysis import (
    analyze_many_categorical_features,
    analyze_many_numeric_features,
    detect_strong_categorical_patterns,
    detect_strong_numeric_patterns,
    resolve_min_count,
)


def run_fraud_pattern_mining(
    df: pd.DataFrame,
    categorical_features: Iterable[str],
    numeric_features: Iterable[str],
    target_col: str = "fraud",
    bins: int = 10,
    min_count: int | None = None,
    min_count_ratio: float = 0.005,
    default_min_count: int = 100,
) -> dict[str, pd.DataFrame]:
    """
    Chạy full pattern mining:

    1. Tạo business bins + interaction features
    2. Analyze categorical features
    3. Analyze numeric features bằng quantile
    4. Lọc strong categorical patterns
    5. Lọc strong numeric patterns
    """
    df = df.copy()
    df = create_fraud_insight_features(df)

    categorical_features = list(categorical_features)
    numeric_features = list(numeric_features)

    auto_categorical_features = [
        "ip_score_risk_group",
        "decline_rate_risk_group",
        "credit_util_risk_group",
        "txn_count_7d_group",
        "txn_velocity_group",
        "night_ratio_group",
        "amount_to_mean_group",
        "amount_to_max_group",
        "night_unusual_group",
        "high_ip_high_amount_flag",
        "high_decline_high_amount_flag",
        "private_ip_high_score_flag",
        "high_velocity_high_amount_flag",
        "low_mcc_entropy_x_low_night_ratio",
        "low_mcc_entropy_x_high_max_amount_30d",
        "low_mcc_entropy_x_low_spending_trend",
        "low_mcc_entropy_x_card_not_present",
        "low_night_ratio_x_high_max_amount_30d",
        "cross_border_x_low_country_diversity",
        "high_ip_score_x_card_not_present",
    ]

    for col in auto_categorical_features:
        if col in df.columns and col not in categorical_features:
            categorical_features.append(col)

    auto_numeric_features = [
        "amount_to_mean_30d",
        "amount_to_max_30d",
        "night_unusual_score",
    ]

    for col in auto_numeric_features:
        if col in df.columns and col not in numeric_features:
            numeric_features.append(col)

    categorical_report = analyze_many_categorical_features(
        df=df,
        features=categorical_features,
        target_col=target_col,
        min_count=min_count,
        min_count_ratio=min_count_ratio,
        default_min_count=default_min_count,
    )

    numeric_report = analyze_many_numeric_features(
        df=df,
        features=numeric_features,
        target_col=target_col,
        bins=bins,
        min_count=min_count,
        min_count_ratio=min_count_ratio,
        default_min_count=default_min_count,
        method="quantile",
    )

    used_min_count = resolve_min_count(
        df=df,
        min_count=min_count,
        min_count_ratio=min_count_ratio,
        default_min_count=default_min_count,
    )

    strong_categorical_patterns = detect_strong_categorical_patterns(
        categorical_report=categorical_report,
        min_lift=1.5,
        min_count=used_min_count,
    )

    strong_numeric_patterns = detect_strong_numeric_patterns(
        numeric_bin_report=numeric_report,
        min_lift=2.0,
        min_count=used_min_count,
    )

    return {
        "df_with_insight_features": df,
        "categorical_report": categorical_report,
        "numeric_report": numeric_report,
        "strong_categorical_patterns": strong_categorical_patterns,
        "strong_numeric_patterns": strong_numeric_patterns,
    }