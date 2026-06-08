from __future__ import annotations

import pandas as pd


def _ensure_binary(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Lấy một cột flag và ép về 0/1.
    Nếu cột không tồn tại thì trả về toàn 0.
    """
    if col not in df.columns:
        return pd.Series(0, index=df.index)

    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo interaction features từ insight ổn định giai đoạn 3.

    Lưu ý:
    - Các base flags phải được tạo trước bởi behavior_features.py
    - Hàm này không fit threshold.
    - Hàm này chỉ nhân các flag 0/1 để tạo interaction.
    """
    df = df.copy()

    low_mcc = _ensure_binary(df, "low_mcc_entropy_flag")
    very_low_mcc = _ensure_binary(df, "very_low_mcc_entropy_flag")
    low_night = _ensure_binary(df, "low_night_ratio_flag")
    high_max = _ensure_binary(df, "high_max_amount_30d_flag")
    low_spending = _ensure_binary(df, "low_spending_trend_flag")
    low_country = _ensure_binary(df, "low_country_diversity_flag")
    night_txn = _ensure_binary(df, "night_transaction_flag")
    card_not_present = _ensure_binary(df, "card_not_present_flag")
    cross_border = _ensure_binary(df, "cross_border_flag")
    high_ip = _ensure_binary(df, "high_ip_score_flag")
    tokenised = _ensure_binary(df, "tokenised_flag")
    high_decline = _ensure_binary(df, "high_decline_rate_flag")
    high_device = _ensure_binary(df, "high_device_diversity_flag")

    # Nhóm interaction mạnh nhất từ stability check
    df["low_mcc_entropy_x_low_night_ratio"] = low_mcc * low_night

    df["low_mcc_entropy_x_low_night_ratio_x_card_not_present"] = (
        low_mcc * low_night * card_not_present
    )

    df["low_mcc_entropy_x_high_max_amount_30d"] = low_mcc * high_max

    df["low_mcc_entropy_x_low_spending_trend"] = low_mcc * low_spending

    df["low_night_ratio_x_high_max_amount_30d"] = low_night * high_max

    df["low_night_ratio_x_night_transaction"] = low_night * night_txn

    df["low_mcc_entropy_x_card_not_present"] = low_mcc * card_not_present

    df["cross_border_x_low_country_diversity"] = cross_border * low_country

    # Nhóm interaction phụ, ưu tiên thấp hơn nhưng vẫn có ý nghĩa
    df["very_low_mcc_entropy_x_low_night_ratio"] = very_low_mcc * low_night

    df["very_low_mcc_entropy_x_card_not_present"] = (
        very_low_mcc * card_not_present
    )

    df["high_ip_score_x_card_not_present"] = high_ip * card_not_present

    df["high_decline_rate_x_card_not_present"] = (
        high_decline * card_not_present
    )

    df["high_device_diversity_x_card_not_present"] = (
        high_device * card_not_present
    )

    df["high_max_amount_30d_x_low_country_diversity"] = (
        high_max * low_country
    )

    df["high_max_amount_30d_x_tokenised"] = high_max * tokenised

    return df


def create_fraud_insight_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm giữ lại để tương thích với code giai đoạn 3 cũ.

    Khuyến nghị:
    - Trong pipeline chính thức nên gọi riêng:
        create_behavior_features(df, thresholds)
        create_interaction_features(df)

    Hàm này chỉ nên dùng cho EDA nhanh, vì thresholds=None sẽ fit ngưỡng ngay trên df truyền vào.
    """
    from src.features.behavior_features import create_behavior_features

    df = df.copy()
    df = create_behavior_features(df)
    df = create_interaction_features(df)

    return df