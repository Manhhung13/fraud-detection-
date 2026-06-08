from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class BehaviorThresholds:
    """
    Các threshold phải được fit từ train rồi apply sang validation/test.
    """
    very_low_mcc_entropy_q05: float | None = None
    low_mcc_entropy_q10: float | None = None
    low_night_ratio_q10: float | None = None
    high_max_amount_q90: float | None = None
    low_spending_trend_q10: float | None = None
    low_country_diversity_q10: float | None = None
    high_device_diversity_q90: float | None = None
    high_decline_rate_q90: float | None = None
    high_ip_score_q90: float | None = None


def _find_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col

    return None


def _safe_quantile(df: pd.DataFrame, col: str | None, q: float) -> float | None:
    if col is None or col not in df.columns:
        return None

    x = pd.to_numeric(df[col], errors="coerce").dropna()

    if x.empty:
        return None

    return float(x.quantile(q))


def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    return pd.to_numeric(df[col], errors="coerce")


def _bool_true(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)

    return (
        df[col]
        .astype(str)
        .str.lower()
        .str.strip()
        .isin(["true", "1", "yes", "y"])
    )


def _bool_false(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)

    return (
        df[col]
        .astype(str)
        .str.lower()
        .str.strip()
        .isin(["false", "0", "no", "n"])
    )


def fit_behavior_thresholds(train_df: pd.DataFrame) -> BehaviorThresholds:
    """
    Fit threshold chỉ trên train để tránh leakage.
    """
    spending_col = _find_existing_col(
        train_df,
        ["spending_trend_30d", "spending_trend"],
    )

    return BehaviorThresholds(
        very_low_mcc_entropy_q05=_safe_quantile(train_df, "mcc_entropy_30d", 0.05),
        low_mcc_entropy_q10=_safe_quantile(train_df, "mcc_entropy_30d", 0.10),
        low_night_ratio_q10=_safe_quantile(train_df, "night_ratio_30d", 0.10),
        high_max_amount_q90=_safe_quantile(train_df, "max_amount_30d", 0.90),
        low_spending_trend_q10=_safe_quantile(train_df, spending_col, 0.10),
        low_country_diversity_q10=_safe_quantile(
            train_df,
            "distinct_countries_30d",
            0.10,
        ),
        high_device_diversity_q90=_safe_quantile(
            train_df,
            "device_diversity_30d",
            0.90,
        ),
        high_decline_rate_q90=_safe_quantile(train_df, "decline_rate_30d", 0.90),
        high_ip_score_q90=_safe_quantile(train_df, "ip_score", 0.90),
    )


def thresholds_to_dataframe(thresholds: BehaviorThresholds) -> pd.DataFrame:
    return pd.DataFrame([asdict(thresholds)])


def thresholds_to_dict(thresholds: BehaviorThresholds) -> dict:
    return asdict(thresholds)


def create_business_risk_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các business bins cố định.
    Đây là categorical feature cho OneHotEncoder ở giai đoạn model.
    """
    df = df.copy()

    if "ip_score" in df.columns:
        df["ip_score_risk_group"] = pd.cut(
            _to_numeric(df, "ip_score"),
            bins=[-np.inf, 0.2, 0.5, 0.8, np.inf],
            labels=["low", "medium", "high", "very_high"],
            include_lowest=True,
        ).astype("object")

    if "decline_rate_30d" in df.columns:
        df["decline_rate_risk_group"] = pd.cut(
            _to_numeric(df, "decline_rate_30d"),
            bins=[-np.inf, 0.1, 0.3, 0.6, np.inf],
            labels=["low", "medium", "high", "very_high"],
            include_lowest=True,
        ).astype("object")

    if "credit_util_today" in df.columns:
        df["credit_util_risk_group"] = pd.cut(
            _to_numeric(df, "credit_util_today"),
            bins=[-np.inf, 0.3, 0.6, 0.9, np.inf],
            labels=["low", "medium", "high", "very_high"],
            include_lowest=True,
        ).astype("object")

    if "txn_count_7d" in df.columns:
        df["txn_count_7d_group"] = pd.cut(
            _to_numeric(df, "txn_count_7d"),
            bins=[-np.inf, 1, 5, 10, 20, np.inf],
            labels=["0_1", "2_5", "6_10", "11_20", "gt_20"],
            include_lowest=True,
        ).astype("object")

    if "txn_count_ratio_7d_30d" in df.columns:
        df["txn_velocity_group"] = pd.cut(
            _to_numeric(df, "txn_count_ratio_7d_30d"),
            bins=[-np.inf, 0.2, 0.5, 0.8, 1.2, np.inf],
            labels=["very_low", "low", "medium", "high", "very_high"],
            include_lowest=True,
        ).astype("object")

    if "night_ratio_30d" in df.columns:
        df["night_ratio_group"] = pd.cut(
            _to_numeric(df, "night_ratio_30d"),
            bins=[-np.inf, 0, 0.1, 0.3, 0.6, np.inf],
            labels=["zero", "very_low", "low", "medium", "high"],
            include_lowest=True,
        ).astype("object")

    if "amount_to_mean_30d" in df.columns:
        df["amount_to_mean_group"] = pd.cut(
            _to_numeric(df, "amount_to_mean_30d"),
            bins=[-np.inf, 1, 2, 5, 10, np.inf],
            labels=["normal", "slightly_high", "high", "very_high", "extreme"],
            include_lowest=True,
        ).astype("object")

    if "amount_to_max_30d" in df.columns:
        df["amount_to_max_group"] = pd.cut(
            _to_numeric(df, "amount_to_max_30d"),
            bins=[-np.inf, 0.5, 1, 2, 5, np.inf],
            labels=[
                "below_half_max",
                "near_max",
                "above_max",
                "much_above_max",
                "extreme",
            ],
            include_lowest=True,
        ).astype("object")

    return df


def create_behavior_flags(
    df: pd.DataFrame,
    thresholds: BehaviorThresholds,
) -> pd.DataFrame:
    """
    Tạo các binary flags từ threshold fit trên train.
    """
    df = df.copy()

    if thresholds.very_low_mcc_entropy_q05 is not None and "mcc_entropy_30d" in df.columns:
        df["very_low_mcc_entropy_flag"] = (
            _to_numeric(df, "mcc_entropy_30d") <= thresholds.very_low_mcc_entropy_q05
        ).astype(int)

    if thresholds.low_mcc_entropy_q10 is not None and "mcc_entropy_30d" in df.columns:
        df["low_mcc_entropy_flag"] = (
            _to_numeric(df, "mcc_entropy_30d") <= thresholds.low_mcc_entropy_q10
        ).astype(int)

    if thresholds.low_night_ratio_q10 is not None and "night_ratio_30d" in df.columns:
        df["low_night_ratio_flag"] = (
            _to_numeric(df, "night_ratio_30d") <= thresholds.low_night_ratio_q10
        ).astype(int)

    if thresholds.high_max_amount_q90 is not None and "max_amount_30d" in df.columns:
        df["high_max_amount_30d_flag"] = (
            _to_numeric(df, "max_amount_30d") >= thresholds.high_max_amount_q90
        ).astype(int)

    spending_col = _find_existing_col(df, ["spending_trend_30d", "spending_trend"])

    if thresholds.low_spending_trend_q10 is not None and spending_col is not None:
        df["low_spending_trend_flag"] = (
            _to_numeric(df, spending_col) <= thresholds.low_spending_trend_q10
        ).astype(int)

    if thresholds.low_country_diversity_q10 is not None and "distinct_countries_30d" in df.columns:
        df["low_country_diversity_flag"] = (
            _to_numeric(df, "distinct_countries_30d") <= thresholds.low_country_diversity_q10
        ).astype(int)

    if thresholds.high_device_diversity_q90 is not None and "device_diversity_30d" in df.columns:
        df["high_device_diversity_flag"] = (
            _to_numeric(df, "device_diversity_30d") >= thresholds.high_device_diversity_q90
        ).astype(int)

    if thresholds.high_decline_rate_q90 is not None and "decline_rate_30d" in df.columns:
        df["high_decline_rate_flag"] = (
            _to_numeric(df, "decline_rate_30d") >= thresholds.high_decline_rate_q90
        ).astype(int)

    if thresholds.high_ip_score_q90 is not None and "ip_score" in df.columns:
        df["high_ip_score_flag"] = (
            _to_numeric(df, "ip_score") >= thresholds.high_ip_score_q90
        ).astype(int)

    if "is_night_txn" in df.columns:
        df["night_transaction_flag"] = (
            _to_numeric(df, "is_night_txn").fillna(0) == 1
        ).astype(int)

    if "card_present" in df.columns:
        df["card_not_present_flag"] = _bool_false(df, "card_present").astype(int)

    if "cross_border" in df.columns:
        df["cross_border_flag"] = _bool_true(df, "cross_border").astype(int)

    if "tokenised" in df.columns:
        df["tokenised_flag"] = _bool_true(df, "tokenised").astype(int)

    return df


def create_time_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo feature liên quan thời gian và thói quen giao dịch ban đêm.
    """
    df = df.copy()

    if "is_night_txn" in df.columns and "night_ratio_30d" in df.columns:
        is_night = _to_numeric(df, "is_night_txn").fillna(0)
        night_ratio = _to_numeric(df, "night_ratio_30d").fillna(0)

        df["night_unusual_score"] = is_night * (1 - night_ratio)

        df["night_unusual_group"] = pd.cut(
            df["night_unusual_score"],
            bins=[-np.inf, 0, 0.3, 0.6, 0.9, np.inf],
            labels=["not_night", "low", "medium", "high", "very_high"],
            include_lowest=True,
        ).astype("object")

    return df


def create_behavior_features(
    df: pd.DataFrame,
    thresholds: BehaviorThresholds | None = None,
) -> pd.DataFrame:
    """
    Tạo feature hành vi.

    Trong pipeline chính:
    - fit thresholds trên train
    - apply thresholds sang train/validation/test
    """
    df = df.copy()

    if thresholds is None:
        thresholds = fit_behavior_thresholds(df)

    df = create_business_risk_bins(df)
    df = create_behavior_flags(df, thresholds=thresholds)
    df = create_time_behavior_features(df)

    return df