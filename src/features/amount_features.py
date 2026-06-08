from __future__ import annotations

import numpy as np
import pandas as pd


def _to_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    return pd.to_numeric(df[col], errors="coerce")


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    eps: float = 1.0,
) -> pd.Series:
    """
    Chia an toàn, tránh chia cho 0.
    """
    return numerator / (denominator + eps)


def create_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo feature liên quan amount.

    Nhóm này giúp model học được giao dịch hiện tại
    lệch khỏi lịch sử 30 ngày như thế nào.
    """
    df = df.copy()

    if "amount" not in df.columns:
        return df

    amount = _to_numeric(df, "amount")

    df["amount"] = amount
    df["log_amount"] = np.log1p(amount.clip(lower=0))
    df["amount_missing_flag"] = amount.isna().astype(int)
    df["amount_invalid_flag"] = ((amount.isna()) | (amount <= 0)).astype(int)

    if "mean_amount_30d" in df.columns:
        mean_amount_30d = _to_numeric(df, "mean_amount_30d")

        df["mean_amount_30d"] = mean_amount_30d
        df["log_mean_amount_30d"] = np.log1p(mean_amount_30d.clip(lower=0))
        df["mean_amount_30d_missing_flag"] = mean_amount_30d.isna().astype(int)
        df["mean_amount_30d_zero_flag"] = (mean_amount_30d.fillna(0) == 0).astype(int)

        df["amount_to_mean_30d"] = safe_divide(amount, mean_amount_30d, eps=1.0)
        df["log_amount_to_mean_30d"] = np.log1p(
            df["amount_to_mean_30d"].clip(lower=0)
        )

    if "max_amount_30d" in df.columns:
        max_amount_30d = _to_numeric(df, "max_amount_30d")

        df["max_amount_30d"] = max_amount_30d
        df["log_max_amount_30d"] = np.log1p(max_amount_30d.clip(lower=0))
        df["max_amount_30d_missing_flag"] = max_amount_30d.isna().astype(int)
        df["max_amount_30d_zero_flag"] = (max_amount_30d.fillna(0) == 0).astype(int)

        df["amount_to_max_30d"] = safe_divide(amount, max_amount_30d, eps=1.0)
        df["log_amount_to_max_30d"] = np.log1p(
            df["amount_to_max_30d"].clip(lower=0)
        )

    if "mean_amount_30d" in df.columns and "std_amount_30d" in df.columns:
        mean_amount_30d = _to_numeric(df, "mean_amount_30d")
        std_amount_30d = _to_numeric(df, "std_amount_30d")

        df["std_amount_30d"] = std_amount_30d
        df["std_amount_30d_missing_flag"] = std_amount_30d.isna().astype(int)
        df["std_amount_30d_zero_flag"] = (std_amount_30d.fillna(0) == 0).astype(int)

        df["amount_z_30d"] = (amount - mean_amount_30d) / (std_amount_30d + 1.0)
        df["amount_to_std_30d"] = safe_divide(amount, std_amount_30d, eps=1.0)

    if "amount_to_mean_30d" in df.columns:
        df["is_amount_above_mean"] = (df["amount_to_mean_30d"] > 1).astype(int)
        df["is_amount_2x_mean"] = (df["amount_to_mean_30d"] > 2).astype(int)
        df["is_amount_5x_mean"] = (df["amount_to_mean_30d"] > 5).astype(int)

    if "amount_to_max_30d" in df.columns:
        df["is_amount_near_max"] = (df["amount_to_max_30d"] >= 0.8).astype(int)
        df["is_amount_above_max"] = (df["amount_to_max_30d"] > 1).astype(int)

    return df