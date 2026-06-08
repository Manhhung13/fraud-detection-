import numpy as np
import pandas as pd


def create_time_features(
    df: pd.DataFrame,
    timestamp_col: str = "local_timestamp",
) -> pd.DataFrame:
    """
    Tạo feature thời gian từ local_timestamp.

    Output:
    - transaction_datetime
    - hour
    - day_of_week
    - day_name
    - is_weekend
    - is_night_txn
    - month
    - day
    - year
    - time_period
    """
    df = df.copy()

    if timestamp_col not in df.columns:
        return df

    df["transaction_datetime"] = pd.to_datetime(
        df[timestamp_col],
        errors="coerce",
    )

    df["hour"] = df["transaction_datetime"].dt.hour
    df["day_of_week"] = df["transaction_datetime"].dt.dayofweek
    df["day_name"] = df["transaction_datetime"].dt.day_name()

    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Giao dịch ban đêm: từ 22h đến trước 6h sáng
    df["is_night_txn"] = (
        (df["hour"] >= 22) | (df["hour"] < 6)
    ).astype(int)

    df["month"] = df["transaction_datetime"].dt.month
    df["day"] = df["transaction_datetime"].dt.day
    df["year"] = df["transaction_datetime"].dt.year

    # Khoảng thời gian trong ngày
    df["time_period"] = pd.cut(
        df["hour"],
        bins=[-1, 5, 11, 17, 21, 23],
        labels=["night", "morning", "afternoon", "evening", "late_evening"],
    ).astype("object")

    return df


def create_time_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo feature kết hợp giữa thời điểm giao dịch hiện tại
    và lịch sử hành vi giao dịch ban đêm.
    """
    df = df.copy()

    if "is_night_txn" in df.columns and "night_ratio_30d" in df.columns:
        df["night_unusual_score"] = (
            df["is_night_txn"] * (1 - df["night_ratio_30d"])
        )

        df["is_night_unusual"] = (
            (df["is_night_txn"] == 1)
            & (df["night_ratio_30d"] <= df["night_ratio_30d"].quantile(0.1))
        ).astype(int)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm tổng hợp tạo feature thời gian.
    """
    df = df.copy()

    df = create_time_features(df)
    df = create_time_behavior_features(df)

    return df