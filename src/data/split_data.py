from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split


def _ensure_datetime(
    df: pd.DataFrame,
    timestamp_col: str = "transaction_datetime",
    fallback_col: str = "local_timestamp",
) -> pd.DataFrame:
    df = df.copy()

    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        return df

    if fallback_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[fallback_col], errors="coerce")
        return df

    raise ValueError(
        f"Không tìm thấy cột thời gian '{timestamp_col}' hoặc '{fallback_col}'."
    )


def get_split_summary(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = "fraud",
) -> pd.DataFrame:
    rows = []

    for name, part in [
        ("train", train_df),
        ("validation", valid_df),
        ("test", test_df),
    ]:
        target = pd.to_numeric(part[target_col], errors="coerce")

        rows.append(
            {
                "split": name,
                "rows": len(part),
                "fraud_count": int(target.sum()),
                "non_fraud_count": int((target == 0).sum()),
                "fraud_rate": float(target.mean()),
                "start_time": part["transaction_datetime"].min()
                if "transaction_datetime" in part.columns
                else None,
                "end_time": part["transaction_datetime"].max()
                if "transaction_datetime" in part.columns
                else None,
            }
        )

    return pd.DataFrame(rows)


def time_based_split(
    df: pd.DataFrame,
    target_col: str = "fraud",
    timestamp_col: str = "transaction_datetime",
    fallback_col: str = "local_timestamp",
    train_size: float = 0.70,
    valid_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_size + valid_size + test_size - 1.0) > 1e-9:
        raise ValueError("train_size + valid_size + test_size phải bằng 1.0")

    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy target column: {target_col}")

    df = _ensure_datetime(df, timestamp_col, fallback_col)
    df = df.dropna(subset=[timestamp_col]).copy()
    df = df.sort_values(timestamp_col).reset_index(drop=True)

    n = len(df)

    train_end = int(n * train_size)
    valid_end = int(n * (train_size + valid_size))

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    test_df = df.iloc[valid_end:].copy()

    return train_df, valid_df, test_df


def stratified_split(
    df: pd.DataFrame,
    target_col: str = "fraud",
    train_size: float = 0.70,
    valid_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_size + valid_size + test_size - 1.0) > 1e-9:
        raise ValueError("train_size + valid_size + test_size phải bằng 1.0")

    y = df[target_col]

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=y,
        random_state=random_state,
    )

    relative_valid_size = valid_size / (valid_size + test_size)

    valid_df, test_df = train_test_split(
        temp_df,
        train_size=relative_valid_size,
        stratify=temp_df[target_col],
        random_state=random_state,
    )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def split_fraud_data(
    df: pd.DataFrame,
    target_col: str = "fraud",
    timestamp_col: str = "transaction_datetime",
    fallback_col: str = "local_timestamp",
    train_size: float = 0.70,
    valid_size: float = 0.15,
    test_size: float = 0.15,
    min_fraud_per_split: int = 50,
    random_state: int = 42,
    strategy: Literal["time", "stratified", "auto"] = "time",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chia dữ liệu cho fraud detection.

    Với dataset hiện tại nên dùng strategy='time'.

    strategy:
    - time: chia theo thời gian, đúng nghiệp vụ fraud detection
    - stratified: chia random stratified
    - auto: ưu tiên time, nếu split nào quá ít fraud thì fallback stratified
    """
    if strategy not in ["time", "stratified", "auto"]:
        raise ValueError("strategy phải là 'time', 'stratified', hoặc 'auto'")

    if strategy == "stratified":
        train_df, valid_df, test_df = stratified_split(
            df=df,
            target_col=target_col,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
            random_state=random_state,
        )

        summary = get_split_summary(train_df, valid_df, test_df, target_col)
        summary["split_strategy"] = "stratified"

        return train_df, valid_df, test_df, summary

    try:
        train_df, valid_df, test_df = time_based_split(
            df=df,
            target_col=target_col,
            timestamp_col=timestamp_col,
            fallback_col=fallback_col,
            train_size=train_size,
            valid_size=valid_size,
            test_size=test_size,
        )

        summary = get_split_summary(train_df, valid_df, test_df, target_col)
        summary["split_strategy"] = "time"

        if strategy == "time":
            return train_df, valid_df, test_df, summary

        fraud_counts = summary["fraud_count"].tolist()

        if min(fraud_counts) >= min_fraud_per_split:
            return train_df, valid_df, test_df, summary

        print(
            "[WARNING] Time-based split có split quá ít fraud. "
            "Fallback sang stratified split."
        )

    except Exception as exc:
        if strategy == "time":
            raise exc

        print(f"[WARNING] Không chia được time-based split: {exc}")
        print("[INFO] Fallback sang stratified split.")

    train_df, valid_df, test_df = stratified_split(
        df=df,
        target_col=target_col,
        train_size=train_size,
        valid_size=valid_size,
        test_size=test_size,
        random_state=random_state,
    )

    summary = get_split_summary(train_df, valid_df, test_df, target_col)
    summary["split_strategy"] = "stratified"

    return train_df, valid_df, test_df, summary


def save_data_splits(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(output_dir / "train.parquet", index=False)
    valid_df.to_parquet(output_dir / "validation.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)