from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.features.parse_features import parse_complex_features
from src.features.time_features import add_time_features


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa tên cột:
    - bỏ khoảng trắng đầu/cuối
    - chuyển về chữ thường
    - thay dấu cách và dấu '-' bằng '_'
    """
    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    return df


def convert_boolean_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa các cột boolean nếu có dạng True/False, yes/no, 1/0.
    """
    df = df.copy()

    bool_mapping = {
        "true": 1,
        "false": 0,
        "yes": 1,
        "no": 0,
        "y": 1,
        "n": 0,
        "1": 1,
        "0": 0,
    }

    for col in df.columns:
        if df[col].dtype == "object":
            unique_values = (
                df[col]
                .dropna()
                .astype(str)
                .str.lower()
                .str.strip()
                .unique()
            )

            if len(unique_values) > 0 and set(unique_values).issubset(set(bool_mapping.keys())):
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.lower()
                    .str.strip()
                    .map(bool_mapping)
                )

    return df


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ép kiểu một số cột nên là numeric.

    errors='ignore' để tránh làm hỏng các cột categorical.
    """
    df = df.copy()

    numeric_candidates = [
        "amount",
        "mean_amount_30d",
        "std_amount_30d",
        "max_amount_30d",
        "night_ratio_30d",
        "mcc_entropy_30d",
        "distinct_merchants_7d",
        "distinct_countries_30d",
        "device_diversity_30d",
        "decline_rate_30d",
        "chargebacks_365d",
        "credit_util_today",
        "spending_trend_30d",
        "fraud",
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch cột dạng chuỗi:
    - strip khoảng trắng
    - thay chuỗi rỗng bằng NaN
    """
    df = df.copy()

    object_cols = df.select_dtypes(include=["object"]).columns

    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    return df


def handle_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Xử lý một số giá trị không hợp lệ cơ bản.

    Lưu ý:
    - Không drop quá mạnh ở giai đoạn này.
    - Chủ yếu tạo flag để phân tích.
    """
    df = df.copy()

    if "amount" in df.columns:
        df["amount_invalid_flag"] = (
            (df["amount"].isna()) |
            (df["amount"] <= 0)
        ).astype(int)

    if "std_amount_30d" in df.columns:
        df["std_amount_30d_zero_flag"] = (
            df["std_amount_30d"].fillna(0) == 0
        ).astype(int)

    if "mean_amount_30d" in df.columns:
        df["mean_amount_30d_zero_flag"] = (
            df["mean_amount_30d"].fillna(0) == 0
        ).astype(int)

    return df


def remove_duplicates(
    df: pd.DataFrame,
    drop_duplicates: bool = False,
) -> pd.DataFrame:
    """
    Kiểm tra hoặc xóa duplicate.

    Mặc định không xóa, chỉ tạo flag.
    """
    df = df.copy()

    df["duplicate_row_flag"] = df.duplicated().astype(int)

    if drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)

    return df


def clean_raw_data(
    df: pd.DataFrame,
    drop_duplicates: bool = False,
) -> pd.DataFrame:
    """
    Pipeline làm sạch dữ liệu raw.

    Các bước:
    1. Chuẩn hóa tên cột
    2. Làm sạch string
    3. Ép kiểu numeric
    4. Parse ip_risk, txn_counts
    5. Tạo time features
    6. Tạo flag invalid/duplicate
    """
    df = df.copy()

    df = normalize_column_names(df)
    df = clean_string_columns(df)
    df = convert_boolean_like_columns(df)
    df = convert_numeric_columns(df)

    df = parse_complex_features(df)
    df = add_time_features(df)

    df = handle_invalid_values(df)
    df = remove_duplicates(df, drop_duplicates=drop_duplicates)

    return df


def save_cleaned_data(
    df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Lưu cleaned data.

    Ưu tiên parquet vì nhẹ và giữ dtype tốt hơn CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    elif output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    else:
        raise ValueError("Output path phải có đuôi .parquet hoặc .csv")


def load_cleaned_data(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Load cleaned data.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    if file_path.suffix == ".parquet":
        return pd.read_parquet(file_path)

    if file_path.suffix == ".csv":
        return pd.read_csv(file_path)

    raise ValueError("File phải có đuôi .parquet hoặc .csv")


def get_cleaning_summary(
    raw_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tạo bảng summary trước/sau cleaning.
    """
    summary = {
        "raw_rows": raw_df.shape[0],
        "raw_columns": raw_df.shape[1],
        "cleaned_rows": cleaned_df.shape[0],
        "cleaned_columns": cleaned_df.shape[1],
        "new_columns_added": cleaned_df.shape[1] - raw_df.shape[1],
        "raw_missing_values": raw_df.isna().sum().sum(),
        "cleaned_missing_values": cleaned_df.isna().sum().sum(),
    }

    if "duplicate_row_flag" in cleaned_df.columns:
        summary["duplicate_rows"] = int(cleaned_df["duplicate_row_flag"].sum())

    if "amount_invalid_flag" in cleaned_df.columns:
        summary["invalid_amount_rows"] = int(cleaned_df["amount_invalid_flag"].sum())
    return pd.DataFrame([summary])