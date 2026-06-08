from pathlib import Path
from typing import Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DEFAULT_DATA_PATH = RAW_DATA_DIR / "fraud2.csv"
DEFAULT_METADATA_PATH = RAW_DATA_DIR / "fraud_metadata.xlsx"


def load_fraud_data(file_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load fraud transaction dataset.

    Parameters
    ----------
    file_path : str | Path | None
        Path to CSV file. If None, use data/raw/fraud2.csv.

    Returns
    -------
    pd.DataFrame
        Loaded fraud dataset.
    """
    path = Path(file_path) if file_path else DEFAULT_DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file dữ liệu: {path}\n"
            f"Hãy kiểm tra file CSV đã nằm trong thư mục data/raw/ chưa."
        )

    df = pd.read_csv(path)
    return df


def load_metadata(file_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load metadata file.

    Parameters
    ----------
    file_path : str | Path | None
        Path to Excel metadata file. If None, use data/raw/fraud_metadata.xlsx.

    Returns
    -------
    pd.DataFrame
        Loaded metadata.
    """
    path = Path(file_path) if file_path else DEFAULT_METADATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file metadata: {path}\n"
            f"Hãy kiểm tra file Excel đã nằm trong thư mục data/raw/ chưa."
        )

    metadata = pd.read_excel(path)
    return metadata


def preview_data(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Return first n rows of dataframe.
    """
    return df.head(n)


def get_basic_shape(df: pd.DataFrame) -> dict:
    """
    Return basic shape information.
    """
    return {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
    }


def get_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return column names and pandas data types.
    """
    result = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "non_null_count": [df[col].notna().sum() for col in df.columns],
            "null_count": [df[col].isna().sum() for col in df.columns],
            "unique_count": [df[col].nunique(dropna=True) for col in df.columns],
        }
    )

    result["null_rate"] = result["null_count"] / len(df)

    return result


def infer_feature_groups(df: pd.DataFrame, target_col: str = "fraud") -> dict:
    """
    Infer numeric, categorical, boolean, datetime-like, and high-cardinality columns.
    """
    numeric_cols = []
    categorical_cols = []
    boolean_cols = []
    datetime_like_cols = []
    high_cardinality_cols = []

    for col in df.columns:
        if col == target_col:
            continue

        nunique = df[col].nunique(dropna=True)
        dtype = df[col].dtype

        if pd.api.types.is_bool_dtype(dtype):
            boolean_cols.append(col)

        elif pd.api.types.is_numeric_dtype(dtype):
            numeric_cols.append(col)

        elif "time" in col.lower() or "date" in col.lower():
            datetime_like_cols.append(col)
            categorical_cols.append(col)

        else:
            categorical_cols.append(col)

        if nunique > 1000:
            high_cardinality_cols.append(col)

    return {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "boolean_cols": boolean_cols,
        "datetime_like_cols": datetime_like_cols,
        "high_cardinality_cols": high_cardinality_cols,
    }


def check_target_column(df: pd.DataFrame, target_col: str = "fraud") -> None:
    """
    Check target column existence.
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Không tìm thấy target column '{target_col}'. "
            f"Các cột hiện có: {list(df.columns)}"
        )