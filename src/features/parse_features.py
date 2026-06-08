import ast
import re
from typing import Any

import numpy as np
import pandas as pd


def safe_literal_eval(value: Any):
    """
    Parse string dạng dict/list/tuple an toàn hơn eval.

    Ví dụ:
    "{'ip': '190.3.86.171', 'score': 0.1467}"
    "(13, 21)"
    """
    if pd.isna(value):
        return None

    if isinstance(value, (dict, list, tuple)):
        return value

    if not isinstance(value, str):
        return None

    value = value.strip()

    if value == "":
        return None

    try:
        return ast.literal_eval(value)
    except Exception:
        return None


def parse_ip_risk_value(value: Any) -> dict:
    """
    Parse 1 giá trị ip_risk.

    Input có thể dạng:
    "{'ip': '190.3.86.171', 'score': 0.1467}"

    Output:
    {
        "ip_address": "190.3.86.171",
        "ip_score": 0.1467
    }
    """
    parsed = safe_literal_eval(value)

    if not isinstance(parsed, dict):
        return {
            "ip_address": np.nan,
            "ip_score": np.nan,
        }

    ip_address = parsed.get("ip", np.nan)
    ip_score = parsed.get("score", np.nan)

    try:
        ip_score = float(ip_score)
    except Exception:
        ip_score = np.nan

    return {
        "ip_address": ip_address,
        "ip_score": ip_score,
    }


def parse_ip_risk(df: pd.DataFrame, column: str = "ip_risk") -> pd.DataFrame:
    """
    Tách cột ip_risk thành:
    - ip_address
    - ip_score
    """
    df = df.copy()

    if column not in df.columns:
        return df

    parsed_rows = df[column].apply(parse_ip_risk_value)
    parsed_df = pd.DataFrame(parsed_rows.tolist(), index=df.index)

    df["ip_address"] = parsed_df["ip_address"]
    df["ip_score"] = parsed_df["ip_score"]

    return df


def parse_txn_counts_value(value: Any) -> dict:
    """
    Parse 1 giá trị txn_counts.

    Input có thể dạng:
    "(13, 21)"

    Giả định:
    - phần tử đầu: txn_count_7d
    - phần tử sau: txn_count_30d
    """
    parsed = safe_literal_eval(value)

    if not isinstance(parsed, (tuple, list)):
        return {
            "txn_count_7d": np.nan,
            "txn_count_30d": np.nan,
        }

    if len(parsed) < 2:
        return {
            "txn_count_7d": np.nan,
            "txn_count_30d": np.nan,
        }

    try:
        txn_count_7d = float(parsed[0])
    except Exception:
        txn_count_7d = np.nan

    try:
        txn_count_30d = float(parsed[1])
    except Exception:
        txn_count_30d = np.nan

    return {
        "txn_count_7d": txn_count_7d,
        "txn_count_30d": txn_count_30d,
    }


def parse_txn_counts(df: pd.DataFrame, column: str = "txn_counts") -> pd.DataFrame:
    """
    Tách cột txn_counts thành:
    - txn_count_7d
    - txn_count_30d
    - txn_count_ratio_7d_30d
    """
    df = df.copy()

    if column not in df.columns:
        return df

    parsed_rows = df[column].apply(parse_txn_counts_value)
    parsed_df = pd.DataFrame(parsed_rows.tolist(), index=df.index)

    df["txn_count_7d"] = parsed_df["txn_count_7d"]
    df["txn_count_30d"] = parsed_df["txn_count_30d"]

    df["txn_count_ratio_7d_30d"] = (
        df["txn_count_7d"] / (df["txn_count_30d"] + 1)
    )

    return df


def is_private_ip(ip: Any) -> int:
    """
    Kiểm tra IP private đơn giản.

    Private range phổ biến:
    - 10.x.x.x
    - 172.16.x.x đến 172.31.x.x
    - 192.168.x.x
    """
    if pd.isna(ip):
        return 0

    ip = str(ip)

    if ip.startswith("10."):
        return 1

    if ip.startswith("192.168."):
        return 1

    match = re.match(r"^172\.(\d+)\.", ip)
    if match:
        second_octet = int(match.group(1))
        if 16 <= second_octet <= 31:
            return 1

    return 0


def create_ip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo thêm feature từ IP sau khi parse.
    """
    df = df.copy()

    if "ip_address" in df.columns:
        df["ip_is_private"] = df["ip_address"].apply(is_private_ip)

    if "ip_score" in df.columns:
        df["ip_score_bin"] = pd.cut(
            df["ip_score"],
            bins=[-np.inf, 0.2, 0.5, 0.8, np.inf],
            labels=["low", "medium", "high", "very_high"],
        ).astype("object")

    return df


def parse_complex_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hàm tổng hợp parse các cột phức tạp.
    """
    df = df.copy()

    df = parse_ip_risk(df)
    df = parse_txn_counts(df)
    df = create_ip_features(df)

    return df