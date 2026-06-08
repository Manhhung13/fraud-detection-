from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


def evaluate_rule_condition(
    df: pd.DataFrame,
    condition: pd.Series,
    rule_name: str,
    target_col: str = "fraud",
) -> dict:
    """
    Đánh giá một rule bất kỳ:
    - support
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

    subset_target = target[condition]
    support = int(condition.sum())

    if support == 0:
        return {
            "rule": rule_name,
            "support": 0,
            "support_rate": 0.0,
            "fraud_count": 0,
            "fraud_rate": 0.0,
            "global_fraud_rate": global_fraud_rate,
            "fraud_rate_diff": 0.0,
            "lift": 0.0,
        }

    fraud_count = int(subset_target.sum())
    fraud_rate = float(subset_target.mean())
    lift = float(fraud_rate / global_fraud_rate) if global_fraud_rate > 0 else np.nan

    return {
        "rule": rule_name,
        "support": support,
        "support_rate": support / total_count if total_count > 0 else 0.0,
        "fraud_count": fraud_count,
        "fraud_rate": fraud_rate,
        "global_fraud_rate": global_fraud_rate,
        "fraud_rate_diff": fraud_rate - global_fraud_rate,
        "lift": lift,
    }


def mine_boolean_interactions(
    df: pd.DataFrame,
    flag_cols: Iterable[str],
    target_col: str = "fraud",
    min_support: int = 50,
    max_order: int = 3,
) -> pd.DataFrame:
    """
    Tìm interaction giữa các feature dạng flag 0/1.

    Ví dụ:
    - low_mcc_entropy_flag AND low_night_ratio_flag
    - low_mcc_entropy_flag AND high_max_amount_30d_flag
    """
    rows = []

    flag_cols = [col for col in flag_cols if col in df.columns]

    if len(flag_cols) < 2:
        return pd.DataFrame()

    for order in range(2, max_order + 1):
        for cols in combinations(flag_cols, order):
            condition = pd.Series(True, index=df.index)

            for col in cols:
                condition = condition & (pd.to_numeric(df[col], errors="coerce").fillna(0) == 1)

            rule_name = " AND ".join(cols)

            row = evaluate_rule_condition(
                df=df,
                condition=condition,
                rule_name=rule_name,
                target_col=target_col,
            )

            if row["support"] >= min_support:
                row["order"] = order
                rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    return result.sort_values(
        ["lift", "fraud_rate", "support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def mine_categorical_pair_interactions(
    df: pd.DataFrame,
    categorical_pairs: list[tuple[str, str]],
    target_col: str = "fraud",
    min_support: int = 50,
) -> pd.DataFrame:
    """
    Tìm fraud rate theo tổ hợp 2 biến categorical.

    Ví dụ:
    - payment_channel + merchant_country
    - card_entry_mode + auth_result
    """
    if target_col not in df.columns:
        raise ValueError(f"Không tìm thấy target column: {target_col}")

    rows = []

    target = pd.to_numeric(df[target_col], errors="coerce")
    valid_mask = target.notna()

    global_rate = float(target[valid_mask].mean())
    total_count = int(valid_mask.sum())

    for col_a, col_b in categorical_pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue

        tmp = df.loc[valid_mask, [col_a, col_b]].copy()
        tmp[target_col] = target[valid_mask]

        grouped = (
            tmp.groupby([col_a, col_b], dropna=False)
            .agg(
                support=(target_col, "size"),
                fraud_count=(target_col, "sum"),
                fraud_rate=(target_col, "mean"),
            )
            .reset_index()
        )

        grouped = grouped[grouped["support"] >= min_support]

        for _, row in grouped.iterrows():
            fraud_rate = float(row["fraud_rate"])
            lift = fraud_rate / global_rate if global_rate > 0 else np.nan

            rows.append(
                {
                    "rule": f"{col_a}={row[col_a]} AND {col_b}={row[col_b]}",
                    "feature_a": col_a,
                    "value_a": row[col_a],
                    "feature_b": col_b,
                    "value_b": row[col_b],
                    "support": int(row["support"]),
                    "support_rate": float(row["support"] / total_count) if total_count > 0 else 0.0,
                    "fraud_count": int(row["fraud_count"]),
                    "fraud_rate": fraud_rate,
                    "global_fraud_rate": global_rate,
                    "fraud_rate_diff": fraud_rate - global_rate,
                    "lift": lift,
                    "order": 2,
                }
            )

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    return result.sort_values(
        ["lift", "fraud_rate", "support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_priority_business_interactions(df: pd.DataFrame) -> dict[str, pd.Series]:
    """
    Các interaction nghiệp vụ ưu tiên.

    Lưu ý:
    File này chỉ đánh giá rule, không tạo feature.
    Các flag phải được tạo trước ở:
    - src/features/behavior_features.py
    - src/features/interaction_features.py
    """
    rules: dict[str, pd.Series] = {}

    def has(col: str) -> bool:
        return col in df.columns

    def flag(col: str) -> pd.Series:
        return pd.to_numeric(df[col], errors="coerce").fillna(0) == 1

    if has("low_mcc_entropy_flag") and has("low_night_ratio_flag"):
        rules["low_mcc_entropy AND low_night_ratio"] = (
            flag("low_mcc_entropy_flag") & flag("low_night_ratio_flag")
        )

    if has("low_mcc_entropy_flag") and has("high_max_amount_30d_flag"):
        rules["low_mcc_entropy AND high_max_amount_30d"] = (
            flag("low_mcc_entropy_flag") & flag("high_max_amount_30d_flag")
        )

    if has("low_mcc_entropy_flag") and has("low_spending_trend_flag"):
        rules["low_mcc_entropy AND low_spending_trend"] = (
            flag("low_mcc_entropy_flag") & flag("low_spending_trend_flag")
        )

    if has("low_mcc_entropy_flag") and has("card_not_present_flag"):
        rules["low_mcc_entropy AND card_not_present"] = (
            flag("low_mcc_entropy_flag") & flag("card_not_present_flag")
        )

    if has("low_night_ratio_flag") and has("night_transaction_flag"):
        rules["low_night_ratio AND night_transaction"] = (
            flag("low_night_ratio_flag") & flag("night_transaction_flag")
        )

    if has("low_night_ratio_flag") and has("high_max_amount_30d_flag"):
        rules["low_night_ratio AND high_max_amount_30d"] = (
            flag("low_night_ratio_flag") & flag("high_max_amount_30d_flag")
        )

    if has("cross_border_flag") and has("low_country_diversity_flag"):
        rules["cross_border AND low_country_diversity"] = (
            flag("cross_border_flag") & flag("low_country_diversity_flag")
        )

    if has("high_ip_score_flag") and has("card_not_present_flag"):
        rules["high_ip_score AND card_not_present"] = (
            flag("high_ip_score_flag") & flag("card_not_present_flag")
        )

    if has("high_device_diversity_flag") and has("card_not_present_flag"):
        rules["high_device_diversity AND card_not_present"] = (
            flag("high_device_diversity_flag") & flag("card_not_present_flag")
        )

    if has("high_decline_rate_flag") and has("card_not_present_flag"):
        rules["high_decline_rate AND card_not_present"] = (
            flag("high_decline_rate_flag") & flag("card_not_present_flag")
        )

    if (
        has("low_mcc_entropy_flag")
        and has("low_night_ratio_flag")
        and has("card_not_present_flag")
    ):
        rules["low_mcc_entropy AND low_night_ratio AND card_not_present"] = (
            flag("low_mcc_entropy_flag")
            & flag("low_night_ratio_flag")
            & flag("card_not_present_flag")
        )

    if (
        has("low_mcc_entropy_flag")
        and has("low_night_ratio_flag")
        and has("high_max_amount_30d_flag")
    ):
        rules["low_mcc_entropy AND low_night_ratio AND high_max_amount_30d"] = (
            flag("low_mcc_entropy_flag")
            & flag("low_night_ratio_flag")
            & flag("high_max_amount_30d_flag")
        )

    return rules


def analyze_priority_business_interactions(
    df: pd.DataFrame,
    target_col: str = "fraud",
    min_support: int = 30,
) -> pd.DataFrame:
    """
    Đánh giá các interaction nghiệp vụ ưu tiên.
    """
    rows = []

    rules = build_priority_business_interactions(df)

    for rule_name, condition in rules.items():
        row = evaluate_rule_condition(
            df=df,
            condition=condition,
            rule_name=rule_name,
            target_col=target_col,
        )

        if row["support"] >= min_support:
            row["order"] = rule_name.count(" AND ") + 1
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    return result.sort_values(
        ["lift", "fraud_rate", "support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def summarize_interactions_for_feature_engineering(
    interaction_report: pd.DataFrame,
    min_lift: float = 2.0,
    min_support: int = 50,
) -> pd.DataFrame:
    """
    Chọn interaction đủ mạnh để đề xuất đưa vào feature engineering.
    """
    if interaction_report.empty:
        return pd.DataFrame()

    result = interaction_report.copy()

    result = result[
        (result["lift"] >= min_lift)
        & (result["support"] >= min_support)
    ]

    if result.empty:
        return pd.DataFrame()

    result["feature_recommendation"] = (
        "Nên tạo binary interaction feature cho Logistic Regression"
    )

    return result.sort_values(
        ["lift", "fraud_rate", "support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)