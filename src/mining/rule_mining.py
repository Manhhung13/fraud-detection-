from __future__ import annotations

from pathlib import Path

import pandas as pd


def classify_rule_strength(
    fraud_rate: float,
    lift: float,
    support: int,
) -> str:
    """
    Phân loại độ mạnh của rule theo nghiệp vụ.
    """
    if support < 30:
        return "WEAK_SUPPORT"

    if lift >= 5 and fraud_rate >= 0.10:
        return "VERY_STRONG"

    if lift >= 3 and fraud_rate >= 0.07:
        return "STRONG"

    if lift >= 2 and fraud_rate >= 0.05:
        return "MEDIUM"

    if lift >= 1.5:
        return "WEAK"

    return "LOW_SIGNAL"


def interpret_rule_business_meaning(rule: str) -> str:
    """
    Diễn giải rule sang ngôn ngữ nghiệp vụ.
    """
    rule_lower = str(rule).lower()

    if "low_mcc_entropy" in rule_lower and "low_night_ratio" in rule_lower:
        return (
            "User có lịch sử ngành hàng ít đa dạng và ít giao dịch ban đêm. "
            "Khi hai tín hiệu này cùng xuất hiện, fraud rate thường tăng mạnh. "
            "Đây là nhóm behavioral anomaly quan trọng."
        )

    if "low_mcc_entropy" in rule_lower and "high_max_amount" in rule_lower:
        return (
            "User có hành vi ngành hàng hẹp nhưng có lịch sử giao dịch giá trị rất cao. "
            "Nhóm này cần kiểm tra vì có thể phản ánh hành vi bất thường trong lịch sử gần."
        )

    if "low_mcc_entropy" in rule_lower and "low_spending_trend" in rule_lower:
        return (
            "User có ngành hàng giao dịch hẹp và xu hướng chi tiêu thấp hoặc giảm. "
            "Nếu phát sinh giao dịch mới lệch khỏi hành vi này thì rủi ro tăng."
        )

    if "card_not_present" in rule_lower:
        return (
            "Giao dịch không có thẻ vật lý. Với fraud online, card-not-present thường cần soi kỹ, "
            "đặc biệt khi đi kèm tín hiệu bất thường khác."
        )

    if "cross_border" in rule_lower and "low_country_diversity" in rule_lower:
        return (
            "Giao dịch xuyên biên giới trong khi user có lịch sử giao dịch ở ít quốc gia. "
            "Đây là tín hiệu country anomaly."
        )

    if "cross_border" in rule_lower:
        return (
            "Giao dịch xuyên biên giới. Risk phụ thuộc vào việc user có quen giao dịch nhiều quốc gia hay không."
        )

    if "high_ip_score" in rule_lower:
        return (
            "IP có điểm rủi ro cao. Đây là tín hiệu kỹ thuật quan trọng, nên kết hợp với hành vi user."
        )

    if "device_diversity" in rule_lower:
        return (
            "User có nhiều thiết bị trong thời gian gần. Có thể là dấu hiệu account sharing hoặc account takeover."
        )

    if "high_decline" in rule_lower:
        return (
            "User có tỷ lệ giao dịch bị từ chối cao. Đây là tín hiệu lịch sử rủi ro cần khai thác."
        )

    if "amount_to_mean" in rule_lower or "high_amount" in rule_lower:
        return (
            "Giao dịch có giá trị cao so với lịch sử. Đây là nhóm amount anomaly quan trọng."
        )

    if "payment_channel" in rule_lower and "merchant_country" in rule_lower:
        return (
            "Một số tổ hợp kênh thanh toán và quốc gia merchant có fraud rate cao hơn baseline."
        )

    return (
        "Rule có tín hiệu thống kê. Cần kiểm tra support, lift, fraud_rate và ý nghĩa nghiệp vụ "
        "trước khi đưa vào model."
    )


def add_rule_interpretation(
    report: pd.DataFrame,
    rule_col: str = "rule",
) -> pd.DataFrame:
    """
    Thêm cột strength và business_interpretation cho report.
    """
    if report is None or report.empty:
        return pd.DataFrame() if report is None else report

    if rule_col not in report.columns:
        return report

    result = report.copy()

    result["strength"] = result.apply(
        lambda row: classify_rule_strength(
            fraud_rate=float(row.get("fraud_rate", 0) or 0),
            lift=float(row.get("lift", 0) or 0),
            support=int(row.get("support", 0) or 0),
        ),
        axis=1,
    )

    result["business_interpretation"] = result[rule_col].apply(
        interpret_rule_business_meaning
    )

    return result


def select_actionable_rules(
    report: pd.DataFrame,
    min_lift: float = 2.0,
    min_support: int = 50,
    min_fraud_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Chọn rule có khả năng hành động:
    - support đủ lớn
    - lift cao
    - fraud_rate đủ cao
    """
    if report is None or report.empty:
        return pd.DataFrame()

    required_cols = ["support", "lift", "fraud_rate"]

    for col in required_cols:
        if col not in report.columns:
            return pd.DataFrame()

    result = report.copy()

    result = result[
        (result["support"] >= min_support)
        & (result["lift"] >= min_lift)
        & (result["fraud_rate"] >= min_fraud_rate)
    ]

    if result.empty:
        return pd.DataFrame()

    result = add_rule_interpretation(result, rule_col="rule")

    return result.sort_values(
        ["lift", "fraud_rate", "support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _df_to_markdown(df: pd.DataFrame, cols: list[str], top_n: int = 20) -> str:
    """
    Convert DataFrame to markdown an toàn.
    """
    if df is None or df.empty:
        return "Không có dữ liệu phù hợp."

    show_cols = [col for col in cols if col in df.columns]

    if not show_cols:
        return df.head(top_n).to_markdown(index=False)

    return df[show_cols].head(top_n).to_markdown(index=False)


def generate_insight_markdown_report(
    output_path: str | Path,
    global_fraud_rate: float,
    segment_report: pd.DataFrame,
    priority_interactions: pd.DataFrame,
    actionable_rules: pd.DataFrame,
    numeric_patterns: pd.DataFrame,
    categorical_patterns: pd.DataFrame,
) -> None:
    """
    Sinh report markdown tổng hợp insight.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    lines.append("# Fraud Detection - Hidden Insight Mining Report\n")

    lines.append("## 1. Tổng quan\n")
    lines.append(
        f"- Fraud rate toàn bộ dataset: **{global_fraud_rate:.4f} "
        f"({global_fraud_rate * 100:.2f}%)**"
    )
    lines.append(
        "- Mục tiêu giai đoạn này là tìm segment/rule có fraud rate cao hơn baseline, "
        "sau đó chuyển các pattern có ý nghĩa thành feature cho model."
    )

    lines.append("\n## 2. Segment nghiệp vụ nổi bật\n")
    lines.append(
        _df_to_markdown(
            segment_report,
            cols=[
                "segment_name",
                "support",
                "support_rate",
                "fraud_count",
                "fraud_rate",
                "lift",
                "strength",
                "description",
            ],
            top_n=15,
        )
    )

    lines.append("\n## 3. Interaction nghiệp vụ ưu tiên\n")
    lines.append(
        _df_to_markdown(
            priority_interactions,
            cols=[
                "rule",
                "support",
                "support_rate",
                "fraud_count",
                "fraud_rate",
                "lift",
                "strength",
                "business_interpretation",
            ],
            top_n=20,
        )
    )

    lines.append("\n## 4. Actionable rules nên chuyển thành feature\n")
    lines.append(
        _df_to_markdown(
            actionable_rules,
            cols=[
                "rule",
                "support",
                "fraud_count",
                "fraud_rate",
                "lift",
                "strength",
                "business_interpretation",
            ],
            top_n=20,
        )
    )

    lines.append("\n## 5. Numeric patterns nổi bật\n")
    lines.append(
        _df_to_markdown(
            numeric_patterns,
            cols=[
                "feature",
                "bin",
                "count",
                "fraud_count",
                "fraud_rate",
                "global_fraud_rate",
                "fraud_rate_diff",
                "lift",
                "min_value",
                "max_value",
            ],
            top_n=20,
        )
    )

    lines.append("\n## 6. Categorical patterns nổi bật\n")
    lines.append(
        _df_to_markdown(
            categorical_patterns,
            cols=[
                "feature",
                "value",
                "count",
                "fraud_count",
                "fraud_rate",
                "global_fraud_rate",
                "fraud_rate_diff",
                "lift",
            ],
            top_n=20,
        )
    )

    lines.append("\n## 7. Khuyến nghị feature engineering\n")
    lines.append(
        """
Dựa trên insight mining, nên ưu tiên tạo các nhóm feature sau:

1. **Behavioral anomaly features**
   - amount_to_mean_30d
   - amount_to_max_30d
   - night_unusual_score
   - txn_count_ratio_7d_30d

2. **MCC behavior features**
   - low_mcc_entropy_flag
   - mcc_entropy_30d
   - low_mcc_entropy_x_low_night_ratio
   - low_mcc_entropy_x_high_max_amount_30d

3. **Time anomaly features**
   - is_night_txn
   - night_ratio_30d
   - night_unusual_group
   - low_night_ratio_x_high_max_amount_30d

4. **Country/channel behavior**
   - cross_border_flag
   - low_country_diversity_flag
   - cross_border_x_low_country_diversity
   - payment_channel_x_merchant_country

5. **Interaction features cho Logistic Regression**
   - low_mcc_entropy_x_low_night_ratio
   - low_mcc_entropy_x_high_max_amount_30d
   - low_mcc_entropy_x_low_spending_trend
   - low_mcc_entropy_x_card_not_present
   - high_ip_score_x_card_not_present
"""
    )

    lines.append("\n## 8. Lưu ý leakage\n")
    lines.append(
        """
Các biến lịch sử như `mean_amount_30d`, `max_amount_30d`, `decline_rate_30d`,
`chargebacks_365d`, `txn_counts` cần đảm bảo được tính từ dữ liệu **trước thời điểm giao dịch hiện tại**.

Nếu các biến này bao gồm giao dịch hiện tại hoặc thông tin tương lai, model sẽ bị leakage.
"""
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")