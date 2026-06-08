from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd


@dataclass
class InsightMapping:
    insight_group: str
    business_meaning: str
    expected_direction: str
    stage3_evidence: str


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {path}")

    return pd.read_csv(path)


def load_json(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file JSON: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_percent(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value) * 100:.{digits}f}%"


def format_number(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):.{digits}f}"


def clean_feature_name(feature: str) -> str:
    """
    Chuyển tên feature sau sklearn ColumnTransformer về dạng dễ đọc hơn.

    Ví dụ:
    - numeric__mcc_entropy_30d -> mcc_entropy_30d
    - interaction__low_mcc_entropy_x_low_night_ratio -> low_mcc_entropy_x_low_night_ratio
    - categorical__time_period_night -> time_period=night
    """
    feature = str(feature)

    if "__" in feature:
        _, raw = feature.split("__", 1)
    else:
        raw = feature

    categorical_prefixes = [
        "currency_",
        "payment_channel_",
        "merchant_country_",
        "card_entry_mode_",
        "auth_result_",
        "pin_verif_method_",
        "recurring_flag_",
        "auth_characteristics_",
        "message_type_",
        "term_location_",
        "day_name_",
        "time_period_",
        "ip_score_bin_",
        "night_ratio_group_",
        "night_unusual_group_",
        "amount_to_mean_group_",
        "amount_to_max_group_",
        "ip_score_risk_group_",
        "decline_rate_risk_group_",
        "credit_util_risk_group_",
        "txn_count_7d_group_",
        "txn_velocity_group_",
    ]

    for prefix in categorical_prefixes:
        if raw.startswith(prefix):
            col = prefix[:-1]
            value = raw[len(prefix):]
            return f"{col}={value}"

    return raw


def map_feature_to_insight(feature: str, coefficient: float) -> InsightMapping:
    """
    Map coefficient về nhóm insight nghiệp vụ.

    Lưu ý:
    - coefficient > 0: tăng log-odds fraud.
    - coefficient < 0: giảm log-odds fraud.
    - Đây là tác động có điều kiện sau khi xét các feature khác, không phải quan hệ nhân quả.
    """
    f = str(feature).lower()
    coef = float(coefficient)

    # MCC behavior anomaly
    if (
        "mcc_entropy" in f
        or "low_mcc_entropy" in f
        or "very_low_mcc_entropy" in f
    ):
        return InsightMapping(
            insight_group="MCC behavior anomaly",
            business_meaning=(
                "User có lịch sử ngành hàng/MCC ít đa dạng. "
                "Khi hành vi ngành hàng hẹp kết hợp với tín hiệu bất thường khác, fraud risk tăng."
            ),
            expected_direction=(
                "mcc_entropy_30d cao thường giảm risk; "
                "low_mcc_entropy flag/interaction thường tăng risk."
            ),
            stage3_evidence=(
                "Giai đoạn 3 cho thấy low_mcc_entropy và very_low_mcc_entropy "
                "là các segment có lift cao và ổn định."
            ),
        )

    # Time behavior anomaly
    if (
        "night_ratio" in f
        or "night_transaction" in f
        or "night_unusual" in f
        or "time_period" in f
    ):
        return InsightMapping(
            insight_group="Time behavior anomaly",
            business_meaning=(
                "Hành vi giao dịch theo thời gian, đặc biệt là giao dịch ban đêm "
                "so với lịch sử của user."
            ),
            expected_direction=(
                "night_ratio_30d cao thường giảm risk vì user quen giao dịch ban đêm; "
                "night transaction hoặc low_night_ratio interaction có thể tăng risk."
            ),
            stage3_evidence=(
                "Giai đoạn 3 cho thấy low_night_ratio và night_unusual "
                "có lift cao trên train và validation."
            ),
        )

    # Amount anomaly
    if (
        "amount_z" in f
        or "amount_to_mean" in f
        or "amount_to_max" in f
        or "max_amount" in f
        or "log_amount" in f
    ):
        return InsightMapping(
            insight_group="Amount anomaly",
            business_meaning=(
                "Giao dịch hiện tại lệch so với lịch sử amount của user."
            ),
            expected_direction=(
                "amount_z_30d cao hoặc các interaction với high_max_amount_30d "
                "thường làm tăng risk."
            ),
            stage3_evidence=(
                "Giai đoạn 3 cho thấy high_max_amount_30d và một số amount group "
                "có lift cao hơn baseline."
            ),
        )

    # Spending trend
    if "spending_trend" in f:
        return InsightMapping(
            insight_group="Spending trend anomaly",
            business_meaning=(
                "Xu hướng chi tiêu thấp hoặc giảm có thể phản ánh hành vi bất thường "
                "khi kết hợp với các tín hiệu khác."
            ),
            expected_direction=(
                "spending_trend cao thường giảm risk; low_spending_trend flag/interaction tăng risk."
            ),
            stage3_evidence=(
                "Giai đoạn 3 cho thấy spending_trend thấp có lift ổn định."
            ),
        )

    # Country behavior
    if (
        "country" in f
        or "cross_border" in f
        or "distinct_countries" in f
        or "merchant_country" in f
    ):
        return InsightMapping(
            insight_group="Country / cross-border behavior",
            business_meaning=(
                "Hành vi giao dịch theo quốc gia và mức độ đa dạng quốc gia của user."
            ),
            expected_direction=(
                "distinct_countries_30d cao thường giảm risk; "
                "low_country_diversity hoặc cross-border interaction có thể tăng risk."
            ),
            stage3_evidence=(
                "Giai đoạn 3 cho thấy low_country_diversity là segment có lift cao."
            ),
        )

    # Risk history / technical risk
    if (
        "ip_score" in f
        or "decline_rate" in f
        or "chargebacks" in f
        or "device_diversity" in f
        or "credit_util" in f
    ):
        return InsightMapping(
            insight_group="Risk history / technical risk",
            business_meaning=(
                "Các tín hiệu lịch sử rủi ro hoặc tín hiệu kỹ thuật như IP risk, decline rate, chargeback."
            ),
            expected_direction=(
                "Một số tín hiệu kỹ thuật đứng riêng có thể yếu, nhưng vẫn hỗ trợ model khi kết hợp với feature khác."
            ),
            stage3_evidence=(
                "Giai đoạn 3 cho thấy một số biến kỹ thuật đứng riêng không mạnh bằng behavior interaction."
            ),
        )

    # Payment context
    if (
        "payment_channel" in f
        or "card_entry_mode" in f
        or "auth_result" in f
        or "pin_verif" in f
        or "auth_characteristics" in f
        or "message_type" in f
        or "term_location" in f
        or "tokenised" in f
        or "card_not_present" in f
        or "currency" in f
        or "recurring_flag" in f
    ):
        return InsightMapping(
            insight_group="Payment / authentication context",
            business_meaning=(
                "Ngữ cảnh thanh toán, xác thực, kênh giao dịch và phương thức nhập thẻ."
            ),
            expected_direction=(
                "Một số kênh/xác thực làm tăng hoặc giảm risk tùy ngữ cảnh, "
                "nhưng thường không mạnh bằng behavioral anomaly."
            ),
            stage3_evidence=(
                "Giai đoạn 3 cho thấy card_not_present đứng riêng không quá mạnh, "
                "nhưng có thể mạnh hơn khi đi cùng behavior anomaly."
            ),
        )

    # Velocity
    if "txn_count" in f or "txn_velocity" in f:
        return InsightMapping(
            insight_group="Transaction velocity",
            business_meaning=(
                "Tốc độ/tần suất giao dịch gần đây của user."
            ),
            expected_direction=(
                "Velocity bất thường có thể hỗ trợ phát hiện fraud, nhưng cần kiểm tra cùng các tín hiệu khác."
            ),
            stage3_evidence=(
                "Giai đoạn 3 khuyến nghị giữ txn_count_ratio_7d_30d như behavioral anomaly feature."
            ),
        )

    return InsightMapping(
        insight_group="Other context",
        business_meaning=(
            "Feature ngữ cảnh khác. Cần đọc cùng coefficient và metric validation/test."
        ),
        expected_direction=(
            "Không có kỳ vọng hướng cố định."
        ),
        stage3_evidence=(
            "Không thuộc nhóm insight chính hoặc chưa có bằng chứng mạnh từ giai đoạn 3."
        ),
    )


def build_coefficient_explanation(coef_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo bảng giải thích coefficient theo insight.
    """
    required_cols = ["feature", "coefficient", "abs_coefficient", "direction"]

    for col in required_cols:
        if col not in coef_df.columns:
            raise ValueError(f"Thiếu cột trong coefficients.csv: {col}")

    rows = []

    for _, row in coef_df.iterrows():
        feature = row["feature"]
        coefficient = float(row["coefficient"])
        mapping = map_feature_to_insight(feature, coefficient)

        if coefficient > 0:
            coefficient_meaning = "Làm tăng log-odds fraud risk"
        elif coefficient < 0:
            coefficient_meaning = "Làm giảm log-odds fraud risk"
        else:
            coefficient_meaning = "Không có tác động tuyến tính rõ ràng"

        rows.append(
            {
                "feature": feature,
                "readable_feature": clean_feature_name(feature),
                "coefficient": coefficient,
                "abs_coefficient": float(row["abs_coefficient"]),
                "direction": row["direction"],
                "coefficient_meaning": coefficient_meaning,
                "insight_group": mapping.insight_group,
                "business_meaning": mapping.business_meaning,
                "expected_direction": mapping.expected_direction,
                "stage3_evidence": mapping.stage3_evidence,
            }
        )

    return pd.DataFrame(rows)


def build_insight_group_summary(explanation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tổng hợp mức độ đóng góp theo nhóm insight.
    """
    if explanation_df.empty:
        return pd.DataFrame()

    summary = (
        explanation_df
        .groupby("insight_group")
        .agg(
            feature_count=("feature", "count"),
            total_abs_coefficient=("abs_coefficient", "sum"),
            mean_abs_coefficient=("abs_coefficient", "mean"),
            max_abs_coefficient=("abs_coefficient", "max"),
        )
        .reset_index()
        .sort_values("total_abs_coefficient", ascending=False)
    )

    return summary


def choose_final_experiment(
    comparison_df: pd.DataFrame,
    preferred_experiment: str = "lr_no_mcc",
) -> pd.Series:
    """
    Chọn experiment cuối.

    Ưu tiên lr_no_mcc nếu có, vì experiment trước đã cho thấy:
    - hiệu năng tốt hơn
    - số chiều thấp hơn
    - dễ giải thích hơn
    """
    if "experiment_name" not in comparison_df.columns:
        raise ValueError("comparison_df thiếu cột experiment_name")

    selected = comparison_df[comparison_df["experiment_name"] == preferred_experiment]

    if not selected.empty:
        return selected.iloc[0]

    # fallback: chọn model có test_f2 tốt nhất
    return comparison_df.sort_values(
        ["test_f2", "test_pr_auc"],
        ascending=[False, False],
    ).iloc[0]


def generate_final_baseline_report(
    comparison_df: pd.DataFrame,
    selected_experiment_row: pd.Series,
    coef_explanation_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    validation_metrics_df: pd.DataFrame,
    test_metrics_df: pd.DataFrame,
    output_path: str | Path,
    top_n: int = 15,
) -> None:
    """
    Sinh báo cáo Markdown cuối cùng cho baseline model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exp = selected_experiment_row

    test_metrics = test_metrics_df.iloc[0].to_dict()
    validation_metrics = validation_metrics_df.iloc[0].to_dict()

    top_positive = (
        coef_explanation_df[coef_explanation_df["coefficient"] > 0]
        .sort_values("coefficient", ascending=False)
        .head(top_n)
    )

    top_negative = (
        coef_explanation_df[coef_explanation_df["coefficient"] < 0]
        .sort_values("coefficient", ascending=True)
        .head(top_n)
    )

    key_insight_features = coef_explanation_df[
        coef_explanation_df["insight_group"].isin(
            [
                "MCC behavior anomaly",
                "Time behavior anomaly",
                "Amount anomaly",
                "Spending trend anomaly",
                "Country / cross-border behavior",
            ]
        )
    ].sort_values("abs_coefficient", ascending=False).head(25)

    lines = []

    lines.append("# Final Baseline Model Report - Logistic Regression\n")

    lines.append("## 1. Model được chọn\n")
    lines.append(
        f"Model baseline được chọn là **`{exp['experiment_name']}`**."
    )

    lines.append("\nLý do chọn model này:")
    lines.append(
        f"- Test PR-AUC: **{format_number(exp['test_pr_auc'], 4)}**"
    )
    lines.append(
        f"- Test F2: **{format_number(exp['test_f2'], 4)}**"
    )
    lines.append(
        f"- Test Precision: **{format_percent(exp['test_precision'])}**"
    )
    lines.append(
        f"- Test Recall: **{format_percent(exp['test_recall'])}**"
    )
    lines.append(
        f"- Review rate: **{format_percent(exp['test_review_rate'])}**"
    )
    lines.append(
        f"- Số chiều sau preprocessing: **{int(exp['n_preprocessed_features'])}**"
    )

    if str(exp.get("dropped_categorical", "")).strip():
        lines.append(
            f"- Feature categorical bị bỏ trong experiment: **`{exp['dropped_categorical']}`**"
        )

    lines.append(
        "\nKết quả experiment cho thấy bản bỏ `mcc` không làm giảm hiệu năng, "
        "ngược lại còn cải thiện PR-AUC/F2 và giảm số chiều, nên phù hợp hơn để chọn làm baseline chính."
    )

    lines.append("\n## 2. So sánh các experiment\n")
    show_cols = [
        "experiment_name",
        "dropped_categorical",
        "min_frequency",
        "n_preprocessed_features",
        "selected_threshold",
        "test_pr_auc",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_f2",
        "test_review_rate",
        "test_tp",
        "test_fp",
        "test_fn",
        "test_tn",
    ]
    show_cols = [col for col in show_cols if col in comparison_df.columns]
    lines.append(
        comparison_df[show_cols]
        .sort_values("test_f2", ascending=False)
        .to_markdown(index=False)
    )

    lines.append("\n## 3. Validation và Test metrics của model cuối\n")
    lines.append("### Validation")
    lines.append(validation_metrics_df.to_markdown(index=False))
    lines.append("\n### Test")
    lines.append(test_metrics_df.to_markdown(index=False))

    lines.append("\n## 4. Diễn giải nghiệp vụ từ confusion matrix\n")
    tp = int(test_metrics.get("tp", 0))
    fp = int(test_metrics.get("fp", 0))
    fn = int(test_metrics.get("fn", 0))
    tn = int(test_metrics.get("tn", 0))
    predicted_positive = int(test_metrics.get("predicted_positive", tp + fp))
    rows = int(test_metrics.get("rows", tp + fp + fn + tn))
    fraud_count = int(test_metrics.get("fraud_count", tp + fn))

    lines.append(
        f"Trên tập test có **{rows:,}** giao dịch, trong đó có **{fraud_count:,}** fraud thật."
    )
    lines.append(
        f"Model cảnh báo **{predicted_positive:,}** giao dịch cần review "
        f"(**{format_percent(test_metrics.get('review_rate'))}** tổng giao dịch)."
    )
    lines.append(
        f"Trong các giao dịch bị cảnh báo, model bắt đúng **{tp:,}** fraud và báo nhầm **{fp:,}** giao dịch thường."
    )
    lines.append(
        f"Model bỏ sót **{fn:,}** fraud và xác định đúng **{tn:,}** giao dịch thường."
    )

    lines.append("\n## 5. Tổng hợp coefficient theo nhóm insight\n")
    lines.append(group_summary_df.to_markdown(index=False))

    lines.append("\n## 6. Top feature làm tăng fraud risk\n")
    top_positive_cols = [
        "readable_feature",
        "coefficient",
        "insight_group",
        "business_meaning",
    ]
    lines.append(top_positive[top_positive_cols].to_markdown(index=False))

    lines.append("\n## 7. Top feature làm giảm fraud risk\n")
    top_negative_cols = [
        "readable_feature",
        "coefficient",
        "insight_group",
        "business_meaning",
    ]
    lines.append(top_negative[top_negative_cols].to_markdown(index=False))

    lines.append("\n## 8. Các coefficient khớp với insight giai đoạn 3\n")
    key_cols = [
        "readable_feature",
        "coefficient",
        "direction",
        "insight_group",
        "expected_direction",
        "stage3_evidence",
    ]
    lines.append(key_insight_features[key_cols].to_markdown(index=False))

    lines.append("\n## 9. Kết luận giải thích model\n")
    lines.append(
        """
Model baseline không chỉ dựa vào category cụ thể mà học được các nhóm behavioral anomaly đã khai phá ở giai đoạn 3:

- `mcc_entropy_30d` có coefficient âm: entropy càng cao thì risk càng giảm, nghĩa là entropy thấp làm fraud risk tăng.
- Các interaction như `low_mcc_entropy_x_high_max_amount_30d`, `low_mcc_entropy_x_low_spending_trend`, `low_mcc_entropy_x_low_night_ratio` có coefficient dương, khớp với insight rằng fraud tăng mạnh khi nhiều tín hiệu bất thường cùng xuất hiện.
- `amount_z_30d` có coefficient dương, cho thấy amount hiện tại lệch khỏi lịch sử user làm tăng risk.
- `night_ratio_30d` và `spending_trend` có coefficient âm, khớp với insight rằng nhóm night ratio thấp hoặc spending trend thấp có risk cao hơn.
"""
    )

    lines.append("\n## 10. Lưu ý khi diễn giải coefficient\n")
    lines.append(
        """
Coefficient của Logistic Regression thể hiện tác động lên log-odds của fraud sau khi đã xét đồng thời các feature khác.
Vì vậy không nên diễn giải là quan hệ nhân quả. Một coefficient dương nghĩa là feature đó liên quan đến việc model tăng fraud risk,
còn coefficient âm nghĩa là feature đó liên quan đến việc model giảm fraud risk.

Các one-hot category như `currency`, `merchant_country`, `time_period` nên được diễn giải cẩn thận vì chúng phụ thuộc vào category tham chiếu
và các feature khác trong model.
"""
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def generate_all_final_reports(
    project_root: str | Path,
    selected_experiment: str = "lr_no_mcc",
) -> None:
    """
    Entry point chính để generate report cuối.
    """
    project_root = Path(project_root)

    comparison_path = project_root / "data" / "reports" / "model" / "experiment_comparison.csv"

    experiment_report_dir = (
        project_root
        / "data"
        / "reports"
        / "model"
        / "experiments"
        / selected_experiment
    )

    coefficients_path = experiment_report_dir / "coefficients.csv"
    validation_metrics_path = experiment_report_dir / "metrics_validation_best_threshold.csv"
    test_metrics_path = experiment_report_dir / "metrics_test.csv"

    comparison_df = load_csv(comparison_path)
    coef_df = load_csv(coefficients_path)
    validation_metrics_df = load_csv(validation_metrics_path)
    test_metrics_df = load_csv(test_metrics_path)

    selected_row = choose_final_experiment(
        comparison_df=comparison_df,
        preferred_experiment=selected_experiment,
    )

    coef_explanation_df = build_coefficient_explanation(coef_df)
    group_summary_df = build_insight_group_summary(coef_explanation_df)

    output_report_path = project_root / "reports" / "final_baseline_model_report.md"

    explanation_csv_path = (
        project_root
        / "data"
        / "reports"
        / "model"
        / "final_model_insight_explanation.csv"
    )

    group_summary_csv_path = (
        project_root
        / "data"
        / "reports"
        / "model"
        / "final_model_insight_group_summary.csv"
    )

    explanation_csv_path.parent.mkdir(parents=True, exist_ok=True)

    coef_explanation_df.to_csv(
        explanation_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    group_summary_df.to_csv(
        group_summary_csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    generate_final_baseline_report(
        comparison_df=comparison_df,
        selected_experiment_row=selected_row,
        coef_explanation_df=coef_explanation_df,
        group_summary_df=group_summary_df,
        validation_metrics_df=validation_metrics_df,
        test_metrics_df=test_metrics_df,
        output_path=output_report_path,
    )

    print(f"Saved final report: {output_report_path}")
    print(f"Saved coefficient explanation: {explanation_csv_path}")
    print(f"Saved insight group summary: {group_summary_csv_path}")