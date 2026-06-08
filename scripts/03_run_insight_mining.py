from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.clean_data import load_cleaned_data
from src.data.split_data import split_fraud_data, save_data_splits

from src.features.behavior_features import (
    fit_behavior_thresholds,
    thresholds_to_dataframe,
    create_behavior_features,
)
from src.features.interaction_features import create_interaction_features

from src.mining.fraud_rate_analysis import (
    calculate_global_fraud_rate,
    analyze_many_categorical_features,
    detect_strong_categorical_patterns,
    detect_strong_numeric_patterns,
    resolve_min_count,
)

from src.mining.segment_analysis import analyze_business_segments

from src.mining.interaction_mining import (
    mine_boolean_interactions,
    mine_categorical_pair_interactions,
    analyze_priority_business_interactions,
    summarize_interactions_for_feature_engineering,
)

from src.mining.rule_mining import (
    add_rule_interpretation,
    select_actionable_rules,
    generate_insight_markdown_report,
)

from src.mining.stability_check import (
    create_numeric_bin_edges_from_train,
    analyze_numeric_features_with_train_bins,
    compare_pattern_stability,
    filter_stable_patterns,
    generate_stability_markdown_report,
)
from src.mining.visualization import (
    generate_data_mining_visualizations,
    write_visualization_gallery,
)


TARGET_COL = "fraud"

CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "interim" / "fraud_cleaned.parquet"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INSIGHT_REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "insights"

MARKDOWN_REPORT_PATH = PROJECT_ROOT / "reports" / "insight_report.md"
STABILITY_MARKDOWN_REPORT_PATH = PROJECT_ROOT / "reports" / "insight_stability_report.md"
MINING_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures" / "data_mining"
MINING_VISUALIZATION_REPORT_PATH = PROJECT_ROOT / "reports" / "data_mining_visualizations.md"


NUMERIC_FEATURES = [
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
    "spending_trend",
    "spending_trend_30d",
    "ip_score",
    "txn_count_7d",
    "txn_count_30d",
    "txn_count_ratio_7d_30d",
    "hour",
    "amount_to_mean_30d",
    "amount_to_max_30d",
    "night_unusual_score",
]


CATEGORICAL_FEATURES = [
    "currency",
    "payment_channel",
    "merchant_country",
    "mcc",
    "card_present",
    "card_entry_mode",
    "auth_result",
    "pin_verif_method",
    "tokenised",
    "recurring_flag",
    "cross_border",
    "auth_characteristics",
    "message_type",
    "term_location",
    "day_name",
    "time_period",
    "ip_score_bin",
    "ip_score_risk_group",
    "decline_rate_risk_group",
    "credit_util_risk_group",
    "txn_count_7d_group",
    "txn_velocity_group",
    "night_ratio_group",
    "amount_to_mean_group",
    "amount_to_max_group",
    "night_unusual_group",
    "high_ip_high_amount_flag",
    "high_decline_high_amount_flag",
    "private_ip_high_score_flag",
    "high_velocity_high_amount_flag",
    "low_mcc_entropy_x_low_night_ratio",
    "low_mcc_entropy_x_high_max_amount_30d",
    "low_mcc_entropy_x_low_spending_trend",
    "low_mcc_entropy_x_card_not_present",
    "low_night_ratio_x_high_max_amount_30d",
    "cross_border_x_low_country_diversity",
    "high_ip_score_x_card_not_present",
]


CATEGORICAL_PAIRS = [
    ("payment_channel", "merchant_country"),
    ("payment_channel", "card_entry_mode"),
    ("payment_channel", "term_location"),
    ("card_entry_mode", "auth_result"),
    ("auth_result", "pin_verif_method"),
    ("cross_border", "merchant_country"),
    ("card_present", "payment_channel"),
    ("tokenised", "payment_channel"),
    ("time_period", "payment_channel"),
    ("day_name", "payment_channel"),
]


def save_report(df: pd.DataFrame, path: Path, preview_rows: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if df is None:
        df = pd.DataFrame()

    output_df = df.copy()

    if preview_rows is not None:
        output_df = output_df.head(preview_rows)

    output_df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Saved: {path}")


def add_interpretation_for_segment(segment_report: pd.DataFrame) -> pd.DataFrame:
    if segment_report is None or segment_report.empty:
        return pd.DataFrame() if segment_report is None else segment_report

    tmp = segment_report.rename(columns={"segment_name": "rule"})
    tmp = add_rule_interpretation(tmp, rule_col="rule")
    tmp = tmp.rename(columns={"rule": "segment_name"})

    return tmp


def prepare_features_with_train_thresholds(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fit thresholds trên train, apply sang train và validation.
    """
    thresholds = fit_behavior_thresholds(train_df)

    train_features = create_behavior_features(train_df, thresholds=thresholds)
    valid_features = create_behavior_features(valid_df, thresholds=thresholds)

    train_features = create_interaction_features(train_features)
    valid_features = create_interaction_features(valid_features)

    threshold_df = thresholds_to_dataframe(thresholds)

    return train_features, valid_features, threshold_df


def main():
    print("=" * 80)
    print("GIAI ĐOẠN 3: TRAIN-ONLY INSIGHT MINING + VALIDATION STABILITY CHECK")
    print("=" * 80)

    print("\n[1] Loading cleaned data...")
    df = load_cleaned_data(CLEANED_DATA_PATH)
    print(f"Loaded cleaned data: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Không tìm thấy target column: {TARGET_COL}")

    INSIGHT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[2] Splitting data...")
    train_df, valid_df, test_df, split_summary = split_fraud_data(
        df=df,
        target_col=TARGET_COL,
        timestamp_col="transaction_datetime",
        fallback_col="local_timestamp",
        train_size=0.70,
        valid_size=0.15,
        test_size=0.15,
        min_fraud_per_split=50,
        random_state=42,
        strategy="time",
    )

    print(split_summary)

    save_data_splits(
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        output_dir=PROCESSED_DIR,
    )

    save_report(
        split_summary,
        INSIGHT_REPORT_DIR / "split_summary.csv",
    )

    print("\n[3] Fit behavior thresholds on TRAIN only, then apply to validation...")
    train_features, valid_features, threshold_df = prepare_features_with_train_thresholds(
        train_df=train_df,
        valid_df=valid_df,
    )

    save_report(
        threshold_df,
        INSIGHT_REPORT_DIR / "train_behavior_thresholds.csv",
    )

    save_report(
        train_features.head(1000),
        INSIGHT_REPORT_DIR / "train_features_preview.csv",
    )

    save_report(
        valid_features.head(1000),
        INSIGHT_REPORT_DIR / "validation_features_preview.csv",
    )

    print("\n[4] Global fraud rate on train / validation...")
    train_global_rate = calculate_global_fraud_rate(train_features, TARGET_COL)
    valid_global_rate = calculate_global_fraud_rate(valid_features, TARGET_COL)

    print(f"Train fraud rate: {train_global_rate:.4f} = {train_global_rate * 100:.2f}%")
    print(f"Valid fraud rate: {valid_global_rate:.4f} = {valid_global_rate * 100:.2f}%")

    used_min_count_train = resolve_min_count(
        df=train_features,
        min_count=None,
        min_count_ratio=0.005,
        default_min_count=100,
    )

    used_min_count_valid = resolve_min_count(
        df=valid_features,
        min_count=None,
        min_count_ratio=0.005,
        default_min_count=50,
    )

    print(f"Train min_count: {used_min_count_train}")
    print(f"Valid min_count: {used_min_count_valid}")

    pd.DataFrame(
        [
            {
                "split": "train",
                "fraud_rate": train_global_rate,
                "used_min_count": used_min_count_train,
            },
            {
                "split": "validation",
                "fraud_rate": valid_global_rate,
                "used_min_count": used_min_count_valid,
            },
        ]
    ).to_csv(INSIGHT_REPORT_DIR / "train_valid_global_fraud_rate.csv", index=False)

    print("\n[5] Numeric fraud-rate reports with TRAIN bin edges...")
    existing_numeric_features = [
        col for col in NUMERIC_FEATURES
        if col in train_features.columns
    ]

    numeric_bin_edges = create_numeric_bin_edges_from_train(
        train_df=train_features,
        features=existing_numeric_features,
        bins=10,
    )

    numeric_train_report = analyze_numeric_features_with_train_bins(
        df=train_features,
        bin_edges=numeric_bin_edges,
        target_col=TARGET_COL,
        min_count=None,
        min_count_ratio=0.005,
        default_min_count=100,
    )

    numeric_valid_report = analyze_numeric_features_with_train_bins(
        df=valid_features,
        bin_edges=numeric_bin_edges,
        target_col=TARGET_COL,
        min_count=None,
        min_count_ratio=0.005,
        default_min_count=50,
    )

    save_report(
        numeric_train_report,
        INSIGHT_REPORT_DIR / "numeric_bin_fraud_rate_train.csv",
    )

    save_report(
        numeric_valid_report,
        INSIGHT_REPORT_DIR / "numeric_bin_fraud_rate_validation.csv",
    )

    strong_numeric_train = detect_strong_numeric_patterns(
        numeric_bin_report=numeric_train_report,
        min_lift=1.5,
        min_count=used_min_count_train,
    )

    save_report(
        strong_numeric_train,
        INSIGHT_REPORT_DIR / "strong_numeric_patterns_train.csv",
    )

    numeric_stability = compare_pattern_stability(
        train_report=numeric_train_report,
        valid_report=numeric_valid_report,
        key_cols=["feature", "bin"],
    )

    stable_numeric = filter_stable_patterns(
        numeric_stability,
        min_lift_train=1.5,
        min_lift_valid=1.3,
        min_support_train=used_min_count_train,
        min_support_valid=used_min_count_valid,
        max_lift_ratio_gap=0.75,
    )

    save_report(
        numeric_stability,
        INSIGHT_REPORT_DIR / "numeric_pattern_stability.csv",
    )

    save_report(
        stable_numeric,
        INSIGHT_REPORT_DIR / "stable_numeric_patterns.csv",
    )

    print("\n[6] Categorical fraud-rate reports...")
    existing_categorical_features = [
        col for col in CATEGORICAL_FEATURES
        if col in train_features.columns
    ]

    categorical_train_report = analyze_many_categorical_features(
        df=train_features,
        features=existing_categorical_features,
        target_col=TARGET_COL,
        min_count=None,
        min_count_ratio=0.005,
        default_min_count=100,
    )

    categorical_valid_report = analyze_many_categorical_features(
        df=valid_features,
        features=existing_categorical_features,
        target_col=TARGET_COL,
        min_count=None,
        min_count_ratio=0.005,
        default_min_count=50,
    )

    save_report(
        categorical_train_report,
        INSIGHT_REPORT_DIR / "categorical_fraud_rate_train.csv",
    )

    save_report(
        categorical_valid_report,
        INSIGHT_REPORT_DIR / "categorical_fraud_rate_validation.csv",
    )

    strong_categorical_train = detect_strong_categorical_patterns(
        categorical_report=categorical_train_report,
        min_lift=1.3,
        min_count=used_min_count_train,
    )

    save_report(
        strong_categorical_train,
        INSIGHT_REPORT_DIR / "strong_categorical_patterns_train.csv",
    )

    categorical_stability = compare_pattern_stability(
        train_report=categorical_train_report,
        valid_report=categorical_valid_report,
        key_cols=["feature", "value"],
    )

    stable_categorical = filter_stable_patterns(
        categorical_stability,
        min_lift_train=1.3,
        min_lift_valid=1.2,
        min_support_train=used_min_count_train,
        min_support_valid=used_min_count_valid,
        max_lift_ratio_gap=0.75,
    )

    save_report(
        categorical_stability,
        INSIGHT_REPORT_DIR / "categorical_pattern_stability.csv",
    )

    save_report(
        stable_categorical,
        INSIGHT_REPORT_DIR / "stable_categorical_patterns.csv",
    )

    print("\n[7] Business segment analysis...")
    segment_train_report = analyze_business_segments(
        df=train_features,
        target_col=TARGET_COL,
    )

    segment_valid_report = analyze_business_segments(
        df=valid_features,
        target_col=TARGET_COL,
    )

    segment_train_report = add_interpretation_for_segment(segment_train_report)
    segment_valid_report = add_interpretation_for_segment(segment_valid_report)

    save_report(
        segment_train_report,
        INSIGHT_REPORT_DIR / "business_segment_report_train.csv",
    )

    save_report(
        segment_valid_report,
        INSIGHT_REPORT_DIR / "business_segment_report_validation.csv",
    )

    segment_stability = compare_pattern_stability(
        train_report=segment_train_report,
        valid_report=segment_valid_report,
        key_cols=["segment_name"],
    )

    stable_segments = filter_stable_patterns(
        segment_stability,
        min_lift_train=1.3,
        min_lift_valid=1.2,
        min_support_train=used_min_count_train,
        min_support_valid=used_min_count_valid,
        max_lift_ratio_gap=0.75,
    )

    save_report(
        segment_stability,
        INSIGHT_REPORT_DIR / "segment_stability.csv",
    )

    save_report(
        stable_segments,
        INSIGHT_REPORT_DIR / "stable_segments.csv",
    )

    print("\n[8] Boolean interaction mining...")
    flag_cols = [
        col for col in train_features.columns
        if col.endswith("_flag") or "_x_" in col
    ]

    boolean_train = mine_boolean_interactions(
        df=train_features,
        flag_cols=flag_cols,
        target_col=TARGET_COL,
        min_support=used_min_count_train,
        max_order=3,
    )

    boolean_valid = mine_boolean_interactions(
        df=valid_features,
        flag_cols=flag_cols,
        target_col=TARGET_COL,
        min_support=used_min_count_valid,
        max_order=3,
    )

    boolean_train = add_rule_interpretation(boolean_train, rule_col="rule")
    boolean_valid = add_rule_interpretation(boolean_valid, rule_col="rule")

    save_report(
        boolean_train,
        INSIGHT_REPORT_DIR / "boolean_interactions_train.csv",
    )

    save_report(
        boolean_valid,
        INSIGHT_REPORT_DIR / "boolean_interactions_validation.csv",
    )

    boolean_stability = compare_pattern_stability(
        train_report=boolean_train,
        valid_report=boolean_valid,
        key_cols=["rule"],
    )

    stable_boolean = filter_stable_patterns(
        boolean_stability,
        min_lift_train=1.5,
        min_lift_valid=1.3,
        min_support_train=used_min_count_train,
        min_support_valid=used_min_count_valid,
        max_lift_ratio_gap=0.75,
    )

    save_report(
        boolean_stability,
        INSIGHT_REPORT_DIR / "boolean_interaction_stability.csv",
    )

    save_report(
        stable_boolean,
        INSIGHT_REPORT_DIR / "stable_boolean_interactions.csv",
    )

    print("\n[9] Priority business interactions...")
    priority_train = analyze_priority_business_interactions(
        df=train_features,
        target_col=TARGET_COL,
        min_support=30,
    )

    priority_valid = analyze_priority_business_interactions(
        df=valid_features,
        target_col=TARGET_COL,
        min_support=30,
    )

    priority_train = add_rule_interpretation(priority_train, rule_col="rule")
    priority_valid = add_rule_interpretation(priority_valid, rule_col="rule")

    save_report(
        priority_train,
        INSIGHT_REPORT_DIR / "priority_business_interactions_train.csv",
    )

    save_report(
        priority_valid,
        INSIGHT_REPORT_DIR / "priority_business_interactions_validation.csv",
    )

    priority_stability = compare_pattern_stability(
        train_report=priority_train,
        valid_report=priority_valid,
        key_cols=["rule"],
    )

    stable_priority = filter_stable_patterns(
        priority_stability,
        min_lift_train=1.5,
        min_lift_valid=1.3,
        min_support_train=50,
        min_support_valid=30,
        max_lift_ratio_gap=0.75,
    )

    save_report(
        priority_stability,
        INSIGHT_REPORT_DIR / "priority_interaction_stability.csv",
    )

    save_report(
        stable_priority,
        INSIGHT_REPORT_DIR / "stable_priority_interactions.csv",
    )

    print("\n[10] Categorical pair interactions...")
    categorical_pair_train = mine_categorical_pair_interactions(
        df=train_features,
        categorical_pairs=CATEGORICAL_PAIRS,
        target_col=TARGET_COL,
        min_support=used_min_count_train,
    )

    categorical_pair_valid = mine_categorical_pair_interactions(
        df=valid_features,
        categorical_pairs=CATEGORICAL_PAIRS,
        target_col=TARGET_COL,
        min_support=used_min_count_valid,
    )

    categorical_pair_train = add_rule_interpretation(
        categorical_pair_train,
        rule_col="rule",
    )

    categorical_pair_valid = add_rule_interpretation(
        categorical_pair_valid,
        rule_col="rule",
    )

    save_report(
        categorical_pair_train,
        INSIGHT_REPORT_DIR / "categorical_pair_interactions_train.csv",
    )

    save_report(
        categorical_pair_valid,
        INSIGHT_REPORT_DIR / "categorical_pair_interactions_validation.csv",
    )

    categorical_pair_stability = compare_pattern_stability(
        train_report=categorical_pair_train,
        valid_report=categorical_pair_valid,
        key_cols=["feature_a", "value_a", "feature_b", "value_b"],
    )

    stable_categorical_pairs = filter_stable_patterns(
        categorical_pair_stability,
        min_lift_train=1.3,
        min_lift_valid=1.2,
        min_support_train=used_min_count_train,
        min_support_valid=used_min_count_valid,
        max_lift_ratio_gap=0.75,
    )

    save_report(
        categorical_pair_stability,
        INSIGHT_REPORT_DIR / "categorical_pair_stability.csv",
    )

    save_report(
        stable_categorical_pairs,
        INSIGHT_REPORT_DIR / "stable_categorical_pair_interactions.csv",
    )

    print("\n[11] Selecting actionable rules from TRAIN only...")
    all_train_rule_reports = []

    for report in [
        boolean_train,
        priority_train,
        categorical_pair_train,
    ]:
        if report is not None and not report.empty:
            all_train_rule_reports.append(report)

    if all_train_rule_reports:
        all_train_rules = pd.concat(all_train_rule_reports, ignore_index=True)
    else:
        all_train_rules = pd.DataFrame()

    actionable_rules_train = select_actionable_rules(
        all_train_rules,
        min_lift=2.0,
        min_support=used_min_count_train,
        min_fraud_rate=train_global_rate * 2,
    )

    save_report(
        actionable_rules_train,
        INSIGHT_REPORT_DIR / "actionable_rules_train.csv",
    )

    print("\n[12] Feature engineering recommendations from stable priority interactions...")
    feature_recommendations = summarize_interactions_for_feature_engineering(
        priority_train,
        min_lift=2.0,
        min_support=50,
    )

    save_report(
        feature_recommendations,
        INSIGHT_REPORT_DIR / "feature_recommendations_train.csv",
    )

    print("\n[13] Generating markdown reports...")

    generate_insight_markdown_report(
        output_path=MARKDOWN_REPORT_PATH,
        global_fraud_rate=train_global_rate,
        segment_report=segment_train_report,
        priority_interactions=priority_train,
        actionable_rules=actionable_rules_train,
        numeric_patterns=strong_numeric_train,
        categorical_patterns=strong_categorical_train,
    )

    generate_stability_markdown_report(
        output_path=STABILITY_MARKDOWN_REPORT_PATH,
        split_summary=split_summary,
        stable_numeric=stable_numeric,
        stable_categorical=stable_categorical,
        stable_priority_interactions=stable_priority,
        stable_boolean_interactions=stable_boolean,
        stable_categorical_pairs=stable_categorical_pairs,
    )

    print(f"Saved insight report: {MARKDOWN_REPORT_PATH}")
    print(f"Saved stability report: {STABILITY_MARKDOWN_REPORT_PATH}")

    print("\n[14] Generating data mining visualizations...")
    image_paths = generate_data_mining_visualizations(
        insight_dir=INSIGHT_REPORT_DIR,
        figures_dir=MINING_FIGURES_DIR,
    )
    write_visualization_gallery(
        image_paths=image_paths,
        output_path=MINING_VISUALIZATION_REPORT_PATH,
        project_root=PROJECT_ROOT,
    )
    print(f"Saved {len(image_paths)} mining figures to: {MINING_FIGURES_DIR}")
    print(f"Saved visualization gallery: {MINING_VISUALIZATION_REPORT_PATH}")

    print("\nDone!")
    print("=" * 80)


if __name__ == "__main__":
    main()
