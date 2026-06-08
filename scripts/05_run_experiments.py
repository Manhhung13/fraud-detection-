from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.training.experiment_runner import (
    ExperimentConfig,
    generate_experiment_comparison_report,
    load_feature_metadata,
    run_single_experiment,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FEATURES_PATH = PROCESSED_DIR / "model_train_features.parquet"
VALIDATION_FEATURES_PATH = PROCESSED_DIR / "model_validation_features.parquet"
TEST_FEATURES_PATH = PROCESSED_DIR / "model_test_features.parquet"
FEATURE_METADATA_PATH = PROCESSED_DIR / "feature_metadata.json"

MODEL_EXPERIMENT_ROOT = PROJECT_ROOT / "models" / "experiments"
REPORT_EXPERIMENT_ROOT = PROJECT_ROOT / "data" / "reports" / "model" / "experiments"

COMPARISON_CSV_PATH = PROJECT_ROOT / "data" / "reports" / "model" / "experiment_comparison.csv"
COMPARISON_REPORT_PATH = PROJECT_ROOT / "reports" / "model_experiment_comparison.md"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {path}\n"
            f"Hãy chạy các bước trước:\n"
            f"1. python scripts/03_run_insight_mining.py\n"
            f"2. python scripts/04_build_features.py"
        )


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [
        TRAIN_FEATURES_PATH,
        VALIDATION_FEATURES_PATH,
        TEST_FEATURES_PATH,
        FEATURE_METADATA_PATH,
    ]:
        require_file(path)

    train_df = pd.read_parquet(TRAIN_FEATURES_PATH)
    validation_df = pd.read_parquet(VALIDATION_FEATURES_PATH)
    test_df = pd.read_parquet(TEST_FEATURES_PATH)

    return train_df, validation_df, test_df


def build_experiment_configs() -> list[ExperimentConfig]:
    """
    Các experiment chính để kiểm tra độ ổn định của mcc.

    max_review_rate=0.05:
    - ép các model so sánh trong cùng điều kiện review khoảng tối đa 5%.
    - phù hợp với report trước đó khi threshold tốt cũng cho review rate ~5%.
    """
    return [
        ExperimentConfig(
            experiment_name="lr_mcc_minfreq_50",
            description=(
                "Baseline hiện tại: giữ mcc, gom category hiếm với min_frequency=50."
            ),
            class_weight="balanced",
            C=1.0,
            use_rare_category_grouping=True,
            min_frequency=50,
            drop_categorical_features=[],
            threshold_metric="f2",
            max_review_rate=0.05,
        ),
        ExperimentConfig(
            experiment_name="lr_no_mcc",
            description=(
                "Bỏ mcc để kiểm tra model có phụ thuộc quá nhiều vào mã ngành hàng hay không."
            ),
            class_weight="balanced",
            C=1.0,
            use_rare_category_grouping=True,
            min_frequency=50,
            drop_categorical_features=["mcc"],
            threshold_metric="f2",
            max_review_rate=0.05,
        ),
        ExperimentConfig(
            experiment_name="lr_mcc_minfreq_100",
            description=(
                "Giữ mcc nhưng gom category hiếm mạnh hơn với min_frequency=100."
            ),
            class_weight="balanced",
            C=1.0,
            use_rare_category_grouping=True,
            min_frequency=100,
            drop_categorical_features=[],
            threshold_metric="f2",
            max_review_rate=0.05,
        ),
        ExperimentConfig(
            experiment_name="lr_mcc_minfreq_200",
            description=(
                "Giữ mcc nhưng gom category hiếm rất mạnh với min_frequency=200."
            ),
            class_weight="balanced",
            C=1.0,
            use_rare_category_grouping=True,
            min_frequency=200,
            drop_categorical_features=[],
            threshold_metric="f2",
            max_review_rate=0.05,
        ),
    ]


def main():
    print("=" * 80)
    print("GIAI ĐOẠN 5: LOGISTIC REGRESSION EXPERIMENTS")
    print("=" * 80)

    print("\n[1] Loading datasets...")
    train_df, validation_df, test_df = load_datasets()

    print(f"Train shape:      {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")
    print(f"Test shape:       {test_df.shape}")

    print("\n[2] Loading feature metadata...")
    metadata = load_feature_metadata(FEATURE_METADATA_PATH)

    print(f"Numeric features:      {len(metadata['numeric_features'])}")
    print(f"Categorical features:  {len(metadata['categorical_features'])}")
    print(f"Binary features:       {len(metadata['binary_features'])}")
    print(f"Interaction features:  {len(metadata['interaction_features'])}")

    print("\n[3] Building experiment configs...")
    configs = build_experiment_configs()

    summaries = []

    for config in configs:
        summary = run_single_experiment(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            base_metadata=metadata,
            config=config,
            model_output_root=MODEL_EXPERIMENT_ROOT,
            report_output_root=REPORT_EXPERIMENT_ROOT,
        )

        summaries.append(summary)

    print("\n[4] Saving comparison report...")
    comparison_df = pd.DataFrame(summaries)

    COMPARISON_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(
        COMPARISON_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    generate_experiment_comparison_report(
        comparison_df=comparison_df,
        output_path=COMPARISON_REPORT_PATH,
    )

    print(f"Saved comparison CSV: {COMPARISON_CSV_PATH}")
    print(f"Saved comparison report: {COMPARISON_REPORT_PATH}")

    print("\n[5] Experiment comparison:")
    show_cols = [
        "experiment_name",
        "min_frequency",
        "dropped_categorical",
        "selected_threshold",
        "n_preprocessed_features",
        "test_pr_auc",
        "test_precision",
        "test_recall",
        "test_f2",
        "test_review_rate",
    ]

    show_cols = [col for col in show_cols if col in comparison_df.columns]

    print(
        comparison_df[show_cols]
        .sort_values("test_f2", ascending=False)
        .to_string(index=False)
    )

    print("\nDone!")
    print("=" * 80)


if __name__ == "__main__":
    main()