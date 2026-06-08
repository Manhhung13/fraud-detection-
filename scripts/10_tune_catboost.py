from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.training.model_experiment_runner import (
    ModelExperimentConfig,
    generate_model_comparison_report,
    load_feature_metadata,
    run_model_experiment,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FEATURES_PATH = PROCESSED_DIR / "model_train_features.parquet"
VALIDATION_FEATURES_PATH = PROCESSED_DIR / "model_validation_features.parquet"
TEST_FEATURES_PATH = PROCESSED_DIR / "model_test_features.parquet"
FEATURE_METADATA_PATH = PROCESSED_DIR / "feature_metadata.json"

MODEL_OUTPUT_ROOT = PROJECT_ROOT / "models" / "catboost_tuning_experiments"
REPORT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "reports" / "catboost_tuning"

COMPARISON_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "catboost_tuning"
    / "catboost_tuning_comparison.csv"
)

COMPARISON_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "catboost_tuning_comparison.md"
)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {path}\n"
            f"Hãy chạy trước: python scripts/04_build_features.py"
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


def build_catboost_tuning_configs() -> list[ModelExperimentConfig]:
    """
    Tuning nhẹ CatBoost final model.

    Base model tốt nhất hiện tại:
    - mcc min_frequency=100
    - no manual interactions
    - iterations=700
    - learning_rate=0.03
    - depth=6
    - l2_leaf_reg=5
    """
    common = {
        "drop_categorical_features": [],
        "drop_interaction_features": True,
        "use_rare_category_grouping": True,
        "min_frequency": 100,
        "threshold_metric": "f2",
        "max_review_rate": 0.05,
    }

    return [
        # Base final model để so sánh lại trong tuning report
        ModelExperimentConfig(
            experiment_name="cat_tune_base_depth6_l2_5_lr003_iter700",
            model_name="catboost",
            description="Base final CatBoost: depth=6, l2=5, lr=0.03, iter=700.",
            **common,
            model_params={
                "iterations": 700,
                "learning_rate": 0.03,
                "depth": 6,
                "l2_leaf_reg": 5.0,
                "scale_pos_weight": "auto",
                "random_state": 42,
                "use_gpu": True,
                "gpu_devices": "0",
            },
        ),

        # Chống overfit hơn
        ModelExperimentConfig(
            experiment_name="cat_tune_depth5_l2_8_lr003_iter700",
            model_name="catboost",
            description="CatBoost chống overfit hơn: depth=5, l2=8, lr=0.03, iter=700.",
            **common,
            model_params={
                "iterations": 700,
                "learning_rate": 0.03,
                "depth": 5,
                "l2_leaf_reg": 8.0,
                "scale_pos_weight": "auto",
                "random_state": 42,
                "use_gpu": True,
                "gpu_devices": "0",
            },
        ),

        # Học chậm hơn, nhiều vòng hơn
        ModelExperimentConfig(
            experiment_name="cat_tune_depth6_l2_5_lr002_iter1000",
            model_name="catboost",
            description="CatBoost học chậm hơn: depth=6, l2=5, lr=0.02, iter=1000.",
            **common,
            model_params={
                "iterations": 1000,
                "learning_rate": 0.02,
                "depth": 6,
                "l2_leaf_reg": 5.0,
                "scale_pos_weight": "auto",
                "random_state": 42,
                "use_gpu": True,
                "gpu_devices": "0",
            },
        ),

        # Mạnh hơn một chút nhưng regularize cao hơn
        ModelExperimentConfig(
            experiment_name="cat_tune_depth7_l2_8_lr003_iter700",
            model_name="catboost",
            description="CatBoost mạnh hơn nhẹ: depth=7, l2=8, lr=0.03, iter=700.",
            **common,
            model_params={
                "iterations": 700,
                "learning_rate": 0.03,
                "depth": 7,
                "l2_leaf_reg": 8.0,
                "scale_pos_weight": "auto",
                "random_state": 42,
                "use_gpu": True,
                "gpu_devices": "0",
            },
        ),

        # Regularize mạnh hơn
        ModelExperimentConfig(
            experiment_name="cat_tune_depth6_l2_10_lr003_iter700",
            model_name="catboost",
            description="CatBoost regularize mạnh hơn: depth=6, l2=10, lr=0.03, iter=700.",
            **common,
            model_params={
                "iterations": 700,
                "learning_rate": 0.03,
                "depth": 6,
                "l2_leaf_reg": 10.0,
                "scale_pos_weight": "auto",
                "random_state": 42,
                "use_gpu": True,
                "gpu_devices": "0",
            },
        ),
    ]


def main():
    print("=" * 80)
    print("CATBOOST TUNING EXPERIMENTS")
    print("=" * 80)

    print("\n[1] Loading datasets...")
    train_df, validation_df, test_df = load_datasets()

    print(f"Train shape:      {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")
    print(f"Test shape:       {test_df.shape}")

    print("\n[2] Loading feature metadata...")
    metadata = load_feature_metadata(FEATURE_METADATA_PATH)

    configs = build_catboost_tuning_configs()

    summaries = []

    for config in configs:
        summary = run_model_experiment(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            base_metadata=metadata,
            config=config,
            model_output_root=MODEL_OUTPUT_ROOT,
            report_output_root=REPORT_OUTPUT_ROOT,
        )
        summaries.append(summary)

    comparison_df = pd.DataFrame(summaries)

    COMPARISON_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(COMPARISON_CSV_PATH, index=False, encoding="utf-8-sig")

    generate_model_comparison_report(
        comparison_df=comparison_df,
        output_path=COMPARISON_REPORT_PATH,
    )

    print(f"\nSaved tuning CSV: {COMPARISON_CSV_PATH}")
    print(f"Saved tuning report: {COMPARISON_REPORT_PATH}")

    show_cols = [
        "experiment_name",
        "selected_threshold",
        "n_preprocessed_features",
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
        "elapsed_seconds",
    ]

    show_cols = [col for col in show_cols if col in comparison_df.columns]

    print("\nTuning summary:")
    print(
        comparison_df[show_cols]
        .sort_values(["test_f2", "test_pr_auc"], ascending=False)
        .to_string(index=False)
    )

    print("\nDone!")
    print("=" * 80)


if __name__ == "__main__":
    main()