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

MODEL_OUTPUT_ROOT = PROJECT_ROOT / "models" / "gpu_boosting_experiments"
REPORT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "reports" / "gpu_boosting_models"

COMPARISON_CSV_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "gpu_boosting_models"
    / "gpu_boosting_model_comparison.csv"
)

COMPARISON_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "gpu_boosting_model_comparison.md"
)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {path}\n"
            f"Hãy chạy trước:\n"
            f"python scripts/04_build_features.py"
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


def build_gpu_boosting_configs() -> list[ModelExperimentConfig]:
    """
    GPU boosting experiments.

    Dựa trên kết quả hiện tại:
    - CatBoost GPU mcc_minfreq_100 đang là model advanced tốt nhất.
    - Thêm experiment bỏ manual interaction features để kiểm tra
      CatBoost có tự học interaction từ feature gốc không.
    """
    common_threshold = {
        "threshold_metric": "f2",
        "max_review_rate": 0.05,
    }

    return [
        ModelExperimentConfig(
            experiment_name="lgbm_gpu_no_mcc",
            model_name="lightgbm",
            description="LightGBM GPU không dùng mcc.",
            drop_categorical_features=["mcc"],
            drop_interaction_features=False,
            use_rare_category_grouping=True,
            min_frequency=100,
            **common_threshold,
            model_params={
                "n_estimators": 700,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 50,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "scale_pos_weight": "auto",
                "random_state": 42,
                "n_jobs": -1,
                "use_gpu": True,
                "gpu_device_type": "gpu",
            },
        ),
        ModelExperimentConfig(
            experiment_name="lgbm_gpu_mcc_minfreq_100",
            model_name="lightgbm",
            description="LightGBM GPU dùng mcc với min_frequency=100.",
            drop_categorical_features=[],
            drop_interaction_features=False,
            use_rare_category_grouping=True,
            min_frequency=100,
            **common_threshold,
            model_params={
                "n_estimators": 700,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "max_depth": -1,
                "min_child_samples": 50,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "scale_pos_weight": "auto",
                "random_state": 42,
                "n_jobs": -1,
                "use_gpu": True,
                "gpu_device_type": "gpu",
            },
        ),
        ModelExperimentConfig(
            experiment_name="xgb_gpu_no_mcc",
            model_name="xgboost",
            description="XGBoost GPU không dùng mcc.",
            drop_categorical_features=["mcc"],
            drop_interaction_features=False,
            use_rare_category_grouping=True,
            min_frequency=100,
            **common_threshold,
            model_params={
                "n_estimators": 700,
                "learning_rate": 0.03,
                "max_depth": 5,
                "min_child_weight": 5.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "scale_pos_weight": "auto",
                "random_state": 42,
                "n_jobs": -1,
                "use_gpu": True,
            },
        ),
        ModelExperimentConfig(
            experiment_name="xgb_gpu_mcc_minfreq_100",
            model_name="xgboost",
            description="XGBoost GPU dùng mcc với min_frequency=100.",
            drop_categorical_features=[],
            drop_interaction_features=False,
            use_rare_category_grouping=True,
            min_frequency=100,
            **common_threshold,
            model_params={
                "n_estimators": 700,
                "learning_rate": 0.03,
                "max_depth": 5,
                "min_child_weight": 5.0,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "scale_pos_weight": "auto",
                "random_state": 42,
                "n_jobs": -1,
                "use_gpu": True,
            },
        ),
        ModelExperimentConfig(
            experiment_name="cat_gpu_no_mcc",
            model_name="catboost",
            description="CatBoost GPU không dùng mcc.",
            drop_categorical_features=["mcc"],
            drop_interaction_features=False,
            use_rare_category_grouping=True,
            min_frequency=100,
            **common_threshold,
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
        ModelExperimentConfig(
            experiment_name="cat_gpu_mcc_minfreq_100",
            model_name="catboost",
            description="CatBoost GPU dùng mcc với min_frequency=100.",
            drop_categorical_features=[],
            drop_interaction_features=False,
            use_rare_category_grouping=True,
            min_frequency=100,
            **common_threshold,
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

        # Experiment mới: CatBoost giữ MCC nhưng bỏ manual interaction features.
        ModelExperimentConfig(
            experiment_name="cat_gpu_mcc_minfreq_100_no_interactions",
            model_name="catboost",
            description=(
                "CatBoost GPU dùng mcc với min_frequency=100 nhưng bỏ toàn bộ "
                "manual interaction features để kiểm tra CatBoost có tự học interaction "
                "từ các feature gốc hay không."
            ),
            drop_categorical_features=[],
            drop_interaction_features=True,
            use_rare_category_grouping=True,
            min_frequency=100,
            **common_threshold,
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
    ]


def main():
    print("=" * 80)
    print("TRAIN GPU BOOSTING MODELS")
    print("=" * 80)

    print("\n[1] Loading datasets...")
    train_df, validation_df, test_df = load_datasets()

    print(f"Train shape:      {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")
    print(f"Test shape:       {test_df.shape}")

    print("\n[2] Loading metadata...")
    metadata = load_feature_metadata(FEATURE_METADATA_PATH)

    print(f"Numeric features:      {len(metadata['numeric_features'])}")
    print(f"Categorical features:  {len(metadata['categorical_features'])}")
    print(f"Binary features:       {len(metadata['binary_features'])}")
    print(f"Interaction features:  {len(metadata['interaction_features'])}")

    configs = build_gpu_boosting_configs()

    summaries = []

    for config in configs:
        try:
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

        except ImportError as exc:
            print(f"[SKIP] {config.experiment_name}: {exc}")

        except Exception as exc:
            print(f"[ERROR] {config.experiment_name}: {exc}")
            print(
                "Gợi ý: nếu lỗi GPU, thử chạy bản CPU hoặc kiểm tra CUDA/OpenCL/driver."
            )
            raise

    if not summaries:
        raise RuntimeError("Không có GPU boosting experiment nào chạy thành công.")

    print("\n[3] Saving comparison...")
    comparison_df = pd.DataFrame(summaries)

    COMPARISON_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(
        COMPARISON_CSV_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    generate_model_comparison_report(
        comparison_df=comparison_df,
        output_path=COMPARISON_REPORT_PATH,
    )

    print(f"Saved comparison CSV: {COMPARISON_CSV_PATH}")
    print(f"Saved comparison report: {COMPARISON_REPORT_PATH}")

    show_cols = [
        "experiment_name",
        "model_name",
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

    print("\n[4] Result summary:")
    print(
        comparison_df[show_cols]
        .sort_values(["test_f2", "test_pr_auc"], ascending=False)
        .to_string(index=False)
    )

    print("\nDone!")
    print("=" * 80)


if __name__ == "__main__":
    main()