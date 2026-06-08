from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.training.train import (
    load_feature_metadata,
    train_logistic_regression_baseline,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FEATURES_PATH = PROCESSED_DIR / "model_train_features.parquet"
VALIDATION_FEATURES_PATH = PROCESSED_DIR / "model_validation_features.parquet"
TEST_FEATURES_PATH = PROCESSED_DIR / "model_test_features.parquet"
FEATURE_METADATA_PATH = PROCESSED_DIR / "feature_metadata.json"

MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
MODEL_REPORT_DIR = PROJECT_ROOT / "data" / "reports" / "model"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {path}\n"
            f"Hãy chạy giai đoạn 4 trước: python scripts/04_build_features.py"
        )


def main():
    print("=" * 80)
    print("GIAI ĐOẠN 5: TRAIN LOGISTIC REGRESSION BASELINE")
    print("=" * 80)

    print("\n[1] Checking input files...")
    for path in [
        TRAIN_FEATURES_PATH,
        VALIDATION_FEATURES_PATH,
        TEST_FEATURES_PATH,
        FEATURE_METADATA_PATH,
    ]:
        require_file(path)
        print(f"Found: {path}")

    print("\n[2] Loading datasets...")
    train_df = pd.read_parquet(TRAIN_FEATURES_PATH)
    validation_df = pd.read_parquet(VALIDATION_FEATURES_PATH)
    test_df = pd.read_parquet(TEST_FEATURES_PATH)

    print(f"Train shape:      {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")
    print(f"Test shape:       {test_df.shape}")

    print("\n[3] Loading feature metadata...")
    metadata = load_feature_metadata(FEATURE_METADATA_PATH)

    print(f"Numeric features:      {len(metadata['numeric_features'])}")
    print(f"Categorical features:  {len(metadata['categorical_features'])}")
    print(f"Binary features:       {len(metadata['binary_features'])}")
    print(f"Interaction features:  {len(metadata['interaction_features'])}")

    print("\n[4] Training Logistic Regression...")
    model_metadata = train_logistic_regression_baseline(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        metadata=metadata,
        output_dir=MODEL_OUTPUT_DIR,
        report_dir=MODEL_REPORT_DIR,
        model_name="logistic_regression",
        class_weight="balanced",
        C=1.0,

        # Quan trọng cho mcc nhiều category hiếm
        use_rare_category_grouping=True,
        min_frequency=50,

        # Fraud detection ưu tiên recall hơn precision
        threshold_metric="f2",
    )

    print("\n[5] Done!")
    print(f"Model saved to: {model_metadata['model_path']}")
    print(f"Selected threshold: {model_metadata['selected_threshold']}")
    print("=" * 80)


if __name__ == "__main__":
    main()