from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.features.build_features import (
    FraudFeatureBuilder,
    create_feature_report,
    generate_feature_engineering_markdown_report,
    select_model_dataframe,
)


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_PATH = PROCESSED_DIR / "train.parquet"
VALIDATION_PATH = PROCESSED_DIR / "validation.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"

TRAIN_FEATURES_PATH = PROCESSED_DIR / "train_features.parquet"
VALIDATION_FEATURES_PATH = PROCESSED_DIR / "validation_features.parquet"
TEST_FEATURES_PATH = PROCESSED_DIR / "test_features.parquet"

MODEL_TRAIN_FEATURES_PATH = PROCESSED_DIR / "model_train_features.parquet"
MODEL_VALIDATION_FEATURES_PATH = PROCESSED_DIR / "model_validation_features.parquet"
MODEL_TEST_FEATURES_PATH = PROCESSED_DIR / "model_test_features.parquet"

FEATURE_METADATA_PATH = PROCESSED_DIR / "feature_metadata.json"
FEATURE_REPORT_PATH = PROJECT_ROOT / "data" / "reports" / "feature_report.csv"
FEATURE_MARKDOWN_REPORT_PATH = PROJECT_ROOT / "reports" / "feature_engineering_report.md"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file: {path}\n"
            f"Hãy chạy giai đoạn 3 trước để tạo train/validation/test parquet."
        )


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    require_file(TRAIN_PATH)
    require_file(VALIDATION_PATH)
    require_file(TEST_PATH)

    train_df = pd.read_parquet(TRAIN_PATH)
    validation_df = pd.read_parquet(VALIDATION_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    return train_df, validation_df, test_df


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved: {path}")


def main():
    print("=" * 80)
    print("GIAI ĐOẠN 4: FEATURE ENGINEERING")
    print("=" * 80)

    print("\n[1] Loading train/validation/test splits...")
    train_df, validation_df, test_df = load_splits()

    print(f"Train shape:      {train_df.shape}")
    print(f"Validation shape: {validation_df.shape}")
    print(f"Test shape:       {test_df.shape}")

    print("\n[2] Fitting feature builder on TRAIN only...")
    builder = FraudFeatureBuilder(target_col="fraud")
    train_features = builder.fit_transform(train_df)

    print("\n[3] Transforming validation/test using TRAIN thresholds...")
    validation_features = builder.transform(validation_df)
    test_features = builder.transform(test_df)

    print(f"Train features shape:      {train_features.shape}")
    print(f"Validation features shape: {validation_features.shape}")
    print(f"Test features shape:       {test_features.shape}")

    print("\n[4] Building metadata...")
    metadata = builder.build_metadata(train_features)

    print(f"Numeric features:      {len(metadata.numeric_features)}")
    print(f"Categorical features:  {len(metadata.categorical_features)}")
    print(f"Binary features:       {len(metadata.binary_features)}")
    print(f"Interaction features:  {len(metadata.interaction_features)}")
    print(f"All model features:    {len(metadata.all_model_features)}")

    print("\n[5] Saving full feature datasets...")
    save_parquet(train_features, TRAIN_FEATURES_PATH)
    save_parquet(validation_features, VALIDATION_FEATURES_PATH)
    save_parquet(test_features, TEST_FEATURES_PATH)

    print("\n[6] Saving model-ready feature datasets...")
    model_train = select_model_dataframe(train_features, metadata)
    model_validation = select_model_dataframe(validation_features, metadata)
    model_test = select_model_dataframe(test_features, metadata)

    save_parquet(model_train, MODEL_TRAIN_FEATURES_PATH)
    save_parquet(model_validation, MODEL_VALIDATION_FEATURES_PATH)
    save_parquet(model_test, MODEL_TEST_FEATURES_PATH)

    print("\n[7] Saving feature metadata...")
    builder.save_metadata(FEATURE_METADATA_PATH)
    print(f"Saved: {FEATURE_METADATA_PATH}")

    print("\n[8] Saving feature report...")
    feature_report = create_feature_report(
        train_features=train_features,
        validation_features=validation_features,
        test_features=test_features,
        metadata=metadata,
    )

    FEATURE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_report.to_csv(FEATURE_REPORT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved: {FEATURE_REPORT_PATH}")

    generate_feature_engineering_markdown_report(
        output_path=FEATURE_MARKDOWN_REPORT_PATH,
        feature_report=feature_report,
        metadata=metadata,
    )
    print(f"Saved: {FEATURE_MARKDOWN_REPORT_PATH}")

    print("\n[9] Quick check fraud rate after feature engineering...")
    for name, df in [
        ("train", model_train),
        ("validation", model_validation),
        ("test", model_test),
    ]:
        print(
            f"{name}: rows={len(df):,}, "
            f"fraud_count={int(df['fraud'].sum()):,}, "
            f"fraud_rate={df['fraud'].mean():.4f}"
        )

    print("\nDone!")
    print("=" * 80)


if __name__ == "__main__":
    main()