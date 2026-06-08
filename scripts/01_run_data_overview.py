from pathlib import Path
import sys

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import (
    load_fraud_data,
    load_metadata,
    get_basic_shape,
    get_column_types,
    infer_feature_groups,
    check_target_column,
)
from src.eda.overview import save_overview_outputs


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fraud2.csv"
METADATA_PATH = PROJECT_ROOT / "data" / "raw" / "fraud_metadata.xlsx"

DATA_REPORT_DIR = PROJECT_ROOT / "data" / "reports"
REPORT_DIR = PROJECT_ROOT / "reports"

TARGET_COL = "fraud"


def main():
    print("=" * 80)
    print("GIAI ĐOẠN 1: DATA UNDERSTANDING")
    print("=" * 80)

    print("\n[1] Loading fraud dataset...")
    df = load_fraud_data(DATA_PATH)
    print(f"Loaded dataset: {DATA_PATH}")
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    print("\n[2] Checking target column...")
    check_target_column(df, TARGET_COL)
    print(f"Target column found: {TARGET_COL}")

    print("\n[3] Loading metadata...")
    try:
        metadata = load_metadata(METADATA_PATH)
        print(f"Loaded metadata: {METADATA_PATH}")
        print(f"Metadata shape: {metadata.shape[0]:,} rows x {metadata.shape[1]:,} columns")
        metadata.to_csv(DATA_REPORT_DIR / "metadata_preview.csv", index=False)
    except FileNotFoundError as exc:
        print(f"Warning: {exc}")
        metadata = None

    print("\n[4] Basic shape...")
    shape_info = get_basic_shape(df)
    print(shape_info)

    print("\n[5] Column types...")
    column_types = get_column_types(df)
    DATA_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    column_types.to_csv(DATA_REPORT_DIR / "column_types.csv", index=False)
    print(column_types.head(10))

    print("\n[6] Inferring feature groups...")
    feature_groups = infer_feature_groups(df, target_col=TARGET_COL)

    for group_name, cols in feature_groups.items():
        print(f"\n{group_name}: {len(cols)} columns")
        print(cols)

    print("\n[7] Saving overview outputs...")
    save_overview_outputs(
        df=df,
        output_dir=DATA_REPORT_DIR,
        report_dir=REPORT_DIR,
        target_col=TARGET_COL,
    )

    print("\nDone!")
    print(f"CSV reports saved to: {DATA_REPORT_DIR}")
    print(f"Markdown report saved to: {REPORT_DIR / 'eda_report.md'}")
    print("=" * 80)


if __name__ == "__main__":
    main()