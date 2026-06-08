from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.data.load_data import load_fraud_data
from src.data.clean_data import clean_raw_data, save_cleaned_data, get_cleaning_summary
from src.eda.overview import save_overview_outputs


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fraud2.csv"

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
GLOBAL_REPORT_DIR = PROJECT_ROOT / "reports"

CLEANED_DATA_PATH = INTERIM_DIR / "fraud_cleaned.parquet"

TARGET_COL = "fraud"


def print_new_columns(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> None:
    raw_cols = set(raw_df.columns)
    cleaned_cols = set(cleaned_df.columns)

    new_cols = sorted(list(cleaned_cols - raw_cols))

    print("\nNew columns created:")
    for col in new_cols:
        print(f"- {col}")


def main():
    print("=" * 80)
    print("GIAI ĐOẠN 2: CLEANING + PARSING DATA")
    print("=" * 80)

    print("\n[1] Loading raw data...")
    raw_df = load_fraud_data(DATA_PATH)

    print(f"Raw shape: {raw_df.shape[0]:,} rows x {raw_df.shape[1]:,} columns")

    print("\n[2] Cleaning and parsing data...")
    cleaned_df = clean_raw_data(raw_df, drop_duplicates=False)

    print(f"Cleaned shape: {cleaned_df.shape[0]:,} rows x {cleaned_df.shape[1]:,} columns")

    print_new_columns(raw_df, cleaned_df)

    print("\n[3] Saving cleaned data...")
    save_cleaned_data(cleaned_df, CLEANED_DATA_PATH)
    print(f"Saved cleaned data to: {CLEANED_DATA_PATH}")

    print("\n[4] Saving cleaning summary...")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cleaning_summary = get_cleaning_summary(raw_df, cleaned_df)
    cleaning_summary_path = REPORT_DIR / "cleaning_summary.csv"
    cleaning_summary.to_csv(cleaning_summary_path, index=False)

    print(cleaning_summary)
    print(f"Saved cleaning summary to: {cleaning_summary_path}")

    print("\n[5] Saving overview report for cleaned data...")
    cleaned_report_dir = REPORT_DIR / "cleaned_data"
    cleaned_markdown_dir = GLOBAL_REPORT_DIR / "cleaned_data"

    save_overview_outputs(
        df=cleaned_df,
        output_dir=cleaned_report_dir,
        report_dir=cleaned_markdown_dir,
        target_col=TARGET_COL,
    )

    print(f"Saved cleaned data reports to: {cleaned_report_dir}")
    print(f"Saved markdown report to: {cleaned_markdown_dir}")

    print("\n[6] Quick check important parsed columns...")

    important_cols = [
        "ip_risk",
        "ip_address",
        "ip_score",
        "txn_counts",
        "txn_count_7d",
        "txn_count_30d",
        "txn_count_ratio_7d_30d",
        "local_timestamp",
        "transaction_datetime",
        "hour",
        "day_of_week",
        "is_weekend",
        "is_night_txn",
        "night_unusual_score",
        "is_night_unusual",
    ]

    existing_cols = [col for col in important_cols if col in cleaned_df.columns]

    print(cleaned_df[existing_cols].head(10))

    print("\nDone!")
    print("=" * 80)


if __name__ == "__main__":
    main()