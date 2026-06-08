from __future__ import annotations

from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


def write_notebook(filename: str, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    path = NOTEBOOK_DIR / filename
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {path}")


COMMON_SETUP = r"""
from pathlib import Path
import sys
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.append(str(PROJECT_ROOT))

pd.set_option("display.max_columns", 300)
pd.set_option("display.width", 200)

print("PROJECT_ROOT:", PROJECT_ROOT)
"""


def generate_01_data_understanding() -> None:
    cells = [
        md_cell(
            """
            # 01 - Data Understanding

            Mục tiêu notebook này:

            - Hiểu bộ dữ liệu gốc.
            - Kiểm tra shape, column, dtype.
            - Kiểm tra missing value.
            - Kiểm tra phân phối target `fraud`.
            - Đọc các report giai đoạn 1 nếu đã sinh từ script.
            """
        ),
        code_cell(COMMON_SETUP),
        code_cell(
            r"""
RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

print("Raw files:")
for p in RAW_DIR.glob("*"):
    print("-", p.name)
"""
        ),
        code_cell(
            r"""
# Ưu tiên tên file chuẩn. Nếu bạn đang dùng tên khác, sửa lại biến DATA_PATH.
DATA_PATH = RAW_DIR / "fraud2.csv"
METADATA_PATH = RAW_DIR / "fraud_metadata.xlsx"

if not DATA_PATH.exists():
    csv_candidates = list(RAW_DIR.glob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError("Không tìm thấy file CSV trong data/raw")
    DATA_PATH = csv_candidates[0]

print("Using DATA_PATH:", DATA_PATH)

df = pd.read_csv(DATA_PATH)
df.head()
"""
        ),
        code_cell(
            r"""
print("Shape:", df.shape)
display(df.head())
display(df.tail())
"""
        ),
        code_cell(
            r"""
df.info()
"""
        ),
        code_cell(
            r"""
display(df.describe(include="all").T)
"""
        ),
        code_cell(
            r"""
TARGET_COL = "fraud"

if TARGET_COL in df.columns:
    target_dist = (
        df[TARGET_COL]
        .value_counts(dropna=False)
        .rename_axis(TARGET_COL)
        .reset_index(name="count")
    )
    target_dist["ratio"] = target_dist["count"] / len(df)
    display(target_dist)

    plt.figure(figsize=(5, 4))
    plt.bar(target_dist[TARGET_COL].astype(str), target_dist["count"])
    plt.title("Target distribution")
    plt.xlabel(TARGET_COL)
    plt.ylabel("count")
    plt.show()
else:
    print(f"Không tìm thấy target column: {TARGET_COL}")
"""
        ),
        code_cell(
            r"""
missing = (
    df.isna()
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
missing.columns = ["column", "missing_rate"]
display(missing.head(40))
"""
        ),
        code_cell(
            r"""
dtype_report = pd.DataFrame({
    "column": df.columns,
    "dtype": [str(df[c].dtype) for c in df.columns],
    "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
    "missing_rate": [df[c].isna().mean() for c in df.columns],
})
display(dtype_report.sort_values(["missing_rate", "n_unique"], ascending=[False, False]).head(50))
"""
        ),
        code_cell(
            r"""
# Đọc các report giai đoạn 1 nếu tồn tại
candidate_reports = [
    "dataset_overview.csv",
    "column_overview.csv",
    "column_types.csv",
    "missing_report.csv",
    "target_distribution.csv",
    "high_cardinality_report.csv",
]

for file in candidate_reports:
    path = REPORT_DIR / file
    if path.exists():
        print("\n" + "=" * 100)
        print(file)
        display(pd.read_csv(path).head(30))
"""
        ),
        md_cell(
            """
            ## Kết luận cần ghi sau khi chạy

            - Dataset có bao nhiêu dòng/cột?
            - Target fraud bị mất cân bằng bao nhiêu?
            - Nhóm cột nào nhiều missing?
            - Có cột categorical cardinality cao nào cần xử lý không?
            """
        ),
    ]

    write_notebook("01_data_understanding.ipynb", cells)


def generate_02_univariate_eda() -> None:
    cells = [
        md_cell(
            """
            # 02 - Univariate EDA

            Mục tiêu:

            - Phân tích từng feature riêng lẻ.
            - Kiểm tra phân phối numeric.
            - Kiểm tra cardinality categorical.
            - Phát hiện skew/outlier/missing.
            """
        ),
        code_cell(COMMON_SETUP),
        code_cell(
            r"""
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

candidate_paths = [
    INTERIM_DIR / "fraud_cleaned.parquet",
    PROCESSED_DIR / "cleaned_data.parquet",
]

for path in candidate_paths:
    if path.exists():
        DATA_PATH = path
        break
else:
    raw_csvs = list((PROJECT_ROOT / "data" / "raw").glob("*.csv"))
    if not raw_csvs:
        raise FileNotFoundError("Không tìm thấy dữ liệu cleaned hoặc raw csv.")
    DATA_PATH = raw_csvs[0]

print("Using:", DATA_PATH)

if DATA_PATH.suffix == ".parquet":
    df = pd.read_parquet(DATA_PATH)
else:
    df = pd.read_csv(DATA_PATH)

df.head()
"""
        ),
        code_cell(
            r"""
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

print("Numeric columns:", len(numeric_cols))
print("Categorical columns:", len(categorical_cols))
"""
        ),
        code_cell(
            r"""
numeric_summary = df[numeric_cols].describe().T
numeric_summary["missing_rate"] = df[numeric_cols].isna().mean()
numeric_summary["skew"] = df[numeric_cols].skew(numeric_only=True)
display(numeric_summary.sort_values("missing_rate", ascending=False).head(50))
"""
        ),
        code_cell(
            r"""
important_numeric = [
    "amount",
    "mean_amount_30d",
    "max_amount_30d",
    "std_amount_30d",
    "mcc_entropy_30d",
    "night_ratio_30d",
    "spending_trend",
    "ip_score",
    "decline_rate_30d",
    "credit_util_today",
    "txn_count_7d",
    "txn_count_30d",
]

for col in important_numeric:
    if col in df.columns:
        plt.figure(figsize=(7, 4))
        df[col].dropna().hist(bins=50)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("count")
        plt.show()
"""
        ),
        code_cell(
            r"""
cardinality = pd.DataFrame({
    "column": categorical_cols,
    "n_unique": [df[c].nunique(dropna=True) for c in categorical_cols],
    "missing_rate": [df[c].isna().mean() for c in categorical_cols],
}).sort_values("n_unique", ascending=False)

display(cardinality)
"""
        ),
        code_cell(
            r"""
for col in categorical_cols[:30]:
    print("\n" + "=" * 100)
    print(col)
    display(df[col].astype(str).value_counts(dropna=False).head(20))
"""
        ),
        code_cell(
            r"""
# Đọc report EDA nếu có
for file in [
    "numeric_summary.csv",
    "categorical_summary.csv",
    "missing_report.csv",
    "high_cardinality_report.csv",
]:
    path = REPORT_DIR / file
    if path.exists():
        print("\n" + "=" * 100)
        print(file)
        display(pd.read_csv(path).head(30))
"""
        ),
        md_cell(
            """
            ## Kết luận cần ghi sau khi chạy

            - Feature numeric nào skew mạnh?
            - Feature nào có outlier?
            - Categorical nào nhiều giá trị hiếm?
            - Cột nào nên log/bin/group rare category?
            """
        ),
    ]

    write_notebook("02_univariate_eda.ipynb", cells)


def generate_03_bivariate_eda() -> None:
    cells = [
        md_cell(
            """
            # 03 - Bivariate EDA with Target

            Mục tiêu:

            - Phân tích quan hệ từng feature với target `fraud`.
            - Tính fraud rate, lift theo category/bin.
            - Tìm tín hiệu ban đầu để đưa sang insight mining.
            """
        ),
        code_cell(COMMON_SETUP),
        code_cell(
            r"""
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "data" / "reports"

candidate_paths = [
    INTERIM_DIR / "fraud_cleaned.parquet",
    PROCESSED_DIR / "cleaned_data.parquet",
]

for path in candidate_paths:
    if path.exists():
        DATA_PATH = path
        break
else:
    raw_csvs = list((PROJECT_ROOT / "data" / "raw").glob("*.csv"))
    if not raw_csvs:
        raise FileNotFoundError("Không tìm thấy dữ liệu cleaned hoặc raw csv.")
    DATA_PATH = raw_csvs[0]

if DATA_PATH.suffix == ".parquet":
    df = pd.read_parquet(DATA_PATH)
else:
    df = pd.read_csv(DATA_PATH)

TARGET_COL = "fraud"
BASE_FRAUD_RATE = df[TARGET_COL].mean()

print("Using:", DATA_PATH)
print("Base fraud rate:", BASE_FRAUD_RATE)
df.head()
"""
        ),
        code_cell(
            r"""
def fraud_rate_table(df: pd.DataFrame, col: str, target_col: str = "fraud", min_count: int = 50) -> pd.DataFrame:
    out = (
        df.groupby(col, dropna=False)[target_col]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"sum": "fraud_count", "mean": "fraud_rate"})
    )
    out["lift"] = out["fraud_rate"] / df[target_col].mean()
    out = out[out["count"] >= min_count]
    return out.sort_values("lift", ascending=False)


def numeric_bin_fraud_rate(df: pd.DataFrame, col: str, target_col: str = "fraud", q: int = 10) -> pd.DataFrame:
    tmp = df[[col, target_col]].copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    tmp = tmp.dropna(subset=[col])

    if tmp[col].nunique() < 3:
        return pd.DataFrame()

    try:
        tmp[f"{col}_bin"] = pd.qcut(tmp[col], q=q, duplicates="drop")
    except ValueError:
        return pd.DataFrame()

    return fraud_rate_table(tmp, f"{col}_bin", target_col=target_col, min_count=1)
"""
        ),
        code_cell(
            r"""
categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

for col in categorical_cols:
    if col == TARGET_COL:
        continue

    print("\n" + "=" * 100)
    print(col)
    display(fraud_rate_table(df, col, target_col=TARGET_COL, min_count=50).head(15))
"""
        ),
        code_cell(
            r"""
important_numeric = [
    "amount",
    "mcc_entropy_30d",
    "night_ratio_30d",
    "spending_trend",
    "max_amount_30d",
    "mean_amount_30d",
    "distinct_countries_30d",
    "device_diversity_30d",
    "decline_rate_30d",
    "credit_util_today",
    "ip_score",
    "txn_count_7d",
    "txn_count_30d",
]

for col in important_numeric:
    if col in df.columns:
        print("\n" + "=" * 100)
        print(col)
        table = numeric_bin_fraud_rate(df, col, target_col=TARGET_COL, q=10)
        display(table)

        if not table.empty:
            plt.figure(figsize=(10, 4))
            plt.bar(table[f"{col}_bin"].astype(str), table["fraud_rate"])
            plt.axhline(BASE_FRAUD_RATE, linestyle="--")
            plt.title(f"Fraud rate by {col} bin")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.show()
"""
        ),
        code_cell(
            r"""
INSIGHT_DIR = REPORT_DIR / "insights"

if INSIGHT_DIR.exists():
    for file in sorted(INSIGHT_DIR.glob("*.csv")):
        print("\n" + "=" * 100)
        print(file.name)
        display(pd.read_csv(file).head(20))
else:
    print("Chưa có data/reports/insights")
"""
        ),
        md_cell(
            """
            ## Kết luận cần ghi sau khi chạy

            - Category/bin nào có lift cao?
            - Numeric nào có fraud rate khác biệt rõ?
            - Insight nào nên kiểm tra stability trên train/validation?
            """
        ),
    ]

    write_notebook("03_bivariate_eda_with_target.ipynb", cells)


def generate_04_hidden_insight() -> None:
    cells = [
        md_cell(
            """
            # 04 - Hidden Insight Mining

            Mục tiêu:

            - Đọc kết quả insight mining.
            - Kiểm tra các pattern/rule mạnh.
            - Kết nối insight với feature engineering.
            - Tránh leakage: insight nên được tìm trên train và kiểm tra stability validation.
            """
        ),
        code_cell(COMMON_SETUP),
        code_cell(
            r"""
REPORT_DIR = PROJECT_ROOT / "data" / "reports"
INSIGHT_DIR = REPORT_DIR / "insights"

print("INSIGHT_DIR exists:", INSIGHT_DIR.exists())
"""
        ),
        code_cell(
            r"""
if INSIGHT_DIR.exists():
    files = sorted(INSIGHT_DIR.glob("*.csv"))
    print("Insight files:", [f.name for f in files])

    for file in files:
        print("\n" + "=" * 100)
        print(file.name)
        display(pd.read_csv(file).head(30))
else:
    print("Chưa có thư mục insight. Hãy chạy scripts/03_run_insight_mining.py trước.")
"""
        ),
        code_cell(
            r"""
for md_file in [
    REPORT_DIR / "insight_report.md",
    REPORT_DIR / "insight_stability_report.md",
    PROJECT_ROOT / "reports" / "insight_report.md",
    PROJECT_ROOT / "reports" / "insight_stability_report.md",
]:
    if md_file.exists():
        print("\n" + "=" * 100)
        print(md_file)
        print(md_file.read_text(encoding="utf-8")[:5000])
"""
        ),
        code_cell(
            r"""
selected_insights = pd.DataFrame([
    {
        "insight": "low_mcc_entropy",
        "business_meaning": "User có lịch sử ngành hàng/MCC ít đa dạng.",
        "feature_action": "Giữ mcc_entropy_30d, tạo low_mcc_entropy_flag/very_low_mcc_entropy_flag.",
        "expected_model_effect": "Entropy thấp làm tăng fraud risk.",
    },
    {
        "insight": "low_night_ratio",
        "business_meaning": "User ít có lịch sử giao dịch ban đêm.",
        "feature_action": "Giữ night_ratio_30d, tạo low_night_ratio_flag, night_unusual_score.",
        "expected_model_effect": "Giao dịch ban đêm bất thường làm tăng fraud risk.",
    },
    {
        "insight": "amount_anomaly",
        "business_meaning": "Amount hiện tại lệch khỏi lịch sử user.",
        "feature_action": "Tạo amount_z_30d, amount_to_mean_30d, log_amount_to_max_30d.",
        "expected_model_effect": "Amount anomaly cao làm tăng fraud risk.",
    },
    {
        "insight": "low_spending_trend",
        "business_meaning": "Xu hướng chi tiêu thấp/bất thường.",
        "feature_action": "Giữ spending_trend, tạo low_spending_trend_flag.",
        "expected_model_effect": "Spending trend thấp có thể làm tăng risk.",
    },
    {
        "insight": "low_country_diversity",
        "business_meaning": "Lịch sử quốc gia ít đa dạng.",
        "feature_action": "Giữ distinct_countries_30d, tạo low_country_diversity_flag.",
        "expected_model_effect": "Country diversity thấp làm tăng risk.",
    },
])

display(selected_insights)
"""
        ),
        md_cell(
            """
            ## Kết luận cần ghi sau khi chạy

            - Insight nào ổn định trên train/validation?
            - Insight nào đã chuyển thành feature?
            - Feature nào cần cẩn thận leakage?
            """
        ),
    ]

    write_notebook("04_hidden_insight_mining.ipynb", cells)


def generate_05_feature_engineering() -> None:
    cells = [
        md_cell(
            """
            # 05 - Feature Engineering Experiment

            Mục tiêu:

            - Kiểm tra output giai đoạn feature engineering.
            - Kiểm tra feature metadata.
            - Kiểm tra train/validation/test split.
            - Kiểm tra missing/correlation để đảm bảo data sẵn sàng train model.
            """
        ),
        code_cell(COMMON_SETUP),
        code_cell(
            r"""
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

train_path = PROCESSED_DIR / "model_train_features.parquet"
val_path = PROCESSED_DIR / "model_validation_features.parquet"
test_path = PROCESSED_DIR / "model_test_features.parquet"
metadata_path = PROCESSED_DIR / "feature_metadata.json"

for path in [train_path, val_path, test_path, metadata_path]:
    print(path, path.exists())
"""
        ),
        code_cell(
            r"""
train_df = pd.read_parquet(train_path)
val_df = pd.read_parquet(val_path)
test_df = pd.read_parquet(test_path)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)

display(train_df.head(3).T.rename(columns=lambda i: f"row_{i}"))
"""
        ),
        code_cell(
            r"""
with open(metadata_path, "r", encoding="utf-8") as f:
    metadata = json.load(f)

for key in ["numeric_features", "categorical_features", "binary_features", "interaction_features"]:
    print("\n" + "=" * 100)
    values = metadata.get(key, [])
    print(key, len(values))
    print(values[:15])
    if len(values) > 15:
        print(f"... {len(values) - 15} more")
"""
        ),
        code_cell(
            r"""
for name, d in [("train", train_df), ("validation", val_df), ("test", test_df)]:
    print("\n" + "=" * 100)
    print(name)
    target_dist = d["fraud"].value_counts(normalize=True).rename("ratio").reset_index()
    display(target_dist)
"""
        ),
        code_cell(
            r"""
all_features = metadata["all_model_features"]

missing = (
    train_df[all_features]
    .isna()
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
missing.columns = ["feature", "missing_rate"]
display(missing.head(40))
"""
        ),
        code_cell(
            r"""
numeric_features = metadata["numeric_features"]
corr = train_df[numeric_features].corr(numeric_only=True)

corr_abs_pairs = (
    corr.abs()
    .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .sort_values(ascending=False)
    .reset_index()
)
corr_abs_pairs.columns = ["feature_1", "feature_2", "abs_corr"]

display(corr_abs_pairs.head(30))
"""
        ),
        code_cell(
            r"""
print("Dropped columns:", metadata.get("dropped_columns"))
print("Thresholds:", metadata.get("thresholds"))
"""
        ),
        md_cell(
            """
            ## Kết luận cần ghi sau khi chạy

            - Feature set đã đủ cho model chưa?
            - Có feature missing bất thường không?
            - Có feature tương quan cao cần loại cho Logistic Regression không?
            - Với CatBoost, interaction thủ công có thể bỏ được không?
            """
        ),
    ]

    write_notebook("05_feature_engineering_experiment.ipynb", cells)


def generate_06_logistic() -> None:
    cells = [
        md_cell(
            """
            # 06 - Logistic Regression Baseline

            Mục tiêu:

            - Kiểm tra baseline Logistic Regression.
            - Đọc metrics train/validation/test.
            - Đọc coefficient để giải thích model.
            - Làm mốc so sánh với CatBoost.
            """
        ),
        code_cell(COMMON_SETUP),
        code_cell(
            r"""
LR_DIR = PROJECT_ROOT / "data" / "reports" / "model" / "experiments" / "lr_no_mcc"

print("LR_DIR exists:", LR_DIR.exists())
print([p.name for p in LR_DIR.glob("*")] if LR_DIR.exists() else [])
"""
        ),
        code_cell(
            r"""
metrics_files = [
    "metrics_train_threshold_05.csv",
    "metrics_validation_best_threshold.csv",
    "metrics_test.csv",
]

metrics = []

for file in metrics_files:
    path = LR_DIR / file
    if path.exists():
        metrics.append(pd.read_csv(path))
    else:
        print("Missing:", path)

metrics_df = pd.concat(metrics, ignore_index=True)
display(metrics_df[[
    "split", "threshold", "pr_auc", "roc_auc", "precision",
    "recall", "f1", "f2", "review_rate", "tp", "fp", "fn", "tn"
]])
"""
        ),
        code_cell(
            r"""
cm_path = LR_DIR / "confusion_matrix_test.csv"
if cm_path.exists():
    cm = pd.read_csv(cm_path, index_col=0)
    display(cm)
else:
    print("Không tìm thấy confusion_matrix_test.csv")
"""
        ),
        code_cell(
            r"""
coef_path = LR_DIR / "coefficients.csv"

coef = pd.read_csv(coef_path)
display(coef.head(20))
"""
        ),
        code_cell(
            r"""
top_positive = (
    coef[coef["coefficient"] > 0]
    .sort_values("coefficient", ascending=False)
    .head(20)
)

top_negative = (
    coef[coef["coefficient"] < 0]
    .sort_values("coefficient")
    .head(20)
)

print("Top positive coefficients")
display(top_positive)

print("Top negative coefficients")
display(top_negative)
"""
        ),
        code_cell(
            r"""
plt.figure(figsize=(10, 8))
top = coef.sort_values("abs_coefficient", ascending=False).head(20)
plt.barh(top["feature"][::-1], top["abs_coefficient"][::-1])
plt.title("Top 20 absolute coefficients - Logistic Regression")
plt.xlabel("abs coefficient")
plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """
            ## Kết luận cần ghi sau khi chạy

            - Logistic Regression là baseline dễ giải thích.
            - Coefficient có khớp insight không?
            - Baseline còn hạn chế vì mô hình tuyến tính.
            - Advanced model có cải thiện không?
            """
        ),
    ]

    write_notebook("06_model_baseline_logistic.ipynb", cells)


def generate_07_advanced() -> None:
    cells = [
        md_cell(
            """
            # 07 - Advanced Model Comparison

            Mục tiêu:

            - So sánh LightGBM, XGBoost, CatBoost.
            - So sánh CatBoost base, CatBoost no interactions, CatBoost tuning.
            - Chọn final model theo PR-AUC, F2, recall, precision, review rate.
            """
        ),
        code_cell(COMMON_SETUP),
        code_cell(
            r"""
GPU_REPORT = PROJECT_ROOT / "data" / "reports" / "gpu_boosting_models" / "gpu_boosting_model_comparison.csv"
TUNING_REPORT = PROJECT_ROOT / "data" / "reports" / "catboost_tuning" / "catboost_tuning_comparison.csv"

print("GPU_REPORT:", GPU_REPORT.exists())
print("TUNING_REPORT:", TUNING_REPORT.exists())
"""
        ),
        code_cell(
            r"""
dfs = []

if GPU_REPORT.exists():
    gpu_df = pd.read_csv(GPU_REPORT)
    gpu_df["source"] = "gpu_boosting"
    dfs.append(gpu_df)
    display(gpu_df.sort_values(["test_f2", "test_pr_auc"], ascending=False))

if TUNING_REPORT.exists():
    tune_df = pd.read_csv(TUNING_REPORT)
    tune_df["source"] = "catboost_tuning"
    dfs.append(tune_df)
    display(tune_df.sort_values(["test_f2", "test_pr_auc"], ascending=False))

all_models = pd.concat(dfs, ignore_index=True)
"""
        ),
        code_cell(
            r"""
compare_cols = [
    "source",
    "experiment_name",
    "model_name",
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

compare_cols = [c for c in compare_cols if c in all_models.columns]

best_models = all_models[compare_cols].sort_values(
    ["test_f2", "test_pr_auc"],
    ascending=False,
)

display(best_models.head(15))
"""
        ),
        code_cell(
            r"""
final_experiment = "cat_tune_depth7_l2_8_lr003_iter700"

final_row = best_models[best_models["experiment_name"] == final_experiment]
display(final_row)
"""
        ),
        code_cell(
            r"""
plot_df = best_models.head(10).copy()

plt.figure(figsize=(12, 5))
plt.bar(plot_df["experiment_name"], plot_df["test_f2"])
plt.title("Top models by Test F2")
plt.ylabel("test_f2")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.bar(plot_df["experiment_name"], plot_df["test_pr_auc"])
plt.title("Top models by Test PR-AUC")
plt.ylabel("test_pr_auc")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
"""
        ),
        md_cell(
            """
            ## Kết luận cần ghi sau khi chạy

            - CatBoost tuned depth7_l2_8 có F2/PR-AUC cao nhất.
            - Trade-off: recall tăng, false positive tăng so với base.
            - Review rate vẫn dưới 5%, phù hợp nghiệp vụ ưu tiên bắt fraud.
            """
        ),
    ]

    write_notebook("07_advanced_model_comparison.ipynb", cells)


def generate_08_explainability() -> None:
    cells = [
        md_cell(
            """
            # 08 - Final Model Explainability

            Mục tiêu:

            - Đọc feature importance của final CatBoost.
            - Kiểm tra model có học đúng insight không.
            - Gom feature importance theo nhóm nghiệp vụ.
            """
        ),
        code_cell(COMMON_SETUP),
        code_cell(
            r"""
FI_PATH = (
    PROJECT_ROOT
    / "data"
    / "reports"
    / "catboost_tuning"
    / "cat_tune_depth7_l2_8_lr003_iter700"
    / "feature_importance.csv"
)

print("FI_PATH exists:", FI_PATH.exists())

fi = pd.read_csv(FI_PATH)
display(fi.head(30))
"""
        ),
        code_cell(
            r"""
top = fi.sort_values("importance", ascending=False).head(25)

plt.figure(figsize=(10, 9))
plt.barh(top["feature"][::-1], top["importance"][::-1])
plt.title("Top 25 Feature Importance - Final CatBoost")
plt.xlabel("importance")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            r"""
def map_insight_group(feature: str) -> str:
    f = feature.lower()

    if "amount" in f:
        return "Amount anomaly"
    if "mcc_entropy" in f or "mcc" in f:
        return "MCC behavior"
    if "night" in f or "time_period" in f or "hour" in f:
        return "Time/night behavior"
    if "spending_trend" in f:
        return "Spending trend"
    if "country" in f or "cross_border" in f or "countries" in f:
        return "Country behavior"
    if (
        "decline" in f
        or "ip_score" in f
        or "device" in f
        or "chargeback" in f
        or "credit" in f
    ):
        return "Risk/technical"
    if "txn_count" in f or "velocity" in f:
        return "Velocity"
    if (
        "payment" in f
        or "auth" in f
        or "card" in f
        or "pin" in f
        or "term_location" in f
    ):
        return "Payment/auth context"

    return "Other"


fi["insight_group"] = fi["feature"].apply(map_insight_group)

group_summary = (
    fi.groupby("insight_group")["importance"]
    .sum()
    .sort_values(ascending=False)
    .reset_index()
)

display(group_summary)
"""
        ),
        code_cell(
            r"""
plt.figure(figsize=(10, 5))
plt.bar(group_summary["insight_group"], group_summary["importance"])
plt.title("Feature importance by insight group")
plt.ylabel("total importance")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            r"""
# Kiểm tra model có phụ thuộc trực tiếp vào MCC category không
mcc_rows = fi[fi["feature"].str.contains("mcc", case=False, na=False)]
display(mcc_rows)
"""
        ),
        md_cell(
            """
            ## Kết luận cần ghi sau khi chạy

            - Top feature có khớp insight giai đoạn 3 không?
            - Model có phụ thuộc vào mã MCC cụ thể không?
            - Feature importance không cho biết chiều tác động; muốn biết chiều tác động cần SHAP/PDP hoặc tham khảo Logistic Regression coefficient.
            """
        ),
    ]

    write_notebook("08_final_model_explainability.ipynb", cells)


def generate_09_error_threshold() -> None:
    cells = [
        md_cell(
            """
            # 09 - Error Analysis and Threshold Policy

            Mục tiêu:

            - Phân tích FP/FN của final model.
            - Kiểm tra threshold policy theo năng lực review 3%, 5%, 10%.
            - Diễn giải kết quả theo nghiệp vụ.
            """
        ),
        code_cell(COMMON_SETUP),
        code_cell(
            r"""
ERROR_DIR = PROJECT_ROOT / "data" / "reports" / "final_error_analysis"
POLICY_DIR = PROJECT_ROOT / "data" / "reports" / "threshold_policy"

print("ERROR_DIR exists:", ERROR_DIR.exists())
print("POLICY_DIR exists:", POLICY_DIR.exists())

if ERROR_DIR.exists():
    print("Error files:", [p.name for p in ERROR_DIR.glob("*")])

if POLICY_DIR.exists():
    print("Policy files:", [p.name for p in POLICY_DIR.glob("*")])
"""
        ),
        code_cell(
            r"""
error_counts_path = ERROR_DIR / "error_type_counts.csv"

if error_counts_path.exists():
    error_counts = pd.read_csv(error_counts_path)
    display(error_counts)

    plt.figure(figsize=(6, 4))
    plt.bar(error_counts["error_type"], error_counts["count"])
    plt.title("Error type counts")
    plt.xlabel("error_type")
    plt.ylabel("count")
    plt.show()
else:
    print("Chưa có error_type_counts.csv. Hãy chạy scripts/12_run_error_analysis.py")
"""
        ),
        code_cell(
            r"""
numeric_summary_path = ERROR_DIR / "error_numeric_summary.csv"

if numeric_summary_path.exists():
    numeric_summary = pd.read_csv(numeric_summary_path)
    display(numeric_summary)
else:
    print("Chưa có error_numeric_summary.csv")
"""
        ),
        code_cell(
            r"""
categorical_summary_path = ERROR_DIR / "error_categorical_summary.csv"

if categorical_summary_path.exists():
    categorical_summary = pd.read_csv(categorical_summary_path)
    display(categorical_summary.head(80))
else:
    print("Chưa có error_categorical_summary.csv")
"""
        ),
        code_cell(
            r"""
policy_path = POLICY_DIR / "threshold_policy_review_capacity.csv"

if policy_path.exists():
    policy = pd.read_csv(policy_path)
    display(policy)

    show_cols = [
        "review_capacity",
        "selected_threshold_from_validation",
        "test_precision",
        "test_recall",
        "test_f2",
        "test_review_rate",
        "test_tp",
        "test_fp",
        "test_fn",
    ]
    show_cols = [c for c in show_cols if c in policy.columns]
    display(policy[show_cols])
else:
    print("Chưa có threshold policy. Hãy chạy scripts/13_generate_threshold_policy.py")
"""
        ),
        code_cell(
            r"""
if policy_path.exists():
    plt.figure(figsize=(7, 4))
    plt.plot(policy["review_capacity"], policy["test_recall"], marker="o", label="Recall")
    plt.plot(policy["review_capacity"], policy["test_precision"], marker="o", label="Precision")
    plt.plot(policy["review_capacity"], policy["test_f2"], marker="o", label="F2")
    plt.title("Metrics by review capacity")
    plt.xlabel("review_capacity")
    plt.ylabel("metric")
    plt.legend()
    plt.grid(True)
    plt.show()
"""
        ),
        md_cell(
            """
            ## Kết luận cần ghi sau khi chạy

            - FN là nhóm nguy hiểm nhất vì fraud bị bỏ sót.
            - FP gây tải review và ảnh hưởng trải nghiệm khách hàng.
            - Threshold policy giúp chọn ngưỡng theo năng lực vận hành.
            - Review capacity 5% thường là trade-off hợp lý cho bài toán này.
            """
        ),
    ]

    write_notebook("09_error_analysis_and_threshold_policy.ipynb", cells)


def main() -> None:
    generate_01_data_understanding()
    generate_02_univariate_eda()
    generate_03_bivariate_eda()
    generate_04_hidden_insight()
    generate_05_feature_engineering()
    generate_06_logistic()
    generate_07_advanced()
    generate_08_explainability()
    generate_09_error_threshold()

    print("\nDone generating notebooks.")


if __name__ == "__main__":
    main()
