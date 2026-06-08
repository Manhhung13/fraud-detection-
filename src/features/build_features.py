from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

import pandas as pd

from src.features.amount_features import create_amount_features
from src.features.behavior_features import (
    BehaviorThresholds,
    create_behavior_features,
    fit_behavior_thresholds,
    thresholds_to_dict,
)
from src.features.interaction_features import create_interaction_features


TARGET_COL = "fraud"


BASE_NUMERIC_FEATURES = [
    # Không đưa amount raw vào model chính.
    # amount trong dataset không lệch đáng kể, nhưng tín hiệu amount tuyệt đối
    # không mạnh bằng các feature anomaly như amount_to_mean_30d, amount_z_30d.

    # Không đưa mean_amount_30d raw vì lệch phải mạnh, dùng log_mean_amount_30d.
    # Không đưa std_amount_30d nếu đã dùng amount_z_30d để tránh trùng ý nghĩa.
    # Không đưa max_amount_30d raw nếu dùng high_max_amount_30d_flag và amount_to_max feature.

    "night_ratio_30d",
    "mcc_entropy_30d",
    "distinct_merchants_7d",
    "distinct_countries_30d",
    "device_diversity_30d",
    "decline_rate_30d",
    "chargebacks_365d",
    "credit_util_today",

    # Dataset của bạn dùng spending_trend.
    # Nếu có spending_trend_30d thì code tự lọc, nhưng nên ưu tiên spending_trend.
    "spending_trend",

    "ip_score",
    "txn_count_7d",
    "txn_count_30d",
    "txn_count_ratio_7d_30d",
    "hour",
]


ENGINEERED_NUMERIC_FEATURES = [
    # mean_amount_30d lệch phải mạnh, nên dùng log thay raw.
    "log_mean_amount_30d",

    # amount_to_mean_30d không lệch mạnh, giữ raw ratio.
    "amount_to_mean_30d",

    # amount_to_max_30d lệch phải rất mạnh, dùng log version thay raw ratio.
    "log_amount_to_max_30d",

    # Chuẩn hóa amount theo lịch sử user.
    # Không dùng amount_to_std_30d vì gần trùng ý nghĩa với amount_z_30d.
    "amount_z_30d",

    # Time anomaly từ insight.
    "night_unusual_score",
]

BASE_CATEGORICAL_FEATURES = [
    "currency",
    "payment_channel",
    "merchant_country",
    "mcc",
    "card_entry_mode",
    "auth_result",
    "pin_verif_method",
    "recurring_flag",
    "auth_characteristics",
    "message_type",
    "term_location",
    "day_name",
    "time_period",
    #"ip_score_bin",
]


ENGINEERED_CATEGORICAL_FEATURES = [
    "night_ratio_group",
    "night_unusual_group",
    "amount_to_max_group",
    "ip_score_risk_group",
    "decline_rate_risk_group",
    "credit_util_risk_group",
    "txn_count_7d_group",
    "txn_velocity_group",
]


BINARY_FEATURES = [

    # Behavior flags từ insight
    "very_low_mcc_entropy_flag",
    "low_mcc_entropy_flag",
    "low_night_ratio_flag",
    "high_max_amount_30d_flag",
    "low_spending_trend_flag",
    "low_country_diversity_flag",
    "night_transaction_flag",
    "card_not_present_flag",
    "cross_border_flag",
    "tokenised_flag",
]


INTERACTION_FEATURES = [
    "low_mcc_entropy_x_low_night_ratio",
    "low_mcc_entropy_x_low_night_ratio_x_card_not_present",
    "low_mcc_entropy_x_high_max_amount_30d",
    "low_mcc_entropy_x_low_spending_trend",
    "low_night_ratio_x_high_max_amount_30d",
    "low_night_ratio_x_night_transaction",
    "low_mcc_entropy_x_card_not_present",
    "cross_border_x_low_country_diversity",
]


DROP_COLUMNS = [
    "ip_risk",
    "txn_counts",
    "ip_address",
    "transaction_datetime",
]


@dataclass
class FeatureMetadata:
    target_col: str
    numeric_features: list[str]
    categorical_features: list[str]
    binary_features: list[str]
    interaction_features: list[str]
    all_model_features: list[str]
    dropped_columns: list[str]
    thresholds: dict


class FraudFeatureBuilder:
    """
    Feature builder cho fraud detection.

    Nguyên tắc:
    - fit threshold trên train
    - transform train/validation/test bằng cùng threshold
    - không fit scaler/encoder ở đây
    """

    def __init__(self, target_col: str = TARGET_COL):
        self.target_col = target_col
        self.thresholds: BehaviorThresholds | None = None
        self.metadata: FeatureMetadata | None = None

    def fit(self, train_df: pd.DataFrame) -> "FraudFeatureBuilder":
        self.thresholds = fit_behavior_thresholds(train_df)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.thresholds is None:
            raise RuntimeError("Bạn cần gọi fit(train_df) trước khi transform().")

        out = df.copy()

        out = create_amount_features(out)
        out = create_behavior_features(out, thresholds=self.thresholds)
        out = create_interaction_features(out)

        out = self._normalize_model_columns(out)

        return out

    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        self.fit(train_df)
        return self.transform(train_df)

    def _normalize_model_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ép kiểu nhẹ cho các nhóm feature để chuẩn bị cho model pipeline.
        """
        df = df.copy()

        numeric_cols = self.get_existing_numeric_features(df)
        binary_cols = self.get_existing_binary_features(df)
        interaction_cols = self.get_existing_interaction_features(df)
        categorical_cols = self.get_existing_categorical_features(df)

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in binary_cols + interaction_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        for col in categorical_cols:
            df[col] = df[col].astype("object").where(df[col].notna(), "missing")

        return df

    def get_existing_numeric_features(self, df: pd.DataFrame) -> list[str]:
        candidates = BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
        return [col for col in candidates if col in df.columns]

    def get_existing_categorical_features(self, df: pd.DataFrame) -> list[str]:
        candidates = BASE_CATEGORICAL_FEATURES + ENGINEERED_CATEGORICAL_FEATURES
        return [col for col in candidates if col in df.columns]

    def get_existing_binary_features(self, df: pd.DataFrame) -> list[str]:
        return [col for col in BINARY_FEATURES if col in df.columns]

    def get_existing_interaction_features(self, df: pd.DataFrame) -> list[str]:
        return [col for col in INTERACTION_FEATURES if col in df.columns]

    def build_metadata(self, df: pd.DataFrame) -> FeatureMetadata:
        numeric_features = self.get_existing_numeric_features(df)
        categorical_features = self.get_existing_categorical_features(df)
        binary_features = self.get_existing_binary_features(df)
        interaction_features = self.get_existing_interaction_features(df)

        all_model_features = (
            numeric_features
            + categorical_features
            + binary_features
            + interaction_features
        )

        thresholds = (
            thresholds_to_dict(self.thresholds)
            if self.thresholds is not None
            else {}
        )

        self.metadata = FeatureMetadata(
            target_col=self.target_col,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            binary_features=binary_features,
            interaction_features=interaction_features,
            all_model_features=all_model_features,
            dropped_columns=[col for col in DROP_COLUMNS if col in df.columns],
            thresholds=thresholds,
        )

        return self.metadata

    def save_metadata(self, path: str | Path) -> None:
        if self.metadata is None:
            raise RuntimeError("Bạn cần gọi build_metadata(df) trước khi save_metadata().")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self.metadata), f, ensure_ascii=False, indent=2)


def select_model_dataframe(
    df: pd.DataFrame,
    metadata: FeatureMetadata,
) -> pd.DataFrame:
    """
    Lấy dataframe chỉ gồm model features + target.
    """
    keep_cols = metadata.all_model_features.copy()

    if metadata.target_col in df.columns:
        keep_cols.append(metadata.target_col)

    keep_cols = [col for col in keep_cols if col in df.columns]

    return df[keep_cols].copy()


def create_feature_report(
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    test_features: pd.DataFrame,
    metadata: FeatureMetadata,
) -> pd.DataFrame:
    rows = []

    for split_name, df in [
        ("train", train_features),
        ("validation", validation_features),
        ("test", test_features),
    ]:
        rows.append(
            {
                "split": split_name,
                "rows": len(df),
                "columns": df.shape[1],
                "numeric_features": len(metadata.numeric_features),
                "categorical_features": len(metadata.categorical_features),
                "binary_features": len(metadata.binary_features),
                "interaction_features": len(metadata.interaction_features),
                "all_model_features": len(metadata.all_model_features),
                "target_col": metadata.target_col,
                "fraud_count": int(df[metadata.target_col].sum())
                if metadata.target_col in df.columns
                else None,
                "fraud_rate": float(df[metadata.target_col].mean())
                if metadata.target_col in df.columns
                else None,
            }
        )

    return pd.DataFrame(rows)


def generate_feature_engineering_markdown_report(
    output_path: str | Path,
    feature_report: pd.DataFrame,
    metadata: FeatureMetadata,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []

    lines.append("# Fraud Detection - Feature Engineering Report\n")

    lines.append("## 1. Tổng quan output\n")
    lines.append(feature_report.to_markdown(index=False))

    lines.append("\n## 2. Numeric features\n")
    lines.append("\n".join([f"- `{col}`" for col in metadata.numeric_features]))

    lines.append("\n## 3. Categorical features\n")
    lines.append("\n".join([f"- `{col}`" for col in metadata.categorical_features]))

    lines.append("\n## 4. Binary behavior features\n")
    lines.append("\n".join([f"- `{col}`" for col in metadata.binary_features]))

    lines.append("\n## 5. Interaction features\n")
    lines.append("\n".join([f"- `{col}`" for col in metadata.interaction_features]))

    lines.append("\n## 6. Thresholds fit từ train\n")
    for key, value in metadata.thresholds.items():
        lines.append(f"- `{key}`: `{value}`")

    lines.append("\n## 7. Ghi chú leakage\n")
    lines.append(
        """
Các threshold trong file này được fit từ train và apply sang validation/test.
Giai đoạn này không fit scaler hoặc encoder. Việc scale numeric và one-hot categorical
sẽ được thực hiện trong model pipeline ở giai đoạn 5.
"""
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")