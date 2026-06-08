from __future__ import annotations

from pathlib import Path
import json
import shutil
from datetime import datetime

import pandas as pd


def read_one_row_csv(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"File rỗng: {path}")

    return df.iloc[0].to_dict()


def promote_final_model(
    project_root: str | Path,
    final_experiment_name: str = "cat_tune_depth7_l2_8_lr003_iter700",
) -> None:
    """
    Copy model tốt nhất từ tuning experiments sang models/final_model/.
    """
    project_root = Path(project_root)

    source_model_dir = (
        project_root
        / "models"
        / "catboost_tuning_experiments"
        / final_experiment_name
    )

    source_report_dir = (
        project_root
        / "data"
        / "reports"
        / "catboost_tuning"
        / final_experiment_name
    )

    final_dir = project_root / "models" / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)

    source_model_path = source_model_dir / "pipeline.joblib"
    source_metadata_path = source_model_dir / "experiment_metadata.json"
    source_test_metrics_path = source_report_dir / "metrics_test.csv"
    source_validation_metrics_path = source_report_dir / "metrics_validation_best_threshold.csv"
    source_feature_importance_path = source_report_dir / "feature_importance.csv"

    if not source_model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {source_model_path}")

    shutil.copy2(source_model_path, final_dir / "fraud_catboost_final.joblib")

    if source_metadata_path.exists():
        shutil.copy2(source_metadata_path, final_dir / "source_experiment_metadata.json")

    if source_feature_importance_path.exists():
        shutil.copy2(source_feature_importance_path, final_dir / "final_feature_importance.csv")

    test_metrics = read_one_row_csv(source_test_metrics_path)
    validation_metrics = read_one_row_csv(source_validation_metrics_path)

    final_metadata = {
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "final_experiment_name": final_experiment_name,
        "model_name": "catboost",
        "model_file": "fraud_catboost_final.joblib",
        "source_model_path": str(source_model_path),
        "selected_threshold": float(test_metrics["threshold"]),
        "business_threshold_source": "selected on validation, applied to test",
        "mcc_strategy": "use mcc with rare category grouping min_frequency=100",
        "drop_manual_interactions": True,
        "test_metrics": test_metrics,
        "validation_metrics": validation_metrics,
        "business_policy": {
            "low": {
                "condition": "fraud_probability < 0.30",
                "risk_level": "LOW",
                "decision": "APPROVE",
            },
            "medium": {
                "condition": "0.30 <= fraud_probability < selected_threshold",
                "risk_level": "MEDIUM",
                "decision": "MANUAL_REVIEW",
            },
            "high": {
                "condition": "fraud_probability >= selected_threshold",
                "risk_level": "HIGH",
                "decision": "FRAUD_ALERT",
            },
        },
    }

    with (final_dir / "final_model_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(final_metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved final model to: {final_dir / 'fraud_catboost_final.joblib'}")
    print(f"Saved final metadata to: {final_dir / 'final_model_metadata.json'}")