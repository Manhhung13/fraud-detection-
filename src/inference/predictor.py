from __future__ import annotations

from pathlib import Path
import json

import joblib
import pandas as pd


class FraudPredictor:
    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy model: {self.model_path}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Không tìm thấy metadata: {self.metadata_path}")

        self.model = joblib.load(self.model_path)

        with self.metadata_path.open("r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        self.threshold = float(self.metadata["selected_threshold"])

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        proba = self.model.predict_proba(df)[:, 1]
        return pd.Series(proba, index=df.index, name="fraud_probability")

    def assign_decision(self, probability: float) -> dict:
        if probability < 0.30:
            return {
                "risk_level": "LOW",
                "decision": "APPROVE",
                "reason": "Fraud probability thấp hơn 0.30.",
            }

        if probability < self.threshold:
            return {
                "risk_level": "MEDIUM",
                "decision": "MANUAL_REVIEW",
                "reason": f"Fraud probability nằm giữa 0.30 và threshold {self.threshold:.2f}.",
            }

        return {
            "risk_level": "HIGH",
            "decision": "FRAUD_ALERT",
            "reason": f"Fraud probability vượt threshold {self.threshold:.2f}.",
        }

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        proba = self.predict_proba(df)

        rows = []

        for idx, p in proba.items():
            decision = self.assign_decision(float(p))
            rows.append(
                {
                    "fraud_probability": float(p),
                    "prediction": int(p >= self.threshold),
                    "threshold": self.threshold,
                    **decision,
                }
            )

        return pd.DataFrame(rows, index=df.index)


def load_final_predictor(project_root: str | Path) -> FraudPredictor:
    project_root = Path(project_root)

    return FraudPredictor(
        model_path=project_root / "models" / "final_model" / "fraud_catboost_final.joblib",
        metadata_path=project_root / "models" / "final_model" / "final_model_metadata.json",
    )