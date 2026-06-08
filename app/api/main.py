from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI

# app/api/main.py -> parents[2] = fraud-detection-project
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from app.api.predict import router as predict_router
from app.api.schemas import HealthResponse
from src.inference.predictor import load_final_predictor


app = FastAPI(
    title="Fraud Detection API",
    version="1.0.0",
    description="API demo cho mô hình phát hiện gian lận giao dịch tài chính.",
)

predictor = load_final_predictor(PROJECT_ROOT)

# Lưu predictor vào app.state để các router dùng chung.
app.state.predictor = predictor

app.include_router(predict_router)


@app.get("/", response_model=HealthResponse)
def root():
    return {
        "status": "ok",
        "model": predictor.metadata.get("final_experiment_name"),
        "threshold": predictor.threshold,
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "model": predictor.metadata.get("final_experiment_name"),
        "threshold": predictor.threshold,
    }