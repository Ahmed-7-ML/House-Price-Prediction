"""
FastAPI inference service for house price prediction.
"""
from contextlib import asynccontextmanager
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from rich.console import Console

from src.api.schemas import BatchPredictionResponse, HouseFeatures, PredictionResponse
from src.config.settings import get_settings
from src.data.preprocess import create_engineered_features

settings = get_settings()
console = Console()

_model: Any = None
_feature_names: list[str] = []
_model_path = settings.models_dir / "best_model.joblib"
_feature_path = settings.models_dir / "feature_names.joblib"


def _load_artifacts() -> None:
    global _model, _feature_names
    if not _model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {_model_path}. Run training first: house-price train"
        )
    _model = joblib.load(_model_path)
    if _feature_path.exists():
        _feature_names = joblib.load(_feature_path)
    else:
        _feature_names = []
    console.print(f"[green]Loaded model from {_model_path}[/green]")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _model, _feature_names
    try:
        _load_artifacts()
    except FileNotFoundError as exc:
        console.print(f"[yellow]Warning: {exc}[/yellow]")
    yield
    console.print("[cyan]Shutting down...[/cyan]")
    _model = None
    _feature_names = []


app = FastAPI(
    title="House Price Prediction API",
    description="End-to-end MLOps reference – California Housing price prediction",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _predict_frame(features: HouseFeatures) -> float:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    df = create_engineered_features(pd.DataFrame([features.model_dump()]))
    if _feature_names:
        for col in _feature_names:
            if col not in df.columns:
                df[col] = 0.0
        df = df[_feature_names]

    try:
        return float(_model.predict(df)[0])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc


@app.get("/")
def root():
    return {
        "message": "House Price Prediction API",
        "docs": "/docs",
        "health": "/health",
        "prediction": "POST /predict",
        "batch_prediction": "POST /predict/batch",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if _model is not None else "model_not_loaded",
        "model_loaded": _model is not None,
        "model_name": settings.model_name,
        "model_path": str(_model_path),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    pred = _predict_frame(features)
    return PredictionResponse(predicted_price=round(pred, 4))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(examples: list[HouseFeatures]):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    results = [PredictionResponse(predicted_price=round(_predict_frame(item), 4)) for item in examples]
    return BatchPredictionResponse(predictions=results)
