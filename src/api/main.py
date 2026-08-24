"""
    FastAPI inference service for house price prediction.
"""
# ---> Imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Any
import pandas as pd
import joblib
from rich.console import Console

from src.api.schemas import HouseFeatures, PredictionResponse
from src.config.settings import get_settings

# ---> Define Console & Settings
settings = get_settings()
console = Console()

# Global model handle
_model: Any = None
_feature_names: list[str] = []
model_path = settings.models_dir / "best_model.joblib"
feature_path = settings.models_dir / "feature_names.joblib"

# ---> Load the Artifacts
def _load_artifacts() -> None:
    """
        Load the trained model and feature names from disk into memory.

        This function is called once at application startup (via lifespan).
        Raises FileNotFoundError if the model file does not exist.
    """
    global _model, _feature_names
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first: house-price train")

    _model = joblib.load(model_path)
    _feature_names = joblib.load(feature_path)
    console.print(f"[green]Loaded model from {model_path}[/green]")


# ---> Define the Lifespan
# Before the API accept requests -> Load the Model & Features (_load_artifacts)
# Load the Model Once (not load it with each request) and make it ready in memory
# You can clean resources at the end
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
        Manage application startup and shutdown events.
            - On startup: load model artifacts into memory.
            - On shutdown: currently does nothing (can be used for cleanup).
    """
    # ---> Startup
    try:
        _load_artifacts()
    except FileNotFoundError as e:
        console.print(f"[yellow]Warning: {e}[/yellow]")
    yield
    
    # ---> Shutdown
    console.print("[cyan]Shutting down... cleaning up resources[/cyan]")
    _model = None
    _feature_names = None
    console.print("[green]Cleanup done[/green]")

# ---> Define FastAPI Application
app = FastAPI(
    title="House Price Prediction API",
    description="End-to-end MLOps reference – California Housing price prediction",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS = Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---> Define API Endpoints

# 1. Root
@app.get("/")
def root():
    """
        Root endpoint – returns basic API information and available routes.
        Useful as a quick check that the service is running.
    """
    return {
        "message": "House Price Prediction API",
        "swagger_docs": "/docs",
        "health": "/health",
        "prediction": "POST /predict"
    }

# 2. Health
@app.get("/health")
def health():
    """
        Health check endpoint.
        - Returns whether the model is successfully loaded.
        - Used by Docker, Kubernetes, load balancers, and monitoring tools.
    """
    return {
        "status": "ok" if _model is not None else "model_not_loaded",
        "model_loaded": _model is not None
    }

# 3. Single Prediction
# 1 Request -> 1 Response
@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures):
    """
        Predict the median house value for a single set of features.
        Steps:
            1. Check that the model is loaded.
            2. Convert input to dict and add engineered features.
            3. Align columns with the features the model was trained on.
            4. Run prediction and return the result.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train the model first."
        )
    
    # Build the Feature Vector with Engineered Features
    data = features.model_dump()
    data["RoomsPerHousehold"] = data["AveRooms"]
    data["BedroomsPerRoom"] = data["AveBedrms"] / data["AveRooms"] if data["AveRooms"] else 0.0
    data["PopulationPerHousehold"] = data["AveOccup"]
    
    # 1 Row = List of dict
    df = pd.DataFrame([data])
    
    # Align columns to what the model was trained on
    if _feature_names:
        for col in _feature_names:
            if col not in df.columns:
                df[col] = 0.0
        
        df = df[_feature_names]
    
    try:
        pred = float(_model.predict(df)[0])
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {e}"
        ) from e
    
    return PredictionResponse(
        predicted_price=round(pred, 4)
    )


# 4. Batch Prediction
# List of requests -> List of responses
@app.post("/predict/batch")
def predict_batch(examples: list[HouseFeatures]):
    """
        Predict house values for multiple examples at once (batch inference).
        Accepts a list of HouseFeatures and returns a list of predictions.
        Internally reuses the single `predict` function for each item.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    results = []
    for example in examples:
        # Call the /predict function
        results.append(predict(example))
    
    return results
