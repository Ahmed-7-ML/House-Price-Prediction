"""
    Standalone evaluation utilities.
    Evaluation of the Best Model
"""

# ---> Imports
import joblib
import numpy as np
from rich.console import Console
from sklearn.metrics import (
    mean_absolute_error as MAE,
    mean_squared_error  as MSE,
    r2_score            as R2,
)
from src.data.preprocess import preprocess_split
from src.models.train import evaluate
from src.config.settings import get_settings

# ---> Define the Console & Settings
settings = get_settings()
console = Console()

# ---> Load the Best Model
def load_best_model():
    path = settings.models_dir / "best_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No model found at {path}. Train first.")
    return joblib.load(path)

# ---> Evaluate the Best Model
def evaluate_best_model() -> dict[str, float]:
    model = load_best_model()
    _, X_test, _, y_test = preprocess_split()
    features = joblib.load(settings.models_dir / "feature_names.joblib")
    X_test = X_test[[c for c in features if c in X_test.columns]]
    y_pred = model.predict(X_test)
    metrics = evaluate(y_test, y_pred)
    console.print(f"[green]Evaluation metrics: {metrics}[/green]")
    return metrics
