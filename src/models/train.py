# Command -> uv run python -m src.models.train
"""
    Model training with MLflow experiment tracking.
"""
"""
التحسينات :
عاوز اعمل Hyperparameter Tuning
عاوز اضيف التصدير الى ONNX
عاوز استخدم ال Feature store
"""

# ---> Imports
from typing import Any
import numpy as np
import os
from rich.console import Console

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import mlflow
import mlflow.sklearn

import joblib

from src.config.settings import get_settings
from src.data.preprocess import preprocess_split
from src.features.pipeline import build_preprocessor, NUMERICAL_FEATURES

# ---> Define Console, Settings
console = Console()
settings = get_settings()

# ---> Get Model Portfolio
def get_model_zoo(
    random_state: int = 42
) -> dict[str, Any]:
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0, random_state=random_state),
        "lasso": Lasso(alpha=1.0, random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            random_state=random_state
        ),
        "support_vector_machine": SVR(
            kernel='poly',
            degree=3
        ),
        "decision_tree": DecisionTreeRegressor(
            criterion="absolute_error",
            max_depth=12,
            min_samples_leaf=5,
            random_state=random_state
        )
    }

# ---> Evaluation Metrics
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae" : float(mean_absolute_error(y_true, y_pred)),
        "r2"  : float(r2_score(y_true, y_pred)),
    }

# ---> Start Training & Experiment Tracking(MLFlow)
def train_log(
    model_name: str | None = None,
    use_feature_store: bool = False
) -> dict[str, Any]:
    """
        Train candidate models, log everything to MLflow, register the best one.
        Returns metrics of the best model.
    """
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)
    
    X_train, X_test, y_train, y_test = preprocess_split()
    
    # Ensure we only use columns the preprocessor expects
    available = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
    X_train = X_train[available]
    X_test = X_test[available]
    
    models = get_model_zoo(settings.seed)
    if model_name:
        models = {model_name: models[model_name]}
    
    best_score = -np.inf
    best_name = None
    best_pipeline = None
    best_run_id = None
    best_metrics: dict[str, float] = {}
    
    for name, estimator in models.items():
        console.print(f"\n[bold cyan]Training {name}...[/bold cyan]")
        preprocessor = build_preprocessor(use_poly=True)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])
        
        with mlflow.start_run(run_name=name) as run:
            # Cross-Validation
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv = settings.cv_folds,
                scoring="r2",
                n_jobs=-1
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            metrics = evaluate(y_test.values, y_pred)
            metrics['cv_r2_mean'] = float(cv_scores.mean())
            metrics['cv_r2_std']  = float(cv_scores.std())
            
            # Log Params & Metrics
            mlflow.log_params({
                "model_type": name,
                "n_features": len(available),
                "train_size": len(X_train),
                "test_size" : len(X_test)
            })
            mlflow.log_metrics(metrics=metrics)
            
            # Log Model
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path = "model",
                serialization_format = "pickle",
                registered_model_name=settings.model_name if name == "gradient_boosting" else None
            )
            
            console.print(
                f"  R²={metrics['r2']:.4f} | RMSE={metrics['rmse']:.4f} | "
                f"MAE={metrics['mae']:.4f} | CV R²={metrics['cv_r2_mean']:.4f}±{metrics['cv_r2_std']:.4f}"
            )
            
            if metrics['r2'] > best_score:
                best_score = metrics['r2']
                best_name = name
                best_pipeline = pipeline
                best_metrics = metrics
                best_run_id = run.info.run_id

    # Persist best model locally for FastAPI Serving
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = settings.models_dir / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)
    console.print(f"\n[bold green]Best model: {best_name} (R²={best_score:.4f})[/bold green]")
    console.print(f"[green]Saved to {model_path}[/green]")
    
    # Also -> Save Feature List for Inference
    joblib.dump(available, settings.models_dir / "feature_names.joblib")
    
    return {
        "best_model": best_name,
        "metrics": best_metrics,
        "model_path": str(model_path),
        "run_id": best_run_id
    }


if __name__ == "__main__":
    train_log()
