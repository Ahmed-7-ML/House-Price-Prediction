"""
Model training with MLflow experiment tracking.
"""

# ---> Imports
from __future__ import annotations
from typing import Any
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.config.settings import get_settings
from src.data.preprocess import preprocess_split
from src.features.pipeline import NUMERICAL_FEATURES, build_preprocessor
from src.features.store import get_training_features_from_store

# ---> Define the Console & Settings
console = Console()
settings = get_settings()

# ---> To Get the Model Portfolio
def get_model_zoo(random_state: int = 42) -> dict[str, Any]:
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=1.0, random_state=random_state),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            random_state=random_state,
        ),
        "decision_tree": DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=5,
            random_state=random_state,
        ),
    }

# ---> Regression Model Evaluation
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

# ---> Prepare & Split Data into Train-Test
def _split_from_feature_store() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    df = get_training_features_from_store()
    if "MedHouseVal" not in df.columns:
        raise ValueError("Feature store data is missing MedHouseVal")
    feature_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    X = df[feature_cols]
    y = df["MedHouseVal"]
    return train_test_split(X, y, test_size=settings.test_size, random_state=settings.seed)

# ---> Log Params/Metrics/Model using MLFlow
def _log_sklearn_model(pipeline: Pipeline) -> None:
    trusted = ["numpy.dtype", "numpy._core.multiarray._reconstruct", "numpy.ndarray"]
    try:
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            registered_model_name=settings.model_name,
            skops_trusted_types=trusted,
            )
    except TypeError:
        # older mlflow that still uses artifact_path
        mlflow.sklearn.log_model(
            sk_model=pipeline, 
            artifact_path="model",
            registered_model_name=settings.model_name,
            skops_trusted_types=trusted,
            )

# ---> Start Model Training & Logging
def train_log(
    model_name: str | None = None,
    use_feature_store: bool = False,
    use_poly: bool = False,
) -> dict[str, Any]:
    """Train candidate models, log to MLflow, persist the best pipeline."""
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    (settings.root_dir / "tmp").mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    if use_feature_store:
        X_train, X_test, y_train, y_test = _split_from_feature_store()
    else:
        X_train, X_test, y_train, y_test = preprocess_split()

    available = [c for c in NUMERICAL_FEATURES if c in X_train.columns]
    X_train = X_train[available]
    X_test = X_test[available]

    zoo = get_model_zoo(settings.seed)
    if model_name:
        if model_name not in zoo:
            raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(zoo)}")
        models = {model_name: zoo[model_name]}
    else:
        selected = [name for name in settings.models_to_try if name in zoo]
        if not selected:
            selected = list(zoo)
        models = {name: zoo[name] for name in selected}

    best_score = -np.inf
    best_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_run_id: str | None = None
    best_metrics: dict[str, float] = {}

    for name, estimator in models.items():
        console.print(f"\n[bold cyan]Training {name}...[/bold cyan]")
        pipeline = Pipeline(
            [
                ("preprocessor", build_preprocessor(use_poly=use_poly, feature_names=available)),
                ("model", estimator),
            ]
        )

        with mlflow.start_run(run_name=name) as run:
            cv_scores = cross_val_score(
                pipeline,
                X_train,
                y_train,
                cv=settings.cv_folds,
                scoring="r2",
                n_jobs=-1,
            )
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            metrics = evaluate(np.asarray(y_test), y_pred)
            metrics["cv_r2_mean"] = float(cv_scores.mean())
            metrics["cv_r2_std"] = float(cv_scores.std())

            mlflow.log_params(
                {
                    "model_type": name,
                    "n_features": len(available),
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                    "use_poly": use_poly,
                }
            )
            mlflow.log_metrics(metrics)
            _log_sklearn_model(pipeline)

            console.print(
                f"  R²={metrics['r2']:.4f} | RMSE={metrics['rmse']:.4f} | "
                f"MAE={metrics['mae']:.4f} | "
                f"CV R²={metrics['cv_r2_mean']:.4f}±{metrics['cv_r2_std']:.4f}"
            )

            metric_key = settings.best_model_metric
            score = metrics.get(metric_key, metrics["r2"])
            if score > best_score:
                best_score = score
                best_name = name
                best_pipeline = pipeline
                best_metrics = metrics
                best_run_id = run.info.run_id

    if best_pipeline is None or best_name is None:
        raise RuntimeError("Training produced no models")

    model_path = settings.models_dir / "best_model.joblib"
    joblib.dump(best_pipeline, model_path)
    joblib.dump(available, settings.models_dir / "feature_names.joblib")

    try:
        mlflow.register_model(f"runs:/{best_run_id}/model", settings.model_name)
    except Exception as exc:  # noqa: BLE001  # registry may be unavailable
        console.print(f"[yellow]Model registry skipped: {exc}[/yellow]")

    console.print(f"\n[bold green]Best model: {best_name} (R²={best_metrics['r2']:.4f})[/bold green]")
    console.print(f"[green]Saved to {model_path}[/green]")

    return {
        "best_model": best_name,
        "metrics": best_metrics,
        "model_path": str(model_path),
        "run_id": best_run_id,
    }


if __name__ == "__main__":
    train_log()
