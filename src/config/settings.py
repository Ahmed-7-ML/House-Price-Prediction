"""
Central configuration using YAML + pydantic-settings.
"""
# ---> Imports
from functools import lru_cache
from pathlib import Path
from typing import Any
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---> Root Directory
ROOT_DIR = Path(__file__).resolve().parents[2]

# print(ROOT_DIR) --> E:\DataScience Bootcamp\Different Platforms\Flyrank.AI

# ---> Configuration Class
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HP_",
        extra="ignore"
    )
    
    # Project
    project_name : str = "house-price-prediction"
    seed: int = 42
    
    # Paths
    root_dir : Path = ROOT_DIR
    raw_data_path: Path = ROOT_DIR / "data" / "raw" / "california_housing.csv"
    processed_data_path: Path = ROOT_DIR / "data" / "processed" / "housing_processed.parquet"
    features_data_path: Path = ROOT_DIR / "data" / "features" / "housing_features.parquet"
    feature_repo_path: Path = ROOT_DIR / "feature_repo"
    models_dir: Path = ROOT_DIR / "models" / "artifacts"
    eda_output_dir: Path = ROOT_DIR / "notebooks" / "eda_figures"
    mlflow_tracking_uri: str = "sqlite:///./tmp/mlflow_house_price.db"

    # Data Split
    test_size : float = 0.2
    validation_size : float = 0.1
    
    # Model
    cv_folds : int = 5
    best_model_metric : str = "r2"
    model_name: str = "house_price_model"
    experiment_name: str = "house_price_prediction"
    models_to_try: list[str] = [
        "linear_regression",
        "ridge",
        "random_forest",
        "gradient_boosting",
    ]

    # Serving
    api_host : str = "0.0.0.0"
    api_port : int = 8000

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "Settings":
        path = path or ROOT_DIR / "configs" / "config.yaml"
        if not path.exists():
            return cls()

        with open(path, encoding="utf-8") as f:
            cfg: dict[str, Any] = yaml.safe_load(f) or {}

        project = cfg.get("project", {})
        data = cfg.get("data", {})
        model = cfg.get("model", {})
        serving = cfg.get("serving", {})

        tracking_uri = model.get("tracking_uri")
        if not tracking_uri:
            tracking_uri = f"sqlite:///{(ROOT_DIR / 'tmp' / 'mlflow_house_price.db').as_posix()}"
        elif tracking_uri.startswith("sqlite:///tmp/"):
            db_name = tracking_uri.split("/")[-1]
            tracking_uri = f"sqlite:///{(ROOT_DIR / 'tmp' / db_name).as_posix()}"

        return cls(
            project_name=project.get("name", "house-price-prediction"),
            seed=project.get("seed", 42),
            test_size=data.get("test_size", 0.2),
            validation_size=data.get("validation_size", 0.1),
            experiment_name=model.get("experiment_name", "house_price_prediction"),
            cv_folds=model.get("cv_folds", 5),
            best_model_metric=model.get("best_model_metric", "r2"),
            models_to_try=model.get(
                "models_to_try",
                ["linear_regression", "ridge", "random_forest", "gradient_boosting"],
            ),
            api_host=serving.get("host", "0.0.0.0"),
            api_port=serving.get("port", 8000),
            model_name=serving.get("model_name", "house_price_model"),
            mlflow_tracking_uri=tracking_uri,
        )

# ---> Get the Settings (Read only once and Save it)
@lru_cache
def get_settings() -> Settings:
    return Settings.from_yaml()
