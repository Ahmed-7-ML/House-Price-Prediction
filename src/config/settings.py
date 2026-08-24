"""
    Central Configuration using YAML + Pydantic Settings
"""
# ---> Imports
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any
from pathlib import Path
from functools import lru_cache

ROOT_DIR = Path(__file__).resolve().parents[3]

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
    raw_data_path: Path = ROOT_DIR / "House Price Prediction" / "data" / "raw" / "california_housing.csv"
    processed_data_path: Path = ROOT_DIR / "House Price Prediction" / "data" / "processed" / "housing_processed.parquet"
    features_data_path: Path = ROOT_DIR / "House Price Prediction" / "data" / "features" / "housing_features.parquet"
    feature_repo_path: Path = ROOT_DIR / "House Price Prediction" / "feature_repo"
    models_dir: Path = ROOT_DIR / "House Price Prediction" / "models"
    mlflow_tracking_uri: str = "sqlite:///tmp/mlflow_house_price.db"

    # Data Split
    test_size : float = 0.2
    validation_size : float = 0.1
    
    # Model
    cv_folds : int = 5
    best_model_metric : str = "r2"
    model_name: str = "house_price_model"
    experiment_name: str = "house_price_prediction"

    # Serving
    api_host : str = "0.0.0.0"
    api_port : int = 8000

    @classmethod
    def from_yaml(cls, path:Path | None = None) -> "Settings":
        path = path or ROOT_DIR / "configs" / "config.yaml"
        if not path.exists():
            return cls()
        
        with open(path) as f:
            cfg: dict[str, Any] = yaml.safe_load(f) or {}
        
        return cls(
            project_name = cfg.get("project", {}).get("name", "house-price-prediction"),
            seed = cfg.get("project", {}).get("seed", 42),
            test_size = cfg.get("data", {}).get("test_size", 0.2),
            validation_size = cfg.get("data", {}).get("validation_size", 0.1),
            experiment_name = cfg.get("model", {}).get("experiment_name", "house_price_prediction"),
            cv_folds = cfg.get("model", {}).get("cv_folds", 5),
            best_model_metric = cfg.get("model", {}).get("best_model_metric", "r2"),
            api_host = cfg.get("serving", {}).get("host", "0.0.0.0"),
            api_port = cfg.get("serving", {}).get("port", 8000),
            model_name = cfg.get("serving", {}).get("model_name", "house_price_model"),
            mlflow_tracking_uri = cfg.get("model", {}).get("tracking_uri", "sqlite:////tmp/mlflow_house_price.db")
        )

# ---> Get the Settings (Read only once and Save it)
@lru_cache
def get_settings() -> Settings:
    return Settings.from_yaml()
