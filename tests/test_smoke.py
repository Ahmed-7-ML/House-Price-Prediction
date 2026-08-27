"""
Smoke tests for the house price MLOps system.
"""
from pathlib import Path

import joblib
import pytest

from src.config.settings import get_settings
from src.data.ingest import load_raw_data
from src.data.preprocess import clean_data, create_engineered_features
from src.features.pipeline import NUMERICAL_FEATURES, build_preprocessor


def test_settings():
    s = get_settings()
    assert s.seed == 42
    assert s.project_name == "house-price-mlops"
    assert s.models_dir.name == "artifacts"


def test_ingest_and_preprocess():
    df = load_raw_data()
    assert len(df) > 20000

    df = create_engineered_features(clean_data(df))
    assert "BedroomsPerRoom" in df.columns
    assert "MedHouseVal" in df.columns
    assert df["BedroomsPerRoom"].notna().all()


def test_preprocessor_builds():
    preprocessor = build_preprocessor(use_poly=False, feature_names=NUMERICAL_FEATURES)
    assert preprocessor is not None


def test_model_exists():
    path = get_settings().models_dir / "best_model.joblib"
    if not path.exists():
        pytest.skip("Model not trained yet")
    model = joblib.load(path)
    assert model is not None
    features_path = get_settings().models_dir / "feature_names.joblib"
    assert features_path.exists()
    names = joblib.load(features_path)
    assert isinstance(names, list)
    assert Path(path).stat().st_size > 0
