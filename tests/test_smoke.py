"""
    Smoke tests for the house price MLOps system.
"""
import sys
from pathlib import Path
import joblib
import pytest
from src.config.settings import get_settings
from src.data.ingest import load_raw_data
from src.data.preprocess import clean_data, create_engineered_features

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def test_settings():
    s = get_settings()
    assert s.seed == 42
    assert s.project_name == "house-price-mlops"


def test_ingest_and_preprocess():
    df = load_raw_data()
    assert len(df) > 20000
    
    df = clean_data(df)
    df = create_engineered_features(df)
    assert "BedroomsPerRoom" in df.columns
    assert "MedHouseVal" in df.columns


def test_model_exists():
    path = get_settings().models_dir / "best_model.joblib"
    # May not exist in CI before train; skip if missing
    if not path.exists():
        pytest.skip("Model not trained yet")
    model = joblib.load(path)
    assert model is not None
