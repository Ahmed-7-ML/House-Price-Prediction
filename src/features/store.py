# Command -> uv run python -m src.features.store
"""
    Feast feature store integration helpers.
"""
# ---> Imports
from feast import FeatureStore
from pathlib import Path
import pandas as pd
from rich.console import Console
from datetime import datetime, timedelta
from src.config.settings import get_settings


# ---> Define Console & Settings
console = Console()
settings = get_settings()


# ---> Load the Features
def _load_features_df() -> pd.DataFrame:
    base = settings.features_data_path
    for suffix in [".pkl", ".csv", ".parquet"]:
        p = base.with_suffix(suffix=suffix)
        if p.exists() and p.stat().st_size > 0:
            if suffix == ".csv":
                return pd.read_csv(p)
            elif suffix == ".pkl":
                return pd.read_pickle(p)
            elif suffix == ".parquet":
                return pd.read_parquet(p)
    raise FileNotFoundError(f"Features file not found near {base}. Run preprocessing first.")

# ---> Prepare Feature Store
def prepare_feature_store_data() -> None:
    """
        Prepare parquet/csv files that Feast will use as offline sources.
        Adds entity_id and event_timestamp required by Feast.
    """
    df = _load_features_df().reset_index(drop=True)
    df['entity_id'] = df.index.astype(int)
    
    base = datetime(2025, 1, 1)
    df['event_timestamp'] = [base + timedelta(hours=i) for i in range(len(df))]

    repo_data = settings.feature_repo_path / "data"
    repo_data.mkdir(parents=True, exist_ok=True)
    
    feature_cols = [
        "entity_id",
        "event_timestamp",
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "RoomsPerHousehold",
        "BedroomsPerRoom",
        "PopulationPerHousehold",
    ]
    out_feat = df[feature_cols]
    out_feat.to_csv(repo_data / "housing_features.csv", index=False)
    try:
        out_feat.to_parquet(repo_data / "housing_features.parquet", index=False)
    except :
        pass

    target_cols = ["entity_id", "event_timestamp", "MedHouseVal"]
    out_tgt = df[target_cols]
    out_tgt.to_csv(repo_data / "housing_target.csv", index=False)
    try:
        out_tgt.to_parquet(repo_data / "housing_target.parquet", index=False)
    except :
        pass

    console.print(f"[green]Feature store data prepared in {repo_data}[/green]")


# ---> Get the Training Features from Store
def get_training_features_from_store() -> pd.DataFrame:
    """
        Retrieve historical features via Feast (or fallback to local files).
    """
    try:
        store = FeatureStore(
            repo_path=str(settings.feature_repo_path)
        )
        entity_df = pd.read_csv(
            settings.feature_repo_path / "data" / "housing_target.csv"
        )[["entity_id", "event_timestamp"]]
        
        entity_df["event_timestamp"] = pd.to_datetime(entity_df["event_timestamp"])

        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "housing_features:MedInc",
                "housing_features:HouseAge",
                "housing_features:AveRooms",
                "housing_features:AveBedrms",
                "housing_features:Population",
                "housing_features:AveOccup",
                "housing_features:Latitude",
                "housing_features:Longitude",
                "housing_features:RoomsPerHousehold",
                "housing_features:BedroomsPerRoom",
                "housing_features:PopulationPerHousehold",
            ]
        ).to_df()
        
        target = pd.read_csv(settings.feature_repo_path / "data" / "housing_target.csv")
        training_df = training_df.merge(target[["entity_id", "MedHouseVal"]], on="entity_id", how="left")
        console.print("[green]Retrieved features via Feast Feature Store[/green]")
        return training_df
    
    except Exception as e:
        console.print(f"[yellow]Feast not ready ({e}). Falling back to local data.[/yellow]")
        return _load_features_df()
