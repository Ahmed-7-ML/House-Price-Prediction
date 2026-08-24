# Command -> uv run python -m src.data.preprocess
"""
    ETL preprocessing: (Extract -> Transform -> Load)
        1. clean
        2. split
        3. persist processed data.
"""

# ---> Imports
from rich.console import Console
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.config.settings import get_settings
from src.data.ingest import load_raw_data

# ---> Define the Console & Settings
console = Console()
settings = get_settings()

# ---> Data Cleaning
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
        Checking :
            - Missing Values
            - Duplicates
            - Filteration
    """
    df = df.copy()
    
    # 1. Remove Duplicates
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)
    
    if after < before:
        console.print(f"[yellow]Dropped {before - len(df)} duplicate rows[/yellow]")
    
    # 2. Remove Outliers
    q99 = df['AveOccup'].quantile(0.99)
    df.loc[df['AveOccup'] > q99, "AveOccup"] = q99
    
    # 3. Fill Missing Values
    if df.isnull().any().any():
        console.print("[yellow]Nulls detected — filling numeric with median[/yellow]")
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
    
    return df


# ---> Feature Engineering
def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Domain-Inspired Engineered Features
    """
    df = df.copy()
    
    # 1. Name these with useful names (descriptive)
    df["RoomsPerHousehold"] = df["AveRooms"]
    df["PopulationPerHousehold"] = df["AveOccup"]
    # 2. Calc. from another feature and fill missing.
    df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"].replace(0, np.nan)         # عشان لو فيه صفر فى المقام ميقسمش عليه
    df["BedroomsPerRoom"] = df["BedroomsPerRoom"].fillna(df["BedroomsPerRoom"].median())

    return df

# ---> Save new, modified DataFrame
# Private/Internal -> Used only inside this script
def _save_df(df:pd.DataFrame, path:Path) -> None:
    """
        Save DataFrame robustly 
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        if path.with_suffix(".parquet").stat().st_size > 0:
            console.print(f"[yellow]Saved as parquet [/yellow]")
            return
    except :
        pass
    
    # Fallback to Pickle
    pkl = path.with_suffix(".pkl")
    df.to_pickle(pkl)    
    console.print(f"[yellow]Saved as pickle: {pkl}[/yellow]")

# ---> Load the Dataframe for further usage
# Private/Internal -> Used only inside this script
def _load_df(path: Path) -> pd.DataFrame:
    path = Path(path)
    for suffix in [".parquet", ".pkl", ".csv"]:
        p = path.with_suffix(suffix)
        if p.exists() and p.stat().st_size > 0:
            if suffix == ".parquet":
                return pd.read_parquet(p)
            elif suffix == ".pkl":
                return pd.read_pickle(p)
            elif suffix == ".csv":
                return pd.read_csv(p)
    raise FileNotFoundError(f"No usable data file near {path}")

# ---> Split dataframe into Training/Testing Splits
def preprocess_split(
    test_size: float | None = None,
    random_state: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
        Full ETL: load → clean → engineer → split → save.
    """
    test_size = test_size or settings.test_size
    random_state = random_state or settings.seed
    
    # 1. Load the Raw Data
    df = load_raw_data()
    
    # 2. Clean it
    df = clean_data(df)
    
    # 3. Engineer Features
    df = create_engineered_features(df)
    
    # 4. Define Features & Target
    target_col = "MedHouseVal"
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]
    
    # 5. Split df into Train/Test Splits
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
    
    # 6. Persist
    _save_df(df, settings.processed_data_path)
    _save_df(df, settings.features_data_path)
    # Also always write CSV for maximum compatibility
    settings.processed_data_path.with_suffix(".csv").write_text("")  # ensure parent
    df.to_csv(settings.processed_data_path.with_suffix(".csv"), index=False)
    df.to_csv(settings.features_data_path.with_suffix(".csv"), index=False)
    
    console.print(f"[green]Train: {len(X_train)} | Test: {len(X_test)} | Features: {list(X.columns)}[/green]")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_split()
