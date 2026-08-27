"""
ETL preprocessing: clean, engineer features, split, persist processed data.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from sklearn.model_selection import train_test_split

from src.config.settings import get_settings
from src.data.ingest import load_raw_data

console = Console()
settings = get_settings()


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates, cap occupancy outliers, fill numeric nulls."""
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates()
    if len(df) < before:
        console.print(f"[yellow]Dropped {before - len(df)} duplicate rows[/yellow]")

    q99 = df["AveOccup"].quantile(0.99)
    df.loc[df["AveOccup"] > q99, "AveOccup"] = q99

    if df.isnull().any().any():
        console.print("[yellow]Nulls detected — filling numeric columns with median[/yellow]")
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Domain-inspired ratios. AveRooms/AveOccup already are per-household averages."""
    df = df.copy()
    df["RoomsPerHousehold"] = df["AveRooms"]
    df["PopulationPerHousehold"] = df["AveOccup"]
    df["BedroomsPerRoom"] = df["AveBedrms"] / df["AveRooms"].replace(0, np.nan)
    median_ratio = df["BedroomsPerRoom"].median()
    df["BedroomsPerRoom"] = df["BedroomsPerRoom"].fillna(0.0 if pd.isna(median_ratio) else median_ratio)
    return df


def _save_df(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        console.print(f"[green]Saved parquet: {parquet_path}[/green]")
        return
    except Exception as exc:
        console.print(f"[yellow]Parquet save failed ({exc}); falling back to pickle[/yellow]")

    pkl = path.with_suffix(".pkl")
    df.to_pickle(pkl)
    console.print(f"[yellow]Saved as pickle: {pkl}[/yellow]")


def load_processed_df(path: Path | None = None) -> pd.DataFrame:
    path = Path(path or settings.processed_data_path)
    for suffix in (".parquet", ".pkl", ".csv"):
        candidate = path.with_suffix(suffix)
        if candidate.exists() and candidate.stat().st_size > 0:
            if suffix == ".parquet":
                return pd.read_parquet(candidate)
            if suffix == ".pkl":
                return pd.read_pickle(candidate)
            return pd.read_csv(candidate)
    raise FileNotFoundError(f"No usable data file near {path}")


def preprocess_split(
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Full ETL: load → clean → engineer → split → save."""
    test_size = test_size if test_size is not None else settings.test_size
    random_state = random_state if random_state is not None else settings.seed

    df = create_engineered_features(clean_data(load_raw_data()))

    target_col = "MedHouseVal"
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    _save_df(df, settings.processed_data_path)
    _save_df(df, settings.features_data_path)

    settings.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
    settings.features_data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(settings.processed_data_path.with_suffix(".csv"), index=False)
    df.to_csv(settings.features_data_path.with_suffix(".csv"), index=False)

    console.print(
        f"[green]Train: {len(X_train)} | Test: {len(X_test)} | Features: {list(X.columns)}[/green]"
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    preprocess_split()
