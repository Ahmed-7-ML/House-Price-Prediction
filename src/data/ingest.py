# Command -> uv run python -m src.data.ingest
"""
    Data ingestion: load California Housing dataset and persist as CSV/Parquet.
"""

# ---> Imports
from sklearn.datasets import fetch_california_housing
import pandas as pd
from pathlib import Path
from rich.console import Console        # Rich text and beautiful formatting in the terminal.
from src.config.settings import get_settings

# ---> Define the Console Interface & Settings
console = Console()
settings = get_settings()

# ---> Ingest and Load the Raw Data
def ingest_raw_data(force: bool = False) -> pd.DataFrame:
    """
        Fetch California Housing and save to data/raw path.
    """
    raw_path = settings.raw_data_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and not force:
        console.print(f"[green]Raw Data already exists at {raw_path}[/green]")
        return pd.read_csv(raw_path)     # DataFrame
    
    console.print("[cyan]Fetching California Housing Dataset from Scikit-Learn...[/cyan]")
    df = fetch_california_housing(as_frame=True)
    df = df.frame  # Includes Target(MedHouseVal)
    df.to_csv(raw_path, index=False)
    console.print(f"[green]Saved {len(df)} rows to {raw_path}[/green]")
    return df

def load_raw_data() -> pd.DataFrame:
    if not settings.raw_data_path.exists():
        return ingest_raw_data()
    return pd.read_csv(settings.raw_data_path)

# Run the Data Pipeline
if __name__ == "__main__":
    ingest_raw_data(force=False)
