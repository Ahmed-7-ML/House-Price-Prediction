# Command -> uv run python -m src.data.ingest
"""
Data ingestion: load California Housing dataset and persist as CSV.
"""

# ---> Imports
from sklearn.datasets import fetch_california_housing
import pandas as pd
from src.config.settings import get_settings
from rich.console import Console

# ---> Console and Settings
console = Console()
settings = get_settings()

# ---> Ingest Raw Data
def ingest_raw_data(force: bool = False) -> pd.DataFrame:
    """Fetch California Housing and save to data/raw path."""
    raw_path = settings.raw_data_path
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists() and not force:
        console.print(f"[green]Raw data already exists at {raw_path}[/green]")
        return pd.read_csv(raw_path)

    console.print("[cyan]Fetching California Housing dataset from scikit-learn...[/cyan]")
    bunch = fetch_california_housing(as_frame=True)
    df = bunch.frame
    df.to_csv(raw_path, index=False)
    console.print(f"[green]Saved {len(df)} rows to {raw_path}[/green]")
    return df

# ---> Load Raw Data
def load_raw_data() -> pd.DataFrame:
    if not settings.raw_data_path.exists():
        return ingest_raw_data()
    return pd.read_csv(settings.raw_data_path)

# Run the Data Pipeline
if __name__ == "__main__":
    ingest_raw_data(force=False)
