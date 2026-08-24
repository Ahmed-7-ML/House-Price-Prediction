# Command -> uv run python -m src.data.eda
"""
    Explorratory Data Analysis with Matplotlib + Seaborn
"""

# ---> Imports
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from rich.console import Console
from src.config.settings import get_settings
from src.data.ingest import load_raw_data

# ---> Define Console & Settings
console = Console()
settings = get_settings()

# ---> Set the Theme of Charts
sns.set_theme(
    style="whitegrid",
    palette="muted"
)

# ---> Function of EDA
def run_eda(output_dir: Path | None = None) -> None:
    output_dir = output_dir or (settings.root_dir / "House Price Prediction" / "eda_figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    console.print(f"[cyan]EDA on {len(df)} samples, {df.shape[1]} columns[/cyan]")
    console.print(df.describe().T)
    
    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df['MedHouseVal'], kde=True, ax=ax)
    ax.set_title("Median House Value Distribution (×$100k)")
    ax.set_xlabel("MedHouseVal")
    fig.tight_layout()
    fig.savefig(output_dir / "01_target_distribution.png", dpi=120)
    plt.close()
    
    # 2. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "02_correlation_heatmap.png", dpi=120)
    plt.close()
    
    # 3. MedInc vs. Target (strongest predictor)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="MedInc", y="MedHouseVal", alpha=0.3, ax=ax)
    ax.set_title("Median Income vs House Value")
    fig.tight_layout()
    fig.savefig(output_dir / "03_medinc_vs_target.png", dpi=120)
    plt.close()
    
    # 4. Geographic price map
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        df["Longitude"],
        df["Latitude"],
        c=df["MedHouseVal"],
        cmap="viridis",
        alpha=0.4,
        s=10,
    )
    plt.colorbar(scatter, ax=ax, label="MedHouseVal")
    ax.set_title("California Housing Prices by Location")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(output_dir / "04_geo_prices.png", dpi=120)
    plt.close()
    
    # 5. Pairplot of key features (sample for speed)
    sample = df.sample(n=min(2000, len(df)), random_state=42)
    key_cols = ["MedInc", "HouseAge", "AveRooms", "AveOccup", "MedHouseVal"]
    g = sns.pairplot(sample[key_cols], diag_kind="kde", corner=True)
    g.fig.suptitle("Pairplot of Key Features", y=1.02)
    g.savefig(output_dir / "05_pairplot.png", dpi=100)
    plt.close()

    console.print(f"[green]EDA figures saved to {output_dir}[/green]")


if __name__ == "__main__":
    run_eda()
