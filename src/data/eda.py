"""
Exploratory data analysis with matplotlib and seaborn.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console

from src.config.settings import get_settings
from src.data.ingest import load_raw_data

console = Console()
settings = get_settings()

sns.set_theme(style="whitegrid", palette="muted")


def run_eda(output_dir: Path | None = None) -> None:
    output_dir = output_dir or settings.eda_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_data()
    console.print(f"[cyan]EDA on {len(df)} samples, {df.shape[1]} columns[/cyan]")
    console.print(df.describe().T)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["MedHouseVal"], kde=True, ax=ax)
    ax.set_title("Median House Value Distribution (×$100k)")
    ax.set_xlabel("MedHouseVal")
    fig.tight_layout()
    fig.savefig(output_dir / "01_target_distribution.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(output_dir / "02_correlation_heatmap.png", dpi=120)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="MedInc", y="MedHouseVal", alpha=0.3, ax=ax)
    ax.set_title("Median Income vs House Value")
    fig.tight_layout()
    fig.savefig(output_dir / "03_medinc_vs_target.png", dpi=120)
    plt.close()

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

    sample = df.sample(n=min(2000, len(df)), random_state=settings.seed)
    key_cols = ["MedInc", "HouseAge", "AveRooms", "AveOccup", "MedHouseVal"]
    pair = sns.pairplot(sample[key_cols], diag_kind="kde", corner=True)
    pair.fig.suptitle("Pairplot of Key Features", y=1.02)
    pair.savefig(output_dir / "05_pairplot.png", dpi=100)
    plt.close()

    console.print(f"[green]EDA figures saved to {output_dir}[/green]")


if __name__ == "__main__":
    run_eda()
