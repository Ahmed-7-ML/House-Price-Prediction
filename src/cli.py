# Command -> uv run python -m src.cli
"""
CLI entrypoint using Typer.

Run with: uv run house-price --help
          uv run python -m src.cli
"""
from typing import Optional
import typer
import uvicorn
from rich.console import Console

# Ingest, Preprocess and Show EDA
from src.data.ingest import ingest_raw_data
from src.data.preprocess import preprocess_split
from src.features.store import prepare_feature_store_data
from src.models.evaluate import evaluate_best_model
from src.models.train import train_log

# ---> Define the Typer App & Console
console = Console()
app = typer.Typer(
    name="house_price_prediction",
    help="House Price MLOps – end-to-end reference system",
    add_completion=False,
    no_args_is_help=True,
)

# ---> Define Commands
@app.command()
def ingest(force: bool = typer.Option(False, help="Re-download even if exists")):
    """Ingest California Housing dataset."""
    ingest_raw_data(force=force)


@app.command()
def preprocess():
    """Run ETL: clean, engineer features, split, save Parquet."""
    preprocess_split()


@app.command()
def eda():
    """Run exploratory data analysis and save figures."""
    run_eda()


@app.command()
def prepare_features():
    """Prepare data for Feast feature store."""
    prepare_feature_store_data()


@app.command()
def train(model: Optional[str] = typer.Option(None, help="Train only this model")):
    """Train models, log to MLflow, save best model."""
    result = train_log(model_name=model)
    console.print(result)


@app.command()
def evaluate():
    """Evaluate the saved best model on the test set."""
    evaluate_best_model()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0"),
    port: int = typer.Option(8000),
):
    """Start FastAPI prediction server."""
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=True,
    )


@app.command()
def pipeline():
    """Run full pipeline: ingest → preprocess → prepare features → train."""
    console.print("[bold]1/4 Ingest[/bold]")
    ingest_raw_data()

    console.print("[bold]2/4 Preprocess[/bold]")
    preprocess_split()

    console.print("[bold]3/4 Prepare feature store data[/bold]")
    prepare_feature_store_data()

    console.print("[bold]4/4 Train & log to MLflow[/bold]")
    train_log()

    console.print("[bold green]Full pipeline complete![/bold green]")


if __name__ == "__main__":
    app()
