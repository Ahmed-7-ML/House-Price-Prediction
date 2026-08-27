# House Price MLOps – End-to-End Reference System

A production-style, modular reference implementation of an end-to-end machine learning system for California house price prediction.

Use this repository as a template / blueprint for future ML projects.

## What this system covers

| Layer | Tools | Purpose |
| --- | --- | --- |
| Package & env | `uv` | Fast, reproducible dependency & project management |
| Data ingestion | scikit-learn, Pandas | Load California Housing dataset |
| ETL pipeline | Pandas | Clean, engineer features, train/test split, Parquet |
| EDA | Matplotlib, Seaborn | Distribution, correlation, geo, pairplots |
| Feature store | Feast (local) | Offline/online feature definitions & retrieval |
| Feature pipeline | scikit-learn `Pipeline` + `ColumnTransformer` | Imputation, scaling, optional polynomial |
| Modeling | scikit-learn (Linear, Ridge, RF, GradientBoosting) | Cross-validated training |
| Experiment tracking | MLflow | Params, metrics, model logging & registry |
| Serving | FastAPI + Pydantic | REST API with `/predict`, `/predict/batch`, `/health` |
| Containerization | Docker + Docker Compose | API + optional MLflow server |
| CLI | Typer + Rich | One-command pipeline orchestration |

## Project structure

```
.
├── configs/config.yaml          # Central configuration
├── data/
│   ├── raw/                     # Original CSV
│   ├── processed/               # Cleaned Parquet
│   └── features/                # Feature-store ready data
├── feature_repo/                # Feast feature store
├── models/artifacts/            # best_model.joblib + feature_names
├── notebooks/eda_figures/       # EDA plots
├── src/
│   ├── api/                     # FastAPI app + schemas
│   ├── config/                  # pydantic-settings
│   ├── data/                    # ingest, preprocess, eda
│   ├── features/                # sklearn pipeline + Feast helpers
│   ├── models/                  # train + evaluate
│   ├── utils/
│   └── cli.py                   # Typer CLI
├── docker/Dockerfile.api
├── Dockerfile
├── docker-compose.yaml
├── pyproject.toml
└── README.md
```

## Quick start

### 1. Prerequisites

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker (optional, for serving)

### 2. Install

```bash
uv sync --extra dev
```

### 3. Run the full pipeline

```bash
uv run house-price pipeline
```

This executes ingest → preprocess → prepare Feast offline data → train (MLflow + best model).

### 4. Individual commands

```bash
uv run house-price ingest
uv run house-price preprocess
uv run house-price eda
uv run house-price prepare-features
uv run house-price train
uv run house-price train --model gradient_boosting
uv run house-price evaluate
uv run house-price serve
```

### 5. MLflow UI

```bash
uv run mlflow ui --backend-store-uri sqlite:./tmp/mlflow_house_price.db --port 5000
```

Open http://localhost:5000

### 6. API usage

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3252,
    "HouseAge": 41.0,
    "AveRooms": 6.9841,
    "AveBedrms": 1.0238,
    "Population": 322.0,
    "AveOccup": 2.5556,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

Interactive docs: http://localhost:8000/docs

## Docker

Train a model first so `models/artifacts/best_model.joblib` exists, then:

```bash
docker compose up --build api
docker compose up mlflow
```

## Feature store (Feast)

After `house-price preprocess` and `house-price prepare-features`:

```bash
cd feature_repo
uv run feast apply
uv run feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

Historical features can then be retrieved via `src.features.store.get_training_features_from_store()`.

## Expected performance (California Housing)

| Model | Typical R² (test) | Notes |
| --- | --- | --- |
| Linear Regression | ~0.60 | Baseline |
| Ridge | ~0.60 | Regularized |
| Random Forest | ~0.80 | Strong |
| Gradient Boosting | ~0.83–0.85 | Usually best |

Median income (`MedInc`) is by far the strongest predictor.

## How to reuse this as a template

1. Copy the folder structure.
2. Replace the dataset in `src/data/ingest.py`.
3. Update feature lists in `src/features/pipeline.py` and `configs/config.yaml`.
4. Adjust the model zoo in `src/models/train.py`.
5. Update Pydantic schemas in `src/api/schemas.py`.
6. Keep the same CLI + MLflow + Docker pattern.

## Development

```bash
uv run ruff check src/
uv run pytest tests/
```

## License

MIT – free to use as a reference for your own projects.
