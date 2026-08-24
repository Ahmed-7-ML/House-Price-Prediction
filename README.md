
# 🏠 House Price MLOps – End-to-End Reference System

A **production-style, modular, reference implementation** of an end-to-end machine learning system for **California house price prediction**.

Use this repository as a **template / blueprint** for future ML projects.

---

## 🎯 What This System Covers

| Layer                         | Tools                                              | Purpose                                             |
| ----------------------------- | -------------------------------------------------- | --------------------------------------------------- |
| **Package & Env**       | `uv`                                             | Fast, reproducible dependency & project management  |
| **Data Ingestion**      | scikit-learn, Pandas                               | Load California Housing dataset                     |
| **ETL Pipeline**        | Pandas / Polars-ready                              | Clean, engineer features, train/test split, Parquet |
| **EDA**                 | Matplotlib, Seaborn                                | Distribution, correlation, geo, pairplots           |
| **Feature Store**       | Feast (local)                                      | Offline/online feature definitions & retrieval      |
| **Feature Pipeline**    | scikit-learn`Pipeline` + `ColumnTransformer`   | Imputation, scaling, optional polynomial            |
| **Modeling**            | scikit-learn (Linear, Ridge, RF, GradientBoosting) | Cross-validated training                            |
| **Experiment Tracking** | MLflow                                             | Params, metrics, model logging & registry           |
| **Serving**             | FastAPI + Pydantic                                 | REST API with`/predict` & `/health`             |
| **Containerization**    | Docker + Docker Compose                            | API + optional MLflow server                        |
| **CLI**                 | Typer + Rich                                       | One-command pipeline orchestration                  |

---

## 📂 Project Structure

```
house-price-mlops/
├── configs/
│   └── config.yaml              # Central configuration
├── data/
│   ├── raw/                     # Original CSV
│   ├── processed/               # Cleaned Parquet
│   └── features/                # Feature-store ready data
├── feature_repo/                # Feast feature store
│   ├── feature_store.yaml
│   ├── entities.py
│   ├── features.py
│   └── data/
├── models/
│   └── artifacts/               # best_model.joblib + feature_names
├── mlruns/                      # MLflow local tracking
├── notebooks/                   # Optional Jupyter exploration
├── src/house_price_mlops/
│   ├── api/                     # FastAPI app + schemas
│   ├── config/                  # pydantic-settings
│   ├── data/                    # ingest, preprocess, eda
│   ├── features/                # sklearn pipeline + Feast helpers
│   ├── models/                  # train + evaluate
│   ├── utils/
│   └── cli.py                   # Typer CLI
├── docker/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml               # uv / project metadata
└── README.md
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Docker (optional, for serving)

### 2. Install

```bash
cd house-price-mlops
uv sync                          # creates .venv + installs all deps
# or with dev tools:
uv sync --extra dev
```

### 3. Run the full pipeline (one command)

```bash
uv run house-price pipeline
```

This executes:

1. **Ingest** California Housing dataset
2. **Preprocess** (clean + engineer features + split)
3. **Prepare** Feast offline data
4. **Train** multiple models, log to MLflow, save best model

### 4. Individual commands

```bash
uv run house-price ingest          # download raw data
uv run house-price preprocess      # ETL
uv run house-price eda             # generate EDA figures → notebooks/eda_figures/
uv run house-price prepare-features
uv run house-price train           # train all models + MLflow
uv run house-price train --model gradient_boosting
uv run house-price evaluate
uv run house-price serve           # start FastAPI on :8000
```

### 5. MLflow UI

```bash
uv run mlflow ui --backend-store-uri ./mlruns --port 5000
# open http://localhost:5000
```

### 6. API usage

```bash
# Health
curl http://localhost:8000/health

# Prediction
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

---

## 🐳 Docker

```bash
# Build & run API (model must already exist in models/artifacts/)
docker compose up --build api

# Optional MLflow tracking server
docker compose up mlflow
```

---

## 🧪 Feature Store (Feast)

After `house-price prepare-features` and `house-price preprocess`:

```bash
cd feature_repo
uv run feast apply
# materialize if you want online store
uv run feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

Historical features can then be retrieved via `house_price_mlops.features.store.get_training_features_from_store()`.

---

## 📊 Expected Performance (California Housing)

| Model             | Typical R² (test)    | Notes        |
| ----------------- | --------------------- | ------------ |
| Linear Regression | ~0.60                 | Baseline     |
| Ridge             | ~0.60                 | Regularized  |
| Random Forest     | ~0.80                 | Strong       |
| Gradient Boosting | **~0.83–0.85** | Usually best |

Median Income (`MedInc`) is by far the strongest predictor.

---

## 🧩 How to Reuse This as a Template

1. Copy the repo / folder structure.
2. Replace the dataset in `data/ingest.py`.
3. Update feature lists in `features/pipeline.py` and `configs/config.yaml`.
4. Adjust model zoo in `models/train.py`.
5. Update Pydantic schemas in `api/schemas.py`.
6. Keep the same CLI + MLflow + Docker pattern.

The separation of concerns (ingest → preprocess → features → train → serve) is intentional and production-friendly.

---

## 🛠️ Development

```bash
uv run ruff check src/
uv run pytest tests/          # add your tests
uv run jupyter lab            # if you installed --extra dev
```

---

## License

MIT – free to use as a reference for your own projects.
