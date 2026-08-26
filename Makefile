.PHONY: install pipeline ingest preprocess eda prepare-features train evaluate serve docker-up clean test

install:
	uv sync --extra dev

pipeline:
	uv run house-price pipeline

ingest:
	uv run python -m src.data.ingest

preprocess:
	uv run python -m src.data.preprocess

eda:
	uv run python -m src.data.eda

prepare-features:
	uv run python -m src.features.store

train:
	uv run python -m src.models.train

evaluate:
	uv run python -m src.models.evaluate

mlflow-ui:
	uv run mlflow ui --backend-store-uri sqlite:./tmp/mlflow_house_price.db --port 5000

serve:
	uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	uv run pytest tests/ -q

docker-up:
	docker compose up --build

clean:
	rm -rf data/raw/* data/processed/* data/features/* models/artifacts/* mlruns tmp/*.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
