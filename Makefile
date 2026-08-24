.PHONY: install pipeline train serve eda docker-up clean

install:
	uv sync

# pipeline:
# 	uv run house-price pipeline

ingest:
	uv run python -m src.data.ingest

train:
	uv run python -m src.models.train

eda:
	uv run python -m src.data.eda

serve:
	uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

docker-up:
	docker compose up --build

clean:
	rm -rf data/raw/* data/processed/* data/features/* models/artifacts/* mlruns mlflow.db
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
