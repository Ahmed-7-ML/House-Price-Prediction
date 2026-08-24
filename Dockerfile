FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
COPY models/ models/

# Install with pip for container simplicity
RUN pip install --no-cache-dir \
    pandas numpy scikit-learn joblib \
    fastapi uvicorn pydantic pydantic-settings \
    pyyaml pyarrow rich typer httpx

ENV PYTHONPATH=/app/src
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]