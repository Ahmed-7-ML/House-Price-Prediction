FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY configs/ configs/
COPY models/ models/

RUN pip install --no-cache-dir \
    pandas numpy scikit-learn joblib pyarrow pyyaml \
    fastapi "uvicorn[standard]" pydantic pydantic-settings \
    rich typer

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
