FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copiar primero lo mínimo para instalar el paquete (mejor uso de la caché de capas).
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install .

COPY evals/ ./evals/

RUN useradd --create-home --uid 10001 harness
USER harness

# Corre la suite de evals deterministas; sale con código distinto de 0 si algo falla.
CMD ["python", "-m", "centro_harness.evals", "--cases", "evals/cases.jsonl"]
