FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# CPU-only PyTorch: the default wheel bundles CUDA libraries and is several times larger.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-core.txt ./
RUN pip install -r requirements.txt

# Download the embedding model at build time so the container runs without network access.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')" \
    && chmod -R a+rX /opt/hf-cache

COPY src ./src
COPY api ./api
COPY evaluation ./evaluation
COPY data ./data

ENV DATA_DIR=/app/data \
    INDEX_DIR=/app/index \
    UPLOAD_DIR=/app/uploads \
    EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    HF_HUB_OFFLINE=1

RUN useradd --create-home appuser && mkdir -p /app/index /app/uploads && chown appuser /app/index /app/uploads
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=20s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
