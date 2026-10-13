from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException

from api.schemas import (
    HealthResponse,
    IndexInfo,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)
from src.embeddings import load_embedder
from src.generator import generate_answer
from src.pipeline import IndexBuild, build_or_load_retriever

logger = logging.getLogger("rag.api")


@dataclass(frozen=True)
class Settings:
    """Runtime configuration. Every field can be set with an environment variable."""

    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    index_dir: Path = field(default_factory=lambda: Path(os.getenv("INDEX_DIR", "index")))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "180")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "35")))
    index_type: str = field(default_factory=lambda: os.getenv("INDEX_TYPE", "flat"))


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory; the model and index are loaded once, at startup."""
    config = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading embedding model %s", config.embedding_model)
        embedder = load_embedder(config.embedding_model)

        # Fails fast (the app does not start) if the data folder is missing or empty.
        build = build_or_load_retriever(
            config.data_dir,
            embedder,
            index_dir=config.index_dir,
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
            index_type=config.index_type,
        )
        logger.info(
            "Index ready: %d chunks from %d documents (%s)",
            build.chunks,
            build.documents,
            "built" if build.rebuilt else "loaded from disk",
        )
        app.state.build = build
        yield
        app.state.build = None

    app = FastAPI(
        title="AI Research Assistant (RAG)",
        description="Question answering over local documents using FAISS retrieval.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.build = None
    app.state.settings = config

    def get_build() -> IndexBuild:
        build = app.state.build
        if build is None:
            raise HTTPException(status_code=503, detail="Index is not ready yet.")
        return build

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        ready = app.state.build is not None
        return HealthResponse(status="ok" if ready else "starting", index_ready=ready)

    @app.get("/index", response_model=IndexInfo, tags=["meta"])
    def index_info() -> IndexInfo:
        build = get_build()
        meta = build.metadata
        return IndexInfo(
            embedding_model=meta["model_name"],
            dimension=meta["dimension"],
            index_type=meta["index_type"],
            documents=build.documents,
            chunks=build.chunks,
            chunk_size=meta["chunk_size"],
            chunk_overlap=meta["chunk_overlap"],
            rebuilt_on_startup=build.rebuilt,
        )

    # A plain `def` endpoint runs in FastAPI's thread pool, so the CPU-bound embedding
    # call does not block the event loop.
    @app.post("/query", response_model=QueryResponse, tags=["rag"])
    def query(request: QueryRequest) -> QueryResponse:
        build = get_build()
        started = time.perf_counter()

        retrieved = build.retriever.retrieve(request.question, top_k=request.top_k)
        answer = generate_answer(request.question, retrieved)
        latency_ms = (time.perf_counter() - started) * 1000

        logger.info(
            "query top_k=%d chunks=%s scores=%s latency_ms=%.1f",
            request.top_k,
            [r.chunk.chunk_id for r in retrieved],
            [round(r.score, 3) for r in retrieved],
            latency_ms,
        )

        return QueryResponse(
            question=request.question,
            answer=answer,
            retrieved=[
                RetrievedChunk(
                    rank=rank,
                    source_file=Path(r.chunk.source).name,
                    chunk_id=r.chunk.chunk_id,
                    score=round(r.score, 4),
                    text=r.chunk.text if request.include_chunks else None,
                )
                for rank, r in enumerate(retrieved, start=1)
            ],
            latency_ms=round(latency_ms, 2),
        )

    return app


app = create_app()
