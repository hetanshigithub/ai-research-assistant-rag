from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.citations import locate_citations
from api.schemas import (
    Answer,
    DocumentInfo,
    DocumentList,
    HealthResponse,
    IndexInfo,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
    UploadResponse,
)
from src.data_loader import SUPPORTED_EXTENSIONS, list_document_paths
from src.embeddings import load_embedder
from src.generator import generate_answer
from src.pipeline import IndexBuild, build_or_load_retriever

logger = logging.getLogger("rag.api")

SAFE_FILENAME = re.compile(r"^[\w][\w .()\-]{0,100}$")


def _origins() -> tuple[str, ...]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:5173")
    return tuple(o.strip() for o in raw.split(",") if o.strip())


@dataclass(frozen=True)
class Settings:
    """Runtime configuration. Every field can be set with an environment variable."""

    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    upload_dir: Path = field(default_factory=lambda: Path(os.getenv("UPLOAD_DIR", "uploads")))
    index_dir: Path = field(default_factory=lambda: Path(os.getenv("INDEX_DIR", "index")))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "180")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "35")))
    index_type: str = field(default_factory=lambda: os.getenv("INDEX_TYPE", "flat"))
    confidence_threshold: float = field(
        default_factory=lambda: float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
    )
    max_upload_bytes: int = field(
        default_factory=lambda: int(float(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024)
    )
    cors_origins: tuple[str, ...] = field(default_factory=_origins)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory; the model and index are loaded once, at startup."""
    config = settings or Settings()
    # Uploads and deletes rebuild the index, so only one may run at a time.
    write_lock = threading.Lock()

    def prepare_index(embedder) -> IndexBuild:
        return build_or_load_retriever(
            config.data_dir,
            embedder,
            index_dir=config.index_dir,
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
            index_type=config.index_type,
            extra_dirs=[config.upload_dir],
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading embedding model %s", config.embedding_model)
        embedder = load_embedder(config.embedding_model)
        config.upload_dir.mkdir(parents=True, exist_ok=True)

        # Fails fast (the app does not start) if the data folder is missing or empty.
        build = prepare_index(embedder)
        logger.info(
            "Index ready: %d chunks from %d documents (%s)",
            build.chunks,
            build.documents,
            "built" if build.rebuilt else "loaded from disk",
        )
        app.state.embedder = embedder
        app.state.build = build
        yield
        app.state.build = None

    app = FastAPI(
        title="AI Research Assistant (RAG)",
        description="Question answering over local documents using FAISS retrieval.",
        version="1.1.0",
        lifespan=lifespan,
    )
    app.state.build = None
    app.state.settings = config

    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.cors_origins),
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Content-Type"],
        )

    def get_build() -> IndexBuild:
        build = app.state.build
        if build is None:
            raise HTTPException(status_code=503, detail="Index is not ready yet.")
        return build

    def index_info_for(build: IndexBuild) -> IndexInfo:
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

    def describe_documents(build: IndexBuild) -> list[DocumentInfo]:
        chunk_counts = Counter(Path(c.source).name for c in build.retriever.chunks)
        infos: list[DocumentInfo] = []
        for origin, directory in (("sample", config.data_dir), ("uploaded", config.upload_dir)):
            if not directory.exists():
                continue
            for path in list_document_paths(directory):
                infos.append(
                    DocumentInfo(
                        name=path.name,
                        origin=origin,
                        chunks=chunk_counts.get(path.name, 0),
                        size_bytes=path.stat().st_size,
                    )
                )
        return infos

    def existing_names() -> set[str]:
        names: set[str] = set()
        for directory in (config.data_dir, config.upload_dir):
            if directory.exists():
                names.update(p.name.lower() for p in list_document_paths(directory))
        return names

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        ready = app.state.build is not None
        return HealthResponse(status="ok" if ready else "starting", index_ready=ready)

    @app.get("/index", response_model=IndexInfo, tags=["meta"])
    def index_info() -> IndexInfo:
        return index_info_for(get_build())

    # A plain `def` endpoint runs in FastAPI's thread pool, so the CPU-bound embedding
    # call does not block the event loop.
    @app.post("/query", response_model=QueryResponse, tags=["rag"])
    def query(request: QueryRequest) -> QueryResponse:
        build = get_build()
        started = time.perf_counter()

        retrieved = build.retriever.retrieve(request.question, top_k=request.top_k)
        answer = generate_answer(request.question, retrieved)
        citations = locate_citations(answer["key_insights"], retrieved)
        latency_ms = (time.perf_counter() - started) * 1000

        logger.info(
            "query top_k=%d chunks=%s scores=%s latency_ms=%.1f",
            request.top_k,
            [r.chunk.chunk_id for r in retrieved],
            [round(r.score, 3) for r in retrieved],
            latency_ms,
        )

        confidence = answer["confidence_score"]
        return QueryResponse(
            question=request.question,
            answer=Answer(**answer, citations=citations),
            answerable=confidence >= config.confidence_threshold,
            confidence_threshold=config.confidence_threshold,
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

    @app.get("/documents", response_model=DocumentList, tags=["documents"])
    def list_documents() -> DocumentList:
        build = get_build()
        return DocumentList(documents=describe_documents(build), total_chunks=build.chunks)

    @app.post("/documents", response_model=UploadResponse, status_code=201, tags=["documents"])
    def upload_document(file: UploadFile) -> UploadResponse:
        """Add a .txt or .pdf file and rebuild the index so it can be searched."""
        get_build()
        name = Path(file.filename or "").name
        if not SAFE_FILENAME.match(name) or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=415, detail="Only .txt and .pdf files with a plain file name are accepted."
            )

        content = file.file.read(config.max_upload_bytes + 1)
        if len(content) > config.max_upload_bytes:
            limit_mb = config.max_upload_bytes / (1024 * 1024)
            raise HTTPException(status_code=413, detail=f"File is larger than {limit_mb:g} MB.")
        if not content.strip():
            raise HTTPException(status_code=422, detail="The file is empty.")

        with write_lock:
            if name.lower() in existing_names():
                raise HTTPException(status_code=409, detail=f"A document named {name} already exists.")

            target = config.upload_dir / name
            target.write_bytes(content)
            try:
                build = prepare_index(app.state.embedder)
            except Exception as exc:  # unreadable PDF, no extractable text, and so on
                target.unlink(missing_ok=True)
                logger.warning("Rejected upload %s: %s", name, exc)
                raise HTTPException(
                    status_code=422, detail="No text could be extracted from this file."
                ) from exc

            if not any(Path(c.source).name == name for c in build.retriever.chunks):
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=422, detail="No text could be extracted from this file.")

            app.state.build = build
            document = next(d for d in describe_documents(build) if d.name == name)

        logger.info("Uploaded %s (%d chunks)", name, document.chunks)
        return UploadResponse(document=document, index=index_info_for(build))

    @app.delete("/documents/{name}", status_code=204, tags=["documents"])
    def delete_document(name: str) -> None:
        get_build()
        with write_lock:
            target = config.upload_dir / Path(name).name
            if not target.is_file():
                in_samples = name.lower() in {p.name.lower() for p in list_document_paths(config.data_dir)}
                if in_samples:
                    raise HTTPException(status_code=403, detail="Sample documents cannot be deleted.")
                raise HTTPException(status_code=404, detail=f"No uploaded document named {name}.")

            target.unlink()
            app.state.build = prepare_index(app.state.embedder)
        logger.info("Deleted %s", name)

    return app


app = create_app()
