from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence
from datetime import datetime, timezone
from pathlib import Path

from src.chunking import TextChunk, chunk_documents
from src.data_loader import Document, load_corpus
from src.retriever import SemanticRetriever
from src.vector_store import read_metadata


@dataclass(frozen=True)
class IndexBuild:
    """Result of preparing an index: the retriever plus how it was obtained."""

    retriever: SemanticRetriever
    metadata: dict
    documents: int
    chunks: int
    rebuilt: bool


def corpus_fingerprint(
    documents: list[Document],
    model_name: str,
    chunk_size: int,
    overlap: int,
    index_type: str,
) -> str:
    """Hash everything that determines index contents.

    If any input changes (a document is edited, the chunking settings change, a
    different embedding model is selected) the fingerprint changes and a saved
    index is treated as stale.
    """
    payload = {
        "model": model_name,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "index_type": index_type,
        "documents": [
            {"source": Path(d.source).as_posix(), "sha": hashlib.sha256(d.text.encode()).hexdigest()}
            for d in sorted(documents, key=lambda d: d.source)
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_or_load_retriever(
    data_dir: Path,
    embedding_model,
    index_dir: Path | None = None,
    chunk_size: int = 180,
    overlap: int = 35,
    index_type: str = "flat",
    force_rebuild: bool = False,
    extra_dirs: Sequence[Path] = (),
) -> IndexBuild:
    """Return a ready retriever, reusing a saved index when it is still valid.

    With ``index_dir=None`` nothing is read from or written to disk.
    """
    documents = load_corpus([Path(data_dir), *map(Path, extra_dirs)])
    fingerprint = corpus_fingerprint(
        documents, embedding_model.model_name, chunk_size, overlap, index_type
    )

    if index_dir is not None and not force_rebuild:
        saved = read_metadata(index_dir)
        if saved is not None and saved.get("fingerprint") == fingerprint:
            try:
                retriever, meta = SemanticRetriever.load(index_dir, embedding_model)
                return IndexBuild(
                    retriever=retriever,
                    metadata=meta,
                    documents=len(documents),
                    chunks=len(retriever.chunks),
                    rebuilt=False,
                )
            except (ValueError, OSError, KeyError):
                pass  # unreadable or inconsistent on disk: fall through and rebuild

    chunks: list[TextChunk] = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError(f"No usable text chunks found in {Path(data_dir).resolve()}.")

    retriever = SemanticRetriever(embedding_model, index_type=index_type)
    retriever.index(chunks)

    metadata = {
        "fingerprint": fingerprint,
        "model_name": embedding_model.model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "documents": len(documents),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if index_dir is not None:
        retriever.save(index_dir, metadata)

    metadata.update(
        {
            "dimension": retriever.store.dimension,
            "index_type": index_type,
            "num_chunks": len(chunks),
        }
    )
    return IndexBuild(
        retriever=retriever,
        metadata=metadata,
        documents=len(documents),
        chunks=len(chunks),
        rebuilt=True,
    )
