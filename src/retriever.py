from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.chunking import TextChunk
from src.vector_store import FaissVectorStore


@dataclass(frozen=True)
class RetrievalResult:
    chunk: TextChunk
    score: float


class SemanticRetriever:
    """Semantic search over document chunks, backed by a FAISS vector store."""

    def __init__(self, embedding_model, index_type: str = "flat") -> None:
        self.embedding_model = embedding_model
        self.index_type = index_type
        self.store: FaissVectorStore | None = None

    @property
    def chunks(self) -> list[TextChunk]:
        return self.store.chunks if self.store is not None else []

    def index(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot index an empty chunk list.")

        embeddings = self.embedding_model.encode([chunk.text for chunk in chunks])
        store = FaissVectorStore(dimension=embeddings.shape[1], index_type=self.index_type)
        store.add(chunks, embeddings)
        self.store = store

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievalResult]:
        if self.store is None:
            raise RuntimeError("Retriever has not been indexed.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_embedding = self.embedding_model.encode([query])
        hits = self.store.search(query_embedding, top_k)
        return [RetrievalResult(chunk=chunk, score=score) for chunk, score in hits]

    def save(self, directory: str | Path, metadata: dict | None = None) -> None:
        if self.store is None:
            raise RuntimeError("Retriever has not been indexed.")
        self.store.save(directory, metadata)

    @classmethod
    def load(cls, directory: str | Path, embedding_model) -> tuple["SemanticRetriever", dict]:
        store, meta = FaissVectorStore.load(directory)

        saved_model = meta.get("model_name")
        if saved_model is not None and saved_model != embedding_model.model_name:
            raise ValueError(
                f"Index was built with {saved_model!r} but the retriever uses "
                f"{embedding_model.model_name!r}. Rebuild the index."
            )

        retriever = cls(embedding_model, index_type=store.index_type)
        retriever.store = store
        return retriever, meta
