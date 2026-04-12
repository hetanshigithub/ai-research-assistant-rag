from __future__ import annotations

from dataclasses import dataclass

from src.chunking import TextChunk
from src.embeddings import EmbeddingModel


@dataclass(frozen=True)
class RetrievalResult:
    chunk: TextChunk
    score: float


class SemanticRetriever:
    """In-memory semantic search over document chunks."""

    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model
        self.chunks: list[TextChunk] = []
        self.embeddings = None

    def index(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            raise ValueError("Cannot index an empty chunk list.")

        self.chunks = chunks
        self.embeddings = self.embedding_model.encode([chunk.text for chunk in chunks])

    def retrieve(self, query: str, top_k: int = 4) -> list[RetrievalResult]:
        if self.embeddings is None:
            raise RuntimeError("Retriever has not been indexed.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        query_embedding = self.embedding_model.encode([query])
        scores = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            RetrievalResult(chunk=self.chunks[index], score=float(scores[index]))
            for index in top_indices
        ]
