from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.chunking import TextChunk

INDEX_FILE = "index.faiss"
CHUNKS_FILE = "chunks.json"
META_FILE = "meta.json"

INDEX_TYPES = ("flat", "hnsw")


class FaissVectorStore:
    """Vector store backed by a FAISS index plus the chunk records it points to.

    Embeddings are expected to be L2-normalized, so inner product equals cosine
    similarity and the returned scores are cosine similarities.

    - ``flat``: exact search, no training step. Right choice for small and medium corpora.
    - ``hnsw``: approximate graph index. Faster on large corpora, may miss a true neighbor.
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "flat",
        hnsw_m: int = 32,
        hnsw_ef_search: int = 64,
        _index: Any = None,
    ) -> None:
        import faiss

        if index_type not in INDEX_TYPES:
            raise ValueError(f"index_type must be one of {INDEX_TYPES}, got {index_type!r}.")
        if dimension <= 0:
            raise ValueError("dimension must be positive.")

        self.dimension = dimension
        self.index_type = index_type
        self.chunks: list[TextChunk] = []

        if _index is not None:
            self.index = _index
        elif index_type == "flat":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            self.index = faiss.IndexHNSWFlat(dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = hnsw_ef_search

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[TextChunk], embeddings: Any) -> None:
        import numpy as np

        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embeddings of shape (n, {self.dimension}), got {vectors.shape}."
            )
        if len(chunks) != vectors.shape[0]:
            raise ValueError("Number of chunks and embeddings must match.")

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_embedding: Any, k: int) -> list[tuple[TextChunk, float]]:
        import numpy as np

        if k <= 0:
            raise ValueError("k must be positive.")
        if not self.chunks:
            return []

        query = np.ascontiguousarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.dimension:
            raise ValueError(
                f"Query has dimension {query.shape[1]}, index has dimension {self.dimension}."
            )

        scores, ids = self.index.search(query, min(k, len(self.chunks)))
        return [
            (self.chunks[int(i)], float(score))
            for score, i in zip(scores[0], ids[0])
            if i != -1  # FAISS pads with -1 when fewer than k neighbors exist
        ]

    def save(self, directory: str | Path, metadata: dict | None = None) -> None:
        """Persist the index, the chunk records, and metadata describing how they were built."""
        import faiss

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / INDEX_FILE))
        records = [
            {"chunk_id": c.chunk_id, "source": c.source, "text": c.text} for c in self.chunks
        ]
        (path / CHUNKS_FILE).write_text(json.dumps(records), encoding="utf-8")

        meta = dict(metadata or {})
        meta.update(
            {
                "dimension": self.dimension,
                "index_type": self.index_type,
                "num_chunks": len(self.chunks),
            }
        )
        (path / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> tuple["FaissVectorStore", dict]:
        import faiss

        path = Path(directory)
        meta = read_metadata(path)
        if meta is None:
            raise FileNotFoundError(f"No saved index found in {path}.")

        index = faiss.read_index(str(path / INDEX_FILE))
        records = json.loads((path / CHUNKS_FILE).read_text(encoding="utf-8"))

        if index.ntotal != len(records):
            raise ValueError(
                f"Corrupt index: {index.ntotal} vectors but {len(records)} chunk records."
            )

        store = cls(dimension=int(meta["dimension"]), index_type=meta["index_type"], _index=index)
        store.chunks = [
            TextChunk(chunk_id=r["chunk_id"], source=r["source"], text=r["text"]) for r in records
        ]
        return store, meta


def read_metadata(directory: str | Path) -> dict | None:
    """Return the saved metadata, or None if the directory holds no complete index."""
    path = Path(directory)
    required = (INDEX_FILE, CHUNKS_FILE, META_FILE)
    if not all((path / name).is_file() for name in required):
        return None
    try:
        return json.loads((path / META_FILE).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
