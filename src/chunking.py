from __future__ import annotations

from dataclasses import dataclass

from src.data_loader import Document


@dataclass(frozen=True)
class TextChunk:
    """Searchable text unit with source metadata."""

    chunk_id: int
    source: str
    text: str


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 180,
    overlap: int = 35,
) -> list[TextChunk]:
    """Split documents into overlapping word chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0:
        raise ValueError("overlap cannot be negative.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: list[TextChunk] = []
    next_chunk_id = 1

    for document in documents:
        words = document.text.split()
        if not words:
            continue

        step = chunk_size - overlap
        for start in range(0, len(words), step):
            end = start + chunk_size
            chunk_words = words[start:end]
            if not chunk_words:
                continue

            chunks.append(
                TextChunk(
                    chunk_id=next_chunk_id,
                    source=document.source,
                    text=" ".join(chunk_words),
                )
            )
            next_chunk_id += 1

            if end >= len(words):
                break

    return chunks
