from __future__ import annotations

from typing import Any

from api.schemas import Citation


def locate_citations(statements: list[str], retrieved: list[Any]) -> list[Citation]:
    """Find where each answer statement appears in the retrieved chunks.

    The generator copies sentences out of the chunk text, so a statement can be
    matched word for word. When it is missing the final period (the generator adds
    one to sentences that were cut off by a chunk boundary), the period is dropped
    before matching. A statement found nowhere is returned with ``found=False``
    rather than guessed at.
    """
    citations: list[Citation] = []
    for statement in statements:
        needles = [statement]
        if statement.endswith("."):
            needles.append(statement[:-1])

        match = None
        for rank, result in enumerate(retrieved, start=1):
            for needle in needles:
                start = result.chunk.text.find(needle)
                if start != -1:
                    match = (rank, result, start, start + len(needle))
                    break
            if match:
                break

        if match is None:
            citations.append(Citation(statement=statement, found=False))
            continue

        rank, result, start, end = match
        citations.append(
            Citation(
                statement=statement,
                found=True,
                rank=rank,
                chunk_id=result.chunk.chunk_id,
                source_file=result.chunk.source.replace("\\", "/").rsplit("/", 1)[-1],
                start=start,
                end=end,
            )
        )
    return citations
