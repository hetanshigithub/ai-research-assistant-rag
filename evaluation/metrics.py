"""Retrieval metrics computed against text-labeled evidence.

A "relevant" chunk is one whose text contains an evidence phrase. Recall is measured per
*evidence item* (is each required piece of evidence present in the top k?) rather than per
chunk, so overlapping chunks that repeat the same sentence are not double counted.
"""

from __future__ import annotations

import re


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def contains_evidence(chunk_text: str, phrase: str) -> bool:
    return normalize(phrase) in normalize(chunk_text)


def is_relevant(chunk_text: str, evidence: list[str]) -> bool:
    return any(contains_evidence(chunk_text, phrase) for phrase in evidence)


def recall_at_k(chunk_texts: list[str], evidence: list[str], k: int) -> float:
    """Fraction of evidence items found in at least one of the top-k chunks."""
    if not evidence:
        raise ValueError("recall requires at least one evidence phrase")
    top = chunk_texts[:k]
    found = sum(1 for phrase in evidence if any(contains_evidence(t, phrase) for t in top))
    return found / len(evidence)


def hit_at_k(chunk_texts: list[str], evidence: list[str], k: int) -> float:
    """1.0 if any top-k chunk is relevant, else 0.0."""
    return 1.0 if any(is_relevant(t, evidence) for t in chunk_texts[:k]) else 0.0


def precision_at_k(chunk_texts: list[str], evidence: list[str], k: int) -> float:
    """Fraction of the top-k chunks that are relevant."""
    top = chunk_texts[:k]
    if not top:
        return 0.0
    return sum(1 for t in top if is_relevant(t, evidence)) / len(top)


def reciprocal_rank(chunk_texts: list[str], evidence: list[str]) -> float:
    """1/rank of the first relevant chunk, or 0.0 if none was retrieved."""
    for rank, text in enumerate(chunk_texts, start=1):
        if is_relevant(text, evidence):
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
