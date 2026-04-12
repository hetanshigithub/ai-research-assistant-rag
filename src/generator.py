from __future__ import annotations

import re
from collections import Counter
from typing import Any


STOPWORDS = {
    "across",
    "about",
    "after",
    "also",
    "because",
    "been",
    "being",
    "from",
    "have",
    "into",
    "more",
    "over",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "using",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
}

RISK_TERMS = {
    "risk",
    "risks",
    "limitation",
    "limitations",
    "hallucination",
    "hallucinations",
    "bias",
    "privacy",
    "compliance",
    "governance",
    "audit",
    "drift",
    "overreliance",
    "explainability",
    "validation",
}


def generate_answer(query: str, retrieved: list[Any]) -> dict:
    """Generate a structured answer using only retrieved context."""
    if not retrieved:
        return {
            "summary": "No relevant context was retrieved.",
            "key_insights": [],
            "risks_limitations": [
                "The system could not retrieve supporting evidence, so no grounded answer should be trusted."
            ],
            "supporting_evidence": [],
            "confidence_score": 0.0,
        }

    query_terms = extract_terms(query)
    ranked_sentences = rank_sentences(query_terms, retrieved)
    best_sentences = [sentence for sentence, _ in ranked_sentences[:5]]

    summary = build_summary(query, best_sentences)
    key_insights = build_key_insights(best_sentences)
    risks_limitations = build_risks_limitations(retrieved)
    evidence = build_supporting_evidence(retrieved)
    confidence_score = compute_confidence_score(retrieved, query_terms)

    return {
        "summary": summary,
        "key_insights": key_insights,
        "risks_limitations": risks_limitations,
        "supporting_evidence": evidence,
        "confidence_score": confidence_score,
    }


def extract_terms(text: str) -> set[str]:
    words = re.findall(r"\b[a-zA-Z][a-zA-Z-]{2,}\b", text.lower())
    return {word for word in words if word not in STOPWORDS}


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if len(sentence.split()) >= 6]


def rank_sentences(
    query_terms: set[str],
    retrieved: list[Any],
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []

    for result in retrieved:
        for sentence in split_sentences(result.chunk.text):
            sentence_terms = extract_terms(sentence)
            overlap = len(query_terms.intersection(sentence_terms))
            score = result.score + (0.08 * overlap)
            scored.append((sentence, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return dedupe_sentences(scored)


def dedupe_sentences(scored: list[tuple[str, float]]) -> list[tuple[str, float]]:
    seen: set[str] = set()
    unique: list[tuple[str, float]] = []

    for sentence, score in scored:
        fingerprint = " ".join(sentence.lower().split()[:12])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append((sentence, score))

    return unique


def build_summary(query: str, sentences: list[str]) -> str:
    if not sentences:
        return (
            "The retrieved context did not contain enough direct evidence to "
            f"answer: {query}"
        )

    concise = " ".join(sentences[:2])
    return f"Based on the retrieved evidence, {concise}"


def build_key_insights(sentences: list[str]) -> list[str]:
    if not sentences:
        return ["The retrieved documents do not provide a strong answer."]

    insights = []
    for sentence in sentences[:4]:
        cleaned = sentence.strip()
        if cleaned[-1] not in ".!?":
            cleaned += "."
        insights.append(cleaned)
    return insights


def build_risks_limitations(retrieved: list[Any]) -> list[str]:
    risk_sentences: list[str] = []

    for result in retrieved:
        for sentence in split_sentences(result.chunk.text):
            sentence_terms = extract_terms(sentence)
            risk_matches = sentence_terms.intersection(RISK_TERMS)
            if "risk language" in sentence.lower() and len(risk_matches) == 1:
                continue
            if risk_matches:
                risk_sentences.append(sentence)

    if risk_sentences:
        return [normalize_sentence(sentence) for sentence in risk_sentences[:3]]

    return [
        "The answer is limited to the documents currently available in the local data folder.",
        "The rule-based generator summarizes retrieved evidence but does not reason like a full language model.",
        "Low or narrow retrieval scores may indicate that the document set does not fully answer the query.",
    ]


def normalize_sentence(sentence: str) -> str:
    cleaned = sentence.strip()
    if cleaned and cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def build_supporting_evidence(retrieved: list[Any]) -> list[dict]:
    evidence = []
    for result in retrieved:
        excerpt = result.chunk.text.replace("\n", " ")
        excerpt = " ".join(excerpt.split())
        if len(excerpt) > 240:
            excerpt = excerpt[:240].rstrip() + "..."
        evidence.append(
            {
                "source": result.chunk.source,
                "chunk_id": result.chunk.chunk_id,
                "score": round(result.score, 4),
                "evidence": excerpt,
            }
        )
    return evidence


def compute_relevance_score(retrieved: list[Any]) -> float:
    top_scores = [max(0.0, result.score) for result in retrieved[:3]]
    if not top_scores:
        return 0.0
    return round(sum(top_scores) / len(top_scores), 4)


def compute_confidence_score(
    retrieved: list[Any],
    query_terms: set[str],
) -> float:
    """Estimate answer confidence from retrieval quality and query coverage."""
    if not retrieved:
        return 0.0

    combined_context = " ".join(result.chunk.text.lower() for result in retrieved)
    term_hits = Counter(term for term in query_terms if term in combined_context)
    coverage = len(term_hits) / max(1, len(query_terms))
    avg_similarity = compute_relevance_score(retrieved)
    evidence_available = 1.0 if retrieved and max(result.score for result in retrieved) > 0 else 0.0
    confidence = (
        (0.55 * avg_similarity)
        + (0.35 * coverage)
        + (0.10 * evidence_available)
    )
    return round(min(1.0, confidence), 4)
