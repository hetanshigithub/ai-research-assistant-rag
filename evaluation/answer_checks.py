"""Answer-quality checks that need no LLM judge.

These are deliberately simple, transparent proxies:

- fact coverage: does the answer text mention each fact the question requires?
- groundedness: is each claim in the answer supported by the retrieved text?
- abstention signal: does the system's confidence separate answerable from
  unanswerable questions?

Groundedness is near-trivially high for an extractive generator (it copies sentences),
so here it acts as a regression guard: if the generator is ever replaced by one that
paraphrases or generates freely, unsupported claims will show up in this number.
"""

from __future__ import annotations

import re

from evaluation.metrics import normalize
from src.generator import split_sentences

SUMMARY_PREFIX = "Based on the retrieved evidence, "
NO_ANSWER_PREFIXES = (
    "The retrieved context did not contain",
    "No relevant context",
    "The retrieved documents do not provide",
)


def answer_text(answer: dict) -> str:
    """The generated prose of an answer (excludes the raw supporting-evidence excerpts)."""
    parts = [answer.get("summary", "")]
    parts.extend(answer.get("key_insights", []))
    parts.extend(answer.get("risks_limitations", []))
    return " ".join(parts)


def fact_coverage(answer: dict, required_facts: list[list[str]]) -> float:
    """Fraction of required facts that appear in the answer (any keyword of a fact counts)."""
    if not required_facts:
        raise ValueError("fact coverage requires at least one required fact")
    text = normalize(answer_text(answer))
    covered = sum(
        1 for alternatives in required_facts if any(normalize(k) in text for k in alternatives)
    )
    return covered / len(required_facts)


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 3]


def claim_units(answer: dict) -> list[str]:
    """The sentences that assert something: summary sentences and key insights."""
    units: list[str] = []
    summary = answer.get("summary", "")
    if summary and not summary.startswith(NO_ANSWER_PREFIXES):
        if summary.startswith(SUMMARY_PREFIX):
            summary = summary[len(SUMMARY_PREFIX):]
        units.extend(split_sentences(summary))
    for insight in answer.get("key_insights", []):
        if not insight.startswith(NO_ANSWER_PREFIXES):
            units.append(insight)
    return units


def groundedness(
    answer: dict,
    retrieved_texts: list[str],
    support_threshold: float = 0.9,
) -> tuple[float | None, list[str]]:
    """Share of claims whose words are (almost) all present in the retrieved text.

    Returns (score, unsupported_claims). The score is None when the answer makes no
    claims (for example when it abstained), so it can be excluded from averages.
    """
    units = claim_units(answer)
    if not units:
        return None, []

    context_tokens = set(_tokens(" ".join(retrieved_texts)))
    unsupported: list[str] = []
    for unit in units:
        tokens = _tokens(unit)
        if not tokens:
            continue
        supported = sum(1 for t in tokens if t in context_tokens) / len(tokens)
        if supported < support_threshold:
            unsupported.append(unit)

    return 1.0 - len(unsupported) / len(units), unsupported


def best_threshold(positives: list[float], negatives: list[float]) -> tuple[float, float] | None:
    """Threshold that best separates the groups, as (threshold, balanced_accuracy).

    A score >= threshold is predicted "answerable". Fitting a threshold on the same
    questions it is then measured on is optimistic; treat the result as a starting
    point to validate on fresh questions, not as a guarantee.
    """
    if not positives or not negatives:
        return None
    values = sorted(set(positives) | set(negatives))
    if len(values) == 1:
        return values[0], 0.5
    candidates = [(a + b) / 2 for a, b in zip(values, values[1:])]

    best: tuple[float, float] | None = None
    for t in candidates:
        tpr = sum(1 for p in positives if p >= t) / len(positives)
        tnr = sum(1 for n in negatives if n < t) / len(negatives)
        balanced = (tpr + tnr) / 2
        if best is None or balanced > best[1]:
            best = (t, balanced)
    return best


def auroc(positives: list[float], negatives: list[float]) -> float | None:
    """Probability that a random positive scores higher than a random negative.

    0.5 means the score carries no signal; 1.0 means it separates the groups perfectly.
    Returns None if either group is empty.
    """
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))
