from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class EvalQuestion:
    """One labeled question.

    Evidence is labeled by *text*: a chunk counts as relevant if it contains one of the
    evidence phrases. That keeps labels valid when chunk size or overlap changes.

    ``required_facts`` is a list of facts; each fact is a list of acceptable keywords
    (any one of them satisfies it). It is used to check the generated answer.
    """

    id: str
    question: str
    answerable: bool
    sources: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    required_facts: list[list[str]] = field(default_factory=list)


def load_eval_set(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    seen: set[str] = set()

    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            question = EvalQuestion(
                id=raw["id"],
                question=raw["question"],
                answerable=bool(raw["answerable"]),
                sources=list(raw.get("sources", [])),
                evidence=list(raw.get("evidence", [])),
                required_facts=[list(f) for f in raw.get("required_facts", [])],
            )
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid eval record ({exc!r})") from exc

        if question.id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate question id {question.id!r}")
        seen.add(question.id)

        if question.answerable and not question.evidence:
            raise ValueError(f"{path}:{line_number}: answerable question {question.id} has no evidence")
        if not question.answerable and (question.evidence or question.sources):
            raise ValueError(f"{path}:{line_number}: unanswerable question {question.id} lists evidence")

        questions.append(question)

    if not questions:
        raise ValueError(f"{path}: no questions found")
    return questions
