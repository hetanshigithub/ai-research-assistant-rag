import json
from pathlib import Path

import pytest

from evaluation.answer_checks import (
    auroc,
    best_threshold,
    claim_units,
    fact_coverage,
    groundedness,
)
from evaluation.dataset import load_eval_set
from evaluation.metrics import (
    contains_evidence,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from evaluation.run_eval import run_evaluation

CHUNKS = [
    "Flat search is exact and needs no training step.",
    "Sourdough bread needs flour water and salt.",
    "HNSW builds a layered graph of neighbors.",
]


# ---------------------------------------------------------------- retrieval metrics


def test_evidence_matching_ignores_case_and_whitespace() -> None:
    assert contains_evidence("Flat   search\nis EXACT", "flat search is exact")
    assert not contains_evidence("flat search", "flat search is exact")


def test_recall_counts_each_evidence_item_once() -> None:
    evidence = ["flat search is exact", "layered graph of neighbors"]
    assert recall_at_k(CHUNKS, evidence, k=1) == 0.5
    assert recall_at_k(CHUNKS, evidence, k=3) == 1.0
    assert recall_at_k(CHUNKS, ["nothing like this"], k=3) == 0.0


def test_recall_requires_evidence() -> None:
    with pytest.raises(ValueError):
        recall_at_k(CHUNKS, [], k=1)


def test_hit_precision_and_mrr() -> None:
    evidence = ["layered graph of neighbors"]
    assert hit_at_k(CHUNKS, evidence, k=2) == 0.0
    assert hit_at_k(CHUNKS, evidence, k=3) == 1.0
    assert precision_at_k(CHUNKS, evidence, k=3) == pytest.approx(1 / 3)
    assert reciprocal_rank(CHUNKS, evidence) == pytest.approx(1 / 3)
    assert reciprocal_rank(CHUNKS, ["absent"]) == 0.0
    assert reciprocal_rank(CHUNKS, ["flat search is exact"]) == 1.0


def test_precision_of_nothing_retrieved_is_zero() -> None:
    assert precision_at_k([], ["x"], k=3) == 0.0


# ---------------------------------------------------------------- answer checks


def _answer(summary: str, insights: list[str]) -> dict:
    return {"summary": summary, "key_insights": insights, "risks_limitations": []}


def test_fact_coverage_accepts_any_alternative_keyword() -> None:
    answer = _answer("Flat search is exact.", ["Approximate indexes trade accuracy for speed."])
    facts = [["exact"], ["approximate"], ["speed", "latency"], ["quantum"]]
    assert fact_coverage(answer, facts) == 0.75


def test_fact_coverage_requires_facts() -> None:
    with pytest.raises(ValueError):
        fact_coverage(_answer("x", []), [])


def test_groundedness_flags_claims_not_in_the_context() -> None:
    context = ["Flat search is exact and needs no training step before queries."]
    grounded = _answer(
        "Based on the retrieved evidence, Flat search is exact and needs no training step.",
        ["Flat search is exact and needs no training step before queries."],
    )
    score, unsupported = groundedness(grounded, context)
    assert score == 1.0 and unsupported == []

    invented = _answer(
        "Based on the retrieved evidence, Flat search is exact and needs no training step.",
        ["Quantum annealing hardware accelerates every nearest neighbor query dramatically."],
    )
    score, unsupported = groundedness(invented, context)
    assert score == 0.5
    assert unsupported == ["Quantum annealing hardware accelerates every nearest neighbor query dramatically."]


def test_abstaining_answers_make_no_claims() -> None:
    abstained = _answer(
        "The retrieved context did not contain enough direct evidence to answer: why?",
        ["The retrieved documents do not provide a strong answer."],
    )
    assert claim_units(abstained) == []
    assert groundedness(abstained, ["anything"]) == (None, [])


def test_auroc_perfect_random_and_empty() -> None:
    assert auroc([0.9, 0.8], [0.2, 0.1]) == 1.0
    assert auroc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert auroc([0.5], [0.5]) == 0.5
    assert auroc([], [0.5]) is None


def test_best_threshold_separates_groups() -> None:
    threshold, balanced = best_threshold([0.6, 0.7, 0.8], [0.1, 0.2, 0.3])
    assert 0.3 < threshold < 0.6
    assert balanced == 1.0
    assert best_threshold([], [0.1]) is None


# ---------------------------------------------------------------- eval set + end to end


def test_eval_set_is_well_formed(eval_set_path) -> None:
    questions = load_eval_set(eval_set_path)
    answerable = [q for q in questions if q.answerable]
    unanswerable = [q for q in questions if not q.answerable]
    assert len(answerable) >= 25
    assert len(unanswerable) >= 5
    assert all(q.required_facts for q in answerable)


def test_every_evidence_phrase_exists_in_its_listed_sources(eval_set_path, data_dir) -> None:
    """Keeps labels honest: editing a document can't silently invalidate the eval set."""
    docs = {p.name: p.read_text(encoding="utf-8") for p in data_dir.glob("*.txt")}
    for q in load_eval_set(eval_set_path):
        for source in q.sources:
            assert source in docs, f"{q.id}: unknown source {source}"
        for phrase in q.evidence:
            assert any(
                contains_evidence(docs[s], phrase) for s in q.sources
            ), f"{q.id}: evidence not found in {q.sources}: {phrase!r}"


def test_loader_rejects_bad_records(tmp_path) -> None:
    def write(*records: dict) -> Path:
        path = tmp_path / "set.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in records))
        return path

    ok = {"id": "a", "question": "q?", "answerable": True, "evidence": ["x"], "required_facts": [["x"]]}
    with pytest.raises(ValueError, match="duplicate"):
        load_eval_set(write(ok, ok))
    with pytest.raises(ValueError, match="no evidence"):
        load_eval_set(write({**ok, "evidence": []}))
    with pytest.raises(ValueError, match="unanswerable"):
        load_eval_set(write({**ok, "answerable": False}))
    with pytest.raises(ValueError, match="invalid eval record"):
        load_eval_set(write({"id": "a"}))


def test_end_to_end_evaluation_with_the_offline_embedder(embedder, data_dir, eval_set_path) -> None:
    report = run_evaluation(data_dir, eval_set_path, embedder, chunk_size=180, overlap=35)

    retrieval = report["summary"]["retrieval"]
    # Regression floor for the pipeline plumbing with the deterministic offline embedder,
    # NOT a statement about semantic-model quality.
    assert retrieval["recall@5"] >= 0.9
    assert retrieval["recall@1"] <= retrieval["recall@3"] <= retrieval["recall@5"]
    assert 0.0 <= retrieval["mrr"] <= 1.0
    assert report["config"]["answerable"] == 30
    assert report["config"]["unanswerable"] == 7
    assert report["summary"]["answer"]["groundedness"] >= 0.95
    assert len(report["per_question"]) == 37
