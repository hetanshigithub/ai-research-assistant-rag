from types import SimpleNamespace

from src.generator import generate_answer


def test_generate_answer_returns_required_sections() -> None:
    retrieved = [
        SimpleNamespace(
            chunk=SimpleNamespace(
                chunk_id=1,
                source="sample.txt",
                text=(
                    "RAG systems retrieve trusted context before generating an "
                    "answer. This improves grounding and makes evidence easier "
                    "to inspect."
                ),
            ),
            score=0.82,
        )
    ]

    answer = generate_answer("Why is RAG useful?", retrieved)

    assert "summary" in answer
    assert "key_insights" in answer
    assert "risks_limitations" in answer
    assert "supporting_evidence" in answer
    assert "confidence_score" in answer
    assert answer["confidence_score"] > 0
