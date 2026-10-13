"""Evaluate retrieval and answer quality on a labeled question set.

    python -m evaluation.run_eval
    python -m evaluation.run_eval --embedding-model hashing          # offline, no download
    python -m evaluation.run_eval --sweep-chunk-sizes 60,100,180     # chunk-size ablation
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.answer_checks import auroc, best_threshold, fact_coverage, groundedness
from evaluation.dataset import EvalQuestion, load_eval_set
from evaluation.metrics import (
    hit_at_k,
    is_relevant,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from src.embeddings import load_embedder
from src.generator import generate_answer
from src.pipeline import build_or_load_retriever

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def run_evaluation(
    data_dir: Path,
    eval_set_path: Path,
    embedding_model,
    chunk_size: int = 180,
    overlap: int = 35,
    ks: tuple[int, ...] = (1, 3, 5),
    answer_top_k: int = 4,
    confidence_threshold: float = 0.5,
    index_type: str = "flat",
) -> dict:
    questions = load_eval_set(eval_set_path)
    build = build_or_load_retriever(
        data_dir,
        embedding_model,
        index_dir=None,  # always evaluate a fresh in-memory index
        chunk_size=chunk_size,
        overlap=overlap,
        index_type=index_type,
    )
    retriever = build.retriever
    depth = max(max(ks), answer_top_k)

    per_question: list[dict] = []
    for q in questions:
        retrieved = retriever.retrieve(q.question, top_k=depth)
        texts = [r.chunk.text for r in retrieved]
        answer = generate_answer(q.question, retrieved[:answer_top_k])

        record: dict = {
            "id": q.id,
            "question": q.question,
            "answerable": q.answerable,
            "confidence": answer["confidence_score"],
            "top1_similarity": retrieved[0].score if retrieved else 0.0,
            "retrieved": [
                {
                    "rank": rank,
                    "source_file": Path(r.chunk.source).name,
                    "chunk_id": r.chunk.chunk_id,
                    "score": round(r.score, 4),
                    "relevant": is_relevant(r.chunk.text, q.evidence) if q.answerable else None,
                }
                for rank, r in enumerate(retrieved[: max(ks)], start=1)
            ],
        }

        if q.answerable:
            for k in ks:
                record[f"recall@{k}"] = recall_at_k(texts, q.evidence, k)
                record[f"hit@{k}"] = hit_at_k(texts, q.evidence, k)
                record[f"precision@{k}"] = precision_at_k(texts, q.evidence, k)
            record["mrr"] = reciprocal_rank(texts, q.evidence)
            record["fact_coverage"] = fact_coverage(answer, q.required_facts)
            score, unsupported = groundedness(answer, texts[:answer_top_k])
            record["groundedness"] = score
            record["unsupported_claims"] = unsupported

        per_question.append(record)

    return {
        "config": {
            "embedding_model": embedding_model.model_name,
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "index_type": index_type,
            "chunks_indexed": build.chunks,
            "documents": build.documents,
            "questions": len(questions),
            "answerable": sum(q.answerable for q in questions),
            "unanswerable": sum(not q.answerable for q in questions),
            "answer_top_k": answer_top_k,
            "confidence_threshold": confidence_threshold,
        },
        "summary": summarize(per_question, ks, confidence_threshold),
        "per_question": per_question,
    }


def summarize(per_question: list[dict], ks: tuple[int, ...], threshold: float) -> dict:
    answerable = [r for r in per_question if r["answerable"]]
    unanswerable = [r for r in per_question if not r["answerable"]]

    retrieval: dict[str, float] = {}
    for k in ks:
        retrieval[f"recall@{k}"] = mean([r[f"recall@{k}"] for r in answerable])
        retrieval[f"hit@{k}"] = mean([r[f"hit@{k}"] for r in answerable])
        retrieval[f"precision@{k}"] = mean([r[f"precision@{k}"] for r in answerable])
    retrieval["mrr"] = mean([r["mrr"] for r in answerable])

    grounded = [r["groundedness"] for r in answerable if r["groundedness"] is not None]
    answer = {
        "fact_coverage": mean([r["fact_coverage"] for r in answerable]),
        "all_facts_present_rate": mean([1.0 if r["fact_coverage"] == 1.0 else 0.0 for r in answerable]),
        "groundedness": mean(grounded),
    }

    conf_ans = [r["confidence"] for r in answerable]
    conf_un = [r["confidence"] for r in unanswerable]
    sim_ans = [r["top1_similarity"] for r in answerable]
    sim_un = [r["top1_similarity"] for r in unanswerable]
    suggested = best_threshold(conf_ans, conf_un)
    abstention = {
        "mean_confidence_answerable": mean(conf_ans),
        "mean_confidence_unanswerable": mean(conf_un),
        "confidence_auroc": auroc(conf_ans, conf_un),
        "top1_similarity_auroc": auroc(sim_ans, sim_un),
        "unanswerable_flagged_rate": mean([1.0 if c < threshold else 0.0 for c in conf_un]),
        "answerable_wrongly_flagged_rate": mean([1.0 if c < threshold else 0.0 for c in conf_ans]),
        # Fit on this same question set, so optimistic. Validate on fresh questions.
        "suggested_threshold": suggested[0] if suggested else None,
        "balanced_accuracy_at_suggested": suggested[1] if suggested else None,
    }
    return {"retrieval": retrieval, "answer": answer, "abstention": abstention}


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def print_report(report: dict, ks: tuple[int, ...]) -> None:
    cfg, s = report["config"], report["summary"]
    print(
        f"\nModel: {cfg['embedding_model']} | chunks: {cfg['chunks_indexed']} "
        f"(size {cfg['chunk_size']}, overlap {cfg['chunk_overlap']}, {cfg['index_type']}) | "
        f"{cfg['answerable']} answerable + {cfg['unanswerable']} unanswerable questions"
    )

    print("\nRetrieval (answerable questions)")
    print("| k | recall@k | hit@k | precision@k |")
    print("|---|---------:|------:|------------:|")
    for k in ks:
        r = s["retrieval"]
        print(f"| {k} | {fmt(r[f'recall@{k}'])} | {fmt(r[f'hit@{k}'])} | {fmt(r[f'precision@{k}'])} |")
    print(f"\nMRR: {fmt(s['retrieval']['mrr'])}")

    a = s["answer"]
    print("\nAnswer quality (answerable questions)")
    print(f"- fact coverage:          {fmt(a['fact_coverage'])}")
    print(f"- all facts present:      {fmt(a['all_facts_present_rate'])}")
    print(f"- groundedness:           {fmt(a['groundedness'])}")

    ab = s["abstention"]
    print(f"\nKnowing when not to answer (confidence threshold {cfg['confidence_threshold']})")
    print(f"- mean confidence, answerable / unanswerable: "
          f"{fmt(ab['mean_confidence_answerable'])} / {fmt(ab['mean_confidence_unanswerable'])}")
    print(f"- confidence AUROC:       {fmt(ab['confidence_auroc'])}   (0.5 = no signal)")
    print(f"- top-1 similarity AUROC: {fmt(ab['top1_similarity_auroc'])}")
    print(f"- unanswerable flagged:   {fmt(ab['unanswerable_flagged_rate'])}")
    print(f"- answerable wrongly flagged: {fmt(ab['answerable_wrongly_flagged_rate'])}")
    print(f"- suggested threshold:    {fmt(ab['suggested_threshold'])} "
          f"(balanced accuracy {fmt(ab['balanced_accuracy_at_suggested'])}; fit on this same set, so optimistic)")

    misses = [r for r in report["per_question"] if r["answerable"] and r["hit@3"] == 0.0]
    if misses:
        print("\nQuestions with no relevant chunk in the top 3:")
        for r in misses:
            print(f"- {r['id']}: {r['question']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate the RAG pipeline on a labeled question set.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--eval-set", default="evaluation/eval_set.jsonl")
    p.add_argument("--embedding-model", default=DEFAULT_MODEL,
                   help="sentence-transformers model name, or 'hashing' for the offline embedder")
    p.add_argument("--chunk-size", type=int, default=180)
    p.add_argument("--chunk-overlap", type=int, default=35)
    p.add_argument("--index-type", choices=["flat", "hnsw"], default="flat")
    p.add_argument("--answer-top-k", type=int, default=4)
    p.add_argument("--confidence-threshold", type=float, default=0.5)
    p.add_argument("--sweep-chunk-sizes", default="",
                   help="comma-separated chunk sizes to compare, e.g. 60,100,180")
    p.add_argument("--report", default="outputs/eval_report.json", help="where to write the JSON report")
    p.add_argument("--min-recall-at-5", type=float, default=None,
                   help="exit with status 1 if recall@5 falls below this value (regression gate)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ks = (1, 3, 5)
    model = load_embedder(args.embedding_model)

    if args.sweep_chunk_sizes:
        sizes = [int(s) for s in args.sweep_chunk_sizes.split(",") if s.strip()]
        print(f"Chunk-size sweep with {model.model_name}")
        print("| chunk size | chunks | recall@1 | recall@3 | recall@5 | MRR | fact coverage |")
        print("|-----------:|-------:|---------:|---------:|---------:|----:|--------------:|")
        for size in sizes:
            overlap = max(1, round(size * 0.2))
            rep = run_evaluation(
                Path(args.data_dir), Path(args.eval_set), model, size, overlap, ks,
                args.answer_top_k, args.confidence_threshold, args.index_type,
            )
            r, a = rep["summary"]["retrieval"], rep["summary"]["answer"]
            print(f"| {size} | {rep['config']['chunks_indexed']} | {fmt(r['recall@1'])} | "
                  f"{fmt(r['recall@3'])} | {fmt(r['recall@5'])} | {fmt(r['mrr'])} | {fmt(a['fact_coverage'])} |")
        return 0

    report = run_evaluation(
        Path(args.data_dir), Path(args.eval_set), model, args.chunk_size, args.chunk_overlap, ks,
        args.answer_top_k, args.confidence_threshold, args.index_type,
    )
    print_report(report, ks)

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved report to {out}")

    if args.min_recall_at_5 is not None:
        recall = report["summary"]["retrieval"]["recall@5"]
        if recall < args.min_recall_at_5:
            print(f"FAIL: recall@5 {recall:.3f} is below the required {args.min_recall_at_5:.3f}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
