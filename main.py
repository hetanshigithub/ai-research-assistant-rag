from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

from src.chunking import chunk_documents
from src.data_loader import load_documents
from src.embeddings import EmbeddingModel
from src.generator import generate_answer
from src.retriever import SemanticRetriever


warnings.filterwarnings(
    "ignore",
    message=".*resume_download.*",
    category=FutureWarning,
)


def parse_bool(value: str | bool) -> bool:
    """Parse CLI booleans like true/false without surprising argparse behavior."""
    if isinstance(value, bool):
        return value

    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a local RAG research assistant over TXT/PDF documents."
    )
    parser.add_argument("query", nargs="?", help="Research question to answer.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Folder containing .txt and .pdf documents.",
    )
    parser.add_argument(
        "--top_k",
        "--top-k",
        dest="top_k",
        type=int,
        default=4,
        help="Number of relevant chunks to retrieve.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=180,
        help="Target chunk size in words.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=35,
        help="Chunk overlap in words.",
    )
    parser.add_argument(
        "--results-file",
        default="outputs/results.json",
        help="Path where results should be saved.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformers embedding model.",
    )
    parser.add_argument(
        "--show_chunks",
        "--show-chunks",
        dest="show_chunks",
        type=parse_bool,
        default=True,
        help="Show retrieved evidence before the answer. Use true or false.",
    )
    parser.add_argument(
        "--save_json",
        "--save-json",
        dest="save_json",
        type=parse_bool,
        default=True,
        help="Save run output to JSON. Use true or false.",
    )
    return parser.parse_args()


def print_retrieved_chunks(results: list) -> None:
    print("\n" + "=" * 88)
    print("Top Retrieved Evidence")
    print("=" * 88)
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        preview = compact_preview(chunk.text)

        print(f"\nRank: {index}")
        print(f"Similarity Score: {result.score:.3f}")
        print(f"Source File: {Path(chunk.source).name}")
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Preview: {preview}")


def compact_preview(text: str, max_chars: int = 320) -> str:
    preview = " ".join(text.split())
    if len(preview) > max_chars:
        return preview[:max_chars].rstrip() + "..."
    return preview


def print_answer(answer: dict) -> None:
    print("\n" + "=" * 88)
    print("STRUCTURED ANSWER")
    print("=" * 88)
    print(f"\nSummary:\n{answer['summary']}")

    print("\nKey Insights:")
    for insight in answer["key_insights"]:
        print(f"- {insight}")

    print("\nRisks / Limitations:")
    for risk in answer["risks_limitations"]:
        print(f"- {risk}")

    print("\nSupporting Evidence:")
    for item in answer["supporting_evidence"]:
        print(
            f"- {Path(item['source']).name} | chunk {item['chunk_id']} | "
            f"score={item['score']:.3f}: {item['evidence']}"
        )

    print("\nConfidence Score:")
    print(f"{answer['confidence_score']:.3f}")


def save_result(payload: dict, results_file: str) -> None:
    output_path = Path(results_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {"runs": []}
    else:
        existing = {"runs": []}

    runs = existing.get("runs", [])
    runs.append(payload)
    output_path.write_text(json.dumps({"runs": runs}, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    query = args.query or input("Enter your research question:\n> ").strip()
    if not query:
        raise SystemExit("Query cannot be empty.")

    data_dir = Path(args.data_dir)
    documents = load_documents(data_dir)
    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
    )

    if not chunks:
        raise SystemExit(f"No usable text chunks found in {data_dir.resolve()}.")

    embedding_model = EmbeddingModel(model_name=args.model_name)
    retriever = SemanticRetriever(embedding_model)
    retriever.index(chunks)
    retrieved = retriever.retrieve(query, top_k=args.top_k)

    answer = generate_answer(query, retrieved)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "retrieved_sources": [
            {
                "source": item.chunk.source,
                "source_file": Path(item.chunk.source).name,
                "chunk_id": item.chunk.chunk_id,
                "similarity_score": round(item.score, 4),
                "preview": compact_preview(item.chunk.text),
            }
            for item in retrieved
        ],
        "similarity_scores": [round(item.score, 4) for item in retrieved],
        "final_answer": answer,
        "confidence_score": answer["confidence_score"],
        "metadata": {
            "data_dir": str(data_dir),
            "documents_loaded": len(documents),
            "chunks_indexed": len(chunks),
            "top_k": args.top_k,
        },
    }

    if args.show_chunks:
        print_retrieved_chunks(retrieved)
    print_answer(answer)
    if args.save_json:
        save_result(payload, args.results_file)
        print(f"\nSaved result to {Path(args.results_file).resolve()}")


if __name__ == "__main__":
    main()
