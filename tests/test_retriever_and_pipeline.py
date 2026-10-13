from pathlib import Path

import pytest

from src.embeddings import HashingEmbedder
from src.pipeline import build_or_load_retriever
from src.retriever import SemanticRetriever


def write_docs(folder: Path, **docs: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in docs.items():
        (folder / f"{name}.txt").write_text(text, encoding="utf-8")
    return folder


DOCS = {
    "faiss": "A flat FAISS index performs exact nearest neighbor search over stored vectors.",
    "bread": "Sourdough bread needs flour, water, salt, and a long fermentation overnight.",
}


def test_retriever_returns_the_most_relevant_chunk(tmp_path, embedder) -> None:
    data = write_docs(tmp_path / "data", **DOCS)

    build = build_or_load_retriever(data, embedder, index_dir=None, chunk_size=50, overlap=5)
    results = build.retriever.retrieve("exact nearest neighbor search with FAISS", top_k=2)

    assert Path(results[0].chunk.source).stem == "faiss"
    assert results[0].score > results[1].score


def test_retrieve_validates_arguments_and_state(embedder) -> None:
    retriever = SemanticRetriever(embedder)
    with pytest.raises(RuntimeError):
        retriever.retrieve("anything")
    with pytest.raises(ValueError):
        retriever.index([])


def test_second_run_loads_the_saved_index(tmp_path, embedder) -> None:
    data = write_docs(tmp_path / "data", **DOCS)
    index_dir = tmp_path / "index"

    first = build_or_load_retriever(data, embedder, index_dir=index_dir, chunk_size=50, overlap=5)
    second = build_or_load_retriever(data, embedder, index_dir=index_dir, chunk_size=50, overlap=5)

    assert first.rebuilt is True
    assert second.rebuilt is False
    assert second.chunks == first.chunks
    query = "flat FAISS index exact search"
    assert (
        first.retriever.retrieve(query, 1)[0].chunk.chunk_id
        == second.retriever.retrieve(query, 1)[0].chunk.chunk_id
    )


def test_editing_a_document_invalidates_the_saved_index(tmp_path, embedder) -> None:
    data = write_docs(tmp_path / "data", **DOCS)
    index_dir = tmp_path / "index"
    build_or_load_retriever(data, embedder, index_dir=index_dir, chunk_size=50, overlap=5)

    write_docs(data, bread="Sourdough bread now also needs a hot dutch oven for the first bake.")
    rebuilt = build_or_load_retriever(data, embedder, index_dir=index_dir, chunk_size=50, overlap=5)

    assert rebuilt.rebuilt is True


@pytest.mark.parametrize(
    "changed",
    [
        {"chunk_size": 20},  # chunk size alone
        {"overlap": 2},  # overlap alone
        {"index_type": "hnsw"},  # index type alone
    ],
)
def test_changing_any_single_index_setting_invalidates_the_saved_index(
    tmp_path, embedder, changed
) -> None:
    data = write_docs(tmp_path / "data", **DOCS)
    index_dir = tmp_path / "index"
    settings = {"chunk_size": 50, "overlap": 5, "index_type": "flat"}
    assert build_or_load_retriever(data, embedder, index_dir=index_dir, **settings).rebuilt is True
    assert build_or_load_retriever(data, embedder, index_dir=index_dir, **settings).rebuilt is False

    rebuilt = build_or_load_retriever(data, embedder, index_dir=index_dir, **{**settings, **changed})

    assert rebuilt.rebuilt is True


def test_changing_the_embedding_model_invalidates_the_saved_index(tmp_path, embedder) -> None:
    data = write_docs(tmp_path / "data", **DOCS)
    index_dir = tmp_path / "index"
    build_or_load_retriever(data, embedder, index_dir=index_dir, chunk_size=50, overlap=5)

    other_model = build_or_load_retriever(
        data, HashingEmbedder(dimension=256), index_dir=index_dir, chunk_size=50, overlap=5
    )

    assert other_model.rebuilt is True
    assert other_model.metadata["dimension"] == 256


def test_loading_an_index_with_a_different_embedding_model_is_refused(tmp_path, embedder) -> None:
    """Vectors from one model are meaningless to another, even if the dimensions agree."""
    data = write_docs(tmp_path / "data", **DOCS)
    index_dir = tmp_path / "index"
    built = build_or_load_retriever(data, embedder, index_dir=index_dir, chunk_size=50, overlap=5)
    assert built.metadata["model_name"] == "hashing-2048"

    with pytest.raises(ValueError, match="Rebuild the index"):
        SemanticRetriever.load(index_dir, HashingEmbedder(dimension=256))

    loaded, meta = SemanticRetriever.load(index_dir, embedder)
    assert len(loaded.chunks) == built.chunks and meta["model_name"] == "hashing-2048"


def test_force_rebuild_ignores_a_valid_index(tmp_path, embedder) -> None:
    data = write_docs(tmp_path / "data", **DOCS)
    index_dir = tmp_path / "index"
    build_or_load_retriever(data, embedder, index_dir=index_dir, chunk_size=50, overlap=5)

    forced = build_or_load_retriever(
        data, embedder, index_dir=index_dir, chunk_size=50, overlap=5, force_rebuild=True
    )
    assert forced.rebuilt is True


def test_a_corrupt_saved_index_is_rebuilt_instead_of_crashing(tmp_path, embedder) -> None:
    data = write_docs(tmp_path / "data", **DOCS)
    index_dir = tmp_path / "index"
    build_or_load_retriever(data, embedder, index_dir=index_dir, chunk_size=50, overlap=5)
    (index_dir / "chunks.json").write_text("[]")  # vectors and records no longer agree

    recovered = build_or_load_retriever(data, embedder, index_dir=index_dir, chunk_size=50, overlap=5)

    assert recovered.rebuilt is True
    assert recovered.chunks > 0


def test_no_index_dir_means_nothing_is_written(tmp_path, embedder) -> None:
    data = write_docs(tmp_path / "data", **DOCS)
    before = {p.name for p in tmp_path.iterdir()}

    build_or_load_retriever(data, embedder, index_dir=None, chunk_size=50, overlap=5)

    assert {p.name for p in tmp_path.iterdir()} == before


def test_missing_or_empty_data_folder_fails_fast(tmp_path, embedder) -> None:
    with pytest.raises(FileNotFoundError):
        build_or_load_retriever(tmp_path / "nope", embedder, index_dir=None)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        build_or_load_retriever(tmp_path / "empty", embedder, index_dir=None)
