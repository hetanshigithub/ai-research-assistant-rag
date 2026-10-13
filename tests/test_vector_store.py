import numpy as np
import pytest

from src.chunking import TextChunk
from src.vector_store import FaissVectorStore


def make_chunks(n: int) -> list[TextChunk]:
    return [TextChunk(chunk_id=i + 1, source=f"doc{i}.txt", text=f"chunk {i}") for i in range(n)]


def unit_vectors(n: int, dim: int = 16, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_search_returns_nearest_chunk_first_with_cosine_scores() -> None:
    vectors = unit_vectors(10)
    store = FaissVectorStore(dimension=16)
    store.add(make_chunks(10), vectors)

    hits = store.search(vectors[3], k=3)

    assert hits[0][0].chunk_id == 4  # chunk_id is index + 1
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)  # identical unit vectors: cosine 1
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_k_larger_than_index_returns_everything_without_padding() -> None:
    vectors = unit_vectors(3)
    store = FaissVectorStore(dimension=16)
    store.add(make_chunks(3), vectors)

    assert len(store.search(vectors[0], k=10)) == 3


def test_faiss_padding_ids_are_never_turned_into_chunks() -> None:
    """Approximate indexes can return id -1 for 'no neighbor'. Using it as a Python index
    would silently return the LAST chunk, so the store must drop it."""

    class PaddedIndex:
        ntotal = 2

        def search(self, query, k):
            return np.array([[0.9, -3.4e38]], dtype=np.float32), np.array([[0, -1]])

    store = FaissVectorStore(dimension=4, _index=PaddedIndex())
    store.chunks = make_chunks(2)

    hits = store.search(np.ones(4, dtype=np.float32), k=2)

    assert [chunk.chunk_id for chunk, _ in hits] == [1]


def test_empty_store_returns_no_hits() -> None:
    assert FaissVectorStore(dimension=16).search(unit_vectors(1)[0], k=3) == []


def test_dimension_and_length_mismatches_are_rejected() -> None:
    store = FaissVectorStore(dimension=16)
    with pytest.raises(ValueError):
        store.add(make_chunks(2), unit_vectors(2, dim=8))
    with pytest.raises(ValueError):
        store.add(make_chunks(3), unit_vectors(2))
    store.add(make_chunks(2), unit_vectors(2))
    with pytest.raises(ValueError):
        store.search(unit_vectors(1, dim=8)[0], k=1)
    with pytest.raises(ValueError):
        store.search(unit_vectors(1)[0], k=0)


def test_unknown_index_type_is_rejected() -> None:
    with pytest.raises(ValueError):
        FaissVectorStore(dimension=16, index_type="ivf")


def test_save_and_load_round_trip(tmp_path) -> None:
    vectors = unit_vectors(20)
    store = FaissVectorStore(dimension=16)
    store.add(make_chunks(20), vectors)
    store.save(tmp_path, {"model_name": "test-model"})

    loaded, meta = FaissVectorStore.load(tmp_path)

    assert meta["model_name"] == "test-model"
    assert meta["num_chunks"] == 20
    assert [c.chunk_id for c in loaded.chunks] == [c.chunk_id for c in store.chunks]
    before = store.search(vectors[7], k=5)
    after = loaded.search(vectors[7], k=5)
    assert [c.chunk_id for c, _ in before] == [c.chunk_id for c, _ in after]


def test_loading_a_missing_index_fails_clearly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        FaissVectorStore.load(tmp_path)


def test_loading_an_inconsistent_index_is_detected(tmp_path) -> None:
    store = FaissVectorStore(dimension=16)
    store.add(make_chunks(5), unit_vectors(5))
    store.save(tmp_path)
    # Simulate a half-written save: drop a chunk record but keep the vectors.
    chunks_file = tmp_path / "chunks.json"
    import json

    records = json.loads(chunks_file.read_text())
    chunks_file.write_text(json.dumps(records[:-1]))

    with pytest.raises(ValueError, match="Corrupt index"):
        FaissVectorStore.load(tmp_path)


def test_hnsw_index_finds_the_exact_match_and_round_trips(tmp_path) -> None:
    vectors = unit_vectors(50)
    store = FaissVectorStore(dimension=16, index_type="hnsw")
    store.add(make_chunks(50), vectors)

    assert store.search(vectors[12], k=1)[0][0].chunk_id == 13

    store.save(tmp_path)
    loaded, meta = FaissVectorStore.load(tmp_path)
    assert meta["index_type"] == "hnsw"
    assert loaded.search(vectors[12], k=1)[0][0].chunk_id == 13
