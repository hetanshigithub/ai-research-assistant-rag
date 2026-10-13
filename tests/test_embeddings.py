import numpy as np
import pytest

from src.embeddings import HashingEmbedder, load_embedder


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder()
    first = embedder.encode(["vector search with FAISS", "chunking strategies"])
    second = embedder.encode(["vector search with FAISS", "chunking strategies"])

    assert first.shape == (2, embedder.dimension)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)


def test_similar_text_scores_higher_than_unrelated_text() -> None:
    embedder = HashingEmbedder()
    query, related, unrelated = embedder.encode(
        [
            "how does a FAISS flat index search vectors",
            "a flat FAISS index compares the query against every stored vector",
            "sourdough bread needs flour water and salt",
        ]
    )
    assert float(query @ related) > float(query @ unrelated)


def test_empty_input_and_empty_text() -> None:
    embedder = HashingEmbedder(dimension=64)
    assert embedder.encode([]).shape == (0, 64)
    # Text with no usable tokens becomes a zero vector, not NaN.
    assert not np.isnan(embedder.encode(["the of and"])).any()


def test_load_embedder_selects_hashing_variants() -> None:
    assert load_embedder("hashing").model_name == "hashing-2048"
    assert load_embedder("hashing-128").dimension == 128


def test_invalid_dimension_is_rejected() -> None:
    with pytest.raises(ValueError):
        HashingEmbedder(dimension=0)
