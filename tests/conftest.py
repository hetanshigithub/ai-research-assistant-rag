from pathlib import Path

import pytest

from src.embeddings import HashingEmbedder

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return REPO_ROOT / "data"


@pytest.fixture(scope="session")
def eval_set_path() -> Path:
    return REPO_ROOT / "evaluation" / "eval_set.jsonl"


@pytest.fixture()
def embedder() -> HashingEmbedder:
    """Offline embedder: lets the whole suite run without downloading a model."""
    return HashingEmbedder()
