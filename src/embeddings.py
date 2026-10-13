from __future__ import annotations

import contextlib
import io
import logging
import math
import os
import re
import zlib
from collections import Counter
from typing import Any


class EmbeddingModel:
    """Thin wrapper around sentence-transformers for local embeddings."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = self._load_model(model_name)

    @staticmethod
    def _load_model(model_name: str):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)

        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Install dependencies with "
                "`pip install -r requirements.txt`."
            ) from exc

        try:
            from transformers.utils import logging as transformers_logging

            transformers_logging.set_verbosity_error()
        except ImportError:
            pass

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            return SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> Any:
        """Return L2-normalized embeddings for cosine similarity."""
        import numpy as np

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            embeddings = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return embeddings.astype(np.float32)


_HASHING_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "their this to was were what when where which who why with how do does can "
    "should would not but into than then there these those they we you your".split()
)


class HashingEmbedder:
    """Deterministic lexical embedder: hashed unigrams and bigrams, no model download.

    This is NOT a semantic model. It exists so the pipeline, the FAISS index, the API
    and the evaluation harness can run in tests and CI without network access. Use a
    real sentence-transformers model for actual retrieval quality.
    """

    def __init__(self, dimension: int = 2048) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive.")
        self.dimension = dimension
        self.model_name = f"hashing-{dimension}"

    @staticmethod
    def _tokens(text: str) -> list[str]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        tokens = []
        for word in words:
            if word in _HASHING_STOPWORDS:
                continue
            if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
                word = word[:-1]  # crude plural folding
            tokens.append(word)
        return tokens

    def encode(self, texts: list[str]) -> Any:
        import numpy as np

        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = self._tokens(text)
            features = Counter(tokens)
            features.update(f"{a}_{b}" for a, b in zip(tokens, tokens[1:]))
            for feature, count in features.items():
                data = feature.encode("utf-8")
                # crc32/adler32 are stable across runs, unlike Python's salted hash().
                index = zlib.crc32(data) % self.dimension
                sign = 1.0 if zlib.adler32(data) & 1 else -1.0
                matrix[row, index] += sign * (1.0 + math.log(count))

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32)


def load_embedder(model_name: str):
    """Return an embedder for `model_name`; 'hashing' or 'hashing-N' selects the offline one."""
    name = model_name.strip()
    if name == "hashing":
        return HashingEmbedder()
    if name.startswith("hashing-"):
        return HashingEmbedder(dimension=int(name.split("-", 1)[1]))
    return EmbeddingModel(name)
