from __future__ import annotations

import contextlib
import io
import logging
import os
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

    def encode(self, texts: list[str]) -> Any:
        """Return L2-normalized embeddings for cosine similarity."""
        import numpy as np

        if not texts:
            return np.empty((0, 0), dtype=np.float32)

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
