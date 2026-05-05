"""
Dense text embeddings via sentence-transformers.

Vectors are L2-normalized so inner product equals cosine similarity when used
with FAISS ``IndexFlatIP`` (see ``faiss_store``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer


class Embedder:
    """
    Wraps a single SentenceTransformer model for query and corpus encoding.

    The same model instance must be used at index-build time and at query time,
    otherwise vectors live in incompatible spaces.
    """

    def __init__(self, model_name: str, device: str | None = None) -> None:
        """
        Args:
            model_name: HuggingFace / sentence-transformers model id or local path.
            device: e.g. ``"cuda"``, ``"cpu"``; ``None`` lets the library decide.
        """
        self._model_name = model_name
        kwargs: dict[str, str] = {}
        if device is not None:
            kwargs["device"] = device
        self._model = SentenceTransformer(model_name, **kwargs)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Vector dimensionality produced by this model."""
        return int(self._model.get_sentence_embedding_dimension())

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
    ) -> NDArray[np.float32]:
        """
        Encode texts to a single precision matrix of shape ``(len(texts), dim)``.

        Embeddings are L2-normalized row-wise for cosine-equivalent IP search.
        """
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32, copy=False)
