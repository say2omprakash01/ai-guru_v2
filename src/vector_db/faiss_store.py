"""
FAISS vector store with integer ids and sidecar metadata.

Uses ``IndexFlatIP`` on **L2-normalized** vectors so scores are cosine
similarity in [-1, 1] (higher is better).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import faiss  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FaissStoreManifest:
    """Serialized next to the index for reproducibility and sanity checks."""

    model_name: str
    embedding_dim: int
    num_vectors: int
    metric: str = "inner_product_on_l2_normalized_vectors"
    source_csv: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(raw: str) -> "FaissStoreManifest":
        data = json.loads(raw)
        return FaissStoreManifest(**data)


class FaissStore:
    """
    Owns a FAISS index aligned with ``metadata[i]`` for internal id ``i``.

    Internal ids are contiguous ``0 .. n-1`` (FAISS row positions). Map them to
    verse keys in ``metadata`` when resolving hits for the retriever.
    """

    def __init__(self, embedding_dim: int) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self._dim = embedding_dim
        self._index = faiss.IndexFlatIP(embedding_dim)
        self._metadata: list[dict[str, Any]] = []

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal)

    def add(
        self,
        vectors: NDArray[np.float32],
        metadata_rows: Sequence[dict[str, Any]],
    ) -> None:
        """
        Append normalized vectors and one metadata dict per vector.

        Args:
            vectors: Shape ``(n, dim)``, float32, L2-normalized rows.
            metadata_rows: Length ``n``; stored in order (FAISS id = row index).
        """
        if vectors.ndim != 2 or vectors.shape[1] != self._dim:
            raise ValueError(
                f"Expected vectors shape (n, {self._dim}), got {vectors.shape}"
            )
        if vectors.shape[0] != len(metadata_rows):
            raise ValueError("vectors and metadata_rows length mismatch")
        if vectors.shape[0] == 0:
            return
        # FAISS expects float32 contiguous
        x = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
        self._index.add(x)
        self._metadata.extend(list(metadata_rows))

    def search(
        self, query_vector: NDArray[np.float32], k: int
    ) -> list[tuple[int, float]]:
        """
        Return up to ``k`` pairs ``(internal_id, score)`` sorted by score desc.

        ``query_vector`` must be shape ``(dim,)`` or ``(1, dim)``, L2-normalized.
        """
        if k <= 0:
            return []
        q = np.ascontiguousarray(query_vector.astype(np.float32, copy=False))
        if q.ndim == 1:
            q = q.reshape(1, -1)
        if q.shape[1] != self._dim:
            raise ValueError(f"Query dim {q.shape[1]} != index dim {self._dim}")
        k_eff = min(k, self.ntotal)
        if k_eff == 0:
            return []
        scores, ids = self._index.search(q, k_eff)
        out: list[tuple[int, float]] = []
        for i in range(k_eff):
            idx = int(ids[0, i])
            if idx < 0:  # FAISS pad when empty
                continue
            out.append((idx, float(scores[0, i])))
        return out

    def get_metadata(self, internal_id: int) -> dict[str, Any]:
        return dict(self._metadata[internal_id])

    def save(self, directory: Path, manifest: FaissStoreManifest) -> None:
        """Write ``index.faiss``, ``metadata.pkl``, ``manifest.json``."""
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        with (directory / "metadata.pkl").open("wb") as f:
            pickle.dump(self._metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        (directory / "manifest.json").write_text(manifest.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> tuple["FaissStore", FaissStoreManifest]:
        """Load store from ``save()`` output directory."""
        manifest = FaissStoreManifest.from_json(
            (directory / "manifest.json").read_text(encoding="utf-8")
        )
        store = cls(manifest.embedding_dim) 
        store._index = faiss.read_index(str(directory / "index.faiss"))
        with (directory / "metadata.pkl").open("rb") as f:
            store._metadata = pickle.load(f)
        if store._index.ntotal != len(store._metadata):
            raise ValueError("Index size and metadata length mismatch")
        return store, manifest
