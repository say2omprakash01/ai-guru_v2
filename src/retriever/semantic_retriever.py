"""
Semantic (dense vector) retrieval over the Gītā FAISS index.

Encodes the user query with the **same** sentence-transformer named in the
index manifest, runs Top-K inner-product search, returns structured hits for
RAG / API layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.embeddings.embedder import Embedder
from src.vector_db.faiss_store import FaissStore, FaissStoreManifest


@dataclass(frozen=True)
class RetrievalHit:
    """One retrieved verse: similarity score plus fields needed for UI / LLM context."""

    rank: int
    score: float
    internal_id: int
    verse_id: int
    scripture_name: str
    chapter: str
    verse: str
    shloka_text: str
    meaning: str
    themes: str
    emotion_tags: str

    @staticmethod
    def from_metadata(
        rank: int,
        internal_id: int,
        score: float,
        md: dict[str, Any],
    ) -> "RetrievalHit":
        vid = md.get("verse_id", internal_id)
        try:
            verse_id_int = int(vid)
        except (TypeError, ValueError):
            verse_id_int = internal_id
        return RetrievalHit(
            rank=rank,
            score=score,
            internal_id=internal_id,
            verse_id=verse_id_int,
            scripture_name=str(md.get("scripture_name", "")),
            chapter=str(md.get("chapter", "")),
            verse=str(md.get("verse", "")),
            shloka_text=str(md.get("shloka_text", "")),
            meaning=str(md.get("meaning", "")),
            themes=str(md.get("themes", "")),
            emotion_tags=str(md.get("emotion_tags", "")),
        )


class SemanticRetriever:
    """
    Production-style thin layer: Embedder + FaissStore + manifest.

    Load with ``from_index_dir`` so the embedding model always matches the index.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: FaissStore,
        manifest: FaissStoreManifest,
    ) -> None:
        if embedder.embedding_dim != store.embedding_dim:
            raise ValueError("Embedder and FAISS store dimension mismatch")
        if manifest.embedding_dim != store.embedding_dim:
            raise ValueError("Manifest embedding_dim does not match store")
        if embedder.model_name != manifest.model_name:
            # Same architecture with different checkpoint path strings could false-positive;
            # still warn by strict equality—indexes must be built with this model id.
            raise ValueError(
                f"Embedder model {embedder.model_name!r} != manifest {manifest.model_name!r}"
            )
        self._embedder = embedder
        self._store = store
        self._manifest = manifest

    @property
    def manifest(self) -> FaissStoreManifest:
        return self._manifest

    @classmethod
    def from_index_dir(
        cls,
        index_dir: Path,
        device: str | None = None,
    ) -> "SemanticRetriever":
        """
        Load ``index.faiss`` + metadata + manifest and construct the matching Embedder.

        Args:
            index_dir: Directory passed to ``FaissStore.save`` (e.g. ``models/index/gita``).
            device: Optional torch device override for the embedder.
        """
        store, manifest = FaissStore.load(index_dir)
        embedder = Embedder(manifest.model_name, device=device)
        return cls(embedder=embedder, store=store, manifest=manifest)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievalHit]:
        """
        Return up to ``k`` verses ranked by semantic similarity to ``query``.

        The query string is embedded as provided (no keyword preprocessing).
        """
        q = query.strip()
        if not q:
            return []
        qv = self._embedder.encode([q], show_progress_bar=False)[0]
        raw_hits = self._store.search(qv, k=k)
        out: list[RetrievalHit] = []
        for rank, (internal_id, score) in enumerate(raw_hits, start=1):
            md = self._store.get_metadata(internal_id)
            out.append(RetrievalHit.from_metadata(rank, internal_id, score, md))
        return out
