"""
End-to-end orchestration entry point (grows with phases).

Phase 3: fixed **multi-verse Top-K** semantic retrieval — the contract later
stages (RAG prompt, API) depend on. Default ``retrieval_top_k=5``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from src.retriever.semantic_retriever import RetrievalHit, SemanticRetriever

if TYPE_CHECKING:
    from src.emotion.emotion_model import EmotionClassifier, EmotionPrediction
    from src.llm.guidance_schema import GuidanceResponse
    from src.llm.rag_generator import RagGenerator


def _default_project_root() -> Path:
    """``src/pipeline/pipeline.py`` → parents[2] == repository root."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable settings for ``GuruPipeline``."""

    index_dir: Path
    retrieval_top_k: int = 5
    device: str | None = None

    @staticmethod
    def default_index_dir() -> Path:
        return _default_project_root() / "models" / "index" / "gita"

    @classmethod
    def defaults(cls) -> "PipelineConfig":
        """Sensible paths when you run from a dev checkout with a built index."""
        return cls(index_dir=cls.default_index_dir(), retrieval_top_k=5, device=None)


@dataclass(frozen=True)
class RetrievalPhaseResult:
    """Output of the retrieval stage only (Phase 3)."""

    user_message: str
    top_k: int
    hits: tuple[RetrievalHit, ...]

    @property
    def hit_count(self) -> int:
        return len(self.hits)


@dataclass(frozen=True)
class GuidanceTurnResult:
    """Retrieval + structured LLM output + optional transformer emotion (Phase 5)."""

    retrieval: RetrievalPhaseResult
    guidance: "GuidanceResponse"
    emotion_prediction: "EmotionPrediction | None" = None


class GuruPipeline:
    """
    Facade over subsystems: retrieval, optional transformer emotion (Phase 5), RAG LLM (Phase 4).
    """

    def __init__(
        self,
        config: PipelineConfig,
        rag: "RagGenerator | None" = None,
        emotion: "EmotionClassifier | None" = None,
    ) -> None:
        if config.retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be positive")
        self._config = config
        self._rag = rag
        self._emotion = emotion

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def rag(self) -> "RagGenerator | None":
        return self._rag

    @property
    def emotion(self) -> "EmotionClassifier | None":
        return self._emotion

    @cached_property
    def _retriever(self) -> SemanticRetriever:
        return SemanticRetriever.from_index_dir(
            self._config.index_dir,
            device=self._config.device,
        )

    def run_retrieval(
        self,
        user_message: str,
        *,
        top_k: int | None = None,
    ) -> RetrievalPhaseResult:
        """
        Run semantic Top-K over the Gītā index (multi-verse bundle for RAG).

        ``top_k`` overrides ``PipelineConfig.retrieval_top_k`` when set (e.g. API query).

        Empty/whitespace ``user_message`` yields zero hits without calling FAISS.
        """
        k = top_k if top_k is not None else self._config.retrieval_top_k
        if k <= 0:
            raise ValueError("top_k must be positive")
        msg = user_message.strip()
        if not msg:
            return RetrievalPhaseResult(
                user_message=user_message,
                top_k=k,
                hits=(),
            )
        hits = self._retriever.retrieve(msg, k=k)
        return RetrievalPhaseResult(
            user_message=msg,
            top_k=k,
            hits=tuple(hits),
        )

    @staticmethod
    def format_context_for_llm(hits: Sequence[RetrievalHit]) -> str:
        """
        Build a single grounding block from retrieved verses (Phase 4 will inject this).

        Only information present in ``hits`` is included — no invented citations.
        """
        blocks: list[str] = []
        for h in hits:
            ref = f"{h.scripture_name.strip()} {h.chapter}:{h.verse}".strip()
            block = "\n".join(
                [
                    f"### Passage {h.rank} — {ref} (similarity={h.score:.4f})",
                    "**Sanskrit:**",
                    h.shloka_text.strip() if h.shloka_text else "(none)",
                    "**Translation (indexed meaning):**",
                    h.meaning.strip() if h.meaning else "(none)",
                ]
            )
            blocks.append(block)
        return "\n\n".join(blocks).strip()

    def run_guidance_turn(
        self,
        user_message: str,
        *,
        top_k: int | None = None,
        use_emotion: bool | None = None,
    ) -> GuidanceTurnResult:
        """
        Retrieve Top-K verses, build CONTEXT, call the LLM, return structured guidance.

        Requires ``rag`` to be passed to ``GuruPipeline`` (see ``RagGenerator``).

        Args:
            top_k: Override number of verses retrieved (default: config).
            use_emotion: If ``False``, skip the transformer emotion path even when configured.
        """
        if self._rag is None:
            raise ValueError(
                "RAG is not configured: pass RagGenerator to GuruPipeline(..., rag=...)"
            )
        retrieval = self.run_retrieval(user_message, top_k=top_k)
        context = self.format_context_for_llm(retrieval.hits)

        do_emotion = use_emotion is not False
        emotion_prediction = None
        if (
            do_emotion
            and self._emotion is not None
            and retrieval.user_message.strip()
        ):
            emotion_prediction = self._emotion.predict(retrieval.user_message)

        if emotion_prediction is not None:
            guidance = self._rag.generate(
                retrieval.user_message,
                context,
                emotion_label=emotion_prediction.label,
                emotion_confidence=emotion_prediction.confidence,
            )
        else:
            guidance = self._rag.generate(retrieval.user_message, context)
        if emotion_prediction is not None:
            guidance = guidance.model_copy(
                update={
                    "emotion": (
                        f"{emotion_prediction.label} "
                        f"(transformer confidence {emotion_prediction.confidence:.2f})"
                    )
                }
            )
        return GuidanceTurnResult(
            retrieval=retrieval,
            guidance=guidance,
            emotion_prediction=emotion_prediction,
        )
