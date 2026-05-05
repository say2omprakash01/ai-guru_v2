"""
AI Guru v2 — FastAPI entry (Phase 6).

Run from repository root:
  PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment:
  OPENAI_API_KEY       — required for POST /v1/guidance
  AI_GURU_INDEX_DIR    — optional; default models/index/gita under repo root
  AI_GURU_USE_EMOTION  — optional; "0" disables transformer emotion at startup
  LLM_MODEL, OPENAI_BASE_URL — same as Phase 4
  CORS_ORIGINS         — optional; comma-separated browser origins (default Vite :5173)
  AI_GURU_EMBED_DEVICE — optional; torch device for query embeddings, e.g. cuda or cpu (default: auto)
"""

from __future__ import annotations

import asyncio
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.env_loader import load_dotenv_from_root

load_dotenv_from_root(ROOT)

from src.emotion.emotion_model import EmotionClassifier
from src.llm.rag_generator import RagGenerator
from src.pipeline.pipeline import GuruPipeline, PipelineConfig


class GuidanceRequest(BaseModel):
    """Client payload for one guidance turn."""

    message: str = Field(..., min_length=1, max_length=8000)
    top_k: int = Field(5, ge=1, le=20, description="Verses to retrieve for RAG context")
    use_emotion_model: bool = Field(
        True,
        description="If true, run HF emotion classifier when the server loaded one",
    )

    @field_validator("message")
    @classmethod
    def strip_message(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("message cannot be empty or whitespace only")
        return s


class CitationOut(BaseModel):
    scripture_name: str
    chapter: str
    verse: str
    similarity: float
    internal_id: int


class RetrievedVerseOut(BaseModel):
    """Full grounded passage (what the LLM saw in CONTEXT)."""

    rank: int
    citation: str
    chapter: str
    verse: str
    shloka_text: str
    meaning: str
    similarity: float


class EmotionPredictionOut(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]


class GuidanceBodyOut(BaseModel):
    emotion: str
    insight: str
    explanation: str
    practical_guidance: str
    reflection_question: str
    disclaimer: str


class GuidanceResponseOut(BaseModel):
    ok: bool = True
    message: str = Field("", description="Echo of trimmed user input")
    top_k: int
    citations: list[CitationOut]
    retrieved_verses: list[RetrievedVerseOut]
    emotion_prediction: EmotionPredictionOut | None
    guidance: GuidanceBodyOut
    llm_model: str | None = None


class HealthOut(BaseModel):
    status: str
    index_ready: bool
    rag_ready: bool
    emotion_enabled: bool


def _build_pipeline() -> GuruPipeline:
    raw = os.environ.get("AI_GURU_INDEX_DIR", "").strip()
    index_dir = Path(raw) if raw else PipelineConfig.default_index_dir()
    if not (index_dir / "index.faiss").is_file():
        raise RuntimeError(
            f"FAISS index not found at {index_dir}. "
            "Run: PYTHONPATH=. python scripts/build_faiss_index.py"
        )

    rag: RagGenerator | None = None
    try:
        rag = RagGenerator()
    except ValueError:
        pass

    emotion: EmotionClassifier | None = None
    if os.environ.get("AI_GURU_USE_EMOTION", "1").strip() != "0":
        emotion = EmotionClassifier()

    raw_dev = os.environ.get("AI_GURU_EMBED_DEVICE", "").strip()
    embed_device: str | None = raw_dev if raw_dev else None
    cfg = PipelineConfig(index_dir=index_dir, retrieval_top_k=5, device=embed_device)
    return GuruPipeline(cfg, rag=rag, emotion=emotion)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pipeline = _build_pipeline()
    yield


app = FastAPI(
    title="AI Guru v2",
    description="Emotion-aware, RAG-grounded Bhagavad Gītā guidance API",
    version="0.2.0",
    lifespan=lifespan,
)


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return ["http://localhost:5173", "http://127.0.0.1:5173"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    pipe: GuruPipeline = app.state.pipeline
    index_ok = (pipe.config.index_dir / "index.faiss").is_file()
    return HealthOut(
        status="ok" if index_ok else "degraded",
        index_ready=index_ok,
        rag_ready=pipe.rag is not None,
        emotion_enabled=pipe.emotion is not None,
    )


def _run_turn_sync(body: GuidanceRequest) -> GuidanceResponseOut:
    pipe: GuruPipeline = app.state.pipeline
    if pipe.rag is None:
        raise HTTPException(
            status_code=503,
            detail="LLM not configured: set OPENAI_API_KEY (and optional OPENAI_BASE_URL).",
        )
    try:
        result = pipe.run_guidance_turn(
            body.message,
            top_k=body.top_k,
            use_emotion=body.use_emotion_model,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Guidance failed: {e!s}") from e

    cites = [
        CitationOut(
            scripture_name=h.scripture_name,
            chapter=h.chapter,
            verse=h.verse,
            similarity=h.score,
            internal_id=h.internal_id,
        )
        for h in result.retrieval.hits
    ]
    retrieved = [
        RetrievedVerseOut(
            rank=h.rank,
            citation=f"{h.scripture_name} {h.chapter}:{h.verse}".strip(),
            chapter=h.chapter,
            verse=h.verse,
            shloka_text=h.shloka_text,
            meaning=h.meaning,
            similarity=h.score,
        )
        for h in result.retrieval.hits
    ]
    ep = result.emotion_prediction
    emotion_out = None
    if ep is not None:
        emotion_out = EmotionPredictionOut(
            label=ep.label, confidence=ep.confidence, scores=ep.scores
        )
    g = result.guidance
    return GuidanceResponseOut(
        message=result.retrieval.user_message,
        top_k=result.retrieval.top_k,
        citations=cites,
        retrieved_verses=retrieved,
        emotion_prediction=emotion_out,
        guidance=GuidanceBodyOut(
            emotion=g.emotion,
            insight=g.insight,
            explanation=g.explanation,
            practical_guidance=g.practical_guidance,
            reflection_question=g.reflection_question,
            disclaimer=g.disclaimer,
        ),
        llm_model=pipe.rag.model,
    )


@app.post("/v1/guidance", response_model=GuidanceResponseOut)
async def guidance(body: GuidanceRequest) -> GuidanceResponseOut:
    """
    One full turn: retrieve Top-K, optional emotion, LLM JSON guidance.

    CPU/GPU-heavy work runs in a thread pool so the event loop stays responsive.
    """
    return await asyncio.to_thread(_run_turn_sync, body)
