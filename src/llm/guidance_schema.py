"""Structured API / RAG output (single schema for LLM JSON and FastAPI later)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GuidanceResponse(BaseModel):
    """
    Grounded guidance fields returned to the client.

    The LLM must not invent new scripture; verses appear only via the CONTEXT
    block assembled from retrieval (enforced in the prompt + pipeline).
    """

    emotion: str = Field(
        ...,
        description="Short label for the user's apparent emotional tone (from user message until Phase 5 model).",
    )
    insight: str = Field(..., description="One tight takeaway connecting user situation to the passages.")
    explanation: str = Field(
        ...,
        description="Explanation that only references ideas present in the retrieved CONTEXT.",
    )
    practical_guidance: str = Field(..., description="Concrete, respectful steps or attitudes to try.")
    reflection_question: str = Field(..., description="One open question for the user to ponder.")
    disclaimer: str = Field(
        ...,
        description="Safety / non-clinical / non-replacement for tradition or teacher.",
    )
