"""
LLM call for grounded spiritual guidance (Phase 4).

Uses an OpenAI-compatible Chat Completions API (OpenAI, OpenRouter, Azure, etc.).
Scripture content must come **only** from the provided CONTEXT string.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from src.llm.guidance_schema import GuidanceResponse

SYSTEM_RAG = """You are AI Guru — a careful, respectful guide for Bhagavad Gītā-based reflection.

STRICT RULES:
1) Use ONLY the provided CONTEXT passages for any quotation, paraphrase, or claim about what the Gītā says. If CONTEXT is empty, say you have no retrieved verses and do NOT invent verses.
2) Do NOT output Sanskrit or English "verse text" that is not present inside CONTEXT.
3) You MAY interpret and connect ideas to the USER MESSAGE using plain language, but scripture claims must trace to CONTEXT.
4) Respond with a single JSON object only — no markdown fences, no preamble.
5) JSON keys exactly: emotion, insight, explanation, practical_guidance, reflection_question, disclaimer.
6) If EMOTION FROM CLASSIFIER is provided, set "emotion" to that exact string (the server may normalize it). Otherwise infer a short mood label from the USER MESSAGE only.
7) "disclaimer" must state this is not medical/clinical advice and not a substitute for a qualified human teacher or tradition.

Tone: warm, concise, non-authoritarian."""

USER_TEMPLATE = """USER MESSAGE:
{user_message}

{emotion_section}

RETRIEVED CONTEXT (only source for scripture claims):
{context}

Return one JSON object with keys: emotion, insight, explanation, practical_guidance, reflection_question, disclaimer."""


# Always appended in code so deploys keep a baseline safety line even if the model shortens disclaimer.
_FALLBACK_DISCLAIMER = (
    "Educational reflection only — not medical, psychiatric, or legal advice. "
    "Not a substitute for a qualified teacher or your own tradition. "
    "Verses above are only those retrieved for this turn."
)


class RagGenerator:
    """
    Wraps chat completions with JSON-shaped ``GuidanceResponse`` parsing.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        """
        Args:
            model: Chat model id. Defaults to env ``LLM_MODEL`` or ``gpt-4o-mini``.
            api_key: Defaults to env ``OPENAI_API_KEY``.
            base_url: Optional OpenAI-compatible API root (e.g. OpenRouter).
            timeout: HTTP timeout seconds.
        """
        self._model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "Missing API key: set OPENAI_API_KEY in the environment (or pass api_key=)."
            )
        b = base_url if base_url is not None else os.environ.get("OPENAI_BASE_URL")
        kwargs: dict[str, Any] = {"api_key": key, "timeout": timeout}
        if b:
            kwargs["base_url"] = b
        # OpenRouter recommends optional HTTP-Referer / X-Title for analytics (see openrouter.ai/docs)
        if b and "openrouter.ai" in b:
            hdrs: dict[str, str] = {}
            ref = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
            if ref:
                hdrs["HTTP-Referer"] = ref
            title = os.environ.get("OPENROUTER_APP_TITLE", "AI Guru v2").strip()
            hdrs["X-Title"] = title or "AI Guru v2"
            if hdrs:
                kwargs["default_headers"] = hdrs
        self._client = OpenAI(**kwargs)

    @property
    def model(self) -> str:
        return self._model

    def generate(
        self,
        user_message: str,
        context_block: str,
        *,
        emotion_label: str | None = None,
        emotion_confidence: float | None = None,
    ) -> GuidanceResponse:
        """
        Call the LLM with user text + retrieved CONTEXT; validate JSON into ``GuidanceResponse``.

        When ``emotion_label`` is set (from a transformer classifier), the user prompt
        includes EMOTION FROM CLASSIFIER so the model aligns tone; the pipeline may
        still overwrite the JSON ``emotion`` field for API consistency.

        Empty ``context_block`` still calls the model with an explicit empty section so it
        refuses to fabricate verses (per SYSTEM_RAG).
        """
        ctx = context_block.strip() if context_block else "(No passages retrieved.)"
        emotion_section = _build_emotion_section(emotion_label, emotion_confidence)
        user_content = USER_TEMPLATE.format(
            user_message=user_message.strip(),
            emotion_section=emotion_section,
            context=ctx,
        )

        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_RAG},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.4,
            )
        except Exception:
            # Some OpenRouter/local models reject json_object — retry without it.
            completion = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_RAG + " Output valid JSON only."},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.4,
            )

        raw = completion.choices[0].message.content
        if not raw:
            raise RuntimeError("LLM returned empty content")
        raw = raw.strip()
        # Strip accidental ``` fences if the provider ignored response_format
        if raw.startswith("```"):
            raw = raw[3:].lstrip()
            if raw.lower().startswith("json"):
                raw = raw[4:].lstrip()
            end = raw.rfind("```")
            if end != -1:
                raw = raw[:end].strip()

        data = json.loads(raw)
        guidance = GuidanceResponse.model_validate(data)
        return _ensure_disclaimer(guidance)


def _build_emotion_section(
    emotion_label: str | None,
    emotion_confidence: float | None,
) -> str:
    if not emotion_label:
        return (
            "EMOTION FROM CLASSIFIER: (not used for this request — infer a short "
            'mood label for JSON "emotion" from USER MESSAGE only.)'
        )
    conf = emotion_confidence if emotion_confidence is not None else 0.0
    exact = f"{emotion_label} (transformer confidence {conf:.2f})"
    return (
        "EMOTION FROM CLASSIFIER (use this EXACT string for JSON key \"emotion\"):\n"
        f"{exact}\n"
        "Tailor insight, explanation, and reflection_question tone to this mood."
    )


def _ensure_disclaimer(g: GuidanceResponse) -> GuidanceResponse:
    d = g.disclaimer.strip()
    if len(d) < 40:
        return g.model_copy(update={"disclaimer": _FALLBACK_DISCLAIMER})
    if "not medical" not in d.lower() and "advice" not in d.lower():
        return g.model_copy(
            update={"disclaimer": f"{d} {_FALLBACK_DISCLAIMER}"}
        )
    return g
