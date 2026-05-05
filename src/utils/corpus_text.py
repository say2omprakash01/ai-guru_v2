"""
Build the exact string that was embedded for each CSV row at index time.

The retriever encodes **user queries** as-is, but any future query expansion or
filters must stay consistent with what ``build_faiss_index`` stored in FAISS.
"""

from __future__ import annotations


def verse_retrieval_text(row: dict[str, str]) -> str:
    """
    Concatenate ``meaning`` with optional ``themes`` and ``emotion_tags``.

    Same logic as Phase 1 index build: empty optional fields are skipped.
    """
    parts: list[str] = []
    m = (row.get("meaning") or "").strip()
    if m:
        parts.append(m)
    for key in ("themes", "emotion_tags"):
        t = (row.get(key) or "").strip()
        if t:
            parts.append(t)
    return "\n".join(parts)
