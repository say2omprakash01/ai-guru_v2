#!/usr/bin/env python3
"""
Full turn — semantic Top-K + transformer emotion (Phase 5) + LLM RAG (Phase 4).

Requires:
  export OPENAI_API_KEY=...
Optional:
  export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # OpenRouter
  export LLM_MODEL=openai/gpt-4o-mini                   # example for OpenRouter

Usage:
  PYTHONPATH=. python scripts/run_rag_turn.py -q "I feel anxious about my future"
  PYTHONPATH=. python scripts/run_rag_turn.py -q "hello" --no-emotion
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.env_loader import load_dotenv_from_root

load_dotenv_from_root(ROOT)

from src.emotion.emotion_model import DEFAULT_EMOTION_MODEL, EmotionClassifier
from src.llm.rag_generator import RagGenerator
from src.pipeline.pipeline import GuruPipeline, PipelineConfig


def main() -> None:
    p = argparse.ArgumentParser(description="RAG guidance turn (retrieval + emotion + LLM).")
    p.add_argument("-q", "--query", required=True)
    p.add_argument("-k", type=int, default=5)
    p.add_argument("--index-dir", type=Path, default=None)
    p.add_argument("--device", default=None, help="FAISS / sentence-transformers device")
    p.add_argument(
        "--emotion-device",
        default=None,
        help='Transformers emotion model device (default: same as --device or auto)',
    )
    p.add_argument("--model", default=None, help="Override LLM_MODEL env")
    p.add_argument(
        "--emotion-model",
        default=None,
        help="HF id for emotion classifier (default: j-hartmann emotion distilroberta)",
    )
    p.add_argument(
        "--no-emotion",
        action="store_true",
        help="Skip transformer emotion (LLM infers mood from text only)",
    )
    args = p.parse_args()

    index_dir = args.index_dir or PipelineConfig.default_index_dir()
    if not (index_dir / "index.faiss").is_file():
        print(f"Missing index: {index_dir}", file=sys.stderr)
        print("Build: PYTHONPATH=. python scripts/build_faiss_index.py", file=sys.stderr)
        sys.exit(1)

    try:
        rag = RagGenerator(model=args.model)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(2)

    cfg = PipelineConfig(
        index_dir=index_dir,
        retrieval_top_k=args.k,
        device=args.device,
    )
    emotion = None
    if not args.no_emotion:
        em_dev = args.emotion_device if args.emotion_device is not None else args.device
        emotion = EmotionClassifier(
            model_name=args.emotion_model or DEFAULT_EMOTION_MODEL,
            device=em_dev,
        )

    pipe = GuruPipeline(cfg, rag=rag, emotion=emotion)
    result = pipe.run_guidance_turn(args.query, top_k=args.k)

    emotion_out = None
    if result.emotion_prediction is not None:
        ep = result.emotion_prediction
        emotion_out = {
            "label": ep.label,
            "confidence": ep.confidence,
            "scores": ep.scores,
        }

    retrieved_verses = [
        {
            "rank": h.rank,
            "citation": f"{h.scripture_name} {h.chapter}:{h.verse}".strip(),
            "chapter": h.chapter,
            "verse": h.verse,
            "shloka_text": h.shloka_text,
            "meaning": h.meaning,
            "similarity": h.score,
        }
        for h in result.retrieval.hits
    ]

    out = {
        "llm_model": rag.model,
        "emotion_classifier": emotion.model_name if emotion else None,
        "retrieval_top_k": result.retrieval.top_k,
        "hit_count": result.retrieval.hit_count,
        "citations": [
            f"{h.scripture_name} {h.chapter}:{h.verse}" for h in result.retrieval.hits
        ],
        "retrieved_verses": retrieved_verses,
        "emotion_prediction": emotion_out,
        "guidance": result.guidance.model_dump(),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
