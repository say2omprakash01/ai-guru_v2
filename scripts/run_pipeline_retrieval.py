#!/usr/bin/env python3
"""
Phase 3: run multi-verse retrieval through GuruPipeline (default Top-K=5).

Usage:
  PYTHONPATH=. python scripts/run_pipeline_retrieval.py -q "I feel anxious about my future"
  PYTHONPATH=. python scripts/run_pipeline_retrieval.py -q "karma yoga" -k 8 --show-context
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline.pipeline import GuruPipeline, PipelineConfig


def main() -> None:
    p = argparse.ArgumentParser(description="GuruPipeline retrieval stage (Phase 3).")
    p.add_argument("-q", "--query", required=True, help="User message")
    p.add_argument(
        "-k",
        type=int,
        default=5,
        help="Top-K verses (default 5 — RAG contract)",
    )
    p.add_argument(
        "--index-dir",
        type=Path,
        default=None,
        help="FAISS directory (default: models/index/gita under repo root)",
    )
    p.add_argument("--device", default=None, help='Torch device, e.g. "cpu"')
    p.add_argument(
        "--show-context",
        action="store_true",
        help="Print LLM-ready CONTEXT block (Phase 4 preview)",
    )
    args = p.parse_args()

    index_dir = args.index_dir or PipelineConfig.default_index_dir()
    if not (index_dir / "index.faiss").is_file():
        print(f"Missing index: {index_dir}", file=sys.stderr)
        print("Build: PYTHONPATH=. python scripts/build_faiss_index.py", file=sys.stderr)
        sys.exit(1)

    cfg = PipelineConfig(
        index_dir=index_dir,
        retrieval_top_k=args.k,
        device=args.device,
    )
    pipe = GuruPipeline(cfg)
    result = pipe.run_retrieval(args.query)

    print(f"top_k={result.top_k}  hits={result.hit_count}  query={result.user_message!r}\n")
    for h in result.hits:
        ref = f"{h.chapter}:{h.verse}"
        snip = h.meaning.replace("\n", " ")[:160]
        print(f"  {h.rank}. score={h.score:.4f}  {ref}  id={h.internal_id}")
        print(f"      {snip}...")
        print()

    if args.show_context and result.hits:
        print("--- CONTEXT (for future RAG prompt) ---\n")
        print(GuruPipeline.format_context_for_llm(result.hits))


if __name__ == "__main__":
    main()
