#!/usr/bin/env python3
"""
Run semantic Top-K retrieval using Phase 2 SemanticRetriever (no index rebuild).

Requires an existing FAISS directory from:
  PYTHONPATH=. python scripts/build_faiss_index.py

Usage:
  PYTHONPATH=. python scripts/query_semantic.py -q "I feel anxious about my future"
  PYTHONPATH=. python scripts/query_semantic.py -q "duty without attachment" -k 3 --device cpu
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retriever.semantic_retriever import SemanticRetriever


def main() -> None:
    p = argparse.ArgumentParser(description="Semantic search over Gita FAISS index.")
    p.add_argument("-q", "--query", required=True, help="User query text")
    p.add_argument("-k", type=int, default=5, help="Top-K verses")
    p.add_argument(
        "--index-dir",
        type=Path,
        default=ROOT / "models" / "index" / "gita",
        help="FAISS directory (index.faiss + metadata + manifest)",
    )
    p.add_argument("--device", default=None, help='Torch device, e.g. "cpu" or "cuda"')
    args = p.parse_args()

    if not (args.index_dir / "index.faiss").is_file():
        print(f"Missing index: {args.index_dir}", file=sys.stderr)
        print("Build first: PYTHONPATH=. python scripts/build_faiss_index.py", file=sys.stderr)
        sys.exit(1)

    retriever = SemanticRetriever.from_index_dir(args.index_dir, device=args.device)
    hits = retriever.retrieve(args.query, k=args.k)
    print(f"model={retriever.manifest.model_name!r}  k={args.k}  query={args.query!r}\n")
    for h in hits:
        ref = f"{h.chapter}:{h.verse}"
        snip = h.meaning.replace("\n", " ")[:200]
        print(f"{h.rank}. score={h.score:.4f}  {ref}  (id={h.internal_id})")
        print(f"   {snip}...")
        print()


if __name__ == "__main__":
    main()
