#!/usr/bin/env python3
"""
Build FAISS index from data/processed/gita_verses.csv (Phase 1).

Embeds ``meaning`` (+ optional non-empty themes / emotion_tags) and saves:
  models/index/gita/{index.faiss,metadata.pkl,manifest.json}

Usage:
  PYTHONPATH=. python scripts/build_faiss_index.py
  PYTHONPATH=. python scripts/build_faiss_index.py --query "I feel anxious about my future"
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.embeddings.embedder import Embedder
from src.utils.corpus_text import verse_retrieval_text
from src.vector_db.faiss_store import FaissStore, FaissStoreManifest


def load_verses_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def row_metadata(row: dict[str, str], internal_id: int) -> dict[str, object]:
    """Stored alongside FAISS id for retriever / API (full verse row + id)."""
    return {
        "verse_id": internal_id,
        "scripture_name": row.get("scripture_name", "").strip(),
        "chapter": row.get("chapter", "").strip(),
        "verse": row.get("verse", "").strip(),
        "shloka_text": row.get("shloka_text", "").strip(),
        "meaning": row.get("meaning", "").strip(),
        "themes": row.get("themes", "").strip(),
        "emotion_tags": row.get("emotion_tags", "").strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FAISS index from Gita CSV.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT / "data" / "processed" / "gita_verses.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "models" / "index" / "gita",
        help="Output directory for index.faiss + metadata + manifest",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="sentence-transformers model name",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Torch device, e.g. "cuda" or "cpu"',
    )
    parser.add_argument(
        "--query",
        default=None,
        help="If set, run one Top-5 search after building (smoke test)",
    )
    args = parser.parse_args()

    rows = load_verses_csv(args.csv)
    texts = [verse_retrieval_text(r) for r in rows]
    if any(not t for t in texts):
        bad = [i for i, t in enumerate(texts) if not t]
        print(f"Abort: empty retrieval text for CSV row indices {bad[:10]}...", file=sys.stderr)
        sys.exit(1)

    print(f"Loading embedder: {args.model}", flush=True)
    embedder = Embedder(args.model, device=args.device)
    print(f"Encoding {len(texts)} verses (dim={embedder.embedding_dim})...", flush=True)
    vectors = embedder.encode(texts, batch_size=32, show_progress_bar=True)

    store = FaissStore(embedder.embedding_dim)
    meta = [row_metadata(r, i) for i, r in enumerate(rows)]
    store.add(vectors, meta)

    manifest = FaissStoreManifest(
        model_name=args.model,
        embedding_dim=embedder.embedding_dim,
        num_vectors=store.ntotal,
        source_csv=str(args.csv.resolve()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    store.save(args.out, manifest)
    print(f"Saved FAISS store → {args.out}", flush=True)

    if args.query:
        qv = embedder.encode([args.query], show_progress_bar=False)[0]
        hits = store.search(qv, k=5)
        print("\nTop-5 for query:", repr(args.query))
        for rank, (vid, score) in enumerate(hits, 1):
            md = store.get_metadata(vid)
            ref = f"{md.get('chapter')}:{md.get('verse')}"
            snippet = (md.get("meaning") or "")[:160].replace("\n", " ")
            print(f"  {rank}. id={vid} score={score:.4f} {ref} — {snippet}...")


if __name__ == "__main__":
    main()
