#!/usr/bin/env python3
"""
Build data/processed/gita_verses.csv from the vedicscriptures/bhagavad-gita JSON corpus.

Source (GPL-3.0): https://github.com/vedicscriptures/bhagavad-gita
Uses raw.githubusercontent.com (no GitHub REST API — avoids rate limits).

Default translator field: ``prabhu`` → A.C. Bhaktivedanta Swami Prabhupada (English ``et``).

Other keys (examples): ``siva`` Swami Sivananda, ``gambir`` Swami Gambhirananda, ``rams`` Ramsukhdas.

Usage:
    python scripts/build_gita_csv_from_vedicscriptures.py -o data/processed/gita_verses.csv
    python scripts/build_gita_csv_from_vedicscriptures.py --translator siva -o data/processed/gita_sivananda.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE_URL = (
    "https://raw.githubusercontent.com/vedicscriptures/bhagavad-gita/main/slok/{name}"
)

# Stop scanning a chapter after this many consecutive misses (handles gaps if any).
CONSECUTIVE_404_LIMIT = 8


def fetch_json(filename: str, timeout: float = 45.0) -> dict[str, Any] | None:
    url = BASE_URL.format(name=filename)
    req = urllib.request.Request(url, headers={"User-Agent": "ai-guru-v2-ingest/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except urllib.error.URLError:
        raise


def extract_meaning(payload: dict[str, Any], translator_key: str) -> str:
    node = payload.get(translator_key)
    if node is None:
        return ""
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, dict):
        et = (node.get("et") or "").strip()
        if et:
            return et
        return (node.get("translation") or "").strip()
    return ""


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def scan_chapter(chapter: int, translator_key: str) -> list[dict[str, str]]:
    """Fetch each sloka JSON once; stop after CONSECUTIVE_404_LIMIT consecutive 404s."""
    rows: list[dict[str, str]] = []
    misses = 0
    v = 1
    while v <= 130 and misses < CONSECUTIVE_404_LIMIT:
        name = f"bhagavadgita_chapter_{chapter}_slok_{v}.json"
        data = fetch_json(name)
        if data is None:
            misses += 1
        else:
            misses = 0
            slok = normalize_whitespace(str(data.get("slok") or ""))
            meaning = normalize_whitespace(extract_meaning(data, translator_key))
            if not meaning:
                meaning = "[translation missing for this author in source JSON]"
            rows.append(
                {
                    "scripture_name": "Bhagavad Gita",
                    "chapter": str(chapter),
                    "verse": str(data.get("verse") or v),
                    "shloka_text": slok,
                    "meaning": meaning,
                    "themes": "",
                    "emotion_tags": "",
                }
            )
        v += 1
        time.sleep(0.05)  # be polite to GitHub raw
    return rows


def build_rows(translator_key: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for ch in range(1, 19):
        rows.extend(scan_chapter(ch, translator_key))
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "scripture_name",
        "chapter",
        "verse",
        "shloka_text",
        "meaning",
        "themes",
        "emotion_tags",
    )
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gita_verses.csv from vedicscriptures JSON.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/processed/gita_verses.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--translator",
        default="prabhu",
        help="JSON key for translator block (default: prabhu = Prabhupada)",
    )
    args = parser.parse_args()

    print("Fetching ślokas from vedicscriptures/bhagavad-gita (raw GitHub)...", flush=True)
    rows = build_rows(args.translator)
    if not rows:
        print("No rows built; check network or translator key.", file=sys.stderr)
        sys.exit(1)
    write_csv(args.output, rows)
    print(f"Wrote {len(rows)} rows → {args.output}", flush=True)


if __name__ == "__main__":
    main()
