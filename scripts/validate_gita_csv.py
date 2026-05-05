#!/usr/bin/env python3
"""
Validate processed Gita CSV schema for AI Guru v2.

Usage:
    python scripts/validate_gita_csv.py data/processed/gita_verses.csv

Expects UTF-8, header row, required columns exactly as in ARCHITECTURE.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REQUIRED_COLUMNS: tuple[str, ...] = (
    "scripture_name",
    "chapter",
    "verse",
    "shloka_text",
    "meaning",
    "themes",
    "emotion_tags",
)


def validate(path: Path) -> list[str]:
    """Return list of error messages; empty means OK."""
    errors: list[str] = []
    if not path.is_file():
        return [f"File not found: {path}"]

    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return ["CSV has no header row."]
        header = [h.strip() for h in reader.fieldnames if h is not None]
        for col in REQUIRED_COLUMNS:
            if col not in header:
                errors.append(f"Missing required column: {col!r}")
        if errors:
            return errors

        rows = list(reader)
        if not rows:
            errors.append("CSV has header but zero data rows.")
            return errors

        for i, row in enumerate(rows, start=2):
            for col in REQUIRED_COLUMNS:
                val = (row.get(col) or "").strip()
                if not val:
                    errors.append(f"Row {i}: empty {col!r}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate gita_verses.csv schema.")
    parser.add_argument("csv_path", type=Path, help="Path to gita_verses.csv")
    args = parser.parse_args()
    errs = validate(args.csv_path)
    if errs:
        print("Validation failed:", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {args.csv_path}")


if __name__ == "__main__":
    main()
