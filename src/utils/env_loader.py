"""Load ``.env`` from repository root if ``python-dotenv`` is installed."""

from __future__ import annotations

from pathlib import Path


def load_dotenv_from_root(root: Path) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = root / ".env"
    if env_path.is_file():
        load_dotenv(env_path)
