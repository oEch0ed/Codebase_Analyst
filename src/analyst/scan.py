"""Discover source files in a local repository."""

import logging
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

SKIP_DIRS: frozenset[str] = frozenset(
    {".git", ".venv", "venv", "__pycache__", "node_modules", "build", "dist"}
)


class FileInfo(BaseModel):
    path: Path  # relative to the repo root
    lines: int


def count_lines(file: Path) -> int:
    with file.open(encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def scan_repo(root: Path, max_file_lines: int) -> list[FileInfo]:
    """Return every .py file (no SKIP_DIRS, oversized files), sorted by path."""
    dirs: list[Path] = [root]
    python_files: list[FileInfo] = []
    while dirs:
        dir = dirs.pop()
        for file in dir.iterdir():
            if file.is_dir() and file.name not in SKIP_DIRS:
                dirs.append(file)
            elif file.is_file() and file.suffix == ".py":
                lines = count_lines(file)
                if lines <= max_file_lines:
                    python_files.append(
                        FileInfo(path=file.relative_to(root), lines=lines)
                    )
                else:
                    logger.info("Skipping %s (too large: %d lines)", file, lines)

    return sorted(python_files, key=lambda f: f.path)
