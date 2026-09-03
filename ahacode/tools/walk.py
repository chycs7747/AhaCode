"""Shared file traversal for grep and glob, with one skip list so the two cannot drift."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

# Directories never worth searching: caches, build output, vendored source. Every
# dot-directory is skipped too, which covers .git, .venv and .ahacode.
SKIP_DIRS = {
    "__pycache__", "node_modules", "venv", "dist", "build", "target",
    "reference",  # vendored reference clones
    "sessions",  # private transcripts
}

# A file bigger than this is data, not source.
MAX_FILE_BYTES = 2_000_000


def is_skipped_dir(name: str) -> bool:
    """Whether traversal refuses to descend into a directory of this name."""
    return name in SKIP_DIRS or name.startswith(".")


def iter_files(root: Path, pattern: str = "**/*") -> Iterator[Path]:
    """Yield the paths under `root` matching a glob pattern, skipping noise directories.

    The skip list applies to directories discovered while walking, never to `root`
    itself, so an explicit search inside a skipped directory still works. A `root`
    that is a file yields that one file, so a tool can be pointed at a spilled log.

    Args:
        root: The directory (or file) to search.
        pattern: A glob pattern relative to `root`.

    Returns:
        The matching files and directories.
    """
    if root.is_file():
        yield root
        return
    for path in root.glob(pattern):
        if any(is_skipped_dir(part) for part in path.relative_to(root).parts[:-1]):
            continue
        if path.is_dir():
            if is_skipped_dir(path.name):
                continue
        elif not path.is_file():
            continue
        yield path


def read_text_or_none(path: Path) -> str | None:
    """A file's text, when it is searchable text.

    Args:
        path: The file.

    Returns:
        The text, or None for a binary (a decode error is the binary test),
        oversized or unreadable file.
    """
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
